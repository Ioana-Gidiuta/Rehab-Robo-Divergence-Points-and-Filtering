import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, butter, cheby1, filtfilt
from scipy.interpolate import UnivariateSpline

from constants import SIGMA, MAX_GAP, CONF_TH, WIN, GAUSS_SIG, GAUSS_WIN, SEC, FPS, dt
from utils import min_jerk_1d

def gauss_conf_weighted_1d(y, conf, sigma=SIGMA, max_gap=MAX_GAP):
    """
    Confidence-weighted Gaussian smoother (σ in *frames*).
        y       : 1-D numpy array of coordinates
        conf    : per-frame confidence in [0,1]
        sigma   : Gaussian sigma in frames
        max_gap : ≤ max_gap consecutive low-conf frames are interpolated
    """
    y = y.astype(float).copy()
    ok = conf >= CONF_TH

    # ---- 1  interpolate short gaps ---------------------------------
    if not ok.all():
        t = np.arange(len(y))
        y[~ok] = np.nan
        y = pd.Series(y).interpolate(
                method="linear", limit=max_gap,
                limit_area="inside").to_numpy()

    # ---- 2  confidence-weighted Gaussian smoothing -----------------
    w = conf.copy()
    # avoid dividing by ~zero where conf is tiny but > 0
    w[w < 1e-3] = 1e-3
    num = gaussian_filter1d(y * w, sigma=sigma, mode='nearest', truncate=WIN/(2*sigma))
    den = gaussian_filter1d(w     , sigma=sigma, mode='nearest', truncate=WIN/(2*sigma))
    return num / den

def gauss_smooth(x, y):
    # truncate so the window is about 5 frames (±2.5 σ)
    kw = dict(sigma=GAUSS_SIG, mode='nearest', truncate=GAUSS_WIN/(2*GAUSS_SIG))
    return gaussian_filter1d(x, **kw), gaussian_filter1d(y, **kw)

def kalman_smooth_xy(x_obs, y_obs, var_proc=2.0, var_meas=16.0):
    A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    Q = var_proc * np.diag([0,0,1,1])
    H = np.array([[1,0,0,0],[0,1,0,0]])
    R = var_meas * np.eye(2)
    n = len(x_obs)
    xs = np.zeros((n,4)); Ps = np.zeros((n,4,4))
    x_pred = np.array([x_obs[0], y_obs[0], 0, 0],float)
    P_pred = np.diag([var_meas,var_meas,100,100])
    # fwd
    for t in range(n):
        z = np.array([x_obs[t], y_obs[t]])
        S = H@P_pred@H.T + R
        K = P_pred@H.T@np.linalg.inv(S)
        x_upd = x_pred + K@(z - H@x_pred)
        P_upd = (np.eye(4)-K@H)@P_pred
        xs[t], Ps[t] = x_upd, P_upd
        x_pred = A@x_upd
        P_pred = A@P_upd@A.T + Q
    # RTS
    for t in range(n-2,-1,-1):
        P_up = Ps[t]
        P_pr = A@P_up@A.T + Q
        C = P_up@A.T@np.linalg.inv(P_pr)
        xs[t] += C@(xs[t+1]-A@xs[t])
    return xs[:,0], xs[:,1]

def baseline_filter(sig):
    return median_filter(
                uniform_filter1d(sig, SEC, mode="nearest"),
                SEC, mode="nearest")

filter_fns = {
    "raw"      : lambda x, y, fr: (x, y),
    "gauss"   : lambda x, y, fr: gauss_smooth(x, y),
    "baseline" : lambda x, y, fr: (baseline_filter(x),
                                   baseline_filter(y)),
    "minjerk"  : lambda x, y, fr: (min_jerk_1d(x), min_jerk_1d(y)),
    "savgol"   : lambda x, y, fr: (savgol_filter(x,11,3),
                                   savgol_filter(y,11,3)),
    "butter"   : lambda x, y, fr: (filtfilt(*butter(4,5/(0.5*FPS)), x),
                                   filtfilt(*butter(4,5/(0.5*FPS)), y)),
    "cheby"    : lambda x, y, fr: (filtfilt(*cheby1(4,0.05,5/(0.5*FPS)), x),
                                   filtfilt(*cheby1(4,0.05,5/(0.5*FPS)), y)),
    "spline"   : lambda x, y, fr: (UnivariateSpline(fr,x,s=None,k=3)(fr),
                                   UnivariateSpline(fr,y,s=None,k=3)(fr)),
    "kalman"   : lambda x, y, fr:  kalman_smooth_xy(x, y)
}