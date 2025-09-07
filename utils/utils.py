import numpy as np
from scipy.signal   import correlate
from math import sqrt
from scipy.sparse.linalg import spsolve
from scipy import sparse

from constants import dt, FPS, MAX_LAG_FR, JOINTS
# ================================================================
# FUNCTION TO EXTRACT DISPLACEMENT METRICS FOR ONE SESSION
def disp_metrics(session, filt_name, filt_fn):
    """Return list of dicts – one per joint."""
    fr  = session["frame_id"].to_numpy(dtype=float)
    # joints of interest (odd = left, even = right)
    rows = []
    for joint_name, k in JOINTS.items():
        xs, ys = filt_fn(session[f"x{k}"].to_numpy(),
                         session[f"y{k}"].to_numpy(), fr)
        # frame-wise displacement (first sample gets NaN → drop)
        disp = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        rows.append({
            "participant": int(session["participant_id"].iloc[0]),
            "date"      : session["date"].iloc[0],
            "camera"    : int(session["camera"].iloc[0]),
            "filter"    : filt_name,
            "joint"     : joint_name,
            "mean_disp" : np.nanmean(disp),
            "median_disp":np.nanmedian(disp),
            "iqr_disp"  : np.subtract(*np.nanpercentile(disp, [75, 25])),
            "path_len"  : np.nansum(disp),           # total travelled pixels
            "n_frames"  : len(session)
        })
    return rows

def peak_lag(a, b, max_lag=MAX_LAG_FR):
    """
    Cross-correlate 1-D signals a (raw) and b (filtered) and
    return (lag_frames, peak_corr).  Positive lag → b lags behind a.
    """
    corr = correlate(b - b.mean(), a - a.mean(), mode="full")
    lags = np.arange(-len(a)+1, len(a))
    # restrict to ±max_lag for stability
    m = np.abs(lags) <= max_lag
    idx = np.argmax(corr[m])
    return lags[m][idx], corr[m][idx] / (np.std(a)*np.std(b)*len(a))

# --------------------------------------------------------------
# helper: displacement at lag L
# --------------------------------------------------------------
def lagged_disp(x, y, lag):
    """Euclidean distance between (x_t,y_t) and (x_{t+lag},y_{t+lag})."""
    return np.sqrt((x[lag:] - x[:-lag])**2 + (y[lag:] - y[:-lag])**2)

def part_confidence(row, idxs):
    return row[[f"score{i}" for i in idxs]].mean()

def se(values):
    return np.std(values) / sqrt(len(values))

def mean_sq_jerk(coord):
    return np.mean(np.diff(coord, n=3)**2) / dt**6

def min_jerk_1d(y, lam=1e4):
    """
    Penalised‑least‑squares smoother that minimises
        Σ (y_i - x_i)^2  +  λ Σ (Δ³ x_i)^2
    where Δ³ is the 3rd finite difference (discrete jerk).

    Parameters
    ----------
    y : 1‑D numpy array (length n)
    lam : float, smoothing strength (higher = smoother)

    Returns
    -------
    x_hat : 1‑D numpy array, smoothed trajectory
    """
    n = len(y)
    # build 3rd‑difference operator: shape (n-3) × n
    diags = np.array([1, -3, 3, -1])
    offsets = np.arange(4)
    D = sparse.diags(diagonals=[diags[0]*np.ones(n-3),
                                diags[1]*np.ones(n-3),
                                diags[2]*np.ones(n-3),
                                diags[3]*np.ones(n-3)],
                     offsets=offsets,
                     shape=(n-3, n))
    A = sparse.eye(n) + lam * (D.T @ D)
    x_hat = spsolve(A.tocsc(), y)
    return x_hat

def jerk(coord):
    return np.diff(coord, n=3) / dt**3

def msj(x, y):
    return np.mean(jerk(x)**2 + jerk(y)**2)