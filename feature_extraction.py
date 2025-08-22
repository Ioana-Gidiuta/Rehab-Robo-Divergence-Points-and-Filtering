import numpy as np
import pandas as pd
from scipy.stats import iqr
from feature_computation import vel, acc, median_position, median_velocity, signal_entropy, cross_corr
from feature_computation import angle, angular_acc, angular_vel

# ankles = [15, 16]  # left and right ankle key‑points
# wrists = [9, 10]  # left and right wrist key‑points
# knees = [13, 14]  # left and right knee key‑points
# elbows = [7, 8]  # left and right elbow key‑points
# even - right
# odd - left

# for ankles and wrists look at -  cross-correlation, entropu, IQR Acceleration, IQR - position, IQR velocity, median position, median velocuty
# for knees and elbows look at - cross-correlation, entropy, IQR Angular acceleration, IQR angular velocity
# mean joint angle, meadian angluar velocuty, stdev jpooint angles


def features_wrist_block(df, fps=30.0):
    """
    Returns a dict of features for the two wrists in one window / trial.
    `df` must have columns ['x9','y9','x10','y10'] (left & right wrist).
    """
    dt = 1.0 / fps
    xl, yl = df["x9"].to_numpy(),  df["y9"].to_numpy()   # left
    xr, yr = df["x10"].to_numpy(), df["y10"].to_numpy()  # right

    # combine X/Y so correlation works on 1-D series
    pos_l = np.vstack([xl, yl]).T.reshape(-1)
    pos_r = np.vstack([xr, yr]).T.reshape(-1)

    # magnitude-based scalar signals
    vmag_l = np.linalg.norm(np.stack([vel(xl,dt), vel(yl,dt)]), axis=0)
    vmag_r = np.linalg.norm(np.stack([vel(xr,dt), vel(yr,dt)]), axis=0)
    amag_l = np.linalg.norm(np.stack([acc(xl,dt), acc(yl,dt)]), axis=0)
    amag_r = np.linalg.norm(np.stack([acc(xr,dt), acc(yr,dt)]), axis=0)

    feats = {
        # ------- bilateral coordination / symmetry -------
        "wr_cross_corr"        : cross_corr(pos_l, pos_r),

        # ------- variability / complexity -------
        "wr_entropy"           : signal_entropy(np.hstack([pos_l, pos_r])),

        # ------- abruptness & smoothness -------
        "wr_iqr_acc"           : np.mean([iqr(amag_l), iqr(amag_r)]),
        "wr_iqr_pos"           : np.mean([iqr(pos_l), iqr(pos_r)]),
        "wr_iqr_vel"           : np.mean([iqr(vmag_l), iqr(vmag_r)]),

        # ------- postural bias & effort -------
        "wr_median_pos"        : np.mean([median_position(pos_l),
                                          median_position(pos_r)]),
        "wr_median_vel"        : np.mean([median_velocity(vmag_l, dt),
                                          median_velocity(vmag_r, dt)]),
    }
    return feats


def features_elbow_block(df, fps: float = 30.0) -> dict:
    """
    Compute Table-2 angular features for the two elbows.

    Required columns in `df`
        left  shoulder/elbow/wrist :  x5,y5 | x7,y7 | x9,y9
        right shoulder/elbow/wrist :  x6,y6 | x8,y8 | x10,y10
    Returns
    -------
    dict  –  feature-name ➜ scalar value
    """
    dt = 1.0 / fps

    # ---- build XY arrays (n_frames, 2) ----
    sh_l  = df[["x5",  "y5"]].to_numpy()
    el_l  = df[["x7",  "y7"]].to_numpy()
    wr_l  = df[["x9",  "y9"]].to_numpy()

    sh_r  = df[["x6",  "y6"]].to_numpy()
    el_r  = df[["x8",  "y8"]].to_numpy()
    wr_r  = df[["x10", "y10"]].to_numpy()

    # ---- joint angles (radians) ----
    theta_l = angle(sh_l, el_l, wr_l)      # left elbow flex/ext
    theta_r = angle(sh_r, el_r, wr_r)      # right elbow

    # ---- angular velocity & acceleration ----
    ω_l, ω_r = angular_vel(theta_l, dt),    angular_vel(theta_r, dt)
    α_l, α_r = angular_acc(theta_l, dt),    angular_acc(theta_r, dt)

    # ---- assemble features ----
    feats = {
        # bilateral coordination / symmetry
        "el_cross_corr"      : cross_corr(theta_l, theta_r),

        # variability / complexity
        "el_entropy"         : signal_entropy(np.hstack([theta_l, theta_r])),

        # spasticity / smoothness proxies
        "el_iqr_ang_acc"     : np.mean([iqr(α_l),  iqr(α_r)]),
        "el_iqr_ang_vel"     : np.mean([iqr(ω_l),  iqr(ω_r)]),

        # tone / rigidity / range
        "el_mean_angle"      : np.mean([theta_l.mean(),  theta_r.mean()]),
        "el_median_ang_vel"  : np.median(np.abs(np.hstack([ω_l, ω_r]))),
        "el_std_angle"       : np.std(np.hstack([theta_l,  theta_r])),
    }
    return feats


def features_ankle_block(df, fps=30.0):
    """
    Returns a dict of features for the two ankles in one window / trial.
    `df` must have columns ['x15','y15','x16','y10'] (left & right ankle).
    """
    dt = 1.0 / fps
    xl, yl = df["x15"].to_numpy(),  df["y15"].to_numpy()   # left
    xr, yr = df["x16"].to_numpy(), df["y16"].to_numpy()  # right

    # combine X/Y so correlation works on 1-D series
    pos_l = np.vstack([xl, yl]).T.reshape(-1)
    pos_r = np.vstack([xr, yr]).T.reshape(-1)

    # magnitude-based scalar signals
    vmag_l = np.linalg.norm(np.stack([vel(xl,dt), vel(yl,dt)]), axis=0)
    vmag_r = np.linalg.norm(np.stack([vel(xr,dt), vel(yr,dt)]), axis=0)
    amag_l = np.linalg.norm(np.stack([acc(xl,dt), acc(yl,dt)]), axis=0)
    amag_r = np.linalg.norm(np.stack([acc(xr,dt), acc(yr,dt)]), axis=0)

    feats = {
        # ------- bilateral coordination / symmetry -------
        "an_cross_corr"        : cross_corr(pos_l, pos_r),

        # ------- variability / complexity -------
        "an_entropy"           : signal_entropy(np.hstack([pos_l, pos_r])),

        # ------- abruptness & smoothness -------
        "an_iqr_acc"           : np.mean([iqr(amag_l), iqr(amag_r)]),
        "an_iqr_pos"           : np.mean([iqr(pos_l), iqr(pos_r)]),
        "an_iqr_vel"           : np.mean([iqr(vmag_l), iqr(vmag_r)]),

        # ------- postural bias & effort -------
        "an_median_pos"        : np.mean([median_position(pos_l),
                                          median_position(pos_r)]),
        "an_median_vel"        : np.mean([median_velocity(vmag_l, dt),
                                          median_velocity(vmag_r, dt)]),
    }
    return feats

def features_knee_block(df, fps=30.0):
    dt = 1.0 / fps

    # XY matrices (n_frames, 2)
    hip_l  = df[["x11","y11"]].to_numpy()
    knee_l = df[["x13","y13"]].to_numpy()
    ank_l  = df[["x15","y15"]].to_numpy()
    hip_r  = df[["x12","y12"]].to_numpy()
    knee_r = df[["x14","y14"]].to_numpy()
    ank_r  = df[["x16","y16"]].to_numpy()

    theta_l = angle(hip_l, knee_l, ank_l)        # radians
    theta_r = angle(hip_r, knee_r, ank_r)

    ω_l,  ω_r  = angular_vel(theta_l, dt),  angular_vel(theta_r, dt)
    α_l,  α_r  = angular_acc(theta_l, dt),  angular_acc(theta_r, dt)

    feats = {
        "kn_cross_corr"        : cross_corr(theta_l, theta_r),
        "kn_entropy"           : signal_entropy(np.hstack([theta_l, theta_r])),
        # spasticity / smoothness proxies
        "kn_iqr_ang_acc"       : np.mean([iqr(α_l), iqr(α_r)]),
        "kn_iqr_ang_vel"       : np.mean([iqr(ω_l), iqr(ω_r)]),
        # tone / rigidity / range
        "kn_mean_angle"        : np.mean([theta_l.mean(), theta_r.mean()]),
        "kn_median_ang_vel"    : np.median(np.abs(np.hstack([ω_l, ω_r]))),
        "kn_std_angle"         : np.std(np.hstack([theta_l, theta_r])),
    }
    return feats

def all_features_for_session(session, smoother_name, smoother_fn):
    """Return dict of ALL features for one session & one filter."""
    fr   = session["frame_id"].to_numpy(dtype=float)

    # we need BOTH the smoothed coords **and** the original scores
    smoothed = {}
    for k in range(17):
        xs, ys = smoother_fn(session[f"x{k}"].to_numpy(),
                             session[f"y{k}"].to_numpy(), fr)
        smoothed[f"x{k}"] = xs
        smoothed[f"y{k}"] = ys
    smooth_df = pd.DataFrame(smoothed)

    # stitch the smoothed coords next to original meta-columns
    tmp = pd.concat([session[["frame_id"]].reset_index(drop=True),
                     smooth_df], axis=1)

    feats = {}
    # position-based
    feats.update(features_wrist_block(tmp))
    feats.update(features_ankle_block(tmp))
    # angular
    feats.update(features_knee_block(tmp))
    feats.update(features_elbow_block(tmp))

    # identifiers
    first = session.iloc[0]
    feats.update({
        "participant": int(first["participant_id"]),
        "date"       : first["date"],
        "camera"     : int(first["camera"]),
        "filter"     : smoother_name,
        "n_frames"   : len(session)
    })
    return feats
