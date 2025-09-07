import numpy as np
from scipy.stats import iqr, entropy

# ---------- generic helpers ----------
def vel(sig, dt):        # first derivative
    return np.gradient(sig, dt, edge_order=2)

def acc(sig, dt):        # second derivative
    return np.gradient(vel(sig, dt), dt, edge_order=2)

# ---------- 1. CROSS-CORRELATION ----------
def cross_corr(a, b):
    """Pearson ρ between two equal-length signals."""
    a, b = np.asarray(a), np.asarray(b)
    return np.corrcoef(a, b)[0, 1]

# ---------- 2. ENTROPY (Shannon, base e) ----------
def signal_entropy(sig, n_bins=32):
    """Histogram-based entropy of a 1-D signal."""
    p, _ = np.histogram(sig, bins=n_bins, density=True)
    p = p[p > 0]
    return entropy(p)

# ---------- 3. IQR-based statistics ----------
def iqr_stat(sig, order, dt):
    """Inter-quartile range of pos/vel/acc depending on order."""
    if order == 0:
        s = sig
    elif order == 1:
        s = vel(sig, dt)
    elif order == 2:
        s = acc(sig, dt)
    else:
        raise ValueError("order must be 0, 1 or 2")
    return iqr(s)

# ---------- 4. MEDIAN statistics ----------
def median_position(sig):
    return np.median(sig)

def median_velocity(sig, dt):
    return np.median(np.abs(vel(sig, dt)))

# ---------- 5. ANGULAR helpers ----------
def angle(p_prox, p_joint, p_dist):
    """
    Angle (rad) at 'p_joint' formed by three XY points of shape (n,2).
    """
    v1 = p_prox - p_joint
    v2 = p_dist - p_joint
    cosang = np.einsum('ij,ij->i', v1, v2) / (
             np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.arccos(np.clip(cosang, -1.0, 1.0))

def angular_vel(theta, dt):
    return np.gradient(theta, dt, edge_order=2)

def angular_acc(theta, dt):
    return np.gradient(angular_vel(theta, dt), dt, edge_order=2)

def rms_jerk(sig, dt):
    """Root-mean-square jerk (3rd derivative of position)."""
    jerk = np.gradient(acc(sig, dt), dt, edge_order=2)
    return np.sqrt(np.mean(jerk**2))
