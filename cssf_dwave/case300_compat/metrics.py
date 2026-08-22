from __future__ import annotations
import numpy as np

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel(); b = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]; b = b[mask]
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def compute_metrics(y_pred, y_true, meta_test=None, non_slack=None, *, rho_dc_vs_ac=None):
    """Explicit global field metrics; missing DC baseline remains unavailable."""
    yp = np.asarray(y_pred, dtype=float); yt = np.asarray(y_true, dtype=float)
    if yp.shape != yt.shape:
        raise ValueError(f"Prediction/target shape mismatch: {yp.shape} != {yt.shape}")
    if non_slack is not None:
        cols = np.asarray(list(non_slack), dtype=int)
        yp_eval, yt_eval = yp[:, cols], yt[:, cols]
    else:
        yp_eval, yt_eval = yp, yt
    err = yp_eval - yt_eval
    rho = _corr(yp_eval, yt_eval)
    rho_dc = None if rho_dc_vs_ac is None else float(rho_dc_vs_ac)
    beats = None if rho_dc is None else bool(rho > rho_dc)
    return {
        "rho_global": rho,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "rho_dc_vs_ac": rho_dc,
        "beats_dc": beats,
    }
