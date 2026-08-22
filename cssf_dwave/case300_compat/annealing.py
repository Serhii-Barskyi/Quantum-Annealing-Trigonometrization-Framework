from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from core.gcv import gcv_lambda, tikhonov_solve

def build_standard_schedule(n: int = 101) -> np.ndarray:
    s = np.linspace(0.0, 1.0, int(n))
    A = 5.0 * np.cos(0.5 * np.pi * s) ** 2
    B = 5.0 * np.sin(0.5 * np.pi * s) ** 2
    return np.column_stack([s, A, B, np.zeros_like(s)])

def compute_phi(delta_vec, schedule, T_us: float, s_freeze: float) -> np.ndarray:
    """Map logical anneal offsets to accumulated driver phase coordinates.

    The physical schedule is interpolated and integrated up to the declared
    freeze point.  GHz*microsecond is converted to cycles (x1000) and then to
    radians.  Returned phases are wrapped to [-pi, pi] for numerical stability.
    """
    delta = np.asarray(delta_vec, dtype=float).ravel()
    sch = np.asarray(schedule, dtype=float)
    if sch.ndim != 2 or sch.shape[1] < 2:
        raise ValueError("schedule must be an N x >=2 array [s, A(s), ...]")
    s = sch[:, 0]; A = sch[:, 1]
    grid = np.linspace(0.0, float(s_freeze), 512)
    out = np.empty(delta.size, dtype=float)
    for k, d in enumerate(delta):
        # Offset changes the local anneal fraction. This is a deterministic
        # coordinate transform for response modeling, not a claim of exact
        # closed-system dynamics.
        local_s = np.clip(grid + float(d), 0.0, 1.0)
        a_local = np.interp(local_s, s, A)
        area_ghz_us = float(np.trapz(a_local, grid)) * float(T_us)
        phase = 2.0 * np.pi * 1000.0 * area_ghz_us
        out[k] = (phase + np.pi) % (2.0 * np.pi) - np.pi
    return out

def _build_feature_matrix(phi_arr, edges) -> np.ndarray:
    phi = np.asarray(phi_arr, dtype=float)
    if phi.ndim == 1: phi = phi[None, :]
    cols = []
    for i, j in edges:
        d = phi[:, int(i)] - phi[:, int(j)]
        cols.extend([np.exp(1j * d), np.exp(-1j * d)])
    if not cols:
        return np.ones((phi.shape[0], 1), dtype=complex)
    return np.column_stack(cols)

@dataclass(slots=True)
class CSNNTAnnealingModel:
    K: int
    M0: int
    edges: list[tuple[int, int]]
    h: np.ndarray
    lam_opt: float
    residual: float
    def predict(self, phi_arr):
        X = _build_feature_matrix(phi_arr, self.edges)
        return np.real(X @ self.h).reshape(-1)

def fit_csnn_t_annealing(phi_arr, y_arr, J_dict, K: int) -> CSNNTAnnealingModel:
    phi = np.asarray(phi_arr, dtype=float); y = np.asarray(y_arr, dtype=float).reshape(-1)
    edges = sorted((int(i), int(j)) for i, j in J_dict.keys())
    X = _build_feature_matrix(phi, edges)
    lam, _, _ = gcv_lambda(X, y, n_lambdas=60, lam_range=(-10, 6))
    h = tikhonov_solve(X, y, lam).reshape(-1)
    pred = np.real(X @ h)
    denom = max(float(np.linalg.norm(y)), 1e-12)
    resid = float(np.linalg.norm(pred - y) / denom)
    return CSNNTAnnealingModel(K=int(K), M0=len(y), edges=edges, h=h, lam_opt=float(lam), residual=resid)

__all__ = ["build_standard_schedule", "compute_phi", "_build_feature_matrix", "fit_csnn_t_annealing", "CSNNTAnnealingModel"]
