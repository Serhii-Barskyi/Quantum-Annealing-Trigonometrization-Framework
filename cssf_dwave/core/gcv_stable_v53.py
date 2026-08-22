"""Numerically stable CSSF(QA) v53 Tikhonov solve and identifiability audit.

The frozen ``core/gcv.py`` remains the canonical historical implementation and
continues to supply the GCV lambda selection.  This versioned module changes
only the numerical linear-algebra realization of the same ridge estimator:

    argmin_H ||Y-XH||_F^2 + lambda ||H||_F^2.

It avoids explicitly forming X^H X or X X^H, whose condition number is the
square of the condition number of X.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


class StableTikhonovError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SpectralConditionAudit:
    n_samples: int
    n_features: int
    numerical_rank: int
    sigma_max: float
    sigma_min_raw: float
    sigma_min_effective: float
    condition_number_x_raw: float
    condition_number_x_effective: float
    condition_number_normal_equations_raw_estimate: float
    condition_number_normal_equations_effective_estimate: float
    lambda_value: float
    lambda_over_sigma_max_sq: float
    lambda_over_sigma_min_sq: float
    gcv_boundary_hit: bool
    solver: str = "svd_tikhonov_v53"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _svd_and_tol(X: NDArray[np.complexfloating], rcond: float | None = None):
    Xc = np.asarray(X, dtype=np.complex128)
    if Xc.ndim != 2 or min(Xc.shape) < 1:
        raise StableTikhonovError("X must be a non-empty two-dimensional matrix.")
    if not np.isfinite(Xc.real).all() or not np.isfinite(Xc.imag).all():
        raise StableTikhonovError("X contains non-finite values.")
    U, s, Vh = np.linalg.svd(Xc, full_matrices=False)
    if rcond is None:
        rcond = np.finfo(np.float64).eps * max(Xc.shape)
    rcond = float(rcond)
    if not math.isfinite(rcond) or rcond < 0.0:
        raise StableTikhonovError("rcond must be finite and non-negative.")
    tol = (0.0 if s.size == 0 else float(s[0])) * rcond
    return Xc, U, s, Vh, tol


def tikhonov_solve_svd_v53(
    X: NDArray[np.complexfloating],
    y: NDArray,
    lam: float,
    *,
    rcond: float | None = None,
) -> NDArray[np.complex128]:
    """Solve the exact Tikhonov objective through SVD spectral filtering."""
    lam = float(lam)
    if not math.isfinite(lam) or lam < 0.0:
        raise StableTikhonovError("lam must be finite and non-negative.")
    Xc, U, s, Vh, tol = _svd_and_tol(X, rcond=rcond)
    yc = np.asarray(y)
    if yc.ndim not in (1, 2) or yc.shape[0] != Xc.shape[0]:
        raise StableTikhonovError("y must have shape (N,) or (N,p) aligned with X.")
    yc = np.asarray(yc, dtype=np.complex128)
    if not np.isfinite(yc.real).all() or not np.isfinite(yc.imag).all():
        raise StableTikhonovError("y contains non-finite values.")

    if lam == 0.0:
        filt = np.zeros_like(s, dtype=np.float64)
        keep = s > tol
        filt[keep] = 1.0 / s[keep]
    else:
        # Stable even for very small singular values; no normal equations formed.
        filt = s / (s * s + lam)

    projected = U.conj().T @ yc
    solution = (Vh.conj().T * filt) @ projected
    return np.asarray(solution, dtype=np.complex128)


def spectral_condition_audit_v53(
    X: NDArray[np.complexfloating],
    lam: float,
    *,
    lam_grid: NDArray | None = None,
    rcond: float | None = None,
) -> SpectralConditionAudit:
    """Quantify conditioning without using the unstable Gram solve."""
    Xc, _U, s, _Vh, tol = _svd_and_tol(X, rcond=rcond)
    rank = int(np.count_nonzero(s > tol))
    sigma_max = float(s[0]) if s.size else 0.0
    if rank:
        sigma_min_effective = float(s[rank - 1])
        kappa_effective = sigma_max / sigma_min_effective if sigma_min_effective > 0.0 else math.inf
    else:
        sigma_min_effective = 0.0
        kappa_effective = math.inf
    sigma_min_raw = float(s[-1]) if s.size else 0.0
    kappa_raw = sigma_max / sigma_min_raw if sigma_min_raw > 0.0 else math.inf
    kappa_gram_raw = kappa_raw * kappa_raw if math.isfinite(kappa_raw) else math.inf
    kappa_gram_effective = kappa_effective * kappa_effective if math.isfinite(kappa_effective) else math.inf
    lam = float(lam)
    if lam < 0.0 or not math.isfinite(lam):
        raise StableTikhonovError("lam must be finite and non-negative.")
    lam_grid_arr = None if lam_grid is None else np.asarray(lam_grid, dtype=float).reshape(-1)
    boundary = bool(
        lam_grid_arr is not None
        and lam_grid_arr.size > 0
        and (np.isclose(lam, lam_grid_arr[0]) or np.isclose(lam, lam_grid_arr[-1]))
    )
    smax2 = sigma_max * sigma_max
    smin2 = sigma_min_raw * sigma_min_raw
    return SpectralConditionAudit(
        n_samples=int(Xc.shape[0]),
        n_features=int(Xc.shape[1]),
        numerical_rank=rank,
        sigma_max=sigma_max,
        sigma_min_raw=sigma_min_raw,
        sigma_min_effective=sigma_min_effective,
        condition_number_x_raw=float(kappa_raw),
        condition_number_x_effective=float(kappa_effective),
        condition_number_normal_equations_raw_estimate=float(kappa_gram_raw),
        condition_number_normal_equations_effective_estimate=float(kappa_gram_effective),
        lambda_value=lam,
        lambda_over_sigma_max_sq=(float(lam / smax2) if smax2 > 0 else math.inf),
        lambda_over_sigma_min_sq=(float(lam / smin2) if smin2 > 0 else math.inf),
        gcv_boundary_hit=boundary,
    )


__all__ = [
    "StableTikhonovError",
    "SpectralConditionAudit",
    "tikhonov_solve_svd_v53",
    "spectral_condition_audit_v53",
]
