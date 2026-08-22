# -*- coding: utf-8 -*-
# Author: Serhii Barskyi | https://www.linkedin.com/in/serhii-barskyi/
# Data Science Course: https://preply.com/en/tutor/7756455
# Framework: Complex Spectral Surrogate Framework (CSSF)
#
# Licensed under the Apache License, Version 2.0.
# You may not use this file except in compliance with the License.
# Full license text: https://www.apache.org/licenses/LICENSE-2.0
#
# Attribution required: if you use this code, please cite:
# Serhii Barskyi, Complex Spectral Surrogate Framework (CSSF),
# https://www.linkedin.com/in/serhii-barskyi/
"""
core/gcv.py — selection of regularization parameter λ via GCV.

Mathematics (formula 2.12 of the document):
    λ̂ = argmin_{λ>0} GCV(λ),  GCV(λ) = ‖y - X h_λ‖² / [N - tr(H_λ)]²

    Hat matrix: H_λ = X (X^H X + λ I)^{-1} X^H ∈ ℂ^{N×N}
    Tikhonov solution: h_λ = (X^H X + λ I)^{-1} X^H y

    Via SVD: X = U Σ V^H (full or truncated)
        tr(H_λ) = ∑_k σ_k² / (σ_k² + λ)   — effective parameters
        ‖y - H_λ y‖² = ‖(I - H_λ) y‖²      — computed efficiently

    Computational complexity:
        SVD: O(N · min(N,M)²) — once
        GCV on grid of n_lambdas points: O(r · n_lambdas) where r = rank(X)

    Applied to all three cases (delta_r > 0 everywhere):
        case14: delta_r = 4,  case30: delta_r = 5,  case57: delta_r = 3

    Interpretation: at λ→0: tr(H)→rank(X), high solution variance.
                    at λ→∞: tr(H)→0, high bias (underfitting).
                    GCV finds the balance without a validation set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def gcv_lambda(
    X: NDArray[np.complexfloating],
    y: NDArray[np.floating],
    n_lambdas: int = 100,
    lam_range: Tuple[float, float] = (-12, 4),
) -> Tuple[float, NDArray, NDArray]:
    """
    Selects optimal λ via GCV using SVD.

    Algorithm:
        1. SVD: X = U Σ V^H
        2. Project y: Ũ = U^H y (into the space of left singular vectors)
        3. For each λ on the grid:
           tr(H_λ) = ∑_k d_k / (d_k + λ),  d_k = σ_k²
           ‖(I-H_λ)y‖² = ‖y‖² - ∑_k (d_k/(d_k+λ)) · |Ũ_k|² · 2 + ... (via SVD)
           GCV(λ) = residual² / [N - tr(H_λ)]²
        4. Return λ with minimum GCV

    Parameters
    ----------
    X          : (N, M) complex — feature matrix
    y          : (N, p) or (N,) float — target values
    n_lambdas  : number of points on the logarithmic grid
    lam_range  : (log10_min, log10_max) for λ

    Returns
    -------
    lam_opt    : float — optimal λ
    lam_grid   : (n_lambdas,) — λ grid
    gcv_values : (n_lambdas,) — GCV(λ) values
    """
    N, M = X.shape
    y = y[:, None] if y.ndim == 1 else y
    # y: (N, p)

    # ── SVD ───────────────────────────────────────────────────────────────────
    # full_matrices=False: U ∈ ℂ^{N×r}, s ∈ ℝ^r, Vt ∈ ℂ^{r×M}, r=min(N,M)
    U, sv, Vt = np.linalg.svd(X, full_matrices=False)
    d = sv ** 2  # (r,) — eigenvalues of X^H X

    # Project y onto left singular vectors: Uy ∈ ℂ^{r × p}
    Uy = U.conj().T @ y  # (r, p)

    # ‖y‖_F² for each column
    y_sq = np.sum(np.real(y) ** 2)  # scalar (total sum)

    # ── Lambda grid ───────────────────────────────────────────────────────────
    lam_grid = np.logspace(lam_range[0], lam_range[1], n_lambdas)
    gcv_values = np.full(n_lambdas, np.inf)

    for idx, lam in enumerate(lam_grid):
        # d_k / (d_k + λ) — vector of filter coefficients
        filt = d / (d + lam)  # (r,)

        # tr(H_λ) = ∑_k filt_k
        tr_H = float(np.sum(filt))

        # Effective degrees of freedom: N - tr(H_λ)
        dof = N - tr_H
        if abs(dof) < 1.0:
            # Degenerate case — skip
            continue

        # ‖(I - H_λ) y‖_F²:
        # H_λ y = U diag(filt) U^H y = U diag(filt) Uy → back
        # (I - H_λ) y = y - U diag(filt) Uy
        # ‖...‖_F² = ‖y‖_F² - 2 Re<U diag(filt) Uy, y> + ‖U diag(filt) Uy‖_F²
        # Efficiently via Uy:
        # ‖y‖_F² - ∑_k filt_k · |Uy_k|² · 2 + ∑_k filt_k² · |Uy_k|²
        #        = ‖y‖_F² - ∑_k filt_k(2 - filt_k) · |Uy_k|²
        Uy_sq = np.sum(np.abs(Uy) ** 2, axis=1)  # (r,) — sum over y columns
        res_sq = y_sq - float(np.sum(filt * (2.0 - filt) * Uy_sq))
        res_sq = max(res_sq, 0.0)  # numerical safeguard

        gcv_values[idx] = res_sq / (dof ** 2 + 1e-15)

    best_idx = int(np.argmin(gcv_values))
    lam_opt  = float(lam_grid[best_idx])

    return lam_opt, lam_grid, gcv_values


def tikhonov_solve(
    X: NDArray[np.complexfloating],
    y: NDArray,
    lam: float,
) -> NDArray[np.complexfloating]:
    """
    Solves the Tikhonov problem: h = (X^H X + λ I)^{-1} X^H y.

    Parameters
    ----------
    X   : (N, M) complex
    y   : (N, p) or (N,) float/complex
    lam : regularization parameter

    Returns
    -------
    H : (M, p) complex — spectral coefficient matrix
    """
    import scipy.linalg as _sla
    N, M = X.shape

    if N < M:
        # Woodbury path: h = X^H (X X^H + λI)^{-1} y  [N×N system, O(N²M)]
        # Memory: N×N instead of M×M — critical for large networks (N << M)
        XXT  = X @ X.conj().T                            # (N, N)
        A    = XXT + lam * np.eye(N, dtype=XXT.dtype)   # (N, N)
        coef = _sla.solve(A, y, check_finite=False)      # (N, p)
        return X.conj().T @ coef                         # (M, p)
    else:
        # Standard path: (X^H X + λI)^{-1} X^H y  [M×M system]
        XHX = X.conj().T @ X                             # (M, M)
        XHy = X.conj().T @ y                             # (M, p)
        return _sla.solve(XHX + lam * np.eye(M, dtype=XHX.dtype),
                          XHy, check_finite=False)       # (M, p)


def effective_rank(
    X: NDArray[np.complexfloating],
    lam: float,
) -> float:
    """
    Effective model rank at given λ: tr(H_λ) = ∑ σ_k²/(σ_k²+λ).
    At λ→0: tr(H)→rank(X). At λ→∞: tr(H)→0.
    """
    _, sv, _ = np.linalg.svd(X, full_matrices=False)
    d = sv ** 2
    return float(np.sum(d / (d + lam)))
