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
core/csnn_t.py — CSNN-T^OPF: analytical LSF surrogate.

Mathematics (Step 1, formula 1.8):
    H_λ̂ = (X_tr^H X_tr + λ̂ I)^{-1} X_tr^H · y_tr,  H_λ̂ ∈ ℂ^{M × n}

    λ̂ is selected by GCV (gcv.py) for all cases (delta_r > 0 everywhere).

    Prediction: ŷ^new = Re(X^new @ H_λ̂)  — linear, O(M·n), no iterations.

    Physical meaning of H_λ̂: j-th column h_j ∈ ℂ^M — vector of spectral
    coefficients of LSF_j decomposition in basis {e^{ik(θ_i-θ_j)}}_{(i,j)∈E,k=±1}.
    Each element h_{j,m} — amplitude of m-th harmonic of power flow on edge (i,j)
    in the loss sensitivity decomposition of bus j.

    Theorem 1: ‖f_AC - f_K‖_{L²} ≤ ‖f_AC‖_{H^s} · K^{-s}
    → CSNN-T outperforms DC approximation: ρ(ŷ, y_AC) > rho_dc_vs_ac.

    Theorem 2: support of LSF_DC ⊆ Λ_DC = {±(e_i-e_j):(i,j)∈E}
    → basis X is analytically optimal, not data-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from core.dataset import BESSDataset
from core.gcv import gcv_lambda, tikhonov_solve


# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass
class CSNNTModel:
    """
    Trained CSNN-T^OPF model.

    Attributes
    ----------
    H        : (M_complex, n) complex — spectral coefficient matrix
    lam_opt  : float — optimal λ by GCV
    case     : str — case name
    M        : int — number of features M_complex
    n        : int — number of buses
    """
    H:       NDArray[np.complexfloating]
    lam_opt: float
    case:    str
    M:       int
    n:       int

    def predict(self, X_new: NDArray[np.complexfloating]) -> NDArray[np.floating]:
        """
        LSF prediction for new scenarios.

        ŷ = Re(X_new @ H)

        Parameters
        ----------
        X_new : (N_new, M_complex) complex

        Returns
        -------
        (N_new, n) float — predicted LSF
        """
        assert X_new.shape[1] == self.M, (
            f"X_new.shape[1]={X_new.shape[1]} ≠ M={self.M}"
        )
        return np.real(X_new @ self.H)

    def mean_lsf(self, X: NDArray[np.complexfloating]) -> NDArray[np.floating]:
        """
        Mean predicted LSF across scenarios: E_ξ[LSF_i].

        Returns
        -------
        (n,) float — mean LSF across all provided scenarios
        """
        return self.predict(X).mean(axis=0)


# ── Training ──────────────────────────────────────────────────────────────────

def fit_csnn_t(
    ds: BESSDataset,
    n_lambdas: int = 100,
    lam_range: tuple = (-12, 4),
    lam_fixed: Optional[float] = None,
) -> CSNNTModel:
    """
    Trains CSNN-T^OPF on a dataset.

    Algorithm:
        1. GCV selects λ̂ from X_train, y_train
        2. h_λ̂ = (X_tr^H X_tr + λ̂ I)^{-1} X_tr^H · y_tr

    Parameters
    ----------
    ds         : BESSDataset
    n_lambdas  : number of λ points for GCV
    lam_range  : (log10_min, log10_max) for λ
    lam_fixed  : fixed λ (without GCV) — for testing

    Returns
    -------
    CSNNTModel
    """
    X_tr = ds.X_train   # (N_train, M_complex) complex
    y_tr = ds.y_train   # (N_train, n) float

    # ── Lambda selection ──────────────────────────────────────────────────────
    if lam_fixed is not None:
        lam_opt = float(lam_fixed)
    else:
        # GCV for all cases (delta_r > 0 for all three)
        lam_opt, _, _ = gcv_lambda(X_tr, y_tr, n_lambdas=n_lambdas,
                                   lam_range=lam_range)

    # ── Tikhonov solution ─────────────────────────────────────────────────────
    # H = (X^H X + λI)^{-1} X^H y  ∈ ℂ^{M × n}
    H = tikhonov_solve(X_tr, y_tr, lam_opt)

    return CSNNTModel(
        H=H,
        lam_opt=lam_opt,
        case=ds.case,
        M=ds.M_complex,
        n=ds.n,
    )
