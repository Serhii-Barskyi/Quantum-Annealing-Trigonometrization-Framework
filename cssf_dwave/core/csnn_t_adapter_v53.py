"""Production-safe CSNN-T adapter for CSSF(QA) v53.

Scientific semantics are unchanged relative to the frozen CSNN-T/GCV model.
Only the numerical solve is replaced by an algebraically equivalent SVD
Tikhonov filter.  The original F0 files remain byte-identical and are verified
before and after fitting when ``verify_integrity=True``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.csnn_t import CSNNTModel
from core.csnn_t_adapter import (
    CSNNTAdapterError,
    CSNNTFitDiagnostics,
    CSNNTSurrogateModel,
    _normalized_target_names,
    as_bess_dataset,
)
from core.dataset import BESSDataset, CSSFDataset
from core.gcv import effective_rank, gcv_lambda
from core.gcv_stable_v53 import spectral_condition_audit_v53, tikhonov_solve_svd_v53
from core.types import SurrogateLevel
from core.validation import COLAB_PROJECT_ROOT, verify_frozen_core


def fit_csnn_t_surrogate_v53(
    training: CSSFDataset | BESSDataset,
    *,
    case: str | None = None,
    level: SurrogateLevel = SurrogateLevel.OPF,
    validation: CSSFDataset | None = None,
    test: CSSFDataset | None = None,
    target_names: Sequence[str] | None = None,
    n_lambdas: int = 100,
    lam_range: tuple[float, float] = (-12.0, 4.0),
    lam_fixed: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    verify_integrity: bool = True,
    project_root: str | Path = COLAB_PROJECT_ROOT,
) -> CSNNTSurrogateModel:
    """Fit the canonical CSSF ridge objective without normal equations."""
    if not isinstance(level, SurrogateLevel):
        raise TypeError("level must be SurrogateLevel.")
    if isinstance(n_lambdas, bool) or not isinstance(n_lambdas, int) or n_lambdas < 2:
        raise CSNNTAdapterError("n_lambdas must be an integer >= 2.")
    if len(lam_range) != 2:
        raise CSNNTAdapterError("lam_range must contain two values.")
    lam_low, lam_high = map(float, lam_range)
    if not (math.isfinite(lam_low) and math.isfinite(lam_high) and lam_low < lam_high):
        raise CSNNTAdapterError("lam_range must contain finite increasing log10 bounds.")
    if lam_fixed is not None:
        lam_fixed = float(lam_fixed)
        if not math.isfinite(lam_fixed) or lam_fixed <= 0.0:
            raise CSNNTAdapterError("lam_fixed must be finite and positive.")

    if verify_integrity:
        verify_frozen_core(project_root)

    dataset = as_bess_dataset(
        training,
        case=case,
        validation=validation,
        test=test,
        metadata=metadata,
    )

    if lam_fixed is None:
        lam_opt, lam_grid, gcv_values = gcv_lambda(
            dataset.X_train,
            dataset.y_train,
            n_lambdas=n_lambdas,
            lam_range=(lam_low, lam_high),
        )
        used_fixed_lambda = False
        diagnostic_n_lambdas = n_lambdas
    else:
        lam_opt = lam_fixed
        lam_grid = np.empty(0, dtype=np.float64)
        gcv_values = np.empty(0, dtype=np.float64)
        used_fixed_lambda = True
        diagnostic_n_lambdas = 0

    H = tikhonov_solve_svd_v53(dataset.X_train, dataset.y_train, float(lam_opt))
    H = np.ascontiguousarray(H, dtype=np.complex128)
    H.setflags(write=False)
    legacy_model = CSNNTModel(
        H=H,
        lam_opt=float(lam_opt),
        case=dataset.case,
        M=dataset.M_complex,
        n=dataset.n,
    )

    diagnostics = CSNNTFitDiagnostics(
        lam_opt=float(lam_opt),
        effective_rank=effective_rank(dataset.X_train, float(lam_opt)),
        n_lambdas=diagnostic_n_lambdas,
        lam_range=(lam_low, lam_high),
        used_fixed_lambda=used_fixed_lambda,
        lam_grid=lam_grid,
        gcv_values=gcv_values,
    )
    condition_audit = spectral_condition_audit_v53(
        dataset.X_train,
        float(lam_opt),
        lam_grid=lam_grid if not used_fixed_lambda else None,
    )
    names = _normalized_target_names(target_names, count=dataset.n)
    model_metadata = {
        **dict(dataset.metadata),
        **({} if metadata is None else dict(metadata)),
        "surrogate_level": level.value,
        "frozen_scientific_core": True,
        "numerical_realization": "svd_tikhonov_v53",
        "normal_equations_formed": False,
        "conditioning": condition_audit.as_dict(),
    }

    result = CSNNTSurrogateModel(
        frozen_model=legacy_model,
        level=level,
        target_names=names,
        training_fingerprint=dataset.fingerprint(),
        diagnostics=diagnostics,
        metadata=model_metadata,
    )
    if verify_integrity:
        verify_frozen_core(project_root)
    return result


# Same public name for versioned callers.
fit_csnn_t_surrogate = fit_csnn_t_surrogate_v53

__all__ = [
    "CSNNTAdapterError",
    "CSNNTFitDiagnostics",
    "CSNNTSurrogateModel",
    "as_bess_dataset",
    "fit_csnn_t_surrogate",
    "fit_csnn_t_surrogate_v53",
]
