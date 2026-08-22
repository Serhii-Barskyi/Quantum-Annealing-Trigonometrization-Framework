"""Full CSSF QA-response surrogate and active-control layer.

The implementation deliberately reuses the original CSSF mathematical path:
``spectral.frequency_support`` -> ``spectral.feature_matrix`` ->
``core.csnn_t_adapter`` -> frozen ``core/csnn_t.py`` and ``core/gcv.py``.
No alternate ridge solver or replacement CSSF estimator is implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Mapping, Any
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.dataset import CSSFDataset
from core.types import SurrogateLevel
from core.csnn_t_adapter_v53 import CSNNTSurrogateModel, fit_csnn_t_surrogate
from spectral.frequency_support import FrequencySupport, signed_axis_support, pairwise_support, total_l1_support
from spectral.feature_matrix import toric_feature_matrix


class CSSFControlError(RuntimeError):
    """Raised when a QA-response CSSF control model violates its contract."""


def build_support(
    n_dimensions: int,
    *,
    mode: str = "signed_axes",
    order: int = 1,
    include_zero: bool = True,
) -> FrequencySupport:
    normalized = str(mode).strip().lower()
    if normalized == "signed_axes":
        return signed_axis_support(n_dimensions, max_harmonic=int(order), include_zero=include_zero)
    if normalized == "pairwise":
        if int(order) != 1:
            raise CSSFControlError("pairwise support is first-order in the v38 D-Wave protocol")
        return pairwise_support(n_dimensions, include_axes=True, include_sums=True, include_differences=True, include_zero=include_zero)
    if normalized == "total_l1":
        return total_l1_support(n_dimensions, max_l1_order=int(order), include_zero=include_zero)
    raise CSSFControlError(f"Unknown CSSF support mode {mode!r}")


def _target_names(names: Sequence[str], count: int) -> tuple[str, ...]:
    values = tuple(str(x).strip() for x in names)
    if len(values) != count or any(not x for x in values) or len(set(values)) != len(values):
        raise CSSFControlError("target_names must be unique, non-empty, and match target columns")
    return values


@dataclass(frozen=True)
class CSSFQAResponseModel:
    """Original-core CSNN-T model plus calibrated decision uncertainty.

    Uncertainty is an external decision layer: the predictive mean always comes
    from the frozen CSNN-T model.  The scale combines held-out residual variance
    with Tikhonov leverage, preserving the implemented CSSF elite-plus-information
    acquisition while exposing an uncertainty-aware score.
    """

    support: FrequencySupport
    model: CSNNTSurrogateModel
    target_names: tuple[str, ...]
    residual_scale: NDArray[np.float64]
    gram_inverse: NDArray[np.complex128]
    coverage_quantile: NDArray[np.float64]
    calibration_fingerprint: str

    def _features(self, operator_phase: ArrayLike) -> NDArray[np.complex128]:
        return toric_feature_matrix(operator_phase, self.support, wrap_coordinates=True)

    def predict(self, operator_phase: ArrayLike) -> NDArray[np.float64]:
        return self.model.predict(self._features(operator_phase))

    def leverage(self, operator_phase: ArrayLike) -> NDArray[np.float64]:
        phi = self._features(operator_phase)
        values = np.real(np.einsum("bi,ij,bj->b", phi.conj(), self.gram_inverse, phi))
        return np.maximum(values, 0.0)

    def predict_std(self, operator_phase: ArrayLike) -> NDArray[np.float64]:
        lev = self.leverage(operator_phase)[:, None]
        return np.sqrt(1.0 + lev) * self.residual_scale[None, :]

    def intervals(self, operator_phase: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mean = self.predict(operator_phase)
        std = self.predict_std(operator_phase)
        radius = std * self.coverage_quantile[None, :]
        return mean - radius, mean + radius

    def empirical_coverage(self, operator_phase: ArrayLike, targets: ArrayLike) -> NDArray[np.float64]:
        y = np.asarray(targets, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        lo, hi = self.intervals(operator_phase)
        if y.shape != lo.shape:
            raise CSSFControlError("coverage target shape mismatch")
        return np.mean((y >= lo) & (y <= hi), axis=0)

    def acquisition(
        self,
        candidates_operator_phase: ArrayLike,
        *,
        target: str = "elite_probability",
        maximize: bool = True,
        uncertainty_weight: float = 1.0,
        leverage_weight: float = 0.25,
        feasibility_target: str | None = "feasibility_probability",
        minimum_feasibility: float = 0.0,
    ) -> tuple[int, NDArray[np.float64], Mapping[str, Any]]:
        if target not in self.target_names:
            raise CSSFControlError(f"Unknown target {target!r}")
        theta = np.asarray(candidates_operator_phase, dtype=np.float64)
        pred = self.predict(theta)
        std = self.predict_std(theta)
        lev = self.leverage(theta)
        j = self.target_names.index(target)
        base = pred[:, j] if maximize else -pred[:, j]
        if np.ptp(lev) > 1e-15:
            lev_norm = (lev - np.min(lev)) / np.ptp(lev)
        else:
            lev_norm = np.zeros_like(lev)
        score = base + float(uncertainty_weight) * std[:, j] + float(leverage_weight) * lev_norm
        feasible_mask = np.ones(theta.shape[0], dtype=bool)
        if feasibility_target is not None and feasibility_target in self.target_names:
            fj = self.target_names.index(feasibility_target)
            feasible_mask = pred[:, fj] >= float(minimum_feasibility)
            score = np.where(feasible_mask, score, -np.inf)
        if not np.any(np.isfinite(score)):
            raise CSSFControlError("No candidate passes the acquisition feasibility gate")
        idx = int(np.nanargmax(score))
        return idx, score, {
            "target": target,
            "maximize": bool(maximize),
            "predicted_target": float(pred[idx, j]),
            "predictive_std": float(std[idx, j]),
            "leverage": float(lev[idx]),
            "acquisition_score": float(score[idx]),
            "passes_feasibility_gate": bool(feasible_mask[idx]),
        }


def fit_cssf_qa_response(
    train_operator_phase: ArrayLike,
    train_targets: ArrayLike,
    *,
    calibration_operator_phase: ArrayLike,
    calibration_targets: ArrayLike,
    target_names: Sequence[str],
    project_root: str | Path,
    support_mode: str = "signed_axes",
    support_order: int = 1,
    n_lambdas: int = 100,
    lam_range: tuple[float, float] = (-12.0, 4.0),
    nominal_coverage: float = 0.90,
    metadata: Mapping[str, Any] | None = None,
) -> CSSFQAResponseModel:
    theta = np.asarray(train_operator_phase, dtype=np.float64)
    y = np.asarray(train_targets, dtype=np.float64)
    theta_cal = np.asarray(calibration_operator_phase, dtype=np.float64)
    y_cal = np.asarray(calibration_targets, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[0] < 4:
        raise CSSFControlError("Training operator-phase coordinates must be a non-empty matrix")
    if y.ndim == 1:
        y = y[:, None]
    if y_cal.ndim == 1:
        y_cal = y_cal[:, None]
    if y.shape[0] != theta.shape[0] or theta_cal.ndim != 2 or theta_cal.shape[1] != theta.shape[1] or y_cal.shape[0] != theta_cal.shape[0] or y_cal.shape[1] != y.shape[1]:
        raise CSSFControlError("Training/calibration arrays are not aligned")
    names = _target_names(target_names, y.shape[1])
    if not (0.5 < float(nominal_coverage) < 1.0):
        raise CSSFControlError("nominal_coverage must lie in (0.5,1)")
    support = build_support(theta.shape[1], mode=support_mode, order=support_order, include_zero=True)
    phi = toric_feature_matrix(theta, support, wrap_coordinates=True)
    phi_cal = toric_feature_matrix(theta_cal, support, wrap_coordinates=True)
    # Fail closed on an underidentified dictionary, following the contract's
    # initial feature-count rule unless an explicit stronger proof is supplied.
    if phi.shape[1] > theta.shape[0] // 2:
        raise CSSFControlError(
            f"CSSF support is underidentified: features={phi.shape[1]} > floor(train/2)={theta.shape[0]//2}"
        )
    ds = CSSFDataset(phi, y, metadata={"role": "qa_response", **({} if metadata is None else dict(metadata))})
    model = fit_csnn_t_surrogate(
        ds,
        case="cssf_qa_response",
        level=SurrogateLevel.DIGITIZED_QA,
        target_names=names,
        n_lambdas=int(n_lambdas),
        lam_range=lam_range,
        metadata={"surrogated_system": "quantum_annealing_response", **({} if metadata is None else dict(metadata))},
        verify_integrity=True,
        project_root=project_root,
    )
    pred_cal = model.predict(phi_cal)
    residual = y_cal - pred_cal
    scale = np.sqrt(np.mean(residual**2, axis=0) + 1e-15)
    standardized = np.abs(residual) / scale[None, :]
    q = np.quantile(standardized, float(nominal_coverage), axis=0, method="higher")
    gram = phi.conj().T @ phi + model.lam_opt * np.eye(phi.shape[1], dtype=np.complex128)
    gram_inv = np.linalg.pinv(gram, hermitian=True)
    for arr in (scale, q, gram_inv):
        arr.setflags(write=False)
    import hashlib, json
    payload = {
        "support": support.frequencies.tolist(),
        "training_fingerprint": model.training_fingerprint,
        "lam_opt": model.lam_opt,
        "calibration_rows": int(theta_cal.shape[0]),
        "nominal_coverage": float(nominal_coverage),
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CSSFQAResponseModel(
        support=support,
        model=model,
        target_names=names,
        residual_scale=np.ascontiguousarray(scale),
        gram_inverse=np.ascontiguousarray(gram_inv),
        coverage_quantile=np.ascontiguousarray(q),
        calibration_fingerprint=fingerprint,
    )
