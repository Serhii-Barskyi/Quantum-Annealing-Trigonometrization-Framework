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

"""Leakage-safe model-selection primitives for QA-response surrogates.

The module contains deterministic numerical helpers only.  It does not fit the
frozen CSNN-T implementation itself and it never initializes CUDA, Qiskit,
Ocean, HiGHS, pandapower, or a QPU client.  The Colab protocol uses these
helpers to compare candidate CSSF coordinate/support/target-scaling designs on
one declared validation partition, confirm the locked winner on a separate
calibration partition, and open the strict QA-OOD partition only afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

from spectral.frequency_support import (
    FrequencySupport,
    pairwise_support,
    signed_axis_support,
    total_l1_support,
)


REAL_DTYPE = np.dtype(np.float64)
SUPPORTED_SUPPORT_FAMILIES = ("signed_axes", "pairwise", "total_l1")
SUPPORTED_TARGET_MODES = (
    "raw_joint",
    "standardized_joint",
    "grouped_standardized",
)
DEFAULT_SELECTION_TARGETS = (
    "feasibility_probability",
    "success_probability",
    "elite_probability",
    "cvar_05",
    "mean_energy",
)


class SurrogateSelectionError(ValueError):
    """Raised when a surrogate-selection contract is invalid."""


def _finite_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.ascontiguousarray(values, dtype=REAL_DTYPE)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise SurrogateSelectionError(f"{name} must be a non-empty matrix.")
    if not np.isfinite(result).all():
        raise SurrogateSelectionError(f"{name} contains non-finite values.")
    return result


def _finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.ascontiguousarray(values, dtype=REAL_DTYPE).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise SurrogateSelectionError(
            f"{name} must be a non-empty finite vector."
        )
    return result


def _bool_mask(values: ArrayLike, *, size: int, name: str) -> NDArray[np.bool_]:
    result = np.ascontiguousarray(values, dtype=np.bool_).reshape(-1)
    if result.size != size:
        raise SurrogateSelectionError(
            f"{name} has size {result.size}; expected {size}."
        )
    if not result.any():
        raise SurrogateSelectionError(f"{name} must select at least one row.")
    return result


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    """Train-only affine map from raw coordinates to a stable phase scale."""

    center: NDArray[np.float64]
    scale: NDArray[np.float64]
    phase_radius: float = math.pi

    def __post_init__(self) -> None:
        center = _finite_vector(self.center, name="center")
        scale = _finite_vector(self.scale, name="scale")
        if center.shape != scale.shape:
            raise SurrogateSelectionError("center and scale shapes must match.")
        if np.any(scale <= 0.0):
            raise SurrogateSelectionError("scale values must be positive.")
        radius = float(self.phase_radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise SurrogateSelectionError("phase_radius must be positive.")
        center.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "phase_radius", radius)

    @property
    def n_dimensions(self) -> int:
        return int(self.center.size)

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = _finite_matrix(values, name="coordinates")
        if matrix.shape[1] != self.n_dimensions:
            raise SurrogateSelectionError(
                "Coordinate dimension does not match fitted transform."
            )
        return np.ascontiguousarray(
            self.phase_radius * (matrix - self.center) / self.scale,
            dtype=REAL_DTYPE,
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-CoordinateTransform-v1\0")
        digest.update(self.center.tobytes(order="C"))
        digest.update(self.scale.tobytes(order="C"))
        digest.update(np.asarray([self.phase_radius], dtype=REAL_DTYPE).tobytes())
        return digest.hexdigest()


def fit_coordinate_transform(
    values: ArrayLike,
    fit_mask: ArrayLike,
    *,
    phase_radius: float = math.pi,
) -> CoordinateTransform:
    """Fit a robust train-only center/range transform without OOD leakage."""

    matrix = _finite_matrix(values, name="coordinates")
    mask = _bool_mask(fit_mask, size=matrix.shape[0], name="fit_mask")
    fitted = matrix[mask]
    lower = np.min(fitted, axis=0)
    upper = np.max(fitted, axis=0)
    center = 0.5 * (lower + upper)
    scale = upper - lower
    fallback = np.std(fitted, axis=0, ddof=1) if fitted.shape[0] > 1 else np.ones(matrix.shape[1])
    scale = np.where(scale > 1.0e-12, scale, fallback)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return CoordinateTransform(center=center, scale=scale, phase_radius=phase_radius)


@dataclass(frozen=True, slots=True)
class TargetTransform:
    """Train-only target normalization used before CSNN-T/GCV fitting."""

    center: NDArray[np.float64]
    scale: NDArray[np.float64]

    def __post_init__(self) -> None:
        center = _finite_vector(self.center, name="target center")
        scale = _finite_vector(self.scale, name="target scale")
        if center.shape != scale.shape:
            raise SurrogateSelectionError(
                "Target center and scale shapes must match."
            )
        if np.any(scale <= 0.0):
            raise SurrogateSelectionError("Target scales must be positive.")
        center.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = _finite_matrix(values, name="targets")
        if matrix.shape[1] != self.center.size:
            raise SurrogateSelectionError("Target dimension mismatch.")
        return np.ascontiguousarray((matrix - self.center) / self.scale)

    def inverse(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = _finite_matrix(values, name="scaled predictions")
        if matrix.shape[1] != self.center.size:
            raise SurrogateSelectionError("Prediction dimension mismatch.")
        return np.ascontiguousarray(matrix * self.scale + self.center)


def fit_target_transform(values: ArrayLike, fit_mask: ArrayLike) -> TargetTransform:
    matrix = _finite_matrix(values, name="targets")
    mask = _bool_mask(fit_mask, size=matrix.shape[0], name="fit_mask")
    fitted = matrix[mask]
    center = np.mean(fitted, axis=0)
    scale = np.std(fitted, axis=0, ddof=1) if fitted.shape[0] > 1 else np.ones(matrix.shape[1])
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return TargetTransform(center=center, scale=scale)


@dataclass(frozen=True, slots=True)
class SurrogateCandidateSpec:
    """One predeclared CSSF architecture candidate."""

    coordinate_family: str
    support_family: str
    support_order: int
    target_mode: str

    def __post_init__(self) -> None:
        coordinate = str(self.coordinate_family).strip()
        support = str(self.support_family).strip().lower()
        target = str(self.target_mode).strip().lower()
        if not coordinate:
            raise SurrogateSelectionError("coordinate_family must be non-empty.")
        if support not in SUPPORTED_SUPPORT_FAMILIES:
            raise SurrogateSelectionError(
                f"Unsupported support_family: {support!r}."
            )
        if target not in SUPPORTED_TARGET_MODES:
            raise SurrogateSelectionError(f"Unsupported target_mode: {target!r}.")
        if isinstance(self.support_order, bool) or not isinstance(self.support_order, int):
            raise TypeError("support_order must be an integer.")
        if self.support_order < 1:
            raise SurrogateSelectionError("support_order must be positive.")
        object.__setattr__(self, "coordinate_family", coordinate)
        object.__setattr__(self, "support_family", support)
        object.__setattr__(self, "target_mode", target)

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.coordinate_family}__{self.support_family}_"
            f"{self.support_order}__{self.target_mode}"
        )

    def fingerprint(self) -> str:
        payload = {
            "coordinate_family": self.coordinate_family,
            "support_family": self.support_family,
            "support_order": self.support_order,
            "target_mode": self.target_mode,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def build_support(self, n_dimensions: int) -> FrequencySupport:
        if self.support_family == "signed_axes":
            return signed_axis_support(
                n_dimensions,
                max_harmonic=self.support_order,
                include_zero=True,
            )
        if self.support_family == "pairwise":
            if self.support_order != 1:
                raise SurrogateSelectionError(
                    "pairwise support currently has first-order interactions only."
                )
            return pairwise_support(
                n_dimensions,
                include_axes=True,
                include_sums=True,
                include_differences=True,
                include_zero=True,
            )
        return total_l1_support(
            n_dimensions,
            max_l1_order=self.support_order,
            include_zero=True,
            max_terms=50_000,
        )



def frequency_support_fingerprint(support: FrequencySupport) -> str:
    """Return a deterministic SHA-256 identity for one frequency support."""

    if not isinstance(support, FrequencySupport):
        raise TypeError("support must be FrequencySupport.")
    digest = hashlib.sha256()
    digest.update(b"CSSF-FrequencySupportSelection-v1\0")
    digest.update(support.kind.value.encode("ascii"))
    digest.update(b"1" if support.include_zero else b"0")
    digest.update(b"1" if support.require_conjugate_symmetry else b"0")
    digest.update(support.frequencies.tobytes(order="C"))
    return digest.hexdigest()

def _rank_correlation(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    if actual.size < 2:
        return 1.0
    actual_rank = rankdata(actual, method="average")
    predicted_rank = rankdata(predicted, method="average")
    actual_centered = actual_rank - np.mean(actual_rank)
    predicted_centered = predicted_rank - np.mean(predicted_rank)
    denominator = float(
        np.linalg.norm(actual_centered) * np.linalg.norm(predicted_centered)
    )
    if denominator == 0.0:
        return 1.0 if np.allclose(actual, predicted) else 0.0
    return float(np.dot(actual_centered, predicted_centered) / denominator)


def _top_k_indices(values: NDArray[np.float64], *, k: int, lower: bool) -> set[int]:
    order = np.argsort(values if lower else -values, kind="stable")
    return set(int(index) for index in order[:k])


def lexicographic_control_order(
    values: ArrayLike,
    target_names: Sequence[str],
) -> NDArray[np.int64]:
    """Order controls by the declared feasibility-first production policy."""

    matrix = _finite_matrix(values, name="control targets")
    names = tuple(str(name) for name in target_names)
    if matrix.shape[1] != len(names):
        raise SurrogateSelectionError("target_names do not match target columns.")
    missing = [name for name in DEFAULT_SELECTION_TARGETS if name not in names]
    if missing:
        raise SurrogateSelectionError(f"Missing selection targets: {missing}.")
    column = {name: names.index(name) for name in names}
    return np.lexsort(
        (
            matrix[:, column["mean_energy"]],
            matrix[:, column["cvar_05"]],
            -matrix[:, column["elite_probability"]],
            -matrix[:, column["success_probability"]],
            -matrix[:, column["feasibility_probability"]],
        )
    ).astype(np.int64, copy=False)


def prediction_metrics(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    target_names: Sequence[str],
    lower_is_better: Sequence[bool],
    train_scale: ArrayLike,
    top_k: int,
) -> dict[str, Any]:
    """Compute scale-balanced accuracy and control-selection diagnostics."""

    actual_matrix = _finite_matrix(actual, name="actual")
    predicted_matrix = _finite_matrix(predicted, name="predicted")
    if actual_matrix.shape != predicted_matrix.shape:
        raise SurrogateSelectionError("actual and predicted shapes must match.")
    names = tuple(str(name) for name in target_names)
    directions = tuple(bool(value) for value in lower_is_better)
    if actual_matrix.shape[1] != len(names) or len(directions) != len(names):
        raise SurrogateSelectionError("Target metadata does not match matrices.")
    if not 1 <= top_k <= actual_matrix.shape[0]:
        raise SurrogateSelectionError("top_k is outside the sample range.")
    scale = _finite_vector(train_scale, name="train_scale")
    if scale.size != actual_matrix.shape[1] or np.any(scale <= 0.0):
        raise SurrogateSelectionError("train_scale is invalid.")

    error = predicted_matrix - actual_matrix
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error * error, axis=0))
    nrmse = rmse / scale
    centered = actual_matrix - np.mean(actual_matrix, axis=0)
    denominator = np.sum(centered * centered, axis=0)
    numerator = np.sum(error * error, axis=0)
    r2 = np.empty_like(denominator, dtype=REAL_DTYPE)
    nonconstant = denominator > 0.0
    r2[nonconstant] = 1.0 - numerator[nonconstant] / denominator[nonconstant]
    r2[~nonconstant] = np.where(
        numerator[~nonconstant] == 0.0,
        1.0,
        0.0,
    )
    spearman = np.asarray(
        [
            _rank_correlation(actual_matrix[:, i], predicted_matrix[:, i])
            for i in range(actual_matrix.shape[1])
        ],
        dtype=REAL_DTYPE,
    )
    top_k_recall = np.asarray(
        [
            len(
                _top_k_indices(actual_matrix[:, i], k=top_k, lower=directions[i])
                & _top_k_indices(predicted_matrix[:, i], k=top_k, lower=directions[i])
            )
            / top_k
            for i in range(actual_matrix.shape[1])
        ],
        dtype=REAL_DTYPE,
    )

    actual_order = lexicographic_control_order(actual_matrix, names)
    predicted_order = lexicographic_control_order(predicted_matrix, names)
    selected = int(predicted_order[0])
    actual_rank_positions = np.empty(actual_matrix.shape[0], dtype=np.int64)
    actual_rank_positions[actual_order] = np.arange(actual_matrix.shape[0])
    selected_rank = int(actual_rank_positions[selected])
    normalized_rank_regret = selected_rank / max(1, actual_matrix.shape[0] - 1)

    per_target = {
        name: {
            "mae": float(mae[index]),
            "rmse": float(rmse[index]),
            "nrmse_train_scale": float(nrmse[index]),
            "r2": float(r2[index]),
            "spearman": float(spearman[index]),
            "top_k_recall": float(top_k_recall[index]),
        }
        for index, name in enumerate(names)
    }
    probability_indices = [
        index for index, name in enumerate(names) if name.endswith("_probability")
    ]
    return {
        "macro_nrmse_train_scale": float(np.mean(nrmse)),
        "macro_r2": float(np.mean(r2)),
        "macro_spearman": float(np.mean(spearman)),
        "macro_top_k_recall": float(np.mean(top_k_recall)),
        "max_probability_mae": (
            0.0
            if not probability_indices
            else float(np.max(mae[probability_indices]))
        ),
        "selected_control_index": selected,
        "selected_control_actual_rank": selected_rank,
        "selection_rank_regret": float(normalized_rank_regret),
        "per_target": per_target,
    }


def candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Deterministic validation-only ordering of CSSF candidates."""

    return (
        0 if bool(row.get("gate_passed", False)) else 1,
        float(row["selection_rank_regret"]),
        float(row["macro_nrmse_train_scale"]),
        -float(row["macro_spearman"]),
        -float(row["macro_top_k_recall"]),
        int(row["feature_count"]),
        str(row["candidate_id"]),
    )


def select_best_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select the locked winner without inspecting calibration or OOD data."""

    if not rows:
        raise SurrogateSelectionError("At least one candidate row is required.")
    normalized = [dict(row) for row in rows]
    required = {
        "candidate_id",
        "gate_passed",
        "selection_rank_regret",
        "macro_nrmse_train_scale",
        "macro_spearman",
        "macro_top_k_recall",
        "feature_count",
    }
    for row in normalized:
        missing = sorted(required - set(row))
        if missing:
            raise SurrogateSelectionError(
                f"Candidate row is missing fields: {missing}."
            )
    winner = min(normalized, key=candidate_sort_key)
    if not bool(winner["gate_passed"]):
        raise SurrogateSelectionError(
            "No CSSF candidate passed the validation architecture gate."
        )
    return winner


__all__ = [
    "CoordinateTransform",
    "DEFAULT_SELECTION_TARGETS",
    "SUPPORTED_SUPPORT_FAMILIES",
    "SUPPORTED_TARGET_MODES",
    "SurrogateCandidateSpec",
    "SurrogateSelectionError",
    "TargetTransform",
    "candidate_sort_key",
    "fit_coordinate_transform",
    "frequency_support_fingerprint",
    "fit_target_transform",
    "lexicographic_control_order",
    "prediction_metrics",
    "select_best_candidate",
]
