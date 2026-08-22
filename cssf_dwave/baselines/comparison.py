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

"""Quality-only comparison of CSSF-QA and the exact HiGHS reference.

The project surrogates only the quantum-annealing response.  QAOA and
MA-QAOA are mathematical modelling/decomposition tools and are not independent
surrogate products.  This module therefore compares two distinct quality axes:

* approximation quality of the QA-response surrogate on held-out or OOD data;
* AC-post-verified quality of the BESS placements returned by CSSF-QA and
  HiGHS on exactly the same scenarios and with exactly the same objective.

Runtime, time limits, time-to-solution, and wall-clock superiority are not
accepted by any public API in this module.  The report is descriptive; a claim
of statistical superiority requires a separate preregistered inferential gate.
No OPF, HiGHS, Ocean, QPU, CUDA, or dataset runtime is initialized here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baselines import (
    HIGHS_SOLVER_NAME,
    QUALITY_METRIC_AC_POST_VERIFIED_OBJECTIVE,
    QUALITY_METRIC_FEASIBILITY,
    QUALITY_METRIC_SURROGATE_MAE,
    QUALITY_METRIC_SURROGATE_R2,
    QUALITY_METRIC_SURROGATE_RMSE,
    QUALITY_METRIC_SURROGATE_SPEARMAN,
    QUALITY_METRIC_SURROGATE_TOP_K_RECALL,
    RUNTIME_ROLE,
    SURROGATED_SYSTEM,
)
from baselines.highs import HighsBESSResult
from dwave_backend.sampler import SamplerMode
from dwave_backend.solver import DWaveSolveResult
from opf.bess_constraints import BESSPlacement


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
BOOL_DTYPE: Final[np.dtype[np.bool_]] = np.dtype(np.bool_)

QA_BACKENDS: Final[tuple[str, ...]] = (
    SamplerMode.EMULATOR.value,
    SamplerMode.QPU.value,
)
SUPPORTED_PARTITIONS: Final[tuple[str, ...]] = (
    "validation",
    "test",
    "ood",
)
COMPARISON_EVIDENCE_STATUS: Final[str] = "descriptive_quality_only"
DEFAULT_OBJECTIVE_TOLERANCE: Final[float] = 1.0e-9


class QualityComparisonError(ValueError):
    """Raised when a quality-only comparison is invalid or mismatched."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise QualityComparisonError(f"{name} must be finite.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized <= 0.0:
        raise QualityComparisonError(f"{name} must be strictly positive.")
    return normalized


def _sha256_digest(value: object, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise QualityComparisonError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise QualityComparisonError(
            f"{name} must be a lowercase SHA-256 digest."
        ) from exc
    return normalized


def _json_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    source = {} if metadata is None else dict(metadata)
    forbidden = (
        "token",
        "password",
        "secret",
        "credential",
        "api_key",
        "apikey",
    )
    for key in source:
        normalized_key = str(key).strip().lower()
        if any(fragment in normalized_key for fragment in forbidden):
            raise QualityComparisonError(
                "metadata must not contain secrets or credential fields."
            )

    try:
        encoded = json.dumps(
            source,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise QualityComparisonError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _readonly_real_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int,
) -> NDArray[np.float64]:
    result = np.array(
        values,
        dtype=REAL_DTYPE,
        order="C",
        copy=True,
    )
    if result.ndim != ndim:
        raise QualityComparisonError(
            f"{name} must be {ndim}-dimensional."
        )
    if result.size == 0 or any(size == 0 for size in result.shape):
        raise QualityComparisonError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(result)):
        raise QualityComparisonError(f"{name} contains non-finite values.")
    result = np.ascontiguousarray(result, dtype=REAL_DTYPE)
    result.setflags(write=False)
    return result


def _readonly_bool_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_size: int,
) -> NDArray[np.bool_]:
    array = np.array(values, order="C", copy=True)
    if array.ndim != 1 or array.size != expected_size:
        raise QualityComparisonError(
            f"{name} must contain exactly {expected_size} values."
        )
    if array.dtype.kind not in {"b", "i", "u"}:
        raise QualityComparisonError(f"{name} must be boolean-valued.")
    if not np.all((array == 0) | (array == 1)):
        raise QualityComparisonError(f"{name} must be boolean-valued.")
    result = np.ascontiguousarray(array, dtype=BOOL_DTYPE)
    result.setflags(write=False)
    return result


def _unique_strings(
    values: Sequence[object],
    *,
    name: str,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for position, value in enumerate(values):
        normalized = str(value).strip()
        if not normalized:
            raise QualityComparisonError(
                f"{name}[{position}] must not be empty."
            )
        if normalized in seen:
            raise QualityComparisonError(
                f"{name} contains duplicate value {normalized!r}."
            )
        seen.add(normalized)
        result.append(normalized)
    if not result:
        raise QualityComparisonError(f"{name} must not be empty.")
    return tuple(result)


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return deterministic one-based average ranks with tie handling."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=REAL_DTYPE)
    start = 0
    while start < values.size:
        end = start + 1
        current = values[order[start]]
        while end < values.size and values[order[end]] == current:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> float:
    actual_rank = _average_ranks(actual)
    predicted_rank = _average_ranks(predicted)
    actual_centered = actual_rank - actual_rank.mean()
    predicted_centered = predicted_rank - predicted_rank.mean()
    denominator = float(
        np.linalg.norm(actual_centered)
        * np.linalg.norm(predicted_centered)
    )
    if denominator == 0.0:
        return 1.0 if np.array_equal(actual_rank, predicted_rank) else 0.0
    return float(actual_centered @ predicted_centered / denominator)


def _top_k_indices(
    values: NDArray[np.float64],
    *,
    top_k: int,
    lower_is_better: bool,
) -> NDArray[np.int64]:
    primary = values if lower_is_better else -values
    indices = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((indices, primary))
    return order[:top_k]


@dataclass(frozen=True, slots=True, init=False)
class QASurrogateQuality:
    """Held-out/OOD approximation quality of the QA-response surrogate."""

    sample_ids: tuple[str, ...]
    target_names: tuple[str, ...]
    actual: NDArray[np.float64]
    predicted: NDArray[np.float64]
    lower_is_better: tuple[bool, ...]
    top_k: int
    partition: str
    source_fingerprint: str
    metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        sample_ids: Sequence[object],
        target_names: Sequence[object],
        actual: ArrayLike,
        predicted: ArrayLike,
        lower_is_better: Sequence[bool],
        top_k: int,
        partition: str,
        source_fingerprint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        ids = _unique_strings(sample_ids, name="sample_ids")
        targets = _unique_strings(target_names, name="target_names")
        actual_array = _readonly_real_array(
            actual,
            name="actual",
            ndim=2,
        )
        predicted_array = _readonly_real_array(
            predicted,
            name="predicted",
            ndim=2,
        )
        if actual_array.shape != predicted_array.shape:
            raise QualityComparisonError(
                "actual and predicted shapes must match."
            )
        if actual_array.shape != (len(ids), len(targets)):
            raise QualityComparisonError(
                "actual/predicted shape must equal "
                "(len(sample_ids), len(target_names))."
            )

        directions = tuple(lower_is_better)
        if len(directions) != len(targets) or not all(
            isinstance(value, bool) for value in directions
        ):
            raise QualityComparisonError(
                "lower_is_better must contain one boolean per target."
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if not 1 <= top_k <= len(ids):
            raise QualityComparisonError(
                "top_k must lie in [1, number of samples]."
            )
        normalized_partition = str(partition).strip().lower()
        if normalized_partition not in SUPPORTED_PARTITIONS:
            raise QualityComparisonError(
                f"partition must be one of {SUPPORTED_PARTITIONS}."
            )

        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "target_names", targets)
        object.__setattr__(self, "actual", actual_array)
        object.__setattr__(self, "predicted", predicted_array)
        object.__setattr__(self, "lower_is_better", directions)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "partition", normalized_partition)
        object.__setattr__(
            self,
            "source_fingerprint",
            _sha256_digest(
                source_fingerprint,
                name="source_fingerprint",
            ),
        )
        object.__setattr__(self, "metadata", _json_metadata(metadata))

    @property
    def n_samples(self) -> int:
        return self.actual.shape[0]

    @property
    def n_targets(self) -> int:
        return self.actual.shape[1]

    def metric_arrays(self) -> Mapping[str, NDArray[np.float64]]:
        error = self.predicted - self.actual
        mae = np.mean(np.abs(error), axis=0)
        rmse = np.sqrt(np.mean(error * error, axis=0))
        actual_centered = self.actual - np.mean(self.actual, axis=0)
        denominator = np.sum(actual_centered * actual_centered, axis=0)
        numerator = np.sum(error * error, axis=0)
        r2 = np.empty_like(denominator, dtype=REAL_DTYPE)
        nonconstant = denominator > 0.0
        r2[nonconstant] = (
            1.0 - numerator[nonconstant] / denominator[nonconstant]
        )
        r2[~nonconstant] = np.where(
            numerator[~nonconstant] == 0.0,
            1.0,
            0.0,
        )
        spearman = np.asarray(
            [
                _spearman(self.actual[:, index], self.predicted[:, index])
                for index in range(self.n_targets)
            ],
            dtype=REAL_DTYPE,
        )
        top_k_recall = np.asarray(
            [
                len(
                    set(
                        _top_k_indices(
                            self.actual[:, index],
                            top_k=self.top_k,
                            lower_is_better=self.lower_is_better[index],
                        ).tolist()
                    )
                    & set(
                        _top_k_indices(
                            self.predicted[:, index],
                            top_k=self.top_k,
                            lower_is_better=self.lower_is_better[index],
                        ).tolist()
                    )
                )
                / self.top_k
                for index in range(self.n_targets)
            ],
            dtype=REAL_DTYPE,
        )

        result: dict[str, NDArray[np.float64]] = {}
        for name, values in (
            (QUALITY_METRIC_SURROGATE_MAE, mae),
            (QUALITY_METRIC_SURROGATE_RMSE, rmse),
            (QUALITY_METRIC_SURROGATE_R2, r2),
            (QUALITY_METRIC_SURROGATE_SPEARMAN, spearman),
            (QUALITY_METRIC_SURROGATE_TOP_K_RECALL, top_k_recall),
        ):
            array = np.ascontiguousarray(values, dtype=REAL_DTYPE)
            array.setflags(write=False)
            result[name] = array
        return MappingProxyType(result)

    def summary(self) -> dict[str, Any]:
        metrics = self.metric_arrays()
        per_target = {
            target: {
                metric: float(values[index])
                for metric, values in metrics.items()
            }
            for index, target in enumerate(self.target_names)
        }
        macro = {
            metric: float(np.mean(values))
            for metric, values in metrics.items()
        }
        return {
            "surrogated_system": SURROGATED_SYSTEM,
            "partition": self.partition,
            "n_samples": self.n_samples,
            "n_targets": self.n_targets,
            "top_k": self.top_k,
            "per_target": per_target,
            "macro": macro,
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QASurrogateQuality-v1\0")
        digest.update(self.source_fingerprint.encode("ascii"))
        digest.update(self.partition.encode("ascii"))
        digest.update(
            json.dumps(
                [self.sample_ids, self.target_names, self.lower_is_better],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(np.asarray([self.top_k], dtype=np.int64).tobytes())
        digest.update(self.actual.tobytes(order="C"))
        digest.update(self.predicted.tobytes(order="C"))
        digest.update(
            json.dumps(
                dict(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class ACPostVerifiedPlacementQuality:
    """Physical placement quality evaluated on a fixed scenario set."""

    backend: str
    placement: BESSPlacement
    scenario_ids: tuple[str, ...]
    objective_values: NDArray[np.float64]
    feasible_mask: NDArray[np.bool_]
    partition: str
    source_fingerprint: str
    verification_fingerprint: str
    metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        backend: str,
        placement: BESSPlacement,
        scenario_ids: Sequence[object],
        objective_values: ArrayLike,
        feasible_mask: ArrayLike,
        partition: str,
        source_fingerprint: str,
        verification_fingerprint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_backend = str(backend).strip().lower()
        if normalized_backend not in (HIGHS_SOLVER_NAME, *QA_BACKENDS):
            raise QualityComparisonError(
                "backend must be highs, local_sqa_gpu, or pegasus_qpu."
            )
        if not isinstance(placement, BESSPlacement):
            raise TypeError("placement must be BESSPlacement.")
        ids = _unique_strings(scenario_ids, name="scenario_ids")
        objectives = _readonly_real_array(
            objective_values,
            name="objective_values",
            ndim=1,
        )
        if objectives.size != len(ids):
            raise QualityComparisonError(
                "objective_values size must equal len(scenario_ids)."
            )
        feasible = _readonly_bool_vector(
            feasible_mask,
            name="feasible_mask",
            expected_size=len(ids),
        )
        normalized_partition = str(partition).strip().lower()
        if normalized_partition not in {"test", "ood"}:
            raise QualityComparisonError(
                "AC post-verification partition must be test or ood."
            )

        object.__setattr__(self, "backend", normalized_backend)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "scenario_ids", ids)
        object.__setattr__(self, "objective_values", objectives)
        object.__setattr__(self, "feasible_mask", feasible)
        object.__setattr__(self, "partition", normalized_partition)
        object.__setattr__(
            self,
            "source_fingerprint",
            _sha256_digest(
                source_fingerprint,
                name="source_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "verification_fingerprint",
            _sha256_digest(
                verification_fingerprint,
                name="verification_fingerprint",
            ),
        )
        object.__setattr__(self, "metadata", _json_metadata(metadata))

    @property
    def mean_objective(self) -> float:
        return float(np.mean(self.objective_values))

    @property
    def median_objective(self) -> float:
        return float(np.median(self.objective_values))

    @property
    def feasibility_rate(self) -> float:
        return float(np.mean(self.feasible_mask))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-ACPostVerifiedPlacementQuality-v1\0")
        digest.update(self.backend.encode("ascii"))
        digest.update(self.placement.fleet.fingerprint().encode("ascii"))
        digest.update(self.placement.selection.tobytes(order="C"))
        digest.update(
            json.dumps(
                self.scenario_ids,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(self.objective_values.tobytes(order="C"))
        digest.update(self.feasible_mask.tobytes(order="C"))
        digest.update(self.partition.encode("ascii"))
        digest.update(self.source_fingerprint.encode("ascii"))
        digest.update(self.verification_fingerprint.encode("ascii"))
        digest.update(
            json.dumps(
                dict(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class BESSQualityComparison:
    """Immutable descriptive comparison of HiGHS and CSSF-QA quality."""

    highs_quality: ACPostVerifiedPlacementQuality
    cssf_qa_quality: ACPostVerifiedPlacementQuality
    qa_surrogate_quality: QASurrogateQuality
    objective_delta_highs_minus_cssf: NDArray[np.float64]
    mean_objective_advantage: float
    median_objective_advantage: float
    cssf_scenario_win_rate: float
    feasibility_rate_advantage: float
    outcome: str
    evidence_status: str
    highs_result_fingerprint: str
    cssf_qa_result_fingerprint: str
    metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        highs_quality: ACPostVerifiedPlacementQuality,
        cssf_qa_quality: ACPostVerifiedPlacementQuality,
        qa_surrogate_quality: QASurrogateQuality,
        objective_tolerance: float,
        highs_result_fingerprint: str,
        cssf_qa_result_fingerprint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        tolerance = _positive_float(
            objective_tolerance,
            name="objective_tolerance",
        )
        delta = np.ascontiguousarray(
            highs_quality.objective_values
            - cssf_qa_quality.objective_values,
            dtype=REAL_DTYPE,
        )
        delta.setflags(write=False)
        mean_advantage = float(np.mean(delta))
        median_advantage = float(np.median(delta))
        win_rate = float(np.mean(delta > tolerance))
        feasibility_advantage = (
            cssf_qa_quality.feasibility_rate
            - highs_quality.feasibility_rate
        )

        if (
            feasibility_advantage >= -tolerance
            and mean_advantage > tolerance
        ):
            outcome = "cssf_qa_better_descriptively"
        elif (
            feasibility_advantage <= tolerance
            and mean_advantage < -tolerance
        ):
            outcome = "highs_better_descriptively"
        else:
            outcome = "mixed_or_indistinguishable_descriptively"

        object.__setattr__(self, "highs_quality", highs_quality)
        object.__setattr__(self, "cssf_qa_quality", cssf_qa_quality)
        object.__setattr__(self, "qa_surrogate_quality", qa_surrogate_quality)
        object.__setattr__(self, "objective_delta_highs_minus_cssf", delta)
        object.__setattr__(self, "mean_objective_advantage", mean_advantage)
        object.__setattr__(
            self,
            "median_objective_advantage",
            median_advantage,
        )
        object.__setattr__(self, "cssf_scenario_win_rate", win_rate)
        object.__setattr__(
            self,
            "feasibility_rate_advantage",
            float(feasibility_advantage),
        )
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self,
            "evidence_status",
            COMPARISON_EVIDENCE_STATUS,
        )
        object.__setattr__(
            self,
            "highs_result_fingerprint",
            _sha256_digest(
                highs_result_fingerprint,
                name="highs_result_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "cssf_qa_result_fingerprint",
            _sha256_digest(
                cssf_qa_result_fingerprint,
                name="cssf_qa_result_fingerprint",
            ),
        )
        object.__setattr__(self, "metadata", _json_metadata(metadata))

    def manifest(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint(),
            "surrogated_system": SURROGATED_SYSTEM,
            "runtime_role": RUNTIME_ROLE,
            "evidence_status": self.evidence_status,
            "outcome": self.outcome,
            "metrics": {
                QUALITY_METRIC_AC_POST_VERIFIED_OBJECTIVE: {
                    "highs_mean": self.highs_quality.mean_objective,
                    "cssf_qa_mean": self.cssf_qa_quality.mean_objective,
                    "highs_minus_cssf_mean": (
                        self.mean_objective_advantage
                    ),
                    "highs_minus_cssf_median": (
                        self.median_objective_advantage
                    ),
                    "cssf_scenario_win_rate": (
                        self.cssf_scenario_win_rate
                    ),
                },
                QUALITY_METRIC_FEASIBILITY: {
                    "highs_rate": self.highs_quality.feasibility_rate,
                    "cssf_qa_rate": self.cssf_qa_quality.feasibility_rate,
                    "cssf_minus_highs": (
                        self.feasibility_rate_advantage
                    ),
                },
                "qa_surrogate": self.qa_surrogate_quality.summary(),
            },
            "highs_quality_fingerprint": self.highs_quality.fingerprint(),
            "cssf_qa_quality_fingerprint": (
                self.cssf_qa_quality.fingerprint()
            ),
            "qa_surrogate_quality_fingerprint": (
                self.qa_surrogate_quality.fingerprint()
            ),
            "highs_result_fingerprint": self.highs_result_fingerprint,
            "cssf_qa_result_fingerprint": (
                self.cssf_qa_result_fingerprint
            ),
            "metadata": dict(self.metadata),
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-BESSQualityComparison-v1\0")
        digest.update(self.highs_quality.fingerprint().encode("ascii"))
        digest.update(self.cssf_qa_quality.fingerprint().encode("ascii"))
        digest.update(self.qa_surrogate_quality.fingerprint().encode("ascii"))
        digest.update(self.objective_delta_highs_minus_cssf.tobytes(order="C"))
        digest.update(
            np.asarray(
                [
                    self.mean_objective_advantage,
                    self.median_objective_advantage,
                    self.cssf_scenario_win_rate,
                    self.feasibility_rate_advantage,
                ],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        digest.update(self.outcome.encode("ascii"))
        digest.update(self.evidence_status.encode("ascii"))
        digest.update(self.highs_result_fingerprint.encode("ascii"))
        digest.update(self.cssf_qa_result_fingerprint.encode("ascii"))
        digest.update(
            json.dumps(
                dict(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


def compare_bess_solution_quality(
    *,
    highs_result: HighsBESSResult,
    cssf_qa_result: DWaveSolveResult,
    highs_quality: ACPostVerifiedPlacementQuality,
    cssf_qa_quality: ACPostVerifiedPlacementQuality,
    qa_surrogate_quality: QASurrogateQuality,
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE,
    metadata: Mapping[str, Any] | None = None,
) -> BESSQualityComparison:
    """Compare solution and QA-surrogate quality without any time metric."""

    if not isinstance(highs_result, HighsBESSResult):
        raise TypeError("highs_result must be HighsBESSResult.")
    if not isinstance(cssf_qa_result, DWaveSolveResult):
        raise TypeError("cssf_qa_result must be DWaveSolveResult.")
    if not isinstance(highs_quality, ACPostVerifiedPlacementQuality):
        raise TypeError(
            "highs_quality must be ACPostVerifiedPlacementQuality."
        )
    if not isinstance(cssf_qa_quality, ACPostVerifiedPlacementQuality):
        raise TypeError(
            "cssf_qa_quality must be ACPostVerifiedPlacementQuality."
        )
    if not isinstance(qa_surrogate_quality, QASurrogateQuality):
        raise TypeError(
            "qa_surrogate_quality must be QASurrogateQuality."
        )

    if highs_quality.backend != HIGHS_SOLVER_NAME:
        raise QualityComparisonError(
            "highs_quality backend must be highs."
        )
    if cssf_qa_quality.backend not in QA_BACKENDS:
        raise QualityComparisonError(
            "cssf_qa_quality backend must be a Pegasus QA backend."
        )
    if cssf_qa_result.backend != cssf_qa_quality.backend:
        raise QualityComparisonError(
            "CSSF-QA result backend differs from AC verification backend."
        )
    if cssf_qa_result.source_kind != "bess_placement_qubo":
        raise QualityComparisonError(
            "CSSF-QA result must originate from BESSPlacementQUBO."
        )
    if cssf_qa_result.placement is None:
        raise QualityComparisonError(
            "CSSF-QA result must contain a decoded BESS placement."
        )

    if highs_result.source_fingerprint != cssf_qa_result.source_fingerprint:
        raise QualityComparisonError(
            "HiGHS and CSSF-QA results must use the same BESS problem."
        )
    if highs_quality.source_fingerprint != highs_result.source_fingerprint:
        raise QualityComparisonError(
            "HiGHS AC verification source differs from its solve result."
        )
    if cssf_qa_quality.source_fingerprint != cssf_qa_result.source_fingerprint:
        raise QualityComparisonError(
            "CSSF-QA AC verification source differs from its solve result."
        )
    if qa_surrogate_quality.source_fingerprint != cssf_qa_result.source_fingerprint:
        raise QualityComparisonError(
            "QA-surrogate quality must refer to the same BESS problem."
        )

    if not np.array_equal(
        highs_quality.placement.selection,
        highs_result.placement.selection,
    ):
        raise QualityComparisonError(
            "HiGHS AC verification placement differs from HiGHS result."
        )
    if not np.array_equal(
        cssf_qa_quality.placement.selection,
        cssf_qa_result.placement.selection,
    ):
        raise QualityComparisonError(
            "CSSF-QA AC verification placement differs from solve result."
        )
    if (
        highs_quality.placement.fleet.fingerprint()
        != cssf_qa_quality.placement.fleet.fingerprint()
    ):
        raise QualityComparisonError(
            "HiGHS and CSSF-QA placements must use the same fleet."
        )
    if highs_quality.scenario_ids != cssf_qa_quality.scenario_ids:
        raise QualityComparisonError(
            "AC verification must use identical ordered scenario IDs."
        )
    if highs_quality.partition != cssf_qa_quality.partition:
        raise QualityComparisonError(
            "AC verification partitions must match."
        )
    if (
        highs_quality.verification_fingerprint
        != cssf_qa_quality.verification_fingerprint
    ):
        raise QualityComparisonError(
            "AC verification models/objectives must match exactly."
        )

    return BESSQualityComparison(
        highs_quality=highs_quality,
        cssf_qa_quality=cssf_qa_quality,
        qa_surrogate_quality=qa_surrogate_quality,
        objective_tolerance=objective_tolerance,
        highs_result_fingerprint=highs_result.fingerprint(),
        cssf_qa_result_fingerprint=cssf_qa_result.fingerprint(),
        metadata=metadata,
    )


__all__ = [
    "REAL_DTYPE",
    "BOOL_DTYPE",
    "QA_BACKENDS",
    "SUPPORTED_PARTITIONS",
    "COMPARISON_EVIDENCE_STATUS",
    "DEFAULT_OBJECTIVE_TOLERANCE",
    "QualityComparisonError",
    "QASurrogateQuality",
    "ACPostVerifiedPlacementQuality",
    "BESSQualityComparison",
    "compare_bess_solution_quality",
]
