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

"""Deterministic case300 BESS candidate reduction.

Candidate reduction is computed only from the locked training partition.  The
held-out test scenarios never influence bus ranking.  The selector excludes
the slack bus, preserves a complete auditable score table, and produces the
exact candidate count required by :class:`config.schema.QUBOConfig`.

This module performs NumPy algebra only.  It does not initialize Qiskit, Aer,
Ocean, a D-Wave sampler, CUDA, HiGHS, or an OPF solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import NDArray

from bess import validate_candidate_buses, validate_placement_cardinality
from bess.case300 import Case300ModeAData
from config.schema import QUBOConfig
from opf.bess_constraints import BESSFleetSpec, BESSUnitSpec


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
DEFAULT_TAIL_QUANTILE: Final[float] = 0.95
DEFAULT_MEAN_ABS_WEIGHT: Final[float] = 0.35
DEFAULT_RMS_WEIGHT: Final[float] = 0.25
DEFAULT_STD_WEIGHT: Final[float] = 0.20
DEFAULT_TAIL_WEIGHT: Final[float] = 0.20


class BESSCandidateError(ValueError):
    """Raised when case300 candidate reduction violates its contract."""


def _finite_nonnegative(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise BESSCandidateError(f"{name} must be finite and non-negative.")
    return normalized


def _readonly_vector(values: Any, *, dtype: np.dtype[Any]) -> NDArray[Any]:
    result = np.array(values, dtype=dtype, order="C", copy=True).reshape(-1)
    result.setflags(write=False)
    return result


def _normalized_component(values: NDArray[np.float64]) -> NDArray[np.float64]:
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return np.zeros_like(values, dtype=REAL_DTYPE)
    return np.asarray(values / maximum, dtype=REAL_DTYPE)


@dataclass(frozen=True, slots=True)
class CandidateSelectionConfig:
    """Weights and cardinality for deterministic candidate reduction."""

    candidate_count: int
    bess_units: int
    tail_quantile: float = DEFAULT_TAIL_QUANTILE
    mean_abs_weight: float = DEFAULT_MEAN_ABS_WEIGHT
    rms_weight: float = DEFAULT_RMS_WEIGHT
    std_weight: float = DEFAULT_STD_WEIGHT
    tail_weight: float = DEFAULT_TAIL_WEIGHT
    exclude_slack: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.candidate_count, bool) or not isinstance(
            self.candidate_count, int
        ):
            raise TypeError("candidate_count must be an integer.")
        if isinstance(self.bess_units, bool) or not isinstance(self.bess_units, int):
            raise TypeError("bess_units must be an integer.")
        validate_placement_cardinality(
            self.bess_units,
            candidate_count=self.candidate_count,
        )
        quantile = float(self.tail_quantile)
        if not math.isfinite(quantile) or not 0.5 < quantile < 1.0:
            raise BESSCandidateError("tail_quantile must lie strictly in (0.5, 1).")
        weights = (
            _finite_nonnegative(self.mean_abs_weight, name="mean_abs_weight"),
            _finite_nonnegative(self.rms_weight, name="rms_weight"),
            _finite_nonnegative(self.std_weight, name="std_weight"),
            _finite_nonnegative(self.tail_weight, name="tail_weight"),
        )
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise BESSCandidateError("candidate score weights must sum to 1.0.")
        if not isinstance(self.exclude_slack, bool):
            raise TypeError("exclude_slack must be boolean.")
        object.__setattr__(self, "tail_quantile", quantile)
        object.__setattr__(self, "mean_abs_weight", weights[0])
        object.__setattr__(self, "rms_weight", weights[1])
        object.__setattr__(self, "std_weight", weights[2])
        object.__setattr__(self, "tail_weight", weights[3])

    @classmethod
    def from_qubo_config(cls, config: QUBOConfig) -> "CandidateSelectionConfig":
        if not isinstance(config, QUBOConfig):
            raise TypeError("config must be QUBOConfig.")
        return cls(
            candidate_count=int(config.candidate_count),
            bess_units=int(config.bess_units),
        )

    def fingerprint(self) -> str:
        payload = {
            "candidate_count": self.candidate_count,
            "bess_units": self.bess_units,
            "tail_quantile": self.tail_quantile,
            "weights": [
                self.mean_abs_weight,
                self.rms_weight,
                self.std_weight,
                self.tail_weight,
            ],
            "exclude_slack": self.exclude_slack,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateSelectionResult:
    """Immutable ranked case300 bus-candidate result."""

    candidate_buses: tuple[int, ...]
    ranked_buses: NDArray[np.int64]
    scores: NDArray[np.float64]
    mean_absolute_lsf: NDArray[np.float64]
    rms_lsf: NDArray[np.float64]
    std_lsf: NDArray[np.float64]
    tail_absolute_lsf: NDArray[np.float64]
    source_fingerprint: str
    config_fingerprint: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        size = self.ranked_buses.size
        arrays = (
            self.scores,
            self.mean_absolute_lsf,
            self.rms_lsf,
            self.std_lsf,
            self.tail_absolute_lsf,
        )
        if size == 0 or any(array.shape != (size,) for array in arrays):
            raise BESSCandidateError("ranked arrays must be non-empty and aligned.")
        if len(set(int(bus) for bus in self.ranked_buses)) != size:
            raise BESSCandidateError("ranked_buses must be unique.")
        if tuple(int(bus) for bus in self.ranked_buses[: len(self.candidate_buses)]) != (
            self.candidate_buses
        ):
            raise BESSCandidateError("candidate_buses must be the ranked prefix.")
        if any(array.flags.writeable for array in (self.ranked_buses, *arrays)):
            raise BESSCandidateError("candidate result arrays must be immutable.")

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-CandidateSelectionResult-v1\0")
        digest.update(self.source_fingerprint.encode("ascii"))
        digest.update(self.config_fingerprint.encode("ascii"))
        digest.update(self.ranked_buses.tobytes(order="C"))
        digest.update(self.scores.tobytes(order="C"))
        return digest.hexdigest()


def select_case300_candidates(
    data: Case300ModeAData,
    config: CandidateSelectionConfig,
) -> CandidateSelectionResult:
    """Rank buses from training LSF statistics and return a strict prefix."""

    if not isinstance(data, Case300ModeAData):
        raise TypeError("data must be Case300ModeAData.")
    if not isinstance(config, CandidateSelectionConfig):
        raise TypeError("config must be CandidateSelectionConfig.")

    eligible = np.ones(data.n, dtype=bool)
    if config.exclude_slack:
        eligible[np.asarray(data.slack_buses, dtype=np.int64)] = False
    eligible_buses = np.flatnonzero(eligible).astype(np.int64, copy=False)
    if config.candidate_count > eligible_buses.size:
        raise BESSCandidateError(
            "candidate_count exceeds the number of eligible case300 buses."
        )

    train_targets = np.asarray(data.targets[data.train_slice], dtype=REAL_DTYPE)
    absolute = np.abs(train_targets)
    mean_absolute = np.mean(absolute, axis=0, dtype=REAL_DTYPE)
    rms = np.sqrt(np.mean(np.square(train_targets), axis=0, dtype=REAL_DTYPE))
    std = np.std(train_targets, axis=0, dtype=REAL_DTYPE)
    tail = np.quantile(absolute, config.tail_quantile, axis=0)

    score = (
        config.mean_abs_weight * _normalized_component(mean_absolute)
        + config.rms_weight * _normalized_component(rms)
        + config.std_weight * _normalized_component(std)
        + config.tail_weight * _normalized_component(tail)
    )
    eligible_scores = score[eligible_buses]
    order = np.lexsort((eligible_buses, -eligible_scores))
    ranked_buses = eligible_buses[order]
    ranked_scores = eligible_scores[order]
    candidates = tuple(int(bus) for bus in ranked_buses[: config.candidate_count])
    validate_candidate_buses(candidates, n_buses=data.n)

    metadata = MappingProxyType(
        {
            "case": data.case,
            "training_scenarios": data.n_train,
            "held_out_scenarios_used": 0,
            "excluded_slack_buses": tuple(data.slack_buses) if config.exclude_slack else (),
            "score_order": "descending_score_then_ascending_bus",
        }
    )
    return CandidateSelectionResult(
        candidate_buses=candidates,
        ranked_buses=_readonly_vector(ranked_buses, dtype=np.dtype(np.int64)),
        scores=_readonly_vector(ranked_scores, dtype=REAL_DTYPE),
        mean_absolute_lsf=_readonly_vector(mean_absolute[ranked_buses], dtype=REAL_DTYPE),
        rms_lsf=_readonly_vector(rms[ranked_buses], dtype=REAL_DTYPE),
        std_lsf=_readonly_vector(std[ranked_buses], dtype=REAL_DTYPE),
        tail_absolute_lsf=_readonly_vector(tail[ranked_buses], dtype=REAL_DTYPE),
        source_fingerprint=data.fingerprint(),
        config_fingerprint=config.fingerprint(),
        metadata=metadata,
    )


def build_case300_fleet(
    selection: CandidateSelectionResult,
    *,
    bess_units: int,
    unit: BESSUnitSpec,
    metadata: Mapping[str, Any] | None = None,
) -> BESSFleetSpec:
    """Build the exact-cardinality fleet consumed by QUBO construction."""

    if not isinstance(selection, CandidateSelectionResult):
        raise TypeError("selection must be CandidateSelectionResult.")
    validate_placement_cardinality(
        bess_units,
        candidate_count=len(selection.candidate_buses),
    )
    merged = {} if metadata is None else dict(metadata)
    merged.update(
        {
            "candidate_selection_fingerprint": selection.fingerprint(),
            "candidate_source_fingerprint": selection.source_fingerprint,
        }
    )
    return BESSFleetSpec(
        selection.candidate_buses,
        units_to_place=bess_units,
        unit=unit,
        metadata=merged,
    )


__all__ = [
    "DEFAULT_TAIL_QUANTILE",
    "BESSCandidateError",
    "CandidateSelectionConfig",
    "CandidateSelectionResult",
    "select_case300_candidates",
    "build_case300_fleet",
]
