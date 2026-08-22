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

"""Quality-only classical baseline contracts for CSSF BESS placement.

HiGHS is used as a deterministic classical quality reference for the same
BESS placement model and for independent comparison of the final placement
quality.  Runtime may be recorded for reproducibility diagnostics, but it is
never a competition criterion, an optimization target, or evidence of
scientific superiority.

The only surrogate studied by this project is the quantum-annealing response.
QAOA and MA-QAOA are mathematical teacher/decomposition models used to obtain
and validate coordinates for digitized quantum annealing; they are not
independent surrogate products and are not classical baselines.

This initializer contains lightweight, solver-independent validation only. It
performs no eager import of ``highspy``, NumPy, Ocean, Qiskit, CUDA, an OPF
solver, or project datasets.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Final


BASELINES_PACKAGE_NAME: Final[str] = "cssf.baselines"
BASELINES_PACKAGE_VERSION: Final[str] = "0.1.0"

HIGHS_SOLVER_NAME: Final[str] = "highs"
HIGHS_ROLE: Final[str] = "classical_solution_quality_reference"
SURROGATED_SYSTEM: Final[str] = "quantum_annealing_response"

PRIMARY_COMPARISON_AXIS: Final[str] = "solution_quality"
SECONDARY_COMPARISON_AXIS: Final[str] = "surrogate_approximation_quality"
RUNTIME_ROLE: Final[str] = "diagnostic_only"
WALL_CLOCK_COMPETITION_ALLOWED: Final[bool] = False
TIME_TO_SOLUTION_SUPERIORITY_CLAIMS_ALLOWED: Final[bool] = False

QUALITY_METRIC_AC_POST_VERIFIED_OBJECTIVE: Final[str] = (
    "ac_post_verified_objective"
)
QUALITY_METRIC_QUBO_OBJECTIVE: Final[str] = "qubo_objective"
QUALITY_METRIC_FEASIBILITY: Final[str] = "placement_feasibility"
QUALITY_METRIC_SURROGATE_MAE: Final[str] = "qa_surrogate_mae"
QUALITY_METRIC_SURROGATE_RMSE: Final[str] = "qa_surrogate_rmse"
QUALITY_METRIC_SURROGATE_R2: Final[str] = "qa_surrogate_r2"
QUALITY_METRIC_SURROGATE_SPEARMAN: Final[str] = "qa_surrogate_spearman"
QUALITY_METRIC_SURROGATE_TOP_K_RECALL: Final[str] = (
    "qa_surrogate_top_k_recall"
)

SUPPORTED_QUALITY_METRICS: Final[tuple[str, ...]] = (
    QUALITY_METRIC_AC_POST_VERIFIED_OBJECTIVE,
    QUALITY_METRIC_QUBO_OBJECTIVE,
    QUALITY_METRIC_FEASIBILITY,
    QUALITY_METRIC_SURROGATE_MAE,
    QUALITY_METRIC_SURROGATE_RMSE,
    QUALITY_METRIC_SURROGATE_R2,
    QUALITY_METRIC_SURROGATE_SPEARMAN,
    QUALITY_METRIC_SURROGATE_TOP_K_RECALL,
)

REQUIRED_PRIMARY_QUALITY_METRICS: Final[tuple[str, ...]] = (
    QUALITY_METRIC_AC_POST_VERIFIED_OBJECTIVE,
    QUALITY_METRIC_FEASIBILITY,
)


class BaselineContractError(ValueError):
    """Raised when a quality-baseline contract is violated."""


def _string_value(value: object, *, name: str) -> str:
    """Normalize a string or a string-valued enum."""

    candidate = value.value if isinstance(value, Enum) else value
    if not isinstance(candidate, str):
        raise TypeError(f"{name} must be a string or string-valued enum.")
    normalized = candidate.strip()
    if not normalized:
        raise BaselineContractError(f"{name} must not be empty.")
    return normalized


def validate_highs_solver_name(solver_name: object) -> str:
    """Require the single supported classical solver name."""

    normalized = _string_value(solver_name, name="solver_name").lower()
    if normalized != HIGHS_SOLVER_NAME:
        raise BaselineContractError(
            f"solver_name must be {HIGHS_SOLVER_NAME!r}; "
            f"received {normalized!r}."
        )
    return normalized


def validate_quality_metrics(metrics: Iterable[object]) -> tuple[str, ...]:
    """Validate an ordered, unique collection of quality metrics."""

    try:
        values = tuple(metrics)
    except TypeError as exc:
        raise TypeError("metrics must be an iterable.") from exc

    if not values:
        raise BaselineContractError("metrics must not be empty.")

    normalized: list[str] = []
    seen: set[str] = set()
    for position, value in enumerate(values):
        metric = _string_value(value, name=f"metrics[{position}]").lower()
        if metric not in SUPPORTED_QUALITY_METRICS:
            raise BaselineContractError(
                "Unsupported quality metric "
                f"{metric!r}; expected one of {SUPPORTED_QUALITY_METRICS}."
            )
        if metric in seen:
            raise BaselineContractError(
                f"metrics contains duplicate entry {metric!r}."
            )
        seen.add(metric)
        normalized.append(metric)

    return tuple(normalized)


def validate_quality_only_comparison(
    *,
    compare_solution_quality: bool,
    compare_surrogate_approximation: bool,
    compare_wall_clock: bool = False,
) -> tuple[bool, bool, bool]:
    """Validate that a benchmark is quality-only and never time-based."""

    for name, value in (
        ("compare_solution_quality", compare_solution_quality),
        ("compare_surrogate_approximation", compare_surrogate_approximation),
        ("compare_wall_clock", compare_wall_clock),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")

    if not compare_solution_quality:
        raise BaselineContractError(
            "compare_solution_quality must be enabled."
        )
    if not compare_surrogate_approximation:
        raise BaselineContractError(
            "compare_surrogate_approximation must be enabled."
        )
    if compare_wall_clock:
        raise BaselineContractError(
            "Wall-clock competition is forbidden; runtime is diagnostic only."
        )

    return True, True, False


__all__ = [
    "BASELINES_PACKAGE_NAME",
    "BASELINES_PACKAGE_VERSION",
    "HIGHS_SOLVER_NAME",
    "HIGHS_ROLE",
    "SURROGATED_SYSTEM",
    "PRIMARY_COMPARISON_AXIS",
    "SECONDARY_COMPARISON_AXIS",
    "RUNTIME_ROLE",
    "WALL_CLOCK_COMPETITION_ALLOWED",
    "TIME_TO_SOLUTION_SUPERIORITY_CLAIMS_ALLOWED",
    "QUALITY_METRIC_AC_POST_VERIFIED_OBJECTIVE",
    "QUALITY_METRIC_QUBO_OBJECTIVE",
    "QUALITY_METRIC_FEASIBILITY",
    "QUALITY_METRIC_SURROGATE_MAE",
    "QUALITY_METRIC_SURROGATE_RMSE",
    "QUALITY_METRIC_SURROGATE_R2",
    "QUALITY_METRIC_SURROGATE_SPEARMAN",
    "QUALITY_METRIC_SURROGATE_TOP_K_RECALL",
    "SUPPORTED_QUALITY_METRICS",
    "REQUIRED_PRIMARY_QUALITY_METRICS",
    "BaselineContractError",
    "validate_highs_solver_name",
    "validate_quality_metrics",
    "validate_quality_only_comparison",
]
