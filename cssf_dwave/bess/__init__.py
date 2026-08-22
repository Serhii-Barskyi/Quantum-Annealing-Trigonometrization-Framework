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

"""Stable BESS validation contracts for the CSSF framework.

This package is the common validation boundary between placement candidates,
QUBO-derived samples, the local Pegasus GPU emulator, Ocean/Leap QPU results,
matched HiGHS baselines, and AC post-verification.  It does not solve an OPF,
build a QUBO, create a sampler, load the case300 dataset, or import optional
solver runtimes during package import.

The helpers defined here deliberately operate on plain Python values.  They
provide one canonical interpretation of bus identifiers, placement
cardinality, scenario partitions, and backend labels before later BESS modules
perform numerical validation and reporting.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from pathlib import PurePosixPath
from typing import Final


BESS_PACKAGE_NAME: Final[str] = "cssf.bess"
BESS_PACKAGE_VERSION: Final[str] = "0.1.0"

CASE300_CASE_NAME: Final[str] = "case300"
CASE300_BUS_COUNT: Final[int] = 300
CASE300_DATASET_FILENAME: Final[str] = (
    "case300_full_modeA_Barskyi_Serhii.json"
)
CASE300_DATASET_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data",
    CASE300_DATASET_FILENAME,
)

PLACEMENT_BACKEND_QAOA: Final[str] = "qaoa"
PLACEMENT_BACKEND_MAQAOA: Final[str] = "maqaoa"
PLACEMENT_BACKEND_DIGITIZED_QA: Final[str] = "digitized_qa"
PLACEMENT_BACKEND_LOCAL_PEGASUS_GPU: Final[str] = "local_sqa_gpu"
PLACEMENT_BACKEND_PEGASUS_QPU: Final[str] = "pegasus_qpu"
PLACEMENT_BACKEND_HIGHS: Final[str] = "highs"
SUPPORTED_PLACEMENT_BACKENDS: Final[tuple[str, ...]] = (
    PLACEMENT_BACKEND_QAOA,
    PLACEMENT_BACKEND_MAQAOA,
    PLACEMENT_BACKEND_DIGITIZED_QA,
    PLACEMENT_BACKEND_LOCAL_PEGASUS_GPU,
    PLACEMENT_BACKEND_PEGASUS_QPU,
    PLACEMENT_BACKEND_HIGHS,
)

REQUIRED_CASE300_KEYS: Final[tuple[str, ...]] = (
    "case",
    "n",
    "n_train",
    "n_test",
    "n_scenarios",
    "bus_types",
    "edges",
    "M",
    "M_complex",
    "rank_modeA",
    "delta_r",
    "theta_rad",
    "X_modeA_re",
    "X_modeA_im",
    "y_lsf",
)

REQUIRED_PLACEMENT_RESULT_FIELDS: Final[tuple[str, ...]] = (
    "backend",
    "selected_buses",
    "placement_cardinality",
    "objective_value",
    "feasible",
    "source_fingerprint",
)

MINIMUM_BESS_UNITS: Final[int] = 1
MINIMUM_CANDIDATE_BUSES: Final[int] = 2
BUS_INDEX_ORIGIN: Final[int] = 0


class BESSContractError(ValueError):
    """Raised when a stable BESS package contract is violated."""


def _string_value(value: object, *, name: str) -> str:
    """Normalize a string or string-valued enum."""

    candidate = value.value if isinstance(value, Enum) else value
    if not isinstance(candidate, str):
        raise TypeError(f"{name} must be a string or string-valued enum.")
    normalized = candidate.strip()
    if not normalized:
        raise BESSContractError(f"{name} must not be empty.")
    return normalized


def validate_backend_name(backend: object) -> str:
    """Validate a placement backend shared by comparison reports."""

    normalized = _string_value(backend, name="backend")
    if normalized not in SUPPORTED_PLACEMENT_BACKENDS:
        raise BESSContractError(
            "backend must be one of "
            f"{SUPPORTED_PLACEMENT_BACKENDS}; received {normalized!r}."
        )
    return normalized


def validate_bus_count(n_buses: int) -> int:
    """Validate the total number of buses in a placement case."""

    if isinstance(n_buses, bool) or not isinstance(n_buses, int):
        raise TypeError("n_buses must be an integer.")
    if n_buses < MINIMUM_CANDIDATE_BUSES:
        raise BESSContractError(
            "n_buses must be at least "
            f"{MINIMUM_CANDIDATE_BUSES}; received {n_buses}."
        )
    return n_buses


def validate_candidate_buses(
    candidate_buses: Iterable[int],
    *,
    n_buses: int,
) -> tuple[int, ...]:
    """Validate and preserve an ordered candidate-bus collection."""

    total_buses = validate_bus_count(n_buses)
    try:
        values = tuple(candidate_buses)
    except TypeError as exc:
        raise TypeError("candidate_buses must be an iterable of integers.") from exc

    if len(values) < MINIMUM_CANDIDATE_BUSES:
        raise BESSContractError(
            "candidate_buses must contain at least "
            f"{MINIMUM_CANDIDATE_BUSES} entries."
        )

    normalized: list[int] = []
    seen: set[int] = set()
    for position, bus in enumerate(values):
        if isinstance(bus, bool) or not isinstance(bus, int):
            raise TypeError(
                "candidate_buses entries must be integers; "
                f"position {position} contains {type(bus).__name__}."
            )
        if not BUS_INDEX_ORIGIN <= bus < total_buses:
            raise BESSContractError(
                "candidate bus index is outside the valid range "
                f"[{BUS_INDEX_ORIGIN}, {total_buses - 1}]: {bus}."
            )
        if bus in seen:
            raise BESSContractError(
                f"candidate_buses contains duplicate bus {bus}."
            )
        seen.add(bus)
        normalized.append(bus)

    return tuple(normalized)


def validate_placement_cardinality(
    bess_units: int,
    *,
    candidate_count: int,
) -> int:
    """Require a non-empty strict subset of the candidate buses."""

    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise TypeError("candidate_count must be an integer.")
    if candidate_count < MINIMUM_CANDIDATE_BUSES:
        raise BESSContractError(
            "candidate_count must be at least "
            f"{MINIMUM_CANDIDATE_BUSES}."
        )
    if isinstance(bess_units, bool) or not isinstance(bess_units, int):
        raise TypeError("bess_units must be an integer.")
    if not MINIMUM_BESS_UNITS <= bess_units < candidate_count:
        raise BESSContractError(
            "bess_units must satisfy "
            f"{MINIMUM_BESS_UNITS} <= bess_units < candidate_count; "
            f"received bess_units={bess_units}, "
            f"candidate_count={candidate_count}."
        )
    return bess_units


def validate_selected_buses(
    selected_buses: Iterable[int],
    *,
    candidate_buses: Iterable[int],
    bess_units: int,
    n_buses: int,
) -> tuple[int, ...]:
    """Validate one ordered BESS placement against its candidate set."""

    candidates = validate_candidate_buses(
        candidate_buses,
        n_buses=n_buses,
    )
    expected_count = validate_placement_cardinality(
        bess_units,
        candidate_count=len(candidates),
    )

    try:
        selected = tuple(selected_buses)
    except TypeError as exc:
        raise TypeError("selected_buses must be an iterable of integers.") from exc

    if len(selected) != expected_count:
        raise BESSContractError(
            "selected_buses cardinality mismatch: expected "
            f"{expected_count}, received {len(selected)}."
        )

    candidate_set = set(candidates)
    normalized: list[int] = []
    seen: set[int] = set()
    for position, bus in enumerate(selected):
        if isinstance(bus, bool) or not isinstance(bus, int):
            raise TypeError(
                "selected_buses entries must be integers; "
                f"position {position} contains {type(bus).__name__}."
            )
        if bus not in candidate_set:
            raise BESSContractError(
                f"selected bus {bus} is not present in candidate_buses."
            )
        if bus in seen:
            raise BESSContractError(
                f"selected_buses contains duplicate bus {bus}."
            )
        seen.add(bus)
        normalized.append(bus)

    return tuple(normalized)


def validate_scenario_partition(
    *,
    n_scenarios: int,
    n_train: int,
    n_test: int,
) -> tuple[int, int, int]:
    """Validate the fixed train/test partition stored by a dataset."""

    named_values = {
        "n_scenarios": n_scenarios,
        "n_train": n_train,
        "n_test": n_test,
    }
    for name, value in named_values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 1:
            raise BESSContractError(f"{name} must be positive.")

    if n_train + n_test != n_scenarios:
        raise BESSContractError(
            "n_train + n_test must equal n_scenarios; received "
            f"{n_train} + {n_test} != {n_scenarios}."
        )
    return n_scenarios, n_train, n_test


__all__ = [
    "BESS_PACKAGE_NAME",
    "BESS_PACKAGE_VERSION",
    "CASE300_CASE_NAME",
    "CASE300_BUS_COUNT",
    "CASE300_DATASET_FILENAME",
    "CASE300_DATASET_RELATIVE_PATH",
    "PLACEMENT_BACKEND_QAOA",
    "PLACEMENT_BACKEND_MAQAOA",
    "PLACEMENT_BACKEND_DIGITIZED_QA",
    "PLACEMENT_BACKEND_LOCAL_PEGASUS_GPU",
    "PLACEMENT_BACKEND_PEGASUS_QPU",
    "PLACEMENT_BACKEND_HIGHS",
    "SUPPORTED_PLACEMENT_BACKENDS",
    "REQUIRED_CASE300_KEYS",
    "REQUIRED_PLACEMENT_RESULT_FIELDS",
    "MINIMUM_BESS_UNITS",
    "MINIMUM_CANDIDATE_BUSES",
    "BUS_INDEX_ORIGIN",
    "BESSContractError",
    "validate_backend_name",
    "validate_bus_count",
    "validate_candidate_buses",
    "validate_placement_cardinality",
    "validate_selected_buses",
    "validate_scenario_partition",
]
