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

"""Strict Ocean/Pegasus backend contracts for CSSF.

The package covers two solver paths that share an Ocean-compatible sampling
boundary:

* a local GPU simulated-annealing backend on a Pegasus graph;
* an explicitly selected real Pegasus QPU.

Real-QPU access is restricted to ``Advantage_system4.*`` and
``Advantage_system6.*`` solver identifiers. Non-Pegasus topology, Zephyr,
implicit solver selection, classical fallback, and silent backend substitution
are prohibited.

This initializer exposes lightweight constants and validators only. It does
not import Ocean, dimod, minorminer, NetworkX, GPU libraries, create samplers,
read credentials, access the filesystem, or perform network operations.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Mapping


DWAVE_BACKEND_PACKAGE_NAME: Final[str] = "cssf.dwave_backend"
DWAVE_BACKEND_PACKAGE_VERSION: Final[str] = "0.1.0"

PEGASUS_TOPOLOGY_TYPE: Final[str] = "pegasus"
REJECTED_TOPOLOGY_TYPES: Final[tuple[str, ...]] = ("zephyr",)
ALLOWED_PEGASUS_SOLVER_PREFIXES: Final[tuple[str, ...]] = (
    "Advantage_system4.",
    "Advantage_system6.",
)

LOCAL_GPU_BACKEND_KIND: Final[str] = "local_sqa_gpu"
PEGASUS_QPU_BACKEND_KIND: Final[str] = "pegasus_qpu"
SUPPORTED_BACKEND_KINDS: Final[tuple[str, ...]] = (
    LOCAL_GPU_BACKEND_KIND,
    PEGASUS_QPU_BACKEND_KIND,
)

OCEAN_SAMPLE_VARTYPE: Final[str] = "BINARY"
OCEAN_SAMPLESET_CLASS: Final[str] = "dimod.SampleSet"
REQUIRED_SAMPLER_METHODS: Final[tuple[str, ...]] = (
    "sample",
    "sample_qubo",
)
REQUIRED_SAMPLER_MAPPINGS: Final[tuple[str, ...]] = (
    "parameters",
    "properties",
)

GPU_REQUIRED_FOR_LOCAL_EMULATOR: Final[bool] = True
CLASSICAL_FALLBACK_ALLOWED: Final[bool] = False
SOLVER_FALLBACK_ALLOWED: Final[bool] = False
IMPLICIT_QPU_SELECTION_ALLOWED: Final[bool] = False

MINIMUM_PEGASUS_M: Final[int] = 2
MAXIMUM_PEGASUS_M: Final[int] = 64
MINIMUM_NUM_READS: Final[int] = 1


class DWaveBackendContractError(ValueError):
    """Raised when a stable Ocean/Pegasus backend contract is violated."""


def _string_value(value: object, *, name: str) -> str:
    """Normalize a string or string-valued enum without importing config."""

    candidate = value.value if isinstance(value, Enum) else value
    if not isinstance(candidate, str):
        raise TypeError(f"{name} must be a string or string-valued enum.")

    normalized = candidate.strip()
    if not normalized:
        raise DWaveBackendContractError(f"{name} must not be empty.")
    return normalized


def validate_backend_kind(backend: object) -> str:
    """Validate a backend kind supported by this package."""

    normalized = _string_value(backend, name="backend")
    if normalized not in SUPPORTED_BACKEND_KINDS:
        raise DWaveBackendContractError(
            f"backend must be one of {SUPPORTED_BACKEND_KINDS}; "
            f"received {normalized!r}."
        )
    return normalized


def validate_topology_type(topology_type: object) -> str:
    """Require the exact Pegasus topology label and reject Zephyr."""

    normalized = _string_value(
        topology_type,
        name="topology_type",
    ).lower()

    if normalized in REJECTED_TOPOLOGY_TYPES:
        raise DWaveBackendContractError(
            "Zephyr topology is prohibited; CSSF targets Pegasus only."
        )
    if normalized != PEGASUS_TOPOLOGY_TYPE:
        raise DWaveBackendContractError(
            "topology_type must be exactly 'pegasus'; "
            f"received {normalized!r}."
        )
    return normalized


def validate_solver_id(solver_id: object) -> str:
    """Validate an explicit allowed real-QPU solver identifier."""

    normalized = _string_value(solver_id, name="solver_id")
    lowered = normalized.lower()
    if "zephyr" in lowered:
        raise DWaveBackendContractError(
            "Zephyr solver identifiers are prohibited."
        )

    prefix = next(
        (
            item
            for item in ALLOWED_PEGASUS_SOLVER_PREFIXES
            if normalized.startswith(item)
        ),
        None,
    )
    if prefix is None:
        raise DWaveBackendContractError(
            "solver_id must start with Advantage_system4. or "
            "Advantage_system6."
        )
    if len(normalized) == len(prefix):
        raise DWaveBackendContractError(
            "solver_id must identify a concrete solver after the allowed "
            f"prefix {prefix!r}."
        )
    return normalized


def validate_pegasus_m(pegasus_m: int) -> int:
    """Validate the supported local Pegasus graph size parameter."""

    if isinstance(pegasus_m, bool) or not isinstance(pegasus_m, int):
        raise TypeError("pegasus_m must be an integer.")
    if not MINIMUM_PEGASUS_M <= pegasus_m <= MAXIMUM_PEGASUS_M:
        raise DWaveBackendContractError(
            "pegasus_m must be in the inclusive range "
            f"[{MINIMUM_PEGASUS_M}, {MAXIMUM_PEGASUS_M}]."
        )
    return pegasus_m


def validate_num_reads(num_reads: int) -> int:
    """Validate the number of requested Ocean samples."""

    if isinstance(num_reads, bool) or not isinstance(num_reads, int):
        raise TypeError("num_reads must be an integer.")
    if num_reads < MINIMUM_NUM_READS:
        raise DWaveBackendContractError(
            f"num_reads must be at least {MINIMUM_NUM_READS}."
        )
    return num_reads


def validate_sampler_interface(sampler: object) -> object:
    """Validate an Ocean-compatible sampler without executing it.

    The check is intentionally structural. It does not import Ocean or dimod,
    submit a problem, read credentials, or access a remote solver.
    """

    missing_methods = tuple(
        name
        for name in REQUIRED_SAMPLER_METHODS
        if not callable(getattr(sampler, name, None))
    )
    if missing_methods:
        raise DWaveBackendContractError(
            "Sampler is missing callable methods: "
            f"{missing_methods}."
        )

    invalid_mappings = tuple(
        name
        for name in REQUIRED_SAMPLER_MAPPINGS
        if not isinstance(getattr(sampler, name, None), Mapping)
    )
    if invalid_mappings:
        raise DWaveBackendContractError(
            "Sampler attributes must be mappings: "
            f"{invalid_mappings}."
        )

    return sampler


__all__ = [
    "DWAVE_BACKEND_PACKAGE_NAME",
    "DWAVE_BACKEND_PACKAGE_VERSION",
    "PEGASUS_TOPOLOGY_TYPE",
    "REJECTED_TOPOLOGY_TYPES",
    "ALLOWED_PEGASUS_SOLVER_PREFIXES",
    "LOCAL_GPU_BACKEND_KIND",
    "PEGASUS_QPU_BACKEND_KIND",
    "SUPPORTED_BACKEND_KINDS",
    "OCEAN_SAMPLE_VARTYPE",
    "OCEAN_SAMPLESET_CLASS",
    "REQUIRED_SAMPLER_METHODS",
    "REQUIRED_SAMPLER_MAPPINGS",
    "GPU_REQUIRED_FOR_LOCAL_EMULATOR",
    "CLASSICAL_FALLBACK_ALLOWED",
    "SOLVER_FALLBACK_ALLOWED",
    "IMPLICIT_QPU_SELECTION_ALLOWED",
    "MINIMUM_PEGASUS_M",
    "MAXIMUM_PEGASUS_M",
    "MINIMUM_NUM_READS",
    "DWaveBackendContractError",
    "validate_backend_kind",
    "validate_topology_type",
    "validate_solver_id",
    "validate_pegasus_m",
    "validate_num_reads",
    "validate_sampler_interface",
]
