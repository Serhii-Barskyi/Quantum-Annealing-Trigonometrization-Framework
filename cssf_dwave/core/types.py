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

"""Shared strongly typed records for CSSF-QA-D-Wave.

This module contains only stable domain types. It has no filesystem, GPU,
Ocean, QPU, OPF, or network side effects and does not import the frozen
``core/gcv.py`` or ``core/csnn_t.py`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Final, TypeAlias


Variable: TypeAlias = int
Coupler: TypeAlias = tuple[Variable, Variable]

UINT32_MAX: Final[int] = 2**32 - 1
ALLOWED_PEGASUS_SOLVER_PREFIXES: Final[tuple[str, ...]] = (
    "Advantage_system4.",
    "Advantage_system6.",
)


class SurrogateLevel(str, Enum):
    """Mathematical levels of the CSSF surrogate hierarchy."""

    OPF = "opf"
    QAOA = "qaoa"
    MA_QAOA = "ma_qaoa"
    DIGITIZED_QA = "digitized_qa"
    HARDWARE_RESIDUAL = "hardware_residual"


class ExecutionTarget(str, Enum):
    """Supported execution targets."""

    AER_GPU = "aer_gpu"
    SQA_GPU = "sqa_gpu"
    PEGASUS_QPU = "pegasus_qpu"
    HIGHS = "highs"


class ProblemDomain(str, Enum):
    """Variable domain of an optimization model."""

    BINARY = "BINARY"
    SPIN = "SPIN"


class DatasetPartition(str, Enum):
    """Standard experiment-data partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OOD = "ood"


class RunStatus(str, Enum):
    """Lifecycle state of an experiment run."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Reproducibility seeds shared across the complete pipeline."""

    global_seed: int = 42
    scenario_seed: int = 101
    qaoa_seed: int = 202
    maqaoa_seed: int = 303
    qa_seed: int = 404
    sampler_seed: int = 505

    def __post_init__(self) -> None:
        for field_name in (
            "global_seed",
            "scenario_seed",
            "qaoa_seed",
            "maqaoa_seed",
            "qa_seed",
            "sampler_seed",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if not 0 <= value <= UINT32_MAX:
                raise ValueError(
                    f"{field_name} must lie in [0, {UINT32_MAX}]."
                )

    def as_dict(self) -> dict[str, int]:
        """Return a serialization-ready seed mapping."""

        return {
            "global_seed": self.global_seed,
            "scenario_seed": self.scenario_seed,
            "qaoa_seed": self.qaoa_seed,
            "maqaoa_seed": self.maqaoa_seed,
            "qa_seed": self.qa_seed,
            "sampler_seed": self.sampler_seed,
        }


@dataclass(frozen=True, slots=True)
class ProblemShape:
    """Validated structural size of a BQM/QUBO/Ising problem."""

    n_variables: int
    n_linear_terms: int
    n_quadratic_terms: int

    def __post_init__(self) -> None:
        for field_name in (
            "n_variables",
            "n_linear_terms",
            "n_quadratic_terms",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")

        if self.n_variables == 0:
            raise ValueError("n_variables must be positive.")
        if self.n_linear_terms > self.n_variables:
            raise ValueError(
                "n_linear_terms cannot exceed n_variables."
            )

        maximum_couplers = (
            self.n_variables * (self.n_variables - 1) // 2
        )
        if self.n_quadratic_terms > maximum_couplers:
            raise ValueError(
                "n_quadratic_terms exceeds the number of unique "
                "undirected variable pairs."
            )

    @property
    def density(self) -> float:
        """Quadratic interaction density in the complete undirected graph."""

        maximum_couplers = (
            self.n_variables * (self.n_variables - 1) // 2
        )
        if maximum_couplers == 0:
            return 0.0
        return self.n_quadratic_terms / maximum_couplers


@dataclass(frozen=True, slots=True)
class SolverIdentity:
    """Validated identity of an execution backend.

    Real-QPU identities are restricted to Pegasus solvers whose IDs begin with
    ``Advantage_system4.`` or ``Advantage_system6.``.
    """

    target: ExecutionTarget
    solver_id: str | None = None
    topology: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, ExecutionTarget):
            raise TypeError("target must be an ExecutionTarget.")

        normalized_solver_id = (
            None if self.solver_id is None else self.solver_id.strip()
        )
        normalized_topology = (
            None if self.topology is None else self.topology.strip().lower()
        )

        object.__setattr__(self, "solver_id", normalized_solver_id)
        object.__setattr__(self, "topology", normalized_topology)

        if self.target is ExecutionTarget.PEGASUS_QPU:
            if not normalized_solver_id:
                raise ValueError(
                    "Pegasus QPU requires an explicit solver_id."
                )
            if not normalized_solver_id.startswith(
                ALLOWED_PEGASUS_SOLVER_PREFIXES
            ):
                raise ValueError(
                    "Pegasus QPU solver_id must start with "
                    "Advantage_system4. or Advantage_system6."
                )
            if normalized_topology != "pegasus":
                raise ValueError(
                    "Real QPU topology must be explicitly set to pegasus."
                )
        else:
            if normalized_solver_id is not None:
                raise ValueError(
                    "solver_id is reserved for the real Pegasus QPU target."
                )
            if normalized_topology not in (None, "pegasus"):
                raise ValueError(
                    "Only Pegasus topology metadata is permitted."
                )


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One finite scalar metric with optional finite uncertainty."""

    name: str
    value: float
    uncertainty: float | None = None
    partition: DatasetPartition | None = None

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("Metric name must not be empty.")
        object.__setattr__(self, "name", normalized_name)

        numeric_value = float(self.value)
        if not math.isfinite(numeric_value):
            raise ValueError("Metric value must be finite.")
        object.__setattr__(self, "value", numeric_value)

        if self.uncertainty is not None:
            numeric_uncertainty = float(self.uncertainty)
            if not math.isfinite(numeric_uncertainty):
                raise ValueError("Metric uncertainty must be finite.")
            if numeric_uncertainty < 0.0:
                raise ValueError(
                    "Metric uncertainty must be non-negative."
                )
            object.__setattr__(
                self,
                "uncertainty",
                numeric_uncertainty,
            )

        if self.partition is not None and not isinstance(
            self.partition,
            DatasetPartition,
        ):
            raise TypeError(
                "partition must be a DatasetPartition or None."
            )


@dataclass(frozen=True, slots=True)
class RunDescriptor:
    """Immutable identity of one reproducible experiment run."""

    run_id: str
    level: SurrogateLevel
    target: ExecutionTarget
    seed: int
    status: RunStatus = RunStatus.CREATED

    def __post_init__(self) -> None:
        normalized_run_id = self.run_id.strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty.")
        object.__setattr__(self, "run_id", normalized_run_id)

        if not isinstance(self.level, SurrogateLevel):
            raise TypeError("level must be a SurrogateLevel.")
        if not isinstance(self.target, ExecutionTarget):
            raise TypeError("target must be an ExecutionTarget.")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if not 0 <= self.seed <= UINT32_MAX:
            raise ValueError(
                f"seed must lie in [0, {UINT32_MAX}]."
            )

        if (
            self.level is SurrogateLevel.OPF
            and self.target in {
                ExecutionTarget.AER_GPU,
                ExecutionTarget.SQA_GPU,
                ExecutionTarget.PEGASUS_QPU,
            }
        ):
            raise ValueError(
                "The OPF surrogate level cannot use a quantum backend."
            )

        if (
            self.level is SurrogateLevel.HARDWARE_RESIDUAL
            and self.target is not ExecutionTarget.PEGASUS_QPU
        ):
            raise ValueError(
                "Hardware residual data must come from Pegasus QPU runs."
            )


def canonical_coupler(u: Variable, v: Variable) -> Coupler:
    """Return a sorted undirected coupler and reject self-loops."""

    if isinstance(u, bool) or not isinstance(u, int):
        raise TypeError("u must be an integer variable label.")
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError("v must be an integer variable label.")
    if u < 0 or v < 0:
        raise ValueError("Variable labels must be non-negative.")
    if u == v:
        raise ValueError("Self-couplers are forbidden.")
    return (u, v) if u < v else (v, u)


__all__ = [
    "Variable",
    "Coupler",
    "UINT32_MAX",
    "ALLOWED_PEGASUS_SOLVER_PREFIXES",
    "SurrogateLevel",
    "ExecutionTarget",
    "ProblemDomain",
    "DatasetPartition",
    "RunStatus",
    "SeedBundle",
    "ProblemShape",
    "SolverIdentity",
    "MetricRecord",
    "RunDescriptor",
    "canonical_coupler",
]
