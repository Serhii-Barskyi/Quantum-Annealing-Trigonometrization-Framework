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

"""Deterministic Qiskit circuits for the CSSF digitized-QA teacher level.

The fixed research direction is::

    QA schedule -> interval integration -> MA-QAOA coordinates -> circuit

The schedule owns the driver and problem envelope integrals. The Ising
Hamiltonian owns the non-identity Z/ZZ coefficients. Consequently, a problem
rotation is ``RZ(2*h_i*gamma_li)`` or ``RZZ(2*J_ij*gamma_lij)`` and a driver
rotation is ``RX(2*beta_li)``. Ising coefficients are never multiplied into
schedule coordinates twice.

First-order digitization uses a problem block followed by a driver block in
each slice, matching the existing MA-QAOA gate convention. Second-order
digitization uses the explicit symmetric product formula
``driver/2 -> problem -> driver/2``. The Ising identity offset is omitted from
the circuit because it contributes only a global phase; expectation modules
must retain it when reporting energies.

Qiskit is imported lazily only while materializing a circuit. Importing this
module performs no simulator, optimizer, filesystem, network, emulator, or QPU
operation. Exact statevector execution remains restricted to Qiskit Aer on an
NVIDIA GPU and to at most 22 qubits by the package-level contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping

from config.schema import QAConfig
from maqaoa.parameters import (
    MAQAOAParameterError,
    MAQAOAParameterLayout,
    MAQAOAParameterValues,
    build_maqaoa_parameter_layout,
)
from qa import (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    QA_ALGORITHM_NAME,
    QA_TO_MAQAOA_DIRECTION,
    validate_exact_statevector_qubit_count,
)
from qa.schedule import (
    AnnealingSchedule,
    DigitizedQASchedule,
    QAScheduleError,
    digitize_schedule_from_config,
)
from qaoa.hamiltonian import IsingHamiltonian


DEFAULT_CIRCUIT_NAME: Final[str] = "cssf_digitized_qa"
DEFAULT_BARRIER_POLICY: Final[str] = "between_slices"
INITIAL_STATE_POLICY: Final[str] = "uniform_plus_state"
IDENTITY_OFFSET_CIRCUIT_POLICY: Final[str] = (
    "global_phase_omitted_expectation_retained"
)
FIRST_ORDER_SPLITTING: Final[str] = "problem_full_then_driver_full"
SECOND_ORDER_SPLITTING: Final[str] = (
    "driver_half_problem_full_driver_half"
)

BarrierPolicy = Literal[
    "none",
    "between_blocks",
    "between_slices",
    "both",
]
OperationKind = Literal["h", "rz", "rzz", "rx", "barrier"]
EvolutionBlock = Literal[
    "initial",
    "problem",
    "driver",
    "driver_pre",
    "driver_post",
    "barrier",
]


class QACircuitError(ValueError):
    """Raised when a digitized-QA circuit contract is violated."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise QACircuitError(f"{name} must be finite.")
    if normalized == 0.0:
        normalized = 0.0
    return normalized


def _nonempty_token(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise QACircuitError(f"{name} must be non-empty.")
    if any(character.isspace() for character in normalized):
        raise QACircuitError(f"{name} must not contain whitespace.")
    return normalized


def _angles_equal(first: float, second: float) -> bool:
    scale = max(1.0, abs(first), abs(second))
    tolerance = 32.0 * math.ulp(scale)
    return math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance)


@dataclass(frozen=True, slots=True)
class QACircuitConfig:
    """Validated deterministic digitized-QA circuit configuration."""

    barrier_policy: BarrierPolicy = DEFAULT_BARRIER_POLICY
    circuit_name: str = DEFAULT_CIRCUIT_NAME

    def __post_init__(self) -> None:
        policy = str(self.barrier_policy).strip().lower()
        allowed = {
            "none",
            "between_blocks",
            "between_slices",
            "both",
        }
        if policy not in allowed:
            raise QACircuitError(
                "barrier_policy must be one of "
                f"{tuple(sorted(allowed))}."
            )
        name = _nonempty_token(self.circuit_name, name="circuit_name")
        object.__setattr__(self, "barrier_policy", policy)
        object.__setattr__(self, "circuit_name", name)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QACircuitConfig-v1\0")
        digest.update(
            json.dumps(
                {
                    "barrier_policy": self.barrier_policy,
                    "circuit_name": self.circuit_name,
                    "initial_state_policy": INITIAL_STATE_POLICY,
                    "identity_offset_circuit_policy": (
                        IDENTITY_OFFSET_CIRCUIT_POLICY
                    ),
                    "first_order_splitting": FIRST_ORDER_SPLITTING,
                    "second_order_splitting": SECOND_ORDER_SPLITTING,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAGateOperation:
    """One fully numeric gate instruction in a digitized-QA plan."""

    kind: OperationKind
    slice_index: int
    block: EvolutionBlock
    qubits: tuple[int, ...]
    rotation_angle: float
    operator_coefficient: float
    applied_integral: float
    channel_index: int | None = None
    cost_term_index: int | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind)
        if kind not in {"h", "rz", "rzz", "rx", "barrier"}:
            raise QACircuitError(f"Unsupported operation kind {kind!r}.")

        block = str(self.block)
        allowed_blocks = {
            "initial",
            "problem",
            "driver",
            "driver_pre",
            "driver_post",
            "barrier",
        }
        if block not in allowed_blocks:
            raise QACircuitError(f"Unsupported evolution block {block!r}.")

        slice_index = self.slice_index
        if isinstance(slice_index, bool) or not isinstance(slice_index, int):
            raise TypeError("slice_index must be an integer.")
        if kind == "h":
            if slice_index != -1 or block != "initial":
                raise QACircuitError(
                    "Initial H operations require slice_index=-1 and "
                    "block='initial'."
                )
        elif slice_index < 0:
            raise QACircuitError(
                "Evolution operations require non-negative slice_index."
            )

        qubits = tuple(self.qubits)
        if any(
            isinstance(qubit, bool)
            or not isinstance(qubit, int)
            or qubit < 0
            for qubit in qubits
        ):
            raise QACircuitError(
                "qubits must contain non-negative integer indices."
            )
        if len(set(qubits)) != len(qubits):
            raise QACircuitError("qubits must not contain duplicates.")
        expected_locality = {"h": 1, "rz": 1, "rzz": 2, "rx": 1}
        if kind in expected_locality and len(qubits) != expected_locality[kind]:
            raise QACircuitError(
                f"{kind} requires {expected_locality[kind]} qubit(s)."
            )
        if kind == "barrier" and not qubits:
            raise QACircuitError("barrier must cover at least one qubit.")
        if kind == "rzz" and tuple(sorted(qubits)) != qubits:
            raise QACircuitError("rzz qubits must be sorted.")

        angle = _finite_float(
            self.rotation_angle,
            name="rotation_angle",
        )
        coefficient = _finite_float(
            self.operator_coefficient,
            name="operator_coefficient",
        )
        integral = _finite_float(
            self.applied_integral,
            name="applied_integral",
        )
        if integral < 0.0:
            raise QACircuitError("applied_integral must be non-negative.")

        channel = self.channel_index
        term = self.cost_term_index
        if kind in {"h", "barrier"}:
            if angle != 0.0 or coefficient != 0.0 or integral != 0.0:
                raise QACircuitError(
                    f"{kind} requires zero angle, coefficient, and integral."
                )
            if channel is not None or term is not None:
                raise QACircuitError(
                    f"{kind} must not reference schedule channels or terms."
                )
            if kind == "barrier" and block != "barrier":
                raise QACircuitError("barrier operations require block='barrier'.")
        elif kind == "rx":
            if block not in {"driver", "driver_pre", "driver_post"}:
                raise QACircuitError(
                    "RX operations require a driver evolution block."
                )
            if coefficient != 1.0:
                raise QACircuitError(
                    "Driver RX operations require operator_coefficient=1."
                )
            if isinstance(channel, bool) or not isinstance(channel, int):
                raise TypeError("Driver channel_index must be an integer.")
            if channel < 0:
                raise QACircuitError(
                    "Driver channel_index must be non-negative."
                )
            if term is not None:
                raise QACircuitError(
                    "Driver operations must not reference a cost term."
                )
            expected = 2.0 * integral
            if not _angles_equal(angle, expected):
                raise QACircuitError(
                    "Driver rotation_angle must equal 2*applied_integral."
                )
        else:
            if block != "problem":
                raise QACircuitError(
                    "RZ/RZZ operations require block='problem'."
                )
            if coefficient == 0.0:
                raise QACircuitError(
                    "Problem rotations require a non-zero coefficient."
                )
            if isinstance(channel, bool) or not isinstance(channel, int):
                raise TypeError("Problem channel_index must be an integer.")
            if channel < 0:
                raise QACircuitError(
                    "Problem channel_index must be non-negative."
                )
            if isinstance(term, bool) or not isinstance(term, int):
                raise TypeError("cost_term_index must be an integer.")
            if term < 0:
                raise QACircuitError(
                    "cost_term_index must be non-negative."
                )
            expected = 2.0 * coefficient * integral
            if not _angles_equal(angle, expected):
                raise QACircuitError(
                    "Problem rotation_angle must equal "
                    "2*operator_coefficient*applied_integral."
                )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "rotation_angle", angle)
        object.__setattr__(self, "operator_coefficient", coefficient)
        object.__setattr__(self, "applied_integral", integral)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAGateOperation-v1\0")
        digest.update(
            json.dumps(
                {
                    "kind": self.kind,
                    "slice_index": self.slice_index,
                    "block": self.block,
                    "qubits": self.qubits,
                    "rotation_angle": self.rotation_angle,
                    "operator_coefficient": self.operator_coefficient,
                    "applied_integral": self.applied_integral,
                    "channel_index": self.channel_index,
                    "cost_term_index": self.cost_term_index,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


def _barrier_operation(
    *,
    slice_index: int,
    qubits: tuple[int, ...],
) -> QAGateOperation:
    return QAGateOperation(
        kind="barrier",
        slice_index=slice_index,
        block="barrier",
        qubits=qubits,
        rotation_angle=0.0,
        operator_coefficient=0.0,
        applied_integral=0.0,
    )


def _problem_operations(
    *,
    slice_index: int,
    layout: MAQAOAParameterLayout,
    values: MAQAOAParameterValues,
) -> tuple[QAGateOperation, ...]:
    gamma = values.gamma[slice_index]
    operations: list[QAGateOperation] = []
    for descriptor in layout.cost_terms:
        integral = float(gamma[descriptor.term_index])
        kind: OperationKind = "rz" if descriptor.locality == 1 else "rzz"
        coefficient = float(descriptor.coefficient)
        operations.append(
            QAGateOperation(
                kind=kind,
                slice_index=slice_index,
                block="problem",
                qubits=descriptor.qubits,
                rotation_angle=2.0 * coefficient * integral,
                operator_coefficient=coefficient,
                applied_integral=integral,
                channel_index=descriptor.term_index,
                cost_term_index=descriptor.term_index,
            )
        )
    return tuple(operations)


def _driver_operations(
    *,
    slice_index: int,
    layout: MAQAOAParameterLayout,
    values: MAQAOAParameterValues,
    block: Literal["driver", "driver_pre", "driver_post"],
    fraction: float,
) -> tuple[QAGateOperation, ...]:
    beta = values.beta[slice_index]
    operations: list[QAGateOperation] = []
    for qubit in range(layout.n_qubits):
        integral = fraction * float(beta[qubit])
        operations.append(
            QAGateOperation(
                kind="rx",
                slice_index=slice_index,
                block=block,
                qubits=(qubit,),
                rotation_angle=2.0 * integral,
                operator_coefficient=1.0,
                applied_integral=integral,
                channel_index=qubit,
            )
        )
    return tuple(operations)


def _build_operations(
    *,
    schedule: DigitizedQASchedule,
    layout: MAQAOAParameterLayout,
    values: MAQAOAParameterValues,
    config: QACircuitConfig,
) -> tuple[QAGateOperation, ...]:
    operations: list[QAGateOperation] = []
    all_qubits = tuple(range(layout.n_qubits))

    for qubit in all_qubits:
        operations.append(
            QAGateOperation(
                kind="h",
                slice_index=-1,
                block="initial",
                qubits=(qubit,),
                rotation_angle=0.0,
                operator_coefficient=0.0,
                applied_integral=0.0,
            )
        )

    for slice_index in range(schedule.slice_count):
        problem = _problem_operations(
            slice_index=slice_index,
            layout=layout,
            values=values,
        )

        if schedule.trotter_order == 1:
            driver = _driver_operations(
                slice_index=slice_index,
                layout=layout,
                values=values,
                block="driver",
                fraction=1.0,
            )
            operations.extend(problem)
            if config.barrier_policy in {"between_blocks", "both"}:
                operations.append(
                    _barrier_operation(
                        slice_index=slice_index,
                        qubits=all_qubits,
                    )
                )
            operations.extend(driver)
        else:
            driver_pre = _driver_operations(
                slice_index=slice_index,
                layout=layout,
                values=values,
                block="driver_pre",
                fraction=0.5,
            )
            driver_post = _driver_operations(
                slice_index=slice_index,
                layout=layout,
                values=values,
                block="driver_post",
                fraction=0.5,
            )
            operations.extend(driver_pre)
            if config.barrier_policy in {"between_blocks", "both"}:
                operations.append(
                    _barrier_operation(
                        slice_index=slice_index,
                        qubits=all_qubits,
                    )
                )
            operations.extend(problem)
            if config.barrier_policy in {"between_blocks", "both"}:
                operations.append(
                    _barrier_operation(
                        slice_index=slice_index,
                        qubits=all_qubits,
                    )
                )
            operations.extend(driver_post)

        if (
            slice_index + 1 < schedule.slice_count
            and config.barrier_policy in {"between_slices", "both"}
        ):
            operations.append(
                _barrier_operation(
                    slice_index=slice_index,
                    qubits=all_qubits,
                )
            )

    return tuple(operations)


@dataclass(frozen=True, slots=True)
class QACircuitPlan:
    """Immutable numeric gate plan before any Qiskit object is created."""

    hamiltonian: IsingHamiltonian
    schedule: DigitizedQASchedule
    config: QACircuitConfig
    parameter_layout: MAQAOAParameterLayout
    mapped_values: MAQAOAParameterValues
    operations: tuple[QAGateOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hamiltonian, IsingHamiltonian):
            raise TypeError("hamiltonian must be IsingHamiltonian.")
        if not isinstance(self.schedule, DigitizedQASchedule):
            raise TypeError("schedule must be DigitizedQASchedule.")
        if not isinstance(self.config, QACircuitConfig):
            raise TypeError("config must be QACircuitConfig.")
        if not isinstance(self.parameter_layout, MAQAOAParameterLayout):
            raise TypeError(
                "parameter_layout must be MAQAOAParameterLayout."
            )
        if not isinstance(self.mapped_values, MAQAOAParameterValues):
            raise TypeError("mapped_values must be MAQAOAParameterValues.")

        validate_exact_statevector_qubit_count(self.hamiltonian.n_qubits)
        self.parameter_layout.validate_hamiltonian(self.hamiltonian)
        if self.parameter_layout.repetitions != self.schedule.slice_count:
            raise QACircuitError(
                "Parameter-layout repetitions must equal schedule slices."
            )

        normalized_values = self.parameter_layout.values(
            gamma=self.mapped_values.gamma,
            beta=self.mapped_values.beta,
        )
        try:
            expected_values = self.schedule.to_maqaoa_values(
                self.parameter_layout
            )
        except QAScheduleError as exc:
            raise QACircuitError(
                f"Cannot validate QA-to-MA-QAOA coordinates: {exc}"
            ) from exc
        if normalized_values.fingerprint() != expected_values.fingerprint():
            raise QACircuitError(
                "mapped_values differ from the schedule-derived "
                "MA-QAOA coordinates."
            )

        operations = tuple(self.operations)
        if not operations:
            raise QACircuitError("operations must not be empty.")
        if any(
            not isinstance(operation, QAGateOperation)
            for operation in operations
        ):
            raise TypeError(
                "operations must contain QAGateOperation objects."
            )
        for operation in operations:
            if any(
                qubit >= self.hamiltonian.n_qubits
                for qubit in operation.qubits
            ):
                raise QACircuitError(
                    "An operation references a qubit outside Hamiltonian."
                )
            if operation.slice_index >= self.schedule.slice_count:
                raise QACircuitError(
                    "An operation references a slice outside schedule."
                )
            if operation.channel_index is not None:
                if operation.kind == "rx":
                    channel_limit = self.parameter_layout.n_qubits
                else:
                    channel_limit = self.parameter_layout.cost_term_count
                if operation.channel_index >= channel_limit:
                    raise QACircuitError(
                        "An operation references an invalid mapped channel."
                    )
            if (
                operation.cost_term_index is not None
                and operation.cost_term_index
                >= self.parameter_layout.cost_term_count
            ):
                raise QACircuitError(
                    "An operation references an invalid cost term."
                )

        expected_operations = _build_operations(
            schedule=self.schedule,
            layout=self.parameter_layout,
            values=expected_values,
            config=self.config,
        )
        actual_fingerprints = tuple(
            operation.fingerprint() for operation in operations
        )
        expected_fingerprints = tuple(
            operation.fingerprint() for operation in expected_operations
        )
        if actual_fingerprints != expected_fingerprints:
            raise QACircuitError(
                "operations differ from the deterministic schedule-derived "
                "product-formula plan."
            )

        object.__setattr__(self, "mapped_values", normalized_values)
        object.__setattr__(self, "operations", operations)

    @property
    def n_qubits(self) -> int:
        return self.hamiltonian.n_qubits

    @property
    def slice_count(self) -> int:
        return self.schedule.slice_count

    @property
    def trotter_order(self) -> int:
        return self.schedule.trotter_order

    @property
    def splitting_policy(self) -> str:
        if self.trotter_order == 1:
            return FIRST_ORDER_SPLITTING
        return SECOND_ORDER_SPLITTING

    def operation_counts(self) -> Mapping[str, int]:
        counts = {
            kind: 0
            for kind in ("h", "rz", "rzz", "rx", "barrier")
        }
        for operation in self.operations:
            counts[operation.kind] += 1
        return MappingProxyType(counts)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QACircuitPlan-v1\0")
        digest.update(self.hamiltonian.fingerprint().encode("ascii"))
        digest.update(self.schedule.fingerprint().encode("ascii"))
        digest.update(self.config.fingerprint().encode("ascii"))
        digest.update(self.parameter_layout.fingerprint().encode("ascii"))
        digest.update(self.mapped_values.fingerprint().encode("ascii"))
        for operation in self.operations:
            digest.update(operation.fingerprint().encode("ascii"))
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "cssf-digitized-qa-circuit-plan-v1",
            "mapping_direction": QA_TO_MAQAOA_DIRECTION,
            "algorithm": QA_ALGORITHM_NAME,
            "n_qubits": self.n_qubits,
            "slice_count": self.slice_count,
            "trotter_order": self.trotter_order,
            "splitting_policy": self.splitting_policy,
            "barrier_policy": self.config.barrier_policy,
            "variable_order": list(self.hamiltonian.variable_order),
            "operation_counts": dict(self.operation_counts()),
            "hamiltonian_fingerprint": self.hamiltonian.fingerprint(),
            "schedule_fingerprint": self.schedule.fingerprint(),
            "parameter_layout_fingerprint": (
                self.parameter_layout.fingerprint()
            ),
            "mapped_values_fingerprint": self.mapped_values.fingerprint(),
            "identity_offset_circuit_policy": (
                IDENTITY_OFFSET_CIRCUIT_POLICY
            ),
            "statevector_qubit_limit": (
                COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
            ),
            "fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True, slots=True)
class QACircuitArtifact:
    """Materialized fixed-angle Qiskit circuit and its exact plan."""

    plan: QACircuitPlan
    circuit: Any
    qiskit_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, QACircuitPlan):
            raise TypeError("plan must be QACircuitPlan.")
        version = _nonempty_token(
            self.qiskit_version,
            name="qiskit_version",
        )

        num_qubits = getattr(self.circuit, "num_qubits", None)
        if num_qubits != self.plan.n_qubits:
            raise QACircuitError(
                "Materialized circuit qubit count differs from the plan."
            )
        parameters = getattr(self.circuit, "parameters", None)
        if parameters is None:
            raise QACircuitError(
                "Materialized circuit does not expose parameters."
            )
        try:
            remaining_parameters = tuple(parameters)
        except TypeError as exc:
            raise QACircuitError(
                "Materialized circuit parameters are not iterable."
            ) from exc
        if remaining_parameters:
            raise QACircuitError(
                "Digitized-QA circuit must contain only fixed numeric angles."
            )

        metadata = getattr(self.circuit, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise QACircuitError(
                "Materialized circuit metadata is unavailable."
            )
        if metadata.get("plan_fingerprint") != self.plan.fingerprint():
            raise QACircuitError(
                "Materialized circuit metadata does not match the plan."
            )

        object.__setattr__(self, "qiskit_version", version)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QACircuitArtifact-v1\0")
        digest.update(self.plan.fingerprint().encode("ascii"))
        digest.update(self.qiskit_version.encode("utf-8"))
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "cssf-digitized-qa-circuit-artifact-v1",
            "plan_fingerprint": self.plan.fingerprint(),
            "artifact_fingerprint": self.fingerprint(),
            "qiskit_version": self.qiskit_version,
            "n_qubits": self.plan.n_qubits,
            "slice_count": self.plan.slice_count,
            "trotter_order": self.plan.trotter_order,
            "fixed_angle_circuit": True,
        }


def build_qa_circuit_plan(
    hamiltonian: IsingHamiltonian,
    schedule: DigitizedQASchedule,
    config: QACircuitConfig | None = None,
) -> QACircuitPlan:
    """Build a deterministic fixed-angle plan from a digitized schedule."""

    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError("hamiltonian must be IsingHamiltonian.")
    if not isinstance(schedule, DigitizedQASchedule):
        raise TypeError("schedule must be DigitizedQASchedule.")

    validate_exact_statevector_qubit_count(hamiltonian.n_qubits)
    circuit_config = QACircuitConfig() if config is None else config
    if not isinstance(circuit_config, QACircuitConfig):
        raise TypeError("config must be QACircuitConfig or None.")

    try:
        layout = build_maqaoa_parameter_layout(
            hamiltonian,
            repetitions=schedule.slice_count,
        )
        values = schedule.to_maqaoa_values(layout)
    except (MAQAOAParameterError, QAScheduleError) as exc:
        raise QACircuitError(
            f"Cannot map digitized QA to MA-QAOA coordinates: {exc}"
        ) from exc

    operations = _build_operations(
        schedule=schedule,
        layout=layout,
        values=values,
        config=circuit_config,
    )
    return QACircuitPlan(
        hamiltonian=hamiltonian,
        schedule=schedule,
        config=circuit_config,
        parameter_layout=layout,
        mapped_values=values,
        operations=operations,
    )


def build_qa_circuit_plan_from_config(
    hamiltonian: IsingHamiltonian,
    schedule: AnnealingSchedule,
    qa_config: QAConfig,
    circuit_config: QACircuitConfig | None = None,
) -> QACircuitPlan:
    """Digitize an annealing schedule and build its exact circuit plan."""

    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError("hamiltonian must be IsingHamiltonian.")
    if not isinstance(schedule, AnnealingSchedule):
        raise TypeError("schedule must be AnnealingSchedule.")
    if not isinstance(qa_config, QAConfig):
        raise TypeError("qa_config must be QAConfig.")

    validate_exact_statevector_qubit_count(hamiltonian.n_qubits)
    try:
        digitized = digitize_schedule_from_config(schedule, qa_config)
    except QAScheduleError as exc:
        raise QACircuitError(f"Cannot digitize QA schedule: {exc}") from exc
    return build_qa_circuit_plan(
        hamiltonian,
        digitized,
        circuit_config,
    )


def _load_qiskit_circuit_api() -> tuple[Any, str]:
    """Load QuantumCircuit lazily without importing simulator runtimes."""

    try:
        qiskit = importlib.import_module("qiskit")
        circuit_module = importlib.import_module("qiskit.circuit")
    except ImportError as exc:
        raise QACircuitError(
            "Qiskit is unavailable in the Google Colab runtime."
        ) from exc

    quantum_circuit = getattr(circuit_module, "QuantumCircuit", None)
    if quantum_circuit is None or not callable(quantum_circuit):
        raise QACircuitError(
            "qiskit.circuit.QuantumCircuit is unavailable."
        )
    return quantum_circuit, str(getattr(qiskit, "__version__", "unknown"))


def materialize_qiskit_qa_circuit(
    plan: QACircuitPlan,
) -> QACircuitArtifact:
    """Materialize one fixed-angle Qiskit circuit from an exact QA plan."""

    if not isinstance(plan, QACircuitPlan):
        raise TypeError("plan must be QACircuitPlan.")
    validate_exact_statevector_qubit_count(plan.n_qubits)

    QuantumCircuit, qiskit_version = _load_qiskit_circuit_api()
    try:
        circuit = QuantumCircuit(
            plan.n_qubits,
            name=plan.config.circuit_name,
        )
    except Exception as exc:
        raise QACircuitError(
            "Cannot initialize the Qiskit QuantumCircuit."
        ) from exc

    try:
        for operation in plan.operations:
            if operation.kind == "h":
                circuit.h(operation.qubits[0])
            elif operation.kind == "rz":
                circuit.rz(
                    operation.rotation_angle,
                    operation.qubits[0],
                )
            elif operation.kind == "rzz":
                circuit.rzz(
                    operation.rotation_angle,
                    operation.qubits[0],
                    operation.qubits[1],
                )
            elif operation.kind == "rx":
                circuit.rx(
                    operation.rotation_angle,
                    operation.qubits[0],
                )
            elif operation.kind == "barrier":
                circuit.barrier(*operation.qubits)
            else:
                raise QACircuitError(
                    f"Unhandled operation kind {operation.kind!r}."
                )
    except QACircuitError:
        raise
    except Exception as exc:
        raise QACircuitError(
            "Qiskit digitized-QA circuit materialization failed."
        ) from exc

    circuit.metadata = {
        "framework": "CSSF",
        "algorithm": QA_ALGORITHM_NAME,
        "mapping_direction": QA_TO_MAQAOA_DIRECTION,
        "plan_fingerprint": plan.fingerprint(),
        "hamiltonian_fingerprint": plan.hamiltonian.fingerprint(),
        "schedule_fingerprint": plan.schedule.fingerprint(),
        "source_schedule_fingerprint": (
            plan.schedule.source_schedule_fingerprint
        ),
        "parameter_layout_fingerprint": (
            plan.parameter_layout.fingerprint()
        ),
        "mapped_values_fingerprint": plan.mapped_values.fingerprint(),
        "n_qubits": plan.n_qubits,
        "slice_count": plan.slice_count,
        "trotter_order": plan.trotter_order,
        "splitting_policy": plan.splitting_policy,
        "variable_order": list(plan.hamiltonian.variable_order),
        "fixed_angle_circuit": True,
        "initial_state_policy": INITIAL_STATE_POLICY,
        "identity_offset_circuit_policy": (
            IDENTITY_OFFSET_CIRCUIT_POLICY
        ),
        "statevector_qubit_limit": COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    }

    return QACircuitArtifact(
        plan=plan,
        circuit=circuit,
        qiskit_version=qiskit_version,
    )


def build_digitized_qa_circuit(
    hamiltonian: IsingHamiltonian,
    schedule: DigitizedQASchedule,
    config: QACircuitConfig | None = None,
) -> QACircuitArtifact:
    """Build and materialize a fixed-angle digitized-QA circuit."""

    return materialize_qiskit_qa_circuit(
        build_qa_circuit_plan(hamiltonian, schedule, config)
    )


def build_qa_circuit_from_config(
    hamiltonian: IsingHamiltonian,
    schedule: AnnealingSchedule,
    qa_config: QAConfig,
    circuit_config: QACircuitConfig | None = None,
) -> QACircuitArtifact:
    """Digitize from QAConfig and materialize the resulting Qiskit circuit."""

    return materialize_qiskit_qa_circuit(
        build_qa_circuit_plan_from_config(
            hamiltonian,
            schedule,
            qa_config,
            circuit_config,
        )
    )


__all__ = [
    "DEFAULT_CIRCUIT_NAME",
    "DEFAULT_BARRIER_POLICY",
    "INITIAL_STATE_POLICY",
    "IDENTITY_OFFSET_CIRCUIT_POLICY",
    "FIRST_ORDER_SPLITTING",
    "SECOND_ORDER_SPLITTING",
    "BarrierPolicy",
    "OperationKind",
    "EvolutionBlock",
    "QACircuitError",
    "QACircuitConfig",
    "QAGateOperation",
    "QACircuitPlan",
    "QACircuitArtifact",
    "build_qa_circuit_plan",
    "build_qa_circuit_plan_from_config",
    "materialize_qiskit_qa_circuit",
    "build_digitized_qa_circuit",
    "build_qa_circuit_from_config",
]
