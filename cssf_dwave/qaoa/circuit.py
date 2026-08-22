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

"""Deterministic construction of parameterized standard-QAOA circuits.

The cost Hamiltonian convention is

    H_C = c I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j,

and one QAOA layer applies

    exp(-i gamma_l H_C) exp(-i beta_l sum_i X_i).

Under Qiskit's rotation conventions this becomes ``RZ(2*gamma*h)``,
``RZZ(2*gamma*J)``, and ``RX(2*beta)``. The constant cost offset is retained
as an exact parameterized global phase when requested.

Qiskit is imported lazily only when materializing a circuit. Importing this
module does not initialize a simulator, backend, optimizer, filesystem, or
network connection.
The exact statevector execution limit is enforced by the Aer GPU runner,
not by circuit construction.  This permits full-size circuits to be handed to
Qiskit Aer GPU tensor-network execution without weakening exact-statevector
safety.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qaoa.hamiltonian import (
    IsingHamiltonian,
    MAX_EXACT_BASIS_ENUMERATION_QUBITS,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
DEFAULT_REPETITIONS: Final[int] = 1
DEFAULT_CIRCUIT_NAME: Final[str] = "cssf_qaoa"
DEFAULT_GAMMA_PREFIX: Final[str] = "gamma"
DEFAULT_BETA_PREFIX: Final[str] = "beta"
COLAB_FREE_STATEVECTOR_QUBIT_LIMIT: Final[int] = (
    MAX_EXACT_BASIS_ENUMERATION_QUBITS
)

BarrierPolicy = Literal[
    "none",
    "between_blocks",
    "between_layers",
    "both",
]
OperationKind = Literal[
    "h",
    "rz",
    "rzz",
    "rx",
    "global_phase",
    "barrier",
]
ParameterFamily = Literal["gamma", "beta"]


class QAOACircuitError(ValueError):
    """Raised when a QAOA circuit specification is invalid."""


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise QAOACircuitError(
            f"{name} must be strictly positive."
        )
    return value


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise QAOACircuitError(f"{name} must be finite.")
    return normalized


def _nonempty_token(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise QAOACircuitError(
            f"{name} must be a non-empty string."
        )
    if any(character.isspace() for character in normalized):
        raise QAOACircuitError(
            f"{name} must not contain whitespace."
        )
    return normalized


def _readonly_parameter_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_size: int,
) -> NDArray[np.float64]:
    result = np.ascontiguousarray(
        np.asarray(values, dtype=REAL_DTYPE).reshape(-1),
        dtype=REAL_DTYPE,
    )
    if result.size != expected_size:
        raise QAOACircuitError(
            f"{name} must contain {expected_size} values; "
            f"received {result.size}."
        )
    if not np.all(np.isfinite(result)):
        raise QAOACircuitError(
            f"{name} contains non-finite values."
        )
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, init=False)
class QAOAParameterValues:
    """Immutable numerical gamma and beta vectors."""

    gamma: NDArray[np.float64]
    beta: NDArray[np.float64]

    def __init__(
        self,
        gamma: ArrayLike,
        beta: ArrayLike,
    ) -> None:
        gamma_array = np.asarray(gamma, dtype=REAL_DTYPE).reshape(-1)
        beta_array = np.asarray(beta, dtype=REAL_DTYPE).reshape(-1)

        if gamma_array.size == 0:
            raise QAOACircuitError(
                "gamma must contain at least one value."
            )
        if gamma_array.size != beta_array.size:
            raise QAOACircuitError(
                "gamma and beta must have equal lengths."
            )

        repetitions = int(gamma_array.size)
        object.__setattr__(
            self,
            "gamma",
            _readonly_parameter_vector(
                gamma_array,
                name="gamma",
                expected_size=repetitions,
            ),
        )
        object.__setattr__(
            self,
            "beta",
            _readonly_parameter_vector(
                beta_array,
                name="beta",
                expected_size=repetitions,
            ),
        )

    @property
    def repetitions(self) -> int:
        return int(self.gamma.size)

    def flat(self) -> NDArray[np.float64]:
        """Return ``[gamma_0..gamma_p-1, beta_0..beta_p-1]``."""

        result = np.ascontiguousarray(
            np.concatenate((self.gamma, self.beta)),
            dtype=REAL_DTYPE,
        )
        result.setflags(write=False)
        return result

    def interleaved(self) -> NDArray[np.float64]:
        """Return ``[gamma_0, beta_0, ..., gamma_p-1, beta_p-1]``."""

        result = np.empty(2 * self.repetitions, dtype=REAL_DTYPE)
        result[0::2] = self.gamma
        result[1::2] = self.beta
        result.setflags(write=False)
        return result

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 parameter fingerprint."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOAParameterValues-v1\0")
        digest.update(
            np.asarray([self.repetitions], dtype=np.int64).tobytes(
                order="C"
            )
        )
        digest.update(self.gamma.tobytes(order="C"))
        digest.update(self.beta.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAOAParameterLayout:
    """Stable names and vector layout for one repetition count."""

    repetitions: int
    gamma_prefix: str = DEFAULT_GAMMA_PREFIX
    beta_prefix: str = DEFAULT_BETA_PREFIX

    def __post_init__(self) -> None:
        repetitions = _positive_integer(
            self.repetitions,
            name="repetitions",
        )
        gamma_prefix = _nonempty_token(
            self.gamma_prefix,
            name="gamma_prefix",
        )
        beta_prefix = _nonempty_token(
            self.beta_prefix,
            name="beta_prefix",
        )
        if gamma_prefix == beta_prefix:
            raise QAOACircuitError(
                "gamma_prefix and beta_prefix must differ."
            )
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "gamma_prefix", gamma_prefix)
        object.__setattr__(self, "beta_prefix", beta_prefix)

    @property
    def gamma_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.gamma_prefix}[{index}]"
            for index in range(self.repetitions)
        )

    @property
    def beta_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.beta_prefix}[{index}]"
            for index in range(self.repetitions)
        )

    @property
    def flat_names(self) -> tuple[str, ...]:
        return self.gamma_names + self.beta_names

    @property
    def n_parameters(self) -> int:
        return 2 * self.repetitions

    def split(self, values: ArrayLike) -> QAOAParameterValues:
        """Split a flat gamma-first vector into immutable families."""

        vector = _readonly_parameter_vector(
            values,
            name="values",
            expected_size=self.n_parameters,
        )
        return QAOAParameterValues(
            vector[: self.repetitions],
            vector[self.repetitions :],
        )

    def from_interleaved(
        self,
        values: ArrayLike,
    ) -> QAOAParameterValues:
        """Split an interleaved gamma/beta optimizer vector."""

        vector = _readonly_parameter_vector(
            values,
            name="values",
            expected_size=self.n_parameters,
        )
        return QAOAParameterValues(
            vector[0::2],
            vector[1::2],
        )

    def constant(
        self,
        *,
        gamma: float,
        beta: float,
    ) -> QAOAParameterValues:
        """Create a constant initialization for every layer."""

        gamma_value = _finite_float(gamma, name="gamma")
        beta_value = _finite_float(beta, name="beta")
        return QAOAParameterValues(
            np.full(
                self.repetitions,
                gamma_value,
                dtype=REAL_DTYPE,
            ),
            np.full(
                self.repetitions,
                beta_value,
                dtype=REAL_DTYPE,
            ),
        )

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 layout fingerprint."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOAParameterLayout-v1\0")
        digest.update(
            json.dumps(
                {
                    "repetitions": self.repetitions,
                    "gamma_prefix": self.gamma_prefix,
                    "beta_prefix": self.beta_prefix,
                    "flat_names": self.flat_names,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAOACircuitConfig:
    """Validated standard-QAOA circuit construction options."""

    repetitions: int = DEFAULT_REPETITIONS
    barrier_policy: BarrierPolicy = "both"
    include_cost_offset_phase: bool = True
    circuit_name: str = DEFAULT_CIRCUIT_NAME
    gamma_prefix: str = DEFAULT_GAMMA_PREFIX
    beta_prefix: str = DEFAULT_BETA_PREFIX

    def __post_init__(self) -> None:
        repetitions = _positive_integer(
            self.repetitions,
            name="repetitions",
        )
        if self.barrier_policy not in (
            "none",
            "between_blocks",
            "between_layers",
            "both",
        ):
            raise QAOACircuitError(
                "barrier_policy must be one of: none, "
                "between_blocks, between_layers, both."
            )
        if not isinstance(self.include_cost_offset_phase, bool):
            raise TypeError(
                "include_cost_offset_phase must be boolean."
            )

        circuit_name = _nonempty_token(
            self.circuit_name,
            name="circuit_name",
        )
        gamma_prefix = _nonempty_token(
            self.gamma_prefix,
            name="gamma_prefix",
        )
        beta_prefix = _nonempty_token(
            self.beta_prefix,
            name="beta_prefix",
        )
        if gamma_prefix == beta_prefix:
            raise QAOACircuitError(
                "gamma_prefix and beta_prefix must differ."
            )

        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "circuit_name", circuit_name)
        object.__setattr__(self, "gamma_prefix", gamma_prefix)
        object.__setattr__(self, "beta_prefix", beta_prefix)

    @property
    def parameter_layout(self) -> QAOAParameterLayout:
        return QAOAParameterLayout(
            repetitions=self.repetitions,
            gamma_prefix=self.gamma_prefix,
            beta_prefix=self.beta_prefix,
        )

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 configuration fingerprint."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOACircuitConfig-v1\0")
        digest.update(
            json.dumps(
                {
                    "repetitions": self.repetitions,
                    "barrier_policy": self.barrier_policy,
                    "include_cost_offset_phase": (
                        self.include_cost_offset_phase
                    ),
                    "circuit_name": self.circuit_name,
                    "gamma_prefix": self.gamma_prefix,
                    "beta_prefix": self.beta_prefix,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAOAGateOperation:
    """One deterministic operation in a solver-independent QAOA plan."""

    kind: OperationKind
    layer: int
    qubits: tuple[int, ...]
    parameter_family: ParameterFamily | None
    rotation_scale: float

    def __post_init__(self) -> None:
        if self.kind not in (
            "h",
            "rz",
            "rzz",
            "rx",
            "global_phase",
            "barrier",
        ):
            raise QAOACircuitError(
                f"Unsupported operation kind {self.kind!r}."
            )
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise TypeError("layer must be an integer.")
        if self.layer < -1:
            raise QAOACircuitError(
                "layer must be -1 for initialization or non-negative."
            )

        qubits = tuple(self.qubits)
        if any(
            isinstance(qubit, bool)
            or not isinstance(qubit, int)
            or qubit < 0
            for qubit in qubits
        ):
            raise QAOACircuitError(
                "qubits must be non-negative integers."
            )
        if len(set(qubits)) != len(qubits):
            raise QAOACircuitError(
                "qubits must be unique within an operation."
            )

        expected_qubit_count = {
            "h": 1,
            "rz": 1,
            "rzz": 2,
            "rx": 1,
            "global_phase": 0,
        }.get(self.kind)
        if (
            expected_qubit_count is not None
            and len(qubits) != expected_qubit_count
        ):
            raise QAOACircuitError(
                f"{self.kind} requires {expected_qubit_count} qubits."
            )
        if self.kind == "rzz" and tuple(sorted(qubits)) != qubits:
            raise QAOACircuitError(
                "rzz qubits must be sorted."
            )

        family = self.parameter_family
        expected_family: ParameterFamily | None
        if self.kind in ("rz", "rzz", "global_phase"):
            expected_family = "gamma"
        elif self.kind == "rx":
            expected_family = "beta"
        else:
            expected_family = None
        if family != expected_family:
            raise QAOACircuitError(
                f"{self.kind} requires parameter_family="
                f"{expected_family!r}."
            )

        scale = _finite_float(
            self.rotation_scale,
            name="rotation_scale",
        )
        if self.kind in ("rz", "rzz") and scale == 0.0:
            raise QAOACircuitError(
                f"{self.kind} rotation_scale must be non-zero."
            )
        if self.kind == "rx" and scale != 2.0:
            raise QAOACircuitError(
                "Standard X mixer requires RX rotation_scale=2."
            )
        if self.kind in ("h", "barrier") and scale != 0.0:
            raise QAOACircuitError(
                f"{self.kind} rotation_scale must be zero."
            )

        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "rotation_scale", scale)


@dataclass(frozen=True, slots=True)
class QAOACircuitPlan:
    """Immutable complete gate plan before any Qiskit object is created."""

    hamiltonian: IsingHamiltonian
    config: QAOACircuitConfig
    parameter_layout: QAOAParameterLayout
    operations: tuple[QAOAGateOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hamiltonian, IsingHamiltonian):
            raise TypeError(
                "hamiltonian must be IsingHamiltonian."
            )
        if not isinstance(self.config, QAOACircuitConfig):
            raise TypeError("config must be QAOACircuitConfig.")
        if not isinstance(
            self.parameter_layout,
            QAOAParameterLayout,
        ):
            raise TypeError(
                "parameter_layout must be QAOAParameterLayout."
            )
        if (
            self.parameter_layout.repetitions
            != self.config.repetitions
        ):
            raise QAOACircuitError(
                "Parameter layout and config repetitions differ."
            )
        if self.parameter_layout.gamma_prefix != self.config.gamma_prefix:
            raise QAOACircuitError(
                "Parameter layout gamma prefix differs from config."
            )
        if self.parameter_layout.beta_prefix != self.config.beta_prefix:
            raise QAOACircuitError(
                "Parameter layout beta prefix differs from config."
            )

        operations = tuple(self.operations)
        if not operations:
            raise QAOACircuitError(
                "operations must not be empty."
            )
        for operation in operations:
            if not isinstance(operation, QAOAGateOperation):
                raise TypeError(
                    "Every operation must be QAOAGateOperation."
                )
            if any(
                qubit >= self.hamiltonian.n_qubits
                for qubit in operation.qubits
            ):
                raise QAOACircuitError(
                    "Operation references a qubit outside Hamiltonian."
                )
            if operation.layer >= self.config.repetitions:
                raise QAOACircuitError(
                    "Operation layer exceeds configured repetitions."
                )

        object.__setattr__(self, "operations", operations)

    @property
    def n_qubits(self) -> int:
        return self.hamiltonian.n_qubits

    @property
    def parameter_count(self) -> int:
        return self.parameter_layout.n_parameters

    def operation_counts(self) -> Mapping[str, int]:
        counts = {
            kind: 0
            for kind in (
                "h",
                "rz",
                "rzz",
                "rx",
                "global_phase",
                "barrier",
            )
        }
        for operation in self.operations:
            counts[operation.kind] += 1
        return MappingProxyType(counts)

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 plan fingerprint."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOACircuitPlan-v1\0")
        digest.update(
            self.hamiltonian.fingerprint().encode("ascii")
        )
        digest.update(self.config.fingerprint().encode("ascii"))
        digest.update(
            self.parameter_layout.fingerprint().encode("ascii")
        )
        digest.update(
            json.dumps(
                [
                    {
                        "kind": operation.kind,
                        "layer": operation.layer,
                        "qubits": operation.qubits,
                        "parameter_family": (
                            operation.parameter_family
                        ),
                        "rotation_scale": (
                            operation.rotation_scale
                        ),
                    }
                    for operation in self.operations
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


def build_qaoa_circuit_plan(
    hamiltonian: IsingHamiltonian,
    config: QAOACircuitConfig | None = None,
) -> QAOACircuitPlan:
    """Construct the exact standard-QAOA gate plan."""

    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError(
            "hamiltonian must be IsingHamiltonian."
        )
    circuit_config = (
        QAOACircuitConfig()
        if config is None
        else config
    )
    if not isinstance(circuit_config, QAOACircuitConfig):
        raise TypeError(
            "config must be QAOACircuitConfig or None."
        )

    operations: list[QAOAGateOperation] = []

    for qubit in range(hamiltonian.n_qubits):
        operations.append(
            QAOAGateOperation(
                kind="h",
                layer=-1,
                qubits=(qubit,),
                parameter_family=None,
                rotation_scale=0.0,
            )
        )

    for layer in range(circuit_config.repetitions):
        for qubit, coefficient in enumerate(
            hamiltonian.linear_z
        ):
            value = float(coefficient)
            if value != 0.0:
                operations.append(
                    QAOAGateOperation(
                        kind="rz",
                        layer=layer,
                        qubits=(qubit,),
                        parameter_family="gamma",
                        rotation_scale=2.0 * value,
                    )
                )

        for first in range(hamiltonian.n_qubits):
            for second in range(
                first + 1,
                hamiltonian.n_qubits,
            ):
                value = float(
                    hamiltonian.quadratic_zz[first, second]
                )
                if value != 0.0:
                    operations.append(
                        QAOAGateOperation(
                            kind="rzz",
                            layer=layer,
                            qubits=(first, second),
                            parameter_family="gamma",
                            rotation_scale=2.0 * value,
                        )
                    )

        if (
            circuit_config.include_cost_offset_phase
            and hamiltonian.offset != 0.0
        ):
            operations.append(
                QAOAGateOperation(
                    kind="global_phase",
                    layer=layer,
                    qubits=tuple(),
                    parameter_family="gamma",
                    rotation_scale=-hamiltonian.offset,
                )
            )

        if circuit_config.barrier_policy in (
            "between_blocks",
            "both",
        ):
            operations.append(
                QAOAGateOperation(
                    kind="barrier",
                    layer=layer,
                    qubits=tuple(range(hamiltonian.n_qubits)),
                    parameter_family=None,
                    rotation_scale=0.0,
                )
            )

        for qubit in range(hamiltonian.n_qubits):
            operations.append(
                QAOAGateOperation(
                    kind="rx",
                    layer=layer,
                    qubits=(qubit,),
                    parameter_family="beta",
                    rotation_scale=2.0,
                )
            )

        if (
            layer + 1 < circuit_config.repetitions
            and circuit_config.barrier_policy in (
                "between_layers",
                "both",
            )
        ):
            operations.append(
                QAOAGateOperation(
                    kind="barrier",
                    layer=layer,
                    qubits=tuple(range(hamiltonian.n_qubits)),
                    parameter_family=None,
                    rotation_scale=0.0,
                )
            )

    return QAOACircuitPlan(
        hamiltonian=hamiltonian,
        config=circuit_config,
        parameter_layout=circuit_config.parameter_layout,
        operations=tuple(operations),
    )


@dataclass(frozen=True, slots=True)
class QAOACircuitArtifact:
    """Materialized Qiskit circuit and its stable parameter handles."""

    plan: QAOACircuitPlan
    circuit: Any
    gamma_parameters: tuple[Any, ...]
    beta_parameters: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, QAOACircuitPlan):
            raise TypeError("plan must be QAOACircuitPlan.")
        gamma = tuple(self.gamma_parameters)
        beta = tuple(self.beta_parameters)
        repetitions = self.plan.config.repetitions
        if len(gamma) != repetitions or len(beta) != repetitions:
            raise QAOACircuitError(
                "Materialized parameter counts do not match repetitions."
            )
        object.__setattr__(self, "gamma_parameters", gamma)
        object.__setattr__(self, "beta_parameters", beta)

    @property
    def parameter_order(self) -> tuple[Any, ...]:
        return self.gamma_parameters + self.beta_parameters

    def parameter_bindings(
        self,
        values: QAOAParameterValues | ArrayLike,
    ) -> Mapping[Any, float]:
        """Return a strict Qiskit parameter-binding mapping."""

        if isinstance(values, QAOAParameterValues):
            normalized = values
        else:
            normalized = self.plan.parameter_layout.split(values)
        if normalized.repetitions != self.plan.config.repetitions:
            raise QAOACircuitError(
                "Parameter values use a different repetition count."
            )

        bindings = {
            parameter: float(value)
            for parameter, value in zip(
                self.gamma_parameters,
                normalized.gamma,
            )
        }
        bindings.update(
            {
                parameter: float(value)
                for parameter, value in zip(
                    self.beta_parameters,
                    normalized.beta,
                )
            }
        )
        return MappingProxyType(bindings)

    def bind(
        self,
        values: QAOAParameterValues | ArrayLike,
    ) -> Any:
        """Return a new fully assigned Qiskit circuit."""

        assign_parameters = getattr(
            self.circuit,
            "assign_parameters",
            None,
        )
        if assign_parameters is None or not callable(assign_parameters):
            raise QAOACircuitError(
                "Materialized circuit lacks assign_parameters()."
            )
        return assign_parameters(
            dict(self.parameter_bindings(values)),
            inplace=False,
            strict=True,
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOACircuitArtifact-v1\0")
        digest.update(self.plan.fingerprint().encode("ascii"))
        digest.update(
            json.dumps(
                [str(parameter) for parameter in self.parameter_order],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


def _load_qiskit_circuit_api() -> tuple[Any, Any]:
    """Load QuantumCircuit and ParameterVector without eager imports."""

    try:
        module = importlib.import_module("qiskit.circuit")
    except ImportError as exc:
        raise QAOACircuitError(
            "Qiskit is unavailable in the Google Colab runtime."
        ) from exc

    quantum_circuit = getattr(module, "QuantumCircuit", None)
    parameter_vector = getattr(module, "ParameterVector", None)

    if quantum_circuit is None or not callable(quantum_circuit):
        raise QAOACircuitError(
            "qiskit.circuit.QuantumCircuit is unavailable."
        )
    if parameter_vector is None or not callable(parameter_vector):
        raise QAOACircuitError(
            "qiskit.circuit.ParameterVector is unavailable."
        )

    return quantum_circuit, parameter_vector


def materialize_qiskit_circuit(
    plan: QAOACircuitPlan,
) -> QAOACircuitArtifact:
    """Materialize a parameterized Qiskit circuit from an exact plan."""

    if not isinstance(plan, QAOACircuitPlan):
        raise TypeError("plan must be QAOACircuitPlan.")

    QuantumCircuit, ParameterVector = _load_qiskit_circuit_api()
    gamma = tuple(
        ParameterVector(
            plan.config.gamma_prefix,
            plan.config.repetitions,
        )
    )
    beta = tuple(
        ParameterVector(
            plan.config.beta_prefix,
            plan.config.repetitions,
        )
    )
    circuit = QuantumCircuit(
        plan.n_qubits,
        name=plan.config.circuit_name,
    )

    for operation in plan.operations:
        if operation.kind == "h":
            circuit.h(operation.qubits[0])
        elif operation.kind == "rz":
            circuit.rz(
                operation.rotation_scale
                * gamma[operation.layer],
                operation.qubits[0],
            )
        elif operation.kind == "rzz":
            circuit.rzz(
                operation.rotation_scale
                * gamma[operation.layer],
                operation.qubits[0],
                operation.qubits[1],
            )
        elif operation.kind == "rx":
            circuit.rx(
                operation.rotation_scale
                * beta[operation.layer],
                operation.qubits[0],
            )
        elif operation.kind == "global_phase":
            circuit.global_phase = (
                circuit.global_phase
                + operation.rotation_scale
                * gamma[operation.layer]
            )
        elif operation.kind == "barrier":
            circuit.barrier(*operation.qubits)
        else:
            raise QAOACircuitError(
                f"Unhandled operation kind {operation.kind!r}."
            )

    circuit.metadata = {
        "framework": "CSSF",
        "algorithm": "CSNN-T^QAOA",
        "plan_fingerprint": plan.fingerprint(),
        "hamiltonian_fingerprint": (
            plan.hamiltonian.fingerprint()
        ),
        "config_fingerprint": plan.config.fingerprint(),
        "parameter_layout_fingerprint": (
            plan.parameter_layout.fingerprint()
        ),
        "repetitions": plan.config.repetitions,
        "n_qubits": plan.n_qubits,
        "parameter_count": plan.parameter_count,
        "variable_order": list(
            plan.hamiltonian.variable_order
        ),
        "parameter_order": list(
            plan.parameter_layout.flat_names
        ),
        "statevector_qubit_limit": (
            COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
        ),
    }

    return QAOACircuitArtifact(
        plan=plan,
        circuit=circuit,
        gamma_parameters=gamma,
        beta_parameters=beta,
    )


def build_parameterized_qaoa_circuit(
    hamiltonian: IsingHamiltonian,
    config: QAOACircuitConfig | None = None,
) -> QAOACircuitArtifact:
    """Build the exact plan and materialize its Qiskit circuit."""

    return materialize_qiskit_circuit(
        build_qaoa_circuit_plan(hamiltonian, config)
    )


__all__ = [
    "REAL_DTYPE",
    "DEFAULT_REPETITIONS",
    "DEFAULT_CIRCUIT_NAME",
    "DEFAULT_GAMMA_PREFIX",
    "DEFAULT_BETA_PREFIX",
    "COLAB_FREE_STATEVECTOR_QUBIT_LIMIT",
    "BarrierPolicy",
    "OperationKind",
    "ParameterFamily",
    "QAOACircuitError",
    "QAOAParameterValues",
    "QAOAParameterLayout",
    "QAOACircuitConfig",
    "QAOAGateOperation",
    "QAOACircuitPlan",
    "QAOACircuitArtifact",
    "build_qaoa_circuit_plan",
    "materialize_qiskit_circuit",
    "build_parameterized_qaoa_circuit",
]
