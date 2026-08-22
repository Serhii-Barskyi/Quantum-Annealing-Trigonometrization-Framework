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

"""Deterministic Qiskit circuit construction for multi-angle QAOA.

Every non-identity Ising Z/ZZ term receives an independent cost angle in each
layer, and every qubit receives an independent mixer angle in each layer. For
an Ising term with coefficient ``w`` the Qiskit rotation is
``RZ(2*w*gamma[layer, term])`` or
``RZZ(2*w*gamma[layer, term])``. The mixer rotation is
``RX(2*beta[layer, qubit])``.

The identity offset is intentionally absent from the parameterized circuit. It
is retained exactly by Hamiltonian-based expectation evaluation and therefore
cannot create redundant trainable global-phase parameters.

Qiskit is imported lazily only when a circuit is materialized. Importing this
module does not initialize a simulator, optimizer, filesystem, or network
connection.
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
from typing import Any, Final, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike

from maqaoa import (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    MAQAOA_ALGORITHM_NAME,
)
from maqaoa.parameters import (
    MAQAOACostTerm,
    MAQAOAParameterError,
    MAQAOAParameterLayout,
    MAQAOAParameterValues,
    build_maqaoa_parameter_layout,
)
from qaoa.hamiltonian import IsingHamiltonian


DEFAULT_CIRCUIT_NAME: Final[str] = "cssf_maqaoa"
DEFAULT_BARRIER_POLICY: Final[str] = "between_layers"
INITIAL_STATE_POLICY: Final[str] = "uniform_plus_state"
IDENTITY_OFFSET_CIRCUIT_POLICY: Final[str] = "excluded"

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
    "barrier",
]
ParameterFamily = Literal["gamma", "beta"]


class MAQAOACircuitError(ValueError):
    """Raised when an MA-QAOA circuit specification is invalid."""


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise MAQAOACircuitError(f"{name} must be positive.")
    return value


def _nonempty_token(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise MAQAOACircuitError(
            f"{name} must be a non-empty string."
        )
    if any(character.isspace() for character in normalized):
        raise MAQAOACircuitError(
            f"{name} must not contain whitespace."
        )
    return normalized


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MAQAOACircuitError(f"{name} must be finite.")
    return normalized


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = str(value)
    if len(normalized) != 64:
        raise MAQAOACircuitError(
            f"{name} must be a SHA-256 digest."
        )
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise MAQAOACircuitError(
            f"{name} must contain hexadecimal SHA-256 data."
        ) from exc
    return normalized


@dataclass(frozen=True, slots=True)
class MAQAOACircuitConfig:
    """Validated deterministic circuit-construction configuration."""

    repetitions: int = 1
    barrier_policy: BarrierPolicy = DEFAULT_BARRIER_POLICY
    circuit_name: str = DEFAULT_CIRCUIT_NAME

    def __post_init__(self) -> None:
        repetitions = _positive_integer(
            self.repetitions,
            name="repetitions",
        )
        barrier_policy = str(self.barrier_policy).strip().lower()
        allowed = {
            "none",
            "between_blocks",
            "between_layers",
            "both",
        }
        if barrier_policy not in allowed:
            raise MAQAOACircuitError(
                "barrier_policy must be one of "
                f"{tuple(sorted(allowed))}."
            )
        circuit_name = _nonempty_token(
            self.circuit_name,
            name="circuit_name",
        )

        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "barrier_policy", barrier_policy)
        object.__setattr__(self, "circuit_name", circuit_name)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOACircuitConfig-v1\0")
        digest.update(
            json.dumps(
                {
                    "repetitions": self.repetitions,
                    "barrier_policy": self.barrier_policy,
                    "circuit_name": self.circuit_name,
                    "initial_state_policy": INITIAL_STATE_POLICY,
                    "identity_offset_circuit_policy": (
                        IDENTITY_OFFSET_CIRCUIT_POLICY
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MAQAOAGateOperation:
    """One exact gate instruction in an MA-QAOA circuit plan."""

    kind: OperationKind
    layer: int
    qubits: tuple[int, ...]
    parameter_family: ParameterFamily | None
    parameter_index: int | None
    coefficient: float
    rotation_scale: float

    def __post_init__(self) -> None:
        kind = str(self.kind)
        allowed_kinds = {"h", "rz", "rzz", "rx", "barrier"}
        if kind not in allowed_kinds:
            raise MAQAOACircuitError(
                f"Unsupported operation kind {kind!r}."
            )

        layer = self.layer
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer.")
        if kind == "h":
            if layer != -1:
                raise MAQAOACircuitError(
                    "Initial H operations must use layer=-1."
                )
        elif layer < 0:
            raise MAQAOACircuitError(
                "Layered operations require a non-negative layer."
            )

        qubits = tuple(self.qubits)
        if any(
            isinstance(qubit, bool)
            or not isinstance(qubit, int)
            or qubit < 0
            for qubit in qubits
        ):
            raise MAQAOACircuitError(
                "qubits must be non-negative integers."
            )
        if len(set(qubits)) != len(qubits):
            raise MAQAOACircuitError(
                "qubits must not contain duplicates."
            )

        expected_locality = {
            "h": 1,
            "rz": 1,
            "rzz": 2,
            "rx": 1,
        }
        if kind in expected_locality and len(qubits) != expected_locality[kind]:
            raise MAQAOACircuitError(
                f"{kind} requires {expected_locality[kind]} qubit(s)."
            )
        if kind == "barrier" and not qubits:
            raise MAQAOACircuitError(
                "barrier must cover at least one qubit."
            )
        if kind == "rzz" and tuple(sorted(qubits)) != qubits:
            raise MAQAOACircuitError(
                "rzz qubits must be sorted."
            )

        family = self.parameter_family
        index = self.parameter_index
        coefficient = _finite_float(
            self.coefficient,
            name="coefficient",
        )
        scale = _finite_float(
            self.rotation_scale,
            name="rotation_scale",
        )

        if kind in {"rz", "rzz"}:
            if family != "gamma":
                raise MAQAOACircuitError(
                    "Cost rotations require gamma parameters."
                )
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(
                    "Cost parameter_index must be an integer."
                )
            if index < 0:
                raise MAQAOACircuitError(
                    "Cost parameter_index must be non-negative."
                )
            if coefficient == 0.0:
                raise MAQAOACircuitError(
                    "Cost-operation coefficient must be non-zero."
                )
            if scale != 2.0 * coefficient:
                raise MAQAOACircuitError(
                    "Cost rotation_scale must equal 2*coefficient."
                )
        elif kind == "rx":
            if family != "beta":
                raise MAQAOACircuitError(
                    "Mixer rotations require beta parameters."
                )
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(
                    "Mixer parameter_index must be an integer."
                )
            if index < 0:
                raise MAQAOACircuitError(
                    "Mixer parameter_index must be non-negative."
                )
            if coefficient != 1.0 or scale != 2.0:
                raise MAQAOACircuitError(
                    "Mixer rotations require coefficient=1 and scale=2."
                )
        else:
            if family is not None or index is not None:
                raise MAQAOACircuitError(
                    f"{kind} must not reference trainable parameters."
                )
            if coefficient != 0.0 or scale != 0.0:
                raise MAQAOACircuitError(
                    f"{kind} must use zero coefficient and scale."
                )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "rotation_scale", scale)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAGateOperation-v1\0")
        digest.update(
            json.dumps(
                {
                    "kind": self.kind,
                    "layer": self.layer,
                    "qubits": self.qubits,
                    "parameter_family": self.parameter_family,
                    "parameter_index": self.parameter_index,
                    "coefficient": self.coefficient,
                    "rotation_scale": self.rotation_scale,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MAQAOACircuitPlan:
    """Immutable exact gate plan before any Qiskit object is created."""

    hamiltonian: IsingHamiltonian
    config: MAQAOACircuitConfig
    parameter_layout: MAQAOAParameterLayout
    operations: tuple[MAQAOAGateOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hamiltonian, IsingHamiltonian):
            raise TypeError("hamiltonian must be IsingHamiltonian.")
        if not isinstance(self.config, MAQAOACircuitConfig):
            raise TypeError("config must be MAQAOACircuitConfig.")
        if not isinstance(self.parameter_layout, MAQAOAParameterLayout):
            raise TypeError(
                "parameter_layout must be MAQAOAParameterLayout."
            )
        self.parameter_layout.validate_hamiltonian(self.hamiltonian)
        if self.parameter_layout.repetitions != self.config.repetitions:
            raise MAQAOACircuitError(
                "Parameter layout and config repetitions differ."
            )

        operations = tuple(self.operations)
        if not operations:
            raise MAQAOACircuitError("operations must not be empty.")
        if any(
            not isinstance(operation, MAQAOAGateOperation)
            for operation in operations
        ):
            raise TypeError(
                "operations must contain MAQAOAGateOperation objects."
            )

        for operation in operations:
            if any(
                qubit >= self.hamiltonian.n_qubits
                for qubit in operation.qubits
            ):
                raise MAQAOACircuitError(
                    "Operation references a qubit outside Hamiltonian."
                )
            if operation.layer >= self.config.repetitions:
                raise MAQAOACircuitError(
                    "Operation layer exceeds configured repetitions."
                )
            if operation.parameter_family == "gamma":
                if operation.parameter_index is None or (
                    operation.parameter_index
                    >= self.parameter_layout.cost_term_count
                ):
                    raise MAQAOACircuitError(
                        "Cost operation references an invalid cost term."
                    )
            if operation.parameter_family == "beta":
                if operation.parameter_index is None or (
                    operation.parameter_index >= self.n_qubits
                ):
                    raise MAQAOACircuitError(
                        "Mixer operation references an invalid qubit angle."
                    )

        object.__setattr__(self, "operations", operations)

    @property
    def n_qubits(self) -> int:
        return self.hamiltonian.n_qubits

    @property
    def parameter_count(self) -> int:
        return self.parameter_layout.total_parameter_count

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
        digest.update(b"CSSF-MAQAOACircuitPlan-v1\0")
        digest.update(self.hamiltonian.fingerprint().encode("ascii"))
        digest.update(self.config.fingerprint().encode("ascii"))
        digest.update(
            self.parameter_layout.fingerprint().encode("ascii")
        )
        for operation in self.operations:
            digest.update(operation.fingerprint().encode("ascii"))
        return digest.hexdigest()


def _cost_operation(
    *,
    layer: int,
    descriptor: MAQAOACostTerm,
) -> MAQAOAGateOperation:
    kind: OperationKind = "rz" if descriptor.locality == 1 else "rzz"
    return MAQAOAGateOperation(
        kind=kind,
        layer=layer,
        qubits=descriptor.qubits,
        parameter_family="gamma",
        parameter_index=descriptor.term_index,
        coefficient=descriptor.coefficient,
        rotation_scale=2.0 * descriptor.coefficient,
    )


def build_maqaoa_circuit_plan(
    hamiltonian: IsingHamiltonian,
    config: MAQAOACircuitConfig | None = None,
) -> MAQAOACircuitPlan:
    """Construct the exact deterministic multi-angle QAOA gate plan."""

    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError("hamiltonian must be IsingHamiltonian.")
    circuit_config = (
        MAQAOACircuitConfig() if config is None else config
    )
    if not isinstance(circuit_config, MAQAOACircuitConfig):
        raise TypeError(
            "config must be MAQAOACircuitConfig or None."
        )
    try:
        layout = build_maqaoa_parameter_layout(
            hamiltonian,
            repetitions=circuit_config.repetitions,
        )
    except MAQAOAParameterError as exc:
        raise MAQAOACircuitError(
            f"Cannot build MA-QAOA parameter layout: {exc}"
        ) from exc

    operations: list[MAQAOAGateOperation] = []

    for qubit in range(hamiltonian.n_qubits):
        operations.append(
            MAQAOAGateOperation(
                kind="h",
                layer=-1,
                qubits=(qubit,),
                parameter_family=None,
                parameter_index=None,
                coefficient=0.0,
                rotation_scale=0.0,
            )
        )

    all_qubits = tuple(range(hamiltonian.n_qubits))
    for layer in range(circuit_config.repetitions):
        operations.extend(
            _cost_operation(layer=layer, descriptor=descriptor)
            for descriptor in layout.cost_terms
        )

        if circuit_config.barrier_policy in {
            "between_blocks",
            "both",
        }:
            operations.append(
                MAQAOAGateOperation(
                    kind="barrier",
                    layer=layer,
                    qubits=all_qubits,
                    parameter_family=None,
                    parameter_index=None,
                    coefficient=0.0,
                    rotation_scale=0.0,
                )
            )

        for qubit in range(hamiltonian.n_qubits):
            operations.append(
                MAQAOAGateOperation(
                    kind="rx",
                    layer=layer,
                    qubits=(qubit,),
                    parameter_family="beta",
                    parameter_index=qubit,
                    coefficient=1.0,
                    rotation_scale=2.0,
                )
            )

        if (
            layer + 1 < circuit_config.repetitions
            and circuit_config.barrier_policy
            in {"between_layers", "both"}
        ):
            operations.append(
                MAQAOAGateOperation(
                    kind="barrier",
                    layer=layer,
                    qubits=all_qubits,
                    parameter_family=None,
                    parameter_index=None,
                    coefficient=0.0,
                    rotation_scale=0.0,
                )
            )

    return MAQAOACircuitPlan(
        hamiltonian=hamiltonian,
        config=circuit_config,
        parameter_layout=layout,
        operations=tuple(operations),
    )


@dataclass(frozen=True, slots=True)
class MAQAOACircuitArtifact:
    """Materialized Qiskit circuit with stable parameter handles."""

    plan: MAQAOACircuitPlan
    circuit: Any
    gamma_parameters: tuple[tuple[Any, ...], ...]
    beta_parameters: tuple[tuple[Any, ...], ...]
    qiskit_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, MAQAOACircuitPlan):
            raise TypeError("plan must be MAQAOACircuitPlan.")

        gamma = tuple(tuple(row) for row in self.gamma_parameters)
        beta = tuple(tuple(row) for row in self.beta_parameters)
        if len(gamma) != self.plan.config.repetitions:
            raise MAQAOACircuitError(
                "gamma_parameters layer count differs from the plan."
            )
        if len(beta) != self.plan.config.repetitions:
            raise MAQAOACircuitError(
                "beta_parameters layer count differs from the plan."
            )
        if any(
            len(row) != self.plan.parameter_layout.cost_term_count
            for row in gamma
        ):
            raise MAQAOACircuitError(
                "gamma_parameters shape differs from the layout."
            )
        if any(len(row) != self.plan.n_qubits for row in beta):
            raise MAQAOACircuitError(
                "beta_parameters shape differs from the layout."
            )
        qiskit_version = _nonempty_token(
            self.qiskit_version,
            name="qiskit_version",
        )

        object.__setattr__(self, "gamma_parameters", gamma)
        object.__setattr__(self, "beta_parameters", beta)
        object.__setattr__(self, "qiskit_version", qiskit_version)

    @property
    def gamma_parameter_order(self) -> tuple[Any, ...]:
        return tuple(
            parameter
            for row in self.gamma_parameters
            for parameter in row
        )

    @property
    def beta_parameter_order(self) -> tuple[Any, ...]:
        return tuple(
            parameter
            for row in self.beta_parameters
            for parameter in row
        )

    @property
    def parameter_order(self) -> tuple[Any, ...]:
        return self.gamma_parameter_order + self.beta_parameter_order

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(str(parameter) for parameter in self.parameter_order)

    def parameter_bindings(
        self,
        values: MAQAOAParameterValues | ArrayLike,
    ) -> Mapping[Any, float]:
        """Return a strict immutable Qiskit parameter-binding mapping."""

        if isinstance(values, MAQAOAParameterValues):
            normalized = self.plan.parameter_layout.values(
                gamma=values.gamma,
                beta=values.beta,
            )
        else:
            normalized = self.plan.parameter_layout.split(values)

        bindings = {
            parameter: float(value)
            for parameter, value in zip(
                self.gamma_parameter_order,
                normalized.gamma.reshape(-1, order="C"),
                strict=True,
            )
        }
        bindings.update(
            {
                parameter: float(value)
                for parameter, value in zip(
                    self.beta_parameter_order,
                    normalized.beta.reshape(-1, order="C"),
                    strict=True,
                )
            }
        )
        if len(bindings) != self.plan.parameter_count:
            raise MAQAOACircuitError(
                "Parameter binding count differs from the circuit plan."
            )
        return MappingProxyType(bindings)

    def bind(
        self,
        values: MAQAOAParameterValues | ArrayLike,
    ) -> Any:
        """Return a new fully assigned Qiskit circuit."""

        assign_parameters = getattr(
            self.circuit,
            "assign_parameters",
            None,
        )
        if assign_parameters is None or not callable(assign_parameters):
            raise MAQAOACircuitError(
                "Materialized circuit lacks assign_parameters()."
            )
        try:
            return assign_parameters(
                dict(self.parameter_bindings(values)),
                inplace=False,
                strict=True,
            )
        except Exception as exc:
            raise MAQAOACircuitError(
                "Qiskit parameter assignment failed."
            ) from exc

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOACircuitArtifact-v1\0")
        digest.update(self.plan.fingerprint().encode("ascii"))
        digest.update(self.qiskit_version.encode("utf-8"))
        digest.update(
            json.dumps(
                self.parameter_names,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


def _load_qiskit_circuit_api() -> tuple[Any, Any, str]:
    """Load QuantumCircuit and Parameter without eager imports."""

    try:
        qiskit = importlib.import_module("qiskit")
        circuit_module = importlib.import_module("qiskit.circuit")
    except ImportError as exc:
        raise MAQAOACircuitError(
            "Qiskit is unavailable in the Google Colab runtime."
        ) from exc

    quantum_circuit = getattr(
        circuit_module,
        "QuantumCircuit",
        None,
    )
    parameter = getattr(circuit_module, "Parameter", None)
    if quantum_circuit is None or not callable(quantum_circuit):
        raise MAQAOACircuitError(
            "qiskit.circuit.QuantumCircuit is unavailable."
        )
    if parameter is None or not callable(parameter):
        raise MAQAOACircuitError(
            "qiskit.circuit.Parameter is unavailable."
        )

    return (
        quantum_circuit,
        parameter,
        str(getattr(qiskit, "__version__", "unknown")),
    )


def _parameter_matrices(
    layout: MAQAOAParameterLayout,
    parameter_class: Any,
) -> tuple[tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]]:
    gamma = tuple(
        tuple(
            parameter_class(layout.gamma_name(layer, term))
            for term in range(layout.cost_term_count)
        )
        for layer in range(layout.repetitions)
    )
    beta = tuple(
        tuple(
            parameter_class(layout.beta_name(layer, qubit))
            for qubit in range(layout.n_qubits)
        )
        for layer in range(layout.repetitions)
    )
    return gamma, beta


def materialize_qiskit_maqaoa_circuit(
    plan: MAQAOACircuitPlan,
) -> MAQAOACircuitArtifact:
    """Materialize a parameterized Qiskit circuit from an exact plan."""

    if not isinstance(plan, MAQAOACircuitPlan):
        raise TypeError("plan must be MAQAOACircuitPlan.")

    QuantumCircuit, Parameter, qiskit_version = (
        _load_qiskit_circuit_api()
    )
    gamma, beta = _parameter_matrices(
        plan.parameter_layout,
        Parameter,
    )
    try:
        circuit = QuantumCircuit(
            plan.n_qubits,
            name=plan.config.circuit_name,
        )
    except Exception as exc:
        raise MAQAOACircuitError(
            "Cannot initialize the Qiskit QuantumCircuit."
        ) from exc

    try:
        for operation in plan.operations:
            if operation.kind == "h":
                circuit.h(operation.qubits[0])
            elif operation.kind == "rz":
                circuit.rz(
                    operation.rotation_scale
                    * gamma[operation.layer][operation.parameter_index],
                    operation.qubits[0],
                )
            elif operation.kind == "rzz":
                circuit.rzz(
                    operation.rotation_scale
                    * gamma[operation.layer][operation.parameter_index],
                    operation.qubits[0],
                    operation.qubits[1],
                )
            elif operation.kind == "rx":
                circuit.rx(
                    operation.rotation_scale
                    * beta[operation.layer][operation.parameter_index],
                    operation.qubits[0],
                )
            elif operation.kind == "barrier":
                circuit.barrier(*operation.qubits)
            else:
                raise MAQAOACircuitError(
                    f"Unhandled operation kind {operation.kind!r}."
                )
    except MAQAOACircuitError:
        raise
    except Exception as exc:
        raise MAQAOACircuitError(
            "Qiskit circuit materialization failed."
        ) from exc

    circuit.metadata = {
        "framework": "CSSF",
        "algorithm": MAQAOA_ALGORITHM_NAME,
        "plan_fingerprint": plan.fingerprint(),
        "hamiltonian_fingerprint": plan.hamiltonian.fingerprint(),
        "parameter_layout_fingerprint": (
            plan.parameter_layout.fingerprint()
        ),
        "repetitions": plan.config.repetitions,
        "n_qubits": plan.n_qubits,
        "cost_term_count": plan.parameter_layout.cost_term_count,
        "parameter_count": plan.parameter_count,
        "variable_order": list(plan.hamiltonian.variable_order),
        "parameter_order": list(
            plan.parameter_layout.parameter_names()
        ),
        "initial_state_policy": INITIAL_STATE_POLICY,
        "identity_offset_circuit_policy": (
            IDENTITY_OFFSET_CIRCUIT_POLICY
        ),
        "statevector_qubit_limit": (
            COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
        ),
    }

    return MAQAOACircuitArtifact(
        plan=plan,
        circuit=circuit,
        gamma_parameters=gamma,
        beta_parameters=beta,
        qiskit_version=qiskit_version,
    )


def build_parameterized_maqaoa_circuit(
    hamiltonian: IsingHamiltonian,
    config: MAQAOACircuitConfig | None = None,
) -> MAQAOACircuitArtifact:
    """Build the exact plan and materialize its Qiskit circuit."""

    return materialize_qiskit_maqaoa_circuit(
        build_maqaoa_circuit_plan(hamiltonian, config)
    )


__all__ = [
    "DEFAULT_CIRCUIT_NAME",
    "DEFAULT_BARRIER_POLICY",
    "INITIAL_STATE_POLICY",
    "IDENTITY_OFFSET_CIRCUIT_POLICY",
    "BarrierPolicy",
    "OperationKind",
    "ParameterFamily",
    "MAQAOACircuitError",
    "MAQAOACircuitConfig",
    "MAQAOAGateOperation",
    "MAQAOACircuitPlan",
    "MAQAOACircuitArtifact",
    "build_maqaoa_circuit_plan",
    "materialize_qiskit_maqaoa_circuit",
    "build_parameterized_maqaoa_circuit",
]
