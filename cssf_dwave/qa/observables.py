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

"""Exact digitized-QA observables from strict Aer GPU statevectors.

This module evaluates the Ising problem Hamiltonian for a statevector produced
by :mod:`qa.aer_gpu`. Quantum execution remains exclusively owned by Qiskit Aer
with ``method="statevector"``, ``device="GPU"``, and double precision. The
observable layer reuses the audited Qiskit ``SparsePauliOp`` and
``Statevector.expectation_value`` kernel from :mod:`qaoa.expectation`; it does
not implement a quantum simulator.

Every result preserves digitized-QA ownership: schedule, source schedule,
Hamiltonian, circuit plan, materialized circuit, Aer configuration, and GPU
environment fingerprints are checked before a value is accepted. A chunked
diagonal audit supplies variance, spectral extrema, ground-state probability,
and little-endian diagnostic samples without allocating a dense basis matrix.
The project-wide exact-statevector ceiling remains 22 qubits.
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

from qa import (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    QA_ALGORITHM_NAME,
    QISKIT_QUBIT_ORDER,
    STATEVECTOR_DEVICE,
    STATEVECTOR_METHOD,
    STATEVECTOR_PRECISION,
    validate_exact_statevector_qubit_count,
)
from qa.aer_gpu import (
    QAAerGPUConfig,
    QAStatevectorResult,
    run_qa_statevector_gpu,
)
from qa.circuit import QACircuitArtifact
from qaoa.aer_gpu import AerStatevectorResult
from qaoa.expectation import (
    QAOAExpectationConfig,
    QAOAExpectationError,
    QAOAExpectationResult,
    QiskitCostOperatorArtifact,
    build_qiskit_cost_operator,
    evaluate_statevector_expectation,
)
from qaoa.hamiltonian import IsingHamiltonian


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
DEFAULT_EXPECTATION_ATOL: Final[float] = 1.0e-10
DEFAULT_IMAGINARY_ATOL: Final[float] = 1.0e-10
DEFAULT_VARIANCE_ATOL: Final[float] = 1.0e-10
DEFAULT_GROUND_ENERGY_ATOL: Final[float] = 1.0e-10
DEFAULT_BASIS_CHUNK_SIZE: Final[int] = 65536
OBSERVABLE_ENGINE: Final[str] = (
    "qiskit.quantum_info.Statevector.expectation_value"
)
OPERATOR_CLASS: Final[str] = "qiskit.quantum_info.SparsePauliOp"


class QAObservablesError(RuntimeError):
    """Raised when digitized-QA observable ownership or algebra fails."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise QAObservablesError(f"{name} must be finite.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized <= 0.0:
        raise QAObservablesError(f"{name} must be strictly positive.")
    return normalized


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise QAObservablesError(f"{name} must be strictly positive.")
    return value


def _sha256_digest(value: str, *, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64:
        raise QAObservablesError(f"{name} must be a SHA-256 digest.")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise QAObservablesError(
            f"{name} must contain hexadecimal characters."
        ) from exc
    return normalized


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _immutable_json_mapping(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    source = {} if metadata is None else dict(metadata)
    try:
        encoded = json.dumps(
            source,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise QAObservablesError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    frozen = _freeze_json(json.loads(encoded))
    if not isinstance(frozen, Mapping):
        raise QAObservablesError("metadata normalization failed.")
    return frozen


@dataclass(frozen=True, slots=True)
class QAObservablesConfig:
    """Numerical controls for exact QA expectation and spectral audits."""

    expectation_atol: float = DEFAULT_EXPECTATION_ATOL
    imaginary_atol: float = DEFAULT_IMAGINARY_ATOL
    variance_atol: float = DEFAULT_VARIANCE_ATOL
    ground_energy_atol: float = DEFAULT_GROUND_ENERGY_ATOL
    basis_chunk_size: int = DEFAULT_BASIS_CHUNK_SIZE
    require_diagonal_crosscheck: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expectation_atol",
            _positive_float(
                self.expectation_atol,
                name="expectation_atol",
            ),
        )
        object.__setattr__(
            self,
            "imaginary_atol",
            _positive_float(
                self.imaginary_atol,
                name="imaginary_atol",
            ),
        )
        object.__setattr__(
            self,
            "variance_atol",
            _positive_float(
                self.variance_atol,
                name="variance_atol",
            ),
        )
        object.__setattr__(
            self,
            "ground_energy_atol",
            _positive_float(
                self.ground_energy_atol,
                name="ground_energy_atol",
            ),
        )
        object.__setattr__(
            self,
            "basis_chunk_size",
            _positive_integer(
                self.basis_chunk_size,
                name="basis_chunk_size",
            ),
        )
        if not isinstance(self.require_diagonal_crosscheck, bool):
            raise TypeError(
                "require_diagonal_crosscheck must be boolean."
            )

    def shared_config(self) -> QAOAExpectationConfig:
        """Return the audited shared Ising-observable configuration."""

        return QAOAExpectationConfig(
            expectation_atol=self.expectation_atol,
            imaginary_atol=self.imaginary_atol,
            variance_atol=self.variance_atol,
            ground_energy_atol=self.ground_energy_atol,
            basis_chunk_size=self.basis_chunk_size,
            require_diagonal_crosscheck=(
                self.require_diagonal_crosscheck
            ),
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAObservablesConfig-v1\0")
        digest.update(
            json.dumps(
                {
                    "expectation_atol": self.expectation_atol,
                    "imaginary_atol": self.imaginary_atol,
                    "variance_atol": self.variance_atol,
                    "ground_energy_atol": self.ground_energy_atol,
                    "basis_chunk_size": self.basis_chunk_size,
                    "require_diagonal_crosscheck": (
                        self.require_diagonal_crosscheck
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QACostOperatorArtifact:
    """QA-owned wrapper around the shared exact Qiskit Ising operator."""

    shared_artifact: QiskitCostOperatorArtifact

    def __post_init__(self) -> None:
        if not isinstance(
            self.shared_artifact,
            QiskitCostOperatorArtifact,
        ):
            raise TypeError(
                "shared_artifact must be QiskitCostOperatorArtifact."
            )
        validate_exact_statevector_qubit_count(self.n_qubits)

    @property
    def operator(self) -> Any:
        return self.shared_artifact.operator

    @property
    def terms(self) -> tuple[tuple[str, float], ...]:
        return self.shared_artifact.terms

    @property
    def term_count(self) -> int:
        return self.shared_artifact.term_count

    @property
    def n_qubits(self) -> int:
        return self.shared_artifact.n_qubits

    @property
    def hamiltonian_fingerprint(self) -> str:
        return self.shared_artifact.hamiltonian_fingerprint

    @property
    def qiskit_version(self) -> str:
        return self.shared_artifact.qiskit_version

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QACostOperatorArtifact-v1\0")
        digest.update(self.shared_artifact.fingerprint().encode("ascii"))
        digest.update(QA_ALGORITHM_NAME.encode("utf-8"))
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "cssf-digitized-qa-cost-operator-v1",
            "operator_class": OPERATOR_CLASS,
            "n_qubits": self.n_qubits,
            "term_count": self.term_count,
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "qiskit_version": self.qiskit_version,
            "operator_fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True, slots=True)
class QAObservablesResult:
    """Exact QA energy statistics with complete source ownership."""

    audit_result: QAOAExpectationResult
    schedule_fingerprint: str
    source_schedule_fingerprint: str
    plan_fingerprint: str
    circuit_fingerprint: str
    qa_statevector_fingerprint: str
    shared_statevector_fingerprint: str
    operator_fingerprint: str
    config_fingerprint: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.audit_result, QAOAExpectationResult):
            raise TypeError(
                "audit_result must be QAOAExpectationResult."
            )
        validate_exact_statevector_qubit_count(
            self.audit_result.n_qubits
        )
        for name in (
            "schedule_fingerprint",
            "source_schedule_fingerprint",
            "plan_fingerprint",
            "circuit_fingerprint",
            "qa_statevector_fingerprint",
            "shared_statevector_fingerprint",
            "operator_fingerprint",
            "config_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _sha256_digest(getattr(self, name), name=name),
            )
        if (
            self.audit_result.statevector_fingerprint
            != self.shared_statevector_fingerprint
        ):
            raise QAObservablesError(
                "Shared observable audit belongs to another statevector."
            )
        object.__setattr__(
            self,
            "metadata",
            _immutable_json_mapping(self.metadata),
        )

    @property
    def expected_energy(self) -> float:
        return self.audit_result.expected_energy

    @property
    def qiskit_expected_energy(self) -> float:
        return self.audit_result.qiskit_expected_energy

    @property
    def diagonal_expected_energy(self) -> float:
        return self.audit_result.diagonal_expected_energy

    @property
    def expectation_discrepancy(self) -> float:
        return self.audit_result.expectation_discrepancy

    @property
    def variance(self) -> float:
        return self.audit_result.variance

    @property
    def standard_deviation(self) -> float:
        return self.audit_result.standard_deviation

    @property
    def minimum_energy(self) -> float:
        return self.audit_result.minimum_energy

    @property
    def maximum_energy(self) -> float:
        return self.audit_result.maximum_energy

    @property
    def ground_probability(self) -> float:
        return self.audit_result.ground_probability

    @property
    def ground_degeneracy(self) -> int:
        return self.audit_result.ground_degeneracy

    @property
    def representative_ground_index(self) -> int:
        return self.audit_result.representative_ground_index

    @property
    def representative_ground_sample(self) -> NDArray[np.int8]:
        return self.audit_result.representative_ground_sample

    @property
    def representative_ground_probability(self) -> float:
        return self.audit_result.representative_ground_probability

    @property
    def most_probable_index(self) -> int:
        return self.audit_result.most_probable_index

    @property
    def most_probable_sample(self) -> NDArray[np.int8]:
        return self.audit_result.most_probable_sample

    @property
    def most_probable_probability(self) -> float:
        return self.audit_result.most_probable_probability

    @property
    def most_probable_energy(self) -> float:
        return self.audit_result.most_probable_energy

    @property
    def n_qubits(self) -> int:
        return self.audit_result.n_qubits

    @property
    def variable_order(self) -> tuple[str, ...]:
        return self.audit_result.variable_order

    @property
    def hamiltonian_fingerprint(self) -> str:
        return self.audit_result.hamiltonian_fingerprint

    def labeled_ground_sample(self) -> dict[str, int]:
        return self.audit_result.labeled_ground_sample()

    def labeled_most_probable_sample(self) -> dict[str, int]:
        return self.audit_result.labeled_most_probable_sample()

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAObservablesResult-v1\0")
        digest.update(self.audit_result.fingerprint().encode("ascii"))
        for value in (
            self.schedule_fingerprint,
            self.source_schedule_fingerprint,
            self.plan_fingerprint,
            self.circuit_fingerprint,
            self.qa_statevector_fingerprint,
            self.shared_statevector_fingerprint,
            self.operator_fingerprint,
            self.config_fingerprint,
        ):
            digest.update(value.encode("ascii"))
        digest.update(
            json.dumps(
                _thaw_json(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "cssf-digitized-qa-observables-result-v1",
            "algorithm": QA_ALGORITHM_NAME,
            "observable_engine": OBSERVABLE_ENGINE,
            "operator_class": OPERATOR_CLASS,
            "method": STATEVECTOR_METHOD,
            "device": STATEVECTOR_DEVICE,
            "precision": STATEVECTOR_PRECISION,
            "qubit_order": QISKIT_QUBIT_ORDER,
            "n_qubits": self.n_qubits,
            "variable_order": list(self.variable_order),
            "expected_energy": self.expected_energy,
            "variance": self.variance,
            "minimum_energy": self.minimum_energy,
            "maximum_energy": self.maximum_energy,
            "ground_probability": self.ground_probability,
            "ground_degeneracy": self.ground_degeneracy,
            "most_probable_index": self.most_probable_index,
            "most_probable_probability": (
                self.most_probable_probability
            ),
            "most_probable_energy": self.most_probable_energy,
            "schedule_fingerprint": self.schedule_fingerprint,
            "source_schedule_fingerprint": (
                self.source_schedule_fingerprint
            ),
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "circuit_fingerprint": self.circuit_fingerprint,
            "qa_statevector_fingerprint": (
                self.qa_statevector_fingerprint
            ),
            "operator_fingerprint": self.operator_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "result_fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True, slots=True)
class QAGPUEvaluation:
    """Fixed QA circuit, strict GPU statevector, and exact observables."""

    circuit_artifact: QACircuitArtifact
    statevector_result: QAStatevectorResult
    observables_result: QAObservablesResult

    def __post_init__(self) -> None:
        if not isinstance(self.circuit_artifact, QACircuitArtifact):
            raise TypeError(
                "circuit_artifact must be QACircuitArtifact."
            )
        if not isinstance(self.statevector_result, QAStatevectorResult):
            raise TypeError(
                "statevector_result must be QAStatevectorResult."
            )
        if not isinstance(self.observables_result, QAObservablesResult):
            raise TypeError(
                "observables_result must be QAObservablesResult."
            )

        artifact_fingerprint = self.circuit_artifact.fingerprint()
        plan_fingerprint = self.circuit_artifact.plan.fingerprint()
        statevector_fingerprint = self.statevector_result.fingerprint()
        if self.statevector_result.circuit_fingerprint != artifact_fingerprint:
            raise QAObservablesError(
                "Statevector belongs to another QA circuit artifact."
            )
        if self.statevector_result.plan_fingerprint != plan_fingerprint:
            raise QAObservablesError(
                "Statevector belongs to another QA circuit plan."
            )
        if (
            self.observables_result.qa_statevector_fingerprint
            != statevector_fingerprint
        ):
            raise QAObservablesError(
                "Observable result belongs to another QA statevector."
            )
        if self.observables_result.circuit_fingerprint != artifact_fingerprint:
            raise QAObservablesError(
                "Observable result belongs to another QA circuit artifact."
            )
        if self.observables_result.plan_fingerprint != plan_fingerprint:
            raise QAObservablesError(
                "Observable result belongs to another QA circuit plan."
            )

    @property
    def objective_value(self) -> float:
        return self.observables_result.expected_energy

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAGPUEvaluation-v1\0")
        digest.update(self.circuit_artifact.fingerprint().encode("ascii"))
        digest.update(self.statevector_result.fingerprint().encode("ascii"))
        digest.update(self.observables_result.fingerprint().encode("ascii"))
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "cssf-digitized-qa-gpu-evaluation-v1",
            "algorithm": QA_ALGORITHM_NAME,
            "circuit_fingerprint": self.circuit_artifact.fingerprint(),
            "statevector_fingerprint": (
                self.statevector_result.fingerprint()
            ),
            "observables_fingerprint": (
                self.observables_result.fingerprint()
            ),
            "objective_value": self.objective_value,
            "evaluation_fingerprint": self.fingerprint(),
        }


def build_qa_qiskit_cost_operator(
    hamiltonian: IsingHamiltonian,
) -> QACostOperatorArtifact:
    """Build the exact QA Ising observable as a Qiskit SparsePauliOp."""

    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError("hamiltonian must be IsingHamiltonian.")
    validate_exact_statevector_qubit_count(hamiltonian.n_qubits)
    try:
        shared = build_qiskit_cost_operator(hamiltonian)
    except QAOAExpectationError as exc:
        raise QAObservablesError(str(exc)) from exc
    return QACostOperatorArtifact(shared_artifact=shared)


def _shared_statevector_adapter(
    statevector_result: QAStatevectorResult,
) -> AerStatevectorResult:
    return AerStatevectorResult(
        statevector=statevector_result.statevector,
        probabilities=statevector_result.probabilities,
        n_qubits=statevector_result.n_qubits,
        variable_order=statevector_result.variable_order,
        circuit_fingerprint=statevector_result.circuit_fingerprint,
        config_fingerprint=statevector_result.config_fingerprint,
        environment_fingerprint=(
            statevector_result.environment_fingerprint
        ),
        transpiled_depth=statevector_result.transpiled_depth,
        transpiled_size=statevector_result.transpiled_size,
        metadata={
            "framework": "CSSF",
            "algorithm": QA_ALGORITHM_NAME,
            "adapter": "qa_to_shared_ising_observables",
            "qa_statevector_fingerprint": (
                statevector_result.fingerprint()
            ),
            "schedule_fingerprint": (
                statevector_result.schedule_fingerprint
            ),
            "source_schedule_fingerprint": (
                statevector_result.source_schedule_fingerprint
            ),
            "plan_fingerprint": statevector_result.plan_fingerprint,
        },
    )


def evaluate_qa_statevector_observables(
    statevector_result: QAStatevectorResult,
    hamiltonian: IsingHamiltonian,
    *,
    config: QAObservablesConfig | None = None,
    operator_artifact: QACostOperatorArtifact | None = None,
) -> QAObservablesResult:
    """Evaluate exact Ising observables for one QA Aer GPU statevector."""

    if not isinstance(statevector_result, QAStatevectorResult):
        raise TypeError(
            "statevector_result must be QAStatevectorResult."
        )
    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError("hamiltonian must be IsingHamiltonian.")

    validate_exact_statevector_qubit_count(hamiltonian.n_qubits)
    if statevector_result.n_qubits != hamiltonian.n_qubits:
        raise QAObservablesError(
            "Statevector and Hamiltonian qubit counts differ."
        )
    if statevector_result.variable_order != hamiltonian.variable_order:
        raise QAObservablesError(
            "Statevector and Hamiltonian variable orders differ."
        )
    hamiltonian_fingerprint = hamiltonian.fingerprint()
    if (
        statevector_result.hamiltonian_fingerprint
        != hamiltonian_fingerprint
    ):
        raise QAObservablesError(
            "QA statevector belongs to another Hamiltonian."
        )

    run_config = QAObservablesConfig() if config is None else config
    if not isinstance(run_config, QAObservablesConfig):
        raise TypeError(
            "config must be QAObservablesConfig or None."
        )

    artifact = (
        build_qa_qiskit_cost_operator(hamiltonian)
        if operator_artifact is None
        else operator_artifact
    )
    if not isinstance(artifact, QACostOperatorArtifact):
        raise TypeError(
            "operator_artifact must be QACostOperatorArtifact."
        )
    if artifact.hamiltonian_fingerprint != hamiltonian_fingerprint:
        raise QAObservablesError(
            "Qiskit operator belongs to another Hamiltonian."
        )

    shared_statevector = _shared_statevector_adapter(statevector_result)
    try:
        audit_result = evaluate_statevector_expectation(
            shared_statevector,
            hamiltonian,
            config=run_config.shared_config(),
            operator_artifact=artifact.shared_artifact,
        )
    except QAOAExpectationError as exc:
        raise QAObservablesError(str(exc)) from exc

    if audit_result.operator_fingerprint != (
        artifact.shared_artifact.fingerprint()
    ):
        raise QAObservablesError(
            "Shared observable audit returned another operator."
        )

    return QAObservablesResult(
        audit_result=audit_result,
        schedule_fingerprint=(
            statevector_result.schedule_fingerprint
        ),
        source_schedule_fingerprint=(
            statevector_result.source_schedule_fingerprint
        ),
        plan_fingerprint=statevector_result.plan_fingerprint,
        circuit_fingerprint=statevector_result.circuit_fingerprint,
        qa_statevector_fingerprint=statevector_result.fingerprint(),
        shared_statevector_fingerprint=(
            shared_statevector.fingerprint()
        ),
        operator_fingerprint=artifact.fingerprint(),
        config_fingerprint=run_config.fingerprint(),
        metadata={
            "framework": "CSSF",
            "algorithm": QA_ALGORITHM_NAME,
            "execution_engine": "qiskit_aer.AerSimulator",
            "execution_method": STATEVECTOR_METHOD,
            "execution_device": STATEVECTOR_DEVICE,
            "execution_precision": STATEVECTOR_PRECISION,
            "observable_engine": OBSERVABLE_ENGINE,
            "operator_class": OPERATOR_CLASS,
            "basis_order": QISKIT_QUBIT_ORDER,
            "basis_chunk_size": run_config.basis_chunk_size,
            "diagonal_crosscheck": (
                run_config.require_diagonal_crosscheck
            ),
            "qiskit_version": artifact.qiskit_version,
            "statevector_qubit_limit": (
                COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
            ),
            "cpu_fallback": False,
        },
    )


def evaluate_qa_circuit_gpu(
    artifact: QACircuitArtifact,
    hamiltonian: IsingHamiltonian,
    *,
    aer_config: QAAerGPUConfig | None = None,
    observables_config: QAObservablesConfig | None = None,
    operator_artifact: QACostOperatorArtifact | None = None,
) -> QAGPUEvaluation:
    """Run one fixed QA circuit on Aer GPU and evaluate exact observables."""

    if not isinstance(artifact, QACircuitArtifact):
        raise TypeError("artifact must be QACircuitArtifact.")
    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError("hamiltonian must be IsingHamiltonian.")

    validate_exact_statevector_qubit_count(hamiltonian.n_qubits)
    if artifact.plan.hamiltonian.fingerprint() != hamiltonian.fingerprint():
        raise QAObservablesError(
            "Circuit artifact and Hamiltonian differ."
        )

    statevector_result = run_qa_statevector_gpu(
        artifact,
        config=aer_config,
    )
    observables_result = evaluate_qa_statevector_observables(
        statevector_result,
        hamiltonian,
        config=observables_config,
        operator_artifact=operator_artifact,
    )
    return QAGPUEvaluation(
        circuit_artifact=artifact,
        statevector_result=statevector_result,
        observables_result=observables_result,
    )


def evaluate_digitized_qa_gpu(
    artifact: QACircuitArtifact,
    hamiltonian: IsingHamiltonian,
    *,
    aer_config: QAAerGPUConfig | None = None,
    observables_config: QAObservablesConfig | None = None,
    operator_artifact: QACostOperatorArtifact | None = None,
) -> QAGPUEvaluation:
    """Explicitly named alias for :func:`evaluate_qa_circuit_gpu`."""

    return evaluate_qa_circuit_gpu(
        artifact,
        hamiltonian,
        aer_config=aer_config,
        observables_config=observables_config,
        operator_artifact=operator_artifact,
    )


__all__ = [
    "REAL_DTYPE",
    "DEFAULT_EXPECTATION_ATOL",
    "DEFAULT_IMAGINARY_ATOL",
    "DEFAULT_VARIANCE_ATOL",
    "DEFAULT_GROUND_ENERGY_ATOL",
    "DEFAULT_BASIS_CHUNK_SIZE",
    "OBSERVABLE_ENGINE",
    "OPERATOR_CLASS",
    "QAObservablesError",
    "QAObservablesConfig",
    "QACostOperatorArtifact",
    "QAObservablesResult",
    "QAGPUEvaluation",
    "build_qa_qiskit_cost_operator",
    "evaluate_qa_statevector_observables",
    "evaluate_qa_circuit_gpu",
    "evaluate_digitized_qa_gpu",
]
