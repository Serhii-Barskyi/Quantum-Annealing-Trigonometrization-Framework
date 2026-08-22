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

"""Production orchestration for exact digitized-QA on Qiskit Aer GPU.

The solver composes the complete fixed project direction::

    QUBO -> Ising audit -> QA schedule -> digitized circuit
    -> Qiskit Aer GPU statevector -> exact observables
    -> digitized-QA surrogate targets.

A second product-formula resolution is executed for the mandatory Trotter
convergence audit.  Generic QUBOs must provide explicit feasibility semantics
when the configured surrogate targets request feasibility probability.  A
BESS-placement QUBO receives an exact cardinality mask automatically.

No custom quantum simulator, CPU fallback, direct optimizer, filesystem,
network, emulator, or QPU operation is implemented here.  Qiskit and Aer stay
lazy in the delegated runtime modules, and every exact statevector path is
rejected above the project-wide 22-qubit limit before runtime import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config.schema import QAConfig, SurrogateTarget
from opf.bess_constraints import BESSPlacement
from qa import (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    MAXIMUM_TROTTER_SLICES,
    MINIMUM_TROTTER_SLICES,
    QA_ALGORITHM_NAME,
    QA_TO_MAQAOA_DIRECTION,
    validate_exact_statevector_qubit_count,
)
from qa.aer_gpu import QAAerGPUConfig
from qa.circuit import (
    QACircuitArtifact,
    QACircuitConfig,
    build_digitized_qa_circuit,
)
from qa.observables import (
    QACostOperatorArtifact,
    QAGPUEvaluation,
    QAObservablesConfig,
    build_qa_qiskit_cost_operator,
    evaluate_digitized_qa_gpu,
)
from qa.schedule import (
    AnnealingSchedule,
    DigitizedQASchedule,
    build_linear_forward_schedule,
)
from qa.surrogate import (
    DEFAULT_BASIS_CHUNK_SIZE,
    DEFAULT_ENERGY_ATOL,
    DEFAULT_LOWER_TAIL_PROBABILITY,
    QASurrogateObservation,
    build_qa_surrogate_observation,
    targets_from_config,
)
from qaoa.hamiltonian import (
    IsingHamiltonian,
    QUBOIsingAudit,
    qubo_to_ising,
    require_qubo_ising_equivalence,
)
from qubo.builder import BESSPlacementQUBO
from qubo.model import QUBOModel


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
INTEGER_DTYPE: Final[np.dtype[np.int8]] = np.dtype(np.int8)
BOOLEAN_DTYPE: Final[np.dtype[np.bool_]] = np.dtype(np.bool_)
DEFAULT_AUDIT_EXACT_LIMIT: Final[int] = 18
DEFAULT_AUDIT_RANDOM_SAMPLES: Final[int] = 4096
DEFAULT_AUDIT_SEED: Final[int] = 0
DEFAULT_AUDIT_TOLERANCE: Final[float] = 1.0e-10
DEFAULT_FINAL_ENERGY_TOLERANCE: Final[float] = 1.0e-10
DEFAULT_SURROGATE_GROUND_ENERGY_ATOL: Final[float] = 1.0e-9
DEFAULT_CONVERGENCE_TOLERANCE: Final[float] = 1.0e-3
DEFAULT_MASK_CHUNK_SIZE: Final[int] = 65_536

SourceKind = Literal["qubo_model", "bess_placement_qubo"]


class QASolverError(RuntimeError):
    """Raised when an end-to-end digitized-QA solve contract is violated."""


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise QASolverError(f"{name} must be non-negative.")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    normalized = _nonnegative_integer(value, name=name)
    if normalized == 0:
        raise QASolverError(f"{name} must be positive.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise QASolverError(
            f"{name} must be finite and strictly positive."
        )
    return normalized


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise QASolverError(f"{name} must be finite.")
    return normalized


def _probability(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if not 0.0 < normalized <= 1.0:
        raise QASolverError(f"{name} must lie in (0, 1].")
    return normalized


def _sha256_digest(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise QASolverError(f"{name} must be a SHA-256 digest.")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise QASolverError(
            f"{name} must be a hexadecimal SHA-256 digest."
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
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise QASolverError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    frozen = _freeze_json(json.loads(encoded))
    if not isinstance(frozen, Mapping):
        raise QASolverError("metadata normalization failed.")
    return frozen


def _readonly_binary_sample(
    values: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.int8]:
    source = np.asarray(values)
    if source.ndim != 1 or source.size != expected_size:
        raise QASolverError(
            f"{name} must be one-dimensional with {expected_size} values."
        )
    numeric = np.asarray(source, dtype=REAL_DTYPE)
    if not np.all(np.isfinite(numeric)):
        raise QASolverError(f"{name} contains non-finite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise QASolverError(f"{name} must contain only binary values.")
    result = np.array(numeric, dtype=INTEGER_DTYPE, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_bool_mask(
    values: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.bool_]:
    source = np.asarray(values)
    if source.ndim != 1 or source.size != expected_size:
        raise QASolverError(
            f"{name} must be one-dimensional with {expected_size} entries."
        )
    if source.dtype.kind not in {"b", "i", "u", "f"}:
        raise QASolverError(f"{name} must contain boolean values.")
    numeric = np.asarray(source, dtype=REAL_DTYPE)
    if not np.all(np.isfinite(numeric)):
        raise QASolverError(f"{name} contains non-finite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise QASolverError(f"{name} must contain only 0/1 values.")
    result = np.array(numeric != 0.0, dtype=BOOLEAN_DTYPE, order="C")
    result.setflags(write=False)
    return result


def _qa_config_payload(config: QAConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise QASolverError("QAConfig serialization did not return an object.")
    return payload


def _default_reference_slices(primary_slices: int) -> int:
    primary = _positive_integer(primary_slices, name="primary_slices")
    if primary > MINIMUM_TROTTER_SLICES:
        reference = max(MINIMUM_TROTTER_SLICES, primary // 2)
    else:
        reference = min(MAXIMUM_TROTTER_SLICES, primary * 2)
    if reference == primary:
        raise QASolverError(
            "Cannot derive a distinct Trotter convergence resolution."
        )
    return reference


def _bess_feasibility_mask(
    placement_qubo: BESSPlacementQUBO,
    *,
    chunk_size: int,
) -> NDArray[np.bool_]:
    """Enumerate the exact cardinality mask in Qiskit little-endian order."""

    if not isinstance(placement_qubo, BESSPlacementQUBO):
        raise TypeError("placement_qubo must be BESSPlacementQUBO.")
    n_qubits = validate_exact_statevector_qubit_count(
        placement_qubo.model.n_variables
    )
    normalized_chunk = _positive_integer(chunk_size, name="chunk_size")
    expected_size = 1 << n_qubits
    required = int(placement_qubo.fleet.units_to_place)
    mask = np.empty(expected_size, dtype=BOOLEAN_DTYPE)

    for start in range(0, expected_size, normalized_chunk):
        stop = min(expected_size, start + normalized_chunk)
        indices = np.arange(start, stop, dtype=np.uint64)
        counts = np.zeros(stop - start, dtype=np.uint8)
        for qubit in range(n_qubits):
            counts += np.asarray(
                (indices >> np.uint64(qubit)) & np.uint64(1),
                dtype=np.uint8,
            )
        mask[start:stop] = counts == required

    mask.setflags(write=False)
    return mask


@dataclass(frozen=True, slots=True)
class QASolverConfig:
    """Validated configuration for one exact digitized-QA GPU solve."""

    qa: QAConfig = field(default_factory=QAConfig)
    circuit: QACircuitConfig = field(default_factory=QACircuitConfig)
    aer: QAAerGPUConfig = field(default_factory=QAAerGPUConfig)
    observables: QAObservablesConfig = field(
        default_factory=QAObservablesConfig
    )
    audit_exact_limit: int = DEFAULT_AUDIT_EXACT_LIMIT
    audit_random_samples: int = DEFAULT_AUDIT_RANDOM_SAMPLES
    audit_seed: int = DEFAULT_AUDIT_SEED
    audit_tolerance: float = DEFAULT_AUDIT_TOLERANCE
    final_energy_tolerance: float = DEFAULT_FINAL_ENERGY_TOLERANCE
    lower_tail_probability: float = DEFAULT_LOWER_TAIL_PROBABILITY
    ground_energy_atol: float = DEFAULT_SURROGATE_GROUND_ENERGY_ATOL
    surrogate_energy_atol: float = DEFAULT_ENERGY_ATOL
    surrogate_basis_chunk_size: int = DEFAULT_BASIS_CHUNK_SIZE
    mask_chunk_size: int = DEFAULT_MASK_CHUNK_SIZE
    convergence_reference_slices: int | None = None
    convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE
    require_converged: bool = False
    require_most_probable_feasible: bool = False
    require_representative_ground_feasible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.qa, QAConfig):
            raise TypeError("qa must be QAConfig.")
        if not isinstance(self.circuit, QACircuitConfig):
            raise TypeError("circuit must be QACircuitConfig.")
        if not isinstance(self.aer, QAAerGPUConfig):
            raise TypeError("aer must be QAAerGPUConfig.")
        if not isinstance(self.observables, QAObservablesConfig):
            raise TypeError("observables must be QAObservablesConfig.")
        if not self.qa.enabled:
            raise QASolverError("QAConfig.enabled must be true.")
        if self.qa.schedule_mapping != QA_TO_MAQAOA_DIRECTION:
            raise QASolverError(
                "Only the fixed QA-to-MA-QAOA schedule mapping is supported."
            )
        if self.aer.max_statevector_qubits > (
            COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
        ):
            raise QASolverError(
                "Aer configuration exceeds the project-wide 22-qubit limit."
            )

        exact_limit = _nonnegative_integer(
            self.audit_exact_limit,
            name="audit_exact_limit",
        )
        if exact_limit > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT:
            raise QASolverError(
                "audit_exact_limit exceeds the project-wide 22-qubit limit."
            )
        object.__setattr__(self, "audit_exact_limit", exact_limit)
        object.__setattr__(
            self,
            "audit_random_samples",
            _positive_integer(
                self.audit_random_samples,
                name="audit_random_samples",
            ),
        )
        object.__setattr__(
            self,
            "audit_seed",
            _nonnegative_integer(self.audit_seed, name="audit_seed"),
        )
        object.__setattr__(
            self,
            "audit_tolerance",
            _positive_float(self.audit_tolerance, name="audit_tolerance"),
        )
        object.__setattr__(
            self,
            "final_energy_tolerance",
            _positive_float(
                self.final_energy_tolerance,
                name="final_energy_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "lower_tail_probability",
            _probability(
                self.lower_tail_probability,
                name="lower_tail_probability",
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
            "surrogate_energy_atol",
            _positive_float(
                self.surrogate_energy_atol,
                name="surrogate_energy_atol",
            ),
        )
        object.__setattr__(
            self,
            "surrogate_basis_chunk_size",
            _positive_integer(
                self.surrogate_basis_chunk_size,
                name="surrogate_basis_chunk_size",
            ),
        )
        object.__setattr__(
            self,
            "mask_chunk_size",
            _positive_integer(self.mask_chunk_size, name="mask_chunk_size"),
        )
        object.__setattr__(
            self,
            "convergence_tolerance",
            _positive_float(
                self.convergence_tolerance,
                name="convergence_tolerance",
            ),
        )

        reference = self.convergence_reference_slices
        if reference is not None:
            reference = _positive_integer(
                reference,
                name="convergence_reference_slices",
            )
            if not MINIMUM_TROTTER_SLICES <= reference <= (
                MAXIMUM_TROTTER_SLICES
            ):
                raise QASolverError(
                    "convergence_reference_slices is outside the supported "
                    "Trotter range."
                )
            if reference == int(self.qa.trotter_slices):
                raise QASolverError(
                    "convergence_reference_slices must differ from the "
                    "primary Trotter resolution."
                )
        object.__setattr__(self, "convergence_reference_slices", reference)

        for name in (
            "require_converged",
            "require_most_probable_feasible",
            "require_representative_ground_feasible",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")

    @property
    def reference_slices(self) -> int:
        if self.convergence_reference_slices is not None:
            return self.convergence_reference_slices
        return _default_reference_slices(int(self.qa.trotter_slices))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QASolverConfig-v1\0")
        digest.update(
            json.dumps(
                _qa_config_payload(self.qa),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(self.circuit.fingerprint().encode("ascii"))
        digest.update(self.aer.fingerprint().encode("ascii"))
        digest.update(self.observables.fingerprint().encode("ascii"))
        digest.update(
            json.dumps(
                {
                    "audit_exact_limit": self.audit_exact_limit,
                    "audit_random_samples": self.audit_random_samples,
                    "audit_seed": self.audit_seed,
                    "audit_tolerance": self.audit_tolerance,
                    "final_energy_tolerance": self.final_energy_tolerance,
                    "lower_tail_probability": self.lower_tail_probability,
                    "ground_energy_atol": self.ground_energy_atol,
                    "surrogate_energy_atol": self.surrogate_energy_atol,
                    "surrogate_basis_chunk_size": (
                        self.surrogate_basis_chunk_size
                    ),
                    "mask_chunk_size": self.mask_chunk_size,
                    "reference_slices": self.reference_slices,
                    "convergence_tolerance": self.convergence_tolerance,
                    "require_converged": self.require_converged,
                    "require_most_probable_feasible": (
                        self.require_most_probable_feasible
                    ),
                    "require_representative_ground_feasible": (
                        self.require_representative_ground_feasible
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QATrotterConvergenceAudit:
    """Comparison between primary and reference digitization resolutions."""

    primary_slices: int
    reference_slices: int
    primary_schedule_fingerprint: str
    reference_schedule_fingerprint: str
    primary_evaluation_fingerprint: str
    reference_evaluation_fingerprint: str
    primary_observation_fingerprint: str
    reference_observation_fingerprint: str
    expected_energy_absolute_difference: float
    target_linf_difference: float
    tolerance: float
    within_tolerance: bool

    def __post_init__(self) -> None:
        primary = _positive_integer(self.primary_slices, name="primary_slices")
        reference = _positive_integer(
            self.reference_slices,
            name="reference_slices",
        )
        if primary == reference:
            raise QASolverError(
                "Convergence audit requires distinct Trotter resolutions."
            )
        object.__setattr__(self, "primary_slices", primary)
        object.__setattr__(self, "reference_slices", reference)
        for name in (
            "primary_schedule_fingerprint",
            "reference_schedule_fingerprint",
            "primary_evaluation_fingerprint",
            "reference_evaluation_fingerprint",
            "primary_observation_fingerprint",
            "reference_observation_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _sha256_digest(getattr(self, name), name=name),
            )
        energy_difference = _finite_float(
            self.expected_energy_absolute_difference,
            name="expected_energy_absolute_difference",
        )
        target_difference = _finite_float(
            self.target_linf_difference,
            name="target_linf_difference",
        )
        if energy_difference < 0.0 or target_difference < 0.0:
            raise QASolverError(
                "Convergence differences must be non-negative."
            )
        tolerance = _positive_float(self.tolerance, name="tolerance")
        expected_within = max(energy_difference, target_difference) <= tolerance
        if not isinstance(self.within_tolerance, bool):
            raise TypeError("within_tolerance must be boolean.")
        if self.within_tolerance != expected_within:
            raise QASolverError(
                "within_tolerance disagrees with convergence differences."
            )
        object.__setattr__(
            self,
            "expected_energy_absolute_difference",
            energy_difference,
        )
        object.__setattr__(self, "target_linf_difference", target_difference)
        object.__setattr__(self, "tolerance", tolerance)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QATrotterConvergenceAudit-v1\0")
        digest.update(
            json.dumps(
                self.manifest(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "primary_slices": self.primary_slices,
            "reference_slices": self.reference_slices,
            "primary_schedule_fingerprint": (
                self.primary_schedule_fingerprint
            ),
            "reference_schedule_fingerprint": (
                self.reference_schedule_fingerprint
            ),
            "primary_evaluation_fingerprint": (
                self.primary_evaluation_fingerprint
            ),
            "reference_evaluation_fingerprint": (
                self.reference_evaluation_fingerprint
            ),
            "primary_observation_fingerprint": (
                self.primary_observation_fingerprint
            ),
            "reference_observation_fingerprint": (
                self.reference_observation_fingerprint
            ),
            "expected_energy_absolute_difference": (
                self.expected_energy_absolute_difference
            ),
            "target_linf_difference": self.target_linf_difference,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True, slots=True)
class QASolution:
    """Immutable end-to-end digitized-QA solution and audit record."""

    source_kind: SourceKind
    qubo_model: QUBOModel
    placement_qubo: BESSPlacementQUBO | None
    hamiltonian: IsingHamiltonian
    equivalence_audit: QUBOIsingAudit
    source_schedule: AnnealingSchedule
    digitized_schedule: DigitizedQASchedule
    circuit_artifact: QACircuitArtifact
    operator_artifact: QACostOperatorArtifact
    evaluation: QAGPUEvaluation
    surrogate_observation: QASurrogateObservation
    convergence_audit: QATrotterConvergenceAudit
    most_probable_sample: NDArray[np.int8]
    representative_ground_sample: NDArray[np.int8]
    most_probable_energy: float
    representative_ground_energy: float
    expected_energy: float
    energy_variance: float
    ground_probability: float
    most_probable_probability: float
    most_probable_is_feasible: bool | None
    representative_ground_is_feasible: bool | None
    most_probable_placement: BESSPlacement | None
    representative_ground_placement: BESSPlacement | None
    config_fingerprint: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.source_kind not in ("qubo_model", "bess_placement_qubo"):
            raise QASolverError("source_kind is unsupported.")
        if not isinstance(self.qubo_model, QUBOModel):
            raise TypeError("qubo_model must be QUBOModel.")
        if not isinstance(self.hamiltonian, IsingHamiltonian):
            raise TypeError("hamiltonian must be IsingHamiltonian.")
        if not isinstance(self.equivalence_audit, QUBOIsingAudit):
            raise TypeError("equivalence_audit must be QUBOIsingAudit.")
        if not isinstance(self.source_schedule, AnnealingSchedule):
            raise TypeError("source_schedule must be AnnealingSchedule.")
        if not isinstance(self.digitized_schedule, DigitizedQASchedule):
            raise TypeError("digitized_schedule must be DigitizedQASchedule.")
        if not isinstance(self.circuit_artifact, QACircuitArtifact):
            raise TypeError("circuit_artifact must be QACircuitArtifact.")
        if not isinstance(self.operator_artifact, QACostOperatorArtifact):
            raise TypeError("operator_artifact must be QACostOperatorArtifact.")
        if not isinstance(self.evaluation, QAGPUEvaluation):
            raise TypeError("evaluation must be QAGPUEvaluation.")
        if not isinstance(self.surrogate_observation, QASurrogateObservation):
            raise TypeError(
                "surrogate_observation must be QASurrogateObservation."
            )
        if not isinstance(self.convergence_audit, QATrotterConvergenceAudit):
            raise TypeError(
                "convergence_audit must be QATrotterConvergenceAudit."
            )

        if self.source_kind == "bess_placement_qubo":
            if not isinstance(self.placement_qubo, BESSPlacementQUBO):
                raise TypeError(
                    "placement_qubo must be BESSPlacementQUBO for BESS solves."
                )
            if self.placement_qubo.model.fingerprint() != (
                self.qubo_model.fingerprint()
            ):
                raise QASolverError(
                    "placement_qubo and qubo_model fingerprints differ."
                )
        elif self.placement_qubo is not None:
            raise QASolverError(
                "Generic QUBO solutions must not contain placement_qubo."
            )

        model_fingerprint = self.qubo_model.fingerprint()
        hamiltonian_fingerprint = self.hamiltonian.fingerprint()
        if self.hamiltonian.variable_order != self.qubo_model.variable_order:
            raise QASolverError(
                "QUBO and Ising variable orders differ."
            )
        if self.equivalence_audit.qubo_fingerprint != model_fingerprint:
            raise QASolverError("Equivalence audit belongs to another QUBO.")
        if self.equivalence_audit.ising_fingerprint != hamiltonian_fingerprint:
            raise QASolverError(
                "Equivalence audit belongs to another Ising Hamiltonian."
            )
        if not self.equivalence_audit.equivalent:
            raise QASolverError("QUBO-to-Ising equivalence audit failed.")

        if self.digitized_schedule.source_schedule_fingerprint != (
            self.source_schedule.fingerprint()
        ):
            raise QASolverError(
                "Digitized schedule belongs to another source schedule."
            )
        if self.circuit_artifact.plan.schedule.fingerprint() != (
            self.digitized_schedule.fingerprint()
        ):
            raise QASolverError(
                "Circuit plan belongs to another digitized schedule."
            )
        if self.circuit_artifact.plan.hamiltonian.fingerprint() != (
            hamiltonian_fingerprint
        ):
            raise QASolverError("Circuit belongs to another Hamiltonian.")
        if self.operator_artifact.hamiltonian_fingerprint != (
            hamiltonian_fingerprint
        ):
            raise QASolverError("Operator belongs to another Hamiltonian.")
        if self.evaluation.circuit_artifact.fingerprint() != (
            self.circuit_artifact.fingerprint()
        ):
            raise QASolverError("Evaluation belongs to another circuit.")
        if self.evaluation.statevector_result.hamiltonian_fingerprint != (
            hamiltonian_fingerprint
        ):
            raise QASolverError("Statevector belongs to another Hamiltonian.")
        if self.evaluation.observables_result.operator_fingerprint != (
            self.operator_artifact.fingerprint()
        ):
            raise QASolverError("Observable result belongs to another operator.")
        if self.surrogate_observation.evaluation_fingerprint != (
            self.evaluation.fingerprint()
        ):
            raise QASolverError(
                "Surrogate observation belongs to another evaluation."
            )
        if self.surrogate_observation.hamiltonian_fingerprint != (
            hamiltonian_fingerprint
        ):
            raise QASolverError(
                "Surrogate observation belongs to another Hamiltonian."
            )

        n_variables = self.qubo_model.n_variables
        most_sample = _readonly_binary_sample(
            self.most_probable_sample,
            expected_size=n_variables,
            name="most_probable_sample",
        )
        ground_sample = _readonly_binary_sample(
            self.representative_ground_sample,
            expected_size=n_variables,
            name="representative_ground_sample",
        )
        object.__setattr__(self, "most_probable_sample", most_sample)
        object.__setattr__(
            self,
            "representative_ground_sample",
            ground_sample,
        )

        most_energy = _finite_float(
            self.most_probable_energy,
            name="most_probable_energy",
        )
        ground_energy = _finite_float(
            self.representative_ground_energy,
            name="representative_ground_energy",
        )
        expected_energy = _finite_float(
            self.expected_energy,
            name="expected_energy",
        )
        variance = _finite_float(
            self.energy_variance,
            name="energy_variance",
        )
        if variance < 0.0:
            raise QASolverError("energy_variance must be non-negative.")
        ground_probability = _finite_float(
            self.ground_probability,
            name="ground_probability",
        )
        most_probability = _finite_float(
            self.most_probable_probability,
            name="most_probable_probability",
        )
        if not 0.0 <= ground_probability <= 1.0:
            raise QASolverError("ground_probability must lie in [0, 1].")
        if not 0.0 <= most_probability <= 1.0:
            raise QASolverError(
                "most_probable_probability must lie in [0, 1]."
            )

        tolerance = max(self.qubo_model.zero_tolerance, 1.0e-10)
        if not math.isclose(
            self.qubo_model.energy(most_sample),
            most_energy,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise QASolverError(
                "Most-probable sample energy disagrees with QUBO."
            )
        if not math.isclose(
            self.qubo_model.energy(ground_sample),
            ground_energy,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise QASolverError(
                "Representative ground energy disagrees with QUBO."
            )
        if not math.isclose(
            expected_energy,
            self.evaluation.observables_result.expected_energy,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise QASolverError(
                "Solution expected energy disagrees with QA observables."
            )
        if not math.isclose(
            variance,
            self.evaluation.observables_result.variance,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise QASolverError(
                "Solution variance disagrees with QA observables."
            )

        for name in (
            "most_probable_is_feasible",
            "representative_ground_is_feasible",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None.")

        object.__setattr__(self, "most_probable_energy", most_energy)
        object.__setattr__(
            self,
            "representative_ground_energy",
            ground_energy,
        )
        object.__setattr__(self, "expected_energy", expected_energy)
        object.__setattr__(self, "energy_variance", variance)
        object.__setattr__(self, "ground_probability", ground_probability)
        object.__setattr__(
            self,
            "most_probable_probability",
            most_probability,
        )
        object.__setattr__(
            self,
            "config_fingerprint",
            _sha256_digest(
                self.config_fingerprint,
                name="config_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_json_mapping(self.metadata),
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QASolution-v1\0")
        digest.update(self.source_kind.encode("ascii"))
        digest.update(self.qubo_model.fingerprint().encode("ascii"))
        digest.update(self.hamiltonian.fingerprint().encode("ascii"))
        digest.update(self.source_schedule.fingerprint().encode("ascii"))
        digest.update(self.digitized_schedule.fingerprint().encode("ascii"))
        digest.update(self.circuit_artifact.fingerprint().encode("ascii"))
        digest.update(self.operator_artifact.fingerprint().encode("ascii"))
        digest.update(self.evaluation.fingerprint().encode("ascii"))
        digest.update(
            self.surrogate_observation.fingerprint().encode("ascii")
        )
        digest.update(self.convergence_audit.fingerprint().encode("ascii"))
        digest.update(self.most_probable_sample.tobytes(order="C"))
        digest.update(
            self.representative_ground_sample.tobytes(order="C")
        )
        digest.update(self.config_fingerprint.encode("ascii"))
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
        most_selected = (
            None
            if self.most_probable_placement is None
            else list(self.most_probable_placement.selected_buses)
        )
        ground_selected = (
            None
            if self.representative_ground_placement is None
            else list(self.representative_ground_placement.selected_buses)
        )
        return {
            "schema": "cssf-digitized-qa-solution-v1",
            "algorithm": QA_ALGORITHM_NAME,
            "source_kind": self.source_kind,
            "n_variables": self.qubo_model.n_variables,
            "expected_energy": self.expected_energy,
            "energy_variance": self.energy_variance,
            "ground_probability": self.ground_probability,
            "most_probable_probability": self.most_probable_probability,
            "most_probable_energy": self.most_probable_energy,
            "representative_ground_energy": (
                self.representative_ground_energy
            ),
            "most_probable_sample": self.most_probable_sample.tolist(),
            "representative_ground_sample": (
                self.representative_ground_sample.tolist()
            ),
            "most_probable_is_feasible": self.most_probable_is_feasible,
            "representative_ground_is_feasible": (
                self.representative_ground_is_feasible
            ),
            "most_probable_selected_buses": most_selected,
            "representative_ground_selected_buses": ground_selected,
            "qubo_fingerprint": self.qubo_model.fingerprint(),
            "hamiltonian_fingerprint": self.hamiltonian.fingerprint(),
            "schedule_fingerprint": self.source_schedule.fingerprint(),
            "digitized_schedule_fingerprint": (
                self.digitized_schedule.fingerprint()
            ),
            "circuit_fingerprint": self.circuit_artifact.fingerprint(),
            "operator_fingerprint": self.operator_artifact.fingerprint(),
            "evaluation_fingerprint": self.evaluation.fingerprint(),
            "observation_fingerprint": (
                self.surrogate_observation.fingerprint()
            ),
            "convergence": self.convergence_audit.manifest(),
            "config_fingerprint": self.config_fingerprint,
            "solution_fingerprint": self.fingerprint(),
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class _QALevelExecution:
    digitized_schedule: DigitizedQASchedule
    circuit_artifact: QACircuitArtifact
    evaluation: QAGPUEvaluation
    observation: QASurrogateObservation


def _execute_level(
    hamiltonian: IsingHamiltonian,
    schedule: AnnealingSchedule,
    *,
    trotter_slices: int,
    trotter_order: int,
    config: QASolverConfig,
    operator_artifact: QACostOperatorArtifact,
    feasible_mask: NDArray[np.bool_] | None,
    success_mask: NDArray[np.bool_] | None,
    elite_energy_threshold: float | None,
    metadata: Mapping[str, Any] | None,
) -> _QALevelExecution:
    digitized = schedule.digitize(
        trotter_slices=trotter_slices,
        trotter_order=trotter_order,
    )
    circuit_artifact = build_digitized_qa_circuit(
        hamiltonian,
        digitized,
        config.circuit,
    )
    evaluation = evaluate_digitized_qa_gpu(
        circuit_artifact,
        hamiltonian,
        aer_config=config.aer,
        observables_config=config.observables,
        operator_artifact=operator_artifact,
    )
    observation = build_qa_surrogate_observation(
        evaluation,
        hamiltonian,
        targets=targets_from_config(config.qa),
        feasible_mask=feasible_mask,
        success_mask=success_mask,
        elite_energy_threshold=elite_energy_threshold,
        lower_tail_probability=config.lower_tail_probability,
        ground_energy_atol=config.ground_energy_atol,
        energy_atol=config.surrogate_energy_atol,
        basis_chunk_size=config.surrogate_basis_chunk_size,
        metadata={
            **({} if metadata is None else dict(metadata)),
            "trotter_slices": trotter_slices,
            "trotter_order": trotter_order,
        },
    )
    return _QALevelExecution(
        digitized_schedule=digitized,
        circuit_artifact=circuit_artifact,
        evaluation=evaluation,
        observation=observation,
    )


def _placement_decode(
    placement_qubo: BESSPlacementQUBO | None,
    sample: NDArray[np.int8],
) -> tuple[bool | None, BESSPlacement | None]:
    if placement_qubo is None:
        return None, None
    feasible = placement_qubo.is_feasible(sample)
    placement = placement_qubo.decode(sample) if feasible else None
    return feasible, placement


def _normalize_optional_masks(
    model: QUBOModel,
    *,
    feasible_mask: ArrayLike | None,
    success_mask: ArrayLike | None,
) -> tuple[NDArray[np.bool_] | None, NDArray[np.bool_] | None]:
    expected_size = 1 << model.n_variables
    normalized_feasible = (
        None
        if feasible_mask is None
        else _readonly_bool_mask(
            feasible_mask,
            expected_size=expected_size,
            name="feasible_mask",
        )
    )
    normalized_success = (
        None
        if success_mask is None
        else _readonly_bool_mask(
            success_mask,
            expected_size=expected_size,
            name="success_mask",
        )
    )
    return normalized_feasible, normalized_success


def _solve_qa_gpu(
    model: QUBOModel,
    *,
    source_kind: SourceKind,
    placement_qubo: BESSPlacementQUBO | None,
    config: QASolverConfig,
    schedule: AnnealingSchedule | None,
    feasible_mask: ArrayLike | None,
    success_mask: ArrayLike | None,
    elite_energy_threshold: float | None,
    metadata: Mapping[str, Any] | None,
) -> QASolution:
    validate_exact_statevector_qubit_count(model.n_variables)
    if model.n_variables > config.aer.max_statevector_qubits:
        raise QASolverError(
            f"QUBO has {model.n_variables} variables but Aer config allows "
            f"only {config.aer.max_statevector_qubits}."
        )

    hamiltonian = qubo_to_ising(model)
    equivalence_audit = require_qubo_ising_equivalence(
        model,
        hamiltonian,
        exact_limit=config.audit_exact_limit,
        random_samples=config.audit_random_samples,
        seed=config.audit_seed,
        tolerance=config.audit_tolerance,
    )

    source_schedule = (
        build_linear_forward_schedule(
            config.qa,
            metadata={
                "framework": "CSSF",
                "algorithm": QA_ALGORITHM_NAME,
                "source_kind": source_kind,
            },
        )
        if schedule is None
        else schedule
    )
    if not isinstance(source_schedule, AnnealingSchedule):
        raise TypeError("schedule must be AnnealingSchedule or None.")
    if not math.isclose(
        source_schedule.total_annealing_time,
        float(config.qa.total_annealing_time),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise QASolverError(
            "Schedule duration differs from QAConfig.total_annealing_time."
        )

    normalized_feasible, normalized_success = _normalize_optional_masks(
        model,
        feasible_mask=feasible_mask,
        success_mask=success_mask,
    )
    requested_targets = targets_from_config(config.qa)
    if (
        SurrogateTarget.FEASIBILITY_PROBABILITY in requested_targets
        and normalized_feasible is None
    ):
        raise QASolverError(
            "feasible_mask is required when QA targets include "
            "feasibility_probability."
        )

    operator_artifact = build_qa_qiskit_cost_operator(hamiltonian)
    primary = _execute_level(
        hamiltonian,
        source_schedule,
        trotter_slices=int(config.qa.trotter_slices),
        trotter_order=int(config.qa.trotter_order),
        config=config,
        operator_artifact=operator_artifact,
        feasible_mask=normalized_feasible,
        success_mask=normalized_success,
        elite_energy_threshold=elite_energy_threshold,
        metadata=metadata,
    )
    reference = _execute_level(
        hamiltonian,
        source_schedule,
        trotter_slices=config.reference_slices,
        trotter_order=int(config.qa.trotter_order),
        config=config,
        operator_artifact=operator_artifact,
        feasible_mask=normalized_feasible,
        success_mask=normalized_success,
        elite_energy_threshold=elite_energy_threshold,
        metadata={
            **({} if metadata is None else dict(metadata)),
            "convergence_reference": True,
        },
    )

    expected_difference = abs(
        primary.evaluation.observables_result.expected_energy
        - reference.evaluation.observables_result.expected_energy
    )
    target_difference = float(
        np.max(np.abs(primary.observation.values - reference.observation.values))
    )
    within_tolerance = max(expected_difference, target_difference) <= (
        config.convergence_tolerance
    )
    convergence_audit = QATrotterConvergenceAudit(
        primary_slices=primary.digitized_schedule.slice_count,
        reference_slices=reference.digitized_schedule.slice_count,
        primary_schedule_fingerprint=(
            primary.digitized_schedule.fingerprint()
        ),
        reference_schedule_fingerprint=(
            reference.digitized_schedule.fingerprint()
        ),
        primary_evaluation_fingerprint=primary.evaluation.fingerprint(),
        reference_evaluation_fingerprint=reference.evaluation.fingerprint(),
        primary_observation_fingerprint=primary.observation.fingerprint(),
        reference_observation_fingerprint=reference.observation.fingerprint(),
        expected_energy_absolute_difference=expected_difference,
        target_linf_difference=target_difference,
        tolerance=config.convergence_tolerance,
        within_tolerance=within_tolerance,
    )
    if config.require_converged and not convergence_audit.within_tolerance:
        raise QASolverError(
            "Digitized-QA Trotter convergence audit exceeded tolerance."
        )

    observables = primary.evaluation.observables_result
    most_sample = _readonly_binary_sample(
        observables.most_probable_sample,
        expected_size=model.n_variables,
        name="most_probable_sample",
    )
    ground_sample = _readonly_binary_sample(
        observables.representative_ground_sample,
        expected_size=model.n_variables,
        name="representative_ground_sample",
    )
    most_energy = model.energy(most_sample)
    ground_energy = model.energy(ground_sample)

    if not math.isclose(
        most_energy,
        observables.most_probable_energy,
        rel_tol=0.0,
        abs_tol=config.final_energy_tolerance,
    ):
        raise QASolverError(
            "Most-probable QUBO energy disagrees with QA observables."
        )
    if not math.isclose(
        ground_energy,
        observables.minimum_energy,
        rel_tol=0.0,
        abs_tol=config.final_energy_tolerance,
    ):
        raise QASolverError(
            "Representative ground energy disagrees with QUBO."
        )

    most_feasible, most_placement = _placement_decode(
        placement_qubo,
        most_sample,
    )
    ground_feasible, ground_placement = _placement_decode(
        placement_qubo,
        ground_sample,
    )
    if (
        source_kind == "bess_placement_qubo"
        and config.require_most_probable_feasible
        and most_feasible is not True
    ):
        raise QASolverError(
            "Most-probable digitized-QA sample violates BESS cardinality."
        )
    if (
        source_kind == "bess_placement_qubo"
        and config.require_representative_ground_feasible
        and ground_feasible is not True
    ):
        raise QASolverError(
            "Representative ground state violates BESS cardinality; "
            "inspect the QUBO penalty construction."
        )

    result_metadata = {
        **({} if metadata is None else dict(metadata)),
        "framework": "CSSF",
        "algorithm": QA_ALGORITHM_NAME,
        "source_kind": source_kind,
        "execution_engine": "qiskit_aer.AerSimulator",
        "execution_method": "statevector",
        "execution_device": "GPU",
        "execution_precision": "double",
        "cpu_fallback": False,
        "statevector_qubit_limit": COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
        "solver_config_fingerprint": config.fingerprint(),
        "equivalence_audit_exhaustive": equivalence_audit.exhaustive,
        "equivalence_audit_samples": equivalence_audit.checked_samples,
        "primary_trotter_slices": primary.digitized_schedule.slice_count,
        "reference_trotter_slices": reference.digitized_schedule.slice_count,
        "convergence_within_tolerance": (
            convergence_audit.within_tolerance
        ),
    }

    return QASolution(
        source_kind=source_kind,
        qubo_model=model,
        placement_qubo=placement_qubo,
        hamiltonian=hamiltonian,
        equivalence_audit=equivalence_audit,
        source_schedule=source_schedule,
        digitized_schedule=primary.digitized_schedule,
        circuit_artifact=primary.circuit_artifact,
        operator_artifact=operator_artifact,
        evaluation=primary.evaluation,
        surrogate_observation=primary.observation,
        convergence_audit=convergence_audit,
        most_probable_sample=most_sample,
        representative_ground_sample=ground_sample,
        most_probable_energy=most_energy,
        representative_ground_energy=ground_energy,
        expected_energy=observables.expected_energy,
        energy_variance=observables.variance,
        ground_probability=observables.ground_probability,
        most_probable_probability=observables.most_probable_probability,
        most_probable_is_feasible=most_feasible,
        representative_ground_is_feasible=ground_feasible,
        most_probable_placement=most_placement,
        representative_ground_placement=ground_placement,
        config_fingerprint=config.fingerprint(),
        metadata=result_metadata,
    )


def solve_qubo_qa_gpu(
    model: QUBOModel,
    *,
    config: QASolverConfig | None = None,
    schedule: AnnealingSchedule | None = None,
    feasible_mask: ArrayLike | None = None,
    success_mask: ArrayLike | None = None,
    elite_energy_threshold: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QASolution:
    """Solve one generic QUBO with exact digitized-QA on Qiskit Aer GPU."""

    if not isinstance(model, QUBOModel):
        raise TypeError("model must be QUBOModel.")
    run_config = QASolverConfig() if config is None else config
    if not isinstance(run_config, QASolverConfig):
        raise TypeError("config must be QASolverConfig or None.")
    return _solve_qa_gpu(
        model,
        source_kind="qubo_model",
        placement_qubo=None,
        config=run_config,
        schedule=schedule,
        feasible_mask=feasible_mask,
        success_mask=success_mask,
        elite_energy_threshold=elite_energy_threshold,
        metadata=metadata,
    )


def solve_bess_placement_qa_gpu(
    placement_qubo: BESSPlacementQUBO,
    *,
    config: QASolverConfig | None = None,
    schedule: AnnealingSchedule | None = None,
    success_mask: ArrayLike | None = None,
    elite_energy_threshold: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QASolution:
    """Solve and decode one BESS-placement QUBO through digitized QA."""

    if not isinstance(placement_qubo, BESSPlacementQUBO):
        raise TypeError("placement_qubo must be BESSPlacementQUBO.")
    run_config = QASolverConfig() if config is None else config
    if not isinstance(run_config, QASolverConfig):
        raise TypeError("config must be QASolverConfig or None.")

    validate_exact_statevector_qubit_count(
        placement_qubo.model.n_variables
    )
    feasibility_mask = _bess_feasibility_mask(
        placement_qubo,
        chunk_size=run_config.mask_chunk_size,
    )
    return _solve_qa_gpu(
        placement_qubo.model,
        source_kind="bess_placement_qubo",
        placement_qubo=placement_qubo,
        config=run_config,
        schedule=schedule,
        feasible_mask=feasibility_mask,
        success_mask=success_mask,
        elite_energy_threshold=elite_energy_threshold,
        metadata=metadata,
    )


__all__ = [
    "REAL_DTYPE",
    "INTEGER_DTYPE",
    "BOOLEAN_DTYPE",
    "DEFAULT_AUDIT_EXACT_LIMIT",
    "DEFAULT_AUDIT_RANDOM_SAMPLES",
    "DEFAULT_AUDIT_SEED",
    "DEFAULT_AUDIT_TOLERANCE",
    "DEFAULT_FINAL_ENERGY_TOLERANCE",
    "DEFAULT_SURROGATE_GROUND_ENERGY_ATOL",
    "DEFAULT_CONVERGENCE_TOLERANCE",
    "DEFAULT_MASK_CHUNK_SIZE",
    "SourceKind",
    "QASolverError",
    "QASolverConfig",
    "QATrotterConvergenceAudit",
    "QASolution",
    "solve_qubo_qa_gpu",
    "solve_bess_placement_qa_gpu",
]
