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

"""Production orchestration for exact statevector QAOA on Google Colab.

This module is the public QAOA solve layer. It composes the solver-independent
QUBO model, the exact QUBO-to-Ising audit, Qiskit circuit/operator creation,
``qiskit-aer-gpu`` statevector evaluation, and Qiskit Algorithms classical
optimization into one reproducible result.

No custom quantum simulator and no direct SciPy optimizer are implemented.
Exact statevector execution is rejected above the project-wide 22-qubit limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from opf.bess_constraints import BESSPlacement
from qaoa.aer_gpu import (
    AerGPUConfig,
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
)
from qaoa.circuit import (
    QAOACircuitArtifact,
    QAOACircuitConfig,
    QAOAParameterValues,
    build_parameterized_qaoa_circuit,
)
from qaoa.expectation import (
    QAOAExpectationConfig,
    QAOAGPUEvaluation,
    QiskitCostOperatorArtifact,
    build_qiskit_cost_operator,
)
from qaoa.hamiltonian import (
    IsingHamiltonian,
    QUBOIsingAudit,
    qubo_to_ising,
    require_qubo_ising_equivalence,
)
from qaoa.objective import (
    QAOAGPUObjective,
    QAOAObjectiveConfig,
    QAOAObjectiveSnapshot,
)
from qaoa.optimizer import (
    QAOAOptimizationConfig,
    QAOAOptimizationResult,
    optimize_qaoa_gpu,
)
from qubo.builder import BESSPlacementQUBO
from qubo.model import QUBOModel


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
INTEGER_DTYPE: Final[np.dtype[np.int8]] = np.dtype(np.int8)
DEFAULT_AUDIT_EXACT_LIMIT: Final[int] = 18
DEFAULT_AUDIT_RANDOM_SAMPLES: Final[int] = 4096
DEFAULT_AUDIT_SEED: Final[int] = 0
DEFAULT_AUDIT_TOLERANCE: Final[float] = 1.0e-10
DEFAULT_FINAL_ENERGY_TOLERANCE: Final[float] = 1.0e-10

SourceKind = Literal["qubo_model", "bess_placement_qubo"]


class QAOASolverError(RuntimeError):
    """Raised when the end-to-end QAOA solve contract is violated."""


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise QAOASolverError(f"{name} must be non-negative.")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    normalized = _nonnegative_integer(value, name=name)
    if normalized == 0:
        raise QAOASolverError(f"{name} must be positive.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise QAOASolverError(
            f"{name} must be finite and strictly positive."
        )
    return normalized


def _readonly_binary_sample(
    values: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.int8]:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.ndim != 1 or array.size != expected_size:
        raise QAOASolverError(
            f"{name} must contain exactly {expected_size} binary values."
        )
    if not np.all(np.isfinite(array)):
        raise QAOASolverError(f"{name} contains non-finite values.")
    if not np.all((array == 0.0) | (array == 1.0)):
        raise QAOASolverError(f"{name} must be exactly binary.")
    result = np.array(array, dtype=INTEGER_DTYPE, copy=True)
    result.setflags(write=False)
    return result


def _json_mapping(
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
        raise QAOASolverError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _require_digest(value: str, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64:
        raise QAOASolverError(
            f"{name} must be a SHA-256 digest."
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise QAOASolverError(
            f"{name} must be hexadecimal."
        ) from exc
    return digest


def _evaluation_fingerprint(
    evaluation: QAOAGPUEvaluation,
) -> str:
    """Fingerprint an evaluation without depending on mutable APIs."""

    if not isinstance(evaluation, QAOAGPUEvaluation):
        raise TypeError(
            "evaluation must be QAOAGPUEvaluation."
        )
    digest = hashlib.sha256()
    digest.update(b"CSSF-QAOAGPUEvaluation-v1\0")
    digest.update(
        evaluation.parameters.flat().tobytes(order="C")
    )
    digest.update(
        evaluation.statevector_result.fingerprint().encode(
            "ascii"
        )
    )
    digest.update(
        evaluation.expectation_result.fingerprint().encode(
            "ascii"
        )
    )
    return digest.hexdigest()


def _circuit_config_fingerprint(
    config: QAOACircuitConfig,
) -> str:
    """Fingerprint the validated QAOA circuit configuration."""

    if not isinstance(config, QAOACircuitConfig):
        raise TypeError(
            "config must be QAOACircuitConfig."
        )
    digest = hashlib.sha256()
    digest.update(b"CSSF-QAOACircuitConfig-v1\0")
    digest.update(
        json.dumps(
            {
                "repetitions": config.repetitions,
                "barrier_policy": config.barrier_policy,
                "include_cost_offset_phase": (
                    config.include_cost_offset_phase
                ),
                "circuit_name": config.circuit_name,
                "gamma_prefix": config.gamma_prefix,
                "beta_prefix": config.beta_prefix,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAOASolverConfig:
    """Validated configuration for one exact Qiskit Aer GPU solve."""

    circuit: QAOACircuitConfig = field(
        default_factory=QAOACircuitConfig
    )
    aer: AerGPUConfig = field(default_factory=AerGPUConfig)
    expectation: QAOAExpectationConfig = field(
        default_factory=QAOAExpectationConfig
    )
    objective: QAOAObjectiveConfig = field(
        default_factory=lambda: QAOAObjectiveConfig(
            retain_best_evaluation=True
        )
    )
    optimization: QAOAOptimizationConfig = field(
        default_factory=QAOAOptimizationConfig
    )
    audit_exact_limit: int = DEFAULT_AUDIT_EXACT_LIMIT
    audit_random_samples: int = DEFAULT_AUDIT_RANDOM_SAMPLES
    audit_seed: int = DEFAULT_AUDIT_SEED
    audit_tolerance: float = DEFAULT_AUDIT_TOLERANCE
    final_energy_tolerance: float = (
        DEFAULT_FINAL_ENERGY_TOLERANCE
    )
    require_most_probable_feasible: bool = False
    require_representative_ground_feasible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, QAOACircuitConfig):
            raise TypeError("circuit must be QAOACircuitConfig.")
        if not isinstance(self.aer, AerGPUConfig):
            raise TypeError("aer must be AerGPUConfig.")
        if not isinstance(
            self.expectation,
            QAOAExpectationConfig,
        ):
            raise TypeError(
                "expectation must be QAOAExpectationConfig."
            )
        if not isinstance(self.objective, QAOAObjectiveConfig):
            raise TypeError(
                "objective must be QAOAObjectiveConfig."
            )
        if not isinstance(
            self.optimization,
            QAOAOptimizationConfig,
        ):
            raise TypeError(
                "optimization must be QAOAOptimizationConfig."
            )

        exact_limit = _nonnegative_integer(
            self.audit_exact_limit,
            name="audit_exact_limit",
        )
        if exact_limit > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT:
            raise QAOASolverError(
                "audit_exact_limit exceeds the project-wide "
                f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}-qubit limit."
            )

        random_samples = _positive_integer(
            self.audit_random_samples,
            name="audit_random_samples",
        )
        audit_seed = _nonnegative_integer(
            self.audit_seed,
            name="audit_seed",
        )

        if not isinstance(
            self.require_most_probable_feasible,
            bool,
        ):
            raise TypeError(
                "require_most_probable_feasible must be boolean."
            )
        if not isinstance(
            self.require_representative_ground_feasible,
            bool,
        ):
            raise TypeError(
                "require_representative_ground_feasible must be boolean."
            )

        object.__setattr__(self, "audit_exact_limit", exact_limit)
        object.__setattr__(
            self,
            "audit_random_samples",
            random_samples,
        )
        object.__setattr__(self, "audit_seed", audit_seed)
        object.__setattr__(
            self,
            "audit_tolerance",
            _positive_float(
                self.audit_tolerance,
                name="audit_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "final_energy_tolerance",
            _positive_float(
                self.final_energy_tolerance,
                name="final_energy_tolerance",
            ),
        )

        if (
            self.aer.max_statevector_qubits
            > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
        ):
            raise QAOASolverError(
                "Aer configuration exceeds the 22-qubit limit."
            )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOASolverConfig-v1\0")
        digest.update(_circuit_config_fingerprint(self.circuit).encode("ascii"))
        digest.update(self.aer.fingerprint().encode("ascii"))
        digest.update(
            self.expectation.fingerprint().encode("ascii")
        )
        digest.update(self.objective.fingerprint().encode("ascii"))
        digest.update(
            self.optimization.fingerprint().encode("ascii")
        )
        digest.update(
            json.dumps(
                {
                    "audit_exact_limit": self.audit_exact_limit,
                    "audit_random_samples": self.audit_random_samples,
                    "audit_seed": self.audit_seed,
                    "audit_tolerance": self.audit_tolerance,
                    "final_energy_tolerance": (
                        self.final_energy_tolerance
                    ),
                    "require_most_probable_feasible": (
                        self.require_most_probable_feasible
                    ),
                    "require_representative_ground_feasible": (
                        self.require_representative_ground_feasible
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAOASolution:
    """Immutable end-to-end QAOA solution and scientific audit record."""

    source_kind: SourceKind
    qubo_model: QUBOModel
    placement_qubo: BESSPlacementQUBO | None
    hamiltonian: IsingHamiltonian
    equivalence_audit: QUBOIsingAudit
    circuit_artifact: QAOACircuitArtifact
    operator_artifact: QiskitCostOperatorArtifact
    optimization_result: QAOAOptimizationResult
    final_evaluation: QAOAGPUEvaluation
    objective_snapshot: QAOAObjectiveSnapshot
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
        if self.source_kind not in (
            "qubo_model",
            "bess_placement_qubo",
        ):
            raise QAOASolverError("source_kind is unsupported.")
        if not isinstance(self.qubo_model, QUBOModel):
            raise TypeError("qubo_model must be QUBOModel.")
        if not isinstance(self.hamiltonian, IsingHamiltonian):
            raise TypeError(
                "hamiltonian must be IsingHamiltonian."
            )
        if not isinstance(self.equivalence_audit, QUBOIsingAudit):
            raise TypeError(
                "equivalence_audit must be QUBOIsingAudit."
            )
        if not isinstance(
            self.circuit_artifact,
            QAOACircuitArtifact,
        ):
            raise TypeError(
                "circuit_artifact must be QAOACircuitArtifact."
            )
        if not isinstance(
            self.operator_artifact,
            QiskitCostOperatorArtifact,
        ):
            raise TypeError(
                "operator_artifact must be QiskitCostOperatorArtifact."
            )
        if not isinstance(
            self.optimization_result,
            QAOAOptimizationResult,
        ):
            raise TypeError(
                "optimization_result must be QAOAOptimizationResult."
            )
        if not isinstance(
            self.final_evaluation,
            QAOAGPUEvaluation,
        ):
            raise TypeError(
                "final_evaluation must be QAOAGPUEvaluation."
            )
        if not isinstance(
            self.objective_snapshot,
            QAOAObjectiveSnapshot,
        ):
            raise TypeError(
                "objective_snapshot must be QAOAObjectiveSnapshot."
            )

        n_variables = self.qubo_model.n_variables
        most_probable = _readonly_binary_sample(
            self.most_probable_sample,
            expected_size=n_variables,
            name="most_probable_sample",
        )
        representative = _readonly_binary_sample(
            self.representative_ground_sample,
            expected_size=n_variables,
            name="representative_ground_sample",
        )

        if self.qubo_model.variable_order != self.hamiltonian.variable_order:
            raise QAOASolverError(
                "QUBO and Ising variable orders differ."
            )
        if not self.equivalence_audit.equivalent:
            raise QAOASolverError(
                "QUBO-to-Ising audit is not equivalent."
            )
        if (
            self.operator_artifact.hamiltonian_fingerprint
            != self.hamiltonian.fingerprint()
        ):
            raise QAOASolverError(
                "Qiskit operator belongs to another Hamiltonian."
            )
        if (
            self.final_evaluation.expectation_result.variable_order
            != self.qubo_model.variable_order
        ):
            raise QAOASolverError(
                "Final evaluation variable order differs from QUBO."
            )

        if self.source_kind == "bess_placement_qubo":
            if not isinstance(
                self.placement_qubo,
                BESSPlacementQUBO,
            ):
                raise TypeError(
                    "placement_qubo is required for BESS source."
                )
            if self.placement_qubo.model.fingerprint() != (
                self.qubo_model.fingerprint()
            ):
                raise QAOASolverError(
                    "placement_qubo and qubo_model differ."
                )
        elif self.placement_qubo is not None:
            raise QAOASolverError(
                "placement_qubo must be None for a generic QUBO."
            )

        expected_energy = float(self.expected_energy)
        variance = float(self.energy_variance)
        ground_probability = float(self.ground_probability)
        most_probability = float(self.most_probable_probability)

        for name, value in (
            ("most_probable_energy", self.most_probable_energy),
            (
                "representative_ground_energy",
                self.representative_ground_energy,
            ),
            ("expected_energy", expected_energy),
            ("energy_variance", variance),
            ("ground_probability", ground_probability),
            ("most_probable_probability", most_probability),
        ):
            if not math.isfinite(float(value)):
                raise QAOASolverError(f"{name} must be finite.")

        if variance < 0.0:
            raise QAOASolverError(
                "energy_variance must be non-negative."
            )
        if not 0.0 <= ground_probability <= 1.0:
            raise QAOASolverError(
                "ground_probability must lie in [0, 1]."
            )
        if not 0.0 <= most_probability <= 1.0:
            raise QAOASolverError(
                "most_probable_probability must lie in [0, 1]."
            )

        for name, value in (
            (
                "most_probable_is_feasible",
                self.most_probable_is_feasible,
            ),
            (
                "representative_ground_is_feasible",
                self.representative_ground_is_feasible,
            ),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None.")

        object.__setattr__(
            self,
            "most_probable_sample",
            most_probable,
        )
        object.__setattr__(
            self,
            "representative_ground_sample",
            representative,
        )
        object.__setattr__(
            self,
            "config_fingerprint",
            _require_digest(
                self.config_fingerprint,
                name="config_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata),
        )

    @property
    def n_qubits(self) -> int:
        return self.qubo_model.n_variables

    @property
    def best_parameters(self) -> QAOAParameterValues:
        return self.optimization_result.best_parameter_values

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOASolution-v1\0")
        digest.update(self.source_kind.encode("ascii"))
        digest.update(self.qubo_model.fingerprint().encode("ascii"))
        digest.update(self.hamiltonian.fingerprint().encode("ascii"))
        digest.update(
            self.circuit_artifact.fingerprint().encode("ascii")
        )
        digest.update(
            self.operator_artifact.fingerprint().encode("ascii")
        )
        digest.update(
            self.optimization_result.fingerprint().encode("ascii")
        )
        digest.update(
            _evaluation_fingerprint(
                self.final_evaluation
            ).encode("ascii")
        )
        digest.update(self.most_probable_sample.tobytes(order="C"))
        digest.update(
            self.representative_ground_sample.tobytes(order="C")
        )
        digest.update(
            np.asarray(
                [
                    self.most_probable_energy,
                    self.representative_ground_energy,
                    self.expected_energy,
                    self.energy_variance,
                    self.ground_probability,
                    self.most_probable_probability,
                ],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        digest.update(self.config_fingerprint.encode("ascii"))
        digest.update(
            json.dumps(
                dict(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def manifest(self) -> dict[str, Any]:
        most_selected = (
            None
            if self.most_probable_placement is None
            else list(self.most_probable_placement.selected_buses)
        )
        representative_selected = (
            None
            if self.representative_ground_placement is None
            else list(
                self.representative_ground_placement.selected_buses
            )
        )

        return {
            "fingerprint": self.fingerprint(),
            "source_kind": self.source_kind,
            "n_qubits": self.n_qubits,
            "qubit_limit": COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
            "repetitions": self.circuit_artifact.plan.config.repetitions,
            "execution_engine": "qiskit_aer.AerSimulator",
            "execution_method": "statevector",
            "execution_device": "GPU",
            "optimizer_engine": "qiskit_algorithms.optimizers",
            "optimizer_name": self.optimization_result.optimizer_name,
            "expected_energy": self.expected_energy,
            "energy_variance": self.energy_variance,
            "ground_probability": self.ground_probability,
            "most_probable_probability": (
                self.most_probable_probability
            ),
            "most_probable_energy": self.most_probable_energy,
            "representative_ground_energy": (
                self.representative_ground_energy
            ),
            "most_probable_sample": (
                self.most_probable_sample.tolist()
            ),
            "representative_ground_sample": (
                self.representative_ground_sample.tolist()
            ),
            "most_probable_is_feasible": (
                self.most_probable_is_feasible
            ),
            "representative_ground_is_feasible": (
                self.representative_ground_is_feasible
            ),
            "most_probable_selected_buses": most_selected,
            "representative_ground_selected_buses": (
                representative_selected
            ),
            "qubo_fingerprint": self.qubo_model.fingerprint(),
            "hamiltonian_fingerprint": (
                self.hamiltonian.fingerprint()
            ),
            "circuit_fingerprint": (
                self.circuit_artifact.fingerprint()
            ),
            "operator_fingerprint": (
                self.operator_artifact.fingerprint()
            ),
            "optimization_fingerprint": (
                self.optimization_result.fingerprint()
            ),
            "final_evaluation_fingerprint": (
                _evaluation_fingerprint(self.final_evaluation)
            ),
            "config_fingerprint": self.config_fingerprint,
            "metadata": dict(self.metadata),
        }


def _placement_decode(
    placement_qubo: BESSPlacementQUBO | None,
    sample: NDArray[np.int8],
) -> tuple[bool | None, BESSPlacement | None]:
    if placement_qubo is None:
        return None, None

    feasible = placement_qubo.is_feasible(sample)
    placement = placement_qubo.decode(sample) if feasible else None
    return feasible, placement


def _solve_qaoa_gpu(
    model: QUBOModel,
    *,
    source_kind: SourceKind,
    placement_qubo: BESSPlacementQUBO | None,
    config: QAOASolverConfig,
    initial_points: Sequence[ArrayLike] | None,
    metadata: Mapping[str, Any] | None,
) -> QAOASolution:
    if model.n_variables > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT:
        raise QAOASolverError(
            f"Exact QAOA statevector solve has {model.n_variables} "
            "variables; the Google Colab project limit is "
            f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}."
        )
    if model.n_variables > config.aer.max_statevector_qubits:
        raise QAOASolverError(
            f"QUBO has {model.n_variables} variables but Aer config "
            f"allows only {config.aer.max_statevector_qubits}."
        )

    hamiltonian = qubo_to_ising(model)
    audit = require_qubo_ising_equivalence(
        model,
        hamiltonian,
        exact_limit=config.audit_exact_limit,
        random_samples=config.audit_random_samples,
        seed=config.audit_seed,
        tolerance=config.audit_tolerance,
    )

    circuit_artifact = build_parameterized_qaoa_circuit(
        hamiltonian,
        config.circuit,
    )
    operator_artifact = build_qiskit_cost_operator(hamiltonian)

    objective_metadata = {
        "framework": "CSSF",
        "algorithm": "CSNN-T^QAOA",
        "source_kind": source_kind,
        "execution_engine": "qiskit_aer.AerSimulator",
        "execution_device": "GPU",
        "qubit_limit": COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
        **({} if metadata is None else dict(metadata)),
    }

    objective = QAOAGPUObjective(
        circuit_artifact,
        hamiltonian,
        aer_config=config.aer,
        expectation_config=config.expectation,
        operator_artifact=operator_artifact,
        config=config.objective,
        metadata=objective_metadata,
    )

    optimization_result = optimize_qaoa_gpu(
        objective,
        config=config.optimization,
        initial_points=initial_points,
        metadata=objective_metadata,
    )

    final_evaluation = objective.evaluate(
        optimization_result.best_parameters
    )
    snapshot = objective.snapshot()
    expectation_result = final_evaluation.expectation_result

    if not np.array_equal(
        final_evaluation.parameters.flat(),
        optimization_result.best_parameters,
    ):
        raise QAOASolverError(
            "Final evaluation parameters differ from optimizer output."
        )

    if not math.isclose(
        final_evaluation.objective_value,
        optimization_result.best_value,
        rel_tol=0.0,
        abs_tol=config.final_energy_tolerance,
    ):
        raise QAOASolverError(
            "Final GPU objective disagrees with optimization best value."
        )

    most_sample = _readonly_binary_sample(
        expectation_result.most_probable_sample,
        expected_size=model.n_variables,
        name="most_probable_sample",
    )
    representative_sample = _readonly_binary_sample(
        expectation_result.representative_ground_sample,
        expected_size=model.n_variables,
        name="representative_ground_sample",
    )

    most_energy = model.energy(most_sample)
    representative_energy = model.energy(representative_sample)

    if not math.isclose(
        most_energy,
        expectation_result.most_probable_energy,
        rel_tol=0.0,
        abs_tol=config.final_energy_tolerance,
    ):
        raise QAOASolverError(
            "Most-probable QUBO energy disagrees with Qiskit result."
        )
    if not math.isclose(
        representative_energy,
        expectation_result.minimum_energy,
        rel_tol=0.0,
        abs_tol=config.final_energy_tolerance,
    ):
        raise QAOASolverError(
            "Representative ground energy disagrees with QUBO."
        )

    most_feasible, most_placement = _placement_decode(
        placement_qubo,
        most_sample,
    )
    ground_feasible, ground_placement = _placement_decode(
        placement_qubo,
        representative_sample,
    )

    if (
        source_kind == "bess_placement_qubo"
        and config.require_most_probable_feasible
        and most_feasible is not True
    ):
        raise QAOASolverError(
            "Most-probable QAOA sample violates BESS cardinality."
        )
    if (
        source_kind == "bess_placement_qubo"
        and config.require_representative_ground_feasible
        and ground_feasible is not True
    ):
        raise QAOASolverError(
            "The exact representative ground state violates BESS "
            "cardinality; inspect penalty construction."
        )

    result_metadata = {
        **objective_metadata,
        "solver_config_fingerprint": config.fingerprint(),
        "equivalence_audit_exhaustive": audit.exhaustive,
        "equivalence_audit_samples": audit.checked_samples,
    }

    return QAOASolution(
        source_kind=source_kind,
        qubo_model=model,
        placement_qubo=placement_qubo,
        hamiltonian=hamiltonian,
        equivalence_audit=audit,
        circuit_artifact=circuit_artifact,
        operator_artifact=operator_artifact,
        optimization_result=optimization_result,
        final_evaluation=final_evaluation,
        objective_snapshot=snapshot,
        most_probable_sample=most_sample,
        representative_ground_sample=representative_sample,
        most_probable_energy=most_energy,
        representative_ground_energy=representative_energy,
        expected_energy=expectation_result.expected_energy,
        energy_variance=expectation_result.variance,
        ground_probability=expectation_result.ground_probability,
        most_probable_probability=(
            expectation_result.most_probable_probability
        ),
        most_probable_is_feasible=most_feasible,
        representative_ground_is_feasible=ground_feasible,
        most_probable_placement=most_placement,
        representative_ground_placement=ground_placement,
        config_fingerprint=config.fingerprint(),
        metadata=result_metadata,
    )


def solve_qubo_qaoa_gpu(
    model: QUBOModel,
    *,
    config: QAOASolverConfig | None = None,
    initial_points: Sequence[ArrayLike] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QAOASolution:
    """Solve one generic QUBO with exact Qiskit Aer GPU QAOA."""

    if not isinstance(model, QUBOModel):
        raise TypeError("model must be QUBOModel.")
    run_config = QAOASolverConfig() if config is None else config
    if not isinstance(run_config, QAOASolverConfig):
        raise TypeError(
            "config must be QAOASolverConfig or None."
        )

    return _solve_qaoa_gpu(
        model,
        source_kind="qubo_model",
        placement_qubo=None,
        config=run_config,
        initial_points=initial_points,
        metadata=metadata,
    )


def solve_bess_placement_qaoa_gpu(
    placement_qubo: BESSPlacementQUBO,
    *,
    config: QAOASolverConfig | None = None,
    initial_points: Sequence[ArrayLike] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QAOASolution:
    """Solve and decode one BESS-placement QUBO through Qiskit GPU."""

    if not isinstance(placement_qubo, BESSPlacementQUBO):
        raise TypeError(
            "placement_qubo must be BESSPlacementQUBO."
        )
    run_config = QAOASolverConfig() if config is None else config
    if not isinstance(run_config, QAOASolverConfig):
        raise TypeError(
            "config must be QAOASolverConfig or None."
        )

    return _solve_qaoa_gpu(
        placement_qubo.model,
        source_kind="bess_placement_qubo",
        placement_qubo=placement_qubo,
        config=run_config,
        initial_points=initial_points,
        metadata=metadata,
    )


__all__ = [
    "REAL_DTYPE",
    "INTEGER_DTYPE",
    "DEFAULT_AUDIT_EXACT_LIMIT",
    "DEFAULT_AUDIT_RANDOM_SAMPLES",
    "DEFAULT_AUDIT_SEED",
    "DEFAULT_AUDIT_TOLERANCE",
    "DEFAULT_FINAL_ENERGY_TOLERANCE",
    "SourceKind",
    "QAOASolverError",
    "QAOASolverConfig",
    "QAOASolution",
    "solve_qubo_qaoa_gpu",
    "solve_bess_placement_qaoa_gpu",
]
