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

"""Production orchestration for exact statevector MA-QAOA on Google Colab.

This module is the public MA-QAOA solve layer. It composes the
solver-independent QUBO model, the exact QUBO-to-Ising audit,
Qiskit circuit and operator creation,
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
from maqaoa.aer_gpu import (
    MAQAOAAerGPUConfig,
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
)
from maqaoa.circuit import (
    MAQAOACircuitArtifact,
    MAQAOACircuitConfig,
    build_parameterized_maqaoa_circuit,
)
from maqaoa.parameters import MAQAOAParameterValues
from maqaoa.expectation import (
    MAQAOAExpectationConfig,
    MAQAOAGPUEvaluation,
    MAQAOACostOperatorArtifact,
    build_maqaoa_qiskit_cost_operator,
)
from qaoa.hamiltonian import (
    IsingHamiltonian,
    QUBOIsingAudit,
    qubo_to_ising,
    require_qubo_ising_equivalence,
)
from maqaoa.objective import (
    MAQAOAGPUObjective,
    MAQAOAObjectiveConfig,
    MAQAOAObjectiveSnapshot,
)
from maqaoa.optimizer import (
    MAQAOAOptimizationConfig,
    MAQAOAOptimizationResult,
    optimize_maqaoa_gpu,
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


class MAQAOASolverError(RuntimeError):
    """Raised when the end-to-end MA-QAOA solve contract is violated."""


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise MAQAOASolverError(f"{name} must be non-negative.")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    normalized = _nonnegative_integer(value, name=name)
    if normalized == 0:
        raise MAQAOASolverError(f"{name} must be positive.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise MAQAOASolverError(
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
        raise MAQAOASolverError(
            f"{name} must contain exactly {expected_size} binary values."
        )
    if not np.all(np.isfinite(array)):
        raise MAQAOASolverError(f"{name} contains non-finite values.")
    if not np.all((array == 0.0) | (array == 1.0)):
        raise MAQAOASolverError(f"{name} must be exactly binary.")
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
        raise MAQAOASolverError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _require_digest(value: str, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64:
        raise MAQAOASolverError(
            f"{name} must be a SHA-256 digest."
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise MAQAOASolverError(
            f"{name} must be hexadecimal."
        ) from exc
    return digest


def _evaluation_fingerprint(
    evaluation: MAQAOAGPUEvaluation,
) -> str:
    """Fingerprint an evaluation without depending on mutable APIs."""

    if not isinstance(evaluation, MAQAOAGPUEvaluation):
        raise TypeError(
            "evaluation must be MAQAOAGPUEvaluation."
        )
    digest = hashlib.sha256()
    digest.update(b"CSSF-MAQAOAGPUEvaluation-v1\0")
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
    config: MAQAOACircuitConfig,
) -> str:
    """Return the circuit configuration's validated fingerprint."""

    if not isinstance(config, MAQAOACircuitConfig):
        raise TypeError(
            "config must be MAQAOACircuitConfig."
        )
    return config.fingerprint()


@dataclass(frozen=True, slots=True)
class MAQAOASolverConfig:
    """Validated configuration for one exact Qiskit Aer GPU solve."""

    circuit: MAQAOACircuitConfig = field(
        default_factory=MAQAOACircuitConfig
    )
    aer: MAQAOAAerGPUConfig = field(default_factory=MAQAOAAerGPUConfig)
    expectation: MAQAOAExpectationConfig = field(
        default_factory=MAQAOAExpectationConfig
    )
    objective: MAQAOAObjectiveConfig = field(
        default_factory=lambda: MAQAOAObjectiveConfig(
            retain_best_evaluation=True
        )
    )
    optimization: MAQAOAOptimizationConfig = field(
        default_factory=MAQAOAOptimizationConfig
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
        if not isinstance(self.circuit, MAQAOACircuitConfig):
            raise TypeError("circuit must be MAQAOACircuitConfig.")
        if not isinstance(self.aer, MAQAOAAerGPUConfig):
            raise TypeError("aer must be MAQAOAAerGPUConfig.")
        if not isinstance(
            self.expectation,
            MAQAOAExpectationConfig,
        ):
            raise TypeError(
                "expectation must be MAQAOAExpectationConfig."
            )
        if not isinstance(self.objective, MAQAOAObjectiveConfig):
            raise TypeError(
                "objective must be MAQAOAObjectiveConfig."
            )
        if not isinstance(
            self.optimization,
            MAQAOAOptimizationConfig,
        ):
            raise TypeError(
                "optimization must be MAQAOAOptimizationConfig."
            )

        exact_limit = _nonnegative_integer(
            self.audit_exact_limit,
            name="audit_exact_limit",
        )
        if exact_limit > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT:
            raise MAQAOASolverError(
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
            raise MAQAOASolverError(
                "Aer configuration exceeds the 22-qubit limit."
            )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOASolverConfig-v1\0")
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
class MAQAOASolution:
    """Immutable end-to-end MA-QAOA solution and scientific audit record."""

    source_kind: SourceKind
    qubo_model: QUBOModel
    placement_qubo: BESSPlacementQUBO | None
    hamiltonian: IsingHamiltonian
    equivalence_audit: QUBOIsingAudit
    circuit_artifact: MAQAOACircuitArtifact
    operator_artifact: MAQAOACostOperatorArtifact
    optimization_result: MAQAOAOptimizationResult
    final_evaluation: MAQAOAGPUEvaluation
    objective_snapshot: MAQAOAObjectiveSnapshot
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
            raise MAQAOASolverError("source_kind is unsupported.")
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
            MAQAOACircuitArtifact,
        ):
            raise TypeError(
                "circuit_artifact must be MAQAOACircuitArtifact."
            )
        if not isinstance(
            self.operator_artifact,
            MAQAOACostOperatorArtifact,
        ):
            raise TypeError(
                "operator_artifact must be MAQAOACostOperatorArtifact."
            )
        if not isinstance(
            self.optimization_result,
            MAQAOAOptimizationResult,
        ):
            raise TypeError(
                "optimization_result must be MAQAOAOptimizationResult."
            )
        if not isinstance(
            self.final_evaluation,
            MAQAOAGPUEvaluation,
        ):
            raise TypeError(
                "final_evaluation must be MAQAOAGPUEvaluation."
            )
        if not isinstance(
            self.objective_snapshot,
            MAQAOAObjectiveSnapshot,
        ):
            raise TypeError(
                "objective_snapshot must be MAQAOAObjectiveSnapshot."
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
            raise MAQAOASolverError(
                "QUBO and Ising variable orders differ."
            )
        if not self.equivalence_audit.equivalent:
            raise MAQAOASolverError(
                "QUBO-to-Ising audit is not equivalent."
            )
        hamiltonian_fingerprint = self.hamiltonian.fingerprint()
        circuit_plan = self.circuit_artifact.plan
        parameter_layout = circuit_plan.parameter_layout
        expectation_result = self.final_evaluation.expectation_result

        if (
            circuit_plan.hamiltonian.fingerprint()
            != hamiltonian_fingerprint
        ):
            raise MAQAOASolverError(
                "Circuit artifact belongs to another Hamiltonian."
            )
        if (
            self.operator_artifact.hamiltonian_fingerprint
            != hamiltonian_fingerprint
        ):
            raise MAQAOASolverError(
                "Qiskit operator belongs to another Hamiltonian."
            )
        if (
            expectation_result.hamiltonian_fingerprint
            != hamiltonian_fingerprint
        ):
            raise MAQAOASolverError(
                "Final expectation belongs to another Hamiltonian."
            )
        if (
            expectation_result.operator_fingerprint
            != self.operator_artifact.fingerprint()
        ):
            raise MAQAOASolverError(
                "Final expectation belongs to another operator."
            )
        if (
            self.final_evaluation.statevector_result.circuit_fingerprint
            != self.circuit_artifact.fingerprint()
        ):
            raise MAQAOASolverError(
                "Final statevector belongs to another circuit."
            )
        if (
            expectation_result.variable_order
            != self.qubo_model.variable_order
        ):
            raise MAQAOASolverError(
                "Final evaluation variable order differs from QUBO."
            )
        if (
            self.optimization_result.parameter_layout_fingerprint
            != parameter_layout.fingerprint()
        ):
            raise MAQAOASolverError(
                "Optimizer result belongs to another parameter layout."
            )
        if (
            self.optimization_result.gamma_shape
            != parameter_layout.gamma_shape
            or self.optimization_result.beta_shape
            != parameter_layout.beta_shape
        ):
            raise MAQAOASolverError(
                "Optimizer parameter shapes differ from the circuit."
            )
        if (
            self.optimization_result.objective_fingerprint
            != self.objective_snapshot.objective_fingerprint
        ):
            raise MAQAOASolverError(
                "Optimizer and objective snapshot identities differ."
            )
        if not np.array_equal(
            self.final_evaluation.parameters.flat(),
            self.optimization_result.best_parameters,
        ):
            raise MAQAOASolverError(
                "Final parameters differ from the optimizer result."
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
                raise MAQAOASolverError(
                    "placement_qubo and qubo_model differ."
                )
        elif self.placement_qubo is not None:
            raise MAQAOASolverError(
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
                raise MAQAOASolverError(f"{name} must be finite.")

        scalar_pairs = (
            (
                "expected_energy",
                expected_energy,
                expectation_result.expected_energy,
            ),
            (
                "energy_variance",
                variance,
                expectation_result.variance,
            ),
            (
                "ground_probability",
                ground_probability,
                expectation_result.ground_probability,
            ),
            (
                "most_probable_probability",
                most_probability,
                expectation_result.most_probable_probability,
            ),
            (
                "most_probable_energy",
                float(self.most_probable_energy),
                expectation_result.most_probable_energy,
            ),
            (
                "representative_ground_energy",
                float(self.representative_ground_energy),
                expectation_result.minimum_energy,
            ),
        )
        for name, supplied, evaluated in scalar_pairs:
            if not math.isclose(
                supplied,
                float(evaluated),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise MAQAOASolverError(
                    f"{name} disagrees with the final evaluation."
                )

        if variance < 0.0:
            raise MAQAOASolverError(
                "energy_variance must be non-negative."
            )
        if not 0.0 <= ground_probability <= 1.0:
            raise MAQAOASolverError(
                "ground_probability must lie in [0, 1]."
            )
        if not 0.0 <= most_probability <= 1.0:
            raise MAQAOASolverError(
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
    def best_parameters(self) -> MAQAOAParameterValues:
        return self.optimization_result.best_parameter_values

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOASolution-v1\0")
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


def _solve_maqaoa_gpu(
    model: QUBOModel,
    *,
    source_kind: SourceKind,
    placement_qubo: BESSPlacementQUBO | None,
    config: MAQAOASolverConfig,
    initial_points: Sequence[ArrayLike] | None,
    metadata: Mapping[str, Any] | None,
) -> MAQAOASolution:
    if model.n_variables > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT:
        raise MAQAOASolverError(
            f"Exact MA-QAOA statevector solve has {model.n_variables} "
            "variables; the Google Colab project limit is "
            f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}."
        )
    if model.n_variables > config.aer.max_statevector_qubits:
        raise MAQAOASolverError(
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

    circuit_artifact = build_parameterized_maqaoa_circuit(
        hamiltonian,
        config.circuit,
    )
    operator_artifact = build_maqaoa_qiskit_cost_operator(hamiltonian)

    objective_metadata = {
        "framework": "CSSF",
        "algorithm": "CSNN-T^MA-QAOA",
        "source_kind": source_kind,
        "execution_engine": "qiskit_aer.AerSimulator",
        "execution_device": "GPU",
        "qubit_limit": COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
        **({} if metadata is None else dict(metadata)),
    }

    objective = MAQAOAGPUObjective(
        circuit_artifact,
        hamiltonian,
        aer_config=config.aer,
        expectation_config=config.expectation,
        operator_artifact=operator_artifact,
        config=config.objective,
        metadata=objective_metadata,
    )

    optimization_result = optimize_maqaoa_gpu(
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
        raise MAQAOASolverError(
            "Final evaluation parameters differ from optimizer output."
        )

    if not math.isclose(
        final_evaluation.objective_value,
        optimization_result.best_value,
        rel_tol=0.0,
        abs_tol=config.final_energy_tolerance,
    ):
        raise MAQAOASolverError(
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
        raise MAQAOASolverError(
            "Most-probable QUBO energy disagrees with Qiskit result."
        )
    if not math.isclose(
        representative_energy,
        expectation_result.minimum_energy,
        rel_tol=0.0,
        abs_tol=config.final_energy_tolerance,
    ):
        raise MAQAOASolverError(
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
        raise MAQAOASolverError(
            "Most-probable MA-QAOA sample violates BESS cardinality."
        )
    if (
        source_kind == "bess_placement_qubo"
        and config.require_representative_ground_feasible
        and ground_feasible is not True
    ):
        raise MAQAOASolverError(
            "The exact representative ground state violates BESS "
            "cardinality; inspect penalty construction."
        )

    result_metadata = {
        **objective_metadata,
        "solver_config_fingerprint": config.fingerprint(),
        "equivalence_audit_exhaustive": audit.exhaustive,
        "equivalence_audit_samples": audit.checked_samples,
    }

    return MAQAOASolution(
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


def solve_qubo_maqaoa_gpu(
    model: QUBOModel,
    *,
    config: MAQAOASolverConfig | None = None,
    initial_points: Sequence[ArrayLike] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MAQAOASolution:
    """Solve one generic QUBO with exact Qiskit Aer GPU MA-QAOA."""

    if not isinstance(model, QUBOModel):
        raise TypeError("model must be QUBOModel.")
    run_config = MAQAOASolverConfig() if config is None else config
    if not isinstance(run_config, MAQAOASolverConfig):
        raise TypeError(
            "config must be MAQAOASolverConfig or None."
        )

    return _solve_maqaoa_gpu(
        model,
        source_kind="qubo_model",
        placement_qubo=None,
        config=run_config,
        initial_points=initial_points,
        metadata=metadata,
    )


def solve_bess_placement_maqaoa_gpu(
    placement_qubo: BESSPlacementQUBO,
    *,
    config: MAQAOASolverConfig | None = None,
    initial_points: Sequence[ArrayLike] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MAQAOASolution:
    """Solve and decode one BESS-placement QUBO through Qiskit GPU."""

    if not isinstance(placement_qubo, BESSPlacementQUBO):
        raise TypeError(
            "placement_qubo must be BESSPlacementQUBO."
        )
    run_config = MAQAOASolverConfig() if config is None else config
    if not isinstance(run_config, MAQAOASolverConfig):
        raise TypeError(
            "config must be MAQAOASolverConfig or None."
        )

    return _solve_maqaoa_gpu(
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
    "MAQAOASolverError",
    "MAQAOASolverConfig",
    "MAQAOASolution",
    "solve_qubo_maqaoa_gpu",
    "solve_bess_placement_maqaoa_gpu",
]
