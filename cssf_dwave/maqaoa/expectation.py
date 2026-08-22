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

"""Exact MA-QAOA cost expectation from Qiskit Aer GPU statevectors.

The quantum state is produced exclusively by
``qiskit_aer.AerSimulator(method="statevector", device="GPU")`` through
:mod:`maqaoa.aer_gpu`. This module materializes the Ising cost Hamiltonian as a
Qiskit ``SparsePauliOp`` and evaluates its expectation with Qiskit
``Statevector.expectation_value``.

A chunked diagonal audit independently verifies Qiskit bit ordering and
computes variance, spectral extrema, ground-state probability, and diagnostic
samples without allocating a dense ``2**n by n`` basis matrix. The project-wide
free-Colab ceiling remains 22 qubits.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import NDArray

from maqaoa import (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    MAQAOA_ALGORITHM_NAME,
)
from maqaoa.aer_gpu import (
    MAQAOAAerGPUConfig,
    MAQAOAStatevectorResult,
    run_maqaoa_statevector_gpu,
)
from maqaoa.circuit import MAQAOACircuitArtifact
from maqaoa.parameters import MAQAOAParameterValues
from qaoa.hamiltonian import IsingHamiltonian


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
COMPLEX_DTYPE: Final[np.dtype[np.complex128]] = np.dtype(np.complex128)
DEFAULT_EXPECTATION_ATOL: Final[float] = 1.0e-10
DEFAULT_IMAGINARY_ATOL: Final[float] = 1.0e-10
DEFAULT_VARIANCE_ATOL: Final[float] = 1.0e-10
DEFAULT_GROUND_ENERGY_ATOL: Final[float] = 1.0e-10
DEFAULT_BASIS_CHUNK_SIZE: Final[int] = 65536
REQUIRED_OPERATOR_CLASS: Final[str] = "qiskit.quantum_info.SparsePauliOp"
REQUIRED_STATE_CLASS: Final[str] = "qiskit.quantum_info.Statevector"


class MAQAOAExpectationError(RuntimeError):
    """Base error for exact MA-QAOA expectation evaluation."""


class MAQAOAQiskitExpectationUnavailableError(MAQAOAExpectationError):
    """Raised when the required Qiskit quantum-info API is unavailable."""


class MAQAOAQiskitExpectationResultError(MAQAOAExpectationError):
    """Raised when Qiskit or the diagonal audit returns invalid results."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MAQAOAExpectationError(f"{name} must be finite.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized <= 0.0:
        raise MAQAOAExpectationError(
            f"{name} must be strictly positive."
        )
    return normalized


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise MAQAOAExpectationError(
            f"{name} must be strictly positive."
        )
    return value


def _sha256_digest(value: str, *, name: str) -> str:
    normalized = str(value)
    if len(normalized) != 64:
        raise MAQAOAExpectationError(
            f"{name} must be a SHA-256 digest."
        )
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise MAQAOAExpectationError(
            f"{name} must contain hexadecimal characters."
        ) from exc
    return normalized.lower()


def _readonly_binary_sample(
    values: NDArray[np.int8] | list[int] | tuple[int, ...],
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.int8]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size != expected_size:
        raise MAQAOAExpectationError(
            f"{name} must contain {expected_size} binary values."
        )
    if not np.all((array == 0) | (array == 1)):
        raise MAQAOAExpectationError(
            f"{name} must contain only zero and one."
        )
    result = np.ascontiguousarray(array, dtype=np.int8)
    result.setflags(write=False)
    return result


def _json_metadata(
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
        raise MAQAOAExpectationError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _load_qiskit_quantum_info() -> tuple[Any, Any, str]:
    """Load Qiskit quantum-info classes lazily."""

    try:
        qiskit = importlib.import_module("qiskit")
        quantum_info = importlib.import_module(
            "qiskit.quantum_info"
        )
    except Exception as exc:
        raise MAQAOAQiskitExpectationUnavailableError(
            "Qiskit is required for exact expectation evaluation."
        ) from exc

    sparse_pauli_op = getattr(
        quantum_info,
        "SparsePauliOp",
        None,
    )
    statevector = getattr(
        quantum_info,
        "Statevector",
        None,
    )

    if sparse_pauli_op is None:
        raise MAQAOAQiskitExpectationUnavailableError(
            f"Missing {REQUIRED_OPERATOR_CLASS}."
        )
    if statevector is None:
        raise MAQAOAQiskitExpectationUnavailableError(
            f"Missing {REQUIRED_STATE_CLASS}."
        )

    version = str(getattr(qiskit, "__version__", "unknown"))
    return sparse_pauli_op, statevector, version


def _operator_terms(
    hamiltonian: IsingHamiltonian,
) -> tuple[tuple[str, float], ...]:
    identity = "I" * hamiltonian.n_qubits
    terms: list[tuple[str, float]] = [
        (identity, float(hamiltonian.offset))
    ]
    terms.extend(
        (
            term.qiskit_label(hamiltonian.n_qubits),
            float(term.coefficient),
        )
        for term in hamiltonian.pauli_z_terms()
    )
    return tuple(terms)


@dataclass(frozen=True, slots=True)
class MAQAOAExpectationConfig:
    """Numerical controls for exact expectation and spectral diagnostics."""

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
        if not isinstance(
            self.require_diagonal_crosscheck,
            bool,
        ):
            raise TypeError(
                "require_diagonal_crosscheck must be boolean."
            )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAExpectationConfig-v1\0")
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
class MAQAOACostOperatorArtifact:
    """Qiskit SparsePauliOp plus deterministic construction metadata."""

    operator: Any
    terms: tuple[tuple[str, float], ...]
    n_qubits: int
    hamiltonian_fingerprint: str
    qiskit_version: str

    def __post_init__(self) -> None:
        n_qubits = _positive_integer(
            self.n_qubits,
            name="n_qubits",
        )
        terms = tuple(
            (str(label), _finite_float(value, name="coefficient"))
            for label, value in self.terms
        )
        if not terms:
            raise MAQAOAExpectationError(
                "Qiskit cost operator must contain at least identity."
            )
        if any(len(label) != n_qubits for label, _ in terms):
            raise MAQAOAExpectationError(
                "Every Pauli label must have length n_qubits."
            )
        if any(
            set(label) - {"I", "Z"}
            for label, _ in terms
        ):
            raise MAQAOAExpectationError(
                "Cost operator may contain only I and Z Paulis."
            )

        operator_qubits = getattr(
            self.operator,
            "num_qubits",
            n_qubits,
        )
        if int(operator_qubits) != n_qubits:
            raise MAQAOAExpectationError(
                "Qiskit operator qubit count differs from n_qubits."
            )

        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(
            self,
            "hamiltonian_fingerprint",
            _sha256_digest(
                self.hamiltonian_fingerprint,
                name="hamiltonian_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "qiskit_version",
            str(self.qiskit_version),
        )

    @property
    def term_count(self) -> int:
        return len(self.terms)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOACostOperatorArtifact-v1\0")
        digest.update(
            self.hamiltonian_fingerprint.encode("ascii")
        )
        digest.update(self.qiskit_version.encode("utf-8"))
        digest.update(
            json.dumps(
                self.terms,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


def build_maqaoa_qiskit_cost_operator(
    hamiltonian: IsingHamiltonian,
) -> MAQAOACostOperatorArtifact:
    """Materialize the exact Ising Hamiltonian as SparsePauliOp."""

    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError(
            "hamiltonian must be IsingHamiltonian."
        )
    if (
        hamiltonian.n_qubits
        > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
    ):
        raise MAQAOAExpectationError(
            "Qiskit statevector expectation is limited to "
            f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT} qubits."
        )

    sparse_pauli_op, _, qiskit_version = (
        _load_qiskit_quantum_info()
    )
    terms = _operator_terms(hamiltonian)

    try:
        operator = sparse_pauli_op.from_list(list(terms))
    except Exception as exc:
        raise MAQAOAQiskitExpectationUnavailableError(
            "Cannot construct Qiskit SparsePauliOp."
        ) from exc

    return MAQAOACostOperatorArtifact(
        operator=operator,
        terms=terms,
        n_qubits=hamiltonian.n_qubits,
        hamiltonian_fingerprint=hamiltonian.fingerprint(),
        qiskit_version=qiskit_version,
    )


def _basis_chunk_energies(
    hamiltonian: IsingHamiltonian,
    *,
    start: int,
    stop: int,
) -> NDArray[np.float64]:
    indices = np.arange(start, stop, dtype=np.uint64)
    shifts = np.arange(
        hamiltonian.n_qubits,
        dtype=np.uint64,
    )
    binary = (
        (indices[:, None] >> shifts[None, :]) & np.uint64(1)
    ).astype(np.float64)
    spins = 1.0 - 2.0 * binary
    energies = (
        hamiltonian.offset
        + spins @ hamiltonian.linear_z
        + np.einsum(
            "bi,ij,bj->b",
            spins,
            hamiltonian.quadratic_zz,
            spins,
            optimize=True,
        )
    )
    return np.ascontiguousarray(energies, dtype=REAL_DTYPE)


@dataclass(frozen=True, slots=True)
class MAQAOAExpectationResult:
    """Exact energy statistics for one Qiskit Aer GPU statevector."""

    expected_energy: float
    qiskit_expected_energy: float
    diagonal_expected_energy: float
    expectation_discrepancy: float
    variance: float
    standard_deviation: float
    minimum_energy: float
    maximum_energy: float
    ground_probability: float
    ground_degeneracy: int
    representative_ground_index: int
    representative_ground_sample: NDArray[np.int8]
    representative_ground_probability: float
    most_probable_index: int
    most_probable_sample: NDArray[np.int8]
    most_probable_probability: float
    most_probable_energy: float
    n_qubits: int
    variable_order: tuple[str, ...]
    statevector_fingerprint: str
    hamiltonian_fingerprint: str
    operator_fingerprint: str
    config_fingerprint: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        n_qubits = _positive_integer(
            self.n_qubits,
            name="n_qubits",
        )
        variables = tuple(self.variable_order)
        if len(variables) != n_qubits:
            raise MAQAOAExpectationError(
                "variable_order length must equal n_qubits."
            )

        finite_fields = (
            "expected_energy",
            "qiskit_expected_energy",
            "diagonal_expected_energy",
            "expectation_discrepancy",
            "variance",
            "standard_deviation",
            "minimum_energy",
            "maximum_energy",
            "ground_probability",
            "representative_ground_probability",
            "most_probable_probability",
            "most_probable_energy",
        )
        for field_name in finite_fields:
            object.__setattr__(
                self,
                field_name,
                _finite_float(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )

        if self.expectation_discrepancy < 0.0:
            raise MAQAOAExpectationError(
                "expectation_discrepancy must be non-negative."
            )
        if self.variance < 0.0:
            raise MAQAOAExpectationError(
                "variance must be non-negative."
            )
        if self.standard_deviation < 0.0:
            raise MAQAOAExpectationError(
                "standard_deviation must be non-negative."
            )
        if self.minimum_energy > self.maximum_energy:
            raise MAQAOAExpectationError(
                "minimum_energy cannot exceed maximum_energy."
            )

        for field_name in (
            "ground_probability",
            "representative_ground_probability",
            "most_probable_probability",
        ):
            probability = float(getattr(self, field_name))
            if probability < 0.0 or probability > 1.0 + 1.0e-10:
                raise MAQAOAExpectationError(
                    f"{field_name} must be in [0, 1]."
                )

        degeneracy = _positive_integer(
            self.ground_degeneracy,
            name="ground_degeneracy",
        )
        dimension = 1 << n_qubits
        for field_name in (
            "representative_ground_index",
            "most_probable_index",
        ):
            index = getattr(self, field_name)
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"{field_name} must be an integer.")
            if index < 0 or index >= dimension:
                raise MAQAOAExpectationError(
                    f"{field_name} is outside the statevector."
                )

        ground_sample = _readonly_binary_sample(
            self.representative_ground_sample,
            expected_size=n_qubits,
            name="representative_ground_sample",
        )
        probable_sample = _readonly_binary_sample(
            self.most_probable_sample,
            expected_size=n_qubits,
            name="most_probable_sample",
        )

        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "variable_order", variables)
        object.__setattr__(self, "ground_degeneracy", degeneracy)
        object.__setattr__(
            self,
            "representative_ground_sample",
            ground_sample,
        )
        object.__setattr__(
            self,
            "most_probable_sample",
            probable_sample,
        )
        for field_name in (
            "statevector_fingerprint",
            "hamiltonian_fingerprint",
            "operator_fingerprint",
            "config_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256_digest(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(self.metadata),
        )

    def labeled_ground_sample(self) -> dict[str, int]:
        return {
            variable: int(value)
            for variable, value in zip(
                self.variable_order,
                self.representative_ground_sample,
            )
        }

    def labeled_most_probable_sample(self) -> dict[str, int]:
        return {
            variable: int(value)
            for variable, value in zip(
                self.variable_order,
                self.most_probable_sample,
            )
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAExpectationResult-v1\0")
        digest.update(
            np.asarray(
                [
                    self.expected_energy,
                    self.qiskit_expected_energy,
                    self.diagonal_expected_energy,
                    self.expectation_discrepancy,
                    self.variance,
                    self.standard_deviation,
                    self.minimum_energy,
                    self.maximum_energy,
                    self.ground_probability,
                    self.representative_ground_probability,
                    self.most_probable_probability,
                    self.most_probable_energy,
                ],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        digest.update(
            np.asarray(
                [
                    self.n_qubits,
                    self.ground_degeneracy,
                    self.representative_ground_index,
                    self.most_probable_index,
                ],
                dtype=np.int64,
            ).tobytes(order="C")
        )
        digest.update(
            self.representative_ground_sample.tobytes(order="C")
        )
        digest.update(self.most_probable_sample.tobytes(order="C"))
        digest.update(
            json.dumps(
                {
                    "variable_order": self.variable_order,
                    "statevector_fingerprint": (
                        self.statevector_fingerprint
                    ),
                    "hamiltonian_fingerprint": (
                        self.hamiltonian_fingerprint
                    ),
                    "operator_fingerprint": self.operator_fingerprint,
                    "config_fingerprint": self.config_fingerprint,
                    "metadata": dict(self.metadata),
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MAQAOAGPUEvaluation:
    """Parameters, Aer GPU statevector, and exact cost statistics."""

    parameters: MAQAOAParameterValues
    statevector_result: MAQAOAStatevectorResult
    expectation_result: MAQAOAExpectationResult

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, MAQAOAParameterValues):
            raise TypeError(
                "parameters must be MAQAOAParameterValues."
            )
        if not isinstance(
            self.statevector_result,
            MAQAOAStatevectorResult,
        ):
            raise TypeError(
                "statevector_result must be MAQAOAStatevectorResult."
            )
        if not isinstance(
            self.expectation_result,
            MAQAOAExpectationResult,
        ):
            raise TypeError(
                "expectation_result must be MAQAOAExpectationResult."
            )
        if (
            self.statevector_result.parameter_fingerprint
            != self.parameters.fingerprint()
        ):
            raise MAQAOAExpectationError(
                "Statevector result belongs to another parameter tensor."
            )
        if (
            self.statevector_result.fingerprint()
            != self.expectation_result.statevector_fingerprint
        ):
            raise MAQAOAExpectationError(
                "Expectation result belongs to another statevector."
            )
        if (
            self.statevector_result.n_qubits
            != self.expectation_result.n_qubits
            or self.statevector_result.variable_order
            != self.expectation_result.variable_order
        ):
            raise MAQAOAExpectationError(
                "Statevector and expectation result dimensions differ."
            )

    @property
    def objective_value(self) -> float:
        return self.expectation_result.expected_energy

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAGPUEvaluation-v1\0")
        digest.update(self.parameters.fingerprint().encode("ascii"))
        digest.update(
            self.statevector_result.fingerprint().encode("ascii")
        )
        digest.update(
            self.expectation_result.fingerprint().encode("ascii")
        )
        return digest.hexdigest()


def evaluate_maqaoa_statevector_expectation(
    statevector_result: MAQAOAStatevectorResult,
    hamiltonian: IsingHamiltonian,
    *,
    config: MAQAOAExpectationConfig | None = None,
    operator_artifact: MAQAOACostOperatorArtifact | None = None,
) -> MAQAOAExpectationResult:
    """Evaluate exact MA-QAOA cost and audit it against the diagonal spectrum."""

    if not isinstance(
        statevector_result,
        MAQAOAStatevectorResult,
    ):
        raise TypeError(
            "statevector_result must be MAQAOAStatevectorResult."
        )
    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError(
            "hamiltonian must be IsingHamiltonian."
        )

    run_config = (
        MAQAOAExpectationConfig()
        if config is None
        else config
    )
    if not isinstance(run_config, MAQAOAExpectationConfig):
        raise TypeError(
            "config must be MAQAOAExpectationConfig or None."
        )

    if statevector_result.n_qubits != hamiltonian.n_qubits:
        raise MAQAOAExpectationError(
            "Statevector and Hamiltonian qubit counts differ."
        )
    if (
        statevector_result.variable_order
        != hamiltonian.variable_order
    ):
        raise MAQAOAExpectationError(
            "Statevector and Hamiltonian variable orders differ."
        )
    if (
        hamiltonian.n_qubits
        > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
    ):
        raise MAQAOAExpectationError(
            "Exact statevector expectation exceeds the project-wide "
            f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}-qubit limit."
        )

    artifact = (
        build_maqaoa_qiskit_cost_operator(hamiltonian)
        if operator_artifact is None
        else operator_artifact
    )
    if not isinstance(artifact, MAQAOACostOperatorArtifact):
        raise TypeError(
            "operator_artifact must be MAQAOACostOperatorArtifact."
        )
    if (
        artifact.hamiltonian_fingerprint
        != hamiltonian.fingerprint()
    ):
        raise MAQAOAExpectationError(
            "Qiskit operator belongs to another Hamiltonian."
        )

    _, statevector_class, _ = _load_qiskit_quantum_info()
    try:
        qiskit_state = statevector_class(
            np.asarray(
                statevector_result.statevector,
                dtype=COMPLEX_DTYPE,
            )
        )
        raw_expectation = qiskit_state.expectation_value(
            artifact.operator
        )
    except Exception as exc:
        raise MAQAOAQiskitExpectationResultError(
            "Qiskit Statevector.expectation_value failed."
        ) from exc

    complex_expectation = complex(raw_expectation)
    if not (
        math.isfinite(complex_expectation.real)
        and math.isfinite(complex_expectation.imag)
    ):
        raise MAQAOAQiskitExpectationResultError(
            "Qiskit expectation is non-finite."
        )
    if abs(complex_expectation.imag) > run_config.imaginary_atol:
        raise MAQAOAQiskitExpectationResultError(
            "A Hermitian Ising expectation has an excessive "
            f"imaginary component: {complex_expectation.imag}."
        )
    qiskit_expectation = float(complex_expectation.real)

    probabilities = statevector_result.probabilities
    dimension = probabilities.size
    diagonal_expectation = 0.0
    second_moment = 0.0
    minimum_energy = math.inf
    maximum_energy = -math.inf
    ground_probability = 0.0
    ground_degeneracy = 0
    representative_ground_index = 0
    representative_ground_probability = -1.0

    for start in range(0, dimension, run_config.basis_chunk_size):
        stop = min(start + run_config.basis_chunk_size, dimension)
        energies = _basis_chunk_energies(
            hamiltonian,
            start=start,
            stop=stop,
        )
        chunk_probabilities = probabilities[start:stop]

        diagonal_expectation += float(
            np.dot(chunk_probabilities, energies)
        )
        second_moment += float(
            np.dot(chunk_probabilities, energies * energies)
        )
        chunk_minimum = float(np.min(energies))
        chunk_maximum = float(np.max(energies))
        maximum_energy = max(maximum_energy, chunk_maximum)

        if (
            not math.isfinite(minimum_energy)
            or chunk_minimum
            < minimum_energy - run_config.ground_energy_atol
        ):
            minimum_energy = chunk_minimum
            ground_probability = 0.0
            ground_degeneracy = 0
            representative_ground_probability = -1.0

        if math.isclose(
            chunk_minimum,
            minimum_energy,
            rel_tol=0.0,
            abs_tol=run_config.ground_energy_atol,
        ):
            ground_mask = np.isclose(
                energies,
                minimum_energy,
                rtol=0.0,
                atol=run_config.ground_energy_atol,
            )
            ground_indices = np.flatnonzero(ground_mask)
            ground_probabilities = chunk_probabilities[ground_mask]
            ground_probability += float(
                np.sum(ground_probabilities)
            )
            ground_degeneracy += int(ground_indices.size)

            if ground_indices.size:
                local_choice = int(
                    np.argmax(ground_probabilities)
                )
                local_probability = float(
                    ground_probabilities[local_choice]
                )
                if (
                    local_probability
                    > representative_ground_probability
                ):
                    representative_ground_probability = (
                        local_probability
                    )
                    representative_ground_index = int(
                        start + ground_indices[local_choice]
                    )

    if not all(
        math.isfinite(value)
        for value in (
            diagonal_expectation,
            second_moment,
            minimum_energy,
            maximum_energy,
            ground_probability,
        )
    ):
        raise MAQAOAQiskitExpectationResultError(
            "Diagonal spectral audit produced non-finite values."
        )

    discrepancy = abs(
        qiskit_expectation - diagonal_expectation
    )
    if (
        run_config.require_diagonal_crosscheck
        and discrepancy > run_config.expectation_atol
    ):
        raise MAQAOAQiskitExpectationResultError(
            "Qiskit expectation and diagonal audit disagree: "
            f"difference={discrepancy:.17g}."
        )

    variance = second_moment - diagonal_expectation**2
    if variance < -run_config.variance_atol:
        raise MAQAOAQiskitExpectationResultError(
            f"Computed energy variance is negative: {variance}."
        )
    variance = max(0.0, float(variance))

    most_probable_index = statevector_result.most_probable_index
    most_probable_sample = (
        statevector_result.most_probable_sample()
    )
    most_probable_energy = hamiltonian.binary_energy(
        most_probable_sample
    )
    representative_ground_sample = (
        statevector_result.binary_sample(
            representative_ground_index
        )
    )

    return MAQAOAExpectationResult(
        expected_energy=qiskit_expectation,
        qiskit_expected_energy=qiskit_expectation,
        diagonal_expected_energy=diagonal_expectation,
        expectation_discrepancy=discrepancy,
        variance=variance,
        standard_deviation=math.sqrt(variance),
        minimum_energy=minimum_energy,
        maximum_energy=maximum_energy,
        ground_probability=min(1.0, ground_probability),
        ground_degeneracy=ground_degeneracy,
        representative_ground_index=representative_ground_index,
        representative_ground_sample=representative_ground_sample,
        representative_ground_probability=(
            representative_ground_probability
        ),
        most_probable_index=most_probable_index,
        most_probable_sample=most_probable_sample,
        most_probable_probability=(
            statevector_result.most_probable_probability
        ),
        most_probable_energy=most_probable_energy,
        n_qubits=hamiltonian.n_qubits,
        variable_order=hamiltonian.variable_order,
        statevector_fingerprint=(
            statevector_result.fingerprint()
        ),
        hamiltonian_fingerprint=hamiltonian.fingerprint(),
        operator_fingerprint=artifact.fingerprint(),
        config_fingerprint=run_config.fingerprint(),
        metadata={
            "framework": "CSSF",
            "algorithm": MAQAOA_ALGORITHM_NAME,
            "execution_engine": "qiskit_aer.AerSimulator",
            "execution_method": "statevector",
            "execution_device": "GPU",
            "expectation_engine": (
                "qiskit.quantum_info.Statevector.expectation_value"
            ),
            "operator_class": REQUIRED_OPERATOR_CLASS,
            "basis_order": "qiskit_little_endian",
            "basis_chunk_size": run_config.basis_chunk_size,
            "qiskit_version": artifact.qiskit_version,
        },
    )


def evaluate_maqaoa_parameters_gpu(
    artifact: MAQAOACircuitArtifact,
    parameters: MAQAOAParameterValues,
    hamiltonian: IsingHamiltonian,
    *,
    aer_config: MAQAOAAerGPUConfig | None = None,
    expectation_config: MAQAOAExpectationConfig | None = None,
    operator_artifact: MAQAOACostOperatorArtifact | None = None,
) -> MAQAOAGPUEvaluation:
    """Run the parameterized MA-QAOA circuit on Aer GPU and evaluate exact cost."""

    if not isinstance(artifact, MAQAOACircuitArtifact):
        raise TypeError(
            "artifact must be MAQAOACircuitArtifact."
        )
    if not isinstance(parameters, MAQAOAParameterValues):
        raise TypeError(
            "parameters must be MAQAOAParameterValues."
        )
    if artifact.plan.hamiltonian.fingerprint() != hamiltonian.fingerprint():
        raise MAQAOAExpectationError(
            "Circuit artifact and Hamiltonian differ."
        )

    statevector_result = run_maqaoa_statevector_gpu(
        artifact,
        parameters,
        config=aer_config,
    )
    expectation_result = evaluate_maqaoa_statevector_expectation(
        statevector_result,
        hamiltonian,
        config=expectation_config,
        operator_artifact=operator_artifact,
    )
    return MAQAOAGPUEvaluation(
        parameters=parameters,
        statevector_result=statevector_result,
        expectation_result=expectation_result,
    )


__all__ = [
    "REAL_DTYPE",
    "COMPLEX_DTYPE",
    "DEFAULT_EXPECTATION_ATOL",
    "DEFAULT_IMAGINARY_ATOL",
    "DEFAULT_VARIANCE_ATOL",
    "DEFAULT_GROUND_ENERGY_ATOL",
    "DEFAULT_BASIS_CHUNK_SIZE",
    "REQUIRED_OPERATOR_CLASS",
    "REQUIRED_STATE_CLASS",
    "MAQAOAExpectationError",
    "MAQAOAQiskitExpectationUnavailableError",
    "MAQAOAQiskitExpectationResultError",
    "MAQAOAExpectationConfig",
    "MAQAOACostOperatorArtifact",
    "MAQAOAExpectationResult",
    "MAQAOAGPUEvaluation",
    "build_maqaoa_qiskit_cost_operator",
    "evaluate_maqaoa_statevector_expectation",
    "evaluate_maqaoa_parameters_gpu",
]
