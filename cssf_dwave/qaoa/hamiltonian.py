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

"""Exact solver-independent conversion between QUBO and Ising Hamiltonians.

The QUBO convention is

    E_Q(x) = offset_Q + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j,

for binary variables ``x_i in {0, 1}``. The Ising convention is

    E_I(z) = offset_I + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j,

for spin variables ``z_i in {-1, +1}``, with the exact substitution

    x_i = (1 - z_i) / 2.

All matrices are stored strictly upper triangular. The module contains no
Qiskit, Aer, dimod, sampler, filesystem, or network side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qubo.model import (
    BINARY_TOLERANCE,
    DEFAULT_ZERO_TOLERANCE,
    QUBOModel,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
SPIN_TOLERANCE: Final[float] = 1.0e-9
DEFAULT_EXACT_AUDIT_LIMIT: Final[int] = 18
DEFAULT_RANDOM_AUDIT_SAMPLES: Final[int] = 4096
DEFAULT_AUDIT_ATOL: Final[float] = 1.0e-10
MAX_EXACT_BASIS_ENUMERATION_QUBITS: Final[int] = 22
DEFAULT_BASIS_ENUMERATION_LIMIT: Final[int] = (
    MAX_EXACT_BASIS_ENUMERATION_QUBITS
)

BasisOrder = Literal["qiskit", "lexicographic"]


class IsingHamiltonianError(ValueError):
    """Raised when an Ising Hamiltonian or state violates its contract."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)

    if not math.isfinite(normalized):
        raise IsingHamiltonianError(f"{name} must be finite.")

    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)

    if normalized <= 0.0:
        raise IsingHamiltonianError(
            f"{name} must be strictly positive."
        )

    return normalized


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise IsingHamiltonianError(
            f"{name} must be strictly positive."
        )
    return value


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise IsingHamiltonianError(
            f"{name} must be non-negative."
        )
    return value


def _variable_order(values: Sequence[str]) -> tuple[str, ...]:
    variables = tuple(str(value).strip() for value in values)

    if not variables:
        raise IsingHamiltonianError(
            "variable_order must contain at least one label."
        )
    if any(not variable for variable in variables):
        raise IsingHamiltonianError(
            "variable_order must not contain empty labels."
        )
    if len(set(variables)) != len(variables):
        raise IsingHamiltonianError(
            "variable_order must contain unique labels."
        )

    return variables


def _readonly_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_size: int,
    zero_tolerance: float,
) -> NDArray[np.float64]:
    result = np.ascontiguousarray(
        np.asarray(values, dtype=REAL_DTYPE).reshape(-1),
        dtype=REAL_DTYPE,
    )

    if result.size != expected_size:
        raise IsingHamiltonianError(
            f"{name} must contain {expected_size} values; "
            f"received {result.size}."
        )
    if not np.all(np.isfinite(result)):
        raise IsingHamiltonianError(
            f"{name} contains non-finite values."
        )

    result[np.abs(result) <= zero_tolerance] = 0.0
    result.setflags(write=False)
    return result


def _readonly_upper_matrix(
    values: ArrayLike,
    *,
    name: str,
    n_variables: int,
    zero_tolerance: float,
) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=REAL_DTYPE)

    if matrix.shape != (n_variables, n_variables):
        raise IsingHamiltonianError(
            f"{name} must have shape "
            f"({n_variables}, {n_variables}); "
            f"received {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise IsingHamiltonianError(
            f"{name} contains non-finite values."
        )
    if np.any(np.abs(np.tril(matrix, k=-1)) > zero_tolerance):
        raise IsingHamiltonianError(
            f"{name} must be strictly upper triangular."
        )
    if np.any(np.abs(np.diag(matrix)) > zero_tolerance):
        raise IsingHamiltonianError(
            f"{name} diagonal must be zero."
        )

    result = np.ascontiguousarray(
        np.triu(matrix, k=1),
        dtype=REAL_DTYPE,
    )
    result[np.abs(result) <= zero_tolerance] = 0.0
    result.setflags(write=False)
    return result


def _binary_matrix(
    values: ArrayLike,
    *,
    tolerance: float,
    expected_width: int | None = None,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise IsingHamiltonianError(
            "Binary states must be one- or two-dimensional."
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise IsingHamiltonianError(
            "Binary states must be non-empty."
        )
    if expected_width is not None and array.shape[1] != expected_width:
        raise IsingHamiltonianError(
            f"Binary states contain {array.shape[1]} variables; "
            f"expected {expected_width}."
        )
    if not np.all(np.isfinite(array)):
        raise IsingHamiltonianError(
            "Binary states contain non-finite values."
        )

    close_zero = np.abs(array) <= tolerance
    close_one = np.abs(array - 1.0) <= tolerance

    if not np.all(close_zero | close_one):
        raise IsingHamiltonianError(
            "Binary states must contain only zero and one."
        )

    return np.ascontiguousarray(
        np.where(close_one, 1.0, 0.0),
        dtype=REAL_DTYPE,
    )


def _spin_matrix(
    values: ArrayLike,
    *,
    tolerance: float,
    expected_width: int | None = None,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise IsingHamiltonianError(
            "Spin states must be one- or two-dimensional."
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise IsingHamiltonianError(
            "Spin states must be non-empty."
        )
    if expected_width is not None and array.shape[1] != expected_width:
        raise IsingHamiltonianError(
            f"Spin states contain {array.shape[1]} variables; "
            f"expected {expected_width}."
        )
    if not np.all(np.isfinite(array)):
        raise IsingHamiltonianError(
            "Spin states contain non-finite values."
        )

    close_minus = np.abs(array + 1.0) <= tolerance
    close_plus = np.abs(array - 1.0) <= tolerance

    if not np.all(close_minus | close_plus):
        raise IsingHamiltonianError(
            "Spin states must contain only -1 and +1."
        )

    return np.ascontiguousarray(
        np.where(close_plus, 1.0, -1.0),
        dtype=REAL_DTYPE,
    )


def binary_to_spin(
    binary: ArrayLike,
    *,
    tolerance: float = BINARY_TOLERANCE,
) -> NDArray[np.float64]:
    """Apply ``z = 1 - 2x`` while preserving one/batch dimensionality."""

    normalized_tolerance = _positive_float(
        tolerance,
        name="tolerance",
    )
    original_ndim = np.asarray(binary).ndim
    matrix = _binary_matrix(
        binary,
        tolerance=normalized_tolerance,
    )
    result = np.ascontiguousarray(
        1.0 - 2.0 * matrix,
        dtype=REAL_DTYPE,
    )

    if original_ndim == 1:
        result = result[0]

    result.setflags(write=False)
    return result


def spin_to_binary(
    spins: ArrayLike,
    *,
    tolerance: float = SPIN_TOLERANCE,
) -> NDArray[np.float64]:
    """Apply ``x = (1 - z) / 2`` while preserving dimensionality."""

    normalized_tolerance = _positive_float(
        tolerance,
        name="tolerance",
    )
    original_ndim = np.asarray(spins).ndim
    matrix = _spin_matrix(
        spins,
        tolerance=normalized_tolerance,
    )
    result = np.ascontiguousarray(
        0.5 * (1.0 - matrix),
        dtype=REAL_DTYPE,
    )

    if original_ndim == 1:
        result = result[0]

    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PauliZTerm:
    """One non-identity diagonal Pauli-Z term."""

    qubits: tuple[int, ...]
    coefficient: float

    def __post_init__(self) -> None:
        qubits = tuple(self.qubits)

        if len(qubits) not in (1, 2):
            raise IsingHamiltonianError(
                "PauliZTerm must act on one or two qubits."
            )
        if any(
            isinstance(qubit, bool)
            or not isinstance(qubit, int)
            or qubit < 0
            for qubit in qubits
        ):
            raise IsingHamiltonianError(
                "PauliZTerm qubits must be non-negative integers."
            )
        if tuple(sorted(qubits)) != qubits:
            raise IsingHamiltonianError(
                "PauliZTerm qubits must be sorted."
            )
        if len(set(qubits)) != len(qubits):
            raise IsingHamiltonianError(
                "PauliZTerm qubits must be unique."
            )

        coefficient = _finite_float(
            self.coefficient,
            name="coefficient",
        )
        if coefficient == 0.0:
            raise IsingHamiltonianError(
                "PauliZTerm coefficient must be non-zero."
            )

        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "coefficient", coefficient)

    def qiskit_label(self, n_qubits: int) -> str:
        """Return a Qiskit Pauli label, where qubit zero is rightmost."""

        normalized_count = _positive_integer(
            n_qubits,
            name="n_qubits",
        )

        if self.qubits[-1] >= normalized_count:
            raise IsingHamiltonianError(
                "Pauli term references a qubit outside n_qubits."
            )

        label = ["I"] * normalized_count
        for qubit in self.qubits:
            label[normalized_count - 1 - qubit] = "Z"
        return "".join(label)


@dataclass(frozen=True, slots=True, init=False)
class IsingHamiltonian:
    """Immutable diagonal Ising cost Hamiltonian."""

    variable_order: tuple[str, ...]
    linear_z: NDArray[np.float64]
    quadratic_zz: NDArray[np.float64]
    offset: float
    zero_tolerance: float

    def __init__(
        self,
        *,
        variable_order: Sequence[str],
        linear_z: ArrayLike,
        quadratic_zz: ArrayLike,
        offset: float = 0.0,
        zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
    ) -> None:
        variables = _variable_order(variable_order)
        tolerance = _positive_float(
            zero_tolerance,
            name="zero_tolerance",
        )
        linear = _readonly_vector(
            linear_z,
            name="linear_z",
            expected_size=len(variables),
            zero_tolerance=tolerance,
        )
        quadratic = _readonly_upper_matrix(
            quadratic_zz,
            name="quadratic_zz",
            n_variables=len(variables),
            zero_tolerance=tolerance,
        )
        normalized_offset = _finite_float(
            offset,
            name="offset",
        )

        object.__setattr__(self, "variable_order", variables)
        object.__setattr__(self, "linear_z", linear)
        object.__setattr__(self, "quadratic_zz", quadratic)
        object.__setattr__(self, "offset", normalized_offset)
        object.__setattr__(self, "zero_tolerance", tolerance)

    @property
    def n_qubits(self) -> int:
        return len(self.variable_order)

    @property
    def n_interactions(self) -> int:
        return int(np.count_nonzero(self.quadratic_zz))

    @classmethod
    def from_qubo(cls, model: QUBOModel) -> "IsingHamiltonian":
        """Convert a QUBO model exactly under ``x=(1-z)/2``."""

        if not isinstance(model, QUBOModel):
            raise TypeError("model must be QUBOModel.")

        pair_sum_by_variable = (
            np.sum(model.quadratic, axis=0)
            + np.sum(model.quadratic, axis=1)
        )
        linear_z = (
            -0.5 * model.linear
            - 0.25 * pair_sum_by_variable
        )
        quadratic_zz = 0.25 * model.quadratic
        offset = float(
            model.offset
            + 0.5 * np.sum(model.linear)
            + 0.25 * np.sum(model.quadratic)
        )

        return cls(
            variable_order=model.variable_order,
            linear_z=linear_z,
            quadratic_zz=quadratic_zz,
            offset=offset,
            zero_tolerance=model.zero_tolerance,
        )

    def to_qubo(self) -> QUBOModel:
        """Invert the Ising conversion exactly up to floating-point roundoff."""

        pair_sum_by_variable = (
            np.sum(self.quadratic_zz, axis=0)
            + np.sum(self.quadratic_zz, axis=1)
        )
        linear = -2.0 * (
            self.linear_z + pair_sum_by_variable
        )
        quadratic = 4.0 * self.quadratic_zz
        qubo_offset = float(
            self.offset
            - 0.5 * np.sum(linear)
            - 0.25 * np.sum(quadratic)
        )

        return QUBOModel(
            variable_order=self.variable_order,
            linear=linear,
            quadratic=quadratic,
            offset=qubo_offset,
            zero_tolerance=self.zero_tolerance,
        )

    def spin_vector(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
        *,
        tolerance: float = SPIN_TOLERANCE,
    ) -> NDArray[np.float64]:
        """Convert one labeled or ordered spin sample to model order."""

        normalized_tolerance = _positive_float(
            tolerance,
            name="tolerance",
        )

        if isinstance(sample, Mapping):
            expected = set(self.variable_order)
            supplied = set(sample)
            missing = expected - supplied
            extra = supplied - expected

            if missing or extra:
                raise IsingHamiltonianError(
                    "Spin labels differ from variable_order; "
                    f"missing={sorted(missing)!r}, "
                    f"extra={sorted(extra)!r}."
                )

            values = [
                sample[variable]
                for variable in self.variable_order
            ]
        else:
            values = sample

        result = _spin_matrix(
            values,
            tolerance=normalized_tolerance,
            expected_width=self.n_qubits,
        )[0]
        result.setflags(write=False)
        return result

    def energy(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
        *,
        tolerance: float = SPIN_TOLERANCE,
    ) -> float:
        """Evaluate one spin assignment."""

        spins = self.spin_vector(
            sample,
            tolerance=tolerance,
        )
        return float(
            self.offset
            + spins @ self.linear_z
            + spins @ self.quadratic_zz @ spins
        )

    def energies(
        self,
        samples: ArrayLike,
        *,
        tolerance: float = SPIN_TOLERANCE,
    ) -> NDArray[np.float64]:
        """Evaluate an ordered batch of spin assignments."""

        normalized_tolerance = _positive_float(
            tolerance,
            name="tolerance",
        )
        spins = _spin_matrix(
            samples,
            tolerance=normalized_tolerance,
            expected_width=self.n_qubits,
        )
        linear_energy = spins @ self.linear_z
        quadratic_energy = np.einsum(
            "bi,ij,bj->b",
            spins,
            self.quadratic_zz,
            spins,
            optimize=True,
        )

        return np.ascontiguousarray(
            self.offset + linear_energy + quadratic_energy,
            dtype=REAL_DTYPE,
        )

    def binary_energy(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
        *,
        tolerance: float = BINARY_TOLERANCE,
    ) -> float:
        """Evaluate one binary assignment through the Ising mapping."""

        if isinstance(sample, Mapping):
            expected = set(self.variable_order)
            supplied = set(sample)
            missing = expected - supplied
            extra = supplied - expected

            if missing or extra:
                raise IsingHamiltonianError(
                    "Binary labels differ from variable_order; "
                    f"missing={sorted(missing)!r}, "
                    f"extra={sorted(extra)!r}."
                )
            values: Any = [
                sample[variable]
                for variable in self.variable_order
            ]
        else:
            values = sample

        spins = binary_to_spin(
            values,
            tolerance=tolerance,
        )
        return self.energy(spins)

    def binary_energies(
        self,
        samples: ArrayLike,
        *,
        tolerance: float = BINARY_TOLERANCE,
    ) -> NDArray[np.float64]:
        """Evaluate an ordered batch of binary assignments."""

        binary = _binary_matrix(
            samples,
            tolerance=_positive_float(
                tolerance,
                name="tolerance",
            ),
            expected_width=self.n_qubits,
        )
        return self.energies(1.0 - 2.0 * binary)

    def pauli_z_terms(self) -> tuple[PauliZTerm, ...]:
        """Return non-zero Z and ZZ terms in deterministic order."""

        terms: list[PauliZTerm] = []

        for qubit, coefficient in enumerate(self.linear_z):
            value = float(coefficient)
            if value != 0.0:
                terms.append(
                    PauliZTerm(
                        qubits=(qubit,),
                        coefficient=value,
                    )
                )

        for first in range(self.n_qubits):
            for second in range(first + 1, self.n_qubits):
                value = float(
                    self.quadratic_zz[first, second]
                )
                if value != 0.0:
                    terms.append(
                        PauliZTerm(
                            qubits=(first, second),
                            coefficient=value,
                        )
                    )

        return tuple(terms)

    def basis_energies(
        self,
        *,
        order: BasisOrder = "qiskit",
        max_qubits: int = DEFAULT_BASIS_ENUMERATION_LIMIT,
    ) -> NDArray[np.float64]:
        """Return diagonal energies in a declared computational-basis order.

        ``qiskit`` uses integer statevector indexing with qubit zero as the
        least-significant bit. ``lexicographic`` makes the first variable the
        most-significant displayed bit.
        """

        normalized_limit = _positive_integer(
            max_qubits,
            name="max_qubits",
        )
        if normalized_limit > MAX_EXACT_BASIS_ENUMERATION_QUBITS:
            raise IsingHamiltonianError(
                "max_qubits cannot exceed the project-wide exact-basis "
                f"limit of {MAX_EXACT_BASIS_ENUMERATION_QUBITS}."
            )
        if self.n_qubits > normalized_limit:
            raise IsingHamiltonianError(
                f"Refusing to enumerate 2^{self.n_qubits} basis states; "
                f"max_qubits={normalized_limit}."
            )
        if order not in ("qiskit", "lexicographic"):
            raise IsingHamiltonianError(
                "order must be 'qiskit' or 'lexicographic'."
            )

        indices = np.arange(
            1 << self.n_qubits,
            dtype=np.uint64,
        )

        if order == "qiskit":
            shifts = np.arange(
                self.n_qubits,
                dtype=np.uint64,
            )
        else:
            shifts = np.arange(
                self.n_qubits - 1,
                -1,
                -1,
                dtype=np.int64,
            ).astype(np.uint64)

        binary = (
            (indices[:, None] >> shifts[None, :]) & 1
        ).astype(REAL_DTYPE)
        result = self.binary_energies(binary)
        result.setflags(write=False)
        return result

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 Hamiltonian fingerprint."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-IsingHamiltonian-v1\0")
        digest.update(
            json.dumps(
                self.variable_order,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(self.linear_z.tobytes(order="C"))
        digest.update(self.quadratic_zz.tobytes(order="C"))
        digest.update(
            np.asarray(
                [self.offset, self.zero_tolerance],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QUBOIsingAudit:
    """Numerical equivalence report for one QUBO-to-Ising conversion."""

    qubo_fingerprint: str
    ising_fingerprint: str
    checked_samples: int
    exhaustive: bool
    max_absolute_error: float
    tolerance: float
    equivalent: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("qubo_fingerprint", self.qubo_fingerprint),
            ("ising_fingerprint", self.ising_fingerprint),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise IsingHamiltonianError(
                    f"{name} must be a SHA-256 digest."
                )

        checked = _positive_integer(
            self.checked_samples,
            name="checked_samples",
        )
        error = _finite_float(
            self.max_absolute_error,
            name="max_absolute_error",
        )
        tolerance = _positive_float(
            self.tolerance,
            name="tolerance",
        )

        if error < 0.0:
            raise IsingHamiltonianError(
                "max_absolute_error must be non-negative."
            )
        if not isinstance(self.exhaustive, bool):
            raise TypeError("exhaustive must be boolean.")
        if not isinstance(self.equivalent, bool):
            raise TypeError("equivalent must be boolean.")
        if self.equivalent != (error <= tolerance):
            raise IsingHamiltonianError(
                "equivalent flag disagrees with error and tolerance."
            )

        object.__setattr__(self, "checked_samples", checked)
        object.__setattr__(self, "max_absolute_error", error)
        object.__setattr__(self, "tolerance", tolerance)


def qubo_to_ising(model: QUBOModel) -> IsingHamiltonian:
    """Public functional alias for :meth:`IsingHamiltonian.from_qubo`."""

    return IsingHamiltonian.from_qubo(model)


def ising_to_qubo(hamiltonian: IsingHamiltonian) -> QUBOModel:
    """Public functional alias for :meth:`IsingHamiltonian.to_qubo`."""

    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError("hamiltonian must be IsingHamiltonian.")
    return hamiltonian.to_qubo()


def _audit_samples(
    n_variables: int,
    *,
    exact_limit: int,
    random_samples: int,
    seed: int,
) -> tuple[NDArray[np.float64], bool]:
    if n_variables <= exact_limit:
        indices = np.arange(
            1 << n_variables,
            dtype=np.uint64,
        )
        shifts = np.arange(
            n_variables,
            dtype=np.uint64,
        )
        samples = (
            (indices[:, None] >> shifts[None, :]) & 1
        ).astype(REAL_DTYPE)
        return samples, True

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    deterministic = [
        np.zeros(n_variables, dtype=np.int8),
        np.ones(n_variables, dtype=np.int8),
    ]
    deterministic.extend(np.eye(n_variables, dtype=np.int8))
    deterministic.extend(1 - np.eye(n_variables, dtype=np.int8))

    rng = np.random.default_rng(seed)
    random_block = rng.integers(
        0,
        2,
        size=(random_samples, n_variables),
        dtype=np.int8,
    )
    samples = np.unique(
        np.vstack((*deterministic, random_block)),
        axis=0,
    ).astype(REAL_DTYPE)
    return samples, False


def audit_qubo_ising_equivalence(
    model: QUBOModel,
    hamiltonian: IsingHamiltonian | None = None,
    *,
    exact_limit: int = DEFAULT_EXACT_AUDIT_LIMIT,
    random_samples: int = DEFAULT_RANDOM_AUDIT_SAMPLES,
    seed: int = 0,
    tolerance: float = DEFAULT_AUDIT_ATOL,
) -> QUBOIsingAudit:
    """Audit energy equivalence exhaustively or by deterministic sampling."""

    if not isinstance(model, QUBOModel):
        raise TypeError("model must be QUBOModel.")

    converted = (
        IsingHamiltonian.from_qubo(model)
        if hamiltonian is None
        else hamiltonian
    )
    if not isinstance(converted, IsingHamiltonian):
        raise TypeError(
            "hamiltonian must be IsingHamiltonian or None."
        )
    if converted.variable_order != model.variable_order:
        raise IsingHamiltonianError(
            "QUBO and Ising variable orders differ."
        )

    normalized_exact_limit = _nonnegative_integer(
        exact_limit,
        name="exact_limit",
    )
    if normalized_exact_limit > MAX_EXACT_BASIS_ENUMERATION_QUBITS:
        raise IsingHamiltonianError(
            "exact_limit cannot exceed the project-wide exact-audit "
            f"limit of {MAX_EXACT_BASIS_ENUMERATION_QUBITS}."
        )
    normalized_random_samples = _positive_integer(
        random_samples,
        name="random_samples",
    )
    normalized_tolerance = _positive_float(
        tolerance,
        name="tolerance",
    )

    samples, exhaustive = _audit_samples(
        model.n_variables,
        exact_limit=normalized_exact_limit,
        random_samples=normalized_random_samples,
        seed=seed,
    )
    qubo_energies = model.energies(samples)
    ising_energies = converted.binary_energies(samples)
    maximum_error = float(
        np.max(np.abs(qubo_energies - ising_energies))
    )

    return QUBOIsingAudit(
        qubo_fingerprint=model.fingerprint(),
        ising_fingerprint=converted.fingerprint(),
        checked_samples=int(samples.shape[0]),
        exhaustive=exhaustive,
        max_absolute_error=maximum_error,
        tolerance=normalized_tolerance,
        equivalent=maximum_error <= normalized_tolerance,
    )


def require_qubo_ising_equivalence(
    model: QUBOModel,
    hamiltonian: IsingHamiltonian | None = None,
    **audit_kwargs: Any,
) -> QUBOIsingAudit:
    """Run an audit and raise when the energies are not equivalent."""

    report = audit_qubo_ising_equivalence(
        model,
        hamiltonian,
        **audit_kwargs,
    )

    if not report.equivalent:
        raise IsingHamiltonianError(
            "QUBO-to-Ising energy audit failed: "
            f"max_absolute_error={report.max_absolute_error}, "
            f"tolerance={report.tolerance}."
        )

    return report


__all__ = [
    "REAL_DTYPE",
    "SPIN_TOLERANCE",
    "DEFAULT_EXACT_AUDIT_LIMIT",
    "DEFAULT_RANDOM_AUDIT_SAMPLES",
    "DEFAULT_AUDIT_ATOL",
    "MAX_EXACT_BASIS_ENUMERATION_QUBITS",
    "DEFAULT_BASIS_ENUMERATION_LIMIT",
    "BasisOrder",
    "IsingHamiltonianError",
    "binary_to_spin",
    "spin_to_binary",
    "PauliZTerm",
    "IsingHamiltonian",
    "QUBOIsingAudit",
    "qubo_to_ising",
    "ising_to_qubo",
    "audit_qubo_ising_equivalence",
    "require_qubo_ising_equivalence",
]
