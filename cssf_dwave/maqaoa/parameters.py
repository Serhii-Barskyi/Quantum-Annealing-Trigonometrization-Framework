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

"""Deterministic multi-angle QAOA parameter layouts and values.

The module assigns one trainable cost angle to every non-identity Ising Z/ZZ
term in every layer and one trainable mixer angle to every qubit in every
layer. It performs exact parameter algebra only; quantum-circuit construction,
statevector execution, expectation evaluation, and sampling remain delegated
to Qiskit and ``qiskit-aer-gpu`` in the corresponding execution modules.

Flat parameter order is stable and explicit:

1. all ``gamma[layer, cost_term]`` values in C row-major order;
2. all ``beta[layer, qubit]`` values in C row-major order.

No parameter sharing, coefficient absorption, hidden reordering, or numerical
rounding is performed.

Exact statevector limits are enforced only by execution backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from maqaoa import (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    DEFAULT_INITIAL_COST_ANGLE,
    DEFAULT_INITIAL_MIXER_ANGLE,
    DEFAULT_REPETITIONS,
    parameter_count,
)
from qaoa.hamiltonian import IsingHamiltonian, PauliZTerm


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
DEFAULT_GAMMA_BOUNDS: Final[tuple[float, float]] = (
    0.0,
    2.0 * math.pi,
)
DEFAULT_BETA_BOUNDS: Final[tuple[float, float]] = (
    0.0,
    math.pi,
)


class MAQAOAParameterError(ValueError):
    """Raised when a multi-angle parameter layout or value is invalid."""


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise MAQAOAParameterError(f"{name} must be positive.")
    return value


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MAQAOAParameterError(f"{name} must be finite.")
    return normalized


def _readonly_matrix(
    values: ArrayLike,
    *,
    shape: tuple[int, int],
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.shape != shape:
        raise MAQAOAParameterError(
            f"{name} has shape {array.shape}; expected {shape}."
        )
    if not np.all(np.isfinite(array)):
        raise MAQAOAParameterError(
            f"{name} contains non-finite values."
        )
    result = np.array(array, dtype=REAL_DTYPE, order="C", copy=True)
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _readonly_vector(
    values: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.ndim != 1 or array.size != expected_size:
        raise MAQAOAParameterError(
            f"{name} must be a one-dimensional vector of length "
            f"{expected_size}; received shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise MAQAOAParameterError(
            f"{name} contains non-finite values."
        )
    result = np.array(array, dtype=REAL_DTYPE, order="C", copy=True)
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _normalized_bounds(
    bounds: Sequence[float],
    *,
    name: str,
) -> tuple[float, float]:
    if len(bounds) != 2:
        raise MAQAOAParameterError(
            f"{name} must contain exactly two values."
        )
    lower = _finite_float(bounds[0], name=f"{name}[0]")
    upper = _finite_float(bounds[1], name=f"{name}[1]")
    if lower >= upper:
        raise MAQAOAParameterError(
            f"{name} lower bound must be less than upper bound."
        )
    return lower, upper


@dataclass(frozen=True, slots=True)
class MAQAOACostTerm:
    """One deterministic trainable Ising cost-term descriptor."""

    term_index: int
    qubits: tuple[int, ...]
    variables: tuple[str, ...]
    coefficient: float
    qiskit_label: str

    def __post_init__(self) -> None:
        index = self.term_index
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("term_index must be an integer.")
        if index < 0:
            raise MAQAOAParameterError(
                "term_index must be non-negative."
            )

        qubits = tuple(self.qubits)
        variables = tuple(str(item) for item in self.variables)
        if len(qubits) not in (1, 2):
            raise MAQAOAParameterError(
                "A trainable cost term must act on one or two qubits."
            )
        if len(variables) != len(qubits):
            raise MAQAOAParameterError(
                "variables must correspond one-to-one with qubits."
            )
        if tuple(sorted(qubits)) != qubits or len(set(qubits)) != len(
            qubits
        ):
            raise MAQAOAParameterError(
                "qubits must be sorted and unique."
            )
        if any(
            isinstance(qubit, bool)
            or not isinstance(qubit, int)
            or qubit < 0
            for qubit in qubits
        ):
            raise MAQAOAParameterError(
                "qubits must be non-negative integers."
            )
        if any(not variable for variable in variables):
            raise MAQAOAParameterError(
                "variables must contain non-empty labels."
            )

        coefficient = _finite_float(
            self.coefficient,
            name="coefficient",
        )
        if coefficient == 0.0:
            raise MAQAOAParameterError(
                "Trainable cost-term coefficient must be non-zero."
            )

        label = str(self.qiskit_label)
        if not label or any(symbol not in {"I", "Z"} for symbol in label):
            raise MAQAOAParameterError(
                "qiskit_label must contain only I and Z symbols."
            )
        if label.count("Z") != len(qubits):
            raise MAQAOAParameterError(
                "qiskit_label support does not match qubits."
            )

        object.__setattr__(self, "term_index", index)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "qiskit_label", label)

    @property
    def locality(self) -> int:
        return len(self.qubits)

    @property
    def parameter_suffix(self) -> str:
        return "_".join(str(qubit) for qubit in self.qubits)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOACostTerm-v1\0")
        digest.update(
            json.dumps(
                {
                    "term_index": self.term_index,
                    "qubits": self.qubits,
                    "variables": self.variables,
                    "coefficient": self.coefficient,
                    "qiskit_label": self.qiskit_label,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class MAQAOAParameterValues:
    """Immutable independent cost and mixer angles for all layers."""

    gamma: NDArray[np.float64]
    beta: NDArray[np.float64]

    def __init__(
        self,
        *,
        gamma: ArrayLike,
        beta: ArrayLike,
        gamma_shape: tuple[int, int] | None = None,
        beta_shape: tuple[int, int] | None = None,
    ) -> None:
        gamma_array = np.asarray(gamma, dtype=REAL_DTYPE)
        beta_array = np.asarray(beta, dtype=REAL_DTYPE)

        if gamma_shape is None:
            if gamma_array.ndim != 2:
                raise MAQAOAParameterError(
                    "gamma must be a two-dimensional matrix."
                )
            normalized_gamma_shape = tuple(gamma_array.shape)
        else:
            normalized_gamma_shape = tuple(gamma_shape)

        if beta_shape is None:
            if beta_array.ndim != 2:
                raise MAQAOAParameterError(
                    "beta must be a two-dimensional matrix."
                )
            normalized_beta_shape = tuple(beta_array.shape)
        else:
            normalized_beta_shape = tuple(beta_shape)

        if len(normalized_gamma_shape) != 2 or len(
            normalized_beta_shape
        ) != 2:
            raise MAQAOAParameterError(
                "gamma_shape and beta_shape must be matrix shapes."
            )
        if normalized_gamma_shape[0] != normalized_beta_shape[0]:
            raise MAQAOAParameterError(
                "gamma and beta must contain the same layer count."
            )
        for name, shape in (
            ("gamma_shape", normalized_gamma_shape),
            ("beta_shape", normalized_beta_shape),
        ):
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in shape
            ):
                raise MAQAOAParameterError(
                    f"{name} entries must be positive integers."
                )

        object.__setattr__(
            self,
            "gamma",
            _readonly_matrix(
                gamma_array,
                shape=normalized_gamma_shape,
                name="gamma",
            ),
        )
        object.__setattr__(
            self,
            "beta",
            _readonly_matrix(
                beta_array,
                shape=normalized_beta_shape,
                name="beta",
            ),
        )

    @property
    def repetitions(self) -> int:
        return int(self.gamma.shape[0])

    @property
    def cost_term_count(self) -> int:
        return int(self.gamma.shape[1])

    @property
    def n_qubits(self) -> int:
        return int(self.beta.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(self.gamma.size + self.beta.size)

    def flat(self) -> NDArray[np.float64]:
        """Return all gamma values followed by all beta values."""

        result = np.concatenate(
            (
                self.gamma.reshape(-1, order="C"),
                self.beta.reshape(-1, order="C"),
            )
        ).astype(REAL_DTYPE, copy=False)
        result.setflags(write=False)
        return result

    def layer(self, layer_index: int) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return immutable gamma and beta vectors for one layer."""

        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer.")
        if layer_index < 0 or layer_index >= self.repetitions:
            raise MAQAOAParameterError(
                "layer_index lies outside the parameter tensor."
            )
        gamma_layer = np.array(
            self.gamma[layer_index],
            dtype=REAL_DTYPE,
            copy=True,
        )
        beta_layer = np.array(
            self.beta[layer_index],
            dtype=REAL_DTYPE,
            copy=True,
        )
        gamma_layer.setflags(write=False)
        beta_layer.setflags(write=False)
        return gamma_layer, beta_layer

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAParameterValues-v1\0")
        digest.update(
            np.asarray(
                [
                    self.repetitions,
                    self.cost_term_count,
                    self.n_qubits,
                ],
                dtype=np.int64,
            ).tobytes(order="C")
        )
        digest.update(self.gamma.tobytes(order="C"))
        digest.update(self.beta.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MAQAOAParameterLayout:
    """Deterministic independent-angle layout for one Ising Hamiltonian."""

    repetitions: int
    variable_order: tuple[str, ...]
    cost_terms: tuple[MAQAOACostTerm, ...]
    hamiltonian_fingerprint: str

    def __post_init__(self) -> None:
        repetitions = _positive_integer(
            self.repetitions,
            name="repetitions",
        )
        variable_order = tuple(str(item) for item in self.variable_order)
        if not variable_order or any(not item for item in variable_order):
            raise MAQAOAParameterError(
                "variable_order must contain non-empty labels."
            )
        if len(set(variable_order)) != len(variable_order):
            raise MAQAOAParameterError(
                "variable_order must contain unique labels."
            )

        terms = tuple(self.cost_terms)
        if not terms:
            raise MAQAOAParameterError(
                "MA-QAOA requires at least one non-identity Ising term."
            )
        if any(not isinstance(term, MAQAOACostTerm) for term in terms):
            raise TypeError(
                "cost_terms must contain MAQAOACostTerm objects."
            )
        expected_indices = tuple(range(len(terms)))
        actual_indices = tuple(term.term_index for term in terms)
        if actual_indices != expected_indices:
            raise MAQAOAParameterError(
                "cost_terms must use contiguous deterministic indices."
            )
        if any(term.qubits[-1] >= len(variable_order) for term in terms):
            raise MAQAOAParameterError(
                "A cost term references a qubit outside variable_order."
            )
        if len({term.qubits for term in terms}) != len(terms):
            raise MAQAOAParameterError(
                "cost_terms contain duplicate Pauli supports."
            )

        digest = str(self.hamiltonian_fingerprint)
        if len(digest) != 64:
            raise MAQAOAParameterError(
                "hamiltonian_fingerprint must be a SHA-256 digest."
            )

        parameter_count(
            repetitions=repetitions,
            cost_term_count=len(terms),
            n_qubits=len(variable_order),
        )

        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "variable_order", variable_order)
        object.__setattr__(self, "cost_terms", terms)
        object.__setattr__(self, "hamiltonian_fingerprint", digest)

    @property
    def n_qubits(self) -> int:
        return len(self.variable_order)

    @property
    def cost_term_count(self) -> int:
        return len(self.cost_terms)

    @property
    def gamma_shape(self) -> tuple[int, int]:
        return self.repetitions, self.cost_term_count

    @property
    def beta_shape(self) -> tuple[int, int]:
        return self.repetitions, self.n_qubits

    @property
    def gamma_parameter_count(self) -> int:
        return self.repetitions * self.cost_term_count

    @property
    def beta_parameter_count(self) -> int:
        return self.repetitions * self.n_qubits

    @property
    def total_parameter_count(self) -> int:
        return parameter_count(
            repetitions=self.repetitions,
            cost_term_count=self.cost_term_count,
            n_qubits=self.n_qubits,
        )

    @property
    def gamma_slice(self) -> slice:
        return slice(0, self.gamma_parameter_count)

    @property
    def beta_slice(self) -> slice:
        return slice(
            self.gamma_parameter_count,
            self.total_parameter_count,
        )

    def gamma_name(self, layer: int, cost_term: int) -> str:
        self._validate_layer(layer)
        if isinstance(cost_term, bool) or not isinstance(cost_term, int):
            raise TypeError("cost_term must be an integer.")
        if cost_term < 0 or cost_term >= self.cost_term_count:
            raise MAQAOAParameterError(
                "cost_term lies outside the parameter layout."
            )
        descriptor = self.cost_terms[cost_term]
        return (
            f"gamma[{layer},{cost_term}]"
            f"_z_{descriptor.parameter_suffix}"
        )

    def beta_name(self, layer: int, qubit: int) -> str:
        self._validate_layer(layer)
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise TypeError("qubit must be an integer.")
        if qubit < 0 or qubit >= self.n_qubits:
            raise MAQAOAParameterError(
                "qubit lies outside the parameter layout."
            )
        return f"beta[{layer},{qubit}]_x_{self.variable_order[qubit]}"

    def parameter_names(self) -> tuple[str, ...]:
        gamma_names = tuple(
            self.gamma_name(layer, term)
            for layer in range(self.repetitions)
            for term in range(self.cost_term_count)
        )
        beta_names = tuple(
            self.beta_name(layer, qubit)
            for layer in range(self.repetitions)
            for qubit in range(self.n_qubits)
        )
        return gamma_names + beta_names

    def split(self, values: ArrayLike) -> MAQAOAParameterValues:
        """Split one flat vector according to the deterministic layout."""

        flat = _readonly_vector(
            values,
            expected_size=self.total_parameter_count,
            name="values",
        )
        gamma = flat[self.gamma_slice].reshape(
            self.gamma_shape,
            order="C",
        )
        beta = flat[self.beta_slice].reshape(
            self.beta_shape,
            order="C",
        )
        return MAQAOAParameterValues(
            gamma=gamma,
            beta=beta,
            gamma_shape=self.gamma_shape,
            beta_shape=self.beta_shape,
        )

    def values(
        self,
        *,
        gamma: ArrayLike,
        beta: ArrayLike,
    ) -> MAQAOAParameterValues:
        """Validate matrices against this exact layout."""

        return MAQAOAParameterValues(
            gamma=gamma,
            beta=beta,
            gamma_shape=self.gamma_shape,
            beta_shape=self.beta_shape,
        )

    def initial_values(
        self,
        *,
        cost_angle: float = DEFAULT_INITIAL_COST_ANGLE,
        mixer_angle: float = DEFAULT_INITIAL_MIXER_ANGLE,
    ) -> MAQAOAParameterValues:
        """Create deterministic constant-valued initial angles."""

        normalized_gamma = _finite_float(
            cost_angle,
            name="cost_angle",
        )
        normalized_beta = _finite_float(
            mixer_angle,
            name="mixer_angle",
        )
        return self.values(
            gamma=np.full(
                self.gamma_shape,
                normalized_gamma,
                dtype=REAL_DTYPE,
            ),
            beta=np.full(
                self.beta_shape,
                normalized_beta,
                dtype=REAL_DTYPE,
            ),
        )

    def bounds(
        self,
        *,
        gamma_bounds: Sequence[float] = DEFAULT_GAMMA_BOUNDS,
        beta_bounds: Sequence[float] = DEFAULT_BETA_BOUNDS,
    ) -> tuple[tuple[float, float], ...]:
        """Return flat optimizer bounds in the exact parameter order."""

        gamma_pair = _normalized_bounds(
            gamma_bounds,
            name="gamma_bounds",
        )
        beta_pair = _normalized_bounds(
            beta_bounds,
            name="beta_bounds",
        )
        return (
            (gamma_pair,) * self.gamma_parameter_count
            + (beta_pair,) * self.beta_parameter_count
        )

    def validate_hamiltonian(self, hamiltonian: IsingHamiltonian) -> None:
        """Reject use with a foreign or modified Ising Hamiltonian."""

        if not isinstance(hamiltonian, IsingHamiltonian):
            raise TypeError("hamiltonian must be IsingHamiltonian.")
        if hamiltonian.fingerprint() != self.hamiltonian_fingerprint:
            raise MAQAOAParameterError(
                "Parameter layout does not match the supplied Hamiltonian."
            )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAParameterLayout-v1\0")
        digest.update(
            json.dumps(
                {
                    "repetitions": self.repetitions,
                    "variable_order": self.variable_order,
                    "cost_term_fingerprints": tuple(
                        term.fingerprint() for term in self.cost_terms
                    ),
                    "hamiltonian_fingerprint": (
                        self.hamiltonian_fingerprint
                    ),
                    "flat_order": (
                        "gamma[layer,cost_term]",
                        "beta[layer,qubit]",
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def _validate_layer(self, layer: int) -> None:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer.")
        if layer < 0 or layer >= self.repetitions:
            raise MAQAOAParameterError(
                "layer lies outside the parameter layout."
            )


def _cost_term_descriptor(
    term_index: int,
    term: PauliZTerm,
    hamiltonian: IsingHamiltonian,
) -> MAQAOACostTerm:
    return MAQAOACostTerm(
        term_index=term_index,
        qubits=term.qubits,
        variables=tuple(
            hamiltonian.variable_order[qubit]
            for qubit in term.qubits
        ),
        coefficient=term.coefficient,
        qiskit_label=term.qiskit_label(hamiltonian.n_qubits),
    )


def build_maqaoa_parameter_layout(
    hamiltonian: IsingHamiltonian,
    *,
    repetitions: int = DEFAULT_REPETITIONS,
) -> MAQAOAParameterLayout:
    """Build the exact independent-angle layout for an Ising Hamiltonian."""

    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError("hamiltonian must be IsingHamiltonian.")
    normalized_repetitions = _positive_integer(
        repetitions,
        name="repetitions",
    )

    terms = tuple(
        _cost_term_descriptor(index, term, hamiltonian)
        for index, term in enumerate(hamiltonian.pauli_z_terms())
    )
    return MAQAOAParameterLayout(
        repetitions=normalized_repetitions,
        variable_order=hamiltonian.variable_order,
        cost_terms=terms,
        hamiltonian_fingerprint=hamiltonian.fingerprint(),
    )


__all__ = [
    "REAL_DTYPE",
    "DEFAULT_GAMMA_BOUNDS",
    "DEFAULT_BETA_BOUNDS",
    "MAQAOAParameterError",
    "MAQAOACostTerm",
    "MAQAOAParameterValues",
    "MAQAOAParameterLayout",
    "build_maqaoa_parameter_layout",
]
