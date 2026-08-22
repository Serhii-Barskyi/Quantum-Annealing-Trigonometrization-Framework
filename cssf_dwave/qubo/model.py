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

"""Immutable solver-independent QUBO representation.

The objective convention is

    E(x) = x.T @ Q @ x + offset,

with binary ``x`` and an upper-triangular matrix ``Q``. Diagonal entries are
stored separately as linear biases because ``x_i**2 = x_i`` for binary
variables. Strictly upper-triangular entries are pairwise quadratic biases.

This module has no dependency on dimod or any sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
DEFAULT_ZERO_TOLERANCE: Final[float] = 1.0e-12
BINARY_TOLERANCE: Final[float] = 1.0e-9


class QUBOModelError(ValueError):
    """Raised when a QUBO model or binary sample is invalid."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)

    if not math.isfinite(normalized):
        raise QUBOModelError(f"{name} must be finite.")

    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)

    if normalized <= 0.0:
        raise QUBOModelError(
            f"{name} must be strictly positive."
        )

    return normalized


def _normalize_variable_order(
    variable_order: Sequence[str],
) -> tuple[str, ...]:
    variables = tuple(str(value).strip() for value in variable_order)

    if not variables:
        raise QUBOModelError(
            "variable_order must contain at least one variable."
        )
    if any(not value for value in variables):
        raise QUBOModelError(
            "variable_order must not contain empty labels."
        )
    if len(set(variables)) != len(variables):
        raise QUBOModelError(
            "variable_order must contain unique labels."
        )

    return variables


def _readonly_vector(
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
        raise QUBOModelError(
            f"{name} must contain {expected_size} values; "
            f"received {result.size}."
        )
    if not np.all(np.isfinite(result)):
        raise QUBOModelError(
            f"{name} contains non-finite values."
        )

    result.setflags(write=False)
    return result


def _readonly_quadratic(
    values: ArrayLike,
    *,
    n_variables: int,
    zero_tolerance: float,
) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=REAL_DTYPE)

    if matrix.shape != (n_variables, n_variables):
        raise QUBOModelError(
            "quadratic must have shape "
            f"({n_variables}, {n_variables}); "
            f"received {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise QUBOModelError(
            "quadratic contains non-finite values."
        )

    lower = np.tril(matrix, k=-1)
    diagonal = np.diag(matrix)

    if np.any(np.abs(lower) > zero_tolerance):
        raise QUBOModelError(
            "quadratic must be upper triangular."
        )
    if np.any(np.abs(diagonal) > zero_tolerance):
        raise QUBOModelError(
            "quadratic diagonal must be zero; "
            "use linear biases for self-interactions."
        )

    result = np.ascontiguousarray(
        np.triu(matrix, k=1),
        dtype=REAL_DTYPE,
    )
    result[np.abs(result) <= zero_tolerance] = 0.0
    result.setflags(write=False)
    return result


def _binary_matrix(
    samples: ArrayLike,
    *,
    n_variables: int,
    tolerance: float,
) -> NDArray[np.float64]:
    array = np.asarray(samples, dtype=REAL_DTYPE)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise QUBOModelError(
            "samples must be one- or two-dimensional."
        )

    if array.shape[0] == 0:
        raise QUBOModelError(
            "samples must contain at least one row."
        )
    if array.shape[1] != n_variables:
        raise QUBOModelError(
            f"samples contain {array.shape[1]} variables; "
            f"expected {n_variables}."
        )
    if not np.all(np.isfinite(array)):
        raise QUBOModelError(
            "samples contain non-finite values."
        )

    close_zero = np.abs(array) <= tolerance
    close_one = np.abs(array - 1.0) <= tolerance

    if not np.all(close_zero | close_one):
        raise QUBOModelError(
            "samples must contain only binary values."
        )

    return np.ascontiguousarray(
        np.where(close_one, 1.0, 0.0),
        dtype=REAL_DTYPE,
    )


@dataclass(frozen=True, slots=True, init=False)
class QUBOModel:
    """Immutable linear and pairwise binary objective."""

    variable_order: tuple[str, ...]
    linear: NDArray[np.float64]
    quadratic: NDArray[np.float64]
    offset: float
    zero_tolerance: float

    def __init__(
        self,
        *,
        variable_order: Sequence[str],
        linear: ArrayLike,
        quadratic: ArrayLike,
        offset: float = 0.0,
        zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
    ) -> None:
        variables = _normalize_variable_order(variable_order)
        tolerance = _positive_float(
            zero_tolerance,
            name="zero_tolerance",
        )
        linear_vector = _readonly_vector(
            linear,
            name="linear",
            expected_size=len(variables),
        )
        quadratic_matrix = _readonly_quadratic(
            quadratic,
            n_variables=len(variables),
            zero_tolerance=tolerance,
        )
        normalized_offset = _finite_float(
            offset,
            name="offset",
        )

        normalized_linear = np.array(
            linear_vector,
            dtype=REAL_DTYPE,
            copy=True,
        )
        normalized_linear[
            np.abs(normalized_linear) <= tolerance
        ] = 0.0
        normalized_linear.setflags(write=False)

        object.__setattr__(
            self,
            "variable_order",
            variables,
        )
        object.__setattr__(
            self,
            "linear",
            normalized_linear,
        )
        object.__setattr__(
            self,
            "quadratic",
            quadratic_matrix,
        )
        object.__setattr__(
            self,
            "offset",
            normalized_offset,
        )
        object.__setattr__(
            self,
            "zero_tolerance",
            tolerance,
        )

    @property
    def n_variables(self) -> int:
        return len(self.variable_order)

    @property
    def n_interactions(self) -> int:
        return int(np.count_nonzero(self.quadratic))

    @classmethod
    def zeros(
        cls,
        variable_order: Sequence[str],
        *,
        offset: float = 0.0,
        zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
    ) -> "QUBOModel":
        variables = _normalize_variable_order(variable_order)
        n_variables = len(variables)

        return cls(
            variable_order=variables,
            linear=np.zeros(n_variables, dtype=REAL_DTYPE),
            quadratic=np.zeros(
                (n_variables, n_variables),
                dtype=REAL_DTYPE,
            ),
            offset=offset,
            zero_tolerance=zero_tolerance,
        )

    @classmethod
    def from_coefficients(
        cls,
        *,
        linear: Mapping[str, float],
        quadratic: Mapping[tuple[str, str], float],
        offset: float = 0.0,
        variable_order: Sequence[str] | None = None,
        zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
    ) -> "QUBOModel":
        """Construct a model from labeled coefficient mappings."""

        if not isinstance(linear, Mapping):
            raise TypeError("linear must be a mapping.")
        if not isinstance(quadratic, Mapping):
            raise TypeError("quadratic must be a mapping.")

        normalized_linear = {
            str(label).strip(): _finite_float(
                value,
                name=f"linear[{label!r}]",
            )
            for label, value in linear.items()
        }

        if any(not label for label in normalized_linear):
            raise QUBOModelError(
                "linear contains an empty variable label."
            )

        referenced: set[str] = set(normalized_linear)
        normalized_pairs: list[tuple[str, str, float]] = []

        for pair, value in quadratic.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise QUBOModelError(
                    "Every quadratic key must be a pair of labels."
                )

            first = str(pair[0]).strip()
            second = str(pair[1]).strip()

            if not first or not second:
                raise QUBOModelError(
                    "quadratic contains an empty variable label."
                )

            coefficient = _finite_float(
                value,
                name=f"quadratic[{pair!r}]",
            )
            referenced.update((first, second))
            normalized_pairs.append(
                (first, second, coefficient)
            )

        if variable_order is None:
            variables = tuple(sorted(referenced))
        else:
            variables = _normalize_variable_order(variable_order)

            missing = referenced - set(variables)
            if missing:
                raise QUBOModelError(
                    "variable_order is missing referenced variables: "
                    f"{sorted(missing)!r}."
                )

        if not variables:
            raise QUBOModelError(
                "At least one referenced variable is required."
            )

        index = {
            variable: position
            for position, variable in enumerate(variables)
        }

        linear_vector = np.zeros(
            len(variables),
            dtype=REAL_DTYPE,
        )
        quadratic_matrix = np.zeros(
            (len(variables), len(variables)),
            dtype=REAL_DTYPE,
        )

        for variable, coefficient in normalized_linear.items():
            linear_vector[index[variable]] += coefficient

        for first, second, coefficient in normalized_pairs:
            first_index = index[first]
            second_index = index[second]

            if first_index == second_index:
                linear_vector[first_index] += coefficient
            else:
                lower_index = min(first_index, second_index)
                upper_index = max(first_index, second_index)
                quadratic_matrix[
                    lower_index,
                    upper_index,
                ] += coefficient

        return cls(
            variable_order=variables,
            linear=linear_vector,
            quadratic=quadratic_matrix,
            offset=offset,
            zero_tolerance=zero_tolerance,
        )

    @classmethod
    def from_dense_matrix(
        cls,
        matrix: ArrayLike,
        *,
        variable_order: Sequence[str],
        offset: float = 0.0,
        zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
    ) -> "QUBOModel":
        """Canonicalize any square matrix under ``x.T @ Q @ x``."""

        variables = _normalize_variable_order(variable_order)
        dense = np.asarray(matrix, dtype=REAL_DTYPE)

        if dense.shape != (len(variables), len(variables)):
            raise QUBOModelError(
                "matrix shape does not match variable_order."
            )
        if not np.all(np.isfinite(dense)):
            raise QUBOModelError(
                "matrix contains non-finite values."
            )

        linear = np.diag(dense).copy()
        quadratic = np.zeros_like(dense)

        for first in range(len(variables)):
            for second in range(first + 1, len(variables)):
                quadratic[first, second] = (
                    dense[first, second]
                    + dense[second, first]
                )

        return cls(
            variable_order=variables,
            linear=linear,
            quadratic=quadratic,
            offset=offset,
            zero_tolerance=zero_tolerance,
        )

    @classmethod
    def from_qubo_dict(
        cls,
        qubo: Mapping[tuple[str, str], float],
        *,
        offset: float = 0.0,
        variable_order: Sequence[str] | None = None,
        zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
    ) -> "QUBOModel":
        """Construct from the common ``{(u, v): bias}`` format."""

        if not isinstance(qubo, Mapping):
            raise TypeError("qubo must be a mapping.")

        linear: dict[str, float] = {}
        quadratic: dict[tuple[str, str], float] = {}

        for pair, coefficient in qubo.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise QUBOModelError(
                    "Every QUBO dictionary key must be a pair."
                )

            first = str(pair[0]).strip()
            second = str(pair[1]).strip()
            value = _finite_float(
                coefficient,
                name=f"qubo[{pair!r}]",
            )

            if first == second:
                linear[first] = linear.get(first, 0.0) + value
            else:
                quadratic[(first, second)] = (
                    quadratic.get((first, second), 0.0)
                    + value
                )

        return cls.from_coefficients(
            linear=linear,
            quadratic=quadratic,
            offset=offset,
            variable_order=variable_order,
            zero_tolerance=zero_tolerance,
        )

    def to_dense_matrix(self) -> NDArray[np.float64]:
        """Return upper-triangular ``Q`` including linear diagonal."""

        matrix = np.array(
            self.quadratic,
            dtype=REAL_DTYPE,
            copy=True,
        )
        np.fill_diagonal(matrix, self.linear)
        return np.ascontiguousarray(matrix, dtype=REAL_DTYPE)

    def to_qubo_dict(
        self,
    ) -> dict[tuple[str, str], float]:
        """Return non-zero labeled QUBO coefficients."""

        result: dict[tuple[str, str], float] = {}

        for index, variable in enumerate(self.variable_order):
            coefficient = float(self.linear[index])
            if coefficient != 0.0:
                result[(variable, variable)] = coefficient

        for first in range(self.n_variables):
            for second in range(first + 1, self.n_variables):
                coefficient = float(
                    self.quadratic[first, second]
                )
                if coefficient != 0.0:
                    result[
                        (
                            self.variable_order[first],
                            self.variable_order[second],
                        )
                    ] = coefficient

        return result

    def sample_vector(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
        *,
        binary_tolerance: float = BINARY_TOLERANCE,
    ) -> NDArray[np.float64]:
        """Convert one labeled or ordered binary sample to a vector."""

        tolerance = _positive_float(
            binary_tolerance,
            name="binary_tolerance",
        )

        if isinstance(sample, Mapping):
            missing = set(self.variable_order) - set(sample)
            extra = set(sample) - set(self.variable_order)

            if missing or extra:
                raise QUBOModelError(
                    "Sample labels differ from variable_order; "
                    f"missing={sorted(missing)!r}, "
                    f"extra={sorted(extra)!r}."
                )

            values = [
                sample[variable]
                for variable in self.variable_order
            ]
        else:
            values = sample

        return _binary_matrix(
            values,
            n_variables=self.n_variables,
            tolerance=tolerance,
        )[0]

    def energy(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
        *,
        binary_tolerance: float = BINARY_TOLERANCE,
    ) -> float:
        """Evaluate one binary sample."""

        vector = self.sample_vector(
            sample,
            binary_tolerance=binary_tolerance,
        )

        return float(
            self.offset
            + vector @ self.linear
            + vector @ self.quadratic @ vector
        )

    def energies(
        self,
        samples: ArrayLike,
        *,
        binary_tolerance: float = BINARY_TOLERANCE,
    ) -> NDArray[np.float64]:
        """Evaluate an ordered batch of binary samples."""

        tolerance = _positive_float(
            binary_tolerance,
            name="binary_tolerance",
        )
        matrix = _binary_matrix(
            samples,
            n_variables=self.n_variables,
            tolerance=tolerance,
        )

        linear_energy = matrix @ self.linear
        quadratic_energy = np.einsum(
            "bi,ij,bj->b",
            matrix,
            self.quadratic,
            matrix,
            optimize=True,
        )

        return np.ascontiguousarray(
            self.offset + linear_energy + quadratic_energy,
            dtype=REAL_DTYPE,
        )

    def scaled(self, factor: float) -> "QUBOModel":
        """Return a uniformly scaled objective."""

        normalized_factor = _finite_float(
            factor,
            name="factor",
        )

        return QUBOModel(
            variable_order=self.variable_order,
            linear=normalized_factor * self.linear,
            quadratic=normalized_factor * self.quadratic,
            offset=normalized_factor * self.offset,
            zero_tolerance=self.zero_tolerance,
        )

    def with_offset(self, offset: float) -> "QUBOModel":
        """Return the same biases with a replaced constant offset."""

        return QUBOModel(
            variable_order=self.variable_order,
            linear=self.linear,
            quadratic=self.quadratic,
            offset=offset,
            zero_tolerance=self.zero_tolerance,
        )

    def add(self, other: "QUBOModel") -> "QUBOModel":
        """Add two models with exactly the same variable order."""

        if not isinstance(other, QUBOModel):
            raise TypeError("other must be QUBOModel.")
        if other.variable_order != self.variable_order:
            raise QUBOModelError(
                "Models must have identical variable_order."
            )

        return QUBOModel(
            variable_order=self.variable_order,
            linear=self.linear + other.linear,
            quadratic=self.quadratic + other.quadratic,
            offset=self.offset + other.offset,
            zero_tolerance=min(
                self.zero_tolerance,
                other.zero_tolerance,
            ),
        )

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 model fingerprint."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-QUBOModel-v1\0")
        digest.update(
            json.dumps(
                self.variable_order,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(self.linear.tobytes(order="C"))
        digest.update(self.quadratic.tobytes(order="C"))
        digest.update(
            np.asarray(
                [self.offset, self.zero_tolerance],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        return digest.hexdigest()


__all__ = [
    "REAL_DTYPE",
    "DEFAULT_ZERO_TOLERANCE",
    "BINARY_TOLERANCE",
    "QUBOModelError",
    "QUBOModel",
]
