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

"""Integer Fourier-frequency supports for toric CSSF surrogates.

For periodic coordinates

    theta_s in R^d / (2*pi*Z)^d,

each frequency vector ``k in Z^d`` defines the complex feature

    exp(i * k^T theta_s).

The same representation is used for OPF angle differences, QAOA parameters,
MA-QAOA term-wise angles, and digitized-QA schedule coordinates. This module
defines only immutable frequency supports; feature evaluation is implemented
separately in ``spectral/feature_matrix.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Final, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


INTEGER_DTYPE: Final[np.dtype[np.int64]] = np.dtype(np.int64)
REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)


class FrequencySupportError(ValueError):
    """Raised when a Fourier support violates the toric contract."""


class SupportKind(str, Enum):
    """Named deterministic support families."""

    CUSTOM = "custom"
    SIGNED_AXES = "signed_axes"
    TOTAL_L1 = "total_l1"
    PAIRWISE = "pairwise"


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise FrequencySupportError(f"{name} must be positive.")
    return value


def _frequency_matrix(
    frequencies: ArrayLike,
) -> NDArray[np.int64]:
    """Return a contiguous two-dimensional integer frequency matrix."""

    array = np.asarray(frequencies)

    if array.ndim != 2:
        raise FrequencySupportError(
            "frequencies must have shape (n_terms, n_dimensions)."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise FrequencySupportError(
            "frequencies must contain at least one row and one dimension."
        )

    if np.issubdtype(array.dtype, np.integer):
        result = np.ascontiguousarray(array, dtype=INTEGER_DTYPE)
    else:
        numeric = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise FrequencySupportError(
                "frequencies contain non-finite values."
            )
        rounded = np.rint(numeric)
        if not np.array_equal(numeric, rounded):
            raise FrequencySupportError(
                "Every Fourier frequency must be an integer."
            )
        result = np.ascontiguousarray(rounded, dtype=INTEGER_DTYPE)

    result.setflags(write=False)
    return result


def _sorted_unique_rows(
    rows: Iterable[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic unique ordering by L1 norm then lexicographically."""

    unique = set(rows)
    return tuple(
        sorted(
            unique,
            key=lambda row: (
                sum(abs(value) for value in row),
                row,
            ),
        )
    )


@dataclass(frozen=True, slots=True, init=False)
class FrequencySupport:
    """Immutable set of integer Fourier vectors."""

    frequencies: NDArray[np.int64]
    kind: SupportKind
    include_zero: bool
    require_conjugate_symmetry: bool

    def __init__(
        self,
        frequencies: ArrayLike,
        *,
        kind: SupportKind = SupportKind.CUSTOM,
        include_zero: bool | None = None,
        require_conjugate_symmetry: bool = True,
    ) -> None:
        matrix = _frequency_matrix(frequencies)

        if not isinstance(kind, SupportKind):
            raise TypeError("kind must be a SupportKind.")
        if not isinstance(require_conjugate_symmetry, bool):
            raise TypeError(
                "require_conjugate_symmetry must be boolean."
            )

        rows = [tuple(int(value) for value in row) for row in matrix]
        if len(set(rows)) != len(rows):
            raise FrequencySupportError(
                "frequencies must not contain duplicate rows."
            )

        has_zero = any(all(value == 0 for value in row) for row in rows)

        if include_zero is None:
            normalized_include_zero = has_zero
        elif not isinstance(include_zero, bool):
            raise TypeError("include_zero must be boolean or None.")
        else:
            normalized_include_zero = include_zero
            if include_zero != has_zero:
                raise FrequencySupportError(
                    "include_zero does not match the supplied support."
                )

        row_set = set(rows)
        if require_conjugate_symmetry:
            missing = [
                row
                for row in rows
                if tuple(-value for value in row) not in row_set
            ]
            if missing:
                raise FrequencySupportError(
                    "Support is not conjugate symmetric; "
                    f"missing negatives for {missing[:3]!r}."
                )

        object.__setattr__(self, "frequencies", matrix)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "include_zero",
            normalized_include_zero,
        )
        object.__setattr__(
            self,
            "require_conjugate_symmetry",
            require_conjugate_symmetry,
        )

    @property
    def n_terms(self) -> int:
        """Number of Fourier vectors."""

        return int(self.frequencies.shape[0])

    @property
    def n_dimensions(self) -> int:
        """Dimension of the toric coordinate space."""

        return int(self.frequencies.shape[1])

    @property
    def max_l1_order(self) -> int:
        """Maximum L1 norm of any frequency vector."""

        return int(np.max(np.sum(np.abs(self.frequencies), axis=1)))

    @property
    def max_linf_order(self) -> int:
        """Maximum absolute coordinate frequency."""

        return int(np.max(np.abs(self.frequencies)))

    @property
    def is_conjugate_symmetric(self) -> bool:
        """Whether every ``k`` is accompanied by ``-k``."""

        rows = {
            tuple(int(value) for value in row)
            for row in self.frequencies
        }
        return all(
            tuple(-value for value in row) in rows
            for row in rows
        )

    def contains(self, frequency: ArrayLike) -> bool:
        """Check whether one integer vector belongs to the support."""

        vector = np.asarray(frequency)
        if vector.ndim != 1 or vector.shape[0] != self.n_dimensions:
            raise FrequencySupportError(
                f"frequency must have shape ({self.n_dimensions},)."
            )
        if not np.issubdtype(vector.dtype, np.integer):
            numeric = np.asarray(vector, dtype=np.float64)
            if not np.all(np.isfinite(numeric)):
                raise FrequencySupportError(
                    "frequency contains non-finite values."
                )
            rounded = np.rint(numeric)
            if not np.array_equal(numeric, rounded):
                raise FrequencySupportError(
                    "frequency must be integer-valued."
                )
            vector = rounded

        candidate = tuple(int(value) for value in vector)
        return candidate in {
            tuple(int(value) for value in row)
            for row in self.frequencies
        }

    def phase(self, coordinates: ArrayLike) -> NDArray[np.float64]:
        """Return ``coordinates @ frequencies.T``.

        Coordinates may be unwrapped real angles. Adding integer multiples of
        ``2*pi`` leaves the corresponding complex feature matrix unchanged.
        """

        matrix = np.asarray(coordinates)

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim != 2:
            raise FrequencySupportError(
                "coordinates must be one- or two-dimensional."
            )

        if matrix.shape[1] != self.n_dimensions:
            raise FrequencySupportError(
                f"coordinates contain {matrix.shape[1]} dimensions; "
                f"expected {self.n_dimensions}."
            )

        real_matrix = np.ascontiguousarray(matrix, dtype=REAL_DTYPE)
        if not np.all(np.isfinite(real_matrix)):
            raise FrequencySupportError(
                "coordinates contain non-finite values."
            )

        return np.ascontiguousarray(
            real_matrix @ self.frequencies.T,
            dtype=REAL_DTYPE,
        )

    def labels(self) -> tuple[str, ...]:
        """Return deterministic human-readable frequency labels."""

        return tuple(
            "k=(" + ",".join(str(int(value)) for value in row) + ")"
            for row in self.frequencies
        )


def signed_axis_support(
    n_dimensions: int,
    *,
    max_harmonic: int = 1,
    include_zero: bool = False,
) -> FrequencySupport:
    """Build ``±m e_j`` frequencies for all dimensions and harmonics."""

    d = _validate_positive_integer(
        n_dimensions,
        name="n_dimensions",
    )
    maximum = _validate_positive_integer(
        max_harmonic,
        name="max_harmonic",
    )
    if not isinstance(include_zero, bool):
        raise TypeError("include_zero must be boolean.")

    rows: list[tuple[int, ...]] = []

    if include_zero:
        rows.append((0,) * d)

    for harmonic in range(1, maximum + 1):
        for dimension in range(d):
            positive = [0] * d
            positive[dimension] = harmonic
            negative = [0] * d
            negative[dimension] = -harmonic
            rows.extend((tuple(negative), tuple(positive)))

    ordered = _sorted_unique_rows(rows)

    return FrequencySupport(
        ordered,
        kind=SupportKind.SIGNED_AXES,
        include_zero=include_zero,
        require_conjugate_symmetry=True,
    )


def total_l1_support(
    n_dimensions: int,
    *,
    max_l1_order: int,
    include_zero: bool = False,
    max_terms: int = 100_000,
) -> FrequencySupport:
    """Enumerate all integer vectors with L1 norm at most ``max_l1_order``."""

    d = _validate_positive_integer(
        n_dimensions,
        name="n_dimensions",
    )
    order = _validate_positive_integer(
        max_l1_order,
        name="max_l1_order",
    )
    budget = _validate_positive_integer(
        max_terms,
        name="max_terms",
    )
    if not isinstance(include_zero, bool):
        raise TypeError("include_zero must be boolean.")

    rows: list[tuple[int, ...]] = []

    for row in itertools.product(
        range(-order, order + 1),
        repeat=d,
    ):
        l1_order = sum(abs(value) for value in row)

        if l1_order == 0:
            if include_zero:
                rows.append(tuple(row))
            continue

        if l1_order <= order:
            rows.append(tuple(row))
            if len(rows) > budget:
                raise FrequencySupportError(
                    "Requested total-L1 support exceeds max_terms="
                    f"{budget}. Reduce dimensions or order."
                )

    ordered = _sorted_unique_rows(rows)

    return FrequencySupport(
        ordered,
        kind=SupportKind.TOTAL_L1,
        include_zero=include_zero,
        require_conjugate_symmetry=True,
    )


def pairwise_support(
    n_dimensions: int,
    *,
    include_axes: bool = True,
    include_sums: bool = True,
    include_differences: bool = True,
    include_zero: bool = False,
) -> FrequencySupport:
    """Build first-order axes and two-coordinate interaction frequencies.

    Pairwise sums generate ``±(e_i + e_j)``.
    Pairwise differences generate ``±(e_i - e_j)``.
    """

    d = _validate_positive_integer(
        n_dimensions,
        name="n_dimensions",
    )

    for name, value in (
        ("include_axes", include_axes),
        ("include_sums", include_sums),
        ("include_differences", include_differences),
        ("include_zero", include_zero),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")

    if not (
        include_axes
        or include_sums
        or include_differences
        or include_zero
    ):
        raise FrequencySupportError(
            "At least one frequency family must be enabled."
        )

    rows: list[tuple[int, ...]] = []

    if include_zero:
        rows.append((0,) * d)

    if include_axes:
        for dimension in range(d):
            vector = [0] * d
            vector[dimension] = 1
            rows.append(tuple(vector))
            rows.append(tuple(-value for value in vector))

    for first in range(d):
        for second in range(first + 1, d):
            if include_sums:
                vector = [0] * d
                vector[first] = 1
                vector[second] = 1
                rows.append(tuple(vector))
                rows.append(tuple(-value for value in vector))

            if include_differences:
                vector = [0] * d
                vector[first] = 1
                vector[second] = -1
                rows.append(tuple(vector))
                rows.append(tuple(-value for value in vector))

    ordered = _sorted_unique_rows(rows)

    return FrequencySupport(
        ordered,
        kind=SupportKind.PAIRWISE,
        include_zero=include_zero,
        require_conjugate_symmetry=True,
    )


__all__ = [
    "INTEGER_DTYPE",
    "REAL_DTYPE",
    "FrequencySupportError",
    "SupportKind",
    "FrequencySupport",
    "signed_axis_support",
    "total_l1_support",
    "pairwise_support",
]
