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

"""Complex toric feature matrices for all CSSF surrogate levels.

For real periodic coordinates ``Theta in R^(N x d)`` and an integer Fourier
support ``K in Z^(M x d)``, the feature matrix is

    X[n, m] = exp(i * K[m]^T * Theta[n]).

The same construction is used by CSNN-T^OPF, CSNN-T^QAOA,
CSNN-T^MA-QAOA, and CSNN-T^digitized-QA. This module only constructs and
validates spectral features; the frozen fitting primitive remains unchanged in
``core/gcv.py`` and ``core/csnn_t.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spectral.frequency_support import (
    FrequencySupport,
    FrequencySupportError,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
COMPLEX_DTYPE: Final[np.dtype[np.complex128]] = np.dtype(np.complex128)
TWO_PI: Final[float] = 2.0 * math.pi


class FeatureMatrixError(ValueError):
    """Raised when toric coordinates or feature matrices are invalid."""


def _real_coordinate_matrix(
    coordinates: ArrayLike,
    *,
    n_dimensions: int,
) -> NDArray[np.float64]:
    """Return a finite contiguous coordinate matrix with shape ``(N, d)``."""

    array = np.asarray(coordinates)

    if np.iscomplexobj(array):
        if np.any(np.asarray(array.imag) != 0.0):
            raise FeatureMatrixError(
                "Periodic coordinates must be real-valued."
            )
        array = array.real

    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise FeatureMatrixError(
            "coordinates must be one- or two-dimensional."
        )

    if array.shape[0] == 0:
        raise FeatureMatrixError(
            "coordinates must contain at least one sample."
        )
    if array.shape[1] != n_dimensions:
        raise FeatureMatrixError(
            f"coordinates contain {array.shape[1]} dimensions; "
            f"expected {n_dimensions}."
        )

    result = np.ascontiguousarray(array, dtype=REAL_DTYPE)

    if not np.all(np.isfinite(result)):
        raise FeatureMatrixError(
            "coordinates contain non-finite values."
        )

    return result


def wrap_periodic_coordinates(
    coordinates: ArrayLike,
    *,
    period: float = TWO_PI,
    center: float = 0.0,
) -> NDArray[np.float64]:
    """Wrap real coordinates to ``[center-period/2, center+period/2)``."""

    period = float(period)
    center = float(center)

    if not math.isfinite(period) or period <= 0.0:
        raise FeatureMatrixError(
            "period must be finite and strictly positive."
        )
    if not math.isfinite(center):
        raise FeatureMatrixError("center must be finite.")

    array = np.asarray(coordinates)

    if np.iscomplexobj(array):
        if np.any(np.asarray(array.imag) != 0.0):
            raise FeatureMatrixError(
                "coordinates must be real-valued."
            )
        array = array.real

    if array.ndim not in (1, 2):
        raise FeatureMatrixError(
            "coordinates must be one- or two-dimensional."
        )
    if array.size == 0:
        raise FeatureMatrixError("coordinates must be non-empty.")

    real_array = np.ascontiguousarray(array, dtype=REAL_DTYPE)

    if not np.all(np.isfinite(real_array)):
        raise FeatureMatrixError(
            "coordinates contain non-finite values."
        )

    lower = center - 0.5 * period
    wrapped = np.mod(real_array - lower, period) + lower

    return np.ascontiguousarray(wrapped, dtype=REAL_DTYPE)


def toric_feature_matrix(
    coordinates: ArrayLike,
    support: FrequencySupport,
    *,
    wrap_coordinates: bool = True,
    check_unit_modulus: bool = True,
) -> NDArray[np.complex128]:
    """Construct the immutable complex matrix ``exp(i * Theta K^T)``."""

    if not isinstance(support, FrequencySupport):
        raise TypeError("support must be a FrequencySupport.")
    if not isinstance(wrap_coordinates, bool):
        raise TypeError("wrap_coordinates must be boolean.")
    if not isinstance(check_unit_modulus, bool):
        raise TypeError("check_unit_modulus must be boolean.")

    theta = _real_coordinate_matrix(
        coordinates,
        n_dimensions=support.n_dimensions,
    )

    if wrap_coordinates:
        theta = wrap_periodic_coordinates(theta)

    try:
        phase = support.phase(theta)
    except FrequencySupportError as exc:
        raise FeatureMatrixError(
            f"Cannot evaluate Fourier phases: {exc}"
        ) from exc

    features = np.ascontiguousarray(
        np.exp(1j * phase),
        dtype=COMPLEX_DTYPE,
    )

    if features.shape != (theta.shape[0], support.n_terms):
        raise FeatureMatrixError(
            "Constructed feature matrix has an unexpected shape."
        )
    if not np.all(np.isfinite(features.real)):
        raise FeatureMatrixError(
            "Feature matrix contains non-finite real components."
        )
    if not np.all(np.isfinite(features.imag)):
        raise FeatureMatrixError(
            "Feature matrix contains non-finite imaginary components."
        )

    if check_unit_modulus and not np.allclose(
        np.abs(features),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    ):
        raise FeatureMatrixError(
            "Toric Fourier features must have unit modulus."
        )

    features.setflags(write=False)
    return features


def toric_feature_jacobian(
    coordinates: ArrayLike,
    support: FrequencySupport,
    *,
    wrap_coordinates: bool = True,
) -> NDArray[np.complex128]:
    """Return ``dX[n,m]/dTheta[n,j] = i*K[m,j]*X[n,m]``.

    The returned shape is ``(n_samples, n_terms, n_dimensions)``.
    """

    theta = _real_coordinate_matrix(
        coordinates,
        n_dimensions=support.n_dimensions,
    )
    features = toric_feature_matrix(
        theta,
        support,
        wrap_coordinates=wrap_coordinates,
    )

    jacobian = (
        1j
        * features[:, :, np.newaxis]
        * support.frequencies[np.newaxis, :, :]
    )
    jacobian = np.ascontiguousarray(
        jacobian,
        dtype=COMPLEX_DTYPE,
    )

    if jacobian.shape != (
        theta.shape[0],
        support.n_terms,
        support.n_dimensions,
    ):
        raise FeatureMatrixError(
            "Constructed feature Jacobian has an unexpected shape."
        )

    jacobian.setflags(write=False)
    return jacobian


def verify_conjugate_feature_pairs(
    features: ArrayLike,
    support: FrequencySupport,
    *,
    tolerance: float = 1.0e-12,
) -> None:
    """Verify ``X(-k) = conjugate(X(k))`` for a symmetric support."""

    if not isinstance(support, FrequencySupport):
        raise TypeError("support must be a FrequencySupport.")

    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise FeatureMatrixError(
            "tolerance must be finite and positive."
        )

    matrix = np.asarray(features)

    if matrix.ndim != 2:
        raise FeatureMatrixError(
            "features must be two-dimensional."
        )
    if matrix.shape[1] != support.n_terms:
        raise FeatureMatrixError(
            f"features contain {matrix.shape[1]} columns; "
            f"expected {support.n_terms}."
        )

    complex_matrix = np.ascontiguousarray(
        matrix,
        dtype=COMPLEX_DTYPE,
    )

    row_to_index = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(support.frequencies)
    }

    for row, index in row_to_index.items():
        negative = tuple(-value for value in row)
        negative_index = row_to_index.get(negative)

        if negative_index is None:
            raise FeatureMatrixError(
                f"Support is missing conjugate partner for {row!r}."
            )

        if not np.allclose(
            complex_matrix[:, negative_index],
            np.conjugate(complex_matrix[:, index]),
            rtol=0.0,
            atol=tolerance,
        ):
            raise FeatureMatrixError(
                f"Conjugate feature relation failed for k={row!r}."
            )


@dataclass(frozen=True, slots=True, init=False)
class ToricFeatureBatch:
    """Immutable coordinates, support, features, and deterministic fingerprint."""

    coordinates: NDArray[np.float64]
    support: FrequencySupport
    features: NDArray[np.complex128]
    wrapped: bool

    def __init__(
        self,
        coordinates: ArrayLike,
        support: FrequencySupport,
        *,
        wrap_coordinates: bool = True,
        verify_conjugates: bool = True,
    ) -> None:
        if not isinstance(support, FrequencySupport):
            raise TypeError("support must be a FrequencySupport.")
        if not isinstance(wrap_coordinates, bool):
            raise TypeError("wrap_coordinates must be boolean.")
        if not isinstance(verify_conjugates, bool):
            raise TypeError("verify_conjugates must be boolean.")

        theta = _real_coordinate_matrix(
            coordinates,
            n_dimensions=support.n_dimensions,
        )
        if wrap_coordinates:
            theta = wrap_periodic_coordinates(theta)

        theta = np.ascontiguousarray(theta, dtype=REAL_DTYPE)
        theta.setflags(write=False)

        features = toric_feature_matrix(
            theta,
            support,
            wrap_coordinates=False,
        )

        if verify_conjugates and support.require_conjugate_symmetry:
            verify_conjugate_feature_pairs(features, support)

        object.__setattr__(self, "coordinates", theta)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "wrapped", wrap_coordinates)

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_terms(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_dimensions(self) -> int:
        return int(self.coordinates.shape[1])

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint."""

        digest = hashlib.sha256()
        digest.update(b"ToricFeatureBatch-v1\0")
        digest.update(str(self.coordinates.shape).encode("ascii"))
        digest.update(self.coordinates.tobytes(order="C"))
        digest.update(str(self.support.frequencies.shape).encode("ascii"))
        digest.update(self.support.frequencies.tobytes(order="C"))
        digest.update(str(self.features.shape).encode("ascii"))
        digest.update(self.features.tobytes(order="C"))
        digest.update(b"wrapped=1" if self.wrapped else b"wrapped=0")
        return digest.hexdigest()


__all__ = [
    "REAL_DTYPE",
    "COMPLEX_DTYPE",
    "TWO_PI",
    "FeatureMatrixError",
    "wrap_periodic_coordinates",
    "toric_feature_matrix",
    "toric_feature_jacobian",
    "verify_conjugate_feature_pairs",
    "ToricFeatureBatch",
]
