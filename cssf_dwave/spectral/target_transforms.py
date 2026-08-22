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

"""Invertible target transforms for CSSF surrogate levels.

The frozen CSNN-T primitive always learns real-valued targets. Different
physical observables may benefit from different invertible transforms before
fitting:

* identity for unconstrained observables such as mean energy;
* standardization for heterogeneous target scales;
* log1p for non-negative heavy-tailed observables;
* signed-log1p for signed heavy-tailed observables;
* logit for probabilities strictly inside ``(0, 1)``.

Every fitted transform is immutable and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Final, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)


class TargetTransformError(ValueError):
    """Raised when a target transform contract is violated."""


class TransformKind(str, Enum):
    """Supported target-transform families."""

    IDENTITY = "identity"
    STANDARDIZE = "standardize"
    LOG1P = "log1p"
    SIGNED_LOG1P = "signed_log1p"
    LOGIT = "logit"


def _as_real_matrix(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return a finite contiguous real matrix with shape ``(N, P)``."""

    array = np.asarray(values)

    if np.iscomplexobj(array):
        imaginary = np.asarray(array.imag)
        if np.any(imaginary != 0.0):
            raise TargetTransformError(
                f"{name} must be real-valued."
            )
        array = array.real

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise TargetTransformError(
            f"{name} must be one- or two-dimensional; "
            f"received shape {array.shape}."
        )

    if 0 in array.shape:
        raise TargetTransformError(f"{name} must be non-empty.")

    result = np.ascontiguousarray(array, dtype=REAL_DTYPE)

    if not np.all(np.isfinite(result)):
        raise TargetTransformError(
            f"{name} contains non-finite values."
        )

    return result


def _readonly(
    values: ArrayLike,
) -> NDArray[np.float64]:
    """Return an immutable contiguous float64 array."""

    result = np.ascontiguousarray(values, dtype=REAL_DTYPE)
    result.setflags(write=False)
    return result


@runtime_checkable
class FittedTargetTransform(Protocol):
    """Protocol implemented by every fitted transform."""

    kind: TransformKind
    n_targets: int

    def transform(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        """Map physical targets to surrogate coordinates."""

    def inverse_transform(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        """Map surrogate coordinates back to physical targets."""


@dataclass(frozen=True, slots=True)
class IdentityTransform:
    """No-op target transform."""

    n_targets: int
    kind: TransformKind = TransformKind.IDENTITY

    def __post_init__(self) -> None:
        if isinstance(self.n_targets, bool) or not isinstance(
            self.n_targets,
            int,
        ):
            raise TypeError("n_targets must be an integer.")
        if self.n_targets < 1:
            raise TargetTransformError("n_targets must be positive.")

    def _validate(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_real_matrix(values, name="values")
        if matrix.shape[1] != self.n_targets:
            raise TargetTransformError(
                f"Expected {self.n_targets} target columns; "
                f"received {matrix.shape[1]}."
            )
        return matrix

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        return self._validate(values).copy()

    def inverse_transform(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        return self._validate(values).copy()


@dataclass(frozen=True, slots=True)
class StandardizeTransform:
    """Column-wise affine standardization."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    kind: TransformKind = TransformKind.STANDARDIZE

    def __post_init__(self) -> None:
        mean = _readonly(np.asarray(self.mean).reshape(-1))
        scale = _readonly(np.asarray(self.scale).reshape(-1))

        if mean.size == 0:
            raise TargetTransformError("mean must not be empty.")
        if mean.shape != scale.shape:
            raise TargetTransformError(
                "mean and scale must have equal shapes."
            )
        if not np.all(np.isfinite(mean)):
            raise TargetTransformError("mean contains non-finite values.")
        if not np.all(np.isfinite(scale)):
            raise TargetTransformError("scale contains non-finite values.")
        if np.any(scale <= 0.0):
            raise TargetTransformError("scale must be strictly positive.")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @property
    def n_targets(self) -> int:
        return int(self.mean.size)

    def _validate(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_real_matrix(values, name="values")
        if matrix.shape[1] != self.n_targets:
            raise TargetTransformError(
                f"Expected {self.n_targets} target columns; "
                f"received {matrix.shape[1]}."
            )
        return matrix

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = self._validate(values)
        return np.ascontiguousarray(
            (matrix - self.mean) / self.scale,
            dtype=REAL_DTYPE,
        )

    def inverse_transform(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        matrix = self._validate(values)
        return np.ascontiguousarray(
            matrix * self.scale + self.mean,
            dtype=REAL_DTYPE,
        )


@dataclass(frozen=True, slots=True)
class Log1pTransform:
    """Element-wise ``log1p`` transform for non-negative targets."""

    n_targets: int
    kind: TransformKind = TransformKind.LOG1P

    def __post_init__(self) -> None:
        if isinstance(self.n_targets, bool) or not isinstance(
            self.n_targets,
            int,
        ):
            raise TypeError("n_targets must be an integer.")
        if self.n_targets < 1:
            raise TargetTransformError("n_targets must be positive.")

    def _validate(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_real_matrix(values, name="values")
        if matrix.shape[1] != self.n_targets:
            raise TargetTransformError(
                f"Expected {self.n_targets} target columns; "
                f"received {matrix.shape[1]}."
            )
        return matrix

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = self._validate(values)
        if np.any(matrix < 0.0):
            raise TargetTransformError(
                "Log1pTransform requires non-negative targets."
            )
        return np.ascontiguousarray(np.log1p(matrix), dtype=REAL_DTYPE)

    def inverse_transform(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        matrix = self._validate(values)
        result = np.expm1(matrix)
        if not np.all(np.isfinite(result)):
            raise TargetTransformError(
                "inverse log1p produced non-finite values."
            )
        if np.any(result < -1.0e-12):
            raise TargetTransformError(
                "inverse log1p produced negative physical targets."
            )
        return np.ascontiguousarray(
            np.maximum(result, 0.0),
            dtype=REAL_DTYPE,
        )


@dataclass(frozen=True, slots=True)
class SignedLog1pTransform:
    """Signed element-wise ``log1p`` transform."""

    n_targets: int
    kind: TransformKind = TransformKind.SIGNED_LOG1P

    def __post_init__(self) -> None:
        if isinstance(self.n_targets, bool) or not isinstance(
            self.n_targets,
            int,
        ):
            raise TypeError("n_targets must be an integer.")
        if self.n_targets < 1:
            raise TargetTransformError("n_targets must be positive.")

    def _validate(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_real_matrix(values, name="values")
        if matrix.shape[1] != self.n_targets:
            raise TargetTransformError(
                f"Expected {self.n_targets} target columns; "
                f"received {matrix.shape[1]}."
            )
        return matrix

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = self._validate(values)
        result = np.sign(matrix) * np.log1p(np.abs(matrix))
        return np.ascontiguousarray(result, dtype=REAL_DTYPE)

    def inverse_transform(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        matrix = self._validate(values)
        result = np.sign(matrix) * np.expm1(np.abs(matrix))
        if not np.all(np.isfinite(result)):
            raise TargetTransformError(
                "inverse signed-log1p produced non-finite values."
            )
        return np.ascontiguousarray(result, dtype=REAL_DTYPE)


@dataclass(frozen=True, slots=True)
class LogitTransform:
    """Probability transform using clipped log-odds."""

    n_targets: int
    epsilon: float = 1.0e-6
    kind: TransformKind = TransformKind.LOGIT

    def __post_init__(self) -> None:
        if isinstance(self.n_targets, bool) or not isinstance(
            self.n_targets,
            int,
        ):
            raise TypeError("n_targets must be an integer.")
        if self.n_targets < 1:
            raise TargetTransformError("n_targets must be positive.")

        epsilon = float(self.epsilon)
        if not math.isfinite(epsilon):
            raise TargetTransformError("epsilon must be finite.")
        if not 0.0 < epsilon < 0.5:
            raise TargetTransformError(
                "epsilon must lie strictly in (0, 0.5)."
            )

        object.__setattr__(self, "epsilon", epsilon)

    def _validate(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_real_matrix(values, name="values")
        if matrix.shape[1] != self.n_targets:
            raise TargetTransformError(
                f"Expected {self.n_targets} target columns; "
                f"received {matrix.shape[1]}."
            )
        return matrix

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = self._validate(values)

        if np.any(matrix < 0.0) or np.any(matrix > 1.0):
            raise TargetTransformError(
                "LogitTransform requires targets inside [0, 1]."
            )

        clipped = np.clip(
            matrix,
            self.epsilon,
            1.0 - self.epsilon,
        )
        result = np.log(clipped) - np.log1p(-clipped)
        return np.ascontiguousarray(result, dtype=REAL_DTYPE)

    def inverse_transform(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        matrix = self._validate(values)

        result = np.empty_like(matrix)
        nonnegative = matrix >= 0.0

        result[nonnegative] = 1.0 / (
            1.0 + np.exp(-matrix[nonnegative])
        )

        exp_values = np.exp(matrix[~nonnegative])
        result[~nonnegative] = exp_values / (1.0 + exp_values)

        result = np.clip(result, 0.0, 1.0)

        if not np.all(np.isfinite(result)):
            raise TargetTransformError(
                "inverse logit produced non-finite values."
            )

        return np.ascontiguousarray(result, dtype=REAL_DTYPE)


TargetTransform: type = (
    IdentityTransform
    | StandardizeTransform
    | Log1pTransform
    | SignedLog1pTransform
    | LogitTransform
)


def fit_standardize(
    values: ArrayLike,
    *,
    minimum_scale: float = 1.0e-12,
) -> StandardizeTransform:
    """Fit a deterministic column-wise standardization transform."""

    matrix = _as_real_matrix(values, name="values")
    minimum_scale = float(minimum_scale)

    if not math.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise TargetTransformError(
            "minimum_scale must be finite and positive."
        )

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    scale = np.maximum(scale, minimum_scale)

    return StandardizeTransform(mean=mean, scale=scale)


def build_transform(
    kind: TransformKind | str,
    *,
    n_targets: int,
    values: ArrayLike | None = None,
    epsilon: float = 1.0e-6,
    minimum_scale: float = 1.0e-12,
) -> FittedTargetTransform:
    """Construct or fit one target transform."""

    normalized_kind = (
        kind if isinstance(kind, TransformKind) else TransformKind(kind)
    )

    if normalized_kind is TransformKind.IDENTITY:
        return IdentityTransform(n_targets=n_targets)
    if normalized_kind is TransformKind.LOG1P:
        return Log1pTransform(n_targets=n_targets)
    if normalized_kind is TransformKind.SIGNED_LOG1P:
        return SignedLog1pTransform(n_targets=n_targets)
    if normalized_kind is TransformKind.LOGIT:
        return LogitTransform(
            n_targets=n_targets,
            epsilon=epsilon,
        )
    if normalized_kind is TransformKind.STANDARDIZE:
        if values is None:
            raise TargetTransformError(
                "values are required to fit StandardizeTransform."
            )
        fitted = fit_standardize(
            values,
            minimum_scale=minimum_scale,
        )
        if fitted.n_targets != n_targets:
            raise TargetTransformError(
                f"values contain {fitted.n_targets} targets; "
                f"expected {n_targets}."
            )
        return fitted

    raise TargetTransformError(
        f"Unsupported transform kind: {normalized_kind!r}."
    )


@dataclass(frozen=True, slots=True)
class TargetTransformBundle:
    """Column-block transform composition for heterogeneous targets."""

    transforms: tuple[FittedTargetTransform, ...]

    def __post_init__(self) -> None:
        transforms = tuple(self.transforms)
        if not transforms:
            raise TargetTransformError(
                "At least one fitted transform is required."
            )
        for transform in transforms:
            if not isinstance(transform, FittedTargetTransform):
                raise TypeError(
                    "Every bundle member must implement "
                    "FittedTargetTransform."
                )
        object.__setattr__(self, "transforms", transforms)

    @property
    def n_targets(self) -> int:
        return sum(transform.n_targets for transform in self.transforms)

    def _split(
        self,
        values: ArrayLike,
    ) -> list[NDArray[np.float64]]:
        matrix = _as_real_matrix(values, name="values")

        if matrix.shape[1] != self.n_targets:
            raise TargetTransformError(
                f"Expected {self.n_targets} target columns; "
                f"received {matrix.shape[1]}."
            )

        blocks: list[NDArray[np.float64]] = []
        start = 0

        for transform in self.transforms:
            stop = start + transform.n_targets
            blocks.append(matrix[:, start:stop])
            start = stop

        return blocks

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        blocks = self._split(values)
        return np.ascontiguousarray(
            np.column_stack(
                [
                    transform.transform(block)
                    for transform, block in zip(self.transforms, blocks)
                ]
            ),
            dtype=REAL_DTYPE,
        )

    def inverse_transform(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        blocks = self._split(values)
        return np.ascontiguousarray(
            np.column_stack(
                [
                    transform.inverse_transform(block)
                    for transform, block in zip(self.transforms, blocks)
                ]
            ),
            dtype=REAL_DTYPE,
        )


__all__ = [
    "REAL_DTYPE",
    "TargetTransformError",
    "TransformKind",
    "FittedTargetTransform",
    "IdentityTransform",
    "StandardizeTransform",
    "Log1pTransform",
    "SignedLog1pTransform",
    "LogitTransform",
    "TargetTransform",
    "fit_standardize",
    "build_transform",
    "TargetTransformBundle",
]
