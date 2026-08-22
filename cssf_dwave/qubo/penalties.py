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

"""Solver-independent QUBO penalties for binary placement constraints.

For binary variables, a linear equality

    sum_i a_i x_i = b

is encoded exactly by

    strength * (sum_i a_i x_i - b)^2.

The expansion is returned as an immutable :class:`qubo.model.QUBOModel`.
This module also provides exact-cardinality, one-hot, at-most-one, fixed-value,
forbidden-pair, and implication penalties.
"""

from __future__ import annotations

import math
from typing import Final, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qubo.model import DEFAULT_ZERO_TOLERANCE, QUBOModel


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)


class QUBOPenaltyError(ValueError):
    """Raised when a binary penalty cannot be constructed."""


def _variables(variable_order: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value).strip() for value in variable_order)

    if not values:
        raise QUBOPenaltyError(
            "variable_order must contain at least one variable."
        )
    if any(not value for value in values):
        raise QUBOPenaltyError(
            "variable_order must not contain empty labels."
        )
    if len(set(values)) != len(values):
        raise QUBOPenaltyError(
            "variable_order must contain unique labels."
        )

    return values


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)

    if not math.isfinite(normalized):
        raise QUBOPenaltyError(f"{name} must be finite.")

    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)

    if normalized <= 0.0:
        raise QUBOPenaltyError(
            f"{name} must be strictly positive."
        )

    return normalized


def _coefficient_vector(
    variable_order: tuple[str, ...],
    coefficients: Mapping[str, float] | ArrayLike,
) -> NDArray[np.float64]:
    """Return coefficients aligned with ``variable_order``."""

    if isinstance(coefficients, Mapping):
        normalized_mapping: dict[str, float] = {}

        for label, value in coefficients.items():
            normalized_label = str(label).strip()

            if not normalized_label:
                raise QUBOPenaltyError(
                    "coefficients contain an empty variable label."
                )
            if normalized_label in normalized_mapping:
                raise QUBOPenaltyError(
                    f"Duplicate coefficient label {normalized_label!r}."
                )

            normalized_mapping[normalized_label] = _finite_float(
                value,
                name=f"coefficients[{normalized_label!r}]",
            )

        extra = set(normalized_mapping) - set(variable_order)

        if extra:
            raise QUBOPenaltyError(
                "coefficients reference unknown variables: "
                f"{sorted(extra)!r}."
            )

        vector = np.array(
            [
                normalized_mapping.get(variable, 0.0)
                for variable in variable_order
            ],
            dtype=REAL_DTYPE,
        )
    else:
        vector = np.asarray(
            coefficients,
            dtype=REAL_DTYPE,
        ).reshape(-1)

        if vector.size != len(variable_order):
            raise QUBOPenaltyError(
                f"coefficients must contain {len(variable_order)} "
                f"values; received {vector.size}."
            )
        if not np.all(np.isfinite(vector)):
            raise QUBOPenaltyError(
                "coefficients contain non-finite values."
            )

        vector = np.ascontiguousarray(vector, dtype=REAL_DTYPE)

    if not np.any(vector != 0.0):
        raise QUBOPenaltyError(
            "At least one equality coefficient must be non-zero."
        )

    return vector


def _require_variable(
    variable_order: tuple[str, ...],
    variable: str,
    *,
    name: str,
) -> str:
    normalized = str(variable).strip()

    if not normalized:
        raise QUBOPenaltyError(
            f"{name} must be a non-empty label."
        )
    if normalized not in variable_order:
        raise QUBOPenaltyError(
            f"{name}={normalized!r} is not in variable_order."
        )

    return normalized


def linear_equality_penalty(
    variable_order: Sequence[str],
    coefficients: Mapping[str, float] | ArrayLike,
    *,
    rhs: float,
    strength: float,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> QUBOModel:
    """Encode ``strength * (a.T @ x - rhs)**2`` exactly."""

    variables = _variables(variable_order)
    vector = _coefficient_vector(variables, coefficients)
    normalized_rhs = _finite_float(rhs, name="rhs")
    normalized_strength = _positive_float(
        strength,
        name="strength",
    )

    linear = normalized_strength * (
        vector * vector - 2.0 * normalized_rhs * vector
    )
    quadratic = np.zeros(
        (len(variables), len(variables)),
        dtype=REAL_DTYPE,
    )

    for first in range(len(variables)):
        for second in range(first + 1, len(variables)):
            quadratic[first, second] = (
                2.0
                * normalized_strength
                * vector[first]
                * vector[second]
            )

    offset = normalized_strength * normalized_rhs * normalized_rhs

    return QUBOModel(
        variable_order=variables,
        linear=linear,
        quadratic=quadratic,
        offset=offset,
        zero_tolerance=zero_tolerance,
    )


def exact_cardinality_penalty(
    variable_order: Sequence[str],
    *,
    selected_count: int,
    strength: float,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> QUBOModel:
    """Encode ``strength * (sum(x) - selected_count)**2``."""

    variables = _variables(variable_order)

    if isinstance(selected_count, bool) or not isinstance(
        selected_count,
        int,
    ):
        raise TypeError("selected_count must be an integer.")
    if not 0 <= selected_count <= len(variables):
        raise QUBOPenaltyError(
            "selected_count must lie in "
            f"[0, {len(variables)}]."
        )

    return linear_equality_penalty(
        variables,
        np.ones(len(variables), dtype=REAL_DTYPE),
        rhs=float(selected_count),
        strength=strength,
        zero_tolerance=zero_tolerance,
    )


def one_hot_penalty(
    variable_order: Sequence[str],
    *,
    strength: float,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> QUBOModel:
    """Require exactly one selected variable."""

    return exact_cardinality_penalty(
        variable_order,
        selected_count=1,
        strength=strength,
        zero_tolerance=zero_tolerance,
    )


def at_most_one_penalty(
    variable_order: Sequence[str],
    *,
    strength: float,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> QUBOModel:
    """Penalize every selected pair.

    The energy equals ``strength * C(k, 2)`` when ``k`` variables are one.
    """

    variables = _variables(variable_order)
    normalized_strength = _positive_float(
        strength,
        name="strength",
    )

    quadratic = np.zeros(
        (len(variables), len(variables)),
        dtype=REAL_DTYPE,
    )

    for first in range(len(variables)):
        for second in range(first + 1, len(variables)):
            quadratic[first, second] = normalized_strength

    return QUBOModel(
        variable_order=variables,
        linear=np.zeros(len(variables), dtype=REAL_DTYPE),
        quadratic=quadratic,
        offset=0.0,
        zero_tolerance=zero_tolerance,
    )


def fixed_binary_penalty(
    variable_order: Sequence[str],
    variable: str,
    *,
    value: int,
    strength: float,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> QUBOModel:
    """Require one variable to equal zero or one."""

    variables = _variables(variable_order)
    normalized_variable = _require_variable(
        variables,
        variable,
        name="variable",
    )

    if isinstance(value, bool):
        normalized_value = int(value)
    elif isinstance(value, int) and value in (0, 1):
        normalized_value = value
    else:
        raise QUBOPenaltyError("value must be binary 0 or 1.")

    return linear_equality_penalty(
        variables,
        {normalized_variable: 1.0},
        rhs=float(normalized_value),
        strength=strength,
        zero_tolerance=zero_tolerance,
    )


def forbidden_pair_penalty(
    variable_order: Sequence[str],
    first: str,
    second: str,
    *,
    strength: float,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> QUBOModel:
    """Forbid ``first = second = 1``."""

    variables = _variables(variable_order)
    normalized_first = _require_variable(
        variables,
        first,
        name="first",
    )
    normalized_second = _require_variable(
        variables,
        second,
        name="second",
    )

    if normalized_first == normalized_second:
        raise QUBOPenaltyError(
            "first and second must be different variables."
        )

    normalized_strength = _positive_float(
        strength,
        name="strength",
    )

    return QUBOModel.from_coefficients(
        linear={},
        quadratic={
            (normalized_first, normalized_second): normalized_strength
        },
        offset=0.0,
        variable_order=variables,
        zero_tolerance=zero_tolerance,
    )


def implication_penalty(
    variable_order: Sequence[str],
    antecedent: str,
    consequent: str,
    *,
    strength: float,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> QUBOModel:
    """Encode ``antecedent => consequent``."""

    variables = _variables(variable_order)
    normalized_antecedent = _require_variable(
        variables,
        antecedent,
        name="antecedent",
    )
    normalized_consequent = _require_variable(
        variables,
        consequent,
        name="consequent",
    )

    if normalized_antecedent == normalized_consequent:
        raise QUBOPenaltyError(
            "antecedent and consequent must be different."
        )

    normalized_strength = _positive_float(
        strength,
        name="strength",
    )

    return QUBOModel.from_coefficients(
        linear={normalized_antecedent: normalized_strength},
        quadratic={
            (
                normalized_antecedent,
                normalized_consequent,
            ): -normalized_strength,
        },
        offset=0.0,
        variable_order=variables,
        zero_tolerance=zero_tolerance,
    )


def objective_absolute_bound(model: QUBOModel) -> float:
    """Return a safe absolute bound on non-constant bias contributions."""

    if not isinstance(model, QUBOModel):
        raise TypeError("model must be QUBOModel.")

    return float(
        np.sum(np.abs(model.linear))
        + np.sum(np.abs(model.quadratic))
    )


def recommended_penalty_strength(
    objective: QUBOModel,
    *,
    multiplier: float = 10.0,
    minimum: float = 1.0,
) -> float:
    """Choose a deterministic strength above the objective-bias scale."""

    if not isinstance(objective, QUBOModel):
        raise TypeError("objective must be QUBOModel.")

    normalized_multiplier = _positive_float(
        multiplier,
        name="multiplier",
    )
    normalized_minimum = _positive_float(
        minimum,
        name="minimum",
    )

    scale = max(
        normalized_minimum,
        objective_absolute_bound(objective),
    )

    return normalized_multiplier * scale


__all__ = [
    "REAL_DTYPE",
    "QUBOPenaltyError",
    "linear_equality_penalty",
    "exact_cardinality_penalty",
    "one_hot_penalty",
    "at_most_one_penalty",
    "fixed_binary_penalty",
    "forbidden_pair_penalty",
    "implication_penalty",
    "objective_absolute_bound",
    "recommended_penalty_strength",
]
