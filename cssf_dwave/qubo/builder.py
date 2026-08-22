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

"""Build deterministic BESS-placement QUBO models.

The builder combines a bus-indexed placement objective with the exact
cardinality required by :class:`opf.bess_constraints.BESSFleetSpec`.

Negative objective coefficients favor a bus or bus pair; positive
coefficients discourage it. Missing linear coefficients default to zero.
Reversed quadratic bus pairs are canonicalized and accumulated.

No sampler is imported or initialized by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from opf.bess_constraints import BESSFleetSpec, BESSPlacement
from qubo.encoding import (
    BESSPlacementEncoding,
    DEFAULT_VARIABLE_PREFIX,
    PlacementEncodingError,
)
from qubo.model import (
    DEFAULT_ZERO_TOLERANCE,
    QUBOModel,
    QUBOModelError,
)
from qubo.penalties import (
    objective_absolute_bound,
    recommended_penalty_strength,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
FLOAT64_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
ENERGY_AUDIT_SAFETY_FACTOR: Final[float] = 8.0
DEFAULT_PENALTY_MULTIPLIER: Final[float] = 10.0
DEFAULT_MINIMUM_PENALTY: Final[float] = 1.0


class QUBOBuilderError(ValueError):
    """Raised when a placement QUBO cannot be constructed safely."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)

    if not math.isfinite(normalized):
        raise QUBOBuilderError(f"{name} must be finite.")

    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)

    if normalized <= 0.0:
        raise QUBOBuilderError(
            f"{name} must be strictly positive."
        )

    return normalized


def _absolute_energy_scale(
    model: QUBOModel,
    vector: NDArray[np.float64],
) -> float:
    """Return the absolute coefficient sum active in one sample."""

    return float(
        abs(model.offset)
        + vector @ np.abs(model.linear)
        + vector @ np.abs(model.quadratic) @ vector
    )


def _energy_audit_tolerance(
    models: tuple[QUBOModel, ...],
    vector: NDArray[np.float64],
) -> float:
    """Bound floating roundoff for independently evaluated QUBOs.

    The three placement energies are evaluated by separate floating-point
    matrix products.  Exact-cardinality penalties can contain large terms
    that cancel for feasible samples, so a fixed absolute tolerance is not a
    valid audit.  This bound scales with the active absolute coefficient sum
    and with a conservative count of additions and multiplications.
    """

    operation_count = sum(
        3 + model.n_variables + 2 * model.n_interactions
        for model in models
    )
    absolute_scale = sum(
        _absolute_energy_scale(model, vector)
        for model in models
    )
    roundoff_bound = (
        ENERGY_AUDIT_SAFETY_FACTOR
        * FLOAT64_EPSILON
        * max(1, operation_count)
        * max(1.0, absolute_scale)
    )
    return max(
        *(model.zero_tolerance for model in models),
        roundoff_bound,
    )


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
        )
    except (TypeError, ValueError) as exc:
        raise QUBOBuilderError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc

    return MappingProxyType(json.loads(encoded))


def _normalize_linear_by_bus(
    fleet: BESSFleetSpec,
    linear_by_bus: Mapping[Any, float] | None,
) -> dict[Any, float]:
    source: Mapping[Any, float] = (
        {} if linear_by_bus is None else linear_by_bus
    )

    if not isinstance(source, Mapping):
        raise TypeError("linear_by_bus must be a mapping or None.")

    candidate_set = set(fleet.candidate_buses)
    extra = set(source) - candidate_set

    if extra:
        raise QUBOBuilderError(
            "linear_by_bus references non-candidate buses: "
            f"{sorted(extra, key=str)!r}."
        )

    result = {
        bus: 0.0
        for bus in fleet.candidate_buses
    }

    for bus, coefficient in source.items():
        result[bus] = _finite_float(
            coefficient,
            name=f"linear_by_bus[{bus!r}]",
        )

    return result


def _normalize_quadratic_by_bus_pair(
    encoding: BESSPlacementEncoding,
    quadratic_by_bus_pair: Mapping[
        tuple[Any, Any],
        float,
    ]
    | None,
) -> tuple[
    dict[str, float],
    dict[tuple[str, str], float],
]:
    """Canonicalize bus-pair coefficients into labeled QUBO biases."""

    source: Mapping[tuple[Any, Any], float] = (
        {}
        if quadratic_by_bus_pair is None
        else quadratic_by_bus_pair
    )

    if not isinstance(source, Mapping):
        raise TypeError(
            "quadratic_by_bus_pair must be a mapping or None."
        )

    candidate_set = set(encoding.fleet.candidate_buses)
    variable_position = {
        variable: index
        for index, variable in enumerate(
            encoding.variable_order
        )
    }

    diagonal_additions = {
        variable: 0.0
        for variable in encoding.variable_order
    }
    pair_coefficients: dict[tuple[str, str], float] = {}

    for pair, raw_coefficient in source.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise QUBOBuilderError(
                "Every quadratic bus key must be a pair."
            )

        first_bus, second_bus = pair
        unknown = {
            bus
            for bus in pair
            if bus not in candidate_set
        }

        if unknown:
            raise QUBOBuilderError(
                "quadratic_by_bus_pair references "
                "non-candidate buses: "
                f"{sorted(unknown, key=str)!r}."
            )

        coefficient = _finite_float(
            raw_coefficient,
            name=f"quadratic_by_bus_pair[{pair!r}]",
        )

        first_variable = encoding.variable_for_bus(first_bus)
        second_variable = encoding.variable_for_bus(second_bus)

        if first_variable == second_variable:
            diagonal_additions[first_variable] += coefficient
            continue

        first_position = variable_position[first_variable]
        second_position = variable_position[second_variable]

        if first_position < second_position:
            canonical_pair = (
                first_variable,
                second_variable,
            )
        else:
            canonical_pair = (
                second_variable,
                first_variable,
            )

        pair_coefficients[canonical_pair] = (
            pair_coefficients.get(canonical_pair, 0.0)
            + coefficient
        )

    return diagonal_additions, pair_coefficients


@dataclass(frozen=True, slots=True)
class QUBOBuildConfig:
    """Configuration for safe placement-QUBO composition."""

    penalty_strength: float | None = None
    penalty_multiplier: float = DEFAULT_PENALTY_MULTIPLIER
    minimum_penalty_strength: float = DEFAULT_MINIMUM_PENALTY
    require_penalty_dominance: bool = True
    variable_prefix: str = DEFAULT_VARIABLE_PREFIX
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE

    def __post_init__(self) -> None:
        if self.penalty_strength is None:
            normalized_strength = None
        else:
            normalized_strength = _positive_float(
                self.penalty_strength,
                name="penalty_strength",
            )

        multiplier = _positive_float(
            self.penalty_multiplier,
            name="penalty_multiplier",
        )
        minimum = _positive_float(
            self.minimum_penalty_strength,
            name="minimum_penalty_strength",
        )
        tolerance = _positive_float(
            self.zero_tolerance,
            name="zero_tolerance",
        )

        if not isinstance(self.require_penalty_dominance, bool):
            raise TypeError(
                "require_penalty_dominance must be boolean."
            )
        if not isinstance(self.variable_prefix, str):
            raise TypeError("variable_prefix must be a string.")

        prefix = self.variable_prefix.strip()

        if not prefix:
            raise QUBOBuilderError(
                "variable_prefix must not be empty."
            )
        if any(character.isspace() for character in prefix):
            raise QUBOBuilderError(
                "variable_prefix must not contain whitespace."
            )

        object.__setattr__(
            self,
            "penalty_strength",
            normalized_strength,
        )
        object.__setattr__(
            self,
            "penalty_multiplier",
            multiplier,
        )
        object.__setattr__(
            self,
            "minimum_penalty_strength",
            minimum,
        )
        object.__setattr__(
            self,
            "variable_prefix",
            prefix,
        )
        object.__setattr__(
            self,
            "zero_tolerance",
            tolerance,
        )


@dataclass(frozen=True, slots=True)
class BESSPlacementQUBO:
    """Complete objective, constraint, and encoding bundle."""

    encoding: BESSPlacementEncoding
    objective_model: QUBOModel
    cardinality_penalty: QUBOModel
    model: QUBOModel
    penalty_strength: float
    objective_bound: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(
            self.encoding,
            BESSPlacementEncoding,
        ):
            raise TypeError(
                "encoding must be BESSPlacementEncoding."
            )

        for field_name in (
            "objective_model",
            "cardinality_penalty",
            "model",
        ):
            if not isinstance(getattr(self, field_name), QUBOModel):
                raise TypeError(
                    f"{field_name} must be QUBOModel."
                )

        expected_order = self.encoding.variable_order

        if self.objective_model.variable_order != expected_order:
            raise QUBOBuilderError(
                "objective_model variable order differs from encoding."
            )
        if self.cardinality_penalty.variable_order != expected_order:
            raise QUBOBuilderError(
                "cardinality_penalty variable order differs "
                "from encoding."
            )
        if self.model.variable_order != expected_order:
            raise QUBOBuilderError(
                "model variable order differs from encoding."
            )

        normalized_strength = _positive_float(
            self.penalty_strength,
            name="penalty_strength",
        )
        normalized_bound = _finite_float(
            self.objective_bound,
            name="objective_bound",
        )

        if normalized_bound < 0.0:
            raise QUBOBuilderError(
                "objective_bound must be non-negative."
            )

        reconstructed = self.objective_model.add(
            self.cardinality_penalty
        )

        if not np.array_equal(
            reconstructed.linear,
            self.model.linear,
        ):
            raise QUBOBuilderError(
                "model linear biases do not equal component sum."
            )
        if not np.array_equal(
            reconstructed.quadratic,
            self.model.quadratic,
        ):
            raise QUBOBuilderError(
                "model quadratic biases do not equal component sum."
            )
        if reconstructed.offset != self.model.offset:
            raise QUBOBuilderError(
                "model offset does not equal component sum."
            )

        object.__setattr__(
            self,
            "penalty_strength",
            normalized_strength,
        )
        object.__setattr__(
            self,
            "objective_bound",
            normalized_bound,
        )
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(self.metadata),
        )

    @property
    def fleet(self) -> BESSFleetSpec:
        return self.encoding.fleet

    @property
    def variable_order(self) -> tuple[str, ...]:
        return self.encoding.variable_order

    def is_feasible(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
    ) -> bool:
        """Return whether the sample has the required cardinality."""

        vector = self.encoding.sample_vector(sample)
        return (
            int(vector.sum())
            == self.fleet.units_to_place
        )

    def decode(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
    ) -> BESSPlacement:
        """Decode one feasible sample into a BESS placement."""

        try:
            return self.encoding.decode_sample(sample)
        except PlacementEncodingError as exc:
            raise QUBOBuilderError(
                f"Cannot decode placement sample: {exc}"
            ) from exc

    def energy_breakdown(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
    ) -> Mapping[str, float]:
        """Return objective, cardinality, and total energies."""

        vector = self.encoding.sample_vector(sample)
        objective_energy = self.objective_model.energy(vector)
        cardinality_energy = self.cardinality_penalty.energy(
            vector
        )
        total_energy = self.model.energy(vector)
        audit_tolerance = _energy_audit_tolerance(
            (
                self.objective_model,
                self.cardinality_penalty,
                self.model,
            ),
            vector,
        )

        if not math.isclose(
            objective_energy + cardinality_energy,
            total_energy,
            rel_tol=0.0,
            abs_tol=audit_tolerance,
        ):
            raise QUBOBuilderError(
                "Energy components do not sum to total energy."
            )

        return MappingProxyType(
            {
                "objective": objective_energy,
                "placement_cardinality": cardinality_energy,
                "total": total_energy,
            }
        )

    def fingerprint(self) -> str:
        """Return a deterministic fingerprint of the complete build."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-BESSPlacementQUBO-v1\0")
        digest.update(
            self.encoding.fingerprint().encode("ascii")
        )
        digest.update(
            self.objective_model.fingerprint().encode("ascii")
        )
        digest.update(
            self.cardinality_penalty.fingerprint().encode(
                "ascii"
            )
        )
        digest.update(self.model.fingerprint().encode("ascii"))
        digest.update(
            np.asarray(
                [
                    self.penalty_strength,
                    self.objective_bound,
                ],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        digest.update(
            json.dumps(
                dict(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-ready reproducibility manifest."""

        return {
            "fingerprint": self.fingerprint(),
            "encoding_fingerprint": (
                self.encoding.fingerprint()
            ),
            "objective_fingerprint": (
                self.objective_model.fingerprint()
            ),
            "penalty_fingerprint": (
                self.cardinality_penalty.fingerprint()
            ),
            "model_fingerprint": self.model.fingerprint(),
            "n_variables": self.model.n_variables,
            "n_interactions": self.model.n_interactions,
            "units_to_place": self.fleet.units_to_place,
            "penalty_strength": self.penalty_strength,
            "objective_bound": self.objective_bound,
            "variable_order": list(self.variable_order),
            "metadata": dict(self.metadata),
        }


def build_bess_placement_qubo(
    fleet: BESSFleetSpec,
    *,
    linear_by_bus: Mapping[Any, float] | None = None,
    quadratic_by_bus_pair: Mapping[
        tuple[Any, Any],
        float,
    ]
    | None = None,
    objective_offset: float = 0.0,
    config: QUBOBuildConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BESSPlacementQUBO:
    """Build an objective plus exact-cardinality placement penalty."""

    if not isinstance(fleet, BESSFleetSpec):
        raise TypeError("fleet must be BESSFleetSpec.")

    build_config = (
        QUBOBuildConfig()
        if config is None
        else config
    )

    if not isinstance(build_config, QUBOBuildConfig):
        raise TypeError(
            "config must be QUBOBuildConfig or None."
        )

    encoding = BESSPlacementEncoding(
        fleet,
        prefix=build_config.variable_prefix,
    )

    linear_bus_coefficients = _normalize_linear_by_bus(
        fleet,
        linear_by_bus,
    )
    diagonal_additions, pair_coefficients = (
        _normalize_quadratic_by_bus_pair(
            encoding,
            quadratic_by_bus_pair,
        )
    )

    linear_by_variable = {
        encoding.variable_for_bus(bus): (
            linear_bus_coefficients[bus]
            + diagonal_additions[
                encoding.variable_for_bus(bus)
            ]
        )
        for bus in fleet.candidate_buses
    }

    try:
        objective = QUBOModel.from_coefficients(
            linear=linear_by_variable,
            quadratic=pair_coefficients,
            offset=_finite_float(
                objective_offset,
                name="objective_offset",
            ),
            variable_order=encoding.variable_order,
            zero_tolerance=build_config.zero_tolerance,
        )
    except QUBOModelError as exc:
        raise QUBOBuilderError(
            f"Cannot construct placement objective: {exc}"
        ) from exc

    objective_bound = objective_absolute_bound(objective)

    dominance_floor = (
        2.0 * objective_bound
        + build_config.zero_tolerance
    )
    required_strength = max(
        build_config.minimum_penalty_strength,
        dominance_floor
        if build_config.require_penalty_dominance
        else 0.0,
    )

    if build_config.penalty_strength is None:
        proposed_strength = recommended_penalty_strength(
            objective,
            multiplier=build_config.penalty_multiplier,
            minimum=build_config.minimum_penalty_strength,
        )
        penalty_strength = max(
            proposed_strength,
            required_strength,
        )
    else:
        penalty_strength = build_config.penalty_strength

        if penalty_strength < required_strength:
            raise QUBOBuilderError(
                f"penalty_strength={penalty_strength} is below "
                f"the required safe value {required_strength}."
            )

    try:
        cardinality_penalty = encoding.cardinality_penalty(
            strength=penalty_strength
        )
        model = objective.add(cardinality_penalty)
    except (PlacementEncodingError, QUBOModelError) as exc:
        raise QUBOBuilderError(
            f"Cannot compose placement QUBO: {exc}"
        ) from exc

    merged_metadata = {
        **({} if metadata is None else dict(metadata)),
        "objective_component": "opf_surrogate_cost",
        "constraint_component": "placement_cardinality",
        "fleet_fingerprint": fleet.fingerprint(),
        "encoding_fingerprint": encoding.fingerprint(),
    }

    return BESSPlacementQUBO(
        encoding=encoding,
        objective_model=objective,
        cardinality_penalty=cardinality_penalty,
        model=model,
        penalty_strength=penalty_strength,
        objective_bound=objective_bound,
        metadata=merged_metadata,
    )


__all__ = [
    "REAL_DTYPE",
    "DEFAULT_PENALTY_MULTIPLIER",
    "DEFAULT_MINIMUM_PENALTY",
    "QUBOBuilderError",
    "QUBOBuildConfig",
    "BESSPlacementQUBO",
    "build_bess_placement_qubo",
]
