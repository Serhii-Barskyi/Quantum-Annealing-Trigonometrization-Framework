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

"""Deterministic binary encoding of BESS placement decisions.

Each candidate bus receives exactly one binary variable. The candidate-bus
order from ``BESSFleetSpec`` is preserved across QUBO construction, sampling,
decoding, reporting, and reproducibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence
from urllib.parse import quote

import numpy as np
from numpy.typing import ArrayLike, NDArray

from opf.bess_constraints import BESSFleetSpec, BESSPlacement
from qubo.model import BINARY_TOLERANCE, QUBOModel
from qubo.penalties import exact_cardinality_penalty


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
INTEGER_DTYPE: Final[np.dtype[np.int8]] = np.dtype(np.int8)
DEFAULT_VARIABLE_PREFIX: Final[str] = "bess_bus"


class PlacementEncodingError(ValueError):
    """Raised when a placement encoding or binary sample is invalid."""


def _normalize_prefix(prefix: str) -> str:
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string.")

    normalized = prefix.strip()

    if not normalized:
        raise PlacementEncodingError(
            "prefix must be a non-empty string."
        )
    if any(character.isspace() for character in normalized):
        raise PlacementEncodingError(
            "prefix must not contain whitespace."
        )

    return normalized


def _bus_token(bus: Any) -> str:
    """Return a stable readable token for one hashable bus index."""

    try:
        hash(bus)
    except TypeError as exc:
        raise PlacementEncodingError(
            f"Candidate bus {bus!r} is not hashable."
        ) from exc

    if isinstance(bus, bool):
        canonical = f"bool:{int(bus)}"
    elif isinstance(bus, (int, np.integer)):
        canonical = f"int:{int(bus)}"
    elif isinstance(bus, (float, np.floating)):
        numeric = float(bus)
        if not math.isfinite(numeric):
            raise PlacementEncodingError(
                "Floating-point bus labels must be finite."
            )
        canonical = f"float:{format(numeric, '.17g')}"
    elif isinstance(bus, str):
        canonical = f"str:{bus}"
    else:
        type_name = (
            f"{type(bus).__module__}.{type(bus).__qualname__}"
        )
        canonical = f"{type_name}:{bus}"

    return quote(canonical, safe="-_.~")


def placement_variable_label(
    bus: Any,
    *,
    prefix: str = DEFAULT_VARIABLE_PREFIX,
) -> str:
    """Return the deterministic QUBO variable assigned to one bus."""

    return f"{_normalize_prefix(prefix)}[{_bus_token(bus)}]"


def _binary_vector(
    values: ArrayLike,
    *,
    expected_size: int,
    tolerance: float,
) -> NDArray[np.int8]:
    array = np.asarray(values, dtype=REAL_DTYPE)

    if array.ndim != 1:
        raise PlacementEncodingError(
            "Placement sample must be one-dimensional."
        )
    if array.size != expected_size:
        raise PlacementEncodingError(
            f"Placement sample contains {array.size} values; "
            f"expected {expected_size}."
        )
    if not np.all(np.isfinite(array)):
        raise PlacementEncodingError(
            "Placement sample contains non-finite values."
        )

    close_zero = np.abs(array) <= tolerance
    close_one = np.abs(array - 1.0) <= tolerance

    if not np.all(close_zero | close_one):
        raise PlacementEncodingError(
            "Placement sample must contain only binary values."
        )

    result = np.where(close_one, 1, 0).astype(INTEGER_DTYPE)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, init=False)
class BESSPlacementEncoding:
    """Immutable mapping between candidate buses and binary variables."""

    fleet: BESSFleetSpec
    prefix: str
    variable_order: tuple[str, ...]
    bus_to_variable: Mapping[Any, str]
    variable_to_bus: Mapping[str, Any]

    def __init__(
        self,
        fleet: BESSFleetSpec,
        *,
        prefix: str = DEFAULT_VARIABLE_PREFIX,
    ) -> None:
        if not isinstance(fleet, BESSFleetSpec):
            raise TypeError("fleet must be BESSFleetSpec.")

        normalized_prefix = _normalize_prefix(prefix)
        variables = tuple(
            placement_variable_label(
                bus,
                prefix=normalized_prefix,
            )
            for bus in fleet.candidate_buses
        )

        if len(set(variables)) != len(variables):
            raise PlacementEncodingError(
                "Candidate buses produced colliding variable labels."
            )

        bus_to_variable = dict(
            zip(fleet.candidate_buses, variables)
        )
        variable_to_bus = {
            variable: bus
            for bus, variable in bus_to_variable.items()
        }

        object.__setattr__(self, "fleet", fleet)
        object.__setattr__(self, "prefix", normalized_prefix)
        object.__setattr__(self, "variable_order", variables)
        object.__setattr__(
            self,
            "bus_to_variable",
            MappingProxyType(bus_to_variable),
        )
        object.__setattr__(
            self,
            "variable_to_bus",
            MappingProxyType(variable_to_bus),
        )

    @property
    def n_variables(self) -> int:
        return len(self.variable_order)

    def variable_for_bus(self, bus: Any) -> str:
        try:
            return self.bus_to_variable[bus]
        except KeyError as exc:
            raise PlacementEncodingError(
                f"Bus {bus!r} is not a placement candidate."
            ) from exc

    def bus_for_variable(self, variable: str) -> Any:
        normalized = str(variable).strip()

        try:
            return self.variable_to_bus[normalized]
        except KeyError as exc:
            raise PlacementEncodingError(
                f"Variable {variable!r} is not part of this encoding."
            ) from exc

    def encode_selected_buses(
        self,
        selected_buses: Sequence[Any],
        *,
        enforce_cardinality: bool = True,
    ) -> NDArray[np.int8]:
        """Encode selected buses in frozen candidate order."""

        if not isinstance(enforce_cardinality, bool):
            raise TypeError(
                "enforce_cardinality must be boolean."
            )

        selected = tuple(selected_buses)

        if len(set(selected)) != len(selected):
            raise PlacementEncodingError(
                "selected_buses must be unique."
            )

        unknown = set(selected) - set(self.fleet.candidate_buses)
        if unknown:
            raise PlacementEncodingError(
                "Selected buses are not candidates: "
                f"{sorted(unknown, key=str)!r}."
            )

        if (
            enforce_cardinality
            and len(selected) != self.fleet.units_to_place
        ):
            raise PlacementEncodingError(
                f"Selected {len(selected)} buses; expected "
                f"{self.fleet.units_to_place}."
            )

        selected_set = set(selected)
        result = np.array(
            [
                int(bus in selected_set)
                for bus in self.fleet.candidate_buses
            ],
            dtype=INTEGER_DTYPE,
        )
        result.setflags(write=False)
        return result

    def encode_placement(
        self,
        placement: BESSPlacement,
    ) -> NDArray[np.int8]:
        """Encode a placement belonging to this exact fleet."""

        if not isinstance(placement, BESSPlacement):
            raise TypeError("placement must be BESSPlacement.")
        if placement.fleet.fingerprint() != self.fleet.fingerprint():
            raise PlacementEncodingError(
                "Placement belongs to a different fleet."
            )

        result = np.array(
            placement.selection,
            dtype=INTEGER_DTYPE,
            copy=True,
        )
        result.setflags(write=False)
        return result

    def sample_vector(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
        *,
        tolerance: float = BINARY_TOLERANCE,
    ) -> NDArray[np.int8]:
        """Convert a labeled or ordered sample to candidate order."""

        normalized_tolerance = float(tolerance)

        if (
            not math.isfinite(normalized_tolerance)
            or normalized_tolerance <= 0.0
        ):
            raise PlacementEncodingError(
                "tolerance must be finite and positive."
            )

        if isinstance(sample, Mapping):
            expected = set(self.variable_order)
            supplied = set(sample)
            missing = expected - supplied
            extra = supplied - expected

            if missing or extra:
                raise PlacementEncodingError(
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

        return _binary_vector(
            values,
            expected_size=self.n_variables,
            tolerance=normalized_tolerance,
        )

    def selected_buses(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
        *,
        tolerance: float = BINARY_TOLERANCE,
    ) -> tuple[Any, ...]:
        """Return selected buses without enforcing cardinality."""

        vector = self.sample_vector(
            sample,
            tolerance=tolerance,
        )

        return tuple(
            bus
            for bus, value in zip(
                self.fleet.candidate_buses,
                vector,
            )
            if int(value) == 1
        )

    def decode_sample(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
        *,
        tolerance: float = BINARY_TOLERANCE,
    ) -> BESSPlacement:
        """Decode one feasible exact-cardinality sample."""

        vector = self.sample_vector(
            sample,
            tolerance=tolerance,
        )
        selected_count = int(vector.sum())

        if selected_count != self.fleet.units_to_place:
            raise PlacementEncodingError(
                f"Sample selects {selected_count} buses; expected "
                f"{self.fleet.units_to_place}."
            )

        return BESSPlacement(self.fleet, vector)

    def labeled_sample(
        self,
        sample: Mapping[str, int | float] | ArrayLike,
        *,
        tolerance: float = BINARY_TOLERANCE,
    ) -> dict[str, int]:
        """Return a plain labeled sample in frozen variable order."""

        vector = self.sample_vector(
            sample,
            tolerance=tolerance,
        )

        return {
            variable: int(value)
            for variable, value in zip(
                self.variable_order,
                vector,
            )
        }

    def coefficient_vector(
        self,
        coefficients_by_bus: Mapping[Any, float],
        *,
        default: float | None = None,
    ) -> NDArray[np.float64]:
        """Align bus-indexed objective coefficients to variable order."""

        if not isinstance(coefficients_by_bus, Mapping):
            raise TypeError(
                "coefficients_by_bus must be a mapping."
            )

        candidate_set = set(self.fleet.candidate_buses)
        extra = set(coefficients_by_bus) - candidate_set

        if extra:
            raise PlacementEncodingError(
                "Coefficients reference non-candidate buses: "
                f"{sorted(extra, key=str)!r}."
            )

        if default is None:
            missing = candidate_set - set(coefficients_by_bus)
            if missing:
                raise PlacementEncodingError(
                    "Missing coefficients for candidate buses: "
                    f"{sorted(missing, key=str)!r}."
                )
            default_value = 0.0
        else:
            default_value = float(default)
            if not math.isfinite(default_value):
                raise PlacementEncodingError(
                    "default must be finite."
                )

        values = np.array(
            [
                float(coefficients_by_bus.get(bus, default_value))
                for bus in self.fleet.candidate_buses
            ],
            dtype=REAL_DTYPE,
        )

        if not np.all(np.isfinite(values)):
            raise PlacementEncodingError(
                "Bus coefficients contain non-finite values."
            )

        values.setflags(write=False)
        return values

    def cardinality_penalty(
        self,
        *,
        strength: float,
    ) -> QUBOModel:
        """Return the fleet exact-cardinality penalty."""

        try:
            return exact_cardinality_penalty(
                self.variable_order,
                selected_count=self.fleet.units_to_place,
                strength=strength,
            )
        except ValueError as exc:
            raise PlacementEncodingError(
                f"Cannot construct cardinality penalty: {exc}"
            ) from exc

    def fingerprint(self) -> str:
        """Return a deterministic encoding fingerprint."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-BESSPlacementEncoding-v1\0")
        digest.update(self.fleet.fingerprint().encode("ascii"))
        digest.update(self.prefix.encode("utf-8"))
        digest.update(
            json.dumps(
                self.variable_order,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


__all__ = [
    "REAL_DTYPE",
    "INTEGER_DTYPE",
    "DEFAULT_VARIABLE_PREFIX",
    "PlacementEncodingError",
    "placement_variable_label",
    "BESSPlacementEncoding",
]
