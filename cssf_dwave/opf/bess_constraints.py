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

"""Pure mathematical BESS placement and dispatch constraints.

The project convention is:

* ``p_charge_mw >= 0`` absorbs active power from the grid;
* ``p_discharge_mw >= 0`` injects active power into the grid;
* net grid injection is ``p_discharge_mw - p_charge_mw``;
* energy is tracked in MWh at interval boundaries.

No solver is initialized and no network is modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
BINARY_TOLERANCE: Final[float] = 1.0e-9
SIMULTANEOUS_POWER_TOLERANCE_MW: Final[float] = 1.0e-10


class BESSConstraintError(ValueError):
    """Raised when a BESS decision violates a physical constraint."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise BESSConstraintError(f"{name} must be finite.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized <= 0.0:
        raise BESSConstraintError(
            f"{name} must be strictly positive."
        )
    return normalized


def _probability(
    value: float,
    *,
    name: str,
    allow_zero: bool = True,
) -> float:
    normalized = _finite_float(value, name=name)
    lower_ok = normalized >= 0.0 if allow_zero else normalized > 0.0

    if not lower_ok or normalized > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise BESSConstraintError(
            f"{name} must lie in {interval}."
        )

    return normalized


def _readonly_matrix(
    values: ArrayLike,
    *,
    name: str,
    expected_columns: int | None = None,
) -> NDArray[np.float64]:
    array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise BESSConstraintError(
            f"{name} must be one- or two-dimensional."
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise BESSConstraintError(f"{name} must be non-empty.")

    if (
        expected_columns is not None
        and array.shape[1] != expected_columns
    ):
        raise BESSConstraintError(
            f"{name} must contain {expected_columns} columns; "
            f"received {array.shape[1]}."
        )

    result = np.ascontiguousarray(array, dtype=REAL_DTYPE)

    if not np.all(np.isfinite(result)):
        raise BESSConstraintError(
            f"{name} contains non-finite values."
        )

    result.setflags(write=False)
    return result


def _readonly_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_size: int | None = None,
) -> NDArray[np.float64]:
    result = np.ascontiguousarray(
        np.asarray(values, dtype=REAL_DTYPE).reshape(-1),
        dtype=REAL_DTYPE,
    )

    if result.size == 0:
        raise BESSConstraintError(f"{name} must be non-empty.")
    if expected_size is not None and result.size != expected_size:
        raise BESSConstraintError(
            f"{name} must contain {expected_size} values; "
            f"received {result.size}."
        )
    if not np.all(np.isfinite(result)):
        raise BESSConstraintError(
            f"{name} contains non-finite values."
        )

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
        )
    except (TypeError, ValueError) as exc:
        raise BESSConstraintError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc

    return MappingProxyType(json.loads(encoded))


def _network_table(
    network: Any,
    name: str,
) -> pd.DataFrame | None:
    if hasattr(network, name):
        value = getattr(network, name)
    elif isinstance(network, Mapping):
        value = network.get(name)
    else:
        try:
            value = network[name]
        except (KeyError, TypeError, AttributeError):
            value = None

    if value is None:
        return None
    if not isinstance(value, pd.DataFrame):
        raise BESSConstraintError(
            f"Network table {name!r} must be a pandas DataFrame."
        )

    return value


@dataclass(frozen=True, slots=True)
class BESSUnitSpec:
    """Uniform physical specification of one installed BESS unit."""

    power_mw: float
    energy_mwh: float
    minimum_soc: float = 0.10
    maximum_soc: float = 0.90
    initial_soc: float = 0.50
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    self_discharge_per_hour: float = 0.0

    def __post_init__(self) -> None:
        power = _positive_float(self.power_mw, name="power_mw")
        energy = _positive_float(
            self.energy_mwh,
            name="energy_mwh",
        )
        minimum_soc = _probability(
            self.minimum_soc,
            name="minimum_soc",
        )
        maximum_soc = _probability(
            self.maximum_soc,
            name="maximum_soc",
        )
        initial_soc = _probability(
            self.initial_soc,
            name="initial_soc",
        )
        charge_efficiency = _probability(
            self.charge_efficiency,
            name="charge_efficiency",
            allow_zero=False,
        )
        discharge_efficiency = _probability(
            self.discharge_efficiency,
            name="discharge_efficiency",
            allow_zero=False,
        )
        self_discharge = _probability(
            self.self_discharge_per_hour,
            name="self_discharge_per_hour",
        )

        if minimum_soc >= maximum_soc:
            raise BESSConstraintError(
                "minimum_soc must be smaller than maximum_soc."
            )
        if not minimum_soc <= initial_soc <= maximum_soc:
            raise BESSConstraintError(
                "initial_soc must lie inside the permitted SOC range."
            )
        if self_discharge >= 1.0:
            raise BESSConstraintError(
                "self_discharge_per_hour must be smaller than 1."
            )

        object.__setattr__(self, "power_mw", power)
        object.__setattr__(self, "energy_mwh", energy)
        object.__setattr__(self, "minimum_soc", minimum_soc)
        object.__setattr__(self, "maximum_soc", maximum_soc)
        object.__setattr__(self, "initial_soc", initial_soc)
        object.__setattr__(
            self,
            "charge_efficiency",
            charge_efficiency,
        )
        object.__setattr__(
            self,
            "discharge_efficiency",
            discharge_efficiency,
        )
        object.__setattr__(
            self,
            "self_discharge_per_hour",
            self_discharge,
        )

    @property
    def minimum_energy_mwh(self) -> float:
        return self.minimum_soc * self.energy_mwh

    @property
    def maximum_energy_mwh(self) -> float:
        return self.maximum_soc * self.energy_mwh

    @property
    def initial_energy_mwh(self) -> float:
        return self.initial_soc * self.energy_mwh

    @property
    def duration_hours(self) -> float:
        return self.energy_mwh / self.power_mw

    def as_dict(self) -> dict[str, float]:
        return {
            "power_mw": self.power_mw,
            "energy_mwh": self.energy_mwh,
            "minimum_soc": self.minimum_soc,
            "maximum_soc": self.maximum_soc,
            "initial_soc": self.initial_soc,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency,
            "self_discharge_per_hour": (
                self.self_discharge_per_hour
            ),
        }


@dataclass(frozen=True, slots=True, init=False)
class BESSFleetSpec:
    """Exact-cardinality placement specification over candidate buses."""

    candidate_buses: tuple[Any, ...]
    units_to_place: int
    unit: BESSUnitSpec
    metadata: Mapping[str, Any]

    def __init__(
        self,
        candidate_buses: Sequence[Any],
        *,
        units_to_place: int,
        unit: BESSUnitSpec,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        candidates = tuple(candidate_buses)

        if not candidates:
            raise BESSConstraintError(
                "candidate_buses must not be empty."
            )
        if len(set(candidates)) != len(candidates):
            raise BESSConstraintError(
                "candidate_buses must be unique."
            )
        if isinstance(units_to_place, bool) or not isinstance(
            units_to_place,
            int,
        ):
            raise TypeError("units_to_place must be an integer.")
        if not 1 <= units_to_place < len(candidates):
            raise BESSConstraintError(
                "units_to_place must satisfy "
                "1 <= units_to_place < len(candidate_buses)."
            )
        if not isinstance(unit, BESSUnitSpec):
            raise TypeError("unit must be BESSUnitSpec.")

        object.__setattr__(self, "candidate_buses", candidates)
        object.__setattr__(
            self,
            "units_to_place",
            units_to_place,
        )
        object.__setattr__(self, "unit", unit)
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(metadata),
        )

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_buses)

    @property
    def total_power_mw(self) -> float:
        return self.units_to_place * self.unit.power_mw

    @property
    def total_energy_mwh(self) -> float:
        return self.units_to_place * self.unit.energy_mwh

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-BESSFleetSpec-v1\0")
        digest.update(
            json.dumps(
                list(self.candidate_buses),
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(str(self.units_to_place).encode("ascii"))
        digest.update(
            json.dumps(
                self.unit.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
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


def validate_candidate_buses(
    network: Any,
    candidate_buses: Sequence[Any],
    *,
    exclude_slack_buses: bool = True,
    require_in_service: bool = True,
) -> tuple[Any, ...]:
    """Validate candidate buses against one network."""

    if not isinstance(exclude_slack_buses, bool):
        raise TypeError("exclude_slack_buses must be boolean.")
    if not isinstance(require_in_service, bool):
        raise TypeError("require_in_service must be boolean.")

    candidates = tuple(candidate_buses)

    if not candidates:
        raise BESSConstraintError(
            "candidate_buses must not be empty."
        )
    if len(set(candidates)) != len(candidates):
        raise BESSConstraintError(
            "candidate_buses must be unique."
        )

    bus = _network_table(network, "bus")
    if bus is None or bus.empty:
        raise BESSConstraintError(
            "Network must contain a non-empty bus table."
        )

    missing = set(candidates) - set(bus.index.tolist())
    if missing:
        raise BESSConstraintError(
            f"Unknown candidate buses: {sorted(missing, key=str)[:5]!r}."
        )

    if require_in_service and "in_service" in bus.columns:
        inactive = [
            bus_index
            for bus_index in candidates
            if not bool(bus.loc[bus_index, "in_service"])
        ]
        if inactive:
            raise BESSConstraintError(
                f"Candidate buses are out of service: {inactive[:5]!r}."
            )

    if exclude_slack_buses:
        slack_buses: set[Any] = set()

        ext_grid = _network_table(network, "ext_grid")
        if ext_grid is not None and not ext_grid.empty:
            active = (
                np.ones(len(ext_grid), dtype=bool)
                if "in_service" not in ext_grid.columns
                else ext_grid["in_service"].fillna(True).astype(bool)
            )
            slack_buses.update(
                ext_grid.loc[active, "bus"].tolist()
            )

        gen = _network_table(network, "gen")
        if (
            gen is not None
            and not gen.empty
            and "slack" in gen.columns
        ):
            active = gen["slack"].fillna(False).astype(bool)
            if "in_service" in gen.columns:
                active &= gen["in_service"].fillna(True).astype(bool)
            slack_buses.update(gen.loc[active, "bus"].tolist())

        forbidden = set(candidates) & slack_buses
        if forbidden:
            raise BESSConstraintError(
                "Slack buses cannot be placement candidates: "
                f"{sorted(forbidden, key=str)!r}."
            )

    return candidates


def validate_binary_selection(
    selection: ArrayLike,
    *,
    expected_count: int,
    tolerance: float = BINARY_TOLERANCE,
) -> NDArray[np.int8]:
    """Validate an exact-cardinality binary placement vector."""

    if isinstance(expected_count, bool) or not isinstance(
        expected_count,
        int,
    ):
        raise TypeError("expected_count must be an integer.")
    if expected_count < 1:
        raise BESSConstraintError(
            "expected_count must be positive."
        )

    tolerance = _positive_float(
        tolerance,
        name="tolerance",
    )

    values = np.asarray(selection, dtype=REAL_DTYPE)

    if values.ndim != 1 or values.size == 0:
        raise BESSConstraintError(
            "selection must be a non-empty one-dimensional vector."
        )
    if not np.all(np.isfinite(values)):
        raise BESSConstraintError(
            "selection contains non-finite values."
        )

    close_to_zero = np.abs(values) <= tolerance
    close_to_one = np.abs(values - 1.0) <= tolerance

    if not np.all(close_to_zero | close_to_one):
        raise BESSConstraintError(
            "selection must contain only binary values."
        )

    binary = np.where(close_to_one, 1, 0).astype(np.int8)

    if int(binary.sum()) != expected_count:
        raise BESSConstraintError(
            f"selection contains {int(binary.sum())} selected buses; "
            f"expected {expected_count}."
        )

    binary.setflags(write=False)
    return binary


@dataclass(frozen=True, slots=True, init=False)
class BESSPlacement:
    """Validated selected buses for one fleet specification."""

    fleet: BESSFleetSpec
    selection: NDArray[np.int8]
    selected_buses: tuple[Any, ...]

    def __init__(
        self,
        fleet: BESSFleetSpec,
        selection: ArrayLike,
    ) -> None:
        if not isinstance(fleet, BESSFleetSpec):
            raise TypeError("fleet must be BESSFleetSpec.")

        binary = validate_binary_selection(
            selection,
            expected_count=fleet.units_to_place,
        )

        if binary.size != fleet.candidate_count:
            raise BESSConstraintError(
                "selection length must equal candidate_count."
            )

        selected = tuple(
            bus
            for bus, value in zip(
                fleet.candidate_buses,
                binary,
            )
            if int(value) == 1
        )

        object.__setattr__(self, "fleet", fleet)
        object.__setattr__(self, "selection", binary)
        object.__setattr__(
            self,
            "selected_buses",
            selected,
        )

    @classmethod
    def from_selected_buses(
        cls,
        fleet: BESSFleetSpec,
        selected_buses: Sequence[Any],
    ) -> "BESSPlacement":
        selected = tuple(selected_buses)

        if len(set(selected)) != len(selected):
            raise BESSConstraintError(
                "selected_buses must be unique."
            )

        unknown = set(selected) - set(fleet.candidate_buses)
        if unknown:
            raise BESSConstraintError(
                f"Selected buses are not candidates: "
                f"{sorted(unknown, key=str)!r}."
            )

        selected_set = set(selected)
        selection = np.array(
            [
                1 if bus in selected_set else 0
                for bus in fleet.candidate_buses
            ],
            dtype=np.int8,
        )

        return cls(fleet, selection)


@dataclass(frozen=True, slots=True, init=False)
class BESSDispatchSchedule:
    """Validated multi-period dispatch for one placement."""

    placement: BESSPlacement
    timestep_hours: float
    p_charge_mw: NDArray[np.float64]
    p_discharge_mw: NDArray[np.float64]
    energy_mwh: NDArray[np.float64]
    terminal_soc_target: float | None
    terminal_soc_tolerance: float

    def __init__(
        self,
        placement: BESSPlacement,
        *,
        timestep_hours: float,
        p_charge_mw: ArrayLike,
        p_discharge_mw: ArrayLike,
        initial_soc: ArrayLike | None = None,
        terminal_soc_target: float | None = None,
        terminal_soc_tolerance: float = 1.0e-6,
    ) -> None:
        if not isinstance(placement, BESSPlacement):
            raise TypeError("placement must be BESSPlacement.")

        timestep = _positive_float(
            timestep_hours,
            name="timestep_hours",
        )
        n_units = placement.fleet.units_to_place

        charge = _readonly_matrix(
            p_charge_mw,
            name="p_charge_mw",
            expected_columns=n_units,
        )
        discharge = _readonly_matrix(
            p_discharge_mw,
            name="p_discharge_mw",
            expected_columns=n_units,
        )

        if charge.shape != discharge.shape:
            raise BESSConstraintError(
                "p_charge_mw and p_discharge_mw shapes must match."
            )
        if np.any(charge < 0.0) or np.any(discharge < 0.0):
            raise BESSConstraintError(
                "Charge and discharge powers must be non-negative."
            )

        unit = placement.fleet.unit

        if np.any(charge > unit.power_mw + BINARY_TOLERANCE):
            raise BESSConstraintError(
                "Charge power exceeds the unit power limit."
            )
        if np.any(discharge > unit.power_mw + BINARY_TOLERANCE):
            raise BESSConstraintError(
                "Discharge power exceeds the unit power limit."
            )
        if np.any(
            (charge > SIMULTANEOUS_POWER_TOLERANCE_MW)
            & (discharge > SIMULTANEOUS_POWER_TOLERANCE_MW)
        ):
            raise BESSConstraintError(
                "Simultaneous charging and discharging is forbidden."
            )

        if initial_soc is None:
            initial_soc_vector = np.full(
                n_units,
                unit.initial_soc,
                dtype=REAL_DTYPE,
            )
        else:
            initial_soc_vector = _readonly_vector(
                initial_soc,
                name="initial_soc",
                expected_size=n_units,
            )

        if np.any(
            initial_soc_vector < unit.minimum_soc - BINARY_TOLERANCE
        ) or np.any(
            initial_soc_vector > unit.maximum_soc + BINARY_TOLERANCE
        ):
            raise BESSConstraintError(
                "initial_soc values violate unit SOC bounds."
            )

        n_periods = charge.shape[0]
        energy = np.empty(
            (n_periods + 1, n_units),
            dtype=REAL_DTYPE,
        )
        energy[0] = initial_soc_vector * unit.energy_mwh

        retention = (
            1.0 - unit.self_discharge_per_hour
        ) ** timestep

        for period in range(n_periods):
            energy[period + 1] = (
                retention * energy[period]
                + unit.charge_efficiency
                * charge[period]
                * timestep
                - discharge[period]
                * timestep
                / unit.discharge_efficiency
            )

        if np.any(
            energy < unit.minimum_energy_mwh - BINARY_TOLERANCE
        ) or np.any(
            energy > unit.maximum_energy_mwh + BINARY_TOLERANCE
        ):
            raise BESSConstraintError(
                "Dispatch violates BESS energy/SOC bounds."
            )

        normalized_target: float | None
        if terminal_soc_target is None:
            normalized_target = None
        else:
            normalized_target = _probability(
                terminal_soc_target,
                name="terminal_soc_target",
            )
            if not (
                unit.minimum_soc
                <= normalized_target
                <= unit.maximum_soc
            ):
                raise BESSConstraintError(
                    "terminal_soc_target violates unit SOC bounds."
                )

        terminal_tolerance = _positive_float(
            terminal_soc_tolerance,
            name="terminal_soc_tolerance",
        )

        if normalized_target is not None:
            terminal_soc = energy[-1] / unit.energy_mwh
            if np.any(
                np.abs(terminal_soc - normalized_target)
                > terminal_tolerance
            ):
                raise BESSConstraintError(
                    "Terminal SOC target is not satisfied."
                )

        energy = np.ascontiguousarray(
            energy,
            dtype=REAL_DTYPE,
        )
        energy.setflags(write=False)

        object.__setattr__(self, "placement", placement)
        object.__setattr__(
            self,
            "timestep_hours",
            timestep,
        )
        object.__setattr__(
            self,
            "p_charge_mw",
            charge,
        )
        object.__setattr__(
            self,
            "p_discharge_mw",
            discharge,
        )
        object.__setattr__(self, "energy_mwh", energy)
        object.__setattr__(
            self,
            "terminal_soc_target",
            normalized_target,
        )
        object.__setattr__(
            self,
            "terminal_soc_tolerance",
            terminal_tolerance,
        )

    @property
    def n_periods(self) -> int:
        return int(self.p_charge_mw.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.p_charge_mw.shape[1])

    @property
    def net_grid_injection_mw(self) -> NDArray[np.float64]:
        """Positive values inject active power into the grid."""

        result = np.ascontiguousarray(
            self.p_discharge_mw - self.p_charge_mw,
            dtype=REAL_DTYPE,
        )
        result.setflags(write=False)
        return result

    @property
    def state_of_charge(self) -> NDArray[np.float64]:
        result = np.ascontiguousarray(
            self.energy_mwh
            / self.placement.fleet.unit.energy_mwh,
            dtype=REAL_DTYPE,
        )
        result.setflags(write=False)
        return result

    @property
    def total_grid_injection_mw(self) -> NDArray[np.float64]:
        result = np.ascontiguousarray(
            self.net_grid_injection_mw.sum(axis=1),
            dtype=REAL_DTYPE,
        )
        result.setflags(write=False)
        return result


__all__ = [
    "REAL_DTYPE",
    "BINARY_TOLERANCE",
    "SIMULTANEOUS_POWER_TOLERANCE_MW",
    "BESSConstraintError",
    "BESSUnitSpec",
    "BESSFleetSpec",
    "validate_candidate_buses",
    "validate_binary_selection",
    "BESSPlacement",
    "BESSDispatchSchedule",
]
