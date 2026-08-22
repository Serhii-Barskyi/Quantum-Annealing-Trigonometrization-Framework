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

"""Strict IEEE/MATPOWER case loading through pandapower.

The module imports pandapower lazily and never downloads data, runs a solver,
caches a mutable network, or substitutes a different case after an error.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib
import math
from typing import Any, Final, Mapping

import numpy as np
import pandas as pd


SUPPORTED_CASE_FACTORIES: Final[Mapping[str, str]] = {
    "case14": "case14",
    "case30": "case30",
    "case57": "case57",
    "case118": "case118",
    "case300": "case300",
}

FINGERPRINT_TABLES: Final[tuple[str, ...]] = (
    "bus",
    "line",
    "trafo",
    "trafo3w",
    "load",
    "gen",
    "sgen",
    "ext_grid",
    "storage",
    "shunt",
    "ward",
    "xward",
    "impedance",
    "dcline",
    "switch",
)

BUS_REFERENCE_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
    "line": ("from_bus", "to_bus"),
    "trafo": ("hv_bus", "lv_bus"),
    "trafo3w": ("hv_bus", "mv_bus", "lv_bus"),
    "load": ("bus",),
    "gen": ("bus",),
    "sgen": ("bus",),
    "ext_grid": ("bus",),
    "storage": ("bus",),
    "shunt": ("bus",),
    "ward": ("bus",),
    "xward": ("bus",),
    "impedance": ("from_bus", "to_bus"),
    "dcline": ("from_bus", "to_bus"),
    "switch": ("bus",),
}


class PowerCaseError(RuntimeError):
    """Raised when a power-system case cannot be loaded or validated."""


def normalize_case_name(case_name: str) -> str:
    """Validate and normalize a registered case name."""

    if not isinstance(case_name, str):
        raise TypeError("case_name must be a string.")

    normalized = case_name.strip().lower()

    if normalized not in SUPPORTED_CASE_FACTORIES:
        supported = ", ".join(SUPPORTED_CASE_FACTORIES)
        raise PowerCaseError(
            f"Unsupported case {case_name!r}. Supported cases: {supported}."
        )

    return normalized


def available_power_cases() -> tuple[str, ...]:
    """Return the deterministic case registry."""

    return tuple(SUPPORTED_CASE_FACTORIES)


def _table(network: Any, table_name: str) -> pd.DataFrame | None:
    """Return one network table without assuming one access style."""

    if hasattr(network, table_name):
        value = getattr(network, table_name)
    elif isinstance(network, Mapping):
        value = network.get(table_name)
    else:
        try:
            value = network[table_name]
        except (KeyError, TypeError, AttributeError):
            value = None

    if value is None:
        return None
    if not isinstance(value, pd.DataFrame):
        raise PowerCaseError(
            f"Network table {table_name!r} must be a pandas DataFrame."
        )

    return value


def _required_table(network: Any, table_name: str) -> pd.DataFrame:
    table = _table(network, table_name)
    if table is None:
        raise PowerCaseError(
            f"Network is missing required table {table_name!r}."
        )
    return table


def _numeric_column(
    table: pd.DataFrame,
    table_name: str,
    column_name: str,
) -> np.ndarray:
    if column_name not in table.columns:
        raise PowerCaseError(
            f"Table {table_name!r} is missing column {column_name!r}."
        )

    try:
        values = pd.to_numeric(
            table[column_name],
            errors="raise",
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PowerCaseError(
            f"{table_name}.{column_name} must be numeric."
        ) from exc

    if not np.all(np.isfinite(values)):
        raise PowerCaseError(
            f"{table_name}.{column_name} contains non-finite values."
        )

    return values


def _has_slack_source(network: Any) -> bool:
    ext_grid = _table(network, "ext_grid")

    if ext_grid is not None and not ext_grid.empty:
        if "in_service" not in ext_grid.columns:
            return True
        if bool(ext_grid["in_service"].fillna(True).astype(bool).any()):
            return True

    gen = _table(network, "gen")

    if gen is None or gen.empty or "slack" not in gen.columns:
        return False

    slack = gen["slack"].fillna(False).astype(bool)

    if "in_service" in gen.columns:
        slack &= gen["in_service"].fillna(True).astype(bool)

    return bool(slack.any())


def validate_power_case(network: Any) -> None:
    """Validate structural invariants required by the OPF pipeline."""

    if network is None:
        raise PowerCaseError("network must not be None.")

    bus = _required_table(network, "bus")

    if bus.empty:
        raise PowerCaseError("The bus table must not be empty.")
    if not bus.index.is_unique:
        raise PowerCaseError("Bus indices must be unique.")
    if bus.index.hasnans:
        raise PowerCaseError("Bus indices must not contain missing values.")

    voltage = _numeric_column(bus, "bus", "vn_kv")
    if np.any(voltage <= 0.0):
        raise PowerCaseError(
            "Every bus.vn_kv value must be strictly positive."
        )

    valid_buses = set(bus.index.tolist())

    for table_name, columns in BUS_REFERENCE_COLUMNS.items():
        table = _table(network, table_name)

        if table is None or table.empty:
            continue

        for column_name in columns:
            if column_name not in table.columns:
                raise PowerCaseError(
                    f"Table {table_name!r} is missing "
                    f"bus-reference column {column_name!r}."
                )

            missing = (
                set(table[column_name].dropna().tolist())
                - valid_buses
            )
            if missing:
                preview = sorted(missing, key=str)[:5]
                raise PowerCaseError(
                    f"{table_name}.{column_name} references unknown buses: "
                    f"{preview!r}."
                )

    line = _table(network, "line")
    if line is not None and not line.empty and "length_km" in line.columns:
        length = _numeric_column(line, "line", "length_km")
        if np.any(length <= 0.0):
            raise PowerCaseError(
                "Every line.length_km value must be positive."
            )

    for table_name in ("load", "gen", "sgen", "storage"):
        table = _table(network, table_name)

        if table is None or table.empty:
            continue

        for column_name in ("p_mw", "q_mvar"):
            if column_name in table.columns:
                _numeric_column(table, table_name, column_name)

    if not _has_slack_source(network):
        raise PowerCaseError(
            "Network must contain an active ext_grid or slack generator."
        )


def _canonical_table_bytes(
    table_name: str,
    table: pd.DataFrame,
) -> bytes:
    columns = sorted(table.columns.tolist(), key=str)
    normalized = table.loc[:, columns].copy()
    normalized = normalized.sort_index(
        kind="mergesort",
        key=lambda index: index.map(str),
    )

    dtypes = "|".join(
        f"{column!s}:{normalized[column].dtype!s}"
        for column in columns
    )
    csv_text = normalized.to_csv(
        index=True,
        header=True,
        na_rep="<NA>",
        float_format="%.17g",
        lineterminator="\n",
    )

    return f"[{table_name}]\n{dtypes}\n{csv_text}".encode("utf-8")


def power_case_fingerprint(network: Any) -> str:
    """Return a deterministic SHA-256 of validated input tables."""

    validate_power_case(network)

    digest = hashlib.sha256()
    digest.update(b"CSSF-PowerCase-v1\0")

    found = 0
    for table_name in FINGERPRINT_TABLES:
        table = _table(network, table_name)
        if table is None:
            continue
        digest.update(_canonical_table_bytes(table_name, table))
        found += 1

    if found == 0:
        raise PowerCaseError("No fingerprintable tables were found.")

    return digest.hexdigest()


def _row_count(network: Any, table_name: str) -> int:
    table = _table(network, table_name)
    return 0 if table is None else int(len(table))


def _sum_column(
    network: Any,
    table_name: str,
    column_name: str,
) -> float:
    table = _table(network, table_name)

    if table is None or table.empty or column_name not in table.columns:
        return 0.0

    return float(_numeric_column(table, table_name, column_name).sum())


@dataclass(frozen=True, slots=True)
class PowerCaseSummary:
    """Immutable network-size and active-power summary."""

    case_name: str
    n_bus: int
    n_line: int
    n_trafo: int
    n_trafo3w: int
    n_load: int
    n_gen: int
    n_sgen: int
    n_ext_grid: int
    n_storage: int
    total_load_p_mw: float
    scheduled_generation_p_mw: float
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_name",
            normalize_case_name(self.case_name),
        )

        for field_name in (
            "n_bus",
            "n_line",
            "n_trafo",
            "n_trafo3w",
            "n_load",
            "n_gen",
            "n_sgen",
            "n_ext_grid",
            "n_storage",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise PowerCaseError(
                    f"{field_name} must be non-negative."
                )

        if self.n_bus < 1:
            raise PowerCaseError("n_bus must be positive.")

        for field_name in (
            "total_load_p_mw",
            "scheduled_generation_p_mw",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise PowerCaseError(
                    f"{field_name} must be finite."
                )
            object.__setattr__(self, field_name, value)

        if len(self.fingerprint) != 64:
            raise PowerCaseError(
                "fingerprint must be a SHA-256 digest."
            )

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


def summarize_power_case(
    case_name: str,
    network: Any,
) -> PowerCaseSummary:
    """Validate and summarize one loaded network."""

    normalized = normalize_case_name(case_name)
    validate_power_case(network)

    return PowerCaseSummary(
        case_name=normalized,
        n_bus=_row_count(network, "bus"),
        n_line=_row_count(network, "line"),
        n_trafo=_row_count(network, "trafo"),
        n_trafo3w=_row_count(network, "trafo3w"),
        n_load=_row_count(network, "load"),
        n_gen=_row_count(network, "gen"),
        n_sgen=_row_count(network, "sgen"),
        n_ext_grid=_row_count(network, "ext_grid"),
        n_storage=_row_count(network, "storage"),
        total_load_p_mw=_sum_column(network, "load", "p_mw"),
        scheduled_generation_p_mw=(
            _sum_column(network, "gen", "p_mw")
            + _sum_column(network, "sgen", "p_mw")
            + _sum_column(network, "storage", "p_mw")
        ),
        fingerprint=power_case_fingerprint(network),
    )


@dataclass(frozen=True, slots=True)
class LoadedPowerCase:
    """Loaded mutable pandapower network with immutable metadata."""

    name: str
    network: Any
    summary: PowerCaseSummary
    source: str = "pandapower.networks"

    def __post_init__(self) -> None:
        normalized = normalize_case_name(self.name)
        object.__setattr__(self, "name", normalized)

        if self.summary.case_name != normalized:
            raise PowerCaseError(
                "summary.case_name must match name."
            )
        if not isinstance(self.source, str) or not self.source.strip():
            raise PowerCaseError("source must be a non-empty string.")

    def clone_network(self) -> Any:
        """Return an independent deep copy for scenario mutation."""

        return copy.deepcopy(self.network)


def load_power_case(
    case_name: str = "case300",
) -> LoadedPowerCase:
    """Load exactly one registered network from pandapower.networks."""

    normalized = normalize_case_name(case_name)

    try:
        networks = importlib.import_module("pandapower.networks")
    except ImportError as exc:
        raise PowerCaseError(
            "pandapower is unavailable. Install requirements-colab.txt "
            "in Google Colab and restart the runtime."
        ) from exc

    factory_name = SUPPORTED_CASE_FACTORIES[normalized]
    factory = getattr(networks, factory_name, None)

    if factory is None or not callable(factory):
        raise PowerCaseError(
            f"pandapower.networks has no callable {factory_name!r}."
        )

    try:
        network = factory()
    except Exception as exc:
        raise PowerCaseError(
            f"Failed to construct {normalized!r} through "
            f"pandapower.networks.{factory_name}()."
        ) from exc

    return LoadedPowerCase(
        name=normalized,
        network=network,
        summary=summarize_power_case(normalized, network),
    )


__all__ = [
    "SUPPORTED_CASE_FACTORIES",
    "FINGERPRINT_TABLES",
    "BUS_REFERENCE_COLUMNS",
    "PowerCaseError",
    "normalize_case_name",
    "available_power_cases",
    "validate_power_case",
    "power_case_fingerprint",
    "PowerCaseSummary",
    "summarize_power_case",
    "LoadedPowerCase",
    "load_power_case",
]
