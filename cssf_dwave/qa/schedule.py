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

"""Validated annealing schedules and QA-to-MA-QAOA interval mapping.

The project direction is deliberately one-way::

    QA schedule -> digitized interval integrals -> MA-QAOA coordinates

This module does not claim physical equivalence between a D-Wave processor and
MA-QAOA. It provides a deterministic research mapping for the digitized-QA
teacher level described by the CSSF architecture.

Schedule amplitudes are represented on normalized time ``s in [0, 1]``. The
physical/model annealing duration multiplies every interval integral. The
Hamiltonian convention absorbs ``hbar`` into the amplitude units, so the
resulting integrals are dimensionless rotation coordinates.

A single driver/problem channel is broadcast only when explicitly mapped to a
compatible MA-QAOA layout. Term-wise schedules may instead supply one driver
channel per qubit and one problem channel per non-identity Ising cost term.
No Qiskit, Aer, Ocean, simulator, optimizer, filesystem, or network operation
is performed during import.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
import math
from typing import Final, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config.schema import QAConfig, TrotterOrder
from maqaoa.parameters import (
    MAQAOAParameterLayout,
    MAQAOAParameterValues,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
NORMALIZED_TIME_START: Final[float] = 0.0
NORMALIZED_TIME_END: Final[float] = 1.0
MINIMUM_SCHEDULE_POINTS: Final[int] = 3
MINIMUM_TROTTER_SLICES: Final[int] = 2
MAXIMUM_TROTTER_SLICES: Final[int] = 4096
SUPPORTED_TROTTER_ORDERS: Final[tuple[int, int]] = (1, 2)
DEFAULT_SCHEDULE_NAME: Final[str] = "linear_forward_anneal"
QA_TO_MAQAOA_DIRECTION: Final[str] = "qa_to_maqaoa"


class QAScheduleError(ValueError):
    """Raised when an annealing schedule or digitization is invalid."""


def _finite_positive(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise QAScheduleError(f"{name} must be finite and positive.")
    return normalized


def _positive_integer(
    value: int,
    *,
    name: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise QAScheduleError(
            f"{name} must be at least {minimum}; received {value}."
        )
    if maximum is not None and value > maximum:
        raise QAScheduleError(
            f"{name} must not exceed {maximum}; received {value}."
        )
    return value


def _nonempty_token(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise QAScheduleError(f"{name} must be non-empty.")
    return normalized


def _readonly_time_grid(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.ndim != 1:
        raise QAScheduleError(
            "normalized_time must be a one-dimensional array."
        )
    if array.size < MINIMUM_SCHEDULE_POINTS:
        raise QAScheduleError(
            "normalized_time must contain at least "
            f"{MINIMUM_SCHEDULE_POINTS} points."
        )
    if not np.all(np.isfinite(array)):
        raise QAScheduleError("normalized_time contains non-finite values.")

    result = np.array(array, dtype=REAL_DTYPE, order="C", copy=True)
    result[result == 0.0] = 0.0
    if result[0] != NORMALIZED_TIME_START:
        raise QAScheduleError("normalized_time must start exactly at 0.0.")
    if result[-1] != NORMALIZED_TIME_END:
        raise QAScheduleError("normalized_time must end exactly at 1.0.")
    if np.any(np.diff(result) <= 0.0):
        raise QAScheduleError("normalized_time must be strictly increasing.")

    result.setflags(write=False)
    return result


def _readonly_amplitudes(
    values: ArrayLike,
    *,
    point_count: int,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise QAScheduleError(
            f"{name} must be one- or two-dimensional."
        )
    if array.shape[0] != point_count:
        raise QAScheduleError(
            f"{name} has {array.shape[0]} rows; expected {point_count}."
        )
    if array.shape[1] < 1:
        raise QAScheduleError(f"{name} must contain at least one channel.")
    if not np.all(np.isfinite(array)):
        raise QAScheduleError(f"{name} contains non-finite values.")
    if np.any(array < 0.0):
        raise QAScheduleError(f"{name} must be non-negative.")

    result = np.array(array, dtype=REAL_DTYPE, order="C", copy=True)
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _readonly_matrix(
    values: ArrayLike,
    *,
    row_count: int,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.ndim != 2 or array.shape[0] != row_count:
        raise QAScheduleError(
            f"{name} must be a matrix with {row_count} rows; "
            f"received shape {array.shape}."
        )
    if array.shape[1] < 1:
        raise QAScheduleError(f"{name} must contain at least one channel.")
    if not np.all(np.isfinite(array)):
        raise QAScheduleError(f"{name} contains non-finite values.")
    if np.any(array < 0.0):
        raise QAScheduleError(f"{name} must be non-negative.")

    result = np.array(array, dtype=REAL_DTYPE, order="C", copy=True)
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _readonly_boundaries(
    values: ArrayLike,
    *,
    slice_count: int,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)
    expected_size = slice_count + 1
    if array.ndim != 1 or array.size != expected_size:
        raise QAScheduleError(
            "slice_boundaries must be a one-dimensional array of length "
            f"{expected_size}; received shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise QAScheduleError("slice_boundaries contains non-finite values.")

    result = np.array(array, dtype=REAL_DTYPE, order="C", copy=True)
    result[result == 0.0] = 0.0
    if result[0] != NORMALIZED_TIME_START:
        raise QAScheduleError("slice_boundaries must start exactly at 0.0.")
    if result[-1] != NORMALIZED_TIME_END:
        raise QAScheduleError("slice_boundaries must end exactly at 1.0.")
    if np.any(np.diff(result) <= 0.0):
        raise QAScheduleError(
            "slice_boundaries must be strictly increasing."
        )

    result.setflags(write=False)
    return result


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _normalize_trotter_order(value: int | TrotterOrder) -> int:
    if isinstance(value, TrotterOrder):
        normalized = int(value.value)
    elif isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("trotter_order must be TrotterOrder or integer.")
    else:
        normalized = value
    if normalized not in SUPPORTED_TROTTER_ORDERS:
        raise QAScheduleError(
            "trotter_order must be first or second order."
        )
    return normalized


def _hash_array(digest: "hashlib._Hash", array: NDArray[np.float64]) -> None:
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))


def _piecewise_linear_primitive(
    grid: NDArray[np.float64],
    values: NDArray[np.float64],
    query: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate exact antiderivatives of sampled piecewise-linear channels."""

    widths = np.diff(grid)
    slopes = np.diff(values, axis=0) / widths[:, None]
    segment_integrals = 0.5 * (
        values[:-1, :] + values[1:, :]
    ) * widths[:, None]
    cumulative = np.vstack(
        (
            np.zeros((1, values.shape[1]), dtype=REAL_DTYPE),
            np.cumsum(segment_integrals, axis=0, dtype=REAL_DTYPE),
        )
    )

    indices = np.searchsorted(grid, query, side="right") - 1
    indices = np.clip(indices, 0, grid.size - 2)
    delta = query - grid[indices]
    return (
        cumulative[indices, :]
        + values[indices, :] * delta[:, None]
        + 0.5 * slopes[indices, :] * delta[:, None] ** 2
    )


def _piecewise_linear_integrals(
    grid: NDArray[np.float64],
    values: NDArray[np.float64],
    boundaries: NDArray[np.float64],
    *,
    total_annealing_time: float,
) -> NDArray[np.float64]:
    primitive = _piecewise_linear_primitive(grid, values, boundaries)
    integrals = total_annealing_time * np.diff(primitive, axis=0)
    integrals[integrals == 0.0] = 0.0
    tolerance = 64.0 * np.finfo(REAL_DTYPE).eps
    if np.any(integrals < -tolerance):
        raise QAScheduleError(
            "Piecewise-linear integration produced a negative interval."
        )
    integrals[integrals < 0.0] = 0.0
    return np.asarray(integrals, dtype=REAL_DTYPE)


def _broadcast_channels(
    values: NDArray[np.float64],
    *,
    expected_channels: int,
    name: str,
) -> NDArray[np.float64]:
    actual_channels = values.shape[1]
    if actual_channels == expected_channels:
        expanded = np.array(values, dtype=REAL_DTYPE, order="C", copy=True)
    elif actual_channels == 1:
        expanded = np.repeat(values, expected_channels, axis=1)
    else:
        raise QAScheduleError(
            f"{name} has {actual_channels} channels; expected one shared "
            f"channel or {expected_channels} term-wise channels."
        )
    expanded.setflags(write=False)
    return expanded


@dataclass(frozen=True, slots=True, init=False)
class AnnealingSchedule:
    """Immutable sampled forward-annealing schedule on normalized time."""

    normalized_time: NDArray[np.float64]
    driver_amplitudes: NDArray[np.float64]
    problem_amplitudes: NDArray[np.float64]
    total_annealing_time: float
    name: str
    metadata: Mapping[str, object]

    def __init__(
        self,
        *,
        normalized_time: ArrayLike,
        driver_amplitudes: ArrayLike,
        problem_amplitudes: ArrayLike,
        total_annealing_time: float,
        name: str = DEFAULT_SCHEDULE_NAME,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        time_grid = _readonly_time_grid(normalized_time)
        driver = _readonly_amplitudes(
            driver_amplitudes,
            point_count=time_grid.size,
            name="driver_amplitudes",
        )
        problem = _readonly_amplitudes(
            problem_amplitudes,
            point_count=time_grid.size,
            name="problem_amplitudes",
        )
        duration = _finite_positive(
            total_annealing_time,
            name="total_annealing_time",
        )
        normalized_name = _nonempty_token(name, name="name")

        raw_metadata = {} if metadata is None else dict(metadata)
        try:
            detached_metadata = json.loads(
                json.dumps(
                    raw_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise QAScheduleError(
                "metadata must be JSON-serializable without NaN values."
            ) from exc

        object.__setattr__(self, "normalized_time", time_grid)
        object.__setattr__(self, "driver_amplitudes", driver)
        object.__setattr__(self, "problem_amplitudes", problem)
        object.__setattr__(self, "total_annealing_time", duration)
        object.__setattr__(self, "name", normalized_name)
        frozen_metadata = _freeze_json(detached_metadata)
        if not isinstance(frozen_metadata, Mapping):
            raise QAScheduleError("metadata must be a JSON object.")
        object.__setattr__(self, "metadata", frozen_metadata)

    @property
    def point_count(self) -> int:
        return int(self.normalized_time.size)

    @property
    def driver_channel_count(self) -> int:
        return int(self.driver_amplitudes.shape[1])

    @property
    def problem_channel_count(self) -> int:
        return int(self.problem_amplitudes.shape[1])

    @property
    def physical_time(self) -> NDArray[np.float64]:
        values = np.array(
            self.normalized_time * self.total_annealing_time,
            dtype=REAL_DTYPE,
            order="C",
            copy=True,
        )
        values.setflags(write=False)
        return values

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-AnnealingSchedule-v1\0")
        digest.update(self.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.total_annealing_time.hex().encode("ascii"))
        digest.update(b"\0")
        _hash_array(digest, self.normalized_time)
        _hash_array(digest, self.driver_amplitudes)
        _hash_array(digest, self.problem_amplitudes)
        digest.update(
            json.dumps(
                _thaw_json(self.metadata),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def digitize(
        self,
        *,
        trotter_slices: int,
        trotter_order: int | TrotterOrder,
    ) -> "DigitizedQASchedule":
        """Integrate all channels exactly over uniform Trotter intervals."""

        slice_count = _positive_integer(
            trotter_slices,
            name="trotter_slices",
            minimum=MINIMUM_TROTTER_SLICES,
            maximum=MAXIMUM_TROTTER_SLICES,
        )
        order = _normalize_trotter_order(trotter_order)
        boundaries = np.linspace(
            NORMALIZED_TIME_START,
            NORMALIZED_TIME_END,
            slice_count + 1,
            dtype=REAL_DTYPE,
        )
        driver_integrals = _piecewise_linear_integrals(
            self.normalized_time,
            self.driver_amplitudes,
            boundaries,
            total_annealing_time=self.total_annealing_time,
        )
        problem_integrals = _piecewise_linear_integrals(
            self.normalized_time,
            self.problem_amplitudes,
            boundaries,
            total_annealing_time=self.total_annealing_time,
        )
        return DigitizedQASchedule(
            slice_boundaries=boundaries,
            driver_integrals=driver_integrals,
            problem_integrals=problem_integrals,
            trotter_order=order,
            source_schedule_fingerprint=self.fingerprint(),
            source_schedule_name=self.name,
            total_annealing_time=self.total_annealing_time,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "cssf-annealing-schedule-v1",
            "mapping_direction": QA_TO_MAQAOA_DIRECTION,
            "name": self.name,
            "point_count": self.point_count,
            "driver_channel_count": self.driver_channel_count,
            "problem_channel_count": self.problem_channel_count,
            "total_annealing_time": self.total_annealing_time,
            "fingerprint": self.fingerprint(),
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True, init=False)
class DigitizedQASchedule:
    """Immutable interval integrals for a first- or second-order QA model."""

    slice_boundaries: NDArray[np.float64]
    driver_integrals: NDArray[np.float64]
    problem_integrals: NDArray[np.float64]
    trotter_order: int
    source_schedule_fingerprint: str
    source_schedule_name: str
    total_annealing_time: float

    def __init__(
        self,
        *,
        slice_boundaries: ArrayLike,
        driver_integrals: ArrayLike,
        problem_integrals: ArrayLike,
        trotter_order: int | TrotterOrder,
        source_schedule_fingerprint: str,
        source_schedule_name: str,
        total_annealing_time: float,
    ) -> None:
        boundary_array = np.asarray(slice_boundaries, dtype=REAL_DTYPE)
        if boundary_array.ndim != 1 or boundary_array.size < 3:
            raise QAScheduleError(
                "slice_boundaries must define at least two slices."
            )
        slice_count = _positive_integer(
            int(boundary_array.size - 1),
            name="slice_count",
            minimum=MINIMUM_TROTTER_SLICES,
            maximum=MAXIMUM_TROTTER_SLICES,
        )
        boundaries = _readonly_boundaries(
            boundary_array,
            slice_count=slice_count,
        )
        driver = _readonly_matrix(
            driver_integrals,
            row_count=slice_count,
            name="driver_integrals",
        )
        problem = _readonly_matrix(
            problem_integrals,
            row_count=slice_count,
            name="problem_integrals",
        )
        order = _normalize_trotter_order(trotter_order)
        digest = str(source_schedule_fingerprint)
        if len(digest) != 64:
            raise QAScheduleError(
                "source_schedule_fingerprint must be a SHA-256 digest."
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise QAScheduleError(
                "source_schedule_fingerprint must be hexadecimal."
            ) from exc

        object.__setattr__(self, "slice_boundaries", boundaries)
        object.__setattr__(self, "driver_integrals", driver)
        object.__setattr__(self, "problem_integrals", problem)
        object.__setattr__(self, "trotter_order", order)
        object.__setattr__(
            self,
            "source_schedule_fingerprint",
            digest.lower(),
        )
        object.__setattr__(
            self,
            "source_schedule_name",
            _nonempty_token(
                source_schedule_name,
                name="source_schedule_name",
            ),
        )
        object.__setattr__(
            self,
            "total_annealing_time",
            _finite_positive(
                total_annealing_time,
                name="total_annealing_time",
            ),
        )

    @property
    def slice_count(self) -> int:
        return int(self.slice_boundaries.size - 1)

    @property
    def driver_channel_count(self) -> int:
        return int(self.driver_integrals.shape[1])

    @property
    def problem_channel_count(self) -> int:
        return int(self.problem_integrals.shape[1])

    @property
    def interval_widths(self) -> NDArray[np.float64]:
        widths = np.array(
            np.diff(self.slice_boundaries),
            dtype=REAL_DTYPE,
            order="C",
            copy=True,
        )
        widths.setflags(write=False)
        return widths

    def to_maqaoa_values(
        self,
        layout: MAQAOAParameterLayout,
    ) -> MAQAOAParameterValues:
        """Map interval integrals to one exact MA-QAOA parameter layout.

        ``gamma`` contains problem-envelope integrals only. Ising coefficients
        remain owned by the layout/circuit and are not multiplied a second
        time. ``beta`` contains driver-envelope integrals.
        """

        if not isinstance(layout, MAQAOAParameterLayout):
            raise TypeError("layout must be MAQAOAParameterLayout.")
        if layout.repetitions != self.slice_count:
            raise QAScheduleError(
                "MA-QAOA repetitions must equal the digitized slice count: "
                f"{layout.repetitions} != {self.slice_count}."
            )

        gamma = _broadcast_channels(
            self.problem_integrals,
            expected_channels=layout.cost_term_count,
            name="problem_integrals",
        )
        beta = _broadcast_channels(
            self.driver_integrals,
            expected_channels=layout.n_qubits,
            name="driver_integrals",
        )
        return layout.values(gamma=gamma, beta=beta)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-DigitizedQASchedule-v1\0")
        digest.update(self.source_schedule_fingerprint.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.source_schedule_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(self.trotter_order).encode("ascii"))
        digest.update(b"\0")
        digest.update(self.total_annealing_time.hex().encode("ascii"))
        digest.update(b"\0")
        _hash_array(digest, self.slice_boundaries)
        _hash_array(digest, self.driver_integrals)
        _hash_array(digest, self.problem_integrals)
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "cssf-digitized-qa-schedule-v1",
            "mapping_direction": QA_TO_MAQAOA_DIRECTION,
            "source_schedule_name": self.source_schedule_name,
            "source_schedule_fingerprint": (
                self.source_schedule_fingerprint
            ),
            "slice_count": self.slice_count,
            "trotter_order": self.trotter_order,
            "driver_channel_count": self.driver_channel_count,
            "problem_channel_count": self.problem_channel_count,
            "total_annealing_time": self.total_annealing_time,
            "fingerprint": self.fingerprint(),
        }


def build_linear_forward_schedule(
    config: QAConfig,
    *,
    driver_channels: int = 1,
    problem_channels: int = 1,
    name: str = DEFAULT_SCHEDULE_NAME,
    metadata: Mapping[str, object] | None = None,
) -> AnnealingSchedule:
    """Build the deterministic forward schedule ``A=1-s, B=s``."""

    if not isinstance(config, QAConfig):
        raise TypeError("config must be QAConfig.")
    if config.schedule_mapping != QA_TO_MAQAOA_DIRECTION:
        raise QAScheduleError(
            "Only the fixed QA-to-MA-QAOA mapping direction is supported."
        )
    point_count = _positive_integer(
        config.schedule_points,
        name="config.schedule_points",
        minimum=MINIMUM_SCHEDULE_POINTS,
    )
    n_driver = _positive_integer(
        driver_channels,
        name="driver_channels",
    )
    n_problem = _positive_integer(
        problem_channels,
        name="problem_channels",
    )

    normalized_time = np.linspace(
        NORMALIZED_TIME_START,
        NORMALIZED_TIME_END,
        point_count,
        dtype=REAL_DTYPE,
    )
    driver = np.repeat(
        (1.0 - normalized_time)[:, None],
        n_driver,
        axis=1,
    )
    problem = np.repeat(
        normalized_time[:, None],
        n_problem,
        axis=1,
    )
    combined_metadata = {
        "construction": "A(s)=1-s; B(s)=s",
        "schedule_mapping": config.schedule_mapping,
    }
    if metadata is not None:
        combined_metadata.update(dict(metadata))

    return AnnealingSchedule(
        normalized_time=normalized_time,
        driver_amplitudes=driver,
        problem_amplitudes=problem,
        total_annealing_time=config.total_annealing_time,
        name=name,
        metadata=combined_metadata,
    )


def digitize_schedule_from_config(
    schedule: AnnealingSchedule,
    config: QAConfig,
) -> DigitizedQASchedule:
    """Digitize a validated schedule using the strict project QA config."""

    if not isinstance(schedule, AnnealingSchedule):
        raise TypeError("schedule must be AnnealingSchedule.")
    if not isinstance(config, QAConfig):
        raise TypeError("config must be QAConfig.")
    if config.schedule_mapping != QA_TO_MAQAOA_DIRECTION:
        raise QAScheduleError(
            "Only the fixed QA-to-MA-QAOA mapping direction is supported."
        )
    if not math.isclose(
        schedule.total_annealing_time,
        float(config.total_annealing_time),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise QAScheduleError(
            "Schedule duration differs from QAConfig.total_annealing_time."
        )
    return schedule.digitize(
        trotter_slices=config.trotter_slices,
        trotter_order=config.trotter_order,
    )


__all__ = [
    "REAL_DTYPE",
    "NORMALIZED_TIME_START",
    "NORMALIZED_TIME_END",
    "MINIMUM_SCHEDULE_POINTS",
    "MINIMUM_TROTTER_SLICES",
    "MAXIMUM_TROTTER_SLICES",
    "SUPPORTED_TROTTER_ORDERS",
    "DEFAULT_SCHEDULE_NAME",
    "QA_TO_MAQAOA_DIRECTION",
    "QAScheduleError",
    "AnnealingSchedule",
    "DigitizedQASchedule",
    "build_linear_forward_schedule",
    "digitize_schedule_from_config",
]
