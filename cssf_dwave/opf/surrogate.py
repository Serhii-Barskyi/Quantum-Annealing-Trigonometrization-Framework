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

"""Dataset construction and physical prediction for CSNN-T^OPF.

This module connects deterministic scenario factors, AC-OPF outputs, toric
Fourier features, and invertible target transforms. It does not run an OPF
solver and does not alter the frozen CSNN-T implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.dataset import CSSFDataset
from opf.acopf import ACOPFResult
from opf.scenario_generator import ScenarioBatch
from spectral.feature_matrix import toric_feature_matrix
from spectral.frequency_support import FrequencySupport
from spectral.target_transforms import (
    FittedTargetTransform,
    TransformKind,
    build_transform,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
DEFAULT_ANGLE_LIMIT: Final[float] = 0.95 * math.pi
BOUND_TOLERANCE: Final[float] = 1.0e-12


class OPFSurrogateError(ValueError):
    """Raised when an OPF surrogate dataset violates its contract."""


class OPFTargetField(str, Enum):
    """AC-OPF quantities available as surrogate targets."""

    OBJECTIVE_COST = "objective_cost"
    MAXIMUM_LOADING_PERCENT = "maximum_loading_percent"
    BUS_VM_PU = "bus_vm_pu"
    BUS_VA_DEGREE = "bus_va_degree"
    BUS_P_MW = "bus_p_mw"
    BUS_Q_MVAR = "bus_q_mvar"
    LINE_LOADING_PERCENT = "line_loading_percent"
    TRAFO_LOADING_PERCENT = "trafo_loading_percent"


DEFAULT_TARGET_FIELDS: Final[tuple[OPFTargetField, ...]] = (
    OPFTargetField.OBJECTIVE_COST,
    OPFTargetField.MAXIMUM_LOADING_PERCENT,
    OPFTargetField.BUS_VM_PU,
    OPFTargetField.BUS_VA_DEGREE,
    OPFTargetField.LINE_LOADING_PERCENT,
    OPFTargetField.TRAFO_LOADING_PERCENT,
)


def _readonly_real_matrix(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise OPFSurrogateError(
            f"{name} must be one- or two-dimensional."
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise OPFSurrogateError(f"{name} must be non-empty.")

    result = np.ascontiguousarray(array, dtype=REAL_DTYPE)

    if not np.all(np.isfinite(result)):
        raise OPFSurrogateError(
            f"{name} contains non-finite values."
        )

    result.setflags(write=False)
    return result


def _normalize_fields(
    fields: Sequence[OPFTargetField | str],
) -> tuple[OPFTargetField, ...]:
    normalized: list[OPFTargetField] = []

    for value in fields:
        if isinstance(value, OPFTargetField):
            field = value
        elif isinstance(value, str):
            try:
                field = OPFTargetField(value.strip())
            except ValueError as exc:
                raise OPFSurrogateError(
                    f"Unsupported OPF target field: {value!r}."
                ) from exc
        else:
            raise TypeError(
                "Every target field must be OPFTargetField or str."
            )

        normalized.append(field)

    result = tuple(normalized)

    if not result:
        raise OPFSurrogateError(
            "At least one OPF target field is required."
        )
    if len(set(result)) != len(result):
        raise OPFSurrogateError(
            "OPF target fields must be unique."
        )

    return result


def _index_label(value: Any) -> str:
    return str(value).replace(" ", "_")


@dataclass(frozen=True, slots=True, init=False)
class ScenarioAngleEncoder:
    """Map bounded scenario factors into a non-colliding toric interval."""

    load_count: int
    renewable_count: int
    load_bounds: tuple[float, float]
    renewable_bounds: tuple[float, float]
    angle_limit: float
    coordinate_names: tuple[str, ...]

    def __init__(
        self,
        *,
        load_count: int,
        renewable_count: int,
        load_bounds: Sequence[float],
        renewable_bounds: Sequence[float],
        load_indices: Sequence[Any],
        renewable_indices: Sequence[Any],
        angle_limit: float = DEFAULT_ANGLE_LIMIT,
    ) -> None:
        for name, value in (
            ("load_count", load_count),
            ("renewable_count", renewable_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise OPFSurrogateError(
                    f"{name} must be non-negative."
                )

        if load_count + renewable_count < 1:
            raise OPFSurrogateError(
                "At least one scenario coordinate is required."
            )

        load_index_tuple = tuple(load_indices)
        renewable_index_tuple = tuple(renewable_indices)

        if len(load_index_tuple) != load_count:
            raise OPFSurrogateError(
                "load_indices length must equal load_count."
            )
        if len(renewable_index_tuple) != renewable_count:
            raise OPFSurrogateError(
                "renewable_indices length must equal renewable_count."
            )

        normalized_load_bounds = self._bounds(
            load_bounds,
            name="load_bounds",
        )
        normalized_renewable_bounds = self._bounds(
            renewable_bounds,
            name="renewable_bounds",
        )

        normalized_angle_limit = float(angle_limit)
        if (
            not math.isfinite(normalized_angle_limit)
            or not 0.0 < normalized_angle_limit < math.pi
        ):
            raise OPFSurrogateError(
                "angle_limit must lie strictly in (0, pi)."
            )

        names = tuple(
            [
                f"load_factor[index={_index_label(index)}]"
                for index in load_index_tuple
            ]
            + [
                f"renewable_factor[index={_index_label(index)}]"
                for index in renewable_index_tuple
            ]
        )

        object.__setattr__(self, "load_count", load_count)
        object.__setattr__(
            self,
            "renewable_count",
            renewable_count,
        )
        object.__setattr__(
            self,
            "load_bounds",
            normalized_load_bounds,
        )
        object.__setattr__(
            self,
            "renewable_bounds",
            normalized_renewable_bounds,
        )
        object.__setattr__(
            self,
            "angle_limit",
            normalized_angle_limit,
        )
        object.__setattr__(self, "coordinate_names", names)

    @staticmethod
    def _bounds(
        values: Sequence[float],
        *,
        name: str,
    ) -> tuple[float, float]:
        if len(values) != 2:
            raise OPFSurrogateError(
                f"{name} must contain two values."
            )

        lower = float(values[0])
        upper = float(values[1])

        if not math.isfinite(lower) or not math.isfinite(upper):
            raise OPFSurrogateError(
                f"{name} values must be finite."
            )
        if lower >= upper:
            raise OPFSurrogateError(
                f"{name} must satisfy lower < upper."
            )

        return lower, upper

    @classmethod
    def from_batch(
        cls,
        batch: ScenarioBatch,
        *,
        angle_limit: float = DEFAULT_ANGLE_LIMIT,
    ) -> "ScenarioAngleEncoder":
        if not isinstance(batch, ScenarioBatch):
            raise TypeError("batch must be ScenarioBatch.")

        return cls(
            load_count=len(batch.load_indices),
            renewable_count=len(batch.renewable_indices),
            load_bounds=batch.config.load_bounds,
            renewable_bounds=batch.config.renewable_bounds,
            load_indices=batch.load_indices,
            renewable_indices=batch.renewable_indices,
            angle_limit=angle_limit,
        )

    @property
    def n_dimensions(self) -> int:
        return self.load_count + self.renewable_count

    def _encode_block(
        self,
        values: ArrayLike,
        *,
        count: int,
        bounds: tuple[float, float],
        name: str,
        n_samples: int | None,
    ) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=REAL_DTYPE)

        if array.ndim != 2:
            raise OPFSurrogateError(
                f"{name} must be two-dimensional."
            )
        if array.shape[1] != count:
            raise OPFSurrogateError(
                f"{name} contains {array.shape[1]} columns; "
                f"expected {count}."
            )
        if n_samples is not None and array.shape[0] != n_samples:
            raise OPFSurrogateError(
                f"{name} sample count does not match."
            )
        if not np.all(np.isfinite(array)):
            raise OPFSurrogateError(
                f"{name} contains non-finite values."
            )

        lower, upper = bounds

        if np.any(array < lower - BOUND_TOLERANCE):
            raise OPFSurrogateError(
                f"{name} contains values below {lower}."
            )
        if np.any(array > upper + BOUND_TOLERANCE):
            raise OPFSurrogateError(
                f"{name} contains values above {upper}."
            )

        clipped = np.clip(array, lower, upper)
        center = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)

        return np.ascontiguousarray(
            self.angle_limit * (clipped - center) / half_width,
            dtype=REAL_DTYPE,
        )

    def encode(
        self,
        load_factors: ArrayLike,
        renewable_factors: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return coordinates with shape ``(N, n_dimensions)``."""

        loads = np.asarray(load_factors, dtype=REAL_DTYPE)
        renewables = np.asarray(
            renewable_factors,
            dtype=REAL_DTYPE,
        )

        if loads.ndim != 2 or renewables.ndim != 2:
            raise OPFSurrogateError(
                "Scenario-factor arrays must be two-dimensional."
            )
        if loads.shape[0] != renewables.shape[0]:
            raise OPFSurrogateError(
                "Load and renewable sample counts must match."
            )
        if loads.shape[0] == 0:
            raise OPFSurrogateError(
                "Scenario-factor arrays must be non-empty."
            )

        encoded_loads = self._encode_block(
            loads,
            count=self.load_count,
            bounds=self.load_bounds,
            name="load_factors",
            n_samples=loads.shape[0],
        )
        encoded_renewables = self._encode_block(
            renewables,
            count=self.renewable_count,
            bounds=self.renewable_bounds,
            name="renewable_factors",
            n_samples=loads.shape[0],
        )

        result = np.ascontiguousarray(
            np.column_stack(
                (encoded_loads, encoded_renewables)
            ),
            dtype=REAL_DTYPE,
        )
        result.setflags(write=False)
        return result

    def encode_batch(
        self,
        batch: ScenarioBatch,
    ) -> NDArray[np.float64]:
        if not isinstance(batch, ScenarioBatch):
            raise TypeError("batch must be ScenarioBatch.")
        if len(batch.load_indices) != self.load_count:
            raise OPFSurrogateError(
                "Batch load dimension does not match encoder."
            )
        if len(batch.renewable_indices) != self.renewable_count:
            raise OPFSurrogateError(
                "Batch renewable dimension does not match encoder."
            )

        return self.encode(
            batch.load_factors,
            batch.renewable_factors,
        )


@dataclass(frozen=True, slots=True, init=False)
class OPFTargetLayout:
    """Fixed output topology and deterministic target-column names."""

    fields: tuple[OPFTargetField, ...]
    bus_indices: tuple[Any, ...]
    line_indices: tuple[Any, ...]
    trafo_indices: tuple[Any, ...]
    target_names: tuple[str, ...]

    def __init__(
        self,
        *,
        fields: Sequence[OPFTargetField | str],
        bus_indices: Sequence[Any],
        line_indices: Sequence[Any],
        trafo_indices: Sequence[Any],
    ) -> None:
        normalized_fields = _normalize_fields(fields)
        buses = tuple(bus_indices)
        lines = tuple(line_indices)
        trafos = tuple(trafo_indices)

        names: list[str] = []

        for field in normalized_fields:
            if field is OPFTargetField.OBJECTIVE_COST:
                names.append(field.value)
            elif field is OPFTargetField.MAXIMUM_LOADING_PERCENT:
                names.append(field.value)
            elif field in (
                OPFTargetField.BUS_VM_PU,
                OPFTargetField.BUS_VA_DEGREE,
                OPFTargetField.BUS_P_MW,
                OPFTargetField.BUS_Q_MVAR,
            ):
                names.extend(
                    f"{field.value}[bus={_index_label(index)}]"
                    for index in buses
                )
            elif field is OPFTargetField.LINE_LOADING_PERCENT:
                names.extend(
                    f"{field.value}[line={_index_label(index)}]"
                    for index in lines
                )
            elif field is OPFTargetField.TRAFO_LOADING_PERCENT:
                names.extend(
                    f"{field.value}[trafo={_index_label(index)}]"
                    for index in trafos
                )

        if not names:
            raise OPFSurrogateError(
                "Selected fields produce no target columns."
            )
        if len(set(names)) != len(names):
            raise OPFSurrogateError(
                "Generated target names are not unique."
            )

        object.__setattr__(self, "fields", normalized_fields)
        object.__setattr__(self, "bus_indices", buses)
        object.__setattr__(self, "line_indices", lines)
        object.__setattr__(self, "trafo_indices", trafos)
        object.__setattr__(self, "target_names", tuple(names))

    @classmethod
    def from_result(
        cls,
        result: ACOPFResult,
        *,
        fields: Sequence[OPFTargetField | str] = DEFAULT_TARGET_FIELDS,
    ) -> "OPFTargetLayout":
        if not isinstance(result, ACOPFResult):
            raise TypeError("result must be ACOPFResult.")

        return cls(
            fields=fields,
            bus_indices=result.bus_indices,
            line_indices=result.line_indices,
            trafo_indices=result.trafo_indices,
        )

    @property
    def n_targets(self) -> int:
        return len(self.target_names)

    def _validate_topology(
        self,
        result: ACOPFResult,
    ) -> None:
        if result.bus_indices != self.bus_indices:
            raise OPFSurrogateError(
                "AC-OPF bus topology does not match target layout."
            )
        if result.line_indices != self.line_indices:
            raise OPFSurrogateError(
                "AC-OPF line topology does not match target layout."
            )
        if result.trafo_indices != self.trafo_indices:
            raise OPFSurrogateError(
                "AC-OPF transformer topology does not match target layout."
            )

    def extract(
        self,
        result: ACOPFResult,
    ) -> NDArray[np.float64]:
        """Flatten one AC-OPF result according to this layout."""

        if not isinstance(result, ACOPFResult):
            raise TypeError("result must be ACOPFResult.")

        self._validate_topology(result)
        blocks: list[NDArray[np.float64]] = []

        for field in self.fields:
            if field is OPFTargetField.OBJECTIVE_COST:
                blocks.append(
                    np.array([result.objective_cost], dtype=REAL_DTYPE)
                )
            elif field is OPFTargetField.MAXIMUM_LOADING_PERCENT:
                blocks.append(
                    np.array(
                        [result.maximum_loading_percent()],
                        dtype=REAL_DTYPE,
                    )
                )
            elif field is OPFTargetField.BUS_VM_PU:
                blocks.append(result.bus_vm_pu)
            elif field is OPFTargetField.BUS_VA_DEGREE:
                blocks.append(result.bus_va_degree)
            elif field is OPFTargetField.BUS_P_MW:
                blocks.append(result.bus_p_mw)
            elif field is OPFTargetField.BUS_Q_MVAR:
                blocks.append(result.bus_q_mvar)
            elif field is OPFTargetField.LINE_LOADING_PERCENT:
                blocks.append(result.line_loading_percent)
            elif field is OPFTargetField.TRAFO_LOADING_PERCENT:
                blocks.append(result.trafo_loading_percent)

        values = np.ascontiguousarray(
            np.concatenate(blocks),
            dtype=REAL_DTYPE,
        )

        if values.size != self.n_targets:
            raise OPFSurrogateError(
                "Extracted target size does not match layout."
            )
        if not np.all(np.isfinite(values)):
            raise OPFSurrogateError(
                "Extracted targets contain non-finite values."
            )

        values.setflags(write=False)
        return values

    def extract_matrix(
        self,
        results: Sequence[ACOPFResult],
    ) -> NDArray[np.float64]:
        if not results:
            raise OPFSurrogateError(
                "results must not be empty."
            )

        matrix = np.ascontiguousarray(
            np.vstack(
                [self.extract(result) for result in results]
            ),
            dtype=REAL_DTYPE,
        )
        matrix.setflags(write=False)
        return matrix


@dataclass(frozen=True, slots=True)
class OPFSurrogateDataset:
    """Immutable complete input/output preparation for CSNN-T^OPF."""

    dataset: CSSFDataset
    physical_targets: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    encoder: ScenarioAngleEncoder
    support: FrequencySupport
    target_layout: OPFTargetLayout
    target_transform: FittedTargetTransform
    scenario_ids: tuple[str, ...]
    batch_fingerprint: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, CSSFDataset):
            raise TypeError("dataset must be CSSFDataset.")
        if not isinstance(self.encoder, ScenarioAngleEncoder):
            raise TypeError("encoder must be ScenarioAngleEncoder.")
        if not isinstance(self.support, FrequencySupport):
            raise TypeError("support must be FrequencySupport.")
        if not isinstance(self.target_layout, OPFTargetLayout):
            raise TypeError(
                "target_layout must be OPFTargetLayout."
            )
        if not isinstance(
            self.target_transform,
            FittedTargetTransform,
        ):
            raise TypeError(
                "target_transform must implement FittedTargetTransform."
            )

        physical = _readonly_real_matrix(
            self.physical_targets,
            name="physical_targets",
        )
        coordinates = _readonly_real_matrix(
            self.coordinates,
            name="coordinates",
        )
        scenario_ids = tuple(self.scenario_ids)

        if physical.shape != (
            self.dataset.n_samples,
            self.target_layout.n_targets,
        ):
            raise OPFSurrogateError(
                "physical_targets shape does not match dataset/layout."
            )
        if coordinates.shape != (
            self.dataset.n_samples,
            self.encoder.n_dimensions,
        ):
            raise OPFSurrogateError(
                "coordinates shape does not match dataset/encoder."
            )
        if self.support.n_dimensions != self.encoder.n_dimensions:
            raise OPFSurrogateError(
                "support and encoder dimensions differ."
            )
        if self.dataset.n_features != self.support.n_terms:
            raise OPFSurrogateError(
                "dataset feature count does not match support."
            )
        if self.dataset.n_targets != self.target_layout.n_targets:
            raise OPFSurrogateError(
                "dataset target count does not match layout."
            )
        if self.target_transform.n_targets != self.dataset.n_targets:
            raise OPFSurrogateError(
                "target-transform count does not match dataset."
            )
        if scenario_ids != self.dataset.sample_ids:
            raise OPFSurrogateError(
                "scenario_ids must equal dataset.sample_ids."
            )
        if len(self.batch_fingerprint) != 64:
            raise OPFSurrogateError(
                "batch_fingerprint must be a SHA-256 digest."
            )

        object.__setattr__(
            self,
            "physical_targets",
            physical,
        )
        object.__setattr__(
            self,
            "coordinates",
            coordinates,
        )
        object.__setattr__(
            self,
            "scenario_ids",
            scenario_ids,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    def inverse_targets(
        self,
        transformed_targets: ArrayLike,
    ) -> NDArray[np.float64]:
        """Convert surrogate-coordinate predictions to physical units."""

        return self.target_transform.inverse_transform(
            transformed_targets
        )


def build_opf_surrogate_dataset(
    batch: ScenarioBatch,
    results: Sequence[ACOPFResult],
    support: FrequencySupport,
    *,
    fields: Sequence[OPFTargetField | str] = DEFAULT_TARGET_FIELDS,
    transform_kind: TransformKind | str = TransformKind.STANDARDIZE,
    target_transform: FittedTargetTransform | None = None,
    angle_limit: float = DEFAULT_ANGLE_LIMIT,
    metadata: Mapping[str, Any] | None = None,
) -> OPFSurrogateDataset:
    """Build complex Fourier features and transformed AC-OPF targets."""

    if not isinstance(batch, ScenarioBatch):
        raise TypeError("batch must be ScenarioBatch.")
    if not isinstance(support, FrequencySupport):
        raise TypeError("support must be FrequencySupport.")

    result_tuple = tuple(results)

    if len(result_tuple) != batch.n_scenarios:
        raise OPFSurrogateError(
            f"Received {len(result_tuple)} AC-OPF results; "
            f"expected {batch.n_scenarios}."
        )
    if not result_tuple:
        raise OPFSurrogateError(
            "At least one AC-OPF result is required."
        )

    encoder = ScenarioAngleEncoder.from_batch(
        batch,
        angle_limit=angle_limit,
    )

    if support.n_dimensions != encoder.n_dimensions:
        raise OPFSurrogateError(
            f"Frequency support has {support.n_dimensions} dimensions; "
            f"scenario encoder requires {encoder.n_dimensions}."
        )

    coordinates = encoder.encode_batch(batch)
    features = toric_feature_matrix(
        coordinates,
        support,
        wrap_coordinates=False,
    )

    layout = OPFTargetLayout.from_result(
        result_tuple[0],
        fields=fields,
    )
    physical_targets = layout.extract_matrix(result_tuple)

    if target_transform is None:
        fitted_transform = build_transform(
            transform_kind,
            n_targets=layout.n_targets,
            values=physical_targets,
        )
    else:
        if not isinstance(
            target_transform,
            FittedTargetTransform,
        ):
            raise TypeError(
                "target_transform must implement FittedTargetTransform."
            )
        if target_transform.n_targets != layout.n_targets:
            raise OPFSurrogateError(
                "target_transform.n_targets does not match layout."
            )
        fitted_transform = target_transform

    transformed_targets = fitted_transform.transform(
        physical_targets
    )

    merged_metadata = {
        **({} if metadata is None else dict(metadata)),
        "surrogate_level": "opf",
        "case_name": batch.case_name,
        "scenario_batch_fingerprint": batch.fingerprint(),
        "target_fields": [field.value for field in layout.fields],
        "coordinate_names": list(encoder.coordinate_names),
        "target_names": list(layout.target_names),
        "target_transform": fitted_transform.kind.value,
    }

    dataset = CSSFDataset(
        features,
        transformed_targets,
        sample_ids=batch.scenario_ids,
        metadata=merged_metadata,
    )

    return OPFSurrogateDataset(
        dataset=dataset,
        physical_targets=physical_targets,
        coordinates=coordinates,
        encoder=encoder,
        support=support,
        target_layout=layout,
        target_transform=fitted_transform,
        scenario_ids=batch.scenario_ids,
        batch_fingerprint=batch.fingerprint(),
        metadata=merged_metadata,
    )


@runtime_checkable
class TransformedTargetPredictor(Protocol):
    """Predictor returning transformed OPF target columns."""

    def predict(self, features: ArrayLike) -> ArrayLike:
        """Return a real matrix with one column per target."""


@dataclass(frozen=True, slots=True)
class PhysicalOPFSurrogate:
    """Attach target inversion and names to a transformed-target predictor."""

    predictor: TransformedTargetPredictor
    support: FrequencySupport
    target_layout: OPFTargetLayout
    target_transform: FittedTargetTransform

    def __post_init__(self) -> None:
        if not isinstance(
            self.predictor,
            TransformedTargetPredictor,
        ):
            raise TypeError(
                "predictor must implement predict(features)."
            )
        if not isinstance(self.support, FrequencySupport):
            raise TypeError("support must be FrequencySupport.")
        if not isinstance(self.target_layout, OPFTargetLayout):
            raise TypeError(
                "target_layout must be OPFTargetLayout."
            )
        if not isinstance(
            self.target_transform,
            FittedTargetTransform,
        ):
            raise TypeError(
                "target_transform must implement FittedTargetTransform."
            )
        if (
            self.target_transform.n_targets
            != self.target_layout.n_targets
        ):
            raise OPFSurrogateError(
                "Target transform and layout sizes differ."
            )

    def predict_transformed(
        self,
        features: ArrayLike,
    ) -> NDArray[np.float64]:
        matrix = _readonly_real_matrix(
            self.predictor.predict(features),
            name="transformed_prediction",
        )

        if matrix.shape[1] != self.target_layout.n_targets:
            raise OPFSurrogateError(
                f"Predictor returned {matrix.shape[1]} targets; "
                f"expected {self.target_layout.n_targets}."
            )

        return matrix.copy()

    def predict(
        self,
        features: ArrayLike,
    ) -> NDArray[np.float64]:
        """Predict targets in physical AC-OPF units."""

        return self.target_transform.inverse_transform(
            self.predict_transformed(features)
        )

    def predict_named(
        self,
        features: ArrayLike,
    ) -> dict[str, NDArray[np.float64]]:
        prediction = self.predict(features)

        return {
            name: prediction[:, index].copy()
            for index, name in enumerate(
                self.target_layout.target_names
            )
        }


__all__ = [
    "REAL_DTYPE",
    "DEFAULT_ANGLE_LIMIT",
    "BOUND_TOLERANCE",
    "OPFSurrogateError",
    "OPFTargetField",
    "DEFAULT_TARGET_FIELDS",
    "ScenarioAngleEncoder",
    "OPFTargetLayout",
    "OPFSurrogateDataset",
    "build_opf_surrogate_dataset",
    "TransformedTargetPredictor",
    "PhysicalOPFSurrogate",
]
