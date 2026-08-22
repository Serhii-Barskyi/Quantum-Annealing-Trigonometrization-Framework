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

"""Hierarchical residual composition for the CSSF surrogate chain.

The supported additive hierarchy is

    y_OPF
    + r_QAOA
    + r_MA-QAOA
    + r_digitized-QA
    + r_hardware,

where each residual is trained against the prediction accumulated through the
preceding level. The frozen CSNN-T primitive remains unchanged; this module
only composes predictors and constructs residual-target datasets.

Quantum Walk is intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.dataset import CSSFDataset
from core.types import SurrogateLevel


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)

SURROGATE_ORDER: Final[tuple[SurrogateLevel, ...]] = (
    SurrogateLevel.OPF,
    SurrogateLevel.QAOA,
    SurrogateLevel.MA_QAOA,
    SurrogateLevel.DIGITIZED_QA,
    SurrogateLevel.HARDWARE_RESIDUAL,
)

_LEVEL_INDEX: Final[dict[SurrogateLevel, int]] = {
    level: index for index, level in enumerate(SURROGATE_ORDER)
}


class ResidualSurrogateError(ValueError):
    """Raised when a residual hierarchy violates the CSSF contract."""


@runtime_checkable
class RealPredictor(Protocol):
    """Minimal predictor interface required by the residual chain."""

    def predict(self, features: ArrayLike) -> ArrayLike:
        """Return a real matrix with shape ``(n_samples, n_targets)``."""


def _normalize_level(level: SurrogateLevel | str) -> SurrogateLevel:
    """Return one validated surrogate level."""

    if isinstance(level, SurrogateLevel):
        return level

    if not isinstance(level, str):
        raise TypeError("level must be SurrogateLevel or str.")

    try:
        return SurrogateLevel(level.strip())
    except ValueError as exc:
        raise ResidualSurrogateError(
            f"Unsupported surrogate level: {level!r}."
        ) from exc


def _real_matrix(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return a finite contiguous real matrix."""

    array = np.asarray(values)

    if np.iscomplexobj(array):
        if np.any(np.asarray(array.imag) != 0.0):
            raise ResidualSurrogateError(
                f"{name} must be real-valued."
            )
        array = array.real

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ResidualSurrogateError(
            f"{name} must be one- or two-dimensional; "
            f"received shape {array.shape}."
        )

    if 0 in array.shape:
        raise ResidualSurrogateError(f"{name} must be non-empty.")

    result = np.ascontiguousarray(array, dtype=REAL_DTYPE)

    if not np.all(np.isfinite(result)):
        raise ResidualSurrogateError(
            f"{name} contains non-finite values."
        )

    return result


def _readonly_real_matrix(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return an immutable validated real matrix."""

    result = _real_matrix(values, name=name)
    result.setflags(write=False)
    return result


def _json_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Detach and validate JSON-safe immutable metadata."""

    source = {} if metadata is None else dict(metadata)

    try:
        encoded = json.dumps(
            source,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ResidualSurrogateError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc

    return MappingProxyType(json.loads(encoded))


def residual_targets(
    reference_targets: ArrayLike,
    accumulated_prediction: ArrayLike,
) -> NDArray[np.float64]:
    """Return ``reference_targets - accumulated_prediction``."""

    reference = _real_matrix(
        reference_targets,
        name="reference_targets",
    )
    prediction = _real_matrix(
        accumulated_prediction,
        name="accumulated_prediction",
    )

    if reference.shape != prediction.shape:
        raise ResidualSurrogateError(
            "reference_targets and accumulated_prediction must have "
            f"equal shapes; received {reference.shape} and "
            f"{prediction.shape}."
        )

    return np.ascontiguousarray(
        reference - prediction,
        dtype=REAL_DTYPE,
    )


def build_residual_dataset(
    features: ArrayLike,
    reference_targets: ArrayLike,
    accumulated_prediction: ArrayLike,
    *,
    level: SurrogateLevel | str,
    baseline_level: SurrogateLevel | str,
    sample_ids: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CSSFDataset:
    """Construct a generic CSSF dataset whose targets are residuals.

    ``level`` must immediately follow ``baseline_level`` in the frozen project
    hierarchy. This prevents accidental training against the wrong baseline.
    """

    normalized_level = _normalize_level(level)
    normalized_baseline = _normalize_level(baseline_level)

    if _LEVEL_INDEX[normalized_level] != _LEVEL_INDEX[normalized_baseline] + 1:
        raise ResidualSurrogateError(
            f"{normalized_level.value} must immediately follow "
            f"{normalized_baseline.value}."
        )

    targets = residual_targets(
        reference_targets,
        accumulated_prediction,
    )

    merged_metadata = {
        **({} if metadata is None else dict(metadata)),
        "target_semantics": "additive_residual",
        "surrogate_level": normalized_level.value,
        "baseline_level": normalized_baseline.value,
    }

    return CSSFDataset(
        features,
        targets,
        sample_ids=sample_ids,
        metadata=merged_metadata,
    )


@dataclass(frozen=True, slots=True)
class ResidualComponent:
    """One predictor in the ordered additive hierarchy."""

    level: SurrogateLevel
    predictor: RealPredictor
    weight: float = 1.0
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.level, SurrogateLevel):
            raise TypeError("level must be SurrogateLevel.")
        if not isinstance(self.predictor, RealPredictor):
            raise TypeError(
                "predictor must implement predict(features)."
            )

        weight = float(self.weight)
        if not math.isfinite(weight):
            raise ResidualSurrogateError(
                "component weight must be finite."
            )
        if weight <= 0.0:
            raise ResidualSurrogateError(
                "component weight must be strictly positive."
            )

        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class ResidualPrediction:
    """Immutable decomposition of one hierarchical prediction."""

    total: NDArray[np.float64]
    contributions: Mapping[SurrogateLevel, NDArray[np.float64]]
    through_level: SurrogateLevel

    def __post_init__(self) -> None:
        total = _readonly_real_matrix(self.total, name="total")

        normalized: dict[SurrogateLevel, NDArray[np.float64]] = {}
        for level, values in self.contributions.items():
            if not isinstance(level, SurrogateLevel):
                raise TypeError(
                    "Contribution keys must be SurrogateLevel."
                )
            contribution = _readonly_real_matrix(
                values,
                name=f"contribution[{level.value}]",
            )
            if contribution.shape != total.shape:
                raise ResidualSurrogateError(
                    f"Contribution {level.value} has shape "
                    f"{contribution.shape}; expected {total.shape}."
                )
            normalized[level] = contribution

        if self.through_level not in normalized:
            raise ResidualSurrogateError(
                "through_level must be present in contributions."
            )

        object.__setattr__(self, "total", total)
        object.__setattr__(
            self,
            "contributions",
            MappingProxyType(normalized),
        )


@dataclass(frozen=True, slots=True, init=False)
class ResidualSurrogateChain:
    """Strict ordered additive composition of CSSF predictors."""

    components: tuple[ResidualComponent, ...]
    target_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        components: Sequence[ResidualComponent],
        *,
        target_names: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_components = tuple(components)

        if not normalized_components:
            raise ResidualSurrogateError(
                "At least the OPF base component is required."
            )

        expected_levels = SURROGATE_ORDER[: len(normalized_components)]
        actual_levels = tuple(
            component.level for component in normalized_components
        )

        if actual_levels != expected_levels:
            raise ResidualSurrogateError(
                "Components must form a contiguous hierarchy beginning with "
                f"OPF. Expected {[level.value for level in expected_levels]}, "
                f"received {[level.value for level in actual_levels]}."
            )

        names = tuple(str(value).strip() for value in target_names)
        if not names:
            raise ResidualSurrogateError(
                "target_names must not be empty."
            )
        if any(not name for name in names):
            raise ResidualSurrogateError(
                "target_names must not contain empty values."
            )
        if len(set(names)) != len(names):
            raise ResidualSurrogateError(
                "target_names must be unique."
            )

        object.__setattr__(
            self,
            "components",
            normalized_components,
        )
        object.__setattr__(self, "target_names", names)
        object.__setattr__(self, "metadata", _json_metadata(metadata))

    @property
    def n_targets(self) -> int:
        """Expected number of physical target columns."""

        return len(self.target_names)

    @property
    def highest_level(self) -> SurrogateLevel:
        """Highest level currently present in the chain."""

        return self.components[-1].level

    @property
    def levels(self) -> tuple[SurrogateLevel, ...]:
        """Ordered levels present in the chain."""

        return tuple(component.level for component in self.components)

    def _normalize_feature_mapping(
        self,
        features_by_level: Mapping[
            SurrogateLevel | str,
            ArrayLike,
        ],
    ) -> dict[SurrogateLevel, ArrayLike]:
        if not isinstance(features_by_level, Mapping):
            raise TypeError("features_by_level must be a mapping.")

        normalized: dict[SurrogateLevel, ArrayLike] = {}

        for level, features in features_by_level.items():
            normalized_level = _normalize_level(level)
            if normalized_level in normalized:
                raise ResidualSurrogateError(
                    f"Duplicate feature mapping for "
                    f"{normalized_level.value}."
                )
            normalized[normalized_level] = features

        return normalized

    def decompose(
        self,
        features_by_level: Mapping[
            SurrogateLevel | str,
            ArrayLike,
        ],
        *,
        through_level: SurrogateLevel | str | None = None,
    ) -> ResidualPrediction:
        """Evaluate and return every additive contribution."""

        features = self._normalize_feature_mapping(features_by_level)
        requested_level = (
            self.highest_level
            if through_level is None
            else _normalize_level(through_level)
        )

        if requested_level not in self.levels:
            raise ResidualSurrogateError(
                f"through_level={requested_level.value!r} is not present "
                "in this chain."
            )

        stop_index = self.levels.index(requested_level)
        contributions: dict[
            SurrogateLevel,
            NDArray[np.float64],
        ] = {}
        expected_shape: tuple[int, int] | None = None

        for component in self.components[: stop_index + 1]:
            if component.level not in features:
                raise ResidualSurrogateError(
                    f"Missing features for level "
                    f"{component.level.value!r}."
                )

            prediction = _real_matrix(
                component.predictor.predict(
                    features[component.level]
                ),
                name=f"prediction[{component.level.value}]",
            )

            if prediction.shape[1] != self.n_targets:
                raise ResidualSurrogateError(
                    f"Predictor {component.level.value} returned "
                    f"{prediction.shape[1]} targets; expected "
                    f"{self.n_targets}."
                )

            if expected_shape is None:
                expected_shape = prediction.shape
            elif prediction.shape != expected_shape:
                raise ResidualSurrogateError(
                    f"Predictor {component.level.value} returned shape "
                    f"{prediction.shape}; expected {expected_shape}."
                )

            weighted = np.ascontiguousarray(
                component.weight * prediction,
                dtype=REAL_DTYPE,
            )
            contributions[component.level] = weighted

        assert expected_shape is not None

        total = np.zeros(expected_shape, dtype=REAL_DTYPE)
        for contribution in contributions.values():
            total += contribution

        return ResidualPrediction(
            total=total,
            contributions=contributions,
            through_level=requested_level,
        )

    def predict(
        self,
        features_by_level: Mapping[
            SurrogateLevel | str,
            ArrayLike,
        ],
        *,
        through_level: SurrogateLevel | str | None = None,
    ) -> NDArray[np.float64]:
        """Return the accumulated physical prediction."""

        return self.decompose(
            features_by_level,
            through_level=through_level,
        ).total.copy()


__all__ = [
    "REAL_DTYPE",
    "SURROGATE_ORDER",
    "ResidualSurrogateError",
    "RealPredictor",
    "residual_targets",
    "build_residual_dataset",
    "ResidualComponent",
    "ResidualPrediction",
    "ResidualSurrogateChain",
]
