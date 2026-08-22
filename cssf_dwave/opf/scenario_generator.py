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

"""Deterministic load and renewable scenarios for OPF experiments.

Scenario factors are generated once, stored in immutable arrays, and applied
only to independent copies of a validated base network. Active and reactive
power of each element use the same multiplier, preserving the base power
factor. This module never runs a power-flow or OPF solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from core.randomness import derive_seed, make_generator, validate_seed
from opf.case_loader import (
    LoadedPowerCase,
    power_case_fingerprint,
    validate_power_case,
)


REAL_DTYPE = np.dtype(np.float64)


class ScenarioGenerationError(ValueError):
    """Raised when a scenario-generation contract is violated."""


def _validate_bounds(
    bounds: Sequence[float],
    *,
    name: str,
    allow_zero_lower: bool,
) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ScenarioGenerationError(
            f"{name} must contain exactly two values."
        )

    lower = float(bounds[0])
    upper = float(bounds[1])

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ScenarioGenerationError(f"{name} values must be finite.")
    if lower >= upper:
        raise ScenarioGenerationError(f"{name} must satisfy lower < upper.")
    if allow_zero_lower:
        if lower < 0.0:
            raise ScenarioGenerationError(
                f"{name} lower bound must be non-negative."
            )
    elif lower <= 0.0:
        raise ScenarioGenerationError(
            f"{name} lower bound must be strictly positive."
        )

    return lower, upper


def _nonnegative_finite(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ScenarioGenerationError(f"{name} must be finite.")
    if normalized < 0.0:
        raise ScenarioGenerationError(f"{name} must be non-negative.")
    return normalized


def _table(network: Any, table_name: str) -> Any:
    if hasattr(network, table_name):
        return getattr(network, table_name)
    if isinstance(network, Mapping):
        return network.get(table_name)
    try:
        return network[table_name]
    except (KeyError, TypeError, AttributeError):
        return None


def _readonly_matrix(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.ascontiguousarray(values, dtype=REAL_DTYPE)
    result.setflags(write=False)
    return result


def _json_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    source = {} if metadata is None else dict(metadata)
    try:
        encoded = json.dumps(
            source,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ScenarioGenerationError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    return MappingProxyType(json.loads(encoded))


@dataclass(frozen=True, slots=True)
class ScenarioGeneratorConfig:
    """Parameters of the correlated multiplicative scenario model."""

    n_scenarios: int = 256
    seed: int = 101
    load_mean: float = 1.0
    load_sigma: float = 0.08
    renewable_mean: float = 1.0
    renewable_sigma: float = 0.20
    intra_family_correlation: float = 0.35
    load_bounds: tuple[float, float] = (0.75, 1.25)
    renewable_bounds: tuple[float, float] = (0.0, 2.0)

    def __post_init__(self) -> None:
        if isinstance(self.n_scenarios, bool) or not isinstance(
            self.n_scenarios, int
        ):
            raise TypeError("n_scenarios must be an integer.")
        if self.n_scenarios < 1:
            raise ScenarioGenerationError("n_scenarios must be positive.")

        validate_seed(self.seed)

        load_mean = float(self.load_mean)
        renewable_mean = float(self.renewable_mean)
        if not math.isfinite(load_mean) or load_mean <= 0.0:
            raise ScenarioGenerationError(
                "load_mean must be finite and positive."
            )
        if not math.isfinite(renewable_mean) or renewable_mean < 0.0:
            raise ScenarioGenerationError(
                "renewable_mean must be finite and non-negative."
            )

        load_sigma = _nonnegative_finite(
            self.load_sigma, name="load_sigma"
        )
        renewable_sigma = _nonnegative_finite(
            self.renewable_sigma, name="renewable_sigma"
        )

        correlation = float(self.intra_family_correlation)
        if not math.isfinite(correlation) or not 0.0 <= correlation <= 1.0:
            raise ScenarioGenerationError(
                "intra_family_correlation must lie in [0, 1]."
            )

        object.__setattr__(self, "load_mean", load_mean)
        object.__setattr__(self, "renewable_mean", renewable_mean)
        object.__setattr__(self, "load_sigma", load_sigma)
        object.__setattr__(self, "renewable_sigma", renewable_sigma)
        object.__setattr__(
            self, "intra_family_correlation", correlation
        )
        object.__setattr__(
            self,
            "load_bounds",
            _validate_bounds(
                self.load_bounds,
                name="load_bounds",
                allow_zero_lower=False,
            ),
        )
        object.__setattr__(
            self,
            "renewable_bounds",
            _validate_bounds(
                self.renewable_bounds,
                name="renewable_bounds",
                allow_zero_lower=True,
            ),
        )

    def as_dict(self) -> dict[str, int | float | list[float]]:
        return {
            "n_scenarios": self.n_scenarios,
            "seed": self.seed,
            "load_mean": self.load_mean,
            "load_sigma": self.load_sigma,
            "renewable_mean": self.renewable_mean,
            "renewable_sigma": self.renewable_sigma,
            "intra_family_correlation": self.intra_family_correlation,
            "load_bounds": list(self.load_bounds),
            "renewable_bounds": list(self.renewable_bounds),
        }


def _correlated_lognormal_factors(
    *,
    n_scenarios: int,
    n_elements: int,
    mean: float,
    sigma: float,
    correlation: float,
    bounds: tuple[float, float],
    seed: int,
) -> NDArray[np.float64]:
    if n_elements == 0:
        return np.empty((n_scenarios, 0), dtype=REAL_DTYPE)

    generator = make_generator(seed)

    if sigma == 0.0:
        factors = np.full(
            (n_scenarios, n_elements), mean, dtype=REAL_DTYPE
        )
    else:
        shared = generator.normal(size=(n_scenarios, 1))
        local = generator.normal(size=(n_scenarios, n_elements))
        shocks = (
            math.sqrt(correlation) * shared
            + math.sqrt(1.0 - correlation) * local
        )
        factors = mean * np.exp(
            sigma * shocks - 0.5 * sigma * sigma
        )

    factors = np.clip(factors, bounds[0], bounds[1])
    if not np.all(np.isfinite(factors)):
        raise ScenarioGenerationError(
            "Generated factors contain non-finite values."
        )
    return np.ascontiguousarray(factors, dtype=REAL_DTYPE)


@dataclass(frozen=True, slots=True, init=False)
class ScenarioBatch:
    """Immutable factors tied to one exact base-network fingerprint."""

    case_name: str
    base_fingerprint: str
    scenario_ids: tuple[str, ...]
    load_indices: tuple[Any, ...]
    renewable_indices: tuple[Any, ...]
    load_factors: NDArray[np.float64]
    renewable_factors: NDArray[np.float64]
    config: ScenarioGeneratorConfig
    metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        case_name: str,
        base_fingerprint: str,
        scenario_ids: Sequence[str],
        load_indices: Sequence[Any],
        renewable_indices: Sequence[Any],
        load_factors: NDArray[np.float64],
        renewable_factors: NDArray[np.float64],
        config: ScenarioGeneratorConfig,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_case = str(case_name).strip().lower()
        if not normalized_case:
            raise ScenarioGenerationError("case_name must not be empty.")
        if len(base_fingerprint) != 64:
            raise ScenarioGenerationError(
                "base_fingerprint must be a SHA-256 digest."
            )
        if not isinstance(config, ScenarioGeneratorConfig):
            raise TypeError("config must be ScenarioGeneratorConfig.")

        ids = tuple(str(value).strip() for value in scenario_ids)
        if len(ids) != config.n_scenarios:
            raise ScenarioGenerationError(
                "scenario_ids length must equal n_scenarios."
            )
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ScenarioGenerationError(
                "scenario_ids must be non-empty and unique."
            )

        load_index_tuple = tuple(load_indices)
        renewable_index_tuple = tuple(renewable_indices)
        loads = _readonly_matrix(load_factors)
        renewables = _readonly_matrix(renewable_factors)

        if loads.shape != (
            config.n_scenarios,
            len(load_index_tuple),
        ):
            raise ScenarioGenerationError(
                "load_factors has an invalid shape."
            )
        if renewables.shape != (
            config.n_scenarios,
            len(renewable_index_tuple),
        ):
            raise ScenarioGenerationError(
                "renewable_factors has an invalid shape."
            )

        object.__setattr__(self, "case_name", normalized_case)
        object.__setattr__(self, "base_fingerprint", base_fingerprint)
        object.__setattr__(self, "scenario_ids", ids)
        object.__setattr__(self, "load_indices", load_index_tuple)
        object.__setattr__(
            self, "renewable_indices", renewable_index_tuple
        )
        object.__setattr__(self, "load_factors", loads)
        object.__setattr__(self, "renewable_factors", renewables)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "metadata", _json_metadata(metadata))

    @property
    def n_scenarios(self) -> int:
        return self.config.n_scenarios

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-ScenarioBatch-v1\0")
        digest.update(self.case_name.encode("utf-8"))
        digest.update(self.base_fingerprint.encode("ascii"))
        digest.update(
            json.dumps(
                self.scenario_ids,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(
            json.dumps(
                list(self.load_indices),
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(
            json.dumps(
                list(self.renewable_indices),
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(self.load_factors.tobytes(order="C"))
        digest.update(self.renewable_factors.tobytes(order="C"))
        digest.update(
            json.dumps(
                self.config.as_dict(),
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

    def _validate_base_case(self, loaded_case: LoadedPowerCase) -> None:
        if not isinstance(loaded_case, LoadedPowerCase):
            raise TypeError("loaded_case must be LoadedPowerCase.")
        if loaded_case.name != self.case_name:
            raise ScenarioGenerationError(
                f"Scenario batch belongs to {self.case_name!r}, "
                f"not {loaded_case.name!r}."
            )
        if power_case_fingerprint(loaded_case.network) != self.base_fingerprint:
            raise ScenarioGenerationError(
                "Base-network fingerprint mismatch."
            )

    def apply(
        self,
        loaded_case: LoadedPowerCase,
        scenario_index: int,
    ) -> Any:
        """Return an independent network with one scenario applied."""

        self._validate_base_case(loaded_case)

        if isinstance(scenario_index, bool) or not isinstance(
            scenario_index, int
        ):
            raise TypeError("scenario_index must be an integer.")
        if not 0 <= scenario_index < self.n_scenarios:
            raise ScenarioGenerationError(
                "scenario_index is outside batch bounds."
            )

        network = loaded_case.clone_network()
        load = _table(network, "load")

        if self.load_indices:
            if load is None:
                raise ScenarioGenerationError(
                    "Base network has no load table."
                )
            for column_name in ("p_mw", "q_mvar"):
                if column_name in load.columns:
                    base_values = load.loc[
                        list(self.load_indices), column_name
                    ].to_numpy(dtype=np.float64)
                    load.loc[
                        list(self.load_indices), column_name
                    ] = base_values * self.load_factors[scenario_index]

        renewable = _table(network, "sgen")
        if self.renewable_indices:
            if renewable is None:
                raise ScenarioGenerationError(
                    "Base network has no sgen table."
                )
            for column_name in ("p_mw", "q_mvar"):
                if column_name in renewable.columns:
                    base_values = renewable.loc[
                        list(self.renewable_indices), column_name
                    ].to_numpy(dtype=np.float64)
                    renewable.loc[
                        list(self.renewable_indices), column_name
                    ] = (
                        base_values
                        * self.renewable_factors[scenario_index]
                    )

        validate_power_case(network)
        return network

    def iter_networks(
        self,
        loaded_case: LoadedPowerCase,
    ) -> Iterator[tuple[str, Any]]:
        for index, scenario_id in enumerate(self.scenario_ids):
            yield scenario_id, self.apply(loaded_case, index)


def generate_scenarios(
    loaded_case: LoadedPowerCase,
    config: ScenarioGeneratorConfig,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ScenarioBatch:
    """Generate a deterministic batch for one exact base network."""

    if not isinstance(loaded_case, LoadedPowerCase):
        raise TypeError("loaded_case must be LoadedPowerCase.")
    if not isinstance(config, ScenarioGeneratorConfig):
        raise TypeError("config must be ScenarioGeneratorConfig.")

    validate_power_case(loaded_case.network)
    base_fingerprint = power_case_fingerprint(loaded_case.network)

    if loaded_case.summary.fingerprint != base_fingerprint:
        raise ScenarioGenerationError(
            "LoadedPowerCase summary fingerprint is stale."
        )

    load = _table(loaded_case.network, "load")
    renewable = _table(loaded_case.network, "sgen")

    load_indices = tuple() if load is None else tuple(load.index.tolist())
    renewable_indices = (
        tuple()
        if renewable is None
        else tuple(renewable.index.tolist())
    )

    load_factors = _correlated_lognormal_factors(
        n_scenarios=config.n_scenarios,
        n_elements=len(load_indices),
        mean=config.load_mean,
        sigma=config.load_sigma,
        correlation=config.intra_family_correlation,
        bounds=config.load_bounds,
        seed=derive_seed(config.seed, "opf_load_scenarios"),
    )
    renewable_factors = _correlated_lognormal_factors(
        n_scenarios=config.n_scenarios,
        n_elements=len(renewable_indices),
        mean=config.renewable_mean,
        sigma=config.renewable_sigma,
        correlation=config.intra_family_correlation,
        bounds=config.renewable_bounds,
        seed=derive_seed(config.seed, "opf_renewable_scenarios"),
    )

    scenario_ids = tuple(
        f"scenario_{index:06d}"
        for index in range(config.n_scenarios)
    )

    return ScenarioBatch(
        case_name=loaded_case.name,
        base_fingerprint=base_fingerprint,
        scenario_ids=scenario_ids,
        load_indices=load_indices,
        renewable_indices=renewable_indices,
        load_factors=load_factors,
        renewable_factors=renewable_factors,
        config=config,
        metadata={
            **({} if metadata is None else dict(metadata)),
            "case_name": loaded_case.name,
            "base_fingerprint": base_fingerprint,
        },
    )


__all__ = [
    "REAL_DTYPE",
    "ScenarioGenerationError",
    "ScenarioGeneratorConfig",
    "ScenarioBatch",
    "generate_scenarios",
]
