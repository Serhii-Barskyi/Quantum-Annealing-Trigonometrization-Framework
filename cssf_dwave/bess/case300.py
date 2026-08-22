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

"""Strict immutable loader for the fixed CSSF case300 Mode-A dataset.

The dataset is a release input, not a mutable runtime cache.  This module
therefore verifies the exact project path, byte size, SHA-256 digest, JSON
schema, numerical shapes, finite values, Mode-A conjugate structure, scenario
partition, slack-bus convention, and immutable array ownership before exposing
any data to candidate selection or later BESS stages.

No OPF, QUBO, quantum, Ocean, D-Wave, or HiGHS runtime is imported here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from bess import (
    CASE300_BUS_COUNT,
    CASE300_CASE_NAME,
    CASE300_DATASET_FILENAME,
    REQUIRED_CASE300_KEYS,
    validate_scenario_partition,
)
from project_paths import DATA_DIR


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
COMPLEX_DTYPE: Final[np.dtype[np.complex128]] = np.dtype(np.complex128)
INTEGER_DTYPE: Final[np.dtype[np.int64]] = np.dtype(np.int64)

CASE300_DATASET_SHA256: Final[str] = (
    "2764826242d9f9ecfdc6e8f0d78a8cac452c39b9091ed84a74f1771e6df52539"
)
CASE300_DATASET_SIZE_BYTES: Final[int] = 93_897_330
CASE300_DATASET_PATH: Final[Path] = DATA_DIR / CASE300_DATASET_FILENAME

CASE300_SCENARIO_LABELS: Final[tuple[str, ...]] = (
    "low",
    "n1",
    "normal",
    "peak",
    "rei",
)
CASE300_BUS_TYPES: Final[tuple[int, ...]] = (1, 2, 3)
CASE300_MODE_A_PAIR_TOLERANCE: Final[float] = 1.0e-12
CASE300_UNIT_MODULUS_TOLERANCE: Final[float] = 1.0e-12
CASE300_EXACT_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    (*REQUIRED_CASE300_KEYS, "meta", "params")
)


class Case300DataError(ValueError):
    """Raised when the fixed case300 dataset violates its release contract."""


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Case300DataError(f"JSON object contains duplicate key {key!r}.")
        result[key] = value
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Case300DataError(f"{name} must be an integer.")
    if value < minimum:
        raise Case300DataError(f"{name} must be >= {minimum}; received {value}.")
    return value




def _bus_tuple(values: object, *, name: str, n_buses: int) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise Case300DataError(f"{name} must be a sequence of bus IDs.")
    if len(values) == 0:
        raise Case300DataError(f"{name} must not be empty.")
    result: list[int] = []
    seen: set[int] = set()
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise Case300DataError(
                f"{name}[{position}] must be an integer bus ID."
            )
        if not 0 <= value < n_buses:
            raise Case300DataError(
                f"{name}[{position}] is outside [0, {n_buses - 1}]."
            )
        if value in seen:
            raise Case300DataError(f"{name} contains duplicate bus {value}.")
        seen.add(value)
        result.append(value)
    return tuple(result)

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly_real_matrix(
    values: object,
    *,
    name: str,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    try:
        result = np.array(values, dtype=REAL_DTYPE, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Case300DataError(f"{name} must be a real numeric matrix.") from exc
    if result.shape != shape:
        raise Case300DataError(
            f"{name} must have shape {shape}; received {result.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise Case300DataError(f"{name} contains NaN or infinite values.")
    result.setflags(write=False)
    return result


def _readonly_integer_vector(
    values: object,
    *,
    name: str,
    size: int,
) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.shape != (size,):
        raise Case300DataError(
            f"{name} must have shape {(size,)}; received {raw.shape}."
        )
    if np.issubdtype(raw.dtype, np.bool_):
        raise Case300DataError(f"{name} must not contain booleans.")
    try:
        numeric = np.asarray(raw, dtype=REAL_DTYPE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Case300DataError(f"{name} must contain integers.") from exc
    if not np.all(np.isfinite(numeric)):
        raise Case300DataError(f"{name} contains NaN or infinite values.")
    if not np.array_equal(numeric, np.rint(numeric)):
        raise Case300DataError(f"{name} contains non-integral values.")
    result = np.array(numeric, dtype=INTEGER_DTYPE, order="C", copy=True)
    result.setflags(write=False)
    return result


def _freeze_json(value: object, *, path: str = "params") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Case300DataError(f"{path} contains NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise Case300DataError(f"{path} keys must be non-empty strings.")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise Case300DataError(
        f"{path} contains unsupported value type {type(value).__name__}."
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_hex_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Case300DataError(f"{name} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise Case300DataError(f"{name} is not hexadecimal.") from exc
    return value.lower()


def _hash_array(digest: Any, name: str, array: NDArray[Any]) -> None:
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))


def _dataset_fingerprint(
    *,
    source_sha256: str,
    n: int,
    n_train: int,
    n_test: int,
    n_scenarios: int,
    m: int,
    m_complex: int,
    rank_mode_a: int,
    delta_r: int,
    bus_types: NDArray[np.int64],
    edges: NDArray[np.float64],
    theta_rad: NDArray[np.float64],
    targets: NDArray[np.float64],
    features_re: NDArray[np.float64],
    features_im: NDArray[np.float64],
    labels: tuple[str, ...],
    params: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"CSSF-Case300ModeAData-v1\0")
    digest.update(source_sha256.encode("ascii"))
    digest.update(b"\0")
    scalars = {
        "case": CASE300_CASE_NAME,
        "n": n,
        "n_train": n_train,
        "n_test": n_test,
        "n_scenarios": n_scenarios,
        "M": m,
        "M_complex": m_complex,
        "rank_modeA": rank_mode_a,
        "delta_r": delta_r,
    }
    digest.update(
        json.dumps(scalars, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, array in (
        ("bus_types", bus_types),
        ("edges", edges),
        ("theta_rad", theta_rad),
        ("y_lsf", targets),
        ("X_modeA_re", features_re),
        ("X_modeA_im", features_im),
    ):
        _hash_array(digest, name, array)
    digest.update(
        json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    digest.update(
        json.dumps(
            _thaw_json(params),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _validate_array_ownership(name: str, array: NDArray[Any]) -> None:
    if array.flags.writeable:
        raise Case300DataError(f"{name} must be immutable.")
    if not array.flags.c_contiguous:
        raise Case300DataError(f"{name} must be C-contiguous.")
    if not array.flags.owndata:
        raise Case300DataError(f"{name} must own its memory.")


@dataclass(frozen=True, slots=True)
class Case300ModeAData:
    """Validated immutable case300 Mode-A arrays and release provenance."""

    case: str
    n: int
    n_train: int
    n_test: int
    n_scenarios: int
    bus_types: NDArray[np.int64]
    edges: NDArray[np.float64]
    M: int
    M_complex: int
    rank_modeA: int
    delta_r: int
    theta_rad: NDArray[np.float64]
    targets: NDArray[np.float64]
    features_re: NDArray[np.float64]
    features_im: NDArray[np.float64]
    features: NDArray[np.complex128]
    scenario_labels: tuple[str, ...]
    params: Mapping[str, Any]
    slack_buses: tuple[int, ...]
    source_path: str
    source_sha256: str
    _fingerprint_value: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.case != CASE300_CASE_NAME or self.n != CASE300_BUS_COUNT:
            raise Case300DataError("Case300 identity is inconsistent.")
        validate_scenario_partition(
            n_scenarios=self.n_scenarios,
            n_train=self.n_train,
            n_test=self.n_test,
        )
        expected_shapes = {
            "bus_types": (self.n,),
            "edges": (self.M, 3),
            "theta_rad": (self.n_scenarios, self.n),
            "targets": (self.n_scenarios, self.n),
            "features_re": (self.n_scenarios, self.M_complex),
            "features_im": (self.n_scenarios, self.M_complex),
            "features": (self.n_scenarios, self.M_complex),
        }
        for name, shape in expected_shapes.items():
            array = getattr(self, name)
            if array.shape != shape:
                raise Case300DataError(
                    f"{name} must have shape {shape}; received {array.shape}."
                )
            _validate_array_ownership(name, array)
        if self.M_complex != 2 * self.M:
            raise Case300DataError("M_complex must equal 2 * M.")
        if len(self.scenario_labels) != self.n_scenarios:
            raise Case300DataError("scenario_labels length mismatch.")
        _bus_tuple(self.slack_buses, name="slack_buses", n_buses=self.n)
        _validate_hex_digest(self.source_sha256, name="source_sha256")
        _validate_hex_digest(self._fingerprint_value, name="fingerprint")
        if not isinstance(self.params, MappingProxyType):
            raise Case300DataError("params must be an immutable mapping proxy.")

    @property
    def train_slice(self) -> slice:
        return slice(0, self.n_train)

    @property
    def test_slice(self) -> slice:
        return slice(self.n_train, self.n_scenarios)

    @property
    def y_lsf(self) -> NDArray[np.float64]:
        return self.targets

    @property
    def X_modeA_re(self) -> NDArray[np.float64]:
        return self.features_re

    @property
    def X_modeA_im(self) -> NDArray[np.float64]:
        return self.features_im

    @property
    def meta(self) -> tuple[str, ...]:
        return self.scenario_labels

    def fingerprint(self) -> str:
        return self._fingerprint_value


def _validated_dataset_path(path: str | Path | None) -> Path:
    expected = CASE300_DATASET_PATH.resolve(strict=False)
    candidate = expected if path is None else Path(path).expanduser().resolve(strict=False)
    if candidate != expected:
        raise Case300DataError(
            "case300 must be loaded only from the fixed project path "
            f"{expected}; received {candidate}."
        )
    if not candidate.exists():
        raise FileNotFoundError(f"case300 dataset does not exist: {candidate}")
    if not candidate.is_file():
        raise Case300DataError(f"case300 dataset path is not a file: {candidate}")
    return candidate


def verify_case300_dataset(path: str | Path | None = None) -> str:
    """Verify fixed path, exact byte size, and canonical SHA-256."""

    candidate = _validated_dataset_path(path)
    size = candidate.stat().st_size
    if size != CASE300_DATASET_SIZE_BYTES:
        raise Case300DataError(
            "case300 dataset byte-size mismatch: expected "
            f"{CASE300_DATASET_SIZE_BYTES}, received {size}."
        )
    digest = _sha256_file(candidate)
    if digest != CASE300_DATASET_SHA256:
        raise Case300DataError(
            "case300 dataset SHA-256 mismatch: expected "
            f"{CASE300_DATASET_SHA256}, received {digest}."
        )
    return digest


def load_case300_mode_a(path: str | Path | None = None) -> Case300ModeAData:
    """Load and fully validate the canonical case300 Mode-A release dataset."""

    candidate = _validated_dataset_path(path)
    source_sha256 = verify_case300_dataset(candidate)

    try:
        with candidate.open("r", encoding="utf-8", newline="") as stream:
            payload = json.load(stream, object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Case300DataError("case300 dataset is not valid strict UTF-8 JSON.") from exc

    if not isinstance(payload, dict):
        raise Case300DataError("case300 top-level JSON value must be an object.")
    actual_keys = frozenset(payload)
    if actual_keys != CASE300_EXACT_TOP_LEVEL_KEYS:
        missing = sorted(CASE300_EXACT_TOP_LEVEL_KEYS - actual_keys)
        extra = sorted(actual_keys - CASE300_EXACT_TOP_LEVEL_KEYS)
        raise Case300DataError(
            f"case300 top-level key mismatch; missing={missing}, extra={extra}."
        )

    case = payload.pop("case")
    if case != CASE300_CASE_NAME:
        raise Case300DataError(
            f"case must be {CASE300_CASE_NAME!r}; received {case!r}."
        )
    n = _integer(payload.pop("n"), name="n", minimum=2)
    if n != CASE300_BUS_COUNT:
        raise Case300DataError(
            f"n must equal {CASE300_BUS_COUNT}; received {n}."
        )
    n_train = _integer(payload.pop("n_train"), name="n_train", minimum=1)
    n_test = _integer(payload.pop("n_test"), name="n_test", minimum=1)
    n_scenarios = _integer(
        payload.pop("n_scenarios"), name="n_scenarios", minimum=1
    )
    validate_scenario_partition(
        n_scenarios=n_scenarios,
        n_train=n_train,
        n_test=n_test,
    )

    m = _integer(payload.pop("M"), name="M", minimum=1)
    m_complex = _integer(payload.pop("M_complex"), name="M_complex", minimum=2)
    rank_mode_a = _integer(
        payload.pop("rank_modeA"), name="rank_modeA", minimum=1
    )
    delta_r = _integer(payload.pop("delta_r"), name="delta_r", minimum=1)
    if m_complex != 2 * m:
        raise Case300DataError(
            f"M_complex must equal 2 * M; received {m_complex} != {2 * m}."
        )
    if rank_mode_a > min(n_scenarios, m_complex):
        raise Case300DataError("rank_modeA exceeds the matrix rank upper bound.")

    bus_types = _readonly_integer_vector(
        payload.pop("bus_types"), name="bus_types", size=n
    )
    observed_bus_types = tuple(int(value) for value in np.unique(bus_types))
    if observed_bus_types != CASE300_BUS_TYPES:
        raise Case300DataError(
            f"bus_types must contain exactly {CASE300_BUS_TYPES}; "
            f"received {observed_bus_types}."
        )

    edges = _readonly_real_matrix(
        payload.pop("edges"), name="edges", shape=(m, 3)
    )
    endpoints = edges[:, :2]
    if not np.array_equal(endpoints, np.rint(endpoints)):
        raise Case300DataError("edges endpoint columns must be integral bus IDs.")
    integer_endpoints = np.asarray(endpoints, dtype=INTEGER_DTYPE)
    if np.any(integer_endpoints < 0) or np.any(integer_endpoints >= n):
        raise Case300DataError("edges reference a bus outside [0, n - 1].")
    if np.any(integer_endpoints[:, 0] == integer_endpoints[:, 1]):
        raise Case300DataError("edges must not contain self-loops.")
    canonical_pairs = np.sort(integer_endpoints, axis=1)
    if np.unique(canonical_pairs, axis=0).shape[0] != m:
        raise Case300DataError("edges contain duplicate undirected bus pairs.")

    theta_rad = _readonly_real_matrix(
        payload.pop("theta_rad"),
        name="theta_rad",
        shape=(n_scenarios, n),
    )
    targets = _readonly_real_matrix(
        payload.pop("y_lsf"),
        name="y_lsf",
        shape=(n_scenarios, n),
    )
    features_re = _readonly_real_matrix(
        payload.pop("X_modeA_re"),
        name="X_modeA_re",
        shape=(n_scenarios, m_complex),
    )
    features_im = _readonly_real_matrix(
        payload.pop("X_modeA_im"),
        name="X_modeA_im",
        shape=(n_scenarios, m_complex),
    )
    features = np.array(
        features_re + 1j * features_im,
        dtype=COMPLEX_DTYPE,
        order="C",
        copy=True,
    )
    if not np.allclose(
        features[:, 1::2],
        np.conjugate(features[:, 0::2]),
        rtol=0.0,
        atol=CASE300_MODE_A_PAIR_TOLERANCE,
    ):
        raise Case300DataError(
            "Mode-A feature columns must form exact interleaved conjugate pairs."
        )
    if not np.allclose(
        np.abs(features),
        1.0,
        rtol=0.0,
        atol=CASE300_UNIT_MODULUS_TOLERANCE,
    ):
        raise Case300DataError("Mode-A complex features must have unit modulus.")
    features.setflags(write=False)

    raw_labels = payload.pop("meta")
    if not isinstance(raw_labels, list) or len(raw_labels) != n_scenarios:
        raise Case300DataError(
            f"meta must be a list of {n_scenarios} scenario labels."
        )
    if any(not isinstance(label, str) or not label for label in raw_labels):
        raise Case300DataError("meta labels must be non-empty strings.")
    labels = tuple(raw_labels)
    counts = Counter(labels)
    if tuple(sorted(counts)) != tuple(sorted(CASE300_SCENARIO_LABELS)):
        raise Case300DataError(
            f"meta labels must be exactly {CASE300_SCENARIO_LABELS}."
        )

    raw_params = payload.pop("params")
    if not isinstance(raw_params, dict):
        raise Case300DataError("params must be a JSON object.")
    params = _freeze_json(raw_params)
    if not isinstance(params, MappingProxyType):
        raise Case300DataError("params normalization failed.")

    n_per_type = params.get("N_PER_TYPE")
    if isinstance(n_per_type, bool) or not isinstance(n_per_type, int):
        raise Case300DataError("params.N_PER_TYPE must be an integer.")
    if n_per_type * len(CASE300_SCENARIO_LABELS) != n_scenarios:
        raise Case300DataError("params.N_PER_TYPE is inconsistent with n_scenarios.")
    if any(counts[label] != n_per_type for label in CASE300_SCENARIO_LABELS):
        raise Case300DataError("Scenario label counts violate params.N_PER_TYPE.")

    raw_slack = params.get("slack_buses")
    if not isinstance(raw_slack, tuple):
        raise Case300DataError("params.slack_buses must be a JSON array.")
    slack_buses = _bus_tuple(raw_slack, name="params.slack_buses", n_buses=n)
    type_three_buses = tuple(int(index) for index in np.flatnonzero(bus_types == 3))
    if slack_buses != type_three_buses:
        raise Case300DataError(
            "params.slack_buses must equal the bus_type=3 indices."
        )

    if payload:
        raise Case300DataError(
            f"Internal loader error: unconsumed keys {sorted(payload)}."
        )

    fingerprint = _dataset_fingerprint(
        source_sha256=source_sha256,
        n=n,
        n_train=n_train,
        n_test=n_test,
        n_scenarios=n_scenarios,
        m=m,
        m_complex=m_complex,
        rank_mode_a=rank_mode_a,
        delta_r=delta_r,
        bus_types=bus_types,
        edges=edges,
        theta_rad=theta_rad,
        targets=targets,
        features_re=features_re,
        features_im=features_im,
        labels=labels,
        params=params,
    )

    return Case300ModeAData(
        case=case,
        n=n,
        n_train=n_train,
        n_test=n_test,
        n_scenarios=n_scenarios,
        bus_types=bus_types,
        edges=edges,
        M=m,
        M_complex=m_complex,
        rank_modeA=rank_mode_a,
        delta_r=delta_r,
        theta_rad=theta_rad,
        targets=targets,
        features_re=features_re,
        features_im=features_im,
        features=features,
        scenario_labels=labels,
        params=params,
        slack_buses=slack_buses,
        source_path=candidate.as_posix(),
        source_sha256=source_sha256,
        _fingerprint_value=fingerprint,
    )


__all__ = [
    "REAL_DTYPE",
    "COMPLEX_DTYPE",
    "INTEGER_DTYPE",
    "CASE300_DATASET_SHA256",
    "CASE300_DATASET_SIZE_BYTES",
    "CASE300_DATASET_PATH",
    "CASE300_SCENARIO_LABELS",
    "CASE300_MODE_A_PAIR_TOLERANCE",
    "CASE300_UNIT_MODULUS_TOLERANCE",
    "CASE300_EXACT_TOP_LEVEL_KEYS",
    "Case300DataError",
    "Case300ModeAData",
    "verify_case300_dataset",
    "load_case300_mode_a",
]
