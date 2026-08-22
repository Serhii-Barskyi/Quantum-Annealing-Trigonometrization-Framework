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

"""Strict normalization and auditing of Ocean ``dimod.SampleSet`` results.

Both CSSF Pegasus execution paths terminate at the same boundary:

``local_sqa_gpu``
    The local CUDA SQA emulator wrapped by Ocean embedding.

``pegasus_qpu``
    An explicitly selected D-Wave Leap QPU wrapped by Ocean embedding.

This module converts either returned ``dimod.SampleSet`` into one immutable,
solver-independent representation. It independently re-evaluates every QUBO
energy with :class:`qubo.model.QUBOModel`, restores the model's exact variable
order, validates read counts and backend provenance, preserves chain-break
statistics, and removes secrets from copied metadata.

``dimod`` is intentionally not imported here. Validation is structural so that
project imports and non-Ocean unit tests remain side-effect free. The actual
sampler boundary in :mod:`dwave_backend.sampler` still enforces that runtime
results are genuine ``dimod.SampleSet`` instances.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from dwave_backend import (
    LOCAL_GPU_BACKEND_KIND,
    OCEAN_SAMPLE_VARTYPE,
    PEGASUS_QPU_BACKEND_KIND,
    validate_backend_kind,
    validate_num_reads,
    validate_solver_id,
)
from qubo.model import BINARY_TOLERANCE, QUBOModel


SAMPLE_DTYPE: Final[np.dtype[np.int8]] = np.dtype(np.int8)
ENERGY_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
OCCURRENCE_DTYPE: Final[np.dtype[np.int64]] = np.dtype(np.int64)
DEFAULT_ENERGY_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-8
DEFAULT_ENERGY_RELATIVE_TOLERANCE: Final[float] = 1.0e-10
CHAIN_BREAK_FIELD: Final[str] = "chain_break_fraction"
REQUIRED_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "sample",
    "energy",
    "num_occurrences",
)
SENSITIVE_METADATA_TOKENS: Final[tuple[str, ...]] = (
    "token",
    "password",
    "secret",
    "credential",
    "authorization",
    "api_key",
    "apikey",
)
REDACTED_METADATA_VALUE: Final[str] = "<redacted>"


class OceanSampleSetError(ValueError):
    """Raised when an Ocean sample result violates CSSF contracts."""


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar.") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise OceanSampleSetError(f"{name} must be finite and positive.")
    return normalized


def _optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a hexadecimal SHA-256 string.")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise OceanSampleSetError(
            f"{name} must be a 64-character hexadecimal SHA-256 digest."
        )
    return normalized


def _metadata_key_is_sensitive(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(token in normalized for token in SENSITIVE_METADATA_TOKENS)


def _json_safe(value: Any, *, key: object | None = None) -> Any:
    """Return a deterministic JSON-safe copy with secret redaction."""

    if key is not None and _metadata_key_is_sensitive(key):
        return REDACTED_METADATA_VALUE
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), key=key)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {
            str(item_key): _json_safe(item_value, key=item_key)
            for item_key, item_value in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_json_safe(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ),
        )
    return str(value)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _immutable_metadata(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OceanSampleSetError(f"{name} must be a mapping.")
    normalized = _json_safe(value)
    try:
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise OceanSampleSetError(
            f"{name} cannot be represented as deterministic JSON."
        ) from exc
    decoded = json.loads(encoded)
    frozen = _deep_freeze(decoded)
    if not isinstance(frozen, Mapping):  # pragma: no cover - defensive
        raise OceanSampleSetError(f"{name} did not normalize to a mapping.")
    return frozen


def _metadata_json(value: Mapping[str, Any]) -> bytes:
    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return [thaw(child) for child in item]
        return item

    return json.dumps(
        thaw(value),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _record_names(record: object) -> tuple[str, ...]:
    dtype = getattr(record, "dtype", None)
    names = getattr(dtype, "names", None)
    if names is None:
        raise OceanSampleSetError(
            "SampleSet.record must be a structured NumPy record array."
        )
    return tuple(str(name) for name in names)


def _record_field(record: object, name: str) -> NDArray[Any]:
    try:
        value = getattr(record, name)
    except AttributeError:
        try:
            value = record[name]  # type: ignore[index]
        except Exception as exc:
            raise OceanSampleSetError(
                f"SampleSet.record does not expose {name!r}."
            ) from exc
    return np.asarray(value)


def _vartype_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip().upper()
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized.startswith("VARTYPE."):
            normalized = normalized.split(".", maxsplit=1)[1]
        return normalized
    text = str(value).strip().upper()
    if text.startswith("VARTYPE."):
        text = text.split(".", maxsplit=1)[1]
    return text


def _variable_order(sampleset: object) -> tuple[str, ...]:
    raw_variables = getattr(sampleset, "variables", None)
    if raw_variables is None:
        raise OceanSampleSetError("SampleSet.variables is missing.")
    try:
        variables = tuple(str(value).strip() for value in raw_variables)
    except TypeError as exc:
        raise OceanSampleSetError(
            "SampleSet.variables must be iterable."
        ) from exc
    if not variables or any(not value for value in variables):
        raise OceanSampleSetError(
            "SampleSet.variables must contain non-empty labels."
        )
    if len(set(variables)) != len(variables):
        raise OceanSampleSetError(
            "SampleSet.variables must contain unique labels."
        )
    return variables


def _binary_samples(
    values: object,
    *,
    n_variables: int,
) -> NDArray[np.int8]:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise OceanSampleSetError(
            "SampleSet.record.sample must be a two-dimensional matrix."
        )
    if matrix.shape[0] < 1:
        raise OceanSampleSetError("SampleSet must contain at least one row.")
    if matrix.shape[1] != n_variables:
        raise OceanSampleSetError(
            "SampleSet sample width does not match SampleSet.variables."
        )
    try:
        numeric = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise OceanSampleSetError(
            "SampleSet samples must be numeric binary values."
        ) from exc
    if not np.all(np.isfinite(numeric)):
        raise OceanSampleSetError("SampleSet samples contain non-finite values.")
    close_zero = np.abs(numeric) <= BINARY_TOLERANCE
    close_one = np.abs(numeric - 1.0) <= BINARY_TOLERANCE
    if not np.all(close_zero | close_one):
        raise OceanSampleSetError(
            "SampleSet samples must use BINARY values {0, 1}."
        )
    result = np.ascontiguousarray(np.where(close_one, 1, 0), dtype=SAMPLE_DTYPE)
    result.setflags(write=False)
    return result


def _finite_vector(
    values: object,
    *,
    name: str,
    expected_size: int,
) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=ENERGY_DTYPE).reshape(-1)
    if vector.size != expected_size:
        raise OceanSampleSetError(
            f"{name} must contain {expected_size} values; received "
            f"{vector.size}."
        )
    if not np.all(np.isfinite(vector)):
        raise OceanSampleSetError(f"{name} contains non-finite values.")
    result = np.ascontiguousarray(vector, dtype=ENERGY_DTYPE)
    result.setflags(write=False)
    return result


def _occurrences(values: object, *, expected_size: int) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.reshape(-1).size != expected_size:
        raise OceanSampleSetError(
            "num_occurrences length does not match the number of sample rows."
        )
    try:
        numeric = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise OceanSampleSetError(
            "num_occurrences must contain positive integers."
        ) from exc
    if not np.all(np.isfinite(numeric)):
        raise OceanSampleSetError("num_occurrences contains non-finite values.")
    rounded = np.rint(numeric)
    if not np.all(numeric == rounded) or np.any(rounded < 1):
        raise OceanSampleSetError(
            "num_occurrences must contain strictly positive integers."
        )
    maximum = np.iinfo(OCCURRENCE_DTYPE).max
    if np.any(rounded > maximum):
        raise OceanSampleSetError("num_occurrences exceeds int64 capacity.")
    result = np.ascontiguousarray(rounded, dtype=OCCURRENCE_DTYPE)
    result.setflags(write=False)
    return result


def _chain_breaks(
    record: object,
    *,
    record_names: Sequence[str],
    expected_size: int,
) -> NDArray[np.float64] | None:
    if CHAIN_BREAK_FIELD not in record_names:
        return None
    vector = _finite_vector(
        _record_field(record, CHAIN_BREAK_FIELD),
        name=CHAIN_BREAK_FIELD,
        expected_size=expected_size,
    )
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise OceanSampleSetError(
            "chain_break_fraction values must lie in [0, 1]."
        )
    return vector


def _canonical_order(
    samples: NDArray[np.int8],
    energies: NDArray[np.float64],
) -> NDArray[np.int64]:
    keys: list[NDArray[Any]] = [
        samples[:, index] for index in range(samples.shape[1] - 1, -1, -1)
    ]
    keys.append(energies)
    order = np.lexsort(tuple(keys))
    return np.ascontiguousarray(order, dtype=np.int64)


def _validate_expected_metadata(
    info: Mapping[str, Any],
    *,
    expected_backend: object | None,
    expected_solver_id: object | None,
    expected_topology_fingerprint: object | None,
    expected_bundle_fingerprint: object | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    backend = info.get("cssf_backend")
    normalized_backend = (
        None if backend is None else validate_backend_kind(backend)
    )
    normalized_expected_backend = (
        None
        if expected_backend is None
        else validate_backend_kind(expected_backend)
    )
    if (
        normalized_expected_backend is not None
        and normalized_backend != normalized_expected_backend
    ):
        raise OceanSampleSetError(
            "SampleSet backend provenance does not match the selected bundle."
        )

    solver_id = info.get("cssf_solver_id")
    normalized_solver_id = (
        None if solver_id is None else validate_solver_id(solver_id)
    )
    normalized_expected_solver_id = (
        None
        if expected_solver_id is None
        else validate_solver_id(expected_solver_id)
    )
    if (
        expected_solver_id is not None
        and normalized_expected_solver_id != normalized_solver_id
    ):
        raise OceanSampleSetError(
            "SampleSet solver provenance does not match the selected bundle."
        )
    if (
        normalized_backend == LOCAL_GPU_BACKEND_KIND
        and normalized_solver_id is not None
    ):
        raise OceanSampleSetError(
            "Local emulator SampleSet must not claim a real QPU solver ID."
        )
    if (
        normalized_backend == PEGASUS_QPU_BACKEND_KIND
        and normalized_solver_id is None
    ):
        raise OceanSampleSetError(
            "QPU SampleSet must include its explicit solver ID."
        )

    topology_fingerprint = _optional_digest(
        info.get("cssf_topology_fingerprint"),
        name="cssf_topology_fingerprint",
    )
    expected_topology = _optional_digest(
        expected_topology_fingerprint,
        name="expected_topology_fingerprint",
    )
    if expected_topology is not None and topology_fingerprint != expected_topology:
        raise OceanSampleSetError(
            "SampleSet topology fingerprint does not match the selected bundle."
        )

    bundle_fingerprint = _optional_digest(
        info.get("cssf_bundle_fingerprint"),
        name="cssf_bundle_fingerprint",
    )
    expected_bundle = _optional_digest(
        expected_bundle_fingerprint,
        name="expected_bundle_fingerprint",
    )
    if expected_bundle is not None and bundle_fingerprint != expected_bundle:
        raise OceanSampleSetError(
            "SampleSet bundle fingerprint does not match the selected bundle."
        )

    return (
        normalized_backend,
        normalized_solver_id,
        topology_fingerprint,
        bundle_fingerprint,
    )


@dataclass(frozen=True, slots=True, init=False)
class OceanSampleBatch:
    """Immutable, model-ordered, independently audited Ocean samples."""

    variable_order: tuple[str, ...]
    samples: NDArray[np.int8]
    energies: NDArray[np.float64]
    num_occurrences: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    chain_break_fraction: NDArray[np.float64] | None
    total_reads: int
    model_fingerprint: str
    backend: str | None
    solver_id: str | None
    topology_fingerprint: str | None
    bundle_fingerprint: str | None
    info: Mapping[str, Any]

    def __init__(
        self,
        *,
        variable_order: Sequence[str],
        samples: NDArray[np.int8],
        energies: NDArray[np.float64],
        num_occurrences: NDArray[np.int64],
        chain_break_fraction: NDArray[np.float64] | None,
        model_fingerprint: str,
        backend: str | None,
        solver_id: str | None,
        topology_fingerprint: str | None,
        bundle_fingerprint: str | None,
        info: Mapping[str, Any],
    ) -> None:
        variables = tuple(str(value) for value in variable_order)
        sample_copy = np.ascontiguousarray(samples, dtype=SAMPLE_DTYPE)
        energy_copy = np.ascontiguousarray(energies, dtype=ENERGY_DTYPE)
        occurrence_copy = np.ascontiguousarray(
            num_occurrences,
            dtype=OCCURRENCE_DTYPE,
        )
        if sample_copy.ndim != 2 or sample_copy.shape[1] != len(variables):
            raise OceanSampleSetError(
                "samples shape must match variable_order."
            )
        row_count = sample_copy.shape[0]
        if energy_copy.shape != (row_count,) or occurrence_copy.shape != (
            row_count,
        ):
            raise OceanSampleSetError(
                "energies and num_occurrences must match the sample rows."
            )
        total_reads = sum(int(value) for value in occurrence_copy)
        if total_reads < 1:
            raise OceanSampleSetError("total_reads must be positive.")
        probabilities = np.ascontiguousarray(
            occurrence_copy.astype(np.float64) / float(total_reads),
            dtype=ENERGY_DTYPE,
        )
        chain_copy: NDArray[np.float64] | None
        if chain_break_fraction is None:
            chain_copy = None
        else:
            chain_copy = np.ascontiguousarray(
                chain_break_fraction,
                dtype=ENERGY_DTYPE,
            )
            if chain_copy.shape != (row_count,):
                raise OceanSampleSetError(
                    "chain_break_fraction must match the sample rows."
                )
        for array in (
            sample_copy,
            energy_copy,
            occurrence_copy,
            probabilities,
            chain_copy,
        ):
            if array is not None:
                array.setflags(write=False)

        model_digest = _optional_digest(
            model_fingerprint,
            name="model_fingerprint",
        )
        if model_digest is None:  # pragma: no cover - input is required
            raise OceanSampleSetError("model_fingerprint is required.")
        normalized_backend = (
            None if backend is None else validate_backend_kind(backend)
        )
        normalized_solver = (
            None if solver_id is None else validate_solver_id(solver_id)
        )
        object.__setattr__(self, "variable_order", variables)
        object.__setattr__(self, "samples", sample_copy)
        object.__setattr__(self, "energies", energy_copy)
        object.__setattr__(self, "num_occurrences", occurrence_copy)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "chain_break_fraction", chain_copy)
        object.__setattr__(self, "total_reads", total_reads)
        object.__setattr__(self, "model_fingerprint", model_digest)
        object.__setattr__(self, "backend", normalized_backend)
        object.__setattr__(self, "solver_id", normalized_solver)
        object.__setattr__(
            self,
            "topology_fingerprint",
            _optional_digest(
                topology_fingerprint,
                name="topology_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "bundle_fingerprint",
            _optional_digest(bundle_fingerprint, name="bundle_fingerprint"),
        )
        object.__setattr__(
            self,
            "info",
            _immutable_metadata(info, name="SampleSet info"),
        )

    @property
    def n_rows(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_variables(self) -> int:
        return int(self.samples.shape[1])

    @property
    def best_energy(self) -> float:
        return float(self.energies[0])

    @property
    def best_sample(self) -> NDArray[np.int8]:
        result = np.array(self.samples[0], dtype=SAMPLE_DTYPE, copy=True)
        result.setflags(write=False)
        return result

    @property
    def best_probability(self) -> float:
        mask = np.isclose(
            self.energies,
            self.best_energy,
            rtol=0.0,
            atol=DEFAULT_ENERGY_ABSOLUTE_TOLERANCE,
        )
        return float(np.sum(self.probabilities[mask], dtype=np.float64))

    @property
    def weighted_mean_energy(self) -> float:
        return float(np.dot(self.probabilities, self.energies))

    @property
    def weighted_energy_variance(self) -> float:
        differences = self.energies - self.weighted_mean_energy
        return float(np.dot(self.probabilities, differences * differences))

    @property
    def weighted_chain_break_fraction(self) -> float | None:
        if self.chain_break_fraction is None:
            return None
        return float(
            np.dot(self.probabilities, self.chain_break_fraction)
        )

    def probability_of(self, mask: object) -> float:
        values = np.asarray(mask)
        if values.shape != (self.n_rows,) or values.dtype != np.bool_:
            raise OceanSampleSetError(
                "mask must be a boolean vector with one value per row."
            )
        return float(np.sum(self.probabilities[values], dtype=np.float64))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-OceanSampleBatch-v1\0")
        digest.update(
            json.dumps(
                self.variable_order,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(self.samples.tobytes(order="C"))
        digest.update(self.energies.tobytes(order="C"))
        digest.update(self.num_occurrences.tobytes(order="C"))
        if self.chain_break_fraction is None:
            digest.update(b"no-chain-break-field")
        else:
            digest.update(self.chain_break_fraction.tobytes(order="C"))
        digest.update(self.model_fingerprint.encode("ascii"))
        digest.update((self.backend or "").encode("utf-8"))
        digest.update((self.solver_id or "").encode("utf-8"))
        digest.update((self.topology_fingerprint or "").encode("ascii"))
        digest.update((self.bundle_fingerprint or "").encode("ascii"))
        digest.update(_metadata_json(self.info))
        return digest.hexdigest()


def normalize_ocean_sampleset(
    sampleset: object,
    model: QUBOModel,
    *,
    expected_num_reads: int | None = None,
    expected_backend: object | None = None,
    expected_solver_id: object | None = None,
    expected_topology_fingerprint: object | None = None,
    expected_bundle_fingerprint: object | None = None,
    energy_absolute_tolerance: float = DEFAULT_ENERGY_ABSOLUTE_TOLERANCE,
    energy_relative_tolerance: float = DEFAULT_ENERGY_RELATIVE_TOLERANCE,
) -> OceanSampleBatch:
    """Validate and normalize one emulator or Leap QPU ``SampleSet``.

    The returned rows are sorted deterministically by energy and then by the
    binary sample in the QUBO model's exact variable order. Reported energies
    are never trusted blindly: each is recalculated with ``model``.
    """

    if not isinstance(model, QUBOModel):
        raise TypeError("model must be QUBOModel.")
    absolute_tolerance = _positive_float(
        energy_absolute_tolerance,
        name="energy_absolute_tolerance",
    )
    relative_tolerance = _positive_float(
        energy_relative_tolerance,
        name="energy_relative_tolerance",
    )
    vartype = _vartype_name(getattr(sampleset, "vartype", None))
    if vartype != OCEAN_SAMPLE_VARTYPE:
        raise OceanSampleSetError(
            "CSSF QUBO results must use dimod BINARY vartype; "
            f"received {vartype!r}."
        )

    variables = _variable_order(sampleset)
    if set(variables) != set(model.variable_order):
        missing = tuple(sorted(set(model.variable_order) - set(variables)))
        extra = tuple(sorted(set(variables) - set(model.variable_order)))
        raise OceanSampleSetError(
            "SampleSet variables differ from the QUBO model; "
            f"missing={missing}, extra={extra}."
        )

    record = getattr(sampleset, "record", None)
    if record is None:
        raise OceanSampleSetError("SampleSet.record is missing.")
    names = _record_names(record)
    missing_fields = tuple(
        field for field in REQUIRED_RECORD_FIELDS if field not in names
    )
    if missing_fields:
        raise OceanSampleSetError(
            f"SampleSet.record is missing required fields {missing_fields}."
        )

    raw_samples = _binary_samples(
        _record_field(record, "sample"),
        n_variables=len(variables),
    )
    reported_energies = _finite_vector(
        _record_field(record, "energy"),
        name="energy",
        expected_size=raw_samples.shape[0],
    )
    num_occurrences = _occurrences(
        _record_field(record, "num_occurrences"),
        expected_size=raw_samples.shape[0],
    )
    chain_break_fraction = _chain_breaks(
        record,
        record_names=names,
        expected_size=raw_samples.shape[0],
    )

    column_index = {label: index for index, label in enumerate(variables)}
    model_columns = [column_index[label] for label in model.variable_order]
    ordered_samples = np.ascontiguousarray(
        raw_samples[:, model_columns],
        dtype=SAMPLE_DTYPE,
    )
    audited_energies = model.energies(ordered_samples)
    if not np.allclose(
        reported_energies,
        audited_energies,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    ):
        differences = np.abs(reported_energies - audited_energies)
        worst_index = int(np.argmax(differences))
        raise OceanSampleSetError(
            "SampleSet energy audit failed at row "
            f"{worst_index}: reported={reported_energies[worst_index]!r}, "
            f"recomputed={audited_energies[worst_index]!r}, "
            f"absolute_error={differences[worst_index]!r}."
        )

    total_reads = sum(int(value) for value in num_occurrences)
    if expected_num_reads is not None:
        normalized_expected_reads = validate_num_reads(expected_num_reads)
        if total_reads != normalized_expected_reads:
            raise OceanSampleSetError(
                "SampleSet occurrence total does not match expected_num_reads; "
                f"received {total_reads}, expected {normalized_expected_reads}."
            )

    raw_info = getattr(sampleset, "info", None)
    if not isinstance(raw_info, Mapping):
        raise OceanSampleSetError("SampleSet.info must be a mapping.")
    (
        backend,
        solver_id,
        topology_fingerprint,
        bundle_fingerprint,
    ) = _validate_expected_metadata(
        raw_info,
        expected_backend=expected_backend,
        expected_solver_id=expected_solver_id,
        expected_topology_fingerprint=expected_topology_fingerprint,
        expected_bundle_fingerprint=expected_bundle_fingerprint,
    )

    order = _canonical_order(ordered_samples, audited_energies)
    sorted_chain_break = (
        None
        if chain_break_fraction is None
        else np.ascontiguousarray(
            chain_break_fraction[order],
            dtype=ENERGY_DTYPE,
        )
    )
    return OceanSampleBatch(
        variable_order=model.variable_order,
        samples=np.ascontiguousarray(ordered_samples[order], dtype=SAMPLE_DTYPE),
        energies=np.ascontiguousarray(audited_energies[order], dtype=ENERGY_DTYPE),
        num_occurrences=np.ascontiguousarray(
            num_occurrences[order],
            dtype=OCCURRENCE_DTYPE,
        ),
        chain_break_fraction=sorted_chain_break,
        model_fingerprint=model.fingerprint(),
        backend=backend,
        solver_id=solver_id,
        topology_fingerprint=topology_fingerprint,
        bundle_fingerprint=bundle_fingerprint,
        info=raw_info,
    )


def normalize_bundle_sampleset(
    sampleset: object,
    model: QUBOModel,
    bundle: object,
    *,
    expected_num_reads: int | None = None,
    energy_absolute_tolerance: float = DEFAULT_ENERGY_ABSOLUTE_TOLERANCE,
    energy_relative_tolerance: float = DEFAULT_ENERGY_RELATIVE_TOLERANCE,
) -> OceanSampleBatch:
    """Normalize a result and prove ownership by one sampler bundle."""

    topology = getattr(bundle, "topology", None)
    topology_fingerprint_method = getattr(topology, "fingerprint", None)
    bundle_fingerprint_method = getattr(bundle, "fingerprint", None)
    if not callable(topology_fingerprint_method):
        raise OceanSampleSetError(
            "bundle.topology must expose fingerprint()."
        )
    if not callable(bundle_fingerprint_method):
        raise OceanSampleSetError("bundle must expose fingerprint().")
    return normalize_ocean_sampleset(
        sampleset,
        model,
        expected_num_reads=expected_num_reads,
        expected_backend=getattr(bundle, "mode", None),
        expected_solver_id=getattr(bundle, "solver_id", None),
        expected_topology_fingerprint=topology_fingerprint_method(),
        expected_bundle_fingerprint=bundle_fingerprint_method(),
        energy_absolute_tolerance=energy_absolute_tolerance,
        energy_relative_tolerance=energy_relative_tolerance,
    )


__all__ = [
    "SAMPLE_DTYPE",
    "ENERGY_DTYPE",
    "OCCURRENCE_DTYPE",
    "DEFAULT_ENERGY_ABSOLUTE_TOLERANCE",
    "DEFAULT_ENERGY_RELATIVE_TOLERANCE",
    "CHAIN_BREAK_FIELD",
    "REQUIRED_RECORD_FIELDS",
    "SENSITIVE_METADATA_TOKENS",
    "REDACTED_METADATA_VALUE",
    "OceanSampleSetError",
    "OceanSampleBatch",
    "normalize_ocean_sampleset",
    "normalize_bundle_sampleset",
]
