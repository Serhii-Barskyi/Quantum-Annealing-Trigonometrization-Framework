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

"""Unified Pegasus solve orchestration for emulator and D-Wave Leap QPU.

The module is the common execution layer for both notebook-selectable modes::

    QUBOModel
    -> explicit local Pegasus GPU emulator or explicit Leap Pegasus QPU
    -> Ocean ``dimod.SampleSet``
    -> independent CSSF energy and provenance audit
    -> optional BESS placement decoding.

The selected backend is never changed implicitly.  Emulator failure cannot
trigger QPU submission, QPU failure cannot trigger emulator execution, and no
classical fallback is implemented.  Ocean and dimod are imported lazily only
when a solve is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config.schema import EmulatorConfig, PegasusQPUConfig
from dwave_backend import (
    LOCAL_GPU_BACKEND_KIND,
    PEGASUS_QPU_BACKEND_KIND,
    validate_backend_kind,
    validate_num_reads,
)
from dwave_backend.sampler import (
    DEFAULT_EMULATOR_MEMORY_FRACTION,
    DEFAULT_QPU_LABEL,
    OceanSamplerBundle,
    SamplerMode,
    build_ocean_sampler,
)
from dwave_backend.sampleset import (
    DEFAULT_ENERGY_ABSOLUTE_TOLERANCE,
    DEFAULT_ENERGY_RELATIVE_TOLERANCE,
    OceanSampleBatch,
    normalize_bundle_sampleset,
)
from opf.bess_constraints import BESSPlacement
from qubo.builder import BESSPlacementQUBO
from qubo.model import QUBOModel


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
INTEGER_DTYPE: Final[np.dtype[np.int8]] = np.dtype(np.int8)
BOOLEAN_DTYPE: Final[np.dtype[np.bool_]] = np.dtype(np.bool_)
DIMOD_MODULE_NAME: Final[str] = "dimod"
DEFAULT_SOLVER_SEED: Final[int] = 505
DEFAULT_REQUIRE_FEASIBLE_BESS_SAMPLE: Final[bool] = True
DEFAULT_CLOSE_EXTERNAL_BUNDLE: Final[bool] = False
DEFAULT_CLOSE_INTERNAL_BUNDLE: Final[bool] = True

SourceKind = Literal["qubo_model", "bess_placement_qubo"]


class DWaveSolverError(RuntimeError):
    """Raised when unified Pegasus solve orchestration fails."""


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise DWaveSolverError(f"{name} must be non-negative.")
    return value


def _positive_float(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise DWaveSolverError(
            f"{name} must be finite and strictly positive."
        )
    return normalized


def _memory_fraction(value: object) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized <= 1.0:
        raise DWaveSolverError("memory_fraction must lie in (0, 1].")
    return normalized


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool.")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise DWaveSolverError(f"{name} must not be empty.")
    return normalized


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name=name)


def _sha256_digest(value: object, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise DWaveSolverError(f"{name} must be a SHA-256 digest.")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise DWaveSolverError(
            f"{name} must be a hexadecimal SHA-256 digest."
        ) from exc
    return normalized


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


def _immutable_json_mapping(
    metadata: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    source = {} if metadata is None else dict(metadata)
    try:
        encoded = json.dumps(
            source,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise DWaveSolverError(
            f"{name} must be JSON-serializable and contain no NaN."
        ) from exc
    frozen = _freeze_json(json.loads(encoded))
    if not isinstance(frozen, Mapping):
        raise DWaveSolverError(f"{name} normalization failed.")
    return frozen


def _readonly_binary_sample(
    sample: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.int8]:
    values = np.asarray(sample)
    if values.ndim != 1 or values.size != expected_size:
        raise DWaveSolverError(
            f"{name} must contain exactly {expected_size} values."
        )
    numeric = np.asarray(values, dtype=REAL_DTYPE)
    if not np.all(np.isfinite(numeric)):
        raise DWaveSolverError(f"{name} contains non-finite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise DWaveSolverError(f"{name} must be exactly binary.")
    result = np.array(numeric, dtype=INTEGER_DTYPE, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_bool_mask(
    mask: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.bool_]:
    values = np.asarray(mask)
    if values.ndim != 1 or values.size != expected_size:
        raise DWaveSolverError(
            f"{name} must contain exactly {expected_size} values."
        )
    if values.dtype.kind not in {"b", "i", "u", "f"}:
        raise DWaveSolverError(f"{name} must be boolean-compatible.")
    numeric = np.asarray(values, dtype=REAL_DTYPE)
    if not np.all(np.isfinite(numeric)):
        raise DWaveSolverError(f"{name} contains non-finite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise DWaveSolverError(f"{name} must contain only 0/1 values.")
    result = np.array(numeric != 0.0, dtype=BOOLEAN_DTYPE, order="C")
    result.setflags(write=False)
    return result


def _load_dimod() -> Any:
    try:
        return importlib.import_module(DIMOD_MODULE_NAME)
    except Exception as exc:
        raise DWaveSolverError(
            "dimod is unavailable; install the D-Wave Ocean SDK."
        ) from exc


def _build_binary_quadratic_model(model: QUBOModel) -> object:
    """Create a dimod BINARY BQM without dropping zero-bias variables."""

    if not isinstance(model, QUBOModel):
        raise TypeError("model must be QUBOModel.")
    dimod = _load_dimod()
    bqm_class = getattr(dimod, "BinaryQuadraticModel", None)
    binary_vartype = getattr(dimod, "BINARY", None)
    if bqm_class is None or not callable(bqm_class) or binary_vartype is None:
        raise DWaveSolverError(
            "dimod.BinaryQuadraticModel or dimod.BINARY is unavailable."
        )

    linear = {
        variable: float(model.linear[index])
        for index, variable in enumerate(model.variable_order)
    }
    quadratic = {
        (model.variable_order[first], model.variable_order[second]): float(
            model.quadratic[first, second]
        )
        for first in range(model.n_variables)
        for second in range(first + 1, model.n_variables)
        if float(model.quadratic[first, second]) != 0.0
    }
    try:
        bqm = bqm_class(
            linear,
            quadratic,
            float(model.offset),
            binary_vartype,
        )
    except Exception as exc:
        raise DWaveSolverError(
            "dimod could not construct the CSSF binary quadratic model."
        ) from exc

    variables = tuple(str(value) for value in getattr(bqm, "variables", ()))
    if set(variables) != set(model.variable_order):
        raise DWaveSolverError(
            "Constructed BQM variables differ from QUBOModel.variable_order."
        )
    vartype_name = str(getattr(getattr(bqm, "vartype", None), "name", ""))
    if vartype_name.upper() != "BINARY":
        raise DWaveSolverError("Constructed BQM must use BINARY vartype.")
    return bqm


def _source_model(
    source: QUBOModel | BESSPlacementQUBO,
) -> tuple[QUBOModel, SourceKind, BESSPlacementQUBO | None]:
    if isinstance(source, BESSPlacementQUBO):
        return source.model, "bess_placement_qubo", source
    if isinstance(source, QUBOModel):
        return source, "qubo_model", None
    raise TypeError("source must be QUBOModel or BESSPlacementQUBO.")


def _resolved_num_reads(
    mode: str,
    *,
    emulator_config: EmulatorConfig | None,
    qpu_config: PegasusQPUConfig | None,
    override: int | None,
) -> int:
    if override is not None:
        return validate_num_reads(override)
    if mode == LOCAL_GPU_BACKEND_KIND:
        if emulator_config is None:
            raise DWaveSolverError(
                "emulator_config is required in local_sqa_gpu mode."
            )
        return validate_num_reads(emulator_config.num_reads)
    if qpu_config is None:
        raise DWaveSolverError(
            "qpu_config is required in pegasus_qpu mode."
        )
    return validate_num_reads(qpu_config.num_reads)


def _resolve_mode(value: SamplerMode | str) -> str:
    raw = value.value if isinstance(value, SamplerMode) else value
    return validate_backend_kind(raw)


def _feasible_mask(
    batch: OceanSampleBatch,
    placement_qubo: BESSPlacementQUBO | None,
) -> NDArray[np.bool_]:
    if placement_qubo is None:
        mask = np.ones(batch.n_rows, dtype=BOOLEAN_DTYPE)
        mask.setflags(write=False)
        return mask
    mask = np.fromiter(
        (
            placement_qubo.is_feasible(sample)
            for sample in batch.samples
        ),
        dtype=BOOLEAN_DTYPE,
        count=batch.n_rows,
    )
    mask = np.ascontiguousarray(mask, dtype=BOOLEAN_DTYPE)
    mask.setflags(write=False)
    return mask


def _best_selected_index(
    batch: OceanSampleBatch,
    feasible_mask: NDArray[np.bool_],
    *,
    require_feasible: bool,
) -> int:
    feasible_positions = np.flatnonzero(feasible_mask)
    if feasible_positions.size:
        return int(feasible_positions[0])
    if require_feasible:
        raise DWaveSolverError(
            "No feasible BESS placement was returned by the selected backend."
        )
    return 0


@dataclass(frozen=True, slots=True)
class DWaveSolveConfig:
    """Backend-independent controls for one Ocean/Pegasus solve."""

    mode: SamplerMode | str = SamplerMode.EMULATOR
    num_reads: int | None = None
    label: str = DEFAULT_QPU_LABEL
    seed: int = DEFAULT_SOLVER_SEED
    memory_fraction: float = DEFAULT_EMULATOR_MEMORY_FRACTION
    energy_absolute_tolerance: float = DEFAULT_ENERGY_ABSOLUTE_TOLERANCE
    energy_relative_tolerance: float = DEFAULT_ENERGY_RELATIVE_TOLERANCE
    require_feasible_bess_sample: bool = (
        DEFAULT_REQUIRE_FEASIBLE_BESS_SAMPLE
    )
    close_external_bundle: bool = DEFAULT_CLOSE_EXTERNAL_BUNDLE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = _resolve_mode(self.mode)
        reads = (
            None if self.num_reads is None else validate_num_reads(self.num_reads)
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "num_reads", reads)
        object.__setattr__(self, "label", _nonempty_string(self.label, name="label"))
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, name="seed"))
        object.__setattr__(self, "memory_fraction", _memory_fraction(self.memory_fraction))
        object.__setattr__(
            self,
            "energy_absolute_tolerance",
            _positive_float(
                self.energy_absolute_tolerance,
                name="energy_absolute_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "energy_relative_tolerance",
            _positive_float(
                self.energy_relative_tolerance,
                name="energy_relative_tolerance",
            ),
        )
        require_feasible = _strict_bool(
            self.require_feasible_bess_sample,
            name="require_feasible_bess_sample",
        )
        if not require_feasible:
            raise DWaveSolverError(
                "BESS solve results must require at least one feasible sample."
            )
        object.__setattr__(
            self,
            "require_feasible_bess_sample",
            require_feasible,
        )
        object.__setattr__(
            self,
            "close_external_bundle",
            _strict_bool(
                self.close_external_bundle,
                name="close_external_bundle",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_json_mapping(self.metadata, name="solver metadata"),
        )

    def fingerprint(self) -> str:
        payload = {
            "mode": self.mode,
            "num_reads": self.num_reads,
            "label": self.label,
            "seed": self.seed,
            "memory_fraction": self.memory_fraction,
            "energy_absolute_tolerance": self.energy_absolute_tolerance,
            "energy_relative_tolerance": self.energy_relative_tolerance,
            "require_feasible_bess_sample": self.require_feasible_bess_sample,
            "close_external_bundle": self.close_external_bundle,
            "metadata": _thaw_json(self.metadata),
        }
        digest = hashlib.sha256()
        digest.update(b"CSSF-DWaveSolveConfig-v1\0")
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DWaveSolveResult:
    """Immutable audited result shared by generic and BESS QUBO solves."""

    source_kind: SourceKind
    batch: OceanSampleBatch
    selected_index: int
    selected_sample: NDArray[np.int8]
    selected_energy: float
    feasible_mask: NDArray[np.bool_]
    feasible_probability: float
    placement: BESSPlacement | None
    energy_breakdown: Mapping[str, float] | None
    solve_config_fingerprint: str
    source_fingerprint: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.source_kind not in {"qubo_model", "bess_placement_qubo"}:
            raise DWaveSolverError("source_kind is invalid.")
        if not isinstance(self.batch, OceanSampleBatch):
            raise TypeError("batch must be OceanSampleBatch.")
        selected_index = _nonnegative_integer(
            self.selected_index,
            name="selected_index",
        )
        if selected_index >= self.batch.n_rows:
            raise DWaveSolverError("selected_index is outside the sample batch.")
        selected_sample = _readonly_binary_sample(
            self.selected_sample,
            expected_size=self.batch.n_variables,
            name="selected_sample",
        )
        if not np.array_equal(selected_sample, self.batch.samples[selected_index]):
            raise DWaveSolverError(
                "selected_sample does not match batch[selected_index]."
            )
        selected_energy = float(self.selected_energy)
        if not math.isfinite(selected_energy):
            raise DWaveSolverError("selected_energy must be finite.")
        if not math.isclose(
            selected_energy,
            float(self.batch.energies[selected_index]),
            rel_tol=0.0,
            abs_tol=DEFAULT_ENERGY_ABSOLUTE_TOLERANCE,
        ):
            raise DWaveSolverError(
                "selected_energy does not match the audited sample batch."
            )
        feasible_mask = _readonly_bool_mask(
            self.feasible_mask,
            expected_size=self.batch.n_rows,
            name="feasible_mask",
        )
        feasible_probability = float(self.feasible_probability)
        if not math.isfinite(feasible_probability) or not (
            0.0 <= feasible_probability <= 1.0
        ):
            raise DWaveSolverError(
                "feasible_probability must lie in [0, 1]."
            )
        expected_probability = self.batch.probability_of(feasible_mask)
        if not math.isclose(
            feasible_probability,
            expected_probability,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise DWaveSolverError(
                "feasible_probability does not match feasible_mask."
            )
        if self.source_kind == "qubo_model":
            if self.placement is not None or self.energy_breakdown is not None:
                raise DWaveSolverError(
                    "Generic QUBO results cannot contain BESS placement data."
                )
        else:
            if not isinstance(self.placement, BESSPlacement):
                raise DWaveSolverError(
                    "BESS result must contain a decoded BESSPlacement."
                )
            if not feasible_mask[selected_index]:
                raise DWaveSolverError(
                    "Selected BESS sample must satisfy placement constraints."
                )
            if not isinstance(self.energy_breakdown, Mapping):
                raise DWaveSolverError(
                    "BESS result must contain an energy breakdown."
                )
        object.__setattr__(self, "selected_index", selected_index)
        object.__setattr__(self, "selected_sample", selected_sample)
        object.__setattr__(self, "selected_energy", selected_energy)
        object.__setattr__(self, "feasible_mask", feasible_mask)
        object.__setattr__(self, "feasible_probability", feasible_probability)
        object.__setattr__(
            self,
            "energy_breakdown",
            None
            if self.energy_breakdown is None
            else _immutable_json_mapping(
                self.energy_breakdown,
                name="energy_breakdown",
            ),
        )
        object.__setattr__(
            self,
            "solve_config_fingerprint",
            _sha256_digest(
                self.solve_config_fingerprint,
                name="solve_config_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "source_fingerprint",
            _sha256_digest(self.source_fingerprint, name="source_fingerprint"),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_json_mapping(self.metadata, name="result metadata"),
        )

    @property
    def backend(self) -> str | None:
        return self.batch.backend

    @property
    def solver_id(self) -> str | None:
        return self.batch.solver_id

    @property
    def best_energy(self) -> float:
        return self.batch.best_energy

    @property
    def best_probability(self) -> float:
        return self.batch.best_probability

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-DWaveSolveResult-v1\0")
        digest.update(self.source_kind.encode("utf-8"))
        digest.update(self.batch.fingerprint().encode("ascii"))
        digest.update(
            np.asarray([self.selected_index], dtype=np.int64).tobytes(order="C")
        )
        digest.update(self.selected_sample.tobytes(order="C"))
        digest.update(
            np.asarray(
                [self.selected_energy, self.feasible_probability],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        digest.update(self.feasible_mask.tobytes(order="C"))
        digest.update(self.solve_config_fingerprint.encode("ascii"))
        digest.update(self.source_fingerprint.encode("ascii"))
        if self.placement is not None:
            digest.update(self.placement.selection.tobytes(order="C"))
            digest.update(
                json.dumps(
                    list(self.placement.selected_buses),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
        if self.energy_breakdown is not None:
            digest.update(
                json.dumps(
                    _thaw_json(self.energy_breakdown),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        digest.update(
            json.dumps(
                _thaw_json(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint(),
            "source_kind": self.source_kind,
            "source_fingerprint": self.source_fingerprint,
            "solve_config_fingerprint": self.solve_config_fingerprint,
            "batch_fingerprint": self.batch.fingerprint(),
            "backend": self.backend,
            "solver_id": self.solver_id,
            "topology_fingerprint": self.batch.topology_fingerprint,
            "bundle_fingerprint": self.batch.bundle_fingerprint,
            "total_reads": self.batch.total_reads,
            "unique_samples": self.batch.n_rows,
            "selected_index": self.selected_index,
            "selected_sample": self.selected_sample.astype(int).tolist(),
            "selected_energy": self.selected_energy,
            "best_energy": self.best_energy,
            "best_probability": self.best_probability,
            "feasible_probability": self.feasible_probability,
            "selected_buses": (
                None
                if self.placement is None
                else list(self.placement.selected_buses)
            ),
            "energy_breakdown": (
                None
                if self.energy_breakdown is None
                else _thaw_json(self.energy_breakdown)
            ),
            "metadata": _thaw_json(self.metadata),
        }


def solve_pegasus_qubo(
    source: QUBOModel | BESSPlacementQUBO,
    *,
    config: DWaveSolveConfig | None = None,
    emulator_config: EmulatorConfig | None = None,
    qpu_config: PegasusQPUConfig | None = None,
    bundle: OceanSamplerBundle | None = None,
    token: str | None = None,
    endpoint: str | None = None,
    region: str | None = None,
) -> DWaveSolveResult:
    """Solve one QUBO through the explicitly selected Pegasus backend.

    ``token`` is passed directly to Ocean sampler construction and is never
    copied into the result, manifest, metadata, or fingerprints.
    """

    solve_config = DWaveSolveConfig() if config is None else config
    if not isinstance(solve_config, DWaveSolveConfig):
        raise TypeError("config must be DWaveSolveConfig or None.")
    model, source_kind, placement_qubo = _source_model(source)
    mode = _resolve_mode(solve_config.mode)
    explicit_token = _optional_string(token, name="token")
    explicit_endpoint = _optional_string(endpoint, name="endpoint")
    explicit_region = _optional_string(region, name="region")

    owns_bundle = bundle is None
    active_bundle: OceanSamplerBundle
    if bundle is None:
        active_bundle = build_ocean_sampler(
            mode,
            emulator_config=emulator_config,
            qpu_config=qpu_config,
            seed=solve_config.seed,
            memory_fraction=solve_config.memory_fraction,
            token=explicit_token,
            endpoint=explicit_endpoint,
            region=explicit_region,
        )
    else:
        if not isinstance(bundle, OceanSamplerBundle):
            raise TypeError("bundle must be OceanSamplerBundle or None.")
        if emulator_config is not None or qpu_config is not None:
            raise DWaveSolverError(
                "emulator_config and qpu_config must be omitted when bundle "
                "is supplied explicitly."
            )
        if any(value is not None for value in (explicit_token, explicit_endpoint, explicit_region)):
            raise DWaveSolverError(
                "token, endpoint, and region must be omitted with an existing bundle."
            )
        if bundle.mode != mode:
            raise DWaveSolverError(
                "Existing bundle mode differs from DWaveSolveConfig.mode."
            )
        active_bundle = bundle

    if solve_config.num_reads is not None:
        resolved_reads = validate_num_reads(solve_config.num_reads)
    elif owns_bundle:
        resolved_reads = _resolved_num_reads(
            mode,
            emulator_config=emulator_config,
            qpu_config=qpu_config,
            override=None,
        )
    else:
        bundle_reads = active_bundle.default_sample_parameters.get("num_reads")
        resolved_reads = validate_num_reads(bundle_reads)

    close_bundle = (
        DEFAULT_CLOSE_INTERNAL_BUNDLE
        if owns_bundle
        else solve_config.close_external_bundle
    )
    try:
        bqm = _build_binary_quadratic_model(model)
        raw_sampleset = active_bundle.sample_bqm(
            bqm,
            num_reads=resolved_reads,
            label=solve_config.label,
        )
        batch = normalize_bundle_sampleset(
            raw_sampleset,
            model,
            active_bundle,
            expected_num_reads=resolved_reads,
            energy_absolute_tolerance=solve_config.energy_absolute_tolerance,
            energy_relative_tolerance=solve_config.energy_relative_tolerance,
        )
    finally:
        if close_bundle:
            active_bundle.close()

    feasible_mask = _feasible_mask(batch, placement_qubo)
    selected_index = _best_selected_index(
        batch,
        feasible_mask,
        require_feasible=(
            placement_qubo is not None
            and solve_config.require_feasible_bess_sample
        ),
    )
    selected_sample = batch.samples[selected_index]
    placement: BESSPlacement | None = None
    energy_breakdown: Mapping[str, float] | None = None
    if placement_qubo is not None:
        placement = placement_qubo.decode(selected_sample)
        energy_breakdown = placement_qubo.energy_breakdown(selected_sample)
        if not math.isclose(
            float(energy_breakdown["total"]),
            float(batch.energies[selected_index]),
            rel_tol=solve_config.energy_relative_tolerance,
            abs_tol=solve_config.energy_absolute_tolerance,
        ):
            raise DWaveSolverError(
                "BESS energy breakdown differs from the audited batch energy."
            )

    source_fingerprint = (
        source.fingerprint()
        if isinstance(source, BESSPlacementQUBO)
        else model.fingerprint()
    )
    metadata = {
        "algorithm": "pegasus_binary_quadratic_sampling",
        "backend_mode": mode,
        "fallback_allowed": False,
        "source_kind": source_kind,
        "model_fingerprint": model.fingerprint(),
        "source_fingerprint": source_fingerprint,
        "bundle_owned_by_solver": owns_bundle,
        "bundle_closed_by_solver": close_bundle,
        "requested_num_reads": resolved_reads,
        "token_stored": False,
        "user_metadata": _thaw_json(solve_config.metadata),
    }
    return DWaveSolveResult(
        source_kind=source_kind,
        batch=batch,
        selected_index=selected_index,
        selected_sample=selected_sample,
        selected_energy=float(batch.energies[selected_index]),
        feasible_mask=feasible_mask,
        feasible_probability=batch.probability_of(feasible_mask),
        placement=placement,
        energy_breakdown=energy_breakdown,
        solve_config_fingerprint=solve_config.fingerprint(),
        source_fingerprint=source_fingerprint,
        metadata=metadata,
    )


__all__ = [
    "REAL_DTYPE",
    "INTEGER_DTYPE",
    "BOOLEAN_DTYPE",
    "DIMOD_MODULE_NAME",
    "DEFAULT_SOLVER_SEED",
    "DEFAULT_REQUIRE_FEASIBLE_BESS_SAMPLE",
    "DEFAULT_CLOSE_EXTERNAL_BUNDLE",
    "DEFAULT_CLOSE_INTERNAL_BUNDLE",
    "DWaveSolverError",
    "DWaveSolveConfig",
    "DWaveSolveResult",
    "solve_pegasus_qubo",
]
