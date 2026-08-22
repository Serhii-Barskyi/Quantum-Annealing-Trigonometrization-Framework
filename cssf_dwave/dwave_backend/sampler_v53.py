"""Versioned Ocean sampler factory for CSSF(QA) v53.

Repairs both known backend blockers while leaving v51 frozen modules untouched:
(1) local P16 programmable-fabric semantics; (2) D-Wave's 2026 solver-name
change from ``Advantage_system6.4``/``Advantage_system4.1`` to exact family IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from types import MappingProxyType
from typing import Any, Mapping

import dwave_backend.sampler as _legacy
from dwave_backend.pegasus_fabric_v53 import (
    topology_from_emulator_config_v53,
    topology_from_qpu_sampler_v53,
    validate_pegasus_solver_id_v53,
)

OceanSamplerError = _legacy.OceanSamplerError
OceanRuntimeUnavailableError = _legacy.OceanRuntimeUnavailableError
OceanSamplerContractError = _legacy.OceanSamplerContractError
OceanSamplerExecutionError = _legacy.OceanSamplerExecutionError
SamplerMode = _legacy.SamplerMode

LOCAL_GPU_BACKEND_KIND = "local_sqa_gpu"
PEGASUS_QPU_BACKEND_KIND = "pegasus_qpu"
DEFAULT_EMULATOR_MEMORY_FRACTION = _legacy.DEFAULT_EMULATOR_MEMORY_FRACTION
DEFAULT_QPU_LABEL = _legacy.DEFAULT_QPU_LABEL


def _json_mapping_v53(value: Mapping[str, Any]) -> Mapping[str, Any]:
    encoded = json.dumps(dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
    return MappingProxyType(json.loads(encoded))


@dataclass(frozen=True, slots=True)
class OceanSamplerBundleV53:
    """v53 sampler provenance container accepting current Pegasus solver IDs."""

    mode: str
    sampler: object
    raw_sampler: object
    topology: object
    solver_id: str | None
    dry_run: bool
    default_sample_parameters: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        mode = _legacy._normalized_mode(self.mode)
        _legacy.validate_sampler_interface(self.sampler)
        _legacy.validate_sampler_interface(self.raw_sampler)
        solver_id = self.solver_id
        if mode == PEGASUS_QPU_BACKEND_KIND:
            if solver_id is None:
                raise OceanSamplerContractError("QPU bundle requires an explicit solver_id.")
            solver_id = validate_pegasus_solver_id_v53(solver_id)
        elif solver_id is not None:
            raise OceanSamplerContractError("Local emulator bundle must not carry a QPU solver_id.")
        if not isinstance(self.dry_run, bool):
            raise TypeError("dry_run must be boolean.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "solver_id", solver_id)
        object.__setattr__(self, "default_sample_parameters", _json_mapping_v53(self.default_sample_parameters))
        object.__setattr__(self, "metadata", _json_mapping_v53(self.metadata))

    def fingerprint(self) -> str:
        payload = {
            "mode": self.mode,
            "solver_id": self.solver_id,
            "dry_run": self.dry_run,
            "topology_fingerprint": self.topology.fingerprint(),
            "default_sample_parameters": dict(self.default_sample_parameters),
            "metadata": dict(self.metadata),
        }
        return hashlib.sha256(
            b"CSSF-OceanSamplerBundle-v53\0" +
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()

    def sample_bqm(self, bqm: object, *, num_reads: int | None = None, label: str = DEFAULT_QPU_LABEL, **parameters: Any) -> object:
        if self.dry_run:
            raise OceanSamplerExecutionError("QPU dry_run is enabled; submission is blocked.")
        kwargs = dict(self.default_sample_parameters)
        kwargs.update(parameters)
        kwargs["label"] = _legacy._nonempty_string(label, name="label")
        if num_reads is not None:
            kwargs["num_reads"] = _legacy.validate_num_reads(num_reads)
        try:
            sampleset = self.sampler.sample(bqm, **kwargs)
        except Exception as exc:
            raise OceanSamplerExecutionError(f"{self.mode} sampling failed.") from exc
        info = getattr(sampleset, "info", None)
        if isinstance(info, dict):
            info.update({
                "cssf_bundle_fingerprint": self.fingerprint(),
                "cssf_backend": self.mode,
                "cssf_solver_id": self.solver_id,
                "cssf_topology_fingerprint": self.topology.fingerprint(),
            })
        return sampleset

    def close(self) -> None:
        _legacy._close_sampler_quietly(self.sampler)
        if self.raw_sampler is not self.sampler:
            _legacy._close_sampler_quietly(self.raw_sampler)

    def __enter__(self) -> "OceanSamplerBundleV53":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


# For the local path the frozen bundle is safe because solver_id=None.  QPU uses
# OceanSamplerBundleV53.  Consumers in the v53 experimental path are duck-typed.
OceanSamplerBundle = OceanSamplerBundleV53


def build_local_pegasus_gpu_sampler_v53(
    config: object,
    *,
    seed: int = 505,
    memory_fraction: float = DEFAULT_EMULATOR_MEMORY_FRACTION,
) -> OceanSamplerBundleV53:
    backend = _legacy._normalized_mode(getattr(config, "backend", None))
    if backend != LOCAL_GPU_BACKEND_KIND:
        raise OceanSamplerContractError("Local emulator requires backend='local_sqa_gpu'.")
    _legacy.validate_topology_type(getattr(config, "topology_type", None))
    if getattr(config, "require_gpu", None) is not True:
        raise OceanSamplerContractError("Local emulator must require GPU.")
    if getattr(config, "allow_classical_fallback", None) is not False:
        raise OceanSamplerContractError("Local emulator classical fallback is prohibited.")
    if getattr(config, "return_dimod_sampleset", None) is not True:
        raise OceanSamplerContractError("Local emulator must return dimod.SampleSet.")
    normalized_seed = _legacy._nonnegative_integer(seed, name="seed")
    normalized_memory_fraction = _legacy._memory_fraction(memory_fraction)

    topology = topology_from_emulator_config_v53(config)
    dimod = _legacy._load_dimod()
    torch = _legacy._load_torch_cuda()
    ocean_system = _legacy._load_ocean_system()
    embedding_composite = getattr(ocean_system, "EmbeddingComposite", None)
    if embedding_composite is None or not callable(embedding_composite):
        raise OceanRuntimeUnavailableError("dwave.system.EmbeddingComposite is unavailable.")
    raw_sampler = _legacy._build_structured_gpu_sampler(
        dimod=dimod,
        torch=torch,
        topology=topology,
        config=config,
        seed=normalized_seed,
        memory_fraction=normalized_memory_fraction,
    )
    try:
        sampler = embedding_composite(raw_sampler)
    except Exception as exc:
        raise OceanRuntimeUnavailableError("Ocean could not wrap GPU emulator with EmbeddingComposite.") from exc
    return OceanSamplerBundleV53(
        mode=LOCAL_GPU_BACKEND_KIND,
        sampler=sampler,
        raw_sampler=raw_sampler,
        topology=topology,
        solver_id=None,
        dry_run=False,
        default_sample_parameters={"num_reads": _legacy.validate_num_reads(getattr(config, "num_reads", None))},
        metadata={
            "provider": _legacy.LOCAL_EMULATOR_PROVIDER,
            "device": _legacy.LOCAL_EMULATOR_DEVICE,
            "pegasus_m": topology.pegasus_m,
            "classical_fallback": False,
            "seed": normalized_seed,
            "memory_fraction": normalized_memory_fraction,
            "topology_semantics": "synthetic_programmable_fabric_v53",
            "fabric_only": True,
        },
    )


def _sampler_solver_name_v53(sampler: object) -> str | None:
    solver = getattr(sampler, "solver", None)
    for attribute in ("name", "id"):
        value = getattr(solver, attribute, None) if solver is not None else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    properties = getattr(sampler, "properties", None)
    if isinstance(properties, Mapping):
        chip_id = properties.get("chip_id")
        if isinstance(chip_id, str) and chip_id.strip():
            return chip_id.strip()
    return None


def _validate_qpu_runtime_sampler_v53(sampler: object, *, expected_solver_id: str) -> None:
    _legacy.validate_sampler_interface(sampler)
    actual = _sampler_solver_name_v53(sampler)
    if actual is None:
        raise OceanSamplerContractError("Ocean QPU sampler exposes no concrete solver identity.")
    actual = validate_pegasus_solver_id_v53(actual)
    if actual != expected_solver_id:
        raise OceanSamplerContractError(
            f"Ocean selected a different solver: expected {expected_solver_id!r}, received {actual!r}."
        )
    solver = getattr(sampler, "solver", None)
    if getattr(solver, "qpu", None) is not True:
        raise OceanSamplerContractError("Selected Ocean solver is not QPU-category.")
    if getattr(solver, "online", None) is not True:
        raise OceanSamplerContractError("Selected Ocean QPU solver is not online.")


def build_leap_pegasus_qpu_sampler_v53(
    config: object,
    *,
    token: str | None = None,
    endpoint: str | None = None,
    region: str | None = None,
) -> OceanSamplerBundleV53:
    """Build one explicitly pinned current/historical System4/System6 QPU."""
    backend = str(getattr(getattr(config, "backend", None), "value", getattr(config, "backend", ""))).strip().lower()
    if backend != PEGASUS_QPU_BACKEND_KIND:
        raise OceanSamplerContractError("Leap QPU requires backend='pegasus_qpu'.")
    if str(getattr(config, "topology_type", "")).strip().lower() != "pegasus":
        raise OceanSamplerContractError("QPU topology must be Pegasus.")
    if getattr(config, "enabled", None) is not True:
        raise OceanSamplerContractError("QPU configuration must set enabled=True.")
    if getattr(config, "require_explicit_solver_id", None) is not True:
        raise OceanSamplerContractError("Explicit solver identity is mandatory.")
    if getattr(config, "allow_solver_fallback", None) is not False:
        raise OceanSamplerContractError("QPU solver fallback is prohibited.")
    if getattr(config, "reject_zephyr", None) is not True:
        raise OceanSamplerContractError("Zephyr rejection must remain enabled.")

    solver_id = validate_pegasus_solver_id_v53(getattr(config, "solver_id", None))
    dry_run = getattr(config, "dry_run", None)
    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be boolean.")
    num_reads = _legacy.validate_num_reads(getattr(config, "num_reads", None))
    annealing_time = float(getattr(config, "annealing_time", 0.0))
    if not annealing_time > 0.0:
        raise OceanSamplerContractError("annealing_time must be positive.")

    explicit_token = _legacy._optional_string(token, name="token")
    explicit_endpoint = _legacy._optional_string(endpoint, name="endpoint")
    explicit_region = _legacy._optional_string(region, name="region")
    ocean_system = _legacy._load_ocean_system()
    dwave_sampler_class = getattr(ocean_system, "DWaveSampler", None)
    embedding_composite = getattr(ocean_system, "EmbeddingComposite", None)
    if dwave_sampler_class is None or not callable(dwave_sampler_class):
        raise OceanRuntimeUnavailableError("dwave.system.DWaveSampler is unavailable.")
    if embedding_composite is None or not callable(embedding_composite):
        raise OceanRuntimeUnavailableError("dwave.system.EmbeddingComposite is unavailable.")

    kwargs: dict[str, Any] = {"solver": {"name": solver_id}}
    if explicit_token is not None:
        kwargs["token"] = explicit_token
    if explicit_endpoint is not None:
        kwargs["endpoint"] = explicit_endpoint
    if explicit_region is not None:
        kwargs["region"] = explicit_region

    raw_sampler: object | None = None
    try:
        raw_sampler = dwave_sampler_class(**kwargs)
        _validate_qpu_runtime_sampler_v53(raw_sampler, expected_solver_id=solver_id)
        topology = topology_from_qpu_sampler_v53(raw_sampler, solver_id=solver_id)
        sampler = embedding_composite(raw_sampler)
        _legacy.validate_sampler_interface(sampler)
    except Exception:
        _legacy._close_sampler_quietly(raw_sampler)
        raise

    token_source = (
        "explicit_argument" if explicit_token is not None else
        ("environment_or_ocean_config" if os.environ.get(_legacy.DWAVE_API_TOKEN_ENV) else "ocean_config_resolution")
    )
    return OceanSamplerBundleV53(
        mode=PEGASUS_QPU_BACKEND_KIND,
        sampler=sampler,
        raw_sampler=raw_sampler,
        topology=topology,
        solver_id=solver_id,
        dry_run=dry_run,
        default_sample_parameters={"num_reads": num_reads, "annealing_time": annealing_time},
        metadata={
            "provider": "dwave_leap_ocean",
            "solver_id": solver_id,
            "topology_type": "pegasus",
            "token_source": token_source,
            "token_stored": False,
            "annealing_time": annealing_time,
            "solver_fallback": False,
            "embedding": "EmbeddingComposite",
            "solver_name_contract": "current_exact_or_historical_minor_v53",
        },
    )


def build_ocean_sampler_v53(
    mode: SamplerMode | str,
    *,
    emulator_config: object | None = None,
    qpu_config: object | None = None,
    seed: int = 505,
    memory_fraction: float = DEFAULT_EMULATOR_MEMORY_FRACTION,
    token: str | None = None,
    endpoint: str | None = None,
    region: str | None = None,
) -> OceanSamplerBundleV53:
    normalized_mode = _legacy._normalized_mode(mode)
    if normalized_mode == LOCAL_GPU_BACKEND_KIND:
        if emulator_config is None or qpu_config is not None:
            raise OceanSamplerContractError("Local mode requires emulator_config only.")
        return build_local_pegasus_gpu_sampler_v53(emulator_config, seed=seed, memory_fraction=memory_fraction)
    if qpu_config is None or emulator_config is not None:
        raise OceanSamplerContractError("QPU mode requires qpu_config only.")
    return build_leap_pegasus_qpu_sampler_v53(qpu_config, token=token, endpoint=endpoint, region=region)


build_local_pegasus_gpu_sampler = build_local_pegasus_gpu_sampler_v53
build_leap_pegasus_qpu_sampler = build_leap_pegasus_qpu_sampler_v53
build_ocean_sampler = build_ocean_sampler_v53

__all__ = [
    "OceanSamplerBundleV53", "OceanSamplerBundle", "OceanSamplerError",
    "OceanRuntimeUnavailableError", "OceanSamplerContractError",
    "OceanSamplerExecutionError", "SamplerMode", "build_local_pegasus_gpu_sampler_v53",
    "build_leap_pegasus_qpu_sampler_v53", "build_ocean_sampler_v53",
    "build_local_pegasus_gpu_sampler", "build_leap_pegasus_qpu_sampler", "build_ocean_sampler",
]
