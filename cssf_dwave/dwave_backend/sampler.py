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

"""Unified Ocean sampler selection for CSSF Pegasus execution.

The module gives the Colab notebook one explicit runtime switch:

``local_sqa_gpu``
    A local Pegasus-constrained simulated-quantum-annealing emulator whose
    sampling kernel runs on a CUDA device through PyTorch. The public factory
    refuses to construct this backend when CUDA is unavailable and contains no
    CPU fallback.

``pegasus_qpu``
    A real D-Wave Leap QPU selected through Ocean's :class:`DWaveSampler` and
    wrapped by :class:`EmbeddingComposite`. The solver identifier must be
    explicit and must belong to ``Advantage_system4.*`` or
    ``Advantage_system6.*``. Zephyr, automatic solver selection, hybrid
    solvers, and silent fallback are rejected.

Both paths expose an Ocean-compatible ``dimod.Sampler`` boundary and return
``dimod.SampleSet`` objects. Optional runtimes are imported lazily, so merely
importing this module performs no CUDA initialization, credential lookup,
network request, solver selection, or QPU submission.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from dwave_backend import (
    LOCAL_GPU_BACKEND_KIND,
    PEGASUS_QPU_BACKEND_KIND,
    validate_backend_kind,
    validate_num_reads,
    validate_sampler_interface,
    validate_solver_id,
    validate_topology_type,
)
from dwave_backend.topology import (
    PegasusEdge,
    PegasusNode,
    PegasusTopology,
    topology_from_emulator_config,
    topology_from_qpu_config,
)


DWAVE_API_TOKEN_ENV: Final[str] = "DWAVE_API_TOKEN"
OCEAN_DIMOD_MODULE: Final[str] = "dimod"
OCEAN_SYSTEM_MODULE: Final[str] = "dwave.system"
TORCH_MODULE: Final[str] = "torch"

LOCAL_EMULATOR_PROVIDER: Final[str] = "cssf_torch_sqa_gpu"
LOCAL_EMULATOR_DEVICE: Final[str] = "cuda"
LOCAL_EMULATOR_CATEGORY: Final[str] = "software"
QPU_CATEGORY: Final[str] = "qpu"

DEFAULT_EMULATOR_BETA_RANGE: Final[tuple[float, float]] = (0.10, 5.00)
DEFAULT_EMULATOR_TRANSVERSE_FIELD_RANGE: Final[tuple[float, float]] = (
    4.00,
    1.0e-3,
)
DEFAULT_EMULATOR_MEMORY_FRACTION: Final[float] = 0.25
DEFAULT_QPU_LABEL: Final[str] = "CSSF BESS Pegasus"
NOTEBOOK_MODE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "emulator": LOCAL_GPU_BACKEND_KIND,
        "qpu": PEGASUS_QPU_BACKEND_KIND,
    }
)
MINIMUM_TROTTER_REPLICAS: Final[int] = 2
MAXIMUM_TROTTER_REPLICAS: Final[int] = 1024
MINIMUM_SQA_SWEEPS: Final[int] = 1
MAXIMUM_GPU_MEMORY_FRACTION: Final[float] = 0.80
SQA_STATE_BYTES_PER_SPIN_ESTIMATE: Final[int] = 64
SQA_EDGE_WORK_BYTES_PER_COUPLER_ESTIMATE: Final[int] = 12
SQA_OOM_BATCH_REDUCTION_FACTOR: Final[int] = 2

# Production incremental-field kernel constants. These do not alter the SQA
# transition; they only control memory planning and occasional numerical
# re-anchoring of the cached local fields.
SQA_LOCAL_FIELD_REBASE_INTERVAL: Final[int] = 256
SQA_INCREMENTAL_PERSISTENT_BYTES_PER_SPIN_ESTIMATE: Final[int] = 8
SQA_INCREMENTAL_SELECTED_BYTES_PER_SPIN_ESTIMATE: Final[int] = 36
SQA_INCREMENTAL_EDGE_BYTES_PER_COUPLER_ESTIMATE: Final[int] = 8

# CUDA production path: one int8 spin tensor plus bounded selected-color RNG
# buffers. The selected-color CSR kernel recomputes only the exact fields
# required by the current Metropolis proposals and needs no persistent local field.
SQA_FUSED_BYTES_PER_SPIN_ESTIMATE: Final[int] = 8
SQA_TRITON_MAX_BLOCK_REPLICAS: Final[int] = 512
SQA_TRITON_REQUIRED_ON_CUDA: Final[bool] = True


class OceanSamplerError(RuntimeError):
    """Base exception for sampler construction or execution failures."""


class OceanRuntimeUnavailableError(OceanSamplerError):
    """Raised when a required optional Ocean or CUDA runtime is unavailable."""


class OceanSamplerContractError(OceanSamplerError):
    """Raised when a sampler violates the strict CSSF backend contract."""


class OceanSamplerExecutionError(OceanSamplerError):
    """Raised when a validated sampler cannot complete a sampling request."""


class SamplerMode(str, Enum):
    """Notebook-visible runtime choices."""

    EMULATOR = LOCAL_GPU_BACKEND_KIND
    QPU = PEGASUS_QPU_BACKEND_KIND


def _enum_or_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _normalized_mode(value: object) -> str:
    candidate = _enum_or_value(value)
    if isinstance(candidate, str):
        candidate = NOTEBOOK_MODE_ALIASES.get(
            candidate.strip().lower(),
            candidate,
        )
    return validate_backend_kind(candidate)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise OceanSamplerContractError(f"{name} must be positive.")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise OceanSamplerContractError(f"{name} must be non-negative.")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar.") from exc
    if not math.isfinite(result):
        raise OceanSamplerContractError(f"{name} must be finite.")
    return result


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean.")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise OceanSamplerContractError(f"{name} must not be empty.")
    return normalized


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name=name)


def _json_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    source = {} if value is None else dict(value)
    try:
        encoded = json.dumps(
            source,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise OceanSamplerContractError(
            f"{name} must be JSON-serializable."
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _two_positive_floats(
    value: Sequence[object],
    *,
    name: str,
    descending: bool | None = None,
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise OceanSamplerContractError(
            f"{name} must contain exactly two values."
        )
    first = _finite_float(value[0], name=f"{name}[0]")
    second = _finite_float(value[1], name=f"{name}[1]")
    if first <= 0.0 or second <= 0.0:
        raise OceanSamplerContractError(f"{name} values must be positive.")
    if descending is True and first <= second:
        raise OceanSamplerContractError(
            f"{name} must be strictly descending."
        )
    if descending is False and first >= second:
        raise OceanSamplerContractError(
            f"{name} must be strictly ascending."
        )
    return (first, second)


def _memory_fraction(value: object) -> float:
    result = _finite_float(value, name="memory_fraction")
    if not 0.0 < result <= MAXIMUM_GPU_MEMORY_FRACTION:
        raise OceanSamplerContractError(
            "memory_fraction must be in (0, "
            f"{MAXIMUM_GPU_MEMORY_FRACTION}]."
        )
    return result


def _load_module(module_name: str, *, purpose: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise OceanRuntimeUnavailableError(
            f"{module_name} is required for {purpose}. Install "
            "requirements-colab.txt in the Colab runtime."
        ) from exc


def _load_dimod() -> Any:
    return _load_module(
        OCEAN_DIMOD_MODULE,
        purpose="the Ocean-compatible SampleSet boundary",
    )


def _load_ocean_system() -> Any:
    return _load_module(
        OCEAN_SYSTEM_MODULE,
        purpose="D-Wave Leap QPU access and minor embedding",
    )


def _load_torch_cuda() -> Any:
    torch = _load_module(
        TORCH_MODULE,
        purpose="the local Pegasus GPU emulator",
    )
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        raise OceanRuntimeUnavailableError(
            "PyTorch does not expose a CUDA runtime."
        )
    if not bool(cuda.is_available()):
        raise OceanRuntimeUnavailableError(
            "CUDA is unavailable. The local Pegasus emulator refuses CPU "
            "fallback; select a Colab GPU runtime or choose pegasus_qpu."
        )
    return torch


def _sampler_solver_name(sampler: object) -> str | None:
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


def _validate_qpu_runtime_sampler(
    sampler: object,
    *,
    expected_solver_id: str,
) -> object:
    validate_sampler_interface(sampler)
    actual_solver_id = _sampler_solver_name(sampler)
    if actual_solver_id is None:
        raise OceanSamplerContractError(
            "Ocean QPU sampler does not expose its concrete solver identity."
        )
    actual_solver_id = validate_solver_id(actual_solver_id)
    if actual_solver_id != expected_solver_id:
        raise OceanSamplerContractError(
            "Ocean selected a different solver than requested: "
            f"expected {expected_solver_id!r}, received {actual_solver_id!r}."
        )

    solver = getattr(sampler, "solver", None)
    qpu_flag = getattr(solver, "qpu", None) if solver is not None else None
    online_flag = getattr(solver, "online", None) if solver is not None else None
    if qpu_flag is not True:
        raise OceanSamplerContractError(
            "Selected Ocean solver is not a QPU-category solver."
        )
    if online_flag is not True:
        raise OceanSamplerContractError(
            "Selected Ocean QPU solver is not online."
        )

    properties = getattr(sampler, "properties", None)
    if not isinstance(properties, Mapping):
        raise OceanSamplerContractError(
            "QPU sampler properties must be a mapping."
        )
    category = properties.get("category")
    if category is not None and str(category).strip().lower() != QPU_CATEGORY:
        raise OceanSamplerContractError(
            f"QPU sampler category must be {QPU_CATEGORY!r}."
        )
    topology = properties.get("topology")
    if not isinstance(topology, Mapping):
        raise OceanSamplerContractError(
            "QPU sampler properties must expose topology metadata."
        )
    validate_topology_type(topology.get("type"))
    chip_id = properties.get("chip_id")
    if chip_id is not None and str(chip_id).strip() != expected_solver_id:
        raise OceanSamplerContractError(
            "QPU chip_id differs from the explicitly requested solver_id."
        )
    return sampler


def _close_sampler_quietly(sampler: object | None) -> None:
    if sampler is None:
        return
    close = getattr(sampler, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            return


def _node_sort_key(node: PegasusNode) -> tuple[int, tuple[int, ...]]:
    if isinstance(node, int):
        return (0, (node,))
    return (1 if len(node) == 4 else 2, tuple(node))


def _canonical_edge_set(
    edges: Iterable[PegasusEdge],
) -> frozenset[frozenset[PegasusNode]]:
    return frozenset(frozenset(edge) for edge in edges)


def _active_graph_coloring(
    node_count: int,
    edge_indices: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    """Greedily color an active Ising graph for conflict-free GPU updates."""

    normalized_node_count = _positive_integer(
        node_count,
        name="node_count",
    )
    adjacency: list[set[int]] = [set() for _ in range(normalized_node_count)]
    for position, edge in enumerate(edge_indices):
        if len(edge) != 2:
            raise OceanSamplerContractError(
                f"edge_indices[{position}] must contain two indices."
            )
        first = _nonnegative_integer(edge[0], name=f"edge[{position}][0]")
        second = _nonnegative_integer(edge[1], name=f"edge[{position}][1]")
        if first >= normalized_node_count or second >= normalized_node_count:
            raise OceanSamplerContractError(
                "Edge index exceeds the active node count."
            )
        if first == second:
            raise OceanSamplerContractError("Self-loops are prohibited.")
        adjacency[first].add(second)
        adjacency[second].add(first)

    order = sorted(
        range(normalized_node_count),
        key=lambda item: (-len(adjacency[item]), item),
    )
    assigned = [-1] * normalized_node_count
    for node in order:
        unavailable = {
            assigned[neighbour]
            for neighbour in adjacency[node]
            if assigned[neighbour] >= 0
        }
        color = 0
        while color in unavailable:
            color += 1
        assigned[node] = color

    color_count = max(assigned) + 1
    groups = tuple(
        tuple(index for index, color in enumerate(assigned) if color == group)
        for group in range(color_count)
    )
    if any(not group for group in groups):
        raise OceanSamplerContractError("Graph coloring produced an empty group.")
    for first, second in edge_indices:
        if assigned[first] == assigned[second]:
            raise OceanSamplerContractError(
                "Graph coloring failed to separate adjacent variables."
            )
    return groups


def _geometric_schedule(
    start: float,
    stop: float,
    count: int,
) -> NDArray[np.float64]:
    normalized_start = _finite_float(start, name="schedule start")
    normalized_stop = _finite_float(stop, name="schedule stop")
    normalized_count = _positive_integer(count, name="schedule count")
    if normalized_start <= 0.0 or normalized_stop <= 0.0:
        raise OceanSamplerContractError(
            "Geometric schedule endpoints must be positive."
        )
    schedule = np.geomspace(
        normalized_start,
        normalized_stop,
        num=normalized_count,
        dtype=np.float64,
    )
    schedule.setflags(write=False)
    return schedule


def _estimate_gpu_batch_size(
    *,
    free_bytes: int,
    num_reads: int,
    replicas: int,
    variable_count: int,
    memory_fraction: float,
    edge_count: int = 0,
) -> int:
    """Estimate a conservative CUDA micro-batch size for the SQA kernel.

    The kernel allocates substantially more than the persistent int8 spin state:
    local fields are float32, edge-gather workspaces scale with the active
    coupler count, and each color/parity update creates additional temporary
    float32 tensors.  The estimate therefore accounts for both active variables
    and active couplers and deliberately leaves the remainder of free CUDA
    memory available for allocator fragmentation and other notebook workloads.
    """

    normalized_free = _positive_integer(free_bytes, name="free_bytes")
    normalized_reads = validate_num_reads(num_reads)
    normalized_replicas = _positive_integer(replicas, name="replicas")
    normalized_variables = _positive_integer(
        variable_count,
        name="variable_count",
    )
    normalized_edges = _nonnegative_integer(edge_count, name="edge_count")
    normalized_fraction = _memory_fraction(memory_fraction)

    bytes_per_replica = (
        normalized_variables * SQA_STATE_BYTES_PER_SPIN_ESTIMATE
        + normalized_edges * SQA_EDGE_WORK_BYTES_PER_COUPLER_ESTIMATE
    )
    bytes_per_read = normalized_replicas * bytes_per_replica
    capacity = int(
        (normalized_free * normalized_fraction) // max(1, bytes_per_read)
    )
    return max(1, min(normalized_reads, capacity))


def _estimate_incremental_gpu_batch_size(
    *,
    free_bytes: int,
    num_reads: int,
    replicas: int,
    variable_count: int,
    memory_fraction: float,
    max_color_size: int,
    max_color_directed_edges: int,
) -> int:
    """Estimate a CUDA micro-batch for the persistent-local-field kernel.

    Unlike the reference kernel, the production kernel never materializes a
    full edge workspace for every color/parity. The dominant allocations are
    the persistent spin/local-field tensors and selected-color workspaces.
    Adaptive OOM retry remains the final safety net and never reduces reads.
    """

    normalized_free = _positive_integer(free_bytes, name="free_bytes")
    normalized_reads = validate_num_reads(num_reads)
    normalized_replicas = _positive_integer(replicas, name="replicas")
    normalized_variables = _positive_integer(variable_count, name="variable_count")
    normalized_fraction = _memory_fraction(memory_fraction)
    normalized_color_size = _positive_integer(max_color_size, name="max_color_size")
    normalized_color_edges = _nonnegative_integer(
        max_color_directed_edges,
        name="max_color_directed_edges",
    )

    parity_replicas = (normalized_replicas + 1) // 2
    persistent_bytes = (
        normalized_replicas
        * normalized_variables
        * SQA_INCREMENTAL_PERSISTENT_BYTES_PER_SPIN_ESTIMATE
    )
    selected_bytes = (
        parity_replicas
        * normalized_color_size
        * SQA_INCREMENTAL_SELECTED_BYTES_PER_SPIN_ESTIMATE
    )
    incremental_edge_bytes = (
        parity_replicas
        * normalized_color_edges
        * SQA_INCREMENTAL_EDGE_BYTES_PER_COUPLER_ESTIMATE
    )
    bytes_per_read = max(1, persistent_bytes + selected_bytes + incremental_edge_bytes)
    capacity = int((normalized_free * normalized_fraction) // bytes_per_read)
    return max(1, min(normalized_reads, capacity))


def _estimate_fused_gpu_batch_size(
    *,
    free_bytes: int,
    num_reads: int,
    replicas: int,
    variable_count: int,
    memory_fraction: float,
) -> int:
    """Estimate a micro-batch for the fused selected-field CUDA kernel.

    The fused path stores spins as int8 in [B,N,R] layout plus two temporary
    float32 RNG tensors for the current color. No B*R*E edge workspace and no
    persistent float32 local-field tensor are allocated.
    """

    normalized_free = _positive_integer(free_bytes, name="free_bytes")
    normalized_reads = validate_num_reads(num_reads)
    normalized_replicas = _positive_integer(replicas, name="replicas")
    normalized_variables = _positive_integer(variable_count, name="variable_count")
    normalized_fraction = _memory_fraction(memory_fraction)
    bytes_per_read = max(
        1,
        normalized_replicas
        * normalized_variables
        * SQA_FUSED_BYTES_PER_SPIN_ESTIMATE,
    )
    capacity = int((normalized_free * normalized_fraction) // bytes_per_read)
    return max(1, min(normalized_reads, capacity))


@dataclass(frozen=True, slots=True)
class SQASettings:
    """Validated local GPU simulated-quantum-annealing settings."""

    num_reads: int
    trotter_replicas: int
    sweeps: int
    burn_in_sweeps: int
    beta_range: tuple[float, float]
    transverse_field_range: tuple[float, float]
    seed: int
    memory_fraction: float

    def __post_init__(self) -> None:
        num_reads = validate_num_reads(self.num_reads)
        replicas = _positive_integer(
            self.trotter_replicas,
            name="trotter_replicas",
        )
        if not MINIMUM_TROTTER_REPLICAS <= replicas <= MAXIMUM_TROTTER_REPLICAS:
            raise OceanSamplerContractError(
                "trotter_replicas must be in the inclusive range "
                f"[{MINIMUM_TROTTER_REPLICAS}, {MAXIMUM_TROTTER_REPLICAS}]."
            )
        if replicas % 2:
            raise OceanSamplerContractError(
                "trotter_replicas must be even for conflict-free ring updates."
            )
        sweeps = _positive_integer(self.sweeps, name="sweeps")
        burn_in = _nonnegative_integer(
            self.burn_in_sweeps,
            name="burn_in_sweeps",
        )
        if burn_in >= sweeps:
            raise OceanSamplerContractError(
                "burn_in_sweeps must be smaller than sweeps."
            )
        beta_range = _two_positive_floats(
            self.beta_range,
            name="beta_range",
            descending=False,
        )
        field_range = _two_positive_floats(
            self.transverse_field_range,
            name="transverse_field_range",
            descending=True,
        )
        seed = _nonnegative_integer(self.seed, name="seed")
        if seed > 2**63 - 1:
            raise OceanSamplerContractError(
                "seed must not exceed 2**63 - 1."
            )
        memory_fraction = _memory_fraction(self.memory_fraction)

        object.__setattr__(self, "num_reads", num_reads)
        object.__setattr__(self, "trotter_replicas", replicas)
        object.__setattr__(self, "sweeps", sweeps)
        object.__setattr__(self, "burn_in_sweeps", burn_in)
        object.__setattr__(self, "beta_range", beta_range)
        object.__setattr__(self, "transverse_field_range", field_range)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "memory_fraction", memory_fraction)

    def fingerprint(self) -> str:
        payload = {
            "num_reads": self.num_reads,
            "trotter_replicas": self.trotter_replicas,
            "sweeps": self.sweeps,
            "burn_in_sweeps": self.burn_in_sweeps,
            "beta_range": self.beta_range,
            "transverse_field_range": self.transverse_field_range,
            "seed": self.seed,
            "memory_fraction": self.memory_fraction,
        }
        digest = hashlib.sha256()
        digest.update(b"CSSF-SQASettings-v1\0")
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


def _problem_local_fields_torch(
    torch: Any,
    state: Any,
    h_tensor: Any,
    edge_u: Any,
    edge_v: Any,
    edge_j: Any,
) -> Any:
    batch, replicas, variable_count = state.shape
    flat_state = state.reshape(batch * replicas, variable_count).to(
        dtype=torch.float32
    )
    local = h_tensor.reshape(1, variable_count).expand(
        batch * replicas,
        variable_count,
    ).clone()

    if int(edge_j.numel()) > 0:
        edge_weights = edge_j.reshape(1, -1)

        # Build one edge workspace at a time.  Keeping source_u and source_v
        # alive simultaneously can double the largest edge-scaled allocation.
        source = flat_state.index_select(1, edge_v)
        source.mul_(edge_weights)
        local.index_add_(1, edge_u, source)
        del source

        source = flat_state.index_select(1, edge_u)
        source.mul_(edge_weights)
        local.index_add_(1, edge_v, source)
        del source

    return local.reshape(batch, replicas, variable_count)


def _single_sqa_sweep_torch_reference(
    *,
    torch: Any,
    state: Any,
    h_tensor: Any,
    edge_u: Any,
    edge_v: Any,
    edge_j: Any,
    color_tensors: Sequence[Any],
    beta: float,
    transverse_field: float,
    generator: Any,
) -> None:
    replicas = int(state.shape[1])
    beta_value = _finite_float(beta, name="beta")
    field_value = _finite_float(
        transverse_field,
        name="transverse_field",
    )
    scaled_field = beta_value * field_value / replicas
    scaled_field = max(scaled_field, np.finfo(np.float64).tiny)
    trotter_coupling = 0.5 * math.log(1.0 / math.tanh(scaled_field))

    for color_indices in color_tensors:
        for parity in (0, 1):
            local = _problem_local_fields_torch(
                torch,
                state,
                h_tensor,
                edge_u,
                edge_v,
                edge_j,
            )

            replica_indices = torch.arange(
                parity,
                replicas,
                2,
                dtype=torch.long,
                device=state.device,
            )
            previous_indices = torch.remainder(replica_indices - 1, replicas)
            following_indices = torch.remainder(replica_indices + 1, replicas)

            state_view = state.index_select(1, replica_indices)
            local_selected = local.index_select(1, replica_indices).index_select(
                2,
                color_indices,
            )
            current = state_view.index_select(2, color_indices).to(
                dtype=torch.float32
            )
            previous_selected = (
                state.index_select(1, previous_indices)
                .index_select(2, color_indices)
                .to(dtype=torch.float32)
            )
            following_selected = (
                state.index_select(1, following_indices)
                .index_select(2, color_indices)
                .to(dtype=torch.float32)
            )

            # Reuse temporary buffers aggressively to keep CUDA peak memory
            # bounded.  This is algebraically identical to the previous
            # Metropolis log-ratio expression.
            log_ratio = current * local_selected
            log_ratio.mul_(2.0 * beta_value / replicas)

            previous_selected.add_(following_selected)
            previous_selected.mul_(current)
            previous_selected.mul_(-2.0 * trotter_coupling)
            log_ratio.add_(previous_selected)
            log_ratio.clamp_max_(0.0)

            random_values = torch.rand(
                log_ratio.shape,
                dtype=torch.float32,
                device=state.device,
                generator=generator,
            )
            random_values.clamp_min_(1.0e-30).log_()
            accepted = random_values < log_ratio
            updated = torch.where(accepted, -current, current).to(
                dtype=state.dtype
            )

            # index_select returns a copy, so write the accepted color update
            # back through the original parity view.
            parity_view = state[:, parity::2, :]
            parity_view[:, :, color_indices] = updated


def _single_sqa_sweep_torch(
    *,
    torch: Any,
    state: Any,
    h_tensor: Any,
    edge_u: Any,
    edge_v: Any,
    edge_j: Any,
    color_tensors: Sequence[Any],
    beta: float,
    transverse_field: float,
    generator: Any,
) -> None:
    """Compatibility/reference sweep preserving the original implementation.

    Production execution uses ``_single_sqa_sweep_incremental_torch`` below.
    Keeping the original entry point makes transition regression tests possible.
    """
    _single_sqa_sweep_torch_reference(
        torch=torch,
        state=state,
        h_tensor=h_tensor,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_j=edge_j,
        color_tensors=color_tensors,
        beta=beta,
        transverse_field=transverse_field,
        generator=generator,
    )


def _incremental_color_plan_arrays(
    *,
    variable_count: int,
    edge_u: NDArray[np.int64],
    edge_v: NDArray[np.int64],
    edge_j: NDArray[np.float64],
    color_groups: Sequence[Sequence[int]],
) -> tuple[
    tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]],
    ...,
]:
    """Build directed per-color neighbour updates for cached local fields.

    Every source spin contributes one directed record for each incident Ising
    coupler. Summed over colors there are exactly two records per undirected edge.
    """
    normalized_variables = _positive_integer(variable_count, name="variable_count")
    u_array = np.asarray(edge_u, dtype=np.int64)
    v_array = np.asarray(edge_v, dtype=np.int64)
    j_array = np.asarray(edge_j, dtype=np.float64)
    if not (u_array.ndim == v_array.ndim == j_array.ndim == 1):
        raise OceanSamplerContractError("Edge arrays must be one-dimensional.")
    if not (u_array.size == v_array.size == j_array.size):
        raise OceanSamplerContractError("Edge arrays must have equal length.")

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(normalized_variables)]
    for first, second, weight in zip(
        u_array.tolist(), v_array.tolist(), j_array.tolist(), strict=True
    ):
        if first == second:
            raise OceanSamplerContractError("Self-loops are prohibited.")
        if not (0 <= first < normalized_variables and 0 <= second < normalized_variables):
            raise OceanSamplerContractError("Edge index exceeds active variable count.")
        adjacency[first].append((second, float(weight)))
        adjacency[second].append((first, float(weight)))

    seen = np.zeros(normalized_variables, dtype=np.int8)
    plans: list[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]
    ] = []
    directed_records = 0

    for group in color_groups:
        color = np.asarray(tuple(group), dtype=np.int64)
        if color.ndim != 1 or color.size == 0:
            raise OceanSamplerContractError(
                "color_groups must contain non-empty one-dimensional groups."
            )
        if int(color.min()) < 0 or int(color.max()) >= normalized_variables:
            raise OceanSamplerContractError("Color index exceeds active variable count.")
        if np.unique(color).size != color.size:
            raise OceanSamplerContractError("A color group contains duplicate variables.")
        if np.any(seen[color] != 0):
            raise OceanSamplerContractError("Color groups overlap.")
        seen[color] = 1

        source_positions: list[int] = []
        destinations: list[int] = []
        weights: list[float] = []
        color_members = set(int(item) for item in color.tolist())
        for source_position, source in enumerate(color.tolist()):
            for destination, weight in adjacency[source]:
                if destination in color_members:
                    raise OceanSamplerContractError(
                        "Incremental plan found an edge inside one color group."
                    )
                source_positions.append(source_position)
                destinations.append(destination)
                weights.append(weight)

        source_array = np.asarray(source_positions, dtype=np.int64)
        destination_array = np.asarray(destinations, dtype=np.int64)
        weight_array = np.asarray(weights, dtype=np.float32)
        directed_records += int(source_array.size)
        plans.append((color, source_array, destination_array, weight_array))

    if not np.all(seen == 1):
        raise OceanSamplerContractError("Color groups do not partition all active variables.")
    if directed_records != 2 * int(j_array.size):
        raise OceanSamplerContractError(
            "Incremental color plan must contain exactly two directed records per edge."
        )
    return tuple(plans)


def _single_sqa_sweep_incremental_torch(
    *,
    torch: Any,
    state: Any,
    local: Any,
    color_plans: Sequence[tuple[Any, Any, Any, Any]],
    parity_plans: Sequence[tuple[Any, Any, Any]],
    beta: float,
    transverse_field: float,
    generator: Any,
) -> None:
    """Execute one mathematically identical SQA sweep using cached fields.

    The Metropolis log-ratio is unchanged. After an accepted problem-spin flip,
    only neighbouring cached fields are updated: delta_local_j = J_ij * delta_s_i.
    """
    replicas = int(state.shape[1])
    beta_value = _finite_float(beta, name="beta")
    field_value = _finite_float(transverse_field, name="transverse_field")
    scaled_field = beta_value * field_value / replicas
    scaled_field = max(scaled_field, np.finfo(np.float64).tiny)
    trotter_coupling = 0.5 * math.log(1.0 / math.tanh(scaled_field))

    for color_indices, edge_source_positions, edge_destinations, edge_weights in color_plans:
        for parity, (_, previous_indices, following_indices) in enumerate(parity_plans):
            # Re-gather after parity 0 because parity 1 must see the accepted
            # parity-0 spins through the Trotter neighbours, exactly as reference.
            # This still copies only B x R x |color|, never B x R x N.
            color_state = state.index_select(2, color_indices)
            local_selected = local[:, parity::2, :].index_select(2, color_indices)
            current = color_state[:, parity::2, :].to(dtype=torch.float32)
            previous_selected = color_state.index_select(1, previous_indices).to(
                dtype=torch.float32
            )
            following_selected = color_state.index_select(1, following_indices).to(
                dtype=torch.float32
            )

            # Preserve the reference arithmetic order for the transition probability.
            log_ratio = current * local_selected
            log_ratio.mul_(2.0 * beta_value / replicas)
            previous_selected.add_(following_selected)
            previous_selected.mul_(current)
            previous_selected.mul_(-2.0 * trotter_coupling)
            log_ratio.add_(previous_selected)
            log_ratio.clamp_max_(0.0)

            random_values = torch.rand(
                log_ratio.shape,
                dtype=torch.float32,
                device=state.device,
                generator=generator,
            )
            random_values.clamp_min_(1.0e-30).log_()
            accepted = random_values < log_ratio
            updated = torch.where(accepted, -current, current).to(dtype=state.dtype)

            state_parity = state[:, parity::2, :]
            state_parity[:, :, color_indices] = updated

            if int(edge_source_positions.numel()) > 0:
                delta_spin = accepted.to(dtype=torch.float32)
                delta_spin.mul_(current)
                delta_spin.mul_(-2.0)
                edge_delta = delta_spin.index_select(2, edge_source_positions)
                edge_delta.mul_(edge_weights.reshape(1, 1, -1))

                local_parity = local[:, parity::2, :]
                scatter_index = edge_destinations.reshape(1, 1, -1).expand(
                    edge_delta.shape[0], edge_delta.shape[1], -1
                )
                local_parity.scatter_add_(2, scatter_index, edge_delta)


def _reference_order_csr_arrays(
    *,
    variable_count: int,
    edge_u: NDArray[np.int64],
    edge_v: NDArray[np.int64],
    edge_j: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32], int]:
    """Build CSR adjacency in the same per-node accumulation order as reference.

    ``_problem_local_fields_torch`` first accumulates all u<-v contributions
    in edge order and then all v<-u contributions in edge order.  Keeping that
    ordering reduces avoidable floating-point differences in the fused path.
    """

    normalized_variables = _positive_integer(variable_count, name="variable_count")
    u_array = np.asarray(edge_u, dtype=np.int64)
    v_array = np.asarray(edge_v, dtype=np.int64)
    j_array = np.asarray(edge_j, dtype=np.float64)
    if not (u_array.ndim == v_array.ndim == j_array.ndim == 1):
        raise OceanSamplerContractError("Edge arrays must be one-dimensional.")
    if not (u_array.size == v_array.size == j_array.size):
        raise OceanSamplerContractError("Edge arrays must have equal length.")

    first_pass: list[list[tuple[int, float]]] = [
        [] for _ in range(normalized_variables)
    ]
    second_pass: list[list[tuple[int, float]]] = [
        [] for _ in range(normalized_variables)
    ]
    for first, second, weight in zip(
        u_array.tolist(), v_array.tolist(), j_array.tolist(), strict=True
    ):
        if first == second:
            raise OceanSamplerContractError("Self-loops are prohibited.")
        if not (0 <= first < normalized_variables and 0 <= second < normalized_variables):
            raise OceanSamplerContractError("Edge index exceeds active variable count.")
        first_pass[first].append((second, float(weight)))
        second_pass[second].append((first, float(weight)))

    row_ptr = np.zeros(normalized_variables + 1, dtype=np.int64)
    neighbours: list[int] = []
    weights: list[float] = []
    max_degree = 0
    for variable in range(normalized_variables):
        records = first_pass[variable] + second_pass[variable]
        max_degree = max(max_degree, len(records))
        for neighbour, weight in records:
            neighbours.append(neighbour)
            weights.append(weight)
        row_ptr[variable + 1] = len(neighbours)

    if len(neighbours) != 2 * int(j_array.size):
        raise OceanSamplerContractError(
            "CSR adjacency must contain exactly two directed records per edge."
        )
    return (
        row_ptr,
        np.asarray(neighbours, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        max_degree,
    )


_TRITON_KERNEL_CACHE: tuple[Any, Any, str] | None = None


def _load_triton_selected_field_kernel() -> tuple[Any, Any, str]:
    """Lazily build the fused Triton selected-field SQA color kernel."""

    global _TRITON_KERNEL_CACHE
    if _TRITON_KERNEL_CACHE is not None:
        return _TRITON_KERNEL_CACHE

    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
    except ImportError as exc:
        raise OceanRuntimeUnavailableError(
            "The fused CUDA SQA kernel requires Triton. The selected CUDA "
            "runtime has no Triton installation; refusing the slow eager-CUDA "
            "fallback in production."
        ) from exc

    @triton.jit
    def _selected_field_color_kernel(
        state_ptr,
        random_even_ptr,
        random_odd_ptr,
        h_ptr,
        csr_row_ptr,
        csr_col_ptr,
        csr_weight_ptr,
        color_ptr,
        beta_scale,
        trotter_coupling,
        batch_size: tl.constexpr,
        replicas: tl.constexpr,
        variable_count: tl.constexpr,
        color_size: tl.constexpr,
        max_degree: tl.constexpr,
        block_replicas: tl.constexpr,
    ):
        # One Triton program owns one (read, variable) pair and vectorizes the
        # complete Trotter ring. State layout [B,N,R] makes replica loads contiguous.
        pid = tl.program_id(0)
        read_index = pid // color_size
        color_position = pid - read_index * color_size
        variable = tl.load(color_ptr + color_position).to(tl.int64)

        half_replicas: tl.constexpr = replicas // 2
        lane = tl.arange(0, block_replicas)
        active = lane < half_replicas
        even_replica = 2 * lane
        odd_replica = even_replica + 1

        state_base = (read_index * variable_count + variable) * replicas
        even_ptr = state_ptr + state_base + even_replica
        odd_ptr = state_ptr + state_base + odd_replica
        random_index = (read_index * half_replicas + lane) * color_size + color_position

        current_even = tl.load(even_ptr, mask=active, other=1).to(tl.float32)
        current_odd = tl.load(odd_ptr, mask=active, other=1).to(tl.float32)
        local_even = tl.full([block_replicas], 0.0, tl.float32)
        local_odd = tl.full([block_replicas], 0.0, tl.float32)
        h_value = tl.load(h_ptr + variable).to(tl.float32)
        local_even += h_value
        local_odd += h_value

        row_start = tl.load(csr_row_ptr + variable).to(tl.int64)
        row_stop = tl.load(csr_row_ptr + variable + 1).to(tl.int64)
        for degree_offset in tl.static_range(0, max_degree):
            edge_position = row_start + degree_offset
            edge_valid = edge_position < row_stop
            neighbour = tl.load(
                csr_col_ptr + edge_position, mask=edge_valid, other=0
            ).to(tl.int64)
            weight = tl.load(
                csr_weight_ptr + edge_position, mask=edge_valid, other=0.0
            ).to(tl.float32)
            neighbour_base = (read_index * variable_count + neighbour) * replicas
            neighbour_even = tl.load(
                state_ptr + neighbour_base + even_replica,
                mask=active & edge_valid,
                other=0,
            ).to(tl.float32)
            neighbour_odd = tl.load(
                state_ptr + neighbour_base + odd_replica,
                mask=active & edge_valid,
                other=0,
            ).to(tl.float32)
            local_even += weight * neighbour_even
            local_odd += weight * neighbour_odd

        # Reference order: parity 0 first. Even Trotter neighbours are odd spins
        # and therefore have not yet been modified in this color update.
        previous_even_replica = (even_replica + replicas - 1) % replicas
        following_even_replica = (even_replica + 1) % replicas
        previous_even = tl.load(
            state_ptr + state_base + previous_even_replica, mask=active, other=1
        ).to(tl.float32)
        following_even = tl.load(
            state_ptr + state_base + following_even_replica, mask=active, other=1
        ).to(tl.float32)
        log_ratio_even = current_even * local_even * beta_scale
        log_ratio_even += (
            -2.0
            * trotter_coupling
            * current_even
            * (previous_even + following_even)
        )
        log_ratio_even = tl.minimum(log_ratio_even, 0.0)
        uniform_even = tl.load(
            random_even_ptr + random_index, mask=active, other=0.5
        ).to(tl.float32)
        uniform_even = tl.maximum(uniform_even, 1.0e-30)
        accept_even = tl.log(uniform_even) < log_ratio_even
        updated_even = tl.where(accept_even, -current_even, current_even)
        tl.store(even_ptr, updated_even.to(tl.int8), mask=active)
        # Odd lanes read even spins written by neighbouring lanes (k and k+1).
        # A block barrier makes the reference parity-0 -> parity-1 ordering explicit.
        tl.debug_barrier()

        # Parity 1 must observe the just-accepted parity-0 spins of the same
        # variable. The store/load sequence is intra-program and no other
        # program writes this variable because colors contain no adjacent nodes.
        previous_odd_replica = (odd_replica + replicas - 1) % replicas
        following_odd_replica = (odd_replica + 1) % replicas
        previous_odd = tl.load(
            state_ptr + state_base + previous_odd_replica, mask=active, other=1
        ).to(tl.float32)
        following_odd = tl.load(
            state_ptr + state_base + following_odd_replica, mask=active, other=1
        ).to(tl.float32)
        log_ratio_odd = current_odd * local_odd * beta_scale
        log_ratio_odd += (
            -2.0
            * trotter_coupling
            * current_odd
            * (previous_odd + following_odd)
        )
        log_ratio_odd = tl.minimum(log_ratio_odd, 0.0)
        uniform_odd = tl.load(
            random_odd_ptr + random_index, mask=active, other=0.5
        ).to(tl.float32)
        uniform_odd = tl.maximum(uniform_odd, 1.0e-30)
        accept_odd = tl.log(uniform_odd) < log_ratio_odd
        updated_odd = tl.where(accept_odd, -current_odd, current_odd)
        tl.store(odd_ptr, updated_odd.to(tl.int8), mask=active)

    version = str(getattr(triton, "__version__", "unknown"))
    _TRITON_KERNEL_CACHE = (triton, _selected_field_color_kernel, version)
    return _TRITON_KERNEL_CACHE


def _run_sqa_triton_fused(
    *,
    torch: Any,
    h: NDArray[np.float64],
    edge_u: NDArray[np.int64],
    edge_v: NDArray[np.int64],
    edge_j: NDArray[np.float64],
    color_groups: Sequence[Sequence[int]],
    settings: SQASettings,
    device: str,
    batch_size: int,
    diagnostics: dict[str, Any] | None = None,
) -> NDArray[np.int8]:
    """Run the exact color/parity Metropolis SQA transition with a fused kernel.

    No local-field approximation is used. For every proposed variable, the
    kernel evaluates h_i + sum_j J_ij s_j directly from the current state using
    bounded-degree CSR adjacency. The color/parity update order is unchanged.
    """

    h_array = np.asarray(h, dtype=np.float64)
    u_array = np.asarray(edge_u, dtype=np.int64)
    v_array = np.asarray(edge_v, dtype=np.int64)
    j_array = np.asarray(edge_j, dtype=np.float64)
    if h_array.ndim != 1 or h_array.size == 0:
        raise OceanSamplerContractError("h must be a non-empty vector.")
    if not (u_array.ndim == v_array.ndim == j_array.ndim == 1):
        raise OceanSamplerContractError("Edge arrays must be one-dimensional.")
    if not (u_array.size == v_array.size == j_array.size):
        raise OceanSamplerContractError("Edge arrays must have equal length.")
    if not np.all(np.isfinite(h_array)) or not np.all(np.isfinite(j_array)):
        raise OceanSamplerContractError("Ising coefficients must be finite.")

    normalized_batch = _positive_integer(batch_size, name="batch_size")
    normalized_device = _nonempty_string(device, name="device")
    if not normalized_device.startswith("cuda"):
        raise OceanSamplerContractError(
            "The fused Triton runner is CUDA-only."
        )
    if settings.trotter_replicas % 2:
        raise OceanSamplerContractError(
            "Fused color/parity kernel requires an even Trotter replica count."
        )

    triton, color_kernel, triton_version = _load_triton_selected_field_kernel()
    half_replicas = settings.trotter_replicas // 2
    block_replicas = 1 << max(0, half_replicas - 1).bit_length()
    if block_replicas > SQA_TRITON_MAX_BLOCK_REPLICAS:
        raise OceanSamplerContractError(
            "Trotter replica count exceeds the fused Triton block limit."
        )
    row_ptr, csr_col, csr_weight, max_degree = _reference_order_csr_arrays(
        variable_count=int(h_array.size),
        edge_u=u_array,
        edge_v=v_array,
        edge_j=j_array,
    )
    if max_degree < 1 and j_array.size:
        raise OceanSamplerContractError("Invalid CSR maximum degree.")

    h_tensor = torch.as_tensor(h_array, dtype=torch.float32, device=normalized_device)
    row_ptr_tensor = torch.as_tensor(row_ptr, dtype=torch.long, device=normalized_device)
    csr_col_tensor = torch.as_tensor(csr_col, dtype=torch.long, device=normalized_device)
    csr_weight_tensor = torch.as_tensor(
        csr_weight, dtype=torch.float32, device=normalized_device
    )
    color_tensors = tuple(
        torch.as_tensor(
            np.asarray(group, dtype=np.int64),
            dtype=torch.long,
            device=normalized_device,
        )
        for group in color_groups
    )
    if not color_tensors or any(int(group.numel()) == 0 for group in color_tensors):
        raise OceanSamplerContractError(
            "color_groups must contain non-empty index groups."
        )

    annealing_steps = settings.sweeps - settings.burn_in_sweeps
    beta_schedule = _geometric_schedule(
        settings.beta_range[0], settings.beta_range[1], annealing_steps
    )
    field_schedule = _geometric_schedule(
        settings.transverse_field_range[0],
        settings.transverse_field_range[1],
        annealing_steps,
    )
    full_beta = np.concatenate(
        (
            np.full(settings.burn_in_sweeps, settings.beta_range[0], dtype=np.float64),
            beta_schedule,
        )
    )
    full_field = np.concatenate(
        (
            np.full(
                settings.burn_in_sweeps,
                settings.transverse_field_range[0],
                dtype=np.float64,
            ),
            field_schedule,
        )
    )
    if full_beta.size != settings.sweeps or full_field.size != settings.sweeps:
        raise OceanSamplerExecutionError("Invalid fused SQA schedule length.")

    # Precompute host scalars once. They are the exact same SQA coefficients
    # used by the reference implementation, only moved out of the hot loop.
    beta_scales = np.empty(settings.sweeps, dtype=np.float32)
    trotter_couplings = np.empty(settings.sweeps, dtype=np.float32)
    for sweep_index, (beta_value, field_value) in enumerate(
        zip(full_beta, full_field, strict=True)
    ):
        scaled_field = float(beta_value) * float(field_value) / settings.trotter_replicas
        scaled_field = max(scaled_field, np.finfo(np.float64).tiny)
        beta_scales[sweep_index] = np.float32(
            2.0 * float(beta_value) / settings.trotter_replicas
        )
        trotter_couplings[sweep_index] = np.float32(
            0.5 * math.log(1.0 / math.tanh(scaled_field))
        )

    generator = torch.Generator(device=normalized_device)
    generator.manual_seed(settings.seed)
    batches: list[NDArray[np.int8]] = []
    completed = 0
    adaptive_batch = min(normalized_batch, settings.num_reads)
    oom_retries = 0
    smallest_successful_batch = adaptive_batch
    total_color_launches = 0

    while completed < settings.num_reads:
        current_batch = min(adaptive_batch, settings.num_reads - completed)
        generator_state = generator.get_state()
        state = None
        initial_state_brn = None
        random_even = None
        random_odd = None
        try:
            # Preserve the reference RNG request for the initial state exactly:
            # same [B,R,N] shape and same torch.Generator. Only after drawing do
            # we transpose to [B,N,R] for coalesced fused-kernel memory access.
            initial_state_brn = torch.randint(
                low=0,
                high=2,
                size=(current_batch, settings.trotter_replicas, h_array.size),
                dtype=torch.int8,
                device=normalized_device,
                generator=generator,
            )
            initial_state_brn.mul_(2).sub_(1)
            state = initial_state_brn.permute(0, 2, 1).contiguous()
            initial_state_brn = None

            half_replicas_runtime = settings.trotter_replicas // 2
            for sweep_index in range(settings.sweeps):
                beta_scale = float(beta_scales[sweep_index])
                trotter_coupling = float(trotter_couplings[sweep_index])
                for color_indices in color_tensors:
                    color_size = int(color_indices.numel())
                    # Preserve the reference random-draw API, shape, generator,
                    # and parity order. Drawing odd uniforms before the fused
                    # launch is equivalent because RNG is independent of state.
                    random_even = torch.rand(
                        (current_batch, half_replicas_runtime, color_size),
                        dtype=torch.float32,
                        device=normalized_device,
                        generator=generator,
                    )
                    random_odd = torch.rand(
                        (current_batch, half_replicas_runtime, color_size),
                        dtype=torch.float32,
                        device=normalized_device,
                        generator=generator,
                    )
                    grid = (current_batch * color_size,)
                    color_kernel[grid](
                        state,
                        random_even,
                        random_odd,
                        h_tensor,
                        row_ptr_tensor,
                        csr_col_tensor,
                        csr_weight_tensor,
                        color_indices,
                        beta_scale,
                        trotter_coupling,
                        batch_size=current_batch,
                        replicas=settings.trotter_replicas,
                        variable_count=int(h_array.size),
                        color_size=color_size,
                        max_degree=max(1, max_degree),
                        block_replicas=block_replicas,
                        num_warps=4,
                    )
                    total_color_launches += 1

            replica_sum = state.to(dtype=torch.int16).sum(dim=2)
            first_replica = state[:, :, 0]
            majority = torch.where(
                replica_sum > 0,
                torch.ones_like(replica_sum, dtype=torch.int8),
                torch.where(
                    replica_sum < 0,
                    -torch.ones_like(replica_sum, dtype=torch.int8),
                    first_replica,
                ),
            )
            binary = ((majority + 1) // 2).to(dtype=torch.int8)
            batches.append(binary.detach().cpu().numpy().astype(np.int8, copy=False))
            completed += current_batch
            smallest_successful_batch = min(smallest_successful_batch, current_batch)

        except Exception as exc:
            if not _is_cuda_oom(torch, exc):
                raise
            oom_retries += 1
            try:
                generator.set_state(generator_state)
            except Exception as restore_exc:
                raise OceanSamplerExecutionError(
                    "CUDA OOM retry could not restore the SQA RNG state."
                ) from restore_exc
            if current_batch <= 1:
                raise OceanSamplerExecutionError(
                    "CUDA memory is insufficient even for one fused SQA read."
                ) from None
            adaptive_batch = max(1, current_batch // SQA_OOM_BATCH_REDUCTION_FACTOR)
            try:
                exc.__traceback__ = None
            except Exception:
                pass
            state = None
            initial_state_brn = None
            random_even = None
            random_odd = None
            gc.collect()
            _release_cuda_cache(torch)
            continue
        finally:
            state = None
            initial_state_brn = None
            random_even = None
            random_odd = None

    result = np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.int8)
    if result.shape != (settings.num_reads, h_array.size):
        raise OceanSamplerExecutionError(
            "Fused GPU emulator returned an invalid sample matrix shape."
        )
    if not np.all((result == 0) | (result == 1)):
        raise OceanSamplerExecutionError(
            "Fused GPU emulator returned non-binary samples."
        )

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "requested_batch_size": normalized_batch,
                "effective_batch_size": adaptive_batch,
                "smallest_successful_batch_size": smallest_successful_batch,
                "oom_retries": oom_retries,
                "kernel_strategy": "triton_fused_selected_csr_fields",
                "local_field_strategy": "exact_on_demand_csr",
                "state_layout": "B,N,R",
                "rng_strategy": "reference_shape_torch_rand_per_color_parity",
                "triton_version": triton_version,
                "color_count": len(color_tensors),
                "max_active_degree": max_degree,
                "triton_block_replicas": block_replicas,
                "color_kernel_launches": total_color_launches,
                "local_field_rebase_interval": 0,
                "local_field_rebuilds_last_batch": 0,
            }
        )

    result.setflags(write=False)
    return result


def _is_cuda_oom(torch: Any, exc: BaseException) -> bool:
    """Return True only for PyTorch CUDA out-of-memory exceptions."""

    oom_types: list[type[BaseException]] = []
    for owner in (torch, getattr(torch, "cuda", None)):
        candidate = getattr(owner, "OutOfMemoryError", None) if owner is not None else None
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            if candidate not in oom_types:
                oom_types.append(candidate)
    return bool(oom_types) and isinstance(exc, tuple(oom_types))


def _release_cuda_cache(torch: Any) -> None:
    """Release unused cached CUDA blocks without changing active tensors."""

    cuda = getattr(torch, "cuda", None)
    empty_cache = getattr(cuda, "empty_cache", None) if cuda is not None else None
    if callable(empty_cache):
        empty_cache()


def _run_sqa_torch(
    *,
    torch: Any,
    h: NDArray[np.float64],
    edge_u: NDArray[np.int64],
    edge_v: NDArray[np.int64],
    edge_j: NDArray[np.float64],
    color_groups: Sequence[Sequence[int]],
    settings: SQASettings,
    device: str,
    batch_size: int,
    diagnostics: dict[str, Any] | None = None,
) -> NDArray[np.int8]:
    """Execute SQA with adaptive CUDA micro-batching.

    ``settings.num_reads`` is never reduced.  If a micro-batch exceeds the
    available CUDA memory, only the in-memory micro-batch is halved and retried.
    CPU fallback is never introduced.
    """

    normalized_device_for_dispatch = _nonempty_string(device, name="device")
    if normalized_device_for_dispatch.startswith("cuda"):
        return _run_sqa_triton_fused(
            torch=torch,
            h=h,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_j=edge_j,
            color_groups=color_groups,
            settings=settings,
            device=normalized_device_for_dispatch,
            batch_size=batch_size,
            diagnostics=diagnostics,
        )

    h_array = np.asarray(h, dtype=np.float64)
    u_array = np.asarray(edge_u, dtype=np.int64)
    v_array = np.asarray(edge_v, dtype=np.int64)
    j_array = np.asarray(edge_j, dtype=np.float64)
    if h_array.ndim != 1 or h_array.size == 0:
        raise OceanSamplerContractError("h must be a non-empty vector.")
    if not (u_array.ndim == v_array.ndim == j_array.ndim == 1):
        raise OceanSamplerContractError("Edge arrays must be one-dimensional.")
    if not (u_array.size == v_array.size == j_array.size):
        raise OceanSamplerContractError("Edge arrays must have equal length.")
    if not np.all(np.isfinite(h_array)) or not np.all(np.isfinite(j_array)):
        raise OceanSamplerContractError("Ising coefficients must be finite.")
    if u_array.size:
        if int(u_array.min()) < 0 or int(v_array.min()) < 0:
            raise OceanSamplerContractError("Edge indices must be non-negative.")
        if int(u_array.max()) >= h_array.size or int(v_array.max()) >= h_array.size:
            raise OceanSamplerContractError("Edge index exceeds h size.")

    normalized_batch = _positive_integer(batch_size, name="batch_size")
    normalized_device = _nonempty_string(device, name="device")
    generator = torch.Generator(device=normalized_device)
    generator.manual_seed(settings.seed)

    h_tensor = torch.as_tensor(
        h_array,
        dtype=torch.float32,
        device=normalized_device,
    )
    edge_u_tensor = torch.as_tensor(
        u_array,
        dtype=torch.long,
        device=normalized_device,
    )
    edge_v_tensor = torch.as_tensor(
        v_array,
        dtype=torch.long,
        device=normalized_device,
    )
    edge_j_tensor = torch.as_tensor(
        j_array,
        dtype=torch.float32,
        device=normalized_device,
    )
    color_tensors = tuple(
        torch.as_tensor(
            np.asarray(group, dtype=np.int64),
            dtype=torch.long,
            device=normalized_device,
        )
        for group in color_groups
    )
    if not color_tensors or any(int(group.numel()) == 0 for group in color_tensors):
        raise OceanSamplerContractError(
            "color_groups must contain non-empty index groups."
        )

    plan_arrays = _incremental_color_plan_arrays(
        variable_count=int(h_array.size),
        edge_u=u_array,
        edge_v=v_array,
        edge_j=j_array,
        color_groups=color_groups,
    )
    color_plans = tuple(
        (
            torch.as_tensor(color, dtype=torch.long, device=normalized_device),
            torch.as_tensor(source_positions, dtype=torch.long, device=normalized_device),
            torch.as_tensor(destinations, dtype=torch.long, device=normalized_device),
            torch.as_tensor(weights, dtype=torch.float32, device=normalized_device),
        )
        for color, source_positions, destinations, weights in plan_arrays
    )
    replicas = settings.trotter_replicas
    parity_plans = tuple(
        (
            torch.arange(parity, replicas, 2, dtype=torch.long, device=normalized_device),
            torch.remainder(
                torch.arange(parity, replicas, 2, dtype=torch.long, device=normalized_device) - 1,
                replicas,
            ),
            torch.remainder(
                torch.arange(parity, replicas, 2, dtype=torch.long, device=normalized_device) + 1,
                replicas,
            ),
        )
        for parity in (0, 1)
    )

    annealing_steps = settings.sweeps - settings.burn_in_sweeps
    beta_schedule = _geometric_schedule(
        settings.beta_range[0],
        settings.beta_range[1],
        annealing_steps,
    )
    field_schedule = _geometric_schedule(
        settings.transverse_field_range[0],
        settings.transverse_field_range[1],
        annealing_steps,
    )

    batches: list[NDArray[np.int8]] = []
    completed = 0
    adaptive_batch = min(normalized_batch, settings.num_reads)
    oom_retries = 0
    smallest_successful_batch = adaptive_batch

    while completed < settings.num_reads:
        current_batch = min(adaptive_batch, settings.num_reads - completed)
        generator_state = generator.get_state()
        state = None
        local = None
        local_field_rebuilds = 0

        try:
            state = torch.randint(
                low=0,
                high=2,
                size=(
                    current_batch,
                    settings.trotter_replicas,
                    h_array.size,
                ),
                dtype=torch.int8,
                device=normalized_device,
                generator=generator,
            )
            state.mul_(2).sub_(1)
            local = _problem_local_fields_torch(
                torch,
                state,
                h_tensor,
                edge_u_tensor,
                edge_v_tensor,
                edge_j_tensor,
            )
            local_field_rebuilds += 1
            sweep_counter = 0

            for _ in range(settings.burn_in_sweeps):
                if (
                    sweep_counter > 0
                    and sweep_counter % SQA_LOCAL_FIELD_REBASE_INTERVAL == 0
                ):
                    local = _problem_local_fields_torch(
                        torch,
                        state,
                        h_tensor,
                        edge_u_tensor,
                        edge_v_tensor,
                        edge_j_tensor,
                    )
                    local_field_rebuilds += 1
                _single_sqa_sweep_incremental_torch(
                    torch=torch,
                    state=state,
                    local=local,
                    color_plans=color_plans,
                    parity_plans=parity_plans,
                    beta=settings.beta_range[0],
                    transverse_field=settings.transverse_field_range[0],
                    generator=generator,
                )
                sweep_counter += 1

            for beta_value, field_value in zip(
                beta_schedule,
                field_schedule,
                strict=True,
            ):
                if (
                    sweep_counter > 0
                    and sweep_counter % SQA_LOCAL_FIELD_REBASE_INTERVAL == 0
                ):
                    local = _problem_local_fields_torch(
                        torch,
                        state,
                        h_tensor,
                        edge_u_tensor,
                        edge_v_tensor,
                        edge_j_tensor,
                    )
                    local_field_rebuilds += 1
                _single_sqa_sweep_incremental_torch(
                    torch=torch,
                    state=state,
                    local=local,
                    color_plans=color_plans,
                    parity_plans=parity_plans,
                    beta=float(beta_value),
                    transverse_field=float(field_value),
                    generator=generator,
                )
                sweep_counter += 1

            replica_sum = state.to(dtype=torch.int16).sum(dim=1)
            majority = torch.where(
                replica_sum > 0,
                torch.ones_like(replica_sum, dtype=torch.int8),
                torch.where(
                    replica_sum < 0,
                    -torch.ones_like(replica_sum, dtype=torch.int8),
                    state[:, 0, :],
                ),
            )
            binary = ((majority + 1) // 2).to(dtype=torch.int8)
            batches.append(
                binary.detach().cpu().numpy().astype(np.int8, copy=False)
            )
            completed += current_batch
            smallest_successful_batch = min(smallest_successful_batch, adaptive_batch)

        except Exception as exc:
            if not _is_cuda_oom(torch, exc):
                raise

            oom_retries += 1
            try:
                generator.set_state(generator_state)
            except Exception as restore_exc:
                raise OceanSamplerExecutionError(
                    "CUDA OOM retry could not restore the SQA RNG state."
                ) from restore_exc

            if current_batch <= 1:
                free_bytes = None
                total_bytes = None
                mem_get_info = getattr(getattr(torch, "cuda", None), "mem_get_info", None)
                if callable(mem_get_info):
                    try:
                        free_bytes, total_bytes = mem_get_info()
                    except Exception:
                        free_bytes = None
                        total_bytes = None
                raise OceanSamplerExecutionError(
                    "CUDA memory is insufficient even for one SQA read; "
                    f"replicas={settings.trotter_replicas}, "
                    f"variables={h_array.size}, edges={j_array.size}, "
                    f"free_bytes={free_bytes}, total_bytes={total_bytes}."
                ) from None

            adaptive_batch = max(
                1,
                current_batch // SQA_OOM_BATCH_REDUCTION_FACTOR,
            )

            # Break traceback references to failed CUDA temporaries before
            # returning unused cached blocks to the allocator.
            try:
                exc.__traceback__ = None
            except Exception:
                pass
            state = None
            local = None
            gc.collect()
            _release_cuda_cache(torch)
            continue

        finally:
            state = None
            local = None

    result = np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.int8)
    if result.shape != (settings.num_reads, h_array.size):
        raise OceanSamplerExecutionError(
            "GPU emulator returned an invalid sample matrix shape."
        )
    if not np.all((result == 0) | (result == 1)):
        raise OceanSamplerExecutionError(
            "GPU emulator returned non-binary samples."
        )

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "requested_batch_size": normalized_batch,
                "effective_batch_size": adaptive_batch,
                "smallest_successful_batch_size": smallest_successful_batch,
                "oom_retries": oom_retries,
                "kernel_strategy": "incremental_local_fields_torch",
                "color_count": len(color_plans),
                "directed_incremental_edges": 2 * int(j_array.size),
                "local_field_rebase_interval": SQA_LOCAL_FIELD_REBASE_INTERVAL,
                "local_field_rebuilds_last_batch": local_field_rebuilds,
            }
        )

    result.setflags(write=False)
    return result


def _build_structured_gpu_sampler(
    *,
    dimod: Any,
    torch: Any,
    topology: PegasusTopology,
    config: object,
    seed: int,
    memory_fraction: float,
) -> object:
    parameters = {
        "num_reads": [],
        "seed": [],
        "trotter_replicas": [],
        "sweeps": [],
        "burn_in_sweeps": [],
        "beta_range": [],
        "transverse_field_range": [],
        "memory_fraction": [],
        "label": [],
    }
    properties = {
        "category": LOCAL_EMULATOR_CATEGORY,
        "provider": LOCAL_EMULATOR_PROVIDER,
        "device": LOCAL_EMULATOR_DEVICE,
        "qpu": False,
        "topology": {
            "type": "pegasus",
            "shape": [topology.pegasus_m],
        },
        "chip_id": f"CSSF_Pegasus_GPU_Emulator_P{topology.pegasus_m}",
        "topology_fingerprint": topology.fingerprint(),
        "classical_fallback": False,
        "kernel_strategy": "triton_fused_selected_csr_fields",
        "local_field_strategy": "exact_on_demand_csr",
        "state_layout": "B,N,R",
    }
    node_order = tuple(sorted(topology.nodes, key=_node_sort_key))
    edge_order = tuple(topology.edges)
    edge_lookup = _canonical_edge_set(edge_order)
    default_num_reads = validate_num_reads(getattr(config, "num_reads", None))
    default_replicas = _positive_integer(
        getattr(config, "sqa_trotter_replicas", None),
        name="sqa_trotter_replicas",
    )
    default_sweeps = _positive_integer(
        getattr(config, "sqa_sweeps", None),
        name="sqa_sweeps",
    )
    default_burn_in = _nonnegative_integer(
        getattr(config, "sqa_burn_in_sweeps", None),
        name="sqa_burn_in_sweeps",
    )

    class PegasusSQAEmulatorSampler(dimod.Sampler, dimod.Structured):
        @property
        def parameters(self) -> Mapping[str, list[Any]]:
            return parameters

        @property
        def properties(self) -> Mapping[str, Any]:
            return properties

        @property
        def nodelist(self) -> list[PegasusNode]:
            return list(node_order)

        @property
        def edgelist(self) -> list[PegasusEdge]:
            return list(edge_order)

        @property
        def adjacency(self) -> Mapping[PegasusNode, set[PegasusNode]]:
            return {
                node: set(neighbours)
                for node, neighbours in topology.adjacency.items()
            }

        def sample(self, bqm: object, **kwargs: Any) -> object:
            accepted = set(parameters)
            unknown = tuple(sorted(set(kwargs) - accepted))
            if unknown:
                raise OceanSamplerContractError(
                    f"Unsupported emulator sampling parameters: {unknown}."
                )
            num_reads = validate_num_reads(
                kwargs.get("num_reads", default_num_reads)
            )
            local_seed = _nonnegative_integer(
                kwargs.get("seed", seed),
                name="seed",
            )
            replicas = _positive_integer(
                kwargs.get("trotter_replicas", default_replicas),
                name="trotter_replicas",
            )
            sweeps = _positive_integer(
                kwargs.get("sweeps", default_sweeps),
                name="sweeps",
            )
            burn_in = _nonnegative_integer(
                kwargs.get("burn_in_sweeps", default_burn_in),
                name="burn_in_sweeps",
            )
            beta_range = kwargs.get(
                "beta_range",
                DEFAULT_EMULATOR_BETA_RANGE,
            )
            transverse_field_range = kwargs.get(
                "transverse_field_range",
                DEFAULT_EMULATOR_TRANSVERSE_FIELD_RANGE,
            )
            local_memory_fraction = _memory_fraction(
                kwargs.get("memory_fraction", memory_fraction)
            )
            label = _optional_string(kwargs.get("label"), name="label")
            settings = SQASettings(
                num_reads=num_reads,
                trotter_replicas=replicas,
                sweeps=sweeps,
                burn_in_sweeps=burn_in,
                beta_range=beta_range,
                transverse_field_range=transverse_field_range,
                seed=local_seed,
                memory_fraction=local_memory_fraction,
            )

            variables = tuple(bqm.variables)
            if not variables:
                raise OceanSamplerContractError(
                    "Cannot sample an empty binary quadratic model."
                )
            topology_nodes = set(node_order)
            missing_nodes = tuple(
                variable for variable in variables if variable not in topology_nodes
            )
            if missing_nodes:
                raise OceanSamplerContractError(
                    "Embedded BQM uses variables outside the Pegasus topology: "
                    f"{missing_nodes[:8]}."
                )
            invalid_edges = tuple(
                edge
                for edge in bqm.quadratic
                if frozenset(edge) not in edge_lookup
            )
            if invalid_edges:
                raise OceanSamplerContractError(
                    "Embedded BQM contains interactions outside the Pegasus "
                    f"topology: {invalid_edges[:8]}."
                )

            vartype_name = str(getattr(bqm.vartype, "name", bqm.vartype)).upper()
            if vartype_name not in {"BINARY", "SPIN"}:
                raise OceanSamplerContractError(
                    "Emulator accepts only BINARY or SPIN BQMs."
                )
            binary_bqm = bqm.change_vartype(dimod.BINARY, inplace=False)
            h_map, j_map, _ = binary_bqm.to_ising()
            variable_index = {
                variable: index for index, variable in enumerate(variables)
            }
            h = np.asarray(
                [float(h_map.get(variable, 0.0)) for variable in variables],
                dtype=np.float64,
            )
            indexed_edges: list[tuple[int, int]] = []
            edge_biases: list[float] = []
            for (first, second), bias in j_map.items():
                indexed_edges.append(
                    (variable_index[first], variable_index[second])
                )
                edge_biases.append(float(bias))
            color_groups = _active_graph_coloring(
                len(variables),
                indexed_edges,
            )

            _release_cuda_cache(torch)
            free_bytes, _ = torch.cuda.mem_get_info()
            batch_size = _estimate_fused_gpu_batch_size(
                free_bytes=int(free_bytes),
                num_reads=settings.num_reads,
                replicas=settings.trotter_replicas,
                variable_count=len(variables),
                memory_fraction=settings.memory_fraction,
            )
            execution_diagnostics: dict[str, Any] = {}
            samples = _run_sqa_torch(
                torch=torch,
                h=h,
                edge_u=np.asarray(
                    [edge[0] for edge in indexed_edges],
                    dtype=np.int64,
                ),
                edge_v=np.asarray(
                    [edge[1] for edge in indexed_edges],
                    dtype=np.int64,
                ),
                edge_j=np.asarray(edge_biases, dtype=np.float64),
                color_groups=color_groups,
                settings=settings,
                device=LOCAL_EMULATOR_DEVICE,
                batch_size=batch_size,
                diagnostics=execution_diagnostics,
            )
            info = {
                "cssf_backend": LOCAL_GPU_BACKEND_KIND,
                "provider": LOCAL_EMULATOR_PROVIDER,
                "device": LOCAL_EMULATOR_DEVICE,
                "topology_type": "pegasus",
                "pegasus_m": topology.pegasus_m,
                "topology_fingerprint": topology.fingerprint(),
                "settings_fingerprint": settings.fingerprint(),
                "batch_size": execution_diagnostics.get(
                    "effective_batch_size",
                    batch_size,
                ),
                "requested_batch_size": execution_diagnostics.get(
                    "requested_batch_size",
                    batch_size,
                ),
                "smallest_successful_batch_size": execution_diagnostics.get(
                    "smallest_successful_batch_size",
                    batch_size,
                ),
                "oom_retries": execution_diagnostics.get("oom_retries", 0),
                "kernel_strategy": execution_diagnostics.get(
                    "kernel_strategy", "triton_fused_selected_csr_fields"
                ),
                "color_count": execution_diagnostics.get("color_count", len(color_groups)),
                "directed_incremental_edges": execution_diagnostics.get(
                    "directed_incremental_edges", 2 * len(indexed_edges)
                ),
                "local_field_rebase_interval": execution_diagnostics.get(
                    "local_field_rebase_interval", SQA_LOCAL_FIELD_REBASE_INTERVAL
                ),
                "local_field_rebuilds_last_batch": execution_diagnostics.get(
                    "local_field_rebuilds_last_batch", 0
                ),
                "label": label,
                "classical_fallback": False,
            }
            output_samples = (
                samples
                if vartype_name == "BINARY"
                else np.asarray(samples * 2 - 1, dtype=np.int8)
            )
            try:
                return dimod.SampleSet.from_samples_bqm(
                    (output_samples, variables),
                    bqm,
                    info=info,
                    aggregate_samples=True,
                    sort_labels=False,
                )
            except Exception as exc:
                raise OceanSamplerExecutionError(
                    "dimod could not construct the emulator SampleSet."
                ) from exc

    sampler = PegasusSQAEmulatorSampler()
    return validate_sampler_interface(sampler)


@dataclass(frozen=True, slots=True)
class OceanSamplerBundle:
    """A validated sampler plus immutable backend provenance."""

    mode: str
    sampler: object
    raw_sampler: object
    topology: PegasusTopology
    solver_id: str | None
    dry_run: bool
    default_sample_parameters: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        mode = _normalized_mode(self.mode)
        validate_sampler_interface(self.sampler)
        validate_sampler_interface(self.raw_sampler)
        if not isinstance(self.topology, PegasusTopology):
            raise TypeError("topology must be PegasusTopology.")
        solver_id = (
            None
            if self.solver_id is None
            else validate_solver_id(self.solver_id)
        )
        dry_run = _strict_bool(self.dry_run, name="dry_run")
        if mode == PEGASUS_QPU_BACKEND_KIND and solver_id is None:
            raise OceanSamplerContractError(
                "QPU bundle requires a concrete solver_id."
            )
        if mode == LOCAL_GPU_BACKEND_KIND and solver_id is not None:
            raise OceanSamplerContractError(
                "Local emulator bundle must not carry a real QPU solver_id."
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "solver_id", solver_id)
        object.__setattr__(self, "dry_run", dry_run)
        object.__setattr__(
            self,
            "default_sample_parameters",
            _json_mapping(
                self.default_sample_parameters,
                name="default sample parameters",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, name="bundle metadata"),
        )

    def fingerprint(self) -> str:
        payload = {
            "mode": self.mode,
            "solver_id": self.solver_id,
            "dry_run": self.dry_run,
            "topology_fingerprint": self.topology.fingerprint(),
            "default_sample_parameters": dict(self.default_sample_parameters),
            "metadata": dict(self.metadata),
        }
        digest = hashlib.sha256()
        digest.update(b"CSSF-OceanSamplerBundle-v1\0")
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def sample_bqm(
        self,
        bqm: object,
        *,
        num_reads: int | None = None,
        label: str = DEFAULT_QPU_LABEL,
        **parameters: Any,
    ) -> object:
        """Sample one BQM through the selected emulator or QPU backend."""

        if self.dry_run:
            raise OceanSamplerExecutionError(
                "QPU dry_run is enabled; validation succeeded but submission "
                "is intentionally blocked."
            )
        resolved_label = _nonempty_string(label, name="label")
        resolved_reads = (
            None if num_reads is None else validate_num_reads(num_reads)
        )
        kwargs = dict(self.default_sample_parameters)
        kwargs.update(parameters)
        kwargs["label"] = resolved_label
        if resolved_reads is not None:
            kwargs["num_reads"] = resolved_reads
        try:
            sampleset = self.sampler.sample(bqm, **kwargs)
        except Exception as exc:
            raise OceanSamplerExecutionError(
                f"{self.mode} sampling failed."
            ) from exc

        dimod = _load_dimod()
        sample_set_class = getattr(dimod, "SampleSet", None)
        if sample_set_class is None or not isinstance(sampleset, sample_set_class):
            raise OceanSamplerContractError(
                "Sampler did not return dimod.SampleSet."
            )
        info = getattr(sampleset, "info", None)
        if not isinstance(info, dict):
            raise OceanSamplerContractError(
                "dimod.SampleSet.info must be a mutable dictionary."
            )
        info.update(
            {
                "cssf_bundle_fingerprint": self.fingerprint(),
                "cssf_backend": self.mode,
                "cssf_solver_id": self.solver_id,
                "cssf_topology_fingerprint": self.topology.fingerprint(),
            }
        )
        return sampleset

    def close(self) -> None:
        """Close Ocean resources without submitting additional work."""

        _close_sampler_quietly(self.sampler)
        if self.raw_sampler is not self.sampler:
            _close_sampler_quietly(self.raw_sampler)

    def __enter__(self) -> "OceanSamplerBundle":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def build_local_pegasus_gpu_sampler(
    config: object,
    *,
    seed: int = 505,
    memory_fraction: float = DEFAULT_EMULATOR_MEMORY_FRACTION,
) -> OceanSamplerBundle:
    """Build the local CUDA SQA emulator and its Ocean embedding layer."""

    backend = _normalized_mode(getattr(config, "backend", None))
    if backend != LOCAL_GPU_BACKEND_KIND:
        raise OceanSamplerContractError(
            "Local emulator requires backend='local_sqa_gpu'."
        )
    validate_topology_type(getattr(config, "topology_type", None))
    if getattr(config, "require_gpu", None) is not True:
        raise OceanSamplerContractError(
            "Local emulator must require GPU."
        )
    if getattr(config, "allow_classical_fallback", None) is not False:
        raise OceanSamplerContractError(
            "Local emulator classical fallback is prohibited."
        )
    if getattr(config, "return_dimod_sampleset", None) is not True:
        raise OceanSamplerContractError(
            "Local emulator must return dimod.SampleSet."
        )
    normalized_seed = _nonnegative_integer(seed, name="seed")
    normalized_memory_fraction = _memory_fraction(memory_fraction)

    topology = topology_from_emulator_config(config)
    dimod = _load_dimod()
    torch = _load_torch_cuda()
    ocean_system = _load_ocean_system()
    embedding_composite = getattr(ocean_system, "EmbeddingComposite", None)
    if embedding_composite is None or not callable(embedding_composite):
        raise OceanRuntimeUnavailableError(
            "dwave.system.EmbeddingComposite is unavailable."
        )

    raw_sampler = _build_structured_gpu_sampler(
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
        raise OceanRuntimeUnavailableError(
            "Ocean could not wrap the GPU emulator with EmbeddingComposite."
        ) from exc
    validate_sampler_interface(sampler)
    return OceanSamplerBundle(
        mode=LOCAL_GPU_BACKEND_KIND,
        sampler=sampler,
        raw_sampler=raw_sampler,
        topology=topology,
        solver_id=None,
        dry_run=False,
        default_sample_parameters={
            "num_reads": validate_num_reads(getattr(config, "num_reads", None)),
        },
        metadata={
            "provider": LOCAL_EMULATOR_PROVIDER,
            "device": LOCAL_EMULATOR_DEVICE,
            "pegasus_m": topology.pegasus_m,
            "classical_fallback": False,
            "seed": normalized_seed,
            "memory_fraction": normalized_memory_fraction,
        },
    )


def build_leap_pegasus_qpu_sampler(
    config: object,
    *,
    token: str | None = None,
    endpoint: str | None = None,
    region: str | None = None,
) -> OceanSamplerBundle:
    """Build an explicit Pegasus QPU sampler through D-Wave Leap/Ocean."""

    backend = _normalized_mode(getattr(config, "backend", None))
    if backend != PEGASUS_QPU_BACKEND_KIND:
        raise OceanSamplerContractError(
            "Leap QPU requires backend='pegasus_qpu'."
        )
    validate_topology_type(getattr(config, "topology_type", None))
    if getattr(config, "enabled", None) is not True:
        raise OceanSamplerContractError(
            "QPU configuration must set enabled=True before connection."
        )
    if getattr(config, "require_explicit_solver_id", None) is not True:
        raise OceanSamplerContractError(
            "QPU configuration must require an explicit solver_id."
        )
    if getattr(config, "allow_solver_fallback", None) is not False:
        raise OceanSamplerContractError(
            "QPU solver fallback is prohibited."
        )
    if getattr(config, "reject_zephyr", None) is not True:
        raise OceanSamplerContractError(
            "QPU configuration must reject Zephyr."
        )
    solver_id = validate_solver_id(getattr(config, "solver_id", None))
    dry_run = _strict_bool(getattr(config, "dry_run", None), name="dry_run")
    validate_num_reads(getattr(config, "num_reads", None))
    annealing_time = _finite_float(
        getattr(config, "annealing_time", None),
        name="annealing_time",
    )
    if annealing_time <= 0.0:
        raise OceanSamplerContractError("annealing_time must be positive.")

    explicit_token = _optional_string(token, name="token")
    explicit_endpoint = _optional_string(endpoint, name="endpoint")
    explicit_region = _optional_string(region, name="region")
    ocean_system = _load_ocean_system()
    dwave_sampler_class = getattr(ocean_system, "DWaveSampler", None)
    embedding_composite = getattr(ocean_system, "EmbeddingComposite", None)
    if dwave_sampler_class is None or not callable(dwave_sampler_class):
        raise OceanRuntimeUnavailableError(
            "dwave.system.DWaveSampler is unavailable."
        )
    if embedding_composite is None or not callable(embedding_composite):
        raise OceanRuntimeUnavailableError(
            "dwave.system.EmbeddingComposite is unavailable."
        )

    constructor_kwargs: dict[str, Any] = {
        "solver": {"name": solver_id},
    }
    if explicit_token is not None:
        constructor_kwargs["token"] = explicit_token
    if explicit_endpoint is not None:
        constructor_kwargs["endpoint"] = explicit_endpoint
    if explicit_region is not None:
        constructor_kwargs["region"] = explicit_region

    raw_sampler: object | None = None
    try:
        raw_sampler = dwave_sampler_class(**constructor_kwargs)
        _validate_qpu_runtime_sampler(
            raw_sampler,
            expected_solver_id=solver_id,
        )
        topology = topology_from_qpu_config(config, raw_sampler)
        sampler = embedding_composite(raw_sampler)
        validate_sampler_interface(sampler)
    except Exception:
        _close_sampler_quietly(raw_sampler)
        raise

    token_source = (
        "explicit_argument"
        if explicit_token is not None
        else (
            "environment_or_ocean_config"
            if os.environ.get(DWAVE_API_TOKEN_ENV)
            else "ocean_config_resolution"
        )
    )
    return OceanSamplerBundle(
        mode=PEGASUS_QPU_BACKEND_KIND,
        sampler=sampler,
        raw_sampler=raw_sampler,
        topology=topology,
        solver_id=solver_id,
        dry_run=dry_run,
        default_sample_parameters={
            "num_reads": validate_num_reads(getattr(config, "num_reads", None)),
            "annealing_time": annealing_time,
        },
        metadata={
            "provider": "dwave_leap_ocean",
            "solver_id": solver_id,
            "topology_type": "pegasus",
            "token_source": token_source,
            "token_stored": False,
            "annealing_time": annealing_time,
            "solver_fallback": False,
            "embedding": "EmbeddingComposite",
        },
    )


def build_ocean_sampler(
    mode: SamplerMode | str,
    *,
    emulator_config: object | None = None,
    qpu_config: object | None = None,
    seed: int = 505,
    memory_fraction: float = DEFAULT_EMULATOR_MEMORY_FRACTION,
    token: str | None = None,
    endpoint: str | None = None,
    region: str | None = None,
) -> OceanSamplerBundle:
    """Build the notebook-selected emulator or real Leap QPU sampler.

    No automatic fallback occurs between modes. A failure in the selected
    branch is raised to the caller and never triggers the other backend.
    """

    normalized_mode = _normalized_mode(mode)
    if normalized_mode == LOCAL_GPU_BACKEND_KIND:
        if emulator_config is None:
            raise OceanSamplerContractError(
                "emulator_config is required for local_sqa_gpu."
            )
        if qpu_config is not None:
            raise OceanSamplerContractError(
                "qpu_config must be omitted when emulator mode is selected."
            )
        return build_local_pegasus_gpu_sampler(
            emulator_config,
            seed=seed,
            memory_fraction=memory_fraction,
        )

    if qpu_config is None:
        raise OceanSamplerContractError(
            "qpu_config is required for pegasus_qpu."
        )
    if emulator_config is not None:
        raise OceanSamplerContractError(
            "emulator_config must be omitted when QPU mode is selected."
        )
    return build_leap_pegasus_qpu_sampler(
        qpu_config,
        token=token,
        endpoint=endpoint,
        region=region,
    )


__all__ = [
    "DWAVE_API_TOKEN_ENV",
    "OCEAN_DIMOD_MODULE",
    "OCEAN_SYSTEM_MODULE",
    "TORCH_MODULE",
    "LOCAL_EMULATOR_PROVIDER",
    "LOCAL_EMULATOR_DEVICE",
    "DEFAULT_EMULATOR_BETA_RANGE",
    "DEFAULT_EMULATOR_TRANSVERSE_FIELD_RANGE",
    "DEFAULT_EMULATOR_MEMORY_FRACTION",
    "DEFAULT_QPU_LABEL",
    "NOTEBOOK_MODE_ALIASES",
    "OceanSamplerError",
    "OceanRuntimeUnavailableError",
    "OceanSamplerContractError",
    "OceanSamplerExecutionError",
    "SamplerMode",
    "SQASettings",
    "OceanSamplerBundle",
    "build_local_pegasus_gpu_sampler",
    "build_leap_pegasus_qpu_sampler",
    "build_ocean_sampler",
]
