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

"""Strict Pegasus topology discovery and validation for CSSF.

The module provides one immutable topology representation shared by the local
GPU emulator and the real D-Wave QPU path.  It validates Ocean-compatible
graphs and samplers, rejects every non-Pegasus family, and preserves the exact
physical node and coupler labels reported by the source.

``dwave_networkx`` is imported lazily only when an ideal local Pegasus graph is
explicitly requested.  Importing this module performs no network access,
credential lookup, sampler construction, graph generation, or backend
selection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import importlib
import json
from types import MappingProxyType
from typing import Any, Final, Iterable, Mapping, Sequence, TypeAlias

from dwave_backend import (
    DWaveBackendContractError,
    LOCAL_GPU_BACKEND_KIND,
    PEGASUS_QPU_BACKEND_KIND,
    PEGASUS_TOPOLOGY_TYPE,
    validate_backend_kind,
    validate_pegasus_m,
    validate_sampler_interface,
    validate_solver_id,
    validate_topology_type,
)


PEGASUS_GRAPH_FACTORY_MODULE: Final[str] = "dwave_networkx"
PEGASUS_GRAPH_FACTORY_NAME: Final[str] = "pegasus_graph"
PEGASUS_LINEAR_LABEL_TYPE: Final[str] = "linear"
PEGASUS_COORDINATE_LABEL_TYPE: Final[str] = "coordinate"
PEGASUS_NICE_COORDINATE_LABEL_TYPE: Final[str] = "nice_coordinate"
SUPPORTED_PEGASUS_LABEL_TYPES: Final[tuple[str, ...]] = (
    PEGASUS_LINEAR_LABEL_TYPE,
    PEGASUS_COORDINATE_LABEL_TYPE,
    PEGASUS_NICE_COORDINATE_LABEL_TYPE,
)
PEGASUS_COORDINATE_LENGTH: Final[int] = 4
PEGASUS_NICE_COORDINATE_LENGTH: Final[int] = 5
PEGASUS_NOMINAL_QUBITS_PER_M_PRODUCT: Final[int] = 24

PegasusCoordinate: TypeAlias = tuple[int, int, int, int]
PegasusNiceCoordinate: TypeAlias = tuple[int, int, int, int, int]
PegasusNode: TypeAlias = int | PegasusCoordinate | PegasusNiceCoordinate
PegasusEdge: TypeAlias = tuple[PegasusNode, PegasusNode]


class PegasusTopologyError(DWaveBackendContractError):
    """Raised when a graph or sampler violates the Pegasus contract."""


class PegasusTopologyRuntimeError(RuntimeError):
    """Raised when the optional Ocean graph factory cannot be used."""


def _validated_topology_type(value: object) -> str:
    try:
        return validate_topology_type(value)
    except DWaveBackendContractError as exc:
        raise PegasusTopologyError(str(exc)) from exc


def _validated_solver_id(value: object) -> str:
    try:
        return validate_solver_id(value)
    except DWaveBackendContractError as exc:
        raise PegasusTopologyError(str(exc)) from exc


def _validated_backend_kind(value: object) -> str:
    try:
        return validate_backend_kind(value)
    except DWaveBackendContractError as exc:
        raise PegasusTopologyError(str(exc)) from exc


def _validated_pegasus_m(value: object) -> int:
    try:
        return validate_pegasus_m(value)  # type: ignore[arg-type]
    except (TypeError, DWaveBackendContractError) as exc:
        raise PegasusTopologyError(str(exc)) from exc


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise PegasusTopologyError(
            f"{name} must be at least {minimum}; received {value}."
        )
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean.")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise PegasusTopologyError(f"{name} must not be empty.")
    return normalized


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
        raise PegasusTopologyError(
            f"{name} must be JSON-serializable."
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _tuple_integer_node(
    value: Sequence[object],
    *,
    expected_length: int,
    name: str,
) -> tuple[int, ...]:
    if len(value) != expected_length:
        raise PegasusTopologyError(
            f"{name} must contain {expected_length} integer coordinates."
        )
    coordinates = tuple(
        _integer(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )
    return coordinates


def normalize_pegasus_node(value: object) -> PegasusNode:
    """Normalize one supported Ocean Pegasus node label."""

    if isinstance(value, bool):
        raise TypeError("Pegasus node labels cannot be boolean.")
    if isinstance(value, int):
        return _integer(value, name="node")
    if isinstance(value, tuple):
        if len(value) == PEGASUS_COORDINATE_LENGTH:
            return _tuple_integer_node(
                value,
                expected_length=PEGASUS_COORDINATE_LENGTH,
                name="Pegasus coordinate",
            )  # type: ignore[return-value]
        if len(value) == PEGASUS_NICE_COORDINATE_LENGTH:
            return _tuple_integer_node(
                value,
                expected_length=PEGASUS_NICE_COORDINATE_LENGTH,
                name="Pegasus nice coordinate",
            )  # type: ignore[return-value]
    raise PegasusTopologyError(
        "Pegasus nodes must use non-negative integer, four-coordinate, "
        "or five-coordinate Ocean labels."
    )


def pegasus_node_label_type(node: PegasusNode) -> str:
    """Return the stable label-family name for one normalized node."""

    if isinstance(node, int):
        return PEGASUS_LINEAR_LABEL_TYPE
    if len(node) == PEGASUS_COORDINATE_LENGTH:
        return PEGASUS_COORDINATE_LABEL_TYPE
    return PEGASUS_NICE_COORDINATE_LABEL_TYPE


def _node_sort_key(node: PegasusNode) -> tuple[int, tuple[int, ...]]:
    if isinstance(node, int):
        return (0, (node,))
    if len(node) == PEGASUS_COORDINATE_LENGTH:
        return (1, tuple(node))
    return (2, tuple(node))


def normalize_pegasus_edge(value: object) -> PegasusEdge:
    """Normalize and canonically order one undirected Pegasus coupler."""

    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise PegasusTopologyError(
            "Each Pegasus edge must contain exactly two node labels."
        )
    first = normalize_pegasus_node(value[0])
    second = normalize_pegasus_node(value[1])
    if first == second:
        raise PegasusTopologyError("Pegasus self-loops are prohibited.")
    if pegasus_node_label_type(first) != pegasus_node_label_type(second):
        raise PegasusTopologyError(
            "Both endpoints of a Pegasus edge must use one label family."
        )
    return (
        (first, second)
        if _node_sort_key(first) < _node_sort_key(second)
        else (second, first)
    )


def nominal_pegasus_qubit_count(pegasus_m: int) -> int:
    """Return the nominal node count ``24*m*(m-1)`` for ``P_m``."""

    normalized_m = _validated_pegasus_m(pegasus_m)
    return (
        PEGASUS_NOMINAL_QUBITS_PER_M_PRODUCT
        * normalized_m
        * (normalized_m - 1)
    )


def _graph_mapping(graph: object) -> Mapping[str, Any]:
    metadata = getattr(graph, "graph", None)
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise PegasusTopologyError("graph.graph must be a mapping.")
    return metadata


def _graph_nodes(graph: object) -> tuple[object, ...]:
    nodes = getattr(graph, "nodes", None)
    if nodes is None:
        raise PegasusTopologyError("Graph does not expose nodes.")
    try:
        values = nodes() if callable(nodes) else nodes
        return tuple(values)
    except Exception as exc:
        raise PegasusTopologyError("Cannot enumerate graph nodes.") from exc


def _graph_edges(graph: object) -> tuple[object, ...]:
    edges = getattr(graph, "edges", None)
    if edges is None:
        raise PegasusTopologyError("Graph does not expose edges.")
    try:
        values = edges() if callable(edges) else edges
        return tuple(values)
    except Exception as exc:
        raise PegasusTopologyError("Cannot enumerate graph edges.") from exc


def _infer_m(
    metadata: Mapping[str, Any],
    *,
    expected_m: int | None,
) -> int:
    if expected_m is not None:
        normalized_expected = _validated_pegasus_m(expected_m)
    else:
        normalized_expected = None

    candidates: list[int] = []
    for key in ("m", "rows"):
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            candidates.append(value)

    shape = metadata.get("shape")
    if isinstance(shape, (tuple, list)) and shape:
        first = shape[0]
        if isinstance(first, int) and not isinstance(first, bool):
            candidates.append(first)

    if normalized_expected is None and not candidates:
        raise PegasusTopologyError(
            "Pegasus graph metadata must provide m, rows, or shape."
        )

    normalized_candidates = tuple(_validated_pegasus_m(item) for item in candidates)
    if normalized_candidates and len(set(normalized_candidates)) != 1:
        raise PegasusTopologyError(
            f"Pegasus graph metadata contains inconsistent sizes: "
            f"{normalized_candidates}."
        )

    inferred = (
        normalized_expected
        if normalized_expected is not None
        else normalized_candidates[0]
    )
    if normalized_candidates and inferred != normalized_candidates[0]:
        raise PegasusTopologyError(
            f"Expected Pegasus m={inferred}, but graph reports "
            f"m={normalized_candidates[0]}."
        )
    return inferred


def _validate_family(metadata: Mapping[str, Any]) -> str:
    raw_family = metadata.get(
        "family",
        metadata.get("topology_type", metadata.get("type")),
    )
    if raw_family is None:
        raise PegasusTopologyError(
            "Graph metadata must explicitly declare family='pegasus'."
        )
    return _validated_topology_type(raw_family)


def _extract_topology_properties(
    properties: Mapping[str, Any],
) -> tuple[str, int]:
    raw_topology = properties.get("topology")
    if not isinstance(raw_topology, Mapping):
        raise PegasusTopologyError(
            "Sampler properties must include a topology mapping."
        )

    topology_type = _validated_topology_type(raw_topology.get("type"))
    shape = raw_topology.get("shape")
    if not isinstance(shape, (tuple, list)) or not shape:
        raise PegasusTopologyError(
            "Sampler topology.shape must contain the Pegasus m value."
        )
    pegasus_m = validate_pegasus_m(
        _integer(shape[0], name="topology.shape[0]", minimum=2)
    )
    return topology_type, pegasus_m


def _sampler_solver_id(sampler: object, explicit: str | None) -> str:
    if explicit is not None:
        return _validated_solver_id(explicit)

    solver = getattr(sampler, "solver", None)
    solver_id = getattr(solver, "id", None) if solver is not None else None
    if solver_id is None:
        properties = getattr(sampler, "properties", None)
        if isinstance(properties, Mapping):
            solver_id = properties.get("chip_id")
    if solver_id is None:
        raise PegasusTopologyError(
            "Real QPU topology requires an explicit concrete solver_id."
        )
    return _validated_solver_id(solver_id)


def _deeply_immutable_adjacency(
    nodes: tuple[PegasusNode, ...],
    edges: tuple[PegasusEdge, ...],
) -> Mapping[PegasusNode, tuple[PegasusNode, ...]]:
    adjacency: dict[PegasusNode, set[PegasusNode]] = {
        node: set() for node in nodes
    }
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    frozen = {
        node: tuple(sorted(neighbours, key=_node_sort_key))
        for node, neighbours in adjacency.items()
    }
    return MappingProxyType(frozen)


def _is_connected(
    nodes: tuple[PegasusNode, ...],
    adjacency: Mapping[PegasusNode, tuple[PegasusNode, ...]],
) -> bool:
    if not nodes:
        return False
    visited: set[PegasusNode] = {nodes[0]}
    queue: deque[PegasusNode] = deque((nodes[0],))
    while queue:
        current = queue.popleft()
        for neighbour in adjacency[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return len(visited) == len(nodes)


@dataclass(frozen=True, slots=True)
class PegasusTopologyAudit:
    """Immutable summary of one validated Pegasus topology."""

    topology_type: str
    pegasus_m: int
    node_count: int
    edge_count: int
    label_type: str
    connected: bool
    ideal: bool
    source: str
    solver_id: str | None
    nominal_node_count: int
    missing_nominal_nodes: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "topology_type",
            _validated_topology_type(self.topology_type),
        )
        pegasus_m = validate_pegasus_m(self.pegasus_m)
        node_count = _integer(self.node_count, name="node_count", minimum=1)
        edge_count = _integer(self.edge_count, name="edge_count")
        if self.label_type not in SUPPORTED_PEGASUS_LABEL_TYPES:
            raise PegasusTopologyError(
                f"Unsupported Pegasus label_type {self.label_type!r}."
            )
        connected = _boolean(self.connected, name="connected")
        ideal = _boolean(self.ideal, name="ideal")
        source = _string(self.source, name="source")
        solver_id = (
            None if self.solver_id is None else _validated_solver_id(self.solver_id)
        )
        nominal = nominal_pegasus_qubit_count(pegasus_m)
        missing = _integer(
            self.missing_nominal_nodes,
            name="missing_nominal_nodes",
        )
        if self.nominal_node_count != nominal:
            raise PegasusTopologyError(
                "nominal_node_count does not match Pegasus m."
            )
        if missing != max(0, nominal - node_count):
            raise PegasusTopologyError(
                "missing_nominal_nodes is inconsistent with node_count."
            )
        if ideal and node_count != nominal:
            raise PegasusTopologyError(
                "An ideal topology must expose the nominal Pegasus nodes."
            )
        object.__setattr__(self, "pegasus_m", pegasus_m)
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "edge_count", edge_count)
        object.__setattr__(self, "connected", connected)
        object.__setattr__(self, "ideal", ideal)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "solver_id", solver_id)
        object.__setattr__(self, "nominal_node_count", nominal)
        object.__setattr__(self, "missing_nominal_nodes", missing)
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, name="audit metadata"),
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-PegasusTopologyAudit-v1\0")
        digest.update(
            json.dumps(
                {
                    "topology_type": self.topology_type,
                    "pegasus_m": self.pegasus_m,
                    "node_count": self.node_count,
                    "edge_count": self.edge_count,
                    "label_type": self.label_type,
                    "connected": self.connected,
                    "ideal": self.ideal,
                    "source": self.source,
                    "solver_id": self.solver_id,
                    "nominal_node_count": self.nominal_node_count,
                    "missing_nominal_nodes": self.missing_nominal_nodes,
                    "metadata": dict(self.metadata),
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class PegasusTopology:
    """Canonical immutable physical Pegasus graph."""

    pegasus_m: int
    nodes: tuple[PegasusNode, ...]
    edges: tuple[PegasusEdge, ...]
    label_type: str
    source: str
    solver_id: str | None
    ideal: bool
    metadata: Mapping[str, Any]
    adjacency: Mapping[PegasusNode, tuple[PegasusNode, ...]]
    connected: bool

    def __init__(
        self,
        *,
        pegasus_m: int,
        nodes: Iterable[object],
        edges: Iterable[object],
        source: str,
        solver_id: str | None = None,
        ideal: bool = False,
        metadata: Mapping[str, Any] | None = None,
        require_connected: bool = True,
    ) -> None:
        normalized_m = _validated_pegasus_m(pegasus_m)
        normalized_nodes = tuple(
            sorted(
                {normalize_pegasus_node(node) for node in nodes},
                key=_node_sort_key,
            )
        )
        if not normalized_nodes:
            raise PegasusTopologyError("Pegasus topology has no nodes.")

        label_types = {
            pegasus_node_label_type(node) for node in normalized_nodes
        }
        if len(label_types) != 1:
            raise PegasusTopologyError(
                "A Pegasus topology cannot mix node label families."
            )
        label_type = next(iter(label_types))

        node_set = set(normalized_nodes)
        normalized_edges = tuple(
            sorted(
                {normalize_pegasus_edge(edge) for edge in edges},
                key=lambda edge: (
                    _node_sort_key(edge[0]),
                    _node_sort_key(edge[1]),
                ),
            )
        )
        for first, second in normalized_edges:
            if first not in node_set or second not in node_set:
                raise PegasusTopologyError(
                    "Pegasus edge references a node absent from the graph."
                )
            if pegasus_node_label_type(first) != label_type:
                raise PegasusTopologyError(
                    "Pegasus edge label type differs from graph nodes."
                )

        adjacency = _deeply_immutable_adjacency(
            normalized_nodes,
            normalized_edges,
        )
        connected = _is_connected(normalized_nodes, adjacency)
        if _boolean(require_connected, name="require_connected") and not connected:
            raise PegasusTopologyError(
                "Pegasus topology must be connected for solver execution."
            )

        normalized_source = _string(source, name="source")
        normalized_solver = (
            None if solver_id is None else _validated_solver_id(solver_id)
        )
        normalized_ideal = _boolean(ideal, name="ideal")
        nominal_nodes = nominal_pegasus_qubit_count(normalized_m)
        if normalized_ideal and len(normalized_nodes) != nominal_nodes:
            raise PegasusTopologyError(
                "Ideal Pegasus topology node count must equal "
                f"{nominal_nodes} for m={normalized_m}; "
                f"received {len(normalized_nodes)}."
            )
        if normalized_solver is not None and normalized_ideal:
            raise PegasusTopologyError(
                "A real QPU topology cannot be marked as ideal."
            )

        object.__setattr__(self, "pegasus_m", normalized_m)
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "edges", normalized_edges)
        object.__setattr__(self, "label_type", label_type)
        object.__setattr__(self, "source", normalized_source)
        object.__setattr__(self, "solver_id", normalized_solver)
        object.__setattr__(self, "ideal", normalized_ideal)
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(metadata, name="topology metadata"),
        )
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "connected", connected)

    @property
    def topology_type(self) -> str:
        return PEGASUS_TOPOLOGY_TYPE

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def nominal_node_count(self) -> int:
        return nominal_pegasus_qubit_count(self.pegasus_m)

    @property
    def missing_nominal_nodes(self) -> int:
        return max(0, self.nominal_node_count - self.node_count)

    @property
    def defective(self) -> bool:
        return self.missing_nominal_nodes > 0

    def degree(self, node: object) -> int:
        normalized = normalize_pegasus_node(node)
        try:
            return len(self.adjacency[normalized])
        except KeyError as exc:
            raise PegasusTopologyError(
                f"Node {normalized!r} is not in this topology."
            ) from exc

    def has_node(self, node: object) -> bool:
        try:
            normalized = normalize_pegasus_node(node)
        except (TypeError, PegasusTopologyError):
            return False
        return normalized in self.adjacency

    def has_edge(self, first: object, second: object) -> bool:
        try:
            edge = normalize_pegasus_edge((first, second))
        except (TypeError, PegasusTopologyError):
            return False
        return edge in set(self.edges)

    def require_problem_graph(
        self,
        *,
        nodes: Iterable[object],
        edges: Iterable[object],
    ) -> tuple[tuple[PegasusNode, ...], tuple[PegasusEdge, ...]]:
        """Validate a directly executable physical problem subgraph."""

        problem_nodes = tuple(
            sorted(
                {normalize_pegasus_node(node) for node in nodes},
                key=_node_sort_key,
            )
        )
        if not problem_nodes:
            raise PegasusTopologyError("Problem graph has no nodes.")
        missing_nodes = tuple(
            node for node in problem_nodes if node not in self.adjacency
        )
        if missing_nodes:
            raise PegasusTopologyError(
                f"Problem graph references unavailable Pegasus nodes: "
                f"{missing_nodes}."
            )

        topology_edges = set(self.edges)
        problem_edges = tuple(
            sorted(
                {normalize_pegasus_edge(edge) for edge in edges},
                key=lambda edge: (
                    _node_sort_key(edge[0]),
                    _node_sort_key(edge[1]),
                ),
            )
        )
        missing_edges = tuple(
            edge for edge in problem_edges if edge not in topology_edges
        )
        if missing_edges:
            raise PegasusTopologyError(
                f"Problem graph contains unavailable Pegasus couplers: "
                f"{missing_edges}."
            )
        problem_node_set = set(problem_nodes)
        for first, second in problem_edges:
            if first not in problem_node_set or second not in problem_node_set:
                raise PegasusTopologyError(
                    "Every problem edge endpoint must appear in problem nodes."
                )
        return problem_nodes, problem_edges

    def audit(self) -> PegasusTopologyAudit:
        return PegasusTopologyAudit(
            topology_type=self.topology_type,
            pegasus_m=self.pegasus_m,
            node_count=self.node_count,
            edge_count=self.edge_count,
            label_type=self.label_type,
            connected=self.connected,
            ideal=self.ideal,
            source=self.source,
            solver_id=self.solver_id,
            nominal_node_count=self.nominal_node_count,
            missing_nominal_nodes=self.missing_nominal_nodes,
            metadata={
                "topology_fingerprint": self.fingerprint(),
                **dict(self.metadata),
            },
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-PegasusTopology-v1\0")
        digest.update(
            json.dumps(
                {
                    "topology_type": self.topology_type,
                    "pegasus_m": self.pegasus_m,
                    "nodes": self.nodes,
                    "edges": self.edges,
                    "label_type": self.label_type,
                    "source": self.source,
                    "solver_id": self.solver_id,
                    "ideal": self.ideal,
                    "connected": self.connected,
                    "metadata": dict(self.metadata),
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()

    @classmethod
    def from_graph(
        cls,
        graph: object,
        *,
        expected_m: int | None = None,
        source: str = "ocean_graph",
        solver_id: str | None = None,
        ideal: bool = False,
        require_connected: bool = True,
    ) -> "PegasusTopology":
        """Create a topology from an Ocean/NetworkX-compatible graph."""

        metadata = _graph_mapping(graph)
        _validate_family(metadata)
        pegasus_m = _infer_m(metadata, expected_m=expected_m)
        topology = cls(
            pegasus_m=pegasus_m,
            nodes=_graph_nodes(graph),
            edges=_graph_edges(graph),
            source=source,
            solver_id=solver_id,
            ideal=ideal,
            metadata={
                "graph_metadata": dict(metadata),
                "graph_class": (
                    f"{type(graph).__module__}.{type(graph).__qualname__}"
                ),
            },
            require_connected=require_connected,
        )
        return topology

    @classmethod
    def from_qpu_sampler(
        cls,
        sampler: object,
        *,
        solver_id: str | None = None,
        require_connected: bool = True,
    ) -> "PegasusTopology":
        """Create a physical topology from an explicit real-QPU sampler."""

        validate_sampler_interface(sampler)
        properties = getattr(sampler, "properties", None)
        if not isinstance(properties, Mapping):
            raise PegasusTopologyError(
                "QPU sampler properties must be a mapping."
            )
        topology_type, pegasus_m = _extract_topology_properties(properties)
        _validated_topology_type(topology_type)
        concrete_solver_id = _sampler_solver_id(sampler, solver_id)

        nodelist = getattr(sampler, "nodelist", None)
        edgelist = getattr(sampler, "edgelist", None)
        if nodelist is None or edgelist is None:
            raise PegasusTopologyError(
                "QPU sampler must expose nodelist and edgelist."
            )
        return cls(
            pegasus_m=pegasus_m,
            nodes=tuple(nodelist),
            edges=tuple(edgelist),
            source="pegasus_qpu_sampler",
            solver_id=concrete_solver_id,
            ideal=False,
            metadata={
                "topology_properties": dict(properties.get("topology", {})),
                "chip_id": properties.get("chip_id"),
            },
            require_connected=require_connected,
        )


def _load_pegasus_graph_factory() -> Any:
    try:
        module = importlib.import_module(PEGASUS_GRAPH_FACTORY_MODULE)
    except ImportError as exc:
        raise PegasusTopologyRuntimeError(
            "dwave_networkx is required to construct an ideal Pegasus graph."
        ) from exc
    factory = getattr(module, PEGASUS_GRAPH_FACTORY_NAME, None)
    if factory is None or not callable(factory):
        raise PegasusTopologyRuntimeError(
            "dwave_networkx.pegasus_graph is unavailable."
        )
    return factory


def build_ideal_pegasus_graph(
    pegasus_m: int,
    *,
    coordinates: bool = False,
    nice_coordinates: bool = False,
    fabric_only: bool = True,
) -> object:
    """Build an ideal Pegasus graph through the Ocean graph factory."""

    normalized_m = _validated_pegasus_m(pegasus_m)
    normalized_coordinates = _boolean(coordinates, name="coordinates")
    normalized_nice = _boolean(
        nice_coordinates,
        name="nice_coordinates",
    )
    normalized_fabric = _boolean(fabric_only, name="fabric_only")
    if normalized_coordinates and normalized_nice:
        raise PegasusTopologyError(
            "coordinates and nice_coordinates are mutually exclusive."
        )

    factory = _load_pegasus_graph_factory()
    try:
        graph = factory(
            normalized_m,
            coordinates=normalized_coordinates,
            nice_coordinates=normalized_nice,
            fabric_only=normalized_fabric,
            data=True,
        )
    except Exception as exc:
        raise PegasusTopologyRuntimeError(
            "Ocean failed to construct the requested Pegasus graph."
        ) from exc

    metadata = _graph_mapping(graph)
    _validate_family(metadata)
    _infer_m(metadata, expected_m=normalized_m)
    return graph


def build_ideal_pegasus_topology(
    pegasus_m: int,
    *,
    coordinates: bool = False,
    nice_coordinates: bool = False,
    fabric_only: bool = True,
    require_connected: bool = True,
) -> PegasusTopology:
    """Build and validate one immutable local-emulator topology."""

    graph = build_ideal_pegasus_graph(
        pegasus_m,
        coordinates=coordinates,
        nice_coordinates=nice_coordinates,
        fabric_only=fabric_only,
    )
    return PegasusTopology.from_graph(
        graph,
        expected_m=pegasus_m,
        source="dwave_networkx.pegasus_graph",
        ideal=True,
        require_connected=require_connected,
    )


def topology_from_emulator_config(
    config: object,
    *,
    coordinates: bool = False,
    nice_coordinates: bool = False,
) -> PegasusTopology:
    """Build a local Pegasus topology from ``EmulatorConfig`` contract."""

    backend = _validated_backend_kind(getattr(config, "backend", None))
    if backend != LOCAL_GPU_BACKEND_KIND:
        raise PegasusTopologyError(
            "Emulator topology requires backend='local_sqa_gpu'."
        )
    _validated_topology_type(getattr(config, "topology_type", None))
    if getattr(config, "require_gpu", None) is not True:
        raise PegasusTopologyError("Local emulator must require GPU.")
    if getattr(config, "allow_classical_fallback", None) is not False:
        raise PegasusTopologyError(
            "Local emulator classical fallback must remain disabled."
        )
    pegasus_m = _validated_pegasus_m(getattr(config, "pegasus_m", None))
    return build_ideal_pegasus_topology(
        pegasus_m,
        coordinates=coordinates,
        nice_coordinates=nice_coordinates,
        fabric_only=True,
        require_connected=True,
    )


def topology_from_qpu_config(
    config: object,
    sampler: object,
) -> PegasusTopology:
    """Validate ``PegasusQPUConfig`` and bind it to a physical sampler."""

    backend = _validated_backend_kind(getattr(config, "backend", None))
    if backend != PEGASUS_QPU_BACKEND_KIND:
        raise PegasusTopologyError(
            "QPU topology requires backend='pegasus_qpu'."
        )
    _validated_topology_type(getattr(config, "topology_type", None))
    if getattr(config, "require_explicit_solver_id", None) is not True:
        raise PegasusTopologyError(
            "QPU configuration must require an explicit solver_id."
        )
    if getattr(config, "allow_solver_fallback", None) is not False:
        raise PegasusTopologyError("QPU solver fallback is prohibited.")
    if getattr(config, "reject_zephyr", None) is not True:
        raise PegasusTopologyError("QPU configuration must reject Zephyr.")
    solver_id = _validated_solver_id(getattr(config, "solver_id", None))
    return PegasusTopology.from_qpu_sampler(
        sampler,
        solver_id=solver_id,
        require_connected=True,
    )


__all__ = [
    "PEGASUS_GRAPH_FACTORY_MODULE",
    "PEGASUS_GRAPH_FACTORY_NAME",
    "PEGASUS_LINEAR_LABEL_TYPE",
    "PEGASUS_COORDINATE_LABEL_TYPE",
    "PEGASUS_NICE_COORDINATE_LABEL_TYPE",
    "SUPPORTED_PEGASUS_LABEL_TYPES",
    "PEGASUS_COORDINATE_LENGTH",
    "PEGASUS_NICE_COORDINATE_LENGTH",
    "PEGASUS_NOMINAL_QUBITS_PER_M_PRODUCT",
    "PegasusCoordinate",
    "PegasusNiceCoordinate",
    "PegasusNode",
    "PegasusEdge",
    "PegasusTopologyError",
    "PegasusTopologyRuntimeError",
    "normalize_pegasus_node",
    "pegasus_node_label_type",
    "normalize_pegasus_edge",
    "nominal_pegasus_qubit_count",
    "PegasusTopologyAudit",
    "PegasusTopology",
    "build_ideal_pegasus_graph",
    "build_ideal_pegasus_topology",
    "topology_from_emulator_config",
    "topology_from_qpu_config",
]
