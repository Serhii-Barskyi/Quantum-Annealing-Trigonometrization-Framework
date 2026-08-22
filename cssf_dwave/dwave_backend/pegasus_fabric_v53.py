"""CSSF(QA) v53 Pegasus topology and 2026 solver-identity compatibility.

The frozen v51 topology module remains byte-identical.  This adapter separates:

* the full nominal P_m prelattice;
* the connected programmable Pegasus fabric used by local SQA;
* the real QPU working graph;
* current post-2026-04-06 solver IDs (no minor suffix) from historical IDs.
"""
from __future__ import annotations

from typing import Mapping

from dwave_backend.topology import (
    PegasusTopology,
    PegasusTopologyError,
    build_ideal_pegasus_graph,
    nominal_pegasus_qubit_count,
)
from dwave_backend import validate_sampler_interface

LOCAL_GPU_BACKEND_KIND = "local_sqa_gpu"
PEGASUS_TOPOLOGY_TYPE = "pegasus"
PEGASUS_SOLVER_FAMILIES_V53 = ("Advantage_system4", "Advantage_system6")


def validate_pegasus_solver_id_v53(value: object) -> str:
    """Accept exact current IDs and historical minor-version IDs only."""
    if not isinstance(value, str):
        raise PegasusTopologyError("solver_id must be a string.")
    solver_id = value.strip()
    if not solver_id:
        raise PegasusTopologyError("solver_id must not be empty.")
    lowered = solver_id.lower()
    if "zephyr" in lowered or solver_id.startswith("Advantage2_"):
        raise PegasusTopologyError("Only Pegasus Advantage_system4/System6 are admissible.")
    if not any(
        solver_id == family or solver_id.startswith(family + ".")
        for family in PEGASUS_SOLVER_FAMILIES_V53
    ):
        raise PegasusTopologyError(
            "solver_id must be Advantage_system4/Advantage_system6 or a historical "
            "minor-version ID in the same Pegasus family."
        )
    return solver_id


def pegasus_solver_family_v53(value: object) -> str:
    solver_id = validate_pegasus_solver_id_v53(value)
    for family in PEGASUS_SOLVER_FAMILIES_V53:
        if solver_id == family or solver_id.startswith(family + "."):
            return family
    raise AssertionError("validated solver must belong to an allowed family")


def programmable_pegasus_fabric_qubit_count(pegasus_m: int) -> int:
    """Return Ocean's default-offset ``fabric_only=True`` node count."""
    if isinstance(pegasus_m, bool) or not isinstance(pegasus_m, int) or pegasus_m < 2:
        raise PegasusTopologyError("pegasus_m must be an integer >= 2.")
    return 24 * pegasus_m * (pegasus_m - 1) - 8 * (pegasus_m - 1)


def topology_from_emulator_config_v53(
    config: object,
    *,
    coordinates: bool = False,
    nice_coordinates: bool = False,
) -> PegasusTopology:
    """Build the connected synthetic programmable fabric for local SQA."""
    raw_backend = getattr(config, "backend", "")
    backend = str(getattr(raw_backend, "value", raw_backend)).strip().lower()
    if backend != LOCAL_GPU_BACKEND_KIND:
        raise PegasusTopologyError("Emulator topology requires backend='local_sqa_gpu'.")
    topology_type = str(getattr(config, "topology_type", "")).strip().lower()
    if topology_type != PEGASUS_TOPOLOGY_TYPE:
        raise PegasusTopologyError("CSSF v53 local emulator requires Pegasus topology.")
    if getattr(config, "require_gpu", None) is not True:
        raise PegasusTopologyError("Local emulator must require GPU.")
    if getattr(config, "allow_classical_fallback", None) is not False:
        raise PegasusTopologyError("Local emulator classical fallback must remain disabled.")

    pegasus_m = getattr(config, "pegasus_m", None)
    if isinstance(pegasus_m, bool) or not isinstance(pegasus_m, int) or pegasus_m < 2:
        raise PegasusTopologyError("pegasus_m must be an integer >= 2.")

    graph = build_ideal_pegasus_graph(
        pegasus_m,
        coordinates=coordinates,
        nice_coordinates=nice_coordinates,
        fabric_only=True,
    )
    nodes = tuple(graph.nodes)
    edges = tuple(graph.edges)
    expected_fabric = programmable_pegasus_fabric_qubit_count(pegasus_m)
    nominal = nominal_pegasus_qubit_count(pegasus_m)
    if len(nodes) != expected_fabric:
        raise PegasusTopologyError(
            "Programmable Pegasus fabric node count mismatch: "
            f"expected {expected_fabric} for m={pegasus_m}, received {len(nodes)}."
        )

    topology = PegasusTopology(
        pegasus_m=pegasus_m,
        nodes=nodes,
        edges=edges,
        source="dwave_networkx.pegasus_graph:fabric_only",
        solver_id=None,
        # Frozen v51 `ideal=True` means *all full nominal nodes*, not
        # "synthetic defect-free programmable fabric".
        ideal=False,
        metadata={
            "schema": "CSSF-Pegasus-Topology-Semantics-v53",
            "topology_role": "synthetic_programmable_fabric",
            "fabric_only": True,
            "synthetic": True,
            "physical_defects": False,
            "nominal_full_node_count": nominal,
            "programmable_fabric_node_count": expected_fabric,
            "structurally_excluded_nominal_nodes": nominal - expected_fabric,
            "legacy_ideal_flag_semantics": "full_nominal_graph_only",
        },
        require_connected=True,
    )
    if not topology.connected:
        raise PegasusTopologyError("Programmable Pegasus fabric must be connected.")
    return topology


def _sampler_solver_name_v53(sampler: object) -> str | None:
    solver = getattr(sampler, "solver", None)
    for attribute in ("name", "id"):
        value = getattr(solver, attribute, None) if solver is not None else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    properties = getattr(sampler, "properties", None)
    if isinstance(properties, Mapping):
        for key in ("chip_id", "name"):
            value = properties.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def topology_from_qpu_sampler_v53(
    sampler: object,
    *,
    solver_id: object,
    require_connected: bool = True,
) -> PegasusTopology:
    """Bind the *actual* real-QPU working graph to a current/historical solver ID.

    A narrow versioned compatibility shim is required because v51's frozen
    validator predates D-Wave's 2026 removal of minor-version suffixes.  The
    object is first constructed with ``solver_id=None`` through the frozen
    topology implementation and only then receives the already-validated v53
    identity.  No topology node/coupler is synthesized or repaired.
    """
    validate_sampler_interface(sampler)
    expected = validate_pegasus_solver_id_v53(solver_id)
    actual = _sampler_solver_name_v53(sampler)
    if actual is None:
        raise PegasusTopologyError("QPU sampler does not expose a concrete solver identity.")
    actual = validate_pegasus_solver_id_v53(actual)
    if actual != expected:
        raise PegasusTopologyError(
            f"Ocean selected {actual!r}, but CSSF pinned {expected!r}."
        )

    properties = getattr(sampler, "properties", None)
    if not isinstance(properties, Mapping):
        raise PegasusTopologyError("QPU sampler properties must be a mapping.")
    topology_props = properties.get("topology")
    if not isinstance(topology_props, Mapping):
        raise PegasusTopologyError("QPU sampler must expose properties['topology'].")
    if str(topology_props.get("type", "")).strip().lower() != PEGASUS_TOPOLOGY_TYPE:
        raise PegasusTopologyError("Only real Pegasus topology is admissible.")
    shape = topology_props.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise PegasusTopologyError("QPU topology.shape must expose Pegasus m.")
    try:
        pegasus_m = int(shape[0])
    except (TypeError, ValueError) as exc:
        raise PegasusTopologyError("QPU topology.shape[0] must be an integer.") from exc
    if pegasus_m != 16:
        raise PegasusTopologyError(f"Commercial CSSF target is Pegasus P16; received P{pegasus_m}.")

    nodelist = getattr(sampler, "nodelist", None)
    edgelist = getattr(sampler, "edgelist", None)
    if nodelist is None or edgelist is None:
        raise PegasusTopologyError("QPU sampler must expose nodelist and edgelist.")

    topology = PegasusTopology(
        pegasus_m=pegasus_m,
        nodes=tuple(nodelist),
        edges=tuple(edgelist),
        source="pegasus_qpu_working_graph_v53",
        solver_id=None,  # bypass only the stale v51 name grammar
        ideal=False,
        metadata={
            "schema": "CSSF-Pegasus-Physical-WorkingGraph-v53",
            "topology_role": "real_qpu_working_graph",
            "solver_id_v53": expected,
            "solver_family": pegasus_solver_family_v53(expected),
            "topology_properties": dict(topology_props),
            "chip_id": properties.get("chip_id"),
            "graph_id": properties.get("graph_id"),
            "node_count": len(tuple(nodelist)),
            "edge_count": len(tuple(edgelist)),
            "physical_defects": True,
            "fabricated_programmable_node_count": programmable_pegasus_fabric_qubit_count(16),
        },
        require_connected=require_connected,
    )
    # The frozen object is immutable by normal API.  This explicit assignment is
    # restricted to this compatibility adapter after exact identity matching.
    object.__setattr__(topology, "solver_id", expected)
    return topology


__all__ = [
    "PEGASUS_SOLVER_FAMILIES_V53",
    "validate_pegasus_solver_id_v53",
    "pegasus_solver_family_v53",
    "programmable_pegasus_fabric_qubit_count",
    "topology_from_emulator_config_v53",
    "topology_from_qpu_sampler_v53",
]
