"""Versioned QPU-config compatibility for current 2026 Pegasus solver names.

The frozen v51 Pydantic schema is intentionally left byte-identical.  It can
load the historical YAML defaults but its solver-id validator predates D-Wave's
2026 removal of minor-version suffixes.  This tiny runtime view carries the
same policy fields while validating the solver identity through the v53
Pegasus adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dwave_backend.pegasus_fabric_v53 import validate_pegasus_solver_id_v53


@dataclass(frozen=True, slots=True)
class PegasusQPUConfigV53:
    enabled: bool
    backend: str
    topology_type: str
    solver_id: str
    require_explicit_solver_id: bool
    reject_zephyr: bool
    allow_solver_fallback: bool
    dry_run: bool
    num_reads: int
    annealing_time: float


def qpu_config_v53(base: Any, *, solver_id: str, live_qpu: bool) -> PegasusQPUConfigV53:
    sid = validate_pegasus_solver_id_v53(solver_id)
    num_reads = int(getattr(base, "num_reads"))
    annealing_time = float(getattr(base, "annealing_time"))
    if num_reads < 1:
        raise ValueError("num_reads must be positive")
    if annealing_time <= 0.0:
        raise ValueError("annealing_time must be positive")
    return PegasusQPUConfigV53(
        enabled=True,
        backend="pegasus_qpu",
        topology_type="pegasus",
        solver_id=sid,
        require_explicit_solver_id=True,
        reject_zephyr=True,
        allow_solver_fallback=False,
        dry_run=not bool(live_qpu),
        num_reads=num_reads,
        annealing_time=annealing_time,
    )


__all__ = ["PegasusQPUConfigV53", "qpu_config_v53"]
