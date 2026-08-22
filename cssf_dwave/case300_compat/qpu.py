from __future__ import annotations
from dataclasses import dataclass
import os
import time
from typing import Any
import numpy as np
from .annealing import compute_phi

_ALLOWED = ("Advantage_system4", "Advantage_system6")

def validate_solver_id(solver_id: str | None) -> str:
    sid = (solver_id or "").strip()
    if not sid or not sid.startswith(_ALLOWED):
        raise ValueError(
            "An explicit Pegasus solver is required: Advantage_system4 or "
            "Advantage_system6 (a provider suffix is accepted)."
        )
    return sid

def get_sampler(backend: str = "qpu", *, solver_id: str | None = None):
    if backend != "qpu":
        raise RuntimeError("The monograph notebook is QPU-only; no simulator fallback is permitted.")
    sid = validate_solver_id(solver_id or os.environ.get("CSSF_DWAVE_SOLVER_ID"))
    try:
        from dwave.system import DWaveSampler
    except Exception as exc:
        raise RuntimeError("dwave-ocean-sdk is required for live QPU execution") from exc
    sampler = DWaveSampler(solver=sid)
    solver = sampler.solver
    if not bool(getattr(solver, "qpu", False)) or not bool(getattr(solver, "online", False)):
        raise RuntimeError(f"Solver {getattr(solver, 'name', sid)!r} is not an online QPU")
    topology = sampler.properties.get("topology", {}).get("type")
    if str(topology).lower() != "pegasus":
        raise RuntimeError(f"Solver {sid!r} is not Pegasus (topology={topology!r})")
    return sampler

def compute_lsf_offsets(lsf_weights, *, delta_max: float = 0.25) -> np.ndarray:
    w = np.abs(np.asarray(lsf_weights, dtype=float).ravel())
    if w.size == 0: return w
    lo, hi = float(np.min(w)), float(np.max(w))
    norm = np.zeros_like(w) if hi <= lo else (w - lo) / (hi - lo)
    return -abs(float(delta_max)) * norm

def compute_lsf_h_bias(lsf_weights, *, gamma: float = 0.05) -> np.ndarray:
    return -abs(float(gamma)) * np.abs(np.asarray(lsf_weights, dtype=float).ravel())

@dataclass(slots=True)
class DWaveResult:
    buses_opt: list[int]
    energy_best: float
    r_quality: float
    n_occurrences: int
    total_reads: int
    timing_s: float
    backend: str
    feasible: bool
    offsets_used: np.ndarray | None
    phi: np.ndarray | None
    solver_id: str
    embedding: dict[int, list[int]] | None = None


def _unwrap_qpu(sampler):
    cur = sampler
    for _ in range(6):
        if hasattr(cur, "edgelist") and hasattr(cur, "properties") and hasattr(cur, "solver"):
            return cur
        children = getattr(cur, "children", None)
        if children:
            cur = children[0]
        else:
            break
    return None

def solve_ising_dwave(ising, lsf_weights, brute_force_ref, *, backend="qpu", num_reads=4096,
                      delta_max=0.25, gamma_lsf=0.0, sampler=None,
                      offsets_override=None, return_phi=False, schedule=None,
                      annealing_time_us=20.0, s_freeze=0.7, chain_strength=None):
    if backend != "qpu":
        raise RuntimeError("QPU-only notebook: backend substitution is forbidden")
    if sampler is None:
        sampler = get_sampler("qpu")
    qpu = _unwrap_qpu(sampler) or sampler
    sid = validate_solver_id(getattr(getattr(qpu, "solver", None), "name", os.environ.get("CSSF_DWAVE_SOLVER_ID")))
    if str(getattr(qpu, "properties", {}).get("topology", {}).get("type", "")).lower() != "pegasus":
        raise RuntimeError("Live sampler topology is not Pegasus")
    try:
        import minorminer
        from dwave.system import FixedEmbeddingComposite
    except Exception as exc:
        raise RuntimeError("dwave-ocean-sdk/minorminer is required for Pegasus embedding") from exc

    K = int(ising.K)
    h_orig = np.asarray(ising.h, dtype=float).reshape(K)
    Jmat = np.asarray(ising.J, dtype=float).reshape(K, K)
    logical_edges = [(i, j) for i in range(K) for j in range(i + 1, K) if abs(Jmat[i, j]) > 0]
    source_edges = logical_edges + [(i, i) for i in range(K)]
    target_edges = list(qpu.edgelist)
    emb = minorminer.find_embedding(source_edges, target_edges, random_seed=20260822)
    if len(emb) != K or any(not emb.get(i) for i in range(K)):
        raise RuntimeError(f"No complete Pegasus embedding found for K={K}")
    fixed = FixedEmbeddingComposite(qpu, emb)

    h_submit = h_orig + compute_lsf_h_bias(lsf_weights, gamma=gamma_lsf)
    Jdict = {(i, j): float(Jmat[i, j]) for i, j in logical_edges}
    logical_offsets = (np.asarray(offsets_override, dtype=float).reshape(K)
                       if offsets_override is not None else compute_lsf_offsets(lsf_weights, delta_max=delta_max))
    nqubits = int(qpu.properties.get("num_qubits", max(qpu.nodelist) + 1))
    physical_offsets = [0.0] * nqubits
    for logical, chain in emb.items():
        for physical in chain:
            if 0 <= int(physical) < nqubits:
                physical_offsets[int(physical)] = float(logical_offsets[int(logical)])

    kwargs: dict[str, Any] = {
        "num_reads": int(num_reads), "annealing_time": float(annealing_time_us),
        "anneal_offsets": physical_offsets,
    }
    if chain_strength is not None:
        kwargs["chain_strength"] = float(chain_strength)
    t0 = time.perf_counter()
    sampleset = fixed.sample_ising({i: float(h_submit[i]) for i in range(K)}, Jdict, **kwargs)
    wall = time.perf_counter() - t0
    vars_order = list(sampleset.variables)
    occurrences = np.asarray(sampleset.record.num_occurrences, dtype=int)
    total_reads = int(np.sum(occurrences))
    if total_reads != int(num_reads):
        raise RuntimeError(f"Returned occurrences {total_reads} != requested reads {num_reads}")

    best_idx = None; best_e = float("inf"); best_x = None
    for r, sample in enumerate(sampleset.record.sample):
        zmap = {int(v): float(sample[k]) for k, v in enumerate(vars_order)}
        z = np.array([zmap[i] for i in range(K)], dtype=float)
        x = (1.0 - z) / 2.0
        if not np.all((x == 0.0) | (x == 1.0)) or int(np.sum(x)) != int(ising.B_max):
            continue
        e = float(ising.const + np.dot(h_orig, z))
        for i, j in logical_edges:
            e += float(Jmat[i, j] * z[i] * z[j])
        if e < best_e:
            best_e, best_idx, best_x = e, r, x
    if best_idx is None:
        raise RuntimeError("QPU response contains no exact-cardinality feasible sample")
    buses = [int(ising.candidates[i]) for i in np.flatnonzero(best_x > 0.5)]
    eopt = brute_force_ref.get("energy_opt") if isinstance(brute_force_ref, dict) else None
    eworst = brute_force_ref.get("energy_worst") if isinstance(brute_force_ref, dict) else None
    if eopt is not None and eworst is not None and float(eworst) > float(eopt):
        rq = float(np.clip(1.0 - (best_e - float(eopt)) / (float(eworst) - float(eopt)), 0.0, 1.0))
    else:
        rq = float("nan")
    timing = sampleset.info.get("timing", {}) if isinstance(sampleset.info, dict) else {}
    qpu_us = timing.get("qpu_access_time") or timing.get("qpu_sampling_time")
    timing_s = float(qpu_us) / 1e6 if qpu_us is not None else float(wall)
    phi = None
    if return_phi:
        if schedule is None:
            raise ValueError("schedule is required when return_phi=True")
        phi = compute_phi(logical_offsets, schedule, annealing_time_us, s_freeze)
    return DWaveResult(
        buses_opt=buses, energy_best=float(best_e), r_quality=rq,
        n_occurrences=int(occurrences[best_idx]), total_reads=total_reads,
        timing_s=timing_s, backend="pegasus_qpu", feasible=True,
        offsets_used=logical_offsets.copy(), phi=phi, solver_id=sid,
        embedding={int(k): [int(v) for v in chain] for k, chain in emb.items()},
    )
