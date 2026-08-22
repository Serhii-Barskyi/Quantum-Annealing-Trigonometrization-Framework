"""Fixed-embedding Pegasus control evaluation for Simulator and real QPU evidence.

Both evidence notebooks use this module so that the physical control domain,
embedding, chain strength, response metrics, and accounting remain identical.
The only backend difference is the response source: calibration-resolved CUDA
SQA for the Simulator notebook or an explicitly pinned Pegasus QPU for the
hardware notebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Callable
import hashlib
import json
import math
import time

import numpy as np

from benchmarks import reference_competitors as rc
from experiments_dwave.operator_phase import ABReferenceCurve, operator_action_coordinates
from dwave_backend.pegasus_fabric_v53 import validate_pegasus_solver_id_v53, pegasus_solver_family_v53


class PegasusControlError(RuntimeError):
    pass


def _hash_json(payload: Mapping[str, Any], prefix: str) -> str:
    raw=json.dumps(dict(payload),sort_keys=True,separators=(",",":"),default=str).encode()
    return hashlib.sha256(prefix.encode()+b"\0"+raw).hexdigest()


def _logical_edges(bqm: Any) -> list[tuple[Any, Any]]:
    edges=list(bqm.quadratic)
    # Ensure isolated variables participate in embedding through a harmless
    # self-free source graph by linking them to a deterministic anchor when needed.
    if not edges and len(bqm.variables)>1:
        vars_=list(bqm.variables)
        edges=[(vars_[0],v) for v in vars_[1:]]
    return edges


@dataclass
class FixedPegasusControlBackend:
    mode: str
    bundle: Any
    logical_bqm: Any
    calibration: ABReferenceCurve
    embedding: Mapping[Any, list[Any]]
    chain_strength: float
    seed: int
    simulator_trotter_replicas: int = 96
    simulator_sweeps: int = 4000
    simulator_burn_in_sweeps: int = 1000
    simulator_memory_fraction: float = 0.65

    @classmethod
    def build(
        cls,
        *,
        mode: str,
        bundle: Any,
        logical_bqm: Any,
        calibration: ABReferenceCurve,
        seed: int,
        embedding_timeout_seconds: int = 180,
        chain_strength: float | None = None,
        frozen_embedding: Mapping[Any, list[Any]] | None = None,
    ) -> "FixedPegasusControlBackend":
        normalized=str(mode).strip().lower()
        if normalized not in {"simulator","qpu"}:
            raise PegasusControlError("mode must be simulator or qpu")
        expected_bundle_mode="local_sqa_gpu" if normalized=="simulator" else "pegasus_qpu"
        if getattr(bundle,"mode",None)!=expected_bundle_mode:
            raise PegasusControlError(f"bundle mode mismatch: expected {expected_bundle_mode!r}")
        if getattr(bundle.topology,"pegasus_m",None)!=16:
            raise PegasusControlError("Only Pegasus P16 is admissible")
        if normalized=="qpu":
            sid=validate_pegasus_solver_id_v53(getattr(bundle,"solver_id",None))
            if pegasus_solver_family_v53(sid) != calibration.family:
                raise PegasusControlError("QPU solver family and A/B calibration family must match")
        try:
            import minorminer
            from dwave.embedding import verify_embedding
            from dwave.embedding.chain_strength import uniform_torque_compensation
        except Exception as exc:  # pragma: no cover - target environment dependency
            raise PegasusControlError("D-Wave embedding dependencies are unavailable") from exc
        source_edges=_logical_edges(logical_bqm)
        target_edges=list(getattr(bundle.raw_sampler,"edgelist",()))
        if not target_edges:
            raise PegasusControlError("Pegasus raw sampler exposes no target edges")
        variables=set(logical_bqm.variables)
        if frozen_embedding is None:
            embedding=minorminer.find_embedding(source_edges,target_edges,timeout=int(embedding_timeout_seconds),random_seed=int(seed))
        else:
            embedding={k:list(v) for k,v in dict(frozen_embedding).items()}
        if set(embedding)!=variables:
            missing=sorted(map(str,variables-set(embedding)))
            extra=sorted(map(str,set(embedding)-variables))
            raise PegasusControlError(f"Embedding variable mismatch; missing={missing[:8]}, extra={extra[:8]}")
        if not verify_embedding(embedding,source_edges,target_edges):
            raise PegasusControlError("Frozen embedding failed D-Wave verification")
        strength=float(uniform_torque_compensation(logical_bqm,embedding)) if chain_strength is None else float(chain_strength)
        if not math.isfinite(strength) or strength<=0:
            raise PegasusControlError("chain_strength must be finite and positive")
        return cls(normalized,bundle,logical_bqm,calibration,{k:list(v) for k,v in embedding.items()},strength,int(seed))

    @property
    def embedding_fingerprint(self) -> str:
        canonical={str(k):list(map(int,v)) for k,v in sorted(self.embedding.items(),key=lambda kv:str(kv[0]))}
        return _hash_json({"embedding":canonical,"chain_strength":self.chain_strength},"CSSF-Pegasus-FixedEmbedding-v53")

    def embedding_manifest(self) -> dict[str, Any]:
        return {
            "embedding": {str(k): list(map(int,v)) for k,v in sorted(self.embedding.items(),key=lambda kv:str(kv[0]))},
            "chain_strength": float(self.chain_strength),
            "embedding_fingerprint": self.embedding_fingerprint,
            "topology_fingerprint": self.bundle.topology.fingerprint(),
            "solver_id": getattr(self.bundle,"solver_id",None),
            "calibration_family": self.calibration.family,
            "calibration_sha256": self.calibration.source_sha256,
        }

    def _embed(self) -> Any:
        try:
            from dwave.embedding import embed_bqm
        except Exception as exc:  # pragma: no cover
            raise PegasusControlError("dwave.embedding.embed_bqm is unavailable") from exc
        return embed_bqm(self.logical_bqm,self.embedding,self.bundle.raw_sampler.adjacency,self.chain_strength)

    def evaluate_schedule(
        self,
        anneal_t_us: np.ndarray,
        anneal_s: np.ndarray,
        *,
        num_reads: int,
        elite_threshold: float,
        feasibility: Callable[[Mapping[Any,int]], bool] | None = None,
        success_energy: float | None = None,
        label: str = "CSSF(QA) schedule evaluation",
        control: np.ndarray | None = None,
        sampling_seed: int | None = None,
    ) -> dict[str, Any]:
        t=np.asarray(anneal_t_us,dtype=float)
        s=np.asarray(anneal_s,dtype=float)
        if t.ndim!=1 or s.ndim!=1 or len(t)!=len(s) or len(t)<3:
            raise PegasusControlError("anneal_t_us and anneal_s must be equal-length one-dimensional arrays with at least three points")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(s)):
            raise PegasusControlError("anneal schedule must contain only finite values")
        if abs(float(t[0]))>1e-12 or np.any(np.diff(t)<=0):
            raise PegasusControlError("anneal_t_us must start at zero and be strictly increasing")
        if abs(float(s[0]))>1e-12 or abs(float(s[-1])-1.0)>1e-12 or np.any(np.diff(s)<-1e-12):
            raise PegasusControlError("forward anneal_s must be monotone from 0 to 1")
        if np.any(s< -1e-12) or np.any(s>1.0+1e-12):
            raise PegasusControlError("anneal_s must remain in [0,1]")
        action=operator_action_coordinates(t,s,self.calibration,n_segments=8)
        embedded=self._embed()
        start=time.perf_counter()
        if self.mode=="simulator":
            raw=rc.cuda_schedule_sqa_embedded_bqm(
                embedded,anneal_t_us=t,anneal_s=s,
                calibration={"family":self.calibration.family,"sha256":self.calibration.source_sha256,
                             "s":self.calibration.s,"A_GHz":self.calibration.A_GHz,"B_GHz":self.calibration.B_GHz},
                num_reads=int(num_reads),trotter_replicas=self.simulator_trotter_replicas,
                sweeps=self.simulator_sweeps,burn_in_sweeps=self.simulator_burn_in_sweeps,
                seed=int(self.seed if sampling_seed is None else sampling_seed),memory_fraction=float(self.simulator_memory_fraction),
                raw_pegasus_sampler=self.bundle.raw_sampler,
            )
        else:
            validate=getattr(self.bundle.raw_sampler,"validate_anneal_schedule",None)
            schedule=[[float(tt),float(ss)] for tt,ss in zip(t,s,strict=True)]
            if callable(validate):
                validate(schedule)
            raw=self.bundle.raw_sampler.sample(
                embedded,num_reads=int(num_reads),anneal_schedule=schedule,label=str(label)
            )
        elapsed=time.perf_counter()-start
        try:
            from dwave.embedding import unembed_sampleset
        except Exception as exc:  # pragma: no cover
            raise PegasusControlError("dwave.embedding.unembed_sampleset is unavailable") from exc
        logical=unembed_sampleset(raw,self.embedding,self.logical_bqm,chain_break_method=None,chain_break_fraction=True)
        energies=np.asarray(logical.record.energy,dtype=float)
        occurrences=np.asarray(logical.record.num_occurrences,dtype=np.int64)
        total=int(np.sum(occurrences))
        if total!=int(num_reads):
            raise PegasusControlError(f"SampleSet occurrence total {total} != requested reads {num_reads}")
        mean_energy=float(np.average(energies,weights=occurrences))
        variance=float(np.average((energies-mean_energy)**2,weights=occurrences))
        expanded=np.repeat(energies,occurrences)
        q05=float(np.quantile(expanded,0.05))
        cvar05=float(np.mean(expanded[expanded<=q05]))
        elite_probability=float(np.sum(occurrences[energies<=float(elite_threshold)])/total)
        feasible_reads=0
        if feasibility is None:
            feasibility_probability=1.0
        else:
            for sample,occ in zip(logical.samples(),occurrences,strict=True):
                if feasibility(sample): feasible_reads+=int(occ)
            feasibility_probability=float(feasible_reads/total)
        best=float(np.min(energies))
        if success_energy is None:
            success_probability=float(np.sum(occurrences[np.isclose(energies,best,rtol=0,atol=1e-12)])/total)
            success_reference="best_observed_diagnostic"
        else:
            success_probability=float(np.sum(occurrences[energies<=float(success_energy)+1e-10])/total)
            success_reference="declared_exact_or_certified_target"
        info=dict(getattr(raw,"info",{}) or {})
        return {
            "control":None if control is None else np.asarray(control,dtype=float),
            "schedule_t_us":t,"schedule_s":s,"operator_action":action,
            "mean_energy":mean_energy,"energy_variance":variance,"energy_quantile_05":q05,"cvar_05":cvar05,
            "feasibility_probability":feasibility_probability,"elite_probability":elite_probability,
            "success_probability":success_probability,"success_reference":success_reference,
            "best_energy":best,"num_reads":total,"elapsed_seconds":float(elapsed),
            "embedding_fingerprint":self.embedding_fingerprint,"chain_strength":self.chain_strength,
            "calibration_family":self.calibration.family,"calibration_sha256":self.calibration.source_sha256,
            "backend_info":info,"sampling_seed":None if self.mode=="qpu" else int(self.seed if sampling_seed is None else sampling_seed),
            "sampleset":logical,
        }

    def evaluate(
        self,
        control: np.ndarray,
        *,
        order: int,
        num_reads: int,
        elite_threshold: float,
        feasibility: Callable[[Mapping[Any,int]], bool] | None = None,
        success_energy: float | None = None,
        label: str = "CSSF(QA) control evaluation",
        sampling_seed: int | None = None,
    ) -> dict[str, Any]:
        t,s=rc.fourier_forward_schedule(control,order=int(order),grid_points=129,reject_nonmonotone=True)
        return self.evaluate_schedule(
            t,s,num_reads=int(num_reads),elite_threshold=float(elite_threshold),
            feasibility=feasibility,success_energy=success_energy,label=label,control=np.asarray(control,dtype=float),sampling_seed=sampling_seed,
        )



@dataclass
class BoundPegasusResponseEvaluator:
    """Callable response evaluator with a query-free operator-action map.

    The same interface is consumed by CSSF and all matched competitors.  CSSF
    may evaluate ``operator_action(control)`` for candidate acquisition without
    spending an annealer query; only ``__call__`` reaches the SQA/QPU backend.
    """
    backend: FixedPegasusControlBackend
    order: int
    elite_threshold: float
    feasibility: Callable[[Mapping[Any,int]], bool] | None = None
    success_energy: float | None = None
    label: str = "CSSF(QA) matched control evaluation"

    def operator_action(self, control: np.ndarray) -> np.ndarray:
        t,s=rc.fourier_forward_schedule(control,order=int(self.order),grid_points=129,reject_nonmonotone=True)
        return operator_action_coordinates(t,s,self.backend.calibration,n_segments=8)

    def __call__(self, control: np.ndarray, *, num_reads: int, sampling_seed: int | None = None) -> dict[str, Any]:
        return self.backend.evaluate(
            control,order=int(self.order),num_reads=int(num_reads),elite_threshold=float(self.elite_threshold),
            feasibility=self.feasibility,success_energy=self.success_energy,label=self.label,sampling_seed=sampling_seed,
        )
