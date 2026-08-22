"""Restartable evidence tasks shared by the CSSF Simulator and Pegasus notebooks.

Every function performs one scientifically declared unit of work and persists
its result atomically.  Expensive annealer campaigns advance by one control
query at a time so Colab runtime boundaries never trigger silent reductions in
reads, sweeps, model size, optimizer state, or competitor fidelity.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import time

import numpy as np

from benchmarks import reference_competitors as rc
from experiments_dwave.benchmark_protocol import MatchedControlProtocol, TARGET_NAMES
from experiments_dwave.evidence_program import atomic_json, worldline_schedule_for_assets
from experiments_dwave.sequential_campaign import (
    AtomicCampaignStore,
    append_observation,
    campaign_fingerprint,
    campaign_summary,
    next_proposal,
)
from experiments_dwave.surrogate_validation import audit_surrogates, validation_design


class EvidenceTaskError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_jsonable(v) for v in value]
    return value


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists(): return default
    return json.loads(path.read_text(encoding="utf-8"))


def collect_surrogate_response_step(
    evaluator: Any,
    *,
    protocol: MatchedControlProtocol,
    output_path: str | Path,
    control_index: int | None = None,
) -> dict[str, Any]:
    """Collect one member of the frozen 96-control QA-response validation design."""
    path=Path(output_path); design=validation_design(protocol)
    payload=_load_json(path,{"schema":"CSSF-QA-SURROGATE-CORPUS-v38","records":[]})
    records=list(payload.get("records",[]))
    existing={int(r["design_index"]) for r in records}
    if control_index is None:
        pending=[i for i in range(len(design)) if i not in existing]
        if not pending:
            return {"complete":True,"records":len(records),"path":str(path)}
        idx=pending[0]
    else:
        idx=int(control_index)
        if not 0<=idx<len(design): raise EvidenceTaskError("control_index is outside the frozen surrogate design")
        if idx in existing:
            row=next(r for r in records if int(r["design_index"])==idx)
            return {"complete":len(records)==len(design),"already_present":True,"record":row,"path":str(path)}
    row=design[idx]; control=np.asarray(row["control"],dtype=float)
    response=evaluator(control,num_reads=protocol.reads_per_control)
    record={
        "design_index":idx,"control_id":row["control_id"],"partition":row["partition"],"control":control.tolist(),
        "response":{k:_jsonable(v) for k,v in response.items() if k!="sampleset"},
    }
    records.append(record); records.sort(key=lambda r:int(r["design_index"]))
    payload={"schema":"CSSF-QA-SURROGATE-CORPUS-v38","protocol":asdict(protocol),"records":records,"complete":len(records)==len(design)}
    atomic_json(path,payload)
    return {"complete":payload["complete"],"records":len(records),"latest":record,"path":str(path)}


def run_surrogate_audit(*,corpus_path:str|Path,project_root:str|Path,output_path:str|Path,target:str="elite_probability") -> dict[str,Any]:
    payload=_load_json(Path(corpus_path),{})
    records=payload.get("records",[])
    result=audit_surrogates(records,project_root=str(project_root),target=target)
    atomic_json(output_path,result)
    return result


def advance_matched_campaign(
    evaluator: Any,
    *,
    method: str,
    protocol: MatchedControlProtocol,
    project_root: str | Path,
    campaign_root: str | Path,
    backend_identity: Mapping[str,Any],
    problem_fingerprint: str,
    candidate_pool_size: int = 4096,
) -> dict[str, Any]:
    """Advance one full matched optimizer by exactly one annealer query."""
    allowed={"CSSF-full","GP+EI-full","Finzgar-BO-matched-full","TuRBO-matched-full","Random-search"}
    if method not in allowed: raise EvidenceTaskError(f"Unsupported matched campaign method {method!r}")
    fp=campaign_fingerprint(method=method,protocol=protocol,backend_identity=backend_identity,problem_fingerprint=problem_fingerprint)
    store=AtomicCampaignStore(campaign_root,fingerprint=fp,method=method)
    payload=store.load()
    if payload.get("complete"):
        return {"complete":True,"summary":campaign_summary(payload),"checkpoint":str(store.path)}
    control,reads,diag,state=next_proposal(method,payload,protocol,project_root=project_root,
                                           operator_action=evaluator.operator_action,candidate_pool_size=int(candidate_pool_size))
    response=evaluator(control,num_reads=int(reads))
    payload=append_observation(method,payload,protocol,control=control,reads=int(reads),response=response,diagnostics=diag,pre_state=state)
    store.save(payload)
    return {"complete":bool(payload["complete"]),"summary":campaign_summary(payload),"latest_query":len(payload["records"])-1,
            "latest_control":control.tolist(),"latest_reads":int(reads),"diagnostics":_jsonable(diag),"checkpoint":str(store.path)}


def campaign_status(*,campaign_root:str|Path,method:str)->dict[str,Any]:
    path=Path(campaign_root)/f"{method.replace('/','_')}.json"
    payload=_load_json(path,{})
    if not payload: return {"method":method,"exists":False,"complete":False,"queries":0}
    out=campaign_summary(payload);out["exists"]=True;out["checkpoint"]=str(path);return out


def evaluate_worldline_schedule(
    evaluator: Any,
    assets: Any,
    *,
    project_root: str | Path,
    calibration_family: str,
    output_path: str | Path,
    num_reads: int,
    total_time_us: float = 20.0,
    grid_points: int = 21,
) -> dict[str,Any]:
    """Construct actual SQA susceptibility and evaluate the resulting raw schedule."""
    construction=worldline_schedule_for_assets(
        assets,project_root,calibration_family,total_time_us=float(total_time_us),grid_points=int(grid_points)
    )
    backend=evaluator.backend
    response=backend.evaluate_schedule(
        np.asarray(construction["t_us"],float),np.asarray(construction["schedule_s"],float),
        num_reads=int(num_reads),elite_threshold=float(evaluator.elite_threshold),
        feasibility=evaluator.feasibility,success_energy=evaluator.success_energy,
        label="Worldline susceptibility schedule",
    )
    result={
        "method":"Worldline-Susceptibility-full",
        "construction":{k:_jsonable(v) for k,v in construction.items() if k!="diagnostics"},
        "response":{k:_jsonable(v) for k,v in response.items() if k!="sampleset"},
        "worldline_diagnostics":_jsonable(construction.get("diagnostics",[])),
    }
    atomic_json(output_path,result);return result


def _elite_probability_score(sampleset:Any,threshold:float)->float:
    energies=np.asarray(sampleset.record.energy,float);occ=np.asarray(sampleset.record.num_occurrences,int);total=int(np.sum(occ))
    return float(np.sum(occ[energies<=float(threshold)])/total)


def run_strong_classical_benchmark(
    bqm: Any,
    *,
    elite_threshold: float,
    output_path: str | Path,
    num_reads: int = 512,
    train_seeds: Sequence[int] = (101,202,303),
    holdout_seeds: Sequence[int] = (404,505,606),
) -> dict[str,Any]:
    """Tune and evaluate strong SA/Tabu arms with all tuning work accounted."""
    sa_grid=[]
    for sweeps in (1000,2000,4000):
        for beta_range in (None,(0.05,5.0),(0.1,10.0)):
            for kind in ("linear","geometric"):
                for randomize in (False,True):
                    for acceptance in ("Metropolis","Gibbs"):
                        cfg={"num_sweeps":sweeps,"beta_schedule_type":kind,"randomize_order":randomize,
                             "proposal_acceptance_criteria":acceptance}
                        if beta_range is not None: cfg["beta_range"]=beta_range
                        sa_grid.append(cfg)
    for beta_lo,beta_hi in ((0.05,5.0),(0.1,10.0)):
        for points in (256,512):
            for randomize in (False,True):
                for acceptance in ("Metropolis","Gibbs"):
                    sa_grid.append({
                        "beta_schedule_type":"custom",
                        "beta_schedule":np.geomspace(beta_lo,beta_hi,points),
                        "num_sweeps_per_beta":1,
                        "randomize_order":randomize,
                        "proposal_acceptance_criteria":acceptance,
                    })
    tabu_grid=[
        {"tenure":tenure,"timeout":timeout,"num_restarts":restarts,
         "energy_threshold":float(elite_threshold),"coefficient_z_first":zfirst,"coefficient_z_restart":zrestart}
        for tenure in (5,10,20) for timeout in (50,100) for restarts in (0,2,5)
        for zfirst,zrestart in ((5000,1250),(10000,2500),(25000,6250))
    ]
    score=lambda ss:_elite_probability_score(ss,float(elite_threshold))
    sa=rc.tune_dwave_sa(bqm,train_seeds=train_seeds,num_reads=int(num_reads),config_grid=sa_grid,score=score)
    tabu=rc.tune_dwave_tabu(bqm,train_seeds=train_seeds,num_reads=int(num_reads),config_grid=tabu_grid,score=score)
    try:
        from dwave.samplers import SimulatedAnnealingSampler,TabuSampler
    except ImportError as exc: raise EvidenceTaskError("dwave-samplers is required for strong classical evaluation") from exc
    sa_sampler=SimulatedAnnealingSampler();tabu_sampler=TabuSampler()
    sa_hold=[];tabu_hold=[]
    start=time.perf_counter()
    for seed in holdout_seeds:
        sa_hold.append(score(sa_sampler.sample(bqm,num_reads=int(num_reads),seed=int(seed),**dict(sa["best_config"]))))
    sa_hold_seconds=time.perf_counter()-start
    start=time.perf_counter()
    for seed in holdout_seeds:
        tabu_hold.append(score(tabu_sampler.sample(bqm,num_reads=int(num_reads),seed=int(seed),**dict(tabu["best_config"]))))
    tabu_hold_seconds=time.perf_counter()-start
    result={
        "Strong-SA":{"tuning":sa,"holdout_values":sa_hold,"holdout_mean":float(np.mean(sa_hold)),"holdout_seconds":sa_hold_seconds,
                     "total_classical_seconds":float(sa["tuning_seconds"]+sa_hold_seconds)},
        "Strong-Tabu":{"tuning":tabu,"holdout_values":tabu_hold,"holdout_mean":float(np.mean(tabu_hold)),"holdout_seconds":tabu_hold_seconds,
                       "total_classical_seconds":float(tabu["tuning_seconds"]+tabu_hold_seconds)},
        "elite_threshold":float(elite_threshold),"num_reads_per_call":int(num_reads),
        "train_seeds":list(map(int,train_seeds)),"holdout_seeds":list(map(int,holdout_seeds)),
    }
    atomic_json(output_path,result);return result


def qzero_claim_gate(corpus_path:str|Path)->dict[str,Any]:
    """Validate presence of a full provenance-bearing QZero pretraining corpus.

    The notebook deliberately refuses to substitute a toy MCTS/NN corpus for
    the full comparator.  Corpus generation can be sharded, but a numerical
    QZero superiority row is unavailable until this gate passes.
    """
    path=Path(corpus_path)
    if not path.exists():
        return {"pass":False,"status":"MISSING_PRETRAINING_CORPUS","path":str(path)}
    payload=_load_json(path,{})
    required={"schema","contexts","X","P","V","hidden_environment_queries","source_provenance","context_encoding"}
    missing=sorted(required-set(payload))
    contexts=payload.get("contexts",[])
    X=payload.get("X",[]); P=payload.get("P",[]); V=payload.get("V",[])
    source=payload.get("source_provenance")
    passed=(not missing and len(contexts)>=45 and int(payload.get("hidden_environment_queries",0))>0
            and bool(X) and bool(P) and bool(V) and len(X)==len(P)==len(V) and bool(source)
            and payload.get("context_encoding")=="normalized_ising_h_plus_upper_J_v38")
    return {"pass":bool(passed),"status":"PASS" if passed else "INCOMPLETE_PRETRAINING_CORPUS","missing":missing,
            "contexts":len(contexts),"training_rows":len(X),"hidden_environment_queries":int(payload.get("hidden_environment_queries",0) or 0),
            "context_encoding":payload.get("context_encoding"),"source_provenance_present":bool(source),"path":str(path)}



def run_paired_confirmation(
    evaluator: Any,
    *,
    cssf_control: Sequence[float],
    competitor_control: Sequence[float],
    output_path: str | Path,
    num_reads: int,
    replicates: int = 12,
    seed: int = 20260817,
    margin: float = 0.0,
    alpha: float = 0.05,
) -> dict[str,Any]:
    """Directly confirm two frozen controls under matched read budgets.

    Simulator confirmation uses common-random-number seed pairs with an
    independent seed per replicate.  QPU confirmation uses the same paired
    submission order and equal reads; no pseudo-seed is reported for hardware.
    The result is claim-locked unless the paired lower confidence bound exceeds
    the declared margin.
    """
    if int(replicates) < 4:
        raise EvidenceTaskError("paired confirmation requires at least four replicates")
    cssf=np.asarray(cssf_control,dtype=float).reshape(-1)
    comp=np.asarray(competitor_control,dtype=float).reshape(-1)
    if cssf.shape != comp.shape:
        raise EvidenceTaskError("confirmation controls must have equal dimensions")
    cssf_values=[];comp_values=[];rows=[]
    is_simulator=getattr(getattr(evaluator,"backend",None),"mode",None)=="simulator"
    for r in range(int(replicates)):
        pair_seed=int(seed)+r if is_simulator else None
        a=evaluator(cssf,num_reads=int(num_reads),sampling_seed=pair_seed)
        b=evaluator(comp,num_reads=int(num_reads),sampling_seed=pair_seed)
        av=float(a["elite_probability"]);bv=float(b["elite_probability"])
        cssf_values.append(av);comp_values.append(bv)
        rows.append({"replicate":r,"sampling_seed":pair_seed,"cssf_elite_probability":av,
                     "competitor_elite_probability":bv,"reads_each":int(num_reads)})
    lcb=rc.paired_cssf_vs_competitor_confirmation_lcb(
        cssf_values,comp_values,alpha=float(alpha),margin=float(margin),bootstrap=5000,seed=int(seed)
    )
    result={
        "schema":"CSSF-QA-PAIRED-CONFIRMATION-v38",
        "cssf_control":cssf.tolist(),"competitor_control":comp.tolist(),
        "replicates":int(replicates),"reads_per_control_per_replicate":int(num_reads),
        "pairs":rows,"statistics":_jsonable(lcb),
        "claim_pass":bool(lcb.get("pass",False)),
    }
    atomic_json(output_path,result);return result

def confirmation_lcb(
    cssf_values:Sequence[float],competitor_values:Sequence[float],*,output_path:str|Path,margin:float=0.0,alpha:float=0.05
)->dict[str,Any]:
    result=rc.paired_cssf_vs_competitor_confirmation_lcb(cssf_values,competitor_values,alpha=float(alpha),margin=float(margin),bootstrap=5000,seed=20260817)
    atomic_json(output_path,result);return result


__all__=[
    "EvidenceTaskError","collect_surrogate_response_step","run_surrogate_audit","advance_matched_campaign","campaign_status",
    "evaluate_worldline_schedule","run_strong_classical_benchmark","qzero_claim_gate","run_paired_confirmation","confirmation_lcb",
]
