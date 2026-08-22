"""Restartable, one-annealer-query-at-a-time CSSF(QA) benchmark campaign.

The campaign preserves the fail-closed atomic-task execution discipline defined by the research contract
framework while supporting matched full-function schedule optimizers.  Each
invocation proposes at most one new annealing control, and state/history are
checkpointed with a frozen campaign fingerprint.  No scientific parameter is
silently reduced to meet wall time.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import os
import tempfile

import numpy as np

from benchmarks import reference_competitors as rc
from experiments_dwave.benchmark_protocol import MatchedControlProtocol, TARGET_NAMES, _tuRBO_matched_reads
from experiments_dwave.cssf_control_v53 import fit_cssf_qa_response
from experiments_dwave.control_design import admissible_latin_hypercube


class CampaignError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Mapping): return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_jsonable(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None: return value
    return str(value)


def campaign_fingerprint(*,method:str,protocol:MatchedControlProtocol,backend_identity:Mapping[str,Any],problem_fingerprint:str) -> str:
    payload={"schema":"CSSF-QA-SEQUENTIAL-CAMPAIGN-v38","method":str(method),"protocol":asdict(protocol),
             "backend_identity":_jsonable(backend_identity),"problem_fingerprint":str(problem_fingerprint)}
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(b"CSSF-QA-SEQUENTIAL-CAMPAIGN-v38\0"+raw).hexdigest()


class AtomicCampaignStore:
    def __init__(self,root:str|Path,*,fingerprint:str,method:str):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/f"{method.replace('/','_')}.json"
        self.fingerprint=str(fingerprint); self.method=str(method)

    def load(self)->dict[str,Any]:
        if not self.path.exists():
            return {"schema":"CSSF-QA-SEQUENTIAL-CAMPAIGN-v38","fingerprint":self.fingerprint,"method":self.method,
                    "records":[],"optimizer_state":None,"complete":False}
        payload=json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("fingerprint")!=self.fingerprint or payload.get("method")!=self.method:
            raise CampaignError("Campaign checkpoint fingerprint/method mismatch")
        if not isinstance(payload.get("records"),list): raise CampaignError("Campaign records must be a list")
        return payload

    def save(self,payload:Mapping[str,Any])->None:
        out=dict(payload); out["fingerprint"]=self.fingerprint;out["method"]=self.method;out["schema"]="CSSF-QA-SEQUENTIAL-CAMPAIGN-v38"
        self.root.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix=self.path.name+".",suffix=".tmp",dir=str(self.root))
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(_jsonable(out),f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)


def _response_matrix(records:Sequence[Mapping[str,Any]])->np.ndarray:
    return np.asarray([[float(r["response"][k]) for k in TARGET_NAMES] for r in records],dtype=float)


def _controls(records:Sequence[Mapping[str,Any]])->np.ndarray:
    return np.asarray([r["control"] for r in records],dtype=float)


def _turbo_dump(opt:rc.JeongTuRBO)->dict[str,Any]:
    state={f.name:_jsonable(getattr(opt.state,f.name)) for f in fields(type(opt.state))}
    return {"state":state,"X":[_jsonable(x) for x in opt.X],"y":[float(v) for v in opt.y],
            "restart_queue":[_jsonable(x) for x in opt.restart_queue]}


def _turbo_load(protocol:MatchedControlProtocol,payload:Mapping[str,Any]|None)->rc.JeongTuRBO:
    opt=rc.JeongTuRBO(protocol.bounds,order=protocol.order,seed=protocol.seed,minimize_objective=False)
    if not payload: return opt
    for k,v in dict(payload.get("state",{})).items():
        if hasattr(opt.state,k):
            if k=="incumbent_x" and v is not None: v=np.asarray(v,dtype=float)
            setattr(opt.state,k,v)
    opt.X=[np.asarray(x,dtype=float) for x in payload.get("X",[])]
    opt.y=[float(v) for v in payload.get("y",[])]
    opt.restart_queue=[np.asarray(x,dtype=float) for x in payload.get("restart_queue",[])]
    if len(opt.X)!=len(opt.y): raise CampaignError("TuRBO checkpoint X/y length mismatch")
    return opt


def next_proposal(
    method:str, payload:Mapping[str,Any], protocol:MatchedControlProtocol, *, project_root:str|Path,
    operator_action:Callable[[np.ndarray],np.ndarray], candidate_pool_size:int=4096,
) -> tuple[np.ndarray,int,dict[str,Any],dict[str,Any]|None]:
    """Return one next control, read count, diagnostics and optional optimizer state."""
    records=list(payload.get("records",[])); n=len(records)
    if n>=protocol.total_control_budget: raise CampaignError("Campaign control-query budget is complete")
    bounds=protocol.bounds; method=str(method)

    if method=="CSSF-full":
        initial=admissible_latin_hypercube(bounds,protocol.cssf_initial_count,order=protocol.order,seed=protocol.seed)
        if n<protocol.cssf_initial_count:
            return initial[n].copy(),protocol.reads_per_control,{"phase":"initial_design","index":n},None
        theta=np.vstack([np.asarray(r["response"]["operator_action"],dtype=float) for r in records])
        Y=_response_matrix(records);nt=protocol.cssf_initial_train;nc=protocol.cssf_initial_calibration
        cal=np.arange(nt,nt+nc,dtype=int); active=np.arange(nt+nc,n,dtype=int); train=np.concatenate([np.arange(nt,dtype=int),active])
        model=fit_cssf_qa_response(theta[train],Y[train],calibration_operator_phase=theta[cal],calibration_targets=Y[cal],
                                   target_names=TARGET_NAMES,project_root=project_root,support_mode="signed_axes",support_order=1,
                                   metadata={"method":method,"active_refit_observations":int(active.size),"protocol":asdict(protocol)})
        pool=rc.latin_hypercube(bounds,int(candidate_pool_size),seed=protocol.seed+1000+n)
        feasible=[];phase=[]
        for c in pool:
            try: phase.append(np.asarray(operator_action(c),dtype=float));feasible.append(c)
            except Exception: continue
        if not feasible: raise CampaignError("No feasible CSSF acquisition candidates")
        idx,score,diag=model.acquisition(np.vstack(phase),target="elite_probability",maximize=True,uncertainty_weight=1.0,
                                         leverage_weight=0.25,feasibility_target="feasibility_probability",minimum_feasibility=0.0)
        return np.asarray(feasible[idx],dtype=float),protocol.reads_per_control,{"phase":"active_cssf",**dict(diag),"score":float(score[idx])},None

    if method=="GP+EI-full":
        opt=rc.SequentialGPEI(bounds,minimize_objective=False,seed=protocol.seed,n_restarts_optimizer=2)
        init=admissible_latin_hypercube(bounds,10,order=protocol.order,seed=protocol.seed)
        for r in records: opt.add(np.asarray(r["control"],dtype=float),float(r["response"]["elite_probability"]))
        if n<10: return init[n].copy(),protocol.reads_per_control,{"phase":"initial_design","index":n},None
        c,d=opt.suggest(feasibility=lambda x:rc.feasible_control(x,bounds,order=protocol.order))
        return c,protocol.reads_per_control,{"phase":"sequential_gp_ei",**dict(d)},None

    if method=="Finzgar-BO-matched-full":
        opt=rc.FinzgarBO(bounds,seed=protocol.seed,total_paper_iterations=50)
        linear=rc.canonical_control(protocol.order,float(np.mean(protocol.annealing_time_range_us)))
        init=opt.initial_design(linear,feasibility=lambda x:rc.feasible_control(x,bounds,order=protocol.order))
        for r in records: opt.add(np.asarray(r["control"],dtype=float),float(r["response"]["elite_probability"]))
        if n<len(init): return init[n].copy(),protocol.reads_per_control,{"phase":"linear_plus_9_random","index":n},None
        c,d=opt.suggest(iteration=n,feasibility=lambda x:rc.feasible_control(x,bounds,order=protocol.order))
        return c,protocol.reads_per_control,{"phase":"finzgar_ucb",**dict(d)},None

    if method=="TuRBO-matched-full":
        opt=_turbo_load(protocol,payload.get("optimizer_state"))
        init=admissible_latin_hypercube(bounds,10,order=protocol.order,seed=protocol.seed)
        reads_used=sum(int(r["response"].get("num_reads",r.get("reads",0))) for r in records)
        if n<10:
            c=init[n].copy(); reads=_tuRBO_matched_reads(opt,c,protocol,reads_used=reads_used,queries_used=n)
            return c,reads,{"phase":"space_filling_initial","adaptive_reads":reads,"index":n},_turbo_dump(opt)
        c,d=opt.suggest(feasibility=lambda x:rc.feasible_control(x,bounds,order=protocol.order))
        reads=_tuRBO_matched_reads(opt,c,protocol,reads_used=reads_used,queries_used=n)
        return c,reads,{"phase":"ei_trust_region","adaptive_reads":reads,**dict(d)},_turbo_dump(opt)

    if method=="Random-search":
        design=admissible_latin_hypercube(bounds,protocol.total_control_budget,order=protocol.order,seed=protocol.seed+77)
        return design[n].copy(),protocol.reads_per_control,{"phase":"random_search","index":n},None

    raise CampaignError(f"Unsupported sequential schedule optimizer {method!r}")


def append_observation(
    method:str,payload:Mapping[str,Any],protocol:MatchedControlProtocol,*,control:np.ndarray,reads:int,
    response:Mapping[str,Any],diagnostics:Mapping[str,Any],pre_state:Mapping[str,Any]|None,
) -> dict[str,Any]:
    out=dict(payload); records=list(out.get("records",[]))
    if len(records)>=protocol.total_control_budget: raise CampaignError("Cannot append beyond control-query budget")
    resp={k:_jsonable(v) for k,v in response.items() if k!="sampleset"}
    required=set(TARGET_NAMES)|{"operator_action","num_reads"}
    missing=sorted(required-set(resp));
    if missing: raise CampaignError(f"Response is missing required fields: {missing}")
    if int(resp["num_reads"])!=int(reads): raise CampaignError("Returned read count differs from reserved/read request")
    records.append({"query_index":len(records),"control":_jsonable(np.asarray(control,dtype=float)),"reads":int(reads),
                    "response":resp,"diagnostics":_jsonable(diagnostics)})
    out["records"]=records
    if method=="TuRBO-matched-full":
        opt=_turbo_load(protocol,pre_state)
        if len(records)<=10:
            opt.add_initial(np.asarray(control,dtype=float),float(resp["elite_probability"]))
        else:
            obs=opt.observe(np.asarray(control,dtype=float),float(resp["elite_probability"]))
            records[-1]["diagnostics"].update(_jsonable(obs))
        out["optimizer_state"]=_turbo_dump(opt)
    out["complete"]=bool(len(records)==protocol.total_control_budget)
    if out["complete"] and method=="TuRBO-matched-full":
        total=sum(int(r["response"]["num_reads"]) for r in records)
        if total!=protocol.total_read_budget: raise CampaignError(f"TuRBO final reads {total} != matched {protocol.total_read_budget}")
    return out


def campaign_summary(payload:Mapping[str,Any],*,target:str="elite_probability")->dict[str,Any]:
    records=list(payload.get("records",[]))
    if not records: return {"method":payload.get("method"),"queries":0,"complete":False}
    values=np.asarray([float(r["response"][target]) for r in records])
    i=int(np.argmax(values))
    return {"method":payload.get("method"),"queries":len(records),"reads":sum(int(r["response"]["num_reads"]) for r in records),
            "best_target":float(values[i]),"best_control":records[i]["control"],"complete":bool(payload.get("complete",False))}
