"""Restartable D0--D3 CSSF campaigns for the integrated BESS proof.

Each invocation advances exactly one annealer query and persists the full
observation history.  D0/D1 use normalized raw schedule coordinates; D2/D3 use
calibration-resolved operator-action coordinates.  Fitting, GCV, uncertainty,
acquisition, candidate pools, seeds and read budgets are otherwise identical.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from benchmarks import reference_competitors as rc
from experiments_dwave.benchmark_protocol import MatchedControlProtocol, MethodTrace, TARGET_NAMES
from experiments_dwave.evidence_v38 import qpu_access_time_us
from experiments_dwave.cssf_control_v53 import fit_cssf_qa_response
from experiments_dwave.integrated_bess_v38 import raw_control_coordinates
from experiments_dwave.control_design import admissible_latin_hypercube


class FactorialCampaignError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_jsonable(v) for v in value]
    return value


def _fingerprint(*,arm_id:str,domain_representation:str,control_representation:str,protocol:MatchedControlProtocol,backend_identity:Mapping[str,Any],problem_fingerprint:str)->str:
    payload={"schema":"CSSF-QA-FACTORIAL-CAMPAIGN-v38","arm_id":arm_id,"domain_representation":domain_representation,"control_representation":control_representation,"protocol":asdict(protocol),"backend_identity":_jsonable(backend_identity),"problem_fingerprint":str(problem_fingerprint)}
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(b"CSSF-QA-FACTORIAL-CAMPAIGN-v38\0"+raw).hexdigest()


def _path(root: str|Path, arm_id: str) -> Path:
    return Path(root)/f"{arm_id}.json"


def load_factorial_campaign(root:str|Path,arm_id:str)->dict[str,Any]:
    p=_path(root,arm_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _atomic_write(path:Path,payload:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=str(path.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(_jsonable(payload),f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _response_matrix(records:list[dict[str,Any]])->np.ndarray:
    return np.asarray([[float(r["response"][k]) for k in TARGET_NAMES] for r in records],dtype=float)


def _coordinate(control:np.ndarray,response:Mapping[str,Any]|None,*,mode:str,evaluator:Any,protocol:MatchedControlProtocol)->np.ndarray:
    if mode=="raw": return raw_control_coordinates(control,protocol.bounds)
    if response is not None and response.get("operator_action") is not None:
        return np.asarray(response["operator_action"],dtype=float)
    pure=getattr(evaluator,"operator_action",None)
    if pure is None or not callable(pure): raise FactorialCampaignError("operator-phase campaign requires evaluator.operator_action(control)")
    return np.asarray(pure(control),dtype=float)


def advance_factorial_campaign(
    evaluator:Any, *, arm_id:str, domain_representation:str, control_representation:str,
    protocol:MatchedControlProtocol, project_root:str|Path, campaign_root:str|Path,
    backend_identity:Mapping[str,Any], problem_fingerprint:str, candidate_pool_size:int=4096,
)->dict[str,Any]:
    aid=str(arm_id); domain=str(domain_representation); control_mode=str(control_representation)
    if aid not in {"D0","D1","D2","D3"}: raise FactorialCampaignError("arm_id must be D0..D3")
    expected={"D0":("raw","raw"),"D1":("trig","raw"),"D2":("raw","operator_phase"),"D3":("trig","operator_phase")}[aid]
    if (domain,control_mode)!=expected: raise FactorialCampaignError(f"{aid} factor identity mismatch: {(domain,control_mode)} != {expected}")
    fp=_fingerprint(arm_id=aid,domain_representation=domain,control_representation=control_mode,protocol=protocol,backend_identity=backend_identity,problem_fingerprint=problem_fingerprint)
    path=_path(campaign_root,aid)
    if path.is_file():
        payload=json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint")!=fp: raise FactorialCampaignError(f"{aid} checkpoint fingerprint mismatch")
    else:
        payload={"schema":"CSSF-QA-FACTORIAL-CAMPAIGN-v38","fingerprint":fp,"arm_id":aid,"domain_representation":domain,"control_representation":control_mode,"method":"CSSF-trig" if control_mode=="operator_phase" else "CSSF-raw/no-trig","records":[],"complete":False,"protocol":asdict(protocol),"problem_fingerprint":str(problem_fingerprint),"backend_identity":_jsonable(backend_identity)}
    records=list(payload.get("records",[])); n=len(records)
    if payload.get("complete"):
        return {"complete":True,"queries":n,"checkpoint":str(path),"summary":factorial_campaign_summary(payload)}
    if n>=protocol.total_control_budget: raise FactorialCampaignError("campaign exceeds frozen control budget")
    initial=admissible_latin_hypercube(protocol.bounds,protocol.cssf_initial_count,order=protocol.order,seed=protocol.seed)
    if n<protocol.cssf_initial_count:
        candidate=np.asarray(initial[n],dtype=float); diag={"phase":"initial_design","index":n}
    else:
        theta=np.vstack([_coordinate(np.asarray(r["control"],float),r["response"],mode=control_mode,evaluator=evaluator,protocol=protocol) for r in records])
        Y=_response_matrix(records);nt=protocol.cssf_initial_train;nc=protocol.cssf_initial_calibration
        cal=np.arange(nt,nt+nc,dtype=int); active=np.arange(nt+nc,n,dtype=int); train=np.concatenate([np.arange(nt,dtype=int),active])
        model=fit_cssf_qa_response(theta[train],Y[train],calibration_operator_phase=theta[cal],calibration_targets=Y[cal],target_names=TARGET_NAMES,project_root=project_root,support_mode="signed_axes",support_order=1,n_lambdas=100,lam_range=(-12.0,4.0),nominal_coverage=0.90,metadata={"contract":"v38","arm_id":aid,"domain_representation":domain,"control_representation":control_mode,"active_refit_observations":int(active.size)})
        pool=rc.latin_hypercube(protocol.bounds,int(candidate_pool_size),seed=protocol.seed+1000+n)
        feasible=[]; coords=[]
        for c in pool:
            try:
                rc.fourier_forward_schedule(c,order=protocol.order,grid_points=129,reject_nonmonotone=True)
                coords.append(_coordinate(np.asarray(c,float),None,mode=control_mode,evaluator=evaluator,protocol=protocol));feasible.append(np.asarray(c,float))
            except Exception: continue
        if not feasible: raise FactorialCampaignError("no physically feasible acquisition candidate")
        idx,score,acq=model.acquisition(np.vstack(coords),target="elite_probability",maximize=True,uncertainty_weight=1.0,leverage_weight=0.25,feasibility_target="feasibility_probability",minimum_feasibility=0.0)
        candidate=np.asarray(feasible[idx],dtype=float);diag={"phase":"active_cssf","arm_id":aid,"control_representation":control_mode,**dict(acq),"score":float(score[idx])}
    response=evaluator(candidate,num_reads=int(protocol.reads_per_control))
    resp={k:_jsonable(v) for k,v in response.items() if k!="sampleset"}
    required=set(TARGET_NAMES)|{"operator_action","num_reads"}
    missing=sorted(required-set(resp))
    if missing: raise FactorialCampaignError(f"response missing fields: {missing}")
    if int(resp["num_reads"])!=int(protocol.reads_per_control): raise FactorialCampaignError("factorial arms require fixed matched reads/control")
    records.append({"query_index":n,"control":candidate.tolist(),"response":resp,"diagnostics":_jsonable(diag)})
    payload["records"]=records;payload["complete"]=len(records)==protocol.total_control_budget
    _atomic_write(path,payload)
    return {"complete":bool(payload["complete"]),"queries":len(records),"checkpoint":str(path),"summary":factorial_campaign_summary(payload)}


def factorial_campaign_summary(payload:Mapping[str,Any])->dict[str,Any]:
    records=list(payload.get("records",[]))
    if not records: return {"arm_id":payload.get("arm_id"),"queries":0,"complete":False}
    values=np.asarray([float(r["response"]["elite_probability"]) for r in records]); i=int(np.argmax(values))
    return {"arm_id":payload.get("arm_id"),"method":payload.get("method"),"queries":len(records),"reads":sum(int(r["response"]["num_reads"]) for r in records),"best_target":float(values[i]),"best_control":records[i]["control"],"complete":bool(payload.get("complete"))}


def campaign_to_trace(payload:Mapping[str,Any])->MethodTrace:
    if not payload.get("complete"): raise FactorialCampaignError("campaign must be complete before conversion to MethodTrace")
    method=f"{payload.get('arm_id')}:{payload.get('method')}"
    ledger=rc.ResourceLedger();controls=[];responses=[];diags=[]
    for row in payload.get("records",[]):
        control=np.asarray(row["control"],dtype=float);response=dict(row["response"]);diag=dict(row.get("diagnostics",{}))
        controls.append(control);responses.append(response);diags.append(diag)
        ledger.add(rc.CostEntry(method=method,stage="search",control_query=1,reads=int(response["num_reads"]),annealing_time_us=float(control[0]),qpu_access_time_us=qpu_access_time_us(response),simulator_seconds=float(response.get("elapsed_seconds",0.0)),metadata=diag))
    return MethodTrace(method,controls,responses,ledger,diags)


__all__=["FactorialCampaignError","advance_factorial_campaign","load_factorial_campaign","factorial_campaign_summary","campaign_to_trace"]
