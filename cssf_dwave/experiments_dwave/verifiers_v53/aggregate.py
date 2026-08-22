"""Aggregate contract verifier. Notebook booleans are never trusted."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .common import VerificationContext, PASS
from . import (
    v00_repository,v01_execution_graph,v02_csnnt,v03_three_level,v04_factorial,
    v05_raw_ablation,v06_residual,v07_operator_action,v08_simulator_sensitivity,
    v09_qpu_scope,v10_competitor_fidelity,v11_resources,v12_application_endpoint,
    v13_highs,v14_statistics,v15_cost_to_target,v16_provenance,v17_reproducibility,
    v18_claim_gate,v19_external_export,
)

BASE=[v00_repository,v01_execution_graph,v02_csnnt,v03_three_level,v04_factorial,v05_raw_ablation,v06_residual,v07_operator_action,v08_simulator_sensitivity,v09_qpu_scope,v10_competitor_fidelity,v11_resources,v12_application_endpoint,v13_highs,v14_statistics,v15_cost_to_target,v16_provenance,v17_reproducibility]

def run_all(ctx:VerificationContext, *, write:bool=True)->dict:
    reports={}
    out=ctx.evidence_root/"verifiers"; out.mkdir(parents=True,exist_ok=True)
    for mod in BASE:
        r=mod.verify(ctx); reports[r["verifier_id"]]=r
        if write: (out/f'{r["verifier_id"]}.json').write_text(json.dumps(r,indent=2,sort_keys=True,default=str),encoding="utf-8")
    r18=v18_claim_gate.verify(ctx,reports); reports["V18"]=r18
    if write: (out/"V18.json").write_text(json.dumps(r18,indent=2,sort_keys=True,default=str),encoding="utf-8")
    r19=v19_external_export.verify(ctx,reports); reports["V19"]=r19
    if write: (out/"V19.json").write_text(json.dumps(r19,indent=2,sort_keys=True,default=str),encoding="utf-8")
    aggregate={
        "schema":"CSSF-QA-VERIFIER-AGGREGATE-v38","mode":ctx.mode,
        "statuses":{k:v["status"] for k,v in sorted(reports.items())},
        "claim_grade":r18["status"]==PASS and r19["status"]==PASS,
    }
    if write: (out/"aggregate.json").write_text(json.dumps(aggregate,indent=2,sort_keys=True),encoding="utf-8")
    return {"reports":reports,"aggregate":aggregate}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",required=True); ap.add_argument("--evidence-root",required=True); ap.add_argument("--notebook"); ap.add_argument("--mode",choices=["simulator","qpu"],default="simulator"); ap.add_argument("--live-hardware-expected",action="store_true")
    ns=ap.parse_args(); ctx=VerificationContext.build(ns.project_root,ns.evidence_root,notebook_path=None if not ns.notebook else Path(ns.notebook),mode=ns.mode,live_hardware_expected=ns.live_hardware_expected)
    result=run_all(ctx); print(json.dumps(result["aggregate"],indent=2)); return 0 if result["aggregate"]["claim_grade"] else 2
if __name__=="__main__": raise SystemExit(main())
