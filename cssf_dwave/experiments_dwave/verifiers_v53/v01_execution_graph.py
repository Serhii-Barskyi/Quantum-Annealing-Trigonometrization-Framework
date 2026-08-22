from __future__ import annotations
import json
from .common import *

REQUIRED=["domain_spectral","bess_qubo","operator_phase","qa_response_surrogate","annealer","placement","application_endpoint"]

def verify(ctx:VerificationContext)->dict:
    checks=[]
    lineage=maybe_json(ctx.evidence_root/"lineage.json")
    if not lineage: return report("V01",ctx,[check("runtime_lineage_present",False)])
    nodes=list(lineage.get("nodes",[])); ok,path=graph_path(nodes,REQUIRED)
    checks.append(check("integrated_runtime_path",ok,observed=[next((n.get('stage') for n in nodes if n.get('evidence_id')==x),x) for x in path],expected=REQUIRED))
    unique=len({n.get("evidence_id") for n in nodes})==len(nodes)
    checks.append(check("unique_evidence_ids",unique,observed=len(nodes),expected=len({n.get('evidence_id') for n in nodes})))
    if ctx.notebook_path and ctx.notebook_path.is_file():
        nb=json.loads(ctx.notebook_path.read_text(encoding="utf-8")); code="\n".join("".join(c.get("source",[])) for c in nb.get("cells",[]) if c.get("cell_type")=="code")
        checks.append(check("no_generic_run_task_dispatcher","run_task(" not in code))
        for stage in ("D0","D1","D2","D3","V00","V19"):
            checks.append(check(f"notebook_exposes_{stage}",stage in code))
    return report("V01",ctx,checks,evidence_ids=path)
