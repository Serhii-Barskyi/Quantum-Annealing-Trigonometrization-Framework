from __future__ import annotations
from .common import *

def verify(ctx:VerificationContext)->dict:
    lineage=maybe_json(ctx.evidence_root/"lineage.json") or {}; nodes=list(lineage.get("nodes",[]))
    required=["domain_spectral","bess_qubo","operator_phase","qa_response_surrogate","annealer","placement","application_endpoint"]
    ok,path=graph_path(nodes,required)
    factor=maybe_json(ctx.evidence_root/"factorial.json") or {}
    d3=next((a for a in factor.get("arms",[]) if a.get("arm_id")=="D3"),{})
    checks=[
        check("joint_end_to_end_path",ok,observed=path,expected=required),
        check("D3_domain_trig",d3.get("domain_representation")=="trig"),
        check("D3_operator_phase",d3.get("control_representation")=="operator_phase"),
        check("D3_final_endpoint_evidence",bool(d3.get("application_endpoint_evidence_id"))),
    ]
    return report("V03",ctx,checks,evidence_ids=path)
