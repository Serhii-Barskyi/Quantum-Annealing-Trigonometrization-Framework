from __future__ import annotations
from .common import *

def verify(ctx:VerificationContext)->dict:
    if ctx.mode!="qpu": return report("V09",ctx,[check("not_qpu_claim",True)])
    p=maybe_json(ctx.evidence_root/"qpu_scope.json")
    if not p: return report("V09",ctx,[check("live_qpu_evidence_present",False)],status=NOT_RUN_HARDWARE)
    sid=str(p.get("solver_id","")); checks=[
        check("approved_solver",sid.startswith(("Advantage_system4.","Advantage_system6.")),observed=sid,expected="Advantage_system4.*|Advantage_system6.*"),
        check("pegasus_p16",str(p.get("topology","")).lower()=="pegasus" and int(p.get("pegasus_m",-1))==16),
        check("fixed_embedding_match",bool(p.get("embedding_fingerprint")) and p.get("embedding_fingerprint")==p.get("expected_embedding_fingerprint")),
        check("schedule_validated",p.get("schedule_validated") is True),
        check("no_fallback",p.get("fallback_backend") in (None,"")),
        check("qpu_timing_present",bool(p.get("qpu_timing")) and int(p.get("num_reads",0))>0),
    ]
    return report("V09",ctx,checks)
