from __future__ import annotations
from .common import *
REQUIRED=("V00","V01","V02","V03","V04","V05","V06","V07","V10","V11","V12","V13","V14","V15","V16","V17")

def verify(ctx:VerificationContext, reports:dict[str,dict] | None=None)->dict:
    if reports is None:
        reports={}
        folder=ctx.evidence_root/"verifiers"
        for p in folder.glob("V*.json"):
            try: reports[p.stem]=load_json(p)
            except Exception: pass
    checks=[]
    for vid in REQUIRED: checks.append(check(f"{vid}:required_pass",reports.get(vid,{}).get("status")==PASS,observed=reports.get(vid,{}).get("status"),expected=PASS))
    if ctx.mode=="simulator": checks.append(check("V08:simulator_hardware",reports.get("V08",{}).get("status")==PASS,observed=reports.get("V08",{}).get("status"),expected=PASS))
    if ctx.mode=="qpu": checks.append(check("V09:qpu_hardware",reports.get("V09",{}).get("status")==PASS,observed=reports.get("V09",{}).get("status"),expected=PASS))
    qzero=maybe_json(ctx.evidence_root/"qzero_gate.json") or {}
    checks.append(check("QZero_full_ready",qzero.get("full_qzero_ready") is True))
    final=all(c["status"]==PASS for c in checks)
    checks.append(check("G_final",final,observed=final,expected=True))
    return report("V18",ctx,checks,status=PASS if final else FAIL)
