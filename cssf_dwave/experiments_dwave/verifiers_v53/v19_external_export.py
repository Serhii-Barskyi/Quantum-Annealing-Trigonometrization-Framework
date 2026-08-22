from __future__ import annotations
from .common import *

def verify(ctx:VerificationContext, reports:dict[str,dict] | None=None)->dict:
    if reports is None:
        reports={}
        for p in (ctx.evidence_root/"verifiers").glob("V*.json"):
            try: reports[p.stem]=load_json(p)
            except Exception: pass
    export=maybe_json(ctx.evidence_root/"external_export.json") or {}
    mandatory=[f"V{i:02d}" for i in range(19)]
    missing=[v for v in mandatory if v not in reports]
    bad={v:reports.get(v,{}).get("status") for v in mandatory if reports.get(v,{}).get("status")!=PASS}
    # For non-QPU simulator evidence, V09 is not an active claim gate; it may PASS as not-applicable.
    checks=[check("all_verifier_reports_present",not missing,observed=missing,expected=[]),check("all_applicable_verifiers_pass",not bad,observed=bad,expected={}),check("V18_final_gate",reports.get("V18",{}).get("status")==PASS)]
    rows=export.get("rows",[])
    checks.append(check("table_rows_traceable",all(r.get("evidence_id") and r.get("verifier_id") in reports for r in rows)))
    checks.append(check("claim_language_locked_to_gate",export.get("claim_grade") is True and reports.get("V18",{}).get("status")==PASS))
    return report("V19",ctx,checks)
