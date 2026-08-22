from __future__ import annotations
from collections import defaultdict
from .common import *

def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"cost_to_target.json"); events=read_jsonl(ctx.evidence_root/"resource_events.jsonl")
    if not p or not events: return report("V15",ctx,[check("cost_to_target_inputs_present",False)])
    target=p.get("target"); checks=[check("target_predeclared",bool(target) and bool(p.get("target_frozen_sha256")))]
    by=defaultdict(list)
    for e in events:
        if e.get("event_type")=="evaluation": by[str(e.get("method"))].append(e)
    for row in p.get("methods",[]):
        method=str(row.get("method")); seq=by.get(method,[]); hit=None; cum=0.0
        for e in seq:
            cum+=float(e.get("amount",1.0)); u=e.get("metadata",{}).get("utility")
            if u is not None and ((p.get("maximize",True) and float(u)>=float(target)) or (not p.get("maximize",True) and float(u)<=float(target))): hit=cum; break
        if hit is None:
            checks.append(check(f"{method}:censored",row.get("status")=="NOT_REACHED" and row.get("cost") is None))
        else:
            checks.append(check(f"{method}:first_hit",row.get("status")=="REACHED" and abs(float(row.get("cost"))-hit)<=1e-12,observed=row.get("cost"),expected=hit))
    return report("V15",ctx,checks)
