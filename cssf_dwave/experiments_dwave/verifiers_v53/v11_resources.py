from __future__ import annotations
from collections import defaultdict
from .common import *


def verify(ctx:VerificationContext)->dict:
    events=read_jsonl(ctx.evidence_root/"resource_events.jsonl"); p=maybe_json(ctx.evidence_root/"resource_budget.json")
    if not events or not p:
        return report("V11",ctx,[check("event_level_accounting_present",False,observed=len(events),expected=">0")])
    totals=defaultdict(lambda:defaultdict(float))
    for e in events:
        totals[str(e["method"])][str(e["resource"])]+=float(e.get("amount",1.0))
    checks=[]
    for method,decl in p.get("methods",{}).items():
        for resource,expected in decl.get("totals",{}).items():
            obs=totals[method][resource]
            checks.append(check(f"{method}:{resource}",abs(obs-float(expected))<=1e-12,observed=obs,expected=float(expected)))
    groups=p.get("matched_groups")
    if groups:
        for group in groups:
            name=str(group.get("name","group")); methods=list(group.get("methods",[])); resources=list(group.get("resources",[])); mode=str(group.get("mode","identical"))
            checks.append(check(f"{name}:methods_present",bool(methods) and all(m in p.get("methods",{}) for m in methods),observed=methods))
            for resource in resources:
                vals=[totals[m][resource] for m in methods]
                if mode=="identical":
                    checks.append(check(f"{name}:matched:{resource}",len(set(vals))<=1,observed=vals,expected="identical"))
                elif mode=="charged":
                    checks.append(check(f"{name}:charged:{resource}",all(v>=0 for v in vals),observed=vals,expected="fully charged/nonnegative"))
                else:
                    checks.append(check(f"{name}:mode",False,observed=mode,expected="identical|charged"))
    else:
        matched_resources=tuple(p.get("matched_resources",["control_evaluation","annealer_reads","confirmation_reads"]))
        for r in matched_resources:
            vals=[totals[m][r] for m in p.get("comparison_methods",[])]; checks.append(check(f"matched:{r}",len(set(vals))<=1,observed=vals,expected="identical"))
    declared=set(p.get("methods",{})); observed=set(totals)
    checks.append(check("all_observed_methods_declared",observed<=declared,observed=sorted(observed-declared),expected=[]))
    return report("V11",ctx,checks)
