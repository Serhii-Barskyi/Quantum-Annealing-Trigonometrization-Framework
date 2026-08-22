from __future__ import annotations
import numpy as np
from .common import *
EXPECTED={"D0":("raw","raw"),"D1":("trig","raw"),"D2":("raw","operator_phase"),"D3":("trig","operator_phase")}

def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"factorial.json")
    if not p: return report("V04",ctx,[check("factorial_evidence_present",False)])
    arms={str(a.get("arm_id")):a for a in p.get("arms",[])}; checks=[check("exact_arm_ids",set(arms)==set(EXPECTED),observed=sorted(arms),expected=sorted(EXPECTED))]
    matched=set()
    util={}
    for arm,(d,c) in EXPECTED.items():
        a=arms.get(arm,{})
        checks.extend([check(f"{arm}:domain",a.get("domain_representation")==d,observed=a.get("domain_representation"),expected=d),check(f"{arm}:control",a.get("control_representation")==c,observed=a.get("control_representation"),expected=c)])
        if a.get("matched_config_sha256"): matched.add(a["matched_config_sha256"])
        if isinstance(a.get("utility"),(int,float)): util[arm]=float(a["utility"])
    checks.append(check("all_nonfactor_config_matched",len(matched)==1,observed=sorted(matched),expected="one shared hash"))
    if len(util)==4:
        delta=util["D3"]-util["D2"]-util["D1"]+util["D0"]
        stored=float(p.get("interaction",{}).get("delta",float("nan")))
        checks.append(check("interaction_recomputed",np.isfinite(stored) and np.isclose(delta,stored,rtol=0,atol=1e-12),observed=stored,expected=delta))
        checks.append(check("paired_uncertainty_present",bool(p.get("interaction",{}).get("paired_uncertainty"))))
    else: checks.append(check("utilities_complete",False,observed=sorted(util),expected=sorted(EXPECTED)))
    return report("V04",ctx,checks)
