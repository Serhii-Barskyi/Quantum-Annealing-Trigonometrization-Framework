from __future__ import annotations
from .common import *
BASE=["opf","qaoa","ma_qaoa","digitized_qa"]

def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"residual_hierarchy.json")
    if not p: return report("V06",ctx,[check("residual_hierarchy_present",False)])
    comps=p.get("components",[]); levels=[c.get("level") for c in comps]; expected=BASE+(["hardware_residual"] if ctx.mode=="qpu" else [])
    checks=[check("ordered_components",levels==expected,observed=levels,expected=expected)]
    for c in comps:
        level=str(c.get("level"));
        for key in ("source_evidence_id","training_partition_fingerprint","target_definition","model_hash"):
            checks.append(check(f"{level}:{key}",bool(c.get(key))))
        checks.append(check(f"{level}:enters_final_predictor",c.get("enters_final_predictor") is True))
        checks.append(check(f"{level}:contribution_hash",bool(c.get("contribution_hash"))))
    checks.append(check("final_prediction_hash",bool(p.get("final_prediction_hash"))))
    return report("V06",ctx,checks)
