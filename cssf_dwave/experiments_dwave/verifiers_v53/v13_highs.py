from __future__ import annotations
import math
from .common import *

def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"highs_reference.json")
    if not p: return report("V13",ctx,[check("highs_reference_present",False)])
    checks=[
        check("certified_optimal",p.get("certified_optimal") is True),
        check("frozen_problem_fingerprint",bool(p.get("problem_fingerprint"))),
        check("placement_present",bool(p.get("selected_buses"))),
        check("independent_objective_recheck",math.isfinite(float(p.get("recomputed_objective",float("nan")))) and abs(float(p.get("objective"))-float(p.get("recomputed_objective")))<=float(p.get("tolerance",1e-8)),observed=p.get("recomputed_objective"),expected=p.get("objective")),
        check("no_wall_clock_competition",p.get("wall_clock_competition") is False),
    ]
    return report("V13",ctx,checks)
