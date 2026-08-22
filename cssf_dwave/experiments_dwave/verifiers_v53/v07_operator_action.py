from __future__ import annotations
import numpy as np
from experiments_dwave.operator_phase import APPROVED_FAMILIES, load_calibration, operator_action_coordinates
from .common import *

def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"operator_action.json")
    if not p: return report("V07",ctx,[check("operator_action_evidence_present",False)])
    checks=[]
    for fam_block in p.get("families",[]):
        fam=fam_block.get("family"); checks.append(check(f"{fam}:approved_family",fam in APPROVED_FAMILIES))
        if fam not in APPROVED_FAMILIES: continue
        filename,expected_hash=APPROVED_FAMILIES[fam]; curve=load_calibration(ctx.project_root/"calibration"/filename,fam)
        checks.append(check(f"{fam}:calibration_hash",curve.source_sha256==expected_hash,observed=curve.source_sha256,expected=expected_hash))
        for row in fam_block.get("controls",[]):
            t=np.asarray(row.get("t_us"),dtype=float); s=np.asarray(row.get("s"),dtype=float); stored=np.asarray(row.get("operator_action"),dtype=float)
            calc=operator_action_coordinates(t,s,curve,n_segments=int(row.get("n_segments",8)))
            err=float(np.max(np.abs(calc-stored))) if calc.shape==stored.shape else float("inf")
            checks.append(check(f"{fam}:{row.get('control_id')}:operator_action",calc.shape==stored.shape and np.allclose(calc,stored,rtol=1e-10,atol=1e-8),observed=err,expected="<=1e-8 abs"))
    return report("V07",ctx,checks)
