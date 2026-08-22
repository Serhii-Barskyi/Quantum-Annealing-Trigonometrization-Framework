from __future__ import annotations
import numpy as np
from benchmarks.reference_competitors import calibrated_sqa_drive_schedule
from experiments_dwave.operator_phase import APPROVED_FAMILIES, load_calibration
from .common import *

def verify(ctx:VerificationContext)->dict:
    if ctx.mode!="simulator": return report("V08",ctx,[check("not_simulator_claim",True)])
    p=maybe_json(ctx.evidence_root/"simulator_schedule_sensitivity.json")
    if not p:
        return report("V08",ctx,[check("cuda_schedule_evidence_present",False)],status=NOT_RUN_HARDWARE if not ctx.live_hardware_expected else FAIL)
    checks=[check("device_cuda",str(p.get("device","")).lower()=="cuda"),check("no_classical_fallback",p.get("classical_fallback") is False)]
    rows=p.get("schedules",[]); checks.append(check("two_distinct_schedules",len(rows)>=2))
    drive_hashes=[]; outcome_hashes=[]
    for row in rows:
        fam=row.get("family");
        if fam not in APPROVED_FAMILIES: checks.append(check(f"{row.get('control_id')}:family",False)); continue
        filename,_=APPROVED_FAMILIES[fam]; curve=load_calibration(ctx.project_root/"calibration"/filename,fam)
        cal={"s":curve.s,"A_GHz":curve.A_GHz,"B_GHz":curve.B_GHz}
        drive=calibrated_sqa_drive_schedule(row["t_us"],row["s"],cal,anneal_steps=int(row["anneal_steps"]),beta_range=tuple(row.get("beta_range",[0.1,5.0])))
        stored=np.asarray(row.get("beta_eff"),dtype=float); field=np.asarray(row.get("field_eff"),dtype=float)
        checks.append(check(f"{row.get('control_id')}:beta_eff",stored.shape==drive["beta_eff"].shape and np.allclose(stored,drive["beta_eff"],rtol=1e-10,atol=1e-12)))
        checks.append(check(f"{row.get('control_id')}:field_eff",field.shape==drive["field_eff"].shape and np.allclose(field,drive["field_eff"],rtol=1e-10,atol=1e-12)))
        checks.append(check(f"{row.get('control_id')}:mapping_invariant",float(drive["mapping_product_max_abs_error"])<=1e-10,observed=float(drive["mapping_product_max_abs_error"]),expected="<=1e-10"))
        drive_hashes.append(json_hash({"b":stored.tolist(),"f":field.tolist()})); outcome_hashes.append(str(row.get("outcome_fingerprint","")))
    if len(rows)>=2:
        checks.append(check("distinct_calibrated_trajectories",len(set(drive_hashes))>1))
        checks.append(check("schedule_sensitive_outcomes",len(set(outcome_hashes))>1 and all(outcome_hashes)))
    return report("V08",ctx,checks)
