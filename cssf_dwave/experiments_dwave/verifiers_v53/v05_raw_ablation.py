from __future__ import annotations
from .common import *

ALLOWED_DIFF={"representation.coordinate_transform","representation.name"}


def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"raw_trig_ablation.json")
    if not p:
        return report("V05",ctx,[check("raw_trig_ablation_present",False)])
    raw=p.get("CSSF-raw/no-trig",{}); trig=p.get("CSSF-trig",{})
    fr=flatten_dict(raw.get("config",{})); ft=flatten_dict(trig.get("config",{})); keys=set(fr)|set(ft)
    diffs={k:(fr.get(k),ft.get(k)) for k in keys if fr.get(k)!=ft.get(k)}
    checks=[
        check("only_representation_diff",set(diffs)==ALLOWED_DIFF,observed=diffs,expected=sorted(ALLOWED_DIFF)),
        check("same_observation_ids",raw.get("observation_ids")==trig.get("observation_ids") and bool(raw.get("observation_ids"))),
        check("same_train_partition",raw.get("train_observation_ids")==trig.get("train_observation_ids") and bool(raw.get("train_observation_ids"))),
        check("same_calibration_partition",raw.get("calibration_observation_ids")==trig.get("calibration_observation_ids") and bool(raw.get("calibration_observation_ids"))),
        check("same_heldout_partition",raw.get("heldout_observation_ids")==trig.get("heldout_observation_ids") and bool(raw.get("heldout_observation_ids"))),
        check("same_candidate_pool",raw.get("candidate_pool_sha256")==trig.get("candidate_pool_sha256") and bool(raw.get("candidate_pool_sha256"))),
        check("same_targets",raw.get("target_names")==trig.get("target_names") and bool(raw.get("target_names"))),
        check("same_budget",raw.get("budget")==trig.get("budget") and bool(raw.get("budget"))),
        check("same_seeds",raw.get("seeds")==trig.get("seeds") and bool(raw.get("seeds"))),
    ]
    return report("V05",ctx,checks)
