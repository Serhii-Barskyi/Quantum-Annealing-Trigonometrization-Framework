from __future__ import annotations
from .common import *

def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"reproducibility.json")
    if not p: return report("V17",ctx,[check("reproducibility_manifest_present",False)])
    checks=[
        check("seeds_present",isinstance(p.get("seeds"),dict) and bool(p.get("seeds"))),
        check("config_hashes_present",isinstance(p.get("config_hashes"),dict) and bool(p.get("config_hashes"))),
        check("environment_hash",bool(p.get("environment_manifest_sha256"))),
        check("code_hashes",isinstance(p.get("code_hashes"),dict) and bool(p.get("code_hashes"))),
        check("deterministic_artifacts_reproduced",p.get("deterministic_regeneration_passed") is True),
        check("stochastic_replay_state",p.get("stochastic_replay_state_recorded") is True),
    ]
    return report("V17",ctx,checks)
