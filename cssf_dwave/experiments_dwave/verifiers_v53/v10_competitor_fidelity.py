from __future__ import annotations
from experiments_dwave.claim_set_v38 import PRIMARY_EXTERNAL_COMPARATORS
from .common import *
FIELDS=("algorithm","state","init","surrogate","acquisition","adaptation","restart","feasibility","budget","objective","accounting","provenance")

def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"competitor_fidelity.json")
    if not p or not p.get("competitors"): return report("V10",ctx,[check("competitor_manifest_present",False)])
    rows={str(c.get("name")):c for c in p["competitors"]}; declared=maybe_json(ctx.evidence_root/"external_claim_set.json") or {}
    required=tuple(declared.get("comparators",PRIMARY_EXTERNAL_COMPARATORS))
    checks=[check("external_claim_set_exact",set(rows)>=set(required),observed=sorted(rows),expected=sorted(required))]
    for name in required:
        c=rows.get(name,{}) ; fidelity=c.get("fidelity",{})
        checks.append(check(f"{name}:fidelity_product",all(int(fidelity.get(k,0))==1 for k in FIELDS),observed={k:fidelity.get(k) for k in FIELDS},expected={k:1 for k in FIELDS}))
        for k in ("implementation_sha256","reference_provenance","mechanism_test_report","hyperparameters","budget_manifest"):
            checks.append(check(f"{name}:{k}",bool(c.get(k))))
        checks.append(check(f"{name}:full_function",c.get("full_function") is True))
    return report("V10",ctx,checks)
