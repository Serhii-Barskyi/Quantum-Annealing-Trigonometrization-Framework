from __future__ import annotations
import numpy as np
from experiments_dwave.claim_set_v38 import PRIMARY_EXTERNAL_COMPARATORS
from .common import *

def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"statistics.json")
    if not p: return report("V14",ctx,[check("statistics_present",False)])
    declared=maybe_json(ctx.evidence_root/"external_claim_set.json") or {}; required=tuple(declared.get("comparators",PRIMARY_EXTERNAL_COMPARATORS))
    rows={str(r.get("competitor")):r for r in p.get("comparisons",[])}; checks=[check("all_external_comparators_confirmed",set(rows)>=set(required),observed=sorted(rows),expected=sorted(required))]
    for name in required:
        row=rows.get(name,{}) ; d=np.asarray(row.get("paired_differences",[]),dtype=float); alpha=float(row.get("alpha",0.05)); margin=float(row.get("margin",0.0)); seed=int(row.get("bootstrap_seed",0)); B=int(row.get("bootstrap_samples",4000))
        if d.size<2 or not np.isfinite(d).all(): checks.append(check(f"{name}:paired_data",False)); continue
        rng=np.random.default_rng(seed); means=np.empty(B)
        for i in range(B): means[i]=float(np.mean(d[rng.integers(0,d.size,d.size)]))
        lcb=float(np.quantile(means,alpha)); stored=float(row.get("lcb",float("nan")))
        checks.append(check(f"{name}:lcb_recomputed",np.isclose(lcb,stored,rtol=0,atol=1e-12),observed=stored,expected=lcb))
        checks.append(check(f"{name}:superiority_gate",bool(row.get("superiority_passed"))==(lcb>margin),observed=row.get("superiority_passed"),expected=lcb>margin))
        checks.append(check(f"{name}:independent_confirmation",row.get("confirmation_partition") in {"confirmation","application_confirmation"}))
    return report("V14",ctx,checks)
