from __future__ import annotations
from collections import Counter
from .common import *
PROTECTED=("train","calibration","validation","ood","n1","confirmation")

def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"partitions.json")
    if not p: return report("V16",ctx,[check("partition_manifest_present",False)])
    checks=[]; seen={}
    for name in PROTECTED:
        ids=list(p.get("partitions",{}).get(name,[])); checks.append(check(f"{name}:unique",len(ids)==len(set(ids))))
        for x in ids:
            if x in seen and not p.get("shared_initial_design",{}).get(x,False): checks.append(check(f"duplicate:{x}",False,observed=[seen[x],name],expected="disjoint"))
            else: seen[x]=name
    obs=p.get("observations",[]); prov=[x.get("provenance_id") for x in obs]
    checks.append(check("unique_provenance_ids",len(prov)==len(set(prov)) and all(prov)))
    checks.append(check("all_observations_partitioned",all(x.get("partition") in PROTECTED or x.get("partition")=="selection" for x in obs)))
    return report("V16",ctx,checks)
