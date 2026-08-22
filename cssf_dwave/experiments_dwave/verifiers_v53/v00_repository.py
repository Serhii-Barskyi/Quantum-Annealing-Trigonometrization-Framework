from __future__ import annotations
import json
from .common import *

FORBIDDEN_TOKENS=("advantage2","zephyr")

def verify(ctx:VerificationContext)->dict:
    manifest_path=ctx.project_root/"releases"/"scientific_manifests"/"FROZEN_SOURCE_MANIFEST_v51.json"
    checks=[]; inputs=[]
    if not manifest_path.is_file(): return report("V00",ctx,[check("f0_manifest_present",False)])
    data=load_json(manifest_path); files=data.get("files",{})
    checks.append(check("frozen_source_count",len(files)==67,observed=len(files),expected=67))
    for rel,digest in files.items():
        p=ctx.project_root/rel; ok=p.is_file() and sha256_file(p)==digest
        checks.append(check(f"F0:{rel}",ok,observed=sha256_file(p) if p.is_file() else None,expected=digest))
    f1_path=ctx.project_root/"releases"/"scientific_manifests"/"FROZEN_INPUT_MANIFEST_v51.json"
    f1=load_json(f1_path).get("files",{}) if f1_path.is_file() else {}
    checks.append(check("frozen_input_manifest_present",bool(f1),observed=len(f1),expected=5))
    for rel,digest in f1.items():
        p=ctx.project_root/rel; checks.append(check(f"F1:{rel}",p.is_file() and sha256_file(p)==digest,observed=sha256_file(p) if p.is_file() else None,expected=digest))
    active=[]
    for p in list((ctx.project_root/"calibration").glob("*")) + list((ctx.project_root/"config").glob("*")):
        if p.is_file(): active.append(p.name.lower())
    forbidden=[x for x in active if any(t in x for t in FORBIDDEN_TOKENS)]
    checks.append(check("no_forbidden_hardware_family",not forbidden,observed=forbidden,expected=[]))

    # Active-release inventory is normative for the STATIC release tree.
    # Runtime evidence/checkpoints are intentionally outside this path-set check: they
    # are verified by evidence manifests/verifiers and must not invalidate source integrity.
    inventory_dir=ctx.project_root/"releases"/"scientific_manifests"
    inventory_candidates=[]
    for path in inventory_dir.glob("ACTIVE_RELEASE_INVENTORY_v*.json"):
        try:
            version=int(path.stem.rsplit("v",1)[1])
        except Exception:
            continue
        inventory_candidates.append((version,path))
    inventory_path=max(inventory_candidates,key=lambda item:item[0])[1] if inventory_candidates else None
    if inventory_path is not None and inventory_path.is_file():
        inventory=load_json(inventory_path)
        declared={str(row.get("path")) for row in inventory.get("files",[]) if row.get("path")}
        ignored_parts={"__pycache__",".pytest_cache"}
        runtime_roots={
            "results","checkpoints","outputs","logs","models","cache","artifacts",
            "embeddings","samplesets","solver_metadata",
        }
        actual=set()
        for candidate in ctx.project_root.rglob("*"):
            if not candidate.is_file():
                continue
            rel=candidate.relative_to(ctx.project_root)
            if rel.parts and rel.parts[0] in runtime_roots:
                continue
            if any(part in ignored_parts for part in rel.parts):
                continue
            if candidate.suffix in {".pyc",".pyo"} or candidate.name.endswith(".tmp"):
                continue
            actual.add(rel.as_posix())
        unexpected=sorted(actual-declared)
        missing=sorted(declared-actual)
        checks.append(check("active_inventory_selected",True,observed=inventory_path.name,expected=inventory_path.name))
        checks.append(check("active_inventory_exact_static_path_set",not unexpected and not missing,observed={"unexpected":unexpected,"missing":missing},expected={"unexpected":[],"missing":[]}))
    else:
        checks.append(check("active_inventory_present",False,observed=None,expected="ACTIVE_RELEASE_INVENTORY_v*.json"))
    return report("V00",ctx,checks,inputs=inputs)
