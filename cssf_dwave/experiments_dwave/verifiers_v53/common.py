"""Shared fail-closed verifier primitives for Contract v36/v38."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN_HARDWARE = "NOT_RUN_HARDWARE"
VERIFIER_VERSION = "v38"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def json_hash(value: Any, prefix: str = "CSSF-verifier-v38") -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False,default=str).encode()
    return hashlib.sha256(prefix.encode()+b"\0"+raw).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_json(path: Path) -> Any | None:
    return load_json(path) if path.is_file() else None


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    if not path.is_file(): return []
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows


def check(name: str, ok: bool, *, observed: Any = None, expected: Any = None) -> dict[str,Any]:
    return {"name":name,"status":PASS if bool(ok) else FAIL,"observed":observed,"expected":expected}


@dataclass(frozen=True)
class VerificationContext:
    project_root: Path
    evidence_root: Path
    notebook_path: Path | None = None
    mode: str = "simulator"
    live_hardware_expected: bool = False
    execute_expensive_checks: bool = False

    @classmethod
    def build(cls, project_root: str | Path, evidence_root: str | Path, **kwargs: Any) -> "VerificationContext":
        return cls(Path(project_root).resolve(),Path(evidence_root).resolve(),**kwargs)

    @property
    def manifest(self) -> dict[str,Any]:
        return maybe_json(self.evidence_root/"run_manifest.json") or {}


def software_manifest_sha256(project_root: Path) -> str:
    candidates=[
        project_root/"releases"/"scientific_manifests"/"FROZEN_SOURCE_MANIFEST_v51.json",
    ]
    for p in candidates:
        if p.is_file(): return sha256_file(p)
    return "0"*64


def report(verifier_id: str, ctx: VerificationContext, checks: Iterable[Mapping[str,Any]], *, status: str | None = None, evidence_ids: Iterable[str] = (), inputs: Iterable[Mapping[str,Any]] = ()) -> dict[str,Any]:
    rows=[dict(x) for x in checks]
    if status is None:
        status=PASS if rows and all(x.get("status")==PASS for x in rows) else FAIL
    if status not in {PASS,FAIL,NOT_RUN_HARDWARE}: raise ValueError(status)
    return {
        "verifier_id":verifier_id,"version":VERIFIER_VERSION,"status":status,
        "inputs":list(inputs),"evidence_ids":list(evidence_ids),"checks":rows,
        "timestamp_utc":utc_now(),"software_manifest_sha256":software_manifest_sha256(ctx.project_root),
    }


def flatten_dict(value: Mapping[str,Any], prefix: str = "") -> dict[str,Any]:
    out={}
    for key,val in value.items():
        name=f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val,Mapping): out.update(flatten_dict(val,name))
        else: out[name]=val
    return out


def graph_path(nodes: list[dict[str,Any]], stages: list[str]) -> tuple[bool,list[str]]:
    """Find any parent-linked evidence path through stage labels in order.

    Multiple factorial arms may emit the same stage.  The verifier therefore
    searches all candidates rather than trusting insertion order.
    """
    by_id={str(n.get("evidence_id")):n for n in nodes if n.get("evidence_id")}
    parents={eid:set(map(str,n.get("parents",[]))) for eid,n in by_id.items()}
    memo: dict[str,set[str]]={}
    def anc(eid:str, trail:set[str]|None=None)->set[str]:
        if eid in memo: return memo[eid]
        trail=set() if trail is None else set(trail)
        if eid in trail: return set()
        trail.add(eid); seen=set()
        for p in parents.get(eid,set()):
            seen.add(p); seen.update(anc(p,trail))
        memo[eid]=seen; return seen
    candidates=[[eid for eid,n in by_id.items() if n.get("stage")==stage] for stage in stages]
    if any(not c for c in candidates): return False,[]
    paths=[[eid] for eid in candidates[0]]
    for group in candidates[1:]:
        next_paths=[]
        for path in paths:
            prev=path[-1]
            for eid in group:
                if prev in anc(eid): next_paths.append(path+[eid])
        paths=next_paths
        if not paths: return False,[]
    paths.sort(key=lambda p:tuple(p))
    return True,paths[0]
