"""Versioned machine-readable evidence utilities for the D-Wave proof program.

This extension never modifies the frozen CSSF scientific source.  It records
runtime lineage, event accounting and artifact hashes consumed by contract
verifiers V00--V19.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SCHEMA_VERSION = "CSSF-QA-EVIDENCE-v38"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_hash(value: Any, *, prefix: str = "CSSF-v38") -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(prefix.encode("utf-8") + b"\0" + payload).hexdigest()


def array_hash(value: Any, *, prefix: str = "CSSF-array-v38") -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256(prefix.encode("ascii") + b"\0")
    h.update(arr.dtype.str.encode("ascii")); h.update(b"\0")
    h.update(str(tuple(arr.shape)).encode("ascii")); h.update(b"\0")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def qpu_access_time_us(response: Mapping[str, Any]) -> float:
    """Extract D-Wave QPU access time from normalized or raw Ocean info."""
    info = dict(response.get("backend_info", {}) or {})
    direct = info.get("qpu_access_time")
    if direct is not None:
        return float(direct or 0.0)
    timing = dict(info.get("timing", {}) or {})
    return float(timing.get("qpu_access_time", 0.0) or 0.0)


def evidence_id(stage: str, payload: Mapping[str, Any]) -> str:
    return f"{stage}:{canonical_json_hash(dict(payload), prefix='CSSF-evidence-id-v38')[:24]}"


@dataclass(frozen=True)
class LineageNode:
    evidence_id: str
    stage: str
    parents: tuple[str, ...]
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    metadata: Mapping[str, Any]


class EvidenceRecorder:
    """Append-only recorder for one proof run."""

    def __init__(self, root: str | Path, *, run_id: str, mode: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.mode = str(mode)
        self.nodes: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
        return path

    def record_node(
        self,
        *,
        stage: str,
        parents: Iterable[str] = (),
        input_ids: Iterable[str] = (),
        output_ids: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
        node_id: str | None = None,
    ) -> str:
        meta = {} if metadata is None else dict(metadata)
        payload = {
            "stage": str(stage),
            "parents": list(parents),
            "input_ids": list(input_ids),
            "output_ids": list(output_ids),
            "metadata": meta,
        }
        eid = str(node_id or evidence_id(str(stage), payload))
        self.nodes.append({"evidence_id": eid, **payload})
        return eid

    def record_event(self, *, method: str, event_type: str, amount: float = 1.0, resource: str = "control_evaluation", metadata: Mapping[str, Any] | None = None) -> None:
        self.events.append({
            "run_id": self.run_id,
            "method": str(method),
            "event_type": str(event_type),
            "resource": str(resource),
            "amount": float(amount),
            "metadata": {} if metadata is None else dict(metadata),
            "timestamp_utc": utc_now(),
        })

    def finalize(self, *, run_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
        lineage = {"schema": SCHEMA_VERSION, "run_id": self.run_id, "mode": self.mode, "nodes": self.nodes}
        self.write_json("lineage.json", lineage)
        event_path = self.root / "resource_events.jsonl"
        # Resource accounting may be independently reconstructed from MethodTrace
        # objects before lineage finalization.  Never erase that authoritative
        # ledger merely because the recorder itself has no additional events.
        if self.events or not event_path.exists():
            with event_path.open("w", encoding="utf-8") as stream:
                for row in self.events:
                    stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        manifest = {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": self.mode,
            "created_utc": utc_now(),
            **({} if run_manifest is None else dict(run_manifest)),
        }
        self.write_json("run_manifest.json", manifest)
        return manifest


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "SCHEMA_VERSION", "utc_now", "sha256_file", "canonical_json_hash", "array_hash",
    "evidence_id", "qpu_access_time_us", "LineageNode", "EvidenceRecorder", "load_json",
]
