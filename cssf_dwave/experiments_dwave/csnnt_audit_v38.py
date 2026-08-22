"""Runtime audit artifacts for every claim-bearing CSNN-T/GCV fit.

The verifier must be able to recompute the frozen Tikhonov solution from the
actual training matrix and target matrix used in the experiment.  These
artifacts are generated at experiment runtime; they are not scientific source
replacements and do not alter the immutable CSSF core.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from experiments_dwave.evidence_v38 import array_hash, canonical_json_hash


class CSNNTAuditError(RuntimeError):
    pass


def _unwrap_model(model: Any) -> tuple[Any, str]:
    """Return an object exposing H/lam_opt/predict and its surrogate level."""
    # CSSFQAResponseModel -> CSNNTSurrogateModel
    if hasattr(model, "model") and hasattr(model.model, "H"):
        model = model.model
    if hasattr(model, "frozen_model") and hasattr(model, "level"):
        level = getattr(model.level, "value", str(model.level))
        return model, str(level)
    if hasattr(model, "H") and hasattr(model, "lam_opt"):
        return model, "OPF"
    raise CSNNTAuditError(f"Unsupported CSNN-T model type: {type(model).__name__}")


def _coefficient_hash(H: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(H, dtype=np.complex128))
    h = hashlib.sha256(b"CSSF-CSNNT-H-v38\0")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def write_csnnt_fit_audit(
    evidence_root: str | Path,
    *,
    evidence_id: str,
    model: Any,
    training_features: Any,
    training_targets: Any,
    training_partition_ids: Iterable[str],
    call_path: str,
    metadata: Mapping[str, Any] | None = None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Append one independently recomputable CSNN-T fit record.

    ``training_features`` and ``training_targets`` must be exactly the matrices
    passed to the frozen fit path.  The complete matrices are compressed into
    an NPZ because V02 must not trust a notebook-provided scalar diagnostic.
    """
    root = Path(evidence_root)
    audit_dir = root / "csnnt_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    eid = str(evidence_id).strip()
    if not eid:
        raise CSNNTAuditError("evidence_id is required")
    X = np.ascontiguousarray(np.asarray(training_features, dtype=np.complex128))
    y = np.asarray(training_targets)
    if y.ndim == 1:
        y = y[:, None]
    y = np.ascontiguousarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 2 or X.shape[0] != y.shape[0]:
        raise CSNNTAuditError("Training feature/target matrices are not aligned")
    if not np.isfinite(X.real).all() or not np.isfinite(X.imag).all() or not np.isfinite(y).all():
        raise CSNNTAuditError("Audit matrices contain non-finite values")
    wrapped, level = _unwrap_model(model)
    H = np.ascontiguousarray(np.asarray(wrapped.H, dtype=np.complex128))
    pred = np.ascontiguousarray(np.real(X @ H), dtype=np.float64)
    if H.shape != (X.shape[1], y.shape[1]):
        raise CSNNTAuditError(f"Coefficient shape {H.shape} is incompatible with X/y")
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in eid)
    rel = Path("csnnt_audits") / f"{safe}.npz"
    np.savez_compressed(root / rel, X=X, y=y, H=H, pred=pred)
    ids = tuple(map(str, training_partition_ids))
    if len(ids) != X.shape[0] or len(set(ids)) != len(ids):
        raise CSNNTAuditError("training_partition_ids must be unique and aligned with X")
    rec = {
        "evidence_id": eid,
        "audit_npz": rel.as_posix(),
        "lambda": float(wrapped.lam_opt),
        "tolerance": float(tolerance),
        "call_path": str(call_path),
        "surrogate_level": level,
        "feature_fingerprint": array_hash(X, prefix="CSSF-CSNNT-X-v38"),
        "target_fingerprint": array_hash(y, prefix="CSSF-CSNNT-y-v38"),
        "train_partition_fingerprint": canonical_json_hash(ids, prefix="CSSF-CSNNT-partition-v38"),
        "model_coefficient_sha256": _coefficient_hash(H),
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "targets": int(y.shape[1]),
        "metadata": {} if metadata is None else dict(metadata),
    }
    manifest_path = root / "csnnt_fits.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema": "CSSF-QA-CSNNT-FITS-v38", "fits": []}
    existing = [r for r in manifest.get("fits", []) if str(r.get("evidence_id")) != eid]
    existing.append(rec)
    existing.sort(key=lambda r: str(r.get("evidence_id")))
    manifest["fits"] = existing
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return rec


__all__ = ["CSNNTAuditError", "write_csnnt_fit_audit"]
