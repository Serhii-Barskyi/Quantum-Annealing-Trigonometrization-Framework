"""Claim-gated EEG phase/syntax utilities for CSSF(QA) v51.

This module implements mathematical and evidence-integrity machinery for the
active EEG microstate application.  It deliberately does not manufacture
real-EEG or live-QPU evidence: external-data/hardware experiments remain
fail-closed until their protected artifacts are present.

The canonical CSSF estimator is imported from the frozen core.  This module
never edits or substitutes ``core/csnn_t.py`` or ``core/gcv.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import os
import tempfile

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr

from core.gcv import effective_rank, gcv_lambda

SCHEMA = "CSSF-QA-EEG-PHASE-SYNTAX-v51"
EEG_SEED = 20260817


class EEGPhaseSyntaxError(RuntimeError):
    """Fail-closed error for EEG application evidence."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(dict(payload)), fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return target


def _hash_payload(payload: Mapping[str, Any], prefix: str) -> str:
    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(prefix.encode() + b"\0" + raw).hexdigest()


def wrap_phase(theta: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(theta, dtype=float)
    if not np.isfinite(x).all():
        raise EEGPhaseSyntaxError("phase values must be finite")
    return np.angle(np.exp(1j * x)).astype(np.float64)


def aligned_lagged_features(
    signal: ArrayLike,
    labels: ArrayLike,
    *,
    window: int,
    horizon: int = 1,
) -> tuple[NDArray[np.float64], NDArray, NDArray[np.int64], NDArray[np.int64]]:
    """Build past-only features X[t-window+1:t+1] -> label[t+horizon].

    Returned ``feature_end`` and ``target_time`` arrays make the no-leakage
    relation auditable.  ``horizon`` must be strictly positive.
    """
    x = np.asarray(signal, dtype=float)
    y = np.asarray(labels)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise EEGPhaseSyntaxError("signal must be (time, channels) and align with 1D labels")
    w, h = int(window), int(horizon)
    if w < 1 or h < 1 or x.shape[0] <= w + h:
        raise EEGPhaseSyntaxError("window/horizon are invalid for the available sequence")
    rows: list[np.ndarray] = []
    targets: list[Any] = []
    feature_end: list[int] = []
    target_time: list[int] = []
    for end in range(w - 1, x.shape[0] - h):
        target = end + h
        rows.append(x[end - w + 1 : end + 1].reshape(-1))
        targets.append(y[target])
        feature_end.append(end)
        target_time.append(target)
    X = np.vstack(rows).astype(np.float64)
    Y = np.asarray(targets)
    fe = np.asarray(feature_end, dtype=np.int64)
    tt = np.asarray(target_time, dtype=np.int64)
    assert_prediction_alignment(fe, tt, minimum_horizon=h)
    return X, Y, fe, tt


def assert_prediction_alignment(
    feature_end: ArrayLike,
    target_time: ArrayLike,
    *,
    minimum_horizon: int = 1,
) -> dict[str, Any]:
    fe = np.asarray(feature_end, dtype=int).reshape(-1)
    tt = np.asarray(target_time, dtype=int).reshape(-1)
    if fe.shape != tt.shape or fe.size == 0:
        raise EEGPhaseSyntaxError("feature/target time arrays must be aligned and non-empty")
    gap = tt - fe
    h = int(minimum_horizon)
    if h < 1 or np.any(gap < h):
        raise EEGPhaseSyntaxError("target leakage or prediction-horizon mismatch detected")
    return {
        "schema": SCHEMA + "-ALIGNMENT",
        "execution_complete": True,
        "n_examples": int(fe.size),
        "minimum_observed_horizon": int(gap.min()),
        "maximum_observed_horizon": int(gap.max()),
        "required_minimum_horizon": h,
        "leakage_detected": False,
    }


def phase_locking_curve(phases: ArrayLike, *, max_lag: int) -> NDArray[np.float64]:
    """Mean multichannel phase-locking magnitude versus positive time lag."""
    th = np.asarray(phases, dtype=float)
    if th.ndim == 1:
        th = th[:, None]
    if th.ndim != 2 or th.shape[0] < 8 or not np.isfinite(th).all():
        raise EEGPhaseSyntaxError("phases must be finite with shape (time, channels)")
    ml = int(max_lag)
    if ml < 1 or ml >= th.shape[0] // 2:
        raise EEGPhaseSyntaxError("max_lag is invalid")
    curve = np.empty(ml, dtype=float)
    for lag in range(1, ml + 1):
        delta = th[lag:] - th[:-lag]
        per_channel = np.abs(np.mean(np.exp(1j * delta), axis=0))
        curve[lag - 1] = float(np.mean(per_channel))
    return curve


def phase_periodicity_audit(
    phases: ArrayLike,
    *,
    max_lag: int,
    candidate_lag: int | None = None,
    local_radius: int = 2,
) -> dict[str, Any]:
    """Audit repeatable phase structure without claiming a physiological mechanism."""
    curve = phase_locking_curve(phases, max_lag=max_lag)
    if candidate_lag is None:
        # Avoid trivial lag-1 selection where possible.
        start = min(2, curve.size - 1)
        idx = int(np.argmax(curve[start:]) + start)
        lag = idx + 1
    else:
        lag = int(candidate_lag)
        if lag < 1 or lag > curve.size:
            raise EEGPhaseSyntaxError("candidate_lag is outside the audited range")
        idx = lag - 1
    r = max(1, int(local_radius))
    left = max(0, idx - r)
    right = min(curve.size, idx + r + 1)
    neighbors = np.concatenate([curve[left:idx], curve[idx + 1 : right]])
    local_prom = float(curve[idx] - np.median(neighbors)) if neighbors.size else 0.0
    return {
        "schema": SCHEMA + "-PHASE-PERIODICITY",
        "execution_complete": True,
        "candidate_lag": lag,
        "phase_locking_at_candidate": float(curve[idx]),
        "local_prominence": local_prom,
        "max_lag": int(max_lag),
        "curve": curve.tolist(),
        "claim_boundary": "numerical phase recurrence only; native EEG periodicity still requires protected real-data confirmation and preprocessing sensitivity",
    }


def markov_transition_matrix(labels: ArrayLike, *, n_states: int | None = None, pseudocount: float = 0.0) -> NDArray[np.float64]:
    seq = np.asarray(labels, dtype=int).reshape(-1)
    if seq.size < 3 or np.any(seq < 0):
        raise EEGPhaseSyntaxError("labels must contain at least three nonnegative states")
    k = int(n_states) if n_states is not None else int(seq.max()) + 1
    if k <= int(seq.max()):
        raise EEGPhaseSyntaxError("n_states is smaller than an observed label")
    pc = float(pseudocount)
    if pc < 0:
        raise EEGPhaseSyntaxError("pseudocount must be nonnegative")
    counts = np.full((k, k), pc, dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        counts[a, b] += 1.0
    rows = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, rows, out=np.full_like(counts, 1.0 / k), where=rows > 0)
    return P


def first_order_markov_surrogates(
    labels: ArrayLike,
    *,
    n_surrogates: int,
    seed: int = EEG_SEED,
) -> NDArray[np.int64]:
    seq = np.asarray(labels, dtype=int).reshape(-1)
    P = markov_transition_matrix(seq)
    k = P.shape[0]
    n = int(n_surrogates)
    if n < 1:
        raise EEGPhaseSyntaxError("n_surrogates must be positive")
    rng = np.random.default_rng(int(seed))
    out = np.empty((n, seq.size), dtype=np.int64)
    initial_p = np.bincount(seq, minlength=k).astype(float)
    initial_p /= initial_p.sum()
    for r in range(n):
        out[r, 0] = int(rng.choice(k, p=initial_p))
        for t in range(1, seq.size):
            out[r, t] = int(rng.choice(k, p=P[out[r, t - 1]]))
    return out


def _validate_counts(counts: ArrayLike) -> NDArray[np.float64]:
    c = np.asarray(counts, dtype=float)
    if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] < 2 or np.any(c < 0) or not np.isfinite(c).all():
        raise EEGPhaseSyntaxError("count matrix must be finite nonnegative (contexts, states)")
    if np.any(c.sum(axis=1) <= 0):
        raise EEGPhaseSyntaxError("each context must contain observations")
    return c


def pooled_training_nll(counts: ArrayLike, assignment: ArrayLike, *, pseudocount: float = 0.5) -> float:
    """Exact group-level multinomial NLL on the supplied count matrix."""
    c = _validate_counts(counts)
    a = np.asarray(assignment, dtype=int).reshape(-1)
    if a.size != c.shape[0] or np.any(a < 0):
        raise EEGPhaseSyntaxError("assignment must map every context to a nonnegative group")
    pc = float(pseudocount)
    if pc <= 0:
        raise EEGPhaseSyntaxError("pseudocount must be positive")
    nll = 0.0
    for g in np.unique(a):
        members = c[a == g]
        pooled = members.sum(axis=0) + pc
        p = pooled / pooled.sum()
        nll -= float(np.sum(members * np.log(p)))
    return nll


def heldout_partition_nll(
    train_counts: ArrayLike,
    confirm_counts: ArrayLike,
    assignment: ArrayLike,
    *,
    pseudocount: float = 0.5,
) -> float:
    tr = _validate_counts(train_counts)
    cf = _validate_counts(confirm_counts)
    if tr.shape != cf.shape:
        raise EEGPhaseSyntaxError("train/confirmation count matrices must have identical shape")
    a = np.asarray(assignment, dtype=int).reshape(-1)
    if a.size != tr.shape[0] or np.any(a < 0):
        raise EEGPhaseSyntaxError("assignment must map every context")
    pc = float(pseudocount)
    if pc <= 0:
        raise EEGPhaseSyntaxError("pseudocount must be positive")
    nll = 0.0
    for g in np.unique(a):
        pooled = tr[a == g].sum(axis=0) + pc
        p = pooled / pooled.sum()
        nll -= float(np.sum(cf[a == g] * np.log(p)))
    return nll


def pairwise_merge_cost_matrix(counts: ArrayLike) -> NDArray[np.float64]:
    """Pairwise classification-loss bridge used only as a surrogate diagnostic."""
    c = _validate_counts(counts)
    n = c.shape[0]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            separate = float(c[i].max() + c[j].max())
            merged = c[i] + c[j]
            pooled_correct = float(c[i, np.argmax(merged)] + c[j, np.argmax(merged)])
            out[i, j] = out[j, i] = separate - pooled_correct
    return out


def pairwise_partition_energy(cost_matrix: ArrayLike, assignment: ArrayLike) -> float:
    W = np.asarray(cost_matrix, dtype=float)
    a = np.asarray(assignment, dtype=int).reshape(-1)
    if W.shape != (a.size, a.size) or not np.allclose(W, W.T) or not np.isfinite(W).all():
        raise EEGPhaseSyntaxError("cost matrix must be finite symmetric and align with assignment")
    e = 0.0
    for i in range(a.size):
        for j in range(i + 1, a.size):
            if a[i] == a[j]:
                e += float(W[i, j])
    return e


def partition_bridge_fidelity(
    true_scores: ArrayLike,
    bridge_scores: ArrayLike,
    *,
    lower_is_better: bool = True,
    top_k: int = 5,
) -> dict[str, Any]:
    y = np.asarray(true_scores, dtype=float).reshape(-1)
    q = np.asarray(bridge_scores, dtype=float).reshape(-1)
    if y.shape != q.shape or y.size < 4 or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise EEGPhaseSyntaxError("bridge fidelity requires aligned finite score arrays with n>=4")
    sign = -1.0 if lower_is_better else 1.0
    uy, uq = sign * y, sign * q
    rho = float(spearmanr(uy, uq).statistic)
    if not np.isfinite(rho):
        rho = 0.0
    k = max(1, min(int(top_k), y.size))
    t = set(np.argsort(-uy)[:k].tolist())
    p = set(np.argsort(-uq)[:k].tolist())
    best_bridge_idx = int(np.argmax(uq))
    regret = float(np.max(uy) - uy[best_bridge_idx])
    return {
        "schema": SCHEMA + "-PARTITION-BRIDGE-FIDELITY",
        "execution_complete": True,
        "n_partitions": int(y.size),
        "spearman": rho,
        "topk_recall": float(len(t & p) / k),
        "decision_regret": regret,
        "lower_is_better": bool(lower_is_better),
        "claim_boundary": "diagnostic of quadratic bridge fidelity; does not certify global discrete-search optimality",
    }


def qubo_assignment_graph_stats(cost_matrix: ArrayLike, *, n_groups: int) -> dict[str, Any]:
    W = np.asarray(cost_matrix, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] < 2 or not np.allclose(W, W.T):
        raise EEGPhaseSyntaxError("cost matrix must be symmetric square")
    n_ctx = W.shape[0]
    G = int(n_groups)
    if G < 2:
        raise EEGPhaseSyntaxError("n_groups must be >=2")
    nz_pairs = int(np.count_nonzero(np.triu(np.abs(W) > 0, k=1)))
    objective_edges = nz_pairs * G
    onehot_edges = n_ctx * (G * (G - 1) // 2)
    logical_variables = n_ctx * G
    # Degree for variable (context i, group g): nonzero context neighbors + G-1 one-hot peers.
    ctx_degree = np.count_nonzero(np.abs(W) > 0, axis=1)
    logical_degrees = np.repeat(ctx_degree + (G - 1), G)
    return {
        "schema": SCHEMA + "-QUBO-GRAPH-STATS",
        "execution_complete": True,
        "n_contexts": int(n_ctx),
        "n_groups": G,
        "logical_variables": int(logical_variables),
        "nonzero_context_pairs": nz_pairs,
        "objective_quadratic_interactions": int(objective_edges),
        "onehot_quadratic_interactions": int(onehot_edges),
        "total_quadratic_interactions": int(objective_edges + onehot_edges),
        "degree_min": int(logical_degrees.min()),
        "degree_median": float(np.median(logical_degrees)),
        "degree_max": int(logical_degrees.max()),
        "embedding_required": True,
        "claim_boundary": "logical graph statistics only; physical Pegasus embedding must be measured independently",
    }


def gcv_real_target_diagnostics(
    X: ArrayLike,
    Y: ArrayLike,
    *,
    n_lambdas: int = 80,
    lam_range: tuple[float, float] = (-8.0, 8.0),
    boundary_tol: int = 0,
) -> dict[str, Any]:
    """Run frozen GCV with an explicit real-target and boundary-stability gate."""
    x = np.asarray(X, dtype=complex)
    y = np.asarray(Y)
    if np.iscomplexobj(y) and np.any(np.abs(np.imag(y)) > 0):
        raise EEGPhaseSyntaxError("canonical frozen GCV is restricted to real-valued EEG targets")
    yr = np.asarray(np.real(y), dtype=float)
    if x.ndim != 2 or yr.shape[0] != x.shape[0] or not np.isfinite(x).all() or not np.isfinite(yr).all():
        raise EEGPhaseSyntaxError("X/Y must be finite and row-aligned")
    lam, grid, vals = gcv_lambda(x, yr, n_lambdas=int(n_lambdas), lam_range=lam_range)
    idx = int(np.argmin(vals))
    tol = max(0, int(boundary_tol))
    boundary = bool(idx <= tol or idx >= len(grid) - 1 - tol)
    rank = float(effective_rank(x, lam))
    dof = float(x.shape[0] - rank)
    return {
        "schema": SCHEMA + "-GCV-DIAGNOSTICS",
        "execution_complete": True,
        "lambda": float(lam),
        "grid_index": idx,
        "grid_size": int(len(grid)),
        "grid_log10_range": [float(lam_range[0]), float(lam_range[1])],
        "boundary_optimum": boundary,
        "effective_rank": rank,
        "residual_dof": dof,
        "stable_for_claim": bool(not boundary and dof >= 1.0),
    }


def weak_coupling_exponent_audit(
    eps: ArrayLike,
    residual_scale: ArrayLike,
    *,
    admissible: ArrayLike | None = None,
) -> dict[str, Any]:
    e = np.asarray(eps, dtype=float).reshape(-1)
    r = np.asarray(residual_scale, dtype=float).reshape(-1)
    if e.shape != r.shape or e.size < 4 or np.any(e <= 0) or np.any(r <= 0) or not np.isfinite(e).all() or not np.isfinite(r).all():
        raise EEGPhaseSyntaxError("weak-coupling audit requires >=4 positive finite aligned points")
    keep = np.ones(e.size, dtype=bool) if admissible is None else np.asarray(admissible, dtype=bool).reshape(-1)
    if keep.shape != e.shape or np.count_nonzero(keep) < 4:
        raise EEGPhaseSyntaxError("at least four predeclared admissible points are required")
    x = np.log(e[keep]); y = np.log(r[keep])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / denom if denom > 0 else 0.0
    return {
        "schema": SCHEMA + "-WEAK-COUPLING-EXPONENT",
        "execution_complete": True,
        "n_total": int(e.size),
        "n_admissible": int(np.count_nonzero(keep)),
        "slope": float(slope),
        "r2_loglog": float(r2),
        "claim_boundary": "numerical reduction-error scaling only; exact Stuart-Landau phase identity is a separate analytical statement",
    }


@dataclass(frozen=True)
class EEGResultSchema:
    experiment_id: str
    requires_real_eeg: bool = False
    requires_live_qpu: bool = False
    requires_replication: bool = False
    required_fields: tuple[str, ...] = ("execution_complete", "claim_passed")


def validate_result_payload(payload: Mapping[str, Any], schema: EEGResultSchema) -> dict[str, Any]:
    missing = [k for k in schema.required_fields if k not in payload]
    if missing:
        return {"valid": False, "missing_fields": missing, "claim_ready": False}
    claim = bool(payload.get("claim_passed", False))
    if schema.requires_real_eeg and not bool(payload.get("real_eeg_confirmed", False)):
        claim = False
    if schema.requires_live_qpu and not bool(payload.get("live_qpu_confirmed", False)):
        claim = False
    if schema.requires_replication and not bool(payload.get("replication_confirmed", False)):
        claim = False
    return {"valid": True, "missing_fields": [], "claim_ready": bool(payload.get("execution_complete", False) and claim)}


def eeg_fail_closed_gate(
    evidence_root: str | Path,
    experiment_schemas: Sequence[EEGResultSchema],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(evidence_root)
    rows: list[dict[str, Any]] = []
    for schema in experiment_schemas:
        p = root / (schema.experiment_id.lower().replace("/", "_") + ".json")
        payload = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        audit = validate_result_payload(payload, schema) if p.exists() else {"valid": False, "missing_fields": list(schema.required_fields), "claim_ready": False}
        rows.append({
            "experiment_id": schema.experiment_id,
            "exists": p.exists(),
            "execution_complete": bool(payload.get("execution_complete", False)),
            "claim_passed": bool(payload.get("claim_passed", False)),
            "claim_ready": bool(audit["claim_ready"]),
            "path": str(p),
        })
    out = {
        "schema": SCHEMA + "-FAIL-CLOSED-GATE",
        "execution_complete": True,
        "experiments": rows,
        "all_executed": bool(rows and all(r["execution_complete"] for r in rows)),
        "all_claim_ready": bool(rows and all(r["claim_ready"] for r in rows)),
        "claim_status": "READY_FOR_CLAIM_REVIEW" if rows and all(r["claim_ready"] for r in rows) else "PENDING_OR_NULL",
        "placeholder_is_evidence": False,
    }
    if output_path is not None:
        _atomic_json(output_path, out)
    return out


__all__ = [
    "SCHEMA", "EEG_SEED", "EEGPhaseSyntaxError", "EEGResultSchema",
    "wrap_phase", "aligned_lagged_features", "assert_prediction_alignment",
    "phase_locking_curve", "phase_periodicity_audit", "markov_transition_matrix",
    "first_order_markov_surrogates", "pooled_training_nll", "heldout_partition_nll",
    "pairwise_merge_cost_matrix", "pairwise_partition_energy", "partition_bridge_fidelity",
    "qubo_assignment_graph_stats", "gcv_real_target_diagnostics", "weak_coupling_exponent_audit",
    "validate_result_payload", "eeg_fail_closed_gate",
]
