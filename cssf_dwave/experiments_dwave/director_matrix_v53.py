"""D-Wave director-grade evidence extensions for CSSF(QA) v53.

This additive module implements the ten evidence closures D45-01..D45-10
retained from the validated evidence matrix and bound to the frozen v53 evidence protocol. It never modifies or bypasses the frozen
CSNN-T/GCV core.  Expensive sampling is restartable and one annealer evaluation
is the largest sampling unit performed by any ``advance_*`` function.

Scientific scope is immutable:
  * periodic/cyclic/toric application processes only;
  * quantum-annealing trigonometrization is the principal mechanism;
  * Pegasus P16, current/historical Advantage_system4 / Advantage_system6 Pegasus IDs only;
  * canonical core/csnn_t.py + core/gcv.py for CSSF response fitting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import math
import os
import tempfile

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal
from scipy.stats import spearmanr

from benchmarks import reference_competitors as rc
from core.dataset import CSSFDataset
from core.csnn_t_adapter_v53 import fit_csnn_t_surrogate
from core.types import SurrogateLevel
from experiments_dwave.application_endpoint_v38 import (
    build_endpoint_protocol,
    evaluate_confirmation_buses,
    select_placement_on_validation,
)
from experiments_dwave.benchmark_protocol import MatchedControlProtocol, TARGET_NAMES
from experiments_dwave.cssf_control_v53 import build_support, fit_cssf_qa_response
from experiments_dwave.evidence_v38 import canonical_json_hash
from experiments_dwave.factorial_runtime_v53 import select_placement_from_sampleset
from experiments_dwave.integrated_bess_v38 import raw_control_coordinates
from experiments_dwave.operator_phase import (
    APPROVED_FAMILIES,
    TWO_PI_GHZ_US,
    load_calibration,
    operator_action_coordinates,
)
from spectral.feature_matrix import toric_feature_matrix
from spectral.frequency_support import FrequencySupport, SupportKind

SCHEMA = "CSSF-QA-DIRECTOR-MATRIX-v53"
DIRECTOR_SEED = 20260817
SOFT_WALL_SECONDS = 1080
HARD_WALL_SECONDS = 1200
CCJJ_BANDWIDTH_MHZ = 6.5  # official 2026 D-Wave I/O-system cutoff for Advantage anneal waveform


class DirectorMatrixError(RuntimeError):
    """Fail-closed error for director-matrix evidence."""


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


def _load_json(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def _hash_payload(payload: Mapping[str, Any], prefix: str) -> str:
    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(prefix.encode() + b"\0" + raw).hexdigest()


def _metric(y: ArrayLike, pred: ArrayLike, *, top_k: int = 5) -> dict[str, float]:
    a = np.asarray(y, dtype=float).reshape(-1)
    b = np.asarray(pred, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise DirectorMatrixError("metric arrays must be aligned and non-empty")
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    rho = float(spearmanr(a, b).statistic) if a.size > 2 else float("nan")
    k = max(1, min(int(top_k), a.size))
    top_true = set(np.argsort(-a)[:k].tolist())
    top_pred = set(np.argsort(-b)[:k].tolist())
    regret = float(np.max(a) - a[int(np.argmax(b))])
    return {
        "mse": mse,
        "mae": mae,
        "spearman": rho,
        "topk_recall": float(len(top_true & top_pred) / k),
        "decision_regret": regret,
    }


def _paired_bootstrap_lcb(diff: ArrayLike, *, alpha: float = 0.05, samples: int = 8000, seed: int = DIRECTOR_SEED) -> dict[str, Any]:
    d = np.asarray(diff, dtype=float).reshape(-1)
    if d.size < 4 or not np.isfinite(d).all():
        raise DirectorMatrixError("paired bootstrap requires at least four finite paired values")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(samples), d.size))
    means = np.mean(d[idx], axis=1)
    return {
        "n_pairs": int(d.size),
        "mean_difference": float(np.mean(d)),
        "lcb": float(np.quantile(means, float(alpha))),
        "ucb": float(np.quantile(means, 1.0 - float(alpha))),
        "alpha": float(alpha),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def global_integrated_action(segmented_action: ArrayLike) -> NDArray[np.float64]:
    """Map (beta_1,gamma_1,...,beta_p,gamma_p) to (sum beta, sum gamma)."""
    x = np.asarray(segmented_action, dtype=np.float64)
    was_1d = x.ndim == 1
    if was_1d:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[1] % 2:
        raise DirectorMatrixError("segmented action must have an even positive number of columns")
    y = x.reshape(x.shape[0], -1, 2).sum(axis=1)
    return y[0] if was_1d else y



# ---------------------------------------------------------------------------
# Atomic execution adapter for the already-mandatory V05 raw-vs-trig ablation.
# This is an execution decomposition, not an eleventh logical D45 experiment.
# ---------------------------------------------------------------------------

def _raw_trig_feasible_design(protocol: MatchedControlProtocol, n: int) -> NDArray[np.float64]:
    rows: list[np.ndarray] = []
    seen: set[bytes] = set()
    design_round = 0
    while len(rows) < int(n):
        batch = rc.latin_hypercube(protocol.bounds, max(4 * int(n), 128), seed=protocol.seed + 3705 + design_round)
        design_round += 1
        for c in batch:
            try:
                rc.fourier_forward_schedule(c, order=protocol.order, grid_points=129, reject_nonmonotone=True)
            except Exception:
                continue
            cc = np.asarray(c, dtype=float)
            key = np.ascontiguousarray(cc, dtype=np.float64).tobytes(order="C")
            if key in seen:
                continue
            seen.add(key); rows.append(cc)
            if len(rows) == int(n):
                break
        if design_round > 100:
            raise DirectorMatrixError("could not construct the V05 shared feasible corpus")
    return np.vstack(rows)


def advance_raw_trig_corpus_v47_step(
    evaluator: Any, *, protocol: MatchedControlProtocol, output_path: str | Path, corpus_size: int = 74,
) -> dict[str, Any]:
    """Collect exactly one shared V05 response per invocation."""
    n = int(corpus_size)
    if n < protocol.cssf_initial_count + 2 or n > protocol.total_control_budget:
        raise DirectorMatrixError("invalid V05 shared-corpus size")
    controls = _raw_trig_feasible_design(protocol, n)
    path = Path(output_path)
    payload = _load_json(path, {"schema": SCHEMA + "-V05-CORPUS", "records": []})
    records = list(payload.get("records", []))
    existing = {int(r["index"]) for r in records}
    pending = [i for i in range(n) if i not in existing]
    if not pending:
        payload["complete"] = True; _atomic_json(path, payload)
        return {"complete": True, "records": len(records), "required": n, "path": str(path)}
    i = pending[0]; c = controls[i]
    response = evaluator(c, num_reads=int(protocol.reads_per_control))
    records.append({"index": i, "control": c.tolist(), "response": {k: _jsonable(v) for k, v in response.items() if k != "sampleset"}})
    records.sort(key=lambda r: int(r["index"]))
    payload = {
        "schema": SCHEMA + "-V05-CORPUS", "records": records, "complete": len(records) == n,
        "corpus_size": n, "protocol": asdict(protocol),
        "design_sha256": hashlib.sha256(np.ascontiguousarray(controls).tobytes(order="C")).hexdigest(),
        "atomic_rule": "one shared annealer control evaluation per invocation",
    }
    _atomic_json(path, payload)
    return {"complete": bool(payload["complete"]), "records": len(records), "required": n, "latest_index": i, "path": str(path)}


def analyze_raw_trig_corpus_v47(
    *, evaluator: Any, protocol: MatchedControlProtocol, project_root: str | Path, corpus_path: str | Path,
    output_path: str | Path, candidate_pool_size: int = 4096,
) -> dict[str, Any]:
    """Reproduce the V05 causal analysis from an atomically collected corpus."""
    payload = _load_json(corpus_path, {})
    if not payload.get("complete"):
        raise DirectorMatrixError("V05 shared corpus is incomplete")
    records = list(payload["records"]); n = len(records)
    controls = np.asarray([r["control"] for r in records], dtype=float)
    responses = [r["response"] for r in records]
    Y = np.asarray([[float(r[k]) for k in TARGET_NAMES] for r in responses], dtype=float)
    nt, nc = int(protocol.cssf_initial_train), int(protocol.cssf_initial_calibration)
    train_idx = np.arange(nt, dtype=int); cal_idx = np.arange(nt, nt + nc, dtype=int); test_idx = np.arange(nt + nc, n, dtype=int)
    if test_idx.size == 0:
        raise DirectorMatrixError("V05 requires held-out observations")
    pure = getattr(evaluator, "operator_action", None)
    if pure is None or not callable(pure):
        raise DirectorMatrixError("V05 requires query-free evaluator.operator_action")
    raw_theta = np.vstack([raw_control_coordinates(c, protocol.bounds) for c in controls])
    trig_theta = np.vstack([np.asarray(r.get("operator_action", pure(c)), dtype=float) for c, r in zip(controls, responses, strict=True)])
    common_fit = dict(
        calibration_targets=Y[cal_idx], target_names=TARGET_NAMES, project_root=str(project_root),
        support_mode="signed_axes", support_order=1, n_lambdas=100, lam_range=(-12.0, 4.0), nominal_coverage=0.90,
    )
    raw_model = fit_cssf_qa_response(raw_theta[train_idx], Y[train_idx], calibration_operator_phase=raw_theta[cal_idx], metadata={"experiment": "V05-shared-corpus", "representation": "raw/no-operator-phase"}, **common_fit)
    trig_model = fit_cssf_qa_response(trig_theta[train_idx], Y[train_idx], calibration_operator_phase=trig_theta[cal_idx], metadata={"experiment": "V05-shared-corpus", "representation": "operator-phase"}, **common_fit)
    pool = rc.latin_hypercube(protocol.bounds, int(candidate_pool_size), seed=protocol.seed + 37505)
    feasible, raw_pool, trig_pool = [], [], []
    for c in pool:
        try:
            rc.fourier_forward_schedule(c, order=protocol.order, grid_points=129, reject_nonmonotone=True)
            feasible.append(np.asarray(c, dtype=float)); raw_pool.append(raw_control_coordinates(c, protocol.bounds)); trig_pool.append(np.asarray(pure(c), dtype=float))
        except Exception:
            continue
    if not feasible:
        raise DirectorMatrixError("V05 candidate pool has no feasible schedule")
    feasible_arr = np.vstack(feasible); raw_pool_arr = np.vstack(raw_pool); trig_pool_arr = np.vstack(trig_pool)
    acq = dict(target="elite_probability", maximize=True, uncertainty_weight=1.0, leverage_weight=0.25, feasibility_target="feasibility_probability", minimum_feasibility=0.0)
    raw_idx, _, raw_diag = raw_model.acquisition(raw_pool_arr, **acq); trig_idx, _, trig_diag = trig_model.acquisition(trig_pool_arr, **acq)
    raw_pred = raw_model.predict(raw_theta[test_idx]); trig_pred = trig_model.predict(trig_theta[test_idx])
    candidate_pool_sha = hashlib.sha256(np.ascontiguousarray(feasible_arr).tobytes(order="C")).hexdigest()
    observation_ids = [f"shared-corpus-{i:04d}" for i in range(n)]
    shared_config = {
        "surrogate": {"family": "CSNN-T", "gcv_n_lambdas": 100, "gcv_lam_range": [-12.0, 4.0], "support_mode": "signed_axes", "support_order": 1},
        "target_transform": "none", "uncertainty": {"nominal_coverage": 0.90, "policy": "calibration-residual-scale+leverage"},
        "acquisition": {"target": "elite_probability", "uncertainty_weight": 1.0, "leverage_weight": 0.25, "minimum_feasibility": 0.0},
        "candidate_pool": {"size": int(feasible_arr.shape[0]), "sha256": candidate_pool_sha},
        "stopping_rule": {"kind": "fixed_shared_corpus", "controls": n}, "reads_per_control": int(protocol.reads_per_control),
    }
    common = {
        "observation_ids": observation_ids, "train_observation_ids": [observation_ids[i] for i in train_idx],
        "calibration_observation_ids": [observation_ids[i] for i in cal_idx], "heldout_observation_ids": [observation_ids[i] for i in test_idx],
        "budget": {"control_evaluations": n, "annealer_reads": n * int(protocol.reads_per_control)},
        "seeds": {"protocol": int(protocol.seed), "shared_design_base": int(protocol.seed + 3705), "candidate_pool": int(protocol.seed + 37505)},
        "candidate_pool_sha256": candidate_pool_sha, "target_names": list(TARGET_NAMES),
    }
    evidence = {
        "schema": "CSSF-QA-RAW-TRIG-CAUSAL-ABLATION-v38",
        "CSSF-raw/no-trig": {**common, "config": {**shared_config, "representation": {"name": "CSSF-raw/no-trig", "coordinate_transform": "normalized_raw_schedule_parameters"}}, "heldout_mse": float(np.mean((raw_pred - Y[test_idx]) ** 2)), "heldout_coverage_by_target": raw_model.empirical_coverage(raw_theta[test_idx], Y[test_idx]).tolist(), "selected_candidate": feasible_arr[raw_idx].tolist(), "acquisition_diagnostics": dict(raw_diag)},
        "CSSF-trig": {**common, "config": {**shared_config, "representation": {"name": "CSSF-trig", "coordinate_transform": "calibration_resolved_operator_action"}}, "heldout_mse": float(np.mean((trig_pred - Y[test_idx]) ** 2)), "heldout_coverage_by_target": trig_model.empirical_coverage(trig_theta[test_idx], Y[test_idx]).tolist(), "selected_candidate": feasible_arr[trig_idx].tolist(), "acquisition_diagnostics": dict(trig_diag)},
    }
    _atomic_json(output_path, evidence)
    return evidence

# ---------------------------------------------------------------------------
# Shared large protected corpus used by D45-01/02/03/06/07.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DirectorCorpusPlan:
    train: int = 1400
    selection: int = 128
    calibration: int = 128
    confirmation: int = 128
    ood: int = 128
    reads_per_control: int = 512
    seed: int = DIRECTOR_SEED

    @property
    def total(self) -> int:
        return self.train + self.selection + self.calibration + self.confirmation + self.ood

    def validate_dictionary_capacity(self, n_dimensions: int = 16) -> dict[str, int]:
        d = int(n_dimensions)
        # Exact closed-form counts avoid the frozen total_l1_support implementation's
        # Cartesian enumeration at d=16.  For ||k||_1<=2:
        # 1 zero + 2d first harmonics + 2d second harmonics + 4*C(d,2) pair terms.
        counts = {
            "signed_axes_1": 1 + 2 * d,
            "pairwise_1": 1 + 2 * d + 4 * math.comb(d, 2),
            "total_l1_2": 1 + 4 * d + 4 * math.comb(d, 2),
        }
        required = 2 * max(counts.values())
        if self.train < required:
            raise DirectorMatrixError(
                f"Director corpus train={self.train} is insufficient for the largest frozen dictionary; required >= {required}"
            )
        return counts


DEFAULT_DIRECTOR_PLAN = DirectorCorpusPlan()


def _interior_bounds(bounds: NDArray[np.float64]) -> NDArray[np.float64]:
    interior = np.asarray(bounds, dtype=float).copy()
    mid = np.mean(interior, axis=1)
    half = 0.34 * (interior[:, 1] - interior[:, 0])
    half[0] = 0.36 * (interior[0, 1] - interior[0, 0])
    interior[:, 0] = mid - half
    interior[:, 1] = mid + half
    return interior


def _feasible_lhs(bounds: NDArray[np.float64], n: int, *, order: int, seed: int) -> NDArray[np.float64]:
    rows: list[np.ndarray] = []
    seen: set[bytes] = set()
    batch = 0
    while len(rows) < int(n):
        candidates = rc.latin_hypercube(bounds, max(256, 2 * (int(n) - len(rows))), seed=int(seed) + batch)
        batch += 1
        for c in candidates:
            try:
                rc.fourier_forward_schedule(c, order=int(order), grid_points=129, reject_nonmonotone=True)
            except Exception:
                continue
            key = np.ascontiguousarray(c, dtype=np.float64).round(14).tobytes()
            if key in seen:
                continue
            seen.add(key)
            rows.append(np.asarray(c, dtype=float))
            if len(rows) == int(n):
                break
        if batch > 500:
            raise DirectorMatrixError("could not construct the complete physically admissible director corpus")
    return np.vstack(rows)


def director_validation_design(protocol: MatchedControlProtocol, plan: DirectorCorpusPlan = DEFAULT_DIRECTOR_PLAN) -> list[dict[str, Any]]:
    """Deterministic protected design sized to identify the full bounded dictionary family."""
    plan.validate_dictionary_capacity(2 * 8)
    full = np.asarray(protocol.bounds, dtype=float)
    interior = _interior_bounds(full)
    n_inside = plan.train + plan.selection + plan.calibration + plan.confirmation
    inside = _feasible_lhs(interior, n_inside, order=protocol.order, seed=plan.seed + 5000)
    # OOD remains inside the same physically admissible global schedule domain but outside the interior box.
    ood: list[np.ndarray] = []
    batch = 0
    while len(ood) < plan.ood:
        cand = _feasible_lhs(full, max(64, 2 * (plan.ood - len(ood))), order=protocol.order, seed=plan.seed + 7000 + batch)
        batch += 1
        for c in cand:
            if np.any((c < interior[:, 0]) | (c > interior[:, 1])):
                ood.append(c)
                if len(ood) == plan.ood:
                    break
        if batch > 500:
            raise DirectorMatrixError("could not construct protected OOD director controls")
    labels = (
        ["train"] * plan.train
        + ["selection"] * plan.selection
        + ["calibration"] * plan.calibration
        + ["confirmation"] * plan.confirmation
        + ["ood"] * plan.ood
    )
    controls = [*inside, *ood]
    return [
        {"design_index": i, "control_id": f"director:{i:05d}", "partition": labels[i], "control": np.asarray(c).tolist()}
        for i, c in enumerate(controls)
    ]


def advance_director_corpus_step(
    evaluator: Any,
    *,
    protocol: MatchedControlProtocol,
    output_path: str | Path,
    plan: DirectorCorpusPlan = DEFAULT_DIRECTOR_PLAN,
) -> dict[str, Any]:
    """Collect exactly one annealer response from the protected director corpus."""
    design = director_validation_design(protocol, plan)
    path = Path(output_path)
    payload = _load_json(path, {"schema": SCHEMA + "-CORPUS", "records": []})
    records = list(payload.get("records", []))
    existing = {int(r["design_index"]) for r in records}
    pending = [row for row in design if int(row["design_index"]) not in existing]
    if not pending:
        return {"complete": True, "records": len(records), "required": len(design), "path": str(path)}
    row = pending[0]
    control = np.asarray(row["control"], dtype=float)
    response = evaluator(control, num_reads=int(plan.reads_per_control))
    record = {
        **row,
        "response": {k: _jsonable(v) for k, v in response.items() if k != "sampleset"},
    }
    records.append(record)
    records.sort(key=lambda r: int(r["design_index"]))
    payload = {
        "schema": SCHEMA + "-CORPUS",
        "protocol": asdict(protocol),
        "plan": asdict(plan),
        "design_sha256": _hash_payload({"design": design}, "CSSF-director-design-v47"),
        "records": records,
        "complete": len(records) == len(design),
        "atomic_rule": "one annealer control evaluation per invocation; no scientific reduction",
    }
    _atomic_json(path, payload)
    return {
        "complete": bool(payload["complete"]),
        "records": len(records),
        "required": len(design),
        "latest_control_id": row["control_id"],
        "partition": row["partition"],
        "path": str(path),
    }


def _load_complete_director_corpus(path: str | Path) -> tuple[list[dict[str, Any]], NDArray[np.str_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    payload = _load_json(path, {})
    if not payload.get("complete"):
        raise DirectorMatrixError("director corpus is incomplete; continue atomic collection before analysis")
    records = list(payload.get("records", []))
    part = np.asarray([str(r["partition"]) for r in records])
    raw = np.asarray([r["control"] for r in records], dtype=float)
    phase = np.asarray([r["response"]["operator_action"] for r in records], dtype=float)
    Y = np.asarray([[float(r["response"][k]) for k in TARGET_NAMES] for r in records], dtype=float)
    return records, part, raw, phase, Y


# ---------------------------------------------------------------------------
# D45-01 — segmented action vs global integrated resources.
# ---------------------------------------------------------------------------

def run_d45_01_segmented_vs_global(
    *, corpus_path: str | Path, project_root: str | Path, output_path: str | Path,
    primary_target: str = "elite_probability", bootstrap_samples: int = 8000,
) -> dict[str, Any]:
    """Causal representation test against the 2026 global-resource prior-art class.

    Only the coordinate map is changed.  Training observations, targets,
    canonical CSNN-T/GCV fitting, support order, calibration partition and
    protected evaluation controls are identical.
    """
    records, part, raw, phase, Y = _load_complete_director_corpus(corpus_path)
    del records, raw
    ti = TARGET_NAMES.index(primary_target)
    train, cal, confirm, ood = part == "train", part == "calibration", part == "confirmation", part == "ood"
    global_phase = global_integrated_action(phase)
    common = dict(
        calibration_targets=Y[cal], target_names=TARGET_NAMES, project_root=project_root,
        support_mode="signed_axes", support_order=1, nominal_coverage=0.90,
    )
    segmented = fit_cssf_qa_response(
        phase[train], Y[train], calibration_operator_phase=phase[cal],
        metadata={"experiment": "D45-01", "representation": "segmented"}, **common,
    )
    global_model = fit_cssf_qa_response(
        global_phase[train], Y[train], calibration_operator_phase=global_phase[cal],
        metadata={"experiment": "D45-01", "representation": "global_integrated_resource"}, **common,
    )
    rows: dict[str, Any] = {}
    for label, mask, seed_off in (("confirmation", confirm, 1), ("ood", ood, 2)):
        y = Y[mask, ti]
        ps = segmented.predict(phase[mask])[:, ti]
        pg = global_model.predict(global_phase[mask])[:, ti]
        diff = (pg - y) ** 2 - (ps - y) ** 2  # positive => segmented lower error
        rows[label] = {
            "segmented": _metric(y, ps),
            "global": _metric(y, pg),
            "paired_squared_error_improvement": _paired_bootstrap_lcb(
                diff, samples=bootstrap_samples, seed=DIRECTOR_SEED + seed_off
            ),
        }
    claim_passed = bool(
        rows["confirmation"]["paired_squared_error_improvement"]["lcb"] > 0.0
        and rows["ood"]["paired_squared_error_improvement"]["lcb"] > 0.0
    )
    payload = {
        "schema": SCHEMA + "-D45-01",
        "execution_complete": True,
        "primary_target": primary_target,
        "same_observations": True,
        "same_canonical_csnn_t_gcv": True,
        "same_harmonic_support_family": True,
        "global_definition": "(sum_k beta_k, sum_k gamma_k)",
        "segmented_definition": "(beta_1,gamma_1,...,beta_p,gamma_p)",
        "partitions": rows,
        "acceptance": "separate one-sided protected LCBs > 0 on confirmation and OOD; no pooling may rescue a failed protected split",
        "claim_passed": claim_passed,
        "passed": claim_passed,
    }
    _atomic_json(output_path, payload)
    return payload

def _safe_total_l1_support(n_dimensions: int, max_l1_order: int) -> FrequencySupport:
    """Generate exactly {k in Z^d: ||k||_1<=q} with pruning, not Cartesian  (2q+1)^d enumeration."""
    d = int(n_dimensions); q = int(max_l1_order)
    if d < 1 or q < 1:
        raise DirectorMatrixError("invalid safe total-L1 support dimensions/order")
    rows: list[tuple[int, ...]] = []
    cur = [0] * d
    def rec(j: int, remaining: int) -> None:
        if j == d:
            rows.append(tuple(cur)); return
        for v in range(-remaining, remaining + 1):
            cur[j] = v
            rec(j + 1, remaining - abs(v))
        cur[j] = 0
    rec(0, q)
    unique = sorted(set(rows), key=lambda r: (sum(abs(v) for v in r), r))
    return FrequencySupport(unique, kind=SupportKind.TOTAL_L1, include_zero=True, require_conjugate_symmetry=True)


def _dictionary_support(n_dimensions: int, mode: str, order: int) -> FrequencySupport:
    if mode == "total_l1" and int(order) >= 2:
        return _safe_total_l1_support(n_dimensions, int(order))
    return build_support(n_dimensions, mode=mode, order=order)


def _fit_custom_support(
    theta_train: NDArray[np.float64], y_train: NDArray[np.float64], *, support: FrequencySupport,
    project_root: str | Path, target_names: Sequence[str], metadata: Mapping[str, Any],
):
    phi = toric_feature_matrix(theta_train, support, wrap_coordinates=True)
    if phi.shape[1] > phi.shape[0] // 2:
        raise DirectorMatrixError(
            f"custom dictionary underidentified: features={phi.shape[1]} > floor(train/2)={phi.shape[0]//2}"
        )
    ds = CSSFDataset(phi, y_train, metadata={"role": "qa_response_dictionary_gate", **dict(metadata)})
    model = fit_csnn_t_surrogate(
        ds, case="cssf_qa_dictionary_gate", level=SurrogateLevel.DIGITIZED_QA,
        target_names=tuple(target_names), n_lambdas=100, lam_range=(-12.0, 4.0),
        metadata={"surrogated_system": "quantum_annealing_response", **dict(metadata)},
        verify_integrity=True, project_root=project_root,
    )
    return model, phi


# ---------------------------------------------------------------------------
# D45-02 — protected dictionary selection / frequency semantics.
# ---------------------------------------------------------------------------

def _support_candidates(n_dimensions: int) -> list[tuple[str, str, int]]:
    return [
        ("signed_axes_1", "signed_axes", 1),
        ("pairwise_1", "pairwise", 1),
        ("total_l1_2", "total_l1", 2),
    ]


def _condition_report(phi: NDArray[np.complex128], lam: float) -> dict[str, Any]:
    sv = np.linalg.svd(phi, compute_uv=False)
    tol = max(phi.shape) * np.finfo(float).eps * (float(sv[0]) if sv.size else 0.0)
    rank = int(np.sum(sv > tol))
    cond = float(np.inf if sv.size == 0 or sv[-1] <= 0 else sv[0] / sv[-1])
    eff = float(np.sum((sv ** 2) / (sv ** 2 + float(lam))))
    return {
        "rows": int(phi.shape[0]), "features": int(phi.shape[1]), "rank": rank,
        "effective_rank": eff, "condition_number": cond,
        "singular_value_max": float(sv[0]), "singular_value_min": float(sv[-1]),
    }


def run_d45_02_dictionary_gate(
    *, corpus_path: str | Path, project_root: str | Path, output_path: str | Path,
    primary_target: str = "elite_probability", bootstrap_resamples: int = 200,
) -> dict[str, Any]:
    """Protected dictionary-order selection with conditioning and stability audit.

    Dictionary choice is made on D_selection only.  D_confirmation and D_ood are
    opened only after the label is frozen.  Frequency semantics remain empirical
    integer harmonics in operator-action coordinates; this experiment does not
    relabel them as exact generator-gap frequencies.
    """
    _, part, _, phase, Y = _load_complete_director_corpus(corpus_path)
    train, selection, confirm, ood = (part == x for x in ("train", "selection", "confirmation", "ood"))
    ti = TARGET_NAMES.index(primary_target)
    candidates: list[dict[str, Any]] = []
    for label, mode, order in _support_candidates(phase.shape[1]):
        support = _dictionary_support(phase.shape[1], mode, order)
        required_train = 2 * int(support.n_terms)
        if int(np.sum(train)) < required_train:
            candidates.append({
                "label": label, "mode": mode, "order": order, "features": support.n_terms,
                "required_train": required_train, "status": "UNDERIDENTIFIED",
            })
            continue
        model, phi = _fit_custom_support(
            phase[train], Y[train], support=support, project_root=project_root,
            target_names=TARGET_NAMES, metadata={"experiment": "D45-02", "dictionary": label},
        )
        sel_pred = model.predict(toric_feature_matrix(phase[selection], support, wrap_coordinates=True))[:, ti]
        sel_y = Y[selection, ti]
        sel_metric = _metric(sel_y, sel_pred)
        cr = _condition_report(phi, model.model.lam_opt)
        nsel = int(np.sum(selection))
        # Predeclared generalized-complexity criterion.  The effective-rank term
        # penalizes poorly supported high-order bases while preserving GCV as the
        # canonical regularization selector inside each candidate model.
        penalized = math.log(max(sel_metric["mse"], 1e-15)) + math.log(max(nsel, 2)) * cr["effective_rank"] / nsel
        candidates.append({
            "label": label, "mode": mode, "order": order, "features": support.n_terms,
            "required_train": required_train, "status": "ADMISSIBLE", "lambda_gcv": float(model.model.lam_opt),
            "conditioning": cr, "selection_metric": sel_metric, "selection_complexity_score": float(penalized),
            "_model": model, "_support": support,
        })
    admissible = [x for x in candidates if x.get("status") == "ADMISSIBLE"]
    clean_candidates = lambda: [{k: v for k, v in x.items() if k not in {"_model", "_support"}} for x in candidates]
    if len(admissible) < 2:
        payload = {
            "schema": SCHEMA + "-D45-02", "execution_complete": True,
            "status": "INSUFFICIENT_ADMISSIBLE_DICTIONARIES", "candidates": clean_candidates(),
            "dictionary_gate_passed": False, "claim_passed": False, "passed": False,
        }
        _atomic_json(output_path, payload)
        return payload

    chosen = min(admissible, key=lambda x: float(x["selection_complexity_score"]))
    chosen_model, chosen_support = chosen["_model"], chosen["_support"]

    final: dict[str, Any] = {}
    for label, mask in (("confirmation", confirm), ("ood", ood)):
        y = Y[mask, ti]
        pred = chosen_model.predict(toric_feature_matrix(phase[mask], chosen_support, wrap_coordinates=True))[:, ti]
        final[label] = _metric(y, pred)

    # Bootstrap stability of *dictionary identity* using selection partition only.
    selection_indices = np.flatnonzero(selection)
    rng = np.random.default_rng(DIRECTOR_SEED + 202)
    selected_labels: list[str] = []
    for _ in range(int(bootstrap_resamples)):
        idx = selection_indices[rng.integers(0, selection_indices.size, selection_indices.size)]
        scores: dict[str, float] = {}
        for row in admissible:
            pred = row["_model"].predict(toric_feature_matrix(phase[idx], row["_support"], wrap_coordinates=True))[:, ti]
            m = _metric(Y[idx, ti], pred)
            n = len(idx); er = float(row["conditioning"]["effective_rank"])
            scores[row["label"]] = math.log(max(m["mse"], 1e-15)) + math.log(max(n, 2)) * er / n
        selected_labels.append(min(scores, key=scores.get))
    selection_frequency = {row["label"]: float(selected_labels.count(row["label"]) / len(selected_labels)) for row in admissible}

    # Coefficient/prediction stability is audited without reducing the frozen
    # feature set.  80% train subsamples remain identifiable because the v47
    # director corpus is deliberately sized above 2*M for the largest support.
    train_idx = np.flatnonzero(train)
    coef_stability = []
    for rep in range(3):
        rr = np.random.default_rng(DIRECTOR_SEED + 220 + rep)
        sub = np.sort(rr.choice(train_idx, size=int(math.floor(0.80 * train_idx.size)), replace=False))
        if sub.size < 2 * int(chosen_support.n_terms):
            raise DirectorMatrixError("v47 corpus no longer supports the frozen coefficient-stability subsample")
        sub_model, _ = _fit_custom_support(
            phase[sub], Y[sub], support=chosen_support, project_root=project_root,
            target_names=TARGET_NAMES, metadata={"experiment": "D45-02", "role": "coefficient_stability", "replicate": rep},
        )
        h0 = np.asarray(chosen_model.model.H, dtype=np.complex128)
        h1 = np.asarray(sub_model.model.H, dtype=np.complex128)
        rel_h = float(np.linalg.norm(h1 - h0) / max(np.linalg.norm(h0), 1e-15))
        p0 = chosen_model.predict(toric_feature_matrix(phase[selection], chosen_support, wrap_coordinates=True))[:, ti]
        p1 = sub_model.predict(toric_feature_matrix(phase[selection], chosen_support, wrap_coordinates=True))[:, ti]
        rho = float(spearmanr(p0, p1).statistic)
        coef_stability.append({"replicate": rep, "train_rows": int(sub.size), "relative_H_frobenius_drift": rel_h, "selection_prediction_spearman": rho})

    stable_freq = bool(selection_frequency.get(chosen["label"], 0.0) >= 0.70)
    stable_predictions = bool(all(np.isfinite(r["selection_prediction_spearman"]) and r["selection_prediction_spearman"] >= 0.90 for r in coef_stability))
    finite_protected = bool(np.isfinite(final["confirmation"]["mse"]) and np.isfinite(final["ood"]["mse"]))
    gate = bool(stable_freq and stable_predictions and finite_protected)
    payload = {
        "schema": SCHEMA + "-D45-02", "execution_complete": True, "status": "COMPLETE",
        "primary_target": primary_target, "selection_partition_only_for_dictionary_choice": True,
        "chosen_dictionary": chosen["label"], "candidates": clean_candidates(),
        "selection_bootstrap_resamples": int(bootstrap_resamples), "dictionary_selection_frequency": selection_frequency,
        "dictionary_frequency_stability_threshold": 0.70,
        "coefficient_prediction_stability": coef_stability, "prediction_stability_spearman_threshold": 0.90,
        "protected_final_metrics": final,
        "claim_semantics": "empirical operator-action-coordinate-informed integer harmonic support; not exact generator-gap spectrum",
        "dictionary_gate_passed": gate, "claim_passed": gate, "passed": gate,
    }
    _atomic_json(output_path, payload)
    return payload

# ---------------------------------------------------------------------------
# D45-03 — multi-output vs scalar CSSF.
# ---------------------------------------------------------------------------

def run_d45_03_multioutput_vs_scalar(
    *, corpus_path: str | Path, project_root: str | Path, output_path: str | Path,
    primary_target: str = "elite_probability", minimum_feasibility: float = 0.0,
) -> dict[str, Any]:
    """Same-coordinate, same-core ablation of multi-output response composition."""
    _, part, _, phase, Y = _load_complete_director_corpus(corpus_path)
    train, cal, selection, confirm, ood = (part == x for x in ("train", "calibration", "selection", "confirmation", "ood"))
    ti = TARGET_NAMES.index(primary_target)
    multi = fit_cssf_qa_response(
        phase[train], Y[train], calibration_operator_phase=phase[cal], calibration_targets=Y[cal],
        target_names=TARGET_NAMES, project_root=project_root, support_mode="signed_axes", support_order=1,
        metadata={"experiment": "D45-03", "response": "multi_output"},
    )
    scalar = fit_cssf_qa_response(
        phase[train], Y[train, ti], calibration_operator_phase=phase[cal], calibration_targets=Y[cal, ti],
        target_names=(primary_target,), project_root=project_root, support_mode="signed_axes", support_order=1,
        metadata={"experiment": "D45-03", "response": "scalar"},
    )
    candidate_phase = phase[selection]
    im, _, dm = multi.acquisition(
        candidate_phase, target=primary_target, maximize=True, uncertainty_weight=1.0, leverage_weight=0.25,
        feasibility_target="feasibility_probability", minimum_feasibility=float(minimum_feasibility),
    )
    is_, _, ds = scalar.acquisition(
        candidate_phase, target=primary_target, maximize=True, uncertainty_weight=1.0, leverage_weight=0.25,
        feasibility_target=None,
    )
    selection_idx = np.flatnonzero(selection)
    multi_global, scalar_global = int(selection_idx[im]), int(selection_idx[is_])
    observed_multi = {k: float(Y[multi_global, j]) for j, k in enumerate(TARGET_NAMES)}
    observed_scalar = {k: float(Y[scalar_global, j]) for j, k in enumerate(TARGET_NAMES)}
    final: dict[str, Any] = {}
    for label, mask, seed_off in (("confirmation", confirm, 0), ("ood", ood, 1)):
        y = Y[mask, ti]
        pm = multi.predict(phase[mask])[:, ti]
        ps = scalar.predict(phase[mask])[:, 0]
        diff = (ps - y) ** 2 - (pm - y) ** 2
        final[label] = {
            "multi": _metric(y, pm), "scalar": _metric(y, ps),
            "paired_squared_error_improvement": _paired_bootstrap_lcb(diff, seed=DIRECTOR_SEED + 30 + seed_off),
        }
    decision_delta = float(observed_multi[primary_target] - observed_scalar[primary_target])
    feasibility_delta = float(observed_multi["feasibility_probability"] - observed_scalar["feasibility_probability"])
    predictive_gate = bool(
        final["confirmation"]["paired_squared_error_improvement"]["lcb"] > 0.0
        and final["ood"]["paired_squared_error_improvement"]["lcb"] > 0.0
    )
    decision_gate = bool(decision_delta > 0.0 and observed_multi["feasibility_probability"] >= float(minimum_feasibility))
    claim_passed = bool(predictive_gate and decision_gate)
    payload = {
        "schema": SCHEMA + "-D45-03", "execution_complete": True, "primary_target": primary_target,
        "same_coordinates": True, "same_core": True, "same_train_calibration_selection_partitions": True,
        "multi_selected_control_index": multi_global, "scalar_selected_control_index": scalar_global,
        "multi_acquisition": dict(dm), "scalar_acquisition": dict(ds),
        "observed_multi_selected_response": observed_multi, "observed_scalar_selected_response": observed_scalar,
        "selected_target_delta": decision_delta, "selected_feasibility_delta": feasibility_delta,
        "protected_predictive_metrics": final, "predictive_gate_passed": predictive_gate,
        "decision_gate_passed": decision_gate,
        "interpretation_boundary": "selection-partition decision ablation plus untouched predictive confirmation/OOD; final physical superiority is established only by D45-09/D45-10 and the locked endpoint",
        "claim_passed": claim_passed, "passed": claim_passed,
    }
    _atomic_json(output_path, payload)
    return payload

# ---------------------------------------------------------------------------
# D45-04 — D-Wave-native default / pausing incumbent.
# ---------------------------------------------------------------------------

def dwave_incumbent_schedules(calibration: Any, protocol: MatchedControlProtocol) -> list[dict[str, Any]]:
    """Predeclared native incumbent set: default linear plus crossover-local pauses.

    The pause center is derived from the frozen per-system A/B calibration only;
    no QPU/SQA response is used to define the candidate grid.  Every ramp has
    the standard 20-us slope and every total duration stays inside the common
    matched control domain.
    """
    T0 = float(np.clip(20.0, *protocol.annealing_time_range_us))
    s_cross = float(calibration.s[int(np.argmin(np.abs(calibration.A_GHz - calibration.B_GHz)))])
    pause_locations = sorted({float(np.clip(s_cross + d, 0.15, 0.85)) for d in (-0.10, 0.0, 0.10)})
    max_pause = max(0.0, float(protocol.annealing_time_range_us[1]) - T0)
    pause_durations = sorted({float(np.clip(v, 0.0, max_pause)) for v in (2.0, 10.0, 30.0) if max_pause > 0.0})
    rows = [{"schedule_id": "dwave_default_linear", "t_us": [0.0, T0], "s": [0.0, 1.0], "role": "default_linear"}]
    for loc in pause_locations:
        for pause_us in pause_durations:
            t1 = T0 * loc
            t2 = t1 + pause_us
            tf = T0 + pause_us
            rows.append({
                "schedule_id": f"dwave_pause_s{loc:.4f}_{pause_us:g}us",
                "t_us": [0.0, t1, t2, tf], "s": [0.0, loc, loc, 1.0],
                "role": "calibration_crossover_local_pause", "pause_s": loc, "pause_us": pause_us,
                "definition_source": "predeclared frozen A/B crossover neighborhood; no response-data tuning of grid",
            })
    return rows

def advance_d45_04_incumbent_step(
    evaluator: Any, *, protocol: MatchedControlProtocol, output_path: str | Path,
    confirmation_replicates: int = 16,
) -> dict[str, Any]:
    """Advance one incumbent-candidate or one frozen-winner confirmation batch."""
    curve = evaluator.backend.calibration
    schedules = dwave_incumbent_schedules(curve, protocol)
    path = Path(output_path)
    payload = _load_json(path, {"schema": SCHEMA + "-D45-04", "candidate_records": [], "confirmation_records": []})
    candidate_records = list(payload.get("candidate_records", []))
    done = {r["schedule_id"] for r in candidate_records}
    pending = [r for r in schedules if r["schedule_id"] not in done]
    if pending:
        row = pending[0]
        response = evaluator.backend.evaluate_schedule(
            np.asarray(row["t_us"], float), np.asarray(row["s"], float), num_reads=int(protocol.reads_per_control),
            elite_threshold=float(evaluator.elite_threshold), feasibility=evaluator.feasibility,
            success_energy=evaluator.success_energy, label=row["schedule_id"],
        )
        candidate_records.append({**row, "response": {k: _jsonable(v) for k, v in response.items() if k != "sampleset"}})
        payload.update({
            "candidate_records": candidate_records, "confirmation_records": list(payload.get("confirmation_records", [])),
            "complete": False, "execution_complete": False, "stage": "candidate_selection",
            "selection_rule": "highest elite_probability among predeclared native default/crossover-pause schedules",
        })
        _atomic_json(path, payload)
        return {"complete": False, "stage": "candidate_selection", "records": len(candidate_records), "required": len(schedules), "latest": row["schedule_id"]}

    if "selected_best_practice" not in payload:
        values = np.asarray([float(r["response"]["elite_probability"]) for r in candidate_records])
        best = candidate_records[int(np.argmax(values))]
        payload["selected_best_practice"] = best["schedule_id"]
        payload["selected_schedule"] = {"t_us": best["t_us"], "s": best["s"]}
        payload["winner_frozen_before_confirmation"] = True
        _atomic_json(path, payload)

    confirmations = list(payload.get("confirmation_records", []))
    if len(confirmations) < int(confirmation_replicates):
        schedule = payload["selected_schedule"]
        response = evaluator.backend.evaluate_schedule(
            np.asarray(schedule["t_us"], float), np.asarray(schedule["s"], float), num_reads=int(protocol.reads_per_control),
            elite_threshold=float(evaluator.elite_threshold), feasibility=evaluator.feasibility,
            success_energy=evaluator.success_energy,
            label=f"{payload['selected_best_practice']}:independent_confirmation:{len(confirmations):03d}",
        )
        confirmations.append({"replicate": len(confirmations), "response": {k: _jsonable(v) for k, v in response.items() if k != "sampleset"}})
        payload.update({"confirmation_records": confirmations, "stage": "independent_confirmation", "complete": False, "execution_complete": False})
        _atomic_json(path, payload)
        return {"complete": False, "stage": "independent_confirmation", "confirmation_replicates": len(confirmations), "required": int(confirmation_replicates)}

    vals = np.asarray([float(r["response"]["elite_probability"]) for r in confirmations], float)
    boot = _paired_bootstrap_lcb(vals, alpha=0.05, samples=8000, seed=DIRECTOR_SEED + 404)
    # For a single method this bootstrap is a mean CI; field names are retained
    # from the shared bootstrap primitive but no difference-to-zero superiority
    # inference is made here.
    payload.update({
        "stage": "complete", "complete": True, "execution_complete": True,
        "confirmation_summary": {"mean_elite_probability": float(np.mean(vals)), "lcb_mean": float(boot["lcb"]), "ucb_mean": float(boot["ucb"]), "replicates": int(vals.size)},
        "incumbent_reference_valid": True, "claim_passed": True, "passed": True,
        "not_a_novelty_claim": True,
    })
    _atomic_json(path, payload)
    return payload

# ---------------------------------------------------------------------------
# D45-05 — 2026 query-efficiency frontier fidelity gate.
# ---------------------------------------------------------------------------

def run_d45_05_query_frontier_gate(
    *, original_qzero_gate: Mapping[str, Any], optimized_qzero_asset: str | Path | None, output_path: str | Path,
) -> dict[str, Any]:
    """Close or explicitly lock the broad 2026 query-efficiency claim.

    Existence of a file is not sufficient.  A claim-grade optimized-QZero asset
    must bind an implementation hash, source/reconstruction provenance, paper-
    domain reproduction, matched-domain execution and complete hidden-query
    accounting.  Missing publication details may be reconstructed only when the
    choices are explicitly declared and the paper-level numerical bar is first
    reproduced within its stated tolerance.
    """
    asset = None if optimized_qzero_asset is None else Path(optimized_qzero_asset)
    manifest: Mapping[str, Any] = {}
    if asset is not None and asset.is_file():
        try:
            manifest = json.loads(asset.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    required_manifest = {
        "schema", "implementation_source_sha256", "source_provenance",
        "reconstruction_assumptions", "paper_domain_reproduction",
        "matched_domain_trace", "hidden_query_accounting",
    }
    manifest_fields = bool(manifest and required_manifest.issubset(manifest))
    source_hash_ok = bool(manifest_fields and isinstance(manifest.get("implementation_source_sha256"), str) and len(manifest["implementation_source_sha256"]) == 64)
    repro = dict(manifest.get("paper_domain_reproduction", {}) or {})
    reproduction_ok = bool(repro.get("passed", False) and np.isfinite(float(repro.get("observed_query_reduction_percent", float("nan")))))
    matched = dict(manifest.get("matched_domain_trace", {}) or {})
    matched_ok = bool(matched.get("complete", False) and int(matched.get("counted_hidden_queries", 0)) >= 0)
    hidden = dict(manifest.get("hidden_query_accounting", {}) or {})
    hidden_ok = bool(hidden.get("complete", False) and hidden.get("pretraining_and_target_queries_charged", False))
    asset_ready = bool(manifest_fields and source_hash_ok and reproduction_ok and matched_ok and hidden_ok)
    original_ready = bool(original_qzero_gate.get("pass", False))
    payload = {
        "schema": SCHEMA + "-D45-05", "execution_complete": True,
        "original_qzero_full_ready": original_ready,
        "optimized_qzero_2026_asset": None if asset is None else str(asset),
        "optimized_qzero_2026_manifest_checks": {
            "required_fields": manifest_fields, "source_hash": source_hash_ok,
            "paper_domain_reproduction": reproduction_ok, "matched_domain_trace": matched_ok,
            "hidden_query_accounting": hidden_ok,
        },
        "required_disclosed_mechanisms": [
            "three-stage selection/expansion+evaluation/backprop MCTS",
            "U=W/N + C*p*sqrt(N_t)/(1+N)", "exponentially decaying exploration coefficient",
            "partially shared policy/value MLP", "ELU hidden activations", "batch normalization",
            "skip connections", "dropout=0.20", "learning_rate_initial=0.008",
            "learning_rate_decay=0.96 per 1000 steps", "policy+value+L2 loss with entropy exploration term",
            "complete pretraining and hidden target-query accounting",
        ],
        "status": "QUERY_FRONTIER_EXECUTABLE" if (original_ready and asset_ready) else "BROAD_QUERY_EFFICIENCY_CLAIM_LOCKED",
        "broad_query_efficiency_claim_ready": bool(original_ready and asset_ready),
        "narrowed_matched_competitor_claim_allowed": True,
        "claim_passed": bool(original_ready and asset_ready), "passed": bool(original_ready and asset_ready),
    }
    _atomic_json(output_path, payload)
    return payload

# ---------------------------------------------------------------------------
# D45-06 — control leverage / abstention.
# ---------------------------------------------------------------------------

def run_d45_06_control_leverage(
    *, corpus_path: str | Path, output_path: str | Path, target: str = "elite_probability",
    minimum_practical_delta: float = 0.02, bootstrap_samples: int = 12000,
) -> dict[str, Any]:
    """Estimate exploitable response variation and enforce the abstention state."""
    _, part, _, _, Y = _load_complete_director_corpus(corpus_path)
    ti = TARGET_NAMES.index(target)
    mask = (part == "confirmation") | (part == "ood")
    p = np.clip(Y[mask, ti], 0.0, 1.0)
    R = float(DEFAULT_DIRECTOR_PLAN.reads_per_control)
    observed_var = float(np.var(p, ddof=1))
    binomial_noise = float(np.mean(p * (1.0 - p) / R))
    between = max(0.0, observed_var - binomial_noise)
    reliability = between / max(observed_var, 1e-15)
    rng = np.random.default_rng(DIRECTOR_SEED + 60)
    ranges = np.empty(int(bootstrap_samples), dtype=float)
    for b in range(int(bootstrap_samples)):
        sample = p[rng.integers(0, p.size, p.size)]
        ranges[b] = float(np.max(sample) - np.min(sample))
    range_lcb = float(np.quantile(ranges, 0.05))
    state = "CONTROL_LEVERAGE_ESTABLISHED" if (between > 0.0 and range_lcb > float(minimum_practical_delta)) else "CONTROL_LEVERAGE_NOT_ESTABLISHED"
    payload = {
        "schema": SCHEMA + "-D45-06", "execution_complete": True, "target": target, "n_controls": int(p.size),
        "observed_between_control_variance": observed_var, "mean_binomial_measurement_variance": binomial_noise,
        "deattenuated_control_variance": between, "control_reliability_ratio": reliability,
        "observed_range": float(np.max(p) - np.min(p)), "range_bootstrap_lcb": range_lcb,
        "minimum_practical_delta": float(minimum_practical_delta), "state": state,
        "abstention_required": state != "CONTROL_LEVERAGE_ESTABLISHED",
        "decision_valid": True,
        "positive_leverage_claim_passed": state == "CONTROL_LEVERAGE_ESTABLISHED",
        "claim_passed": state == "CONTROL_LEVERAGE_ESTABLISHED", "passed": state == "CONTROL_LEVERAGE_ESTABLISHED",
        "simulator_boundary": "live QPU campaign additionally requires programming-cycle replication; simulator variation cannot replace hardware repeatability evidence",
    }
    _atomic_json(output_path, payload)
    return payload

# ---------------------------------------------------------------------------
# D45-07 — filter-aware sensitivity of operator coordinates.
# ---------------------------------------------------------------------------

def approximate_ccjj_filtered_schedule(t_us: ArrayLike, s: ArrayLike, *, bandwidth_mhz: float = CCJJ_BANDWIDTH_MHZ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t = np.asarray(t_us, dtype=float).reshape(-1); q = np.asarray(s, dtype=float).reshape(-1)
    if t.size < 2 or t.size != q.size or np.any(np.diff(t) <= 0):
        raise DirectorMatrixError("filter approximation requires aligned increasing PWL schedule")
    # Reproduce the D-Wave-documented second-order Bessel approximation form,
    # applied here to normalized anneal fraction as a sensitivity model only.
    sampling_rate = max(100, int(float(bandwidth_mhz) * 100))
    n = max(256, int(np.ceil(sampling_rate * (t[-1] - t[0]))))
    td = np.linspace(t[0], t[-1], n)
    sig = np.interp(td, t, q)
    b, a = signal.bessel(2, 2 / 100, btype="lowpass", analog=False, output="ba", norm="mag")
    filt = np.clip(signal.lfilter(b, a, sig), 0.0, 1.0)
    return td.astype(float), filt.astype(float)


def _operator_action_unconstrained(t_us: ArrayLike, s: ArrayLike, calibration: Any, *, n_segments: int = 8) -> NDArray[np.float64]:
    t = np.asarray(t_us, dtype=float).reshape(-1); q = np.clip(np.asarray(s, dtype=float).reshape(-1), 0.0, 1.0)
    if t.size < 2 or t.size != q.size or np.any(np.diff(t) <= 0):
        raise DirectorMatrixError("invalid sensitivity waveform")
    A, B = calibration.interpolate(q)
    edges = np.linspace(t[0], t[-1], int(n_segments) + 1)
    out = []
    for j in range(int(n_segments)):
        lo, hi = edges[j], edges[j + 1]
        mask = (t >= lo) & (t <= hi)
        tt = t[mask]
        if tt.size < 2:
            td = np.linspace(lo, hi, 32); sd = np.interp(td, t, q); aa, bb = calibration.interpolate(sd); tt = td
        else:
            aa, bb = A[mask], B[mask]
        out.extend((TWO_PI_GHZ_US * float(np.trapezoid(aa, tt)), TWO_PI_GHZ_US * float(np.trapezoid(bb, tt))))
    return np.asarray(out, dtype=float)


def run_d45_07_filter_sensitivity(
    *, corpus_path: str | Path, project_root: str | Path, calibration_family: str, protocol: MatchedControlProtocol,
    output_path: str | Path, sample_controls: int = 128,
) -> dict[str, Any]:
    """Sensitivity analysis for finite-bandwidth distortion of requested schedules.

    The official 6.5-MHz cutoff is used.  The second-order Bessel transfer form
    is adapted from D-Wave's documented filtered-waveform approximation example
    and is explicitly treated as a proxy sensitivity model, not a measurement
    or exact transfer function of the anneal control line.
    """
    _, part, raw, phase, _ = _load_complete_director_corpus(corpus_path)
    if calibration_family not in APPROVED_FAMILIES:
        raise DirectorMatrixError("filter sensitivity is restricted to frozen System4/System6 calibration families")
    filename, _ = APPROVED_FAMILIES[calibration_family]
    curve = load_calibration(Path(project_root) / "calibration" / filename, calibration_family)
    idx = np.flatnonzero((part == "confirmation") | (part == "ood"))[: int(sample_controls)]
    if idx.size == 0:
        raise DirectorMatrixError("filter sensitivity requires protected controls")
    rows = []
    for i in idx:
        c = raw[i]
        t, s = rc.fourier_forward_schedule(c, order=protocol.order, grid_points=129, reject_nonmonotone=True)
        tf, sf = approximate_ccjj_filtered_schedule(t, s, bandwidth_mhz=CCJJ_BANDWIDTH_MHZ)
        nominal = phase[i]
        filtered = _operator_action_unconstrained(tf, sf, curve, n_segments=nominal.size // 2)
        rel = float(np.linalg.norm(filtered - nominal) / max(np.linalg.norm(nominal), 1e-15))
        rows.append({"design_index": int(i), "relative_operator_action_distortion": rel, "nominal": nominal.tolist(), "filter_proxy": filtered.tolist()})
    distort = np.asarray([r["relative_operator_action_distortion"] for r in rows])
    payload = {
        "schema": SCHEMA + "-D45-07", "execution_complete": True,
        "family": calibration_family, "bandwidth_mhz": CCJJ_BANDWIDTH_MHZ,
        "filter_model": "second-order digital Bessel proxy sensitivity model; not a measured delivered anneal waveform",
        "n_controls": len(rows), "median_relative_distortion": float(np.median(distort)),
        "q95_relative_distortion": float(np.quantile(distort, 0.95)), "max_relative_distortion": float(np.max(distort)),
        "rows": rows, "waveform_characterization_valid": True, "claim_passed": True, "passed": True,
    }
    _atomic_json(output_path, payload)
    return payload

# ---------------------------------------------------------------------------
# D45-08 — System4 target residual transfer vs scratch.
# ---------------------------------------------------------------------------

TRANSFER_TARGET_COUNT = 256
TRANSFER_CAL_COUNT = 32
TRANSFER_CONFIRM_COUNT = 32
TRANSFER_TRAIN_MAX = TRANSFER_TARGET_COUNT - TRANSFER_CAL_COUNT - TRANSFER_CONFIRM_COUNT
TRANSFER_LEARNING_CURVE = (66, 80, 96, 128, 160, 192)
TRANSFER_PRIMARY_MSE = 0.010
TRANSFER_MIN_SPEARMAN = 0.80


def advance_d45_08_target_step(
    target_evaluator: Any, *, source_corpus_path: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    _, part, raw, _, _ = _load_complete_director_corpus(source_corpus_path)
    eligible = np.flatnonzero(part == "train")[:TRANSFER_TARGET_COUNT]
    if eligible.size != TRANSFER_TARGET_COUNT:
        raise DirectorMatrixError("source director corpus does not contain the frozen 256 transfer controls")
    payload = _load_json(output_path, {"schema": SCHEMA + "-D45-08-TARGET-CORPUS", "records": []})
    records = list(payload.get("records", [])); existing = {int(r["source_index"]) for r in records}
    pending = [int(i) for i in eligible if int(i) not in existing]
    if not pending:
        payload["complete"] = True; _atomic_json(output_path, payload); return {"complete": True, "records": len(records)}
    i = pending[0]; control = raw[i]
    response = target_evaluator(control, num_reads=DEFAULT_DIRECTOR_PLAN.reads_per_control)
    records.append({"source_index": i, "control": control.tolist(), "response": {k: _jsonable(v) for k, v in response.items() if k != "sampleset"}})
    records.sort(key=lambda r: int(r["source_index"]))
    payload = {"schema": SCHEMA + "-D45-08-TARGET-CORPUS", "records": records, "complete": len(records) == TRANSFER_TARGET_COUNT,
               "atomic_rule": "one target-context annealer evaluation per invocation"}
    _atomic_json(output_path, payload)
    return {"complete": payload["complete"], "records": len(records), "required": TRANSFER_TARGET_COUNT, "latest_source_index": i}


def run_d45_08_transfer_analysis(
    *, source_corpus_path: str | Path, target_corpus_path: str | Path, project_root: str | Path, output_path: str | Path,
    target_name: str = "elite_probability", primary_mse_target: float = TRANSFER_PRIMARY_MSE,
    minimum_spearman: float = TRANSFER_MIN_SPEARMAN,
) -> dict[str, Any]:
    """Pre-frozen learning-curve test of cross-context residual transfer."""
    _, part, _, source_phase, source_Y = _load_complete_director_corpus(source_corpus_path)
    target_payload = _load_json(target_corpus_path, {})
    if not target_payload.get("complete"):
        raise DirectorMatrixError("System4 target corpus is incomplete")
    rows = list(target_payload["records"])
    indices = np.asarray([int(r["source_index"]) for r in rows], dtype=int)
    target_phase = np.asarray([r["response"]["operator_action"] for r in rows], dtype=float)
    target_Y = np.asarray([[float(r["response"][k]) for k in TARGET_NAMES] for r in rows], dtype=float)
    source_phase_sub = source_phase[indices]; source_Y_sub = source_Y[indices]
    ti = TARGET_NAMES.index(target_name)
    ntrain, ncal = TRANSFER_TRAIN_MAX, TRANSFER_CAL_COUNT
    cal_idx = np.arange(ntrain, ntrain + ncal)
    confirm_idx = np.arange(ntrain + ncal, TRANSFER_TARGET_COUNT)
    source_model = fit_cssf_qa_response(
        source_phase_sub[:ntrain], source_Y_sub[:ntrain], calibration_operator_phase=source_phase_sub[cal_idx],
        calibration_targets=source_Y_sub[cal_idx], target_names=TARGET_NAMES, project_root=project_root,
        support_mode="signed_axes", support_order=1, metadata={"experiment": "D45-08", "role": "source_context"},
    )
    source_baseline_all = source_model.predict(source_phase_sub)
    curve_rows = []
    for n in TRANSFER_LEARNING_CURVE:
        if n > ntrain:
            continue
        scratch = fit_cssf_qa_response(
            target_phase[:n], target_Y[:n], calibration_operator_phase=target_phase[cal_idx], calibration_targets=target_Y[cal_idx],
            target_names=TARGET_NAMES, project_root=project_root, support_mode="signed_axes", support_order=1,
            metadata={"experiment": "D45-08", "role": "scratch", "target_observations": n},
        )
        residual_targets = target_Y[:n] - source_baseline_all[:n]
        residual_cal = target_Y[cal_idx] - source_baseline_all[cal_idx]
        residual = fit_cssf_qa_response(
            target_phase[:n], residual_targets, calibration_operator_phase=target_phase[cal_idx], calibration_targets=residual_cal,
            target_names=TARGET_NAMES, project_root=project_root, support_mode="signed_axes", support_order=1,
            metadata={"experiment": "D45-08", "role": "source_plus_target_residual", "target_observations": n},
        )
        y = target_Y[confirm_idx, ti]
        ps = scratch.predict(target_phase[confirm_idx])[:, ti]
        pt = source_baseline_all[confirm_idx, ti] + residual.predict(target_phase[confirm_idx])[:, ti]
        diff = (ps - y) ** 2 - (pt - y) ** 2
        curve_rows.append({
            "target_observations": int(n), "scratch": _metric(y, ps), "transfer": _metric(y, pt),
            "paired_error_improvement": _paired_bootstrap_lcb(diff, seed=DIRECTOR_SEED + 800 + n),
        })
    def meets(metric: Mapping[str, Any]) -> bool:
        rho = float(metric["spearman"])
        return bool(float(metric["mse"]) <= float(primary_mse_target) and np.isfinite(rho) and rho >= float(minimum_spearman))
    transfer_hit = next((r for r in curve_rows if meets(r["transfer"])), None)
    scratch_hit = next((r for r in curve_rows if meets(r["scratch"])), None)
    positive = bool(transfer_hit is not None and scratch_hit is not None and transfer_hit["target_observations"] < scratch_hit["target_observations"])
    negative = bool(transfer_hit is None and scratch_hit is not None)
    payload = {
        "schema": SCHEMA + "-D45-08", "execution_complete": True, "target": target_name,
        "learning_curve": curve_rows,
        "prefrozen_quality_target": {"mse_max": float(primary_mse_target), "spearman_min": float(minimum_spearman)},
        "transfer_first_hit": transfer_hit, "scratch_first_hit": scratch_hit,
        "negative_transfer_detected": negative,
        "negative_transfer_gate": "broad transfer claim fails if target scratch reaches the pre-frozen quality target but transferred residual model does not, or if transfer does not reduce target observations",
        "positive_transfer_claim_passed": positive, "claim_passed": positive, "passed": positive,
    }
    _atomic_json(output_path, payload)
    return payload

# ---------------------------------------------------------------------------
# D45-10 — exact validation-defined physical superior-set mass p_gamma.
# ---------------------------------------------------------------------------

def _sampleset_placement_counts(sampleset: Any, arm: Any) -> tuple[dict[tuple[int, ...], int], int]:
    counts: dict[tuple[int, ...], int] = {}; total = 0
    for datum in sampleset.data(fields=["sample", "num_occurrences"], sorted_by=None):
        occ = int(datum.num_occurrences); total += occ
        sample = dict(datum.sample)
        if not arm.problem.is_feasible(sample):
            continue
        placement = arm.problem.decode(sample); key = tuple(map(int, placement.selected_buses))
        counts[key] = counts.get(key, 0) + occ
    return counts, total


def advance_d45_10_physical_mass_step(
    *, project_root: str | Path, arm: Any, evaluator: Any, control: ArrayLike, highs_placement: Any,
    output_path: str | Path, gamma: float = 0.0, reads: int = 512,
) -> dict[str, Any]:
    """Advance exact p_gamma construction by one atomic stage.

    Stage 1 performs one annealer evaluation and serializes all unique feasible
    placement occurrence counts.  Each later invocation physically labels one
    unique placement on the full validation partition.  No unique placement is
    dropped; this is exact with respect to the sampled read batch.
    """
    path = Path(output_path)
    payload = _load_json(path, {"schema": SCHEMA + "-D45-10", "stage": "sampling", "labels": {}})
    if payload.get("complete"):
        return payload
    if payload.get("stage") == "sampling":
        response = evaluator(np.asarray(control, float), num_reads=int(reads))
        ss = response.get("sampleset")
        if ss is None:
            raise DirectorMatrixError("p_gamma bridge requires the logical sampleset")
        counts, total = _sampleset_placement_counts(ss, arm)
        payload.update({
            "stage": "labeling", "control": np.asarray(control, float).tolist(), "reads": int(total),
            "placement_counts": {"|".join(map(str, k)): int(v) for k, v in counts.items()}, "labels": {},
            "gamma": float(gamma), "response_summary": {k: _jsonable(v) for k, v in response.items() if k != "sampleset"},
        })
        _atomic_json(path, payload)
        return {"complete": False, "stage": "labeling", "unique_placements": len(counts), "labeled": 0}
    counts = dict(payload.get("placement_counts", {})); labels = dict(payload.get("labels", {}))
    # HiGHS validation utility is comparator-only and can be computed once without touching OOD/N-1.
    if "__highs__" not in labels:
        _, meta = select_placement_on_validation(project_root, {"HiGHS": highs_placement})
        summary = meta["candidates"][0]["summary"]
        labels["__highs__"] = {"utility": -float(summary["mean_penalized_objective"]), "selected_buses": list(map(int, highs_placement.selected_buses))}
        payload["labels"] = labels; _atomic_json(path, payload)
        return {"complete": False, "stage": "labeling", "labeled": len(labels) - 1, "remaining": len(counts)}
    pending = [k for k in counts if k not in labels]
    if pending:
        key = pending[0]; buses = tuple(int(x) for x in key.split("|") if x)
        # BESSPlacement is immutable and is reconstructed exactly from the arm's frozen candidate fleet.
        from opf.bess_constraints import BESSPlacement
        placement = BESSPlacement.from_selected_buses(arm.fleet, buses)
        _, meta = select_placement_on_validation(project_root, {key: placement})
        summary = meta["candidates"][0]["summary"]
        labels[key] = {"utility": -float(summary["mean_penalized_objective"]), "selected_buses": list(buses), "occurrences": int(counts[key])}
        payload["labels"] = labels; _atomic_json(path, payload)
        return {"complete": False, "stage": "labeling", "labeled": len(labels) - 1, "remaining": len(counts) - (len(labels) - 1), "latest": key}
    u_ref = float(labels["__highs__"]["utility"]); threshold = u_ref + float(payload["gamma"])
    superior_occ = sum(int(counts[k]) for k in counts if float(labels[k]["utility"]) >= threshold)
    p_gamma = float(superior_occ / max(1, int(payload["reads"])))
    payload.update({
        "stage": "complete", "complete": True, "reference_utility": u_ref, "superior_threshold": threshold,
        "superior_occurrences": int(superior_occ), "p_gamma_validation": p_gamma,
        "definition": "probability mass of sampled placements whose full validation-partition physical utility >= HiGHS validation utility + gamma",
        "ood_n1_used_for_control_selection": False,
        "execution_complete": True,
        "physical_bridge_complete": True,
        "positive_physical_mass_claim_passed": bool(p_gamma > 0.0),
        "claim_passed": bool(p_gamma > 0.0),
        "passed": bool(p_gamma > 0.0),
    })
    _atomic_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# D45-09 — cost to independently confirmed physical target.
# ---------------------------------------------------------------------------

def run_d45_09_physical_cost_to_target(
    *, method_costs: Mapping[str, Mapping[str, float]], confirmation_by_method: Mapping[str, Mapping[str, Any]],
    reference_method: str, output_path: str | Path, gamma: float = 0.0,
    cssf_method: str = "CSSF-full", primary_cost_axis: str = "control_evaluation",
) -> dict[str, Any]:
    """Fully charged campaign cost to an independently confirmed physical target.

    This deliberately does *not* fabricate an intermediate physical first-hit
    time when physical utility was not evaluated after every search query.
    Methods that do not reach the predeclared reference+gamma confirmation target
    are censored.  Among methods that do reach it, the predeclared primary cost
    axis is compared while the complete observed resource vector is retained.
    """
    if reference_method not in confirmation_by_method:
        raise DirectorMatrixError("physical target requires a reference confirmation row")
    ref_vals = np.asarray([-float(x["penalized_objective"]) for x in confirmation_by_method[reference_method]["rows"]], float)
    target = float(np.mean(ref_vals) + float(gamma))
    rows = []
    for method, confirmation in confirmation_by_method.items():
        vals = np.asarray([-float(x["penalized_objective"]) for x in confirmation["rows"]], float)
        mean = float(np.mean(vals)); reached = bool(mean >= target)
        cost = {str(k): float(v) for k, v in dict(method_costs.get(method, {})).items()}
        rows.append({
            "method": method, "confirmed_mean_utility": mean, "target": target, "reached": reached,
            "fully_charged_campaign_cost": cost if reached else None,
            "primary_cost": (float(cost[primary_cost_axis]) if reached and primary_cost_axis in cost else None),
        })
    cssf_row = next((r for r in rows if r["method"] == cssf_method), None)
    competitor_rows = [r for r in rows if r["method"] not in {cssf_method, reference_method} and r["reached"] and r["primary_cost"] is not None]
    strongest = min(competitor_rows, key=lambda r: float(r["primary_cost"])) if competitor_rows else None
    claim_passed = bool(
        cssf_row is not None and cssf_row["reached"] and cssf_row["primary_cost"] is not None
        and strongest is not None and float(cssf_row["primary_cost"]) < float(strongest["primary_cost"])
    )
    payload = {
        "schema": SCHEMA + "-D45-09", "execution_complete": True,
        "reference_method": reference_method, "gamma": float(gamma), "physical_target": target,
        "primary_cost_axis": primary_cost_axis, "rows": rows,
        "strongest_reached_competitor_on_primary_axis": strongest,
        "censoring_rule": "methods not reaching the independent confirmation target remain NOT_REACHED; no imputation",
        "first_hit_boundary": "this is fully charged campaign-to-confirmed-target, not an invented intermediate first-hit cost; first-hit physical cost requires physical evaluation at each intermediate query",
        "commercial_cost_claim_passed": claim_passed, "claim_passed": claim_passed, "passed": claim_passed,
    }
    _atomic_json(output_path, payload)
    return payload

# ---------------------------------------------------------------------------
# Aggregate director gate.
# ---------------------------------------------------------------------------

D45_FILES = {
    "D45-01": "d45_01_segmented_vs_global.json",
    "D45-02": "d45_02_dictionary_gate.json",
    "D45-03": "d45_03_multi_vs_scalar.json",
    "D45-04": "d45_04_dwave_incumbent.json",
    "D45-05": "d45_05_query_frontier.json",
    "D45-06": "d45_06_control_leverage.json",
    "D45-07": "d45_07_filter_sensitivity.json",
    "D45-08": "d45_08_transfer_analysis.json",
    "D45-09": "d45_09_physical_cost_to_target.json",
    "D45-10": "d45_10_physical_mass.json",
}


def director_matrix_gate(evidence_root: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    """Separate protocol completeness from positive scientific outcomes.

    A rigorously executed falsification experiment is not converted into an
    engineering failure merely because the CSSF hypothesis did not win.  Claim
    booleans are therefore reported independently from execution completeness.
    """
    root = Path(evidence_root); rows = []
    payloads: dict[str, Mapping[str, Any]] = {}
    for eid, filename in D45_FILES.items():
        path = root / filename
        if not path.is_file():
            rows.append({"experiment": eid, "execution_status": "MISSING", "execution_complete": False, "claim_passed": False, "path": str(path)})
            continue
        payload = _load_json(path, {}); payloads[eid] = payload
        complete = bool(payload.get("execution_complete", payload.get("complete", payload.get("status") == "COMPLETE")))
        claim = bool(payload.get("claim_passed", payload.get("passed", False)))
        rows.append({
            "experiment": eid, "execution_status": "COMPLETE" if complete else "INCOMPLETE",
            "execution_complete": complete, "claim_passed": claim, "path": str(path),
        })
    evidence_complete = bool(len(rows) == len(D45_FILES) and all(r["execution_complete"] for r in rows))
    mechanism_claim = bool(
        payloads.get("D45-01", {}).get("claim_passed", False)
        and payloads.get("D45-02", {}).get("dictionary_gate_passed", False)
        and payloads.get("D45-03", {}).get("claim_passed", False)
    )
    transfer_claim = bool(payloads.get("D45-08", {}).get("positive_transfer_claim_passed", False))
    commercial_cost_claim = bool(payloads.get("D45-09", {}).get("commercial_cost_claim_passed", False))
    physical_bridge = bool(payloads.get("D45-10", {}).get("physical_bridge_complete", False))
    broad_query_claim = bool(payloads.get("D45-05", {}).get("broad_query_efficiency_claim_ready", False))
    payload = {
        "schema": SCHEMA + "-AGGREGATE-GATE", "experiments": rows,
        "evidence_execution_complete": evidence_complete,
        "mechanism_claim_passed": mechanism_claim,
        "positive_transfer_claim_passed": transfer_claim,
        "commercial_cost_claim_passed": commercial_cost_claim,
        "physical_bridge_complete": physical_bridge,
        "broad_query_efficiency_claim_unlocked": broad_query_claim,
        "simulator_composition_claim_ready": bool(evidence_complete and mechanism_claim and physical_bridge),
        "simulator_commercial_value_hypothesis_passed": bool(evidence_complete and mechanism_claim and transfer_claim and commercial_cost_claim and physical_bridge),
        "hardware_claim_boundary": "Simulator evidence cannot satisfy live System4/System6 QPU timing or hardware-confirmation gates",
        "no_predetermined_superiority": True,
    }
    if output_path is None:
        output_path = root / "director_matrix_gate_v47.json"
    _atomic_json(output_path, payload)
    return payload

__all__ = [
    "DirectorMatrixError", "DirectorCorpusPlan", "DEFAULT_DIRECTOR_PLAN", "director_validation_design",
    "advance_raw_trig_corpus_v47_step", "analyze_raw_trig_corpus_v47", "advance_director_corpus_step", "global_integrated_action", "run_d45_01_segmented_vs_global",
    "run_d45_02_dictionary_gate", "run_d45_03_multioutput_vs_scalar", "dwave_incumbent_schedules",
    "advance_d45_04_incumbent_step", "run_d45_05_query_frontier_gate", "run_d45_06_control_leverage",
    "approximate_ccjj_filtered_schedule", "run_d45_07_filter_sensitivity", "advance_d45_08_target_step",
    "run_d45_08_transfer_analysis", "advance_d45_10_physical_mass_step", "run_d45_09_physical_cost_to_target",
    "director_matrix_gate", "D45_FILES", "SOFT_WALL_SECONDS", "HARD_WALL_SECONDS",
]

# ---------------------------------------------------------------------------
# v51 synchronized four-task / three-domain registry and scientific gates.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProgramExperiment:
    experiment_id: str
    program: str
    question: str
    requires: tuple[str, ...] = ()
    external_result_field: str = ""
    physical_confirmation_required: bool = True


PROGRAM_CASE300 = "case300_ac_bess"
PROGRAM_IEEE33 = "ieee33_resilience"
PROGRAM_MOTOR = "few_fem_outer_rotor_bldc"
PROGRAM_EEG = "eeg_phase_syntax_microstates"


@dataclass(frozen=True)
class ApplicationResultSchema:
    """Fail-closed evidence requirements for non-EEG application experiments.

    A completed record may be reportable even when a superiority hypothesis fails,
    but a positive claim cannot become ready unless every required confirmation tier
    is explicitly present.
    """
    experiment_id: str
    program: str
    requires_application_data: bool = True
    requires_independent_confirmation: bool = False
    requires_live_qpu: bool = False
    requires_prototype: bool = False


def validate_application_result_payload(payload: Mapping[str, Any], schema: ApplicationResultSchema) -> dict[str, Any]:
    execution_complete = bool(payload.get("execution_complete", False))
    application_data_confirmed = bool(payload.get("application_data_confirmed", False))
    independent_confirmation = bool(payload.get("independent_confirmation", False))
    live_qpu_confirmed = bool(payload.get("live_qpu_confirmed", False))
    prototype_confirmed = bool(payload.get("prototype_confirmed", False))
    requirements = {
        "application_data": (not schema.requires_application_data) or application_data_confirmed,
        "independent_confirmation": (not schema.requires_independent_confirmation) or independent_confirmation,
        "live_qpu": (not schema.requires_live_qpu) or live_qpu_confirmed,
        "prototype": (not schema.requires_prototype) or prototype_confirmed,
    }
    confirmation_ready = bool(all(requirements.values()))
    report_ready = bool(execution_complete and confirmation_ready)
    claim_passed = bool(payload.get("claim_passed", False))
    positive_claim_ready = bool(report_ready and claim_passed)
    return {
        "experiment_id": schema.experiment_id,
        "program": schema.program,
        "execution_complete": execution_complete,
        "claim_passed": claim_passed,
        "confirmation_requirements": requirements,
        "confirmation_ready": confirmation_ready,
        "report_ready": report_ready,
        "positive_claim_ready": positive_claim_ready,
        "evidence_tier": str(payload.get("evidence_tier", "UNDECLARED")),
    }


def application_fail_closed_gate(
    evidence_root: str | Path, schemas: Sequence[ApplicationResultSchema], *, output_path: str | Path | None = None
) -> dict[str, Any]:
    root = Path(evidence_root)
    rows: list[dict[str, Any]] = []
    for schema in schemas:
        p = root / (schema.experiment_id.lower().replace("/", "_") + ".json")
        payload = _load_json(p, {}) if p.exists() else {}
        row = validate_application_result_payload(payload, schema)
        row.update({"exists": p.exists(), "path": str(p)})
        rows.append(row)
    all_report_ready = bool(rows and all(r["report_ready"] for r in rows))
    any_positive = bool(any(r["positive_claim_ready"] for r in rows))
    out = {
        "schema": SCHEMA + "-APPLICATION-FAIL-CLOSED",
        "execution_complete": True,
        "experiments": rows,
        "all_report_ready": all_report_ready,
        "any_positive_claim_ready": any_positive,
        "claim_status": "READY_FOR_RESULT_REVIEW" if all_report_ready else "PENDING_OR_UNCONFIRMED",
        "null_and_adverse_results_are_reportable_when_confirmed": True,
    }
    if output_path is not None:
        _atomic_json(output_path, out)
    return out


def ieee33_result_schemas() -> tuple[ApplicationResultSchema, ...]:
    # Every IEEE-33 result must be based on the declared application/fault corpus and
    # independently evaluated AC/restoration/WRAP physics before it is reportable.
    return tuple(
        ApplicationResultSchema(
            e.experiment_id, PROGRAM_IEEE33, requires_application_data=True,
            requires_independent_confirmation=True,
        )
        for e in IEEE33_EXPERIMENTS
    )


def motor_result_schemas() -> tuple[ApplicationResultSchema, ...]:
    live = {"M59-09", "M59-10", "M59-14"}
    independent = {"M59-11", "M59-12", "M59-13", "M59-14", "M59-15"}
    prototype = {"M59-12", "M59-15"}
    return tuple(
        ApplicationResultSchema(
            e.experiment_id, PROGRAM_MOTOR, requires_application_data=True,
            requires_independent_confirmation=e.experiment_id in independent,
            requires_live_qpu=e.experiment_id in live,
            requires_prototype=e.experiment_id in prototype,
        )
        for e in MOTOR_EXPERIMENTS
    )


def _exp(eid: str, program: str, question: str, requires: Sequence[str] = (), field: str = "") -> ProgramExperiment:
    return ProgramExperiment(eid, program, question, tuple(requires), field, True)


CASE300_EXPERIMENTS: tuple[ProgramExperiment, ...] = (
    _exp("P1-L1", PROGRAM_CASE300, "canonical application-domain trigonometrization versus raw representation", field="case300_level1"),
    *tuple(_exp(f"P1-{eid}", PROGRAM_CASE300, f"retained director evidence closure {eid}", requires=("P1-L1",), field=eid.lower()) for eid in D45_FILES),
)

IEEE33_EXPERIMENTS: tuple[ProgramExperiment, ...] = (
    _exp("R33-RQ1", PROGRAM_IEEE33, "loss-oriented CSSF+QA BESS placement transferred to independent WRAP evaluation", field="ieee33_rq1_loss_transfer"),
    _exp("R33-RQ2", PROGRAM_IEEE33, "resilience-aware BESS objective/QUBO after the loss-only causal test", requires=("R33-RQ1",), field="ieee33_rq2_resilience_bess"),
    _exp("R33-E0", PROGRAM_IEEE33, "faithful Mishra IEEE-33 reference reconstruction", field="ieee33_reference"),
    _exp("R33-E1", PROGRAM_IEEE33, "complete full-resource reference baseline reproduction", requires=("R33-E0",), field="ieee33_full_resource_baseline"),
    _exp("R33-E2", PROGRAM_IEEE33, "full-resource CSSF joint planning/control", requires=("R33-RQ2", "R33-E1"), field="ieee33_full_resource_cssf"),
    _exp("R33-E3", PROGRAM_IEEE33, "loss-oriented versus resilience-aware Pareto under identical resources", requires=("R33-E2",), field="ieee33_pareto"),
    _exp("R33-E4", PROGRAM_IEEE33, "raw versus toric/complex application representation", requires=("R33-E1",), field="ieee33_toric_causality"),
    _exp("R33-E5", PROGRAM_IEEE33, "three-level CSSF plus D0/D1/D2/D3 and response ablations", requires=("R33-E4",), field="ieee33_three_level"),
    _exp("R33-E6", PROGRAM_IEEE33, "causal resource deletions and interaction effects", requires=("R33-E2",), field="ieee33_resource_ablation"),
    _exp("R33-E7", PROGRAM_IEEE33, "matched classical resilience competitor frontier", requires=("R33-E1",), field="ieee33_classical_frontier"),
    _exp("R33-E8", PROGRAM_IEEE33, "endogenous/worst-case contingency discovery", requires=("R33-E2",), field="ieee33_worst_case"),
    _exp("R33-E9", PROGRAM_IEEE33, "coupled transportation-power constraints for mobile storage", requires=("R33-E1",), field="ieee33_transport_power"),
    _exp("R33-E10", PROGRAM_IEEE33, "protected fault OOD and topology transfer", requires=("R33-E5", "R33-E8"), field="ieee33_ood"),
    _exp("R33-E11", PROGRAM_IEEE33, "cost to independently confirmed resilience target", requires=("R33-E7", "R33-E10"), field="ieee33_cost_target"),
)

MOTOR_EXPERIMENTS: tuple[ProgramExperiment, ...] = (
    _exp("M59-01", PROGRAM_MOTOR, "rotor/stator symmetry and physical period audit", field="motor_period"),
    _exp("M59-02", PROGRAM_MOTOR, "periodic CSNN-T versus raw/no-trig and Fourier-reduced alternatives", requires=("M59-01",), field="motor_periodic_causality"),
    _exp("M59-03", PROGRAM_MOTOR, "Few-FEM learning curve at matched charged FEM budgets", requires=("M59-02",), field="motor_few_fem"),
    _exp("M59-04", PROGRAM_MOTOR, "harmonic/design-support stability and OOD design-region tests", requires=("M59-02",), field="motor_support_stability"),
    _exp("M59-05", PROGRAM_MOTOR, "sparse pairwise QUBO bridge fidelity against untouched FEM", requires=("M59-03", "M59-04"), field="motor_qubo_fidelity"),
    _exp("M59-06", PROGRAM_MOTOR, "direct electric-machine QA prior-art comparator", requires=("M59-05",), field="motor_qa_prior_art"),
    _exp("M59-07", PROGRAM_MOTOR, "identical frozen-QUBO classical discrete-solver frontier", requires=("M59-05",), field="motor_classical_frontier"),
    _exp("M59-08", PROGRAM_MOTOR, "QA trigonometrization causal matrix and control-leverage abstention", requires=("M59-05",), field="motor_qa_trig"),
    _exp("M59-09", PROGRAM_MOTOR, "primary System6 candidate campaign", requires=("M59-08",), field="motor_system6"),
    _exp("M59-10", PROGRAM_MOTOR, "System4 reproducibility and transfer", requires=("M59-09",), field="motor_system4_transfer"),
    _exp("M59-11", PROGRAM_MOTOR, "independent electromagnetic/thermal/demagnetization/stress confirmation", requires=("M59-07", "M59-09"), field="motor_multiphysics"),
    _exp("M59-12", PROGRAM_MOTOR, "manufactured prototype and dyno tier", requires=("M59-11",), field="motor_prototype_dyno"),
    _exp("M59-13", PROGRAM_MOTOR, "Few-FEM full cost-to-confirmed-target", requires=("M59-11",), field="motor_cost_target"),
    _exp("M59-14", PROGRAM_MOTOR, "separate QPU contribution at matched budget", requires=("M59-07", "M59-09", "M59-11"), field="motor_qpu_contribution"),
    _exp("M59-15", PROGRAM_MOTOR, "production-value extrapolation only after physical confirmation", requires=("M59-12", "M59-13"), field="motor_production_value"),
)

EEG_EXPERIMENTS: tuple[ProgramExperiment, ...] = (
    _exp("N67-01", PROGRAM_EEG, "artifact/result provenance audit and quarantine of unreproduced supplied numbers", field="eeg_provenance"),
    _exp("N67-02", PROGRAM_EEG, "native periodicity audit on protected real/public EEG", requires=("N67-01",), field="eeg_native_periodicity"),
    _exp("N67-03", PROGRAM_EEG, "phase extraction and preprocessing sensitivity", requires=("N67-02",), field="eeg_preprocessing_sensitivity"),
    _exp("N67-04", PROGRAM_EEG, "same-corpus toric CSNN-T representation causality", requires=("N67-02", "N67-03"), field="eeg_toric_causality"),
    _exp("N67-05", PROGRAM_EEG, "Stuart-Landau weak-coupling approximation audit", requires=("N67-01",), field="eeg_stuart_landau"),
    _exp("N67-06", PROGRAM_EEG, "reduced-connectome versus protected-rich phase dictionary", requires=("N67-04", "N67-05"), field="eeg_dictionary_gate"),
    _exp("N67-07", PROGRAM_EEG, "data-processing-inequality structural demonstration and real-EEG information-gap separation", requires=("N67-01",), field="eeg_information_gap"),
    _exp("N67-08", PROGRAM_EEG, "aligned one-step-ahead symbolic versus continuous prediction", requires=("N67-04", "N67-07"), field="eeg_aligned_prediction"),
    _exp("N67-09", PROGRAM_EEG, "causal-state algorithm frontier including epsilon-automata/CSSR/kernel epsilon-machines", requires=("N67-08",), field="eeg_causal_state_frontier"),
    _exp("N67-10", PROGRAM_EEG, "nested selection of predictive state/class complexity", requires=("N67-09",), field="eeg_state_complexity"),
    _exp("N67-11", PROGRAM_EEG, "true group-level objective and exact/fidelity-gated syntax QUBO", requires=("N67-10",), field="eeg_qubo_fidelity"),
    _exp("N67-12", PROGRAM_EEG, "certified discrete-search diagnosis on nested/full instances", requires=("N67-11",), field="eeg_search_certificate"),
    _exp("N67-13", PROGRAM_EEG, "Pegasus-P16 embedding, precision and chain feasibility", requires=("N67-11", "N67-12"), field="eeg_pegasus_feasibility"),
    _exp("N67-14", PROGRAM_EEG, "QA trigonometrization causal matrix and response-leverage abstention", requires=("N67-13",), field="eeg_qa_trig"),
    _exp("N67-15", PROGRAM_EEG, "primary System6 live-QPU campaign", requires=("N67-14",), field="eeg_system6"),
    _exp("N67-16", PROGRAM_EEG, "System4 reproducibility and transfer/negative-transfer campaign", requires=("N67-15",), field="eeg_system4_transfer"),
    _exp("N67-17", PROGRAM_EEG, "predeclared test-retest reliability endpoint", requires=("N67-08",), field="eeg_test_retest"),
    _exp("N67-18", PROGRAM_EEG, "multi-subject scaling with participant leakage barriers", requires=("N67-10", "N67-12"), field="eeg_multisubject"),
    _exp("N67-19", PROGRAM_EEG, "subjective-state association with multiplicity control", requires=("N67-17",), field="eeg_subjective_state"),
    _exp("N67-20", PROGRAM_EEG, "independent frozen-method public replication", requires=("N67-17", "N67-18"), field="eeg_replication"),
    _exp("N67-21", PROGRAM_EEG, "cost to independently confirmed neuroscience target", requires=("N67-12", "N67-15", "N67-20"), field="eeg_cost_target"),
)


FOUR_TASK_EXPERIMENTS: tuple[ProgramExperiment, ...] = CASE300_EXPERIMENTS + IEEE33_EXPERIMENTS + MOTOR_EXPERIMENTS + EEG_EXPERIMENTS
# Backward-compatible alias for v48 consumers; v51 claim authority is FOUR_TASK_EXPERIMENTS.
THREE_PROGRAM_EXPERIMENTS = FOUR_TASK_EXPERIMENTS


def validate_four_task_registry(experiments: Sequence[ProgramExperiment] = FOUR_TASK_EXPERIMENTS) -> dict[str, Any]:
    ids = [e.experiment_id for e in experiments]
    if len(ids) != len(set(ids)):
        raise DirectorMatrixError("four-task experiment ids must be unique")
    by_id = {e.experiment_id: e for e in experiments}
    required_programs = {PROGRAM_CASE300, PROGRAM_IEEE33, PROGRAM_MOTOR, PROGRAM_EEG}
    present_programs = {e.program for e in experiments}
    if present_programs != required_programs:
        raise DirectorMatrixError(f"program registry mismatch: {sorted(present_programs)}")
    for e in experiments:
        missing = [r for r in e.requires if r not in by_id]
        if missing:
            raise DirectorMatrixError(f"{e.experiment_id} has missing dependencies {missing}")
    # Exact cycle check.
    visiting: set[str] = set(); done: set[str] = set()
    def dfs(eid: str) -> None:
        if eid in done: return
        if eid in visiting: raise DirectorMatrixError(f"dependency cycle at {eid}")
        visiting.add(eid)
        for dep in by_id[eid].requires: dfs(dep)
        visiting.remove(eid); done.add(eid)
    for eid in ids: dfs(eid)
    return {
        "schema": SCHEMA + "-FOUR-TASK-REGISTRY",
        "execution_complete": True,
        "n_experiments": len(ids),
        "program_counts": {p: sum(e.program == p for e in experiments) for p in sorted(required_programs)},
        "registry_sha256": _hash_payload({"experiments": [asdict(e) for e in experiments]}, "four-task-registry"),
    }


def write_four_task_manifest(path: str | Path, experiments: Sequence[ProgramExperiment] = FOUR_TASK_EXPERIMENTS) -> dict[str, Any]:
    audit = validate_four_task_registry(experiments)
    payload = {
        **audit,
        "experiments": [asdict(e) for e in experiments],
        "result_policy": "pending fields remain explicit; simulator/QPU/physical evidence are never conflated",
    }
    _atomic_json(path, payload)
    return payload


def pairwise_binary_features(X: ArrayLike, *, interactions: Sequence[tuple[int, int]] | None = None) -> tuple[NDArray[np.float64], list[str]]:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise DirectorMatrixError("X must be a non-empty 2D binary design matrix")
    if not np.isin(x, [0.0, 1.0]).all():
        raise DirectorMatrixError("pairwise bridge requires binary 0/1 design variables")
    n, d = x.shape
    pairs = list(interactions) if interactions is not None else [(i, j) for i in range(d) for j in range(i + 1, d)]
    for i, j in pairs:
        if not (0 <= i < j < d):
            raise DirectorMatrixError(f"invalid interaction ({i},{j}) for d={d}")
    cols = [np.ones(n, dtype=float)] + [x[:, i] for i in range(d)] + [x[:, i] * x[:, j] for i, j in pairs]
    names = ["intercept"] + [f"x{i}" for i in range(d)] + [f"x{i}*x{j}" for i, j in pairs]
    return np.column_stack(cols), names


def validate_qubo_bridge(
    true_utility: ArrayLike, predicted_utility: ArrayLike, *, true_feasible: ArrayLike | None = None,
    predicted_feasible: ArrayLike | None = None, top_k: int = 5, max_regret: float | None = None,
    min_spearman: float = 0.70, min_topk_recall: float = 0.50, min_feasibility_accuracy: float = 0.95,
) -> dict[str, Any]:
    y = np.asarray(true_utility, dtype=float).reshape(-1)
    p = np.asarray(predicted_utility, dtype=float).reshape(-1)
    if y.shape != p.shape or y.size < 4 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise DirectorMatrixError("QUBO bridge validation requires aligned finite arrays with n>=4")
    m = _metric(y, p, top_k=top_k)
    if not np.isfinite(m["spearman"]):
        m["spearman"] = 0.0
    feas_acc = 1.0
    if true_feasible is not None or predicted_feasible is not None:
        if true_feasible is None or predicted_feasible is None:
            raise DirectorMatrixError("both feasibility arrays must be supplied together")
        a = np.asarray(true_feasible, dtype=bool).reshape(-1); b = np.asarray(predicted_feasible, dtype=bool).reshape(-1)
        if a.shape != y.shape or b.shape != y.shape:
            raise DirectorMatrixError("feasibility arrays must align with utilities")
        feas_acc = float(np.mean(a == b))
    regret_cap = float(max_regret) if max_regret is not None else max(1e-12, 0.05 * float(np.ptp(y)))
    passed = bool(m["spearman"] >= min_spearman and m["topk_recall"] >= min_topk_recall and m["decision_regret"] <= regret_cap and feas_acc >= min_feasibility_accuracy)
    return {
        "schema": SCHEMA + "-QUBO-BRIDGE-VALIDATION", "execution_complete": True, "passed": passed,
        "metrics": m, "feasibility_accuracy": feas_acc,
        "thresholds": {"min_spearman": min_spearman, "min_topk_recall": min_topk_recall, "max_regret": regret_cap, "min_feasibility_accuracy": min_feasibility_accuracy},
    }


def fit_pairwise_qubo_bridge(
    X: ArrayLike, utility: ArrayLike, *, train_indices: Sequence[int], validation_indices: Sequence[int],
    interactions: Sequence[tuple[int, int]] | None = None, ridge: float = 1e-8, top_k: int = 5,
) -> dict[str, Any]:
    x = np.asarray(X, dtype=float); y = np.asarray(utility, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise DirectorMatrixError("X and utility must align")
    tr = np.asarray(train_indices, dtype=int); va = np.asarray(validation_indices, dtype=int)
    if tr.size < 3 or va.size < 4 or set(tr.tolist()) & set(va.tolist()):
        raise DirectorMatrixError("train/validation partitions must be disjoint and sufficiently large")
    if min(tr.min(), va.min()) < 0 or max(tr.max(), va.max()) >= y.size:
        raise DirectorMatrixError("partition index out of range")
    F, names = pairwise_binary_features(x, interactions=interactions)
    gram = F[tr].T @ F[tr]
    penalty = np.eye(gram.shape[0]) * float(ridge); penalty[0, 0] = 0.0
    coef = np.linalg.solve(gram + penalty, F[tr].T @ y[tr])
    pred = F @ coef
    validation = validate_qubo_bridge(y[va], pred[va], top_k=top_k)
    return {
        "schema": SCHEMA + "-PAIRWISE-QUBO-BRIDGE", "execution_complete": True,
        "feature_names": names, "coefficients": coef.tolist(),
        "train_indices": tr.tolist(), "validation_indices": va.tolist(),
        "validation": validation, "predicted_utility": pred.tolist(),
        "interpretation_boundary": "empirical pairwise bridge; never evidence that the underlying nonlinear physics is quadratic",
    }


def compare_wrap_resilience(
    reference: Mapping[str, ArrayLike], candidate: Mapping[str, ArrayLike], *, alpha: float = 0.05,
    bootstrap_samples: int = 8000, seed: int = DIRECTOR_SEED,
) -> dict[str, Any]:
    """Paired same-scenario comparison with Mishra directions normalized to higher-is-better."""
    direction = {"withstand": 1.0, "recover": 1.0, "adapt": -1.0, "prevent": -1.0}
    if set(reference) != set(direction) or set(candidate) != set(direction):
        raise DirectorMatrixError("WRAP comparison requires withstand/recover/adapt/prevent exactly")
    rows: dict[str, Any] = {}
    for k, sign in direction.items():
        a = np.asarray(reference[k], dtype=float).reshape(-1); b = np.asarray(candidate[k], dtype=float).reshape(-1)
        if a.shape != b.shape or a.size < 4:
            raise DirectorMatrixError(f"WRAP metric {k} must be paired with n>=4")
        stats = _paired_bootstrap_lcb(sign * (b - a), alpha=alpha, samples=bootstrap_samples, seed=seed + len(rows))
        stats["direction_normalized"] = "higher_is_better"
        stats["positive_transfer"] = bool(stats["lcb"] > 0.0)
        rows[k] = stats
    return {
        "schema": SCHEMA + "-WRAP-TRANSFER", "execution_complete": True,
        "metrics": rows, "all_four_positive": bool(all(v["positive_transfer"] for v in rows.values())),
        "claim_boundary": "same-scenario paired resilience transfer only; does not establish broad robust-resilience superiority",
    }


def periodic_signal_audit(theta: ArrayLike, values: ArrayLike, *, harmonics: int = 12) -> dict[str, Any]:
    th = np.asarray(theta, dtype=float).reshape(-1); y = np.asarray(values, dtype=float).reshape(-1)
    if th.shape != y.shape or th.size < max(8, 2 * int(harmonics) + 1) or not np.isfinite(th).all() or not np.isfinite(y).all():
        raise DirectorMatrixError("period audit requires aligned finite angle/signal arrays")
    if np.ptp(th) <= 0:
        raise DirectorMatrixError("angle grid must span a positive interval")
    # Scale the observed interval to one candidate period; this audits periodic representability, not physical symmetry identification by itself.
    phase = 2.0 * np.pi * (th - th.min()) / np.ptp(th)
    cols = [np.ones_like(phase)]
    for k in range(1, int(harmonics) + 1):
        cols.extend([np.cos(k * phase), np.sin(k * phase)])
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    scale = max(float(np.std(y)), 1e-12)
    nrmse = float(np.sqrt(np.mean((y - pred) ** 2)) / scale)
    endpoint_gap = float(abs(y[0] - y[-1]) / scale)
    return {
        "schema": SCHEMA + "-PERIODIC-SIGNAL-AUDIT", "execution_complete": True,
        "harmonics": int(harmonics), "normalized_rmse": nrmse, "normalized_endpoint_gap": endpoint_gap,
        "candidate_interval": float(np.ptp(th)),
        "claim_boundary": "numerical periodic representability only; the motor fundamental sector must also be justified by pole/slot/winding symmetry",
    }


def few_fem_cost_to_target(records: Sequence[Mapping[str, Any]], *, target_utility: float) -> dict[str, Any]:
    if not records:
        raise DirectorMatrixError("Few-FEM cost-to-target requires records")
    out: dict[str, Any] = {}
    for r in records:
        method = str(r["method"]); calls = np.asarray(r["fem_calls"], dtype=int); utility = np.asarray(r["confirmed_utility"], dtype=float)
        if calls.ndim != 1 or utility.ndim != 1 or calls.size != utility.size or calls.size == 0 or np.any(np.diff(calls) <= 0):
            raise DirectorMatrixError(f"invalid Few-FEM learning curve for {method}")
        hit = np.flatnonzero(utility >= float(target_utility))
        out[method] = {"status": "REACHED" if hit.size else "NOT_REACHED", "fem_calls_to_target": int(calls[hit[0]]) if hit.size else None, "best_confirmed_utility": float(np.max(utility))}
    return {"schema": SCHEMA + "-FEW-FEM-COST-TO-TARGET", "execution_complete": True, "target_utility": float(target_utility), "methods": out}


def conditional_result_placeholders() -> dict[str, str]:
    """External proposal fields. These are deliberately not results."""
    return {e.external_result_field: f"[[VERIFY AGAINST RESULTS: {e.experiment_id}]]" for e in FOUR_TASK_EXPERIMENTS if e.external_result_field}


def four_task_gate(evidence_root: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(evidence_root)
    registry = validate_four_task_registry()
    # New application programs are fail-closed until their evidence files exist. Case300 remains governed by the retained director matrix gate.
    case300_gate = director_matrix_gate(root / "case300") if (root / "case300").exists() else {"evidence_complete": False, "claim_status": "PENDING"}
    expected_new = [e for e in IEEE33_EXPERIMENTS + MOTOR_EXPERIMENTS + EEG_EXPERIMENTS]
    rows = []
    for e in expected_new:
        p = root / (e.experiment_id.lower().replace("/", "_") + ".json")
        payload = _load_json(p, {}) if p.exists() else {}
        rows.append({"experiment_id": e.experiment_id, "exists": p.exists(), "execution_complete": bool(payload.get("execution_complete", False)), "claim_passed": bool(payload.get("claim_passed", False)), "path": str(p)})
    payload = {
        "schema": SCHEMA + "-FOUR-TASK-GATE", "execution_complete": True,
        "registry": registry, "case300": case300_gate, "new_experiments": rows,
        "new_evidence_complete": bool(rows and all(r["execution_complete"] for r in rows)),
        "claim_status": "READY_FOR_CLAIM_REVIEW" if rows and all(r["execution_complete"] for r in rows) else "PENDING_NEW_EXPERIMENTS",
        "no_placeholder_is_evidence": True,
    }
    if output_path is not None: _atomic_json(output_path, payload)
    return payload



def validate_three_program_registry(experiments: Sequence[ProgramExperiment] = FOUR_TASK_EXPERIMENTS) -> dict[str, Any]:
    """Deprecated v48 name retained for import compatibility; validates the v51 four-task registry."""
    return validate_four_task_registry(experiments)


def write_three_program_manifest(path: str | Path, experiments: Sequence[ProgramExperiment] = FOUR_TASK_EXPERIMENTS) -> dict[str, Any]:
    """Deprecated v48 name retained for import compatibility."""
    return write_four_task_manifest(path, experiments)


def three_program_gate(evidence_root: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    """Deprecated v48 name retained for import compatibility."""
    return four_task_gate(evidence_root, output_path=output_path)

__all__ = sorted(set(__all__ + [
    "ProgramExperiment", "ApplicationResultSchema", "PROGRAM_CASE300", "PROGRAM_IEEE33", "PROGRAM_MOTOR", "PROGRAM_EEG",
    "validate_application_result_payload", "application_fail_closed_gate", "ieee33_result_schemas", "motor_result_schemas",
    "CASE300_EXPERIMENTS", "IEEE33_EXPERIMENTS", "MOTOR_EXPERIMENTS", "EEG_EXPERIMENTS",
    "FOUR_TASK_EXPERIMENTS", "THREE_PROGRAM_EXPERIMENTS",
    "validate_four_task_registry", "write_four_task_manifest", "four_task_gate",
    "validate_three_program_registry", "write_three_program_manifest", "three_program_gate",
    "pairwise_binary_features", "validate_qubo_bridge", "fit_pairwise_qubo_bridge",
    "compare_wrap_resilience", "periodic_signal_audit", "few_fem_cost_to_target",
    "conditional_result_placeholders",
]))
