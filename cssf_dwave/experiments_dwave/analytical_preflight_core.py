"""CPU-only analytical preflight for CSSF(QA) SQA experiments.

The preflight never invokes an annealer.  It validates the experiment before
expensive simulation by checking schedule admissibility, operator-action and
raw-control information geometry, BQM/embedding consistency, D0-D3 causal
contrast completeness, and read-budget resolution.  The result is a forecast
and experiment-validity certificate, never a substitute for SQA/QPU evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from experiments_dwave.benchmark_protocol import MatchedControlProtocol
from experiments_dwave.control_design import admissible_latin_hypercube, assert_admissible_forward_design
from experiments_dwave.integrated_bess_v38 import raw_control_coordinates
from experiments_dwave.operator_phase import APPROVED_FAMILIES, load_calibration, operator_action_coordinates
from benchmarks import reference_competitors as rc
from spectral.feature_matrix import toric_feature_matrix
from spectral.frequency_support import FrequencySupport, SupportKind, signed_axis_support, pairwise_support


SCHEMA = "CSSF-QA-ANALYTICAL-PREFLIGHT-v55"
DEFAULT_SQA_EXPERIMENTS = (
    "D0", "D1", "D2", "D3",
    "CSSF-full", "GP+EI-full", "Finzgar-BO-matched-full", "TuRBO-matched-full",
    "operator-action-family-check", "residual-hierarchy", "production-sampling",
    "director-corpus", "D45-04", "D45-08", "D45-10",
)


class AnalyticalPreflightError(RuntimeError):
    """Raised when an experiment is not analytically cleared for annealing."""


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


def _design_metrics(theta: ArrayLike) -> dict[str, Any]:
    x = np.asarray(theta, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise AnalyticalPreflightError("operator/control coordinate matrix must be two-dimensional")
    support = signed_axis_support(x.shape[1], max_harmonic=1, include_zero=True)
    phi = toric_feature_matrix(x, support, wrap_coordinates=True)
    sv = np.linalg.svd(phi, compute_uv=False)
    tol = float(sv[0] * max(phi.shape) * np.finfo(np.float64).eps)
    rank = int(np.sum(sv > tol))
    gram = (phi.conj().T @ phi) / float(phi.shape[0])
    ev = np.linalg.eigvalsh(gram)
    norms = np.sqrt(np.maximum(np.real(np.diag(phi.conj().T @ phi)), 0.0))
    denom = norms[:, None] * norms[None, :]
    corr = np.divide(
        np.abs(phi.conj().T @ phi), denom,
        out=np.zeros_like(denom, dtype=float), where=denom > 0,
    )
    np.fill_diagonal(corr, 0.0)
    cond_phi = float(np.inf if sv[-1] == 0 else sv[0] / sv[-1])
    cond_gram = float(np.inf if ev[0] <= 0 else ev[-1] / ev[0])
    eps_amp = float(cond_phi * np.finfo(np.float64).eps)
    return {
        "rows": int(phi.shape[0]),
        "features": int(phi.shape[1]),
        "rank": rank,
        "full_column_rank": bool(rank == phi.shape[1]),
        "features_le_half_train": bool(phi.shape[1] <= x.shape[0] // 2),
        "sigma_min": float(sv[-1]),
        "sigma_max": float(sv[0]),
        "condition_phi_2": cond_phi,
        "condition_gram_2": cond_gram,
        "condition_times_machine_epsilon": eps_amp,
        "gram_eigen_min": float(ev[0]),
        "gram_eigen_max": float(ev[-1]),
        "max_column_coherence": float(np.max(corr)),
        "coordinate_span_min": float(np.min(np.ptp(x, axis=0))),
        "coordinate_span_median": float(np.median(np.ptp(x, axis=0))),
        "coordinate_span_max": float(np.max(np.ptp(x, axis=0))),
    }



def _total_l1_2_support(d: int) -> FrequencySupport:
    rows = [tuple([0] * d)]
    for j in range(d):
        for value in (-2, -1, 1, 2):
            row = [0] * d; row[j] = value; rows.append(tuple(row))
    for i in range(d):
        for j in range(i + 1, d):
            for a in (-1, 1):
                for b in (-1, 1):
                    row = [0] * d; row[i] = a; row[j] = b; rows.append(tuple(row))
    rows = sorted(set(rows), key=lambda row: (sum(abs(v) for v in row), row))
    return FrequencySupport(
        np.asarray(rows, dtype=int), kind=SupportKind.TOTAL_L1,
        include_zero=True, require_conjugate_symmetry=True,
    )


def _design_metrics_for_support(theta: ArrayLike, support: FrequencySupport) -> dict[str, Any]:
    x = np.asarray(theta, dtype=np.float64)
    phi = toric_feature_matrix(x, support, wrap_coordinates=True)
    sv = np.linalg.svd(phi, compute_uv=False)
    tol = float(sv[0] * max(phi.shape) * np.finfo(np.float64).eps)
    rank = int(np.sum(sv > tol))
    gram = (phi.conj().T @ phi) / float(phi.shape[0])
    ev = np.linalg.eigvalsh(gram)
    return {
        "rows": int(phi.shape[0]), "features": int(phi.shape[1]), "rank": rank,
        "full_column_rank": bool(rank == phi.shape[1]),
        "sigma_min": float(sv[-1]), "sigma_max": float(sv[0]),
        "condition_phi_2": float(np.inf if sv[-1] == 0 else sv[0] / sv[-1]),
        "condition_gram_2": float(np.inf if ev[0] <= 0 else ev[-1] / ev[0]),
    }


def _embedding_metrics(arm: Any, embedding: Mapping[Any, Sequence[Any]]) -> dict[str, Any]:
    variables = tuple(arm.problem.variable_order)
    keys = tuple(embedding.keys())
    missing = sorted(set(variables) - set(keys), key=str)
    extra = sorted(set(keys) - set(variables), key=str)
    chains = [tuple(embedding[v]) for v in variables if v in embedding]
    empty = [str(v) for v in variables if v in embedding and len(tuple(embedding[v])) == 0]
    flat = [q for chain in chains for q in chain]
    overlap = len(flat) - len(set(flat))
    lengths = [len(c) for c in chains]
    return {
        "variables": len(variables),
        "embedding_keys": len(keys),
        "missing_variables": missing,
        "extra_variables": extra,
        "empty_chains": empty,
        "physical_qubits_used": len(set(flat)),
        "overlap_count": int(overlap),
        "chain_length_min": int(min(lengths)) if lengths else 0,
        "chain_length_median": float(np.median(lengths)) if lengths else 0.0,
        "chain_length_max": int(max(lengths)) if lengths else 0,
        "passed": bool(not missing and not extra and not empty and overlap == 0),
    }


def _qubo_metrics(arm: Any) -> dict[str, Any]:
    model = arm.problem.model
    linear = np.asarray(model.linear, dtype=np.float64).reshape(-1)
    quadratic = np.asarray(model.quadratic, dtype=np.float64)
    quad_vals = quadratic[np.triu_indices(quadratic.shape[0], k=1)] if quadratic.size else np.asarray([], dtype=np.float64)
    nonzero = np.abs(quad_vals) > float(getattr(model, "zero_tolerance", 0.0))
    finite = bool(np.isfinite(linear).all() and np.isfinite(quadratic).all())
    return {
        "variables": int(len(model.variable_order)),
        "quadratic_terms": int(np.sum(nonzero)),
        "linear_abs_max": float(np.max(np.abs(linear))) if linear.size else 0.0,
        "quadratic_abs_max": float(np.max(np.abs(quad_vals))) if quad_vals.size else 0.0,
        "finite_coefficients": finite,
    }


def _factorial_contrast_metrics() -> dict[str, Any]:
    # columns: intercept, trig application representation, operator-action control, interaction
    X = np.asarray([
        [1.0, 0.0, 0.0, 0.0],  # D0
        [1.0, 1.0, 0.0, 0.0],  # D1
        [1.0, 0.0, 1.0, 0.0],  # D2
        [1.0, 1.0, 1.0, 1.0],  # D3
    ])
    return {
        "matrix": X.tolist(),
        "rank": int(np.linalg.matrix_rank(X)),
        "parameters": 4,
        "full_rank": bool(np.linalg.matrix_rank(X) == 4),
        "estimands": ["intercept", "application_trig_effect", "operator_action_effect", "interaction"],
    }


def _sampling_resolution(protocol: MatchedControlProtocol) -> dict[str, Any]:
    R = int(protocol.reads_per_control)
    n = int(protocol.total_control_budget)
    p95_one_hit = float(1.0 - 0.05 ** (1.0 / R))
    return {
        "reads_per_control": R,
        "controls_per_arm": n,
        "total_reads_per_fixed_read_arm": int(R * n),
        "worst_case_bernoulli_se_one_control": float(math.sqrt(0.25 / R)),
        "zero_hits_upper95_probability": p95_one_hit,
        "minimum_event_probability_for_95pct_chance_of_at_least_one_hit": p95_one_hit,
        "worst_case_95pct_halfwidth_arm_mean_difference": float(1.96 * math.sqrt(2 * 0.25 / (R * n))),
        "event_hit_probability_one_control": {
            str(p): float(1.0 - (1.0 - p) ** R) for p in (0.001, 0.002, 0.005, 0.01, 0.02)
        },
    }


def _forecast(geometry: Mapping[str, Any], sampling: Mapping[str, Any]) -> dict[str, Any]:
    full_rank = bool(geometry.get("full_column_rank", False))
    half_rule = bool(geometry.get("features_le_half_train", False))
    cond = float(geometry.get("condition_phi_2", np.inf))
    span = float(geometry.get("coordinate_span_min", 0.0))
    if not full_rank or not half_rule or not np.isfinite(cond) or span <= 0:
        level = "LOW"
        reason = "design is underidentified, degenerate, or non-finite"
    elif cond <= 30:
        level = "HIGH"
        reason = "full-rank first-order design with strong numerical separation"
    elif cond <= 300:
        level = "MEDIUM_HIGH"
        reason = "full-rank design with usable but nontrivial conditioning"
    elif cond <= 3000:
        level = "MEDIUM"
        reason = "identifiable design with elevated conditioning sensitivity"
    else:
        level = "LOW_MEDIUM"
        reason = "identifiable on rank but vulnerable to noise/conditioning"
    return {
        "cssf_response_identifiability_forecast": level,
        "basis": reason,
        "not_a_gpu_outcome_prediction": True,
        "rare_event_resolution_note": (
            "Events substantially rarer than the reported one-hit threshold may remain unseen at one control; "
            "absence of hits must not be interpreted as zero probability."
        ),
        "one_hit_95_probability_threshold": float(sampling["minimum_event_probability_for_95pct_chance_of_at_least_one_hit"]),
    }


@dataclass(frozen=True)
class AnalyticalPreflight:
    payload: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.payload.get("overall_pass", False))

    def require(self, experiment_id: str) -> None:
        experiment_id = str(experiment_id)
        experiments = self.payload.get("experiments", {})
        if experiment_id not in experiments:
            raise AnalyticalPreflightError(f"no analytical preflight record for experiment {experiment_id!r}")
        record = experiments[experiment_id]
        if not bool(record.get("pass", False)):
            raise AnalyticalPreflightError(
                f"analytical preflight blocks {experiment_id}: {record.get('reasons', ['unspecified failure'])}"
            )

    def summary(self) -> dict[str, Any]:
        exp = self.payload.get("experiments", {})
        return {
            "schema": self.payload.get("schema"),
            "overall_pass": self.passed,
            "annealer_calls": self.payload.get("annealer_calls"),
            "experiments_cleared": sum(bool(v.get("pass")) for v in exp.values()),
            "experiments_total": len(exp),
            "operator_forecast": self.payload.get("operator_action", {}).get("forecast", {}),
        }


def run_analytical_preflight(
    project_root: str | Path,
    *,
    protocol: MatchedControlProtocol,
    raw_arm: Any,
    trig_arm: Any,
    source_embedding: Mapping[Any, Sequence[Any]] | None = None,
    calibration_family: str = "Advantage_system6",
    comparison_family: str = "Advantage_system4",
    experiment_ids: Sequence[str] = DEFAULT_SQA_EXPERIMENTS,
    output_path: str | Path | None = None,
) -> AnalyticalPreflight:
    """Run one complete zero-annealer-call preflight for the planned SQA program."""
    root = Path(project_root)
    design = admissible_latin_hypercube(
        protocol.bounds, protocol.cssf_initial_count,
        order=protocol.order, seed=protocol.seed,
    )
    assert_admissible_forward_design(design, protocol.bounds, order=protocol.order)
    train_controls = design[: protocol.cssf_initial_train]

    raw_theta = np.vstack([raw_control_coordinates(c, protocol.bounds) for c in train_controls])
    raw_geometry = _design_metrics(raw_theta)

    family_geometry: dict[str, Any] = {}
    for family in (calibration_family, comparison_family):
        if family not in APPROVED_FAMILIES:
            raise AnalyticalPreflightError(f"unsupported calibration family {family!r}")
        filename, _ = APPROVED_FAMILIES[family]
        curve = load_calibration(root / "calibration" / filename, family)
        theta = []
        for c in train_controls:
            t, s = rc.fourier_forward_schedule(c, order=protocol.order, grid_points=129, reject_nonmonotone=True)
            theta.append(operator_action_coordinates(t, s, curve, n_segments=8))
        geom = _design_metrics(np.asarray(theta, dtype=np.float64))
        family_geometry[family] = geom

    primary_geometry = family_geometry[calibration_family]

    # Protected director design: evaluate the information geometry of the richer
    # dictionaries before any protected-corpus SQA sample is collected.
    try:
        from experiments_dwave.director_matrix_v53 import director_validation_design, DEFAULT_DIRECTOR_PLAN
        director_rows = director_validation_design(protocol, DEFAULT_DIRECTOR_PLAN)
        director_controls = np.asarray([r["control"] for r in director_rows if r["partition"] == "train"], dtype=np.float64)
        assert_admissible_forward_design(director_controls, protocol.bounds, order=protocol.order)
        filename, _ = APPROVED_FAMILIES[calibration_family]
        curve = load_calibration(root / "calibration" / filename, calibration_family)
        director_theta = []
        for c in director_controls:
            t, ss = rc.fourier_forward_schedule(c, order=protocol.order, grid_points=129, reject_nonmonotone=True)
            director_theta.append(operator_action_coordinates(t, ss, curve, n_segments=8))
        director_theta = np.asarray(director_theta, dtype=np.float64)
        d = director_theta.shape[1]
        director_geometry = {
            "train_controls": int(director_theta.shape[0]),
            "signed_axes_1": _design_metrics_for_support(director_theta, signed_axis_support(d, max_harmonic=1, include_zero=True)),
            "pairwise_1": _design_metrics_for_support(director_theta, pairwise_support(d, include_axes=True, include_sums=True, include_differences=True, include_zero=True)),
            "total_l1_2": _design_metrics_for_support(director_theta, _total_l1_2_support(d)),
        }
    except Exception as exc:
        director_geometry = {"error": f"{type(exc).__name__}: {exc}"}

    sampling = _sampling_resolution(protocol)
    factorial = _factorial_contrast_metrics()
    raw_qubo = _qubo_metrics(raw_arm)
    trig_qubo = _qubo_metrics(trig_arm)
    embedding = None if source_embedding is None else _embedding_metrics(trig_arm, source_embedding)
    same_cardinality = raw_qubo["variables"] == trig_qubo["variables"]
    same_rank_order = len(tuple(raw_arm.problem.variable_order)) == len(tuple(trig_arm.problem.variable_order))

    base_reasons = []
    if not factorial["full_rank"]:
        base_reasons.append("D0-D3 factorial contrast matrix is rank deficient")
    if not same_cardinality or not same_rank_order:
        base_reasons.append("raw/trig QUBO candidate cardinality mismatch")
    if not raw_qubo["finite_coefficients"] or not trig_qubo["finite_coefficients"]:
        base_reasons.append("QUBO contains non-finite coefficients")
    if embedding is not None and not embedding["passed"]:
        base_reasons.append("embedding is incomplete, overlapping, or contains empty chains")
    if not raw_geometry["full_column_rank"] or not raw_geometry["features_le_half_train"]:
        base_reasons.append("raw-control first-order CSSF design is underidentified")
    for fam, geom in family_geometry.items():
        if not geom["full_column_rank"] or not geom["features_le_half_train"]:
            base_reasons.append(f"{fam} operator-action CSSF design is underidentified")
        if geom["coordinate_span_min"] <= 0:
            base_reasons.append(f"{fam} contains a degenerate operator-action coordinate")
    if "error" in director_geometry:
        base_reasons.append("director protected-design geometry could not be evaluated")
    else:
        for name in ("signed_axes_1", "pairwise_1", "total_l1_2"):
            if not director_geometry[name]["full_column_rank"]:
                base_reasons.append(f"director dictionary {name} is rank deficient")
    base_pass = not base_reasons

    experiments = {}
    for experiment_id in map(str, experiment_ids):
        reasons = list(base_reasons)
        # Every SQA experiment shares the admissible schedule domain and runtime physics.
        # D0/D1 additionally rely on raw-control identifiability; D2/D3 and all schedule-control
        # experiments rely on operator-action identifiability.
        if experiment_id in {"D0", "D1"} and not raw_geometry["full_column_rank"]:
            reasons.append("raw-control design lacks full rank")
        if experiment_id not in {"D0", "D1"} and not primary_geometry["full_column_rank"]:
            reasons.append("operator-action design lacks full rank")
        experiments[experiment_id] = {
            "pass": bool(not reasons),
            "reasons": reasons,
            "forecast": _forecast(raw_geometry if experiment_id in {"D0", "D1"} else primary_geometry, sampling),
            "gpu_or_qpu_calls_used_by_preflight": 0,
        }

    payload = {
        "schema": SCHEMA,
        "annealer_calls": 0,
        "purpose": "forecast-and-validity-gate-not-substitute-for-SQA",
        "protocol": asdict(protocol),
        "schedule_design": {
            "construction": "admissible_latin_hypercube",
            "initial_controls": int(design.shape[0]),
            "all_forward_schedule_admissible": True,
            "sha256": hashlib.sha256(np.ascontiguousarray(design).tobytes(order="C")).hexdigest(),
        },
        "raw_control": {"geometry": raw_geometry, "forecast": _forecast(raw_geometry, sampling)},
        "operator_action": {
            "primary_family": calibration_family,
            "families": family_geometry,
            "forecast": _forecast(primary_geometry, sampling),
        },
        "sampling_resolution": sampling,
        "director_design_geometry": director_geometry,
        "factorial_causal_completeness": factorial,
        "qubo": {"raw": raw_qubo, "trig": trig_qubo, "same_candidate_cardinality": same_cardinality},
        "embedding": embedding,
        "embedding_preflight_status": "checked" if embedding is not None else "deferred_to_runtime_topology_gate",
        "experiments": experiments,
        "overall_pass": bool(base_pass and all(v["pass"] for v in experiments.values())),
        "interpretation": {
            "can_forecast": [
                "schedule executability", "spectral identifiability", "conditioning", "control-coordinate degeneracy",
                "causal contrast completeness", "read-budget rare-event resolution", "embedding/QUBO consistency",
            ],
            "cannot_replace": [
                "SQA response distribution", "elite probability", "freeze-out behavior", "hardware noise", "QPU advantage",
            ],
        },
    }
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result = AnalyticalPreflight(payload=payload)
    if not result.passed:
        # Return the detailed object to callers that want to inspect it, but notebook
        # execution uses require() / overall gate to fail closed before annealing.
        pass
    return result


def require_experiment_preflight(preflight: AnalyticalPreflight, experiment_id: str) -> None:
    preflight.require(experiment_id)


__all__ = [
    "SCHEMA", "DEFAULT_SQA_EXPERIMENTS", "AnalyticalPreflightError", "AnalyticalPreflight",
    "run_analytical_preflight", "require_experiment_preflight",
]
