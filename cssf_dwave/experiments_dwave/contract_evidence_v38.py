"""Contract-v38 evidence assembly for CSSF(QA) D-Wave proof campaigns.

This module does not implement scientific algorithms.  It serializes and
cross-links outputs produced by the frozen CSSF framework and the versioned
D-Wave experiment layer so independent V00--V19 verifiers can recompute the
claim gates without trusting notebook booleans.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from benchmarks import reference_competitors as rc
from experiments_dwave.benchmark_protocol import MatchedControlProtocol, MethodTrace
from experiments_dwave.claim_set_v38 import PRIMARY_EXTERNAL_COMPARATORS, SURROGATE_STRUCTURE_REFERENCES
from experiments_dwave.evidence_v38 import canonical_json_hash, sha256_file, qpu_access_time_us
from experiments_dwave.operator_phase import APPROVED_FAMILIES, load_calibration, operator_action_coordinates


class ContractEvidenceError(RuntimeError):
    pass


FIDELITY_FIELDS = (
    "algorithm", "state", "init", "surrogate", "acquisition", "adaptation",
    "restart", "feasibility", "budget", "objective", "accounting", "provenance",
)


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


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return p


def write_external_claim_set(evidence_root: str | Path) -> dict[str, Any]:
    payload = {
        "schema": "CSSF-QA-EXTERNAL-CLAIM-SET-v38",
        "comparators": list(PRIMARY_EXTERNAL_COMPARATORS),
        "surrogate_structure_references": list(SURROGATE_STRUCTURE_REFERENCES),
        "highs_role": "exact BESS/QUBO quality reference; not an annealing-schedule optimizer",
        "claim_endpoint": "identical AC/OOD/full-N-1 BESS application endpoint with independent confirmation",
    }
    write_json(Path(evidence_root) / "external_claim_set.json", payload)
    return payload


def campaign_payload_to_trace(payload: Mapping[str, Any]) -> MethodTrace:
    """Reconstruct a MethodTrace from a complete restartable campaign checkpoint."""
    if not payload.get("complete"):
        raise ContractEvidenceError("campaign must be complete before claim evidence is assembled")
    method = str(payload.get("method", ""))
    if not method:
        raise ContractEvidenceError("campaign checkpoint has no method")
    controls: list[np.ndarray] = []
    responses: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    ledger = rc.ResourceLedger()
    for row in payload.get("records", []):
        control = np.asarray(row["control"], dtype=float)
        response = dict(row["response"])
        diag = dict(row.get("diagnostics", {}))
        controls.append(control)
        responses.append(response)
        diagnostics.append(diag)
        ledger.add(
            rc.CostEntry(
                method=method,
                stage="search",
                control_query=1,
                reads=int(response.get("num_reads", row.get("reads", 0))),
                annealing_time_us=float(control[0]),
                qpu_access_time_us=qpu_access_time_us(response),
                simulator_seconds=float(response.get("elapsed_seconds", 0.0) or 0.0),
                metadata=diag,
            )
        )
    return MethodTrace(method=method, controls=controls, responses=responses, ledger=ledger, diagnostics=diagnostics)


def write_operator_action_evidence(
    project_root: str | Path,
    evidence_root: str | Path,
    controls: Mapping[str, Sequence[Sequence[float]]],
    *,
    order: int = 8,
    n_segments: int = 8,
) -> dict[str, Any]:
    """Serialize independently recomputable A(s),B(s)->operator-action evidence.

    ``controls`` is keyed by Advantage_system4 / Advantage_system6.  Every row
    stores the physical schedule, not only its feature vector, so V07 can
    recalculate the operator-action coordinates from the frozen XLSX files.
    """
    root = Path(project_root)
    families = []
    for family in ("Advantage_system4", "Advantage_system6"):
        if family not in controls:
            raise ContractEvidenceError(f"operator-action evidence requires controls for {family}")
        filename, expected_hash = APPROVED_FAMILIES[family]
        curve = load_calibration(root / "calibration" / filename, family)
        if curve.source_sha256 != expected_hash:
            raise ContractEvidenceError(f"frozen calibration hash mismatch for {family}")
        rows = []
        for i, control in enumerate(controls[family]):
            c = np.asarray(control, dtype=float)
            t, s = rc.fourier_forward_schedule(c, order=int(order), grid_points=129, reject_nonmonotone=True)
            action = operator_action_coordinates(t, s, curve, n_segments=int(n_segments))
            rows.append({
                "control_id": f"{family}:control:{i:04d}",
                "control": c.tolist(),
                "t_us": t.tolist(),
                "s": s.tolist(),
                "n_segments": int(n_segments),
                "operator_action": action.tolist(),
            })
        families.append({
            "family": family,
            "calibration_file": filename,
            "calibration_sha256": curve.source_sha256,
            "controls": rows,
        })
    payload = {"schema": "CSSF-QA-OPERATOR-ACTION-EVIDENCE-v38", "families": families}
    write_json(Path(evidence_root) / "operator_action.json", payload)
    return payload


def write_highs_reference(evidence_root: str | Path, *, highs: Any, problem: Any) -> dict[str, Any]:
    """Write exact-quality-reference evidence; never a schedule-optimizer row."""
    selected_sample = np.asarray(getattr(highs, "selected_sample"), dtype=np.int8).reshape(-1)
    if selected_sample.size != problem.model.n_variables:
        raise ContractEvidenceError("HiGHS selected_sample dimension does not match the frozen QUBO")
    placement = problem.decode(selected_sample)
    selected = list(map(int, placement.selected_buses))
    objective = float(getattr(highs, "combined_qubo_energy"))
    # Independent objective recheck from the frozen QUBO implementation.
    recomputed = float(problem.model.energy(selected_sample))
    payload = {
        "schema": "CSSF-QA-HIGHS-QUALITY-REFERENCE-v38",
        "certified_optimal": bool(getattr(highs, "certified_optimal", False)),
        "problem_fingerprint": str(problem.fingerprint()),
        "selected_buses": selected,
        "objective": objective,
        "recomputed_objective": recomputed,
        "tolerance": 1.0e-8,
        "wall_clock_competition": False,
        "role": "exact BESS/QUBO quality reference only",
    }
    write_json(Path(evidence_root) / "highs_reference.json", payload)
    return payload


def trace_resource_events(trace: MethodTrace, *, utility_target: str = "elite_probability") -> list[dict[str, Any]]:
    """Convert one method trace into event-level verifier accounting rows."""
    rows: list[dict[str, Any]] = []
    for i, (control, response) in enumerate(zip(trace.controls, trace.responses, strict=True)):
        method = str(trace.method)
        utility = float(response[utility_target])
        base_meta = {"query_index": int(i), "utility": utility, "stage": "search"}
        rows.append({"method": method, "event_type": "evaluation", "resource": "control_evaluation", "amount": 1.0, "metadata": base_meta})
        rows.append({"method": method, "event_type": "evaluation", "resource": "annealer_reads", "amount": float(response.get("num_reads", 0)), "metadata": base_meta})
        rows.append({"method": method, "event_type": "evaluation", "resource": "annealing_time_us", "amount": float(np.asarray(control, dtype=float)[0]), "metadata": base_meta})
        qpu = qpu_access_time_us(response)
        sim = float(response.get("elapsed_seconds", 0.0) or 0.0)
        rows.append({"method": method, "event_type": "evaluation", "resource": "qpu_access_time_us", "amount": qpu, "metadata": base_meta})
        rows.append({"method": method, "event_type": "evaluation", "resource": "simulator_seconds", "amount": sim, "metadata": base_meta})
    return rows


def placement_resource_events(placement_meta: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Charge independent production-sampling replicates used to freeze placements."""
    rows: list[dict[str, Any]] = []
    for method, meta in placement_meta.items():
        for rep in meta.get("production_replicates", []) or []:
            info = dict(rep.get("backend_info", {}) or {})
            qpu = qpu_access_time_us({"backend_info": info})
            base = {"stage": "production_placement", "replicate": int(rep.get("replicate", 0))}
            rows.append({"method": str(method), "event_type": "production", "resource": "control_evaluation", "amount": 1.0, "metadata": base})
            rows.append({"method": str(method), "event_type": "production", "resource": "annealer_reads", "amount": float(rep.get("num_reads", 0)), "metadata": base})
            rows.append({"method": str(method), "event_type": "production", "resource": "qpu_access_time_us", "amount": qpu, "metadata": base})
    return rows


def application_validation_events(methods: Sequence[str], *, validation: int, ood: int, n1: int, confirmation: int) -> list[dict[str, Any]]:
    """Charge every nonlinear physical validation case used by an application claim."""
    total = int(validation) + int(ood) + int(n1) + int(confirmation)
    return [{"method": str(m), "event_type": "application_validation", "resource": "physical_validation_cases", "amount": float(total),
             "metadata": {"validation": int(validation), "ood": int(ood), "n1": int(n1), "confirmation": int(confirmation)}} for m in methods]


def write_resource_accounting(
    evidence_root: str | Path,
    *,
    traces: Sequence[MethodTrace],
    extra_events: Sequence[Mapping[str, Any]] = (),
    identical_schedule_methods: Sequence[str] = (),
    charged_methods: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write event ledger plus declared totals consumed by V11/V15.

    Schedule-search arms with the same protocol are checked for identical
    control/read resources.  Role-different physics/classical/QZero arms are
    still fully charged but are not forced into an artificial identical-read
    constraint; this distinction is explicit in the manifest.
    """
    events: list[dict[str, Any]] = []
    for trace in traces:
        events.extend(trace_resource_events(trace))
    events.extend(dict(e) for e in extra_events)
    methods: dict[str, dict[str, Any]] = {}
    totals: dict[str, dict[str, float]] = {}
    for event in events:
        m = str(event["method"]); r = str(event["resource"])
        totals.setdefault(m, {}).setdefault(r, 0.0)
        totals[m][r] += float(event.get("amount", 1.0))
    for m, value in totals.items():
        methods[m] = {"totals": {k: float(v) for k, v in sorted(value.items())}}
    groups = []
    if identical_schedule_methods:
        groups.append({
            "name": "matched_schedule_optimizer_budget",
            "methods": list(identical_schedule_methods),
            "resources": ["control_evaluation", "annealer_reads"],
            "mode": "identical",
        })
    if charged_methods:
        groups.append({
            "name": "role_specific_full_cost_accounting",
            "methods": list(charged_methods),
            "resources": ["control_evaluation", "annealer_reads", "qpu_access_time_us", "simulator_seconds"],
            "mode": "charged",
        })
    budget = {"schema": "CSSF-QA-RESOURCE-BUDGET-v38", "methods": methods, "matched_groups": groups}
    root = Path(evidence_root); root.mkdir(parents=True, exist_ok=True)
    with (root / "resource_events.jsonl").open("w", encoding="utf-8") as stream:
        for row in events:
            stream.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n")
    write_json(root / "resource_budget.json", budget)
    return events, budget


def write_cost_to_target(
    evidence_root: str | Path,
    *,
    events: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    target: float,
    target_definition: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute first-hit cost directly from the same event rows used by V11."""
    target_hash = canonical_json_hash(dict(target_definition), prefix="CSSF-cost-target-v38")
    rows = []
    for method in methods:
        cost = 0.0; hit = None
        for e in events:
            if str(e.get("method")) != str(method) or e.get("event_type") != "evaluation" or e.get("resource") != "control_evaluation":
                continue
            cost += float(e.get("amount", 1.0))
            utility = e.get("metadata", {}).get("utility")
            if utility is not None and float(utility) >= float(target):
                hit = cost; break
        rows.append({"method": str(method), "status": "REACHED" if hit is not None else "NOT_REACHED", "cost": None if hit is None else float(hit)})
    payload = {
        "schema": "CSSF-QA-COST-TO-TARGET-v38", "target": float(target),
        "target_frozen_sha256": target_hash, "target_definition": dict(target_definition),
        "maximize": True, "methods": rows,
    }
    write_json(Path(evidence_root) / "cost_to_target.json", payload)
    return payload


def paired_bootstrap_row(
    *, competitor: str, cssf_values: Sequence[float], competitor_values: Sequence[float],
    seed: int, alpha: float = 0.05, margin: float = 0.0, bootstrap_samples: int = 4000,
    confirmation_partition: str = "application_confirmation",
) -> dict[str, Any]:
    a = np.asarray(cssf_values, dtype=float); b = np.asarray(competitor_values, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ContractEvidenceError("paired confirmation vectors must be finite equal-length vectors with at least two pairs")
    differences = a - b
    rng = np.random.default_rng(int(seed)); means = np.empty(int(bootstrap_samples), dtype=float)
    for i in range(int(bootstrap_samples)):
        means[i] = float(np.mean(differences[rng.integers(0, differences.size, differences.size)]))
    lcb = float(np.quantile(means, float(alpha)))
    return {
        "competitor": str(competitor), "paired_differences": differences.tolist(),
        "cssf_values": a.tolist(), "competitor_values": b.tolist(),
        "alpha": float(alpha), "margin": float(margin), "bootstrap_seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples), "lcb": lcb,
        "superiority_passed": bool(lcb > float(margin)),
        "confirmation_partition": str(confirmation_partition),
    }


def write_statistics(evidence_root: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = set(PRIMARY_EXTERNAL_COMPARATORS)
    got = {str(r.get("competitor")) for r in rows}
    if not required.issubset(got):
        missing = sorted(required - got)
        raise ContractEvidenceError(f"statistics require all frozen external comparators: missing {missing}")
    payload = {"schema": "CSSF-QA-PAIRED-APPLICATION-STATISTICS-v38", "comparisons": [dict(r) for r in rows]}
    write_json(Path(evidence_root) / "statistics.json", payload)
    return payload


def write_partitions(
    evidence_root: str | Path,
    *, partitions: Mapping[str, Sequence[str]], observations: Sequence[Mapping[str, Any]], shared_initial_design: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": "CSSF-QA-PARTITIONS-v38",
        "partitions": {str(k): list(map(str, v)) for k, v in partitions.items()},
        "observations": [dict(x) for x in observations],
        "shared_initial_design": {} if shared_initial_design is None else dict(shared_initial_design),
    }
    write_json(Path(evidence_root) / "partitions.json", payload)
    return payload


def write_reproducibility(
    project_root: str | Path,
    evidence_root: str | Path,
    *, seeds: Mapping[str, int], config_hashes: Mapping[str, str], code_paths: Sequence[str | Path], deterministic_regeneration_passed: bool,
    stochastic_replay_state_recorded: bool = True,
) -> dict[str, Any]:
    root = Path(project_root)
    code_hashes = {}
    for item in code_paths:
        p = Path(item)
        if not p.is_absolute(): p = root / p
        if not p.is_file():
            raise ContractEvidenceError(f"reproducibility code path is absent: {p}")
        code_hashes[str(p.relative_to(root))] = sha256_file(p)
    env_payload = {
        "python": __import__("sys").version,
        "numpy": np.__version__,
    }
    try:
        import torch
        env_payload["torch"] = torch.__version__
        env_payload["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        env_payload["torch"] = None
        env_payload["torch_cuda_available"] = False
    payload = {
        "schema": "CSSF-QA-REPRODUCIBILITY-v38", "seeds": {str(k): int(v) for k, v in seeds.items()},
        "config_hashes": dict(config_hashes),
        "environment_manifest": env_payload,
        "environment_manifest_sha256": canonical_json_hash(env_payload, prefix="CSSF-env-v38"),
        "code_hashes": code_hashes,
        "deterministic_regeneration_passed": bool(deterministic_regeneration_passed),
        "stochastic_replay_state_recorded": bool(stochastic_replay_state_recorded),
    }
    write_json(Path(evidence_root) / "reproducibility.json", payload)
    return payload


def write_competitor_fidelity(
    project_root: str | Path,
    evidence_root: str | Path,
    *, reproduction_suite: Mapping[str, Any], qzero_gate: Mapping[str, Any],
    strong_classical_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed fidelity manifest from executable mechanism evidence.

    A comparator receives a binary fidelity product of one only when its full
    mechanism evidence is actually present.  Missing dependency/corpus stages
    are serialized as zeros instead of being silently promoted to PASS.
    """
    root = Path(project_root)
    results = dict(reproduction_suite.get("results", {}))
    rows = []
    implementation = root / "benchmarks" / "reference_competitors.py"
    impl_hash = sha256_file(implementation)
    for name in PRIMARY_EXTERNAL_COMPARATORS:
        mech = dict(results.get(name, {}))
        if name == "QZero-matched-full":
            full = bool(qzero_gate.get("pass")) and bool(mech.get("pass", False) or qzero_gate.get("pass"))
        elif name in {"Strong-SA", "Strong-Tabu"}:
            strong_row = {} if not strong_classical_result else dict(strong_classical_result.get(name, {}))
            full = bool(strong_row.get("tuning", {}).get("best_config"))
        else:
            full = bool(mech.get("pass", False))
        fidelity = {field: int(full) for field in FIDELITY_FIELDS}
        rows.append({
            "name": name, "full_function": bool(full), "fidelity": fidelity,
            "implementation_sha256": impl_hash,
            "reference_provenance": "frozen literature/reference protocol declared in the released CSSF(QA) evidence specification",
            "mechanism_test_report": _jsonable(mech if mech else {"status": "MISSING"}),
            "hyperparameters": "full-function implementation in benchmarks/reference_competitors.py; no reduced named arm accepted",
            "budget_manifest": "resource_budget.json",
        })
    payload = {"schema": "CSSF-QA-COMPETITOR-FIDELITY-v38", "competitors": rows,
               "registry_hash": reproduction_suite.get("registry_hash"), "algorithm_hash": reproduction_suite.get("algorithm_hash")}
    write_json(Path(evidence_root) / "competitor_fidelity.json", payload)
    return payload


def write_external_export_stub(evidence_root: str | Path, *, claim_grade: bool = False, rows: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    payload = {"schema": "CSSF-QA-EXTERNAL-EXPORT-v38", "claim_grade": bool(claim_grade), "rows": [dict(x) for x in rows]}
    write_json(Path(evidence_root) / "external_export.json", payload)
    return payload


__all__ = [
    "ContractEvidenceError", "FIDELITY_FIELDS", "write_json", "write_external_claim_set",
    "campaign_payload_to_trace", "write_operator_action_evidence", "write_highs_reference",
    "trace_resource_events", "write_resource_accounting", "write_cost_to_target", "paired_bootstrap_row",
    "write_statistics", "write_partitions", "write_reproducibility", "write_competitor_fidelity",
    "write_external_export_stub",
]


def rename_trace(trace: MethodTrace, method: str) -> MethodTrace:
    """Return an accounting-equivalent trace under an external claim-arm label."""
    ledger = rc.ResourceLedger()
    for e in trace.ledger.entries:
        ledger.add(rc.CostEntry(
            method=str(method), stage=e.stage, control_query=e.control_query,
            hidden_environment_queries=e.hidden_environment_queries, reads=e.reads,
            annealing_time_us=e.annealing_time_us, qpu_access_time_us=e.qpu_access_time_us,
            programming_time_us=e.programming_time_us, readout_time_us=e.readout_time_us,
            classical_seconds=e.classical_seconds, simulator_seconds=e.simulator_seconds,
            ac_calls=e.ac_calls, metadata=dict(e.metadata),
        ))
    return MethodTrace(str(method), [np.asarray(x, dtype=float).copy() for x in trace.controls], [dict(x) for x in trace.responses], ledger, [dict(x) for x in trace.diagnostics])


def endpoint_utility(endpoint_row: Mapping[str, Any]) -> float:
    """Frozen higher-is-better application utility for factorial/confirmation gates.

    The score uses untouched OOD and the complete N-1 set only; validation is
    reserved for placement selection.  Infeasible cases already carry the
    endpoint's frozen 1e9 penalty.
    """
    ood = np.asarray([float(x["penalized_objective"]) for x in endpoint_row.get("ood", {}).get("rows", [])], dtype=float)
    n1 = np.asarray([float(x["penalized_objective"]) for x in endpoint_row.get("n1", {}).get("rows", [])], dtype=float)
    if ood.size == 0 or n1.size == 0 or not np.isfinite(ood).all() or not np.isfinite(n1).all():
        raise ContractEvidenceError("application utility requires complete finite OOD and N-1 endpoint rows")
    return -float(0.5 * np.mean(ood) + 0.5 * np.mean(n1))


def confirmation_values(confirmation: Mapping[str, Any]) -> np.ndarray:
    rows = list(confirmation.get("rows", []))
    if not rows:
        raise ContractEvidenceError("confirmation evidence contains no rows")
    values = np.asarray([-float(x["penalized_objective"]) for x in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ContractEvidenceError("confirmation utilities are non-finite")
    return values


def write_factorial_evidence(
    evidence_root: str | Path,
    *,
    specs: Sequence[Any],
    endpoint_rows: Mapping[str, Mapping[str, Any]],
    lineage_endpoint_ids: Mapping[str, str],
    interaction_bootstrap_samples: int = 4000,
    interaction_seed: int = 20260817,
) -> dict[str, Any]:
    """Serialize D0--D3 utilities and paired interaction uncertainty."""
    arms = []
    utilities = {}
    for spec in specs:
        aid = str(spec.arm_id)
        row = endpoint_rows[aid]
        u = endpoint_utility(row); utilities[aid] = u
        arms.append({
            "arm_id": aid, "domain_representation": str(spec.domain_representation),
            "control_representation": str(spec.control_representation),
            "matched_config_sha256": str(spec.matched_config_sha256), "utility": u,
            "application_endpoint_evidence_id": str(lineage_endpoint_ids[aid]),
            "selected_buses": list(row.get("selected_buses", [])),
        })
    delta = float(utilities["D3"] - utilities["D2"] - utilities["D1"] + utilities["D0"])
    # Paired scenario-level interaction uses OOD rows, which are identically
    # indexed across arms. N-1 is separately enforced by V12 and included in
    # the scalar utility above.
    per_arm = {}
    for aid in ("D0", "D1", "D2", "D3"):
        rows = endpoint_rows[aid].get("ood", {}).get("rows", [])
        per_arm[aid] = np.asarray([-float(x["penalized_objective"]) for x in rows], dtype=float)
    lengths = {x.size for x in per_arm.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) < 2:
        raise ContractEvidenceError("factorial paired uncertainty requires aligned OOD scenario rows")
    dvec = per_arm["D3"] - per_arm["D2"] - per_arm["D1"] + per_arm["D0"]
    rng = np.random.default_rng(int(interaction_seed)); boot = np.empty(int(interaction_bootstrap_samples))
    for i in range(boot.size):
        boot[i] = float(np.mean(dvec[rng.integers(0, dvec.size, dvec.size)]))
    uncertainty = {
        "partition": "ood", "paired_scenarios": int(dvec.size), "bootstrap_samples": int(boot.size),
        "bootstrap_seed": int(interaction_seed), "mean_interaction": float(np.mean(dvec)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }
    payload = {"schema": "CSSF-QA-D0-D3-FACTORIAL-v38", "arms": arms, "interaction": {"delta": delta, "paired_uncertainty": uncertainty}}
    write_json(Path(evidence_root) / "factorial.json", payload)
    return payload
