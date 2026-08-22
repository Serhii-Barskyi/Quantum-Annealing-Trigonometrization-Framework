"""External D-Wave evidence-notebook orchestration for the full CSSF framework.

The notebooks expose one task at a time so expensive response measurements can
be checkpointed without altering the declared scientific budgets.  All CSSF
model fitting is delegated to the original framework; this module coordinates
only evidence collection, competitor execution, provenance, and claim gates.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np

from benchmarks import reference_competitors as rc
from experiments_dwave.benchmark_protocol import MatchedControlProtocol
from experiments_dwave.bess_evidence import build_case300_bess_assets, to_dimod_bqm
from experiments_dwave.competitor_fidelity import run_reference_reproduction_suite
from experiments_dwave.evidence_program import (
    PROGRAM_REVISION,
    atomic_json,
    build_response_evaluator,
    calibration_report,
    environment_report,
    framework_identity_report,
    run_level1_csnn_t_evidence,
)
from experiments_dwave.experiment_tasks import (
    advance_matched_campaign,
    campaign_status,
    collect_surrogate_response_step,
    evaluate_worldline_schedule,
    qzero_claim_gate,
    run_paired_confirmation,
    run_strong_classical_benchmark,
    run_surrogate_audit,
)


TASKS = (
    "PREFLIGHT",
    "LEVEL_I",
    "REFERENCE_FIDELITY",
    "FREEZE_CONTEXT",
    "SURROGATE_STEP",
    "SURROGATE_AUDIT",
    "SCHEDULE_STEP",
    "WORLDLINE",
    "STRONG_CLASSICAL",
    "QZERO_GATE",
    "STATUS",
    "CONFIRM",
    "ASSEMBLE",
)
MATCHED_METHODS = (
    "CSSF-full",
    "GP+EI-full",
    "Finzgar-BO-matched-full",
    "TuRBO-matched-full",
    "Random-search",
)
CLAIM_COMPETITORS = (
    "GP+EI-full",
    "Finzgar-BO-matched-full",
    "TuRBO-matched-full",
)


class NotebookProgramError(RuntimeError):
    pass


def _read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_method(method: str) -> str:
    return str(method).replace("/", "_").replace(" ", "_")


def output_layout(project_root: str | Path, mode: str, calibration_family: str, solver_id: str | None = None) -> dict[str, Path]:
    root = Path(project_root)
    mode = str(mode).lower()
    identity = f"{mode}_{calibration_family}"
    if mode == "qpu" and solver_id:
        identity += "_" + str(solver_id).replace(".", "_")
    results = root / "results" / "dwave_evidence_v38" / identity
    checkpoints = root / "checkpoints" / "dwave_evidence_v38" / identity
    results.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    return {
        "results": results,
        "checkpoints": checkpoints,
        "context": checkpoints / "pegasus_context.json",
        "embedding": checkpoints / "fixed_embedding.json",
        "surrogate_corpus": checkpoints / "qa_response_surrogate_corpus.json",
        "surrogate_audit": results / "qa_response_surrogate_audit.json",
        "campaigns": checkpoints / "matched_campaigns",
        "worldline": results / "worldline_susceptibility.json",
        "strong_classical": results / "strong_classical.json",
        "qzero_corpus": checkpoints / "qzero_full_pretraining_corpus.json",
        "reference_fidelity": results / "competitor_reference_fidelity.json",
        "level1": results / "level1_original_csnnt.json",
        "assembly": results / "evidence_assembly.json",
    }


def _context_expected(layout: Mapping[str, Path]) -> str | None:
    payload = _read(layout["context"], {}) or {}
    value = payload.get("topology_fingerprint")
    return None if value is None else str(value)


def _require_live_qpu(mode: str, task: str, live_qpu: bool) -> None:
    if str(mode).lower() == "qpu" and task in {
        "FREEZE_CONTEXT", "SURROGATE_STEP", "SCHEDULE_STEP", "WORLDLINE", "CONFIRM"
    } and not bool(live_qpu):
        raise NotebookProgramError(f"{task} requires LIVE_QPU=True for the Pegasus notebook")


def _backend_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": manifest["mode"],
        "solver_id": manifest.get("solver_id"),
        "topology_fingerprint": manifest["topology_fingerprint"],
        "embedding_fingerprint": manifest["embedding_fingerprint"],
        "calibration_family": manifest["calibration"]["family"],
        "calibration_sha256": manifest["calibration"]["sha256"],
    }


def _build_evaluator(
    project_root: Path,
    *,
    mode: str,
    calibration_family: str,
    solver_id: str | None,
    live_qpu: bool,
    layout: Mapping[str, Path],
):
    expected = _context_expected(layout)
    if str(mode).lower() == "qpu" and layout["context"].exists() and not expected:
        raise NotebookProgramError("Frozen Pegasus context has no topology fingerprint")
    evaluator, assets, bundle, manifest = build_response_evaluator(
        project_root,
        mode=mode,
        calibration_family=calibration_family,
        solver_id=solver_id,
        live_qpu=live_qpu,
        solve_highs=True,
        expected_topology_fingerprint=expected,
        embedding_manifest_path=layout["embedding"],
    )
    return evaluator, assets, bundle, manifest


def _freeze_context(layout: Mapping[str, Path], manifest: Mapping[str, Any]) -> dict[str, Any]:
    existing = _read(layout["context"], None)
    payload = {
        "schema": "CSSF-QA-PEGASUS-CONTEXT-v38",
        "mode": manifest["mode"],
        "solver_id": manifest.get("solver_id"),
        "topology_fingerprint": manifest["topology_fingerprint"],
        "embedding_fingerprint": manifest["embedding_fingerprint"],
        "chain_strength": manifest["chain_strength"],
        "calibration": manifest["calibration"],
        "bess": manifest["bess"],
        "protocol": manifest["protocol"],
    }
    if existing is not None:
        keys = ("mode", "solver_id", "topology_fingerprint", "embedding_fingerprint", "calibration")
        mismatch = [k for k in keys if existing.get(k) != payload.get(k)]
        if mismatch:
            raise NotebookProgramError(f"Frozen context mismatch: {mismatch}")
    atomic_json(layout["context"], payload)
    return payload


def _best_campaign_control(path: Path) -> tuple[np.ndarray, float]:
    payload = _read(path, {}) or {}
    records = list(payload.get("records", []))
    if not records:
        raise NotebookProgramError(f"Campaign has no observations: {path}")
    values = np.asarray([float(r["response"]["elite_probability"]) for r in records])
    idx = int(np.argmax(values))
    return np.asarray(records[idx]["control"], dtype=float), float(values[idx])


def _assemble(project_root: Path, layout: Mapping[str, Path], protocol: MatchedControlProtocol) -> dict[str, Any]:
    methods = list(MATCHED_METHODS)
    status = {m: campaign_status(campaign_root=layout["campaigns"], method=m) for m in methods}
    surrogate = _read(layout["surrogate_audit"], None)
    fidelity = _read(layout["reference_fidelity"], None)
    qzero = qzero_claim_gate(layout["qzero_corpus"])
    confirmations = {}
    for competitor in CLAIM_COMPETITORS:
        p = layout["results"] / f"confirmation_CSSF-full_vs_{_safe_method(competitor)}.json"
        if p.exists():
            confirmations[competitor] = _read(p, {})
    all_matched_complete = all(status[m].get("complete", False) for m in CLAIM_COMPETITORS + ("CSSF-full",))
    confirmation_pass = all(
        bool(confirmations.get(m, {}).get("claim_pass", False)) for m in CLAIM_COMPETITORS
    ) if all_matched_complete else False
    fidelity_results = {} if not fidelity else fidelity.get("results", {})
    named_fidelity_pass = all(bool(fidelity_results.get(m, {}).get("pass", False)) for m in (
        "GP+EI-full", "TuRBO-paper", "TuRBO-matched-full", "Periodic-GP", "Torus-Riemannian-Matern-GP", "Worldline-Susceptibility-full"
    )) and bool(fidelity_results.get("Finzgar-BO-paper", {}).get("pass", False)) and bool(qzero.get("pass", False))
    claim_ready = bool(all_matched_complete and confirmation_pass and named_fidelity_pass)
    result = {
        "schema": "CSSF-QA-EVIDENCE-ASSEMBLY-v38",
        "program_revision": PROGRAM_REVISION,
        "protocol": asdict(protocol),
        "campaign_status": status,
        "surrogate_audit_available": surrogate is not None,
        "competitor_fidelity": fidelity,
        "qzero_gate": qzero,
        "confirmations": confirmations,
        "claim_ready": claim_ready,
        "claim_status": "UNLOCKED" if claim_ready else "LOCKED",
        "claim_rule": "Commercial superiority remains locked unless all full-function fidelity, matched-campaign, and independent confirmation gates pass.",
    }
    atomic_json(layout["assembly"], result)
    return result


def run_task(
    project_root: str | Path,
    *,
    mode: str,
    task: str,
    calibration_family: str,
    method: str = "CSSF-full",
    control_index: int | None = None,
    solver_id: str | None = None,
    live_qpu: bool = False,
    confirmation_replicates: int = 12,
) -> dict[str, Any]:
    """Execute exactly one declared evidence task."""
    root = Path(project_root)
    mode = str(mode).lower()
    task = str(task).strip().upper()
    if mode not in {"simulator", "qpu"}:
        raise NotebookProgramError("mode must be simulator or qpu")
    if task not in TASKS:
        raise NotebookProgramError(f"Unknown task {task!r}; allowed={TASKS}")
    if calibration_family not in {"Advantage_system4", "Advantage_system6"}:
        raise NotebookProgramError("Only Advantage_system4 and Advantage_system6 are supported")
    if mode == "qpu":
        if not solver_id or not str(solver_id).startswith(("Advantage_system4.", "Advantage_system6.")):
            if task not in {"PREFLIGHT", "LEVEL_I", "REFERENCE_FIDELITY", "QZERO_GATE", "STATUS", "ASSEMBLE"}:
                raise NotebookProgramError("Pegasus execution requires an explicit Advantage_system4.* or Advantage_system6.* solver_id")
        if solver_id:
            expected_family = solver_id.split(".", 1)[0]
            if expected_family != calibration_family:
                raise NotebookProgramError("solver_id and calibration_family must belong to the same Pegasus family")
    _require_live_qpu(mode, task, live_qpu)

    layout = output_layout(root, mode, calibration_family, solver_id)
    protocol = MatchedControlProtocol()

    if task == "PREFLIGHT":
        result = {
            "schema": "CSSF-QA-DWAVE-PREFLIGHT-v38",
            "mode": mode,
            "program_revision": PROGRAM_REVISION,
            "framework": framework_identity_report(root),
            "calibration": calibration_report(root, calibration_family),
            "environment": environment_report(),
            "protocol": asdict(protocol),
            "competitor_registry_hash": rc.REGISTRY_HASH,
            "competitor_algorithm_hash": rc.ALGORITHM_HASH,
            "matched_methods": list(MATCHED_METHODS),
            "hardware_status": "NOT_RUN_HARDWARE",
        }
        atomic_json(layout["results"] / "preflight.json", result)
        return result

    if task == "LEVEL_I":
        result = run_level1_csnn_t_evidence(root)
        atomic_json(layout["level1"], result)
        return result

    if task == "REFERENCE_FIDELITY":
        result = run_reference_reproduction_suite()
        atomic_json(layout["reference_fidelity"], result)
        return result

    if task == "QZERO_GATE":
        return qzero_claim_gate(layout["qzero_corpus"])

    if task == "STATUS":
        return {
            "mode": mode,
            "context": _read(layout["context"], None),
            "surrogate_corpus": _read(layout["surrogate_corpus"], {"records": []}),
            "campaigns": {m: campaign_status(campaign_root=layout["campaigns"], method=m) for m in MATCHED_METHODS},
            "qzero_gate": qzero_claim_gate(layout["qzero_corpus"]),
        }

    if task == "SURROGATE_AUDIT":
        return run_surrogate_audit(
            corpus_path=layout["surrogate_corpus"], project_root=root,
            output_path=layout["surrogate_audit"], target="elite_probability"
        )

    if task == "STRONG_CLASSICAL":
        assets = build_case300_bess_assets(root, solve_highs=True)
        if assets.elite_energy_threshold is None:
            raise NotebookProgramError("Certified HiGHS threshold is required")
        return run_strong_classical_benchmark(
            to_dimod_bqm(assets.problem), elite_threshold=float(assets.elite_energy_threshold),
            output_path=layout["strong_classical"], num_reads=protocol.reads_per_control,
        )

    if task == "ASSEMBLE":
        return _assemble(root, layout, protocol)

    evaluator = assets = bundle = manifest = None
    try:
        evaluator, assets, bundle, manifest = _build_evaluator(
            root, mode=mode, calibration_family=calibration_family,
            solver_id=solver_id, live_qpu=live_qpu, layout=layout,
        )
        if task == "FREEZE_CONTEXT":
            return _freeze_context(layout, manifest)

        # A scientific response task always requires a frozen backend context.
        if not layout["context"].exists():
            _freeze_context(layout, manifest)
        backend_identity = _backend_identity(manifest)

        if task == "SURROGATE_STEP":
            return collect_surrogate_response_step(
                evaluator, protocol=protocol, output_path=layout["surrogate_corpus"], control_index=control_index
            )

        if task == "SCHEDULE_STEP":
            if method not in MATCHED_METHODS:
                raise NotebookProgramError(f"SCHEDULE_STEP requires one of {MATCHED_METHODS}")
            return advance_matched_campaign(
                evaluator, method=method, protocol=protocol, project_root=root,
                campaign_root=layout["campaigns"], backend_identity=backend_identity,
                problem_fingerprint=assets.problem.fingerprint(), candidate_pool_size=4096,
            )

        if task == "WORLDLINE":
            return evaluate_worldline_schedule(
                evaluator, assets, project_root=root, calibration_family=calibration_family,
                output_path=layout["worldline"], num_reads=protocol.reads_per_control,
            )

        if task == "CONFIRM":
            if method not in CLAIM_COMPETITORS:
                raise NotebookProgramError(f"CONFIRM requires a competitor in {CLAIM_COMPETITORS}")
            cssf_path = layout["campaigns"] / "CSSF-full.json"
            comp_path = layout["campaigns"] / f"{_safe_method(method)}.json"
            cssf_control, cssf_best = _best_campaign_control(cssf_path)
            comp_control, comp_best = _best_campaign_control(comp_path)
            out = layout["results"] / f"confirmation_CSSF-full_vs_{_safe_method(method)}.json"
            result = run_paired_confirmation(
                evaluator, cssf_control=cssf_control, competitor_control=comp_control,
                output_path=out, num_reads=protocol.reads_per_control,
                replicates=int(confirmation_replicates), seed=protocol.seed,
            )
            result["selection_evidence"] = {
                "cssf_campaign_best_elite_probability": cssf_best,
                "competitor_campaign_best_elite_probability": comp_best,
                "competitor": method,
            }
            atomic_json(out, result)
            return result

        raise NotebookProgramError(f"Task {task!r} reached no execution branch")
    finally:
        if bundle is not None:
            try:
                bundle.close()
            except Exception:
                pass


__all__ = [
    "TASKS", "MATCHED_METHODS", "CLAIM_COMPETITORS", "NotebookProgramError",
    "output_layout", "run_task",
]
