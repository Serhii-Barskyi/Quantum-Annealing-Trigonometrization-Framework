"""Comprehensive CPU-only Analytical Preflight for the v56 CSSF(QA) GPU program.

This module performs every pre-annealing check that can be evaluated without
GPU SQA or live-QPU samples.  It may PASS, EXPAND, REPAIR, or BLOCK a planned
experiment according to rules frozen before any annealer response is observed.
It never substitutes analytical predictions for GPU/QPU evidence.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import importlib.util
import json
import math
import re

import numpy as np

from experiments_dwave.analytical_preflight_core import (
    AnalyticalPreflight,
    AnalyticalPreflightError,
    _design_metrics,
    _design_metrics_for_support,
    _embedding_metrics,
    _factorial_contrast_metrics,
    _forecast,
    _jsonable,
    _qubo_metrics,
    _sampling_resolution,
    _total_l1_2_support,
    run_analytical_preflight as _run_v55,
)
from experiments_dwave.benchmark_protocol import MatchedControlProtocol
from experiments_dwave.control_design import admissible_latin_hypercube, assert_admissible_forward_design
from experiments_dwave.integrated_bess_v38 import raw_control_coordinates
from experiments_dwave.operator_phase import APPROVED_FAMILIES, load_calibration, operator_action_coordinates
from experiments_dwave.experiment_tasks import qzero_claim_gate
from experiments_dwave.director_matrix_v53 import (
    DEFAULT_DIRECTOR_PLAN,
    FOUR_TASK_EXPERIMENTS,
    IEEE33_EXPERIMENTS,
    MOTOR_EXPERIMENTS,
    EEG_EXPERIMENTS,
    director_validation_design,
    validate_four_task_registry,
    dwave_incumbent_schedules,
    _raw_trig_feasible_design,
)
from benchmarks import reference_competitors as rc
from spectral.frequency_support import signed_axis_support, pairwise_support

SCHEMA = "CSSF-QA-ANALYTICAL-PREFLIGHT-v56"
STATUS_PASS = "PASS"
STATUS_PASS_SCOPE = "PASS_WITH_SCOPE"
STATUS_EXPAND = "EXPAND"
STATUS_REPAIR = "REPAIR"
STATUS_BLOCK = "BLOCK"
STATUS_BLOCK_ASSET = "BLOCK_MISSING_ASSET"

# Every current notebook path that can invoke the GPU annealer directly or
# indirectly.  CPU-only/AC/classical cells are intentionally absent.
DEFAULT_SQA_EXPERIMENTS = (
    "D0", "D1", "D2", "D3",
    "raw-trig-shared-corpus",
    "operator-action-family-check",
    "residual-hierarchy",
    "production-sampling",
    "CSSF-full",
    "GP+EI-full",
    "Finzgar-BO-matched-full",
    "TuRBO-matched-full",
    "QZero-matched-full",
    "Worldline-Susceptibility-full",
    "director-corpus",
    "D45-04",
    "D45-08",
    "D45-10",
)

APPLICATION_ASSETS = {
    "ieee33": (
        "data/ieee33_mishra_reference.json",
        "data/ieee33_mishra_fault_scenarios.json",
        "data/ieee33_transport_network.json",
    ),
    "motor": (
        "data/motor_baseline_cad_manifest.json",
        "data/motor_design_space.json",
        "data/motor_fem_corpus.npz",
        "data/motor_bench_protocol.json",
    ),
    "eeg": (
        "data/eeg_antonova_manifest.json",
        "data/eeg_public_replication_manifest.json",
        "data/eeg_preprocessing_protocol.json",
        "data/eeg_microstate_maps_manifest.json",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _required_reads_for_hit_probability(p: float, target: float = 0.95) -> int:
    p = float(p); target = float(target)
    if not (0.0 < p < 1.0 and 0.0 < target < 1.0):
        raise ValueError("p and target must lie in (0,1)")
    return int(math.ceil(math.log(1.0 - target) / math.log(1.0 - p)))


def _sampling_power_table(read_counts: Sequence[int], p_values=(1e-3, 2e-3, 5e-3, 1e-2)) -> dict[str, Any]:
    out = {}
    for R in map(int, read_counts):
        out[str(R)] = {
            "worst_case_bernoulli_se": float(math.sqrt(0.25 / R)),
            "zero_hit_one_sided_95_upper": float(1.0 - 0.05 ** (1.0 / R)),
            "hit_probability": {str(p): float(1.0 - (1.0-p)**R) for p in p_values},
        }
    return out


def _qubo_deep_metrics(arm: Any) -> dict[str, Any]:
    p = arm.problem
    base = _qubo_metrics(arm)
    objective_bound = float(p.objective_bound)
    penalty = float(p.penalty_strength)
    required_floor = float(2.0 * objective_bound + p.model.zero_tolerance)
    audit = getattr(arm, "audit", None)
    return {
        **base,
        "units_to_place": int(p.fleet.units_to_place),
        "candidate_count": int(len(p.variable_order)),
        "variable_order_matches_model": tuple(p.variable_order) == tuple(p.model.variable_order),
        "variable_order_matches_encoding": tuple(p.variable_order) == tuple(p.encoding.variable_order),
        "penalty_strength": penalty,
        "objective_absolute_bound": objective_bound,
        "penalty_required_floor": required_floor,
        "penalty_dominance_pass": bool(penalty >= required_floor),
        "qubo_ising_audit_present": audit is not None,
        "qubo_ising_audit": _jsonable(audit) if audit is not None else None,
    }


def _requirements_lock(root: Path) -> dict[str, Any]:
    text = (root / "requirements-colab.txt").read_text(encoding="utf-8")
    pins = {}
    for package in ("dwave-ocean-sdk", "qiskit-aer-gpu", "qiskit", "numpy", "scipy"):
        m = re.search(rf"^{re.escape(package)}\s*==\s*([^\s#]+)", text, flags=re.M)
        pins[package] = None if m is None else m.group(1)
    return {
        "pins": pins,
        "ocean_9_4_locked": pins.get("dwave-ocean-sdk") == "9.4.0",
        "qiskit_aer_gpu_locked": pins.get("qiskit-aer-gpu") == "0.15.1",
        "cpu_fallback_prohibited_in_requirements": "No CPU fallback is permitted" in text,
    }


def _topology_capacity_preflight(root: Path, K: int) -> dict[str, Any]:
    # Exact ideal-Pegasus counts from the project topology contract.  The ideal
    # P_m native-clique capacity used as a necessary/sufficient structural
    # certificate for the defect-free simulator fabric is 12(m-1).
    m = 16
    nominal = 24*m*(m-1)
    fabric = nominal - 8*(m-1)
    clique_capacity = 12*(m-1)
    req = _requirements_lock(root)
    local_ocean_available = all(importlib.util.find_spec(name) is not None for name in ("dwave_networkx", "minorminer", "dimod"))
    return {
        "pegasus_m": m,
        "nominal_nodes": nominal,
        "programmable_fabric_nodes": fabric,
        "structurally_excluded_nominal_nodes": nominal-fabric,
        "ideal_native_clique_capacity": clique_capacity,
        "logical_variables": int(K),
        "structural_clique_capacity_pass": bool(int(K) <= clique_capacity),
        "target_ocean_lock_pass": bool(req["ocean_9_4_locked"]),
        "local_ocean_packages_available_for_exact_chain_map": bool(local_ocean_available),
        "exact_chain_map_status": (
            "LOCAL_EXACT_MAP_CHECK_AVAILABLE" if local_ocean_available
            else "NOT_EXECUTABLE_IN_ASSISTANT_CONTAINER; target notebook performs deterministic runtime embedding verification before first GPU sample"
        ),
        "scope": "simulator structural topology certificate; real-QPU working graph must be read from live sampler and cannot be known analytically",
    }


def _control_design_preflight(root: Path, protocol: MatchedControlProtocol, calibration_family: str) -> dict[str, Any]:
    bounds = protocol.bounds
    designs = {
        "cssf_initial": admissible_latin_hypercube(bounds, protocol.cssf_initial_count, order=protocol.order, seed=protocol.seed),
        "gp_initial": admissible_latin_hypercube(bounds, 10, order=protocol.order, seed=protocol.seed),
        "turbo_initial": admissible_latin_hypercube(bounds, 10, order=protocol.order, seed=protocol.seed),
        "random_search": admissible_latin_hypercube(bounds, protocol.total_control_budget, order=protocol.order, seed=protocol.seed+77),
        "raw_trig_shared": _raw_trig_feasible_design(protocol, 74),
    }
    linear = rc.canonical_control(protocol.order, float(np.mean(protocol.annealing_time_range_us)))
    finzgar = rc.FinzgarBO(bounds, seed=protocol.seed, total_paper_iterations=50)
    designs["finzgar_initial"] = np.vstack(finzgar.initial_design(linear, feasibility=lambda x: rc.feasible_control(x,bounds,order=protocol.order)))
    for X in designs.values():
        assert_admissible_forward_design(X, bounds, order=protocol.order)
    director_rows = director_validation_design(protocol, DEFAULT_DIRECTOR_PLAN)
    director = np.asarray([r["control"] for r in director_rows], dtype=float)
    assert_admissible_forward_design(director, bounds, order=protocol.order)
    designs["director_all"] = director

    filename, _ = APPROVED_FAMILIES[calibration_family]
    curve = load_calibration(root / "calibration" / filename, calibration_family)
    native = dwave_incumbent_schedules(curve, protocol)
    native_ok = True
    native_rows = []
    for row in native:
        t=np.asarray(row["t_us"],float); s=np.asarray(row["s"],float)
        ok=bool(t.ndim==s.ndim==1 and len(t)==len(s) and len(t)>=3 and abs(t[0])<=1e-12 and np.all(np.diff(t)>0) and abs(s[0])<=1e-12 and abs(s[-1]-1)<=1e-12 and np.all(np.diff(s)>=-1e-12) and np.all((s>=-1e-12)&(s<=1+1e-12)))
        native_ok &= ok; native_rows.append({"schedule_id":row["schedule_id"],"pass":ok,"points":len(t)})

    return {
        "all_generated_forward_admissible": True,
        "designs": {name:{"rows":int(X.shape[0]),"dimensions":int(X.shape[1]),"sha256":hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()} for name,X in designs.items()},
        "native_incumbent_schedules": native_rows,
        "native_incumbent_schedules_pass": bool(native_ok),
    }


def _worldline_preflight(arm: Any, calibration: Any) -> dict[str, Any]:
    """Static CPU design audit for the worldline comparator.

    The susceptibility Monte Carlo itself is an experiment-side construction and
    is deliberately not executed during analytical preflight.  Here we verify
    the frozen inputs and the schedule-domain invariants that can be known before
    sampling.
    """
    h=np.asarray(arm.hamiltonian.linear_z,float)
    J=np.asarray(arm.hamiltonian.quadratic_zz,float); J=J+J.T
    s_grid=np.linspace(0.0,1.0,21)
    A=np.interp(s_grid,calibration.s,calibration.A_GHz)
    B=np.interp(s_grid,calibration.s,calibration.B_GHz)
    passed=bool(
        h.ndim==1 and J.shape==(h.size,h.size) and np.isfinite(h).all() and np.isfinite(J).all()
        and np.allclose(J,J.T,rtol=0,atol=1e-12)
        and s_grid.size>=5 and abs(s_grid[0])<=1e-12 and abs(s_grid[-1]-1)<=1e-12 and np.all(np.diff(s_grid)>0)
        and np.isfinite(A).all() and np.isfinite(B).all() and np.all(A>=0) and np.all(B>=0)
    )
    return {
        "pass":passed,
        "logical_variables":int(h.size),
        "s_grid_points":int(s_grid.size),
        "calibration_finite_nonnegative":bool(np.isfinite(A).all() and np.isfinite(B).all() and np.all(A>=0) and np.all(B>=0)),
        "ising_symmetric_finite":bool(np.isfinite(J).all() and np.allclose(J,J.T,rtol=0,atol=1e-12)),
        "susceptibility_sampling_status":"EXECUTE_ON_GPU_CAMPAIGN_SIDE; not an analytical pretest",
    }


def _application_asset_preflight(root: Path) -> dict[str, Any]:
    registry=validate_four_task_registry(FOUR_TASK_EXPERIMENTS)
    by_program={
        "case300_ac_bess":{"assets_required":[],"available":True},
        "ieee33_resilience":{},"few_fem_outer_rotor_bldc":{},"eeg_phase_syntax_microstates":{},
    }
    asset_program_map={
        "ieee33":"ieee33_resilience",
        "motor":"few_fem_outer_rotor_bldc",
        "eeg":"eeg_phase_syntax_microstates",
    }
    for short_name, rels in APPLICATION_ASSETS.items():
        program=asset_program_map[short_name]
        rows={rel:(root/rel).is_file() for rel in rels}
        by_program[program]={"assets":rows,"available":bool(all(rows.values())),"missing":[k for k,v in rows.items() if not v]}
    exp_rows={}
    for e in FOUR_TASK_EXPERIMENTS:
        p=e.program
        if p == "case300_ac_bess":
            status=STATUS_PASS_SCOPE; reason="case300 application/QUBO inputs are present; GPU outcome remains unobserved in this preflight"
        elif p == "ieee33_resilience":
            ok=by_program[p]["available"]; status=STATUS_PASS_SCOPE if ok else STATUS_BLOCK_ASSET; reason="inputs present" if ok else "required IEEE-33 application/fault assets are missing"
        elif p == "few_fem_outer_rotor_bldc":
            ok=by_program[p]["available"]; status=STATUS_PASS_SCOPE if ok else STATUS_BLOCK_ASSET; reason="inputs present" if ok else "required CAD/FEM/bench assets are missing"
        elif p == "eeg_phase_syntax_microstates":
            ok=by_program[p]["available"]; status=STATUS_PASS_SCOPE if ok else STATUS_BLOCK_ASSET; reason="inputs present" if ok else "required real/public EEG/preprocessing/microstate assets are missing"
        else:
            status=STATUS_BLOCK; reason="unknown application program"
        exp_rows[e.experiment_id]={"program":p,"status":status,"reason":reason,"dependencies":list(e.requires)}
    return {"registry":registry,"programs":by_program,"experiments":exp_rows}


def _artifact_integrity(root: Path) -> dict[str, Any]:
    out={}
    manifest_path=root/"releases"/"scientific_manifests"/"FROZEN_SOURCE_MANIFEST_v51.json"
    if not manifest_path.is_file():
        return {"f0_manifest_present":False,"pass":False}
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    # tolerate either {files:{path:hash}} or list records.
    source=manifest.get("files", manifest)
    pairs=[]
    if isinstance(source, Mapping):
        for k,v in source.items():
            if isinstance(v,str): pairs.append((k,v))
            elif isinstance(v,Mapping) and "sha256" in v: pairs.append((k,str(v["sha256"])))
    elif isinstance(source,list):
        for r in source:
            if isinstance(r,Mapping) and "path" in r and "sha256" in r: pairs.append((str(r["path"]),str(r["sha256"])))
    mismatches=[]
    for rel,expected in pairs:
        p=root/rel
        if not p.is_file() or _sha256(p)!=expected: mismatches.append(rel)
    return {"f0_manifest_present":True,"manifest_entries":len(pairs),"mismatches":mismatches,"pass":bool(pairs and not mismatches)}


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
    root=Path(project_root)
    # The exact heavy SVD geometry is precomputed once by the assistant and
    # fingerprinted.  Runtime CPU preflight verifies that the current deterministic
    # designs match that certificate; users do not need to repeat the expensive SVD.
    cert_path=root/"releases"/"scientific_manifests"/"ANALYTICAL_GEOMETRY_CERTIFICATE_v56.json"
    if not cert_path.is_file():
        raise AnalyticalPreflightError(f"missing analytical geometry certificate: {cert_path}")
    cert=json.loads(cert_path.read_text(encoding="utf-8"))
    if cert.get("schema")!="CSSF-QA-ANALYTICAL-GEOMETRY-CERTIFICATE-v56" or not cert.get("validated_exact_svd"):
        raise AnalyticalPreflightError("invalid analytical geometry certificate")

    controls=_control_design_preflight(root,protocol,calibration_family)
    if controls["designs"]["cssf_initial"]["sha256"] != cert.get("cssf_initial_design_sha256"):
        raise AnalyticalPreflightError("CSSF initial design changed after geometry certification")
    director_rows=director_validation_design(protocol,DEFAULT_DIRECTOR_PLAN)
    director_X=np.asarray([r["control"] for r in director_rows],dtype=float)
    director_sha=hashlib.sha256(np.ascontiguousarray(director_X).tobytes()).hexdigest()
    if director_sha != cert.get("director_full_design_sha256"):
        raise AnalyticalPreflightError("director design changed after geometry certification")

    sampling=dict(cert["sampling_resolution"])
    raw_geometry=dict(cert["raw_control"]["geometry"])
    family_geometry=dict(cert["operator_action"]["families"])
    primary_geometry=dict(family_geometry[calibration_family])
    factorial=dict(cert["factorial_causal_completeness"])
    director_geometry=dict(cert["director_design_geometry"])
    core_ids=tuple(e for e in experiment_ids if e not in {"raw-trig-shared-corpus","QZero-matched-full","Worldline-Susceptibility-full"})
    experiments={}
    for eid in core_ids:
        geom=raw_geometry if eid in {"D0","D1"} else primary_geometry
        reasons=[]
        if not factorial.get("full_rank",False): reasons.append("D0-D3 factorial contrast matrix is rank deficient")
        if not geom.get("full_column_rank",False): reasons.append("CSSF design lacks full column rank")
        experiments[eid]={"pass":not reasons,"reasons":reasons,"forecast":_forecast(geom,sampling),"gpu_or_qpu_calls_used_by_preflight":0}
    payload={
        "schema":SCHEMA,"annealer_calls":0,"protocol":asdict(protocol),
        "raw_control":cert["raw_control"],"operator_action":cert["operator_action"],
        "sampling_resolution":sampling,"director_design_geometry":director_geometry,
        "factorial_causal_completeness":factorial,"experiments":experiments,
        "geometry_certificate":{"path":str(cert_path),"sha256":_sha256(cert_path),"director_design_sha256_verified":director_sha},
    }
    raw_q=_qubo_deep_metrics(raw_arm); trig_q=_qubo_deep_metrics(trig_arm)
    topology=_topology_capacity_preflight(root,len(trig_arm.problem.variable_order))
    app=_application_asset_preflight(root)
    integrity=_artifact_integrity(root)
    filename,_=APPROVED_FAMILIES[calibration_family]
    curve=load_calibration(root/"calibration"/filename,calibration_family)
    worldline=_worldline_preflight(trig_arm,curve)
    qzero=qzero_claim_gate(root/"data"/"qzero_full_pretraining_corpus_v38.json")

    # Sampling is a design parameter, not a post-hoc outcome-dependent knob.
    p_reference=1e-3
    r95=_required_reads_for_hit_probability(p_reference,0.95)
    confirm_reads=max(4096,r95)
    sampling_power=_sampling_power_table([protocol.reads_per_control,confirm_reads,8192,8192*4])

    experiments={str(k):dict(v) for k,v in payload.get("experiments",{}).items()}
    # Promote base pass records to explicit v56 status.
    for eid,row in experiments.items():
        row["status"]=STATUS_PASS if row.get("pass") else STATUS_BLOCK
        row["predeclared_action"]="EXECUTE_UNCHANGED" if row.get("pass") else "BLOCK"

    # Exact missing paths from v55 are now governed explicitly.
    shared_ok=controls["all_generated_forward_admissible"]
    experiments["raw-trig-shared-corpus"]={"pass":bool(shared_ok),"status":STATUS_PASS if shared_ok else STATUS_BLOCK,"reasons":[] if shared_ok else ["shared control design invalid"],"predeclared_action":"EXECUTE_UNCHANGED" if shared_ok else "BLOCK","gpu_or_qpu_calls_used_by_preflight":0}

    qz_pass=bool(qzero.get("pass",False))
    experiments["QZero-matched-full"]={"pass":qz_pass,"status":STATUS_PASS if qz_pass else STATUS_BLOCK_ASSET,"reasons":[] if qz_pass else [qzero.get("status","QZero corpus unavailable")],"predeclared_action":"EXECUTE" if qz_pass else "DO_NOT_EXECUTE_QZERO; keep comparator row unavailable/claim locked","gpu_or_qpu_calls_used_by_preflight":0}

    experiments["Worldline-Susceptibility-full"]={"pass":bool(worldline["pass"]),"status":STATUS_PASS_SCOPE if worldline["pass"] else STATUS_BLOCK,"reasons":[] if worldline["pass"] else ["worldline schedule invalid"],"predeclared_action":"EXECUTE_CONSTRUCTION; use matched independent production sampling before application comparison","gpu_or_qpu_calls_used_by_preflight":0,"forecast_scope":"512 search reads are not a rare-tail confirmation budget"}

    # D45-10 is the exact p_gamma bridge.  512 reads provide only ~40% chance of
    # seeing at least one p=1e-3 event, so the CPU gate changes its future GPU
    # resource plan prospectively to 4096 before any D45-10 sample exists.
    if "D45-10" in experiments:
        experiments["D45-10"].update({
            "pass":True,
            "status":STATUS_EXPAND,
            "predeclared_action":"EXECUTE_WITH_EXPANDED_READ_BUDGET",
            "planned_reads_v55":512,
            "locked_reads_v56":confirm_reads,
            "rare_event_reference_probability":p_reference,
            "required_reads_for_95pct_one_hit":r95,
            "hit_probability_at_512":float(1-(1-p_reference)**512),
            "hit_probability_at_locked_reads":float(1-(1-p_reference)**confirm_reads),
        })

    # Director corpus may estimate a response surface at 512 reads/control, but
    # individual zero-hit records are explicitly not treated as proof p_elite=0.
    if "director-corpus" in experiments:
        experiments["director-corpus"].update({
            "status":STATUS_PASS_SCOPE,
            "predeclared_action":"EXECUTE_512_PER_CONTROL_FOR_COVERAGE; reserve >=4096 for frozen-control tail confirmation",
            "rare_event_scope":"response-identification corpus, not single-control rare-tail certification",
        })

    # Production sampling already has 4*8192 reads per frozen schedule.
    if "production-sampling" in experiments:
        experiments["production-sampling"].update({"status":STATUS_PASS,"predeclared_action":"EXECUTE_4x8192_PER_FROZEN_CONTROL"})

    # The residual hierarchy is a mechanism/decomposition experiment.  Its 16 observed
    # controls and 16 operator-action coordinates are structurally auditable, but this
    # square design is not promoted to broad held-out predictive evidence by preflight.
    if "residual-hierarchy" in experiments:
        experiments["residual-hierarchy"].update({
            "status":STATUS_PASS_SCOPE,
            "predeclared_action":"EXECUTE_AS_MECHANISM_DIAGNOSTIC; do not treat as broad predictive confirmation",
            "scope":{
                "logical_qubits":int(len(trig_arm.problem.variable_order)),
                "observed_controls":16,
                "operator_action_coordinates":16,
                "teacher_backend_contract":"Aer GPU tensor_network; no 2^K statevector requirement",
                "predictive_scope":"mechanism decomposition/ranking on observed controls; separate held-out gate required for control authority",
            },
        })

    # Structural gates common to all executable case300 SQA paths.
    common_fail=[]
    if not integrity.get("pass"): common_fail.append("frozen source integrity failed")
    if not controls["all_generated_forward_admissible"]: common_fail.append("one or more planned control designs are non-admissible")
    if not raw_q["penalty_dominance_pass"] or not trig_q["penalty_dominance_pass"]: common_fail.append("QUBO cardinality penalty dominance failed")
    if not raw_q["variable_order_matches_model"] or not trig_q["variable_order_matches_model"]: common_fail.append("candidate/QUBO variable order mismatch")
    if not topology["structural_clique_capacity_pass"]: common_fail.append("K exceeds ideal P16 clique capacity")
    if not topology["target_ocean_lock_pass"]: common_fail.append("target Ocean lock missing")
    if not worldline["pass"]: common_fail.append("worldline schedule construction failed")

    # QZero is optional-but-primary comparator evidence: missing corpus blocks only
    # QZero and any broad all-comparator claim, not unrelated SQA execution.
    for eid,row in experiments.items():
        if eid=="QZero-matched-full": continue
        if common_fail:
            row["pass"]=False; row["status"]=STATUS_BLOCK; row.setdefault("reasons",[]).extend(common_fail); row["predeclared_action"]="BLOCK"

    executable_non_qzero=[eid for eid in experiment_ids if eid!="QZero-matched-full"]
    core_gpu_ready=bool(all(experiments.get(eid,{}).get("pass",False) for eid in executable_non_qzero))

    payload.update({
        "annealer_calls":0,
        "purpose":"complete CPU analytical design gate before any GPU SQA/QPU evidence",
        "governance":{
            "rule":"E_GPU_locked = G(E_planned, Pi_CPU), with G frozen before GPU responses",
            "allowed_actions":["PASS","EXPAND","REPAIR","BLOCK"],
            "forbidden_posthoc_actions":["drop a mandatory arm because forecast is unfavorable","select CSSF support after GPU Y is inspected","weaken comparator budget after seeing outcomes","retune claim thresholds after locked results"],
        },
        "control_designs":controls,
        "qubo_deep_audit":{"raw":raw_q,"trig":trig_q},
        "pegasus_p16_structural_preflight":topology,
        "worldline_preflight":worldline,
        "qzero_preflight":qzero,
        "sampling_power":{
            "reference_rare_event_probability":p_reference,
            "required_reads_for_95pct_one_hit":r95,
            "v56_confirmatory_floor_reads":confirm_reads,
            "table":sampling_power,
        },
        "application_preflight":app,
        "artifact_integrity":integrity,
        "experiments":experiments,
        "execution_plan":{
            "D45-10":{"reads":confirm_reads,"reason":"prospective rare-event resolution repair"},
            "production":{"reads_per_replicate":8192,"replicates":4},
            "Worldline-production":{"reads_per_replicate":8192,"replicates":4,"reason":"matched final-production confirmation, independent of 512-read search construction"},
            "director":{"reads_per_control":DEFAULT_DIRECTOR_PLAN.reads_per_control,"scope":"identification, not individual rare-tail certification"},
            "QZero":{"execute":qz_pass,"status":qzero.get("status")},
        },
        "gpu_program_ready_excluding_missing_qzero":core_gpu_ready,
        "broad_all_comparator_claim_ready_before_gpu":False,
        "broad_all_comparator_claim_blocker":"QZero full pretraining corpus missing" if not qz_pass else "GPU/application outcomes not yet observed",
        "overall_pass":core_gpu_ready,
        "overall_interpretation":"PASS means the executable non-QZero GPU/SQA design is mathematically cleared after the prospective D45-10 read-budget expansion; it does not predict a favorable GPU outcome.",
    })
    if output_path is not None:
        p=Path(output_path); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(_jsonable(payload),indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    return AnalyticalPreflight(payload=payload)


def require_experiment_preflight(preflight: AnalyticalPreflight, experiment_id: str) -> None:
    preflight.require(experiment_id)


__all__=["SCHEMA","DEFAULT_SQA_EXPERIMENTS","AnalyticalPreflightError","AnalyticalPreflight","run_analytical_preflight","require_experiment_preflight"]
