"""Integrated three-level CSSF BESS proof primitives (v38).

This module is additive: the original 75 scientific source files remain byte
frozen.  It connects the original domain CSNN-T representation, the frozen
BESS/QUBO machinery, calibration-resolved operator-action controls and the
canonical CSNN-T QA-response model into one experiment lineage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bess.case300 import Case300ModeAData, load_case300_mode_a
from bess.candidates import CandidateSelectionConfig, CandidateSelectionResult, build_case300_fleet
from config.loader import load_config
from core.csnn_t import CSNNTModel, fit_csnn_t
from core.dataset import BESSDataset
from opf.bess_constraints import BESSUnitSpec
from qaoa.hamiltonian import qubo_to_ising, require_qubo_ising_equivalence
from qubo.builder import build_bess_placement_qubo
from experiments_dwave.benchmark_protocol import MatchedControlProtocol, TARGET_NAMES
from experiments_dwave.cssf_control_v53 import fit_cssf_qa_response
from benchmarks import reference_competitors as rc


class IntegratedBESSError(RuntimeError):
    pass


def _readonly(values: Any, dtype: Any) -> np.ndarray:
    arr=np.array(values,dtype=dtype,order="C",copy=True).reshape(-1); arr.setflags(write=False); return arr


def _normalized_component(values: np.ndarray) -> np.ndarray:
    maximum=float(np.max(values)); return np.zeros_like(values,dtype=float) if maximum<=0 else np.asarray(values/maximum,dtype=float)


def candidate_selection_from_lsf(
    data: Case300ModeAData,
    config: CandidateSelectionConfig,
    train_lsf: np.ndarray,
    *,
    representation: str,
    model_fingerprint: str,
) -> CandidateSelectionResult:
    """Apply the frozen candidate-score formula to a declared train-only LSF representation."""
    y=np.asarray(train_lsf,dtype=np.float64)
    if y.shape!=(data.n_train,data.n):
        raise IntegratedBESSError(f"train_lsf shape {y.shape} != {(data.n_train,data.n)}")
    if not np.isfinite(y).all(): raise IntegratedBESSError("train_lsf contains non-finite values")
    eligible=np.ones(data.n,dtype=bool)
    if config.exclude_slack: eligible[np.asarray(data.slack_buses,dtype=int)]=False
    buses=np.flatnonzero(eligible).astype(np.int64,copy=False)
    absolute=np.abs(y); mean_abs=np.mean(absolute,axis=0); rms=np.sqrt(np.mean(y*y,axis=0)); std=np.std(y,axis=0); tail=np.quantile(absolute,config.tail_quantile,axis=0)
    score=(config.mean_abs_weight*_normalized_component(mean_abs)+config.rms_weight*_normalized_component(rms)+config.std_weight*_normalized_component(std)+config.tail_weight*_normalized_component(tail))
    eligible_scores=score[buses]; order=np.lexsort((buses,-eligible_scores)); ranked=buses[order]; ranked_scores=eligible_scores[order]
    candidates=tuple(int(x) for x in ranked[:config.candidate_count])
    meta=MappingProxyType({
        "case":data.case,"training_scenarios":data.n_train,"held_out_scenarios_used":0,
        "excluded_slack_buses":tuple(data.slack_buses) if config.exclude_slack else (),
        "score_order":"descending_score_then_ascending_bus",
        "domain_representation":str(representation),"domain_model_fingerprint":str(model_fingerprint),
        "formula_identity":"bess.candidates.select_case300_candidates score formula",
    })
    return CandidateSelectionResult(
        candidate_buses=candidates,ranked_buses=_readonly(ranked,np.int64),scores=_readonly(ranked_scores,np.float64),
        mean_absolute_lsf=_readonly(mean_abs[ranked],np.float64),rms_lsf=_readonly(rms[ranked],np.float64),
        std_lsf=_readonly(std[ranked],np.float64),tail_absolute_lsf=_readonly(tail[ranked],np.float64),
        source_fingerprint=data.fingerprint(),config_fingerprint=config.fingerprint(),metadata=meta,
    )


@dataclass(frozen=True)
class DomainRepresentation:
    name: str
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    model_fingerprint: str
    diagnostics: Mapping[str,Any]
    csnnt_model: CSNNTModel | None = None


def build_domain_representation(project_root: str | Path, representation: str) -> tuple[Case300ModeAData,DomainRepresentation]:
    root=Path(project_root); data=load_case300_mode_a(root/"data"/"case300_full_modeA_Barskyi_Serhii.json")
    rep=str(representation).strip().lower()
    if rep=="trig":
        ds=BESSDataset(case=data.case,X_train=data.features[data.train_slice],y_train=data.targets[data.train_slice],X_test=data.features[data.test_slice],y_test=data.targets[data.test_slice],metadata={"representation":"modeA_trigonometric"})
        model=fit_csnn_t(ds,n_lambdas=100,lam_range=(-12,4)); train=model.predict(ds.X_train); test=model.predict(ds.X_test)
        fp=hashlib.sha256(model.H.tobytes(order="C")+repr(model.lam_opt).encode()).hexdigest()
        return data,DomainRepresentation("trig",train,test,fp,{"lambda_gcv":float(model.lam_opt),"features":int(model.M),"call_path":"core.csnn_t.fit_csnn_t"},model)
    if rep=="raw":
        try: from sklearn.linear_model import RidgeCV
        except Exception as exc: raise IntegratedBESSError("scikit-learn is required for the raw-domain factorial arm") from exc
        grid=np.logspace(-12,4,100); model=RidgeCV(alphas=grid,fit_intercept=True)
        model.fit(data.theta_rad[data.train_slice],data.targets[data.train_slice]); train=np.asarray(model.predict(data.theta_rad[data.train_slice]),dtype=float); test=np.asarray(model.predict(data.theta_rad[data.test_slice]),dtype=float)
        payload={"coef":np.asarray(model.coef_).tolist(),"intercept":np.asarray(model.intercept_).tolist(),"alpha":float(model.alpha_)}
        fp=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        return data,DomainRepresentation("raw",train,test,fp,{"ridge_alpha":float(model.alpha_),"raw_dimensions":int(data.theta_rad.shape[1]),"role":"factorial raw-domain ablation"},None)
    raise IntegratedBESSError("representation must be raw or trig")


@dataclass(frozen=True)
class BESSArmProblem:
    representation: str
    data: Case300ModeAData
    domain: DomainRepresentation
    selection: CandidateSelectionResult
    fleet: Any
    problem: Any
    hamiltonian: Any
    ising_audit: Any


def build_bess_arm_problem(project_root: str | Path, representation: str) -> BESSArmProblem:
    root=Path(project_root); data,domain=build_domain_representation(root,representation)
    cfg=load_config([root/"config"/"base.yaml",root/"config"/"case300.yaml",root/"config"/"emulator_gpu.yaml"])
    selection_cfg=CandidateSelectionConfig.from_qubo_config(cfg.qubo)
    selection=candidate_selection_from_lsf(data,selection_cfg,domain.train_predictions,representation=domain.name,model_fingerprint=domain.model_fingerprint)
    fleet=build_case300_fleet(selection,bess_units=cfg.qubo.bess_units,unit=BESSUnitSpec(power_mw=25.0,energy_mwh=100.0),metadata={"strategy":f"case300_{domain.name}_domain_train_only"})
    linear={bus:-float(selection.scores[i]) for i,bus in enumerate(selection.candidate_buses)}
    problem=build_bess_placement_qubo(fleet,linear_by_bus=linear,metadata={"candidate_selection_fingerprint":selection.fingerprint(),"domain_representation":domain.name,"domain_model_fingerprint":domain.model_fingerprint})
    h=qubo_to_ising(problem.model); audit=require_qubo_ising_equivalence(problem.model,h,exact_limit=18,random_samples=8192,seed=cfg.random.global_seed,tolerance=cfg.qubo.verify_qubo_ising_tolerance)
    return BESSArmProblem(domain.name,data,domain,selection,fleet,problem,h,audit)


@dataclass(frozen=True)
class FactorialArmSpec:
    arm_id: str
    domain_representation: str
    control_representation: str
    matched_config_sha256: str


def factorial_specs(protocol: MatchedControlProtocol | None = None) -> tuple[FactorialArmSpec,...]:
    p=MatchedControlProtocol() if protocol is None else protocol
    matched={"protocol":asdict(p),"objective":"application_utility","endpoint":"AC/OOD/full-N-1","surrogate":"frozen_CSNN-T_GCV","acquisition":"mean+uncertainty+leverage","confirmation":"paired_independent"}
    digest=hashlib.sha256(json.dumps(matched,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return (
        FactorialArmSpec("D0","raw","raw",digest),FactorialArmSpec("D1","trig","raw",digest),
        FactorialArmSpec("D2","raw","operator_phase",digest),FactorialArmSpec("D3","trig","operator_phase",digest),
    )


def raw_control_coordinates(control: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Map raw schedule parameters to dimensionless coordinates without A/B action integration.

    The CSNN-T/GCV model and acquisition are unchanged; only the coordinate map
    differs from operator-action CSSF.  This is the mandatory causal ablation.
    """
    x=np.asarray(control,dtype=float).reshape(-1); b=np.asarray(bounds,dtype=float)
    if b.shape!=(x.size,2): raise IntegratedBESSError("raw control/bounds shape mismatch")
    width=b[:,1]-b[:,0]
    if np.any(width<=0): raise IntegratedBESSError("invalid control bounds")
    return 2.0*np.pi*(x-b[:,0])/width-np.pi


def run_cssf_same_machinery(
    evaluator: Callable[...,dict[str,Any]],
    protocol: MatchedControlProtocol,
    *,
    project_root: str | Path,
    coordinate_mode: str,
    candidate_pool_size: int = 4096,
) -> Any:
    """Run the identical CSSF fit/acquisition loop with one selectable coordinate map.

    ``coordinate_mode='operator_phase'`` calls the physical A/B action map.
    ``coordinate_mode='raw'`` uses only normalized raw control parameters.
    Every other optimizer setting, observation, seed, budget and acquisition
    operation is shared by this single function.
    """
    from experiments_dwave.benchmark_protocol import MethodTrace, _evaluate, _response_matrix, shared_initial_design
    mode=str(coordinate_mode).strip().lower()
    if mode not in {"raw","operator_phase"}: raise IntegratedBESSError("coordinate_mode must be raw or operator_phase")
    method="CSSF-trig" if mode=="operator_phase" else "CSSF-raw/no-trig"
    ledger=rc.ResourceLedger(); controls=[]; responses=[]; diags=[]; bounds=protocol.bounds
    initial=shared_initial_design(protocol)
    for c in initial:
        r=_evaluate(method,c,evaluator,protocol.reads_per_control,ledger,{"phase":"initial_design","coordinate_mode":mode}); controls.append(c.copy()); responses.append(r); diags.append({"phase":"initial_design","coordinate_mode":mode})
    def coord(c:np.ndarray, response:Mapping[str,Any]|None=None)->np.ndarray:
        if mode=="raw": return raw_control_coordinates(c,bounds)
        if response is not None and "operator_action" in response: return np.asarray(response["operator_action"],dtype=float)
        pure=getattr(evaluator,"operator_action",None)
        if pure is None or not callable(pure): raise IntegratedBESSError("operator-phase arm requires evaluator.operator_action(control)")
        return np.asarray(pure(c),dtype=float)
    while len(controls)<protocol.total_control_budget:
        theta=np.vstack([coord(c,r) for c,r in zip(controls,responses,strict=True)]); Y=_response_matrix(responses)
        nt=protocol.cssf_initial_train; nc=protocol.cssf_initial_calibration; cal=np.arange(nt,nt+nc,dtype=int); active=np.arange(nt+nc,len(responses),dtype=int); train=np.concatenate([np.arange(nt,dtype=int),active])
        model=fit_cssf_qa_response(theta[train],Y[train],calibration_operator_phase=theta[cal],calibration_targets=Y[cal],target_names=TARGET_NAMES,project_root=str(project_root),support_mode="signed_axes",support_order=1,metadata={"method":method,"coordinate_mode":mode,"protocol":asdict(protocol),"active_refit_observations":int(active.size)})
        pool=rc.latin_hypercube(bounds,int(candidate_pool_size),seed=protocol.seed+1000+len(controls)); feasible=[]; coords=[]
        for c in pool:
            try:
                # Raw arm still obeys physical schedule feasibility; it differs only in representation used by the surrogate.
                rc.fourier_forward_schedule(c,order=protocol.order,grid_points=129,reject_nonmonotone=True)
                coords.append(coord(c)); feasible.append(c)
            except Exception: continue
        if not feasible: raise IntegratedBESSError("candidate pool contains no physically feasible schedule")
        idx,score,diag=model.acquisition(np.vstack(coords),target="elite_probability",maximize=True,uncertainty_weight=1.0,leverage_weight=0.25,feasibility_target="feasibility_probability",minimum_feasibility=0.0)
        c=np.asarray(feasible[idx],dtype=float); d={**dict(diag),"coordinate_mode":mode}
        r=_evaluate(method,c,evaluator,protocol.reads_per_control,ledger,d); controls.append(c.copy()); responses.append(r); diags.append(d)
    return MethodTrace(method,controls,responses,ledger,diags)



@dataclass(frozen=True)
class SharedCorpusAblationResult:
    evidence: Mapping[str, Any]
    raw_model: Any
    trig_model: Any
    controls: np.ndarray
    responses: tuple[Mapping[str, Any], ...]


def run_shared_corpus_raw_trig_ablation(
    evaluator: Callable[..., dict[str, Any]],
    protocol: MatchedControlProtocol,
    *,
    project_root: str | Path,
    corpus_size: int | None = None,
    candidate_pool_size: int = 4096,
) -> SharedCorpusAblationResult:
    """Causal raw-vs-operator-phase ablation on one immutable response corpus.

    The annealer is queried exactly once for every shared observation.  Both
    representations receive the same observation ids, targets, train/cal/test
    split, CSNN-T/GCV policy, uncertainty construction, acquisition weights,
    feasible candidate pool, seeds and read budget.  The only factor changed is
    the coordinate transformation presented to the CSSF spectral surrogate.
    """
    from experiments_dwave.benchmark_protocol import _response_matrix, shared_initial_design

    n = int(protocol.total_control_budget if corpus_size is None else corpus_size)
    if n < protocol.cssf_initial_count + 2:
        raise IntegratedBESSError("shared ablation corpus must contain train, calibration and held-out observations")
    if n > protocol.total_control_budget:
        raise IntegratedBESSError("shared ablation corpus may not exceed the matched control budget")
    # A deterministic *feasible* design is used once; neither arm may adaptively modify it.
    # Feasibility filtering is query-free and therefore cannot advantage either representation.
    proposed: list[np.ndarray] = []
    design_round = 0
    seen: set[bytes] = set()
    while len(proposed) < n:
        batch_n = max(4*n, 128)
        batch = rc.latin_hypercube(protocol.bounds, batch_n, seed=protocol.seed+3705+design_round)
        design_round += 1
        for c in batch:
            try:
                rc.fourier_forward_schedule(c, order=protocol.order, grid_points=129, reject_nonmonotone=True)
            except Exception:
                continue
            key=np.ascontiguousarray(np.asarray(c,dtype=np.float64)).tobytes(order="C")
            if key in seen:
                continue
            seen.add(key); proposed.append(np.asarray(c,dtype=float))
            if len(proposed) >= n:
                break
        if design_round > 100 and len(proposed) < n:
            raise IntegratedBESSError("could not construct the shared feasible response corpus")
    controls = np.vstack(proposed[:n])
    responses: list[Mapping[str, Any]] = []
    observation_ids: list[str] = []
    for i, control in enumerate(controls):
        response = evaluator(np.asarray(control, dtype=float), num_reads=int(protocol.reads_per_control))
        responses.append(dict(response))
        observation_ids.append(f"shared-corpus-{i:04d}")
    Y = _response_matrix(responses)
    nt, nc = int(protocol.cssf_initial_train), int(protocol.cssf_initial_calibration)
    train_idx = np.arange(nt, dtype=int)
    cal_idx = np.arange(nt, nt+nc, dtype=int)
    test_idx = np.arange(nt+nc, n, dtype=int)
    if test_idx.size == 0:
        raise IntegratedBESSError("shared causal ablation requires a held-out response subset")

    pure = getattr(evaluator, "operator_action", None)
    if pure is None or not callable(pure):
        raise IntegratedBESSError("shared causal ablation requires evaluator.operator_action(control) without an annealer query")
    raw_theta = np.vstack([raw_control_coordinates(c, protocol.bounds) for c in controls])
    trig_theta = np.vstack([np.asarray(r.get("operator_action", pure(c)), dtype=float) for c, r in zip(controls, responses, strict=True)])

    shared_fit = dict(
        calibration_targets=Y[cal_idx], target_names=TARGET_NAMES, project_root=str(project_root),
        support_mode="signed_axes", support_order=1, n_lambdas=100, lam_range=(-12.0, 4.0), nominal_coverage=0.90,
    )
    raw_model = fit_cssf_qa_response(
        raw_theta[train_idx], Y[train_idx], calibration_operator_phase=raw_theta[cal_idx],
        metadata={"experiment":"V05-shared-corpus","representation":"raw/no-operator-phase"}, **shared_fit,
    )
    trig_model = fit_cssf_qa_response(
        trig_theta[train_idx], Y[train_idx], calibration_operator_phase=trig_theta[cal_idx],
        metadata={"experiment":"V05-shared-corpus","representation":"operator-phase"}, **shared_fit,
    )

    pool = rc.latin_hypercube(protocol.bounds, int(candidate_pool_size), seed=protocol.seed+37505)
    feasible: list[np.ndarray] = []
    raw_pool: list[np.ndarray] = []
    trig_pool: list[np.ndarray] = []
    for c in pool:
        try:
            rc.fourier_forward_schedule(c, order=protocol.order, grid_points=129, reject_nonmonotone=True)
            feasible.append(np.asarray(c, dtype=float))
            raw_pool.append(raw_control_coordinates(c, protocol.bounds))
            trig_pool.append(np.asarray(pure(c), dtype=float))
        except Exception:
            continue
    if not feasible:
        raise IntegratedBESSError("shared causal ablation candidate pool contains no feasible schedule")
    raw_pool_arr = np.vstack(raw_pool); trig_pool_arr = np.vstack(trig_pool); feasible_arr = np.vstack(feasible)
    acq_kwargs=dict(target="elite_probability",maximize=True,uncertainty_weight=1.0,leverage_weight=0.25,feasibility_target="feasibility_probability",minimum_feasibility=0.0)
    raw_idx, _, raw_diag = raw_model.acquisition(raw_pool_arr, **acq_kwargs)
    trig_idx, _, trig_diag = trig_model.acquisition(trig_pool_arr, **acq_kwargs)

    raw_pred = raw_model.predict(raw_theta[test_idx]); trig_pred = trig_model.predict(trig_theta[test_idx])
    raw_mse = float(np.mean((raw_pred-Y[test_idx])**2)); trig_mse = float(np.mean((trig_pred-Y[test_idx])**2))
    raw_cov = raw_model.empirical_coverage(raw_theta[test_idx], Y[test_idx]).tolist()
    trig_cov = trig_model.empirical_coverage(trig_theta[test_idx], Y[test_idx]).tolist()
    candidate_pool_sha = hashlib.sha256(np.ascontiguousarray(feasible_arr).tobytes(order="C")).hexdigest()
    shared_config = {
        "surrogate": {"family":"CSNN-T","gcv_n_lambdas":100,"gcv_lam_range":[-12.0,4.0],"support_mode":"signed_axes","support_order":1},
        "target_transform": "none",
        "uncertainty": {"nominal_coverage":0.90,"policy":"calibration-residual-scale+leverage"},
        "acquisition": {"target":"elite_probability","uncertainty_weight":1.0,"leverage_weight":0.25,"minimum_feasibility":0.0},
        "candidate_pool": {"size":int(feasible_arr.shape[0]),"sha256":candidate_pool_sha},
        "stopping_rule": {"kind":"fixed_shared_corpus","controls":n},
        "reads_per_control": int(protocol.reads_per_control),
    }
    common = {
        "observation_ids": observation_ids,
        "train_observation_ids": [observation_ids[i] for i in train_idx],
        "calibration_observation_ids": [observation_ids[i] for i in cal_idx],
        "heldout_observation_ids": [observation_ids[i] for i in test_idx],
        "budget": {"control_evaluations":n,"annealer_reads":n*int(protocol.reads_per_control)},
        "seeds": {"protocol":int(protocol.seed),"shared_design_base":int(protocol.seed+3705),"candidate_pool":int(protocol.seed+37505)},
        "candidate_pool_sha256": candidate_pool_sha,
        "target_names": list(TARGET_NAMES),
    }
    evidence = {
        "schema": "CSSF-QA-RAW-TRIG-CAUSAL-ABLATION-v38",
        "CSSF-raw/no-trig": {**common, "config":{**shared_config,"representation":{"name":"CSSF-raw/no-trig","coordinate_transform":"normalized_raw_schedule_parameters"}},"heldout_mse":raw_mse,"heldout_coverage_by_target":raw_cov,"selected_candidate":feasible_arr[raw_idx].tolist(),"acquisition_diagnostics":dict(raw_diag)},
        "CSSF-trig": {**common, "config":{**shared_config,"representation":{"name":"CSSF-trig","coordinate_transform":"calibration_resolved_operator_action"}},"heldout_mse":trig_mse,"heldout_coverage_by_target":trig_cov,"selected_candidate":feasible_arr[trig_idx].tolist(),"acquisition_diagnostics":dict(trig_diag)},
    }
    return SharedCorpusAblationResult(evidence=evidence, raw_model=raw_model, trig_model=trig_model, controls=np.ascontiguousarray(controls), responses=tuple(responses))

def factorial_interaction(utilities: Mapping[str,float]) -> float:
    missing=set(("D0","D1","D2","D3"))-set(utilities)
    if missing: raise IntegratedBESSError(f"missing factorial utilities: {sorted(missing)}")
    return float(utilities["D3"]-utilities["D2"]-utilities["D1"]+utilities["D0"])


__all__=[
    "IntegratedBESSError","DomainRepresentation","candidate_selection_from_lsf","build_domain_representation",
    "BESSArmProblem","build_bess_arm_problem","FactorialArmSpec","factorial_specs","raw_control_coordinates",
    "run_cssf_same_machinery","SharedCorpusAblationResult","run_shared_corpus_raw_trig_ablation","factorial_interaction",
]
