"""Matched-budget schedule-control benchmarking for CSSF(QA).

The protocol keeps CSSF on the original framework path and uses the additive
reference competitors from ``benchmarks.reference_competitors``.  It is shared
by Simulator and Pegasus notebooks; only the supplied response evaluator changes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Mapping, Any, Sequence
import time

import numpy as np

from benchmarks import reference_competitors as rc
from experiments_dwave.cssf_control_v53 import fit_cssf_qa_response
from experiments_dwave.evidence_v38 import qpu_access_time_us
from experiments_dwave.control_design import admissible_latin_hypercube

TARGET_NAMES=(
    "mean_energy","energy_variance","energy_quantile_05","cvar_05",
    "feasibility_probability","elite_probability","success_probability",
)


class BenchmarkProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatchedControlProtocol:
    order: int = 4
    alpha: float = 0.30
    annealing_time_range_us: tuple[float,float] = (5.0,50.0)
    total_control_budget: int = 96
    cssf_initial_train: int = 66
    cssf_initial_calibration: int = 6
    reads_per_control: int = 512
    turbo_min_reads: int = 250
    turbo_max_reads: int = 900
    seed: int = 20260817

    def __post_init__(self) -> None:
        if self.order<1 or self.alpha<=0: raise BenchmarkProtocolError("invalid Fourier control domain")
        if self.total_control_budget<20: raise BenchmarkProtocolError("total control budget is too small")
        if self.cssf_initial_train<4 or self.cssf_initial_calibration<2: raise BenchmarkProtocolError("CSSF initial partitions are too small")
        if self.cssf_initial_train+self.cssf_initial_calibration>=self.total_control_budget:
            raise BenchmarkProtocolError("CSSF must retain an active-selection budget after initialization")
        if not (1 <= self.turbo_min_reads <= self.reads_per_control <= self.turbo_max_reads):
            raise BenchmarkProtocolError("TuRBO adaptive-read bounds must bracket the matched mean reads/control")
        features=1+2*(2*8)  # operator-action dimension is beta/gamma x 8 segments
        if features>self.cssf_initial_train//2:
            raise BenchmarkProtocolError(
                f"CSSF first-order dictionary violates identifiability: features={features}, train={self.cssf_initial_train}"
            )

    @property
    def bounds(self) -> np.ndarray:
        return rc.fourier_bounds(self.order,self.annealing_time_range_us,self.alpha)

    @property
    def cssf_initial_count(self) -> int:
        return self.cssf_initial_train+self.cssf_initial_calibration

    @property
    def total_read_budget(self) -> int:
        return int(self.total_control_budget*self.reads_per_control)

    @property
    def turbo_runtime_guard_us(self) -> float:
        # Full Jeong-TuRBO requires an explicit pre-submission QPU-runtime guard.
        # The matched comparison itself is fixed by query/read budget; this cap is
        # deliberately conservative so it prevents runaway submission without
        # silently changing the common read budget.
        per_query = 20_000.0 + self.turbo_max_reads*(self.annealing_time_range_us[1]+100.0+20.0)
        return float(self.total_control_budget*per_query)


@dataclass
class MethodTrace:
    method: str
    controls: list[np.ndarray]
    responses: list[dict[str,Any]]
    ledger: rc.ResourceLedger
    diagnostics: list[dict[str,Any]]

    def target_values(self,target:str="elite_probability") -> np.ndarray:
        return np.asarray([float(r[target]) for r in self.responses],dtype=float)

    def best_index(self,target:str="elite_probability",maximize:bool=True) -> int:
        values=self.target_values(target)
        return int(np.argmax(values) if maximize else np.argmin(values))

    def summary(self,target:str="elite_probability",maximize:bool=True) -> dict[str,Any]:
        i=self.best_index(target,maximize)
        return {
            "method":self.method,"queries":len(self.controls),"best_control":self.controls[i].tolist(),
            "best_target":float(self.responses[i][target]),"best_response":{k:v for k,v in self.responses[i].items() if k in TARGET_NAMES},
            "cost":self.ledger.totals(self.method),
        }


def _evaluate(
    method:str, control:np.ndarray, evaluator:Callable[...,dict[str,Any]], reads:int,
    ledger:rc.ResourceLedger, diagnostics:Mapping[str,Any]|None=None,
) -> dict[str,Any]:
    start=time.perf_counter()
    response=evaluator(np.asarray(control,dtype=float),num_reads=int(reads))
    elapsed=time.perf_counter()-start
    ledger.add(rc.CostEntry(
        method=method,stage="search",control_query=1,reads=int(response.get("num_reads",reads)),
        annealing_time_us=float(np.asarray(control)[0]),qpu_access_time_us=qpu_access_time_us(response),
        simulator_seconds=float(response.get("elapsed_seconds",elapsed) if "simulator" in str(response.get("backend_info",{})).lower() else 0.0),
        classical_seconds=float(max(0.0,elapsed-float(response.get("elapsed_seconds",0.0)))),
        metadata={} if diagnostics is None else dict(diagnostics),
    ))
    return response


def _response_matrix(rows:Sequence[Mapping[str,Any]]) -> np.ndarray:
    return np.asarray([[float(r[k]) for k in TARGET_NAMES] for r in rows],dtype=float)


def shared_initial_design(protocol:MatchedControlProtocol) -> np.ndarray:
    return admissible_latin_hypercube(protocol.bounds,protocol.cssf_initial_count,order=protocol.order,seed=protocol.seed)


def run_cssf_full(
    evaluator:Callable[...,dict[str,Any]], protocol:MatchedControlProtocol,
    *, project_root:str, candidate_pool_size:int=4096,
) -> MethodTrace:
    method="CSSF-full"
    ledger=rc.ResourceLedger(); controls=[]; responses=[]; diags=[]
    bounds=protocol.bounds
    initial=shared_initial_design(protocol)
    for c in initial:
        r=_evaluate(method,c,evaluator,protocol.reads_per_control,ledger,{"phase":"initial_design"})
        controls.append(c.copy()); responses.append(r); diags.append({"phase":"initial_design"})
    rng=np.random.default_rng(protocol.seed+101)
    while len(controls)<protocol.total_control_budget:
        theta=np.vstack([np.asarray(r["operator_action"],dtype=float) for r in responses])
        Y=_response_matrix(responses)
        nt=protocol.cssf_initial_train
        nc=protocol.cssf_initial_calibration
        calibration_idx=np.arange(nt,nt+nc,dtype=int)
        active_idx=np.arange(nt+nc,len(responses),dtype=int)
        train_idx=np.concatenate([np.arange(nt,dtype=int),active_idx])
        model=fit_cssf_qa_response(
            theta[train_idx],Y[train_idx],calibration_operator_phase=theta[calibration_idx],calibration_targets=Y[calibration_idx],
            target_names=TARGET_NAMES,project_root=project_root,support_mode="signed_axes",support_order=1,
            metadata={"method":method,"protocol":asdict(protocol),"active_refit_observations":int(active_idx.size)},
        )
        pool=rc.latin_hypercube(bounds,int(candidate_pool_size),seed=protocol.seed+1000+len(controls))
        feasible=[]; phase=[]
        # Evaluator must expose a pure control->operator_action helper to avoid hidden QA calls.
        pure=getattr(evaluator,"operator_action",None)
        if pure is None or not callable(pure):
            raise BenchmarkProtocolError("CSSF evaluator must expose pure operator_action(control) without an annealer query")
        for c in pool:
            try:
                phase.append(np.asarray(pure(c),dtype=float)); feasible.append(c)
            except Exception:
                continue
        if not feasible: raise BenchmarkProtocolError("CSSF candidate pool contains no feasible schedule")
        idx,score,diag=model.acquisition(np.vstack(phase),target="elite_probability",maximize=True,
                                         uncertainty_weight=1.0,leverage_weight=0.25,
                                         feasibility_target="feasibility_probability",minimum_feasibility=0.0)
        c=np.asarray(feasible[idx],dtype=float)
        r=_evaluate(method,c,evaluator,protocol.reads_per_control,ledger,diag)
        controls.append(c.copy()); responses.append(r); diags.append(dict(diag))
    return MethodTrace(method,controls,responses,ledger,diags)


def run_gp_ei_full(evaluator:Callable[...,dict[str,Any]],protocol:MatchedControlProtocol) -> MethodTrace:
    method="GP+EI-full"; ledger=rc.ResourceLedger(); controls=[]; responses=[]; diags=[]
    opt=rc.SequentialGPEI(protocol.bounds,minimize_objective=False,seed=protocol.seed,n_restarts_optimizer=2)
    init=admissible_latin_hypercube(protocol.bounds,10,order=protocol.order,seed=protocol.seed)
    for c in init:
        r=_evaluate(method,c,evaluator,protocol.reads_per_control,ledger,{"phase":"initial_design"}); v=float(r["elite_probability"])
        opt.add(c,v); controls.append(c.copy());responses.append(r);diags.append({"phase":"initial_design"})
    while len(controls)<protocol.total_control_budget:
        c,d=opt.suggest(feasibility=lambda x:rc.feasible_control(x,protocol.bounds,order=protocol.order))
        r=_evaluate(method,c,evaluator,protocol.reads_per_control,ledger,d);opt.add(c,float(r["elite_probability"]))
        controls.append(c.copy());responses.append(r);diags.append(dict(d))
    return MethodTrace(method,controls,responses,ledger,diags)


def run_finzgar_matched_full(evaluator:Callable[...,dict[str,Any]],protocol:MatchedControlProtocol) -> MethodTrace:
    method="Finzgar-BO-matched-full"; ledger=rc.ResourceLedger();controls=[];responses=[];diags=[]
    opt=rc.FinzgarBO(protocol.bounds,seed=protocol.seed,total_paper_iterations=50)
    linear=rc.canonical_control(protocol.order,float(np.mean(protocol.annealing_time_range_us)))
    init=opt.initial_design(linear,feasibility=lambda x:rc.feasible_control(x,protocol.bounds,order=protocol.order))
    for c in init:
        r=_evaluate(method,c,evaluator,protocol.reads_per_control,ledger,{"phase":"linear_plus_9_random"});opt.add(c,float(r["elite_probability"]))
        controls.append(c.copy());responses.append(r);diags.append({"phase":"linear_plus_9_random"})
    while len(controls)<protocol.total_control_budget:
        c,d=opt.suggest(iteration=len(controls),feasibility=lambda x:rc.feasible_control(x,protocol.bounds,order=protocol.order))
        r=_evaluate(method,c,evaluator,protocol.reads_per_control,ledger,d);opt.add(c,float(r["elite_probability"]))
        controls.append(c.copy());responses.append(r);diags.append(dict(d))
    return MethodTrace(method,controls,responses,ledger,diags)


def _tuRBO_matched_reads(
    opt:rc.JeongTuRBO, candidate:np.ndarray, protocol:MatchedControlProtocol,
    *, reads_used:int, queries_used:int,
) -> int:
    """Adaptive Jeong read allocation with an *exact* matched total-read budget.

    The desired read count is the full optimizer's incumbent/progress-dependent
    allocation.  It is then clipped only enough to guarantee that all remaining
    control queries can still be executed within the frozen total read budget.
    Consequently the matched TuRBO arm uses adaptive reads while ending with
    exactly the same total reads as CSSF/GP/Finzgar.
    """
    desired=int(opt.adaptive_reads(candidate,min_reads=protocol.turbo_min_reads,max_reads=protocol.turbo_max_reads))
    remaining_total=int(protocol.total_read_budget-reads_used)
    slots=int(protocol.total_control_budget-queries_used)
    if slots<=0: raise BenchmarkProtocolError("TuRBO read allocator called after budget exhaustion")
    future_slots=slots-1
    lower=max(protocol.turbo_min_reads,remaining_total-future_slots*protocol.turbo_max_reads)
    upper=min(protocol.turbo_max_reads,remaining_total-future_slots*protocol.turbo_min_reads)
    if lower>upper:
        raise BenchmarkProtocolError(f"TuRBO matched-read budget infeasible: lower={lower}, upper={upper}")
    return int(np.clip(desired,lower,upper))


def run_turbo_matched_full(evaluator:Callable[...,dict[str,Any]],protocol:MatchedControlProtocol) -> MethodTrace:
    method="TuRBO-matched-full"; ledger=rc.ResourceLedger();controls=[];responses=[];diags=[]
    opt=rc.JeongTuRBO(protocol.bounds,order=protocol.order,seed=protocol.seed,minimize_objective=False)
    runtime_guard=rc.QPURuntimeBudget(total_us=protocol.turbo_runtime_guard_us)
    reads_used=0
    init=admissible_latin_hypercube(protocol.bounds,10,order=protocol.order,seed=protocol.seed)
    for c in init:
        reads=_tuRBO_matched_reads(opt,c,protocol,reads_used=reads_used,queries_used=len(controls))
        reserved=runtime_guard.estimate(float(c[0]),reads)
        if reserved>float(runtime_guard.remaining_us): raise rc.BudgetError("TuRBO runtime guard exhausted during initial design")
        runtime_guard.remaining_us=float(runtime_guard.remaining_us)-reserved
        r=_evaluate(method,c,evaluator,reads,ledger,{"phase":"space_filling_initial","adaptive_reads":reads,"runtime_reserved_us":reserved})
        reads_used+=int(r.get("num_reads",reads)); opt.add_initial(c,float(r["elite_probability"]))
        controls.append(c.copy());responses.append(r);diags.append({"phase":"space_filling_initial","adaptive_reads":reads,"runtime_reserved_us":reserved})
    while len(controls)<protocol.total_control_budget:
        c,d=opt.suggest(feasibility=lambda x:rc.feasible_control(x,protocol.bounds,order=protocol.order))
        reads=_tuRBO_matched_reads(opt,c,protocol,reads_used=reads_used,queries_used=len(controls))
        reserved=runtime_guard.estimate(float(c[0]),reads)
        if reserved>float(runtime_guard.remaining_us): raise rc.BudgetError("TuRBO runtime guard exhausted before control evaluation")
        runtime_guard.remaining_us=float(runtime_guard.remaining_us)-reserved
        diag={**dict(d),"adaptive_reads":reads,"runtime_reserved_us":reserved,"runtime_remaining_us":float(runtime_guard.remaining_us)}
        r=_evaluate(method,c,evaluator,reads,ledger,diag); reads_used+=int(r.get("num_reads",reads))
        obs=opt.observe(c,float(r["elite_probability"]))
        controls.append(c.copy());responses.append(r);diags.append({**diag,**dict(obs)})
    if reads_used!=protocol.total_read_budget:
        raise BenchmarkProtocolError(f"TuRBO matched arm consumed {reads_used} reads; expected {protocol.total_read_budget}")
    return MethodTrace(method,controls,responses,ledger,diags)


def run_random_search(evaluator:Callable[...,dict[str,Any]],protocol:MatchedControlProtocol) -> MethodTrace:
    method="Random-search";ledger=rc.ResourceLedger();controls=[];responses=[];diags=[]
    design=admissible_latin_hypercube(protocol.bounds,protocol.total_control_budget,order=protocol.order,seed=protocol.seed+77)
    for c in design:
        r=_evaluate(method,c,evaluator,protocol.reads_per_control,ledger,{"phase":"random_search"});controls.append(c.copy());responses.append(r);diags.append({"phase":"random_search"})
    return MethodTrace(method,controls,responses,ledger,diags)


def cost_to_target_table(traces:Sequence[MethodTrace],target_value:float,target:str="elite_probability") -> list[dict[str,Any]]:
    rows=[]
    for tr in traces:
        values=tr.target_values(target)
        costs=[float(r.get("num_reads",0)) for r in tr.responses]
        row=rc.cost_to_target(values,costs,float(target_value),maximize=True); row["method"]=tr.method;rows.append(row)
    return rows
