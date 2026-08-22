"""Runtime glue for integrated D0-D3 CSSF BESS evidence (v54 integration-safe runtime).

All scientific primitives are imported from the frozen framework; this module
only binds each factorial BESS/QUBO arm to the same Pegasus-P16 response domain
and converts an independently sampled control into a validation-selected BESS
placement.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np

from baselines.highs import HighsSolveConfig, solve_bess_with_highs
from dwave_backend.sampler_v53 import build_ocean_sampler
from dwave_backend.pegasus_fabric_v53 import validate_pegasus_solver_id_v53, pegasus_solver_family_v53
from experiments_dwave.application_endpoint_v38 import select_placement_on_validation
from experiments_dwave.bess_evidence import logical_feasibility, to_dimod_bqm
from experiments_dwave.benchmark_protocol import MatchedControlProtocol, MethodTrace
from experiments_dwave.integrated_bess_v54 import BESSArmProblem
from experiments_dwave.operator_phase import APPROVED_FAMILIES, load_calibration
from experiments_dwave.pegasus_control_backend_v53 import BoundPegasusResponseEvaluator, FixedPegasusControlBackend
from config.loader import load_config

class FactorialRuntimeError(RuntimeError): pass

from config.qpu_compat_v53 import qpu_config_v53

@dataclass
class ArmRuntime:
    arm: BESSArmProblem
    evaluator: BoundPegasusResponseEvaluator
    bundle: Any
    highs: Any
    backend_manifest: Mapping[str,Any]

    def close(self)->None:
        close=getattr(self.bundle,"close",None)
        if callable(close): close()


def build_arm_runtime(project_root: str|Path, arm: BESSArmProblem, *, mode:str, calibration_family:str, solver_id:str|None=None, live_qpu:bool=False, embedding_manifest:Mapping[Any,list[Any]]|None=None, frozen_embedding:Mapping[Any,list[Any]]|None=None, chain_strength:float|None=None) -> ArmRuntime:
    root=Path(project_root); mode=str(mode).lower()
    if embedding_manifest is not None and frozen_embedding is not None:
        raise FactorialRuntimeError('Specify only one of embedding_manifest or frozen_embedding')
    resolved_embedding = frozen_embedding if frozen_embedding is not None else embedding_manifest
    if calibration_family not in APPROVED_FAMILIES: raise FactorialRuntimeError("Only Advantage_system4/System6 calibrations are admissible")
    cfg_files=[root/'config'/'base.yaml',root/'config'/'case300.yaml',root/'config'/('emulator_gpu.yaml' if mode=='simulator' else 'pegasus_qpu.yaml')]
    cfg=load_config(cfg_files)
    highs=solve_bess_with_highs(arm.problem,config=HighsSolveConfig(random_seed=cfg.random.global_seed,threads=1,metadata={'role':'exact_quality_reference_v38','arm_representation':arm.representation}))
    if not highs.certified_optimal: raise FactorialRuntimeError("HiGHS reference is not certified optimal")
    elite_gap=0.02*max(1.0,abs(float(highs.combined_qubo_energy))); threshold=float(highs.combined_qubo_energy)+elite_gap
    bqm=to_dimod_bqm(arm.problem)
    if mode=='simulator':
        bundle=build_ocean_sampler('local_sqa_gpu',emulator_config=cfg.emulator,seed=cfg.random.global_seed)
    elif mode=='qpu':
        sid=validate_pegasus_solver_id_v53(str(solver_id or '').strip())
        if pegasus_solver_family_v53(sid) != calibration_family:
            raise FactorialRuntimeError('QPU family/calibration mismatch')
        qpu_cfg=qpu_config_v53(cfg.qpu,solver_id=sid,live_qpu=bool(live_qpu))
        bundle=build_ocean_sampler('pegasus_qpu',qpu_config=qpu_cfg)
    else: raise FactorialRuntimeError('mode must be simulator or qpu')
    filename,_=APPROVED_FAMILIES[calibration_family]; curve=load_calibration(root/'calibration'/filename,calibration_family)
    backend=FixedPegasusControlBackend.build(mode=mode,bundle=bundle,logical_bqm=bqm,calibration=curve,seed=cfg.random.global_seed,frozen_embedding=resolved_embedding,chain_strength=chain_strength)
    evaluator=BoundPegasusResponseEvaluator(backend=backend,order=MatchedControlProtocol().order,elite_threshold=threshold,feasibility=logical_feasibility(arm.problem),success_energy=float(highs.combined_qubo_energy),label=f'CSSF(QA) v54 {arm.representation} {mode}')
    manifest={**backend.embedding_manifest(),'mode':mode,'problem_fingerprint':arm.problem.fingerprint(),'domain_representation':arm.representation,'highs_fingerprint':highs.fingerprint(),'elite_threshold':threshold}
    return ArmRuntime(arm,evaluator,bundle,highs,manifest)



def remap_embedding_by_candidate_rank(source_arm:BESSArmProblem, target_arm:BESSArmProblem, source_embedding:Mapping[Any,list[Any]]) -> dict[Any,list[Any]]:
    """Preserve identical physical chains while relabeling BESS variables by candidate rank.

    D0/D2 and D1/D3 may contain different bus labels because the domain
    representation changes train-only candidate ranking.  The logical QUBO
    graph has the same ranked-variable cardinality; this mapping makes the
    physical embedding a controlled non-factor variable rather than silently
    allowing minorminer to choose a different chain layout per arm.
    """
    # QUBOModel exposes ``variable_order`` (not ``variables``).  The enclosing
    # BESSPlacementQUBO exposes the same canonical order and is the preferred
    # semantic API because it is tied to candidate rank.
    src=tuple(source_arm.problem.variable_order); dst=tuple(target_arm.problem.variable_order)
    src_model=tuple(source_arm.problem.model.variable_order); dst_model=tuple(target_arm.problem.model.variable_order)
    if src != src_model or dst != dst_model:
        raise FactorialRuntimeError("BESS QUBO encoding/model variable order mismatch")
    if len(src)!=len(dst):
        raise FactorialRuntimeError("Factorial arms have different QUBO cardinality")
    source_keys=set(source_embedding)
    if source_keys != set(src):
        missing=set(src)-source_keys; extra=source_keys-set(src)
        raise FactorialRuntimeError(
            f"Source embedding does not exactly cover the source factorial arm; missing={sorted(map(str,missing))}, extra={sorted(map(str,extra))}"
        )
    remapped={dst[i]:list(source_embedding[src[i]]) for i in range(len(src))}
    # Fail closed on empty/duplicate qubits inside a chain.  Cross-chain overlap
    # is checked by the backend embedding validator against the Pegasus graph.
    for var, chain in remapped.items():
        if not chain:
            raise FactorialRuntimeError(f"Remapped embedding chain is empty for {var!r}")
        if len(set(chain)) != len(chain):
            raise FactorialRuntimeError(f"Remapped embedding chain contains duplicate qubits for {var!r}")
    return remapped

def _response_placements(response: Mapping[str,Any], arm:BESSArmProblem) -> list[tuple[float,int,Any]]:
    ss=response.get('sampleset')
    if ss is None: raise FactorialRuntimeError('Production response does not expose the logical SampleSet')
    rows=[]
    for datum in ss.data(fields=['sample','energy','num_occurrences'],sorted_by='energy'):
        sample=dict(datum.sample)
        if not arm.problem.is_feasible(sample): continue
        placement=arm.problem.decode(sample)
        rows.append((float(datum.energy),int(datum.num_occurrences),placement))
    return rows



def select_placement_from_sampleset(
    project_root:str|Path, arm:BESSArmProblem, sampleset:Any, *, method:str, portfolio_size:int=12,
) -> tuple[Any,dict[str,Any]]:
    """Decode a logical BQM SampleSet and select a placement on validation AC only."""
    pool={}
    for datum in sampleset.data(fields=['sample','energy','num_occurrences'],sorted_by='energy'):
        sample=dict(datum.sample)
        if not arm.problem.is_feasible(sample):
            continue
        placement=arm.problem.decode(sample); key=tuple(map(int,placement.selected_buses))
        row=pool.setdefault(key,{'placement':placement,'best_energy':float(datum.energy),'occurrences':0})
        row['best_energy']=min(float(row['best_energy']),float(datum.energy)); row['occurrences']+=int(datum.num_occurrences)
    ranked=sorted(pool.items(),key=lambda kv:(float(kv[1]['best_energy']),-int(kv[1]['occurrences']),kv[0]))[:int(portfolio_size)]
    if not ranked: raise FactorialRuntimeError(f'{method} produced no feasible BESS placement')
    candidates={f'{method}_{i:03d}':row['placement'] for i,(_,row) in enumerate(ranked)}
    winner_id,validation=select_placement_on_validation(project_root,candidates); winner=candidates[winner_id]
    return winner,{'method':str(method),'selected_buses':list(map(int,winner.selected_buses)),'validation_selection':validation,
                   'portfolio':[{'candidate_id':f'{method}_{i:03d}','selected_buses':list(key),'best_energy':float(row['best_energy']),'occurrences':int(row['occurrences'])} for i,(key,row) in enumerate(ranked)]}


def select_placement_from_response(project_root:str|Path, arm:BESSArmProblem, response:Mapping[str,Any], *, method:str, portfolio_size:int=12) -> tuple[Any,dict[str,Any]]:
    ss=response.get('sampleset')
    if ss is None: raise FactorialRuntimeError(f'{method} response does not expose logical SampleSet')
    return select_placement_from_sampleset(project_root,arm,ss,method=method,portfolio_size=portfolio_size)


def select_production_placement_for_control(project_root:str|Path, runtime:ArmRuntime, control:Any, *, method:str, production_reads:int=8192, replicates:int=4, portfolio_size:int=12, seed:int=20260817) -> tuple[Any,dict[str,Any]]:
    """Independent production sampling for an explicitly frozen schedule."""
    c=np.asarray(control,dtype=float).reshape(-1)
    if replicates<2: raise FactorialRuntimeError('At least two independent production replicates are required')
    pool={}; rep_meta=[]
    for r in range(int(replicates)):
        response=runtime.evaluator(c,num_reads=int(production_reads),sampling_seed=int(seed)+10000*r if runtime.evaluator.backend.mode=='simulator' else None)
        rep_meta.append({'replicate':r,'num_reads':int(response.get('num_reads',production_reads)),'backend_info':dict(response.get('backend_info',{})),'operator_action':np.asarray(response['operator_action'],dtype=float).tolist()})
        for energy,occ,placement in _response_placements(response,runtime.arm):
            key=tuple(map(int,placement.selected_buses)); row=pool.setdefault(key,{'placement':placement,'best_energy':energy,'occurrences':0}); row['best_energy']=min(float(row['best_energy']),energy); row['occurrences']+=occ
    ranked=sorted(pool.items(),key=lambda kv:(float(kv[1]['best_energy']),-int(kv[1]['occurrences']),kv[0]))[:int(portfolio_size)]
    if not ranked: raise FactorialRuntimeError(f'{method} produced no feasible BESS placement')
    candidates={f'{method}_{i:03d}':row['placement'] for i,(_,row) in enumerate(ranked)}; winner_id,validation=select_placement_on_validation(project_root,candidates); winner=candidates[winner_id]
    return winner,{'selected_control':c.tolist(),'method':str(method),'production_reads_per_replicate':int(production_reads),'replicates':int(replicates),'production_replicates':rep_meta,'portfolio':[{'candidate_id':f'{method}_{i:03d}','selected_buses':list(key),'best_energy':float(row['best_energy']),'occurrences':int(row['occurrences'])} for i,(key,row) in enumerate(ranked)],'validation_selection':validation,'selected_buses':list(map(int,winner.selected_buses))}

def select_production_placement(project_root:str|Path, runtime:ArmRuntime, trace:MethodTrace, *, production_reads:int=8192, replicates:int=4, portfolio_size:int=12, seed:int=20260817) -> tuple[Any,dict[str,Any]]:
    if replicates<2: raise FactorialRuntimeError('At least two independent production replicates are required')
    best=trace.best_index('elite_probability',True); control=np.asarray(trace.controls[best],dtype=float)
    pool={}; rep_meta=[]
    for r in range(int(replicates)):
        response=runtime.evaluator(control,num_reads=int(production_reads),sampling_seed=int(seed)+10000*r)
        rep_meta.append({'replicate':r,'num_reads':int(response.get('num_reads',production_reads)),'backend_info':dict(response.get('backend_info',{})),'operator_action':np.asarray(response['operator_action'],dtype=float).tolist()})
        for energy,occ,placement in _response_placements(response,runtime.arm):
            key=tuple(map(int,placement.selected_buses)); row=pool.setdefault(key,{'placement':placement,'best_energy':energy,'occurrences':0}); row['best_energy']=min(float(row['best_energy']),energy); row['occurrences']+=occ
    ranked=sorted(pool.items(),key=lambda kv:(float(kv[1]['best_energy']),-int(kv[1]['occurrences']),kv[0]))[:int(portfolio_size)]
    if not ranked: raise FactorialRuntimeError('No feasible BESS placement was produced by the selected schedule')
    candidates={f'qa_{i:03d}':row['placement'] for i,(_,row) in enumerate(ranked)}
    winner_id,validation=select_placement_on_validation(project_root,candidates)
    winner=candidates[winner_id]
    return winner,{
        'selected_control':control.tolist(),'search_method':trace.method,'search_best_elite_probability':float(trace.responses[best]['elite_probability']),
        'production_reads_per_replicate':int(production_reads),'replicates':int(replicates),'production_replicates':rep_meta,
        'portfolio':[{'candidate_id':f'qa_{i:03d}','selected_buses':list(key),'best_energy':float(row['best_energy']),'occurrences':int(row['occurrences'])} for i,(key,row) in enumerate(ranked)],
        'validation_selection':validation,'selected_buses':list(map(int,winner.selected_buses)),
    }

def select_production_placement_for_schedule(project_root:str|Path, runtime:ArmRuntime, anneal_t_us:Any, anneal_s:Any, *, method:str, production_reads:int=8192, replicates:int=4, portfolio_size:int=12, seed:int=20260817) -> tuple[Any,dict[str,Any]]:
    """Independent production sampling for a frozen explicit anneal schedule.

    This is the schedule-form analogue of ``select_production_placement_for_control``
    and is used for comparators such as the worldline-susceptibility schedule so
    application comparison is not based on a smaller final sampling budget.
    """
    t=np.asarray(anneal_t_us,dtype=float).reshape(-1); ss=np.asarray(anneal_s,dtype=float).reshape(-1)
    if replicates<2: raise FactorialRuntimeError('At least two independent production replicates are required')
    pool={}; rep_meta=[]
    backend=runtime.evaluator.backend
    for r in range(int(replicates)):
        response=backend.evaluate_schedule(
            t,ss,num_reads=int(production_reads),elite_threshold=runtime.evaluator.elite_threshold,
            feasibility=runtime.evaluator.feasibility,success_energy=runtime.evaluator.success_energy,
            label=f'{method} production replicate {r}',
            sampling_seed=(int(seed)+10000*r if backend.mode=='simulator' else None),
        )
        rep_meta.append({'replicate':r,'num_reads':int(response.get('num_reads',production_reads)),'backend_info':dict(response.get('backend_info',{})),'operator_action':np.asarray(response['operator_action'],dtype=float).tolist()})
        for energy,occ,placement in _response_placements(response,runtime.arm):
            key=tuple(map(int,placement.selected_buses)); row=pool.setdefault(key,{'placement':placement,'best_energy':energy,'occurrences':0}); row['best_energy']=min(float(row['best_energy']),energy); row['occurrences']+=occ
    ranked=sorted(pool.items(),key=lambda kv:(float(kv[1]['best_energy']),-int(kv[1]['occurrences']),kv[0]))[:int(portfolio_size)]
    if not ranked: raise FactorialRuntimeError(f'{method} produced no feasible BESS placement')
    candidates={f'{method}_{i:03d}':row['placement'] for i,(_,row) in enumerate(ranked)}
    winner_id,validation=select_placement_on_validation(project_root,candidates); winner=candidates[winner_id]
    return winner,{
        'method':str(method),'schedule_t_us':t.tolist(),'schedule_s':ss.tolist(),
        'production_reads_per_replicate':int(production_reads),'replicates':int(replicates),'production_replicates':rep_meta,
        'portfolio':[{'candidate_id':f'{method}_{i:03d}','selected_buses':list(key),'best_energy':float(row['best_energy']),'occurrences':int(row['occurrences'])} for i,(key,row) in enumerate(ranked)],
        'validation_selection':validation,'selected_buses':list(map(int,winner.selected_buses)),
    }


__all__=['FactorialRuntimeError','ArmRuntime','build_arm_runtime','remap_embedding_by_candidate_rank','select_placement_from_sampleset','select_placement_from_response','select_production_placement_for_control','select_production_placement_for_schedule','select_production_placement']
