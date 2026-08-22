"""Residual-hierarchy refinement for the claim-grade CSSF-full arm (v38)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from benchmarks import reference_competitors as rc
from core.types import SurrogateLevel
from experiments_dwave.benchmark_protocol import MatchedControlProtocol
from experiments_dwave.residual_program_v38 import FittedResidualHierarchy


class HierarchicalRefinementError(RuntimeError): pass


@dataclass(frozen=True)
class HierarchicalProposal:
    control: np.ndarray
    predicted_mean_energy: float
    predicted_feasibility_probability: float
    score: float
    diagnostics: Mapping[str,Any]


def propose_with_full_residual_hierarchy(
    hierarchy:FittedResidualHierarchy,
    evaluator:Any,
    protocol:MatchedControlProtocol,
    *,
    candidate_pool_size:int=4096,
    seed:int=20260817,
    minimum_predicted_feasibility:float=0.50,
)->HierarchicalProposal:
    """Select a new D3 schedule from the full additive hierarchy prediction."""
    rng_seed=int(seed)+37061
    pool=rc.latin_hypercube(protocol.bounds,int(candidate_pool_size),seed=rng_seed)
    pure=getattr(evaluator,"operator_action",None)
    if pure is None or not callable(pure): raise HierarchicalRefinementError("evaluator.operator_action(control) is required")
    controls=[]; phase=[]
    for c in pool:
        try:
            x=np.asarray(pure(np.asarray(c,float)),dtype=float).reshape(-1)
        except Exception:
            continue
        if np.isfinite(x).all(): controls.append(np.asarray(c,float)); phase.append(x)
    if not controls: raise HierarchicalRefinementError("No feasible candidate schedule survived the control gate")
    X=np.asarray(phase,dtype=float).astype(np.complex128)
    features={level:X for level in hierarchy.chain.levels}
    pred=np.asarray(hierarchy.chain.predict(features),dtype=float)
    if pred.shape[1]!=2: raise HierarchicalRefinementError("Hierarchy must predict [mean_energy, feasibility_probability]")
    energy=pred[:,0]; feas=pred[:,1]
    e_scale=max(float(np.std(energy)),1e-12); e_center=float(np.median(energy))
    score=-(energy-e_center)/e_scale + 2.0*np.clip(feas,0.0,1.0)
    admissible=np.flatnonzero(feas>=float(minimum_predicted_feasibility))
    if admissible.size:
        idx=int(admissible[np.argmax(score[admissible])])
    else:
        idx=int(np.lexsort((energy,-feas))[0])
    control=np.asarray(controls[idx],dtype=float)
    return HierarchicalProposal(control,float(energy[idx]),float(feas[idx]),float(score[idx]),{
        "schema":"CSSF-HIERARCHICAL-PROPOSAL-v38","candidate_pool_size_requested":int(candidate_pool_size),
        "candidate_pool_size_feasible":int(len(controls)),"seed":rng_seed,"minimum_predicted_feasibility":float(minimum_predicted_feasibility),
        "hierarchy_levels":[x.value for x in hierarchy.chain.levels],"final_predictor_used":True,
        "selected_index":idx,
    })


def rank_observed_controls_with_full_hierarchy(
    hierarchy:FittedResidualHierarchy, evaluator:Any, controls:np.ndarray, *, minimum_predicted_feasibility:float=0.50
)->HierarchicalProposal:
    """Rerank already-observed D3 controls; this consumes no extra annealer query."""
    C=np.asarray(controls,dtype=float)
    pure=getattr(evaluator,"operator_action",None)
    if C.ndim!=2 or C.shape[0]<2 or pure is None or not callable(pure):
        raise HierarchicalRefinementError("Aligned observed controls and evaluator.operator_action are required")
    X=np.asarray([np.asarray(pure(c),dtype=float) for c in C],dtype=float).astype(np.complex128)
    pred=np.asarray(hierarchy.chain.predict({level:X for level in hierarchy.chain.levels}),dtype=float)
    energy=pred[:,0]; feas=pred[:,1]; e_scale=max(float(np.std(energy)),1e-12); e_center=float(np.median(energy))
    score=-(energy-e_center)/e_scale+2.0*np.clip(feas,0.0,1.0); admissible=np.flatnonzero(feas>=float(minimum_predicted_feasibility))
    idx=int(admissible[np.argmax(score[admissible])]) if admissible.size else int(np.lexsort((energy,-feas))[0])
    return HierarchicalProposal(C[idx].copy(),float(energy[idx]),float(feas[idx]),float(score[idx]),{
        "schema":"CSSF-HIERARCHICAL-OBSERVED-RERANK-v38","observed_controls":int(C.shape[0]),"selected_index":idx,
        "minimum_predicted_feasibility":float(minimum_predicted_feasibility),"hierarchy_levels":[x.value for x in hierarchy.chain.levels],
        "final_predictor_used":True,"additional_annealer_queries":0,
    })


__all__=["HierarchicalRefinementError","HierarchicalProposal","propose_with_full_residual_hierarchy","rank_observed_controls_with_full_hierarchy"]
