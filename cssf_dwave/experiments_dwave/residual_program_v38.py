"""Versioned full CSSF residual hierarchy experiment utilities.

The frozen hierarchy remains OPF -> QAOA -> MA-QAOA -> digitized-QA ->
hardware residual.  QAOA/MA-QAOA remain teacher/decomposition levels; this
module does not promote them to production optimizers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.csnn_t_adapter import CSNNTSurrogateModel, fit_csnn_t_surrogate
from core.dataset import CSSFDataset
from core.types import SurrogateLevel
from spectral.residual_surrogate import ResidualComponent, ResidualSurrogateChain, build_residual_dataset
from experiments_dwave.evidence_v38 import array_hash, canonical_json_hash


class ResidualProgramError(RuntimeError): pass


LEVELS=(SurrogateLevel.OPF,SurrogateLevel.QAOA,SurrogateLevel.MA_QAOA,SurrogateLevel.DIGITIZED_QA,SurrogateLevel.HARDWARE_RESIDUAL)


@dataclass(frozen=True)
class FittedResidualHierarchy:
    chain: ResidualSurrogateChain
    models: Mapping[SurrogateLevel,CSNNTSurrogateModel]
    training_features: Mapping[SurrogateLevel,np.ndarray]
    reference_targets: Mapping[SurrogateLevel,np.ndarray]
    contribution_arrays: Mapping[SurrogateLevel,np.ndarray]
    final_prediction: np.ndarray
    evidence: Mapping[str,Any]


def _validate_inputs(features_by_level: Mapping[SurrogateLevel,np.ndarray], references_by_level: Mapping[SurrogateLevel,np.ndarray], levels: Sequence[SurrogateLevel])->tuple[int,int]:
    n=None; p=None
    for level in levels:
        if level not in features_by_level or level not in references_by_level: raise ResidualProgramError(f"Missing data for {level.value}")
        X=np.asarray(features_by_level[level]); y=np.asarray(references_by_level[level],dtype=float)
        if X.ndim!=2 or y.ndim!=2 or X.shape[0]!=y.shape[0] or X.shape[0]==0: raise ResidualProgramError(f"Invalid aligned matrices for {level.value}")
        if not np.isfinite(X.real).all() or not np.isfinite(X.imag if np.iscomplexobj(X) else X).all() or not np.isfinite(y).all(): raise ResidualProgramError(f"Non-finite data for {level.value}")
        if n is None: n,p=X.shape[0],y.shape[1]
        if X.shape[0]!=n or y.shape!=(n,p): raise ResidualProgramError("All hierarchy levels must share aligned samples and target dimension")
    return int(n),int(p)


def fit_residual_hierarchy(
    project_root: str|Path,
    features_by_level: Mapping[SurrogateLevel,np.ndarray],
    references_by_level: Mapping[SurrogateLevel,np.ndarray],
    *,
    target_names: Sequence[str],
    mode: str="simulator",
    sample_ids: Sequence[str]|None=None,
    n_lambdas: int=100,
    lam_range: tuple[float,float]=(-12.0,4.0),
    source_evidence_ids: Mapping[SurrogateLevel,str]|None=None,
) -> FittedResidualHierarchy:
    mode=str(mode).lower(); levels=LEVELS if mode=="qpu" else LEVELS[:-1]
    n,p=_validate_inputs(features_by_level,references_by_level,levels)
    names=tuple(map(str,target_names))
    if len(names)!=p: raise ResidualProgramError("target_names length mismatch")
    ids=tuple(sample_ids) if sample_ids is not None else tuple(f"hierarchy_{i:06d}" for i in range(n))
    if len(ids)!=n or len(set(ids))!=n: raise ResidualProgramError("sample_ids must be unique and aligned")
    models={}; components=[]; contributions={}; accumulated=np.zeros((n,p),dtype=float); evidence_components=[]
    sources={} if source_evidence_ids is None else dict(source_evidence_ids)
    for idx,level in enumerate(levels):
        X=np.asarray(features_by_level[level],dtype=np.complex128); y=np.asarray(references_by_level[level],dtype=float)
        if idx==0:
            ds=CSSFDataset(X,y,sample_ids=ids,metadata={"target_semantics":"absolute_OPF","surrogate_level":level.value,"contract":"v38"})
            target_definition="absolute_OPF_reference"
        else:
            ds=build_residual_dataset(X,y,accumulated,level=level,baseline_level=levels[idx-1],sample_ids=ids,metadata={"contract":"v38","teacher_role":level.value})
            target_definition=f"{level.value}_reference_minus_accumulated_{levels[idx-1].value}"
        model=fit_csnn_t_surrogate(ds,case="case300",level=level,target_names=names,n_lambdas=n_lambdas,lam_range=lam_range,metadata={"hierarchy":"full_CSSF_v38","production_optimizer":False if level in {SurrogateLevel.QAOA,SurrogateLevel.MA_QAOA} else True},project_root=project_root)
        pred=model.predict(X); accumulated=np.ascontiguousarray(accumulated+pred,dtype=float)
        models[level]=model; contributions[level]=pred
        components.append(ResidualComponent(level=level,predictor=model,metadata={"teacher_or_target":level.value}))
        h=hashlib.sha256(np.ascontiguousarray(model.H).tobytes(order="C")).hexdigest()
        evidence_components.append({
            "level":level.value,"source_evidence_id":sources.get(level,f"source:{level.value}"),"training_partition_fingerprint":ds.fingerprint(),
            "target_definition":target_definition,"model_hash":h,"contribution_hash":array_hash(pred,prefix=f"CSSF-residual-{level.value}"),
            "enters_final_predictor":True,"lambda_gcv":float(model.lam_opt),"call_path":"core.csnn_t_adapter.fit_csnn_t_surrogate",
        })
    chain=ResidualSurrogateChain(components,target_names=names,metadata={"contract":"v38","mode":mode,"qaoa_maqaoa_role":"teacher/decomposition; not independent production optimizers"})
    # Exercise the frozen chain itself, not only a manual sum.
    decomp=chain.decompose({level:features_by_level[level] for level in levels})
    if not np.allclose(decomp.total,accumulated,rtol=1e-12,atol=1e-12): raise ResidualProgramError("Frozen residual-chain composition disagrees with sequential fit")
    evidence={"schema":"CSSF-RESIDUAL-HIERARCHY-v38","mode":mode,"components":evidence_components,"final_prediction_hash":array_hash(decomp.total,prefix="CSSF-residual-final"),"levels":[x.value for x in levels],"sample_ids_hash":canonical_json_hash(list(ids),prefix="CSSF-residual-samples")}
    return FittedResidualHierarchy(chain,models,{k:np.asarray(v) for k,v in features_by_level.items()},{k:np.asarray(v) for k,v in references_by_level.items()},contributions,decomp.total,evidence)


__all__=["ResidualProgramError","LEVELS","FittedResidualHierarchy","fit_residual_hierarchy"]
