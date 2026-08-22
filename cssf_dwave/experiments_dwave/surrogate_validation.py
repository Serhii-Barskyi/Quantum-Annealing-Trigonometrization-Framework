"""Locked QA-response surrogate validation for the D-Wave evidence notebooks."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from benchmarks import reference_competitors as rc
from experiments_dwave.benchmark_protocol import MatchedControlProtocol, TARGET_NAMES
from experiments_dwave.cssf_control_v53 import fit_cssf_qa_response
from experiments_dwave.control_design import admissible_latin_hypercube

PARTITION_COUNTS={"train":66,"selection":10,"calibration":10,"ood":10}


def validation_design(protocol:MatchedControlProtocol)->list[dict[str,Any]]:
    """Frozen 96-control design with a boundary OOD partition.

    Train/selection/calibration occupy an interior Fourier box; OOD points are
    drawn from the full admissible box and accepted only when at least one
    coordinate lies outside the interior box.  All points remain physically
    admissible members of the same Pegasus control domain.
    """
    full=protocol.bounds.copy(); interior=full.copy()
    Tmid=0.5*(full[0,0]+full[0,1]); Thalf=0.36*(full[0,1]-full[0,0])
    interior[0]=[Tmid-Thalf,Tmid+Thalf]
    for j in range(1,full.shape[0]):
        mid=0.5*(full[j,0]+full[j,1]); half=0.34*(full[j,1]-full[j,0])
        interior[j]=[mid-half,mid+half]
    n_in=PARTITION_COUNTS['train']+PARTITION_COUNTS['selection']+PARTITION_COUNTS['calibration']
    inside=admissible_latin_hypercube(interior,n_in,order=protocol.order,seed=protocol.seed+500)
    labels=(['train']*PARTITION_COUNTS['train']+['selection']*PARTITION_COUNTS['selection']+['calibration']*PARTITION_COUNTS['calibration'])
    rng=np.random.default_rng(protocol.seed+501); ood=[];batch=0
    while len(ood)<PARTITION_COUNTS['ood']:
        cand=rc.latin_hypercube(full,64,seed=protocol.seed+600+batch);batch+=1
        for c in cand:
            boundary=bool(np.any((c<interior[:,0])|(c>interior[:,1])))
            if boundary and rc.feasible_control(c,full,order=protocol.order): ood.append(c.copy())
            if len(ood)>=PARTITION_COUNTS['ood']: break
        if batch>100: raise RuntimeError('Could not construct locked physical OOD controls')
    controls=list(inside)+ood
    return [{"control_id":f"surrogate_{i:03d}","partition":labels[i] if i<n_in else 'ood',"control":np.asarray(c,float).tolist()} for i,c in enumerate(controls)]


def _metric(y:np.ndarray,p:np.ndarray)->dict[str,float]:
    y=np.asarray(y,float);p=np.asarray(p,float)
    mse=float(np.mean((p-y)**2));mae=float(np.mean(np.abs(p-y)))
    rho=float(spearmanr(y,p).statistic) if y.size>2 else float('nan')
    k=max(1,min(5,y.size)); top=set(np.argsort(-y)[:k]); pred=set(np.argsort(-p)[:k])
    return {"mse":mse,"mae":mae,"spearman":rho,"top5_recall":float(len(top&pred)/k)}


def audit_surrogates(records:Sequence[Mapping[str,Any]],*,project_root:str,target:str='elite_probability')->dict[str,Any]:
    if len(records)!=sum(PARTITION_COUNTS.values()): raise ValueError('Surrogate audit requires the complete frozen 96-control corpus')
    part=np.asarray([str(r['partition']) for r in records]);raw=np.asarray([r['control'] for r in records],float)
    phase=np.asarray([r['response']['operator_action'] for r in records],float)
    Y=np.asarray([[float(r['response'][k]) for k in TARGET_NAMES] for r in records],float);ti=TARGET_NAMES.index(target)
    train=part=='train';cal=part=='calibration'
    cssf=fit_cssf_qa_response(phase[train],Y[train],calibration_operator_phase=phase[cal],calibration_targets=Y[cal],
                              target_names=TARGET_NAMES,project_root=project_root,support_mode='signed_axes',support_order=1,
                              metadata={'role':'locked_surrogate_validation'})
    alphas=np.logspace(-12,4,80); ridge=RidgeCV(alphas=alphas).fit(raw[train],Y[train,ti])
    gp=rc.fit_gp(rc.normalized(raw[train],np.column_stack([raw.min(0),raw.max(0)])),Y[train,ti],ard=True,random_state=20260817,n_restarts_optimizer=2)
    periodic=rc.PeriodicGP(seed=20260817).fit(phase[train],Y[train,ti])
    torus=rc.TorusSpectralMaternGP(truncation=2).fit(phase[train],Y[train,ti])
    bounds=np.column_stack([raw.min(0),raw.max(0)])
    out={"target":target,"frozen_core":True,"cssf_level":cssf.model.level.value,"cssf_lambda_gcv":float(cssf.model.lam_opt),"partitions":{}}
    for name in ('selection','calibration','ood'):
        m=part==name; actual=Y[m,ti]
        pred_cssf=cssf.predict(phase[m])[:,ti]
        pred_ridge=ridge.predict(raw[m])
        pred_gp=gp.predict(rc.normalized(raw[m],bounds))
        pred_per=periodic.predict(phase[m])
        pred_torus=torus.predict(phase[m])
        out['partitions'][name]={"CSSF":_metric(actual,pred_cssf),"Raw-Ridge":_metric(actual,pred_ridge),
                                 "Raw-GP":_metric(actual,pred_gp),"Periodic-GP":_metric(actual,pred_per),
                                 "Torus-Matern-GP":_metric(actual,pred_torus)}
    return out
