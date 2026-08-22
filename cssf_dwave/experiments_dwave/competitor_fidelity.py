"""Executable reference-reproduction checks for full benchmark competitors.

The checks are deliberately small and tractable.  Their purpose is to prove
that every named algorithm executes its defining state, surrogate, acquisition,
adaptation and accounting mechanisms before a costly Pegasus campaign.  They
are not substitutes for the production BESS benchmark and never generate a
commercial superiority claim by themselves.
"""
from __future__ import annotations

from typing import Any
import importlib.util
import math
import numpy as np

from benchmarks import reference_competitors as rc
from experiments_dwave.control_design import admissible_latin_hypercube


def _harmonic_merit(control: np.ndarray) -> float:
    c=np.asarray(control,float)
    return float(0.5 + 0.15*np.cos(0.12*c[0]) + np.sum(0.05*np.cos(np.arange(1,c.size)*c[1:])))


def _binary_terminal(value: float) -> bool:
    return bool(float(value) >= 0.55)


def run_reference_reproduction_suite(seed:int=20260817, *, run_heavy_finzgar: bool = False, run_heavy_qzero: bool = False)->dict[str,Any]:
    results:dict[str,Any]={}
    # Sequential GP+EI.
    b=rc.fourier_bounds(2,(5.0,50.0),0.2); feasible=lambda x:rc.feasible_control(x,b,order=2)
    gp=rc.SequentialGPEI(b,minimize_objective=False,seed=seed,n_restarts_optimizer=1)
    init=admissible_latin_hypercube(b,8,order=2,seed=seed)
    for x in init: gp.add(x,_harmonic_merit(x))
    x,d=gp.suggest(feasibility=feasible);gp.add(x,_harmonic_merit(x))
    results['GP+EI-full']={'pass':bool(feasible(x) and len(gp.X)==9),'diagnostics':d}

    # Finzgar paper lifecycle is 10 initialization queries plus 50 sequential GP/UCB
    # updates.  It is not reduced for local convenience.  The full reproduction
    # is optional because repeated GP hyperparameter optimization is intentionally
    # expensive on the local validation host.
    if run_heavy_finzgar:
        fin=rc.FinzgarBO(b,seed=seed,total_paper_iterations=50)
        linear=rc.canonical_control(2,20.0);design=fin.initial_design(linear,feasibility=feasible)
        for x0 in design: fin.add(x0,_harmonic_merit(x0))
        kappas=[]
        for iteration in range(10,60):
            x0,d0=fin.suggest(iteration=iteration,feasibility=feasible);fin.add(x0,_harmonic_merit(x0));kappas.append(float(d0['kappa']))
        passed=bool(len(fin.X)==60 and abs(rc.finzgar_kappa(0)-2.0)<1e-12 and abs(rc.finzgar_kappa(50)-0.01)<1e-12)
        results['Finzgar-BO-paper']={'pass':passed,'status':'PASS' if passed else 'FAIL',
                                     'queries':len(fin.X),'kappa_first':kappas[0],'kappa_last':kappas[-1]}
        results['Finzgar-BO-matched-full']=dict(results['Finzgar-BO-paper'])
    else:
        # Execute its exact initialization and schedule semantics, but do not
        # grant paper-fidelity PASS without all 50 sequential updates.
        fin=rc.FinzgarBO(b,seed=seed,total_paper_iterations=50)
        linear=rc.canonical_control(2,20.0);design=fin.initial_design(linear,feasibility=feasible)
        for x0 in design: fin.add(x0,_harmonic_merit(x0))
        results['Finzgar-BO-paper']={
            'pass':False,'status':'NOT_RUN_LOCAL_RESOURCE',
            'initial_queries':len(fin.X),'kappa_initial':rc.finzgar_kappa(0),'kappa_final':rc.finzgar_kappa(50),
            'reason':'The complete 10+50 sequential GP/UCB paper lifecycle is retained in production code and is not shortened for local fidelity PASS.',
        }
        results['Finzgar-BO-matched-full']=dict(results['Finzgar-BO-paper'])

    # Jeong-TuRBO: persistent state, EI, adaptive reads, runtime guard, update and restart-capable lifecycle.
    tb=rc.fourier_bounds(8,(5.0,50.0),0.30); f8=lambda x:rc.feasible_control(x,tb,order=8)
    turbo=rc.JeongTuRBO(tb,order=8,seed=seed,minimize_objective=False)
    init8=admissible_latin_hypercube(tb,12,order=8,seed=seed)
    for x in init8: turbo.add_initial(x,_harmonic_merit(x))
    before=turbo.state.iteration;x,d=turbo.suggest(feasibility=f8);reads=turbo.adaptive_reads(x,min_reads=250,max_reads=900);obs=turbo.observe(x,_harmonic_merit(x))
    guard=rc.QPURuntimeBudget(total_us=1e9);reads,reserved=guard.allocate(float(x[0]),int(reads),min_reads=250)
    t,s=rc.fourier_forward_schedule(x,order=8,grid_points=129,reject_nonmonotone=False)
    results['TuRBO-paper']={'pass':bool(turbo.state.iteration==before+1 and 250<=reads<=900 and reserved>0 and len(t)==129),
                            'diagnostics':{**d,**obs,'adaptive_reads':int(reads)}}
    results['TuRBO-matched-full']=dict(results['TuRBO-paper'])

    # Periodic and torus/Riemannian GP models.
    rng=np.random.default_rng(seed);theta=rng.uniform(-math.pi,math.pi,size=(36,3));y=np.cos(theta[:,0])+0.2*np.sin(theta[:,1]-theta[:,2])
    per=rc.PeriodicGP(seed=seed).fit(theta,y);p1=per.predict(theta[:5]);p2=per.predict(theta[:5]+2*math.pi)
    tor=rc.TorusSpectralMaternGP(truncation=2).fit(theta,y);q1=tor.predict(theta[:5]);q2=tor.predict(theta[:5]+2*math.pi)
    results['Periodic-GP']={'pass':bool(np.max(np.abs(p1-p2))<1e-8)}
    results['Torus-Riemannian-Matern-GP']={'pass':bool(np.max(np.abs(q1-q2))<1e-8)}

    # Actual worldline susceptibility -> cumulative allocation -> inverse schedule.
    h=np.asarray([0.1,-0.1,0.0]);J=np.asarray([[0,-.5,0],[-.5,0,-.3],[0,-.3,0]],float)
    wc=rc.WorldlineSQAConfig(beta=1.5,replicas=8,burn_in_sweeps=8,measurement_sweeps=16,thin=2,seed=seed)
    w=rc.construct_worldline_schedule(h,J,s_grid=np.linspace(0,1,7),A_of_s=lambda s:1-s+0.05,B_of_s=lambda s:s,T_us=20,chi0=1e-4,config=wc)
    results['Worldline-Susceptibility-full']={'pass':bool(np.all(np.asarray(w['chi'])>=0) and np.all(np.diff(w['schedule_s'])>=-1e-12))}

    # QZero paper arm is intentionally heavy at the published M=5, l=0.2, delta=0.01
    # action discretization.  A local mechanism test must never weaken that paper arm
    # merely to finish quickly.  The full paper reproduction therefore runs only when
    # explicitly requested; otherwise it remains fail-closed and its separate unit
    # tests cover the policy/value/MCTS mechanics without conferring fidelity PASS.
    if run_heavy_qzero:
        space=rc.FourierDiscreteActionSpace(order=5,l=0.2,delta=0.01)
        qz=rc.QZero(space,context_dim=2,T_us=70.0,seed=seed,C_start=3.0,C_end=0.5,Nplayout=6,epsilon=0.01,maximize_merit=True)
        contexts=[np.asarray([0.2,-0.1]),np.asarray([-0.3,0.4])]
        def env_factory(shift:float):
            return lambda c: float(0.5+0.2*np.cos(np.sum(np.asarray(c)[1:])+shift))
        envs=[env_factory(0.0),env_factory(0.4)]
        X,P,V=qz.build_pretraining_data(contexts,envs,mcts_episodes=1);loss=qz.pretrain(X,P,V,epochs=2)
        best,diag=qz.fine_tune(contexts[0],envs[0],terminal_success=_binary_terminal,episodes=1,epochs_per_round=2)
        arch=qz.net.architecture_manifest()
        results['QZero-paper']={'pass':bool(arch['separate_policy_value_networks'] and qz.pretraining_queries>0 and diag['target_queries']>0 and diag['terminal_value_semantics']=='binary_pm1'),
                                'status':'PASS' if arch['separate_policy_value_networks'] and qz.pretraining_queries>0 and diag['target_queries']>0 else 'FAIL',
                                'pretraining_queries':qz.pretraining_queries,'target_queries':diag['target_queries'],'loss_final':loss[-1]}
        results['QZero-matched-full']=dict(results['QZero-paper'])
    else:
        results['QZero-paper']={
            'pass':False,
            'status':'NOT_RUN_LOCAL_RESOURCE',
            'reason':'Published QZero action discretization M=5, l=0.2, delta=0.01 is retained; no reduced proxy is accepted for paper-fidelity PASS.',
        }
        results['QZero-matched-full']={
            'pass':False,
            'status':'NOT_RUN_PRODUCTION_CORPUS',
            'reason':'Matched full QZero remains locked until the provenance-bearing pretraining/fine-tuning corpus and hidden-query ledger are supplied.',
        }

    # Strong classical wrappers can be reproduced only when Ocean samplers are installed.
    try:
        samplers_available=importlib.util.find_spec('dwave.samplers') is not None
    except ModuleNotFoundError:
        samplers_available=False
    results['Strong-SA']={'pass':False,'status':'NOT_RUN_DEPENDENCY' if not samplers_available else 'AVAILABLE_FOR_BQM_REPRODUCTION'}
    results['Strong-Tabu']={'pass':False,'status':'NOT_RUN_DEPENDENCY' if not samplers_available else 'AVAILABLE_FOR_BQM_REPRODUCTION'}

    highs_available=importlib.util.find_spec('highspy') is not None
    results['HiGHS-quality-reference']={'pass':False,
                                        'status':'AVAILABLE_NOT_EXECUTED' if highs_available else 'NOT_RUN_DEPENDENCY'}
    return {'registry_hash':rc.REGISTRY_HASH,'algorithm_hash':rc.ALGORITHM_HASH,'results':results}


__all__=['run_reference_reproduction_suite']
