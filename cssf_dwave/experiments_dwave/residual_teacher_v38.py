"""Physical teacher data for the v38 CSSF residual hierarchy.

The teacher stack is aligned on the same annealing controls and target vector:
application/QUBO ideal -> QAOA -> physical MA-QAOA -> digitized QA/SQA -> QPU.
QAOA and MA-QAOA are reference/decomposition levels only.  The final learned
quantity remains the annealer response used by CSSF schedule selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

from benchmarks import reference_competitors as rc
from core.types import SurrogateLevel
from experiments_dwave.evidence_v38 import canonical_json_hash
from experiments_dwave.operator_phase import operator_action_coordinates
from qa.schedule import AnnealingSchedule


class ResidualTeacherError(RuntimeError):
    pass


TARGET_NAMES = ("mean_energy", "feasibility_probability")


def calibrated_schedule_projection(control: Sequence[float], calibration: Any, *, order: int = 8, trotter_slices: int = 2) -> tuple[AnnealingSchedule, Any, np.ndarray]:
    """Map one admissible Fourier D-Wave schedule through frozen A(s),B(s).

    The normalized driver/problem amplitudes are derived from the physical
    calibration and integrated by the frozen QA->MA-QAOA digitizer.
    """
    c=np.asarray(control,dtype=float).reshape(-1)
    t_us,s=rc.fourier_forward_schedule(c,order=int(order),grid_points=129,reject_nonmonotone=True)
    A=np.interp(s,np.asarray(calibration.s,float),np.asarray(calibration.A_GHz,float))
    B=np.interp(s,np.asarray(calibration.s,float),np.asarray(calibration.B_GHz,float))
    a_scale=max(float(np.max(np.abs(A))),1e-15); b_scale=max(float(np.max(np.abs(B))),1e-15)
    driver=np.clip(A/a_scale,0.0,None); problem=np.clip(B/b_scale,0.0,None)
    schedule=AnnealingSchedule(
        normalized_time=np.asarray(t_us,float)/float(t_us[-1]),
        driver_amplitudes=driver,
        problem_amplitudes=problem,
        total_annealing_time=float(t_us[-1]),
        name="calibration_resolved_pegasus_forward_qa",
        metadata={"family":str(calibration.family),"calibration_sha256":str(calibration.source_sha256),"source":"A(s),B(s)"},
    )
    digitized=schedule.digitize(trotter_slices=int(trotter_slices),trotter_order=2)
    phase=operator_action_coordinates(t_us,s,calibration,n_segments=8)
    return schedule,digitized,phase


def _counts_metrics(counts: Mapping[str,int], model: Any, *, units_to_place: int) -> dict[str,float]:
    samples=[]; occurrences=[]
    for bits,count in counts.items():
        clean=str(bits).replace(" ","")
        x=np.asarray([int(b) for b in clean[::-1]],dtype=np.int8)
        if x.size!=int(model.n_variables):
            raise ResidualTeacherError("Qiskit bitstring dimension differs from BESS QUBO")
        samples.append(x); occurrences.append(int(count))
    if not samples: raise ResidualTeacherError("Teacher circuit returned no samples")
    X=np.asarray(samples,dtype=np.int8); occ=np.asarray(occurrences,dtype=np.int64); prob=occ/float(occ.sum())
    energy=np.asarray(model.energies(X),dtype=float); feasible=np.sum(X,axis=1)==int(units_to_place)
    mean=float(np.dot(prob,energy)); return {"mean_energy":mean,"feasibility_probability":float(np.sum(prob[feasible]))}


def qaoa_maqaoa_teacher(control: Sequence[float], *, calibration: Any, hamiltonian: Any, qubo_model: Any, units_to_place: int, seed: int, shots: int = 4096, order: int = 8, trotter_slices: int = 2) -> dict[str,Any]:
    """Run tied QAOA and physical MA-QAOA references on Aer GPU tensor network."""
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qaoa.circuit import QAOACircuitConfig,QAOAParameterValues,build_parameterized_qaoa_circuit
        from maqaoa.circuit import MAQAOACircuitConfig,build_parameterized_maqaoa_circuit
    except Exception as exc:  # pragma: no cover - target environment
        raise ResidualTeacherError("Qiskit Aer GPU dependencies are required") from exc
    _,digitized,phase=calibrated_schedule_projection(control,calibration,order=order,trotter_slices=trotter_slices)
    backend=AerSimulator(method="tensor_network",device="GPU",precision="double",tensor_network_num_sampling_qubits=10,use_cuTensorNet_autotuning=True,shot_branching_enable=True,shot_branching_sampling_enable=True)
    if "GPU" not in set(backend.available_devices()) or "tensor_network" not in set(backend.available_methods()):
        raise ResidualTeacherError("Aer GPU tensor_network backend is unavailable")
    q_art=build_parameterized_qaoa_circuit(hamiltonian,QAOACircuitConfig(repetitions=int(trotter_slices),barrier_policy="between_layers"))
    q_values=QAOAParameterValues(gamma=digitized.problem_integrals[:,0],beta=digitized.driver_integrals[:,0])
    m_art=build_parameterized_maqaoa_circuit(hamiltonian,MAQAOACircuitConfig(repetitions=int(trotter_slices),barrier_policy="between_layers"))
    m_values=digitized.to_maqaoa_values(m_art.plan.parameter_layout)
    def run(circuit:Any,run_seed:int)->dict[str,float]:
        measured=circuit.copy(); measured.measure_all(); compiled=transpile(measured,backend,optimization_level=1,seed_transpiler=int(run_seed))
        result=backend.run(compiled,shots=int(shots),seed_simulator=int(run_seed)).result()
        if not result.success: raise ResidualTeacherError("Aer GPU teacher execution failed")
        return _counts_metrics(result.get_counts(compiled),qubo_model,units_to_place=units_to_place)
    q=run(q_art.bind(q_values),int(seed)); m=run(m_art.bind(m_values),int(seed))
    return {"qaoa":q,"ma_qaoa":m,"operator_action":phase.tolist(),"shots":int(shots),"schedule_fingerprint":canonical_json_hash(np.asarray(control,float).tolist(),prefix="CSSF-v38-residual-control")}


def collect_residual_reference_stack(
    controls: Sequence[Sequence[float]], *, evaluator: Any, calibration: Any, arm: Any,
    ideal_energy: float, mode: str, seed: int = 20260817, shots: int = 4096,
    qpu_evaluator: Any | None = None, cached_digitized: Mapping[str,Mapping[str,Any]] | None = None,
    cached_hardware: Mapping[str,Mapping[str,Any]] | None = None,
) -> tuple[dict[SurrogateLevel,np.ndarray],dict[SurrogateLevel,np.ndarray],dict[str,Any]]:
    """Collect aligned residual teachers for full CSSF hierarchy.

    ``evaluator`` is the digitized-QA/SQA or live-QPU response evaluator.  In
    QPU mode a separate digitized simulator evaluator must be supplied through
    ``qpu_evaluator`` only when the main evaluator is the live QPU; this keeps
    the hardware residual explicitly separated from digitized-QA.
    """
    mode=str(mode).lower(); rows=[]; features=[]; refs={level:[] for level in (SurrogateLevel.OPF,SurrogateLevel.QAOA,SurrogateLevel.MA_QAOA,SurrogateLevel.DIGITIZED_QA,SurrogateLevel.HARDWARE_RESIDUAL)}
    def key(control:np.ndarray)->str:
        return canonical_json_hash(np.asarray(control,float).round(12).tolist(),prefix="CSSF-v38-residual-control-key")
    cached_digitized={} if cached_digitized is None else dict(cached_digitized); cached_hardware={} if cached_hardware is None else dict(cached_hardware)
    for i,c0 in enumerate(controls):
        c=np.asarray(c0,dtype=float); ck=key(c); _,_,phase=calibrated_schedule_projection(c,calibration,order=8,trotter_slices=2); features.append(phase)
        teacher=qaoa_maqaoa_teacher(c,calibration=calibration,hamiltonian=arm.hamiltonian,qubo_model=arm.problem.model,units_to_place=arm.fleet.units_to_place,seed=int(seed)+i,shots=int(shots))
        opf=[float(ideal_energy),1.0]; refs[SurrogateLevel.OPF].append(opf); refs[SurrogateLevel.QAOA].append([teacher["qaoa"]["mean_energy"],teacher["qaoa"]["feasibility_probability"]]); refs[SurrogateLevel.MA_QAOA].append([teacher["ma_qaoa"]["mean_energy"],teacher["ma_qaoa"]["feasibility_probability"]])
        if mode=="simulator":
            d=dict(cached_digitized[ck]) if ck in cached_digitized else evaluator(c,num_reads=2048,sampling_seed=int(seed)+1000+i); refs[SurrogateLevel.DIGITIZED_QA].append([d["mean_energy"],d["feasibility_probability"]])
        elif mode=="qpu":
            if qpu_evaluator is None and ck not in cached_digitized: raise ResidualTeacherError("QPU hierarchy requires digitized-QA evidence for hardware residual separation")
            d=dict(cached_digitized[ck]) if ck in cached_digitized else qpu_evaluator(c,num_reads=2048,sampling_seed=int(seed)+1000+i)
            h=dict(cached_hardware[ck]) if ck in cached_hardware else evaluator(c,num_reads=2048)
            refs[SurrogateLevel.DIGITIZED_QA].append([d["mean_energy"],d["feasibility_probability"]]); refs[SurrogateLevel.HARDWARE_RESIDUAL].append([h["mean_energy"],h["feasibility_probability"]])
        else: raise ResidualTeacherError("mode must be simulator or qpu")
        rows.append({"sample_id":f"residual:{i:04d}","control":c.tolist(),"operator_action":phase.tolist(),"teacher":teacher})
    X=np.asarray(features,dtype=float).astype(np.complex128)
    levels=(SurrogateLevel.OPF,SurrogateLevel.QAOA,SurrogateLevel.MA_QAOA,SurrogateLevel.DIGITIZED_QA)+( (SurrogateLevel.HARDWARE_RESIDUAL,) if mode=="qpu" else () )
    features_by_level={level:X.copy() for level in levels}; refs_by_level={level:np.asarray(refs[level],dtype=float) for level in levels}
    return features_by_level,refs_by_level,{"schema":"CSSF-RESIDUAL-TEACHER-STACK-v38","mode":mode,"rows":rows,"target_names":list(TARGET_NAMES),"levels":[x.value for x in levels]}


__all__=["ResidualTeacherError","TARGET_NAMES","calibrated_schedule_projection","qaoa_maqaoa_teacher","collect_residual_reference_stack"]
