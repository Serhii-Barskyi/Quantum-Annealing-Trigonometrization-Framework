"""Case300 BESS evidence assets for CSSF(QA) D-Wave experiments.

This module composes the original framework's train-only candidate selection,
BESS QUBO/Ising construction and certified HiGHS quality reference.  It does
not replace any CSSF surrogate component and contains no alternative optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from baselines.highs import HighsSolveConfig, solve_bess_with_highs
from bess.case300 import load_case300_mode_a
from bess.candidates import CandidateSelectionConfig, build_case300_fleet, select_case300_candidates
from config.loader import load_config
from opf.bess_constraints import BESSUnitSpec
from qaoa.hamiltonian import qubo_to_ising, require_qubo_ising_equivalence
from qubo.builder import build_bess_placement_qubo


@dataclass(frozen=True)
class Case300BESSEvidenceAssets:
    config: Any
    case300: Any
    selection: Any
    fleet: Any
    problem: Any
    hamiltonian: Any
    ising_audit: Any
    highs_result: Any | None
    elite_relative_gap: float
    elite_energy_threshold: float | None

    @property
    def exact_success_energy(self) -> float | None:
        if self.highs_result is None:
            return None
        return float(self.highs_result.combined_qubo_energy)

    def manifest(self) -> dict[str, Any]:
        return {
            "case": self.case300.case,
            "dataset_sha256": self.case300.source_sha256,
            "candidate_count": len(self.selection.candidate_buses),
            "bess_units": int(self.fleet.units_to_place),
            "candidate_selection_fingerprint": self.selection.fingerprint(),
            "qubo_variables": int(self.problem.model.n_variables),
            "qubo_interactions": int(self.problem.model.n_interactions),
            "qubo_fingerprint": self.problem.fingerprint(),
            "ising_max_absolute_error": float(self.ising_audit.max_absolute_error),
            "highs_certified_optimal": None if self.highs_result is None else bool(self.highs_result.certified_optimal),
            "highs_combined_qubo_energy": None if self.highs_result is None else float(self.highs_result.combined_qubo_energy),
            "elite_relative_gap": float(self.elite_relative_gap),
            "elite_energy_threshold": self.elite_energy_threshold,
        }


def build_case300_bess_assets(
    project_root: str | Path,
    *,
    solve_highs: bool = True,
    elite_relative_gap: float = 0.01,
    highs_threads: int = 1,
) -> Case300BESSEvidenceAssets:
    root=Path(project_root)
    cfg=load_config([root/'config'/'base.yaml',root/'config'/'case300.yaml',root/'config'/'emulator_gpu.yaml'])
    if cfg.qa.surrogated_system != 'quantum_annealing_response':
        raise RuntimeError('The production surrogate target must be quantum_annealing_response.')
    if cfg.qaoa.independently_surrogated or cfg.maqaoa.independently_surrogated:
        raise RuntimeError('QAOA and MA-QAOA are decomposition/reference levels, not independent production surrogate targets.')
    if cfg.benchmark.compare_wall_clock:
        raise RuntimeError('HiGHS is an exact quality reference; wall-clock competition is outside its declared role.')
    if cfg.emulator.allow_classical_fallback:
        raise RuntimeError('The Pegasus simulator experiment prohibits CPU/classical fallback.')

    case300=load_case300_mode_a(root/'data'/'case300_full_modeA_Barskyi_Serhii.json')
    selection_cfg=CandidateSelectionConfig.from_qubo_config(cfg.qubo)
    selection=select_case300_candidates(case300,selection_cfg)
    fleet=build_case300_fleet(
        selection,bess_units=cfg.qubo.bess_units,
        unit=BESSUnitSpec(power_mw=25.0,energy_mwh=100.0),
        metadata={'strategy':'case300_lsf_train_only'},
    )
    linear_by_bus={bus:-float(selection.scores[i]) for i,bus in enumerate(selection.candidate_buses)}
    problem=build_bess_placement_qubo(
        fleet,linear_by_bus=linear_by_bus,
        metadata={'candidate_selection_fingerprint':selection.fingerprint(),'strategy':'CSSF_QA_LSF'},
    )
    hamiltonian=qubo_to_ising(problem.model)
    ising_audit=require_qubo_ising_equivalence(
        problem.model,hamiltonian,exact_limit=18,random_samples=8192,
        seed=cfg.random.global_seed,tolerance=cfg.qubo.verify_qubo_ising_tolerance,
    )
    highs_result=None
    threshold=None
    if solve_highs:
        highs_result=solve_bess_with_highs(problem,config=HighsSolveConfig(random_seed=cfg.random.global_seed,threads=int(highs_threads)))
        if not highs_result.certified_optimal:
            raise RuntimeError('HiGHS quality reference must be certified optimal.')
        gap=float(elite_relative_gap)*max(1.0,abs(float(highs_result.combined_qubo_energy)))
        threshold=float(highs_result.combined_qubo_energy)+gap
    return Case300BESSEvidenceAssets(cfg,case300,selection,fleet,problem,hamiltonian,ising_audit,highs_result,float(elite_relative_gap),threshold)


def to_dimod_bqm(problem: Any) -> Any:
    try:
        import dimod
    except Exception as exc:  # pragma: no cover - installed in target Colab environment
        raise RuntimeError('dimod/D-Wave Ocean is required for the annealing experiment.') from exc
    return dimod.BinaryQuadraticModel.from_qubo(problem.model.to_qubo_dict(),offset=float(problem.model.offset))


def logical_feasibility(problem: Any):
    def predicate(sample: Mapping[Any,int]) -> bool:
        return bool(problem.is_feasible(sample))
    return predicate
