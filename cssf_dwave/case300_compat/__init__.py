"""Compatibility layer for the monograph's preserved case300 notebook.

This package exposes the narrow historical notebook API without modifying the
frozen CSSF v51 scientific source modules.  It is intentionally fail-closed:
real-QPU functions require an explicit Pegasus solver and never substitute a
simulator.
"""
from .dataset import Case300Dataset, load_dataset
from .metrics import compute_metrics
from .mpf import compute_mpf
from .qubo import (
    CandidateScreen,
    HamiltonianData,
    IsingProblem,
    QUBOProblem,
    build_hamiltonian,
    build_qubo,
    qubo_to_ising,
    screen_candidates,
    verify_ising_identity,
    verify_qubo,
)
from .qpu import (
    DWaveResult,
    compute_lsf_h_bias,
    compute_lsf_offsets,
    get_sampler,
    solve_ising_dwave,
    validate_solver_id,
)
from .validation import validation_report

__all__ = [
    "Case300Dataset", "load_dataset", "compute_metrics", "compute_mpf",
    "CandidateScreen", "HamiltonianData", "IsingProblem", "QUBOProblem",
    "screen_candidates", "build_qubo", "verify_qubo", "qubo_to_ising",
    "verify_ising_identity", "build_hamiltonian", "DWaveResult",
    "compute_lsf_h_bias", "compute_lsf_offsets", "get_sampler",
    "solve_ising_dwave", "validate_solver_id", "validation_report",
]
