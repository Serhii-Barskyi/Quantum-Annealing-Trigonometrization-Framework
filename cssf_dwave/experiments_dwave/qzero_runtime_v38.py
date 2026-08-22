"""Full matched QZero runtime for the v38 D-Wave evidence program.

The implementation deliberately consumes a provenance-bearing pretraining
corpus.  It never fabricates or downsizes the 45+ source contexts.  Target
adaptation uses the full policy/value/MCTS implementation in
``benchmarks.reference_competitors.QZero`` with the paper hyperparameters and
matched control box.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from benchmarks import reference_competitors as rc
from experiments_dwave.benchmark_protocol import MatchedControlProtocol, MethodTrace
from experiments_dwave.evidence_v38 import qpu_access_time_us


class QZeroRuntimeError(RuntimeError):
    pass


CONTEXT_ENCODING = "normalized_ising_h_plus_upper_J_v38"


def ising_context(hamiltonian: Any) -> np.ndarray:
    h = np.asarray(hamiltonian.linear_z, dtype=float).reshape(-1)
    J = np.asarray(hamiltonian.quadratic_zz, dtype=float)
    iu = np.triu_indices(J.shape[0], k=1)
    x = np.r_[h, J[iu]]
    scale = max(1.0, float(np.max(np.abs(x))))
    return np.ascontiguousarray(x / scale, dtype=float)


def load_pretraining_corpus(path: str | Path, *, expected_context_dim: int, action_dim: int, input_dim: int) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise QZeroRuntimeError(f"Full QZero pretraining corpus is absent: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    required = {"schema", "contexts", "X", "P", "V", "hidden_environment_queries", "source_provenance", "context_encoding"}
    missing = sorted(required - set(payload))
    if missing:
        raise QZeroRuntimeError(f"QZero corpus missing fields: {missing}")
    if payload.get("context_encoding") != CONTEXT_ENCODING:
        raise QZeroRuntimeError(f"QZero context encoding must be {CONTEXT_ENCODING!r}")
    if len(payload.get("contexts", [])) < 45:
        raise QZeroRuntimeError("QZero requires at least 45 provenance-bearing pretraining contexts")
    if int(payload.get("hidden_environment_queries", 0)) <= 0:
        raise QZeroRuntimeError("QZero pretraining hidden-environment query accounting is absent")
    X = np.asarray(payload["X"], dtype=float); P = np.asarray(payload["P"], dtype=float); V = np.asarray(payload["V"], dtype=float).reshape(-1)
    if X.ndim != 2 or P.ndim != 2 or X.shape[0] != P.shape[0] or X.shape[0] != V.size:
        raise QZeroRuntimeError("QZero X/P/V shapes are inconsistent")
    if X.shape[1] != int(input_dim) or P.shape[1] != int(action_dim):
        raise QZeroRuntimeError(f"QZero corpus dimensionality mismatch: X={X.shape}, P={P.shape}, expected input={input_dim}, action={action_dim}")
    contexts = np.asarray(payload["contexts"], dtype=float)
    if contexts.ndim != 2 or contexts.shape[1] != int(expected_context_dim):
        raise QZeroRuntimeError("QZero context dimension differs from the target Hamiltonian encoding")
    if not np.isfinite(X).all() or not np.isfinite(P).all() or not np.isfinite(V).all() or not np.isfinite(contexts).all():
        raise QZeroRuntimeError("QZero corpus contains non-finite values")
    return {**payload, "_X": X, "_P": P, "_V": V, "_contexts": contexts}


@dataclass(frozen=True)
class QZeroMatchedResult:
    trace: MethodTrace
    diagnostics: Mapping[str, Any]
    selected_control: np.ndarray


def run_qzero_matched_full(
    evaluator: Any,
    hamiltonian: Any,
    *,
    protocol: MatchedControlProtocol,
    corpus_path: str | Path,
    seed: int = 20260817,
    pretrain_epochs: int = 30,
    min_rounds: int = 8,
    max_rounds: int = 64,
    epochs_per_round: int = 10,
    loss_tolerance: float = 1.0e-2,
    patience: int = 3,
    terminal_success_threshold: float | None = None,
) -> QZeroMatchedResult:
    """Execute full QZero pretraining + target fine tuning on the matched D3 domain."""
    space = rc.MatchedBoxDiscreteActionSpace(protocol.bounds, n_actions_grid=41)
    context = ising_context(hamiltonian)
    qz = rc.QZero(
        space, context_dim=context.size, T_us=float(np.mean(protocol.annealing_time_range_us)),
        seed=int(seed), C_start=3.0, C_end=0.5, Nplayout=6, epsilon=0.01, maximize_merit=True,
    )
    corpus = load_pretraining_corpus(
        corpus_path, expected_context_dim=context.size, action_dim=qz.action_dim, input_dim=qz.input_dim,
    )
    losses = qz.pretrain(corpus["_X"], corpus["_P"], corpus["_V"], epochs=int(pretrain_epochs))
    qz.pretraining_queries = int(corpus["hidden_environment_queries"])

    ledger = rc.ResourceLedger(); controls: list[np.ndarray] = []; responses: list[dict[str, Any]] = []; diags: list[dict[str, Any]] = []

    # Every hidden target-environment query is a real annealer evaluation and is
    # therefore charged.  The QZero class calls this function from MCTS leaves.
    def environment(control: np.ndarray) -> float:
        response = evaluator(np.asarray(control, dtype=float), num_reads=int(protocol.reads_per_control))
        controls.append(np.asarray(control, dtype=float).copy()); responses.append(response)
        diag = {"phase": "qzero_hidden_target_query", "query_index": len(controls)-1}
        diags.append(diag)
        ledger.add(rc.CostEntry(
            method="QZero-matched-full", stage="fine_tune_hidden_query", control_query=0,
            hidden_environment_queries=1, reads=int(response.get("num_reads", protocol.reads_per_control)),
            annealing_time_us=float(np.asarray(control)[0]),
            qpu_access_time_us=qpu_access_time_us(response),
            simulator_seconds=float(response.get("elapsed_seconds", 0.0) or 0.0), metadata=diag,
        ))
        return float(response["elite_probability"])

    # The binary terminal criterion must be predeclared outside target QZero
    # fine tuning (for example from the common initial design).
    if terminal_success_threshold is None or not np.isfinite(float(terminal_success_threshold)):
        raise QZeroRuntimeError("A finite predeclared terminal_success_threshold is required")
    initial_target = float(terminal_success_threshold)
    best, diag = qz.fine_tune_to_convergence(
        context, environment, terminal_success=lambda value: float(value) >= initial_target,
        min_rounds=int(min_rounds), max_rounds=int(max_rounds), epochs_per_round=int(epochs_per_round),
        loss_tolerance=float(loss_tolerance), patience=int(patience),
    )
    # Independent final returned-control evaluation is an ordinary control query.
    final_response = evaluator(np.asarray(best, dtype=float), num_reads=int(protocol.reads_per_control))
    controls.append(np.asarray(best, dtype=float).copy()); responses.append(final_response)
    final_diag = {"phase": "qzero_returned_control_confirmation"}; diags.append(final_diag)
    ledger.add(rc.CostEntry(
        method="QZero-matched-full", stage="returned_control", control_query=1,
        hidden_environment_queries=0, reads=int(final_response.get("num_reads", protocol.reads_per_control)),
        annealing_time_us=float(np.asarray(best)[0]),
        qpu_access_time_us=qpu_access_time_us(final_response),
        simulator_seconds=float(final_response.get("elapsed_seconds", 0.0) or 0.0), metadata=final_diag,
    ))
    diagnostics = {
        **dict(diag), "pretraining_loss_final": float(losses[-1]) if losses else None,
        "pretraining_hidden_environment_queries": int(corpus["hidden_environment_queries"]),
        "source_provenance": corpus["source_provenance"], "context_encoding": CONTEXT_ENCODING,
        "network_architecture": qz.net.architecture_manifest(), "Nplayout": 6,
        "C_start": 3.0, "C_end": 0.5, "epsilon": 0.01, "n_actions_per_depth": 41,
        "target_context_dim": int(context.size), "matched_control_dimensions": int(space.order),
        "terminal_success_threshold": initial_target,
    }
    return QZeroMatchedResult(MethodTrace("QZero-matched-full", controls, responses, ledger, diags), diagnostics, np.asarray(best, dtype=float))


__all__ = ["QZeroRuntimeError", "CONTEXT_ENCODING", "ising_context", "load_pretraining_corpus", "QZeroMatchedResult", "run_qzero_matched_full"]
