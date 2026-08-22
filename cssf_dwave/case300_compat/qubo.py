from __future__ import annotations
from dataclasses import dataclass
import itertools
import math
import numpy as np

@dataclass(slots=True)
class CandidateScreen:
    candidates: list[int]
    scores: np.ndarray

@dataclass(slots=True)
class QUBOProblem:
    K: int
    B_max: int
    Q: np.ndarray
    c: np.ndarray
    candidates: list[int]
    lam_pen: float
    pairwise_source: str
    edge_map: dict[tuple[int, int], float]
    S_matrix: np.ndarray | None = None
    physical_Q: np.ndarray | None = None

    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float).reshape(self.K)
        return float(x @ self.Q @ x)

    def brute_force(self, max_configs: int = 1_000_000):
        ncfg = math.comb(self.K, self.B_max)
        if ncfg > max_configs:
            return {"buses_opt": [], "energy_opt": None, "energy_worst": None, "n_configs": ncfg}
        best_e = float("inf"); worst_e = -float("inf"); best = None
        for combo in itertools.combinations(range(self.K), self.B_max):
            x = np.zeros(self.K); x[list(combo)] = 1.0
            e = self.energy(x)
            if e < best_e:
                best_e, best = e, combo
            worst_e = max(worst_e, e)
        return {
            "buses_opt": [self.candidates[i] for i in best] if best is not None else [],
            "energy_opt": float(best_e) if best is not None else None,
            "energy_worst": float(worst_e) if best is not None else None,
            "n_configs": ncfg,
        }

@dataclass(slots=True)
class IsingProblem:
    h: np.ndarray
    J: np.ndarray
    const: float
    K: int
    B_max: int
    candidates: list[int]

@dataclass(slots=True)
class HamiltonianData:
    J_mat: np.ndarray
    lsf_w: np.ndarray

def _minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo, hi = float(np.min(v)), float(np.max(v))
    return np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)

def screen_candidates(ds, mdl, mpf, K: int, *, alpha: float = 0.795, beta: float = 0.205) -> CandidateScreen:
    """Train-only LSF + MPF screening used by the monograph notebook.

    alpha/beta are compatibility defaults for this released notebook, not
    universal physical constants.  They are fixed before OOD evaluation.
    """
    ids = np.asarray(ds.non_slack, dtype=int)
    mean_lsf = np.asarray(mdl.mean_lsf(ds.X_train), dtype=float)
    s_lsf = _minmax(np.abs(mean_lsf[ids]))
    s_mpf = _minmax(np.asarray(mpf, dtype=float)[ids])
    score = float(alpha) * s_lsf + float(beta) * s_mpf
    order = np.argsort(-score, kind="stable")
    K = min(max(1, int(K)), len(ids))
    chosen = order[:K]
    return CandidateScreen(candidates=ids[chosen].tolist(), scores=score[chosen].copy())

def _edge_map(ds, candidates):
    local = {bus: k for k, bus in enumerate(candidates)}
    out = {}
    for i, j, b in ds.edges:
        if i in local and j in local:
            a, c = sorted((local[i], local[j]))
            out[(a, c)] = float(b)
    return out

def build_qubo(ds, cands: CandidateScreen, mdl, mpf, *, B_max: int, pairwise_source: str = "topology") -> QUBOProblem:
    candidates = list(cands.candidates); K = len(candidates); B = int(B_max)
    if not 1 <= B <= K:
        raise ValueError("B_max must lie in [1, K]")
    if pairwise_source not in {"topology", "csnnt"}:
        raise ValueError("pairwise_source must be 'topology' or 'csnnt'")
    scores = np.asarray(cands.scores, dtype=float)
    # Higher screening score means more attractive placement -> lower energy.
    c = -scores.copy()
    physical = np.diag(c)
    edge_map = _edge_map(ds, candidates)
    S = None
    if pairwise_source == "topology":
        if edge_map:
            scale = max(abs(v) for v in edge_map.values())
            for (i, j), b in edge_map.items():
                pair = -0.10 * (abs(b) / max(scale, 1e-12))
                physical[i, j] += pair / 2.0; physical[j, i] += pair / 2.0
    else:
        # A bounded, symmetric first-order interaction proxy derived from
        # correlated predicted LSF columns.  It is a candidate hypothesis,
        # not physical confirmation; the notebook AC-verifies alternatives.
        pred = np.real(mdl.predict(ds.X_train))[:, candidates]
        if pred.shape[0] > 1:
            S = np.corrcoef(pred, rowvar=False)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            S = np.zeros((K, K))
        np.fill_diagonal(S, 0.0)
        gamma = 0.05
        physical += -gamma * S / 2.0
        physical = (physical + physical.T) / 2.0
        np.fill_diagonal(physical, c)
    # Conservative exact-cardinality penalty dominance.
    objective_bound = float(np.sum(np.abs(physical)))
    lam = max(10.0, 5.0 * objective_bound + 1.0)
    Q = physical.copy()
    Q[np.diag_indices(K)] += lam * (1.0 - 2.0 * B)
    for i in range(K):
        for j in range(i + 1, K):
            Q[i, j] += lam; Q[j, i] += lam
    return QUBOProblem(K, B, Q, c, candidates, lam, pairwise_source, edge_map, S, physical)

def verify_qubo(prob: QUBOProblem):
    sym = bool(np.allclose(prob.Q, prob.Q.T, atol=1e-12))
    finite = bool(np.isfinite(prob.Q).all())
    bound = float(np.sum(np.abs(prob.physical_Q))) if prob.physical_Q is not None else 0.0
    return {
        "Q_symmetric": sym,
        "finite": finite,
        "lam_sufficient": bool(prob.lam_pen > bound),
        "objective_absolute_bound": bound,
        "synergy_asymmetry": float(np.max(np.abs(prob.Q - prob.Q.T))),
    }

def qubo_to_ising(prob: QUBOProblem) -> IsingProblem:
    Q = np.asarray(prob.Q, dtype=float); one = np.ones(prob.K)
    h = -0.5 * (Q @ one)
    J = np.zeros_like(Q)
    for i in range(prob.K):
        for j in range(i + 1, prob.K):
            J[i, j] = J[j, i] = Q[i, j] / 2.0
    const = 0.25 * (float(one @ Q @ one) + float(np.trace(Q)))
    return IsingProblem(h=h, J=J, const=const, K=prob.K, B_max=prob.B_max, candidates=list(prob.candidates))

def _ising_energy(ising: IsingProblem, z: np.ndarray) -> float:
    e = float(ising.const + np.dot(ising.h, z))
    for i in range(ising.K):
        for j in range(i + 1, ising.K):
            e += float(ising.J[i, j] * z[i] * z[j])
    return e

def verify_ising_identity(prob: QUBOProblem, ising: IsingProblem, *, trials: int = 256, seed: int = 20260822):
    rng = np.random.default_rng(seed); max_err = 0.0
    controls = [np.zeros(prob.K), np.ones(prob.K)]
    controls += [rng.integers(0, 2, prob.K).astype(float) for _ in range(trials)]
    for x in controls:
        z = 1.0 - 2.0 * x
        max_err = max(max_err, abs(prob.energy(x) - _ising_energy(ising, z)))
    return {"passed": bool(max_err <= 1e-8), "max_err": float(max_err)}

def build_hamiltonian(ising: IsingProblem, lsf_raw) -> HamiltonianData:
    w = np.abs(np.asarray(lsf_raw, dtype=float)).reshape(ising.K)
    mx = float(np.max(w)) if w.size else 0.0
    if mx > 0: w = w / mx
    return HamiltonianData(J_mat=np.asarray(ising.J, dtype=float).copy(), lsf_w=w)
