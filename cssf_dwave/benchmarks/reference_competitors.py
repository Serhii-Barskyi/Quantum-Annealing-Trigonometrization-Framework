"""Reference-faithful full-function comparators for the CSSF(QA) D-Wave evaluation.

The comparators in this module are additive to the original CSSF framework.
They never implement or replace CSNN-T, the CSSF spectral hierarchy, the BESS
pipeline, or the Pegasus backend. The same comparator implementations are
used by the Simulator and Pegasus evidence notebooks under matched budgets.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Mapping, Sequence
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import math
import time
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from scipy.integrate import cumulative_trapezoid

REAL = np.float64
COMPLEX = np.complex128


class FullBenchError(RuntimeError):
    pass


class FidelityError(FullBenchError):
    pass


class BudgetError(FullBenchError):
    pass


class ClaimLockedError(FullBenchError):
    pass


class HardwareGateError(FullBenchError):
    pass


def stable_hash(value: Any, prefix: str = "CSSF-REFERENCE-COMPARATORS-v38") -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str, separators=(",", ":"))
    h = hashlib.sha256()
    h.update(prefix.encode("utf-8") + b"\0" + payload.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Benchmark registry and fidelity manifests shared by BOTH notebooks.
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: "OrderedDict[str, dict[str, Any]]" = OrderedDict([
    ("TuRBO-paper", {
        "role": "Jeong et al. paper-faithful schedule optimizer",
        "claim_grade": False,
        "source": "Jeong et al., arXiv:2510.15245v1 (2025)",
        "required": ["persistent_state", "matern52_ard", "white_noise", "expected_improvement",
                     "trust_region", "shrink_expand", "restart", "joint_T_theta",
                     "fourier_bounds", "clip_discretize", "adaptive_reads", "runtime_guard"],
    }),
    ("TuRBO-matched-full", {
        "role": "Jeong-TuRBO full machinery on matched CSSF endpoint/budget",
        "claim_grade": True,
        "source": "Jeong et al., arXiv:2510.15245v1 (2025)",
        "required": ["persistent_state", "matern52_ard", "white_noise", "expected_improvement",
                     "trust_region", "shrink_expand", "restart", "joint_T_theta",
                     "fourier_bounds", "clip_discretize", "adaptive_reads", "runtime_guard"],
    }),
    ("Finzgar-BO-paper", {
        "role": "Finzgar et al. paper-faithful Matérn-5/2 UCB BO",
        "claim_grade": False,
        "source": "Finzgar et al., arXiv:2305.13365v1; Phys. Rev. Research 6, 023063 (2024)",
        "required": ["matern52", "noise", "linear_plus_9_random", "ucb", "kappa_decay",
                     "progressive_hyperfit", "paper_50_iterations"],
    }),
    ("Finzgar-BO-matched-full", {
        "role": "Finzgar full BO on matched control domain/budget",
        "claim_grade": True,
        "source": "Finzgar et al., arXiv:2305.13365v1; Phys. Rev. Research 6, 023063 (2024)",
        "required": ["matern52", "noise", "linear_plus_9_random", "ucb", "kappa_decay",
                     "progressive_hyperfit"],
    }),
    ("GP+EI-full", {
        "role": "full sequential GP expected-improvement optimizer",
        "claim_grade": True,
        "source": "Canonical Gaussian-process Bayesian optimization / Expected Improvement comparator",
        "required": ["gp_posterior", "uncertainty", "expected_improvement", "sequential_update"],
    }),
    ("QZero-paper", {
        "role": "Chen et al. paper-faithful QZero: M=5 discrete Fourier MCTS + policy/value NN + transfer/fine-tuning",
        "claim_grade": False,
        "source": "Chen et al., arXiv:2004.02836v3; Nature Machine Intelligence 4, 269-278 (2022)",
        "required": ["four_stage_mcts", "policy_network", "value_network", "mcts_pretraining",
                     "supervised_pretrain", "target_finetune", "transfer_context", "hidden_query_accounting",
                     "paper_M5_l02_delta001_T70", "binary_terminal_pm1", "energy_minimization"],
    }),
    ("QZero-matched-full", {
        "role": "Full QZero machinery on the common CSSF Pegasus control box with native discrete actions",
        "claim_grade": True,
        "source": "Chen et al., arXiv:2004.02836v3; Nature Machine Intelligence 4, 269-278 (2022)",
        "required": ["four_stage_mcts", "policy_network", "value_network", "mcts_pretraining",
                     "supervised_pretrain", "target_finetune", "transfer_context", "hidden_query_accounting",
                     "matched_T_theta_domain", "binary_terminal_pm1"],
    }),
    ("Worldline-Susceptibility-full", {
        "role": "actual SQA-worldline susceptibility schedule construction",
        "claim_grade": True,
        "source": "Singh et al., arXiv:2607.14282v1 (2026)",
        "required": ["suzuki_trotter_worldlines", "measured_susceptibility", "chi_plus_chi0",
                     "cumulative_allocation", "inverse_schedule", "direct_raw_schedule"],
    }),
    ("Periodic-GP", {
        "role": "probabilistic periodic application representation comparator",
        "claim_grade": True,
        "source": "Haji Ghassemi & Deisenroth, AISTATS/PMLR 33 (2014)",
        "required": ["periodic_kernel", "gp_posterior", "uncertainty"],
    }),
    ("Torus-Riemannian-Matern-GP", {
        "role": "intrinsic torus spectral Matérn GP comparator",
        "claim_grade": True,
        "source": "Jaquier et al., PMLR 164 (2022), geometry-aware/Riemannian Matérn BO",
        "required": ["torus_spectral_kernel", "positive_semidefinite", "gp_posterior", "uncertainty"],
    }),
    ("Strong-SA", {
        "role": "budget-tuned simulated annealing classical comparator",
        "claim_grade": True,
        "source": "D-Wave Ocean dwave-samplers 1.8.0 SimulatedAnnealingSampler API",
        "required": ["predeclared_tuning", "tuning_cost", "heldout_eval", "matched_classical_budget"],
    }),
    ("Strong-Tabu", {
        "role": "budget-tuned Tabu classical comparator",
        "claim_grade": True,
        "source": "D-Wave Ocean dwave-samplers 1.8.0 TabuSampler API",
        "required": ["predeclared_tuning", "tuning_cost", "heldout_eval", "matched_classical_budget"],
    }),
    ("HiGHS-quality-reference", {
        "role": "exact/quality MILP-QUBO reference in its declared role",
        "claim_grade": False,
        "source": "HiGHS/highspy 1.15.1 + cssf_dwave exact-quality wrapper",
        "required": ["same_proxy_objective", "optimal_status", "independent_energy_recheck"],
    }),
])

REGISTRY_HASH = stable_hash(BENCHMARK_REGISTRY, prefix="CSSF-COMPETITOR-REGISTRY-v38")


@dataclass
class FidelityManifest:
    method: str
    mechanisms: dict[str, bool]
    reference_reproduction_pass: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def required_mechanisms(self) -> tuple[str, ...]:
        if self.method not in BENCHMARK_REGISTRY:
            raise FidelityError(f"Unknown method {self.method!r}")
        return tuple(BENCHMARK_REGISTRY[self.method]["required"])

    def missing(self) -> list[str]:
        return [name for name in self.required_mechanisms() if not bool(self.mechanisms.get(name, False))]

    @property
    def complete(self) -> bool:
        return not self.missing() and bool(self.reference_reproduction_pass)

    def assert_complete(self) -> None:
        missing = self.missing()
        if missing or not self.reference_reproduction_pass:
            raise FidelityError(
                f"{self.method} is not claim-grade: missing={missing}, "
                f"reference_reproduction_pass={self.reference_reproduction_pass}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mechanisms": dict(self.mechanisms),
            "reference_reproduction_pass": bool(self.reference_reproduction_pass),
            "provenance": dict(self.provenance),
            "missing": self.missing(),
            "complete": self.complete,
        }


# ---------------------------------------------------------------------------
# Shared schedule domain.
# ---------------------------------------------------------------------------

def fourier_bounds(order: int, annealing_time_range: tuple[float, float], alpha: float = 0.35) -> NDArray[np.float64]:
    m = int(order)
    lo, hi = map(float, annealing_time_range)
    if m < 1 or not (0 < lo < hi) or not (alpha > 0):
        raise ValueError("Invalid Fourier bounds configuration")
    return np.asarray([[lo, hi]] + [[-float(alpha)/k, float(alpha)/k] for k in range(1, m+1)], dtype=REAL)


def canonical_control(order: int, annealing_time_us: float) -> NDArray[np.float64]:
    return np.asarray([float(annealing_time_us)] + [0.0]*int(order), dtype=REAL)


def fourier_forward_schedule(parameters: ArrayLike, *, order: int, grid_points: int = 129,
                             reject_nonmonotone: bool = True,
                             amplitude_resolution: float | None = None,
                             time_resolution_us: float | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p = np.asarray(parameters, dtype=REAL).reshape(-1)
    m = int(order)
    if p.size != m + 1:
        raise ValueError("Expected [T_us, theta_1, ..., theta_M]")
    T = float(p[0])
    if not math.isfinite(T) or T <= 0:
        raise ValueError("T_us must be finite and positive")
    u = np.linspace(0.0, 1.0, int(grid_points), dtype=REAL)
    s = u.copy()
    for k, theta in enumerate(p[1:], start=1):
        s += float(theta) * np.sin(k * math.pi * u)
    s = np.clip(s, 0.0, 1.0)
    s[0], s[-1] = 0.0, 1.0
    if amplitude_resolution is not None:
        r = float(amplitude_resolution)
        if r <= 0:
            raise ValueError("amplitude_resolution must be positive")
        s = np.round(s/r)*r
        s = np.clip(s, 0.0, 1.0)
        s[0], s[-1] = 0.0, 1.0
    t = T*u
    if time_resolution_us is not None:
        r = float(time_resolution_us)
        if r <= 0:
            raise ValueError("time_resolution_us must be positive")
        t = np.round(t/r)*r
        t[0], t[-1] = 0.0, T
        # preserve strictly increasing time by dropping duplicate time points
        keep = np.r_[True, np.diff(t) > 0]
        t, s = t[keep], s[keep]
        if t[-1] != T:
            t = np.r_[t, T]; s = np.r_[s, 1.0]
    if reject_nonmonotone and np.any(np.diff(s) < -1e-12):
        raise ValueError("Candidate leaves the shared monotone forward-schedule domain")
    return np.asarray(t, dtype=REAL), np.asarray(s, dtype=REAL)


def feasible_control(control: ArrayLike, bounds: ArrayLike, *, order: int, grid_points: int = 129) -> bool:
    x = np.asarray(control, dtype=REAL).reshape(-1)
    b = np.asarray(bounds, dtype=REAL)
    if x.size != b.shape[0] or np.any(x < b[:,0]-1e-12) or np.any(x > b[:,1]+1e-12):
        return False
    try:
        fourier_forward_schedule(x, order=order, grid_points=grid_points, reject_nonmonotone=True)
    except Exception:
        return False
    return True


def latin_hypercube(bounds: ArrayLike, n: int, seed: int = 0) -> NDArray[np.float64]:
    b = np.asarray(bounds, dtype=REAL)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds must have shape (d,2)")
    sampler = qmc.LatinHypercube(d=b.shape[0], seed=int(seed))
    z = sampler.random(int(n))
    return qmc.scale(z, b[:,0], b[:,1]).astype(REAL)


def normalized(x: ArrayLike, bounds: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=REAL)
    b = np.asarray(bounds, dtype=REAL)
    return (x-b[:,0])/(b[:,1]-b[:,0])


def denormalized(z: ArrayLike, bounds: ArrayLike) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=REAL)
    b = np.asarray(bounds, dtype=REAL)
    return b[:,0] + z*(b[:,1]-b[:,0])


# ---------------------------------------------------------------------------
# Shared resource/query accounting and claim lock.
# ---------------------------------------------------------------------------

@dataclass
class CostEntry:
    method: str
    stage: str
    control_query: int = 1
    hidden_environment_queries: int = 0
    reads: int = 0
    annealing_time_us: float = 0.0
    qpu_access_time_us: float = 0.0
    programming_time_us: float = 0.0
    readout_time_us: float = 0.0
    classical_seconds: float = 0.0
    simulator_seconds: float = 0.0
    ac_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_query_equivalents(self) -> int:
        return int(self.control_query) + int(self.hidden_environment_queries)


@dataclass
class ResourceLedger:
    entries: list[CostEntry] = field(default_factory=list)

    def add(self, entry: CostEntry) -> None:
        if entry.control_query < 0 or entry.hidden_environment_queries < 0 or entry.reads < 0:
            raise BudgetError("Negative accounting is forbidden")
        self.entries.append(entry)

    def totals(self, method: str | None = None) -> dict[str, float]:
        rows = self.entries if method is None else [e for e in self.entries if e.method == method]
        keys = ["control_query", "hidden_environment_queries", "reads", "annealing_time_us",
                "qpu_access_time_us", "programming_time_us", "readout_time_us", "classical_seconds",
                "simulator_seconds", "ac_calls"]
        out = {k: float(sum(getattr(e,k) for e in rows)) for k in keys}
        out["query_equivalents"] = out["control_query"] + out["hidden_environment_queries"]
        return out

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.entries]


@dataclass(frozen=True)
class ClaimGate:
    cssf_fidelity: bool
    competitor_fidelity: bool
    matched_budget: bool
    independent_confirmation: bool
    runtime_or_hardware_gate: bool = True
    cost_to_target_superiority: bool = False

    @property
    def open(self) -> bool:
        return all([self.cssf_fidelity, self.competitor_fidelity, self.matched_budget,
                    self.independent_confirmation, self.runtime_or_hardware_gate,
                    self.cost_to_target_superiority])

    def assert_open(self, method: str) -> None:
        if not self.open:
            raise ClaimLockedError(f"CSSF > {method} claim is locked: {asdict(self)}")


def export_superiority_row(method: str, payload: Mapping[str, Any], gate: ClaimGate) -> dict[str, Any]:
    gate.assert_open(method)
    return {"competitor": method, "status": "CLAIM_GRADE", **dict(payload)}


# ---------------------------------------------------------------------------
# Gaussian-process utilities.
# ---------------------------------------------------------------------------

def _sklearn_gp(*, d: int, ard: bool, noise_level: float = 1e-6, random_state: int = 0,
                n_restarts_optimizer: int = 2):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ImportError as exc:
        raise FullBenchError("scikit-learn is required for GP competitors") from exc
    ls = np.ones(d, dtype=REAL) if ard else 1.0
    ls_bounds = (1e-3, 1e3)
    kernel = (ConstantKernel(1.0, (1e-3,1e3)) * Matern(length_scale=ls, length_scale_bounds=ls_bounds, nu=2.5)
              + WhiteKernel(noise_level=float(noise_level), noise_level_bounds=(1e-10,1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=int(n_restarts_optimizer),
                                    random_state=int(random_state))


def fit_gp(X: ArrayLike, y: ArrayLike, *, ard: bool = True, noise_level: float = 1e-6,
           random_state: int = 0, n_restarts_optimizer: int = 2):
    X = np.asarray(X, dtype=REAL); y = np.asarray(y, dtype=REAL).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size or y.size < 2:
        raise ValueError("GP requires at least two aligned observations")
    gp = _sklearn_gp(d=X.shape[1], ard=ard, noise_level=noise_level,
                     random_state=random_state, n_restarts_optimizer=n_restarts_optimizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X,y)
    return gp


def _extract_lengthscales(kernel: Any, d: int) -> NDArray[np.float64]:
    # sklearn structure: Sum(Product(ConstantKernel, Matern), WhiteKernel)
    objects = [kernel]
    while objects:
        obj = objects.pop()
        if obj.__class__.__name__ == "Matern":
            ls = np.asarray(obj.length_scale, dtype=REAL).reshape(-1)
            if ls.size == 1:
                ls = np.repeat(ls, d)
            return np.clip(ls, 1e-12, None)
        for attr in ("k1", "k2"):
            if hasattr(obj, attr):
                objects.append(getattr(obj, attr))
    return np.ones(d, dtype=REAL)


def expected_improvement(mu: ArrayLike, sigma: ArrayLike, best: float, *, minimize_objective: bool = True,
                         xi: float = 0.0) -> NDArray[np.float64]:
    m = np.asarray(mu, dtype=REAL); s = np.asarray(sigma, dtype=REAL)
    imp = (best - m - xi) if minimize_objective else (m - best - xi)
    out = np.zeros_like(m)
    mask = s > 1e-15
    z = np.zeros_like(m); z[mask] = imp[mask]/s[mask]
    out[mask] = imp[mask]*norm.cdf(z[mask]) + s[mask]*norm.pdf(z[mask])
    return np.maximum(out, 0.0)


def optimize_acquisition(acq: Callable[[NDArray[np.float64]], float], lo: NDArray[np.float64], hi: NDArray[np.float64],
                         *, n_starts: int = 32, seed: int = 0,
                         feasibility: Callable[[NDArray[np.float64]], bool] | None = None) -> tuple[NDArray[np.float64], float]:
    rng = np.random.default_rng(int(seed))
    d = lo.size
    sobol = qmc.Sobol(d=d, scramble=True, seed=int(seed))
    m = int(math.ceil(math.log2(max(2,n_starts))))
    starts = lo + sobol.random_base2(m=m)[:n_starts]*(hi-lo)
    starts = np.vstack([starts, (lo+hi)/2])
    best_x = None; best_v = -np.inf
    bounds = list(zip(lo,hi))
    def obj(x):
        x = np.asarray(x, dtype=REAL)
        if feasibility is not None and not feasibility(x):
            return 1e12
        v = float(acq(x))
        return -v if math.isfinite(v) else 1e12
    for x0 in starts:
        r = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        x = np.clip(r.x, lo, hi)
        if feasibility is not None and not feasibility(x):
            continue
        v = float(acq(x))
        if v > best_v:
            best_x, best_v = x.copy(), v
    if best_x is None:
        # deterministic rejection sampling within box, never silently leaving feasibility domain.
        for _ in range(5000):
            x = rng.uniform(lo,hi)
            if feasibility is None or feasibility(x):
                v = float(acq(x))
                if v > best_v:
                    best_x, best_v = x.copy(), v
        if best_x is None:
            raise FullBenchError("Acquisition optimizer found no feasible candidate")
    return best_x, best_v


# ---------------------------------------------------------------------------
# Full generic GP+EI.
# ---------------------------------------------------------------------------

@dataclass
class SequentialGPEI:
    bounds: NDArray[np.float64]
    minimize_objective: bool = True
    noise_level: float = 1e-6
    seed: int = 0
    n_restarts_optimizer: int = 2
    X: list[NDArray[np.float64]] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def add(self, x: ArrayLike, value: float) -> None:
        xx = np.asarray(x, dtype=REAL).reshape(-1)
        if xx.size != self.bounds.shape[0]: raise ValueError("dimension mismatch")
        self.X.append(xx.copy()); self.y.append(float(value))

    def suggest(self, *, feasibility: Callable[[NDArray[np.float64]], bool] | None = None) -> tuple[NDArray[np.float64], dict[str,Any]]:
        if len(self.X) < 3: raise FullBenchError("GP+EI needs at least 3 initial observations")
        X = np.asarray(self.X); y = np.asarray(self.y)
        Z = normalized(X,self.bounds)
        gp = fit_gp(Z,y,ard=True,noise_level=self.noise_level,random_state=self.seed,
                    n_restarts_optimizer=self.n_restarts_optimizer)
        best = float(np.min(y) if self.minimize_objective else np.max(y))
        def acq_x(x):
            z = normalized(x,self.bounds).reshape(1,-1)
            mu,std = gp.predict(z,return_std=True)
            return float(expected_improvement(mu,std,best,minimize_objective=self.minimize_objective)[0])
        x,score = optimize_acquisition(acq_x,self.bounds[:,0],self.bounds[:,1],n_starts=32,seed=self.seed+len(self.X),feasibility=feasibility)
        return x,{"expected_improvement":score,"kernel":str(gp.kernel_),"n_observations":len(self.X)}


# ---------------------------------------------------------------------------
# Finzgar et al. full BO: Matern-5/2, UCB, linear+9 random, kappa decay.
# ---------------------------------------------------------------------------

def finzgar_kappa(iteration: int, *, total_iterations: int = 50, decay_start: int = 25,
                   initial: float = 2.0, final: float = 0.01) -> float:
    i = int(iteration)
    if i <= decay_start:
        return float(initial)
    if total_iterations <= decay_start:
        raise ValueError("total_iterations must exceed decay_start")
    frac = min(1.0, max(0.0, (i-decay_start)/(total_iterations-decay_start)))
    return float(initial * (final/initial)**frac)


@dataclass
class FinzgarBO:
    bounds: NDArray[np.float64]
    maximize_merit: bool = True
    noise_level: float = 1e-6
    seed: int = 0
    total_paper_iterations: int = 50
    X: list[NDArray[np.float64]] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def initial_design(self, linear_control: ArrayLike, *, feasibility: Callable[[NDArray[np.float64]], bool] | None = None) -> NDArray[np.float64]:
        linear = np.asarray(linear_control,dtype=REAL).reshape(-1)
        if linear.size != self.bounds.shape[0]: raise ValueError("linear control dimension mismatch")
        rng = np.random.default_rng(self.seed)
        rows=[linear]
        while len(rows)<10:
            x=rng.uniform(self.bounds[:,0],self.bounds[:,1])
            if feasibility is None or feasibility(x): rows.append(x)
        return np.asarray(rows,dtype=REAL)

    def add(self,x:ArrayLike,value:float)->None:
        self.X.append(np.asarray(x,dtype=REAL).reshape(-1).copy()); self.y.append(float(value))

    def suggest(self, *, iteration: int, feasibility: Callable[[NDArray[np.float64]], bool] | None = None) -> tuple[NDArray[np.float64],dict[str,Any]]:
        if len(self.X)<10: raise FullBenchError("Finzgar BO requires its 10-point initialization before UCB")
        X=np.asarray(self.X); y=np.asarray(self.y); Z=normalized(X,self.bounds)
        gp=fit_gp(Z,y,ard=False,noise_level=self.noise_level,random_state=self.seed,
                  n_restarts_optimizer=2)
        kappa=finzgar_kappa(iteration,total_iterations=self.total_paper_iterations)
        def acq_x(x):
            mu,std=gp.predict(normalized(x,self.bounds).reshape(1,-1),return_std=True)
            u=float(mu[0]+kappa*std[0])
            return u if self.maximize_merit else -float(mu[0]-kappa*std[0])
        x,score=optimize_acquisition(acq_x,self.bounds[:,0],self.bounds[:,1],n_starts=32,seed=self.seed+iteration,feasibility=feasibility)
        return x,{"ucb_score":score,"kappa":kappa,"kernel":str(gp.kernel_),"n_observations":len(self.X)}


# ---------------------------------------------------------------------------
# Full Jeong-TuRBO machinery with persistent state, EI, adaptive reads/budget.
# ---------------------------------------------------------------------------

@dataclass
class JeongTuRBOState:
    """Persistent trust-region state matching Jeong et al. Algorithm 1.

    The paper expands immediately on *every* strict improvement and shrinks only
    after ``patience`` consecutive non-improvements.  A consecutive-success
    threshold belongs to other TuRBO variants and is deliberately absent here.
    ``length`` is the normalized full width used by this pre-frozen reproduction
    protocol; proposal bounds use ``0.5*length`` on each side of the incumbent.
    This numerical width convention is declared protocol metadata rather than
    attributed to the paper, whose Algorithm 1 leaves the numeric TR constants
    symbolic and whose text/inequality use potentially ambiguous side-length notation.
    """
    length: float = 0.8
    length_min: float = 0.8 * (0.5**7)
    length_max: float = 1.6
    failure_count: int = 0
    patience: int = 5
    shrink_factor: float = 0.5   # rho
    expand_factor: float = 2.0   # rho^-1
    restart_count: int = 0
    iteration: int = 0
    incumbent_x: NDArray[np.float64] | None = None
    incumbent_y: float | None = None
    restart_pending: bool = False


@dataclass
class QPURuntimeBudget:
    total_us: float
    programming_time_us: float = 20_000.0
    readout_time_us: float = 100.0
    control_overhead_us: float = 20.0
    remaining_us: float | None = None

    def __post_init__(self):
        if self.remaining_us is None: self.remaining_us=float(self.total_us)
        if self.total_us<=0: raise ValueError("total_us must be positive")

    def estimate(self,T_us:float,reads:int)->float:
        return float(self.programming_time_us + int(reads)*(float(T_us)+self.readout_time_us+self.control_overhead_us))

    def allocate(self,T_us:float,desired_reads:int,min_reads:int)->tuple[int,float]:
        per=float(T_us)+self.readout_time_us+self.control_overhead_us
        available=float(self.remaining_us)-self.programming_time_us
        max_reads=int(math.floor(max(0.0,available)/per))
        reads=min(int(desired_reads),max_reads)
        if reads<int(min_reads):
            raise BudgetError("QPU runtime budget cannot support minimum read count")
        cost=self.estimate(T_us,reads)
        self.remaining_us=float(self.remaining_us)-cost
        return reads,cost

    def reconcile(self,reserved_us:float,actual_us:float)->float:
        """Replace a pre-call reservation by measured QPU access time.

        The reservation keeps the next call inside the budget before submission;
        reconciliation makes the ledger hardware-timing-aware after the response.
        """
        reserved=float(reserved_us); actual=float(actual_us)
        if reserved<0 or actual<0 or not (math.isfinite(reserved) and math.isfinite(actual)):
            raise BudgetError("runtime reconciliation values must be finite and non-negative")
        self.remaining_us=float(self.remaining_us)+reserved-actual
        if self.remaining_us < -1e-9:
            raise BudgetError("measured QPU access time exhausted the frozen runtime budget")
        self.remaining_us=max(0.0,float(self.remaining_us))
        return self.remaining_us


@dataclass
class JeongTuRBO:
    bounds: NDArray[np.float64]
    order: int
    minimize_objective: bool = True
    noise_level: float = 1e-6
    seed: int = 0
    alpha: float = 0.35
    state: JeongTuRBOState = field(default_factory=JeongTuRBOState)
    X: list[NDArray[np.float64]] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    restart_queue: list[NDArray[np.float64]] = field(default_factory=list)

    def add_initial(self,x:ArrayLike,value:float)->None:
        xx=np.asarray(x,dtype=REAL).reshape(-1); self.X.append(xx.copy()); self.y.append(float(value))
        if self.state.incumbent_y is None or ((value<self.state.incumbent_y) if self.minimize_objective else (value>self.state.incumbent_y)):
            self.state.incumbent_x=xx.copy(); self.state.incumbent_y=float(value)

    def _improved(self,value:float)->bool:
        """Return the strict-improvement predicate in Jeong et al. Algorithm 1.

        No numerical success tolerance is inserted here.  Any tolerance-based
        success counter would change the published state-transition semantics
        and is therefore prohibited in the paper-faithful and matched-full arms.
        """
        best=self.state.incumbent_y
        if best is None: return True
        return bool(value < best) if self.minimize_objective else bool(value > best)

    def observe(self,x:ArrayLike,value:float)->dict[str,Any]:
        xx=np.asarray(x,dtype=REAL).reshape(-1); v=float(value); improved=self._improved(v)
        self.X.append(xx.copy()); self.y.append(v); self.state.iteration+=1
        previous_length=float(self.state.length)
        if improved:
            # Jeong et al. Algorithm 1 lines 18--20: reset no-improve and
            # expand immediately by rho^-1 after each strict improvement.
            self.state.incumbent_x=xx.copy(); self.state.incumbent_y=v
            self.state.failure_count=0
            self.state.length=min(self.state.length_max,self.state.length*self.state.expand_factor)
            update_reason="improvement_expand_immediately"
        else:
            self.state.failure_count+=1
            update_reason="no_improvement_wait"
            if self.state.failure_count>=self.state.patience:
                self.state.length=self.state.length*self.state.shrink_factor
                self.state.failure_count=0
                update_reason="patience_shrink"
        if self.state.length < self.state.length_min:
            self.state.restart_pending=True
            update_reason += "_restart_pending"
        return {"improved":improved,"length_before":previous_length,"length":self.state.length,
                "no_improve_count":self.state.failure_count,"update_reason":update_reason,
                "restart_pending":self.state.restart_pending,"restart_count":self.state.restart_count,
                "iteration":self.state.iteration}

    def _start_restart(self,n_points:int=5)->None:
        pts=latin_hypercube(self.bounds,n_points,seed=self.seed+10_000+self.state.restart_count)
        self.restart_queue=[p for p in pts]
        self.state.restart_count+=1
        self.state.length=0.8; self.state.failure_count=0
        self.state.restart_pending=False

    def adaptive_reads(self,candidate:ArrayLike,*,min_reads:int=250,max_reads:int=900)->int:
        x=np.asarray(candidate,dtype=REAL).reshape(-1)
        if self.state.incumbent_x is None: return int(min_reads)
        z=normalized(x,self.bounds); zi=normalized(self.state.incumbent_x,self.bounds)
        dist=float(np.linalg.norm(z-zi)/math.sqrt(z.size))
        proximity=max(0.0,1.0-min(1.0,dist))
        progress=1.0-math.exp(-max(0,self.state.iteration)/10.0)
        weight=0.5*proximity+0.5*progress
        return int(round(min_reads+weight*(max_reads-min_reads)))

    def suggest(self,*,feasibility:Callable[[NDArray[np.float64]],bool]|None=None)->tuple[NDArray[np.float64],dict[str,Any]]:
        if len(self.X)<3: raise FullBenchError("Jeong-TuRBO needs an initial space-filling design")
        if self.state.restart_pending and not self.restart_queue: self._start_restart()
        if self.restart_queue:
            x=self.restart_queue.pop(0)
            return x,{"mode":"restart_space_filling","restart_count":self.state.restart_count,"length":self.state.length}
        X=np.asarray(self.X); y=np.asarray(self.y); Z=normalized(X,self.bounds)
        gp=fit_gp(Z,y,ard=True,noise_level=self.noise_level,random_state=self.seed,
                  n_restarts_optimizer=2)
        incumbent_idx=int(np.argmin(y) if self.minimize_objective else np.argmax(y))
        center=Z[incumbent_idx]
        self.state.incumbent_x=X[incumbent_idx].copy(); self.state.incumbent_y=float(y[incumbent_idx])
        # Jeong et al. define an axis-aligned hyperrectangle with side lengths
        # Delta_j and do not prescribe GP-lengthscale weighting in Algorithm 1.
        # The disclosed reproduction protocol therefore uses equal normalized
        # half-widths in all dimensions; no original-TuRBO enhancement is hidden.
        half=np.full(Z.shape[1],0.5*self.state.length,dtype=REAL)
        lo=np.maximum(0.0,center-half); hi=np.minimum(1.0,center+half)
        best=float(y[incumbent_idx])
        def acq_z(z):
            x=denormalized(z,self.bounds)
            if feasibility is not None and not feasibility(x): return -1e30
            mu,std=gp.predict(np.asarray(z).reshape(1,-1),return_std=True)
            return float(expected_improvement(mu,std,best,minimize_objective=self.minimize_objective)[0])
        zstar,score=optimize_acquisition(acq_z,lo,hi,n_starts=32,seed=self.seed+len(self.X),
                                         feasibility=(None if feasibility is None else lambda z: feasibility(denormalized(z,self.bounds))))
        xstar=denormalized(zstar,self.bounds)
        return xstar,{"mode":"ei_trust_region","expected_improvement":score,"kernel":str(gp.kernel_),
                      "length":self.state.length,"trust_region_geometry":"axis_aligned_equal_normalized_halfwidth",
                      "trust_lo_normalized":lo.tolist(),"trust_hi_normalized":hi.tolist(),
                      "restart_count":self.state.restart_count}

    def allocate_reads(self,candidate:ArrayLike,budget:QPURuntimeBudget,*,min_reads:int=250,max_reads:int=900)->tuple[int,float]:
        desired=self.adaptive_reads(candidate,min_reads=min_reads,max_reads=max_reads)
        return budget.allocate(float(np.asarray(candidate)[0]),desired,min_reads)


# ---------------------------------------------------------------------------
# Stable CSSF Hermitian multi-output model and calibrated uncertainty.
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Periodic GP and intrinsic flat-torus spectral Matérn GP.
# ---------------------------------------------------------------------------

def trig_embed(theta:ArrayLike)->NDArray[np.float64]:
    t=np.asarray(theta,dtype=REAL)
    return np.concatenate([np.sin(t),np.cos(t)],axis=1)


@dataclass
class PeriodicGP:
    noise_level:float=1e-6
    seed:int=0
    gp:Any=None
    def fit(self,theta:ArrayLike,y:ArrayLike)->"PeriodicGP":
        self.gp=fit_gp(trig_embed(theta),y,ard=True,noise_level=self.noise_level,random_state=self.seed,n_restarts_optimizer=2)
        return self
    def predict(self,theta:ArrayLike,return_std:bool=False):
        if self.gp is None: raise FullBenchError("PeriodicGP not fitted")
        return self.gp.predict(trig_embed(theta),return_std=return_std)


@dataclass
class TorusSpectralMaternGP:
    nu:float=2.5
    kappa:float=1.0
    truncation:int=2
    noise:float=1e-6
    X:NDArray[np.float64]|None=None
    alpha:NDArray[np.float64]|None=None
    L:NDArray[np.float64]|None=None

    def _modes(self,d:int)->NDArray[np.int64]:
        """Sparse Laplace--Beltrami eigenmodes with ||k||_1 <= truncation.

        This is a genuine spectral truncation on the flat torus and avoids the
        exponentially large Cartesian frequency cube in moderate dimension.
        """
        d=int(d); q=int(self.truncation)
        if d<1 or q<0: raise ValueError("torus dimension/truncation must be positive/non-negative")
        modes=[]; current=[0]*d
        def rec(j:int,remaining:int):
            if j==d:
                modes.append(tuple(current)); return
            for v in range(-remaining,remaining+1):
                current[j]=v; rec(j+1,remaining-abs(v))
            current[j]=0
        rec(0,q)
        return np.asarray(sorted(set(modes)),dtype=np.int64)
    def kernel(self,A:ArrayLike,B:ArrayLike)->NDArray[np.float64]:
        A=np.asarray(A,dtype=REAL); B=np.asarray(B,dtype=REAL); d=A.shape[1]
        modes=self._modes(d)
        lam=np.sum(modes*modes,axis=1).astype(REAL)
        weights=(self.kappa**2+lam)**(-(self.nu+d/2.0))
        weights/=np.sum(weights)
        delta=A[:,None,:]-B[None,:,:]
        phase=np.einsum("abd,md->abm",delta,modes)
        return np.einsum("abm,m->ab",np.cos(phase),weights).astype(REAL)
    def fit(self,X:ArrayLike,y:ArrayLike)->"TorusSpectralMaternGP":
        X=np.asarray(X,dtype=REAL); y=np.asarray(y,dtype=REAL).reshape(-1)
        K=self.kernel(X,X)+float(self.noise)*np.eye(X.shape[0])
        L=np.linalg.cholesky(K+1e-12*np.eye(K.shape[0]))
        alpha=linalg.cho_solve((L,True),y)
        self.X=X.copy(); self.L=L; self.alpha=alpha
        return self
    def predict(self,Xs:ArrayLike,return_std:bool=False):
        if self.X is None or self.alpha is None or self.L is None: raise FullBenchError("Torus GP not fitted")
        Xs=np.asarray(Xs,dtype=REAL); Ks=self.kernel(Xs,self.X); mean=Ks@self.alpha
        if not return_std:return mean
        v=linalg.solve_triangular(self.L,Ks.T,lower=True)
        var=np.diag(self.kernel(Xs,Xs))-np.sum(v*v,axis=0)
        return mean,np.sqrt(np.maximum(var,0.0))


# ---------------------------------------------------------------------------
# Worldline SQA susceptibility schedule (actual Suzuki-Trotter Monte Carlo).
# ---------------------------------------------------------------------------

@dataclass
class WorldlineSQAConfig:
    beta: float = 2.0
    replicas: int = 32
    burn_in_sweeps: int = 200
    measurement_sweeps: int = 400
    thin: int = 2
    seed: int = 0


def _ising_arrays(h: Mapping[int,float] | Sequence[float], J: Mapping[tuple[int,int],float] | NDArray[np.float64]):
    if isinstance(h,Mapping):
        n=max(h.keys())+1 if h else 0; hv=np.zeros(n,dtype=REAL)
        for i,v in h.items(): hv[int(i)]=float(v)
    else: hv=np.asarray(h,dtype=REAL).reshape(-1); n=hv.size
    JM=np.zeros((n,n),dtype=REAL)
    if isinstance(J,Mapping):
        for (i,j),v in J.items(): JM[int(i),int(j)]=JM[int(j),int(i)]=float(v)
    else:
        JM=np.asarray(J,dtype=REAL)
        if JM.shape!=(n,n): raise ValueError("J shape mismatch")
    return hv,JM


def worldline_susceptibility(h:Mapping[int,float]|Sequence[float],J:Mapping[tuple[int,int],float]|NDArray[np.float64],
                             *,A:float,B:float,config:WorldlineSQAConfig)->tuple[float,dict[str,Any]]:
    hv,JM=_ising_arrays(h,J); n=hv.size; M=int(config.replicas); beta=float(config.beta)
    if n<1 or M<2 or A<=0 or B<0: raise ValueError("invalid SQA inputs")
    rng=np.random.default_rng(int(config.seed)); spins=rng.choice([-1,1],size=(n,M)).astype(np.int8)
    x=max(1e-12,beta*float(A)/M)
    Jperp=0.5*math.log(1.0/math.tanh(x))
    spatial_scale=beta*float(B)/M
    mags=[]; accepts=0; attempts=0
    total_sweeps=int(config.burn_in_sweeps)+int(config.measurement_sweeps)
    for sweep in range(total_sweeps):
        for i in rng.permutation(n):
            for k in rng.permutation(M):
                s=int(spins[i,k])
                spatial=hv[i]+float(JM[i]@spins[:,k])
                # effective action change for flip; distribution proportional exp(-S_eff)
                dS=-2.0*spatial_scale*s*spatial + 2.0*Jperp*s*(int(spins[i,(k-1)%M])+int(spins[i,(k+1)%M]))
                attempts+=1
                if dS<=0 or rng.random()<math.exp(-dS):
                    spins[i,k]=-s; accepts+=1
        if sweep>=config.burn_in_sweeps and ((sweep-config.burn_in_sweeps)%max(1,int(config.thin))==0):
            mags.append(float(np.mean(spins)))
    m=np.asarray(mags,dtype=REAL)
    chi=float(n*M*(np.mean(m*m)-np.mean(np.abs(m))**2)) if m.size else float("nan")
    return max(0.0,chi),{"acceptance_rate":accepts/max(1,attempts),"n_measurements":int(m.size),"J_perp":Jperp}


def construct_worldline_schedule(h:Mapping[int,float]|Sequence[float],J:Mapping[tuple[int,int],float]|NDArray[np.float64],
                                 *,s_grid:ArrayLike,A_of_s:Callable[[NDArray[np.float64]],NDArray[np.float64]],
                                 B_of_s:Callable[[NDArray[np.float64]],NDArray[np.float64]],T_us:float,
                                 chi0:float=1e-6,config:WorldlineSQAConfig=WorldlineSQAConfig())->dict[str,Any]:
    s=np.asarray(s_grid,dtype=REAL).reshape(-1)
    if s.size<5 or abs(s[0])>1e-12 or abs(s[-1]-1)>1e-12 or np.any(np.diff(s)<=0):
        raise ValueError("s_grid must be strictly increasing from 0 to 1")
    A=np.asarray(A_of_s(s),dtype=REAL); B=np.asarray(B_of_s(s),dtype=REAL)
    chis=[]; diagnostics=[]
    # endpoints may have A~0; evaluate slightly inside to avoid singular Jperp, then copy nearest.
    for idx,(si,ai,bi) in enumerate(zip(s,A,B)):
        aa=max(float(ai),1e-6)
        cfg=WorldlineSQAConfig(**{**asdict(config),"seed":int(config.seed)+idx})
        chi,diag=worldline_susceptibility(h,J,A=aa,B=max(float(bi),0.0),config=cfg)
        chis.append(chi); diagnostics.append(diag)
    chi=np.asarray(chis,dtype=REAL); w=chi+float(chi0)
    cumulative=np.r_[0.0,cumulative_trapezoid(w,s)]
    if cumulative[-1]<=0: raise FullBenchError("worldline cumulative allocation is degenerate")
    tau=cumulative/cumulative[-1]
    u=np.linspace(0.0,1.0,s.size)
    s_of_t=np.interp(u,tau,s)
    t=float(T_us)*u
    return {"s_grid":s,"chi":chi,"weight":w,"tau":tau,"t_us":t,"schedule_s":s_of_t,"diagnostics":diagnostics}


# ---------------------------------------------------------------------------
# QZero original: four-stage MCTS + policy/value NN + pretraining/fine tuning.
# ---------------------------------------------------------------------------

@dataclass
class QueryCounter:
    environment_queries:int=0
    def inc(self,n:int=1):self.environment_queries+=int(n)


@dataclass
class FourierDiscreteActionSpace:
    order:int=5
    l:float=0.2
    delta:float=0.01
    def values(self)->NDArray[np.float64]:
        n=int(round(2*self.l/self.delta))+1
        return np.linspace(-self.l,self.l,n,dtype=REAL)
    @property
    def n_actions(self)->int:return int(self.values().size)
    def control_from_indices(self,indices:Sequence[int],T_us:float)->NDArray[np.float64]:
        vals=self.values(); idx=list(map(int,indices))
        if len(idx)!=self.order: raise ValueError("complete QZero path required")
        return np.asarray([float(T_us)]+[float(vals[i]) for i in idx],dtype=REAL)
    def indices_from_control(self,control:ArrayLike)->list[int]:
        c=np.asarray(control,dtype=REAL).reshape(-1); vals=self.values()
        coeff=c[1:] if c.size==self.order+1 else c
        if coeff.size!=self.order: raise ValueError("QZero paper control dimension mismatch")
        return [int(np.argmin(np.abs(vals-v))) for v in coeff]
    def state_vector(self,prefix:Sequence[int])->NDArray[np.float64]:
        # Original QZero encodes the partial coefficient vector with future
        # coefficients left at zero and appends Hamiltonian information.
        # No extra completion mask is injected into the paper-faithful state.
        vals=self.values(); x=np.zeros(self.order,dtype=REAL)
        for j,a in enumerate(prefix): x[j]=vals[int(a)]/max(self.l,1e-12)
        return x


@dataclass
class MatchedBoxDiscreteActionSpace:
    """QZero-compatible discrete representation of a continuous matched control box.

    Every decision depth has the same number of actions, but each depth maps the
    action index to its own physical interval. The first depth is annealing time;
    remaining depths are Fourier coefficients. Thus the algorithmic QZero tree,
    policy/value networks, visit-count policy and training logic are unchanged,
    while the matched arm receives exactly the CSSF T+theta dimensions/bounds.
    """
    bounds:NDArray[np.float64]
    n_actions_grid:int=41
    def __post_init__(self):
        self.bounds=np.asarray(self.bounds,dtype=REAL)
        if self.bounds.ndim!=2 or self.bounds.shape[1]!=2 or self.bounds.shape[0]<2: raise ValueError("matched bounds shape")
        if int(self.n_actions_grid)<3: raise ValueError("n_actions_grid must be >=3")
    @property
    def order(self)->int:return int(self.bounds.shape[0])  # decision depths = T + all theta coordinates
    @property
    def n_actions(self)->int:return int(self.n_actions_grid)
    def depth_values(self,depth:int)->NDArray[np.float64]:
        d=int(depth); lo,hi=map(float,self.bounds[d])
        if d==0: return np.geomspace(lo,hi,self.n_actions,dtype=REAL)
        return np.linspace(lo,hi,self.n_actions,dtype=REAL)
    def values(self)->NDArray[np.float64]:
        # Compatibility only: policy width is shared; physical values are depth-specific.
        return np.arange(self.n_actions,dtype=REAL)
    def control_from_indices(self,indices:Sequence[int],T_us:float|None=None)->NDArray[np.float64]:
        idx=list(map(int,indices))
        if len(idx)!=self.order: raise ValueError("complete matched QZero path required")
        return np.asarray([self.depth_values(d)[a] for d,a in enumerate(idx)],dtype=REAL)
    def indices_from_control(self,control:ArrayLike)->list[int]:
        c=np.asarray(control,dtype=REAL).reshape(-1)
        if c.size!=self.order: raise ValueError("matched QZero control dimension mismatch")
        return [int(np.argmin(np.abs(self.depth_values(d)-c[d]))) for d in range(self.order)]
    def state_vector(self,prefix:Sequence[int])->NDArray[np.float64]:
        x=np.zeros(self.order,dtype=REAL)
        for d,a in enumerate(prefix):
            vals=self.depth_values(d); v=float(vals[int(a)]); lo,hi=map(float,self.bounds[d]); x[d]=2.0*(v-lo)/(hi-lo)-1.0
        return x


@dataclass
class MCTSNode:
    prefix:tuple[int,...]
    parent:"MCTSNode|None"=None
    children:dict[int,"MCTSNode" ]=field(default_factory=dict)
    visits:int=0
    value_sum:float=0.0
    prior:float=0.0
    @property
    def q(self)->float:return self.value_sum/max(1,self.visits)


class PureMCTS:
    def __init__(self,space:FourierDiscreteActionSpace,environment:Callable[[NDArray[np.float64]],float],*,T_us:float,
                 C:float=2.0,Nexp:int=10,Nsim:int=5,seed:int=0,counter:QueryCounter|None=None,maximize_merit:bool=True):
        self.space=space; self.environment=environment; self.T_us=float(T_us); self.C=float(C); self.Nexp=int(Nexp); self.Nsim=int(Nsim)
        self.maximize_merit=bool(maximize_merit)
        self.rng=np.random.default_rng(int(seed)); self.counter=counter or QueryCounter(); self.root=MCTSNode(prefix=())
        self.best_indices:tuple[int,...]|None=None; self.best_reward=(-np.inf if self.maximize_merit else np.inf)
    def _ucb(self,parent:MCTSNode,child:MCTSNode)->float:
        if child.visits==0:return float("inf")
        return child.q+self.C*math.sqrt(2.0*math.log(max(1,parent.visits))/child.visits)
    def _select(self)->tuple[MCTSNode,list[MCTSNode]]:
        node=self.root; path=[node]
        while len(node.prefix)<self.space.order and node.children and len(node.children)>=self.space.n_actions:
            node=max(node.children.values(),key=lambda c:self._ucb(path[-1],c)); path.append(node)
        while len(node.prefix)<self.space.order and node.children:
            unvisited=[c for c in node.children.values() if c.visits==0]
            if unvisited: break
            node=max(node.children.values(),key=lambda c:self._ucb(path[-1],c)); path.append(node)
        return node,path
    def _expand(self,node:MCTSNode)->list[MCTSNode]:
        if len(node.prefix)>=self.space.order:return [node]
        available=[a for a in range(self.space.n_actions) if a not in node.children]
        self.rng.shuffle(available); out=[]
        for a in available[:self.Nexp]:
            child=MCTSNode(prefix=node.prefix+(int(a),),parent=node); node.children[int(a)]=child; out.append(child)
        return out or list(node.children.values())
    def _rollout(self,node:MCTSNode)->tuple[float,tuple[int,...]]:
        idx=list(node.prefix)
        while len(idx)<self.space.order: idx.append(int(self.rng.integers(self.space.n_actions)))
        control=self.space.control_from_indices(idx,self.T_us); reward=float(self.environment(control)); self.counter.inc()
        tup=tuple(idx)
        better = reward>self.best_reward if self.maximize_merit else reward<self.best_reward
        if better:self.best_reward=reward;self.best_indices=tup
        # UCB always maximizes node value; convert a minimization objective to search merit.
        search_merit = reward if self.maximize_merit else -reward
        return search_merit,tup
    def _backprop(self,node:MCTSNode,reward:float)->None:
        cur=node
        while cur is not None:
            cur.visits+=1;cur.value_sum+=float(reward);cur=cur.parent
    def search(self,episodes:int=20)->tuple[NDArray[np.float64],dict[str,Any]]:
        for _ in range(int(episodes)):
            node,path=self._select(); expanded=self._expand(node)
            for child in expanded:
                for _ in range(self.Nsim):
                    reward,_=self._rollout(child); self._backprop(child,reward)
        if self.best_indices is None: raise FullBenchError("MCTS produced no complete schedule")
        return self.space.control_from_indices(self.best_indices,self.T_us),{"best_reward":self.best_reward,"environment_queries":self.counter.environment_queries,"maximize_merit":self.maximize_merit}


def _torch():
    try: import torch; return torch
    except ImportError as exc: raise FullBenchError("PyTorch is required for QZero") from exc


class PolicyValueNet:
    """Original-QZero-style *separate* policy and value networks.

    Chen et al. describe a policy network with hidden widths 256, 128 and a
    policy output, and a value network with hidden widths 256, 128, 64 and a
    scalar value output.  Sharing a trunk would be a MindSpore-style/engineering
    simplification and is therefore deliberately prohibited in the v38
    reference-faithful baseline.
    """
    def __init__(self,input_dim:int,action_dim:int,seed:int=0):
        torch=_torch(); torch.manual_seed(int(seed)); nn=torch.nn
        class PolicyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net=nn.Sequential(
                    nn.Linear(input_dim,256),nn.ReLU(),
                    nn.Linear(256,128),nn.ReLU(),
                    nn.Linear(128,action_dim),
                )
            def forward(self,x): return self.net(x)
        class ValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net=nn.Sequential(
                    nn.Linear(input_dim,256),nn.ReLU(),
                    nn.Linear(256,128),nn.ReLU(),
                    nn.Linear(128,64),nn.ReLU(),
                    nn.Linear(64,1),nn.Tanh(),
                )
            def forward(self,x): return self.net(x).squeeze(-1)
        self.policy_model=PolicyNet()
        self.value_model=ValueNet()

    def predict(self,x:ArrayLike)->tuple[NDArray[np.float64],float]:
        torch=_torch(); self.policy_model.eval(); self.value_model.eval()
        with torch.no_grad():
            t=torch.as_tensor(np.asarray(x,dtype=np.float32)).reshape(1,-1)
            logits=self.policy_model(t); v=self.value_model(t)
            p=torch.softmax(logits,dim=-1)[0].cpu().numpy()
        return p.astype(REAL),float(v.item())

    def train_supervised(self,X:ArrayLike,policy_targets:ArrayLike,value_targets:ArrayLike,*,epochs:int=30,lr_start:float=0.008,lr_end:float=0.0008,l2:float=1e-4)->list[float]:
        torch=_torch(); X=np.asarray(X,dtype=np.float32); P=np.asarray(policy_targets,dtype=np.float32); V=np.asarray(value_targets,dtype=np.float32).reshape(-1)
        params=list(self.policy_model.parameters())+list(self.value_model.parameters())
        opt=torch.optim.Adam(params,lr=float(lr_start),weight_decay=float(l2))
        xt=torch.tensor(X); pt=torch.tensor(P); vt=torch.tensor(V)
        losses=[]
        for e in range(int(epochs)):
            frac=e/max(1,epochs-1); lr=lr_start*(lr_end/lr_start)**frac
            for g in opt.param_groups:g["lr"]=lr
            self.policy_model.train(); self.value_model.train(); opt.zero_grad()
            logits=self.policy_model(xt); v=self.value_model(xt)
            logp=torch.log_softmax(logits,dim=-1)
            policy_loss=-(pt*logp).sum(dim=1).mean()
            value_loss=((v-vt)**2).mean()
            loss=policy_loss+value_loss
            loss.backward(); opt.step(); losses.append(float(loss.item()))
        return losses

    def architecture_manifest(self)->dict[str,Any]:
        return {
            "policy_hidden":[256,128],
            "value_hidden":[256,128,64],
            "shared_trunk":False,
            "separate_policy_value_networks":True,
        }


class QZero:
    def __init__(self,space:FourierDiscreteActionSpace,context_dim:int,*,T_us:float,seed:int=0,
                 C_start:float=3.0,C_end:float=0.5,Nplayout:int=6,epsilon:float=0.01,maximize_merit:bool=True):
        self.space=space;self.context_dim=int(context_dim);self.T_us=float(T_us);self.seed=int(seed);self.C_start=float(C_start);self.C_end=float(C_end);self.Nplayout=int(Nplayout);self.epsilon=float(epsilon);self.maximize_merit=bool(maximize_merit)
        self.input_dim=int(np.asarray(space.state_vector([]),dtype=REAL).size)+self.context_dim;self.action_dim=space.order*space.n_actions
        self.net=PolicyValueNet(self.input_dim,self.action_dim,seed=seed);self.counter=QueryCounter();self.pretraining_queries=0;self.target_queries=0
        self.rng=np.random.default_rng(seed)
    def _input(self,prefix:Sequence[int],context:ArrayLike)->NDArray[np.float64]:return np.r_[self.space.state_vector(prefix),np.asarray(context,dtype=REAL).reshape(-1)]
    def build_pretraining_data(self,contexts:Sequence[ArrayLike],environments:Sequence[Callable[[NDArray[np.float64]],float]],*,mcts_episodes:int=5)->tuple[NDArray,NDArray,NDArray]:
        X=[];P=[];V=[]
        for idx,(ctx,env) in enumerate(zip(contexts,environments)):
            counter=QueryCounter();mcts=PureMCTS(self.space,env,T_us=self.T_us,C=2.0,Nexp=10,Nsim=5,seed=self.seed+idx,counter=counter,maximize_merit=self.maximize_merit)
            control,diag=mcts.search(episodes=mcts_episodes);chosen=self.space.indices_from_control(control)
            self.pretraining_queries+=counter.environment_queries
            for depth in range(self.space.order):
                prefix=chosen[:depth]; x=self._input(prefix,ctx); p=np.zeros(self.action_dim,dtype=REAL); p[depth*self.space.n_actions+chosen[depth]]=1.0
                X.append(x);P.append(p);V.append(1.0)
        return np.asarray(X,dtype=REAL),np.asarray(P,dtype=REAL),np.asarray(V,dtype=REAL)
    def pretrain(self,X:ArrayLike,P:ArrayLike,V:ArrayLike,*,epochs:int=30)->list[float]:return self.net.train_supervised(X,P,V,epochs=epochs)
    def _guided_episode(self,context:ArrayLike,environment:Callable[[NDArray[np.float64]],float],episode_idx:int,total_episodes:int,terminal_success:Callable[[float],bool])->tuple[NDArray[np.float64],list[tuple[NDArray,np.ndarray,float]],float]:
        root=MCTSNode(prefix=());trajectory=[];C=self.C_start*(self.C_end/self.C_start)**(episode_idx/max(1,total_episodes-1))
        for depth in range(self.space.order):
            # Run Nplayout simulations from current root. Partial leaves use value NN; complete leaves query environment.
            for _ in range(self.Nplayout):
                node=root;path=[node]
                while len(node.prefix)<self.space.order and node.children:
                    inp=self._input(node.prefix,context); policy,_=self.net.predict(inp); seg=policy[len(node.prefix)*self.space.n_actions:(len(node.prefix)+1)*self.space.n_actions]
                    parent_vis=max(1,sum(c.visits for c in node.children.values()))
                    def score(c):
                        prior=float(seg[c.prefix[-1]]) if seg.size else 1.0/self.space.n_actions
                        return c.q+C*prior*math.sqrt(parent_vis)/(1+c.visits)
                    node=max(node.children.values(),key=score);path.append(node)
                if len(node.prefix)<self.space.order:
                    inp=self._input(node.prefix,context);policy,vpred=self.net.predict(inp);seg=policy[len(node.prefix)*self.space.n_actions:(len(node.prefix)+1)*self.space.n_actions]
                    if not node.children:
                        for a in range(self.space.n_actions):node.children[a]=MCTSNode(prefix=node.prefix+(a,),parent=node,prior=float(seg[a]))
                    child=max(node.children.values(),key=lambda c:c.prior if c.visits==0 else -1.0);node=child;path.append(node)
                if len(node.prefix)==self.space.order:
                    reward=float(environment(self.space.control_from_indices(node.prefix,self.T_us)));self.counter.inc();self.target_queries+=1;v=(1.0 if bool(terminal_success(reward)) else -1.0)
                else:
                    _,v=self.net.predict(self._input(node.prefix,context))
                for pnode in path:pnode.visits+=1;pnode.value_sum+=float(v)
            # actual move from visit-count policy
            if not root.children:
                for a in range(self.space.n_actions):root.children[a]=MCTSNode(prefix=root.prefix+(a,),parent=root)
            visits=np.asarray([root.children[a].visits for a in range(self.space.n_actions)],dtype=REAL);pi=visits/(visits.sum()+1e-12)
            a=int(np.argmax(visits));trajectory.append((self._input(root.prefix,context),self._global_policy(depth,pi),0.0));root=root.children[a]
        control=self.space.control_from_indices(root.prefix,self.T_us);reward=float(environment(control));self.counter.inc();self.target_queries+=1
        terminal_z=(1.0 if bool(terminal_success(reward)) else -1.0)
        trajectory=[(x,p,terminal_z) for x,p,_ in trajectory]
        return control,trajectory,reward
    def _global_policy(self,depth:int,pi:NDArray[np.float64])->NDArray[np.float64]:
        p=np.zeros(self.action_dim,dtype=REAL);p[depth*self.space.n_actions:(depth+1)*self.space.n_actions]=pi;return p
    def fine_tune(self,context:ArrayLike,environment:Callable[[NDArray[np.float64]],float],*,terminal_success:Callable[[float],bool],episodes:int=4,epochs_per_round:int=10)->tuple[NDArray[np.float64],dict[str,Any]]:
        if not callable(terminal_success): raise FidelityError("QZero requires an explicit terminal-success predicate; continuous reward clipping is forbidden")
        best=None;best_reward=(-np.inf if self.maximize_merit else np.inf);losses=[]
        for e in range(int(episodes)):
            control,traj,reward=self._guided_episode(context,environment,e,episodes,terminal_success)
            X=np.asarray([t[0] for t in traj]);P=np.asarray([t[1] for t in traj]);V=np.asarray([t[2] for t in traj]);losses.extend(self.net.train_supervised(X,P,V,epochs=epochs_per_round))
            better=reward>best_reward if self.maximize_merit else reward<best_reward
            if better:best_reward=reward;best=control.copy()
        if best is None:raise FullBenchError("QZero fine-tuning produced no control")
        return best,{"best_reward":best_reward,"pretraining_queries":self.pretraining_queries,"target_queries":self.target_queries,"total_environment_queries":self.counter.environment_queries+self.pretraining_queries,"loss_final":losses[-1] if losses else None,"terminal_value_semantics":"binary_pm1"}

    def fine_tune_to_convergence(self,context:ArrayLike,environment:Callable[[NDArray[np.float64]],float],*,
                                 min_rounds:int=8,max_rounds:int=64,epochs_per_round:int=10,
                                 loss_tolerance:float=1e-2,patience:int=3,terminal_success:Callable[[float],bool]|None=None)->tuple[NDArray[np.float64],dict[str,Any]]:
        """Full QZero target adaptation with an explicit convergence gate.

        The paper describes repeated MCTS-guided policy/value recalibration until a steady
        state is reached. A tiny fixed number of rounds is therefore not claim-grade. This
        implementation repeats complete guided episodes and retraining until a frozen loss
        tolerance is satisfied for ``patience`` consecutive rounds, or fails closed at
        ``max_rounds``.
        """
        min_rounds=int(min_rounds);max_rounds=int(max_rounds);epochs_per_round=int(epochs_per_round);patience=int(patience)
        if min_rounds<1 or max_rounds<min_rounds or epochs_per_round<1 or patience<1 or not (loss_tolerance>0):
            raise ValueError("invalid QZero convergence protocol")
        if not callable(terminal_success): raise FidelityError("QZero requires an explicit terminal-success predicate; original terminal z is binary ±1")
        best=None;best_reward=(-np.inf if self.maximize_merit else np.inf);round_losses=[];stable=0
        for e in range(max_rounds):
            control,traj,reward=self._guided_episode(context,environment,e,max_rounds,terminal_success)
            X=np.asarray([t[0] for t in traj]);P=np.asarray([t[1] for t in traj]);V=np.asarray([t[2] for t in traj])
            losses=self.net.train_supervised(X,P,V,epochs=epochs_per_round);last=float(losses[-1]);round_losses.append(last)
            better=reward>best_reward if self.maximize_merit else reward<best_reward
            if better:best_reward=float(reward);best=control.copy()
            stable = stable+1 if last<=float(loss_tolerance) else 0
            if (e+1)>=min_rounds and stable>=patience:
                if best is None: raise FullBenchError("QZero convergence reached without a control")
                return best,{"best_reward":best_reward,"pretraining_queries":self.pretraining_queries,"target_queries":self.target_queries,
                             "total_environment_queries":self.counter.environment_queries+self.pretraining_queries,"loss_final":last,
                             "fine_tune_rounds":e+1,"converged":True,"loss_tolerance":float(loss_tolerance),"patience":patience,
                             "round_losses":round_losses,"terminal_value_semantics":"binary_pm1","maximize_merit":self.maximize_merit}
        raise FidelityError(f"QZero target adaptation did not reach the frozen convergence tolerance {loss_tolerance} within {max_rounds} full rounds")


# ---------------------------------------------------------------------------
# Strong classical tuning wrappers (official D-Wave samplers when installed).
# ---------------------------------------------------------------------------

def tune_dwave_sa(bqm:Any,*,train_seeds:Sequence[int],num_reads:int,config_grid:Sequence[Mapping[str,Any]],score:Callable[[Any],float])->dict[str,Any]:
    try: from dwave.samplers import SimulatedAnnealingSampler
    except ImportError as exc: raise FullBenchError("dwave-samplers is required for Strong-SA") from exc
    sampler=SimulatedAnnealingSampler();records=[]
    for cfg in config_grid:
        vals=[];start=time.perf_counter()
        for seed in train_seeds:
            ss=sampler.sample(bqm,num_reads=int(num_reads),seed=int(seed),**dict(cfg));vals.append(float(score(ss)))
        records.append({"config":dict(cfg),"mean_score":float(np.mean(vals)),"seconds":time.perf_counter()-start})
    best=max(records,key=lambda r:r["mean_score"])
    return {"best_config":best["config"],"tuning_records":records,"tuning_seconds":float(sum(r["seconds"] for r in records))}


def tune_dwave_tabu(bqm:Any,*,train_seeds:Sequence[int],num_reads:int,config_grid:Sequence[Mapping[str,Any]],score:Callable[[Any],float])->dict[str,Any]:
    try: from dwave.samplers import TabuSampler
    except ImportError as exc: raise FullBenchError("dwave-samplers is required for Strong-Tabu") from exc
    sampler=TabuSampler();records=[]
    for cfg in config_grid:
        vals=[];start=time.perf_counter()
        for seed in train_seeds:
            ss=sampler.sample(bqm,num_reads=int(num_reads),seed=int(seed),**dict(cfg));vals.append(float(score(ss)))
        records.append({"config":dict(cfg),"mean_score":float(np.mean(vals)),"seconds":time.perf_counter()-start})
    best=max(records,key=lambda r:r["mean_score"])
    return {"best_config":best["config"],"tuning_records":records,"tuning_seconds":float(sum(r["seconds"] for r in records))}



# ---------------------------------------------------------------------------
# Schedule-sensitive CUDA SQA extension of the project Pegasus-P16 emulator.
# ---------------------------------------------------------------------------


def calibrated_sqa_drive_schedule(
    anneal_t_us: ArrayLike,
    anneal_s: ArrayLike,
    calibration: Mapping[str, Any],
    *,
    anneal_steps: int,
    beta_range: tuple[float, float] = (0.10, 5.00),
) -> dict[str, Any]:
    """Pure, CPU-testable mapping from requested s(t) and A/B table to SQA drive.

    This function performs no sampling.  It exists so the exact schedule semantics
    used by the CUDA backend can be regression-tested on hosts without a GPU.
    """
    t=np.asarray(anneal_t_us,dtype=REAL).reshape(-1)
    ss=np.asarray(anneal_s,dtype=REAL).reshape(-1)
    if t.size<2 or t.size!=ss.size or np.any(~np.isfinite(t)) or np.any(~np.isfinite(ss)):
        raise HardwareGateError("anneal schedule must contain finite equal-length t/s arrays")
    if abs(float(t[0]))>1e-12 or np.any(np.diff(t)<=0) or float(t[-1])<=0:
        raise HardwareGateError("anneal schedule times must start at zero and increase strictly")
    if np.any(ss < -1e-12) or np.any(ss > 1.0+1e-12):
        raise HardwareGateError("anneal schedule s values must lie in [0,1]")
    steps=int(anneal_steps)
    if steps<1: raise HardwareGateError("anneal_steps must be positive")
    beta0,beta1=map(float,beta_range)
    if not (0<beta0<beta1): raise HardwareGateError("beta_range must be positive ascending")
    cs=np.asarray(calibration.get("s"),dtype=REAL).reshape(-1)
    ca=np.asarray(calibration.get("A_GHz"),dtype=REAL).reshape(-1)
    cb=np.asarray(calibration.get("B_GHz"),dtype=REAL).reshape(-1)
    if cs.size<2 or not (cs.size==ca.size==cb.size) or np.any(np.diff(cs)<=0):
        raise HardwareGateError("invalid frozen A/B calibration arrays")
    if abs(float(cs[0]))>1e-12 or abs(float(cs[-1])-1.0)>1e-6:
        raise HardwareGateError("calibration s grid must cover [0,1]")
    base_beta=np.geomspace(beta0,beta1,steps,dtype=REAL)
    t_eval=np.linspace(float(t[0]),float(t[-1]),steps,dtype=REAL)
    s_eval=np.interp(t_eval,t,ss)
    A=np.interp(s_eval,cs,ca); B=np.interp(s_eval,cs,cb)
    Eref=max(float(np.max(np.abs(ca))),float(np.max(np.abs(cb))),1e-12)
    a=A/Eref; b=B/Eref
    eps=np.finfo(np.float64).eps**0.5
    b_safe=np.maximum(b,eps)
    beta_eff=base_beta*b_safe
    field_eff=a/b_safe
    product_error=float(np.max(np.abs(beta_eff*field_eff-base_beta*a)))
    if product_error>1e-10:
        raise HardwareGateError(f"A/B schedule mapping invariant failed: {product_error:.3e}")
    return {
        "t_eval_us":t_eval,"s_eval":s_eval,"A_GHz":A,"B_GHz":B,
        "base_beta":base_beta,"beta_eff":beta_eff,"field_eff":field_eff,
        "energy_reference_GHz":Eref,"mapping_product_max_abs_error":product_error,
        "schedule_sha256":stable_hash({"t_us":t.tolist(),"s":ss.tolist()},prefix="CSSF-SQA-SCHEDULE-v38"),
    }


def cuda_schedule_sqa_embedded_bqm(
    embedded_bqm: Any,
    *,
    anneal_t_us: ArrayLike,
    anneal_s: ArrayLike,
    calibration: Mapping[str, Any],
    num_reads: int,
    trotter_replicas: int = 96,
    sweeps: int = 4000,
    burn_in_sweeps: int = 1000,
    beta_range: tuple[float, float] = (0.10, 5.00),
    seed: int = 0,
    memory_fraction: float = 0.72,
    raw_pegasus_sampler: Any | None = None,
) -> Any:
    """Sample an *embedded* BQM with the project CUDA SQA transition kernel.

    The production project emulator originally anneals along geometric beta and
    transverse-field ramps.  This claim-grade extension keeps the same exact
    color/parity CUDA Metropolis transition and Pegasus-P16 structured graph,
    but drives it with the requested PWL schedule through the frozen per-system
    D-Wave calibration tables A(s), B(s):

        beta_problem(k) = beta_base(k) * B(s_k) / E_ref
        Gamma(k)        = [A(s_k)/E_ref] / [B(s_k)/E_ref]

    so beta_problem * Gamma = beta_base * A/E_ref while the Ising contribution
    is weighted by B/E_ref.  This is a schedule-sensitive SQA *reference model*,
    not a claim that SQA reproduces open-system D-Wave dynamics exactly.

    CPU fallback is deliberately absent.  CUDA unavailability or OOM raises a
    fail-closed HardwareGateError and never changes reads/replicas/sweeps.
    """
    # Import the exact transition machinery from the frozen project source.
    try:
        import dimod
        import torch
        from dwave_backend import sampler as project_sqa
    except Exception as exc:
        raise HardwareGateError("CUDA Pegasus SQA dependencies are unavailable") from exc

    require_cuda_claim_runtime()
    if not torch.cuda.is_available():
        raise HardwareGateError("CUDA is required for schedule-sensitive Pegasus SQA")

    t=np.asarray(anneal_t_us,dtype=REAL).reshape(-1)
    ss=np.asarray(anneal_s,dtype=REAL).reshape(-1)
    if t.size<2 or t.size!=ss.size or np.any(~np.isfinite(t)) or np.any(~np.isfinite(ss)):
        raise HardwareGateError("anneal schedule must contain finite equal-length t/s arrays")
    if abs(float(t[0]))>1e-12 or np.any(np.diff(t)<=0) or float(t[-1])<=0:
        raise HardwareGateError("anneal schedule times must start at zero and increase strictly")
    if np.any(ss < -1e-12) or np.any(ss > 1.0+1e-12):
        raise HardwareGateError("anneal schedule s values must lie in [0,1]")

    reads=int(num_reads); replicas=int(trotter_replicas); total_sweeps=int(sweeps); burn=int(burn_in_sweeps)
    if reads<1 or replicas<2 or replicas%2 or total_sweeps<=burn or burn<0:
        raise HardwareGateError("invalid CUDA SQA scientific parameters")
    beta0,beta1=map(float,beta_range)
    if not (0<beta0<beta1): raise HardwareGateError("beta_range must be positive ascending")

    cs=np.asarray(calibration.get("s"),dtype=REAL).reshape(-1)
    ca=np.asarray(calibration.get("A_GHz"),dtype=REAL).reshape(-1)
    cb=np.asarray(calibration.get("B_GHz"),dtype=REAL).reshape(-1)
    if cs.size<2 or not (cs.size==ca.size==cb.size) or np.any(np.diff(cs)<=0):
        raise HardwareGateError("invalid frozen A/B calibration arrays")
    if abs(float(cs[0]))>1e-12 or abs(float(cs[-1])-1.0)>1e-6:
        raise HardwareGateError("calibration s grid must cover [0,1]")

    variables=tuple(embedded_bqm.variables)
    if not variables: raise HardwareGateError("embedded BQM is empty")
    if raw_pegasus_sampler is not None:
        props=getattr(raw_pegasus_sampler,"properties",{}) or {}
        topology=props.get("topology",{}) if isinstance(props,Mapping) else {}
        if str(topology.get("type","")).lower()!="pegasus":
            raise HardwareGateError("schedule-sensitive CUDA SQA requires Pegasus structured sampler")
        raw_nodes=set(getattr(raw_pegasus_sampler,"nodelist",()))
        if raw_nodes and any(v not in raw_nodes for v in variables):
            raise HardwareGateError("embedded BQM contains nodes outside frozen Pegasus topology")
        raw_edges={frozenset(e) for e in getattr(raw_pegasus_sampler,"edgelist",())}
        if raw_edges and any(frozenset(e) not in raw_edges for e in embedded_bqm.quadratic):
            raise HardwareGateError("embedded BQM contains couplers outside frozen Pegasus topology")

    binary_bqm=embedded_bqm.change_vartype(dimod.BINARY,inplace=False)
    h_map,j_map,_offset=binary_bqm.to_ising()
    variable_index={v:i for i,v in enumerate(variables)}
    h=np.asarray([float(h_map.get(v,0.0)) for v in variables],dtype=REAL)
    edge_pairs=[]; edge_bias=[]
    for (u,v),bias in j_map.items():
        edge_pairs.append((variable_index[u],variable_index[v])); edge_bias.append(float(bias))
    edge_u=np.asarray([e[0] for e in edge_pairs],dtype=np.int64)
    edge_v=np.asarray([e[1] for e in edge_pairs],dtype=np.int64)
    edge_j=np.asarray(edge_bias,dtype=REAL)
    color_groups=project_sqa._active_graph_coloring(len(variables),edge_pairs)

    device="cuda"
    generator=torch.Generator(device=device); generator.manual_seed(int(seed))
    h_tensor=torch.as_tensor(h,dtype=torch.float32,device=device)
    eu=torch.as_tensor(edge_u,dtype=torch.long,device=device)
    ev=torch.as_tensor(edge_v,dtype=torch.long,device=device)
    ej=torch.as_tensor(edge_j,dtype=torch.float32,device=device)
    plan_arrays=project_sqa._incremental_color_plan_arrays(
        variable_count=len(variables),edge_u=edge_u,edge_v=edge_v,edge_j=edge_j,color_groups=color_groups)
    color_plans=tuple((
        torch.as_tensor(color,dtype=torch.long,device=device),
        torch.as_tensor(src,dtype=torch.long,device=device),
        torch.as_tensor(dst,dtype=torch.long,device=device),
        torch.as_tensor(w,dtype=torch.float32,device=device),
    ) for color,src,dst,w in plan_arrays)
    parity_plans=tuple((
        torch.arange(parity,replicas,2,dtype=torch.long,device=device),
        torch.remainder(torch.arange(parity,replicas,2,dtype=torch.long,device=device)-1,replicas),
        torch.remainder(torch.arange(parity,replicas,2,dtype=torch.long,device=device)+1,replicas),
    ) for parity in (0,1))

    anneal_steps=total_sweeps-burn
    drive=calibrated_sqa_drive_schedule(t,ss,calibration,anneal_steps=anneal_steps,beta_range=(beta0,beta1))
    base_beta=np.asarray(drive["base_beta"],dtype=REAL); beta_eff=np.asarray(drive["beta_eff"],dtype=REAL)
    field_eff=np.asarray(drive["field_eff"],dtype=REAL); Eref=float(drive["energy_reference_GHz"])
    product_error=float(drive["mapping_product_max_abs_error"])
    eps=np.finfo(np.float64).eps**0.5

    project_sqa._release_cuda_cache(torch)
    free_bytes,_total_bytes=torch.cuda.mem_get_info()
    batch_size=project_sqa._estimate_fused_gpu_batch_size(
        free_bytes=int(free_bytes),num_reads=reads,replicas=replicas,
        variable_count=len(variables),memory_fraction=float(memory_fraction))
    batch_size=max(1,min(int(batch_size),reads))

    batches=[]; completed=0; local_rebuilds=0
    start=time.perf_counter()
    while completed<reads:
        current=min(batch_size,reads-completed)
        try:
            state=torch.randint(0,2,(current,replicas,len(variables)),dtype=torch.int8,device=device,generator=generator)
            state.mul_(2).sub_(1)
            local=project_sqa._problem_local_fields_torch(torch,state,h_tensor,eu,ev,ej)
            local_rebuilds+=1; sweep_counter=0

            # Burn-in at the first requested physical schedule point; no hidden alternative path.
            a0=float(np.interp(float(ss[0]),cs,ca)/Eref)
            b0=float(np.interp(float(ss[0]),cs,cb)/Eref)
            b0s=max(b0,eps); burn_beta=beta0*b0s; burn_field=a0/b0s
            for _ in range(burn):
                if sweep_counter>0 and sweep_counter % project_sqa.SQA_LOCAL_FIELD_REBASE_INTERVAL==0:
                    local=project_sqa._problem_local_fields_torch(torch,state,h_tensor,eu,ev,ej); local_rebuilds+=1
                project_sqa._single_sqa_sweep_incremental_torch(
                    torch=torch,state=state,local=local,color_plans=color_plans,parity_plans=parity_plans,
                    beta=float(burn_beta),transverse_field=float(burn_field),generator=generator)
                sweep_counter+=1

            for be,fe in zip(beta_eff,field_eff,strict=True):
                if sweep_counter>0 and sweep_counter % project_sqa.SQA_LOCAL_FIELD_REBASE_INTERVAL==0:
                    local=project_sqa._problem_local_fields_torch(torch,state,h_tensor,eu,ev,ej); local_rebuilds+=1
                project_sqa._single_sqa_sweep_incremental_torch(
                    torch=torch,state=state,local=local,color_plans=color_plans,parity_plans=parity_plans,
                    beta=float(be),transverse_field=float(fe),generator=generator)
                sweep_counter+=1

            replica_sum=state.to(dtype=torch.int16).sum(dim=1)
            majority=torch.where(replica_sum>0,torch.ones_like(replica_sum,dtype=torch.int8),
                                 torch.where(replica_sum<0,-torch.ones_like(replica_sum,dtype=torch.int8),state[:,0,:]))
            binary=((majority+1)//2).to(dtype=torch.int8)
            batches.append(binary.detach().cpu().numpy().astype(np.int8,copy=False)); completed+=current
            del state,local,binary,majority,replica_sum
        except RuntimeError as exc:
            # OOM is fail-closed: never shrink scientific parameters or fall back to CPU.
            if "out of memory" in str(exc).lower():
                raise HardwareGateError(
                    "CUDA OOM in claim-grade schedule-sensitive SQA; scientific parameters were NOT reduced"
                ) from exc
            raise
    elapsed=time.perf_counter()-start
    samples=np.concatenate(batches,axis=0)
    energies=np.asarray(binary_bqm.energies(samples),dtype=REAL)
    info={
        "cssf_backend":"local_sqa_gpu_schedule_sensitive",
        "provider":"cssf_torch_sqa_gpu",
        "device":"cuda",
        "topology_type":"pegasus",
        "schedule_sensitive":True,
        "schedule_mapping":"beta*B_and_A_over_B_exact_product",
        "schedule_sha256":stable_hash({"t_us":t.tolist(),"s":ss.tolist()},prefix="CSSF-SQA-SCHEDULE-v38"),
        "calibration_family":str(calibration.get("family","")),
        "calibration_sha256":str(calibration.get("sha256","")),
        "num_reads":reads,"trotter_replicas":replicas,"sweeps":total_sweeps,"burn_in_sweeps":burn,
        "beta_range":[beta0,beta1],"energy_reference_GHz":Eref,
        "mapping_product_max_abs_error":product_error,"cuda_batch_size":batch_size,
        "local_field_rebuilds":local_rebuilds,"elapsed_seconds":elapsed,
        "classical_fallback":False,
    }
    return dimod.SampleSet.from_samples((samples,list(variables)),vartype=dimod.BINARY,energy=energies,info=info)

# ---------------------------------------------------------------------------
# Calibration and Pegasus hardware gates.
# ---------------------------------------------------------------------------

SYSTEM4_CALIBRATION_SHA256="03350bb86bab2f752697e1a8c37f3e4c2100c6596d0f6c6bf8f6d2e3e97de4f1"
SYSTEM6_CALIBRATION_SHA256="d266ee71c8a0611cc392781da4df65e20969aed658b5df60453ac099202fdc06"
ADVANTAGE2_REJECT_SHA256="80292381a6d7f3b7b4826e1ae9cb316a878931474b96c3e11d82422395bdd678"


def file_sha256(path:Path|str)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def load_ab_calibration(path:Path|str,expected_family:str)->dict[str,Any]:
    path=Path(path); digest=file_sha256(path)
    if digest==ADVANTAGE2_REJECT_SHA256: raise HardwareGateError("Advantage2 calibration is OUT-OF-SCOPE / DO-NOT-USE")
    expected_hash={"Advantage_system4":SYSTEM4_CALIBRATION_SHA256,"Advantage_system6":SYSTEM6_CALIBRATION_SHA256}.get(expected_family)
    if expected_hash is None: raise HardwareGateError("Only Advantage_system4/system6 calibration families are permitted")
    if digest!=expected_hash: raise HardwareGateError(f"Calibration hash mismatch for {expected_family}")
    # Dependency-free, deterministic XLSX parsing of the frozen D-Wave table.
    # We intentionally read only the named worksheet and first four columns.
    import zipfile
    import xml.etree.ElementTree as ET
    ns={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pr":"http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(path,"r") as zf:
        wb_root=ET.fromstring(zf.read("xl/workbook.xml"))
        rel_root=ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rels={el.attrib["Id"]:el.attrib["Target"] for el in rel_root.findall("pr:Relationship",ns)}
        sheet_target=None
        for sh in wb_root.findall("m:sheets/m:sheet",ns):
            if sh.attrib.get("name")=="Standard-Annealing Schedule":
                rid=sh.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                sheet_target=rels.get(rid)
                break
        if not sheet_target: raise HardwareGateError("Standard-Annealing Schedule worksheet is missing")
        sheet_path="xl/"+sheet_target.lstrip("/") if not sheet_target.startswith("xl/") else sheet_target
        shared=[]
        if "xl/sharedStrings.xml" in zf.namelist():
            ssroot=ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ssroot.findall("m:si",ns):
                shared.append("".join(t.text or "" for t in si.iterfind(".//m:t",ns)))
        root=ET.fromstring(zf.read(sheet_path))
        rows=[]
        for row in root.findall(".//m:sheetData/m:row",ns):
            if int(row.attrib.get("r","0")) < 2: continue
            vals=[None,None,None,None]
            for cell in row.findall("m:c",ns):
                ref=cell.attrib.get("r","")
                letters="".join(ch for ch in ref if ch.isalpha())
                if letters not in {"A","B","C","D"}: continue
                idx={"A":0,"B":1,"C":2,"D":3}[letters]
                v=cell.find("m:v",ns)
                if v is None: continue
                text=v.text or ""
                if cell.attrib.get("t")=="s":
                    text=shared[int(text)]
                vals[idx]=text
            if vals[0] is not None:
                rows.append([float(vals[0]),float(vals[1]),float(vals[2]),float(vals[3])])
        arr=np.asarray(rows,dtype=REAL)
    if arr.shape[0]<100 or np.any(np.diff(arr[:,0])<=0) or abs(arr[0,0])>1e-12 or abs(arr[-1,0]-1)>1e-6: raise HardwareGateError("Malformed A/B calibration grid")
    return {"family":expected_family,"sha256":digest,"s":arr[:,0],"A_GHz":arr[:,1],"B_GHz":arr[:,2],"c":arr[:,3],"path":str(path)}


def assert_pegasus_sampler(raw_sampler:Any,*,requested_solver:str,required_graph_id:str)->dict[str,Any]:
    if not requested_solver.startswith(("Advantage_system4","Advantage_system6")):
        raise HardwareGateError("Only Advantage_system4.* or Advantage_system6.* is permitted")
    props=getattr(raw_sampler,"properties",{}) or {}; topology=props.get("topology",{}) if isinstance(props,Mapping) else {}
    if str(topology.get("type","")).lower()!="pegasus": raise HardwareGateError("Live solver is not Pegasus")
    solver_id=str(getattr(raw_sampler,"solver",getattr(raw_sampler,"id",requested_solver)))
    # Ocean objects vary; requested exact name remains authoritative and is checked by DWaveSampler constructor/project backend.
    graph_id=str(props.get("graph_id","") or props.get("problem_timing_data",{}).get("graph_id","") or required_graph_id)
    if not required_graph_id: raise HardwareGateError("graph_id must be explicitly frozen before claim-grade QPU execution")
    if graph_id and graph_id!=required_graph_id: raise HardwareGateError(f"graph_id mismatch: live={graph_id!r}, required={required_graph_id!r}")
    return {"requested_solver":requested_solver,"graph_id":required_graph_id,"topology":"pegasus","topology_shape":topology.get("shape")}


# ---------------------------------------------------------------------------
# Runtime / semantic helpers.
# ---------------------------------------------------------------------------

def require_cuda_claim_runtime()->dict[str,Any]:
    import subprocess,platform,sys
    try:
        probe=subprocess.run(["nvidia-smi"],text=True,capture_output=True)
    except FileNotFoundError as exc:
        raise HardwareGateError("NVIDIA GPU is required; nvidia-smi is unavailable and CPU fallback is prohibited") from exc
    if probe.returncode!=0: raise HardwareGateError("NVIDIA GPU is required; CPU fallback is prohibited")
    try: import torch
    except ImportError as exc: raise HardwareGateError("PyTorch must be importable before scientific installation") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count()<1: raise HardwareGateError("CUDA is not available to PyTorch")
    x=torch.tensor([1.,2.,3.],dtype=torch.float64,device="cuda");val=(x*x).sum()
    if not bool(torch.isfinite(val)): raise HardwareGateError("CUDA float64 probe failed")
    return {"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0),"capability":list(torch.cuda.get_device_capability(0))}


def source_fingerprint(obj:Any)->str:
    import inspect
    try: src=inspect.getsource(obj)
    except Exception: src=repr(obj)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


ALGORITHM_SOURCE_HASHES={
    "JeongTuRBO":source_fingerprint(JeongTuRBO),
    "FinzgarBO":source_fingerprint(FinzgarBO),
    "SequentialGPEI":source_fingerprint(SequentialGPEI),
    "QZero":source_fingerprint(QZero),
    "construct_worldline_schedule":source_fingerprint(construct_worldline_schedule),
    "PeriodicGP":source_fingerprint(PeriodicGP),
    "TorusSpectralMaternGP":source_fingerprint(TorusSpectralMaternGP),
}
ALGORITHM_HASH=stable_hash(ALGORITHM_SOURCE_HASHES,prefix="CSSF-COMPETITOR-SOURCE-HASHES-v38")


def default_fidelity_manifests(reference_pass:bool=False)->dict[str,FidelityManifest]:
    out={}
    for name,spec in BENCHMARK_REGISTRY.items():
        out[name]=FidelityManifest(method=name,mechanisms={m:True for m in spec["required"]},reference_reproduction_pass=bool(reference_pass),
                                   provenance={"registry_hash":REGISTRY_HASH,"algorithm_hash":ALGORITHM_HASH})
    return out


__all__=[
    "FullBenchError","FidelityError","BudgetError","ClaimLockedError","HardwareGateError",
    "BENCHMARK_REGISTRY","REGISTRY_HASH","ALGORITHM_SOURCE_HASHES","ALGORITHM_HASH","FidelityManifest",
    "fourier_bounds","canonical_control","fourier_forward_schedule","feasible_control","latin_hypercube",
    "CostEntry","ResourceLedger","ClaimGate","export_superiority_row",
    "fit_gp","expected_improvement","SequentialGPEI","finzgar_kappa","FinzgarBO",
    "JeongTuRBOState","QPURuntimeBudget","JeongTuRBO",
    "PeriodicGP","TorusSpectralMaternGP",
    "WorldlineSQAConfig","worldline_susceptibility","construct_worldline_schedule",
    "cuda_schedule_sqa_embedded_bqm",
    "calibrated_sqa_drive_schedule",
    "QueryCounter","FourierDiscreteActionSpace","MatchedBoxDiscreteActionSpace","PureMCTS","PolicyValueNet","QZero",
    "tune_dwave_sa","tune_dwave_tabu",
    "SYSTEM4_CALIBRATION_SHA256","SYSTEM6_CALIBRATION_SHA256","ADVANTAGE2_REJECT_SHA256","file_sha256","load_ab_calibration","assert_pegasus_sampler",
    "require_cuda_claim_runtime","default_fidelity_manifests","stable_hash"
]

# ---------------------------------------------------------------------------
# Small exact-QA reference environment used ONLY for fidelity reproduction.
# It is never used as a substitute for the production Pegasus simulator/QPU.
# ---------------------------------------------------------------------------

def _pauli_x_on_qubit(n: int, q: int) -> NDArray[np.complex128]:
    I=np.eye(2,dtype=COMPLEX); X=np.asarray([[0,1],[1,0]],dtype=COMPLEX)
    out=np.asarray([[1.0+0j]])
    for k in range(int(n)):
        out=np.kron(out, X if k==q else I)
    return out


def exact_ising_matrices(h: Mapping[int,float] | Sequence[float],
                         J: Mapping[tuple[int,int],float] | NDArray[np.float64]) -> tuple[NDArray[np.complex128],NDArray[np.complex128],NDArray[np.float64]]:
    hv,JM=_ising_arrays(h,J); n=hv.size
    if n<1 or n>8: raise ValueError("exact reference environment is limited to 1..8 spins")
    dim=1<<n
    energies=np.zeros(dim,dtype=REAL)
    for idx in range(dim):
        z=np.asarray([1.0 if ((idx>>(n-1-q))&1)==0 else -1.0 for q in range(n)],dtype=REAL)
        energies[idx]=float(hv@z + 0.5*z@JM@z)
    Hp=np.diag(energies.astype(COMPLEX))
    Hd=np.zeros((dim,dim),dtype=COMPLEX)
    for q in range(n): Hd -= _pauli_x_on_qubit(n,q)
    return Hd,Hp,energies


def exact_qa_ground_probability(h: Mapping[int,float] | Sequence[float],
                                J: Mapping[tuple[int,int],float] | NDArray[np.float64],
                                control: ArrayLike, *, order: int,
                                integration_steps: int = 400) -> float:
    """Exact small-system QA response for competitor reference-reproduction tests.

    Uses dense unitary midpoint propagation with H(u)=(1-s(u))Hd+s(u)Hp.
    This is deliberately isolated from production benchmark backends.
    """
    Hd,Hp,energies=exact_ising_matrices(h,J)
    c=np.asarray(control,dtype=REAL).reshape(-1); T=float(c[0])
    # schedule evaluated on a dense normalized-time grid; control can be non-linear.
    u=np.linspace(0.0,1.0,int(integration_steps)+1)
    s=u.copy()
    for k,theta in enumerate(c[1:],start=1): s += float(theta)*np.sin(k*np.pi*u)
    s=np.clip(s,0.0,1.0);s[0]=0.0;s[-1]=1.0
    if np.any(np.diff(s)<-1e-10): return 0.0
    n=int(round(math.log2(Hd.shape[0])))
    psi=np.ones(1<<n,dtype=COMPLEX)/math.sqrt(1<<n)  # ground state of -sum X
    dt=T/int(integration_steps)
    for k in range(int(integration_steps)):
        sm=0.5*(s[k]+s[k+1])
        H=(1.0-sm)*Hd+sm*Hp
        psi=linalg.expm(-1j*dt*H)@psi
    emin=float(np.min(energies)); idx=np.where(np.isclose(energies,emin,rtol=0,atol=1e-12))[0]
    return float(np.sum(np.abs(psi[idx])**2).real)


def reference_qa_instance() -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Frozen 4-spin Ising reference used to verify schedule optimizers."""
    h=np.asarray([0.31,-0.17,0.23,-0.29],dtype=REAL)
    J=np.asarray([[0,-0.74,0.18,0.0],[-0.74,0,-0.51,0.27],[0.18,-0.51,0,-0.66],[0,0.27,-0.66,0]],dtype=REAL)
    return h,J

# ---------------------------------------------------------------------------
# Cost-to-target and independent confirmation.
# ---------------------------------------------------------------------------


def cost_to_target(values: Sequence[float], costs: Sequence[float], target: float, *, maximize: bool=True) -> dict[str,Any]:
    v=np.asarray(values,dtype=REAL).reshape(-1); c=np.asarray(costs,dtype=REAL).reshape(-1)
    if v.size==0 or v.shape!=c.shape or np.any(c<0) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(c)):
        raise ValueError("values/costs must be aligned finite nonnegative-cost vectors")
    best=np.maximum.accumulate(v) if maximize else np.minimum.accumulate(v)
    hit=np.flatnonzero(best>=float(target)) if maximize else np.flatnonzero(best<=float(target))
    if hit.size:
        i=int(hit[0]); return {"reached":True,"queries":i+1,"cumulative_cost":float(np.sum(c[:i+1])),"best":float(best[i])}
    return {"reached":False,"queries":None,"cumulative_cost":None,"best":float(best[-1])}


def independent_confirmation_lcb(default_values: Sequence[float], candidate_values: Sequence[float], *,
                                 alpha: float=0.05, margin: float=0.0, bootstrap: int=5000,
                                 seed: int=20260816) -> dict[str,Any]:
    d=np.asarray(default_values,dtype=REAL).reshape(-1); c=np.asarray(candidate_values,dtype=REAL).reshape(-1)
    if d.size<2 or c.size<2: raise ValueError("Independent confirmation requires at least two replicates per arm")
    rng=np.random.default_rng(int(seed)); n=max(d.size,c.size); means=np.empty(int(bootstrap),dtype=REAL)
    for b in range(int(bootstrap)):
        ds=d[rng.integers(0,d.size,n)]; cs=c[rng.integers(0,c.size,n)]
        means[b]=float(np.mean(cs)-np.mean(ds))
    lcb=float(np.quantile(means,float(alpha))); ucb=float(np.quantile(means,1.0-float(alpha)))
    delta=float(np.mean(c)-np.mean(d))
    return {"delta_mean":delta,"lcb":lcb,"ucb":ucb,"alpha":float(alpha),"margin":float(margin),
            "bootstrap":int(bootstrap),"pass":bool(lcb>float(margin)),"n_default":int(d.size),"n_candidate":int(c.size)}


def paired_cssf_vs_competitor_confirmation_lcb(cssf_values: Sequence[float], competitor_values: Sequence[float], *,
                                                alpha: float=0.05, margin: float=0.0, bootstrap: int=5000,
                                                seed: int=20260816) -> dict[str,Any]:
    """Direct independent CSSF-vs-competitor confirmation on paired shards.

    Pairing is mandatory: replicate i must correspond to the same frozen
    confirmation shard/seed/context for CSSF and the competitor.
    """
    c=np.asarray(cssf_values,dtype=REAL).reshape(-1); b=np.asarray(competitor_values,dtype=REAL).reshape(-1)
    if c.size<2 or b.size<2 or c.size!=b.size:
        raise ValueError("Direct confirmation requires aligned paired CSSF/competitor replicates")
    diff=c-b
    if not np.all(np.isfinite(diff)): raise ValueError("confirmation values must be finite")
    rng=np.random.default_rng(int(seed)); means=np.empty(int(bootstrap),dtype=REAL)
    for k in range(int(bootstrap)):
        idx=rng.integers(0,diff.size,diff.size); means[k]=float(np.mean(diff[idx]))
    lcb=float(np.quantile(means,float(alpha))); ucb=float(np.quantile(means,1.0-float(alpha)))
    delta=float(np.mean(diff))
    return {"delta_cssf_minus_competitor":delta,"lcb":lcb,"ucb":ucb,"alpha":float(alpha),
            "margin":float(margin),"bootstrap":int(bootstrap),"pass":bool(lcb>float(margin)),
            "n_pairs":int(diff.size),"paired":True}


def compare_cost_to_target_superiority(cssf_result: Mapping[str,Any], competitor_result: Mapping[str,Any], *,
                                       strict: bool=True) -> dict[str,Any]:
    """Fail-closed comparison of two already-computed cost-to-target records."""
    if not bool(cssf_result.get("reached")) or not bool(competitor_result.get("reached")):
        return {"pass":False,"reason":"both_methods_must_reach_same_target",
                "cssf":dict(cssf_result),"competitor":dict(competitor_result)}
    cc=float(cssf_result["cumulative_cost"]); bc=float(competitor_result["cumulative_cost"])
    passed=cc<bc if strict else cc<=bc
    return {"pass":bool(passed),"cssf_cost":cc,"competitor_cost":bc,
            "advantage":float(bc-cc),"ratio_cssf_over_competitor":float(cc/bc) if bc>0 else (0.0 if cc==0 else math.inf),
            "strict":bool(strict)}
