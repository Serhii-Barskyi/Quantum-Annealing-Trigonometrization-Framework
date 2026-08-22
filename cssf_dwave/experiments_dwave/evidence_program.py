"""D-Wave evidence program for the full original CSSF framework.

This module is an orchestration layer.  It does not reimplement CSNN-T, GCV,
the CSSF spectral hierarchy, the BESS model, QUBO/Ising construction, or the
Pegasus backend.  The two D-Wave evidence notebooks call these functions so
that Simulator and QPU experiments share one scientific protocol and differ
only in the annealing-response backend.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np

from bess.case300 import load_case300_mode_a
from core.csnn_t import fit_csnn_t
from core.dataset import BESSDataset
from core.validation import verify_frozen_core
from experiments_dwave.benchmark_protocol import MatchedControlProtocol
from experiments_dwave.bess_evidence import (
    Case300BESSEvidenceAssets,
    build_case300_bess_assets,
    logical_feasibility,
    to_dimod_bqm,
)
from experiments_dwave.operator_phase import load_calibration
from experiments_dwave.pegasus_control_backend import (
    BoundPegasusResponseEvaluator,
    FixedPegasusControlBackend,
)
from benchmarks import reference_competitors as rc


PROGRAM_REVISION = "CSSF-QA-DWAVE-EVIDENCE-v51-20260819"
REQUIRED_ORIGINAL_PACKAGES = (
    "core", "spectral", "opf", "bess", "qubo", "qaoa", "maqaoa", "qa",
    "dwave_backend", "pipeline", "baselines", "config",
)
CALIBRATION_FILES = {
    "Advantage_system4": "09-1263A-C_Advantage_system4_annealing_schedule.xlsx",
    "Advantage_system6": "09-1273A-F_Advantage_system6_annealing_schedule.xlsx",
}
TARGET_NAMES = (
    "mean_energy", "energy_variance", "energy_quantile_05", "cvar_05",
    "feasibility_probability", "elite_probability", "success_probability",
)


class EvidenceProgramError(RuntimeError):
    """Raised when an evidence-program invariant is violated."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


def framework_identity_report(project_root: str | Path) -> dict[str, Any]:
    """Verify the original CSSF packages and immutable CSNN-T/GCV core."""
    root = Path(project_root)
    missing = [name for name in REQUIRED_ORIGINAL_PACKAGES if not (root / name).is_dir()]
    if missing:
        raise EvidenceProgramError(f"Original CSSF packages are missing: {missing}")
    frozen = verify_frozen_core(root)
    if not frozen.valid:
        raise EvidenceProgramError("Frozen CSNN-T/GCV core integrity verification failed")
    data = load_case300_mode_a(root / "data" / "case300_full_modeA_Barskyi_Serhii.json")
    return {
        "program_revision": PROGRAM_REVISION,
        "original_packages": list(REQUIRED_ORIGINAL_PACKAGES),
        "frozen_core_valid": True,
        "frozen_core": {item.relative_path: item.actual_sha256 for item in frozen.files},
        "case300_sha256": data.source_sha256,
        "case300_fingerprint": data.fingerprint(),
        "case300_shape": [data.n_scenarios, data.n],
        "case300_modeA_features": [data.n_scenarios, data.M_complex],
    }


def _pearson_flat(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if np.std(a) <= 0 or np.std(p) <= 0:
        return float("nan")
    return float(np.corrcoef(a, p)[0, 1])


def run_level1_csnn_t_evidence(
    project_root: str | Path,
    *,
    bootstrap_samples: int = 4000,
    seed: int = 20260817,
) -> dict[str, Any]:
    """Run the full original case300 CSNN-T Level-I trigonometric evidence.

    The CSSF arm uses the immutable complex Mode-A feature matrix and the
    original ``core.csnn_t.fit_csnn_t`` / GCV implementation.  The comparator
    is a raw-angle RidgeCV model trained on the identical train/test split.
    """
    root = Path(project_root)
    identity = framework_identity_report(root)
    data = load_case300_mode_a(root / "data" / "case300_full_modeA_Barskyi_Serhii.json")
    ds = BESSDataset(
        case="case300_level1_full_csnnt",
        X_train=data.features[data.train_slice],
        y_train=data.targets[data.train_slice],
        X_test=data.features[data.test_slice],
        y_test=data.targets[data.test_slice],
    )
    cssf = fit_csnn_t(ds, n_lambdas=100, lam_range=(-12, 4))
    cssf_pred = cssf.predict(ds.X_test)

    try:
        from sklearn.linear_model import RidgeCV
    except Exception as exc:  # pragma: no cover - project requirement in target environment
        raise EvidenceProgramError("scikit-learn is required for the raw-angle comparator") from exc
    alphas = np.logspace(-12, 4, 100)
    raw = RidgeCV(alphas=alphas, fit_intercept=True)
    raw.fit(data.theta_rad[data.train_slice], data.targets[data.train_slice])
    raw_pred = np.asarray(raw.predict(data.theta_rad[data.test_slice]), dtype=float)

    actual = data.targets[data.test_slice]
    cssf_sq = np.mean((actual - cssf_pred) ** 2, axis=1)
    raw_sq = np.mean((actual - raw_pred) ** 2, axis=1)
    delta = raw_sq - cssf_sq
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(bootstrap_samples), dtype=float)
    for i in range(int(bootstrap_samples)):
        idx = rng.integers(0, delta.size, delta.size)
        boot[i] = float(np.mean(delta[idx]))
    ci = np.quantile(boot, [0.025, 0.975])

    cssf_mse = float(np.mean((actual - cssf_pred) ** 2))
    raw_mse = float(np.mean((actual - raw_pred) ** 2))
    return {
        "experiment": "Level-I full original CSNN-T trigonometric response identification",
        "framework_identity": identity,
        "train_rows": int(data.n_train),
        "test_rows": int(data.n_test),
        "raw_dimensions": int(data.theta_rad.shape[1]),
        "complex_modeA_features": int(data.M_complex),
        "cssf_lambda_gcv": float(cssf.lam_opt),
        "cssf_mse": cssf_mse,
        "raw_ridge_mse": raw_mse,
        "relative_mse_reduction": float((raw_mse - cssf_mse) / raw_mse),
        "cssf_pearson": _pearson_flat(actual, cssf_pred),
        "raw_ridge_pearson": _pearson_flat(actual, raw_pred),
        "heldout_scenarios_cssf_better": int(np.sum(cssf_sq < raw_sq)),
        "heldout_scenarios_total": int(delta.size),
        "paired_mean_mse_improvement": float(np.mean(delta)),
        "paired_bootstrap_95_ci": [float(ci[0]), float(ci[1])],
        "raw_ridge_alpha": float(raw.alpha_),
        "claim_boundary": "This experiment isolates response-identification value of the original CSSF trigonometric representation; it is not a QPU-advantage claim.",
    }


def calibration_report(project_root: str | Path, family: str) -> dict[str, Any]:
    root = Path(project_root)
    if family not in CALIBRATION_FILES:
        raise EvidenceProgramError("Only Advantage_system4 and Advantage_system6 calibration families are admissible")
    curve = load_calibration(root / "calibration" / CALIBRATION_FILES[family], family)
    return {
        "family": curve.family,
        "sha256": curve.source_sha256,
        "rows": int(curve.s.size),
        "A_GHz_range": [float(np.min(curve.A_GHz)), float(np.max(curve.A_GHz))],
        "B_GHz_range": [float(np.min(curve.B_GHz)), float(np.max(curve.B_GHz))],
    }


def load_runtime_config(project_root: str | Path, mode: str, *, solver_id: str | None = None, live_qpu: bool = False):
    """Load the original validated CSSF configuration for one evidence backend."""
    root = Path(project_root)
    from config.loader import load_config
    normalized = str(mode).strip().lower()
    if normalized == "simulator":
        return load_config([root / "config" / "base.yaml", root / "config" / "case300.yaml", root / "config" / "emulator_gpu.yaml"])
    if normalized != "qpu":
        raise EvidenceProgramError("mode must be simulator or qpu")
    if not solver_id or not str(solver_id).startswith(("Advantage_system4.", "Advantage_system6.")):
        raise EvidenceProgramError("An explicit Advantage_system4.* or Advantage_system6.* solver_id is required")
    cfg = load_config([root / "config" / "base.yaml", root / "config" / "case300.yaml", root / "config" / "pegasus_qpu.yaml"])
    cfg.qpu.enabled = True
    cfg.qpu.solver_id = str(solver_id)
    cfg.qpu.dry_run = not bool(live_qpu)
    return cfg


def build_response_evaluator(
    project_root: str | Path,
    *,
    mode: str,
    calibration_family: str,
    solver_id: str | None = None,
    live_qpu: bool = False,
    solve_highs: bool = True,
    expected_topology_fingerprint: str | None = None,
    embedding_manifest_path: str | Path | None = None,
) -> tuple[BoundPegasusResponseEvaluator, Case300BESSEvidenceAssets, Any, dict[str, Any]]:
    """Build one fixed-embedding Pegasus response evaluator from original CSSF assets."""
    root = Path(project_root)
    normalized = str(mode).strip().lower()
    cfg = load_runtime_config(root, normalized, solver_id=solver_id, live_qpu=live_qpu)
    assets = build_case300_bess_assets(root, solve_highs=bool(solve_highs))
    if assets.elite_energy_threshold is None:
        raise EvidenceProgramError("A certified HiGHS reference is required to define the frozen elite threshold")
    bqm = to_dimod_bqm(assets.problem)
    curve = load_calibration(root / "calibration" / CALIBRATION_FILES[calibration_family], calibration_family)
    from dwave_backend.sampler import build_ocean_sampler
    if normalized == "simulator":
        bundle = build_ocean_sampler("local_sqa_gpu", emulator_config=cfg.emulator, seed=cfg.random.global_seed)
    elif normalized == "qpu":
        bundle = build_ocean_sampler("pegasus_qpu", qpu_config=cfg.qpu)
    else:
        raise EvidenceProgramError("mode must be simulator or qpu")
    topology_fingerprint = bundle.topology.fingerprint()
    if expected_topology_fingerprint is not None and topology_fingerprint != str(expected_topology_fingerprint):
        bundle.close()
        raise EvidenceProgramError(
            f"Pegasus topology fingerprint mismatch: expected {expected_topology_fingerprint}, observed {topology_fingerprint}"
        )
    frozen_embedding = None
    frozen_chain_strength = None
    embedding_path = None if embedding_manifest_path is None else Path(embedding_manifest_path)
    if embedding_path is not None and embedding_path.exists():
        saved = json.loads(embedding_path.read_text(encoding="utf-8"))
        if saved.get("topology_fingerprint") != topology_fingerprint:
            bundle.close()
            raise EvidenceProgramError("Saved embedding belongs to a different Pegasus topology fingerprint")
        if saved.get("calibration_family") != calibration_family:
            bundle.close()
            raise EvidenceProgramError("Saved embedding calibration family differs from the requested experiment")
        frozen_embedding = saved.get("embedding")
        frozen_chain_strength = saved.get("chain_strength")
    backend = FixedPegasusControlBackend.build(
        mode=normalized,
        bundle=bundle,
        logical_bqm=bqm,
        calibration=curve,
        seed=cfg.random.global_seed,
        frozen_embedding=frozen_embedding,
        chain_strength=frozen_chain_strength,
    )
    if embedding_path is not None and not embedding_path.exists():
        atomic_json(embedding_path, backend.embedding_manifest())
    evaluator = BoundPegasusResponseEvaluator(
        backend=backend,
        order=MatchedControlProtocol().order,
        elite_threshold=float(assets.elite_energy_threshold),
        feasibility=logical_feasibility(assets.problem),
        success_energy=assets.exact_success_energy,
        label=f"CSSF(QA) {PROGRAM_REVISION} {normalized}",
    )
    manifest = {
        "mode": normalized,
        "solver_id": getattr(bundle, "solver_id", None),
        "topology_fingerprint": topology_fingerprint,
        "embedding_fingerprint": backend.embedding_fingerprint,
        "chain_strength": float(backend.chain_strength),
        "calibration": calibration_report(root, calibration_family),
        "bess": assets.manifest(),
        "protocol": asdict(MatchedControlProtocol()),
    }
    return evaluator, assets, bundle, manifest


def worldline_schedule_for_assets(
    assets: Case300BESSEvidenceAssets,
    project_root: str | Path,
    calibration_family: str,
    *,
    total_time_us: float = 20.0,
    grid_points: int = 21,
    beta: float = 2.0,
    replicas: int = 32,
    burn_in_sweeps: int = 200,
    measurement_sweeps: int = 800,
    thin: int = 4,
    seed: int = 20260817,
) -> dict[str, Any]:
    """Construct the complete worldline-susceptibility schedule on the BESS Ising model."""
    curve = load_calibration(Path(project_root) / "calibration" / CALIBRATION_FILES[calibration_family], calibration_family)
    ham = assets.hamiltonian
    h = np.asarray(ham.linear_z, dtype=float)
    J = np.asarray(ham.quadratic_zz, dtype=float)
    J = J + J.T
    s_grid = np.linspace(0.0, 1.0, int(grid_points))
    cfg = rc.WorldlineSQAConfig(
        beta=float(beta), replicas=int(replicas), burn_in_sweeps=int(burn_in_sweeps),
        measurement_sweeps=int(measurement_sweeps), thin=int(thin), seed=int(seed),
    )
    return rc.construct_worldline_schedule(
        h, J, s_grid=s_grid,
        A_of_s=lambda s: np.interp(s, curve.s, curve.A_GHz),
        B_of_s=lambda s: np.interp(s, curve.s, curve.B_GHz),
        T_us=float(total_time_us), chi0=1e-6, config=cfg,
    )


def environment_report() -> dict[str, Any]:
    """Return a non-secret reproducibility snapshot."""
    report: dict[str, Any] = {
        "program_revision": PROGRAM_REVISION,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        from importlib import metadata
        packages = ("numpy", "pandas", "scipy", "scikit-learn", "qiskit", "qiskit-aer-gpu", "dwave-ocean-sdk", "pandapower", "highspy", "torch")
        report["packages"] = {}
        for name in packages:
            try:
                report["packages"][name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                report["packages"][name] = None
    except Exception:
        report["packages"] = {}
    return report


__all__ = [
    "PROGRAM_REVISION", "REQUIRED_ORIGINAL_PACKAGES", "CALIBRATION_FILES", "TARGET_NAMES",
    "EvidenceProgramError", "atomic_json", "framework_identity_report", "run_level1_csnn_t_evidence",
    "calibration_report", "load_runtime_config", "build_response_evaluator",
    "worldline_schedule_for_assets", "environment_report",
]
