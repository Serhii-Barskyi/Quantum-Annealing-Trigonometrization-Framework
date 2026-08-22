"""Stable integration wrapper for BESS D0-D3 construction (v54).

v54 fixes an integration leak discovered in the Colab D0-D3 cell: the legacy
``integrated_bess_v38`` module built the trigonometric domain arm through the
frozen ``core.csnn_t.fit_csnn_t`` path, whose final coefficient calculation
forms normal equations.  That bypassed the v53 SVD-Tikhonov corrective layer
and could emit a SciPy ``LinAlgWarning`` for the case300 design.

The physical/scientific construction is unchanged.  Only the algebraically
identical numerical realization of the Tikhonov minimizer is routed through
``core.csnn_t_adapter_v53.fit_csnn_t_surrogate_v53``.  All BESS/QUBO classes,
candidate selection, Ising conversion, and factorial definitions remain the
v38 objects.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import experiments_dwave.integrated_bess_v38 as _v38
from core.csnn_t_adapter_v53 import fit_csnn_t_surrogate_v53
from core.dataset import BESSDataset
from core.types import SurrogateLevel

# Re-export the canonical v38 data structures so downstream type identity and
# serialization remain unchanged.
IntegratedBESSError = _v38.IntegratedBESSError
DomainRepresentation = _v38.DomainRepresentation
BESSArmProblem = _v38.BESSArmProblem
FactorialArmSpec = _v38.FactorialArmSpec
factorial_specs = _v38.factorial_specs
raw_control_coordinates = _v38.raw_control_coordinates
run_cssf_same_machinery = _v38.run_cssf_same_machinery


def build_domain_representation(
    project_root: str | Path,
    representation: str,
) -> tuple[Any, DomainRepresentation]:
    """Build raw/trigonometric domain arms with a stable trig coefficient solve."""
    root = Path(project_root)
    data = _v38.load_case300_mode_a(
        root / "data" / "case300_full_modeA_Barskyi_Serhii.json"
    )
    rep = str(representation).strip().lower()

    if rep == "trig":
        ds = BESSDataset(
            case=data.case,
            X_train=data.features[data.train_slice],
            y_train=data.targets[data.train_slice],
            X_test=data.features[data.test_slice],
            y_test=data.targets[data.test_slice],
            metadata={"representation": "modeA_trigonometric"},
        )
        wrapped = fit_csnn_t_surrogate_v53(
            ds,
            level=SurrogateLevel.OPF,
            n_lambdas=100,
            lam_range=(-12.0, 4.0),
            project_root=root,
        )
        model = wrapped.frozen_model
        train = wrapped.predict(ds.X_train)
        test = wrapped.predict(ds.X_test)
        fp = hashlib.sha256(
            model.H.tobytes(order="C") + repr(model.lam_opt).encode()
        ).hexdigest()
        conditioning = dict(wrapped.metadata.get("conditioning", {}))
        diagnostics = {
            "lambda_gcv": float(model.lam_opt),
            "features": int(model.M),
            "call_path": "core.csnn_t_adapter_v53.fit_csnn_t_surrogate_v53",
            "numerical_realization": "svd_tikhonov_v53",
            "normal_equations_formed": False,
            "conditioning": conditioning,
        }
        return data, DomainRepresentation(
            "trig", train, test, fp, diagnostics, model
        )

    if rep == "raw":
        try:
            from sklearn.linear_model import RidgeCV
        except Exception as exc:  # pragma: no cover - environment gate
            raise IntegratedBESSError(
                "scikit-learn is required for the raw-domain factorial arm"
            ) from exc

        grid = np.logspace(-12, 4, 100)
        model = RidgeCV(alphas=grid, fit_intercept=True)
        model.fit(
            data.theta_rad[data.train_slice],
            data.targets[data.train_slice],
        )
        train = np.asarray(
            model.predict(data.theta_rad[data.train_slice]), dtype=float
        )
        test = np.asarray(
            model.predict(data.theta_rad[data.test_slice]), dtype=float
        )
        payload = {
            "coef": np.asarray(model.coef_).tolist(),
            "intercept": np.asarray(model.intercept_).tolist(),
            "alpha": float(model.alpha_),
        }
        fp = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        return data, DomainRepresentation(
            "raw",
            train,
            test,
            fp,
            {
                "ridge_alpha": float(model.alpha_),
                "raw_dimensions": int(data.theta_rad.shape[1]),
                "role": "factorial raw-domain ablation",
            },
            None,
        )

    raise IntegratedBESSError("representation must be raw or trig")


def build_bess_arm_problem(
    project_root: str | Path,
    representation: str,
) -> BESSArmProblem:
    """Build the canonical BESS/QUBO arm using the v54 domain representation."""
    root = Path(project_root)
    data, domain = build_domain_representation(root, representation)
    cfg = _v38.load_config(
        [
            root / "config" / "base.yaml",
            root / "config" / "case300.yaml",
            root / "config" / "emulator_gpu.yaml",
        ]
    )
    selection_cfg = _v38.CandidateSelectionConfig.from_qubo_config(cfg.qubo)
    selection = _v38.candidate_selection_from_lsf(
        data,
        selection_cfg,
        domain.train_predictions,
        representation=domain.name,
        model_fingerprint=domain.model_fingerprint,
    )
    fleet = _v38.build_case300_fleet(
        selection,
        bess_units=cfg.qubo.bess_units,
        unit=_v38.BESSUnitSpec(power_mw=25.0, energy_mwh=100.0),
        metadata={"strategy": f"case300_{domain.name}_domain_train_only"},
    )
    linear = {
        bus: -float(selection.scores[i])
        for i, bus in enumerate(selection.candidate_buses)
    }
    problem = _v38.build_bess_placement_qubo(
        fleet,
        linear_by_bus=linear,
        metadata={
            "candidate_selection_fingerprint": selection.fingerprint(),
            "domain_representation": domain.name,
            "domain_model_fingerprint": domain.model_fingerprint,
        },
    )
    hamiltonian = _v38.qubo_to_ising(problem.model)
    audit = _v38.require_qubo_ising_equivalence(
        problem.model,
        hamiltonian,
        exact_limit=18,
        random_samples=8192,
        seed=cfg.random.global_seed,
        tolerance=cfg.qubo.verify_qubo_ising_tolerance,
    )
    return BESSArmProblem(
        domain.name,
        data,
        domain,
        selection,
        fleet,
        problem,
        hamiltonian,
        audit,
    )


__all__ = [
    "IntegratedBESSError",
    "DomainRepresentation",
    "BESSArmProblem",
    "FactorialArmSpec",
    "build_domain_representation",
    "build_bess_arm_problem",
    "factorial_specs",
    "raw_control_coordinates",
    "run_cssf_same_machinery",
]
