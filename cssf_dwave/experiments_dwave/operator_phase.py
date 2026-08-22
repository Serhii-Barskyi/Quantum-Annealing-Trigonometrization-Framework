"""Calibration-resolved operator-action coordinates for CSSF(QA).

This module maps an executed physical anneal schedule ``s(t)`` and an approved
Pegasus System4/System6 ``A(s), B(s)`` calibration to the dimensionless action
coordinates used by the original CSSF spectral machinery.  It does not fit a
surrogate and it does not alter the frozen CSNN-T/GCV core.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any
import hashlib
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

SYSTEM4_CALIBRATION_SHA256 = "03350bb86bab2f752697e1a8c37f3e4c2100c6596d0f6c6bf8f6d2e3e97de4f1"
SYSTEM6_CALIBRATION_SHA256 = "d266ee71c8a0611cc392781da4df65e20969aed658b5df60453ac099202fdc06"
APPROVED_FAMILIES = {
    "Advantage_system4": ("09-1263A-C_Advantage_system4_annealing_schedule.xlsx", SYSTEM4_CALIBRATION_SHA256),
    "Advantage_system6": ("09-1273A-F_Advantage_system6_annealing_schedule.xlsx", SYSTEM6_CALIBRATION_SHA256),
}
TWO_PI_GHZ_US = 2.0 * math.pi * 1_000.0


class OperatorPhaseError(RuntimeError):
    """Raised when physical schedule or calibration provenance is invalid."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ABReferenceCurve:
    family: str
    s: NDArray[np.float64]
    A_GHz: NDArray[np.float64]
    B_GHz: NDArray[np.float64]
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.family not in APPROVED_FAMILIES:
            raise OperatorPhaseError(f"Unsupported calibration family: {self.family!r}")
        s = np.asarray(self.s, dtype=np.float64).reshape(-1)
        A = np.asarray(self.A_GHz, dtype=np.float64).reshape(-1)
        B = np.asarray(self.B_GHz, dtype=np.float64).reshape(-1)
        if s.size < 2 or not (s.size == A.size == B.size):
            raise OperatorPhaseError("A/B calibration arrays must be aligned and non-empty")
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            raise OperatorPhaseError("A/B calibration arrays must be finite")
        if abs(float(s[0])) > 1e-12 or abs(float(s[-1]) - 1.0) > 1e-6 or np.any(np.diff(s) <= 0):
            raise OperatorPhaseError("Calibration s grid must be strictly increasing over [0,1]")
        expected = APPROVED_FAMILIES[self.family][1]
        if self.source_sha256 != expected:
            raise OperatorPhaseError(
                f"Calibration hash mismatch for {self.family}: {self.source_sha256} != {expected}"
            )
        for name, arr in (("s", s), ("A_GHz", A), ("B_GHz", B)):
            copy = np.ascontiguousarray(arr, dtype=np.float64)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)

    def interpolate(self, s_values: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        values = np.asarray(s_values, dtype=np.float64)
        if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
            raise OperatorPhaseError("Schedule s values must lie in [0,1]")
        values = np.clip(values, 0.0, 1.0)
        return np.interp(values, self.s, self.A_GHz), np.interp(values, self.s, self.B_GHz)


def load_calibration(path: str | Path, family: str) -> ABReferenceCurve:
    """Load one approved D-Wave standard-annealing A/B spreadsheet."""
    if family not in APPROVED_FAMILIES:
        raise OperatorPhaseError("Only Advantage_system4 and Advantage_system6 are admissible")
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    expected_name, expected_hash = APPROVED_FAMILIES[family]
    if source.name != expected_name or digest != expected_hash:
        raise OperatorPhaseError(
            f"Frozen calibration identity mismatch for {family}: name={source.name!r}, sha256={digest}"
        )
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - target environment dependency
        raise OperatorPhaseError("pandas is required to load D-Wave calibration spreadsheets") from exc
    frame = pd.read_excel(source, sheet_name="Standard-Annealing Schedule", header=None)
    if frame.shape[0] < 3 or frame.shape[1] < 3:
        raise OperatorPhaseError("Calibration spreadsheet has an unexpected shape")
    data = frame.iloc[1:, :3].astype(float).to_numpy()
    return ABReferenceCurve(
        family=family,
        s=data[:, 0],
        A_GHz=data[:, 1],
        B_GHz=data[:, 2],
        source_path=str(source),
        source_sha256=digest,
    )


def validate_executed_schedule(t_us: ArrayLike, s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t = np.asarray(t_us, dtype=np.float64).reshape(-1)
    q = np.asarray(s, dtype=np.float64).reshape(-1)
    if t.size < 2 or t.size != q.size:
        raise OperatorPhaseError("Executed schedule requires aligned t_us and s arrays")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(q))):
        raise OperatorPhaseError("Executed schedule must be finite")
    if abs(float(t[0])) > 1e-12 or np.any(np.diff(t) <= 0.0):
        raise OperatorPhaseError("Schedule time must start at zero and increase strictly")
    if np.any(q < -1e-12) or np.any(q > 1.0 + 1e-12):
        raise OperatorPhaseError("Schedule s must remain in [0,1]")
    if abs(float(q[0])) > 1e-8 or abs(float(q[-1]) - 1.0) > 1e-8:
        raise OperatorPhaseError("Forward schedule must start at s=0 and end at s=1")
    return np.ascontiguousarray(t), np.ascontiguousarray(np.clip(q, 0.0, 1.0))


def operator_action_coordinates(
    t_us: ArrayLike,
    s: ArrayLike,
    calibration: ABReferenceCurve,
    *,
    n_segments: int = 8,
) -> NDArray[np.float64]:
    """Return ``(beta_1,gamma_1,...,beta_p,gamma_p)`` in radians.

    The integral is taken in physical time.  Since the calibration files report
    ``A/h`` and ``B/h`` in GHz, ``2*pi*1000`` converts ``GHz * microsecond`` to
    dimensionless radians.
    """
    if isinstance(n_segments, bool) or int(n_segments) < 1:
        raise OperatorPhaseError("n_segments must be a positive integer")
    p = int(n_segments)
    t, q = validate_executed_schedule(t_us, s)
    # Dense resampling makes the segment integrals stable for PWL schedules.
    dense_n = max(2049, 256 * p + 1)
    td = np.linspace(float(t[0]), float(t[-1]), dense_n, dtype=np.float64)
    sd = np.interp(td, t, q)
    A, B = calibration.interpolate(sd)
    edges = np.linspace(float(t[0]), float(t[-1]), p + 1, dtype=np.float64)
    coords: list[float] = []
    for j in range(p):
        lo, hi = edges[j], edges[j + 1]
        mask = (td >= lo) & (td <= hi)
        tt = td[mask]
        aa = A[mask]
        bb = B[mask]
        if tt.size < 2:
            raise OperatorPhaseError("Insufficient quadrature points in action segment")
        beta = TWO_PI_GHZ_US * float(np.trapezoid(aa, tt))
        gamma = TWO_PI_GHZ_US * float(np.trapezoid(bb, tt))
        coords.extend((beta, gamma))
    out = np.asarray(coords, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise OperatorPhaseError("Operator-action coordinates are non-finite")
    return out


def operator_action_batch(
    schedules: list[tuple[ArrayLike, ArrayLike]],
    calibration: ABReferenceCurve,
    *,
    n_segments: int = 8,
) -> NDArray[np.float64]:
    if not schedules:
        raise OperatorPhaseError("At least one schedule is required")
    return np.vstack([
        operator_action_coordinates(t, s, calibration, n_segments=n_segments)
        for t, s in schedules
    ])


def calibration_provenance(curve: ABReferenceCurve) -> Mapping[str, Any]:
    return {
        "family": curve.family,
        "source_path": curve.source_path,
        "source_sha256": curve.source_sha256,
        "rows": int(curve.s.size),
        "operator_phase_unit": "radian",
        "phase_factor_GHz_us": TWO_PI_GHZ_US,
    }
