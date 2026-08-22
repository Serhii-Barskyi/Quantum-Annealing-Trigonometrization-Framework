"""Admissible schedule designs for CSSF(QA) annealing-control experiments.

This module is CPU-only.  It constructs experimental controls exclusively from
forward annealing schedules that pass the physical schedule validator.  It is
used by every matched SQA campaign before any annealer evaluation is requested.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
from numpy.typing import ArrayLike, NDArray

from benchmarks import reference_competitors as rc


class ControlDesignError(RuntimeError):
    """Raised when a requested admissible design cannot be constructed."""


def is_admissible_forward_control(control: ArrayLike, bounds: ArrayLike, *, order: int) -> bool:
    x = np.asarray(control, dtype=np.float64).reshape(-1)
    b = np.asarray(bounds, dtype=np.float64)
    if b.shape != (x.size, 2):
        return False
    if np.any(x < b[:, 0]) or np.any(x > b[:, 1]):
        return False
    try:
        rc.fourier_forward_schedule(x, order=int(order), grid_points=129, reject_nonmonotone=True)
    except Exception:
        return False
    return True


def assert_admissible_forward_design(design: ArrayLike, bounds: ArrayLike, *, order: int) -> NDArray[np.float64]:
    X = np.asarray(design, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.float64)
    if X.ndim != 2 or b.ndim != 2 or b.shape[1] != 2 or X.shape[1] != b.shape[0]:
        raise ControlDesignError("control design and bounds have incompatible dimensions")
    invalid = [i for i, row in enumerate(X) if not is_admissible_forward_control(row, b, order=order)]
    if invalid:
        head = invalid[:12]
        raise ControlDesignError(
            f"design contains {len(invalid)} non-admissible forward schedules; first indices={head}"
        )
    if not np.isfinite(X).all():
        raise ControlDesignError("control design contains non-finite values")
    return np.ascontiguousarray(X)


def admissible_latin_hypercube(
    bounds: ArrayLike,
    n: int,
    *,
    order: int,
    seed: int,
    oversample_factor: int = 4,
    max_rounds: int = 256,
) -> NDArray[np.float64]:
    """Return exactly ``n`` distinct LHS-derived physically admissible controls.

    Candidate batches are deterministic for a fixed seed.  Rejection occurs
    before any SQA/QPU call.  The returned matrix is checked again fail-closed.
    """
    b = np.asarray(bounds, dtype=np.float64)
    n = int(n)
    if n < 1 or b.ndim != 2 or b.shape[1] != 2:
        raise ControlDesignError("invalid admissible LHS request")
    rows: list[np.ndarray] = []
    seen: set[bytes] = set()
    for round_idx in range(int(max_rounds)):
        remaining = n - len(rows)
        if remaining <= 0:
            break
        batch_size = max(128, int(oversample_factor) * remaining)
        batch = rc.latin_hypercube(b, batch_size, seed=int(seed) + 104729 * round_idx)
        for candidate in batch:
            c = np.ascontiguousarray(candidate, dtype=np.float64)
            if not is_admissible_forward_control(c, b, order=int(order)):
                continue
            key = c.tobytes(order="C")
            if key in seen:
                continue
            seen.add(key)
            rows.append(c.copy())
            if len(rows) == n:
                break
    if len(rows) != n:
        raise ControlDesignError(
            f"could construct only {len(rows)} admissible controls out of requested {n} after {max_rounds} rounds"
        )
    return assert_admissible_forward_design(np.vstack(rows), b, order=int(order))


def admissible_fraction(design: ArrayLike, bounds: ArrayLike, *, order: int) -> float:
    X = np.asarray(design, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ControlDesignError("design must be a non-empty matrix")
    ok = sum(is_admissible_forward_control(row, bounds, order=order) for row in X)
    return float(ok / X.shape[0])


__all__ = [
    "ControlDesignError",
    "is_admissible_forward_control",
    "assert_admissible_forward_design",
    "admissible_latin_hypercube",
    "admissible_fraction",
]
