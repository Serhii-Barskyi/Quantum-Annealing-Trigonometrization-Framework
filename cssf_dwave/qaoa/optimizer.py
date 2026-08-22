# -*- coding: utf-8 -*-
# Author: Serhii Barskyi | https://www.linkedin.com/in/serhii-barskyi/
# Data Science Course: https://preply.com/en/tutor/7756455
# Framework: Complex Spectral Surrogate Framework (CSSF)
#
# Licensed under the Apache License, Version 2.0.
# You may not use this file except in compliance with the License.
# Full license text: https://www.apache.org/licenses/LICENSE-2.0
#
# Attribution required: if you use this code, please cite:
# Serhii Barskyi, Complex Spectral Surrogate Framework (CSSF),
# https://www.linkedin.com/in/serhii-barskyi/

"""Qiskit Algorithms optimization orchestration for exact GPU QAOA.

This module never calls SciPy directly and never implements a classical
optimizer. Every minimization run is delegated to an optimizer from
``qiskit_algorithms.optimizers``. Every objective call is delegated to
:class:`qaoa.objective.QAOAGPUObjective`, whose quantum evaluations use only
``qiskit_aer.AerSimulator(method="statevector", device="GPU")``.

The production wrapper adds deterministic multi-start initialization, strict
parameter bounds, Qiskit random-seed control, immutable run records, best-point
tracking, and reproducibility manifests. QAOA parameters are kept as exact
``float64`` values and are never rounded or silently wrapped.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qaoa.aer_gpu import COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
from qaoa.circuit import QAOAParameterValues
from qaoa.objective import QAOAGPUObjective


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
OptimizerName = Literal["COBYLA", "SPSA", "SLSQP", "L_BFGS_B"]
InitialPointStrategy = Literal["midpoint_then_uniform", "uniform"]

SUPPORTED_OPTIMIZERS: Final[tuple[str, ...]] = (
    "COBYLA",
    "SPSA",
    "SLSQP",
    "L_BFGS_B",
)
DEFAULT_OPTIMIZER: Final[str] = "COBYLA"
DEFAULT_MAXITER: Final[int] = 250
DEFAULT_RESTARTS: Final[int] = 8
DEFAULT_SEED: Final[int] = 20250308
DEFAULT_TOLERANCE: Final[float] = 1.0e-7
DEFAULT_RHOBEG: Final[float] = 0.5
DEFAULT_FTOL: Final[float] = 1.0e-9
DEFAULT_FINITE_DIFFERENCE_STEP: Final[float] = 1.0e-8
DEFAULT_GAMMA_BOUNDS: Final[tuple[float, float]] = (
    0.0,
    2.0 * math.pi,
)
DEFAULT_BETA_BOUNDS: Final[tuple[float, float]] = (
    0.0,
    math.pi,
)


class QAOAOptimizerError(RuntimeError):
    """Raised when Qiskit optimization cannot be executed safely."""


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise QAOAOptimizerError(
            f"{name} must be strictly positive."
        )
    return value


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise QAOAOptimizerError(
            f"{name} must be non-negative."
        )
    return value


def _positive_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise QAOAOptimizerError(
            f"{name} must be finite and strictly positive."
        )
    return normalized


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise QAOAOptimizerError(f"{name} must be finite.")
    return normalized


def _optional_positive_float(
    value: float | None,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name=name)


def _readonly_vector(
    values: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.ndim != 1:
        raise QAOAOptimizerError(
            f"{name} must be one-dimensional."
        )
    if array.size != expected_size:
        raise QAOAOptimizerError(
            f"{name} contains {array.size} values; "
            f"expected {expected_size}."
        )
    if not np.all(np.isfinite(array)):
        raise QAOAOptimizerError(
            f"{name} contains non-finite values."
        )
    result = np.array(
        array,
        dtype=REAL_DTYPE,
        copy=True,
        order="C",
    )
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _json_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    source = {} if metadata is None else dict(metadata)
    try:
        encoded = json.dumps(
            source,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise QAOAOptimizerError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _normalize_bounds(
    bounds: Sequence[float],
    *,
    name: str,
) -> tuple[float, float]:
    values = tuple(bounds)
    if len(values) != 2:
        raise QAOAOptimizerError(
            f"{name} must contain exactly two values."
        )
    lower = _finite_float(values[0], name=f"{name}[0]")
    upper = _finite_float(values[1], name=f"{name}[1]")
    if not lower < upper:
        raise QAOAOptimizerError(
            f"{name} lower bound must be smaller than upper bound."
        )
    return lower, upper


def _qiskit_algorithms_version() -> str:
    try:
        return version("qiskit-algorithms")
    except PackageNotFoundError:
        try:
            module = import_module("qiskit_algorithms")
        except ImportError as exc:
            raise QAOAOptimizerError(
                "qiskit-algorithms is required. Install the pinned "
                "Google Colab dependency before optimization."
            ) from exc
        return str(getattr(module, "__version__", "unknown"))


def _load_qiskit_optimizer_components() -> tuple[Any, Any, str]:
    """Load Qiskit Algorithms lazily and return its optimizer namespace."""

    try:
        optimizers = import_module("qiskit_algorithms.optimizers")
        utilities = import_module("qiskit_algorithms.utils")
    except ImportError as exc:
        raise QAOAOptimizerError(
            "qiskit-algorithms is required. The CSSF optimizer does not "
            "fall back to scipy.optimize or a custom implementation."
        ) from exc

    algorithm_globals = getattr(
        utilities,
        "algorithm_globals",
        None,
    )
    if algorithm_globals is None:
        raise QAOAOptimizerError(
            "qiskit_algorithms.utils.algorithm_globals is unavailable."
        )

    return optimizers, algorithm_globals, _qiskit_algorithms_version()


@dataclass(frozen=True, slots=True)
class QAOAOptimizationConfig:
    """Validated configuration for Qiskit Algorithms multi-start runs."""

    optimizer_name: OptimizerName = DEFAULT_OPTIMIZER
    maxiter: int = DEFAULT_MAXITER
    restarts: int = DEFAULT_RESTARTS
    seed: int = DEFAULT_SEED
    tolerance: float | None = DEFAULT_TOLERANCE
    use_bounds: bool = True
    gamma_bounds: tuple[float, float] = DEFAULT_GAMMA_BOUNDS
    beta_bounds: tuple[float, float] = DEFAULT_BETA_BOUNDS
    initial_point_strategy: InitialPointStrategy = (
        "midpoint_then_uniform"
    )
    rhobeg: float = DEFAULT_RHOBEG
    ftol: float = DEFAULT_FTOL
    finite_difference_step: float = DEFAULT_FINITE_DIFFERENCE_STEP
    spsa_blocking: bool = False
    spsa_trust_region: bool = False
    spsa_second_order: bool = False
    spsa_resamplings: int = 1
    stop_after_first_success: bool = False

    def __post_init__(self) -> None:
        name = str(self.optimizer_name).upper()
        if name not in SUPPORTED_OPTIMIZERS:
            raise QAOAOptimizerError(
                "optimizer_name must be one of: "
                + ", ".join(SUPPORTED_OPTIMIZERS)
                + "."
            )
        if self.initial_point_strategy not in (
            "midpoint_then_uniform",
            "uniform",
        ):
            raise QAOAOptimizerError(
                "initial_point_strategy must be midpoint_then_uniform "
                "or uniform."
            )

        for field_name in (
            "use_bounds",
            "spsa_blocking",
            "spsa_trust_region",
            "spsa_second_order",
            "stop_after_first_success",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean.")

        object.__setattr__(self, "optimizer_name", name)
        object.__setattr__(
            self,
            "maxiter",
            _positive_integer(self.maxiter, name="maxiter"),
        )
        object.__setattr__(
            self,
            "restarts",
            _positive_integer(self.restarts, name="restarts"),
        )
        object.__setattr__(
            self,
            "seed",
            _nonnegative_integer(self.seed, name="seed"),
        )
        object.__setattr__(
            self,
            "tolerance",
            _optional_positive_float(
                self.tolerance,
                name="tolerance",
            ),
        )
        object.__setattr__(
            self,
            "gamma_bounds",
            _normalize_bounds(
                self.gamma_bounds,
                name="gamma_bounds",
            ),
        )
        object.__setattr__(
            self,
            "beta_bounds",
            _normalize_bounds(
                self.beta_bounds,
                name="beta_bounds",
            ),
        )
        object.__setattr__(
            self,
            "rhobeg",
            _positive_float(self.rhobeg, name="rhobeg"),
        )
        object.__setattr__(
            self,
            "ftol",
            _positive_float(self.ftol, name="ftol"),
        )
        object.__setattr__(
            self,
            "finite_difference_step",
            _positive_float(
                self.finite_difference_step,
                name="finite_difference_step",
            ),
        )
        object.__setattr__(
            self,
            "spsa_resamplings",
            _positive_integer(
                self.spsa_resamplings,
                name="spsa_resamplings",
            ),
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOAOptimizationConfig-v1\0")
        digest.update(
            json.dumps(
                {
                    "optimizer_name": self.optimizer_name,
                    "maxiter": self.maxiter,
                    "restarts": self.restarts,
                    "seed": self.seed,
                    "tolerance": self.tolerance,
                    "use_bounds": self.use_bounds,
                    "gamma_bounds": self.gamma_bounds,
                    "beta_bounds": self.beta_bounds,
                    "initial_point_strategy": (
                        self.initial_point_strategy
                    ),
                    "rhobeg": self.rhobeg,
                    "ftol": self.ftol,
                    "finite_difference_step": (
                        self.finite_difference_step
                    ),
                    "spsa_blocking": self.spsa_blocking,
                    "spsa_trust_region": self.spsa_trust_region,
                    "spsa_second_order": self.spsa_second_order,
                    "spsa_resamplings": self.spsa_resamplings,
                    "stop_after_first_success": (
                        self.stop_after_first_success
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAOAOptimizationRun:
    """Immutable result of one Qiskit optimizer restart."""

    restart_index: int
    seed: int
    initial_parameters: NDArray[np.float64]
    optimizer_parameters: NDArray[np.float64]
    optimizer_value: float
    best_parameters: NDArray[np.float64]
    best_value: float
    objective_requests: int
    objective_executions: int
    objective_cache_hits: int
    nfev: int | None
    nit: int | None
    njev: int | None
    success: bool | None
    message: str | None

    def __post_init__(self) -> None:
        restart_index = _nonnegative_integer(
            self.restart_index,
            name="restart_index",
        )
        seed = _nonnegative_integer(self.seed, name="seed")
        initial = np.asarray(
            self.initial_parameters,
            dtype=REAL_DTYPE,
        )
        optimizer_point = np.asarray(
            self.optimizer_parameters,
            dtype=REAL_DTYPE,
        )
        best = np.asarray(self.best_parameters, dtype=REAL_DTYPE)

        if initial.ndim != 1 or initial.size == 0:
            raise QAOAOptimizerError(
                "initial_parameters must be a non-empty vector."
            )
        parameter_count = int(initial.size)
        initial = _readonly_vector(
            initial,
            expected_size=parameter_count,
            name="initial_parameters",
        )
        optimizer_point = _readonly_vector(
            optimizer_point,
            expected_size=parameter_count,
            name="optimizer_parameters",
        )
        best = _readonly_vector(
            best,
            expected_size=parameter_count,
            name="best_parameters",
        )

        for field_name in (
            "objective_requests",
            "objective_executions",
            "objective_cache_hits",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )
        if self.objective_executions > self.objective_requests:
            raise QAOAOptimizerError(
                "objective_executions cannot exceed requests."
            )
        if self.objective_cache_hits > self.objective_requests:
            raise QAOAOptimizerError(
                "objective_cache_hits cannot exceed requests."
            )

        for field_name in ("nfev", "nit", "njev"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _nonnegative_integer(value, name=field_name),
                )
        if self.success is not None and not isinstance(
            self.success,
            bool,
        ):
            raise TypeError("success must be boolean or None.")

        object.__setattr__(self, "restart_index", restart_index)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "initial_parameters", initial)
        object.__setattr__(
            self,
            "optimizer_parameters",
            optimizer_point,
        )
        object.__setattr__(self, "best_parameters", best)
        object.__setattr__(
            self,
            "optimizer_value",
            _finite_float(
                self.optimizer_value,
                name="optimizer_value",
            ),
        )
        object.__setattr__(
            self,
            "best_value",
            _finite_float(self.best_value, name="best_value"),
        )
        object.__setattr__(
            self,
            "message",
            None if self.message is None else str(self.message),
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOAOptimizationRun-v1\0")
        digest.update(self.initial_parameters.tobytes(order="C"))
        digest.update(self.optimizer_parameters.tobytes(order="C"))
        digest.update(self.best_parameters.tobytes(order="C"))
        digest.update(
            np.asarray(
                [
                    self.restart_index,
                    self.seed,
                    self.objective_requests,
                    self.objective_executions,
                    self.objective_cache_hits,
                    -1 if self.nfev is None else self.nfev,
                    -1 if self.nit is None else self.nit,
                    -1 if self.njev is None else self.njev,
                ],
                dtype=np.int64,
            ).tobytes(order="C")
        )
        digest.update(
            np.asarray(
                [self.optimizer_value, self.best_value],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        digest.update(str(self.success).encode("ascii"))
        digest.update(str(self.message).encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAOAOptimizationResult:
    """Complete multi-start Qiskit optimization result."""

    optimizer_name: str
    qiskit_algorithms_version: str
    config_fingerprint: str
    objective_fingerprint: str
    runs: tuple[QAOAOptimizationRun, ...]
    best_restart_index: int
    best_parameters: NDArray[np.float64]
    best_value: float
    total_requests: int
    total_executions: int
    total_cache_hits: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        optimizer_name = str(self.optimizer_name).upper()
        if optimizer_name not in SUPPORTED_OPTIMIZERS:
            raise QAOAOptimizerError(
                "optimizer_name is unsupported."
            )
        runs = tuple(self.runs)
        if not runs:
            raise QAOAOptimizerError(
                "runs must contain at least one restart."
            )
        if not all(
            isinstance(run, QAOAOptimizationRun)
            for run in runs
        ):
            raise TypeError(
                "runs must contain QAOAOptimizationRun objects."
            )
        best_restart_index = _nonnegative_integer(
            self.best_restart_index,
            name="best_restart_index",
        )
        if best_restart_index >= len(runs):
            raise QAOAOptimizerError(
                "best_restart_index is outside runs."
            )
        parameter_count = int(runs[0].best_parameters.size)
        if any(
            run.best_parameters.size != parameter_count
            for run in runs
        ):
            raise QAOAOptimizerError(
                "All runs must use the same parameter dimension."
            )
        best_parameters = _readonly_vector(
            self.best_parameters,
            expected_size=parameter_count,
            name="best_parameters",
        )
        best_value = _finite_float(
            self.best_value,
            name="best_value",
        )

        for field_name in (
            "total_requests",
            "total_executions",
            "total_cache_hits",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )
        if self.total_executions > self.total_requests:
            raise QAOAOptimizerError(
                "total_executions cannot exceed total_requests."
            )

        object.__setattr__(self, "optimizer_name", optimizer_name)
        object.__setattr__(
            self,
            "qiskit_algorithms_version",
            str(self.qiskit_algorithms_version),
        )
        object.__setattr__(self, "runs", runs)
        object.__setattr__(
            self,
            "best_restart_index",
            best_restart_index,
        )
        object.__setattr__(
            self,
            "best_parameters",
            best_parameters,
        )
        object.__setattr__(self, "best_value", best_value)
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(self.metadata),
        )

        for field_name in (
            "config_fingerprint",
            "objective_fingerprint",
        ):
            digest = str(getattr(self, field_name)).lower()
            if len(digest) != 64:
                raise QAOAOptimizerError(
                    f"{field_name} must be a SHA-256 digest."
                )
            try:
                int(digest, 16)
            except ValueError as exc:
                raise QAOAOptimizerError(
                    f"{field_name} must be hexadecimal."
                ) from exc
            object.__setattr__(self, field_name, digest)

    @property
    def best_values_by_restart(self) -> NDArray[np.float64]:
        result = np.asarray(
            [run.best_value for run in self.runs],
            dtype=REAL_DTYPE,
        )
        result.setflags(write=False)
        return result

    @property
    def best_parameter_values(self) -> QAOAParameterValues:
        repetitions = self.best_parameters.size // 2
        return QAOAParameterValues(
            self.best_parameters[:repetitions],
            self.best_parameters[repetitions:],
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAOAOptimizationResult-v1\0")
        digest.update(self.optimizer_name.encode("ascii"))
        digest.update(
            self.qiskit_algorithms_version.encode("utf-8")
        )
        digest.update(self.config_fingerprint.encode("ascii"))
        digest.update(self.objective_fingerprint.encode("ascii"))
        for run in self.runs:
            digest.update(run.fingerprint().encode("ascii"))
        digest.update(self.best_parameters.tobytes(order="C"))
        digest.update(
            np.asarray(
                [
                    self.best_restart_index,
                    self.total_requests,
                    self.total_executions,
                    self.total_cache_hits,
                ],
                dtype=np.int64,
            ).tobytes(order="C")
        )
        digest.update(
            np.asarray(
                [self.best_value],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        digest.update(
            json.dumps(
                dict(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint(),
            "optimizer_name": self.optimizer_name,
            "qiskit_algorithms_version": (
                self.qiskit_algorithms_version
            ),
            "config_fingerprint": self.config_fingerprint,
            "objective_fingerprint": self.objective_fingerprint,
            "restart_count": len(self.runs),
            "best_restart_index": self.best_restart_index,
            "best_value": self.best_value,
            "best_parameters": self.best_parameters.tolist(),
            "best_values_by_restart": (
                self.best_values_by_restart.tolist()
            ),
            "total_requests": self.total_requests,
            "total_executions": self.total_executions,
            "total_cache_hits": self.total_cache_hits,
            "execution_engine": "qiskit_aer.AerSimulator",
            "execution_method": "statevector",
            "execution_device": "GPU",
            "optimizer_engine": "qiskit_algorithms.optimizers",
            "metadata": dict(self.metadata),
        }


def qaoa_parameter_bounds(
    repetitions: int,
    config: QAOAOptimizationConfig,
) -> tuple[tuple[float, float], ...]:
    """Return gamma-first bounds matching the circuit parameter layout."""

    repetitions = _positive_integer(
        repetitions,
        name="repetitions",
    )
    if not isinstance(config, QAOAOptimizationConfig):
        raise TypeError(
            "config must be QAOAOptimizationConfig."
        )
    return (
        (config.gamma_bounds,) * repetitions
        + (config.beta_bounds,) * repetitions
    )


def generate_initial_points(
    parameter_count: int,
    config: QAOAOptimizationConfig,
    *,
    initial_points: Sequence[ArrayLike] | None = None,
) -> tuple[NDArray[np.float64], ...]:
    """Generate deterministic bounded gamma-first restart points."""

    parameter_count = _positive_integer(
        parameter_count,
        name="parameter_count",
    )
    if parameter_count % 2 != 0:
        raise QAOAOptimizerError(
            "QAOA parameter_count must be even."
        )
    if not isinstance(config, QAOAOptimizationConfig):
        raise TypeError(
            "config must be QAOAOptimizationConfig."
        )

    repetitions = parameter_count // 2
    bounds = qaoa_parameter_bounds(repetitions, config)
    supplied = () if initial_points is None else tuple(initial_points)
    if len(supplied) > config.restarts:
        raise QAOAOptimizerError(
            "initial_points contains more entries than restarts."
        )

    result: list[NDArray[np.float64]] = []
    for index, values in enumerate(supplied):
        point = _readonly_vector(
            values,
            expected_size=parameter_count,
            name=f"initial_points[{index}]",
        )
        for parameter_index, (value, bound) in enumerate(
            zip(point, bounds)
        ):
            if value < bound[0] or value > bound[1]:
                raise QAOAOptimizerError(
                    f"initial_points[{index}][{parameter_index}] "
                    "lies outside configured bounds."
                )
        result.append(point)

    rng = np.random.default_rng(config.seed)
    while len(result) < config.restarts:
        if (
            config.initial_point_strategy
            == "midpoint_then_uniform"
            and not result
        ):
            point = np.asarray(
                [
                    0.5 * (lower + upper)
                    for lower, upper in bounds
                ],
                dtype=REAL_DTYPE,
            )
        else:
            lower = np.asarray(
                [bound[0] for bound in bounds],
                dtype=REAL_DTYPE,
            )
            upper = np.asarray(
                [bound[1] for bound in bounds],
                dtype=REAL_DTYPE,
            )
            point = rng.uniform(lower, upper)
        readonly = np.array(point, dtype=REAL_DTYPE, copy=True)
        readonly[readonly == 0.0] = 0.0
        readonly.setflags(write=False)
        result.append(readonly)

    return tuple(result)


def _build_qiskit_optimizer(
    config: QAOAOptimizationConfig,
    optimizers: Any,
) -> Any:
    """Instantiate an official Qiskit Algorithms optimizer."""

    if config.optimizer_name == "COBYLA":
        optimizer_class = getattr(optimizers, "COBYLA", None)
        kwargs = {
            "maxiter": config.maxiter,
            "disp": False,
            "rhobeg": config.rhobeg,
            "tol": config.tolerance,
        }
    elif config.optimizer_name == "SPSA":
        optimizer_class = getattr(optimizers, "SPSA", None)
        kwargs = {
            "maxiter": config.maxiter,
            "blocking": config.spsa_blocking,
            "trust_region": config.spsa_trust_region,
            "second_order": config.spsa_second_order,
            "resamplings": config.spsa_resamplings,
        }
    elif config.optimizer_name == "SLSQP":
        optimizer_class = getattr(optimizers, "SLSQP", None)
        kwargs = {
            "maxiter": config.maxiter,
            "disp": False,
            "ftol": config.ftol,
            "tol": config.tolerance,
            "eps": config.finite_difference_step,
            "max_evals_grouped": 1,
        }
    elif config.optimizer_name == "L_BFGS_B":
        optimizer_class = getattr(optimizers, "L_BFGS_B", None)
        kwargs = {
            "maxfun": config.maxiter,
            "maxiter": config.maxiter,
            "ftol": config.ftol,
            "iprint": -1,
            "eps": config.finite_difference_step,
            "max_evals_grouped": 1,
        }
    else:
        raise QAOAOptimizerError(
            f"Unsupported optimizer {config.optimizer_name!r}."
        )

    if optimizer_class is None:
        raise QAOAOptimizerError(
            f"Qiskit Algorithms does not expose "
            f"{config.optimizer_name}."
        )
    try:
        return optimizer_class(**kwargs)
    except Exception as exc:
        raise QAOAOptimizerError(
            f"Cannot initialize Qiskit optimizer "
            f"{config.optimizer_name}: {exc}"
        ) from exc


def _optional_result_integer(
    result: Any,
    name: str,
) -> int | None:
    value = getattr(result, name, None)
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise QAOAOptimizerError(
            f"Qiskit OptimizerResult.{name} is invalid."
        ) from exc
    return _nonnegative_integer(normalized, name=name)


def optimize_qaoa_gpu(
    objective: QAOAGPUObjective,
    *,
    config: QAOAOptimizationConfig | None = None,
    initial_points: Sequence[ArrayLike] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QAOAOptimizationResult:
    """Optimize exact QAOA through Qiskit Algorithms and Aer GPU."""

    if not isinstance(objective, QAOAGPUObjective):
        raise TypeError(
            "objective must be QAOAGPUObjective."
        )
    optimization_config = (
        QAOAOptimizationConfig() if config is None else config
    )
    if not isinstance(
        optimization_config,
        QAOAOptimizationConfig,
    ):
        raise TypeError(
            "config must be QAOAOptimizationConfig or None."
        )
    if objective.hamiltonian.n_qubits > (
        COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
    ):
        raise QAOAOptimizerError(
            "Exact QAOA optimization exceeds the project-wide "
            f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}-qubit limit."
        )

    optimizers, algorithm_globals, algorithms_version = (
        _load_qiskit_optimizer_components()
    )
    points = generate_initial_points(
        objective.parameter_count,
        optimization_config,
        initial_points=initial_points,
    )
    repetitions = objective.parameter_count // 2
    bounds = (
        list(qaoa_parameter_bounds(repetitions, optimization_config))
        if optimization_config.use_bounds
        else None
    )

    starting_snapshot = objective.snapshot()
    runs: list[QAOAOptimizationRun] = []

    for restart_index, initial_point in enumerate(points):
        restart_seed = optimization_config.seed + restart_index
        algorithm_globals.random_seed = restart_seed
        qiskit_optimizer = _build_qiskit_optimizer(
            optimization_config,
            optimizers,
        )

        run_start = objective.snapshot()
        local_best_value = math.inf
        local_best_parameters = np.array(
            initial_point,
            dtype=REAL_DTYPE,
            copy=True,
        )

        def tracked_objective(values: ArrayLike) -> float:
            nonlocal local_best_value, local_best_parameters
            value = float(objective(values))
            point = _readonly_vector(
                values,
                expected_size=objective.parameter_count,
                name="optimizer parameters",
            )
            if (
                value < local_best_value
                or (
                    math.isclose(
                        value,
                        local_best_value,
                        rel_tol=0.0,
                        abs_tol=objective.config.tie_atol,
                    )
                    and point.tobytes(order="C")
                    < local_best_parameters.tobytes(order="C")
                )
            ):
                local_best_value = value
                local_best_parameters = np.array(
                    point,
                    dtype=REAL_DTYPE,
                    copy=True,
                )
            return value

        try:
            raw_result = qiskit_optimizer.minimize(
                fun=tracked_objective,
                x0=np.array(
                    initial_point,
                    dtype=REAL_DTYPE,
                    copy=True,
                ),
                bounds=bounds,
            )
        except Exception as exc:
            raise QAOAOptimizerError(
                f"Qiskit {optimization_config.optimizer_name} failed "
                f"at restart {restart_index}: {exc}"
            ) from exc

        raw_x = getattr(raw_result, "x", None)
        raw_fun = getattr(raw_result, "fun", None)
        if raw_x is None or raw_fun is None:
            raise QAOAOptimizerError(
                "Qiskit OptimizerResult must expose x and fun."
            )
        optimizer_parameters = _readonly_vector(
            raw_x,
            expected_size=objective.parameter_count,
            name="OptimizerResult.x",
        )
        optimizer_value = _finite_float(
            raw_fun,
            name="OptimizerResult.fun",
        )

        if not math.isfinite(local_best_value):
            local_best_value = float(
                tracked_objective(optimizer_parameters)
            )
        local_best_parameters = _readonly_vector(
            local_best_parameters,
            expected_size=objective.parameter_count,
            name="local_best_parameters",
        )

        run_end = objective.snapshot()
        run = QAOAOptimizationRun(
            restart_index=restart_index,
            seed=restart_seed,
            initial_parameters=initial_point,
            optimizer_parameters=optimizer_parameters,
            optimizer_value=optimizer_value,
            best_parameters=local_best_parameters,
            best_value=local_best_value,
            objective_requests=(
                run_end.requests - run_start.requests
            ),
            objective_executions=(
                run_end.executions - run_start.executions
            ),
            objective_cache_hits=(
                run_end.cache_hits - run_start.cache_hits
            ),
            nfev=_optional_result_integer(raw_result, "nfev"),
            nit=_optional_result_integer(raw_result, "nit"),
            njev=_optional_result_integer(raw_result, "njev"),
            success=(
                None
                if getattr(raw_result, "success", None) is None
                else bool(getattr(raw_result, "success"))
            ),
            message=(
                None
                if getattr(raw_result, "message", None) is None
                else str(getattr(raw_result, "message"))
            ),
        )
        runs.append(run)

        if (
            optimization_config.stop_after_first_success
            and run.success is True
        ):
            break

    if not runs:
        raise QAOAOptimizerError(
            "Qiskit optimization produced no runs."
        )

    best_restart_index = min(
        range(len(runs)),
        key=lambda index: (
            runs[index].best_value,
            runs[index].best_parameters.tobytes(order="C"),
            runs[index].restart_index,
        ),
    )
    best_run = runs[best_restart_index]
    ending_snapshot = objective.snapshot()

    return QAOAOptimizationResult(
        optimizer_name=optimization_config.optimizer_name,
        qiskit_algorithms_version=algorithms_version,
        config_fingerprint=optimization_config.fingerprint(),
        objective_fingerprint=objective.fingerprint(),
        runs=tuple(runs),
        best_restart_index=best_restart_index,
        best_parameters=best_run.best_parameters,
        best_value=best_run.best_value,
        total_requests=(
            ending_snapshot.requests - starting_snapshot.requests
        ),
        total_executions=(
            ending_snapshot.executions
            - starting_snapshot.executions
        ),
        total_cache_hits=(
            ending_snapshot.cache_hits
            - starting_snapshot.cache_hits
        ),
        metadata=metadata,
    )


__all__ = [
    "REAL_DTYPE",
    "OptimizerName",
    "InitialPointStrategy",
    "SUPPORTED_OPTIMIZERS",
    "DEFAULT_OPTIMIZER",
    "DEFAULT_MAXITER",
    "DEFAULT_RESTARTS",
    "DEFAULT_SEED",
    "DEFAULT_TOLERANCE",
    "DEFAULT_RHOBEG",
    "DEFAULT_FTOL",
    "DEFAULT_FINITE_DIFFERENCE_STEP",
    "DEFAULT_GAMMA_BOUNDS",
    "DEFAULT_BETA_BOUNDS",
    "QAOAOptimizerError",
    "QAOAOptimizationConfig",
    "QAOAOptimizationRun",
    "QAOAOptimizationResult",
    "qaoa_parameter_bounds",
    "generate_initial_points",
    "optimize_qaoa_gpu",
]
