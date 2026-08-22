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

"""Production objective callable for exact MA-QAOA on Qiskit Aer GPU.

The objective delegates every quantum evaluation to
:func:`maqaoa.expectation.evaluate_maqaoa_parameters_gpu`, which in turn uses only
``qiskit_aer.AerSimulator(method="statevector", device="GPU")`` and Qiskit
``Statevector.expectation_value``. This module implements no circuit simulator,
state evolution, or custom expectation engine.

The wrapper adds strict parameter validation, one-entry-safe LRU caching,
thread-safe accounting, bounded scalar history, deterministic tie-breaking,
and reproducibility manifests. Exact cache keys use float64 bytes; parameters
are never rounded or reduced modulo an assumed period.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import math
from threading import RLock
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from maqaoa import COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
from maqaoa.aer_gpu import MAQAOAAerGPUConfig
from maqaoa.circuit import MAQAOACircuitArtifact
from maqaoa.expectation import (
    MAQAOACostOperatorArtifact,
    MAQAOAExpectationConfig,
    MAQAOAExpectationError,
    MAQAOAGPUEvaluation,
    build_maqaoa_qiskit_cost_operator,
    evaluate_maqaoa_parameters_gpu,
)
from maqaoa.parameters import (
    MAQAOAParameterError,
    MAQAOAParameterValues,
)
from qaoa.hamiltonian import IsingHamiltonian


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
DEFAULT_CACHE_CAPACITY: Final[int] = 1
DEFAULT_HISTORY_CAPACITY: Final[int] = 4096
DEFAULT_TIE_ATOL: Final[float] = 1.0e-12


class MAQAOAObjectiveError(RuntimeError):
    """Raised when the exact GPU objective cannot be evaluated safely."""


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise MAQAOAObjectiveError(
            f"{name} must be strictly positive."
        )
    return value


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise MAQAOAObjectiveError(
            f"{name} must be non-negative."
        )
    return value


def _positive_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise MAQAOAObjectiveError(
            f"{name} must be finite and strictly positive."
        )
    return normalized


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MAQAOAObjectiveError(f"{name} must be finite.")
    return normalized


def _sha256_digest(value: str, *, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64:
        raise MAQAOAObjectiveError(
            f"{name} must be a SHA-256 digest."
        )
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise MAQAOAObjectiveError(
            f"{name} must contain hexadecimal characters."
        ) from exc
    return normalized


def _readonly_parameters(
    values: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.ndim != 1:
        raise MAQAOAObjectiveError(
            f"{name} must be one-dimensional."
        )
    if array.size != expected_size:
        raise MAQAOAObjectiveError(
            f"{name} contains {array.size} values; "
            f"expected {expected_size}."
        )
    if not np.all(np.isfinite(array)):
        raise MAQAOAObjectiveError(
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
        raise MAQAOAObjectiveError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    return MappingProxyType(json.loads(encoded))



def _gpu_evaluation_fingerprint(
    evaluation: MAQAOAGPUEvaluation,
) -> str:
    """Fingerprint a GPU evaluation without relying on mutable APIs."""

    if not isinstance(evaluation, MAQAOAGPUEvaluation):
        raise TypeError(
            "evaluation must be MAQAOAGPUEvaluation."
        )
    digest = hashlib.sha256()
    digest.update(b"CSSF-MAQAOAGPUEvaluation-v1\0")
    digest.update(
        evaluation.parameters.flat().tobytes(order="C")
    )
    digest.update(
        evaluation.statevector_result.fingerprint().encode("ascii")
    )
    digest.update(
        evaluation.expectation_result.fingerprint().encode("ascii")
    )
    return digest.hexdigest()

def _parameter_key(parameters: NDArray[np.float64]) -> bytes:
    """Return an exact canonical float64 cache key."""

    return parameters.tobytes(order="C")


@dataclass(frozen=True, slots=True)
class MAQAOAObjectiveConfig:
    """Memory, history, and tie-breaking policy for objective calls."""

    cache_enabled: bool = True
    cache_capacity: int = DEFAULT_CACHE_CAPACITY
    history_capacity: int = DEFAULT_HISTORY_CAPACITY
    record_cache_hits: bool = True
    retain_best_evaluation: bool = False
    tie_atol: float = DEFAULT_TIE_ATOL

    def __post_init__(self) -> None:
        for field_name in (
            "cache_enabled",
            "record_cache_hits",
            "retain_best_evaluation",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean.")

        cache_capacity = _positive_integer(
            self.cache_capacity,
            name="cache_capacity",
        )
        history_capacity = _positive_integer(
            self.history_capacity,
            name="history_capacity",
        )
        tie_atol = _positive_float(
            self.tie_atol,
            name="tie_atol",
        )

        object.__setattr__(
            self,
            "cache_capacity",
            cache_capacity,
        )
        object.__setattr__(
            self,
            "history_capacity",
            history_capacity,
        )
        object.__setattr__(self, "tie_atol", tie_atol)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAObjectiveConfig-v1\0")
        digest.update(
            json.dumps(
                {
                    "cache_enabled": self.cache_enabled,
                    "cache_capacity": self.cache_capacity,
                    "history_capacity": self.history_capacity,
                    "record_cache_hits": self.record_cache_hits,
                    "retain_best_evaluation": (
                        self.retain_best_evaluation
                    ),
                    "tie_atol": self.tie_atol,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MAQAOAObjectiveRecord:
    """Compact scalar record for one optimizer objective request."""

    request_index: int
    execution_index: int
    cache_hit: bool
    parameters: NDArray[np.float64]
    objective_value: float
    evaluation_fingerprint: str
    statevector_fingerprint: str
    expectation_fingerprint: str

    def __post_init__(self) -> None:
        request_index = _positive_integer(
            self.request_index,
            name="request_index",
        )
        execution_index = _positive_integer(
            self.execution_index,
            name="execution_index",
        )
        if not isinstance(self.cache_hit, bool):
            raise TypeError("cache_hit must be boolean.")
        parameters = np.asarray(
            self.parameters,
            dtype=REAL_DTYPE,
        )
        if parameters.ndim != 1 or parameters.size == 0:
            raise MAQAOAObjectiveError(
                "parameters must be a non-empty vector."
            )
        if not np.all(np.isfinite(parameters)):
            raise MAQAOAObjectiveError(
                "parameters contain non-finite values."
            )
        readonly = np.array(
            parameters,
            dtype=REAL_DTYPE,
            copy=True,
            order="C",
        )
        readonly[readonly == 0.0] = 0.0
        readonly.setflags(write=False)

        object.__setattr__(
            self,
            "request_index",
            request_index,
        )
        object.__setattr__(
            self,
            "execution_index",
            execution_index,
        )
        object.__setattr__(self, "parameters", readonly)
        object.__setattr__(
            self,
            "objective_value",
            _finite_float(
                self.objective_value,
                name="objective_value",
            ),
        )
        for field_name in (
            "evaluation_fingerprint",
            "statevector_fingerprint",
            "expectation_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256_digest(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAObjectiveRecord-v1\0")
        digest.update(self.parameters.tobytes(order="C"))
        digest.update(
            np.asarray(
                [
                    self.request_index,
                    self.execution_index,
                    int(self.cache_hit),
                ],
                dtype=np.int64,
            ).tobytes(order="C")
        )
        digest.update(
            np.asarray(
                [self.objective_value],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        digest.update(
            self.evaluation_fingerprint.encode("ascii")
        )
        digest.update(
            self.statevector_fingerprint.encode("ascii")
        )
        digest.update(
            self.expectation_fingerprint.encode("ascii")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MAQAOAObjectiveSnapshot:
    """Immutable state summary for checkpoints and optimizer callbacks."""

    requests: int
    executions: int
    cache_hits: int
    best_value: float | None
    best_parameters: NDArray[np.float64] | None
    best_evaluation_fingerprint: str | None
    records: tuple[MAQAOAObjectiveRecord, ...]
    objective_fingerprint: str

    def __post_init__(self) -> None:
        requests = _nonnegative_integer(
            self.requests,
            name="requests",
        )
        executions = _nonnegative_integer(
            self.executions,
            name="executions",
        )
        cache_hits = _nonnegative_integer(
            self.cache_hits,
            name="cache_hits",
        )
        if executions > requests:
            raise MAQAOAObjectiveError(
                "executions cannot exceed requests."
            )
        if cache_hits > requests:
            raise MAQAOAObjectiveError(
                "cache_hits cannot exceed requests."
            )

        if self.best_value is None:
            if (
                self.best_parameters is not None
                or self.best_evaluation_fingerprint is not None
            ):
                raise MAQAOAObjectiveError(
                    "Best-result fields must be all present or all absent."
                )
            best_parameters = None
        else:
            _finite_float(self.best_value, name="best_value")
            if self.best_parameters is None:
                raise MAQAOAObjectiveError(
                    "best_parameters are required with best_value."
                )
            best_parameters = np.array(
                np.asarray(
                    self.best_parameters,
                    dtype=REAL_DTYPE,
                ).reshape(-1),
                dtype=REAL_DTYPE,
                copy=True,
                order="C",
            )
            if not np.all(np.isfinite(best_parameters)):
                raise MAQAOAObjectiveError(
                    "best_parameters contain non-finite values."
                )
            best_parameters[best_parameters == 0.0] = 0.0
            best_parameters.setflags(write=False)
            if self.best_evaluation_fingerprint is None:
                raise MAQAOAObjectiveError(
                    "best_evaluation_fingerprint is required."
                )
            _sha256_digest(
                self.best_evaluation_fingerprint,
                name="best_evaluation_fingerprint",
            )

        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "executions", executions)
        object.__setattr__(self, "cache_hits", cache_hits)
        object.__setattr__(
            self,
            "best_parameters",
            best_parameters,
        )
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(
            self,
            "objective_fingerprint",
            _sha256_digest(
                self.objective_fingerprint,
                name="objective_fingerprint",
            ),
        )


class MAQAOAGPUObjective:
    """Thread-safe callable evaluated exclusively through Qiskit Aer GPU."""

    def __init__(
        self,
        artifact: MAQAOACircuitArtifact,
        hamiltonian: IsingHamiltonian,
        *,
        aer_config: MAQAOAAerGPUConfig | None = None,
        expectation_config: MAQAOAExpectationConfig | None = None,
        operator_artifact: MAQAOACostOperatorArtifact | None = None,
        config: MAQAOAObjectiveConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(artifact, MAQAOACircuitArtifact):
            raise TypeError(
                "artifact must be MAQAOACircuitArtifact."
            )
        if not isinstance(hamiltonian, IsingHamiltonian):
            raise TypeError(
                "hamiltonian must be IsingHamiltonian."
            )
        if (
            artifact.plan.hamiltonian.fingerprint()
            != hamiltonian.fingerprint()
        ):
            raise MAQAOAObjectiveError(
                "Circuit artifact and Hamiltonian differ."
            )
        if (
            hamiltonian.n_qubits
            > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
        ):
            raise MAQAOAObjectiveError(
                "Exact MA-QAOA objective exceeds the project-wide "
                f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}-qubit limit."
            )

        self.artifact = artifact
        self.hamiltonian = hamiltonian
        self.aer_config = (
            MAQAOAAerGPUConfig() if aer_config is None else aer_config
        )
        self.expectation_config = (
            MAQAOAExpectationConfig()
            if expectation_config is None
            else expectation_config
        )
        self.config = (
            MAQAOAObjectiveConfig()
            if config is None
            else config
        )

        if not isinstance(self.aer_config, MAQAOAAerGPUConfig):
            raise TypeError(
                "aer_config must be MAQAOAAerGPUConfig or None."
            )
        if not isinstance(
            self.expectation_config,
            MAQAOAExpectationConfig,
        ):
            raise TypeError(
                "expectation_config must be MAQAOAExpectationConfig "
                "or None."
            )
        if not isinstance(self.config, MAQAOAObjectiveConfig):
            raise TypeError(
                "config must be MAQAOAObjectiveConfig or None."
            )

        self.operator_artifact = (
            build_maqaoa_qiskit_cost_operator(hamiltonian)
            if operator_artifact is None
            else operator_artifact
        )
        if not isinstance(
            self.operator_artifact,
            MAQAOACostOperatorArtifact,
        ):
            raise TypeError(
                "operator_artifact must be "
                "MAQAOACostOperatorArtifact or None."
            )
        if (
            self.operator_artifact.hamiltonian_fingerprint
            != hamiltonian.fingerprint()
        ):
            raise MAQAOAObjectiveError(
                "Qiskit operator belongs to another Hamiltonian."
            )

        self.metadata = _json_metadata(metadata)
        self._lock = RLock()
        self._cache: OrderedDict[
            bytes,
            MAQAOAGPUEvaluation,
        ] = OrderedDict()
        self._records: deque[MAQAOAObjectiveRecord] = deque(
            maxlen=self.config.history_capacity
        )
        self._requests = 0
        self._executions = 0
        self._cache_hits = 0
        self._best_value: float | None = None
        self._best_parameters: NDArray[np.float64] | None = None
        self._best_evaluation_fingerprint: str | None = None
        self._best_evaluation: MAQAOAGPUEvaluation | None = None

    @property
    def parameter_count(self) -> int:
        return self.artifact.plan.parameter_layout.total_parameter_count

    @property
    def requests(self) -> int:
        with self._lock:
            return self._requests

    @property
    def executions(self) -> int:
        with self._lock:
            return self._executions

    @property
    def cache_hits(self) -> int:
        with self._lock:
            return self._cache_hits

    @property
    def best_value(self) -> float | None:
        with self._lock:
            return self._best_value

    @property
    def best_parameters(self) -> NDArray[np.float64] | None:
        with self._lock:
            if self._best_parameters is None:
                return None
            result = np.array(
                self._best_parameters,
                dtype=REAL_DTYPE,
                copy=True,
            )
            result.setflags(write=False)
            return result

    @property
    def best_evaluation(self) -> MAQAOAGPUEvaluation | None:
        with self._lock:
            return self._best_evaluation

    def _normalize_parameters(
        self,
        values: MAQAOAParameterValues | ArrayLike,
    ) -> tuple[MAQAOAParameterValues, NDArray[np.float64]]:
        try:
            if isinstance(values, MAQAOAParameterValues):
                parameters = values
                layout = self.artifact.plan.parameter_layout
                if parameters.repetitions != layout.repetitions:
                    raise MAQAOAObjectiveError(
                        "Parameter repetition count differs from circuit."
                    )
                if parameters.cost_term_count != layout.cost_term_count:
                    raise MAQAOAObjectiveError(
                        "Parameter cost-term count differs from circuit."
                    )
                if parameters.n_qubits != layout.n_qubits:
                    raise MAQAOAObjectiveError(
                        "Parameter qubit count differs from circuit."
                    )
            else:
                parameters = (
                    self.artifact.plan.parameter_layout.split(values)
                )
        except MAQAOAParameterError as exc:
            raise MAQAOAObjectiveError(
                f"Invalid MA-QAOA parameters: {exc}"
            ) from exc

        flat = _readonly_parameters(
            parameters.flat(),
            expected_size=self.parameter_count,
            name="parameters",
        )
        normalized = self.artifact.plan.parameter_layout.split(flat)
        return normalized, flat

    def _is_better(
        self,
        value: float,
        parameters: NDArray[np.float64],
    ) -> bool:
        if self._best_value is None or self._best_parameters is None:
            return True
        if value < self._best_value - self.config.tie_atol:
            return True
        if math.isclose(
            value,
            self._best_value,
            rel_tol=0.0,
            abs_tol=self.config.tie_atol,
        ):
            return tuple(parameters.tolist()) < tuple(
                self._best_parameters.tolist()
            )
        return False

    def _record(
        self,
        *,
        parameters: NDArray[np.float64],
        evaluation: MAQAOAGPUEvaluation,
        cache_hit: bool,
        request_index: int,
        execution_index: int,
    ) -> None:
        record = MAQAOAObjectiveRecord(
            request_index=request_index,
            execution_index=execution_index,
            cache_hit=cache_hit,
            parameters=parameters,
            objective_value=evaluation.objective_value,
            evaluation_fingerprint=_gpu_evaluation_fingerprint(evaluation),
            statevector_fingerprint=(
                evaluation.statevector_result.fingerprint()
            ),
            expectation_fingerprint=(
                evaluation.expectation_result.fingerprint()
            ),
        )
        if not cache_hit or self.config.record_cache_hits:
            self._records.append(record)

    def evaluate(
        self,
        values: MAQAOAParameterValues | ArrayLike,
    ) -> MAQAOAGPUEvaluation:
        """Run or reuse one exact Qiskit Aer GPU evaluation."""

        parameters, flat = self._normalize_parameters(values)
        key = _parameter_key(flat)

        with self._lock:
            self._requests += 1
            request_index = self._requests

            if self.config.cache_enabled and key in self._cache:
                evaluation = self._cache.pop(key)
                self._cache[key] = evaluation
                self._cache_hits += 1
                self._record(
                    parameters=flat,
                    evaluation=evaluation,
                    cache_hit=True,
                    request_index=request_index,
                    execution_index=self._executions,
                )
                return evaluation

            try:
                evaluation = evaluate_maqaoa_parameters_gpu(
                    self.artifact,
                    parameters,
                    self.hamiltonian,
                    aer_config=self.aer_config,
                    expectation_config=self.expectation_config,
                    operator_artifact=self.operator_artifact,
                )
            except MAQAOAExpectationError as exc:
                raise MAQAOAObjectiveError(
                    f"Qiskit Aer GPU objective failed: {exc}"
                ) from exc

            if not isinstance(evaluation, MAQAOAGPUEvaluation):
                raise MAQAOAObjectiveError(
                    "GPU evaluator returned an unexpected result type."
                )
            objective_value = _finite_float(
                evaluation.objective_value,
                name="objective_value",
            )
            self._executions += 1
            execution_index = self._executions

            if self.config.cache_enabled:
                self._cache[key] = evaluation
                while (
                    len(self._cache)
                    > self.config.cache_capacity
                ):
                    self._cache.popitem(last=False)

            if self._is_better(objective_value, flat):
                best = np.array(
                    flat,
                    dtype=REAL_DTYPE,
                    copy=True,
                )
                best.setflags(write=False)
                self._best_value = objective_value
                self._best_parameters = best
                self._best_evaluation_fingerprint = (
                    _gpu_evaluation_fingerprint(evaluation)
                )
                self._best_evaluation = (
                    evaluation
                    if self.config.retain_best_evaluation
                    else None
                )

            self._record(
                parameters=flat,
                evaluation=evaluation,
                cache_hit=False,
                request_index=request_index,
                execution_index=execution_index,
            )
            return evaluation

    def __call__(
        self,
        values: MAQAOAParameterValues | ArrayLike,
    ) -> float:
        """Return the exact expected energy for optimizer interfaces."""

        return float(self.evaluate(values).objective_value)

    def clear_cache(self) -> None:
        """Release cached statevectors without altering accounting."""

        with self._lock:
            self._cache.clear()

    def reset_tracking(self, *, clear_cache: bool = False) -> None:
        """Reset counters, history, and best-result tracking."""

        if not isinstance(clear_cache, bool):
            raise TypeError("clear_cache must be boolean.")
        with self._lock:
            self._records.clear()
            self._requests = 0
            self._executions = 0
            self._cache_hits = 0
            self._best_value = None
            self._best_parameters = None
            self._best_evaluation_fingerprint = None
            self._best_evaluation = None
            if clear_cache:
                self._cache.clear()

    def fingerprint(self) -> str:
        """Return a stable identity for the objective definition."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAGPUObjective-v1\0")
        digest.update(self.artifact.fingerprint().encode("ascii"))
        digest.update(self.hamiltonian.fingerprint().encode("ascii"))
        digest.update(self.aer_config.fingerprint().encode("ascii"))
        digest.update(
            self.expectation_config.fingerprint().encode("ascii")
        )
        digest.update(
            self.operator_artifact.fingerprint().encode("ascii")
        )
        digest.update(self.config.fingerprint().encode("ascii"))
        digest.update(
            json.dumps(
                dict(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def snapshot(self) -> MAQAOAObjectiveSnapshot:
        """Return immutable scalar history and best parameters."""

        with self._lock:
            best_parameters = (
                None
                if self._best_parameters is None
                else np.array(
                    self._best_parameters,
                    dtype=REAL_DTYPE,
                    copy=True,
                )
            )
            if best_parameters is not None:
                best_parameters.setflags(write=False)
            return MAQAOAObjectiveSnapshot(
                requests=self._requests,
                executions=self._executions,
                cache_hits=self._cache_hits,
                best_value=self._best_value,
                best_parameters=best_parameters,
                best_evaluation_fingerprint=(
                    self._best_evaluation_fingerprint
                ),
                records=tuple(self._records),
                objective_fingerprint=self.fingerprint(),
            )

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-ready objective and execution manifest."""

        snapshot = self.snapshot()
        return {
            "fingerprint": snapshot.objective_fingerprint,
            "artifact_fingerprint": self.artifact.fingerprint(),
            "hamiltonian_fingerprint": (
                self.hamiltonian.fingerprint()
            ),
            "operator_fingerprint": (
                self.operator_artifact.fingerprint()
            ),
            "aer_config_fingerprint": (
                self.aer_config.fingerprint()
            ),
            "expectation_config_fingerprint": (
                self.expectation_config.fingerprint()
            ),
            "objective_config_fingerprint": (
                self.config.fingerprint()
            ),
            "n_qubits": self.hamiltonian.n_qubits,
            "repetitions": self.artifact.plan.config.repetitions,
            "parameter_count": self.parameter_count,
            "cost_term_count": (
                self.artifact.plan.parameter_layout.cost_term_count
            ),
            "execution_engine": "qiskit_aer.AerSimulator",
            "execution_method": "statevector",
            "execution_device": "GPU",
            "requests": snapshot.requests,
            "executions": snapshot.executions,
            "cache_hits": snapshot.cache_hits,
            "best_value": snapshot.best_value,
            "best_parameters": (
                None
                if snapshot.best_parameters is None
                else snapshot.best_parameters.tolist()
            ),
            "best_evaluation_fingerprint": (
                snapshot.best_evaluation_fingerprint
            ),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "REAL_DTYPE",
    "DEFAULT_CACHE_CAPACITY",
    "DEFAULT_HISTORY_CAPACITY",
    "DEFAULT_TIE_ATOL",
    "MAQAOAObjectiveError",
    "MAQAOAObjectiveConfig",
    "MAQAOAObjectiveRecord",
    "MAQAOAObjectiveSnapshot",
    "MAQAOAGPUObjective",
]
