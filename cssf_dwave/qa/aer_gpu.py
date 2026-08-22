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

"""Strict Qiskit Aer GPU execution for CSSF digitized-QA circuits.

The module accepts only a fully materialized, fixed-angle
:class:`qa.circuit.QACircuitArtifact`. It copies the circuit, appends the Aer
statevector save instruction, transpiles through Qiskit, and executes only
``qiskit_aer.AerSimulator(method="statevector", device="GPU",
precision="double")``.

No CPU fallback, custom simulator, approximate statevector engine, or silent
backend substitution is permitted. The exact statevector path is capped at
22 qubits before Qiskit/Aer loading, backend construction, transpilation, or
statevector allocation. Qiskit and Aer remain lazy runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qa import (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    QA_ALGORITHM_NAME,
    QISKIT_QUBIT_ORDER,
    STATEVECTOR_DEVICE,
    STATEVECTOR_METHOD,
    STATEVECTOR_PRECISION,
    validate_exact_statevector_qubit_count,
)
from qa.circuit import QACircuitArtifact
from qaoa.aer_gpu import (
    DEFAULT_NORM_TOLERANCE,
    DEFAULT_OPTIMIZATION_LEVEL,
    DEFAULT_SEED_SIMULATOR,
    DEFAULT_SEED_TRANSPILER,
    DEFAULT_VALIDATION_THRESHOLD,
    DEFAULT_ZERO_THRESHOLD,
    AerGPUConfig,
    AerGPUEnvironment,
    AerGPUError,
    AerGPUUnavailableError,
    create_gpu_statevector_backend,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
COMPLEX_DTYPE: Final[np.dtype[np.complex128]] = np.dtype(np.complex128)
DEFAULT_MAX_STATEVECTOR_QUBITS: Final[int] = (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
)
DEFAULT_PROBABILITY_CONSISTENCY_TOLERANCE: Final[float] = 1.0e-12


class QAAerGPUError(RuntimeError):
    """Base error for strict digitized-QA Aer GPU execution."""


class QAAerGPUUnavailableError(QAAerGPUError):
    """Raised when the required Qiskit Aer GPU path is unavailable."""


class QAAerGPUExecutionError(QAAerGPUError):
    """Raised when circuit preparation, transpilation, or execution fails."""


class QAAerGPUResultError(QAAerGPUError):
    """Raised when Aer returns a malformed digitized-QA statevector."""


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise QAAerGPUError(f"{name} must be positive.")
    return value


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise QAAerGPUError(f"{name} must be non-negative.")
    return value


def _sha256_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise QAAerGPUResultError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return normalized


def _readonly_complex_vector(
    values: ArrayLike,
    *,
    expected_size: int,
) -> NDArray[np.complex128]:
    vector = np.array(
        np.asarray(values, dtype=COMPLEX_DTYPE).reshape(-1),
        dtype=COMPLEX_DTYPE,
        order="C",
        copy=True,
    )
    if vector.size != expected_size:
        raise QAAerGPUResultError(
            f"statevector contains {vector.size} amplitudes; "
            f"expected {expected_size}."
        )
    if not np.all(np.isfinite(vector.real)) or not np.all(
        np.isfinite(vector.imag)
    ):
        raise QAAerGPUResultError(
            "statevector contains non-finite amplitudes."
        )
    vector.setflags(write=False)
    return vector


def _readonly_probability_vector(
    values: ArrayLike,
    *,
    expected_size: int,
) -> NDArray[np.float64]:
    probabilities = np.array(
        np.asarray(values, dtype=REAL_DTYPE).reshape(-1),
        dtype=REAL_DTYPE,
        order="C",
        copy=True,
    )
    if probabilities.size != expected_size:
        raise QAAerGPUResultError(
            "probability-vector size does not match the statevector."
        )
    if not np.all(np.isfinite(probabilities)):
        raise QAAerGPUResultError(
            "probabilities contain non-finite values."
        )
    if np.any(probabilities < 0.0):
        raise QAAerGPUResultError("probabilities contain negative values.")
    probabilities.setflags(write=False)
    return probabilities


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _immutable_json_mapping(
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    try:
        encoded = json.dumps(
            dict(metadata),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise QAAerGPUResultError(
            "metadata must be JSON-serializable."
        ) from exc
    decoded = json.loads(encoded)
    frozen = _freeze_json(decoded)
    if not isinstance(frozen, Mapping):
        raise QAAerGPUResultError("metadata normalization failed.")
    return frozen


@dataclass(frozen=True, slots=True)
class QAAerGPUConfig(AerGPUConfig):
    """Digitized-QA specialization of the shared strict Aer GPU config."""

    def __post_init__(self) -> None:
        try:
            super(QAAerGPUConfig, self).__post_init__()
        except (AerGPUError, TypeError) as exc:
            raise QAAerGPUError(str(exc)) from exc

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAAerGPUConfig-v1\0")
        digest.update(AerGPUConfig.fingerprint(self).encode("ascii"))
        digest.update(QA_ALGORITHM_NAME.encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAAerGPUEnvironment:
    """Audited Qiskit/Aer capabilities for digitized-QA execution."""

    qiskit_version: str
    qiskit_aer_version: str
    backend_name: str
    available_devices: tuple[str, ...]
    available_methods: tuple[str, ...]
    config_fingerprint: str

    def __post_init__(self) -> None:
        devices = tuple(str(item) for item in self.available_devices)
        methods = tuple(str(item) for item in self.available_methods)
        if STATEVECTOR_DEVICE not in devices:
            raise QAAerGPUUnavailableError(
                f"Aer devices {devices!r} do not include GPU."
            )
        if methods and STATEVECTOR_METHOD not in methods:
            raise QAAerGPUUnavailableError(
                f"Aer methods {methods!r} do not include statevector."
            )
        object.__setattr__(self, "available_devices", devices)
        object.__setattr__(self, "available_methods", methods)
        object.__setattr__(
            self,
            "config_fingerprint",
            _sha256_digest(
                self.config_fingerprint,
                name="config_fingerprint",
            ),
        )

    @classmethod
    def from_shared(
        cls,
        environment: AerGPUEnvironment,
    ) -> "QAAerGPUEnvironment":
        if not isinstance(environment, AerGPUEnvironment):
            raise TypeError("environment must be AerGPUEnvironment.")
        return cls(
            qiskit_version=environment.qiskit_version,
            qiskit_aer_version=environment.qiskit_aer_version,
            backend_name=environment.backend_name,
            available_devices=environment.available_devices,
            available_methods=environment.available_methods,
            config_fingerprint=environment.config_fingerprint,
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAAerGPUEnvironment-v1\0")
        digest.update(
            json.dumps(
                {
                    "qiskit_version": self.qiskit_version,
                    "qiskit_aer_version": self.qiskit_aer_version,
                    "backend_name": self.backend_name,
                    "available_devices": self.available_devices,
                    "available_methods": self.available_methods,
                    "config_fingerprint": self.config_fingerprint,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QAStatevectorResult:
    """Immutable exact statevector with digitized-QA ownership metadata."""

    statevector: NDArray[np.complex128]
    probabilities: NDArray[np.float64]
    n_qubits: int
    variable_order: tuple[str, ...]
    schedule_fingerprint: str
    source_schedule_fingerprint: str
    hamiltonian_fingerprint: str
    plan_fingerprint: str
    circuit_fingerprint: str
    config_fingerprint: str
    environment_fingerprint: str
    transpiled_depth: int
    transpiled_size: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        n_qubits = _positive_integer(self.n_qubits, name="n_qubits")
        try:
            validate_exact_statevector_qubit_count(n_qubits)
        except (TypeError, ValueError) as exc:
            raise QAAerGPUResultError(str(exc)) from exc

        expected_size = 1 << n_qubits
        statevector = _readonly_complex_vector(
            self.statevector,
            expected_size=expected_size,
        )
        probabilities = _readonly_probability_vector(
            self.probabilities,
            expected_size=expected_size,
        )
        expected_probabilities = np.abs(statevector) ** 2
        if not np.allclose(
            probabilities,
            expected_probabilities,
            rtol=0.0,
            atol=DEFAULT_PROBABILITY_CONSISTENCY_TOLERANCE,
        ):
            raise QAAerGPUResultError(
                "probabilities are inconsistent with the statevector."
            )

        variable_order = tuple(str(item) for item in self.variable_order)
        if len(variable_order) != n_qubits:
            raise QAAerGPUResultError(
                "variable_order length must equal n_qubits."
            )
        if len(set(variable_order)) != len(variable_order):
            raise QAAerGPUResultError(
                "variable_order must contain unique labels."
            )
        if any(not item for item in variable_order):
            raise QAAerGPUResultError(
                "variable_order labels must be non-empty."
            )

        for name in (
            "schedule_fingerprint",
            "source_schedule_fingerprint",
            "hamiltonian_fingerprint",
            "plan_fingerprint",
            "circuit_fingerprint",
            "config_fingerprint",
            "environment_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _sha256_digest(getattr(self, name), name=name),
            )

        object.__setattr__(self, "statevector", statevector)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "variable_order", variable_order)
        object.__setattr__(
            self,
            "transpiled_depth",
            _nonnegative_integer(
                self.transpiled_depth,
                name="transpiled_depth",
            ),
        )
        object.__setattr__(
            self,
            "transpiled_size",
            _nonnegative_integer(
                self.transpiled_size,
                name="transpiled_size",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_json_mapping(self.metadata),
        )

    @property
    def norm(self) -> float:
        return float(np.sum(self.probabilities))

    @property
    def most_probable_index(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def most_probable_probability(self) -> float:
        return float(self.probabilities[self.most_probable_index])

    def binary_sample(self, basis_index: int) -> NDArray[np.int8]:
        """Decode a Qiskit basis index in little-endian qubit order."""

        index = _nonnegative_integer(basis_index, name="basis_index")
        if index >= self.probabilities.size:
            raise QAAerGPUResultError(
                "basis_index is outside the statevector."
            )
        sample = (
            (
                np.uint64(index)
                >> np.arange(self.n_qubits, dtype=np.uint64)
            )
            & np.uint64(1)
        ).astype(np.int8)
        sample.setflags(write=False)
        return sample

    def most_probable_sample(self) -> NDArray[np.int8]:
        return self.binary_sample(self.most_probable_index)

    def labeled_sample(self, basis_index: int) -> dict[str, int]:
        sample = self.binary_sample(basis_index)
        return {
            variable: int(value)
            for variable, value in zip(self.variable_order, sample)
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QAStatevectorResult-v1\0")
        digest.update(self.statevector.tobytes(order="C"))
        digest.update(self.probabilities.tobytes(order="C"))
        for value in (
            self.schedule_fingerprint,
            self.source_schedule_fingerprint,
            self.hamiltonian_fingerprint,
            self.plan_fingerprint,
            self.circuit_fingerprint,
            self.config_fingerprint,
            self.environment_fingerprint,
        ):
            digest.update(value.encode("ascii"))
        digest.update(
            json.dumps(
                {
                    "n_qubits": self.n_qubits,
                    "variable_order": self.variable_order,
                    "transpiled_depth": self.transpiled_depth,
                    "transpiled_size": self.transpiled_size,
                    "metadata": _thaw_json(self.metadata),
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "cssf-digitized-qa-statevector-result-v1",
            "algorithm": QA_ALGORITHM_NAME,
            "execution_engine": "qiskit_aer.AerSimulator",
            "method": STATEVECTOR_METHOD,
            "device": STATEVECTOR_DEVICE,
            "precision": STATEVECTOR_PRECISION,
            "qubit_order": QISKIT_QUBIT_ORDER,
            "n_qubits": self.n_qubits,
            "variable_order": list(self.variable_order),
            "schedule_fingerprint": self.schedule_fingerprint,
            "source_schedule_fingerprint": (
                self.source_schedule_fingerprint
            ),
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "circuit_fingerprint": self.circuit_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "transpiled_depth": self.transpiled_depth,
            "transpiled_size": self.transpiled_size,
            "probability_norm": self.norm,
            "most_probable_index": self.most_probable_index,
            "most_probable_probability": self.most_probable_probability,
            "result_fingerprint": self.fingerprint(),
        }


def _load_qiskit_transpile() -> tuple[Any, str]:
    """Load qiskit.transpile lazily without selecting another backend."""

    try:
        qiskit = importlib.import_module("qiskit")
    except ImportError as exc:
        raise QAAerGPUUnavailableError(
            "Qiskit is unavailable in the Google Colab runtime."
        ) from exc
    transpile = getattr(qiskit, "transpile", None)
    if transpile is None or not callable(transpile):
        raise QAAerGPUUnavailableError("qiskit.transpile is unavailable.")
    return transpile, str(getattr(qiskit, "__version__", "unknown"))


def _require_fixed_circuit(circuit: Any) -> None:
    parameters = getattr(circuit, "parameters", None)
    if parameters is None:
        raise QAAerGPUExecutionError(
            "Digitized-QA circuit does not expose parameters."
        )
    try:
        remaining = tuple(parameters)
    except TypeError as exc:
        raise QAAerGPUExecutionError(
            "Cannot inspect digitized-QA circuit parameters."
        ) from exc
    if remaining:
        raise QAAerGPUExecutionError(
            "Digitized-QA circuit must contain only fixed numeric angles."
        )


def _copy_circuit(circuit: Any) -> Any:
    copy_method = getattr(circuit, "copy", None)
    if copy_method is None or not callable(copy_method):
        raise QAAerGPUExecutionError(
            "Qiskit circuit copy() is unavailable; immutable execution "
            "cannot be guaranteed."
        )
    try:
        copied = copy_method()
    except Exception as exc:
        raise QAAerGPUExecutionError(
            "Cannot copy the digitized-QA circuit for execution."
        ) from exc
    if copied is circuit:
        raise QAAerGPUExecutionError(
            "Qiskit circuit copy() returned the original object."
        )
    return copied


def _save_statevector(circuit: Any) -> None:
    save_statevector = getattr(circuit, "save_statevector", None)
    if save_statevector is None or not callable(save_statevector):
        raise QAAerGPUExecutionError(
            "Qiskit Aer save_statevector instruction is unavailable."
        )
    try:
        save_statevector()
    except Exception as exc:
        raise QAAerGPUExecutionError(
            "Cannot append the Qiskit Aer save_statevector instruction."
        ) from exc


def _integer_metric(circuit: Any, name: str) -> int:
    value = getattr(circuit, name, None)
    if value is None:
        return 0
    try:
        metric = value() if callable(value) else value
        normalized = int(metric)
    except Exception as exc:
        raise QAAerGPUResultError(
            f"Cannot read transpiled circuit {name}."
        ) from exc
    return _nonnegative_integer(normalized, name=f"transpiled_{name}")


def create_qa_gpu_statevector_backend(
    config: QAAerGPUConfig | None = None,
) -> tuple[Any, QAAerGPUEnvironment]:
    """Create and audit the shared strict Aer GPU backend for QA."""

    run_config = QAAerGPUConfig() if config is None else config
    if not isinstance(run_config, QAAerGPUConfig):
        raise TypeError("config must be QAAerGPUConfig or None.")
    try:
        backend, shared_environment = create_gpu_statevector_backend(
            run_config
        )
    except AerGPUUnavailableError as exc:
        raise QAAerGPUUnavailableError(str(exc)) from exc
    except AerGPUError as exc:
        raise QAAerGPUError(str(exc)) from exc
    return backend, QAAerGPUEnvironment.from_shared(shared_environment)


def audit_qa_aer_gpu_environment(
    config: QAAerGPUConfig | None = None,
) -> QAAerGPUEnvironment:
    """Audit strict Qiskit Aer GPU availability without running a circuit."""

    _, environment = create_qa_gpu_statevector_backend(config)
    return environment


def run_qa_statevector_gpu(
    artifact: QACircuitArtifact,
    config: QAAerGPUConfig | None = None,
) -> QAStatevectorResult:
    """Execute one fixed-angle digitized-QA circuit on Qiskit Aer GPU."""

    if not isinstance(artifact, QACircuitArtifact):
        raise TypeError("artifact must be QACircuitArtifact.")
    run_config = QAAerGPUConfig() if config is None else config
    if not isinstance(run_config, QAAerGPUConfig):
        raise TypeError("config must be QAAerGPUConfig or None.")

    n_qubits = artifact.plan.n_qubits
    try:
        validate_exact_statevector_qubit_count(n_qubits)
    except (TypeError, ValueError) as exc:
        raise QAAerGPUExecutionError(str(exc)) from exc
    if n_qubits > run_config.max_statevector_qubits:
        raise QAAerGPUExecutionError(
            f"Circuit has {n_qubits} qubits; configured statevector limit "
            f"is {run_config.max_statevector_qubits}."
        )

    circuit = artifact.circuit
    if getattr(circuit, "num_qubits", None) != n_qubits:
        raise QAAerGPUExecutionError(
            "Circuit qubit count differs from its digitized-QA plan."
        )
    _require_fixed_circuit(circuit)

    metadata = getattr(circuit, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise QAAerGPUExecutionError(
            "Circuit metadata is unavailable."
        )
    expected_metadata = {
        "plan_fingerprint": artifact.plan.fingerprint(),
        "hamiltonian_fingerprint": (
            artifact.plan.hamiltonian.fingerprint()
        ),
        "schedule_fingerprint": artifact.plan.schedule.fingerprint(),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise QAAerGPUExecutionError(
                f"Circuit metadata field {key!r} does not match its plan."
            )

    execution_circuit = _copy_circuit(circuit)
    _require_fixed_circuit(execution_circuit)
    _save_statevector(execution_circuit)

    transpile, transpile_qiskit_version = _load_qiskit_transpile()
    backend, environment = create_qa_gpu_statevector_backend(run_config)
    if (
        artifact.qiskit_version != "unknown"
        and transpile_qiskit_version != "unknown"
        and artifact.qiskit_version != transpile_qiskit_version
    ):
        raise QAAerGPUExecutionError(
            "Circuit and execution Qiskit versions differ."
        )

    try:
        transpiled = transpile(
            execution_circuit,
            backend=backend,
            optimization_level=run_config.optimization_level,
            seed_transpiler=run_config.seed_transpiler,
        )
    except Exception as exc:
        raise QAAerGPUExecutionError(
            "Qiskit transpilation for digitized-QA Aer GPU failed."
        ) from exc
    _require_fixed_circuit(transpiled)

    try:
        job = backend.run(transpiled, **run_config.run_options())
        aer_result = job.result()
    except Exception as exc:
        raise QAAerGPUExecutionError(
            "Qiskit Aer GPU digitized-QA statevector job failed."
        ) from exc

    success = getattr(aer_result, "success", True)
    if not bool(success):
        status = getattr(aer_result, "status", "unknown")
        raise QAAerGPUExecutionError(
            f"Qiskit Aer reported an unsuccessful job: {status}."
        )
    get_statevector = getattr(aer_result, "get_statevector", None)
    if get_statevector is None or not callable(get_statevector):
        raise QAAerGPUResultError(
            "Aer result lacks get_statevector()."
        )
    try:
        raw_statevector = get_statevector(transpiled)
    except Exception as exc:
        raise QAAerGPUResultError(
            "Cannot extract the saved statevector from Aer result."
        ) from exc

    expected_size = 1 << n_qubits
    statevector = _readonly_complex_vector(
        raw_statevector,
        expected_size=expected_size,
    )
    probabilities = _readonly_probability_vector(
        np.abs(statevector) ** 2,
        expected_size=expected_size,
    )
    norm = float(np.sum(probabilities))
    if not math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=run_config.norm_tolerance,
    ):
        raise QAAerGPUResultError(
            f"Statevector probability norm is {norm:.17g}, not 1."
        )

    result_metadata: dict[str, Any] = {
        "framework": "CSSF",
        "algorithm": QA_ALGORITHM_NAME,
        "execution_engine": "qiskit_aer.AerSimulator",
        "method": STATEVECTOR_METHOD,
        "device": STATEVECTOR_DEVICE,
        "precision": run_config.precision,
        "qubit_order": QISKIT_QUBIT_ORDER,
        "cpu_fallback": False,
        "seed_transpiler": run_config.seed_transpiler,
        "seed_simulator": run_config.seed_simulator,
        "probability_norm": norm,
        "slice_count": artifact.plan.slice_count,
        "trotter_order": artifact.plan.trotter_order,
        "splitting_policy": artifact.plan.splitting_policy,
        "fixed_angle_circuit": True,
    }
    backend_metadata = getattr(aer_result, "metadata", None)
    if isinstance(backend_metadata, Mapping):
        result_metadata["aer_metadata"] = dict(backend_metadata)

    return QAStatevectorResult(
        statevector=statevector,
        probabilities=probabilities,
        n_qubits=n_qubits,
        variable_order=artifact.plan.hamiltonian.variable_order,
        schedule_fingerprint=artifact.plan.schedule.fingerprint(),
        source_schedule_fingerprint=(
            artifact.plan.schedule.source_schedule_fingerprint
        ),
        hamiltonian_fingerprint=(
            artifact.plan.hamiltonian.fingerprint()
        ),
        plan_fingerprint=artifact.plan.fingerprint(),
        circuit_fingerprint=artifact.fingerprint(),
        config_fingerprint=run_config.fingerprint(),
        environment_fingerprint=environment.fingerprint(),
        transpiled_depth=_integer_metric(transpiled, "depth"),
        transpiled_size=_integer_metric(transpiled, "size"),
        metadata=result_metadata,
    )


def run_digitized_qa_statevector_gpu(
    artifact: QACircuitArtifact,
    config: QAAerGPUConfig | None = None,
) -> QAStatevectorResult:
    """Explicitly named alias for :func:`run_qa_statevector_gpu`."""

    return run_qa_statevector_gpu(artifact, config)


__all__ = [
    "REAL_DTYPE",
    "COMPLEX_DTYPE",
    "STATEVECTOR_METHOD",
    "STATEVECTOR_DEVICE",
    "STATEVECTOR_PRECISION",
    "DEFAULT_OPTIMIZATION_LEVEL",
    "DEFAULT_SEED_TRANSPILER",
    "DEFAULT_SEED_SIMULATOR",
    "DEFAULT_NORM_TOLERANCE",
    "DEFAULT_ZERO_THRESHOLD",
    "DEFAULT_VALIDATION_THRESHOLD",
    "DEFAULT_PROBABILITY_CONSISTENCY_TOLERANCE",
    "COLAB_FREE_STATEVECTOR_QUBIT_LIMIT",
    "DEFAULT_MAX_STATEVECTOR_QUBITS",
    "QAAerGPUError",
    "QAAerGPUUnavailableError",
    "QAAerGPUExecutionError",
    "QAAerGPUResultError",
    "QAAerGPUConfig",
    "QAAerGPUEnvironment",
    "QAStatevectorResult",
    "create_qa_gpu_statevector_backend",
    "audit_qa_aer_gpu_environment",
    "run_qa_statevector_gpu",
    "run_digitized_qa_statevector_gpu",
]
