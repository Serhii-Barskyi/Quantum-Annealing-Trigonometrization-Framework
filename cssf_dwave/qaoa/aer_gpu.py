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

"""Strict Qiskit Aer GPU execution for exact QAOA statevectors.

This module materializes no circuit and implements no custom simulator. It
accepts a :class:`qaoa.circuit.QAOACircuitArtifact`, binds its parameters,
transpiles through Qiskit, and executes only
``qiskit_aer.AerSimulator(method="statevector", device="GPU")``.

CPU fallback is intentionally forbidden. Exact statevector execution is
rejected above the project-wide 22-qubit ceiling before Qiskit/Aer loading,
backend construction, transpilation, or statevector allocation. A missing GPU,
unsupported statevector method, failed Aer job, malformed statevector, or
normalization error raises a dedicated exception.
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

from qaoa.circuit import (
    QAOACircuitArtifact,
    QAOACircuitError,
    QAOAParameterValues,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
COMPLEX_DTYPE: Final[np.dtype[np.complex128]] = np.dtype(np.complex128)
REQUIRED_METHOD: Final[str] = "statevector"
REQUIRED_DEVICE: Final[str] = "GPU"
DEFAULT_OPTIMIZATION_LEVEL: Final[int] = 1
DEFAULT_SEED_TRANSPILER: Final[int] = 271828
DEFAULT_SEED_SIMULATOR: Final[int] = 314159
DEFAULT_NORM_TOLERANCE: Final[float] = 1.0e-10
DEFAULT_ZERO_THRESHOLD: Final[float] = 1.0e-14
DEFAULT_VALIDATION_THRESHOLD: Final[float] = 1.0e-10
COLAB_FREE_STATEVECTOR_QUBIT_LIMIT: Final[int] = 22
DEFAULT_MAX_STATEVECTOR_QUBITS: Final[int] = (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
)


class AerGPUError(RuntimeError):
    """Base error for strict Qiskit Aer GPU execution."""


class AerGPUUnavailableError(AerGPUError):
    """Raised when the required Qiskit Aer GPU path is unavailable."""


class AerGPUExecutionError(AerGPUError):
    """Raised when transpilation or Aer execution fails."""


class AerGPUResultError(AerGPUError):
    """Raised when Aer returns an invalid statevector result."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise AerGPUError(f"{name} must be finite.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized <= 0.0:
        raise AerGPUError(f"{name} must be strictly positive.")
    return normalized


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise AerGPUError(f"{name} must be non-negative.")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    normalized = _nonnegative_integer(value, name=name)
    if normalized == 0:
        raise AerGPUError(f"{name} must be positive.")
    return normalized


def _readonly_complex_vector(
    values: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.complex128]:
    result = np.ascontiguousarray(
        np.asarray(values, dtype=COMPLEX_DTYPE).reshape(-1),
        dtype=COMPLEX_DTYPE,
    )
    if result.size != expected_size:
        raise AerGPUResultError(
            f"{name} contains {result.size} amplitudes; "
            f"expected {expected_size}."
        )
    if not np.all(np.isfinite(result.real)) or not np.all(
        np.isfinite(result.imag)
    ):
        raise AerGPUResultError(f"{name} contains non-finite amplitudes.")
    result.setflags(write=False)
    return result


def _readonly_probability_vector(
    values: ArrayLike,
    *,
    expected_size: int,
) -> NDArray[np.float64]:
    result = np.ascontiguousarray(
        np.asarray(values, dtype=REAL_DTYPE).reshape(-1),
        dtype=REAL_DTYPE,
    )
    if result.size != expected_size:
        raise AerGPUResultError(
            "Probability vector size does not match the statevector."
        )
    if not np.all(np.isfinite(result)):
        raise AerGPUResultError("Probabilities contain non-finite values.")
    if np.any(result < 0.0):
        raise AerGPUResultError("Probabilities contain negative values.")
    result.setflags(write=False)
    return result


def _json_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        encoded = json.dumps(
            dict(metadata),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise AerGPUResultError(
            "Result metadata must be JSON-serializable."
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def _backend_sequence(backend: Any, name: str) -> tuple[str, ...]:
    value = getattr(backend, name, None)
    if value is None:
        return tuple()
    try:
        raw = _call_or_value(value)
    except Exception as exc:
        raise AerGPUUnavailableError(
            f"Cannot query Aer backend {name}()."
        ) from exc
    return tuple(str(item) for item in raw)


@dataclass(frozen=True, slots=True)
class AerGPUConfig:
    """Validated strict configuration for Qiskit Aer statevector GPU."""

    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER
    seed_simulator: int = DEFAULT_SEED_SIMULATOR
    precision: str = "double"
    zero_threshold: float = DEFAULT_ZERO_THRESHOLD
    validation_threshold: float = DEFAULT_VALIDATION_THRESHOLD
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE
    max_statevector_qubits: int = DEFAULT_MAX_STATEVECTOR_QUBITS
    fusion_enable: bool = True
    fusion_threshold: int = 14

    def __post_init__(self) -> None:
        optimization_level = _nonnegative_integer(
            self.optimization_level,
            name="optimization_level",
        )
        if optimization_level > 3:
            raise AerGPUError("optimization_level must lie in [0, 3].")

        seed_transpiler = _nonnegative_integer(
            self.seed_transpiler,
            name="seed_transpiler",
        )
        seed_simulator = _nonnegative_integer(
            self.seed_simulator,
            name="seed_simulator",
        )
        maximum_qubits = _positive_integer(
            self.max_statevector_qubits,
            name="max_statevector_qubits",
        )
        if maximum_qubits > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT:
            raise AerGPUError(
                "max_statevector_qubits exceeds the project-wide "
                f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}-qubit limit."
            )
        fusion_threshold = _nonnegative_integer(
            self.fusion_threshold,
            name="fusion_threshold",
        )

        precision = str(self.precision).strip().lower()
        if precision != "double":
            raise AerGPUError(
                "Scientific production execution requires double precision."
            )
        if not isinstance(self.fusion_enable, bool):
            raise TypeError("fusion_enable must be boolean.")

        object.__setattr__(self, "optimization_level", optimization_level)
        object.__setattr__(self, "seed_transpiler", seed_transpiler)
        object.__setattr__(self, "seed_simulator", seed_simulator)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(
            self,
            "zero_threshold",
            _positive_float(self.zero_threshold, name="zero_threshold"),
        )
        object.__setattr__(
            self,
            "validation_threshold",
            _positive_float(
                self.validation_threshold,
                name="validation_threshold",
            ),
        )
        object.__setattr__(
            self,
            "norm_tolerance",
            _positive_float(self.norm_tolerance, name="norm_tolerance"),
        )
        object.__setattr__(self, "max_statevector_qubits", maximum_qubits)
        object.__setattr__(self, "fusion_threshold", fusion_threshold)

    def backend_options(self) -> dict[str, Any]:
        """Return the exact options used to construct AerSimulator."""

        return {
            "method": REQUIRED_METHOD,
            "device": REQUIRED_DEVICE,
            "precision": self.precision,
            "zero_threshold": self.zero_threshold,
            "validation_threshold": self.validation_threshold,
            "fusion_enable": self.fusion_enable,
            "fusion_threshold": self.fusion_threshold,
        }

    def run_options(self) -> dict[str, Any]:
        """Return deterministic Aer job options."""

        return {
            "shots": None,
            "seed_simulator": self.seed_simulator,
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-AerGPUConfig-v1\0")
        digest.update(
            json.dumps(
                {
                    "optimization_level": self.optimization_level,
                    "seed_transpiler": self.seed_transpiler,
                    "seed_simulator": self.seed_simulator,
                    "precision": self.precision,
                    "zero_threshold": self.zero_threshold,
                    "validation_threshold": self.validation_threshold,
                    "norm_tolerance": self.norm_tolerance,
                    "max_statevector_qubits": self.max_statevector_qubits,
                    "fusion_enable": self.fusion_enable,
                    "fusion_threshold": self.fusion_threshold,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AerGPUEnvironment:
    """Audited Qiskit/Aer runtime capabilities."""

    qiskit_version: str
    qiskit_aer_version: str
    backend_name: str
    available_devices: tuple[str, ...]
    available_methods: tuple[str, ...]
    config_fingerprint: str

    def __post_init__(self) -> None:
        devices = tuple(self.available_devices)
        methods = tuple(self.available_methods)
        if REQUIRED_DEVICE not in devices:
            raise AerGPUUnavailableError(
                f"Aer devices {devices!r} do not include GPU."
            )
        if methods and REQUIRED_METHOD not in methods:
            raise AerGPUUnavailableError(
                f"Aer methods {methods!r} do not include statevector."
            )
        if len(self.config_fingerprint) != 64:
            raise AerGPUError(
                "config_fingerprint must be a SHA-256 digest."
            )
        object.__setattr__(self, "available_devices", devices)
        object.__setattr__(self, "available_methods", methods)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-AerGPUEnvironment-v1\0")
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
class AerStatevectorResult:
    """Immutable exact statevector returned by Qiskit Aer GPU."""

    statevector: NDArray[np.complex128]
    probabilities: NDArray[np.float64]
    n_qubits: int
    variable_order: tuple[str, ...]
    circuit_fingerprint: str
    config_fingerprint: str
    environment_fingerprint: str
    transpiled_depth: int
    transpiled_size: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        n_qubits = _positive_integer(self.n_qubits, name="n_qubits")
        expected_size = 1 << n_qubits
        statevector = _readonly_complex_vector(
            self.statevector,
            expected_size=expected_size,
            name="statevector",
        )
        probabilities = _readonly_probability_vector(
            self.probabilities,
            expected_size=expected_size,
        )
        variable_order = tuple(self.variable_order)
        if len(variable_order) != n_qubits:
            raise AerGPUResultError(
                "variable_order length must equal n_qubits."
            )
        for name, value in (
            ("circuit_fingerprint", self.circuit_fingerprint),
            ("config_fingerprint", self.config_fingerprint),
            ("environment_fingerprint", self.environment_fingerprint),
        ):
            if len(value) != 64:
                raise AerGPUResultError(
                    f"{name} must be a SHA-256 digest."
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
        object.__setattr__(self, "metadata", _json_metadata(self.metadata))

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
            raise AerGPUResultError("basis_index is outside the statevector.")
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
        digest.update(b"CSSF-AerStatevectorResult-v1\0")
        digest.update(self.statevector.tobytes(order="C"))
        digest.update(self.probabilities.tobytes(order="C"))
        digest.update(self.circuit_fingerprint.encode("ascii"))
        digest.update(self.config_fingerprint.encode("ascii"))
        digest.update(self.environment_fingerprint.encode("ascii"))
        digest.update(
            json.dumps(
                {
                    "n_qubits": self.n_qubits,
                    "variable_order": self.variable_order,
                    "transpiled_depth": self.transpiled_depth,
                    "transpiled_size": self.transpiled_size,
                    "metadata": dict(self.metadata),
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        return digest.hexdigest()


def _load_qiskit_aer_api() -> tuple[Any, Any, str, str]:
    """Load Qiskit transpilation and AerSimulator lazily."""

    try:
        qiskit = importlib.import_module("qiskit")
    except ImportError as exc:
        raise AerGPUUnavailableError(
            "Qiskit is unavailable in the Google Colab runtime."
        ) from exc

    transpile = getattr(qiskit, "transpile", None)
    if transpile is None or not callable(transpile):
        raise AerGPUUnavailableError("qiskit.transpile is unavailable.")

    try:
        qiskit_aer = importlib.import_module("qiskit_aer")
    except ImportError as exc:
        raise AerGPUUnavailableError(
            "qiskit-aer-gpu is unavailable in the Google Colab runtime."
        ) from exc

    simulator_class = getattr(qiskit_aer, "AerSimulator", None)
    if simulator_class is None or not callable(simulator_class):
        raise AerGPUUnavailableError(
            "qiskit_aer.AerSimulator is unavailable."
        )

    return (
        transpile,
        simulator_class,
        str(getattr(qiskit, "__version__", "unknown")),
        str(getattr(qiskit_aer, "__version__", "unknown")),
    )


def create_gpu_statevector_backend(
    config: AerGPUConfig | None = None,
) -> tuple[Any, AerGPUEnvironment]:
    """Create and audit a strict Aer statevector GPU backend."""

    run_config = AerGPUConfig() if config is None else config
    if not isinstance(run_config, AerGPUConfig):
        raise TypeError("config must be AerGPUConfig or None.")

    _, simulator_class, qiskit_version, aer_version = (
        _load_qiskit_aer_api()
    )

    try:
        backend = simulator_class(**run_config.backend_options())
    except Exception as exc:
        raise AerGPUUnavailableError(
            "Cannot initialize AerSimulator with statevector GPU."
        ) from exc

    devices = _backend_sequence(backend, "available_devices")
    methods = _backend_sequence(backend, "available_methods")

    if REQUIRED_DEVICE not in devices:
        raise AerGPUUnavailableError(
            f"Aer GPU is unavailable; reported devices={devices!r}. "
            "CPU fallback is forbidden."
        )
    if methods and REQUIRED_METHOD not in methods:
        raise AerGPUUnavailableError(
            "Aer statevector method is unavailable."
        )

    name_value = getattr(backend, "name", None)
    backend_name = str(_call_or_value(name_value) if name_value else type(backend).__name__)

    environment = AerGPUEnvironment(
        qiskit_version=qiskit_version,
        qiskit_aer_version=aer_version,
        backend_name=backend_name,
        available_devices=devices,
        available_methods=methods,
        config_fingerprint=run_config.fingerprint(),
    )
    return backend, environment


def audit_aer_gpu_environment(
    config: AerGPUConfig | None = None,
) -> AerGPUEnvironment:
    """Validate the strict Qiskit Aer GPU path without running a circuit."""

    _, environment = create_gpu_statevector_backend(config)
    return environment


def _require_bound_circuit(circuit: Any) -> None:
    parameters = getattr(circuit, "parameters", None)
    if parameters is None:
        return
    try:
        remaining = tuple(parameters)
    except TypeError as exc:
        raise AerGPUExecutionError(
            "Cannot inspect bound-circuit parameters."
        ) from exc
    if remaining:
        raise AerGPUExecutionError(
            f"Circuit still contains {len(remaining)} unbound parameters."
        )


def _save_statevector(circuit: Any) -> None:
    save_statevector = getattr(circuit, "save_statevector", None)
    if save_statevector is None or not callable(save_statevector):
        raise AerGPUExecutionError(
            "Qiskit Aer save_statevector instruction is unavailable."
        )
    try:
        save_statevector()
    except Exception as exc:
        raise AerGPUExecutionError(
            "Cannot append the Qiskit Aer save_statevector instruction."
        ) from exc


def _integer_metric(circuit: Any, name: str) -> int:
    value = getattr(circuit, name, None)
    if value is None:
        return 0
    try:
        metric = _call_or_value(value)
    except Exception as exc:
        raise AerGPUResultError(
            f"Cannot read transpiled circuit {name}."
        ) from exc
    return _nonnegative_integer(int(metric), name=f"transpiled_{name}")


def run_qaoa_statevector_gpu(
    artifact: QAOACircuitArtifact,
    parameters: QAOAParameterValues | ArrayLike,
    config: AerGPUConfig | None = None,
) -> AerStatevectorResult:
    """Bind, transpile, and execute one QAOA circuit on Aer GPU."""

    if not isinstance(artifact, QAOACircuitArtifact):
        raise TypeError("artifact must be QAOACircuitArtifact.")

    run_config = AerGPUConfig() if config is None else config
    if not isinstance(run_config, AerGPUConfig):
        raise TypeError("config must be AerGPUConfig or None.")

    n_qubits = artifact.plan.n_qubits
    if n_qubits > run_config.max_statevector_qubits:
        raise AerGPUExecutionError(
            f"Circuit has {n_qubits} qubits; configured statevector limit "
            f"is {run_config.max_statevector_qubits}."
        )

    try:
        bound_circuit = artifact.bind(parameters)
    except QAOACircuitError as exc:
        raise AerGPUExecutionError(
            f"QAOA parameter binding failed: {exc}"
        ) from exc

    _require_bound_circuit(bound_circuit)
    _save_statevector(bound_circuit)

    transpile, _, _, _ = _load_qiskit_aer_api()
    backend, environment = create_gpu_statevector_backend(run_config)

    try:
        transpiled = transpile(
            bound_circuit,
            backend=backend,
            optimization_level=run_config.optimization_level,
            seed_transpiler=run_config.seed_transpiler,
        )
    except Exception as exc:
        raise AerGPUExecutionError(
            "Qiskit transpilation for Aer GPU failed."
        ) from exc

    _require_bound_circuit(transpiled)

    try:
        job = backend.run(transpiled, **run_config.run_options())
        aer_result = job.result()
    except Exception as exc:
        raise AerGPUExecutionError(
            "Qiskit Aer GPU statevector job failed."
        ) from exc

    success = getattr(aer_result, "success", True)
    if not bool(success):
        status = getattr(aer_result, "status", "unknown")
        raise AerGPUExecutionError(
            f"Qiskit Aer reported an unsuccessful job: {status}."
        )

    get_statevector = getattr(aer_result, "get_statevector", None)
    if get_statevector is None or not callable(get_statevector):
        raise AerGPUResultError(
            "Aer result lacks get_statevector()."
        )

    try:
        raw_statevector = get_statevector(transpiled)
    except Exception as exc:
        raise AerGPUResultError(
            "Cannot extract the saved statevector from Aer result."
        ) from exc

    expected_size = 1 << n_qubits
    statevector = _readonly_complex_vector(
        raw_statevector,
        expected_size=expected_size,
        name="statevector",
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
        raise AerGPUResultError(
            f"Statevector probability norm is {norm:.17g}, not 1."
        )

    result_metadata = {
        "framework": "CSSF",
        "algorithm": "CSNN-T^QAOA",
        "execution_engine": "qiskit_aer.AerSimulator",
        "method": REQUIRED_METHOD,
        "device": REQUIRED_DEVICE,
        "precision": run_config.precision,
        "seed_transpiler": run_config.seed_transpiler,
        "seed_simulator": run_config.seed_simulator,
        "probability_norm": norm,
    }

    backend_metadata = getattr(aer_result, "metadata", None)
    if isinstance(backend_metadata, Mapping):
        result_metadata["aer_metadata"] = dict(backend_metadata)

    return AerStatevectorResult(
        statevector=statevector,
        probabilities=probabilities,
        n_qubits=n_qubits,
        variable_order=artifact.plan.hamiltonian.variable_order,
        circuit_fingerprint=artifact.fingerprint(),
        config_fingerprint=run_config.fingerprint(),
        environment_fingerprint=environment.fingerprint(),
        transpiled_depth=_integer_metric(transpiled, "depth"),
        transpiled_size=_integer_metric(transpiled, "size"),
        metadata=result_metadata,
    )


__all__ = [
    "REAL_DTYPE",
    "COMPLEX_DTYPE",
    "REQUIRED_METHOD",
    "REQUIRED_DEVICE",
    "DEFAULT_OPTIMIZATION_LEVEL",
    "DEFAULT_SEED_TRANSPILER",
    "DEFAULT_SEED_SIMULATOR",
    "DEFAULT_NORM_TOLERANCE",
    "DEFAULT_ZERO_THRESHOLD",
    "DEFAULT_VALIDATION_THRESHOLD",
    "COLAB_FREE_STATEVECTOR_QUBIT_LIMIT",
    "DEFAULT_MAX_STATEVECTOR_QUBITS",
    "AerGPUError",
    "AerGPUUnavailableError",
    "AerGPUExecutionError",
    "AerGPUResultError",
    "AerGPUConfig",
    "AerGPUEnvironment",
    "AerStatevectorResult",
    "create_gpu_statevector_backend",
    "audit_aer_gpu_environment",
    "run_qaoa_statevector_gpu",
]
