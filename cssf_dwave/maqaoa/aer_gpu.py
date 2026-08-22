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

"""Strict Qiskit Aer GPU execution for exact MA-QAOA statevectors.

The module accepts a materialized :class:`MAQAOACircuitArtifact`, binds the
independent cost and mixer angles, transpiles with Qiskit, and executes only
``qiskit_aer.AerSimulator(method="statevector", device="GPU")``.

No CPU fallback, custom statevector engine, or NumPy circuit simulation is
implemented. NumPy is used only for result validation, immutable data
transforms, probability extraction, deterministic sampling diagnostics, and
fingerprinting.

Free Google Colab exact-statevector execution is hard-limited to 22 qubits.
The limit cannot be raised through configuration.
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

from maqaoa import (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    MAQAOA_ALGORITHM_NAME,
    STATEVECTOR_BACKEND_CLASS,
    STATEVECTOR_DEVICE,
    STATEVECTOR_METHOD,
    STATEVECTOR_PRECISION,
)
from maqaoa.circuit import (
    MAQAOACircuitArtifact,
    MAQAOACircuitError,
)
from maqaoa.parameters import (
    MAQAOAParameterError,
    MAQAOAParameterValues,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
COMPLEX_DTYPE: Final[np.dtype[np.complex128]] = np.dtype(np.complex128)

REQUIRED_METHOD: Final[str] = STATEVECTOR_METHOD
REQUIRED_DEVICE: Final[str] = STATEVECTOR_DEVICE
REQUIRED_PRECISION: Final[str] = STATEVECTOR_PRECISION

DEFAULT_OPTIMIZATION_LEVEL: Final[int] = 1
DEFAULT_SEED_TRANSPILER: Final[int] = 271828
DEFAULT_SEED_SIMULATOR: Final[int] = 314159
DEFAULT_NORM_TOLERANCE: Final[float] = 1.0e-10
DEFAULT_ZERO_THRESHOLD: Final[float] = 1.0e-14
DEFAULT_VALIDATION_THRESHOLD: Final[float] = 1.0e-10
DEFAULT_MAX_STATEVECTOR_QUBITS: Final[int] = (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
)
DEFAULT_FUSION_THRESHOLD: Final[int] = 14


class MAQAOAAerGPUError(RuntimeError):
    """Base error for strict MA-QAOA Qiskit Aer GPU execution."""


class MAQAOAAerGPUUnavailableError(MAQAOAAerGPUError):
    """Raised when the required Qiskit Aer GPU path is unavailable."""


class MAQAOAAerGPUExecutionError(MAQAOAAerGPUError):
    """Raised when parameter binding, transpilation, or execution fails."""


class MAQAOAAerGPUResultError(MAQAOAAerGPUError):
    """Raised when Qiskit Aer returns an invalid statevector result."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MAQAOAAerGPUError(f"{name} must be finite.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized <= 0.0:
        raise MAQAOAAerGPUError(f"{name} must be strictly positive.")
    return normalized


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise MAQAOAAerGPUError(f"{name} must be non-negative.")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    normalized = _nonnegative_integer(value, name=name)
    if normalized == 0:
        raise MAQAOAAerGPUError(f"{name} must be positive.")
    return normalized


def _sha256_digest(value: str, *, name: str) -> str:
    normalized = str(value)
    if len(normalized) != 64:
        raise MAQAOAAerGPUError(
            f"{name} must be a 64-character SHA-256 digest."
        )
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise MAQAOAAerGPUError(
            f"{name} must contain hexadecimal SHA-256 data."
        ) from exc
    return normalized


def _readonly_complex_vector(
    values: ArrayLike,
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=COMPLEX_DTYPE).reshape(-1)
    if array.size != expected_size:
        raise MAQAOAAerGPUResultError(
            f"{name} contains {array.size} amplitudes; "
            f"expected {expected_size}."
        )
    if not np.all(np.isfinite(array.real)) or not np.all(
        np.isfinite(array.imag)
    ):
        raise MAQAOAAerGPUResultError(
            f"{name} contains non-finite amplitudes."
        )
    result = np.array(
        array,
        dtype=COMPLEX_DTYPE,
        order="C",
        copy=True,
    )
    result.setflags(write=False)
    return result


def _readonly_probability_vector(
    values: ArrayLike,
    *,
    expected_size: int,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=REAL_DTYPE).reshape(-1)
    if array.size != expected_size:
        raise MAQAOAAerGPUResultError(
            "Probability-vector size does not match the statevector."
        )
    if not np.all(np.isfinite(array)):
        raise MAQAOAAerGPUResultError(
            "Probabilities contain non-finite values."
        )
    if np.any(array < 0.0):
        raise MAQAOAAerGPUResultError(
            "Probabilities contain negative values."
        )
    result = np.array(
        array,
        dtype=REAL_DTYPE,
        order="C",
        copy=True,
    )
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _json_mapping(
    metadata: Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    try:
        encoded = json.dumps(
            dict(metadata),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise MAQAOAAerGPUResultError(
            f"{name} must be JSON-serializable."
        ) from exc
    return MappingProxyType(json.loads(encoded))


def _call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def _backend_sequence(backend: Any, name: str) -> tuple[str, ...]:
    attribute = getattr(backend, name, None)
    if attribute is None:
        return tuple()
    try:
        raw = _call_or_value(attribute)
    except Exception as exc:
        raise MAQAOAAerGPUUnavailableError(
            f"Cannot query Aer backend {name}."
        ) from exc
    return tuple(str(item) for item in raw)


@dataclass(frozen=True, slots=True)
class MAQAOAAerGPUConfig:
    """Validated strict configuration for exact MA-QAOA Aer GPU runs."""

    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER
    seed_simulator: int = DEFAULT_SEED_SIMULATOR
    precision: str = REQUIRED_PRECISION
    zero_threshold: float = DEFAULT_ZERO_THRESHOLD
    validation_threshold: float = DEFAULT_VALIDATION_THRESHOLD
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE
    max_statevector_qubits: int = DEFAULT_MAX_STATEVECTOR_QUBITS
    fusion_enable: bool = True
    fusion_threshold: int = DEFAULT_FUSION_THRESHOLD

    def __post_init__(self) -> None:
        optimization_level = _nonnegative_integer(
            self.optimization_level,
            name="optimization_level",
        )
        if optimization_level > 3:
            raise MAQAOAAerGPUError(
                "optimization_level must lie in [0, 3]."
            )

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
            raise MAQAOAAerGPUError(
                f"max_statevector_qubits={maximum_qubits} exceeds the "
                f"hard free Google Colab statevector limit of "
                f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}."
            )

        fusion_threshold = _nonnegative_integer(
            self.fusion_threshold,
            name="fusion_threshold",
        )
        precision = str(self.precision).strip().lower()
        if precision != REQUIRED_PRECISION:
            raise MAQAOAAerGPUError(
                "Scientific production execution requires double "
                "precision."
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
            _positive_float(
                self.zero_threshold,
                name="zero_threshold",
            ),
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
            _positive_float(
                self.norm_tolerance,
                name="norm_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "max_statevector_qubits",
            maximum_qubits,
        )
        object.__setattr__(
            self,
            "fusion_threshold",
            fusion_threshold,
        )

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
        """Return deterministic exact-statevector job options."""

        return {
            "shots": None,
            "seed_simulator": self.seed_simulator,
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAAerGPUConfig-v1\0")
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
                    "max_statevector_qubits": (
                        self.max_statevector_qubits
                    ),
                    "hard_statevector_qubit_limit": (
                        COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
                    ),
                    "fusion_enable": self.fusion_enable,
                    "fusion_threshold": self.fusion_threshold,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MAQAOAAerGPUEnvironment:
    """Audited Qiskit and Qiskit Aer GPU runtime capabilities."""

    qiskit_version: str
    qiskit_aer_version: str
    backend_name: str
    available_devices: tuple[str, ...]
    available_methods: tuple[str, ...]
    config_fingerprint: str

    def __post_init__(self) -> None:
        qiskit_version = str(self.qiskit_version).strip()
        aer_version = str(self.qiskit_aer_version).strip()
        backend_name = str(self.backend_name).strip()
        if not qiskit_version:
            raise MAQAOAAerGPUError(
                "qiskit_version must be non-empty."
            )
        if not aer_version:
            raise MAQAOAAerGPUError(
                "qiskit_aer_version must be non-empty."
            )
        if not backend_name:
            raise MAQAOAAerGPUError(
                "backend_name must be non-empty."
            )

        devices = tuple(str(item) for item in self.available_devices)
        methods = tuple(str(item) for item in self.available_methods)
        if REQUIRED_DEVICE not in devices:
            raise MAQAOAAerGPUUnavailableError(
                f"Aer devices {devices!r} do not include GPU."
            )
        if methods and REQUIRED_METHOD not in methods:
            raise MAQAOAAerGPUUnavailableError(
                f"Aer methods {methods!r} do not include statevector."
            )

        object.__setattr__(self, "qiskit_version", qiskit_version)
        object.__setattr__(self, "qiskit_aer_version", aer_version)
        object.__setattr__(self, "backend_name", backend_name)
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

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAAerGPUEnvironment-v1\0")
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
class MAQAOAStatevectorResult:
    """Immutable exact MA-QAOA statevector returned by Qiskit Aer GPU."""

    statevector: NDArray[np.complex128]
    probabilities: NDArray[np.float64]
    n_qubits: int
    variable_order: tuple[str, ...]
    parameter_fingerprint: str
    circuit_fingerprint: str
    config_fingerprint: str
    environment_fingerprint: str
    transpiled_depth: int
    transpiled_size: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        n_qubits = _positive_integer(self.n_qubits, name="n_qubits")
        if n_qubits > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT:
            raise MAQAOAAerGPUResultError(
                f"n_qubits={n_qubits} exceeds the hard statevector limit "
                f"of {COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}."
            )
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
        variable_order = tuple(str(item) for item in self.variable_order)
        if len(variable_order) != n_qubits:
            raise MAQAOAAerGPUResultError(
                "variable_order length must equal n_qubits."
            )
        if any(not item for item in variable_order):
            raise MAQAOAAerGPUResultError(
                "variable_order must contain non-empty labels."
            )
        if len(set(variable_order)) != len(variable_order):
            raise MAQAOAAerGPUResultError(
                "variable_order must contain unique labels."
            )

        object.__setattr__(self, "statevector", statevector)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "variable_order", variable_order)
        object.__setattr__(
            self,
            "parameter_fingerprint",
            _sha256_digest(
                self.parameter_fingerprint,
                name="parameter_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "circuit_fingerprint",
            _sha256_digest(
                self.circuit_fingerprint,
                name="circuit_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "config_fingerprint",
            _sha256_digest(
                self.config_fingerprint,
                name="config_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "environment_fingerprint",
            _sha256_digest(
                self.environment_fingerprint,
                name="environment_fingerprint",
            ),
        )
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
            _json_mapping(self.metadata, name="metadata"),
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

        index = _nonnegative_integer(
            basis_index,
            name="basis_index",
        )
        if index >= self.probabilities.size:
            raise MAQAOAAerGPUResultError(
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
        """Return the most probable binary sample in variable order."""

        return self.binary_sample(self.most_probable_index)

    def labeled_sample(self, basis_index: int) -> Mapping[str, int]:
        """Return an immutable variable-to-bit mapping."""

        sample = self.binary_sample(basis_index)
        return MappingProxyType(
            {
                variable: int(value)
                for variable, value in zip(
                    self.variable_order,
                    sample,
                    strict=True,
                )
            }
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-MAQAOAStatevectorResult-v1\0")
        digest.update(self.statevector.tobytes(order="C"))
        digest.update(self.probabilities.tobytes(order="C"))
        digest.update(self.parameter_fingerprint.encode("ascii"))
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
        raise MAQAOAAerGPUUnavailableError(
            "Qiskit is unavailable in the Google Colab runtime."
        ) from exc

    transpile = getattr(qiskit, "transpile", None)
    if transpile is None or not callable(transpile):
        raise MAQAOAAerGPUUnavailableError(
            "qiskit.transpile is unavailable."
        )

    try:
        qiskit_aer = importlib.import_module("qiskit_aer")
    except ImportError as exc:
        raise MAQAOAAerGPUUnavailableError(
            "qiskit-aer-gpu is unavailable in the Google Colab runtime."
        ) from exc

    simulator_class = getattr(qiskit_aer, "AerSimulator", None)
    if simulator_class is None or not callable(simulator_class):
        raise MAQAOAAerGPUUnavailableError(
            "qiskit_aer.AerSimulator is unavailable."
        )

    return (
        transpile,
        simulator_class,
        str(getattr(qiskit, "__version__", "unknown")),
        str(getattr(qiskit_aer, "__version__", "unknown")),
    )


def create_maqaoa_gpu_statevector_backend(
    config: MAQAOAAerGPUConfig | None = None,
) -> tuple[Any, MAQAOAAerGPUEnvironment]:
    """Create and audit a strict Qiskit Aer statevector GPU backend."""

    run_config = MAQAOAAerGPUConfig() if config is None else config
    if not isinstance(run_config, MAQAOAAerGPUConfig):
        raise TypeError(
            "config must be MAQAOAAerGPUConfig or None."
        )

    _, simulator_class, qiskit_version, aer_version = (
        _load_qiskit_aer_api()
    )
    try:
        backend = simulator_class(**run_config.backend_options())
    except Exception as exc:
        raise MAQAOAAerGPUUnavailableError(
            "Cannot initialize AerSimulator with statevector GPU."
        ) from exc

    devices = _backend_sequence(backend, "available_devices")
    methods = _backend_sequence(backend, "available_methods")
    if REQUIRED_DEVICE not in devices:
        raise MAQAOAAerGPUUnavailableError(
            f"Aer GPU is unavailable; reported devices={devices!r}. "
            "CPU fallback is forbidden."
        )
    if methods and REQUIRED_METHOD not in methods:
        raise MAQAOAAerGPUUnavailableError(
            "Aer statevector method is unavailable."
        )

    name_value = getattr(backend, "name", None)
    backend_name = str(
        _call_or_value(name_value)
        if name_value is not None
        else type(backend).__name__
    )
    environment = MAQAOAAerGPUEnvironment(
        qiskit_version=qiskit_version,
        qiskit_aer_version=aer_version,
        backend_name=backend_name,
        available_devices=devices,
        available_methods=methods,
        config_fingerprint=run_config.fingerprint(),
    )
    return backend, environment


def audit_maqaoa_aer_gpu_environment(
    config: MAQAOAAerGPUConfig | None = None,
) -> MAQAOAAerGPUEnvironment:
    """Validate the strict Qiskit Aer GPU path without executing a circuit."""

    _, environment = create_maqaoa_gpu_statevector_backend(config)
    return environment


def _require_bound_circuit(circuit: Any) -> None:
    parameters = getattr(circuit, "parameters", None)
    if parameters is None:
        return
    try:
        remaining = tuple(parameters)
    except TypeError as exc:
        raise MAQAOAAerGPUExecutionError(
            "Cannot inspect bound-circuit parameters."
        ) from exc
    if remaining:
        raise MAQAOAAerGPUExecutionError(
            f"Circuit still contains {len(remaining)} unbound parameters."
        )


def _save_statevector(circuit: Any) -> None:
    save_statevector = getattr(circuit, "save_statevector", None)
    if save_statevector is None or not callable(save_statevector):
        raise MAQAOAAerGPUExecutionError(
            "Qiskit Aer save_statevector instruction is unavailable."
        )
    try:
        save_statevector()
    except Exception as exc:
        raise MAQAOAAerGPUExecutionError(
            "Cannot append the Qiskit Aer save_statevector instruction."
        ) from exc


def _integer_metric(circuit: Any, name: str) -> int:
    value = getattr(circuit, name, None)
    if value is None:
        return 0
    try:
        metric = _call_or_value(value)
    except Exception as exc:
        raise MAQAOAAerGPUResultError(
            f"Cannot read transpiled circuit {name}."
        ) from exc
    return _nonnegative_integer(
        int(metric),
        name=f"transpiled_{name}",
    )


def _normalize_parameter_values(
    artifact: MAQAOACircuitArtifact,
    parameters: MAQAOAParameterValues | ArrayLike,
) -> MAQAOAParameterValues:
    try:
        if isinstance(parameters, MAQAOAParameterValues):
            return artifact.plan.parameter_layout.values(
                gamma=parameters.gamma,
                beta=parameters.beta,
            )
        return artifact.plan.parameter_layout.split(parameters)
    except MAQAOAParameterError as exc:
        raise MAQAOAAerGPUExecutionError(
            f"Invalid MA-QAOA parameter values: {exc}"
        ) from exc


def run_maqaoa_statevector_gpu(
    artifact: MAQAOACircuitArtifact,
    parameters: MAQAOAParameterValues | ArrayLike,
    config: MAQAOAAerGPUConfig | None = None,
) -> MAQAOAStatevectorResult:
    """Bind, transpile, and execute one MA-QAOA circuit on Aer GPU."""

    if not isinstance(artifact, MAQAOACircuitArtifact):
        raise TypeError("artifact must be MAQAOACircuitArtifact.")

    run_config = MAQAOAAerGPUConfig() if config is None else config
    if not isinstance(run_config, MAQAOAAerGPUConfig):
        raise TypeError(
            "config must be MAQAOAAerGPUConfig or None."
        )

    n_qubits = artifact.plan.n_qubits
    if n_qubits > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT:
        raise MAQAOAAerGPUExecutionError(
            f"Circuit has {n_qubits} qubits; the hard free Google Colab "
            f"statevector limit is "
            f"{COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}."
        )
    if n_qubits > run_config.max_statevector_qubits:
        raise MAQAOAAerGPUExecutionError(
            f"Circuit has {n_qubits} qubits; configured statevector limit "
            f"is {run_config.max_statevector_qubits}."
        )

    normalized_parameters = _normalize_parameter_values(
        artifact,
        parameters,
    )
    try:
        bound_circuit = artifact.bind(normalized_parameters)
    except (MAQAOACircuitError, MAQAOAParameterError) as exc:
        raise MAQAOAAerGPUExecutionError(
            f"MA-QAOA parameter binding failed: {exc}"
        ) from exc

    _require_bound_circuit(bound_circuit)

    transpile, _, _, _ = _load_qiskit_aer_api()
    backend, environment = create_maqaoa_gpu_statevector_backend(
        run_config
    )
    _save_statevector(bound_circuit)

    try:
        transpiled = transpile(
            bound_circuit,
            backend=backend,
            optimization_level=run_config.optimization_level,
            seed_transpiler=run_config.seed_transpiler,
        )
    except Exception as exc:
        raise MAQAOAAerGPUExecutionError(
            "Qiskit transpilation for MA-QAOA Aer GPU failed."
        ) from exc

    _require_bound_circuit(transpiled)
    try:
        job = backend.run(
            transpiled,
            **run_config.run_options(),
        )
        aer_result = job.result()
    except Exception as exc:
        raise MAQAOAAerGPUExecutionError(
            "Qiskit Aer GPU statevector job failed."
        ) from exc

    success = getattr(aer_result, "success", True)
    if not bool(success):
        status = getattr(aer_result, "status", "unknown")
        raise MAQAOAAerGPUExecutionError(
            f"Qiskit Aer reported an unsuccessful job: {status}."
        )

    get_statevector = getattr(aer_result, "get_statevector", None)
    if get_statevector is None or not callable(get_statevector):
        raise MAQAOAAerGPUResultError(
            "Aer result lacks get_statevector()."
        )
    try:
        raw_statevector = get_statevector(transpiled)
    except Exception as exc:
        raise MAQAOAAerGPUResultError(
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
        raise MAQAOAAerGPUResultError(
            f"Statevector probability norm is {norm:.17g}, not 1."
        )

    result_metadata: dict[str, Any] = {
        "framework": "CSSF",
        "algorithm": MAQAOA_ALGORITHM_NAME,
        "execution_engine": STATEVECTOR_BACKEND_CLASS,
        "method": REQUIRED_METHOD,
        "device": REQUIRED_DEVICE,
        "precision": run_config.precision,
        "seed_transpiler": run_config.seed_transpiler,
        "seed_simulator": run_config.seed_simulator,
        "probability_norm": norm,
        "n_qubits": n_qubits,
        "repetitions": artifact.plan.config.repetitions,
        "cost_term_count": (
            artifact.plan.parameter_layout.cost_term_count
        ),
        "parameter_count": artifact.plan.parameter_count,
        "hard_statevector_qubit_limit": (
            COLAB_FREE_STATEVECTOR_QUBIT_LIMIT
        ),
        "cpu_fallback": False,
    }
    backend_metadata = getattr(aer_result, "metadata", None)
    if isinstance(backend_metadata, Mapping):
        result_metadata["aer_metadata"] = dict(backend_metadata)

    return MAQAOAStatevectorResult(
        statevector=statevector,
        probabilities=probabilities,
        n_qubits=n_qubits,
        variable_order=artifact.plan.hamiltonian.variable_order,
        parameter_fingerprint=normalized_parameters.fingerprint(),
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
    "REQUIRED_PRECISION",
    "DEFAULT_OPTIMIZATION_LEVEL",
    "DEFAULT_SEED_TRANSPILER",
    "DEFAULT_SEED_SIMULATOR",
    "DEFAULT_NORM_TOLERANCE",
    "DEFAULT_ZERO_THRESHOLD",
    "DEFAULT_VALIDATION_THRESHOLD",
    "DEFAULT_MAX_STATEVECTOR_QUBITS",
    "DEFAULT_FUSION_THRESHOLD",
    "MAQAOAAerGPUError",
    "MAQAOAAerGPUUnavailableError",
    "MAQAOAAerGPUExecutionError",
    "MAQAOAAerGPUResultError",
    "MAQAOAAerGPUConfig",
    "MAQAOAAerGPUEnvironment",
    "MAQAOAStatevectorResult",
    "create_maqaoa_gpu_statevector_backend",
    "audit_maqaoa_aer_gpu_environment",
    "run_maqaoa_statevector_gpu",
]
