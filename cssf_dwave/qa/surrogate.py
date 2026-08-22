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

"""Exact digitized-QA teacher targets and CSSF residual datasets.

This module converts one audited :class:`qa.observables.QAGPUEvaluation`
into the target vector declared by :class:`config.schema.QAConfig`.  It then
builds either a raw teacher dataset or the mandatory hierarchical residual
training dataset

    CSNN-T^MA-QAOA -> residual -> CSNN-T^digitized-QA.

The statevector itself is produced only by the strict Qiskit Aer GPU runtime.
NumPy is used here solely for deterministic algebra, probability aggregation,
validation, and dataset assembly.  The module performs no eager Qiskit, Aer,
Ocean, filesystem, network, or hardware operation during import.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config.schema import QAConfig, SurrogateTarget
from core.dataset import CSSFDataset
from core.types import SurrogateLevel
from qa import (
    COLAB_FREE_STATEVECTOR_QUBIT_LIMIT,
    QA_ALGORITHM_NAME,
    QISKIT_QUBIT_ORDER,
    validate_exact_statevector_qubit_count,
)
from qa.observables import QAGPUEvaluation
from qaoa.hamiltonian import IsingHamiltonian
from spectral.residual_surrogate import build_residual_dataset


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
DEFAULT_LOWER_TAIL_PROBABILITY: Final[float] = 0.05
DEFAULT_ENERGY_ATOL: Final[float] = 1.0e-9
DEFAULT_PROBABILITY_ATOL: Final[float] = 1.0e-12
DEFAULT_BASIS_CHUNK_SIZE: Final[int] = 65_536
SURROGATE_LEVEL: Final[SurrogateLevel] = SurrogateLevel.DIGITIZED_QA
BASELINE_LEVEL: Final[SurrogateLevel] = SurrogateLevel.MA_QAOA

DEFAULT_SURROGATE_TARGETS: Final[tuple[SurrogateTarget, ...]] = (
    SurrogateTarget.MEAN_ENERGY,
    SurrogateTarget.ENERGY_VARIANCE,
    SurrogateTarget.ENERGY_QUANTILE_05,
    SurrogateTarget.CVAR_05,
    SurrogateTarget.FEASIBILITY_PROBABILITY,
    SurrogateTarget.ELITE_PROBABILITY,
    SurrogateTarget.SUCCESS_PROBABILITY,
)


class QASurrogateError(ValueError):
    """Raised when exact QA targets or residual datasets are inconsistent."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise QASurrogateError(f"{name} must be finite.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized <= 0.0:
        raise QASurrogateError(f"{name} must be strictly positive.")
    return normalized


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise QASurrogateError(f"{name} must be strictly positive.")
    return value


def _probability(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if not 0.0 < normalized <= 1.0:
        raise QASurrogateError(f"{name} must lie in (0, 1].")
    return normalized


def _sha256_digest(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise QASurrogateError(f"{name} must be a SHA-256 digest.")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise QASurrogateError(
            f"{name} must be a hexadecimal SHA-256 digest."
        ) from exc
    return normalized


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
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _immutable_json_mapping(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    source = {} if metadata is None else dict(metadata)
    try:
        encoded = json.dumps(
            source,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise QASurrogateError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc
    frozen = _freeze_json(json.loads(encoded))
    if not isinstance(frozen, Mapping):
        raise QASurrogateError("metadata normalization failed.")
    return frozen


def _readonly_real_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_size: int | None = None,
) -> NDArray[np.float64]:
    result = np.array(
        np.asarray(values, dtype=REAL_DTYPE).reshape(-1),
        dtype=REAL_DTYPE,
        order="C",
        copy=True,
    )
    if result.size == 0:
        raise QASurrogateError(f"{name} must not be empty.")
    if expected_size is not None and result.size != expected_size:
        raise QASurrogateError(
            f"{name} must contain {expected_size} values; "
            f"received {result.size}."
        )
    if not np.all(np.isfinite(result)):
        raise QASurrogateError(f"{name} contains non-finite values.")
    result.setflags(write=False)
    return result


def _readonly_bool_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_size: int,
) -> NDArray[np.bool_]:
    source = np.asarray(values)
    if source.ndim != 1 or source.size != expected_size:
        raise QASurrogateError(
            f"{name} must be one-dimensional with {expected_size} entries."
        )
    if source.dtype.kind not in {"b", "i", "u", "f"}:
        raise QASurrogateError(f"{name} must contain boolean values.")
    numeric = np.asarray(source, dtype=REAL_DTYPE)
    if not np.all(np.isfinite(numeric)):
        raise QASurrogateError(f"{name} contains non-finite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise QASurrogateError(f"{name} must contain only 0/1 values.")
    result = np.array(numeric != 0.0, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


def _normalize_targets(
    targets: Sequence[SurrogateTarget | str],
) -> tuple[SurrogateTarget, ...]:
    normalized: list[SurrogateTarget] = []
    for value in targets:
        if isinstance(value, SurrogateTarget):
            target = value
        elif isinstance(value, str):
            try:
                target = SurrogateTarget(value.strip())
            except ValueError as exc:
                raise QASurrogateError(
                    f"Unsupported QA surrogate target: {value!r}."
                ) from exc
        else:
            raise TypeError(
                "targets must contain SurrogateTarget or str values."
            )
        normalized.append(target)

    result = tuple(normalized)
    if not result:
        raise QASurrogateError("At least one surrogate target is required.")
    if len(set(result)) != len(result):
        raise QASurrogateError("Surrogate targets must be unique.")
    return result


def targets_from_config(config: QAConfig) -> tuple[SurrogateTarget, ...]:
    """Return the validated target order declared by ``QAConfig``."""

    if not isinstance(config, QAConfig):
        raise TypeError("config must be QAConfig.")
    return _normalize_targets(config.targets)




def _basis_energies_qiskit_chunked(
    hamiltonian: IsingHamiltonian,
    *,
    chunk_size: int,
) -> NDArray[np.float64]:
    """Enumerate exact Ising energies in Qiskit order with bounded workspace."""

    size = 1 << hamiltonian.n_qubits
    result = np.empty(size, dtype=REAL_DTYPE)
    shifts = np.arange(hamiltonian.n_qubits, dtype=np.uint64)
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        indices = np.arange(start, stop, dtype=np.uint64)
        binary = (
            (indices[:, None] >> shifts[None, :]) & np.uint64(1)
        ).astype(REAL_DTYPE)
        spins = 1.0 - 2.0 * binary
        result[start:stop] = (
            hamiltonian.offset
            + spins @ hamiltonian.linear_z
            + np.einsum(
                "bi,ij,bj->b",
                spins,
                hamiltonian.quadratic_zz,
                spins,
                optimize=True,
            )
        )
    result.setflags(write=False)
    return result


def weighted_lower_quantile(
    values: ArrayLike,
    probabilities: ArrayLike,
    *,
    probability_mass: float = DEFAULT_LOWER_TAIL_PROBABILITY,
) -> float:
    """Return the smallest value whose cumulative probability reaches mass."""

    mass = _probability(probability_mass, name="probability_mass")
    value_array = _readonly_real_vector(values, name="values")
    probability_array = _readonly_real_vector(
        probabilities,
        name="probabilities",
        expected_size=value_array.size,
    )
    if np.any(probability_array < 0.0):
        raise QASurrogateError("probabilities must be non-negative.")
    total = float(np.sum(probability_array, dtype=np.float64))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise QASurrogateError(
            f"probabilities must sum to one; received {total:.17g}."
        )

    order = np.argsort(value_array, kind="stable")
    ordered_values = value_array[order]
    ordered_probabilities = probability_array[order]
    cumulative = np.cumsum(ordered_probabilities, dtype=np.float64)
    index = int(np.searchsorted(cumulative, mass, side="left"))
    index = min(index, ordered_values.size - 1)
    return float(ordered_values[index])


def weighted_lower_cvar(
    values: ArrayLike,
    probabilities: ArrayLike,
    *,
    probability_mass: float = DEFAULT_LOWER_TAIL_PROBABILITY,
) -> float:
    """Return the exact lower-tail CVaR with fractional cutoff mass."""

    mass = _probability(probability_mass, name="probability_mass")
    value_array = _readonly_real_vector(values, name="values")
    probability_array = _readonly_real_vector(
        probabilities,
        name="probabilities",
        expected_size=value_array.size,
    )
    if np.any(probability_array < 0.0):
        raise QASurrogateError("probabilities must be non-negative.")
    total = float(np.sum(probability_array, dtype=np.float64))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise QASurrogateError(
            f"probabilities must sum to one; received {total:.17g}."
        )

    order = np.argsort(value_array, kind="stable")
    ordered_values = value_array[order]
    ordered_probabilities = probability_array[order]

    remaining = mass
    weighted_sum = 0.0
    for value, probability in zip(ordered_values, ordered_probabilities):
        if remaining <= 0.0:
            break
        included = min(float(probability), remaining)
        weighted_sum += included * float(value)
        remaining -= included

    if remaining > 1.0e-10:
        raise QASurrogateError(
            "probabilities do not contain enough mass for CVaR."
        )
    return float(weighted_sum / mass)


@dataclass(frozen=True, slots=True)
class QASurrogateObservation:
    """One immutable exact teacher-target vector with ownership metadata."""

    targets: tuple[SurrogateTarget, ...]
    values: NDArray[np.float64]
    evaluation_fingerprint: str
    statevector_fingerprint: str
    hamiltonian_fingerprint: str
    schedule_fingerprint: str
    selection_fingerprint: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        targets = _normalize_targets(self.targets)
        values = _readonly_real_vector(
            self.values,
            name="values",
            expected_size=len(targets),
        )
        for target, value in zip(targets, values):
            if target in {
                SurrogateTarget.FEASIBILITY_PROBABILITY,
                SurrogateTarget.ELITE_PROBABILITY,
                SurrogateTarget.SUCCESS_PROBABILITY,
            } and not -DEFAULT_PROBABILITY_ATOL <= value <= (
                1.0 + DEFAULT_PROBABILITY_ATOL
            ):
                raise QASurrogateError(
                    f"{target.value} must be a probability in [0, 1]."
                )
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "values", values)
        for name in (
            "evaluation_fingerprint",
            "statevector_fingerprint",
            "hamiltonian_fingerprint",
            "schedule_fingerprint",
            "selection_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "metadata",
            _immutable_json_mapping(self.metadata),
        )

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(target.value for target in self.targets)

    def as_mapping(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                target.value: float(value)
                for target, value in zip(self.targets, self.values)
            }
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-QASurrogateObservation-v1\0")
        digest.update(
            json.dumps(
                self.target_names,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(self.values.tobytes(order="C"))
        for value in (
            self.evaluation_fingerprint,
            self.statevector_fingerprint,
            self.hamiltonian_fingerprint,
            self.schedule_fingerprint,
            self.selection_fingerprint,
        ):
            digest.update(value.encode("ascii"))
        digest.update(
            json.dumps(
                _thaw_json(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "cssf-digitized-qa-surrogate-observation-v1",
            "algorithm": QA_ALGORITHM_NAME,
            "surrogate_level": SURROGATE_LEVEL.value,
            "baseline_level": BASELINE_LEVEL.value,
            "qubit_order": QISKIT_QUBIT_ORDER,
            "target_names": list(self.target_names),
            "values": [float(value) for value in self.values],
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "statevector_fingerprint": self.statevector_fingerprint,
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "schedule_fingerprint": self.schedule_fingerprint,
            "selection_fingerprint": self.selection_fingerprint,
            "observation_fingerprint": self.fingerprint(),
            "metadata": _thaw_json(self.metadata),
        }


def _selection_fingerprint(
    *,
    feasible_mask: NDArray[np.bool_] | None,
    success_mask: NDArray[np.bool_],
    elite_energy_threshold: float,
    lower_tail_probability: float,
    ground_energy_atol: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"CSSF-QASurrogateSelection-v1\0")
    if feasible_mask is None:
        digest.update(b"feasible:none\0")
    else:
        digest.update(b"feasible:present\0")
        digest.update(feasible_mask.tobytes(order="C"))
    digest.update(success_mask.tobytes(order="C"))
    digest.update(
        np.asarray(
            [
                elite_energy_threshold,
                lower_tail_probability,
                ground_energy_atol,
            ],
            dtype=REAL_DTYPE,
        ).tobytes(order="C")
    )
    return digest.hexdigest()


def build_qa_surrogate_observation(
    evaluation: QAGPUEvaluation,
    hamiltonian: IsingHamiltonian,
    *,
    targets: Sequence[SurrogateTarget | str] = DEFAULT_SURROGATE_TARGETS,
    feasible_mask: ArrayLike | None = None,
    success_mask: ArrayLike | None = None,
    elite_energy_threshold: float | None = None,
    lower_tail_probability: float = DEFAULT_LOWER_TAIL_PROBABILITY,
    ground_energy_atol: float = DEFAULT_ENERGY_ATOL,
    energy_atol: float = DEFAULT_ENERGY_ATOL,
    basis_chunk_size: int = DEFAULT_BASIS_CHUNK_SIZE,
    metadata: Mapping[str, Any] | None = None,
) -> QASurrogateObservation:
    """Build one exact target vector from an audited QA GPU evaluation.

    ``feasible_mask`` follows Qiskit's little-endian statevector index order
    and is mandatory when feasibility probability is requested.  When no
    explicit success mask is supplied, success means probability of the exact
    ground-state manifold.  When no elite threshold is supplied, the lower
    weighted quantile at ``lower_tail_probability`` defines the elite set.
    """

    if not isinstance(evaluation, QAGPUEvaluation):
        raise TypeError("evaluation must be QAGPUEvaluation.")
    if not isinstance(hamiltonian, IsingHamiltonian):
        raise TypeError("hamiltonian must be IsingHamiltonian.")

    normalized_targets = _normalize_targets(targets)
    n_qubits = validate_exact_statevector_qubit_count(
        hamiltonian.n_qubits
    )
    statevector_result = evaluation.statevector_result
    observables_result = evaluation.observables_result
    if statevector_result.n_qubits != n_qubits:
        raise QASurrogateError(
            "Statevector and Hamiltonian qubit counts differ."
        )
    if statevector_result.variable_order != hamiltonian.variable_order:
        raise QASurrogateError(
            "Statevector and Hamiltonian variable orders differ."
        )

    hamiltonian_fingerprint = hamiltonian.fingerprint()
    if statevector_result.hamiltonian_fingerprint != hamiltonian_fingerprint:
        raise QASurrogateError(
            "Statevector belongs to another Hamiltonian."
        )
    if observables_result.hamiltonian_fingerprint != hamiltonian_fingerprint:
        raise QASurrogateError(
            "Observable result belongs to another Hamiltonian."
        )

    mass = _probability(
        lower_tail_probability,
        name="lower_tail_probability",
    )
    if mass != DEFAULT_LOWER_TAIL_PROBABILITY:
        raise QASurrogateError(
            "ENERGY_QUANTILE_05 and CVAR_05 require exactly 0.05 "
            "lower-tail probability mass."
        )
    normalized_ground_atol = _positive_float(
        ground_energy_atol,
        name="ground_energy_atol",
    )
    normalized_energy_atol = _positive_float(
        energy_atol,
        name="energy_atol",
    )
    normalized_chunk_size = _positive_integer(
        basis_chunk_size,
        name="basis_chunk_size",
    )

    probabilities = statevector_result.probabilities
    expected_size = 1 << n_qubits
    if probabilities.size != expected_size:
        raise QASurrogateError(
            "Statevector probability dimension is inconsistent."
        )
    energies = _basis_energies_qiskit_chunked(
        hamiltonian,
        chunk_size=normalized_chunk_size,
    )
    if energies.size != expected_size:
        raise QASurrogateError(
            "Hamiltonian basis dimension is inconsistent."
        )

    mean_energy = float(np.dot(probabilities, energies))
    centered = energies - mean_energy
    energy_variance = float(np.dot(probabilities, centered * centered))
    if energy_variance < 0.0 and abs(energy_variance) <= normalized_energy_atol:
        energy_variance = 0.0

    if not math.isclose(
        mean_energy,
        observables_result.expected_energy,
        rel_tol=0.0,
        abs_tol=normalized_energy_atol,
    ):
        raise QASurrogateError(
            "Recomputed mean energy disagrees with QA observables."
        )
    if not math.isclose(
        energy_variance,
        observables_result.variance,
        rel_tol=0.0,
        abs_tol=normalized_energy_atol,
    ):
        raise QASurrogateError(
            "Recomputed energy variance disagrees with QA observables."
        )

    energy_quantile = weighted_lower_quantile(
        energies,
        probabilities,
        probability_mass=mass,
    )
    cvar = weighted_lower_cvar(
        energies,
        probabilities,
        probability_mass=mass,
    )

    normalized_feasible_mask: NDArray[np.bool_] | None = None
    if feasible_mask is not None:
        normalized_feasible_mask = _readonly_bool_vector(
            feasible_mask,
            name="feasible_mask",
            expected_size=expected_size,
        )
    if (
        SurrogateTarget.FEASIBILITY_PROBABILITY in normalized_targets
        and normalized_feasible_mask is None
    ):
        raise QASurrogateError(
            "feasible_mask is required for feasibility_probability."
        )

    if elite_energy_threshold is None:
        elite_threshold = energy_quantile
    else:
        elite_threshold = _finite_float(
            elite_energy_threshold,
            name="elite_energy_threshold",
        )
    elite_mask = energies <= elite_threshold + normalized_energy_atol

    if success_mask is None:
        ground_energy = float(np.min(energies))
        normalized_success_mask = np.asarray(
            energies <= ground_energy + normalized_ground_atol,
            dtype=np.bool_,
        )
        normalized_success_mask.setflags(write=False)
        success_semantics = "ground_state_manifold"
    else:
        normalized_success_mask = _readonly_bool_vector(
            success_mask,
            name="success_mask",
            expected_size=expected_size,
        )
        success_semantics = "explicit_success_mask"

    feasibility_probability = (
        float(np.sum(probabilities[normalized_feasible_mask]))
        if normalized_feasible_mask is not None
        else math.nan
    )
    elite_probability = float(np.sum(probabilities[elite_mask]))
    success_probability = float(
        np.sum(probabilities[normalized_success_mask])
    )
    if normalized_feasible_mask is not None:
        feasibility_probability = float(
            np.clip(feasibility_probability, 0.0, 1.0)
        )
    elite_probability = float(np.clip(elite_probability, 0.0, 1.0))
    success_probability = float(np.clip(success_probability, 0.0, 1.0))

    target_values: dict[SurrogateTarget, float] = {
        SurrogateTarget.MEAN_ENERGY: mean_energy,
        SurrogateTarget.ENERGY_VARIANCE: energy_variance,
        SurrogateTarget.ENERGY_QUANTILE_05: energy_quantile,
        SurrogateTarget.CVAR_05: cvar,
        SurrogateTarget.FEASIBILITY_PROBABILITY: (
            feasibility_probability
        ),
        SurrogateTarget.ELITE_PROBABILITY: elite_probability,
        SurrogateTarget.SUCCESS_PROBABILITY: success_probability,
    }
    values = np.asarray(
        [target_values[target] for target in normalized_targets],
        dtype=REAL_DTYPE,
    )

    selection_fingerprint = _selection_fingerprint(
        feasible_mask=normalized_feasible_mask,
        success_mask=normalized_success_mask,
        elite_energy_threshold=elite_threshold,
        lower_tail_probability=mass,
        ground_energy_atol=normalized_ground_atol,
    )

    merged_metadata = {
        **({} if metadata is None else dict(metadata)),
        "algorithm": QA_ALGORITHM_NAME,
        "surrogate_level": SURROGATE_LEVEL.value,
        "baseline_level": BASELINE_LEVEL.value,
        "target_semantics": "exact_statevector_teacher_observables",
        "qubit_order": QISKIT_QUBIT_ORDER,
        "n_qubits": n_qubits,
        "lower_tail_probability": mass,
        "elite_energy_threshold": elite_threshold,
        "success_semantics": success_semantics,
        "feasibility_mask_supplied": normalized_feasible_mask is not None,
        "basis_chunk_size": normalized_chunk_size,
    }

    return QASurrogateObservation(
        targets=normalized_targets,
        values=values,
        evaluation_fingerprint=evaluation.fingerprint(),
        statevector_fingerprint=statevector_result.fingerprint(),
        hamiltonian_fingerprint=hamiltonian_fingerprint,
        schedule_fingerprint=statevector_result.schedule_fingerprint,
        selection_fingerprint=selection_fingerprint,
        metadata=merged_metadata,
    )


def _normalize_observations(
    observations: Sequence[QASurrogateObservation],
) -> tuple[QASurrogateObservation, ...]:
    normalized = tuple(observations)
    if not normalized:
        raise QASurrogateError("observations must not be empty.")
    if any(
        not isinstance(observation, QASurrogateObservation)
        for observation in normalized
    ):
        raise TypeError(
            "observations must contain QASurrogateObservation values."
        )
    reference_targets = normalized[0].targets
    if any(
        observation.targets != reference_targets
        for observation in normalized[1:]
    ):
        raise QASurrogateError(
            "All observations must use the same ordered target set."
        )
    fingerprints = tuple(
        observation.fingerprint() for observation in normalized
    )
    if len(set(fingerprints)) != len(fingerprints):
        raise QASurrogateError(
            "observations must not contain duplicate fingerprints."
        )
    return normalized


def qa_target_matrix(
    observations: Sequence[QASurrogateObservation],
) -> NDArray[np.float64]:
    """Return an immutable matrix in the shared observation target order."""

    normalized = _normalize_observations(observations)
    result = np.array(
        np.vstack([observation.values for observation in normalized]),
        dtype=REAL_DTYPE,
        order="C",
        copy=True,
    )
    result.setflags(write=False)
    return result


def _dataset_metadata(
    observations: tuple[QASurrogateObservation, ...],
    *,
    target_semantics: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        **({} if metadata is None else dict(metadata)),
        "algorithm": QA_ALGORITHM_NAME,
        "surrogate_level": SURROGATE_LEVEL.value,
        "baseline_level": BASELINE_LEVEL.value,
        "target_semantics": target_semantics,
        "target_names": list(observations[0].target_names),
        "observation_fingerprints": [
            observation.fingerprint() for observation in observations
        ],
        "evaluation_fingerprints": [
            observation.evaluation_fingerprint
            for observation in observations
        ],
        "hamiltonian_fingerprints": [
            observation.hamiltonian_fingerprint
            for observation in observations
        ],
        "schedule_fingerprints": [
            observation.schedule_fingerprint
            for observation in observations
        ],
    }


def build_qa_teacher_dataset(
    features: ArrayLike,
    observations: Sequence[QASurrogateObservation],
    *,
    sample_ids: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CSSFDataset:
    """Build a raw exact digitized-QA teacher dataset for audits/ablation."""

    normalized = _normalize_observations(observations)
    return CSSFDataset(
        features,
        qa_target_matrix(normalized),
        sample_ids=sample_ids,
        metadata=_dataset_metadata(
            normalized,
            target_semantics="exact_digitized_qa_teacher",
            metadata=metadata,
        ),
    )


def build_qa_surrogate_dataset(
    features: ArrayLike,
    observations: Sequence[QASurrogateObservation],
    accumulated_maqaoa_prediction: ArrayLike,
    *,
    sample_ids: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CSSFDataset:
    """Build the mandatory MA-QAOA -> digitized-QA residual dataset."""

    normalized = _normalize_observations(observations)
    reference_targets = qa_target_matrix(normalized)
    return build_residual_dataset(
        features,
        reference_targets,
        accumulated_maqaoa_prediction,
        level=SURROGATE_LEVEL,
        baseline_level=BASELINE_LEVEL,
        sample_ids=sample_ids,
        metadata=_dataset_metadata(
            normalized,
            target_semantics="additive_digitized_qa_residual",
            metadata=metadata,
        ),
    )


build_qa_residual_dataset = build_qa_surrogate_dataset


__all__ = [
    "REAL_DTYPE",
    "DEFAULT_LOWER_TAIL_PROBABILITY",
    "DEFAULT_ENERGY_ATOL",
    "DEFAULT_PROBABILITY_ATOL",
    "DEFAULT_BASIS_CHUNK_SIZE",
    "SURROGATE_LEVEL",
    "BASELINE_LEVEL",
    "DEFAULT_SURROGATE_TARGETS",
    "QASurrogateError",
    "QASurrogateObservation",
    "targets_from_config",
    "weighted_lower_quantile",
    "weighted_lower_cvar",
    "build_qa_surrogate_observation",
    "qa_target_matrix",
    "build_qa_teacher_dataset",
    "build_qa_surrogate_dataset",
    "build_qa_residual_dataset",
]
