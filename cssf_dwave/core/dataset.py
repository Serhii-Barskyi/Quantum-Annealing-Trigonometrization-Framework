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

"""Validated immutable datasets for all CSSF surrogate levels.

The frozen CSNN-T implementation consumes the historical ``BESSDataset``
contract while newer project modules use the generic ``CSSFDataset`` contract.
This module provides both contracts without changing ``core/gcv.py`` or
``core/csnn_t.py``.

Every scientific array is copied into an owning C-contiguous NumPy buffer,
validated for finite values, and marked read-only. Metadata is recursively
validated as JSON data and exposed through immutable mappings and tuples.
Fingerprints are deterministic SHA-256 digests over complete dataset content.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray


COMPLEX_DTYPE: Final[np.dtype[np.complex128]] = np.dtype(np.complex128)
REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)


class DatasetError(ValueError):
    """Raised when a CSSF dataset violates the mathematical contract."""


def _as_complex_matrix(values: ArrayLike, *, name: str) -> NDArray[np.complex128]:
    """Return a finite, owning, contiguous, read-only complex matrix."""

    array = np.asarray(values)

    if array.ndim != 2:
        raise DatasetError(
            f"{name} must be two-dimensional; received shape {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DatasetError(
            f"{name} must contain at least one sample and one feature."
        )

    try:
        result = np.array(array, dtype=COMPLEX_DTYPE, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetError(f"{name} must be a complex numeric matrix.") from exc

    if not np.all(np.isfinite(result.real)):
        raise DatasetError(f"{name} contains non-finite real components.")
    if not np.all(np.isfinite(result.imag)):
        raise DatasetError(f"{name} contains non-finite imaginary components.")

    result.setflags(write=False)
    return result


def _as_real_target_matrix(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return a finite, owning, contiguous, read-only real target matrix."""

    array = np.asarray(values)

    if np.iscomplexobj(array):
        imaginary = np.asarray(array.imag)
        if np.any(imaginary != 0.0):
            raise DatasetError(
                f"{name} must be real-valued; non-zero imaginary values found."
            )
        array = array.real

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise DatasetError(
            f"{name} must be one- or two-dimensional; "
            f"received shape {array.shape}."
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DatasetError(
            f"{name} must contain at least one sample and one target."
        )

    try:
        result = np.array(array, dtype=REAL_DTYPE, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetError(f"{name} must be a real numeric matrix.") from exc

    if not np.all(np.isfinite(result)):
        raise DatasetError(f"{name} contains non-finite values.")

    result.setflags(write=False)
    return result


def _normalize_sample_ids(
    sample_ids: Sequence[str] | None,
    *,
    sample_count: int,
) -> tuple[str, ...]:
    """Validate sample identifiers or create deterministic defaults."""

    if sample_ids is None:
        return tuple(f"sample_{index:08d}" for index in range(sample_count))
    if isinstance(sample_ids, (str, bytes, bytearray)):
        raise DatasetError("sample_ids must be a sequence of identifiers.")

    normalized = tuple(str(value).strip() for value in sample_ids)

    if len(normalized) != sample_count:
        raise DatasetError(
            "sample_ids length must equal the number of samples: "
            f"{len(normalized)} != {sample_count}."
        )
    if any(not value for value in normalized):
        raise DatasetError("sample_ids must not contain empty identifiers.")
    if len(set(normalized)) != len(normalized):
        raise DatasetError("sample_ids must be unique.")

    return normalized


def _freeze_json(value: object, *, path: str = "metadata") -> object:
    """Validate and recursively freeze one JSON-compatible value."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetError(f"{path} contains NaN or infinity.")
        return float(value)
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise DatasetError(
                    f"{path} keys must be non-empty strings."
                )
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise DatasetError(
        f"{path} contains unsupported value type {type(value).__name__}."
    )


def _normalize_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Return deeply immutable, detached, JSON-compatible metadata."""

    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise DatasetError("metadata must be a mapping or None.")

    frozen = _freeze_json(metadata)
    if not isinstance(frozen, Mapping):  # pragma: no cover - defensive branch
        raise DatasetError("metadata must normalize to a mapping.")
    return frozen


def _thaw_json(value: object) -> object:
    """Return a plain JSON-compatible copy of recursively frozen data."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _thaw_json(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _update_array_digest(
    digest: Any,
    *,
    name: str,
    array: NDArray[Any] | None,
) -> None:
    """Append a stable typed array representation to a SHA-256 digest."""

    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    if array is None:
        digest.update(b"NONE\0")
        return
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    digest.update(b"\0")


def _validate_array_ownership(name: str, array: NDArray[Any]) -> None:
    """Require an independent immutable C-contiguous NumPy buffer."""

    if array.flags.writeable:
        raise DatasetError(f"{name} must be read-only.")
    if not array.flags.c_contiguous:
        raise DatasetError(f"{name} must be C-contiguous.")
    if not array.flags.owndata:
        raise DatasetError(f"{name} must own its memory.")


def _validate_split_pair(
    features: ArrayLike | None,
    targets: ArrayLike | None,
    *,
    split_name: str,
) -> tuple[NDArray[np.complex128] | None, NDArray[np.float64] | None]:
    """Validate that an optional feature/target split is supplied as a pair."""

    if (features is None) != (targets is None):
        raise DatasetError(
            f"{split_name} features and targets must either both be provided "
            "or both be None."
        )
    if features is None:
        return None, None

    validated_features = _as_complex_matrix(
        features,
        name=f"X_{split_name}",
    )
    validated_targets = _as_real_target_matrix(
        targets,
        name=f"y_{split_name}",
    )
    if validated_features.shape[0] != validated_targets.shape[0]:
        raise DatasetError(
            f"X_{split_name} and y_{split_name} must contain the same number "
            f"of samples: {validated_features.shape[0]} != "
            f"{validated_targets.shape[0]}."
        )
    return validated_features, validated_targets


@dataclass(frozen=True, slots=True)
class CSSFDataset:
    """Immutable matrix dataset consumed by CSNN-T adapters.

    Parameters
    ----------
    features:
        Complex matrix ``X`` with shape ``(n_samples, n_features)``.
    targets:
        Real vector or matrix ``Y`` with shape ``(n_samples,)`` or
        ``(n_samples, n_targets)``.
    sample_ids:
        Optional unique identifiers. Deterministic identifiers are generated
        when omitted.
    metadata:
        JSON-compatible metadata, recursively detached and frozen.
    """

    features: NDArray[np.complex128]
    targets: NDArray[np.float64]
    sample_ids: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        features: ArrayLike,
        targets: ArrayLike,
        *,
        sample_ids: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        validated_features = _as_complex_matrix(features, name="features")
        validated_targets = _as_real_target_matrix(targets, name="targets")

        if validated_features.shape[0] != validated_targets.shape[0]:
            raise DatasetError(
                "features and targets must contain the same number of samples: "
                f"{validated_features.shape[0]} != "
                f"{validated_targets.shape[0]}."
            )

        object.__setattr__(self, "features", validated_features)
        object.__setattr__(self, "targets", validated_targets)
        object.__setattr__(
            self,
            "sample_ids",
            _normalize_sample_ids(
                sample_ids,
                sample_count=validated_features.shape[0],
            ),
        )
        object.__setattr__(self, "metadata", _normalize_metadata(metadata))

        _validate_array_ownership("features", self.features)
        _validate_array_ownership("targets", self.targets)

    @property
    def n_samples(self) -> int:
        """Number of samples."""

        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        """Number of complex spectral features."""

        return int(self.features.shape[1])

    @property
    def n_targets(self) -> int:
        """Number of real target columns."""

        return int(self.targets.shape[1])

    def take(self, indices: ArrayLike) -> "CSSFDataset":
        """Return a validated subset in the requested order."""

        index_array = np.asarray(indices)

        if index_array.ndim != 1:
            raise DatasetError("indices must be one-dimensional.")
        if index_array.size == 0:
            raise DatasetError("indices must not be empty.")
        if np.issubdtype(index_array.dtype, np.bool_):
            raise DatasetError("indices must not contain booleans.")
        if not np.issubdtype(index_array.dtype, np.integer):
            raise DatasetError("indices must contain integers.")

        normalized = np.asarray(index_array, dtype=np.int64)

        if np.any(normalized < 0) or np.any(normalized >= self.n_samples):
            raise DatasetError("indices contain values outside dataset bounds.")
        if np.unique(normalized).size != normalized.size:
            raise DatasetError("indices must not contain duplicates.")

        return CSSFDataset(
            self.features[normalized],
            self.targets[normalized],
            sample_ids=tuple(self.sample_ids[index] for index in normalized),
            metadata=self.metadata,
        )

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of dataset content."""

        digest = hashlib.sha256()
        digest.update(b"CSSFDataset-v1\0")
        _update_array_digest(digest, name="features", array=self.features)
        _update_array_digest(digest, name="targets", array=self.targets)
        digest.update(_canonical_json_bytes(self.sample_ids))
        digest.update(b"\0")
        digest.update(_canonical_json_bytes(self.metadata))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BESSDataset:
    """Immutable legacy dataset required by the frozen CSNN-T implementation.

    The training split is mandatory. Validation and test splits are optional,
    but each must be supplied as a complete feature/target pair. All supplied
    splits share the same number of complex features and real target columns.

    Attributes use the historical names expected by ``core/csnn_t.py``:

    ``X_train``
        Complex training matrix with shape ``(n_train, M_complex)``.
    ``y_train``
        Real training targets with shape ``(n_train, n)``.
    ``M_complex``
        Number of complex spectral features.
    ``n``
        Number of BESS/OPF target columns.
    """

    case: str
    X_train: NDArray[np.complex128]
    y_train: NDArray[np.float64]
    X_validation: NDArray[np.complex128] | None
    y_validation: NDArray[np.float64] | None
    X_test: NDArray[np.complex128] | None
    y_test: NDArray[np.float64] | None
    metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        case: str,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_validation: ArrayLike | None = None,
        y_validation: ArrayLike | None = None,
        X_test: ArrayLike | None = None,
        y_test: ArrayLike | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_case = str(case).strip()
        if not normalized_case:
            raise DatasetError("case must be a non-empty string.")

        train_features, train_targets = _validate_split_pair(
            X_train,
            y_train,
            split_name="train",
        )
        if train_features is None or train_targets is None:
            raise DatasetError("The training split is mandatory.")

        validation_features, validation_targets = _validate_split_pair(
            X_validation,
            y_validation,
            split_name="validation",
        )
        test_features, test_targets = _validate_split_pair(
            X_test,
            y_test,
            split_name="test",
        )

        expected_features = int(train_features.shape[1])
        expected_targets = int(train_targets.shape[1])

        for split_name, features, targets in (
            ("validation", validation_features, validation_targets),
            ("test", test_features, test_targets),
        ):
            if features is None or targets is None:
                continue
            if features.shape[1] != expected_features:
                raise DatasetError(
                    f"X_{split_name} contains {features.shape[1]} features; "
                    f"expected {expected_features}."
                )
            if targets.shape[1] != expected_targets:
                raise DatasetError(
                    f"y_{split_name} contains {targets.shape[1]} targets; "
                    f"expected {expected_targets}."
                )

        object.__setattr__(self, "case", normalized_case)
        object.__setattr__(self, "X_train", train_features)
        object.__setattr__(self, "y_train", train_targets)
        object.__setattr__(self, "X_validation", validation_features)
        object.__setattr__(self, "y_validation", validation_targets)
        object.__setattr__(self, "X_test", test_features)
        object.__setattr__(self, "y_test", test_targets)
        object.__setattr__(self, "metadata", _normalize_metadata(metadata))

        for name in (
            "X_train",
            "y_train",
            "X_validation",
            "y_validation",
            "X_test",
            "y_test",
        ):
            array = getattr(self, name)
            if array is not None:
                _validate_array_ownership(name, array)

    @property
    def M_complex(self) -> int:
        """Number of complex spectral features."""

        return int(self.X_train.shape[1])

    @property
    def n(self) -> int:
        """Number of real target columns, conventionally the bus count."""

        return int(self.y_train.shape[1])

    @property
    def n_train(self) -> int:
        """Number of training scenarios."""

        return int(self.X_train.shape[0])

    @property
    def n_validation(self) -> int:
        """Number of validation scenarios, or zero when absent."""

        return 0 if self.X_validation is None else int(self.X_validation.shape[0])

    @property
    def n_test(self) -> int:
        """Number of test scenarios, or zero when absent."""

        return 0 if self.X_test is None else int(self.X_test.shape[0])

    @property
    def has_validation(self) -> bool:
        """Whether a complete validation split is present."""

        return self.X_validation is not None

    @property
    def has_test(self) -> bool:
        """Whether a complete test split is present."""

        return self.X_test is not None

    def training_dataset(self) -> CSSFDataset:
        """Return the training split through the generic dataset contract."""

        return CSSFDataset(
            self.X_train,
            self.y_train,
            metadata={**dict(self.metadata), "case": self.case, "split": "train"},
        )

    def validation_dataset(self) -> CSSFDataset | None:
        """Return the validation split through the generic contract."""

        if self.X_validation is None or self.y_validation is None:
            return None
        return CSSFDataset(
            self.X_validation,
            self.y_validation,
            metadata={
                **dict(self.metadata),
                "case": self.case,
                "split": "validation",
            },
        )

    def test_dataset(self) -> CSSFDataset | None:
        """Return the test split through the generic contract."""

        if self.X_test is None or self.y_test is None:
            return None
        return CSSFDataset(
            self.X_test,
            self.y_test,
            metadata={**dict(self.metadata), "case": self.case, "split": "test"},
        )

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of all splits."""

        digest = hashlib.sha256()
        digest.update(b"BESSDataset-v1\0")
        digest.update(self.case.encode("utf-8"))
        digest.update(b"\0")
        for name in (
            "X_train",
            "y_train",
            "X_validation",
            "y_validation",
            "X_test",
            "y_test",
        ):
            _update_array_digest(
                digest,
                name=name,
                array=getattr(self, name),
            )
        digest.update(_canonical_json_bytes(self.metadata))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Deterministic non-overlapping train/validation/test partition."""

    train: CSSFDataset
    validation: CSSFDataset
    test: CSSFDataset
    seed: int

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, CSSFDataset)
            for item in (self.train, self.validation, self.test)
        ):
            raise TypeError("All split members must be CSSFDataset instances.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if not 0 <= self.seed <= 2**32 - 1:
            raise DatasetError("seed must lie in [0, 2**32 - 1].")

        dimensions = {
            (item.n_features, item.n_targets)
            for item in (self.train, self.validation, self.test)
        }
        if len(dimensions) != 1:
            raise DatasetError(
                "All splits must have equal feature and target dimensions."
            )

        train_ids = set(self.train.sample_ids)
        validation_ids = set(self.validation.sample_ids)
        test_ids = set(self.test.sample_ids)

        if train_ids & validation_ids:
            raise DatasetError("Train and validation splits overlap.")
        if train_ids & test_ids:
            raise DatasetError("Train and test splits overlap.")
        if validation_ids & test_ids:
            raise DatasetError("Validation and test splits overlap.")


def split_dataset(
    dataset: CSSFDataset,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    """Create a reproducible shuffled train/validation/test split."""

    if not isinstance(dataset, CSSFDataset):
        raise TypeError("dataset must be a CSSFDataset instance.")
    if isinstance(validation_fraction, bool) or not isinstance(
        validation_fraction,
        (int, float),
    ):
        raise TypeError("validation_fraction must be a real number.")
    if isinstance(test_fraction, bool) or not isinstance(
        test_fraction,
        (int, float),
    ):
        raise TypeError("test_fraction must be a real number.")
    validation_fraction = float(validation_fraction)
    test_fraction = float(test_fraction)
    if not math.isfinite(validation_fraction):
        raise DatasetError("validation_fraction must be finite.")
    if not math.isfinite(test_fraction):
        raise DatasetError("test_fraction must be finite.")
    if not 0.0 < validation_fraction < 1.0:
        raise DatasetError("validation_fraction must lie strictly in (0, 1).")
    if not 0.0 < test_fraction < 1.0:
        raise DatasetError("test_fraction must lie strictly in (0, 1).")
    if validation_fraction + test_fraction >= 1.0:
        raise DatasetError(
            "validation_fraction + test_fraction must be smaller than 1."
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not 0 <= seed <= 2**32 - 1:
        raise DatasetError("seed must lie in [0, 2**32 - 1].")
    if dataset.n_samples < 3:
        raise DatasetError(
            "At least three samples are required for a three-way split."
        )

    validation_count = max(
        1,
        int(np.floor(dataset.n_samples * validation_fraction)),
    )
    test_count = max(
        1,
        int(np.floor(dataset.n_samples * test_fraction)),
    )
    train_count = dataset.n_samples - validation_count - test_count

    if train_count < 1:
        raise DatasetError(
            "Requested fractions leave no sample for the training split."
        )

    generator = np.random.default_rng(seed)
    permutation = generator.permutation(dataset.n_samples)

    train_indices = permutation[:train_count]
    validation_indices = permutation[
        train_count : train_count + validation_count
    ]
    test_indices = permutation[train_count + validation_count :]

    return DatasetSplit(
        train=dataset.take(train_indices),
        validation=dataset.take(validation_indices),
        test=dataset.take(test_indices),
        seed=seed,
    )


__all__ = [
    "COMPLEX_DTYPE",
    "REAL_DTYPE",
    "DatasetError",
    "CSSFDataset",
    "BESSDataset",
    "DatasetSplit",
    "split_dataset",
]
