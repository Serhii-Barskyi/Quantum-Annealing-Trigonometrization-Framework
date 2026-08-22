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

"""Strict validation utilities for CSSF-QA-D-Wave.

The validators enforce the project contracts without modifying the frozen
``core/gcv.py`` and ``core/csnn_t.py`` implementation.

Production invariants:

* project root: ``/content/drive/MyDrive/cssf_dwave``;
* statevector backend: Qiskit Aer GPU only;
* QPU topology: Pegasus only;
* QPU families: ``Advantage_system4.*`` and ``Advantage_system6.*`` only;
* silent fallback: forbidden.

No validation runs automatically during import.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.types import ALLOWED_PEGASUS_SOLVER_PREFIXES


COLAB_PROJECT_ROOT: Final[Path] = Path(
    "/content/drive/MyDrive/cssf_dwave"
)
FROZEN_MANIFEST_RELATIVE_PATH: Final[Path] = (
    Path("core") / "frozen_manifest.json"
)


class CSSFValidationError(RuntimeError):
    """Raised when a strict CSSF contract is violated."""


@dataclass(frozen=True, slots=True)
class FrozenFileVerification:
    """Result of one frozen-file integrity check."""

    relative_path: str
    expected_sha256: str
    actual_sha256: str
    expected_size_bytes: int
    actual_size_bytes: int

    @property
    def valid(self) -> bool:
        """Whether hash and size both match the manifest."""

        return (
            self.expected_sha256 == self.actual_sha256
            and self.expected_size_bytes == self.actual_size_bytes
        )


@dataclass(frozen=True, slots=True)
class FrozenCoreReport:
    """Complete immutable-core verification report."""

    project_root: Path
    manifest_path: Path
    files: tuple[FrozenFileVerification, ...]

    @property
    def valid(self) -> bool:
        """Whether every frozen file passed verification."""

        return bool(self.files) and all(item.valid for item in self.files)

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-ready report."""

        return {
            "project_root": str(self.project_root),
            "manifest_path": str(self.manifest_path),
            "valid": self.valid,
            "files": [
                {
                    "relative_path": item.relative_path,
                    "expected_sha256": item.expected_sha256,
                    "actual_sha256": item.actual_sha256,
                    "expected_size_bytes": item.expected_size_bytes,
                    "actual_size_bytes": item.actual_size_bytes,
                    "valid": item.valid,
                }
                for item in self.files
            ],
        }


def sha256_file(path: str | Path) -> str:
    """Calculate SHA-256 without changing the file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise CSSFValidationError(
            f"File does not exist or is not regular: {file_path}"
        )

    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_colab_project_root(path: str | Path) -> Path:
    """Require the exact production project root."""

    normalized = Path(path)
    if normalized != COLAB_PROJECT_ROOT:
        raise CSSFValidationError(
            "CSSF production root must be exactly "
            f"{COLAB_PROJECT_ROOT}; received {normalized}."
        )
    return normalized


def _load_frozen_manifest(project_root: Path) -> tuple[Path, dict[str, Any]]:
    """Load and structurally validate ``frozen_manifest.json``."""

    manifest_path = project_root / FROZEN_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise CSSFValidationError(
            f"Frozen manifest is missing: {manifest_path}"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CSSFValidationError(
            f"Frozen manifest is invalid UTF-8 JSON: {manifest_path}"
        ) from exc

    if not isinstance(manifest, dict):
        raise CSSFValidationError(
            "Frozen manifest top level must be a JSON object."
        )
    if manifest.get("hash_algorithm") != "sha256":
        raise CSSFValidationError("Frozen manifest must use SHA-256.")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"gcv.py", "csnn_t.py"}:
        raise CSSFValidationError(
            "Frozen manifest must contain exactly gcv.py and csnn_t.py."
        )

    policy = manifest.get("frozen_core_policy")
    required_policy = {
        "files_are_immutable": True,
        "copy_mode": "byte_for_byte",
        "monkey_patching_allowed": False,
        "automatic_reformatting_allowed": False,
        "line_ending_conversion_allowed": False,
    }
    if not isinstance(policy, dict):
        raise CSSFValidationError(
            "Frozen manifest is missing frozen_core_policy."
        )
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise CSSFValidationError(
                f"Frozen policy {key!r} must equal {expected!r}."
            )

    return manifest_path, manifest


def verify_frozen_core(
    project_root: str | Path = COLAB_PROJECT_ROOT,
    *,
    raise_on_mismatch: bool = True,
) -> FrozenCoreReport:
    """Verify frozen files against their registered hashes.

    The explicit ``project_root`` argument exists for isolated tests of a copied
    project tree. Production code uses the fixed default Colab root.
    """

    root = Path(project_root)
    manifest_path, manifest = _load_frozen_manifest(root)
    checks: list[FrozenFileVerification] = []

    for filename in ("gcv.py", "csnn_t.py"):
        entry = manifest["files"][filename]
        if not isinstance(entry, dict):
            raise CSSFValidationError(
                f"Manifest entry for {filename} must be an object."
            )

        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise CSSFValidationError(
                f"Invalid SHA-256 entry for {filename}."
            )
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 1
        ):
            raise CSSFValidationError(
                f"Invalid size_bytes entry for {filename}."
            )
        if entry.get("immutable") is not True:
            raise CSSFValidationError(
                f"Manifest entry {filename} must be immutable."
            )
        if entry.get("copy_policy") != "byte_for_byte":
            raise CSSFValidationError(
                f"Manifest entry {filename} must use byte_for_byte copy."
            )
        if entry.get("modification_allowed") is not False:
            raise CSSFValidationError(
                f"Manifest entry {filename} must forbid modification."
            )

        file_path = root / "core" / filename
        if not file_path.is_file():
            raise CSSFValidationError(
                f"Frozen file is missing: {file_path}"
            )

        check = FrozenFileVerification(
            relative_path=f"core/{filename}",
            expected_sha256=expected_hash,
            actual_sha256=sha256_file(file_path),
            expected_size_bytes=expected_size,
            actual_size_bytes=file_path.stat().st_size,
        )
        checks.append(check)

        if raise_on_mismatch and not check.valid:
            raise CSSFValidationError(
                f"Frozen file integrity failure for {file_path}: "
                f"expected sha256={check.expected_sha256}, "
                f"size={check.expected_size_bytes}; actual "
                f"sha256={check.actual_sha256}, "
                f"size={check.actual_size_bytes}."
            )

    report = FrozenCoreReport(
        project_root=root,
        manifest_path=manifest_path,
        files=tuple(checks),
    )
    if raise_on_mismatch and not report.valid:
        raise CSSFValidationError("Frozen core verification failed.")
    return report


def validate_feature_target_contract(
    features: ArrayLike,
    targets: ArrayLike,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Validate the matrix contract consumed by frozen CSNN-T.

    Targets with shape ``(N,)`` are converted to ``(N, 1)``. Returned arrays
    are copied into contiguous ``complex128`` and ``float64`` buffers.
    """

    feature_array = np.asarray(features)
    target_array = np.asarray(targets)

    if feature_array.ndim != 2 or 0 in feature_array.shape:
        raise CSSFValidationError(
            "features must be a non-empty matrix (n_samples, n_features)."
        )

    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    elif target_array.ndim != 2:
        raise CSSFValidationError(
            "targets must have shape (n_samples,) or "
            "(n_samples, n_targets)."
        )
    if 0 in target_array.shape:
        raise CSSFValidationError("targets must be non-empty.")
    if feature_array.shape[0] != target_array.shape[0]:
        raise CSSFValidationError(
            "features and targets must contain the same number of samples."
        )

    if np.iscomplexobj(target_array):
        if np.any(np.asarray(target_array.imag) != 0.0):
            raise CSSFValidationError(
                "CSNN-T targets must be real-valued."
            )
        target_array = target_array.real

    complex_features = np.ascontiguousarray(
        feature_array,
        dtype=np.complex128,
    )
    real_targets = np.ascontiguousarray(
        target_array,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(complex_features.real)):
        raise CSSFValidationError(
            "features contain non-finite real components."
        )
    if not np.all(np.isfinite(complex_features.imag)):
        raise CSSFValidationError(
            "features contain non-finite imaginary components."
        )
    if not np.all(np.isfinite(real_targets)):
        raise CSSFValidationError("targets contain non-finite values.")

    return complex_features, real_targets


def validate_aer_gpu_metadata(metadata: Mapping[str, Any]) -> None:
    """Require explicit Qiskit Aer GPU execution metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    device = str(metadata.get("device", "")).strip().upper()
    if device != "GPU":
        raise CSSFValidationError(
            "Qiskit Aer CPU fallback is forbidden; "
            f"reported device={metadata.get('device')!r}."
        )


def validate_pegasus_solver(*, solver_id: str, topology_type: str) -> None:
    """Validate the strict real-QPU identity."""

    if not isinstance(solver_id, str):
        raise TypeError("solver_id must be a string.")
    if not isinstance(topology_type, str):
        raise TypeError("topology_type must be a string.")

    normalized_solver = solver_id.strip()
    normalized_topology = topology_type.strip().lower()

    if not normalized_solver:
        raise CSSFValidationError("An explicit solver_id is required.")
    if normalized_topology != "pegasus":
        raise CSSFValidationError(
            "Only Pegasus QPU topology is permitted."
        )
    if not normalized_solver.startswith(ALLOWED_PEGASUS_SOLVER_PREFIXES):
        raise CSSFValidationError(
            "solver_id must start with Advantage_system4. or "
            "Advantage_system6."
        )


def validate_no_fallback(
    *,
    allow_fallback: bool,
    field_name: str = "allow_fallback",
) -> None:
    """Reject any enabled fallback flag."""

    if not isinstance(allow_fallback, bool):
        raise TypeError(f"{field_name} must be boolean.")
    if allow_fallback:
        raise CSSFValidationError(
            f"{field_name}=True violates the strict no-fallback policy."
        )


__all__ = [
    "COLAB_PROJECT_ROOT",
    "FROZEN_MANIFEST_RELATIVE_PATH",
    "CSSFValidationError",
    "FrozenFileVerification",
    "FrozenCoreReport",
    "sha256_file",
    "validate_colab_project_root",
    "verify_frozen_core",
    "validate_feature_target_contract",
    "validate_aer_gpu_metadata",
    "validate_pegasus_solver",
    "validate_no_fallback",
]
