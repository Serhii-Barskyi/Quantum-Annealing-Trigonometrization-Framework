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

"""Reproducible scientific manifests for validated CSSF-QA runs.

The manifest binds one :class:`pipeline.runner.PipelineRunResult` to the exact
case300 input, BESS-QUBO, QA-response surrogate evidence, Pegasus result,
identical AC/OOD verification, and certified HiGHS quality reference.

The sole learned target is the response of quantum annealing.  QAOA remains a
tied-angle regression reference and MA-QAOA remains the term-wise coordinate
decomposition of digitized QA; neither is represented as an independent
surrogate or production optimizer.

The unchanged ``core/gcv.py`` and ``core/csnn_t.py`` files are verified before
a manifest can be built.  This proves implementation identity, not the final
mathematical adequacy of CSNN-T for D-Wave QA.  That stronger claim remains a
separate final scientific gate and is recorded explicitly as pending.

No filesystem or solver operation is performed at import.  Writing a manifest
is explicit and atomic.  Credentials, secrets, wall-clock competitions,
time-to-solution claims, and speedup claims are rejected recursively.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from bess.case300 import (
    CASE300_DATASET_PATH,
    CASE300_DATASET_SHA256,
    CASE300_DATASET_SIZE_BYTES,
    verify_case300_dataset,
)
from core.validation import verify_frozen_core
from pipeline import (
    MAQAOA_ROLE,
    PRODUCTION_STAGE_ORDER,
    QAOA_ROLE,
    QA_COORDINATE_MODEL,
    QA_SCHEDULE_MAPPING,
    RUNTIME_ROLE,
    SOLE_SURROGATED_SYSTEM,
)
from pipeline.runner import PipelineRunResult
from project_paths import PROJECT_ROOT


SCIENTIFIC_MANIFEST_SCHEMA: Final[str] = (
    "cssf-qa-scientific-run-manifest-v1"
)
SCIENTIFIC_MANIFEST_VERSION: Final[str] = "0.1.0"
HASH_ALGORITHM: Final[str] = "sha256"
SCIENTIFIC_EVIDENCE_STATUS: Final[str] = "descriptive_quality_only"
CSNNT_IMPLEMENTATION_STATUS: Final[str] = (
    "frozen_implementation_identity_verified"
)
CSNNT_MATHEMATICAL_STATUS: Final[str] = (
    "pending_final_qa_surrogation_verification_gate"
)
EXPECTED_FROZEN_CORE_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "core/gcv.py": (
            "3a4f4298a65d486269c6d4657d21f98f864b590e869fe80eefe2e9cdbb4c8ff7"
        ),
        "core/csnn_t.py": (
            "afcc08ebd3237ec722669b703f2b8087e457d07d43655823988d60ce7c87d8c8"
        ),
    }
)
CSNNT_MODEL_DECLARATIONS: Final[frozenset[str]] = frozenset(
    {
        "csnn_t",
        "cssf_csnn_t",
        "csnn_t_quantum_annealing_response",
        "csnn-t_quantum-annealing-response",
    }
)
CRITICAL_SOURCE_FILES: Final[tuple[str, ...]] = (
    "core/gcv.py",
    "core/csnn_t.py",
    "core/csnn_t_adapter.py",
    "qa/schedule.py",
    "qa/surrogate.py",
    "qubo/builder.py",
    "dwave_backend/solver.py",
    "baselines/highs.py",
    "baselines/comparison.py",
    "pipeline/__init__.py",
    "pipeline/runner.py",
    "pipeline/manifest.py",
    "config/schema.py",
    "config/base.yaml",
    "config/case300.yaml",
)
SENSITIVE_KEY_TOKENS: Final[tuple[str, ...]] = (
    "token",
    "password",
    "secret",
    "credential",
    "authorization",
    "api_key",
    "apikey",
)
FORBIDDEN_CLAIM_TOKENS: Final[tuple[str, ...]] = (
    "compare_wall_clock",
    "same_wall_clock_budget",
    "time_to_solution",
    "runtime_superiority",
    "runtime_advantage",
    "speedup_claim",
    "faster_than_highs",
)


class ScientificManifestError(ValueError):
    """Raised when a scientific manifest would be unsafe or inconsistent."""


def _sha256_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string.")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ScientificManifestError(
            f"{name} must be a 64-character hexadecimal SHA-256 digest."
        )
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _sensitive_key(value: object) -> bool:
    normalized = _normalized_key(value)
    return any(token in normalized for token in SENSITIVE_KEY_TOKENS)


def _forbidden_claim_key(value: object) -> bool:
    normalized = _normalized_key(value)
    return any(token in normalized for token in FORBIDDEN_CLAIM_TOKENS)


def _freeze_json(value: object, *, path: str = "manifest") -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScientificManifestError(
                f"{path} must contain only finite floating-point values."
            )
        return float(value)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ScientificManifestError(f"{path} contains an empty key.")
            if _sensitive_key(key):
                raise ScientificManifestError(
                    f"{path} must not contain credentials or secrets."
                )
            if _forbidden_claim_key(key):
                raise ScientificManifestError(
                    f"{path} contains forbidden time-comparison claim {key!r}."
                )
            result[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ScientificManifestError(
        f"{path} contains unsupported value type {type(value).__name__}."
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        _thaw_json(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _critical_source_manifest(project_root: Path) -> dict[str, object]:
    files: dict[str, object] = {}
    for relative in CRITICAL_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise ScientificManifestError(
                f"Critical project file is missing: {path}."
            )
        files[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return files


def _verified_frozen_core(project_root: Path) -> dict[str, object]:
    report = verify_frozen_core(project_root)
    if not report.valid:
        raise ScientificManifestError("Frozen CSNN-T/GCV core is invalid.")

    files: dict[str, object] = {}
    for item in report.files:
        expected = EXPECTED_FROZEN_CORE_SHA256.get(item.relative_path)
        if expected is None or item.actual_sha256 != expected:
            raise ScientificManifestError(
                f"Unexpected frozen-core identity for {item.relative_path}."
            )
        files[item.relative_path] = {
            "sha256": item.actual_sha256,
            "size_bytes": item.actual_size_bytes,
            "byte_for_byte_verified": item.valid,
        }
    if set(files) != set(EXPECTED_FROZEN_CORE_SHA256):
        raise ScientificManifestError(
            "Frozen-core report must contain exactly gcv.py and csnn_t.py."
        )
    return {
        "status": CSNNT_IMPLEMENTATION_STATUS,
        "files": files,
        "monkey_patching_allowed": False,
        "mathematical_application_status": CSNNT_MATHEMATICAL_STATUS,
    }


def _csnnt_declaration(run: PipelineRunResult) -> str:
    metadata = dict(run.qa_surrogate_quality.metadata)
    declaration = ""
    for key in ("model", "model_family", "surrogate_model"):
        if key in metadata:
            declaration = str(metadata[key]).strip().lower()
            break
    if declaration not in CSNNT_MODEL_DECLARATIONS:
        raise ScientificManifestError(
            "QA-surrogate quality metadata must declare the CSNN-T model "
            "used for quantum-annealing-response surrogation."
        )
    return declaration


def _validate_run_lineage(run: PipelineRunResult) -> None:
    if tuple(record.stage for record in run.stage_records) != (
        PRODUCTION_STAGE_ORDER
    ):
        raise ScientificManifestError(
            "Pipeline stage records do not follow the canonical order."
        )
    if run.data.source_sha256 != CASE300_DATASET_SHA256:
        raise ScientificManifestError("Run does not use the canonical case300.")
    if run.qa_surrogate_quality.source_fingerprint != run.bess_qubo.fingerprint():
        raise ScientificManifestError(
            "QA-surrogate quality belongs to a different BESS-QUBO."
        )
    if run.qa_surrogate_quality.partition not in {"validation", "test", "ood"}:
        raise ScientificManifestError(
            "QA-surrogate evidence must be held-out validation/test/OOD."
        )
    if run.cssf_qa_result.source_fingerprint != run.bess_qubo.fingerprint():
        raise ScientificManifestError(
            "CSSF-QA solve result belongs to a different BESS-QUBO."
        )
    if run.highs_reference.solve_result.source_fingerprint != (
        run.bess_qubo.fingerprint()
    ):
        raise ScientificManifestError(
            "HiGHS reference belongs to a different BESS-QUBO."
        )
    if not run.highs_reference.solve_result.certified_optimal:
        raise ScientificManifestError(
            "HiGHS quality reference must be certified optimal."
        )
    if run.comparison.evidence_status != SCIENTIFIC_EVIDENCE_STATUS:
        raise ScientificManifestError(
            "Comparison evidence must remain descriptive_quality_only."
        )
    if run.comparison.qa_surrogate_quality.fingerprint() != (
        run.qa_surrogate_quality.fingerprint()
    ):
        raise ScientificManifestError(
            "Comparison contains a different QA-surrogate quality record."
        )
    if run.comparison.cssf_qa_result_fingerprint != (
        run.cssf_qa_result.fingerprint()
    ):
        raise ScientificManifestError(
            "Comparison contains a different CSSF-QA result fingerprint."
        )
    if run.comparison.highs_result_fingerprint != (
        run.highs_reference.solve_result.fingerprint()
    ):
        raise ScientificManifestError(
            "Comparison contains a different HiGHS result fingerprint."
        )


def _ac_quality_manifest(value: object) -> dict[str, object]:
    return {
        "fingerprint": value.fingerprint(),
        "backend": value.backend,
        "partition": value.partition,
        "scenario_count": len(value.scenario_ids),
        "scenario_ids_sha256": hashlib.sha256(
            json.dumps(
                list(value.scenario_ids),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "placement_buses": list(value.placement.selected_buses),
        "mean_objective": value.mean_objective,
        "median_objective": value.median_objective,
        "feasibility_rate": value.feasibility_rate,
        "source_fingerprint": value.source_fingerprint,
        "verification_fingerprint": value.verification_fingerprint,
        "metadata": dict(value.metadata),
    }


def _highs_manifest(run: PipelineRunResult) -> dict[str, object]:
    result = run.highs_reference.solve_result
    return {
        "fingerprint": result.fingerprint(),
        "role": "classical_solution_quality_reference",
        "certified_optimal": result.certified_optimal,
        "model_status": result.model_status,
        "solver_version": result.solver_version,
        "source_fingerprint": result.source_fingerprint,
        "linearization_fingerprint": result.linearization_fingerprint,
        "config_fingerprint": result.config_fingerprint,
        "selected_buses": list(result.placement.selected_buses),
        "objective_value": result.objective_value,
        "combined_qubo_energy": result.combined_qubo_energy,
        "solver_objective_value": result.solver_objective_value,
        "metadata": dict(result.metadata),
    }


@dataclass(frozen=True, slots=True)
class ScientificRunManifest:
    """Immutable, canonical, JSON-serializable scientific run manifest."""

    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        frozen = _freeze_json(self.payload)
        if not isinstance(frozen, MappingProxyType):
            raise ScientificManifestError("Manifest normalization failed.")
        if frozen.get("schema") != SCIENTIFIC_MANIFEST_SCHEMA:
            raise ScientificManifestError("Unexpected scientific manifest schema.")
        if frozen.get("manifest_version") != SCIENTIFIC_MANIFEST_VERSION:
            raise ScientificManifestError("Unexpected manifest version.")
        object.__setattr__(self, "payload", frozen)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-ScientificRunManifest-v1\0")
        digest.update(_canonical_json_bytes(self.payload))
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        document = _thaw_json(self.payload)
        if not isinstance(document, dict):
            raise ScientificManifestError("Manifest payload is not a mapping.")
        document["manifest_fingerprint"] = self.fingerprint()
        return document

    def to_json(self, *, indent: int = 2) -> str:
        if isinstance(indent, bool) or not isinstance(indent, int):
            raise TypeError("indent must be an integer.")
        if indent < 0:
            raise ScientificManifestError("indent must be non-negative.")
        return json.dumps(
            self.manifest(),
            sort_keys=True,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        ) + "\n"


def build_scientific_manifest(
    run: PipelineRunResult,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ScientificRunManifest:
    """Build and validate the complete reproducibility document for one run."""

    if not isinstance(run, PipelineRunResult):
        raise TypeError("run must be PipelineRunResult.")

    _validate_run_lineage(run)
    csnnt_declaration = _csnnt_declaration(run)
    dataset_digest = verify_case300_dataset()
    if dataset_digest != CASE300_DATASET_SHA256:
        raise ScientificManifestError("Canonical case300 verification failed.")

    frozen_core = _verified_frozen_core(PROJECT_ROOT)
    source_files = _critical_source_manifest(PROJECT_ROOT)
    comparison_manifest = run.comparison.manifest()
    runner_manifest = run.manifest()

    payload: dict[str, object] = {
        "schema": SCIENTIFIC_MANIFEST_SCHEMA,
        "manifest_version": SCIENTIFIC_MANIFEST_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "framework": "Complex Spectral Surrogate Framework (CSSF)",
        "author": "Serhii Barskyi",
        "project_root": str(PROJECT_ROOT),
        "scientific_scope": {
            "sole_surrogated_system": SOLE_SURROGATED_SYSTEM,
            "qa_coordinate_model": QA_COORDINATE_MODEL,
            "qa_schedule_mapping": QA_SCHEDULE_MAPPING,
            "qaoa_role": QAOA_ROLE,
            "maqaoa_role": MAQAOA_ROLE,
            "qaoa_independently_surrogated": False,
            "maqaoa_independently_surrogated": False,
            "highs_role": "classical_solution_quality_reference",
            "runtime_role": RUNTIME_ROLE,
            "wall_clock_competition": False,
            "evidence_status": SCIENTIFIC_EVIDENCE_STATUS,
        },
        "csnn_t_qa_application": {
            "declared_model": csnnt_declaration,
            "implementation": frozen_core,
            "coordinate_domain": (
                "physical_image_of_qa_to_maqaoa_schedule_mapping"
            ),
            "prediction_rule": "real_part_of_complex_spectral_linear_map",
            "regularization_implementation": "frozen_gcv_tikhonov",
            "final_mathematical_gate": "tests/test_final_gates.py",
            "final_mathematical_gate_status": "pending",
            "claim_limit": (
                "implementation identity and held-out/OOD empirical quality; "
                "no universal QA-surrogation theorem is claimed"
            ),
        },
        "input_identity": {
            "case": run.data.case,
            "dataset_path": str(CASE300_DATASET_PATH),
            "dataset_sha256": dataset_digest,
            "dataset_size_bytes": CASE300_DATASET_SIZE_BYTES,
            "dataset_fingerprint": run.data.fingerprint(),
            "n_scenarios": run.data.n_scenarios,
            "n_train": run.data.n_train,
            "n_test": run.data.n_test,
            "feature_shape": list(run.data.features.shape),
            "target_shape": list(run.data.targets.shape),
        },
        "source_identity": {
            "critical_files": source_files,
        },
        "pipeline": runner_manifest,
        "quality_evidence": {
            "qa_surrogate": {
                "fingerprint": run.qa_surrogate_quality.fingerprint(),
                "source_fingerprint": (
                    run.qa_surrogate_quality.source_fingerprint
                ),
                "partition": run.qa_surrogate_quality.partition,
                "target_names": list(run.qa_surrogate_quality.target_names),
                "sample_count": run.qa_surrogate_quality.n_samples,
                "summary": run.qa_surrogate_quality.summary(),
                "metadata": dict(run.qa_surrogate_quality.metadata),
            },
            "cssf_qa_solver": run.cssf_qa_result.manifest(),
            "cssf_qa_ac_post_verification": _ac_quality_manifest(
                run.cssf_qa_ac_quality
            ),
            "highs_reference": _highs_manifest(run),
            "highs_ac_post_verification": _ac_quality_manifest(
                run.highs_reference.ac_quality
            ),
            "comparison": comparison_manifest,
        },
        "execution_claim_policy": {
            "reported_qa_backend": run.cssf_qa_result.backend,
            "reported_solver_id": run.cssf_qa_result.solver_id,
            "physical_execution_inferred_by_manifest_builder": False,
            "backend_provenance_source": "validated_solver_result_only",
            "runtime_is_quality_metric": False,
            "performance_superiority_claims": "forbidden",
        },
        "metadata": {} if metadata is None else dict(metadata),
    }
    return ScientificRunManifest(payload)


def verify_scientific_manifest(
    document: Mapping[str, object],
    *,
    expected_run_fingerprint: str | None = None,
) -> str:
    """Validate one exported manifest and return its fingerprint."""

    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping.")
    source = dict(document)
    supplied = _sha256_digest(
        source.pop("manifest_fingerprint", ""),
        name="manifest_fingerprint",
    )
    manifest = ScientificRunManifest(source)
    actual = manifest.fingerprint()
    if supplied != actual:
        raise ScientificManifestError(
            "Scientific manifest fingerprint does not match its content."
        )
    if expected_run_fingerprint is not None:
        expected = _sha256_digest(
            expected_run_fingerprint,
            name="expected_run_fingerprint",
        )
        pipeline = source.get("pipeline")
        if not isinstance(pipeline, Mapping):
            raise ScientificManifestError("Manifest pipeline section is missing.")
        run_fingerprint = _sha256_digest(
            pipeline.get("fingerprint"),
            name="pipeline.fingerprint",
        )
        if run_fingerprint != expected:
            raise ScientificManifestError(
                "Manifest belongs to a different pipeline run."
            )
    return actual


def write_scientific_manifest(
    manifest: ScientificRunManifest,
    destination: str | Path,
) -> Path:
    """Atomically write a manifest below the fixed project root."""

    if not isinstance(manifest, ScientificRunManifest):
        raise TypeError("manifest must be ScientificRunManifest.")
    path = Path(destination).expanduser()
    if path.suffix.lower() != ".json":
        raise ScientificManifestError("Manifest destination must use .json.")

    root = PROJECT_ROOT.resolve(strict=False)
    normalized = path.resolve(strict=False)
    if normalized != root and root not in normalized.parents:
        raise ScientificManifestError(
            "Manifest destination must be below the fixed project root."
        )
    normalized.parent.mkdir(parents=True, exist_ok=True)

    temporary = normalized.with_name(f".{normalized.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(manifest.to_json(), encoding="utf-8", newline="\n")
        os.replace(temporary, normalized)
    finally:
        if temporary.exists():
            temporary.unlink()
    return normalized


__all__ = [
    "SCIENTIFIC_MANIFEST_SCHEMA",
    "SCIENTIFIC_MANIFEST_VERSION",
    "HASH_ALGORITHM",
    "SCIENTIFIC_EVIDENCE_STATUS",
    "CSNNT_IMPLEMENTATION_STATUS",
    "CSNNT_MATHEMATICAL_STATUS",
    "EXPECTED_FROZEN_CORE_SHA256",
    "CSNNT_MODEL_DECLARATIONS",
    "CRITICAL_SOURCE_FILES",
    "SENSITIVE_KEY_TOKENS",
    "FORBIDDEN_CLAIM_TOKENS",
    "ScientificManifestError",
    "ScientificRunManifest",
    "build_scientific_manifest",
    "verify_scientific_manifest",
    "write_scientific_manifest",
]
