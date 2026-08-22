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

"""Strict execution runner for the QA-only CSSF BESS pipeline.

The runner executes the canonical stage order declared by :mod:`pipeline` and
checks the complete fingerprint lineage between case300, train-only candidate
selection, the BESS QUBO, held-out/OOD QA-response surrogate quality, Pegasus
sampling, identical AC post-verification, and the certified HiGHS quality
reference.

Heavy stages are supplied explicitly as callables.  This prevents implicit
backend substitution, hidden CPU fallback, credential lookup, or accidental
execution during import.  QAOA and MA-QAOA do not appear as production solver
stages: they remain regression and QA-coordinate validation levels only.

Runtime may be recorded elsewhere as a diagnostic, but this module neither
measures nor compares wall-clock or time-to-solution values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from baselines.comparison import (
    ACPostVerifiedPlacementQuality,
    BESSQualityComparison,
    QASurrogateQuality,
    compare_bess_solution_quality,
)
from baselines.highs import HighsBESSResult
from bess.case300 import (
    CASE300_DATASET_SHA256,
    Case300ModeAData,
)
from bess.candidates import CandidateSelectionResult
from dwave_backend.solver import DWaveSolveResult
from pipeline import (
    PRODUCTION_STAGE_ORDER,
    RUNTIME_ROLE,
    SOLE_SURROGATED_SYSTEM,
    PipelinePlan,
    PipelineStage,
)
from qubo.builder import BESSPlacementQUBO


RUNNER_SCHEMA: Final[str] = "cssf-pipeline-runner-v1"
RUNNER_VERSION: Final[str] = "0.1.0"
SENSITIVE_METADATA_TOKENS: Final[tuple[str, ...]] = (
    "token",
    "password",
    "secret",
    "credential",
    "authorization",
    "api_key",
    "apikey",
)
FORBIDDEN_COMPARISON_TOKENS: Final[tuple[str, ...]] = (
    "compare_wall_clock",
    "time_to_solution",
    "runtime_superiority",
    "runtime_advantage",
    "speedup_claim",
)


class PipelineRunError(RuntimeError):
    """Raised when one stage violates the end-to-end pipeline lineage."""


def _sha256_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string.")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise PipelineRunError(
            f"{name} must be a 64-character hexadecimal SHA-256 digest."
        )
    return normalized


def _metadata_key_is_sensitive(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(token in normalized for token in SENSITIVE_METADATA_TOKENS)


def _freeze_json(value: object, *, path: str = "metadata") -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise PipelineRunError(f"{path} must contain only finite values.")
        return float(value)
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                raise PipelineRunError(f"{path} contains an empty key.")
            if _metadata_key_is_sensitive(normalized_key):
                raise PipelineRunError(
                    f"{path} must not contain credentials or secrets."
                )
            canonical_key = normalized_key.lower().replace("-", "_")
            if canonical_key in FORBIDDEN_COMPARISON_TOKENS:
                raise PipelineRunError(
                    f"{path} contains forbidden time-comparison field "
                    f"{normalized_key!r}."
                )
            frozen[normalized_key] = _freeze_json(
                item,
                path=f"{path}.{normalized_key}",
            )
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise PipelineRunError(
        f"{path} contains unsupported value type {type(value).__name__}."
    )


def _immutable_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    source = {} if metadata is None else dict(metadata)
    frozen = _freeze_json(source)
    if not isinstance(frozen, MappingProxyType):
        raise PipelineRunError("metadata normalization failed.")
    return frozen


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _callable(value: object, *, name: str) -> Callable[..., object]:
    if not callable(value):
        raise TypeError(f"{name} must be callable.")
    return value


def _fingerprint(value: object, *, name: str) -> str:
    method = getattr(value, "fingerprint", None)
    if not callable(method):
        raise PipelineRunError(f"{name} must expose fingerprint().")
    return _sha256_digest(method(), name=f"{name}.fingerprint()")


def _same_selection(first: object, second: object) -> bool:
    first_selection = getattr(first, "selection", None)
    second_selection = getattr(second, "selection", None)
    if first_selection is None or second_selection is None:
        return False
    return bool(np.array_equal(first_selection, second_selection))


@dataclass(frozen=True, slots=True)
class HighsReferenceStageResult:
    """Certified HiGHS result plus identical AC/OOD post-verification."""

    solve_result: HighsBESSResult
    ac_quality: ACPostVerifiedPlacementQuality

    def __post_init__(self) -> None:
        if not isinstance(self.solve_result, HighsBESSResult):
            raise TypeError("solve_result must be HighsBESSResult.")
        if not isinstance(self.ac_quality, ACPostVerifiedPlacementQuality):
            raise TypeError(
                "ac_quality must be ACPostVerifiedPlacementQuality."
            )
        if not self.solve_result.certified_optimal:
            raise PipelineRunError(
                "HiGHS quality reference must be certified optimal."
            )
        if self.ac_quality.backend != "highs":
            raise PipelineRunError("HiGHS AC quality backend must be highs.")
        if (
            self.ac_quality.source_fingerprint
            != self.solve_result.source_fingerprint
        ):
            raise PipelineRunError(
                "HiGHS solve and AC verification source fingerprints differ."
            )
        if not _same_selection(
            self.ac_quality.placement,
            self.solve_result.placement,
        ):
            raise PipelineRunError(
                "HiGHS AC verification placement differs from solve result."
            )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-HighsReferenceStageResult-v1\0")
        digest.update(self.solve_result.fingerprint().encode("ascii"))
        digest.update(self.ac_quality.fingerprint().encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PipelineExecutors:
    """Explicit stage callables; no stage has an implicit fallback."""

    load_case300: Callable[[], Case300ModeAData]
    select_bess_candidates: Callable[
        [Case300ModeAData],
        CandidateSelectionResult,
    ]
    build_bess_qubo: Callable[
        [Case300ModeAData, CandidateSelectionResult],
        BESSPlacementQUBO,
    ]
    predict_qa_response: Callable[
        [Case300ModeAData, BESSPlacementQUBO],
        QASurrogateQuality,
    ]
    execute_pegasus: Callable[
        [BESSPlacementQUBO],
        DWaveSolveResult,
    ]
    ac_ood_post_verify: Callable[
        [Case300ModeAData, BESSPlacementQUBO, DWaveSolveResult],
        ACPostVerifiedPlacementQuality,
    ]
    solve_highs_quality_reference: Callable[
        [Case300ModeAData, BESSPlacementQUBO],
        HighsReferenceStageResult,
    ]

    def __post_init__(self) -> None:
        for name in (
            "load_case300",
            "select_bess_candidates",
            "build_bess_qubo",
            "predict_qa_response",
            "execute_pegasus",
            "ac_ood_post_verify",
            "solve_highs_quality_reference",
        ):
            _callable(getattr(self, name), name=name)


@dataclass(frozen=True, slots=True)
class PipelineStageRecord:
    """One immutable stage output and its parent-fingerprint lineage."""

    stage: str
    artifact_type: str
    artifact_fingerprint: str
    parent_fingerprints: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stage = str(self.stage).strip()
        if stage not in PRODUCTION_STAGE_ORDER:
            raise PipelineRunError(f"Unknown production stage {stage!r}.")
        artifact_type = str(self.artifact_type).strip()
        if not artifact_type:
            raise PipelineRunError("artifact_type must not be empty.")
        parents = tuple(
            _sha256_digest(value, name="parent_fingerprint")
            for value in self.parent_fingerprints
        )
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(
            self,
            "artifact_fingerprint",
            _sha256_digest(
                self.artifact_fingerprint,
                name="artifact_fingerprint",
            ),
        )
        object.__setattr__(self, "parent_fingerprints", parents)
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-PipelineStageRecord-v1\0")
        digest.update(self.stage.encode("ascii"))
        digest.update(self.artifact_type.encode("utf-8"))
        digest.update(self.artifact_fingerprint.encode("ascii"))
        for parent in self.parent_fingerprints:
            digest.update(parent.encode("ascii"))
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
            "stage": self.stage,
            "artifact_type": self.artifact_type,
            "artifact_fingerprint": self.artifact_fingerprint,
            "parent_fingerprints": list(self.parent_fingerprints),
            "record_fingerprint": self.fingerprint(),
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Complete immutable output of one validated CSSF-QA pipeline run."""

    plan: PipelinePlan
    data: Case300ModeAData
    candidate_selection: CandidateSelectionResult
    bess_qubo: BESSPlacementQUBO
    qa_surrogate_quality: QASurrogateQuality
    cssf_qa_result: DWaveSolveResult
    cssf_qa_ac_quality: ACPostVerifiedPlacementQuality
    highs_reference: HighsReferenceStageResult
    comparison: BESSQualityComparison
    stage_records: tuple[PipelineStageRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PipelinePlan):
            raise TypeError("plan must be PipelinePlan.")
        expected_types = (
            ("data", self.data, Case300ModeAData),
            (
                "candidate_selection",
                self.candidate_selection,
                CandidateSelectionResult,
            ),
            ("bess_qubo", self.bess_qubo, BESSPlacementQUBO),
            (
                "qa_surrogate_quality",
                self.qa_surrogate_quality,
                QASurrogateQuality,
            ),
            ("cssf_qa_result", self.cssf_qa_result, DWaveSolveResult),
            (
                "cssf_qa_ac_quality",
                self.cssf_qa_ac_quality,
                ACPostVerifiedPlacementQuality,
            ),
            (
                "highs_reference",
                self.highs_reference,
                HighsReferenceStageResult,
            ),
            ("comparison", self.comparison, BESSQualityComparison),
        )
        for name, value, expected in expected_types:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}.")

        records = tuple(self.stage_records)
        if tuple(record.stage for record in records) != PRODUCTION_STAGE_ORDER:
            raise PipelineRunError(
                "stage_records must exactly follow PRODUCTION_STAGE_ORDER."
            )
        if len({record.stage for record in records}) != len(records):
            raise PipelineRunError("stage_records must not contain duplicates.")

        object.__setattr__(self, "stage_records", records)
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CSSF-PipelineRunResult-v1\0")
        digest.update(self.plan.fingerprint().encode("ascii"))
        for record in self.stage_records:
            digest.update(record.fingerprint().encode("ascii"))
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
            "schema": RUNNER_SCHEMA,
            "runner_version": RUNNER_VERSION,
            "fingerprint": self.fingerprint(),
            "plan": self.plan.manifest(),
            "surrogated_system": SOLE_SURROGATED_SYSTEM,
            "runtime_role": RUNTIME_ROLE,
            "wall_clock_competition": False,
            "dataset_fingerprint": self.data.fingerprint(),
            "candidate_selection_fingerprint": (
                self.candidate_selection.fingerprint()
            ),
            "bess_qubo_fingerprint": self.bess_qubo.fingerprint(),
            "qa_surrogate_quality_fingerprint": (
                self.qa_surrogate_quality.fingerprint()
            ),
            "cssf_qa_result_fingerprint": self.cssf_qa_result.fingerprint(),
            "cssf_qa_ac_quality_fingerprint": (
                self.cssf_qa_ac_quality.fingerprint()
            ),
            "highs_reference_fingerprint": self.highs_reference.fingerprint(),
            "comparison_fingerprint": self.comparison.fingerprint(),
            "stage_records": [
                record.manifest() for record in self.stage_records
            ],
            "metadata": _thaw_json(self.metadata),
        }


def _record(
    stage: PipelineStage,
    artifact: object,
    *,
    parents: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> PipelineStageRecord:
    return PipelineStageRecord(
        stage=stage.value,
        artifact_type=type(artifact).__name__,
        artifact_fingerprint=_fingerprint(
            artifact,
            name=f"{stage.value} artifact",
        ),
        parent_fingerprints=tuple(parents),
        metadata={} if metadata is None else metadata,
    )


def run_cssf_pipeline(
    *,
    plan: PipelinePlan,
    executors: PipelineExecutors,
    metadata: Mapping[str, Any] | None = None,
) -> PipelineRunResult:
    """Execute and validate the complete QA-only production pipeline.

    Every heavy operation is explicit in ``executors``.  The canonical
    comparison implementation is not injectable, preventing a caller from
    replacing quality-only comparison with a wall-clock criterion.
    """

    if not isinstance(plan, PipelinePlan):
        raise TypeError("plan must be PipelinePlan.")
    if not isinstance(executors, PipelineExecutors):
        raise TypeError("executors must be PipelineExecutors.")

    normalized_metadata = _immutable_metadata(metadata)
    records: list[PipelineStageRecord] = []
    plan_fingerprint = plan.fingerprint()

    data = executors.load_case300()
    if not isinstance(data, Case300ModeAData):
        raise TypeError("load_case300 must return Case300ModeAData.")
    if data.source_sha256 != CASE300_DATASET_SHA256:
        raise PipelineRunError("case300 source SHA-256 is not canonical.")
    data_fingerprint = data.fingerprint()
    records.append(
        _record(
            PipelineStage.LOAD_CASE300,
            data,
            parents=(plan_fingerprint,),
            metadata={"source_sha256": data.source_sha256},
        )
    )

    selection = executors.select_bess_candidates(data)
    if not isinstance(selection, CandidateSelectionResult):
        raise TypeError(
            "select_bess_candidates must return CandidateSelectionResult."
        )
    if selection.source_fingerprint != data_fingerprint:
        raise PipelineRunError(
            "Candidate selection does not belong to the loaded case300 data."
        )
    selection_fingerprint = selection.fingerprint()
    records.append(
        _record(
            PipelineStage.SELECT_BESS_CANDIDATES,
            selection,
            parents=(data_fingerprint,),
            metadata={
                "candidate_count": len(selection.candidate_buses),
                "held_out_scenarios_used": selection.metadata.get(
                    "held_out_scenarios_used"
                ),
            },
        )
    )

    bess_qubo = executors.build_bess_qubo(data, selection)
    if not isinstance(bess_qubo, BESSPlacementQUBO):
        raise TypeError("build_bess_qubo must return BESSPlacementQUBO.")
    fleet_selection = bess_qubo.fleet.metadata.get(
        "candidate_selection_fingerprint"
    )
    if fleet_selection != selection_fingerprint:
        raise PipelineRunError(
            "BESS fleet does not belong to the candidate selection."
        )
    qubo_selection = bess_qubo.metadata.get(
        "candidate_selection_fingerprint"
    )
    if qubo_selection not in {None, selection_fingerprint}:
        raise PipelineRunError(
            "BESS QUBO metadata identifies a different candidate selection."
        )
    qubo_fingerprint = bess_qubo.fingerprint()
    records.append(
        _record(
            PipelineStage.BUILD_BESS_QUBO,
            bess_qubo,
            parents=(data_fingerprint, selection_fingerprint),
            metadata={
                "n_variables": bess_qubo.model.n_variables,
                "units_to_place": bess_qubo.fleet.units_to_place,
            },
        )
    )

    qa_quality = executors.predict_qa_response(data, bess_qubo)
    if not isinstance(qa_quality, QASurrogateQuality):
        raise TypeError(
            "predict_qa_response must return QASurrogateQuality."
        )
    if qa_quality.source_fingerprint != qubo_fingerprint:
        raise PipelineRunError(
            "QA-surrogate quality refers to a different BESS QUBO."
        )
    if qa_quality.partition not in {"test", "ood"}:
        raise PipelineRunError(
            "QA-surrogate evidence must be held-out test or OOD data."
        )
    qa_quality_fingerprint = qa_quality.fingerprint()
    records.append(
        _record(
            PipelineStage.PREDICT_QA_RESPONSE,
            qa_quality,
            parents=(qubo_fingerprint,),
            metadata={
                "partition": qa_quality.partition,
                "n_samples": qa_quality.n_samples,
                "n_targets": qa_quality.n_targets,
            },
        )
    )

    cssf_qa_result = executors.execute_pegasus(bess_qubo)
    if not isinstance(cssf_qa_result, DWaveSolveResult):
        raise TypeError("execute_pegasus must return DWaveSolveResult.")
    if cssf_qa_result.source_kind != "bess_placement_qubo":
        raise PipelineRunError(
            "Pegasus stage must solve a BESSPlacementQUBO."
        )
    if cssf_qa_result.source_fingerprint != qubo_fingerprint:
        raise PipelineRunError(
            "Pegasus result refers to a different BESS QUBO."
        )
    if cssf_qa_result.backend != plan.qa_backend:
        raise PipelineRunError(
            "Pegasus result backend differs from the validated pipeline plan."
        )
    if cssf_qa_result.placement is None:
        raise PipelineRunError("Pegasus result must contain a BESS placement.")
    cssf_result_fingerprint = cssf_qa_result.fingerprint()
    records.append(
        _record(
            PipelineStage.EXECUTE_PEGASUS,
            cssf_qa_result,
            parents=(qubo_fingerprint, qa_quality_fingerprint),
            metadata={
                "backend": cssf_qa_result.backend,
                "feasible_probability": (
                    cssf_qa_result.feasible_probability
                ),
            },
        )
    )

    cssf_ac_quality = executors.ac_ood_post_verify(
        data,
        bess_qubo,
        cssf_qa_result,
    )
    if not isinstance(cssf_ac_quality, ACPostVerifiedPlacementQuality):
        raise TypeError(
            "ac_ood_post_verify must return "
            "ACPostVerifiedPlacementQuality."
        )
    if cssf_ac_quality.backend != plan.qa_backend:
        raise PipelineRunError(
            "CSSF-QA AC verification backend differs from the plan."
        )
    if cssf_ac_quality.source_fingerprint != qubo_fingerprint:
        raise PipelineRunError(
            "CSSF-QA AC verification refers to a different BESS QUBO."
        )
    if not _same_selection(
        cssf_ac_quality.placement,
        cssf_qa_result.placement,
    ):
        raise PipelineRunError(
            "CSSF-QA AC verification placement differs from Pegasus result."
        )
    cssf_ac_fingerprint = cssf_ac_quality.fingerprint()
    records.append(
        _record(
            PipelineStage.AC_OOD_POST_VERIFY,
            cssf_ac_quality,
            parents=(cssf_result_fingerprint,),
            metadata={
                "partition": cssf_ac_quality.partition,
                "scenario_count": len(cssf_ac_quality.scenario_ids),
                "verification_fingerprint": (
                    cssf_ac_quality.verification_fingerprint
                ),
            },
        )
    )

    highs_reference = executors.solve_highs_quality_reference(
        data,
        bess_qubo,
    )
    if not isinstance(highs_reference, HighsReferenceStageResult):
        raise TypeError(
            "solve_highs_quality_reference must return "
            "HighsReferenceStageResult."
        )
    if highs_reference.solve_result.source_fingerprint != qubo_fingerprint:
        raise PipelineRunError(
            "HiGHS quality reference refers to a different BESS QUBO."
        )
    if (
        highs_reference.ac_quality.verification_fingerprint
        != cssf_ac_quality.verification_fingerprint
    ):
        raise PipelineRunError(
            "HiGHS and CSSF-QA must use the identical AC verification."
        )
    if (
        highs_reference.ac_quality.scenario_ids
        != cssf_ac_quality.scenario_ids
    ):
        raise PipelineRunError(
            "HiGHS and CSSF-QA must use identical ordered scenarios."
        )
    if highs_reference.ac_quality.partition != cssf_ac_quality.partition:
        raise PipelineRunError(
            "HiGHS and CSSF-QA AC partitions must match."
        )
    highs_fingerprint = highs_reference.fingerprint()
    records.append(
        _record(
            PipelineStage.SOLVE_HIGHS_REFERENCE,
            highs_reference,
            parents=(qubo_fingerprint, cssf_ac_fingerprint),
            metadata={
                "certified_optimal": True,
                "model_status": (
                    highs_reference.solve_result.model_status
                ),
            },
        )
    )

    comparison = compare_bess_solution_quality(
        highs_result=highs_reference.solve_result,
        cssf_qa_result=cssf_qa_result,
        highs_quality=highs_reference.ac_quality,
        cssf_qa_quality=cssf_ac_quality,
        qa_surrogate_quality=qa_quality,
        metadata={
            "pipeline_plan_fingerprint": plan_fingerprint,
            "comparison_scope": "quality_only",
        },
    )
    comparison_fingerprint = comparison.fingerprint()
    records.append(
        _record(
            PipelineStage.COMPARE_SOLUTION_QUALITY,
            comparison,
            parents=(
                qa_quality_fingerprint,
                cssf_ac_fingerprint,
                highs_fingerprint,
            ),
            metadata={
                "evidence_status": comparison.evidence_status,
                "runtime_role": RUNTIME_ROLE,
            },
        )
    )

    result = PipelineRunResult(
        plan=plan,
        data=data,
        candidate_selection=selection,
        bess_qubo=bess_qubo,
        qa_surrogate_quality=qa_quality,
        cssf_qa_result=cssf_qa_result,
        cssf_qa_ac_quality=cssf_ac_quality,
        highs_reference=highs_reference,
        comparison=comparison,
        stage_records=tuple(records),
        metadata=normalized_metadata,
    )
    if result.stage_records[-1].artifact_fingerprint != comparison_fingerprint:
        raise PipelineRunError(
            "Final stage record does not match comparison fingerprint."
        )
    return result


__all__ = [
    "RUNNER_SCHEMA",
    "RUNNER_VERSION",
    "SENSITIVE_METADATA_TOKENS",
    "FORBIDDEN_COMPARISON_TOKENS",
    "PipelineRunError",
    "HighsReferenceStageResult",
    "PipelineExecutors",
    "PipelineStageRecord",
    "PipelineRunResult",
    "run_cssf_pipeline",
]
