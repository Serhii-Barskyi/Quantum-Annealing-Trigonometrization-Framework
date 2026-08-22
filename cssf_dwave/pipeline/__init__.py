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

"""Stable end-to-end pipeline contracts for CSSF-QA BESS placement.

The sole learned surrogate target is the response of quantum annealing.
Ordinary QAOA is used only as a tied-angle regression reference, and
MA-QAOA is used only as the term-wise coordinate decomposition of digitized
quantum annealing. Neither QAOA nor MA-QAOA is an independent surrogate
product or a production placement backend.

The production flow is fixed to

``case300 -> candidates -> BESS QUBO -> QA-response surrogate ->
Pegasus execution -> identical AC/OOD verification -> HiGHS quality
reference -> quality comparison``.

Runtime may be recorded as a reproducibility diagnostic, but wall-clock or
time-to-solution competition is forbidden. This initializer performs only
lightweight validation and imports no NumPy, Qiskit, Ocean, HiGHS, CUDA,
OPF solver, project dataset, or network resource.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Final


PIPELINE_PACKAGE_NAME: Final[str] = "cssf.pipeline"
PIPELINE_PACKAGE_VERSION: Final[str] = "0.1.0"

SOLE_SURROGATED_SYSTEM: Final[str] = "quantum_annealing_response"
QA_COORDINATE_MODEL: Final[str] = "maqaoa_decomposition"
QA_SCHEDULE_MAPPING: Final[str] = "qa_to_maqaoa"
QAOA_ROLE: Final[str] = "qa_modeling_and_regression"
MAQAOA_ROLE: Final[str] = (
    "qa_coordinate_decomposition_and_regression"
)

LOCAL_QA_BACKEND: Final[str] = "local_sqa_gpu"
QPU_QA_BACKEND: Final[str] = "pegasus_qpu"
HIGHS_REFERENCE_BACKEND: Final[str] = "highs"
SUPPORTED_QA_BACKENDS: Final[tuple[str, ...]] = (
    LOCAL_QA_BACKEND,
    QPU_QA_BACKEND,
)

RUNTIME_ROLE: Final[str] = "diagnostic_only"
WALL_CLOCK_COMPETITION_ALLOWED: Final[bool] = False
QAOA_INDEPENDENT_SURROGATE_ALLOWED: Final[bool] = False
MAQAOA_INDEPENDENT_SURROGATE_ALLOWED: Final[bool] = False


class PipelineStage(str, Enum):
    """Ordered stages of the production BESS-placement pipeline."""

    LOAD_CASE300 = "load_case300"
    SELECT_BESS_CANDIDATES = "select_bess_candidates"
    BUILD_BESS_QUBO = "build_bess_qubo"
    PREDICT_QA_RESPONSE = "predict_quantum_annealing_response"
    EXECUTE_PEGASUS = "execute_pegasus"
    AC_OOD_POST_VERIFY = "ac_ood_post_verify"
    SOLVE_HIGHS_REFERENCE = "solve_highs_quality_reference"
    COMPARE_SOLUTION_QUALITY = "compare_solution_quality"


PRODUCTION_STAGE_ORDER: Final[tuple[str, ...]] = tuple(
    stage.value for stage in PipelineStage
)
RESEARCH_VALIDATION_LEVELS: Final[tuple[str, ...]] = (
    "qaoa_tied_angle_regression",
    "maqaoa_qa_coordinate_decomposition",
    "digitized_qa_convergence",
)
FORBIDDEN_PRODUCTION_STAGES: Final[tuple[str, ...]] = (
    "surrogate_qaoa",
    "surrogate_maqaoa",
    "optimize_with_qaoa",
    "optimize_with_maqaoa",
    "compare_wall_clock",
    "compare_time_to_solution",
)


class PipelineContractError(ValueError):
    """Raised when the scientific pipeline contract is violated."""


def _value(value: object, *, name: str) -> str:
    """Normalize strings and string-valued enums."""

    candidate = value.value if isinstance(value, Enum) else value
    if not isinstance(candidate, str):
        raise TypeError(f"{name} must be a string or string-valued enum.")
    normalized = candidate.strip()
    if not normalized:
        raise PipelineContractError(f"{name} must not be empty.")
    return normalized


def _attribute(value: object, name: str, *, owner: str) -> object:
    """Read one required attribute from a duck-typed project object."""

    if not hasattr(value, name):
        raise PipelineContractError(
            f"{owner} must expose required attribute {name!r}."
        )
    return getattr(value, name)


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean.")
    return value


def validate_stage_order(stages: object) -> tuple[str, ...]:
    """Require the complete canonical production-stage sequence."""

    try:
        values = tuple(stages)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("stages must be an iterable.") from exc

    normalized = tuple(
        _value(stage, name=f"stages[{index}]")
        for index, stage in enumerate(values)
    )
    if normalized != PRODUCTION_STAGE_ORDER:
        raise PipelineContractError(
            "Production stages must exactly match the canonical order "
            f"{PRODUCTION_STAGE_ORDER}; received {normalized}."
        )
    if any(stage in FORBIDDEN_PRODUCTION_STAGES for stage in normalized):
        raise PipelineContractError(
            "QAOA/MA-QAOA surrogate products and time competitions are "
            "forbidden production stages."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """Immutable, solver-independent production pipeline plan."""

    qa_backend: str = LOCAL_QA_BACKEND
    surrogated_system: str = SOLE_SURROGATED_SYSTEM
    stage_order: tuple[str, ...] = PRODUCTION_STAGE_ORDER
    qaoa_independently_surrogated: bool = False
    maqaoa_independently_surrogated: bool = False
    include_highs_quality_reference: bool = True
    compare_wall_clock: bool = False
    runtime_role: str = RUNTIME_ROLE

    def __post_init__(self) -> None:
        backend = _value(self.qa_backend, name="qa_backend").lower()
        if backend not in SUPPORTED_QA_BACKENDS:
            raise PipelineContractError(
                "qa_backend must be local_sqa_gpu or pegasus_qpu; "
                f"received {backend!r}."
            )

        target = _value(
            self.surrogated_system,
            name="surrogated_system",
        ).lower()
        if target != SOLE_SURROGATED_SYSTEM:
            raise PipelineContractError(
                "The sole surrogate target must be "
                f"{SOLE_SURROGATED_SYSTEM!r}."
            )

        qaoa_independent = _strict_bool(
            self.qaoa_independently_surrogated,
            name="qaoa_independently_surrogated",
        )
        maqaoa_independent = _strict_bool(
            self.maqaoa_independently_surrogated,
            name="maqaoa_independently_surrogated",
        )
        include_highs = _strict_bool(
            self.include_highs_quality_reference,
            name="include_highs_quality_reference",
        )
        compare_wall_clock = _strict_bool(
            self.compare_wall_clock,
            name="compare_wall_clock",
        )
        runtime_role = _value(self.runtime_role, name="runtime_role").lower()

        if qaoa_independent or maqaoa_independent:
            raise PipelineContractError(
                "QAOA and MA-QAOA cannot be independently surrogated."
            )
        if not include_highs:
            raise PipelineContractError(
                "The certified HiGHS solution-quality reference is required."
            )
        if compare_wall_clock:
            raise PipelineContractError(
                "Wall-clock competition is forbidden; runtime is diagnostic."
            )
        if runtime_role != RUNTIME_ROLE:
            raise PipelineContractError(
                f"runtime_role must be {RUNTIME_ROLE!r}."
            )

        object.__setattr__(self, "qa_backend", backend)
        object.__setattr__(self, "surrogated_system", target)
        object.__setattr__(
            self,
            "stage_order",
            validate_stage_order(self.stage_order),
        )
        object.__setattr__(self, "runtime_role", runtime_role)

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of the plan."""

        payload = {
            "qa_backend": self.qa_backend,
            "surrogated_system": self.surrogated_system,
            "stage_order": list(self.stage_order),
            "qaoa_independently_surrogated": (
                self.qaoa_independently_surrogated
            ),
            "maqaoa_independently_surrogated": (
                self.maqaoa_independently_surrogated
            ),
            "include_highs_quality_reference": (
                self.include_highs_quality_reference
            ),
            "compare_wall_clock": self.compare_wall_clock,
            "runtime_role": self.runtime_role,
        }
        digest = hashlib.sha256()
        digest.update(b"CSSF-PipelinePlan-v1\0")
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        """Return a JSON-ready reproducibility manifest."""

        return {
            "fingerprint": self.fingerprint(),
            "qa_backend": self.qa_backend,
            "surrogated_system": self.surrogated_system,
            "stage_order": list(self.stage_order),
            "qaoa_role": QAOA_ROLE,
            "maqaoa_role": MAQAOA_ROLE,
            "qaoa_independently_surrogated": False,
            "maqaoa_independently_surrogated": False,
            "highs_role": "classical_solution_quality_reference",
            "runtime_role": self.runtime_role,
            "wall_clock_competition": False,
        }


def validate_cssf_pipeline_config(config: object) -> PipelinePlan:
    """Validate a duck-typed ``config.schema.CSSFConfig`` instance.

    The function deliberately avoids importing Pydantic or project solver
    modules, keeping ``import pipeline`` free of optional runtime side effects.
    """

    project = _attribute(config, "project", owner="config")
    qaoa = _attribute(config, "qaoa", owner="config")
    maqaoa = _attribute(config, "maqaoa", owner="config")
    qa = _attribute(config, "qa", owner="config")
    emulator = _attribute(config, "emulator", owner="config")
    qpu = _attribute(config, "qpu", owner="config")
    benchmark = _attribute(config, "benchmark", owner="config")

    if _strict_bool(
        _attribute(project, "strict_no_fallbacks", owner="config.project"),
        name="config.project.strict_no_fallbacks",
    ) is not True:
        raise PipelineContractError("Project fallbacks must remain forbidden.")

    if _value(
        _attribute(qaoa, "role", owner="config.qaoa"),
        name="config.qaoa.role",
    ) != QAOA_ROLE:
        raise PipelineContractError("QAOA role is inconsistent with QA modeling.")
    if _strict_bool(
        _attribute(
            qaoa,
            "independently_surrogated",
            owner="config.qaoa",
        ),
        name="config.qaoa.independently_surrogated",
    ):
        raise PipelineContractError("QAOA cannot be independently surrogated.")

    if _value(
        _attribute(maqaoa, "role", owner="config.maqaoa"),
        name="config.maqaoa.role",
    ) != MAQAOA_ROLE:
        raise PipelineContractError(
            "MA-QAOA role is inconsistent with QA coordinate decomposition."
        )
    if _strict_bool(
        _attribute(
            maqaoa,
            "independently_surrogated",
            owner="config.maqaoa",
        ),
        name="config.maqaoa.independently_surrogated",
    ):
        raise PipelineContractError(
            "MA-QAOA cannot be independently surrogated."
        )

    if _value(
        _attribute(qa, "surrogated_system", owner="config.qa"),
        name="config.qa.surrogated_system",
    ) != SOLE_SURROGATED_SYSTEM:
        raise PipelineContractError(
            "QA response must be the sole surrogate target."
        )
    if _value(
        _attribute(qa, "coordinate_model", owner="config.qa"),
        name="config.qa.coordinate_model",
    ) != QA_COORDINATE_MODEL:
        raise PipelineContractError(
            "QA coordinates must use the MA-QAOA decomposition."
        )
    if _value(
        _attribute(qa, "schedule_mapping", owner="config.qa"),
        name="config.qa.schedule_mapping",
    ) != QA_SCHEDULE_MAPPING:
        raise PipelineContractError(
            "QA schedule mapping must be qa_to_maqaoa."
        )

    emulator_backend = _value(
        _attribute(emulator, "backend", owner="config.emulator"),
        name="config.emulator.backend",
    ).lower()
    if emulator_backend != LOCAL_QA_BACKEND:
        raise PipelineContractError(
            "The local production emulator must be local_sqa_gpu."
        )
    if _value(
        _attribute(emulator, "topology_type", owner="config.emulator"),
        name="config.emulator.topology_type",
    ).lower() != "pegasus":
        raise PipelineContractError("The emulator topology must be Pegasus.")
    if _strict_bool(
        _attribute(
            emulator,
            "allow_classical_fallback",
            owner="config.emulator",
        ),
        name="config.emulator.allow_classical_fallback",
    ):
        raise PipelineContractError("Classical emulator fallback is forbidden.")

    if _value(
        _attribute(qpu, "backend", owner="config.qpu"),
        name="config.qpu.backend",
    ).lower() != QPU_QA_BACKEND:
        raise PipelineContractError("The real-QPU backend must be pegasus_qpu.")
    if _value(
        _attribute(qpu, "topology_type", owner="config.qpu"),
        name="config.qpu.topology_type",
    ).lower() != "pegasus":
        raise PipelineContractError("The real-QPU topology must be Pegasus.")
    if _strict_bool(
        _attribute(
            qpu,
            "allow_solver_fallback",
            owner="config.qpu",
        ),
        name="config.qpu.allow_solver_fallback",
    ):
        raise PipelineContractError("QPU solver fallback is forbidden.")

    if _value(
        _attribute(
            benchmark,
            "baseline_solver",
            owner="config.benchmark",
        ),
        name="config.benchmark.baseline_solver",
    ).lower() != HIGHS_REFERENCE_BACKEND:
        raise PipelineContractError("HiGHS must be the quality reference.")
    if _value(
        _attribute(
            benchmark,
            "surrogated_system",
            owner="config.benchmark",
        ),
        name="config.benchmark.surrogated_system",
    ) != SOLE_SURROGATED_SYSTEM:
        raise PipelineContractError(
            "Benchmark and QA sections must share the QA-response target."
        )

    required_true = (
        "compare_solution_quality",
        "compare_qa_surrogate_approximation",
        "compare_same_objective",
        "compare_same_constraints",
        "ac_post_verify_both_methods",
        "use_ood_scenarios",
    )
    for name in required_true:
        if _strict_bool(
            _attribute(benchmark, name, owner="config.benchmark"),
            name=f"config.benchmark.{name}",
        ) is not True:
            raise PipelineContractError(
                f"config.benchmark.{name} must be enabled."
            )

    if _strict_bool(
        _attribute(
            benchmark,
            "compare_wall_clock",
            owner="config.benchmark",
        ),
        name="config.benchmark.compare_wall_clock",
    ):
        raise PipelineContractError("Wall-clock competition is forbidden.")
    if _value(
        _attribute(benchmark, "runtime_role", owner="config.benchmark"),
        name="config.benchmark.runtime_role",
    ).lower() != RUNTIME_ROLE:
        raise PipelineContractError(
            "Benchmark runtime must remain diagnostic only."
        )

    qpu_enabled = _strict_bool(
        _attribute(qpu, "enabled", owner="config.qpu"),
        name="config.qpu.enabled",
    )
    qa_backend = QPU_QA_BACKEND if qpu_enabled else LOCAL_QA_BACKEND
    return PipelinePlan(qa_backend=qa_backend)


__all__ = [
    "PIPELINE_PACKAGE_NAME",
    "PIPELINE_PACKAGE_VERSION",
    "SOLE_SURROGATED_SYSTEM",
    "QA_COORDINATE_MODEL",
    "QA_SCHEDULE_MAPPING",
    "QAOA_ROLE",
    "MAQAOA_ROLE",
    "LOCAL_QA_BACKEND",
    "QPU_QA_BACKEND",
    "HIGHS_REFERENCE_BACKEND",
    "SUPPORTED_QA_BACKENDS",
    "RUNTIME_ROLE",
    "WALL_CLOCK_COMPETITION_ALLOWED",
    "QAOA_INDEPENDENT_SURROGATE_ALLOWED",
    "MAQAOA_INDEPENDENT_SURROGATE_ALLOWED",
    "PipelineStage",
    "PRODUCTION_STAGE_ORDER",
    "RESEARCH_VALIDATION_LEVELS",
    "FORBIDDEN_PRODUCTION_STAGES",
    "PipelineContractError",
    "PipelinePlan",
    "validate_stage_order",
    "validate_cssf_pipeline_config",
]
