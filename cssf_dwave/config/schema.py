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

"""Strict Pydantic configuration schema for CSSF-QA-D-Wave.

The framework is intentionally restricted to:

* Google Colab;
* the fixed project root ``/content/drive/MyDrive/cssf_dwave``;
* Qiskit Aer GPU for every statevector experiment;
* Pegasus D-Wave topology;
* Advantage_system4.* or Advantage_system6.* real-QPU families;
* explicit failures instead of silent fallbacks.

The schema contains only configuration contracts. It performs no filesystem,
GPU, Ocean, QPU, OPF, or network operations during import.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


COLAB_PROJECT_ROOT: Final[Path] = Path(
    "/content/drive/MyDrive/cssf_dwave"
)
ALLOWED_PEGASUS_SOLVER_PREFIXES: Final[tuple[str, ...]] = (
    "Advantage_system4.",
    "Advantage_system6.",
)


class StrictConfigModel(BaseModel):
    """Base class forbidding unknown fields and validating assignment."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


class BackendKind(str, Enum):
    """Supported execution backends."""

    LOCAL_SA = "local_sa"
    LOCAL_DIGITIZED_QA_GPU = "local_digitized_qa_gpu"
    LOCAL_SQA_GPU = "local_sqa_gpu"
    PEGASUS_QPU = "pegasus_qpu"


class TrotterOrder(int, Enum):
    """Supported product-formula orders."""

    FIRST = 1
    SECOND = 2


class SurrogateTarget(str, Enum):
    """Observable targets allowed for QA-response surrogate learning."""

    MEAN_ENERGY = "mean_energy"
    ENERGY_VARIANCE = "energy_variance"
    ENERGY_QUANTILE_05 = "energy_quantile_05"
    CVAR_05 = "cvar_05"
    FEASIBILITY_PROBABILITY = "feasibility_probability"
    ELITE_PROBABILITY = "elite_probability"
    SUCCESS_PROBABILITY = "success_probability"


class ProjectConfig(StrictConfigModel):
    """Global project invariants."""

    name: Literal["cssf-qa-dwave"] = "cssf-qa-dwave"
    root: Path = COLAB_PROJECT_ROOT
    execution_environment: Literal["google_colab"] = "google_colab"
    strict_no_fallbacks: Literal[True] = True
    quantum_walk_enabled: Literal[False] = False

    @field_validator("root")
    @classmethod
    def validate_fixed_root(cls, value: Path) -> Path:
        normalized = Path(value)
        if normalized != COLAB_PROJECT_ROOT:
            raise ValueError(
                "The project root is fixed to "
                f"{COLAB_PROJECT_ROOT}; received {normalized}."
            )
        return normalized


class RuntimeConfig(StrictConfigModel):
    """Google Colab runtime contract."""

    require_gpu: Literal[True] = True
    require_cuda: Literal[True] = True
    statevector_provider: Literal["qiskit_aer_gpu"] = "qiskit_aer_gpu"
    aer_method: Literal["statevector"] = "statevector"
    aer_device: Literal["GPU"] = "GPU"
    fail_if_gpu_unavailable: Literal[True] = True
    install_from: Path = (
        COLAB_PROJECT_ROOT / "requirements-colab.txt"
    )

    @field_validator("install_from")
    @classmethod
    def validate_requirements_path(cls, value: Path) -> Path:
        expected = COLAB_PROJECT_ROOT / "requirements-colab.txt"
        normalized = Path(value)
        if normalized != expected:
            raise ValueError(
                f"Colab requirements path must be {expected}; "
                f"received {normalized}."
            )
        return normalized


class RandomConfig(StrictConfigModel):
    """Reproducibility seeds."""

    global_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42
    scenario_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 101
    qaoa_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 202
    maqaoa_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 303
    qa_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 404
    sampler_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 505


class OPFConfig(StrictConfigModel):
    """AC-OPF and scenario-generation settings."""

    network_case: Literal["case300"] = "case300"
    train_scenarios: PositiveInt = 400
    validation_scenarios: PositiveInt = 100
    ood_scenarios: PositiveInt = 100
    train_load_scale_min: PositiveFloat = 0.85
    train_load_scale_max: PositiveFloat = 1.15
    ood_load_scale_min: PositiveFloat = 1.32
    ood_load_scale_max: PositiveFloat = 1.50
    require_ac_convergence: Literal[True] = True
    finite_difference_step_mw: PositiveFloat = 1.0e-3

    @model_validator(mode="after")
    def validate_scenario_ranges(self) -> "OPFConfig":
        if self.train_load_scale_min >= self.train_load_scale_max:
            raise ValueError(
                "train_load_scale_min must be smaller than "
                "train_load_scale_max."
            )
        if self.ood_load_scale_min >= self.ood_load_scale_max:
            raise ValueError(
                "ood_load_scale_min must be smaller than "
                "ood_load_scale_max."
            )
        if self.ood_load_scale_min <= self.train_load_scale_max:
            raise ValueError(
                "OOD and training load-scale ranges must not overlap."
            )
        return self


class QUBOConfig(StrictConfigModel):
    """BESS QUBO construction settings."""

    candidate_count: Annotated[int, Field(ge=2, le=300)] = 80
    bess_units: Annotated[int, Field(ge=1, le=299)] = 15
    enforce_exact_cardinality: Literal[True] = True
    verify_qubo_ising_tolerance: PositiveFloat = 1.0e-10
    preserve_argmin_after_scaling: Literal[True] = True
    coefficient_range_linear: PositiveFloat = 2.0
    coefficient_range_quadratic: PositiveFloat = 1.0

    @model_validator(mode="after")
    def validate_cardinality(self) -> "QUBOConfig":
        if self.bess_units >= self.candidate_count:
            raise ValueError(
                "bess_units must be strictly smaller than candidate_count."
            )
        return self


class QAOATeacherConfig(StrictConfigModel):
    """Ordinary QAOA used only to model and regress digitized QA limits.

    QAOA is not an independently surrogated system in this project.  It is a
    tied-angle reference model used to validate the QA-to-MA-QAOA
    decomposition and related ablation experiments.
    """

    enabled: bool = True
    role: Literal["qa_modeling_and_regression"] = (
        "qa_modeling_and_regression"
    )
    independently_surrogated: Literal[False] = False
    depth: Annotated[int, Field(ge=1, le=12)] = 1
    training_points: Annotated[int, Field(ge=8)] = 60
    shots: Annotated[int, Field(ge=1)] = 4096
    statevector_provider: Literal["qiskit_aer_gpu"] = "qiskit_aer_gpu"
    aer_method: Literal["statevector"] = "statevector"
    aer_device: Literal["GPU"] = "GPU"
    weighted_lsf_mixer: Literal[True] = True


class MAQAOATeacherConfig(StrictConfigModel):
    """MA-QAOA coordinate decomposition of digitized quantum annealing.

    MA-QAOA is not an independently surrogated system.  Its term-wise angles
    are the coordinate language used to decompose and validate digitized QA.
    """

    enabled: bool = True
    role: Literal["qa_coordinate_decomposition_and_regression"] = (
        "qa_coordinate_decomposition_and_regression"
    )
    independently_surrogated: Literal[False] = False
    depth: Annotated[int, Field(ge=1, le=12)] = 1
    training_points: Annotated[int, Field(ge=8)] = 120
    shots: Annotated[int, Field(ge=1)] = 4096
    statevector_provider: Literal["qiskit_aer_gpu"] = "qiskit_aer_gpu"
    aer_method: Literal["statevector"] = "statevector"
    aer_device: Literal["GPU"] = "GPU"
    termwise_mixer_angles: Literal[True] = True
    termwise_linear_cost_angles: Literal[True] = True
    termwise_quadratic_cost_angles: Literal[True] = True
    require_tied_angle_qaoa_equivalence: Literal[True] = True


class QAConfig(StrictConfigModel):
    """The sole surrogate-learning target: quantum-annealing response."""

    enabled: bool = True
    surrogated_system: Literal["quantum_annealing_response"] = (
        "quantum_annealing_response"
    )
    coordinate_model: Literal["maqaoa_decomposition"] = (
        "maqaoa_decomposition"
    )
    schedule_mapping: Literal["qa_to_maqaoa"] = "qa_to_maqaoa"
    trotter_slices: Annotated[int, Field(ge=2, le=4096)] = 64
    trotter_order: TrotterOrder = TrotterOrder.SECOND
    total_annealing_time: PositiveFloat = 20.0
    schedule_points: Annotated[int, Field(ge=3)] = 101
    verify_trotter_convergence: Literal[True] = True
    targets: tuple[SurrogateTarget, ...] = (
        SurrogateTarget.MEAN_ENERGY,
        SurrogateTarget.ENERGY_VARIANCE,
        SurrogateTarget.ENERGY_QUANTILE_05,
        SurrogateTarget.CVAR_05,
        SurrogateTarget.FEASIBILITY_PROBABILITY,
        SurrogateTarget.ELITE_PROBABILITY,
        SurrogateTarget.SUCCESS_PROBABILITY,
    )

    @field_validator("targets")
    @classmethod
    def validate_targets(
        cls,
        value: tuple[SurrogateTarget, ...],
    ) -> tuple[SurrogateTarget, ...]:
        if not value:
            raise ValueError("At least one QA surrogate target is required.")
        if len(set(value)) != len(value):
            raise ValueError("QA surrogate targets must be unique.")
        return value


class EmulatorConfig(StrictConfigModel):
    """Local Ocean-compatible emulator settings."""

    backend: BackendKind = BackendKind.LOCAL_SQA_GPU
    num_reads: Annotated[int, Field(ge=1)] = 4096
    return_dimod_sampleset: Literal[True] = True
    topology_type: Literal["pegasus"] = "pegasus"
    pegasus_m: Annotated[int, Field(ge=2, le=64)] = 16
    require_gpu: Literal[True] = True
    allow_classical_fallback: Literal[False] = False
    sqa_trotter_replicas: Annotated[int, Field(ge=2, le=1024)] = 64
    sqa_sweeps: Annotated[int, Field(ge=1)] = 2000
    sqa_burn_in_sweeps: Annotated[int, Field(ge=0)] = 500

    @model_validator(mode="after")
    def validate_sqa_sweeps(self) -> "EmulatorConfig":
        if self.sqa_burn_in_sweeps >= self.sqa_sweeps:
            raise ValueError(
                "sqa_burn_in_sweeps must be smaller than sqa_sweeps."
            )
        if self.backend == BackendKind.PEGASUS_QPU:
            raise ValueError(
                "EmulatorConfig cannot select the real Pegasus QPU backend."
            )
        return self


class PegasusQPUConfig(StrictConfigModel):
    """Strict real-QPU selection policy."""

    enabled: bool = False
    backend: Literal["pegasus_qpu"] = "pegasus_qpu"
    topology_type: Literal["pegasus"] = "pegasus"
    allowed_solver_prefixes: tuple[str, ...] = (
        ALLOWED_PEGASUS_SOLVER_PREFIXES
    )
    solver_id: str | None = None
    require_explicit_solver_id: Literal[True] = True
    reject_zephyr: Literal[True] = True
    allow_solver_fallback: Literal[False] = False
    dry_run: bool = True
    num_reads: Annotated[int, Field(ge=1)] = 4096
    annealing_time: PositiveFloat = 20.0

    @field_validator("allowed_solver_prefixes")
    @classmethod
    def validate_solver_prefixes(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if value != ALLOWED_PEGASUS_SOLVER_PREFIXES:
            raise ValueError(
                "Allowed QPU prefixes are fixed to "
                f"{ALLOWED_PEGASUS_SOLVER_PREFIXES}."
            )
        return value

    @model_validator(mode="after")
    def validate_solver_selection(self) -> "PegasusQPUConfig":
        if self.enabled and not self.solver_id:
            raise ValueError(
                "An explicit Pegasus solver_id is required when QPU use "
                "is enabled."
            )
        if self.solver_id is not None and not self.solver_id.startswith(
            ALLOWED_PEGASUS_SOLVER_PREFIXES
        ):
            raise ValueError(
                "solver_id must start with Advantage_system4. or "
                "Advantage_system6."
            )
        return self


class BenchmarkConfig(StrictConfigModel):
    """Quality-only CSSF-QA-versus-HiGHS comparison contract.

    Runtime may be recorded as a reproducibility diagnostic, but wall-clock
    competition and time-to-solution superiority claims are forbidden.  The
    scientific axes are final BESS solution quality and approximation quality
    of the quantum-annealing-response surrogate.
    """

    baseline_solver: Literal["highs"] = "highs"
    surrogated_system: Literal["quantum_annealing_response"] = (
        "quantum_annealing_response"
    )
    compare_solution_quality: Literal[True] = True
    compare_qa_surrogate_approximation: Literal[True] = True
    compare_same_objective: Literal[True] = True
    compare_same_constraints: Literal[True] = True
    compare_wall_clock: Literal[False] = False
    runtime_role: Literal["diagnostic_only"] = "diagnostic_only"
    ac_post_verify_both_methods: Literal[True] = True
    use_ood_scenarios: Literal[True] = True
    bootstrap_samples: Annotated[int, Field(ge=100)] = 2000
    confidence_level: Annotated[
        float,
        Field(gt=0.5, lt=1.0),
    ] = 0.95
    require_positive_ac_advantage_ci: Literal[True] = True


class CSSFConfig(StrictConfigModel):
    """Top-level validated project configuration."""

    schema_version: Literal["1.0"] = "1.0"
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    random: RandomConfig = Field(default_factory=RandomConfig)
    opf: OPFConfig = Field(default_factory=OPFConfig)
    qubo: QUBOConfig = Field(default_factory=QUBOConfig)
    qaoa: QAOATeacherConfig = Field(default_factory=QAOATeacherConfig)
    maqaoa: MAQAOATeacherConfig = Field(
        default_factory=MAQAOATeacherConfig
    )
    qa: QAConfig = Field(default_factory=QAConfig)
    emulator: EmulatorConfig = Field(default_factory=EmulatorConfig)
    qpu: PegasusQPUConfig = Field(default_factory=PegasusQPUConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @model_validator(mode="after")
    def validate_cross_level_contract(self) -> "CSSFConfig":
        if self.qa.enabled and not self.maqaoa.enabled:
            raise ValueError(
                "The digitized-QA level requires the MA-QAOA bridge."
            )
        if self.maqaoa.enabled and not self.qaoa.enabled:
            raise ValueError(
                "The MA-QAOA decomposition requires ordinary QAOA for "
                "tied-angle regression validation."
            )
        if self.qa.surrogated_system != self.benchmark.surrogated_system:
            raise ValueError(
                "QA and benchmark sections must identify the same sole "
                "surrogated system."
            )
        return self


def default_config() -> CSSFConfig:
    """Return the strict default CSSF configuration."""

    return CSSFConfig()


__all__ = [
    "COLAB_PROJECT_ROOT",
    "ALLOWED_PEGASUS_SOLVER_PREFIXES",
    "BackendKind",
    "TrotterOrder",
    "SurrogateTarget",
    "ProjectConfig",
    "RuntimeConfig",
    "RandomConfig",
    "OPFConfig",
    "QUBOConfig",
    "QAOATeacherConfig",
    "MAQAOATeacherConfig",
    "QAConfig",
    "EmulatorConfig",
    "PegasusQPUConfig",
    "BenchmarkConfig",
    "CSSFConfig",
    "default_config",
]
