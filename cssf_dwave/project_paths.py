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

"""Centralized filesystem paths for the CSSF-QA-D-Wave project.

The Google Colab project root is intentionally fixed and must not be
overridden through environment variables:

    /content/drive/MyDrive/cssf_dwave

Importing this module has no filesystem side effects. Directories are
created only through :func:`ensure_directories`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# Fixed Google Colab project root
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path("/content/drive/MyDrive/cssf_dwave")

# Source-code and configuration directories
CONFIG_DIR: Final[Path] = PROJECT_ROOT / "config"
CORE_DIR: Final[Path] = PROJECT_ROOT / "core"
SPECTRAL_DIR: Final[Path] = PROJECT_ROOT / "spectral"
OPF_DIR: Final[Path] = PROJECT_ROOT / "opf"
QUBO_DIR: Final[Path] = PROJECT_ROOT / "qubo"
QAOA_DIR: Final[Path] = PROJECT_ROOT / "qaoa"
MAQAOA_DIR: Final[Path] = PROJECT_ROOT / "maqaoa"
QA_DIR: Final[Path] = PROJECT_ROOT / "qa"
DWAVE_BACKEND_DIR: Final[Path] = PROJECT_ROOT / "dwave_backend"
BESS_DIR: Final[Path] = PROJECT_ROOT / "bess"
BASELINES_DIR: Final[Path] = PROJECT_ROOT / "baselines"
PIPELINE_DIR: Final[Path] = PROJECT_ROOT / "pipeline"
TESTS_DIR: Final[Path] = PROJECT_ROOT / "tests"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

# Data directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

# Runtime and generated-artifact directories
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
CHECKPOINTS_DIR: Final[Path] = PROJECT_ROOT / "checkpoints"
CACHE_DIR: Final[Path] = PROJECT_ROOT / "cache"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
EMBEDDINGS_DIR: Final[Path] = PROJECT_ROOT / "embeddings"
SAMPLESETS_DIR: Final[Path] = PROJECT_ROOT / "samplesets"
SOLVER_METADATA_DIR: Final[Path] = PROJECT_ROOT / "solver_metadata"

# Frequently used files
BASE_CONFIG_PATH: Final[Path] = CONFIG_DIR / "base.yaml"
CASE300_CONFIG_PATH: Final[Path] = CONFIG_DIR / "case300.yaml"
EMULATOR_GPU_CONFIG_PATH: Final[Path] = CONFIG_DIR / "emulator_gpu.yaml"
PEGASUS_QPU_CONFIG_PATH: Final[Path] = CONFIG_DIR / "pegasus_qpu.yaml"
FROZEN_MANIFEST_PATH: Final[Path] = CORE_DIR / "frozen_manifest.json"

# Directories that may be created for runtime use.
RUNTIME_DIRECTORIES: Final[tuple[Path, ...]] = (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    CACHE_DIR,
    ARTIFACTS_DIR,
    EMBEDDINGS_DIR,
    SAMPLESETS_DIR,
    SOLVER_METADATA_DIR,
)


def project_path(*parts: str | Path) -> Path:
    """Return a normalized path strictly contained in ``PROJECT_ROOT``.

    Parameters
    ----------
    *parts:
        Relative path components below ``PROJECT_ROOT``.

    Returns
    -------
    pathlib.Path
        Absolute normalized path below the fixed project root.

    Raises
    ------
    ValueError
        If an absolute component is supplied or if the resolved path escapes
        the project root through ``..`` traversal.
    """
    candidate = PROJECT_ROOT

    for part in parts:
        component = Path(part)
        if component.is_absolute():
            raise ValueError(
                f"Absolute path components are forbidden: {component!s}"
            )
        candidate = candidate / component

    normalized_root = PROJECT_ROOT.resolve(strict=False)
    normalized_candidate = candidate.resolve(strict=False)

    if normalized_candidate != normalized_root and normalized_root not in (
        normalized_candidate.parents
    ):
        raise ValueError(
            f"Path escapes PROJECT_ROOT: {normalized_candidate!s}"
        )

    return normalized_candidate


def ensure_project_root(*, create: bool = False) -> Path:
    """Validate the fixed project root and optionally create it.

    This function should be called only after Google Drive has been mounted
    in Colab.

    Parameters
    ----------
    create:
        When ``True``, create the project root recursively if it does not
        exist. When ``False``, require it to exist.

    Returns
    -------
    pathlib.Path
        The fixed project root.

    Raises
    ------
    FileNotFoundError
        If the project root does not exist and ``create=False``.
    NotADirectoryError
        If the path exists but is not a directory.
    """
    if create:
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            "CSSF project root does not exist. Mount Google Drive and create "
            f"the directory: {PROJECT_ROOT}"
        )

    if not PROJECT_ROOT.is_dir():
        raise NotADirectoryError(
            f"CSSF project root is not a directory: {PROJECT_ROOT}"
        )

    return PROJECT_ROOT


def ensure_directories(
    directories: Iterable[Path] = RUNTIME_DIRECTORIES,
) -> tuple[Path, ...]:
    """Create validated project directories and return them.

    Every directory must be located below ``PROJECT_ROOT``. The function
    refuses to create paths outside the project tree.
    """
    ensure_project_root(create=True)

    created: list[Path] = []
    normalized_root = PROJECT_ROOT.resolve(strict=False)

    for directory in directories:
        normalized = Path(directory).resolve(strict=False)

        if normalized != normalized_root and normalized_root not in (
            normalized.parents
        ):
            raise ValueError(
                f"Refusing to create a directory outside PROJECT_ROOT: "
                f"{normalized}"
            )

        normalized.mkdir(parents=True, exist_ok=True)
        created.append(normalized)

    return tuple(created)


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "CORE_DIR",
    "SPECTRAL_DIR",
    "OPF_DIR",
    "QUBO_DIR",
    "QAOA_DIR",
    "MAQAOA_DIR",
    "QA_DIR",
    "DWAVE_BACKEND_DIR",
    "BESS_DIR",
    "BASELINES_DIR",
    "PIPELINE_DIR",
    "TESTS_DIR",
    "NOTEBOOKS_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "OUTPUTS_DIR",
    "RESULTS_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "CHECKPOINTS_DIR",
    "CACHE_DIR",
    "ARTIFACTS_DIR",
    "EMBEDDINGS_DIR",
    "SAMPLESETS_DIR",
    "SOLVER_METADATA_DIR",
    "BASE_CONFIG_PATH",
    "CASE300_CONFIG_PATH",
    "EMULATOR_GPU_CONFIG_PATH",
    "PEGASUS_QPU_CONFIG_PATH",
    "FROZEN_MANIFEST_PATH",
    "RUNTIME_DIRECTORIES",
    "project_path",
    "ensure_project_root",
    "ensure_directories",
]
