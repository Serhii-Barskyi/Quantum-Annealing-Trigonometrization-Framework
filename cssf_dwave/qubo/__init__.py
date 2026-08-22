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

"""QUBO layer of the CSSF BESS-placement framework.

The package will convert OPF-surrogate objectives and BESS placement
constraints into a deterministic binary quadratic model. This initializer
defines only stable conventions and performs no eager imports, model
construction, sampler initialization, filesystem access, or network access.
"""

from __future__ import annotations

from typing import Final


QUBO_PACKAGE_NAME: Final[str] = "cssf.qubo"
QUBO_PACKAGE_VERSION: Final[str] = "0.1.0"

QUBO_OBJECTIVE_CONVENTION: Final[str] = (
    "minimize x.T @ Q @ x + offset"
)
QUBO_MATRIX_STORAGE: Final[str] = "upper_triangular"
QUBO_VARIABLE_DOMAIN: Final[tuple[int, int]] = (0, 1)

PLACEMENT_VARIABLE_PREFIX: Final[str] = "bess_bus"
AUXILIARY_VARIABLE_PREFIX: Final[str] = "aux"
SLACK_VARIABLE_PREFIX: Final[str] = "slack"

OBJECTIVE_COMPONENTS: Final[tuple[str, ...]] = (
    "opf_surrogate_cost",
    "voltage_violation",
    "thermal_violation",
    "placement_cardinality",
    "capacity_consistency",
)

REQUIRED_MODEL_FIELDS: Final[tuple[str, ...]] = (
    "linear",
    "quadratic",
    "offset",
    "variable_order",
)

DEFAULT_PENALTY_SCALE: Final[float] = 10.0
DEFAULT_ZERO_TOLERANCE: Final[float] = 1.0e-12

SUPPORTED_EXPORT_FORMATS: Final[tuple[str, ...]] = (
    "dimod_bqm",
    "qubo_dict",
    "dense_matrix",
)


__all__ = [
    "QUBO_PACKAGE_NAME",
    "QUBO_PACKAGE_VERSION",
    "QUBO_OBJECTIVE_CONVENTION",
    "QUBO_MATRIX_STORAGE",
    "QUBO_VARIABLE_DOMAIN",
    "PLACEMENT_VARIABLE_PREFIX",
    "AUXILIARY_VARIABLE_PREFIX",
    "SLACK_VARIABLE_PREFIX",
    "OBJECTIVE_COMPONENTS",
    "REQUIRED_MODEL_FIELDS",
    "DEFAULT_PENALTY_SCALE",
    "DEFAULT_ZERO_TOLERANCE",
    "SUPPORTED_EXPORT_FORMATS",
]
