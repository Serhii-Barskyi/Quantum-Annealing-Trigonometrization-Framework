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

"""Optimal-power-flow layer of the CSSF BESS-placement framework.

The package will provide deterministic IEEE/MATPOWER case loading, scenario
generation, AC/DC OPF reference calculations, BESS operating constraints, and
the CSNN-T^OPF training interface.

The frozen mathematical primitive remains in ``core/gcv.py`` and
``core/csnn_t.py``. This initializer intentionally performs no eager imports,
solver initialization, filesystem access, or network access.
"""

from __future__ import annotations

from typing import Final


OPF_PACKAGE_NAME: Final[str] = "cssf.opf"
OPF_PACKAGE_VERSION: Final[str] = "0.1.0"

SUPPORTED_CASES: Final[tuple[str, ...]] = (
    "case14",
    "case30",
    "case57",
    "case118",
    "case300",
)

SUPPORTED_FORMULATIONS: Final[tuple[str, ...]] = (
    "acopf",
    "dcopf",
)

DEFAULT_CASE: Final[str] = "case300"
DEFAULT_FORMULATION: Final[str] = "acopf"

BESS_DECISION_FIELDS: Final[tuple[str, ...]] = (
    "bus_index",
    "power_mw",
    "energy_mwh",
)

FROZEN_CORE_MODULES: Final[tuple[str, ...]] = (
    "core/gcv.py",
    "core/csnn_t.py",
)

QUANTUM_WALK_ENABLED: Final[bool] = False


__all__ = [
    "OPF_PACKAGE_NAME",
    "OPF_PACKAGE_VERSION",
    "SUPPORTED_CASES",
    "SUPPORTED_FORMULATIONS",
    "DEFAULT_CASE",
    "DEFAULT_FORMULATION",
    "BESS_DECISION_FIELDS",
    "FROZEN_CORE_MODULES",
    "QUANTUM_WALK_ENABLED",
]
