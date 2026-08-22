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

"""Spectral utilities for the multi-level CSSF hierarchy.

This package defines target transformations, toric/Fourier frequency supports,
complex feature matrices, and residual-surrogate composition for:

* CSNN-T^OPF;
* CSNN-T^QAOA;
* CSNN-T^MA-QAOA;
* CSNN-T^digitized-QA;
* a future Pegasus hardware-residual level.

The frozen mathematical primitive remains in ``core/gcv.py`` and
``core/csnn_t.py``. This package must never modify or monkey-patch it.

The initializer intentionally performs no eager imports so that spectral
modules can be added and tested one file at a time.
"""

from __future__ import annotations

from typing import Final


SPECTRAL_PACKAGE_NAME: Final[str] = "cssf.spectral"
SPECTRAL_PACKAGE_VERSION: Final[str] = "0.1.0"

SUPPORTED_SURROGATE_LEVELS: Final[tuple[str, ...]] = (
    "opf",
    "qaoa",
    "ma_qaoa",
    "digitized_qa",
    "hardware_residual",
)

FROZEN_CORE_MODULES: Final[tuple[str, ...]] = (
    "core/gcv.py",
    "core/csnn_t.py",
)

QUANTUM_WALK_ENABLED: Final[bool] = False


__all__ = [
    "SPECTRAL_PACKAGE_NAME",
    "SPECTRAL_PACKAGE_VERSION",
    "SUPPORTED_SURROGATE_LEVELS",
    "FROZEN_CORE_MODULES",
    "QUANTUM_WALK_ENABLED",
]
