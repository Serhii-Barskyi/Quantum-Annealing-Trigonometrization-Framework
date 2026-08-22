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

"""Frozen mathematical core package for CSSF.

The theoretical implementation files

    core/gcv.py
    core/csnn_t.py

are inherited from the original CSSF/Aizenberg formulation and must remain
byte-for-byte unchanged after their hashes are registered in
``core/frozen_manifest.json``.

This package initializer intentionally performs no eager imports. That policy:

* prevents accidental execution while the project is assembled file by file;
* avoids circular imports;
* preserves independent hash verification of the frozen modules;
* keeps adapters and extensions outside the immutable mathematical core.
"""

from __future__ import annotations

from typing import Final


CORE_PACKAGE_NAME: Final[str] = "cssf.core"
CORE_PACKAGE_VERSION: Final[str] = "0.1.0"

FROZEN_MODULES: Final[tuple[str, ...]] = (
    "gcv.py",
    "csnn_t.py",
)

FROZEN_MANIFEST_FILENAME: Final[str] = "frozen_manifest.json"

CORE_MUTATION_POLICY: Final[str] = (
    "core/gcv.py and core/csnn_t.py are immutable; extensions must use "
    "core/csnn_t_adapter.py without modifying the frozen implementation."
)


__all__ = [
    "CORE_PACKAGE_NAME",
    "CORE_PACKAGE_VERSION",
    "FROZEN_MODULES",
    "FROZEN_MANIFEST_FILENAME",
    "CORE_MUTATION_POLICY",
]
