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

"""Configuration package for CSSF-QA-D-Wave.

The package contains validated configuration schemas and YAML loaders for the
Google Colab project rooted at:

    /content/drive/MyDrive/cssf_dwave

This module intentionally performs no eager imports so that partially created
project stages remain importable while the framework is assembled file by file.
"""

from __future__ import annotations

from typing import Final


CONFIG_PACKAGE_NAME: Final[str] = "cssf.config"
CONFIG_PACKAGE_VERSION: Final[str] = "0.1.0"
PROJECT_ROOT_LITERAL: Final[str] = "/content/drive/MyDrive/cssf_dwave"


__all__ = [
    "CONFIG_PACKAGE_NAME",
    "CONFIG_PACKAGE_VERSION",
    "PROJECT_ROOT_LITERAL",
]
