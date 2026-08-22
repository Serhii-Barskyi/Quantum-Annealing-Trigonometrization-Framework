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

"""QAOA layer of the CSSF BESS-placement framework.

The package converts the solver-independent placement QUBO into an Ising cost
Hamiltonian and evaluates standard QAOA with deterministic statevector
simulation in Google Colab. This initializer contains only stable conventions
and performs no eager imports, simulator initialization, optimization, file
access, or network access.
"""

from __future__ import annotations

from typing import Final


QAOA_PACKAGE_NAME: Final[str] = "cssf.qaoa"
QAOA_PACKAGE_VERSION: Final[str] = "0.1.0"

QAOA_ALGORITHM_NAME: Final[str] = "CSNN-T^QAOA"
QAOA_OBJECTIVE_SENSE: Final[str] = "minimize"
QAOA_VARIABLE_DOMAIN: Final[tuple[int, int]] = (0, 1)
QAOA_SPIN_DOMAIN: Final[tuple[int, int]] = (-1, 1)

QUBO_TO_SPIN_CONVENTION: Final[str] = "x_i = (1 - z_i) / 2"
COST_HAMILTONIAN_BASIS: Final[str] = "Z"
MIXER_HAMILTONIAN_BASIS: Final[str] = "X"

STATEVECTOR_BACKEND_CLASS: Final[str] = "qiskit_aer.AerSimulator"
STATEVECTOR_METHOD: Final[str] = "statevector"
STATEVECTOR_DEVICE: Final[str] = "GPU"

PARAMETER_FAMILIES: Final[tuple[str, ...]] = (
    "gamma",
    "beta",
)

DEFAULT_REPETITIONS: Final[int] = 1
DEFAULT_INITIAL_GAMMA: Final[float] = 0.1
DEFAULT_INITIAL_BETA: Final[float] = 0.1
DEFAULT_SHOTS_FOR_SAMPLING: Final[int] = 4096
DEFAULT_PROBABILITY_TOLERANCE: Final[float] = 1.0e-12

SUPPORTED_EXPECTATION_MODES: Final[tuple[str, ...]] = (
    "exact_statevector",
    "sampled_counts",
)

REQUIRED_RESULT_FIELDS: Final[tuple[str, ...]] = (
    "optimal_parameters",
    "expected_energy",
    "best_sample",
    "best_sample_energy",
    "probabilities",
)


__all__ = [
    "QAOA_PACKAGE_NAME",
    "QAOA_PACKAGE_VERSION",
    "QAOA_ALGORITHM_NAME",
    "QAOA_OBJECTIVE_SENSE",
    "QAOA_VARIABLE_DOMAIN",
    "QAOA_SPIN_DOMAIN",
    "QUBO_TO_SPIN_CONVENTION",
    "COST_HAMILTONIAN_BASIS",
    "MIXER_HAMILTONIAN_BASIS",
    "STATEVECTOR_BACKEND_CLASS",
    "STATEVECTOR_METHOD",
    "STATEVECTOR_DEVICE",
    "PARAMETER_FAMILIES",
    "DEFAULT_REPETITIONS",
    "DEFAULT_INITIAL_GAMMA",
    "DEFAULT_INITIAL_BETA",
    "DEFAULT_SHOTS_FOR_SAMPLING",
    "DEFAULT_PROBABILITY_TOLERANCE",
    "SUPPORTED_EXPECTATION_MODES",
    "REQUIRED_RESULT_FIELDS",
]
