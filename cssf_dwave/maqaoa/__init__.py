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

"""Multi-angle QAOA layer of the CSSF BESS-placement framework.

The package implements CSNN-T^MA-QAOA for solver-independent placement QUBOs.
Each non-identity Ising cost term receives an independent cost angle in every
layer, and each binary variable receives an independent mixer angle in every
layer. Exact statevectors are executed only through Qiskit Aer on an NVIDIA GPU
inside Google Colab.

This initializer exposes stable package conventions only. It performs no eager
Qiskit imports, simulator initialization, optimization, file access, or network
access.
"""

from __future__ import annotations

from typing import Final


MAQAOA_PACKAGE_NAME: Final[str] = "cssf.maqaoa"
MAQAOA_PACKAGE_VERSION: Final[str] = "0.1.0"

MAQAOA_ALGORITHM_NAME: Final[str] = "CSNN-T^MA-QAOA"
MAQAOA_OBJECTIVE_SENSE: Final[str] = "minimize"
MAQAOA_VARIABLE_DOMAIN: Final[tuple[int, int]] = (0, 1)
MAQAOA_SPIN_DOMAIN: Final[tuple[int, int]] = (-1, 1)

QUBO_TO_SPIN_CONVENTION: Final[str] = "x_i = (1 - z_i) / 2"
COST_HAMILTONIAN_BASIS: Final[str] = "Z"
MIXER_HAMILTONIAN_BASIS: Final[str] = "X"

STATEVECTOR_BACKEND_CLASS: Final[str] = "qiskit_aer.AerSimulator"
STATEVECTOR_METHOD: Final[str] = "statevector"
STATEVECTOR_DEVICE: Final[str] = "GPU"
STATEVECTOR_PRECISION: Final[str] = "double"
COLAB_FREE_STATEVECTOR_QUBIT_LIMIT: Final[int] = 22

COST_ANGLE_FAMILY: Final[str] = "gamma"
MIXER_ANGLE_FAMILY: Final[str] = "beta"
PARAMETER_FAMILIES: Final[tuple[str, ...]] = (
    COST_ANGLE_FAMILY,
    MIXER_ANGLE_FAMILY,
)

COST_PARAMETER_GRANULARITY: Final[str] = (
    "one angle per non-identity Ising Z/ZZ term per layer"
)
MIXER_PARAMETER_GRANULARITY: Final[str] = (
    "one angle per qubit per layer"
)
PARAMETER_FLAT_ORDER: Final[tuple[str, ...]] = (
    "gamma[layer, cost_term]",
    "beta[layer, qubit]",
)
QISKIT_QUBIT_ORDER: Final[str] = "little_endian"

IDENTITY_OFFSET_POLICY: Final[str] = (
    "exclude identity offset from trainable cost angles and retain it "
    "exactly in energy evaluation"
)
ZERO_COEFFICIENT_POLICY: Final[str] = (
    "exclude zero-valued Ising terms from the trainable parameter layout"
)
TERM_ORDER_POLICY: Final[str] = (
    "linear Z terms in variable order, followed by upper-triangular ZZ "
    "terms in lexicographic variable-index order"
)

DEFAULT_REPETITIONS: Final[int] = 1
DEFAULT_INITIAL_COST_ANGLE: Final[float] = 0.1
DEFAULT_INITIAL_MIXER_ANGLE: Final[float] = 0.1
DEFAULT_PROBABILITY_TOLERANCE: Final[float] = 1.0e-12
DEFAULT_PARAMETER_TOLERANCE: Final[float] = 1.0e-12

SUPPORTED_EXPECTATION_MODES: Final[tuple[str, ...]] = (
    "exact_statevector",
)
SUPPORTED_PARAMETER_SHARING_POLICIES: Final[tuple[str, ...]] = (
    "none",
)

REQUIRED_RESULT_FIELDS: Final[tuple[str, ...]] = (
    "optimal_cost_angles",
    "optimal_mixer_angles",
    "expected_energy",
    "energy_variance",
    "most_probable_sample",
    "most_probable_energy",
    "ground_probability",
    "probabilities",
)


def parameter_count(
    *,
    repetitions: int,
    cost_term_count: int,
    n_qubits: int,
) -> int:
    """Return the exact number of trainable MA-QAOA parameters.

    The layout contains ``repetitions * cost_term_count`` independent cost
    angles and ``repetitions * n_qubits`` independent mixer angles.
    """

    for name, value in (
        ("repetitions", repetitions),
        ("cost_term_count", cost_term_count),
        ("n_qubits", n_qubits),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 1:
            raise ValueError(f"{name} must be positive.")

    return repetitions * (cost_term_count + n_qubits)


__all__ = [
    "MAQAOA_PACKAGE_NAME",
    "MAQAOA_PACKAGE_VERSION",
    "MAQAOA_ALGORITHM_NAME",
    "MAQAOA_OBJECTIVE_SENSE",
    "MAQAOA_VARIABLE_DOMAIN",
    "MAQAOA_SPIN_DOMAIN",
    "QUBO_TO_SPIN_CONVENTION",
    "COST_HAMILTONIAN_BASIS",
    "MIXER_HAMILTONIAN_BASIS",
    "STATEVECTOR_BACKEND_CLASS",
    "STATEVECTOR_METHOD",
    "STATEVECTOR_DEVICE",
    "STATEVECTOR_PRECISION",
    "COLAB_FREE_STATEVECTOR_QUBIT_LIMIT",
    "COST_ANGLE_FAMILY",
    "MIXER_ANGLE_FAMILY",
    "PARAMETER_FAMILIES",
    "COST_PARAMETER_GRANULARITY",
    "MIXER_PARAMETER_GRANULARITY",
    "PARAMETER_FLAT_ORDER",
    "QISKIT_QUBIT_ORDER",
    "IDENTITY_OFFSET_POLICY",
    "ZERO_COEFFICIENT_POLICY",
    "TERM_ORDER_POLICY",
    "DEFAULT_REPETITIONS",
    "DEFAULT_INITIAL_COST_ANGLE",
    "DEFAULT_INITIAL_MIXER_ANGLE",
    "DEFAULT_PROBABILITY_TOLERANCE",
    "DEFAULT_PARAMETER_TOLERANCE",
    "SUPPORTED_EXPECTATION_MODES",
    "SUPPORTED_PARAMETER_SHARING_POLICIES",
    "REQUIRED_RESULT_FIELDS",
    "parameter_count",
]
