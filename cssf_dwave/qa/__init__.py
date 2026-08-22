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

"""Digitized-QA layer of the CSSF BESS-placement framework.

The package maps a validated annealing schedule to deterministic product-
formula circuit coordinates in the fixed direction
``QA schedule -> MA-QAOA coordinates``. Exact statevector execution is allowed
only through Qiskit Aer on an NVIDIA GPU in Google Colab and is capped at
22 qubits. CPU and silent fallbacks are forbidden.

This initializer exposes stable package conventions and lightweight contract
validators only. It performs no eager Qiskit, Aer, optimizer, filesystem,
network, emulator, or QPU operation during import.
"""

from __future__ import annotations

from typing import Final


QA_PACKAGE_NAME: Final[str] = "cssf.qa"
QA_PACKAGE_VERSION: Final[str] = "0.1.0"
QA_ALGORITHM_NAME: Final[str] = "CSNN-T^digitized-QA"

QA_TO_MAQAOA_DIRECTION: Final[str] = "qa_to_maqaoa"
QA_HAMILTONIAN_FORM: Final[str] = "H(t)=A(t)H_X+B(t)H_P"
NORMALIZED_TIME_DOMAIN: Final[tuple[float, float]] = (0.0, 1.0)
DRIVER_HAMILTONIAN_BASIS: Final[str] = "X"
PROBLEM_HAMILTONIAN_BASIS: Final[str] = "Z"

STATEVECTOR_BACKEND_CLASS: Final[str] = "qiskit_aer.AerSimulator"
STATEVECTOR_PROVIDER: Final[str] = "qiskit_aer_gpu"
STATEVECTOR_METHOD: Final[str] = "statevector"
STATEVECTOR_DEVICE: Final[str] = "GPU"
STATEVECTOR_PRECISION: Final[str] = "double"
COLAB_FREE_STATEVECTOR_QUBIT_LIMIT: Final[int] = 22
CPU_FALLBACK_ALLOWED: Final[bool] = False
SILENT_FALLBACK_ALLOWED: Final[bool] = False
QISKIT_QUBIT_ORDER: Final[str] = "little_endian"

MINIMUM_SCHEDULE_POINTS: Final[int] = 3
MINIMUM_TROTTER_SLICES: Final[int] = 2
MAXIMUM_TROTTER_SLICES: Final[int] = 4096
SUPPORTED_TROTTER_ORDERS: Final[tuple[int, int]] = (1, 2)
DEFAULT_SCHEDULE_NAME: Final[str] = "linear_forward_anneal"
DEFAULT_PROBABILITY_TOLERANCE: Final[float] = 1.0e-12

REQUIRED_RESULT_FIELDS: Final[tuple[str, ...]] = (
    "schedule_fingerprint",
    "hamiltonian_fingerprint",
    "circuit_fingerprint",
    "expected_energy",
    "energy_variance",
    "probabilities",
)


class QAContractError(ValueError):
    """Raised when a stable digitized-QA package contract is violated."""


def validate_exact_statevector_qubit_count(n_qubits: int) -> int:
    """Validate and return a qubit count allowed by the Colab GPU contract."""

    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
        raise TypeError("n_qubits must be an integer.")
    if n_qubits < 1:
        raise QAContractError("n_qubits must be positive.")
    if n_qubits > COLAB_FREE_STATEVECTOR_QUBIT_LIMIT:
        raise QAContractError(
            f"n_qubits={n_qubits} exceeds the exact Google Colab "
            f"statevector limit of {COLAB_FREE_STATEVECTOR_QUBIT_LIMIT}."
        )
    return n_qubits


def validate_trotter_contract(
    *,
    trotter_slices: int,
    trotter_order: int,
) -> tuple[int, int]:
    """Validate and return the supported product-formula configuration."""

    if isinstance(trotter_slices, bool) or not isinstance(
        trotter_slices,
        int,
    ):
        raise TypeError("trotter_slices must be an integer.")
    if not MINIMUM_TROTTER_SLICES <= trotter_slices <= MAXIMUM_TROTTER_SLICES:
        raise QAContractError(
            "trotter_slices must be in the inclusive range "
            f"[{MINIMUM_TROTTER_SLICES}, {MAXIMUM_TROTTER_SLICES}]."
        )

    if isinstance(trotter_order, bool) or not isinstance(trotter_order, int):
        raise TypeError("trotter_order must be an integer.")
    if trotter_order not in SUPPORTED_TROTTER_ORDERS:
        raise QAContractError(
            f"trotter_order must be one of {SUPPORTED_TROTTER_ORDERS}."
        )

    return trotter_slices, trotter_order


__all__ = [
    "QA_PACKAGE_NAME",
    "QA_PACKAGE_VERSION",
    "QA_ALGORITHM_NAME",
    "QA_TO_MAQAOA_DIRECTION",
    "QA_HAMILTONIAN_FORM",
    "NORMALIZED_TIME_DOMAIN",
    "DRIVER_HAMILTONIAN_BASIS",
    "PROBLEM_HAMILTONIAN_BASIS",
    "STATEVECTOR_BACKEND_CLASS",
    "STATEVECTOR_PROVIDER",
    "STATEVECTOR_METHOD",
    "STATEVECTOR_DEVICE",
    "STATEVECTOR_PRECISION",
    "COLAB_FREE_STATEVECTOR_QUBIT_LIMIT",
    "CPU_FALLBACK_ALLOWED",
    "SILENT_FALLBACK_ALLOWED",
    "QISKIT_QUBIT_ORDER",
    "MINIMUM_SCHEDULE_POINTS",
    "MINIMUM_TROTTER_SLICES",
    "MAXIMUM_TROTTER_SLICES",
    "SUPPORTED_TROTTER_ORDERS",
    "DEFAULT_SCHEDULE_NAME",
    "DEFAULT_PROBABILITY_TOLERANCE",
    "REQUIRED_RESULT_FIELDS",
    "QAContractError",
    "validate_exact_statevector_qubit_count",
    "validate_trotter_contract",
]
