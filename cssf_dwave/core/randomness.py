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

"""Deterministic random-seed utilities for CSSF experiments.

The module has no import-time side effects. Random generators are created only
when explicitly requested.

Statevector and sampler backends must still receive their own explicit seeds:

* Qiskit Aer GPU: ``seed_simulator`` and ``seed_transpiler``;
* SQA GPU: sampler-specific seed;
* D-Wave QPU: gauge and read configuration recorded separately.

This module does not silently seed or reconfigure any backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Final

import numpy as np
from numpy.random import Generator, SeedSequence

from core.types import SeedBundle, UINT32_MAX


DERIVATION_PERSONALIZATION: Final[bytes] = b"CSSF-SEED-v1"


class RandomnessError(ValueError):
    """Raised when a seed or deterministic stream name is invalid."""


def validate_seed(seed: int, *, name: str = "seed") -> int:
    """Validate and return one unsigned 32-bit seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"{name} must be an integer.")
    if not 0 <= seed <= UINT32_MAX:
        raise RandomnessError(
            f"{name} must lie in [0, {UINT32_MAX}]."
        )
    return seed


def normalize_stream_name(stream_name: str) -> str:
    """Normalize and validate a deterministic stream name."""

    if not isinstance(stream_name, str):
        raise TypeError("stream_name must be a string.")

    normalized = stream_name.strip()

    if not normalized:
        raise RandomnessError("stream_name must not be empty.")

    return normalized


def derive_seed(
    parent_seed: int,
    stream_name: str,
    *,
    index: int = 0,
) -> int:
    """Derive a stable unsigned 32-bit child seed.

    The result is independent of Python's randomized hash function and remains
    stable across processes and Google Colab runtime restarts.
    """

    validated_parent = validate_seed(parent_seed, name="parent_seed")
    normalized_name = normalize_stream_name(stream_name)

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer.")
    if index < 0:
        raise RandomnessError("index must be non-negative.")

    digest = hashlib.blake2s(
        digest_size=16,
        person=DERIVATION_PERSONALIZATION[:8],
    )
    digest.update(validated_parent.to_bytes(4, byteorder="big"))
    digest.update(index.to_bytes(8, byteorder="big", signed=False))
    digest.update(normalized_name.encode("utf-8"))

    return int.from_bytes(
        digest.digest()[:4],
        byteorder="big",
        signed=False,
    )


def make_generator(seed: int) -> Generator:
    """Create an independent NumPy ``Generator`` from one validated seed."""

    return np.random.default_rng(validate_seed(seed))


def make_named_generator(
    parent_seed: int,
    stream_name: str,
    *,
    index: int = 0,
) -> Generator:
    """Create a deterministic named NumPy generator."""

    return make_generator(
        derive_seed(
            parent_seed,
            stream_name,
            index=index,
        )
    )


def make_seed_sequence(seed: int) -> SeedSequence:
    """Create a NumPy ``SeedSequence`` from one validated seed."""

    return SeedSequence(validate_seed(seed))


def spawn_generators(
    seed: int,
    count: int,
) -> tuple[Generator, ...]:
    """Spawn statistically independent deterministic NumPy generators."""

    validated_seed = validate_seed(seed)

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer.")
    if count < 1:
        raise RandomnessError("count must be positive.")

    child_sequences = SeedSequence(validated_seed).spawn(count)
    return tuple(np.random.default_rng(child) for child in child_sequences)


def seed_python_and_numpy_legacy(seed: int) -> int:
    """Seed Python ``random`` and NumPy's legacy global RNG.

    New CSSF code should prefer :func:`make_generator`. This function exists
    only for third-party libraries that still consume global random state.
    """

    validated_seed = validate_seed(seed)
    random.seed(validated_seed)
    np.random.seed(validated_seed)
    return validated_seed


@dataclass(frozen=True, slots=True)
class BackendSeeds:
    """Explicit seeds passed to quantum and optimization backends."""

    aer_simulator: int
    aer_transpiler: int
    sqa_sampler: int
    scenario_generator: int
    qaoa_training: int
    maqaoa_training: int
    qa_training: int

    def __post_init__(self) -> None:
        for field_name in (
            "aer_simulator",
            "aer_transpiler",
            "sqa_sampler",
            "scenario_generator",
            "qaoa_training",
            "maqaoa_training",
            "qa_training",
        ):
            validate_seed(getattr(self, field_name), name=field_name)

    def as_dict(self) -> dict[str, int]:
        """Return a serialization-ready mapping."""

        return {
            "aer_simulator": self.aer_simulator,
            "aer_transpiler": self.aer_transpiler,
            "sqa_sampler": self.sqa_sampler,
            "scenario_generator": self.scenario_generator,
            "qaoa_training": self.qaoa_training,
            "maqaoa_training": self.maqaoa_training,
            "qa_training": self.qa_training,
        }


def backend_seeds(bundle: SeedBundle) -> BackendSeeds:
    """Derive explicit backend seeds from the project ``SeedBundle``."""

    if not isinstance(bundle, SeedBundle):
        raise TypeError("bundle must be a SeedBundle.")

    return BackendSeeds(
        aer_simulator=derive_seed(
            bundle.qa_seed,
            "aer_simulator",
        ),
        aer_transpiler=derive_seed(
            bundle.qa_seed,
            "aer_transpiler",
        ),
        sqa_sampler=derive_seed(
            bundle.sampler_seed,
            "sqa_sampler",
        ),
        scenario_generator=derive_seed(
            bundle.scenario_seed,
            "scenario_generator",
        ),
        qaoa_training=derive_seed(
            bundle.qaoa_seed,
            "qaoa_training",
        ),
        maqaoa_training=derive_seed(
            bundle.maqaoa_seed,
            "maqaoa_training",
        ),
        qa_training=derive_seed(
            bundle.qa_seed,
            "qa_training",
        ),
    )


__all__ = [
    "DERIVATION_PERSONALIZATION",
    "RandomnessError",
    "validate_seed",
    "normalize_stream_name",
    "derive_seed",
    "make_generator",
    "make_named_generator",
    "make_seed_sequence",
    "spawn_generators",
    "seed_python_and_numpy_legacy",
    "BackendSeeds",
    "backend_seeds",
]
