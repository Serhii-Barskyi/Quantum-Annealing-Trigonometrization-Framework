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

"""Exact quality-reference MILP for the CSSF BESS placement QUBO.

HiGHS is used here only as a classical solution-quality reference.  The
module does not compare wall-clock time and does not introduce a time limit.
It solves the same bus-indexed objective and exact-cardinality feasible set as
:class:`qubo.builder.BESSPlacementQUBO`.

HiGHS does not solve mixed-integer nonconvex QUBO objectives directly.  Every
active binary product ``x_i * x_j`` is therefore replaced by a continuous
auxiliary variable ``y_ij`` and its exact binary McCormick hull::

    y_ij <= x_i
    y_ij <= x_j
    y_ij >= x_i + x_j - 1
    0 <= y_ij <= 1

When ``x_i`` and ``x_j`` are binary, these constraints force
``y_ij = x_i * x_j`` exactly.  The cardinality penalty is not optimized as a
proxy: the physical placement constraint ``sum(x) = units_to_place`` is added
explicitly, while the original objective model is preserved coefficient for
coefficient.  The returned solution is independently re-evaluated by the
project QUBO implementation before it is accepted.

``highspy`` is imported lazily only inside :func:`solve_bess_with_highs`.
Importing this module never initializes HiGHS and has no GPU, QPU, Ocean, OPF,
dataset, or filesystem side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import NDArray

from baselines import (
    HIGHS_ROLE,
    HIGHS_SOLVER_NAME,
    RUNTIME_ROLE,
    SURROGATED_SYSTEM,
)
from opf.bess_constraints import BESSPlacement
from qubo.builder import BESSPlacementQUBO


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
INTEGER_DTYPE: Final[np.dtype[np.int8]] = np.dtype(np.int8)
INDEX_DTYPE: Final[np.dtype[np.int32]] = np.dtype(np.int32)

DEFAULT_RANDOM_SEED: Final[int] = 0
DEFAULT_THREADS: Final[int] = 1
DEFAULT_MIP_REL_GAP: Final[float] = 0.0
DEFAULT_MIP_ABS_GAP: Final[float] = 0.0
DEFAULT_INTEGRALITY_TOLERANCE: Final[float] = 1.0e-6
DEFAULT_AUDIT_TOLERANCE: Final[float] = 1.0e-8

LINEARIZATION_KIND: Final[str] = "exact_binary_mccormick_milp"
OBJECTIVE_KIND: Final[str] = "same_bess_objective"
CONSTRAINT_KIND: Final[str] = "same_exact_cardinality"


class HighsBaselineError(ValueError):
    """Raised when the exact HiGHS quality baseline is invalid."""


class HighsUnavailableError(ImportError):
    """Raised when the optional ``highspy`` runtime is unavailable."""


def _finite_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise HighsBaselineError(f"{name} must be finite.")
    return normalized


def _nonnegative_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized < 0.0:
        raise HighsBaselineError(f"{name} must be non-negative.")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if normalized <= 0.0:
        raise HighsBaselineError(f"{name} must be strictly positive.")
    return normalized


def _json_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    source = {} if metadata is None else dict(metadata)
    forbidden_fragments = (
        "token",
        "password",
        "secret",
        "credential",
        "api_key",
        "apikey",
    )

    for key in source:
        normalized_key = str(key).strip().lower()
        if any(fragment in normalized_key for fragment in forbidden_fragments):
            raise HighsBaselineError(
                "metadata must not contain secrets or credential fields."
            )

    try:
        encoded = json.dumps(
            source,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HighsBaselineError(
            "metadata must be JSON-serializable and contain no NaN."
        ) from exc

    return MappingProxyType(json.loads(encoded))


def _readonly_binary_vector(
    values: NDArray[np.float64] | NDArray[np.int8],
    *,
    expected_size: int,
    tolerance: float,
) -> NDArray[np.int8]:
    array = np.asarray(values, dtype=REAL_DTYPE).reshape(-1)

    if array.size != expected_size:
        raise HighsBaselineError(
            "HiGHS returned an unexpected number of primary variables: "
            f"expected {expected_size}, received {array.size}."
        )
    if not np.all(np.isfinite(array)):
        raise HighsBaselineError("HiGHS returned non-finite variable values.")

    close_zero = np.abs(array) <= tolerance
    close_one = np.abs(array - 1.0) <= tolerance
    if not np.all(close_zero | close_one):
        bad = int(np.flatnonzero(~(close_zero | close_one))[0])
        raise HighsBaselineError(
            "HiGHS returned a non-binary primary variable at position "
            f"{bad}: {array[bad]!r}."
        )

    result = np.where(close_one, 1, 0).astype(INTEGER_DTYPE, copy=False)
    result = np.ascontiguousarray(result, dtype=INTEGER_DTYPE)
    result.setflags(write=False)
    return result


def _scaled_audit_tolerance(
    *values: float,
    base_tolerance: float,
) -> float:
    scale = max(1.0, *(abs(float(value)) for value in values))
    return max(base_tolerance, 32.0 * np.finfo(np.float64).eps * scale)


@dataclass(frozen=True, slots=True)
class HighsSolveConfig:
    """Deterministic, quality-only HiGHS configuration.

    No time-limit or matched-budget field exists by design.  The solver must
    reach ``HighsModelStatus.kOptimal`` before a result is accepted.
    """

    random_seed: int = DEFAULT_RANDOM_SEED
    threads: int = DEFAULT_THREADS
    presolve: str = "on"
    parallel: str = "off"
    output_flag: bool = False
    mip_rel_gap: float = DEFAULT_MIP_REL_GAP
    mip_abs_gap: float = DEFAULT_MIP_ABS_GAP
    integrality_tolerance: float = DEFAULT_INTEGRALITY_TOLERANCE
    audit_tolerance: float = DEFAULT_AUDIT_TOLERANCE
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed,
            int,
        ):
            raise TypeError("random_seed must be an integer.")
        if not 0 <= self.random_seed <= 2_147_483_647:
            raise HighsBaselineError(
                "random_seed must lie in [0, 2147483647]."
            )
        if isinstance(self.threads, bool) or not isinstance(self.threads, int):
            raise TypeError("threads must be an integer.")
        if self.threads < 1:
            raise HighsBaselineError("threads must be at least one.")
        if not isinstance(self.presolve, str):
            raise TypeError("presolve must be a string.")
        if not isinstance(self.parallel, str):
            raise TypeError("parallel must be a string.")
        if not isinstance(self.output_flag, bool):
            raise TypeError("output_flag must be boolean.")

        presolve = self.presolve.strip().lower()
        parallel = self.parallel.strip().lower()
        if presolve not in {"off", "choose", "on"}:
            raise HighsBaselineError(
                "presolve must be one of 'off', 'choose', or 'on'."
            )
        if parallel != "off":
            raise HighsBaselineError(
                "parallel must be 'off' for deterministic quality reference."
            )

        relative_gap = _nonnegative_float(
            self.mip_rel_gap,
            name="mip_rel_gap",
        )
        absolute_gap = _nonnegative_float(
            self.mip_abs_gap,
            name="mip_abs_gap",
        )
        if relative_gap != 0.0 or absolute_gap != 0.0:
            raise HighsBaselineError(
                "mip_rel_gap and mip_abs_gap must both be zero; the quality "
                "reference must require certified optimality."
            )

        integrality_tolerance = _positive_float(
            self.integrality_tolerance,
            name="integrality_tolerance",
        )
        audit_tolerance = _positive_float(
            self.audit_tolerance,
            name="audit_tolerance",
        )

        object.__setattr__(self, "presolve", presolve)
        object.__setattr__(self, "parallel", parallel)
        object.__setattr__(self, "mip_rel_gap", relative_gap)
        object.__setattr__(self, "mip_abs_gap", absolute_gap)
        object.__setattr__(
            self,
            "integrality_tolerance",
            integrality_tolerance,
        )
        object.__setattr__(self, "audit_tolerance", audit_tolerance)
        object.__setattr__(self, "metadata", _json_metadata(self.metadata))

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 configuration fingerprint."""

        payload = {
            "random_seed": self.random_seed,
            "threads": self.threads,
            "presolve": self.presolve,
            "parallel": self.parallel,
            "output_flag": self.output_flag,
            "mip_rel_gap": self.mip_rel_gap,
            "mip_abs_gap": self.mip_abs_gap,
            "integrality_tolerance": self.integrality_tolerance,
            "audit_tolerance": self.audit_tolerance,
            "metadata": dict(self.metadata),
        }
        digest = hashlib.sha256()
        digest.update(b"CSSF-HighsSolveConfig-v1\0")
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class HighsProductTerm:
    """One exact auxiliary representation of ``x_i * x_j``."""

    first_index: int
    second_index: int
    coefficient: float

    def __post_init__(self) -> None:
        for name in ("first_index", "second_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise HighsBaselineError(f"{name} must be non-negative.")
        if self.first_index >= self.second_index:
            raise HighsBaselineError(
                "Product terms must satisfy first_index < second_index."
            )
        object.__setattr__(
            self,
            "coefficient",
            _finite_float(self.coefficient, name="coefficient"),
        )


@dataclass(frozen=True, slots=True)
class HighsLinearizedBESSModel:
    """Pure, solver-independent exact MILP linearization manifest."""

    variable_order: tuple[str, ...]
    linear_costs: tuple[float, ...]
    product_terms: tuple[HighsProductTerm, ...]
    units_to_place: int
    objective_offset: float
    source_fingerprint: str

    def __post_init__(self) -> None:
        if not self.variable_order:
            raise HighsBaselineError("variable_order must not be empty.")
        if len(set(self.variable_order)) != len(self.variable_order):
            raise HighsBaselineError("variable_order must be unique.")
        if len(self.linear_costs) != len(self.variable_order):
            raise HighsBaselineError(
                "linear_costs length must equal variable_order length."
            )
        normalized_costs = tuple(
            _finite_float(value, name=f"linear_costs[{index}]")
            for index, value in enumerate(self.linear_costs)
        )
        if isinstance(self.units_to_place, bool) or not isinstance(
            self.units_to_place,
            int,
        ):
            raise TypeError("units_to_place must be an integer.")
        if not 1 <= self.units_to_place < len(self.variable_order):
            raise HighsBaselineError(
                "units_to_place must define a non-empty strict subset."
            )
        for term in self.product_terms:
            if not isinstance(term, HighsProductTerm):
                raise TypeError("product_terms must contain HighsProductTerm.")
            if term.second_index >= len(self.variable_order):
                raise HighsBaselineError(
                    "product term index exceeds the primary variable count."
                )
        if len(
            {
                (term.first_index, term.second_index)
                for term in self.product_terms
            }
        ) != len(self.product_terms):
            raise HighsBaselineError("product_terms must contain unique pairs.")

        offset = _finite_float(self.objective_offset, name="objective_offset")
        fingerprint = str(self.source_fingerprint).strip().lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise HighsBaselineError(
                "source_fingerprint must be a lowercase SHA-256 digest."
            )

        object.__setattr__(self, "linear_costs", normalized_costs)
        object.__setattr__(self, "objective_offset", offset)
        object.__setattr__(self, "source_fingerprint", fingerprint)

    @property
    def n_primary_variables(self) -> int:
        return len(self.variable_order)

    @property
    def n_product_variables(self) -> int:
        return len(self.product_terms)

    @property
    def n_columns(self) -> int:
        return self.n_primary_variables + self.n_product_variables

    @property
    def n_rows(self) -> int:
        return 1 + 3 * self.n_product_variables

    def linearized_objective(
        self,
        primary_values: NDArray[np.float64] | NDArray[np.int8],
        product_values: NDArray[np.float64] | None = None,
    ) -> float:
        """Evaluate the linearized objective including its constant offset."""

        primary = np.asarray(primary_values, dtype=REAL_DTYPE).reshape(-1)
        if primary.size != self.n_primary_variables:
            raise HighsBaselineError(
                "primary_values length does not match the linearized model."
            )
        if not np.all(np.isfinite(primary)):
            raise HighsBaselineError("primary_values contain non-finite values.")

        if product_values is None:
            products = np.asarray(
                [
                    primary[term.first_index] * primary[term.second_index]
                    for term in self.product_terms
                ],
                dtype=REAL_DTYPE,
            )
        else:
            products = np.asarray(product_values, dtype=REAL_DTYPE).reshape(-1)
            if products.size != self.n_product_variables:
                raise HighsBaselineError(
                    "product_values length does not match the linearized model."
                )
            if not np.all(np.isfinite(products)):
                raise HighsBaselineError(
                    "product_values contain non-finite values."
                )

        return float(
            self.objective_offset
            + primary @ np.asarray(self.linear_costs, dtype=REAL_DTYPE)
            + sum(
                term.coefficient * products[index]
                for index, term in enumerate(self.product_terms)
            )
        )

    def fingerprint(self) -> str:
        """Return a deterministic fingerprint of the exact MILP mapping."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-HighsLinearizedBESSModel-v1\0")
        digest.update(self.source_fingerprint.encode("ascii"))
        digest.update(
            json.dumps(
                self.variable_order,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(
            np.asarray(self.linear_costs, dtype=REAL_DTYPE).tobytes(order="C")
        )
        digest.update(
            np.asarray(
                [
                    (term.first_index, term.second_index)
                    for term in self.product_terms
                ],
                dtype=np.int64,
            ).tobytes(order="C")
        )
        digest.update(
            np.asarray(
                [term.coefficient for term in self.product_terms],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        digest.update(
            np.asarray(
                [self.units_to_place, self.objective_offset],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class HighsBESSResult:
    """Audited certified-optimal HiGHS quality-reference result."""

    selected_sample: NDArray[np.int8]
    placement: BESSPlacement
    objective_value: float
    combined_qubo_energy: float
    solver_objective_value: float
    model_status: str
    solver_version: str
    source_fingerprint: str
    linearization_fingerprint: str
    config_fingerprint: str
    certified_optimal: bool
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        sample = np.ascontiguousarray(
            np.asarray(self.selected_sample, dtype=INTEGER_DTYPE).reshape(-1),
            dtype=INTEGER_DTYPE,
        )
        if sample.size == 0 or not np.all((sample == 0) | (sample == 1)):
            raise HighsBaselineError(
                "selected_sample must be a non-empty binary vector."
            )
        sample.setflags(write=False)
        if not isinstance(self.placement, BESSPlacement):
            raise TypeError("placement must be BESSPlacement.")
        if not isinstance(self.certified_optimal, bool):
            raise TypeError("certified_optimal must be boolean.")
        if not self.certified_optimal:
            raise HighsBaselineError(
                "HiGHS result must be certified optimal for quality reference."
            )

        for name in (
            "objective_value",
            "combined_qubo_energy",
            "solver_objective_value",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name),
            )
        for name in (
            "model_status",
            "solver_version",
            "source_fingerprint",
            "linearization_fingerprint",
            "config_fingerprint",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise HighsBaselineError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)

        object.__setattr__(self, "selected_sample", sample)
        object.__setattr__(self, "metadata", _json_metadata(self.metadata))

    def fingerprint(self) -> str:
        """Return a deterministic fingerprint excluding diagnostic runtime."""

        digest = hashlib.sha256()
        digest.update(b"CSSF-HighsBESSResult-v1\0")
        digest.update(self.selected_sample.tobytes(order="C"))
        digest.update(
            np.asarray(
                [
                    self.objective_value,
                    self.combined_qubo_energy,
                    self.solver_objective_value,
                ],
                dtype=REAL_DTYPE,
            ).tobytes(order="C")
        )
        for value in (
            self.model_status,
            self.solver_version,
            self.source_fingerprint,
            self.linearization_fingerprint,
            self.config_fingerprint,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        digest.update(
            json.dumps(
                dict(self.metadata),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


def linearize_bess_placement_qubo(
    problem: BESSPlacementQUBO,
) -> HighsLinearizedBESSModel:
    """Build the exact solver-independent MILP representation.

    Only the original BESS objective is linearized.  Exact placement
    cardinality is represented as a hard MILP row rather than through the QUBO
    penalty.  Consequently the feasible objective and constraints are exactly
    the same as the project BESS problem, without optimizing an altered proxy.
    """

    if not isinstance(problem, BESSPlacementQUBO):
        raise TypeError("problem must be BESSPlacementQUBO.")

    objective = problem.objective_model
    terms: list[HighsProductTerm] = []
    for first in range(objective.n_variables):
        for second in range(first + 1, objective.n_variables):
            coefficient = float(objective.quadratic[first, second])
            if coefficient != 0.0:
                terms.append(
                    HighsProductTerm(
                        first_index=first,
                        second_index=second,
                        coefficient=coefficient,
                    )
                )

    return HighsLinearizedBESSModel(
        variable_order=problem.variable_order,
        linear_costs=tuple(float(value) for value in objective.linear),
        product_terms=tuple(terms),
        units_to_place=problem.fleet.units_to_place,
        objective_offset=objective.offset,
        source_fingerprint=problem.fingerprint(),
    )


def _import_highspy() -> Any:
    try:
        import highspy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HighsUnavailableError(
            "highspy is required only for the HiGHS quality baseline. "
            "Install the pinned project requirements in Google Colab."
        ) from exc
    return highspy


def _require_highs_ok(status: Any, highspy: Any, *, operation: str) -> None:
    if status != highspy.HighsStatus.kOk:
        raise HighsBaselineError(
            f"HiGHS operation {operation!r} failed with status {status!r}."
        )


def _build_highs_model(
    linearized: HighsLinearizedBESSModel,
    config: HighsSolveConfig,
    highspy: Any,
) -> Any:
    solver = highspy.Highs()

    options = (
        ("output_flag", config.output_flag),
        ("log_to_console", config.output_flag),
        ("random_seed", config.random_seed),
        ("threads", config.threads),
        ("parallel", config.parallel),
        ("presolve", config.presolve),
        ("mip_rel_gap", config.mip_rel_gap),
        ("mip_abs_gap", config.mip_abs_gap),
    )
    for name, value in options:
        status = solver.setOptionValue(name, value)
        _require_highs_ok(status, highspy, operation=f"setOptionValue({name})")

    costs = np.ascontiguousarray(
        [
            *linearized.linear_costs,
            *(term.coefficient for term in linearized.product_terms),
        ],
        dtype=REAL_DTYPE,
    )
    lower = np.zeros(linearized.n_columns, dtype=REAL_DTYPE)
    upper = np.ones(linearized.n_columns, dtype=REAL_DTYPE)

    status = solver.addCols(
        linearized.n_columns,
        costs,
        lower,
        upper,
        0,
        0,
        0,
        0,
    )
    _require_highs_ok(status, highspy, operation="addCols")

    for index in range(linearized.n_primary_variables):
        status = solver.changeColIntegrality(
            index,
            highspy.HighsVarType.kInteger,
        )
        _require_highs_ok(
            status,
            highspy,
            operation=f"changeColIntegrality({index})",
        )

    row_lower: list[float] = []
    row_upper: list[float] = []
    row_start: list[int] = []
    column_index: list[int] = []
    coefficient: list[float] = []

    row_start.append(len(column_index))
    column_index.extend(range(linearized.n_primary_variables))
    coefficient.extend([1.0] * linearized.n_primary_variables)
    cardinality = float(linearized.units_to_place)
    row_lower.append(cardinality)
    row_upper.append(cardinality)

    infinity = float(highspy.kHighsInf)
    product_start = linearized.n_primary_variables
    for product_position, term in enumerate(linearized.product_terms):
        product_index = product_start + product_position

        row_start.append(len(column_index))
        column_index.extend((term.first_index, product_index))
        coefficient.extend((-1.0, 1.0))
        row_lower.append(-infinity)
        row_upper.append(0.0)

        row_start.append(len(column_index))
        column_index.extend((term.second_index, product_index))
        coefficient.extend((-1.0, 1.0))
        row_lower.append(-infinity)
        row_upper.append(0.0)

        row_start.append(len(column_index))
        column_index.extend(
            (term.first_index, term.second_index, product_index)
        )
        coefficient.extend((-1.0, -1.0, 1.0))
        row_lower.append(-1.0)
        row_upper.append(infinity)

    status = solver.addRows(
        linearized.n_rows,
        np.asarray(row_lower, dtype=REAL_DTYPE),
        np.asarray(row_upper, dtype=REAL_DTYPE),
        len(column_index),
        np.asarray(row_start, dtype=INDEX_DTYPE),
        np.asarray(column_index, dtype=INDEX_DTYPE),
        np.asarray(coefficient, dtype=REAL_DTYPE),
    )
    _require_highs_ok(status, highspy, operation="addRows")

    return solver


def solve_bess_with_highs(
    problem: BESSPlacementQUBO,
    *,
    config: HighsSolveConfig | None = None,
) -> HighsBESSResult:
    """Solve the exact BESS quality-reference MILP with HiGHS.

    The function accepts only a certified-optimal solution.  It does not set a
    time limit and does not use runtime as a comparison metric.  The primary
    placement, auxiliary products, cardinality, original objective, and full
    penalized QUBO energy are independently audited before returning.
    """

    if not isinstance(problem, BESSPlacementQUBO):
        raise TypeError("problem must be BESSPlacementQUBO.")
    solve_config = HighsSolveConfig() if config is None else config
    if not isinstance(solve_config, HighsSolveConfig):
        raise TypeError("config must be HighsSolveConfig or None.")

    linearized = linearize_bess_placement_qubo(problem)
    highspy = _import_highspy()
    solver = _build_highs_model(linearized, solve_config, highspy)

    run_status = solver.run()
    if run_status == highspy.HighsStatus.kError:
        raise HighsBaselineError("HiGHS failed while solving the BESS MILP.")

    model_status = solver.getModelStatus()
    status_name = str(solver.modelStatusToString(model_status)).strip()
    if model_status != highspy.HighsModelStatus.kOptimal:
        raise HighsBaselineError(
            "HiGHS quality reference requires certified optimality; "
            f"received model status {status_name!r}."
        )

    solution = solver.getSolution()
    all_values = np.asarray(solution.col_value, dtype=REAL_DTYPE).reshape(-1)
    if all_values.size != linearized.n_columns:
        raise HighsBaselineError(
            "HiGHS solution width differs from the exact linearized model."
        )
    if not np.all(np.isfinite(all_values)):
        raise HighsBaselineError("HiGHS returned non-finite solution values.")

    primary_values = _readonly_binary_vector(
        all_values[: linearized.n_primary_variables],
        expected_size=linearized.n_primary_variables,
        tolerance=solve_config.integrality_tolerance,
    )
    product_values = all_values[linearized.n_primary_variables :]
    expected_products = np.asarray(
        [
            primary_values[term.first_index]
            * primary_values[term.second_index]
            for term in linearized.product_terms
        ],
        dtype=REAL_DTYPE,
    )
    if not np.allclose(
        product_values,
        expected_products,
        rtol=0.0,
        atol=solve_config.integrality_tolerance,
    ):
        raise HighsBaselineError(
            "HiGHS auxiliary products violate the exact binary linearization."
        )

    if not problem.is_feasible(primary_values):
        raise HighsBaselineError(
            "HiGHS solution violates the exact BESS cardinality."
        )
    placement = problem.decode(primary_values)
    breakdown = problem.energy_breakdown(primary_values)
    objective_value = float(breakdown["objective"])
    combined_energy = float(breakdown["total"])

    solver_raw_objective = _finite_float(
        solver.getObjectiveValue(),
        name="solver objective",
    )
    solver_objective = solver_raw_objective + linearized.objective_offset
    linearized_objective = linearized.linearized_objective(
        primary_values,
        product_values,
    )
    tolerance = _scaled_audit_tolerance(
        objective_value,
        solver_objective,
        linearized_objective,
        base_tolerance=solve_config.audit_tolerance,
    )
    if not math.isclose(
        solver_objective,
        linearized_objective,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise HighsBaselineError(
            "HiGHS objective disagrees with its exact linearized model."
        )
    if not math.isclose(
        objective_value,
        linearized_objective,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise HighsBaselineError(
            "HiGHS objective disagrees with the original BESS objective."
        )

    solver_version_method = getattr(solver, "version", None)
    solver_version = (
        str(solver_version_method()).strip()
        if callable(solver_version_method)
        else "unknown"
    )
    result_metadata = {
        **dict(solve_config.metadata),
        "solver": HIGHS_SOLVER_NAME,
        "solver_role": HIGHS_ROLE,
        "runtime_role": RUNTIME_ROLE,
        "surrogated_system": SURROGATED_SYSTEM,
        "objective_kind": OBJECTIVE_KIND,
        "constraint_kind": CONSTRAINT_KIND,
        "linearization_kind": LINEARIZATION_KIND,
        "same_objective": True,
        "same_constraints": True,
        "wall_clock_competition": False,
        "time_limit_set": False,
        "certified_optimal": True,
        "n_primary_variables": linearized.n_primary_variables,
        "n_product_variables": linearized.n_product_variables,
        "n_milp_columns": linearized.n_columns,
        "n_milp_rows": linearized.n_rows,
    }

    return HighsBESSResult(
        selected_sample=primary_values,
        placement=placement,
        objective_value=objective_value,
        combined_qubo_energy=combined_energy,
        solver_objective_value=solver_objective,
        model_status=status_name,
        solver_version=solver_version,
        source_fingerprint=problem.fingerprint(),
        linearization_fingerprint=linearized.fingerprint(),
        config_fingerprint=solve_config.fingerprint(),
        certified_optimal=True,
        metadata=result_metadata,
    )


__all__ = [
    "REAL_DTYPE",
    "INTEGER_DTYPE",
    "INDEX_DTYPE",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_THREADS",
    "DEFAULT_MIP_REL_GAP",
    "DEFAULT_MIP_ABS_GAP",
    "DEFAULT_INTEGRALITY_TOLERANCE",
    "DEFAULT_AUDIT_TOLERANCE",
    "LINEARIZATION_KIND",
    "OBJECTIVE_KIND",
    "CONSTRAINT_KIND",
    "HighsBaselineError",
    "HighsUnavailableError",
    "HighsSolveConfig",
    "HighsProductTerm",
    "HighsLinearizedBESSModel",
    "HighsBESSResult",
    "linearize_bess_placement_qubo",
    "solve_bess_with_highs",
]
