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

"""Strict AC optimal-power-flow execution through pandapower.

The wrapper solves an independent deep copy of a validated network. It calls
only ``pandapower.runopp`` and never substitutes another formulation after a
failure. Extracted numerical results are immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import importlib
import math
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from opf.case_loader import (
    LoadedPowerCase,
    PowerCaseError,
    power_case_fingerprint,
    validate_power_case,
)


REAL_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
ALLOWED_INITIALIZATIONS: Final[tuple[str, ...]] = (
    "flat",
    "pf",
    "results",
)
ALLOWED_TRAFO3W_LOSS_SIDES: Final[tuple[str, ...]] = (
    "hv",
    "mv",
    "lv",
    "star",
)


class ACOPFError(RuntimeError):
    """Base error for AC-OPF preparation, execution, or extraction."""


class ACOPFExecutionError(ACOPFError):
    """Raised when ``pandapower.runopp`` cannot complete."""


class ACOPFConvergenceError(ACOPFError):
    """Raised when the AC OPF does not converge."""


class ACOPFResultError(ACOPFError):
    """Raised when converged result tables violate the output contract."""


def _network_value(network: Any, name: str) -> Any:
    if hasattr(network, name):
        return getattr(network, name)
    if isinstance(network, Mapping):
        return network.get(name)
    try:
        return network[name]
    except (KeyError, TypeError, AttributeError):
        return None


def _table(
    network: Any,
    name: str,
    *,
    required: bool,
) -> pd.DataFrame | None:
    value = _network_value(network, name)

    if value is None:
        if required:
            raise ACOPFResultError(
                f"AC-OPF output is missing table {name!r}."
            )
        return None

    if not isinstance(value, pd.DataFrame):
        raise ACOPFResultError(
            f"AC-OPF table {name!r} must be a pandas DataFrame."
        )

    return value


def _readonly_vector(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    result = np.ascontiguousarray(
        np.asarray(values, dtype=REAL_DTYPE).reshape(-1),
        dtype=REAL_DTYPE,
    )

    if not np.all(np.isfinite(result)):
        raise ACOPFResultError(
            f"{name} contains non-finite values."
        )

    result.setflags(write=False)
    return result


def _extract_column(
    table: pd.DataFrame,
    column_name: str,
    *,
    table_name: str,
) -> NDArray[np.float64]:
    if column_name not in table.columns:
        raise ACOPFResultError(
            f"{table_name} is missing column {column_name!r}."
        )

    try:
        values = pd.to_numeric(
            table[column_name],
            errors="raise",
        ).to_numpy(dtype=REAL_DTYPE)
    except (TypeError, ValueError) as exc:
        raise ACOPFResultError(
            f"{table_name}.{column_name} must be numeric."
        ) from exc

    return _readonly_vector(
        values,
        name=f"{table_name}.{column_name}",
    )


def _validate_result_index(
    result_table: pd.DataFrame,
    input_table: pd.DataFrame,
    *,
    result_name: str,
    input_name: str,
) -> tuple[Any, ...]:
    if not result_table.index.is_unique:
        raise ACOPFResultError(
            f"{result_name} index must be unique."
        )

    if not result_table.index.equals(input_table.index):
        raise ACOPFResultError(
            f"{result_name} index must exactly match {input_name} index."
        )

    return tuple(result_table.index.tolist())


def _empty_vector() -> NDArray[np.float64]:
    result = np.empty(0, dtype=REAL_DTYPE)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ACOPFConfig:
    """Validated keyword arguments passed to ``pandapower.runopp``."""

    calculate_voltage_angles: bool = True
    check_connectivity: bool = True
    suppress_warnings: bool = True
    switch_rx_ratio: float = 2.0
    delta: float = 1.0e-10
    init: str = "flat"
    numba: bool = True
    trafo3w_losses: str = "hv"
    consider_line_temperature: bool = False
    verbose: bool = False
    require_convergence: bool = True
    verify_input_unchanged: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "calculate_voltage_angles",
            "check_connectivity",
            "suppress_warnings",
            "numba",
            "consider_line_temperature",
            "verbose",
            "require_convergence",
            "verify_input_unchanged",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean.")

        switch_rx_ratio = float(self.switch_rx_ratio)
        delta = float(self.delta)

        if not math.isfinite(switch_rx_ratio) or switch_rx_ratio <= 0.0:
            raise ACOPFError(
                "switch_rx_ratio must be finite and strictly positive."
            )
        if not math.isfinite(delta) or delta <= 0.0:
            raise ACOPFError(
                "delta must be finite and strictly positive."
            )

        initialization = str(self.init).strip().lower()
        if initialization not in ALLOWED_INITIALIZATIONS:
            raise ACOPFError(
                f"init must be one of {ALLOWED_INITIALIZATIONS}."
            )

        loss_side = str(self.trafo3w_losses).strip().lower()
        if loss_side not in ALLOWED_TRAFO3W_LOSS_SIDES:
            raise ACOPFError(
                "trafo3w_losses must be one of "
                f"{ALLOWED_TRAFO3W_LOSS_SIDES}."
            )

        object.__setattr__(self, "switch_rx_ratio", switch_rx_ratio)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "init", initialization)
        object.__setattr__(self, "trafo3w_losses", loss_side)

    def runopp_kwargs(self) -> dict[str, bool | float | str]:
        """Return the exact arguments sent to pandapower."""

        return {
            "verbose": self.verbose,
            "calculate_voltage_angles": self.calculate_voltage_angles,
            "check_connectivity": self.check_connectivity,
            "suppress_warnings": self.suppress_warnings,
            "switch_rx_ratio": self.switch_rx_ratio,
            "delta": self.delta,
            "init": self.init,
            "numba": self.numba,
            "trafo3w_losses": self.trafo3w_losses,
            "consider_line_temperature": (
                self.consider_line_temperature
            ),
        }


@dataclass(frozen=True, slots=True)
class ElementDispatch:
    """Immutable active/reactive dispatch for one element table."""

    indices: tuple[Any, ...]
    p_mw: NDArray[np.float64]
    q_mvar: NDArray[np.float64]

    def __post_init__(self) -> None:
        p_mw = _readonly_vector(self.p_mw, name="p_mw")
        q_mvar = _readonly_vector(self.q_mvar, name="q_mvar")

        if len(self.indices) != p_mw.size:
            raise ACOPFResultError(
                "indices and p_mw lengths differ."
            )
        if p_mw.shape != q_mvar.shape:
            raise ACOPFResultError(
                "p_mw and q_mvar shapes differ."
            )

        object.__setattr__(self, "indices", tuple(self.indices))
        object.__setattr__(self, "p_mw", p_mw)
        object.__setattr__(self, "q_mvar", q_mvar)


@dataclass(frozen=True, slots=True)
class ACOPFResult:
    """Immutable numerical result extracted from one AC-OPF run."""

    objective_cost: float
    input_fingerprint: str
    bus_indices: tuple[Any, ...]
    bus_vm_pu: NDArray[np.float64]
    bus_va_degree: NDArray[np.float64]
    bus_p_mw: NDArray[np.float64]
    bus_q_mvar: NDArray[np.float64]
    line_indices: tuple[Any, ...]
    line_loading_percent: NDArray[np.float64]
    trafo_indices: tuple[Any, ...]
    trafo_loading_percent: NDArray[np.float64]
    ext_grid: ElementDispatch
    gen: ElementDispatch
    sgen: ElementDispatch
    storage: ElementDispatch
    solved_network: Any

    def __post_init__(self) -> None:
        objective = float(self.objective_cost)
        if not math.isfinite(objective):
            raise ACOPFResultError(
                "objective_cost must be finite."
            )
        if len(self.input_fingerprint) != 64:
            raise ACOPFResultError(
                "input_fingerprint must be a SHA-256 digest."
            )

        bus_vectors = {
            "bus_vm_pu": _readonly_vector(
                self.bus_vm_pu,
                name="bus_vm_pu",
            ),
            "bus_va_degree": _readonly_vector(
                self.bus_va_degree,
                name="bus_va_degree",
            ),
            "bus_p_mw": _readonly_vector(
                self.bus_p_mw,
                name="bus_p_mw",
            ),
            "bus_q_mvar": _readonly_vector(
                self.bus_q_mvar,
                name="bus_q_mvar",
            ),
        }

        n_bus = len(self.bus_indices)
        for name, values in bus_vectors.items():
            if values.size != n_bus:
                raise ACOPFResultError(
                    f"{name} length must equal bus_indices length."
                )

        line_loading = _readonly_vector(
            self.line_loading_percent,
            name="line_loading_percent",
        )
        trafo_loading = _readonly_vector(
            self.trafo_loading_percent,
            name="trafo_loading_percent",
        )

        if line_loading.size != len(self.line_indices):
            raise ACOPFResultError(
                "line loading length must match line indices."
            )
        if trafo_loading.size != len(self.trafo_indices):
            raise ACOPFResultError(
                "transformer loading length must match transformer indices."
            )

        object.__setattr__(self, "objective_cost", objective)
        object.__setattr__(self, "bus_indices", tuple(self.bus_indices))
        object.__setattr__(self, "line_indices", tuple(self.line_indices))
        object.__setattr__(self, "trafo_indices", tuple(self.trafo_indices))
        object.__setattr__(self, "line_loading_percent", line_loading)
        object.__setattr__(self, "trafo_loading_percent", trafo_loading)

        for name, values in bus_vectors.items():
            object.__setattr__(self, name, values)

    @property
    def n_bus(self) -> int:
        return len(self.bus_indices)

    def bus_target_matrix(self) -> NDArray[np.float64]:
        """Return ``[vm_pu, va_degree, p_mw, q_mvar]`` per bus."""

        return np.ascontiguousarray(
            np.column_stack(
                (
                    self.bus_vm_pu,
                    self.bus_va_degree,
                    self.bus_p_mw,
                    self.bus_q_mvar,
                )
            ),
            dtype=REAL_DTYPE,
        )

    def maximum_loading_percent(self) -> float:
        """Return maximum line or transformer loading."""

        blocks = tuple(
            values
            for values in (
                self.line_loading_percent,
                self.trafo_loading_percent,
            )
            if values.size
        )
        if not blocks:
            return 0.0
        return max(float(np.max(values)) for values in blocks)


def _optional_loading(
    network: Any,
    *,
    input_name: str,
    result_name: str,
) -> tuple[tuple[Any, ...], NDArray[np.float64]]:
    input_table = _table(network, input_name, required=False)

    if input_table is None or input_table.empty:
        return tuple(), _empty_vector()

    result_table = _table(network, result_name, required=True)
    assert result_table is not None

    indices = _validate_result_index(
        result_table,
        input_table,
        result_name=result_name,
        input_name=input_name,
    )
    loading = _extract_column(
        result_table,
        "loading_percent",
        table_name=result_name,
    )
    return indices, loading


def _dispatch(
    network: Any,
    *,
    input_name: str,
    result_name: str,
) -> ElementDispatch:
    input_table = _table(network, input_name, required=False)

    if input_table is None or input_table.empty:
        return ElementDispatch(
            indices=tuple(),
            p_mw=_empty_vector(),
            q_mvar=_empty_vector(),
        )

    result_table = _table(network, result_name, required=True)
    assert result_table is not None

    indices = _validate_result_index(
        result_table,
        input_table,
        result_name=result_name,
        input_name=input_name,
    )

    return ElementDispatch(
        indices=indices,
        p_mw=_extract_column(
            result_table,
            "p_mw",
            table_name=result_name,
        ),
        q_mvar=_extract_column(
            result_table,
            "q_mvar",
            table_name=result_name,
        ),
    )


def extract_acopf_result(
    solved_network: Any,
    *,
    input_fingerprint: str,
) -> ACOPFResult:
    """Extract strict immutable output from a solved network."""

    converged_value = _network_value(
        solved_network,
        "OPF_converged",
    )
    if not isinstance(converged_value, (bool, np.bool_)):
        raise ACOPFResultError(
            "Solved network is missing boolean OPF_converged."
        )
    if not bool(converged_value):
        raise ACOPFConvergenceError(
            "pandapower reported OPF_converged=False."
        )

    try:
        objective_cost = float(
            _network_value(solved_network, "res_cost")
        )
    except (TypeError, ValueError) as exc:
        raise ACOPFResultError(
            "Solved network is missing numeric res_cost."
        ) from exc

    if not math.isfinite(objective_cost):
        raise ACOPFResultError("res_cost must be finite.")

    bus = _table(solved_network, "bus", required=True)
    res_bus = _table(solved_network, "res_bus", required=True)
    assert bus is not None
    assert res_bus is not None

    bus_indices = _validate_result_index(
        res_bus,
        bus,
        result_name="res_bus",
        input_name="bus",
    )

    line_indices, line_loading = _optional_loading(
        solved_network,
        input_name="line",
        result_name="res_line",
    )
    trafo_indices, trafo_loading = _optional_loading(
        solved_network,
        input_name="trafo",
        result_name="res_trafo",
    )

    return ACOPFResult(
        objective_cost=objective_cost,
        input_fingerprint=input_fingerprint,
        bus_indices=bus_indices,
        bus_vm_pu=_extract_column(
            res_bus,
            "vm_pu",
            table_name="res_bus",
        ),
        bus_va_degree=_extract_column(
            res_bus,
            "va_degree",
            table_name="res_bus",
        ),
        bus_p_mw=_extract_column(
            res_bus,
            "p_mw",
            table_name="res_bus",
        ),
        bus_q_mvar=_extract_column(
            res_bus,
            "q_mvar",
            table_name="res_bus",
        ),
        line_indices=line_indices,
        line_loading_percent=line_loading,
        trafo_indices=trafo_indices,
        trafo_loading_percent=trafo_loading,
        ext_grid=_dispatch(
            solved_network,
            input_name="ext_grid",
            result_name="res_ext_grid",
        ),
        gen=_dispatch(
            solved_network,
            input_name="gen",
            result_name="res_gen",
        ),
        sgen=_dispatch(
            solved_network,
            input_name="sgen",
            result_name="res_sgen",
        ),
        storage=_dispatch(
            solved_network,
            input_name="storage",
            result_name="res_storage",
        ),
        solved_network=solved_network,
    )


def run_acopf(
    network_or_case: Any | LoadedPowerCase,
    config: ACOPFConfig | None = None,
) -> ACOPFResult:
    """Run strict AC OPF on an independent deep copy."""

    run_config = ACOPFConfig() if config is None else config
    if not isinstance(run_config, ACOPFConfig):
        raise TypeError("config must be ACOPFConfig or None.")

    source_network = (
        network_or_case.network
        if isinstance(network_or_case, LoadedPowerCase)
        else network_or_case
    )

    try:
        validate_power_case(source_network)
        source_fingerprint = power_case_fingerprint(source_network)
    except PowerCaseError as exc:
        raise ACOPFError(
            f"Invalid AC-OPF input network: {exc}"
        ) from exc

    solved_network = copy.deepcopy(source_network)

    try:
        pandapower = importlib.import_module("pandapower")
    except ImportError as exc:
        raise ACOPFExecutionError(
            "pandapower is unavailable in the Google Colab runtime."
        ) from exc

    runopp = getattr(pandapower, "runopp", None)
    if runopp is None or not callable(runopp):
        raise ACOPFExecutionError(
            "pandapower.runopp is unavailable."
        )

    try:
        runopp(
            solved_network,
            **run_config.runopp_kwargs(),
        )
    except Exception as exc:
        raise ACOPFExecutionError(
            "pandapower.runopp failed; no alternate formulation was run."
        ) from exc

    converged_value = _network_value(
        solved_network,
        "OPF_converged",
    )
    converged = isinstance(
        converged_value,
        (bool, np.bool_),
    ) and bool(converged_value)

    if run_config.require_convergence and not converged:
        raise ACOPFConvergenceError(
            "AC OPF did not converge."
        )

    if run_config.verify_input_unchanged:
        try:
            solved_fingerprint = power_case_fingerprint(
                solved_network
            )
        except PowerCaseError as exc:
            raise ACOPFResultError(
                "Solved network input tables became invalid."
            ) from exc

        if solved_fingerprint != source_fingerprint:
            raise ACOPFResultError(
                "pandapower.runopp changed protected input tables."
            )

    return extract_acopf_result(
        solved_network,
        input_fingerprint=source_fingerprint,
    )


__all__ = [
    "REAL_DTYPE",
    "ALLOWED_INITIALIZATIONS",
    "ALLOWED_TRAFO3W_LOSS_SIDES",
    "ACOPFError",
    "ACOPFExecutionError",
    "ACOPFConvergenceError",
    "ACOPFResultError",
    "ACOPFConfig",
    "ElementDispatch",
    "ACOPFResult",
    "extract_acopf_result",
    "run_acopf",
]
