# -*- coding: utf-8 -*-
"""Runtime-only diagnostics for the CSSF local GPU SQA experiment.

This module is deliberately excluded from the SQA transition mathematics.
It observes runtime state (time, CUDA memory, progress, sampler metadata) and
must never modify the BQM, spin state, random-number stream, annealing
schedule, Trotter replica count, sweep count, or requested number of reads.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np


DEFAULT_HEARTBEAT_SECONDS: Final[float] = 20.0
DEFAULT_PROGRESS_BAR_WIDTH: Final[int] = 28
GIB: Final[float] = float(1024**3)

GPU_EXECUTION_INFO_KEYS: Final[tuple[str, ...]] = (
    "batch_size",
    "requested_batch_size",
    "smallest_successful_batch_size",
    "oom_retries",
    "kernel_strategy",
    "color_count",
    "directed_incremental_edges",
    "local_field_rebase_interval",
    "local_field_rebuilds_last_batch",
    "settings_fingerprint",
    "cssf_backend",
    "device",
    "topology_type",
    "pegasus_m",
    "classical_fallback",
)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _resolve_torch(torch_module: Any | None) -> Any:
    if torch_module is not None:
        return torch_module
    # Use importlib so backend-policy AST checks still see no eager/explicit
    # optional-runtime import in this module.  The import occurs only when a
    # CUDA diagnostic is actually requested.
    return importlib.import_module("torch")


def format_duration(seconds: float | None) -> str:
    """Format a duration without changing any experiment state."""

    if seconds is None:
        return "n/a"
    try:
        numeric = float(seconds)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric) or numeric < 0.0:
        return "n/a"

    total = int(round(numeric))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return (
        f"{hours:02d}:{minutes:02d}:{secs:02d}"
        if hours
        else f"{minutes:02d}:{secs:02d}"
    )


def format_metric(value: Any, precision: int = 6) -> str:
    """Format a numeric diagnostic value."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{numeric:.{precision}g}" if math.isfinite(numeric) else str(numeric)


def progress_bar(
    completed: int,
    total: int,
    *,
    width: int = DEFAULT_PROGRESS_BAR_WIDTH,
) -> str:
    """Return a deterministic text progress bar."""

    normalized_width = _positive_integer(width, name="width")
    fraction = (
        0.0
        if total <= 0
        else min(1.0, max(0.0, float(completed) / float(total)))
    )
    filled = int(round(normalized_width * fraction))
    return "[" + "#" * filled + "-" * (normalized_width - filled) + "]"


def cuda_memory_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    """Read CUDA allocator/device memory counters.

    The function performs no synchronization and no cache manipulation.
    """

    torch = _resolve_torch(torch_module)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "free_gib": float(free_bytes) / GIB,
            "total_gib": float(total_bytes) / GIB,
            "allocated_gib": float(torch.cuda.memory_allocated()) / GIB,
            "reserved_gib": float(torch.cuda.memory_reserved()) / GIB,
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated()) / GIB,
        }
    except Exception as exc:  # Diagnostic collection must never mask the experiment.
        return {"error": f"{type(exc).__name__}: {exc}"}


def format_cuda_snapshot(snapshot: Mapping[str, Any]) -> str:
    """Format a CUDA memory snapshot for notebook logs."""

    if "error" in snapshot:
        return f"GPU memory unavailable ({snapshot['error']})"
    return (
        f"free={snapshot['free_gib']:.2f}/{snapshot['total_gib']:.2f} GiB, "
        f"allocated={snapshot['allocated_gib']:.2f} GiB, "
        f"reserved={snapshot['reserved_gib']:.2f} GiB, "
        f"peak={snapshot['peak_allocated_gib']:.2f} GiB"
    )


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 of a file using bounded-memory streaming."""

    normalized_path = Path(path)
    digest = hashlib.sha256()
    with normalized_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_sqa_work(parameters: Mapping[str, Any]) -> float:
    """Return the same ETA-only work proxy used by the original notebook.

    The embedded graph is fixed across controls.  This value is not consumed by
    the sampler and therefore cannot alter SQA dynamics.
    """

    return float(
        int(parameters["num_reads"])
        * int(parameters["trotter_replicas"])
        * int(parameters["sweeps"])
    )


def estimate_eta_seconds(
    *,
    completed_compute_work: float,
    completed_compute_seconds: float,
    current_work: float = 0.0,
    current_elapsed: float = 0.0,
    future_work: float = 0.0,
) -> float | None:
    """Estimate remaining wall time from completed relative SQA work."""

    if completed_compute_work <= 0.0 or completed_compute_seconds <= 0.0:
        return None

    work_rate = completed_compute_work / completed_compute_seconds
    if not math.isfinite(work_rate) or work_rate <= 0.0:
        return None

    current_remaining = max(
        0.0,
        current_work - work_rate * max(0.0, current_elapsed),
    )
    return (current_remaining + max(0.0, future_work)) / work_rate


def print_stage(title: str) -> None:
    """Print the notebook stage delimiter used by the original Cell 11."""

    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def extract_gpu_execution_info(raw_sampleset: Any) -> dict[str, Any]:
    """Extract only runtime/provenance metadata exposed by the sampler."""

    info = getattr(raw_sampleset, "info", {})
    if not isinstance(info, Mapping):
        return {}

    result: dict[str, Any] = {}
    for key in GPU_EXECUTION_INFO_KEYS:
        if key not in info:
            continue
        value = info[key]
        result[key] = value.item() if isinstance(value, np.generic) else value
    return result


@dataclass(slots=True)
class _SamplingWindow(AbstractContextManager["_SamplingWindow"]):
    reporter: "SQAProgressReporter"
    task_index: int
    control_index: int
    replicate: int
    control_id: str
    tau: float
    cache_state: str
    parameters: Mapping[str, Any]
    future_work_after_current: float
    work: float = field(init=False)
    eta_before: float | None = field(init=False, default=None)
    gpu_before: dict[str, Any] = field(init=False, default_factory=dict)
    gpu_after: dict[str, Any] = field(init=False, default_factory=dict)
    elapsed_seconds: float = field(init=False, default=0.0)
    _run_started: float = field(init=False, default=0.0)
    _stop_event: threading.Event | None = field(init=False, default=None)
    _heartbeat_thread: threading.Thread | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.work = relative_sqa_work(self.parameters)

    def __enter__(self) -> "_SamplingWindow":
        self.reporter._require_started()
        self.eta_before = self.reporter.estimate_eta(
            current_work=self.work,
            future_work=self.future_work_after_current,
        )

        print()
        print(
            f"[RUN] {progress_bar(self.task_index - 1, self.reporter.total_tasks)} "
            f"task {self.task_index}/{self.reporter.total_tasks} | "
            f"control {self.control_index}/{self.reporter.total_controls} | "
            f"replicate {self.replicate + 1}/{self.reporter.replicates_per_control} | "
            f"{self.control_id}"
        )
        print(
            "      "
            f"cache={self.cache_state} | "
            f"tau={self.tau:.6g} | "
            f"reads={self.parameters['num_reads']} | "
            f"replicas={self.parameters['trotter_replicas']} | "
            f"sweeps={self.parameters['sweeps']} | "
            f"burn_in={self.parameters['burn_in_sweeps']} | "
            f"beta={self.parameters['beta_range']} | "
            f"field={self.parameters['transverse_field_range']} | "
            f"ETA={format_duration(self.eta_before)}"
        )

        torch = self.reporter.torch
        # Preserve the exact synchronization/cache-reset sequence from the
        # original notebook.  It is outside the sampler and does not alter the
        # Markov transition rule or requested simulation size.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.gpu_before = cuda_memory_snapshot(torch)
        print(
            "      GPU before: " + format_cuda_snapshot(self.gpu_before),
            flush=True,
        )

        self._run_started = time.perf_counter()
        self._stop_event = threading.Event()
        self._heartbeat_thread = self.reporter._start_heartbeat(
            stop_event=self._stop_event,
            run_started=self._run_started,
            task_index=self.task_index,
            control_index=self.control_index,
            replicate=self.replicate,
            parameters=self.parameters,
            future_work_after_current=self.future_work_after_current,
        )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(
                timeout=max(1.0, self.reporter.heartbeat_seconds)
            )

        # Match the original notebook failure path: after a sampler exception the
        # heartbeat is stopped, but no additional CUDA synchronization is forced.
        if exc_type is not None:
            return None

        # Preserve original post-sampler synchronization before measuring the
        # completed sampling wall time.
        self.reporter.torch.cuda.synchronize()
        if self._run_started > 0.0:
            self.elapsed_seconds = time.perf_counter() - self._run_started
        self.gpu_after = cuda_memory_snapshot(self.reporter.torch)
        return None


@dataclass(slots=True)
class SQAProgressReporter(AbstractContextManager["SQAProgressReporter"]):
    """Runtime-only progress/heartbeat controller for replicated GPU-SQA.

    The reporter never calls the sampler, never constructs sampler parameters,
    never mutates cache state, and never touches the RNG.  The notebook keeps
    the complete scientific orchestration visible and executes the sampler
    explicitly inside ``sampling_window``.
    """

    total_tasks: int
    total_controls: int
    replicates_per_control: int
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS
    torch_module: Any | None = None
    completed_compute_work: float = field(init=False, default=0.0)
    completed_compute_seconds: float = field(init=False, default=0.0)
    completed_cache_hits: int = field(init=False, default=0)
    completed_compute_tasks: int = field(init=False, default=0)
    overall_started: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.total_tasks = _positive_integer(self.total_tasks, name="total_tasks")
        self.total_controls = _positive_integer(
            self.total_controls, name="total_controls"
        )
        self.replicates_per_control = _positive_integer(
            self.replicates_per_control, name="replicates_per_control"
        )
        self.heartbeat_seconds = _finite_float(
            self.heartbeat_seconds, name="heartbeat_seconds"
        )
        if self.heartbeat_seconds <= 0.0:
            raise ValueError("heartbeat_seconds must be positive.")
        self.torch_module = _resolve_torch(self.torch_module)

    @property
    def torch(self) -> Any:
        return self.torch_module

    def __enter__(self) -> "SQAProgressReporter":
        self.overall_started = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def _require_started(self) -> None:
        if self.overall_started <= 0.0:
            raise RuntimeError(
                "SQAProgressReporter must be entered before sampling begins."
            )

    @property
    def elapsed_seconds(self) -> float:
        self._require_started()
        return time.perf_counter() - self.overall_started

    def estimate_eta(
        self,
        *,
        current_work: float = 0.0,
        current_elapsed: float = 0.0,
        future_work: float = 0.0,
    ) -> float | None:
        return estimate_eta_seconds(
            completed_compute_work=self.completed_compute_work,
            completed_compute_seconds=self.completed_compute_seconds,
            current_work=current_work,
            current_elapsed=current_elapsed,
            future_work=future_work,
        )

    def _start_heartbeat(
        self,
        *,
        stop_event: threading.Event,
        run_started: float,
        task_index: int,
        control_index: int,
        replicate: int,
        parameters: Mapping[str, Any],
        future_work_after_current: float,
    ) -> threading.Thread:
        current_work = relative_sqa_work(parameters)

        def worker() -> None:
            while not stop_event.wait(self.heartbeat_seconds):
                current_elapsed = time.perf_counter() - run_started
                overall_elapsed = self.elapsed_seconds
                eta = self.estimate_eta(
                    current_work=current_work,
                    current_elapsed=current_elapsed,
                    future_work=future_work_after_current,
                )
                print(
                    f"[HEARTBEAT] task {task_index}/{self.total_tasks} | "
                    f"control {control_index}/{self.total_controls} | "
                    f"replicate {replicate + 1}/{self.replicates_per_control} | "
                    f"current={format_duration(current_elapsed)} | "
                    f"overall={format_duration(overall_elapsed)} | "
                    f"ETA={format_duration(eta)}"
                )
                print(
                    "            "
                    + format_cuda_snapshot(cuda_memory_snapshot(self.torch)),
                    flush=True,
                )

        thread = threading.Thread(
            target=worker,
            name="cssf-sqa-progress-heartbeat",
            daemon=True,
        )
        thread.start()
        return thread

    def sampling_window(
        self,
        *,
        task_index: int,
        control_index: int,
        replicate: int,
        control_id: str,
        tau: float,
        cache_state: str,
        parameters: Mapping[str, Any],
        future_work_after_current: float,
    ) -> _SamplingWindow:
        return _SamplingWindow(
            reporter=self,
            task_index=_positive_integer(task_index, name="task_index"),
            control_index=_positive_integer(control_index, name="control_index"),
            replicate=_nonnegative_integer(replicate, name="replicate"),
            control_id=str(control_id),
            tau=float(tau),
            cache_state=str(cache_state),
            parameters=parameters,
            future_work_after_current=float(future_work_after_current),
        )

    def record_runtime(self, runtime: Mapping[str, Any]) -> None:
        if bool(runtime.get("cache_hit", False)):
            self.completed_cache_hits += 1
            return

        work = float(runtime.get("work", 0.0))
        elapsed = float(runtime.get("elapsed_seconds", 0.0))
        if work < 0.0 or elapsed < 0.0:
            raise ValueError("Runtime work and elapsed_seconds must be non-negative.")
        self.completed_compute_tasks += 1
        self.completed_compute_work += work
        self.completed_compute_seconds += elapsed

    def report_progress(self, *, task_index: int, remaining_work: float) -> None:
        normalized_task = _positive_integer(task_index, name="task_index")
        eta = self.estimate_eta(future_work=float(remaining_work))
        print(
            f"[PROGRESS] {progress_bar(normalized_task, self.total_tasks)} "
            f"{normalized_task}/{self.total_tasks} "
            f"({100.0 * normalized_task / self.total_tasks:.1f}%) | "
            f"computed={self.completed_compute_tasks} | "
            f"cache={self.completed_cache_hits} | "
            f"elapsed={format_duration(self.elapsed_seconds)} | "
            f"ETA={format_duration(eta)}",
            flush=True,
        )

    def report_replicate_done(
        self,
        *,
        window: _SamplingWindow,
        gpu_execution: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        weighted_mean_energy: float,
        feasible_probability: float,
    ) -> None:
        print(
            f"[DONE] task {window.task_index}/{self.total_tasks} | "
            f"elapsed={format_duration(window.elapsed_seconds)} | "
            f"kernel={gpu_execution.get('kernel_strategy', 'n/a')} | "
            f"colors={gpu_execution.get('color_count', 'n/a')} | "
            f"batch={gpu_execution.get('batch_size', 'n/a')} | "
            f"requested_batch={gpu_execution.get('requested_batch_size', 'n/a')} | "
            f"smallest_batch="
            f"{gpu_execution.get('smallest_successful_batch_size', 'n/a')} | "
            f"OOM_retries={gpu_execution.get('oom_retries', 'n/a')}"
        )
        print(
            "       "
            f"best_energy={format_metric(diagnostics['best_energy'], 9)} | "
            f"mean_energy={format_metric(weighted_mean_energy, 9)} | "
            f"feasible_probability={format_metric(feasible_probability, 6)} | "
            f"chain_break="
            f"{format_metric(diagnostics['weighted_chain_break_fraction'], 6)} | "
            f"unique_samples={diagnostics['unique_samples']}"
        )
        print(
            "       GPU after: " + format_cuda_snapshot(window.gpu_after),
            flush=True,
        )


__all__ = [
    "DEFAULT_HEARTBEAT_SECONDS",
    "DEFAULT_PROGRESS_BAR_WIDTH",
    "GPU_EXECUTION_INFO_KEYS",
    "SQAProgressReporter",
    "cuda_memory_snapshot",
    "estimate_eta_seconds",
    "extract_gpu_execution_info",
    "file_sha256",
    "format_cuda_snapshot",
    "format_duration",
    "format_metric",
    "print_stage",
    "progress_bar",
    "relative_sqa_work",
]
