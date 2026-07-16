"""Atomic benchmark checkpoints and actively bounded subprocess execution."""

from __future__ import annotations

import ctypes
import gc
import importlib
import json
import os
import platform
import time
from collections.abc import Callable, Mapping
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter, process_time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BENCHMARK_SCHEMA_VERSION = "v0.5.3"
MEMORY_SCOPE = "isolated_subprocess_peak"

StageStatus = Literal["completed", "timeout", "failed", "skipped"]
MeasurementKind = Literal["measured", "estimated", "unavailable"]


class BenchmarkValue(BaseModel):
    """A value whose provenance cannot be confused in reports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: MeasurementKind
    value: float | None = None
    unit: str = Field(min_length=1)
    method: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_kind_and_value(self) -> BenchmarkValue:
        if self.kind == "unavailable" and self.value is not None:
            raise ValueError("unavailable benchmark values must not contain a value")
        if self.kind != "unavailable" and self.value is None:
            raise ValueError("measured and estimated benchmark values require a value")
        return self


class IsolatedProcessMetrics(BaseModel):
    """Wall, CPU, GC and process-local peak memory captured by one child."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_scope: Literal["isolated_subprocess_peak"] = MEMORY_SCOPE
    peak_memory_mb: float | None = Field(default=None, ge=0)
    memory_error: str | None = None
    wall_clock_seconds: float = Field(ge=0)
    cpu_seconds: float = Field(ge=0)
    gc_collection_counts: tuple[int, int, int]
    process_exit_status: int
    timed_out: bool = False


class IsolatedStageOutcome(BaseModel):
    """Parent-observed result of one actively bounded stage process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(min_length=1)
    status: StageStatus
    checkpoint_path: str
    elapsed_seconds: float = Field(ge=0)
    process_exit_status: int | None
    reused_checkpoint: bool = False


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Write bytes beside the destination and atomically replace it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Persist canonical UTF-8 JSON using an atomic same-directory replacement."""
    content = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        ).encode("utf-8")
        + b"\n"
    )
    return atomic_write_bytes(path, content)


def load_checkpoint(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load a checkpoint and reject schema or identity/hash mismatches."""
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"benchmark checkpoint must contain an object: {target}")
    if data.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"benchmark checkpoint schema mismatch: {target}")
    for key, expected in (expected_identity or {}).items():
        if expected is not None and data.get(key) != expected:
            raise ValueError(f"benchmark checkpoint identity mismatch for {key}: {target}")
    return data


def _peak_process_memory_mb() -> tuple[float | None, str | None]:
    """Read this process's historical peak without failing the benchmark."""
    try:
        if os.name == "nt":

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = (
                    ("cb", ctypes.c_ulong),
                    ("page_fault_count", ctypes.c_ulong),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                )

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ProcessMemoryCounters),
                ctypes.c_ulong,
            )
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            handle = kernel32.GetCurrentProcess()
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
            return counters.peak_working_set_size / (1024 * 1024), None
        resource = importlib.import_module("resource")
        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        divisor = 1024 if platform.system() != "Darwin" else 1024 * 1024
        return peak_rss / divisor, None
    except Exception as error:  # pragma: no cover - platform API failure path
        return None, f"{type(error).__name__}: {error}"


def _resolve_worker(worker_path: str) -> Callable[..., Mapping[str, object]]:
    module_name, separator, function_name = worker_path.partition(":")
    if not separator:
        raise ValueError("worker path must use module:function syntax")
    worker = getattr(importlib.import_module(module_name), function_name)
    if not callable(worker):
        raise TypeError(f"benchmark worker is not callable: {worker_path}")
    return worker


def benchmark_probe_worker(
    *,
    delay_seconds: float = 0.0,
    value: str = "ok",
) -> Mapping[str, object]:
    """Minimal deterministic worker used to verify active timeout semantics."""
    if delay_seconds < 0:
        raise ValueError("probe delay must be non-negative")
    time.sleep(delay_seconds)
    return {"probe_value": value}


def _isolated_worker_entry(
    worker_path: str,
    worker_kwargs: dict[str, object],
    checkpoint_path: str,
    checkpoint_identity: dict[str, object],
) -> None:
    wall_started = perf_counter()
    cpu_started = process_time()
    gc_before = tuple(item["collections"] for item in gc.get_stats())
    try:
        result = dict(_resolve_worker(worker_path)(**worker_kwargs))
        status: StageStatus = "completed"
        error: str | None = None
        exit_status = 0
    except Exception as exception:  # pragma: no cover - asserted through parent artifact
        result = {}
        status = "failed"
        error = f"{type(exception).__name__}: {exception}"
        exit_status = 1
    peak_memory, memory_error = _peak_process_memory_mb()
    gc_after = tuple(item["collections"] for item in gc.get_stats())
    metrics = IsolatedProcessMetrics(
        peak_memory_mb=peak_memory,
        memory_error=memory_error,
        wall_clock_seconds=perf_counter() - wall_started,
        cpu_seconds=process_time() - cpu_started,
        gc_collection_counts=tuple(
            int(after - before) for before, after in zip(gc_before, gc_after, strict=True)
        ),
        process_exit_status=exit_status,
    )
    atomic_write_json(
        checkpoint_path,
        {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            **checkpoint_identity,
            "status": status,
            "error": error,
            "execution": metrics.model_dump(mode="json"),
            **result,
        },
    )
    if exit_status:
        raise RuntimeError(error)


def run_isolated_stage(
    *,
    stage: str,
    worker_path: str,
    worker_kwargs: Mapping[str, object],
    checkpoint_path: str | Path,
    checkpoint_identity: Mapping[str, object],
    timeout_seconds: float,
) -> IsolatedStageOutcome:
    """Run one path in a spawned child and actively terminate it at the deadline."""
    if timeout_seconds <= 0:
        raise ValueError("stage timeout must be positive")
    target = Path(checkpoint_path)
    if target.exists():
        checkpoint = load_checkpoint(target, expected_identity=checkpoint_identity)
        if checkpoint.get("status") == "completed":
            return IsolatedStageOutcome(
                stage=stage,
                status="skipped",
                checkpoint_path=str(target),
                elapsed_seconds=0.0,
                process_exit_status=0,
                reused_checkpoint=True,
            )

    context = get_context("spawn")
    process = context.Process(
        target=_isolated_worker_entry,
        args=(
            worker_path,
            dict(worker_kwargs),
            str(target),
            dict(checkpoint_identity),
        ),
        name=f"dlt-benchmark-{stage}",
    )
    started = perf_counter()
    process.start()
    process.join(timeout_seconds)
    elapsed = perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():  # pragma: no cover - terminate should stop a Python child
            process.kill()
            process.join(5)
        existing_completed = False
        if target.exists():
            existing = load_checkpoint(target, expected_identity=checkpoint_identity)
            existing_completed = existing.get("status") == "completed"
        if not existing_completed:
            atomic_write_json(
                target,
                {
                    "schema_version": BENCHMARK_SCHEMA_VERSION,
                    **dict(checkpoint_identity),
                    "status": "timeout",
                    "error": f"stage exceeded active timeout of {timeout_seconds:.3f} seconds",
                    "execution": IsolatedProcessMetrics(
                        wall_clock_seconds=elapsed,
                        cpu_seconds=0.0,
                        gc_collection_counts=(0, 0, 0),
                        process_exit_status=int(process.exitcode or -1),
                        timed_out=True,
                    ).model_dump(mode="json"),
                },
            )
        return IsolatedStageOutcome(
            stage=stage,
            status="timeout",
            checkpoint_path=str(target),
            elapsed_seconds=elapsed,
            process_exit_status=process.exitcode,
        )

    status: StageStatus = "failed"
    if target.exists():
        checkpoint = load_checkpoint(target, expected_identity=checkpoint_identity)
        status = str(checkpoint.get("status"))  # type: ignore[assignment]
    return IsolatedStageOutcome(
        stage=stage,
        status=status,
        checkpoint_path=str(target),
        elapsed_seconds=elapsed,
        process_exit_status=process.exitcode,
    )
