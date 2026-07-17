"""Deterministic ProcessPool scheduling for experiment specs or target chunks."""

from __future__ import annotations

import os
import platform
import socket
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.cohort import (
    CohortDefinitionIdentity,
    task_targets_sha256,
)
from dlt_number_analysis.experiments.shared_computation import deterministic_subseed
from dlt_number_analysis.experiments.specs import ExperimentSpec

TaskPartitionMode = Literal["experiment_spec", "target_chunk"]


def default_process_count() -> int:
    """Use all but one logical CPU, capped at eight and never below one."""
    return min(max((os.cpu_count() or 1) - 1, 1), 8)


class ExperimentProcessTask(BaseModel):
    """Pickle-safe unit partitioned by a spec group or a target chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    partition_mode: TaskPartitionMode
    experiment_specs: tuple[ExperimentSpec, ...] = Field(min_length=1)
    target_issues: tuple[str, ...] = Field(min_length=1)
    logical_cohort: CohortDefinitionIdentity | None = None
    task_index: int = Field(default=0, ge=0)
    task_targets_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    seed: int
    deterministic_subseed: int = Field(ge=0)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_task_chunk(self) -> ExperimentProcessTask:
        expected_hash = task_targets_sha256(self.target_issues)
        if self.task_targets_sha256 is not None and self.task_targets_sha256 != expected_hash:
            raise ValueError("task target hash differs from task target issues")
        if self.logical_cohort is not None:
            cohort_targets = set(self.logical_cohort.payload.ordered_target_issues)
            if not set(self.target_issues).issubset(cohort_targets):
                raise ValueError("task targets are outside the logical cohort")
        return self


class FailedExperimentTask(BaseModel):
    """Failure retained in scheduler metadata rather than silently discarded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    exception_type: str
    message: str


class CompletedExperimentTask(BaseModel):
    """Serializable worker return and measured child CPU time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    result: dict[str, JsonValue]
    cpu_seconds: float = Field(ge=0)


class ProcessSchedulerReport(BaseModel):
    """Host, timing, process count, throughput, and failed-task audit record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    process_count: int = Field(ge=1)
    hostname: str
    platform: str
    python_version: str
    started_at: datetime
    ended_at: datetime
    wall_clock_seconds: float = Field(ge=0)
    total_cpu_time_seconds: float = Field(ge=0)
    experiment_throughput_per_hour: float = Field(ge=0)
    estimated_remaining_runtime_seconds: float = Field(ge=0)
    completed_tasks: tuple[CompletedExperimentTask, ...]
    failed_tasks: tuple[FailedExperimentTask, ...]
    risk_disclaimer: str = DISCLAIMER


ProcessTaskWorker = Callable[[ExperimentProcessTask], dict[str, JsonValue]]


def _timed_worker(
    worker: ProcessTaskWorker,
    task: ExperimentProcessTask,
) -> tuple[dict[str, JsonValue], float]:
    started = process_time()
    return worker(task), process_time() - started


def build_process_tasks(
    specifications: Sequence[ExperimentSpec],
    target_issues: Sequence[str],
    *,
    master_seed: int,
    chunk_size: int = 25,
    partition_mode: TaskPartitionMode = "target_chunk",
    parameters: dict[str, JsonValue] | None = None,
    logical_cohort: CohortDefinitionIdentity | None = None,
) -> tuple[ExperimentProcessTask, ...]:
    """Create deterministic tasks that keep shared specs together for target chunks."""
    if not specifications or not target_issues:
        return ()
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    tasks: list[ExperimentProcessTask] = []
    if partition_mode == "target_chunk":
        seeds = sorted({seed for spec in specifications for seed in spec.seeds})
        for seed in seeds:
            specs = tuple(
                spec.model_copy(update={"seeds": (seed,)})
                for spec in specifications
                if seed in spec.seeds
            )
            for task_index, offset in enumerate(range(0, len(target_issues), chunk_size)):
                chunk = tuple(str(issue) for issue in target_issues[offset : offset + chunk_size])
                task_id = f"targets-{chunk[0]}-{chunk[-1]}-seed-{seed}"
                tasks.append(
                    ExperimentProcessTask(
                        task_id=task_id,
                        partition_mode=partition_mode,
                        experiment_specs=specs,
                        target_issues=chunk,
                        logical_cohort=logical_cohort,
                        task_index=task_index,
                        task_targets_sha256=task_targets_sha256(chunk),
                        seed=seed,
                        deterministic_subseed=deterministic_subseed(master_seed, task_id, seed),
                        parameters=dict(parameters or {}),
                    )
                )
    else:
        for spec in specifications:
            for seed in spec.seeds:
                for task_index, offset in enumerate(range(0, len(target_issues), chunk_size)):
                    chunk = tuple(
                        str(issue) for issue in target_issues[offset : offset + chunk_size]
                    )
                    task_id = f"{spec.experiment_id}-{chunk[0]}-{chunk[-1]}-seed-{seed}"
                    tasks.append(
                        ExperimentProcessTask(
                            task_id=task_id,
                            partition_mode=partition_mode,
                            experiment_specs=(spec.model_copy(update={"seeds": (seed,)}),),
                            target_issues=chunk,
                            logical_cohort=logical_cohort,
                            task_index=task_index,
                            task_targets_sha256=task_targets_sha256(chunk),
                            seed=seed,
                            deterministic_subseed=deterministic_subseed(master_seed, task_id, seed),
                            parameters=dict(parameters or {}),
                        )
                    )
    return tuple(tasks)


def run_process_scheduler(
    tasks: Sequence[ExperimentProcessTask],
    worker: ProcessTaskWorker,
    *,
    workers: int | None = None,
    total_planned_tasks: int | None = None,
) -> ProcessSchedulerReport:
    """Execute with ProcessPoolExecutor; Python Portfolio search never uses threads."""
    process_count = default_process_count() if workers is None else workers
    if process_count < 1:
        raise ValueError("workers must be positive")
    if total_planned_tasks is not None and total_planned_tasks < len(tasks):
        raise ValueError("total_planned_tasks cannot be below submitted task count")
    started_at = datetime.now(UTC)
    wall_started = perf_counter()
    completed: list[CompletedExperimentTask] = []
    failed: list[FailedExperimentTask] = []
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = {executor.submit(_timed_worker, worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result, cpu_seconds = future.result()
            except Exception as error:
                failed.append(
                    FailedExperimentTask(
                        task_id=task.task_id,
                        exception_type=type(error).__name__,
                        message=str(error),
                    )
                )
            else:
                completed.append(
                    CompletedExperimentTask(
                        task_id=task.task_id,
                        result=result,
                        cpu_seconds=cpu_seconds,
                    )
                )
    wall_seconds = perf_counter() - wall_started
    ended_at = datetime.now(UTC)
    completed_count = len(completed)
    throughput = 0.0 if wall_seconds == 0 else completed_count / wall_seconds * 3600
    planned = len(tasks) if total_planned_tasks is None else total_planned_tasks
    remaining = max(planned - completed_count - len(failed), 0)
    estimated_remaining = 0.0 if throughput == 0 else remaining / throughput * 3600
    return ProcessSchedulerReport(
        process_count=process_count,
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        started_at=started_at,
        ended_at=ended_at,
        wall_clock_seconds=wall_seconds,
        total_cpu_time_seconds=sum(item.cpu_seconds for item in completed),
        experiment_throughput_per_hour=throughput,
        estimated_remaining_runtime_seconds=estimated_remaining,
        completed_tasks=tuple(sorted(completed, key=lambda item: item.task_id)),
        failed_tasks=tuple(sorted(failed, key=lambda item: item.task_id)),
    )


def write_process_scheduler_report(
    report: ProcessSchedulerReport,
    path: str | Path,
) -> Path:
    """Persist scheduler metadata including failed tasks and host information."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return target
