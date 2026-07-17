"""Deterministic ProcessPool scheduling for experiment specs or target chunks."""

from __future__ import annotations

import hashlib
import json
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
    failure_stage: Literal["worker", "callback", "scheduler"] = "worker"
    exception_type: str
    message: str


class CompletedExperimentTask(BaseModel):
    """Serializable worker return and measured child CPU time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    result: dict[str, JsonValue]
    cpu_seconds: float = Field(ge=0)
    worker_started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    worker_completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CompletedExperimentTaskSummary(BaseModel):
    """Bounded scheduler metadata retained after a streamed formal commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    cpu_seconds: float = Field(ge=0)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_count: int = Field(ge=0)
    committed: bool


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
    completed_task_summaries: tuple[CompletedExperimentTaskSummary, ...] = ()
    planned_task_count: int = Field(default=0, ge=0)
    worker_completed_task_count: int = Field(default=0, ge=0)
    committed_task_count: int = Field(default=0, ge=0)
    failed_task_count: int = Field(default=0, ge=0)
    pending_task_count: int = Field(default=0, ge=0)
    first_commit_at: datetime | None = None
    last_commit_at: datetime | None = None
    mean_task_commit_seconds: float = Field(default=0.0, ge=0)
    maximum_task_commit_seconds: float = Field(default=0.0, ge=0)
    interrupted: bool = False
    resumed_from_completed_target_count: int = Field(default=0, ge=0)
    incremental_commit_enabled: bool = False
    maximum_uncommitted_task_count: int = Field(default=0, ge=0)
    risk_disclaimer: str = DISCLAIMER


ProcessTaskWorker = Callable[[ExperimentProcessTask], dict[str, JsonValue]]
CompletedTaskCallback = Callable[
    [ExperimentProcessTask, CompletedExperimentTask],
    None,
]
FailedTaskCallback = Callable[[ExperimentProcessTask, FailedExperimentTask], None]


class ProcessSchedulerFatalError(RuntimeError):
    """Fatal scheduler/callback failure carrying all already-committed progress."""

    def __init__(
        self,
        message: str,
        *,
        report: ProcessSchedulerReport,
        task_id: str | None,
        interrupted: bool,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.task_id = task_id
        self.interrupted = interrupted


def _timed_worker(
    worker: ProcessTaskWorker,
    task: ExperimentProcessTask,
) -> tuple[dict[str, JsonValue], float, datetime, datetime]:
    started_at = datetime.now(UTC)
    started = process_time()
    result = worker(task)
    return result, process_time() - started, started_at, datetime.now(UTC)


def _result_sha256(result: dict[str, JsonValue]) -> str:
    payload = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _observation_count(result: dict[str, JsonValue]) -> int:
    payload = result.get("observations_json")
    if not isinstance(payload, str):
        return 0
    try:
        records = json.loads(payload)
    except json.JSONDecodeError:
        return 0
    return len(records) if isinstance(records, list) else 0


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
    on_task_completed: CompletedTaskCallback | None = None,
    on_task_failed: FailedTaskCallback | None = None,
    retain_full_results: bool = True,
    resumed_from_completed_target_count: int = 0,
) -> ProcessSchedulerReport:
    """Synchronously commit each finished task in the parent before counting it complete."""
    process_count = default_process_count() if workers is None else workers
    if process_count < 1:
        raise ValueError("workers must be positive")
    if total_planned_tasks is not None and total_planned_tasks < len(tasks):
        raise ValueError("total_planned_tasks cannot be below submitted task count")
    started_at = datetime.now(UTC)
    wall_started = perf_counter()
    completed: list[CompletedExperimentTask] = []
    summaries: list[CompletedExperimentTaskSummary] = []
    failed: list[FailedExperimentTask] = []
    worker_completed_count = 0
    commit_durations: list[float] = []
    commit_times: list[datetime] = []
    interrupted = False
    fatal_error: BaseException | None = None
    fatal_task_id: str | None = None
    executor = ProcessPoolExecutor(max_workers=process_count)
    try:
        futures = {
            executor.submit(_timed_worker, worker, task): task for task in tasks[:process_count]
        }
        next_task_index = len(futures)
        try:
            while futures:
                future = next(as_completed(tuple(futures)))
                task = futures.pop(future)
                try:
                    result, cpu_seconds, worker_started_at, worker_completed_at = future.result()
                except Exception as error:
                    failure = FailedExperimentTask(
                        task_id=task.task_id,
                        failure_stage="worker",
                        exception_type=type(error).__name__,
                        message=str(error),
                    )
                    failed.append(failure)
                    if on_task_failed is not None:
                        try:
                            on_task_failed(task, failure)
                        except KeyboardInterrupt as callback_error:
                            interrupted = True
                            fatal_error = callback_error
                            fatal_task_id = task.task_id
                            break
                        except Exception as callback_error:
                            fatal_error = callback_error
                            fatal_task_id = task.task_id
                            failed.append(
                                FailedExperimentTask(
                                    task_id=task.task_id,
                                    failure_stage="callback",
                                    exception_type=type(callback_error).__name__,
                                    message=str(callback_error),
                                )
                            )
                            break
                    if next_task_index < len(tasks):
                        next_task = tasks[next_task_index]
                        futures[executor.submit(_timed_worker, worker, next_task)] = next_task
                        next_task_index += 1
                    continue
                worker_completed_count += 1
                item = CompletedExperimentTask(
                    task_id=task.task_id,
                    result=result,
                    cpu_seconds=cpu_seconds,
                    worker_started_at=worker_started_at,
                    worker_completed_at=worker_completed_at,
                )
                commit_started = perf_counter()
                try:
                    if on_task_completed is not None:
                        on_task_completed(task, item)
                except KeyboardInterrupt as error:
                    interrupted = True
                    fatal_error = error
                    fatal_task_id = task.task_id
                    failed.append(
                        FailedExperimentTask(
                            task_id=task.task_id,
                            failure_stage="callback",
                            exception_type=type(error).__name__,
                            message="task commit interrupted",
                        )
                    )
                    break
                except Exception as error:
                    fatal_error = error
                    fatal_task_id = task.task_id
                    failed.append(
                        FailedExperimentTask(
                            task_id=task.task_id,
                            failure_stage="callback",
                            exception_type=type(error).__name__,
                            message=str(error),
                        )
                    )
                    break
                commit_durations.append(perf_counter() - commit_started)
                commit_times.append(datetime.now(UTC))
                if retain_full_results:
                    completed.append(item)
                summaries.append(
                    CompletedExperimentTaskSummary(
                        task_id=task.task_id,
                        cpu_seconds=cpu_seconds,
                        result_sha256=_result_sha256(result),
                        observation_count=_observation_count(result),
                        committed=on_task_completed is not None,
                    )
                )
                if next_task_index < len(tasks):
                    next_task = tasks[next_task_index]
                    futures[executor.submit(_timed_worker, worker, next_task)] = next_task
                    next_task_index += 1
        except KeyboardInterrupt as error:
            interrupted = True
            fatal_error = error
        if fatal_error is not None:
            for future in futures:
                if not future.done():
                    future.cancel()
    finally:
        executor.shutdown(wait=fatal_error is None, cancel_futures=fatal_error is not None)
    wall_seconds = perf_counter() - wall_started
    ended_at = datetime.now(UTC)
    completed_count = len(summaries)
    throughput = 0.0 if wall_seconds == 0 else completed_count / wall_seconds * 3600
    planned = len(tasks) if total_planned_tasks is None else total_planned_tasks
    remaining = max(planned - completed_count, 0)
    estimated_remaining = 0.0 if throughput == 0 else remaining / throughput * 3600
    report = ProcessSchedulerReport(
        process_count=process_count,
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        started_at=started_at,
        ended_at=ended_at,
        wall_clock_seconds=wall_seconds,
        total_cpu_time_seconds=sum(item.cpu_seconds for item in summaries),
        experiment_throughput_per_hour=throughput,
        estimated_remaining_runtime_seconds=estimated_remaining,
        completed_tasks=tuple(sorted(completed, key=lambda item: item.task_id)),
        completed_task_summaries=tuple(summaries),
        failed_tasks=tuple(sorted(failed, key=lambda item: item.task_id)),
        planned_task_count=planned,
        worker_completed_task_count=worker_completed_count,
        committed_task_count=(completed_count if on_task_completed is not None else 0),
        failed_task_count=len(failed),
        pending_task_count=remaining,
        first_commit_at=commit_times[0] if commit_times else None,
        last_commit_at=commit_times[-1] if commit_times else None,
        mean_task_commit_seconds=(
            sum(commit_durations) / len(commit_durations) if commit_durations else 0.0
        ),
        maximum_task_commit_seconds=max(commit_durations, default=0.0),
        interrupted=interrupted,
        resumed_from_completed_target_count=resumed_from_completed_target_count,
        incremental_commit_enabled=on_task_completed is not None,
        maximum_uncommitted_task_count=min(process_count, len(tasks)),
    )
    if fatal_error is not None:
        raise ProcessSchedulerFatalError(
            f"scheduler stopped while committing task {fatal_task_id or 'unknown'}: {fatal_error}",
            report=report,
            task_id=fatal_task_id,
            interrupted=interrupted,
        ) from fatal_error
    return report


def write_process_scheduler_report(
    report: ProcessSchedulerReport,
    path: str | Path,
) -> Path:
    """Persist scheduler metadata including failed tasks and host information."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return target
