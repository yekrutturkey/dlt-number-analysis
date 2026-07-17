"""Parent-process task commit transactions for resumable formal experiments."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.cohort import (
    CohortDefinitionIdentity,
    task_targets_sha256,
)
from dlt_number_analysis.experiments.identity import (
    ExperimentExecutionIdentity,
    RunContextIdentity,
    current_git_commit_sha,
)
from dlt_number_analysis.experiments.scheduler import (
    CompletedExperimentTask,
    ExperimentProcessTask,
    FailedExperimentTask,
)
from dlt_number_analysis.experiments.specs import ExperimentSpec
from dlt_number_analysis.experiments.splits import experiment_config_sha256
from dlt_number_analysis.experiments.storage_v4 import ExperimentResultStoreV4

TASK_COMMIT_SCHEMA_VERSION = "formal-task-commit-v1"
TaskCommitStatus = Literal["prepared", "completed", "failed"]
TaskCommitFaultPoint = Literal[
    "before_prepared_audit",
    "after_prepared_audit",
    "before_store_append",
    "after_store_append",
    "before_completed_audit",
]
TaskCommitFaultInjector = Callable[[TaskCommitFaultPoint], None]


class TaskCommitAudit(BaseModel):
    """Append-attempt evidence; schema-v4 CURRENT remains the completion authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["formal-task-commit-v1"] = TASK_COMMIT_SCHEMA_VERSION
    operation_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    task_id: str = Field(min_length=1)
    task_index: int = Field(ge=0)
    partition_mode: str
    seed: int
    experiment_ids: tuple[str, ...] = Field(min_length=1)
    target_issues: tuple[str, ...] = Field(min_length=1)
    task_targets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_count: int = Field(ge=0)
    observations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cohort_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_config_sha256_by_experiment: dict[str, str]
    started_at: datetime
    worker_completed_at: datetime
    commit_started_at: datetime
    commit_completed_at: datetime | None = None
    status: TaskCommitStatus
    committed_generation_by_experiment: dict[str, str] = Field(default_factory=dict)
    failure_message: str | None = None
    git_commit_sha: str
    hostname: str
    risk_disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def validate_derived_identity(self) -> TaskCommitAudit:
        if task_targets_sha256(self.target_issues) != self.task_targets_sha256:
            raise ValueError("task commit target hash differs from target issues")
        if set(self.execution_config_sha256_by_experiment) != set(self.experiment_ids):
            raise ValueError("task commit execution identities differ from experiments")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.execution_config_sha256_by_experiment.values()
        ):
            raise ValueError("task commit execution config contains an invalid SHA-256")
        if self.status == "completed" and self.commit_completed_at is None:
            raise ValueError("completed task commit audit requires completion time")
        if self.status == "failed" and not self.failure_message:
            raise ValueError("failed task commit audit requires failure message")
        return self


class TaskCommitAuditReport(BaseModel):
    """Read-only classification of task audit state against schema-v4 CURRENT."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    completed_count: int = Field(ge=0)
    prepared_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    inconsistent_count: int = Field(ge=0)
    damaged_count: int = Field(ge=0)
    inconsistent_audit_paths: tuple[str, ...]
    risk_disclaimer: str = DISCLAIMER


def _inject(injector: TaskCommitFaultInjector | None, point: TaskCommitFaultPoint) -> None:
    if injector is not None:
        injector(point)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _empty_observations_sha256() -> str:
    return _sha256_text("[]")


def _safe_task_id(task_id: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in task_id
    )
    if not safe:
        raise ValueError("task ID cannot produce an empty audit path")
    return safe


def task_commit_audit_path(root: str | Path, audit: TaskCommitAudit) -> Path:
    """Use an append-only operation file so failed attempts remain auditable."""
    return Path(root) / "task_commits" / _safe_task_id(audit.task_id) / f"{audit.operation_id}.json"


def atomic_write_task_commit_audit(path: str | Path, audit: TaskCommitAudit) -> Path:
    """Atomically replace only one audit attempt, retaining its prepared predecessor."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(audit.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        try:
            descriptor = os.open(target.parent, os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                with suppress(OSError):
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _load_observations(
    task: ExperimentProcessTask,
    completed_task: CompletedExperimentTask,
    *,
    identities_by_experiment: Mapping[str, ExperimentExecutionIdentity],
    contexts_by_experiment: Mapping[str, RunContextIdentity],
    cohort: CohortDefinitionIdentity,
) -> tuple[pd.DataFrame, str]:
    if completed_task.task_id != task.task_id:
        raise ValueError("completed task ID differs from scheduled task")
    result = completed_task.result
    if result.get("task_id") != task.task_id:
        raise ValueError("worker result task ID differs from scheduled task")
    if result.get("task_index") != task.task_index:
        raise ValueError("worker result task index differs from scheduled task")
    expected_task_hash = task.task_targets_sha256 or task_targets_sha256(task.target_issues)
    if result.get("task_targets_sha256") != expected_task_hash:
        raise ValueError("worker result target hash differs from scheduled task")
    payload = result.get("observations_json")
    if not isinstance(payload, str):
        raise ValueError("worker result is missing observations_json")
    try:
        records = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("worker observations_json is invalid") from error
    if (
        not isinstance(records, list)
        or not records
        or not all(isinstance(record, dict) for record in records)
    ):
        raise ValueError("worker observations_json must be a non-empty record list")
    frame = pd.DataFrame.from_records(records)
    required = {
        "task_id",
        "task_index",
        "task_targets_sha256",
        "target_issue",
        "experiment_id",
        "experiment_version",
        "experiment_spec_json",
        "experiment_config_sha256",
        "run_context_sha256",
        "execution_config_sha256",
        "cohort_definition_json",
        "cohort_definition_sha256",
        "expected_targets_sha256",
        "seed",
        "history_sha256",
        "evaluation_mode",
        "profile",
        "requested_portfolio_scoring_method",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"worker observations are missing fields: {sorted(missing)}")
    expected_experiments = {spec.experiment_id: spec for spec in task.experiment_specs}
    if set(frame["experiment_id"].astype(str)) != set(expected_experiments):
        raise ValueError("worker observation experiments differ from scheduled specs")
    expected_targets = set(map(str, task.target_issues))
    if set(frame["target_issue"].astype(str)) != expected_targets:
        raise ValueError("worker observation targets differ from scheduled targets")
    if not expected_targets.issubset(cohort.payload.ordered_target_issues):
        raise ValueError("scheduled targets are outside the logical cohort")
    if task.logical_cohort is not None and task.logical_cohort != cohort:
        raise ValueError("scheduled logical cohort differs from commit cohort")
    scalar_expectations: dict[str, object] = {
        "task_id": task.task_id,
        "task_index": task.task_index,
        "task_targets_sha256": expected_task_hash,
        "seed": task.seed,
        "cohort_definition_sha256": cohort.cohort_definition_sha256,
        "expected_targets_sha256": cohort.payload.expected_targets_sha256,
    }
    for column, expected in scalar_expectations.items():
        if set(frame[column].astype(str)) != {str(expected)}:
            raise ValueError(f"worker observation {column} differs from scheduled identity")
    for raw in frame["cohort_definition_json"].astype(str):
        if CohortDefinitionIdentity.model_validate_json(raw) != cohort:
            raise ValueError("worker embedded cohort differs from scheduled cohort")
    for experiment_id, spec in expected_experiments.items():
        partition = frame.loc[frame["experiment_id"].astype(str) == experiment_id]
        if len(partition) != len(expected_targets):
            raise ValueError("worker experiment partition has the wrong observation count")
        if set(partition["target_issue"].astype(str)) != expected_targets:
            raise ValueError("worker experiment partition does not cover the task targets")
        if partition["target_issue"].astype(str).duplicated().any():
            raise ValueError("worker experiment partition contains duplicate targets")
        embedded_specs = tuple(
            ExperimentSpec.model_validate_json(raw) for raw in partition["experiment_spec_json"]
        )
        if not embedded_specs or any(embedded != spec for embedded in embedded_specs):
            raise ValueError("worker embedded ExperimentSpec differs from scheduled spec")
        identity = identities_by_experiment[experiment_id]
        context = contexts_by_experiment[experiment_id]
        expected_values = {
            "experiment_version": spec.experiment_version,
            "experiment_config_sha256": experiment_config_sha256(spec),
            "run_context_sha256": context.run_context_sha256,
            "execution_config_sha256": identity.execution_config_sha256,
            "history_sha256": context.payload.history_sha256,
            "evaluation_mode": context.payload.evaluation_mode,
            "profile": context.payload.profile,
            "requested_portfolio_scoring_method": context.payload.portfolio_scoring_method,
        }
        for column, expected in expected_values.items():
            if set(partition[column].astype(str)) != {str(expected)}:
                raise ValueError(f"worker {experiment_id} partition differs in {column}")
    return frame, payload


def _current_generations_for_task(
    task: ExperimentProcessTask,
    store: ExperimentResultStoreV4,
    *,
    identities_by_experiment: Mapping[str, ExperimentExecutionIdentity],
    contexts_by_experiment: Mapping[str, RunContextIdentity],
    cohort: CohortDefinitionIdentity,
) -> dict[str, str]:
    generations: dict[str, str] = {}
    targets = set(map(str, task.target_issues))
    for spec in task.experiment_specs:
        identity = identities_by_experiment[spec.experiment_id]
        status = store.status(
            spec,
            identity,
            contexts_by_experiment[spec.experiment_id],
            cohort,
            seed=task.seed,
            expected_target_issues=cohort.payload.ordered_target_issues,
        )
        if targets.issubset(status.completed_target_issues):
            if status.current_generation_id is None:
                raise ValueError("completed task targets have no CURRENT generation")
            generations[spec.experiment_id] = status.current_generation_id
    return generations


def commit_completed_task(
    task: ExperimentProcessTask,
    completed_task: CompletedExperimentTask,
    store: ExperimentResultStoreV4,
    scheduler_checkpoint_root: str | Path,
    *,
    identities_by_experiment: Mapping[str, ExperimentExecutionIdentity],
    contexts_by_experiment: Mapping[str, RunContextIdentity],
    cohort: CohortDefinitionIdentity,
    _fault_injector: TaskCommitFaultInjector | None = None,
) -> TaskCommitAudit:
    """Validate and synchronously append one worker result before scheduler completion."""
    frame, observations_json = _load_observations(
        task,
        completed_task,
        identities_by_experiment=identities_by_experiment,
        contexts_by_experiment=contexts_by_experiment,
        cohort=cohort,
    )
    run_hashes = {
        contexts_by_experiment[spec.experiment_id].run_context_sha256
        for spec in task.experiment_specs
    }
    if len(run_hashes) != 1:
        raise ValueError("one task commit requires exactly one run context")
    now = datetime.now(UTC)
    audit = TaskCommitAudit(
        operation_id=f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex}",
        task_id=task.task_id,
        task_index=task.task_index,
        partition_mode=task.partition_mode,
        seed=task.seed,
        experiment_ids=tuple(sorted(spec.experiment_id for spec in task.experiment_specs)),
        target_issues=tuple(task.target_issues),
        task_targets_sha256=(task.task_targets_sha256 or task_targets_sha256(task.target_issues)),
        observation_count=len(frame),
        observations_sha256=_sha256_text(observations_json),
        run_context_sha256=next(iter(run_hashes)),
        cohort_definition_sha256=cohort.cohort_definition_sha256,
        execution_config_sha256_by_experiment={
            spec.experiment_id: identities_by_experiment[spec.experiment_id].execution_config_sha256
            for spec in task.experiment_specs
        },
        started_at=completed_task.worker_started_at,
        worker_completed_at=completed_task.worker_completed_at,
        commit_started_at=now,
        status="prepared",
        git_commit_sha=current_git_commit_sha(),
        hostname=socket.gethostname(),
    )
    path = task_commit_audit_path(scheduler_checkpoint_root, audit)
    _inject(_fault_injector, "before_prepared_audit")
    atomic_write_task_commit_audit(path, audit)
    _inject(_fault_injector, "after_prepared_audit")
    try:
        _inject(_fault_injector, "before_store_append")
        store.append(frame)
        _inject(_fault_injector, "after_store_append")
    except Exception as error:
        try:
            generations = _current_generations_for_task(
                task,
                store,
                identities_by_experiment=identities_by_experiment,
                contexts_by_experiment=contexts_by_experiment,
                cohort=cohort,
            )
        except (OSError, ValueError):
            generations = {}
        failed = audit.model_copy(
            update={
                "status": "failed",
                "commit_completed_at": datetime.now(UTC),
                "committed_generation_by_experiment": generations,
                "failure_message": str(error),
            }
        )
        with suppress(OSError):
            atomic_write_task_commit_audit(path, failed)
        raise
    generations = _current_generations_for_task(
        task,
        store,
        identities_by_experiment=identities_by_experiment,
        contexts_by_experiment=contexts_by_experiment,
        cohort=cohort,
    )
    if set(generations) != {spec.experiment_id for spec in task.experiment_specs}:
        raise RuntimeError("schema-v4 CURRENT does not contain every committed task target")
    completed = audit.model_copy(
        update={
            "status": "completed",
            "commit_completed_at": datetime.now(UTC),
            "committed_generation_by_experiment": generations,
        }
    )
    _inject(_fault_injector, "before_completed_audit")
    atomic_write_task_commit_audit(path, completed)
    return completed


def record_failed_task_audit(
    task: ExperimentProcessTask,
    failure: FailedExperimentTask,
    scheduler_checkpoint_root: str | Path,
    *,
    identities_by_experiment: Mapping[str, ExperimentExecutionIdentity],
    contexts_by_experiment: Mapping[str, RunContextIdentity],
    cohort: CohortDefinitionIdentity,
) -> TaskCommitAudit:
    """Persist a worker failure without claiming any schema-v4 completion."""
    now = datetime.now(UTC)
    run_hashes = {
        contexts_by_experiment[spec.experiment_id].run_context_sha256
        for spec in task.experiment_specs
    }
    audit = TaskCommitAudit(
        operation_id=f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex}",
        task_id=task.task_id,
        task_index=task.task_index,
        partition_mode=task.partition_mode,
        seed=task.seed,
        experiment_ids=tuple(sorted(spec.experiment_id for spec in task.experiment_specs)),
        target_issues=tuple(task.target_issues),
        task_targets_sha256=(task.task_targets_sha256 or task_targets_sha256(task.target_issues)),
        observation_count=0,
        observations_sha256=_empty_observations_sha256(),
        run_context_sha256=next(iter(run_hashes)),
        cohort_definition_sha256=cohort.cohort_definition_sha256,
        execution_config_sha256_by_experiment={
            spec.experiment_id: identities_by_experiment[spec.experiment_id].execution_config_sha256
            for spec in task.experiment_specs
        },
        started_at=now,
        worker_completed_at=now,
        commit_started_at=now,
        commit_completed_at=now,
        status="failed",
        failure_message=f"{failure.exception_type}: {failure.message}",
        git_commit_sha=current_git_commit_sha(),
        hostname=socket.gethostname(),
    )
    atomic_write_task_commit_audit(
        task_commit_audit_path(scheduler_checkpoint_root, audit),
        audit,
    )
    return audit


def audit_task_commits(
    scheduler_checkpoint_root: str | Path,
    *,
    completed_targets_by_partition: Mapping[tuple[str, int], set[str]],
) -> TaskCommitAuditReport:
    """Classify audits without using them to decide resume targets."""
    root = Path(scheduler_checkpoint_root) / "task_commits"
    completed = prepared = failed = damaged = 0
    inconsistent: list[str] = []
    if not root.exists():
        return TaskCommitAuditReport(
            completed_count=0,
            prepared_count=0,
            failed_count=0,
            inconsistent_count=0,
            damaged_count=0,
            inconsistent_audit_paths=(),
        )
    for path in sorted(root.glob("*/*.json")):
        try:
            audit = TaskCommitAudit.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            damaged += 1
            inconsistent.append(str(path))
            continue
        if audit.status == "completed":
            completed += 1
        elif audit.status == "prepared":
            prepared += 1
        else:
            failed += 1
        all_current = all(
            set(audit.target_issues).issubset(
                completed_targets_by_partition.get((experiment_id, audit.seed), set())
            )
            for experiment_id in audit.experiment_ids
        )
        any_current = any(
            bool(
                set(audit.target_issues).intersection(
                    completed_targets_by_partition.get((experiment_id, audit.seed), set())
                )
            )
            for experiment_id in audit.experiment_ids
        )
        if (audit.status == "completed" and not all_current) or (
            audit.status == "prepared" and any_current
        ):
            inconsistent.append(str(path))
    return TaskCommitAuditReport(
        completed_count=completed,
        prepared_count=prepared,
        failed_count=failed,
        inconsistent_count=len(inconsistent),
        damaged_count=damaged,
        inconsistent_audit_paths=tuple(inconsistent),
    )
