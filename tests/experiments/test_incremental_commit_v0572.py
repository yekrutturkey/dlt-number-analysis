"""Synthetic v0.5.7.2 immediate-commit and interruption recovery tests."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from pydantic import JsonValue

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments import command as command_module
from dlt_number_analysis.experiments.cohort import (
    CohortDefinitionIdentity,
    build_cohort_definition_identity,
)
from dlt_number_analysis.experiments.command import (
    _pending_groups,
    _reload_completion_gate,
)
from dlt_number_analysis.experiments.identity import (
    EXECUTION_IDENTITY_SCHEMA_VERSION,
    RUNNER_VERSION,
    ExperimentExecutionIdentity,
    RunContextIdentity,
    build_experiment_execution_identity,
    build_run_context_identity,
)
from dlt_number_analysis.experiments.scheduler import (
    CompletedExperimentTask,
    ExperimentProcessTask,
    ProcessSchedulerFatalError,
    ProcessSchedulerReport,
    build_process_tasks,
    run_process_scheduler,
)
from dlt_number_analysis.experiments.specs import ExperimentSpec, baseline_experiment_specs
from dlt_number_analysis.experiments.splits import experiment_config_sha256
from dlt_number_analysis.experiments.storage import STORAGE_SCHEMA_VERSION_V4
from dlt_number_analysis.experiments.storage_v4 import (
    ExperimentPartitionStatusV4,
    ExperimentResultStoreV4,
    PartitionGenerationAudit,
)
from dlt_number_analysis.experiments.task_commit import (
    TaskCommitAudit,
    audit_task_commits,
    commit_completed_task,
    record_failed_task_audit,
)


@dataclass(frozen=True)
class SyntheticEnvironment:
    specifications: tuple[ExperimentSpec, ...]
    contexts: dict[str, RunContextIdentity]
    identities: dict[str, ExperimentExecutionIdentity]
    cohort: CohortDefinitionIdentity
    store: ExperimentResultStoreV4
    audit_root: Path


def _history(count: int = 24) -> pd.DataFrame:
    start = date(2020, 1, 1)
    return pd.DataFrame(
        [
            [
                str(20001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                1,
                2,
                3,
                4,
                5 + index % 15,
                1,
                2 + index % 10,
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def _environment(tmp_path: Path, targets: tuple[str, ...]) -> SyntheticEnvironment:
    draws = _history()
    specifications = baseline_experiment_specs(seeds=(77,), phase="development")[1:3]
    contexts = {
        spec.experiment_id: build_run_context_identity(
            draws,
            data_split_spec=spec.data_split,
            phase="development",
            evaluation_mode="raw_observation",
            profile="fast",
            portfolio_scoring_method="numpy_vectorized",
            minimum_history=3,
            minimum_bank_size=500,
            maximum_bank_search_trials=80_000,
        )
        for spec in specifications
    }
    identities = {
        spec.experiment_id: build_experiment_execution_identity(contexts[spec.experiment_id], spec)
        for spec in specifications
    }
    cohort = build_cohort_definition_identity(
        targets,
        cohort_purpose="development_smoke",
        phase="development",
        cohort_id="v0572-synthetic",
        phase_target_issues=tuple(draws.iloc[:16]["issue"].astype(str)),
    )
    return SyntheticEnvironment(
        specifications=specifications,
        contexts=contexts,
        identities=identities,
        cohort=cohort,
        store=ExperimentResultStoreV4(tmp_path / "schema_v4"),
        audit_root=tmp_path / "formal_run",
    )


def _task_parameters(environment: SyntheticEnvironment) -> dict[str, JsonValue]:
    context = next(iter(environment.contexts.values()))
    return {
        "run_context_sha256": context.run_context_sha256,
        "history_sha256": context.payload.history_sha256,
        "evaluation_mode": context.payload.evaluation_mode,
        "profile": context.payload.profile,
        "portfolio_scoring_method": context.payload.portfolio_scoring_method,
        "execution_config_sha256_by_experiment": {
            experiment_id: identity.execution_config_sha256
            for experiment_id, identity in environment.identities.items()
        },
    }


def _synthetic_worker(task: ExperimentProcessTask) -> dict[str, JsonValue]:
    parameters = task.parameters
    cohort = task.logical_cohort
    if cohort is None:
        raise ValueError("synthetic worker requires a logical cohort")
    delay_by_task = parameters.get("delay_by_task", {})
    if isinstance(delay_by_task, dict):
        raw_delay = delay_by_task.get(task.task_id, 0.0)
        if isinstance(raw_delay, int | float):
            time.sleep(float(raw_delay))
    execution_hashes = cast(dict[str, str], parameters["execution_config_sha256_by_experiment"])
    rows: list[dict[str, object]] = []
    for spec in task.experiment_specs:
        for target in task.target_issues:
            rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "experiment_version": spec.experiment_version,
                    "experiment_spec_json": spec.model_dump_json(),
                    "experiment_config_sha256": experiment_config_sha256(spec),
                    "run_context_sha256": parameters["run_context_sha256"],
                    "execution_config_sha256": execution_hashes[spec.experiment_id],
                    "execution_identity_schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
                    "runner_version": RUNNER_VERSION,
                    "history_sha256": parameters["history_sha256"],
                    "evaluation_mode": parameters["evaluation_mode"],
                    "profile": parameters["profile"],
                    "requested_portfolio_scoring_method": parameters["portfolio_scoring_method"],
                    "portfolio_scoring_method": "synthetic",
                    "storage_schema_version": STORAGE_SCHEMA_VERSION_V4,
                    "identity_status": "formal_verified",
                    "formal_inference_eligible": True,
                    "phase": spec.data_split.phase,
                    "seed": task.seed,
                    "target_issue": target,
                    "data_cutoff_issue": str(int(target) - 1),
                    "cohort_definition_json": cohort.model_dump_json(),
                    "cohort_identity_schema_version": (
                        cohort.payload.cohort_identity_schema_version
                    ),
                    "cohort_purpose": cohort.payload.cohort_purpose,
                    "cohort_id": cohort.payload.cohort_id,
                    "cohort_definition_sha256": cohort.cohort_definition_sha256,
                    "expected_targets_sha256": cohort.payload.expected_targets_sha256,
                    "cohort_target_count": cohort.payload.target_count,
                    "cohort_start_issue": cohort.payload.start_issue,
                    "cohort_end_issue": cohort.payload.end_issue,
                    "task_id": task.task_id,
                    "task_index": task.task_index,
                    "task_targets_sha256": task.task_targets_sha256,
                    "task_target_count": len(task.target_issues),
                    "best_front_hits": 0,
                }
            )
    return {
        "task_id": task.task_id,
        "task_index": task.task_index,
        "task_targets_sha256": task.task_targets_sha256,
        "observations_json": json.dumps(rows, ensure_ascii=False),
        "executions": [],
    }


def _failing_worker(task: ExperimentProcessTask) -> dict[str, JsonValue]:
    raise RuntimeError(f"synthetic worker failure: {task.task_id}")


def _tasks(
    environment: SyntheticEnvironment,
    targets: tuple[str, ...],
    *,
    chunk_size: int,
    specifications: tuple[ExperimentSpec, ...] | None = None,
    delays: dict[str, float] | None = None,
) -> tuple[ExperimentProcessTask, ...]:
    tasks = build_process_tasks(
        specifications or environment.specifications,
        targets,
        master_seed=77,
        chunk_size=chunk_size,
        logical_cohort=environment.cohort,
        parameters=_task_parameters(environment),
    )
    if delays is None:
        return tasks
    return tuple(
        task.model_copy(
            update={
                "parameters": {
                    **task.parameters,
                    "delay_by_task": delays,
                }
            }
        )
        for task in tasks
    )


def _status_map(
    environment: SyntheticEnvironment,
) -> dict[tuple[str, int], ExperimentPartitionStatusV4]:
    return {
        (spec.experiment_id, 77): environment.store.status(
            spec,
            environment.identities[spec.experiment_id],
            environment.contexts[spec.experiment_id],
            environment.cohort,
            seed=77,
            expected_target_issues=environment.cohort.payload.ordered_target_issues,
        )
        for spec in environment.specifications
    }


def _commit_callback(environment: SyntheticEnvironment):
    def callback(
        task: ExperimentProcessTask,
        completed_task: CompletedExperimentTask,
    ) -> None:
        commit_completed_task(
            task,
            completed_task,
            environment.store,
            environment.audit_root,
            identities_by_experiment=environment.identities,
            contexts_by_experiment=environment.contexts,
            cohort=environment.cohort,
        )

    return callback


def _audit_records(environment: SyntheticEnvironment) -> tuple[TaskCommitAudit, ...]:
    return tuple(
        TaskCommitAudit.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((environment.audit_root / "task_commits").glob("*/*.json"))
    )


def test_workers_one_interrupt_preserves_commit_and_resume_uses_only_current_pending(
    tmp_path: Path,
) -> None:
    targets = ("20005", "20006", "20007", "20008", "20009", "20010")
    environment = _environment(tmp_path, targets)
    tasks = _tasks(environment, targets, chunk_size=2)
    callback_count = 0

    def interrupt_second(
        task: ExperimentProcessTask,
        completed_task: CompletedExperimentTask,
    ) -> None:
        nonlocal callback_count
        callback_count += 1
        if callback_count == 2:
            raise KeyboardInterrupt
        _commit_callback(environment)(task, completed_task)

    with pytest.raises(ProcessSchedulerFatalError) as captured:
        run_process_scheduler(
            tasks,
            _synthetic_worker,
            workers=1,
            on_task_completed=interrupt_second,
            retain_full_results=False,
        )
    first_report = captured.value.report
    assert first_report.interrupted
    assert first_report.committed_task_count == 1
    assert first_report.pending_task_count == 2
    statuses = _status_map(environment)
    assert all(status.completed_target_issues == ("20005", "20006") for status in statuses.values())
    assert all(status.pending_target_issues == targets[2:] for status in statuses.values())
    assert all(status.current_path.exists() for status in statuses.values())
    first_audits = _audit_records(environment)
    assert {audit.status for audit in first_audits} == {"completed"}
    assert first_audits[0].schema_version == "formal-task-commit-v1"
    assert first_audits[0].observation_count == 4
    assert first_audits[0].experiment_ids == ("B1", "B2")
    assert set(first_audits[0].committed_generation_by_experiment) == {"B1", "B2"}
    _, blockers = _reload_completion_gate(
        environment.store,
        environment.specifications,
        environment.identities,
        environment.contexts,
        environment.cohort,
        [first_report],
    )
    assert blockers

    groups = _pending_groups(
        environment.store,
        environment.specifications,
        {spec.experiment_id: targets for spec in environment.specifications},
        environment.identities,
        environment.contexts,
        environment.cohort,
    )
    assert list(groups) == [(77, targets[2:])]
    resume_tasks = _tasks(
        environment,
        targets[2:],
        chunk_size=2,
        specifications=tuple(groups[(77, targets[2:])]),
    )
    assert all("20005" not in task.target_issues for task in resume_tasks)
    second_report = run_process_scheduler(
        resume_tasks,
        _synthetic_worker,
        workers=1,
        on_task_completed=_commit_callback(environment),
        retain_full_results=False,
        resumed_from_completed_target_count=2,
    )
    final_statuses = _status_map(environment)
    assert all(status.is_complete for status in final_statuses.values())
    assert all(status.completed_target_issues == targets for status in final_statuses.values())
    _, blockers = _reload_completion_gate(
        environment.store,
        environment.specifications,
        environment.identities,
        environment.contexts,
        environment.cohort,
        [second_report],
    )
    assert not blockers
    assert second_report.resumed_from_completed_target_count == 2
    assert len(_audit_records(environment)) == 3


def test_workers_two_callbacks_follow_completion_order_run_in_parent_and_are_slim(
    tmp_path: Path,
) -> None:
    targets = ("20005", "20006", "20007", "20008")
    environment = _environment(tmp_path, targets)
    base_tasks = _tasks(environment, targets, chunk_size=1)
    delays = {
        base_tasks[0].task_id: 0.20,
        base_tasks[1].task_id: 0.01,
        base_tasks[2].task_id: 0.10,
        base_tasks[3].task_id: 0.01,
    }
    tasks = _tasks(environment, targets, chunk_size=1, delays=delays)
    parent_pid = os.getpid()
    callback_pids: list[int] = []
    completion_order: list[str] = []

    def callback(
        task: ExperimentProcessTask,
        completed_task: CompletedExperimentTask,
    ) -> None:
        callback_pids.append(os.getpid())
        completion_order.append(task.task_id)
        _commit_callback(environment)(task, completed_task)

    report = run_process_scheduler(
        tasks,
        _synthetic_worker,
        workers=2,
        on_task_completed=callback,
        retain_full_results=False,
    )
    assert callback_pids == [parent_pid] * 4
    assert completion_order[0] == tasks[1].task_id
    assert completion_order != [task.task_id for task in tasks]
    assert not report.completed_tasks
    assert len(report.completed_task_summaries) == 4
    assert all(summary.committed for summary in report.completed_task_summaries)
    assert all(status.is_complete for status in _status_map(environment).values())
    for spec in environment.specifications:
        frame = environment.store.read_partition(
            spec,
            environment.identities[spec.experiment_id],
            environment.cohort,
            seed=77,
        )
        assert frame["target_issue"].tolist() == list(targets)


def test_partial_experiment_append_resumes_only_missing_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = ("20005", "20006")
    environment = _environment(tmp_path, targets)
    task = _tasks(environment, targets, chunk_size=2)[0]
    completed = CompletedExperimentTask(
        task_id=task.task_id, result=_synthetic_worker(task), cpu_seconds=0
    )
    original_append = environment.store.append
    first_experiment = environment.specifications[0].experiment_id

    def partial_append(frame: pd.DataFrame):
        original_append(frame.loc[frame["experiment_id"] == first_experiment])
        raise RuntimeError("synthetic later partition failure")

    monkeypatch.setattr(environment.store, "append", partial_append)
    with pytest.raises(RuntimeError, match="later partition failure"):
        commit_completed_task(
            task,
            completed,
            environment.store,
            environment.audit_root,
            identities_by_experiment=environment.identities,
            contexts_by_experiment=environment.contexts,
            cohort=environment.cohort,
        )
    monkeypatch.setattr(environment.store, "append", original_append)
    statuses = _status_map(environment)
    assert statuses[(first_experiment, 77)].is_complete
    second_experiment = environment.specifications[1].experiment_id
    assert statuses[(second_experiment, 77)].pending_target_issues == targets
    failed_audit = _audit_records(environment)[0]
    assert failed_audit.status == "failed"
    assert set(failed_audit.committed_generation_by_experiment) == {first_experiment}

    resume_spec = (environment.specifications[1],)
    resume_task = _tasks(
        environment,
        targets,
        chunk_size=2,
        specifications=resume_spec,
    )[0]
    resume_report = run_process_scheduler(
        (resume_task,),
        _synthetic_worker,
        workers=1,
        on_task_completed=_commit_callback(environment),
        retain_full_results=False,
    )
    assert resume_report.committed_task_count == 1
    assert all(status.is_complete for status in _status_map(environment).values())


def test_current_success_with_final_audit_failure_remains_prepared_and_resume_uses_current(
    tmp_path: Path,
) -> None:
    targets = ("20005", "20006")
    environment = _environment(tmp_path, targets)
    task = _tasks(environment, targets, chunk_size=2)[0]
    completed = CompletedExperimentTask(
        task_id=task.task_id, result=_synthetic_worker(task), cpu_seconds=0
    )

    def fail(point: str) -> None:
        if point == "before_completed_audit":
            raise OSError("synthetic final audit failure")

    with pytest.raises(OSError, match="final audit failure"):
        commit_completed_task(
            task,
            completed,
            environment.store,
            environment.audit_root,
            identities_by_experiment=environment.identities,
            contexts_by_experiment=environment.contexts,
            cohort=environment.cohort,
            _fault_injector=fail,
        )
    statuses = _status_map(environment)
    assert all(status.is_complete for status in statuses.values())
    assert _audit_records(environment)[0].status == "prepared"
    audit_report = audit_task_commits(
        environment.audit_root,
        completed_targets_by_partition={
            key: set(status.completed_target_issues) for key, status in statuses.items()
        },
    )
    assert audit_report.prepared_count == 1
    assert audit_report.inconsistent_count == 1
    groups = _pending_groups(
        environment.store,
        environment.specifications,
        {spec.experiment_id: targets for spec in environment.specifications},
        environment.identities,
        environment.contexts,
        environment.cohort,
    )
    assert not groups


def test_store_failure_marks_audit_failed_and_duplicate_callback_cannot_duplicate_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = ("20005", "20006")
    environment = _environment(tmp_path, targets)
    task = _tasks(environment, targets, chunk_size=2)[0]
    completed = CompletedExperimentTask(
        task_id=task.task_id, result=_synthetic_worker(task), cpu_seconds=0
    )

    def fail_append(_frame: pd.DataFrame):
        raise RuntimeError("synthetic store failure")

    original_append = environment.store.append
    monkeypatch.setattr(environment.store, "append", fail_append)
    with pytest.raises(RuntimeError, match="store failure"):
        commit_completed_task(
            task,
            completed,
            environment.store,
            environment.audit_root,
            identities_by_experiment=environment.identities,
            contexts_by_experiment=environment.contexts,
            cohort=environment.cohort,
        )
    assert _audit_records(environment)[0].status == "failed"
    assert all(not status.completed_target_issues for status in _status_map(environment).values())

    monkeypatch.setattr(environment.store, "append", original_append)
    commit_completed_task(
        task,
        completed,
        environment.store,
        environment.audit_root,
        identities_by_experiment=environment.identities,
        contexts_by_experiment=environment.contexts,
        cohort=environment.cohort,
    )
    with pytest.raises(Exception, match="already contains"):
        commit_completed_task(
            task,
            completed,
            environment.store,
            environment.audit_root,
            identities_by_experiment=environment.identities,
            contexts_by_experiment=environment.contexts,
            cohort=environment.cohort,
        )
    assert all(
        len(
            environment.store.read_partition(
                spec,
                environment.identities[spec.experiment_id],
                environment.cohort,
                seed=77,
            )
        )
        == 2
        for spec in environment.specifications
    )


def test_worker_failure_keeps_targets_pending_and_creates_failed_audit(tmp_path: Path) -> None:
    targets = ("20005", "20006")
    environment = _environment(tmp_path, targets)
    task = _tasks(environment, targets, chunk_size=2)[0]

    def on_failed(task_value, failure) -> None:
        record_failed_task_audit(
            task_value,
            failure,
            environment.audit_root,
            identities_by_experiment=environment.identities,
            contexts_by_experiment=environment.contexts,
            cohort=environment.cohort,
        )

    report = run_process_scheduler(
        (task,),
        _failing_worker,
        workers=1,
        on_task_completed=_commit_callback(environment),
        on_task_failed=on_failed,
        retain_full_results=False,
    )
    assert report.worker_completed_task_count == 0
    assert report.committed_task_count == 0
    assert report.failed_task_count == 1
    assert report.pending_task_count == 1
    assert _audit_records(environment)[0].status == "failed"
    assert all(
        status.pending_target_issues == targets for status in _status_map(environment).values()
    )


def test_command_streams_callback_and_never_reappends_scheduler_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = _history()

    class FakeStore:
        def __init__(self) -> None:
            self.completed = False
            self.append_calls = 0

        def current_path(self, spec, identity, cohort, *, seed: int) -> Path:
            del identity, cohort
            return tmp_path / spec.experiment_id / f"seed_{seed}" / "CURRENT"

        def audit_partition_generations(self, spec, identity, cohort, *, seed: int):
            del identity, cohort, seed
            return PartitionGenerationAudit(
                partition_directory=tmp_path / spec.experiment_id,
                current_generation_id="generation-1" if self.completed else None,
                valid_generation_ids=("generation-1",) if self.completed else (),
                orphan_generation_ids=(),
                invalid_generation_ids=(),
                current_is_valid=True,
                recovery_possible=False,
                warnings=(),
            )

        def status(
            self,
            spec,
            identity,
            run_context,
            cohort,
            *,
            seed: int,
            expected_target_issues,
        ) -> ExperimentPartitionStatusV4:
            del run_context
            targets = tuple(expected_target_issues)
            return ExperimentPartitionStatusV4(
                phase=spec.data_split.phase,
                experiment_id=spec.experiment_id,
                experiment_version=spec.experiment_version,
                run_context_sha256=identity.run_context_sha256,
                execution_config_sha256=identity.execution_config_sha256,
                cohort_definition_sha256=cohort.cohort_definition_sha256,
                expected_targets_sha256=cohort.payload.expected_targets_sha256,
                seed=seed,
                current_path=self.current_path(spec, identity, cohort, seed=seed),
                current_generation_id="generation-1" if self.completed else None,
                completed_target_issues=targets if self.completed else (),
                pending_target_issues=() if self.completed else targets,
                is_complete=self.completed,
            )

        def append(self, _frame: pd.DataFrame) -> None:
            self.append_calls += 1
            pytest.fail("command repeated store.append after scheduler return")

        def load_cohort(self, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    store = FakeStore()

    def resolve_outputs(args, *, run_context_sha256, cohort_definition_sha256):
        formal = tmp_path / "formal" / run_context_sha256 / cohort_definition_sha256
        args.task_commit_root = formal
        args.scheduler_report_dir = formal / "scheduler"
        args.experiment_runtime_output = formal / "runtime.md"
        args.summary_output = formal / "summary.md"
        args.ablation_output = formal / "ablation.csv"
        args.holdout_output = formal / "holdout.md"
        args.contexts_audit_output = formal / "contexts.csv"
        args.run_log_dir = formal / "logs"
        return formal, ()

    def fake_commit(task, completed_task, store_arg, root, **_kwargs):
        assert completed_task.task_id == task.task_id
        assert store_arg is store
        assert root.is_relative_to(tmp_path)
        store.completed = True
        return SimpleNamespace(
            experiment_ids=("B1",),
            observation_count=1,
            committed_generation_by_experiment={"B1": "generation-1"},
        )

    def fake_scheduler(tasks, _worker, **kwargs):
        assert kwargs["retain_full_results"] is False
        completed = CompletedExperimentTask(
            task_id=tasks[0].task_id,
            result={},
            cpu_seconds=0,
        )
        kwargs["on_task_completed"](tasks[0], completed)
        now = datetime.now(UTC)
        return ProcessSchedulerReport(
            process_count=1,
            hostname="test",
            platform="test",
            python_version="3.13",
            started_at=now,
            ended_at=now,
            wall_clock_seconds=0,
            total_cpu_time_seconds=0,
            experiment_throughput_per_hour=0,
            estimated_remaining_runtime_seconds=0,
            completed_tasks=(),
            failed_tasks=(),
            planned_task_count=1,
            worker_completed_task_count=1,
            committed_task_count=1,
            pending_task_count=0,
            incremental_commit_enabled=True,
        )

    monkeypatch.setattr(command_module, "ExperimentResultStoreV4", lambda _root: store)
    monkeypatch.setattr(command_module, "load_verified_history", lambda _path: draws)
    monkeypatch.setattr(
        command_module,
        "load_prize_rule_schedule",
        lambda _path: SimpleNamespace(tables=()),
    )
    monkeypatch.setattr(command_module, "_resolve_output_paths", resolve_outputs)
    monkeypatch.setattr(
        command_module,
        "build_experiment_preflight",
        lambda **_kwargs: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(command_module, "write_experiment_preflight", lambda *_args: None)
    monkeypatch.setattr(command_module, "commit_completed_task", fake_commit)
    monkeypatch.setattr(command_module, "run_process_scheduler", fake_scheduler)
    monkeypatch.setattr(command_module, "write_process_scheduler_report", lambda *_args: None)
    monkeypatch.setattr(command_module, "write_experiment_runtime_report", lambda *_args: None)
    monkeypatch.setattr(command_module, "validate_single_formal_report_context", lambda _x: None)
    monkeypatch.setattr(
        command_module,
        "compare_to_constraint_matched_baseline",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        command_module,
        "write_experiment_summary_report",
        lambda *_args, **_kwargs: None,
    )

    result = command_module.main(
        [
            "--experiment-ids",
            "B1",
            "--target-issues",
            "20005",
            "--minimum-history",
            "3",
            "--results-root",
            str(tmp_path / "schema_v4"),
            "--preflight-root",
            str(tmp_path / "preflight"),
            "--workers",
            "1",
        ]
    )
    assert result == 0
    assert store.completed
    assert store.append_calls == 0
