"""Deterministic ProcessPool experiment scheduler tests."""

from __future__ import annotations

from dlt_number_analysis.experiments import (
    ExperimentProcessTask,
    baseline_experiment_specs,
    build_process_tasks,
    default_process_count,
    run_process_scheduler,
)


def _echo_worker(task: ExperimentProcessTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "subseed": task.deterministic_subseed,
        "target_count": len(task.target_issues),
    }


def test_target_chunk_tasks_have_deterministic_subseeds() -> None:
    specifications = baseline_experiment_specs(seeds=(10,))[1:3]
    first = build_process_tasks(
        specifications,
        ("26001", "26002", "26003"),
        master_seed=99,
        chunk_size=2,
    )
    second = build_process_tasks(
        specifications,
        ("26001", "26002", "26003"),
        master_seed=99,
        chunk_size=2,
    )

    assert first == second
    assert len(first) == 2
    assert all(len(task.experiment_specs) == 2 for task in first)
    assert 1 <= default_process_count() <= 8


def test_process_scheduler_records_host_timing_and_results() -> None:
    tasks = build_process_tasks(
        baseline_experiment_specs(seeds=(10,))[1:2],
        ("26001", "26002"),
        master_seed=99,
        chunk_size=1,
    )

    report = run_process_scheduler(tasks, _echo_worker, workers=2)

    assert report.process_count == 2
    assert len(report.completed_tasks) == 2
    assert not report.failed_tasks
    assert report.hostname
    assert report.wall_clock_seconds > 0
    assert report.total_cpu_time_seconds >= 0
    assert report.experiment_throughput_per_hour > 0
