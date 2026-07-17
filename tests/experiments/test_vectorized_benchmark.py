"""Bounded vectorized benchmark artifact tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments import (
    VectorizedBenchmarkRun,
    VectorizedTargetBenchmark,
    load_vectorized_benchmark_json,
    write_vectorized_benchmark_json,
    write_vectorized_benchmark_report,
)


def _target(issue: str, speedup: float = 3.0) -> VectorizedTargetBenchmark:
    return VectorizedTargetBenchmark(
        target_issue=issue,
        data_cutoff_issue=str(int(issue) - 1),
        candidate_count=10_000,
        bank_size=1_300,
        candidate_generation_seconds=1.0,
        feasible_bank_generation_seconds=0.2,
        candidate_array_conversion_seconds=0.1,
        index_bank_build_seconds=0.1,
        object_reference_total_seconds=3.0,
        numpy_score_view_seconds=0.2,
        numpy_vectorized_scoring_seconds=0.05,
        final_object_construction_seconds=0.05,
        numpy_vectorized_total_seconds=1.0,
        object_reference_peak_memory_mb=100,
        numpy_vectorized_peak_memory_mb=10,
        speedup=speedup,
        b1_entry_unchanged=True,
        all_final_tickets_identical=True,
    )


def test_vectorized_benchmark_json_and_report_round_trip(tmp_path: Path) -> None:
    started = datetime(2026, 7, 15, tzinfo=UTC)
    run = VectorizedBenchmarkRun(
        workers=1,
        target_issues=("08009", "08010", "08011"),
        started_at=started,
        ended_at=started + timedelta(seconds=9),
        wall_clock_seconds=9,
        average_seconds_per_target=3,
        targets=(_target("08009"), _target("08010"), _target("08011")),
    )

    json_path = write_vectorized_benchmark_json(run, tmp_path / "benchmark.json")
    loaded = load_vectorized_benchmark_json(json_path)
    report = write_vectorized_benchmark_report((loaded,), tmp_path / "benchmark.md")
    content = report.read_text(encoding="utf-8")

    assert loaded == run
    assert "08009" in content
    assert "3.000x" in content
    assert "达到2倍目标" in content
    assert DISCLAIMER in content
