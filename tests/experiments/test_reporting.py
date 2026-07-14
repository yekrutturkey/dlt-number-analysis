"""Mandatory phase separation and risk-statement report tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import DataQualityReport
from dlt_number_analysis.experiments import (
    CompletedExperimentTask,
    ProcessSchedulerReport,
    RuntimeBenchmarkRecord,
    compare_to_constraint_matched_baseline,
    write_ablation_results_csv,
    write_data_quality_report,
    write_experiment_runtime_report,
    write_experiment_summary_report,
    write_holdout_results_report,
    write_runtime_benchmark_report,
)


def _observations() -> pd.DataFrame:
    records = []
    for experiment_id, offset in (("B1", 0), ("B5", 1)):
        for index in range(8):
            records.append(
                {
                    "phase": "development",
                    "experiment_id": experiment_id,
                    "target_issue": str(26001 + index),
                    "seed": 100 + index % 2,
                    "best_front_hits": 1 + offset,
                    "best_back_hits": 0,
                    "best_total_hits": 1 + offset,
                    "at_least_three_front": bool(offset),
                    "at_least_2_plus_1": bool(offset),
                    "unique_hit_concentration": 0.2 + offset * 0.1,
                    "any_prize": bool(offset),
                    "total_prize": 5 * offset,
                    "roi": -1 + offset,
                }
            )
    return pd.DataFrame(records)


def test_all_required_report_types_include_phase_or_risk_context(tmp_path: Path) -> None:
    observations = _observations()
    comparisons = compare_to_constraint_matched_baseline(
        observations,
        bootstrap_resamples=100,
        permutations=1000,
        minimum_paired_observations=2,
    )
    quality = DataQualityReport(
        record_count=2,
        data_start_issue="26001",
        data_cutoff_issue="26002",
        source_names=("official", "independent"),
        findings=(),
        conflict_count=0,
        is_valid=True,
        blocks_backtest=False,
    )
    benchmark = RuntimeBenchmarkRecord(
        profile="fast",
        parallel_workers=1,
        random_seed=1,
        runtime_seconds=1,
        peak_memory_mb=2,
        candidate_generation_seconds=0.1,
        candidate_scoring_seconds=0.4,
        portfolio_search_seconds=0.5,
        feasible_portfolios_evaluated=20,
    )
    scheduler = ProcessSchedulerReport(
        process_count=1,
        hostname="test-host",
        platform="test-platform",
        python_version="3.13",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        ended_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        wall_clock_seconds=1,
        total_cpu_time_seconds=0.5,
        experiment_throughput_per_hour=3600,
        estimated_remaining_runtime_seconds=0,
        completed_tasks=(
            CompletedExperimentTask(
                task_id="one",
                result={
                    "executions": [
                        {
                            "experiment_id": "B1",
                            "phase": "development",
                            "seed": 1,
                            "period_count": 1,
                            "candidate_cache_hit_rate": 0,
                            "feasible_bank_generation_seconds": 0.2,
                            "portfolio_bank_reuse_count": 0,
                            "experiment_throughput_per_hour": 3600,
                            "estimated_remaining_runtime_seconds": 0,
                            "total_cpu_time_seconds": 0.5,
                            "wall_clock_seconds": 1,
                        }
                    ]
                },
                cpu_seconds=0.5,
            ),
        ),
        failed_tasks=(),
    )

    paths = (
        write_data_quality_report(quality, None, tmp_path / "data_quality.md"),
        write_experiment_summary_report(
            observations, comparisons, tmp_path / "experiment_summary.md"
        ),
        write_holdout_results_report(observations, tmp_path / "holdout_results.md"),
        write_runtime_benchmark_report((benchmark,), tmp_path / "runtime_benchmark.md"),
        write_experiment_runtime_report((scheduler,), tmp_path / "experiment_runtime.md"),
    )
    ablation = write_ablation_results_csv(observations.iloc[:1], tmp_path / "ablation_results.csv")

    assert all(DISCLAIMER in path.read_text(encoding="utf-8") for path in paths)
    assert DISCLAIMER in ablation.read_text(encoding="utf-8")
    summary = paths[1].read_text(encoding="utf-8")
    assert "历史拟合" in summary
    assert "校准" in summary
    assert "最终留出" in summary
    assert "配对置换检验" in summary
    assert all(item.holm_adjusted_p_value >= item.permutation_p_value for item in comparisons)
    assert all(
        item.statistically_significant_advantage == item.holm_significant_advantage
        for item in comparisons
    )
