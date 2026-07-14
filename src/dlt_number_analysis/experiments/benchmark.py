"""Reproducible pipeline runtime and Python-allocation memory benchmarks."""

from __future__ import annotations

import tracemalloc
from collections.abc import Sequence
from datetime import datetime
from statistics import median
from time import perf_counter, process_time
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis.pipeline import (
    PipelineConfig,
    PipelineProfile,
    PipelineSeeds,
    PredictionPipeline,
)


class RuntimeBenchmarkRecord(BaseModel):
    """Measured timings and Python peak allocations for one pipeline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: PipelineProfile
    parallel_workers: int = Field(ge=1)
    candidate_scoring_method: Literal["scalar", "numpy_batch_features"] = "numpy_batch_features"
    random_seed: int
    runtime_seconds: float = Field(ge=0)
    peak_memory_mb: float = Field(ge=0)
    candidate_generation_seconds: float = Field(ge=0)
    candidate_scoring_seconds: float = Field(ge=0)
    portfolio_search_seconds: float = Field(ge=0)
    feasible_portfolios_evaluated: int = Field(ge=0)
    candidate_cache_hit_rate: float = Field(default=0.0, ge=0, le=1)
    feasible_bank_generation_seconds: float = Field(default=0.0, ge=0)
    portfolio_bank_reuse_count: int = Field(default=0, ge=0)
    experiment_throughput_per_hour: float = Field(default=0.0, ge=0)
    estimated_remaining_runtime_seconds: float = Field(default=0.0, ge=0)
    process_count: int = Field(default=1, ge=1)
    total_cpu_time_seconds: float = Field(default=0.0, ge=0)
    wall_clock_seconds: float = Field(default=0.0, ge=0)
    memory_method: str = "tracemalloc_python_allocations"


def benchmark_pipeline_profiles(
    history: pd.DataFrame,
    *,
    target_issue: str,
    generated_at: datetime,
    profiles: Sequence[PipelineProfile] = ("fast", "standard", "final"),
    parallel_workers: Sequence[int] = (1, 4),
    candidate_scoring_methods: Sequence[Literal["scalar", "numpy_batch_features"]] = (
        "numpy_batch_features",
    ),
    random_seed: int = 20260714,
) -> tuple[RuntimeBenchmarkRecord, ...]:
    """Measure every requested profile/worker combination using identical history and seed."""
    records: list[RuntimeBenchmarkRecord] = []
    for scoring_method in candidate_scoring_methods:
        for profile in profiles:
            for worker_count in parallel_workers:
                config = PipelineConfig(
                    profile=profile,
                    parallel_workers=worker_count,
                    candidate_scoring_method=scoring_method,
                )
                was_tracing = tracemalloc.is_tracing()
                if not was_tracing:
                    tracemalloc.start()
                tracemalloc.reset_peak()
                started = perf_counter()
                cpu_started = process_time()
                result = PredictionPipeline(config).run(
                    history,
                    target_issue=target_issue,
                    generated_at=generated_at,
                    random_seeds=PipelineSeeds.from_base_seed(random_seed),
                )
                runtime_seconds = perf_counter() - started
                cpu_seconds = process_time() - cpu_started
                _, peak_bytes = tracemalloc.get_traced_memory()
                if not was_tracing:
                    tracemalloc.stop()
                optimizer = result.portfolio_selection.optimizer_parameters
                records.append(
                    RuntimeBenchmarkRecord(
                        profile=profile,
                        parallel_workers=worker_count,
                        candidate_scoring_method=scoring_method,
                        random_seed=random_seed,
                        runtime_seconds=runtime_seconds,
                        peak_memory_mb=peak_bytes / (1024 * 1024),
                        candidate_generation_seconds=(
                            result.candidate_pool_summary.candidate_generation_seconds
                        ),
                        candidate_scoring_seconds=(
                            result.candidate_pool_summary.candidate_scoring_seconds
                        ),
                        portfolio_search_seconds=float(optimizer["portfolio_search_seconds"]),
                        feasible_portfolios_evaluated=int(
                            optimizer["feasible_portfolios_evaluated"]
                        ),
                        experiment_throughput_per_hour=(
                            0.0 if runtime_seconds == 0 else 3600 / runtime_seconds
                        ),
                        process_count=1,
                        total_cpu_time_seconds=cpu_seconds,
                        wall_clock_seconds=runtime_seconds,
                    )
                )
    return tuple(records)


def choose_default_parallel_workers(
    records: Sequence[RuntimeBenchmarkRecord],
    *,
    profile: PipelineProfile = "final",
    candidate_scoring_method: Literal["scalar", "numpy_batch_features"] = ("numpy_batch_features"),
    minimum_stable_speedup: float = 1.05,
) -> int:
    """Choose four workers only after a stable scoring-time speedup is measured."""
    selected = [
        record
        for record in records
        if record.profile == profile and record.candidate_scoring_method == candidate_scoring_method
    ]
    one = [record.candidate_scoring_seconds for record in selected if record.parallel_workers == 1]
    four = [record.candidate_scoring_seconds for record in selected if record.parallel_workers == 4]
    if not one or not four:
        return 1
    paired_count = min(len(one), len(four))
    speedups = [one[index] / four[index] for index in range(paired_count) if four[index] > 0]
    if (
        speedups
        and min(speedups) >= minimum_stable_speedup
        and median(speedups) >= (minimum_stable_speedup)
    ):
        return 4
    return 1
