"""Conservative parallel-worker benchmark decision tests."""

from __future__ import annotations

from dlt_number_analysis.experiments import (
    RuntimeBenchmarkRecord,
    choose_default_parallel_workers,
)


def _record(workers: int, scoring_seconds: float) -> RuntimeBenchmarkRecord:
    return RuntimeBenchmarkRecord(
        profile="final",
        parallel_workers=workers,
        random_seed=1,
        runtime_seconds=10,
        peak_memory_mb=20,
        candidate_generation_seconds=1,
        candidate_scoring_seconds=scoring_seconds,
        portfolio_search_seconds=4,
        feasible_portfolios_evaluated=100,
    )


def test_four_threads_require_stable_scoring_speedup() -> None:
    unstable = (_record(1, 4.0), _record(4, 3.5), _record(1, 4.0), _record(4, 4.2))
    stable = (_record(1, 4.0), _record(4, 3.0), _record(1, 4.2), _record(4, 3.1))

    assert choose_default_parallel_workers(unstable) == 1
    assert choose_default_parallel_workers(stable) == 4
    assert choose_default_parallel_workers(stable[:1]) == 1
