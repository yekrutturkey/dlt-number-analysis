"""Shared rolling experiment-runner tests."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments import (
    baseline_experiment_specs,
    build_experiment_pipeline_config,
    run_experiment,
)


def _history(count: int = 10) -> pd.DataFrame:
    start = date(2026, 1, 1)
    return pd.DataFrame(
        [
            [
                str(26001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                1 + index % 5,
                7 + index % 5,
                13 + index % 5,
                19 + index % 5,
                25 + index % 5,
                1 + index % 3,
                7 + index % 3,
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def test_uniform_baseline_runner_uses_only_declared_development_targets() -> None:
    spec = baseline_experiment_specs(seeds=(77,))[0]
    result = run_experiment(
        _history(),
        spec,
        minimum_history=3,
        random_baseline_seed_count=1000,
        bootstrap_resamples=100,
    )

    assert result.observations["target_issue"].tolist() == ["26004", "26005", "26006"]
    assert set(result.observations["data_cutoff_issue"]) == {"26003", "26004", "26005"}
    assert set(result.observations["experiment_id"]) == {"B0"}
    assert result.executions[0].period_count == 3


def test_constraint_matched_random_removes_score_objective_preferences() -> None:
    constraint_matched = build_experiment_pipeline_config(baseline_experiment_specs()[1])
    optimized = build_experiment_pipeline_config(baseline_experiment_specs()[5])

    assert constraint_matched.single_ticket_weight == 1
    assert constraint_matched.diversity_weight == 0
    assert constraint_matched.core_weight == 0
    assert constraint_matched.structure_weight == 0
    assert constraint_matched.repeat_penalty_weight == 0
    assert optimized.diversity_weight > 0
    assert optimized.scorer_spec.name == "recency_weighted_frequency_score"
