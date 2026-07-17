"""Shared rolling experiment-runner tests."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments import (
    ablation_experiment_specs,
    baseline_experiment_specs,
    build_experiment_pipeline_config,
    run_experiment,
    run_experiment_batch,
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


def test_constraint_matched_random_never_enters_optimized_pipeline() -> None:
    optimized = build_experiment_pipeline_config(baseline_experiment_specs()[5])

    with pytest.raises(ValueError, match="does not use the optimized pipeline"):
        build_experiment_pipeline_config(baseline_experiment_specs()[1])
    assert optimized.diversity_weight > 0
    assert optimized.scorer_spec.name == "recency_weighted_frequency_score"


def test_true_issue_gap_blocks_formal_experiment() -> None:
    history = _history().drop(index=4).reset_index(drop=True)

    with pytest.raises(ValueError, match="integrity errors block"):
        run_experiment(
            history,
            baseline_experiment_specs(seeds=(77,))[0],
            minimum_history=3,
            random_baseline_seed_count=1000,
            bootstrap_resamples=100,
        )


def test_b1_and_b2_share_candidates_bank_and_constraints_for_one_target() -> None:
    specs = baseline_experiment_specs(seeds=(77,))[1:3]

    result = run_experiment_batch(
        _history(),
        specs,
        minimum_history=3,
        target_issues=("26004",),
        random_baseline_seed_count=1000,
        bootstrap_resamples=100,
    )
    observations = result.observations.set_index("experiment_id")

    assert set(observations.index) == {"B1", "B2"}
    assert observations.loc["B1", "bank_hash"] == observations.loc["B2", "bank_hash"]
    assert (
        observations.loc["B1", "candidate_numbers_hash"]
        == observations.loc["B2", "candidate_numbers_hash"]
    )
    assert (
        observations.loc["B1", "constraints_signature"]
        == observations.loc["B2", "constraints_signature"]
    )
    assert result.executions[0].candidate_cache_hit_rate == 0.5
    assert result.executions[0].portfolio_bank_reuse_count == 1
    assert set(result.observations["target_candidate_generation_count"]) == {1}
    assert set(result.observations["target_candidate_reuse_count"]) == {1}
    assert set(result.observations["target_bank_generation_count"]) == {1}
    assert set(result.observations["target_portfolio_bank_reuse_count"]) == {1}
    assert set(result.observations["evaluation_mode"]) == {"full_resampling"}
    assert set(result.observations["random_baseline_seed_count"]) == {1000}


def test_b1_through_b6_raw_build_one_bank_and_skip_full_backtest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = baseline_experiment_specs(seeds=(79,))[1:7]

    def fail_if_called(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("raw observation must not call run_rolling_backtest")

    monkeypatch.setattr(
        "dlt_number_analysis.experiments.runner.run_rolling_backtest",
        fail_if_called,
    )

    result = run_experiment_batch(
        _history(),
        specs,
        minimum_history=3,
        target_issues=("26004",),
        random_baseline_seed_count=1,
        bootstrap_resamples=1,
        minimum_bank_size=100,
        evaluation_mode="raw_observation",
    )

    assert len(result.observations) == 6
    assert result.observations["run_context_sha256"].nunique() == 1
    assert result.observations["execution_config_sha256"].nunique() == 6
    assert result.observations["history_sha256"].nunique() == 1
    assert set(result.observations["requested_portfolio_scoring_method"]) == {"numpy_vectorized"}
    assert set(result.observations["formal_inference_eligible"]) == {True}
    assert result.observations["experiment_version"].tolist() == [
        f"v0.5.5-b{index}-shared-bank-v2" for index in range(1, 7)
    ]
    assert result.observations["candidate_numbers_hash"].nunique() == 1
    assert result.observations["constraints_signature"].nunique() == 1
    assert result.observations["bank_hash"].nunique() == 1
    assert set(result.observations["target_bank_generation_count"]) == {1}
    assert set(result.observations["target_portfolio_bank_reuse_count"]) == {5}
    assert set(
        result.observations.loc[
            result.observations["experiment_id"] != "B1", "portfolio_scoring_method"
        ]
    ) == {"numpy_vectorized"}
    assert set(
        result.observations.loc[
            result.observations["experiment_id"] == "B1", "portfolio_scoring_method"
        ]
    ) == {"random_bank_sample"}
    assert all(execution.portfolio_bank_reuse_count == 5 for execution in result.executions)
    assert result.observations["bank_size"].min() >= 100
    assert set(result.observations["evaluation_mode"]) == {"raw_observation"}
    assert {timing.evaluation_mode for timing in result.target_timings} == {"raw_observation"}
    assert result.target_timings[0].raw_evaluation_seconds >= 0


def test_final_holdout_rejects_raw_observation() -> None:
    spec = baseline_experiment_specs(seeds=(79,), phase="final_holdout")[1]

    with pytest.raises(ValueError, match="full_resampling"):
        run_experiment_batch(
            _history(),
            (spec,),
            minimum_history=3,
            target_issues=("26009",),
            evaluation_mode="raw_observation",
            holdout_lock_path="unused.json",
        )


def test_small_ablation_batch_uses_shared_cube_and_constraint_bank() -> None:
    registered = ablation_experiment_specs(seeds=(78,))
    specs = (registered[0], registered[6])

    result = run_experiment_batch(
        _history(),
        specs,
        minimum_history=3,
        target_issues=("26004",),
        random_baseline_seed_count=1000,
        bootstrap_resamples=100,
    )

    assert len(result.observations) == 2
    assert result.observations["bank_hash"].nunique() == 1
    assert result.observations["candidate_numbers_hash"].nunique() == 1
    assert result.executions[0].candidate_cache_hit_rate == 0.5
    assert result.executions[0].portfolio_bank_reuse_count == 1
