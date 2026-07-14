"""Shared end-to-end prediction pipeline and strict rolling integration tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from random import Random

import pandas as pd
import pytest
from pydantic import ValidationError

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.backtesting import run_rolling_backtest
from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.pipeline import (
    PipelineConfig,
    PipelineSeeds,
    PredictionPipeline,
)
from dlt_number_analysis.scoring import ScorerSpec


def make_history(count: int = 40) -> pd.DataFrame:
    rng = Random(404)
    start = date(2025, 1, 1)
    return pd.DataFrame(
        [
            [
                str(25001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                *sorted(rng.sample(range(1, 36), 5)),
                *sorted(rng.sample(range(1, 13), 2)),
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


@pytest.fixture(scope="module")
def pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        scorer_spec=ScorerSpec(name="uniform_score"),
        optimizer_search_trials=5_000,
        minimum_history_size=40,
    )


def test_prediction_pipeline_persists_complete_bound_configuration(
    pipeline_config: PipelineConfig,
) -> None:
    history = make_history()
    seeds = PipelineSeeds(candidate_pool=111, portfolio_optimizer=112)

    result = PredictionPipeline(pipeline_config).run(
        history,
        target_issue="25041",
        generated_at=datetime(2026, 7, 14, tzinfo=UTC),
        random_seeds=seeds,
    )

    assert result.prediction.strategy_name == "optimized_portfolio_strategy"
    assert result.prediction.data_cutoff_issue == "25040"
    assert result.prediction.parameters["pipeline_config"] == pipeline_config.model_dump(
        mode="json"
    )
    assert result.prediction.parameters["random_seeds"] == seeds.model_dump(mode="json")
    assert result.prediction.parameters["short_history_override"] is False
    assert result.config.profile == "standard"
    assert result.config.parallel_workers == 1
    assert result.candidate_pool_summary.candidate_count == 10_000
    assert result.candidate_pool_summary.scorer_spec == pipeline_config.scorer_spec
    assert len(result.prediction.tickets) == 5
    assert result.risk_disclaimer == DISCLAIMER


def test_pipeline_rejects_history_that_contains_target_or_future_draw(
    pipeline_config: PipelineConfig,
) -> None:
    with pytest.raises(ValueError, match="cutoff must precede"):
        PredictionPipeline(pipeline_config).run(
            make_history(),
            target_issue="25040",
            generated_at=datetime(2026, 7, 14, tzinfo=UTC),
            random_seeds=PipelineSeeds.from_base_seed(1),
        )


def test_pipeline_rejects_history_below_default_minimum_without_override() -> None:
    with pytest.raises(ValueError, match="shorter than minimum_history_size"):
        PredictionPipeline().run(
            make_history(40),
            target_issue="25041",
            generated_at=datetime(2026, 7, 14, tzinfo=UTC),
            random_seeds=PipelineSeeds.from_base_seed(1),
        )


def test_pipeline_profiles_expand_all_runtime_parameters() -> None:
    fast = PipelineConfig(profile="fast")
    standard = PipelineConfig(profile="standard")
    final = PipelineConfig(profile="final")

    assert (fast.candidate_count, fast.optimizer_search_trials, fast.parallel_workers) == (
        10_000,
        5_000,
        1,
    )
    assert (
        standard.candidate_count,
        standard.optimizer_search_trials,
        standard.parallel_workers,
    ) == (10_000, 25_000, 1)
    assert (final.candidate_count, final.optimizer_search_trials, final.parallel_workers) == (
        25_000,
        100_000,
        4,
    )
    assert PipelineConfig(profile="final", candidate_count=30_000).candidate_count == 30_000


def test_pipeline_config_rejects_structure_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        PipelineConfig(
            structure_feature_weights={
                **PipelineConfig().structure_feature_weights,
                "front_sum": 0.2,
            }
        )


def test_rolling_backtest_invokes_same_optimized_pipeline_with_full_prior_history(
    pipeline_config: PipelineConfig,
) -> None:
    draws = make_history(41)

    report = run_rolling_backtest(
        draws,
        pipeline_configs={"optimized_portfolio_strategy": pipeline_config},
        min_history=40,
        base_random_seed=500,
    )

    result = next(
        item for item in report.results if item.strategy_name == "optimized_portfolio_strategy"
    )
    assert result.target_issue == "25041"
    assert result.data_cutoff_issue == "25040"
    assert set(result.random_metric_percentiles) == {
        "best_front_hits",
        "best_back_hits",
        "best_total_hits",
        "at_least_three_front",
        "at_least_2_plus_1",
        "front_pool_coverage",
        "back_pool_coverage",
        "unique_hit_concentration",
    }
