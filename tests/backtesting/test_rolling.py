"""严格向前滚动回测测试。"""

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from pydantic import JsonValue

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.backtesting import (
    BacktestPeriodResult,
    calculate_raw_hit_metrics,
    clear_random_baseline_cache,
    run_rolling_backtest,
)
from dlt_number_analysis.data import CSV_COLUMNS, DrawRecord
from dlt_number_analysis.evaluation import IssuePrizeRecord, load_prize_table
from dlt_number_analysis.models import PredictionRecord, TicketRecord
from dlt_number_analysis.scoring import NumberArea, uniform_score

PRIZE_TABLE_PATH = Path(__file__).resolve().parents[2] / "config" / "prize_tiers.json"


def make_draws() -> pd.DataFrame:
    """构造按时间递增的四期测试开奖。"""
    return pd.DataFrame(
        [
            ["26014", "2026-02-02", 3, 9, 15, 22, 34, 4, 10],
            ["26015", "2026-02-04", 5, 11, 17, 24, 35, 6, 9],
            ["26016", "2026-02-07", 1, 2, 10, 20, 30, 1, 12],
            ["26017", "2026-02-09", 2, 3, 11, 21, 31, 2, 11],
        ],
        columns=CSV_COLUMNS,
    )


def fixed_strategy(
    *,
    target_issue: str,
    data_cutoff_issue: str,
    generated_at: datetime,
    random_seed: int,
    parameters: Mapping[str, JsonValue] | None = None,
    front_scores: Mapping[int, float] | None = None,
    back_scores: Mapping[int, float] | None = None,
    model_version: str = "fixed-test-v1",
) -> PredictionRecord:
    """返回与未来开奖无关的固定组合，用于验证回测指标。"""
    del front_scores, back_scores
    number_sets = [
        ((1, 2, 6, 7, 8), (1, 3)),
        ((4, 9, 14, 19, 24), (2, 4)),
        ((5, 10, 15, 20, 25), (5, 7)),
        ((11, 16, 21, 26, 31), (6, 8)),
        ((12, 17, 22, 27, 32), (9, 10)),
    ]
    return PredictionRecord(
        target_issue=target_issue,
        tickets=tuple(
            TicketRecord(
                target_issue=target_issue,
                ticket_id=f"{target_issue}-fixed-{index}",
                front_numbers=front,
                back_numbers=back,
            )
            for index, (front, back) in enumerate(number_sets, start=1)
        ),
        strategy_name="fixed",
        model_version=model_version,
        data_cutoff_issue=data_cutoff_issue,
        generated_at=generated_at,
        random_seed=random_seed,
        parameters=dict(parameters or {}),
        prediction_origin="generated",
    )


def test_rolling_backtest_outputs_requested_metrics_and_random_baseline() -> None:
    clear_random_baseline_cache()
    report = run_rolling_backtest(
        make_draws(),
        load_prize_table(PRIZE_TABLE_PATH),
        strategies={"fixed": fixed_strategy},
        min_history=2,
        base_random_seed=10,
        prize_context_by_issue={
            "26016": "pool_below_800m",
            "26017": "pool_below_800m",
        },
    )

    assert report.includes_random_baseline is True
    assert report.risk_disclaimer == DISCLAIMER
    assert "不声称任何策略优于" in report.comparative_claim
    assert {result.strategy_name for result in report.results} == {"fixed", "random_baseline"}
    fixed_26016 = next(
        result
        for result in report.results
        if result.strategy_name == "fixed" and result.target_issue == "26016"
    )
    assert fixed_26016.data_cutoff_issue == "26015"
    assert fixed_26016.best_front_hits == 2
    assert fixed_26016.best_back_hits == 1
    assert fixed_26016.best_total_hits == 3
    assert fixed_26016.at_least_three_front is False
    assert fixed_26016.at_least_2_plus_1 is True
    assert fixed_26016.ticket_hit_share == pytest.approx(0.6)
    assert fixed_26016.hit_concentration_ratio == fixed_26016.ticket_hit_share
    assert fixed_26016.unique_hit_concentration == pytest.approx(0.6)
    assert fixed_26016.any_prize is True
    assert fixed_26016.total_cost == Decimal("10")
    assert fixed_26016.total_prize == Decimal("5")
    assert fixed_26016.roi == Decimal("-0.5")
    assert fixed_26016.random_baseline_seed_count == 1000
    assert 0 <= fixed_26016.random_metric_percentiles["best_total_hits"] <= 100
    assert set(fixed_26016.random_metric_percentiles) == {
        "best_front_hits",
        "best_back_hits",
        "best_total_hits",
        "at_least_three_front",
        "at_least_2_plus_1",
        "front_pool_coverage",
        "back_pool_coverage",
        "unique_hit_concentration",
    }
    assert report.random_baseline_summaries[0].seed_count == 1000
    assert report.random_baseline_summaries[0].cache_reused is False
    assert report.random_baseline_summaries[1].cache_reused is True
    assert (
        report.random_baseline_summaries[0]
        .metric_summaries["best_total_hits"]
        .monte_carlo_error_interval.method
        == "monte_carlo_normal_error_interval"
    )
    fixed_summary = next(
        summary for summary in report.strategy_summaries if summary.strategy_name == "fixed"
    )
    assert fixed_summary.period_count == 2
    assert (
        fixed_summary.bootstrap_performance_intervals["best_total_hits"].method
        == "cross_historical_period_bootstrap"
    )


def test_reduced_resampling_requires_explicit_research_override() -> None:
    draws = make_draws().iloc[:3].copy()
    with pytest.raises(ValueError, match="at least 1000"):
        run_rolling_backtest(
            draws,
            strategies={"fixed": fixed_strategy},
            min_history=2,
            random_baseline_seed_count=10,
            bootstrap_resamples=10,
        )

    report = run_rolling_backtest(
        draws,
        strategies={"fixed": fixed_strategy},
        min_history=2,
        random_baseline_seed_count=10,
        bootstrap_resamples=10,
        allow_reduced_resampling=True,
    )

    assert all(result.random_baseline_seed_count == 10 for result in report.results)
    assert report.random_baseline_summaries[0].seed_count == 10


def test_raw_hit_metrics_skip_resampling_outputs() -> None:
    draws = make_draws()
    prediction = fixed_strategy(
        target_issue="26016",
        data_cutoff_issue="26015",
        generated_at=datetime.fromisoformat("2026-02-04T23:59:00+08:00"),
        random_seed=1,
    )
    actual = DrawRecord.model_validate(draws.iloc[2].to_dict())

    metrics = calculate_raw_hit_metrics(prediction, actual)

    assert metrics == {
        "best_front_hits": 2,
        "best_back_hits": 1,
        "best_total_hits": 3,
        "at_least_three_front": False,
        "at_least_2_plus_1": True,
        "ticket_hit_share": pytest.approx(0.6),
        "unique_hit_concentration": pytest.approx(0.6),
    }


def test_legacy_hit_concentration_input_migrates_to_ticket_hit_share() -> None:
    report = run_rolling_backtest(
        make_draws().iloc[:3].copy(),
        strategies={"fixed": fixed_strategy},
        min_history=2,
    )
    result = next(item for item in report.results if item.strategy_name == "fixed")
    payload = result.model_dump()
    old_value = payload.pop("ticket_hit_share")
    payload["hit_concentration_ratio"] = old_value

    migrated = BacktestPeriodResult.model_validate(payload)

    assert migrated.ticket_hit_share == old_value
    assert "hit_concentration_ratio" not in migrated.model_dump()


def test_repeated_core_hits_do_not_reduce_unique_hit_concentration() -> None:
    def repeated_core_strategy(**kwargs: object) -> PredictionRecord:
        prediction = fixed_strategy(**kwargs)  # type: ignore[arg-type]
        tickets = list(prediction.tickets)
        replacements = (
            (1, 2, 6, 7, 8),
            (1, 10, 14, 19, 24),
            (1, 15, 20, 25, 29),
        )
        for index, front in enumerate(replacements):
            tickets[index] = tickets[index].model_copy(update={"front_numbers": front})
        return prediction.model_copy(update={"tickets": tuple(tickets)})

    report = run_rolling_backtest(
        make_draws().iloc[:3].copy(),
        strategies={"fixed": repeated_core_strategy},
        min_history=2,
    )
    result = next(item for item in report.results if item.strategy_name == "fixed")

    assert result.ticket_hit_share == pytest.approx(3 / 7)
    assert result.unique_hit_concentration == pytest.approx(3 / 5)
    assert result.unique_hit_concentration > result.ticket_hit_share


def test_unique_hit_concentration_is_zero_when_pool_has_no_hits() -> None:
    def no_hit_strategy(**kwargs: object) -> PredictionRecord:
        prediction = fixed_strategy(**kwargs)  # type: ignore[arg-type]
        number_sets = (
            ((3, 4, 5, 6, 7), (2, 3)),
            ((8, 9, 11, 12, 13), (4, 5)),
            ((14, 15, 16, 17, 18), (6, 7)),
            ((19, 21, 22, 23, 24), (8, 9)),
            ((25, 26, 27, 28, 29), (10, 11)),
        )
        tickets = tuple(
            ticket.model_copy(update={"front_numbers": front, "back_numbers": back})
            for ticket, (front, back) in zip(prediction.tickets, number_sets, strict=True)
        )
        return prediction.model_copy(update={"tickets": tickets})

    report = run_rolling_backtest(
        make_draws().iloc[:3].copy(),
        strategies={"fixed": no_hit_strategy},
        min_history=2,
    )
    result = next(item for item in report.results if item.strategy_name == "fixed")

    assert result.front_pool_coverage == 0
    assert result.back_pool_coverage == 0
    assert result.unique_hit_concentration == 0
    assert result.ticket_hit_share == 0


def test_adding_future_draw_does_not_change_earlier_backtest_result() -> None:
    draws = make_draws()
    prize_table = load_prize_table(PRIZE_TABLE_PATH)
    short_report = run_rolling_backtest(
        draws.iloc[:3].copy(),
        prize_table,
        strategies={"fixed": fixed_strategy},
        min_history=2,
        base_random_seed=10,
        prize_context_by_issue={"26016": "pool_below_800m"},
    )
    full_report = run_rolling_backtest(
        draws,
        prize_table,
        strategies={"fixed": fixed_strategy},
        min_history=2,
        base_random_seed=10,
        prize_context_by_issue={
            "26016": "pool_below_800m",
            "26017": "pool_below_800m",
        },
    )

    earlier_results = tuple(
        result for result in full_report.results if result.target_issue == "26016"
    )
    assert short_report.results == earlier_results


def test_number_scorer_only_receives_history_before_each_target() -> None:
    observed: list[tuple[NumberArea, int, str]] = []

    def observing_scorer(history: pd.DataFrame, *, area: NumberArea) -> dict[int, float]:
        observed.append((area, len(history), str(history.iloc[-1]["issue"])))
        return uniform_score(history, area=area)

    run_rolling_backtest(
        make_draws(),
        load_prize_table(PRIZE_TABLE_PATH),
        strategies={"fixed": fixed_strategy},
        min_history=2,
        number_scorer=observing_scorer,
        prize_context_by_issue={
            "26016": "pool_below_800m",
            "26017": "pool_below_800m",
        },
    )

    assert observed == [
        ("front", 2, "26015"),
        ("back", 2, "26015"),
        ("front", 3, "26016"),
        ("back", 3, "26016"),
    ]


def test_current_seven_tier_rule_is_not_used_for_older_issues() -> None:
    old_draws = make_draws().iloc[:3].copy()
    old_draws["issue"] = ["25001", "25002", "25003"]

    report = run_rolling_backtest(
        old_draws,
        load_prize_table(PRIZE_TABLE_PATH),
        strategies={"fixed": fixed_strategy},
        min_history=2,
    )

    fixed = next(result for result in report.results if result.strategy_name == "fixed")
    assert fixed.best_total_hits >= 0
    assert fixed.prize_rule_version is None
    assert fixed.any_prize is None
    assert fixed.total_prize is None
    assert fixed.roi is None
    assert fixed.prize_data_available is False
    summary = next(item for item in report.strategy_summaries if item.strategy_name == "fixed")
    assert summary.average_prize is None
    assert summary.median_prize is None
    assert summary.longest_no_prize_streak is None
    assert summary.monetary_metrics_complete is False


def test_issue_actual_prize_record_overrides_rule_amount() -> None:
    table = load_prize_table(PRIZE_TABLE_PATH)
    record = IssuePrizeRecord(
        issue="26016",
        prize_table_version=table.version,
        prize_context="pool_below_800m",
        actual_amount_by_tier={"七等奖": Decimal("9")},
    )

    report = run_rolling_backtest(
        make_draws().iloc[:3].copy(),
        table,
        strategies={"fixed": fixed_strategy},
        min_history=2,
        issue_prize_records={"26016": record},
    )

    fixed = next(result for result in report.results if result.strategy_name == "fixed")
    assert fixed.total_prize == Decimal("9")
    assert fixed.roi == Decimal("-0.1")
    assert fixed.prize_rule_version == table.version


def test_missing_historical_pool_context_keeps_hits_but_disables_roi() -> None:
    report = run_rolling_backtest(
        make_draws().iloc[:3].copy(),
        load_prize_table(PRIZE_TABLE_PATH),
        strategies={"fixed": fixed_strategy},
        min_history=2,
    )

    fixed = next(result for result in report.results if result.strategy_name == "fixed")
    assert fixed.best_total_hits == 3
    assert fixed.any_prize is True
    assert fixed.total_prize is None
    assert fixed.roi is None
    assert fixed.prize_data_available is False
