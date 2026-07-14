"""严格向前滚动回测测试。"""

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from pydantic import JsonValue

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.backtesting import run_rolling_backtest
from dlt_number_analysis.data import CSV_COLUMNS
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
    assert fixed_26016.hit_concentration_ratio == pytest.approx(0.6)
    assert fixed_26016.any_prize is True
    assert fixed_26016.total_cost == Decimal("10")
    assert fixed_26016.total_prize == Decimal("5")
    assert fixed_26016.roi == Decimal("-0.5")
    assert fixed_26016.random_baseline_seed_count == 1000
    assert 0 <= fixed_26016.best_total_hits_percentile <= 100
    assert (
        fixed_26016.best_total_hits_percentile_ci_lower
        <= fixed_26016.best_total_hits_percentile_ci_upper
    )
    assert report.random_baseline_summaries[0].seed_count == 1000
    fixed_summary = next(
        summary for summary in report.strategy_summaries if summary.strategy_name == "fixed"
    )
    assert fixed_summary.period_count == 2


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
