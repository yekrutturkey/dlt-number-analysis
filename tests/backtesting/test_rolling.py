"""严格向前滚动回测测试。"""

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from pydantic import JsonValue

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.backtesting import run_rolling_backtest
from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.evaluation import load_prize_table
from dlt_number_analysis.models import PredictionRecord, TicketRecord

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
    assert fixed_26016.any_prize is True
    assert fixed_26016.total_cost == Decimal("10")
    assert fixed_26016.total_prize == Decimal("5")
    assert fixed_26016.roi == Decimal("-0.5")


def test_adding_future_draw_does_not_change_earlier_backtest_result() -> None:
    draws = make_draws()
    prize_table = load_prize_table(PRIZE_TABLE_PATH)
    short_report = run_rolling_backtest(
        draws.iloc[:3].copy(),
        prize_table,
        strategies={"fixed": fixed_strategy},
        min_history=2,
        base_random_seed=10,
    )
    full_report = run_rolling_backtest(
        draws,
        prize_table,
        strategies={"fixed": fixed_strategy},
        min_history=2,
        base_random_seed=10,
    )

    earlier_results = tuple(
        result for result in full_report.results if result.target_issue == "26016"
    )
    assert short_report.results == earlier_results
