"""配置驱动的奖级评估和复盘测试。"""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from dlt_number_analysis.data import DrawRecord
from dlt_number_analysis.evaluation import (
    evaluate_prediction,
    evaluate_ticket,
    load_prize_table,
    render_prediction_review,
)
from dlt_number_analysis.models import PredictionRecord, TicketRecord

PRIZE_TABLE_PATH = Path(__file__).resolve().parents[2] / "config" / "prize_tiers.json"


def make_manual_prediction() -> PredictionRecord:
    """构造第 26078 期用户提供的五注手工事前票据。"""
    number_sets = [
        ((7, 16, 20, 29, 35), (5, 12)),
        ((4, 12, 18, 25, 32), (2, 8)),
        ((2, 10, 17, 24, 31), (1, 9)),
        ((6, 15, 22, 27, 33), (4, 10)),
        ((8, 13, 21, 26, 34), (3, 11)),
    ]
    return PredictionRecord(
        target_issue="26078",
        tickets=tuple(
            TicketRecord(
                target_issue="26078",
                ticket_id=f"26078-manual-{index}",
                front_numbers=front,
                back_numbers=back,
                ticket_role="manual_purchase",
            )
            for index, (front, back) in enumerate(number_sets, start=1)
        ),
        strategy_name="manual_purchase",
        model_version="manual-v0",
        data_cutoff_issue="26077",
        generated_at=None,
        recorded_at=datetime(2026, 7, 14, tzinfo=UTC),
        random_seed=None,
        parameters={"original_generated_at_status": "not_provided"},
        prediction_origin="manual_chat",
    )


def make_actual_draw() -> DrawRecord:
    """返回用户提供的第 26078 期开奖结果。"""
    return DrawRecord(
        issue="26078",
        draw_date="2026-07-13",
        front_1=2,
        front_2=13,
        front_3=20,
        front_4=25,
        front_5=32,
        back_1=8,
        back_2=11,
    )


def test_ticket_two_is_seventh_prize_from_configuration() -> None:
    prediction = make_manual_prediction()
    prize_table = load_prize_table(PRIZE_TABLE_PATH)

    evaluation = evaluate_ticket(
        prediction.tickets[1],
        make_actual_draw(),
        prize_table,
        prize_context="pool_at_or_above_800m",
    )

    assert evaluation.front_hits == 2
    assert evaluation.back_hits == 1
    assert evaluation.prize_tier == "七等奖"
    assert evaluation.prize_amount == Decimal("7")


def test_manual_prediction_review_has_full_pool_but_only_ticket_level_prize() -> None:
    prediction = make_manual_prediction()
    draw = make_actual_draw()
    review = evaluate_prediction(
        prediction,
        draw,
        load_prize_table(PRIZE_TABLE_PATH),
        prize_context="pool_at_or_above_800m",
    )

    assert review.best_front_hits == 2
    assert review.best_back_hits == 1
    assert review.any_prize is True
    assert review.front_pool_coverage == 5
    assert review.back_pool_coverage == 2
    assert review.total_cost == Decimal("10")
    assert review.total_prize == Decimal("7")
    assert review.roi == Decimal("-0.3")

    report = render_prediction_review(prediction, draw, review)
    assert "不是当前代码生成结果" in report
    assert "第 2 注命中 2 个前区和 1 个后区" in report
    assert "5 注号码池覆盖全部 5 个前区和 2 个后区" in report
    assert "号码池全覆盖不代表单注预测成功" in report


def test_prize_amount_is_read_from_configuration_not_evaluator() -> None:
    prediction = make_manual_prediction()
    table = load_prize_table(PRIZE_TABLE_PATH)
    changed_tiers = tuple(
        tier.model_copy(
            update={
                "prize_amount_by_context": {
                    **tier.prize_amount_by_context,
                    "pool_below_800m": Decimal("99"),
                }
            }
        )
        if tier.prize_tier == "七等奖"
        else tier
        for tier in table.tiers
    )
    changed_table = table.model_copy(update={"tiers": changed_tiers})

    evaluation = evaluate_ticket(prediction.tickets[1], make_actual_draw(), changed_table)

    assert evaluation.prize_amount == Decimal("99")


def test_seventh_prize_remains_five_when_pool_is_below_800m() -> None:
    prediction = make_manual_prediction()
    evaluation = evaluate_ticket(
        prediction.tickets[1],
        make_actual_draw(),
        load_prize_table(PRIZE_TABLE_PATH),
        prize_context="pool_below_800m",
    )

    assert evaluation.prize_tier == "七等奖"
    assert evaluation.prize_amount == Decimal("5")
