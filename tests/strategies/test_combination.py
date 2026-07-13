"""五注组合策略测试。"""

from collections import Counter
from datetime import UTC, datetime

import pytest

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.models import PredictionRecord
from dlt_number_analysis.strategies import (
    core_rotation,
    hybrid_portfolio,
    max_coverage,
    random_baseline,
)

GENERATED_AT = datetime(2026, 7, 14, 8, tzinfo=UTC)


def strategy_arguments() -> dict[str, object]:
    """返回所有测试策略共享的审计参数。"""
    return {
        "target_issue": "26079",
        "data_cutoff_issue": "26078",
        "generated_at": GENERATED_AT,
        "random_seed": 20260714,
    }


def assert_valid_five_ticket_prediction(prediction: PredictionRecord) -> None:
    """检查所有策略必须满足的五注输出契约。"""
    assert len(prediction.tickets) == 5
    combinations = {(ticket.front_numbers, ticket.back_numbers) for ticket in prediction.tickets}
    assert len(combinations) == 5
    assert prediction.random_seed == 20260714
    assert prediction.parameters["ticket_count"] == 5
    assert prediction.risk_disclaimer == DISCLAIMER


def test_max_coverage_avoids_cross_ticket_number_reuse() -> None:
    prediction = max_coverage(**strategy_arguments())  # type: ignore[arg-type]

    assert_valid_five_ticket_prediction(prediction)
    assert len({number for ticket in prediction.tickets for number in ticket.front_numbers}) == 25
    assert len({number for ticket in prediction.tickets for number in ticket.back_numbers}) == 10


@pytest.mark.parametrize("repetitions", [2, 3])
def test_core_rotation_reuses_each_core_two_or_three_times(repetitions: int) -> None:
    prediction = core_rotation(
        **strategy_arguments(),  # type: ignore[arg-type]
        parameters={"core_front_numbers": [1, 2, 3, 4, 5], "core_repetitions": repetitions},
    )

    assert_valid_five_ticket_prediction(prediction)
    counts = Counter(number for ticket in prediction.tickets for number in ticket.front_numbers)
    assert all(counts[number] == repetitions for number in range(1, 6))


def test_hybrid_portfolio_has_three_core_and_two_exploration_tickets() -> None:
    prediction = hybrid_portfolio(**strategy_arguments())  # type: ignore[arg-type]

    assert_valid_five_ticket_prediction(prediction)
    roles = Counter(ticket.ticket_role for ticket in prediction.tickets)
    assert roles == {"core_rotation": 3, "exploration": 2}


def test_random_baseline_is_seeded_and_ignores_scores() -> None:
    first = random_baseline(
        **strategy_arguments(),  # type: ignore[arg-type]
        front_scores={1: 999.0},
        back_scores={1: 999.0},
    )
    second = random_baseline(**strategy_arguments())  # type: ignore[arg-type]

    assert_valid_five_ticket_prediction(first)
    assert [ticket.front_numbers for ticket in first.tickets] == [
        ticket.front_numbers for ticket in second.tickets
    ]
    assert first.parameters["front_scores"] == {}
