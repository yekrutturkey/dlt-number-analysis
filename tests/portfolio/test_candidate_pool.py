"""候选票池、评分明细和五注 Portfolio 优化测试。"""

from collections import Counter
from datetime import UTC, date, datetime, timedelta
from itertools import combinations
from pathlib import Path
from random import Random

import pandas as pd
import pytest
from pydantic import ValidationError

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.portfolio import (
    CandidatePool,
    PortfolioConstraints,
    PortfolioSelection,
    generate_candidate_pool,
    load_candidate_score_details,
    optimize_portfolio,
    write_candidate_score_details,
)
from dlt_number_analysis.scoring import ScorerSpec, recency_weighted_frequency_score


def make_history(count: int = 80) -> pd.DataFrame:
    """构造多样且严格递增的测试历史。"""
    rng = Random(818)
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


def thirty_period_scorer(history: pd.DataFrame, *, area: str) -> dict[int, float]:
    """为测试候选池固定使用最近 30 期评分。"""
    return recency_weighted_frequency_score(history, area=area, window=30)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def candidate_pool() -> CandidatePool:
    """生成一次一万注候选，供持久化和优化测试复用。"""
    return generate_candidate_pool(
        make_history(),
        target_issue="26081",
        generated_at=datetime(2026, 7, 18, tzinfo=UTC),
        random_seed=26081,
        scorer_spec=ScorerSpec(
            name="recency_weighted_frequency_score",
            parameters={"window": 30, "decay": 0.93},
        ),
    )


@pytest.fixture(scope="module")
def portfolio_selection(candidate_pool: CandidatePool) -> PortfolioSelection:
    return optimize_portfolio(
        candidate_pool,
        random_seed=20260718,
        search_trials=8_000,
    )


def test_candidate_pool_has_at_least_ten_thousand_unique_scored_tickets(
    candidate_pool: CandidatePool,
) -> None:
    assert len(candidate_pool.candidates) == 10_000
    assert (
        len(
            {
                (candidate.front_numbers, candidate.back_numbers)
                for candidate in candidate_pool.candidates
            }
        )
        == 10_000
    )
    assert candidate_pool.random_seed == 26081
    assert candidate_pool.data_cutoff_issue == str(make_history().iloc[-1]["issue"])
    assert candidate_pool.scorer_parameters["window"] == 30
    assert candidate_pool.scorer_spec.name == "recency_weighted_frequency_score"
    assert candidate_pool.risk_disclaimer == DISCLAIMER
    first = candidate_pool.candidates[0]
    assert 0 <= first.number_score <= 1
    assert 0 <= first.structure_score <= 1
    assert 0 <= first.combined_ticket_score <= 1
    assert first.structure_component_scores
    assert first.structure_component_details


def test_candidate_pool_rejects_fewer_than_ten_thousand() -> None:
    with pytest.raises(ValueError, match="at least"):
        generate_candidate_pool(
            make_history(),
            target_issue="26081",
            generated_at=datetime(2026, 7, 18, tzinfo=UTC),
            random_seed=1,
            candidate_count=9_999,
        )


def test_candidate_score_details_round_trip(
    candidate_pool: CandidatePool,
    tmp_path: Path,
) -> None:
    target = tmp_path / "candidate_scores.jsonl"

    write_candidate_score_details(candidate_pool, target)
    loaded = load_candidate_score_details(target)

    assert loaded == candidate_pool
    assert DISCLAIMER in target.read_text(encoding="utf-8").splitlines()[0]


def test_portfolio_optimizer_satisfies_all_default_constraints(
    portfolio_selection: PortfolioSelection,
) -> None:
    selection = portfolio_selection

    assert len(selection.tickets) == 5
    assert 16 <= selection.front_pool_size <= 21
    assert len({ticket.back_numbers for ticket in selection.tickets}) == 5
    assert len({ticket.sum_interval for ticket in selection.tickets}) >= 3
    assert len({ticket.zone_structure for ticket in selection.tickets}) >= 3
    assert sum(ticket.ticket_role == "core_stable" for ticket in selection.tickets) == 3
    assert sum(ticket.ticket_role == "exploration" for ticket in selection.tickets) == 2
    assert all(
        len(set(left.front_numbers).intersection(right.front_numbers)) <= 2
        for left, right in combinations(selection.tickets, 2)
    )
    counts = Counter(number for ticket in selection.tickets for number in ticket.front_numbers)
    assert 2 <= len(selection.core_front_numbers) <= 3
    assert all(counts[number] in (2, 3) for number in selection.core_front_numbers)
    assert sum(counts[number] == 3 for number in selection.core_front_numbers) <= 1
    assert 2 <= len(selection.support_front_numbers) <= 4
    assert all(counts[number] == 2 for number in selection.support_front_numbers)
    repeated = {number for number, count in counts.items() if count >= 2}
    assert repeated == set(selection.core_front_numbers).union(selection.support_front_numbers)
    assert max(counts.values()) <= 3
    assert selection.risk_disclaimer == DISCLAIMER

    prediction = selection.to_prediction_record()
    assert prediction.random_seed == 20260718
    assert prediction.parameters["constraints"]["total_budget"] == 10.0
    assert prediction.risk_disclaimer == DISCLAIMER


def test_portfolio_defaults_use_target_pool_not_largest_pool_reward() -> None:
    constraints = PortfolioConstraints()

    assert constraints.min_front_pool_size == 16
    assert constraints.target_front_pool_size == 18
    assert constraints.max_front_pool_size == 21


def test_hard_constraint_rejects_repeated_back_combination(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][1]["back_numbers"] = payload["tickets"][0]["back_numbers"]

    with pytest.raises(ValidationError, match="back combinations must not repeat"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_pairwise_front_overlap_above_two(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][1]["front_numbers"] = payload["tickets"][0]["front_numbers"]

    with pytest.raises(ValidationError, match="pairwise front overlap"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_requires_every_repeat_to_be_classified(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["support_front_numbers"] = payload["support_front_numbers"][1:]

    with pytest.raises(ValidationError, match="every repeated number"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_wrong_stable_exploration_split(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][0]["ticket_role"] = "exploration"

    with pytest.raises(ValidationError, match="stable ticket count"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_insufficient_structure_coverage(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    for ticket in payload["tickets"]:
        ticket["sum_interval"] = "one_interval"

    with pytest.raises(ValidationError, match="sum interval coverage"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_insufficient_zone_coverage(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    for ticket in payload["tickets"]:
        ticket["zone_structure"] = "one_zone_signature"

    with pytest.raises(ValidationError, match="zone structure coverage"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_pool_outside_configured_range(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    pool_size = payload["front_pool_size"]
    payload["constraints"]["min_front_pool_size"] = 5
    payload["constraints"]["max_front_pool_size"] = pool_size - 1
    payload["constraints"]["target_front_pool_size"] = pool_size - 1

    with pytest.raises(ValidationError, match="front pool size"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_more_than_three_occurrences(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][3]["front_numbers"] = (9, 18, 21, 27, 28)
    payload["front_pool_size"] = len(
        {number for ticket in payload["tickets"] for number in ticket["front_numbers"]}
    )

    with pytest.raises(ValidationError, match="maximum occurrence"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_core_occurrence_below_configured_minimum(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["constraints"]["core_min_occurrences"] = 3

    with pytest.raises(ValidationError, match="core front number occurrences"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_too_many_core_numbers(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["constraints"]["max_core_front_numbers"] = 2

    with pytest.raises(ValidationError, match="core front number count"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_too_many_support_numbers(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["constraints"]["max_support_front_numbers"] = 2

    with pytest.raises(ValidationError, match="support front number count"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_support_number_above_two_occurrences(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][1]["front_numbers"] = (9, 21, 22, 25, 35)
    payload["front_pool_size"] = len(
        {number for ticket in payload["tickets"] for number in ticket["front_numbers"]}
    )

    with pytest.raises(ValidationError, match="support front number occurrences"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_wrong_ticket_count(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"] = payload["tickets"][:4]

    with pytest.raises(ValidationError):
        PortfolioSelection.model_validate(payload)


def test_constraint_model_rejects_budget_mismatch() -> None:
    with pytest.raises(ValidationError, match="total budget"):
        PortfolioConstraints(total_budget=12)
