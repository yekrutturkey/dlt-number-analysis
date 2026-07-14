"""候选票池、评分明细和五注 Portfolio 优化测试。"""

from collections import Counter
from datetime import UTC, date, datetime, timedelta
from itertools import combinations
from pathlib import Path
from random import Random

import pandas as pd
import pytest

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.portfolio import (
    CandidatePool,
    generate_candidate_pool,
    load_candidate_score_details,
    optimize_portfolio,
    write_candidate_score_details,
)
from dlt_number_analysis.scoring import recency_weighted_frequency_score


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
        number_scorer=thirty_period_scorer,
        scorer_name="recency_weighted_frequency_score",
        scorer_parameters={"window": 30, "decay": 0.93},
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
    assert candidate_pool.risk_disclaimer == DISCLAIMER
    first = candidate_pool.candidates[0]
    assert 0 <= first.number_score <= 1
    assert 0 <= first.structure_score <= 1
    assert 0 <= first.combined_ticket_score <= 1
    assert first.structure_component_scores


def test_candidate_pool_rejects_fewer_than_ten_thousand() -> None:
    with pytest.raises(ValueError, match="至少"):
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
    candidate_pool: CandidatePool,
) -> None:
    selection = optimize_portfolio(
        candidate_pool,
        random_seed=20260718,
        search_trials=8_000,
    )

    assert len(selection.tickets) == 5
    assert 16 <= selection.front_pool_size <= 20
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
    assert selection.risk_disclaimer == DISCLAIMER

    prediction = selection.to_prediction_record()
    assert prediction.random_seed == 20260718
    assert prediction.parameters["constraints"]["total_budget"] == 10.0
    assert prediction.risk_disclaimer == DISCLAIMER
