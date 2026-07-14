"""可替换号码评分器测试。"""

from datetime import date, timedelta
from functools import partial
from random import Random

import pandas as pd
import pytest

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.scoring import (
    ScorerSpec,
    build_number_scorer,
    cumulative_frequency_score,
    hot_cold_blend_score,
    recency_weighted_frequency_score,
    uniform_score,
)


def make_history(count: int = 120) -> pd.DataFrame:
    """构造按时间递增且不代表真实开奖的历史窗口。"""
    rng = Random(20260714)
    start = date(2025, 1, 1)
    records = []
    for index in range(count):
        records.append(
            [
                str(25001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                *sorted(rng.sample(range(1, 36), 5)),
                *sorted(rng.sample(range(1, 13), 2)),
            ]
        )
    return pd.DataFrame(records, columns=CSV_COLUMNS)


def test_uniform_score_is_equal_over_the_complete_number_space() -> None:
    history = make_history(3)

    front = uniform_score(history, area="front")
    back = uniform_score(history, area="back")

    assert set(front) == set(range(1, 36))
    assert set(back) == set(range(1, 13))
    assert len(set(front.values())) == 1
    assert len(set(back.values())) == 1


def test_cumulative_frequency_uses_only_supplied_history() -> None:
    history = pd.DataFrame(
        [
            ["26001", "2026-01-01", 1, 2, 3, 4, 5, 1, 2],
            ["26002", "2026-01-03", 1, 6, 7, 8, 9, 1, 3],
        ],
        columns=CSV_COLUMNS,
    )

    front = cumulative_frequency_score(history, area="front")
    back = cumulative_frequency_score(history, area="back")

    assert front[1] == pytest.approx(2 / 10)
    assert front[35] == 0
    assert back[1] == pytest.approx(2 / 4)
    assert back[12] == 0


@pytest.mark.parametrize("window", [10, 30, 100])
def test_recency_weighted_frequency_supports_required_windows(window: int) -> None:
    scorer = partial(recency_weighted_frequency_score, window=window)

    scores = scorer(make_history(), area="front")

    assert set(scores) == set(range(1, 36))
    assert sum(scores.values()) == pytest.approx(1.0)


def test_recency_weighted_frequency_rejects_unsupported_window() -> None:
    with pytest.raises(ValueError, match="window"):
        recency_weighted_frequency_score(make_history(), area="front", window=20)


def test_hot_cold_blend_is_finite_bounded_and_seed_free() -> None:
    history = make_history()

    first = hot_cold_blend_score(history, area="back")
    second = hot_cold_blend_score(history, area="back")

    assert first == second
    assert set(first) == set(range(1, 13))
    assert all(0 <= value <= 1 for value in first.values())


def test_scorer_spec_binds_logged_parameters_to_actual_invocation() -> None:
    history = make_history()
    spec = ScorerSpec(
        name="recency_weighted_frequency_score",
        parameters={"window": 10, "decay": 0.8},
    )

    scorer = build_number_scorer(spec)

    assert scorer.spec == spec
    assert scorer(history, area="front") == recency_weighted_frequency_score(
        history,
        area="front",
        window=10,
        decay=0.8,
    )


def test_scorer_spec_rejects_parameters_not_consumed_by_implementation() -> None:
    with pytest.raises(ValueError, match="unsupported scorer parameters"):
        ScorerSpec(name="uniform_score", parameters={"window": 10})


def test_scorer_spec_materializes_defaults_for_audit_logs() -> None:
    spec = ScorerSpec(name="recency_weighted_frequency_score")

    assert spec.parameters == {"window": 30, "decay": 0.93}
