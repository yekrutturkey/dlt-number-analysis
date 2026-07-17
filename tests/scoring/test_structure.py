"""基于历史经验分布的动态结构评分测试。"""

from datetime import date, timedelta
from random import Random

import numpy as np
import pandas as pd
import pytest

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import CSV_COLUMNS, DrawRecord
from dlt_number_analysis.scoring import (
    compute_ticket_structure,
    compute_ticket_structures_batch,
    fit_structure_profile,
    score_ticket_structure,
)


def make_history(count: int = 40) -> pd.DataFrame:
    """构造具有多样结构的测试历史。"""
    rng = Random(303)
    start = date(2026, 1, 1)
    return pd.DataFrame(
        [
            [
                str(26001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                *sorted(rng.sample(range(1, 36), 5)),
                *sorted(rng.sample(range(1, 13), 2)),
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def last_draw(history: pd.DataFrame) -> DrawRecord:
    """把测试历史最后一期转成 DrawRecord。"""
    return DrawRecord.model_validate(history.iloc[-1].to_dict())


def test_structure_profile_uses_historical_quantiles_not_fixed_ranges() -> None:
    history = make_history()
    profile = fit_structure_profile(history)

    expected = np.quantile(profile.distributions["front_sum"], (1 / 3, 2 / 3))

    assert profile.front_sum_quantile_edges == (float(expected[0]), float(expected[1]))
    assert profile.data_cutoff_issue == str(history.iloc[-1]["issue"])
    assert profile.risk_disclaimer == DISCLAIMER


def test_ticket_structure_contains_all_requested_front_and_back_features() -> None:
    history = make_history()
    previous = last_draw(history)

    features = compute_ticket_structure(
        [2, 13, 20, 25, 32],
        [8, 11],
        previous_draw=previous,
    )

    assert features.front_sum == 92
    assert features.front_span == 30
    assert features.front_odd_count + features.front_even_count == 5
    assert features.front_large_count + features.front_small_count == 5
    assert (
        features.front_zone_1_count + features.front_zone_2_count + features.front_zone_3_count == 5
    )
    assert features.back_sum == 19
    assert features.back_odd_count + features.back_even_count == 2
    assert features.back_large_count + features.back_small_count == 2


def test_numpy_batch_features_exactly_match_scalar_features() -> None:
    history = make_history()
    previous = last_draw(history)
    fronts = np.asarray(
        [
            [2, 13, 20, 25, 32],
            [1, 2, 11, 21, 35],
            [7, 12, 18, 24, 30],
        ],
        dtype=np.int64,
    )
    backs = np.asarray([[8, 11], [1, 2], [6, 7]], dtype=np.int64)

    batch = compute_ticket_structures_batch(fronts, backs, previous_draw=previous)
    scalar = tuple(
        compute_ticket_structure(front, back, previous_draw=previous)
        for front, back in zip(fronts, backs, strict=True)
    )

    assert batch == scalar


def test_structure_score_saves_each_empirical_component() -> None:
    history = make_history()
    profile = fit_structure_profile(history)
    features = compute_ticket_structure(
        [2, 13, 20, 25, 32],
        [8, 11],
        previous_draw=last_draw(history),
    )

    result = score_ticket_structure(features, profile)

    assert 0 <= result.overall_score <= 1
    assert "front_sum" in result.component_scores
    assert "front_repeat_from_previous_count" in result.component_scores
    assert "back_consecutive_pair_count" in result.component_scores
    assert result.risk_disclaimer == DISCLAIMER


def test_structure_profile_has_non_duplicated_concepts_and_normalized_weights() -> None:
    profile = fit_structure_profile(make_history())
    definitions = {item.name: item for item in profile.feature_definitions}

    assert set(definitions) == {
        "front_sum",
        "front_span",
        "front_odd_count",
        "front_large_count",
        "front_zone_signature",
        "front_consecutive_pair_count",
        "front_same_tail_pair_count",
        "front_repeat_from_previous_count",
        "back_sum",
        "back_odd_count",
        "back_large_count",
        "back_consecutive_pair_count",
        "back_repeat_from_previous_count",
    }
    assert "front_even_count" not in definitions
    assert "front_small_count" not in definitions
    assert sum(item.weight for item in definitions.values()) == pytest.approx(1.0)
    assert definitions["front_sum"].method == "empirical_quantile_centrality"
    assert definitions["front_zone_signature"].method == "laplace_smoothed_empirical_frequency"


def test_structure_score_persists_method_parameters_and_weight_per_component() -> None:
    history = make_history()
    profile = fit_structure_profile(history, laplace_alpha=2.0)
    features = compute_ticket_structure(
        [2, 13, 20, 25, 32],
        [8, 11],
        previous_draw=last_draw(history),
    )

    result = score_ticket_structure(features, profile)

    continuous = result.component_details["front_sum"]
    discrete = result.component_details["front_zone_signature"]
    assert continuous.method == "empirical_quantile_centrality"
    assert continuous.parameters["history_observations"] == len(history)
    assert discrete.method == "laplace_smoothed_empirical_frequency"
    assert discrete.parameters["alpha"] == 2.0
    assert discrete.score > 0
    assert sum(item.weight for item in result.component_details.values()) == pytest.approx(1.0)
