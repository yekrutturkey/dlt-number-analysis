"""统计特征工程测试。"""

import pandas as pd
import pytest

from dlt_number_analysis.analysis import (
    compute_back_features,
    compute_front_features,
    engineer_features,
)
from dlt_number_analysis.data import CSV_COLUMNS


def make_valid_draws() -> pd.DataFrame:
    """构造不代表真实开奖的合法测试数据。"""
    return pd.DataFrame(
        [
            ["07001", "2007-05-28", 1, 2, 11, 20, 35, 3, 8],
            ["07002", "2007-05-30", 2, 7, 12, 23, 31, 4, 9],
            ["07003", "2007-06-02", 5, 15, 16, 25, 35, 1, 12],
        ],
        columns=CSV_COLUMNS,
    )


def test_compute_front_features_uses_documented_pair_definitions() -> None:
    features = compute_front_features([1, 2, 3, 11, 21])

    assert features.front_sum == 38
    assert features.front_span == 20
    assert features.front_odd_count == 4
    assert features.front_even_count == 1
    assert features.front_large_count == 1
    assert features.front_small_count == 4
    assert features.front_zone_1_count == 4
    assert features.front_zone_2_count == 1
    assert features.front_zone_3_count == 0
    assert features.odd_even_ratio == "4:1"
    assert features.large_small_ratio == "1:4"
    assert features.zone_ratio == "4:1:0"
    assert features.consecutive_pair_count == 2
    assert features.same_tail_pair_count == 3


def test_compute_back_features_uses_six_as_small_number_boundary() -> None:
    features = compute_back_features([6, 7])

    assert features.back_sum == 13
    assert features.back_odd_count == 1
    assert features.back_even_count == 1
    assert features.back_large_count == 1
    assert features.back_small_count == 1
    assert features.back_consecutive_pair_count == 1


@pytest.mark.parametrize("numbers", [[1], [1, 1], [1, 13], [1, 2.0]])
def test_compute_back_features_rejects_invalid_inputs(numbers: list[object]) -> None:
    with pytest.raises(ValueError):
        compute_back_features(numbers)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "numbers",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4, 36],
        [1, 2, 3, 3, 4],
        [1, 2, 3, 4, 5.0],
    ],
)
def test_compute_front_features_rejects_invalid_inputs(numbers: list[object]) -> None:
    with pytest.raises(ValueError):
        compute_front_features(numbers)  # type: ignore[arg-type]


def test_engineer_features_calculates_all_requested_features() -> None:
    result = engineer_features(make_valid_draws())

    first = result.iloc[0]
    assert first["front_sum"] == 69
    assert first["front_span"] == 34
    assert first["front_odd_count"] == 3
    assert first["front_even_count"] == 2
    assert first["front_large_count"] == 2
    assert first["front_small_count"] == 3
    assert first["front_zone_1_count"] == 3
    assert first["front_zone_2_count"] == 1
    assert first["front_zone_3_count"] == 1
    assert first["odd_even_ratio"] == "3:2"
    assert first["large_small_ratio"] == "2:3"
    assert first["zone_ratio"] == "3:1:1"
    assert first["consecutive_pair_count"] == 1
    assert first["same_tail_pair_count"] == 1
    assert pd.isna(first["repeat_from_previous_count"])
    assert first["back_sum"] == 11
    assert first["back_odd_count"] == 1
    assert first["back_even_count"] == 1
    assert first["back_large_count"] == 1
    assert first["back_small_count"] == 1
    assert first["back_consecutive_pair_count"] == 0
    assert pd.isna(first["back_repeat_from_previous_count"])

    assert result.loc[1, "repeat_from_previous_count"] == 1
    assert result.loc[2, "repeat_from_previous_count"] == 0
    assert result.loc[1, "back_repeat_from_previous_count"] == 0
    assert result.loc[2, "back_repeat_from_previous_count"] == 0
    assert str(result["repeat_from_previous_count"].dtype) == "Int64"
    assert str(result["back_repeat_from_previous_count"].dtype) == "Int64"


def test_future_draw_does_not_change_earlier_features() -> None:
    draws = make_valid_draws()
    baseline = engineer_features(draws.iloc[:2].copy())

    draws.loc[2, ["front_1", "front_2", "front_3", "front_4", "front_5"]] = [
        2,
        6,
        17,
        28,
        34,
    ]
    with_future_changed = engineer_features(draws)

    pd.testing.assert_frame_equal(
        baseline.reset_index(drop=True),
        with_future_changed.iloc[:2].reset_index(drop=True),
    )
