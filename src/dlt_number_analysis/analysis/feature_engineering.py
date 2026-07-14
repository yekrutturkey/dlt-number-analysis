"""仅使用当前及过去开奖数据构建可解释统计特征。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import comb
from numbers import Integral

import pandas as pd

from dlt_number_analysis.data.data_validator import (
    BACK_COLUMNS,
    FRONT_COLUMNS,
    validate_draw_dataframe,
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "front_sum",
    "front_span",
    "front_odd_count",
    "front_even_count",
    "front_large_count",
    "front_small_count",
    "front_zone_1_count",
    "front_zone_2_count",
    "front_zone_3_count",
    "odd_even_ratio",
    "large_small_ratio",
    "zone_ratio",
    "consecutive_pair_count",
    "same_tail_pair_count",
    "repeat_from_previous_count",
    "back_sum",
    "back_odd_count",
    "back_even_count",
    "back_large_count",
    "back_small_count",
    "back_consecutive_pair_count",
    "back_repeat_from_previous_count",
)


@dataclass(frozen=True, slots=True)
class FrontFeatures:
    """单期前区号码的静态统计特征。"""

    front_sum: int
    front_span: int
    front_odd_count: int
    front_even_count: int
    front_large_count: int
    front_small_count: int
    front_zone_1_count: int
    front_zone_2_count: int
    front_zone_3_count: int
    odd_even_ratio: str
    large_small_ratio: str
    zone_ratio: str
    consecutive_pair_count: int
    same_tail_pair_count: int


@dataclass(frozen=True, slots=True)
class BackFeatures:
    """单期后区号码的静态统计特征。"""

    back_sum: int
    back_odd_count: int
    back_even_count: int
    back_large_count: int
    back_small_count: int
    back_consecutive_pair_count: int


def _normalize_front_numbers(front_numbers: Sequence[int]) -> tuple[int, ...]:
    """校验独立的前区号码序列并返回升序元组。"""
    if len(front_numbers) != 5:
        raise ValueError("前区必须恰好包含 5 个号码")
    if any(
        isinstance(number, bool) or not isinstance(number, Integral) for number in front_numbers
    ):
        raise ValueError("前区号码必须是整数")

    normalized = tuple(sorted(int(number) for number in front_numbers))
    if any(number < 1 or number > 35 for number in normalized):
        raise ValueError("前区号码必须在 1 到 35 之间")
    if len(set(normalized)) != len(normalized):
        raise ValueError("前区号码不得重复")
    return normalized


def compute_front_features(front_numbers: Sequence[int]) -> FrontFeatures:
    """计算不依赖其他期次的单期前区统计特征。"""
    numbers = _normalize_front_numbers(front_numbers)
    odd_count = sum(number % 2 == 1 for number in numbers)
    even_count = len(numbers) - odd_count
    large_count = sum(number >= 18 for number in numbers)
    small_count = len(numbers) - large_count
    zone_1_count = sum(number <= 12 for number in numbers)
    zone_2_count = sum(13 <= number <= 24 for number in numbers)
    zone_3_count = len(numbers) - zone_1_count - zone_2_count
    consecutive_pair_count = sum(current - previous == 1 for previous, current in pairwise(numbers))
    tail_counts = Counter(number % 10 for number in numbers)
    same_tail_pair_count = sum(comb(count, 2) for count in tail_counts.values() if count >= 2)

    return FrontFeatures(
        front_sum=sum(numbers),
        front_span=numbers[-1] - numbers[0],
        front_odd_count=odd_count,
        front_even_count=even_count,
        front_large_count=large_count,
        front_small_count=small_count,
        front_zone_1_count=zone_1_count,
        front_zone_2_count=zone_2_count,
        front_zone_3_count=zone_3_count,
        odd_even_ratio=f"{odd_count}:{even_count}",
        large_small_ratio=f"{large_count}:{small_count}",
        zone_ratio=f"{zone_1_count}:{zone_2_count}:{zone_3_count}",
        consecutive_pair_count=consecutive_pair_count,
        same_tail_pair_count=same_tail_pair_count,
    )


def compute_back_features(back_numbers: Sequence[int]) -> BackFeatures:
    """计算后区和值、奇偶、大小和连号等静态特征。

    后区 1–6 定义为小号，7–12 定义为大号。
    """
    if len(back_numbers) != 2:
        raise ValueError("后区必须恰好包含 2 个号码")
    if any(isinstance(number, bool) or not isinstance(number, Integral) for number in back_numbers):
        raise ValueError("后区号码必须是整数")

    numbers = tuple(sorted(int(number) for number in back_numbers))
    if any(number < 1 or number > 12 for number in numbers):
        raise ValueError("后区号码必须在 1 到 12 之间")
    if len(set(numbers)) != len(numbers):
        raise ValueError("后区号码不得重复")

    odd_count = sum(number % 2 == 1 for number in numbers)
    large_count = sum(number >= 7 for number in numbers)
    return BackFeatures(
        back_sum=sum(numbers),
        back_odd_count=odd_count,
        back_even_count=len(numbers) - odd_count,
        back_large_count=large_count,
        back_small_count=len(numbers) - large_count,
        back_consecutive_pair_count=int(numbers[1] - numbers[0] == 1),
    )


def engineer_features(draws: pd.DataFrame) -> pd.DataFrame:
    """校验开奖表并按时间向前计算特征，不读取任何未来期次。

    第一行没有可比较的上一期，因此 ``repeat_from_previous_count`` 为
    ``pandas.NA``，其余行只与紧邻的上一行比较。
    """
    validated = validate_draw_dataframe(draws)
    feature_records: list[dict[str, int | str]] = []
    front_repeat_counts: list[int | None] = []
    back_repeat_counts: list[int | None] = []
    previous_front: set[int] | None = None
    previous_back: set[int] | None = None

    selected_columns = (*FRONT_COLUMNS, *BACK_COLUMNS)
    for values in validated.loc[:, selected_columns].itertuples(index=False, name=None):
        front_values = values[: len(FRONT_COLUMNS)]
        back_values = values[len(FRONT_COLUMNS) :]
        current_front = {int(number) for number in front_values}
        current_back = {int(number) for number in back_values}
        front_features = compute_front_features(front_values)
        back_features = compute_back_features(back_values)
        feature_records.append({**asdict(front_features), **asdict(back_features)})
        front_repeat_counts.append(
            None if previous_front is None else len(current_front.intersection(previous_front))
        )
        back_repeat_counts.append(
            None if previous_back is None else len(current_back.intersection(previous_back))
        )
        previous_front = current_front
        previous_back = current_back

    result = validated.copy()
    static_features = pd.DataFrame(feature_records, index=result.index)
    for column in static_features.columns:
        result[column] = static_features[column]
    result["repeat_from_previous_count"] = pd.array(front_repeat_counts, dtype="Int64")
    result["back_repeat_from_previous_count"] = pd.array(back_repeat_counts, dtype="Int64")
    result = result.loc[:, (*validated.columns, *FEATURE_COLUMNS)]
    return result
