"""仅使用当前及过去开奖数据构建可解释统计特征。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import comb
from numbers import Integral

import pandas as pd

from dlt_number_analysis.data.data_validator import FRONT_COLUMNS, validate_draw_dataframe

FEATURE_COLUMNS: tuple[str, ...] = (
    "front_sum",
    "front_span",
    "odd_even_ratio",
    "large_small_ratio",
    "zone_ratio",
    "consecutive_pair_count",
    "same_tail_pair_count",
    "repeat_from_previous_count",
)


@dataclass(frozen=True, slots=True)
class FrontFeatures:
    """单期前区号码的静态统计特征。"""

    front_sum: int
    front_span: int
    odd_even_ratio: str
    large_small_ratio: str
    zone_ratio: str
    consecutive_pair_count: int
    same_tail_pair_count: int


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
        odd_even_ratio=f"{odd_count}:{even_count}",
        large_small_ratio=f"{large_count}:{small_count}",
        zone_ratio=f"{zone_1_count}:{zone_2_count}:{zone_3_count}",
        consecutive_pair_count=consecutive_pair_count,
        same_tail_pair_count=same_tail_pair_count,
    )


def engineer_features(draws: pd.DataFrame) -> pd.DataFrame:
    """校验开奖表并按时间向前计算特征，不读取任何未来期次。

    第一行没有可比较的上一期，因此 ``repeat_from_previous_count`` 为
    ``pandas.NA``，其余行只与紧邻的上一行比较。
    """
    validated = validate_draw_dataframe(draws)
    feature_records: list[dict[str, int | str]] = []
    repeat_counts: list[int | None] = []
    previous_front: set[int] | None = None

    for front_values in validated.loc[:, FRONT_COLUMNS].itertuples(index=False, name=None):
        current_front = {int(number) for number in front_values}
        features = compute_front_features(front_values)
        feature_records.append(asdict(features))
        repeat_counts.append(
            None if previous_front is None else len(current_front.intersection(previous_front))
        )
        previous_front = current_front

    result = validated.copy()
    static_features = pd.DataFrame(feature_records, index=result.index)
    for column in FEATURE_COLUMNS[:-1]:
        result[column] = static_features[column]
    result["repeat_from_previous_count"] = pd.array(repeat_counts, dtype="Int64")
    return result
