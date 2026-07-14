"""只消费目标期之前历史数据的可替换号码评分器。"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Literal, Protocol

import pandas as pd

from dlt_number_analysis.data import BACK_COLUMNS, FRONT_COLUMNS
from dlt_number_analysis.data.data_validator import validate_draw_dataframe

NumberArea = Literal["front", "back"]
RECENCY_WINDOWS: tuple[int, int, int] = (10, 30, 100)


class NumberScorer(Protocol):
    """候选生成器和滚动回测使用的统一号码评分接口。"""

    def __call__(
        self,
        history: pd.DataFrame,
        *,
        area: NumberArea,
    ) -> Mapping[int, float]: ...


def _area_definition(area: NumberArea) -> tuple[tuple[str, ...], int]:
    """返回指定区域的列和号码上限。"""
    if area == "front":
        return FRONT_COLUMNS, 35
    if area == "back":
        return BACK_COLUMNS, 12
    raise ValueError(f"未知号码区域：{area}")


def _frequency_score(
    history: pd.DataFrame,
    *,
    area: NumberArea,
    row_weights: tuple[float, ...],
) -> dict[int, float]:
    """按行权重计算完整号码空间的加权出现频率。"""
    columns, maximum = _area_definition(area)
    counts = dict.fromkeys(range(1, maximum + 1), 0.0)
    for weight, values in zip(
        row_weights,
        history.loc[:, columns].itertuples(index=False, name=None),
        strict=True,
    ):
        for value in values:
            counts[int(value)] += weight
    denominator = sum(row_weights) * len(columns)
    return {number: count / denominator for number, count in counts.items()}


def _min_max_scale(scores: Mapping[int, float]) -> dict[int, float]:
    """把任意有限评分缩放到 0–1；完全相同时给中性分 0.5。"""
    if not scores or any(not isfinite(float(value)) for value in scores.values()):
        raise ValueError("号码评分必须是非空有限数值映射")
    minimum = min(float(value) for value in scores.values())
    maximum = max(float(value) for value in scores.values())
    if maximum == minimum:
        return {number: 0.5 for number in scores}
    return {
        number: (float(value) - minimum) / (maximum - minimum) for number, value in scores.items()
    }


def uniform_score(history: pd.DataFrame, *, area: NumberArea) -> dict[int, float]:
    """为指定区域所有号码返回完全相同的基准分。"""
    validate_draw_dataframe(history)
    _, maximum = _area_definition(area)
    score = 1.0 / maximum
    return dict.fromkeys(range(1, maximum + 1), score)


def cumulative_frequency_score(
    history: pd.DataFrame,
    *,
    area: NumberArea,
) -> dict[int, float]:
    """用传入历史窗口内的累计出现频率评分。"""
    validated = validate_draw_dataframe(history)
    return _frequency_score(
        validated,
        area=area,
        row_weights=tuple(1.0 for _ in range(len(validated))),
    )


def recency_weighted_frequency_score(
    history: pd.DataFrame,
    *,
    area: NumberArea,
    window: int = 30,
    decay: float = 0.93,
) -> dict[int, float]:
    """在最近 10、30 或 100 期中对越新的开奖记录赋予越高权重。"""
    if window not in RECENCY_WINDOWS:
        raise ValueError(f"window 必须是 {RECENCY_WINDOWS} 之一")
    if not 0 < decay <= 1 or not isfinite(decay):
        raise ValueError("decay 必须是 (0, 1] 内的有限数值")
    validated = validate_draw_dataframe(history)
    selected = validated.iloc[-window:].copy()
    row_weights = tuple(decay**age for age in reversed(range(len(selected))))
    return _frequency_score(selected, area=area, row_weights=row_weights)


def hot_cold_blend_score(
    history: pd.DataFrame,
    *,
    area: NumberArea,
    hot_window: int = 10,
    cold_window: int = 100,
    hot_weight: float = 0.65,
    decay: float = 0.93,
) -> dict[int, float]:
    """混合近期热度与较长窗口中的低频度，输出工程评分而非概率。"""
    if hot_window not in RECENCY_WINDOWS or cold_window not in RECENCY_WINDOWS:
        raise ValueError(f"hot_window 和 cold_window 必须来自 {RECENCY_WINDOWS}")
    if hot_window > cold_window:
        raise ValueError("hot_window 不得大于 cold_window")
    if not 0 <= hot_weight <= 1 or not isfinite(hot_weight):
        raise ValueError("hot_weight 必须是 [0, 1] 内的有限数值")

    validated = validate_draw_dataframe(history)
    hot = recency_weighted_frequency_score(
        validated,
        area=area,
        window=hot_window,
        decay=decay,
    )
    cold_history = validated.iloc[-cold_window:].copy()
    long_frequency = cumulative_frequency_score(cold_history, area=area)
    hot_scaled = _min_max_scale(hot)
    long_scaled = _min_max_scale(long_frequency)
    cold_component = {number: 1.0 - value for number, value in long_scaled.items()}
    return {
        number: hot_weight * hot_scaled[number] + (1.0 - hot_weight) * cold_component[number]
        for number in hot_scaled
    }
