"""只消费目标期之前历史数据的可替换号码评分器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationInfo, field_validator

from dlt_number_analysis.data import BACK_COLUMNS, FRONT_COLUMNS
from dlt_number_analysis.data.data_validator import validate_draw_dataframe

NumberArea = Literal["front", "back"]
RECENCY_WINDOWS: tuple[int, int, int] = (10, 30, 100)
NUMBER_SCORER_VERSION = "number-scorer-v1"
ScorerName = Literal[
    "uniform_score",
    "cumulative_frequency_score",
    "recency_weighted_frequency_score",
    "hot_cold_blend_score",
]


class ScorerSpec(BaseModel):
    """A versioned scorer configuration that is also the invocation source of truth."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    name: ScorerName
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    version: str = NUMBER_SCORER_VERSION

    @field_validator("parameters", mode="before")
    @classmethod
    def resolve_parameters(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> dict[str, JsonValue]:
        """Fill implementation defaults and reject parameters the scorer will not consume."""
        if not isinstance(value, Mapping):
            raise ValueError("parameters must be a mapping")
        name = info.data.get("name")
        supplied = dict(value)
        defaults: dict[str, dict[str, JsonValue]] = {
            "uniform_score": {},
            "cumulative_frequency_score": {},
            "recency_weighted_frequency_score": {"window": 30, "decay": 0.93},
            "hot_cold_blend_score": {
                "hot_window": 10,
                "cold_window": 100,
                "hot_weight": 0.65,
                "decay": 0.93,
            },
        }
        if name not in defaults:
            return supplied
        unknown = set(supplied).difference(defaults[name])
        if unknown:
            raise ValueError(f"unsupported scorer parameters: {sorted(unknown)}")
        return {**defaults[name], **supplied}

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """Only bind specs to an implementation version available in this package."""
        if value != NUMBER_SCORER_VERSION:
            raise ValueError(f"unsupported scorer version: {value}")
        return value


class NumberScorer(Protocol):
    """候选生成器和滚动回测使用的统一号码评分接口。"""

    def __call__(
        self,
        history: pd.DataFrame,
        *,
        area: NumberArea,
    ) -> Mapping[int, float]: ...


@dataclass(frozen=True, slots=True)
class BoundNumberScorer:
    """Callable scorer whose audit spec is inseparable from its behavior."""

    spec: ScorerSpec

    def __call__(self, history: pd.DataFrame, *, area: NumberArea) -> Mapping[int, float]:
        parameters = dict(self.spec.parameters)
        if self.spec.name == "uniform_score":
            return uniform_score(history, area=area)
        if self.spec.name == "cumulative_frequency_score":
            return cumulative_frequency_score(history, area=area)
        if self.spec.name == "recency_weighted_frequency_score":
            return recency_weighted_frequency_score(
                history,
                area=area,
                window=int(parameters["window"]),
                decay=float(parameters["decay"]),
            )
        return hot_cold_blend_score(
            history,
            area=area,
            hot_window=int(parameters["hot_window"]),
            cold_window=int(parameters["cold_window"]),
            hot_weight=float(parameters["hot_weight"]),
            decay=float(parameters["decay"]),
        )


def build_number_scorer(spec: ScorerSpec) -> BoundNumberScorer:
    """Build the exact scorer described by ``spec`` after validating its parameters."""
    bound = BoundNumberScorer(spec=spec)
    # Exercise scalar validation without consuming or depending on any history.
    parameters = spec.parameters
    if spec.name == "recency_weighted_frequency_score":
        window = int(parameters["window"])
        decay = float(parameters["decay"])
        if window not in RECENCY_WINDOWS:
            raise ValueError(f"window must be one of {RECENCY_WINDOWS}")
        if not 0 < decay <= 1 or not isfinite(decay):
            raise ValueError("decay must be finite and in (0, 1]")
    elif spec.name == "hot_cold_blend_score":
        hot_window = int(parameters["hot_window"])
        cold_window = int(parameters["cold_window"])
        hot_weight = float(parameters["hot_weight"])
        decay = float(parameters["decay"])
        if hot_window not in RECENCY_WINDOWS or cold_window not in RECENCY_WINDOWS:
            raise ValueError(f"windows must come from {RECENCY_WINDOWS}")
        if hot_window > cold_window:
            raise ValueError("hot_window must not exceed cold_window")
        if not 0 <= hot_weight <= 1 or not isfinite(hot_weight):
            raise ValueError("hot_weight must be finite and in [0, 1]")
        if not 0 < decay <= 1 or not isfinite(decay):
            raise ValueError("decay must be finite and in (0, 1]")
    return bound


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
