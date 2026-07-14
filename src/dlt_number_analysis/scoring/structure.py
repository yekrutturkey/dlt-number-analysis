"""依据预测时点之前经验分布计算候选票据结构分数。"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.analysis import (
    compute_back_features,
    compute_front_features,
    engineer_features,
)
from dlt_number_analysis.data import DrawRecord
from dlt_number_analysis.data.data_validator import validate_draw_dataframe


class TicketStructureFeatures(BaseModel):
    """一注候选的前后区结构特征。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    front_sum: int
    front_span: int
    front_odd_count: int
    front_even_count: int
    front_large_count: int
    front_small_count: int
    front_zone_1_count: int
    front_zone_2_count: int
    front_zone_3_count: int
    front_consecutive_pair_count: int
    front_same_tail_pair_count: int
    front_repeat_from_previous_count: int
    back_sum: int
    back_odd_count: int
    back_even_count: int
    back_large_count: int
    back_small_count: int
    back_consecutive_pair_count: int
    back_repeat_from_previous_count: int

    @property
    def zone_signature(self) -> str:
        """返回用于 Portfolio 结构覆盖的三区结构标识。"""
        return f"{self.front_zone_1_count}:{self.front_zone_2_count}:{self.front_zone_3_count}"


class HistoricalStructureProfile(BaseModel):
    """完全由目标期之前历史数据拟合的经验结构分布。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_start_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    history_size: int = Field(ge=1)
    distributions: dict[str, tuple[float, ...]]
    front_sum_quantile_edges: tuple[float, float]
    scoring_method: str = "empirical_midrank_centrality"
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        """结构经验分布输出必须包含固定风险声明。"""
        if value != DISCLAIMER:
            raise ValueError("结构经验分布缺少固定风险声明")
        return value

    def front_sum_interval(self, value: int) -> str:
        """按历史三分位边界返回动态和值区间标签。"""
        lower, upper = self.front_sum_quantile_edges
        if value <= lower:
            return "historical_lower_third"
        if value <= upper:
            return "historical_middle_third"
        return "historical_upper_third"


class StructureScore(BaseModel):
    """结构总分及逐项经验分布得分。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_score: float = Field(ge=0, le=1)
    component_scores: dict[str, float]
    scoring_method: str = "empirical_midrank_centrality"
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        """结构评分输出必须包含固定风险声明。"""
        if value != DISCLAIMER:
            raise ValueError("结构评分缺少固定风险声明")
        return value


_PROFILE_COLUMN_MAP: Mapping[str, str] = {
    "front_sum": "front_sum",
    "front_span": "front_span",
    "front_odd_count": "front_odd_count",
    "front_even_count": "front_even_count",
    "front_large_count": "front_large_count",
    "front_small_count": "front_small_count",
    "front_zone_1_count": "front_zone_1_count",
    "front_zone_2_count": "front_zone_2_count",
    "front_zone_3_count": "front_zone_3_count",
    "front_consecutive_pair_count": "consecutive_pair_count",
    "front_same_tail_pair_count": "same_tail_pair_count",
    "front_repeat_from_previous_count": "repeat_from_previous_count",
    "back_sum": "back_sum",
    "back_odd_count": "back_odd_count",
    "back_even_count": "back_even_count",
    "back_large_count": "back_large_count",
    "back_small_count": "back_small_count",
    "back_consecutive_pair_count": "back_consecutive_pair_count",
    "back_repeat_from_previous_count": "back_repeat_from_previous_count",
}


def fit_structure_profile(history: pd.DataFrame) -> HistoricalStructureProfile:
    """从传入的历史窗口拟合经验分布，不读取窗口之外或未来数据。"""
    validated = validate_draw_dataframe(history)
    featured = engineer_features(validated)
    distributions: dict[str, tuple[float, ...]] = {}
    for feature_name, column in _PROFILE_COLUMN_MAP.items():
        values = featured[column].dropna().astype(float).tolist()
        distributions[feature_name] = tuple(sorted(values))
    front_sums = np.asarray(distributions["front_sum"], dtype=float)
    lower, upper = np.quantile(front_sums, (1 / 3, 2 / 3), method="linear")
    return HistoricalStructureProfile(
        data_start_issue=str(validated.iloc[0]["issue"]),
        data_cutoff_issue=str(validated.iloc[-1]["issue"]),
        history_size=len(validated),
        distributions=distributions,
        front_sum_quantile_edges=(float(lower), float(upper)),
    )


def compute_ticket_structure(
    front_numbers: Sequence[int],
    back_numbers: Sequence[int],
    *,
    previous_draw: DrawRecord | None,
) -> TicketStructureFeatures:
    """计算候选票据结构，并只与预测时点最后一期历史开奖比较重号。"""
    front_features = compute_front_features(front_numbers)
    back_features = compute_back_features(back_numbers)
    front = tuple(sorted(int(number) for number in front_numbers))
    back = tuple(sorted(int(number) for number in back_numbers))
    front_values = asdict(front_features)
    back_values = asdict(back_features)
    return TicketStructureFeatures(
        front_sum=front_values["front_sum"],
        front_span=front_values["front_span"],
        front_odd_count=front_values["front_odd_count"],
        front_even_count=front_values["front_even_count"],
        front_large_count=front_values["front_large_count"],
        front_small_count=front_values["front_small_count"],
        front_zone_1_count=front_values["front_zone_1_count"],
        front_zone_2_count=front_values["front_zone_2_count"],
        front_zone_3_count=front_values["front_zone_3_count"],
        front_consecutive_pair_count=front_values["consecutive_pair_count"],
        front_same_tail_pair_count=front_values["same_tail_pair_count"],
        front_repeat_from_previous_count=(
            0
            if previous_draw is None
            else len(set(front).intersection(previous_draw.front_numbers))
        ),
        back_sum=back_values["back_sum"],
        back_odd_count=back_values["back_odd_count"],
        back_even_count=back_values["back_even_count"],
        back_large_count=back_values["back_large_count"],
        back_small_count=back_values["back_small_count"],
        back_consecutive_pair_count=back_values["back_consecutive_pair_count"],
        back_repeat_from_previous_count=(
            0 if previous_draw is None else len(set(back).intersection(previous_draw.back_numbers))
        ),
    )


def _empirical_midrank_centrality(value: float, distribution: tuple[float, ...]) -> float:
    """以经验中位秩计算中心性，不使用固定号码结构区间。"""
    if not distribution:
        raise ValueError("经验分布不能为空")
    if value < distribution[0] or value > distribution[-1]:
        return 0.0
    lower = bisect_left(distribution, value)
    upper = bisect_right(distribution, value)
    midrank = (lower + (upper - lower) / 2) / len(distribution)
    return max(0.0, 1.0 - 2.0 * abs(midrank - 0.5))


def score_ticket_structure(
    features: TicketStructureFeatures,
    profile: HistoricalStructureProfile,
) -> StructureScore:
    """按历史经验分布逐项评分并取平均值。"""
    values = features.model_dump()
    component_scores = {
        name: _empirical_midrank_centrality(float(values[name]), distribution)
        for name, distribution in profile.distributions.items()
        if distribution
    }
    if not component_scores:
        raise ValueError("历史结构分布没有可评分的特征")
    return StructureScore(
        overall_score=sum(component_scores.values()) / len(component_scores),
        component_scores=component_scores,
    )
