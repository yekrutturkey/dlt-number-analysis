"""滚动回测的逐期、随机分布和跨期汇总指标模型。"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dlt_number_analysis import DISCLAIMER


class BacktestPeriodResult(BaseModel):
    """一个策略在一个历史目标期的事后回测指标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    strategy_name: str = Field(min_length=1)
    random_seed: int
    best_front_hits: int = Field(ge=0, le=5)
    best_back_hits: int = Field(ge=0, le=2)
    best_total_hits: int = Field(ge=0, le=7)
    at_least_three_front: bool
    at_least_2_plus_1: bool
    hit_concentration_ratio: float = Field(ge=0, le=1)
    any_prize: bool | None
    front_pool_coverage: int = Field(ge=0, le=5)
    back_pool_coverage: int = Field(ge=0, le=2)
    total_cost: Decimal = Field(gt=0)
    total_prize: Decimal | None = Field(ge=0)
    roi: Decimal | None
    prize_rule_version: str | None
    prize_data_available: bool
    random_baseline_seed_count: int = Field(ge=1000)
    best_total_hits_percentile: float = Field(ge=0, le=100)
    best_total_hits_percentile_ci_lower: float = Field(ge=0, le=100)
    best_total_hits_percentile_ci_upper: float = Field(ge=0, le=100)
    random_best_total_hits_mean_ci_lower: float = Field(ge=0, le=7)
    random_best_total_hits_mean_ci_upper: float = Field(ge=0, le=7)


class RandomBaselineSummary(BaseModel):
    """一个历史时点至少 1000 个随机种子的命中分布摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    seed_start: int
    seed_count: int = Field(ge=1000)
    metric: str = "best_total_hits"
    mean: float = Field(ge=0, le=7)
    bootstrap_confidence_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_ci_lower: float = Field(ge=0, le=7)
    bootstrap_ci_upper: float = Field(ge=0, le=7)


class BacktestStrategySummary(BaseModel):
    """一个策略跨全部已回测历史时点的汇总指标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_name: str = Field(min_length=1)
    period_count: int = Field(ge=1)
    at_least_three_front_rate: float = Field(ge=0, le=1)
    at_least_2_plus_1_rate: float = Field(ge=0, le=1)
    hit_concentration_ratio: float = Field(ge=0, le=1)
    average_prize: Decimal | None = Field(ge=0)
    median_prize: Decimal | None = Field(ge=0)
    longest_no_prize_streak: int | None = Field(ge=0)
    monetary_metrics_complete: bool


class BacktestReport(BaseModel):
    """包含随机 baseline 的严格向前滚动回测输出。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[BacktestPeriodResult, ...]
    strategy_summaries: tuple[BacktestStrategySummary, ...]
    random_baseline_summaries: tuple[RandomBaselineSummary, ...]
    includes_random_baseline: bool
    methodology: str
    comparative_claim: str
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_risk_disclaimer(cls, value: str) -> str:
        """回测输出必须携带固定风险声明。"""
        if value != DISCLAIMER:
            raise ValueError("回测输出缺少固定风险声明")
        return value
