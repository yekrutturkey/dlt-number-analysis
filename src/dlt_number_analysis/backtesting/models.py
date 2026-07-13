"""滚动回测的逐期指标与报告模型。"""

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
    any_prize: bool
    front_pool_coverage: int = Field(ge=0, le=5)
    back_pool_coverage: int = Field(ge=0, le=2)
    total_cost: Decimal = Field(gt=0)
    total_prize: Decimal = Field(ge=0)
    roi: Decimal


class BacktestReport(BaseModel):
    """包含随机 baseline 的严格向前滚动回测输出。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[BacktestPeriodResult, ...]
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
