"""Rolling-backtest period, random distribution, and historical summary models."""

from __future__ import annotations

from decimal import Decimal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from dlt_number_analysis import DISCLAIMER


class ConfidenceInterval(BaseModel):
    """Named interval whose method distinguishes MC error from historical uncertainty."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str = Field(min_length=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    lower: float
    upper: float


class RandomMetricSummary(BaseModel):
    """Monte Carlo sampling summary for one completely random five-ticket metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    standard_error: float = Field(ge=0)
    monte_carlo_error_interval: ConfidenceInterval


class BacktestPeriodResult(BaseModel):
    """One strategy evaluated after generating from strictly pre-target history."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    target_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    strategy_name: str = Field(min_length=1)
    random_seed: int
    best_front_hits: int = Field(ge=0, le=5)
    best_back_hits: int = Field(ge=0, le=2)
    best_total_hits: int = Field(ge=0, le=7)
    at_least_three_front: bool
    at_least_2_plus_1: bool
    ticket_hit_share: float = Field(
        ge=0,
        le=1,
        validation_alias=AliasChoices("ticket_hit_share", "hit_concentration_ratio"),
    )
    unique_hit_concentration: float = Field(ge=0, le=1)
    any_prize: bool | None
    front_pool_coverage: int = Field(ge=0, le=5)
    back_pool_coverage: int = Field(ge=0, le=2)
    total_cost: Decimal = Field(gt=0)
    total_prize: Decimal | None = Field(ge=0)
    roi: Decimal | None
    prize_rule_version: str | None
    prize_data_available: bool
    random_baseline_seed_count: int = Field(ge=1)
    random_metric_percentiles: dict[str, float]

    @property
    def hit_concentration_ratio(self) -> float:
        """Deprecated v0.3 read accessor; migrate to ``ticket_hit_share``."""
        return self.ticket_hit_share


class RawObservationResult(BaseModel):
    """Direct prediction evaluation without Monte Carlo or bootstrap fields."""

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
    ticket_hit_share: float = Field(ge=0, le=1)
    unique_hit_concentration: float = Field(ge=0, le=1)
    front_pool_coverage: int = Field(ge=0, le=5)
    back_pool_coverage: int = Field(ge=0, le=2)
    total_cost: Decimal = Field(gt=0)
    total_prize: Decimal | None = Field(ge=0)
    roi: Decimal | None
    any_prize: bool | None
    prize_rule_version: str | None
    prize_data_available: bool


class RandomBaselineSummary(BaseModel):
    """Cached random five-ticket distributions summarized for one target issue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    cache_key: str = Field(min_length=1)
    cache_reused: bool
    seed_start: int
    seed_count: int = Field(ge=1)
    metric_summaries: dict[str, RandomMetricSummary]


class BacktestStrategySummary(BaseModel):
    """Cross-period performance and historical-period bootstrap intervals."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    strategy_name: str = Field(min_length=1)
    period_count: int = Field(ge=1)
    at_least_three_front_rate: float = Field(ge=0, le=1)
    at_least_2_plus_1_rate: float = Field(ge=0, le=1)
    ticket_hit_share: float = Field(
        ge=0,
        le=1,
        validation_alias=AliasChoices("ticket_hit_share", "hit_concentration_ratio"),
    )
    unique_hit_concentration: float = Field(ge=0, le=1)
    average_prize: Decimal | None = Field(ge=0)
    median_prize: Decimal | None = Field(ge=0)
    longest_no_prize_streak: int | None = Field(ge=0)
    monetary_metrics_complete: bool
    bootstrap_performance_intervals: dict[str, ConfidenceInterval]

    @property
    def hit_concentration_ratio(self) -> float:
        """Deprecated v0.3 read accessor; migrate to ``ticket_hit_share``."""
        return self.ticket_hit_share


class BacktestReport(BaseModel):
    """Strict expanding-window report with cached completely random baseline."""

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
        if value != DISCLAIMER:
            raise ValueError("backtest report is missing the fixed risk disclaimer")
        return value
