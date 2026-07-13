"""历史来源、预测日志和预测评估的 Pydantic 模型。"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from numbers import Integral
from typing import Literal, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from dlt_number_analysis import DISCLAIMER

PredictionOrigin = Literal["generated", "manual_chat"]


def _normalize_ticket_numbers(
    value: object, *, count: int, maximum: int, area: str
) -> tuple[int, ...]:
    """把票据号码严格校验并规范化为升序元组。"""
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{area}必须恰好包含 {count} 个号码")
    if any(isinstance(number, bool) or not isinstance(number, Integral) for number in value):
        raise ValueError(f"{area}号码必须是整数")
    numbers = tuple(int(number) for number in value)
    if any(number < 1 or number > maximum for number in numbers):
        raise ValueError(f"{area}号码必须在 1 到 {maximum} 之间")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{area}号码不得重复")
    if numbers != tuple(sorted(numbers)):
        raise ValueError(f"{area}号码必须严格升序")
    return numbers


def _validate_aware_datetime(value: datetime | None, *, field_name: str) -> datetime | None:
    """确保存在的时间带有明确时区。"""
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{field_name}必须包含时区")
    return value


class DrawSourceRecord(BaseModel):
    """单期开奖记录的来源和内容指纹。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    issue: str = Field(pattern=r"^\d+$")
    source_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    fetched_at: datetime
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        """要求来源地址至少具有协议，允许 chat:// 等内部来源标识。"""
        if not urlparse(value).scheme:
            raise ValueError("source_url 必须包含 URL 协议")
        return value

    @field_validator("fetched_at")
    @classmethod
    def validate_fetched_at(cls, value: datetime) -> datetime:
        """来源获取时间必须带时区。"""
        validated = _validate_aware_datetime(value, field_name="fetched_at")
        assert validated is not None
        return validated


class TicketRecord(BaseModel):
    """一注大乐透候选或已购买票据。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    target_issue: str = Field(pattern=r"^\d+$")
    ticket_id: str = Field(min_length=1)
    front_numbers: tuple[int, ...]
    back_numbers: tuple[int, ...]
    ticket_role: str = Field(default="standard", min_length=1)

    @field_validator("front_numbers", mode="before")
    @classmethod
    def validate_front_numbers(cls, value: object) -> tuple[int, ...]:
        """校验一注的五个前区号码。"""
        return _normalize_ticket_numbers(value, count=5, maximum=35, area="前区")

    @field_validator("back_numbers", mode="before")
    @classmethod
    def validate_back_numbers(cls, value: object) -> tuple[int, ...]:
        """校验一注的两个后区号码。"""
        return _normalize_ticket_numbers(value, count=2, maximum=12, area="后区")


class PredictionRecord(BaseModel):
    """一次五注预测或手工事前票据的完整可审计记录。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    target_issue: str = Field(pattern=r"^\d+$")
    tickets: tuple[TicketRecord, ...] = Field(min_length=5, max_length=5)
    strategy_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    generated_at: datetime | None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    random_seed: int | None
    parameters: dict[str, JsonValue]
    prediction_origin: PredictionOrigin
    risk_disclaimer: str = DISCLAIMER

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime | None) -> datetime | None:
        """已知的原始生成时间必须带时区。"""
        return _validate_aware_datetime(value, field_name="generated_at")

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(cls, value: datetime) -> datetime:
        """补录时间必须带时区。"""
        validated = _validate_aware_datetime(value, field_name="recorded_at")
        assert validated is not None
        return validated

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_risk_disclaimer(cls, value: str) -> str:
        """预测记录必须包含逐字一致的固定风险声明。"""
        if value != DISCLAIMER:
            raise ValueError("预测记录缺少固定风险声明")
        return value

    @model_validator(mode="after")
    def validate_prediction_consistency(self) -> Self:
        """禁止未来数据、跨期票据和重复票据。"""
        if int(self.data_cutoff_issue) >= int(self.target_issue):
            raise ValueError("data_cutoff_issue 必须早于 target_issue")
        if self.prediction_origin == "generated":
            if self.generated_at is None:
                raise ValueError("代码生成记录必须包含 generated_at")
            if self.random_seed is None:
                raise ValueError("代码生成记录必须包含 random_seed")
        if any(ticket.target_issue != self.target_issue for ticket in self.tickets):
            raise ValueError("所有票据的 target_issue 必须与预测记录一致")
        ticket_ids = [ticket.ticket_id for ticket in self.tickets]
        if len(set(ticket_ids)) != len(ticket_ids):
            raise ValueError("ticket_id 不得重复")
        combinations = [(ticket.front_numbers, ticket.back_numbers) for ticket in self.tickets]
        if len(set(combinations)) != len(combinations):
            raise ValueError("任意两注号码不得完全相同")
        return self


class PredictionEvaluation(BaseModel):
    """单注票据在实际开奖后的评估。"""

    model_config = ConfigDict(extra="forbid")

    target_issue: str = Field(pattern=r"^\d+$")
    ticket_id: str = Field(min_length=1)
    front_hits: int = Field(ge=0, le=5)
    back_hits: int = Field(ge=0, le=2)
    prize_tier: str | None
    prize_amount: Decimal | None = Field(ge=0)
    prize_context: str = Field(min_length=1)


class PredictionReview(BaseModel):
    """五注预测在单期开奖后的汇总复盘。"""

    model_config = ConfigDict(extra="forbid")

    target_issue: str = Field(pattern=r"^\d+$")
    evaluations: tuple[PredictionEvaluation, ...] = Field(min_length=1)
    best_front_hits: int = Field(ge=0, le=5)
    best_back_hits: int = Field(ge=0, le=2)
    any_prize: bool
    front_pool_coverage: int = Field(ge=0, le=5)
    back_pool_coverage: int = Field(ge=0, le=2)
    total_cost: Decimal = Field(ge=0)
    total_prize: Decimal | None = Field(ge=0)
    roi: Decimal | None
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_risk_disclaimer(cls, value: str) -> str:
        """复盘汇总必须携带固定风险声明。"""
        if value != DISCLAIMER:
            raise ValueError("复盘汇总缺少固定风险声明")
        return value
