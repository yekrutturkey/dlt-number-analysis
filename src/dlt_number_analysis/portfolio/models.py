"""候选票评分与 Portfolio 选择结果的数据模型。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from itertools import combinations
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.models import PredictionRecord, TicketRecord
from dlt_number_analysis.scoring import TicketStructureFeatures


class CandidateTicketScore(BaseModel):
    """一注合法候选及其完整评分明细。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    front_numbers: tuple[int, int, int, int, int]
    back_numbers: tuple[int, int]
    features: TicketStructureFeatures
    front_number_score: float = Field(ge=0, le=1)
    back_number_score: float = Field(ge=0, le=1)
    number_score: float = Field(ge=0, le=1)
    structure_score: float = Field(ge=0, le=1)
    combined_ticket_score: float = Field(ge=0, le=1)
    structure_component_scores: dict[str, float]
    sum_interval: str = Field(min_length=1)
    zone_structure: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_numbers(self) -> Self:
        """候选号码必须分别唯一、升序并位于合法范围。"""
        if self.front_numbers != tuple(sorted(self.front_numbers)):
            raise ValueError("候选前区号码必须严格升序")
        if len(set(self.front_numbers)) != 5 or not all(
            1 <= number <= 35 for number in self.front_numbers
        ):
            raise ValueError("候选前区必须是 1 到 35 内的 5 个不同号码")
        if self.back_numbers != tuple(sorted(self.back_numbers)):
            raise ValueError("候选后区号码必须严格升序")
        if len(set(self.back_numbers)) != 2 or not all(
            1 <= number <= 12 for number in self.back_numbers
        ):
            raise ValueError("候选后区必须是 1 到 12 内的 2 个不同号码")
        return self


class CandidatePool(BaseModel):
    """带完整生成元数据和风险声明的候选票池。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    generated_at: datetime
    random_seed: int
    scorer_name: str = Field(min_length=1)
    scorer_parameters: dict[str, JsonValue]
    generation_parameters: dict[str, JsonValue]
    candidates: tuple[CandidateTicketScore, ...] = Field(min_length=1)
    risk_disclaimer: str = DISCLAIMER

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        """候选票池生成时间必须带时区。"""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at 必须包含时区")
        return value

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        """候选池输出必须包含固定风险声明。"""
        if value != DISCLAIMER:
            raise ValueError("候选票池缺少固定风险声明")
        return value

    @model_validator(mode="after")
    def validate_pool(self) -> Self:
        """禁止未来截止期、候选 ID 重复和候选组合重复。"""
        if int(self.data_cutoff_issue) >= int(self.target_issue):
            raise ValueError("data_cutoff_issue 必须早于 target_issue")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id 不得重复")
        combinations_seen = [
            (candidate.front_numbers, candidate.back_numbers) for candidate in self.candidates
        ]
        if len(set(combinations_seen)) != len(combinations_seen):
            raise ValueError("候选票池不得包含完全相同票据")
        return self


class PortfolioConstraints(BaseModel):
    """十元五注预算下的默认 Portfolio 约束。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticket_count: int = Field(default=5, ge=5, le=5)
    ticket_cost: float = Field(default=2.0, gt=0)
    total_budget: float = Field(default=10.0, gt=0)
    min_front_pool_size: int = Field(default=16, ge=5, le=25)
    max_front_pool_size: int = Field(default=20, ge=5, le=25)
    min_core_front_numbers: int = Field(default=2, ge=1, le=5)
    max_core_front_numbers: int = Field(default=3, ge=1, le=5)
    core_min_occurrences: int = Field(default=2, ge=2, le=3)
    max_core_numbers_with_three_occurrences: int = Field(default=1, ge=0, le=1)
    max_pairwise_front_overlap: int = Field(default=2, ge=0, le=5)
    min_sum_intervals: int = Field(default=3, ge=1, le=5)
    min_zone_structures: int = Field(default=3, ge=1, le=5)
    stable_ticket_count: int = Field(default=3, ge=0, le=5)
    exploration_ticket_count: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """检查上下界和角色数量一致。"""
        if self.min_front_pool_size > self.max_front_pool_size:
            raise ValueError("前区号码池下界不得大于上界")
        if self.min_core_front_numbers > self.max_core_front_numbers:
            raise ValueError("核心号码数量下界不得大于上界")
        if self.stable_ticket_count + self.exploration_ticket_count != self.ticket_count:
            raise ValueError("稳健票和探索票数量之和必须等于 5")
        if self.total_budget != self.ticket_count * self.ticket_cost:
            raise ValueError("总预算必须等于票数乘以单注成本")
        return self


class PortfolioTicket(BaseModel):
    """被 Portfolio 选中的候选票据及其角色。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    front_numbers: tuple[int, int, int, int, int]
    back_numbers: tuple[int, int]
    ticket_role: Literal["core_stable", "exploration"]
    number_score: float = Field(ge=0, le=1)
    structure_score: float = Field(ge=0, le=1)
    combined_ticket_score: float = Field(ge=0, le=1)
    sum_interval: str
    zone_structure: str


class PortfolioScoreBreakdown(BaseModel):
    """Portfolio 目标函数的逐项分数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    single_ticket_score: float = Field(ge=0, le=1)
    portfolio_diversity: float = Field(ge=0, le=1)
    core_concentration: float = Field(ge=0, le=1)
    structure_coverage: float = Field(ge=0, le=1)
    excessive_repeat_penalty: float = Field(ge=0, le=1)
    combined_portfolio_score: float = Field(ge=0, le=1)


class PortfolioSelection(BaseModel):
    """满足约束的五注 Portfolio 选择及可审计元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    generated_at: datetime
    random_seed: int
    model_version: str = "portfolio-optimizer-v0.3"
    tickets: tuple[PortfolioTicket, ...] = Field(min_length=5, max_length=5)
    core_front_numbers: tuple[int, ...] = Field(min_length=2, max_length=3)
    front_pool_size: int
    sum_interval_count: int
    zone_structure_count: int
    constraints: PortfolioConstraints
    scores: PortfolioScoreBreakdown
    optimizer_parameters: dict[str, JsonValue]
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        """Portfolio 输出必须包含固定风险声明。"""
        if value != DISCLAIMER:
            raise ValueError("Portfolio 输出缺少固定风险声明")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """再次验证五注选择满足保存的全部硬约束。"""
        constraints = self.constraints
        if int(self.data_cutoff_issue) >= int(self.target_issue):
            raise ValueError("data_cutoff_issue 必须早于 target_issue")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at 必须包含时区")
        if len(self.tickets) != constraints.ticket_count:
            raise ValueError("Portfolio 必须固定包含 5 注")
        fronts = [set(ticket.front_numbers) for ticket in self.tickets]
        combinations_seen = [(ticket.front_numbers, ticket.back_numbers) for ticket in self.tickets]
        if len(set(combinations_seen)) != len(combinations_seen):
            raise ValueError("任意两注不得完全相同")
        if len({ticket.back_numbers for ticket in self.tickets}) != len(self.tickets):
            raise ValueError("后区组合不得完全重复")
        if any(
            len(left.intersection(right)) > constraints.max_pairwise_front_overlap
            for left, right in combinations(fronts, 2)
        ):
            raise ValueError("任意两注前区交集超过约束")
        front_pool_size = len(set().union(*fronts))
        if (
            not constraints.min_front_pool_size
            <= front_pool_size
            <= constraints.max_front_pool_size
        ):
            raise ValueError("前区总号码池不在约束范围")
        if front_pool_size != self.front_pool_size:
            raise ValueError("front_pool_size 与票据不一致")

        counts = Counter(number for ticket in self.tickets for number in ticket.front_numbers)
        core_counts = [counts[number] for number in self.core_front_numbers]
        if not (
            constraints.min_core_front_numbers
            <= len(self.core_front_numbers)
            <= constraints.max_core_front_numbers
        ):
            raise ValueError("核心前区号码数量不满足约束")
        if any(count not in (constraints.core_min_occurrences, 3) for count in core_counts):
            raise ValueError("核心前区号码出现次数不满足约束")
        if (
            sum(count == 3 for count in core_counts)
            > constraints.max_core_numbers_with_three_occurrences
        ):
            raise ValueError("出现 3 次的核心号码过多")

        stable_count = sum(ticket.ticket_role == "core_stable" for ticket in self.tickets)
        if stable_count != constraints.stable_ticket_count:
            raise ValueError("核心/稳健票数量不满足约束")
        if len(self.tickets) - stable_count != constraints.exploration_ticket_count:
            raise ValueError("探索票数量不满足约束")
        if any(
            not set(ticket.front_numbers).intersection(self.core_front_numbers)
            for ticket in self.tickets
            if ticket.ticket_role == "core_stable"
        ):
            raise ValueError("每注核心/稳健票至少应包含一个指定核心号码")

        sum_count = len({ticket.sum_interval for ticket in self.tickets})
        zone_count = len({ticket.zone_structure for ticket in self.tickets})
        if sum_count < constraints.min_sum_intervals or sum_count != self.sum_interval_count:
            raise ValueError("和值区间覆盖不足或记录不一致")
        if zone_count < constraints.min_zone_structures or zone_count != self.zone_structure_count:
            raise ValueError("三区结构覆盖不足或记录不一致")
        return self

    def to_prediction_record(self) -> PredictionRecord:
        """转换为标准五注预测日志，保留优化器种子、参数和分数。"""
        return PredictionRecord(
            target_issue=self.target_issue,
            tickets=tuple(
                TicketRecord(
                    target_issue=self.target_issue,
                    ticket_id=f"{self.target_issue}-optimized-{index:02d}",
                    front_numbers=ticket.front_numbers,
                    back_numbers=ticket.back_numbers,
                    ticket_role=ticket.ticket_role,
                )
                for index, ticket in enumerate(self.tickets, start=1)
            ),
            strategy_name="optimized_portfolio",
            model_version=self.model_version,
            data_cutoff_issue=self.data_cutoff_issue,
            generated_at=self.generated_at,
            random_seed=self.random_seed,
            parameters={
                "candidate_ids": [ticket.candidate_id for ticket in self.tickets],
                "core_front_numbers": list(self.core_front_numbers),
                "constraints": self.constraints.model_dump(mode="json"),
                "portfolio_scores": self.scores.model_dump(mode="json"),
                "optimizer_parameters": self.optimizer_parameters,
            },
            prediction_origin="generated",
        )
