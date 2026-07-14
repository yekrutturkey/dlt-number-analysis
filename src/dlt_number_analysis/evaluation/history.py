"""按历史生效边界解析奖级规则和逐期实际奖金。"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlt_number_analysis.evaluation.prize import PrizeTable
from dlt_number_analysis.models import PredictionReview


class IssuePrizeRecord(BaseModel):
    """某一期与规则版本绑定的实际单注奖金数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    issue: str = Field(pattern=r"^\d+$")
    prize_table_version: str = Field(min_length=1)
    prize_context: str = Field(min_length=1)
    actual_amount_by_tier: dict[str, Decimal]

    @model_validator(mode="after")
    def validate_amounts(self) -> Self:
        """实际奖金不得为负数。"""
        if any(amount < 0 for amount in self.actual_amount_by_tier.values()):
            raise ValueError("实际单注奖金不得为负数")
        return self


class PrizeRuleSchedule(BaseModel):
    """按 effective_from_issue 排序的历史规则表集合。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tables: tuple[PrizeTable, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        """每个生效起点和版本必须唯一。"""
        starts = [int(table.effective_from_issue) for table in self.tables]
        versions = [table.version for table in self.tables]
        if len(set(starts)) != len(starts):
            raise ValueError("历史奖级规则生效起点不得重复")
        if len(set(versions)) != len(versions):
            raise ValueError("历史奖级规则版本不得重复")
        return self

    def table_for_issue(self, issue: str) -> PrizeTable | None:
        """返回该期之前最近生效的规则；早于全部规则时返回 None。"""
        numeric_issue = int(issue)
        eligible = [
            table for table in self.tables if int(table.effective_from_issue) <= numeric_issue
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda table: int(table.effective_from_issue))


def apply_issue_prize_record(
    review: PredictionReview,
    table: PrizeTable,
    record: IssuePrizeRecord,
) -> PredictionReview:
    """用逐期实际奖金覆盖配置中的浮动或规则金额。"""
    if review.target_issue != record.issue:
        raise ValueError("复盘期号与实际奖金记录期号不一致")
    if table.version != record.prize_table_version:
        raise ValueError("实际奖金记录绑定了错误的奖级规则版本")
    if record.prize_context not in table.contexts:
        raise ValueError("实际奖金记录使用了未知奖金上下文")
    configured_tiers = {tier.prize_tier for tier in table.tiers}
    if not set(record.actual_amount_by_tier).issubset(configured_tiers):
        raise ValueError("实际奖金记录包含规则表中不存在的奖级")

    evaluations = tuple(
        evaluation.model_copy(
            update={
                "prize_amount": (
                    Decimal("0")
                    if evaluation.prize_tier is None
                    else record.actual_amount_by_tier.get(
                        evaluation.prize_tier,
                        evaluation.prize_amount,
                    )
                ),
                "prize_context": record.prize_context,
            }
        )
        for evaluation in review.evaluations
    )
    amounts = [evaluation.prize_amount for evaluation in evaluations]
    total_prize = (
        None
        if any(amount is None for amount in amounts)
        else sum((amount for amount in amounts if amount is not None), Decimal("0"))
    )
    roi = None if total_prize is None else (total_prize - review.total_cost) / review.total_cost
    return review.model_copy(
        update={"evaluations": evaluations, "total_prize": total_prize, "roi": roi}
    )
