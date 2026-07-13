"""从配置表加载大乐透奖级，并评估预测票据。"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlt_number_analysis.data import DrawRecord
from dlt_number_analysis.models import (
    PredictionEvaluation,
    PredictionRecord,
    PredictionReview,
    TicketRecord,
)


class MatchPattern(BaseModel):
    """某一奖级对应的前后区命中数组合。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    front_hits: int = Field(ge=0, le=5)
    back_hits: int = Field(ge=0, le=2)


class PrizeTierRule(BaseModel):
    """奖级匹配条件及各奖池上下文对应奖金。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prize_tier: str = Field(min_length=1)
    matches: tuple[MatchPattern, ...] = Field(min_length=1)
    prize_amount_by_context: dict[str, Decimal | None]


class PrizeTable(BaseModel):
    """一个生效区间内的完整奖级和票价配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    game: str = Field(min_length=1)
    version: str = Field(min_length=1)
    effective_from_issue: str = Field(pattern=r"^\d+$")
    source_url: str = Field(min_length=1)
    ticket_cost: Decimal = Field(gt=0)
    contexts: tuple[str, ...] = Field(min_length=1)
    default_context: str = Field(min_length=1)
    tiers: tuple[PrizeTierRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_table_consistency(self) -> Self:
        """确保上下文和命中模式没有遗漏或冲突。"""
        context_set = set(self.contexts)
        if len(context_set) != len(self.contexts):
            raise ValueError("奖金上下文不得重复")
        if self.default_context not in context_set:
            raise ValueError("default_context 必须出现在 contexts 中")

        seen_patterns: set[tuple[int, int]] = set()
        for tier in self.tiers:
            if set(tier.prize_amount_by_context) != context_set:
                raise ValueError(f"{tier.prize_tier} 的奖金上下文不完整")
            for match in tier.matches:
                pattern = (match.front_hits, match.back_hits)
                if pattern in seen_patterns:
                    raise ValueError(f"命中模式重复配置：{pattern}")
                seen_patterns.add(pattern)
        return self


def load_prize_table(path: str | Path) -> PrizeTable:
    """从 UTF-8 JSON 文件读取并校验奖级配置。"""
    source = Path(path)
    return PrizeTable.model_validate_json(source.read_text(encoding="utf-8"))


def evaluate_ticket(
    ticket: TicketRecord,
    actual_draw: DrawRecord,
    prize_table: PrizeTable,
    *,
    prize_context: str | None = None,
) -> PredictionEvaluation:
    """根据配置表评估一注号码，不在函数中硬编码奖级或奖金。"""
    if ticket.target_issue != actual_draw.issue:
        raise ValueError("票据期号与实际开奖期号不一致")
    context = prize_context or prize_table.default_context
    if context not in prize_table.contexts:
        raise ValueError(f"未知奖金上下文：{context}")

    front_hits = len(set(ticket.front_numbers).intersection(actual_draw.front_numbers))
    back_hits = len(set(ticket.back_numbers).intersection(actual_draw.back_numbers))
    prize_tier: str | None = None
    prize_amount: Decimal | None = Decimal("0")
    for rule in prize_table.tiers:
        if any(
            match.front_hits == front_hits and match.back_hits == back_hits
            for match in rule.matches
        ):
            prize_tier = rule.prize_tier
            prize_amount = rule.prize_amount_by_context[context]
            break

    return PredictionEvaluation(
        target_issue=ticket.target_issue,
        ticket_id=ticket.ticket_id,
        front_hits=front_hits,
        back_hits=back_hits,
        prize_tier=prize_tier,
        prize_amount=prize_amount,
        prize_context=context,
    )


def evaluate_prediction(
    prediction: PredictionRecord,
    actual_draw: DrawRecord,
    prize_table: PrizeTable,
    *,
    prize_context: str | None = None,
) -> PredictionReview:
    """汇总五注票据的命中、号码池覆盖、成本、奖金和 ROI。"""
    if prediction.target_issue != actual_draw.issue:
        raise ValueError("预测期号与实际开奖期号不一致")
    evaluations = tuple(
        evaluate_ticket(ticket, actual_draw, prize_table, prize_context=prize_context)
        for ticket in prediction.tickets
    )
    front_pool = {number for ticket in prediction.tickets for number in ticket.front_numbers}
    back_pool = {number for ticket in prediction.tickets for number in ticket.back_numbers}
    total_cost = prize_table.ticket_cost * len(prediction.tickets)

    known_amounts = [evaluation.prize_amount for evaluation in evaluations]
    total_prize = (
        None
        if any(amount is None for amount in known_amounts)
        else sum((amount for amount in known_amounts if amount is not None), Decimal("0"))
    )
    roi = None if total_prize is None else (total_prize - total_cost) / total_cost
    return PredictionReview(
        target_issue=prediction.target_issue,
        evaluations=evaluations,
        best_front_hits=max(evaluation.front_hits for evaluation in evaluations),
        best_back_hits=max(evaluation.back_hits for evaluation in evaluations),
        any_prize=any(evaluation.prize_tier is not None for evaluation in evaluations),
        front_pool_coverage=len(front_pool.intersection(actual_draw.front_numbers)),
        back_pool_coverage=len(back_pool.intersection(actual_draw.back_numbers)),
        total_cost=total_cost,
        total_prize=total_prize,
        roi=roi,
    )
