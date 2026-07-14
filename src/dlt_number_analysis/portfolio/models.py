"""Candidate scoring and constrained five-ticket portfolio models."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from itertools import combinations
from math import isclose
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.models import PredictionRecord, TicketRecord
from dlt_number_analysis.scoring import (
    ScorerSpec,
    StructureComponentScore,
    TicketStructureFeatures,
)


class CandidateTicketScore(BaseModel):
    """One legal candidate ticket with complete number and structure score details."""

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
    structure_component_details: dict[str, StructureComponentScore]
    sum_interval: str = Field(min_length=1)
    zone_structure: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_numbers(self) -> Self:
        if self.front_numbers != tuple(sorted(self.front_numbers)):
            raise ValueError("candidate front numbers must be strictly ascending")
        if len(set(self.front_numbers)) != 5 or not all(
            1 <= number <= 35 for number in self.front_numbers
        ):
            raise ValueError("candidate front numbers must be five unique values from 1 to 35")
        if self.back_numbers != tuple(sorted(self.back_numbers)):
            raise ValueError("candidate back numbers must be strictly ascending")
        if len(set(self.back_numbers)) != 2 or not all(
            1 <= number <= 12 for number in self.back_numbers
        ):
            raise ValueError("candidate back numbers must be two unique values from 1 to 12")
        return self


class CandidatePool(BaseModel):
    """Seeded candidate pool whose scorer spec is the invocation source of truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    generated_at: datetime
    random_seed: int
    scorer_spec: ScorerSpec
    generation_parameters: dict[str, JsonValue]
    candidates: tuple[CandidateTicketScore, ...] = Field(min_length=1)
    risk_disclaimer: str = DISCLAIMER

    @property
    def scorer_name(self) -> str:
        """v0.3 compatibility accessor derived from ``scorer_spec``."""
        return self.scorer_spec.name

    @property
    def scorer_parameters(self) -> dict[str, JsonValue]:
        """v0.3 compatibility accessor that cannot diverge from actual invocation."""
        return dict(self.scorer_spec.parameters)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        if value != DISCLAIMER:
            raise ValueError("candidate pool is missing the fixed risk disclaimer")
        return value

    @model_validator(mode="after")
    def validate_pool(self) -> Self:
        if int(self.data_cutoff_issue) >= int(self.target_issue):
            raise ValueError("data_cutoff_issue must precede target_issue")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id values must be unique")
        tickets = [
            (candidate.front_numbers, candidate.back_numbers) for candidate in self.candidates
        ]
        if len(set(tickets)) != len(tickets):
            raise ValueError("candidate pool cannot contain duplicate tickets")
        return self


class PortfolioConstraints(BaseModel):
    """Hard constraints for a ten-yuan, five-ticket portfolio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticket_count: int = Field(default=5, ge=5, le=5)
    ticket_cost: float = Field(default=2.0, gt=0)
    total_budget: float = Field(default=10.0, gt=0)
    min_front_pool_size: int = Field(default=16, ge=5, le=25)
    max_front_pool_size: int = Field(default=21, ge=5, le=25)
    target_front_pool_size: int = Field(default=18, ge=5, le=25)
    min_core_front_numbers: int = Field(default=2, ge=1, le=5)
    max_core_front_numbers: int = Field(default=3, ge=1, le=5)
    core_min_occurrences: int = Field(default=2, ge=2, le=3)
    core_max_occurrences: int = Field(default=3, ge=2, le=3)
    max_core_numbers_with_three_occurrences: int = Field(default=1, ge=0, le=1)
    min_support_front_numbers: int = Field(default=2, ge=0, le=10)
    max_support_front_numbers: int = Field(default=4, ge=0, le=10)
    support_max_occurrences: int = Field(default=2, ge=2, le=2)
    max_front_number_occurrences: int = Field(default=3, ge=2, le=5)
    max_pairwise_front_overlap: int = Field(default=2, ge=0, le=5)
    min_sum_intervals: int = Field(default=3, ge=1, le=5)
    min_zone_structures: int = Field(default=3, ge=1, le=5)
    stable_ticket_count: int = Field(default=3, ge=0, le=5)
    exploration_ticket_count: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.min_front_pool_size > self.max_front_pool_size:
            raise ValueError("front pool lower bound must not exceed upper bound")
        if not self.min_front_pool_size <= self.target_front_pool_size <= self.max_front_pool_size:
            raise ValueError("target_front_pool_size must be inside the configured range")
        if self.min_core_front_numbers > self.max_core_front_numbers:
            raise ValueError("core count lower bound must not exceed upper bound")
        if self.core_min_occurrences > self.core_max_occurrences:
            raise ValueError("core occurrence lower bound must not exceed upper bound")
        if self.min_support_front_numbers > self.max_support_front_numbers:
            raise ValueError("support count lower bound must not exceed upper bound")
        if self.stable_ticket_count + self.exploration_ticket_count != self.ticket_count:
            raise ValueError("stable and exploration ticket counts must total five")
        if not isclose(self.total_budget, self.ticket_count * self.ticket_cost):
            raise ValueError("total budget must equal ticket count times ticket cost")
        return self


class PortfolioTicket(BaseModel):
    """Selected candidate ticket and its role in the five-ticket portfolio."""

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
    """Auditable portfolio objective components."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    single_ticket_score: float = Field(ge=0, le=1)
    portfolio_diversity: float = Field(ge=0, le=1)
    core_concentration: float = Field(ge=0, le=1)
    structure_coverage: float = Field(ge=0, le=1)
    excessive_repeat_penalty: float = Field(ge=0, le=1)
    combined_portfolio_score: float = Field(ge=0, le=1)


class PortfolioSelection(BaseModel):
    """Validated five-ticket portfolio with explicit core/support classification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    generated_at: datetime
    random_seed: int
    model_version: str = "portfolio-optimizer-v0.4"
    tickets: tuple[PortfolioTicket, ...] = Field(min_length=5, max_length=5)
    core_front_numbers: tuple[int, ...] = Field(min_length=2, max_length=3)
    support_front_numbers: tuple[int, ...] = Field(min_length=2, max_length=4)
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
        if value != DISCLAIMER:
            raise ValueError("portfolio is missing the fixed risk disclaimer")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        constraints = self.constraints
        if int(self.data_cutoff_issue) >= int(self.target_issue):
            raise ValueError("data_cutoff_issue must precede target_issue")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        if len(self.tickets) != constraints.ticket_count:
            raise ValueError("portfolio must contain exactly five tickets")
        fronts = [set(ticket.front_numbers) for ticket in self.tickets]
        tickets = [(ticket.front_numbers, ticket.back_numbers) for ticket in self.tickets]
        if len(set(tickets)) != len(tickets):
            raise ValueError("tickets must not be identical")
        if len({ticket.back_numbers for ticket in self.tickets}) != len(self.tickets):
            raise ValueError("back combinations must not repeat")
        if any(
            len(left.intersection(right)) > constraints.max_pairwise_front_overlap
            for left, right in combinations(fronts, 2)
        ):
            raise ValueError("pairwise front overlap exceeds the hard constraint")

        front_pool_size = len(set().union(*fronts))
        if (
            not constraints.min_front_pool_size
            <= front_pool_size
            <= constraints.max_front_pool_size
        ):
            raise ValueError("front pool size is outside the configured range")
        if front_pool_size != self.front_pool_size:
            raise ValueError("front_pool_size does not match tickets")

        counts = Counter(number for ticket in self.tickets for number in ticket.front_numbers)
        if any(count > constraints.max_front_number_occurrences for count in counts.values()):
            raise ValueError("a front number exceeds the maximum occurrence constraint")
        core_set = set(self.core_front_numbers)
        support_set = set(self.support_front_numbers)
        if core_set.intersection(support_set):
            raise ValueError("core and support front numbers must be disjoint")
        repeated = {number for number, count in counts.items() if count >= 2}
        if repeated != core_set.union(support_set):
            raise ValueError("every repeated number must be classified as core or support")
        if not (
            constraints.min_core_front_numbers
            <= len(core_set)
            <= constraints.max_core_front_numbers
        ):
            raise ValueError("core front number count does not satisfy constraints")
        if any(
            not constraints.core_min_occurrences
            <= counts[number]
            <= constraints.core_max_occurrences
            for number in core_set
        ):
            raise ValueError("core front number occurrences do not satisfy constraints")
        if (
            sum(counts[number] == 3 for number in core_set)
            > constraints.max_core_numbers_with_three_occurrences
        ):
            raise ValueError("too many core numbers occur three times")
        if not (
            constraints.min_support_front_numbers
            <= len(support_set)
            <= constraints.max_support_front_numbers
        ):
            raise ValueError("support front number count does not satisfy constraints")
        if any(
            counts[number] < 2 or counts[number] > constraints.support_max_occurrences
            for number in support_set
        ):
            raise ValueError("support front number occurrences do not satisfy constraints")
        if any(count != 1 for number, count in counts.items() if number not in repeated):
            raise ValueError("exploration front numbers must occur exactly once")

        stable = [ticket for ticket in self.tickets if ticket.ticket_role == "core_stable"]
        if len(stable) != constraints.stable_ticket_count:
            raise ValueError("stable ticket count does not satisfy constraints")
        if len(self.tickets) - len(stable) != constraints.exploration_ticket_count:
            raise ValueError("exploration ticket count does not satisfy constraints")
        if any(not set(ticket.front_numbers).intersection(core_set) for ticket in stable):
            raise ValueError("every stable ticket must contain a core front number")

        sum_count = len({ticket.sum_interval for ticket in self.tickets})
        zone_count = len({ticket.zone_structure for ticket in self.tickets})
        if sum_count < constraints.min_sum_intervals or sum_count != self.sum_interval_count:
            raise ValueError("sum interval coverage is insufficient or inconsistent")
        if zone_count < constraints.min_zone_structures or zone_count != self.zone_structure_count:
            raise ValueError("zone structure coverage is insufficient or inconsistent")
        return self

    def to_prediction_record(self) -> PredictionRecord:
        """Convert the portfolio into the standard five-ticket prediction record."""
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
                "support_front_numbers": list(self.support_front_numbers),
                "constraints": self.constraints.model_dump(mode="json"),
                "portfolio_scores": self.scores.model_dump(mode="json"),
                "optimizer_parameters": self.optimizer_parameters,
            },
            prediction_origin="generated",
        )
