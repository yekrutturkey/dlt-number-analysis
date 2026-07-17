"""Score-independent feasible Portfolio banks for fair strategy comparison."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import MutableMapping
from itertools import combinations
from math import isclose
from random import Random
from time import perf_counter
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.portfolio.models import (
    CandidatePool,
    CandidateTicketScore,
    PortfolioConstraints,
    PortfolioScoreBreakdown,
    PortfolioSelection,
    PortfolioTicket,
)
from dlt_number_analysis.portfolio.optimizer import (
    PortfolioOptimizationError,
    score_portfolio_tickets,
)


class BankTicket(BaseModel):
    """Candidate identity defined only by lottery numbers, never by candidate_id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    front_numbers: tuple[int, int, int, int, int]
    back_numbers: tuple[int, int]

    @model_validator(mode="after")
    def validate_numbers(self) -> Self:
        if (
            self.front_numbers != tuple(sorted(self.front_numbers))
            or len(set(self.front_numbers)) != 5
        ):
            raise ValueError("bank ticket front numbers must be strictly ascending and unique")
        if (
            self.back_numbers != tuple(sorted(self.back_numbers))
            or len(set(self.back_numbers)) != 2
        ):
            raise ValueError("bank ticket back numbers must be strictly ascending and unique")
        if not all(1 <= value <= 35 for value in self.front_numbers):
            raise ValueError("bank ticket front number is outside 1..35")
        if not all(1 <= value <= 12 for value in self.back_numbers):
            raise ValueError("bank ticket back number is outside 1..12")
        return self


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _ticket(candidate: CandidateTicketScore) -> BankTicket:
    return BankTicket(
        front_numbers=candidate.front_numbers,
        back_numbers=candidate.back_numbers,
    )


def portfolio_constraints_signature(constraints: PortfolioConstraints) -> str:
    """Return a stable signature over every hard-constraint parameter."""
    return _sha256(constraints.model_dump(mode="json"))


def candidate_numbers_hash(pool: CandidatePool) -> str:
    """Hash the candidate number set independently of order, IDs, or scores."""
    tickets = sorted(
        (
            candidate.front_numbers,
            candidate.back_numbers,
        )
        for candidate in pool.candidates
    )
    return _sha256(tickets)


class FeasiblePortfolioEntry(BaseModel):
    """One canonical feasible five-ticket set with score-independent roles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tickets: tuple[BankTicket, BankTicket, BankTicket, BankTicket, BankTicket]
    core_front_numbers: tuple[int, ...] = Field(min_length=1)
    support_front_numbers: tuple[int, ...]
    stable_ticket_indices: tuple[int, int, int]
    entry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        identities = [(ticket.front_numbers, ticket.back_numbers) for ticket in self.tickets]
        if identities != sorted(identities) or len(set(identities)) != 5:
            raise ValueError("bank entry tickets must be unique and canonically sorted")
        if tuple(sorted(set(self.stable_ticket_indices))) != self.stable_ticket_indices:
            raise ValueError("stable ticket indices must be three sorted unique values")
        if any(index < 0 or index >= 5 for index in self.stable_ticket_indices):
            raise ValueError("stable ticket index is outside the five-ticket entry")
        expected_hash = _entry_hash(
            self.tickets,
            self.core_front_numbers,
            self.support_front_numbers,
            self.stable_ticket_indices,
        )
        if self.entry_hash != expected_hash:
            raise ValueError("entry_hash does not match the canonical feasible portfolio")
        return self


def _entry_hash(
    tickets: tuple[BankTicket, ...],
    core: tuple[int, ...],
    support: tuple[int, ...],
    stable_indices: tuple[int, ...],
) -> str:
    return _sha256(
        {
            "tickets": [ticket.model_dump(mode="json") for ticket in tickets],
            "core_front_numbers": core,
            "support_front_numbers": support,
            "stable_ticket_indices": stable_indices,
        }
    )


class FeasiblePortfolioBank(BaseModel):
    """Reusable score-independent set generated once per target/seed/constraints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    candidate_pool_seed: int
    bank_seed: int
    constraints: PortfolioConstraints
    constraints_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_numbers_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_search_trials: int = Field(ge=1)
    search_trials: int = Field(ge=1)
    search_expansion_count: int = Field(ge=0)
    minimum_bank_size: int = Field(ge=1)
    attempted_portfolios: int = Field(ge=1)
    feasible_attempt_count: int = Field(ge=1)
    bank_size: int = Field(ge=1)
    acceptance_rate: float = Field(gt=0, le=1)
    generation_seconds: float = Field(ge=0)
    entries: tuple[FeasiblePortfolioEntry, ...] = Field(min_length=1)
    bank_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        if value != DISCLAIMER:
            raise ValueError("feasible Portfolio bank is missing the fixed disclaimer")
        return value

    @model_validator(mode="after")
    def validate_bank(self) -> Self:
        if self.constraints_signature != portfolio_constraints_signature(self.constraints):
            raise ValueError("constraints_signature does not match constraints")
        if self.bank_size != len(self.entries):
            raise ValueError("bank_size does not match entries")
        if self.bank_size < self.minimum_bank_size:
            raise ValueError("bank_size is below the required minimum")
        if self.search_trials < self.initial_search_trials:
            raise ValueError("search_trials must not be below initial_search_trials")
        if self.attempted_portfolios > self.search_trials:
            raise ValueError("attempted_portfolios exceeds the final search budget")
        if len({entry.entry_hash for entry in self.entries}) != len(self.entries):
            raise ValueError("feasible Portfolio bank contains duplicate entries")
        if not isclose(
            self.acceptance_rate,
            self.feasible_attempt_count / self.attempted_portfolios,
            abs_tol=1e-12,
        ):
            raise ValueError("acceptance_rate does not match attempted portfolios")
        if self.bank_hash != _bank_hash(self):
            raise ValueError("bank_hash does not match bank contents")
        return self


def _bank_hash(bank: FeasiblePortfolioBank | dict[str, object]) -> str:
    if isinstance(bank, FeasiblePortfolioBank):
        payload = bank.model_dump(
            mode="json",
            exclude={"bank_hash", "generation_seconds", "risk_disclaimer"},
        )
    else:
        payload = dict(bank)
    return _sha256(payload)


class FeasiblePortfolioSubset(BaseModel):
    """Stable entry-hash prefix of a parent bank for bounded reference scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_bank_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    subset_size: int = Field(ge=1)
    subset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    first_entry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    last_entry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bank: FeasiblePortfolioBank

    @model_validator(mode="after")
    def validate_subset(self) -> Self:
        ordered = tuple(sorted(self.bank.entries, key=lambda entry: entry.entry_hash))
        entry_hashes = tuple(entry.entry_hash for entry in ordered)
        if self.subset_size != self.bank.bank_size or self.subset_size != len(ordered):
            raise ValueError("subset_size does not match the bounded bank")
        if entry_hashes[0] != self.first_entry_hash or entry_hashes[-1] != self.last_entry_hash:
            raise ValueError("subset boundary hashes do not match the bounded bank")
        expected = _sha256(
            {
                "parent_bank_hash": self.parent_bank_hash,
                "entry_hashes": entry_hashes,
            }
        )
        if self.subset_hash != expected:
            raise ValueError("subset_hash does not match the stable entry prefix")
        return self


def build_stable_portfolio_subset(
    bank: FeasiblePortfolioBank,
    subset_size: int,
) -> FeasiblePortfolioSubset:
    """Return the first N entries after deterministic entry-hash sorting."""
    if not 1 <= subset_size <= bank.bank_size:
        raise ValueError("subset_size must be inside the parent bank")
    entries = tuple(sorted(bank.entries, key=lambda entry: entry.entry_hash))[:subset_size]
    payload = bank.model_dump(mode="json", exclude={"bank_hash"})
    payload.update(
        {
            "entries": [entry.model_dump(mode="json") for entry in entries],
            "bank_size": subset_size,
            "minimum_bank_size": min(bank.minimum_bank_size, subset_size),
        }
    )
    hash_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"generation_seconds", "risk_disclaimer"}
    }
    payload["bank_hash"] = _sha256(hash_payload)
    subset_bank = FeasiblePortfolioBank.model_validate(payload)
    entry_hashes = tuple(entry.entry_hash for entry in entries)
    return FeasiblePortfolioSubset(
        parent_bank_hash=bank.bank_hash,
        subset_size=subset_size,
        subset_hash=_sha256(
            {
                "parent_bank_hash": bank.bank_hash,
                "entry_hashes": entry_hashes,
            }
        ),
        first_entry_hash=entry_hashes[0],
        last_entry_hash=entry_hashes[-1],
        bank=subset_bank,
    )


BankCacheKey = tuple[str, int, str]


def _base_constraints_pass(
    candidates: tuple[CandidateTicketScore, ...],
    constraints: PortfolioConstraints,
) -> bool:
    if len({(item.front_numbers, item.back_numbers) for item in candidates}) != 5:
        return False
    if len({item.back_numbers for item in candidates}) != 5:
        return False
    fronts = [set(item.front_numbers) for item in candidates]
    if any(
        len(left.intersection(right)) > constraints.max_pairwise_front_overlap
        for left, right in combinations(fronts, 2)
    ):
        return False
    front_pool_size = len(set().union(*fronts))
    if not constraints.min_front_pool_size <= front_pool_size <= constraints.max_front_pool_size:
        return False
    if len({item.sum_interval for item in candidates}) < constraints.min_sum_intervals:
        return False
    return len({item.zone_structure for item in candidates}) >= constraints.min_zone_structures


def _score_independent_classification(
    candidates: tuple[CandidateTicketScore, ...],
    constraints: PortfolioConstraints,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, int, int]] | None:
    counts = Counter(number for item in candidates for number in item.front_numbers)
    if any(count > constraints.max_front_number_occurrences for count in counts.values()):
        return None
    repeated = tuple(sorted(number for number, count in counts.items() if count >= 2))
    triples = {number for number in repeated if counts[number] == 3}
    if len(triples) > constraints.max_core_numbers_with_three_occurrences:
        return None

    best: tuple[tuple[int, ...], tuple[int, ...], tuple[int, int, int]] | None = None
    best_key: tuple[int, int, tuple[int, ...]] | None = None
    for core_count in range(
        constraints.min_core_front_numbers,
        constraints.max_core_front_numbers + 1,
    ):
        for core_values in combinations(repeated, core_count):
            core = set(core_values)
            if not triples.issubset(core):
                continue
            if any(
                not constraints.core_min_occurrences
                <= counts[number]
                <= constraints.core_max_occurrences
                for number in core
            ):
                continue
            support = tuple(sorted(set(repeated).difference(core)))
            if not (
                constraints.min_support_front_numbers
                <= len(support)
                <= constraints.max_support_front_numbers
            ):
                continue
            if any(counts[number] > constraints.support_max_occurrences for number in support):
                continue
            ranked_indices = sorted(
                range(5),
                key=lambda index: (
                    -len(set(candidates[index].front_numbers).intersection(core)),
                    candidates[index].front_numbers,
                    candidates[index].back_numbers,
                ),
            )
            stable = tuple(sorted(ranked_indices[: constraints.stable_ticket_count]))
            if len(stable) != 3 or any(
                not set(candidates[index].front_numbers).intersection(core) for index in stable
            ):
                continue
            stable_coverage = sum(
                bool(set(item.front_numbers).intersection(core)) for item in candidates
            )
            total_occurrences = sum(counts[number] for number in core)
            key = (stable_coverage, total_occurrences, tuple(-number for number in core_values))
            if best_key is None or key > best_key:
                best_key = key
                best = tuple(core_values), support, stable  # type: ignore[assignment]
    return best


def build_feasible_portfolio_bank(
    pool: CandidatePool,
    *,
    bank_seed: int,
    constraints: PortfolioConstraints | None = None,
    search_trials: int = 25_000,
    maximum_bank_size: int = 2_000,
    minimum_bank_size: int = 1,
    maximum_search_trials: int | None = None,
    search_trial_growth_factor: int = 2,
    cache: MutableMapping[BankCacheKey, FeasiblePortfolioBank] | None = None,
) -> FeasiblePortfolioBank:
    """Generate a reusable bank, expanding the search until its minimum size is met."""
    if search_trials < 1 or maximum_bank_size < 1 or minimum_bank_size < 1:
        raise ValueError("search trials and bank sizes must be positive")
    if minimum_bank_size > maximum_bank_size:
        raise ValueError("minimum_bank_size must not exceed maximum_bank_size")
    final_search_ceiling = maximum_search_trials or search_trials
    if final_search_ceiling < search_trials:
        raise ValueError("maximum_search_trials must not be below search_trials")
    if search_trial_growth_factor < 2:
        raise ValueError("search_trial_growth_factor must be at least 2")
    active_constraints = constraints or PortfolioConstraints()
    signature = portfolio_constraints_signature(active_constraints)
    cache_key = (pool.target_issue, bank_seed, signature)
    number_hash = candidate_numbers_hash(pool)
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        if cached.candidate_numbers_hash != number_hash:
            raise ValueError("cached bank candidate numbers differ for the same bank identity")
        if cached.bank_size < minimum_bank_size:
            raise ValueError("cached bank is below the requested minimum_bank_size")
        return cached

    started = perf_counter()
    canonical_candidates = tuple(
        sorted(
            pool.candidates,
            key=lambda item: (item.front_numbers, item.back_numbers),
        )
    )
    rng = Random(bank_seed)
    entries: dict[str, FeasiblePortfolioEntry] = {}
    feasible_attempt_count = 0
    attempted = 0
    initial_search_trials = search_trials
    active_search_trials = search_trials
    expansion_count = 0
    while True:
        for trial_index in range(attempted + 1, active_search_trials + 1):
            attempted = trial_index
            selected = tuple(
                sorted(
                    rng.sample(canonical_candidates, active_constraints.ticket_count),
                    key=lambda item: (item.front_numbers, item.back_numbers),
                )
            )
            if not _base_constraints_pass(selected, active_constraints):
                continue
            classification = _score_independent_classification(selected, active_constraints)
            if classification is None:
                continue
            feasible_attempt_count += 1
            core, support, stable_indices = classification
            tickets = tuple(_ticket(item) for item in selected)
            entry_hash = _entry_hash(tickets, core, support, stable_indices)
            entries.setdefault(
                entry_hash,
                FeasiblePortfolioEntry(
                    tickets=tickets,
                    core_front_numbers=core,
                    support_front_numbers=support,
                    stable_ticket_indices=stable_indices,
                    entry_hash=entry_hash,
                ),
            )
            if len(entries) >= maximum_bank_size:
                break
        if len(entries) >= minimum_bank_size:
            break
        if active_search_trials >= final_search_ceiling:
            raise PortfolioOptimizationError(
                "feasible Portfolio bank did not reach minimum_bank_size within the "
                "maximum seeded search budget"
            )
        active_search_trials = min(
            final_search_ceiling,
            active_search_trials * search_trial_growth_factor,
        )
        expansion_count += 1
    if not entries:
        raise PortfolioOptimizationError(
            "no feasible Portfolio bank entries were found within the seeded search budget"
        )
    ordered_entries = tuple(entries[key] for key in sorted(entries))
    payload: dict[str, object] = {
        "target_issue": pool.target_issue,
        "data_cutoff_issue": pool.data_cutoff_issue,
        "candidate_pool_seed": pool.random_seed,
        "bank_seed": bank_seed,
        "constraints": active_constraints.model_dump(mode="json"),
        "constraints_signature": signature,
        "candidate_numbers_hash": number_hash,
        "initial_search_trials": initial_search_trials,
        "search_trials": active_search_trials,
        "search_expansion_count": expansion_count,
        "minimum_bank_size": minimum_bank_size,
        "attempted_portfolios": attempted,
        "feasible_attempt_count": feasible_attempt_count,
        "bank_size": len(ordered_entries),
        "acceptance_rate": feasible_attempt_count / attempted,
        "entries": [entry.model_dump(mode="json") for entry in ordered_entries],
    }
    bank = FeasiblePortfolioBank(
        **payload,
        generation_seconds=perf_counter() - started,
        bank_hash=_bank_hash(payload),
    )
    if cache is not None:
        cache[cache_key] = bank
    return bank


def _selection_from_entry(
    pool: CandidatePool,
    bank: FeasiblePortfolioBank,
    entry: FeasiblePortfolioEntry,
    *,
    random_seed: int,
    selection_method: str,
    single_ticket_weight: float,
    diversity_weight: float,
    core_weight: float,
    structure_weight: float,
    repeat_penalty_weight: float,
) -> PortfolioSelection:
    if candidate_numbers_hash(pool) != bank.candidate_numbers_hash:
        raise ValueError("candidate pool numbers do not match the feasible Portfolio bank")
    candidates = {(item.front_numbers, item.back_numbers): item for item in pool.candidates}
    stable = set(entry.stable_ticket_indices)
    tickets = tuple(
        PortfolioTicket(
            candidate_id=candidates[(ticket.front_numbers, ticket.back_numbers)].candidate_id,
            front_numbers=ticket.front_numbers,
            back_numbers=ticket.back_numbers,
            ticket_role="core_stable" if index in stable else "exploration",
            number_score=candidates[(ticket.front_numbers, ticket.back_numbers)].number_score,
            structure_score=candidates[(ticket.front_numbers, ticket.back_numbers)].structure_score,
            combined_ticket_score=candidates[
                (ticket.front_numbers, ticket.back_numbers)
            ].combined_ticket_score,
            sum_interval=candidates[(ticket.front_numbers, ticket.back_numbers)].sum_interval,
            zone_structure=candidates[(ticket.front_numbers, ticket.back_numbers)].zone_structure,
        )
        for index, ticket in enumerate(entry.tickets)
    )
    scores = score_portfolio_tickets(
        tickets,
        entry.core_front_numbers,
        entry.support_front_numbers,
        bank.constraints,
        single_ticket_weight=single_ticket_weight,
        diversity_weight=diversity_weight,
        core_weight=core_weight,
        structure_weight=structure_weight,
        repeat_penalty_weight=repeat_penalty_weight,
    )
    return PortfolioSelection(
        target_issue=pool.target_issue,
        data_cutoff_issue=pool.data_cutoff_issue,
        generated_at=pool.generated_at,
        random_seed=random_seed,
        model_version="portfolio-bank-v0.5.1",
        tickets=tickets,
        core_front_numbers=entry.core_front_numbers,
        support_front_numbers=entry.support_front_numbers,
        front_pool_size=len({number for ticket in tickets for number in ticket.front_numbers}),
        sum_interval_count=len({ticket.sum_interval for ticket in tickets}),
        zone_structure_count=len({ticket.zone_structure for ticket in tickets}),
        constraints=bank.constraints,
        scores=scores,
        optimizer_parameters={
            "selection_method": selection_method,
            "portfolio_scoring_method": (
                "object_reference"
                if selection_method == "highest_objective_from_shared_feasible_portfolio_bank"
                else "random_bank_sample"
            ),
            "bank_seed": bank.bank_seed,
            "bank_size": bank.bank_size,
            "bank_acceptance_rate": bank.acceptance_rate,
            "bank_hash": bank.bank_hash,
            "candidate_numbers_hash": bank.candidate_numbers_hash,
            "constraints_signature": bank.constraints_signature,
            "selected_bank_entry_hash": entry.entry_hash,
            "single_ticket_weight": single_ticket_weight,
            "diversity_weight": diversity_weight,
            "core_weight": core_weight,
            "structure_weight": structure_weight,
            "repeat_penalty_weight": repeat_penalty_weight,
        },
    )


def sample_constraint_matched_portfolio(
    pool: CandidatePool,
    bank: FeasiblePortfolioBank,
    *,
    random_seed: int,
) -> PortfolioSelection:
    """Uniformly sample one bank member; scores are recorded only after selection."""
    ordered = tuple(sorted(bank.entries, key=lambda entry: entry.entry_hash))
    selected = ordered[Random(random_seed).randrange(len(ordered))]
    return _selection_from_entry(
        pool,
        bank,
        selected,
        random_seed=random_seed,
        selection_method="uniform_random_over_feasible_portfolio_bank",
        single_ticket_weight=0.45,
        diversity_weight=0.25,
        core_weight=0.15,
        structure_weight=0.15,
        repeat_penalty_weight=0.20,
    )


class PortfolioBankScoringResult(BaseModel):
    """Best bank member plus the objective score retained for every member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selection: PortfolioSelection
    scores_by_entry_hash: dict[str, PortfolioScoreBreakdown]
    bank_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluated_portfolios: int = Field(ge=1)
    risk_disclaimer: str = DISCLAIMER


def score_portfolio_bank(
    pool: CandidatePool,
    bank: FeasiblePortfolioBank,
    *,
    random_seed: int,
    single_ticket_weight: float = 0.45,
    diversity_weight: float = 0.25,
    core_weight: float = 0.15,
    structure_weight: float = 0.15,
    repeat_penalty_weight: float = 0.20,
) -> PortfolioBankScoringResult:
    """Score every shared feasible member and select the highest objective value."""
    selections: dict[str, PortfolioSelection] = {}
    for entry in sorted(bank.entries, key=lambda item: item.entry_hash):
        selections[entry.entry_hash] = _selection_from_entry(
            pool,
            bank,
            entry,
            random_seed=random_seed,
            selection_method="highest_objective_from_shared_feasible_portfolio_bank",
            single_ticket_weight=single_ticket_weight,
            diversity_weight=diversity_weight,
            core_weight=core_weight,
            structure_weight=structure_weight,
            repeat_penalty_weight=repeat_penalty_weight,
        )
    best_hash, best = max(
        selections.items(),
        key=lambda item: (item[1].scores.combined_portfolio_score, item[0]),
    )
    best = best.model_copy(
        update={
            "optimizer_parameters": {
                **best.optimizer_parameters,
                "selected_bank_entry_hash": best_hash,
                "evaluated_portfolios": len(selections),
                "portfolio_scoring_method": "object_reference",
            }
        }
    )
    return PortfolioBankScoringResult(
        selection=PortfolioSelection.model_validate(best.model_dump()),
        scores_by_entry_hash={key: value.scores for key, value in selections.items()},
        bank_hash=bank.bank_hash,
        evaluated_portfolios=len(selections),
    )
