"""NumPy scoring views and index-only feasible Portfolio evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

from dlt_number_analysis.data.data_validator import validate_draw_dataframe
from dlt_number_analysis.portfolio.bank import (
    FeasiblePortfolioBank,
    candidate_numbers_hash,
    portfolio_constraints_signature,
)
from dlt_number_analysis.portfolio.models import (
    CandidatePool,
    PortfolioSelection,
    PortfolioTicket,
)
from dlt_number_analysis.portfolio.optimizer import score_portfolio_tickets
from dlt_number_analysis.scoring import ScorerSpec, build_number_scorer

TicketIdentity = tuple[tuple[int, ...], tuple[int, ...]]
PortfolioScoringMethod = Literal["object_reference", "numpy_vectorized"]


def _readonly(values: object, *, dtype: np.dtype[object] | type[object]) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    view = array.view()
    view.flags.writeable = False
    return view


def _validate_score_vector(values: np.ndarray, count: int, label: str) -> None:
    if values.shape != (count,):
        raise ValueError(f"{label} must have shape ({count},)")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{label} must contain finite values in [0, 1]")


@dataclass(frozen=True, slots=True)
class CandidateArrayBundle:
    """Immutable array representation preserving one CandidatePool's exact order."""

    source_pool: CandidatePool
    front_numbers: np.ndarray
    back_numbers: np.ndarray
    number_scores: np.ndarray
    structure_scores: np.ndarray
    combined_scores: np.ndarray
    sum_interval_codes: np.ndarray
    zone_structure_codes: np.ndarray
    candidate_ids: tuple[str, ...]
    ticket_identity_to_index: Mapping[TicketIdentity, int]
    candidate_numbers_hash: str
    sum_interval_labels: tuple[str, ...]
    zone_structure_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.source_pool.candidates)
        if self.front_numbers.shape != (count, 5):
            raise ValueError("front_numbers must have shape [candidate_count, 5]")
        if self.back_numbers.shape != (count, 2):
            raise ValueError("back_numbers must have shape [candidate_count, 2]")
        for values, label in (
            (self.number_scores, "number_scores"),
            (self.structure_scores, "structure_scores"),
            (self.combined_scores, "combined_scores"),
        ):
            _validate_score_vector(values, count, label)
        if self.sum_interval_codes.shape != (count,) or self.zone_structure_codes.shape != (count,):
            raise ValueError("categorical candidate codes must align with the candidate count")
        if len(self.candidate_ids) != count or len(set(self.candidate_ids)) != count:
            raise ValueError("candidate IDs must be complete and unique")
        if len(self.ticket_identity_to_index) != count:
            raise ValueError("ticket_identity_to_index must contain every candidate exactly once")
        if self.candidate_numbers_hash != candidate_numbers_hash(self.source_pool):
            raise ValueError("candidate_numbers_hash does not match the source pool")
        if any(array.flags.writeable for array in self.arrays):
            raise ValueError("candidate arrays must be read-only")

    @property
    def arrays(self) -> tuple[np.ndarray, ...]:
        """Return every immutable array for compact validation and memory accounting."""
        return (
            self.front_numbers,
            self.back_numbers,
            self.number_scores,
            self.structure_scores,
            self.combined_scores,
            self.sum_interval_codes,
            self.zone_structure_codes,
        )


def _categorical_codes(values: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    labels = tuple(dict.fromkeys(values))
    lookup = {label: index for index, label in enumerate(labels)}
    return _readonly([lookup[value] for value in values], dtype=np.int32), labels


def candidate_pool_to_array_bundle(pool: CandidatePool) -> CandidateArrayBundle:
    """Convert one CandidatePool once without changing order, scores, IDs, or identity."""
    identities = tuple(
        (candidate.front_numbers, candidate.back_numbers) for candidate in pool.candidates
    )
    if len(set(identities)) != len(identities):
        raise ValueError("candidate pool identities are not unique")
    sum_codes, sum_labels = _categorical_codes(
        tuple(candidate.sum_interval for candidate in pool.candidates)
    )
    zone_codes, zone_labels = _categorical_codes(
        tuple(candidate.zone_structure for candidate in pool.candidates)
    )
    return CandidateArrayBundle(
        source_pool=pool,
        front_numbers=_readonly(
            [candidate.front_numbers for candidate in pool.candidates], dtype=np.int16
        ),
        back_numbers=_readonly(
            [candidate.back_numbers for candidate in pool.candidates], dtype=np.int8
        ),
        number_scores=_readonly(
            [candidate.number_score for candidate in pool.candidates], dtype=np.float64
        ),
        structure_scores=_readonly(
            [candidate.structure_score for candidate in pool.candidates], dtype=np.float64
        ),
        combined_scores=_readonly(
            [candidate.combined_ticket_score for candidate in pool.candidates],
            dtype=np.float64,
        ),
        sum_interval_codes=sum_codes,
        zone_structure_codes=zone_codes,
        candidate_ids=tuple(candidate.candidate_id for candidate in pool.candidates),
        ticket_identity_to_index=MappingProxyType(
            {identity: index for index, identity in enumerate(identities)}
        ),
        candidate_numbers_hash=candidate_numbers_hash(pool),
        sum_interval_labels=sum_labels,
        zone_structure_labels=zone_labels,
    )


@dataclass(frozen=True, slots=True)
class CandidateScoreView:
    """Lightweight score arrays over one immutable candidate bundle."""

    candidate_arrays: CandidateArrayBundle
    number_scores: np.ndarray
    structure_scores: np.ndarray
    combined_scores: np.ndarray
    scorer_spec: ScorerSpec
    number_score_weight: float
    structure_score_weight: float
    scoring_method: str

    def __post_init__(self) -> None:
        count = len(self.candidate_arrays.candidate_ids)
        for values, label in (
            (self.number_scores, "number_scores"),
            (self.structure_scores, "structure_scores"),
            (self.combined_scores, "combined_scores"),
        ):
            _validate_score_vector(values, count, label)
            if values.flags.writeable:
                raise ValueError(f"{label} must be read-only")
        if self.number_score_weight < 0 or self.structure_score_weight < 0:
            raise ValueError("score weights must be non-negative")
        if self.number_score_weight + self.structure_score_weight <= 0:
            raise ValueError("score weights must sum above zero")


def candidate_score_view_from_arrays(
    candidate_arrays: CandidateArrayBundle,
    *,
    number_scores: np.ndarray,
    combined_scores: np.ndarray,
    scorer_spec: ScorerSpec,
    number_score_weight: float,
    structure_score_weight: float,
    scoring_method: str,
) -> CandidateScoreView:
    """Bind existing score vectors without materializing CandidateTicketScore objects."""
    return CandidateScoreView(
        candidate_arrays=candidate_arrays,
        number_scores=_readonly(number_scores, dtype=np.float64),
        structure_scores=_readonly(candidate_arrays.structure_scores, dtype=np.float64),
        combined_scores=_readonly(combined_scores, dtype=np.float64),
        scorer_spec=scorer_spec,
        number_score_weight=number_score_weight,
        structure_score_weight=structure_score_weight,
        scoring_method=scoring_method,
    )


def _normalized_score_lookup(scores: Mapping[int, float], maximum: int) -> np.ndarray:
    expected = set(range(1, maximum + 1))
    if set(scores) != expected:
        raise ValueError(f"number scores must cover every value from 1 to {maximum}")
    values = np.asarray([float(scores[number]) for number in range(1, maximum + 1)])
    if np.any(~np.isfinite(values)):
        raise ValueError("number scores must be finite")
    minimum = float(values.min())
    span = float(values.max()) - minimum
    if span == 0:
        return np.full(maximum, 0.5, dtype=np.float64)
    return (values - minimum) / span


def build_candidate_score_view(
    history: pd.DataFrame,
    candidate_arrays: CandidateArrayBundle,
    *,
    scorer_spec: ScorerSpec,
    number_score_weight: float,
    structure_score_weight: float,
    front_number_weight: float = 5 / 7,
    back_number_weight: float = 2 / 7,
) -> CandidateScoreView:
    """Compute only score arrays for one scorer/weight configuration."""
    validated = validate_draw_dataframe(history)
    if str(validated.iloc[-1]["issue"]) != candidate_arrays.source_pool.data_cutoff_issue:
        raise ValueError("score-view history cutoff does not match the candidate bundle")
    if front_number_weight < 0 or back_number_weight < 0:
        raise ValueError("front/back score weights must be non-negative")
    area_total = front_number_weight + back_number_weight
    score_total = number_score_weight + structure_score_weight
    if area_total <= 0 or score_total <= 0:
        raise ValueError("score weights must sum above zero")
    scorer = build_number_scorer(scorer_spec)
    front_lookup = _normalized_score_lookup(scorer(validated, area="front"), 35)
    back_lookup = _normalized_score_lookup(scorer(validated, area="back"), 12)
    front_scores = front_lookup[candidate_arrays.front_numbers - 1].mean(axis=1)
    back_scores = back_lookup[candidate_arrays.back_numbers - 1].mean(axis=1)
    number_scores = (
        front_number_weight * front_scores + back_number_weight * back_scores
    ) / area_total
    combined_scores = (
        number_score_weight * number_scores
        + structure_score_weight * candidate_arrays.structure_scores
    ) / score_total
    return candidate_score_view_from_arrays(
        candidate_arrays,
        number_scores=number_scores,
        combined_scores=combined_scores,
        scorer_spec=scorer_spec,
        number_score_weight=number_score_weight,
        structure_score_weight=structure_score_weight,
        scoring_method="numpy_candidate_score_view",
    )


@dataclass(frozen=True, slots=True)
class FeasiblePortfolioIndexBank:
    """Index-only representation with score-independent objective components."""

    source_bank: FeasiblePortfolioBank
    candidate_indices: np.ndarray
    stable_ticket_mask: np.ndarray
    core_number_masks: np.ndarray
    support_number_masks: np.ndarray
    front_pool_sizes: np.ndarray
    sum_interval_counts: np.ndarray
    zone_structure_counts: np.ndarray
    portfolio_diversity_scores: np.ndarray
    core_concentration_scores: np.ndarray
    structure_coverage_scores: np.ndarray
    repeat_penalties: np.ndarray
    entry_hashes: tuple[str, ...]
    bank_hash: str
    candidate_numbers_hash: str
    constraints_signature: str

    def __post_init__(self) -> None:
        count = self.source_bank.bank_size
        expected_shapes = {
            "candidate_indices": (count, 5),
            "stable_ticket_mask": (count, 5),
            "core_number_masks": (count, 36),
            "support_number_masks": (count, 36),
            "front_pool_sizes": (count,),
            "sum_interval_counts": (count,),
            "zone_structure_counts": (count,),
            "portfolio_diversity_scores": (count,),
            "core_concentration_scores": (count,),
            "structure_coverage_scores": (count,),
            "repeat_penalties": (count,),
        }
        for name, shape in expected_shapes.items():
            values = getattr(self, name)
            if values.shape != shape:
                raise ValueError(f"{name} has shape {values.shape}, expected {shape}")
            if values.flags.writeable:
                raise ValueError(f"{name} must be read-only")
        if len(self.entry_hashes) != count or len(set(self.entry_hashes)) != count:
            raise ValueError("entry hashes must be complete and unique")
        if self.bank_hash != self.source_bank.bank_hash:
            raise ValueError("bank_hash does not match the source bank")
        if self.candidate_numbers_hash != self.source_bank.candidate_numbers_hash:
            raise ValueError("candidate_numbers_hash does not match the source bank")
        if self.constraints_signature != portfolio_constraints_signature(
            self.source_bank.constraints
        ):
            raise ValueError("constraints_signature does not match source constraints")


def _unique_counts(codes: np.ndarray) -> np.ndarray:
    ordered = np.sort(codes, axis=1)
    return 1 + np.count_nonzero(np.diff(ordered, axis=1), axis=1)


def feasible_portfolio_bank_to_index_bank(
    candidate_arrays: CandidateArrayBundle,
    bank: FeasiblePortfolioBank,
) -> FeasiblePortfolioIndexBank:
    """Resolve every bank ticket to five candidate indices and precompute fixed scores."""
    if candidate_arrays.candidate_numbers_hash != bank.candidate_numbers_hash:
        raise ValueError("candidate bundle numbers do not match the feasible bank")
    if bank.constraints_signature != portfolio_constraints_signature(bank.constraints):
        raise ValueError("bank constraints signature is invalid")
    entries = tuple(sorted(bank.entries, key=lambda entry: entry.entry_hash))
    candidate_indices = np.empty((len(entries), 5), dtype=np.int32)
    stable_mask = np.zeros((len(entries), 5), dtype=bool)
    core_masks = np.zeros((len(entries), 36), dtype=bool)
    support_masks = np.zeros((len(entries), 36), dtype=bool)
    for entry_index, entry in enumerate(entries):
        for ticket_index, ticket in enumerate(entry.tickets):
            identity = (ticket.front_numbers, ticket.back_numbers)
            try:
                candidate_index = candidate_arrays.ticket_identity_to_index[identity]
            except KeyError as error:
                raise ValueError("bank ticket is missing from the candidate bundle") from error
            if (
                tuple(candidate_arrays.front_numbers[candidate_index]) != ticket.front_numbers
                or tuple(candidate_arrays.back_numbers[candidate_index]) != ticket.back_numbers
            ):
                raise ValueError("candidate index does not reproduce the bank ticket")
            candidate_indices[entry_index, ticket_index] = candidate_index
        stable_mask[entry_index, list(entry.stable_ticket_indices)] = True
        core_masks[entry_index, list(entry.core_front_numbers)] = True
        support_masks[entry_index, list(entry.support_front_numbers)] = True

    selected_front = candidate_arrays.front_numbers[candidate_indices]
    selected_back = candidate_arrays.back_numbers[candidate_indices]
    front_ticket_masks = np.zeros((len(entries), 5, 36), dtype=bool)
    back_ticket_masks = np.zeros((len(entries), 5, 13), dtype=bool)
    entry_axis = np.arange(len(entries))[:, None, None]
    ticket_axis = np.arange(5)[None, :, None]
    front_ticket_masks[entry_axis, ticket_axis, selected_front] = True
    back_ticket_masks[entry_axis, ticket_axis, selected_back] = True
    front_counts = front_ticket_masks.sum(axis=1)
    front_pool_sizes = np.count_nonzero(front_counts, axis=1)
    back_pool_sizes = np.count_nonzero(back_ticket_masks.any(axis=1), axis=1)

    pairwise_overlaps = np.stack(
        [
            np.count_nonzero(front_ticket_masks[:, left] & front_ticket_masks[:, right], axis=1)
            for left, right in combinations(range(5), 2)
        ],
        axis=1,
    )
    constraints = bank.constraints
    maximum_distance = max(
        constraints.target_front_pool_size - constraints.min_front_pool_size,
        constraints.max_front_pool_size - constraints.target_front_pool_size,
        1,
    )
    target_pool_scores = np.maximum(
        0.0,
        1.0 - np.abs(front_pool_sizes - constraints.target_front_pool_size) / maximum_distance,
    )
    pairwise_diversity = 1.0 - pairwise_overlaps.sum(axis=1) / 50
    back_diversity = back_pool_sizes / 12
    portfolio_diversity = (target_pool_scores + pairwise_diversity + back_diversity) / 3

    stable_front_masks = front_ticket_masks & stable_mask[:, :, None]
    stable_core_occurrences = np.count_nonzero(
        stable_front_masks & core_masks[:, None, :], axis=(1, 2)
    )
    total_core_occurrences = (front_counts * core_masks).sum(axis=1)
    core_counts = core_masks.sum(axis=1)
    core_number_coverage = (stable_front_masks.any(axis=1) & core_masks).sum(axis=1) / core_counts
    core_occurrence_compliance = (
        core_masks
        & (front_counts >= constraints.core_min_occurrences)
        & (front_counts <= constraints.core_max_occurrences)
    ).sum(axis=1) / core_counts
    stable_ticket_coverage = (
        (front_ticket_masks & core_masks[:, None, :]).any(axis=2) & stable_mask
    ).sum(axis=1) / stable_mask.sum(axis=1)
    stable_occurrence_share = stable_core_occurrences / total_core_occurrences
    core_concentration = (
        core_number_coverage
        + core_occurrence_compliance
        + stable_ticket_coverage
        + stable_occurrence_share
    ) / 4

    sum_counts = _unique_counts(candidate_arrays.sum_interval_codes[candidate_indices])
    zone_counts = _unique_counts(candidate_arrays.zone_structure_codes[candidate_indices])
    structure_coverage = (
        np.minimum(sum_counts / constraints.min_sum_intervals, 1.0)
        + np.minimum(zone_counts / constraints.min_zone_structures, 1.0)
    ) / 2

    classified_masks = core_masks | support_masks
    unclassified_repeat_excess = (
        np.where(
            (front_counts > 1) & ~classified_masks,
            front_counts - 1,
            0,
        ).sum(axis=1)
        / 25
    )
    overlap_excess = np.maximum(pairwise_overlaps - 1, 0).sum(axis=1) / max(
        pairwise_overlaps.shape[1] * constraints.max_pairwise_front_overlap,
        1,
    )
    repeat_penalties = np.minimum(
        (unclassified_repeat_excess + overlap_excess) / 2,
        1.0,
    )
    return FeasiblePortfolioIndexBank(
        source_bank=bank,
        candidate_indices=_readonly(candidate_indices, dtype=np.int32),
        stable_ticket_mask=_readonly(stable_mask, dtype=bool),
        core_number_masks=_readonly(core_masks, dtype=bool),
        support_number_masks=_readonly(support_masks, dtype=bool),
        front_pool_sizes=_readonly(front_pool_sizes, dtype=np.int16),
        sum_interval_counts=_readonly(sum_counts, dtype=np.int8),
        zone_structure_counts=_readonly(zone_counts, dtype=np.int8),
        portfolio_diversity_scores=_readonly(portfolio_diversity, dtype=np.float64),
        core_concentration_scores=_readonly(core_concentration, dtype=np.float64),
        structure_coverage_scores=_readonly(structure_coverage, dtype=np.float64),
        repeat_penalties=_readonly(repeat_penalties, dtype=np.float64),
        entry_hashes=tuple(entry.entry_hash for entry in entries),
        bank_hash=bank.bank_hash,
        candidate_numbers_hash=bank.candidate_numbers_hash,
        constraints_signature=bank.constraints_signature,
    )


@dataclass(frozen=True, slots=True)
class PortfolioScoreArrays:
    """All bank-member objective components without Pydantic object construction."""

    single_ticket_scores: np.ndarray
    diversity_scores: np.ndarray
    core_concentration_scores: np.ndarray
    structure_coverage_scores: np.ndarray
    repeat_penalties: np.ndarray
    combined_portfolio_scores: np.ndarray


@dataclass(frozen=True, slots=True)
class VectorizedPortfolioBankScoringResult:
    """Best selection plus array scores and stage timings."""

    selection: PortfolioSelection
    selected_entry_hash: str
    score_arrays: PortfolioScoreArrays
    bank_hash: str
    evaluated_portfolios: int
    vectorized_scoring_seconds: float
    final_object_construction_seconds: float


def _portfolio_tickets(
    score_view: CandidateScoreView,
    index_bank: FeasiblePortfolioIndexBank,
    entry_index: int,
) -> tuple[PortfolioTicket, ...]:
    indices = index_bank.candidate_indices[entry_index]
    stable = index_bank.stable_ticket_mask[entry_index]
    return tuple(
        PortfolioTicket(
            candidate_id=score_view.candidate_arrays.candidate_ids[int(candidate_index)],
            front_numbers=tuple(
                int(value)
                for value in score_view.candidate_arrays.front_numbers[int(candidate_index)]
            ),
            back_numbers=tuple(
                int(value)
                for value in score_view.candidate_arrays.back_numbers[int(candidate_index)]
            ),
            ticket_role="core_stable" if bool(stable[ticket_index]) else "exploration",
            number_score=float(score_view.number_scores[int(candidate_index)]),
            structure_score=float(score_view.structure_scores[int(candidate_index)]),
            combined_ticket_score=float(score_view.combined_scores[int(candidate_index)]),
            sum_interval=score_view.candidate_arrays.source_pool.candidates[
                int(candidate_index)
            ].sum_interval,
            zone_structure=score_view.candidate_arrays.source_pool.candidates[
                int(candidate_index)
            ].zone_structure,
        )
        for ticket_index, candidate_index in enumerate(indices)
    )


def score_portfolio_bank_vectorized(
    score_view: CandidateScoreView,
    index_bank: FeasiblePortfolioIndexBank,
    *,
    random_seed: int,
    single_ticket_weight: float = 0.45,
    diversity_weight: float = 0.25,
    core_weight: float = 0.15,
    structure_weight: float = 0.15,
    repeat_penalty_weight: float = 0.20,
) -> VectorizedPortfolioBankScoringResult:
    """Score all bank members by array and materialize only the deterministic winner."""
    if score_view.candidate_arrays.candidate_numbers_hash != index_bank.candidate_numbers_hash:
        raise ValueError("score view and index bank use different candidate numbers")
    weights = (
        single_ticket_weight,
        diversity_weight,
        core_weight,
        structure_weight,
        repeat_penalty_weight,
    )
    if any(not isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("portfolio objective weights must be finite and non-negative")
    positive_weight = sum(weights[:4])
    if positive_weight <= 0:
        raise ValueError("positive portfolio objective weights must sum above zero")

    scoring_started = perf_counter()
    ticket_scores = score_view.combined_scores[index_bank.candidate_indices]
    single_scores = (
        ticket_scores[:, 0]
        + ticket_scores[:, 1]
        + ticket_scores[:, 2]
        + ticket_scores[:, 3]
        + ticket_scores[:, 4]
    ) / 5
    positive_scores = (
        single_ticket_weight * single_scores
        + diversity_weight * index_bank.portfolio_diversity_scores
        + core_weight * index_bank.core_concentration_scores
        + structure_weight * index_bank.structure_coverage_scores
    ) / positive_weight
    combined_scores = np.clip(
        positive_scores - repeat_penalty_weight * index_bank.repeat_penalties,
        0.0,
        1.0,
    )
    maximum_score = float(combined_scores.max())
    tied = np.flatnonzero(combined_scores == maximum_score)
    best_index = max(tied.tolist(), key=lambda index: index_bank.entry_hashes[index])
    scoring_seconds = perf_counter() - scoring_started

    object_started = perf_counter()
    core_numbers = tuple(
        int(value) for value in np.flatnonzero(index_bank.core_number_masks[best_index])
    )
    support_numbers = tuple(
        int(value) for value in np.flatnonzero(index_bank.support_number_masks[best_index])
    )
    tickets = _portfolio_tickets(score_view, index_bank, best_index)
    constraints = index_bank.source_bank.constraints
    final_scores = score_portfolio_tickets(
        tickets,
        core_numbers,
        support_numbers,
        constraints,
        single_ticket_weight=single_ticket_weight,
        diversity_weight=diversity_weight,
        core_weight=core_weight,
        structure_weight=structure_weight,
        repeat_penalty_weight=repeat_penalty_weight,
    )
    selection = PortfolioSelection(
        target_issue=score_view.candidate_arrays.source_pool.target_issue,
        data_cutoff_issue=score_view.candidate_arrays.source_pool.data_cutoff_issue,
        generated_at=score_view.candidate_arrays.source_pool.generated_at,
        random_seed=random_seed,
        model_version="portfolio-bank-v0.5.2",
        tickets=tickets,
        core_front_numbers=core_numbers,
        support_front_numbers=support_numbers,
        front_pool_size=int(index_bank.front_pool_sizes[best_index]),
        sum_interval_count=int(index_bank.sum_interval_counts[best_index]),
        zone_structure_count=int(index_bank.zone_structure_counts[best_index]),
        constraints=constraints,
        scores=final_scores,
        optimizer_parameters={
            "selection_method": "highest_objective_from_shared_feasible_portfolio_bank",
            "portfolio_scoring_method": "numpy_vectorized",
            "bank_seed": index_bank.source_bank.bank_seed,
            "bank_size": index_bank.source_bank.bank_size,
            "bank_acceptance_rate": index_bank.source_bank.acceptance_rate,
            "bank_hash": index_bank.bank_hash,
            "candidate_numbers_hash": index_bank.candidate_numbers_hash,
            "constraints_signature": index_bank.constraints_signature,
            "selected_bank_entry_hash": index_bank.entry_hashes[best_index],
            "evaluated_portfolios": index_bank.source_bank.bank_size,
            "single_ticket_weight": single_ticket_weight,
            "diversity_weight": diversity_weight,
            "core_weight": core_weight,
            "structure_weight": structure_weight,
            "repeat_penalty_weight": repeat_penalty_weight,
            "candidate_score_view_method": score_view.scoring_method,
        },
    )
    object_seconds = perf_counter() - object_started
    return VectorizedPortfolioBankScoringResult(
        selection=selection,
        selected_entry_hash=index_bank.entry_hashes[best_index],
        score_arrays=PortfolioScoreArrays(
            single_ticket_scores=_readonly(single_scores, dtype=np.float64),
            diversity_scores=index_bank.portfolio_diversity_scores,
            core_concentration_scores=index_bank.core_concentration_scores,
            structure_coverage_scores=index_bank.structure_coverage_scores,
            repeat_penalties=index_bank.repeat_penalties,
            combined_portfolio_scores=_readonly(combined_scores, dtype=np.float64),
        ),
        bank_hash=index_bank.bank_hash,
        evaluated_portfolios=index_bank.source_bank.bank_size,
        vectorized_scoring_seconds=scoring_seconds,
        final_object_construction_seconds=object_seconds,
    )
