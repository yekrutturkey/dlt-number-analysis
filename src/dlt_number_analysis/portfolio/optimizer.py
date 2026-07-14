"""Seeded search for a valid and auditable five-ticket portfolio."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import isfinite
from random import Random

from dlt_number_analysis.portfolio.models import (
    CandidatePool,
    CandidateTicketScore,
    PortfolioConstraints,
    PortfolioScoreBreakdown,
    PortfolioSelection,
    PortfolioTicket,
)


class PortfolioOptimizationError(ValueError):
    """No feasible portfolio was found within the configured seeded search budget."""


def _passes_base_constraints(
    candidates: tuple[CandidateTicketScore, ...],
    constraints: PortfolioConstraints,
) -> bool:
    if len({candidate.candidate_id for candidate in candidates}) != constraints.ticket_count:
        return False
    if len({candidate.back_numbers for candidate in candidates}) != constraints.ticket_count:
        return False
    fronts = [set(candidate.front_numbers) for candidate in candidates]
    if any(
        len(left.intersection(right)) > constraints.max_pairwise_front_overlap
        for left, right in combinations(fronts, 2)
    ):
        return False
    pool_size = len(set().union(*fronts))
    if not constraints.min_front_pool_size <= pool_size <= constraints.max_front_pool_size:
        return False
    if len({candidate.sum_interval for candidate in candidates}) < constraints.min_sum_intervals:
        return False
    return (
        len({candidate.zone_structure for candidate in candidates})
        >= constraints.min_zone_structures
    )


def _classify_repeated_numbers(
    candidates: tuple[CandidateTicketScore, ...],
    constraints: PortfolioConstraints,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Classify every repeated number as core or support; leave singletons exploratory."""
    counts = Counter(number for candidate in candidates for number in candidate.front_numbers)
    if any(count > constraints.max_front_number_occurrences for count in counts.values()):
        return None
    repeated = sorted(number for number, count in counts.items() if count >= 2)
    triples = {number for number in repeated if counts[number] == 3}
    if len(triples) > constraints.max_core_numbers_with_three_occurrences:
        return None

    number_quality = {
        number: sum(
            candidate.combined_ticket_score
            for candidate in candidates
            if number in candidate.front_numbers
        )
        / counts[number]
        for number in repeated
    }
    best: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    best_score = -1.0
    for core_count in range(
        constraints.min_core_front_numbers,
        constraints.max_core_front_numbers + 1,
    ):
        if core_count > len(repeated):
            continue
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
            support = set(repeated).difference(core)
            if not (
                constraints.min_support_front_numbers
                <= len(support)
                <= constraints.max_support_front_numbers
            ):
                continue
            if any(counts[number] > constraints.support_max_occurrences for number in support):
                continue
            stable_coverage = sum(
                bool(set(candidate.front_numbers).intersection(core)) for candidate in candidates
            )
            if stable_coverage < constraints.stable_ticket_count:
                continue
            score = stable_coverage + sum(number_quality[number] for number in core)
            if score > best_score:
                best = tuple(sorted(core)), tuple(sorted(support))
                best_score = score
    return best


def _assign_roles(
    candidates: tuple[CandidateTicketScore, ...],
    core_numbers: tuple[int, ...],
    constraints: PortfolioConstraints,
) -> tuple[PortfolioTicket, ...] | None:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -len(set(candidate.front_numbers).intersection(core_numbers)),
            -candidate.combined_ticket_score,
            candidate.candidate_id,
        ),
    )
    stable = ranked[: constraints.stable_ticket_count]
    if any(not set(candidate.front_numbers).intersection(core_numbers) for candidate in stable):
        return None
    stable_ids = {candidate.candidate_id for candidate in stable}
    ordered = [
        *stable,
        *(candidate for candidate in ranked if candidate.candidate_id not in stable_ids),
    ]
    return tuple(
        PortfolioTicket(
            candidate_id=candidate.candidate_id,
            front_numbers=candidate.front_numbers,
            back_numbers=candidate.back_numbers,
            ticket_role=("core_stable" if candidate.candidate_id in stable_ids else "exploration"),
            number_score=candidate.number_score,
            structure_score=candidate.structure_score,
            combined_ticket_score=candidate.combined_ticket_score,
            sum_interval=candidate.sum_interval,
            zone_structure=candidate.zone_structure,
        )
        for candidate in ordered
    )


def _score_portfolio(
    tickets: tuple[PortfolioTicket, ...],
    core_numbers: tuple[int, ...],
    support_numbers: tuple[int, ...],
    constraints: PortfolioConstraints,
    *,
    single_ticket_weight: float,
    diversity_weight: float,
    core_weight: float,
    structure_weight: float,
    repeat_penalty_weight: float,
) -> PortfolioScoreBreakdown:
    fronts = [set(ticket.front_numbers) for ticket in tickets]
    front_counts = Counter(number for ticket in tickets for number in ticket.front_numbers)
    front_pool_size = len(set().union(*fronts))
    overlaps = [len(left.intersection(right)) for left, right in combinations(fronts, 2)]
    back_pool_size = len({number for ticket in tickets for number in ticket.back_numbers})

    single_ticket_score = sum(ticket.combined_ticket_score for ticket in tickets) / len(tickets)
    pool_distance = abs(front_pool_size - constraints.target_front_pool_size)
    maximum_distance = max(
        constraints.target_front_pool_size - constraints.min_front_pool_size,
        constraints.max_front_pool_size - constraints.target_front_pool_size,
        1,
    )
    target_pool_score = max(0.0, 1.0 - pool_distance / maximum_distance)
    pairwise_diversity = 1.0 - sum(overlaps) / (len(overlaps) * 5)
    back_diversity = back_pool_size / 12
    portfolio_diversity = (target_pool_score + pairwise_diversity + back_diversity) / 3

    core_set = set(core_numbers)
    stable = [ticket for ticket in tickets if ticket.ticket_role == "core_stable"]
    stable_core_occurrences = sum(
        len(set(ticket.front_numbers).intersection(core_set)) for ticket in stable
    )
    total_core_occurrences = sum(front_counts[number] for number in core_set)
    core_number_coverage = sum(
        any(number in ticket.front_numbers for ticket in stable) for number in core_set
    ) / len(core_set)
    core_occurrence_compliance = sum(
        constraints.core_min_occurrences <= front_counts[number] <= constraints.core_max_occurrences
        for number in core_set
    ) / len(core_set)
    stable_ticket_coverage = sum(
        bool(set(ticket.front_numbers).intersection(core_set)) for ticket in stable
    ) / len(stable)
    stable_occurrence_share = stable_core_occurrences / total_core_occurrences
    core_concentration = (
        core_number_coverage
        + core_occurrence_compliance
        + stable_ticket_coverage
        + stable_occurrence_share
    ) / 4

    sum_coverage = min(
        len({ticket.sum_interval for ticket in tickets}) / constraints.min_sum_intervals,
        1.0,
    )
    zone_coverage = min(
        len({ticket.zone_structure for ticket in tickets}) / constraints.min_zone_structures,
        1.0,
    )
    structure_coverage = (sum_coverage + zone_coverage) / 2

    classified = core_set.union(support_numbers)
    unclassified_repeat_excess = (
        sum(
            count - 1
            for number, count in front_counts.items()
            if count > 1 and number not in classified
        )
        / 25
    )
    overlap_excess = sum(max(overlap - 1, 0) for overlap in overlaps) / max(
        len(overlaps) * constraints.max_pairwise_front_overlap,
        1,
    )
    excessive_repeat_penalty = min(
        (unclassified_repeat_excess + overlap_excess) / 2,
        1.0,
    )

    positive_weight = single_ticket_weight + diversity_weight + core_weight + structure_weight
    positive_score = (
        single_ticket_weight * single_ticket_score
        + diversity_weight * portfolio_diversity
        + core_weight * core_concentration
        + structure_weight * structure_coverage
    ) / positive_weight
    combined = max(0.0, min(1.0, positive_score - repeat_penalty_weight * excessive_repeat_penalty))
    return PortfolioScoreBreakdown(
        single_ticket_score=single_ticket_score,
        portfolio_diversity=portfolio_diversity,
        core_concentration=core_concentration,
        structure_coverage=structure_coverage,
        excessive_repeat_penalty=excessive_repeat_penalty,
        combined_portfolio_score=combined,
    )


def optimize_portfolio(
    pool: CandidatePool,
    *,
    random_seed: int,
    constraints: PortfolioConstraints | None = None,
    search_trials: int = 25_000,
    stable_candidate_limit: int = 2_000,
    single_ticket_weight: float = 0.45,
    diversity_weight: float = 0.25,
    core_weight: float = 0.15,
    structure_weight: float = 0.15,
    repeat_penalty_weight: float = 0.20,
) -> PortfolioSelection:
    """Select five tickets under all v0.4 hard constraints."""
    active_constraints = constraints or PortfolioConstraints()
    if len(pool.candidates) < 10_000:
        raise ValueError("portfolio optimization requires at least 10,000 candidates")
    if search_trials < 1:
        raise ValueError("search_trials must be positive")
    if not 5 <= stable_candidate_limit <= len(pool.candidates):
        raise ValueError("stable_candidate_limit must be between 5 and candidate pool size")
    weights = (
        single_ticket_weight,
        diversity_weight,
        core_weight,
        structure_weight,
        repeat_penalty_weight,
    )
    if any(not isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("portfolio objective weights must be finite and non-negative")
    if sum(weights[:4]) <= 0:
        raise ValueError("positive portfolio objective weights must sum above zero")

    ranked = sorted(
        pool.candidates,
        key=lambda candidate: (-candidate.combined_ticket_score, candidate.candidate_id),
    )
    stable_candidates = ranked[:stable_candidate_limit]
    rng = Random(random_seed)
    best_tickets: tuple[PortfolioTicket, ...] | None = None
    best_core: tuple[int, ...] | None = None
    best_support: tuple[int, ...] | None = None
    best_scores: PortfolioScoreBreakdown | None = None
    feasible_count = 0

    for _ in range(search_trials):
        selected = list(rng.sample(stable_candidates, active_constraints.stable_ticket_count))
        selected_ids = {candidate.candidate_id for candidate in selected}
        while len(selected) < active_constraints.ticket_count:
            candidate = rng.choice(ranked)
            if candidate.candidate_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
        selected_tuple = tuple(selected)
        if not _passes_base_constraints(selected_tuple, active_constraints):
            continue
        classification = _classify_repeated_numbers(selected_tuple, active_constraints)
        if classification is None:
            continue
        core_numbers, support_numbers = classification
        tickets = _assign_roles(selected_tuple, core_numbers, active_constraints)
        if tickets is None:
            continue
        scores = _score_portfolio(
            tickets,
            core_numbers,
            support_numbers,
            active_constraints,
            single_ticket_weight=single_ticket_weight,
            diversity_weight=diversity_weight,
            core_weight=core_weight,
            structure_weight=structure_weight,
            repeat_penalty_weight=repeat_penalty_weight,
        )
        feasible_count += 1
        if best_scores is None or (
            scores.combined_portfolio_score,
            tuple(ticket.candidate_id for ticket in tickets),
        ) > (
            best_scores.combined_portfolio_score,
            tuple(ticket.candidate_id for ticket in best_tickets or ()),
        ):
            best_tickets = tickets
            best_core = core_numbers
            best_support = support_numbers
            best_scores = scores

    if best_tickets is None or best_core is None or best_support is None or best_scores is None:
        raise PortfolioOptimizationError(
            "no portfolio satisfied all hard constraints within the seeded search budget"
        )

    return PortfolioSelection(
        target_issue=pool.target_issue,
        data_cutoff_issue=pool.data_cutoff_issue,
        generated_at=pool.generated_at,
        random_seed=random_seed,
        tickets=best_tickets,
        core_front_numbers=best_core,
        support_front_numbers=best_support,
        front_pool_size=len({number for ticket in best_tickets for number in ticket.front_numbers}),
        sum_interval_count=len({ticket.sum_interval for ticket in best_tickets}),
        zone_structure_count=len({ticket.zone_structure for ticket in best_tickets}),
        constraints=active_constraints,
        scores=best_scores,
        optimizer_parameters={
            "search_method": "seeded_random_feasible_portfolio_search",
            "search_trials": search_trials,
            "stable_candidate_limit": stable_candidate_limit,
            "feasible_portfolios_evaluated": feasible_count,
            "candidate_pool_random_seed": pool.random_seed,
            "candidate_pool_size": len(pool.candidates),
            "single_ticket_weight": single_ticket_weight,
            "diversity_weight": diversity_weight,
            "core_weight": core_weight,
            "structure_weight": structure_weight,
            "repeat_penalty_weight": repeat_penalty_weight,
            "target_front_pool_size": active_constraints.target_front_pool_size,
            "classification": "all_repeats_are_explicitly_core_or_support",
        },
    )
