"""从已评分候选票池选择满足硬约束的五注 Portfolio。"""

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
    """候选池和搜索预算内没有找到满足硬约束的五注组合。"""


def _passes_non_core_constraints(
    candidates: tuple[CandidateTicketScore, ...],
    constraints: PortfolioConstraints,
) -> bool:
    """快速检查号码池、相交、后区和结构覆盖约束。"""
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


def _select_core_numbers(
    candidates: tuple[CandidateTicketScore, ...],
    constraints: PortfolioConstraints,
) -> tuple[int, ...] | None:
    """选择 2–3 个指定核心；其他偶然重号不自动视为核心。"""
    counts = Counter(number for candidate in candidates for number in candidate.front_numbers)
    eligible = [
        number for number, count in counts.items() if count in (constraints.core_min_occurrences, 3)
    ]
    if len(eligible) < constraints.min_core_front_numbers:
        return None

    number_quality = {
        number: sum(
            candidate.combined_ticket_score
            for candidate in candidates
            if number in candidate.front_numbers
        )
        / counts[number]
        for number in eligible
    }
    eligible.sort(key=lambda number: (-number_quality[number], number))
    eligible = eligible[:12]

    best: tuple[int, ...] | None = None
    best_quality = -1.0
    for core_count in range(
        min(constraints.max_core_front_numbers, len(eligible)),
        constraints.min_core_front_numbers - 1,
        -1,
    ):
        for core in combinations(eligible, core_count):
            if (
                sum(counts[number] == 3 for number in core)
                > constraints.max_core_numbers_with_three_occurrences
            ):
                continue
            covered_tickets = sum(
                bool(set(candidate.front_numbers).intersection(core)) for candidate in candidates
            )
            if covered_tickets < constraints.stable_ticket_count:
                continue
            quality = covered_tickets + sum(number_quality[number] for number in core)
            if quality > best_quality:
                best = tuple(sorted(core))
                best_quality = quality
        if best is not None:
            return best
    return None


def _assign_roles(
    candidates: tuple[CandidateTicketScore, ...],
    core_numbers: tuple[int, ...],
    constraints: PortfolioConstraints,
) -> tuple[PortfolioTicket, ...] | None:
    """优先把核心覆盖高且单票分高的三注标记为核心/稳健票。"""
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
    constraints: PortfolioConstraints,
    *,
    single_ticket_weight: float,
    diversity_weight: float,
    core_weight: float,
    structure_weight: float,
    repeat_penalty_weight: float,
) -> PortfolioScoreBreakdown:
    """计算单票、组合多样性、核心集中度、结构覆盖和重复惩罚。"""
    fronts = [set(ticket.front_numbers) for ticket in tickets]
    front_counts = Counter(number for ticket in tickets for number in ticket.front_numbers)
    front_pool_size = len(set().union(*fronts))
    overlaps = [len(left.intersection(right)) for left, right in combinations(fronts, 2)]
    back_pool_size = len({number for ticket in tickets for number in ticket.back_numbers})

    single_ticket_score = sum(ticket.combined_ticket_score for ticket in tickets) / len(tickets)
    front_diversity = min(front_pool_size / constraints.max_front_pool_size, 1.0)
    pairwise_diversity = 1.0 - sum(overlaps) / (len(overlaps) * 5)
    back_diversity = back_pool_size / 12
    portfolio_diversity = (front_diversity + pairwise_diversity + back_diversity) / 3

    triple_count = sum(front_counts[number] == 3 for number in core_numbers)
    core_concentration = 1.0 - 0.2 * triple_count
    sum_coverage = min(
        len({ticket.sum_interval for ticket in tickets}) / constraints.min_sum_intervals,
        1.0,
    )
    zone_coverage = min(
        len({ticket.zone_structure for ticket in tickets}) / constraints.min_zone_structures,
        1.0,
    )
    structure_coverage = (sum_coverage + zone_coverage) / 2

    non_core_repeat_excess = (
        sum(
            max(count - 1, 0)
            for number, count in front_counts.items()
            if number not in core_numbers
        )
        / 25
    )
    overlap_excess = sum(max(overlap - 1, 0) for overlap in overlaps) / max(
        len(overlaps) * constraints.max_pairwise_front_overlap,
        1,
    )
    excessive_repeat_penalty = min((non_core_repeat_excess + overlap_excess) / 2, 1.0)

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
    """使用带种子的随机搜索从至少一万注候选中优化五注 Portfolio。"""
    active_constraints = constraints or PortfolioConstraints()
    if len(pool.candidates) < 10_000:
        raise ValueError("Portfolio 优化要求至少 10000 注候选")
    if search_trials < 1:
        raise ValueError("search_trials 必须大于 0")
    if not 5 <= stable_candidate_limit <= len(pool.candidates):
        raise ValueError("stable_candidate_limit 必须位于 5 和候选数之间")
    weights = (
        single_ticket_weight,
        diversity_weight,
        core_weight,
        structure_weight,
        repeat_penalty_weight,
    )
    if any(not isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Portfolio 目标函数权重必须是非负有限数值")
    if sum(weights[:4]) <= 0:
        raise ValueError("正向目标函数权重之和必须大于 0")

    ranked = sorted(
        pool.candidates,
        key=lambda candidate: (-candidate.combined_ticket_score, candidate.candidate_id),
    )
    stable_candidates = ranked[:stable_candidate_limit]
    rng = Random(random_seed)
    best_tickets: tuple[PortfolioTicket, ...] | None = None
    best_core: tuple[int, ...] | None = None
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
        if not _passes_non_core_constraints(selected_tuple, active_constraints):
            continue
        core_numbers = _select_core_numbers(selected_tuple, active_constraints)
        if core_numbers is None:
            continue
        tickets = _assign_roles(selected_tuple, core_numbers, active_constraints)
        if tickets is None:
            continue
        scores = _score_portfolio(
            tickets,
            core_numbers,
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
            best_scores = scores

    if best_tickets is None or best_core is None or best_scores is None:
        raise PortfolioOptimizationError(
            "在当前候选池和搜索预算内没有找到满足全部 Portfolio 硬约束的组合"
        )

    return PortfolioSelection(
        target_issue=pool.target_issue,
        data_cutoff_issue=pool.data_cutoff_issue,
        generated_at=pool.generated_at,
        random_seed=random_seed,
        tickets=best_tickets,
        core_front_numbers=best_core,
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
            "core_definition": (
                "2_to_3_designated_numbers_with_2_occurrences_and_at_most_one_with_3;"
                "incidental_non_core_repeats_allowed"
            ),
        },
    )
