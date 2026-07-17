"""候选票池、评分明细和五注 Portfolio 优化测试。"""

from collections import Counter
from datetime import UTC, date, datetime, timedelta
from itertools import combinations
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.portfolio import (
    CandidateArrayBundle,
    CandidatePool,
    CandidateScoreView,
    FeasiblePortfolioBank,
    FeasiblePortfolioIndexBank,
    PortfolioConstraints,
    PortfolioSelection,
    build_candidate_score_view,
    build_feasible_portfolio_bank,
    build_stable_portfolio_subset,
    candidate_numbers_hash,
    candidate_pool_to_array_bundle,
    candidate_score_view_from_arrays,
    feasible_portfolio_bank_to_index_bank,
    generate_candidate_pool,
    load_candidate_score_details,
    optimize_portfolio,
    rescore_candidate_pool,
    sample_constraint_matched_portfolio,
    score_portfolio_bank,
    score_portfolio_bank_vectorized,
    write_candidate_score_details,
)
from dlt_number_analysis.scoring import ScorerSpec, recency_weighted_frequency_score


def make_history(count: int = 80) -> pd.DataFrame:
    """构造多样且严格递增的测试历史。"""
    rng = Random(818)
    start = date(2025, 1, 1)
    return pd.DataFrame(
        [
            [
                str(25001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                *sorted(rng.sample(range(1, 36), 5)),
                *sorted(rng.sample(range(1, 13), 2)),
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def thirty_period_scorer(history: pd.DataFrame, *, area: str) -> dict[int, float]:
    """为测试候选池固定使用最近 30 期评分。"""
    return recency_weighted_frequency_score(history, area=area, window=30)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def candidate_pool() -> CandidatePool:
    """生成一次一万注候选，供持久化和优化测试复用。"""
    return generate_candidate_pool(
        make_history(),
        target_issue="26081",
        generated_at=datetime(2026, 7, 18, tzinfo=UTC),
        random_seed=26081,
        scorer_spec=ScorerSpec(
            name="recency_weighted_frequency_score",
            parameters={"window": 30, "decay": 0.93},
        ),
        parallel_workers=2,
    )


@pytest.fixture(scope="module")
def portfolio_selection(candidate_pool: CandidatePool) -> PortfolioSelection:
    return optimize_portfolio(
        candidate_pool,
        random_seed=20260718,
        search_trials=8_000,
    )


@pytest.fixture(scope="module")
def vectorized_bank(candidate_pool: CandidatePool) -> FeasiblePortfolioBank:
    return build_feasible_portfolio_bank(
        candidate_pool,
        bank_seed=5201,
        search_trials=2_000,
        maximum_bank_size=32,
    )


def test_candidate_pool_has_at_least_ten_thousand_unique_scored_tickets(
    candidate_pool: CandidatePool,
) -> None:
    assert len(candidate_pool.candidates) == 10_000
    assert (
        len(
            {
                (candidate.front_numbers, candidate.back_numbers)
                for candidate in candidate_pool.candidates
            }
        )
        == 10_000
    )
    assert candidate_pool.random_seed == 26081
    assert candidate_pool.data_cutoff_issue == str(make_history().iloc[-1]["issue"])
    assert candidate_pool.scorer_parameters["window"] == 30
    assert candidate_pool.scorer_spec.name == "recency_weighted_frequency_score"
    assert candidate_pool.generation_parameters["parallel_workers"] == 2
    assert candidate_pool.risk_disclaimer == DISCLAIMER
    first = candidate_pool.candidates[0]
    assert 0 <= first.number_score <= 1
    assert 0 <= first.structure_score <= 1
    assert 0 <= first.combined_ticket_score <= 1
    assert first.structure_component_scores
    assert first.structure_component_details


def test_candidate_pool_rejects_fewer_than_ten_thousand() -> None:
    with pytest.raises(ValueError, match="at least"):
        generate_candidate_pool(
            make_history(),
            target_issue="26081",
            generated_at=datetime(2026, 7, 18, tzinfo=UTC),
            random_seed=1,
            candidate_count=9_999,
        )


def test_candidate_score_details_round_trip(
    candidate_pool: CandidatePool,
    tmp_path: Path,
) -> None:
    target = tmp_path / "candidate_scores.jsonl"

    write_candidate_score_details(candidate_pool, target)
    loaded = load_candidate_score_details(target)

    assert loaded == candidate_pool
    assert DISCLAIMER in target.read_text(encoding="utf-8").splitlines()[0]


def test_portfolio_optimizer_satisfies_all_default_constraints(
    portfolio_selection: PortfolioSelection,
) -> None:
    selection = portfolio_selection

    assert len(selection.tickets) == 5
    assert 16 <= selection.front_pool_size <= 21
    assert len({ticket.back_numbers for ticket in selection.tickets}) == 5
    assert len({ticket.sum_interval for ticket in selection.tickets}) >= 3
    assert len({ticket.zone_structure for ticket in selection.tickets}) >= 3
    assert sum(ticket.ticket_role == "core_stable" for ticket in selection.tickets) == 3
    assert sum(ticket.ticket_role == "exploration" for ticket in selection.tickets) == 2
    assert all(
        len(set(left.front_numbers).intersection(right.front_numbers)) <= 2
        for left, right in combinations(selection.tickets, 2)
    )
    counts = Counter(number for ticket in selection.tickets for number in ticket.front_numbers)
    assert 2 <= len(selection.core_front_numbers) <= 3
    assert all(counts[number] in (2, 3) for number in selection.core_front_numbers)
    assert sum(counts[number] == 3 for number in selection.core_front_numbers) <= 1
    assert 2 <= len(selection.support_front_numbers) <= 4
    assert all(counts[number] == 2 for number in selection.support_front_numbers)
    repeated = {number for number, count in counts.items() if count >= 2}
    assert repeated == set(selection.core_front_numbers).union(selection.support_front_numbers)
    assert max(counts.values()) <= 3
    assert selection.risk_disclaimer == DISCLAIMER

    prediction = selection.to_prediction_record()
    assert prediction.random_seed == 20260718
    assert prediction.parameters["constraints"]["total_budget"] == 10.0
    assert prediction.risk_disclaimer == DISCLAIMER


def test_portfolio_defaults_use_target_pool_not_largest_pool_reward() -> None:
    constraints = PortfolioConstraints()

    assert constraints.min_front_pool_size == 16
    assert constraints.target_front_pool_size == 18
    assert constraints.max_front_pool_size == 21


def test_hard_constraint_rejects_repeated_back_combination(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][1]["back_numbers"] = payload["tickets"][0]["back_numbers"]

    with pytest.raises(ValidationError, match="back combinations must not repeat"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_pairwise_front_overlap_above_two(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][1]["front_numbers"] = payload["tickets"][0]["front_numbers"]

    with pytest.raises(ValidationError, match="pairwise front overlap"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_requires_every_repeat_to_be_classified(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["support_front_numbers"] = payload["support_front_numbers"][1:]

    with pytest.raises(ValidationError, match="every repeated number"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_wrong_stable_exploration_split(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][0]["ticket_role"] = "exploration"

    with pytest.raises(ValidationError, match="stable ticket count"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_insufficient_structure_coverage(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    for ticket in payload["tickets"]:
        ticket["sum_interval"] = "one_interval"

    with pytest.raises(ValidationError, match="sum interval coverage"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_insufficient_zone_coverage(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    for ticket in payload["tickets"]:
        ticket["zone_structure"] = "one_zone_signature"

    with pytest.raises(ValidationError, match="zone structure coverage"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_pool_outside_configured_range(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    pool_size = payload["front_pool_size"]
    payload["constraints"]["min_front_pool_size"] = 5
    payload["constraints"]["max_front_pool_size"] = pool_size - 1
    payload["constraints"]["target_front_pool_size"] = pool_size - 1

    with pytest.raises(ValidationError, match="front pool size"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_more_than_three_occurrences(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][3]["front_numbers"] = (9, 18, 21, 27, 28)
    payload["front_pool_size"] = len(
        {number for ticket in payload["tickets"] for number in ticket["front_numbers"]}
    )

    with pytest.raises(ValidationError, match="maximum occurrence"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_core_occurrence_below_configured_minimum(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["constraints"]["core_min_occurrences"] = 3

    with pytest.raises(ValidationError, match="core front number occurrences"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_too_many_core_numbers(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["constraints"]["max_core_front_numbers"] = 2

    with pytest.raises(ValidationError, match="core front number count"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_too_many_support_numbers(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["constraints"]["max_support_front_numbers"] = 2

    with pytest.raises(ValidationError, match="support front number count"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_support_number_above_two_occurrences(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"][1]["front_numbers"] = (9, 21, 22, 25, 35)
    payload["front_pool_size"] = len(
        {number for ticket in payload["tickets"] for number in ticket["front_numbers"]}
    )

    with pytest.raises(ValidationError, match="support front number occurrences"):
        PortfolioSelection.model_validate(payload)


def test_hard_constraint_rejects_wrong_ticket_count(
    portfolio_selection: PortfolioSelection,
) -> None:
    payload = portfolio_selection.model_dump()
    payload["tickets"] = payload["tickets"][:4]

    with pytest.raises(ValidationError):
        PortfolioSelection.model_validate(payload)


def test_constraint_model_rejects_budget_mismatch() -> None:
    with pytest.raises(ValidationError, match="total budget"):
        PortfolioConstraints(total_budget=12)


def test_constraint_matched_bank_sampling_ignores_candidate_id_order(
    candidate_pool: CandidatePool,
) -> None:
    bank = build_feasible_portfolio_bank(
        candidate_pool,
        bank_seed=5101,
        search_trials=4_000,
        maximum_bank_size=250,
    )
    reversed_candidates = tuple(
        candidate.model_copy(update={"candidate_id": f"reordered-{index:05d}"})
        for index, candidate in enumerate(reversed(candidate_pool.candidates), start=1)
    )
    reordered = CandidatePool.model_validate(
        candidate_pool.model_copy(update={"candidates": reversed_candidates}).model_dump()
    )

    assert candidate_numbers_hash(reordered) == bank.candidate_numbers_hash
    original_distribution = [
        tuple(
            (ticket.front_numbers, ticket.back_numbers)
            for ticket in sample_constraint_matched_portfolio(
                candidate_pool, bank, random_seed=seed
            ).tickets
        )
        for seed in range(30)
    ]
    reordered_distribution = [
        tuple(
            (ticket.front_numbers, ticket.back_numbers)
            for ticket in sample_constraint_matched_portfolio(
                reordered, bank, random_seed=seed
            ).tickets
        )
        for seed in range(30)
    ]

    assert original_distribution == reordered_distribution
    assert bank.bank_size > 0
    assert bank.acceptance_rate > 0
    assert bank.bank_hash


def test_optimized_selection_scores_the_same_feasible_bank(
    candidate_pool: CandidatePool,
) -> None:
    bank = build_feasible_portfolio_bank(
        candidate_pool,
        bank_seed=5102,
        search_trials=3_000,
        maximum_bank_size=100,
    )

    scored = score_portfolio_bank(candidate_pool, bank, random_seed=88)

    assert scored.evaluated_portfolios == bank.bank_size
    assert scored.bank_hash == bank.bank_hash
    assert scored.selection.optimizer_parameters["bank_hash"] == bank.bank_hash


def test_feasible_bank_expands_search_until_minimum_size(
    candidate_pool: CandidatePool,
) -> None:
    bank = build_feasible_portfolio_bank(
        candidate_pool,
        bank_seed=5103,
        search_trials=100,
        maximum_search_trials=2_000,
        minimum_bank_size=100,
        maximum_bank_size=120,
    )

    assert bank.initial_search_trials == 100
    assert bank.search_trials > bank.initial_search_trials
    assert bank.search_expansion_count > 0
    assert bank.bank_size >= 100
    assert bank.minimum_bank_size == 100


def test_candidate_array_bundle_preserves_pool_order_scores_and_hash(
    candidate_pool: CandidatePool,
) -> None:
    bundle = candidate_pool_to_array_bundle(candidate_pool)

    assert isinstance(bundle, CandidateArrayBundle)
    assert bundle.front_numbers.tolist() == [
        list(candidate.front_numbers) for candidate in candidate_pool.candidates
    ]
    assert bundle.back_numbers.tolist() == [
        list(candidate.back_numbers) for candidate in candidate_pool.candidates
    ]
    assert bundle.number_scores.tolist() == [
        candidate.number_score for candidate in candidate_pool.candidates
    ]
    assert bundle.structure_scores.tolist() == [
        candidate.structure_score for candidate in candidate_pool.candidates
    ]
    assert bundle.combined_scores.tolist() == [
        candidate.combined_ticket_score for candidate in candidate_pool.candidates
    ]
    assert bundle.candidate_numbers_hash == candidate_numbers_hash(candidate_pool)
    assert all(not array.flags.writeable for array in bundle.arrays)


def test_index_bank_reproduces_every_bank_ticket(
    candidate_pool: CandidatePool,
    vectorized_bank: FeasiblePortfolioBank,
) -> None:
    bundle = candidate_pool_to_array_bundle(candidate_pool)
    index_bank = feasible_portfolio_bank_to_index_bank(bundle, vectorized_bank)

    assert isinstance(index_bank, FeasiblePortfolioIndexBank)
    assert index_bank.bank_hash == vectorized_bank.bank_hash
    assert index_bank.candidate_numbers_hash == vectorized_bank.candidate_numbers_hash
    entries = sorted(vectorized_bank.entries, key=lambda entry: entry.entry_hash)
    for entry_index, entry in enumerate(entries):
        identities = tuple(
            (
                tuple(int(value) for value in bundle.front_numbers[index]),
                tuple(int(value) for value in bundle.back_numbers[index]),
            )
            for index in index_bank.candidate_indices[entry_index]
        )
        assert identities == tuple(
            (ticket.front_numbers, ticket.back_numbers) for ticket in entry.tickets
        )


@pytest.mark.parametrize(
    ("scorer_spec", "number_weight", "structure_weight"),
    (
        (ScorerSpec(name="uniform_score"), 0.5, 0.5),
        (
            ScorerSpec(
                name="recency_weighted_frequency_score",
                parameters={"window": 30, "decay": 0.93},
            ),
            1.0,
            0.0,
        ),
        (ScorerSpec(name="uniform_score"), 0.0, 1.0),
        (
            ScorerSpec(
                name="recency_weighted_frequency_score",
                parameters={"window": 30, "decay": 0.93},
            ),
            0.5,
            0.5,
        ),
        (
            ScorerSpec(
                name="hot_cold_blend_score",
                parameters={
                    "hot_window": 10,
                    "cold_window": 100,
                    "hot_weight": 0.7,
                    "decay": 0.93,
                },
            ),
            0.5,
            0.5,
        ),
    ),
    ids=("B2", "B3", "B4", "B5", "B6"),
)
def test_vectorized_bank_scoring_matches_object_reference(
    candidate_pool: CandidatePool,
    vectorized_bank: FeasiblePortfolioBank,
    scorer_spec: ScorerSpec,
    number_weight: float,
    structure_weight: float,
) -> None:
    history = make_history()
    materialized = rescore_candidate_pool(
        history,
        candidate_pool,
        scorer_spec=scorer_spec,
        number_score_weight=number_weight,
        structure_score_weight=structure_weight,
    )
    reference = score_portfolio_bank(materialized, vectorized_bank, random_seed=20260000)
    bundle = candidate_pool_to_array_bundle(candidate_pool)
    score_view = build_candidate_score_view(
        history,
        bundle,
        scorer_spec=scorer_spec,
        number_score_weight=number_weight,
        structure_score_weight=structure_weight,
    )
    index_bank = feasible_portfolio_bank_to_index_bank(bundle, vectorized_bank)
    vectorized = score_portfolio_bank_vectorized(
        score_view,
        index_bank,
        random_seed=20260000,
    )

    assert isinstance(score_view, CandidateScoreView)
    assert (
        vectorized.selected_entry_hash
        == reference.selection.optimizer_parameters["selected_bank_entry_hash"]
    )
    assert [
        (ticket.front_numbers, ticket.back_numbers) for ticket in vectorized.selection.tickets
    ] == [(ticket.front_numbers, ticket.back_numbers) for ticket in reference.selection.tickets]
    assert vectorized.selection.core_front_numbers == reference.selection.core_front_numbers
    assert vectorized.selection.support_front_numbers == reference.selection.support_front_numbers
    for field in (
        "single_ticket_score",
        "portfolio_diversity",
        "core_concentration",
        "structure_coverage",
        "excessive_repeat_penalty",
        "combined_portfolio_score",
    ):
        assert getattr(vectorized.selection.scores, field) == pytest.approx(
            getattr(reference.selection.scores, field), abs=1e-12
        )


def test_b1_sampling_is_independent_of_score_view(
    candidate_pool: CandidatePool,
    vectorized_bank: FeasiblePortfolioBank,
) -> None:
    uniform = rescore_candidate_pool(
        make_history(),
        candidate_pool,
        scorer_spec=ScorerSpec(name="uniform_score"),
        number_score_weight=1.0,
        structure_score_weight=0.0,
    )
    original = sample_constraint_matched_portfolio(
        candidate_pool, vectorized_bank, random_seed=20260000
    )
    rescored = sample_constraint_matched_portfolio(uniform, vectorized_bank, random_seed=20260000)

    assert (
        original.optimizer_parameters["selected_bank_entry_hash"]
        == rescored.optimizer_parameters["selected_bank_entry_hash"]
    )
    assert original.optimizer_parameters["portfolio_scoring_method"] == "random_bank_sample"
    assert [(ticket.front_numbers, ticket.back_numbers) for ticket in original.tickets] == [
        (ticket.front_numbers, ticket.back_numbers) for ticket in rescored.tickets
    ]


def test_stable_portfolio_subset_hash_and_boundaries_are_deterministic(
    vectorized_bank: FeasiblePortfolioBank,
) -> None:
    first = build_stable_portfolio_subset(vectorized_bank, 20)
    second = build_stable_portfolio_subset(vectorized_bank, 20)

    assert first == second
    assert first.parent_bank_hash == vectorized_bank.bank_hash
    assert first.subset_size == 20
    assert first.first_entry_hash == min(entry.entry_hash for entry in vectorized_bank.entries)
    assert (
        first.last_entry_hash == sorted(entry.entry_hash for entry in vectorized_bank.entries)[19]
    )


@pytest.fixture(scope="module")
def benchmark_subset_bank(candidate_pool: CandidatePool) -> FeasiblePortfolioBank:
    return build_feasible_portfolio_bank(
        candidate_pool,
        bank_seed=5301,
        search_trials=2_000,
        maximum_bank_size=250,
    )


def test_b2_object_and_vectorized_match_for_50_100_250_subsets(
    candidate_pool: CandidatePool,
    benchmark_subset_bank: FeasiblePortfolioBank,
) -> None:
    history = make_history()
    scorer = ScorerSpec(name="uniform_score")
    materialized = rescore_candidate_pool(
        history,
        candidate_pool,
        scorer_spec=scorer,
        number_score_weight=0.5,
        structure_score_weight=0.5,
    )
    bundle = candidate_pool_to_array_bundle(candidate_pool)
    score_view = build_candidate_score_view(
        history,
        bundle,
        scorer_spec=scorer,
        number_score_weight=0.5,
        structure_score_weight=0.5,
    )
    for subset_size in (50, 100, 250):
        subset = build_stable_portfolio_subset(benchmark_subset_bank, subset_size)
        reference = score_portfolio_bank(materialized, subset.bank, random_seed=20260000)
        vectorized = score_portfolio_bank_vectorized(
            score_view,
            feasible_portfolio_bank_to_index_bank(bundle, subset.bank),
            random_seed=20260000,
        )

        assert (
            reference.selection.optimizer_parameters["selected_bank_entry_hash"]
            == vectorized.selected_entry_hash
        )
        assert [
            (ticket.front_numbers, ticket.back_numbers) for ticket in reference.selection.tickets
        ] == [
            (ticket.front_numbers, ticket.back_numbers) for ticket in vectorized.selection.tickets
        ]
        assert reference.selection.core_front_numbers == vectorized.selection.core_front_numbers
        assert (
            reference.selection.support_front_numbers == vectorized.selection.support_front_numbers
        )
        for field, value in reference.selection.scores.model_dump().items():
            assert value == pytest.approx(
                vectorized.selection.scores.model_dump()[field],
                abs=1e-12,
            )


def test_vectorized_identity_survives_candidate_id_and_order_changes(
    candidate_pool: CandidatePool,
    vectorized_bank: FeasiblePortfolioBank,
) -> None:
    reversed_candidates = tuple(
        candidate.model_copy(update={"candidate_id": f"changed-{index:05d}"})
        for index, candidate in enumerate(reversed(candidate_pool.candidates), start=1)
    )
    reordered = CandidatePool.model_validate(
        candidate_pool.model_copy(update={"candidates": reversed_candidates}).model_dump()
    )
    original_bundle = candidate_pool_to_array_bundle(candidate_pool)
    reordered_bundle = candidate_pool_to_array_bundle(reordered)
    scorer_spec = ScorerSpec(name="uniform_score")
    original = score_portfolio_bank_vectorized(
        build_candidate_score_view(
            make_history(),
            original_bundle,
            scorer_spec=scorer_spec,
            number_score_weight=0.5,
            structure_score_weight=0.5,
        ),
        feasible_portfolio_bank_to_index_bank(original_bundle, vectorized_bank),
        random_seed=99,
    )
    reordered_result = score_portfolio_bank_vectorized(
        build_candidate_score_view(
            make_history(),
            reordered_bundle,
            scorer_spec=scorer_spec,
            number_score_weight=0.5,
            structure_score_weight=0.5,
        ),
        feasible_portfolio_bank_to_index_bank(reordered_bundle, vectorized_bank),
        random_seed=99,
    )

    assert original.selected_entry_hash == reordered_result.selected_entry_hash
    assert [
        (ticket.front_numbers, ticket.back_numbers) for ticket in original.selection.tickets
    ] == [
        (ticket.front_numbers, ticket.back_numbers) for ticket in reordered_result.selection.tickets
    ]


def test_vectorized_tie_break_uses_largest_entry_hash(
    candidate_pool: CandidatePool,
    vectorized_bank: FeasiblePortfolioBank,
) -> None:
    bundle = candidate_pool_to_array_bundle(candidate_pool)
    constant = np.full(len(bundle.candidate_ids), 0.5, dtype=float)
    score_view = candidate_score_view_from_arrays(
        bundle,
        number_scores=constant,
        combined_scores=constant,
        scorer_spec=ScorerSpec(name="uniform_score"),
        number_score_weight=1.0,
        structure_score_weight=0.0,
        scoring_method="constant_tie_test",
    )
    result = score_portfolio_bank_vectorized(
        score_view,
        feasible_portfolio_bank_to_index_bank(bundle, vectorized_bank),
        random_seed=1,
        single_ticket_weight=1.0,
        diversity_weight=0.0,
        core_weight=0.0,
        structure_weight=0.0,
        repeat_penalty_weight=0.0,
    )

    assert result.selected_entry_hash == max(entry.entry_hash for entry in vectorized_bank.entries)
