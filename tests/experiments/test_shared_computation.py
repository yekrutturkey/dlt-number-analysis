"""Shared candidate and NumPy ablation-score correctness tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from random import Random

import numpy as np
import pandas as pd
import pytest

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments import (
    SharedComputationCache,
    ablation_constraint_variants,
    build_recency_ablation_score_cube,
    materialize_ablation_candidate_pool,
)
from dlt_number_analysis.portfolio import CandidatePool, rescore_candidate_pool
from dlt_number_analysis.scoring import ScorerSpec


def _history(count: int = 120) -> pd.DataFrame:
    rng = Random(918)
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


@pytest.fixture(scope="module")
def shared_pool() -> CandidatePool:
    cache = SharedComputationCache()
    history = _history()
    return cache.get_candidate_pool(
        history,
        target_issue="26001",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        candidate_seed=711,
    )


def test_shared_candidate_cache_generates_once_per_target_and_seed() -> None:
    cache = SharedComputationCache()
    history = _history()
    options = {
        "target_issue": "26001",
        "generated_at": datetime(2026, 1, 1, tzinfo=UTC),
        "candidate_seed": 712,
    }

    first = cache.get_candidate_pool(history, **options)
    second = cache.get_candidate_pool(history, **options)

    assert first is second
    assert cache.candidate_requests == 2
    assert cache.candidate_hits == 1
    assert cache.candidate_cache_hit_rate == 0.5


def test_15_by_4_score_cube_matches_scalar_rescore(
    shared_pool: CandidatePool,
) -> None:
    history = _history()
    cube = build_recency_ablation_score_cube(history, shared_pool)
    scorer_index = next(
        index
        for index, spec in enumerate(cube.scorer_specs)
        if spec.parameters == {"window": 30, "decay": 0.93}
    )
    structure_index = cube.structure_weights.index(0.5)
    materialized = materialize_ablation_candidate_pool(
        shared_pool,
        cube,
        scorer_index=scorer_index,
        structure_weight_index=structure_index,
    )
    scalar = rescore_candidate_pool(
        history,
        shared_pool,
        scorer_spec=ScorerSpec(
            name="recency_weighted_frequency_score",
            parameters={"window": 30, "decay": 0.93},
        ),
        number_score_weight=0.5,
        structure_score_weight=0.5,
    )

    assert cube.number_scores.shape == (15, 10_000)
    assert cube.combined_scores.shape == (15, 4, 10_000)
    assert len(ablation_constraint_variants()) == 6
    np.testing.assert_allclose(
        [item.combined_ticket_score for item in materialized.candidates],
        [item.combined_ticket_score for item in scalar.candidates],
        rtol=0,
        atol=1e-12,
    )
    assert materialized.candidates[0].features == shared_pool.candidates[0].features


def test_feasible_bank_cache_builds_once_per_target_seed_and_constraints(
    shared_pool: CandidatePool,
) -> None:
    cache = SharedComputationCache()
    constraints = ablation_constraint_variants()[0]

    first = cache.get_portfolio_bank(
        shared_pool,
        bank_seed=991,
        constraints=constraints,
        search_trials=4_000,
        maximum_bank_size=100,
    )
    second = cache.get_portfolio_bank(
        shared_pool,
        bank_seed=991,
        constraints=constraints,
        search_trials=4_000,
        maximum_bank_size=100,
    )

    assert first is second
    assert cache.bank_requests == 2
    assert cache.bank_hits == 1
    assert cache.portfolio_bank_reuse_count == 1
