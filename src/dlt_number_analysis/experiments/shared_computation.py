"""Shared candidate, scoring-matrix, and feasible-bank computation for ablations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from math import isfinite

import numpy as np
import pandas as pd

from dlt_number_analysis.data import BACK_COLUMNS, FRONT_COLUMNS, canonical_history_sha256
from dlt_number_analysis.data.data_validator import validate_draw_dataframe
from dlt_number_analysis.portfolio import (
    BankCacheKey,
    CandidatePool,
    FeasiblePortfolioBank,
    PortfolioConstraints,
    build_feasible_portfolio_bank,
    generate_candidate_pool,
    portfolio_constraints_signature,
    rescore_candidate_pool_with_vectors,
)
from dlt_number_analysis.scoring import RECENCY_WINDOWS, ScorerSpec

ABLATION_DECAYS: tuple[float, ...] = (0.85, 0.90, 0.93, 0.97, 1.0)
ABLATION_STRUCTURE_WEIGHTS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)


def deterministic_subseed(base_seed: int, *parts: object) -> int:
    """Derive a platform-independent non-negative child seed."""
    material = "\x1f".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & 0x7FFF_FFFF


def _minmax_rows(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=1, keepdims=True)
    maximum = values.max(axis=1, keepdims=True)
    span = maximum - minimum
    scaled = np.divide(
        values - minimum,
        span,
        out=np.full_like(values, 0.5, dtype=float),
        where=span != 0,
    )
    return scaled


def _recency_space_scores(
    history: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    maximum: int,
    variants: tuple[tuple[int, float], ...],
) -> np.ndarray:
    values = history.loc[:, columns].to_numpy(dtype=np.int64)
    result = np.zeros((len(variants), maximum), dtype=float)
    for index, (window, decay) in enumerate(variants):
        selected = values[-window:]
        row_weights = np.power(decay, np.arange(len(selected) - 1, -1, -1, dtype=float))
        counts = np.zeros(maximum, dtype=float)
        np.add.at(
            counts,
            selected.reshape(-1) - 1,
            np.repeat(row_weights, len(columns)),
        )
        result[index] = counts / (row_weights.sum() * len(columns))
    return _minmax_rows(result)


@dataclass(frozen=True, slots=True)
class AblationScoreCube:
    """Fifteen number-score rows combined with four structure weights by matrix math."""

    scorer_specs: tuple[ScorerSpec, ...]
    structure_weights: tuple[float, ...]
    front_candidate_scores: np.ndarray
    back_candidate_scores: np.ndarray
    number_scores: np.ndarray
    structure_scores: np.ndarray
    combined_scores: np.ndarray
    candidate_numbers_hash: str

    def __post_init__(self) -> None:
        variant_count = len(self.scorer_specs)
        candidate_count = self.structure_scores.size
        if self.front_candidate_scores.shape != (variant_count, candidate_count):
            raise ValueError("front candidate score matrix has the wrong shape")
        if self.back_candidate_scores.shape != (variant_count, candidate_count):
            raise ValueError("back candidate score matrix has the wrong shape")
        if self.number_scores.shape != (variant_count, candidate_count):
            raise ValueError("number score matrix has the wrong shape")
        if self.combined_scores.shape != (
            variant_count,
            len(self.structure_weights),
            candidate_count,
        ):
            raise ValueError("combined score cube has the wrong shape")


def build_recency_ablation_score_cube(
    history: pd.DataFrame,
    pool: CandidatePool,
    *,
    windows: tuple[int, ...] = RECENCY_WINDOWS,
    decays: tuple[float, ...] = ABLATION_DECAYS,
    structure_weights: tuple[float, ...] = ABLATION_STRUCTURE_WEIGHTS,
    front_number_weight: float = 5 / 7,
    back_number_weight: float = 2 / 7,
) -> AblationScoreCube:
    """Compute the full 15x4 candidate-score cube without rerunning the Pipeline."""
    if any(window not in RECENCY_WINDOWS for window in windows):
        raise ValueError(f"recency windows must come from {RECENCY_WINDOWS}")
    if any(not isfinite(decay) or not 0 < decay <= 1 for decay in decays):
        raise ValueError("all recency decays must be finite and in (0, 1]")
    if any(not isfinite(weight) or not 0 <= weight <= 1 for weight in structure_weights):
        raise ValueError("all structure weights must be finite and in [0, 1]")
    validated = validate_draw_dataframe(history)
    if str(validated.iloc[-1]["issue"]) != pool.data_cutoff_issue:
        raise ValueError("ablation history cutoff does not match the candidate pool")
    variants = tuple(product(windows, decays))
    specs = tuple(
        ScorerSpec(
            name="recency_weighted_frequency_score",
            parameters={"window": window, "decay": decay},
        )
        for window, decay in variants
    )
    front_space = _recency_space_scores(
        validated,
        columns=FRONT_COLUMNS,
        maximum=35,
        variants=variants,
    )
    back_space = _recency_space_scores(
        validated,
        columns=BACK_COLUMNS,
        maximum=12,
        variants=variants,
    )
    fronts = np.asarray([item.front_numbers for item in pool.candidates], dtype=np.int64) - 1
    backs = np.asarray([item.back_numbers for item in pool.candidates], dtype=np.int64) - 1
    front_scores = np.take_along_axis(front_space[:, None, :], fronts[None, :, :], axis=2).mean(
        axis=2
    )
    back_scores = np.take_along_axis(back_space[:, None, :], backs[None, :, :], axis=2).mean(axis=2)
    area_total = front_number_weight + back_number_weight
    if area_total <= 0:
        raise ValueError("front and back number weights must sum above zero")
    number_scores = (
        front_number_weight * front_scores + back_number_weight * back_scores
    ) / area_total
    structure_scores = np.asarray(
        [candidate.structure_score for candidate in pool.candidates],
        dtype=float,
    )
    structure_weight_array = np.asarray(structure_weights, dtype=float)
    combined = (
        number_scores[:, None, :] * (1.0 - structure_weight_array)[None, :, None]
        + structure_scores[None, None, :] * structure_weight_array[None, :, None]
    )
    from dlt_number_analysis.portfolio import candidate_numbers_hash

    return AblationScoreCube(
        scorer_specs=specs,
        structure_weights=structure_weights,
        front_candidate_scores=front_scores,
        back_candidate_scores=back_scores,
        number_scores=number_scores,
        structure_scores=structure_scores,
        combined_scores=combined,
        candidate_numbers_hash=candidate_numbers_hash(pool),
    )


def materialize_ablation_candidate_pool(
    pool: CandidatePool,
    cube: AblationScoreCube,
    *,
    scorer_index: int,
    structure_weight_index: int,
) -> CandidatePool:
    """Materialize one view from shared arrays without recomputing raw features."""
    structure_weight = cube.structure_weights[structure_weight_index]
    return rescore_candidate_pool_with_vectors(
        pool,
        scorer_spec=cube.scorer_specs[scorer_index],
        front_candidate_scores=cube.front_candidate_scores[scorer_index],
        back_candidate_scores=cube.back_candidate_scores[scorer_index],
        number_candidate_scores=cube.number_scores[scorer_index],
        combined_candidate_scores=cube.combined_scores[scorer_index, structure_weight_index],
        number_score_weight=1.0 - structure_weight,
        structure_score_weight=structure_weight,
        scoring_method="shared_15x4_numpy_ablation_score_cube",
    )


@dataclass(slots=True)
class SharedComputationCache:
    """Per-process cache with explicit hit/reuse accounting."""

    candidate_pools: dict[tuple[str, int, str], CandidatePool] = field(default_factory=dict)
    portfolio_banks: dict[BankCacheKey, FeasiblePortfolioBank] = field(default_factory=dict)
    candidate_requests: int = 0
    candidate_hits: int = 0
    bank_requests: int = 0
    bank_hits: int = 0

    @property
    def candidate_cache_hit_rate(self) -> float:
        return (
            0.0 if self.candidate_requests == 0 else self.candidate_hits / self.candidate_requests
        )

    @property
    def portfolio_bank_reuse_count(self) -> int:
        return self.bank_hits

    def get_candidate_pool(
        self,
        history: pd.DataFrame,
        *,
        target_issue: str,
        generated_at: datetime,
        candidate_seed: int,
        candidate_count: int = 10_000,
    ) -> CandidatePool:
        """Generate candidates/structures once for one target and deterministic seed."""
        history_hash = canonical_history_sha256(history)
        key = (target_issue, candidate_seed, history_hash)
        self.candidate_requests += 1
        if key in self.candidate_pools:
            self.candidate_hits += 1
            return self.candidate_pools[key]
        pool = generate_candidate_pool(
            history,
            target_issue=target_issue,
            generated_at=generated_at,
            random_seed=candidate_seed,
            scorer_spec=ScorerSpec(name="uniform_score"),
            candidate_count=candidate_count,
            number_score_weight=1.0,
            structure_score_weight=0.0,
            parallel_workers=1,
        )
        self.candidate_pools[key] = pool
        return pool

    def get_portfolio_bank(
        self,
        pool: CandidatePool,
        *,
        bank_seed: int,
        constraints: PortfolioConstraints,
        search_trials: int,
        maximum_bank_size: int,
    ) -> FeasiblePortfolioBank:
        """Build a constraint bank once and count subsequent reuse."""
        key = (pool.target_issue, bank_seed, portfolio_constraints_signature(constraints))
        self.bank_requests += 1
        existed = key in self.portfolio_banks
        bank = build_feasible_portfolio_bank(
            pool,
            bank_seed=bank_seed,
            constraints=constraints,
            search_trials=search_trials,
            maximum_bank_size=maximum_bank_size,
            cache=self.portfolio_banks,
        )
        if existed:
            self.bank_hits += 1
        return bank


def ablation_constraint_variants() -> tuple[PortfolioConstraints, ...]:
    """Return the six pool-size/core-count hard-constraint signatures."""
    return tuple(
        PortfolioConstraints(
            target_front_pool_size=pool_size,
            min_core_front_numbers=core_count,
            max_core_front_numbers=core_count,
        )
        for pool_size, core_count in product((16, 18, 20), (2, 3))
    )


@dataclass(frozen=True, slots=True)
class SharedAblationContext:
    """All reusable work for 360 configurations at one target/seed."""

    candidate_pool: CandidatePool
    score_cube: AblationScoreCube
    banks_by_constraints_signature: dict[str, FeasiblePortfolioBank]
    candidate_cache_hit_rate: float
    feasible_bank_generation_seconds: float
    portfolio_bank_reuse_count: int


def build_shared_ablation_context(
    history: pd.DataFrame,
    *,
    target_issue: str,
    generated_at: datetime,
    seed: int,
    cache: SharedComputationCache | None = None,
    candidate_count: int = 10_000,
    bank_search_trials: int = 5_000,
    maximum_bank_size: int = 1_000,
) -> SharedAblationContext:
    """Build one candidate set, one 15x4 score cube, and exactly six banks."""
    active_cache = cache or SharedComputationCache()
    pool = active_cache.get_candidate_pool(
        history,
        target_issue=target_issue,
        generated_at=generated_at,
        candidate_seed=deterministic_subseed(seed, target_issue, "candidates"),
        candidate_count=candidate_count,
    )
    cube = build_recency_ablation_score_cube(history, pool)
    banks: dict[str, FeasiblePortfolioBank] = {}
    bank_seconds = 0.0
    for constraints in ablation_constraint_variants():
        signature = portfolio_constraints_signature(constraints)
        bank = active_cache.get_portfolio_bank(
            pool,
            bank_seed=deterministic_subseed(seed, target_issue, "bank", signature),
            constraints=constraints,
            search_trials=bank_search_trials,
            maximum_bank_size=maximum_bank_size,
        )
        banks[signature] = bank
        bank_seconds += bank.generation_seconds
    return SharedAblationContext(
        candidate_pool=pool,
        score_cube=cube,
        banks_by_constraints_signature=banks,
        candidate_cache_hit_rate=active_cache.candidate_cache_hit_rate,
        feasible_bank_generation_seconds=bank_seconds,
        portfolio_bank_reuse_count=active_cache.portfolio_bank_reuse_count,
    )
