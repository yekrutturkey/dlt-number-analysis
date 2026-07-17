"""History-to-PredictionRecord pipeline with one auditable configuration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from math import isclose, isfinite
from statistics import fmean
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import validate_draw_dataframe
from dlt_number_analysis.models import PredictionRecord
from dlt_number_analysis.portfolio import (
    CandidatePool,
    PortfolioConstraints,
    PortfolioSelection,
    generate_candidate_pool,
    optimize_portfolio,
)
from dlt_number_analysis.scoring import (
    DEFAULT_STRUCTURE_WEIGHTS,
    HistoricalStructureProfile,
    ScorerSpec,
    fit_structure_profile,
)

PipelineProfile = Literal["fast", "standard", "final"]
PROFILE_DEFAULTS: dict[PipelineProfile, dict[str, int]] = {
    "fast": {
        "candidate_count": 10_000,
        "optimizer_search_trials": 5_000,
        "stable_candidate_limit": 2_000,
        "parallel_workers": 1,
    },
    "standard": {
        "candidate_count": 10_000,
        "optimizer_search_trials": 25_000,
        "stable_candidate_limit": 2_000,
        "parallel_workers": 1,
    },
    "final": {
        "candidate_count": 25_000,
        "optimizer_search_trials": 100_000,
        "stable_candidate_limit": 5_000,
        "parallel_workers": 1,
    },
}


class PipelineSeeds(BaseModel):
    """Every random seed consumed by one pipeline execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_pool: int
    portfolio_optimizer: int

    @classmethod
    def from_base_seed(cls, base_seed: int) -> PipelineSeeds:
        return cls(candidate_pool=base_seed, portfolio_optimizer=base_seed + 1)


class PipelineConfig(BaseModel):
    """Single source of truth for scorer, candidate, and optimizer behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: PipelineProfile = "standard"
    scorer_spec: ScorerSpec = Field(
        default_factory=lambda: ScorerSpec(
            name="recency_weighted_frequency_score",
            parameters={"window": 30, "decay": 0.93},
        )
    )
    structure_feature_weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_STRUCTURE_WEIGHTS)
    )
    laplace_alpha: float = Field(default=1.0, gt=0)
    candidate_count: int = Field(ge=10_000)
    front_number_weight: float = Field(default=5 / 7, ge=0)
    back_number_weight: float = Field(default=2 / 7, ge=0)
    number_score_weight: float = Field(default=0.5, ge=0)
    structure_score_weight: float = Field(default=0.5, ge=0)
    portfolio_constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)
    optimizer_search_trials: int = Field(ge=1)
    stable_candidate_limit: int = Field(ge=5)
    parallel_workers: int = Field(ge=1, le=64)
    candidate_scoring_method: Literal["scalar", "numpy_batch_features"] = "numpy_batch_features"
    minimum_history_size: int = Field(default=100, ge=1)
    single_ticket_weight: float = Field(default=0.45, ge=0)
    diversity_weight: float = Field(default=0.25, ge=0)
    core_weight: float = Field(default=0.15, ge=0)
    structure_weight: float = Field(default=0.15, ge=0)
    repeat_penalty_weight: float = Field(default=0.20, ge=0)
    model_version: str = "prediction-pipeline-v0.5"

    @model_validator(mode="before")
    @classmethod
    def expand_profile_defaults(cls, value: Any) -> Any:
        """Materialize a named run profile while preserving explicit overrides."""
        if not isinstance(value, Mapping):
            return value
        resolved = dict(value)
        profile = resolved.get("profile", "standard")
        defaults = PROFILE_DEFAULTS.get(profile)
        if defaults is None:
            return resolved
        for name, default in defaults.items():
            if resolved.get(name) is None:
                resolved[name] = default
        return resolved

    @model_validator(mode="after")
    def validate_weights_and_limits(self) -> PipelineConfig:
        if set(self.structure_feature_weights) != set(DEFAULT_STRUCTURE_WEIGHTS):
            raise ValueError("structure weights must cover every concept feature")
        if any(
            not isfinite(weight) or weight < 0 for weight in self.structure_feature_weights.values()
        ):
            raise ValueError("structure weights must be finite and non-negative")
        if not isclose(sum(self.structure_feature_weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("structure weights must sum to 1")
        if self.front_number_weight + self.back_number_weight <= 0:
            raise ValueError("front/back number weights must sum above zero")
        if self.number_score_weight + self.structure_score_weight <= 0:
            raise ValueError("number/structure weights must sum above zero")
        if (
            self.single_ticket_weight
            + self.diversity_weight
            + self.core_weight
            + self.structure_weight
            <= 0
        ):
            raise ValueError("positive portfolio objective weights must sum above zero")
        if self.stable_candidate_limit > self.candidate_count:
            raise ValueError("stable_candidate_limit cannot exceed candidate_count")
        return self


class CandidatePoolSummary(BaseModel):
    """Compact candidate-pool statistics stored with the final prediction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count: int = Field(ge=10_000)
    minimum_combined_score: float = Field(ge=0, le=1)
    mean_combined_score: float = Field(ge=0, le=1)
    maximum_combined_score: float = Field(ge=0, le=1)
    candidate_generation_seconds: float = Field(ge=0)
    candidate_scoring_seconds: float = Field(ge=0)
    candidate_scoring_method: Literal["scalar", "numpy_batch_features"]
    scorer_spec: ScorerSpec
    structure_profile_cutoff_issue: str = Field(pattern=r"^\d+$")


class PipelineResult(BaseModel):
    """End-to-end output used identically by live generation and rolling backtests."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    prediction: PredictionRecord
    candidate_pool_summary: CandidatePoolSummary
    portfolio_selection: PortfolioSelection
    structure_profile: HistoricalStructureProfile
    config: PipelineConfig
    random_seeds: PipelineSeeds
    short_history_override: bool = False
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        if value != DISCLAIMER:
            raise ValueError("pipeline result is missing the fixed risk disclaimer")
        return value


def _summarize_pool(
    pool: CandidatePool,
    profile: HistoricalStructureProfile,
) -> CandidatePoolSummary:
    scores = [candidate.combined_ticket_score for candidate in pool.candidates]
    return CandidatePoolSummary(
        candidate_count=len(pool.candidates),
        minimum_combined_score=min(scores),
        mean_combined_score=fmean(scores),
        maximum_combined_score=max(scores),
        candidate_generation_seconds=float(
            pool.generation_parameters["candidate_generation_seconds"]
        ),
        candidate_scoring_seconds=float(pool.generation_parameters["candidate_scoring_seconds"]),
        candidate_scoring_method=str(pool.generation_parameters["candidate_scoring_method"]),
        scorer_spec=pool.scorer_spec,
        structure_profile_cutoff_issue=profile.data_cutoff_issue,
    )


class PredictionPipeline:
    """Execute the shared v0.4 prediction flow without reading any future draw."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(
        self,
        history: pd.DataFrame,
        *,
        target_issue: str,
        generated_at: datetime,
        random_seeds: PipelineSeeds,
        allow_short_history: bool = False,
    ) -> PipelineResult:
        validated = validate_draw_dataframe(history)
        short_history_override = len(validated) < self.config.minimum_history_size
        if short_history_override and not allow_short_history:
            raise ValueError(
                "pipeline history is shorter than minimum_history_size; "
                "use an explicit research override to continue"
            )
        cutoff_issue = str(validated.iloc[-1]["issue"])
        if int(cutoff_issue) >= int(target_issue):
            raise ValueError("pipeline history cutoff must precede target_issue")
        profile = fit_structure_profile(
            validated,
            feature_weights=self.config.structure_feature_weights,
            laplace_alpha=self.config.laplace_alpha,
        )
        pool = generate_candidate_pool(
            validated,
            target_issue=target_issue,
            generated_at=generated_at,
            random_seed=random_seeds.candidate_pool,
            scorer_spec=self.config.scorer_spec,
            structure_profile=profile,
            candidate_count=self.config.candidate_count,
            front_number_weight=self.config.front_number_weight,
            back_number_weight=self.config.back_number_weight,
            number_score_weight=self.config.number_score_weight,
            structure_score_weight=self.config.structure_score_weight,
            parallel_workers=self.config.parallel_workers,
            candidate_scoring_method=self.config.candidate_scoring_method,
        )
        selection = optimize_portfolio(
            pool,
            random_seed=random_seeds.portfolio_optimizer,
            constraints=self.config.portfolio_constraints,
            search_trials=self.config.optimizer_search_trials,
            stable_candidate_limit=self.config.stable_candidate_limit,
            single_ticket_weight=self.config.single_ticket_weight,
            diversity_weight=self.config.diversity_weight,
            core_weight=self.config.core_weight,
            structure_weight=self.config.structure_weight,
            repeat_penalty_weight=self.config.repeat_penalty_weight,
        )
        summary = _summarize_pool(pool, profile)
        base_prediction = selection.to_prediction_record()
        prediction = base_prediction.model_copy(
            update={
                "strategy_name": "optimized_portfolio_strategy",
                "model_version": self.config.model_version,
                "parameters": {
                    **base_prediction.parameters,
                    "pipeline_config": self.config.model_dump(mode="json"),
                    "random_seeds": random_seeds.model_dump(mode="json"),
                    "candidate_pool_summary": summary.model_dump(mode="json"),
                    "short_history_override": short_history_override,
                },
            }
        )
        prediction = PredictionRecord.model_validate(prediction.model_dump())
        return PipelineResult(
            prediction=prediction,
            candidate_pool_summary=summary,
            portfolio_selection=selection,
            structure_profile=profile,
            config=self.config,
            random_seeds=random_seeds,
            short_history_override=short_history_override,
        )


def optimized_portfolio_strategy(
    history: pd.DataFrame,
    *,
    target_issue: str,
    generated_at: datetime,
    random_seed: int,
    pipeline_config: PipelineConfig | None = None,
    allow_short_history: bool = False,
) -> PredictionRecord:
    """Run the shared pipeline as a strict-history rolling-backtest strategy."""
    return (
        PredictionPipeline(pipeline_config)
        .run(
            history,
            target_issue=target_issue,
            generated_at=generated_at,
            random_seeds=PipelineSeeds.from_base_seed(random_seed),
            allow_short_history=allow_short_history,
        )
        .prediction
    )
