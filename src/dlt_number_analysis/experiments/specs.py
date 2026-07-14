"""Versioned strategy experiment and strict ablation specifications."""

from __future__ import annotations

from itertools import product
from math import isclose
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlt_number_analysis.portfolio import PortfolioConstraints
from dlt_number_analysis.scoring import ScorerSpec

ExperimentPhase = Literal["development", "calibration", "final_holdout"]
PortfolioStrategy = Literal[
    "uniform_random",
    "constraint_matched_random",
    "optimized_portfolio",
    "max_coverage",
    "core_rotation",
]
CandidateGenerationMethod = Literal[
    "direct_uniform",
    "seeded_uniform_candidate_pool",
]


class DataSplitSpec(BaseModel):
    """Contiguous time split ratios and the phase this run is allowed to consume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    development_ratio: float = Field(default=0.60, gt=0, lt=1)
    calibration_ratio: float = Field(default=0.20, gt=0, lt=1)
    final_holdout_ratio: float = Field(default=0.20, gt=0, lt=1)
    phase: ExperimentPhase = "development"

    @model_validator(mode="after")
    def validate_ratios(self) -> Self:
        if not isclose(
            self.development_ratio + self.calibration_ratio + self.final_holdout_ratio,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("development, calibration, and holdout ratios must sum to 1")
        return self


class ExperimentSpec(BaseModel):
    """One auditable strategy configuration with no parameters outside the spec."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    experiment_version: str = Field(min_length=1)
    scorer_spec: ScorerSpec
    structure_score_weight: float = Field(ge=0, le=1)
    number_score_weight: float = Field(ge=0, le=1)
    portfolio_strategy: PortfolioStrategy
    portfolio_constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)
    candidate_generation_method: CandidateGenerationMethod
    seeds: tuple[int, ...] = Field(default=(20260000,), min_length=1)
    data_split: DataSplitSpec = Field(default_factory=DataSplitSpec)

    @model_validator(mode="after")
    def validate_score_weights(self) -> Self:
        if not isclose(
            self.structure_score_weight + self.number_score_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("number and structure score weights must sum to 1")
        if self.portfolio_strategy == "uniform_random" and (
            self.candidate_generation_method != "direct_uniform"
        ):
            raise ValueError("uniform random must use direct_uniform generation")
        if self.portfolio_strategy != "uniform_random" and (
            self.candidate_generation_method != "seeded_uniform_candidate_pool"
        ):
            raise ValueError("portfolio experiments require the seeded candidate pool")
        return self


def baseline_experiment_specs(
    *,
    seeds: tuple[int, ...] = (20260000,),
    phase: ExperimentPhase = "development",
) -> tuple[ExperimentSpec, ...]:
    """Return the required B0-B8 comparison set."""
    split = DataSplitSpec(phase=phase)
    uniform = ScorerSpec(name="uniform_score")
    recency = ScorerSpec(
        name="recency_weighted_frequency_score",
        parameters={"window": 30, "decay": 0.93},
    )
    hot_cold = ScorerSpec(
        name="hot_cold_blend_score",
        parameters={
            "hot_window": 10,
            "cold_window": 100,
            "hot_weight": 0.65,
            "decay": 0.93,
        },
    )

    def spec(
        experiment_id: str,
        scorer: ScorerSpec,
        number_weight: float,
        strategy: PortfolioStrategy,
        method: CandidateGenerationMethod = "seeded_uniform_candidate_pool",
    ) -> ExperimentSpec:
        bank_version = experiment_id in {f"B{index}" for index in range(1, 7)}
        return ExperimentSpec(
            experiment_id=experiment_id,
            experiment_version=(
                f"v0.5.1-{experiment_id.lower()}-shared-bank-v1"
                if bank_version
                else f"v0.5-{experiment_id.lower()}-v1"
            ),
            scorer_spec=scorer,
            structure_score_weight=1.0 - number_weight,
            number_score_weight=number_weight,
            portfolio_strategy=strategy,
            candidate_generation_method=method,
            seeds=seeds,
            data_split=split,
        )

    return (
        spec("B0", uniform, 1.0, "uniform_random", "direct_uniform"),
        spec("B1", uniform, 1.0, "constraint_matched_random"),
        spec("B2", uniform, 0.5, "optimized_portfolio"),
        spec("B3", recency, 1.0, "optimized_portfolio"),
        spec("B4", uniform, 0.0, "optimized_portfolio"),
        spec("B5", recency, 0.5, "optimized_portfolio"),
        spec("B6", hot_cold, 0.5, "optimized_portfolio"),
        spec("B7", recency, 1.0, "max_coverage"),
        spec("B8", recency, 1.0, "core_rotation"),
    )


def ablation_experiment_specs(
    *,
    seeds: tuple[int, ...] = (20260000,),
    phase: Literal["development", "calibration"] = "development",
) -> tuple[ExperimentSpec, ...]:
    """Build all 360 allowed parameter combinations outside the final holdout."""
    if phase not in {"development", "calibration"}:
        raise ValueError("ablation parameter selection cannot consume final holdout data")
    specifications: list[ExperimentSpec] = []
    for window, decay, structure_weight, pool_size, core_count in product(
        (10, 30, 100),
        (0.85, 0.90, 0.93, 0.97, 1.0),
        (0.0, 0.25, 0.5, 0.75),
        (16, 18, 20),
        (2, 3),
    ):
        identifier = f"A-w{window}-d{decay:.2f}-s{structure_weight:.2f}-p{pool_size}-c{core_count}"
        constraints = PortfolioConstraints(
            target_front_pool_size=pool_size,
            min_core_front_numbers=core_count,
            max_core_front_numbers=core_count,
        )
        specifications.append(
            ExperimentSpec(
                experiment_id=identifier,
                experiment_version="v0.5.1-ablation-shared-v1",
                scorer_spec=ScorerSpec(
                    name="recency_weighted_frequency_score",
                    parameters={"window": window, "decay": decay},
                ),
                structure_score_weight=structure_weight,
                number_score_weight=1.0 - structure_weight,
                portfolio_strategy="optimized_portfolio",
                portfolio_constraints=constraints,
                candidate_generation_method="seeded_uniform_candidate_pool",
                seeds=seeds,
                data_split=DataSplitSpec(phase=phase),
            )
        )
    return tuple(specifications)
