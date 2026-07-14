"""Leakage-safe staged ablation plans from screening through one-shot holdout."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlt_number_analysis.experiments.specs import (
    DataSplitSpec,
    ExperimentPhase,
    ExperimentSpec,
    ablation_experiment_specs,
)
from dlt_number_analysis.experiments.splits import split_history

AblationStage = Literal[
    "screening",
    "full_development",
    "calibration",
    "final_holdout",
]


class StagedAblationPlan(BaseModel):
    """Frozen stage scope, seeds, target stride, and selected experiment specs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: AblationStage
    phase: ExperimentPhase
    target_stride: int = Field(ge=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    specifications: tuple[ExperimentSpec, ...] = Field(min_length=1)
    selection_rule: str
    final_holdout_locked: bool = False

    @model_validator(mode="after")
    def validate_stage(self) -> StagedAblationPlan:
        if any(spec.data_split.phase != self.phase for spec in self.specifications):
            raise ValueError("every staged specification must use the declared phase")
        if any(spec.seeds != self.seeds for spec in self.specifications):
            raise ValueError("every staged specification must bind the stage seeds")
        if self.stage == "final_holdout" and (
            len(self.specifications) != 1 or not self.final_holdout_locked
        ):
            raise ValueError("final holdout requires exactly one frozen specification")
        return self


def _ranked_specs(
    phase: Literal["development", "calibration"],
    ranked_experiment_ids: Sequence[str] | None,
) -> list[ExperimentSpec]:
    specifications = list(ablation_experiment_specs(phase=phase))
    if ranked_experiment_ids is None:
        return specifications
    by_id = {spec.experiment_id: spec for spec in specifications}
    unknown = set(ranked_experiment_ids).difference(by_id)
    if unknown:
        raise ValueError(f"ranked ablation IDs are unknown: {sorted(unknown)}")
    return [by_id[experiment_id] for experiment_id in ranked_experiment_ids]


def build_staged_ablation_plan(
    stage: AblationStage,
    *,
    ranked_experiment_ids: Sequence[str] | None = None,
    calibration_count: int = 10,
    frozen_specification: ExperimentSpec | None = None,
    base_seed: int = 20260000,
) -> StagedAblationPlan:
    """Create the requested 360/30/5-10/1 stage without touching future phases."""
    if stage == "screening":
        seeds = (base_seed,)
        specs = tuple(
            spec.model_copy(update={"seeds": seeds})
            for spec in _ranked_specs("development", ranked_experiment_ids)
        )
        return StagedAblationPlan(
            stage=stage,
            phase="development",
            target_stride=5,
            seeds=seeds,
            specifications=specs,
            selection_rule="all_360_on_every_fifth_development_target",
        )
    if stage == "full_development":
        seeds = tuple(base_seed + offset for offset in range(3))
        specs = tuple(
            spec.model_copy(update={"seeds": seeds})
            for spec in _ranked_specs("development", ranked_experiment_ids)[:30]
        )
        return StagedAblationPlan(
            stage=stage,
            phase="development",
            target_stride=1,
            seeds=seeds,
            specifications=specs,
            selection_rule="top_30_from_screening_on_full_development",
        )
    if stage == "calibration":
        if not 5 <= calibration_count <= 10:
            raise ValueError("calibration_count must be between 5 and 10")
        seeds = tuple(base_seed + offset for offset in range(5))
        specs = tuple(
            spec.model_copy(update={"seeds": seeds})
            for spec in _ranked_specs("calibration", ranked_experiment_ids)[:calibration_count]
        )
        return StagedAblationPlan(
            stage=stage,
            phase="calibration",
            target_stride=1,
            seeds=seeds,
            specifications=specs,
            selection_rule=f"top_{calibration_count}_from_development_on_full_calibration",
        )
    if frozen_specification is None:
        raise ValueError("final_holdout requires one explicitly frozen specification")
    seed = frozen_specification.seeds[0]
    frozen = frozen_specification.model_copy(
        update={
            "seeds": (seed,),
            "data_split": DataSplitSpec(phase="final_holdout"),
        }
    )
    return StagedAblationPlan(
        stage=stage,
        phase="final_holdout",
        target_stride=1,
        seeds=(seed,),
        specifications=(frozen,),
        selection_rule="one_frozen_configuration_executed_once",
        final_holdout_locked=True,
    )


def select_stage_target_issues(
    draws: pd.DataFrame,
    plan: StagedAblationPlan,
    *,
    minimum_history: int = 100,
) -> tuple[str, ...]:
    """Select chronological targets only inside the plan's declared phase."""
    if minimum_history < 1:
        raise ValueError("minimum_history must be positive")
    split = split_history(draws, plan.specifications[0].data_split)
    phase = split.phase_frame(plan.phase)
    allowed = set(phase["issue"].astype(str).tolist())
    issues = draws["issue"].astype(str).tolist()
    eligible = [
        issue for index, issue in enumerate(issues) if index >= minimum_history and issue in allowed
    ]
    return tuple(eligible[:: plan.target_stride])
