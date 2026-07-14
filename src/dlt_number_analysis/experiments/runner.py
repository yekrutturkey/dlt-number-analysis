"""Strict expanding-window execution for versioned experiment specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis.backtesting import run_rolling_backtest
from dlt_number_analysis.data import validate_draw_dataframe
from dlt_number_analysis.evaluation import IssuePrizeRecord, PrizeTable
from dlt_number_analysis.experiments.specs import ExperimentSpec
from dlt_number_analysis.experiments.splits import (
    acquire_holdout_lock,
    experiment_config_sha256,
    finalize_holdout_lock,
    split_history,
)
from dlt_number_analysis.pipeline import PipelineConfig, PipelineProfile
from dlt_number_analysis.scoring import build_number_scorer
from dlt_number_analysis.strategies import core_rotation, max_coverage


class ExperimentExecutionRecord(BaseModel):
    """Runtime and observation count for one experiment seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    experiment_version: str
    experiment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase: str
    seed: int
    period_count: int = Field(ge=0)
    runtime_seconds: float = Field(ge=0)


@dataclass(frozen=True, slots=True)
class ExperimentBatchResult:
    """Tidy observations plus execution-level runtime records."""

    observations: pd.DataFrame
    executions: tuple[ExperimentExecutionRecord, ...]


def build_experiment_pipeline_config(
    spec: ExperimentSpec,
    *,
    profile: PipelineProfile = "fast",
    parallel_workers: int | None = None,
) -> PipelineConfig:
    """Bind an optimized experiment spec to the exact shared prediction pipeline."""
    if spec.portfolio_strategy in {"uniform_random", "max_coverage", "core_rotation"}:
        raise ValueError("this experiment strategy does not use the optimized pipeline")
    values: dict[str, object] = {
        "profile": profile,
        "scorer_spec": spec.scorer_spec,
        "structure_score_weight": spec.structure_score_weight,
        "number_score_weight": spec.number_score_weight,
        "portfolio_constraints": spec.portfolio_constraints,
        "model_version": "prediction-pipeline-v0.5-experiment",
    }
    if parallel_workers is not None:
        values["parallel_workers"] = parallel_workers
    if spec.portfolio_strategy == "constraint_matched_random":
        values.update(
            {
                "single_ticket_weight": 1.0,
                "diversity_weight": 0.0,
                "core_weight": 0.0,
                "structure_weight": 0.0,
                "repeat_penalty_weight": 0.0,
            }
        )
    return PipelineConfig(**values)


def _phase_bounds(draws: pd.DataFrame, spec: ExperimentSpec) -> tuple[int, int]:
    split = split_history(draws, spec.data_split)
    phase = split.phase_frame(spec.data_split.phase)
    issue_to_index = {
        str(issue): index for index, issue in enumerate(draws["issue"].astype(str).tolist())
    }
    start = issue_to_index[str(phase.iloc[0]["issue"])]
    end = issue_to_index[str(phase.iloc[-1]["issue"])] + 1
    return start, end


def _result_strategy_name(spec: ExperimentSpec) -> str:
    if spec.portfolio_strategy == "uniform_random":
        return "random_baseline"
    if spec.portfolio_strategy in {"max_coverage", "core_rotation"}:
        return spec.portfolio_strategy
    return "optimized_portfolio_strategy"


def _parameter_columns(spec: ExperimentSpec) -> dict[str, object]:
    parameters = spec.scorer_spec.parameters
    return {
        "scorer_name": spec.scorer_spec.name,
        "portfolio_strategy": spec.portfolio_strategy,
        "candidate_generation_method": spec.candidate_generation_method,
        "window": parameters.get("window"),
        "decay": parameters.get("decay"),
        "hot_window": parameters.get("hot_window"),
        "cold_window": parameters.get("cold_window"),
        "hot_weight": parameters.get("hot_weight"),
        "number_score_weight": spec.number_score_weight,
        "structure_score_weight": spec.structure_score_weight,
        "target_front_pool_size": spec.portfolio_constraints.target_front_pool_size,
        "core_number_count": spec.portfolio_constraints.min_core_front_numbers,
    }


def run_experiment(
    draws: pd.DataFrame,
    spec: ExperimentSpec,
    *,
    profile: PipelineProfile = "fast",
    minimum_history: int = 100,
    random_baseline_seed_count: int = 1000,
    bootstrap_resamples: int = 1000,
    prize_tables: Sequence[PrizeTable] | None = None,
    issue_prize_records: Mapping[str, IssuePrizeRecord] | None = None,
    holdout_lock_path: str | Path | None = None,
    parallel_workers: int | None = None,
) -> ExperimentBatchResult:
    """Run one spec on only its declared contiguous phase using pre-target history."""
    validated = validate_draw_dataframe(draws)
    phase_start, phase_end = _phase_bounds(validated, spec)
    if spec.data_split.phase == "final_holdout":
        if holdout_lock_path is None:
            raise ValueError("formal final holdout requires a lock path")
        acquire_holdout_lock(spec, holdout_lock_path)

    target_start = max(minimum_history, phase_start)
    prefix = validated.iloc[:phase_end].copy()
    date_by_issue = {
        str(row["issue"]): row["draw_date"] for row in validated.to_dict(orient="records")
    }
    observations: list[dict[str, object]] = []
    executions: list[ExperimentExecutionRecord] = []
    result_name = _result_strategy_name(spec)
    config_hash = experiment_config_sha256(spec)

    for seed in spec.seeds:
        strategies = None
        pipelines = None
        scorer = build_number_scorer(spec.scorer_spec)
        if spec.portfolio_strategy == "max_coverage":
            strategies = {"max_coverage": max_coverage}
        elif spec.portfolio_strategy == "core_rotation":
            strategies = {"core_rotation": core_rotation}
        elif spec.portfolio_strategy != "uniform_random":
            pipelines = {
                "optimized_portfolio_strategy": build_experiment_pipeline_config(
                    spec,
                    profile=profile,
                    parallel_workers=parallel_workers,
                )
            }
        started = perf_counter()
        report = run_rolling_backtest(
            prefix,
            strategies=strategies,
            pipeline_configs=pipelines,
            min_history=target_start,
            base_random_seed=seed,
            prize_tables=prize_tables,
            issue_prize_records=issue_prize_records,
            number_scorer=scorer,
            number_scorer_name=spec.scorer_spec.name,
            random_baseline_seed_count=random_baseline_seed_count,
            bootstrap_resamples=bootstrap_resamples,
            allow_short_history=minimum_history < 100,
        )
        runtime_seconds = perf_counter() - started
        selected = [result for result in report.results if result.strategy_name == result_name]
        for result in selected:
            observations.append(
                {
                    "experiment_id": spec.experiment_id,
                    "experiment_version": spec.experiment_version,
                    "experiment_config_sha256": config_hash,
                    "phase": spec.data_split.phase,
                    "seed": seed,
                    "prediction_random_seed": result.random_seed,
                    "draw_date": date_by_issue[result.target_issue],
                    **_parameter_columns(spec),
                    **result.model_dump(mode="python"),
                }
            )
        executions.append(
            ExperimentExecutionRecord(
                experiment_id=spec.experiment_id,
                experiment_version=spec.experiment_version,
                experiment_config_sha256=config_hash,
                phase=spec.data_split.phase,
                seed=seed,
                period_count=len(selected),
                runtime_seconds=runtime_seconds,
            )
        )

    frame = pd.DataFrame.from_records(observations)
    batch = ExperimentBatchResult(observations=frame, executions=tuple(executions))
    if spec.data_split.phase == "final_holdout":
        assert holdout_lock_path is not None
        result_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        finalize_holdout_lock(
            holdout_lock_path,
            experiment_version=spec.experiment_version,
            result_bytes=result_bytes,
        )
    return batch


def run_experiment_batch(
    draws: pd.DataFrame,
    specifications: Sequence[ExperimentSpec],
    **run_options: object,
) -> ExperimentBatchResult:
    """Run multiple specs and combine tidy results without changing their configs."""
    observations: list[pd.DataFrame] = []
    executions: list[ExperimentExecutionRecord] = []
    for spec in specifications:
        result = run_experiment(draws, spec, **run_options)
        observations.append(result.observations)
        executions.extend(result.executions)
    combined = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame()
    return ExperimentBatchResult(observations=combined, executions=tuple(executions))
