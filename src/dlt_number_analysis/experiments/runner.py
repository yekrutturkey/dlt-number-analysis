"""Strict expanding-window execution for versioned experiment specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time
from os import getpid
from pathlib import Path
from time import perf_counter, process_time
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis.backtesting import (
    evaluate_prediction_raw_observation,
    run_rolling_backtest,
)
from dlt_number_analysis.data import (
    CSV_COLUMNS,
    DrawRecord,
    generate_history_integrity_report,
    validate_draw_dataframe,
)
from dlt_number_analysis.evaluation import IssuePrizeRecord, PrizeTable
from dlt_number_analysis.experiments.benchmark_checkpoints import peak_process_memory_mb
from dlt_number_analysis.experiments.shared_computation import (
    SharedComputationCache,
    build_shared_ablation_context,
    deterministic_subseed,
)
from dlt_number_analysis.experiments.specs import ExperimentSpec
from dlt_number_analysis.experiments.splits import (
    acquire_holdout_lock,
    experiment_config_sha256,
    finalize_holdout_lock,
    split_history,
)
from dlt_number_analysis.models import PredictionRecord
from dlt_number_analysis.pipeline import PROFILE_DEFAULTS, PipelineConfig, PipelineProfile
from dlt_number_analysis.portfolio import (
    CandidatePool,
    CandidateScoreView,
    FeasiblePortfolioBank,
    FeasiblePortfolioIndexBank,
    PortfolioScoringMethod,
    build_candidate_score_view,
    candidate_pool_to_array_bundle,
    candidate_score_view_from_arrays,
    feasible_portfolio_bank_to_index_bank,
    portfolio_constraints_signature,
    rescore_candidate_pool,
    rescore_candidate_pool_with_vectors,
    sample_constraint_matched_portfolio,
    score_portfolio_bank,
    score_portfolio_bank_vectorized,
)
from dlt_number_analysis.scoring import build_number_scorer, uniform_score
from dlt_number_analysis.strategies import StrategyFunction, core_rotation, max_coverage

EvaluationMode = Literal["raw_observation", "full_resampling"]


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
    candidate_cache_hit_rate: float = Field(default=0.0, ge=0, le=1)
    feasible_bank_generation_seconds: float = Field(default=0.0, ge=0)
    portfolio_bank_reuse_count: int = Field(default=0, ge=0)
    experiment_throughput_per_hour: float = Field(default=0.0, ge=0)
    estimated_remaining_runtime_seconds: float = Field(default=0.0, ge=0)
    process_count: int = Field(default=1, ge=1)
    total_cpu_time_seconds: float = Field(default=0.0, ge=0)
    wall_clock_seconds: float = Field(default=0.0, ge=0)
    candidate_array_conversion_seconds: float = Field(default=0.0, ge=0)
    portfolio_index_bank_seconds: float = Field(default=0.0, ge=0)
    vectorized_scoring_seconds: float = Field(default=0.0, ge=0)
    final_object_construction_seconds: float = Field(default=0.0, ge=0)


class TargetExecutionTiming(BaseModel):
    """Measured phase timings for one target/seed shared Runner invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    seed: int
    evaluation_mode: EvaluationMode
    history_slice_seconds: float = Field(ge=0)
    candidate_generation_seconds: float = Field(ge=0)
    candidate_array_conversion_seconds: float = Field(ge=0)
    bank_generation_seconds: float = Field(ge=0)
    portfolio_index_bank_seconds: float = Field(ge=0)
    b1_sampling_seconds: float = Field(ge=0)
    score_view_seconds: float = Field(ge=0)
    vectorized_scoring_seconds: float = Field(ge=0)
    final_object_construction_seconds: float = Field(ge=0)
    raw_evaluation_seconds: float = Field(ge=0)
    observation_serialization_seconds: float = Field(ge=0)
    checkpoint_write_seconds: float = Field(default=0.0, ge=0)
    target_total_seconds: float = Field(ge=0)
    candidate_count: int = Field(ge=1)
    bank_size: int = Field(ge=1)
    acceptance_rate: float = Field(gt=0, le=1)
    candidate_cache_hit_rate: float = Field(ge=0, le=1)
    bank_reuse_count: int = Field(ge=0)
    peak_memory_mb: float | None = Field(default=None, ge=0)
    process_id: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class ExperimentBatchResult:
    """Tidy observations plus execution-level runtime records."""

    observations: pd.DataFrame
    executions: tuple[ExperimentExecutionRecord, ...]
    target_timings: tuple[TargetExecutionTiming, ...] = ()


def build_experiment_pipeline_config(
    spec: ExperimentSpec,
    *,
    profile: PipelineProfile = "fast",
    parallel_workers: int | None = None,
) -> PipelineConfig:
    """Bind an optimized experiment spec to the exact shared prediction pipeline."""
    if spec.portfolio_strategy in {
        "uniform_random",
        "constraint_matched_random",
        "max_coverage",
        "core_rotation",
    }:
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
        "scorer_spec_json": spec.scorer_spec.model_dump_json(),
        "portfolio_strategy": spec.portfolio_strategy,
        "portfolio_constraints_json": spec.portfolio_constraints.model_dump_json(),
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


def _historical_generation_time(draw_date: object) -> datetime:
    value = pd.Timestamp(draw_date).date()
    return datetime.combine(value, time(23, 59), tzinfo=ZoneInfo("Asia/Shanghai"))


def _target_cohort_id(phase: str, target_issues: Sequence[str]) -> str:
    """Name an exact target range without merging it with a different cohort."""
    targets = tuple(str(issue) for issue in target_issues)
    if targets == tuple(f"{issue:05d}" for issue in range(8009, 8109)):
        return "v051_paired_smoke_100"
    if targets == ("08008",):
        return "single_correctness_smoke"
    if phase == "calibration":
        return "calibration"
    if phase == "final_holdout":
        return "final_holdout"
    if not targets:
        raise ValueError("experiment cohort cannot be empty")
    return f"{phase}_{targets[0]}_{targets[-1]}_{len(targets)}"


def _prediction_strategy(prediction: PredictionRecord) -> StrategyFunction:
    def strategy(**_: object) -> PredictionRecord:
        return prediction

    return strategy


def _shared_bank_experiment_batch(
    draws: pd.DataFrame,
    specifications: Sequence[ExperimentSpec],
    *,
    profile: PipelineProfile = "fast",
    minimum_history: int = 100,
    random_baseline_seed_count: int = 1000,
    bootstrap_resamples: int = 1000,
    prize_tables: Sequence[PrizeTable] | None = None,
    issue_prize_records: Mapping[str, IssuePrizeRecord] | None = None,
    holdout_lock_path: str | Path | None = None,
    parallel_workers: int | None = None,
    target_issues: Sequence[str] | None = None,
    minimum_bank_size: int = 500,
    maximum_bank_search_trials: int = 80_000,
    portfolio_scoring_method: PortfolioScoringMethod = "numpy_vectorized",
    evaluation_mode: EvaluationMode = "full_resampling",
) -> ExperimentBatchResult:
    """Run B1-B6 target-first so candidates and feasible banks are truly shared."""
    del parallel_workers
    if evaluation_mode not in {"raw_observation", "full_resampling"}:
        raise ValueError(f"unsupported evaluation mode: {evaluation_mode}")
    if not specifications:
        return ExperimentBatchResult(pd.DataFrame(), (), ())
    allowed_ids = {f"B{index}" for index in range(1, 7)}
    is_ablation = all(
        spec.experiment_version == "v0.5.1-ablation-shared-v1" for spec in specifications
    )
    if not is_ablation and any(spec.experiment_id not in allowed_ids for spec in specifications):
        raise ValueError("shared feasible-bank execution requires B1-B6 or v0.5.1 ablations")
    first = specifications[0]
    if any(spec.data_split != first.data_split for spec in specifications):
        raise ValueError("shared bank experiments must use the same temporal split")
    if any(spec.seeds != first.seeds for spec in specifications):
        raise ValueError("shared bank experiments must use the same seed set")
    if first.data_split.phase == "final_holdout":
        if evaluation_mode != "full_resampling":
            raise ValueError("formal final holdout requires full_resampling evaluation")
        if holdout_lock_path is None:
            raise ValueError("formal final holdout requires a lock path")
        for spec in specifications:
            acquire_holdout_lock(spec, holdout_lock_path)

    validated = validate_draw_dataframe(draws)
    if generate_history_integrity_report(validated).blocks_backtest:
        raise ValueError("history integrity errors block formal experiments")
    phase_start, phase_end = _phase_bounds(validated, first)
    target_start = max(minimum_history, phase_start)
    allowed_targets = None if target_issues is None else {str(issue) for issue in target_issues}
    target_indices = [
        index
        for index in range(target_start, phase_end)
        if allowed_targets is None or str(validated.iloc[index]["issue"]) in allowed_targets
    ]
    if allowed_targets is not None:
        found = {str(validated.iloc[index]["issue"]) for index in target_indices}
        missing = allowed_targets.difference(found)
        if missing:
            raise ValueError(f"requested targets are outside the declared phase: {sorted(missing)}")
    target_issue_sequence = tuple(str(validated.iloc[index]["issue"]) for index in target_indices)
    cohort_id = (
        "development_full"
        if allowed_targets is None and first.data_split.phase == "development"
        else _target_cohort_id(first.data_split.phase, target_issue_sequence)
    )

    profile_defaults = PROFILE_DEFAULTS[profile]
    objective = PipelineConfig(profile=profile)
    observations: list[dict[str, object]] = []
    execution_rows: list[ExperimentExecutionRecord] = []
    target_timing_rows: list[TargetExecutionTiming] = []
    config_hashes = {spec.experiment_id: experiment_config_sha256(spec) for spec in specifications}
    for seed in first.seeds:
        wall_started = perf_counter()
        cpu_started = process_time()
        period_counts = dict.fromkeys((spec.experiment_id for spec in specifications), 0)
        candidate_requests = 0
        candidate_hits = 0
        bank_reuse_count = 0
        bank_generation_seconds = 0.0
        candidate_array_conversion_seconds = 0.0
        portfolio_index_bank_seconds = 0.0
        vectorized_scoring_seconds = 0.0
        final_object_construction_seconds = 0.0
        for target_index in target_indices:
            target_started = perf_counter()
            history_started = perf_counter()
            history = validated.iloc[:target_index].copy()
            target_history_slice_seconds = perf_counter() - history_started
            target_issue = str(validated.iloc[target_index]["issue"])
            generated_at = _historical_generation_time(history.iloc[-1]["draw_date"])
            candidate_seed = deterministic_subseed(seed, target_issue, "candidates")
            cache = SharedComputationCache()
            ablation_context = None
            candidate_started = perf_counter()
            if is_ablation:
                ablation_context = build_shared_ablation_context(
                    history,
                    target_issue=target_issue,
                    generated_at=generated_at,
                    seed=seed,
                    cache=cache,
                    candidate_count=profile_defaults["candidate_count"],
                    bank_search_trials=profile_defaults["optimizer_search_trials"],
                    maximum_bank_size=min(2_000, profile_defaults["optimizer_search_trials"]),
                    minimum_bank_size=minimum_bank_size,
                    maximum_bank_search_trials=maximum_bank_search_trials,
                )
                base_pool = ablation_context.candidate_pool
                candidate_requests += len(specifications)
                candidate_hits += max(len(specifications) - 1, 0)
                bank_generation_seconds += ablation_context.feasible_bank_generation_seconds
            else:
                requested_pools = [
                    cache.get_candidate_pool(
                        history,
                        target_issue=target_issue,
                        generated_at=generated_at,
                        candidate_seed=candidate_seed,
                        candidate_count=profile_defaults["candidate_count"],
                    )
                    for _ in specifications
                ]
                candidate_requests += cache.candidate_requests
                candidate_hits += cache.candidate_hits
                base_pool = requested_pools[0]
            target_candidate_generation_seconds = perf_counter() - candidate_started
            array_started = perf_counter()
            candidate_arrays = candidate_pool_to_array_bundle(base_pool)
            target_array_seconds = perf_counter() - array_started
            candidate_array_conversion_seconds += target_array_seconds
            rescored: dict[tuple[str, float, float], CandidatePool] = {}
            score_views: dict[tuple[str, float, float], CandidateScoreView] = {}
            banks_by_signature: dict[str, FeasiblePortfolioBank] = {}
            index_banks_by_signature: dict[str, FeasiblePortfolioIndexBank] = {}
            predictions: dict[str, PredictionRecord] = {}
            audit_by_strategy: dict[str, dict[str, object]] = {}
            target_bank_reuse_count = 0
            target_bank_generation_seconds = 0.0
            target_index_bank_seconds = 0.0
            target_b1_sampling_seconds = 0.0
            target_score_view_seconds = 0.0
            target_vectorized_scoring_seconds = 0.0
            target_final_object_seconds = 0.0
            for spec in specifications:
                signature = portfolio_constraints_signature(spec.portfolio_constraints)
                bank_seed = deterministic_subseed(seed, target_issue, "bank", signature)
                was_cached = signature in banks_by_signature
                if ablation_context is None:
                    bank = cache.get_portfolio_bank(
                        base_pool,
                        bank_seed=bank_seed,
                        constraints=spec.portfolio_constraints,
                        search_trials=profile_defaults["optimizer_search_trials"],
                        maximum_bank_size=min(2_000, profile_defaults["optimizer_search_trials"]),
                        minimum_bank_size=minimum_bank_size,
                        maximum_search_trials=maximum_bank_search_trials,
                    )
                else:
                    bank = ablation_context.banks_by_constraints_signature[signature]
                if was_cached:
                    bank_reuse_count += 1
                    target_bank_reuse_count += 1
                else:
                    banks_by_signature[signature] = bank
                    target_bank_generation_seconds += bank.generation_seconds
                    if ablation_context is None:
                        bank_generation_seconds += bank.generation_seconds
                    index_started = perf_counter()
                    index_banks_by_signature[signature] = feasible_portfolio_bank_to_index_bank(
                        candidate_arrays, bank
                    )
                    index_seconds = perf_counter() - index_started
                    portfolio_index_bank_seconds += index_seconds
                    target_index_bank_seconds += index_seconds
                score_key = (
                    spec.scorer_spec.model_dump_json(),
                    spec.number_score_weight,
                    spec.structure_score_weight,
                )
                if (
                    spec.portfolio_strategy != "constraint_matched_random"
                    and portfolio_scoring_method == "object_reference"
                    and score_key not in rescored
                ):
                    if ablation_context is None:
                        rescored[score_key] = rescore_candidate_pool(
                            history,
                            base_pool,
                            scorer_spec=spec.scorer_spec,
                            number_score_weight=spec.number_score_weight,
                            structure_score_weight=spec.structure_score_weight,
                        )
                    else:
                        scorer_index = next(
                            index
                            for index, scorer_spec in enumerate(
                                ablation_context.score_cube.scorer_specs
                            )
                            if scorer_spec == spec.scorer_spec
                        )
                        structure_index = ablation_context.score_cube.structure_weights.index(
                            spec.structure_score_weight
                        )
                        structure_weight = ablation_context.score_cube.structure_weights[
                            structure_index
                        ]
                        rescored[score_key] = rescore_candidate_pool_with_vectors(
                            base_pool,
                            scorer_spec=ablation_context.score_cube.scorer_specs[scorer_index],
                            front_candidate_scores=ablation_context.score_cube.front_candidate_scores[
                                scorer_index
                            ],
                            back_candidate_scores=ablation_context.score_cube.back_candidate_scores[
                                scorer_index
                            ],
                            number_candidate_scores=ablation_context.score_cube.number_scores[
                                scorer_index
                            ],
                            combined_candidate_scores=ablation_context.score_cube.combined_scores[
                                scorer_index, structure_index
                            ],
                            number_score_weight=1.0 - structure_weight,
                            structure_score_weight=structure_weight,
                            scoring_method="shared_15x4_numpy_ablation_score_cube",
                        )
                if (
                    spec.portfolio_strategy != "constraint_matched_random"
                    and portfolio_scoring_method == "numpy_vectorized"
                    and score_key not in score_views
                ):
                    score_view_started = perf_counter()
                    if ablation_context is None:
                        score_views[score_key] = build_candidate_score_view(
                            history,
                            candidate_arrays,
                            scorer_spec=spec.scorer_spec,
                            number_score_weight=spec.number_score_weight,
                            structure_score_weight=spec.structure_score_weight,
                        )
                    else:
                        scorer_index = next(
                            index
                            for index, scorer_spec in enumerate(
                                ablation_context.score_cube.scorer_specs
                            )
                            if scorer_spec == spec.scorer_spec
                        )
                        structure_index = ablation_context.score_cube.structure_weights.index(
                            spec.structure_score_weight
                        )
                        score_views[score_key] = candidate_score_view_from_arrays(
                            candidate_arrays,
                            number_scores=ablation_context.score_cube.number_scores[scorer_index],
                            combined_scores=ablation_context.score_cube.combined_scores[
                                scorer_index, structure_index
                            ],
                            scorer_spec=spec.scorer_spec,
                            number_score_weight=spec.number_score_weight,
                            structure_score_weight=spec.structure_score_weight,
                            scoring_method="shared_15x4_numpy_ablation_score_cube_view",
                        )
                    target_score_view_seconds += perf_counter() - score_view_started
                selection_seed = deterministic_subseed(
                    seed, target_issue, spec.experiment_id, "selection"
                )
                if spec.portfolio_strategy == "constraint_matched_random":
                    b1_started = perf_counter()
                    selection = sample_constraint_matched_portfolio(
                        base_pool,
                        bank,
                        random_seed=selection_seed,
                    )
                    target_b1_sampling_seconds += perf_counter() - b1_started
                    active_scoring_method = "random_bank_sample"
                elif portfolio_scoring_method == "numpy_vectorized":
                    vectorized_result = score_portfolio_bank_vectorized(
                        score_views[score_key],
                        index_banks_by_signature[signature],
                        random_seed=selection_seed,
                        single_ticket_weight=objective.single_ticket_weight,
                        diversity_weight=objective.diversity_weight,
                        core_weight=objective.core_weight,
                        structure_weight=objective.structure_weight,
                        repeat_penalty_weight=objective.repeat_penalty_weight,
                    )
                    selection = vectorized_result.selection
                    vectorized_scoring_seconds += vectorized_result.vectorized_scoring_seconds
                    target_vectorized_scoring_seconds += (
                        vectorized_result.vectorized_scoring_seconds
                    )
                    final_object_construction_seconds += (
                        vectorized_result.final_object_construction_seconds
                    )
                    target_final_object_seconds += (
                        vectorized_result.final_object_construction_seconds
                    )
                    active_scoring_method = "numpy_vectorized"
                else:
                    selection = score_portfolio_bank(
                        rescored[score_key],
                        bank,
                        random_seed=selection_seed,
                        single_ticket_weight=objective.single_ticket_weight,
                        diversity_weight=objective.diversity_weight,
                        core_weight=objective.core_weight,
                        structure_weight=objective.structure_weight,
                        repeat_penalty_weight=objective.repeat_penalty_weight,
                    ).selection
                    active_scoring_method = "object_reference"
                strategy_name = f"experiment_{spec.experiment_id}"
                base_prediction = selection.to_prediction_record()
                prediction = base_prediction.model_copy(
                    update={
                        "strategy_name": strategy_name,
                        "model_version": "shared-feasible-bank-experiment-v0.5.2",
                        "parameters": {
                            **base_prediction.parameters,
                            "experiment_config_sha256": config_hashes[spec.experiment_id],
                            "evaluation_mode": evaluation_mode,
                            "scorer_spec": spec.scorer_spec.model_dump(mode="json"),
                            "number_score_weight": spec.number_score_weight,
                            "structure_score_weight": spec.structure_score_weight,
                            "portfolio_scoring_method": active_scoring_method,
                            "candidate_seed": candidate_seed,
                            "bank_seed": bank.bank_seed,
                            "bank_size": bank.bank_size,
                            "bank_acceptance_rate": bank.acceptance_rate,
                            "bank_initial_search_trials": bank.initial_search_trials,
                            "bank_search_trials": bank.search_trials,
                            "bank_search_expansion_count": bank.search_expansion_count,
                            "bank_minimum_size": bank.minimum_bank_size,
                            "bank_hash": bank.bank_hash,
                            "constraints_signature": bank.constraints_signature,
                            "candidate_numbers_hash": bank.candidate_numbers_hash,
                        },
                    }
                )
                predictions[strategy_name] = PredictionRecord.model_validate(
                    prediction.model_dump()
                )
                audit_by_strategy[strategy_name] = {
                    "candidate_seed": candidate_seed,
                    "selection_seed": selection_seed,
                    "bank_seed": bank.bank_seed,
                    "bank_size": bank.bank_size,
                    "bank_acceptance_rate": bank.acceptance_rate,
                    "bank_initial_search_trials": bank.initial_search_trials,
                    "bank_search_trials": bank.search_trials,
                    "bank_search_expansion_count": bank.search_expansion_count,
                    "bank_minimum_size": bank.minimum_bank_size,
                    "bank_hash": bank.bank_hash,
                    "candidate_numbers_hash": bank.candidate_numbers_hash,
                    "constraints_signature": bank.constraints_signature,
                    "portfolio_scoring_method": active_scoring_method,
                }
            target_bank_generation_count = len(banks_by_signature)
            for audit in audit_by_strategy.values():
                audit.update(
                    {
                        "target_candidate_generation_count": len(cache.candidate_pools),
                        "target_candidate_reuse_count": cache.candidate_hits,
                        "target_bank_generation_count": target_bank_generation_count,
                        "target_portfolio_bank_reuse_count": target_bank_reuse_count,
                        "evaluation_mode": evaluation_mode,
                    }
                )
            raw_evaluation_seconds = 0.0
            if evaluation_mode == "full_resampling":
                strategies = {
                    name: _prediction_strategy(prediction)
                    for name, prediction in predictions.items()
                }
                prefix = validated.iloc[: target_index + 1].copy()
                report = run_rolling_backtest(
                    prefix,
                    strategies=strategies,
                    min_history=target_index,
                    base_random_seed=seed,
                    prize_tables=prize_tables,
                    issue_prize_records=issue_prize_records,
                    number_scorer=uniform_score,
                    number_scorer_name="uniform_score_for_precomputed_bank_predictions",
                    random_baseline_seed_count=random_baseline_seed_count,
                    bootstrap_resamples=bootstrap_resamples,
                    allow_short_history=minimum_history < 100,
                )
                results_by_name = {result.strategy_name: result for result in report.results}
            else:
                target_row = validated.iloc[target_index]
                actual_draw = DrawRecord.model_validate(
                    {column: target_row[column] for column in CSV_COLUMNS}
                )
                raw_started = perf_counter()
                results_by_name = {
                    strategy_name: evaluate_prediction_raw_observation(
                        prediction,
                        actual_draw,
                        prize_tables=prize_tables,
                        issue_prize_records=issue_prize_records,
                    )
                    for strategy_name, prediction in predictions.items()
                }
                raw_evaluation_seconds = perf_counter() - raw_started
            serialization_started = perf_counter()
            for spec in specifications:
                strategy_name = f"experiment_{spec.experiment_id}"
                result = results_by_name[strategy_name]
                period_counts[spec.experiment_id] += 1
                observations.append(
                    {
                        "experiment_id": spec.experiment_id,
                        "experiment_version": spec.experiment_version,
                        "experiment_config_sha256": config_hashes[spec.experiment_id],
                        "phase": spec.data_split.phase,
                        "evaluation_mode": evaluation_mode,
                        "profile": profile,
                        "cohort_id": cohort_id,
                        "seed": seed,
                        "prediction_random_seed": predictions[strategy_name].random_seed,
                        "prediction_model_version": predictions[strategy_name].model_version,
                        "candidate_count": len(base_pool.candidates),
                        "draw_date": validated.iloc[target_index]["draw_date"],
                        **_parameter_columns(spec),
                        **audit_by_strategy[strategy_name],
                        **result.model_dump(mode="python"),
                    }
                )
            serialization_seconds = perf_counter() - serialization_started
            bank_values = tuple(banks_by_signature.values())
            peak_memory, _ = peak_process_memory_mb()
            target_timing_rows.append(
                TargetExecutionTiming(
                    target_issue=target_issue,
                    seed=seed,
                    evaluation_mode=evaluation_mode,
                    history_slice_seconds=target_history_slice_seconds,
                    candidate_generation_seconds=target_candidate_generation_seconds,
                    candidate_array_conversion_seconds=target_array_seconds,
                    bank_generation_seconds=target_bank_generation_seconds,
                    portfolio_index_bank_seconds=target_index_bank_seconds,
                    b1_sampling_seconds=target_b1_sampling_seconds,
                    score_view_seconds=target_score_view_seconds,
                    vectorized_scoring_seconds=target_vectorized_scoring_seconds,
                    final_object_construction_seconds=target_final_object_seconds,
                    raw_evaluation_seconds=raw_evaluation_seconds,
                    observation_serialization_seconds=serialization_seconds,
                    target_total_seconds=perf_counter() - target_started,
                    candidate_count=len(base_pool.candidates),
                    bank_size=min(bank.bank_size for bank in bank_values),
                    acceptance_rate=min(bank.acceptance_rate for bank in bank_values),
                    candidate_cache_hit_rate=cache.candidate_cache_hit_rate,
                    bank_reuse_count=target_bank_reuse_count,
                    peak_memory_mb=peak_memory,
                    process_id=getpid(),
                )
            )
        wall_seconds = perf_counter() - wall_started
        cpu_seconds = process_time() - cpu_started
        cache_hit_rate = 0.0 if candidate_requests == 0 else candidate_hits / candidate_requests
        for spec in specifications:
            period_count = period_counts[spec.experiment_id]
            execution_rows.append(
                ExperimentExecutionRecord(
                    experiment_id=spec.experiment_id,
                    experiment_version=spec.experiment_version,
                    experiment_config_sha256=config_hashes[spec.experiment_id],
                    phase=spec.data_split.phase,
                    seed=seed,
                    period_count=period_count,
                    runtime_seconds=wall_seconds,
                    candidate_cache_hit_rate=cache_hit_rate,
                    feasible_bank_generation_seconds=bank_generation_seconds,
                    portfolio_bank_reuse_count=bank_reuse_count,
                    experiment_throughput_per_hour=(
                        0.0 if wall_seconds == 0 else period_count / wall_seconds * 3600
                    ),
                    process_count=1,
                    total_cpu_time_seconds=cpu_seconds,
                    wall_clock_seconds=wall_seconds,
                    candidate_array_conversion_seconds=candidate_array_conversion_seconds,
                    portfolio_index_bank_seconds=portfolio_index_bank_seconds,
                    vectorized_scoring_seconds=vectorized_scoring_seconds,
                    final_object_construction_seconds=final_object_construction_seconds,
                )
            )
    frame = pd.DataFrame.from_records(observations)
    if first.data_split.phase == "final_holdout":
        assert holdout_lock_path is not None
        for spec in specifications:
            result_bytes = (
                frame.loc[frame["experiment_id"] == spec.experiment_id]
                .to_csv(index=False, lineterminator="\n")
                .encode("utf-8")
            )
            finalize_holdout_lock(
                holdout_lock_path,
                experiment_version=spec.experiment_version,
                result_bytes=result_bytes,
            )
    return ExperimentBatchResult(frame, tuple(execution_rows), tuple(target_timing_rows))


def _run_legacy_experiment(
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
    if generate_history_integrity_report(validated).blocks_backtest:
        raise ValueError("history integrity errors block formal experiments")
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
    target_issues: Sequence[str] | None = None,
    minimum_bank_size: int = 500,
    maximum_bank_search_trials: int = 80_000,
    portfolio_scoring_method: PortfolioScoringMethod = "numpy_vectorized",
    evaluation_mode: EvaluationMode = "full_resampling",
) -> ExperimentBatchResult:
    """Run one experiment, routing B1-B6 through the shared feasible-bank path."""
    options = {
        "profile": profile,
        "minimum_history": minimum_history,
        "random_baseline_seed_count": random_baseline_seed_count,
        "bootstrap_resamples": bootstrap_resamples,
        "prize_tables": prize_tables,
        "issue_prize_records": issue_prize_records,
        "holdout_lock_path": holdout_lock_path,
        "parallel_workers": parallel_workers,
    }
    if spec.experiment_id in {f"B{index}" for index in range(1, 7)} or (
        spec.experiment_version == "v0.5.1-ablation-shared-v1"
    ):
        return _shared_bank_experiment_batch(
            draws,
            (spec,),
            target_issues=target_issues,
            minimum_bank_size=minimum_bank_size,
            maximum_bank_search_trials=maximum_bank_search_trials,
            portfolio_scoring_method=portfolio_scoring_method,
            evaluation_mode=evaluation_mode,
            **options,
        )
    if evaluation_mode != "full_resampling":
        raise ValueError("raw_observation evaluation is supported only by shared B1-B6")
    if target_issues is not None:
        raise ValueError("target_issues filtering is currently restricted to B1-B6")
    return _run_legacy_experiment(draws, spec, **options)


def run_experiment_batch(
    draws: pd.DataFrame,
    specifications: Sequence[ExperimentSpec],
    **run_options: object,
) -> ExperimentBatchResult:
    """Run multiple specs and combine tidy results without changing their configs."""
    observations: list[pd.DataFrame] = []
    executions: list[ExperimentExecutionRecord] = []
    target_timings: list[TargetExecutionTiming] = []
    shared_ids = {f"B{index}" for index in range(1, 7)}
    shared = [spec for spec in specifications if spec.experiment_id in shared_ids]
    ablations = [
        spec for spec in specifications if spec.experiment_version == "v0.5.1-ablation-shared-v1"
    ]
    legacy = [
        spec
        for spec in specifications
        if spec.experiment_id not in shared_ids
        and spec.experiment_version != "v0.5.1-ablation-shared-v1"
    ]
    for shared_group in (shared, ablations):
        if not shared_group:
            continue
        result = _shared_bank_experiment_batch(draws, shared_group, **run_options)
        observations.append(result.observations)
        executions.extend(result.executions)
        target_timings.extend(result.target_timings)
    for spec in legacy:
        result = run_experiment(draws, spec, **run_options)
        observations.append(result.observations)
        executions.extend(result.executions)
        target_timings.extend(result.target_timings)
    combined = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame()
    return ExperimentBatchResult(
        observations=combined,
        executions=tuple(executions),
        target_timings=tuple(target_timings),
    )
