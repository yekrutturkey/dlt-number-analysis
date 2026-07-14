"""Strict expanding-window execution for versioned experiment specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from time import perf_counter, process_time
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis.backtesting import run_rolling_backtest
from dlt_number_analysis.data import generate_history_integrity_report, validate_draw_dataframe
from dlt_number_analysis.evaluation import IssuePrizeRecord, PrizeTable
from dlt_number_analysis.experiments.shared_computation import (
    SharedComputationCache,
    build_shared_ablation_context,
    deterministic_subseed,
    materialize_ablation_candidate_pool,
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
    FeasiblePortfolioBank,
    portfolio_constraints_signature,
    rescore_candidate_pool,
    sample_constraint_matched_portfolio,
    score_portfolio_bank,
)
from dlt_number_analysis.scoring import build_number_scorer, uniform_score
from dlt_number_analysis.strategies import StrategyFunction, core_rotation, max_coverage


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


def _historical_generation_time(draw_date: object) -> datetime:
    value = pd.Timestamp(draw_date).date()
    return datetime.combine(value, time(23, 59), tzinfo=ZoneInfo("Asia/Shanghai"))


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
) -> ExperimentBatchResult:
    """Run B1-B6 target-first so candidates and feasible banks are truly shared."""
    del parallel_workers
    if not specifications:
        return ExperimentBatchResult(pd.DataFrame(), ())
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

    profile_defaults = PROFILE_DEFAULTS[profile]
    objective = PipelineConfig(profile=profile)
    observations: list[dict[str, object]] = []
    execution_rows: list[ExperimentExecutionRecord] = []
    config_hashes = {spec.experiment_id: experiment_config_sha256(spec) for spec in specifications}
    for seed in first.seeds:
        wall_started = perf_counter()
        cpu_started = process_time()
        period_counts = dict.fromkeys((spec.experiment_id for spec in specifications), 0)
        candidate_requests = 0
        candidate_hits = 0
        bank_reuse_count = 0
        bank_generation_seconds = 0.0
        for target_index in target_indices:
            history = validated.iloc[:target_index].copy()
            target_issue = str(validated.iloc[target_index]["issue"])
            generated_at = _historical_generation_time(history.iloc[-1]["draw_date"])
            candidate_seed = deterministic_subseed(seed, target_issue, "candidates")
            cache = SharedComputationCache()
            ablation_context = None
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
            rescored: dict[tuple[str, float, float], CandidatePool] = {}
            banks_by_signature: dict[str, FeasiblePortfolioBank] = {}
            predictions: dict[str, PredictionRecord] = {}
            audit_by_strategy: dict[str, dict[str, object]] = {}
            target_bank_reuse_count = 0
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
                    if ablation_context is None:
                        bank_generation_seconds += bank.generation_seconds
                score_key = (
                    spec.scorer_spec.model_dump_json(),
                    spec.number_score_weight,
                    spec.structure_score_weight,
                )
                if score_key not in rescored:
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
                        rescored[score_key] = materialize_ablation_candidate_pool(
                            base_pool,
                            ablation_context.score_cube,
                            scorer_index=scorer_index,
                            structure_weight_index=structure_index,
                        )
                scored_pool = rescored[score_key]
                selection_seed = deterministic_subseed(
                    seed, target_issue, spec.experiment_id, "selection"
                )
                if spec.portfolio_strategy == "constraint_matched_random":
                    selection = sample_constraint_matched_portfolio(
                        scored_pool,
                        bank,
                        random_seed=selection_seed,
                    )
                else:
                    selection = score_portfolio_bank(
                        scored_pool,
                        bank,
                        random_seed=selection_seed,
                        single_ticket_weight=objective.single_ticket_weight,
                        diversity_weight=objective.diversity_weight,
                        core_weight=objective.core_weight,
                        structure_weight=objective.structure_weight,
                        repeat_penalty_weight=objective.repeat_penalty_weight,
                    ).selection
                strategy_name = f"experiment_{spec.experiment_id}"
                base_prediction = selection.to_prediction_record()
                prediction = base_prediction.model_copy(
                    update={
                        "strategy_name": strategy_name,
                        "model_version": "shared-feasible-bank-experiment-v0.5.1",
                        "parameters": {
                            **base_prediction.parameters,
                            "experiment_config_sha256": config_hashes[spec.experiment_id],
                            "scorer_spec": spec.scorer_spec.model_dump(mode="json"),
                            "number_score_weight": spec.number_score_weight,
                            "structure_score_weight": spec.structure_score_weight,
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
                }
            target_bank_generation_count = len(banks_by_signature)
            for audit in audit_by_strategy.values():
                audit.update(
                    {
                        "target_candidate_generation_count": len(cache.candidate_pools),
                        "target_candidate_reuse_count": cache.candidate_hits,
                        "target_bank_generation_count": target_bank_generation_count,
                        "target_portfolio_bank_reuse_count": target_bank_reuse_count,
                    }
                )
            strategies = {
                name: _prediction_strategy(prediction) for name, prediction in predictions.items()
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
                        "seed": seed,
                        "prediction_random_seed": predictions[strategy_name].random_seed,
                        "draw_date": validated.iloc[target_index]["draw_date"],
                        **_parameter_columns(spec),
                        **audit_by_strategy[strategy_name],
                        **result.model_dump(mode="python"),
                    }
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
    return ExperimentBatchResult(frame, tuple(execution_rows))


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
            **options,
        )
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
    for spec in legacy:
        result = run_experiment(draws, spec, **run_options)
        observations.append(result.observations)
        executions.extend(result.executions)
    combined = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame()
    return ExperimentBatchResult(observations=combined, executions=tuple(executions))
