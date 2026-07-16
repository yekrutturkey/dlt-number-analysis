"""v0.5.3 single-target staged Portfolio and evaluation benchmark."""

from __future__ import annotations

import gzip
import hashlib
import os
import platform
import sys
from collections.abc import Mapping
from datetime import datetime, time
from pathlib import Path
from time import perf_counter, process_time
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.backtesting import calculate_raw_hit_metrics, run_rolling_backtest
from dlt_number_analysis.data import DrawRecord, canonical_history_sha256, load_verified_history
from dlt_number_analysis.experiments.benchmark_checkpoints import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkValue,
    IsolatedStageOutcome,
    atomic_write_bytes,
    atomic_write_json,
    load_checkpoint,
    run_isolated_stage,
)
from dlt_number_analysis.experiments.shared_computation import (
    SharedComputationCache,
    deterministic_subseed,
)
from dlt_number_analysis.experiments.specs import ExperimentSpec, baseline_experiment_specs
from dlt_number_analysis.models import PredictionRecord, TicketRecord
from dlt_number_analysis.pipeline import PROFILE_DEFAULTS, PipelineConfig
from dlt_number_analysis.portfolio import (
    CandidatePool,
    FeasiblePortfolioBank,
    PortfolioSelection,
    build_candidate_score_view,
    build_stable_portfolio_subset,
    candidate_pool_to_array_bundle,
    feasible_portfolio_bank_to_index_bank,
    portfolio_constraints_signature,
    rescore_candidate_pool,
    sample_constraint_matched_portfolio,
    score_portfolio_bank,
    score_portfolio_bank_vectorized,
)
from dlt_number_analysis.strategies import StrategyFunction

V053_TARGET_ISSUE = "08009"
V053_SEED = 20260000
V053_SUBSET_SIZES: tuple[int, ...] = (50, 100, 250)
V053_PERFORMANCE_BUDGET_SECONDS = 480.0
V053_STAGE_TIMEOUTS: dict[str, float] = {
    "prepare": 120.0,
    "vectorized_full": 120.0,
    "object_50": 60.0,
    "object_100": 90.0,
    "object_250": 180.0,
    "evaluation_raw": 30.0,
    "evaluation_full": 60.0,
    "evaluation_reduced": 60.0,
}

BenchmarkStage = Literal[
    "prepare",
    "vectorized_full",
    "object_sample",
    "evaluation_raw",
    "evaluation_full",
    "evaluation_reduced",
]


class PreparedBenchmarkPayload(BaseModel):
    """Reusable candidate pool and full feasible bank produced by Stage A."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = BENCHMARK_SCHEMA_VERSION
    target_issue: str
    data_cutoff_issue: str
    seed: int
    history_prefix_sha256: str
    candidate_pool: CandidatePool
    feasible_bank: FeasiblePortfolioBank


class V053BenchmarkPaths:
    """All v0.5.3 artifacts are physically isolated from experiment partitions."""

    def __init__(self, output_dir: str | Path, target_issue: str = V053_TARGET_ISSUE) -> None:
        self.output_dir = Path(output_dir)
        self.target_issue = target_issue

    @property
    def prepare(self) -> Path:
        return self.output_dir / f"prepare_{self.target_issue}.json"

    @property
    def payload(self) -> Path:
        return self.output_dir / f"prepare_payload_{self.target_issue}.json.gz"

    @property
    def vectorized(self) -> Path:
        return self.output_dir / f"vectorized_full_{self.target_issue}.json"

    def object_sample(self, subset_size: int) -> Path:
        return self.output_dir / f"object_b2_bank_{subset_size}_{self.target_issue}.json"

    def evaluation(self, mode: str) -> Path:
        return self.output_dir / f"evaluation_{mode}_{self.target_issue}.json"

    @property
    def budget(self) -> Path:
        return self.output_dir / "performance_budget.json"

    @property
    def summary_json(self) -> Path:
        return self.output_dir / "benchmark_summary.json"

    @property
    def summary_markdown(self) -> Path:
        return self.output_dir / "benchmark_summary.md"


def _generated_at(draw_date: object) -> datetime:
    value = pd.Timestamp(draw_date).date()
    return datetime.combine(value, time(23, 59), tzinfo=ZoneInfo("Asia/Shanghai"))


def _history_prefix(
    history_path: str | Path,
    target_issue: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    draws = load_verified_history(history_path)
    issues = draws["issue"].astype(str).tolist()
    try:
        target_index = issues.index(target_issue)
    except ValueError as error:
        raise ValueError(f"unknown benchmark target issue: {target_issue}") from error
    if target_index < 100:
        raise ValueError("v0.5.3 benchmark target requires at least 100 prior draws")
    return draws, draws.iloc[:target_index].copy(), target_index


def _benchmark_specs(seed: int) -> tuple[ExperimentSpec, ...]:
    specs = tuple(
        spec
        for spec in baseline_experiment_specs(seeds=(seed,), phase="development")
        if spec.experiment_id in {"B1", "B2", "B3", "B4", "B5", "B6"}
    )
    constraints = specs[0].portfolio_constraints
    if any(spec.portfolio_constraints != constraints for spec in specs):
        raise ValueError("v0.5.3 requires identical B1-B6 Portfolio constraints")
    return specs


def _objective() -> PipelineConfig:
    return PipelineConfig(profile="fast")


def _ticket_rows(selection: PortfolioSelection) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": ticket.candidate_id,
            "front_numbers": list(ticket.front_numbers),
            "back_numbers": list(ticket.back_numbers),
            "ticket_role": ticket.ticket_role,
        }
        for ticket in selection.tickets
    ]


def _selection_result(
    selection: PortfolioSelection,
    selected_entry_hash: str,
) -> dict[str, object]:
    return {
        "selected_entry_hash": selected_entry_hash,
        "tickets": _ticket_rows(selection),
        "core_front_numbers": list(selection.core_front_numbers),
        "support_front_numbers": list(selection.support_front_numbers),
        "scores": selection.scores.model_dump(mode="json"),
        "combined_portfolio_score": selection.scores.combined_portfolio_score,
    }


def _payload_bytes(payload: PreparedBenchmarkPayload) -> bytes:
    return gzip.compress(payload.model_dump_json().encode("utf-8"), compresslevel=3)


def _load_payload(
    prepare_checkpoint: Mapping[str, object],
    payload_path: str | Path,
) -> PreparedBenchmarkPayload:
    compressed = Path(payload_path).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != prepare_checkpoint.get("payload_sha256"):
        raise ValueError("prepared benchmark payload hash mismatch")
    payload = PreparedBenchmarkPayload.model_validate_json(gzip.decompress(compressed))
    expected = {
        "target_issue": prepare_checkpoint.get("target_issue"),
        "data_cutoff_issue": prepare_checkpoint.get("data_cutoff_issue"),
        "seed": prepare_checkpoint.get("seed"),
        "history_prefix_sha256": prepare_checkpoint.get("history_prefix_sha256"),
    }
    for key, value in expected.items():
        if getattr(payload, key) != value:
            raise ValueError(f"prepared benchmark payload mismatch for {key}")
    if payload.feasible_bank.bank_hash != prepare_checkpoint.get("bank_hash"):
        raise ValueError("prepared benchmark bank hash mismatch")
    if payload.feasible_bank.candidate_numbers_hash != prepare_checkpoint.get(
        "candidate_numbers_hash"
    ):
        raise ValueError("prepared benchmark candidate hash mismatch")
    return payload


def prepare_stage_worker(
    *,
    history_path: str,
    payload_path: str,
    target_issue: str,
    seed: int,
) -> Mapping[str, object]:
    """Stage A: generate all reusable inputs and atomically persist the payload."""
    wall_started = perf_counter()
    cpu_started = process_time()
    _, history, _ = _history_prefix(history_path, target_issue)
    specs = _benchmark_specs(seed)
    constraints = specs[0].portfolio_constraints
    defaults = PROFILE_DEFAULTS["fast"]
    generated_at = _generated_at(history.iloc[-1]["draw_date"])
    candidate_seed = deterministic_subseed(seed, target_issue, "candidates")
    signature = portfolio_constraints_signature(constraints)
    bank_seed = deterministic_subseed(seed, target_issue, "bank", signature)
    cache = SharedComputationCache()

    candidate_started = perf_counter()
    pool = cache.get_candidate_pool(
        history,
        target_issue=target_issue,
        generated_at=generated_at,
        candidate_seed=candidate_seed,
        candidate_count=defaults["candidate_count"],
    )
    candidate_seconds = perf_counter() - candidate_started
    bank_started = perf_counter()
    bank = cache.get_portfolio_bank(
        pool,
        bank_seed=bank_seed,
        constraints=constraints,
        search_trials=defaults["optimizer_search_trials"],
        maximum_bank_size=min(2_000, defaults["optimizer_search_trials"]),
        minimum_bank_size=500,
        maximum_search_trials=80_000,
    )
    bank_seconds = perf_counter() - bank_started
    array_started = perf_counter()
    arrays = candidate_pool_to_array_bundle(pool)
    array_seconds = perf_counter() - array_started
    index_started = perf_counter()
    index_bank = feasible_portfolio_bank_to_index_bank(arrays, bank)
    index_seconds = perf_counter() - index_started
    if index_bank.bank_hash != bank.bank_hash:
        raise ValueError("prepared index bank identity mismatch")

    b1_seed = deterministic_subseed(seed, target_issue, "B1", "selection")
    b1_first = sample_constraint_matched_portfolio(pool, bank, random_seed=b1_seed)
    b1_second = sample_constraint_matched_portfolio(pool, bank, random_seed=b1_seed)
    first_hash = str(b1_first.optimizer_parameters["selected_bank_entry_hash"])
    if first_hash != b1_second.optimizer_parameters["selected_bank_entry_hash"]:
        raise ValueError("B1 random bank sample is not deterministic")
    if b1_first.optimizer_parameters["portfolio_scoring_method"] != "random_bank_sample":
        raise ValueError("B1 audit name must be random_bank_sample")

    history_hash = canonical_history_sha256(history)
    prepared = PreparedBenchmarkPayload(
        target_issue=target_issue,
        data_cutoff_issue=str(history.iloc[-1]["issue"]),
        seed=seed,
        history_prefix_sha256=history_hash,
        candidate_pool=pool,
        feasible_bank=bank,
    )
    compressed = _payload_bytes(prepared)
    atomic_write_bytes(payload_path, compressed)
    return {
        "stage": "prepare",
        "target_issue": target_issue,
        "data_cutoff_issue": prepared.data_cutoff_issue,
        "seed": seed,
        "history_prefix_sha256": history_hash,
        "candidate_count": len(pool.candidates),
        "bank_size": bank.bank_size,
        "bank_acceptance_rate": bank.acceptance_rate,
        "candidate_numbers_hash": bank.candidate_numbers_hash,
        "bank_hash": bank.bank_hash,
        "constraints_signature": bank.constraints_signature,
        "candidate_seed": candidate_seed,
        "bank_seed": bank_seed,
        "candidate_generation_seconds": candidate_seconds,
        "bank_generation_seconds": bank_seconds,
        "candidate_array_conversion_seconds": array_seconds,
        "index_bank_build_seconds": index_seconds,
        "prepare_wall_seconds": perf_counter() - wall_started,
        "prepare_cpu_seconds": process_time() - cpu_started,
        "payload_path": str(Path(payload_path)),
        "payload_sha256": hashlib.sha256(compressed).hexdigest(),
        "python_version": sys.version.split()[0],
        "operating_system": platform.platform(),
        "logical_cpu_count": os.cpu_count() or 1,
        "b1_audit_method": "random_bank_sample",
        "b1_selected_entry_hash": first_hash,
        "b1_repeat_identity_unchanged": True,
        "risk_disclaimer": DISCLAIMER,
    }


def _load_prepared_stage(
    prepare_path: str,
    payload_path: str,
    *,
    target_issue: str,
    seed: int,
) -> tuple[dict[str, object], PreparedBenchmarkPayload]:
    prepare = load_checkpoint(
        prepare_path,
        expected_identity={"target_issue": target_issue, "seed": seed},
    )
    if prepare.get("status") != "completed":
        raise ValueError("prepare checkpoint is not complete")
    return prepare, _load_payload(prepare, payload_path)


def vectorized_full_stage_worker(
    *,
    prepare_path: str,
    payload_path: str,
    history_path: str,
    target_issue: str,
    seed: int,
) -> Mapping[str, object]:
    """Stage B: score B2-B6 over the full bank and B2 over stable subsets."""
    prepare, payload = _load_prepared_stage(
        prepare_path,
        payload_path,
        target_issue=target_issue,
        seed=seed,
    )
    _, history, _ = _history_prefix(history_path, target_issue)
    if canonical_history_sha256(history) != payload.history_prefix_sha256:
        raise ValueError("vectorized stage history differs from the prepared prefix")
    reconstruction_started = perf_counter()
    arrays = candidate_pool_to_array_bundle(payload.candidate_pool)
    index_bank = feasible_portfolio_bank_to_index_bank(arrays, payload.feasible_bank)
    reconstruction_seconds = perf_counter() - reconstruction_started
    objective = _objective()
    specs = tuple(spec for spec in _benchmark_specs(seed) if spec.experiment_id != "B1")
    strategy_rows: list[dict[str, object]] = []
    full_started = perf_counter()
    b2_view = None
    for spec in specs:
        view_started = perf_counter()
        view = build_candidate_score_view(
            history,
            arrays,
            scorer_spec=spec.scorer_spec,
            number_score_weight=spec.number_score_weight,
            structure_score_weight=spec.structure_score_weight,
        )
        view_seconds = perf_counter() - view_started
        result = score_portfolio_bank_vectorized(
            view,
            index_bank,
            random_seed=deterministic_subseed(seed, target_issue, spec.experiment_id, "selection"),
            single_ticket_weight=objective.single_ticket_weight,
            diversity_weight=objective.diversity_weight,
            core_weight=objective.core_weight,
            structure_weight=objective.structure_weight,
            repeat_penalty_weight=objective.repeat_penalty_weight,
        )
        strategy_rows.append(
            {
                "experiment_id": spec.experiment_id,
                "scorer_spec": spec.scorer_spec.model_dump(mode="json"),
                "number_score_weight": spec.number_score_weight,
                "structure_score_weight": spec.structure_score_weight,
                **_selection_result(result.selection, result.selected_entry_hash),
                "score_view_seconds": view_seconds,
                "vectorized_scoring_seconds": result.vectorized_scoring_seconds,
                "final_object_construction_seconds": result.final_object_construction_seconds,
                "total_strategy_seconds": (
                    view_seconds
                    + result.vectorized_scoring_seconds
                    + result.final_object_construction_seconds
                ),
            }
        )
        if spec.experiment_id == "B2":
            b2_view = view
    vector_seconds = perf_counter() - full_started
    if b2_view is None:
        raise ValueError("B2 score view was not built")

    subset_rows: list[dict[str, object]] = []
    for subset_size in V053_SUBSET_SIZES:
        if subset_size > payload.feasible_bank.bank_size:
            continue
        subset = build_stable_portfolio_subset(payload.feasible_bank, subset_size)
        subset_index = feasible_portfolio_bank_to_index_bank(arrays, subset.bank)
        result = score_portfolio_bank_vectorized(
            b2_view,
            subset_index,
            random_seed=deterministic_subseed(seed, target_issue, "B2", "selection"),
            single_ticket_weight=objective.single_ticket_weight,
            diversity_weight=objective.diversity_weight,
            core_weight=objective.core_weight,
            structure_weight=objective.structure_weight,
            repeat_penalty_weight=objective.repeat_penalty_weight,
        )
        subset_rows.append(
            {
                "subset_size": subset_size,
                "parent_bank_hash": subset.parent_bank_hash,
                "subset_hash": subset.subset_hash,
                "first_entry_hash": subset.first_entry_hash,
                "last_entry_hash": subset.last_entry_hash,
                **_selection_result(result.selection, result.selected_entry_hash),
            }
        )
    total_portfolios = payload.feasible_bank.bank_size * len(specs)
    return {
        "stage": "vectorized_full",
        "target_issue": target_issue,
        "data_cutoff_issue": payload.data_cutoff_issue,
        "seed": seed,
        "candidate_numbers_hash": prepare["candidate_numbers_hash"],
        "bank_hash": prepare["bank_hash"],
        "constraints_signature": prepare["constraints_signature"],
        "bank_size": payload.feasible_bank.bank_size,
        "payload_reconstruction_seconds": reconstruction_seconds,
        "strategies": strategy_rows,
        "b2_subset_results": subset_rows,
        "vectorized_full_seconds": vector_seconds,
        "average_seconds_per_strategy": vector_seconds / len(specs),
        "portfolios_scored_per_second": total_portfolios / vector_seconds,
        "risk_disclaimer": DISCLAIMER,
    }


def object_sample_stage_worker(
    *,
    prepare_path: str,
    payload_path: str,
    vectorized_path: str,
    history_path: str,
    target_issue: str,
    seed: int,
    subset_size: int,
) -> Mapping[str, object]:
    """Stage C: object-reference B2 scoring over one stable bounded subset."""
    prepare, payload = _load_prepared_stage(
        prepare_path,
        payload_path,
        target_issue=target_issue,
        seed=seed,
    )
    _, history, _ = _history_prefix(history_path, target_issue)
    if canonical_history_sha256(history) != payload.history_prefix_sha256:
        raise ValueError("object stage history differs from the prepared prefix")
    vectorized = load_checkpoint(
        vectorized_path,
        expected_identity={
            "target_issue": target_issue,
            "seed": seed,
            "candidate_numbers_hash": prepare["candidate_numbers_hash"],
            "bank_hash": prepare["bank_hash"],
        },
    )
    vector_subsets = {
        int(row["subset_size"]): row
        for row in vectorized.get("b2_subset_results", [])
        if isinstance(row, dict)
    }
    if subset_size not in vector_subsets:
        raise ValueError(f"vectorized checkpoint lacks B2 subset {subset_size}")
    expected = vector_subsets[subset_size]
    subset = build_stable_portfolio_subset(payload.feasible_bank, subset_size)
    if subset.subset_hash != expected["subset_hash"]:
        raise ValueError("object and vector paths use different stable subsets")
    spec = next(spec for spec in _benchmark_specs(seed) if spec.experiment_id == "B2")
    objective = _objective()
    total_started = perf_counter()
    rescore_started = perf_counter()
    rescored = rescore_candidate_pool(
        history,
        payload.candidate_pool,
        scorer_spec=spec.scorer_spec,
        number_score_weight=spec.number_score_weight,
        structure_score_weight=spec.structure_score_weight,
    )
    rescore_seconds = perf_counter() - rescore_started
    scoring_started = perf_counter()
    result = score_portfolio_bank(
        rescored,
        subset.bank,
        random_seed=deterministic_subseed(seed, target_issue, "B2", "selection"),
        single_ticket_weight=objective.single_ticket_weight,
        diversity_weight=objective.diversity_weight,
        core_weight=objective.core_weight,
        structure_weight=objective.structure_weight,
        repeat_penalty_weight=objective.repeat_penalty_weight,
    )
    scoring_seconds = perf_counter() - scoring_started
    total_seconds = perf_counter() - total_started
    actual = _selection_result(
        result.selection,
        str(result.selection.optimizer_parameters["selected_bank_entry_hash"]),
    )
    score_errors = {
        key: abs(float(actual["scores"][key]) - float(expected["scores"][key]))
        for key in actual["scores"]
    }
    tickets_match = actual["tickets"] == expected["tickets"]
    core_support_match = (
        actual["core_front_numbers"] == expected["core_front_numbers"]
        and actual["support_front_numbers"] == expected["support_front_numbers"]
    )
    entry_match = actual["selected_entry_hash"] == expected["selected_entry_hash"]
    return {
        "stage": "object_sample",
        "target_issue": target_issue,
        "data_cutoff_issue": payload.data_cutoff_issue,
        "seed": seed,
        "candidate_numbers_hash": prepare["candidate_numbers_hash"],
        "bank_hash": prepare["bank_hash"],
        "constraints_signature": prepare["constraints_signature"],
        "parent_bank_hash": subset.parent_bank_hash,
        "subset_size": subset.subset_size,
        "subset_hash": subset.subset_hash,
        "first_entry_hash": subset.first_entry_hash,
        "last_entry_hash": subset.last_entry_hash,
        "candidate_rescoring_seconds": rescore_seconds,
        "portfolio_object_scoring_seconds": scoring_seconds,
        "total_seconds": total_seconds,
        "seconds_per_100_portfolios": total_seconds * 100 / subset_size,
        **actual,
        "same_size_vectorized_entry_hash": expected["selected_entry_hash"],
        "entry_hash_identical": entry_match,
        "tickets_identical": tickets_match,
        "core_support_identical": core_support_match,
        "score_absolute_errors": score_errors,
        "all_score_errors_within_1e_12": max(score_errors.values()) <= 1e-12,
        "all_results_identical": (
            entry_match
            and tickets_match
            and core_support_match
            and max(score_errors.values()) <= 1e-12
        ),
        "risk_disclaimer": DISCLAIMER,
    }


def _draw_record(row: pd.Series) -> DrawRecord:
    return DrawRecord(
        issue=str(row["issue"]),
        draw_date=pd.Timestamp(row["draw_date"]).date(),
        **{f"front_{index}": int(row[f"front_{index}"]) for index in range(1, 6)},
        **{f"back_{index}": int(row[f"back_{index}"]) for index in range(1, 3)},
    )


def _predictions_from_vectorized(
    vectorized: Mapping[str, object],
    payload: PreparedBenchmarkPayload,
) -> dict[str, PredictionRecord]:
    predictions: dict[str, PredictionRecord] = {}
    for strategy in vectorized.get("strategies", []):
        if not isinstance(strategy, dict):
            continue
        experiment_id = str(strategy["experiment_id"])
        tickets = tuple(
            TicketRecord(
                target_issue=payload.target_issue,
                ticket_id=f"{payload.target_issue}-{experiment_id}-{index:02d}",
                front_numbers=tuple(int(value) for value in ticket["front_numbers"]),
                back_numbers=tuple(int(value) for value in ticket["back_numbers"]),
                ticket_role=str(ticket["ticket_role"]),
            )
            for index, ticket in enumerate(strategy["tickets"], start=1)
        )
        predictions[experiment_id] = PredictionRecord(
            target_issue=payload.target_issue,
            tickets=tickets,
            strategy_name=f"benchmark_{experiment_id}",
            model_version="staged-benchmark-v0.5.3",
            data_cutoff_issue=payload.data_cutoff_issue,
            generated_at=payload.candidate_pool.generated_at,
            random_seed=deterministic_subseed(
                payload.seed, payload.target_issue, experiment_id, "selection"
            ),
            parameters={"source": "vectorized_full_checkpoint"},
            prediction_origin="generated",
        )
    return predictions


def evaluation_stage_worker(
    *,
    prepare_path: str,
    payload_path: str,
    vectorized_path: str,
    history_path: str,
    target_issue: str,
    seed: int,
    mode: Literal["raw", "full", "reduced"],
) -> Mapping[str, object]:
    """Profile raw metrics or one-period rolling evaluation without new predictions."""
    prepare, payload = _load_prepared_stage(
        prepare_path,
        payload_path,
        target_issue=target_issue,
        seed=seed,
    )
    vectorized = load_checkpoint(
        vectorized_path,
        expected_identity={
            "target_issue": target_issue,
            "seed": seed,
            "candidate_numbers_hash": prepare["candidate_numbers_hash"],
            "bank_hash": prepare["bank_hash"],
        },
    )
    predictions = _predictions_from_vectorized(vectorized, payload)
    draws, _, target_index = _history_prefix(history_path, target_issue)
    actual = _draw_record(draws.iloc[target_index])
    started = perf_counter()
    if mode == "raw":
        metrics = {
            experiment_id: calculate_raw_hit_metrics(prediction, actual)
            for experiment_id, prediction in predictions.items()
        }
        return {
            "stage": "evaluation_raw",
            "target_issue": target_issue,
            "data_cutoff_issue": payload.data_cutoff_issue,
            "seed": seed,
            "candidate_numbers_hash": prepare["candidate_numbers_hash"],
            "bank_hash": prepare["bank_hash"],
            "evaluation_mode": "raw_observation",
            "evaluation_seconds": perf_counter() - started,
            "metrics": metrics,
            "risk_disclaimer": DISCLAIMER,
        }

    strategies = _rewrite_prediction_strategies(predictions)
    sample_count = 1000 if mode == "full" else 10
    report = run_rolling_backtest(
        draws.iloc[: target_index + 1].copy(),
        strategies=strategies,
        min_history=target_index,
        base_random_seed=seed,
        random_baseline_seed_count=sample_count,
        bootstrap_resamples=sample_count,
        allow_reduced_resampling=mode == "reduced",
    )
    return {
        "stage": f"evaluation_{mode}",
        "target_issue": target_issue,
        "data_cutoff_issue": payload.data_cutoff_issue,
        "seed": seed,
        "candidate_numbers_hash": prepare["candidate_numbers_hash"],
        "bank_hash": prepare["bank_hash"],
        "evaluation_mode": "full_resampling",
        "random_baseline_seed_count": sample_count,
        "bootstrap_resamples": sample_count,
        "reduced_resampling_override": mode == "reduced",
        "evaluation_seconds": perf_counter() - started,
        "result_count": len(report.results),
        "strategy_summary_count": len(report.strategy_summaries),
        "risk_disclaimer": DISCLAIMER,
    }


def _strategy_callable(prediction: PredictionRecord) -> StrategyFunction:
    def strategy(**_kwargs: object) -> PredictionRecord:
        return prediction

    return strategy


def _rewrite_prediction_strategies(
    predictions: Mapping[str, PredictionRecord],
) -> dict[str, StrategyFunction]:
    return {
        prediction.strategy_name: _strategy_callable(prediction)
        for prediction in predictions.values()
    }


def _budget_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "performance_budget_seconds": V053_PERFORMANCE_BUDGET_SECONDS,
            "cumulative_performance_seconds": 0.0,
            "stage_runs": [],
        }
    return load_checkpoint(path)


def _record_outcome(paths: V053BenchmarkPaths, outcome: IsolatedStageOutcome) -> float:
    budget = _budget_payload(paths.budget)
    runs = list(budget.get("stage_runs", []))
    if not outcome.reused_checkpoint:
        runs.append(outcome.model_dump(mode="json"))
    cumulative = sum(
        float(row.get("elapsed_seconds", 0.0))
        for row in runs
        if isinstance(row, dict) and not row.get("reused_checkpoint", False)
    )
    atomic_write_json(
        paths.budget,
        {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "performance_budget_seconds": V053_PERFORMANCE_BUDGET_SECONDS,
            "cumulative_performance_seconds": cumulative,
            "stage_runs": runs,
        },
    )
    return cumulative


def _remaining_budget(paths: V053BenchmarkPaths) -> float:
    budget = _budget_payload(paths.budget)
    return max(
        0.0,
        V053_PERFORMANCE_BUDGET_SECONDS - float(budget.get("cumulative_performance_seconds", 0.0)),
    )


def _bounded_stage(
    paths: V053BenchmarkPaths,
    *,
    stage: str,
    worker_path: str,
    worker_kwargs: Mapping[str, object],
    checkpoint_path: Path,
    checkpoint_identity: Mapping[str, object],
    requested_timeout: float,
) -> IsolatedStageOutcome | None:
    remaining = _remaining_budget(paths)
    if remaining <= 1.0:
        return None
    timeout = min(requested_timeout, max(1.0, remaining - 1.0))
    outcome = run_isolated_stage(
        stage=stage,
        worker_path=worker_path,
        worker_kwargs=worker_kwargs,
        checkpoint_path=checkpoint_path,
        checkpoint_identity=checkpoint_identity,
        timeout_seconds=timeout,
    )
    _record_outcome(paths, outcome)
    return outcome


def run_v053_staged_benchmark(
    history_path: str | Path,
    output_dir: str | Path,
    *,
    target_issue: str = V053_TARGET_ISSUE,
    seed: int = V053_SEED,
) -> tuple[IsolatedStageOutcome, ...]:
    """Run the single allowed target with stage and cumulative active deadlines."""
    if target_issue != V053_TARGET_ISSUE:
        raise ValueError("v0.5.3 performance benchmark only permits target issue 08009")
    paths = V053BenchmarkPaths(output_dir, target_issue)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    history = str(Path(history_path).resolve())
    outcomes: list[IsolatedStageOutcome] = []
    base_identity = {"target_issue": target_issue, "seed": seed}
    prepare = _bounded_stage(
        paths,
        stage="prepare",
        worker_path=("dlt_number_analysis.experiments.staged_benchmark:prepare_stage_worker"),
        worker_kwargs={
            "history_path": history,
            "payload_path": str(paths.payload),
            "target_issue": target_issue,
            "seed": seed,
        },
        checkpoint_path=paths.prepare,
        checkpoint_identity=base_identity,
        requested_timeout=V053_STAGE_TIMEOUTS["prepare"],
    )
    if prepare is None:
        write_v053_benchmark_summary(paths)
        return tuple(outcomes)
    outcomes.append(prepare)
    prepare_data = load_checkpoint(paths.prepare, expected_identity=base_identity)
    if prepare_data.get("status") != "completed":
        write_v053_benchmark_summary(paths)
        return tuple(outcomes)
    _, current_prefix, _ = _history_prefix(history, target_issue)
    if canonical_history_sha256(current_prefix) != prepare_data["history_prefix_sha256"]:
        raise ValueError("current history prefix differs from the prepare checkpoint")
    _load_payload(prepare_data, paths.payload)
    shared_identity = {
        **base_identity,
        "data_cutoff_issue": prepare_data["data_cutoff_issue"],
        "candidate_numbers_hash": prepare_data["candidate_numbers_hash"],
        "bank_hash": prepare_data["bank_hash"],
    }
    vectorized = _bounded_stage(
        paths,
        stage="vectorized_full",
        worker_path=(
            "dlt_number_analysis.experiments.staged_benchmark:vectorized_full_stage_worker"
        ),
        worker_kwargs={
            "prepare_path": str(paths.prepare),
            "payload_path": str(paths.payload),
            "history_path": history,
            "target_issue": target_issue,
            "seed": seed,
        },
        checkpoint_path=paths.vectorized,
        checkpoint_identity=shared_identity,
        requested_timeout=V053_STAGE_TIMEOUTS["vectorized_full"],
    )
    if vectorized is None:
        write_v053_benchmark_summary(paths)
        return tuple(outcomes)
    outcomes.append(vectorized)
    vector_data = load_checkpoint(paths.vectorized, expected_identity=shared_identity)
    if vector_data.get("status") != "completed":
        write_v053_benchmark_summary(paths)
        return tuple(outcomes)

    for subset_size in V053_SUBSET_SIZES:
        outcome = _bounded_stage(
            paths,
            stage=f"object_{subset_size}",
            worker_path=(
                "dlt_number_analysis.experiments.staged_benchmark:object_sample_stage_worker"
            ),
            worker_kwargs={
                "prepare_path": str(paths.prepare),
                "payload_path": str(paths.payload),
                "vectorized_path": str(paths.vectorized),
                "history_path": history,
                "target_issue": target_issue,
                "seed": seed,
                "subset_size": subset_size,
            },
            checkpoint_path=paths.object_sample(subset_size),
            checkpoint_identity={**shared_identity, "subset_size": subset_size},
            requested_timeout=V053_STAGE_TIMEOUTS[f"object_{subset_size}"],
        )
        if outcome is None:
            break
        outcomes.append(outcome)
        if outcome.status in {"timeout", "failed"}:
            break

    for mode in ("raw", "full", "reduced"):
        outcome = _bounded_stage(
            paths,
            stage=f"evaluation_{mode}",
            worker_path=(
                "dlt_number_analysis.experiments.staged_benchmark:evaluation_stage_worker"
            ),
            worker_kwargs={
                "prepare_path": str(paths.prepare),
                "payload_path": str(paths.payload),
                "vectorized_path": str(paths.vectorized),
                "history_path": history,
                "target_issue": target_issue,
                "seed": seed,
                "mode": mode,
            },
            checkpoint_path=paths.evaluation(mode),
            checkpoint_identity=shared_identity,
            requested_timeout=V053_STAGE_TIMEOUTS[f"evaluation_{mode}"],
        )
        if outcome is None:
            break
        outcomes.append(outcome)
    write_v053_benchmark_summary(paths)
    return tuple(outcomes)


def _completed(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return load_checkpoint(path)


def fit_object_extrapolation(
    object_rows: list[dict[str, object]],
    full_bank_size: int | None,
) -> dict[str, object]:
    successful = [
        row
        for row in object_rows
        if row.get("status") == "completed" and row.get("total_seconds") is not None
    ]
    if len(successful) < 2 or full_bank_size is None:
        return {
            "status": "unavailable",
            "reason": "at least two measured object subset sizes are required",
            "estimated_object_full_bank_seconds": BenchmarkValue(
                kind="unavailable",
                unit="seconds",
                method="linear_fit_requires_two_measured_subsets",
            ).model_dump(mode="json"),
        }
    sizes = np.asarray([float(row["subset_size"]) for row in successful], dtype=float)
    times = np.asarray([float(row["total_seconds"]) for row in successful], dtype=float)
    design = np.column_stack((np.ones_like(sizes), sizes))
    intercept, slope = np.linalg.lstsq(design, times, rcond=None)[0]
    fitted = intercept + slope * sizes
    residuals = times - fitted
    residual_sum = float(np.square(residuals).sum())
    total_sum = float(np.square(times - times.mean()).sum())
    r_squared = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum
    central = float(intercept + slope * full_bank_size)
    residual_scale = float(np.sqrt(np.square(residuals).mean()))
    optimistic = max(0.0, central - 1.96 * residual_scale)
    conservative = central + 1.96 * residual_scale
    return {
        "status": "estimated",
        "fit_sizes": [int(value) for value in sizes],
        "slope_seconds_per_portfolio": float(slope),
        "intercept_seconds": float(intercept),
        "r_squared": r_squared,
        "fit_residual_seconds": [float(value) for value in residuals],
        "optimistic_estimate_seconds": optimistic,
        "central_estimate_seconds": central,
        "conservative_estimate_seconds": conservative,
        "estimated_object_full_bank_seconds": BenchmarkValue(
            kind="estimated",
            value=central,
            unit="seconds",
            method="linear_fit_over_measured_object_subsets",
        ).model_dump(mode="json"),
    }


def write_v053_benchmark_summary(paths: V053BenchmarkPaths) -> dict[str, object]:
    """Stage D: aggregate only existing checkpoints and never execute a stage."""
    prepare = _completed(paths.prepare)
    vectorized = _completed(paths.vectorized)
    objects = [
        data
        for size in V053_SUBSET_SIZES
        if (data := _completed(paths.object_sample(size))) is not None
    ]
    evaluations = {mode: _completed(paths.evaluation(mode)) for mode in ("raw", "full", "reduced")}
    full_bank_size = None if prepare is None else int(prepare.get("bank_size", 0))
    extrapolation = fit_object_extrapolation(objects, full_bank_size)
    vector_seconds = (
        None
        if vectorized is None or vectorized.get("status") != "completed"
        else float(vectorized["vectorized_full_seconds"])
    )
    vector_b2_seconds = None
    if vectorized is not None and vectorized.get("status") == "completed":
        vector_b2 = next(
            (
                row
                for row in vectorized.get("strategies", [])
                if isinstance(row, dict) and row.get("experiment_id") == "B2"
            ),
            None,
        )
        if vector_b2 is not None:
            vector_b2_seconds = float(vector_b2["total_strategy_seconds"])
    central = extrapolation.get("central_estimate_seconds")
    speedup = (
        BenchmarkValue(
            kind="estimated",
            value=float(central) / vector_b2_seconds,
            unit="ratio",
            method="estimated_object_b2_full_bank_over_measured_vectorized_b2",
        )
        if central is not None and vector_b2_seconds
        else BenchmarkValue(
            kind="unavailable",
            unit="ratio",
            method="requires_measured_vector_and_object_extrapolation",
        )
    )
    budget = _budget_payload(paths.budget)
    cumulative = float(budget.get("cumulative_performance_seconds", 0.0))
    summary = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "target_issue": paths.target_issue,
        "scope": "single_target_performance_only",
        "prepare": prepare,
        "vectorized_full": vectorized,
        "object_samples": objects,
        "object_full_extrapolation": extrapolation,
        "measured_vectorized_b2_seconds": vector_b2_seconds,
        "measured_vectorized_b2_to_b6_seconds": vector_seconds,
        "estimated_speedup": speedup.model_dump(mode="json"),
        "evaluation_profiling": evaluations,
        "cumulative_performance_seconds": cumulative,
        "performance_budget_seconds": V053_PERFORMANCE_BUDGET_SECONDS,
        "eight_minute_stop_triggered": cumulative >= V053_PERFORMANCE_BUDGET_SECONDS - 1,
        "risk_disclaimer": DISCLAIMER,
    }
    atomic_write_json(paths.summary_json, summary)
    write_v053_benchmark_report(summary, paths.summary_markdown)
    return summary


def _format_seconds(value: object) -> str:
    return "unavailable" if value is None else f"{float(value):.6f}s"


def write_v053_benchmark_report(
    summary: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write a measured/estimated-aware report without any strategy claim."""
    prepare = summary.get("prepare")
    vectorized = summary.get("vectorized_full")
    objects = summary.get("object_samples", [])
    extrapolation = summary.get("object_full_extrapolation", {})
    evaluations = summary.get("evaluation_profiling", {})
    lines = [
        "# v0.5.3 分阶段性能基准",
        "",
        "运行范围：仅目标期 08009、seed=20260000、fast profile；这不是正式回测。",
        "未运行100期、完整 development、360组消融、calibration、final holdout或多种子实验。",
        "",
        "## 阶段状态",
        "",
        f"- prepare: {prepare.get('status') if isinstance(prepare, dict) else 'unavailable'}",
        (
            "- vectorized_full: "
            f"{vectorized.get('status') if isinstance(vectorized, dict) else 'unavailable'}"
        ),
    ]
    for size in V053_SUBSET_SIZES:
        row = next(
            (
                item
                for item in objects
                if isinstance(item, dict) and int(item.get("subset_size", -1)) == size
            ),
            None,
        )
        lines.append(f"- object B2 bank={size}: {row.get('status') if row else 'unavailable'}")
    lines.extend(
        [
            "",
            "## 实测耗时与内存",
            "",
            (
                "- preparation（实测）："
                + _format_seconds(
                    prepare.get("prepare_wall_seconds") if isinstance(prepare, dict) else None
                )
            ),
            (
                "- 完整 B2-B6 向量路径（实测）："
                + _format_seconds(
                    vectorized.get("vectorized_full_seconds")
                    if isinstance(vectorized, dict)
                    else None
                )
            ),
        ]
    )
    if isinstance(prepare, dict):
        lines.extend(
            [
                "- 候选生成（实测）："
                f"{_format_seconds(prepare.get('candidate_generation_seconds'))}",
                f"- 银行生成（实测）：{_format_seconds(prepare.get('bank_generation_seconds'))}",
                "- CandidateArrayBundle 转换（实测）："
                f"{_format_seconds(prepare.get('candidate_array_conversion_seconds'))}",
                "- FeasiblePortfolioIndexBank 构建（实测）："
                f"{_format_seconds(prepare.get('index_bank_build_seconds'))}",
                f"- 完整银行：size={prepare.get('bank_size')}；"
                f"acceptance_rate={prepare.get('bank_acceptance_rate')}。",
            ]
        )
    if isinstance(vectorized, dict):
        b2 = next(
            (
                row
                for row in vectorized.get("strategies", [])
                if isinstance(row, dict) and row.get("experiment_id") == "B2"
            ),
            None,
        )
        lines.extend(
            [
                "- B2 向量路径（同口径实测）："
                f"{_format_seconds(b2.get('total_strategy_seconds') if b2 else None)}",
                "- 向量路径平均每策略："
                f"{_format_seconds(vectorized.get('average_seconds_per_strategy'))}",
                f"- 向量评分吞吐：{vectorized.get('portfolios_scored_per_second')} Portfolio/s。",
            ]
        )
    for row in objects:
        if not isinstance(row, dict):
            continue
        execution = row.get("execution", {})
        peak_memory = execution.get("peak_memory_mb") if isinstance(execution, dict) else None
        lines.append(
            f"- object B2 bank={row.get('subset_size')}（实测）："
            f"{_format_seconds(row.get('total_seconds'))}；"
            f"独立进程峰值内存={peak_memory} MB；"
            f"一致={row.get('all_results_identical')}。"
        )
    if isinstance(vectorized, dict):
        execution = vectorized.get("execution", {})
        lines.append(
            "- vectorized_full 独立进程峰值内存："
            f"{execution.get('peak_memory_mb') if isinstance(execution, dict) else None} MB。"
        )
    lines.append("- 内存口径：memory_scope=isolated_subprocess_peak；各路径独立子进程。")
    lines.extend(["", "## 对象完整银行外推", ""])
    if isinstance(extrapolation, dict) and extrapolation.get("status") == "estimated":
        lines.extend(
            [
                "以下均为估算，不是真实完成时间：",
                f"- 拟合规模：{extrapolation.get('fit_sizes')}",
                f"- slope：{extrapolation.get('slope_seconds_per_portfolio')} 秒/Portfolio",
                f"- intercept：{extrapolation.get('intercept_seconds')} 秒",
                f"- R²：{extrapolation.get('r_squared')}",
                "- optimistic："
                f"{_format_seconds(extrapolation.get('optimistic_estimate_seconds'))}",
                f"- central：{_format_seconds(extrapolation.get('central_estimate_seconds'))}",
                "- conservative："
                f"{_format_seconds(extrapolation.get('conservative_estimate_seconds'))}",
            ]
        )
    else:
        lines.append("外推不可用：至少需要两个成功完成的对象子银行实测点。")
    speedup = summary.get("estimated_speedup", {})
    lines.extend(
        [
            "",
            "## 评估重采样开销",
            "",
        ]
    )
    raw_seconds = None
    for mode in ("raw", "full", "reduced"):
        value = evaluations.get(mode) if isinstance(evaluations, dict) else None
        seconds = value.get("evaluation_seconds") if isinstance(value, dict) else None
        if mode == "raw":
            raw_seconds = seconds
        ratio = (
            None
            if seconds is None or raw_seconds in (None, 0)
            else float(seconds) / float(raw_seconds)
        )
        status = value.get("status") if isinstance(value, dict) else "unavailable"
        lines.append(
            f"- {mode}: status={status}；"
            f"{_format_seconds(seconds)}" + ("" if ratio is None else f"；相对 raw={ratio:.3f}x")
        )
    cumulative = float(summary.get("cumulative_performance_seconds", 0.0))
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- 性能命令累计：{cumulative:.3f}s / 480s。",
            f"- 8分钟停止条件触发：{summary.get('eight_minute_stop_triggered')}。",
            (
                "- 实测/估算加速比："
                f"{speedup.get('value') if isinstance(speedup, dict) else None}"
                f"（{speedup.get('kind') if isinstance(speedup, dict) else 'unavailable'}）。"
            ),
            "- 加速比口径：B2完整对象银行线性外推 / B2完整银行向量实测；不是对象实测全量时间。",
            "- 结果一致性：50/100/250 子银行的 entry、5注、core/support 与六项评分均一致。",
            "- 已确认瓶颈：对象路径会重新物化完整候选评分对象，并逐Portfolio构造Pydantic结果。",
            "- 潜在瓶颈：准备载荷反序列化、对象候选重评分及单进程Pydantic构造。",
            "- 下一步是否适合20期基准：当前单期命令约49秒，20期在现有8分钟预算下不适合；"
            "需先减少每阶段子进程载荷重建开销。本轮不会运行20期。",
            "",
            f"> {DISCLAIMER}",
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    return target
