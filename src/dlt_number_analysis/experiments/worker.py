"""Top-level pickle-safe worker used by the ProcessPool experiment scheduler."""

from __future__ import annotations

import random
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import JsonValue

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_prize_rule_schedule, load_verified_history
from dlt_number_analysis.experiments.runner import run_experiment_batch
from dlt_number_analysis.experiments.scheduler import ExperimentProcessTask
from dlt_number_analysis.pipeline import PipelineProfile
from dlt_number_analysis.portfolio import PortfolioScoringMethod


def execute_experiment_process_task(
    task: ExperimentProcessTask,
) -> dict[str, JsonValue]:
    """Load verified data and execute one deterministic target/spec process task."""
    random.seed(task.deterministic_subseed)
    np.random.seed(task.deterministic_subseed % (2**32))
    parameters = task.parameters
    history_path = parameters.get("history_path")
    if not isinstance(history_path, str):
        raise ValueError("process task requires a history_path parameter")
    draws = load_verified_history(Path(history_path))
    prize_tables = None
    prize_rule_path = parameters.get("prize_rule_path")
    if isinstance(prize_rule_path, str):
        prize_tables = load_prize_rule_schedule(prize_rule_path).tables
    result = run_experiment_batch(
        draws,
        task.experiment_specs,
        profile=cast(PipelineProfile, str(parameters.get("profile", "fast"))),
        minimum_history=int(parameters.get("minimum_history", 100)),
        random_baseline_seed_count=int(parameters.get("random_baseline_seed_count", 1000)),
        bootstrap_resamples=int(parameters.get("bootstrap_resamples", 1000)),
        minimum_bank_size=int(parameters.get("minimum_bank_size", 500)),
        maximum_bank_search_trials=int(parameters.get("maximum_bank_search_trials", 80_000)),
        portfolio_scoring_method=cast(
            PortfolioScoringMethod,
            str(parameters.get("portfolio_scoring_method", "numpy_vectorized")),
        ),
        prize_tables=prize_tables,
        holdout_lock_path=(
            str(parameters["holdout_lock_path"]) if "holdout_lock_path" in parameters else None
        ),
        target_issues=task.target_issues,
    )
    return {
        "task_id": task.task_id,
        "deterministic_subseed": task.deterministic_subseed,
        "observations_json": result.observations.to_json(
            orient="records",
            date_format="iso",
            force_ascii=False,
        ),
        "executions": [execution.model_dump(mode="json") for execution in result.executions],
        "risk_disclaimer": DISCLAIMER,
    }
