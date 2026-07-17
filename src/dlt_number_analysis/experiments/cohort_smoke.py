"""Bounded six-target, two-chunk v0.5.6 logical-cohort correctness smoke."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import pandas as pd

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_prize_rule_schedule, load_verified_history
from dlt_number_analysis.experiments.benchmark_checkpoints import atomic_write_json
from dlt_number_analysis.experiments.cohort import (
    CohortDefinitionIdentity,
    build_cohort_definition_identity,
)
from dlt_number_analysis.experiments.identity import (
    ExperimentExecutionIdentity,
    RunContextIdentity,
    build_experiment_execution_identity,
    build_run_context_identity,
)
from dlt_number_analysis.experiments.preflight import (
    build_experiment_preflight,
    write_experiment_preflight,
)
from dlt_number_analysis.experiments.reporting import (
    compare_to_constraint_matched_baseline,
    validate_single_formal_report_context,
)
from dlt_number_analysis.experiments.scheduler import build_process_tasks
from dlt_number_analysis.experiments.specs import ExperimentSpec, baseline_experiment_specs
from dlt_number_analysis.experiments.splits import split_history
from dlt_number_analysis.experiments.storage import _sha256_file
from dlt_number_analysis.experiments.storage_v3 import (
    EXPERIMENT_PRIMARY_KEY_V3,
    ExperimentResultStoreV3,
)
from dlt_number_analysis.experiments.worker import execute_experiment_process_task

V056_SMOKE_SEED = 20260000
V056_SMOKE_TIMEOUT_SECONDS = 90.0
V056_RESUME_TIMEOUT_SECONDS = 20.0
V056_REPORT_TIMEOUT_SECONDS = 20.0
V056_SMOKE_EXPERIMENT_IDS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "B6")
V056_SMOKE_COHORT_ID = "v056_cohort_smoke_6"


def select_v056_cohort_smoke_targets(
    draws: pd.DataFrame,
    *,
    minimum_history: int = 100,
) -> tuple[str, str, str, str, str, str]:
    """Select 0%, 20%, 40%, 60%, 80%, and last eligible development targets."""
    spec = baseline_experiment_specs(seeds=(V056_SMOKE_SEED,), phase="development")[1]
    development = split_history(draws, spec.data_split).development
    development_issues = set(development["issue"].astype(str))
    eligible = tuple(
        issue
        for index, issue in enumerate(draws["issue"].astype(str))
        if index >= minimum_history and issue in development_issues
    )
    if len(eligible) < 6:
        raise ValueError("development split needs at least six eligible smoke targets")
    last_index = len(eligible) - 1
    indices = (
        0,
        last_index // 5,
        2 * last_index // 5,
        3 * last_index // 5,
        4 * last_index // 5,
        last_index,
    )
    selected = tuple(eligible[index] for index in indices)
    if len(set(selected)) != 6:
        raise ValueError("dynamic v0.5.6 selection did not produce six unique targets")
    return selected  # type: ignore[return-value]


def _smoke_specs() -> tuple[ExperimentSpec, ...]:
    by_id = {
        spec.experiment_id: spec
        for spec in baseline_experiment_specs(
            seeds=(V056_SMOKE_SEED,),
            phase="development",
        )
    }
    return tuple(by_id[experiment_id] for experiment_id in V056_SMOKE_EXPERIMENT_IDS)


def _smoke_identity(
    draws: pd.DataFrame,
    specs: tuple[ExperimentSpec, ...],
    *,
    prize_rule_path: str | Path,
    minimum_history: int,
    minimum_bank_size: int,
    maximum_bank_search_trials: int,
) -> tuple[
    tuple[str, str, str, str, str, str],
    RunContextIdentity,
    dict[str, ExperimentExecutionIdentity],
    CohortDefinitionIdentity,
]:
    targets = select_v056_cohort_smoke_targets(draws, minimum_history=minimum_history)
    prize_tables = load_prize_rule_schedule(prize_rule_path).tables
    context = build_run_context_identity(
        draws,
        data_split_spec=specs[0].data_split,
        phase="development",
        evaluation_mode="raw_observation",
        profile="fast",
        portfolio_scoring_method="numpy_vectorized",
        minimum_history=minimum_history,
        minimum_bank_size=minimum_bank_size,
        maximum_bank_search_trials=maximum_bank_search_trials,
        prize_tables=prize_tables,
        worker_count=1,
        target_issues=targets,
    )
    phase_targets = tuple(
        split_history(draws, specs[0].data_split).development["issue"].astype(str)
    )
    cohort = build_cohort_definition_identity(
        targets,
        cohort_purpose="development_smoke",
        phase="development",
        cohort_id=V056_SMOKE_COHORT_ID,
        phase_target_issues=phase_targets,
    )
    identities = {
        spec.experiment_id: build_experiment_execution_identity(context, spec) for spec in specs
    }
    return targets, context, identities, cohort


def _result_frame(payload: Mapping[str, object]) -> pd.DataFrame:
    raw = payload.get("observations_json")
    if not isinstance(raw, str):
        raise ValueError("v0.5.6 task result is missing observations_json")
    return pd.DataFrame.from_records(json.loads(raw))


def _parquet_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted(root.glob("**/observations.parquet"))
    }


def run_v056_cohort_smoke_worker(
    *,
    project_root: str,
    history_path: str,
    prize_rule_path: str,
    output_dir: str,
    preflight_root: str,
    minimum_history: int = 100,
    minimum_bank_size: int = 500,
    maximum_bank_search_trials: int = 80_000,
) -> Mapping[str, object]:
    """Execute exactly two sequential three-target tasks and atomically append each."""
    draws = load_verified_history(history_path)
    specs = _smoke_specs()
    targets, context, identities, cohort = _smoke_identity(
        draws,
        specs,
        prize_rule_path=prize_rule_path,
        minimum_history=minimum_history,
        minimum_bank_size=minimum_bank_size,
        maximum_bank_search_trials=maximum_bank_search_trials,
    )
    schema_root = Path(output_dir) / "schema_v3"
    store = ExperimentResultStoreV3(schema_root)
    statuses = {
        (spec.experiment_id, V056_SMOKE_SEED): store.status(
            spec,
            identities[spec.experiment_id],
            context,
            cohort,
            seed=V056_SMOKE_SEED,
            expected_target_issues=targets,
        )
        for spec in specs
    }
    tasks = build_process_tasks(
        specs,
        targets,
        master_seed=V056_SMOKE_SEED,
        chunk_size=3,
        partition_mode="target_chunk",
        logical_cohort=cohort,
        parameters={
            "history_path": str(Path(history_path).resolve()),
            "prize_rule_path": str(Path(prize_rule_path).resolve()),
            "profile": "fast",
            "minimum_history": minimum_history,
            "random_baseline_seed_count": 1000,
            "bootstrap_resamples": 1000,
            "evaluation_mode": "raw_observation",
            "minimum_bank_size": minimum_bank_size,
            "maximum_bank_search_trials": maximum_bank_search_trials,
            "portfolio_scoring_method": "numpy_vectorized",
        },
    )
    if len(tasks) != 2 or {len(task.target_issues) for task in tasks} != {3}:
        raise ValueError("v0.5.6 smoke requires exactly two three-target tasks")
    preflight = build_experiment_preflight(
        project_root=project_root,
        history_path=history_path,
        draws=draws,
        specifications=specs,
        run_context=context,
        identities=identities,
        cohort=cohort,
        statuses=statuses,
        task_chunk_plan=tuple(
            {
                "task_id": task.task_id,
                "task_index": task.task_index,
                "target_issues": list(task.target_issues),
                "target_count": len(task.target_issues),
            }
            for task in tasks
        ),
        result_output_paths=tuple(
            store.partition_path(
                spec,
                identities[spec.experiment_id],
                cohort,
                seed=V056_SMOKE_SEED,
            )
            for spec in specs
        ),
        inference_context="smoke",
        allow_dirty=True,
    )
    write_experiment_preflight(preflight, preflight_root)
    started = perf_counter()
    task_seconds: list[float] = []
    for task in tasks:
        task_started = perf_counter()
        frame = _result_frame(execute_experiment_process_task(task))
        store.append(frame)
        task_seconds.append(perf_counter() - task_started)
    run_seconds = perf_counter() - started
    observations = store.load_cohort(
        run_context_sha256=context.run_context_sha256,
        cohort_definition_sha256=cohort.cohort_definition_sha256,
        phase="development",
        experiment_ids=V056_SMOKE_EXPERIMENT_IDS,
        seeds=(V056_SMOKE_SEED,),
    )
    validate_single_formal_report_context(observations)
    if len(observations) != 36:
        raise ValueError("v0.5.6 smoke must produce exactly 36 observations")
    if observations.duplicated(list(EXPERIMENT_PRIMARY_KEY_V3)).any():
        raise ValueError("v0.5.6 smoke produced duplicate schema-v3 primary keys")
    if observations["cohort_definition_sha256"].nunique() != 1:
        raise ValueError("task chunks did not preserve one logical cohort")
    if observations["task_targets_sha256"].nunique() != 2:
        raise ValueError("two task chunks must retain distinct task target hashes")
    experiment_counts = observations.groupby("experiment_id")["target_issue"].nunique()
    if set(experiment_counts.astype(int)) != {6}:
        raise ValueError("each B1-B6 experiment must contain all six targets")
    execution_counts = observations.groupby("experiment_id")["execution_config_sha256"].nunique()
    if set(execution_counts.astype(int)) != {1}:
        raise ValueError("execution identity changed across task chunks")
    return {
        "smoke_version": "v0.5.6-logical-cohort-smoke-v1",
        "targets": list(targets),
        "initial_task_count": len(tasks),
        "initial_task_chunk_sizes": [len(task.target_issues) for task in tasks],
        "task_targets_sha256": [task.task_targets_sha256 for task in tasks],
        "observation_count": len(observations),
        "logical_cohort_count": observations["cohort_definition_sha256"].nunique(),
        "paired_target_count": int(experiment_counts.min()),
        "run_seconds": run_seconds,
        "task_seconds": task_seconds,
        "run_context_sha256": context.run_context_sha256,
        "cohort_id": cohort.payload.cohort_id,
        "cohort_definition_sha256": cohort.cohort_definition_sha256,
        "expected_targets_sha256": cohort.payload.expected_targets_sha256,
        "execution_config_sha256_by_experiment": {
            experiment_id: identity.execution_config_sha256
            for experiment_id, identity in identities.items()
        },
        "storage_schema_version": "experiment-storage-schema-v3",
        "preflight_ready": preflight.ready,
        "parquet_hashes": _parquet_hashes(schema_root),
        "risk_disclaimer": DISCLAIMER,
    }


def validate_v056_cohort_smoke_resume(
    history_path: str | Path,
    prize_rule_path: str | Path,
    output_dir: str | Path,
    *,
    minimum_history: int = 100,
    minimum_bank_size: int = 500,
    maximum_bank_search_trials: int = 80_000,
    chunk_size: int = 2,
) -> dict[str, object]:
    """Read only statuses and prove changed chunk size schedules no generation."""
    draws = load_verified_history(history_path)
    specs = _smoke_specs()
    targets, context, identities, cohort = _smoke_identity(
        draws,
        specs,
        prize_rule_path=prize_rule_path,
        minimum_history=minimum_history,
        minimum_bank_size=minimum_bank_size,
        maximum_bank_search_trials=maximum_bank_search_trials,
    )
    schema_root = Path(output_dir) / "schema_v3"
    before = _parquet_hashes(schema_root)
    statuses = [
        ExperimentResultStoreV3(schema_root).status(
            spec,
            identities[spec.experiment_id],
            context,
            cohort,
            seed=V056_SMOKE_SEED,
            expected_target_issues=targets,
        )
        for spec in specs
    ]
    pending = tuple(
        sorted(
            {issue for status in statuses for issue in status.pending_target_issues},
            key=int,
        )
    )
    tasks = build_process_tasks(
        specs,
        pending,
        master_seed=V056_SMOKE_SEED,
        chunk_size=chunk_size,
        logical_cohort=cohort,
    )
    after = _parquet_hashes(schema_root)
    if pending or tasks or before != after:
        raise ValueError("read-only resume validation found pending work or modified Parquet")
    return {
        "resume_validation": "read_only_no_generation",
        "chunk_size": chunk_size,
        "run_context_sha256": context.run_context_sha256,
        "cohort_definition_sha256": cohort.cohort_definition_sha256,
        "completed_target_count": len(targets),
        "pending_target_count": 0,
        "scheduled_task_count": 0,
        "generation_called": False,
        "candidate_generation_called": False,
        "bank_generation_called": False,
        "observation_count_added": 0,
        "parquet_hashes_unchanged": True,
        "risk_disclaimer": DISCLAIMER,
    }


def write_v056_cohort_smoke_report(
    *,
    store_root: str | Path,
    run_context_sha256: str,
    cohort_definition_sha256: str,
    checkpoint: Mapping[str, object],
    resume: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Read exactly one schema-v3 context/cohort and write a non-advantage smoke report."""
    store = ExperimentResultStoreV3(store_root)
    observations = store.load_cohort(
        run_context_sha256=run_context_sha256,
        cohort_definition_sha256=cohort_definition_sha256,
        phase="development",
        experiment_ids=V056_SMOKE_EXPERIMENT_IDS,
        seeds=(V056_SMOKE_SEED,),
    )
    validate_single_formal_report_context(observations)
    if len(observations) != 36:
        raise ValueError("v0.5.6 report-only requires exactly 36 selected observations")
    counts = observations.groupby("experiment_id").size().astype(int).to_dict()
    comparisons = compare_to_constraint_matched_baseline(
        observations,
        bootstrap_resamples=100,
        permutations=100,
        minimum_paired_observations=2,
    )
    paired_counts = {comparison.common_target_count for comparison in comparisons}
    paired_target_count = min(paired_counts) if paired_counts else 6
    lines = [
        "# v0.5.6 logical cohort smoke",
        "",
        f"- targets: {', '.join(str(value) for value in checkpoint['targets'])}",
        f"- initial tasks: {checkpoint['initial_task_count']}",
        f"- initial task chunk sizes: {checkpoint['initial_task_chunk_sizes']}",
        f"- cohort id: `{checkpoint['cohort_id']}`",
        f"- cohort definition SHA256: `{cohort_definition_sha256}`",
        f"- expected targets SHA256: `{checkpoint['expected_targets_sha256']}`",
        f"- run context SHA256: `{run_context_sha256}`",
        f"- observations by experiment: {counts}",
        f"- total observations: {len(observations)}",
        f"- logical cohort count: {observations['cohort_definition_sha256'].nunique()}",
        f"- paired target count: {paired_target_count}",
        "- initial wrapper checkpoint schema collision repaired without rerunning computation: "
        f"{checkpoint.get('checkpoint_schema_collision_repaired', False)}",
        f"- resume validation: {resume.get('resume_validation')}",
        f"- resume scheduled tasks: {resume.get('scheduled_task_count')}",
        f"- resume generation called: {resume.get('generation_called')}",
        "- report-only generation called: false",
        "- schema version: `experiment-storage-schema-v3`",
        "",
        "This bounded smoke validates identity, chunk independence, resume, and reporting "
        "boundaries only. It is not used for parameter selection and makes no strategy-advantage "
        "claim.",
        "",
        f"> {DISCLAIMER}",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target


def write_v056_resume_audit(payload: Mapping[str, object], path: str | Path) -> Path:
    """Atomically persist the only permitted post-smoke resume audit."""
    target = Path(path)
    atomic_write_json(target, dict(payload))
    return target


def run_v056_resume_validation_worker(**kwargs: object) -> Mapping[str, object]:
    """Pickle-safe isolated wrapper for the read-only resume validation."""
    return validate_v056_cohort_smoke_resume(
        history_path=str(kwargs["history_path"]),
        prize_rule_path=str(kwargs["prize_rule_path"]),
        output_dir=str(kwargs["output_dir"]),
        chunk_size=int(kwargs.get("chunk_size", 2)),
    )


def run_v056_report_only_worker(**kwargs: object) -> Mapping[str, object]:
    """Pickle-safe isolated wrapper for exact-context report-only validation."""
    checkpoint_path = Path(str(kwargs["smoke_checkpoint_path"]))
    resume_path = Path(str(kwargs["resume_checkpoint_path"]))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    report = write_v056_cohort_smoke_report(
        store_root=str(kwargs["store_root"]),
        run_context_sha256=str(kwargs["run_context_sha256"]),
        cohort_definition_sha256=str(kwargs["cohort_definition_sha256"]),
        checkpoint=checkpoint,
        resume=resume,
        path=str(kwargs["report_path"]),
    )
    return {
        "report_path": str(report),
        "selected_observation_count": 36,
        "logical_cohort_count": 1,
        "generation_called": False,
        "candidate_generation_called": False,
        "bank_generation_called": False,
        "risk_disclaimer": DISCLAIMER,
    }
