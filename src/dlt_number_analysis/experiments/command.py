"""Resumable v0.5.1 experiment command implementation backed by Parquet partitions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import pandas as pd

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_verified_history
from dlt_number_analysis.experiments.reporting import (
    compare_to_constraint_matched_baseline,
    write_ablation_results_csv,
    write_experiment_runtime_report,
    write_experiment_summary_report,
    write_holdout_results_report,
)
from dlt_number_analysis.experiments.scheduler import (
    ProcessSchedulerReport,
    build_process_tasks,
    run_process_scheduler,
    write_process_scheduler_report,
)
from dlt_number_analysis.experiments.specs import (
    ExperimentPhase,
    ExperimentSpec,
    ablation_experiment_specs,
    baseline_experiment_specs,
)
from dlt_number_analysis.experiments.splits import split_history
from dlt_number_analysis.experiments.stages import (
    AblationStage,
    build_staged_ablation_plan,
    select_stage_target_issues,
)
from dlt_number_analysis.experiments.storage import EXPERIMENT_PRIMARY_KEY, ExperimentResultStore
from dlt_number_analysis.experiments.worker import execute_experiment_process_task

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    """Build the resumable experiment CLI without starting any expensive run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, default=PROJECT_ROOT / "data/raw/draws.csv")
    parser.add_argument("--experiment-ids", nargs="+", default=["B1"])
    parser.add_argument(
        "--phase",
        choices=("development", "calibration", "final_holdout"),
        default="development",
    )
    parser.add_argument(
        "--stage",
        choices=("screening", "full_development", "calibration", "final_holdout"),
    )
    parser.add_argument("--ranked-experiment-ids", nargs="+")
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument("--frozen-experiment-id")
    parser.add_argument("--target-issues", nargs="+")
    parser.add_argument("--seed", type=int, default=20260000)
    parser.add_argument("--minimum-history", type=int, default=100)
    parser.add_argument("--profile", choices=("fast", "standard", "final"), default="fast")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument(
        "--partition-mode",
        choices=("experiment_spec", "target_chunk"),
        default="target_chunk",
    )
    parser.add_argument("--random-baseline-seeds", type=int, default=1000)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--evaluation-mode",
        choices=("raw_observation", "full_resampling"),
        help="defaults to raw observation except for the mandatory full final holdout",
    )
    parser.add_argument("--minimum-bank-size", type=int, default=500)
    parser.add_argument("--maximum-bank-search-trials", type=int, default=80_000)
    parser.add_argument(
        "--portfolio-scoring-method",
        choices=("object_reference", "numpy_vectorized"),
        default="numpy_vectorized",
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--inference-context",
        choices=("formal", "smoke"),
        default="formal",
    )
    parser.add_argument(
        "--legacy-observations",
        type=Path,
        default=PROJECT_ROOT / "outputs/backtests/v05_observations.csv",
        help="one-time source migrated into immutable Parquet partitions",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/experiment_summary.md",
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/ablation_results.csv",
    )
    parser.add_argument(
        "--holdout-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/holdout_results.md",
    )
    parser.add_argument(
        "--scheduler-report-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/run_metadata",
    )
    parser.add_argument(
        "--experiment-runtime-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/experiment_runtime.md",
    )
    parser.add_argument(
        "--holdout-lock",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/final_holdout/holdout_lock.json",
    )
    return parser


def _selected_specifications(args: argparse.Namespace) -> tuple[ExperimentSpec, ...]:
    if args.stage is not None:
        stage = cast(AblationStage, args.stage)
        frozen = None
        if stage == "final_holdout":
            if args.frozen_experiment_id is None:
                raise ValueError("final_holdout stage requires --frozen-experiment-id")
            by_id = {
                spec.experiment_id: spec for spec in ablation_experiment_specs(phase="calibration")
            }
            try:
                frozen = by_id[args.frozen_experiment_id]
            except KeyError as error:
                raise ValueError("frozen experiment ID is not a registered ablation") from error
        plan = build_staged_ablation_plan(
            stage,
            ranked_experiment_ids=args.ranked_experiment_ids,
            calibration_count=args.calibration_count,
            frozen_specification=frozen,
            base_seed=args.seed,
        )
        return plan.specifications
    phase = cast(ExperimentPhase, args.phase)
    available = baseline_experiment_specs(seeds=(args.seed,), phase=phase)
    selected = tuple(spec for spec in available if spec.experiment_id in args.experiment_ids)
    missing = set(args.experiment_ids).difference(spec.experiment_id for spec in selected)
    if missing:
        raise ValueError(f"unknown experiment IDs: {sorted(missing)}")
    return selected


def _expected_targets(
    draws: pd.DataFrame,
    spec: ExperimentSpec,
    *,
    minimum_history: int,
    explicit_targets: list[str] | None,
    stage: str | None,
) -> tuple[str, ...]:
    if explicit_targets is not None:
        return tuple(str(issue) for issue in explicit_targets)
    if stage is not None:
        plan = build_staged_ablation_plan(
            cast(AblationStage, stage),
            ranked_experiment_ids=[item.experiment_id for item in (spec,)],
            calibration_count=5,
            frozen_specification=(spec if stage == "final_holdout" else None),
            base_seed=spec.seeds[0],
        )
        return select_stage_target_issues(draws, plan, minimum_history=minimum_history)
    phase_frame = split_history(draws, spec.data_split).phase_frame(spec.data_split.phase)
    allowed = set(phase_frame["issue"].astype(str).tolist())
    return tuple(
        issue
        for index, issue in enumerate(draws["issue"].astype(str).tolist())
        if index >= minimum_history and issue in allowed
    )


def _pending_groups(
    store: ExperimentResultStore,
    specifications: tuple[ExperimentSpec, ...],
    targets_by_experiment: dict[str, tuple[str, ...]],
) -> dict[tuple[int, tuple[str, ...]], list[ExperimentSpec]]:
    groups: dict[tuple[int, tuple[str, ...]], list[ExperimentSpec]] = defaultdict(list)
    for spec in specifications:
        expected = targets_by_experiment[spec.experiment_id]
        for seed in spec.seeds:
            status = store.status(spec, seed=seed, expected_target_issues=expected)
            if status.is_complete:
                print(f"skip completed: {spec.data_split.phase}/{spec.experiment_id}/seed_{seed}")
                continue
            groups[(seed, status.pending_target_issues)].append(
                spec.model_copy(update={"seeds": (seed,)})
            )
    return groups


def _ablation_status_rows(observations: pd.DataFrame) -> pd.DataFrame:
    counts = (
        observations.groupby(["phase", "experiment_id"], sort=True).size().to_dict()
        if not observations.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for phase in ("development", "calibration"):
        for spec in ablation_experiment_specs(phase=phase):
            parameters = spec.scorer_spec.parameters
            observation_count = int(counts.get((phase, spec.experiment_id), 0))
            rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "experiment_version": spec.experiment_version,
                    "phase": phase,
                    "window": parameters["window"],
                    "decay": parameters["decay"],
                    "structure_score_weight": spec.structure_score_weight,
                    "target_front_pool_size": spec.portfolio_constraints.target_front_pool_size,
                    "core_number_count": spec.portfolio_constraints.min_core_front_numbers,
                    "observation_count": observation_count,
                    "execution_status": "executed" if observation_count else "not_executed",
                }
            )
    return pd.DataFrame.from_records(rows)


def _migrate_legacy_observations(
    store: ExperimentResultStore,
    legacy_path: Path,
) -> int:
    """Append only legacy keys not already present; never rewrite or delete the source CSV."""
    if not legacy_path.exists():
        return 0
    legacy = pd.read_csv(
        legacy_path,
        dtype={"target_issue": "string", "data_cutoff_issue": "string"},
    )
    if legacy.duplicated(list(EXPERIMENT_PRIMARY_KEY)).any():
        raise ValueError("legacy observations contain duplicate experiment primary keys")
    existing = store.load_all()
    if existing.empty:
        pending = legacy
    else:
        keys = existing.loc[:, list(EXPERIMENT_PRIMARY_KEY)].drop_duplicates()
        marked = legacy.merge(
            keys.assign(_already_stored=True),
            on=list(EXPERIMENT_PRIMARY_KEY),
            how="left",
        )
        pending = marked.loc[marked["_already_stored"].isna()].drop(columns="_already_stored")
    if pending.empty:
        return 0
    store.append(pending)
    return len(pending)


def _frames_from_scheduler_results(
    report: ProcessSchedulerReport,
) -> list[pd.DataFrame]:
    completed = report.completed_tasks
    frames: list[pd.DataFrame] = []
    for item in completed:
        payload = item.result.get("observations_json")
        if not isinstance(payload, str):
            raise ValueError("worker result is missing observations_json")
        records = json.loads(payload)
        frames.append(pd.DataFrame.from_records(records))
    return frames


def main(argv: list[str] | None = None) -> int:
    """Run only missing tasks, append partitions, and aggregate reports from Parquet."""
    args = build_parser().parse_args(argv)
    draws = load_verified_history(args.draws)
    specifications = _selected_specifications(args)
    store = ExperimentResultStore(args.results_root)
    migrated = _migrate_legacy_observations(store, args.legacy_observations)
    if migrated:
        print(f"migrated legacy observations: {migrated}")
    targets_by_experiment = {
        spec.experiment_id: _expected_targets(
            draws,
            spec,
            minimum_history=args.minimum_history,
            explicit_targets=args.target_issues,
            stage=args.stage,
        )
        for spec in specifications
    }
    failures = False
    if not args.report_only:
        for (seed, targets), grouped_specs in _pending_groups(
            store, specifications, targets_by_experiment
        ).items():
            if not targets:
                continue
            final_holdout = grouped_specs[0].data_split.phase == "final_holdout"
            evaluation_mode = args.evaluation_mode or (
                "full_resampling" if final_holdout else "raw_observation"
            )
            if final_holdout and evaluation_mode != "full_resampling":
                raise ValueError("final_holdout requires full_resampling evaluation")
            tasks = build_process_tasks(
                grouped_specs,
                targets,
                master_seed=seed,
                chunk_size=len(targets) if final_holdout else args.chunk_size,
                partition_mode=args.partition_mode,
                parameters={
                    "history_path": str(args.draws.resolve()),
                    "prize_rule_path": str((PROJECT_ROOT / "config/prize_tiers.json").resolve()),
                    "profile": args.profile,
                    "minimum_history": args.minimum_history,
                    "random_baseline_seed_count": args.random_baseline_seeds,
                    "bootstrap_resamples": args.bootstrap_resamples,
                    "evaluation_mode": evaluation_mode,
                    "minimum_bank_size": args.minimum_bank_size,
                    "maximum_bank_search_trials": args.maximum_bank_search_trials,
                    "portfolio_scoring_method": args.portfolio_scoring_method,
                    "holdout_lock_path": str(args.holdout_lock.resolve()),
                },
            )
            report = run_process_scheduler(
                tasks,
                execute_experiment_process_task,
                workers=1 if final_holdout else args.workers,
            )
            timestamp = report.started_at.strftime("%Y%m%dT%H%M%S%fZ")
            write_process_scheduler_report(
                report,
                args.scheduler_report_dir / f"run_{timestamp}.json",
            )
            for frame in _frames_from_scheduler_results(report):
                if not frame.empty:
                    store.append(frame)
            failures = failures or bool(report.failed_tasks)
    observations = store.load_all()
    comparisons = compare_to_constraint_matched_baseline(observations)
    write_experiment_summary_report(
        observations,
        comparisons,
        args.summary_output,
        inference_context=args.inference_context,
    )
    write_ablation_results_csv(_ablation_status_rows(observations), args.ablation_output)
    write_holdout_results_report(observations, args.holdout_output)
    scheduler_reports = tuple(
        ProcessSchedulerReport.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(args.scheduler_report_dir.glob("run_*.json"))
    )
    write_experiment_runtime_report(scheduler_reports, args.experiment_runtime_output)
    print(f"partitioned observations: {len(observations)}")
    print(DISCLAIMER)
    return 1 if failures else 0
