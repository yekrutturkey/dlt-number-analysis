"""Resumable schema-v3 experiment command with exact logical-cohort boundaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import pandas as pd

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_prize_rule_schedule, load_verified_history
from dlt_number_analysis.experiments.cohort import (
    CohortDefinitionIdentity,
    CohortPurpose,
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
from dlt_number_analysis.experiments.storage import (
    LegacyObservationImportStore,
)
from dlt_number_analysis.experiments.storage_v3 import ExperimentResultStoreV3
from dlt_number_analysis.experiments.worker import execute_experiment_process_task
from dlt_number_analysis.pipeline import PROFILE_DEFAULTS

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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--all-contexts-audit", action="store_true")
    parser.add_argument("--run-context-sha256")
    parser.add_argument("--cohort-definition-sha256")
    parser.add_argument("--cohort-id")
    parser.add_argument(
        "--cohort-purpose",
        choices=(
            "development_full",
            "development_smoke",
            "ablation_screening",
            "ablation_full_development",
            "calibration",
            "final_holdout",
        ),
    )
    parser.add_argument(
        "--inference-context",
        choices=("formal", "smoke"),
        default="formal",
    )
    parser.add_argument(
        "--legacy-observations",
        type=Path,
        default=PROJECT_ROOT / "outputs/backtests/v05_observations.csv",
        help="source used only with --migrate-legacy-observations",
    )
    parser.add_argument(
        "--migrate-legacy-observations",
        action="store_true",
        help="explicitly import old CSV rows as legacy_unverified, never as formal results",
    )
    parser.add_argument(
        "--legacy-import-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/legacy_import",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/schema_v3",
    )
    parser.add_argument(
        "--preflight-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/run_plans",
    )
    parser.add_argument(
        "--contexts-audit-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/schema_v3/context_index.csv",
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
    store: ExperimentResultStoreV3,
    specifications: tuple[ExperimentSpec, ...],
    targets_by_experiment: dict[str, tuple[str, ...]],
    identities_by_experiment: dict[str, ExperimentExecutionIdentity],
    contexts_by_experiment: dict[str, RunContextIdentity],
    cohort: CohortDefinitionIdentity,
) -> dict[tuple[int, tuple[str, ...]], list[ExperimentSpec]]:
    groups: dict[tuple[int, tuple[str, ...]], list[ExperimentSpec]] = defaultdict(list)
    for spec in specifications:
        expected = targets_by_experiment[spec.experiment_id]
        for seed in spec.seeds:
            status = store.status(
                spec,
                identities_by_experiment[spec.experiment_id],
                contexts_by_experiment[spec.experiment_id],
                cohort,
                seed=seed,
                expected_target_issues=expected,
            )
            if status.is_complete:
                print(f"skip completed: {spec.data_split.phase}/{spec.experiment_id}/seed_{seed}")
                continue
            groups[(seed, status.pending_target_issues)].append(
                spec.model_copy(update={"seeds": (seed,)})
            )
    return groups


def _cohort_purpose(args: argparse.Namespace, phase: ExperimentPhase) -> CohortPurpose:
    if args.cohort_purpose is not None:
        return cast(CohortPurpose, args.cohort_purpose)
    if phase == "final_holdout":
        return "final_holdout"
    if phase == "calibration":
        return "calibration"
    if args.stage == "screening":
        return "ablation_screening"
    if args.stage == "full_development":
        return "ablation_full_development"
    if args.inference_context == "smoke":
        return "development_smoke"
    return "development_full"


def _task_plan_rows(
    pending_groups: dict[tuple[int, tuple[str, ...]], list[ExperimentSpec]],
    *,
    chunk_size: int,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for (seed, targets), specs in pending_groups.items():
        for task_index, offset in enumerate(range(0, len(targets), chunk_size)):
            chunk = targets[offset : offset + chunk_size]
            rows.append(
                {
                    "seed": seed,
                    "task_index": task_index,
                    "target_issues": list(chunk),
                    "target_count": len(chunk),
                    "experiment_ids": [spec.experiment_id for spec in specs],
                }
            )
    return tuple(rows)


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
    """Preflight, execute, resume, or report one exact schema-v3 logical cohort."""
    args = build_parser().parse_args(argv)
    specifications = _selected_specifications(args)
    store = ExperimentResultStoreV3(args.results_root)
    if args.migrate_legacy_observations:
        if not args.legacy_observations.exists():
            raise ValueError(f"legacy observations do not exist: {args.legacy_observations}")
        migrated = LegacyObservationImportStore(args.legacy_import_root).import_csv(
            args.legacy_observations
        )
        print(f"imported legacy_unverified observations: {migrated}")
    if args.all_contexts_audit:
        index = store.context_index()
        args.contexts_audit_output.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(args.contexts_audit_output, index=False)
        print(f"schema-v3 context audit: {args.contexts_audit_output}")
        print(DISCLAIMER)
        return 0
    phase = specifications[0].data_split.phase
    if any(spec.data_split.phase != phase for spec in specifications):
        raise ValueError("one command cannot mix experiment phases")
    seeds = tuple(sorted({seed for spec in specifications for seed in spec.seeds}))
    if args.report_only:
        if args.run_context_sha256 is None or args.cohort_definition_sha256 is None:
            raise ValueError(
                "report-only requires --run-context-sha256 and --cohort-definition-sha256"
            )
        observations = store.load_cohort(
            run_context_sha256=args.run_context_sha256,
            cohort_definition_sha256=args.cohort_definition_sha256,
            phase=phase,
            experiment_ids=[spec.experiment_id for spec in specifications],
            seeds=seeds,
        )
        validate_single_formal_report_context(observations)
        comparisons = compare_to_constraint_matched_baseline(
            observations,
            minimum_paired_observations=(2 if args.inference_context == "smoke" else 30),
        )
        write_experiment_summary_report(
            observations,
            comparisons,
            args.summary_output,
            inference_context=args.inference_context,
        )
        if phase == "final_holdout":
            write_holdout_results_report(observations, args.holdout_output)
        print(f"selected schema-v3 observations: {len(observations)}")
        print(DISCLAIMER)
        return 0

    draws = load_verified_history(args.draws)
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
    target_sets = {targets for targets in targets_by_experiment.values()}
    if len(target_sets) != 1:
        raise ValueError("one logical cohort requires the same complete targets for every spec")
    complete_targets = next(iter(target_sets))
    final_holdout = phase == "final_holdout"
    evaluation_mode = args.evaluation_mode or (
        "full_resampling" if final_holdout else "raw_observation"
    )
    if final_holdout and evaluation_mode != "full_resampling":
        raise ValueError("final_holdout requires full_resampling evaluation")
    prize_tables = load_prize_rule_schedule(PROJECT_ROOT / "config/prize_tiers.json").tables
    identities_by_experiment: dict[str, ExperimentExecutionIdentity] = {}
    contexts_by_experiment: dict[str, RunContextIdentity] = {}
    for spec in specifications:
        uses_shared_bank = spec.experiment_id in {f"B{index}" for index in range(1, 7)} or (
            spec.experiment_version == "v0.5.1-ablation-shared-v1"
        )
        run_context = build_run_context_identity(
            draws,
            data_split_spec=spec.data_split,
            phase=spec.data_split.phase,
            evaluation_mode=evaluation_mode,
            profile=args.profile,
            portfolio_scoring_method=(
                args.portfolio_scoring_method if uses_shared_bank else "object_reference"
            ),
            minimum_history=args.minimum_history,
            minimum_bank_size=args.minimum_bank_size if uses_shared_bank else 1,
            maximum_bank_search_trials=(
                args.maximum_bank_search_trials
                if uses_shared_bank
                else PROFILE_DEFAULTS[args.profile]["optimizer_search_trials"]
            ),
            prize_tables=prize_tables,
            worker_count=args.workers,
            target_issues=args.target_issues,
        )
        identities_by_experiment[spec.experiment_id] = build_experiment_execution_identity(
            run_context, spec
        )
        contexts_by_experiment[spec.experiment_id] = run_context
    run_hashes = {context.run_context_sha256 for context in contexts_by_experiment.values()}
    if len(run_hashes) != 1:
        raise ValueError("one schema-v3 command requires exactly one shared run context")
    shared_context = next(iter(contexts_by_experiment.values()))
    phase_targets = (
        split_history(draws, specifications[0].data_split).phase_frame(phase)["issue"].astype(str)
    )
    purpose = _cohort_purpose(args, phase)
    provisional_id = args.cohort_id or (
        f"{purpose}_{min(complete_targets, key=int)}_"
        f"{max(complete_targets, key=int)}_{len(complete_targets)}"
    )
    cohort = build_cohort_definition_identity(
        complete_targets,
        cohort_purpose=purpose,
        phase=phase,
        cohort_id=provisional_id,
        phase_target_issues=tuple(phase_targets),
    )
    if args.run_context_sha256 is not None and (
        args.run_context_sha256 != shared_context.run_context_sha256
    ):
        raise ValueError("requested run context differs from the resolved execution context")
    if args.cohort_definition_sha256 is not None and (
        args.cohort_definition_sha256 != cohort.cohort_definition_sha256
    ):
        raise ValueError("requested cohort hash differs from the resolved logical cohort")

    statuses = {
        (spec.experiment_id, seed): store.status(
            spec,
            identities_by_experiment[spec.experiment_id],
            contexts_by_experiment[spec.experiment_id],
            cohort,
            seed=seed,
            expected_target_issues=complete_targets,
        )
        for spec in specifications
        for seed in spec.seeds
    }
    pending_groups = _pending_groups(
        store,
        specifications,
        targets_by_experiment,
        identities_by_experiment,
        contexts_by_experiment,
        cohort,
    )
    effective_chunk_size = len(complete_targets) if final_holdout else args.chunk_size
    task_plan = _task_plan_rows(pending_groups, chunk_size=effective_chunk_size)
    output_paths = [
        store.partition_path(
            spec,
            identities_by_experiment[spec.experiment_id],
            cohort,
            seed=seed,
        )
        for spec in specifications
        for seed in spec.seeds
    ]
    preflight = build_experiment_preflight(
        project_root=PROJECT_ROOT,
        history_path=args.draws,
        draws=draws,
        specifications=specifications,
        run_context=shared_context,
        identities=identities_by_experiment,
        cohort=cohort,
        statuses=statuses,
        task_chunk_plan=task_plan,
        result_output_paths=output_paths,
        inference_context=args.inference_context,
        allow_dirty=args.allow_dirty,
    )
    write_experiment_preflight(preflight, args.preflight_root)
    if args.preflight_only:
        print(f"preflight ready: {preflight.ready}")
        print(DISCLAIMER)
        return 0 if preflight.ready else 1

    failures = False
    scheduler_reports: list[ProcessSchedulerReport] = []
    for (seed, targets), grouped_specs in pending_groups.items():
        if not targets:
            continue
        tasks = build_process_tasks(
            grouped_specs,
            targets,
            master_seed=seed,
            chunk_size=effective_chunk_size,
            partition_mode=args.partition_mode,
            logical_cohort=cohort,
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
        scheduler_reports.append(report)
        timestamp = report.started_at.strftime("%Y%m%dT%H%M%S%fZ")
        write_process_scheduler_report(
            report,
            args.scheduler_report_dir / f"run_{timestamp}.json",
        )
        for frame in _frames_from_scheduler_results(report):
            if not frame.empty:
                store.append(frame)
        failures = failures or bool(report.failed_tasks)
    observations = store.load_cohort(
        run_context_sha256=shared_context.run_context_sha256,
        cohort_definition_sha256=cohort.cohort_definition_sha256,
        phase=phase,
        experiment_ids=[spec.experiment_id for spec in specifications],
        seeds=seeds,
    )
    validate_single_formal_report_context(observations)
    comparisons = compare_to_constraint_matched_baseline(
        observations,
        minimum_paired_observations=(2 if args.inference_context == "smoke" else 30),
    )
    write_experiment_summary_report(
        observations,
        comparisons,
        args.summary_output,
        inference_context=args.inference_context,
    )
    if args.stage is not None:
        write_ablation_results_csv(_ablation_status_rows(observations), args.ablation_output)
    if phase == "final_holdout":
        write_holdout_results_report(observations, args.holdout_output)
    write_experiment_runtime_report(tuple(scheduler_reports), args.experiment_runtime_output)
    print(f"selected schema-v3 observations: {len(observations)}")
    print(f"run_context_sha256: {shared_context.run_context_sha256}")
    print(f"cohort_definition_sha256: {cohort.cohort_definition_sha256}")
    print(DISCLAIMER)
    return 1 if failures else 0
