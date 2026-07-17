"""Resumable schema-v4 experiment command with mandatory formal preflight."""

from __future__ import annotations

import argparse
import json
import sys
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
from dlt_number_analysis.experiments.git_audit import AuthorizedGeneratedPath
from dlt_number_analysis.experiments.identity import (
    ExperimentExecutionIdentity,
    RunContextIdentity,
    build_experiment_execution_identity,
    build_run_context_identity,
)
from dlt_number_analysis.experiments.manual_operations import (
    CurrentPointerRepairAuditV2,
    StaleRunLockClearAuditV2,
    atomic_write_manual_audit,
    audit_manual_operations,
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
from dlt_number_analysis.experiments.run_lock import FormalRunLock, clear_stale_run_lock
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
from dlt_number_analysis.experiments.storage_v4 import (
    ExperimentPartitionStatusV4,
    ExperimentResultStoreV4,
)
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--report-only", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--all-contexts-audit", action="store_true")
    modes.add_argument("--audit-generations", action="store_true")
    modes.add_argument("--repair-current", action="store_true")
    modes.add_argument("--clear-stale-run-lock", action="store_true")
    modes.add_argument("--audit-manual-operations", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--repair-generation-id")
    parser.add_argument("--repair-reason")
    parser.add_argument("--stale-lock-reason")
    parser.add_argument("--manual-operations-audit-output", type=Path)
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
        default=PROJECT_ROOT / "outputs/experiments/schema_v4",
    )
    parser.add_argument(
        "--preflight-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/run_plans",
    )
    parser.add_argument(
        "--contexts-audit-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--holdout-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--scheduler-report-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--experiment-runtime-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--run-log-dir", type=Path)
    parser.add_argument(
        "--holdout-lock",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/final_holdout/holdout_lock.json",
    )
    return parser


def _resolve_output_paths(
    args: argparse.Namespace,
    *,
    run_context_sha256: str,
    cohort_definition_sha256: str,
) -> tuple[Path, tuple[AuthorizedGeneratedPath, ...]]:
    """Resolve identity-scoped defaults and exact Git-authorized output paths."""
    formal_root = (
        PROJECT_ROOT / "outputs/formal_runs" / run_context_sha256 / cohort_definition_sha256
    )
    explicit = {
        name: getattr(args, name) is not None
        for name in (
            "contexts_audit_output",
            "summary_output",
            "ablation_output",
            "holdout_output",
            "scheduler_report_dir",
            "experiment_runtime_output",
            "run_log_dir",
        )
    }
    args.contexts_audit_output = args.contexts_audit_output or (
        formal_root / "reports/context_index.csv"
    )
    args.summary_output = args.summary_output or (formal_root / "reports/experiment_summary.md")
    args.ablation_output = args.ablation_output or (formal_root / "reports/ablation_results.csv")
    args.holdout_output = args.holdout_output or (formal_root / "reports/holdout_results.md")
    args.scheduler_report_dir = args.scheduler_report_dir or (formal_root / "scheduler")
    args.experiment_runtime_output = args.experiment_runtime_output or (
        formal_root / "runtime/experiment_runtime.md"
    )
    args.run_log_dir = args.run_log_dir or (formal_root / "logs")
    default_results = PROJECT_ROOT / "outputs/experiments/schema_v4"
    default_preflight = PROJECT_ROOT / "outputs/experiments/run_plans"
    authorized: list[AuthorizedGeneratedPath] = [
        AuthorizedGeneratedPath(
            path=args.results_root,
            kind="directory",
            explicitly_provided=args.results_root != default_results,
        ),
        AuthorizedGeneratedPath(
            path=args.preflight_root,
            kind="directory",
            explicitly_provided=args.preflight_root != default_preflight,
        ),
        AuthorizedGeneratedPath(
            path=args.scheduler_report_dir,
            kind="directory",
            explicitly_provided=explicit["scheduler_report_dir"],
        ),
        AuthorizedGeneratedPath(
            path=args.run_log_dir,
            kind="directory",
            explicitly_provided=explicit["run_log_dir"],
        ),
        AuthorizedGeneratedPath(
            path=args.summary_output,
            kind="file",
            explicitly_provided=explicit["summary_output"],
        ),
        AuthorizedGeneratedPath(
            path=args.experiment_runtime_output,
            kind="file",
            explicitly_provided=explicit["experiment_runtime_output"],
        ),
        AuthorizedGeneratedPath(
            path=args.ablation_output,
            kind="file",
            explicitly_provided=explicit["ablation_output"],
        ),
        AuthorizedGeneratedPath(
            path=args.holdout_output,
            kind="file",
            explicitly_provided=explicit["holdout_output"],
        ),
        AuthorizedGeneratedPath(
            path=args.contexts_audit_output,
            kind="file",
            explicitly_provided=explicit["contexts_audit_output"],
        ),
    ]
    return formal_root, tuple(authorized)


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
    store: ExperimentResultStoreV4,
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
    """Preflight, execute, resume, audit, repair, or report one schema-v4 cohort."""
    args = build_parser().parse_args(argv)
    if args.repair_current and (
        args.repair_generation_id is None or not (args.repair_reason or "").strip()
    ):
        raise ValueError("--repair-current requires --repair-generation-id and --repair-reason")
    if args.clear_stale_run_lock and not (args.stale_lock_reason or "").strip():
        raise ValueError("--clear-stale-run-lock requires --stale-lock-reason")
    store = ExperimentResultStoreV4(args.results_root)
    if args.all_contexts_audit:
        output = args.contexts_audit_output or (
            PROJECT_ROOT / "outputs/formal_runs/context_audits/context_index.csv"
        )
        index = store.context_index()
        output.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(output, index=False)
        print(f"schema-v4 context audit: {output}")
        print(DISCLAIMER)
        return 0
    specifications = _selected_specifications(args)
    phase = specifications[0].data_split.phase
    if any(spec.data_split.phase != phase for spec in specifications):
        raise ValueError("one command cannot mix experiment phases")
    seeds = tuple(sorted({seed for spec in specifications for seed in spec.seeds}))
    if args.report_only:
        if args.run_context_sha256 is None or args.cohort_definition_sha256 is None:
            raise ValueError(
                "report-only requires --run-context-sha256 and --cohort-definition-sha256"
            )
        _resolve_output_paths(
            args,
            run_context_sha256=args.run_context_sha256,
            cohort_definition_sha256=args.cohort_definition_sha256,
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
        print(f"selected schema-v4 observations: {len(observations)}")
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
    if not complete_targets:
        raise ValueError("logical cohort cannot be empty")
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
        raise ValueError("one schema-v4 command requires exactly one shared run context")
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
    formal_run_root, authorized_paths = _resolve_output_paths(
        args,
        run_context_sha256=shared_context.run_context_sha256,
        cohort_definition_sha256=cohort.cohort_definition_sha256,
    )
    generation_audits = {}
    statuses: dict[tuple[str, int], ExperimentPartitionStatusV4] = {}
    identity_conflicts: list[str] = []
    for spec in specifications:
        identity = identities_by_experiment[spec.experiment_id]
        for seed in spec.seeds:
            key = f"{spec.experiment_id}:seed_{seed}"
            audit = store.audit_partition_generations(spec, identity, cohort, seed=seed)
            generation_audits[key] = audit
            try:
                status = store.status(
                    spec,
                    identity,
                    contexts_by_experiment[spec.experiment_id],
                    cohort,
                    seed=seed,
                    expected_target_issues=complete_targets,
                )
            except (OSError, ValueError) as error:
                identity_conflicts.append(f"{key}: {error}")
                status = ExperimentPartitionStatusV4(
                    phase=phase,
                    experiment_id=spec.experiment_id,
                    experiment_version=spec.experiment_version,
                    run_context_sha256=identity.run_context_sha256,
                    execution_config_sha256=identity.execution_config_sha256,
                    cohort_definition_sha256=cohort.cohort_definition_sha256,
                    expected_targets_sha256=cohort.payload.expected_targets_sha256,
                    seed=seed,
                    current_path=store.current_path(spec, identity, cohort, seed=seed),
                    current_generation_id=None,
                    completed_target_issues=(),
                    pending_target_issues=complete_targets,
                    is_complete=False,
                )
            statuses[(spec.experiment_id, seed)] = status
    if args.audit_generations:
        print(
            json.dumps(
                {key: audit.model_dump(mode="json") for key, audit in generation_audits.items()},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        print(DISCLAIMER)
        return 0
    if args.repair_current:
        if len(specifications) != 1 or len(seeds) != 1:
            raise ValueError("--repair-current requires exactly one experiment and one seed")
        spec = specifications[0]
        path = store.repair_current_pointer(
            spec,
            identities_by_experiment[spec.experiment_id],
            cohort,
            seed=seeds[0],
            generation_id=args.repair_generation_id,
            reason=args.repair_reason,
        )
        record = CurrentPointerRepairAuditV2.model_validate_json(path.read_text(encoding="utf-8"))
        print(f"operation_id: {record.operation_id}")
        print(f"audit_path: {path}")
        print(f"final_status: {record.status}")
        print(f"current_changed: {record.status == 'completed'}")
        print(DISCLAIMER)
        return 0
    run_lock_path = formal_run_root / "RUNNING.lock"
    if args.audit_manual_operations:
        report = audit_manual_operations((args.run_log_dir, args.results_root))
        if args.manual_operations_audit_output is not None:
            atomic_write_manual_audit(args.manual_operations_audit_output, report)
            print(f"manual operations audit: {args.manual_operations_audit_output}")
        print(f"prepared_operations: {len(report.prepared_operations)}")
        print(f"completed_operations: {len(report.completed_operations)}")
        print(f"failed_operations: {len(report.failed_operations)}")
        print(f"inconsistent_operations: {len(report.inconsistent_operations)}")
        print(DISCLAIMER)
        return 0
    if args.clear_stale_run_lock:
        audit_path = clear_stale_run_lock(
            run_lock_path,
            reason=args.stale_lock_reason,
            audit_directory=args.run_log_dir,
        )
        record = StaleRunLockClearAuditV2.model_validate_json(
            audit_path.read_text(encoding="utf-8")
        )
        print(f"operation_id: {record.operation_id}")
        print(f"audit_path: {audit_path}")
        print(f"final_status: {record.status}")
        print(f"lock_deleted: {record.status == 'completed'}")
        print(DISCLAIMER)
        return 0

    pending_groups: dict[tuple[int, tuple[str, ...]], list[ExperimentSpec]] = defaultdict(list)
    for spec in specifications:
        for seed in spec.seeds:
            status = statuses[(spec.experiment_id, seed)]
            if status.is_complete:
                print(f"skip completed: {phase}/{spec.experiment_id}/seed_{seed}")
                continue
            pending_groups[(seed, status.pending_target_issues)].append(
                spec.model_copy(update={"seeds": (seed,)})
            )
    effective_chunk_size = len(complete_targets) if final_holdout else args.chunk_size
    task_plan = _task_plan_rows(pending_groups, chunk_size=effective_chunk_size)
    output_paths = [
        store.current_path(
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
        authorized_generated_paths=authorized_paths,
        generation_audits=generation_audits,
        results_root=args.results_root,
        legacy_migration_enabled=args.migrate_legacy_observations,
        stage=args.stage,
        final_holdout_report=final_holdout,
        reads_holdout_lock=final_holdout,
        identity_conflicts=identity_conflicts,
    )
    write_experiment_preflight(preflight, args.preflight_root)
    if args.preflight_only:
        print(f"preflight ready: {preflight.ready}")
        print(DISCLAIMER)
        return 0 if preflight.ready else 1
    if not preflight.ready:
        raise RuntimeError("formal experiment preflight failed; execution was not started")

    failures = False
    scheduler_reports: list[ProcessSchedulerReport] = []
    command_summary = " ".join(argv if argv is not None else sys.argv[1:])
    lock = FormalRunLock(
        run_lock_path,
        run_context_sha256=shared_context.run_context_sha256,
        cohort_definition_sha256=cohort.cohort_definition_sha256,
        phase=phase,
        seeds=seeds,
        command_summary=command_summary,
    )
    with lock:
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
    print(f"selected schema-v4 observations: {len(observations)}")
    print(f"run_context_sha256: {shared_context.run_context_sha256}")
    print(f"cohort_definition_sha256: {cohort.cohort_definition_sha256}")
    print(DISCLAIMER)
    return 1 if failures else 0
