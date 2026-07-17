"""Read-only preflight-v2 with truthful Git, schema, and holdout audit fields."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import (
    VerifiedHistoryManifest,
    canonical_history_sha256,
    verified_manifest_path,
)
from dlt_number_analysis.experiments.cohort import CohortDefinitionIdentity
from dlt_number_analysis.experiments.git_audit import (
    AuthorizedGeneratedPath,
    GitStatusEntry,
    GitWorktreeAudit,
    audit_git_worktree,
)
from dlt_number_analysis.experiments.identity import (
    ExperimentExecutionIdentity,
    RunContextIdentity,
    current_git_commit_sha,
)
from dlt_number_analysis.experiments.specs import ExperimentSpec
from dlt_number_analysis.experiments.storage import STORAGE_SCHEMA_VERSION_V4
from dlt_number_analysis.experiments.storage_v4 import (
    ExperimentPartitionStatusV4,
    PartitionGenerationAudit,
)


class PreflightCheck(BaseModel):
    """One named, machine-readable preflight assertion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    details: str


class ExperimentPreflightPlan(BaseModel):
    """Complete non-executing plan for one run context and logical cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "experiment-preflight-v2"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    storage_schema_version: str
    inference_context: str
    allow_dirty: bool
    git_dirty: bool
    git_worktree_completely_clean: bool
    git_worktree_clean_for_formal_execution: bool
    authorized_dirty_path_count: int = Field(ge=0)
    blocking_dirty_path_count: int = Field(ge=0)
    authorized_dirty_paths: tuple[str, ...]
    blocking_dirty_paths: tuple[str, ...]
    git_commit_sha: str
    history_path: str
    canonical_history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase: str
    split_boundaries: dict[str, dict[str, object]]
    experiment_versions: dict[str, str]
    evaluation_mode: str
    profile: str
    portfolio_scoring_method: str
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_config_sha256: dict[str, str]
    cohort_id: str
    cohort_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_targets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_target_count: int = Field(ge=1)
    start_issue: str
    end_issue: str
    completed_target_counts: dict[str, int]
    pending_target_counts: dict[str, int]
    current_generation_by_partition: dict[str, str | None]
    orphan_generation_count: int = Field(ge=0)
    invalid_generation_count: int = Field(ge=0)
    task_chunk_plan: tuple[dict[str, object], ...]
    result_output_paths: tuple[str, ...]
    identity_conflicts: tuple[str, ...]
    touches_legacy_or_prior_schema: bool
    reads_final_holdout_results: bool
    estimated_task_count: int = Field(ge=0)
    checks: tuple[PreflightCheck, ...]
    ready: bool
    execution_will_be_blocked: bool
    execution_block_reasons: tuple[str, ...]
    risk_disclaimer: str = DISCLAIMER


def git_worktree_dirty(project_root: str | Path) -> bool:
    """Backward-compatible complete-dirty probe backed by path classification."""
    return not audit_git_worktree(project_root, ()).git_worktree_completely_clean


def touches_prior_schema(
    *,
    project_root: str | Path,
    results_root: str | Path,
    legacy_migration_enabled: bool = False,
    reads_prior_schema_as_formal_input: bool = False,
    report_only_scans_prior_schema: bool = False,
) -> bool:
    """Truthfully detect physical overlap or explicit old-schema use."""
    root = Path(project_root).resolve()
    result = Path(results_root).resolve()
    prior = tuple(
        (root / path).resolve()
        for path in (
            "outputs/experiments/schema_v2",
            "outputs/experiments/schema_v3",
            "outputs/experiments/legacy_import",
        )
    )

    def overlaps(first: Path, second: Path) -> bool:
        try:
            first.relative_to(second)
            return True
        except ValueError:
            pass
        try:
            second.relative_to(first)
            return True
        except ValueError:
            return False

    return (
        legacy_migration_enabled
        or reads_prior_schema_as_formal_input
        or report_only_scans_prior_schema
        or any(overlaps(result, path) for path in prior)
    )


def planned_final_holdout_access(
    *,
    phase: str,
    stage: str | None,
    final_holdout_report: bool,
    reads_holdout_lock: bool,
) -> bool:
    """Record planned final-holdout access instead of hard-coding false."""
    return (
        phase == "final_holdout"
        or stage == "final_holdout"
        or final_holdout_report
        or reads_holdout_lock
    )


def _generation_summary(
    audits: Mapping[str, PartitionGenerationAudit],
) -> tuple[dict[str, str | None], int, int, tuple[str, ...], bool]:
    current = {key: audit.current_generation_id for key, audit in sorted(audits.items())}
    orphan_count = sum(len(audit.orphan_generation_ids) for audit in audits.values())
    invalid_count = sum(len(audit.invalid_generation_ids) for audit in audits.values())
    conflicts: list[str] = []
    for key, audit in audits.items():
        if not audit.current_is_valid:
            conflicts.append(f"{key}: CURRENT is invalid or missing while generations exist")
    return current, orphan_count, invalid_count, tuple(sorted(conflicts)), not conflicts


def build_experiment_preflight(
    *,
    project_root: str | Path,
    history_path: str | Path,
    draws: pd.DataFrame,
    specifications: Sequence[ExperimentSpec],
    run_context: RunContextIdentity,
    identities: Mapping[str, ExperimentExecutionIdentity],
    cohort: CohortDefinitionIdentity,
    statuses: Mapping[tuple[str, int], ExperimentPartitionStatusV4],
    task_chunk_plan: Sequence[Mapping[str, object]],
    result_output_paths: Sequence[str | Path],
    inference_context: str,
    allow_dirty: bool,
    authorized_generated_paths: Sequence[AuthorizedGeneratedPath] = (),
    git_status_entries: Sequence[GitStatusEntry] | None = None,
    generation_audits: Mapping[str, PartitionGenerationAudit] | None = None,
    storage_schema_version: str = STORAGE_SCHEMA_VERSION_V4,
    results_root: str | Path | None = None,
    legacy_migration_enabled: bool = False,
    reads_prior_schema_as_formal_input: bool = False,
    report_only_scans_prior_schema: bool = False,
    stage: str | None = None,
    final_holdout_report: bool = False,
    reads_holdout_lock: bool = False,
    identity_conflicts: Sequence[str] = (),
) -> ExperimentPreflightPlan:
    """Build a truthful read-only gate without importing generation code."""
    if not specifications:
        raise ValueError("preflight requires at least one experiment specification")
    source = Path(history_path)
    manifest = VerifiedHistoryManifest.model_validate_json(
        verified_manifest_path(source).read_text(encoding="utf-8")
    )
    history_hash = canonical_history_sha256(draws)
    history_valid = history_hash == manifest.canonical_history_sha256
    common_context = run_context.payload
    versions_valid = all(
        spec.experiment_version == f"v0.5.5-{spec.experiment_id.lower()}-shared-bank-v2"
        if spec.experiment_id in {f"B{index}" for index in range(1, 7)}
        else bool(spec.experiment_version)
        for spec in specifications
    )
    run_context_valid = all(
        identity.run_context_sha256 == run_context.run_context_sha256
        for identity in identities.values()
    ) and len(identities) == len(specifications)
    cohort_valid = cohort.payload.phase == common_context.phase
    git_audit: GitWorktreeAudit = audit_git_worktree(
        project_root,
        authorized_generated_paths,
        status_entries=git_status_entries,
    )
    source_changes = (
        *git_audit.tracked_source_changes,
        *git_audit.untracked_source_changes,
        *git_audit.staged_source_changes,
    )
    git_allowed = git_audit.git_worktree_clean_for_formal_execution or (
        inference_context == "smoke" and allow_dirty and not source_changes
    )
    completed = {
        f"{experiment_id}:seed_{seed}": len(status.completed_target_issues)
        for (experiment_id, seed), status in statuses.items()
    }
    pending = {
        f"{experiment_id}:seed_{seed}": len(status.pending_target_issues)
        for (experiment_id, seed), status in statuses.items()
    }
    audits = dict(generation_audits or {})
    current_generations, orphan_count, invalid_count, generation_conflicts, currents_valid = (
        _generation_summary(audits)
    )
    conflicts = tuple(sorted(set((*identity_conflicts, *generation_conflicts))))
    active_results_root = Path(results_root or Path(project_root) / "outputs/experiments/schema_v4")
    touches_prior = touches_prior_schema(
        project_root=project_root,
        results_root=active_results_root,
        legacy_migration_enabled=legacy_migration_enabled,
        reads_prior_schema_as_formal_input=reads_prior_schema_as_formal_input,
        report_only_scans_prior_schema=report_only_scans_prior_schema,
    )
    reads_holdout = planned_final_holdout_access(
        phase=cohort.payload.phase,
        stage=stage,
        final_holdout_report=final_holdout_report,
        reads_holdout_lock=reads_holdout_lock,
    )
    holdout_boundary_valid = cohort.payload.phase != "final_holdout" or (
        common_context.evaluation_mode == "full_resampling"
        and cohort.payload.cohort_purpose == "final_holdout"
    )
    schema_valid = storage_schema_version == STORAGE_SCHEMA_VERSION_V4
    reasons: list[str] = []
    gates = (
        (history_valid, "frozen history manifest or canonical SHA failed"),
        (versions_valid, "ExperimentSpec version check failed"),
        (run_context_valid, "run context or execution identities are inconsistent"),
        (cohort_valid, "cohort identity does not match the run phase"),
        (schema_valid, "formal output is not schema_v4"),
        (not conflicts, "schema_v4 partition identity conflict exists"),
        (currents_valid, "one or more CURRENT pointers are invalid"),
        (git_allowed, "blocking Git worktree changes exist"),
        (not touches_prior, "formal execution touches legacy or a prior schema"),
        (holdout_boundary_valid, "final holdout boundary or evaluation mode is invalid"),
    )
    for passed, reason in gates:
        if not passed:
            reasons.append(reason)
    task_plan = tuple(dict(item) for item in task_chunk_plan)
    output_paths = tuple(str(Path(path)) for path in result_output_paths)
    ready = not reasons
    executable_task_count = len(task_plan) if ready else 0
    checks = (
        PreflightCheck(
            name="git_worktree_completely_clean",
            passed=git_audit.git_worktree_completely_clean,
            details=str(git_audit.git_worktree_completely_clean),
        ),
        PreflightCheck(
            name="git_worktree_clean_for_formal_execution",
            passed=git_allowed,
            details=str(git_audit.git_worktree_clean_for_formal_execution),
        ),
        PreflightCheck(
            name="authorized_dirty_paths",
            passed=True,
            details=str(git_audit.authorized_dirty_paths),
        ),
        PreflightCheck(
            name="blocking_dirty_paths",
            passed=git_allowed,
            details=str(git_audit.blocking_dirty_paths),
        ),
        PreflightCheck(
            name="git_commit_sha",
            passed=current_git_commit_sha() != "0" * 40,
            details=current_git_commit_sha(),
        ),
        PreflightCheck(
            name="verified_history",
            passed=history_valid,
            details=manifest.schema_version,
        ),
        PreflightCheck(
            name="canonical_history_sha256",
            passed=history_valid,
            details=history_hash,
        ),
        PreflightCheck(
            name="phase_and_split_boundaries",
            passed=cohort_valid,
            details=cohort.payload.phase,
        ),
        PreflightCheck(
            name="experiment_versions",
            passed=versions_valid,
            details=",".join(
                f"{spec.experiment_id}:{spec.experiment_version}" for spec in specifications
            ),
        ),
        PreflightCheck(
            name="run_context",
            passed=run_context_valid,
            details=run_context.run_context_sha256,
        ),
        PreflightCheck(
            name="cohort_identity",
            passed=cohort_valid,
            details=cohort.cohort_definition_sha256,
        ),
        PreflightCheck(
            name="storage_schema_version",
            passed=schema_valid,
            details=storage_schema_version,
        ),
        PreflightCheck(
            name="current_generations",
            passed=currents_valid,
            details=str(current_generations),
        ),
        PreflightCheck(
            name="orphan_generations",
            passed=True,
            details=str(orphan_count),
        ),
        PreflightCheck(
            name="invalid_generations",
            passed=True,
            details=str(invalid_count),
        ),
        PreflightCheck(
            name="identity_conflicts",
            passed=not conflicts,
            details=";".join(conflicts) or "none",
        ),
        PreflightCheck(
            name="touches_legacy_or_prior_schema",
            passed=not touches_prior,
            details=str(touches_prior),
        ),
        PreflightCheck(
            name="reads_final_holdout_results",
            passed=holdout_boundary_valid,
            details=str(reads_holdout),
        ),
        PreflightCheck(
            name="evaluation_mode",
            passed=holdout_boundary_valid,
            details=common_context.evaluation_mode,
        ),
        PreflightCheck(name="profile", passed=True, details=common_context.profile),
        PreflightCheck(
            name="portfolio_scoring_method",
            passed=True,
            details=common_context.portfolio_scoring_method,
        ),
        PreflightCheck(
            name="execution_config_sha256",
            passed=run_context_valid,
            details=str(len(identities)),
        ),
        PreflightCheck(
            name="expected_targets_sha256",
            passed=True,
            details=cohort.payload.expected_targets_sha256,
        ),
        PreflightCheck(
            name="expected_target_count",
            passed=cohort.payload.target_count > 0,
            details=str(cohort.payload.target_count),
        ),
        PreflightCheck(
            name="cohort_start_end",
            passed=int(cohort.payload.start_issue) <= int(cohort.payload.end_issue),
            details=f"{cohort.payload.start_issue}-{cohort.payload.end_issue}",
        ),
        PreflightCheck(
            name="completed_target_counts",
            passed=True,
            details=str(completed),
        ),
        PreflightCheck(
            name="pending_target_counts",
            passed=True,
            details=str(pending),
        ),
        PreflightCheck(
            name="task_chunk_plan",
            passed=True,
            details=str(len(task_plan)),
        ),
        PreflightCheck(
            name="result_output_paths",
            passed=bool(output_paths),
            details=";".join(output_paths),
        ),
        PreflightCheck(
            name="estimated_task_count",
            passed=True,
            details=str(executable_task_count),
        ),
    )
    return ExperimentPreflightPlan(
        storage_schema_version=storage_schema_version,
        inference_context=inference_context,
        allow_dirty=allow_dirty,
        git_dirty=not git_audit.git_worktree_completely_clean,
        git_worktree_completely_clean=git_audit.git_worktree_completely_clean,
        git_worktree_clean_for_formal_execution=(git_audit.git_worktree_clean_for_formal_execution),
        authorized_dirty_path_count=git_audit.authorized_dirty_path_count,
        blocking_dirty_path_count=git_audit.blocking_dirty_path_count,
        authorized_dirty_paths=git_audit.authorized_dirty_paths,
        blocking_dirty_paths=git_audit.blocking_dirty_paths,
        git_commit_sha=current_git_commit_sha(),
        history_path=str(source.resolve()),
        canonical_history_sha256=history_hash,
        phase=cohort.payload.phase,
        split_boundaries={
            phase: boundary.model_dump(mode="json")
            for phase, boundary in common_context.split_boundaries.items()
        },
        experiment_versions={
            spec.experiment_id: spec.experiment_version for spec in specifications
        },
        evaluation_mode=common_context.evaluation_mode,
        profile=common_context.profile,
        portfolio_scoring_method=common_context.portfolio_scoring_method,
        run_context_sha256=run_context.run_context_sha256,
        execution_config_sha256={
            experiment_id: identity.execution_config_sha256
            for experiment_id, identity in identities.items()
        },
        cohort_id=cohort.payload.cohort_id,
        cohort_definition_sha256=cohort.cohort_definition_sha256,
        expected_targets_sha256=cohort.payload.expected_targets_sha256,
        expected_target_count=cohort.payload.target_count,
        start_issue=cohort.payload.start_issue,
        end_issue=cohort.payload.end_issue,
        completed_target_counts=completed,
        pending_target_counts=pending,
        current_generation_by_partition=current_generations,
        orphan_generation_count=orphan_count,
        invalid_generation_count=invalid_count,
        task_chunk_plan=task_plan,
        result_output_paths=output_paths,
        identity_conflicts=conflicts,
        touches_legacy_or_prior_schema=touches_prior,
        reads_final_holdout_results=reads_holdout,
        estimated_task_count=executable_task_count,
        checks=checks,
        ready=ready,
        execution_will_be_blocked=not ready,
        execution_block_reasons=tuple(reasons),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_experiment_preflight(
    plan: ExperimentPreflightPlan,
    root: str | Path,
) -> tuple[Path, Path]:
    """Atomically write JSON and Markdown beneath exact run/cohort identities."""
    directory = Path(root) / plan.run_context_sha256 / plan.cohort_definition_sha256
    json_path = directory / "preflight.json"
    markdown_path = directory / "preflight.md"
    _atomic_write_text(json_path, plan.model_dump_json(indent=2) + "\n")
    lines = [
        "# Experiment preflight v2",
        "",
        f"- Ready: `{str(plan.ready).lower()}`",
        f"- Execution blocked: `{str(plan.execution_will_be_blocked).lower()}`",
        f"- Block reasons: `{list(plan.execution_block_reasons)}`",
        f"- Storage schema: `{plan.storage_schema_version}`",
        f"- Phase: `{plan.phase}`",
        f"- Cohort: `{plan.cohort_id}`",
        f"- Run context: `{plan.run_context_sha256}`",
        f"- Cohort definition: `{plan.cohort_definition_sha256}`",
        f"- Expected targets: `{plan.expected_target_count}`",
        f"- Completely clean: `{plan.git_worktree_completely_clean}`",
        f"- Clean for formal execution: `{plan.git_worktree_clean_for_formal_execution}`",
        f"- Authorized dirty paths: `{list(plan.authorized_dirty_paths)}`",
        f"- Blocking dirty paths: `{list(plan.blocking_dirty_paths)}`",
        f"- Orphan generations: `{plan.orphan_generation_count}`",
        f"- Invalid generations: `{plan.invalid_generation_count}`",
        f"- Estimated tasks: `{plan.estimated_task_count}`",
        "",
        "## Checks",
        "",
        *[
            f"- {'PASS' if check.passed else 'FAIL'} `{check.name}`: {check.details}"
            for check in plan.checks
        ],
        "",
        DISCLAIMER,
    ]
    _atomic_write_text(markdown_path, "\n".join(lines) + "\n")
    return json_path, markdown_path
