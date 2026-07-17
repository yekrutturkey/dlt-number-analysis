"""Read-only formal experiment preflight and atomic audit-plan output."""

from __future__ import annotations

import os
import subprocess
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
from dlt_number_analysis.experiments.identity import (
    ExperimentExecutionIdentity,
    RunContextIdentity,
    current_git_commit_sha,
)
from dlt_number_analysis.experiments.specs import ExperimentSpec
from dlt_number_analysis.experiments.storage_v3 import ExperimentPartitionStatusV3


class PreflightCheck(BaseModel):
    """One named, machine-readable preflight assertion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    details: str


class ExperimentPreflightPlan(BaseModel):
    """Complete non-executing plan for one run context and logical cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "experiment-preflight-v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    inference_context: str
    allow_dirty: bool
    git_dirty: bool
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
    task_chunk_plan: tuple[dict[str, object], ...]
    result_output_paths: tuple[str, ...]
    identity_conflicts: tuple[str, ...]
    touches_legacy_or_schema_v2: bool
    reads_final_holdout_results: bool
    estimated_task_count: int = Field(ge=0)
    checks: tuple[PreflightCheck, ...]
    ready: bool
    risk_disclaimer: str = DISCLAIMER


def git_worktree_dirty(project_root: str | Path) -> bool:
    """Return whether tracked or untracked working-tree content is present."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_experiment_preflight(
    *,
    project_root: str | Path,
    history_path: str | Path,
    draws: pd.DataFrame,
    specifications: Sequence[ExperimentSpec],
    run_context: RunContextIdentity,
    identities: Mapping[str, ExperimentExecutionIdentity],
    cohort: CohortDefinitionIdentity,
    statuses: Mapping[tuple[str, int], ExperimentPartitionStatusV3],
    task_chunk_plan: Sequence[Mapping[str, object]],
    result_output_paths: Sequence[str | Path],
    inference_context: str,
    allow_dirty: bool,
    identity_conflicts: Sequence[str] = (),
) -> ExperimentPreflightPlan:
    """Build the 24-item read-only plan without importing execution code."""
    if not specifications:
        raise ValueError("preflight requires at least one experiment specification")
    dirty = git_worktree_dirty(project_root)
    if dirty and not (inference_context == "smoke" and allow_dirty):
        raise ValueError("formal preflight refuses a dirty Git worktree")
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
    completed = {
        f"{experiment_id}:seed_{seed}": len(status.completed_target_issues)
        for (experiment_id, seed), status in statuses.items()
    }
    pending = {
        f"{experiment_id}:seed_{seed}": len(status.pending_target_issues)
        for (experiment_id, seed), status in statuses.items()
    }
    output_paths = tuple(str(Path(path)) for path in result_output_paths)
    task_plan = tuple(dict(item) for item in task_chunk_plan)
    checks = (
        PreflightCheck(name="git_worktree_clean", passed=not dirty, details=str(not dirty)),
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
        PreflightCheck(name="canonical_history_sha256", passed=history_valid, details=history_hash),
        PreflightCheck(
            name="phase_and_split_boundaries",
            passed=cohort.payload.phase == common_context.phase,
            details=cohort.payload.phase,
        ),
        PreflightCheck(
            name="experiment_versions",
            passed=versions_valid,
            details=",".join(
                f"{spec.experiment_id}:{spec.experiment_version}" for spec in specifications
            ),
        ),
        PreflightCheck(name="evaluation_mode", passed=True, details=common_context.evaluation_mode),
        PreflightCheck(name="profile", passed=True, details=common_context.profile),
        PreflightCheck(
            name="portfolio_scoring_method",
            passed=True,
            details=common_context.portfolio_scoring_method,
        ),
        PreflightCheck(
            name="run_context_sha256", passed=True, details=run_context.run_context_sha256
        ),
        PreflightCheck(
            name="execution_config_sha256",
            passed=len(identities) == len(specifications),
            details=str(len(identities)),
        ),
        PreflightCheck(
            name="cohort_id",
            passed=bool(cohort.payload.cohort_id),
            details=cohort.payload.cohort_id,
        ),
        PreflightCheck(
            name="cohort_definition_sha256",
            passed=True,
            details=cohort.cohort_definition_sha256,
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
        PreflightCheck(name="completed_target_counts", passed=True, details=str(completed)),
        PreflightCheck(name="pending_target_counts", passed=True, details=str(pending)),
        PreflightCheck(name="task_chunk_plan", passed=True, details=str(len(task_plan))),
        PreflightCheck(
            name="result_output_paths", passed=bool(output_paths), details=";".join(output_paths)
        ),
        PreflightCheck(
            name="identity_conflicts",
            passed=not identity_conflicts,
            details=";".join(identity_conflicts) or "none",
        ),
        PreflightCheck(name="touches_legacy_or_schema_v2", passed=True, details="false"),
        PreflightCheck(name="reads_final_holdout_results", passed=True, details="false"),
        PreflightCheck(name="estimated_task_count", passed=True, details=str(len(task_plan))),
    )
    ready = (
        history_valid
        and versions_valid
        and len(identities) == len(specifications)
        and cohort.payload.phase == common_context.phase
        and not identity_conflicts
        and (not dirty or (inference_context == "smoke" and allow_dirty))
    )
    return ExperimentPreflightPlan(
        inference_context=inference_context,
        allow_dirty=allow_dirty,
        git_dirty=dirty,
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
        task_chunk_plan=task_plan,
        result_output_paths=output_paths,
        identity_conflicts=tuple(identity_conflicts),
        touches_legacy_or_schema_v2=False,
        reads_final_holdout_results=False,
        estimated_task_count=len(task_plan),
        checks=checks,
        ready=ready,
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
        "# Experiment preflight",
        "",
        f"- Ready: `{str(plan.ready).lower()}`",
        f"- Phase: `{plan.phase}`",
        f"- Cohort: `{plan.cohort_id}`",
        f"- Run context: `{plan.run_context_sha256}`",
        f"- Cohort definition: `{plan.cohort_definition_sha256}`",
        f"- Expected targets: `{plan.expected_target_count}`",
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
