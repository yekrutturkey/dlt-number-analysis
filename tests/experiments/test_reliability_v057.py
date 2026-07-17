"""Bounded v0.5.7 Git-audit, CLI-mode, and run-lock reliability tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlt_number_analysis.experiments.command import build_parser
from dlt_number_analysis.experiments.git_audit import (
    AuthorizedGeneratedPath,
    GitStatusEntry,
    audit_git_worktree,
)
from dlt_number_analysis.experiments.preflight import (
    planned_final_holdout_access,
    touches_prior_schema,
)
from dlt_number_analysis.experiments.run_lock import (
    FormalRunLock,
    StaleRunLockClearAudit,
    clear_stale_run_lock,
)


def _authorized_outputs(tmp_path: Path) -> tuple[AuthorizedGeneratedPath, ...]:
    return (
        AuthorizedGeneratedPath(
            path=tmp_path / "outputs/experiments/schema_v4",
            kind="directory",
        ),
        AuthorizedGeneratedPath(
            path=tmp_path / "outputs/formal_runs/run/cohort/logs",
            kind="directory",
        ),
    )


def test_authorized_generated_artifacts_allow_formal_resume(tmp_path: Path) -> None:
    audit = audit_git_worktree(
        tmp_path,
        _authorized_outputs(tmp_path),
        status_entries=(
            GitStatusEntry(
                path="outputs/experiments/schema_v4/development/B1/CURRENT",
                index_status="?",
                worktree_status="?",
            ),
            GitStatusEntry(
                path="outputs/formal_runs/run/cohort/logs/run.json",
                index_status="?",
                worktree_status="?",
            ),
        ),
    )
    assert not audit.git_worktree_completely_clean
    assert audit.git_worktree_clean_for_formal_execution
    assert audit.authorized_dirty_path_count == 2
    assert audit.blocking_dirty_path_count == 0


@pytest.mark.parametrize(
    "path",
    (
        "src/changed.py",
        "tests/test_changed.py",
        "scripts/run.py",
        "config/prize_tiers.json",
        "data/raw/draws.csv",
        "uv.lock",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        ".gitignore",
    ),
)
def test_source_config_and_frozen_data_changes_always_block(
    tmp_path: Path,
    path: str,
) -> None:
    audit = audit_git_worktree(
        tmp_path,
        _authorized_outputs(tmp_path),
        status_entries=(GitStatusEntry(path=path, index_status=" ", worktree_status="M"),),
    )
    assert not audit.git_worktree_clean_for_formal_execution
    assert audit.blocking_dirty_paths == (path,)
    assert path in audit.tracked_source_changes


def test_untracked_unapproved_path_and_staged_source_are_reported(tmp_path: Path) -> None:
    audit = audit_git_worktree(
        tmp_path,
        _authorized_outputs(tmp_path),
        status_entries=(
            GitStatusEntry(path="notes.txt", index_status="?", worktree_status="?"),
            GitStatusEntry(path="src/staged.py", index_status="A", worktree_status=" "),
        ),
    )
    assert audit.blocking_dirty_paths == ("notes.txt", "src/staged.py")
    assert audit.untracked_source_changes == ()
    assert audit.staged_source_changes == ("src/staged.py",)


def test_authorization_cannot_cover_project_root_or_prior_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project root"):
        audit_git_worktree(
            tmp_path,
            (AuthorizedGeneratedPath(path=tmp_path, kind="directory"),),
            status_entries=(),
        )
    with pytest.raises(ValueError, match="prior-schema"):
        audit_git_worktree(
            tmp_path,
            (
                AuthorizedGeneratedPath(
                    path=tmp_path / "outputs/experiments",
                    kind="directory",
                ),
            ),
            status_entries=(),
        )


def test_prior_schema_and_holdout_flags_are_computed_from_real_inputs(
    tmp_path: Path,
) -> None:
    assert touches_prior_schema(
        project_root=tmp_path,
        results_root=tmp_path / "outputs/experiments/schema_v3",
    )
    assert touches_prior_schema(
        project_root=tmp_path,
        results_root=tmp_path / "outputs/experiments/schema_v4",
        legacy_migration_enabled=True,
    )
    assert not touches_prior_schema(
        project_root=tmp_path,
        results_root=tmp_path / "outputs/experiments/schema_v4",
    )
    assert not planned_final_holdout_access(
        phase="development",
        stage=None,
        final_holdout_report=False,
        reads_holdout_lock=False,
    )
    assert planned_final_holdout_access(
        phase="final_holdout",
        stage="final_holdout",
        final_holdout_report=True,
        reads_holdout_lock=True,
    )


def test_cli_operational_modes_are_mutually_exclusive_and_repair_fields_parse() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--report-only", "--repair-current"])
    args = parser.parse_args(
        [
            "--repair-current",
            "--repair-generation-id",
            "generation-1",
            "--repair-reason",
            "operator selected validated generation",
        ]
    )
    assert args.repair_current
    assert args.repair_generation_id == "generation-1"


def test_run_lock_refuses_second_writer_and_releases_owner(tmp_path: Path) -> None:
    path = tmp_path / "RUNNING.lock"
    first = FormalRunLock(
        path,
        run_context_sha256="1" * 64,
        cohort_definition_sha256="2" * 64,
        phase="development",
        seeds=(20260000,),
        command_summary="formal run",
    )
    first.acquire()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["pid"] > 0
    assert stored["hostname"]
    assert stored["run_context_sha256"] == "1" * 64
    second = FormalRunLock(
        path,
        run_context_sha256="1" * 64,
        cohort_definition_sha256="2" * 64,
        phase="development",
        seeds=(20260000,),
        command_summary="second writer",
    )
    with pytest.raises(RuntimeError, match="second writer"):
        second.acquire()
    assert path.exists()
    first.release()
    assert not path.exists()


def test_stale_lock_is_not_auto_removed_and_explicit_clear_is_audited(
    tmp_path: Path,
) -> None:
    path = tmp_path / "RUNNING.lock"
    path.write_text('{"stale": true}', encoding="utf-8")
    lock = FormalRunLock(
        path,
        run_context_sha256="1" * 64,
        cohort_definition_sha256="2" * 64,
        phase="development",
        seeds=(1,),
        command_summary="resume",
    )
    with pytest.raises(RuntimeError):
        lock.acquire()
    assert path.exists()
    with pytest.raises(ValueError, match="requires a reason"):
        clear_stale_run_lock(path, reason="", audit_directory=tmp_path / "audits")
    audit_path = clear_stale_run_lock(
        path,
        reason="operator verified original process exited",
        audit_directory=tmp_path / "audits",
    )
    audit = StaleRunLockClearAudit.model_validate_json(audit_path.read_text(encoding="utf-8"))
    assert not path.exists()
    assert audit.reason.startswith("operator verified")
