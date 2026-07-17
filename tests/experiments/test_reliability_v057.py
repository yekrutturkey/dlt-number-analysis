"""Bounded v0.5.7 Git-audit, CLI-mode, and run-lock reliability tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dlt_number_analysis.experiments import git_audit as git_audit_module
from dlt_number_analysis.experiments import run_lock as run_lock_module
from dlt_number_analysis.experiments.command import build_parser
from dlt_number_analysis.experiments.git_audit import (
    AuthorizedGeneratedPath,
    GitStatusEntry,
    audit_git_worktree,
    read_git_status_entries,
)
from dlt_number_analysis.experiments.manual_operations import (
    StaleRunLockClearAuditV2,
    audit_manual_operations,
)
from dlt_number_analysis.experiments.preflight import (
    planned_final_holdout_access,
    touches_prior_schema,
)
from dlt_number_analysis.experiments.run_lock import (
    FormalRunLock,
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


def test_real_porcelain_z_bytes_preserve_normal_rename_and_copy_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (
        b" M normal file.txt\0"
        + "?? 未跟踪.txt\0".encode()
        + b"R  outputs/formal/new name.txt\0outputs/formal/old name.txt\0"
        + "C  outputs/formal/复制.txt\0config/源 配置.json\0".encode()
    )
    monkeypatch.setattr(
        git_audit_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=raw),
    )
    entries = read_git_status_entries(tmp_path)
    assert entries[0].destination_path == "normal file.txt"
    assert entries[0].source_path is None
    assert entries[1].destination_path == "未跟踪.txt"
    rename = entries[2]
    assert rename.change_type == "rename"
    assert rename.destination_path == "outputs/formal/new name.txt"
    assert rename.source_path == "outputs/formal/old name.txt"
    copy = entries[3]
    assert copy.change_type == "copy"
    assert copy.destination_path == "outputs/formal/复制.txt"
    assert copy.source_path == "config/源 配置.json"


@pytest.mark.parametrize(
    ("source", "destination", "change_type", "allowed", "blocked_endpoint"),
    (
        ("notes/file.py", "src/file.py", "rename", False, "destination"),
        ("src/file.py", "outputs/formal/file.py", "rename", False, "source"),
        (
            "config/settings.json",
            "outputs/formal/settings.json",
            "copy",
            False,
            "source",
        ),
        (
            "outputs/formal/old.json",
            "outputs/formal/new.json",
            "rename",
            True,
            None,
        ),
        (
            "outputs/formal/old.json",
            "notes/new.json",
            "rename",
            False,
            "destination",
        ),
        (
            "notes/old.json",
            "outputs/formal/new.json",
            "rename",
            False,
            "source",
        ),
    ),
)
def test_rename_copy_audit_checks_both_endpoints(
    tmp_path: Path,
    source: str,
    destination: str,
    change_type: str,
    allowed: bool,
    blocked_endpoint: str | None,
) -> None:
    authorized = (
        AuthorizedGeneratedPath(
            path=tmp_path / "outputs/formal",
            kind="directory",
        ),
    )
    entry = GitStatusEntry(
        destination_path=destination,
        source_path=source,
        index_status="R" if change_type == "rename" else "C",
        worktree_status=" ",
        change_type=change_type,
    )
    audit = audit_git_worktree(tmp_path, authorized, status_entries=(entry,))
    assert audit.all_changed_paths == (f"{source} -> {destination}",)
    assert audit.git_worktree_clean_for_formal_execution is allowed
    if allowed:
        assert audit.authorized_dirty_paths == (f"{source} -> {destination}",)
    else:
        assert blocked_endpoint is not None
        assert any(blocked_endpoint in item for item in audit.blocking_dirty_paths)


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
    with pytest.raises(SystemExit):
        parser.parse_args(["--report-only", "--audit-manual-operations"])
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
    audit = StaleRunLockClearAuditV2.model_validate_json(audit_path.read_text(encoding="utf-8"))
    assert not path.exists()
    assert audit.reason.startswith("operator verified")
    assert audit.status == "completed"
    assert audit.completed_at is not None
    report = audit_manual_operations((tmp_path,))
    assert len(report.completed_operations) == 1
    assert not report.inconsistent_operations
    path.write_text("unexpected replacement lock", encoding="utf-8")
    inconsistent = audit_manual_operations((tmp_path,))
    assert len(inconsistent.inconsistent_operations) == 1


def test_stale_lock_prepare_failure_keeps_lock_and_writes_no_audit(tmp_path: Path) -> None:
    path = tmp_path / "RUNNING.lock"
    path.write_text("stale", encoding="utf-8")

    def fail(point: str) -> None:
        if point == "before_prepare_audit_write":
            raise RuntimeError("prepare failed")

    with pytest.raises(RuntimeError, match="prepare failed"):
        clear_stale_run_lock(
            path,
            reason="confirmed stale",
            audit_directory=tmp_path / "audits",
            _fault_injector=fail,
        )
    assert path.exists()
    assert not tuple((tmp_path / "audits").glob("*.json"))


@pytest.mark.parametrize(
    "fault_point",
    ("after_prepare_audit_commit", "before_lock_delete"),
)
def test_stale_lock_is_deleted_only_after_prepared_audit_commit(
    tmp_path: Path,
    fault_point: str,
) -> None:
    path = tmp_path / "RUNNING.lock"
    path.write_text("stale", encoding="utf-8")

    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("crash after prepare")

    with pytest.raises(RuntimeError, match="crash after prepare"):
        clear_stale_run_lock(
            path,
            reason="confirmed stale",
            audit_directory=tmp_path / "audits",
            _fault_injector=fail,
        )
    assert path.exists()
    audit_path = next((tmp_path / "audits").glob("*.json"))
    audit = StaleRunLockClearAuditV2.model_validate_json(audit_path.read_text(encoding="utf-8"))
    assert audit.status == "prepared"


def test_stale_lock_delete_failure_records_failed_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "RUNNING.lock"
    path.write_text("stale", encoding="utf-8")

    def deny_delete(_: Path) -> None:
        raise OSError("access denied")

    monkeypatch.setattr(run_lock_module, "_delete_lock", deny_delete)
    with pytest.raises(RuntimeError, match="deletion failed"):
        clear_stale_run_lock(
            path,
            reason="confirmed stale",
            audit_directory=tmp_path / "audits",
        )
    assert path.exists()
    audit_path = next((tmp_path / "audits").glob("*.json"))
    audit = StaleRunLockClearAuditV2.model_validate_json(audit_path.read_text(encoding="utf-8"))
    assert audit.status == "failed"
    assert audit.failure_message == "access denied"


@pytest.mark.parametrize("fault_point", ("after_lock_delete", "before_completed_audit_update"))
def test_stale_lock_final_audit_failure_retains_prepared_evidence(
    tmp_path: Path,
    fault_point: str,
) -> None:
    path = tmp_path / "RUNNING.lock"
    path.write_text("stale", encoding="utf-8")

    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("final audit interrupted")

    with pytest.raises(RuntimeError, match="final audit interrupted"):
        clear_stale_run_lock(
            path,
            reason="confirmed stale",
            audit_directory=tmp_path / "audits",
            _fault_injector=fail,
        )
    assert not path.exists()
    audit_path = next((tmp_path / "audits").glob("*.json"))
    audit = StaleRunLockClearAuditV2.model_validate_json(audit_path.read_text(encoding="utf-8"))
    assert audit.status == "prepared"
    report = audit_manual_operations((tmp_path,))
    assert len(report.inconsistent_operations) == 1


def test_manual_operations_audit_reports_damaged_audit_file(tmp_path: Path) -> None:
    damaged = tmp_path / "repair_damaged.json"
    damaged.write_text("{not-json", encoding="utf-8")
    report = audit_manual_operations((tmp_path,))
    assert len(report.failed_operations) == 1
    assert len(report.inconsistent_operations) == 1
    assert report.warnings and "damaged" in report.warnings[0]
