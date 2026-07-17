"""Atomic run-level writer lock with explicit stale-lock clearing audit."""

from __future__ import annotations

import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.identity import current_git_commit_sha


class RunLockRecord(BaseModel):
    """Identity of the only allowed writer for one formal logical run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "formal-run-lock-v1"
    lock_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    pid: int = Field(ge=1)
    hostname: str
    started_at: datetime
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cohort_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase: str
    seeds: tuple[int, ...]
    command_summary: str
    risk_disclaimer: str = DISCLAIMER


class StaleRunLockClearAudit(BaseModel):
    """Permanent evidence of one explicitly authorized stale-lock removal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "stale-run-lock-clear-v1"
    original_lock: dict[str, object] | str
    cleared_at: datetime
    reason: str = Field(min_length=1)
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    risk_disclaimer: str = DISCLAIMER


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class FormalRunLock:
    """Context manager that never removes a pre-existing or mismatched lock."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_context_sha256: str,
        cohort_definition_sha256: str,
        phase: str,
        seeds: tuple[int, ...],
        command_summary: str,
    ) -> None:
        self.path = Path(path)
        self.record = RunLockRecord(
            lock_id=uuid4().hex,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            started_at=datetime.now(UTC),
            git_commit_sha=current_git_commit_sha(),
            run_context_sha256=run_context_sha256,
            cohort_definition_sha256=cohort_definition_sha256,
            phase=phase,
            seeds=tuple(sorted(set(seeds))),
            command_summary=command_summary,
        )
        self._acquired = False

    def acquire(self) -> RunLockRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_new_file(
                self.path,
                (self.record.model_dump_json(indent=2) + "\n").encode("utf-8"),
            )
        except FileExistsError as error:
            raise RuntimeError(
                "formal run lock already exists; a second writer was not started"
            ) from error
        self._acquired = True
        return self.record

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            current = RunLockRecord.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError("formal run lock changed while held; refusing removal") from error
        if current.lock_id != self.record.lock_id:
            raise RuntimeError("formal run lock ownership changed; refusing removal")
        self.path.unlink()
        self._acquired = False

    def __enter__(self) -> RunLockRecord:
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def clear_stale_run_lock(
    path: str | Path,
    *,
    reason: str,
    audit_directory: str | Path,
) -> Path:
    """Remove exactly one named lock only after explicit reason and write an audit."""
    if not reason.strip():
        raise ValueError("stale run lock clearing requires a reason")
    target = Path(path)
    if not target.is_file():
        raise ValueError("stale run lock does not exist")
    raw = target.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
        original: dict[str, object] | str = parsed if isinstance(parsed, dict) else raw
    except json.JSONDecodeError:
        original = raw
    target.unlink()
    cleared_at = datetime.now(UTC)
    audit = StaleRunLockClearAudit(
        original_lock=original,
        cleared_at=cleared_at,
        reason=reason.strip(),
        git_commit_sha=current_git_commit_sha(),
    )
    directory = Path(audit_directory)
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = directory / (
        f"stale_lock_clear_{cleared_at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex}.json"
    )
    _write_new_file(
        audit_path,
        (audit.model_dump_json(indent=2) + "\n").encode("utf-8"),
    )
    return audit_path
