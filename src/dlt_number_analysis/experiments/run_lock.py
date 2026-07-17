"""Atomic run-level writer lock with explicit stale-lock clearing audit."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.identity import current_git_commit_sha
from dlt_number_analysis.experiments.manual_operations import (
    StaleRunLockClearAuditV2,
    atomic_write_manual_audit,
    operation_id,
    operator_hostname,
)


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


StaleRunLockClearAudit = StaleRunLockClearAuditV2

StaleLockClearFaultPoint = Literal[
    "before_prepare_audit_write",
    "after_prepare_audit_commit",
    "before_lock_delete",
    "after_lock_delete",
    "before_completed_audit_update",
]
StaleLockClearFaultInjector = Callable[[StaleLockClearFaultPoint], None]


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inject_clear_fault(
    injector: StaleLockClearFaultInjector | None,
    point: StaleLockClearFaultPoint,
) -> None:
    if injector is not None:
        injector(point)


def _delete_lock(path: Path) -> None:
    path.unlink()


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
    _fault_injector: StaleLockClearFaultInjector | None = None,
) -> Path:
    """Persist prepared intent before removing exactly one named stale lock."""
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
    requested_at = datetime.now(UTC)
    operation = operation_id()
    audit_path = Path(audit_directory) / f"stale_lock_clear_{operation}.json"
    prepared = StaleRunLockClearAuditV2(
        operation_id=operation,
        lock_path=target.resolve(),
        original_lock=original,
        requested_at=requested_at,
        status="prepared",
        reason=reason.strip(),
        git_commit_sha=current_git_commit_sha(),
        operator_hostname=operator_hostname(),
    )
    _inject_clear_fault(_fault_injector, "before_prepare_audit_write")
    atomic_write_manual_audit(audit_path, prepared)
    _inject_clear_fault(_fault_injector, "after_prepare_audit_commit")
    _inject_clear_fault(_fault_injector, "before_lock_delete")
    try:
        _delete_lock(target)
    except OSError as error:
        failed = prepared.model_copy(
            update={
                "status": "failed",
                "failure_message": str(error),
            }
        )
        atomic_write_manual_audit(audit_path, failed)
        raise RuntimeError("stale run lock deletion failed") from error
    _inject_clear_fault(_fault_injector, "after_lock_delete")
    _inject_clear_fault(_fault_injector, "before_completed_audit_update")
    completed = prepared.model_copy(
        update={
            "status": "completed",
            "completed_at": datetime.now(UTC),
        }
    )
    atomic_write_manual_audit(audit_path, completed)
    return audit_path
