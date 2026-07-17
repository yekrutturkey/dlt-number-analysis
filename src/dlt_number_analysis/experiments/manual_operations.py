"""Durable audit records for explicit stale-lock clearing and CURRENT repair."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER

ManualOperationStatus = Literal["prepared", "completed", "failed"]
ManualOperationType = Literal[
    "stale_lock_clear",
    "current_pointer_repair",
    "unknown",
]


class StaleRunLockClearAuditV2(BaseModel):
    """Two-phase evidence for one explicitly requested stale-lock removal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["stale-run-lock-clear-v2"] = "stale-run-lock-clear-v2"
    operation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    lock_path: Path
    original_lock: dict[str, object] | str
    requested_at: datetime
    completed_at: datetime | None = None
    status: ManualOperationStatus
    reason: str = Field(min_length=1)
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    operator_hostname: str
    failure_message: str | None = None
    risk_disclaimer: str = DISCLAIMER


class CurrentPointerRepairAuditV2(BaseModel):
    """Two-phase evidence for one explicit schema-v4 CURRENT replacement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["schema-v4-current-repair-v2"] = "schema-v4-current-repair-v2"
    operation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    partition_path: Path
    original_current: dict[str, object] | str | None
    original_current_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    requested_generation_id: str
    validated_generation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposed_current: dict[str, object]
    requested_at: datetime
    completed_at: datetime | None = None
    status: ManualOperationStatus
    reason: str = Field(min_length=1)
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    operator_hostname: str
    failure_message: str | None = None
    risk_disclaimer: str = DISCLAIMER


class ManualOperationSummary(BaseModel):
    """Read-only classification of one manual-operation audit file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str
    operation_type: ManualOperationType
    status: ManualOperationStatus
    audit_path: Path
    inconsistency: str | None = None


class ManualOperationsAuditReport(BaseModel):
    """Prepared, completed, failed, and inconsistent manual operations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "manual-operations-audit-v1"
    prepared_operations: tuple[ManualOperationSummary, ...]
    completed_operations: tuple[ManualOperationSummary, ...]
    failed_operations: tuple[ManualOperationSummary, ...]
    inconsistent_operations: tuple[ManualOperationSummary, ...]
    warnings: tuple[str, ...]
    risk_disclaimer: str = DISCLAIMER


def operation_id() -> str:
    """Return a collision-resistant identifier for one explicit operation."""
    return uuid4().hex


def operator_hostname() -> str:
    """Return the local host recorded as the operation initiator."""
    return socket.gethostname()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_manual_audit(path: str | Path, audit: BaseModel) -> Path:
    """Atomically create or update one audit while retaining the old version on failure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    payload = (audit.model_dump_json(indent=2) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _summary(
    *,
    operation_id_value: str,
    operation_type: ManualOperationType,
    status: ManualOperationStatus,
    path: Path,
    inconsistency: str | None = None,
) -> ManualOperationSummary:
    return ManualOperationSummary(
        operation_id=operation_id_value,
        operation_type=operation_type,
        status=status,
        audit_path=path,
        inconsistency=inconsistency,
    )


def _current_generation_id(partition: Path) -> tuple[str | None, str | None]:
    current = partition / "CURRENT"
    if not current.exists():
        return None, None
    try:
        payload = json.loads(current.read_text(encoding="utf-8"))
        generation = payload.get("generation_id") if isinstance(payload, dict) else None
        if not isinstance(generation, str):
            raise ValueError("CURRENT does not contain generation_id")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"CURRENT cannot be parsed: {error}"
    return generation, None


def _candidate_audit_paths(roots: Sequence[str | Path]) -> tuple[Path, ...]:
    found: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if root.is_file():
            found.add(root.resolve())
            continue
        if not root.exists():
            continue
        for pattern in ("stale_lock_clear_*.json", "repair_*.json"):
            found.update(path.resolve() for path in root.rglob(pattern) if path.is_file())
    return tuple(sorted(found))


def audit_manual_operations(
    audit_roots: Sequence[str | Path],
) -> ManualOperationsAuditReport:
    """Classify manual-operation audits without repairing or deleting anything."""
    prepared: list[ManualOperationSummary] = []
    completed: list[ManualOperationSummary] = []
    failed: list[ManualOperationSummary] = []
    inconsistent: list[ManualOperationSummary] = []
    warnings: list[str] = []
    for path in _candidate_audit_paths(audit_roots):
        try:
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            schema = payload.get("schema_version") if isinstance(payload, dict) else None
            if schema == "stale-run-lock-clear-v2":
                record = StaleRunLockClearAuditV2.model_validate(payload)
                inconsistency = None
                lock_exists = record.lock_path.exists()
                if record.status == "prepared" and not lock_exists:
                    inconsistency = "prepared stale-lock clear has no remaining lock"
                elif record.status == "completed" and lock_exists:
                    inconsistency = "completed stale-lock clear still has a lock"
                summary = _summary(
                    operation_id_value=record.operation_id,
                    operation_type="stale_lock_clear",
                    status=record.status,
                    path=path,
                    inconsistency=inconsistency,
                )
            elif schema == "schema-v4-current-repair-v2":
                record = CurrentPointerRepairAuditV2.model_validate(payload)
                generation, parse_warning = _current_generation_id(record.partition_path)
                requested = record.requested_generation_id
                inconsistency = parse_warning
                if record.status == "prepared" and generation == requested:
                    inconsistency = "prepared CURRENT repair already points to proposed generation"
                elif record.status == "completed" and generation != requested:
                    inconsistency = (
                        "completed CURRENT repair does not point to requested generation"
                    )
                summary = _summary(
                    operation_id_value=record.operation_id,
                    operation_type="current_pointer_repair",
                    status=record.status,
                    path=path,
                    inconsistency=inconsistency,
                )
            else:
                warnings.append(f"unsupported manual-operation audit schema: {path}")
                continue
        except (OSError, ValueError, json.JSONDecodeError) as error:
            warning = f"damaged manual-operation audit {path}: {error}"
            warnings.append(warning)
            summary = _summary(
                operation_id_value=f"damaged-{path.stem}",
                operation_type="unknown",
                status="failed",
                path=path,
                inconsistency=warning,
            )
        if summary.status == "prepared":
            prepared.append(summary)
        elif summary.status == "completed":
            completed.append(summary)
        else:
            failed.append(summary)
        if summary.inconsistency is not None:
            inconsistent.append(summary)
    return ManualOperationsAuditReport(
        prepared_operations=tuple(prepared),
        completed_operations=tuple(completed),
        failed_operations=tuple(failed),
        inconsistent_operations=tuple(inconsistent),
        warnings=tuple(warnings),
    )
