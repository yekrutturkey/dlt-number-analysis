"""Contiguous temporal splits and one-shot final-holdout locking."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis.data import validate_draw_dataframe
from dlt_number_analysis.experiments.specs import DataSplitSpec, ExperimentPhase, ExperimentSpec

if TYPE_CHECKING:
    from dlt_number_analysis.experiments.identity import (
        ExperimentExecutionIdentity,
        RunContextIdentity,
    )


@dataclass(frozen=True, slots=True)
class TemporalDataSplit:
    """Three non-overlapping chronological frames."""

    development: pd.DataFrame
    calibration: pd.DataFrame
    final_holdout: pd.DataFrame

    def phase_frame(self, phase: ExperimentPhase) -> pd.DataFrame:
        """Return a copy of exactly one allowed split."""
        return getattr(self, phase).copy()


class HoldoutLockEntry(BaseModel):
    """Configuration identity reserved before a formal final-holdout run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1)
    experiment_version: str = Field(min_length=1)
    experiment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_mode: str
    locked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = "reserved_before_run"
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class HoldoutLockRegistry(BaseModel):
    """Version-indexed final-holdout execution registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "holdout-lock-v2"
    entries: dict[str, HoldoutLockEntry] = Field(default_factory=dict)


def split_history(
    draws: pd.DataFrame,
    spec: DataSplitSpec | None = None,
) -> TemporalDataSplit:
    """Split strictly by row order, never by random assignment."""
    validated = validate_draw_dataframe(draws)
    active = spec or DataSplitSpec()
    count = len(validated)
    development_end = int(count * active.development_ratio)
    calibration_end = development_end + int(count * active.calibration_ratio)
    if development_end < 1 or calibration_end <= development_end or calibration_end >= count:
        raise ValueError("history is too short for three non-empty temporal splits")
    return TemporalDataSplit(
        development=validated.iloc[:development_end].copy(),
        calibration=validated.iloc[development_end:calibration_end].copy(),
        final_holdout=validated.iloc[calibration_end:].copy(),
    )


def experiment_config_sha256(spec: ExperimentSpec) -> str:
    """Hash a canonical JSON representation of every experiment parameter."""
    canonical = json.dumps(
        spec.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _holdout_identity_key(
    spec: ExperimentSpec,
    identity: ExperimentExecutionIdentity,
) -> str:
    return f"{spec.experiment_id}:{spec.experiment_version}:{identity.execution_config_sha256}"


def _write_holdout_registry(path: Path, registry: HoldoutLockRegistry) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            registry.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_holdout_lock(
    spec: ExperimentSpec,
    path: str | Path,
    *,
    identity: ExperimentExecutionIdentity,
    run_context: RunContextIdentity,
) -> HoldoutLockEntry:
    """Reserve one complete full-resampling execution identity before holdout access."""
    if spec.data_split.phase != "final_holdout":
        raise ValueError("a holdout lock is only valid for final_holdout experiments")
    if run_context.payload.evaluation_mode != "full_resampling":
        raise ValueError("formal final holdout requires full_resampling evaluation")
    if identity.run_context_sha256 != run_context.run_context_sha256:
        raise ValueError("holdout execution identity differs from run context")
    target = Path(path)
    if target.exists():
        registry = HoldoutLockRegistry.model_validate_json(target.read_text(encoding="utf-8"))
    else:
        registry = HoldoutLockRegistry()
    key = _holdout_identity_key(spec, identity)
    if key in registry.entries:
        raise ValueError("this complete execution identity already consumed its formal holdout run")
    same_version = [
        entry
        for entry in registry.entries.values()
        if entry.experiment_id == spec.experiment_id
        and entry.experiment_version == spec.experiment_version
    ]
    if same_version:
        raise ValueError("changed final-holdout parameters must use a new experiment version")
    entry = HoldoutLockEntry(
        experiment_id=spec.experiment_id,
        experiment_version=spec.experiment_version,
        experiment_config_sha256=identity.experiment_config_sha256,
        run_context_sha256=identity.run_context_sha256,
        execution_config_sha256=identity.execution_config_sha256,
        history_sha256=run_context.payload.history_sha256,
        evaluation_mode=run_context.payload.evaluation_mode,
    )
    updated = registry.model_copy(update={"entries": {**registry.entries, key: entry}})
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_holdout_registry(target, updated)
    return entry


def finalize_holdout_lock(
    path: str | Path,
    *,
    spec: ExperimentSpec,
    identity: ExperimentExecutionIdentity,
    result_bytes: bytes,
) -> HoldoutLockEntry:
    """Bind the already-reserved holdout entry to the immutable result hash."""
    target = Path(path)
    registry = HoldoutLockRegistry.model_validate_json(target.read_text(encoding="utf-8"))
    key = _holdout_identity_key(spec, identity)
    current = registry.entries.get(key)
    if current is None:
        raise ValueError("final holdout was not reserved before the run")
    if current.result_sha256 is not None:
        raise ValueError("final holdout result was already finalized")
    finalized = current.model_copy(
        update={
            "status": "completed",
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
    )
    updated = registry.model_copy(update={"entries": {**registry.entries, key: finalized}})
    _write_holdout_registry(target, updated)
    return finalized
