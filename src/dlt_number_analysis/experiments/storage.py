"""Partitioned append-only Parquet storage for resumable experiment observations."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.specs import ExperimentPhase, ExperimentSpec

EXPERIMENT_PRIMARY_KEY: tuple[str, ...] = (
    "experiment_id",
    "experiment_version",
    "phase",
    "seed",
    "target_issue",
)


class ExperimentPartitionStatus(BaseModel):
    """Resume state for one experiment/phase/seed partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: ExperimentPhase
    experiment_id: str
    seed: int
    partition_path: Path
    completed_target_issues: tuple[str, ...]
    pending_target_issues: tuple[str, ...]
    is_complete: bool


class DuplicateExperimentResultError(ValueError):
    """A result row attempted to reuse an immutable experiment primary key."""


class ExperimentResultStore:
    """Store logical appends under phase/experiment/seed physical partitions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def partition_path(
        self,
        phase: ExperimentPhase,
        experiment_id: str,
        seed: int,
    ) -> Path:
        """Return outputs/experiments/{phase}/{experiment_id}/seed_{seed}.parquet."""
        if not experiment_id or any(value in experiment_id for value in ("/", "\\", "..")):
            raise ValueError("experiment_id is not safe for a partition path")
        return self.root / phase / experiment_id / f"seed_{seed}.parquet"

    def read_partition(
        self,
        phase: ExperimentPhase,
        experiment_id: str,
        seed: int,
    ) -> pd.DataFrame:
        """Read one partition while preserving issue identifiers as strings."""
        path = self.partition_path(phase, experiment_id, seed)
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path, engine="pyarrow")
        for column in ("target_issue", "data_cutoff_issue"):
            if column in frame:
                frame[column] = frame[column].astype("string")
        return frame

    def status(
        self,
        spec: ExperimentSpec,
        *,
        seed: int,
        expected_target_issues: tuple[str, ...],
    ) -> ExperimentPartitionStatus:
        """Return missing targets so an interrupted task can resume without recomputation."""
        existing = self.read_partition(spec.data_split.phase, spec.experiment_id, seed)
        completed = (
            set(existing["target_issue"].astype(str).tolist()) if not existing.empty else set()
        )
        pending = tuple(issue for issue in expected_target_issues if issue not in completed)
        return ExperimentPartitionStatus(
            phase=spec.data_split.phase,
            experiment_id=spec.experiment_id,
            seed=seed,
            partition_path=self.partition_path(spec.data_split.phase, spec.experiment_id, seed),
            completed_target_issues=tuple(sorted(completed, key=int)),
            pending_target_issues=pending,
            is_complete=not pending,
        )

    def append(self, observations: pd.DataFrame) -> tuple[Path, ...]:
        """Append new primary keys atomically; never replace an existing logical result."""
        missing = set(EXPERIMENT_PRIMARY_KEY).difference(observations.columns)
        if missing:
            raise ValueError(f"experiment observations are missing keys: {sorted(missing)}")
        if observations.empty:
            return ()
        incoming = observations.copy()
        if incoming.duplicated(list(EXPERIMENT_PRIMARY_KEY)).any():
            raise DuplicateExperimentResultError("incoming observations contain duplicate keys")
        incoming["risk_disclaimer"] = DISCLAIMER
        written: list[Path] = []
        grouped = incoming.groupby(["phase", "experiment_id", "seed"], sort=True, dropna=False)
        for (raw_phase, raw_experiment_id, raw_seed), partition in grouped:
            phase = str(raw_phase)
            if phase not in {"development", "calibration", "final_holdout"}:
                raise ValueError(f"unsupported experiment phase: {phase}")
            experiment_id = str(raw_experiment_id)
            seed = int(raw_seed)
            target = self.partition_path(phase, experiment_id, seed)  # type: ignore[arg-type]
            existing = self.read_partition(phase, experiment_id, seed)  # type: ignore[arg-type]
            if not existing.empty:
                duplicate = existing.loc[:, list(EXPERIMENT_PRIMARY_KEY)].merge(
                    partition.loc[:, list(EXPERIMENT_PRIMARY_KEY)],
                    on=list(EXPERIMENT_PRIMARY_KEY),
                    how="inner",
                )
                if not duplicate.empty:
                    raise DuplicateExperimentResultError(
                        f"partition already contains {len(duplicate)} incoming primary key(s)"
                    )
            combined = (
                partition.copy()
                if existing.empty
                else pd.concat([existing, partition], ignore_index=True)
            )
            if combined.duplicated(list(EXPERIMENT_PRIMARY_KEY)).any():
                raise DuplicateExperimentResultError("combined partition contains duplicate keys")
            combined = combined.sort_values("target_issue", key=lambda values: values.astype(int))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp.parquet")
            try:
                combined.to_parquet(temporary, index=False, engine="pyarrow")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            written.append(target)
        return tuple(written)

    def load_all(self, *, phase: ExperimentPhase | None = None) -> pd.DataFrame:
        """Aggregate reports from partition files rather than a mutable result CSV."""
        search_root = self.root if phase is None else self.root / phase
        paths = sorted(search_root.glob("**/seed_*.parquet")) if search_root.exists() else []
        frames = [pd.read_parquet(path, engine="pyarrow") for path in paths]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if combined.duplicated(list(EXPERIMENT_PRIMARY_KEY)).any():
            raise DuplicateExperimentResultError(
                "stored experiment partitions contain duplicate keys"
            )
        for column in ("target_issue", "data_cutoff_issue"):
            if column in combined:
                combined[column] = combined[column].astype("string")
        return combined
