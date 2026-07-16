"""Partitioned append-only Parquet storage for resumable experiment observations."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.identity import (
    EXECUTION_IDENTITY_SCHEMA_VERSION,
    RUNNER_VERSION,
    ExperimentExecutionIdentity,
    RunContextIdentity,
)
from dlt_number_analysis.experiments.specs import ExperimentPhase, ExperimentSpec

EXPERIMENT_PRIMARY_KEY: tuple[str, ...] = (
    "experiment_id",
    "experiment_version",
    "phase",
    "seed",
    "target_issue",
)

STORAGE_SCHEMA_VERSION = "experiment-storage-schema-v2"
EXPERIMENT_PRIMARY_KEY_V2: tuple[str, ...] = (
    "experiment_id",
    "experiment_version",
    "execution_config_sha256",
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


class FormalPartitionManifest(BaseModel):
    """Identity and content hash committed after one formal Parquet partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    storage_schema_version: str = STORAGE_SCHEMA_VERSION
    experiment_id: str
    experiment_version: str
    experiment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_identity_schema_version: str = EXECUTION_IDENTITY_SCHEMA_VERSION
    runner_version: str = RUNNER_VERSION
    phase: ExperimentPhase
    seed: int
    history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_mode: str
    profile: str
    portfolio_scoring_method: str
    expected_target_count: int = Field(ge=1)
    completed_target_count: int = Field(ge=0)
    parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    updated_at: datetime
    risk_disclaimer: str = DISCLAIMER


class ExperimentPartitionStatusV2(BaseModel):
    """Strict resume state for exactly one schema-v2 execution identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: ExperimentPhase
    experiment_id: str
    experiment_version: str
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    partition_path: Path
    completed_target_issues: tuple[str, ...]
    pending_target_issues: tuple[str, ...]
    is_complete: bool


def _validate_path_segment(value: str, label: str) -> str:
    if not value or value in {".", ".."} or any(token in value for token in ("/", "\\", "..")):
        raise ValueError(f"{label} is not safe for a schema-v2 partition path")
    return value


def _validate_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a complete lowercase SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ExperimentResultStoreV2:
    """Identity-isolated append-only formal storage; legacy partitions are never scanned."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def partition_directory(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        *,
        seed: int,
    ) -> Path:
        phase = _validate_path_segment(spec.data_split.phase, "phase")
        experiment_id = _validate_path_segment(spec.experiment_id, "experiment_id")
        version = _validate_path_segment(spec.experiment_version, "experiment_version")
        run_hash = _validate_sha256(identity.run_context_sha256, "run_context_sha256")
        execution_hash = _validate_sha256(
            identity.execution_config_sha256, "execution_config_sha256"
        )
        return (
            self.root / phase / experiment_id / version / run_hash / execution_hash / f"seed_{seed}"
        )

    def partition_path(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        *,
        seed: int,
    ) -> Path:
        return self.partition_directory(spec, identity, seed=seed) / "observations.parquet"

    def manifest_path(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        *,
        seed: int,
    ) -> Path:
        return self.partition_directory(spec, identity, seed=seed) / "manifest.json"

    def _validate_partition_rows(
        self,
        frame: pd.DataFrame,
        manifest: FormalPartitionManifest,
    ) -> None:
        missing = set(EXPERIMENT_PRIMARY_KEY_V2).difference(frame.columns)
        if missing:
            raise ValueError(f"schema-v2 observations are missing keys: {sorted(missing)}")
        if frame.duplicated(list(EXPERIMENT_PRIMARY_KEY_V2)).any():
            raise DuplicateExperimentResultError("schema-v2 partition contains duplicate keys")
        identity_values = {
            "experiment_id": manifest.experiment_id,
            "experiment_version": manifest.experiment_version,
            "experiment_config_sha256": manifest.experiment_config_sha256,
            "run_context_sha256": manifest.run_context_sha256,
            "execution_config_sha256": manifest.execution_config_sha256,
            "execution_identity_schema_version": manifest.execution_identity_schema_version,
            "runner_version": manifest.runner_version,
            "phase": manifest.phase,
            "seed": manifest.seed,
            "history_sha256": manifest.history_sha256,
            "evaluation_mode": manifest.evaluation_mode,
            "profile": manifest.profile,
            "requested_portfolio_scoring_method": manifest.portfolio_scoring_method,
        }
        for column, expected in identity_values.items():
            if column not in frame or set(frame[column].astype(str)) != {str(expected)}:
                raise ValueError(f"schema-v2 row identity differs from manifest: {column}")
        if (
            "formal_inference_eligible" not in frame
            or not frame["formal_inference_eligible"].astype(bool).all()
        ):
            raise ValueError("schema-v2 formal rows must be inference eligible")

    def validate_manifest_payload(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        *,
        seed: int,
        frame: pd.DataFrame,
        manifest: FormalPartitionManifest,
    ) -> None:
        """Validate an in-memory manifest without mutating the committed partition."""
        expected = {
            "experiment_id": spec.experiment_id,
            "experiment_version": spec.experiment_version,
            "experiment_config_sha256": identity.experiment_config_sha256,
            "run_context_sha256": identity.run_context_sha256,
            "execution_config_sha256": identity.execution_config_sha256,
            "phase": spec.data_split.phase,
            "seed": seed,
        }
        for field, value in expected.items():
            if str(getattr(manifest, field)) != str(value):
                raise ValueError(f"schema-v2 manifest identity mismatch: {field}")
        self._validate_partition_rows(frame, manifest)
        if len(frame) != manifest.completed_target_count:
            raise ValueError("schema-v2 manifest completed count differs from Parquet")

    def read_partition(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        *,
        seed: int,
    ) -> pd.DataFrame:
        parquet_path = self.partition_path(spec, identity, seed=seed)
        manifest_path = self.manifest_path(spec, identity, seed=seed)
        if not parquet_path.exists() and not manifest_path.exists():
            return pd.DataFrame()
        if not parquet_path.exists() or not manifest_path.exists():
            raise ValueError("schema-v2 partition requires both Parquet and manifest")
        manifest = FormalPartitionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if _sha256_file(parquet_path) != manifest.parquet_sha256:
            raise ValueError("schema-v2 Parquet hash differs from manifest")
        frame = pd.read_parquet(parquet_path, engine="pyarrow")
        self.validate_manifest_payload(
            spec,
            identity,
            seed=seed,
            frame=frame,
            manifest=manifest,
        )
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
        identity: ExperimentExecutionIdentity,
        run_context: RunContextIdentity,
    ) -> ExperimentPartitionStatusV2:
        if identity.run_context_sha256 != run_context.run_context_sha256:
            raise ValueError("status execution identity differs from run context")
        existing = self.read_partition(spec, identity, seed=seed)
        if not existing.empty:
            expected_context = {
                "history_sha256": run_context.payload.history_sha256,
                "evaluation_mode": run_context.payload.evaluation_mode,
                "profile": run_context.payload.profile,
                "requested_portfolio_scoring_method": (
                    run_context.payload.portfolio_scoring_method
                ),
            }
            for column, expected_value in expected_context.items():
                if set(existing[column].astype(str)) != {str(expected_value)}:
                    raise ValueError(f"status partition differs from run context: {column}")
        expected = tuple(str(issue) for issue in expected_target_issues)
        completed = (
            set(existing["target_issue"].astype(str).tolist()) if not existing.empty else set()
        )
        unexpected = completed.difference(expected)
        if unexpected:
            raise ValueError(
                f"schema-v2 partition contains unexpected targets: {sorted(unexpected)}"
            )
        pending = tuple(issue for issue in expected if issue not in completed)
        return ExperimentPartitionStatusV2(
            phase=spec.data_split.phase,
            experiment_id=spec.experiment_id,
            experiment_version=spec.experiment_version,
            run_context_sha256=identity.run_context_sha256,
            execution_config_sha256=identity.execution_config_sha256,
            seed=seed,
            partition_path=self.partition_path(spec, identity, seed=seed),
            completed_target_issues=tuple(sorted(completed, key=int)),
            pending_target_issues=pending,
            is_complete=not pending,
        )

    def append(
        self,
        observations: pd.DataFrame,
        *,
        expected_target_issues_by_experiment: Mapping[str, Sequence[str]],
    ) -> tuple[Path, ...]:
        if observations.empty:
            return ()
        required = {
            *EXPERIMENT_PRIMARY_KEY_V2,
            "experiment_config_sha256",
            "run_context_sha256",
            "history_sha256",
            "evaluation_mode",
            "profile",
            "portfolio_scoring_method",
            "execution_identity_schema_version",
            "runner_version",
            "formal_inference_eligible",
        }
        missing = required.difference(observations.columns)
        if missing:
            raise ValueError(f"schema-v2 observations are missing fields: {sorted(missing)}")
        incoming = observations.copy()
        if incoming.duplicated(list(EXPERIMENT_PRIMARY_KEY_V2)).any():
            raise DuplicateExperimentResultError("incoming schema-v2 observations duplicate keys")
        incoming["risk_disclaimer"] = DISCLAIMER
        written: list[Path] = []
        group_columns = [
            "phase",
            "experiment_id",
            "experiment_version",
            "experiment_config_sha256",
            "run_context_sha256",
            "execution_config_sha256",
            "seed",
        ]
        for raw_key, partition in incoming.groupby(group_columns, sort=True, dropna=False):
            (
                raw_phase,
                raw_experiment_id,
                raw_version,
                experiment_hash,
                run_hash,
                execution_hash,
                raw_seed,
            ) = raw_key
            spec = ExperimentSpec.model_validate_json(
                str(partition.iloc[0]["experiment_spec_json"])
            )
            if (
                spec.experiment_id != str(raw_experiment_id)
                or spec.experiment_version != str(raw_version)
                or spec.data_split.phase != str(raw_phase)
            ):
                raise ValueError("schema-v2 embedded ExperimentSpec differs from row identity")
            identity = ExperimentExecutionIdentity(
                experiment_config_sha256=str(experiment_hash),
                run_context_sha256=str(run_hash),
                execution_config_sha256=str(execution_hash),
            )
            seed = int(raw_seed)
            target = self.partition_path(spec, identity, seed=seed)
            manifest_path = self.manifest_path(spec, identity, seed=seed)
            existing = self.read_partition(spec, identity, seed=seed)
            if not existing.empty:
                duplicate = existing.loc[:, list(EXPERIMENT_PRIMARY_KEY_V2)].merge(
                    partition.loc[:, list(EXPERIMENT_PRIMARY_KEY_V2)],
                    on=list(EXPERIMENT_PRIMARY_KEY_V2),
                    how="inner",
                )
                if not duplicate.empty:
                    raise DuplicateExperimentResultError(
                        f"schema-v2 partition already contains {len(duplicate)} key(s)"
                    )
            combined = (
                partition.copy()
                if existing.empty
                else pd.concat([existing, partition], ignore_index=True)
            )
            if combined.duplicated(list(EXPERIMENT_PRIMARY_KEY_V2)).any():
                raise DuplicateExperimentResultError("combined schema-v2 partition duplicates keys")
            expected = tuple(
                str(issue)
                for issue in expected_target_issues_by_experiment.get(spec.experiment_id, ())
            )
            if not expected:
                raise ValueError(f"expected targets missing for {spec.experiment_id}")
            completed = set(combined["target_issue"].astype(str))
            if not completed.issubset(set(expected)):
                raise ValueError("schema-v2 append contains a target outside the expected set")
            immutable_values = {
                "experiment_id": spec.experiment_id,
                "experiment_version": spec.experiment_version,
                "experiment_config_sha256": identity.experiment_config_sha256,
                "run_context_sha256": identity.run_context_sha256,
                "execution_config_sha256": identity.execution_config_sha256,
                "execution_identity_schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
                "runner_version": RUNNER_VERSION,
                "phase": spec.data_split.phase,
                "seed": seed,
            }
            for column, expected_value in immutable_values.items():
                if column not in combined or set(combined[column].astype(str)) != {
                    str(expected_value)
                }:
                    raise ValueError(f"schema-v2 rows contain mixed identity: {column}")
            for column in (
                "history_sha256",
                "evaluation_mode",
                "profile",
                "requested_portfolio_scoring_method",
            ):
                if column not in combined or combined[column].astype(str).nunique() != 1:
                    raise ValueError(f"schema-v2 rows contain mixed context: {column}")
            combined = combined.sort_values("target_issue", key=lambda values: values.astype(int))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp.parquet")
            try:
                combined.to_parquet(temporary, index=False, engine="pyarrow")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            now = datetime.now(UTC)
            previous_created = now
            if manifest_path.exists():
                previous_created = FormalPartitionManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                ).created_at
            manifest = FormalPartitionManifest(
                experiment_id=spec.experiment_id,
                experiment_version=spec.experiment_version,
                experiment_config_sha256=identity.experiment_config_sha256,
                run_context_sha256=identity.run_context_sha256,
                execution_config_sha256=identity.execution_config_sha256,
                phase=spec.data_split.phase,
                seed=seed,
                history_sha256=str(combined.iloc[0]["history_sha256"]),
                evaluation_mode=str(combined.iloc[0]["evaluation_mode"]),
                profile=str(combined.iloc[0]["profile"]),
                portfolio_scoring_method=str(
                    combined.iloc[0]["requested_portfolio_scoring_method"]
                ),
                expected_target_count=len(expected),
                completed_target_count=len(combined),
                parquet_sha256=_sha256_file(target),
                created_at=previous_created,
                updated_at=now,
            )
            self._validate_partition_rows(combined, manifest)
            _atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
            written.append(target)
        return tuple(written)

    def load_all(self, *, phase: ExperimentPhase | None = None) -> pd.DataFrame:
        search_root = self.root if phase is None else self.root / phase
        paths = sorted(search_root.glob("**/observations.parquet")) if search_root.exists() else []
        frames: list[pd.DataFrame] = []
        for path in paths:
            manifest_path = path.with_name("manifest.json")
            if not manifest_path.exists():
                raise ValueError(f"schema-v2 manifest missing: {path}")
            manifest = FormalPartitionManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if _sha256_file(path) != manifest.parquet_sha256:
                raise ValueError(f"schema-v2 Parquet hash differs from manifest: {path}")
            frame = pd.read_parquet(path, engine="pyarrow")
            self._validate_partition_rows(frame, manifest)
            if len(frame) != manifest.completed_target_count:
                raise ValueError(f"schema-v2 manifest completed count differs from Parquet: {path}")
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if combined.duplicated(list(EXPERIMENT_PRIMARY_KEY_V2)).any():
            raise DuplicateExperimentResultError("schema-v2 storage contains duplicate keys")
        for column in ("target_issue", "data_cutoff_issue"):
            if column in combined:
                combined[column] = combined[column].astype("string")
        return combined


class LegacyObservationImportStore:
    """Explicit, non-formal destination for unverified legacy CSV observations."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def path(self) -> Path:
        return self.root / "legacy_observations.parquet"

    def import_csv(self, source: str | Path) -> int:
        if self.path.exists():
            raise FileExistsError("legacy import already exists and cannot be overwritten")
        legacy = pd.read_csv(
            source,
            dtype={"target_issue": "string", "data_cutoff_issue": "string"},
        )
        legacy = legacy.assign(
            identity_status="legacy_unverified",
            storage_schema_version="legacy",
            run_context_sha256=None,
            execution_config_sha256=None,
            formal_inference_eligible=False,
            risk_disclaimer=DISCLAIMER,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp.parquet")
        try:
            legacy.to_parquet(temporary, index=False, engine="pyarrow")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return len(legacy)
