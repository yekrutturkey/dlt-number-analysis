"""Schema-v3 storage isolated by execution identity and complete logical cohort."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.cohort import (
    COHORT_IDENTITY_SCHEMA_VERSION,
    CohortDefinitionIdentity,
    CohortPurpose,
)
from dlt_number_analysis.experiments.identity import (
    EXECUTION_IDENTITY_SCHEMA_VERSION,
    RUNNER_VERSION,
    ExperimentExecutionIdentity,
    RunContextIdentity,
)
from dlt_number_analysis.experiments.specs import ExperimentPhase, ExperimentSpec
from dlt_number_analysis.experiments.splits import experiment_config_sha256
from dlt_number_analysis.experiments.storage import (
    STORAGE_SCHEMA_VERSION,
    DuplicateExperimentResultError,
    _atomic_write_json,
    _sha256_file,
    _validate_path_segment,
    _validate_sha256,
)

EXPERIMENT_PRIMARY_KEY_V3: tuple[str, ...] = (
    "experiment_id",
    "experiment_version",
    "execution_config_sha256",
    "cohort_definition_sha256",
    "phase",
    "seed",
    "target_issue",
)


class FormalPartitionManifestV3(BaseModel):
    """Immutable identities and committed content hash for one schema-v3 partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    storage_schema_version: str = STORAGE_SCHEMA_VERSION
    experiment_id: str
    experiment_version: str
    experiment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_identity_schema_version: str = EXECUTION_IDENTITY_SCHEMA_VERSION
    runner_version: str = RUNNER_VERSION
    cohort_identity_schema_version: str = COHORT_IDENTITY_SCHEMA_VERSION
    cohort_purpose: CohortPurpose
    cohort_id: str
    cohort_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_targets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_target_count: int = Field(ge=1)
    cohort_start_issue: str = Field(pattern=r"^\d+$")
    cohort_end_issue: str = Field(pattern=r"^\d+$")
    completed_target_count: int = Field(ge=0)
    phase: ExperimentPhase
    seed: int
    history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_mode: str
    profile: str
    requested_portfolio_scoring_method: str
    parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    updated_at: datetime
    risk_disclaimer: str = DISCLAIMER


class ExperimentPartitionStatusV3(BaseModel):
    """Resume state for one exact execution, logical cohort, and seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: ExperimentPhase
    experiment_id: str
    experiment_version: str
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cohort_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_targets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    partition_path: Path
    completed_target_issues: tuple[str, ...]
    pending_target_issues: tuple[str, ...]
    is_complete: bool


class ExperimentResultStoreV3:
    """Append-only schema-v3 store that never scans or migrates schema-v2 data."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def partition_directory(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
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
        cohort_hash = _validate_sha256(cohort.cohort_definition_sha256, "cohort_definition_sha256")
        return (
            self.root
            / phase
            / experiment_id
            / version
            / run_hash
            / execution_hash
            / cohort_hash
            / f"seed_{seed}"
        )

    def partition_path(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
    ) -> Path:
        return self.partition_directory(spec, identity, cohort, seed=seed) / "observations.parquet"

    def manifest_path(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
    ) -> Path:
        return self.partition_directory(spec, identity, cohort, seed=seed) / "manifest.json"

    def _validate_rows(
        self,
        frame: pd.DataFrame,
        manifest: FormalPartitionManifestV3,
    ) -> None:
        missing = set(EXPERIMENT_PRIMARY_KEY_V3).difference(frame.columns)
        if missing:
            raise ValueError(f"schema-v3 observations are missing keys: {sorted(missing)}")
        if frame.duplicated(list(EXPERIMENT_PRIMARY_KEY_V3)).any():
            raise DuplicateExperimentResultError("schema-v3 partition contains duplicate keys")
        identity_values = {
            "storage_schema_version": manifest.storage_schema_version,
            "experiment_id": manifest.experiment_id,
            "experiment_version": manifest.experiment_version,
            "experiment_config_sha256": manifest.experiment_config_sha256,
            "run_context_sha256": manifest.run_context_sha256,
            "execution_config_sha256": manifest.execution_config_sha256,
            "execution_identity_schema_version": manifest.execution_identity_schema_version,
            "runner_version": manifest.runner_version,
            "cohort_identity_schema_version": manifest.cohort_identity_schema_version,
            "cohort_purpose": manifest.cohort_purpose,
            "cohort_id": manifest.cohort_id,
            "cohort_definition_sha256": manifest.cohort_definition_sha256,
            "expected_targets_sha256": manifest.expected_targets_sha256,
            "cohort_target_count": manifest.expected_target_count,
            "cohort_start_issue": manifest.cohort_start_issue,
            "cohort_end_issue": manifest.cohort_end_issue,
            "phase": manifest.phase,
            "seed": manifest.seed,
            "history_sha256": manifest.history_sha256,
            "evaluation_mode": manifest.evaluation_mode,
            "profile": manifest.profile,
            "requested_portfolio_scoring_method": (manifest.requested_portfolio_scoring_method),
        }
        for column, expected in identity_values.items():
            if column not in frame or set(frame[column].astype(str)) != {str(expected)}:
                raise ValueError(f"schema-v3 row identity differs from manifest: {column}")
        if (
            "formal_inference_eligible" not in frame
            or not frame["formal_inference_eligible"].astype(bool).all()
        ):
            raise ValueError("schema-v3 rows must be formally inference eligible")
        completed = set(frame["target_issue"].astype(str))
        cohort = CohortDefinitionIdentity.model_validate_json(
            str(frame.iloc[0]["cohort_definition_json"])
        )
        expected = set(cohort.payload.ordered_target_issues)
        if not completed.issubset(expected):
            raise ValueError("schema-v3 completed targets are outside the logical cohort")
        if cohort.cohort_definition_sha256 != manifest.cohort_definition_sha256:
            raise ValueError("schema-v3 embedded cohort differs from manifest")

    def validate_manifest_payload(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
        frame: pd.DataFrame,
        manifest: FormalPartitionManifestV3,
    ) -> None:
        expected = {
            "experiment_id": spec.experiment_id,
            "experiment_version": spec.experiment_version,
            "experiment_config_sha256": identity.experiment_config_sha256,
            "run_context_sha256": identity.run_context_sha256,
            "execution_config_sha256": identity.execution_config_sha256,
            "cohort_identity_schema_version": (cohort.payload.cohort_identity_schema_version),
            "cohort_purpose": cohort.payload.cohort_purpose,
            "cohort_id": cohort.payload.cohort_id,
            "cohort_definition_sha256": cohort.cohort_definition_sha256,
            "expected_targets_sha256": cohort.payload.expected_targets_sha256,
            "expected_target_count": cohort.payload.target_count,
            "cohort_start_issue": cohort.payload.start_issue,
            "cohort_end_issue": cohort.payload.end_issue,
            "phase": spec.data_split.phase,
            "seed": seed,
        }
        for field, value in expected.items():
            if str(getattr(manifest, field)) != str(value):
                raise ValueError(f"schema-v3 manifest identity mismatch: {field}")
        self._validate_rows(frame, manifest)
        if len(frame) != manifest.completed_target_count:
            raise ValueError("schema-v3 manifest completed count differs from Parquet")

    def read_partition(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
    ) -> pd.DataFrame:
        parquet_path = self.partition_path(spec, identity, cohort, seed=seed)
        manifest_path = self.manifest_path(spec, identity, cohort, seed=seed)
        if not parquet_path.exists() and not manifest_path.exists():
            return pd.DataFrame()
        if not parquet_path.exists() or not manifest_path.exists():
            raise ValueError("schema-v3 partition requires both Parquet and manifest")
        manifest = FormalPartitionManifestV3.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if _sha256_file(parquet_path) != manifest.parquet_sha256:
            raise ValueError("schema-v3 Parquet hash differs from manifest")
        frame = pd.read_parquet(parquet_path, engine="pyarrow")
        self.validate_manifest_payload(
            spec,
            identity,
            cohort,
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
        identity: ExperimentExecutionIdentity,
        run_context: RunContextIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
        expected_target_issues: Sequence[str],
    ) -> ExperimentPartitionStatusV3:
        if identity.run_context_sha256 != run_context.run_context_sha256:
            raise ValueError("schema-v3 status execution identity differs from run context")
        canonical_expected = tuple(
            sorted((str(issue) for issue in expected_target_issues), key=int)
        )
        if canonical_expected != cohort.payload.ordered_target_issues:
            raise ValueError("schema-v3 status expected targets differ from logical cohort")
        existing = self.read_partition(spec, identity, cohort, seed=seed)
        if not existing.empty:
            context_values = {
                "history_sha256": run_context.payload.history_sha256,
                "evaluation_mode": run_context.payload.evaluation_mode,
                "profile": run_context.payload.profile,
                "requested_portfolio_scoring_method": (
                    run_context.payload.portfolio_scoring_method
                ),
                "expected_targets_sha256": cohort.payload.expected_targets_sha256,
            }
            for column, expected in context_values.items():
                if set(existing[column].astype(str)) != {str(expected)}:
                    raise ValueError(f"schema-v3 status differs from context: {column}")
        completed = (
            set(existing["target_issue"].astype(str).tolist()) if not existing.empty else set()
        )
        unexpected = completed.difference(canonical_expected)
        if unexpected:
            raise ValueError(f"schema-v3 partition contains unexpected targets: {unexpected}")
        pending = tuple(issue for issue in canonical_expected if issue not in completed)
        return ExperimentPartitionStatusV3(
            phase=spec.data_split.phase,
            experiment_id=spec.experiment_id,
            experiment_version=spec.experiment_version,
            run_context_sha256=identity.run_context_sha256,
            execution_config_sha256=identity.execution_config_sha256,
            cohort_definition_sha256=cohort.cohort_definition_sha256,
            expected_targets_sha256=cohort.payload.expected_targets_sha256,
            seed=seed,
            partition_path=self.partition_path(spec, identity, cohort, seed=seed),
            completed_target_issues=tuple(sorted(completed, key=int)),
            pending_target_issues=pending,
            is_complete=not pending,
        )

    def append(self, observations: pd.DataFrame) -> tuple[Path, ...]:
        if observations.empty:
            return ()
        required = {
            *EXPERIMENT_PRIMARY_KEY_V3,
            "experiment_spec_json",
            "cohort_definition_json",
            "experiment_config_sha256",
            "run_context_sha256",
            "history_sha256",
            "evaluation_mode",
            "profile",
            "requested_portfolio_scoring_method",
            "formal_inference_eligible",
        }
        missing = required.difference(observations.columns)
        if missing:
            raise ValueError(f"schema-v3 observations are missing fields: {sorted(missing)}")
        incoming = observations.copy()
        if incoming.duplicated(list(EXPERIMENT_PRIMARY_KEY_V3)).any():
            raise DuplicateExperimentResultError("incoming schema-v3 observations duplicate keys")
        incoming["risk_disclaimer"] = DISCLAIMER
        group_columns = [
            "phase",
            "experiment_id",
            "experiment_version",
            "experiment_config_sha256",
            "run_context_sha256",
            "execution_config_sha256",
            "cohort_definition_sha256",
            "seed",
        ]
        written: list[Path] = []
        for raw_key, partition in incoming.groupby(group_columns, sort=True, dropna=False):
            (
                raw_phase,
                raw_experiment_id,
                raw_version,
                experiment_hash,
                run_hash,
                execution_hash,
                cohort_hash,
                raw_seed,
            ) = raw_key
            spec = ExperimentSpec.model_validate_json(
                str(partition.iloc[0]["experiment_spec_json"])
            )
            cohort = CohortDefinitionIdentity.model_validate_json(
                str(partition.iloc[0]["cohort_definition_json"])
            )
            if (
                spec.experiment_id != str(raw_experiment_id)
                or spec.experiment_version != str(raw_version)
                or spec.data_split.phase != str(raw_phase)
                or experiment_config_sha256(spec) != str(experiment_hash)
            ):
                raise ValueError("schema-v3 embedded ExperimentSpec differs from row identity")
            if cohort.cohort_definition_sha256 != str(cohort_hash):
                raise ValueError("schema-v3 embedded cohort differs from row identity")
            identity = ExperimentExecutionIdentity(
                experiment_config_sha256=str(experiment_hash),
                run_context_sha256=str(run_hash),
                execution_config_sha256=str(execution_hash),
            )
            seed = int(raw_seed)
            target = self.partition_path(spec, identity, cohort, seed=seed)
            manifest_path = self.manifest_path(spec, identity, cohort, seed=seed)
            existing = self.read_partition(spec, identity, cohort, seed=seed)
            if not existing.empty:
                duplicate = existing.loc[:, list(EXPERIMENT_PRIMARY_KEY_V3)].merge(
                    partition.loc[:, list(EXPERIMENT_PRIMARY_KEY_V3)],
                    on=list(EXPERIMENT_PRIMARY_KEY_V3),
                    how="inner",
                )
                if not duplicate.empty:
                    raise DuplicateExperimentResultError(
                        f"schema-v3 partition already contains {len(duplicate)} key(s)"
                    )
            combined = (
                partition.copy()
                if existing.empty
                else pd.concat([existing, partition], ignore_index=True)
            )
            if combined.duplicated(list(EXPERIMENT_PRIMARY_KEY_V3)).any():
                raise DuplicateExperimentResultError("combined schema-v3 partition duplicates keys")
            completed = set(combined["target_issue"].astype(str))
            expected = set(cohort.payload.ordered_target_issues)
            if not completed.issubset(expected):
                raise ValueError("schema-v3 append contains a target outside the logical cohort")
            combined = combined.sort_values("target_issue", key=lambda values: values.astype(int))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp.parquet")
            try:
                combined.to_parquet(temporary, index=False, engine="pyarrow")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            now = datetime.now(UTC)
            created_at = now
            if manifest_path.exists():
                created_at = FormalPartitionManifestV3.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                ).created_at
            manifest = FormalPartitionManifestV3(
                experiment_id=spec.experiment_id,
                experiment_version=spec.experiment_version,
                experiment_config_sha256=identity.experiment_config_sha256,
                run_context_sha256=identity.run_context_sha256,
                execution_config_sha256=identity.execution_config_sha256,
                cohort_purpose=cohort.payload.cohort_purpose,
                cohort_id=cohort.payload.cohort_id,
                cohort_definition_sha256=cohort.cohort_definition_sha256,
                expected_targets_sha256=cohort.payload.expected_targets_sha256,
                expected_target_count=cohort.payload.target_count,
                cohort_start_issue=cohort.payload.start_issue,
                cohort_end_issue=cohort.payload.end_issue,
                completed_target_count=len(combined),
                phase=spec.data_split.phase,
                seed=seed,
                history_sha256=str(combined.iloc[0]["history_sha256"]),
                evaluation_mode=str(combined.iloc[0]["evaluation_mode"]),
                profile=str(combined.iloc[0]["profile"]),
                requested_portfolio_scoring_method=str(
                    combined.iloc[0]["requested_portfolio_scoring_method"]
                ),
                parquet_sha256=_sha256_file(target),
                created_at=created_at,
                updated_at=now,
            )
            self.validate_manifest_payload(
                spec,
                identity,
                cohort,
                seed=seed,
                frame=combined,
                manifest=manifest,
            )
            _atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
            written.append(target)
        return tuple(written)

    def _read_discovered_partition(self, path: Path) -> pd.DataFrame:
        manifest_path = path.with_name("manifest.json")
        if not manifest_path.exists():
            raise ValueError(f"schema-v3 manifest missing: {path}")
        manifest = FormalPartitionManifestV3.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if _sha256_file(path) != manifest.parquet_sha256:
            raise ValueError(f"schema-v3 Parquet hash differs from manifest: {path}")
        frame = pd.read_parquet(path, engine="pyarrow")
        self._validate_rows(frame, manifest)
        if len(frame) != manifest.completed_target_count:
            raise ValueError("schema-v3 discovered manifest count differs from Parquet")
        return frame

    def load_cohort(
        self,
        *,
        run_context_sha256: str,
        cohort_definition_sha256: str,
        phase: ExperimentPhase,
        experiment_ids: Sequence[str],
        seeds: Sequence[int],
    ) -> pd.DataFrame:
        """Load only one explicitly selected context/cohort; never scan schema-v2 or legacy."""
        run_hash = _validate_sha256(run_context_sha256, "run_context_sha256")
        cohort_hash = _validate_sha256(cohort_definition_sha256, "cohort_definition_sha256")
        selected_seeds = {int(seed) for seed in seeds}
        if not experiment_ids or not selected_seeds:
            raise ValueError("formal cohort selection requires experiment IDs and seeds")
        paths: list[Path] = []
        for experiment_id in experiment_ids:
            safe_id = _validate_path_segment(str(experiment_id), "experiment_id")
            pattern = f"{phase}/{safe_id}/*/{run_hash}/*/{cohort_hash}/seed_*/observations.parquet"
            for path in self.root.glob(pattern):
                seed = int(path.parent.name.removeprefix("seed_"))
                if seed in selected_seeds:
                    paths.append(path)
        frames = [self._read_discovered_partition(path) for path in sorted(paths)]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if combined.duplicated(list(EXPERIMENT_PRIMARY_KEY_V3)).any():
            raise DuplicateExperimentResultError("selected schema-v3 cohort duplicates keys")
        exact_values = {
            "phase": phase,
            "run_context_sha256": run_hash,
            "cohort_definition_sha256": cohort_hash,
        }
        for column, expected in exact_values.items():
            if set(combined[column].astype(str)) != {str(expected)}:
                raise ValueError(f"formal cohort selection mixed {column}")
        requested_ids = {str(experiment_id) for experiment_id in experiment_ids}
        actual_ids = set(combined["experiment_id"].astype(str))
        if actual_ids != requested_ids:
            raise ValueError(
                "formal cohort selection did not return exactly the requested experiments"
            )
        actual_seeds = set(combined["seed"].astype(int))
        if actual_seeds != selected_seeds:
            raise ValueError("formal cohort selection did not return exactly the requested seeds")
        for column in (
            "history_sha256",
            "evaluation_mode",
            "profile",
            "requested_portfolio_scoring_method",
        ):
            if combined[column].astype(str).nunique() != 1:
                raise ValueError(f"formal cohort selection mixed {column}")
        for _, partition in combined.groupby(["experiment_id", "seed"], sort=True):
            cohort = CohortDefinitionIdentity.model_validate_json(
                str(partition.iloc[0]["cohort_definition_json"])
            )
            actual_targets = tuple(sorted(partition["target_issue"].astype(str), key=int))
            if actual_targets != cohort.payload.ordered_target_issues:
                raise ValueError("formal cohort selection contains an incomplete partition")
        if not combined["formal_inference_eligible"].astype(bool).all():
            raise ValueError("formal cohort selection contains ineligible rows")
        for column in ("target_issue", "data_cutoff_issue"):
            if column in combined:
                combined[column] = combined[column].astype("string")
        return combined

    def context_index(self) -> pd.DataFrame:
        """Return an audit-only manifest index without calculating strategy comparisons."""
        records = []
        for path in sorted(self.root.glob("**/manifest.json")):
            manifest = FormalPartitionManifestV3.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            records.append({**manifest.model_dump(mode="python"), "manifest_path": str(path)})
        return pd.DataFrame.from_records(records)
