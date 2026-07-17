"""Schema-v4 immutable-generation storage with an atomic JSON CURRENT pointer."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.cohort import CohortDefinitionIdentity
from dlt_number_analysis.experiments.identity import (
    ExperimentExecutionIdentity,
    RunContextIdentity,
    current_git_commit_sha,
)
from dlt_number_analysis.experiments.manual_operations import (
    CurrentPointerRepairAuditV2,
    atomic_write_manual_audit,
    operation_id,
    operator_hostname,
)
from dlt_number_analysis.experiments.specs import ExperimentPhase, ExperimentSpec
from dlt_number_analysis.experiments.splits import experiment_config_sha256
from dlt_number_analysis.experiments.storage import (
    STORAGE_SCHEMA_VERSION_V4,
    DuplicateExperimentResultError,
    _sha256_file,
    _validate_path_segment,
    _validate_sha256,
)
from dlt_number_analysis.experiments.storage_v3 import (
    EXPERIMENT_PRIMARY_KEY_V3,
    ExperimentResultStoreV3,
    FormalPartitionManifestV3,
)

EXPERIMENT_PRIMARY_KEY_V4 = EXPERIMENT_PRIMARY_KEY_V3

GenerationFaultPoint = Literal[
    "before_parquet_write",
    "after_parquet_before_manifest",
    "after_manifest_before_generation_rename",
    "after_generation_rename_before_current_write",
    "after_current_temp_before_replace",
    "after_current_replace",
]
GenerationFaultInjector = Callable[[GenerationFaultPoint], None]


class CurrentPointerV4(BaseModel):
    """The only committed selector for one schema-v4 partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    storage_schema_version: Literal["experiment-storage-schema-v4"] = STORAGE_SCHEMA_VERSION_V4
    generation_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    updated_at: datetime


class FormalPartitionManifestV4(FormalPartitionManifestV3):
    """A fully validated immutable generation manifest."""

    storage_schema_version: Literal["experiment-storage-schema-v4"] = STORAGE_SCHEMA_VERSION_V4
    generation_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    parent_generation_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    generation_created_at: datetime
    observation_count: int = Field(ge=0)
    committed: Literal[True] = True


class ExperimentPartitionStatusV4(BaseModel):
    """Resume state resolved only from the generation referenced by CURRENT."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: ExperimentPhase
    experiment_id: str
    experiment_version: str
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cohort_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_targets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    current_path: Path
    current_generation_id: str | None
    completed_target_issues: tuple[str, ...]
    pending_target_issues: tuple[str, ...]
    is_complete: bool


class PartitionGenerationAudit(BaseModel):
    """Read-only classification of CURRENT, valid generations, and orphans."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    partition_directory: Path
    current_generation_id: str | None
    valid_generation_ids: tuple[str, ...]
    orphan_generation_ids: tuple[str, ...]
    invalid_generation_ids: tuple[str, ...]
    current_is_valid: bool
    recovery_possible: bool
    warnings: tuple[str, ...]


CurrentPointerRepairAudit = CurrentPointerRepairAuditV2

CurrentRepairFaultPoint = Literal[
    "before_prepare_audit_write",
    "after_prepare_audit_commit",
    "before_current_replace",
    "after_current_replace",
    "before_completed_audit_update",
]
CurrentRepairFaultInjector = Callable[[CurrentRepairFaultPoint], None]


def _fsync_file(path: Path) -> None:
    """Best-effort durable flush for one already-written regular file."""
    try:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync; unsupported Windows calls are harmless."""
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


def _write_json_fsync(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _fsync_file(path)


def _generation_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid4().hex}"


def _inject(
    injector: GenerationFaultInjector | None,
    point: GenerationFaultPoint,
) -> None:
    if injector is not None:
        injector(point)


def _inject_repair_fault(
    injector: CurrentRepairFaultInjector | None,
    point: CurrentRepairFaultPoint,
) -> None:
    if injector is not None:
        injector(point)


def _replace_current(temporary: Path, current: Path) -> None:
    os.replace(temporary, current)


class ExperimentResultStoreV4:
    """Append-only store whose CURRENT switch is the sole commit point."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._v3_validator = ExperimentResultStoreV3(self.root)

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
            identity.execution_config_sha256,
            "execution_config_sha256",
        )
        cohort_hash = _validate_sha256(
            cohort.cohort_definition_sha256,
            "cohort_definition_sha256",
        )
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

    def current_path(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
    ) -> Path:
        return self.partition_directory(spec, identity, cohort, seed=seed) / "CURRENT"

    def generations_directory(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
    ) -> Path:
        return self.partition_directory(spec, identity, cohort, seed=seed) / "generations"

    def generation_directory(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
        generation_id: str,
    ) -> Path:
        safe_generation = _validate_path_segment(generation_id, "generation_id")
        return (
            self.generations_directory(
                spec,
                identity,
                cohort,
                seed=seed,
            )
            / safe_generation
        )

    def _parse_current(self, path: Path) -> CurrentPointerV4:
        try:
            return CurrentPointerV4.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"schema-v4 CURRENT is invalid: {path}") from error

    def _validate_generation_directory(
        self,
        directory: Path,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
        expected_generation_id: str,
        expected_manifest_sha256: str | None = None,
        require_directory_name: bool = True,
    ) -> tuple[pd.DataFrame, FormalPartitionManifestV4]:
        manifest_path = directory / "manifest.json"
        parquet_path = directory / "observations.parquet"
        if not manifest_path.is_file() or not parquet_path.is_file():
            raise ValueError("schema-v4 generation requires Parquet and manifest")
        manifest_sha = _sha256_file(manifest_path)
        if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
            raise ValueError("schema-v4 manifest SHA differs from CURRENT")
        manifest = FormalPartitionManifestV4.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.generation_id != expected_generation_id:
            raise ValueError("schema-v4 generation ID differs from manifest")
        if require_directory_name and directory.name != expected_generation_id:
            raise ValueError("schema-v4 generation ID differs from directory")
        if _sha256_file(parquet_path) != manifest.parquet_sha256:
            raise ValueError("schema-v4 Parquet hash differs from manifest")
        frame = pd.read_parquet(parquet_path, engine="pyarrow")
        self._v3_validator.validate_manifest_payload(
            spec,
            identity,
            cohort,
            seed=seed,
            frame=frame,
            manifest=manifest,
        )
        if manifest.storage_schema_version != STORAGE_SCHEMA_VERSION_V4:
            raise ValueError("schema-v4 generation has the wrong storage schema")
        if manifest.observation_count != len(frame):
            raise ValueError("schema-v4 observation count differs from Parquet")
        if manifest.completed_target_count != len(frame):
            raise ValueError("schema-v4 completed count differs from Parquet")
        if manifest.committed is not True:
            raise ValueError("schema-v4 generation is not committed")
        return frame, manifest

    def _read_current_generation(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
    ) -> tuple[pd.DataFrame, FormalPartitionManifestV4, CurrentPointerV4] | None:
        current_path = self.current_path(spec, identity, cohort, seed=seed)
        generations = self.generations_directory(spec, identity, cohort, seed=seed)
        if not current_path.exists():
            official_generations = (
                [
                    path
                    for path in generations.iterdir()
                    if path.is_dir() and not path.name.startswith(".")
                ]
                if generations.exists()
                else []
            )
            if official_generations:
                raise ValueError("schema-v4 CURRENT is missing while generations exist")
            return None
        current = self._parse_current(current_path)
        directory = self.generation_directory(
            spec,
            identity,
            cohort,
            seed=seed,
            generation_id=current.generation_id,
        )
        if not directory.is_dir():
            raise ValueError("schema-v4 CURRENT points to a missing generation")
        frame, manifest = self._validate_generation_directory(
            directory,
            spec,
            identity,
            cohort,
            seed=seed,
            expected_generation_id=current.generation_id,
            expected_manifest_sha256=current.manifest_sha256,
        )
        return frame, manifest, current

    def read_partition(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
    ) -> pd.DataFrame:
        resolved = self._read_current_generation(
            spec,
            identity,
            cohort,
            seed=seed,
        )
        if resolved is None:
            return pd.DataFrame()
        frame = resolved[0].copy()
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
    ) -> ExperimentPartitionStatusV4:
        if identity.run_context_sha256 != run_context.run_context_sha256:
            raise ValueError("schema-v4 status execution identity differs from run context")
        canonical_expected = tuple(sorted(map(str, expected_target_issues), key=int))
        if canonical_expected != cohort.payload.ordered_target_issues:
            raise ValueError("schema-v4 status expected targets differ from logical cohort")
        resolved = self._read_current_generation(
            spec,
            identity,
            cohort,
            seed=seed,
        )
        existing = pd.DataFrame() if resolved is None else resolved[0]
        current_id = None if resolved is None else resolved[2].generation_id
        if not existing.empty:
            expected_context = {
                "history_sha256": run_context.payload.history_sha256,
                "evaluation_mode": run_context.payload.evaluation_mode,
                "profile": run_context.payload.profile,
                "requested_portfolio_scoring_method": (
                    run_context.payload.portfolio_scoring_method
                ),
                "expected_targets_sha256": cohort.payload.expected_targets_sha256,
            }
            for column, expected in expected_context.items():
                if set(existing[column].astype(str)) != {str(expected)}:
                    raise ValueError(f"schema-v4 status differs from context: {column}")
        completed = set(existing["target_issue"].astype(str)) if not existing.empty else set()
        unexpected = completed.difference(canonical_expected)
        if unexpected:
            raise ValueError(f"schema-v4 partition contains unexpected targets: {unexpected}")
        pending = tuple(issue for issue in canonical_expected if issue not in completed)
        return ExperimentPartitionStatusV4(
            phase=spec.data_split.phase,
            experiment_id=spec.experiment_id,
            experiment_version=spec.experiment_version,
            run_context_sha256=identity.run_context_sha256,
            execution_config_sha256=identity.execution_config_sha256,
            cohort_definition_sha256=cohort.cohort_definition_sha256,
            expected_targets_sha256=cohort.payload.expected_targets_sha256,
            seed=seed,
            current_path=self.current_path(spec, identity, cohort, seed=seed),
            current_generation_id=current_id,
            completed_target_issues=tuple(sorted(completed, key=int)),
            pending_target_issues=pending,
            is_complete=not pending,
        )

    def _new_manifest(
        self,
        *,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        seed: int,
        combined: pd.DataFrame,
        generation_id: str,
        parent_generation_id: str | None,
        parquet_sha256: str,
        created_at: datetime,
        now: datetime,
    ) -> FormalPartitionManifestV4:
        return FormalPartitionManifestV4(
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
            parquet_sha256=parquet_sha256,
            created_at=created_at,
            updated_at=now,
            generation_id=generation_id,
            parent_generation_id=parent_generation_id,
            generation_created_at=now,
            observation_count=len(combined),
            committed=True,
        )

    def append(
        self,
        observations: pd.DataFrame,
        *,
        _fault_injector: GenerationFaultInjector | None = None,
    ) -> tuple[Path, ...]:
        """Commit each partition through a new generation and atomic CURRENT switch."""
        if observations.empty:
            return ()
        required = {
            *EXPERIMENT_PRIMARY_KEY_V4,
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
            raise ValueError(f"schema-v4 observations are missing fields: {sorted(missing)}")
        incoming = observations.copy()
        incoming["storage_schema_version"] = STORAGE_SCHEMA_VERSION_V4
        incoming["risk_disclaimer"] = DISCLAIMER
        if incoming.duplicated(list(EXPERIMENT_PRIMARY_KEY_V4)).any():
            raise DuplicateExperimentResultError("incoming schema-v4 observations duplicate keys")
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
        committed_paths: list[Path] = []
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
                raise ValueError("schema-v4 embedded ExperimentSpec differs from row identity")
            if cohort.cohort_definition_sha256 != str(cohort_hash):
                raise ValueError("schema-v4 embedded cohort differs from row identity")
            identity = ExperimentExecutionIdentity(
                experiment_config_sha256=str(experiment_hash),
                run_context_sha256=str(run_hash),
                execution_config_sha256=str(execution_hash),
            )
            seed = int(raw_seed)
            resolved = self._read_current_generation(
                spec,
                identity,
                cohort,
                seed=seed,
            )
            existing = pd.DataFrame() if resolved is None else resolved[0]
            parent_manifest = None if resolved is None else resolved[1]
            parent_current = None if resolved is None else resolved[2]
            if not existing.empty:
                duplicate = existing.loc[:, list(EXPERIMENT_PRIMARY_KEY_V4)].merge(
                    partition.loc[:, list(EXPERIMENT_PRIMARY_KEY_V4)],
                    on=list(EXPERIMENT_PRIMARY_KEY_V4),
                    how="inner",
                )
                if not duplicate.empty:
                    raise DuplicateExperimentResultError(
                        f"schema-v4 partition already contains {len(duplicate)} key(s)"
                    )
            combined = (
                partition.copy()
                if existing.empty
                else pd.concat([existing, partition], ignore_index=True)
            )
            if combined.duplicated(list(EXPERIMENT_PRIMARY_KEY_V4)).any():
                raise DuplicateExperimentResultError("combined schema-v4 partition duplicates keys")
            completed = set(combined["target_issue"].astype(str))
            if not completed.issubset(cohort.payload.ordered_target_issues):
                raise ValueError("schema-v4 append contains a target outside the logical cohort")
            combined = combined.sort_values(
                "target_issue",
                key=lambda values: values.astype(int),
            )
            generations = self.generations_directory(
                spec,
                identity,
                cohort,
                seed=seed,
            )
            generations.mkdir(parents=True, exist_ok=True)
            generation_id = _generation_id()
            temporary = generations / f".{generation_id}.tmp"
            final = generations / generation_id
            temporary.mkdir()
            _inject(_fault_injector, "before_parquet_write")
            parquet_path = temporary / "observations.parquet"
            combined.to_parquet(parquet_path, index=False, engine="pyarrow")
            _fsync_file(parquet_path)
            _inject(_fault_injector, "after_parquet_before_manifest")
            now = datetime.now(UTC)
            manifest = self._new_manifest(
                spec=spec,
                identity=identity,
                cohort=cohort,
                seed=seed,
                combined=combined,
                generation_id=generation_id,
                parent_generation_id=(
                    None if parent_current is None else parent_current.generation_id
                ),
                parquet_sha256=_sha256_file(parquet_path),
                created_at=(now if parent_manifest is None else parent_manifest.created_at),
                now=now,
            )
            manifest_path = temporary / "manifest.json"
            _write_json_fsync(manifest_path, manifest.model_dump(mode="json"))
            _inject(_fault_injector, "after_manifest_before_generation_rename")
            self._validate_generation_directory(
                temporary,
                spec,
                identity,
                cohort,
                seed=seed,
                expected_generation_id=generation_id,
                require_directory_name=False,
            )
            _fsync_directory(temporary)
            os.replace(temporary, final)
            _fsync_directory(generations)
            _inject(_fault_injector, "after_generation_rename_before_current_write")
            current = CurrentPointerV4(
                generation_id=generation_id,
                manifest_sha256=_sha256_file(final / "manifest.json"),
                updated_at=datetime.now(UTC),
            )
            partition_directory = self.partition_directory(
                spec,
                identity,
                cohort,
                seed=seed,
            )
            temporary_current = partition_directory / f".CURRENT.{uuid4().hex}.tmp"
            _write_json_fsync(temporary_current, current.model_dump(mode="json"))
            _inject(_fault_injector, "after_current_temp_before_replace")
            os.replace(
                temporary_current,
                self.current_path(spec, identity, cohort, seed=seed),
            )
            _fsync_directory(partition_directory)
            _inject(_fault_injector, "after_current_replace")
            committed_paths.append(final / "observations.parquet")
        return tuple(committed_paths)

    def audit_partition_generations(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
    ) -> PartitionGenerationAudit:
        """Classify generations without changing CURRENT or selecting by mtime."""
        partition = self.partition_directory(spec, identity, cohort, seed=seed)
        generations = self.generations_directory(spec, identity, cohort, seed=seed)
        current_path = self.current_path(spec, identity, cohort, seed=seed)
        warnings: list[str] = []
        current: CurrentPointerV4 | None = None
        if current_path.exists():
            try:
                current = self._parse_current(current_path)
            except ValueError as error:
                warnings.append(str(error))
        valid: list[str] = []
        invalid: list[str] = []
        manifest_hashes: dict[str, str] = {}
        if generations.exists():
            for directory in sorted(path for path in generations.iterdir() if path.is_dir()):
                if directory.name.startswith("."):
                    invalid.append(directory.name)
                    warnings.append(f"temporary generation remains: {directory.name}")
                    continue
                try:
                    self._validate_generation_directory(
                        directory,
                        spec,
                        identity,
                        cohort,
                        seed=seed,
                        expected_generation_id=directory.name,
                    )
                except (OSError, ValueError) as error:
                    invalid.append(directory.name)
                    warnings.append(f"invalid generation {directory.name}: {error}")
                else:
                    valid.append(directory.name)
                    manifest_hashes[directory.name] = _sha256_file(directory / "manifest.json")
        current_valid = False
        if current is not None:
            current_valid = (
                current.generation_id in valid
                and manifest_hashes.get(current.generation_id) == current.manifest_sha256
            )
            if not current_valid:
                warnings.append("CURRENT does not resolve to a fully valid generation")
        elif valid:
            warnings.append("valid generations exist without a valid CURRENT")
        empty = (
            not current_path.exists()
            and not valid
            and not [item for item in invalid if not item.startswith(".")]
        )
        current_id = None if current is None else current.generation_id
        orphan = tuple(item for item in valid if item != current_id)
        return PartitionGenerationAudit(
            partition_directory=partition,
            current_generation_id=current_id,
            valid_generation_ids=tuple(valid),
            orphan_generation_ids=orphan,
            invalid_generation_ids=tuple(invalid),
            current_is_valid=current_valid or empty,
            recovery_possible=(not current_valid and bool(valid)),
            warnings=tuple(warnings),
        )

    def repair_current_pointer(
        self,
        spec: ExperimentSpec,
        identity: ExperimentExecutionIdentity,
        cohort: CohortDefinitionIdentity,
        *,
        seed: int,
        generation_id: str,
        reason: str,
        _fault_injector: CurrentRepairFaultInjector | None = None,
    ) -> Path:
        """Persist prepared intent before pointing CURRENT at a valid generation."""
        if not reason.strip():
            raise ValueError("schema-v4 CURRENT repair requires a reason")
        audit = self.audit_partition_generations(
            spec,
            identity,
            cohort,
            seed=seed,
        )
        if generation_id not in audit.valid_generation_ids:
            raise ValueError("schema-v4 repair generation is not fully valid")
        generation = self.generation_directory(
            spec,
            identity,
            cohort,
            seed=seed,
            generation_id=generation_id,
        )
        validated_manifest_sha = _sha256_file(generation / "manifest.json")
        self._validate_generation_directory(
            generation,
            spec,
            identity,
            cohort,
            seed=seed,
            expected_generation_id=generation_id,
            expected_manifest_sha256=validated_manifest_sha,
        )
        current_path = self.current_path(spec, identity, cohort, seed=seed)
        original: dict[str, object] | str | None = None
        original_sha256: str | None = None
        if current_path.exists():
            raw = current_path.read_text(encoding="utf-8")
            original_sha256 = _sha256_file(current_path)
            try:
                parsed = json.loads(raw)
                original = parsed if isinstance(parsed, dict) else raw
            except json.JSONDecodeError:
                original = raw
        replacement = CurrentPointerV4(
            generation_id=generation_id,
            manifest_sha256=validated_manifest_sha,
            updated_at=datetime.now(UTC),
        )
        requested_at = datetime.now(UTC)
        operation = operation_id()
        repair_directory = current_path.parent / "repairs"
        repair_path = repair_directory / f"repair_{operation}.json"
        prepared = CurrentPointerRepairAuditV2(
            operation_id=operation,
            partition_path=current_path.parent.resolve(),
            original_current=original,
            original_current_sha256=original_sha256,
            requested_generation_id=generation_id,
            validated_generation_manifest_sha256=validated_manifest_sha,
            proposed_current=replacement.model_dump(mode="json"),
            requested_at=requested_at,
            status="prepared",
            reason=reason.strip(),
            git_commit_sha=current_git_commit_sha(),
            operator_hostname=operator_hostname(),
        )
        _inject_repair_fault(_fault_injector, "before_prepare_audit_write")
        atomic_write_manual_audit(repair_path, prepared)
        _inject_repair_fault(_fault_injector, "after_prepare_audit_commit")
        temporary = current_path.with_name(f".CURRENT.repair.{operation}.tmp")
        current_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_json_fsync(temporary, replacement.model_dump(mode="json"))
        except OSError as error:
            failed = prepared.model_copy(update={"status": "failed", "failure_message": str(error)})
            atomic_write_manual_audit(repair_path, failed)
            raise RuntimeError("schema-v4 repair temporary CURRENT write failed") from error
        _inject_repair_fault(_fault_injector, "before_current_replace")
        try:
            _replace_current(temporary, current_path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            failed = prepared.model_copy(update={"status": "failed", "failure_message": str(error)})
            atomic_write_manual_audit(repair_path, failed)
            raise RuntimeError("schema-v4 CURRENT replacement failed") from error
        _fsync_directory(current_path.parent)
        _inject_repair_fault(_fault_injector, "after_current_replace")
        resolved = self._read_current_generation(
            spec,
            identity,
            cohort,
            seed=seed,
        )
        if resolved is None or resolved[2].generation_id != generation_id:
            failed = prepared.model_copy(
                update={
                    "status": "failed",
                    "failure_message": "CURRENT post-repair verification failed",
                }
            )
            atomic_write_manual_audit(repair_path, failed)
            raise RuntimeError("schema-v4 CURRENT post-repair verification failed")
        _inject_repair_fault(_fault_injector, "before_completed_audit_update")
        completed = prepared.model_copy(
            update={
                "status": "completed",
                "completed_at": datetime.now(UTC),
            }
        )
        atomic_write_manual_audit(repair_path, completed)
        return repair_path

    def load_cohort(
        self,
        *,
        run_context_sha256: str,
        cohort_definition_sha256: str,
        phase: ExperimentPhase,
        experiment_ids: Sequence[str],
        seeds: Sequence[int],
    ) -> pd.DataFrame:
        """Load only explicitly selected CURRENT generations in this schema-v4 root."""
        run_hash = _validate_sha256(run_context_sha256, "run_context_sha256")
        cohort_hash = _validate_sha256(
            cohort_definition_sha256,
            "cohort_definition_sha256",
        )
        selected_ids = {str(item) for item in experiment_ids}
        selected_seeds = {int(item) for item in seeds}
        if not selected_ids or not selected_seeds:
            raise ValueError("schema-v4 cohort load requires experiment IDs and seeds")
        frames: list[pd.DataFrame] = []
        for experiment_id in sorted(selected_ids):
            safe_id = _validate_path_segment(experiment_id, "experiment_id")
            pattern = f"{phase}/{safe_id}/*/{run_hash}/*/{cohort_hash}/seed_*"
            for partition in sorted(self.root.glob(pattern)):
                if not partition.is_dir():
                    continue
                seed = int(partition.name.removeprefix("seed_"))
                if seed not in selected_seeds:
                    continue
                current_path = partition / "CURRENT"
                if not current_path.exists():
                    generations = partition / "generations"
                    official = (
                        [
                            path
                            for path in generations.iterdir()
                            if path.is_dir() and not path.name.startswith(".")
                        ]
                        if generations.exists()
                        else []
                    )
                    if official:
                        raise ValueError("schema-v4 CURRENT is missing while generations exist")
                    continue
                current = self._parse_current(current_path)
                generation = partition / "generations" / current.generation_id
                manifest = FormalPartitionManifestV4.model_validate_json(
                    (generation / "manifest.json").read_text(encoding="utf-8")
                )
                stored = pd.read_parquet(generation / "observations.parquet")
                spec = ExperimentSpec.model_validate_json(
                    str(stored.iloc[0]["experiment_spec_json"])
                )
                cohort = CohortDefinitionIdentity.model_validate_json(
                    str(stored.iloc[0]["cohort_definition_json"])
                )
                identity = ExperimentExecutionIdentity(
                    experiment_config_sha256=manifest.experiment_config_sha256,
                    run_context_sha256=manifest.run_context_sha256,
                    execution_config_sha256=manifest.execution_config_sha256,
                )
                frame, _ = self._validate_generation_directory(
                    generation,
                    spec,
                    identity,
                    cohort,
                    seed=seed,
                    expected_generation_id=current.generation_id,
                    expected_manifest_sha256=current.manifest_sha256,
                )
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if combined.duplicated(list(EXPERIMENT_PRIMARY_KEY_V4)).any():
            raise DuplicateExperimentResultError("selected schema-v4 cohort duplicates keys")
        exact = {
            "phase": phase,
            "run_context_sha256": run_hash,
            "cohort_definition_sha256": cohort_hash,
        }
        for column, expected in exact.items():
            if set(combined[column].astype(str)) != {str(expected)}:
                raise ValueError(f"schema-v4 formal cohort selection mixed {column}")
        if set(combined["experiment_id"].astype(str)) != selected_ids:
            raise ValueError("schema-v4 cohort load did not find exactly the requested experiments")
        if set(combined["seed"].astype(int)) != selected_seeds:
            raise ValueError("schema-v4 cohort load did not find exactly the requested seeds")
        for column in (
            "history_sha256",
            "evaluation_mode",
            "profile",
            "requested_portfolio_scoring_method",
        ):
            if combined[column].astype(str).nunique() != 1:
                raise ValueError(f"schema-v4 formal cohort selection mixed {column}")
        for _, partition in combined.groupby(["experiment_id", "seed"], sort=True):
            cohort = CohortDefinitionIdentity.model_validate_json(
                str(partition.iloc[0]["cohort_definition_json"])
            )
            targets = tuple(sorted(partition["target_issue"].astype(str), key=int))
            if targets != cohort.payload.ordered_target_issues:
                raise ValueError("schema-v4 formal cohort selection is incomplete")
        for column in ("target_issue", "data_cutoff_issue"):
            if column in combined:
                combined[column] = combined[column].astype("string")
        return combined

    def context_index(self) -> pd.DataFrame:
        """Return CURRENT metadata only; orphan generations never become report inputs."""
        records: list[dict[str, object]] = []
        for path in sorted(self.root.glob("**/CURRENT")):
            try:
                pointer = self._parse_current(path)
            except ValueError as error:
                records.append({"current_path": str(path), "error": str(error)})
            else:
                records.append(
                    {
                        **pointer.model_dump(mode="python"),
                        "current_path": str(path),
                    }
                )
        return pd.DataFrame.from_records(records)
