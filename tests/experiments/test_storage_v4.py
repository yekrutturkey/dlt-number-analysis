"""Schema-v4 immutable-generation transaction and recovery tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments import storage_v4 as storage_v4_module
from dlt_number_analysis.experiments.cohort import build_cohort_definition_identity
from dlt_number_analysis.experiments.identity import (
    EXECUTION_IDENTITY_SCHEMA_VERSION,
    RUNNER_VERSION,
    build_experiment_execution_identity,
    build_run_context_identity,
)
from dlt_number_analysis.experiments.manual_operations import (
    CurrentPointerRepairAuditV2,
    audit_manual_operations,
)
from dlt_number_analysis.experiments.specs import baseline_experiment_specs
from dlt_number_analysis.experiments.storage import (
    STORAGE_SCHEMA_VERSION_V4,
    DuplicateExperimentResultError,
)
from dlt_number_analysis.experiments.storage_v4 import (
    EXPERIMENT_PRIMARY_KEY_V4,
    CurrentPointerRepairAudit,
    CurrentPointerV4,
    ExperimentResultStoreV4,
    FormalPartitionManifestV4,
)


def _history(count: int = 20) -> pd.DataFrame:
    start = date(2020, 1, 1)
    return pd.DataFrame(
        [
            [
                str(20001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                1,
                2,
                3,
                4,
                5 + index % 15,
                1,
                2 + index % 10,
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def _identities(targets: tuple[str, ...] = ("20005", "20006")):
    spec = baseline_experiment_specs(seeds=(77,), phase="development")[1]
    draws = _history()
    context = build_run_context_identity(
        draws,
        data_split_spec=spec.data_split,
        phase="development",
        evaluation_mode="raw_observation",
        profile="fast",
        portfolio_scoring_method="numpy_vectorized",
        minimum_history=3,
        minimum_bank_size=500,
        maximum_bank_search_trials=80_000,
    )
    identity = build_experiment_execution_identity(context, spec)
    cohort = build_cohort_definition_identity(
        targets,
        cohort_purpose="development_smoke",
        phase="development",
        cohort_id="storage-v4-test",
        phase_target_issues=tuple(draws.iloc[:12]["issue"].astype(str)),
    )
    return spec, context, identity, cohort


def _row(target: str, *, spec, context, identity, cohort) -> dict[str, object]:
    return {
        "experiment_id": spec.experiment_id,
        "experiment_version": spec.experiment_version,
        "experiment_spec_json": spec.model_dump_json(),
        "experiment_config_sha256": identity.experiment_config_sha256,
        "run_context_sha256": identity.run_context_sha256,
        "execution_config_sha256": identity.execution_config_sha256,
        "execution_identity_schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "history_sha256": context.payload.history_sha256,
        "evaluation_mode": context.payload.evaluation_mode,
        "profile": context.payload.profile,
        "requested_portfolio_scoring_method": context.payload.portfolio_scoring_method,
        "portfolio_scoring_method": "random_bank_sample",
        "storage_schema_version": STORAGE_SCHEMA_VERSION_V4,
        "identity_status": "formal_verified",
        "formal_inference_eligible": True,
        "phase": spec.data_split.phase,
        "seed": 77,
        "target_issue": target,
        "data_cutoff_issue": str(int(target) - 1),
        "cohort_definition_json": cohort.model_dump_json(),
        "cohort_identity_schema_version": cohort.payload.cohort_identity_schema_version,
        "cohort_purpose": cohort.payload.cohort_purpose,
        "cohort_id": cohort.payload.cohort_id,
        "cohort_definition_sha256": cohort.cohort_definition_sha256,
        "expected_targets_sha256": cohort.payload.expected_targets_sha256,
        "cohort_target_count": cohort.payload.target_count,
        "cohort_start_issue": cohort.payload.start_issue,
        "cohort_end_issue": cohort.payload.end_issue,
        "task_id": "task-0",
        "task_index": 0,
        "task_targets_sha256": "4" * 64,
        "task_target_count": 1,
        "best_front_hits": 1,
        "best_back_hits": 0,
        "best_total_hits": 1,
        "at_least_three_front": False,
        "at_least_2_plus_1": False,
        "unique_hit_concentration": 0.5,
        "any_prize": False,
        "total_prize": 0.0,
        "roi": -1.0,
        "draw_date": "2020-01-01",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_target(store: ExperimentResultStoreV4, target: str, identities) -> None:
    spec, context, identity, cohort = identities
    store.append(
        pd.DataFrame([_row(target, spec=spec, context=context, identity=identity, cohort=cohort)])
    )


def _first_append_orphan(tmp_path: Path):
    identities = _identities(targets=("20005",))
    spec, context, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path)
    frame = pd.DataFrame(
        [_row("20005", spec=spec, context=context, identity=identity, cohort=cohort)]
    )

    def fail(point: str) -> None:
        if point == "after_generation_rename_before_current_write":
            raise RuntimeError("first append interrupted")

    with pytest.raises(RuntimeError, match="first append interrupted"):
        store.append(frame, _fault_injector=fail)
    return identities, store, frame


def test_schema_v4_path_primary_key_and_two_generations(tmp_path: Path) -> None:
    identities = _identities()
    spec, context, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path / "schema_v4")
    status = store.status(
        spec,
        identity,
        context,
        cohort,
        seed=77,
        expected_target_issues=cohort.payload.ordered_target_issues,
    )
    assert status.pending_target_issues == ("20005", "20006")
    assert not status.current_path.exists()
    assert "execution_config_sha256" in EXPERIMENT_PRIMARY_KEY_V4
    assert "cohort_definition_sha256" in EXPERIMENT_PRIMARY_KEY_V4
    partition = store.partition_directory(spec, identity, cohort, seed=77)
    assert identity.execution_config_sha256 in partition.parts
    assert cohort.cohort_definition_sha256 in partition.parts

    _append_target(store, "20005", identities)
    first = CurrentPointerV4.model_validate_json(
        store.current_path(spec, identity, cohort, seed=77).read_text(encoding="utf-8")
    )
    first_manifest = FormalPartitionManifestV4.model_validate_json(
        (
            store.generation_directory(
                spec, identity, cohort, seed=77, generation_id=first.generation_id
            )
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    first_generation = store.generation_directory(
        spec, identity, cohort, seed=77, generation_id=first.generation_id
    )
    first_parquet_sha = _sha256(first_generation / "observations.parquet")
    first_manifest_sha = _sha256(first_generation / "manifest.json")
    _append_target(store, "20006", identities)
    second = CurrentPointerV4.model_validate_json(
        store.current_path(spec, identity, cohort, seed=77).read_text(encoding="utf-8")
    )
    assert second.generation_id != first.generation_id
    assert first_manifest.parent_generation_id is None
    assert first_manifest.observation_count == 1
    assert first_manifest.committed is True
    assert _sha256(first_generation / "observations.parquet") == first_parquet_sha
    assert _sha256(first_generation / "manifest.json") == first_manifest_sha
    second_manifest = FormalPartitionManifestV4.model_validate_json(
        (
            store.generation_directory(
                spec, identity, cohort, seed=77, generation_id=second.generation_id
            )
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert second_manifest.parent_generation_id == first.generation_id
    assert store.read_partition(spec, identity, cohort, seed=77)["target_issue"].tolist() == [
        "20005",
        "20006",
    ]


@pytest.mark.parametrize(
    "fault_point",
    [
        "before_parquet_write",
        "after_parquet_before_manifest",
        "after_manifest_before_generation_rename",
        "after_generation_rename_before_current_write",
        "after_current_temp_before_replace",
    ],
)
def test_faults_before_current_switch_preserve_parent_and_resume(
    tmp_path: Path,
    fault_point: str,
) -> None:
    identities = _identities()
    spec, context, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path / fault_point)
    _append_target(store, "20005", identities)
    current_path = store.current_path(spec, identity, cohort, seed=77)
    old_current = current_path.read_bytes()
    pointer = CurrentPointerV4.model_validate_json(old_current)
    old_generation = store.generation_directory(
        spec, identity, cohort, seed=77, generation_id=pointer.generation_id
    )
    old_parquet_sha = _sha256(old_generation / "observations.parquet")
    old_manifest_sha = _sha256(old_generation / "manifest.json")

    def fail_at(point: str) -> None:
        if point == fault_point:
            raise RuntimeError(f"injected {point}")

    frame = pd.DataFrame(
        [_row("20006", spec=spec, context=context, identity=identity, cohort=cohort)]
    )
    with pytest.raises(RuntimeError, match="injected"):
        store.append(frame, _fault_injector=fail_at)
    assert current_path.read_bytes() == old_current
    assert _sha256(old_generation / "observations.parquet") == old_parquet_sha
    assert _sha256(old_generation / "manifest.json") == old_manifest_sha
    assert store.read_partition(spec, identity, cohort, seed=77)["target_issue"].tolist() == [
        "20005"
    ]
    status = store.status(
        spec,
        identity,
        context,
        cohort,
        seed=77,
        expected_target_issues=cohort.payload.ordered_target_issues,
    )
    assert status.pending_target_issues == ("20006",)
    audit = store.audit_partition_generations(spec, identity, cohort, seed=77)
    assert audit.current_is_valid
    assert audit.orphan_generation_ids or audit.invalid_generation_ids
    store.append(frame)
    assert store.status(
        spec,
        identity,
        context,
        cohort,
        seed=77,
        expected_target_issues=cohort.payload.ordered_target_issues,
    ).is_complete


def test_fault_after_current_switch_commits_and_duplicate_is_rejected(tmp_path: Path) -> None:
    identities = _identities()
    spec, context, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path)
    _append_target(store, "20005", identities)
    old_current = store.current_path(spec, identity, cohort, seed=77).read_bytes()
    frame = pd.DataFrame(
        [_row("20006", spec=spec, context=context, identity=identity, cohort=cohort)]
    )

    def fail_after_switch(point: str) -> None:
        if point == "after_current_replace":
            raise RuntimeError("caller crashed")

    with pytest.raises(RuntimeError, match="caller crashed"):
        store.append(frame, _fault_injector=fail_after_switch)
    assert store.current_path(spec, identity, cohort, seed=77).read_bytes() != old_current
    assert store.status(
        spec,
        identity,
        context,
        cohort,
        seed=77,
        expected_target_issues=cohort.payload.ordered_target_issues,
    ).is_complete
    with pytest.raises(DuplicateExperimentResultError):
        store.append(frame)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("corrupt_current", "CURRENT is invalid"),
        ("missing_generation", "missing generation"),
        ("wrong_manifest_sha", "manifest SHA"),
        ("tamper_manifest", "manifest SHA"),
        ("tamper_parquet", "Parquet hash"),
    ],
)
def test_current_and_generation_tampering_is_rejected(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    identities = _identities()
    spec, _, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path / mutation)
    _append_target(store, "20005", identities)
    current_path = store.current_path(spec, identity, cohort, seed=77)
    current = CurrentPointerV4.model_validate_json(current_path.read_text(encoding="utf-8"))
    generation = store.generation_directory(
        spec, identity, cohort, seed=77, generation_id=current.generation_id
    )
    if mutation == "corrupt_current":
        current_path.write_text("not json", encoding="utf-8")
    elif mutation == "missing_generation":
        current_path.write_text(
            current.model_copy(update={"generation_id": "missing"}).model_dump_json(),
            encoding="utf-8",
        )
    elif mutation == "wrong_manifest_sha":
        current_path.write_text(
            current.model_copy(update={"manifest_sha256": "0" * 64}).model_dump_json(),
            encoding="utf-8",
        )
    elif mutation == "tamper_manifest":
        with (generation / "manifest.json").open("ab") as stream:
            stream.write(b" ")
    else:
        with (generation / "observations.parquet").open("ab") as stream:
            stream.write(b"tamper")
    with pytest.raises(ValueError, match=message):
        store.read_partition(spec, identity, cohort, seed=77)


def test_orphan_is_audited_not_selected_and_repair_is_explicit(tmp_path: Path) -> None:
    identities = _identities()
    spec, _, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path)
    _append_target(store, "20005", identities)
    current_path = store.current_path(spec, identity, cohort, seed=77)
    first = CurrentPointerV4.model_validate_json(current_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(
        [_row("20006", spec=spec, context=identities[1], identity=identity, cohort=cohort)]
    )

    def fail_before_current(point: str) -> None:
        if point == "after_generation_rename_before_current_write":
            raise RuntimeError("orphan")

    with pytest.raises(RuntimeError, match="orphan"):
        store.append(frame, _fault_injector=fail_before_current)
    audit = store.audit_partition_generations(spec, identity, cohort, seed=77)
    assert audit.current_generation_id == first.generation_id
    assert len(audit.orphan_generation_ids) == 1
    assert store.read_partition(spec, identity, cohort, seed=77)["target_issue"].tolist() == [
        "20005"
    ]
    repair_path = store.repair_current_pointer(
        spec,
        identity,
        cohort,
        seed=77,
        generation_id=audit.orphan_generation_ids[0],
        reason="recover fully validated interrupted append",
    )
    repair = CurrentPointerRepairAudit.model_validate_json(repair_path.read_text(encoding="utf-8"))
    assert repair.requested_generation_id == audit.orphan_generation_ids[0]
    assert repair.reason.startswith("recover")
    assert repair.status == "completed"
    assert store.read_partition(spec, identity, cohort, seed=77)["target_issue"].tolist() == [
        "20005",
        "20006",
    ]
    invalid_generation = store.generation_directory(
        spec,
        identity,
        cohort,
        seed=77,
        generation_id=first.generation_id,
    )
    with (invalid_generation / "manifest.json").open("ab") as stream:
        stream.write(b"not-json")
    with pytest.raises(ValueError, match="not fully valid"):
        store.repair_current_pointer(
            spec,
            identity,
            cohort,
            seed=77,
            generation_id=first.generation_id,
            reason="invalid selection",
        )


def test_multiple_valid_orphans_never_change_current_automatically(tmp_path: Path) -> None:
    identities = _identities()
    spec, _, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path)
    _append_target(store, "20005", identities)
    current_path = store.current_path(spec, identity, cohort, seed=77)
    original = current_path.read_bytes()
    frame = pd.DataFrame(
        [_row("20006", spec=spec, context=identities[1], identity=identity, cohort=cohort)]
    )

    def fail(point: str) -> None:
        if point == "after_generation_rename_before_current_write":
            raise RuntimeError("orphan")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="orphan"):
            store.append(frame, _fault_injector=fail)
    audit = store.audit_partition_generations(spec, identity, cohort, seed=77)
    assert len(audit.orphan_generation_ids) == 2
    assert current_path.read_bytes() == original


def test_first_append_interruption_requires_explicit_audited_repair(tmp_path: Path) -> None:
    identities, store, frame = _first_append_orphan(tmp_path)
    spec, context, identity, cohort = identities
    current_path = store.current_path(spec, identity, cohort, seed=77)
    assert not current_path.exists()
    audit = store.audit_partition_generations(spec, identity, cohort, seed=77)
    assert audit.current_generation_id is None
    assert len(audit.valid_generation_ids) == 1
    assert audit.orphan_generation_ids == audit.valid_generation_ids
    assert not audit.current_is_valid
    assert audit.recovery_possible
    with pytest.raises(ValueError, match="CURRENT is missing"):
        store.read_partition(spec, identity, cohort, seed=77)
    with pytest.raises(ValueError, match="CURRENT is missing"):
        store.status(
            spec,
            identity,
            context,
            cohort,
            seed=77,
            expected_target_issues=("20005",),
        )
    repair_path = store.repair_current_pointer(
        spec,
        identity,
        cohort,
        seed=77,
        generation_id=audit.orphan_generation_ids[0],
        reason="recover validated first-append generation",
    )
    repair = CurrentPointerRepairAuditV2.model_validate_json(
        repair_path.read_text(encoding="utf-8")
    )
    assert repair.status == "completed"
    assert current_path.exists()
    assert store.read_partition(spec, identity, cohort, seed=77)["target_issue"].tolist() == [
        "20005"
    ]
    assert store.status(
        spec,
        identity,
        context,
        cohort,
        seed=77,
        expected_target_issues=("20005",),
    ).is_complete
    with pytest.raises(DuplicateExperimentResultError):
        store.append(frame)
    manual = audit_manual_operations((tmp_path,))
    assert len(manual.completed_operations) == 1
    assert not manual.inconsistent_operations
    current_path.unlink()
    inconsistent = audit_manual_operations((tmp_path,))
    assert len(inconsistent.inconsistent_operations) == 1


@pytest.mark.parametrize(
    "fault_point",
    ("before_parquet_write", "after_manifest_before_generation_rename"),
)
def test_first_append_early_fault_has_no_repairable_generation(
    tmp_path: Path,
    fault_point: str,
) -> None:
    identities = _identities(targets=("20005",))
    spec, context, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path / fault_point)
    frame = pd.DataFrame(
        [_row("20005", spec=spec, context=context, identity=identity, cohort=cohort)]
    )

    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("early first-append fault")

    with pytest.raises(RuntimeError, match="early first-append fault"):
        store.append(frame, _fault_injector=fail)
    assert not store.current_path(spec, identity, cohort, seed=77).exists()
    audit = store.audit_partition_generations(spec, identity, cohort, seed=77)
    assert not audit.valid_generation_ids
    assert not audit.orphan_generation_ids
    assert audit.invalid_generation_ids
    assert not audit.recovery_possible
    official = tuple(
        path
        for path in store.generations_directory(spec, identity, cohort, seed=77).iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    assert not official
    assert store.read_partition(spec, identity, cohort, seed=77).empty


@pytest.mark.parametrize(
    ("fault_point", "current_changed", "audit_exists"),
    (
        ("before_prepare_audit_write", False, False),
        ("after_prepare_audit_commit", False, True),
        ("before_current_replace", False, True),
        ("after_current_replace", True, True),
        ("before_completed_audit_update", True, True),
    ),
)
def test_current_repair_faults_preserve_prepared_evidence(
    tmp_path: Path,
    fault_point: str,
    current_changed: bool,
    audit_exists: bool,
) -> None:
    identities, store, _ = _first_append_orphan(tmp_path)
    spec, _, identity, cohort = identities
    generation_id = store.audit_partition_generations(
        spec, identity, cohort, seed=77
    ).orphan_generation_ids[0]

    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("repair interrupted")

    with pytest.raises(RuntimeError, match="repair interrupted"):
        store.repair_current_pointer(
            spec,
            identity,
            cohort,
            seed=77,
            generation_id=generation_id,
            reason="fault-injected repair",
            _fault_injector=fail,
        )
    current_path = store.current_path(spec, identity, cohort, seed=77)
    assert current_path.exists() is current_changed
    audit_paths = tuple(tmp_path.rglob("repair_*.json"))
    assert bool(audit_paths) is audit_exists
    if audit_exists:
        record = CurrentPointerRepairAuditV2.model_validate_json(
            audit_paths[0].read_text(encoding="utf-8")
        )
        assert record.status == "prepared"
        manual = audit_manual_operations((tmp_path,))
        assert bool(manual.inconsistent_operations) is current_changed


def test_current_replace_failure_keeps_original_and_records_failed_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities, store, _ = _first_append_orphan(tmp_path)
    spec, _, identity, cohort = identities
    generation_id = store.audit_partition_generations(
        spec, identity, cohort, seed=77
    ).orphan_generation_ids[0]

    def deny_replace(_: Path, __: Path) -> None:
        raise OSError("replace denied")

    monkeypatch.setattr(storage_v4_module, "_replace_current", deny_replace)
    with pytest.raises(RuntimeError, match="replacement failed"):
        store.repair_current_pointer(
            spec,
            identity,
            cohort,
            seed=77,
            generation_id=generation_id,
            reason="replace failure",
        )
    assert not store.current_path(spec, identity, cohort, seed=77).exists()
    audit_path = next(tmp_path.rglob("repair_*.json"))
    audit = CurrentPointerRepairAuditV2.model_validate_json(audit_path.read_text(encoding="utf-8"))
    assert audit.status == "failed"
    assert audit.failure_message == "replace denied"


def test_current_temporary_write_failure_keeps_original_and_records_failed_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities, store, _ = _first_append_orphan(tmp_path)
    spec, _, identity, cohort = identities
    generation_id = store.audit_partition_generations(
        spec, identity, cohort, seed=77
    ).orphan_generation_ids[0]

    def deny_write(_: Path, __: object) -> None:
        raise OSError("write denied")

    monkeypatch.setattr(storage_v4_module, "_write_json_fsync", deny_write)
    with pytest.raises(RuntimeError, match="temporary CURRENT write failed"):
        store.repair_current_pointer(
            spec,
            identity,
            cohort,
            seed=77,
            generation_id=generation_id,
            reason="temporary write failure",
        )
    assert not store.current_path(spec, identity, cohort, seed=77).exists()
    audit_path = next(tmp_path.rglob("repair_*.json"))
    audit = CurrentPointerRepairAuditV2.model_validate_json(audit_path.read_text(encoding="utf-8"))
    assert audit.status == "failed"
    assert audit.failure_message == "write denied"


def test_missing_current_with_official_generation_is_not_an_empty_partition(
    tmp_path: Path,
) -> None:
    identities = _identities()
    spec, _, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path)
    _append_target(store, "20005", identities)
    store.current_path(spec, identity, cohort, seed=77).unlink()
    with pytest.raises(ValueError, match="CURRENT is missing"):
        store.read_partition(spec, identity, cohort, seed=77)
    with pytest.raises(ValueError, match="CURRENT is missing"):
        store.load_cohort(
            run_context_sha256=identity.run_context_sha256,
            cohort_definition_sha256=cohort.cohort_definition_sha256,
            phase="development",
            experiment_ids=(spec.experiment_id,),
            seeds=(77,),
        )


def test_schema_v4_does_not_scan_schema_v3_sibling(tmp_path: Path) -> None:
    identities = _identities()
    spec, context, identity, cohort = identities
    marker = tmp_path / "schema_v3" / "observations.parquet"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"v3 marker")
    status = ExperimentResultStoreV4(tmp_path / "schema_v4").status(
        spec,
        identity,
        context,
        cohort,
        seed=77,
        expected_target_issues=cohort.payload.ordered_target_issues,
    )
    assert status.completed_target_issues == ()


def test_target_outside_cohort_and_duplicate_primary_key_are_rejected(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    store = ExperimentResultStoreV4(tmp_path)
    outside = pd.DataFrame(
        [_row("20007", spec=spec, context=context, identity=identity, cohort=cohort)]
    )
    with pytest.raises(ValueError, match="outside the logical cohort"):
        store.append(outside)
    duplicate = pd.DataFrame(
        [
            _row("20005", spec=spec, context=context, identity=identity, cohort=cohort),
            _row("20005", spec=spec, context=context, identity=identity, cohort=cohort),
        ]
    )
    with pytest.raises(DuplicateExperimentResultError):
        store.append(duplicate)


def test_completed_count_tamper_is_rejected(tmp_path: Path) -> None:
    identities = _identities()
    spec, _, identity, cohort = identities
    store = ExperimentResultStoreV4(tmp_path)
    _append_target(store, "20005", identities)
    current_path = store.current_path(spec, identity, cohort, seed=77)
    current = CurrentPointerV4.model_validate_json(current_path.read_text(encoding="utf-8"))
    generation = store.generation_directory(
        spec, identity, cohort, seed=77, generation_id=current.generation_id
    )
    manifest_path = generation / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["completed_target_count"] = 2
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    current_path.write_text(
        current.model_copy(update={"manifest_sha256": _sha256(manifest_path)}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed count"):
        store.read_partition(spec, identity, cohort, seed=77)
