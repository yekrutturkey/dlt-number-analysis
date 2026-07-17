"""v0.5.6 logical-cohort identity and task-chunk independence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlt_number_analysis.experiments.cohort import (
    COHORT_IDENTITY_SCHEMA_VERSION,
    build_cohort_definition_identity,
    task_targets_sha256,
)
from dlt_number_analysis.experiments.scheduler import build_process_tasks
from dlt_number_analysis.experiments.specs import baseline_experiment_specs
from dlt_number_analysis.experiments.storage import ExperimentResultStoreV2
from dlt_number_analysis.experiments.storage_v3 import (
    EXPERIMENT_PRIMARY_KEY_V3,
    ExperimentResultStoreV3,
)

PHASE_TARGETS = ("100", "101", "102", "103", "104", "105")


def _cohort(
    targets: tuple[str, ...] = ("100", "101", "102"),
    **kwargs: object,
):
    options = {
        "cohort_purpose": "development_smoke",
        "phase": "development",
        "cohort_id": "smoke",
        "phase_target_issues": PHASE_TARGETS,
    }
    options.update(kwargs)
    return build_cohort_definition_identity(targets, **options)  # type: ignore[arg-type]


def test_target_input_order_does_not_change_cohort_hash() -> None:
    assert (
        _cohort(("102", "100", "101")).cohort_definition_sha256
        == _cohort(("100", "101", "102")).cohort_definition_sha256
    )


@pytest.mark.parametrize(
    ("field", "first", "second"),
    (
        ("chunk_size", 2, 17),
        ("worker_count", 1, 8),
        ("task_order", (1, 2), (2, 1)),
        ("pending_target_issues", ("100",), ("101", "102")),
    ),
)
def test_scheduler_or_resume_metadata_does_not_change_cohort_hash(
    field: str,
    first: object,
    second: object,
) -> None:
    assert (
        _cohort(**{field: first}).cohort_definition_sha256
        == _cohort(**{field: second}).cohort_definition_sha256
    )


def test_complete_target_set_changes_cohort_hash() -> None:
    assert (
        _cohort(("100", "101")).cohort_definition_sha256
        != _cohort(("100", "101", "102")).cohort_definition_sha256
    )


def test_cohort_purpose_changes_cohort_hash() -> None:
    assert (
        _cohort().cohort_definition_sha256
        != _cohort(cohort_purpose="development_full").cohort_definition_sha256
    )


def test_phase_changes_cohort_hash() -> None:
    other = build_cohort_definition_identity(
        ("100", "101", "102"),
        cohort_purpose="calibration",
        phase="calibration",
        cohort_id="smoke",
        phase_target_issues=PHASE_TARGETS,
    )
    assert _cohort().cohort_definition_sha256 != other.cohort_definition_sha256


def test_cohort_id_changes_cohort_hash() -> None:
    assert (
        _cohort().cohort_definition_sha256 != _cohort(cohort_id="another").cohort_definition_sha256
    )


def test_duplicate_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        _cohort(("100", "100"))


def test_out_of_phase_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        _cohort(("100", "999"))


def test_empty_target_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        _cohort(())


def test_identity_contains_expected_schema_and_derived_fields() -> None:
    cohort = _cohort(("102", "100", "101"))
    assert cohort.payload.cohort_identity_schema_version == COHORT_IDENTITY_SCHEMA_VERSION
    assert cohort.payload.ordered_target_issues == ("100", "101", "102")
    assert (cohort.payload.target_count, cohort.payload.start_issue, cohort.payload.end_issue) == (
        3,
        "100",
        "102",
    )


def test_task_hash_depends_on_chunk_not_logical_cohort() -> None:
    assert task_targets_sha256(("100", "101")) != task_targets_sha256(("102",))


def test_two_scheduler_chunks_receive_same_complete_cohort() -> None:
    specs = baseline_experiment_specs(seeds=(20260000,), phase="development")[1:7]
    cohort = _cohort(PHASE_TARGETS)
    tasks = build_process_tasks(
        specs,
        PHASE_TARGETS,
        master_seed=20260000,
        chunk_size=3,
        logical_cohort=cohort,
    )
    assert len(tasks) == 2
    assert {task.logical_cohort.cohort_definition_sha256 for task in tasks} == {
        cohort.cohort_definition_sha256
    }
    assert len({task.task_targets_sha256 for task in tasks}) == 2


def test_schema_v3_path_contains_full_cohort_hash() -> None:
    spec = baseline_experiment_specs(seeds=(1,), phase="development")[1]
    identity_type = __import__(
        "dlt_number_analysis.experiments.identity", fromlist=["ExperimentExecutionIdentity"]
    ).ExperimentExecutionIdentity
    identity = identity_type(
        experiment_config_sha256="1" * 64,
        run_context_sha256="2" * 64,
        execution_config_sha256="3" * 64,
    )
    cohort = _cohort()
    path = ExperimentResultStoreV3("schema_v3").partition_path(spec, identity, cohort, seed=1)
    assert cohort.cohort_definition_sha256 in path.parts


def test_schema_v3_primary_key_contains_cohort_hash() -> None:
    assert "cohort_definition_sha256" in EXPERIMENT_PRIMARY_KEY_V3


def test_schema_v2_path_signature_remains_unchanged(tmp_path: Path) -> None:
    spec = baseline_experiment_specs(seeds=(1,), phase="development")[1]
    identity_type = __import__(
        "dlt_number_analysis.experiments.identity", fromlist=["ExperimentExecutionIdentity"]
    ).ExperimentExecutionIdentity
    identity = identity_type(
        experiment_config_sha256="1" * 64,
        run_context_sha256="2" * 64,
        execution_config_sha256="3" * 64,
    )
    path = ExperimentResultStoreV2(tmp_path / "schema_v2").partition_path(spec, identity, seed=1)
    assert path.parts[-2:] == ("seed_1", "observations.parquet")
    assert _cohort().cohort_definition_sha256 not in path.parts
