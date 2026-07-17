"""Schema-v3 manifest, primary-key, exact-load, and resume tests."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments.cohort import build_cohort_definition_identity
from dlt_number_analysis.experiments.identity import (
    EXECUTION_IDENTITY_SCHEMA_VERSION,
    RUNNER_VERSION,
    build_experiment_execution_identity,
    build_run_context_identity,
)
from dlt_number_analysis.experiments.specs import baseline_experiment_specs
from dlt_number_analysis.experiments.storage import (
    STORAGE_SCHEMA_VERSION,
    DuplicateExperimentResultError,
)
from dlt_number_analysis.experiments.storage_v3 import (
    ExperimentResultStoreV3,
    FormalPartitionManifestV3,
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
        cohort_id="storage-test",
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
        "storage_schema_version": STORAGE_SCHEMA_VERSION,
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


def test_schema_v3_append_and_resume_use_complete_cohort(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    store = ExperimentResultStoreV3(tmp_path / "schema_v3")
    store.append(
        pd.DataFrame([_row("20005", spec=spec, context=context, identity=identity, cohort=cohort)])
    )
    status = store.status(
        spec,
        identity,
        context,
        cohort,
        seed=77,
        expected_target_issues=("20005", "20006"),
    )
    assert status.completed_target_issues == ("20005",)
    assert status.pending_target_issues == ("20006",)


def test_resume_rejects_same_count_but_different_targets(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    with pytest.raises(ValueError, match="expected targets differ"):
        ExperimentResultStoreV3(tmp_path).status(
            spec,
            identity,
            context,
            cohort,
            seed=77,
            expected_target_issues=("20005", "20007"),
        )


def test_duplicate_schema_v3_primary_key_is_rejected(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    store = ExperimentResultStoreV3(tmp_path)
    frame = pd.DataFrame(
        [_row("20005", spec=spec, context=context, identity=identity, cohort=cohort)]
    )
    store.append(frame)
    with pytest.raises(DuplicateExperimentResultError):
        store.append(frame)


def test_manifest_expected_target_hash_tamper_is_rejected(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    store = ExperimentResultStoreV3(tmp_path)
    store.append(
        pd.DataFrame(
            [
                _row("20005", spec=spec, context=context, identity=identity, cohort=cohort),
                _row("20006", spec=spec, context=context, identity=identity, cohort=cohort),
            ]
        )
    )
    path = store.manifest_path(spec, identity, cohort, seed=77)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expected_targets_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_targets_sha256"):
        store.read_partition(spec, identity, cohort, seed=77)


def test_manifest_history_tamper_is_rejected(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    store = ExperimentResultStoreV3(tmp_path)
    store.append(
        pd.DataFrame([_row("20005", spec=spec, context=context, identity=identity, cohort=cohort)])
    )
    path = store.manifest_path(spec, identity, cohort, seed=77)
    manifest = FormalPartitionManifestV3.model_validate_json(path.read_text(encoding="utf-8"))
    path.write_text(
        manifest.model_copy(update={"history_sha256": "0" * 64}).model_dump_json(), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="history_sha256"):
        store.read_partition(spec, identity, cohort, seed=77)


def test_parquet_hash_tamper_is_rejected(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    store = ExperimentResultStoreV3(tmp_path)
    store.append(
        pd.DataFrame([_row("20005", spec=spec, context=context, identity=identity, cohort=cohort)])
    )
    path = store.partition_path(spec, identity, cohort, seed=77)
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="Parquet hash"):
        store.read_partition(spec, identity, cohort, seed=77)


def test_schema_v3_status_does_not_scan_schema_v2_sibling(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    marker = tmp_path / "schema_v2" / "observations.parquet"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"legacy marker")
    status = ExperimentResultStoreV3(tmp_path / "schema_v3").status(
        spec,
        identity,
        context,
        cohort,
        seed=77,
        expected_target_issues=("20005", "20006"),
    )
    assert status.completed_target_issues == ()
    assert status.pending_target_issues == ("20005", "20006")


def test_exact_load_selects_only_requested_cohort(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    store = ExperimentResultStoreV3(tmp_path)
    store.append(
        pd.DataFrame(
            [
                _row("20005", spec=spec, context=context, identity=identity, cohort=cohort),
                _row("20006", spec=spec, context=context, identity=identity, cohort=cohort),
            ]
        )
    )
    selected = store.load_cohort(
        run_context_sha256=context.run_context_sha256,
        cohort_definition_sha256=cohort.cohort_definition_sha256,
        phase="development",
        experiment_ids=("B1",),
        seeds=(77,),
    )
    assert selected["target_issue"].astype(str).tolist() == ["20005", "20006"]


def test_formal_load_rejects_incomplete_partition(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    store = ExperimentResultStoreV3(tmp_path)
    store.append(
        pd.DataFrame([_row("20005", spec=spec, context=context, identity=identity, cohort=cohort)])
    )
    with pytest.raises(ValueError, match="incomplete"):
        store.load_cohort(
            run_context_sha256=context.run_context_sha256,
            cohort_definition_sha256=cohort.cohort_definition_sha256,
            phase="development",
            experiment_ids=("B1",),
            seeds=(77,),
        )


def test_manifest_contains_complete_cohort_and_parquet_identity(tmp_path: Path) -> None:
    spec, context, identity, cohort = _identities()
    store = ExperimentResultStoreV3(tmp_path)
    store.append(
        pd.DataFrame([_row("20005", spec=spec, context=context, identity=identity, cohort=cohort)])
    )
    manifest = FormalPartitionManifestV3.model_validate_json(
        store.manifest_path(spec, identity, cohort, seed=77).read_text(encoding="utf-8")
    )
    assert manifest.cohort_definition_sha256 == cohort.cohort_definition_sha256
    assert manifest.expected_targets_sha256 == cohort.payload.expected_targets_sha256
    assert manifest.completed_target_count == 1
