"""Physical isolation, strict identity, and explicit legacy import tests."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments.command import build_parser
from dlt_number_analysis.experiments.identity import (
    EXECUTION_IDENTITY_SCHEMA_VERSION,
    RUNNER_VERSION,
    build_experiment_execution_identity,
    build_run_context_identity,
)
from dlt_number_analysis.experiments.specs import ExperimentSpec, baseline_experiment_specs
from dlt_number_analysis.experiments.storage import (
    STORAGE_SCHEMA_VERSION,
    DuplicateExperimentResultError,
    ExperimentResultStoreV2,
    FormalPartitionManifest,
    LegacyObservationImportStore,
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


def _spec_and_identity(*, evaluation_mode: str = "raw_observation"):
    spec = baseline_experiment_specs(seeds=(77,), phase="development")[1]
    context = build_run_context_identity(
        _history(),
        data_split_spec=spec.data_split,
        phase="development",
        evaluation_mode=evaluation_mode,  # type: ignore[arg-type]
        profile="fast",
        portfolio_scoring_method="numpy_vectorized",
        minimum_history=10,
        minimum_bank_size=500,
        maximum_bank_search_trials=80_000,
    )
    return spec, context, build_experiment_execution_identity(context, spec)


def _row(
    spec: ExperimentSpec,
    *,
    target_issue: str,
    context,
    identity,
) -> dict[str, object]:
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
        "git_commit_sha": "0" * 40,
        "evaluation_mode": context.payload.evaluation_mode,
        "profile": context.payload.profile,
        "portfolio_scoring_method": "random_bank_sample",
        "requested_portfolio_scoring_method": context.payload.portfolio_scoring_method,
        "storage_schema_version": STORAGE_SCHEMA_VERSION,
        "identity_status": "formal_verified",
        "formal_inference_eligible": True,
        "phase": spec.data_split.phase,
        "seed": 77,
        "target_issue": target_issue,
        "data_cutoff_issue": str(int(target_issue) - 1),
        "best_front_hits": 1,
    }


def test_schema_v2_path_uses_full_hashes_and_is_physically_isolated(tmp_path: Path) -> None:
    spec, _, identity = _spec_and_identity()
    store = ExperimentResultStoreV2(tmp_path / "schema_v2")

    path = store.partition_path(spec, identity, seed=77)

    assert identity.run_context_sha256 in path.parts
    assert identity.execution_config_sha256 in path.parts
    assert len(identity.run_context_sha256) == len(identity.execution_config_sha256) == 64
    assert path.name == "observations.parquet"


@pytest.mark.parametrize("unsafe", ("../B1", "B/1", "B\\1", ".."))
def test_schema_v2_rejects_unsafe_path_segments(tmp_path: Path, unsafe: str) -> None:
    spec, _, identity = _spec_and_identity()
    unsafe_spec = spec.model_copy(update={"experiment_id": unsafe})

    with pytest.raises(ValueError, match="not safe"):
        ExperimentResultStoreV2(tmp_path).partition_path(unsafe_spec, identity, seed=77)


def test_status_requires_complete_v055_identity_and_ignores_old_version(tmp_path: Path) -> None:
    spec, context, identity = _spec_and_identity()
    old_spec = spec.model_copy(update={"experiment_version": "v0.5.1-b1-shared-bank-v1"})
    old_identity = build_experiment_execution_identity(context, old_spec)
    store = ExperimentResultStoreV2(tmp_path / "schema_v2")
    store.append(
        pd.DataFrame(
            [_row(old_spec, target_issue="20011", context=context, identity=old_identity)]
        ),
        expected_target_issues_by_experiment={"B1": ("20011",)},
    )

    status = store.status(
        spec,
        seed=77,
        expected_target_issues=("20011",),
        identity=identity,
        run_context=context,
    )

    assert status.pending_target_issues == ("20011",)
    assert status.is_complete is False


def test_raw_and_full_contexts_do_not_mark_each_other_complete(tmp_path: Path) -> None:
    raw_spec, raw_context, raw_identity = _spec_and_identity()
    full_spec, full_context, full_identity = _spec_and_identity(evaluation_mode="full_resampling")
    store = ExperimentResultStoreV2(tmp_path / "schema_v2")
    store.append(
        pd.DataFrame(
            [_row(raw_spec, target_issue="20011", context=raw_context, identity=raw_identity)]
        ),
        expected_target_issues_by_experiment={"B1": ("20011",)},
    )

    assert store.status(
        full_spec,
        seed=77,
        expected_target_issues=("20011",),
        identity=full_identity,
        run_context=full_context,
    ).pending_target_issues == ("20011",)


def test_different_history_contexts_do_not_mark_each_other_complete(tmp_path: Path) -> None:
    spec, context, identity = _spec_and_identity()
    changed_history = _history()
    changed_history.loc[0, "front_5"] = 6
    changed_context = build_run_context_identity(
        changed_history,
        data_split_spec=spec.data_split,
        phase="development",
        evaluation_mode="raw_observation",
        profile="fast",
        portfolio_scoring_method="numpy_vectorized",
        minimum_history=10,
        minimum_bank_size=500,
        maximum_bank_search_trials=80_000,
    )
    changed_identity = build_experiment_execution_identity(changed_context, spec)
    store = ExperimentResultStoreV2(tmp_path / "schema_v2")
    store.append(
        pd.DataFrame([_row(spec, target_issue="20011", context=context, identity=identity)]),
        expected_target_issues_by_experiment={"B1": ("20011",)},
    )

    assert store.status(
        spec,
        seed=77,
        expected_target_issues=("20011",),
        identity=changed_identity,
        run_context=changed_context,
    ).pending_target_issues == ("20011",)


def test_schema_v2_append_rejects_full_primary_key_duplicate(tmp_path: Path) -> None:
    spec, context, identity = _spec_and_identity()
    row = pd.DataFrame([_row(spec, target_issue="20011", context=context, identity=identity)])
    store = ExperimentResultStoreV2(tmp_path / "schema_v2")
    expected = {"B1": ("20011",)}
    store.append(row, expected_target_issues_by_experiment=expected)

    with pytest.raises(DuplicateExperimentResultError, match="already contains"):
        store.append(row, expected_target_issues_by_experiment=expected)


def test_manifest_tamper_or_row_identity_mismatch_fails_read(tmp_path: Path) -> None:
    spec, context, identity = _spec_and_identity()
    store = ExperimentResultStoreV2(tmp_path / "schema_v2")
    store.append(
        pd.DataFrame([_row(spec, target_issue="20011", context=context, identity=identity)]),
        expected_target_issues_by_experiment={"B1": ("20011",)},
    )
    manifest_path = store.manifest_path(spec, identity, seed=77)
    manifest = FormalPartitionManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest_path.write_text(
        json.dumps(
            manifest.model_copy(update={"history_sha256": "0" * 64}).model_dump(mode="json"),
            default=str,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row identity differs from manifest"):
        store.read_partition(spec, identity, seed=77)


def test_legacy_migration_is_opt_in_and_formally_ineligible(tmp_path: Path) -> None:
    args = build_parser().parse_args([])
    assert args.migrate_legacy_observations is False
    source = tmp_path / "legacy.csv"
    pd.DataFrame([{"experiment_id": "B1", "target_issue": "20011"}]).to_csv(source, index=False)
    store = LegacyObservationImportStore(tmp_path / "legacy_import")

    assert store.import_csv(source) == 1
    imported = pd.read_parquet(store.path)
    assert imported.iloc[0]["identity_status"] == "legacy_unverified"
    assert bool(imported.iloc[0]["formal_inference_eligible"]) is False
    assert pd.isna(imported.iloc[0]["run_context_sha256"])
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        store.import_csv(source)
