"""Partitioned append-only Parquet experiment storage tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments import (
    DataSplitSpec,
    DuplicateExperimentResultError,
    ExperimentResultStore,
    baseline_experiment_specs,
)


def _row(*, target_issue: str, phase: str = "development") -> dict[str, object]:
    return {
        "experiment_id": "B1",
        "experiment_version": "v0.5.1-b1-shared-bank-v1",
        "phase": phase,
        "seed": 77,
        "target_issue": target_issue,
        "data_cutoff_issue": str(int(target_issue) - 1),
        "best_front_hits": 1,
    }


def test_partition_store_appends_resumes_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    store = ExperimentResultStore(tmp_path / "outputs" / "experiments")
    spec = baseline_experiment_specs(seeds=(77,))[1]

    first_path = store.append(pd.DataFrame([_row(target_issue="26001")]))[0]
    partial = store.status(spec, seed=77, expected_target_issues=("26001", "26002"))
    store.append(pd.DataFrame([_row(target_issue="26002")]))
    complete = store.status(spec, seed=77, expected_target_issues=("26001", "26002"))

    assert (
        first_path
        == tmp_path / "outputs" / "experiments" / "development" / "B1" / "seed_77.parquet"
    )
    assert partial.pending_target_issues == ("26002",)
    assert complete.is_complete is True
    assert store.read_partition("development", "B1", 77)["target_issue"].tolist() == [
        "26001",
        "26002",
    ]
    assert set(store.load_all()["risk_disclaimer"]) == {DISCLAIMER}
    with pytest.raises(DuplicateExperimentResultError, match="already contains"):
        store.append(pd.DataFrame([_row(target_issue="26002")]))


def test_phases_are_physically_isolated(tmp_path: Path) -> None:
    store = ExperimentResultStore(tmp_path / "experiments")
    calibration_spec = baseline_experiment_specs(seeds=(77,))[1].model_copy(
        update={"data_split": DataSplitSpec(phase="calibration")}
    )
    store.append(pd.DataFrame([_row(target_issue="26003", phase="calibration")]))

    status = store.status(
        calibration_spec,
        seed=77,
        expected_target_issues=("26003",),
    )

    assert status.is_complete is True
    assert "calibration" in status.partition_path.parts
    assert store.load_all(phase="development").empty


def test_existing_v051_smoke_parquet_remains_readable() -> None:
    project_root = Path(__file__).resolve().parents[2]
    store = ExperimentResultStore(project_root / "outputs" / "experiments")

    b1 = store.read_partition("development", "B1", 20260000)
    b2 = store.read_partition("development", "B2", 20260000)

    assert len(b1) == 101
    assert len(b2) == 100
    assert b1["target_issue"].iloc[0] == "08008"
    assert b2["target_issue"].tolist() == [f"{issue:05d}" for issue in range(8009, 8109)]
