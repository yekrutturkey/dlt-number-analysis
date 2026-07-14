"""Chronological split and final-holdout lock tests."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments import (
    acquire_holdout_lock,
    baseline_experiment_specs,
    finalize_holdout_lock,
    split_history,
)


def _history(count: int = 10) -> pd.DataFrame:
    start = date(2026, 1, 1)
    return pd.DataFrame(
        [
            [
                str(26001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                1,
                2,
                3,
                4,
                5,
                1,
                2,
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def test_time_split_is_contiguous_60_20_20() -> None:
    split = split_history(_history())

    assert list(map(len, (split.development, split.calibration, split.final_holdout))) == [
        6,
        2,
        2,
    ]
    assert split.development.iloc[-1]["issue"] == "26006"
    assert split.calibration.iloc[0]["issue"] == "26007"
    assert split.final_holdout.iloc[0]["issue"] == "26009"


def test_final_holdout_version_can_be_reserved_and_completed_only_once(
    tmp_path: Path,
) -> None:
    spec = baseline_experiment_specs(phase="final_holdout")[5]
    lock_path = tmp_path / "holdout-lock.json"

    reserved = acquire_holdout_lock(spec, lock_path)
    assert reserved.status == "reserved_before_run"
    completed = finalize_holdout_lock(
        lock_path,
        experiment_version=spec.experiment_version,
        result_bytes=b"formal result",
    )
    assert completed.status == "completed"
    assert completed.result_sha256 is not None

    with pytest.raises(ValueError, match="already consumed"):
        acquire_holdout_lock(spec, lock_path)
    with pytest.raises(ValueError, match="already finalized"):
        finalize_holdout_lock(
            lock_path,
            experiment_version=spec.experiment_version,
            result_bytes=b"second result",
        )


def test_changed_parameters_require_a_new_holdout_version(tmp_path: Path) -> None:
    spec = baseline_experiment_specs(phase="final_holdout")[5]
    acquire_holdout_lock(spec, tmp_path / "lock.json")
    changed = spec.model_copy(update={"number_score_weight": 0.75, "structure_score_weight": 0.25})

    with pytest.raises(ValueError, match="use a new experiment version"):
        acquire_holdout_lock(changed, tmp_path / "lock.json")
