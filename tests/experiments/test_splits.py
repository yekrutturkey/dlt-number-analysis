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
    build_experiment_execution_identity,
    build_run_context_identity,
    finalize_holdout_lock,
    split_history,
)


def _identity(spec, *, evaluation_mode: str = "full_resampling"):
    context = build_run_context_identity(
        _history(),
        data_split_spec=spec.data_split,
        phase="final_holdout",
        evaluation_mode=evaluation_mode,  # type: ignore[arg-type]
        profile="fast",
        portfolio_scoring_method="numpy_vectorized",
        minimum_history=1,
        minimum_bank_size=500,
        maximum_bank_search_trials=80_000,
    )
    return context, build_experiment_execution_identity(context, spec)


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
    context, identity = _identity(spec)

    reserved = acquire_holdout_lock(spec, lock_path, identity=identity, run_context=context)
    assert reserved.status == "reserved_before_run"
    completed = finalize_holdout_lock(
        lock_path,
        spec=spec,
        identity=identity,
        result_bytes=b"formal result",
    )
    assert completed.status == "completed"
    assert completed.result_sha256 is not None

    with pytest.raises(ValueError, match="already consumed"):
        acquire_holdout_lock(spec, lock_path, identity=identity, run_context=context)
    with pytest.raises(ValueError, match="already finalized"):
        finalize_holdout_lock(
            lock_path,
            spec=spec,
            identity=identity,
            result_bytes=b"second result",
        )


def test_changed_parameters_require_a_new_holdout_version(tmp_path: Path) -> None:
    spec = baseline_experiment_specs(phase="final_holdout")[5]
    context, identity = _identity(spec)
    acquire_holdout_lock(
        spec,
        tmp_path / "lock.json",
        identity=identity,
        run_context=context,
    )
    changed = spec.model_copy(update={"number_score_weight": 0.75, "structure_score_weight": 0.25})
    changed_context, changed_identity = _identity(changed)

    with pytest.raises(ValueError, match="use a new experiment version"):
        acquire_holdout_lock(
            changed,
            tmp_path / "lock.json",
            identity=changed_identity,
            run_context=changed_context,
        )


def test_final_holdout_lock_rejects_raw_observation(tmp_path: Path) -> None:
    spec = baseline_experiment_specs(phase="final_holdout")[5]
    context, identity = _identity(spec, evaluation_mode="raw_observation")

    with pytest.raises(ValueError, match="requires full_resampling"):
        acquire_holdout_lock(
            spec,
            tmp_path / "lock.json",
            identity=identity,
            run_context=context,
        )
