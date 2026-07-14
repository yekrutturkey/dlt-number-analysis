"""Screening, development, calibration, and locked holdout plan tests."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments import (
    build_staged_ablation_plan,
    select_stage_target_issues,
)


def _history(count: int = 150) -> pd.DataFrame:
    start = date(2026, 1, 1)
    return pd.DataFrame(
        [
            [
                str(26001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                1,
                7,
                13,
                19,
                25,
                1,
                7,
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def test_staged_ablation_counts_and_target_stride() -> None:
    screening = build_staged_ablation_plan("screening")
    development = build_staged_ablation_plan("full_development")
    calibration = build_staged_ablation_plan("calibration", calibration_count=7)

    assert len(screening.specifications) == 360
    assert len(screening.seeds) == 1
    assert len(development.specifications) == 30
    assert len(development.seeds) == 3
    assert len(calibration.specifications) == 7
    assert len(calibration.seeds) == 5
    targets = select_stage_target_issues(_history(), screening, minimum_history=10)
    assert len(targets) > 1
    assert int(targets[1]) - int(targets[0]) == 5


def test_final_holdout_requires_one_explicitly_frozen_specification() -> None:
    with pytest.raises(ValueError, match="explicitly frozen"):
        build_staged_ablation_plan("final_holdout")

    candidate = build_staged_ablation_plan("calibration").specifications[0]
    holdout = build_staged_ablation_plan(
        "final_holdout",
        frozen_specification=candidate,
    )

    assert holdout.final_holdout_locked is True
    assert holdout.phase == "final_holdout"
    assert len(holdout.specifications) == 1
