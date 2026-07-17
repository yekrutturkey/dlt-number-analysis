"""The bounded identity smoke selects and validates exactly three dynamic targets."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments.identity_smoke import (
    V055_SMOKE_EXPERIMENT_IDS,
    _validate_observation_identity,
    select_v055_identity_smoke_targets,
)
from dlt_number_analysis.experiments.specs import baseline_experiment_specs
from dlt_number_analysis.experiments.splits import split_history


def _history(count: int = 300) -> pd.DataFrame:
    start = date(2015, 1, 1)
    return pd.DataFrame(
        [
            [
                str(15001 + index),
                (start + timedelta(days=index)).isoformat(),
                1,
                2,
                3,
                4,
                5 + index % 30,
                1,
                2 + index % 10,
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def test_identity_smoke_selects_first_median_last_eligible_development_targets() -> None:
    history = _history()
    selected = select_v055_identity_smoke_targets(history, minimum_history=100)
    spec = baseline_experiment_specs(seeds=(20260000,), phase="development")[1]
    development = set(split_history(history, spec.data_split).development["issue"].astype(str))
    eligible = tuple(
        issue
        for index, issue in enumerate(history["issue"].astype(str))
        if index >= 100 and issue in development
    )

    assert selected == (eligible[0], eligible[len(eligible) // 2], eligible[-1])
    assert len(set(selected)) == 3


def test_identity_smoke_validator_requires_exactly_three_targets_and_18_rows() -> None:
    targets = ("15101", "15141", "15180")
    rows = []
    for target in targets:
        for experiment_id in V055_SMOKE_EXPERIMENT_IDS:
            rows.append(
                {
                    "target_issue": target,
                    "experiment_id": experiment_id,
                    "run_context_sha256": "a" * 64,
                    "execution_config_sha256": experiment_id.lower().ljust(64, "0"),
                    "requested_portfolio_scoring_method": "numpy_vectorized",
                    "portfolio_scoring_method": (
                        "random_bank_sample" if experiment_id == "B1" else "numpy_vectorized"
                    ),
                    "history_sha256": "b" * 64,
                }
            )

    _validate_observation_identity(pd.DataFrame(rows), targets=targets)
    with pytest.raises(ValueError, match="exactly 18"):
        _validate_observation_identity(pd.DataFrame(rows[:-1]), targets=targets)
