"""Deterministic two-layer identity tests for formal v0.5.5 experiments."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

import pandas as pd

from dlt_number_analysis.data import CSV_COLUMNS
from dlt_number_analysis.experiments import (
    EXECUTION_IDENTITY_SCHEMA_VERSION,
    build_experiment_execution_identity,
    build_run_context_identity,
    canonical_identity_sha256,
    experiment_config_sha256,
)
from dlt_number_analysis.experiments.specs import baseline_experiment_specs


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
                5 + index % 30,
                1,
                2 + index % 10,
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def _context(
    history: pd.DataFrame,
    *,
    evaluation_mode: str = "raw_observation",
    profile: str = "fast",
    workers: int = 1,
    targets: tuple[str, ...] = ("20011",),
):
    spec = baseline_experiment_specs(seeds=(77,), phase="development")[1]
    return build_run_context_identity(
        history,
        data_split_spec=spec.data_split,
        phase="development",
        evaluation_mode=evaluation_mode,  # type: ignore[arg-type]
        profile=profile,  # type: ignore[arg-type]
        portfolio_scoring_method="numpy_vectorized",
        minimum_history=10,
        minimum_bank_size=500,
        maximum_bank_search_trials=80_000,
        worker_count=workers,
        target_issues=targets,
    )


def test_run_context_hash_is_canonical_and_deterministic() -> None:
    first = _context(_history())
    second = _context(_history())

    assert first == second
    assert first.payload.execution_identity_schema_version == EXECUTION_IDENTITY_SCHEMA_VERSION
    assert len(first.run_context_sha256) == 64


def test_dataframe_field_order_does_not_change_run_context_hash() -> None:
    payload = _context(_history()).payload.model_dump(mode="json")
    reordered = dict(reversed(tuple(payload.items())))

    assert canonical_identity_sha256(payload) == canonical_identity_sha256(reordered)


def test_experiment_spec_change_changes_only_execution_identity() -> None:
    specs = baseline_experiment_specs(seeds=(77,), phase="development")
    context = _context(_history())
    original = build_experiment_execution_identity(context, specs[1])
    changed_spec = specs[1].model_copy(update={"experiment_version": "v0.5.5-b1-test-v3"})
    changed = build_experiment_execution_identity(context, changed_spec)

    assert original.run_context_sha256 == changed.run_context_sha256
    assert original.execution_config_sha256 != changed.execution_config_sha256
    expected = hashlib.sha256(
        (context.run_context_sha256 + experiment_config_sha256(specs[1])).encode("ascii")
    ).hexdigest()
    assert original.execution_config_sha256 == expected


def test_evaluation_mode_changes_run_context_hash() -> None:
    assert (
        _context(_history()).run_context_sha256
        != _context(_history(), evaluation_mode="full_resampling").run_context_sha256
    )


def test_profile_changes_run_context_hash() -> None:
    assert (
        _context(_history()).run_context_sha256
        != _context(_history(), profile="standard").run_context_sha256
    )


def test_history_content_changes_run_context_hash() -> None:
    changed = _history()
    changed.loc[0, "front_5"] = 6

    assert _context(_history()).run_context_sha256 != _context(changed).run_context_sha256


def test_worker_count_is_excluded_from_run_context_hash() -> None:
    assert (
        _context(_history(), workers=1).run_context_sha256
        == _context(_history(), workers=8).run_context_sha256
    )


def test_target_chunk_is_excluded_from_run_context_hash() -> None:
    assert (
        _context(_history(), targets=("20011",)).run_context_sha256
        == _context(_history(), targets=("20012", "20013")).run_context_sha256
    )


def test_b1_b6_share_context_but_have_distinct_execution_identities() -> None:
    context = _context(_history())
    specs = baseline_experiment_specs(seeds=(77,), phase="development")[1:7]
    identities = [build_experiment_execution_identity(context, spec) for spec in specs]

    assert {identity.run_context_sha256 for identity in identities} == {context.run_context_sha256}
    assert len({identity.execution_config_sha256 for identity in identities}) == 6
    assert [spec.experiment_version for spec in specs] == [
        f"v0.5.5-b{index}-shared-bank-v2" for index in range(1, 7)
    ]
