"""Formal reporting must never mix incompatible execution identities."""

from __future__ import annotations

import pandas as pd
import pytest

from dlt_number_analysis.experiments.reporting import (
    compare_to_constraint_matched_baseline,
    summarize_observations,
    summarize_raw_observation_partitions,
)


def _row(
    experiment_id: str,
    target_issue: str,
    *,
    run_hash: str = "a" * 64,
    history_hash: str = "b" * 64,
    evaluation_mode: str = "raw_observation",
    formal: bool = True,
) -> dict[str, object]:
    return {
        "phase": "development",
        "cohort_id": "identity_test",
        "experiment_id": experiment_id,
        "experiment_version": f"v-{experiment_id}",
        "experiment_config_sha256": ("c" if experiment_id == "B1" else "d") * 64,
        "execution_config_sha256": ("e" if experiment_id == "B1" else "f") * 64,
        "execution_identity_schema_version": "experiment-execution-identity-v2",
        "run_context_sha256": run_hash,
        "history_sha256": history_hash,
        "evaluation_mode": evaluation_mode,
        "profile": "fast",
        "formal_inference_eligible": formal,
        "identity_status": "formal_verified" if formal else "legacy_unverified",
        "target_issue": target_issue,
        "seed": 77,
        "draw_date": "2020-01-01",
        "best_front_hits": 2,
        "best_total_hits": 3,
        "at_least_three_front": False,
        "at_least_2_plus_1": True,
        "unique_hit_concentration": 0.5,
        "any_prize": False,
        "total_prize": 0.0,
        "roi": -1.0,
    }


def test_legacy_unverified_rows_are_excluded_from_formal_summary() -> None:
    observations = pd.DataFrame(
        [
            _row("B1", "20011"),
            _row("B1", "20012", formal=False),
        ]
    )

    summary = summarize_observations(observations)

    assert summary["observation_count"].tolist() == [1]
    assert summary["formal_inference_eligible"].tolist() == [True]


def test_mixed_run_contexts_are_reported_as_separate_groups() -> None:
    observations = pd.DataFrame(
        [
            _row("B1", "20011", run_hash="a" * 64),
            _row("B1", "20012", run_hash="9" * 64),
        ]
    )

    summary = summarize_observations(observations)

    assert len(summary) == 2
    assert set(summary["run_context_sha256"]) == {"a" * 64, "9" * 64}


def test_raw_statistics_reject_mixed_evaluation_modes() -> None:
    observations = pd.DataFrame(
        [
            _row("B1", "20011"),
            _row("B1", "20012", evaluation_mode="full_resampling"),
        ]
    )

    with pytest.raises(ValueError, match="only raw_observation"):
        summarize_raw_observation_partitions(observations)


def test_paired_comparison_requires_same_run_context_and_history() -> None:
    incompatible = pd.DataFrame(
        [
            _row("B1", "20011", run_hash="a" * 64),
            _row("B1", "20012", run_hash="a" * 64),
            _row("B2", "20011", run_hash="9" * 64),
            _row("B2", "20012", run_hash="9" * 64),
        ]
    )
    compatible = pd.DataFrame(
        [
            _row("B1", "20011"),
            _row("B1", "20012"),
            _row("B2", "20011"),
            _row("B2", "20012"),
        ]
    )

    assert (
        compare_to_constraint_matched_baseline(
            incompatible,
            minimum_paired_observations=2,
            bootstrap_resamples=100,
            permutations=100,
        )
        == ()
    )
    comparisons = compare_to_constraint_matched_baseline(
        compatible,
        minimum_paired_observations=2,
        bootstrap_resamples=100,
        permutations=100,
    )
    assert comparisons
    assert {comparison.run_context_sha256 for comparison in comparisons} == {"a" * 64}
    assert {comparison.history_sha256 for comparison in comparisons} == {"b" * 64}
