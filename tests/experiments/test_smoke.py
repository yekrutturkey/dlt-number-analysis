"""Bounded paired-smoke audit and resume tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dlt_number_analysis.experiments.smoke import (
    SMOKE_EXPERIMENT_IDS,
    comparisons_frame,
    smoke_comparisons,
    validate_paired_smoke_observations,
    verify_smoke_storage,
)
from dlt_number_analysis.experiments.specs import baseline_experiment_specs
from dlt_number_analysis.experiments.storage import ExperimentResultStore


def _observations() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for target_index, target_issue in enumerate(("08009", "08010")):
        for experiment_index, experiment_id in enumerate(SMOKE_EXPERIMENT_IDS):
            records.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_version": f"smoke-{experiment_id}",
                    "phase": "development",
                    "seed": 20260000,
                    "target_issue": target_issue,
                    "data_cutoff_issue": str(int(target_issue) - 1),
                    "candidate_numbers_hash": f"candidate-{target_issue}",
                    "constraints_signature": "shared-constraints",
                    "bank_hash": f"bank-{target_issue}",
                    "bank_size": 500 + target_index * 100,
                    "bank_acceptance_rate": 0.2 + target_index * 0.1,
                    "bank_search_expansion_count": target_index,
                    "target_bank_generation_count": 1,
                    "target_portfolio_bank_reuse_count": 5,
                    "best_front_hits": 1 + (experiment_index % 2),
                    "best_total_hits": 2 + (experiment_index % 2),
                    "at_least_three_front": False,
                    "at_least_2_plus_1": experiment_index > 0,
                    "unique_hit_concentration": 0.4 + experiment_index * 0.01,
                    "any_prize": None,
                    "roi": None,
                }
            )
    return pd.DataFrame.from_records(records)


def test_paired_smoke_validates_sharing_and_bank_distribution() -> None:
    observations = _observations()

    audit = validate_paired_smoke_observations(
        observations,
        target_issues=("08009", "08010"),
        seed=20260000,
        minimum_bank_size=500,
        resume_status_verified=True,
        duplicate_primary_key_rejected=True,
        completed_task_skip_verified=True,
    )
    comparisons = smoke_comparisons(
        observations,
        target_issues=("08009", "08010"),
        seed=20260000,
        bootstrap_resamples=100,
        permutations=100,
    )

    assert audit.target_count == 2
    assert audit.bank_size.minimum == 500
    assert audit.bank_size.median == 550
    assert audit.bank_size.mean == 550
    assert audit.bank_size.maximum == 600
    assert audit.expanded_target_count == 1
    assert set(comparisons_frame(comparisons)["experiment_id"]) == {
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
    }


def test_partial_partitions_resume_then_reject_duplicate(tmp_path: Path) -> None:
    store = ExperimentResultStore(tmp_path / "experiments")
    observations = _observations()
    specifications = tuple(
        spec
        for spec in baseline_experiment_specs(seeds=(20260000,), phase="development")
        if spec.experiment_id in SMOKE_EXPERIMENT_IDS
    )
    first_target = observations.loc[observations["target_issue"] == "08009"]
    second_target = observations.loc[observations["target_issue"] == "08010"]
    store.append(first_target)

    statuses = [
        store.status(
            spec,
            seed=20260000,
            expected_target_issues=("08009", "08010"),
        )
        for spec in specifications
    ]
    assert all(status.pending_target_issues == ("08010",) for status in statuses)

    store.append(second_target)
    resume_complete, duplicate_rejected = verify_smoke_storage(
        store,
        specifications,
        seed=20260000,
        target_issues=("08009", "08010"),
    )

    assert resume_complete is True
    assert duplicate_rejected is True
