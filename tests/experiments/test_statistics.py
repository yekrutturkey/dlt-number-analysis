"""Paired inference and grouped-performance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dlt_number_analysis.experiments import (
    adjust_p_values_benjamini_hochberg,
    adjust_p_values_holm,
    paired_bootstrap_95_interval,
    paired_metric_differences,
    paired_permutation_test,
    parameter_sensitivity_report,
    performance_by_seed,
    performance_by_year,
)


def test_holm_and_benjamini_hochberg_corrections_are_monotone() -> None:
    p_values = (0.01, 0.04, 0.03, 0.002)

    holm = adjust_p_values_holm(p_values)
    benjamini = adjust_p_values_benjamini_hochberg(p_values)

    np.testing.assert_allclose(holm, (0.03, 0.06, 0.06, 0.008))
    np.testing.assert_allclose(benjamini, (0.02, 0.04, 0.04, 0.008))


def _results(offset: int, experiment_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "target_issue": str(25001 + index),
                "draw_date": f"202{5 + index // 4}-01-{index + 1:02d}",
                "seed": 100 + index % 2,
                "best_front_hits": 1 + offset,
                "best_total_hits": 2 + offset,
                "at_least_three_front": bool(offset),
                "at_least_2_plus_1": bool(offset),
                "unique_hit_concentration": 0.2 + offset * 0.1,
                "any_prize": bool(offset),
                "roi": -1.0 + offset * 0.2,
                "window": 10 if index < 4 else 30,
            }
            for index in range(8)
        ]
    )


def test_paired_differences_bootstrap_and_permutation_use_matching_pairs() -> None:
    strategy = _results(1, "B5")
    baseline = _results(0, "B1")
    differences = paired_metric_differences(
        strategy,
        baseline,
        metrics=("best_front_hits", "best_total_hits", "any_prize_rate"),
    )

    front = differences.loc[differences["metric"] == "best_front_hits", "difference"]
    interval = paired_bootstrap_95_interval(front, random_seed=7, resamples=500)
    test = paired_permutation_test(front, random_seed=8, permutations=5000)

    assert len(differences) == 24
    assert front.tolist() == [1.0] * 8
    assert interval.lower == interval.upper == 1.0
    assert test.p_value < 0.05


def test_paired_comparison_rejects_duplicate_pair_keys() -> None:
    strategy = pd.concat([_results(1, "B5"), _results(1, "B5").iloc[:1]])
    with pytest.raises(ValueError, match="duplicate paired"):
        paired_metric_differences(strategy, _results(0, "B1"))


def test_year_seed_and_parameter_sensitivity_reports() -> None:
    results = pd.concat([_results(0, "B1"), _results(1, "B5")], ignore_index=True)

    yearly = performance_by_year(results)
    seeded = performance_by_seed(results)
    sensitivity = parameter_sensitivity_report(results, parameter_columns=("window",))

    assert set(yearly["year"]) == {2025, 2026}
    assert set(seeded["seed"]) == {100, 101}
    assert set(sensitivity["parameter_value"]) == {10, 30}
    assert set(sensitivity["metric"]) == {
        "best_front_hits",
        "best_total_hits",
        "unique_hit_concentration",
    }
