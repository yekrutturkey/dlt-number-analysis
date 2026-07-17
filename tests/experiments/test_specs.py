"""Experiment matrix and parameter-binding tests."""

from __future__ import annotations

from dlt_number_analysis.experiments import (
    ablation_experiment_specs,
    baseline_experiment_specs,
)


def test_required_baselines_are_complete_and_auditable() -> None:
    specs = baseline_experiment_specs(seeds=(11, 12))

    assert tuple(spec.experiment_id for spec in specs) == tuple(f"B{i}" for i in range(9))
    assert len({spec.experiment_version for spec in specs}) == 9
    assert all(spec.seeds == (11, 12) for spec in specs)
    assert all(spec.number_score_weight + spec.structure_score_weight == 1 for spec in specs)
    assert specs[0].portfolio_strategy == "uniform_random"
    assert specs[1].portfolio_strategy == "constraint_matched_random"
    assert specs[4].number_score_weight == 0
    assert specs[7].portfolio_strategy == "max_coverage"
    assert specs[8].portfolio_strategy == "core_rotation"


def test_ablation_grid_contains_all_360_development_combinations() -> None:
    specs = ablation_experiment_specs()

    assert len(specs) == 3 * 5 * 4 * 3 * 2 == 360
    assert len({spec.experiment_id for spec in specs}) == 360
    assert {spec.scorer_spec.parameters["window"] for spec in specs} == {10, 30, 100}
    assert {spec.scorer_spec.parameters["decay"] for spec in specs} == {
        0.85,
        0.90,
        0.93,
        0.97,
        1.0,
    }
    assert {spec.structure_score_weight for spec in specs} == {0, 0.25, 0.5, 0.75}
    assert {spec.portfolio_constraints.target_front_pool_size for spec in specs} == {
        16,
        18,
        20,
    }
    assert {spec.portfolio_constraints.min_core_front_numbers for spec in specs} == {2, 3}
    assert all(spec.data_split.phase == "development" for spec in specs)
