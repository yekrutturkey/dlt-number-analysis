"""v0.5.4 production Runner benchmark scope, checkpoint and resume tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from dlt_number_analysis.data import load_verified_history
from dlt_number_analysis.experiments.batch_runner_benchmark import (
    V054_BENCHMARK_VERSION,
    V054_EXPERIMENT_IDS,
    V054_TARGET_ISSUES,
    TargetExecutionTiming,
    V054BenchmarkConfig,
    V054BenchmarkPaths,
    aggregate_v054_benchmark,
    load_v054_target_checkpoint,
    write_v054_target_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = PROJECT_ROOT / "data/raw/draws.csv"


def _observations(target_issue: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "benchmark_version": V054_BENCHMARK_VERSION,
                "experiment_id": experiment_id,
                "experiment_version": f"test-{experiment_id}",
                "experiment_config_sha256": f"{index + 1:064x}",
                "phase": "development",
                "seed": 20260000,
                "target_issue": target_issue,
                "evaluation_mode": "raw_observation",
                "candidate_numbers_hash": "a" * 64,
                "bank_hash": "b" * 64,
            }
            for index, experiment_id in enumerate(V054_EXPERIMENT_IDS)
        ]
    )


def _timing(target_issue: str, seconds: float = 1.0) -> TargetExecutionTiming:
    return TargetExecutionTiming(
        target_issue=target_issue,
        seed=20260000,
        evaluation_mode="raw_observation",
        history_slice_seconds=0.01,
        candidate_generation_seconds=0.1,
        candidate_array_conversion_seconds=0.01,
        bank_generation_seconds=0.2,
        portfolio_index_bank_seconds=0.01,
        b1_sampling_seconds=0.01,
        score_view_seconds=0.1,
        vectorized_scoring_seconds=0.1,
        final_object_construction_seconds=0.01,
        raw_evaluation_seconds=0.01,
        observation_serialization_seconds=0.01,
        target_total_seconds=seconds,
        candidate_count=10_000,
        bank_size=500,
        acceptance_rate=0.1,
        candidate_cache_hit_rate=5 / 6,
        bank_reuse_count=5,
        peak_memory_mb=100,
        process_id=123,
    )


def test_v054_config_rejects_any_scope_expansion() -> None:
    assert V054BenchmarkConfig().target_issues == V054_TARGET_ISSUES
    with pytest.raises(ValidationError, match="08009 through 08018"):
        V054BenchmarkConfig(target_issues=V054_TARGET_ISSUES[:-1])
    with pytest.raises(ValidationError, match="B1-B6"):
        V054BenchmarkConfig(experiment_ids=("B1",))
    with pytest.raises(ValidationError, match="20260000"):
        V054BenchmarkConfig(seed=1)
    with pytest.raises(ValidationError):
        V054BenchmarkConfig(process_count=2)


def test_target_checkpoint_is_atomic_reusable_and_hash_validated(tmp_path: Path) -> None:
    draws = load_verified_history(HISTORY_PATH)
    paths = V054BenchmarkPaths(tmp_path / "outputs/benchmarks/v054")
    config = V054BenchmarkConfig()
    target = "08009"
    write_v054_target_checkpoint(
        draws,
        paths,
        target,
        config,
        _observations(target),
        _timing(target),
    )

    frame, timing = load_v054_target_checkpoint(draws, paths, target, config)

    assert len(frame) == 6
    assert timing.checkpoint_write_seconds >= 0
    assert not list(paths.checkpoints.glob("*.tmp*"))
    changed = draws.copy()
    first_front = int(changed.iloc[0]["front_1"])
    replacement = next(
        number for number in range(1, int(changed.iloc[0]["front_2"])) if number != first_front
    )
    changed.loc[0, "front_1"] = replacement
    with pytest.raises(ValueError, match="history_prefix_sha256"):
        load_v054_target_checkpoint(changed, paths, target, config)


def test_duplicate_benchmark_primary_key_is_rejected(tmp_path: Path) -> None:
    draws = load_verified_history(HISTORY_PATH)
    frame = _observations("08009")
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        write_v054_target_checkpoint(
            draws,
            V054BenchmarkPaths(tmp_path),
            "08009",
            V054BenchmarkConfig(),
            duplicate,
            _timing("08009"),
        )


def test_checkpoint_aggregation_is_read_only_and_requires_five_for_extrapolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = load_verified_history(HISTORY_PATH)
    paths = V054BenchmarkPaths(tmp_path / "outputs/benchmarks/v054")
    config = V054BenchmarkConfig()
    for index, target in enumerate(V054_TARGET_ISSUES[:5], start=1):
        write_v054_target_checkpoint(
            draws,
            paths,
            target,
            config,
            _observations(target),
            _timing(target, float(index)),
        )

    monkeypatch.setattr(
        "dlt_number_analysis.experiments.batch_runner_benchmark.run_experiment_batch",
        lambda *args, **kwargs: pytest.fail("aggregation regenerated predictions"),
    )
    summary = aggregate_v054_benchmark(HISTORY_PATH, paths.output_dir, config=config)

    assert summary["measured_target_count"] == 5
    assert summary["extrapolation"]["kind"] == "estimated"
    assert paths.observations.exists()
    assert paths.timings.exists()
    assert not (tmp_path / "outputs/experiments").exists()
