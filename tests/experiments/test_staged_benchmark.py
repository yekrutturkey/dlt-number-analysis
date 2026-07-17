"""v0.5.3 staged checkpoint, timeout, estimation and report tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.benchmark_checkpoints import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkValue,
    atomic_write_json,
    load_checkpoint,
    run_isolated_stage,
)
from dlt_number_analysis.experiments.staged_benchmark import (
    V053BenchmarkPaths,
    fit_object_extrapolation,
    write_v053_benchmark_report,
    write_v053_benchmark_summary,
)

PROBE_WORKER = "dlt_number_analysis.experiments.benchmark_checkpoints:benchmark_probe_worker"


def _identity(candidate_hash: str = "a" * 64) -> dict[str, object]:
    return {
        "target_issue": "08009",
        "seed": 20260000,
        "candidate_numbers_hash": candidate_hash,
        "bank_hash": "b" * 64,
    }


def test_atomic_checkpoint_round_trip_and_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "stage.json"
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        **_identity(),
        "status": "completed",
        "value": 1,
    }
    atomic_write_json(path, payload)

    assert load_checkpoint(path, expected_identity=_identity()) == payload
    assert not path.with_name(f".{path.name}.tmp").exists()
    with pytest.raises(ValueError, match="candidate_numbers_hash"):
        load_checkpoint(path, expected_identity=_identity("c" * 64))


def test_completed_stage_is_saved_immediately_and_reused(tmp_path: Path) -> None:
    path = tmp_path / "complete.json"
    first = run_isolated_stage(
        stage="probe",
        worker_path=PROBE_WORKER,
        worker_kwargs={"value": "saved"},
        checkpoint_path=path,
        checkpoint_identity=_identity(),
        timeout_seconds=5,
    )
    saved = load_checkpoint(path, expected_identity=_identity())
    second = run_isolated_stage(
        stage="probe",
        worker_path=PROBE_WORKER,
        worker_kwargs={"value": "must-not-run"},
        checkpoint_path=path,
        checkpoint_identity=_identity(),
        timeout_seconds=5,
    )

    assert first.status == "completed"
    assert saved["probe_value"] == "saved"
    assert saved["execution"]["memory_scope"] == "isolated_subprocess_peak"
    assert second.status == "skipped"
    assert second.reused_checkpoint is True
    assert load_checkpoint(path)["probe_value"] == "saved"


def test_worker_result_cannot_override_checkpoint_reserved_fields(tmp_path: Path) -> None:
    path = tmp_path / "reserved.json"
    outcome = run_isolated_stage(
        stage="reserved_probe",
        worker_path=PROBE_WORKER,
        worker_kwargs={"return_reserved_fields": True},
        checkpoint_path=path,
        checkpoint_identity=_identity(),
        timeout_seconds=5,
    )
    saved = load_checkpoint(path, expected_identity=_identity())

    assert outcome.status == "completed"
    assert saved["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert saved["status"] == "completed"


def test_timeout_actively_terminates_child_and_preserves_other_checkpoint(
    tmp_path: Path,
) -> None:
    preserved = tmp_path / "preserved.json"
    atomic_write_json(
        preserved,
        {"schema_version": BENCHMARK_SCHEMA_VERSION, **_identity(), "status": "completed"},
    )
    timed_out = tmp_path / "timeout.json"

    outcome = run_isolated_stage(
        stage="timeout_probe",
        worker_path=PROBE_WORKER,
        worker_kwargs={"delay_seconds": 5.0},
        checkpoint_path=timed_out,
        checkpoint_identity=_identity(),
        timeout_seconds=0.2,
    )

    timeout_checkpoint = load_checkpoint(timed_out, expected_identity=_identity())
    assert outcome.status == "timeout"
    assert outcome.elapsed_seconds < 3
    assert timeout_checkpoint["execution"]["timed_out"] is True
    assert load_checkpoint(preserved)["status"] == "completed"


def test_measured_estimated_and_unavailable_values_cannot_mix() -> None:
    measured = BenchmarkValue(kind="measured", value=1.5, unit="seconds", method="clock")
    estimated = BenchmarkValue(kind="estimated", value=3.0, unit="seconds", method="linear_fit")
    unavailable = BenchmarkValue(kind="unavailable", unit="seconds", method="insufficient_points")

    assert measured.kind == "measured"
    assert estimated.kind == "estimated"
    assert unavailable.value is None
    with pytest.raises(ValidationError):
        BenchmarkValue(kind="estimated", unit="seconds", method="missing")
    with pytest.raises(ValidationError):
        BenchmarkValue(kind="unavailable", value=1, unit="seconds", method="invalid")


def test_object_extrapolation_requires_two_points_and_is_marked_estimated() -> None:
    unavailable = fit_object_extrapolation(
        [{"status": "completed", "subset_size": 50, "total_seconds": 2.0}],
        1_300,
    )
    estimated = fit_object_extrapolation(
        [
            {"status": "completed", "subset_size": 50, "total_seconds": 2.0},
            {"status": "completed", "subset_size": 100, "total_seconds": 3.0},
            {"status": "completed", "subset_size": 250, "total_seconds": 6.0},
        ],
        1_300,
    )

    assert unavailable["status"] == "unavailable"
    assert estimated["status"] == "estimated"
    assert estimated["estimated_object_full_bank_seconds"]["kind"] == "estimated"
    assert estimated["central_estimate_seconds"] == pytest.approx(27.0)


def test_aggregate_does_not_require_object_stage(tmp_path: Path) -> None:
    paths = V053BenchmarkPaths(tmp_path)
    atomic_write_json(
        paths.prepare,
        {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "target_issue": "08009",
            "seed": 20260000,
            "status": "completed",
            "bank_size": 1_300,
            "prepare_wall_seconds": 1.0,
        },
    )
    atomic_write_json(
        paths.vectorized,
        {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "target_issue": "08009",
            "seed": 20260000,
            "status": "completed",
            "vectorized_full_seconds": 0.5,
            "execution": {"peak_memory_mb": 100.0},
        },
    )

    summary = write_v053_benchmark_summary(paths)
    report_path = write_v053_benchmark_report(summary, tmp_path / "report.md")
    report = report_path.read_text(encoding="utf-8")

    assert summary["vectorized_full"]["status"] == "completed"
    assert summary["object_samples"] == []
    assert summary["estimated_speedup"]["kind"] == "unavailable"
    assert "外推不可用" in report
    assert DISCLAIMER in report
