"""CLI command and next-prediction audit-artifact tests."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from random import Random

import pandas as pd
import pytest
from pydantic import ValidationError

import dlt_number_analysis.cli as cli_module
from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.cli import main
from dlt_number_analysis.data import CSV_COLUMNS, append_draws, load_draws_csv
from dlt_number_analysis.pipeline import (
    GitAudit,
    NextPredictionArtifact,
    load_next_prediction_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_history(path: Path, count: int = 40) -> Path:
    rng = Random(919)
    start = date(2025, 1, 1)
    frame = pd.DataFrame(
        [
            [
                str(26001 + index),
                (start.replace(year=2026) + timedelta(days=index * 2)).isoformat(),
                *sorted(rng.sample(range(1, 36), 5)),
                *sorted(rng.sample(range(1, 13), 2)),
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )
    frame.to_csv(path, index=False)
    return path


def test_validate_data_and_empty_backtest_commands_write_reports(tmp_path: Path) -> None:
    draws = PROJECT_ROOT / "data" / "raw" / "draws.csv"
    quality_output = tmp_path / "quality.json"
    backtest_output = tmp_path / "backtest.json"

    assert main(["validate-data", "--draws", str(draws), "--output", str(quality_output)]) == 0
    assert (
        main(
            [
                "run-backtest",
                "--draws",
                str(draws),
                "--output",
                str(backtest_output),
            ]
        )
        == 0
    )

    assert json.loads(quality_output.read_text(encoding="utf-8"))["is_valid"] is True
    assert json.loads(backtest_output.read_text(encoding="utf-8"))["results"] == []


def test_generate_next_artifact_to_evaluate_latest_closed_loop(tmp_path: Path) -> None:
    draws = write_history(tmp_path / "draws.csv")
    output = tmp_path / "26041.json"
    history_sha256 = hashlib.sha256(draws.read_bytes()).hexdigest()

    result = main(
        [
            "generate-next",
            "--draws",
            str(draws),
            "--target-issue",
            "26041",
            "--random-seed",
            "700",
            "--scorer",
            "hot_cold_blend_score",
            "--hot-window",
            "10",
            "--cold-window",
            "30",
            "--hot-weight",
            "0.6",
            "--decay",
            "0.9",
            "--profile",
            "fast",
            "--parallel-workers",
            "1",
            "--minimum-history-size",
            "100",
            "--allow-short-history",
            "--allow-dirty",
            "--output",
            str(output),
        ]
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert artifact["target_issue"] == "26041"
    assert artifact["data_cutoff_issue"] == "26040"
    assert len(artifact["git_commit_sha"]) == 40
    assert isinstance(artifact["git_dirty"], bool)
    assert len(artifact["git_diff_hash"]) == 64
    assert artifact["history_sha256"] == history_sha256
    assert artifact["history_record_count"] == 40
    assert artifact["history_start_issue"] == "26001"
    assert artifact["history_cutoff_issue"] == "26040"
    assert artifact["short_history_override"] is True
    assert artifact["pipeline_config"]["profile"] == "fast"
    assert artifact["pipeline_config"]["candidate_count"] == 10_000
    assert artifact["pipeline_config"]["optimizer_search_trials"] == 5_000
    assert artifact["pipeline_config"]["stable_candidate_limit"] == 2_000
    assert artifact["pipeline_config"]["parallel_workers"] == 1
    assert artifact["pipeline_config"]["scorer_spec"] == {
        "name": "hot_cold_blend_score",
        "parameters": {
            "hot_window": 10,
            "cold_window": 30,
            "hot_weight": 0.6,
            "decay": 0.9,
        },
        "version": "number-scorer-v1",
    }
    assert artifact["random_seeds"] == {
        "candidate_pool": 700,
        "portfolio_optimizer": 701,
    }
    assert artifact["candidate_pool_summary"]["candidate_count"] == 10_000
    assert len(artifact["prediction"]["tickets"]) == 5
    assert artifact["risk_disclaimer"] == DISCLAIMER
    assert load_next_prediction_artifact(output).prediction.target_issue == "26041"

    for field in (
        "target_issue",
        "data_cutoff_issue",
        "generated_at",
        "pipeline_config",
        "random_seeds",
    ):
        tampered = copy.deepcopy(artifact)
        if field == "target_issue":
            tampered[field] = "26042"
        elif field == "data_cutoff_issue":
            tampered[field] = "26039"
        elif field == "generated_at":
            tampered[field] = "2026-07-14T00:00:00Z"
        elif field == "pipeline_config":
            tampered[field]["candidate_count"] = 11_000
        else:
            tampered[field]["candidate_pool"] = 999
        with pytest.raises(ValidationError):
            NextPredictionArtifact.model_validate(tampered)

    existing = load_draws_csv(draws)
    target_date = (existing.iloc[-1]["draw_date"] + timedelta(days=2)).isoformat()
    incoming = pd.DataFrame(
        [["26041", target_date, 1, 5, 12, 24, 35, 2, 11]],
        columns=CSV_COLUMNS,
    )
    append_draws(existing, incoming).to_csv(draws, index=False)
    review_output = tmp_path / "26041_review.json"

    evaluation_result = main(
        [
            "evaluate-latest",
            "--draws",
            str(draws),
            "--prediction",
            str(output),
            "--output",
            str(review_output),
        ]
    )

    review = json.loads(review_output.read_text(encoding="utf-8"))
    assert evaluation_result == 0
    assert review["target_issue"] == "26041"
    assert len(review["evaluations"]) == 5
    assert review["risk_disclaimer"] == DISCLAIMER


def test_generate_next_rejects_dirty_git_without_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = write_history(tmp_path / "draws.csv")
    monkeypatch.setattr(
        cli_module,
        "collect_git_audit",
        lambda _: GitAudit(commit_sha="0" * 40, dirty=True, diff_hash="1" * 64),
    )

    with pytest.raises(ValueError, match="refuses a dirty Git working tree"):
        main(
            [
                "generate-next",
                "--draws",
                str(draws),
                "--target-issue",
                "26041",
                "--allow-short-history",
            ]
        )


def test_generate_next_requires_explicit_target_issue(tmp_path: Path) -> None:
    draws = write_history(tmp_path / "draws.csv")

    with pytest.raises(SystemExit):
        main(["generate-next", "--draws", str(draws)])


def test_evaluate_latest_command_uses_recorded_prize_context(tmp_path: Path) -> None:
    output = tmp_path / "review.json"

    result = main(
        [
            "evaluate-latest",
            "--output",
            str(output),
        ]
    )

    review = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert review["target_issue"] == "26078"
    assert review["total_prize"] == "7"
    assert review["roi"] == "-0.3"
    assert review["risk_disclaimer"] == DISCLAIMER
