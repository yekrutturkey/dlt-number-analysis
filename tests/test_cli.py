"""CLI command and next-prediction audit-artifact tests."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from random import Random

import pandas as pd

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.cli import main
from dlt_number_analysis.data import CSV_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_history(path: Path, count: int = 40) -> Path:
    rng = Random(919)
    start = date(2025, 1, 1)
    frame = pd.DataFrame(
        [
            [
                str(25001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
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


def test_generate_next_saves_complete_audit_artifact(tmp_path: Path) -> None:
    draws = write_history(tmp_path / "draws.csv")
    output = tmp_path / "25041.json"

    result = main(
        [
            "generate-next",
            "--draws",
            str(draws),
            "--target-issue",
            "25041",
            "--random-seed",
            "700",
            "--scorer",
            "uniform_score",
            "--search-trials",
            "5000",
            "--output",
            str(output),
        ]
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert artifact["target_issue"] == "25041"
    assert artifact["data_cutoff_issue"] == "25040"
    assert len(artifact["git_commit_sha"]) == 40
    assert artifact["pipeline_config"]["scorer_spec"]["name"] == "uniform_score"
    assert artifact["random_seeds"] == {
        "candidate_pool": 700,
        "portfolio_optimizer": 701,
    }
    assert artifact["candidate_pool_summary"]["candidate_count"] == 10_000
    assert len(artifact["prediction"]["tickets"]) == 5
    assert artifact["risk_disclaimer"] == DISCLAIMER


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
