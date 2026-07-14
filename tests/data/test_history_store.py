"""Conflict-safe history and historical prize-data interface tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from dlt_number_analysis.backtesting import run_rolling_backtest
from dlt_number_analysis.data import (
    CSV_COLUMNS,
    DataQualityIssue,
    DataQualityReport,
    DrawConflictError,
    SourceReconciliationResult,
    append_draws,
    assert_backtest_ready,
    generate_data_quality_report,
    load_issue_prizes,
    load_prize_rule_schedule,
    reconcile_sources,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_existing() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["26001", "2026-01-01", 1, 2, 3, 4, 5, 1, 2],
            ["26002", "2026-01-03", 6, 7, 8, 9, 10, 3, 4],
        ],
        columns=CSV_COLUMNS,
    )


def make_incoming() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["26002", "2026-01-03", 6, 7, 8, 9, 10, 3, 4],
            ["26003", "2026-01-05", 11, 12, 13, 14, 15, 5, 6],
        ],
        columns=CSV_COLUMNS,
    )


def test_append_draws_deduplicates_identical_records_and_appends_new_issue() -> None:
    result = append_draws(make_existing(), make_incoming())

    assert result["issue"].tolist() == ["26001", "26002", "26003"]


def test_source_conflict_is_reported_without_automatic_correction() -> None:
    conflicting = make_incoming()
    conflicting.loc[0, ["front_1", "front_2"]] = [5, 7]

    result = reconcile_sources({"official_a": make_existing(), "official_b": conflicting})

    assert result.reconciled_draws is None
    assert result.report.blocks_backtest is True
    assert result.report.conflict_count == 1
    conflict = next(item for item in result.report.findings if item.code == "source_conflict")
    assert conflict.issue == "26002"
    assert conflict.field == "front_1"
    assert conflict.values_by_source == {"official_a": "6", "official_b": "5"}

    with pytest.raises(DrawConflictError) as error:
        append_draws(make_existing(), conflicting)
    assert error.value.report.conflict_count == 1


def test_conflict_report_blocks_backtest() -> None:
    conflicting = make_incoming()
    conflicting.loc[0, ["front_1", "front_2"]] = [5, 7]
    reconciliation = reconcile_sources({"a": make_existing(), "b": conflicting})
    quality = generate_data_quality_report(make_existing(), reconciliation=reconciliation)

    with pytest.raises(ValueError, match="blocks backtesting"):
        run_rolling_backtest(make_existing(), min_history=1, data_quality_report=quality)


def test_warning_is_reported_without_invalidating_or_blocking_history() -> None:
    warning = DataQualityIssue(
        severity="warning",
        code="single_source_only",
        message="history currently has only one verified source",
        sources=("official_a",),
    )
    reconciliation = SourceReconciliationResult(
        reconciled_draws=make_existing(),
        report=DataQualityReport(
            record_count=2,
            data_start_issue="26001",
            data_cutoff_issue="26002",
            source_names=("official_a",),
            findings=(warning,),
            conflict_count=0,
            is_valid=True,
            blocks_backtest=False,
        ),
    )

    quality = generate_data_quality_report(make_existing(), reconciliation=reconciliation)

    assert quality.findings == (warning,)
    assert quality.is_valid is True
    assert quality.blocks_backtest is False
    assert_backtest_ready(quality)


def test_mixed_warning_and_error_is_invalid_and_blocks_backtest() -> None:
    warning = DataQualityIssue(
        severity="warning",
        code="single_source_only",
        message="history currently has only one verified source",
    )
    error = DataQualityIssue(
        severity="error",
        code="source_conflict",
        message="sources disagree",
        issue="26002",
    )
    reconciliation = SourceReconciliationResult(
        reconciled_draws=None,
        report=DataQualityReport(
            record_count=2,
            data_start_issue="26001",
            data_cutoff_issue="26002",
            source_names=("official_a", "official_b"),
            findings=(warning, error),
            conflict_count=1,
            is_valid=False,
            blocks_backtest=True,
        ),
    )

    quality = generate_data_quality_report(make_existing(), reconciliation=reconciliation)

    assert {finding.severity for finding in quality.findings} == {"warning", "error"}
    assert quality.is_valid is False
    assert quality.blocks_backtest is True
    with pytest.raises(ValueError, match="blocks backtesting"):
        assert_backtest_ready(quality)


def test_quality_report_rejects_warning_only_blocking_flags() -> None:
    with pytest.raises(ValidationError, match="is_valid"):
        DataQualityReport(
            record_count=1,
            data_start_issue="26001",
            data_cutoff_issue="26001",
            source_names=("official_a",),
            findings=(
                DataQualityIssue(
                    severity="warning",
                    code="single_source_only",
                    message="warning only",
                ),
            ),
            conflict_count=0,
            is_valid=False,
            blocks_backtest=True,
        )


def test_load_issue_prizes_and_rule_schedule(tmp_path: Path) -> None:
    issue_prizes_path = tmp_path / "issue_prizes.json"
    issue_prizes_path.write_text(
        json.dumps(
            [
                {
                    "issue": "26078",
                    "prize_table_version": "dlt-seven-tier-2026-v1",
                    "prize_context": "pool_at_or_above_800m",
                    "actual_amount_by_tier": {"七等奖": "7"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prizes = load_issue_prizes(issue_prizes_path)
    schedule = load_prize_rule_schedule(PROJECT_ROOT / "config" / "prize_tiers.json")

    assert prizes["26078"].actual_amount_by_tier["七等奖"] == 7
    assert schedule.table_for_issue("26078") is not None
    assert schedule.table_for_issue("25000") is None
