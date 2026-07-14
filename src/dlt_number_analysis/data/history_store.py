"""Conflict-safe history append, reconciliation, quality, and prize-data loaders."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis.data.data_validator import (
    CSV_COLUMNS,
    DrawDataValidationError,
    validate_draw_dataframe,
)
from dlt_number_analysis.evaluation.history import IssuePrizeRecord, PrizeRuleSchedule
from dlt_number_analysis.evaluation.prize import PrizeTable


class DrawConflictError(DrawDataValidationError):
    """Source records disagree and therefore cannot be appended or backtested."""

    def __init__(self, report: DataQualityReport) -> None:
        super().__init__("conflicting draw records require manual resolution")
        self.report = report


class DataQualityIssue(BaseModel):
    """One machine-readable quality or source reconciliation finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Literal["error", "warning"]
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    issue: str | None = None
    field: str | None = None
    sources: tuple[str, ...] = ()
    values_by_source: dict[str, str] = Field(default_factory=dict)


class DataQualityReport(BaseModel):
    """Validation report whose errors explicitly block rolling backtests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    record_count: int = Field(ge=0)
    data_start_issue: str | None
    data_cutoff_issue: str | None
    source_names: tuple[str, ...]
    findings: tuple[DataQualityIssue, ...]
    conflict_count: int = Field(ge=0)
    is_valid: bool
    blocks_backtest: bool


@dataclass(frozen=True, slots=True)
class SourceReconciliationResult:
    """Reconciled frame only exists when every same-issue record agrees."""

    reconciled_draws: pd.DataFrame | None
    report: DataQualityReport


def _value_text(value: object) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def reconcile_sources(sources: Mapping[str, pd.DataFrame]) -> SourceReconciliationResult:
    """Compare sources without choosing a winner or silently fixing conflicts."""
    if not sources:
        raise ValueError("at least one draw source is required")
    validated_sources: dict[str, pd.DataFrame] = {}
    findings: list[DataQualityIssue] = []
    for source_name, frame in sources.items():
        if not source_name.strip():
            raise ValueError("source names must not be empty")
        try:
            validated_sources[source_name] = validate_draw_dataframe(frame)
        except DrawDataValidationError as error:
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="invalid_source_data",
                    message=str(error),
                    sources=(source_name,),
                )
            )

    issue_records: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for source_name, frame in validated_sources.items():
        for record in frame.to_dict(orient="records"):
            issue_records[str(record["issue"])][source_name] = record

    conflicts: list[DataQualityIssue] = []
    for issue, by_source in sorted(issue_records.items(), key=lambda item: int(item[0])):
        for field in CSV_COLUMNS:
            values = {
                source_name: _value_text(record[field]) for source_name, record in by_source.items()
            }
            if len(set(values.values())) > 1:
                conflicts.append(
                    DataQualityIssue(
                        severity="error",
                        code="source_conflict",
                        message=f"sources disagree for issue {issue}, field {field}",
                        issue=issue,
                        field=field,
                        sources=tuple(sorted(values)),
                        values_by_source=dict(sorted(values.items())),
                    )
                )
    findings.extend(conflicts)

    reconciled: pd.DataFrame | None = None
    if not findings:
        unique_records = [
            next(iter(by_source.values()))
            for _, by_source in sorted(issue_records.items(), key=lambda item: int(item[0]))
        ]
        try:
            reconciled = validate_draw_dataframe(pd.DataFrame(unique_records, columns=CSV_COLUMNS))
        except DrawDataValidationError as error:
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="invalid_merged_timeline",
                    message=str(error),
                    sources=tuple(sorted(sources)),
                )
            )
    issues = sorted(issue_records, key=int)
    report = DataQualityReport(
        record_count=len(issues),
        data_start_issue=issues[0] if issues else None,
        data_cutoff_issue=issues[-1] if issues else None,
        source_names=tuple(sorted(sources)),
        findings=tuple(findings),
        conflict_count=len(conflicts),
        is_valid=not findings,
        blocks_backtest=bool(findings),
    )
    return SourceReconciliationResult(reconciled_draws=reconciled, report=report)


def append_draws(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Append identical/new issues while raising on any conflicting issue."""
    result = reconcile_sources({"existing": existing, "incoming": incoming})
    if result.report.blocks_backtest or result.reconciled_draws is None:
        raise DrawConflictError(result.report)
    return result.reconciled_draws


def generate_data_quality_report(
    draws: pd.DataFrame,
    *,
    reconciliation: SourceReconciliationResult | None = None,
) -> DataQualityReport:
    """Generate a non-mutating report; conflicts and validation errors block backtests."""
    findings = list(reconciliation.report.findings if reconciliation else ())
    source_names = reconciliation.report.source_names if reconciliation else ("provided_frame",)
    try:
        validated = validate_draw_dataframe(draws)
    except DrawDataValidationError as error:
        findings.append(
            DataQualityIssue(
                severity="error",
                code="invalid_draw_history",
                message=str(error),
                sources=source_names,
            )
        )
        return DataQualityReport(
            record_count=len(draws),
            data_start_issue=None,
            data_cutoff_issue=None,
            source_names=source_names,
            findings=tuple(findings),
            conflict_count=sum(item.code == "source_conflict" for item in findings),
            is_valid=False,
            blocks_backtest=True,
        )
    return DataQualityReport(
        record_count=len(validated),
        data_start_issue=str(validated.iloc[0]["issue"]),
        data_cutoff_issue=str(validated.iloc[-1]["issue"]),
        source_names=source_names,
        findings=tuple(findings),
        conflict_count=sum(item.code == "source_conflict" for item in findings),
        is_valid=not findings,
        blocks_backtest=bool(findings),
    )


def assert_backtest_ready(report: DataQualityReport) -> None:
    """Stop a backtest when unresolved history findings are present."""
    if report.blocks_backtest:
        raise DrawDataValidationError("data quality report blocks backtesting")


def load_issue_prizes(path: str | Path) -> dict[str, IssuePrizeRecord]:
    """Load JSON array, JSON object, or JSONL issue-specific prize records."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        raw_records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        raw_records = payload.get("records", []) if isinstance(payload, dict) else payload
        if isinstance(payload, dict) and "issue" in payload:
            raw_records = [payload]
    records = [IssuePrizeRecord.model_validate(record) for record in raw_records]
    result: dict[str, IssuePrizeRecord] = {}
    for record in records:
        if record.issue in result:
            raise ValueError(f"duplicate issue prize record: {record.issue}")
        result[record.issue] = record
    return result


def load_prize_rule_schedule(
    paths: str | Path | Sequence[str | Path],
) -> PrizeRuleSchedule:
    """Load one or more versioned prize tables into an effective-issue schedule."""
    source_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    tables: list[PrizeTable] = []
    for raw_path in source_paths:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        raw_tables = payload.get("tables") if isinstance(payload, dict) else None
        if raw_tables is None:
            tables.append(PrizeTable.model_validate(payload))
        else:
            tables.extend(PrizeTable.model_validate(table) for table in raw_tables)
    return PrizeRuleSchedule(tables=tuple(tables))
