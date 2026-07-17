"""Non-mutating timeline, source-page, and verified-prefix integrity audits."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Protocol

import pandas as pd

from dlt_number_analysis.data.data_validator import CSV_COLUMNS, validate_draw_dataframe
from dlt_number_analysis.data.history_store import DataQualityIssue, DataQualityReport
from dlt_number_analysis.data.identity import canonical_history_sha256


class SnapshotAuditInput(Protocol):
    """Minimal immutable-snapshot surface consumed by page audits."""

    source_name: str
    source_format: str
    content: bytes
    content_hash: str


def yearly_issue_gap_report(draws: pd.DataFrame) -> tuple[DataQualityIssue, ...]:
    """Report missing within-year issue numbers without inventing replacement records."""
    validated = validate_draw_dataframe(draws)
    issues = validated["issue"].astype(str).tolist()
    findings: list[DataQualityIssue] = []
    for previous, current in pairwise(issues):
        previous_year, current_year = previous[:2], current[:2]
        previous_sequence, current_sequence = int(previous[2:]), int(current[2:])
        if previous_year == current_year and current_sequence != previous_sequence + 1:
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="yearly_issue_gap",
                    message=(
                        f"issue sequence jumps from {previous} to {current}; "
                        "missing records must be reconciled from sources"
                    ),
                    issue=current,
                )
            )
        elif previous_year != current_year and current_sequence != 1:
            findings.append(
                DataQualityIssue(
                    severity="warning",
                    code="year_boundary_not_starting_at_001",
                    message=f"new issue-year boundary starts at {current}",
                    issue=current,
                )
            )
    return tuple(findings)


def draw_date_anomaly_report(
    draws: pd.DataFrame,
    *,
    long_gap_days: int = 10,
) -> tuple[DataQualityIssue, ...]:
    """Report date-order errors and unusually long pauses without changing dates."""
    if long_gap_days < 1:
        raise ValueError("long_gap_days must be positive")
    missing = set(CSV_COLUMNS).difference(draws.columns)
    if missing:
        raise ValueError(f"draw date audit is missing columns: {sorted(missing)}")
    issues = draws["issue"].astype(str).tolist()
    dates = [pd.Timestamp(value).date() for value in draws["draw_date"].tolist()]
    findings: list[DataQualityIssue] = []
    duplicate_counts = Counter(dates)
    for duplicated in sorted(value for value, count in duplicate_counts.items() if count > 1):
        findings.append(
            DataQualityIssue(
                severity="error",
                code="duplicate_draw_date",
                message=f"draw date {duplicated.isoformat()} occurs more than once",
            )
        )
    for index, (previous, current) in enumerate(pairwise(dates), start=1):
        issue = issues[index]
        delta = (current - previous).days
        if delta <= 0:
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="non_increasing_draw_date",
                    message=f"draw date is not increasing at issue {issue}",
                    issue=issue,
                )
            )
        elif delta > long_gap_days:
            findings.append(
                DataQualityIssue(
                    severity="warning",
                    code="unusually_long_draw_date_gap",
                    message=f"{delta}-day calendar gap precedes issue {issue}",
                    issue=issue,
                )
            )
    for issue, draw_date in zip(issues, dates, strict=True):
        if len(issue) >= 5 and int(issue[:2]) != draw_date.year % 100:
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="issue_year_draw_date_mismatch",
                    message=f"issue {issue} does not match draw year {draw_date.year}",
                    issue=issue,
                )
            )
    return tuple(findings)


def source_page_duplicate_check(snapshot: SnapshotAuditInput) -> tuple[DataQualityIssue, ...]:
    """Detect duplicated/missing official pages and issue overlap across page bodies."""
    if snapshot.source_format != "sporttery_json_pages":
        return ()
    payload = json.loads(snapshot.content.decode("utf-8-sig"))
    pages = payload.get("pages") if isinstance(payload, dict) else None
    if not isinstance(pages, list) or not pages:
        return (
            DataQualityIssue(
                severity="error",
                code="invalid_source_page_bundle",
                message=f"{snapshot.source_name} page bundle is empty or invalid",
                sources=(snapshot.source_name,),
            ),
        )
    findings: list[DataQualityIssue] = []
    page_numbers: list[int] = []
    page_hashes: list[str] = []
    issues_seen: dict[str, int] = {}
    for raw_page in pages:
        if not isinstance(raw_page, dict):
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="invalid_source_page_entry",
                    message=f"{snapshot.source_name} contains a non-object page entry",
                    sources=(snapshot.source_name,),
                )
            )
            continue
        page_number = int(raw_page.get("page_number", -1))
        body = base64.b64decode(str(raw_page.get("body_base64", "")), validate=True)
        body_hash = hashlib.sha256(body).hexdigest()
        page_numbers.append(page_number)
        page_hashes.append(body_hash)
        if body_hash != raw_page.get("content_hash"):
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="source_page_hash_mismatch",
                    message=f"page {page_number} body hash does not match its metadata",
                    sources=(snapshot.source_name,),
                )
            )
        page_payload = json.loads(body.decode("utf-8-sig"))
        value = page_payload.get("value", {}) if isinstance(page_payload, dict) else {}
        records = value.get("list", []) if isinstance(value, dict) else []
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict) or "lotteryDrawNum" not in record:
                continue
            issue = str(record["lotteryDrawNum"])
            if issue in issues_seen:
                findings.append(
                    DataQualityIssue(
                        severity="error",
                        code="source_page_duplicate_issue",
                        message=(
                            f"issue {issue} appears on pages {issues_seen[issue]} and {page_number}"
                        ),
                        issue=issue,
                        sources=(snapshot.source_name,),
                    )
                )
            issues_seen[issue] = page_number
    for value, count in Counter(page_numbers).items():
        if count > 1:
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="source_page_number_duplicate",
                    message=f"page number {value} appears {count} times",
                    sources=(snapshot.source_name,),
                )
            )
    for value, count in Counter(page_hashes).items():
        if count > 1:
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="source_page_content_duplicate",
                    message=f"page content hash {value} appears {count} times",
                    sources=(snapshot.source_name,),
                )
            )
    expected_pages = list(range(1, len(pages) + 1))
    if sorted(page_numbers) != expected_pages:
        findings.append(
            DataQualityIssue(
                severity="error",
                code="source_page_sequence_gap",
                message="official source page numbers are not a contiguous 1-based sequence",
                sources=(snapshot.source_name,),
            )
        )
    return tuple(findings)


def source_record_count_check(
    sources: Mapping[str, pd.DataFrame],
) -> tuple[DataQualityIssue, ...]:
    """Require independent sources to contain the same issue set and record count."""
    if len(sources) < 2:
        raise ValueError("source record-count audit requires at least two sources")
    issue_sets = {
        name: set(validate_draw_dataframe(frame)["issue"].astype(str).tolist())
        for name, frame in sources.items()
    }
    counts = {name: len(issues) for name, issues in issue_sets.items()}
    findings: list[DataQualityIssue] = []
    if len(set(counts.values())) > 1:
        findings.append(
            DataQualityIssue(
                severity="error",
                code="source_record_count_mismatch",
                message="independent source record counts differ",
                sources=tuple(sorted(sources)),
                values_by_source={name: str(count) for name, count in sorted(counts.items())},
            )
        )
    union = set().union(*issue_sets.values())
    for name, issues in sorted(issue_sets.items()):
        missing = sorted(union.difference(issues), key=int)
        if missing:
            preview = ", ".join(missing[:5])
            findings.append(
                DataQualityIssue(
                    severity="error",
                    code="source_missing_issues",
                    message=(f"source {name} is missing {len(missing)} issue(s); first: {preview}"),
                    issue=missing[0],
                    sources=(name,),
                )
            )
    return tuple(findings)


def verified_history_prefix_check(
    history: pd.DataFrame,
    *,
    cutoff_issue: str,
    expected_canonical_sha256: str,
    expected_record_count: int | None = None,
) -> tuple[DataQualityIssue, ...]:
    """Verify a canonical historical prefix while allowing later draws to be appended."""
    validated = validate_draw_dataframe(history)
    matching = validated.index[validated["issue"].astype(str) == cutoff_issue].tolist()
    if len(matching) != 1:
        return (
            DataQualityIssue(
                severity="error",
                code="verified_prefix_cutoff_missing",
                message=f"verified prefix cutoff issue {cutoff_issue} is missing",
                issue=cutoff_issue,
            ),
        )
    prefix = validated.iloc[: matching[0] + 1].copy()
    findings: list[DataQualityIssue] = []
    if expected_record_count is not None and len(prefix) != expected_record_count:
        findings.append(
            DataQualityIssue(
                severity="error",
                code="verified_prefix_record_count_mismatch",
                message=(
                    f"verified prefix expected {expected_record_count} records "
                    f"but found {len(prefix)}"
                ),
                issue=cutoff_issue,
            )
        )
    actual_hash = canonical_history_sha256(prefix)
    if actual_hash != expected_canonical_sha256:
        findings.append(
            DataQualityIssue(
                severity="error",
                code="verified_prefix_hash_mismatch",
                message="verified history prefix was modified",
                issue=cutoff_issue,
                values_by_source={
                    "expected": expected_canonical_sha256,
                    "actual": actual_hash,
                },
            )
        )
    return tuple(findings)


def generate_history_integrity_report(
    draws: pd.DataFrame,
    *,
    source_names: Sequence[str] = ("provided_frame",),
) -> DataQualityReport:
    """Combine timeline audits; warnings remain visible and errors block experiments."""
    validated = validate_draw_dataframe(draws)
    findings = (*yearly_issue_gap_report(validated), *draw_date_anomaly_report(validated))
    has_errors = any(finding.severity == "error" for finding in findings)
    return DataQualityReport(
        record_count=len(validated),
        data_start_issue=str(validated.iloc[0]["issue"]),
        data_cutoff_issue=str(validated.iloc[-1]["issue"]),
        source_names=tuple(source_names),
        findings=findings,
        conflict_count=0,
        is_valid=not has_errors,
        blocks_backtest=has_errors,
    )
