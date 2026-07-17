"""Historical timeline, source page, count, and prefix audit tests."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime

import pandas as pd

from dlt_number_analysis.data import (
    CSV_COLUMNS,
    SourceSnapshot,
    canonical_history_sha256,
    draw_date_anomaly_report,
    generate_history_integrity_report,
    source_page_duplicate_check,
    source_record_count_check,
    verified_history_prefix_check,
    yearly_issue_gap_report,
)


def _draws(issues: tuple[str, ...], dates: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [issue, draw_date, 1, 7, 13, 19, 25, 1, 7]
            for issue, draw_date in zip(issues, dates, strict=True)
        ],
        columns=CSV_COLUMNS,
    )


def test_true_issue_gap_is_error_but_long_date_gap_is_warning() -> None:
    gap = _draws(("26001", "26003"), ("2026-01-01", "2026-01-03"))
    long_pause = _draws(("26001", "26002"), ("2026-01-01", "2026-01-20"))

    assert yearly_issue_gap_report(gap)[0].severity == "error"
    date_findings = draw_date_anomaly_report(long_pause)
    report = generate_history_integrity_report(long_pause)

    assert date_findings[0].code == "unusually_long_draw_date_gap"
    assert date_findings[0].severity == "warning"
    assert report.is_valid is True
    assert report.blocks_backtest is False


def test_source_record_count_mismatch_is_blocking_error() -> None:
    full = _draws(("26001", "26002"), ("2026-01-01", "2026-01-03"))
    short = full.iloc[:1].copy()

    findings = source_record_count_check({"official": full, "independent": short})

    assert {finding.code for finding in findings} == {
        "source_record_count_mismatch",
        "source_missing_issues",
    }
    assert all(finding.severity == "error" for finding in findings)


def test_official_page_overlap_is_detected() -> None:
    def body(page_number: int) -> bytes:
        return json.dumps(
            {
                "value": {
                    "list": [{"lotteryDrawNum": "26001"}],
                    "pages": 2,
                    "pageNo": page_number,
                }
            }
        ).encode()

    pages = []
    for page_number in (1, 2):
        raw = body(page_number)
        pages.append(
            {
                "page_number": page_number,
                "content_hash": hashlib.sha256(raw).hexdigest(),
                "body_base64": base64.b64encode(raw).decode("ascii"),
            }
        )
    content = json.dumps({"pages": pages}).encode()
    snapshot = SourceSnapshot.from_content(
        source_name="official",
        source_url="https://example.test",
        source_format="sporttery_json_pages",
        content=content,
        fetched_at=datetime(2026, 7, 14, tzinfo=UTC),
    )

    findings = source_page_duplicate_check(snapshot)

    assert any(finding.code == "source_page_duplicate_issue" for finding in findings)


def test_verified_prefix_allows_append_but_rejects_history_change() -> None:
    prefix = _draws(("26001", "26002"), ("2026-01-01", "2026-01-03"))
    appended = pd.concat(
        [
            prefix,
            _draws(("26003",), ("2026-01-05",)),
        ],
        ignore_index=True,
    )
    expected_hash = canonical_history_sha256(prefix)

    assert not verified_history_prefix_check(
        appended,
        cutoff_issue="26002",
        expected_canonical_sha256=expected_hash,
        expected_record_count=2,
    )
    modified = appended.copy()
    modified.loc[0, "front_5"] = 26
    findings = verified_history_prefix_check(
        modified,
        cutoff_issue="26002",
        expected_canonical_sha256=expected_hash,
        expected_record_count=2,
    )

    assert findings[0].code == "verified_prefix_hash_mismatch"
