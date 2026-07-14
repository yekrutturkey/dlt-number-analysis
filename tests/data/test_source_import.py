"""Independent source snapshot, reconciliation, and verified-history tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from dlt_number_analysis.data import (
    ConflictResolutionRecord,
    SourceSnapshot,
    load_draws_csv,
    load_source_snapshot,
    load_verified_history,
    normalize_source_draws,
    reconcile_draw_sources,
    resolve_conflict_record,
    write_source_snapshot,
    write_verified_history,
)

FETCHED_AT = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


def _official_snapshot(*, conflicting: bool = False) -> SourceSnapshot:
    second_numbers = "06 07 08 09 11 03 04" if conflicting else "06 07 08 09 10 03 04"
    payload = {
        "value": {
            "list": [
                {
                    "lotteryDrawNum": "26002",
                    "lotteryDrawTime": "2026-01-03",
                    "lotteryDrawResult": second_numbers,
                },
                {
                    "lotteryDrawNum": "26001",
                    "lotteryDrawTime": "2026-01-01",
                    "lotteryDrawResult": "01 02 03 04 05 01 02",
                },
            ]
        }
    }
    return SourceSnapshot.from_content(
        source_name="china_sports_lottery_official",
        source_url="https://example.test/official",
        source_format="sporttery_json",
        content=json.dumps(payload).encode(),
        fetched_at=FETCHED_AT,
    )


def _five_hundred_snapshot() -> SourceSnapshot:
    def row(cells: list[str]) -> str:
        return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"

    fillers = ["-", "-", "-", "-", "-", "-"]
    html = (
        '<html><table><tbody id="tdata">'
        + row(["26001", "01", "02", "03", "04", "05", "01", "02", *fillers, "2026-01-01"])
        + row(["26002", "06", "07", "08", "09", "10", "03", "04", *fillers, "2026-01-03"])
        + "</tbody></table></html>"
    )
    return SourceSnapshot.from_content(
        source_name="500_com",
        source_url="https://example.test/500",
        source_format="five_hundred_html",
        content=html.encode("gb18030"),
        fetched_at=FETCHED_AT,
    )


def test_normalize_and_reconcile_two_independent_sources() -> None:
    official = normalize_source_draws(_official_snapshot())
    independent = normalize_source_draws(_five_hundred_snapshot())
    result = reconcile_draw_sources([_official_snapshot(), _five_hundred_snapshot()])

    pd.testing.assert_frame_equal(official, independent)
    assert result.report.is_valid is True
    assert result.report.conflict_count == 0
    assert result.reconciled_draws is not None


def test_raw_snapshot_is_never_overwritten(tmp_path: Path) -> None:
    snapshot = _official_snapshot()
    raw_path, metadata_path = write_source_snapshot(snapshot, tmp_path)

    assert raw_path.read_bytes() == snapshot.content
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["content_hash"] == (
        snapshot.content_hash
    )
    assert load_source_snapshot(metadata_path) == snapshot
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        write_source_snapshot(snapshot, tmp_path)


def test_conflict_blocks_verified_write_and_requires_explicit_resolution(tmp_path: Path) -> None:
    official = _official_snapshot(conflicting=True)
    independent = _five_hundred_snapshot()
    snapshots = [official, independent]
    result = reconcile_draw_sources(snapshots)

    assert result.reconciled_draws is None
    assert result.report.blocks_backtest is True
    assert result.report.conflict_count == 1
    with pytest.raises(ValueError, match="conflicts block"):
        write_verified_history(result, snapshots, tmp_path / "draws.csv")

    resolution = ConflictResolutionRecord(
        issue="26002",
        selected_source="500_com",
        reason="manual comparison against published draw notice",
        resolved_by="test-reviewer",
        resolved_at=FETCHED_AT,
        source_content_hashes={
            snapshot.source_name: snapshot.content_hash for snapshot in snapshots
        },
    )
    chosen = resolve_conflict_record(snapshots, resolution)
    assert chosen.front_numbers == (6, 7, 8, 9, 10)


def test_verified_history_manifest_detects_later_modification(tmp_path: Path) -> None:
    snapshots = [_official_snapshot(), _five_hundred_snapshot()]
    result = reconcile_draw_sources(snapshots)
    history_path = tmp_path / "draws.csv"
    manifest = write_verified_history(result, snapshots, history_path)

    loaded = load_verified_history(history_path)
    assert len(loaded) == manifest.record_count == 2
    assert len(set(manifest.source_names)) == 2

    modified = load_draws_csv(history_path)
    modified.loc[0, "front_5"] = 11
    modified.to_csv(history_path, index=False)
    with pytest.raises(ValueError, match="modified after reconciliation"):
        load_verified_history(history_path)


def test_reconciliation_rejects_one_source() -> None:
    with pytest.raises(ValueError, match="two independent"):
        reconcile_draw_sources([_official_snapshot()])


def test_missing_issue_in_one_source_blocks_verified_reconciliation() -> None:
    payload = json.loads(_official_snapshot().content)
    payload["value"]["list"] = payload["value"]["list"][:1]
    incomplete = SourceSnapshot.from_content(
        source_name="china_sports_lottery_official",
        source_url="https://example.test/official",
        source_format="sporttery_json",
        content=json.dumps(payload).encode(),
        fetched_at=FETCHED_AT,
    )

    result = reconcile_draw_sources([incomplete, _five_hundred_snapshot()])

    assert result.reconciled_draws is None
    assert result.report.blocks_backtest is True
    assert any(finding.code == "source_record_count_mismatch" for finding in result.report.findings)
