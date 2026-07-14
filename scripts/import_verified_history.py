"""Fetch two independent snapshots and write verified DLT history after exact agreement."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import (
    generate_data_quality_report,
    load_source_snapshot,
    reconcile_draw_sources,
    write_source_snapshot,
    write_verified_history,
)
from dlt_number_analysis.experiments import write_data_quality_report
from dlt_number_analysis.fetchers import FiveHundredHistoryFetcher, SportteryHistoryFetcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-directory",
        type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "draws.csv",
    )
    parser.add_argument("--official-page-size", type=int, default=30)
    parser.add_argument("--request-interval-seconds", type=float, default=0.5)
    parser.add_argument("--end-issue", default="99999")
    parser.add_argument("--official-snapshot-metadata", type=Path)
    parser.add_argument("--independent-snapshot-metadata", type=Path)
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "data_quality.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fetched_at = datetime.now(UTC)
    if bool(args.official_snapshot_metadata) != bool(args.independent_snapshot_metadata):
        raise ValueError("both snapshot metadata paths must be supplied together")
    if args.official_snapshot_metadata:
        official = load_source_snapshot(args.official_snapshot_metadata)
        independent = load_source_snapshot(args.independent_snapshot_metadata)
    else:
        official = SportteryHistoryFetcher(
            page_size=args.official_page_size,
            request_interval_seconds=args.request_interval_seconds,
        ).fetch(fetched_at=fetched_at)
        write_source_snapshot(official, args.snapshot_directory)
        independent = FiveHundredHistoryFetcher(end_issue=args.end_issue).fetch(
            fetched_at=fetched_at
        )
        write_source_snapshot(independent, args.snapshot_directory)
    snapshots = (official, independent)
    reconciliation = reconcile_draw_sources(snapshots)
    if reconciliation.reconciled_draws is None:
        empty_quality = reconciliation.report
        write_data_quality_report(empty_quality, None, args.quality_report)
        print(f"source conflicts: {empty_quality.conflict_count}")
        print(DISCLAIMER)
        return 2
    manifest = write_verified_history(reconciliation, snapshots, args.output)
    quality = generate_data_quality_report(
        reconciliation.reconciled_draws,
        reconciliation=reconciliation,
    )
    write_data_quality_report(quality, manifest, args.quality_report)
    print(
        f"verified history: {manifest.history_start_issue}-{manifest.history_cutoff_issue}; "
        f"records={manifest.record_count}; conflicts={manifest.conflict_count}"
    )
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
