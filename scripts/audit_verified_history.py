"""Regenerate the non-mutating verified-history integrity report."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import (
    VerifiedHistoryManifest,
    generate_history_integrity_report,
    load_verified_history,
    verified_manifest_path,
)
from dlt_number_analysis.experiments import write_data_quality_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, default=PROJECT_ROOT / "data/raw/draws.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/data_quality.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    draws = load_verified_history(args.draws)
    manifest = VerifiedHistoryManifest.model_validate_json(
        verified_manifest_path(args.draws).read_text(encoding="utf-8")
    )
    report = generate_history_integrity_report(draws, source_names=manifest.source_names)
    write_data_quality_report(report, manifest, args.output)
    print(
        f"history integrity: records={report.record_count}; "
        f"warnings={sum(item.severity == 'warning' for item in report.findings)}; "
        f"errors={sum(item.severity == 'error' for item in report.findings)}"
    )
    print(DISCLAIMER)
    return 0 if report.is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
