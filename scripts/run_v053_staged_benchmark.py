"""Run or aggregate the actively bounded v0.5.3 single-target benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.command import PROJECT_ROOT
from dlt_number_analysis.experiments.staged_benchmark import (
    V053BenchmarkPaths,
    run_v053_staged_benchmark,
    write_v053_benchmark_report,
    write_v053_benchmark_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the single-target command without any long-experiment options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, default=PROJECT_ROOT / "data/raw/draws.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmarks/v053",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/v053_staged_benchmark.md",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main() -> int:
    """Execute bounded stages or rebuild reports only from checkpoints."""
    args = build_parser().parse_args()
    paths = V053BenchmarkPaths(args.output_dir)
    if not args.aggregate_only:
        outcomes = run_v053_staged_benchmark(args.draws, args.output_dir)
        for outcome in outcomes:
            print(
                f"{outcome.stage}: {outcome.status}; "
                f"elapsed={outcome.elapsed_seconds:.3f}s; "
                f"checkpoint={outcome.checkpoint_path}"
            )
    summary = write_v053_benchmark_summary(paths)
    write_v053_benchmark_report(summary, args.report_output)
    print(f"summary: {paths.summary_json}")
    print(f"report: {args.report_output}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
