"""Run the one bounded v0.5.4 production Runner benchmark or validate resume state."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.batch_runner_benchmark import (
    V054BenchmarkPaths,
    run_v054_batch_runner_benchmark,
    validate_v054_resume,
    write_v054_report,
)
from dlt_number_analysis.experiments.command import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, default=PROJECT_ROOT / "data/raw/draws.csv")
    parser.add_argument(
        "--prize-rules",
        type=Path,
        default=PROJECT_ROOT / "config/prize_tiers.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmarks/v054",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/v054_batch_runner_benchmark.md",
    )
    parser.add_argument("--resume-check-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = V054BenchmarkPaths(args.output_dir)
    if args.resume_check_only:
        summary = validate_v054_resume(args.draws, args.output_dir)
        print(f"read-only resume validation: {summary['measured_target_count']} targets")
    else:
        outcome = run_v054_batch_runner_benchmark(
            args.draws,
            args.prize_rules,
            args.output_dir,
        )
        print(
            f"benchmark: {outcome.status}; elapsed={outcome.elapsed_seconds:.3f}s; "
            f"checkpoint={outcome.checkpoint_path}"
        )
    write_v054_report(args.draws, args.output_dir, args.report_output)
    print(f"summary: {paths.summary}")
    print(f"report: {args.report_output}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
