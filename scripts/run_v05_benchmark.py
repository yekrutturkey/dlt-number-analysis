"""Benchmark fast/standard/final pipeline profiles with one and four scoring workers."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_verified_history
from dlt_number_analysis.experiments import (
    benchmark_pipeline_profiles,
    write_runtime_benchmark_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draws",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "draws.csv",
    )
    parser.add_argument("--target-issue", default="26079")
    parser.add_argument("--random-seed", type=int, default=20260714)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "runtime_benchmark.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    draws = load_verified_history(args.draws)
    generated_at = datetime.now(UTC)
    numpy_records = benchmark_pipeline_profiles(
        draws,
        target_issue=args.target_issue,
        generated_at=generated_at,
        random_seed=args.random_seed,
    )
    scalar_reference = benchmark_pipeline_profiles(
        draws,
        target_issue=args.target_issue,
        generated_at=generated_at,
        profiles=("fast",),
        parallel_workers=(1,),
        candidate_scoring_methods=("scalar",),
        random_seed=args.random_seed,
    )
    records = (*numpy_records, *scalar_reference)
    write_runtime_benchmark_report(records, args.output)
    for record in records:
        print(
            f"{record.profile} workers={record.parallel_workers} "
            f"method={record.candidate_scoring_method}: "
            f"runtime={record.runtime_seconds:.3f}s, peak={record.peak_memory_mb:.1f}MB"
        )
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
