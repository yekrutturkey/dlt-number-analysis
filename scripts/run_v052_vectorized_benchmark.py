"""Run one bounded worker-count slice of the v0.5.2 three-target benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.command import PROJECT_ROOT
from dlt_number_analysis.experiments.vectorized_benchmark import (
    load_vectorized_benchmark_json,
    run_vectorized_benchmark,
    write_vectorized_benchmark_json,
    write_vectorized_benchmark_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build a command that can run only one bounded worker count at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, default=PROJECT_ROOT / "data/raw/draws.csv")
    parser.add_argument("--workers", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seed", type=int, default=20260000)
    parser.add_argument(
        "--target-issues",
        nargs="+",
        default=("08009", "08010", "08011"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/v052_vectorized_benchmark.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one slice and aggregate every completed one/two-worker JSON result."""
    args = build_parser().parse_args(argv)
    targets = tuple(str(issue) for issue in args.target_issues)
    if len(targets) > 3:
        raise ValueError("the v0.5.2 benchmark is limited to at most three targets")
    result = run_vectorized_benchmark(
        args.draws,
        target_issues=targets,
        seed=args.seed,
        workers=args.workers,
        checkpoint_dir=args.result_dir / "v052_vectorized_checkpoints",
    )
    result_path = args.result_dir / f"v052_vectorized_benchmark_workers_{args.workers}.json"
    write_vectorized_benchmark_json(result, result_path)
    completed = tuple(
        load_vectorized_benchmark_json(path)
        for path in sorted(args.result_dir.glob("v052_vectorized_benchmark_workers_*.json"))
    )
    write_vectorized_benchmark_report(completed, args.report_output)
    print(result.model_dump_json(indent=2))
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
