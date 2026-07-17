"""Run the bounded B1-B6 100-period paired correctness smoke experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_verified_history
from dlt_number_analysis.experiments.command import PROJECT_ROOT
from dlt_number_analysis.experiments.command import main as run_experiments
from dlt_number_analysis.experiments.smoke import (
    SMOKE_EXPERIMENT_IDS,
    select_consecutive_development_targets,
    smoke_comparisons,
    validate_paired_smoke_observations,
    verify_smoke_storage,
    write_paired_smoke_report,
    write_smoke_comparisons_csv,
)
from dlt_number_analysis.experiments.specs import baseline_experiment_specs
from dlt_number_analysis.experiments.storage import ExperimentResultStore


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately bounded smoke command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, default=PROJECT_ROOT / "data/raw/draws.csv")
    parser.add_argument("--start-issue", default="08009")
    parser.add_argument("--period-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--minimum-bank-size", type=int, default=500)
    parser.add_argument("--maximum-bank-search-trials", type=int, default=80_000)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "outputs/experiments")
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/v051_paired_smoke_100.md",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/v051_paired_smoke_comparisons.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run once, rerun to prove skip behavior, and write audited smoke artifacts."""
    args = build_parser().parse_args(argv)
    if args.period_count != 100:
        raise ValueError("the v0.5.1 paired smoke command requires exactly 100 periods")
    draws = load_verified_history(args.draws)
    targets = select_consecutive_development_targets(
        draws,
        count=args.period_count,
        start_issue=args.start_issue,
    )
    command = [
        "--draws",
        str(args.draws),
        "--experiment-ids",
        *SMOKE_EXPERIMENT_IDS,
        "--phase",
        "development",
        "--profile",
        "fast",
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--chunk-size",
        str(args.chunk_size),
        "--minimum-bank-size",
        str(args.minimum_bank_size),
        "--maximum-bank-search-trials",
        str(args.maximum_bank_search_trials),
        "--results-root",
        str(args.results_root),
        "--inference-context",
        "smoke",
        "--target-issues",
        *targets,
    ]
    first_exit = run_experiments(command)
    if first_exit != 0:
        return first_exit
    second_exit = run_experiments(command)
    store = ExperimentResultStore(args.results_root)
    specifications = tuple(
        spec
        for spec in baseline_experiment_specs(seeds=(args.seed,), phase="development")
        if spec.experiment_id in SMOKE_EXPERIMENT_IDS
    )
    resume_verified, duplicate_rejected = verify_smoke_storage(
        store,
        specifications,
        seed=args.seed,
        target_issues=targets,
    )
    observations = store.load_all()
    comparisons = smoke_comparisons(
        observations,
        target_issues=targets,
        seed=args.seed,
    )
    audit = validate_paired_smoke_observations(
        observations,
        target_issues=targets,
        seed=args.seed,
        minimum_bank_size=args.minimum_bank_size,
        resume_status_verified=resume_verified,
        duplicate_primary_key_rejected=duplicate_rejected,
        completed_task_skip_verified=second_exit == 0,
    )
    write_paired_smoke_report(audit, comparisons, args.report_output)
    write_smoke_comparisons_csv(comparisons, args.comparison_output)
    print(audit.model_dump_json(indent=2))
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
