"""Run the one permitted v0.5.6 smoke, resume check, or exact report-only check."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.benchmark_checkpoints import run_isolated_stage
from dlt_number_analysis.experiments.cohort_smoke import (
    V056_REPORT_TIMEOUT_SECONDS,
    V056_RESUME_TIMEOUT_SECONDS,
    V056_SMOKE_TIMEOUT_SECONDS,
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
        default=PROJECT_ROOT / "outputs/benchmarks/v056_cohort_smoke",
    )
    parser.add_argument(
        "--preflight-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/run_plans",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/v056_cohort_smoke.md",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume-check-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--run-context-sha256")
    parser.add_argument("--cohort-definition-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_identity = {
        "benchmark_version": "v0.5.6-logical-cohort-smoke-v1",
        "evaluation_mode": "raw_observation",
        "profile": "fast",
        "portfolio_scoring_method": "numpy_vectorized",
        "process_count": 1,
    }
    smoke_checkpoint = args.output_dir / "cohort_smoke_checkpoint.json"
    resume_checkpoint = args.output_dir / "resume_validation.json"
    report_checkpoint = args.output_dir / "report_only_validation.json"
    if args.report_only:
        if args.run_context_sha256 is None or args.cohort_definition_sha256 is None:
            raise ValueError(
                "report-only requires --run-context-sha256 and --cohort-definition-sha256"
            )
        outcome = run_isolated_stage(
            stage="v056_report_only",
            worker_path=(
                "dlt_number_analysis.experiments.cohort_smoke:run_v056_report_only_worker"
            ),
            worker_kwargs={
                "store_root": str((args.output_dir / "schema_v3").resolve()),
                "run_context_sha256": args.run_context_sha256,
                "cohort_definition_sha256": args.cohort_definition_sha256,
                "smoke_checkpoint_path": str(smoke_checkpoint.resolve()),
                "resume_checkpoint_path": str(resume_checkpoint.resolve()),
                "report_path": str(args.report_output.resolve()),
            },
            checkpoint_path=report_checkpoint,
            checkpoint_identity={
                **common_identity,
                "validation_mode": "report_only",
                "run_context_sha256": args.run_context_sha256,
                "cohort_definition_sha256": args.cohort_definition_sha256,
            },
            timeout_seconds=V056_REPORT_TIMEOUT_SECONDS,
        )
    elif args.resume_check_only:
        if args.chunk_size != 2:
            raise ValueError("the permitted v0.5.6 resume validation requires chunk_size=2")
        outcome = run_isolated_stage(
            stage="v056_resume_validation",
            worker_path=(
                "dlt_number_analysis.experiments.cohort_smoke:run_v056_resume_validation_worker"
            ),
            worker_kwargs={
                "history_path": str(args.draws.resolve()),
                "prize_rule_path": str(args.prize_rules.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "chunk_size": 2,
            },
            checkpoint_path=resume_checkpoint,
            checkpoint_identity={
                **common_identity,
                "validation_mode": "resume_read_only",
                "chunk_size": 2,
            },
            timeout_seconds=V056_RESUME_TIMEOUT_SECONDS,
        )
    else:
        outcome = run_isolated_stage(
            stage="v056_cohort_smoke",
            worker_path=(
                "dlt_number_analysis.experiments.cohort_smoke:run_v056_cohort_smoke_worker"
            ),
            worker_kwargs={
                "project_root": str(PROJECT_ROOT.resolve()),
                "history_path": str(args.draws.resolve()),
                "prize_rule_path": str(args.prize_rules.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "preflight_root": str(args.preflight_root.resolve()),
            },
            checkpoint_path=smoke_checkpoint,
            checkpoint_identity=common_identity,
            timeout_seconds=V056_SMOKE_TIMEOUT_SECONDS,
        )
    print(
        f"{outcome.stage}: {outcome.status}; elapsed={outcome.elapsed_seconds:.3f}s; "
        f"checkpoint={outcome.checkpoint_path}"
    )
    print(DISCLAIMER)
    return 0 if outcome.status in {"completed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
