"""Run the one bounded v0.5.5 identity smoke or its single read-only resume check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.benchmark_checkpoints import (
    load_checkpoint,
    run_isolated_stage,
)
from dlt_number_analysis.experiments.command import PROJECT_ROOT
from dlt_number_analysis.experiments.identity_smoke import (
    V055_SMOKE_TIMEOUT_SECONDS,
    validate_v055_identity_smoke_resume,
    write_v055_identity_smoke_report,
)


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
        default=PROJECT_ROOT / "outputs/benchmarks/v055_identity_smoke",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/v055_identity_smoke.md",
    )
    parser.add_argument("--resume-check-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = args.output_dir / "identity_smoke_checkpoint.json"
    resume_path = args.output_dir / "resume_validation.json"
    identity = {
        "benchmark_version": "v0.5.5-identity-smoke-v1",
        "evaluation_mode": "raw_observation",
        "profile": "fast",
        "portfolio_scoring_method": "numpy_vectorized",
        "process_count": 1,
    }
    if args.resume_check_only:
        resume = validate_v055_identity_smoke_resume(
            args.draws,
            args.prize_rules,
            args.output_dir,
        )
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        resume_path.write_text(
            json.dumps(resume, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        checkpoint = load_checkpoint(checkpoint_path, expected_identity=identity)
        write_v055_identity_smoke_report(checkpoint, resume, args.report_output)
        print("read-only resume validation: no generation scheduled")
    else:
        outcome = run_isolated_stage(
            stage="v055_identity_smoke",
            worker_path=(
                "dlt_number_analysis.experiments.identity_smoke:run_v055_identity_smoke_worker"
            ),
            worker_kwargs={
                "history_path": str(args.draws.resolve()),
                "prize_rule_path": str(args.prize_rules.resolve()),
                "output_dir": str(args.output_dir.resolve()),
            },
            checkpoint_path=checkpoint_path,
            checkpoint_identity=identity,
            timeout_seconds=V055_SMOKE_TIMEOUT_SECONDS,
        )
        checkpoint = load_checkpoint(checkpoint_path, expected_identity=identity)
        resume = (
            json.loads(resume_path.read_text(encoding="utf-8")) if resume_path.exists() else None
        )
        write_v055_identity_smoke_report(checkpoint, resume, args.report_output)
        print(
            f"identity smoke: {outcome.status}; elapsed={outcome.elapsed_seconds:.3f}s; "
            f"checkpoint={outcome.checkpoint_path}"
        )
        if outcome.status != "completed":
            print(DISCLAIMER)
            return 1
    print(f"report: {args.report_output}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
