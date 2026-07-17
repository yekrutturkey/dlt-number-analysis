"""Bounded three-target validation for v0.5.5 execution identity and storage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from time import perf_counter

import pandas as pd

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_prize_rule_schedule, load_verified_history
from dlt_number_analysis.experiments.identity import (
    ExperimentExecutionIdentity,
    RunContextIdentity,
    build_experiment_execution_identity,
    build_run_context_identity,
)
from dlt_number_analysis.experiments.runner import run_experiment_batch
from dlt_number_analysis.experiments.specs import ExperimentSpec, baseline_experiment_specs
from dlt_number_analysis.experiments.splits import split_history
from dlt_number_analysis.experiments.storage import (
    DuplicateExperimentResultError,
    ExperimentResultStoreV2,
    FormalPartitionManifest,
)

V055_SMOKE_SEED = 20260000
V055_SMOKE_TIMEOUT_SECONDS = 120.0
V055_SMOKE_EXPERIMENT_IDS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "B6")


def select_v055_identity_smoke_targets(
    draws: pd.DataFrame,
    *,
    minimum_history: int = 100,
) -> tuple[str, str, str]:
    """Return the first eligible, median, and last development targets dynamically."""
    spec = baseline_experiment_specs(seeds=(V055_SMOKE_SEED,), phase="development")[1]
    development = split_history(draws, spec.data_split).development
    development_issues = set(development["issue"].astype(str))
    eligible = tuple(
        issue
        for index, issue in enumerate(draws["issue"].astype(str))
        if index >= minimum_history and issue in development_issues
    )
    if len(eligible) < 3:
        raise ValueError("development split needs at least three eligible smoke targets")
    selected = (eligible[0], eligible[len(eligible) // 2], eligible[-1])
    if len(set(selected)) != 3:
        raise ValueError("dynamic smoke target selection did not produce three unique issues")
    return selected


def _smoke_specs() -> tuple[ExperimentSpec, ...]:
    baselines = baseline_experiment_specs(
        seeds=(V055_SMOKE_SEED,),
        phase="development",
    )
    by_id = {spec.experiment_id: spec for spec in baselines}
    return tuple(by_id[experiment_id] for experiment_id in V055_SMOKE_EXPERIMENT_IDS)


def _smoke_identities(
    draws: pd.DataFrame,
    specs: tuple[ExperimentSpec, ...],
    *,
    prize_rule_path: str | Path,
    minimum_history: int,
    minimum_bank_size: int,
    maximum_bank_search_trials: int,
    targets: tuple[str, ...],
) -> tuple[RunContextIdentity, dict[str, ExperimentExecutionIdentity]]:
    prize_tables = load_prize_rule_schedule(prize_rule_path).tables
    context = build_run_context_identity(
        draws,
        data_split_spec=specs[0].data_split,
        phase="development",
        evaluation_mode="raw_observation",
        profile="fast",
        portfolio_scoring_method="numpy_vectorized",
        minimum_history=minimum_history,
        minimum_bank_size=minimum_bank_size,
        maximum_bank_search_trials=maximum_bank_search_trials,
        prize_tables=prize_tables,
        worker_count=1,
        target_issues=targets,
    )
    return context, {
        spec.experiment_id: build_experiment_execution_identity(context, spec) for spec in specs
    }


def _validate_observation_identity(
    observations: pd.DataFrame,
    *,
    targets: tuple[str, ...],
) -> None:
    if len(observations) != len(targets) * len(V055_SMOKE_EXPERIMENT_IDS):
        raise ValueError("v0.5.5 identity smoke must produce exactly 18 observations")
    if observations["run_context_sha256"].nunique() != 1:
        raise ValueError("B1-B6 must share one run context")
    execution_counts = observations.groupby("experiment_id")["execution_config_sha256"].nunique()
    if set(execution_counts.index) != set(V055_SMOKE_EXPERIMENT_IDS) or set(
        execution_counts.astype(int)
    ) != {1}:
        raise ValueError("each B1-B6 experiment must retain one execution identity")
    if observations.drop_duplicates("experiment_id")["execution_config_sha256"].nunique() != 6:
        raise ValueError("B1-B6 must have distinct per-experiment execution identities")
    if set(observations["requested_portfolio_scoring_method"]) != {"numpy_vectorized"}:
        raise ValueError("smoke requested scoring method is not uniformly NumPy vectorized")
    actual = observations.groupby("experiment_id")["portfolio_scoring_method"].first().to_dict()
    if actual.get("B1") != "random_bank_sample":
        raise ValueError("B1 must record random bank sampling as its actual method")
    if {actual.get(experiment_id) for experiment_id in V055_SMOKE_EXPERIMENT_IDS[1:]} != {
        "numpy_vectorized"
    }:
        raise ValueError("B2-B6 must record NumPy vectorized scoring")
    if observations["history_sha256"].nunique() != 1:
        raise ValueError("smoke observations do not share one canonical history identity")
    if set(observations["target_issue"].astype(str)) != set(targets):
        raise ValueError("smoke observations differ from the three requested targets")


def _manifest_tamper_probe(
    store: ExperimentResultStoreV2,
    spec: ExperimentSpec,
    identity: ExperimentExecutionIdentity,
) -> bool:
    frame = store.read_partition(spec, identity, seed=V055_SMOKE_SEED)
    manifest = FormalPartitionManifest.model_validate_json(
        store.manifest_path(spec, identity, seed=V055_SMOKE_SEED).read_text(encoding="utf-8")
    )
    tampered = manifest.model_copy(update={"history_sha256": "0" * 64})
    try:
        store.validate_manifest_payload(
            spec,
            identity,
            seed=V055_SMOKE_SEED,
            frame=frame,
            manifest=tampered,
        )
    except ValueError:
        return True
    return False


def run_v055_identity_smoke_worker(
    *,
    history_path: str,
    prize_rule_path: str,
    output_dir: str,
    minimum_history: int = 100,
    minimum_bank_size: int = 500,
    maximum_bank_search_trials: int = 80_000,
) -> Mapping[str, object]:
    """Execute exactly three development targets and commit schema-v2 partitions."""
    draws = load_verified_history(history_path)
    prize_tables = load_prize_rule_schedule(prize_rule_path).tables
    targets = select_v055_identity_smoke_targets(draws, minimum_history=minimum_history)
    specs = _smoke_specs()
    run_context, identities = _smoke_identities(
        draws,
        specs,
        prize_rule_path=prize_rule_path,
        minimum_history=minimum_history,
        minimum_bank_size=minimum_bank_size,
        maximum_bank_search_trials=maximum_bank_search_trials,
        targets=targets,
    )
    store = ExperimentResultStoreV2(Path(output_dir) / "schema_v2")
    pre_run_pending = all(
        store.status(
            spec,
            seed=V055_SMOKE_SEED,
            expected_target_issues=targets,
            identity=identities[spec.experiment_id],
            run_context=run_context,
        ).pending_target_issues
        == targets
        for spec in specs
    )
    started = perf_counter()
    result = run_experiment_batch(
        draws,
        specs,
        profile="fast",
        minimum_history=minimum_history,
        prize_tables=prize_tables,
        target_issues=targets,
        minimum_bank_size=minimum_bank_size,
        maximum_bank_search_trials=maximum_bank_search_trials,
        portfolio_scoring_method="numpy_vectorized",
        evaluation_mode="raw_observation",
        parallel_workers=1,
    )
    run_seconds = perf_counter() - started
    observations = result.observations
    _validate_observation_identity(observations, targets=targets)
    store.append(
        observations,
        expected_target_issues_by_experiment={spec.experiment_id: targets for spec in specs},
    )
    resume_complete = all(
        store.status(
            spec,
            seed=V055_SMOKE_SEED,
            expected_target_issues=targets,
            identity=identities[spec.experiment_id],
            run_context=run_context,
        ).is_complete
        for spec in specs
    )
    duplicate_rejected = False
    try:
        store.append(
            observations.iloc[[0]],
            expected_target_issues_by_experiment={spec.experiment_id: targets for spec in specs},
        )
    except DuplicateExperimentResultError:
        duplicate_rejected = True
    index_by_issue = {
        issue: index for index, issue in enumerate(draws["issue"].astype(str).tolist())
    }
    timings = [timing.model_dump(mode="json") for timing in result.target_timings]
    target_seconds = [float(timing.target_total_seconds) for timing in result.target_timings]
    if len(target_seconds) != 3:
        raise ValueError("v0.5.5 identity smoke requires three complete target timings")
    return {
        "smoke_version": "v0.5.5-identity-smoke-v1",
        "targets": list(targets),
        "history_lengths": {issue: index_by_issue[issue] for issue in targets},
        "observation_count": len(observations),
        "run_seconds": run_seconds,
        "run_context_sha256": str(observations.iloc[0]["run_context_sha256"]),
        "history_sha256": str(observations.iloc[0]["history_sha256"]),
        "execution_config_sha256_by_experiment": {
            experiment_id: str(group.iloc[0]["execution_config_sha256"])
            for experiment_id, group in observations.groupby("experiment_id", sort=True)
        },
        "bank_size": {
            "minimum": int(observations["bank_size"].min()),
            "median": float(observations["bank_size"].median()),
            "maximum": int(observations["bank_size"].max()),
        },
        "acceptance_rate": {
            "minimum": float(observations["bank_acceptance_rate"].min()),
            "median": float(observations["bank_acceptance_rate"].median()),
            "maximum": float(observations["bank_acceptance_rate"].max()),
        },
        "old_results_did_not_skip": pre_run_pending,
        "resume_complete": resume_complete,
        "duplicate_primary_key_rejected": duplicate_rejected,
        "manifest_tamper_rejected": _manifest_tamper_probe(
            store,
            specs[0],
            identities[specs[0].experiment_id],
        ),
        "target_timings": timings,
        "target_seconds_range": {
            "minimum": min(target_seconds),
            "median": median(target_seconds),
            "maximum": max(target_seconds),
        },
        "risk_disclaimer": DISCLAIMER,
    }


def validate_v055_identity_smoke_resume(
    history_path: str | Path,
    prize_rule_path: str | Path,
    output_dir: str | Path,
    *,
    minimum_history: int = 100,
    minimum_bank_size: int = 500,
    maximum_bank_search_trials: int = 80_000,
) -> dict[str, object]:
    """Read only committed partitions and prove a rerun would schedule no generation."""
    draws = load_verified_history(history_path)
    targets = select_v055_identity_smoke_targets(draws, minimum_history=minimum_history)
    specs = _smoke_specs()
    run_context, identities = _smoke_identities(
        draws,
        specs,
        prize_rule_path=prize_rule_path,
        minimum_history=minimum_history,
        minimum_bank_size=minimum_bank_size,
        maximum_bank_search_trials=maximum_bank_search_trials,
        targets=targets,
    )
    store = ExperimentResultStoreV2(Path(output_dir) / "schema_v2")
    statuses = [
        store.status(
            spec,
            seed=V055_SMOKE_SEED,
            expected_target_issues=targets,
            identity=identities[spec.experiment_id],
            run_context=run_context,
        )
        for spec in specs
    ]
    if not all(status.is_complete and not status.pending_target_issues for status in statuses):
        raise ValueError("v0.5.5 read-only resume validation found pending generation work")
    observations = store.load_all(phase="development")
    _validate_observation_identity(observations, targets=targets)
    return {
        "resume_validation": "read_only_no_generation",
        "targets": list(targets),
        "completed_partition_count": len(statuses),
        "observation_count": len(observations),
        "generation_called": False,
        "risk_disclaimer": DISCLAIMER,
    }


def write_v055_identity_smoke_report(
    checkpoint: Mapping[str, object],
    resume: Mapping[str, object] | None,
    path: str | Path,
) -> Path:
    """Write identity and timing evidence without making a strategy claim."""
    lines = [
        "# v0.5.5 execution identity smoke",
        "",
        f"- status: {checkpoint.get('status', 'unknown')}",
        f"- targets: {', '.join(str(value) for value in checkpoint.get('targets', []))}",
        f"- observation_count: {checkpoint.get('observation_count', 0)}",
        f"- run_seconds: {checkpoint.get('run_seconds', 'unavailable')}",
        f"- target_seconds_range: {checkpoint.get('target_seconds_range', 'unavailable')}",
        f"- run_context_sha256: `{checkpoint.get('run_context_sha256', 'unavailable')}`",
        f"- history_sha256: `{checkpoint.get('history_sha256', 'unavailable')}`",
        f"- old_results_did_not_skip: {checkpoint.get('old_results_did_not_skip', False)}",
        "- duplicate_primary_key_rejected: "
        f"{checkpoint.get('duplicate_primary_key_rejected', False)}",
        f"- manifest_tamper_rejected: {checkpoint.get('manifest_tamper_rejected', False)}",
        f"- resume_validation: {json.dumps(dict(resume or {}), ensure_ascii=False)}",
        "",
        "This bounded smoke validates execution identity and resumability only. It is not used "
        "for parameter selection and does not establish a strategy advantage.",
        "",
        f"> {DISCLAIMER}",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target
