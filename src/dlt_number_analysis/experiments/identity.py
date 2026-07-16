"""Canonical two-layer execution identity for formal experiment runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis.data import canonical_history_sha256, validate_draw_dataframe
from dlt_number_analysis.evaluation import IssuePrizeRecord, PrizeTable
from dlt_number_analysis.experiments.specs import DataSplitSpec, ExperimentPhase, ExperimentSpec
from dlt_number_analysis.experiments.splits import experiment_config_sha256, split_history
from dlt_number_analysis.pipeline import PROFILE_DEFAULTS, PipelineProfile
from dlt_number_analysis.portfolio import PortfolioScoringMethod

EXECUTION_IDENTITY_SCHEMA_VERSION = "experiment-execution-identity-v2"
RUNNER_VERSION = "shared-feasible-bank-runner-v0.5.5"
SCORING_IMPLEMENTATION_VERSION = "number-and-structure-scoring-v0.5.5"
RAW_EVALUATION_SEMANTICS_VERSION = "raw-observation-v1"
FULL_RESAMPLING_SEMANTICS_VERSION = "full-resampling-v1"

EvaluationMode = Literal["raw_observation", "full_resampling"]


def canonical_identity_sha256(payload: object) -> str:
    """Hash JSON-compatible identity content independently of mapping field order."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class SplitBoundary(BaseModel):
    """Concrete immutable row and issue boundary for one temporal phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_index: int = Field(ge=0)
    end_index_exclusive: int = Field(gt=0)
    record_count: int = Field(gt=0)
    start_issue: str = Field(pattern=r"^\d+$")
    end_issue: str = Field(pattern=r"^\d+$")


class RunContextPayload(BaseModel):
    """Shared B1-B6 execution conditions, excluding task/chunk and host metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_identity_schema_version: str = EXECUTION_IDENTITY_SCHEMA_VERSION
    runner_version: str = RUNNER_VERSION
    history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_split_spec: DataSplitSpec
    split_boundaries: dict[ExperimentPhase, SplitBoundary]
    phase: ExperimentPhase
    evaluation_mode: EvaluationMode
    profile: PipelineProfile
    portfolio_scoring_method: PortfolioScoringMethod
    minimum_history: int = Field(ge=1)
    minimum_bank_size: int = Field(ge=1)
    maximum_bank_search_trials: int = Field(ge=1)
    candidate_count: int = Field(ge=1)
    optimizer_search_trials: int = Field(ge=1)
    maximum_bank_size: int = Field(ge=1)
    scoring_implementation_version: str = SCORING_IMPLEMENTATION_VERSION
    prize_rule_schedule_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    issue_prize_records_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evaluation_semantics_version: str


class RunContextIdentity(BaseModel):
    """Canonical shared context and its stable SHA-256 identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload: RunContextPayload
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExperimentExecutionIdentity(BaseModel):
    """Per-spec identity derived from one shared run context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _split_boundaries(
    draws: pd.DataFrame, split_spec: DataSplitSpec
) -> dict[ExperimentPhase, SplitBoundary]:
    split = split_history(draws, split_spec)
    development_count = len(split.development)
    calibration_count = len(split.calibration)
    starts = {
        "development": 0,
        "calibration": development_count,
        "final_holdout": development_count + calibration_count,
    }
    result: dict[ExperimentPhase, SplitBoundary] = {}
    for phase in ("development", "calibration", "final_holdout"):
        frame = split.phase_frame(phase)
        start = starts[phase]
        result[phase] = SplitBoundary(
            start_index=start,
            end_index_exclusive=start + len(frame),
            record_count=len(frame),
            start_issue=str(frame.iloc[0]["issue"]),
            end_issue=str(frame.iloc[-1]["issue"]),
        )
    return result


def prize_rule_schedule_sha256(prize_tables: Sequence[PrizeTable] | None) -> str | None:
    """Hash logical prize schedule content, or preserve an explicit null marker."""
    if not prize_tables:
        return None
    payload = [
        table.model_dump(mode="json")
        for table in sorted(prize_tables, key=lambda item: int(item.effective_from_issue))
    ]
    return canonical_identity_sha256(payload)


def issue_prize_records_sha256(
    records: Mapping[str, IssuePrizeRecord] | None,
) -> str | None:
    """Hash issue prize records in stable issue order, or return explicit null."""
    if not records:
        return None
    payload = {issue: records[issue].model_dump(mode="json") for issue in sorted(records, key=int)}
    return canonical_identity_sha256(payload)


def build_run_context_identity(
    draws: pd.DataFrame,
    *,
    data_split_spec: DataSplitSpec,
    phase: ExperimentPhase,
    evaluation_mode: EvaluationMode,
    profile: PipelineProfile,
    portfolio_scoring_method: PortfolioScoringMethod,
    minimum_history: int,
    minimum_bank_size: int,
    maximum_bank_search_trials: int,
    prize_tables: Sequence[PrizeTable] | None = None,
    issue_prize_records: Mapping[str, IssuePrizeRecord] | None = None,
    worker_count: int | None = None,
    target_issues: Sequence[str] | None = None,
) -> RunContextIdentity:
    """Build identity while intentionally excluding worker and target chunk metadata."""
    del worker_count, target_issues
    validated = validate_draw_dataframe(draws)
    defaults = PROFILE_DEFAULTS[profile]
    payload = RunContextPayload(
        history_sha256=canonical_history_sha256(validated),
        data_split_spec=data_split_spec,
        split_boundaries=_split_boundaries(validated, data_split_spec),
        phase=phase,
        evaluation_mode=evaluation_mode,
        profile=profile,
        portfolio_scoring_method=portfolio_scoring_method,
        minimum_history=minimum_history,
        minimum_bank_size=minimum_bank_size,
        maximum_bank_search_trials=maximum_bank_search_trials,
        candidate_count=defaults["candidate_count"],
        optimizer_search_trials=defaults["optimizer_search_trials"],
        maximum_bank_size=min(2_000, defaults["optimizer_search_trials"]),
        prize_rule_schedule_sha256=prize_rule_schedule_sha256(prize_tables),
        issue_prize_records_sha256=issue_prize_records_sha256(issue_prize_records),
        evaluation_semantics_version=(
            RAW_EVALUATION_SEMANTICS_VERSION
            if evaluation_mode == "raw_observation"
            else FULL_RESAMPLING_SEMANTICS_VERSION
        ),
    )
    return RunContextIdentity(
        payload=payload,
        run_context_sha256=canonical_identity_sha256(payload.model_dump(mode="json")),
    )


def build_experiment_execution_identity(
    run_context: RunContextIdentity,
    spec: ExperimentSpec,
) -> ExperimentExecutionIdentity:
    """Combine shared context identity with the canonical individual ExperimentSpec hash."""
    experiment_hash = experiment_config_sha256(spec)
    execution_hash = hashlib.sha256(
        f"{run_context.run_context_sha256}{experiment_hash}".encode("ascii")
    ).hexdigest()
    return ExperimentExecutionIdentity(
        experiment_config_sha256=experiment_hash,
        run_context_sha256=run_context.run_context_sha256,
        execution_config_sha256=execution_hash,
    )


def current_git_commit_sha() -> str:
    """Return the current commit for audit only; it is not part of resumable identity."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40
    value = result.stdout.strip().lower()
    return (
        value
        if len(value) == 40 and all(char in "0123456789abcdef" for char in value)
        else "0" * 40
    )
