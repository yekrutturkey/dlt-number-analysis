"""Logical experiment cohort identity independent of task chunking and execution order."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlt_number_analysis.experiments.identity import canonical_identity_sha256
from dlt_number_analysis.experiments.specs import ExperimentPhase

COHORT_IDENTITY_SCHEMA_VERSION = "logical-cohort-identity-v1"

CohortPurpose = Literal[
    "development_full",
    "development_smoke",
    "ablation_screening",
    "ablation_full_development",
    "calibration",
    "final_holdout",
]


class CohortDefinitionPayload(BaseModel):
    """Canonical complete target population shared by every scheduler chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort_identity_schema_version: str = COHORT_IDENTITY_SCHEMA_VERSION
    cohort_purpose: CohortPurpose
    phase: ExperimentPhase
    cohort_id: str = Field(min_length=1)
    ordered_target_issues: tuple[str, ...] = Field(min_length=1)
    target_count: int = Field(ge=1)
    start_issue: str = Field(pattern=r"^\d+$")
    end_issue: str = Field(pattern=r"^\d+$")
    expected_targets_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_derived_fields(self) -> CohortDefinitionPayload:
        issues = self.ordered_target_issues
        if len(set(issues)) != len(issues):
            raise ValueError("logical cohort target issues must be unique")
        if tuple(sorted(issues, key=int)) != issues:
            raise ValueError("logical cohort target issues must be in canonical integer order")
        if self.target_count != len(issues):
            raise ValueError("logical cohort target count differs from target issues")
        if self.start_issue != issues[0] or self.end_issue != issues[-1]:
            raise ValueError("logical cohort start/end issue differs from target issues")
        expected_hash = canonical_identity_sha256(list(issues))
        if self.expected_targets_sha256 != expected_hash:
            raise ValueError("logical cohort expected target hash is invalid")
        return self


class CohortDefinitionIdentity(BaseModel):
    """Complete logical cohort payload and its stable SHA-256 identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload: CohortDefinitionPayload
    cohort_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity_hash(self) -> CohortDefinitionIdentity:
        expected = canonical_identity_sha256(self.payload.model_dump(mode="json"))
        if self.cohort_definition_sha256 != expected:
            raise ValueError("logical cohort definition hash is invalid")
        return self


def normalize_cohort_targets(
    target_issues: Sequence[str | int],
    *,
    phase_target_issues: Sequence[str | int],
) -> tuple[str, ...]:
    """Normalize against canonical phase issue strings and reject duplicates/out-of-phase rows."""
    raw = tuple(str(issue) for issue in target_issues)
    if not raw:
        raise ValueError("logical cohort target issues cannot be empty")
    raw_integers = tuple(int(issue) for issue in raw)
    if len(set(raw_integers)) != len(raw_integers):
        raise ValueError("logical cohort target issues contain duplicates")
    allowed_by_integer: dict[int, str] = {}
    for raw_allowed in phase_target_issues:
        canonical = str(raw_allowed)
        integer = int(canonical)
        if integer in allowed_by_integer and allowed_by_integer[integer] != canonical:
            raise ValueError("phase contains ambiguous issue string representations")
        allowed_by_integer[integer] = canonical
    missing = sorted(set(raw_integers).difference(allowed_by_integer))
    if missing:
        raise ValueError(f"logical cohort targets are outside the declared phase: {missing}")
    return tuple(allowed_by_integer[issue] for issue in sorted(raw_integers))


def build_cohort_definition_identity(
    target_issues: Sequence[str | int],
    *,
    cohort_purpose: CohortPurpose,
    phase: ExperimentPhase,
    cohort_id: str,
    phase_target_issues: Sequence[str | int],
    chunk_size: int | None = None,
    worker_count: int | None = None,
    task_order: Sequence[int] | None = None,
    pending_target_issues: Sequence[str | int] | None = None,
) -> CohortDefinitionIdentity:
    """Build a cohort identity while explicitly excluding scheduler and resume state."""
    del chunk_size, worker_count, task_order, pending_target_issues
    ordered = normalize_cohort_targets(
        target_issues,
        phase_target_issues=phase_target_issues,
    )
    payload = CohortDefinitionPayload(
        cohort_purpose=cohort_purpose,
        phase=phase,
        cohort_id=cohort_id,
        ordered_target_issues=ordered,
        target_count=len(ordered),
        start_issue=ordered[0],
        end_issue=ordered[-1],
        expected_targets_sha256=canonical_identity_sha256(list(ordered)),
    )
    return CohortDefinitionIdentity(
        payload=payload,
        cohort_definition_sha256=canonical_identity_sha256(payload.model_dump(mode="json")),
    )


def task_targets_sha256(target_issues: Sequence[str | int]) -> str:
    """Hash one ordered task chunk for scheduling audit only."""
    targets = tuple(str(issue) for issue in target_issues)
    if not targets:
        raise ValueError("task target issues cannot be empty")
    if len(set(map(int, targets))) != len(targets):
        raise ValueError("task target issues contain duplicates")
    return canonical_identity_sha256(list(targets))
