"""Persisted metadata envelope for a live next-issue pipeline run."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.models import PredictionRecord
from dlt_number_analysis.pipeline.prediction import (
    CandidatePoolSummary,
    PipelineConfig,
    PipelineSeeds,
)


class NextPredictionArtifact(BaseModel):
    """Complete live-generation audit record required by the v0.4.1 CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    generated_at: datetime
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    git_diff_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        validation_alias=AliasChoices("raw_file_sha256", "history_sha256"),
    )
    canonical_history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_record_count: int = Field(ge=1)
    history_start_issue: str = Field(pattern=r"^\d+$")
    history_cutoff_issue: str = Field(pattern=r"^\d+$")
    history_verified: bool = True
    unverified_history_override: bool = False
    short_history_override: bool = False
    pipeline_config: PipelineConfig
    random_seeds: PipelineSeeds
    candidate_pool_summary: CandidatePoolSummary
    prediction: PredictionRecord
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        if value != DISCLAIMER:
            raise ValueError("next-prediction artifact is missing the fixed disclaimer")
        return value

    @property
    def history_sha256(self) -> str:
        """Deprecated v0.4.1 accessor for the exact raw-file hash."""
        return self.raw_file_sha256

    @model_validator(mode="after")
    def validate_bound_prediction(self) -> Self:
        """Reject an envelope whose duplicated audit fields diverge from its prediction."""
        prediction = self.prediction
        if self.target_issue != prediction.target_issue:
            raise ValueError("artifact target_issue does not match prediction")
        if self.data_cutoff_issue != prediction.data_cutoff_issue:
            raise ValueError("artifact data_cutoff_issue does not match prediction")
        if self.generated_at != prediction.generated_at:
            raise ValueError("artifact generated_at does not match prediction")
        if self.history_cutoff_issue != self.data_cutoff_issue:
            raise ValueError("history_cutoff_issue must equal data_cutoff_issue")
        if int(self.history_start_issue) > int(self.history_cutoff_issue):
            raise ValueError("history_start_issue must not exceed history_cutoff_issue")

        parameters = prediction.parameters
        if parameters.get("pipeline_config") != self.pipeline_config.model_dump(mode="json"):
            raise ValueError("artifact pipeline_config does not match prediction parameters")
        if parameters.get("random_seeds") != self.random_seeds.model_dump(mode="json"):
            raise ValueError("artifact random_seeds do not match prediction parameters")
        if parameters.get("short_history_override") != self.short_history_override:
            raise ValueError("artifact short_history_override does not match prediction parameters")
        if parameters.get("history_verified") != self.history_verified:
            raise ValueError("artifact history_verified does not match prediction parameters")
        if parameters.get("unverified_history_override") != self.unverified_history_override:
            raise ValueError(
                "artifact unverified_history_override does not match prediction parameters"
            )
        if self.history_verified == self.unverified_history_override:
            raise ValueError("exactly one history verification state must be true")
        return self


def load_next_prediction_artifact(path: str | Path) -> NextPredictionArtifact:
    """Load and fully cross-validate a persisted next-prediction artifact."""
    source = Path(path)
    return NextPredictionArtifact.model_validate_json(source.read_text(encoding="utf-8"))
