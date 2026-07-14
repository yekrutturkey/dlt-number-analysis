"""Persisted metadata envelope for a live next-issue pipeline run."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.models import PredictionRecord
from dlt_number_analysis.pipeline.prediction import (
    CandidatePoolSummary,
    PipelineConfig,
    PipelineSeeds,
)


class NextPredictionArtifact(BaseModel):
    """Complete live-generation audit record required by the v0.4 CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    generated_at: datetime
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
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
