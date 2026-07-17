"""End-to-end prediction pipeline shared by live generation and rolling backtests."""

from dlt_number_analysis.data.identity import canonical_history_bytes, canonical_history_sha256
from dlt_number_analysis.pipeline.artifacts import (
    NextPredictionArtifact,
    load_next_prediction_artifact,
)
from dlt_number_analysis.pipeline.audit import (
    GitAudit,
    HistoryAudit,
    build_history_audit,
    collect_git_audit,
)
from dlt_number_analysis.pipeline.prediction import (
    PROFILE_DEFAULTS,
    CandidatePoolSummary,
    PipelineConfig,
    PipelineProfile,
    PipelineResult,
    PipelineSeeds,
    PredictionPipeline,
    optimized_portfolio_strategy,
)

__all__ = [
    "PROFILE_DEFAULTS",
    "CandidatePoolSummary",
    "GitAudit",
    "HistoryAudit",
    "NextPredictionArtifact",
    "PipelineConfig",
    "PipelineProfile",
    "PipelineResult",
    "PipelineSeeds",
    "PredictionPipeline",
    "build_history_audit",
    "canonical_history_bytes",
    "canonical_history_sha256",
    "collect_git_audit",
    "load_next_prediction_artifact",
    "optimized_portfolio_strategy",
]
