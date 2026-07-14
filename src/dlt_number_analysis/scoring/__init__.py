"""可替换的号码评分与动态结构评分接口。"""

from dlt_number_analysis.scoring.number_scorers import (
    RECENCY_WINDOWS,
    NumberArea,
    NumberScorer,
    cumulative_frequency_score,
    hot_cold_blend_score,
    recency_weighted_frequency_score,
    uniform_score,
)
from dlt_number_analysis.scoring.structure import (
    HistoricalStructureProfile,
    StructureScore,
    TicketStructureFeatures,
    compute_ticket_structure,
    fit_structure_profile,
    score_ticket_structure,
)

__all__ = [
    "RECENCY_WINDOWS",
    "HistoricalStructureProfile",
    "NumberArea",
    "NumberScorer",
    "StructureScore",
    "TicketStructureFeatures",
    "compute_ticket_structure",
    "cumulative_frequency_score",
    "fit_structure_profile",
    "hot_cold_blend_score",
    "recency_weighted_frequency_score",
    "score_ticket_structure",
    "uniform_score",
]
