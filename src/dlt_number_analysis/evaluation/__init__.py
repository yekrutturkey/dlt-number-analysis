"""配置驱动的中奖评估与复盘报告。"""

from dlt_number_analysis.evaluation.prize import (
    MatchPattern,
    PrizeTable,
    PrizeTierRule,
    evaluate_prediction,
    evaluate_ticket,
    load_prize_table,
)
from dlt_number_analysis.evaluation.reporting import render_prediction_review, write_review_report

__all__ = [
    "MatchPattern",
    "PrizeTable",
    "PrizeTierRule",
    "evaluate_prediction",
    "evaluate_ticket",
    "load_prize_table",
    "render_prediction_review",
    "write_review_report",
]
