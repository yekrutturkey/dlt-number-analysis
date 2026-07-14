"""历史开奖数据结构、加载和校验。"""

from dlt_number_analysis.data.data_validator import (
    BACK_COLUMNS,
    CSV_COLUMNS,
    FRONT_COLUMNS,
    DrawDataValidationError,
    DrawRecord,
    load_draws_csv,
    validate_draw_dataframe,
    write_validated_draws_csv,
)
from dlt_number_analysis.data.history_store import (
    DataQualityIssue,
    DataQualityReport,
    DrawConflictError,
    SourceReconciliationResult,
    append_draws,
    assert_backtest_ready,
    generate_data_quality_report,
    load_issue_prizes,
    load_prize_rule_schedule,
    reconcile_sources,
)

__all__ = [
    "BACK_COLUMNS",
    "CSV_COLUMNS",
    "FRONT_COLUMNS",
    "DataQualityIssue",
    "DataQualityReport",
    "DrawConflictError",
    "DrawDataValidationError",
    "DrawRecord",
    "SourceReconciliationResult",
    "append_draws",
    "assert_backtest_ready",
    "generate_data_quality_report",
    "load_draws_csv",
    "load_issue_prizes",
    "load_prize_rule_schedule",
    "reconcile_sources",
    "validate_draw_dataframe",
    "write_validated_draws_csv",
]
