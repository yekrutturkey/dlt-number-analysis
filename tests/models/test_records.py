"""预测与来源记录模型测试。"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.models import (
    DrawSourceRecord,
    PredictionRecord,
    TicketRecord,
    load_prediction_record,
    write_prediction_record,
)


def make_tickets() -> tuple[TicketRecord, ...]:
    """构造五注不同的合法测试票据。"""
    return tuple(
        TicketRecord(
            target_issue="26002",
            ticket_id=f"test-{index}",
            front_numbers=(index, index + 5, index + 10, index + 15, index + 20),
            back_numbers=(index, index + 5),
        )
        for index in range(1, 6)
    )


def test_draw_source_record_requires_aware_time_and_sha256() -> None:
    record = DrawSourceRecord(
        issue="26078",
        source_name="manual_chat",
        source_url="chat://current-thread/26078",
        fetched_at=datetime(2026, 7, 14, tzinfo=UTC),
        content_hash="a" * 64,
    )

    assert record.issue == "26078"
    with pytest.raises(ValidationError):
        DrawSourceRecord(
            issue="26078",
            source_name="manual_chat",
            source_url="missing-scheme",
            fetched_at=datetime(2026, 7, 14),
            content_hash="short",
        )


def test_generated_prediction_requires_seed_time_and_past_cutoff() -> None:
    record = PredictionRecord(
        target_issue="26002",
        tickets=make_tickets(),
        strategy_name="max_coverage",
        model_version="simple-v0.2",
        data_cutoff_issue="26001",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        random_seed=42,
        parameters={"ticket_count": 5},
        prediction_origin="generated",
    )

    assert record.risk_disclaimer == DISCLAIMER
    with pytest.raises(ValidationError, match="data_cutoff_issue"):
        record.model_copy(update={"data_cutoff_issue": "26002"}).model_validate(
            {**record.model_dump(), "data_cutoff_issue": "26002"}
        )


def test_prediction_rejects_identical_tickets() -> None:
    tickets = list(make_tickets())
    tickets[-1] = tickets[0].model_copy(update={"ticket_id": "different-id"})

    with pytest.raises(ValidationError, match="任意两注号码不得完全相同"):
        PredictionRecord(
            target_issue="26002",
            tickets=tickets,
            strategy_name="test",
            model_version="test-v1",
            data_cutoff_issue="26001",
            generated_at=datetime(2026, 1, 1, tzinfo=UTC),
            random_seed=1,
            parameters={},
            prediction_origin="generated",
        )


def test_prediction_json_round_trip_preserves_seed_and_parameters(tmp_path: Path) -> None:
    record = PredictionRecord(
        target_issue="26002",
        tickets=make_tickets(),
        strategy_name="max_coverage",
        model_version="simple-v0.2",
        data_cutoff_issue="26001",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        random_seed=42,
        parameters={"nested": {"ticket_count": 5}},
        prediction_origin="generated",
    )
    target = tmp_path / "predictions" / "26002.json"

    write_prediction_record(record, target)
    loaded = load_prediction_record(target)

    assert loaded == record
