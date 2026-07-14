"""第 26078 期真实开奖、手工预测和复盘制品测试。"""

from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_verified_history
from dlt_number_analysis.models import DrawSourceRecord, PredictionRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_issue_26078_artifacts_are_valid_and_auditable() -> None:
    draws_path = PROJECT_ROOT / "data" / "raw" / "draws.csv"
    draws = load_verified_history(draws_path)
    assert len(draws) == 2896
    draw = draws.loc[draws["issue"] == "26078"].iloc[0]
    assert draw[["front_1", "front_2", "front_3", "front_4", "front_5"]].tolist() == [
        2,
        13,
        20,
        25,
        32,
    ]
    assert draw[["back_1", "back_2"]].tolist() == [8, 11]

    source_line = (PROJECT_ROOT / "data" / "raw" / "draw_sources.jsonl").read_text(encoding="utf-8")
    source = DrawSourceRecord.model_validate_json(source_line)
    assert len(source.content_hash) == 64

    prediction = PredictionRecord.model_validate_json(
        (PROJECT_ROOT / "outputs" / "predictions" / "26078_manual_chat.json").read_text(
            encoding="utf-8"
        )
    )
    assert prediction.prediction_origin == "manual_chat"
    assert prediction.model_version == "manual-v0"
    assert prediction.data_cutoff_issue == "26077"
    assert prediction.generated_at is None
    assert prediction.parameters["prize_context"] == "pool_at_or_above_800m"
    assert prediction.risk_disclaimer == DISCLAIMER


def test_issue_26078_review_contains_required_conclusions() -> None:
    report = (PROJECT_ROOT / "outputs" / "reports" / "26078_review.md").read_text(encoding="utf-8")

    assert "第 2 注命中 2 个前区和 1 个后区" in report
    assert "5 注号码池覆盖全部 5 个前区和 2 个后区" in report
    assert "号码池全覆盖不代表单注预测成功" in report
    assert "不是当前代码生成结果" in report
    assert "七等奖 | 7 元" in report
    assert "总奖金：7 元" in report
    assert "ROI：-30.00%" in report
    assert DISCLAIMER in report
