"""从已保存的 26078 开奖与手工预测日志重建复盘报告。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from dlt_number_analysis.data import CSV_COLUMNS, DrawRecord, load_draws_csv
from dlt_number_analysis.evaluation import (
    evaluate_prediction,
    load_prize_table,
    render_prediction_review,
    write_review_report,
)
from dlt_number_analysis.models import DrawSourceRecord, PredictionRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_manual_prediction(path: Path) -> PredictionRecord:
    """读取并校验手工预测 JSON。"""
    return PredictionRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _load_source_record(path: Path) -> DrawSourceRecord:
    """读取并校验单行来源 JSONL。"""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise ValueError("26078 来源日志必须恰好包含一行")
    return DrawSourceRecord.model_validate_json(lines[0])


def generate_review() -> Path:
    """验证所有输入并生成 26078 Markdown 复盘。"""
    draws = load_draws_csv(PROJECT_ROOT / "data" / "raw" / "draws.csv")
    row = draws.loc[draws["issue"] == "26078"]
    if len(row) != 1:
        raise ValueError("历史开奖 CSV 必须恰好包含一条 26078 记录")
    actual_draw = DrawRecord.model_validate({column: row.iloc[0][column] for column in CSV_COLUMNS})
    prediction = _load_manual_prediction(
        PROJECT_ROOT / "outputs" / "predictions" / "26078_manual_chat.json"
    )
    source_record = _load_source_record(PROJECT_ROOT / "data" / "raw" / "draw_sources.jsonl")
    if source_record.issue != actual_draw.issue:
        raise ValueError("来源日志期号与开奖数据不一致")
    if (
        source_record.content_hash
        != sha256((PROJECT_ROOT / "data" / "raw" / "draws.csv").read_bytes()).hexdigest()
    ):
        raise ValueError("开奖 CSV 内容哈希与来源日志不一致")

    prize_table = load_prize_table(PROJECT_ROOT / "config" / "prize_tiers.json")
    review = evaluate_prediction(
        prediction,
        actual_draw,
        prize_table,
        prize_context="pool_below_800m",
    )
    report = render_prediction_review(prediction, actual_draw, review)
    report += (
        "\n## 数据与规则来源\n\n"
        "- 开奖号码与事前购买号码：用户在当前聊天中提供，来源标记为 `manual_chat`。\n"
        "- 原始生成时间未提供，日志中保留为 `null`，没有推测或补造。\n"
        "- 奖级规则：https://m.lottery.gov.cn/ksjz/m/yxgz_dlt/\n"
        "- 26077 期奖池滚存依据：https://www.gdlottery.cn/f_html/kjgg/P085_26077.html\n"
        "- 数据抓取模块尚未实现，当前记录仍需未来通过独立抓取源交叉核验。\n"
    )
    return write_review_report(
        report,
        PROJECT_ROOT / "outputs" / "reports" / "26078_review.md",
    )


if __name__ == "__main__":
    print(generate_review())
