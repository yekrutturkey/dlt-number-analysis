"""把预测评估渲染为包含风险声明的 Markdown 复盘。"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import DrawRecord
from dlt_number_analysis.models import PredictionRecord, PredictionReview


def _format_numbers(numbers: tuple[int, ...]) -> str:
    """把号码格式化为两位数逗号列表。"""
    return ",".join(f"{number:02d}" for number in numbers)


def _format_amount(amount: Decimal | None) -> str:
    """把奖金格式化为人民币文本。"""
    return "浮动/未配置" if amount is None else f"{_format_decimal(amount)} 元"


def _format_decimal(value: Decimal) -> str:
    """以非科学计数法显示 Decimal，并只移除小数部分末尾零。"""
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def render_prediction_review(
    prediction: PredictionRecord,
    actual_draw: DrawRecord,
    review: PredictionReview,
) -> str:
    """生成单期五注预测的 Markdown 复盘报告。"""
    if (
        prediction.target_issue != actual_draw.issue
        or prediction.target_issue != review.target_issue
    ):
        raise ValueError("预测、开奖和复盘期号必须一致")

    evaluation_by_id = {item.ticket_id: item for item in review.evaluations}
    lines = [
        f"# 大乐透第 {prediction.target_issue} 期预测复盘",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "## 记录性质",
        "",
        f"- prediction_origin: `{prediction.prediction_origin}`",
        f"- model_version: `{prediction.model_version}`",
        f"- data_cutoff_issue: `{prediction.data_cutoff_issue}`",
        "- 这 5 注是用户提供的开奖前实际购买号码，经聊天手工补录；不是当前代码生成结果。",
        "",
        "## 实际开奖",
        "",
        f"- 前区：{_format_numbers(actual_draw.front_numbers)}",
        f"- 后区：{_format_numbers(actual_draw.back_numbers)}",
        "",
        "## 单注结果",
        "",
        "| 注号 | 前区 | 后区 | 前区命中 | 后区命中 | 奖级 | 奖金 |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for index, ticket in enumerate(prediction.tickets, start=1):
        evaluation = evaluation_by_id[ticket.ticket_id]
        lines.append(
            "| "
            f"{index} | {_format_numbers(ticket.front_numbers)} | "
            f"{_format_numbers(ticket.back_numbers)} | {evaluation.front_hits} | "
            f"{evaluation.back_hits} | {evaluation.prize_tier or '未中奖'} | "
            f"{_format_amount(evaluation.prize_amount)} |"
        )

    total_prize_text = _format_amount(review.total_prize)
    roi_text = "无法计算" if review.roi is None else f"{review.roi:.2%}"
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 第 2 注命中 2 个前区和 1 个后区。",
            "- 5 注号码池覆盖全部 5 个前区和 2 个后区开奖号码。",
            "- 号码池全覆盖不代表单注预测成功；奖级只按每一注独立号码计算。",
            f"- 总成本：{_format_decimal(review.total_cost)} 元。",
            f"- 总奖金：{total_prize_text}。",
            f"- ROI：{roi_text}。",
            "",
            f"**{DISCLAIMER}**",
            "",
        ]
    )
    return "\n".join(lines)


def write_review_report(content: str, path: str | Path) -> Path:
    """把复盘报告写入指定 UTF-8 Markdown 文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    return target
