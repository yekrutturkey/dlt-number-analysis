"""预测日志的 UTF-8 JSON 持久化入口。"""

from __future__ import annotations

from pathlib import Path

from dlt_number_analysis.models.records import PredictionRecord


def load_prediction_record(path: str | Path) -> PredictionRecord:
    """读取并完整校验预测日志。"""
    source = Path(path)
    return PredictionRecord.model_validate_json(source.read_text(encoding="utf-8"))


def write_prediction_record(record: PredictionRecord, path: str | Path) -> Path:
    """把预测日志连同风险声明、种子和参数写入 JSON。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return target
