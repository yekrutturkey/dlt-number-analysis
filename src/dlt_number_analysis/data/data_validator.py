"""大乐透历史开奖 CSV 的结构与业务规则校验。"""

from __future__ import annotations

import re
from datetime import date, datetime, time
from itertools import pairwise
from numbers import Integral
from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

FRONT_COLUMNS: tuple[str, ...] = tuple(f"front_{index}" for index in range(1, 6))
BACK_COLUMNS: tuple[str, ...] = tuple(f"back_{index}" for index in range(1, 3))
NUMBER_COLUMNS: tuple[str, ...] = FRONT_COLUMNS + BACK_COLUMNS
CSV_COLUMNS: tuple[str, ...] = ("issue", "draw_date", *NUMBER_COLUMNS)


class DrawDataValidationError(ValueError):
    """历史开奖数据不符合规范。"""


class DrawRecord(BaseModel):
    """一条经过字段级和单期业务规则校验的开奖记录。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    issue: str = Field(min_length=1, pattern=r"^\d+$")
    draw_date: date
    front_1: int = Field(ge=1, le=35)
    front_2: int = Field(ge=1, le=35)
    front_3: int = Field(ge=1, le=35)
    front_4: int = Field(ge=1, le=35)
    front_5: int = Field(ge=1, le=35)
    back_1: int = Field(ge=1, le=12)
    back_2: int = Field(ge=1, le=12)

    _issue_pattern: ClassVar[re.Pattern[str]] = re.compile(r"^\d+$")

    @field_validator("issue", mode="before")
    @classmethod
    def validate_issue_type_and_format(cls, value: object) -> str:
        """要求期号显式使用字符串，以免前导零在读取时丢失。"""
        if not isinstance(value, str):
            raise ValueError("期号必须以字符串提供")
        normalized = value.strip()
        if not cls._issue_pattern.fullmatch(normalized):
            raise ValueError("期号只能包含数字")
        return normalized

    @field_validator("draw_date", mode="before")
    @classmethod
    def validate_draw_date_format(cls, value: object) -> object:
        """CSV 日期只接受 YYYY-MM-DD；允许校验结果再次进入校验器。"""
        if isinstance(value, datetime):
            if value.time() != time.min:
                raise ValueError("开奖日期不能包含非零时间")
            return value.date()
        if isinstance(value, date):
            return value
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("开奖日期必须使用 YYYY-MM-DD 格式")
        return value

    @field_validator(*NUMBER_COLUMNS, mode="before")
    @classmethod
    def validate_integer_number(cls, value: object) -> int:
        """拒绝布尔值、小数和数字字符串，避免静默修正脏数据。"""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("号码必须是整数")
        return int(value)

    @model_validator(mode="after")
    def validate_number_order_and_uniqueness(self) -> Self:
        """前区和后区号码必须分别唯一并严格升序。"""
        if len(set(self.front_numbers)) != len(self.front_numbers):
            raise ValueError("前区号码不得重复")
        if self.front_numbers != tuple(sorted(self.front_numbers)):
            raise ValueError("前区号码必须严格升序")
        if len(set(self.back_numbers)) != len(self.back_numbers):
            raise ValueError("后区号码不得重复")
        if self.back_numbers != tuple(sorted(self.back_numbers)):
            raise ValueError("后区号码必须严格升序")
        return self

    @property
    def front_numbers(self) -> tuple[int, int, int, int, int]:
        """返回前区号码。"""
        return (self.front_1, self.front_2, self.front_3, self.front_4, self.front_5)

    @property
    def back_numbers(self) -> tuple[int, int]:
        """返回后区号码。"""
        return (self.back_1, self.back_2)


def _format_pydantic_errors(error: ValidationError) -> str:
    """把 Pydantic 错误压缩成稳定、可读的单行信息。"""
    messages: list[str] = []
    for detail in error.errors(include_url=False):
        location = ".".join(str(part) for part in detail["loc"]) or "record"
        messages.append(f"{location}: {detail['msg']}")
    return "; ".join(messages)


def validate_draw_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """校验历史开奖表并返回列和标量类型规范化后的副本。

    除单行号码规则外，还要求期号、开奖日期唯一且均按时间方向严格递增。
    该函数不对原始 ``frame`` 做原地修改。
    """
    actual_columns = tuple(frame.columns)
    if actual_columns != CSV_COLUMNS:
        raise DrawDataValidationError(
            f"CSV 列名或顺序不正确；期望 {CSV_COLUMNS}，实际 {actual_columns}"
        )
    if frame.empty:
        raise DrawDataValidationError("历史开奖数据不能为空")

    validated_records: list[DrawRecord] = []
    for row_number, raw_record in enumerate(frame.to_dict(orient="records"), start=2):
        try:
            validated_records.append(DrawRecord.model_validate(raw_record))
        except ValidationError as error:
            formatted = _format_pydantic_errors(error)
            raise DrawDataValidationError(f"CSV 第 {row_number} 行校验失败：{formatted}") from error

    issues = [record.issue for record in validated_records]
    if len(set(issues)) != len(issues):
        raise DrawDataValidationError("期号不得重复")
    numeric_issues = [int(issue) for issue in issues]
    if any(current <= previous for previous, current in pairwise(numeric_issues)):
        raise DrawDataValidationError("期号必须按数值严格递增")

    draw_dates = [record.draw_date for record in validated_records]
    if len(set(draw_dates)) != len(draw_dates):
        raise DrawDataValidationError("开奖日期不得重复")
    if any(current <= previous for previous, current in pairwise(draw_dates)):
        raise DrawDataValidationError("开奖日期必须严格递增")

    normalized = pd.DataFrame(
        [record.model_dump(mode="python") for record in validated_records], columns=CSV_COLUMNS
    )
    normalized["issue"] = normalized["issue"].astype("string")
    for column in NUMBER_COLUMNS:
        normalized[column] = normalized[column].astype("int64")
    return normalized


def load_draws_csv(path: str | Path) -> pd.DataFrame:
    """以保留期号前导零的方式读取并校验历史开奖 CSV。"""
    source = Path(path)
    try:
        frame = pd.read_csv(source, dtype={"issue": "string"})
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise DrawDataValidationError(f"无法读取历史开奖 CSV：{source}") from error
    return validate_draw_dataframe(frame)


def write_validated_draws_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """先校验再以规范列顺序保存历史开奖 CSV，并返回保存路径。"""
    validated = validate_draw_dataframe(frame)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    return target
