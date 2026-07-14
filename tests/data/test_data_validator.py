"""历史开奖数据校验测试。"""

from pathlib import Path

import pandas as pd
import pytest

from dlt_number_analysis.data import (
    CSV_COLUMNS,
    DrawDataValidationError,
    load_draws_csv,
    validate_draw_dataframe,
    write_validated_draws_csv,
)


def make_valid_draws() -> pd.DataFrame:
    """构造不代表真实开奖的最小合法测试数据。"""
    return pd.DataFrame(
        [
            ["07001", "2007-05-28", 1, 2, 11, 20, 35, 3, 8],
            ["07002", "2007-05-30", 2, 7, 12, 23, 31, 4, 9],
            ["07003", "2007-06-02", 5, 15, 16, 25, 35, 1, 12],
        ],
        columns=CSV_COLUMNS,
    )


def test_validate_draw_dataframe_normalizes_without_mutating_input() -> None:
    original = make_valid_draws()
    validated = validate_draw_dataframe(original)

    assert list(validated.columns) == list(CSV_COLUMNS)
    assert validated["issue"].tolist() == ["07001", "07002", "07003"]
    assert str(validated["issue"].dtype) == "string"
    assert validated.loc[0, "draw_date"].isoformat() == "2007-05-28"
    assert original.loc[0, "draw_date"] == "2007-05-28"


def test_csv_round_trip_preserves_issue_leading_zero(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "draws.csv"
    written = write_validated_draws_csv(make_valid_draws(), target)
    loaded = load_draws_csv(written)

    assert written == target
    assert loaded["issue"].tolist() == ["07001", "07002", "07003"]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("front_1", 0, "CSV 第 2 行校验失败"),
        ("front_5", 36, "CSV 第 2 行校验失败"),
        ("back_1", 0, "CSV 第 2 行校验失败"),
        ("back_2", 13, "CSV 第 2 行校验失败"),
        ("front_2", 1, "前区号码不得重复"),
        ("front_2", 0.5, "号码必须是整数"),
        ("back_2", 3, "后区号码不得重复"),
    ],
)
def test_rejects_invalid_number_rules(column: str, value: object, message: str) -> None:
    draws = make_valid_draws()
    draws[column] = draws[column].astype(object)
    draws.loc[0, column] = value

    with pytest.raises(DrawDataValidationError, match=message):
        validate_draw_dataframe(draws)


def test_rejects_non_ascending_numbers() -> None:
    draws = make_valid_draws()
    draws.loc[0, ["front_1", "front_2"]] = [2, 1]

    with pytest.raises(DrawDataValidationError, match="前区号码必须严格升序"):
        validate_draw_dataframe(draws)


def test_rejects_missing_extra_or_reordered_columns() -> None:
    draws = make_valid_draws()[list(reversed(CSV_COLUMNS))]

    with pytest.raises(DrawDataValidationError, match="CSV 列名或顺序不正确"):
        validate_draw_dataframe(draws)


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("issue", ["07001", "07001", "07003"], "期号不得重复"),
        ("issue", ["07001", "07003", "07002"], "期号必须按数值严格递增"),
        ("draw_date", ["2007-05-28", "2007-05-28", "2007-06-02"], "开奖日期不得重复"),
        (
            "draw_date",
            ["2007-05-28", "2007-06-02", "2007-05-30"],
            "开奖日期必须严格递增",
        ),
    ],
)
def test_rejects_duplicate_or_non_chronological_rows(
    field: str, values: list[str], message: str
) -> None:
    draws = make_valid_draws()
    draws[field] = values

    with pytest.raises(DrawDataValidationError, match=message):
        validate_draw_dataframe(draws)


def test_rejects_empty_data() -> None:
    with pytest.raises(DrawDataValidationError, match="历史开奖数据不能为空"):
        validate_draw_dataframe(pd.DataFrame(columns=CSV_COLUMNS))
