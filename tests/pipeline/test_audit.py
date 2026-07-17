"""Logical-history identity tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dlt_number_analysis.data import CSV_COLUMNS, load_draws_csv
from dlt_number_analysis.pipeline import build_history_audit, canonical_history_sha256


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["26001", "2026-01-01", 1, 2, 3, 4, 5, 1, 2],
            ["26002", "2026-01-03", 6, 7, 8, 9, 10, 3, 4],
        ],
        columns=CSV_COLUMNS,
    )


def test_canonical_hash_ignores_csv_number_width_and_line_endings(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(
        "issue,draw_date,front_1,front_2,front_3,front_4,front_5,back_1,back_2\n"
        "26001,2026-01-01,1,2,3,4,5,1,2\n"
        "26002,2026-01-03,6,7,8,9,10,3,4\n",
        encoding="utf-8",
        newline="\n",
    )
    second.write_text(
        "issue,draw_date,front_1,front_2,front_3,front_4,front_5,back_1,back_2\r\n"
        "26001,2026-01-01,01,02,03,04,05,01,02\r\n"
        "26002,2026-01-03,06,07,08,09,10,03,04\r\n",
        encoding="utf-8",
        newline="",
    )

    first_draws = load_draws_csv(first)
    second_draws = load_draws_csv(second)
    first_audit = build_history_audit(first, first_draws)
    second_audit = build_history_audit(second, second_draws)

    assert first_audit.raw_file_sha256 != second_audit.raw_file_sha256
    assert first_audit.canonical_history_sha256 == second_audit.canonical_history_sha256


def test_canonical_hash_changes_when_logical_history_changes() -> None:
    original = _history()
    modified = original.copy()
    modified.loc[0, "front_5"] = 11

    assert canonical_history_sha256(original) != canonical_history_sha256(modified)
