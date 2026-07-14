"""Canonical identity for validated logical draw history."""

from __future__ import annotations

import hashlib

import pandas as pd

from dlt_number_analysis.data.data_validator import CSV_COLUMNS, validate_draw_dataframe


def canonical_history_bytes(draws: pd.DataFrame) -> bytes:
    """Serialize logical records with fixed fields, two-digit numbers, and LF endings."""
    validated = validate_draw_dataframe(draws)
    lines = [",".join(CSV_COLUMNS)]
    for row in validated.to_dict(orient="records"):
        fields = [str(row["issue"]), row["draw_date"].isoformat()]
        fields.extend(f"{int(row[column]):02d}" for column in CSV_COLUMNS[2:])
        lines.append(",".join(fields))
    return ("\n".join(lines) + "\n").encode("utf-8")


def canonical_history_sha256(draws: pd.DataFrame) -> str:
    """Hash logical records independently of raw CSV formatting and scalar dtypes."""
    return hashlib.sha256(canonical_history_bytes(draws)).hexdigest()
