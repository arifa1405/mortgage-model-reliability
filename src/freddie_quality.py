"""Data-quality profiling for Freddie Mac origination data."""

import pandas as pd

from .freddie_config import (
    DOCUMENTED_CODE_VALUES,
    DOCUMENTED_SENTINELS,
)


def _normalize_code(value: object) -> str | None:
    """Convert code values to comparable strings."""

    if pd.isna(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def create_origination_quality_report(
    data: pd.DataFrame,
    year: int,
) -> dict[str, pd.DataFrame]:
    """Create code, sentinel, and missingness quality reports."""

    unexpected_code_rows = []
    sentinel_rows = []

    for column, allowed_values in (
        DOCUMENTED_CODE_VALUES.items()
    ):
        if column not in data.columns:
            unexpected_code_rows.append(
                {
                    "year": year,
                    "column": column,
                    "unexpected_value": "<MISSING COLUMN>",
                    "count": 0,
                }
            )
            continue

        normalized_values = data[column].map(
            _normalize_code
        )

        value_counts = normalized_values.value_counts(
            dropna=True
        )

        for value, count in value_counts.items():
            if value not in allowed_values:
                unexpected_code_rows.append(
                    {
                        "year": year,
                        "column": column,
                        "unexpected_value": value,
                        "count": int(count),
                    }
                )

    for column, sentinel in DOCUMENTED_SENTINELS.items():
        if column not in data.columns:
            sentinel_rows.append(
                {
                    "year": year,
                    "column": column,
                    "sentinel": str(sentinel),
                    "count": 0,
                    "percentage": 0.0,
                    "column_missing": True,
                }
            )
            continue

        normalized_values = data[column].map(
            _normalize_code
        )

        normalized_sentinel = _normalize_code(sentinel)

        sentinel_count = int(
            normalized_values.eq(normalized_sentinel).sum()
        )

        sentinel_rows.append(
            {
                "year": year,
                "column": column,
                "sentinel": normalized_sentinel,
                "count": sentinel_count,
                "percentage": (
                    sentinel_count / len(data) * 100
                ),
                "column_missing": False,
            }
        )

    missingness_rows = []

    for column in data.columns:
        missing_count = int(
            data[column].isna().sum()
        )

        missingness_rows.append(
            {
                "year": year,
                "column": column,
                "missing_count": missing_count,
                "missing_percentage": (
                    missing_count / len(data) * 100
                ),
            }
        )

    unexpected_codes = pd.DataFrame(
        unexpected_code_rows,
        columns=[
            "year",
            "column",
            "unexpected_value",
            "count",
        ],
    )

    sentinel_counts = pd.DataFrame(
        sentinel_rows
    )

    missingness = pd.DataFrame(
        missingness_rows
    )

    return {
        "unexpected_codes": unexpected_codes,
        "sentinel_counts": sentinel_counts,
        "missingness": missingness,
    }