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


def _create_high_ratio_warnings(
    data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Flag high LTV or CLTV values without complete HARP support."""

    ltv = pd.to_numeric(
        data["ORIGINAL LOAN-TO-VALUE (LTV)"],
        errors="coerce",
    )

    cltv = pd.to_numeric(
        data[
            "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)"
        ],
        errors="coerce",
    )

    high_ltv_mask = (
        ltv.gt(105)
        & ltv.ne(999)
    )

    high_cltv_mask = (
        cltv.gt(105)
        & cltv.ne(999)
    )

    harp_supported = (
        data["HARP INDICATOR"]
        .astype("string")
        .str.strip()
        .eq("Y")
        .fillna(False)
    )

    pre_harp_sequence = (
        data["PRE-HARP LOAN SEQUENCE NUMBER"]
        .astype("string")
        .str.strip()
    )

    pre_harp_supported = (
        pre_harp_sequence.notna()
        & pre_harp_sequence.ne("")
    )

    unsupported_high_ratio_mask = (
        (high_ltv_mask | high_cltv_mask)
        & ~(
            harp_supported
            & pre_harp_supported
        )
    )

    warnings = data.loc[
        unsupported_high_ratio_mask,
        [
            "LOAN IDENTIFIER",
            "ORIGINAL LOAN-TO-VALUE (LTV)",
            "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)",
            "HARP INDICATOR",
            "PRE-HARP LOAN SEQUENCE NUMBER",
        ],
    ].copy()

    warnings.insert(
        0,
        "year",
        year,
    )

    warnings["HIGH LTV FLAG"] = high_ltv_mask.loc[
        unsupported_high_ratio_mask
    ]

    warnings["HIGH CLTV FLAG"] = high_cltv_mask.loc[
        unsupported_high_ratio_mask
    ]

    return warnings


def create_origination_quality_report(
    data: pd.DataFrame,
    year: int,
) -> dict[str, pd.DataFrame]:
    """Create code, sentinel, missingness, and high-ratio reports."""

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

        normalized_sentinel = _normalize_code(
            sentinel
        )

        sentinel_count = int(
            normalized_values.eq(
                normalized_sentinel
            ).sum()
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

    high_ratio_warnings = _create_high_ratio_warnings(
        data=data,
        year=year,
    )

    return {
        "unexpected_codes": unexpected_codes,
        "sentinel_counts": sentinel_counts,
        "missingness": missingness,
        "high_ratio_warnings": high_ratio_warnings,
    }