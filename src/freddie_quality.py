"""Data-quality profiling for Freddie Mac origination data."""

import pandas as pd

from .freddie_config import (
    DOCUMENTED_CODE_VALUES,
    DOCUMENTED_SENTINELS,
    EXPANDED_RATIO_MAXIMUM,
    PRIOR_CLTV_MAXIMUM,
    PRIOR_LTV_MAXIMUM,
    RATIO_DISCLOSURE_CHANGE_QUARTER,
    RATIO_DISCLOSURE_CHANGE_YEAR,
    RATIO_SENTINEL,
)


def _normalize_code(
    value: object,
) -> str | None:
    """Convert code values to comparable strings."""

    if pd.isna(value):
        return None

    if (
        isinstance(value, float)
        and value.is_integer()
    ):
        return str(
            int(value)
        )

    return str(value).strip()


def _extract_origination_quarter(
    data: pd.DataFrame,
) -> pd.Series:
    """Extract the origination quarter from each loan identifier."""

    loan_identifier = (
        data["LOAN IDENTIFIER"]
        .astype("string")
        .str.strip()
    )

    quarter = loan_identifier.str.extract(
        r"^[A-Z]\d{2}Q([1-4])",
        expand=False,
    )

    return pd.to_numeric(
        quarter,
        errors="coerce",
    ).astype("Int64")


def _create_high_ratio_warnings(
    data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Flag ratio values inconsistent with applicable disclosure rules."""

    required_columns = {
        "LOAN IDENTIFIER",
        "ORIGINAL LOAN-TO-VALUE (LTV)",
        "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)",
        "HARP INDICATOR",
        "PRE-HARP LOAN SEQUENCE NUMBER",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Required high-ratio quality columns are missing: "
            f"{sorted(missing_columns)}"
        )

    ltv = pd.to_numeric(
        data[
            "ORIGINAL LOAN-TO-VALUE (LTV)"
        ],
        errors="coerce",
    )

    cltv = pd.to_numeric(
        data[
            "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)"
        ],
        errors="coerce",
    )

    valid_ltv = ltv.mask(
        ltv.eq(RATIO_SENTINEL)
    )

    valid_cltv = cltv.mask(
        cltv.eq(RATIO_SENTINEL)
    )

    origination_quarter = (
        _extract_origination_quarter(data)
    )

    if year > RATIO_DISCLOSURE_CHANGE_YEAR:
        expanded_ratio_rules = pd.Series(
            True,
            index=data.index,
            dtype="boolean",
        )

    elif year < RATIO_DISCLOSURE_CHANGE_YEAR:
        expanded_ratio_rules = pd.Series(
            False,
            index=data.index,
            dtype="boolean",
        )

    else:
        expanded_ratio_rules = (
            origination_quarter
            .ge(
                RATIO_DISCLOSURE_CHANGE_QUARTER
            )
            .fillna(False)
            .astype("boolean")
        )

    harp_supported = (
        data["HARP INDICATOR"]
        .astype("string")
        .str.strip()
        .eq("Y")
        .fillna(False)
    )

    pre_harp_sequence = (
        data[
            "PRE-HARP LOAN SEQUENCE NUMBER"
        ]
        .astype("string")
        .str.strip()
    )

    pre_harp_supported = (
        pre_harp_sequence.notna()
        & pre_harp_sequence.ne("")
    )

    complete_harp_support = (
        harp_supported
        & pre_harp_supported
    )

    # Before 2018 Q2, LTV above 105% requires complete
    # HARP support.
    unsupported_prior_ltv = (
        ~expanded_ratio_rules
        & valid_ltv.gt(
            PRIOR_LTV_MAXIMUM
        )
        & valid_ltv.le(
            EXPANDED_RATIO_MAXIMUM
        )
        & ~complete_harp_support
    )

    # Before 2018 Q2, CLTV above 200% requires complete
    # HARP support.
    unsupported_prior_cltv = (
        ~expanded_ratio_rules
        & valid_cltv.gt(
            PRIOR_CLTV_MAXIMUM
        )
        & valid_cltv.le(
            EXPANDED_RATIO_MAXIMUM
        )
        & ~complete_harp_support
    )

    # Values above 998% are outside the documented range
    # under both disclosure definitions.
    invalid_ltv_maximum = valid_ltv.gt(
        EXPANDED_RATIO_MAXIMUM
    )

    invalid_cltv_maximum = valid_cltv.gt(
        EXPANDED_RATIO_MAXIMUM
    )

    high_ltv_warning = (
        unsupported_prior_ltv
        | invalid_ltv_maximum
    )

    high_cltv_warning = (
        unsupported_prior_cltv
        | invalid_cltv_maximum
    )

    warning_mask = (
        high_ltv_warning
        | high_cltv_warning
    )

    warnings = data.loc[
        warning_mask,
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

    warnings.insert(
        1,
        "origination_quarter",
        origination_quarter.loc[
            warning_mask
        ],
    )

    warnings["HIGH LTV FLAG"] = (
        high_ltv_warning.loc[
            warning_mask
        ]
    )

    warnings["HIGH CLTV FLAG"] = (
        high_cltv_warning.loc[
            warning_mask
        ]
    )

    return warnings.reset_index(
        drop=True
    )


def create_origination_quality_report(
    data: pd.DataFrame,
    year: int,
) -> dict[str, pd.DataFrame]:
    """Create code, sentinel, missingness, and ratio reports."""

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
                    "unexpected_value": (
                        "<MISSING COLUMN>"
                    ),
                    "count": 0,
                }
            )
            continue

        normalized_values = data[column].map(
            _normalize_code
        )

        value_counts = (
            normalized_values.value_counts(
                dropna=True
            )
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

    for column, sentinel in (
        DOCUMENTED_SENTINELS.items()
    ):
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

        normalized_sentinel = (
            _normalize_code(sentinel)
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
                "sentinel": (
                    normalized_sentinel
                ),
                "count": sentinel_count,
                "percentage": (
                    sentinel_count
                    / len(data)
                    * 100
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
                    missing_count
                    / len(data)
                    * 100
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

    high_ratio_warnings = (
        _create_high_ratio_warnings(
            data=data,
            year=year,
        )
    )

    return {
        "unexpected_codes": unexpected_codes,
        "sentinel_counts": sentinel_counts,
        "missingness": missingness,
        "high_ratio_warnings": (
            high_ratio_warnings
        ),
    }