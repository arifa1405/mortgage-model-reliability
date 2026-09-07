"""Data-quality profiling for Freddie Mac origination data."""

import pandas as pd

from .freddie_config import (
    ACE_PDR_AVAILABLE_YEAR,
    BORROWER_COUNT_CHANGE_QUARTER,
    BORROWER_COUNT_CHANGE_YEAR,
    DOCUMENTED_CODE_VALUES,
    DOCUMENTED_NUMERIC_RANGES,
    DOCUMENTED_SENTINELS,
    EXPANDED_RATIO_MAXIMUM,
    EXPANDED_RATIO_MINIMUM,
    PRIOR_CLTV_MAXIMUM,
    PRIOR_CLTV_MINIMUM,
    PRIOR_LTV_MAXIMUM,
    PRIOR_LTV_MINIMUM,
    PROPERTY_VALUATION_AVAILABLE_YEAR,
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
        return str(int(value))

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


def _effective_date_mask(
    data: pd.DataFrame,
    year: int,
    change_year: int,
    change_quarter: int,
) -> pd.Series:
    """Return rows governed by a year-and-quarter disclosure change."""

    if year > change_year:
        return pd.Series(
            True,
            index=data.index,
            dtype="boolean",
        )

    if year < change_year:
        return pd.Series(
            False,
            index=data.index,
            dtype="boolean",
        )

    return (
        _extract_origination_quarter(data)
        .ge(change_quarter)
        .fillna(False)
        .astype("boolean")
    )


def _create_ratio_warnings(
    data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Flag LTV and CLTV values inconsistent with disclosure rules."""

    required_columns = {
        "LOAN IDENTIFIER",
        "ORIGINAL LOAN-TO-VALUE (LTV)",
        "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)",
        "HARP INDICATOR",
        "PRE-HARP LOAN SEQUENCE NUMBER",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(
            "Required ratio-quality columns are missing: "
            f"{sorted(missing_columns)}"
        )

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

    valid_ltv = ltv.mask(
        ltv.eq(RATIO_SENTINEL)
    )

    valid_cltv = cltv.mask(
        cltv.eq(RATIO_SENTINEL)
    )

    origination_quarter = _extract_origination_quarter(
        data
    )

    expanded_ratio_rules = _effective_date_mask(
        data=data,
        year=year,
        change_year=RATIO_DISCLOSURE_CHANGE_YEAR,
        change_quarter=RATIO_DISCLOSURE_CHANGE_QUARTER,
    )

    harp_indicator = (
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

    pre_harp_present = (
        pre_harp_sequence.notna()
        & pre_harp_sequence.ne("")
    )

    # Before 2018 Q2, expanded ratio ranges apply to documented
    # HARP loans. Requiring both fields makes this a consistency rule.
    expanded_or_harp = (
        expanded_ratio_rules
        | (
            harp_indicator
            & pre_harp_present
        )
    )

    ltv_minimum = pd.Series(
        PRIOR_LTV_MINIMUM,
        index=data.index,
        dtype="float64",
    ).mask(
        expanded_or_harp,
        EXPANDED_RATIO_MINIMUM,
    )

    ltv_maximum = pd.Series(
        PRIOR_LTV_MAXIMUM,
        index=data.index,
        dtype="float64",
    ).mask(
        expanded_or_harp,
        EXPANDED_RATIO_MAXIMUM,
    )

    cltv_minimum = pd.Series(
        PRIOR_CLTV_MINIMUM,
        index=data.index,
        dtype="float64",
    ).mask(
        expanded_or_harp,
        EXPANDED_RATIO_MINIMUM,
    )

    cltv_maximum = pd.Series(
        PRIOR_CLTV_MAXIMUM,
        index=data.index,
        dtype="float64",
    ).mask(
        expanded_or_harp,
        EXPANDED_RATIO_MAXIMUM,
    )

    low_ltv_warning = (
        valid_ltv.notna()
        & valid_ltv.lt(ltv_minimum)
    )

    high_ltv_warning = (
        valid_ltv.notna()
        & valid_ltv.gt(ltv_maximum)
    )

    low_cltv_warning = (
        valid_cltv.notna()
        & valid_cltv.lt(cltv_minimum)
    )

    high_cltv_warning = (
        valid_cltv.notna()
        & valid_cltv.gt(cltv_maximum)
    )

    # Freddie Mac instructs that CLTV should be reported as 999 when
    # the calculated CLTV is lower than LTV.
    cltv_below_ltv_warning = (
        valid_ltv.notna()
        & valid_cltv.notna()
        & valid_cltv.lt(valid_ltv)
    )

    warning_mask = (
        low_ltv_warning
        | high_ltv_warning
        | low_cltv_warning
        | high_cltv_warning
        | cltv_below_ltv_warning
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
        origination_quarter.loc[warning_mask],
    )

    warnings["LOW LTV FLAG"] = (
        low_ltv_warning.loc[warning_mask]
    )

    warnings["HIGH LTV FLAG"] = (
        high_ltv_warning.loc[warning_mask]
    )

    warnings["LOW CLTV FLAG"] = (
        low_cltv_warning.loc[warning_mask]
    )

    warnings["HIGH CLTV FLAG"] = (
        high_cltv_warning.loc[warning_mask]
    )

    warnings["CLTV BELOW LTV FLAG"] = (
        cltv_below_ltv_warning.loc[warning_mask]
    )

    return warnings.reset_index(drop=True)


def _create_numeric_range_warnings(
    data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Summarize non-sentinel values outside documented numeric ranges."""

    warning_rows = []

    for column, (
        minimum,
        maximum,
    ) in DOCUMENTED_NUMERIC_RANGES.items():
        if column not in data.columns:
            continue

        raw_values = data[column]
        normalized_values = raw_values.map(
            _normalize_code
        )
        numeric_values = pd.to_numeric(
            raw_values,
            errors="coerce",
        )

        sentinel = DOCUMENTED_SENTINELS.get(
            column
        )
        normalized_sentinel = _normalize_code(
            sentinel
        )

        eligible = (
            normalized_values.notna()
            & normalized_values.ne(
                normalized_sentinel
            )
        )

        invalid = (
            eligible
            & (
                numeric_values.isna()
                | numeric_values.lt(minimum)
                | numeric_values.gt(maximum)
            )
        )

        invalid_counts = (
            normalized_values.loc[invalid]
            .value_counts(dropna=False)
        )

        for value, count in invalid_counts.items():
            warning_rows.append(
                {
                    "year": year,
                    "column": column,
                    "unexpected_value": value,
                    "count": int(count),
                    "rule": (
                        f"non-sentinel value must be between "
                        f"{minimum} and {maximum}"
                    ),
                }
            )

    return pd.DataFrame(
        warning_rows,
        columns=[
            "year",
            "column",
            "unexpected_value",
            "count",
            "rule",
        ],
    )


def _create_definition_warnings(
    data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Check codes whose documented meaning changes over time."""

    warning_rows = []

    borrower_column = "NUMBER OF BORROWERS"

    if borrower_column in data.columns:
        borrower_count = pd.to_numeric(
            data[borrower_column],
            errors="coerce",
        ).mask(
            lambda values: values.eq(
                DOCUMENTED_SENTINELS[
                    borrower_column
                ]
            )
        )

        exact_count_rules = _effective_date_mask(
            data=data,
            year=year,
            change_year=BORROWER_COUNT_CHANGE_YEAR,
            change_quarter=BORROWER_COUNT_CHANGE_QUARTER,
        )

        invalid_borrower_definition = (
            ~exact_count_rules
            & borrower_count.gt(2)
            & borrower_count.le(10)
        )

        invalid_counts = (
            data.loc[
                invalid_borrower_definition,
                borrower_column,
            ]
            .map(_normalize_code)
            .value_counts()
        )

        for value, count in invalid_counts.items():
            warning_rows.append(
                {
                    "year": year,
                    "column": borrower_column,
                    "unexpected_value": value,
                    "count": int(count),
                    "rule": (
                        "2018 Q1 and earlier permit 1, 2, "
                        "or unavailable code 99"
                    ),
                }
            )

    # The Pre-HARP identifier is populated only for Relief Refinance
    # loans, and all other loans must leave it blank.
    if {
        "HARP INDICATOR",
        "PRE-HARP LOAN SEQUENCE NUMBER",
    }.issubset(data.columns):
        harp_indicator = (
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

        pre_harp_present = (
            pre_harp_sequence.notna()
            & pre_harp_sequence.ne("")
        )

        harp_without_link = (
            harp_indicator
            & ~pre_harp_present
        )

        link_without_harp = (
            ~harp_indicator
            & pre_harp_present
        )

        if harp_without_link.any():
            warning_rows.append(
                {
                    "year": year,
                    "column": (
                        "HARP INDICATOR / "
                        "PRE-HARP LOAN SEQUENCE NUMBER"
                    ),
                    "unexpected_value": (
                        "Y_WITHOUT_PRE_HARP_ID"
                    ),
                    "count": int(
                        harp_without_link.sum()
                    ),
                    "rule": (
                        "HARP indicator Y requires a "
                        "Pre-HARP loan sequence number"
                    ),
                }
            )

        if link_without_harp.any():
            warning_rows.append(
                {
                    "year": year,
                    "column": (
                        "HARP INDICATOR / "
                        "PRE-HARP LOAN SEQUENCE NUMBER"
                    ),
                    "unexpected_value": (
                        "PRE_HARP_ID_WITHOUT_Y"
                    ),
                    "count": int(
                        link_without_harp.sum()
                    ),
                    "rule": (
                        "Pre-HARP loan sequence number "
                        "is populated only when the "
                        "HARP indicator is Y"
                    ),
                }
            )

        # Freddie Mac discloses DTI as unavailable (999) for
        # all HARP/Relief Refinance loans.
        dti_column = (
            "ORIGINAL DEBT-TO-INCOME (DTI) RATIO"
        )

        if dti_column in data.columns:
            dti_code = data[
                dti_column
            ].map(_normalize_code)

            populated_harp_dti = (
                harp_indicator
                & dti_code.ne("999")
            )

            if populated_harp_dti.any():
                warning_rows.append(
                    {
                        "year": year,
                        "column": dti_column,
                        "unexpected_value": (
                            "POPULATED_FOR_HARP"
                        ),
                        "count": int(
                            populated_harp_dti.sum()
                        ),
                        "rule": (
                            "HARP loan DTI must use "
                            "unavailable code 999"
                        ),
                    }
                )

    valuation_column = "PROPERTY VALUATION METHOD"

    if valuation_column in data.columns:
        valuation_method = (
            data[valuation_column]
            .map(_normalize_code)
        )

        if year < PROPERTY_VALUATION_AVAILABLE_YEAR:
            invalid_valuation_definition = (
                valuation_method.notna()
                & valuation_method.ne("7")
            )
            valuation_rule = (
                "usable Property Valuation Method codes "
                "begin with 2017 originations"
            )
        elif year < ACE_PDR_AVAILABLE_YEAR:
            invalid_valuation_definition = (
                valuation_method.eq("4")
            )
            valuation_rule = (
                "Property Valuation Method code 4 "
                "begins in 2022"
            )
        else:
            invalid_valuation_definition = pd.Series(
                False,
                index=data.index,
            )
            valuation_rule = ""

        invalid_counts = (
            valuation_method.loc[
                invalid_valuation_definition
            ]
            .value_counts()
        )

        for value, count in invalid_counts.items():
            warning_rows.append(
                {
                    "year": year,
                    "column": valuation_column,
                    "unexpected_value": value,
                    "count": int(count),
                    "rule": valuation_rule,
                }
            )

    return pd.DataFrame(
        warning_rows,
        columns=[
            "year",
            "column",
            "unexpected_value",
            "count",
            "rule",
        ],
    )


def create_origination_quality_report(
    data: pd.DataFrame,
    year: int,
) -> dict[str, pd.DataFrame]:
    """Create code, sentinel, missingness, range, and ratio reports."""

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

    numeric_range_warnings = (
        _create_numeric_range_warnings(
            data=data,
            year=year,
        )
    )

    definition_warnings = (
        _create_definition_warnings(
            data=data,
            year=year,
        )
    )

    ratio_warnings = _create_ratio_warnings(
        data=data,
        year=year,
    )

    return {
        "unexpected_codes": unexpected_codes,
        "sentinel_counts": sentinel_counts,
        "missingness": missingness,
        "numeric_range_warnings": numeric_range_warnings,
        "definition_warnings": definition_warnings,
        "ratio_warnings": ratio_warnings,
    }
