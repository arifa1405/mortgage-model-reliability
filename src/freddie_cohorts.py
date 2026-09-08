import re

import pandas as pd


FREDDIE_LOAN_ID_PATTERN = re.compile(
    r"^(?P<product>[FA])"
    r"(?P<year>\d{2})"
    r"Q(?P<quarter>[1-4])"
    r"(?P<sequence>\d{7})$"
)


def parse_freddie_loan_identifier(
    loan_identifier: str,
) -> dict[str, int | str]:
    """
    Validate one Freddie Mac loan identifier and extract its
    origination year, quarter, and cohort.

    Example:
        F15Q10000276 -> 2015, Q1, 2015Q1
    """
    if not isinstance(loan_identifier, str):
        raise ValueError(
            "LOAN IDENTIFIER must be a string. "
            f"Received: {loan_identifier!r}"
        )

    match = FREDDIE_LOAN_ID_PATTERN.fullmatch(loan_identifier)

    if match is None:
        raise ValueError(
            "Invalid Freddie Mac LOAN IDENTIFIER format: "
            f"{loan_identifier!r}"
        )

    origination_year = 2000 + int(match.group("year"))
    origination_quarter = f"Q{match.group('quarter')}"
    origination_cohort = (
        f"{origination_year}{origination_quarter}"
    )

    return {
        "ORIGINATION YEAR": origination_year,
        "ORIGINATION QUARTER": origination_quarter,
        "ORIGINATION COHORT": origination_cohort,
    }


def add_origination_cohorts(
    origination: pd.DataFrame,
    expected_year: int,
) -> pd.DataFrame:
    """
    Add origination year, quarter, and cohort columns by parsing
    the Freddie Mac LOAN IDENTIFIER.

    The extracted year must agree with the expected source-file year.
    """
    required_column = "LOAN IDENTIFIER"

    if required_column not in origination.columns:
        raise KeyError(
            f"Required column is missing: {required_column}"
        )

    cohort_data = origination.copy()

    loan_identifiers = cohort_data[required_column].astype("string")

    valid_identifier_mask = loan_identifiers.str.fullmatch(
        FREDDIE_LOAN_ID_PATTERN,
        na=False,
    )

    if not valid_identifier_mask.all():
        invalid_identifiers = (
            cohort_data.loc[
                ~valid_identifier_mask,
                required_column,
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Invalid Freddie Mac LOAN IDENTIFIER values found. "
            f"Examples: {invalid_identifiers}"
        )

    identifier_parts = loan_identifiers.str.extract(
        FREDDIE_LOAN_ID_PATTERN
    )

    origination_year = (
        2000 + identifier_parts["year"].astype("int64")
    )

    origination_quarter = (
        "Q" + identifier_parts["quarter"]
    )

    year_mismatch_mask = origination_year.ne(expected_year)

    if year_mismatch_mask.any():
        mismatched_identifiers = (
            cohort_data.loc[
                year_mismatch_mask,
                required_column,
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "LOAN IDENTIFIER year does not match the expected "
            f"source-file year {expected_year}. "
            f"Examples: {mismatched_identifiers}"
        )

    cohort_data["ORIGINATION YEAR"] = origination_year
    cohort_data["ORIGINATION QUARTER"] = origination_quarter
    cohort_data["ORIGINATION COHORT"] = (
        origination_year.astype("string")
        + origination_quarter
    )

    return cohort_data

def create_origination_cohort_summary(
    origination: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create annual-quarter origination cohort counts and percentages.
    """
    required_columns = [
        "ORIGINATION YEAR",
        "ORIGINATION QUARTER",
        "ORIGINATION COHORT",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in origination.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing origination cohort columns: "
            f"{missing_columns}"
        )

    cohort_summary = (
        origination
        .groupby(
            required_columns,
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .rename(
            columns={
                "ORIGINATION YEAR": "year",
                "ORIGINATION QUARTER": "quarter",
                "ORIGINATION COHORT": "cohort",
            }
        )
    )

    annual_totals = (
        cohort_summary
        .groupby("year")["count"]
        .transform("sum")
    )

    cohort_summary["percentage"] = (
        cohort_summary["count"]
        / annual_totals
        * 100
    )

    return (
        cohort_summary
        .sort_values(
            by=["year", "quarter"]
        )
        .reset_index(drop=True)
    )

def validate_origination_cohorts(
    origination: pd.DataFrame,
    expected_year: int,
) -> dict[str, int]:
    """
    Validate saved origination cohort columns against the
    original Freddie Mac loan identifiers.
    """
    cohort_columns = [
        "ORIGINATION YEAR",
        "ORIGINATION QUARTER",
        "ORIGINATION COHORT",
    ]

    missing_columns = [
        column
        for column in cohort_columns
        if column not in origination.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing origination cohort columns: "
            f"{missing_columns}"
        )

    source_data = origination.drop(
        columns=cohort_columns
    )

    expected_data = add_origination_cohorts(
        origination=source_data,
        expected_year=expected_year,
    )

    actual_year = pd.to_numeric(
        origination["ORIGINATION YEAR"],
        errors="coerce",
    )

    expected_year_values = expected_data[
        "ORIGINATION YEAR"
    ]

    invalid_year_values = int(
        actual_year
        .ne(expected_year_values)
        .sum()
    )

    actual_quarter = origination[
        "ORIGINATION QUARTER"
    ].astype("string")

    expected_quarter = expected_data[
        "ORIGINATION QUARTER"
    ].astype("string")

    invalid_quarter_values = int(
        actual_quarter
        .fillna("<MISSING>")
        .ne(
            expected_quarter.fillna("<MISSING>")
        )
        .sum()
    )

    actual_cohort = origination[
        "ORIGINATION COHORT"
    ].astype("string")

    expected_cohort = expected_data[
        "ORIGINATION COHORT"
    ].astype("string")

    invalid_cohort_values = int(
        actual_cohort
        .fillna("<MISSING>")
        .ne(
            expected_cohort.fillna("<MISSING>")
        )
        .sum()
    )

    errors = []

    if invalid_year_values > 0:
        errors.append(
            f"Found {invalid_year_values} invalid "
            "ORIGINATION YEAR values."
        )

    if invalid_quarter_values > 0:
        errors.append(
            f"Found {invalid_quarter_values} invalid "
            "ORIGINATION QUARTER values."
        )

    if invalid_cohort_values > 0:
        errors.append(
            f"Found {invalid_cohort_values} invalid "
            "ORIGINATION COHORT values."
        )

    if errors:
        raise ValueError(
            "Origination cohort validation failed:\n- "
            + "\n- ".join(errors)
        )

    return {
        "rows": len(origination),
        "invalid_origination_years": (
            invalid_year_values
        ),
        "invalid_origination_quarters": (
            invalid_quarter_values
        ),
        "invalid_origination_cohorts": (
            invalid_cohort_values
        ),
    }