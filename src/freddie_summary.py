"""Standardized annual summaries for Freddie Mac origination data."""

import pandas as pd


NUMERIC_SUMMARY_COLUMNS = {
    "CLASSIC FICO": "FICO",
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO_CLEAN": "DTI",
    "ORIGINAL LOAN-TO-VALUE (LTV)_CLEAN": "LTV",
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)_CLEAN": "CLTV",
    "ORIGINAL UPB": "UPB",
    "ORIGINAL INTEREST RATE": "INTEREST RATE",
    "ORIGINAL LOAN TERM": "LOAN TERM",
    "MORTGAGE INSURANCE PERCENTAGE (MI %)": "MI PERCENTAGE",
}


CATEGORICAL_SUMMARY_COLUMNS = {
    "FIRST TIME HOMEBUYER INDICATOR_CLEAN":
        "FIRST TIME HOMEBUYER",
    "NUMBER OF UNITS":
        "NUMBER OF UNITS",
    "OCCUPANCY STATUS":
        "OCCUPANCY STATUS",
    "CHANNEL":
        "CHANNEL",
    "PROPERTY STATE":
        "PROPERTY STATE",
    "PROPERTY TYPE":
        "PROPERTY TYPE",
    "LOAN PURPOSE":
        "LOAN PURPOSE",
    "NUMBER OF BORROWERS":
        "NUMBER OF BORROWERS",
}


def create_numeric_summary(
    data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Create a standardized numeric summary for one year."""

    missing_columns = [
        column
        for column in NUMERIC_SUMMARY_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing numeric summary columns: {missing_columns}"
        )

    numeric_data = (
        data[list(NUMERIC_SUMMARY_COLUMNS)]
        .rename(columns=NUMERIC_SUMMARY_COLUMNS)
    )

    summary = (
        numeric_data
        .describe()
        .transpose()
        .rename(
            columns={
                "25%": "q1",
                "50%": "median",
                "75%": "q3",
            }
        )
    )

    summary["missing"] = numeric_data.isna().sum()
    summary["missing_percentage"] = (
        numeric_data.isna().mean() * 100
    )

    summary = (
        summary
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    summary.insert(0, "year", year)

    return summary[
        [
            "year",
            "variable",
            "count",
            "missing",
            "missing_percentage",
            "mean",
            "std",
            "min",
            "q1",
            "median",
            "q3",
            "max",
        ]
    ]


def create_categorical_summary(
    data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Create category counts and percentages for one year."""

    missing_columns = [
        column
        for column in CATEGORICAL_SUMMARY_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing categorical summary columns: {missing_columns}"
        )

    summary_rows = []

    for column, variable_name in (
        CATEGORICAL_SUMMARY_COLUMNS.items()
    ):
        values = (
            data[column]
            .astype("string")
            .fillna("<MISSING>")
        )

        counts = values.value_counts(dropna=False)

        for category, count in counts.items():
            summary_rows.append(
                {
                    "year": year,
                    "variable": variable_name,
                    "category": category,
                    "count": int(count),
                    "percentage": (
                        float(count) / len(data) * 100
                    ),
                }
            )

    return pd.DataFrame(summary_rows)


def create_annual_summary(
    data: pd.DataFrame,
    year: int,
) -> dict[str, pd.DataFrame]:
    """Create all standardized summaries for one year."""

    return {
        "numeric": create_numeric_summary(data, year),
        "categorical": create_categorical_summary(data, year),
    }