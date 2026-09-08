"""Cleaning functions for Freddie Mac origination data."""

import pandas as pd

from .freddie_config import (
    CATEGORICAL_SENTINELS,
    NUMERIC_SENTINELS,
    POSTAL_CODE_SENTINEL,
    POSTAL_CODE_WIDTH,
)


def clean_origination(
    origination: pd.DataFrame,
) -> pd.DataFrame:
    """Create cleaned origination fields while preserving raw columns."""

    cleaned = origination.copy()

    required_columns = (
        set(NUMERIC_SENTINELS)
        | set(CATEGORICAL_SENTINELS)
        | {"POSTAL CODE"}
    )

    missing_columns = required_columns - set(
        cleaned.columns
    )

    if missing_columns:
        raise KeyError(
            "Required cleaning columns are missing: "
            f"{sorted(missing_columns)}"
        )

    # Convert documented numeric sentinel values to missing.
    for column, sentinel in NUMERIC_SENTINELS.items():
        numeric_values = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

        cleaned[f"{column}_CLEAN"] = (
            numeric_values.mask(
                numeric_values.eq(sentinel)
            )
        )

    # Convert documented categorical sentinel values to missing.
    for column, sentinel in (
        CATEGORICAL_SENTINELS.items()
    ):
        categorical_values = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

        cleaned[f"{column}_CLEAN"] = (
            categorical_values.mask(
                categorical_values.eq(sentinel)
            )
        )

    # Preserve three-character postal-code formatting and convert
    # documented unavailable code 000 to missing.
    postal_code = (
        cleaned["POSTAL CODE"]
        .astype("string")
        .str.strip()
        .str.zfill(POSTAL_CODE_WIDTH)
    )

    cleaned["POSTAL CODE_CLEAN"] = (
        postal_code.mask(
            postal_code.eq(
                POSTAL_CODE_SENTINEL
            )
        )
    )

    # Harmonize Number of Borrowers across the 2018 Q2
    # disclosure-definition change:
    #   2018 Q1 and earlier: 1 = one borrower, 2 = multiple
    #   2018 Q2 and later:   exact borrower counts from 1 to 10
    number_of_borrowers = cleaned[
        "NUMBER OF BORROWERS_CLEAN"
    ]

    borrower_group = pd.Series(
        pd.NA,
        index=cleaned.index,
        dtype="string",
    )

    borrower_group = borrower_group.mask(
        number_of_borrowers.eq(1),
        "1",
    )

    borrower_group = borrower_group.mask(
        number_of_borrowers.between(
            2,
            10,
            inclusive="both",
        ),
        "2+",
    )

    cleaned[
        "NUMBER OF BORROWERS GROUP_CLEAN"
    ] = borrower_group

    return cleaned