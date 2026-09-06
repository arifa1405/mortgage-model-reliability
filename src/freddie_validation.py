"""Validation checks for Freddie Mac origination data."""

import pandas as pd

from .freddie_config import (
    CATEGORICAL_SENTINELS,
    NUMERIC_SENTINELS,
    ORIGINATION_COLUMNS,
)


def validate_origination(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    expected_rows: int | None = None,
) -> dict[str, int]:
    """Validate schema, identifiers, cleaning, and postal codes."""

    errors = []

    # Validate the raw schema.
    if list(raw.columns) != ORIGINATION_COLUMNS:
        errors.append(
            "Raw origination columns do not match the configured schema."
        )

    # Cleaning must preserve the number of observations.
    if len(raw) != len(cleaned):
        errors.append(
            "Cleaning changed the number of origination rows."
        )

    # Optionally verify the expected sample size.
    if expected_rows is not None and len(raw) != expected_rows:
        errors.append(
            f"Expected {expected_rows} rows but found {len(raw)}."
        )

    # Validate loan identifiers.
    if "LOAN IDENTIFIER" in raw.columns:
        missing_loan_ids = int(
            raw["LOAN IDENTIFIER"].isna().sum()
        )

        duplicate_loan_ids = int(
            raw["LOAN IDENTIFIER"].duplicated().sum()
        )
    else:
        missing_loan_ids = len(raw)
        duplicate_loan_ids = 0

        errors.append(
            "LOAN IDENTIFIER is missing from the raw data."
        )

    if missing_loan_ids > 0:
        errors.append(
            f"Found {missing_loan_ids} missing loan identifiers."
        )

    if duplicate_loan_ids > 0:
        errors.append(
            f"Found {duplicate_loan_ids} duplicate loan identifiers."
        )

    # Determine which cleaned fields should exist.
    expected_clean_columns = (
        [
            f"{column}_CLEAN"
            for column in NUMERIC_SENTINELS
        ]
        + [
            f"{column}_CLEAN"
            for column in CATEGORICAL_SENTINELS
        ]
        + ["POSTAL CODE_CLEAN"]
    )

    missing_clean_columns = [
        column
        for column in expected_clean_columns
        if column not in cleaned.columns
    ]

    if missing_clean_columns:
        errors.append(
            f"Missing clean columns: {missing_clean_columns}"
        )

    # Confirm that numeric sentinels were removed.
    for column, sentinel in NUMERIC_SENTINELS.items():
        clean_column = f"{column}_CLEAN"

        if clean_column in cleaned.columns:
            sentinel_mask = (
                cleaned[clean_column]
                .eq(sentinel)
                .fillna(False)
            )

            remaining_sentinels = int(
                sentinel_mask.sum()
            )

            if remaining_sentinels > 0:
                errors.append(
                    f"{remaining_sentinels} sentinel values remain "
                    f"in {clean_column}."
                )

    # Confirm that categorical sentinels were removed.
    for column, sentinel in CATEGORICAL_SENTINELS.items():
        clean_column = f"{column}_CLEAN"

        if clean_column in cleaned.columns:
            sentinel_mask = (
                cleaned[clean_column]
                .eq(sentinel)
                .fillna(False)
            )

            remaining_sentinels = int(
                sentinel_mask.sum()
            )

            if remaining_sentinels > 0:
                errors.append(
                    f"{remaining_sentinels} sentinel values remain "
                    f"in {clean_column}."
                )

    # Validate the cleaned three-digit postal-code representation.
    if "POSTAL CODE_CLEAN" in cleaned.columns:
        postal_is_valid = (
            cleaned["POSTAL CODE_CLEAN"]
            .astype("string")
            .str.fullmatch(r"\d{3}", na=False)
        )

        invalid_postal_codes = int(
            (~postal_is_valid).sum()
        )
    else:
        invalid_postal_codes = len(cleaned)

    if invalid_postal_codes > 0:
        errors.append(
            f"Found {invalid_postal_codes} invalid postal codes."
        )

    # Stop processing if any critical validation failed.
    if errors:
        raise ValueError(
            "Origination validation failed:\n- "
            + "\n- ".join(errors)
        )

    return {
        "rows": len(cleaned),
        "raw_columns": raw.shape[1],
        "cleaned_columns": cleaned.shape[1],
        "missing_loan_ids": missing_loan_ids,
        "duplicate_loan_ids": duplicate_loan_ids,
        "invalid_postal_codes": invalid_postal_codes,
    }