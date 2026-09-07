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
    """Validate schema, identifiers, cleaning, and derived fields."""

    errors = []

    # The input must follow the current 31-field Freddie Mac layout.
    if list(raw.columns) != ORIGINATION_COLUMNS:
        errors.append(
            "Raw origination columns do not match the configured schema."
        )

    # Cleaning is additive: it must preserve observations and raw fields.
    if len(raw) != len(cleaned):
        errors.append(
            "Cleaning changed the number of origination rows."
        )

    missing_preserved_columns = [
        column
        for column in raw.columns
        if column not in cleaned.columns
    ]

    if missing_preserved_columns:
        errors.append(
            "Cleaning removed raw columns: "
            f"{missing_preserved_columns}"
        )

    changed_raw_columns = [
        column
        for column in raw.columns
        if (
            column in cleaned.columns
            and not raw[column].equals(cleaned[column])
        )
    ]

    if changed_raw_columns:
        errors.append(
            "Cleaning modified preserved raw columns: "
            f"{changed_raw_columns}"
        )

    if expected_rows is not None and len(raw) != expected_rows:
        errors.append(
            f"Expected {expected_rows} rows but found {len(raw)}."
        )

    # Loan identifiers are the record-level key.
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

    # Every documented sentinel receives an additive clean field.
    expected_clean_columns = (
        [
            f"{column}_CLEAN"
            for column in NUMERIC_SENTINELS
        ]
        + [
            f"{column}_CLEAN"
            for column in CATEGORICAL_SENTINELS
        ]
        + [
            "POSTAL CODE_CLEAN",
            "NUMBER OF BORROWERS GROUP_CLEAN",
        ]
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

    # Documented sentinels must not remain in their clean fields.
    for column, sentinel in NUMERIC_SENTINELS.items():
        clean_column = f"{column}_CLEAN"

        if clean_column in cleaned.columns:
            remaining_sentinels = int(
                cleaned[clean_column]
                .eq(sentinel)
                .fillna(False)
                .sum()
            )

            if remaining_sentinels > 0:
                errors.append(
                    f"{remaining_sentinels} sentinel values remain "
                    f"in {clean_column}."
                )

    for column, sentinel in CATEGORICAL_SENTINELS.items():
        clean_column = f"{column}_CLEAN"

        if clean_column in cleaned.columns:
            remaining_sentinels = int(
                cleaned[clean_column]
                .eq(sentinel)
                .fillna(False)
                .sum()
            )

            if remaining_sentinels > 0:
                errors.append(
                    f"{remaining_sentinels} sentinel values remain "
                    f"in {clean_column}."
                )

    # The Release 47 postal field contains three digits. Missing clean
    # values are valid because raw code 000 is documented as unavailable.
    if "POSTAL CODE_CLEAN" in cleaned.columns:
        postal_code = cleaned[
            "POSTAL CODE_CLEAN"
        ].astype("string")

        postal_is_valid = (
            postal_code.isna()
            | postal_code.str.fullmatch(
                r"\d{3}",
                na=False,
            )
        )

        invalid_postal_codes = int(
            (~postal_is_valid).sum()
        )
    else:
        invalid_postal_codes = len(cleaned)

    if invalid_postal_codes > 0:
        errors.append(
            f"Found {invalid_postal_codes} invalid cleaned postal codes."
        )

    # The harmonized borrower group is stable across the 2018 Q2
    # definition change: one borrower versus two or more borrowers.
    borrower_clean_column = "NUMBER OF BORROWERS_CLEAN"
    borrower_group_column = "NUMBER OF BORROWERS GROUP_CLEAN"

    if (
        borrower_clean_column in cleaned.columns
        and borrower_group_column in cleaned.columns
    ):
        borrower_count = pd.to_numeric(
            cleaned[borrower_clean_column],
            errors="coerce",
        )

        expected_group = pd.Series(
            pd.NA,
            index=cleaned.index,
            dtype="string",
        )

        expected_group = expected_group.mask(
            borrower_count.eq(1),
            "1",
        )

        expected_group = expected_group.mask(
            borrower_count.between(
                2,
                10,
                inclusive="both",
            ),
            "2+",
        )

        actual_group = cleaned[
            borrower_group_column
        ].astype("string")

        invalid_borrower_groups = int(
            actual_group.fillna("<MISSING>")
            .ne(
                expected_group.fillna("<MISSING>")
            )
            .sum()
        )
    else:
        invalid_borrower_groups = len(cleaned)

    if invalid_borrower_groups > 0:
        errors.append(
            f"Found {invalid_borrower_groups} invalid borrower groups."
        )

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
        "invalid_borrower_groups": invalid_borrower_groups,
    }
