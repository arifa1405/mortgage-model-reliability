"""Cleaning functions for Freddie Mac origination data."""

import pandas as pd

from .freddie_config import (
    CATEGORICAL_SENTINELS,
    NUMERIC_SENTINELS,
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

    missing_columns = required_columns - set(cleaned.columns)

    if missing_columns:
        raise KeyError(
            f"Required cleaning columns are missing: "
            f"{sorted(missing_columns)}"
        )

    for column, sentinel in NUMERIC_SENTINELS.items():
        numeric_values = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

        cleaned[f"{column}_CLEAN"] = numeric_values.mask(
            numeric_values.eq(sentinel)
        )

    for column, sentinel in CATEGORICAL_SENTINELS.items():
        categorical_values = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

        cleaned[f"{column}_CLEAN"] = categorical_values.mask(
            categorical_values.eq(sentinel)
        )

    postal_code = (
        cleaned["POSTAL CODE"]
        .astype("string")
        .str.strip()
    )

    cleaned["POSTAL CODE_CLEAN"] = postal_code.str.zfill(
        POSTAL_CODE_WIDTH
    )

    return cleaned