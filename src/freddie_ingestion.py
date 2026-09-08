"""Functions for loading Freddie Mac annual sample files."""

from pathlib import Path

import pandas as pd

from .freddie_config import (
    ORIGINATION_COLUMNS,
    ORIGINATION_MAX_LENGTHS,
    ORIGINATION_STRING_COLUMNS,
)


def validate_raw_origination_structure(
    file_path: str | Path,
    max_reported_errors: int = 10,
) -> dict[str, int]:
    """Validate raw field counts and maximum lengths before parsing."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Origination file not found: {file_path}"
        )

    expected_field_count = len(ORIGINATION_COLUMNS)

    rows_scanned = 0
    field_count_errors = 0
    field_length_errors = 0
    error_examples = []

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
    ) as raw_file:
        for line_number, line in enumerate(raw_file, start=1):
            rows_scanned += 1

            fields = line.rstrip("\r\n").split("|")

            if len(fields) != expected_field_count:
                field_count_errors += 1

                if len(error_examples) < max_reported_errors:
                    error_examples.append(
                        f"Line {line_number}: expected "
                        f"{expected_field_count} fields but found "
                        f"{len(fields)}."
                    )

                continue

            for position, column in enumerate(
                ORIGINATION_COLUMNS
            ):
                value = fields[position]
                maximum_length = ORIGINATION_MAX_LENGTHS[column]

                if len(value) > maximum_length:
                    field_length_errors += 1

                    if len(error_examples) < max_reported_errors:
                        error_examples.append(
                            f"Line {line_number}, field "
                            f"{position + 1} ({column}): length "
                            f"{len(value)} exceeds maximum "
                            f"{maximum_length}."
                        )

    if rows_scanned == 0:
        raise ValueError(
            f"Origination file is empty: {file_path}"
        )

    if field_count_errors > 0 or field_length_errors > 0:
        error_message = [
            "Raw origination structure validation failed.",
            f"Field-count errors: {field_count_errors}",
            f"Field-length errors: {field_length_errors}",
        ]

        if error_examples:
            error_message.append("Examples:")
            error_message.extend(error_examples)

        raise ValueError("\n".join(error_message))

    return {
        "rows_scanned": rows_scanned,
        "expected_fields_per_row": expected_field_count,
        "field_count_errors": field_count_errors,
        "field_length_errors": field_length_errors,
    }


def load_origination(
    file_path: str | Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Validate and load a Freddie Mac origination file."""

    file_path = Path(file_path)

    structure_report = validate_raw_origination_structure(
        file_path
    )

    origination = pd.read_csv(
        file_path,
        sep="|",
        header=None,
        names=ORIGINATION_COLUMNS,
        dtype=ORIGINATION_STRING_COLUMNS,
        nrows=nrows,
        low_memory=False,
        encoding="utf-8-sig",
    )

    if origination.shape[1] != len(ORIGINATION_COLUMNS):
        raise ValueError(
            "Unexpected number of parsed origination columns: "
            f"{origination.shape[1]}"
        )

    origination.attrs["structure_validation"] = (
        structure_report
    )

    return origination