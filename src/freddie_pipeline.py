"""End-to-end Freddie Mac origination processing pipeline."""

from pathlib import Path

import pandas as pd

from .freddie_cleaning import clean_origination
from .freddie_ingestion import load_origination
from .freddie_summary import create_annual_summary
from .freddie_validation import validate_origination


def run_origination_pipeline(
    year: int,
    input_file: str | Path,
    output_directory: str | Path,
    expected_rows: int | None = None,
) -> dict[str, object]:
    """Load, clean, validate, summarize, and save one annual sample."""

    input_file = Path(input_file)
    output_directory = Path(output_directory)

    raw = load_origination(input_file)

    cleaned = clean_origination(raw)

    validation_report = validate_origination(
        raw=raw,
        cleaned=cleaned,
        expected_rows=expected_rows,
    )

    summaries = create_annual_summary(
        data=cleaned,
        year=year,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_path = (
        output_directory
        / f"origination_{year}_clean.csv"
    )

    numeric_summary_path = (
        output_directory
        / f"origination_{year}_numeric_summary.csv"
    )

    categorical_summary_path = (
        output_directory
        / f"origination_{year}_categorical_summary.csv"
    )

    cleaned.to_csv(
        cleaned_path,
        index=False,
    )

    summaries["numeric"].to_csv(
        numeric_summary_path,
        index=False,
    )

    summaries["categorical"].to_csv(
        categorical_summary_path,
        index=False,
    )

    return {
        "year": year,
        "raw": raw,
        "cleaned": cleaned,
        "validation": validation_report,
        "numeric_summary": summaries["numeric"],
        "categorical_summary": summaries["categorical"],
        "output_paths": {
            "cleaned": cleaned_path,
            "numeric_summary": numeric_summary_path,
            "categorical_summary": categorical_summary_path,
        },
    }