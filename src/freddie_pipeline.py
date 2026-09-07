"""End-to-end Freddie Mac origination processing pipeline."""

import json
from pathlib import Path

from .freddie_cleaning import clean_origination
from .freddie_ingestion import load_origination
from .freddie_quality import create_origination_quality_report
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

    structure_report = raw.attrs.get(
        "structure_validation",
        {},
    )

    cleaned = clean_origination(raw)

    validation_report = validate_origination(
        raw=raw,
        cleaned=cleaned,
        expected_rows=expected_rows,
    )

    quality_reports = create_origination_quality_report(
        data=raw,
        year=year,
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
        / f"origination_{year}_clean.parquet"
    )

    numeric_summary_path = (
        output_directory
        / f"origination_{year}_numeric_summary.csv"
    )

    categorical_summary_path = (
        output_directory
        / f"origination_{year}_categorical_summary.csv"
    )

    unexpected_codes_path = (
        output_directory
        / f"origination_{year}_unexpected_codes.csv"
    )

    sentinel_counts_path = (
        output_directory
        / f"origination_{year}_sentinel_counts.csv"
    )

    missingness_path = (
        output_directory
        / f"origination_{year}_missingness.csv"
    )

    high_ratio_warnings_path = (
        output_directory
        / f"origination_{year}_high_ratio_warnings.csv"
    )

    validation_path = (
        output_directory
        / f"origination_{year}_validation.json"
    )

    cleaned.to_parquet(
        cleaned_path,
        index=False,
        engine="pyarrow",
    )

    summaries["numeric"].to_csv(
        numeric_summary_path,
        index=False,
    )

    summaries["categorical"].to_csv(
        categorical_summary_path,
        index=False,
    )

    quality_reports["unexpected_codes"].to_csv(
        unexpected_codes_path,
        index=False,
    )

    quality_reports["sentinel_counts"].to_csv(
        sentinel_counts_path,
        index=False,
    )

    quality_reports["missingness"].to_csv(
        missingness_path,
        index=False,
    )

    quality_reports["high_ratio_warnings"].to_csv(
        high_ratio_warnings_path,
        index=False,
    )

    with validation_path.open(
        mode="w",
        encoding="utf-8",
    ) as validation_file:
        json.dump(
            {
                "year": year,
                "structure_validation": structure_report,
                "data_validation": validation_report,
                "unexpected_code_rows": len(
                    quality_reports["unexpected_codes"]
                ),
                "high_ratio_warning_rows": len(
                    quality_reports["high_ratio_warnings"]
                ),
            },
            validation_file,
            indent=2,
        )

    return {
        "year": year,
        "raw": raw,
        "cleaned": cleaned,
        "structure_validation": structure_report,
        "validation": validation_report,
        "numeric_summary": summaries["numeric"],
        "categorical_summary": summaries["categorical"],
        "unexpected_codes": (
            quality_reports["unexpected_codes"]
        ),
        "sentinel_counts": (
            quality_reports["sentinel_counts"]
        ),
        "missingness": quality_reports["missingness"],
        "high_ratio_warnings": (
            quality_reports["high_ratio_warnings"]
        ),
        "output_paths": {
            "cleaned": cleaned_path,
            "numeric_summary": numeric_summary_path,
            "categorical_summary": categorical_summary_path,
            "unexpected_codes": unexpected_codes_path,
            "sentinel_counts": sentinel_counts_path,
            "missingness": missingness_path,
            "high_ratio_warnings": high_ratio_warnings_path,
            "validation": validation_path,
        },
    }