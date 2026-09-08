"""End-to-end Freddie Mac origination processing pipeline."""

import json
from pathlib import Path

from .freddie_cleaning import clean_origination
from .freddie_cohorts import (
    add_origination_cohorts,
    create_origination_cohort_summary,
    validate_origination_cohorts,
)
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
    """
    Load, clean, create cohorts, validate, summarize, and save
    one annual sample.
    """

    input_file = Path(input_file)
    output_directory = Path(output_directory)

    raw = load_origination(input_file)

    structure_report = raw.attrs.get(
        "structure_validation",
        {},
    )

    cleaned = clean_origination(raw)

    cleaned = add_origination_cohorts(
        origination=cleaned,
        expected_year=year,
    )

    validation_report = validate_origination(
        raw=raw,
        cleaned=cleaned,
        expected_rows=expected_rows,
    )

    cohort_validation_report = validate_origination_cohorts(
    origination=cleaned,
    expected_year=year,
)

    quality_reports = create_origination_quality_report(
        data=raw,
        year=year,
    )

    summaries = create_annual_summary(
        data=cleaned,
        year=year,
    )

    cohort_summary = create_origination_cohort_summary(
        origination=cleaned,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = {
        "cleaned": (
            output_directory
            / f"origination_{year}_clean.parquet"
        ),
        "numeric_summary": (
            output_directory
            / f"origination_{year}_numeric_summary.csv"
        ),
        "categorical_summary": (
            output_directory
            / f"origination_{year}_categorical_summary.csv"
        ),
        "cohort_summary": (
            output_directory
            / f"origination_{year}_cohort_summary.csv"
        ),
        "unexpected_codes": (
            output_directory
            / f"origination_{year}_unexpected_codes.csv"
        ),
        "sentinel_counts": (
            output_directory
            / f"origination_{year}_sentinel_counts.csv"
        ),
        "missingness": (
            output_directory
            / f"origination_{year}_missingness.csv"
        ),
        "numeric_range_warnings": (
            output_directory
            / f"origination_{year}_numeric_range_warnings.csv"
        ),
        "definition_warnings": (
            output_directory
            / f"origination_{year}_definition_warnings.csv"
        ),
        "ratio_warnings": (
            output_directory
            / f"origination_{year}_ratio_warnings.csv"
        ),
        "validation": (
            output_directory
            / f"origination_{year}_validation.json"
        ),
    }

    cleaned.to_parquet(
        output_paths["cleaned"],
        index=False,
        engine="pyarrow",
    )

    summaries["numeric"].to_csv(
        output_paths["numeric_summary"],
        index=False,
    )

    summaries["categorical"].to_csv(
        output_paths["categorical_summary"],
        index=False,
    )

    cohort_summary.to_csv(
        output_paths["cohort_summary"],
        index=False,
    )

    for report_name in [
        "unexpected_codes",
        "sentinel_counts",
        "missingness",
        "numeric_range_warnings",
        "definition_warnings",
        "ratio_warnings",
    ]:
        quality_reports[report_name].to_csv(
            output_paths[report_name],
            index=False,
        )

    with output_paths["validation"].open(
        mode="w",
        encoding="utf-8",
    ) as validation_file:
        json.dump(
            {
                "year": year,
                "structure_validation": structure_report,
                "data_validation": validation_report,
                "cohort_validation": cohort_validation_report,
                "cohort_summary_rows": len(
                    cohort_summary
                ),
                "cohort_summary_records": int(
                    cohort_summary["count"].sum()
                ),
                "unexpected_code_rows": len(
                    quality_reports["unexpected_codes"]
                ),
                "numeric_range_warning_rows": len(
                    quality_reports[
                        "numeric_range_warnings"
                    ]
                ),
                "definition_warning_rows": len(
                    quality_reports[
                        "definition_warnings"
                    ]
                ),
                "ratio_warning_rows": len(
                    quality_reports["ratio_warnings"]
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
        "cohort_validation": cohort_validation_report,
        "numeric_summary": summaries["numeric"],
        "categorical_summary": summaries["categorical"],
        "cohort_summary": cohort_summary,
        **quality_reports,
        "output_paths": output_paths,
    }