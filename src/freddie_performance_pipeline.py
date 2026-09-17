"""End-to-end Freddie Mac performance-target processing pipeline."""

import gc
import json
from pathlib import Path

import pandas as pd

from .freddie_performance import (
    TARGET_PERFORMANCE_COLUMNS,
    create_horizon_targets,
    load_performance,
    prepare_performance_timeline,
    summarize_horizon_targets,
    validate_performance_for_target,
)


FINAL_TARGET_COLUMNS = [
    "SOURCE YEAR",
    "LOAN IDENTIFIER",
    "ORIGINATION COHORT",
    "HORIZON_MONTHS",
    "TARGET_90_PLUS",
    "TARGET_COMPOSITE_CREDIT_EVENT",
    "FIRST_SERIOUS_DELINQUENCY_PERIOD",
    "FIRST_TERMINAL_CREDIT_EVENT_PERIOD",
    "FIRST_OBSERVED_MONTH",
    "LAST_OBSERVED_MONTH",
    "OBSERVED_MONTHS",
]


def _component_summary(
    selected_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mutually exclusive components of the selected target."""

    serious_event = selected_targets[
        "SERIOUS_DELINQUENCY_WITHIN_HORIZON"
    ]
    terminal_event = selected_targets[
        "TERMINAL_CREDIT_EVENT_WITHIN_HORIZON"
    ]

    component_masks = {
        "90+ delinquency only": (
            serious_event & ~terminal_event
        ),
        "Terminal credit event only": (
            ~serious_event & terminal_event
        ),
        "Both components": serious_event & terminal_event,
        "Neither component": ~serious_event & ~terminal_event,
    }

    rows = []

    for component, mask in component_masks.items():
        loans = int(mask.sum())
        rows.append(
            {
                "component": component,
                "loans": loans,
                "percentage": (
                    loans / len(selected_targets) * 100
                    if len(selected_targets)
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def _cohort_summary(
    selected_targets: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Summarize the selected target by origination cohort."""

    summary = (
        selected_targets.groupby(
            "ORIGINATION COHORT",
            observed=True,
        )
        .agg(
            eligible_loans=("LOAN IDENTIFIER", "size"),
            serious_delinquency_events=(
                "TARGET_90_PLUS",
                "sum",
            ),
            serious_delinquency_rate=(
                "TARGET_90_PLUS",
                "mean",
            ),
            composite_credit_events=(
                "TARGET_COMPOSITE_CREDIT_EVENT",
                "sum",
            ),
            composite_credit_event_rate=(
                "TARGET_COMPOSITE_CREDIT_EVENT",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["serious_delinquency_rate_percentage"] = (
        summary["serious_delinquency_rate"] * 100
    )
    summary[
        "composite_credit_event_rate_percentage"
    ] = summary["composite_credit_event_rate"] * 100

    expected_cohorts = pd.DataFrame(
        {
            "ORIGINATION COHORT": [
                f"{year}Q{quarter}"
                for quarter in range(1, 5)
            ]
        }
    )
    summary = expected_cohorts.merge(
        summary,
        on="ORIGINATION COHORT",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "eligible_loans",
        "serious_delinquency_events",
        "composite_credit_events",
    ]
    summary[count_columns] = (
        summary[count_columns]
        .fillna(0)
        .astype("int64")
    )

    return summary


def _validate_annual_readback(
    output_paths: dict[str, Path],
    horizons: tuple[int, ...],
    selected_horizon: int,
    expected_loans: int,
) -> dict[str, object]:
    """Read annual outputs back and validate their saved contents."""

    candidates = pd.read_parquet(output_paths["candidates"])
    selected = pd.read_parquet(output_paths["selected_target"])
    horizon_summary = pd.read_csv(
        output_paths["horizon_summary"]
    )
    cohort_summary = pd.read_csv(
        output_paths["cohort_summary"]
    )
    component_summary = pd.read_csv(
        output_paths["component_summary"]
    )

    duplicate_candidate_rows = int(
        candidates.duplicated(
            subset=["LOAN IDENTIFIER", "HORIZON_MONTHS"]
        ).sum()
    )
    unexpected_horizons = sorted(
        set(candidates["HORIZON_MONTHS"])
        - set(horizons)
    )
    duplicate_selected_loans = int(
        selected["LOAN IDENTIFIER"].duplicated().sum()
    )
    unexpected_target_values = sorted(
        set(
            selected[
                "TARGET_COMPOSITE_CREDIT_EVENT"
            ].dropna()
        )
        - {0, 1}
    )

    report = {
        "candidate_rows": len(candidates),
        "expected_candidate_rows": (
            expected_loans * len(horizons)
        ),
        "candidate_duplicate_loan_horizons": (
            duplicate_candidate_rows
        ),
        "unexpected_saved_horizons": unexpected_horizons,
        "selected_target_rows": len(selected),
        "selected_target_duplicate_loans": (
            duplicate_selected_loans
        ),
        "unexpected_target_values": unexpected_target_values,
        "selected_horizon_values": sorted(
            selected["HORIZON_MONTHS"].unique().tolist()
        ),
        "saved_horizon_summary_rows": len(horizon_summary),
        "saved_cohort_summary_rows": len(cohort_summary),
        "saved_component_summary_rows": len(component_summary),
    }

    failed_checks = {}

    if report["candidate_rows"] != report[
        "expected_candidate_rows"
    ]:
        failed_checks["candidate_rows"] = report[
            "candidate_rows"
        ]
    if duplicate_candidate_rows:
        failed_checks["candidate_duplicates"] = (
            duplicate_candidate_rows
        )
    if unexpected_horizons:
        failed_checks["unexpected_horizons"] = (
            unexpected_horizons
        )
    if duplicate_selected_loans:
        failed_checks["selected_duplicates"] = (
            duplicate_selected_loans
        )
    if unexpected_target_values:
        failed_checks["target_values"] = (
            unexpected_target_values
        )
    if report["selected_horizon_values"] != [selected_horizon]:
        failed_checks["selected_horizon_values"] = report[
            "selected_horizon_values"
        ]
    if len(horizon_summary) != len(horizons):
        failed_checks["horizon_summary_rows"] = len(
            horizon_summary
        )
    if len(component_summary) != 4:
        failed_checks["component_summary_rows"] = len(
            component_summary
        )
    if int(component_summary["loans"].sum()) != len(selected):
        failed_checks["component_total"] = int(
            component_summary["loans"].sum()
        )
    if int(cohort_summary["eligible_loans"].sum()) != len(
        selected
    ):
        failed_checks["cohort_total"] = int(
            cohort_summary["eligible_loans"].sum()
        )

    if failed_checks:
        raise ValueError(
            "Saved performance-target validation failed: "
            f"{failed_checks}"
        )

    return report


def run_performance_target_pipeline(
    year: int,
    performance_file: str | Path,
    origination_file: str | Path,
    output_directory: str | Path,
    horizons: tuple[int, ...] = (12, 24, 36),
    selected_horizon: int = 24,
    minimum_observable_coverage_percentage: float = 95.0,
    expected_loans: int = 50000,
) -> dict[str, object]:
    """Build, validate, save, and verify one annual target dataset."""

    performance_file = Path(performance_file)
    origination_file = Path(origination_file)
    output_directory = Path(output_directory)

    if selected_horizon not in horizons:
        raise ValueError(
            "The selected horizon must appear in candidate horizons."
        )
    if not origination_file.exists():
        raise FileNotFoundError(
            f"Origination file not found: {origination_file}"
        )

    performance = load_performance(
        file_path=performance_file,
        usecols=TARGET_PERFORMANCE_COLUMNS,
    )
    structure_report = performance.attrs.get(
        "structure_validation",
        {},
    )
    origination = pd.read_parquet(
        origination_file,
        columns=[
            "LOAN IDENTIFIER",
            "FIRST PAYMENT DATE",
            "ORIGINATION COHORT",
        ],
    )

    if len(origination) != expected_loans:
        raise ValueError(
            f"Expected {expected_loans} origination loans for {year} "
            f"but found {len(origination)}."
        )

    validation_report = validate_performance_for_target(
        performance=performance,
        origination=origination,
    )
    timeline = prepare_performance_timeline(
        performance=performance,
        origination=origination,
    )
    del performance
    gc.collect()

    targets = create_horizon_targets(
        timeline=timeline,
        origination=origination,
        horizons=horizons,
    )
    del timeline
    gc.collect()

    targets.insert(0, "SOURCE YEAR", year)
    horizon_summary = summarize_horizon_targets(targets)
    horizon_summary.insert(0, "year", year)

    selected_summary = horizon_summary.loc[
        horizon_summary["horizon_months"].eq(
            selected_horizon
        )
    ].iloc[0]
    observable_coverage = float(
        selected_summary[
            "observable_cohort_coverage_percentage"
        ]
    )

    if (
        observable_coverage
        < minimum_observable_coverage_percentage
    ):
        raise ValueError(
            f"{year} {selected_horizon}-month observable-cohort "
            f"coverage is {observable_coverage:.3f}%, below "
            f"{minimum_observable_coverage_percentage:.3f}%."
        )

    selected_candidates = targets.loc[
        targets["HORIZON_MONTHS"].eq(selected_horizon)
        & targets["TARGET_ELIGIBLE"]
    ].copy()
    selected_target = (
        selected_candidates[FINAL_TARGET_COLUMNS]
        .sort_values("LOAN IDENTIFIER")
        .reset_index(drop=True)
    )
    component_summary = _component_summary(selected_candidates)
    component_summary.insert(0, "year", year)
    cohort_summary = _cohort_summary(
        selected_targets=selected_candidates,
        year=year,
    )
    cohort_summary.insert(0, "year", year)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "candidates": output_directory
        / f"performance_target_candidates_{year}.parquet",
        "selected_target": output_directory
        / f"performance_target_{year}_{selected_horizon}m.parquet",
        "horizon_summary": output_directory
        / f"performance_target_{year}_horizon_summary.csv",
        "cohort_summary": output_directory
        / f"performance_target_{year}_cohort_summary.csv",
        "component_summary": output_directory
        / f"performance_target_{year}_component_summary.csv",
        "validation": output_directory
        / f"performance_target_{year}_validation.json",
    }

    targets.to_parquet(
        output_paths["candidates"],
        index=False,
        engine="pyarrow",
    )
    selected_target.to_parquet(
        output_paths["selected_target"],
        index=False,
        engine="pyarrow",
    )
    horizon_summary.to_csv(
        output_paths["horizon_summary"],
        index=False,
    )
    cohort_summary.to_csv(
        output_paths["cohort_summary"],
        index=False,
    )
    component_summary.to_csv(
        output_paths["component_summary"],
        index=False,
    )

    readback_report = _validate_annual_readback(
        output_paths=output_paths,
        horizons=horizons,
        selected_horizon=selected_horizon,
        expected_loans=expected_loans,
    )

    validation_payload = {
        "year": year,
        "performance_file": str(performance_file),
        "origination_file": str(origination_file),
        "candidate_horizons": list(horizons),
        "selected_horizon_months": selected_horizon,
        "minimum_observable_coverage_percentage": (
            minimum_observable_coverage_percentage
        ),
        "selected_sample_retention_percentage": float(
            selected_summary["sample_retention_percentage"]
        ),
        "selected_observable_cohort_coverage_percentage": (
            observable_coverage
        ),
        "target_definition": (
            "Numeric delinquency status >= 3 or RA, or Zero "
            "Balance Code in 02, 03, 09, 15, within the "
            f"first {selected_horizon} months from FIRST PAYMENT DATE."
        ),
        "coverage_definition": (
            "Eligible loans divided by loans observable from month 0; "
            "overall sample retention is reported separately."
        ),
        "structure_validation": structure_report,
        "data_validation": validation_report,
        "eligible_loans": len(selected_target),
        "serious_delinquency_events": int(
            selected_target["TARGET_90_PLUS"].sum()
        ),
        "composite_credit_events": int(
            selected_target[
                "TARGET_COMPOSITE_CREDIT_EVENT"
            ].sum()
        ),
        "readback_validation": readback_report,
    }

    with output_paths["validation"].open(
        mode="w",
        encoding="utf-8",
    ) as validation_file:
        json.dump(
            validation_payload,
            validation_file,
            indent=2,
        )

    with output_paths["validation"].open(
        mode="r",
        encoding="utf-8",
    ) as validation_file:
        saved_validation = json.load(validation_file)

    validation_json_matches = (
        saved_validation["year"] == year
        and saved_validation["eligible_loans"]
        == len(selected_target)
        and abs(
            saved_validation[
                "selected_observable_cohort_coverage_percentage"
            ]
            - observable_coverage
        )
        < 1e-12
    )
    if not validation_json_matches:
        raise ValueError(
            f"Saved validation JSON does not match {year} results."
        )
    readback_report["validation_json_matches"] = True

    return {
        "year": year,
        "structure_validation": structure_report,
        "validation": validation_report,
        "horizon_summary": horizon_summary,
        "cohort_summary": cohort_summary,
        "component_summary": component_summary,
        "selected_summary": selected_summary.to_dict(),
        "readback_validation": readback_report,
        "output_paths": output_paths,
    }


def combine_performance_targets(
    years: list[int] | tuple[int, ...],
    output_directory: str | Path,
    selected_horizon: int = 24,
) -> dict[str, object]:
    """Combine and verify annual selected targets and summaries."""

    output_directory = Path(output_directory)
    annual_targets = []
    annual_summaries = []
    cohort_summaries = []

    for year in years:
        target_path = (
            output_directory
            / f"performance_target_{year}_{selected_horizon}m.parquet"
        )
        summary_path = (
            output_directory
            / f"performance_target_{year}_horizon_summary.csv"
        )
        cohort_path = (
            output_directory
            / f"performance_target_{year}_cohort_summary.csv"
        )

        for path in [target_path, summary_path, cohort_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required annual output not found: {path}"
                )

        annual_target = pd.read_parquet(target_path)
        if "SOURCE YEAR" not in annual_target.columns:
            annual_target.insert(0, "SOURCE YEAR", year)
        if not annual_target["SOURCE YEAR"].eq(year).all():
            raise ValueError(
                f"Source-year mismatch in {target_path}."
            )

        annual_targets.append(annual_target)
        annual_summaries.append(
            pd.read_csv(summary_path).loc[
                lambda data: data["horizon_months"].eq(
                    selected_horizon
                )
            ]
        )
        cohort_summaries.append(pd.read_csv(cohort_path))

    combined_target = pd.concat(
        annual_targets,
        ignore_index=True,
    )
    annual_summary = pd.concat(
        annual_summaries,
        ignore_index=True,
    )
    cohort_summary = pd.concat(
        cohort_summaries,
        ignore_index=True,
    )

    duplicate_year_loans = int(
        combined_target.duplicated(
            subset=["SOURCE YEAR", "LOAN IDENTIFIER"]
        ).sum()
    )
    unexpected_years = sorted(
        set(combined_target["SOURCE YEAR"]) - set(years)
    )
    unexpected_horizons = sorted(
        set(combined_target["HORIZON_MONTHS"])
        - {selected_horizon}
    )
    unexpected_target_values = sorted(
        set(
            combined_target[
                "TARGET_COMPOSITE_CREDIT_EVENT"
            ].dropna()
        )
        - {0, 1}
    )

    if (
        duplicate_year_loans
        or unexpected_years
        or unexpected_horizons
        or unexpected_target_values
    ):
        raise ValueError(
            "Combined target validation failed: "
            f"duplicates={duplicate_year_loans}, "
            f"years={unexpected_years}, "
            f"horizons={unexpected_horizons}, "
            f"target_values={unexpected_target_values}."
        )

    year_span = f"{min(years)}_{max(years)}"
    output_paths = {
        "combined_target": output_directory
        / f"performance_target_{year_span}_{selected_horizon}m.parquet",
        "annual_summary": output_directory
        / f"performance_target_{year_span}_annual_summary.csv",
        "quarterly_summary": output_directory
        / f"performance_target_{year_span}_quarterly_summary.csv",
        "validation": output_directory
        / f"performance_target_{year_span}_validation.json",
    }

    combined_target.to_parquet(
        output_paths["combined_target"],
        index=False,
        engine="pyarrow",
    )
    annual_summary.to_csv(
        output_paths["annual_summary"],
        index=False,
    )
    cohort_summary.to_csv(
        output_paths["quarterly_summary"],
        index=False,
    )

    saved_target = pd.read_parquet(output_paths["combined_target"])
    saved_annual_summary = pd.read_csv(
        output_paths["annual_summary"]
    )
    saved_quarterly_summary = pd.read_csv(
        output_paths["quarterly_summary"]
    )

    readback_report = {
        "combined_target_rows": len(saved_target),
        "combined_target_columns": saved_target.shape[1],
        "duplicate_year_loan_rows": int(
            saved_target.duplicated(
                subset=["SOURCE YEAR", "LOAN IDENTIFIER"]
            ).sum()
        ),
        "saved_years": sorted(
            saved_target["SOURCE YEAR"].unique().tolist()
        ),
        "saved_horizons": sorted(
            saved_target["HORIZON_MONTHS"].unique().tolist()
        ),
        "annual_summary_rows": len(saved_annual_summary),
        "quarterly_summary_rows": len(saved_quarterly_summary),
    }

    expected_years = sorted(years)
    if (
        readback_report["duplicate_year_loan_rows"]
        or readback_report["saved_years"] != expected_years
        or readback_report["saved_horizons"] != [selected_horizon]
        or readback_report["annual_summary_rows"] != len(years)
        or readback_report["quarterly_summary_rows"]
        != len(years) * 4
    ):
        raise ValueError(
            "Combined saved-output validation failed: "
            f"{readback_report}"
        )

    validation_payload = {
        "years": list(years),
        "selected_horizon_months": selected_horizon,
        "combined_target_rows": len(combined_target),
        "composite_credit_events": int(
            combined_target[
                "TARGET_COMPOSITE_CREDIT_EVENT"
            ].sum()
        ),
        "readback_validation": readback_report,
    }

    with output_paths["validation"].open(
        mode="w",
        encoding="utf-8",
    ) as validation_file:
        json.dump(
            validation_payload,
            validation_file,
            indent=2,
        )

    with output_paths["validation"].open(
        mode="r",
        encoding="utf-8",
    ) as validation_file:
        saved_validation = json.load(validation_file)

    validation_json_matches = (
        saved_validation["years"] == list(years)
        and saved_validation["combined_target_rows"]
        == len(combined_target)
        and saved_validation["composite_credit_events"]
        == int(
            combined_target[
                "TARGET_COMPOSITE_CREDIT_EVENT"
            ].sum()
        )
    )
    if not validation_json_matches:
        raise ValueError(
            "Saved combined validation JSON does not match results."
        )
    readback_report["validation_json_matches"] = True

    return {
        "combined_target": combined_target,
        "annual_summary": annual_summary,
        "quarterly_summary": cohort_summary,
        "readback_validation": readback_report,
        "output_paths": output_paths,
    }
