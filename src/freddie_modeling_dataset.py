"""Build and validate the Freddie Mac loan-level modeling dataset."""

import json
from pathlib import Path

import pandas as pd


KEY_COLUMNS = [
    "SOURCE YEAR",
    "LOAN IDENTIFIER",
]

COHORT_COLUMNS = [
    "ORIGINATION YEAR",
    "ORIGINATION QUARTER",
    "ORIGINATION COHORT",
]

TARGET_AUDIT_COLUMNS = [
    "HORIZON_MONTHS",
    "TARGET_90_PLUS",
    "TARGET_COMPOSITE_CREDIT_EVENT",
    "FIRST_SERIOUS_DELINQUENCY_PERIOD",
    "FIRST_TERMINAL_CREDIT_EVENT_PERIOD",
    "FIRST_OBSERVED_MONTH",
    "LAST_OBSERVED_MONTH",
    "OBSERVED_MONTHS",
]

LABEL_COLUMNS = [
    "TARGET_90_PLUS",
    "TARGET_COMPOSITE_CREDIT_EVENT",
]


def load_origination_cohorts(
    years: list[int] | tuple[int, ...],
    input_directory: str | Path,
) -> pd.DataFrame:
    """Load and combine annual cleaned origination cohorts."""

    input_directory = Path(input_directory)
    annual_origination = []

    for year in years:
        input_path = (
            input_directory
            / f"origination_{year}_clean.parquet"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Cleaned origination file not found: {input_path}"
            )

        origination = pd.read_parquet(input_path)

        required_columns = {
            "LOAN IDENTIFIER",
            *COHORT_COLUMNS,
        }
        missing_columns = sorted(
            required_columns - set(origination.columns)
        )

        if missing_columns:
            raise KeyError(
                f"{year} origination data is missing columns: "
                f"{missing_columns}"
            )

        source_year = pd.to_numeric(
            origination["ORIGINATION YEAR"],
            errors="coerce",
        )

        if not source_year.eq(year).all():
            raise ValueError(
                f"{year} origination file contains records from "
                "another origination year."
            )

        annual_data = origination.copy()
        annual_data.insert(0, "SOURCE YEAR", year)
        annual_data["LOAN IDENTIFIER"] = annual_data[
            "LOAN IDENTIFIER"
        ].astype("string")
        annual_origination.append(annual_data)

    combined = pd.concat(
        annual_origination,
        ignore_index=True,
    )

    duplicate_keys = int(
        combined.duplicated(subset=KEY_COLUMNS).sum()
    )
    missing_keys = int(
        combined[KEY_COLUMNS].isna().any(axis=1).sum()
    )

    if duplicate_keys or missing_keys:
        raise ValueError(
            "Combined origination-key validation failed: "
            f"duplicates={duplicate_keys}, missing={missing_keys}."
        )

    return combined


def _validate_target_contract(
    targets: pd.DataFrame,
    years: list[int] | tuple[int, ...],
    selected_horizon: int,
) -> None:
    """Validate the combined selected-target input before merging."""

    required_columns = {
        *KEY_COLUMNS,
        "ORIGINATION COHORT",
        *TARGET_AUDIT_COLUMNS,
    }
    missing_columns = sorted(
        required_columns - set(targets.columns)
    )

    if missing_columns:
        raise KeyError(
            "Combined target data is missing columns: "
            f"{missing_columns}"
        )

    duplicate_keys = int(
        targets.duplicated(subset=KEY_COLUMNS).sum()
    )
    missing_keys = int(
        targets[KEY_COLUMNS].isna().any(axis=1).sum()
    )
    observed_years = sorted(
        pd.to_numeric(
            targets["SOURCE YEAR"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    expected_years = sorted(years)
    observed_horizons = sorted(
        pd.to_numeric(
            targets["HORIZON_MONTHS"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    errors = []

    if duplicate_keys:
        errors.append(
            f"Found {duplicate_keys} duplicate target keys."
        )
    if missing_keys:
        errors.append(
            f"Found {missing_keys} missing target keys."
        )
    if observed_years != expected_years:
        errors.append(
            "Target years do not match the requested years: "
            f"observed={observed_years}, expected={expected_years}."
        )
    if observed_horizons != [selected_horizon]:
        errors.append(
            "Target horizons do not match the selected horizon: "
            f"observed={observed_horizons}, "
            f"expected={[selected_horizon]}."
        )

    for column in LABEL_COLUMNS:
        missing_labels = int(targets[column].isna().sum())
        unexpected_values = sorted(
            set(targets[column].dropna().astype(int)) - {0, 1}
        )

        if missing_labels:
            errors.append(
                f"Found {missing_labels} missing values in {column}."
            )
        if unexpected_values:
            errors.append(
                f"Unexpected values in {column}: {unexpected_values}."
            )

    serious_target = targets["TARGET_90_PLUS"].astype(int)
    composite_target = targets[
        "TARGET_COMPOSITE_CREDIT_EVENT"
    ].astype(int)
    serious_period_present = targets[
        "FIRST_SERIOUS_DELINQUENCY_PERIOD"
    ].notna()
    terminal_period_present = targets[
        "FIRST_TERMINAL_CREDIT_EVENT_PERIOD"
    ].notna()

    serious_period_mismatches = int(
        serious_period_present.ne(serious_target.eq(1)).sum()
    )
    composite_component_mismatches = int(
        (composite_target.eq(1)).ne(
            serious_period_present | terminal_period_present
        ).sum()
    )
    target_order_violations = int(
        serious_target.gt(composite_target).sum()
    )
    nonpositive_observation_counts = int(
        pd.to_numeric(
            targets["OBSERVED_MONTHS"],
            errors="coerce",
        )
        .le(0)
        .sum()
    )

    if serious_period_mismatches:
        errors.append(
            "90+ target and first serious-delinquency period disagree "
            f"for {serious_period_mismatches} loans."
        )
    if composite_component_mismatches:
        errors.append(
            "Composite target and event-component periods disagree "
            f"for {composite_component_mismatches} loans."
        )
    if target_order_violations:
        errors.append(
            "TARGET_90_PLUS exceeds the composite target for "
            f"{target_order_violations} loans."
        )
    if nonpositive_observation_counts:
        errors.append(
            "Found nonpositive observation counts for "
            f"{nonpositive_observation_counts} loans."
        )

    if errors:
        raise ValueError(
            "Target-contract validation failed:\n- "
            + "\n- ".join(errors)
        )


def build_modeling_dataset(
    origination: pd.DataFrame,
    targets: pd.DataFrame,
    years: list[int] | tuple[int, ...],
    selected_horizon: int = 24,
) -> pd.DataFrame:
    """Merge eligible targets onto origination-time loan records."""

    _validate_target_contract(
        targets=targets,
        years=years,
        selected_horizon=selected_horizon,
    )

    required_origination_columns = {
        *KEY_COLUMNS,
        *COHORT_COLUMNS,
    }
    missing_origination_columns = sorted(
        required_origination_columns - set(origination.columns)
    )

    if missing_origination_columns:
        raise KeyError(
            "Combined origination data is missing columns: "
            f"{missing_origination_columns}"
        )

    origination_data = origination.copy()
    target_data = targets.copy()

    for data in [origination_data, target_data]:
        data["LOAN IDENTIFIER"] = data[
            "LOAN IDENTIFIER"
        ].astype("string")
        data["SOURCE YEAR"] = pd.to_numeric(
            data["SOURCE YEAR"],
            errors="raise",
        ).astype(int)

    origination_duplicate_keys = int(
        origination_data.duplicated(
            subset=KEY_COLUMNS
        ).sum()
    )
    if origination_duplicate_keys:
        raise ValueError(
            "Origination data contains "
            f"{origination_duplicate_keys} duplicate loan/year keys."
        )

    reconciliation = target_data[
        [*KEY_COLUMNS, "ORIGINATION COHORT"]
    ].merge(
        origination_data[
            [*KEY_COLUMNS, "ORIGINATION COHORT"]
        ],
        on=KEY_COLUMNS,
        how="left",
        suffixes=("_TARGET", "_ORIGINATION"),
        validate="one_to_one",
        indicator=True,
    )

    unmatched_targets = int(
        reconciliation["_merge"].ne("both").sum()
    )
    cohort_mismatches = int(
        reconciliation[
            "ORIGINATION COHORT_TARGET"
        ]
        .astype("string")
        .fillna("<MISSING>")
        .ne(
            reconciliation[
                "ORIGINATION COHORT_ORIGINATION"
            ]
            .astype("string")
            .fillna("<MISSING>")
        )
        .sum()
    )

    if unmatched_targets or cohort_mismatches:
        raise ValueError(
            "Origination-target reconciliation failed: "
            f"unmatched_targets={unmatched_targets}, "
            f"cohort_mismatches={cohort_mismatches}."
        )

    target_columns_to_add = [
        *KEY_COLUMNS,
        *TARGET_AUDIT_COLUMNS,
    ]
    modeling_dataset = origination_data.merge(
        target_data[target_columns_to_add],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    leading_columns = [
        *KEY_COLUMNS,
        *COHORT_COLUMNS,
    ]
    remaining_columns = [
        column
        for column in modeling_dataset.columns
        if column not in leading_columns
    ]
    modeling_dataset = modeling_dataset[
        leading_columns + remaining_columns
    ]

    return modeling_dataset.sort_values(
        KEY_COLUMNS
    ).reset_index(drop=True)


def validate_modeling_dataset(
    origination: pd.DataFrame,
    targets: pd.DataFrame,
    modeling_dataset: pd.DataFrame,
    years: list[int] | tuple[int, ...],
    selected_horizon: int = 24,
    expected_origination_loans_per_year: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Validate row counts, keys, cohorts, labels, and event rates."""

    expected_years = sorted(years)
    duplicate_keys = int(
        modeling_dataset.duplicated(
            subset=KEY_COLUMNS
        ).sum()
    )
    missing_keys = int(
        modeling_dataset[KEY_COLUMNS].isna().any(axis=1).sum()
    )
    missing_labels = int(
        modeling_dataset[LABEL_COLUMNS]
        .isna()
        .any(axis=1)
        .sum()
    )
    observed_years = sorted(
        modeling_dataset["SOURCE YEAR"].unique().tolist()
    )
    observed_horizons = sorted(
        modeling_dataset["HORIZON_MONTHS"]
        .unique()
        .tolist()
    )
    year_cohort_mismatches = int(
        pd.to_numeric(
            modeling_dataset["ORIGINATION YEAR"],
            errors="coerce",
        )
        .ne(modeling_dataset["SOURCE YEAR"])
        .sum()
    )
    merged_row_count_matches_targets = (
        len(modeling_dataset) == len(targets)
    )

    origination_counts = (
        origination.groupby("SOURCE YEAR")
        .size()
        .rename("origination_loans")
    )
    target_counts = (
        targets.groupby("SOURCE YEAR")
        .size()
        .rename("target_eligible_loans")
    )
    annual_summary = (
        modeling_dataset.groupby(
            "SOURCE YEAR",
            as_index=False,
        )
        .agg(
            merged_loans=("LOAN IDENTIFIER", "size"),
            serious_delinquency_events=(
                "TARGET_90_PLUS",
                "sum",
            ),
            composite_credit_events=(
                "TARGET_COMPOSITE_CREDIT_EVENT",
                "sum",
            ),
            composite_event_rate=(
                "TARGET_COMPOSITE_CREDIT_EVENT",
                "mean",
            ),
        )
        .merge(
            origination_counts,
            left_on="SOURCE YEAR",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        .merge(
            target_counts,
            left_on="SOURCE YEAR",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
    )
    annual_summary["sample_retention_percentage"] = (
        annual_summary["merged_loans"]
        / annual_summary["origination_loans"]
        * 100
    )
    annual_summary["composite_event_rate_percentage"] = (
        annual_summary["composite_event_rate"] * 100
    )
    annual_summary["counts_reconcile"] = (
        annual_summary["merged_loans"]
        .eq(annual_summary["target_eligible_loans"])
    )

    expected_annual_rows_match = True
    if expected_origination_loans_per_year is not None:
        expected_annual_rows_match = bool(
            annual_summary["origination_loans"]
            .eq(expected_origination_loans_per_year)
            .all()
        )

    checks = {
        "requested_years": expected_years,
        "selected_horizon_months": selected_horizon,
        "origination_rows": int(len(origination)),
        "target_rows": int(len(targets)),
        "modeling_rows": int(len(modeling_dataset)),
        "modeling_columns": int(modeling_dataset.shape[1]),
        "duplicate_loan_year_keys": duplicate_keys,
        "missing_loan_year_keys": missing_keys,
        "missing_labels": missing_labels,
        "observed_years": observed_years,
        "observed_horizons": observed_horizons,
        "year_cohort_mismatches": year_cohort_mismatches,
        "merged_row_count_matches_targets": (
            merged_row_count_matches_targets
        ),
        "annual_counts_reconcile": bool(
            annual_summary["counts_reconcile"].all()
        ),
        "expected_origination_rows_per_year_match": (
            expected_annual_rows_match
        ),
    }

    failed_checks = []

    if duplicate_keys:
        failed_checks.append("duplicate loan/year keys")
    if missing_keys:
        failed_checks.append("missing loan/year keys")
    if missing_labels:
        failed_checks.append("missing target labels")
    if observed_years != expected_years:
        failed_checks.append("unexpected year coverage")
    if observed_horizons != [selected_horizon]:
        failed_checks.append("unexpected target horizon")
    if year_cohort_mismatches:
        failed_checks.append("source-year/cohort mismatch")
    if not merged_row_count_matches_targets:
        failed_checks.append("merged and target row counts differ")
    if not checks["annual_counts_reconcile"]:
        failed_checks.append("annual counts do not reconcile")
    if not expected_annual_rows_match:
        failed_checks.append("unexpected annual origination row count")

    if failed_checks:
        raise ValueError(
            "Modeling-dataset validation failed: "
            + ", ".join(failed_checks)
            + "."
        )

    return checks, annual_summary


def run_modeling_dataset_pipeline(
    years: list[int] | tuple[int, ...],
    input_directory: str | Path,
    output_directory: str | Path | None = None,
    selected_horizon: int = 24,
    expected_origination_loans_per_year: int | None = None,
) -> dict[str, object]:
    """Load, merge, validate, save, and re-read the modeling dataset."""

    input_directory = Path(input_directory)
    output_directory = Path(
        output_directory
        if output_directory is not None
        else input_directory
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    origination = load_origination_cohorts(
        years=years,
        input_directory=input_directory,
    )

    year_span = f"{min(years)}_{max(years)}"
    target_path = (
        input_directory
        / f"performance_target_{year_span}_{selected_horizon}m.parquet"
    )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Combined performance target not found: {target_path}"
        )

    targets = pd.read_parquet(target_path)
    modeling_dataset = build_modeling_dataset(
        origination=origination,
        targets=targets,
        years=years,
        selected_horizon=selected_horizon,
    )
    validation, annual_summary = validate_modeling_dataset(
        origination=origination,
        targets=targets,
        modeling_dataset=modeling_dataset,
        years=years,
        selected_horizon=selected_horizon,
        expected_origination_loans_per_year=(
            expected_origination_loans_per_year
        ),
    )

    output_paths = {
        "modeling_dataset": (
            output_directory
            / f"freddie_modeling_dataset_{year_span}_{selected_horizon}m.parquet"
        ),
        "annual_summary": (
            output_directory
            / f"freddie_modeling_dataset_{year_span}_{selected_horizon}m_summary.csv"
        ),
        "validation": (
            output_directory
            / f"freddie_modeling_dataset_{year_span}_{selected_horizon}m_validation.json"
        ),
    }

    modeling_dataset.to_parquet(
        output_paths["modeling_dataset"],
        index=False,
        engine="pyarrow",
    )
    annual_summary.to_csv(
        output_paths["annual_summary"],
        index=False,
    )

    saved_dataset = pd.read_parquet(
        output_paths["modeling_dataset"]
    )
    saved_validation, saved_annual_summary = (
        validate_modeling_dataset(
            origination=origination,
            targets=targets,
            modeling_dataset=saved_dataset,
            years=years,
            selected_horizon=selected_horizon,
            expected_origination_loans_per_year=(
                expected_origination_loans_per_year
            ),
        )
    )

    validation_payload = {
        **validation,
        "input_target_path": str(target_path),
        "output_dataset_path": str(
            output_paths["modeling_dataset"]
        ),
        "readback_validation_matches": (
            validation == saved_validation
        ),
        "readback_annual_summary_matches": True,
    }

    try:
        pd.testing.assert_frame_equal(
            annual_summary.reset_index(drop=True),
            saved_annual_summary.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
        )
    except AssertionError as error:
        validation_payload[
            "readback_annual_summary_matches"
        ] = False
        raise ValueError(
            "Saved modeling-dataset annual summary failed readback "
            "validation."
        ) from error

    if not validation_payload["readback_validation_matches"]:
        raise ValueError(
            "Saved modeling dataset failed readback validation."
        )

    with output_paths["validation"].open(
        mode="w",
        encoding="utf-8",
    ) as validation_file:
        json.dump(
            validation_payload,
            validation_file,
            indent=2,
        )

    return {
        "origination": origination,
        "targets": targets,
        "modeling_dataset": modeling_dataset,
        "annual_summary": annual_summary,
        "validation": validation_payload,
        "output_paths": output_paths,
    }
