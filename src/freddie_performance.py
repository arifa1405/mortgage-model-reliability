"""Freddie Mac monthly performance validation and target construction."""

from pathlib import Path

import pandas as pd


PERFORMANCE_COLUMNS = [
    "LOAN IDENTIFIER",
    "PERIOD",
    "CURRENT ACTUAL UPB",
    "CURRENT LOAN DELINQUENCY STATUS",
    "LOAN AGE",
    "REMAINING MONTHS TO LEGAL MATURITY",
    "UNDERWRITING DEFECT AND MAJOR SERVICING DEFECT SETTLEMENT DATE",
    "MODIFICATION FLAG",
    "ZERO BALANCE CODE",
    "ZERO BALANCE EFFECTIVE DATE",
    "CURRENT INTEREST RATE",
    "CURRENT NON-INTEREST BEARING UPB",
    "DUE DATE OF LAST PAID INSTALLMENT (DDLPI)",
    "MI RECOVERIES",
    "NET SALES PROCEEDS",
    "NON MI RECOVERIES",
    "TOTAL EXPENSES",
    "LEGAL COSTS",
    "MAINTENANCE AND PRESERVATION COSTS",
    "TAXES AND INSURANCE",
    "MISCELLANEOUS EXPENSES",
    "ACTUAL LOSS",
    "CUMULATIVE MODIFICATION COSTS",
    "INTEREST RATE STEP INDICATOR",
    "PAYMENT DEFERRAL FLAG",
    "ESTIMATED LOAN-TO-VALUE (ELTV)",
    "ZERO BALANCE REMOVAL UPB",
    "DELINQUENT ACCRUED INTEREST",
    "DELINQUENCY DUE TO DISASTER",
    "BORROWER ASSISTANCE PLAN",
    "CURRENT PERIOD MODIFICATION COSTS",
    "CURRENT INTEREST BEARING UPB",
    "MORTGAGE INSURANCE CANCELLATION INDICATOR",
    "SERVICER NAME",
    "BANKRUPTCY CRAMDOWN COSTS",
]


PERFORMANCE_STRING_COLUMNS = {
    "LOAN IDENTIFIER": "string",
    "PERIOD": "string",
    "CURRENT LOAN DELINQUENCY STATUS": "string",
    "UNDERWRITING DEFECT AND MAJOR SERVICING DEFECT SETTLEMENT DATE": "string",
    "MODIFICATION FLAG": "string",
    "ZERO BALANCE CODE": "string",
    "ZERO BALANCE EFFECTIVE DATE": "string",
    "DUE DATE OF LAST PAID INSTALLMENT (DDLPI)": "string",
    "INTEREST RATE STEP INDICATOR": "string",
    "PAYMENT DEFERRAL FLAG": "string",
    "DELINQUENCY DUE TO DISASTER": "string",
    "BORROWER ASSISTANCE PLAN": "string",
    "MORTGAGE INSURANCE CANCELLATION INDICATOR": "string",
    "SERVICER NAME": "string",
}


TARGET_PERFORMANCE_COLUMNS = [
    "LOAN IDENTIFIER",
    "PERIOD",
    "CURRENT LOAN DELINQUENCY STATUS",
    "ZERO BALANCE CODE",
    "ZERO BALANCE EFFECTIVE DATE",
]


DOCUMENTED_ZERO_BALANCE_CODES = {
    "01",
    "02",
    "03",
    "09",
    "15",
    "16",
    "96",
}


CREDIT_EVENT_ZERO_BALANCE_CODES = {
    "02",
    "03",
    "09",
    "15",
}


def validate_raw_performance_structure(
    file_path: str | Path,
    max_reported_errors: int = 10,
) -> dict[str, int]:
    """Confirm that every raw performance record contains 35 fields."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Performance file not found: {file_path}"
        )

    expected_field_count = len(PERFORMANCE_COLUMNS)
    rows_scanned = 0
    field_count_errors = 0
    error_examples = []

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
    ) as raw_file:
        for line_number, line in enumerate(raw_file, start=1):
            rows_scanned += 1
            observed_field_count = len(
                line.rstrip("\r\n").split("|")
            )

            if observed_field_count != expected_field_count:
                field_count_errors += 1

                if len(error_examples) < max_reported_errors:
                    error_examples.append(
                        f"Line {line_number}: expected "
                        f"{expected_field_count} fields but found "
                        f"{observed_field_count}."
                    )

    if rows_scanned == 0:
        raise ValueError(
            f"Performance file is empty: {file_path}"
        )

    if field_count_errors > 0:
        error_message = [
            "Raw performance structure validation failed.",
            f"Field-count errors: {field_count_errors}",
        ]

        if error_examples:
            error_message.append("Examples:")
            error_message.extend(error_examples)

        raise ValueError("\n".join(error_message))

    return {
        "rows_scanned": rows_scanned,
        "expected_fields_per_row": expected_field_count,
        "field_count_errors": field_count_errors,
    }


def load_performance(
    file_path: str | Path,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Validate and load a Freddie Mac monthly performance file."""

    file_path = Path(file_path)
    structure_report = validate_raw_performance_structure(
        file_path
    )

    if usecols is None:
        selected_columns = PERFORMANCE_COLUMNS
    else:
        unknown_columns = [
            column
            for column in usecols
            if column not in PERFORMANCE_COLUMNS
        ]

        if unknown_columns:
            raise KeyError(
                "Unknown performance columns requested: "
                f"{unknown_columns}"
            )

        selected_columns = usecols

    selected_dtypes = {
        column: dtype
        for column, dtype in PERFORMANCE_STRING_COLUMNS.items()
        if column in selected_columns
    }

    performance = pd.read_csv(
        file_path,
        sep="|",
        header=None,
        names=PERFORMANCE_COLUMNS,
        usecols=selected_columns,
        dtype=selected_dtypes,
        low_memory=False,
        encoding="utf-8-sig",
    )

    performance = performance[selected_columns]
    performance.attrs["structure_validation"] = (
        structure_report
    )

    return performance


def _valid_yyyymm(values: pd.Series) -> pd.Series:
    """Return a mask for real calendar months stored as YYYYMM."""

    text = values.astype("string")
    format_valid = text.str.fullmatch(
        r"\d{6}",
        na=False,
    )
    month = pd.to_numeric(
        text.str[-2:],
        errors="coerce",
    )

    return format_valid & month.between(1, 12)


def validate_performance_for_target(
    performance: pd.DataFrame,
    origination: pd.DataFrame,
) -> dict[str, int]:
    """Validate keys, dates, and codes needed for target construction."""

    required_performance_columns = set(
        TARGET_PERFORMANCE_COLUMNS
    )
    missing_performance_columns = sorted(
        required_performance_columns
        - set(performance.columns)
    )

    if missing_performance_columns:
        raise KeyError(
            "Missing target performance columns: "
            f"{missing_performance_columns}"
        )

    required_origination_columns = {
        "LOAN IDENTIFIER",
        "FIRST PAYMENT DATE",
        "ORIGINATION COHORT",
    }
    missing_origination_columns = sorted(
        required_origination_columns
        - set(origination.columns)
    )

    if missing_origination_columns:
        raise KeyError(
            "Missing origination columns: "
            f"{missing_origination_columns}"
        )

    missing_loan_ids = int(
        performance["LOAN IDENTIFIER"].isna().sum()
    )
    invalid_periods = int(
        (~_valid_yyyymm(performance["PERIOD"])).sum()
    )
    duplicate_loan_periods = int(
        performance.duplicated(
            subset=["LOAN IDENTIFIER", "PERIOD"]
        ).sum()
    )

    delinquency = performance[
        "CURRENT LOAN DELINQUENCY STATUS"
    ].astype("string").str.strip()
    delinquency_is_documented = (
        delinquency.eq("RA")
        | delinquency.str.fullmatch(
            r"\d{1,3}",
            na=False,
        )
    )
    unexpected_delinquency_statuses = int(
        (~delinquency_is_documented).sum()
    )

    zero_balance_code = performance[
        "ZERO BALANCE CODE"
    ].astype("string").str.strip().str.zfill(2)
    zero_balance_is_documented = (
        zero_balance_code.isna()
        | zero_balance_code.isin(
            DOCUMENTED_ZERO_BALANCE_CODES
        )
    )
    unexpected_zero_balance_codes = int(
        (~zero_balance_is_documented).sum()
    )

    origination_ids = set(
        origination["LOAN IDENTIFIER"]
        .dropna()
        .astype("string")
    )
    performance_ids = set(
        performance["LOAN IDENTIFIER"]
        .dropna()
        .astype("string")
    )
    unmatched_performance_loans = len(
        performance_ids - origination_ids
    )
    origination_loans_without_performance = len(
        origination_ids - performance_ids
    )

    critical_errors = {
        "missing_loan_ids": missing_loan_ids,
        "invalid_periods": invalid_periods,
        "duplicate_loan_periods": duplicate_loan_periods,
        "unexpected_delinquency_statuses": (
            unexpected_delinquency_statuses
        ),
        "unexpected_zero_balance_codes": (
            unexpected_zero_balance_codes
        ),
        "unmatched_performance_loans": (
            unmatched_performance_loans
        ),
    }

    failed_checks = {
        name: value
        for name, value in critical_errors.items()
        if value > 0
    }

    if failed_checks:
        raise ValueError(
            "Performance target validation failed: "
            f"{failed_checks}"
        )

    return {
        "performance_rows": len(performance),
        "performance_columns_loaded": performance.shape[1],
        "performance_loans": len(performance_ids),
        "origination_loans": len(origination_ids),
        **critical_errors,
        "origination_loans_without_performance": (
            origination_loans_without_performance
        ),
    }


def prepare_performance_timeline(
    performance: pd.DataFrame,
    origination: pd.DataFrame,
) -> pd.DataFrame:
    """Add a first-payment-relative month index and event flags."""

    validate_performance_for_target(
        performance=performance,
        origination=origination,
    )

    first_payment = origination[
        ["LOAN IDENTIFIER", "FIRST PAYMENT DATE"]
    ].copy()
    first_payment["LOAN IDENTIFIER"] = first_payment[
        "LOAN IDENTIFIER"
    ].astype("string")

    if first_payment["LOAN IDENTIFIER"].duplicated().any():
        raise ValueError(
            "Origination data contains duplicate loan identifiers."
        )

    if (~_valid_yyyymm(first_payment["FIRST PAYMENT DATE"])).any():
        raise ValueError(
            "Origination data contains invalid FIRST PAYMENT DATE values."
        )

    timeline = performance.copy()
    timeline["LOAN IDENTIFIER"] = timeline[
        "LOAN IDENTIFIER"
    ].astype("string")
    timeline = timeline.merge(
        first_payment,
        on="LOAN IDENTIFIER",
        how="left",
        validate="many_to_one",
    )

    reporting_period = timeline["PERIOD"].astype("string")
    first_payment_period = timeline[
        "FIRST PAYMENT DATE"
    ].astype("string")

    reporting_month_number = (
        pd.to_numeric(
            reporting_period.str[:4],
            errors="raise",
        )
        * 12
        + pd.to_numeric(
            reporting_period.str[4:6],
            errors="raise",
        )
    )
    first_payment_month_number = (
        pd.to_numeric(
            first_payment_period.str[:4],
            errors="raise",
        )
        * 12
        + pd.to_numeric(
            first_payment_period.str[4:6],
            errors="raise",
        )
    )

    timeline["MONTHS FROM FIRST PAYMENT"] = (
        reporting_month_number
        - first_payment_month_number
    ).astype("int32")

    delinquency = timeline[
        "CURRENT LOAN DELINQUENCY STATUS"
    ].astype("string").str.strip()
    delinquency_numeric = pd.to_numeric(
        delinquency.where(delinquency.ne("RA")),
        errors="coerce",
    )

    timeline["SERIOUS DELINQUENCY EVENT"] = (
        delinquency_numeric.ge(3)
        | delinquency.eq("RA")
    )

    zero_balance_code = timeline[
        "ZERO BALANCE CODE"
    ].astype("string").str.strip().str.zfill(2)
    timeline["ZERO BALANCE CODE CLEAN"] = zero_balance_code
    timeline["TERMINATION EVENT"] = zero_balance_code.notna()
    timeline["TERMINAL CREDIT EVENT"] = (
        zero_balance_code.isin(
            CREDIT_EVENT_ZERO_BALANCE_CODES
        )
    )
    timeline["COMPOSITE CREDIT EVENT"] = (
        timeline["SERIOUS DELINQUENCY EVENT"]
        | timeline["TERMINAL CREDIT EVENT"]
    )

    return timeline


def create_horizon_targets(
    timeline: pd.DataFrame,
    origination: pd.DataFrame,
    horizons: tuple[int, ...] = (12, 24, 36),
) -> pd.DataFrame:
    """Create leakage-safe loan-level targets for candidate horizons."""

    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError(
            "Every target horizon must be a positive integer."
        )

    base = origination[
        ["LOAN IDENTIFIER", "ORIGINATION COHORT"]
    ].copy()
    base["LOAN IDENTIFIER"] = base[
        "LOAN IDENTIFIER"
    ].astype("string")

    nonnegative_timeline = timeline.loc[
        timeline["MONTHS FROM FIRST PAYMENT"].ge(0)
    ].copy()

    observation_summary = (
        timeline.groupby(
            "LOAN IDENTIFIER",
            observed=True,
        )["MONTHS FROM FIRST PAYMENT"]
        .agg(
            FIRST_OBSERVED_MONTH="min",
            LAST_OBSERVED_MONTH="max",
        )
        .reset_index()
    )

    target_tables = []

    for horizon in horizons:
        in_horizon = nonnegative_timeline[
            "MONTHS FROM FIRST PAYMENT"
        ].lt(horizon)
        horizon_data = nonnegative_timeline.loc[
            in_horizon
        ].copy()

        event_summary = (
            horizon_data.groupby(
                "LOAN IDENTIFIER",
                observed=True,
            )
            .agg(
                OBSERVED_MONTHS=(
                    "MONTHS FROM FIRST PAYMENT",
                    "nunique",
                ),
                LAST_MONTH_WITHIN_HORIZON=(
                    "MONTHS FROM FIRST PAYMENT",
                    "max",
                ),
                SERIOUS_DELINQUENCY_WITHIN_HORIZON=(
                    "SERIOUS DELINQUENCY EVENT",
                    "max",
                ),
                TERMINAL_CREDIT_EVENT_WITHIN_HORIZON=(
                    "TERMINAL CREDIT EVENT",
                    "max",
                ),
                COMPOSITE_CREDIT_EVENT_WITHIN_HORIZON=(
                    "COMPOSITE CREDIT EVENT",
                    "max",
                ),
                TERMINATION_WITHIN_HORIZON=(
                    "TERMINATION EVENT",
                    "max",
                ),
            )
            .reset_index()
        )

        serious_period = (
            horizon_data.loc[
                horizon_data[
                    "SERIOUS DELINQUENCY EVENT"
                ]
            ]
            .groupby(
                "LOAN IDENTIFIER",
                observed=True,
            )["PERIOD"]
            .min()
            .rename("FIRST_SERIOUS_DELINQUENCY_PERIOD")
            .reset_index()
        )
        credit_event_period = (
            horizon_data.loc[
                horizon_data["TERMINAL CREDIT EVENT"]
            ]
            .groupby(
                "LOAN IDENTIFIER",
                observed=True,
            )["PERIOD"]
            .min()
            .rename("FIRST_TERMINAL_CREDIT_EVENT_PERIOD")
            .reset_index()
        )

        target = base.merge(
            observation_summary,
            on="LOAN IDENTIFIER",
            how="left",
            validate="one_to_one",
        )
        target = target.merge(
            event_summary,
            on="LOAN IDENTIFIER",
            how="left",
            validate="one_to_one",
        )
        target = target.merge(
            serious_period,
            on="LOAN IDENTIFIER",
            how="left",
            validate="one_to_one",
        )
        target = target.merge(
            credit_event_period,
            on="LOAN IDENTIFIER",
            how="left",
            validate="one_to_one",
        )

        boolean_columns = [
            "SERIOUS_DELINQUENCY_WITHIN_HORIZON",
            "TERMINAL_CREDIT_EVENT_WITHIN_HORIZON",
            "COMPOSITE_CREDIT_EVENT_WITHIN_HORIZON",
            "TERMINATION_WITHIN_HORIZON",
        ]

        for column in boolean_columns:
            target[column] = target[column].fillna(False).astype(bool)

        target["OBSERVED_MONTHS"] = (
            target["OBSERVED_MONTHS"]
            .fillna(0)
            .astype("int32")
        )
        target["HORIZON_MONTHS"] = horizon
        target["LEFT_TRUNCATED"] = (
            target["FIRST_OBSERVED_MONTH"].isna()
            | target["FIRST_OBSERVED_MONTH"].gt(0)
        )
        target["FOLLOW_UP_COMPLETE"] = (
            target["LAST_OBSERVED_MONTH"].ge(
                horizon - 1
            )
            | target["TERMINATION_WITHIN_HORIZON"]
        )
        target["TARGET_ELIGIBLE"] = (
            ~target["LEFT_TRUNCATED"]
            & target["FOLLOW_UP_COMPLETE"]
        )

        target["TARGET_90_PLUS"] = pd.Series(
            pd.NA,
            index=target.index,
            dtype="Int8",
        )
        target["TARGET_COMPOSITE_CREDIT_EVENT"] = pd.Series(
            pd.NA,
            index=target.index,
            dtype="Int8",
        )

        eligible = target["TARGET_ELIGIBLE"]
        target.loc[eligible, "TARGET_90_PLUS"] = (
            target.loc[
                eligible,
                "SERIOUS_DELINQUENCY_WITHIN_HORIZON",
            ]
            .astype("int8")
        )
        target.loc[
            eligible,
            "TARGET_COMPOSITE_CREDIT_EVENT",
        ] = (
            target.loc[
                eligible,
                "COMPOSITE_CREDIT_EVENT_WITHIN_HORIZON",
            ]
            .astype("int8")
        )

        target_tables.append(target)

    return pd.concat(
        target_tables,
        ignore_index=True,
    )


def summarize_horizon_targets(
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize coverage and event rates for each candidate horizon."""

    summary_rows = []

    for horizon, horizon_data in targets.groupby(
        "HORIZON_MONTHS",
        sort=True,
    ):
        eligible = horizon_data.loc[
            horizon_data["TARGET_ELIGIBLE"]
        ]
        total_loans = len(horizon_data)
        eligible_loans = len(eligible)

        summary_rows.append(
            {
                "horizon_months": int(horizon),
                "total_loans": total_loans,
                "eligible_loans": eligible_loans,
                "coverage_percentage": (
                    eligible_loans / total_loans * 100
                    if total_loans
                    else 0.0
                ),
                "left_truncated_loans": int(
                    horizon_data["LEFT_TRUNCATED"].sum()
                ),
                "incomplete_follow_up_loans": int(
                    (~horizon_data["FOLLOW_UP_COMPLETE"]).sum()
                ),
                "serious_delinquency_events": int(
                    eligible["TARGET_90_PLUS"].sum()
                ),
                "serious_delinquency_rate": (
                    eligible["TARGET_90_PLUS"].mean()
                    if eligible_loans
                    else pd.NA
                ),
                "composite_credit_events": int(
                    eligible[
                        "TARGET_COMPOSITE_CREDIT_EVENT"
                    ].sum()
                ),
                "composite_credit_event_rate": (
                    eligible[
                        "TARGET_COMPOSITE_CREDIT_EVENT"
                    ].mean()
                    if eligible_loans
                    else pd.NA
                ),
            }
        )

    return pd.DataFrame(summary_rows)
