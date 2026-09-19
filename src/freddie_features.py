"""Leakage-safe feature engineering for the Freddie Mac modeling dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_YEARS = (2015, 2016, 2017)
EVALUATION_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)

KEY_COLUMNS = ["SOURCE YEAR", "LOAN IDENTIFIER"]
METADATA_COLUMNS = [
    "SOURCE YEAR",
    "LOAN IDENTIFIER",
    "ORIGINATION YEAR",
    "ORIGINATION QUARTER",
    "ORIGINATION COHORT",
    "COHORT ROLE",
]
LABEL_COLUMNS = ["TARGET_90_PLUS", "TARGET_COMPOSITE_CREDIT_EVENT"]
TARGET_AUDIT_COLUMNS = [
    "HORIZON_MONTHS",
    "FIRST_SERIOUS_DELINQUENCY_PERIOD",
    "FIRST_TERMINAL_CREDIT_EVENT_PERIOD",
    "FIRST_OBSERVED_MONTH",
    "LAST_OBSERVED_MONTH",
    "OBSERVED_MONTHS",
]

NUMERIC_FEATURES = [
    "FICO_SCORE",
    "MI_PERCENT",
    "ORIGINAL_DTI",
    "ORIGINAL_LTV",
    "ORIGINAL_INTEREST_RATE",
    "LOG_ORIGINAL_UPB",
    "CLTV_MINUS_LTV",
    "HAS_MORTGAGE_INSURANCE",
    "HAS_SUBORDINATE_FINANCING",
    "HIGH_LTV_GT_80",
    "FICO_SCORE_MISSING",
    "MI_PERCENT_MISSING",
    "ORIGINAL_DTI_MISSING",
    "ORIGINAL_LTV_MISSING",
    "CLTV_MISSING",
]

CATEGORICAL_FEATURES = [
    "FIRST_TIME_HOMEBUYER",
    "NUMBER_OF_UNITS",
    "OCCUPANCY_STATUS",
    "CHANNEL",
    "AMORTIZATION_TYPE",
    "PROPERTY_STATE",
    "PROPERTY_TYPE",
    "LOAN_PURPOSE",
    "ORIGINAL_LOAN_TERM",
    "NUMBER_OF_BORROWERS_GROUP",
    "SUPER_CONFORMING_FLAG",
    "SPECIAL_ELIGIBILITY_PROGRAM",
    "HARP_INDICATOR",
    "INTEREST_ONLY_INDICATOR",
]

PREDICTOR_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_SOURCE_COLUMNS = [
    *KEY_COLUMNS,
    "ORIGINATION YEAR",
    "ORIGINATION QUARTER",
    "ORIGINATION COHORT",
    "CLASSIC FICO_CLEAN",
    "FIRST TIME HOMEBUYER INDICATOR_CLEAN",
    "MORTGAGE INSURANCE PERCENTAGE (MI %)_CLEAN",
    "NUMBER OF UNITS_CLEAN",
    "OCCUPANCY STATUS_CLEAN",
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)_CLEAN",
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO_CLEAN",
    "ORIGINAL UPB",
    "ORIGINAL LOAN-TO-VALUE (LTV)_CLEAN",
    "ORIGINAL INTEREST RATE",
    "CHANNEL_CLEAN",
    "AMORTIZATION TYPE",
    "PROPERTY STATE",
    "PROPERTY TYPE_CLEAN",
    "LOAN PURPOSE_CLEAN",
    "ORIGINAL LOAN TERM",
    "NUMBER OF BORROWERS GROUP_CLEAN",
    "SUPER CONFORMING FLAG",
    "SPECIAL ELIGIBILITY PROGRAM",
    "HARP INDICATOR",
    "INTEREST ONLY (I/O) INDICATOR",
    *LABEL_COLUMNS,
]


FEATURE_DECISIONS = [
    ("CLASSIC FICO", "selected", "FICO_SCORE", "Core borrower credit-risk measure; use the cleaned field and retain a missingness indicator."),
    ("FIRST PAYMENT DATE", "excluded", "", "Calendar timing is represented by cohort metadata; using the date directly would let the model learn time rather than borrower risk."),
    ("FIRST TIME HOMEBUYER INDICATOR", "selected", "FIRST_TIME_HOMEBUYER", "Origination-time borrower characteristic with plausible risk relevance; use the cleaned categorical field."),
    ("MATURITY DATE", "excluded", "", "Redundant with first-payment timing and original loan term, and not a direct borrower-risk characteristic."),
    ("METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION", "excluded", "", "High-cardinality geography with changing definitions; property state provides a more stable geographic control."),
    ("MORTGAGE INSURANCE PERCENTAGE (MI %)", "selected", "MI_PERCENT; HAS_MORTGAGE_INSURANCE", "Captures mortgage-insurance protection and high-LTV loan structure; preserve amount and presence."),
    ("NUMBER OF UNITS", "selected", "NUMBER_OF_UNITS", "Property structure is known at origination and can distinguish single-unit from multi-unit exposure."),
    ("OCCUPANCY STATUS", "selected", "OCCUPANCY_STATUS", "Primary residence, second home, and investment occupancy have distinct credit-risk profiles."),
    ("ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)", "selected-derived", "CLTV_MINUS_LTV; HAS_SUBORDINATE_FINANCING", "Use the incremental CLTV-LTV gap instead of raw CLTV to capture subordinate financing without duplicating LTV."),
    ("ORIGINAL DEBT-TO-INCOME (DTI) RATIO", "selected", "ORIGINAL_DTI", "Core borrower affordability measure; use the documented cleaned field and a missingness indicator."),
    ("ORIGINAL UPB", "selected-transformed", "LOG_ORIGINAL_UPB", "Loan balance is known at origination; log1p reduces scale skew while preserving ordering."),
    ("ORIGINAL LOAN-TO-VALUE (LTV)", "selected", "ORIGINAL_LTV; HIGH_LTV_GT_80", "Core collateral-leverage measure; retain the continuous value plus the domain-relevant 80% threshold."),
    ("ORIGINAL INTEREST RATE", "selected", "ORIGINAL_INTEREST_RATE", "Contract rate affects payment burden and is available at origination."),
    ("CHANNEL", "selected", "CHANNEL", "Origination channel can reflect underwriting and acquisition differences."),
    ("PREPAYMENT PENALTY INDICATOR", "excluded", "", "Primarily related to prepayment behavior rather than the 24-month credit-event target; exclude from the core model."),
    ("AMORTIZATION TYPE", "selected", "AMORTIZATION_TYPE", "Contract structure is known at origination and may alter payment risk."),
    ("PROPERTY STATE", "selected", "PROPERTY_STATE", "Stable, interpretable geographic control that avoids finer high-cardinality location fields."),
    ("PROPERTY TYPE", "selected", "PROPERTY_TYPE", "Collateral type is known at origination and has plausible loss and default relevance."),
    ("POSTAL CODE", "excluded", "", "High-cardinality local geography increases sparsity and instability; property state is retained instead."),
    ("LOAN IDENTIFIER", "metadata-only", "", "Required for reconciliation and audit, but an arbitrary identifier has no predictive meaning."),
    ("LOAN PURPOSE", "selected", "LOAN_PURPOSE", "Purchase, refinance, and cash-out purpose reflect different borrower incentives and risk."),
    ("ORIGINAL LOAN TERM", "selected", "ORIGINAL_LOAN_TERM", "Contract term changes amortization and payment burden; treat it as categorical."),
    ("NUMBER OF BORROWERS", "selected-harmonized", "NUMBER_OF_BORROWERS_GROUP", "Use the harmonized 1 versus 2+ grouping so the 2018 Q2 definition change is not mistaken for drift."),
    ("SELLER NAME", "excluded", "", "High-cardinality institution identity may overfit the sample and transfer poorly across years and to Fannie Mae."),
    ("SUPER CONFORMING FLAG", "selected", "SUPER_CONFORMING_FLAG", "Identifies a documented product segment and is available at origination."),
    ("PRE-HARP LOAN SEQUENCE NUMBER", "excluded", "", "Identifier-like linkage field, not a borrower characteristic."),
    ("SPECIAL ELIGIBILITY PROGRAM", "selected", "SPECIAL_ELIGIBILITY_PROGRAM", "Program participation captures underwriting/product mix and is retained as a categorical exposure."),
    ("HARP INDICATOR", "selected", "HARP_INDICATOR", "Program status can explain historical product-mix differences and is known at origination."),
    ("PROPERTY VALUATION METHOD", "excluded", "", "Not available throughout the 2015-2017 development window; including it would encode a disclosure-era change."),
    ("INTEREST ONLY (I/O) INDICATOR", "selected", "INTEREST_ONLY_INDICATOR", "Contract feature with direct payment-structure relevance."),
    ("VANTAGESCORE 4.0", "excluded", "", "Introduced after the development period, so it cannot support a stable 2015-2023 feature schema."),
]


def feature_decision_table() -> pd.DataFrame:
    """Return the auditable source-field inclusion and exclusion record."""

    return pd.DataFrame(
        FEATURE_DECISIONS,
        columns=["source_field", "decision", "final_feature", "reason"],
    )


def _numeric(data: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(data[column], errors="coerce")


def _categorical(data: pd.DataFrame, column: str) -> pd.Series:
    return data[column].astype("string").str.strip()


def build_engineered_features(modeling_data: pd.DataFrame) -> pd.DataFrame:
    """Create origination-time predictors while preserving audit metadata."""

    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(modeling_data.columns))
    if missing:
        raise KeyError(f"Feature engineering source columns are missing: {missing}")

    source_year = _numeric(modeling_data, "SOURCE YEAR")
    unsupported_years = sorted(
        set(source_year.dropna().astype(int))
        - set(TRAINING_YEARS)
        - set(EVALUATION_YEARS)
    )
    if unsupported_years:
        raise ValueError(f"Unsupported source years: {unsupported_years}")

    output = modeling_data[KEY_COLUMNS + [
        "ORIGINATION YEAR",
        "ORIGINATION QUARTER",
        "ORIGINATION COHORT",
    ]].copy()
    output["COHORT ROLE"] = np.where(
        source_year.isin(TRAINING_YEARS),
        "DEVELOPMENT_2015_2017",
        "EVALUATION_2018_2023",
    )

    fico = _numeric(modeling_data, "CLASSIC FICO_CLEAN")
    mi = _numeric(modeling_data, "MORTGAGE INSURANCE PERCENTAGE (MI %)_CLEAN")
    dti = _numeric(modeling_data, "ORIGINAL DEBT-TO-INCOME (DTI) RATIO_CLEAN")
    ltv = _numeric(modeling_data, "ORIGINAL LOAN-TO-VALUE (LTV)_CLEAN")
    cltv = _numeric(modeling_data, "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)_CLEAN")
    upb = _numeric(modeling_data, "ORIGINAL UPB")

    output["FICO_SCORE"] = fico
    output["MI_PERCENT"] = mi
    output["ORIGINAL_DTI"] = dti
    output["ORIGINAL_LTV"] = ltv
    output["ORIGINAL_INTEREST_RATE"] = _numeric(modeling_data, "ORIGINAL INTEREST RATE")
    output["LOG_ORIGINAL_UPB"] = np.log1p(upb.where(upb.ge(0)))
    output["CLTV_MINUS_LTV"] = (cltv - ltv).clip(lower=0)
    output["HAS_MORTGAGE_INSURANCE"] = mi.gt(0).astype("int8")
    output["HAS_SUBORDINATE_FINANCING"] = cltv.gt(ltv).astype("int8")
    output["HIGH_LTV_GT_80"] = ltv.gt(80).astype("int8")

    for final_name, values in {
        "FICO_SCORE_MISSING": fico,
        "MI_PERCENT_MISSING": mi,
        "ORIGINAL_DTI_MISSING": dti,
        "ORIGINAL_LTV_MISSING": ltv,
        "CLTV_MISSING": cltv,
    }.items():
        output[final_name] = values.isna().astype("int8")

    categorical_map = {
        "FIRST_TIME_HOMEBUYER": "FIRST TIME HOMEBUYER INDICATOR_CLEAN",
        "NUMBER_OF_UNITS": "NUMBER OF UNITS_CLEAN",
        "OCCUPANCY_STATUS": "OCCUPANCY STATUS_CLEAN",
        "CHANNEL": "CHANNEL_CLEAN",
        "AMORTIZATION_TYPE": "AMORTIZATION TYPE",
        "PROPERTY_STATE": "PROPERTY STATE",
        "PROPERTY_TYPE": "PROPERTY TYPE_CLEAN",
        "LOAN_PURPOSE": "LOAN PURPOSE_CLEAN",
        "ORIGINAL_LOAN_TERM": "ORIGINAL LOAN TERM",
        "NUMBER_OF_BORROWERS_GROUP": "NUMBER OF BORROWERS GROUP_CLEAN",
        "SUPER_CONFORMING_FLAG": "SUPER CONFORMING FLAG",
        "SPECIAL_ELIGIBILITY_PROGRAM": "SPECIAL ELIGIBILITY PROGRAM",
        "HARP_INDICATOR": "HARP INDICATOR",
        "INTEREST_ONLY_INDICATOR": "INTEREST ONLY (I/O) INDICATOR",
    }
    for final_name, source_name in categorical_map.items():
        output[final_name] = _categorical(modeling_data, source_name)

    for label in LABEL_COLUMNS:
        output[label] = _numeric(modeling_data, label).astype("int8")

    return output[METADATA_COLUMNS + PREDICTOR_COLUMNS + LABEL_COLUMNS]


def fit_preprocessing_contract(
    engineered_data: pd.DataFrame,
    training_years: tuple[int, ...] = TRAINING_YEARS,
    minimum_category_count: int = 100,
) -> dict[str, object]:
    """Fit imputation and category rules using development cohorts only."""

    training = engineered_data[
        engineered_data["SOURCE YEAR"].isin(training_years)
    ]
    if training.empty:
        raise ValueError("No rows were found for the requested training years.")

    numeric_medians = {}
    for column in NUMERIC_FEATURES:
        median = pd.to_numeric(training[column], errors="coerce").median()
        if pd.isna(median):
            raise ValueError(f"Training median is unavailable for {column}.")
        numeric_medians[column] = float(median)

    category_levels = {}
    for column in CATEGORICAL_FEATURES:
        values = training[column].astype("string").fillna("MISSING")
        counts = values.value_counts(dropna=False)
        retained = sorted(
            str(value)
            for value, count in counts.items()
            if int(count) >= minimum_category_count
        )
        category_levels[column] = retained

    return {
        "training_years": list(training_years),
        "minimum_category_count": int(minimum_category_count),
        "numeric_medians": numeric_medians,
        "category_levels": category_levels,
        "unknown_category": "OTHER",
        "missing_category": "MISSING",
    }


def apply_preprocessing_contract(
    engineered_data: pd.DataFrame,
    contract: dict[str, object],
) -> pd.DataFrame:
    """Apply frozen development-cohort rules to every temporal cohort."""

    output = engineered_data.copy()
    for column, median in contract["numeric_medians"].items():
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(median)

    missing_category = str(contract["missing_category"])
    unknown_category = str(contract["unknown_category"])
    for column, retained in contract["category_levels"].items():
        values = output[column].astype("string").fillna(missing_category)
        allowed = set(str(value) for value in retained)
        output[column] = values.where(values.isin(allowed), unknown_category)

    return output


def validate_feature_dataset(feature_data: pd.DataFrame) -> dict[str, object]:
    """Validate predictor boundaries, completeness, keys, and cohort roles."""

    required = set(METADATA_COLUMNS + PREDICTOR_COLUMNS + LABEL_COLUMNS)
    missing_columns = sorted(required - set(feature_data.columns))
    forbidden_columns = sorted(set(TARGET_AUDIT_COLUMNS) & set(feature_data.columns))
    duplicate_keys = int(feature_data.duplicated(KEY_COLUMNS).sum())
    missing_keys = int(feature_data[KEY_COLUMNS].isna().any(axis=1).sum())
    missing_predictors = int(feature_data[PREDICTOR_COLUMNS].isna().sum().sum())
    missing_labels = int(feature_data[LABEL_COLUMNS].isna().sum().sum())
    invalid_labels = {
        column: sorted(set(feature_data[column].dropna().astype(int)) - {0, 1})
        for column in LABEL_COLUMNS
    }
    invalid_labels = {key: value for key, value in invalid_labels.items() if value}
    finite_numeric = bool(
        np.isfinite(feature_data[NUMERIC_FEATURES].to_numpy(dtype=float)).all()
    )
    expected_roles = {"DEVELOPMENT_2015_2017", "EVALUATION_2018_2023"}
    observed_roles = set(feature_data["COHORT ROLE"].dropna().astype(str))

    checks = {
        "rows": int(len(feature_data)),
        "columns": int(feature_data.shape[1]),
        "predictor_count": len(PREDICTOR_COLUMNS),
        "numeric_predictor_count": len(NUMERIC_FEATURES),
        "categorical_predictor_count": len(CATEGORICAL_FEATURES),
        "missing_required_columns": missing_columns,
        "forbidden_post_origination_columns": forbidden_columns,
        "duplicate_loan_year_keys": duplicate_keys,
        "missing_loan_year_keys": missing_keys,
        "missing_predictor_values_after_contract": missing_predictors,
        "missing_labels": missing_labels,
        "invalid_label_values": invalid_labels,
        "all_numeric_predictors_finite": finite_numeric,
        "observed_cohort_roles": sorted(observed_roles),
    }

    failures = []
    if missing_columns:
        failures.append("required feature columns are missing")
    if forbidden_columns:
        failures.append("post-origination audit columns entered the feature dataset")
    if duplicate_keys or missing_keys:
        failures.append("loan/year keys are not unique and complete")
    if missing_predictors:
        failures.append("predictors remain missing after applying the contract")
    if missing_labels or invalid_labels:
        failures.append("labels are incomplete or invalid")
    if not finite_numeric:
        failures.append("numeric predictors contain infinite values")
    if observed_roles != expected_roles:
        failures.append("development/evaluation cohort roles are incomplete")
    if failures:
        raise ValueError("Feature-dataset validation failed: " + "; ".join(failures) + ".")

    return checks


def run_feature_engineering_pipeline(
    input_path: str | Path,
    output_directory: str | Path,
    minimum_category_count: int = 100,
) -> dict[str, object]:
    """Build, freeze, validate, save, and read back model features."""

    input_path = Path(input_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    modeling_data = pd.read_parquet(input_path)
    engineered = build_engineered_features(modeling_data)
    contract = fit_preprocessing_contract(
        engineered,
        minimum_category_count=minimum_category_count,
    )
    final_features = apply_preprocessing_contract(engineered, contract)
    validation = validate_feature_dataset(final_features)

    paths = {
        "features": output_directory / "freddie_features_2015_2023_24m.parquet",
        "decisions": output_directory / "freddie_feature_decisions.csv",
        "contract": output_directory / "freddie_feature_contract.json",
        "validation": output_directory / "freddie_feature_validation.json",
    }
    final_features.to_parquet(paths["features"], index=False, engine="pyarrow")
    feature_decision_table().to_csv(paths["decisions"], index=False)
    with paths["contract"].open("w", encoding="utf-8") as file:
        json.dump(contract, file, indent=2)
    with paths["validation"].open("w", encoding="utf-8") as file:
        json.dump(validation, file, indent=2)

    saved = pd.read_parquet(paths["features"])
    saved_validation = validate_feature_dataset(saved)
    if validation != saved_validation:
        raise ValueError("Saved feature dataset failed readback validation.")

    return {
        "engineered_before_contract": engineered,
        "features": final_features,
        "decisions": feature_decision_table(),
        "contract": contract,
        "validation": validation,
        "output_paths": paths,
    }
