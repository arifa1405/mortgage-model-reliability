"""Tests for leakage-safe Freddie Mac feature engineering."""

import unittest

import pandas as pd

from src.freddie_features import (
    CATEGORICAL_FEATURES,
    PREDICTOR_COLUMNS,
    apply_preprocessing_contract,
    build_engineered_features,
    feature_decision_table,
    fit_preprocessing_contract,
    validate_feature_dataset,
)


def _source_row(year: int, loan_id: str, category: str = "P") -> dict:
    return {
        "SOURCE YEAR": year,
        "LOAN IDENTIFIER": loan_id,
        "ORIGINATION YEAR": year,
        "ORIGINATION QUARTER": "Q1",
        "ORIGINATION COHORT": f"{year}Q1",
        "CLASSIC FICO_CLEAN": 740 if year != 2016 else pd.NA,
        "FIRST TIME HOMEBUYER INDICATOR_CLEAN": "N",
        "MORTGAGE INSURANCE PERCENTAGE (MI %)_CLEAN": 12,
        "NUMBER OF UNITS_CLEAN": 1,
        "OCCUPANCY STATUS_CLEAN": "P",
        "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)_CLEAN": 85,
        "ORIGINAL DEBT-TO-INCOME (DTI) RATIO_CLEAN": 36,
        "ORIGINAL UPB": 250000,
        "ORIGINAL LOAN-TO-VALUE (LTV)_CLEAN": 80,
        "ORIGINAL INTEREST RATE": 4.25,
        "CHANNEL_CLEAN": "R",
        "AMORTIZATION TYPE": "FRM",
        "PROPERTY STATE": "NY",
        "PROPERTY TYPE_CLEAN": "SF",
        "LOAN PURPOSE_CLEAN": category,
        "ORIGINAL LOAN TERM": 360,
        "NUMBER OF BORROWERS GROUP_CLEAN": "2+",
        "SUPER CONFORMING FLAG": "N",
        "SPECIAL ELIGIBILITY PROGRAM": pd.NA,
        "HARP INDICATOR": "N",
        "INTEREST ONLY (I/O) INDICATOR": "N",
        "TARGET_90_PLUS": 0,
        "TARGET_COMPOSITE_CREDIT_EVENT": 0,
        "HORIZON_MONTHS": 24,
        "FIRST_SERIOUS_DELINQUENCY_PERIOD": pd.NA,
    }


class FreddieFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = [
            _source_row(2015, "F15Q10000001"),
            _source_row(2016, "F16Q10000002"),
            _source_row(2017, "F17Q10000003"),
            _source_row(2018, "F18Q10000004", category="X"),
        ]
        rows[-1]["CLASSIC FICO_CLEAN"] = pd.NA
        self.source = pd.DataFrame(rows)

    def test_feature_boundary_and_derived_values(self) -> None:
        engineered = build_engineered_features(self.source)

        self.assertEqual(engineered.loc[0, "CLTV_MINUS_LTV"], 5)
        self.assertEqual(engineered.loc[0, "HAS_SUBORDINATE_FINANCING"], 1)
        self.assertEqual(engineered.loc[0, "HIGH_LTV_GT_80"], 0)
        self.assertNotIn("HORIZON_MONTHS", engineered.columns)
        self.assertNotIn("FIRST_SERIOUS_DELINQUENCY_PERIOD", engineered.columns)
        self.assertEqual(set(PREDICTOR_COLUMNS) - set(engineered.columns), set())

    def test_contract_uses_training_only_and_maps_unseen_category(self) -> None:
        engineered = build_engineered_features(self.source)
        contract = fit_preprocessing_contract(
            engineered,
            minimum_category_count=1,
        )
        final = apply_preprocessing_contract(engineered, contract)

        self.assertEqual(contract["numeric_medians"]["FICO_SCORE"], 740.0)
        self.assertEqual(final.loc[3, "LOAN_PURPOSE"], "OTHER")
        self.assertEqual(final.loc[3, "FICO_SCORE"], 740.0)
        self.assertFalse(final[PREDICTOR_COLUMNS].isna().any().any())

    def test_validation_accepts_complete_temporal_roles(self) -> None:
        engineered = build_engineered_features(self.source)
        contract = fit_preprocessing_contract(engineered, minimum_category_count=1)
        final = apply_preprocessing_contract(engineered, contract)
        checks = validate_feature_dataset(final)

        self.assertEqual(checks["predictor_count"], len(PREDICTOR_COLUMNS))
        self.assertEqual(
            checks["categorical_predictor_count"],
            len(CATEGORICAL_FEATURES),
        )
        self.assertEqual(checks["forbidden_post_origination_columns"], [])

    def test_decision_table_covers_all_origination_fields(self) -> None:
        decisions = feature_decision_table()
        self.assertEqual(len(decisions), 31)
        self.assertEqual(decisions["source_field"].nunique(), 31)
        self.assertTrue(decisions["reason"].str.len().gt(20).all())


if __name__ == "__main__":
    unittest.main()
