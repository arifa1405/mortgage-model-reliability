"""Tests for Freddie Mac origination-target dataset construction."""

import unittest

import pandas as pd

from src.freddie_modeling_dataset import (
    build_modeling_dataset,
    validate_modeling_dataset,
)


class FreddieModelingDatasetTests(unittest.TestCase):
    """Exercise merge cardinality and cross-source validation."""

    def setUp(self) -> None:
        self.years = [2015, 2016]
        self.origination = pd.DataFrame(
            {
                "SOURCE YEAR": [2015, 2015, 2016, 2016],
                "LOAN IDENTIFIER": [
                    "F15Q10000001",
                    "F15Q20000002",
                    "F16Q10000003",
                    "F16Q20000004",
                ],
                "ORIGINATION YEAR": [2015, 2015, 2016, 2016],
                "ORIGINATION QUARTER": ["Q1", "Q2", "Q1", "Q2"],
                "ORIGINATION COHORT": [
                    "2015Q1",
                    "2015Q2",
                    "2016Q1",
                    "2016Q2",
                ],
                "ORIGINAL UPB": [100000, 120000, 140000, 160000],
            }
        )
        self.targets = pd.DataFrame(
            {
                "SOURCE YEAR": [2015, 2016, 2016],
                "LOAN IDENTIFIER": [
                    "F15Q10000001",
                    "F16Q10000003",
                    "F16Q20000004",
                ],
                "ORIGINATION COHORT": [
                    "2015Q1",
                    "2016Q1",
                    "2016Q2",
                ],
                "HORIZON_MONTHS": [24, 24, 24],
                "TARGET_90_PLUS": [0, 1, 0],
                "TARGET_COMPOSITE_CREDIT_EVENT": [0, 1, 1],
                "FIRST_SERIOUS_DELINQUENCY_PERIOD": [
                    pd.NA,
                    "201703",
                    pd.NA,
                ],
                "FIRST_TERMINAL_CREDIT_EVENT_PERIOD": [
                    pd.NA,
                    pd.NA,
                    "201711",
                ],
                "FIRST_OBSERVED_MONTH": [0, 0, 0],
                "LAST_OBSERVED_MONTH": [23, 23, 15],
                "OBSERVED_MONTHS": [24, 24, 16],
            }
        )

    def test_build_and_validate_modeling_dataset(self) -> None:
        dataset = build_modeling_dataset(
            origination=self.origination,
            targets=self.targets,
            years=self.years,
        )

        validation, annual_summary = validate_modeling_dataset(
            origination=self.origination,
            targets=self.targets,
            modeling_dataset=dataset,
            years=self.years,
            expected_origination_loans_per_year=2,
        )

        self.assertEqual(len(dataset), len(self.targets))
        self.assertEqual(
            dataset["LOAN IDENTIFIER"].tolist(),
            [
                "F15Q10000001",
                "F16Q10000003",
                "F16Q20000004",
            ],
        )
        self.assertTrue(
            validation["merged_row_count_matches_targets"]
        )
        self.assertTrue(
            annual_summary["counts_reconcile"].all()
        )

    def test_cohort_mismatch_is_rejected(self) -> None:
        targets = self.targets.copy()
        targets.loc[0, "ORIGINATION COHORT"] = "2015Q4"

        with self.assertRaisesRegex(
            ValueError,
            "cohort_mismatches=1",
        ):
            build_modeling_dataset(
                origination=self.origination,
                targets=targets,
                years=self.years,
            )

    def test_duplicate_target_key_is_rejected(self) -> None:
        targets = pd.concat(
            [self.targets, self.targets.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicate target keys",
        ):
            build_modeling_dataset(
                origination=self.origination,
                targets=targets,
                years=self.years,
            )


if __name__ == "__main__":
    unittest.main()
