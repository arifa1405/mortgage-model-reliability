"""Companion contract checks: leakage, category spelling, outcomes and bin boundaries."""
import copy
import unittest

import numpy as np
import pandas as pd

from src.freddie_companion import (
    PRIMARY, SENSITIVITY, TARGET, apply_contract, build_companion_features,
    calibration_table, fit_contract, probability_metrics, select_candidate,
)
from src.freddie_features import REQUIRED_SOURCE_COLUMNS


def source_frame(year, n=300):
    """Synthetic records only; no actual mortgage observations."""
    rng = np.random.default_rng(year)
    i = np.arange(n)
    frame = pd.DataFrame({c: [pd.NA] * n for c in REQUIRED_SOURCE_COLUMNS})
    values = {
        'SOURCE YEAR': year, 'LOAN IDENTIFIER': [f'SYN{year}{j:05d}' for j in i],
        'ORIGINATION YEAR': year, 'ORIGINATION QUARTER': 'Q1',
        'ORIGINATION COHORT': f'{year}Q1', 'CLASSIC FICO_CLEAN': rng.integers(600, 851, n),
        'MORTGAGE INSURANCE PERCENTAGE (MI %)_CLEAN': np.where(i % 4 == 0, 25., 0.),
        'ORIGINAL DEBT-TO-INCOME (DTI) RATIO_CLEAN': rng.uniform(10, 60, n),
        'ORIGINAL LOAN-TO-VALUE (LTV)_CLEAN': rng.uniform(40, 100, n),
        'ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)_CLEAN': 100.,
        'ORIGINAL UPB': rng.integers(50000, 500000, n), 'ORIGINAL INTEREST RATE': rng.uniform(2, 6, n),
        'NUMBER OF UNITS_CLEAN': np.where(i % 5 == 0, 2., 1.),
        'OCCUPANCY STATUS_CLEAN': 'P', 'CHANNEL_CLEAN': np.where(i % 3 == 0, 'B', 'R'),
        'AMORTIZATION TYPE': 'FRM', 'PROPERTY STATE': np.where(i % 2 == 0, 'NY', 'CA'),
        'PROPERTY TYPE_CLEAN': 'SF', 'LOAN PURPOSE_CLEAN': np.array(['P', 'C', 'N'])[i % 3],
        'ORIGINAL LOAN TERM': np.where(i % 4 == 0, 180., 360.),
        'NUMBER OF BORROWERS GROUP_CLEAN': np.where(i % 2 == 0, '1', '2+'),
        'FIRST TIME HOMEBUYER INDICATOR_CLEAN': 'N', 'SUPER CONFORMING FLAG': 'N',
        'SPECIAL ELIGIBILITY PROGRAM': '9', 'HARP INDICATOR': 'N',
        'INTEREST ONLY (I/O) INDICATOR': 'N',
        'TARGET_90_PLUS': (i % 11 == 0).astype('int8'),
        TARGET: (i % 11 == 0).astype('int8'), 'HORIZON_MONTHS': 24,
    }
    for c, v in values.items():
        frame[c] = v
    frame.loc[i % 23 == 0, 'ORIGINAL DEBT-TO-INCOME (DTI) RATIO_CLEAN'] = np.nan
    return frame


class CompanionChecks(unittest.TestCase):
    def setUp(self):
        self.raw = pd.concat([source_frame(year) for year in [2015, 2016, 2017]], ignore_index=True)
        self.engineered = build_companion_features(self.raw)

    def test_original_labels_membership_and_purpose_preserved(self):
        np.testing.assert_array_equal(self.engineered[TARGET], self.raw[TARGET])
        self.assertEqual(len(self.raw), len(self.engineered))
        self.assertEqual(self.engineered.PURCHASE_REFINANCE_GROUP.head(3).tolist(),
                         ['PURCHASE', 'REFINANCE', 'REFINANCE'])
        self.assertEqual(self.engineered.NUMBER_OF_UNITS.iloc[1], '1')
        self.assertEqual(self.engineered.ORIGINAL_LOAN_TERM.iloc[1], '360')

    def test_validation_values_cannot_change_training_contract(self):
        before = fit_contract(self.engineered, 'primary_22', [2015, 2016])
        changed = self.engineered.copy()
        mask = changed['SOURCE YEAR'].eq(2017)
        changed.loc[mask, 'ORIGINAL_DTI'] = 1e12
        changed.loc[mask, 'PROPERTY_STATE'] = 'ZZ'
        changed.loc[mask, TARGET] = 1 - changed.loc[mask, TARGET]
        after = fit_contract(changed, 'primary_22', [2015, 2016])
        self.assertEqual(before, after)
        result = apply_contract(changed.loc[mask], after)
        self.assertTrue(result.PROPERTY_STATE.eq('OTHER').all())
        self.assertNotIn('ZZ', after['category_levels']['PROPERTY_STATE'])

    def test_missing_and_unseen_evaluation_inputs_use_stored_rules(self):
        contract = fit_contract(self.engineered, 'fico_sensitivity_24', [2015, 2016])
        frozen = copy.deepcopy(contract)
        evaluation = self.engineered.iloc[:2].copy()
        evaluation['ORIGINAL_DTI'] = np.nan
        evaluation['PROPERTY_STATE'] = 'ZZ'
        evaluation['NUMBER_OF_UNITS'] = ['1.0', '1']
        evaluation[TARGET] = [1, 0]
        a = apply_contract(evaluation, contract)
        evaluation[TARGET] = [0, 1]
        b = apply_contract(evaluation, contract)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(contract, frozen)
        self.assertEqual(list(a), SENSITIVITY)
        self.assertTrue(a.ORIGINAL_DTI.eq(contract['numeric_medians']['ORIGINAL_DTI']).all())
        self.assertTrue(a.PROPERTY_STATE.eq('OTHER').all())
        self.assertEqual(a.NUMBER_OF_UNITS.iloc[0], a.NUMBER_OF_UNITS.iloc[1])

    def test_no_silent_invalid_labels_or_purpose(self):
        invalid = self.raw.copy()
        invalid[TARGET] = invalid[TARGET].astype(float)
        invalid.loc[0, TARGET] = .5
        with self.assertRaisesRegex(ValueError, 'Invalid TARGET'):
            build_companion_features(invalid)
        invalid = self.raw.copy()
        invalid.loc[0, 'LOAN PURPOSE_CLEAN'] = 'U'
        with self.assertRaisesRegex(ValueError, 'purpose'):
            build_companion_features(invalid)

    def test_no_out_of_time_fit_and_no_all_missing_training_column(self):
        with self.assertRaisesRegex(ValueError, 'fit years'):
            fit_contract(self.engineered, 'primary_22', [2015, 2016, 2017, 2018])
        bad = self.engineered.copy()
        bad.loc[bad['SOURCE YEAR'].isin([2015, 2016]), 'ORIGINAL_DTI'] = np.nan
        with self.assertRaisesRegex(ValueError, 'training median'):
            fit_contract(bad, 'primary_22', [2015, 2016])

    def test_selection_uses_declared_tolerance_rounding_and_order(self):
        frame = pd.DataFrame([
            dict(candidate_order=0, average_precision=.2000, brier=.0200002),
            dict(candidate_order=1, average_precision=.2040, brier=.0200001),
            dict(candidate_order=2, average_precision=.1980, brier=.0100),
        ])
        # Candidate 2 falls outside the AP tolerance; rounded Brier tie favors 0.
        self.assertEqual(int(select_candidate(frame).candidate_order), 0)

    def test_calibration_endpoints_and_empty_bins(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0., .1, .9, 1.])
        bins = calibration_table(y, p)
        self.assertEqual(bins.n.tolist(), [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertEqual(int(bins.events.sum()), 2)
        self.assertAlmostEqual(probability_metrics(y, p)['ece_fixed_10'], .05)
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            calibration_table(y, [0, np.nan, .9, 1])

    def test_contract_and_integer_category_corruption_rejected(self):
        contract = fit_contract(self.engineered, 'primary_22', [2015, 2016])
        corrupt = copy.deepcopy(contract)
        corrupt['predictors'] = list(reversed(PRIMARY))
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            apply_contract(self.engineered, corrupt)
        evaluation = self.engineered.iloc[:2].copy()
        evaluation['NUMBER_OF_UNITS'] = ['1.5', '1']
        with self.assertRaisesRegex(ValueError, 'Noninteger'):
            apply_contract(evaluation, contract)


if __name__ == '__main__':
    unittest.main()
