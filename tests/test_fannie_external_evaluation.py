"""Exact weighted metrics and bootstrap dependence, using synthetic loans only."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.fannie_external_evaluation import (
    METRICS, MetricCache, bootstrap_group, cohort_metadata, intervals, stable_seed,
)


class ExternalEvaluationChecks(unittest.TestCase):
    def test_integer_multiplicities_equal_expanded_sklearn_metrics_with_ties(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        p = np.array([0., .2, .2, .5, .5, 1., .2, .9])
        weights = np.array([2, 0, 3, 2, 1, 4, 0, 2])
        cache = MetricCache(y, p, [0, .2, .5, 1])
        actual = dict(zip(METRICS, cache.vector(weights)))
        ix = np.repeat(np.arange(len(y)), weights)
        self.assertAlmostEqual(actual['roc_auc'], roc_auc_score(y[ix], p[ix]), places=14)
        self.assertAlmostEqual(actual['average_precision'], average_precision_score(y[ix], p[ix]), places=14)
        self.assertAlmostEqual(actual['brier'], brier_score_loss(y[ix], p[ix]), places=14)
        self.assertAlmostEqual(actual['risk_gap_pred_minus_observed'], p[ix].mean() - y[ix].mean(), places=14)
        bins = np.searchsorted([.2, .5], p[ix], side='right')
        ece = sum(abs((p[ix][bins == j] - y[ix][bins == j]).sum()) for j in range(3)) / len(ix)
        self.assertAlmostEqual(actual['ece_development_bins'], ece, places=14)

    def test_many_random_ties_and_zero_weights(self):
        rng = np.random.default_rng(19)
        for _ in range(20):
            y = rng.integers(0, 2, 60)
            p = rng.integers(0, 11, 60) / 10
            w = np.bincount(rng.integers(0, 60, 60), minlength=60)
            ix = np.repeat(np.arange(60), w)
            values = dict(zip(METRICS, MetricCache(y, p, [0, .1, .4, 1]).vector(w)))
            self.assertAlmostEqual(values['roc_auc'], roc_auc_score(y[ix], p[ix]), places=13)
            self.assertAlmostEqual(values['average_precision'], average_precision_score(y[ix], p[ix]), places=13)

    def test_missing_class_does_not_discard_probability_metrics(self):
        cache = MetricCache([0, 0], [.1, .2], [0, 1])
        values = cache.metrics()
        self.assertTrue(np.isnan(values['roc_auc']))
        self.assertTrue(np.isnan(values['average_precision']))
        self.assertAlmostEqual(values['brier'], .025)
        self.assertAlmostEqual(values['risk_gap_pred_minus_observed'], .15)
        self.assertTrue(np.isnan(cache.vector([0, 0])).all())

    def test_exact_bootstrap_rows_and_paired_models(self):
        y = np.array([0, 1, 0, 0, 1, 0])
        p = np.array([.1, .8, .3, .2, .7, .4])
        result = bootstrap_group(y, {'one': p, 'two': p}, {'one': [0, .5, 1], 'two': [0, .5, 1]},
                                 draws=8, seed=3)
        np.testing.assert_allclose(result['draws'][:, 0], result['draws'][:, 1], equal_nan=True)
        rng = np.random.default_rng(3)
        for i in range(8):
            ix = rng.integers(0, len(y), len(y))
            expected = MetricCache(y[ix], p[ix], [0, .5, 1]).vector()
            np.testing.assert_allclose(result['draws'][i, 0], expected, atol=1e-14, rtol=0, equal_nan=True)

    def test_nested_strict_resamples_share_parent_draw(self):
        y = np.array([0, 1, 0, 1, 0, 0])
        p = np.array([.1, .7, .3, .6, .1, .2])
        mask = np.array([True, True, True, False, False, True])
        result = bootstrap_group(y, {'model': p}, {'model': [0, .5, 1]}, draws=12, seed=4, subset=mask)
        rng = np.random.default_rng(4)
        for i in range(12):
            ix = rng.integers(0, len(y), len(y)); strict_ix = ix[mask[ix]]
            self.assertEqual(result['subset_sizes'][i], len(strict_ix))
            expected = MetricCache(y[strict_ix], p[strict_ix], [0, .5, 1]).vector() if len(strict_ix) else np.full(len(METRICS), np.nan)
            np.testing.assert_allclose(result['subset_draws'][i, 0], expected, atol=1e-14, rtol=0, equal_nan=True)
        self.assertGreater(len(set(result['subset_sizes'])), 1)

    def test_whole_population_subset_is_identical(self):
        y = np.array([0, 1, 0, 1]); p = np.array([.1, .7, .3, .9])
        result = bootstrap_group(y, {'model': p}, {'model': [0, 1]}, draws=6, seed=6,
                                 subset=np.ones(4, dtype=bool))
        np.testing.assert_allclose(result['draws'], result['subset_draws'], equal_nan=True)
        self.assertTrue((result['subset_sizes'] == 4).all())

    def test_intervals_keep_invalid_draw_counts(self):
        values = np.ones((10, len(METRICS)))
        values[:3, 0] = np.nan
        table = intervals(values, np.ones(len(METRICS)), dict(population='synthetic'))
        row = table.loc[table.metric.eq('roc_auc')].iloc[0]
        self.assertEqual((row.valid_draws, row.invalid_draws), (7, 3))
        self.assertEqual((row.ci_low, row.ci_high), (1., 1.))

    def test_invalid_probabilities_labels_weights_and_edges_are_rejected(self):
        for y, p, edges in [([0, 1], [np.nan, .2], [0, 1]), ([0, 2], [.1, .2], [0, 1]),
                             ([0, 1], [.1, 1.1], [0, 1]), ([0, 1], [.1, .2], [0, .5, .5, 1])]:
            with self.assertRaises(ValueError):
                MetricCache(y, p, edges)
        with self.assertRaises(ValueError):
            MetricCache([0, 1], [.1, .2], [0, 1]).vector([1, -1])

    def test_cohort_check_catches_shifted_dates_and_duplicate_keys(self):
        frame = pd.DataFrame({'SOURCE YEAR': [2018, 2018], 'LOAN IDENTIFIER': ['F18Q10000001', 'F18Q20000002'],
                              'ORIGINATION YEAR': [2018, 2018], 'ORIGINATION QUARTER': ['Q1', 'Q2'],
                              'ORIGINATION COHORT': ['2018Q1', '2018Q2']})
        result = cohort_metadata(frame)
        self.assertEqual(result.half_year.tolist(), ['2018-H1', '2018-H1'])
        frame.loc[1, 'ORIGINATION COHORT'] = '2018Q1'
        with self.assertRaisesRegex(ValueError, 'cohort text'):
            cohort_metadata(frame)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            cohort_metadata(pd.concat([frame.iloc[:1], frame.iloc[:1]], ignore_index=True))
        fractional = frame.iloc[:1].copy()
        fractional['ORIGINATION QUARTER'] = '1.5'
        with self.assertRaisesRegex(ValueError, 'Noninteger'):
            cohort_metadata(fractional)

    def test_named_random_streams_are_reproducible_and_distinct(self):
        self.assertEqual(stable_seed('Freddie 2018Q1'), stable_seed('Freddie 2018Q1'))
        self.assertNotEqual(stable_seed('Freddie 2018Q1'), stable_seed('Fannie 2018Q1 primary'))


if __name__ == '__main__':
    unittest.main()
