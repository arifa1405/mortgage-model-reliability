"""Frozen companion evaluation and paired loan bootstrap. No model fitting."""
import hashlib

import numpy as np
import pandas as pd

from .freddie_companion import CATEGORICAL, normalize_predictors, require


METRICS = [
    "roc_auc", "average_precision", "brier", "risk_gap_pred_minus_observed",
    "absolute_risk_gap", "ece_development_bins", "ece_fixed_10",
    "prevalence", "mean_predicted",
]


def cohort_metadata(frame):
    """Check Freddie source/cohort alignment without using outcomes."""
    required = ["SOURCE YEAR", "LOAN IDENTIFIER", "ORIGINATION YEAR",
                "ORIGINATION QUARTER", "ORIGINATION COHORT"]
    require(set(required).issubset(frame), "Freddie cohort columns missing.")
    require(not frame[required].isna().any().any(), "Missing Freddie key or cohort.")
    require(not frame.duplicated(["SOURCE YEAR", "LOAN IDENTIFIER"]).any(), "Duplicate Freddie key.")
    parts = frame["LOAN IDENTIFIER"].astype("string").str.extract(r"^[FA]([0-9]{2})Q([1-4])[0-9]{7}$")
    require(not parts.isna().any().any(), "Invalid Freddie loan identifier.")
    year = pd.to_numeric(frame["ORIGINATION YEAR"], errors="raise")
    q = pd.to_numeric(frame["ORIGINATION QUARTER"].astype("string").str.removeprefix("Q"), errors="raise")
    require(year.mod(1).eq(0).all() and q.mod(1).eq(0).all(), "Noninteger Freddie year/quarter.")
    year, q = year.astype(int), q.astype(int)
    require(year.eq(2000 + parts[0].astype(int)).all() and q.eq(parts[1].astype(int)).all(),
            "Freddie loan identifier and cohort disagree.")
    require(pd.to_numeric(frame["SOURCE YEAR"]).eq(year).all(), "Source and origination year differ.")
    quarter = year.astype(str) + "Q" + q.astype(str)
    require(frame["ORIGINATION COHORT"].astype("string").eq(quarter).all(), "Saved cohort text differs.")
    return pd.DataFrame(dict(year=year, quarter=quarter,
                             half_year=year.astype(str) + "-H" + ((q - 1) // 2 + 1).astype(str)), index=frame.index)


def input_coverage(frame, contract, pipeline):
    """Aggregate imputation, rare/unseen mapping and one-hot coverage."""
    x = normalize_predictors(frame, contract["predictors"])
    encoder = pipeline.named_steps["prepare"].named_transformers_["category"]
    encoded_levels = dict(zip(CATEGORICAL, encoder.categories_))
    rows = []
    for c in x:
        missing = x[c].isna()
        other = np.zeros(len(x), dtype=bool)
        unseen = np.zeros(len(x), dtype=bool)
        if c in CATEGORICAL:
            values = x[c].fillna(contract["missing_category"])
            other = ~values.isin(contract["category_levels"][c])
            mapped = values.where(~other, contract["unknown_category"])
            unseen = ~mapped.isin(encoded_levels[c])
        rows.append(dict(predictor=c, n=len(x), missing=int(missing.sum()),
                         missing_fraction=float(missing.mean()), mapped_to_other=int(np.sum(other)),
                         unseen_after_mapping=int(np.sum(unseen))))
    return pd.DataFrame(rows)


class MetricCache:
    """Cache probability order/ties for exact weighted ROC/AP without repeated sorting.

    Integer multiplicities represent ordinary bootstrap loan resamples, not class
    reweighting. Tied probabilities are accumulated together, matching sklearn's
    non-interpolated Average Precision and trapezoidal ROC-AUC definitions.
    """

    def __init__(self, y, p, development_edges):
        y = np.asarray(y)
        p = np.asarray(p, dtype=float)
        require(y.ndim == p.ndim == 1 and len(y) == len(p) and len(y) > 0, "Invalid prediction dimensions.")
        require(np.isin(y, [0, 1]).all(), "Invalid evaluation outcome.")
        require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid probabilities.")
        edges = np.asarray(development_edges, dtype=float)
        require(len(edges) >= 2 and edges[0] == 0 and edges[-1] == 1 and (np.diff(edges) > 0).all(),
                "Invalid frozen calibration edges.")
        self.y, self.p = y.astype(float), p
        self.loss = (p - self.y) ** 2
        self.error = p - self.y
        self.order = np.argsort(-p, kind="stable")
        ordered_p = p[self.order]
        self.tie_ends = np.r_[np.flatnonzero(ordered_p[:-1] != ordered_p[1:]), len(p) - 1]
        self.ordered_y = self.y[self.order]
        self.dev_bin = np.searchsorted(edges[1:-1], p, side="right")
        self.fixed_bin = np.searchsorted(np.linspace(0, 1, 11)[1:-1], p, side="right")

    def vector(self, multiplicities=None):
        w = np.ones(len(self.y)) if multiplicities is None else np.asarray(multiplicities, dtype=float)
        require(w.shape == self.y.shape and np.isfinite(w).all() and (w >= 0).all(), "Invalid resample weights.")
        n = float(w.sum())
        if n == 0:
            return np.full(len(METRICS), np.nan)
        positives = float(np.sum(w * self.y))
        negatives = n - positives
        prevalence = positives / n
        mean = float(np.sum(w * self.p) / n)
        gap = mean - prevalence
        auc = ap = float("nan")
        if positives > 0 and negatives > 0:
            sorted_w = w[self.order]
            tp = np.cumsum(sorted_w * self.ordered_y)[self.tie_ends]
            total = np.cumsum(sorted_w)[self.tie_ends]
            fp = total - tp
            tpr, fpr = tp / positives, fp / negatives
            auc = float(np.sum(np.diff(np.r_[0., fpr]) * (tpr + np.r_[0., tpr[:-1]]) / 2))
            precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
            ap = float(np.sum(np.diff(np.r_[0., tpr]) * precision))
        bin_errors = w * self.error
        ece_dev = np.abs(np.bincount(self.dev_bin, weights=bin_errors)).sum() / n
        ece_fixed = np.abs(np.bincount(self.fixed_bin, weights=bin_errors)).sum() / n
        return np.array([auc, ap, np.sum(w * self.loss) / n, gap, abs(gap),
                         ece_dev, ece_fixed, prevalence, mean], dtype=float)

    def metrics(self):
        return dict(zip(METRICS, map(float, self.vector())))


def summarize(y, predictions, frozen_edges):
    y = np.asarray(y)
    caches = {key: MetricCache(y, p, frozen_edges[key]) for key, p in predictions.items()}
    return pd.DataFrame([dict(model=key, n=len(y), events=int(y.sum()), negatives=int(len(y) - y.sum()),
                              **cache.metrics()) for key, cache in caches.items()])


def bootstrap_group(y, predictions, frozen_edges, draws=1000, seed=42, subset=None, progress=None):
    """Shared draws for all models; optionally derive a nested eligibility subset.

    Resample n parent loans with replacement. The strict subset is selected from
    each same parent resample, so its draw count varies. It is never treated as an
    independent sample from primary. No outcome stratification is applied.
    """
    require(isinstance(draws, int) and draws > 0, "Positive integer bootstrap draws required.")
    keys = list(predictions)
    require(bool(keys), "No predictions for bootstrap.")
    y = np.asarray(y)
    full = [MetricCache(y, predictions[k], frozen_edges[k]) for k in keys]
    require(all(c.y.shape == y.shape for c in full), "Bootstrap prediction length differs.")
    if subset is not None:
        subset = np.asarray(subset)
        require(subset.dtype == bool and subset.shape == y.shape and subset.any(), "Invalid nested subset.")
        nested = [MetricCache(y[subset], np.asarray(predictions[k])[subset], frozen_edges[k]) for k in keys]
    else:
        nested = None
    result = np.full((draws, len(keys), len(METRICS)), np.nan)
    nested_result = np.full_like(result, np.nan) if nested else None
    nested_sizes = np.zeros(draws, dtype=int) if nested else None
    rng = np.random.default_rng(seed)
    for i in range(draws):
        # Exactly the multiplicities of n iid draws from the n observed loans.
        w = np.bincount(rng.integers(0, len(y), size=len(y)), minlength=len(y))
        for j, cache in enumerate(full):
            result[i, j] = cache.vector(w)
        if nested:
            nested_sizes[i] = int(w[subset].sum())
            for j, cache in enumerate(nested):
                nested_result[i, j] = cache.vector(w[subset])
        if progress and ((i + 1) % 100 == 0 or i == draws - 1):
            progress(i + 1, draws)
    return dict(models=keys, draws=result, subset_draws=nested_result, subset_sizes=nested_sizes)


def intervals(draws, point, labels):
    """Percentile intervals with explicit invalid/valid draw counts."""
    values = np.asarray(draws)
    require(values.ndim == 2 and values.shape[1] == len(METRICS), "Interval dimensions differ.")
    point = np.asarray(point)
    rows = []
    for j, metric in enumerate(METRICS):
        valid = values[:, j][np.isfinite(values[:, j])]
        lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        rows.append(dict(**labels, metric=metric, estimate=float(point[j]), ci_low=float(lo), ci_high=float(hi),
                         valid_draws=len(valid), invalid_draws=len(values) - len(valid)))
    return pd.DataFrame(rows)


def stable_seed(name, seed=42):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


def paired_model_intervals(group, point, population):
    """Predeclared family/specification differences on the same observed loans."""
    pairs = [
        ("primary_22__xgboost", "primary_22__logistic"),
        ("fico_sensitivity_24__xgboost", "fico_sensitivity_24__logistic"),
        ("fico_sensitivity_24__logistic", "primary_22__logistic"),
        ("fico_sensitivity_24__xgboost", "primary_22__xgboost"),
    ]
    rows = []
    for left, right in pairs:
        i, j = group["models"].index(left), group["models"].index(right)
        rows.append(intervals(group["draws"][:, i] - group["draws"][:, j], point[i] - point[j],
                              dict(population=population, comparison=f"{left}_minus_{right}", dependence="paired_same_loans")))
    return pd.concat(rows, ignore_index=True)
