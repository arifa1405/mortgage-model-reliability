"""Freddie-only companion development; separate from the original model pipeline."""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .freddie_features import build_engineered_features


TARGET = "TARGET_COMPOSITE_CREDIT_EVENT"
SEED = 42
PRIMARY = [
    "MI_PERCENT", "ORIGINAL_DTI", "ORIGINAL_LTV", "ORIGINAL_INTEREST_RATE",
    "LOG_ORIGINAL_UPB", "CLTV_MINUS_LTV", "HAS_MORTGAGE_INSURANCE",
    "HAS_SUBORDINATE_FINANCING", "HIGH_LTV_GT_80", "MI_PERCENT_MISSING",
    "ORIGINAL_DTI_MISSING", "ORIGINAL_LTV_MISSING", "CLTV_MISSING",
    "NUMBER_OF_UNITS", "OCCUPANCY_STATUS", "CHANNEL", "AMORTIZATION_TYPE",
    "PROPERTY_STATE", "PROPERTY_TYPE", "ORIGINAL_LOAN_TERM",
    "NUMBER_OF_BORROWERS_GROUP", "PURCHASE_REFINANCE_GROUP",
]
CATEGORICAL = PRIMARY[13:]
SENSITIVITY = PRIMARY + ["FICO_SCORE", "FICO_SCORE_MISSING"]
SPECIFICATIONS = {"primary_22": PRIMARY, "fico_sensitivity_24": SENSITIVITY}
# Same candidates and order as Notebook 10, selected separately for each specification.
GRID = {
    "logistic": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "xgboost": [dict(max_depth=d, n_estimators=n, learning_rate=lr)
                for d, n, lr in ((2, 200, .05), (3, 200, .05), (3, 350, .05))],
}


def require(condition, message):
    """Keep audit checks active even when Python optimization is enabled."""
    if not condition:
        raise ValueError(message)


def normalize_predictors(frame, predictors):
    """Canonical pre-imputation spelling shared with the reviewed Fannie inputs."""
    require(not frame.columns.duplicated().any(), "Duplicate input columns.")
    require(set(predictors).issubset(frame.columns), "Required companion predictors missing.")
    out = frame.loc[:, predictors].copy()
    for c in predictors:
        if c not in CATEGORICAL:
            out[c] = pd.to_numeric(out[c], errors="raise").astype(float)
            require(not np.isinf(out[c].to_numpy()).any(), f"Infinite values in {c}.")
        else:
            values = out[c].astype("string").str.strip().replace("", pd.NA)
            if c in {"NUMBER_OF_UNITS", "ORIGINAL_LOAN_TERM"}:
                numbers = pd.to_numeric(values, errors="coerce")
                require(not (values.notna() & numbers.isna()).any(), f"Invalid category in {c}.")
                require(numbers.dropna().mod(1).eq(0).all(), f"Noninteger category in {c}.")
                values = numbers.astype("Int64").astype("string")
            out[c] = values
    return out


def build_companion_features(source):
    """Reuse the original engineering; change only the companion predictor contract."""
    for c in [TARGET, "TARGET_90_PLUS"]:
        require(source[c].notna().all() and source[c].isin([0, 1]).all(), f"Invalid {c}.")
    engineered = build_engineered_features(source)
    purpose = engineered["LOAN_PURPOSE"].astype("string").str.strip().replace("", pd.NA)
    require(set(purpose.dropna()).issubset({"P", "C", "N"}), "Unexpected Freddie purpose code.")
    engineered["PURCHASE_REFINANCE_GROUP"] = purpose.map(
        {"P": "PURCHASE", "C": "REFINANCE", "N": "REFINANCE"})
    predictors = normalize_predictors(engineered, SENSITIVITY)
    metadata = engineered[["SOURCE YEAR", "LOAN IDENTIFIER", "ORIGINATION YEAR",
                           "ORIGINATION QUARTER", "ORIGINATION COHORT", TARGET]].copy()
    return pd.concat([metadata, predictors], axis=1)


def fit_contract(engineered, specification, training_years, minimum_category_count=100):
    """Learn medians and rare-category rules from the named Freddie years only."""
    predictors = SPECIFICATIONS[specification]
    years = list(training_years)
    require(years in [[2015, 2016], [2015, 2016, 2017]], "Unsupported fit years.")
    training = engineered.loc[engineered["SOURCE YEAR"].isin(years)]
    require(set(training["SOURCE YEAR"].unique()) == set(years), "Missing fit-year population.")
    x = normalize_predictors(training, predictors)
    numeric = [c for c in predictors if c not in CATEGORICAL]
    medians = {}
    for c in numeric:
        present = x[c].dropna()
        require(not present.empty, f"No finite training median for {c}.")
        value = present.median()
        require(pd.notna(value) and np.isfinite(value), f"No finite training median for {c}.")
        medians[c] = float(value)
    levels = {}
    for c in CATEGORICAL:
        counts = x[c].fillna("MISSING").value_counts()
        levels[c] = sorted(str(v) for v, count in counts.items() if count >= minimum_category_count)
    return dict(schema_version=1, specification=specification, predictors=list(predictors),
                numeric_predictors=numeric, categorical_predictors=list(CATEGORICAL),
                training_years=years, training_rows=len(training), minimum_category_count=minimum_category_count,
                numeric_medians=medians, category_levels=levels,
                missing_category="MISSING", unknown_category="OTHER",
                integer_categories=["NUMBER_OF_UNITS", "ORIGINAL_LOAN_TERM"],
                target=TARGET, horizon_months=24)


def apply_contract(frame, contract):
    """Apply stored rules without learning from evaluation rows or outcomes."""
    spec = contract["specification"]
    require(contract["predictors"] == SPECIFICATIONS[spec], "Predictor contract differs.")
    numeric = [c for c in contract["predictors"] if c not in CATEGORICAL]
    require(contract["numeric_predictors"] == numeric and contract["categorical_predictors"] == CATEGORICAL,
            "Predictor types differ.")
    require(set(contract["numeric_medians"]) == set(numeric), "Median contract differs.")
    require(set(contract["category_levels"]) == set(CATEGORICAL), "Category contract differs.")
    out = normalize_predictors(frame, contract["predictors"])
    for c, median in contract["numeric_medians"].items():
        require(np.isfinite(median), f"Invalid stored median for {c}.")
        out[c] = out[c].fillna(median)
    for c, levels in contract["category_levels"].items():
        values = out[c].fillna(contract["missing_category"])
        out[c] = values.where(values.isin(levels), contract["unknown_category"]).astype(object)
    require(not out.isna().any().any(), "Missing values after preprocessing.")
    return out


def model_pipeline(family, params, contract):
    numeric = Pipeline([("scale", StandardScaler())]) if family == "logistic" else "passthrough"
    preparation = ColumnTransformer([
        ("numeric", numeric, contract["numeric_predictors"]),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL),
    ], remainder="drop", sparse_threshold=.3)
    if family == "logistic":
        model = LogisticRegression(C=params["C"], class_weight=None, max_iter=2000, random_state=SEED)
    elif family == "xgboost":
        model = XGBClassifier(objective="binary:logistic", **params, subsample=.8,
                              colsample_bytree=.8, reg_lambda=1, eval_metric="logloss",
                              random_state=SEED, n_jobs=4, tree_method="hist", scale_pos_weight=1)
    else:
        raise ValueError(f"Unknown family: {family}")
    return Pipeline([("prepare", preparation), ("model", model)])


def fit_model(pipeline, x, y):
    require(set(np.unique(y)) == {0, 1}, "Training must contain both outcomes.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(x, y)
    return pipeline


def calibration_table(y, p, edges=None):
    """Fixed probability ranges by default; no evaluation-fitted calibration bins."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    require(len(y) == len(p) and len(y) > 0 and np.isin(y, [0, 1]).all(), "Invalid outcomes.")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid probabilities.")
    edges = np.linspace(0., 1., 11) if edges is None else np.asarray(edges, dtype=float)
    require(edges[0] == 0 and edges[-1] == 1 and (np.diff(edges) > 0).all(), "Invalid bin edges.")
    indexes = np.searchsorted(edges[1:-1], p, side="right")
    rows = []
    for i in range(len(edges) - 1):
        mask = indexes == i
        n = int(mask.sum())
        rows.append(dict(bin_index=i, lower_edge=float(edges[i]), upper_edge=float(edges[i+1]),
                         n=n, events=int(y[mask].sum()),
                         mean_predicted=float(p[mask].mean()) if n else None,
                         observed_rate=float(y[mask].mean()) if n else None))
    return pd.DataFrame(rows)


def probability_metrics(y, p):
    bins = calibration_table(y, p)
    require(set(np.unique(y)) == {0, 1}, "Validation must contain both outcomes.")
    ece = ((bins.mean_predicted - bins.observed_rate).abs().fillna(0) * bins.n).sum() / len(y)
    return dict(n=int(len(y)), events=int(np.sum(y)), prevalence=float(np.mean(y)),
                mean_predicted=float(np.mean(p)), average_precision=float(average_precision_score(y, p)),
                roc_auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                risk_gap=float(np.mean(p) - np.mean(y)), ece_fixed_10=float(ece))


def select_candidate(frame):
    """Notebook 10 AP tolerance, then six-decimal Brier, then declared grid order."""
    require(not frame.empty and np.isfinite(frame[["average_precision", "brier"]]).all().all(),
            "Invalid candidate metrics.")
    close = frame.loc[(frame.average_precision.max() - frame.average_precision) < .005].copy()
    close["brier_6"] = close.brier.round(6)
    return close.sort_values(["brier_6", "candidate_order"], kind="stable").iloc[0]


def development_bin_edges(probabilities):
    p = np.asarray(probabilities, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid reference predictions.")
    return np.unique(np.r_[0., np.quantile(p, np.linspace(0, 1, 11))[1:-1], 1.]).tolist()


def train_candidates(engineered):
    """Select four models with separate train-only contracts, never OOT rows."""
    require(set(engineered["SOURCE YEAR"].unique()) == {2015, 2016, 2017}, "Development years differ.")
    fit = engineered["SOURCE YEAR"].isin([2015, 2016])
    val = engineered["SOURCE YEAR"].eq(2017)
    y_fit = engineered.loc[fit, TARGET].to_numpy(dtype=int)
    y_val = engineered.loc[val, TARGET].to_numpy(dtype=int)
    rows, selected, contracts, validation_predictions, validation_edges = [], {}, {}, {}, {}
    for spec in SPECIFICATIONS:
        contract = fit_contract(engineered, spec, [2015, 2016])
        contracts[spec] = contract
        x_fit, x_val = apply_contract(engineered.loc[fit], contract), apply_contract(engineered.loc[val], contract)
        for family, candidates in GRID.items():
            predictions = {}
            pipelines = {}
            for order, params in enumerate(candidates):
                pipe = fit_model(model_pipeline(family, params, contract), x_fit, y_fit)
                p = pipe.predict_proba(x_val)[:, 1]
                predictions[order] = p
                pipelines[order] = pipe
                rows.append(dict(specification=spec, family=family, candidate_order=order,
                                 params=json.dumps(params, sort_keys=True), n_fit=len(y_fit),
                                 events_fit=int(y_fit.sum()), **probability_metrics(y_val, p)))
                print(spec, family, order, "AP", round(rows[-1]["average_precision"], 6),
                      "Brier", round(rows[-1]["brier"], 6), flush=True)
            group = pd.DataFrame([r for r in rows if r["specification"] == spec and r["family"] == family])
            winner = select_candidate(group).to_dict()
            key = (spec, family)
            selected[key] = winner
            validation_predictions[key] = predictions[int(winner["candidate_order"])]
            chosen_pipe = pipelines[int(winner["candidate_order"])]
            validation_edges[key] = development_bin_edges(chosen_pipe.predict_proba(x_fit)[:, 1])
    return pd.DataFrame(rows), selected, contracts, validation_predictions, validation_edges
