# Mortgage Credit-Risk Model Reliability Under Changing Borrower Populations

MS Data Science Capstone Project

## Project Overview

Mortgage credit-risk models are commonly developed on historical borrower
populations, but lending conditions, borrower characteristics, product mix, and
data disclosures change over time. A model can therefore continue producing
predictions while becoming less discriminative, poorly calibrated, or
operationally unreliable.

This project evaluates the complete reliability lifecycle of mortgage
credit-risk models:

```text
Population Drift
        ↓
Model-Relevant and Prediction Drift
        ↓
Observed Discrimination or Calibration Degradation
        ↓
Early-Warning Validation
        ↓
Evidence-Based Retraining Decision
        ↓
External Validation
```

The project deliberately distinguishes distribution change from model failure.
A feature distribution changing does not automatically mean that predictive
performance has degraded, and degradation does not automatically mean that
retraining will improve the next cohort.

## Central Research Question

Can label-free signals observed when a new mortgage cohort arrives anticipate
later discrimination or calibration degradation, and can those signals identify
when retraining is genuinely beneficial?

## Research Scope

The analysis follows two models through development, out-of-time monitoring,
degradation assessment, and retraining evaluation:

- XGBoost as the primary model for nonlinear relationships and feature interactions.
- Logistic Regression as a transparent comparison benchmark using the same temporal split and features.

The objective is model reliability under temporal population change—not a broad
model-accuracy tournament.

## Data Strategy

### Freddie Mac

Freddie Mac Single-Family Loan-Level Dataset annual samples are used for model
development and temporal monitoring.

- 2015–2017: development cohorts for model training and fitted preprocessing.
- 2018–2023: sequential out-of-time cohorts for evaluation and reliability monitoring.
- Each annual sample contains approximately 50,000 loans.
- Annual files remain intact; origination quarter will be retained as metadata
  for later cohort-level analysis.

### Fannie Mae

Fannie Mae Single-Family Loan Performance Data will provide independent
external validation after the Freddie Mac methodology is frozen. It will test
whether the observed drift, degradation, warning, and retraining conclusions
transfer across datasets.

Raw and generated data files are intentionally excluded from Git because of
their size. They must be obtained from the official data providers.

## Completed Work

### 2015 baseline discovery and EDA

- Inspected all 31 Freddie Mac origination fields against official
  documentation and domain logic.
- Preserved raw fields and created cleaned fields only for documented reasons.
- Investigated sentinels, unusual LTV and CLTV values, mortgage insurance,
  postal-code representation, HARP linkage, and disclosure limitations.
- Completed focused univariate and bivariate analysis and a numerical
  correlation heatmap.
- Distinguished legitimate program-supported values from actual data-quality
  problems.

### Reusable origination pipeline

The notebook decisions were converted into reusable Python modules for:

- Pipe-delimited ingestion and structural validation.
- Raw-preserving cleaning and documented sentinel treatment.
- Categorical-code, numeric-range, missingness, definition-timing, and
  cross-field validation.
- Quarter-aware 2018 Q2 LTV, CLTV, and borrower-definition rules.
- Cross-year numeric and categorical summaries.
- Annual Parquet, CSV, and JSON outputs.
- Saved-output and serialization verification.

### Origination harmonization

The reusable pipeline was first verified on the 2015–2019 samples:

- 250,000 sampled loans processed.
- 50,000 loans per year.
- 31 raw fields preserved.
- 47-column cleaned analytical datasets created.
- No structural, undocumented-code, numeric-range, definition-timing, or
  ratio-rule violations detected.
- No missing or duplicate loan identifiers.
- Saved warning-report counts agree with validation JSON files.
- All five saved Parquet datasets pass read-back validation.

The pipeline was then extended through 2023 (nine annual samples, 450,000 origination
rows). The analysis records disclosure and definition changes so they are not
mistaken for ordinary borrower-population drift.

### Performance targets and modeling dataset (2015–2023)

- Notebook 04 developed the 24-month outcome definition on the 2015 sample.
  The composite label records 90+ serious delinquency or a defined terminal
  credit event; the 90+ only label is retained for sensitivity analysis.
- Notebook 05 applied the fixed target contract to all nine annual cohorts,
  with annual, quarterly, coverage, event-component, and read-back checks.
- Notebook 06 joined eligible targets to cleaned origination records by source
  year and loan identifier. The validated modeling dataset has 437,668 rows
  and 59 columns, with no duplicate loan/year keys or missing labels.

### Feature engineering (2015–2023)

- Notebook 07 and `src/freddie_features.py` define 29 origination-time
  predictors (15 numeric, 14 categorical) and document decisions for all
  31 raw origination fields.
- Fitted imputation and category rules use 2015–2017 only. The frozen contract
  applies to 2018–2023; identifiers, cohort metadata, and post-origination
  target-audit fields do not enter the predictor matrix.
- The feature output has 437,668 rows and passed save/read-back validation.
  Tests cover merge integrity, leakage exclusion, unseen categories, and
  development-only preprocessing.

## Current Repository Structure

- `notebooks/01`–`07`: exploration, origination validation, target development,
  multi-year performance validation, modeling-data join, and feature engineering.
- `src/`: reusable origination, cohort, performance, modeling-dataset, and
  feature-engineering modules.
- `tests/`: modeling-dataset and feature-engineering unit tests.
- `data/raw/`, `data/interim/`, `data/processed/`: local data directories; raw
  and generated files are excluded from Git.

## Next Implementation Stages

- Complete the peer-reviewed literature review and lock the drift/degradation
  and early-warning evaluation design.
- Train XGBoost (primary) and Logistic Regression (comparison) on 2015–2017,
  then evaluate frozen models sequentially on 2018–2023.
- Measure population and prediction drift and later discrimination and
  calibration changes as 24-month labels become available.
- Back-test label-free early-warning signals and compare retraining policies
  using only information available at each decision point.
- Perform secondary external validation with harmonized Fannie Mae data.

## Planned Evaluation

### Model reliability

- PR-AUC.
- ROC-AUC.
- Brier score.
- Calibration curve, slope, and intercept.
- Precision and recall at a validation-selected threshold.

### Drift and warning evidence

- PSI and Wasserstein distance for numerical features.
- Categorical proportion and missingness shifts.
- Prediction-distribution drift.
- Drift weighted by Logistic Regression coefficients or XGBoost importance.
- Warning sensitivity, precision, false alarms, missed events, and lead time.

### Retraining benefit

Retraining will be considered warranted only when it produces a practically
meaningful improvement on the same untouched future cohort and that improvement
is supported by uncertainty analysis.

## Reproducibility Principles

- Preserve raw source fields.
- Use documented, period-aware cleaning and validation rules.
- Keep annual artifacts while retaining quarterly cohort metadata.
- Use origination-time predictors only.
- Prevent future cohorts and immature labels from influencing earlier
  decisions.
- Fit preprocessing using training data only.
- Treat schema and disclosure changes separately from population drift.
- Keep claims proportional to the number of independent temporal cohorts.

## Documentation

The current Freddie Mac origination pipeline is based primarily on:

- [Freddie Mac Single-Family Loan-Level Dataset General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf)
- [Freddie Mac Release 47 disclosure changes](https://www.freddiemac.com/fmac-resources/research/pdf/disclosure-changes-summary.pdf)
- [Freddie Mac historical disclosure changes](https://capitalmarkets.freddiemac.com/crt/docs/pdfs/historical-disclosure-changes.pdf)

## Current Status

**Completed:** Freddie Mac origination and performance validation through
2015–2023, the eligible 24-month target, the 437,668-row modeling dataset,
and leakage-safe feature engineering with preprocessing fit on 2015–2017.

**Next:** literature-grounded modeling methodology, then train the two models
and evaluate reliability across 2018–2023.
