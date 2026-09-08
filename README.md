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

- Logistic Regression as a transparent, interpretable benchmark.
- XGBoost for nonlinear relationships and feature interactions.

The objective is model reliability under temporal population change—not a broad
model-accuracy tournament.

## Data Strategy

### Freddie Mac

Freddie Mac Single-Family Loan-Level Dataset annual samples are used for model
development and temporal monitoring.

- 2015–2019: baseline development and temporal-validation window.
- 2020 onward: out-of-time drift and degradation monitoring.
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

### 2015–2019 origination harmonization

The reusable pipeline was successfully applied to five annual samples:

- 250,000 sampled loans processed.
- 50,000 loans per year.
- 31 raw fields preserved.
- 47-column cleaned analytical datasets created.
- No structural, undocumented-code, numeric-range, definition-timing, or
  ratio-rule violations detected.
- No missing or duplicate loan identifiers.
- Saved warning-report counts agree with validation JSON files.
- All five saved Parquet datasets pass read-back validation.

The analysis also records disclosure and definition changes so they are not
mistaken for ordinary borrower-population drift.

## Current Repository Structure

```text
notebooks/
├── 01_freddie_mac_data_understanding.ipynb
├── 02_ingest_2015_sample.ipynb
└── 03_freddie_multi_year_origination_validation.ipynb

src/
├── freddie_config.py
├── freddie_ingestion.py
├── freddie_cleaning.py
├── freddie_validation.py
├── freddie_quality.py
├── freddie_summary.py
└── freddie_pipeline.py
```

## Next Implementation Stages

- Derive and retain origination year, quarter, cohort, and documentation-rule
  period.
- Build reusable performance-file ingestion and validation.
- Research, define, and audit a consistent 24-month 90+ delinquency target.
- Create one leakage-safe modeling row per eligible loan.
- Train and temporally validate Logistic Regression and XGBoost.
- Freeze the 2015–2019 baseline models.
- Measure 2020+ population, prediction, and model-aware drift.
- Confirm later discrimination and calibration degradation after labels mature.
- Back-test early-warning signals without using future information.
- Compare frozen, calendar, conventional-drift, and evidence-based retraining
  policies on untouched future cohorts.
- Perform locked-method external validation with harmonized Fannie Mae data.

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

**Completed:** 2015 baseline origination EDA, reusable origination processing,
and documentation-aware validation and harmonization of the 2015–2019 annual
origination samples.

**Next:** cohort metadata integration, followed by reusable Freddie Mac
performance-data ingestion and validation.
