# Mortgage Credit-Risk Model Reliability Under Changing Borrower Populations

MS Data Science Capstone Project  
Status updated: 21 October 2026

## Project Overview

Mortgage credit-risk models are developed using historical borrower and loan
data, but the populations on which they operate change over time. Population
drift can be detected before reliable credit outcomes become observable.
However, drift alone does not establish that a model has deteriorated, and
observed deterioration does not establish that retraining will improve its
performance on later loans.

This project develops and evaluates a practical framework for relating early,
unlabeled population changes to subsequent model reliability and for assessing
whether retraining produces meaningful improvement. Freddie Mac is the primary
study; Fannie Mae provides the planned independent external evaluation.

The submitted Project Overview Statement (POS) anchors the research question,
objectives and scope.

## Central Research Question

Can early, unlabeled drift signals identify population changes that will later
result in meaningful model degradation, and can those signals support
evidence-based retraining decisions?

## Research Scope and Objectives

1. Develop XGBoost as the primary mortgage credit-risk model and Logistic
   Regression as the secondary comparison, using the 2015–2017 Freddie Mac
   cohorts, the same predictors and the same 24-month target.
2. Measure population drift across the 2018–2023 cohorts relative to the
   2015–2017 development population.
3. Evaluate whether drift is associated with subsequent deterioration in
   discrimination and calibration.
4. Investigate signals available before corresponding outcomes are known.
   When meaningful deterioration is observed in XGBoost, compare the frozen
   model with a retrained model to assess improvement.
5. Evaluate whether the main findings are consistent in compatible Fannie Mae
   data.

The focus is borrower-population drift and model reliability. Direct
identification of concept drift is outside the study's scope. A weak or
inconsistent drift–degradation relationship is a valid research finding;
an unevaluated objective must still be identified as pending.

## Data and Evaluation Design

### Freddie Mac

- Source: Freddie Mac Single-Family Loan-Level Dataset annual samples.
- Development: 2015–2017.
- Internal model selection: fit candidates on 2015–2016 and evaluate on 2017;
  refit the selected pipelines on 2015–2017 and freeze.
- Out-of-time evaluation: 2018–2023.
- Sampled originations: 450,000 loans across nine annual samples.
- Validated eligible modeling dataset: 437,668 rows and 59 columns.
- Predictors: 29 origination-time features.
- Target: a fixed 24-month composite credit event, defined as 90+ day
  delinquency or a qualifying adverse terminal credit event within the
  target window.
- Temporal unit: 12 half-year evaluation cohorts are primary; six annual
  cohorts provide context and 24 quarterly cohorts provide sensitivity checks.

Two population views are retained for drift measurement: all sampled
originations and loans eligible for the modeling target. Each view has its
own pooled 2015–2017 reference. Model reliability is measured only on eligible
loans. Full-sample drift paired with eligible-loan reliability therefore
retains a documented population mismatch.

The sampled originations are not the entire Freddie Mac mortgage population.

### Historical Timing

The frozen-model evaluation is retrospective. Complete 24-month outcomes for
the 2015–2017 development cohorts were not all available at the beginning of
2018. Notebook 12 examines model readiness, first reporting and outcome
maturity using explicit timing proxies.

Reporting dates do not establish verified predictor publication dates.
Retrospectively determined target eligibility also limits claims about
population information available at historical decision dates.

### Fannie Mae

Fannie Mae Single-Family Loan Performance Data will support the secondary
external evaluation after the Freddie Mac methodology is fixed. Initial
structural inspection of one 2018-Q1 file is complete; feature and target
compatibility checks and external evaluation remain pending.

Differences in field definitions, available features, populations and outcomes
may restrict the analysis to comparable subsets.

## Completed Work Through Notebook 12

| Notebook | Purpose and completed work |
|---|---|
| 01 — Data understanding | Inspected the 31 origination fields and monthly performance structure; investigated source definitions, sentinels and data-quality issues. |
| 02 — 2015 ingestion | Established raw-preserving cleaning, ingestion and validation on the 2015 sample. |
| 03 — Multi-year origination validation | Applied the reusable origination pipeline across 2015–2023 and documented disclosure and definition changes. |
| 04 — Performance target development | Defined and checked the fixed 24-month composite credit-event target. |
| 05 — Multi-year performance validation | Applied the target contract across years, with coverage, event-component and saved-output checks. |
| 06 — Modeling dataset validation | Joined eligible targets to cleaned originations by source year and loan identifier; verified counts, uniqueness and label completeness. |
| 07 — Feature engineering | Defined 29 origination-time predictors and development-fitted preprocessing; excluded identifiers, cohort metadata and post-origination audit fields from model inputs. |
| 08 — Temporal granularity feasibility | Assessed cohort size and event support; selected half-year analysis with annual context and quarterly sensitivity. |
| 09 — 2018 target-pattern diagnostic | Investigated the unusual 2018 target pattern and documented diagnostic checks. |
| 10 — Baseline model development | Selected candidates using 2017, refitted on 2015–2017 and froze Logistic Regression and XGBoost pipelines with a model manifest. |
| 11 — Frozen out-of-time reliability | Scored 2018–2023 and evaluated discrimination, probability accuracy and calibration, with half-year bootstrap intervals. |
| 12 — Population drift and reliability | Measured drift, linked five selected signals to reliability, examined uncertainty and temporal sensitivity, audited information timing and evaluated exploratory candidate alerts. |

Internal model selection rebuilds preprocessing using 2015–2016 only.
Final frozen preprocessing and model pipelines use 2015–2017 only.
No 2018–2023 outcomes enter baseline model selection.

The literature review, including the subsequent Rotalini paper review,
informed the evaluation design. The contribution being developed is an
empirical evaluation and decision framework; no new drift statistic is claimed.

## Methods Implemented

### Population Drift

- **Numeric PSI:** development-reference decile bins, with duplicate cut points
  removed and outer boundaries extended to infinity. Count smoothing is 0.5
  per bin; 0.1 and 1.0 are sensitivity settings.
- **Wasserstein distance:** raw numeric distribution distance, with
  development-IQR scaling where available.
- **Total variation distance:** categorical distribution change, including
  missing values.
- **Missingness and category coverage:** separate numeric missing-rate changes,
  missingness flags and novel-category rates.
- Numeric distributions are measured before imputation. Constant-reference
  features use a dedicated change-from-constant percentage; unavailable
  comparisons remain explicit.

All 29 predictors were measured. The five selected signals used for linkage
and candidate alerts are FICO, DTI and LTV distribution change, channel
distribution change, and DTI missingness change.

### Model Reliability

- **ROC-AUC:** discrimination.
- **Average precision (AP):** precision–recall performance, using scikit-learn's
  average-precision implementation rather than trapezoidal PR-AUC.
- **Brier score:** overall probability prediction error, interpreted with
  event prevalence.
- **Signed risk gap:** observed event rate minus mean predicted probability;
  absolute risk gap describes the magnitude of that aggregate difference.
- **Frozen-bin calibration:** development-defined risk bins applied unchanged
  to later cohorts, with counts, events and Wilson intervals.
- **Bootstrap uncertainty:** 1,000 loan-level resamples for half-year
  reliability and subsequent changes from the descriptive 2018-H1 anchor.

Brier score is not a calibration-only measure. ECE, calibration slope/intercept
and threshold-based precision/recall are not primary implemented results in
this workflow.

### Drift–Reliability Linkage and Candidate Alerts

- Descriptive cohort plots and a complete exploratory Spearman-correlation
  grid across five signals, four reliability outcomes, both models and both
  population views.
- Leave-one-cohort-out sensitivity, PSI smoothing sensitivity and
  prevalence/time context.
- Annual and quarterly sensitivity using directly calculated cohort metrics,
  rather than averages of half-year distances or performance metrics.
- Reliability-change uncertainty relative to each model's own 2018-H1 result.
- Five fixed candidate alert cutoffs based on the maximum drift across the
  six development half-years.
- Exploratory alert evaluation across 11 later half-years, using three
  illustrative deterioration sizes per metric and reporting hits, false
  alarms, misses and correct non-alerts.

Candidate cutoffs were not optimized on evaluation outcomes. The deterioration
sizes are illustrative candidates, not validated business tolerances.

## Findings and Limits Through Notebook 12

Population drift does not consistently identify worsening model reliability.
Associations depend on the signal, cohort and metric. Ranking performance and
probability accuracy can move in different directions.

Candidate alerts did not demonstrate a dependable general-purpose warning
rule. Frequent alerts captured deterioration but also produced false alarms;
less frequent alerts could miss deterioration.

These conclusions are exploratory:

- The 2018–2023 outcomes have already been examined and cannot be described
  as an untouched warning-validation sample.
- Historical predictor publication dates remain unverified. Model readiness
  and outcome availability use timing proxies.
- Repeated models, population views, metrics and signals do not add
  independent temporal observations.
- Overlapping 24-month outcome windows and shared economic conditions limit
  temporal independence. Annual/quarterly agreement is sensitivity evidence,
  not independent replication.
- Bootstrap intervals are descriptive and pointwise. They do not capture
  model-fitting uncertainty or all shared shocks and temporal dependence.
- AP and Brier score require event-prevalence context.
- Statistical direction does not establish practical importance, and an
  uncertain change does not establish stable or harmless drift.

Verified historical warnings at cohort arrival, independent warning
validation and operational lead time remain unestablished. Retraining and
recalibration benefits have not been evaluated. No intervention benefit,
cost saving or automatic retraining trigger is claimed.

## Remaining Work

### 1. Notebook 13: Retraining Comparison

Establish a defensible chronological design before fitting an updated model:

- Specify decision dates and information-availability assumptions.
- Identify labels available for updating at each decision date.
- Define candidate practical deterioration and improvement criteria.
- Compare frozen and retrained XGBoost on the same separate, later
  evaluation loans using the existing reliability metrics and uncertainty.
- Preserve the original baseline models, target and feature definitions.

Existing explored evaluation cohorts cannot become independent confirmation
merely by defining a new comparison. Any reuse must be described as
exploratory. Recalibration may be supplementary, but does not replace the
POS-required retraining comparison when meaningful deterioration is observed.

### 2. Consolidate the Reusable Framework

Organize the reviewed workflow into configurable components with dataset
adapters and standardized outputs for input validation, population definition,
reference cohorts, drift measurement, later reliability, timing-aware warning
assessment and retraining comparison.

The process should transfer to compatible datasets. Freddie-specific feature
mappings, fitted models, reference distributions and numerical cutoffs must
not be assumed portable. The framework must permit an **insufficient evidence**
recommendation.

### 3. Fannie Mae External Evaluation

Verify compatible features, outcomes and cohort coverage; apply the fixed
workflow; and assess consistency of the main findings in the independent
dataset. Document restrictions and differences explicitly.

### 4. Final Synthesis

Answer each submitted POS objective, separating observed findings,
exploratory evidence, unresolved timing limitations and questions that could
not be evaluated.

## Repository Structure and Reproduction

- `notebooks/01`–`12`: the analysis sequence described above.
- `src/`: reusable origination, cohort, performance, modeling-dataset and
  feature-engineering modules.
- `freddie_population_audit.py`: population inclusion and timing audit.
- `tests/`: modeling-dataset and feature-engineering tests.
- `requirements.txt`: Python dependencies.
- `data/raw/`, `data/interim/`, `data/processed/`: local data directories.

Raw data, generated datasets, prediction files and model artifacts are
excluded from Git. Obtain source data from the official providers under
their access terms and run the notebooks in sequence with the expected
local inputs.

Saved outputs embedded in committed notebooks can be inspected on GitHub
without downloading the underlying data. Standalone generated CSV, JSON,
Parquet, PDF and model files are not supplied by cloning this repository;
they must be regenerated or provided separately under applicable data terms.
Review bundles contain aggregate summaries rather than loan-level records.

## Reproducibility Principles

- Preserve raw fields and document period-aware cleaning rules.
- Keep target definitions and predictor contracts consistent across cohorts.
- Fit preprocessing only on the applicable training population.
- Keep identifiers and post-origination information outside model inputs.
- Separate retrospective evaluation from historical information availability.
- Do not use future or immature labels for historical update decisions.
- Treat schema and disclosure changes separately from borrower-population drift.
- Retain cohort event counts, prevalence, uncertainty and unavailable results.
- Keep claims proportional to the available independent temporal evidence.

## Source Documentation

- [Freddie Mac Single-Family Loan-Level Dataset General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf)
- [Freddie Mac Release 47 disclosure changes](https://www.freddiemac.com/fmac-resources/research/pdf/disclosure-changes-summary.pdf)
- [Freddie Mac historical disclosure changes](https://capitalmarkets.freddiemac.com/crt/docs/pdfs/historical-disclosure-changes.pdf)

## Current Status

**Completed:** Freddie Mac preparation, baseline model development, frozen
out-of-time reliability, retrospective drift–reliability analysis, temporal
sensitivity, timing assessment and exploratory candidate-alert evaluation
through Notebook 12.

**Next:** design and evaluate the frozen-versus-retrained XGBoost comparison,
consolidate the reusable framework, and conduct the compatible Fannie Mae
external evaluation.
