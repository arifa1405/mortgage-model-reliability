# Notebook 15B — Compatibility decision record

**Project:** Mortgage Credit-Risk Model Reliability Under Changing Borrower Populations  
**Prepared:** 25 November 2026  
**Evidence baseline:** feature branch `feature/freddie-retraining-feasibility`, commit `b55b78296bb16456b3d2bec1c274d27eb20d9d89`  
**Status:** EDA and its computational follow-up are reviewed. The researcher has authorized Notebook 16 companion development using Freddie only. Exact Fannie publication release and original download date were not retained; this is a provenance limitation, not a failed structural check or an automatic training/evaluation block. External scoring has not been approved or performed.

## 1. Decision summary

The Fannie candidate population has been constructed and examined. The raw-snapshot audit, eligibility reconciliation, numerical and categorical EDA, correlations, population distances and dollar-balance precision correction have passed their reviewed checks.

Use separate Freddie-developed companion models for the external study. Preserve the original 29-predictor Freddie models, loan populations, targets, preprocessing and historical findings. The proposed primary companion specification has 22 predictors; a separate 24-predictor specification adds the existing FICO proxy and its missingness indicator.

The external study is a retrospective comparison of related mortgage populations. Its central external comparison is Freddie versus Fannie actual 2018 Q1 originations. The Fannie population is only the subset contained in the available acquisition file. Matching origination quarter does not make the agency populations equivalent in product coverage, sampling or target semantics.

| Issue | Evidence and interpretation | Proposed decision / current status |
|---|---|---|
| Candidate dataset and counts | Notebook 15, executed 15B and the raw audit reconcile. | Construction and EDA checks complete. |
| Dollar-balance PSI precision | Section 11 corrected inverse-log boundary round-off; both old and corrected calculations reconcile with exported bin counts. | Use corrected Section 11 dollar-balance PSI in reporting. Preserve the earlier review bundles as superseded evidence for this one calculation. |
| Freddie extreme ratios | Preserved raw and cleaned values agree. Notebook 03 and the existing quality module already account for the broader HARP ranges. | Preserve existing values and rules. Do not classify the 587 calendar-only field flags as 587 erroneous loans. |
| Fannie raw CLTV | All 260,395 earliest snapshots matched; one missing/invalid CLTV; no observed CLTV below LTV. | Raw-CLTV EDA complete; retain the existing missingness treatment. |
| Fannie 56 conflicting histories | Distinct nonblank values change in selected fields; the earliest snapshot remains the implemented source. | Retain existing values and flags; disclose unresolved reasons. No silent replacement, future backfill or exclusion. |
| Primary and strict eligibility | The additional strict exclusions are 200 repurchases and one reporting gap, with no overlap between these two groups. | Keep both existing populations. Strict Fannie eligibility is a sensitivity, not a claim of matched strict Freddie eligibility. |
| Release provenance | Raw-file identity is fixed by hash and size; the original download history is unavailable. Official provider definitions and earlier positional audits are available. | Record the original publication release as unverified. Missing download date alone is not a training or external-evaluation gate. Evaluate actual retained-field definitions separately; do not infer release vintage from acquisition quarter, last reporting month or field count. |
| Credit-score equivalence | The existing proxy is the minimum available disclosed borrower/co-borrower score. Official disclosures changed during 2026. | Exclude FICO and its missingness flag from the primary specification. Freddie-only training of the 24-feature sensitivity is authorized; its Fannie interpretation remains conditional on score-mapping review. Do not substitute a different field silently. |
| Composite target semantics | Both implementations use numeric serious delinquency and selected terminal codes, but some event definitions differ across agencies. | Propose a mapped 24-month composite outcome with explicit differences; obtain approval of this interpretation before external scoring. Do not claim literal outcome equivalence. |
| Earliest-snapshot timing | The audit confirms which row was selected; it does not establish historical publication availability or prove that every field was known at origination. | Accept only as a disclosed retrospective measurement limitation, subject to milestone approval. No prospective deployment claim. |
| Population coverage | Fannie is an acquisition-file subset; Freddie is an annual sample. Fannie primary coverage excludes HARP whereas the Freddie research includes HARP. | Retain the existing study populations for the proposed pilot and disclose the coverage difference. Do not introduce a new matched-product filter without a separately reviewed specification. |

## 2. Reviewed local evidence

The record uses the aggregate review bundles supplied by the researcher and the committed notebook outputs. No local raw mortgage file or fitted model was available in the review workspace. Therefore, the review independently reconciled the supplied aggregate evidence; it did not independently rescan the researcher's source datasets.

| Population | Loans | Composite events |
|---|---:|---:|
| Fannie available actual 2018 Q1 originations | 260,395 | Not assigned to ineligible loans |
| Fannie primary eligible | 260,244 | 1,960 |
| Fannie strict eligible | 260,043 | 1,947 |
| Freddie development eligible, 2015–2017 | 143,598 | Existing approved Freddie outcomes preserved |
| Freddie 2018 Q1 eligible | 11,884 | Existing approved Freddie outcomes preserved |

Primary-to-strict exclusions remove 201 loans and 13 observed composite events. This reconciles the policies; it is not a basis for selecting an eligibility policy using model performance.

The corrected dollar-balance PSI values use the same reference bins and 0.5 count smoothing. The correction snaps reconstructed values within floating-point tolerance to a bin boundary for counting only; it does not alter model inputs.

| Comparison | Earlier PSI | Corrected PSI |
|---|---:|---:|
| Freddie development eligible → Fannie primary | 0.003464713913 | 0.002951999649 |
| Freddie 2018 Q1 eligible → Fannie primary | 0.002020646347 | 0.004788510099 |
| Freddie development eligible → Freddie 2018 Q1 eligible | 0.005348578388 | 0.005348578388 |

### Evidence fingerprints

| Evidence | SHA-256 |
|---|---|
| Reviewed 15B raw-audit ZIP, uploaded as `notebook_15b_eda_population_review(1).zip` | `e23a40436f7ca57abf3da8e95a899a9bcf7e67489a3ecc683629ae5794f92d3f` |
| Reviewed Section 11 ZIP, `notebook_15b_followup_review.zip` | `5f812b027ecaa9dc2c2356b892df323db59817309ef99a2c4dfb44b76f69e0d8` |
| Fannie raw source, as verified by the researcher's audit | `b89043e7c2967d8f50bb3d7f5f694ee624a77ec86e9390a6333ee9ccf5ed3ebe` |
| Fannie candidate Parquet identified by the reviewed run | `29aebd65b716fb1d21434cb1d3ab109da7640086a3a5b0ab5a39407c7c52c70f` |

The raw source has 6,928,164,686 bytes, 23,563,399 monthly records and 458,830 distinct loans in the acquisition file. These identify the reviewed input; they do not identify its publication release.

### Earlier structural inspection

The researcher subsequently supplied `10_fannie_2018q1_compatibility_inspection.ipynb`, an earlier local notebook absent from the searched GitHub branches. It performed seven positional spot checks and referenced the September 10, 2026 glossary. Its parser removed the leading blank field, shifting the mapping, and its streaming audit stopped at physical line 30. The failed loan-order check cannot independently establish noncontiguous loan histories because the loan-ID mapping was shifted. Treat this notebook as preliminary inspection, not successful full validation. Notebook 14 corrected positional alignment; 14B and subsequent audits provide the corrected full-file evidence. No repeat structural scan is required because the original download history is unavailable.

## 3. Correct interpretation of the Freddie ratios

The Section 11 check intentionally compared ratios with calendar-based reference bounds. It did not read HARP indicators or pre-HARP links. Its 587 eligible field-level flags are consequently a preliminary diagnostic count, with possible within-loan overlap, rather than the output of the complete existing quality policy.

The earlier interpretation that these were a newly unexplained Freddie data-quality problem is superseded by this reconciliation:

1. `src/freddie_quality.py::_create_ratio_warnings` already uses the 2018 Q2 transition and documented HARP status plus a pre-HARP link when assigning the broader ratio ranges.
2. Notebook 03's committed high-ratio table reports the following sampled-loan counts. Every loan in these groups has both HARP status Y and a pre-HARP link.

| Sample year | Loans with LTV or CLTV above 105% | HARP Y | Pre-HARP link present |
|---|---:|---:|---:|
| 2015 | 530 | 530 | 530 |
| 2016 | 227 | 227 | 227 |
| 2017 | 124 | 124 | 124 |
| 2018 | 34 | 34 | 34 |

3. Notebook 03's saved quality results report zero ratio warnings under the complete policy for 2015–2023. The extreme maxima in its high-ratio table agree with the later follow-up.
4. The currently retrieved Freddie guide explicitly gives HARP a 1–998% ratio range [S1].

This is a reconciliation with existing code and saved results, not a newly executed loan-by-loan HARP cross-tab of the 587 flags. It supports retaining the original handling. It does not establish that the property valuations are economically accurate or that the source is error-free. Companion input checks should reuse the existing HARP-aware quality rules rather than the simplified calendar-only flag count.

Freddie–Fannie coverage remains different: the Fannie primary research dataset excludes HARP [S2]. Removing HARP from Freddie would define a new comparison population and cannot be done silently.

## 4. Proposed companion predictor contract

### Primary: 22 predictors, in reviewed order

The order below reproduces the reviewed 14G proposal and Notebook 15 metadata. The rules describe existing candidate transformations; they do not certify release-specific equivalence.

| Order | Predictor | Type | Mapping or treatment |
|---:|---|---|---|
| 1 | `MI_PERCENT` | Numeric | Reviewed insurance-percentage mapping; distinguish no insurance from unavailable coverage. |
| 2 | `ORIGINAL_DTI` | Numeric | Reviewed cleaned origination DTI; retain missing values before preprocessing. |
| 3 | `ORIGINAL_LTV` | Numeric | Reviewed cleaned origination LTV. |
| 4 | `ORIGINAL_INTEREST_RATE` | Numeric | Original contract rate in percentage units; do not use current rate. |
| 5 | `LOG_ORIGINAL_UPB` | Numeric | `log1p` of original balance; do not use current balance. |
| 6 | `CLTV_MINUS_LTV` | Numeric | Existing clipped nonnegative difference after cleaning. |
| 7 | `HAS_MORTGAGE_INSURANCE` | Binary numeric | Preserve the existing `MI_PERCENT > 0` rule; interpret alongside the missingness flag. |
| 8 | `HAS_SUBORDINATE_FINANCING` | Binary numeric | Preserve the existing `CLTV > LTV` rule and its existing missing-input behavior. |
| 9 | `HIGH_LTV_GT_80` | Binary numeric | Preserve the existing `LTV > 80` rule. |
| 10 | `MI_PERCENT_MISSING` | Binary numeric | Missingness after the reviewed insurance mapping. |
| 11 | `ORIGINAL_DTI_MISSING` | Binary numeric | Missingness before imputation. |
| 12 | `ORIGINAL_LTV_MISSING` | Binary numeric | Missingness before imputation. |
| 13 | `CLTV_MISSING` | Binary numeric | Missingness of cleaned source CLTV. |
| 14 | `NUMBER_OF_UNITS` | Categorical | Canonical integer category spelling. |
| 15 | `OCCUPANCY_STATUS` | Categorical | Reviewed primary/second-home/investment categories. |
| 16 | `CHANNEL` | Categorical | Reviewed retail/broker/correspondent categories; monitor unknown levels. |
| 17 | `AMORTIZATION_TYPE` | Categorical | Preserve for contract consistency; observed fixed-rate constancy is disclosed. |
| 18 | `PROPERTY_STATE` | Categorical | Reviewed state categories; handle unseen categories through the frozen Freddie pipeline. |
| 19 | `PROPERTY_TYPE` | Categorical | Reviewed shared property categories. |
| 20 | `ORIGINAL_LOAN_TERM` | Categorical | Months, with canonical integer category spelling. |
| 21 | `NUMBER_OF_BORROWERS_GROUP` | Categorical | Reviewed 1 versus 2+ grouping. |
| 22 | `PURCHASE_REFINANCE_GROUP` | Categorical | Freddie P → purchase, C/N → refinance; Fannie P → purchase, C/R → refinance. Preserve unknowns; do not assert exact fine-category equivalence. |

The primary contract contains 13 numeric/binary predictors and nine categorical predictors. It replaces direct transfer of the original model, not the original Freddie research itself.

### FICO sensitivity: 24 predictors

Append `FICO_SCORE` and `FICO_SCORE_MISSING`, in that order. The Fannie score remains the already constructed minimum-available disclosed-score proxy, conditional on score-mapping review for external interpretation. Training the corresponding Freddie companion does not certify that proxy. Do not backfill it from later reporting rows or replace it with field 111, a current score, or a newer score model without a documented new mapping and approval.

The original `FIRST_TIME_HOMEBUYER`, fine-grained `LOAN_PURPOSE`, `SUPER_CONFORMING_FLAG`, `SPECIAL_ELIGIBILITY_PROGRAM`, `HARP_INDICATOR` and `INTEREST_ONLY_INDICATOR` are not primary companion inputs. Their omission does not mean their underlying populations or definitions have become equivalent.

Mortgage insurance remains conditional on the reviewed joint use of percentage and MI-type fields. The Fannie glossary identifies null MI type as no MI [S3]; the implementation retains separate unavailable-coverage treatment. A false binary insurance flag alone must not be interpreted as independently verified uninsured status.

## 5. Proposed outcome and eligibility decisions

Preserve the approved first-scheduled-payment window, calendar months 0–23, and reporting-month terminal timing. The Fannie working composite is numeric delinquency at least three months or a selected terminal code in that window. The original Freddie serious-event implementation also includes RA; it remains unchanged.

| Component | Freddie implementation / terminology | Fannie working mapping | Interpretation |
|---|---|---|---|
| Numeric serious delinquency | Numeric status at least 3 | Numeric status at least 3 | Operationally related serious-delinquency component. |
| RA / unknown | RA contributes to the existing serious-event flag | XX remains unknown | No RA-to-XX equivalence. |
| 02 | Third-party sale | Third-party sale | Retain selected adverse-terminal mapping. |
| 03 | Short sale or charge-off | Short sale | Different event coverage. |
| 09 | REO disposition | Deed-in-lieu or REO disposition | Different event coverage. |
| 15 | Whole-loan sale terminology | Nonperforming note sale | Same code number does not establish identical selection semantics. |
| 06 | No direct counterpart asserted here | Repurchase | Separate observation/eligibility issue; not automatically a negative outcome. |

The code mappings are supported by the repository implementations and the agency glossaries [S1, S3]. Proposed reporting language is **mapped 24-month composite credit-event outcome**, with these differences disclosed. Do not describe the agencies' labels as identical or claim that their performance difference measures a pure agency effect.

Keep existing primary and strict Fannie eligibility separately. Primary eligibility is based on observation start and completion/termination; it does not guarantee a complete monthly history. The strict rule additionally excludes required-month gaps, unknown active status and in-window repurchases. Among the 201 additional strict exclusions are 13 observed composite events; do not tune this rule using predictive results.

A purely numeric serious-delinquency sensitivity is optional. If later approved, reconstruct numeric-only outcomes for both datasets and train corresponding Freddie-only companion models. The original Freddie `TARGET_90_PLUS` cannot be assumed numeric-only because its implementation includes RA. Do not relabel evaluation outcomes while presenting the composite-trained model as a numeric-only model.

## 6. Earliest snapshots and conflicting fields

Retain the implemented earliest available reporting snapshot. The 56 histories with conflicting nonblank origination fields remain flagged and included under both existing eligibility policies. The audit identified fields and reproduced counts; it did not resolve why values changed or which value is correct.

Changes between blank and populated borrower scores among flagged histories are distinct from conflicts between two nonblank scores. Do not use later availability to fill earlier values. These audit findings are not evidence that every retained value was published at the historical origination date.

The proposed acceptance is limited to a retrospective study of disclosed records. Do not infer verified real-time feature availability, deployment readiness or independent prospective warning validity. Any conflict-exclusion sensitivity would need an explicit outcome-independent rule and separate approval; it is not added to the primary analysis by this record.

## 7. Release identification: a documented provenance limitation

The 2024 glossary alone is insufficient to certify a 113-field local file. Fannie announced additional score fields and a transition in 2026 [S4]. The currently retrieved glossary is dated 9 November 2026 and documents the legacy origination score fields and newer score fields [S3]. A further announced SF dataset change is scheduled for January 2027; its CRT timing is different [S5]. These schedules must not be conflated.

**Inference, not a verified local-release identification:** the observed 113-field structure is compatible with a disclosure generation that contains additional score fields. It does not prove the publication date, release identifier or applicability of every current glossary note to the local file.

The primary 22-feature proposal limits dependence on uncertain score equivalence. It does not, by itself, close provenance for the other retained fields and target codes.

The researcher no longer has the original download-history entry or a retained release-specific glossary. Record the exact original publication release and download date as unverified/not retained. A Windows modification date or an acquisition-quarter filename does not prove either. No further search, redownload or replacement of the established data is required merely to recover the date.

This is a reproducibility note, not a finding that the data failed validation. The source hash, structural audits, reviewed retained-field definitions and transformations identify what was actually analyzed. External readiness depends on defensible predictor mapping, outcome interpretation and population coverage; the missing date alone is not an automatic scoring block. Do not mark the entire compatibility assessment as failed solely because release-identification metadata is unavailable.

## 8. Notebook 16 and external-scoring gates

| Next activity | Protocol / gate |
|---|---|
| Notebook 16 development | Authorized by the researcher: implement the 22-feature primary and separate 24-feature FICO sensitivity using Freddie only. Execute locally and review the generated aggregate bundle before external evaluation. |
| Companion development | Once approved, use Freddie only: train on 2015–2016, validate/select on 2017, then refit on 2015–2017. Keep the original models and original research populations intact. |
| Preprocessing | Fit imputation, category handling and any scaling on the applicable Freddie training population only. Freeze ordering and manifests. No Fannie/out-of-time fitting or outcome-based model selection. |
| Models | Logistic Regression and XGBoost for each approved predictor specification, stored separately from the original 29-feature artifacts. |
| Outlier handling | Reuse existing documented cleaning and HARP-aware checks. Do not drop or cap historical ratios using the simplified 587-flag count. |
| Population interpretation | Full Freddie and available Fannie populations differ in sampling and products. Any narrower product-matched population is a separate reviewed comparison, not an automatic change. |
| Notebook 17 scoring | Requires reviewed retained-field mappings, an approved mapped-target interpretation, approved population/predictor specifications, frozen companion artifacts and explicit evaluation approval. Disclose the unretained original download date/publication release; their absence alone does not block scoring. |
| Generalizability | One acquisition-file subset does not validate the complete Freddie 2018–2023 drift, early-warning or retraining experiment. |

**Disposition:** close the numerical/categorical EDA, raw-snapshot audit and precision follow-up as reviewed. Retain the retrospective, population and provenance limitations. Proceed with authorized Freddie-only companion development in Notebook 16. Review internal development outputs and the external interpretation before Notebook 17. External scoring remains unreleased.

Preparing the original 15B record performed no new raw-data scan, data correction, model training or scoring. The subsequent Notebook 16 implementation saves separate artifacts and has been checked with synthetic data; actual research-model development runs on the researcher's local files. No approval status in an earlier execution manifest was changed.

## 9. Source register

Provider links were checked on 25 November 2026. Live provider URLs may change their contents; retain a release-matched copy locally when available. The currently retrieved documents are interpretive evidence, not proof of the downloaded files' release versions.

### Project sources at the evidence baseline

- [Notebook 03: earlier origination validation and HARP analysis](https://github.com/arifa1405/mortgage-model-reliability/blob/b55b78296bb16456b3d2bec1c274d27eb20d9d89/notebooks/03_freddie_multi_year_origination_validation.ipynb)
- [Existing Freddie quality rules](https://github.com/arifa1405/mortgage-model-reliability/blob/b55b78296bb16456b3d2bec1c274d27eb20d9d89/src/freddie_quality.py)
- [Existing Freddie performance/target implementation](https://github.com/arifa1405/mortgage-model-reliability/blob/b55b78296bb16456b3d2bec1c274d27eb20d9d89/src/freddie_performance.py)
- [Notebook 14G: candidate decisions](https://github.com/arifa1405/mortgage-model-reliability/blob/b55b78296bb16456b3d2bec1c274d27eb20d9d89/notebooks/14g_fannie_final_compatibility_decisions.ipynb)
- [Notebook 15: candidate construction](https://github.com/arifa1405/mortgage-model-reliability/blob/b55b78296bb16456b3d2bec1c274d27eb20d9d89/notebooks/15_fannie_external_dataset_construction.ipynb)
- [Notebook 15B: EDA and computational follow-up](https://github.com/arifa1405/mortgage-model-reliability/blob/b55b78296bb16456b3d2bec1c274d27eb20d9d89/notebooks/15b_fannie_eda_population_comparison.ipynb)

### Provider documentation

- **[S1]** [Freddie Mac General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf), retrieved version: Release 48, October 2026; origination ratio ranges and performance-code definitions.
- **[S2]** [Fannie Mae Loan Performance Dataset FAQs](https://capitalmarkets.fanniemae.com/media/8921/display), retrieved version dated 10 March 2026; primary/HARP population coverage and acquisition-quarter organization.
- **[S3]** [Fannie Mae CRT and Single-Family Loan Performance Glossary](https://capitalmarkets.fanniemae.com/media/6931/display), retrieved version dated 9 November 2026; retained field definitions, score-field transitions, MI type and terminal-code definitions.
- **[S4]** [Fannie Mae disclosure-enhancement announcement](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/enhance-credit-risk-transfer-disclosures), 17 November 2025; distinguishes 2026 CRT and SF dataset publication timing.
- **[S5]** [Fannie Mae multiple-credit-score disclosure announcement](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-announces-crt-disclosure-enhancements-support-multiple-credit-score-models), 22 October 2026; distinguishes November 2026 CRT changes from the January 2027 SF dataset publication.
- **[S6]** [Fannie Mae July 2024 glossary used in the original diagnostic work](https://capitalmarkets.fanniemae.com/sites/capmrkt/files/2024-07/crt-file-layout-and-glossary.pdf), 29 July 2024; historical comparison only, not the sole release-certification source.
