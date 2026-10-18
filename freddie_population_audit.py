"""Notebook 12 population audit. Read-only inputs; new summary directory only.

Run from the repository root:
    python freddie_population_audit.py
Or pass --root "E:\\capstone_project\\mortgage-model-reliability".
Requires the project's existing pandas/numpy/pyarrow environment.
Does not compute drift, train models, change labels, or assign exclusion reasons.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['SOURCE YEAR', 'LOAN IDENTIFIER']
COHORT = ['ORIGINATION YEAR', 'ORIGINATION QUARTER']
TARGET = 'TARGET_COMPOSITE_CREDIT_EVENT'
NUMERIC = {
    'FICO': 'CLASSIC FICO_CLEAN',
    'DTI': 'ORIGINAL DEBT-TO-INCOME (DTI) RATIO_CLEAN',
    'LTV': 'ORIGINAL LOAN-TO-VALUE (LTV)_CLEAN',
}
CATEGORIES = {
    'LOAN_PURPOSE': 'LOAN PURPOSE_CLEAN',
    'OCCUPANCY': 'OCCUPANCY STATUS_CLEAN',
    'CHANNEL': 'CHANNEL_CLEAN',
    'BORROWER_GROUP': 'NUMBER OF BORROWERS GROUP_CLEAN',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize(frame, label):
    frame = frame.copy()
    for column in [*KEYS, *COHORT]:
        require(column in frame, f'{label}: missing {column}')
    require(not frame[[*KEYS, *COHORT]].isna().any().any(),
            f'{label}: missing key/cohort values')
    for column in ['SOURCE YEAR', 'ORIGINATION YEAR']:
        values = pd.to_numeric(frame[column], errors='raise')
        require((values == np.floor(values)).all(), f'{label}: fractional year')
        frame[column] = values.astype(int)
    frame['LOAN IDENTIFIER'] = frame['LOAN IDENTIFIER'].astype('string')
    require(frame['LOAN IDENTIFIER'].str.strip().ne('').all(), f'{label}: blank ID')
    quarters = pd.to_numeric(frame['ORIGINATION QUARTER'].astype('string')
                             .str.strip().str.removeprefix('Q'), errors='raise')
    require(quarters.isin([1, 2, 3, 4]).all(), f'{label}: invalid quarter')
    frame['quarter'] = quarters.astype(int)
    require(frame['SOURCE YEAR'].eq(frame['ORIGINATION YEAR']).all(),
            f'{label}: source/origination year mismatch')
    require(not frame.duplicated(KEYS).any(), f'{label}: duplicate loan/year keys')
    frame['period'] = (frame['SOURCE YEAR'].astype(str) + '-' +
                       np.where(frame['quarter'] <= 2, 'H1', 'H2'))
    return frame


def key_match(left, right, label):
    joined = left[KEYS + ['period']].merge(
        right[KEYS + ['period']], on=KEYS, how='outer',
        suffixes=('_left', '_right'), indicator=True, validate='one_to_one')
    require(joined['_merge'].eq('both').all(), f'{label}: key sets differ')
    require(joined['period_left'].eq(joined['period_right']).all(),
            f'{label}: cohort assignments differ')


def summarize(frame):
    counts, numeric, categorical = [], [], []
    groups = list(frame.groupby('period', sort=True))
    groups.insert(0, ('DEVELOPMENT_2015_2017', frame[frame['SOURCE YEAR'] <= 2017]))
    for period, part in groups:
        n = len(part)
        k = int(part['included'].sum())
        counts.append({'period': period,
                       'role': 'development' if period.startswith(('2015', '2016', '2017', 'DEVELOPMENT')) else 'evaluation',
                       'origination_loans': n, 'included_loans': k,
                       'excluded_loans': n-k, 'inclusion_percentage': 100*k/n if n else np.nan})
        for feature, column in NUMERIC.items():
            values = pd.to_numeric(part[column], errors='raise')
            summaries = {}
            for status, mask in [('included', part['included']), ('excluded', ~part['included'])]:
                v = values[mask]
                observed = v.dropna()
                require(np.isfinite(observed).all(), f'{period}/{feature}: nonfinite value')
                summaries[status] = {'n': len(v), 'nonmissing_n': len(observed),
                                     'missing_percentage': 100*v.isna().mean() if len(v) else np.nan,
                                     'mean': observed.mean(), 'median': observed.median(),
                                     'q25': observed.quantile(.25), 'q75': observed.quantile(.75)}
            row = {'period': period, 'feature': feature}
            for status, stats in summaries.items():
                row.update({status+'_'+name: value for name, value in stats.items()})
            row['excluded_minus_included_mean'] = summaries['excluded']['mean']-summaries['included']['mean']
            row['excluded_minus_included_missing_pp'] = summaries['excluded']['missing_percentage']-summaries['included']['missing_percentage']
            numeric.append(row)
        for feature, column in CATEGORIES.items():
            # Explicit tagged missing level cannot collide with an observed value.
            values = part[column].astype('string').str.strip()
            levels = sorted(values.dropna().unique().tolist())
            for level in [None, *levels]:
                mask_level = values.isna() if level is None else values.eq(level).fillna(False)
                row = {'period': period, 'feature': feature,
                       'level': '(missing)' if level is None else level,
                       'is_missing_level': level is None}
                for status, mask in [('included', part['included']), ('excluded', ~part['included'])]:
                    denominator = int(mask.sum())
                    count = int((mask & mask_level).sum())
                    row[status+'_n'] = count
                    row[status+'_denominator'] = denominator
                    row[status+'_percentage'] = 100*count/denominator if denominator else np.nan
                row['excluded_minus_included_pp'] = row['excluded_percentage']-row['included_percentage']
                categorical.append(row)
    return pd.DataFrame(counts), pd.DataFrame(numeric), pd.DataFrame(categorical)


def run(root, output=None):
    data = root/'data'/'processed'/'Freddie_Mac'
    base = data/'model_baselines_2015_2017'
    oot = base/'oot_evaluation'
    model_path = data/'freddie_modeling_dataset_2015_2023_24m.parquet'
    pred_path = oot/'oot_predictions_2018_2023.parquet'
    summary_path = oot/'cohort_reliability_summary.csv'
    manifest_path = base/'model_manifest.json'
    origins = [data/f'origination_{year}_clean.parquet' for year in range(2015, 2024)]
    inputs = [*origins, model_path, pred_path, summary_path, manifest_path]
    for path in inputs:
        require(path.is_file(), f'Required input missing: {path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['development_years'] == [2015, 2016, 2017], 'Development years changed')
    require(manifest['horizon_months'] == 24 and manifest['target'] == TARGET, 'Target/horizon changed')
    require(sha(model_path) == manifest['input_file_sha256'], 'Modeling data changed since baseline fit')
    frames = []
    columns = ['LOAN IDENTIFIER', *COHORT, *NUMERIC.values(), *CATEGORIES.values()]
    for year, path in zip(range(2015, 2024), origins):
        frame = pd.read_parquet(path, columns=columns)
        frame['SOURCE YEAR'] = year
        frames.append(normalize(frame, str(year)))
    orig = pd.concat(frames, ignore_index=True)
    require(not orig.duplicated(KEYS).any(), 'Duplicate origination keys')
    require(set(orig.period) == {f'{y}-{h}' for y in range(2015, 2024) for h in ['H1', 'H2']},
            'Expected 18 half-year origination cohorts')
    model = normalize(pd.read_parquet(model_path, columns=[*KEYS, *COHORT, TARGET, 'HORIZON_MONTHS']), 'modeling')
    require(model['HORIZON_MONTHS'].eq(24).all(), 'Non-24-month labels present')
    require(model[TARGET].isin([0, 1]).all(), 'Invalid or missing outcomes')
    require(set(model['SOURCE YEAR']) == set(range(2015, 2024)), 'Modeling year coverage changed')
    dev = model[model['SOURCE YEAR'] <= 2017]
    require(len(dev) == manifest['development_rows'], 'Development row count changed')
    require(int(dev[TARGET].sum()) == manifest['development_events'], 'Development event count changed')
    pred = normalize(pd.read_parquet(pred_path, columns=[*KEYS, *COHORT, 'half_year', 'y_true']), 'predictions')
    require(pred['SOURCE YEAR'].between(2018, 2023).all(), 'Predictions include non-OOT years')
    require(pred['half_year'].eq(pred['period']).all(), 'Prediction half-year field mismatch')
    later = model[model['SOURCE YEAR'] >= 2018]
    key_match(later, pred, 'Modeling vs Notebook 11 predictions')
    labels = later[KEYS+[TARGET]].merge(pred[KEYS+['y_true']], on=KEYS, validate='one_to_one')
    require(labels[TARGET].eq(labels['y_true']).all(), 'Notebook 11 outcomes differ from modeling labels')
    coverage = model[KEYS+['period']].merge(orig[KEYS+['period']], on=KEYS, how='left',
                                         suffixes=('_model', '_orig'), indicator=True, validate='one_to_one')
    require(coverage['_merge'].eq('both').all(), 'Eligible loan absent from origination sample')
    require(coverage['period_model'].eq(coverage['period_orig']).all(), 'Eligible origination cohort mismatch')
    orig = orig.merge(model[KEYS].assign(included=True), on=KEYS, how='left', validate='one_to_one')
    orig['included'] = orig['included'].eq(True)
    summary = pd.read_csv(summary_path)
    selected = summary[summary['period_kind'].eq('half_year')].copy()
    expected = {f'{y}-{h}' for y in range(2018, 2024) for h in ['H1', 'H2']}
    require(len(selected) == 24 and not selected.duplicated(['period', 'model']).any(), 'Invalid half-year summary rows')
    require(set(selected['model']) == {'xgboost', 'logistic'}, 'Unexpected model families')
    for family in ['xgboost', 'logistic']:
        require(set(selected.loc[selected.model.eq(family), 'period']) == expected, f'{family}: missing cohorts')
    actual = pred.groupby('period').agg(actual_n=('y_true', 'size'), actual_events=('y_true', 'sum')).reset_index()
    joined = selected.merge(actual, on='period', validate='many_to_one')
    require(joined['n'].eq(joined['actual_n']).all(), 'Summary loan counts differ from predictions')
    require(joined['positives'].eq(joined['actual_events']).all(), 'Summary event counts differ from predictions')
    counts, numeric, categories = summarize(orig)
    paths_hashes = [{'name': str(p.relative_to(root)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in inputs]
    output = output or data/'population_audits'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = output.resolve()
    require(not output.exists(), f'Output already exists; refusing overwrite: {output}')
    # All input validation completes before writing summaries.
    output.mkdir(parents=True, exist_ok=False)
    counts.to_csv(output/'population_audit_counts.csv', index=False)
    numeric.to_csv(output/'population_audit_numeric.csv', index=False)
    categories.to_csv(output/'population_audit_categories.csv', index=False)
    payload = {'status': 'passed', 'scope': 'Population selection/reconciliation only; no drift metrics or thresholds',
               'origination_rows': len(orig), 'eligible_rows': len(model), 'development_rows': len(dev),
               'oot_rows': len(pred), 'oot_events': int(pred.y_true.sum()),
               'half_year_key_and_label_reconciliation': True, 'both_model_summary_counts_reconcile': True,
               'input_files': paths_hashes,
               'limitations': ['Exclusion reasons are not inferred from absence in the modeling dataset.',
                              'Summaries describe selection within annual sampled origination files, not the complete Freddie universe.',
                              'No claim of harmless selection or early-warning validity follows from reconciliation alone.',
                              'Missing values remain missing; no feature-contract imputation or category grouping is applied.',
                              'The development aggregate overlaps its six half-year rows; do not sum both.']}
    (output/'population_audit_validation.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    # Verify byte-identical inputs after the audit.
    require(all(sha(root/item['name']) == item['sha256'] for item in paths_hashes), 'An input changed during execution')
    print(counts.to_string(index=False))
    print(f'\nAudit passed. Four summary files saved to: {output}')
    print('Excluded means absent from the eligible modeling dataset; reasons are not inferred.')
    print('Share the four summary files for interpretation. Keep loan-level data local.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Repository root')
    parser.add_argument('--output', type=Path, help='New output directory; must not already exist')
    args = parser.parse_args()
    run(args.root.resolve(), args.output)
