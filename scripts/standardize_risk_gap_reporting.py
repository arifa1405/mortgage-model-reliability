"""Create versioned aggregate-only reporting corrections for Notebook 16/17.
Usage: python scripts/standardize_risk_gap_reporting.py NB16.zip NB17.zip OUTPUT_DIR
No model fitting/scoring, no raw data, no mutation of original review bundles.
Requires numpy and pandas for reconciliation checks.

The reviewed Notebook 16 and 17 runs use predicted minus observed internally.
Do not edit their protected source modules or historical manifests to change this.
Use the corrected review bundles for cross-notebook reporting.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, zipfile
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np
import pandas as pd

OLD = 'risk_gap_pred_minus_observed'
NEW = 'risk_gap_obs_minus_predicted'
DEFINITION = 'Observed event rate minus mean predicted probability; positive = underprediction.'

def sha(b): return hashlib.sha256(b).hexdigest()
def negate(s):
    if s == '': return s
    assert Decimal(s).is_finite()
    return s[1:] if s.startswith('-') else '-' + s.lstrip('+')

def convert_csv(raw, stage):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    fields = reader.fieldnames
    old = 'risk_gap' if stage == 16 else OLD
    wide = old in fields
    long = 'metric' in fields
    if not wide and not long: return raw, 0
    target_fields = [NEW if c == old and wide else c for c in fields]
    rows, count = [], 0
    for row in reader:
        original = row.copy()
        if wide:
            x = row.pop(old)
            row[NEW] = negate(x)
            if x and {'prevalence','mean_predicted'} <= row.keys():
                assert abs(Decimal(row[NEW]) - (Decimal(row['prevalence'])-Decimal(row['mean_predicted']))) < Decimal('1e-12')
            if 'absolute_risk_gap' in row and x:
                assert abs(Decimal(row[NEW])) == Decimal(row['absolute_risk_gap'])
            assert all(row[c] == original[c] for c in fields if c != old)
            count += 1
        if long and row['metric'] == old:
            row['metric'] = NEW
            row['estimate'] = negate(original['estimate'])
            row['ci_low'], row['ci_high'] = negate(original['ci_high']), negate(original['ci_low'])
            if row['ci_low'] and row['ci_high']:
                assert Decimal(row['ci_low']) <= Decimal(row['ci_high'])
            assert all(row[c] == original[c] for c in fields if c not in ('metric','estimate','ci_low','ci_high'))
            count += 1
        rows.append(row)
    if not count: return raw, 0
    buf = io.StringIO(newline=''); writer = csv.DictWriter(buf, fieldnames=target_fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return buf.getvalue().encode('utf-8'), count

def load_csv(files, name): return pd.read_csv(io.BytesIO(files[name]), float_precision='round_trip')

def reconcile_17(files):
    draws = load_csv(files,'bootstrap_metric_draws.csv')
    pt = load_csv(files,'performance_point_estimates.csv').set_index(['population','model'])
    assert not pt.index.duplicated().any()
    matched = load_csv(files,'matched_2018q1_performance.csv')
    pd.testing.assert_frame_equal(matched.set_index(['population','model']).sort_index(), pt.loc[matched.set_index(['population','model']).index].sort_index())
    grouped = {(p,m):g.set_index('draw_index')[NEW].sort_index() for (p,m),g in draws.groupby(['population','model'])}
    assert len(draws)==60000 and len(grouped)==60 and all(len(v)==1000 and not v.index.duplicated().any() for v in grouped.values())
    checks = {}
    for file in ['bootstrap_metric_intervals.csv','paired_model_difference_intervals.csv','cross_population_difference_intervals.csv']:
        rows=load_csv(files,file); rows=rows.loc[rows.metric.eq(NEW)]; checks[file]=len(rows)
        for r in rows.itertuples(index=False):
            if file.startswith('bootstrap'):
                a=(r.population,r.model); b=None
            elif file.startswith('paired'):
                left,right=r.comparison.split('_minus_'); a=(r.population,left); b=(r.population,right)
            else:
                if r.comparison=='Fannie_strict_minus_primary': left,right='Fannie 2018Q1 strict','Fannie 2018Q1 primary'
                else: left,right=r.comparison.split('_minus_')
                a=(left,r.model); b=(right,r.model)
            vals=grouped[a] if b is None else grouped[a]-grouped[b]
            estimate=pt.loc[a,NEW] if b is None else pt.loc[a,NEW]-pt.loc[b,NEW]
            valid=vals[np.isfinite(vals)]
            assert len(valid)==r.valid_draws and len(vals)-len(valid)==r.invalid_draws
            np.testing.assert_allclose([r.estimate,r.ci_low,r.ci_high],[estimate,*np.quantile(valid,[.025,.975])],rtol=0,atol=1e-12)
    return checks

def correct(src, stage, out):
    manifest_name='model_manifest.json' if stage==16 else 'model_evaluation_manifest.json'
    with zipfile.ZipFile(src) as z:
        assert z.testzip() is None
        files={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
    historical=files.pop(manifest_name); manifest=json.loads(historical)
    expected={k.replace('\\','/'):v for k,v in manifest['output_sha256'].items()}
    for name,raw in files.items():
        assert name in expected and sha(raw)==expected[name], f'Unverified input: {name}'
    absent=sorted(set(expected)-set(files))
    assert all(n.endswith(('.joblib','.parquet')) for n in absent), absent
    corrected, counts={},{}
    for name,raw in files.items():
        if name.endswith('.csv'):
            corrected[name], n=convert_csv(raw,stage)
            if n: counts[name]=n
        elif name=='evaluation_protocol.json' and stage==17:
            d=json.loads(raw); assert OLD in d['metric_definitions']
            d['metric_definitions'].pop(OLD); d['metric_definitions'][NEW]=DEFINITION
            corrected[name]=(json.dumps(d,indent=2)+'\n').encode()
            corrected['provenance/original_evaluation_protocol.json']=raw
        elif name=='review_notes.md':
            corrected['provenance/original_review_notes.md']=raw
            s=raw.decode()
            if stage==17:
                old='Risk gap here is predicted minus observed; this is the\nopposite sign of the original Notebook 11 risk-gap column.'
                s=s.replace('\r\n','\n'); assert old in s
                s=s.replace(old,'Risk gap in these reporting tables is observed minus predicted, matching Notebook 11.')
            s += '\n\nReporting-only correction: '+DEFINITION+'\nHistorical run and its original manifest are retained under provenance. No rerun or model update occurred.\n'
            corrected[name]=s.encode()
        else: corrected[name]=raw
    checks=reconcile_17(corrected) if stage==17 else {}
    if stage==16:
        a=load_csv(corrected,'development_2017_candidate_metrics.csv'); b=load_csv(corrected,'development_2017_selected_metrics.csv')
        for (s,f),g in a.groupby(['specification','family']):
            q=g.loc[(g.average_precision.max()-g.average_precision)<.005].assign(brier_6=lambda d:d.brier.round(6)).sort_values(['brier_6','candidate_order']).iloc[0]
            assert int(q.candidate_order)==int(b.loc[b.specification.eq(s)&b.family.eq(f),'candidate_order'].iloc[0])
        checks['unchanged_selected_candidates']=len(b)
    corrected['provenance/'+manifest_name]=historical
    result={ 'stage':str(stage),'status':'reporting_correction_verified_from_aggregate_exports','created_utc':datetime.now(timezone.utc).isoformat(),
      'source_review_zip':src.name,'source_review_zip_sha256':sha(src.read_bytes()),'original_manifest_sha256':sha(historical),
      'metric_definition':{NEW:DEFINITION},'transformation':'Negate signed gap and signed differences; interval [L,U] becomes [-U,-L]. All other CSV cells retained exactly.',
      'original_artifacts_not_modified':True,'models_refitted':False,'predictions_regenerated':False,'bootstrap_resampled':False,
      'local_only_artifacts_not_available_for_verification':absent,'original_included_output_hashes_verified':len(files),
      'corrected_rows_by_file':counts,'independent_reconciliation_checks':checks,
      'output_sha256':{n:sha(v) for n,v in sorted(corrected.items())}}
    corrected['reporting_correction_manifest.json']=(json.dumps(result,indent=2)+'\n').encode()
    dest=out/f'notebook_{stage}_standardized_risk_gap_review.zip'
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,raw in sorted(corrected.items()):z.writestr(n,raw)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None and all(sha(z.read(n))==v for n,v in result['output_sha256'].items())
    print(json.dumps({'output':str(dest),'corrected_rows':counts,'checks':checks,'verified_original_included_outputs':len(files),'local_only_not_verified':absent},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('notebook16',type=Path); p.add_argument('notebook17',type=Path); p.add_argument('output_dir',type=Path); a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    correct(a.notebook16,16,a.output_dir); correct(a.notebook17,17,a.output_dir)
if __name__=='__main__': main()
