"""Recalculate receptor-macro metrics from unchanged revision score exports.

No neural model is loaded or trained by this script. Python 3.10+ standard library.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ('recall_at_20', 'pair_mrr', 'ep_at_20', 'ef_at_20')


def read_csv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/metrics')
    parser.add_argument('--score-index', type=Path, default=ROOT/'score_index.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; existing results are not overwritten.')
    specs = json.loads(args.score_index.read_text())
    candidates = {r['candidate_name_1369'] for r in read_csv(ROOT/'inputs/common/candidates.csv')}
    labels = read_csv(ROOT/'inputs/common/labels_234.csv')
    targets = {r['hgnc_symbol'] for r in labels}
    assert len(candidates)==1310 and len(targets)==106 and len(labels)==234
    a = read_csv(ROOT/'inputs/cohorts/drugban_train_pair_excluded_104_229_labels.csv')
    b = read_csv(ROOT/'inputs/cohorts/cpi_source_receptor_excluded_42_104_labels.csv')
    a_ids = {r['pair_id'] for r in a}; b_ids = {r['pair_id'] for r in b}
    cohorts = {'full':labels, 'A_pair_excluded':a, 'B_receptor_excluded':b,
               'C_joint_derived':[r for r in labels if r['pair_id'] in a_ids & b_ids]}
    expected = {'full':(106,234),'A_pair_excluded':(104,229),'B_receptor_excluded':(42,104),'C_joint_derived':(41,101)}
    for key, rows in cohorts.items():
        assert (len({r['hgnc_symbol'] for r in rows}),len(rows))==expected[key]
    pair_rows=[]; receptor_rows=[]; run_rows=[]
    for spec in specs:
        path = ROOT/spec['scores']
        scores=defaultdict(dict)
        for row in read_csv(path):
            t,c=row['hgnc_symbol'],row['candidate_name_1369']; value=float(row['score'])
            assert t in targets and c in candidates and c not in scores[t] and math.isfinite(value)
            scores[t][c]=value
        assert set(scores)==targets and all(set(s)==candidates for s in scores.values())
        ranges={}
        for t, sm in scores.items():
            counts=defaultdict(int)
            for v in sm.values(): counts[v]+=1
            offset=0; bounds={}
            for value in sorted(counts, reverse=True):
                bounds[value]=(offset+1,offset+counts[value]);offset+=counts[value]
            ranges[t]={c:bounds[v] for c,v in sm.items()}
        pair_metrics={}
        for row in labels:
            t,c=row['hgnc_symbol'],row['candidate_name_1369']
            first,last=ranges[t][c]; width=last-first+1
            hit=max(0,min(20-first+1,width))/width
            rr=sum(1/r for r in range(first,last+1))/width
            pair_metrics[row['pair_id']]=(hit,rr)
            pair_rows.append(dict(model=spec['model'],unit=spec['unit'],pair_id=row['pair_id'],receptor=t,candidate=c,
                                  rank_min=first,rank_max=last,p_top20=hit,expected_reciprocal_rank=rr))
        for cohort, rows in cohorts.items():
            grouped=defaultdict(list)
            for row in rows:grouped[row['hgnc_symbol']].append(row)
            records=[]
            for t, vals in grouped.items():
                hits=sum(pair_metrics[r['pair_id']][0] for r in vals)
                rec=dict(model=spec['model'],unit=spec['unit'],cohort=cohort,receptor=t,
                         receptor_class=vals[0]['receptor_class'],n_pairs=len(vals),recall_at_20=hits/len(vals),
                         pair_mrr=statistics.mean(pair_metrics[r['pair_id']][1] for r in vals),
                         ep_at_20=hits/20,ef_at_20=hits/len(vals)*1310/20)
                records.append(rec);receptor_rows.append(rec)
            for stratum in ('all','gpcr','nhr'):
                selected=[r for r in records if stratum=='all' or r['receptor_class']==stratum]
                if not selected:continue
                run_rows.append(dict(model=spec['model'],unit=spec['unit'],replicate_type=spec['replicate_type'],
                                     cohort=cohort,stratum=stratum,n_receptors=len(selected),n_pairs=sum(r['n_pairs'] for r in selected),
                                     **{m:statistics.mean(r[m] for r in selected) for m in METRICS}))
        print(spec['model'],spec['unit'],'verified',len(targets)*len(candidates),'scores')
    groups=defaultdict(list)
    for row in run_rows:groups[(row['model'],row['cohort'],row['stratum'])].append(row)
    summary=[]
    for (model,cohort,stratum),rows in groups.items():
        for metric in METRICS:
            vals=[r[metric] for r in rows]
            summary.append(dict(model=model,cohort=cohort,stratum=stratum,metric=metric,n_runs=len(vals),
                                n_receptors=rows[0]['n_receptors'],n_pairs=rows[0]['n_pairs'],
                                mean=statistics.mean(vals),sample_sd=statistics.stdev(vals) if len(vals)>1 else ''))
    args.output.mkdir(parents=True)
    for name,rows in [('pair_ranks',pair_rows),('receptor_metrics',receptor_rows),('per_run',run_rows),('summary',summary)]:
        write_csv(args.output/(name+'.csv'),rows)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/s['scores'] for s in specs]}
    (args.output/'run_manifest.json').write_text(json.dumps(dict(mode='saved_score_reanalysis_not_model_inference',
        score_sha256=hashes,tie_rule='exact exported decimal equality; expected random order within ties',
        C_mask='intersection of the provided A and B pair IDs; newly derived, not recovered historical file'),indent=2)+'\n')
    print('Results:',args.output)

if __name__=='__main__':main()
