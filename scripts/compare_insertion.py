"""Summarize and plot matched baseline/policy evaluation JSON files."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=Path('runs/insertion_v4'))
    a=p.parse_args()
    baseline=json.loads((a.run_dir/'test_baseline.json').read_text())
    policy=json.loads((a.run_dir/'test_policy.json').read_text())
    b=baseline['records']; r=policy['records']
    if [x['seed'] for x in b]!=[x['seed'] for x in r] or baseline['configuration']!=policy['configuration']:
        raise ValueError('Comparison requires identical seeds and configuration')
    times=np.array([[x['seconds'] for x in rows] for rows in (b,r)])
    forces=np.array([[x['peak_load_n'] for x in rows] for rows in (b,r)])
    delta=times[1]-times[0]
    rng=np.random.default_rng(0)
    ci=np.percentile(rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1),[2.5,97.5])
    result={'episodes':len(b),'baseline_successes':sum(x['success'] for x in b),
            'policy_successes':sum(x['success'] for x in r),'mean_seconds_baseline_policy':times.mean(axis=1).tolist(),
            'mean_time_change_seconds':float(delta.mean()),'paired_bootstrap_95_ci_seconds':ci.tolist(),
            'policy_faster_trials':int((delta<0).sum()),'mean_peak_load_baseline_policy':forces.mean(axis=1).tolist(),
            'max_peak_load_baseline_policy':forces.max(axis=1).tolist()}
    (a.run_dir/'comparison.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,values,label in zip(axes,(times,forces),('Completion time (s; failures use timeout)','Peak summed contact load (N)')):
        ax.scatter(values[0],values[1],alpha=.7)
        maximum=values.max()*1.05
        ax.plot([0,maximum],[0,maximum],'--',color='gray')
        ax.set(xlabel='Baseline',ylabel='RL policy',title=label)
        ax.grid(alpha=.2)
    fig.suptitle(f'{len(b)} matched unseen starts — below dashed line favors RL')
    fig.savefig(a.run_dir/'comparison.png',dpi=160)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
