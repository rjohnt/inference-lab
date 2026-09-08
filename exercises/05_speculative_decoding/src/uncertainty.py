"""Paired bootstrap of the ratio of median throughput across repetitions."""
import collections,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);rows=[json.loads(l) for l in (root/'results.jsonl').read_text().splitlines()];g=collections.defaultdict(dict)
for r in rows:g[r['case'],r['mode']][r['rep']]=r['timings']['predicted_per_second']
rng=np.random.default_rng(42);lines=['# Throughput variability','', '95% paired bootstrap intervals for the ratio of medians, resampling repetition blocks 10,000 times. These describe timing variation in this six-prompt suite on this host; they do not cover workload diversity or systematic host interference.','', '| Prompt | Method | N | Median TPS | Min–max TPS | Speedup | 95% interval |','|---|---|---:|---:|---:|---:|---:|']
for case in sorted({r['case'] for r in rows}):
 for mode in ['none','draft-eagle3','draft-dflash']:
  reps=sorted(g[case,mode]);a=np.array([g[case,mode][r] for r in reps]);b=np.array([g[case,'none'][r] for r in reps]);idx=rng.integers(0,len(a),(10000,len(a)));ratios=np.median(a[idx],axis=1)/np.median(b[idx],axis=1);lo,hi=np.quantile(ratios,[.025,.975]);ratio=np.median(a)/np.median(b)
  lines.append(f'| {case} | {mode} | {len(a)} | {np.median(a):.2f} | {min(a):.2f}–{max(a):.2f} | {ratio:.3f}× | {lo:.3f}–{hi:.3f}× |')
(root/'uncertainty.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
