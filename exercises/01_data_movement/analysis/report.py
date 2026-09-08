"""Audit raw block timings and publish compact summaries plus charts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert (args.run/'COMPLETE').is_file(), 'Incomplete run'
    manifest = json.loads((args.run/'manifest.json').read_text())
    cfg = manifest['config']
    rows = json.loads((args.run/'summary.json').read_text())
    samples = [json.loads(x) for x in (args.run/'samples.jsonl').read_text().splitlines()]
    expected_per_repeat = len(cfg['transfer_bytes'])*4 + len(cfg['packing_chunks']) + 8
    assert len(rows) == len(samples) == cfg['repeats']*expected_per_repeat
    seen = set()
    for i, (r, raw) in enumerate(zip(rows,samples)):
        key = (r['repeat'],r['kind'],r.get('bytes'),r.get('direction'),r.get('pinned'),r.get('chunks'),r.get('workload'),r.get('variant'))
        assert key not in seen
        seen.add(key)
        a = np.array(raw['seconds'])
        assert raw['case'] == i and len(a) == r['samples'] >= cfg['min_samples']
        assert np.isfinite(a).all() and (a > 0).all()
        assert np.isclose(a.sum(),r['measured_seconds'],rtol=1e-10)
        assert a.sum() >= cfg['pipeline_seconds' if r['kind']=='pipeline' else 'transfer_seconds']
        assert np.isclose(np.median(a)/r['count'],r['median_seconds'],rtol=1e-10)
        assert np.isclose(np.percentile(a,95)/r['count'],r['p95_seconds'],rtol=1e-10)
        assert r['total_operations'] == len(a)*r['count'] and r['correctness']=='PASS'
    args.out.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json','summary.json','COMPLETE']:
        (args.out/name).write_bytes((args.run/name).read_bytes())
    audit = dict(status='PASS',cases=len(rows),samples=sum(r['samples'] for r in rows),
                 total_operations=sum(r['total_operations'] for r in rows),
                 measured_seconds=sum(r['measured_seconds'] for r in rows),
                 raw_sha256=hashlib.sha256((args.run/'samples.jsonl').read_bytes()).hexdigest())
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (args.out/'timings.csv').open('w') as f:
        writer=csv.DictWriter(f,keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'svg.hashsalt':'data-movement-v1','font.size':11})
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(args.out/(name+'.png'),dpi=170)
        fig.savefig(args.out/(name+'.svg'),metadata={'Date':None})
        p=args.out/(name+'.svg');p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,d in zip(axs,['H2D','D2H']):
        for pinned in [False,True]:
            series=[]
            for size in cfg['transfer_bytes']:
                vals=[size/r['median_seconds']/1e9 for r in rows if r['kind']=='transfer' and r['direction']==d and r['pinned']==pinned and r['bytes']==size]
                series.append((size,np.median(vals),min(vals),max(vals)))
            a=np.array(series);ax.plot(a[:,0],a[:,1],'o-',label='Pinned' if pinned else 'Pageable')
            ax.fill_between(a[:,0],a[:,2],a[:,3],alpha=.15)
        ax.set_xscale('log',base=2);ax.set_xlabel('Payload bytes');ax.set_ylabel('Effective GB/s (payload / wall time)')
        ax.set_title(d);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Host/device transfers · median across repeats; shading = repeat range')
    finish(fig,'transfer-bandwidth')
    variants=['pageable_serial','pinned_serial','overlapped','resident']
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    findings=[]
    for ax,w in zip(axs,['rmsnorm','matmul']):
        vals=[[1/r['median_seconds'] for r in rows if r['kind']=='pipeline' and r['workload']==w and r['variant']==v] for v in variants]
        med=np.array([np.median(v) for v in vals]);lo=np.array([min(v) for v in vals]);hi=np.array([max(v) for v in vals])
        ax.bar(range(4),med,yerr=[med-lo,hi-med],capsize=4,color=['#94a3b8','#22d3ee','#34d399','#a78bfa'])
        ax.set_xticks(range(4),['Pageable\nserial','Pinned\nserial','3-stream\npipeline','GPU\nresident'])
        ax.set_ylabel('Completed batches / second');ax.set_title(w.upper());ax.grid(axis='y',alpha=.2)
        findings.append(f'| {w} | '+ ' | '.join(f'{v:,.1f}' for v in med)+f' | {med[2]/med[0]:.2f}× |')
    fig.suptitle('End-to-end batch throughput · error bars = repeat range')
    finish(fig,'pipeline-throughput')
    fig,ax=plt.subplots(figsize=(9,5))
    for rep in range(1,cfg['repeats']+1):
        subset=sorted((r for r in rows if r['kind']=='packing' and r['repeat']==rep),key=lambda r:r['chunks'])
        ax.plot([r['chunks'] for r in subset],[r['median_seconds']*1e6 for r in subset],'o-',label=f'Repeat {rep}')
    ax.set_xscale('log',base=2);ax.set_xlabel('Copy submissions for the same 4 MiB payload');ax.set_ylabel('Microseconds per payload')
    ax.set_title('Small-copy overhead · pre-existing contiguous views; no packing cost')
    ax.legend();ax.grid(alpha=.2);finish(fig,'copy-batching')
    text='''# Data movement results

All numerical checks and independent timing audits passed. Three repetitions are
separate measurement passes in one process, with independently shuffled case order.
Error bars show the range of repeat medians, not confidence intervals.

| Workload | Pageable serial batches/s | Pinned serial | Overlapped | GPU resident | Overlapped / pageable |
| --- | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(findings)+f'''\n
Measured {audit['cases']} cases, {audit['samples']:,} timed blocks, and
{audit['total_operations']:,} operations (a transfer payload or a pipeline batch).
Total timed-block duration: {audit['measured_seconds']/60:.2f} minutes.

![Transfer bandwidth](transfer-bandwidth.png)

![Pipeline throughput](pipeline-throughput.png)

![Copy batching](copy-batching.png)

See the exercise README for theory, protocol, reproduction commands and limitations.
'''
    (args.out/'report.md').write_text(text)
    print(json.dumps(audit))

if __name__=='__main__': main()
