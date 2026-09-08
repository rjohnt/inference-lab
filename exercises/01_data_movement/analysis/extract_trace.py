"""Extract only reviewed relative GPU activity timings from private Nsight SQLite."""
import argparse,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('sqlite',type=Path);ap.add_argument('--workload',required=True);ap.add_argument('--variant',required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
con=sqlite3.connect(f'file:{args.sqlite}?mode=ro',uri=True)
tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
kernels='CUPTI_ACTIVITY_KIND_KERNEL'
assert kernels in tables, 'No GPU kernel activities captured'
events=[]
for a,b in con.execute(f'SELECT start,end FROM {kernels} ORDER BY start'):
 events.append(dict(stage='compute',start_ns=a,end_ns=b))
assert 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables,'No GPU memcpy activities captured'
for a,b,kind,size in con.execute('SELECT start,end,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start'):
 if kind in [1,2]:events.append(dict(stage='H2D' if kind==1 else 'D2H',start_ns=a,end_ns=b,bytes=size))
assert events
origin=min(e['start_ns'] for e in events)
for stage in ['H2D','compute','D2H']:
 subset=sorted([e for e in events if e['stage']==stage],key=lambda e:e['start_ns'])
 assert len(subset)==24,(stage,len(subset),'Expected one activity per stage per batch')
 for i,e in enumerate(subset):e.update(batch=i,start_ms=(e.pop('start_ns')-origin)/1e6,end_ms=(e.pop('end_ns')-origin)/1e6)
# Merge copy intervals before intersecting so simultaneous H2D and D2H aren't double counted.
intervals=sorted((e['start_ms'],e['end_ms']) for e in events if e['stage']!='compute')
merged=[]
for a,b in intervals:
 if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
 else:merged.append([a,b])
compute=[e for e in events if e['stage']=='compute']
total=sum(e['end_ms']-e['start_ms'] for e in compute)
overlap=sum(max(0,min(e['end_ms'],b)-max(e['start_ms'],a)) for e in compute for a,b in merged)
result=dict(source='Nsight Systems CUDA activity trace',workload=args.workload,variant=args.variant,
 events=sorted(events,key=lambda e:e['start_ms']),compute_ms=total,compute_overlapping_copy_ms=overlap,
 compute_overlap_fraction=overlap/total,notes='Separate profiled run. Fraction of kernel activity time overlapping either copy direction; not a throughput speedup.')
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='events'}))
