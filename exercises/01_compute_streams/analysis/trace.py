"""Publish relative kernel activities from a separate Nsight CUDA-graph-node trace."""
import argparse,json,sqlite3,hashlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('sqlite',type=Path);ap.add_argument('--strategy',required=True);ap.add_argument('--rows',type=int,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
c=sqlite3.connect(f'file:{args.sqlite}?mode=ro',uri=True)
activities=c.execute('SELECT start,end,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall()
expected=8 if args.strategy=='batched' else 32
assert len(activities)==expected,(len(activities),expected)
origin=min(a for a,b,s in activities);streams={s:i for i,s in enumerate(sorted({s for a,b,s in activities}))}
events=[dict(start_us=(a-origin)/1e3,end_us=(b-origin)/1e3,lane=streams[s]) for a,b,s in activities]
# Sweep half-open intervals: ends precede starts at equal timestamps.
points=sorted([(a,1) for a,b,s in activities]+[(b,-1) for a,b,s in activities])
active=0;maximum=0;overlap=0;busy=0;previous=points[0][0]
for t,delta in points:
 if active:busy+=t-previous
 if active>1:overlap+=t-previous
 active+=delta;maximum=max(maximum,active);previous=t
result=dict(raw_sha256=hashlib.sha256(args.sqlite.read_bytes()).hexdigest(),strategy=args.strategy,rows=args.rows,graph_replays=8,kernels=len(events),events=events,
 peak_concurrent_kernel_intervals=maximum,overlap_fraction_of_busy_time=overlap/busy,
 timestamp_method='Nsight Systems with WSL CUPTI conversion; reduced timestamp accuracy',
 notes='Separate instrumented capture. Graph scheduler may use internal streams. Tiny overlaps are not reliable with this timestamp workaround.')
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='events'}))
