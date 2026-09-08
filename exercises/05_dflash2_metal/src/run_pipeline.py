import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for out,extra in [('probe-matched-16k',['--probe']),('results-16k',['--repeats','10'])]:
    dest=ROOT/out
    if not (dest/'COMPLETE').exists():
        subprocess.run([sys.executable,'-u',str(ROOT/'bench.py'),'--context','16384','--ubatch','128','--out',str(dest),*extra],check=True)
    rows=[json.loads(s) for s in (dest/'results.jsonl').read_text().splitlines()]
    assert len(rows)==(12 if extra==['--probe'] else 120)
    assert all(r['finish_reason']=='stop' and r['answer'].strip() and r['reasoning_chars']==0 for r in rows)
    draft=[r for r in rows if r['mode']=='draft-dflash']
    assert sum(r['timings'].get('draft_n_accepted',0) for r in draft)>0
    print('VALIDATED',out,flush=True)
subprocess.run([sys.executable,str(ROOT/'compare.py')],check=True)
(ROOT/'ARTIFACTS_COMPLETE').write_text('Full Mac comparison and four-column report generated\n')
