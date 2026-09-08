import json,math,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);s=json.loads((root/'summary.json').read_text());d={(r['case'],r['mode']):r for r in s};cases=sorted({r['case'] for r in s})
for mode in ['draft-eagle3','draft-dflash']:
    ratios=[d[c,mode]['decode_tps']/d[c,'none']['decode_tps'] for c in cases]
    print(mode,'geometric-mean throughput speedup',math.exp(statistics.mean(map(math.log,ratios))),'range',min(ratios),max(ratios))
    print('Visible answer faster cases',sum(d[c,mode]['first_answer_s']<d[c,'none']['first_answer_s'] for c in cases),'/',len(cases))
rows=[json.loads(l) for l in (root/'results.jsonl').read_text().splitlines()]
for mode in ['none','draft-eagle3','draft-dflash']:
    rs=[r for r in rows if r['mode']==mode]
    draft=sum(r['timings'].get('draft_n',0) for r in rs)
    acc=sum(r['timings'].get('draft_n_accepted',0) for r in rs)
    print(mode,'overall draft acceptance', acc/draft if draft else None)
