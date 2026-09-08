"""Check coverage, truncation, retrieval answers and code syntax without executing model output."""
import ast,collections,json,re,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[json.loads(l) for l in (root/'results.jsonl').read_text().splitlines()]
meta=json.loads((root/'metadata.json').read_text()); repeats=meta['args']['repeats']; total=repeats*18
expected={'owner':'team-amber','retries':3,'timeout_ms':2750,'region':'zone-2'}
reference=ast.parse('def summarize(values):\n    if not values:\n        return {"total": 0, "count": 0, "average": None, "minimum": None, "maximum": None}\n    total = sum(values)\n    count = len(values)\n    average = total / count\n    minimum = min(values)\n    maximum = max(values)\n    return {"total": total, "count": count, "average": average, "minimum": minimum, "maximum": maximum}\n')
checks=[]
for r in rows:
    answer=r['answer'].strip()
    if answer.startswith('```'):answer=re.sub(r'^```[^\n]*\n|\n```$','',answer).strip()
    check={'case':r['case'],'mode':r['mode'],'rep':r['rep'],'stopped_normally':r['finish_reason']=='stop','visible_answer':r['first_answer_s'] is not None,'uncached_prompt':r['timings'].get('cache_n')==0}
    if r['case'].startswith('document'):
        try:check['retrieval_correct']=json.loads(answer)==expected
        except Exception:check['retrieval_correct']=False
    if r['case'].startswith('code'):
        try:
            tree=ast.parse(answer)
            check['matches_reviewed_correct_code']=ast.dump(tree)==ast.dump(reference)
            check['single_summarize_function']=len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name=='summarize'
        except SyntaxError:check['single_summarize_function']=False
    checks.append(check)
counts=collections.Counter((r['case'],r['mode']) for r in rows)
assert len(rows)==total and len(counts)==18 and all(v==repeats for v in counts.values()), counts
assert len({(r['case'],r['mode'],r['rep']) for r in rows})==total
assert all(all(v for k,v in c.items() if isinstance(v,bool)) for c in checks),checks
(root/'validation.json').write_text(json.dumps(checks,indent=2))
print(f'{total} unique requests; {repeats} repeats per cell; all prompts uncached; no truncation; all retrieval answers correct; all code outputs parse as one summarize function.')
