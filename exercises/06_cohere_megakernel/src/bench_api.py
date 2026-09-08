"""Same local streaming HTTP client and public coding tasks for both engines."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

TASKS = [
    'Write a Python function that merges two sorted lists. Include a short explanation and three tests.',
    'Implement binary search in Python. Explain its invariant and include tests for edge cases.',
    'Write a Python LRU cache using OrderedDict, with get and put methods and three tests.',
    'Implement topological sorting in Python with cycle detection. Include an example.',
    'Write a Python function to validate balanced parentheses, brackets and braces. Include tests.',
    'Implement breadth first search for a grid maze in Python. Return the shortest path.',
    'Write a Python function to group anagrams. Explain its complexity and include three tests.',
    'Implement a Python generator that reads a text file in chunks and counts lines. Explain boundary handling.',
]

def request(base_url, model, prompt, max_tokens):
    payload = json.dumps(dict(model=model, messages=[{'role':'user','content':prompt}], temperature=0, top_p=1,
        max_tokens=max_tokens, stream=True, stream_options={'include_usage': True})).encode()
    req = urllib.request.Request(base_url+'/v1/chat/completions', data=payload,
                                 headers={'Content-Type':'application/json'})
    start = time.perf_counter()
    first = last = None
    usage = None
    texts = []
    finish = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            body = raw[6:].strip()
            if body == b'[DONE]':
                break
            event = json.loads(body)
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                delta=choice.get('delta',{})
                chunk=''.join(delta.get(k) or '' for k in ('content','reasoning','reasoning_content'))
                if chunk:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    texts.append(chunk)
                finish = choice.get('finish_reason') or finish
    wall = time.perf_counter()-start
    if first is None or usage is None or finish not in ('stop','length'):
        raise RuntimeError(f'Incomplete stream: first={first}, usage={usage}, finish={finish}')
    n = usage['completion_tokens']
    return dict(ttft_ms=(first-start)*1000, latency_ms=wall*1000,
        tpot_ms=(last-first)*1000/(n-1) if n>1 else None,
        completion_tokens=n, prompt_tokens=usage['prompt_tokens'], finish_reason=finish,
        output_sha256=hashlib.sha256(''.join(texts).encode()).hexdigest())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--model',default='north-mini')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--engine',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=256)
    a=p.parse_args()
    def prompts_for(concurrency, rep):
        # Same deterministic unique prefix on both engines; prevents whole-prompt
        # reuse between warmups/repetitions while retaining identical workloads.
        prompts=[]
        for i, task in enumerate(TASKS):
            nonce=hashlib.sha256(f'{concurrency}:{rep}:{i}'.encode()).hexdigest()
            text=f'Request ID: {nonce}.\n{task}'
            prompts.append(text)
        return prompts
    results=[]
    for concurrency in (1,8):
        # Warm every public fixture, exercising batch-8 JIT when applicable.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(lambda prompt: request(a.base_url,a.model,prompt,a.tokens),prompts_for(concurrency,0)))
        for rep in range(1,4):
            start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows=list(pool.map(lambda prompt: request(a.base_url,a.model,prompt,a.tokens),prompts_for(concurrency,rep)))
            wall=time.perf_counter()-start
            item=dict(engine=a.engine,concurrency=concurrency,rep=rep,requests=rows,
                wall_s=wall,output_tok_s=sum(r['completion_tokens'] for r in rows)/wall,
                mean_ttft_ms=statistics.mean(r['ttft_ms'] for r in rows),
                mean_tpot_ms=statistics.mean(r['tpot_ms'] for r in rows if r['tpot_ms'] is not None))
            results.append(item)
            a.output.write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:v for k,v in item.items() if k!='requests'}),flush=True)

if __name__=='__main__':
    main()
