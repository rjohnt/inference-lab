"""Same fixed-length streaming workload for replicas and disaggregated serving."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import statistics
import random
import time

import httpx
from transformers import AutoTokenizer

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('/workspace/disagg'))
p.add_argument('--mode',choices=['replicas','pd'],required=True)
p.add_argument('--url',default='http://127.0.0.1:8000')
p.add_argument('--raw',type=Path)
p.add_argument('--profiles',nargs='+',choices=['1k','8k','32k','mixed'],default=['1k','8k','32k','mixed'])
p.add_argument('--requests-per-run',type=int,default=16)
p.add_argument('--shape-suite',action='store_true')
p.add_argument('--concurrencies',nargs='+',type=int,default=[4,16])
a=p.parse_args()
raw=a.raw or a.root/'raw'
raw.mkdir(parents=True,exist_ok=True)
tokenizer=AutoTokenizer.from_pretrained(a.root/'model')
fill=tokenizer.encode('The system records incoming requests, processes their documents, and returns a concise summary. ',add_special_tokens=False)

def prompt(length,index):
    prefix=tokenizer.encode(f'Document {index}. Read the following notes.\n',add_special_tokens=False)
    suffix=tokenizer.encode('\nSummarize the document in a short paragraph:\n',add_special_tokens=False)
    middle=length-len(prefix)-len(suffix)
    assert middle>0
    return prefix+(fill*((middle+len(fill)-1)//len(fill)))[:middle]+suffix

def quantile(xs,q):
    xs=sorted(xs)
    return xs[min(len(xs)-1,int((len(xs)-1)*q))]

async def one(client,ids,index,stream=True,tokens=128):
    data={'model':'qwen3','prompt':ids,'temperature':0,'top_p':1,'max_tokens':tokens,
          'ignore_eos':True,'stream':stream}
    if not stream:
        response=await client.post(a.url+'/v1/completions',json=data)
        response.raise_for_status()
        return response.json()
    data['stream_options']={'include_usage':True}
    start=time.perf_counter(); stamps=[]; text=[]; usage=None; finish=None
    async with client.stream('POST',a.url+'/v1/completions',json=data) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith('data: '): continue
            body=line[6:]
            if body=='[DONE]': break
            event=json.loads(body)
            if event.get('usage'): usage=event['usage']
            for choice in event.get('choices',[]):
                if choice.get('text'):
                    stamps.append(time.perf_counter()); text.append(choice['text'])
                finish=choice.get('finish_reason') or finish
    end=time.perf_counter()
    assert usage and usage['prompt_tokens']==len(ids),usage
    assert usage['completion_tokens']==tokens and stamps and finish=='length',(usage,finish)
    gaps=[(y-x)*1000 for x,y in zip(stamps,stamps[1:])]
    return {'index':index,'prompt_tokens':len(ids),'output_tokens':tokens,
            'ttft_ms':(stamps[0]-start)*1000,'latency_ms':(end-start)*1000,
            'tpot_ms':(stamps[-1]-stamps[0])*1000/(tokens-1),
            'chunk_gaps_ms':gaps,'text_chunks':len(stamps),
            'output_sha256':hashlib.sha256(''.join(text).encode()).hexdigest()}

async def metrics(client):
    result={}
    for port in (8100,8200):
        r=await client.get(f'http://127.0.0.1:{port}/metrics'); r.raise_for_status()
        result[str(port)]=r.text
    return result

async def main():
    async with httpx.AsyncClient(timeout=300,limits=httpx.Limits(max_connections=64,max_keepalive_connections=32)) as client:
        checks=[]
        for length in (128,1024,8192,32768):
            for index in (0,1):
                answer=await one(client,prompt(length,index),index,stream=False,tokens=64)
                assert answer['usage']['prompt_tokens']==length
                assert answer['usage']['completion_tokens']==64
                checks.append({'context':length,'index':index,'text':answer['choices'][0]['text'],'usage':answer['usage']})
        (raw/f'correctness-{a.mode}.json').write_text(json.dumps(checks,indent=2)+'\n')
        if a.mode=='pd' or raw!=a.root/'raw':
            baseline=json.loads((a.root/'raw/correctness-replicas.json').read_text())
            assert len(checks)==len(baseline)
            comparison=[{'context':x['context'],'index':x['index'],'exact_text_match':x['text']==y['text']} for x,y in zip(checks,baseline)]
            (raw/'correctness-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
            print('Greedy output agreement:',sum(x['exact_text_match'] for x in comparison),'/',len(comparison),flush=True)
        for profile in a.profiles:
            n=a.requests_per_run
            assert n>0 and n%2==0
            lengths={'1k':[1024]*n,'8k':[8192]*n,'32k':[32768]*n,'mixed':[1024,32768]*(n//2)}[profile]
            output_tokens={'1k':1024,'8k':512,'32k':128}[profile] if a.shape_suite else 128
            random.Random(42).shuffle(lengths)
            for concurrency in a.concurrencies:
                for rep in (0,1,2,3):
                    sem=asyncio.Semaphore(concurrency)
                    async def limited(index,length):
                        async with sem: return await one(client,prompt(length,index),index,tokens=output_tokens)
                    before=await metrics(client)
                    started_unix_s=time.time()
                    start=time.perf_counter()
                    rows=await asyncio.gather(*(limited(i,n) for i,n in enumerate(lengths)))
                    wall=time.perf_counter()-start
                    # vLLM exports connector statistics on an interval. This pause is
                    # outside request timing; final totals are flushed separately.
                    await asyncio.sleep(2)
                    after=await metrics(client)
                    name=f'{a.mode}-{profile}-c{concurrency}-r{rep}'
                    d={'mode':a.mode,'profile':profile,'concurrency':concurrency,'rep':rep,
                       'started_unix_s':started_unix_s,'wall_s':wall,'output_tok_s':sum(x['output_tokens'] for x in rows)/wall,
                       'mean_ttft_ms':statistics.mean(x['ttft_ms'] for x in rows),
                       'p95_ttft_ms':quantile([x['ttft_ms'] for x in rows],.95),
                       'mean_tpot_ms':statistics.mean(x['tpot_ms'] for x in rows),
                       'p95_chunk_gap_ms':quantile([v for x in rows for v in x['chunk_gaps_ms']],.95),
                       'requests':rows}
                    (raw/f'{name}.json').write_text(json.dumps(d,indent=2)+'\n')
                    (raw/f'{name}-metrics.json').write_text(json.dumps({'before':before,'after':after}))
                    print(json.dumps({k:v for k,v in d.items() if k!='requests'}),flush=True)
        await asyncio.sleep(12)
        (raw/f'{a.mode}-final-metrics.json').write_text(json.dumps(await metrics(client)))

asyncio.run(main())
