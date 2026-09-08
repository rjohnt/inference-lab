import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
import os
if not all(os.environ.get(k) for k in ('INFERENCE_BASE_URL', 'GPUSTATION_KEY_FILE')):
    raise SystemExit('Set INFERENCE_BASE_URL and GPUSTATION_KEY_FILE outside Git')
key = Path(os.environ['GPUSTATION_KEY_FILE']).expanduser().read_text().strip()
base = os.environ['INFERENCE_BASE_URL'].rstrip('/')

def models():
    req=urllib.request.Request(base+'/models',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=10) as r:
        return json.load(r)

def chat(model, prompt, stream=False, tokens=128):
    start=time.monotonic()
    data=dict(model=model,messages=[{'role':'system','content':'You are a helpful assistant. /no_think'},{'role':'user','content':prompt}],temperature=0,max_tokens=tokens,stream=stream)
    req=urllib.request.Request(base+'/chat/completions',data=json.dumps(data).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r:
        if stream:
            chunks=[]; done=False
            for line in r:
                if line.startswith(b'data: '):
                    payload=line[6:].strip()
                    if payload==b'[DONE]':
                        done=True
                    else:
                        chunks.append(json.loads(payload))
            assert done, 'Stream interrupted before DONE'
            content=''.join(c['choices'][0].get('delta',{}).get('content','') or '' for c in chunks if c.get('choices'))
        else:
            content=json.load(r)['choices'][0]['message']['content']
    assert content
    return dict(model=model,seconds=round(time.monotonic()-start,2),content=content,stream=stream)

if __name__=='__main__':
    results={'before':models()}
    with concurrent.futures.ThreadPoolExecutor() as pool:
        a=pool.submit(chat,'nemotron-nano-9b-v2','Count from 1 to 180 with every number separated by a comma.',True,600)
        time.sleep(1)
        b=pool.submit(chat,'qwen3.8-27b-dflash2','What is 17 times 23? Answer with just the number.')
        results['overlap_nemotron']=a.result()
        results['overlap_qwen']=b.result()
    assert '391' in results['overlap_qwen']['content']
    results['after_qwen']=models()
    print(json.dumps(results,indent=2),flush=True)
    (ROOT/'router-check.json').write_text(json.dumps(results,indent=2)+'\n')
