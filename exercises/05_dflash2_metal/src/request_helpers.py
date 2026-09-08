"""Run on GPU Station. Isolated llama-server; never changes Ollama configuration."""
import atexit
import argparse, hashlib, json, os, random, subprocess, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
MODEL = '/usr/share/ollama/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f'
SERVER = '/usr/local/lib/ollama/llama-server'
BASE = 'http://127.0.0.1:18081'

def api(path, data=None, base=BASE):
    return urllib.request.urlopen(urllib.request.Request(base+path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type':'application/json'}), timeout=300)

def obj(path, data=None, base=BASE):
    with api(path, data, base) as r:
        return json.load(r)

def stream(prompt, limit=1536, thinking=True):
    payload = {'model':'qwen3-benchmark', 'messages':[
        {'role':'system','content':'You are a helpful assistant. Follow the requested output format and be concise.'},
        {'role':'user','content':prompt}], 'stream':True,
        'stream_options':{'include_usage':True}, 'temperature':0,
        'seed':42, 'max_tokens':limit, 'cache_prompt':False,
        'chat_template_kwargs':{'enable_thinking':thinking}, 'timings_per_token':True}
    t = time.perf_counter(); first = visible = None
    answer = thought = ''; timings = {}; usage = {}; finish = None
    with api('/v1/chat/completions',payload) as r:
        for line in r:
            if not line.startswith(b'data: ') or line[6:].strip()==b'[DONE]': continue
            d=json.loads(line[6:]); elapsed=time.perf_counter()-t
            if d.get('timings'):timings=d['timings']
            if d.get('usage'):usage=d['usage']
            for c in d.get('choices',[]):
                delta=c.get('delta',{})
                a=delta.get('content') or ''; b=delta.get('reasoning_content') or delta.get('reasoning') or ''
                if (a or b) and first is None:first=elapsed
                if a and visible is None:visible=elapsed
                answer+=a; thought+=b
                finish=c.get('finish_reason') or finish
    if not thinking and (thought or '<think>' in answer):
        raise RuntimeError('Unexpected reasoning in non-thinking run')
    return {'ttft_s':first,'first_answer_s':visible,'total_s':time.perf_counter()-t,
        'timings':timings,'usage':usage,'finish_reason':finish,
        'answer':answer,'reasoning_chars':len(thought),
        'output_sha256':hashlib.sha256((thought+'\n'+answer).encode()).hexdigest()}
