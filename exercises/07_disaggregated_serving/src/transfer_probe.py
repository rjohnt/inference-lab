"""Measure NIXL counters for four identical, isolated 32K transfers.

The first request warms the connection. Counter flush waits are outside timing.
The probe bypasses the benchmark router so its events stay separate.
"""
import argparse
import json
from pathlib import Path
import re
import time
import uuid
import httpx
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/disagg'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
t=AutoTokenizer.from_pretrained(a.root/'model')
fill=t.encode('The system records incoming requests, processes their documents, and returns a concise summary. ',add_special_tokens=False)
ids=(fill*((32768+len(fill)-1)//len(fill)))[:32768]
client=httpx.Client(timeout=180)

def counters():
    r=client.get('http://127.0.0.1:8200/metrics');r.raise_for_status();result={}
    for line in r.text.splitlines():
        m=re.match(r'(vllm:nixl_[^{ ]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$',line)
        if m and not m[1].endswith(('_created','_bucket')):result[m[1]]=float(m[2])
    return result

def request():
    headers={'X-Request-Id':str(uuid.uuid4())}
    body={'model':'qwen3','prompt':ids,'temperature':0,'max_tokens':1,'ignore_eos':True,'stream':False}
    prefill={**body,'kv_transfer_params':{'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None}}
    r=client.post('http://127.0.0.1:8100/v1/completions',json=prefill,headers=headers);r.raise_for_status()
    transfer=r.json()['kv_transfer_params'];assert transfer['remote_block_ids']
    r=client.post('http://127.0.0.1:8200/v1/completions',json={**body,'kv_transfer_params':transfer},headers=headers);r.raise_for_status()
    assert r.json()['usage']['completion_tokens']==1
request();time.sleep(12);before=counters()
for _ in range(4):request()
time.sleep(12);after=counters()
delta={k:after[k]-before.get(k,0) for k in after}
assert delta['vllm:nixl_xfer_time_seconds_count']==4,delta
assert delta['vllm:nixl_num_failed_transfers_total']==0
result={'context_tokens':32768,'output_tokens':1,'warmup_requests':1,'measured_transfers':4,'counters':delta}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
