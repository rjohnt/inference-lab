"""Investigate the 8K greedy mismatch using common-prefix token probabilities.

All responses and token IDs are private. The public summary contains only
agreement counts, positions, and probability differences.
"""
import argparse
import json
from pathlib import Path
import urllib.request
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/disagg'));a=p.parse_args()
out=a.root/'raw/diagnostics';out.mkdir(exist_ok=True)
t=AutoTokenizer.from_pretrained(a.root/'model')
fill=t.encode('The system records incoming requests, processes their documents, and returns a concise summary. ',add_special_tokens=False)
prefix=t.encode('Document 0. Read the following notes.\n',add_special_tokens=False)
suffix=t.encode('\nSummarize the document in a short paragraph:\n',add_special_tokens=False)
n=8192-len(prefix)-len(suffix);ids=prefix+(fill*((n+len(fill)-1)//len(fill)))[:n]+suffix

def request(port,prompt,tokens=64):
    data={'model':'qwen3','prompt':prompt,'temperature':0,'top_p':1,'max_tokens':tokens,
          'ignore_eos':True,'stream':False,'logprobs':5,'return_tokens_as_token_ids':True}
    r=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=120) as response:d=json.load(response)
    assert d['usage']['completion_tokens']==tokens
    return d

paths={'pd':8000,'local_decode_gpu':8200,'local_prefill_gpu':8100}
responses={name:[request(port,ids) for _ in range(3)] for name,port in paths.items()}
(out/'responses.json').write_text(json.dumps(responses,indent=2)+'\n')
seq=lambda d:d['choices'][0]['logprobs']['tokens']
summary={'context_tokens':8192,'output_tokens':64,'repetitions':3,
         'within_path_repeat_agreement':{name:all(seq(v)==seq(values[0]) for v in values) for name,values in responses.items()},'comparisons':{}}
pd=responses['pd'][0]
for name in ('local_decode_gpu','local_prefill_gpu'):
    other=responses[name][0];x=seq(pd);y=seq(other)
    first=next((i for i,(v,w) in enumerate(zip(x,y)) if v!=w),None)
    result={'exact_token_sequence_match':x==y,'common_prefix_tokens':len(x) if first is None else first}
    if first is not None:
        common=ids+[int(v.removeprefix('token_id:')) for v in x[:first]]
        probes={key:request(paths[key],common,1) for key in ('pd',name)}
        (out/f'common-prefix-{name}.json').write_text(json.dumps(probes,indent=2)+'\n')
        prob={key:d['choices'][0]['logprobs']['top_logprobs'][0] for key,d in probes.items()}
        shared=set(prob['pd'])&set(prob[name])
        result['common_prefix_probe']={
            'same_argmax':seq(probes['pd'])==seq(probes[name]),
            'shared_top5_tokens':len(shared),
            'max_abs_shared_logprob_delta':max((abs(prob['pd'][k]-prob[name][k]) for k in shared),default=None),
            'top1_top2_logprob_margin':{key:sorted(v.values(),reverse=True)[0]-sorted(v.values(),reverse=True)[1] for key,v in prob.items()}}
    summary['comparisons'][name]=result
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
