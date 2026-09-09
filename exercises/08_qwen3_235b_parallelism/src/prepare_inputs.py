"""Build private, length-controlled serving inputs from pinned LongBench documents."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request
import zipfile
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/qwen235b'));a=p.parse_args()
root=a.root;raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
archive=root/'longbench.zip'
if not archive.exists():urllib.request.urlretrieve(f'https://huggingface.co/datasets/zai-org/LongBench/resolve/{revision}/data.zip',archive)
t=AutoTokenizer.from_pretrained(root/'model')
documents=[]
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if Path(name).name not in ('gov_report.jsonl','qasper.jsonl','hotpotqa.jsonl'):continue
        for line in z.read(name).decode().splitlines():
            d=json.loads(line)
            instruction='Answer the task using the document.\nTask: '+d['input']+'\nDocument:\n'
            documents.append({'task':Path(name).stem,'id':d['_id'],'prefix':instruction,'tokens':t.encode(d['context'],add_special_tokens=False)})
random.Random(42).shuffle(documents)
manifest={'dataset':'zai-org/LongBench','revision':revision,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'seed':42,'scored_evaluation':False,'profiles':{}}
for name,limit,output in [('decode',1024,1024),('intermediate',8192,512),('prefill',32768,128)]:
    rows=[];sources=[]
    for d in documents:
        if len(d['tokens'])<limit:continue
        count=limit-len(t.encode(d['prefix']))-64
        while True:
            text=d['prefix']+t.decode(d['tokens'][:count])
            prompt=t.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
            actual=len(t.encode(prompt,add_special_tokens=False))
            if actual<=limit:break
            count-=actual-limit+1
        rows.append({'prompt':prompt,'output_tokens':output})
        sources.append({'source_id':d['id'],'task':d['task'],'input_tokens':actual})
        if len(rows)==64:break
    assert rows,(name,'no sufficiently long documents')
    # Repeat only when this filtered source pool has fewer than 64 documents.
    unique=len(rows);rows=(rows*((64+unique-1)//unique))[:64]
    out=raw/f'inputs-{name}.jsonl'
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest['profiles'][name]={'requests_available':64,'unique_sources':unique,'input_token_min':min(r['input_tokens'] for r in sources),'input_token_max':max(r['input_tokens'] for r in sources),'output_tokens':output,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'sources':sources}
(raw/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:{x:y for x,y in v.items() if x!='sources'} for k,v in manifest['profiles'].items()}),flush=True)
