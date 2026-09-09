"""Download the user-selected checkpoint at its pinned revision."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/qwen235b'));a=p.parse_args()
model='QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ';revision='38e8e75808020b2c9be573c19629a4d02cd8e8aa'
a.root.mkdir(parents=True,exist_ok=True)
snapshot_download(model,revision=revision,local_dir=a.root/'model',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],max_workers=8)
index=json.loads((a.root/'model/model.safetensors.index.json').read_text())
files=sorted(set(index['weight_map'].values()))
assert all((a.root/'model'/name).is_file() for name in files)
(a.root/'raw').mkdir(exist_ok=True)
(a.root/'raw/checkpoint.json').write_text(json.dumps({'model':model,'revision':revision,'declared_tensor_bytes':index['metadata']['total_size'],'shard_count':len(files),'shard_file_bytes':sum((a.root/'model'/name).stat().st_size for name in files)},indent=2)+'\n')
(a.root/'MODEL_READY').touch()
