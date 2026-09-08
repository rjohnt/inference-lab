import hashlib
import json
from pathlib import Path
import subprocess
import requests

root = Path('/path/to/data/gpustation-models/nemotron')
root.mkdir(parents=True, exist_ok=True)
repo = 'bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF'
filename = 'nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf'
manifest = root / 'manifest.json'
if manifest.exists():
    meta = json.loads(manifest.read_text())
else:
    r = requests.get(f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=60)
    r.raise_for_status()
    info = r.json()
    entry = next(x for x in info['siblings'] if x['rfilename'] == filename)
    meta = dict(repo=repo, revision=info['sha'], filename=filename,
                bytes=entry['size'], sha256=entry['lfs']['sha256'])
    manifest.write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta), flush=True)
target = root / filename
partial = root / (filename + '.partial')
if not target.exists():
    subprocess.run(['curl', '-fL', '--retry', '5', '--retry-delay', '3',
                    '-C', '-', '-o', str(partial),
                    f"https://huggingface.co/{repo}/resolve/{meta['revision']}/{filename}"], check=True)
    source = partial
else:
    source = target
assert source.stat().st_size == meta['bytes'], 'Size mismatch'
digest = hashlib.sha256()
with source.open('rb') as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == meta['sha256'], 'SHA256 mismatch'
if source == partial:
    partial.rename(target)
print('VERIFIED ' + str(target), flush=True)
