"""Download and checksum pinned public GGUFs, keeping all model data on E:."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path('/path/to/data/gpustation-benchmarks/dflash2')
MODELS = ROOT / 'models'
SPECS = [
    ('target', 'unsloth/Qwen3.8-27B-GGUF', '4ca720788d1e01f1bff70c033e0d0028fd02e502', 'Qwen3.8-27B-UD-IQ2_S.gguf'),
    ('draft', 'z-lab/Qwen3.8-27B-DFlash2-GGUF', '2d9571f8ce46e151f61c6499c99dee6079e1d610', 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf'),
]

def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HOME=str(ROOT / 'hf-cache'), HF_HUB_DISABLE_XET='1')
    manifest = {}
    for role, repo, revision, filename in SPECS:
        url = f'https://huggingface.co/api/models/{repo}/tree/{revision}'
        with urllib.request.urlopen(url, timeout=60) as response:
            entry = next(x for x in json.load(response) if x['path'] == filename)
        expected = entry['lfs']['oid']
        print(f'Downloading {role}: {filename} ({entry["size"]:,} bytes)', flush=True)
        subprocess.run([str(Path.home() / '.local/bin/hf'), 'download', repo, filename,
                        '--revision', revision, '--local-dir', str(MODELS)], env=env, check=True)
        path = MODELS / filename
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        assert path.stat().st_size == entry['size'], 'File size mismatch'
        assert actual == expected, 'SHA256 mismatch'
        manifest[role] = dict(repo=repo, revision=revision, filename=filename,
                              path=str(path), bytes=path.stat().st_size, sha256=actual)
        (ROOT / 'models.json').write_text(json.dumps(manifest, indent=2))
        print(f'VERIFIED {role}: {actual}', flush=True)
    (ROOT / 'DOWNLOAD_COMPLETE').write_text('verified\n')

if __name__ == '__main__':
    main()
