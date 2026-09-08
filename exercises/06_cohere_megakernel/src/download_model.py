"""Fetch the same public BF16 checkpoint for both engines."""
import os
from huggingface_hub import snapshot_download

snapshot_download(
    'CohereLabs/North-Mini-Code-1.0',
    revision='d11e61a842617a22dc328552fa5bb86231ee4f37',
    local_dir=os.environ.get('TASK_ROOT','/workspace/cohere-benchmark')+'/checkpoint',
    max_workers=8,
)
