"""Capture one instrumented decode step, separately from performance results."""
import argparse
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.root/'megakernel/src'))
import torch
import runner
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
runner.main(['--mk','--fast','--real-weight','--checkpoint',str(a.root/'checkpoint'),
    '--lib-path',str(a.root/'megakernel/build/libmk_release.so'),
    '--batch-size','1','--fake-prompt-len','8192','--max-new-tokens','2',
    '--num-sms','132','--frac-vram-utilization','0.90',
    '--profile',str(a.root/'raw/mk-sm-trace.json')])
