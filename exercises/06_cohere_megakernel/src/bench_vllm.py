"""Invoke vLLM's own serving benchmark repeatedly in one client process."""
import argparse
import contextlib
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
root=a.root
raw=root/'raw'
from vllm.benchmarks.serve import add_cli_args, main
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser=FlexibleArgumentParser()
add_cli_args(parser)
for context in (8192,1024,32768):
    for batch in (1,2,4,8):
        for rep in (0,1,2,3):
            name=f'vllm-c{context}-b{batch}-r{rep}'
            argv=['--backend','vllm','--base-url','http://127.0.0.1:8000',
                '--model','north-mini','--tokenizer',str(root/'checkpoint'),
                '--dataset-name','random','--random-input-len',str(context),
                '--random-output-len','512','--random-range-ratio','0.0',
                '--num-prompts',str(batch),'--max-concurrency',str(batch),
                '--ignore-eos','--temperature','0','--seed','42','--save-result','--save-detailed',
                '--result-dir',str(raw),'--result-filename',name+'.json',
                '--num-warmups',str(batch),'--percentile-metrics','ttft,tpot,itl,e2el']
            with (raw/(name+'.log')).open('w') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result=main(parser.parse_args(argv))
            if (result['completed']!=batch or result['total_output_tokens']!=batch*512
                    or result['input_lens']!=[context]*batch or result['output_lens']!=[512]*batch):
                raise RuntimeError(f'Incomplete benchmark {name}')
            print(name+' completed',flush=True)
