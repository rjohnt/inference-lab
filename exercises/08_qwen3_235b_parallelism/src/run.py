"""Run one quantized Qwen parallelism mode with vLLM's serving benchmark."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/qwen235b'))
p.add_argument('--mode',choices=['tp','pp','ep'],required=True);p.add_argument('--pilot-only',action='store_true')
a=p.parse_args();root=a.root;raw=root/'raw'/(a.mode+'-pilot' if a.pilot_only else a.mode);raw.mkdir(parents=True,exist_ok=True)
python='/opt/disagg/venv/bin/python';cli='/opt/disagg/venv/bin/vllm';base='http://127.0.0.1:8300'
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0,1';env['VLLM_CACHE_ROOT']='/opt/disagg/vllm-cache-235b'
cmd=[cli,'serve',str(root/'model'),'--served-model-name','qwen235b','--host','127.0.0.1','--port','8300',
 '--dtype','bfloat16','--max-model-len','9216' if a.pilot_only else '33792','--max-num-seqs','4' if a.pilot_only else '16','--max-num-batched-tokens','4096',
 '--gpu-memory-utilization','0.90','--seed','42','--no-enable-prefix-caching','--generation-config','vllm',
 '--distributed-executor-backend','mp','--tensor-parallel-size','1' if a.mode=='pp' else '2',
 '--pipeline-parallel-size','2' if a.mode=='pp' else '1']
if a.mode=='ep':cmd+=['--enable-expert-parallel']
(raw/'launch-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
server_started=time.time()
log=(raw/'server.log').open('w');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=log,start_new_session=True)
def get(path):
    with urllib.request.urlopen(base+path,timeout=10) as r:return r.read()
def post(body):
    req=urllib.request.Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r:return json.load(r)
try:
    deadline=time.monotonic()+1800
    while True:
        if proc.poll() is not None:raise RuntimeError(f'{a.mode} startup exited {proc.returncode}')
        try:get('/v1/models');break
        except Exception:
            if time.monotonic()>deadline:raise TimeoutError('Server startup exceeded 30 minutes')
            time.sleep(3)
    (raw/'startup.json').write_text(json.dumps({'mode':a.mode,'pilot':a.pilot_only,'started_unix_s':server_started,'ready_unix_s':time.time(),'ready_elapsed_s':time.time()-server_started},indent=2)+'\n')
    print(a.mode+' SERVER_READY',flush=True)
    (raw/'initial-metrics.txt').write_bytes(get('/metrics'))
    checks=[]
    for profile in (('decode','intermediate') if a.pilot_only else ('decode','intermediate','prefill')):
        inputs=[json.loads(x) for x in (root/'raw'/f'inputs-{profile}.jsonl').read_text().splitlines()]
        for index,item in enumerate(inputs[:2]):
            out=post({'model':'qwen235b','prompt':item['prompt'],'max_tokens':64,'temperature':0,'ignore_eos':True,'stream':False})
            assert out['usage']['completion_tokens']==64
            checks.append({'profile':profile,'index':index,'usage':out['usage'],'text':out['choices'][0]['text']})
    (raw/'correctness.json').write_text(json.dumps(checks,indent=2)+'\n')
    for profile in (('decode','intermediate') if a.pilot_only else ('decode','intermediate','prefill')):
        for concurrency in ([1] if a.pilot_only else [1,4,16]):
            for rep in ([1] if a.pilot_only else [0,1,2,3]):
                n=4 if a.pilot_only else max(16,4*concurrency) if rep else max(4,concurrency)
                name=f'{a.mode}-{profile}-c{concurrency}-r{rep}'
                if a.pilot_only:name='pilot-'+name
                bench=[cli,'bench','serve','--backend','openai','--base-url',base,'--endpoint','/v1/completions',
                    '--model',str(root/'model'),'--served-model-name','qwen235b','--tokenizer',str(root/'model'),
                    '--dataset-name','custom','--custom-output-len','-1','--dataset-path',str(root/'raw'/f'inputs-{profile}.jsonl'),'--skip-chat-template',
                    '--num-prompts',str(n),'--max-concurrency',str(concurrency),'--request-rate','inf','--seed','42',
                    '--ignore-eos','--temperature','0','--percentile-metrics','ttft,tpot,itl,e2el','--metric-percentiles','50,95,99',
                    '--save-result','--save-detailed','--result-dir',str(raw),'--result-filename',name+'.json']
                before=get('/metrics').decode();start=time.time()
                with (raw/(name+'.log')).open('w') as output:subprocess.run(bench,env=env,stdout=output,stderr=output,check=True,timeout=3600)
                result=json.loads((raw/(name+'.json')).read_text());assert result['completed']==n,(name,result.get('completed'))
                expected_output={'decode':1024,'intermediate':512,'prefill':128}[profile]
                assert result['total_output_tokens']==n*expected_output,(name,result.get('total_output_tokens'),n*expected_output)
                (raw/(name+'-run.json')).write_text(json.dumps({'mode':a.mode,'profile':profile,'concurrency':concurrency,'rep':rep,'num_prompts':n,'pilot':a.pilot_only,'started_unix_s':start,'ended_unix_s':time.time(),'command':bench,'metrics_before':before,'metrics_after':get('/metrics').decode()},indent=2)+'\n')
                print(name+' COMPLETE '+json.dumps({k:result[k] for k in ('completed','output_throughput','mean_ttft_ms','mean_tpot_ms') if k in result}),flush=True)
    (raw/'final-metrics.txt').write_bytes(get('/metrics'))
    (root/(a.mode.upper()+('_PILOT' if a.pilot_only else '')+'_COMPLETE')).touch()
finally:
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    log.close()
