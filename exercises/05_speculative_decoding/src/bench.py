"""Run on GPU Station. Isolated llama-server; never changes Ollama configuration."""
import atexit
from host_monitor import Monitor
import argparse, hashlib, json, os, random, subprocess, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
MODEL = '/usr/share/ollama/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f'
SERVER = '/usr/local/lib/ollama/llama-server'
BASE = 'http://127.0.0.1:18081'

def api(path, data=None, base=BASE):
    return urllib.request.urlopen(urllib.request.Request(base+path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type':'application/json'}), timeout=300)

def obj(path, data=None, base=BASE):
    with api(path, data, base) as r:
        return json.load(r)

def stream(prompt, limit=1536, thinking=True):
    payload = {'model':'qwen3-benchmark', 'messages':[
        {'role':'system','content':'You are a helpful assistant. Follow the requested output format and be concise.'},
        {'role':'user','content':prompt}], 'stream':True,
        'stream_options':{'include_usage':True}, 'temperature':0,
        'seed':42, 'max_tokens':limit, 'cache_prompt':False,
        'chat_template_kwargs':{'enable_thinking':thinking}, 'timings_per_token':True}
    t = time.perf_counter(); first = visible = None
    answer = thought = ''; timings = {}; usage = {}; finish = None
    with api('/v1/chat/completions',payload) as r:
        for line in r:
            if not line.startswith(b'data: ') or line[6:].strip()==b'[DONE]': continue
            d=json.loads(line[6:]); elapsed=time.perf_counter()-t
            if d.get('timings'):timings=d['timings']
            if d.get('usage'):usage=d['usage']
            for c in d.get('choices',[]):
                delta=c.get('delta',{})
                a=delta.get('content') or ''; b=delta.get('reasoning_content') or delta.get('reasoning') or ''
                if (a or b) and first is None:first=elapsed
                if a and visible is None:visible=elapsed
                answer+=a; thought+=b
                finish=c.get('finish_reason') or finish
    if not thinking and (thought or '<think>' in answer):
        raise RuntimeError('Unexpected reasoning in non-thinking run')
    return {'ttft_s':first,'first_answer_s':visible,'total_s':time.perf_counter()-t,
        'timings':timings,'usage':usage,'finish_reason':finish,
        'answer':answer,'reasoning_chars':len(thought),
        'output_sha256':hashlib.sha256((thought+'\n'+answer).encode()).hexdigest()}

def make_prompts():
    cases=[{'id':'chat_short','kind':'chat','prompt':"What is a diffusion model? Explain in three short sentences."}]
    for length in [512,2048,8192]:
        records=[]
        for i in range(800):
            records.append(f'Record {i:04d}: service=worker-{i:04d}; owner=team-{i%17}; retries={i%5}; timeout_ms={1000+i*7}; region=zone-{i%4}.\n')
        tok=obj('/tokenize',{'content':''.join(records),'add_special':False})['tokens'][:length]
        document=obj('/detokenize',{'tokens':tok})['content']
        marker='\nRecord SPECIAL: service=orchid; owner=team-amber; retries=3; timeout_ms=2750; region=zone-2.\n'
        middle=len(document)//2
        document=document[:middle]+marker+document[middle:]
        cases.append({'id':f'document_{length}','kind':'retrieval','target_context_tokens':length,
            'prompt':'Read the following service inventory.\n<inventory>\n'+document+'\n</inventory>\nReturn only a JSON object containing owner, retries, timeout_ms, and region for service orchid.'})
    for length in [512,8192]:
        functions='\n'.join(f'def handler_{i:03d}(value):\n    """Transform value for handler {i:03d}."""\n    result = value + {i}\n    return result\n' for i in range(600))
        tok=obj('/tokenize',{'content':functions,'add_special':False})['tokens'][:length]
        context=obj('/detokenize',{'tokens':tok})['content']
        target='def summarize(values):\n    total = sum(values)\n    count = len(values)\n    average = total / count\n    minimum = min(values)\n    maximum = max(values)\n    return {"total": total, "count": count, "average": average, "minimum": minimum, "maximum": maximum}\n'
        cases.append({'id':f'code_{length}','kind':'code_edit','target_context_tokens':length,
            'prompt':'Context from a Python module (the excerpt may end mid-function):\n```python\n'+context+'\n```\nModify only this function so empty input returns count and total 0 and average, minimum, maximum None. Preserve all nonempty behavior and existing variable names. Return the complete updated function only.\n```python\n'+target+'```'})
    for c in cases:
        c['user_tokens']=len(obj('/tokenize',{'content':c['prompt'],'add_special':False})['tokens'])
    (ROOT/'prompts.json').write_text(json.dumps(cases,indent=2))
    return cases

def start(mode, rep, out, context=32768, ubatch=512, thinking=True):
    cmd=[SERVER,'-m',MODEL,'--alias','qwen3-benchmark','--host','127.0.0.1','--port','18081',
        '-c',str(context),'-np','1','-ngl','all','-fa','on','-ctk','q8_0','-ctv','q8_0','-b','512','-ub',str(ubatch),
        '--jinja','--reasoning-format','deepseek','--reasoning','on' if thinking else 'off','--metrics',
        '--cache-ram','0','--load-mode','dio','--log-verbosity','4','--spec-type',mode]
    if mode=='draft-eagle3':cmd+=['-md',str(ROOT/'models/eagle3-q8.gguf'),'--spec-draft-ngl','all','--spec-draft-n-max','3']
    if mode=='draft-dflash':cmd+=['-md',str(ROOT/'models/dflash-q8.gguf'),'--spec-draft-ngl','all','--spec-draft-n-max','15']
    if mode=='ngram-mod':cmd+=['--spec-ngram-mod-n-match','12','--spec-ngram-mod-n-min','4','--spec-ngram-mod-n-max','16']
    if mode in ('draft-eagle3','draft-dflash'):cmd+=['--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0']
    log=(out/f'{mode}-{rep}.log').open('w'); t=time.perf_counter()
    env=dict(os.environ, LD_LIBRARY_PATH='/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v13:/usr/lib/wsl/lib', GGML_BACKEND_PATH='/usr/local/lib/ollama/cuda_v13/libggml-cuda.so')
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    try:
        while time.perf_counter()-t<180:
            if proc.poll() is not None:raise RuntimeError(f'{mode} exited {proc.returncode}; see log')
            try: healthy=obj('/health').get('status')=='ok'
            except Exception: healthy=False
            if healthy:
                startup_log=(out/f'{mode}-{rep}.log').read_text()
                if 'no usable GPU found' in startup_log or 'offloaded 37/37 layers to GPU' not in startup_log:
                    raise RuntimeError('Full GPU offload not confirmed; refusing invalid benchmark')
                return proc,log,cmd,time.perf_counter()-t
            time.sleep(0.5)
        raise TimeoutError('server startup timed out')
    except BaseException:
        proc.terminate();proc.wait();log.close();raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--modes',nargs='+',default=['none','ngram-mod','draft-eagle3','draft-dflash'])
    parser.add_argument('--context',type=int,default=32768)
    parser.add_argument('--ubatch',type=int,default=512)
    parser.add_argument('--output-root',type=Path,default=ROOT)
    parser.add_argument('--thinking',choices=['on','off'],default='on')
    args=parser.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True)
    out=args.output_root/('results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    metadata={'created':datetime.now(timezone.utc).isoformat(),'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'server_version':subprocess.getoutput(SERVER+' --version'),
        'gpu':subprocess.getoutput('/usr/lib/wsl/lib/nvidia-smi'),
        'method':f'Server-local timing; warm GPU; cache_prompt=false; greedy decoding; thinking {args.thinking}; q8_0 KV for all modes. Rotated method order each repetition. Process startup to health recorded separately, not a disk-cold measurement. Not directly comparable with Mac-to-Ollama baseline.',
        'server_env':{'LD_LIBRARY_PATH':'/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v13:/usr/lib/wsl/lib','GGML_BACKEND_PATH':'/usr/local/lib/ollama/cuda_v13/libggml-cuda.so'},
        'draft_url':'https://huggingface.co/williamliao/Qwen3-8B-EAGLE3-Speculator-GGUF/resolve/44480ff4ea6330788818f7f5fc9a69b326dc4c06/Qwen3-8B-speculator.eagle3-F16.gguf',
        'dflash_url':'https://huggingface.co/AtomicChat/Qwen3-8B-DFlash-GGUF/resolve/788b1a553f50979b99fecf6abe7a4c3fd88a8d89/Qwen3-8B-DFlash.Q8_0.gguf'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    # Do not compete with a loaded Ollama model. Unload just the existing target.
    obj('/api/generate',{'model':'qwen3:8b-32k','keep_alive':0},base='http://127.0.0.1:11434')
    monitor=Monitor(out);monitor.start();atexit.register(monitor.stop)
    monitor.snapshot('initial')
    cases=None
    with (out/'results.jsonl').open('a') as f:
        for rep in range(args.repeats):
            order=args.modes[rep%len(args.modes):]+args.modes[:rep%len(args.modes)]
            for mode in order:
                monitor.phase=f'{mode}/rep-{rep+1}/startup'
                monitor.snapshot(f'{mode}-{rep}-before')
                print(f'START {mode} repetition {rep+1}',flush=True)
                proc,log,cmd,load=start(mode,rep,out,args.context,args.ubatch,args.thinking=='on')
                try:
                    (out/f'{mode}-{rep}-command.json').write_text(json.dumps({'command':cmd,'startup_s':load}))
                    if cases is None:
                        cases=make_prompts();(out/'prompts.json').write_text(json.dumps(cases,indent=2))
                    stream('Reply with the single word Ready.',64,thinking=args.thinking=='on')
                    shuffled=list(cases);random.Random(42+rep).shuffle(shuffled)
                    for case in shuffled:
                        monitor.phase=f'{mode}/rep-{rep+1}/{case["id"]}'
                        request_unix_start=time.time()
                        result=stream(case['prompt'],thinking=args.thinking=='on')
                        result.update(request_unix_start=request_unix_start,request_unix_end=time.time())
                        result.update(thinking=args.thinking,mode=mode,rep=rep+1,case=case['id'],user_tokens=case['user_tokens'])
                        f.write(json.dumps(result)+'\n');f.flush()
                        print(json.dumps({k:v for k,v in result.items() if k!='answer'}),flush=True)
                    with api('/metrics') as response:(out/f'{mode}-{rep}-metrics.txt').write_bytes(response.read())
                finally:
                    proc.terminate()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    log.close()
    monitor.phase='complete';monitor.snapshot('final')
    (out/'COMPLETE').write_text('complete\n')
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
