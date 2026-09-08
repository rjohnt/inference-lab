"""Same shared-weight GEMMs: sequential, multi-stream, or one larger GEMM."""
import argparse,hashlib,json,math,platform,random,sys,time
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
# Reuse the independently audited duration controller, without importing its main.
import importlib.util
helper_path=ROOT.parent/'01_data_movement/src/benchmark.py'
spec=importlib.util.spec_from_file_location('movement_benchmark',helper_path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

class Workload:
    def __init__(self,rows,strategy,cfg):
        self.strategy=strategy;self.requests=cfg['requests'];self.rows=rows;self.width=cfg['width']
        torch.manual_seed(cfg['seed'])
        self.x=torch.randn(self.requests,rows,self.width,device='cuda',dtype=torch.bfloat16)
        self.w=torch.randn(self.width,self.width,device='cuda',dtype=torch.bfloat16)/math.sqrt(self.width)
        self.y=torch.empty_like(self.x)
        self.root=torch.cuda.Stream();self.branches=[torch.cuda.Stream() for _ in range(self.requests)]
        # Pre-create views so batching does not add a timed CPU packing step.
        self.xs=list(self.x.unbind());self.ys=list(self.y.unbind())
        self.flat_x=self.x.view(-1,self.width);self.flat_y=self.y.view(-1,self.width)
        self.expected=(self.x.float()@self.w.float()).bfloat16()
        torch.cuda.synchronize()
        for _ in range(5):self.enqueue()
        torch.cuda.synchronize();self.verify()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=self.root):self.enqueue()
        self.graph.replay();torch.cuda.synchronize();self.verify()

    def enqueue(self):
        with torch.cuda.stream(self.root):
            if self.strategy=='batched':
                torch.mm(self.flat_x,self.w,out=self.flat_y)
            elif self.strategy=='sequential':
                for x,y in zip(self.xs,self.ys):torch.mm(x,self.w,out=y)
            else:
                # Fork every branch BEFORE joining any branch; otherwise this would serialize.
                for branch in self.branches:branch.wait_stream(self.root)
                for branch,x,y in zip(self.branches,self.xs,self.ys):
                    with torch.cuda.stream(branch):torch.mm(x,self.w,out=y)
                for branch in self.branches:self.root.wait_stream(branch)

    def run(self,count,mode):
        for _ in range(count):
            if mode=='graph':self.graph.replay()
            else:self.enqueue()

    def verify(self):
        torch.testing.assert_close(self.y,self.expected,rtol=.02,atol=.02)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--profile',choices=['sequential','streams','batched']);ap.add_argument('--rows',type=int,default=32)
    args=ap.parse_args();cfg=json.loads((ROOT/'configs/standard.json').read_text())
    if args.smoke:cfg.update(repeats=1,rows=[32],seconds=.05,min_samples=3)
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if args.profile:
        w=Workload(args.rows,args.profile,cfg);w.run(10,'graph');torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range(args.profile):w.run(8,'graph');torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        helper.save(args.out/'capture.json',dict(strategy=args.profile,rows=args.rows,graph_replays=8,config=cfg))
        return
    manifest=dict(config=cfg,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  controller_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                  torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                  python=platform.python_version(),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    helper.save(args.out/'manifest.json',manifest);results=[]
    with (args.out/'samples.jsonl').open('w') as raw:
        for rep in range(cfg['repeats']):
            cases=[(rows,strategy,mode) for rows in cfg['rows'] for strategy in ['sequential','streams','batched'] for mode in ['ordinary','graph']]
            random.Random(cfg['seed']+rep).shuffle(cases)
            for index,(rows,strategy,mode) in enumerate(cases):
                w=Workload(rows,strategy,cfg)
                result,samples=helper.measure(lambda n:w.run(n,mode),cfg['seconds'],cfg)
                w.verify()
                results.append(dict(repeat=rep+1,rows=rows,strategy=strategy,mode=mode,correctness='PASS',**result))
                raw.write(json.dumps(dict(case=len(results)-1,seconds=samples))+'\n');raw.flush()
                helper.save(args.out/'summary.json',results);helper.save(args.out/'progress.json',dict(repeat=rep+1,case=index+1,total=len(cases)))
                del w
    manifest['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());helper.save(args.out/'manifest.json',manifest)
    (args.out/'COMPLETE').write_text('All cases completed; numerical checks passed.\n')

if __name__=='__main__':
    with torch.inference_mode():main()
