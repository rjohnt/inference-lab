"""Identical single-GPU workers, optionally producing or consuming KV cache."""
import argparse
import json
import os
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('/workspace/disagg'))
p.add_argument('--gpu',type=int,required=True)
p.add_argument('--port',type=int,required=True)
p.add_argument('--batch-tokens',type=int,default=4096)
p.add_argument('--block-size',type=int,default=16)
p.add_argument('--ucx-proto-info',action='store_true')
p.add_argument('--role',choices=['replica','prefill','decode'],required=True)
a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
os.environ['VLLM_NIXL_SIDE_CHANNEL_HOST']='127.0.0.1'
os.environ['VLLM_NIXL_SIDE_CHANNEL_PORT']=str(5600+a.gpu)
os.environ['UCX_TLS']='all'
if a.ucx_proto_info:os.environ['UCX_PROTO_INFO']='y'
os.environ['VLLM_CACHE_ROOT']='/opt/disagg/vllm-cache'
cmd=['/opt/disagg/venv/bin/python','-m','vllm.entrypoints.cli.main','serve',str(a.root/'model'),
    '--served-model-name','qwen3','--host','127.0.0.1','--port',str(a.port),
    '--dtype','bfloat16','--max-model-len','33792','--max-num-seqs','16',
    '--block-size',str(a.block_size),'--max-num-batched-tokens',str(a.batch_tokens),'--gpu-memory-utilization','0.85',
    '--seed','42','--no-enable-prefix-caching','--generation-config','vllm']
if a.role!='replica':
    cmd+=['--kv-transfer-config',json.dumps({'kv_connector':'NixlConnector',
        'kv_role':'kv_producer' if a.role=='prefill' else 'kv_consumer',
        'kv_load_failure_policy':'fail','kv_connector_extra_config':{'backends':['UCX']}})]
os.execv(cmd[0],cmd)
