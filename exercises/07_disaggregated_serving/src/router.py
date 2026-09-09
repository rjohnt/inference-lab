"""One client-facing router for equal-budget replicas or NIXL P/D serving.

The P/D request protocol follows vLLM v0.24.0's NIXL integration proxy.
Missing transfer metadata is an error, never a local-prefill fallback.
"""
import argparse
from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
import uuid

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response
import uvicorn

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['replicas','pd'],required=True)
p.add_argument('--workers',nargs=2,default=['http://127.0.0.1:8100','http://127.0.0.1:8200'])
p.add_argument('--port',type=int,default=8000)
p.add_argument('--balance',choices=['requests','prompt_tokens'],default='requests')
p.add_argument('--events',type=Path,required=True)
a=p.parse_args()

@asynccontextmanager
async def lifespan(app):
    app.state.client=httpx.AsyncClient(timeout=300,limits=httpx.Limits(max_connections=128,max_keepalive_connections=64))
    a.events.parent.mkdir(parents=True,exist_ok=True)
    app.state.events=a.events.open('a',buffering=1)
    app.state.inflight=[0,0]
    app.state.work=[0,0]
    yield
    await app.state.client.aclose()
    app.state.events.close()

app=FastAPI(lifespan=lifespan)

@app.get('/health')
async def health():
    return {'mode':a.mode}

@app.post('/v1/completions')
async def complete(request:Request):
    data=await request.json()
    weight=len(data.get('prompt',[])) or 1
    request_id=str(uuid.uuid4())
    headers={'X-Request-Id':request_id}
    client=request.app.state.client
    begin=time.perf_counter()
    event={'started_unix_s':time.time(),'mode':a.mode,'request_id':request_id,'prefill_ms':None,'transfer_metadata':False}
    if a.mode=='pd':
        prefill=dict(data)
        prefill.update(stream=False,max_tokens=1,kv_transfer_params={
            'do_remote_decode':True,'do_remote_prefill':False,
            'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None})
        for key in ('stream_options','min_tokens','min_completion_tokens'): prefill.pop(key,None)
        response=await client.post(a.workers[0]+'/v1/completions',json=prefill,headers=headers)
        if response.status_code!=200: raise HTTPException(502,'Prefill worker failed: '+response.text[:500])
        transfer=response.json().get('kv_transfer_params')
        if not transfer or not transfer.get('remote_block_ids'):
            raise HTTPException(502,'Prefill returned no KV-transfer blocks')
        data['kv_transfer_params']=transfer
        event.update(prefill_ms=(time.perf_counter()-begin)*1000,transfer_metadata=True,
                     remote_block_count=len(transfer['remote_block_ids']))
        target=1
    else:
        target=min(range(2),key=lambda i:(request.app.state.inflight if a.balance=='requests' else request.app.state.work)[i])
    event['worker']=target
    request.app.state.inflight[target]+=1
    request.app.state.work[target]+=weight
    outgoing=client.build_request('POST',a.workers[target]+'/v1/completions',json=data,headers=headers)
    try:
        upstream=await client.send(outgoing,stream=True)
    except Exception:
        request.app.state.inflight[target]-=1
        request.app.state.work[target]-=weight
        raise
    if upstream.status_code!=200:
        body=await upstream.aread(); await upstream.aclose()
        request.app.state.inflight[target]-=1
        request.app.state.work[target]-=weight
        raise HTTPException(502,body.decode(errors='replace')[:500])
    if not data.get('stream'):
        body=await upstream.aread(); await upstream.aclose()
        request.app.state.inflight[target]-=1
        request.app.state.work[target]-=weight
        event['wall_ms']=(time.perf_counter()-begin)*1000
        request.app.state.events.write(json.dumps(event)+'\n')
        return Response(body,media_type='application/json')
    async def chunks():
        try:
            async for chunk in upstream.aiter_bytes(): yield chunk
        finally:
            await upstream.aclose()
            request.app.state.inflight[target]-=1
            request.app.state.work[target]-=weight
            event['wall_ms']=(time.perf_counter()-begin)*1000
            request.app.state.events.write(json.dumps(event)+'\n')
    return StreamingResponse(chunks(),media_type='text/event-stream')

if __name__=='__main__': uvicorn.run(app,host='127.0.0.1',port=a.port,access_log=False)
