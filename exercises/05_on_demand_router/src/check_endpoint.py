import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import os
if not all(os.environ.get(k) for k in ('INFERENCE_BASE_URL', 'GPUSTATION_KEY_FILE')):
    raise SystemExit('Set INFERENCE_BASE_URL and GPUSTATION_KEY_FILE outside Git')
key = Path(os.environ['GPUSTATION_KEY_FILE']).expanduser().read_text().strip()
base = os.environ['INFERENCE_BASE_URL'].rstrip('/')

def request(path, body=None, auth=True):
    headers = {'Content-Type':'application/json'}
    if auth:
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)

results = {}
for attempt in range(90):
    try:
        request('/models')
        break
    except (urllib.error.HTTPError, urllib.error.URLError):
        if attempt == 89:
            raise
        time.sleep(2)
try:
    request('/models', auth=False)
    raise AssertionError('Unauthenticated access unexpectedly allowed')
except urllib.error.HTTPError as e:
    assert e.code == 401, e.code
results['authentication'] = '401 without key'
results['models'] = request('/models')
body = dict(model='nemotron-nano-9b-v2', messages=[{'role':'system','content':'You are a helpful assistant. /no_think'}, {'role':'user','content':'What is 17 times 23? Answer with just the number.'}], temperature=0, max_tokens=128)
t=time.monotonic()
reply=request('/chat/completions',body)
assert '391' in reply['choices'][0]['message']['content'], reply
results['chat'] = reply
results['chat_seconds'] = time.monotonic()-t
body['messages'][1]['content'] = 'Use the get_weather tool to get the weather in Chicago.'
body['tools'] = [{'type':'function','function':{'name':'get_weather','description':'Get current weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
body['max_tokens']=512
reply=request('/chat/completions',body)
msg=reply['choices'][0]['message']
assert msg.get('tool_calls'), reply
call=msg['tool_calls'][0]
assert call['function']['name']=='get_weather',reply
assert json.loads(call['function']['arguments'])['city']=='Chicago',reply
results['tool_call']=reply
body['messages'] += [msg, {'role':'tool','tool_call_id':call['id'],'content':'{"city":"Chicago","temperature_c":21,"conditions":"sunny"}'}]
reply=request('/chat/completions',body)
assert '21' in reply['choices'][0]['message']['content'],reply
results['tool_result']=reply
out=Path(__file__).parent/'endpoint-check.json'
out.write_text(json.dumps(results,indent=2)+'\n')
print('PASS: authentication, chat, tool arguments, tool-result round trip')
print('Chat timings:', results['chat'].get('timings'))
