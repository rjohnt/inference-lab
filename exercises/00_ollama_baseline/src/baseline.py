import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

BASE = 'http://127.0.0.1:11434'
MODEL = 'qwen3:8b-32k'
OUT = Path(__file__).parent

def request(path, data=None):
    return urllib.request.urlopen(urllib.request.Request(BASE + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'}), timeout=180)

def get_json(path, data=None):
    with request(path, data) as response:
        return json.load(response)

metadata = {'timestamp': datetime.now(timezone.utc).isoformat(),
    'endpoint': BASE, 'model': MODEL, 'version': get_json('/api/version'),
    'model_details': get_json('/api/show', {'model': MODEL}),
    'initial_residency': get_json('/api/ps'),
    'method': 'Client-side SSE timing over Tailscale; standalone prompts, no OpenCode system prompt/tools; default thinking; temperature 0.2; max_tokens 2048. One forced-cold ping, then three warm repetitions per prompt. Warm runs may reuse prompt cache.'}
(OUT / 'baseline-metadata.json').write_text(json.dumps(metadata, indent=2))
# Unload only the target model to measure a reproducible cold start.
get_json('/api/generate', {'model': MODEL, 'keep_alive': 0})
cases = [('cold', 'ping')]
for _ in range(3):
    cases += [('warm', p) for p in ['ping', "what\u0027s a diffusion model?", 'what model are you']]
with (OUT / 'baseline-results.jsonl').open('w') as output:
    for i, (state, prompt) in enumerate(cases):
        payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'stream': True, 'stream_options': {'include_usage': True},
            'temperature': 0.2, 'max_tokens': 2048}
        start = time.perf_counter()
        first = visible = None
        reasoning = content = ''
        usage = None
        finish = None
        with request('/v1/chat/completions', payload) as response:
            headers_s = time.perf_counter() - start
            for line in response:
                if not line.startswith(b'data: ') or line[6:].strip() == b'[DONE]':
                    continue
                chunk = json.loads(line[6:])
                now = time.perf_counter() - start
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta', {})
                    thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    answer = delta.get('content') or ''
                    if (thought or answer) and first is None:
                        first = now
                    if answer and visible is None:
                        visible = now
                    reasoning += thought
                    content += answer
                    finish = choice.get('finish_reason') or finish
        result = {'run': i + 1, 'state': state, 'prompt': prompt,
            'headers_s': headers_s, 'ttft_s': first, 'first_answer_s': visible,
            'total_s': time.perf_counter() - start,
            'reasoning_chars': len(reasoning), 'answer_chars': len(content),
            'usage': usage, 'finish_reason': finish, 'answer': content}
        output.write(json.dumps(result) + '\n')
        output.flush()
        print(json.dumps(result), flush=True)
(OUT / 'baseline-final-residency.json').write_text(json.dumps(get_json('/api/ps'), indent=2))
