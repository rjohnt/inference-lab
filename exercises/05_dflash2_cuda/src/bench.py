"""Matched, resumable Qwen3.8 benchmark. Run inside tmux on GPUStation."""
import argparse
import collections
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import threading
import time
import traceback

import request_helpers as http

ROOT = Path(__file__).resolve().parent
SERVER = '/usr/local/lib/ollama/llama-server'
ENV = dict(os.environ, LD_LIBRARY_PATH='/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v13:/usr/lib/wsl/lib',
           GGML_BACKEND_PATH='/usr/local/lib/ollama/cuda_v13/libggml-cuda.so')
MODES = ['none', 'draft-dflash']

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

def start(mode, args, out, label, manifest):
    cmd = [SERVER, '-m', manifest['target']['path'], '--alias', 'qwen3-benchmark',
           '--host', '127.0.0.1', '--port', '18081', '-c', str(args.context), '-np', '1',
           '-ngl', 'all', '--fit', 'off', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
           '-b', '512', '-ub', str(args.ubatch), '--jinja', '--reasoning-format', 'deepseek',
           '--reasoning', 'off', '--metrics', '--cache-ram', '0', '--load-mode', 'dio',
           '--log-verbosity', '4', '--spec-type', mode]
    if mode == 'draft-dflash':
        cmd += ['-md', manifest['draft']['path'], '--spec-draft-ngl', 'all',
                '--spec-draft-n-max', '7', '--spec-draft-type-k', 'q8_0',
                '--spec-draft-type-v', 'q8_0']
    name = f'{label}-{mode}-{time.time_ns()}'
    log_path = out / (name + '.log')
    log = log_path.open('w')
    before = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    try:
        while time.monotonic() - before < 300:
            if proc.poll() is not None:
                raise RuntimeError(f'{mode} exited {proc.returncode}: {log_path}')
            try:
                healthy = http.obj('/health').get('status') == 'ok'
            except Exception:
                healthy = False
            if healthy:
                text = log_path.read_text()
                offloads = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', text)
                if not offloads or any(a != b for a, b in offloads) or 'no usable GPU found' in text:
                    raise RuntimeError(f'Full GPU offload not confirmed: {log_path}')
                if mode == 'draft-dflash' and 'selector_top_k' not in text:
                    raise RuntimeError(f'DFlash 2 selector metadata not confirmed: {log_path}')
                save(out / (name + '-command.json'), dict(command=cmd, startup_s=time.monotonic()-before,
                                                        offloads=offloads, pid=proc.pid))
                return proc, log, name
            time.sleep(0.5)
        raise TimeoutError(f'Server load timeout: {log_path}')
    except BaseException:
        stop(proc, log)
        raise

def stop(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.close()

def read_results(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    rows = []
    offset = 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise
            path.with_suffix('.interrupted-backup').write_bytes(data)
            with path.open('r+b') as stream:
                stream.truncate(offset)
            break
        offset += len(line)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=int, default=16384)
    parser.add_argument('--ubatch', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '.benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 18081))
    manifest = json.loads((ROOT / 'models.json').read_text())
    assert (ROOT / 'DOWNLOAD_COMPLETE').exists() and set(manifest) == {'target', 'draft'}
    original = (ROOT / 'source-prompts.json').read_bytes()
    config = dict(model='Qwen3.8-27B UD-IQ2_S', draft='DFlash 2 Q4_K_M',
                  context=args.context, ubatch=args.ubatch, repeats=args.repeats,
                  thinking='off', modes=MODES, draft_max=7, max_tokens=1536,
                  temperature=0, seed=42, cache_prompt=False,
                  prompts_sha256=hashlib.sha256(original).hexdigest(), models=manifest,
                  server_version=subprocess.check_output([SERVER, '--version'], env=ENV, text=True))
    config_path = args.out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Resume configuration differs'
    else:
        save(config_path, config)
    save(args.out / 'environment.json', dict(gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'], text=True),
                                            meminfo=Path('/proc/meminfo').read_text(), server_env={k:ENV[k] for k in ('LD_LIBRARY_PATH','GGML_BACKEND_PATH')}))
    cases = json.loads(original)
    result_path = args.out / 'results.jsonl'
    rows = read_results(result_path)
    done = {(r['rep'], r['mode'], r['case']) for r in rows}
    assert len(done) == len(rows), 'Duplicate result keys'
    phase = {'value': 'starting'}
    end = threading.Event()
    def monitor():
        fields = 'timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu'
        with (args.out / 'telemetry.jsonl').open('a') as file:
            while not end.is_set():
                sample = dict(unix_time=time.time(), phase=phase['value'], meminfo=Path('/proc/meminfo').read_text())
                try:
                    sample['gpu'] = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu='+fields,
                                                            '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
                except Exception as error:
                    sample['error'] = str(error)
                file.write(json.dumps(sample)+'\n')
                file.flush()
                end.wait(2)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        schedule = [(1, mode) for mode in reversed(MODES)] if args.probe else [
            (rep+1, mode) for rep in range(args.repeats)
            for mode in MODES[rep % 2:] + MODES[:rep % 2]]
        for rep, mode in schedule:
            selected = [c for c in cases if (rep, mode, c['id']) not in done]
            if not selected:
                continue
            phase['value'] = f'{mode}/rep-{rep}/startup'
            print('START', phase['value'], flush=True)
            proc, log, name = start(mode, args, args.out, f'rep-{rep}', manifest)
            try:
                for case in cases:
                    case['user_tokens'] = len(http.obj('/tokenize', dict(content=case['prompt'], add_special=False))['tokens'])
                    assert case['user_tokens'] + 1536 + 512 < args.context, 'Prompt and output do not fit'
                save(args.out / 'prompts.json', cases)
                http.stream('Reply with the single word Ready.', 64, thinking=False)
                random.Random(42+rep-1).shuffle(selected)
                for case in selected:
                    phase['value'] = f'{mode}/rep-{rep}/{case["id"]}'
                    save(args.out / 'status.json', dict(state='running', phase=phase['value'], completed=len(done),
                                                       expected=(12 if args.probe else args.repeats*12), updated=time.time()))
                    started = time.time()
                    result = http.stream(case['prompt'], 1536, thinking=False)
                    assert result['timings'].get('predicted_per_second', 0) > 0, 'Missing decode timing'
                    assert result['timings'].get('cache_n') == 0, 'Unexpected cached prompt'
                    result.update(rep=rep, mode=mode, case=case['id'], user_tokens=case['user_tokens'],
                                  thinking='off', request_unix_start=started, request_unix_end=time.time())
                    with result_path.open('a') as file:
                        file.write(json.dumps(result)+'\n')
                        file.flush()
                        os.fsync(file.fileno())
                    done.add((rep, mode, case['id']))
                    print(json.dumps({k:result[k] for k in ('rep','mode','case','total_s','finish_reason','timings')}), flush=True)
                with http.api('/metrics') as response:
                    (args.out / (name+'-metrics.txt')).write_bytes(response.read())
            finally:
                stop(proc, log)
        counts = collections.Counter((r['mode'],r['case']) for r in read_results(result_path))
        expected_repeats = 1 if args.probe else args.repeats
        assert len(counts) == 12 and all(n == expected_repeats for n in counts.values()), counts
        (args.out / 'COMPLETE').write_text(datetime.now(timezone.utc).isoformat()+'\n')
        save(args.out / 'status.json', dict(state='complete', completed=len(done), updated=time.time()))
        print('COMPLETE', args.out, flush=True)
    except BaseException:
        (args.out / 'FAILED.txt').write_text(traceback.format_exc())
        save(args.out / 'status.json', dict(state='failed', phase=phase['value'], completed=len(done), updated=time.time()))
        raise
    finally:
        end.set()
        thread.join(timeout=12)

if __name__ == '__main__':
    main()
