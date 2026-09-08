"""Authenticated, serialized on-demand router for the installed llama-server."""
import configparser
import hmac
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
if not os.environ.get('GPUSTATION_KEY_FILE'):
    raise SystemExit('Set GPUSTATION_KEY_FILE to a local credential file outside Git')
KEY = Path(os.environ['GPUSTATION_KEY_FILE']).read_text().strip()
if not KEY:
    raise SystemExit('GPUSTATION_KEY_FILE must contain a nonempty key')
PRESETS = configparser.ConfigParser(interpolation=None)
PRESETS.read_string('[metadata]\n' + (ROOT / 'models.ini').read_text())
MODELS = [s for s in PRESETS.sections() if s not in ('metadata', '*')]
LOCK = threading.Lock()
PROCESS = None
CURRENT = None
LAST_USED = 0.0
PORT = 18081
IDLE = 120


def stop():
    global PROCESS, CURRENT
    if PROCESS is not None:
        PROCESS.terminate()
        try:
            PROCESS.wait(timeout=30)
        except subprocess.TimeoutExpired:
            PROCESS.kill()
            PROCESS.wait()
        print(f'unloaded {CURRENT}', flush=True)
    PROCESS = CURRENT = None


def ensure(model):
    global PROCESS, CURRENT
    if CURRENT == model and PROCESS is not None and PROCESS.poll() is None:
        return
    stop()
    settings = dict(PRESETS['*'])
    settings.update(PRESETS[model])
    args = ['/usr/local/lib/ollama/llama-server']
    for key, value in settings.items():
        if key in ('load-on-startup', 'sleep-idle-seconds'):
            continue
        if value == 'true':
            args.append('--' + key)
        elif value != 'false':
            args.extend(['--' + key, value])
    args += ['--alias', model, '--host', '127.0.0.1', '--port', str(PORT), '--timeout', '600', '--api-key-file', os.environ['GPUSTATION_KEY_FILE']]
    PROCESS = subprocess.Popen(args)
    CURRENT = model
    deadline = time.monotonic() + 540
    while time.monotonic() < deadline:
        if PROCESS.poll() is not None:
            stop()
            raise RuntimeError('Model process exited; inspect service journal')
        conn = http.client.HTTPConnection('127.0.0.1', PORT, timeout=2)
        try:
            conn.request('GET', '/health', headers={'Authorization': 'Bearer ' + KEY})
            response = conn.getresponse()
            response.read()
            if response.status == 200:
                print(f'loaded {model}', flush=True)
                return
        except OSError:
            pass
        finally:
            conn.close()
        time.sleep(.5)
    stop()
    raise TimeoutError('Model load timed out')


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + KEY):
            return True
        self.reply(401, {'error': {'message': 'Unauthorized'}})
        return False

    def do_GET(self):
        if self.path == '/health':
            self.reply(200, {'status': 'ok'})
        elif self.authorized():
            if self.path in ('/v1/models', '/models'):
                self.reply(200, {'object': 'list', 'data': [{'id': m, 'object': 'model', 'owned_by': 'gpustation', 'status': 'loaded' if CURRENT == m and PROCESS is not None and PROCESS.poll() is None else 'unloaded'} for m in MODELS]})
            else:
                self.reply(404, {'error': {'message': 'Unknown endpoint'}})

    def do_POST(self):
        global LAST_USED
        if not self.authorized():
            self.close_connection = True
            return
        if self.path not in ('/v1/chat/completions', '/v1/completions'):
            self.reply(404, {'error': {'message': 'Unknown endpoint'}})
            self.close_connection = True
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16 * 1024 * 1024:
                raise ValueError('Invalid request size')
            data = self.rfile.read(size)
            model = json.loads(data)['model']
            if model not in MODELS:
                raise ValueError('Unknown model')
        except (ValueError, KeyError):
            self.reply(400, {'error': {'message': 'Invalid request or unknown model'}})
            return
        # Hold through the entire response, including streaming, so switching cannot
        # terminate active inference. Idle shutdown uses this same lock.
        with LOCK:
            conn = None
            started = False
            try:
                ensure(model)
                conn = http.client.HTTPConnection('127.0.0.1', PORT, timeout=600)
                conn.request('POST', self.path, body=data, headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
                response = conn.getresponse()
                self.send_response(response.status)
                self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                started = True
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Stop cancelled work before another request can switch models.
                stop()
            except Exception as exc:
                print(f'request failed: {type(exc).__name__}: {exc}', flush=True)
                stop()
                if not started:
                    self.reply(502, {'error': {'message': str(exc)}})
                self.close_connection = True
            finally:
                if conn is not None:
                    conn.close()
                LAST_USED = time.monotonic()


def idle_monitor():
    while True:
        time.sleep(2)
        if LOCK.acquire(blocking=False):
            try:
                if PROCESS is not None and time.monotonic() - LAST_USED >= IDLE:
                    stop()
            finally:
                LOCK.release()


if __name__ == '__main__':
    threading.Thread(target=idle_monitor, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', 18080), Handler)
    print('router ready; all models unloaded', flush=True)
    try:
        server.serve_forever()
    finally:
        stop()
