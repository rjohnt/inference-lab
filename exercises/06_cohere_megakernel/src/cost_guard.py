"""Delete only the experiment pod at a deadline; keep credentials local to CLI."""
import argparse
import datetime
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--pod-id", required=True)
p.add_argument("--deadline", required=True, help="Unix timestamp")
a = p.parse_args()
deadline = float(a.deadline)
while time.time() < deadline:
    time.sleep(min(30, deadline - time.time()))
for attempt in range(10):
    result = subprocess.run(["runpodctl", "pod", "delete", a.pod_id], capture_output=True, text=True)
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), result.returncode,
          result.stdout, result.stderr, flush=True)
    if result.returncode == 0 or "not_found" in result.stderr:
        break
    time.sleep(30)
else:
    raise SystemExit("Pod cleanup failed; manual cleanup required")
