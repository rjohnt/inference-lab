"""Run a local PowerShell script through the existing WSL SSH connection."""
import base64, subprocess, sys
from pathlib import Path
script=Path(sys.argv[1]).read_text()
encoded=base64.b64encode(script.encode('utf-16le')).decode()
if '--direct' in sys.argv:
    remote='powershell.exe -NoProfile -NonInteractive -EncodedCommand '+encoded
    conn=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-i',str(Path.home()/'.ssh/id_ed25519_example'),'-l','<windows-user>','<windows-host>']
else:
    remote='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -NonInteractive -EncodedCommand '+encoded
    conn=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','gpustation']
result=subprocess.run(conn+[remote],capture_output=True)
print(result.stdout.decode('utf-8',errors='replace').replace('\x00',''),end='')
print(result.stderr.decode('utf-8',errors='replace').replace('\x00',''),end='',file=sys.stderr)
sys.exit(result.returncode)
