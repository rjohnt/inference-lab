"""Apply NVIDIA's WSL timestamp-conversion workaround without replacing other settings."""
from pathlib import Path
import re
import shutil
import subprocess

path=Path(subprocess.check_output(['nsys','-z'],text=True).strip())
if not path.is_absolute():
    raise RuntimeError('Nsight did not return an absolute configuration path')
text=path.read_text() if path.exists() else ''
key='CuptiUseRawGpuTimestamps=false'
pattern=r'^\s*CuptiUseRawGpuTimestamps\s*=.*$'
updated=re.sub(pattern,key,text,flags=re.MULTILINE) if re.search(pattern,text,re.MULTILINE) else text.rstrip()+'\n'+key+'\n'
if updated!=text:
    path.parent.mkdir(parents=True,exist_ok=True)
    backup=path.with_suffix(path.suffix+'.before-wsl-timestamps')
    if path.exists() and not backup.exists():shutil.copyfile(path,backup)
    path.write_text(updated)
print('Configured CUPTI timestamp conversion for WSL; trace timestamps have reduced accuracy.')
