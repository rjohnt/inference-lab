$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='Stop'
Get-Service sshd -ErrorAction SilentlyContinue | Select-Object Name,Status,StartType | ConvertTo-Json -Compress
Get-WindowsCapability -Online -Name 'OpenSSH.Server*' | Select-Object Name,State | ConvertTo-Json -Compress
Get-Item 'C:\Users\example\AppData\Local\wsl\{d26fa7b5-b95f-4ee9-8b0b-8b0406caf14a}\ext4.vhdx' | Select-Object FullName,Length | ConvertTo-Json -Compress
Get-Partition -DriveLetter C,D,E | Get-Disk | Select-Object Number,FriendlyName,BusType,HealthStatus,OperationalStatus | ConvertTo-Json -Compress
Get-CimInstance Win32_PageFileSetting | Select-Object Name,InitialSize,MaximumSize | ConvertTo-Json -Compress
wsl.exe --system --exec sh -c 'command -v e2fsck; command -v lsblk'
wsl.exe --version
