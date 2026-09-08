$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
[pscustomobject]@{User=$identity.Name; Elevated=([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)} | ConvertTo-Json -Compress
Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | ForEach-Object { Get-ItemProperty $_.PSPath | Select-Object DistributionName,BasePath } | ConvertTo-Json -Compress
Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,FreeVirtualMemory,TotalVirtualMemorySize | ConvertTo-Json -Compress
Get-WinEvent -FilterHashtable @{LogName='System';StartTime=(Get-Date).AddHours(-4);Level=2} -MaxEvents 15 -ErrorAction SilentlyContinue | Select-Object TimeCreated,ProviderName,Id,Message | ConvertTo-Json -Depth 3
