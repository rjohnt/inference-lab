$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='Stop'
Get-NetTCPConnection -LocalPort 22 -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort,State,OwningProcess | ConvertTo-Json -Compress
Get-NetFirewallRule -Name 'GPUStation-SSH-Tailscale-Mac' | Select-Object Enabled,Profile,Action | ConvertTo-Json -Compress
$prefs = (& 'C:\Program Files\Tailscale\tailscale.exe' debug prefs | ConvertFrom-Json)
$prefs | Select-Object ShieldsUp,WantRunning,ForceDaemon | ConvertTo-Json -Compress
Get-NetIPAddress -IPAddress <windows-host> | Select-Object InterfaceAlias,IPAddress | ConvertTo-Json -Compress
Test-NetConnection -ComputerName <windows-host> -Port 22 -InformationLevel Quiet
