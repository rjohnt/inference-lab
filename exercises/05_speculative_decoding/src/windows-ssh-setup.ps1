$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='Stop'
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
New-Item -ItemType Directory -Path "$env:ProgramData\ssh" -Force | Out-Null
$keyPath="$env:ProgramData\ssh\administrators_authorized_keys"
$key='<YOUR_SSH_PUBLIC_KEY>'
if (!(Test-Path $keyPath) -or !((Get-Content $keyPath) -contains $key)) { Add-Content -Encoding ascii -Path $keyPath -Value $key }
icacls.exe $keyPath /inheritance:r /grant '*S-1-5-32-544:F' /grant '*S-1-5-18:F' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Key ACL setup failed' }
Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue | Disable-NetFirewallRule
if (!(Get-NetFirewallRule -Name 'GPUStation-SSH-Tailscale-Mac' -ErrorAction SilentlyContinue)) {
 New-NetFirewallRule -Name 'GPUStation-SSH-Tailscale-Mac' -DisplayName 'GPU Station SSH from Mac via Tailscale' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 22 -LocalAddress <windows-host> -RemoteAddress <client-address> | Out-Null
}
Start-Service sshd
Set-Service sshd -StartupType Automatic
Get-Service sshd | Select-Object Name,Status | ConvertTo-Json -Compress
Get-Content "$env:ProgramData\ssh\ssh_host_ed25519_key.pub"
