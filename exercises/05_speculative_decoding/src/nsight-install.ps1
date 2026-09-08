$ErrorActionPreference='Stop'
$msi='E:\nsight-install\NsightSystems-2026.4.1.msi'
$sig=Get-AuthenticodeSignature $msi
if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=NVIDIA Corporation') { throw 'Unexpected installer signature' }
New-Item -ItemType Directory -Force 'E:\nsight-install\temp' | Out-Null
$env:TEMP='E:\nsight-install\temp';$env:TMP=$env:TEMP
$p=Start-Process msiexec.exe -ArgumentList '/i E:\nsight-install\NsightSystems-2026.4.1.msi /qn /norestart INSTALLDIR="E:\NVIDIA\Nsight Systems 2026.4.1" /L*v E:\nsight-install\install.log' -Wait -PassThru
Write-Output "Installer exit code: $($p.ExitCode)"
if ($p.ExitCode -notin @(0,3010)) { throw 'Nsight installation failed; inspect install.log' }
$cli=Get-ChildItem 'E:\NVIDIA\Nsight Systems 2026.4.1' -Filter nsys.exe -Recurse | Select-Object -First 1
if (!$cli) { throw 'Nsight CLI not found' }
Write-Output $cli.FullName
& $cli.FullName --version
