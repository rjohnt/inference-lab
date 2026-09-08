$ErrorActionPreference='Stop'
$p='E:\nsight-install\NsightSystems-2026.4.1.msi'
Get-AuthenticodeSignature $p | Select-Object Status,@{n='Signer';e={$_.SignerCertificate.Subject}} | ConvertTo-Json
$installer=New-Object -ComObject WindowsInstaller.Installer
$db=$installer.OpenDatabase($p,0)
$v=$db.OpenView('SELECT `Directory`,`Directory_Parent`,`DefaultDir` FROM `Directory`');$v.Execute()
while($r=$v.Fetch()) { if($r.StringData(1) -match 'INSTALL|TARGET|NVIDIA|NSIGHT|ProgramFiles') { '{0} | {1} | {2}' -f $r.StringData(1),$r.StringData(2),$r.StringData(3) } }
