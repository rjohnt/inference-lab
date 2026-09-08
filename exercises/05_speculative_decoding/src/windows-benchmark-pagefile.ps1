$ErrorActionPreference = 'Stop'
$before = Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,FreeVirtualMemory,TotalVirtualMemorySize
$path = 'E:\gpustation-benchmark-pagefile.sys'
if (Test-Path $path) { throw 'Benchmark pagefile already exists; inspect before reuse.' }
if ((Get-Volume -DriveLetter E).SizeRemaining -lt 32GB) { throw 'Insufficient E: headroom' }
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class BenchmarkPagefile {
 [StructLayout(LayoutKind.Sequential)] public struct US { public ushort Length; public ushort MaximumLength; public IntPtr Buffer; }
 [DllImport("ntdll.dll")] static extern int RtlAdjustPrivilege(uint privilege, bool enable, bool currentThread, out bool previous);
 [DllImport("ntdll.dll")] static extern int NtCreatePagingFile(ref US name, ref long min, ref long max, uint priority);
 public static void Create(string path) {
  bool previous; int s=RtlAdjustPrivilege(15,true,false,out previous);
  if(s!=0) throw new Exception("Privilege status: "+s.ToString("X8"));
  string nt="\\??\\"+path; IntPtr buffer=Marshal.StringToHGlobalUni(nt);
  try { US name=new US{Length=(ushort)(nt.Length*2),MaximumLength=(ushort)((nt.Length+1)*2),Buffer=buffer};
   long min=16L*1024*1024*1024,max=min;
   s=NtCreatePagingFile(ref name,ref min,ref max,0);
   if(s!=0) throw new Exception("Pagefile status: "+s.ToString("X8"));
  } finally { Marshal.FreeHGlobal(buffer); bool ignored; RtlAdjustPrivilege(15,previous,false,out ignored); }
 }
}
'@
[BenchmarkPagefile]::Create($path)
@{Before=$before;After=(Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,FreeVirtualMemory,TotalVirtualMemorySize);Pagefiles=@(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage)} | ConvertTo-Json -Depth 4
