# Windows commit-capacity workaround — 2026-09-07

DFlash failed CUDA allocations at 32K, including after Q8 draft KV caches and a 128-token target microbatch. A 16K attempt also failed while allocating draft weights. Windows reported only 4,318,184 KiB free virtual memory immediately before the workaround. The only pagefile was C:\pagefile.sys, allocated 4,625 MiB, on a nearly full C: drive.

Created a temporary 16,384 MiB pagefile at `E:\gpustation-benchmark-pagefile.sys` using the Windows native pagefile API. Free virtual memory increased to 20,976,408 KiB. No registry persistence was added, no existing pagefile was removed, and Windows was not restarted. This additional pagefile is active for the current Windows boot. The file may remain on E: after reboot; remove it only once Windows confirms it is no longer an active pagefile. Repeated future benchmarking may need a persistent pagefile configured on E:.

The script is `windows-benchmark-pagefile.ps1`; it refuses to overwrite an existing path. This action increases system commit capacity; it does not deliberately offload model layers or KV caches to CPU.

DFlash draft is stored on E: to avoid growing the WSL VHD on C:. A symlink in `~/benchmarks/specdec/models/` points to `/path/to/data/gpustation-benchmark-models/dflash-q8.gguf`.

Draft bytes: 1,120,250,336. Verified SHA256: `5be4f6b1bfd5c2c1aa753d4c03e30700114654fefbbf29f02257ef37adb00bf0` against pinned Hugging Face repository LFS metadata.
