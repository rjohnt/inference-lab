"""Launch the real workload under ncu; --dry-run needs no CUDA or profiler."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-exercise", type=Path,
                        default=Path(os.environ.get("RMSNORM_EXERCISE_DIR", ROOT.parent / "04_kernel_fusion")))
    parser.add_argument("--python", type=Path, help="CUDA-enabled Python executable; defaults to kernel exercise .venv")
    parser.add_argument("--ncu", default=os.environ.get("NCU_BIN", "ncu"))
    parser.add_argument("--method", choices=["eager", "compiled"], required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--set", choices=["basic", "full"], default="basic")
    parser.add_argument("--out", type=Path, required=True, help="New local directory; never overwritten")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    kernel = args.kernel_exercise.resolve()
    # Keep the venv's path: resolving its symlink would select system Python!
    python = (args.python or kernel / ".venv/bin/python").absolute()
    workload = ROOT / "src/profile_workload.py"
    if not python.is_file() or not os.access(python, os.X_OK):
        parser.error("Set --python to an existing CUDA-enabled Python executable, or prepare the kernel exercise .venv")
    if not (kernel / "src/implementations.py").is_file():
        parser.error("--kernel-exercise must point to the existing kernel-fusion exercise")
    if args.rows < 1 or args.width < 1:
        parser.error("rows and width must be positive")
    ncu = shutil.which(args.ncu)
    if not ncu and not args.dry_run:
        parser.error("ncu was not found; set --ncu or NCU_BIN to the installed Nsight Compute executable")
    out = args.out.resolve()
    command = [ncu or args.ncu, "--mode", "launch-and-attach", "--target-processes", "all",
               "--set", args.set, "--nvtx", "--nvtx-include", "rmsnorm/",
               "--clock-control", "none", "--cache-control", "none", "--export", str(out / "report"),
               str(python), str(workload), "--kernel-exercise", str(kernel), "--method", args.method,
               "--rows", str(args.rows), "--width", str(args.width), "--dtype", args.dtype]
    if args.dry_run:
        print(shlex.join(command))
        return
    out.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output([ncu, "--version"], text=True)
    # Raw command paths/profiler metadata belong only in ignored local storage.
    (out / "metadata.json").write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
                                                   "ncu_version": version, "command": command}, indent=2) + "\n")
    with (out / "launch.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"ncu exited {result.returncode}; inspect {out / 'launch.log'}")
    report = out / "report.ncu-rep"
    if not report.is_file() or report.stat().st_size == 0:
        raise SystemExit("No report produced; inspect launch.log for missing kernels or counter-permission errors")
    print(f"Profile saved: {report}")


if __name__ == "__main__":
    main()
