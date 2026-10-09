"""
Environment details for reproducibility (module requirement: "environment
details"), and the physical-core count used to refuse oversubscribed
scalability runs.

Run directly to print/write the environment:
    python3 src/env_info.py --out results/full/environment.json
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone


def logical_cores():
    return os.cpu_count() or 1


def physical_cores():
    """Physical core count. Tries psutil, then lscpu, then falls back to
    the logical count (an upper bound, so a fallback can only make the
    oversubscription check more permissive, never wrongly strict)."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    try:
        out = subprocess.run(["lscpu", "-p=Core,Socket"], capture_output=True,
                             text=True, timeout=10).stdout
        pairs = {tuple(line.split(",")) for line in out.splitlines()
                 if line and not line.startswith("#")}
        if pairs:
            return len(pairs)
    except Exception:
        pass
    return logical_cores()


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def cpu_model():
    m = re.search(r"model name\s*:\s*(.+)", _read("/proc/cpuinfo"))
    return m.group(1).strip() if m else platform.processor() or "unknown"


def total_ram_gb():
    m = re.search(r"MemTotal:\s+(\d+) kB", _read("/proc/meminfo"))
    return round(int(m.group(1)) / 1024 / 1024, 2) if m else None


def is_wsl():
    return "microsoft" in _read("/proc/version").lower()


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _pkg_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not installed"


def mpi_info():
    info = {"mpi4py": _pkg_version("mpi4py")}
    try:
        from mpi4py import MPI
        info["mpi_library"] = MPI.Get_library_version().strip().splitlines()[0]
    except Exception as e:
        info["mpi_library"] = f"unavailable ({e})"
    try:
        out = subprocess.run(["mpirun", "--version"], capture_output=True,
                             text=True, timeout=10)
        info["mpirun"] = (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        info["mpirun"] = "not found"
    return info


def collect(data_files=()):
    env = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "wsl": is_wsl(),
        "python": sys.version.split()[0],
        "cpu_model": cpu_model(),
        "logical_cores": logical_cores(),
        "physical_cores": physical_cores(),
        "ram_gb": total_ram_gb(),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "matplotlib": _pkg_version("matplotlib"),
        **mpi_info(),
        "data_files_sha256": {},
    }
    for path in data_files:
        if os.path.exists(path):
            env["data_files_sha256"][os.path.basename(path)] = sha256(path)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="write JSON here")
    parser.add_argument("--data", nargs="*", default=[],
                        help="data files whose SHA-256 should be recorded")
    args = parser.parse_args()
    env = collect(args.data)
    text = json.dumps(env, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
