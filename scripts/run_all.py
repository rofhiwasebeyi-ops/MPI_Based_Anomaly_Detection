"""
One command from raw code to every tabler.

    python3 scripts/run_all.py --quick     # verification run 
    python3 scripts/run_all.py --full      # full experiments 
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd, label):
    print(f"\n=== {label} ===\n$ {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"\nSTEP FAILED: {label} (exit {r.returncode}). Fix this before continuing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--processes", type=int, nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--eskom-csv", default="data/ESK19643.csv")
    args = parser.parse_args()

    name = "quick" if args.quick else "full"
    flag = f"--{name}"
    outdir = ROOT / "results" / name
    outdir.mkdir(parents=True, exist_ok=True)

    run([PY, "src/env_info.py", "--out", str(outdir / "environment.json"),
         "--data", args.eskom_csv], "1. record environment")
    if not args.skip_tests:
        run([PY, "-m", "pytest", "tests/", "-q"], "2. tests")

    extra = (["--resume"] if args.resume else [])
    run([PY, "scripts/run_baseline_sweep.py", flag, *extra], "3. sequential baseline sweep")
    mpi = [PY, "scripts/run_experiments.py", flag, *extra]
    if args.processes:
        mpi += ["--processes", *map(str, args.processes)]
    run(mpi, "4. MPI sweep")
    # run_experiments overwrote environment.json with the same fields; keep as is.
    run([PY, "scripts/analyze_results.py", "--dir", str(outdir)], "5. analysis")

    if (ROOT / args.eskom_csv).exists():
        run([PY, "src/run_eskom_case_study.py", args.eskom_csv, "--outdir", str(outdir)],
            "6. Eskom replay case study")
    else:
        print(f"\n(skipping Eskom case study: {args.eskom_csv} not found)")

    manifest = {}
    for p in sorted(outdir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            manifest[str(p.relative_to(outdir))] = {
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nDone. {len(manifest)} files in {outdir} (checksums in MANIFEST.json).")


if __name__ == "__main__":
    main()
