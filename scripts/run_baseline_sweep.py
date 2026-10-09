"""
Sequential-baseline sweep: the same (sensors, readings, window) problems and
seeds as run_experiments.py (both read scripts/experiment_config.py), run
with the sequential program. Needed so speed-up is measured against the true
sequential detector, not against the MPI program at one process.

Usage:
    python3 scripts/run_baseline_sweep.py --quick
    python3 scripts/run_baseline_sweep.py --full 
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_config import CONFIGS, BASE_SEED  # noqa: E402

BASELINE = ROOT / "src" / "sequential_baseline.py"

PATTERNS = {
    "gen_time": re.compile(r"Data generation time\s*:\s*([\d.]+)\s*s"),
    "detect_time": re.compile(r"Detection time\s*:\s*([\d.]+)\s*s"),
    "total_time": re.compile(r"Total time\s*:\s*([\d.]+)\s*s"),
    # two separately labelled figures, each = readings / the time it names
    "total_throughput": re.compile(r"Total throughput\s*:\s*([\d,]+)\s*readings/s"),
    "detection_throughput": re.compile(r"Detection-only throughput:\s*([\d,]+)\s*readings/s"),
    "precision": re.compile(r"Precision\s*:\s*([\d.]+)"),
    "recall": re.compile(r"Recall\s*:\s*([\d.]+)"),
}
KEY = ["sensors", "readings", "window"]
FIELDNAMES = KEY + ["trial", "seed"] + list(PATTERNS.keys())


def run_once(sensors, readings, window, seed):
    cmd = [sys.executable, str(BASELINE), "--sensors", str(sensors),
           "--readings", str(readings), "--window", str(window), "--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"  [FAILED] {' '.join(cmd)}\n{proc.stderr}", file=sys.stderr)
        return None
    parsed = {}
    for k, pat in PATTERNS.items():
        m = pat.search(proc.stdout)
        parsed[k] = float(m.group(1).replace(",", "")) if m else None
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS["quick" if args.quick else "full"]
    outdir = Path(args.outdir) if args.outdir else ROOT / "results" / cfg["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "raw_baseline.csv"

    done = {}
    if args.resume and raw_path.exists():
        with open(raw_path, newline="") as f:
            for row in csv.DictReader(f):
                k = tuple(str(row[c]) for c in KEY)
                done[k] = done.get(k, 0) + 1
    mode_flag = "a" if (args.resume and raw_path.exists()) else "w"

    combos = [(s, w) for s in cfg["sensor_counts"] for w in cfg["windows"]]
    with open(raw_path, mode_flag, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if mode_flag == "w":
            writer.writeheader()
        for i, (s, w) in enumerate(combos, 1):
            if done.get((str(s), str(cfg["readings"]), str(w)), 0) >= cfg["n_trials"]:
                print(f"[{i}/{len(combos)}] skip (already complete): sensors={s} window={w}")
                continue
            print(f"[{i}/{len(combos)}] sensors={s} readings={cfg['readings']} window={w}")
            if cfg["warmup"]:
                print("  warm-up trial (untimed, discarded)...")
                run_once(s, cfg["readings"], w, args.seed)
            for trial in range(1, cfg["n_trials"] + 1):
                seed = args.seed + trial
                r = run_once(s, cfg["readings"], w, seed)
                if r is None:
                    continue
                writer.writerow({"sensors": s, "readings": cfg["readings"], "window": w,
                                 "trial": trial, "seed": seed, **r})
                csvfile.flush()
                print(f"  trial {trial}/{cfg['n_trials']}: detect_time={r['detect_time']}s "
                      f"total_time={r['total_time']}s")
    print(f"\nDone. Baseline results: {raw_path}")


if __name__ == "__main__":
    main()
