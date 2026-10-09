"""
MPI experiment sweep.

Runs src/mpi_detector.py via mpirun for every combination in the chosen
configuration (scripts/experiment_config.py): one untimed warm-up, then
N timed trials per configuration. Raw per-trial rows go to
<outdir>/raw_mpi.csv; the configuration and environment actually used are
recorded beside them (config.json, environment.json).

Usage:
    python3 scripts/run_experiments.py --quick          # verification
    python3 scripts/run_experiments.py --full           # full experiments
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_config import CONFIGS, BASE_SEED  # noqa: E402
import env_info  # noqa: E402

DETECTOR = ROOT / "src" / "mpi_detector.py"

RESULT_PATTERNS = {
    "wall_time": re.compile(r"Wall time \(max over ranks\)\s*:\s*([\d.]+)\s*s"),
    "compute_time": re.compile(r"Compute time \(max over ranks\)\s*:\s*([\d.]+)\s*s"),
    "comm_time": re.compile(r"Comm time \(max over ranks\)\s*:\s*([\d.]+)\s*s"),
    "comm_issue_time": re.compile(r"\(= issue ([\d.]+)\s*s \+ wait"),
    "comm_wait_time": re.compile(r"\+ wait ([\d.]+)\s*s\)"),
    "comm_share_pct": re.compile(r"Comm share of wall time\s*:\s*([\d.]+)%"),
    # detection compute that ran while a non-blocking reduction was
    # outstanding (0 for blocking by definition): the measured overlap
    "overlap_compute_time": re.compile(r"Overlap-eligible compute time\s*:\s*([\d.]+)\s*s"),
    "throughput": re.compile(r"Throughput\s*:\s*([\d,]+)\s*readings/s"),
    "precision": re.compile(r"Detection precision \(vs labels\)\s*:\s*([\d.]+)"),
    "recall": re.compile(r"Detection recall \(vs labels\)\s*:\s*([\d.]+)"),
    # exact agreement vs the sequential baseline; present only with --verify
    # (trial 1 of each configuration), blank otherwise
    "mismatches_vs_sequential": re.compile(r"Mismatching readings vs sequential\s*:\s*(\d+)"),
}

KEY_FIELDS = ["partition", "comm", "processes", "sensors", "readings", "window"]
FIELDNAMES = KEY_FIELDS + ["trial", "seed", "oversubscribed", "external_wall_clock"] \
    + list(RESULT_PATTERNS.keys())


def run_once(partition, comm, n_procs, sensors, readings, window, agg_interval, seed,
             verify=False):
    cmd = ["mpirun", "-np", str(n_procs), sys.executable, str(DETECTOR),
           "--sensors", str(sensors), "--readings", str(readings),
           "--window", str(window), "--partition", partition, "--comm", comm,
           "--agg-interval", str(agg_interval), "--seed", str(seed)]
    if verify:
        cmd.append("--verify")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"  [FAILED] {' '.join(cmd)}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return None
    parsed = {"external_wall_clock": elapsed}
    for key, pattern in RESULT_PATTERNS.items():
        m = pattern.search(proc.stdout)
        parsed[key] = float(m.group(1).replace(",", "")) if m else None
    return parsed


def existing_trial_counts(csv_path):
    counts = {}
    if not csv_path.exists():
        return counts
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = tuple(str(row[k]) for k in KEY_FIELDS)
            counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--processes", type=int, nargs="+", default=None,
                        help="override the configuration's process counts")
    parser.add_argument("--outdir", default=None, help="default: results/<quick|full>")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--resume", action="store_true",
                        help="skip configurations whose trials are already in raw_mpi.csv")
    parser.add_argument("--allow-oversubscribe", action="store_true",
                        help="functional testing only; rows are marked and excluded from analysis")
    args = parser.parse_args()

    cfg = dict(CONFIGS["quick" if args.quick else "full"])
    if args.processes:
        cfg["process_counts"] = args.processes

    cores = env_info.physical_cores()
    too_many = [p for p in cfg["process_counts"] if p > cores]
    if too_many and not args.allow_oversubscribe:
        sys.exit(f"ERROR: process counts {too_many} exceed the {cores} physical core(s) detected. "
                 f"Oversubscribed runs are not valid scalability evidence. Pass --processes with "
                 f"values <= {cores}, or --allow-oversubscribe for functional testing only.")

    outdir = Path(args.outdir) if args.outdir else ROOT / "results" / cfg["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "raw_mpi.csv"

    (outdir / "config.json").write_text(json.dumps(
        {**cfg, "seed": args.seed, "physical_cores_detected": cores,
         "oversubscription_allowed": args.allow_oversubscribe}, indent=2) + "\n")
    (outdir / "environment.json").write_text(json.dumps(
        env_info.collect([str(ROOT / "data" / "ESK19643.csv")]), indent=2) + "\n")

    done = existing_trial_counts(raw_path) if args.resume else {}
    mode_flag = "a" if (args.resume and raw_path.exists()) else "w"

    combos = [(pa, co, p, s, w)
              for pa in cfg["partitions"] for co in cfg["comms"]
              for p in cfg["process_counts"] for s in cfg["sensor_counts"]
              for w in cfg["windows"]]

    with open(raw_path, mode_flag, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if mode_flag == "w":
            writer.writeheader()
        for i, (pa, co, p, s, w) in enumerate(combos, 1):
            key = (pa, co, str(p), str(s), str(cfg["readings"]), str(w))
            if done.get(key, 0) >= cfg["n_trials"]:
                print(f"[{i}/{len(combos)}] skip (already complete): {key}")
                continue
            print(f"\n[{i}/{len(combos)}] partition={pa} comm={co} procs={p} sensors={s} window={w}")
            if cfg["warmup"]:
                print("  warm-up trial (untimed, discarded)...")
                run_once(pa, co, p, s, cfg["readings"], w, cfg["agg_interval"], args.seed)
            for trial in range(1, cfg["n_trials"] + 1):
                seed = args.seed + trial
                result = run_once(pa, co, p, s, cfg["readings"], w, cfg["agg_interval"], seed,
                                  verify=(trial == 1))   # exact-agreement check once per config
                if result is None:
                    continue
                writer.writerow({"partition": pa, "comm": co, "processes": p, "sensors": s,
                                 "readings": cfg["readings"], "window": w, "trial": trial,
                                 "seed": seed, "oversubscribed": int(p > cores), **result})
                csvfile.flush()
                print(f"  trial {trial}/{cfg['n_trials']}: wall={result['wall_time']}s "
                      f"comm_share={result['comm_share_pct']}%")

    print(f"\nDone. Raw results: {raw_path}")


if __name__ == "__main__":
    main()
