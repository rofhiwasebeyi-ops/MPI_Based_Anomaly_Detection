"""
Screen a real Eskom Data Portal export with the SAME detector used
everywhere else (src/sequential_baseline.windowed_zscore_detect).

Usage:
    python3 src/run_eskom_case_study.py data/ESK19643.csv --outdir results/full
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eskom_loader import load_eskom_csv  # noqa: E402
from sequential_baseline import windowed_zscore_detect, variance_floor  # noqa: E402
import env_info  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Eskom replay case study")
    parser.add_argument("csv_path")
    parser.add_argument("--window", type=int, default=24,
                        help="rolling window in readings (24 = 1 day for hourly data)")
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--max-missing-frac", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--outdir", default=None,
                        help="write summary/CSV/figure here (omit to print only)")
    args = parser.parse_args()

    stream, names, timestamps = load_eskom_csv(args.csv_path, args.max_missing_frac)
    n_series, n_readings = stream.shape
    print(f"Loaded {n_series} series x {n_readings} readings from {args.csv_path}")
    print("NOTE: REPLAY of historical records (offline), not a live deployment.")

    if n_readings <= args.window:
        sys.exit(f"ERROR: only {n_readings} readings but window={args.window}. "
                 f"Reduce --window or use a longer export.")

    flags = windowed_zscore_detect(stream, args.window, args.threshold)
    total_flagged = int(flags.sum())
    print(f"\nTotal flagged readings: {total_flagged} "
          f"({100 * total_flagged / flags.size:.2f}% of all readings)")

    counts = []
    for name, row in zip(names, flags):
        counts.append((name, int(row.sum()), 100.0 * row.sum() / len(row)))
        print(f"  {name}: {counts[-1][1]} flagged ({counts[-1][2]:.2f}%)")

    # z-score of each flagged reading, using the detector's own definition:
    # window = the W readings before it, floor from the sensor's whole series
    events = []
    for s_idx, name in enumerate(names):
        x = stream[s_idx]
        floor_std = np.sqrt(variance_floor(x))
        for i in np.flatnonzero(flags[s_idx]):
            w = x[i - args.window:i]
            z = abs(x[i] - w.mean()) / max(w.std(), floor_std)
            events.append((z, timestamps[i], name, float(x[i])))
    events.sort(key=lambda e: e[0], reverse=True)

    print(f"\nTop {args.top_n} flagged events by |z-score|:")
    for z, ts, name, val in events[:args.top_n]:
        print(f"  {ts}  {name:30s} value={val:10.2f}  z={z:.2f}")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        import csv
        with open(os.path.join(args.outdir, "eskom_flag_counts.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["series", "flagged_readings", "flagged_pct"])
            for name, n, pct in counts:
                w.writerow([name, n, round(pct, 3)])
        with open(os.path.join(args.outdir, "eskom_top_events.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "series", "value", "abs_z"])
            for z, ts, name, val in events[:max(args.top_n, 100)]:
                w.writerow([ts, name, round(val, 3), round(float(z), 2)])
        summary = {
            "replay_not_live": True,
            "input_file": os.path.basename(args.csv_path),
            "input_sha256": env_info.sha256(args.csv_path),
            "series_retained": n_series,
            "readings_per_series": n_readings,
            "date_range": [str(timestamps.min()), str(timestamps.max())],
            "window": args.window, "threshold": args.threshold,
            "total_flagged": total_flagged,
            "flagged_pct": round(100 * total_flagged / flags.size, 3),
        }
        with open(os.path.join(args.outdir, "eskom_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if "Residual Demand" in names:
                idx = names.index("Residual Demand")
                fig, ax = plt.subplots(figsize=(11, 3.6))
                ax.plot(timestamps, stream[idx], lw=0.4, color="#4C72B0", label="Residual Demand")
                fi = np.flatnonzero(flags[idx])
                ax.scatter(timestamps[fi], stream[idx][fi], s=10, color="#C44E52",
                           label=f"flagged ({len(fi)})", zorder=3)
                ax.set_ylabel("MW")
                ax.set_title("Replayed Eskom hourly Residual Demand with flagged readings "
                             f"(W={args.window}, threshold={args.threshold})")
                ax.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, "eskom_residual_demand.png"), dpi=150)
                plt.close()
        except Exception as e:  # figure is a convenience; never fail the run
            print(f"(figure skipped: {e})", file=sys.stderr)
        print(f"\nWrote case-study outputs to {args.outdir}")


if __name__ == "__main__":
    main()
