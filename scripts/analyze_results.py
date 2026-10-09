"""
Analysis pipeline for the experiment matrix.

Reads raw per-trial results from scripts/run_experiments.py and
scripts/run_baseline_sweep.py, aggregates to medians + IQR per
configuration, computes speedup/efficiency relative to the TRUE sequential
baseline.
  
Usage:
    python3 scripts/analyze_results.py \\
        --mpi-results results/full_results.csv \\
        --baseline-results results/full_baseline_results.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def load_and_aggregate_mpi(mpi_csv):
    df = pd.read_csv(mpi_csv)
    if "oversubscribed" in df.columns and df["oversubscribed"].astype(int).any():
        n = int(df["oversubscribed"].astype(int).sum())
        print(f"NOTE: excluding {n} oversubscribed trial row(s) from analysis.", file=sys.stderr)
        df = df[df["oversubscribed"].astype(int) == 0]
    group_cols = ["partition", "comm", "processes", "sensors", "readings", "window"]
    agg = df.groupby(group_cols).agg(
        median_wall_time=("wall_time", "median"),
        iqr_wall_time=("wall_time", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        median_comm_time=("comm_time", "median"),
        median_comm_issue_time=("comm_issue_time", "median"),
        median_comm_wait_time=("comm_wait_time", "median"),
        median_overlap_compute_time=("overlap_compute_time", "median"),
        median_compute_time=("compute_time", "median"),
        mean_comm_share_pct=("comm_share_pct", "mean"),
        # exact-agreement check vs the sequential baseline (reported
        # SEPARATELY from precision/recall: agreement = implementation
        # correctness; precision/recall = detection quality vs labels)
        max_mismatches=("mismatches_vs_sequential", "max"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        n_trials=("trial", "count"),
    ).reset_index()
    return agg


def load_and_aggregate_baseline(baseline_csv):
    df = pd.read_csv(baseline_csv)
    group_cols = ["sensors", "readings", "window"]
    agg = df.groupby(group_cols).agg(
        median_baseline_time=("detect_time", "median"),
    ).reset_index()
    return agg


def compute_speedup_efficiency(mpi_agg, baseline_agg):
    merged = mpi_agg.merge(baseline_agg, on=["sensors", "readings", "window"], how="left")

    missing_baseline = merged["median_baseline_time"].isna()
    if missing_baseline.any():
        print(f"WARNING: {missing_baseline.sum()} configuration(s) have no matching "
              f"baseline result (run scripts/run_baseline_sweep.py with matching "
              f"--sensors/--readings/--window). Speedup/efficiency will be NaN for "
              f"these rows.", file=sys.stderr)

    merged["speedup"] = merged["median_baseline_time"] / merged["median_wall_time"]
    merged["efficiency"] = merged["speedup"] / merged["processes"]
    return merged


def make_summary_table(merged, out_path):
    cols = ["partition", "comm", "processes", "sensors", "readings", "window",
            "median_wall_time", "iqr_wall_time", "speedup", "efficiency",
            "mean_comm_share_pct", "median_comm_issue_time", "median_comm_wait_time",
            "median_overlap_compute_time", "max_mismatches",
            "mean_precision", "mean_recall", "n_trials"]
    table = merged[cols].sort_values(["partition", "comm", "sensors", "window", "processes"]).copy()
    round_map = {
        "median_wall_time": 4, "iqr_wall_time": 4, "speedup": 3, "efficiency": 3,
        "mean_comm_share_pct": 2, "median_comm_issue_time": 5,
        "median_comm_wait_time": 5, "median_overlap_compute_time": 4,
        "mean_precision": 3, "mean_recall": 3,
    }
    table = table.round(round_map)
    table.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(table)} rows)")
    return table


def make_overlap_table(merged, out_path):
    keys = ["partition", "processes", "sensors", "readings", "window"]
    blk = merged[merged["comm"] == "blocking"].set_index(keys)
    nbk = merged[merged["comm"] == "nonblocking"].set_index(keys)
    common = blk.index.intersection(nbk.index)
    rows = []
    for k in common:
        b, n = blk.loc[k], nbk.loc[k]
        b_comm, n_comm = b["median_comm_time"], n["median_comm_time"]
        pct = 100.0 * (b_comm - n_comm) / b_comm if b_comm > 0 else float("nan")
        rows.append({
            **dict(zip(keys, k)),
            "blocking_comm_time": b_comm,
            "nonblocking_comm_time": n_comm,
            "nonblocking_issue_time": n["median_comm_issue_time"],
            "nonblocking_wait_time": n["median_comm_wait_time"],
            "overlap_compute_time": n["median_overlap_compute_time"],
            "relative_comm_time_reduction_pct": pct,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        print("WARNING: no configuration has both blocking and non-blocking "
              "results; overlap table not written.", file=sys.stderr)
        return table
    table = table.sort_values(["partition", "sensors", "window", "processes"]).round(5)
    table.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(table)} rows)")
    return table


def make_precision_recall_table(merged, out_path):
    cols = ["partition", "comm", "processes", "sensors", "window",
            "mean_precision", "mean_recall"]
    table = merged[cols].sort_values(["sensors", "window", "partition", "comm", "processes"]).copy()
    table = table.round({"mean_precision": 3, "mean_recall": 3})
    table.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(table)} rows)")
    return table


def plot_speedup_efficiency(merged, out_path):
    """One figure, two panels: speedup and efficiency vs process count,
    one line per (partition, comm) strategy. Faceted by sensor count if
    more than one is present -- otherwise a single pair of panels."""
    sensor_counts = sorted(merged["sensors"].unique())
    strategies = merged[["partition", "comm"]].drop_duplicates().values.tolist()

    n_facets = len(sensor_counts)
    fig, axes = plt.subplots(2, n_facets, figsize=(5.5 * n_facets, 8), squeeze=False)

    for col_idx, sensors in enumerate(sensor_counts):
        sub_all = merged[merged["sensors"] == sensors]
        ax_speedup = axes[0][col_idx]
        ax_eff = axes[1][col_idx]

        for partition, comm in strategies:
            sub = sub_all[(sub_all["partition"] == partition) & (sub_all["comm"] == comm)]
            if sub.empty:
                continue
            # average across window sizes if more than one, for a clean line
            line = sub.groupby("processes").agg(
                speedup=("speedup", "mean"), efficiency=("efficiency", "mean")
            ).reset_index().sort_values("processes")
            label = f"{partition}/{comm}"
            ax_speedup.plot(line["processes"], line["speedup"], marker="o", label=label)
            ax_eff.plot(line["processes"], line["efficiency"], marker="o", label=label)

        max_p = sub_all["processes"].max() if not sub_all.empty else 1
        ax_speedup.plot([1, max_p], [1, max_p], "k--", alpha=0.3, label="ideal")
        ax_speedup.set_title(f"Speedup (sensors={sensors})")
        ax_speedup.set_xlabel("Processes")
        ax_speedup.set_ylabel("Speedup vs. sequential baseline")
        ax_speedup.legend(fontsize=8)
        ax_speedup.grid(alpha=0.3)

        ax_eff.axhline(1.0, color="k", linestyle="--", alpha=0.3, label="ideal")
        ax_eff.set_title(f"Efficiency (sensors={sensors})")
        ax_eff.set_xlabel("Processes")
        ax_eff.set_ylabel("Efficiency (speedup / P)")
        ax_eff.legend(fontsize=8)
        ax_eff.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def plot_comm_compute_breakdown(merged, out_path):
    """Stacked bar: median compute time vs median comm time, one bar per
    configuration -- the explanatory ablation isolating where runtime
    differences originate."""
    merged = merged.sort_values(["sensors", "partition", "comm", "processes"])
    labels = [
        f"{r.partition[:3]}/{r.comm[:5]}\nP={r.processes},S={r.sensors}"
        for r in merged.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(merged)), 6))
    x = np.arange(len(merged))
    ax.bar(x, merged["median_compute_time"], label="Compute time", color="#4C72B0")
    ax.bar(x, merged["median_comm_time"], bottom=merged["median_compute_time"],
           label="Comm time", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Time (s)")
    ax.set_title("Compute vs. communication time breakdown per configuration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Turn raw sweep CSVs into the paper's tables and figures. "
                    "Reads <dir>/raw_mpi.csv and <dir>/raw_baseline.csv; writes "
                    "summary_table.csv, precision_recall_table.csv, overlap_table.csv, "
                    "key_numbers.json and figures/ into the same folder.")
    parser.add_argument("--dir", required=True,
                        help="results folder produced by the sweeps, e.g. results/full")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    mpi_csv = results_dir / "raw_mpi.csv"
    baseline_csv = results_dir / "raw_baseline.csv"
    for f in (mpi_csv, baseline_csv):
        if not f.exists():
            sys.exit(f"ERROR: {f} not found. Run run_experiments.py and "
                     f"run_baseline_sweep.py with the same mode first.")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    mpi_agg = load_and_aggregate_mpi(mpi_csv)
    baseline_agg = load_and_aggregate_baseline(baseline_csv)
    merged = compute_speedup_efficiency(mpi_agg, baseline_agg)

    make_summary_table(merged, results_dir / "summary_table.csv")
    make_precision_recall_table(merged, results_dir / "precision_recall_table.csv")
    overlap = make_overlap_table(merged, results_dir / "overlap_table.csv")
    plot_speedup_efficiency(merged, figures_dir / "speedup_efficiency.png")
    plot_comm_compute_breakdown(merged, figures_dir / "comm_compute_breakdown.png")

    verified = merged["max_mismatches"].notna()
    n_ver = int(verified.sum())
    n_exact = int((merged.loc[verified, "max_mismatches"] == 0).sum())
    n_bad = int((merged.loc[verified, "max_mismatches"] > 0).sum())
    print("\nSequential-vs-MPI output agreement (separate from precision/recall):")
    print(f"  {n_ver} of {len(merged)} configurations verified; "
          f"{n_exact} exact matches, {n_bad} with mismatches")

    best = merged.loc[merged["speedup"].idxmax()]
    worst_comm = merged.loc[merged["mean_comm_share_pct"].idxmax()]
    key_numbers = {
        "configurations": int(len(merged)),
        "agreement": {"verified": n_ver, "exact_matches": n_exact, "with_mismatches": n_bad},
        "best_speedup": {"value": round(float(best["speedup"]), 3),
                         "partition": best["partition"], "comm": best["comm"],
                         "processes": int(best["processes"]), "sensors": int(best["sensors"]),
                         "window": int(best["window"])},
        "highest_comm_share_pct": {"value": round(float(worst_comm["mean_comm_share_pct"]), 2),
                                   "partition": worst_comm["partition"], "comm": worst_comm["comm"],
                                   "processes": int(worst_comm["processes"]),
                                   "sensors": int(worst_comm["sensors"])},
    }
    if overlap is not None and len(overlap):
        key_numbers["relative_comm_time_reduction_pct"] = {
            "median": round(float(overlap["relative_comm_time_reduction_pct"].median()), 2),
            "min": round(float(overlap["relative_comm_time_reduction_pct"].min()), 2),
            "max": round(float(overlap["relative_comm_time_reduction_pct"].max()), 2),
        }
    (results_dir / "key_numbers.json").write_text(json.dumps(key_numbers, indent=2) + "\n")
    print("\nKey numbers for the paper (also written to key_numbers.json):")
    print(json.dumps(key_numbers, indent=2))


if __name__ == "__main__":
    main()
