"""
Sequential windowed z-score anomaly detector -- the fair baseline.

For each sensor's stream, a rolling mean and standard deviation are
maintained over the preceding `window` readings using cumulative sums,
giving O(n) time per sensor rather than a naive O(n*window) recomputation.
A reading is flagged anomalous when its z-score exceeds `threshold`.

Usage:
    python3 src/sequential_baseline.py --sensors 50 --readings 10000 --window 50
"""

import argparse
import time
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_gen import generate_stream, precision_recall  # noqa: E402


def variance_floor(x):
    """Variance floor for one sensor, defined from that sensor's WHOLE
    series: 1% of its standard deviation, squared (never below 1e-6)."""
    return max((0.01 * float(np.std(x))) ** 2, 1e-6)


def windowed_zscore_detect(stream, window, threshold):
    """
    Sequential windowed z-score detector.
    """
    n_sensors, n_readings = stream.shape
    flags = np.zeros((n_sensors, n_readings), dtype=bool)

    for s in range(n_sensors):
        x = stream[s]
        var_floor = variance_floor(x)  # whole-series floor; shared with MPI code

        csum = np.cumsum(x)
        csum2 = np.cumsum(x * x)
        for i in range(window, n_readings):
            lo = i - window - 1
            total = csum[i - 1] - (csum[lo] if lo >= 0 else 0.0)
            total2 = csum2[i - 1] - (csum2[lo] if lo >= 0 else 0.0)
            mean = total / window
            var = max(total2 / window - mean * mean, var_floor)
            std = np.sqrt(var)
            z = abs(x[i] - mean) / std
            flags[s, i] = z > threshold

    return flags


def run(n_sensors, n_readings, window, threshold, anomaly_rate, seed, verbose=True):
    """Run the full baseline pipeline and return a results dict."""
    t0 = time.perf_counter()
    stream, labels = generate_stream(n_sensors, n_readings, anomaly_rate, seed)
    t1 = time.perf_counter()

    flags = windowed_zscore_detect(stream, window, threshold)
    t2 = time.perf_counter()

    precision, recall, tp, fp, fn = precision_recall(flags, labels)

    n_total = n_sensors * n_readings
    gen_time = t1 - t0
    detect_time = t2 - t1
    total_time = t2 - t0

    total_throughput = n_total / total_time if total_time > 0 else float("inf")
    detection_throughput = n_total / detect_time if detect_time > 0 else float("inf")

    result = {
        "gen_time": gen_time,
        "detect_time": detect_time,
        "total_time": total_time,
        "total_throughput": total_throughput,
        "detection_throughput": detection_throughput,
        "n_total_readings": n_total,
        "n_anomalies": int(labels.sum()),
        "n_flagged": int(flags.sum()),
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
    }

    if verbose:
        print(f"Config: sensors={n_sensors} readings={n_readings} "
              f"window={window} threshold={threshold}")
        print("\n--- Sequential baseline results ---")
        print(f"Data generation time    : {result['gen_time']:.4f} s")
        print(f"Detection time          : {result['detect_time']:.4f} s")
        print(f"Total time              : {result['total_time']:.4f} s")
        print(f"Total throughput        : {result['total_throughput']:,.0f} readings/s "
              f"(= {n_total} readings / {total_time:.4f} s total)")
        print(f"Detection-only throughput: {result['detection_throughput']:,.0f} readings/s "
              f"(= {n_total} readings / {detect_time:.4f} s detection)")
        print(f"Precision               : {result['precision']:.3f}")
        print(f"Recall                   : {result['recall']:.3f}")

    return result, flags, labels


def main():
    parser = argparse.ArgumentParser(description="Sequential windowed z-score baseline")
    parser.add_argument("--sensors", type=int, default=50)
    parser.add_argument("--readings", type=int, default=10000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--anomaly-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args.sensors, args.readings, args.window, args.threshold,
        args.anomaly_rate, args.seed)


if __name__ == "__main__":
    main()
