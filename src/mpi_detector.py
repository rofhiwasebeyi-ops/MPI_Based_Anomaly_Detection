"""
MPI streaming anomaly-detection pipeline.

Two partitioning schemes (sensor-based, time-based) crossed with two 
communication strategies (blocking, non-blocking) for a periodic statistics-aggregation step.

Usage:
    mpirun -np 4 python3 src/mpi_detector.py \\
        --sensors 200 --readings 10000 --window 50 \\
        --partition sensor --comm blocking

    mpirun -np 4 python3 src/mpi_detector.py \\
        --partition time --comm nonblocking
"""

import argparse
import time
import sys
import os

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(__file__))
from data_gen import generate_stream, precision_recall  # noqa: E402
from sequential_baseline import variance_floor  # noqa: E402


def split_range(n, n_parts, part_idx):
    """Even-as-possible contiguous split of range(n) into n_parts, return
    the (start, end) half-open slice owned by part_idx."""
    base = n // n_parts
    rem = n % n_parts
    start = part_idx * base + min(part_idx, rem)
    end = start + base + (1 if part_idx < rem else 0)
    return start, end


def local_window_stats(x, window, var_floor):
    """Cumulative-sum rolling mean/std for a single 1-D array.
    Returns (mean, std) arrays aligned with x; undefined for i < window,
    since reading i's window [i-window, i-1] is only complete once
    i >= window.

    `var_floor` is the sensor's WHOLE-SERIES variance floor
    (sequential_baseline.variance_floor), passed in rather than computed
    from `x` here: `x` is only a batch/rank slice, and a floor taken from
    a slice would differ from the sequential detector's and break exact
    agreement on sparse data (e.g. Eskom's mostly-zero metrics)."""
    n = len(x)
    csum = np.cumsum(x)
    csum2 = np.cumsum(x * x)
    mean = np.zeros(n)
    std = np.zeros(n)
    for i in range(window, n):
        lo = i - window - 1
        total = csum[i - 1] - (csum[lo] if lo >= 0 else 0.0)
        total2 = csum2[i - 1] - (csum2[lo] if lo >= 0 else 0.0)
        m = total / window
        var = max(total2 / window - m * m, var_floor)
        mean[i] = m
        std[i] = np.sqrt(var)
    return mean, std


def detect_local(local_stream, window, threshold, var_floors):
    """Per-sensor windowed z-score flagging -- identical arithmetic to the
    sequential baseline, run only on this rank's local rows."""
    flags = np.zeros_like(local_stream, dtype=bool)
    for i, x in enumerate(local_stream):
        mean, std = local_window_stats(x, window, var_floors[i])
        z = np.zeros(len(x))
        z[window:] = np.abs(x[window:] - mean[window:]) / std[window:]
        flags[i] = z > threshold
    return flags


def detect_batch_with_lookback(local_data_2d, b_start, b_end, window, threshold, var_floors):
    """Detection for local columns [b_start, b_end), using a lookback of
    up to `window` preceding local columns so the batch's own first few
    readings still have a complete window (see module docstring, WINDOW
    BOUNDARIES ACROSS BATCHES AND RANKS)."""
    lookback_start = max(0, b_start - window)
    slice_with_lookback = local_data_2d[:, lookback_start:b_end]
    flags_with_lookback = detect_local(slice_with_lookback, window, threshold, var_floors)
    trim = b_start - lookback_start
    return flags_with_lookback[:, trim:]


def process_local_stream(comm, local_data_2d, window, threshold, comm_strategy, agg_interval,
                         var_floors, owned_start=0, owned_end=None):
    """
    Process local_data_2d (n_local_sensors, n_local_columns) in column
    batches of size `agg_interval`. Per batch: compute this rank's
    per-sensor detection flags for the batch (always; this is the
    correctness-critical path, identical regardless of comm_strategy),
    and separately aggregate the batch's pooled (sum, sumsq, count)
    across ranks via Allreduce (blocking) or Iallreduce+Wait
    (non-blocking).

    Returns (flags_2d, compute_time, comm_issue_time, comm_wait_time,
             overlap_compute_time).
    """
    n_sensors_local, n_cols = local_data_2d.shape
    flags = np.zeros((n_sensors_local, n_cols), dtype=bool)

    compute_time = 0.0
    comm_issue_time = 0.0
    comm_wait_time = 0.0
    overlap_compute_time = 0.0

    if owned_end is None:
        owned_end = n_cols

    owned_length = owned_end - owned_start

    if owned_length <= 0:
        return flags, compute_time, comm_issue_time, comm_wait_time, overlap_compute_time

    n_batches = (owned_length + agg_interval - 1) // agg_interval

    for b in range(n_batches):
        b_start = owned_start + b * agg_interval
        b_end = min(b_start + agg_interval, owned_end)

        batch = local_data_2d[:, b_start:b_end]

        local_sum = np.array([batch.sum()], dtype=np.float64)
        local_sumsq = np.array([(batch ** 2).sum()], dtype=np.float64)
        local_count = np.array([batch.size], dtype=np.float64)
        packed_local = np.concatenate([local_sum, local_sumsq, local_count])
        packed_global = np.zeros(3, dtype=np.float64)

        if comm_strategy == "blocking":
            t_compute_start = time.perf_counter()
            batch_flags = detect_batch_with_lookback(local_data_2d, b_start, b_end, window, threshold, var_floors)
            compute_time += time.perf_counter() - t_compute_start

            t_comm_start = time.perf_counter()
            comm.Allreduce(packed_local, packed_global, op=MPI.SUM)
            comm_issue_time += time.perf_counter() - t_comm_start

        else:  # nonblocking
            t_issue_start = time.perf_counter()
            req = comm.Iallreduce(packed_local, packed_global, op=MPI.SUM)
            comm_issue_time += time.perf_counter() - t_issue_start
            t_overlap_start = time.perf_counter()
            batch_flags = detect_batch_with_lookback(local_data_2d, b_start, b_end, window, threshold, var_floors)
            overlap_duration = time.perf_counter() - t_overlap_start
            compute_time += overlap_duration
            overlap_compute_time += overlap_duration

            t_wait_start = time.perf_counter()
            req.Wait()
            comm_wait_time += time.perf_counter() - t_wait_start

        flags[:, b_start:b_end] = batch_flags

    return flags, compute_time, comm_issue_time, comm_wait_time, overlap_compute_time


def run_sensor_partition(comm, rank, size, stream, window, threshold, comm_strategy,
                          agg_interval=2000):
    """Sensor-based partitioning: each rank owns a contiguous block of
    sensors and processes its full local stream through
    process_local_stream (see above)."""
    n_sensors, n_readings = stream.shape
    start, end = split_range(n_sensors, size, rank)
    local_stream = stream[start:end]
    # whole-series floors for the sensors this rank owns (shared definition
    # with the sequential detector)
    var_floors = np.array([variance_floor(stream[i]) for i in range(start, end)])

    local_flags, compute_time, comm_issue, comm_wait, overlap_compute = process_local_stream(
        comm, local_stream, window, threshold, comm_strategy, agg_interval, var_floors)

    all_flags = comm.gather(local_flags, root=0)
    flags = np.concatenate(all_flags, axis=0) if rank == 0 else None

    return flags, compute_time, comm_issue, comm_wait, overlap_compute


def run_time_partition(comm, rank, size, stream, window, threshold, comm_strategy,
                        agg_interval=2000):
    """Time-based partitioning: each rank owns a contiguous block of time
    indices across ALL sensors, with a `window`-sized halo pulled from
    the preceding rank's tail so rolling statistics are correct at
    RANK boundaries (a second, separate halo -- across BATCHES within
    one rank -- is handled inside process_local_stream)."""
    n_sensors, n_readings = stream.shape
    start, end = split_range(n_readings, size, rank)
    halo = window
    halo_start = max(0, start - halo)
    local_with_halo = stream[:, halo_start:end]
    trim = start - halo_start
    var_floors = np.array([variance_floor(stream[i]) for i in range(n_sensors)])
    flags_with_halo, compute_time, comm_issue, comm_wait, overlap_compute = process_local_stream(
        comm, local_with_halo, window, threshold, comm_strategy, agg_interval, var_floors,  owned_start=trim,
        owned_end=trim + (end - start))
    local_flags = flags_with_halo[:, trim:]

    all_flags = comm.gather(local_flags, root=0)
    flags = np.concatenate(all_flags, axis=1) if rank == 0 else None

    return flags, compute_time, comm_issue, comm_wait, overlap_compute


def main():
    parser = argparse.ArgumentParser(description="MPI streaming anomaly detector")
    parser.add_argument("--sensors", type=int, default=50)
    parser.add_argument("--readings", type=int, default=10000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--anomaly-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", choices=["sensor", "time"], default="sensor")
    parser.add_argument("--comm", choices=["blocking", "nonblocking"], default="blocking")
    parser.add_argument("--agg-interval", type=int, default=2000,
                         help="readings per statistics-aggregation batch")
    parser.add_argument("--verify", action="store_true",
                         help="after timing, run the sequential baseline on the same "
                              "data and report exact output agreement (outside the "
                              "timed region; adds serial runtime, so used sparingly)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    stream, labels = generate_stream(args.sensors, args.readings, args.anomaly_rate, args.seed)

    t0 = time.perf_counter()
    if args.partition == "sensor":
        flags, compute_time, comm_issue, comm_wait, overlap_compute = run_sensor_partition(
            comm, rank, size, stream, args.window, args.threshold, args.comm, args.agg_interval)
    else:
        flags, compute_time, comm_issue, comm_wait, overlap_compute = run_time_partition(
            comm, rank, size, stream, args.window, args.threshold, args.comm, args.agg_interval)
    t1 = time.perf_counter()

    comm_time = comm_issue + comm_wait

    max_compute = comm.reduce(compute_time, op=MPI.MAX, root=0)
    max_comm = comm.reduce(comm_time, op=MPI.MAX, root=0)
    max_comm_issue = comm.reduce(comm_issue, op=MPI.MAX, root=0)
    max_comm_wait = comm.reduce(comm_wait, op=MPI.MAX, root=0)
    max_overlap_compute = comm.reduce(overlap_compute, op=MPI.MAX, root=0)

    if rank == 0:
        precision, recall, tp, fp, fn = precision_recall(flags, labels)
        n_total = args.sensors * args.readings
        wall_time = t1 - t0
        # Throughput MUST be readings / elapsed seconds using the SAME
        # elapsed time it is quoted against -- see sequential_baseline.py
        # for why a mismatched denominator produces inconsistent figures.
        throughput = n_total / wall_time if wall_time > 0 else float("inf")

        print(f"Config: sensors={args.sensors} readings={args.readings} "
              f"window={args.window} partition={args.partition} comm={args.comm} "
              f"processes={size}")
        print("\n--- MPI results ---")
        print(f"Wall time (max over ranks)        : {wall_time:.4f} s")
        print(f"Compute time (max over ranks)     : {max_compute:.4f} s")
        print(f"Comm time (max over ranks)        : {max_comm:.6f} s "
              f"(= issue {max_comm_issue:.6f} s + wait {max_comm_wait:.6f} s)")
        print(f"Comm share of wall time           : {100 * max_comm / wall_time:.1f}%")
        print(f"Overlap-eligible compute time      : {max_overlap_compute:.6f} s "
              f"(compute that ran while a non-blocking reduction was outstanding; "
              f"0 for blocking, where no overlap is possible)")
        print(f"Throughput                        : {throughput:,.0f} readings/s "
              f"(= {n_total} readings / {wall_time:.4f} s wall time)")
    
        if args.verify:
            from sequential_baseline import windowed_zscore_detect
            expected = windowed_zscore_detect(stream, args.window, args.threshold)
            mismatches = int((flags != expected).sum())
            verdict = "EXACT MATCH" if mismatches == 0 else "MISMATCH"
            print(f"Agreement with sequential baseline: {verdict} "
                  f"({mismatches} mismatching readings of {n_total})")
            print(f"Mismatching readings vs sequential: {mismatches}")
        print(f"Detection precision (vs labels)    : {precision:.3f}")
        print(f"Detection recall (vs labels)       : {recall:.3f}")
        print(f"TP={tp} FP={fp} FN={fn}")


if __name__ == "__main__":
    main()
