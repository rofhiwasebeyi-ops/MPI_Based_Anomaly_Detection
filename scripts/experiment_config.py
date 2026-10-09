"""
Single source of truth for the experiment matrices.

Both sweeps (run_baseline_sweep.py and run_experiments.py) import these, so
the sequential baseline and the MPI runs are guaranteed to cover identical
(sensors, readings, window) problems.

Seeds: trial k of every configuration uses seed BASE_SEED + k (warm-up uses
BASE_SEED), in BOTH sweeps, so each trial sees identical data in the
baseline and the MPI program. Everything is deterministic given these seeds.
"""

BASE_SEED = 42

QUICK = {
    "name": "quick",
    "partitions": ["sensor", "time"],
    "comms": ["blocking", "nonblocking"],
    "process_counts": [1, 2, 4],
    "sensor_counts": [8],
    "readings": 10000,
    "windows": [50],
    "agg_interval": 500,
    "n_trials": 2,
    "warmup": True,
}

FULL = {
    "name": "full",
    "partitions": ["sensor", "time"],
    "comms": ["blocking", "nonblocking"],
    "process_counts": [1, 2, 4, 8],   # highest count must not exceed physical cores
    "sensor_counts": [50, 200, 500],
    "readings": 10000,
    "windows": [50, 200],
    "agg_interval": 2000,
    "n_trials": 5,
    "warmup": True,
}

CONFIGS = {"quick": QUICK, "full": FULL}
