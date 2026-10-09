"""
Correctness test: for every (partition, comm) strategy combination, running
the MPI detector functions directly (in-process, size=1 -- this sandbox
cannot launch true multi-rank MPI jobs) must produce a flags
array that exactly matches the sequential baseline on the same data.
"""

import sys
import os

import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_gen import generate_stream  # noqa: E402
from sequential_baseline import windowed_zscore_detect  # noqa: E402
from mpi_detector import run_sensor_partition, run_time_partition  # noqa: E402


SENSORS, READINGS, WINDOW, THRESHOLD, SEED = 8, 2000, 50, 3.0, 42


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def data():
    stream, labels = generate_stream(SENSORS, READINGS, seed=SEED)
    return stream, labels


def test_sensor_blocking_matches_sequential(comm, data):
    stream, _ = data
    expected = windowed_zscore_detect(stream, WINDOW, THRESHOLD)
    flags, _, _, _, _ = run_sensor_partition(comm, 0, 1, stream, WINDOW, THRESHOLD, "blocking", 500)
    assert np.array_equal(flags, expected)


def test_sensor_nonblocking_matches_sequential(comm, data):
    stream, _ = data
    expected = windowed_zscore_detect(stream, WINDOW, THRESHOLD)
    flags, _, _, _, _ = run_sensor_partition(comm, 0, 1, stream, WINDOW, THRESHOLD, "nonblocking", 500)
    assert np.array_equal(flags, expected)


def test_time_blocking_matches_sequential(comm, data):
    stream, _ = data
    expected = windowed_zscore_detect(stream, WINDOW, THRESHOLD)
    flags, _, _, _, _ = run_time_partition(comm, 0, 1, stream, WINDOW, THRESHOLD, "blocking", 500)
    assert np.array_equal(flags, expected)


def test_time_nonblocking_matches_sequential(comm, data):
    stream, _ = data
    expected = windowed_zscore_detect(stream, WINDOW, THRESHOLD)
    flags, _, _, _, _ = run_time_partition(comm, 0, 1, stream, WINDOW, THRESHOLD, "nonblocking", 500)
    assert np.array_equal(flags, expected)


def test_nonblocking_has_overlap_eligible_compute_time(comm, data):
    """Under the non-blocking strategy, detection compute for each batch
    must run between Iallreduce issue and Wait (the overlap window) -- so
    overlap-eligible compute time should be > 0 whenever there is any data
    to process. Blocking must report exactly 0 (no overlap is possible by
    definition)."""
    stream, _ = data
    _, _, _, _, overlap_nonblocking = run_sensor_partition(
        comm, 0, 1, stream, WINDOW, THRESHOLD, "nonblocking", 500)
    _, _, _, _, overlap_blocking = run_sensor_partition(
        comm, 0, 1, stream, WINDOW, THRESHOLD, "blocking", 500)
    assert overlap_nonblocking > 0
    assert overlap_blocking == 0


def test_batch_boundaries_match_unbatched_detection(comm, data):
    """Splitting into agg_interval batches (with the per-batch lookback
    halo) must produce exactly the same flags as a single unbatched call."""
    stream, _ = data
    expected = windowed_zscore_detect(stream, WINDOW, THRESHOLD)
    # a small, deliberately-not-evenly-dividing batch size to stress the
    # lookback/trim logic at several boundary positions
    flags, _, _, _, _ = run_sensor_partition(
        comm, 0, 1, stream, WINDOW, THRESHOLD, "blocking", agg_interval=137)
    assert np.array_equal(flags, expected)


def test_split_range_covers_full_range_no_overlap():
    from mpi_detector import split_range
    n, n_parts = 97, 5  # deliberately not evenly divisible
    covered = []
    for p in range(n_parts):
        start, end = split_range(n, n_parts, p)
        covered.extend(range(start, end))
    assert sorted(covered) == list(range(n))


def test_agreement_when_variance_floor_binds(comm):
    """The variance floor must be defined from the same quantity in the
    sequential and MPI detectors: each sensor's WHOLE series. Construct
    data where the floor decides the outcome."""
    stream = np.zeros((2, 1500))
    stream[:, 30] = 5000.0                       # huge early spike
    for t in (400, 700, 1000, 1300):             # small bumps, later batches
        stream[:, t] = 2.0
    expected = windowed_zscore_detect(stream, 24, 3.0)
    assert not expected[:, 400].any()            # sanity: floor suppresses the bump
    for fn in (run_sensor_partition, run_time_partition):
        for strategy in ("blocking", "nonblocking"):
            flags, _, _, _, _ = fn(comm, 0, 1, stream, 24, 3.0, strategy, 97)
            assert np.array_equal(flags, expected), (fn.__name__, strategy)
