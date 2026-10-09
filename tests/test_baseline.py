import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_gen import generate_stream, precision_recall  # noqa: E402
from sequential_baseline import windowed_zscore_detect  # noqa: E402


def test_generate_stream_shape():
    stream, labels = generate_stream(n_sensors=5, n_readings=200, seed=1)
    assert stream.shape == (5, 200)
    assert labels.shape == (5, 200)
    assert labels.dtype == bool


def test_generate_stream_deterministic_with_seed():
    s1, l1 = generate_stream(10, 500, seed=7)
    s2, l2 = generate_stream(10, 500, seed=7)
    assert np.array_equal(s1, s2)
    assert np.array_equal(l1, l2)


def test_generate_stream_different_seeds_differ():
    s1, _ = generate_stream(10, 500, seed=1)
    s2, _ = generate_stream(10, 500, seed=2)
    assert not np.array_equal(s1, s2)


def test_generate_stream_produces_some_anomalies():
    _, labels = generate_stream(20, 2000, anomaly_rate=0.02, seed=3)
    assert labels.sum() > 0


def test_windowed_zscore_detect_shape():
    stream, _ = generate_stream(5, 300, seed=4)
    flags = windowed_zscore_detect(stream, window=50, threshold=3.0)
    assert flags.shape == stream.shape
    assert flags.dtype == bool


def test_windowed_zscore_no_flags_before_window():
    """The first `window` readings of every sensor cannot be flagged,
    since there is not yet a full window of history."""
    stream, _ = generate_stream(5, 300, seed=4)
    window = 50
    flags = windowed_zscore_detect(stream, window=window, threshold=3.0)
    assert not flags[:, :window].any()


def test_windowed_zscore_detects_obvious_spike():
    """A synthetic sensor with a single huge, obvious spike must be flagged."""
    rng = np.random.default_rng(0)
    stream = rng.normal(0, 1.0, size=(1, 200))
    stream[0, 150] += 100.0  # enormous, unmistakable spike
    flags = windowed_zscore_detect(stream, window=50, threshold=3.0)
    assert flags[0, 150]


def test_windowed_zscore_flat_signal_no_false_positives():
    """A perfectly flat (zero-variance) signal should not crash and should
    not spuriously flag every point once variance floor is applied."""
    stream = np.full((2, 200), 42.0)
    flags = windowed_zscore_detect(stream, window=50, threshold=3.0)
    assert flags.shape == (2, 200)
    # constant signal has ~zero variance; with the var floor, z-scores for
    # unchanged values should be ~0, so nothing should be flagged
    assert not flags.any()


def test_precision_recall_perfect_match():
    labels = np.array([True, False, True, False])
    flags = np.array([True, False, True, False])
    precision, recall, tp, fp, fn = precision_recall(flags, labels)
    assert precision == 1.0
    assert recall == 1.0
    assert tp == 2 and fp == 0 and fn == 0


def test_precision_recall_no_predictions():
    labels = np.array([True, True, False])
    flags = np.array([False, False, False])
    precision, recall, tp, fp, fn = precision_recall(flags, labels)
    assert precision == 0.0
    assert recall == 0.0
    assert fn == 2


def test_baseline_precision_recall_reasonable_on_synthetic_data():
    """End-to-end sanity check."""
    stream, labels = generate_stream(n_sensors=8, n_readings=10000, seed=42)
    flags = windowed_zscore_detect(stream, window=50, threshold=3.0)
    precision, recall, _, _, _ = precision_recall(flags, labels)
    assert 0.5 < precision < 1.0
    assert 0.5 < recall < 1.0
