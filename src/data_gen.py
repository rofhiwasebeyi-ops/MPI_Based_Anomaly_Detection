"""
Synthetic multi-sensor electricity-demand stream generator.

Shared by the sequential baseline and the MPI implementation, so both are
guaranteed to see identical data for a given seed -- this is what makes the
numerical-agreement correctness check (parallel vs sequential output)
meaningful.
"""

import numpy as np


def generate_stream(n_sensors, n_readings, anomaly_rate=0.01, seed=42):
    """
    Build a synthetic (n_sensors, n_readings) demand matrix with labelled
    anomalies.

    Each sensor follows a smooth diurnal demand curve (sinusoidal base load)
    plus Gaussian noise. A fraction of readings are overwritten with
    injected anomalies (spikes or short sustained surges) at known index
    positions, which serve as ground-truth labels for precision/recall
    evaluation.

    Returns
    -------
    stream : ndarray, shape (n_sensors, n_readings)
    labels : ndarray of bool, shape (n_sensors, n_readings)
        True where a reading was deliberately injected as anomalous.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_readings)

    phase = rng.uniform(0, 2 * np.pi, size=(n_sensors, 1))
    amplitude = rng.uniform(0.8, 1.2, size=(n_sensors, 1))
    base = 50 + amplitude * 10 * np.sin(2 * np.pi * t / 500.0 + phase)

    noise = rng.normal(0, 1.5, size=(n_sensors, n_readings))
    stream = base + noise

    labels = np.zeros((n_sensors, n_readings), dtype=bool)

    n_anomalies = int(anomaly_rate * n_sensors * n_readings)
    sensor_idx = rng.integers(0, n_sensors, size=n_anomalies)
    time_idx = rng.integers(50, n_readings, size=n_anomalies)  # skip warm-up

    for s, ti in zip(sensor_idx, time_idx):
        if rng.random() < 0.5:
            stream[s, ti] += rng.choice([-1, 1]) * rng.uniform(15, 25)
        else:
            span = rng.integers(3, 6)
            end = min(ti + span, n_readings)
            stream[s, ti:end] += rng.uniform(10, 18)
            labels[s, ti:end] = True
        labels[s, ti] = True

    return stream, labels


def precision_recall(flags, labels):
    """Precision/recall of boolean flag array against boolean label array."""
    tp = int(np.sum(flags & labels))
    fp = int(np.sum(flags & ~labels))
    fn = int(np.sum(~flags & labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, tp, fp, fn
