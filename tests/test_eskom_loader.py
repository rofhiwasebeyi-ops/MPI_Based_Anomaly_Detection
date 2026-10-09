import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eskom_loader import load_eskom_csv  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_eskom_export.csv")


def test_load_eskom_csv_basic_shape():
    stream, names, timestamps = load_eskom_csv(FIXTURE)
    assert stream.ndim == 2
    assert stream.shape[0] == len(names)
    assert stream.shape[1] == len(timestamps)


def test_load_eskom_csv_retains_expected_series():
    stream, names, _ = load_eskom_csv(FIXTURE)
    expected = {"Residual Demand", "RSA Contracted Demand",
                "Available Capacity", "Dispatchable Generation"}
    assert expected.issubset(set(names))


def test_load_eskom_csv_timestamps_parsed_and_sorted():
    _, _, timestamps = load_eskom_csv(FIXTURE)
    assert timestamps.is_monotonic_increasing
    assert timestamps[0].year == 2024


def test_load_eskom_csv_handles_missing_values_via_interpolation():
    """The fixture has 2 injected NaNs in RSA Contracted Demand; with the
    default interpolate fill method, no NaNs should remain."""
    stream, names, _ = load_eskom_csv(FIXTURE, fill_method="interpolate")
    idx = names.index("RSA Contracted Demand")
    assert not np.isnan(stream[idx]).any()


def test_load_eskom_csv_drops_columns_over_missing_threshold():
    """A column with excessive missingness should be dropped rather than
    silently filled, per the loader's documented behaviour."""
    import pandas as pd
    df = pd.read_csv(FIXTURE)
    # introduce a mostly-missing column
    df["Mostly Missing Metric"] = np.nan
    df.loc[0:5, "Mostly Missing Metric"] = 1.0
    tmp_path = "/tmp/test_mostly_missing.csv"
    df.to_csv(tmp_path, index=False)

    stream, names, _ = load_eskom_csv(tmp_path, max_missing_frac=0.2)
    assert "Mostly Missing Metric" not in names
    os.remove(tmp_path)


GAP_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                            "sample_eskom_export_with_gap.csv")


def test_load_eskom_csv_repairs_decimal_comma_rows():
    """Real Eskom exports sometimes write a decimal comma instead of a
    decimal point in one column, unquoted, which splits into two CSV
    fields and shifts every later column. The loader must repair this
    rather than silently misaligning columns (verified here by checking
    the timestamp column parses as real dates, not shifted numeric junk)."""
    stream, names, timestamps = load_eskom_csv(GAP_FIXTURE)
    # if columns were misaligned, timestamps would fail to parse as dates
    assert timestamps.notna().all()
    assert not np.isnan(stream).any()


def test_load_eskom_csv_truncates_structural_actual_forecast_gap():
    """When one column (an 'actual' series) has a large TRAILING block of
    missing values while another column (a 'forecast' series) is fully
    populated over the same range -- a real, observed Eskom export
    characteristic, not scattered missingness -- the loader must truncate
    to the mutually-valid range rather than interpolating/extrapolating
    fake values through the gap."""
    stream, names, timestamps = load_eskom_csv(GAP_FIXTURE)
    # the fixture's "Residual Demand" actual column is NaN from row 250
    # onward (of an original 300), so the truncated series must be shorter
    # than the full 300-row forecast range, and must contain no NaNs or
    # implausibly flat trailing values.
    assert stream.shape[1] < 300
    demand_idx = names.index("Residual Demand")
    # a genuinely truncated (not flatline-filled) series should retain
    # real variance right up to its last value, not repeat one constant
    tail = stream[demand_idx, -10:]
    assert len(set(tail.round(6))) > 1  # not all identical (no flatline)


def test_load_eskom_csv_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_eskom_csv("/tmp/does_not_exist_12345.csv")
