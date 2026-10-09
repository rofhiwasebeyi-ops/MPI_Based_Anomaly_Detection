"""
Loader for real Eskom Data Portal CSV exports.

Eskom Data Portal exports (obtained via the data request form at
https://www.eskom.co.za/dataportal/data-request-form/
"""

import argparse
import sys
import os
import csv
import io

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# Column names commonly seen across different Eskom Data Portal exports.
# The loader tries these in order and falls back to "first column with a
# parseable datetime" if none match.
CANDIDATE_DATETIME_COLS = [
    "Date Time Hour Beginning", "DateTime", "Date Time", "Timestamp", "Date",
]


def _repair_decimal_comma_rows(path):
    """
    Real Eskom Data Portal exports have a known data-quality defect: at
    least one column (observed: "Total UCLF+OCLF") is sometimes written
    with a comma as the decimal separator instead of a period, and that
    comma is not quoted -- so the CSV parser splits it into two fields and
    every later column shifts right by one for that row. Confirmed present
    in ~88% of rows of a real export (ESK19643.csv) used to develop this
    loader.

    This repairs the raw text before handing it to pandas: for any row
    with exactly one extra field compared to the header, it locates the
    two adjacent purely-numeric fields that, when joined with a period,
    parse as a float, and merges them back into one field.

    Returns a file-like object (io.StringIO) of the repaired CSV text.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return io.StringIO("")

    header = rows[0]
    n_expected = len(header)
    repaired = [header]
    n_repaired = 0
    n_dropped = 0

    for row in rows[1:]:
        if len(row) == n_expected:
            repaired.append(row)
        elif len(row) == n_expected + 1:
            merged = None
            for i in range(len(row) - 1):
                candidate = row[i] + "." + row[i + 1]
                try:
                    float(candidate)
                except ValueError:
                    continue
                # only accept if both halves look like a split decimal
                # (second half is a plain digit string -- a real fractional
                # part, not coincidentally-numeric unrelated fields)
                if row[i + 1].isdigit() and row[i].lstrip("-").isdigit():
                    merged = row[:i] + [candidate] + row[i + 2:]
                    break
            if merged and len(merged) == n_expected:
                repaired.append(merged)
                n_repaired += 1
            else:
                n_dropped += 1
        else:
            n_dropped += 1

    if n_repaired or n_dropped:
        print(f"[eskom_loader] repaired {n_repaired} decimal-comma rows, "
              f"dropped {n_dropped} unrecoverable rows out of {len(rows) - 1}",
              file=sys.stderr)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(repaired)
    buf.seek(0)
    return buf


def _find_datetime_column(df):
    for col in CANDIDATE_DATETIME_COLS:
        if col in df.columns:
            return col
    # fall back: first column that parses as a datetime for most rows
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                return col
        except Exception:
            continue
    raise ValueError(
        "Could not identify a datetime column. Expected one of "
        f"{CANDIDATE_DATETIME_COLS} or a column that parses as a date. "
        f"Found columns: {list(df.columns)}"
    )


def load_eskom_csv(path, max_missing_frac=0.2, fill_method="interpolate"):
    """
    Load an Eskom Data Portal CSV export.

    Parameters
    ----------
    path : str
        Path to the downloaded CSV.
    max_missing_frac : float
        A numeric column with more than this fraction of missing values is
        dropped rather than imputed (avoids fabricating a metric that is
        mostly absent).
    fill_method : {"interpolate", "ffill", "drop"}
        How to handle missing values in retained columns.

    Returns
    -------
    stream : ndarray, shape (n_series, n_readings)
        One row per retained numeric column, in chronological order.
    series_names : list of str
        Column name corresponding to each row of `stream`.
    timestamps : pandas.DatetimeIndex
    """
    df = pd.read_csv(_repair_decimal_comma_rows(path))
    if df.empty:
        raise ValueError(f"{path} contains no rows.")

    dt_col = _find_datetime_column(df)
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", format="mixed")
    df = df.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)

    numeric_cols = [
        c for c in df.columns
        if c != dt_col and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
    ]
    if not numeric_cols:
        raise ValueError(f"No numeric data columns found in {path} besides '{dt_col}'.")

    # First pass: decide which columns to retain by overall missing fraction
    # only (no filling yet -- filling happens after truncation below, so we
    # never interpolate/extrapolate across a large structural gap).
    candidate_cols = []
    for c in numeric_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        if col.isna().mean() <= max_missing_frac:
            df[c] = col
            candidate_cols.append(c)

    if not candidate_cols:
        raise ValueError(
            f"All numeric columns in {path} exceeded max_missing_frac={max_missing_frac}."
        )

    first_valid = [df[c].first_valid_index() for c in candidate_cols]
    last_valid = [df[c].last_valid_index() for c in candidate_cols]
    global_first = max(first_valid)
    global_last = min(last_valid)

    if global_first > 0 or global_last < len(df) - 1:
        n_before = len(df)
        dropped_head = global_first
        dropped_tail = n_before - 1 - global_last
        df = df.iloc[global_first:global_last + 1].reset_index(drop=True)
        print(f"[eskom_loader] truncated {dropped_head} leading and {dropped_tail} "
              f"trailing rows: at least one retained column had a structural "
              f"(non-scattered) gap there -- likely an actual/forecast date-range "
              f"mismatch rather than scattered missing readings. Retained range: "
              f"{df[dt_col].iloc[0]} to {df[dt_col].iloc[-1]}.", file=sys.stderr)

    retained = []
    for c in candidate_cols:
        col = df[c]
        if fill_method == "interpolate":
            col = col.interpolate(limit_direction="both")
        elif fill_method == "ffill":
            col = col.ffill().bfill()
        elif fill_method == "drop":
            pass  # leave NaNs; caller's responsibility
        df[c] = col
        retained.append(c)

    if fill_method == "drop":
        df = df.dropna(subset=retained)

    stream = df[retained].to_numpy(dtype=np.float64).T  # (n_series, n_readings)
    timestamps = pd.DatetimeIndex(df[dt_col])

    return stream, retained, timestamps


def main():
    parser = argparse.ArgumentParser(description="Load and summarise an Eskom CSV export")
    parser.add_argument("csv_path")
    parser.add_argument("--max-missing-frac", type=float, default=0.2)
    parser.add_argument("--fill-method", choices=["interpolate", "ffill", "drop"], default="interpolate")
    args = parser.parse_args()

    stream, names, timestamps = load_eskom_csv(
        args.csv_path, args.max_missing_frac, args.fill_method)

    print(f"Loaded {args.csv_path}")
    print(f"Series retained ({len(names)}): {names}")
    print(f"Readings per series: {stream.shape[1]}")
    print(f"Date range: {timestamps.min()} to {timestamps.max()}")
    for name, row in zip(names, stream):
        print(f"  {name}: mean={row.mean():.2f} std={row.std():.2f} "
              f"min={row.min():.2f} max={row.max():.2f}")


if __name__ == "__main__":
    main()
