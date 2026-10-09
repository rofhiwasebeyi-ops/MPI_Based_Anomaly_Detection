# IT18X97 research artefact: blocking vs non-blocking MPI reductions in a streaming anomaly detector

One per-sensor windowed z-score detector, implemented sequentially and in MPI
(sensor- or time-partitioned; blocking `MPI_Allreduce` or non-blocking
`MPI_Iallreduce` + `MPI_Wait` for a periodic pooled-statistics reduction).
Route: **balanced** (2 partitionings x 2 communication strategies, compared
under one experiment harness). Real Eskom hourly records are **replayed**
offline for a case study.

## 1. Quick verification (minutes)

```bash
pip install -r requirements.txt
sudo apt-get install -y mpich libmpich-dev        # Ubuntu / WSL2
python3 scripts/run_all.py --quick                # all steps, small matrix
```

This records the environment, runs the 36 tests, runs the sequential and MPI
sweeps on a small matrix (8 sensors, 2 trials), analyses them, runs the Eskom
case study, and writes `results/quick/` with a `MANIFEST.json` of checksums.
On a machine with fewer than 4 physical cores pass `--processes 1 2` (or `1`):
process counts above the physical core count are **refused**, because
oversubscribed runs are not valid scalability evidence.

## 2. Full experiments (the ones reported in the paper)

```bash
python3 scripts/run_all.py --full                       # 96 MPI configurations, 1-8 processes
python3 scripts/run_all.py --full --processes 1 2 4     # if you have 4 physical cores
```

The matrix is defined once, in `scripts/experiment_config.py`, and both
sweeps read it: 2 partitionings x 2 reductions x process counts {1,2,4,8} x
sensors {50,200,500} x window {50,200}, 10,000 readings per sensor, batch size
2000, one untimed warm-up and 5 timed trials per configuration. It is
recorded in `results/full/config.json`. Expect roughly one to four hours.
If your machine has fewer than 8 physical cores, reduce `--processes` and
state the highest count actually used in the paper.

## 3. Steps run individually

| Step | Command | Output |
|---|---|---|
| Environment | `python3 src/env_info.py --out results/full/environment.json --data data/ESK19643.csv` | `environment.json` |
| Tests | `python3 -m pytest tests/ -v` | pass/fail |
| Sequential baseline | `python3 scripts/run_baseline_sweep.py --full` | `raw_baseline.csv` |
| MPI sweep | `python3 scripts/run_experiments.py --full` | `raw_mpi.csv`, `config.json` |
| Analysis and figures | `python3 scripts/analyze_results.py --dir results/full` | tables, `figures/`, `key_numbers.json` |
| Eskom case study | `python3 src/run_eskom_case_study.py data/ESK19643.csv --outdir results/full` | `eskom_*` |

Single runs with every parameter exposed:

```bash
python3 src/sequential_baseline.py --sensors 50 --readings 10000 --window 50 --threshold 3.0 --seed 42
mpirun -np 4 python3 src/mpi_detector.py --sensors 200 --readings 10000 --window 50 \
    --partition sensor --comm nonblocking --agg-interval 2000 --seed 42 --verify
```

`--verify` also runs the sequential detector on the same data (outside the
timed region) and reports how many readings differ.

## 4. Layout

```
src/        data_gen.py  sequential_baseline.py  mpi_detector.py  eskom_loader.py
            run_eskom_case_study.py  env_info.py
scripts/    experiment_config.py  run_baseline_sweep.py  run_experiments.py
            analyze_results.py  run_all.py
tests/      unit, correctness, loader and artefact tests (+ fixtures/)
data/       ESK19643.csv   (public Eskom Data Portal export; SHA-256 in environment.json)
results/    quick/ and full/  (raw CSVs, processed tables, figures, environment, manifest)
requirements.txt
README.md
```

## 5. Determinism and seeds

Data are generated from a seeded NumPy generator. Trial k of every
configuration uses seed `42 + k` (warm-up uses 42) in **both** sweeps, so each
trial sees identical data in the baseline and the MPI program. Seeds are
stored per row in the raw CSVs. Timings vary between runs; seeds, data, flags,
agreement counts and precision/recall do not.

## 6. What the tests establish

* The MPI detector's flags equal the sequential detector's for all four
  partition/communication combinations, including with batching
  (`test_mpi_correctness.py`), and on sparse data where the variance floor
  decides the outcome (`test_agreement_when_variance_floor_binds`).
* Window membership: the window for reading *i* is readings *i-W..i-1*;
  batching with a lookback of W readings reproduces unbatched flags.
* The non-blocking path reports non-zero overlap-eligible compute time and
  the blocking path exactly zero.
* The Eskom loader repairs decimal-comma rows and truncates structural
  actual/forecast gaps instead of interpolating through them.
* The experiment matrix equals the paper's Table 1, both sweeps share it,
  oversubscription is refused, resume does not duplicate rows.

At one process the tests cannot exercise inter-rank communication; the
multi-rank agreement is checked by `--verify` inside the sweeps and
summarised in `key_numbers.json`.

## 7. Detector definition and design decisions

* Flag reading *i* of sensor *s* when `|x - mean| / max(std, sigma_min) > threshold`,
  with mean/std over that sensor's own window of the W readings before *i*.
  No flag depends on another sensor or rank.
* `sigma_min` is 1% of the sensor's whole-series standard deviation
  (`sequential_baseline.variance_floor`, shared by both programs). This keeps
  z-scores finite after flat windows in sparse metrics, and means the detector
  is defined for replayed/offline data; a live system would need a running estimate.
* The pooled (sum, sum of squares, count) statistic is reduced across ranks
  per batch purely to give the communication comparison a concrete collective.
  It never changes a flag.
* Blocking: compute the batch's flags, then `MPI_Allreduce`. Non-blocking:
  issue `MPI_Iallreduce`, compute the batch's flags while it is pending, then
  `MPI_Wait`. Issue time, wait time and overlap-eligible compute time are
  timed separately; whether overlap is achieved depends on the MPI library and
  is **measured, not assumed** (`overlap_table.csv`).
* Throughput is always readings processed divided by the elapsed seconds
  quoted beside it. The sequential program prints total and detection-only
  throughput separately, each with its formula.

## 8. Environment

Recorded automatically in `results/<mode>/environment.json`: OS, WSL flag,
CPU model, physical and logical cores, RAM, Python, NumPy, pandas, matplotlib,
mpi4py, MPI library and `mpirun` versions, and the SHA-256 of the Eskom file.
Development was done on Ubuntu under WSL2 with MPICH.

## 9. Data and ethics

`data/ESK19643.csv` is a public Eskom Data Portal export (hourly system
data; 43,824 rows, April 2022 to March 2027, of which 38,712 rows up to
30 August 2026 contain observed values). It contains no personal data.
Cite it as: Eskom, Eskom Data Portal, https://www.eskom.co.za/dataportal/
(accessed 2026). The loader repairs ~88% of rows in this file that write a
decimal comma, and truncates the forecast-only tail rather than filling it.
The Eskom file has no ground-truth labels, so the case study reports flagged
events only; precision and recall use synthetic data with injected labels.
