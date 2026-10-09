"""
Tests of the artefact's own promises.
"""

import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import env_info  # noqa: E402
import experiment_config  # noqa: E402
import run_experiments  # noqa: E402
import run_baseline_sweep  # noqa: E402

PY = sys.executable
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "sample_eskom_export.csv")


def _run(args, **kw):
    return subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, **kw)


def test_full_matrix_matches_paper_table_1():
    cfg = experiment_config.FULL
    n_configs = (len(cfg["partitions"]) * len(cfg["comms"]) * len(cfg["process_counts"])
                 * len(cfg["sensor_counts"]) * len(cfg["windows"]))
    assert n_configs == 96                       # "96 configurations" in the paper
    assert cfg["process_counts"] == [1, 2, 4, 8]
    assert cfg["sensor_counts"] == [50, 200, 500]
    assert cfg["windows"] == [50, 200]
    assert cfg["n_trials"] == 5 and cfg["warmup"] is True
    assert cfg["readings"] == 10000 and cfg["agg_interval"] == 2000


def test_both_sweeps_read_the_same_config_object():
    assert run_experiments.CONFIGS is experiment_config.CONFIGS
    assert run_baseline_sweep.CONFIGS is experiment_config.CONFIGS
    assert run_experiments.BASE_SEED == run_baseline_sweep.BASE_SEED == experiment_config.BASE_SEED


def test_quick_config_is_small():
    q = experiment_config.QUICK
    assert q["sensor_counts"] == [8] and q["n_trials"] <= 3


def test_physical_cores_sane():
    assert 1 <= env_info.physical_cores() <= env_info.logical_cores()


def test_environment_record_has_required_fields(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("abc")
    env = env_info.collect([str(f)])
    for key in ("python", "numpy", "mpi4py", "mpi_library", "physical_cores",
                "logical_cores", "ram_gb", "platform", "cpu_model", "data_files_sha256"):
        assert key in env, key
    assert env["data_files_sha256"]["d.txt"] == env_info.sha256(str(f))
    assert env_info.sha256(str(f)) == env_info.sha256(str(f))   # deterministic


def test_same_seed_gives_identical_results():
    outs = [_run(["src/sequential_baseline.py", "--sensors", "6", "--readings", "2000",
                  "--seed", "7"]).stdout for _ in range(2)]
    pick = lambda s: [l for l in s.splitlines() if l.startswith(("Precision", "Recall"))]
    assert pick(outs[0]) == pick(outs[1]) and pick(outs[0])


def test_oversubscription_is_refused():
    r = _run(["scripts/run_experiments.py", "--quick", "--processes", "9999",
              "--outdir", os.path.join(ROOT, "results", "_test_oversub")])
    assert r.returncode != 0
    assert "physical core" in (r.stdout + r.stderr)


def test_baseline_resume_does_not_duplicate_rows(tmp_path):
    out = str(tmp_path / "r")
    assert _run(["scripts/run_baseline_sweep.py", "--quick", "--outdir", out]).returncode == 0
    rows1 = list(csv.DictReader(open(os.path.join(out, "raw_baseline.csv"))))
    assert _run(["scripts/run_baseline_sweep.py", "--quick", "--outdir", out, "--resume"]).returncode == 0
    rows2 = list(csv.DictReader(open(os.path.join(out, "raw_baseline.csv"))))
    assert len(rows1) == len(rows2) == experiment_config.QUICK["n_trials"]
    # trial k uses seed BASE_SEED + k in every sweep
    assert [int(r["seed"]) for r in rows1] == [experiment_config.BASE_SEED + 1,
                                               experiment_config.BASE_SEED + 2]


def test_eskom_case_study_writes_traceable_outputs(tmp_path):
    out = str(tmp_path / "e")
    r = _run(["src/run_eskom_case_study.py", FIXTURE, "--outdir", out, "--window", "24"])
    assert r.returncode == 0, r.stderr
    assert "REPLAY" in r.stdout
    for name in ("eskom_summary.json", "eskom_flag_counts.csv", "eskom_top_events.csv"):
        assert os.path.exists(os.path.join(out, name)), name
    summary = json.load(open(os.path.join(out, "eskom_summary.json")))
    assert summary["replay_not_live"] is True
    assert summary["input_sha256"] == env_info.sha256(FIXTURE)
