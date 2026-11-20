"""The runner's window road (`served_verdicts.sh`, ``ROAD=window``) through `verity run`'s own call (`served_table.served_window`)
on a window's record made on the processor."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity_cli import run as RUN
from verity_cli import served_table as ST

from tests.program.predict.test_close_record import _close
from tests.program.predict.test_window_rows import RUN as RUN_ROOT
from verity_vllm.program.predict.close_record import Record

ROOT = Path(__file__).parents[6]
COMPILER = {"name": "c", "digest": "cd" * 64}


def _serve_dir(tmp_path, monkeypatch):
    """TINY's window at class 2,4,2 closed as a serve's close leaves it (`Record.handoff`, then `Record.make` from the
    handoff alone, as close_record's finish does after the lease), under ``tmp_path/serve``; what only a card's serve writes
    beside it (its configuration, seal, commit verdict, advice registration and predict row's Build) stood in."""
    d = tmp_path / "serve"
    low, units, served, plan, binding, compiled = _close(d, monkeypatch)
    rec = Record("rate:1/4", "unread", tmp_path / "salt_stream", sample=2, workers=2)
    h = rec.handoff(plan, 0, low, units, binding)
    got = rec.make(plan, 0, compiled, low, units, served, SimpleNamespace(rc=binding.rc, com=None), handoff=Path(h["out"]))
    line = {"window": 0, "advice": "ef" * 64, "handoff": {"out": h["out"]}, "record": {"pending": "made after the lease"}}
    (d / "plans.jsonl").write_text(json.dumps(line) + "\n")
    (d / "serve.json").write_text(json.dumps({"scheme": "rows-sha512"}))
    (d / "seals.jsonl").write_text(json.dumps({"window": 0, "run_root": RUN_ROOT}) + "\n")
    commit = d / "windows" / "w0" / "commit"
    commit.mkdir(parents=True)
    (commit / "verdict.json").write_text(json.dumps({"run_root": got["run"], "commit_scheme": "rows-sha512"}))
    (commit / "runs.jsonl").write_text(json.dumps({"commit": {"binding": {"program_digest": "ab" * 64}}}) + "\n")
    (d / "advice" / "w0" / "registration.json").write_text(json.dumps(
        {"compiler": {**COMPILER, "cap_bits": 64}, "row_bits": 8, "binding": {"advice": {"root": "ef" * 64, "bits": 0}}}))
    (d / "advice" / "w0" / "private" / "advice.bin").write_bytes(b"")
    (d / "weights_registration.json").write_text("{}")
    (tmp_path / "predicted").mkdir()
    (tmp_path / "predicted" / "predicted.json").write_text(json.dumps({"compiler": COMPILER, "workload_digest": "12" * 64}))
    return d


def _road(tmp_path, monkeypatch, application, d):
    """``application``'s runner as `verity run` launches it on a node whose environment names the serve dir; its lines."""
    cfg = RUN.load(ROOT / "applications" / f"{application}.toml", RUN.registry())
    for k, v in {"D": str(d), "PRED": str(tmp_path / "predicted"), "SERVE_ROOT": str(tmp_path), "WINDOW_CLASS": "2,4,2",
                 "VENV": sys.prefix, "CUDA_HOME": str(tmp_path / "no-cuda"), "HF_HOME": str(tmp_path / "hf")}.items():
        monkeypatch.setenv(k, v)
    state = tmp_path / f"state-{application}"
    rc = ST.served_window(cfg, state, {})
    return rc, (state / "logs" / "served-window.log").read_text()


@pytest.mark.slow
def test_the_window_road_runs_remote_from_the_record_to_its_first_step_of_another_lane(tmp_path, monkeypatch):
    """remote through the preflight on the window's record (no ROW read), its seal held to the record and predict row's
    compiler, to the sampled proofs, which refuse by name as proofs'; Pearl+ refused as compute accounting's; the end line
    names each step. What no cheaper test reaches: the runner itself on a real window's record. Slow (minutes) for the close."""
    d = _serve_dir(tmp_path, monkeypatch)
    rc, log = _road(tmp_path, monkeypatch, "remote", d)
    assert "PREFLIGHT" not in log and "ROW" not in log.replace("no ROW read", ""), log
    assert "SEAL: the window's seal" in log and ": PASS" in log.split("SEAL: ")[1].splitlines()[0], log
    assert "COMPILER: the record's c cdcdcdcdcdcdcdcd…" in log and "BOUND-PROGRAM: the commit's bound Program digest abababababababab…" in log
    assert ("sampled-proofs: not run: step 3's prove path (prove_path/run.sh) proves the units a partition file names "
            "(PP_PARTITION), and the window's record has none: its proof of the window's circuit is proofs'") in log, log
    assert "pearl-plus: not run: Pearl+'s piece serves its own passes on a card" in log and "compute accounting's" in log
    end = next(x for x in log.splitlines() if "ROAD END" in x)
    assert "ran preflight,seal,window,advice" in end and "sampled-proofs" in end and "pearl-plus" in end and rc == 20, log


@pytest.mark.slow
def test_the_window_road_runs_an_onsite_application_without_compute_accounting(tmp_path, monkeypatch):
    """onsite_network lists no compute accounting, so POUW is empty and the preflight passes it; its network attester and
    wipe refuse by name as their lanes'. Slow (minutes) for the close."""
    d = _serve_dir(tmp_path, monkeypatch)
    rc, log = _road(tmp_path, monkeypatch, "onsite_network", d)
    assert "POUW" not in log and "pearl-plus" not in log, log
    assert "sampled-proofs: not run:" in log and "network-attester: not run:" in log and "wipe: not run:" in log, log
    assert rc == 20 and "ROAD END" in log, log


def test_the_window_road_refuses_by_name_each_thing_its_record_lacks(tmp_path):
    """An empty serve dir: the preflight names every file it wanted, and nothing named ROW."""
    env = {**os.environ, "D": str(tmp_path / "serve"), "PRED": str(tmp_path / "predicted"), "OUT": str(tmp_path / "out"),
           "SERVE_ROOT": str(tmp_path), "VENV": sys.prefix, "POUW": "pearlplus", "PRODUCT": "remote",
           "CUDA_HOME": str(tmp_path), "PEARLC_CHALLENGER_GATE": ""}
    (tmp_path / "serve").mkdir()
    p = subprocess.run(["bash", "benchmarks/one_stage/served_verdicts.sh"], cwd=ROOT, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    assert p.returncode == 3, out
    assert ("PREFLIGHT: the window's record lacks WINDOW_CLASS, the class the serve took (its --window-class), which "
            "serve.json doesn't record; D/plans.jsonl, the close's plan of each window; PRED/predicted.json, predict "
            "row's Build before the lease") in out, out
    assert "ROW" not in out
