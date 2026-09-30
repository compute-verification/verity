"""The disk floor and eviction (2026-09-27/28: the control pod's disk filled six times, the last time ending every guard on it):
`research run` and `research fetch --all` refuse under the floor and name the eviction; `research data custody --publish` removes the
run files it has just preserved on the remote."""
from __future__ import annotations

import json
import os

from research import cli
from research import runs as R
from research.store.cli import main as data_main

from test_run_custody import FIXTURE_RID, _fixture_run


# ---- the disk floor -----------------------------------------------------------------------------------------------------------------------

def test_the_floor_refuses_under_it_names_the_eviction_and_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "free_bytes", lambda p: 1_000_000_000)
    why = R.disk_floor_refusal(tmp_path / "runs" / "not-yet")
    assert why and "1.00 GB free" in why and "1.5 GB floor" in why and "research data evict --target-free-gb 3 --runs" in why
    assert R.disk_floor_refusal(tmp_path, {"RESEARCH_DISK_FLOOR_GB": "0.5"}) is None
    assert R.disk_floor_refusal(tmp_path, {"RESEARCH_DISK_FLOOR_GB": "0"}) is None
    monkeypatch.setattr(R, "free_bytes", lambda p: 2_000_000_000)
    assert R.disk_floor_refusal(tmp_path) is None


def test_research_run_and_fetch_all_refuse_under_the_floor(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "free_bytes", lambda p: 100_000_000)
    runs = tmp_path / "runs"
    assert cli.main(["run", "--runs-dir", str(runs), "--", "true"]) == 3
    assert "research run: refused: 0.10 GB free" in capsys.readouterr().err and not runs.exists()
    assert cli.main(["fetch", "r1", "--all", "--runs-dir", str(runs)]) == 3
    assert "research fetch --all: refused" in capsys.readouterr().err
    assert cli.main(["fetch", "r1", "--runs-dir", str(runs)]) == 1, "the records alone are small: past the floor, to the lookup"
    assert "no run matching" in capsys.readouterr().err


# ---- eviction after custody -----------------------------------------------------------------------------------------------------------------

def test_custody_publish_removes_the_big_run_files_it_just_preserved(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    runs = tmp_path / "runs"
    d = _fixture_run(runs)
    big = d / "out" / "proof.bin"
    big.write_bytes(b"p" * (2 << 20))
    os.utime(big, (1_600_000_000, 1_600_000_000))
    assert data_main(["custody", FIXTURE_RID, "--runs-dir", str(runs), "--publish"]) == 0
    out = capsys.readouterr().out
    assert "removed 1 local file(s), 2.1 MB" in out and not big.exists()
    assert (d / "job.json").exists() and (d / "stdout.log").exists(), "the small records stay"
    ev = json.loads((d / "run-files.evicted.json").read_text())
    assert [f["path"] for f in ev["files"]] == ["out/proof.bin"] and ev["restore"].startswith("research data fetch art:")
    assert data_main(["custody", FIXTURE_RID, "--runs-dir", str(runs)]) == 0, "custody still holds: the record is not a run file"
    assert data_main(["fetch", ev["artifact"], "--to", str(tmp_path / "back")]) == 0
    assert (tmp_path / "back" / "out" / "proof.bin").read_bytes() == b"p" * (2 << 20)


def test_custody_publish_keep_local_and_a_file_changed_since_the_check_stay(tmp_path, monkeypatch, capsys):
    from research.store.evict import evict_custody

    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    runs = tmp_path / "runs"
    d = _fixture_run(runs)
    big = d / "big.bin"
    big.write_bytes(b"q" * (2 << 20))
    os.utime(big, (1_600_000_000, 1_600_000_000))
    assert data_main(["custody", FIXTURE_RID, "--runs-dir", str(runs), "--publish", "--keep-local", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert big.exists() and "evicted" not in rep
    from research.store import evict as EV
    from research.store.local import LocalStore

    plan = EV.plan_runs(LocalStore(os.environ["RESEARCH_STORE"]), runs, quiet_s=0, now=4e9)
    assert [(r["artifact"], [f["path"] for f in r["files"]]) for r in plan] == [(rep["run_record"], ["big.bin"])], \
        "`research data evict --runs` finds a custody run through its run record"
    os.utime(big, (rep["verified_from"] + 1, rep["verified_from"] + 1))
    assert evict_custody(d, rep)["files"] == [] and big.exists(), "touched after the check: not what was verified"
    assert evict_custody(d, {**rep, "preserved": False})["files"] == []
