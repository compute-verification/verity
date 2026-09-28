"""`tools/check/verdicts.py`: the passes one machine stored, carried to another as a pack; and `check.py`'s side of it."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"test_verdicts_{name}", ROOT / "tools" / "check" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m                                      # a module with dataclasses must be importable by name
    spec.loader.exec_module(m)
    return m


V = _load("verdicts")


def _machine(tmp_path: Path, monkeypatch, name: str) -> dict[str, Path]:
    """Point both caches at this machine's directories."""
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / name / "check"))
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / name / "tests"))
    return V.roots()


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _fill(r: dict[str, Path]) -> None:
    _write(r["tests"] / "k1.json", '{"suite": "a"}')
    _write(r["tests"] / "durations.json", "{}")                             # not a verdict
    _write(r["tests"] / "logs" / "x" / "a.json", "{}")                       # a log
    _write(r["check"] / "circuit-check" / "v1" / "d" / "e1.json", '{"report": 1}')
    _write(r["check"] / "circuit-check" / "v1" / "d" / ".e2.tmp", "partial")  # a write in progress
    _write(r["check"] / "lean-audit" / "k2.json", '{"package": "p"}')
    _write(r["check"] / "lean-agreement" / "k3.json", '{"key": "k3"}')
    _write(r["check"] / "lean-deps" / "m" / "manifest.json", "{}")           # fetched dependencies
    _write(r["check"] / "lean-audit-scratch-x1" / "p" / "lake-manifest.json", "{}")  # an audit's scratch tree
    _write(r["check"] / "lean-changed" / "p" / "lean-audit.json", "{}")      # lean_changed.py's built tree
    _write(r["check"] / "lean-changed" / "links.json", "{}")                 # check's early Lean step's tree
    _write(r["check"] / "lean-records" / "build-0" / "p" / "lean-audit.json", "{}")   # the Lean fast path's


VERDICTS = ["check/circuit-check/v1/d/e1.json", "check/lean-agreement/k3.json", "check/lean-audit/k2.json", "tests/k1.json"]


def test_the_roots_are_where_check_and_suites_keep_their_caches(tmp_path, monkeypatch):
    r = _machine(tmp_path, monkeypatch, "a")
    assert r["check"] == _load("check").cache_dir() and r["tests"] == _load("suites").CACHE
    monkeypatch.delenv("CHECK_CACHE")
    monkeypatch.delenv("VERITY_TEST_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert V.roots() == {"check": _load("check").cache_dir(), "tests": _load("suites").CACHE}


def test_a_pack_carries_the_verdicts_and_nothing_else_and_never_replaces_an_entry(tmp_path, monkeypatch):
    _fill(_machine(tmp_path, monkeypatch, "a"))
    assert sorted(V.entries()) == VERDICTS
    pack = tmp_path / "pack.tar.gz"
    assert V.export(pack) == 4
    again = tmp_path / "again.tar.gz"
    V.export(again)
    assert pack.read_bytes() == again.read_bytes()                  # the same entries make the same pack
    b = _machine(tmp_path, monkeypatch, "b")
    _write(b["tests"] / "k1.json", '{"suite": "b, its own"}')
    assert V.import_(pack) == (3, 1)
    assert sorted(V.entries()) == VERDICTS and (b["tests"] / "k1.json").read_text() == '{"suite": "b, its own"}'
    assert not (b["check"] / "lean-deps").exists() and not (b["tests"] / "durations.json").exists()


def test_the_lean_build_trees_are_never_walked(tmp_path, monkeypatch):
    """A recorded check snapshots and exports the caches around its steps, and a Lean build tree holds a package's whole
    dependency copy (some 170k files); its `.json` files (records, `lake-manifest.json`, `links.json`) were packed as verdicts
    (red team, #966)."""
    r = _machine(tmp_path, monkeypatch, "a")
    _fill(r)
    _write(r["check"] / "lean-audit" / ".lake" / "x.json", "{}")
    walked = []
    real = os.walk

    def walk(top, *a, **kw):
        for d, dirs, names in real(top, *a, **kw):
            walked.append(Path(d).relative_to(top).as_posix())
            yield d, dirs, names

    monkeypatch.setattr(V.os, "walk", walk)
    assert sorted(V.entries()) == VERDICTS and "circuit-check/v1/d" in walked
    assert not [d for d in walked if V._skipped("check", d.split("/")[0]) or ".lake" in d.split("/")]


def test_since_a_snapshot_a_pack_holds_only_what_the_run_stored(tmp_path, monkeypatch):
    r = _machine(tmp_path, monkeypatch, "a")
    _fill(r)
    before = V.snapshot()
    _write(r["tests"] / "k4.json", '{"suite": "new"}')
    _write(r["check"] / "lean-audit" / "k2.json", '{"package": "p", "again": true}')
    pack = tmp_path / "run.tar.gz"
    assert V.export(pack, before) == 2
    with tarfile.open(pack) as tar:
        assert sorted(tar.getnames()) == ["check/lean-audit/k2.json", "tests/k4.json"]


def test_an_entry_that_goes_away_while_it_is_read_is_left_out(tmp_path, monkeypatch):
    """Another slot's suites prune a module's older passes, or its audit removes its scratch tree, after this one lists them."""
    r = _machine(tmp_path, monkeypatch, "a")
    _fill(r)
    listed = V.entries()
    (r["tests"] / "k1.json").unlink()
    monkeypatch.setattr(V, "entries", lambda: listed)
    assert sorted(V.snapshot()) == VERDICTS[:-1]
    pack = tmp_path / "pack.tar.gz"
    assert V.export(pack) == 3
    with tarfile.open(pack) as tar:
        assert sorted(tar.getnames()) == VERDICTS[:-1]


@pytest.mark.parametrize("member", ["../escape.json", "/abs.json", "other/x.json", "check/lean-deps/m.json", "tests/logs/a.json",
                                    "check/lean-changed/links.json", "check/lean-records/build-0/p/lean-audit.json",
                                    "tests/durations.json", "check/.hidden.json", "tests/k.txt", "link",
                                    "check/lean-audit-scratch-x1/p/lake-manifest.json", "check/lean-changed/p/lean-audit.json"])
def test_a_pack_holding_anything_but_verdict_entries_is_refused_whole(tmp_path, monkeypatch, member):
    r = _machine(tmp_path, monkeypatch, "b")
    pack = tmp_path / "bad.tar.gz"
    with tarfile.open(pack, "w:gz") as tar:
        good = tarfile.TarInfo("tests/good.json")
        good.size = 2
        tar.addfile(good, io.BytesIO(b"{}"))
        bad = tarfile.TarInfo(member)
        if member == "link":
            bad.type, bad.linkname = tarfile.SYMTYPE, "tests/good.json"
        else:
            bad.size = 2
        tar.addfile(bad, io.BytesIO(b"{}") if member != "link" else None)
    with pytest.raises(V.BadPack):
        V.import_(pack)
    assert not (r["tests"] / "good.json").exists()                 # nothing is written from a refused pack
    assert V.main(["import", str(pack)]) == 1


def test_check_imports_a_pack_from_the_runs_inputs_and_names_it_in_the_result(tmp_path, monkeypatch):
    check = _load("check")
    _fill(_machine(tmp_path, monkeypatch, "a"))
    V.export(tmp_path / "verdicts-union.tar.gz")
    _machine(tmp_path, monkeypatch, "pod")
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (tmp_path / "verdicts-union.tar.gz").rename(run / "inputs" / "verdicts-union.tar.gz")
    got = check.verdicts_in(Path("/elsewhere/verdicts-union.tar.gz"), run)      # the launcher's path; the run has its copy
    assert got["added"] == 4 and got["pack"] == "verdicts-union.tar.gz" and len(got["sha256"]) == 64
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20260929-000000-abcd")
    check.write_result(run, True, [], {"in": got})
    assert json.loads((run / "result.json").read_text())["verdicts"]["in"]["added"] == 4


def test_the_agreement_key_leaves_out_what_names_the_machine(monkeypatch):
    check = _load("check")
    base = {"PATH": f"{check.ROOT}/.venv/bin:/usr/bin", "LEAN_NUM_THREADS": "8"}
    one = check._env_key({**base, "HOSTNAME": "a1b2", "RUNPOD_POD_ID": "abc", "RUNPOD_PUBLIC_IP": "1.2.3.4", "PUBLIC_KEY": "ssh-ed25519 A"})
    other = check._env_key({**base, "HOSTNAME": "c3d4", "RUNPOD_POD_ID": "xyz", "RUNPOD_PUBLIC_IP": "5.6.7.8", "PUBLIC_KEY": "ssh-ed25519 B"})
    assert one == other == {"LEAN_NUM_THREADS": "8", "PATH": "<tree>/.venv/bin:/usr/bin"}
