"""`tools/verity/check/verdicts.py`: the passes one machine stored, carried to another as a pack; and `check.py`'s side of it."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"test_verdicts_{name}", ROOT / "tools" / "verity" / "check" / f"{name}.py")
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
    _write(r["check"] / "lean-builds" / "k" / "ir" / "P" / "M.setup.json", "{}")  # the Lean audit's kept builds
    _write(r["check"] / "lean-artifacts" / "outputs" / "p" / "h.json", "{}")   # Lake's cache of built modules


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


SKIPPED = ["check/lean-deps/m.json", "tests/logs/a.json", "check/lean-changed/links.json",
           "check/lean-records/build-0/p/lean-audit.json", "tests/durations.json", "check/lean-audit-scratch-x1/p/lake-manifest.json",
           "check/lean-changed/p/lean-audit.json", "check/lean-builds/k/ir/P/M.setup.json",
           "check/lean-artifacts/outputs/p/h.json"]


def _pack_with(tmp_path: Path, member: str) -> Path:
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
    return pack


@pytest.mark.parametrize("member", SKIPPED)
def test_a_member_in_a_part_of_the_caches_never_packed_is_left_out_and_the_rest_taken(tmp_path, monkeypatch, member):
    """Packs written before `lean-builds/` was skipped hold its kept builds' `ir/*.setup.json`: their verdicts still count, and
    neither `import` nor `combine` carries the builds (12 Oct: a combine of them ran a 15 GB VM out of disk)."""
    r = _machine(tmp_path, monkeypatch, "b")
    pack = _pack_with(tmp_path, member)
    assert V.refusal(pack) is None
    assert V.import_(pack) == (1, 0) and (r["tests"] / "good.json").exists()
    assert not (r[member.split("/")[0]] / member.split("/", 1)[1]).exists()
    assert V.combine([pack], tmp_path / "all.tar.gz") == 1
    with tarfile.open(tmp_path / "all.tar.gz") as tar:
        assert tar.getnames() == ["tests/good.json"]


@pytest.mark.parametrize("member", ["../escape.json", "/abs.json", "other/x.json", "check/.hidden.json", "tests/k.txt", "link"])
def test_a_pack_holding_anything_but_verdict_entries_is_refused_whole(tmp_path, monkeypatch, member):
    r = _machine(tmp_path, monkeypatch, "b")
    pack = _pack_with(tmp_path, member)
    with pytest.raises(V.BadPack):
        V.import_(pack)
    assert not (r["tests"] / "good.json").exists()                 # nothing is written from a refused pack
    assert V.main(["import", str(pack)]) == 1
    assert V.main(["combine", str(tmp_path / "all.tar.gz"), str(pack)]) == 1 and not (tmp_path / "all.tar.gz").exists()


def test_combine_streams_its_entries_and_writes_the_pack_holding_them_whole_would(tmp_path):
    """`gather_verdicts` combines the newest recorded checks' packs on the lander's VM, and twelve of them held whole outgrew its
    15 GB: combining keeps one entry at a time in memory, writes the same bytes, and leaves no spill behind."""
    import tracemalloc

    big = 8 << 20
    a, b = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    V._write(a, [("check/circuit-check/x.json", os.urandom(big)), ("tests/k.json", b"{\"a\": 1}")])
    V._write(b, [("check/circuit-check/y.json", os.urandom(big)), ("tests/k.json", b"{\"b\": 2}")])
    out = tmp_path / "out" / "all.tar.gz"
    tracemalloc.start()
    assert V.combine([a, b], out) == 3
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < big // 2, peak
    with tarfile.open(a, "r:gz") as ta, tarfile.open(b, "r:gz") as tb:
        found = {**{m.name: tb.extractfile(m).read() for m in tb.getmembers()}, **{m.name: ta.extractfile(m).read() for m in ta.getmembers()}}
    whole = tmp_path / "whole.tar.gz"
    V._write(whole, sorted(found.items()))
    assert out.read_bytes() == whole.read_bytes() and found["tests/k.json"] == b"{\"a\": 1}"
    assert [p.name for p in out.parent.iterdir()] == ["all.tar.gz"]


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


class _Store:
    """The store as `gather_verdicts` reads it: recorded checks, newest first, each with its run files."""

    remote = None

    def __init__(self, tmp: Path, runs: dict[str, Path | None], states: dict[str, str] | None = None):
        self.tmp, self.runs, self.states = tmp, runs, states or {}
        self.index = self

    def sql(self, query: str, params: tuple = ()) -> list[tuple]:
        import sqlite3

        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE attempts (id TEXT, tool TEXT, state TEXT, start_utc TEXT)")
        db.executemany("INSERT INTO attempts VALUES (?, 'check', ?, ?)",
                       [(rid, self.states.get(rid, "done"), f"2026-10-12T{99 - i:02d}") for i, rid in enumerate(self.runs)])
        return db.execute(query, params).fetchall()

    def get_attempt(self, rid: str) -> dict:
        return {"outputs": {"run_files": f"art:{rid}"}}

    def listing(self, art: str) -> list[dict]:
        return [{"path": "verdicts.tar.gz"}] if self.runs[art.removeprefix("art:")] else [{"path": "result.json"}]

    def fetch(self, art: str, to: Path, *, paths: list[str]) -> Path:
        assert paths == ["verdicts.tar.gz"]
        to.mkdir(parents=True)
        (to / "verdicts.tar.gz").write_bytes(self.runs[art.removeprefix("art:")].read_bytes())
        return to


def test_a_recorded_check_gathers_the_newest_recorded_checks_packs_into_one(tmp_path, monkeypatch, capsys):
    """`--record` sends one pack of the newest recorded checks' packs, so a pass any of them earned is reused on any machine.
    A run that stored none is passed over; where two hold one entry, the newer's is taken."""
    check = _load("check")
    packs = {}
    for name, entries in (("new", {"tests/k1.json": "new", "tests/k2.json": "2"}), ("old", {"tests/k1.json": "old", "tests/k3.json": "3"})):
        r = _machine(tmp_path, monkeypatch, name)
        for k, text in entries.items():
            _write(r["tests"] / k.removeprefix("tests/"), text)
        V.export(packs.setdefault(name, tmp_path / f"{name}.tar.gz"))
    store = _Store(tmp_path, {"r-new": packs["new"], "r-none": None, "r-old": packs["old"]})
    got = check.gather_verdicts(tmp_path / "out", store)
    assert "3 entries gathered from 2 recorded checks (r-new, r-old)" in capsys.readouterr().out
    _machine(tmp_path, monkeypatch, "pod")
    assert V.import_(got) == (3, 0)
    assert (V.roots()["tests"] / "k1.json").read_text() == "new"
    assert check.gather_verdicts(tmp_path / "none", _Store(tmp_path, {"r-none": None})) is None


def test_a_recorded_checks_pack_import_refuses_is_passed_over_by_name_and_a_refused_pack_given_is_an_error(tmp_path, monkeypatch,
                                                                                                          capsys):
    """A pack holding anything but verdict entries (9 Oct: a Lean audit's scratch in a recorded check's pack crashed every
    `--record`; such a part is now left out of a pack, and a link or another prefix still refuses it) is skipped and the next one
    taken; none of its entries is reused."""
    check = _load("check")
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        for name in ("tests/k7.json", "other/lean/lake-manifest.json"):
            data = b"{}"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    r = _machine(tmp_path, monkeypatch, "old")
    _write(r["tests"] / "k3.json", "3")
    V.export(old := tmp_path / "old.tar.gz")
    got = check.gather_verdicts(tmp_path / "out", _Store(tmp_path, {"r-bad": bad, "r-old": old}), limit=1)
    out = capsys.readouterr().out
    assert "skipping r-bad's pack, which import refuses (BadPack: member 'other/lean/lake-manifest.json'" in out
    assert "1 entries gathered from 1 recorded checks (r-old)" in out
    _machine(tmp_path, monkeypatch, "pod")
    assert V.import_(got) == (1, 0) and not (V.roots()["tests"] / "k7.json").exists()
    with pytest.raises(SystemExit, match="refused"):
        check.gather_verdicts(tmp_path / "x", _Store(tmp_path, {}), given=bad)


def test_a_failed_or_cancelled_checks_pack_is_gathered_and_a_running_ones_is_not(tmp_path, monkeypatch, capsys):
    """A pack holds only passes, each under a key of everything it read, so a check that failed or was cancelled (its passes
    kept on SIGTERM, recorded `failed`) answers for them like one that passed."""
    check = _load("check")
    packs = {}
    for name in ("failed", "cancelled", "running"):
        r = _machine(tmp_path, monkeypatch, name)
        _write(r["tests"] / f"{name}.json", name)
        V.export(packs.setdefault(name, tmp_path / f"{name}.tar.gz"))
    store = _Store(tmp_path, {f"r-{k}": p for k, p in packs.items()}, {f"r-{k}": k for k in packs})
    got = check.gather_verdicts(tmp_path / "out", store)
    assert "2 entries gathered from 2 recorded checks (r-failed, r-cancelled)" in capsys.readouterr().out
    _machine(tmp_path, monkeypatch, "pod")
    assert V.import_(got) == (2, 0)


def test_a_store_that_cant_be_read_gathers_nothing_and_says_why(tmp_path, capsys):
    check = _load("check")

    class Broken(_Store):
        def sql(self, query, params=()):
            raise OSError("injected")

    assert check.gather_verdicts(tmp_path, Broken(tmp_path, {})) is None
    assert "no verdicts gathered from the store (OSError: injected)" in capsys.readouterr().out


def test_a_pack_given_is_added_to_the_gathered_ones_and_survives_a_store_that_cant_be_read(tmp_path, monkeypatch, capsys):
    check = _load("check")
    packs = {}
    for name, entries in (("given", {"tests/k1.json": "given", "tests/k9.json": "9"}), ("new", {"tests/k1.json": "new", "tests/k2.json": "2"})):
        r = _machine(tmp_path, monkeypatch, name)
        for k, text in entries.items():
            _write(r["tests"] / k.removeprefix("tests/"), text)
        V.export(packs.setdefault(name, tmp_path / f"{name}.tar.gz"))
    got = check.gather_verdicts(tmp_path / "out", _Store(tmp_path, {"r-new": packs["new"]}), given=packs["given"])
    assert "3 entries gathered from 1 recorded checks and the pack given (given.tar.gz (given), r-new)" in capsys.readouterr().out
    _machine(tmp_path, monkeypatch, "pod")
    assert V.import_(got) == (3, 0)

    class Broken(_Store):
        def sql(self, query, params=()):
            raise OSError("injected")

    got = check.gather_verdicts(tmp_path / "broken", Broken(tmp_path, {}), given=packs["given"])
    assert "no verdicts gathered from the store (OSError: injected)" in capsys.readouterr().out
    _machine(tmp_path, monkeypatch, "pod2")
    assert V.import_(got) == (2, 0)
    with pytest.raises(SystemExit, match="no such pack"):
        check.gather_verdicts(tmp_path / "x", _Store(tmp_path, {}), given=tmp_path / "absent.tar.gz")
