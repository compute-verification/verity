"""`tools/verity/check/verdicts.py`: the passes one machine stored, carried to another as a pack; and `check.py`'s side of it."""
from __future__ import annotations

import gzip
import hashlib
import http.client
import importlib.util
import io
import json
import os
import sys
import tarfile
import types
import urllib.error
import zlib
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


def test_an_import_of_one_part_takes_only_its_entries(tmp_path, monkeypatch):
    """`check.py --no-cache --circuit-cache` (drive's after-landing run) imports circuit-check's entries of a pack alone, so no
    suite, Lean or step pass it carries is here for a step to reuse."""
    _fill(_machine(tmp_path, monkeypatch, "a"))
    V.export(pack := tmp_path / "pack.tar.gz")
    _machine(tmp_path, monkeypatch, "b")
    assert V.import_(pack, "check/circuit-check/") == (1, 0)
    assert sorted(V.entries()) == ["check/circuit-check/v1/d/e1.json"]


def test_the_lean_build_trees_are_never_walked(tmp_path, monkeypatch):
    """A recorded check snapshots and exports the caches around its steps, and a Lean build tree holds a package's whole
    dependency copy (some 170k files); its `.json` files (records, `lake-manifest.json`, `links.json`) are never packed as
    verdicts."""
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


def test_snapshot_and_export_read_only_what_the_index_does_not_know(tmp_path, monkeypatch):
    """The caches run to hundreds of GB; re-reading every entry at both ends of a check held a finished run for an hour of
    disk reads once another job had evicted the page cache.  The per-root index memoizes each
    entry's sha256 by (size, mtime_ns): a second snapshot reads nothing, and a closing export reads only the run's own writes."""
    r = _machine(tmp_path, monkeypatch, "a")
    _fill(r)
    reads = []
    real = V._read
    monkeypatch.setattr(V, "_read", lambda p: reads.append(p) or real(p))
    before = V.snapshot()
    assert len(reads) == len(VERDICTS) and (r["check"] / V.INDEX).is_file() and (r["tests"] / V.INDEX).is_file()
    reads.clear()
    assert V.snapshot() == before and reads == []                     # every entry answered by the index
    _write(r["tests"] / "k4.json", '{"suite": "new"}')
    pack = tmp_path / "run.tar.gz"
    assert V.export(pack, before) == 1 and {p.name for p in reads} == {"k4.json"}   # hashed, then streamed into the pack
    with tarfile.open(pack) as tar:
        assert tar.getnames() == ["tests/k4.json"]


def test_a_garbled_or_stale_index_only_costs_the_reread(tmp_path, monkeypatch):
    r = _machine(tmp_path, monkeypatch, "a")
    _fill(r)
    before = V.snapshot()
    (r["tests"] / V.INDEX).write_text("not json")
    (r["check"] / V.INDEX).write_text(json.dumps({"lean-audit/k2.json": [1, 2, "0" * 64]}))   # wrong size and mtime: stale
    assert V.snapshot() == before                                     # recomputed from the files, and the index rewritten
    assert json.loads((r["tests"] / V.INDEX).read_text())
    assert V.export(tmp_path / "none.tar.gz", before) == 0


def test_the_index_is_never_listed_or_packed(tmp_path, monkeypatch):
    r = _machine(tmp_path, monkeypatch, "a")
    _fill(r)
    V.snapshot()
    assert (r["check"] / V.INDEX).is_file() and sorted(V.entries()) == VERDICTS
    assert V.export(pack := tmp_path / "pack.tar.gz") == 4
    with tarfile.open(pack) as tar:
        assert sorted(tar.getnames()) == VERDICTS


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


def test_a_pack_of_the_entries_named_is_xz_and_import_reads_it(tmp_path, monkeypatch):
    """`check.Reused`'s pack: the entries a plan found, each once, in path order; a name that is no verdict entry here (absent,
    hidden, outside the caches, a part never packed) is left out, never read."""
    r = _machine(tmp_path, monkeypatch, "a")
    _fill(r)
    names = ["tests/k1.json", "check/lean-audit/k2.json", "tests/k1.json", "tests/absent.json", "tests/../k1.json",
             "check/circuit-check/v1/d/.e2.tmp", "check/lean-deps/m/manifest.json", "other/k1.json", "tests/durations.json"]
    pack = tmp_path / "reused.tar.xz"
    assert V.export_paths(pack, names, xz=True) == 2
    assert pack.read_bytes()[:6] == b"\xfd7zXZ\x00" and V.refusal(pack) is None
    with tarfile.open(pack) as tar:
        assert tar.getnames() == ["check/lean-audit/k2.json", "tests/k1.json"]
    _machine(tmp_path, monkeypatch, "b")
    assert V.import_(pack) == (2, 0)


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
    `import` never carries the builds."""
    r = _machine(tmp_path, monkeypatch, "b")
    pack = _pack_with(tmp_path, member)
    assert V.refusal(pack) is None
    assert V.import_(pack) == (1, 0) and (r["tests"] / "good.json").exists()
    assert not (r[member.split("/")[0]] / member.split("/", 1)[1]).exists()


@pytest.mark.parametrize("member", ["../escape.json", "/abs.json", "other/x.json", "check/.hidden.json", "tests/k.txt", "link"])
def test_a_pack_holding_anything_but_verdict_entries_is_refused_whole(tmp_path, monkeypatch, member):
    r = _machine(tmp_path, monkeypatch, "b")
    pack = _pack_with(tmp_path, member)
    with pytest.raises(V.BadPack):
        V.import_(pack)
    assert not (r["tests"] / "good.json").exists()                 # nothing is written from a refused pack
    assert V.main(["import", str(pack)]) == 1


def _unreadable(tmp_path: Path) -> dict[str, Path]:
    """{the error reading it raises: a pack}: a file that's no pack, a pack cut short, and a gzipped and an xz pack whose stream
    turns corrupt past their first member, where `tarfile` leaves zlib's and lzma's own errors as they are."""
    items = [(f"tests/k{i}.json", b"{}" * (1 << 16)) for i in range(8)]
    V._write(gz := tmp_path / "whole.tar.gz", items)
    V._write(xz := tmp_path / "whole.tar.xz", items, xz=True)
    tar, z = gzip.decompress(gz.read_bytes()), zlib.compressobj(wbits=-15)
    # deflate data that holds 7/8 of the tar, then a block of the reserved type, which zlib refuses wherever it reads it
    cut = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff" + z.compress(tar[: len(tar) * 7 // 8]) + z.flush(zlib.Z_FULL_FLUSH) + b"\x07"
    flipped = bytearray(xz.read_bytes())
    flipped[len(flipped) * 3 // 4] ^= 0xFF
    data = {"ReadError": b"no pack at all", "EOFError": gz.read_bytes()[: gz.stat().st_size // 2], "error": cut,
            "LZMAError": bytes(flipped)}
    for err, b in data.items():
        (tmp_path / f"{err}.pack").write_bytes(b)
    return {err: tmp_path / f"{err}.pack" for err in data}


def test_a_pack_cut_short_or_corrupt_is_refused_and_nothing_written(tmp_path, monkeypatch):
    """A recorded run imports other machines' packs at its start: one that isn't a tar, or whose compressed stream ends early
    or turns corrupt, is refused like one holding anything but verdict entries, never a crash of the run that imports it."""
    r = _machine(tmp_path, monkeypatch, "b")
    for err, pack in _unreadable(tmp_path).items():
        assert V.refusal(pack).startswith(f"{err}: "), err
        with pytest.raises(V.UNREADABLE) as got:
            V.import_(pack)
        assert type(got.value).__name__ == err and not V.entries() and V.main(["import", str(pack)]) == 1
    assert V.refusal(tmp_path / "absent.tar.gz").startswith("FileNotFoundError: ") and not r["tests"].exists()


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
    check.write_result(run, True, [], {"in": [got]})
    assert json.loads((run / "result.json").read_text())["verdicts"]["in"][0]["added"] == 4


def test_a_pack_under_a_directory_the_machine_cant_read_is_found_in_the_runs_inputs(tmp_path, monkeypatch):
    """The launcher names the pack by its own path, and on another machine that path can sit under a directory the run may not
    read (the control pod's /root): stat raises PermissionError there, not FileNotFoundError, and the run's copy is used."""
    check = _load("check")
    _fill(_machine(tmp_path, monkeypatch, "a"))
    V.export(tmp_path / "verdicts-gathered.tar.gz")
    _machine(tmp_path, monkeypatch, "node")
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (tmp_path / "verdicts-gathered.tar.gz").rename(run / "inputs" / "verdicts-gathered.tar.gz")
    launcher = Path("/root/fast/vqueue-tmp/check-lean-deps-x/verdicts-gathered.tar.gz")
    exists = Path.exists

    def denied(self, *a, **kw):
        if self == launcher:
            raise PermissionError(13, "Permission denied", str(self))
        return exists(self, *a, **kw)

    monkeypatch.setattr(Path, "exists", denied)
    got = check.verdicts_in(launcher, run)
    assert got["added"] == 4 and got["pack"] == "verdicts-gathered.tar.gz"


def test_the_agreement_key_leaves_out_what_names_the_machine(monkeypatch):
    check = _load("check")
    base = {"PATH": f"{check.ROOT}/.venv/bin:/usr/bin", "VERITY_SLOW": "1"}
    one = check._env_key({**base, "HOSTNAME": "a1b2", "RUNPOD_POD_ID": "abc", "RUNPOD_PUBLIC_IP": "1.2.3.4", "PUBLIC_KEY": "ssh-ed25519 A"})
    other = check._env_key({**base, "HOSTNAME": "c3d4", "RUNPOD_POD_ID": "xyz", "RUNPOD_PUBLIC_IP": "5.6.7.8", "PUBLIC_KEY": "ssh-ed25519 B"})
    assert one == other == {"VERITY_SLOW": "1"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _packs(tmp_path: Path, monkeypatch, held: dict[str, dict[str, str]]) -> dict[str, Path]:
    """{name: pack} of the entries each machine ``held`` ({pack path: text}) stored, exported from its own caches."""
    packs = {}
    for name, entries in held.items():
        r = _machine(tmp_path, monkeypatch, name)
        for k, text in entries.items():
            _write(r[k.split("/")[0]] / k.split("/", 1)[1], text)
        V.export(packs.setdefault(name, tmp_path / f"{name}.tar.gz"))
    return packs


def _refused(tmp_path: Path) -> Path:
    """A pack `import` refuses whole: beside a suite's entry, a member under another prefix."""
    with tarfile.open(bad := tmp_path / "bad.tar.gz", "w:gz") as tar:
        for name in ("tests/k7.json", "other/lean/lake-manifest.json"):
            info = tarfile.TarInfo(name)
            info.size = 2
            tar.addfile(info, io.BytesIO(b"{}"))
    return bad


class _Store:
    """The store as `verdict_packs` reads it: recorded checks, each an hour after the next one listed and running ``walls`` seconds
    (none: one), with their run files as it lists them; with ``signs``, a remote holding every pack that signs a `file:` URL of
    each.  Each refresh, fetch and signature is kept."""

    def __init__(self, tmp: Path, runs: dict[str, Path | None], states: dict[str, str] | None = None,
                 walls: dict[str, float] | None = None, signs: bool = False):
        self.tmp, self.runs, self.states, self.walls = tmp, runs, states or {}, walls or {}
        self.index, self.refreshed, self.fetched, self.signed = self, [], [], []
        self.remote = types.SimpleNamespace(presign=self.presign) if signs else None

    def refresh(self, attempts_only: bool = False, since: str | None = None) -> None:
        self.refreshed.append((attempts_only, since))

    def sql(self, query: str, params: tuple = ()) -> list[tuple]:
        import sqlite3

        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE attempts (id TEXT, tool TEXT, state TEXT, start_utc TEXT, wall_s REAL, manifest_json TEXT)")
        db.executemany("INSERT INTO attempts VALUES (?, 'check', ?, ?, ?, '{}')",
                       [(rid, self.states.get(rid, "done"), f"2026-10-12T{23 - i:02d}:00:00Z", self.walls.get(rid, 1.0))
                        for i, rid in enumerate(self.runs)])
        return db.execute(query, params).fetchall()

    def get_attempt(self, rid: str) -> dict:
        return {"outputs": {"run_files": f"art:{rid}"}}

    def listing(self, art: str) -> list[dict]:
        if (pack := self.runs[art.removeprefix("art:")]) is None:
            return [{"path": "result.json", "bytes": 2, "sha256": "0" * 64, "remote": True}]
        return [{"path": "verdicts.tar.gz", "bytes": pack.stat().st_size, "sha256": _sha(pack), "remote": True}]

    def fetch(self, art: str, to: Path, *, paths: list[str]) -> Path:
        assert paths == ["verdicts.tar.gz"]
        self.fetched.append(art)
        to.mkdir(parents=True)
        (to / "verdicts.tar.gz").write_bytes(self.runs[art.removeprefix("art:")].read_bytes())
        return to

    def presign(self, key: str, expires_s: int) -> str:
        from research.store.local import object_key

        self.signed.append((key, expires_s))
        return next(p for p in self.runs.values() if p is not None and object_key(_sha(p)) == key).as_uri()


def test_a_recorded_check_sends_the_newest_recorded_checks_packs_as_urls_and_fetches_none(tmp_path, monkeypatch, capsys):
    """A launch fetched and combined the newest recorded checks' packs on the control pod's network volume, some ten minutes
    of each shard's launch.  It now lists them, newest first, each with a URL of its blob signed for `VERDICT_URL_TTL_S`, in a
    file only its owner reads, and fetches none; a run that stored none is passed over.  The URLs live as long as the Lean
    bundles' (`lean_audit.URL_TTL_S`), which the same run fetches after the same wait for its slot.  The newest checks are listed
    from the store's remote, as only its last `common.RECENT_S` of attempts: listing all of them took half a minute a launch."""
    check = _load("check")
    packs = _packs(tmp_path, monkeypatch, {"new": {"tests/k1.json": "new"}, "old": {"tests/k1.json": "old", "tests/k3.json": "3"}})
    store = _Store(tmp_path, {"r-new": packs["new"], "r-none": None, "r-old": packs["old"]}, signs=True)
    nows: list[float] = []
    monkeypatch.setattr(check.COMMON, "recent", lambda now: nows.append(now) or "r20261108-120000")
    path = check.verdict_url_file(tmp_path, store)
    assert path == tmp_path / check.VERDICT_URLS and path.stat().st_mode & 0o777 == 0o600
    assert [(p["run"], p["sha256"], p["bytes"], p["url"]) for p in json.loads(path.read_text())] == \
        [(f"r-{k}", _sha(packs[k]), packs[k].stat().st_size, packs[k].as_uri()) for k in ("new", "old")]
    assert [ttl for _, ttl in store.signed] == [check._lean_audit().URL_TTL_S] * 2 and store.fetched == []
    assert store.refreshed == [(True, "r20261108-120000")], "the newest checks are listed from the store, not this machine's stale index"
    assert len(nows) == 1 and nows[0] > 1.7e9, "the cutoff is counted back from the time of day, not from a monotonic clock"
    assert "sending presigned URLs for the verdict packs of 2 recorded checks (0 MiB): r-new, r-old" in capsys.readouterr().out
    assert check.verdict_url_file(tmp_path / "none", _Store(tmp_path, {"r-none": None}, signs=True)) is None


def test_a_launch_lists_the_attempts_started_since_before_the_longest_recorded_check_could(monkeypatch):
    """`recent` is the run id prefix `RECENT_S` before a time, in UTC as every run id is on any machine's zone; it reaches back
    past the longest a recorded check runs, or a long one that just ended would go unlisted where the index lacks it."""
    import time

    check = _load("check")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        assert check.COMMON.recent(check.COMMON.RECENT_S + 40 * 86400 + 3661.5) == "r19700210-010101"
    finally:
        monkeypatch.undo()
        time.tzset()
    assert check.COMMON.RECENT_S >= 2 * check.STEP_BUDGET_S


def test_a_run_imports_each_pack_sent_newest_first_and_names_each_in_its_result(tmp_path, monkeypatch, capsys):
    """`import` never replaces an entry, so where two packs hold one the newer's is kept, and one this node holds already is its
    own; with ``only`` (drive's after-landing run) just circuit-check's entries are taken."""
    check = _load("check")
    packs = _packs(tmp_path, monkeypatch, {"new": {"tests/k1.json": "new", "check/circuit-check/v1/d/e.json": "{}"},
                                           "old": {"tests/k1.json": "old", "tests/k3.json": "3"}})
    listed = json.loads(check.verdict_url_file(tmp_path, _Store(tmp_path, {"r-new": packs["new"], "r-old": packs["old"]},
                                                                signs=True)).read_text())
    capsys.readouterr()
    r = _machine(tmp_path, monkeypatch, "node")
    _write(r["tests"] / "k3.json", "the node's own")
    assert check.verdicts_fetched(listed) == [{"run": "r-new", "sha256": _sha(packs["new"]), "added": 2, "already": 0},
                                              {"run": "r-old", "sha256": _sha(packs["old"]), "added": 0, "already": 2}]
    assert (r["tests"] / "k1.json").read_text() == "new" and (r["tests"] / "k3.json").read_text() == "the node's own"
    out = capsys.readouterr().out
    assert "check: verdicts: 2 imported, 2 already here, from 2 of 2 recorded checks' packs in " in out and "file:" not in out
    _machine(tmp_path, monkeypatch, "after-landing")
    got = check.verdicts_fetched(listed, check.CIRCUIT_ENTRIES)
    assert [(g["added"], g["only"]) for g in got] == [(1, check.CIRCUIT_ENTRIES), (0, check.CIRCUIT_ENTRIES)]
    assert sorted(V.entries()) == ["check/circuit-check/v1/d/e.json"]
    assert "1 imported, 0 already here, from 2 of 2 recorded checks' packs (only check/circuit-check/) in " in capsys.readouterr().out


def test_a_pack_that_cant_be_fetched_isnt_the_one_listed_or_that_import_refuses_is_passed_over_by_name(tmp_path, monkeypatch,
                                                                                                          capsys):
    """A pack holding anything but verdict entries (a Lean audit's scratch is left out of a pack, and a link or another prefix
    still refuses it), one cut short or corrupt at its source (its sha256 is what the store lists), one whose bytes aren't those
    the store lists for it, one no fetch reaches and one whose URL doesn't parse are each skipped and the next one taken; none of
    their entries is reused, and no URL is printed, since each grants a read until it expires.  A fetch stops past the size
    listed, so a store that sends more costs a run's start at most a read more than it lists."""
    check = _load("check")
    bad = _refused(tmp_path)
    packs = _packs(tmp_path, monkeypatch, {"old": {"tests/k3.json": "3"}})
    good = {"run": "r-old", "sha256": _sha(packs["old"]), "bytes": packs["old"].stat().st_size, "url": packs["old"].as_uri()}
    unreadable = _unreadable(tmp_path)
    (huge := tmp_path / "huge.tar.gz").write_bytes(bytes(3 << 20))
    listed = [{"run": "r-bad", "sha256": _sha(bad), "bytes": bad.stat().st_size, "url": bad.as_uri()},
              *({"run": f"r-{err}", "sha256": _sha(p), "bytes": p.stat().st_size, "url": p.as_uri()} for err, p in unreadable.items()),
              {**good, "run": "r-gone", "url": (tmp_path / "gone.tar.gz").as_uri()}, {**good, "run": "r-nourl", "url": "no-scheme"},
              {**good, "run": "r-huge", "bytes": 1, "url": huge.as_uri()},
              {**good, "run": "r-other", "sha256": "0" * 64}, {**good, "run": "r-long", "bytes": good["bytes"] - 1}, good]
    _machine(tmp_path, monkeypatch, "node")
    got = check.verdicts_fetched(listed)
    out = capsys.readouterr().out
    assert "skipping r-bad's pack: import refuses it (BadPack: member 'other/lean/lake-manifest.json'" in out
    for err in unreadable:
        assert f"skipping r-{err}'s pack: import refuses it ({err}: " in out
    assert "skipping r-gone's pack: it can't be fetched (URLError); its verdicts are recomputed" in out
    assert "skipping r-nourl's pack: it can't be fetched (ValueError); its verdicts are recomputed" in out
    assert f"skipping r-huge's pack: the {1 << 20} bytes fetched aren't the 1 the store lists under its sha256" in out
    n = good["bytes"]
    assert f"skipping r-other's pack: the {n} bytes fetched aren't the {n} the store lists under its sha256" in out
    assert f"skipping r-long's pack: the {n} bytes fetched aren't the {n - 1} the store lists" in out
    assert [g.get("added") for g in got] == [None] * (len(listed) - 1) + [1] and sorted(V.entries()) == ["tests/k3.json"]
    assert all(f"skipping {g['run']}'s pack: {g['skipped']}; its verdicts are recomputed" in out for g in got[:-1])
    assert f"from 1 of {len(listed)} recorded checks' packs" in out and "file:" not in out


def test_a_fetch_the_store_answers_with_an_error_or_cuts_short_is_passed_over_without_its_url(tmp_path, monkeypatch, capsys):
    """urllib's errors carry the URL they were raised for, which grants a read until it expires: only the error's type and an
    HTTP status are printed or recorded.  Each fetch gives up on a store that stops answering, rather than holding the run's
    start."""
    check = _load("check")
    url, timeouts = "https://store.invalid/objects/sha256/ab?X-Amz-Signature=signed", []

    class Cut:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n: int) -> bytes:
            raise http.client.IncompleteRead(b"x", 9)

    def urlopen(u: str, timeout: float):
        timeouts.append(timeout)
        if u.endswith("&403"):
            raise urllib.error.HTTPError(u, 403, "Forbidden", {}, None)
        return Cut()

    monkeypatch.setattr(check.urllib.request, "urlopen", urlopen)
    _machine(tmp_path, monkeypatch, "node")
    got = check.verdicts_fetched([{"run": "r-403", "sha256": "0" * 64, "bytes": 1, "url": url + "&403"},
                                  {"run": "r-cut", "sha256": "0" * 64, "bytes": 1, "url": url}])
    out = capsys.readouterr().out
    assert "skipping r-403's pack: it can't be fetched (HTTPError 403); its verdicts are recomputed" in out
    assert "skipping r-cut's pack: it can't be fetched (IncompleteRead); its verdicts are recomputed" in out
    assert "signed" not in out and "signed" not in json.dumps(got) and not V.entries()
    assert len(timeouts) == 2 and all(isinstance(t, (int, float)) and 0 < t <= 600 for t in timeouts)


def test_a_pack_that_would_take_a_checks_packs_past_their_byte_budget_is_passed_over_unfetched(tmp_path, monkeypatch, capsys):
    """A recorded check can publish a pack far larger than the rest: circuit-check's per-target stores, near 1 GiB, which no
    gated check reads.  Fetched by the launch, it staged many times its size on the launcher's disk, so no run started within
    the queue's wait; fetched by the run, it would cost each run that before its first step.  A pack whose listed size would
    take the packs past `GATHER_BYTES` is passed over by name, never signed or fetched; older checks' smaller packs are taken."""
    check = _load("check")
    packs = _packs(tmp_path, monkeypatch, {k: {f"tests/{k}.json": k} for k in ("new", "big", "old", "older")})

    class Sized(_Store):
        def listing(self, art: str) -> list[dict]:
            return [{**f, "bytes": 10**9} for f in super().listing(art)] if art == "art:r-big" else super().listing(art)

    monkeypatch.setattr(check, "GATHER_BYTES", packs["new"].stat().st_size + packs["old"].stat().st_size)
    store = Sized(tmp_path, {f"r-{k}": p for k, p in packs.items()}, signs=True)
    sent = json.loads(check.verdict_url_file(tmp_path, store).read_text())
    out = capsys.readouterr().out
    assert "skipping r-big's pack: its 954 MiB would take this check's packs past" in out
    assert "skipping r-older's pack" in out, "r-new's and r-old's sizes fill the budget exactly"
    assert [p["run"] for p in sent] == ["r-new", "r-old"] and len(store.signed) == 2 and store.fetched == []


def test_the_checks_taken_are_the_ones_that_ended_last_so_a_long_ones_passes_reach_the_next_check(tmp_path, monkeypatch):
    """A pre-warm of a cold circuit-check can run 4.5 h while drive starts more than twelve warm checks; ordered by start, its
    pack would be out of the window when the train it warmed ran."""
    check = _load("check")
    packs = _packs(tmp_path, monkeypatch, {k: {f"tests/{k}.json": k} for k in ("short", "long")})
    store = _Store(tmp_path, {"r-short": packs["short"], "r-long": packs["long"]}, walls={"r-long": 3 * 3600.0})
    assert [p["run"] for p in check.verdict_packs(store, limit=1)] == ["r-long"]


def test_a_plan_reads_the_packs_a_check_imports_and_passes_over_one_import_refuses(tmp_path, monkeypatch, capsys):
    """`plan.py` looks passes up in the packs a recorded check imports (`gathered_packs`: `verdict_packs`, fetched here), so
    what it says a check reuses is what one would; a pack import refuses is skipped by name, and so is one gone from the store
    by the time it's fetched."""
    check = _load("check")
    bad = _refused(tmp_path)
    packs = _packs(tmp_path, monkeypatch, {k: {f"tests/{k}.json": k} for k in ("old", "older")})
    store = _Store(tmp_path, {"r-bad": bad, "r-none": None, "r-old": packs["old"], "r-older": packs["older"]})
    got = check.gathered_packs(tmp_path / "out", store, limit=3)
    assert [rid for rid, _ in got] == ["r-old", "r-older"] and got[0][1].read_bytes() == packs["old"].read_bytes()
    assert got[0][1] == tmp_path / "out" / "verdicts" / "r-old" / "verdicts.tar.gz"
    assert "skipping r-bad's pack, which import refuses (BadPack: member 'other/lean/lake-manifest.json'" in capsys.readouterr().out
    assert store.fetched == ["art:r-bad", "art:r-old", "art:r-older"] and store.refreshed == []
    assert [rid for rid, _ in check.gathered_packs(tmp_path / "one", store, limit=2)] == ["r-old"]
    assert check.attempt_pack(store, "r-none", tmp_path / "none") is None

    class Evicted(_Store):
        def listing(self, art: str) -> list[dict]:
            self.listed = getattr(self, "listed", 0) + 1
            return super().listing(art) if self.listed == 1 else []

    assert check.gathered_packs(tmp_path / "gone", Evicted(tmp_path, {"r-old": packs["old"]})) == []


def test_a_failed_or_cancelled_checks_pack_is_taken_and_a_running_ones_is_not(tmp_path, monkeypatch):
    """A pack holds only passes, each under a key of everything it read, so a check that failed or was cancelled (its passes
    kept on SIGTERM, recorded `failed`) answers for them like one that passed."""
    check = _load("check")
    packs = _packs(tmp_path, monkeypatch, {k: {f"tests/{k}.json": k} for k in ("failed", "cancelled", "running")})
    store = _Store(tmp_path, {f"r-{k}": p for k, p in packs.items()}, {f"r-{k}": k for k in packs})
    assert [p["run"] for p in check.verdict_packs(store)] == ["r-failed", "r-cancelled"]


def test_an_attempt_the_store_holds_no_outputs_of_is_passed_over_and_the_rest_listed(tmp_path, monkeypatch):
    """The index lists an attempt before its outputs reach the store (a run that ended without publishing, or one whose record
    another machine hasn't synced): it has no pack to send, and the newer ones' absence never costs the older ones'."""
    check = _load("check")
    packs = _packs(tmp_path, monkeypatch, {"old": {"tests/old.json": "o"}})

    class Bare(_Store):
        def get_attempt(self, rid: str) -> dict | None:
            return {"r-unknown": None, "r-bare": {}}.get(rid) if rid != "r-old" else super().get_attempt(rid)

    store = Bare(tmp_path, {"r-unknown": None, "r-bare": None, "r-old": packs["old"]})
    assert [p["run"] for p in check.verdict_packs(store)] == ["r-old"]


def test_a_sigkilled_checks_pack_is_taken_from_its_run_record(tmp_path, monkeypatch):
    """A cancel's SIGKILL (rc 137) leaves no result.json, so the attempt has no run files listed from it, only the run record
    custody uploaded, which holds the pack `record` declares."""
    check = _load("check")
    packs = _packs(tmp_path, monkeypatch, {"killed": {"tests/killed.json": "k"}})

    class Killed(_Store):
        def get_attempt(self, rid: str) -> dict:
            return {"outputs": {"run_record": f"art:{rid}"}}

    store = Killed(tmp_path, {"r-killed": packs["killed"]}, {"r-killed": "cancelled"})
    assert [p["run"] for p in check.verdict_packs(store)] == ["r-killed"]


def test_a_store_that_cant_be_read_or_signs_nothing_sends_no_urls_and_says_why(tmp_path, monkeypatch, capsys):
    """The run then reuses its node's own passes and a pack given.  A pack the store's remote doesn't hold is never sent, since
    no URL fetches it, though a plan read here still takes it."""
    check = _load("check")

    class Broken(_Store):
        def sql(self, query, params=()):
            raise OSError("injected")

    assert check.verdict_url_file(tmp_path, Broken(tmp_path, {}, signs=True)) is None
    assert "no verdict packs listed from the store (OSError: injected)" in capsys.readouterr().out
    packs = _packs(tmp_path, monkeypatch, {"here": {"tests/k.json": "k"}})
    assert check.verdict_url_file(tmp_path, _Store(tmp_path, {"r-here": packs["here"]})) is None
    out = capsys.readouterr().out
    assert "no store remote here signs URLs, so the run imports no recorded check's pack" in out and "listed" not in out

    class Local(_Store):
        def listing(self, art: str) -> list[dict]:
            return [{**f, "remote": False} for f in super().listing(art)]

    store = Local(tmp_path, {"r-here": packs["here"]}, signs=True)
    assert check.verdict_url_file(tmp_path, store) is None and store.signed == []
    assert [rid for rid, _ in check.gathered_packs(tmp_path / "plan", store)] == ["r-here"]


@pytest.mark.parametrize("given", [True, False])
def test_a_recorded_run_imports_the_packs_sent_beside_one_given_before_its_snapshot(tmp_path, monkeypatch, given):
    """The run fetches the recorded checks' packs before its steps and before the snapshot its own pack is taken against, so its
    pack holds what its steps stored and none of what it imported; a pack given is imported beside them, never in their place.
    The URLs leave its inputs before any step."""
    check = _load("check")
    packs = _packs(tmp_path, monkeypatch, {"earned": {"tests/k1.json": "1", "check/lean-audit/k2.json": "{}"},
                                           "given": {"tests/k9.json": "9"}})
    run_dir = tmp_path / "run"
    (run_dir / "inputs").mkdir(parents=True)
    check.verdict_url_file(run_dir / "inputs", _Store(tmp_path, {"r-earned": packs["earned"]}, signs=True))
    _machine(tmp_path, monkeypatch, "node")
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261110-000000-abcd")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.delenv("VERITY_SKIP_STORE", raising=False)
    monkeypatch.delenv(check.VERDICT_URLS_ENV, raising=False)
    monkeypatch.setattr(check.SUITES, "foreign", lambda members, env: None)
    seen = {}

    def run(out, n, cache, keep_going, only, packs, reused, regenerate, circuit_cache=None, live=None, splits=None):
        seen.update(entries=sorted(V.entries()), urls=(run_dir / "inputs" / check.VERDICT_URLS).exists())
        return True, []

    monkeypatch.setattr(check, "run", run)
    assert check.main(["--keep-going", *(["--verdicts-in", str(packs["given"])] if given else [])]) == 0
    assert seen == {"entries": ["check/lean-audit/k2.json", "tests/k1.json", *(["tests/k9.json"] if given else [])], "urls": False}
    doc = json.loads((run_dir / "result.json").read_text())
    assert [(x.get("run") or x["pack"], x["added"]) for x in doc["verdicts"]["in"]] == \
        [*([("given.tar.gz", 1)] if given else []), ("r-earned", 2)]
    assert doc["verdicts"]["out"]["entries"] == 0


def test_two_imports_on_one_machine_never_share_a_temporary_file(tmp_path, monkeypatch):
    """Two checks on one machine import the same pack at once: each writes its entries through a file of its own, so neither
    renames the other's away (a shared name fails `verdicts_in` with FileNotFoundError)."""
    _fill(_machine(tmp_path, monkeypatch, "a"))
    pack = tmp_path / "pack.tar.gz"
    assert V.export(pack) == 4
    b = _machine(tmp_path, monkeypatch, "b")
    other = b["check"] / "lean-agreement" / f".k3.json.{os.getpid() + 1}.import"
    _write(other, "another import, half written")
    assert V.import_(pack) == (4, 0)
    assert other.read_text() == "another import, half written"
    assert (b["check"] / "lean-agreement" / "k3.json").read_text() == '{"key": "k3"}'
    assert not list((b["check"] / "lean-agreement").glob(f".k3.json.{os.getpid()}.import"))
