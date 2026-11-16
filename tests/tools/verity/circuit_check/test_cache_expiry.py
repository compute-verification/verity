"""A cache's versions expire (`cache.expire`): a run marks its own version used and removes, by rename and then delete, each
other version no run has used for `cache.EXPIRE_S`, never its own; a removal never fails a run.  Every time here is injected."""
from __future__ import annotations

import fcntl
import os
import subprocess
from pathlib import Path

import pytest

from circuit_check import cache as CC

NOW = 2_000_000_000.0
HOUR = 3600.0


def _version(cache: Path, name: str, used: float | None) -> Path:
    """A version's directory holding a few entries, its mark last set at ``used`` (None: no mark)."""
    d = cache / name / "k"
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"{i}.json").write_text("{}")
    if used is not None:
        (cache / CC.USED).mkdir(exist_ok=True)
        (m := cache / CC.USED / name).touch()
        os.utime(m, (used, used))
    return cache / name


def _used(cache: Path, name: str) -> float:
    return os.stat(cache / CC.USED / name).st_mtime


def test_a_version_used_within_the_window_stays_an_older_one_goes_and_the_run_s_own_stays_whatever_its_age(tmp_path):
    own, cache = CC.version(), tmp_path / "cache"
    recent, old, unmarked = "a" * 128, "b" * 128, "c" * 128
    _version(cache, own, NOW - 1000 * HOUR)
    _version(cache, recent, NOW - 71 * HOUR)
    _version(cache, old, NOW - 73 * HOUR)
    _version(cache, unmarked, None)
    (cache / "not-a-version").mkdir()
    assert CC.expire(cache, now=NOW, delete=CC.sweep) == {"gone": [old], "left": []}
    assert sorted(p.name for p in cache.iterdir() if not p.name.startswith(".")) == sorted([own, recent, unmarked, "not-a-version"])
    assert not any(p.name.startswith(CC.EXPIRED) for p in cache.iterdir()), "the renamed directory is deleted"
    assert (_used(cache, own), _used(cache, recent), _used(cache, unmarked)) == (NOW, NOW - 71 * HOUR, NOW)
    assert not (cache / CC.USED / old).exists()


def test_a_version_first_seen_unmarked_goes_a_window_after_that_and_a_use_restarts_the_window(tmp_path):
    cache, seen, used = tmp_path / "cache", "c" * 128, "d" * 128
    _version(cache, seen, None)
    _version(cache, used, None)
    assert CC.expire(cache, now=NOW, delete=CC.sweep)["gone"] == []
    os.utime(cache / CC.USED / used, (NOW + 70 * HOUR, NOW + 70 * HOUR))     # a run of it, as `mark` leaves it
    assert CC.expire(cache, now=NOW + CC.EXPIRE_S - 1, delete=CC.sweep)["gone"] == []
    assert CC.expire(cache, now=NOW + CC.EXPIRE_S, delete=CC.sweep)["gone"] == [seen]
    assert (cache / used).is_dir() and not (cache / seen).exists()


def test_a_start_reads_the_cache_s_top_and_the_marks_never_what_a_version_holds(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    for c in "0123456789":
        _version(cache, c * 128, NOW - HOUR)
    CC.version()
    calls, real = [], {f: getattr(os, f) for f in ("listdir", "scandir", "stat", "lstat")}

    def seen(f):
        def call(p=".", *a, **k):
            if str(p).startswith(str(cache)):
                calls.append((f, Path(p)))
            return real[f](p, *a, **k)
        return call

    for f in real:
        monkeypatch.setattr(os, f, seen(f))
    assert CC.expire(cache, now=NOW, delete=lambda top: pytest.fail("nothing to delete")) == {"gone": [], "left": []}
    marks = {("stat", cache / CC.USED / (c * 128)) for c in "0123456789"}
    assert sorted(set(calls) - marks) == [("listdir", cache), ("stat", cache / CC.USED)] and len(calls) == 12


def test_two_runs_removing_one_version_at_once_both_pass_and_it_goes_once(tmp_path, monkeypatch):
    cache, old = tmp_path / "cache", "b" * 128
    _version(cache, old, NOW - 100 * HOUR)
    rename, other = os.rename, []

    def racing(src, dst):
        if not other and Path(src).name == old:
            other.append(None)
            other[0] = CC.expire(cache, now=NOW, delete=lambda top: None)        # the other run renames it first
        rename(src, dst)

    monkeypatch.setattr(os, "rename", racing)
    mine = CC.expire(cache, now=NOW, delete=lambda top: None)
    assert mine["gone"] == [] and [n for n, _ in mine["left"]] == [old] and "FileNotFoundError" in mine["left"][0][1]
    assert other == [{"gone": [old], "left": []}]
    away = [p for p in cache.iterdir() if p.name.startswith(CC.EXPIRED)]
    assert len(away) == 1 and not (cache / old).exists()
    with open(cache / CC.USED / ".sweep.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)                                          # another sweeper is at work
        assert CC.sweep(cache) == "another sweeper holds the lock"
        assert away[0].exists()
    assert CC.sweep(cache) == "swept 1"
    assert not away[0].exists()


def test_a_version_marked_between_reading_its_mark_and_renaming_it_is_given_back(tmp_path, monkeypatch):
    cache, old = tmp_path / "cache", "b" * 128
    _version(cache, old, NOW - 100 * HOUR)
    rename = os.rename

    def marked_meanwhile(src, dst):
        rename(src, dst)
        if Path(src).name == old:
            os.utime(cache / CC.USED / old, (NOW, NOW))                           # a run of it starts here

    monkeypatch.setattr(os, "rename", marked_meanwhile)
    assert CC.expire(cache, now=NOW, delete=lambda top: pytest.fail("nothing to delete")) == {
        "gone": [], "left": [[old, "marked while it was renamed away: given back"]]}
    assert (cache / old / "k" / "0.json").is_file()


def test_a_directory_left_renamed_away_is_deleted_by_a_later_run_in_a_process_of_its_own(tmp_path):
    cache = tmp_path / "cache"
    away = _version(cache, f"{CC.EXPIRED}{'b' * 128}.1.ab", None)
    started = []
    CC.expire(cache, now=NOW, delete=lambda top: started.append(CC._sweep_apart(top)))
    assert len(started) == 1 and not isinstance(started[0], OSError)
    assert started[0].wait() == 0 and not away.exists()


def test_a_run_whose_version_goes_while_it_stores_stores_nothing_and_does_not_raise(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("a")
    got = {"files": ["a.txt"], "dirs": [], "probes": [], "libs": []}
    k, rep, cache = {"version": "v", "request": {}}, {"failures": [], "warnings": []}, tmp_path / "cache"
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    replace = os.replace

    def gone(src, dst):
        if Path(dst).suffix == ".json" and CC.READS not in Path(dst).parts:
            os.rename(cache / "v", cache / f"{CC.EXPIRED}v")                     # another run removes this version here
        replace(src, dst)

    monkeypatch.setattr(os, "replace", gone)
    monkeypatch.setattr(CC, "LOST", [])
    assert CC.store(cache, k, tmp_path, got, rep) is None
    assert len(CC.LOST) == 1 and "FileNotFoundError" in CC.LOST[0]
    assert CC.lookup(cache, k, tmp_path) is None
    CC._VISIBLE.clear()
    CC._MEMO.clear()


def test_a_read_only_run_marks_its_version_only_where_the_cache_holds_it(tmp_path):
    cache, own = tmp_path / "cache", CC.version()
    assert CC.mark(cache, now=NOW) is None
    assert not cache.exists()
    (cache / own).mkdir(parents=True)
    assert CC.mark(cache, now=NOW) is None
    assert _used(cache, own) == NOW
