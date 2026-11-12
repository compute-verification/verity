"""A machine's own copy of slot.py: a run's slot record says which copy took its place in the line; a line holds tickets of
older and newer copies together; and a copy replaced while a waiter waits changes nothing for that waiter."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import threading
from datetime import datetime, UTC
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "tools" / "verity" / "check" / "slot.py"
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
V1, V2 = "1" * 40, "2" * 40


class Exec(Exception):
    pass


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _hold(path: Path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """``machine.load(name, at=None)``: slot.py as a process that starts now reads it, from the machine's own copy (or from
    ``at``, a tree's), on a machine whose copy, install record, slots, disk and reservations are all under tmp_path. Outside
    any run or slot of the check that runs these tests. A slot one of them takes stays held, as by its command, until
    ``machine.ends()`` closes the oldest such lock."""
    for k in ("RESEARCH_RUN_DIR", "RESEARCH_SOURCE_SHA", "CHECK_SLOT", "CHECK_SLOTS", "CHECK_RUN_MEM_GB", "CHECK_MIN_MEM_GB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: None)

    class Machine:
        copy = tmp_path / "node" / "check-slot" / "tools" / "verity" / "check" / "slot.py"
        record = tmp_path / "node" / "deploy" / "record.jsonl"
        locks = tmp_path / "locks"
        taken: list[int] = []

        def ends(self) -> None:
            os.close(self.taken.pop(0))

        def install(self, text: str, commit: str | None) -> str:
            """As `research deploy install` does: the new file written beside the copy and renamed over it, then the record's
            line (none with no ``commit``: an install cut short there, or a copy changed by hand); the bytes' blob."""
            self.copy.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.copy.with_name(f".{self.copy.name}.deploy-tmp")
            tmp.write_text(text)
            os.replace(tmp, self.copy)
            if commit:
                self.record.parent.mkdir(parents=True, exist_ok=True)
                with self.record.open("a") as f:
                    f.write(json.dumps({"at": "20261001T090000Z", "path": str(self.copy), "blob": _blob(text.encode()),
                                        "commit": commit, "ref": "origin/main", "by": "infra"}) + "\n")
            return _blob(text.encode())

        def load(self, name: str, at: Path | None = None):
            spec = importlib.util.spec_from_file_location(name, at or self.copy)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            monkeypatch.setattr(mod, "NODE_COPY", self.copy)
            monkeypatch.setattr(mod, "DEPLOY_RECORD", self.record)
            monkeypatch.setattr(mod, "SLOTS", self.locks / "slots")
            monkeypatch.setattr(mod, "RAMLOCK", tmp_path / "ramlock")
            monkeypatch.setattr(mod, "DISK", tmp_path / "no-disk")
            monkeypatch.setattr(mod, "DISK_MIN_GB", 0)
            monkeypatch.setattr(mod, "DISK_MIN_INODES", 0)
            monkeypatch.setattr(mod, "allowed", lambda: set(range(1024)))
            monkeypatch.setattr(mod, "memory_gb", lambda root=None: pytest.fail("reads this host's free memory"))
            take = mod.take
            monkeypatch.setattr(mod, "take", lambda *a, **kw: (got := take(*a, **kw)) and (self.taken.append(got[1]) or got))
            return mod

    m = Machine()
    m.locks.mkdir()
    (m.locks / "slots").write_text("a 0\n")
    yield m
    while m.taken:
        m.ends()


def test_a_runs_slot_record_names_the_machines_copy_by_the_commit_its_install_recorded(machine, tmp_path, monkeypatch):
    """The record's newest line for the copy names its commit only while it names the copy's bytes: a copy changed since
    (by hand, or by an install that hasn't written its line yet) is named by its bytes, and the record says why no commit
    is. A record that can't be read is that reason, never a run that doesn't start."""
    text = SOURCE.read_text()
    blob = machine.install(text, V1)
    slot = machine.load("slot_machine_copy")
    assert slot.copy() == {"of": "machine", "blob": blob, "commit": V1}
    with machine.record.open("a") as f:                       # the record's other lines: a restart, a move to the attic, torn ones
        f.write(json.dumps({"at": "x", "unit": "systemd:a.service", "restart": "restarted"}) + "\n")
        f.write(json.dumps({"at": "x", "path": str(machine.copy), "attic": "/attic/slot.py", "by": "infra"}) + "\n{not json\n")
        f.write(json.dumps({"at": "x", "path": str(machine.copy), "blob": "0" * 40})[:-9] + "\n")
    assert slot.copy() == {"of": "machine", "blob": blob, "commit": V1}, "a torn line hides no install written whole"
    newer = machine.install(text + "\n# changed\n", None)
    assert machine.load("slot_machine_changed").copy() == {
        "of": "machine", "blob": newer,
        "no_commit": f"{machine.record} names blob {blob} last for this file, and 1 of its lines for it can't be read"}
    machine.record.unlink()
    got = machine.load("slot_machine_unrecorded").copy()
    assert got == {"of": "machine", "blob": newer, "no_commit": got["no_commit"]}
    assert got["no_commit"].startswith(f"{machine.record}: FileNotFoundError: ")
    machine.record.write_text(json.dumps({"at": "x", "path": str(machine.copy) + ".other", "blob": newer, "commit": V1}) + "\n")
    assert machine.load("slot_machine_unnamed").copy() == {
        "of": "machine", "blob": newer, "no_commit": f"{machine.record} names no install of this file"}
    machine.record.unlink()

    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run))
    machine.install(text, V2)
    slot = machine.load("slot_machine_run")
    monkeypatch.setattr(os, "execvpe", lambda *a: (_ for _ in ()).throw(Exec()))
    with pytest.raises(Exec):
        slot.main(["--by", "lane", "--", "true"], now=lambda: NOW, sleep=lambda s: pytest.fail("a free slot is taken at once"))
    doc = json.loads((run / slot.RUN_SLOT).read_text())
    assert doc["state"] == "held" and doc["copy"] == {"of": "machine", "blob": blob, "commit": V2}


def test_a_trees_copy_says_so_in_one_line_only_where_the_machine_has_none_and_a_pool_is_named(machine, tmp_path, monkeypatch, capsys):
    """A run that names a pool on a machine without a copy of its own waits by its tree's, and its log says so once; a check
    that names no pool (the machine's own check slots) says nothing, whichever copy the machine has."""
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run))
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", V1)
    monkeypatch.setattr(os, "execvpe", lambda *a: (_ for _ in ()).throw(Exec()))
    tree = machine.load("slot_of_a_tree", at=SOURCE)
    want = {"of": "tree", "blob": _blob(SOURCE.read_bytes()), "commit": V1}

    def lines() -> list[str]:
        with pytest.raises(Exec):
            tree.main(["--", "true"], now=lambda: NOW, sleep=lambda s: pytest.fail("a free slot is taken at once"))
        machine.ends()
        assert json.loads((run / tree.RUN_SLOT).read_text())["copy"] == want
        return [ln for ln in capsys.readouterr().err.splitlines() if "of its own" in ln]

    assert lines() == [], "no pool is named: the machine's own check slots, taken by the tree's copy"
    monkeypatch.setenv("CHECK_SLOTS", str(machine.locks / "slots"))
    assert lines() == [f"slot: this machine has no slot.py of its own ({machine.copy}), so the run's tree's copy holds its place in "
                       f"the pool's line"]
    machine.install(SOURCE.read_text(), V2)
    assert lines() == [], "the machine has a copy: the launch says why this run's command names its tree's"
    monkeypatch.delenv("RESEARCH_SOURCE_SHA")
    del want["commit"]
    assert lines() == []


def test_slot_py_names_the_check_cache_as_common_py_does_and_reads_no_file_beside_it(machine, tmp_path, monkeypatch):
    """slot.py is whole wherever it runs: the machine's copy is one file, so it names the check cache itself. The two names
    must stay one, and a common.py beside a copy is never loaded."""
    spec = importlib.util.spec_from_file_location("check_common_for_slot_copy", SOURCE.with_name("common.py"))
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    machine.install(SOURCE.read_text(), V1)
    (machine.copy.parent / "common.py").write_text("raise SystemExit('a copy of slot.py loaded the common.py beside it')\n")
    slot = machine.load("slot_machine_cache")
    for cache in (str(tmp_path / "cache" / "verity-check"), ""):
        monkeypatch.setenv("CHECK_CACHE", cache)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert slot.cache_dir() == common.cache_dir()
    monkeypatch.delenv("CHECK_CACHE")
    assert slot.cache_dir() == common.cache_dir() == tmp_path / "home" / ".cache" / "verity-check"
    assert slot.cache_disk() == tmp_path
    assert slot.blocked(slot.parse("a 0")[0], NOW) is None


def test_a_line_holding_tickets_of_an_older_copy_and_a_newer_one_is_read_whole(machine, tmp_path, monkeypatch, capsys):
    """A pool's line holds the tickets of every copy that has a waiter in it. A ticket from before tickets named what their
    run asks (no `mem_gb`), and one from before tickets named anything, are waiters as any other: listed, in their place, and
    never passed, since nothing says memory is all they wait for. A ticket that names its ask is passed by a small run
    while less than that is free."""
    slot = machine.load("slot_tree_mixed", at=SOURCE)
    line = machine.locks / slot.LINE
    older, o = slot.join(machine.locks, who={"pid": 11, "run": "r-older", "tree": "/src/older", "commit": V1, "by": "lane-a"})
    oldest, e = slot.join(machine.locks)                                          # an empty ticket: its name alone
    asks, a = slot.join(machine.locks, who={"pid": 13, "run": "r-asks", "tree": "/src/newer", "commit": V2, "by": "lane-b", "mem_gb": 400})
    monkeypatch.setenv(slot.RUN_MEM_ENV, "8")
    mine, m = slot.join(machine.locks, who=slot.asker("lane-c"))
    assert [p.name for p in sorted(line.iterdir())] == [older.name, oldest.name, asks.name, mine.name]

    waits = slot.waiting(line, NOW, before=mine.name)
    assert [(w["pid"], w.get("run"), w.get("by"), w.get("mem_gb")) for w in waits] == [
        (11, "r-older", "lane-a", None), (os.getpid(), None, None, None), (13, "r-asks", "lane-b", 400)]
    assert json.loads(mine.read_text())["mem_gb"] == 8
    assert slot.ahead(mine) == 3
    assert slot.ahead(mine, free_gb=100) == 2, "only the ticket that names an ask past what is free is passed"
    assert slot.ahead(mine, free_gb=500) == 3
    assert capsys.readouterr().err == "", "an older ticket is no unreadable one"
    doc = slot.status_doc(NOW)
    assert [w.get("run") for w in doc["waiters"]] == ["r-older", None, "r-asks", None]

    # the same line through main(): a small run behind it waits while an older ticket is ahead, whatever is free, and takes
    # the idle slot once only the one that asks more than is free is ahead of it
    os.close(m)
    mine.unlink()
    monkeypatch.setattr(slot, "memory_gb", lambda root=None: 100.0)
    monkeypatch.setattr(slot, "reserved_gb", lambda **kw: 0.0)
    monkeypatch.setattr(os, "execvpe", lambda *a: (_ for _ in ()).throw(Exec()))
    polls = []

    def sleep(s):
        polls.append(capsys.readouterr().err.splitlines()[0])
        for fd in (o, e):                                                         # the older copies' waiters leave the line
            os.close(fd)

    with pytest.raises(Exec):
        slot.main(["--by", "lane-c", "--", "true"], now=lambda: NOW, sleep=sleep)
    assert polls == [f"slot: 2 check(s) ahead in line; retrying every {slot.POLL_S} s"]
    assert [p.name for p in sorted(line.iterdir())] == [asks.name], "the waiter that asks 400 GB keeps its place"
    os.close(a)


def test_a_copy_replaced_while_a_waiter_waits_leaves_that_waiter_its_rules_and_its_place(machine, tmp_path, monkeypatch):
    """An install renames a new file over the machine's copy between two admissions. A waiter that started under the old copy
    read it whole when it started: it waits on by the old copy's rules and reads neither the new file nor anything beside
    it, and it keeps its place, ahead of a waiter that starts under the new copy, which the new copy admits after it.

    Each waiter is a thread here where the machine has a process, stopped at each of its polls until the test lets it go
    on, so nothing waits on a clock."""
    old_text = SOURCE.read_text()
    new_text = old_text + "\nPOLL_S = 7          # the new copy's rule\n"
    old_blob, order, failed = machine.install(old_text, V1), [], []
    runs = {n: tmp_path / f"run-{n}" for n in ("old", "new")}
    for d in runs.values():
        d.mkdir()
    monkeypatch.setattr(os, "execvpe", lambda file, args, env: (_ for _ in ()).throw(Exec(args[-1])))
    held = _hold(machine.locks / "check-a.lock")

    class Waiter:
        """One waiter's main(), stopped inside each sleep until `step()`; `polls` is what it slept each time."""

        def __init__(self, name: str, mod):
            self.name, self.mod, self.polls = name, mod, []
            self.at_poll, self.go, self.ended = threading.Event(), threading.Event(), threading.Event()
            self.thread = threading.Thread(target=self.run, daemon=True)

        def run(self):
            try:
                self.mod.main(["--by", self.name, "--", "true", self.name], now=lambda: NOW, sleep=self.sleep)
            except Exec as e:
                order.append(str(e))
            except BaseException as e:      # noqa: BLE001  the test's thread reads it: a waiter that died is the failure
                failed.append((self.name, repr(e)))
            finally:
                self.ended.set()
                self.at_poll.set()

        def sleep(self, s):
            self.polls.append(s)
            self.at_poll.set()
            self.go.wait()
            self.go.clear()

        def start(self):
            self.thread.start()
            self.at_poll.wait()

        def step(self):
            """Let it poll once more: until it sleeps again, or ends."""
            self.at_poll.clear()
            self.go.set()
            self.at_poll.wait()

        def record(self) -> dict:
            return json.loads((runs[self.name] / self.mod.RUN_SLOT).read_text())

    monkeypatch.setenv("RESEARCH_RUN_DIR", str(runs["old"]))
    old = Waiter("old", machine.load("slot_copy_old"))
    old.start()
    assert not failed and old.polls == [20] and old.record()["copy"] == {"of": "machine", "blob": old_blob, "commit": V1}

    new_blob = machine.install(new_text, V2)                 # the install: one rename, then the record's line
    assert new_blob != old_blob and machine.copy.read_text() == new_text
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(runs["new"]))
    new = Waiter("new", machine.load("slot_copy_new"))
    new.start()
    assert not failed and new.polls == [7] and new.record()["copy"] == {"of": "machine", "blob": new_blob, "commit": V2}
    assert new.record()["why"] == "1 check(s) ahead in line"

    # from here on neither waiter may read its copy's file again, nor a file beside it
    machine.copy.write_text("raise SystemExit('a waiter read its copy again')\n")
    (machine.copy.parent / "common.py").write_text("raise SystemExit('a waiter loaded the common.py beside its copy')\n")
    os.close(held)                                            # the slot frees: the line decides who takes it
    new.step()
    assert not new.ended.is_set() and new.polls == [7, 7], "the slot is free, and the waiter ahead of it has not taken it yet"
    old.step()
    assert old.ended.is_set() and order == ["old"] and old.polls == [20], "the old copy's waiter takes the slot, by the old copy"
    assert old.record()["state"] == "held" and old.record()["copy"]["commit"] == V1
    new.step()
    assert not new.ended.is_set() and new.record()["why"].startswith("every check slot is taken (a: "), "the head of the line now"
    machine.ends()                                            # the old waiter's run ends
    new.step()
    assert new.ended.is_set() and order == ["old", "new"] and not failed
    assert new.record()["state"] == "held" and new.record()["copy"] == {"of": "machine", "blob": new_blob, "commit": V2}
