"""`manifest.build_components` and the word check's prefetch (`prefetch_word_rules`) take the machine's pool lock only when their
headroom is the host's, shared: a `budget_gb` or a cgroup `memory.max` below the host's RAM is the process's own budget, which they
size their pools from without waiting on other runs.  The lock is one file per machine that every user opens read-only and never
with O_CREAT once it exists (`_open_shared_lock`)."""

from __future__ import annotations

import errno
import fcntl
import os
import pwd
import stat
import tempfile
from contextlib import contextmanager

import pytest

from verity_vllm.pipeline import manifest as M

GiB = 1 << 30


@pytest.fixture
def host(tmp_path, monkeypatch):
    def set_(memory_max: str | None, total_gib: int = 512, head: int | None = 100 * GiB):
        mx = tmp_path / "memory.max"
        if memory_max is None:
            mx.unlink(missing_ok=True)
        else:
            mx.write_text(memory_max + "\n")
        (tmp_path / "meminfo").write_text(f"MemTotal:       {total_gib << 20} kB\nMemAvailable:   1 kB\n")
        monkeypatch.setattr(M, "CGROUP_MEMORY_MAX", str(mx))
        monkeypatch.setattr(M, "MEMINFO", str(tmp_path / "meminfo"))
        monkeypatch.setattr(M, "_headroom", lambda: head)
    return set_


def test_the_budget_is_the_given_one_capped_by_the_headroom_else_a_cgroup_below_the_host_else_none(host):
    host("max")
    assert M._own_budget() is None
    host(None)
    assert M._own_budget() is None
    host(str(512 * GiB))
    assert M._own_budget() is None
    host(str(96 * GiB), head=90 * GiB)
    assert M._own_budget() == 90 * GiB
    host("max")
    assert M._own_budget(40) == 40 * GiB
    host("max", head=30 * GiB)
    assert M._own_budget(40) == 30 * GiB
    host("max", head=None)
    assert M._own_budget(40) == 40 * GiB


def test_build_components_takes_the_lock_only_without_a_budget_of_its_own(host, monkeypatch):
    locks, heads = [], []

    @contextmanager
    def lock():
        locks.append(1)
        yield

    monkeypatch.setattr(M, "_pool_lock", lock)
    monkeypatch.setattr(M, "_build_forked", lambda tasks, jobs, after, head, log: heads.append(head) or [])
    tasks = [("a", {}), ("b", {})]
    host(str(96 * GiB), head=90 * GiB)
    M.build_components(tasks, jobs=2)
    assert (locks, heads) == ([], [90 * GiB])
    host("max", head=70 * GiB)
    M.build_components(tasks, jobs=2)
    assert (locks, heads) == ([1], [90 * GiB, 70 * GiB])


@pytest.mark.parametrize("jobs", [1, 2])
def test_each_component_built_adds_a_progress_line_to_the_log(host, monkeypatch, jobs):
    """A TP manifest with no limit (`row --manifest-timeout 0`) shows its progress by these lines: one as each component is built,
    in turn or in the forked pool."""
    from verity_vllm.pipeline import global_program
    monkeypatch.setattr(M, "build", lambda d, **kw: {"d": d})
    monkeypatch.setattr(global_program, "component_spec_histogram", lambda d: {"Gemm_v1{K=4,N=4}": 1})
    host(str(96 * GiB), head=90 * GiB)
    tasks, lines = [(f"c{i}", {}) for i in range(3)], []
    assert M.build_components(tasks, jobs=jobs, log=lines.append) == [{"d": d} for d, _kw in tasks]
    assert [ln.split(" built in ")[0] for ln in lines if " built in " in ln] == [f"  components: {k}/3" for k in (1, 2, 3)], lines


def test_the_word_rules_prefetch_budgets_its_share_of_that_headroom(host, monkeypatch):
    """`word.prefetch` gets `HEADROOM_SHARE` of the headroom `build_components` would budget, taking the lock where it would; an
    unreadable headroom is no budget, and a Build without a histogram prefetches nothing and takes no lock."""
    from verity_vllm.pipeline import global_program
    locks, budgets = [], []

    @contextmanager
    def lock():
        locks.append(1)
        yield

    monkeypatch.setattr(M, "_pool_lock", lock)
    monkeypatch.setattr(global_program, "component_spec_histogram", lambda d: {} if d == "bare" else {"Gemm_v1{K=4,N=4}": 1})
    monkeypatch.setattr(M.word_query, "prefetch", lambda specs, X, W, extras, *, max_gates, jobs, budget: budgets.append(budget) or 0)
    host(str(96 * GiB), head=90 * GiB)
    M.prefetch_word_rules(["row"], "16/32", jobs=2)
    host("max", head=70 * GiB)
    M.prefetch_word_rules(["row"], "16/32", jobs=2)
    M.prefetch_word_rules(["row"], "16/32", jobs=2, budget_gb=40)
    M.prefetch_word_rules(["bare"], "16/32", jobs=2)
    host("max", head=None)
    M.prefetch_word_rules(["row"], "16/32", jobs=2)
    share = M.HEADROOM_SHARE
    assert locks == [1, 1] and budgets == [int(90 * GiB * share), int(70 * GiB * share), int(40 * GiB * share), None]


# --- the lock file: one per machine, whichever user made it -------------------------------------------------------------------------

@pytest.fixture
def umask():
    old = os.umask(0o022)
    yield os.umask
    os.umask(old)


def held_elsewhere(path) -> bool:
    """True when another open of `path` cannot take the lock without waiting (flock locks belong to the open file description)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def opens_of(monkeypatch, path, before=None) -> list[int]:
    """The flags of every `os.open` of `path`; `before(flags, n)` runs ahead of the n-th (1-based), as another process would."""
    real, calls = os.open, []

    def spy(p, flags, *a, **kw):
        if os.fspath(p) == os.fspath(path):
            calls.append(flags)
            if before:
                before(flags, len(calls))
        return real(p, flags, *a, **kw)
    monkeypatch.setattr(os, "open", spy)
    return calls


def test_an_existing_lock_it_cannot_write_is_opened_read_only_and_excludes_a_second_holder(tmp_path):
    lock = tmp_path / M.POOL_LOCK
    lock.write_text("")
    lock.chmod(0o444)       # to its owner as another user's 0o664 file is to this one: readable, not writable
    with M._pool_lock(str(lock)):
        assert held_elsewhere(lock)
    assert not held_elsewhere(lock)
    assert stat.S_IMODE(lock.stat().st_mode) == 0o444


def test_an_existing_lock_is_never_opened_with_o_creat(tmp_path, monkeypatch):
    """`fs.protected_regular` refuses an O_CREAT open of another user's file in a sticky world-writable directory, whatever its mode."""
    lock = tmp_path / M.POOL_LOCK
    lock.write_text("")
    calls = opens_of(monkeypatch, lock)
    with M._pool_lock(str(lock)):
        pass
    assert calls == [os.O_RDONLY]


@pytest.mark.parametrize("mask", [0o022, 0o077, 0o777], ids=lambda m: f"umask-{m:03o}")
def test_a_missing_lock_is_created_0o666_whatever_the_umask(tmp_path, umask, mask):
    lock = tmp_path / M.POOL_LOCK
    umask(mask)
    with M._pool_lock(str(lock)):
        assert held_elsewhere(lock)
    assert stat.S_IMODE(lock.stat().st_mode) == 0o666


def test_the_default_lock_is_the_one_name_in_the_temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with M._pool_lock():
        assert held_elsewhere(tmp_path / M.POOL_LOCK)


def test_losing_the_creation_race_opens_the_winners_lock_unchanged(tmp_path, monkeypatch):
    lock = tmp_path / M.POOL_LOCK

    def another_creates_it(flags, n):
        if n == 2:          # between this process's plain open (ENOENT) and its create
            lock.write_text("")
            lock.chmod(0o640)
    calls = opens_of(monkeypatch, lock, another_creates_it)
    with M._pool_lock(str(lock)):
        opened = list(calls)
        assert held_elsewhere(lock)
    assert opened == [os.O_RDONLY, os.O_RDONLY | os.O_CREAT | os.O_EXCL, os.O_RDONLY]
    assert stat.S_IMODE(lock.stat().st_mode) == 0o640


def test_a_lock_that_keeps_vanishing_raises_by_name_after_its_attempts(tmp_path, monkeypatch):
    lock = tmp_path / M.POOL_LOCK

    def churn(flags, n):
        if flags & os.O_CREAT:
            lock.write_text("")
        else:
            lock.unlink(missing_ok=True)
    calls = opens_of(monkeypatch, lock, churn)
    with pytest.raises(RuntimeError, match="kept appearing and vanishing over 4 attempts"):
        with M._pool_lock(str(lock)):
            pytest.fail("locked a file that was never there")
    assert calls == [os.O_RDONLY, os.O_RDONLY | os.O_CREAT | os.O_EXCL] * 4


@pytest.mark.skipif(os.geteuid() == 0, reason="root opens a mode-0 file")
def test_a_lock_it_cannot_read_raises_naming_its_path_and_owner(tmp_path):
    lock = tmp_path / M.POOL_LOCK
    lock.write_text("")
    lock.chmod(0)
    with pytest.raises(PermissionError) as e:
        with M._pool_lock(str(lock)):
            pytest.fail("locked a file it cannot read")
    msg = str(e.value)
    assert e.value.errno == errno.EACCES and e.value.filename == str(lock)
    assert f"owner {pwd.getpwuid(os.getuid()).pw_name}, uid {os.getuid()}, mode 0)" in msg and "chmod a+r" in msg
