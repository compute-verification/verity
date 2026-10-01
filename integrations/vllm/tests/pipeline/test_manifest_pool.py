"""`manifest.build_components` takes the machine's pool lock only when its headroom is the host's, shared: a `budget_gb` or a
cgroup `memory.max` below the host's RAM is the process's own budget, which it sizes the pool from without waiting on other runs."""

from __future__ import annotations

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
