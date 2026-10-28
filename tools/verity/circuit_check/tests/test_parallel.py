"""`cli.parallel`, the per-target processes behind `circuit-check --jobs`, on its own, without circuit-check's catalog."""
from __future__ import annotations

import os
import signal

import pytest

from circuit_check import cli


def test_every_item_is_yielded_once():
    assert sorted(v for _, v, _, _ in cli.parallel(abs, [-1, -2, -3], jobs=2)) == [1, 2, 3]


_KEPT: list = []


def _allocate(mb: int) -> int:
    """A fake target that allocates ``mb`` MiB and keeps it, as a check that leaves its memo resident does."""
    block = bytearray(mb << 20)
    block[::4096] = b"x" * len(block[::4096])
    _KEPT.append(block)
    return mb


def test_a_target_s_peak_is_its_own():
    """A 600 MiB target's peak shows it, and the small target after it, in the same single job, does not inherit it: a worker
    that outlived the first would keep its 600 MiB under the second's peak."""
    got = {x: (peak, start) for x, _, peak, start in cli.parallel(_allocate, [600, 1], jobs=1)}
    (big, big_start), (small, small_start) = got[600], got[1]
    assert big - big_start >= 0.55
    assert small - small_start < 0.05 and small < big - 0.4


def _die_on_7(x: int) -> int:
    if x == 7:
        os.kill(os.getpid(), signal.SIGKILL)
    return x


def test_a_target_whose_process_is_killed_is_named_and_the_rest_still_run():
    got = {x: (v, peak) for x, v, peak, _ in cli.parallel(_die_on_7, [1, 7, 2, 3], jobs=2)}
    killed, peak = got.pop(7)
    assert isinstance(killed, cli.Killed) and killed["how"] == "signal 9" and peak > 0
    assert {x: v for x, (v, _) in got.items()} == {1: 1, 2: 2, 3: 3}


def _raise(x: int) -> int:
    raise ValueError(f"no {x}")


def test_a_target_that_raises_is_reported_not_lost():
    [(_, v, _, _)] = cli.parallel(_raise, [5], jobs=1)
    assert isinstance(v, cli.Killed) and v["how"] == "raised ValueError: no 5"


#: three zygotes forked from a process of their own (a fork of pytest's could deadlock), each asked once so that it has started,
#: then the fds each holds of this process's ends of the zygotes' pipes, and twelve items run across them
ZYGOTES = """
import json, os
from circuit_check import cli

def parent(_):
    return os.getppid()

zs = [cli.Zygote(parent, fork=True) for _ in range(3)]
ends = {os.readlink(f"/proc/self/fd/{z.conn.fileno()}") for z in zs}
held = []
for z in zs:
    z.conn.send((0, 0))
    z.conn.recv()
    held.append(sorted(ends & {os.readlink(f"/proc/{z._pid}/fd/{fd}") for fd in os.listdir(f"/proc/{z._pid}/fd")}))
got = [v for _, v, _, _ in cli.parallel(parent, list(range(12)), 6, zygote=zs)]
print(json.dumps({"pids": [z._pid for z in zs], "held": held, "parents": got, "codes": [z.exitcode() for z in zs]}))
"""


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="reads the zygotes' fds in /proc")
def test_items_are_forked_from_every_zygote_and_no_zygote_holds_another_s_pipe():
    """One zygote forks a 45 GiB task list's targets about a second apart however many jobs run (`cli.ZYGOTES`), so items go to
    the zygote with the fewest running.  A zygote forked after another closes this process's end of the other's pipe: else
    the first would not see its pipe end when this process died, and would wait for an item forever."""
    import json
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-c", ZYGOTES], capture_output=True, text=True, check=True)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["held"] == [[], [], []]
    assert len(out["parents"]) == 12 and set(out["parents"]) == set(out["pids"])
    assert out["codes"] == [0, 0, 0]


class _Fn:
    def __init__(self, family: str | None = None) -> None:
        if family is not None:
            self.family = family


def test_a_constructed_family_s_members_go_in_batches_where_the_first_was(monkeypatch):
    """`cli._batched`: the members of a `CONSTRUCTED` family, each a few milliseconds to check, are batched `BATCH` at most to a
    process, each batch where its first member was in the order; every other target stays alone, a template too."""
    monkeypatch.setattr(cli, "BATCH", 2)
    fams = ["Const", None, "Const", "Input", "Const", "And", "Input", "Const"]
    tasks = [("definition", f"t{i}", None, {"roots": [_Fn(f)]}) for i, f in enumerate(fams)] + [("template", "tpl", None, {})]
    assert cli._batched(list(range(len(tasks))), tasks) == [(0, 2), 1, (3, 6), (4, 7), 5, 8]
    assert cli._batched([8, 7, 1], tasks) == [8, (7,), 1]


def test_heavy_targets_run_alone():
    """`admit` under a 160 GB budget: a 150 GiB target starts only when nothing runs, and nothing starts beside it; light ones
    fill what is left, past a heavy one that does not fit; one over the budget runs alone."""
    assert cli.admit([150, 16, 16], [], jobs=8, budget=160) == [0]
    assert cli.admit([16, 16], [150], jobs=8, budget=160) == []
    assert cli.admit([150, 16, 16], [100], jobs=8, budget=160) == [1, 2]
    assert cli.admit([16] * 12, [], jobs=8, budget=160) == list(range(8))
    assert cli.admit([200, 16], [], jobs=8, budget=160) == [0]
    assert cli.admit([16], [200], jobs=8, budget=160) == []


def test_a_budget_still_runs_every_target_each_in_a_process_of_its_own():
    got = list(cli.parallel(_pid, [1, 2, 3], jobs=3, weight=lambda _: 3, budget=4))
    assert sorted(x for x, *_ in got) == [1, 2, 3] and len({v for _, v, _, _ in got}) == 3


def _pid(_: int) -> int:
    return os.getpid()


def test_a_budget_granted_mid_run_admits_what_now_fits(monkeypatch, tmp_path):
    """When pytest ends long before circuit-check does, the targets still queued would run one at a time under the share
    it was given beside pytest.  `parallel` asks a callable budget again at each admission, so once
    check.py grants the slot (`granted_gb`, ``--mem-gb-file``) the targets still queued start together."""
    asked, picked, admit = [], [], cli.admit

    def budget() -> float:
        asked.append(len(asked))
        return 4 if cli.granted_gb(grant) is None else 9

    def recorded(queued, running, jobs, budget):
        picked.append((budget, len(admit(queued, running, jobs, budget))))
        if len(picked) == 1:
            grant.write_text("9\n")
        return admit(queued, running, jobs, budget)

    grant = tmp_path / "circuit-check.mem-gb"
    assert cli.granted_gb(grant) is None and cli.granted_gb(None) is None
    monkeypatch.setattr(cli, "admit", recorded)
    got = list(cli.parallel(_pid, [1, 2, 3], jobs=3, weight=lambda _: 3, budget=budget))
    assert sorted(x for x, *_ in got) == [1, 2, 3]
    assert picked[:2] == [(4, 1), (9, 2)] and len(asked) == len(picked)
    grant.write_text("not yet")
    assert cli.granted_gb(grant) is None


def test_jobs_0_takes_one_worker_per_160_gb_the_tightest_cgroup_leaves(tmp_path, monkeypatch):
    monkeypatch.delenv(cli.SLOT_MEM_ENV, raising=False)
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/meminfo").write_text("MemTotal: 2000000000 kB\nMemAvailable: 1600000000 kB\n")
    (tmp_path / "proc/self").mkdir()
    (tmp_path / "proc/self/cgroup").write_text("0::/user.slice/user-1001.slice/session-7.scope\n")
    scope = tmp_path / "sys/fs/cgroup/user.slice/user-1001.slice/session-7.scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text("max\n")
    (scope.parent / "memory.max").write_text(str(1600 << 30))
    (scope.parent / "memory.current").write_text(str(1200 << 30))
    assert cli.memory_gb(tmp_path) == 400
    assert cli.auto_jobs(tmp_path) == min(2, len(os.sched_getaffinity(0)))
    (scope.parent / "memory.current").write_text(str(1590 << 30))
    assert cli.auto_jobs(tmp_path) == 1


def test_a_cgroups_inactive_page_cache_counts_as_left(tmp_path):
    """A user slice at 462 of its 463 GiB, 278 GiB of it page cache, would leave a budget of ~0, and the targets would run one
    at a time. The kernel reclaims inactive_file at the limit; active_file it may
    still need, and without a memory.stat nothing is taken off."""
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/meminfo").write_text(f"MemAvailable: {380 << 20} kB\n")
    (tmp_path / "proc/self").mkdir()
    (tmp_path / "proc/self/cgroup").write_text("0::/user.slice\n")
    cg = tmp_path / "sys/fs/cgroup/user.slice"
    cg.mkdir(parents=True)
    (cg / "memory.max").write_text(f"{463 << 30}\n")
    (cg / "memory.current").write_text(f"{462 << 30}\n")
    assert cli.memory_gb(tmp_path) == 1
    (cg / "memory.stat").write_text(f"anon {184 << 30}\nfile {278 << 30}\nactive_file {20 << 30}\ninactive_file {258 << 30}\n")
    assert cli.memory_gb(tmp_path) == 259, "the limit less the anonymous memory and the active cache"
    (cg / "memory.stat").write_text(f"inactive_file {500 << 30}\n")
    assert cli.memory_gb(tmp_path) == 380, "never more than the limit (nor MemAvailable)"
    (tmp_path / "proc/meminfo").write_text("")
    assert cli.memory_gb(tmp_path) == 463
    (cg / "memory.current").write_text(f"{600 << 30}\n")
    (cg / "memory.stat").write_text("inactive_file 0\n")
    assert cli.memory_gb(tmp_path) == 0


def test_the_budget_reserves_this_process_s_own_task_list(monkeypatch):
    """`--mem-gb 149` (a 160 GB worker) with the parent holding its 9.6 GiB task list leaves 139.4 GiB for the targets, so a
    151 GiB target is over it (run alone, and warned about) and a 137.8 GiB one fits.  `--mem-gb` is a cap, never more than the
    machine has free: a check slot's mem= on a node whose other check already took most of it leaves only what is free."""
    monkeypatch.delenv(cli.SLOT_MEM_ENV, raising=False)
    monkeypatch.setattr(cli, "_rss_gb", lambda: 9.6)
    monkeypatch.setattr(cli, "memory_gb", lambda: 1500)
    budget = cli.budget_gb(149)
    assert budget == pytest.approx(139.4)
    assert cli.admit([151.02, 16], [], jobs=8, budget=budget) == [0]
    assert cli.admit([137.81], [], jobs=8, budget=budget) == [0] and cli.admit([16], [137.81], jobs=8, budget=budget) == []
    monkeypatch.setattr(cli, "memory_gb", lambda: 200)
    assert cli.budget_gb(None) == 200 and cli.budget_gb(560) == 200


def test_without_mem_gb_a_run_in_a_check_slot_keeps_to_the_slot(monkeypatch):
    """Inside slot.py's slot (CHECK_SLOT_MEM_GB) a raw `circuit-check` without `--mem-gb` budgets the slot's mem=, not
    the machine's MemAvailable; `--mem-gb` still wins, and outside a slot nothing changes."""
    monkeypatch.setattr(cli, "_rss_gb", lambda: 0.0)
    monkeypatch.setattr(cli, "memory_gb", lambda root=None: 1500)
    monkeypatch.setenv(cli.SLOT_MEM_ENV, "320")
    assert cli.budget_gb(None) == 320 and cli.budget_gb(100) == 100
    assert cli.auto_jobs() == min(2, len(os.sched_getaffinity(0)))
    monkeypatch.delenv(cli.SLOT_MEM_ENV)
    assert cli.budget_gb(None) == 1500


def test_a_circuit_check_in_an_xdist_worker_keeps_to_the_worker_s_share(monkeypatch):
    """suites.py sizes the suite's xdist workers at `worker_gb` each from a memory every worker shares, so a
    circuit-check a test starts budgets that worker's share (conftest.py's `worker_slot_gb`), or the slot's when smaller, never
    the slot or the cgroup the others share; outside xdist nothing is set.  A worker holding 25 GiB leaves its targets the rest
    of its share, one job."""
    from conftest import WORKER_GB, worker_slot_gb

    worker = {"PYTEST_XDIST_WORKER": "gw3"}
    assert worker_slot_gb(worker, 32) == 32 and worker_slot_gb({**worker, cli.SLOT_MEM_ENV: "480"}, 32) == 32
    assert worker_slot_gb({**worker, cli.SLOT_MEM_ENV: "20"}, 32) == 20
    assert worker_slot_gb({cli.SLOT_MEM_ENV: "480"}, 32) is None and worker_slot_gb({}, 32) is None
    if os.environ.get("PYTEST_XDIST_WORKER"):
        assert int(os.environ[cli.SLOT_MEM_ENV]) <= WORKER_GB
    monkeypatch.setattr(cli, "_rss_gb", lambda: 25.0)
    monkeypatch.setattr(cli, "memory_gb", lambda root=None: 900)
    monkeypatch.setenv(cli.SLOT_MEM_ENV, str(worker_slot_gb(worker, 32)))
    assert cli.budget_gb(None) == 7.0 and cli.auto_jobs() == 1


def test_the_catalog_s_build_keeps_to_what_the_session_s_workers_were_given(monkeypatch):
    """The xdist worker that builds the catalog (conftest.py's `_budget`) budgets its bindings' processes the workers' shares,
    `worker_gb` each, less what each holds as it waits: not the slot's mem=, which `check`'s suites running beside this one
    share, and never more than the slot or the memory leaves; outside xdist, `cli.budget_gb`."""
    from conftest import _budget

    monkeypatch.setattr(cli, "_rss_gb", lambda: 0.5)
    monkeypatch.setattr(cli, "memory_gb", lambda root=None: 900)
    four = {"PYTEST_XDIST_WORKER": "gw0", "PYTEST_XDIST_WORKER_COUNT": "4"}
    assert _budget({**four, cli.SLOT_MEM_ENV: "480"}, 8) == 30.0 and _budget(four, 8) == 30.0
    assert _budget({**four, cli.SLOT_MEM_ENV: "20"}, 8) == 18.0
    monkeypatch.setattr(cli, "memory_gb", lambda root=None: 10)
    assert _budget(four, 8) == 10.0
    monkeypatch.setenv(cli.SLOT_MEM_ENV, "320")
    assert _budget({cli.SLOT_MEM_ENV: "320"}, 8) == cli.budget_gb(None) == 10
