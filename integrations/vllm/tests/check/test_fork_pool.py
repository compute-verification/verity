"""Forked replay / export workers die with their parent: a SIGKILLed Commit leaves no worker holding its copy-on-write heap."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from verity_vllm.check.replay.fork import die_with_parent

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")

PARENT = """
import os, signal, sys, time
import multiprocessing as mp
from verity_vllm.check.replay.fork import fork_pool
pool = fork_pool(3) if sys.argv[1] == "fork_pool" else mp.get_context("fork").Pool(3)
pool.map_async(time.sleep, [120] * 3)
time.sleep(1.0)
print(" ".join(str(p.pid) for p in pool._pool), flush=True)
os.kill(os.getpid(), signal.SIGKILL)
"""


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _workers_after_parent_sigkill(kind: str) -> tuple[list[int], list[int]]:
    # read the pid line and wait for the parent only: orphaned workers keep the stdout pipe open, so EOF may never come
    p = subprocess.Popen([sys.executable, "-c", PARENT, kind], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = p.stdout.readline()
    assert p.wait(timeout=60) == -signal.SIGKILL
    p.stdout.close()
    pids = [int(x) for x in line.split()]
    assert len(pids) == 3
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(_alive(q) for q in pids):
        time.sleep(0.1)
    return pids, [q for q in pids if _alive(q)]


def test_fork_pool_workers_die_with_a_sigkilled_parent():
    _, alive = _workers_after_parent_sigkill("fork_pool")
    assert alive == []


def test_a_plain_fork_pool_orphans_its_workers():
    """The control: without the initializer the workers outlive a SIGKILLed parent (the #23 Commit's two 54 GB workers)."""
    pids, alive = _workers_after_parent_sigkill("plain")
    try:
        assert alive == pids
    finally:
        for q in alive:
            os.kill(q, signal.SIGKILL)


def test_a_child_forked_after_its_parent_died_exits_at_once():
    pid = os.fork()
    if pid == 0:
        die_with_parent(os.getppid() + 1)   # not this child's parent: as if the parent died before the child ran
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 1
