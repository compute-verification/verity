"""Forked replay / export workers die with their parent: a SIGKILLed Commit leaves no worker holding its copy-on-write heap."""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys

import pytest

from verity_vllm.check.replay.fork import die_with_parent

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")

# each worker enters its task, says so, and blocks on a pipe nobody writes to; once all three are in, the parent prints their pids
# and waits for the test to close its stdin before it SIGKILLs itself
PARENT = """
import os, signal, sys
import multiprocessing as mp
from verity_vllm.check.replay.fork import fork_pool
(block, _), (started, started_w) = os.pipe(), os.pipe()
def work(_):
    os.write(started_w, b".")
    os.read(block, 1)
pool = fork_pool(3) if sys.argv[1] == "fork_pool" else mp.get_context("fork").Pool(3)
pool.map_async(work, range(3))
n = 0
while n < 3:
    n += len(os.read(started, 3 - n))
print(" ".join(str(p.pid) for p in pool._pool), flush=True)
sys.stdin.read()
os.kill(os.getpid(), signal.SIGKILL)
"""


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _workers_after_parent_sigkill(kind: str) -> tuple[list[int], list[int]]:
    """The workers' pids and pidfds once their parent is SIGKILLed and reaped; the pidfds are opened while the parent, and so every
    worker, is alive."""
    p = subprocess.Popen([sys.executable, "-c", PARENT, kind], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True)
    pids = [int(x) for x in p.stdout.readline().split()]
    assert len(pids) == 3
    fds = [os.pidfd_open(q) for q in pids]
    p.stdin.close()
    # wait for the parent only: orphaned workers keep the stdout pipe open, so EOF may never come
    assert p.wait() == -signal.SIGKILL
    p.stdout.close()
    return pids, fds


def test_fork_pool_workers_die_with_a_sigkilled_parent():
    # the parent's exit queued each worker's PR_SET_PDEATHSIG SIGKILL before `wait` returned; a pidfd is readable once its process exited
    _, fds = _workers_after_parent_sigkill("fork_pool")
    for fd in fds:
        select.select([fd], [], [])
        os.close(fd)


def test_a_plain_fork_pool_orphans_its_workers():
    """The control: without the initializer the workers outlive a SIGKILLed parent (the #23 Commit's two 54 GB workers)."""
    pids, fds = _workers_after_parent_sigkill("plain")
    try:
        assert [q for q in pids if _alive(q)] == pids   # blocked on a pipe nobody writes to, only a signal ends them
    finally:
        for fd in fds:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            os.close(fd)


def test_a_child_forked_after_its_parent_died_exits_at_once():
    pid = os.fork()
    if pid == 0:
        die_with_parent(os.getppid() + 1)   # not this child's parent: as if the parent died before the child ran
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 1


def _pdeathsig(_=None) -> tuple[int, int]:
    import ctypes
    sig = ctypes.c_int(0)
    assert ctypes.CDLL(None, use_errno=True).prctl(2, ctypes.byref(sig), 0, 0, 0) == 0   # PR_GET_PDEATHSIG
    return os.getpid(), sig.value


def _die(_=None) -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def test_a_replacement_worker_sets_no_signal_and_the_pool_terminates():
    """The pool forks a dead worker's replacement from its worker-handler thread, which `terminate()` ends: a signal following that
    thread would kill the replacement inside `terminate()`, maybe holding the task queue's read lock, and `terminate()` would hang."""
    from verity_vllm.check.replay.fork import fork_pool

    with fork_pool(1) as pool:
        first, sig = pool.apply(_pdeathsig)
        assert sig == signal.SIGKILL
        pool.apply_async(_die)
        second, sig = pool.apply(_pdeathsig)
        assert second != first and sig == 0
