"""Forked worker pools that die with their parent.

A fork-context `multiprocessing.Pool` worker blocks on its task queue, whose write end it inherited at the fork: when the parent is
SIGKILLed (an OOM kill, a `kill -9`) no EOF ever arrives and the worker outlives it, holding its copy-on-write view of the parent's heap.
Linux: every worker sets `PR_SET_PDEATHSIG` to SIGKILL, so the kernel kills it when the parent dies, and exits at once if the parent
is already gone by the time it runs.  Elsewhere it is a no-op.  `dead_worker` tells the parent, by name, when a worker died mid-task.

The signal follows the parent THREAD that forked the worker.  The pool forks its first workers from the calling thread, but forks the
replacement of a dead worker from its worker-handler thread, which exits in `Pool.terminate()`: the kernel would SIGKILL the
replacement there, possibly while it holds the task queue's read lock, and `terminate()` would wait on the lock forever.  So a
replacement keeps only the ppid check.
"""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import signal
import sys
import threading
from multiprocessing.pool import Pool

PR_SET_PDEATHSIG = 1


def die_with_parent(parent_pid: int, forker: int | None = None) -> None:
    """In a forked child: have the kernel SIGKILL this process when its parent dies (`PR_SET_PDEATHSIG` is per parent thread).  A
    parent that died before the call is caught by the ppid check.  `forker`: the `threading.get_ident()` of the parent thread the
    signal may follow; a child another thread forked (a pool's replacement worker) sets no signal."""
    if not sys.platform.startswith("linux"):
        return
    if forker is not None and threading.get_ident() != forker:
        if os.getppid() != parent_pid:
            os._exit(1)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:
        os._exit(1)


def fork_pool(processes: int) -> Pool:
    """`mp.get_context("fork").Pool(processes)` whose workers die with the calling process."""
    return mp.get_context("fork").Pool(processes=processes, initializer=die_with_parent, initargs=(os.getpid(), threading.get_ident()))


#: seconds between the parent's checks for a dead worker: a worker that dies mid-task (an OOM kill) loses its task, and the pool would
#: otherwise wait for it forever
POOL_POLL_S = 10.0
CGROUP_MEMORY_EVENTS = "/sys/fs/cgroup/memory.events"


class WorkerDied(RuntimeError):
    """A forked replay worker exited mid-task, by name: OOM-killed when the cgroup's `memory.events` counted an OOM kill meanwhile."""


def oom_kills() -> int | None:
    try:
        with open(CGROUP_MEMORY_EVENTS) as f:
            return next(int(ln.split()[1]) for ln in f if ln.split()[:1] == ["oom_kill"])
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def dead_worker(pool: Pool, pids: set[int], oom0: int | None, workers: int, flag: str) -> str | None:
    """Why the pool lost a worker since it started (`pids`, `oom0` = `oom_kills()` then), or None while every worker it forked is
    alive; `flag` is the option that sets the worker count."""
    procs = list(pool._pool)
    gone = sorted(pids - {p.pid for p in procs} | {p.pid for p in procs if p.exitcode not in (None, 0)})
    oom = oom_kills()
    killed = oom is not None and oom0 is not None and oom > oom0
    if not gone and not killed:
        return None
    if killed:
        return (f"a forked replay worker was OOM-KILLED (cgroup memory.events oom_kill {oom0} -> {oom}; {workers} workers, pids exited "
                f"{gone}): the pool does not fit the cgroup's memory.max; pass fewer {flag}")
    return f"forked replay worker(s) {gone} exited mid-task ({workers} workers): their tasks are lost"
