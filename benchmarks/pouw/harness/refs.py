"""The host references an arm's gates compare against, off the lease's critical path.

An arm whose gate recomputes outputs on the host (a numpy twin) can hand that work to `pool()`, a process pool sized by the
job's CPUs (`cpus()`), so a gate's references are computed side by side instead of one after another. An arm may also have
``prefetch(dev, call, shape, flags)`` (optional, arm interface v0.6): before a flag set's first gate, the harness gives it
every call it is about to gate (`bench._arm_gate`), so all their references are in flight while the first gate runs. A
prefetch only starts work: the gate still takes what it compares against from the device as it is at the gate, and ignores
a reference made from other inputs.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from verity_config.settings import of

settings = of(__file__)
_POOL: ProcessPoolExecutor | None = None


def cpus() -> int:
    """The job's CPUs: ``OMP_NUM_THREADS`` (a fill job's ``cpus=``), at most the CPUs the process may run on."""
    allowed = len(os.sched_getaffinity(0))
    try:
        n = settings.OMP_NUM_THREADS
    except ValueError:
        n = None
    return allowed if n is None else max(1, min(n, allowed))


def pool() -> ProcessPoolExecutor | None:
    """The job's pool, made on first use, or None with one CPU (the arm then computes inline). Its workers are forked, so
    they have every module the arm loaded; a forked worker can't use the parent's CUDA context, so what an arm submits
    must not touch the device."""
    global _POOL
    if _POOL is None and cpus() > 1:
        _POOL = ProcessPoolExecutor(cpus(), mp_context=multiprocessing.get_context("fork"))
        atexit.register(_POOL.shutdown, cancel_futures=True)
    return _POOL


def prefetch(arm, dev, calls: list, shape, flags: frozenset[str]) -> None:
    """Has the arm start the references its gate will compare each of `calls` against, if it has ``prefetch``."""
    fn = getattr(arm, "prefetch", None)
    if fn is not None:
        dev.sync()
        for call in calls:
            fn(dev, call, shape, flags)
