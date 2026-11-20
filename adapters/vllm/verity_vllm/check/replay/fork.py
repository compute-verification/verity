"""Forked children that die with their parent.

A forked child blocked on a pipe whose write end it inherited at the fork outlives a SIGKILLed parent (an OOM kill, a `kill -9`):
no EOF ever arrives, and it keeps its copy-on-write view of the parent's heap.  Linux: `die_with_parent` sets `PR_SET_PDEATHSIG` to
SIGKILL, so the kernel kills the child when the parent dies, and exits at once if the parent is already gone by the time it runs.
Elsewhere it is a no-op.  The signal follows the parent THREAD that forked the child.
"""
from __future__ import annotations

import ctypes
import os
import signal
import sys
import threading

__all__ = ["die_with_parent"]

PR_SET_PDEATHSIG = 1


def die_with_parent(parent_pid: int, forker: int | None = None) -> None:
    """In a forked child: have the kernel SIGKILL this process when its parent dies (`PR_SET_PDEATHSIG` is per parent thread).  A
    parent that died before the call is caught by the ppid check.  `forker`: the `threading.get_ident()` of the parent thread the
    signal may follow; a child another thread forked sets no signal."""
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
