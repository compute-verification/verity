"""`python3 treelock.py LOCK WAIT_S TREE -- CMD...`: CMD holding an exclusive flock on LOCK, after waiting up to WAIT_S seconds for it.

`research run --cwd source` runs a workload in the source root, which on a machine is <root>/src/<sha>/, shared by every run of
the commit; two Lean builds there write one `.lake` at once, truncating each other's generated .c files under clang (SIGBUS,
INF-641).  The runner puts a workload that builds Lean there (`builds_lean`) behind this, one lock per tree, so such runs take the
tree in turn.  The wait is the workload's own time, so the run's --timeout bounds it too.  CMD inherits the lock (exec), and so
do the processes it starts.  Stdlib only: it runs by path, without the research package on its path.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import sys
import time
from pathlib import Path

WAIT_S = 4 * 3600
POLL_S = 5.0
EX_NOT_STARTED = 75
#: a command word that builds Lean in its tree: lake itself, the Lean steps' drivers, and what runs them (check.py's Lean steps,
#: suites.py's AFTER_LEAN suites, `verity check|lean`)
LEAN_BUILD = re.compile(r"(?:^|[\s/'\"(;&|])(?:lake|lean_changed\.py|lean_audit\.py|check\.py|suites\.py)(?=$|[\s'\";&|)])"
                        r"|(?:^|[\s/'\"(;&|])verity\s+(?:check|lean)(?=$|[\s'\";&|)])")


def builds_lean(command: list[str]) -> bool:
    return bool(LEAN_BUILD.search(" ".join(command)))


def lock_path(runs_root: Path, tree: Path) -> Path:
    """The tree's lock, beside the runs (`.exclusive.lock`'s directory): the tree itself is the shipped commit's, and a checkout's
    parent may not be writable."""
    return runs_root / f".lean-tree-{hashlib.sha256(os.fsencode(os.path.realpath(tree))).hexdigest()[:16]}.lock"


def wrap(lock: Path, tree: Path, wait_s: float = WAIT_S) -> list[str]:
    # -P: this file's directory, the research package, would shadow the standard library's modules (its queue.py)
    return [sys.executable, "-P", str(Path(__file__).resolve()), str(lock), f"{wait_s:g}", str(tree), "--"]


def acquire(lock: str, wait_s: float, tree: str, *, clock=time.monotonic, sleep=time.sleep) -> int | None:
    """An fd holding LOCK exclusively, or None once WAIT_S has passed without it; says on stderr that it waits, and why."""
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    t0, said = clock(), False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if clock() - t0 >= wait_s:
                os.close(fd)
                return None
        if not said:
            print(f"research: waiting up to {wait_s:g} s for another run's Lean build in the shared tree {tree} to end ({lock}); "
                  f"`research run --cwd clone` gives a run a tree of its own", file=sys.stderr, flush=True)
            said = True
        sleep(POLL_S)


def main(argv: list[str]) -> int:
    if len(argv) < 5 or argv[3] != "--":
        print("usage: treelock.py LOCK WAIT_S TREE -- CMD...", file=sys.stderr)
        return 2
    lock, wait_s, tree, cmd = argv[0], float(argv[1]), argv[2], argv[4:]
    fd = acquire(lock, wait_s, tree)
    if fd is None:
        print(f"research: refused: another run's Lean build held the shared tree {tree} for {wait_s:g} s ({lock}), so nothing ran; "
              f"launch it with `research run --cwd clone` (the run's own tree) to build beside it", file=sys.stderr, flush=True)
        return EX_NOT_STARTED
    os.set_inheritable(fd, True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
