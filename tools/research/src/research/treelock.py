"""`python3 treelock.py LOCK WAIT_S TREE -- CMD...`: CMD holding an exclusive flock on LOCK."""
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
LEAN_BUILD = re.compile(r"(?:^|[\s/'\"(;&|])(?:lake|lean_changed\.py|lean_audit\.py|check\.py|suites\.py)(?=$|[\s'\";&|)])"
                        r"|(?:^|[\s/'\"(;&|])verity\s+(?:check|lean)(?=$|[\s'\";&|)])")


def builds_lean(command: list[str]) -> bool:
    return bool(LEAN_BUILD.search(" ".join(command)))


def lock_path(runs_root: Path, tree: Path) -> Path:
    return runs_root / f".lean-tree-{hashlib.sha256(os.fsencode(os.path.realpath(tree))).hexdigest()[:16]}.lock"


def wrap(lock: Path, tree: Path, wait_s: float | None = None) -> list[str]:
    return [sys.executable, "-P", str(Path(__file__).resolve()), str(lock), f"{WAIT_S if wait_s is None else wait_s:g}", str(tree), "--"]


def acquire(lock: str, wait_s: float, tree: str, *, clock=time.monotonic, sleep=time.sleep) -> int | None:
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
