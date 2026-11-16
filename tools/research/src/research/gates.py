"""What every gate (`research merge`, its push guard, `research queue`) shares, so each fails closed the same way.

`git_env()` is the environment for a gate's subprocesses: the caller's, less what makes git read a repository other than the
one at its cwd (`GIT_LOCAL_ENV`: a hook's GIT_DIR, a scratch GIT_INDEX_FILE) and what a caller's process leaks into code it runs
(`CALLER_ENV`: its import path, the Lean and check slots it holds), with the record merge driver set (`RECORD_DRIVER`).  `refuse()` is the one way a gate says no: a line on stderr
naming why, and the shared exit code `REFUSED`.  `worktrees_shared()` is what each gate's fetch holds: the clone's worktree
lock (`WORKTREES_LOCK`), shared.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import sys
from pathlib import Path

from .clone import RECORD_DRIVER

OK, FINDING, USAGE, REFUSED = 0, 1, 2, 3

GIT_LOCAL_ENV = ("GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_CONFIG", "GIT_DIR", "GIT_GRAFT_FILE",
                 "GIT_IMPLICIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_INTERNAL_SUPER_PREFIX", "GIT_NO_REPLACE_OBJECTS",
                 "GIT_OBJECT_DIRECTORY", "GIT_PREFIX", "GIT_REPLACE_REF_BASE", "GIT_SHALLOW_FILE", "GIT_WORK_TREE")
GIT_KEPT_ENV = ("GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT")
CALLER_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "LEAN_SLOT", "LEAN_SLOT_POOL", "LEAN_SLOTS",
              "CHECK_SLOT")


def git_env(**extra: str) -> dict[str, str]:
    drop = {*GIT_LOCAL_ENV, *CALLER_ENV}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    n = int(env.get("GIT_CONFIG_COUNT") or 0)
    for i, (k, v) in enumerate(RECORD_DRIVER.items(), n):
        env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"] = k, v
    env["GIT_CONFIG_COUNT"] = str(n + len(RECORD_DRIVER))
    return {**env, **extra}


WORKTREES_LOCK = "research-worktrees.lock"


@contextlib.contextmanager
def worktrees_shared(repo: Path, env: dict[str, str] | None = None):
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=repo, capture_output=True, text=True,
                            env=env if env is not None else git_env()).stdout.strip()
    if not common:
        yield
        return
    with open(Path(repo, common) / WORKTREES_LOCK, "a") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        yield


def refuse(prog: str, why: str) -> int:
    print(f"{prog}: refused: {why}", file=sys.stderr)
    return REFUSED
