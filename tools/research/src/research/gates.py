"""What every gate (`research merge`, its push guard, `research queue`) shares, so each fails closed the same way.

`git_env()` is the environment for a gate's subprocesses: the caller's, less what makes git read a repository other than the
one at its cwd (`GIT_LOCAL_ENV`: a hook's GIT_DIR, a scratch GIT_INDEX_FILE) and what a caller's process leaks into code it runs
(`CALLER_ENV`: its import path, the Lean and check slots it holds), with the record merge driver set (`RECORD_DRIVER`).  `refuse()` is the one way a gate says no: a line on stderr
naming why, and the shared exit code `REFUSED`.
"""
from __future__ import annotations

import os
import sys

from .clone import RECORD_DRIVER

OK, FINDING, USAGE, REFUSED = 0, 1, 2, 3

#: the variables `git rev-parse --local-env-vars` names that pick a repository, index or object store other than the one at
#: the cwd (git drops them itself on entering a submodule); GIT_CONFIG_PARAMETERS and GIT_CONFIG_COUNT, which it also names,
#: carry `git -c` settings such as a credential helper and pick no repository, so they stay
GIT_LOCAL_ENV = ("GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_CONFIG", "GIT_DIR", "GIT_GRAFT_FILE",
                 "GIT_IMPLICIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_INTERNAL_SUPER_PREFIX", "GIT_NO_REPLACE_OBJECTS",
                 "GIT_OBJECT_DIRECTORY", "GIT_PREFIX", "GIT_REPLACE_REF_BASE", "GIT_SHALLOW_FILE", "GIT_WORK_TREE")
GIT_KEPT_ENV = ("GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT")
#: what a caller's process leaks into a subprocess: its import path and interpreter, and the Lean build and check slots it holds
#: (a nested `lean_slot.py` or `slot.py` takes the variable as a slot already held)
CALLER_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "LEAN_SLOT", "LEAN_SLOT_POOL", "LEAN_SLOTS",
              "CHECK_SLOT")


def git_env(**extra: str) -> dict[str, str]:
    """The environment for a gate's subprocess: os.environ less GIT_LOCAL_ENV and CALLER_ENV, with `RECORD_DRIVER` after any
    GIT_CONFIG_COUNT settings the caller carries, plus ``extra``."""
    drop = {*GIT_LOCAL_ENV, *CALLER_ENV}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    n = int(env.get("GIT_CONFIG_COUNT") or 0)
    for i, (k, v) in enumerate(RECORD_DRIVER.items(), n):
        env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"] = k, v
    env["GIT_CONFIG_COUNT"] = str(n + len(RECORD_DRIVER))
    return {**env, **extra}


def refuse(prog: str, why: str) -> int:
    """Say on stderr that ``prog`` refused and why (the message names the fix); the exit code for it."""
    print(f"{prog}: refused: {why}", file=sys.stderr)
    return REFUSED
