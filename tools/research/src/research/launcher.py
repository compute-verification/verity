"""What a launcher of `research run` trusts of the tree it ships (`--source`).

Unset, `RESEARCH_LAUNCHER_TREE` changes nothing: the shipped tree is the launcher's own checkout and its own trust, and the
launcher runs that tree's copies of the queue's client, suites.py, lean_audit.py and the pool probe's slot.py.

Set to a trusted checkout (the deployed `main` on a machine that launches other people's commits with its own keys), every
program of the repository the launcher runs on its own side, or sends a machine by ssh outside the run, is that tree's
(`code`), and the shipped tree is data: read as files, shipped as a commit, never a Python's working directory or import
root. A step that can't run the trusted copy against the shipped tree's files without importing them (suites.py's
`--list`) doesn't run, and the launch takes its default. Each git call on the shipped tree (`git`, `git_env`) runs no
hook (`core.hooksPath=/dev/null`) and no fsmonitor command (`core.fsmonitor=false`), and reads neither the system's nor
the user's git config (`GIT_CONFIG_NOSYSTEM`, `GIT_CONFIG_GLOBAL`), so the filter drivers the tree's `.gitattributes` can
name are only those its git directory's own config defines. That config is not overridden: the launcher must make the
git directory itself (no commit can write under `.git`). `research`'s re-execution as origin/main's copy is off
(`research.at_main`): the trusted tree is the copy that runs.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import settings

ENV = "RESEARCH_LAUNCHER_TREE"
GIT_OPTS = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false")
GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}


def tree() -> Path | None:
    """The trusted tree `ENV` names, or None when it is unset."""
    v = (settings.get(ENV) or "").strip()
    return Path(v).expanduser().resolve() if v else None


def code(source: Path | None) -> Path | None:
    """The tree whose programs the launcher runs for a launch from ``source``: the trusted tree when there is one."""
    return tree() or source


def git(*args: str) -> list[str]:
    """`git ARGS` on a shipped tree: with `GIT_OPTS` under a trusted tree."""
    return ["git", *(GIT_OPTS if tree() else ()), *args]


def git_env(env: dict[str, str] | None = None) -> dict[str, str] | None:
    """The environment of a git call on a shipped tree: ``env`` (None: this process's) with `GIT_ENV` under a trusted tree."""
    if not tree():
        return env
    return {**(os.environ if env is None else env), **GIT_ENV}
