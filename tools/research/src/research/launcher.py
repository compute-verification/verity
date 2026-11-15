"""What a launcher of `research run` trusts of the tree it ships (`--source`).

Unset (absent from the environment), `RESEARCH_LAUNCHER_TREE` changes nothing: the shipped tree is the launcher's own
checkout and its own trust, and the launcher runs that tree's copies of the queue's client, suites.py, lean_audit.py and
the pool probe's slot.py.

Set, even to an empty string, it names a trusted checkout (the deployed `main` on a machine that launches other people's
commits with its own keys): every program of the repository the launcher runs on its own side, or sends a machine by ssh
outside the run, is that tree's (`code`), and the shipped tree is data: read as files, shipped as a commit, never a
Python's working directory or import root. A step that can't run the trusted copy against the shipped tree's files without
importing them (suites.py's `--list`) doesn't run, and the launch takes its default. Each git call on the shipped tree
(`git`, `git_env`) runs no hook (`core.hooksPath=/dev/null`) and no fsmonitor command (`core.fsmonitor=false`), and reads
neither the system's nor the user's git config (`GIT_CONFIG_NOSYSTEM`, `GIT_CONFIG_GLOBAL`), so the filter drivers the
tree's `.gitattributes` can name are only those its git directory's own config defines. That config is not overridden: the
launcher must make the git directory itself (no commit can write under `.git`). `research`'s re-execution as origin/main's
copy is off (`research.at_main`): the trusted tree is the copy that runs.

It fails closed: `research run` refuses the launch before anything ships or runs (`refusal`), and every other reader of the
tree raises `Refused`, unless the value is an absolute path (never resolved against the working directory, which is the
--source tree) to an existing directory that is neither the source tree, nor inside it, nor around it (symbolic links
resolved, and the path as written too), and that is a git checkout which is not stale. Stale means either of:
- its HEAD is not its own `origin/main` or an ancestor of it (a branch checked out there), as of its last fetch: the
  launcher fetches nothing;
- a file the launcher runs from it (`RUNS`) is missing, modified, staged or untracked.
How far its `origin/main` is behind the remote's `main` is the deployment's to keep short; nothing here reads the network.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import settings

ENV = "RESEARCH_LAUNCHER_TREE"
GIT_OPTS = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false")
GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
#: what the launcher runs or reads from the trusted tree: research itself; the queue's client, its description and the kinds
#: whose wrappers it places; the pool probe's programs; and lean_audit.py with what it loads by path and the pins it signs.
#: A directory holds all of it. tests/test_launcher_tree.py traces what the launcher opens there, and fails on what this
#: misses.
RUNS = ("tools/research/src/research/launcher.py", "tools/research/src/research",
        "infra/cluster/src/cluster/submit.py", "infra/cluster/src/cluster", "infra/cluster/descriptions/nebius.toml",
        "infra/cluster/kinds", "tools/check/slot.py", "tools/check/lean_slot.py",
        "tools/verity/lean/lean_audit.py", "tools/check/common.py", "tools/verity/lean/module_keys.py",
        "tools/verity/lean/headers.py", "tools/verity/graph/verity_graph/contract.py", "tools/verity/lean/settings.py",
        "tools/verity/lean/lean-deps.json")
CURE = f"set ${ENV} to the absolute path of a clean checkout of origin/main, or unset it to launch from the source tree"
_CHECKED: dict[str, Path] = {}


class Refused(SystemExit):
    """The launch is refused: one line naming the variable, what is wrong with its value and the cure."""

    def __init__(self, value: str, problem: str, cure: str = CURE) -> None:
        super().__init__(f"research: ${ENV}={value!r} {problem}: {cure}")


def _git(at: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *GIT_OPTS, "-C", str(at), *args], capture_output=True, text=True,
                          env={**os.environ, **GIT_ENV})


def _checked(value: str) -> Path:
    if not value.strip():
        raise Refused(value, "is blank, and set: only its absence means no trusted tree")
    if not Path(value).is_absolute():
        raise Refused(value, "is not an absolute path (it would resolve against the working directory, the --source tree)")
    real = Path(value).resolve()
    if not real.is_dir():
        raise Refused(value, "is not an existing directory")
    if missing := [f for f in RUNS if not (real / f).exists()]:
        raise Refused(value, f"lacks what the launcher runs from it: {', '.join(missing)}")
    head = _git(real, "rev-parse", "--verify", "-q", "HEAD^{commit}").stdout.strip()
    main = _git(real, "rev-parse", "--verify", "-q", "refs/remotes/origin/main^{commit}").stdout.strip()
    if not head or not main:
        raise Refused(value, "is not a git checkout with an origin/main")
    if _git(real, "merge-base", "--is-ancestor", head, main).returncode:
        raise Refused(value, f"is stale: its HEAD {head[:12]} is not in its origin/main ({main[:12]})",
                      f"`git -C {value} checkout --detach origin/main` there, or unset ${ENV}")
    st = _git(real, "status", "--porcelain", "--untracked-files=all", "--", *RUNS)
    if st.returncode or st.stdout.strip():
        changed = ", ".join(line[3:] for line in st.stdout.splitlines()[:5]) or st.stderr.strip()[-200:]
        raise Refused(value, f"is stale: what the launcher runs from it is changed or untracked there ({changed})",
                      f"`git -C {value} checkout --detach origin/main` and remove the changes, or unset ${ENV}")
    return real


def tree(source: Path | None = None) -> Path | None:
    """The trusted tree `ENV` names, None when it is absent from the environment; `Refused` when it can't be trusted, or
    isn't apart from ``source``."""
    if not settings.has(ENV):
        return None
    value = settings.get(ENV)
    if value not in _CHECKED:
        _CHECKED[value] = _checked(value)
    if source is not None:
        _apart(_CHECKED[value], Path(source))
    return _CHECKED[value]


def _apart(trusted: Path, source: Path) -> None:
    value = settings.get(ENV)
    src = source.resolve()
    for t in (Path(os.path.abspath(value)), trusted):
        if t == src or src in t.parents or t in src.parents:
            where = "the" if t == src else "inside the" if src in t.parents else "around the"
            raise Refused(value, f"is {where} --source tree {src}: the trusted tree must lie apart from what it launches")


def code(source: Path | None) -> Path | None:
    """The tree whose programs the launcher runs for a launch from ``source``: the trusted tree when there is one."""
    return tree(source) or source


def refusal(source: str | Path | None) -> str | None:
    """Why `research run` refuses to launch from ``source`` (None: the working directory's checkout) under `ENV`, before
    anything ships or runs; None when it is unset or the tree is trusted."""
    try:
        if tree() is None:
            return None
        if source is None:
            top = subprocess.run(git("rev-parse", "--show-toplevel"), capture_output=True, text=True, env=git_env())
            source = top.stdout.strip() or os.getcwd()
        code(Path(source))
    except Refused as e:
        return str(e.code)
    return None


def git(*args: str) -> list[str]:
    """`git ARGS` on a shipped tree: with `GIT_OPTS` under a trusted tree."""
    return ["git", *(GIT_OPTS if tree() else ()), *args]


def git_env(env: dict[str, str] | None = None) -> dict[str, str] | None:
    """The environment of a git call on a shipped tree: ``env`` (None: this process's) with `GIT_ENV` under a trusted tree."""
    if not tree():
        return env
    return {**(os.environ if env is None else env), **GIT_ENV}
