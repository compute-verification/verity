"""Run origin/main's `research` when this checkout's copy is only behind it.

An agent VM's checkout is its branch, often days behind main, and its venv runs that branch's `research` (an editable
install), so commands ran old code against main's config. When the checkout's research code (`PATHS`) carries no change
of its own, neither committed since its merge base with origin/main nor in the working tree, and origin/main's differs,
`research` re-executes itself from a detached worktree of origin/main kept in a cache, moved to origin/main whenever that
moves. Both are as of the checkout's last fetch: no network, so a credential prompt can't hang it. A branch that changes
research code runs its own. `RESEARCH_AT_CHECKOUT=1` runs the checkout's copy (and marks the re-executed process, so it
never loops). The working directory is kept, so `research run` still ships the caller's checkout.

`cli.main` asks this module first, which covers a checkout that has it. A checkout from before it can't ask, so on an
agent VM `/usr/local/bin/research` (`tools/agent-guard/research.sh`, installed at boot) runs origin/main's copy of this
file as a script, with `RESEARCH_CHECKOUT` naming the checkout: it runs origin/main's code or the checkout's.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

PATHS = ("tools/research", "infra/ops", "infra/cluster")
OPT_OUT = "RESEARCH_AT_CHECKOUT"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def behind(repo: Path) -> tuple[str, int] | None:
    """`(origin/main's commit, commits behind)` when `repo`'s research code is only behind origin/main's, else None."""
    main = git(repo, "rev-parse", "-q", "--verify", "origin/main^{commit}").stdout.strip()
    base = git(repo, "merge-base", "HEAD", main).stdout.strip() if main else ""
    if not base:
        return None
    own = git(repo, "diff", "--quiet", base, "HEAD", "--", *PATHS).returncode
    dirty = git(repo, "diff", "--quiet", "HEAD", "--", *PATHS).returncode
    moved = git(repo, "diff", "--quiet", base, main, "--", *PATHS).returncode
    if own != 0 or dirty != 0 or moved != 1:
        return None
    n = git(repo, "rev-list", "--count", f"{base}..{main}", "--", *PATHS).stdout.strip()
    return main, int(n or 0)


def tree(repo: Path, main: str, cache: Path) -> Path:
    """The cached worktree of `repo` at `main`, made or moved there under a lock."""
    common = git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
    home = cache / hashlib.sha256(common.encode()).hexdigest()[:12]
    home.mkdir(parents=True, exist_ok=True)
    wt = home / "tree"
    with open(home / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (wt / ".git").exists() and git(wt, "rev-parse", "HEAD").stdout.strip() == main:
            return wt
        if (wt / ".git").exists() and git(wt, "checkout", "-q", "--detach", main).returncode == 0:
            return wt
        shutil.rmtree(wt, ignore_errors=True)
        git(repo, "worktree", "prune")
        r = git(repo, "worktree", "add", "-q", "--detach", str(wt), main)
        if r.returncode:
            raise OSError(f"git worktree add {wt} {main[:12]}: {r.stderr.strip()[-300:]}")
    return wt


def start(src: Path, argv: list[str], execve: Callable = os.execve) -> None:
    """Replace this process with `research ARGV` run from the package under `src`."""
    path = os.pathsep.join(p for p in (str(src), os.environ.get("PYTHONPATH", "")) if p)
    execve(sys.executable, [sys.executable, "-m", "research", *argv], {**os.environ, OPT_OUT: "1", "PYTHONPATH": path})


def reexec(argv: list[str], repo: Path | None = None, cache: Path | None = None, execve: Callable = os.execve) -> None:
    """Re-execute `research ARGV` from origin/main's code when `repo`'s is only behind it (module doc); else return."""
    repo = repo or Path(__file__).resolve().parents[4]
    if os.environ.get(OPT_OUT) or not (repo / ".git").exists():
        return
    try:
        found = behind(repo)
        if found is None:
            return
        main, n = found
        wt = tree(repo, main, cache or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "research-at-main")
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        print(f"research: running this checkout's copy; origin/main's could not be set up: {e}", file=sys.stderr)
        return
    print(f"research: this checkout's research code is {n} commit(s) behind origin/main ({main[:10]}, as of its last "
          f"fetch), so origin/main's runs, from {wt}; {OPT_OUT}=1 runs this checkout's", file=sys.stderr)
    start(wt / "tools/research/src", argv, execve)


def shim(argv: list[str], repo: Path, cache: Path | None = None, execve: Callable = os.execve) -> None:
    """`research ARGV` for the checkout `repo`: origin/main's code when `repo`'s is only behind it, else `repo`'s."""
    reexec(argv, repo, cache, execve)
    start(repo / "tools/research/src", argv, execve)


if __name__ == "__main__":
    shim(sys.argv[1:], Path(os.environ.get("RESEARCH_CHECKOUT") or "."))
