"""Run origin/main's `research` when this checkout's copy is only behind it.

An agent VM's checkout is its branch, often days behind main, and its venv runs that branch's `research` (an editable
install), so commands ran old code against main's config. When the checkout's research code (`PATHS`) carries no change
of its own, neither committed since its merge base with origin/main nor in the working tree, and origin/main's differs,
`research` re-executes itself from a detached worktree of origin/main kept in a cache, moved to origin/main whenever that
moves. Both are as of the checkout's last fetch: no network, so a credential prompt can't hang it. A branch that changes
research code runs its own. `RESEARCH_AT_CHECKOUT=1` runs the checkout's copy (and marks the re-executed process, so it
never loops). The working directory is kept, so `research run` still ships the caller's checkout.

Setting that worktree up never stops a command: all of it (every git call and the wait for the cache's lock) has `SETUP_S`
together, and when that runs out, or git fails, or the cache can't be written, the checkout's own copy runs and one line
on stderr says why. A worktree a killed git left half-made is never run from: the cache names the commit its tree is
wholly at (`tree`), and the next command makes the tree again.

`cli.main` asks this module first, which covers a checkout that has it. A checkout from before it can't ask, so on an
agent VM `/usr/local/bin/research` (`tools/agent-guard/research.sh`, installed at boot) runs origin/main's copy of this
file as a script, with `RESEARCH_CHECKOUT` naming the checkout: it runs origin/main's code or the checkout's.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

PATHS = ("tools/research", "infra/ops", "infra/cluster")
OPT_OUT = "RESEARCH_AT_CHECKOUT"
#: what one command gives to setting up origin/main's copy, every git call and the wait for the cache's lock together,
#: before it runs the checkout's own: many times a checkout of this repository on a busy machine, and short enough that
#: the command isn't taken for hung
SETUP_S = 60.0
#: how often a command waiting for the cache's lock tries it again
LOCK_POLL_S = 0.1
#: the runner, the clock and the wait of a set-up, which a test stands in for
run, clock, sleep = subprocess.run, time.monotonic, time.sleep


def late(at: str) -> TimeoutError:
    return TimeoutError(f"its {SETUP_S:g} s ran out at {at}")


def left(end: float, at: str) -> float:
    """The seconds until ``end``, a `clock` reading; TimeoutError, naming ``at``, when none are."""
    if (s := end - clock()) <= 0:
        raise late(at)
    return s


def call(name: str, argv: list[str], end: float, check: bool = False) -> subprocess.CompletedProcess:
    """``argv`` (``name`` in a message) within what is left until ``end``: TimeoutError when that runs out, killing it, and
    with ``check`` OSError when it fails."""
    try:
        r = run(argv, capture_output=True, text=True, timeout=left(end, name), env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except subprocess.TimeoutExpired:
        raise late(name) from None
    if check and r.returncode:
        raise OSError(f"{name} exited {r.returncode}: {r.stderr.strip()[-300:]}")
    return r


def git(repo: Path, *args: str, end: float, check: bool = False) -> subprocess.CompletedProcess:
    name = " ".join(["git", *(a for a in args[:2] if not a.startswith("-"))])
    return call(f"`{name}`", ["git", "-C", str(repo), *args], end, check)


def behind(repo: Path, end: float | None = None) -> tuple[str, int] | None:
    """`(origin/main's commit, commits behind)` when `repo`'s research code is only behind origin/main's, else None."""
    end = clock() + SETUP_S if end is None else end
    main = git(repo, "rev-parse", "-q", "--verify", "origin/main^{commit}", end=end).stdout.strip()
    base = git(repo, "merge-base", "HEAD", main, end=end).stdout.strip() if main else ""
    if not base:
        return None
    own = git(repo, "diff", "--quiet", base, "HEAD", "--", *PATHS, end=end).returncode
    dirty = git(repo, "diff", "--quiet", "HEAD", "--", *PATHS, end=end).returncode
    moved = git(repo, "diff", "--quiet", base, main, "--", *PATHS, end=end).returncode
    if own != 0 or dirty != 0 or moved != 1:
        return None
    n = git(repo, "rev-list", "--count", f"{base}..{main}", "--", *PATHS, end=end).stdout.strip()
    return main, int(n or 0)


def tree(repo: Path, main: str, cache: Path, end: float | None = None) -> Path:
    """The cached worktree of `repo` at `main`, made or moved there under a lock, by ``end`` (`SETUP_S` from now).

    Beside the tree, `at` names the commit the tree is wholly at: written once git has checked it out, removed before git
    changes it. So a tree whose checkout was killed (its HEAD already at `main`, its files partly there, its entry in the
    repository still locked) has no `at`, and is removed and made again, over that entry; a whole tree is moved, and made
    again when the move fails. The bound stops whatever writes or removes files: the checkout is a git call of its own
    (`git worktree add` checks out in a child, which outlives it), and `rm` removes the tree."""
    end = clock() + SETUP_S if end is None else end
    common = git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir", end=end).stdout.strip()
    home = cache / hashlib.sha256(common.encode()).hexdigest()[:12]
    home.mkdir(parents=True, exist_ok=True)
    wt, at = home / "tree", home / "at"
    checkout = ("checkout", "-q", "--force", "--detach", main)
    with open(home / "lock", "w") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                left(end, f"the wait for {home / 'lock'}, which another research command holds")
                sleep(LOCK_POLL_S)
        whole = at.read_text().strip() if at.is_file() and (wt / ".git").exists() else ""
        if whole == main:
            return wt
        at.unlink(missing_ok=True)
        if not whole or git(wt, *checkout, end=end).returncode:
            call(f"`rm -rf {wt}`", ["rm", "-rf", "--", str(wt)], end, check=True)
            git(repo, "worktree", "add", "--force", "--force", "-q", "--no-checkout", "--detach", str(wt), main, end=end,
                check=True)
            git(wt, *checkout, end=end, check=True)
        at.write_text(main + "\n")
    return wt


def start(src: Path, argv: list[str], execve: Callable = os.execve) -> None:
    """Replace this process with `research ARGV` run from the package under `src`."""
    path = os.pathsep.join(p for p in (str(src), os.environ.get("PYTHONPATH", "")) if p)
    execve(sys.executable, [sys.executable, "-m", "research", *argv], {**os.environ, OPT_OUT: "1", "PYTHONPATH": path})


def reexec(argv: list[str], repo: Path | None = None, cache: Path | None = None, execve: Callable = os.execve) -> None:
    """Re-execute `research ARGV` from origin/main's code when `repo`'s is only behind it (module doc); else return, with
    one line on stderr when origin/main's should run and couldn't be set up. First, as `research`'s own module (not the
    shim's script, whose `research` comes here again), it says once a session when no shell hook runs on this agent's VM
    (`guard_notice`): the other way a VM's branch is behind main without a word."""
    if __package__:
        from .guard_notice import notice

        notice(os.environ)
    repo = repo or Path(__file__).resolve().parents[4]
    if os.environ.get(OPT_OUT) or not (repo / ".git").exists():
        return
    try:
        end = clock() + SETUP_S
        found = behind(repo, end)
        if found is None:
            return
        main, n = found
        wt = tree(repo, main, cache or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "research-at-main", end)
    except Exception as e:  # noqa: BLE001  whatever keeps origin/main's from being set up, the checkout's own still runs
        print(f"research: running this checkout's copy: origin/main's could not be set up ({' '.join(str(e).split())})",
              file=sys.stderr)
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
