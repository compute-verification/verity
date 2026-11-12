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

The cache holds one entry per repository (a hash of its git common dir: worktrees of one repository share it, each clone
has its own), and each command that sets one up removes the entries no longer needed (`prune`): never its own, never one
whose `lock` another command holds (it is being made or moved), never one a live process runs from (`inuse`, which every
command started from the tree holds shared, carried through its exec, and which the kernel lets go when the last such
process exits), never one a live process of this user names (a copy of this module from before `inuse` holds nothing:
`named` reads `/proc` for an entry about to be removed), never the `KEEP` most recently picked, and only once `IDLE_S` has
passed since it was last picked (the `lock` file's mtime, dated under the lock by every pick). Picking and judging both
happen under the entry's `lock`, so a pruner either sees a pick (and keeps the entry) or removes it first (and the command
makes it again). An entry that holds nothing, which a pruner that died before its `rmdir` leaves, is dated by its
directory and removed by `rmdir` alone. A removal that fails is one line on stderr, and the command runs on.

`cli.main` asks this module first, which covers a checkout that has it. A checkout from before it can't ask, so on an
agent VM `/usr/local/bin/research` (`tools/agent-guard/research.sh`, installed at boot) runs origin/main's copy of this
file as a script, with `RESEARCH_CHECKOUT` naming the checkout: it runs origin/main's code or the checkout's.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Callable

PATHS = ("tools/research", "infra/ops", "infra/cluster")
OPT_OUT = "RESEARCH_AT_CHECKOUT"
#: what one command gives to setting up origin/main's copy, every git call and the wait for the cache's lock together,
#: before it runs the checkout's own: many times a checkout of this repository on a busy machine, and short enough that
#: the command isn't taken for hung
SETUP_S = 60.0
#: how often a command waiting for the cache's lock tries it again
LOCK_POLL_S = 0.1
#: the entries kept whatever their age, the most recently picked first: the checkouts a machine is likely to run from next,
#: each of which a kept tree spares a whole checkout of the repository
KEEP = 4
#: how long after its last pick an entry may be removed: what keeps the tree of a command started by a copy of this module
#: from before `inuse`, which dates its picks (opening `lock` to write) and holds nothing, and short enough that a machine
#: keeps only the entries of the repositories it ran from lately
IDLE_S = 3600.0
#: an entry's name: the hash `tree` names it by; nothing else under the cache is ever removed
HOME = re.compile(r"[0-9a-f]{12}")
#: the runner, the clocks and the wait of a set-up, which a test stands in for: `clock` bounds a set-up, `wall` dates a pick
run, clock, wall, sleep = subprocess.run, time.monotonic, time.time, time.sleep
#: the `inuse` holds this process took, open and inheritable, so the process it starts carries them
held: list[IO] = []
#: where each live process's working directory, executable, command line and environment are read (`named`): Linux's,
#: which a test stands in for
PROC = Path("/proc")


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
    (`git worktree add` checks out in a child, which outlives it), and `rm` removes the tree. Still under the lock, the
    pick is dated and the tree held in use (`use`), which is what keeps a pruner from removing it."""
    end = clock() + SETUP_S if end is None else end
    common = git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir", end=end).stdout.strip()
    home = cache / hashlib.sha256(common.encode()).hexdigest()[:12]
    wt, at = home / "tree", home / "at"
    checkout = ("checkout", "-q", "--force", "--detach", main)
    with lock(home, end):
        whole = at.read_text().strip() if at.is_file() and (wt / ".git").exists() else ""
        if whole != main:
            at.unlink(missing_ok=True)
            if not whole or git(wt, *checkout, end=end).returncode:
                call(f"`rm -rf {wt}`", ["rm", "-rf", "--", str(wt)], end, check=True)
                git(repo, "worktree", "add", "--force", "--force", "-q", "--no-checkout", "--detach", str(wt), main, end=end,
                    check=True)
                git(wt, *checkout, end=end, check=True)
            at.write_text(main + "\n")
        use(home)
    return wt


def same(f: IO, path: Path) -> bool:
    """Whether the open file ``f`` is the one now at ``path``."""
    try:
        s = path.stat()
    except FileNotFoundError:
        return False
    t = os.fstat(f.fileno())
    return (s.st_dev, s.st_ino) == (t.st_dev, t.st_ino)


def lock(home: Path, end: float | None = None) -> IO | None:
    """`home`'s lock, held: waited for until ``end``, making `home` when it is missing, or with no ``end`` tried once (None
    when another command holds it or `home` is gone). A pruner removes the lock file with its entry, so a lock taken on a
    file no longer at `home/lock` is let go and taken again on the one there now."""
    while True:
        if end is not None:
            home.mkdir(parents=True, exist_ok=True)
        try:
            f = open(home / "lock", "a")
        except FileNotFoundError:
            if end is None:
                return None
            left(end, f"the wait for {home / 'lock'}, which another research command holds")
            continue
        try:
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if end is None:
                        f.close()
                        return None
                    left(end, f"the wait for {home / 'lock'}, which another research command holds")
                    sleep(LOCK_POLL_S)
            if same(f, home / "lock"):
                return f
        except BaseException:
            f.close()
            raise
        f.close()


def use(home: Path) -> None:
    """Under `home`'s lock: hold its tree in use (`held`) and date the pick, both of which a pruner reads under that lock."""
    f = open(home / "inuse", "a")
    try:
        fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        os.set_inheritable(f.fileno(), True)
        t = wall()
        os.utime(home / "lock", (t, t))
    except BaseException:
        f.close()
        raise
    held.append(f)


def picked(home: Path) -> float:
    """When `home`'s tree was last picked (`use`): its lock file's mtime. An entry with no lock file is being made or
    removed, so it counts as picked now, unless it holds nothing (what a pruner that died before its `rmdir` leaves, or
    a command's `mkdir` just before its lock file), dated by its directory's last change."""
    try:
        return (home / "lock").stat().st_mtime
    except FileNotFoundError:
        return emptied_at(home)


def emptied_at(home: Path) -> float:
    """When the entry `home`, with no lock file, last changed if it holds nothing; now if it holds something or is gone."""
    try:
        return home.stat().st_mtime if empty(home) else wall()
    except FileNotFoundError:
        return wall()


def empty(home: Path) -> bool:
    """Whether the directory `home` is there and holds nothing."""
    try:
        with os.scandir(home) as entries:
            return next(entries, None) is None
    except FileNotFoundError:
        return False


def rmdir(home: Path) -> bool:
    """Remove the directory `home` if it is empty; whether it did. Gone already, or holding a lock file a command made for
    it meanwhile, it stays as it is."""
    try:
        home.rmdir()
    except OSError as e:
        if e.errno not in (errno.ENOENT, errno.ENOTEMPTY):
            raise
        return False
    return True


def under(word: str, roots: set[str]) -> bool:
    """Whether a word of a command line or an environment, split at `=` and `:`, holds a path in one of `roots`."""
    return any(part == r or part.startswith(r + "/") for part in re.split("[=:]", word) for r in roots)


def says(proc: Path, name: str, roots: set[str]) -> bool:
    """Whether the process at `proc` names a path in `roots` by `name`: its `cwd` or `exe` link, or a word of its
    `cmdline` or `environ`. What can't be read (the process gone, or not readable by this user) names nothing."""
    try:
        if name in ("cwd", "exe"):
            return under(os.readlink(proc / name), roots)
        data = (proc / name).read_bytes()
        return any(os.fsencode(r) in data for r in roots) and any(under(w, roots) for w in os.fsdecode(data).split("\0"))
    except OSError:
        return False


def mine(proc: Path, uid: int) -> bool:
    try:
        return proc.stat().st_uid == uid
    except OSError:
        return False


def named(tree: Path) -> bool:
    """Whether a live process of this user names `tree` (`says`): what keeps a tree a command started by a copy of this
    module from before `inuse` still runs from, which leaves no other mark. One listing of `PROC`, one stat a process,
    and two links and two reads a process of this user's, up to the first that names it. Where `PROC` isn't there to
    list, every tree counts as named."""
    if not PROC.is_dir():
        return True
    roots, uid = {str(tree), os.path.realpath(tree)}, os.getuid()
    with os.scandir(PROC) as procs:
        return any(p.name.isdigit() and mine(Path(p.path), uid)
                   and any(says(Path(p.path), n, roots) for n in ("cwd", "exe", "cmdline", "environ")) for p in procs)


def due(cache: Path, keep: Path | None) -> list[Path]:
    """The entries under `cache` a pruner may remove, the longest unpicked first: past the `KEEP` most recently picked, and
    unpicked for `IDLE_S`. One stat an entry, and nothing inside its tree."""
    if not cache.is_dir():
        return []
    with os.scandir(cache) as entries:
        homes = sorted(((picked(Path(e.path)), Path(e.path)) for e in entries
                        if HOME.fullmatch(e.name) and e.is_dir(follow_symlinks=False)), reverse=True)
    now = wall()
    return [h for t, h in reversed(homes[KEEP:]) if now - t > IDLE_S and h != keep]


def remove(home: Path, end: float) -> bool:
    """Remove the entry `home` by ``end`` unless, judged under its lock, it is being made or moved, was picked since it was
    listed, a live process holds it in use, or a live process names its tree (`named`, read only here, for an entry
    otherwise removed); whether it was removed. OSError when `rm` fails. An entry that holds nothing has no lock to take,
    and is removed by `rmdir` alone."""
    if empty(home):
        return rmdir(home)
    if (f := lock(home)) is None:
        return False
    with f:
        if wall() - picked(home) <= IDLE_S:
            return False
        with open(home / "inuse", "a") as u:
            try:
                fcntl.flock(u, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            if named(home / "tree"):
                return False
            (home / "at").unlink(missing_ok=True)
            call(f"`rm -rf {home / 'tree'}`", ["rm", "-rf", "--", str(home / "tree")], end, check=True)
            (home / "inuse").unlink(missing_ok=True)
            (home / "lock").unlink(missing_ok=True)
        rmdir(home)
    return True


def prune(cache: Path, keep: Path | None, end: float) -> None:
    """Remove the entries under `cache` that are `due`, the longest unpicked first, until one is removed or ``end`` comes:
    each command makes at most one entry, so one removal a command keeps up. Never raises; what it couldn't remove is one
    line on stderr."""
    failed = []
    try:
        for home in due(cache, keep):
            if clock() >= end:
                break
            try:
                if remove(home, end):
                    break
            except Exception as e:  # noqa: BLE001  a removal that fails never stops the command
                failed.append(" ".join(str(e).split()))
    except Exception as e:  # noqa: BLE001  nor does a cache that can't be listed
        failed.append(" ".join(str(e).split()))
    if failed:
        print(f"research: {len(failed)} old tree(s) of origin/main's research under {cache} could not be removed "
              f"({failed[0]})", file=sys.stderr)


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
    cache = cache or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "research-at-main"
    end = clock() + SETUP_S
    try:
        found = behind(repo, end)
        if found is None:
            return
        main, n = found
        wt = tree(repo, main, cache, end)
    except Exception as e:  # noqa: BLE001  whatever keeps origin/main's from being set up, the checkout's own still runs
        print(f"research: running this checkout's copy: origin/main's could not be set up ({' '.join(str(e).split())})",
              file=sys.stderr)
        prune(cache, None, end)
        return
    print(f"research: this checkout's research code is {n} commit(s) behind origin/main ({main[:10]}, as of its last "
          f"fetch), so origin/main's runs, from {wt}; {OPT_OUT}=1 runs this checkout's", file=sys.stderr)
    prune(cache, wt.parent, end)
    start(wt / "tools/research/src", argv, execve)


def shim(argv: list[str], repo: Path, cache: Path | None = None, execve: Callable = os.execve) -> None:
    """`research ARGV` for the checkout `repo`: origin/main's code when `repo`'s is only behind it, else `repo`'s."""
    reexec(argv, repo, cache, execve)
    start(repo / "tools/research/src", argv, execve)


if __name__ == "__main__":
    shim(sys.argv[1:], Path(os.environ.get("RESEARCH_CHECKOUT") or "."))
