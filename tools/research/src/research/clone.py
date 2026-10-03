"""`research run --cwd clone`: the workload's own checkout of the source root.

`--cwd source` runs a workload in the source root, which on a machine is the shipped tree <root>/src/<sha>/ that every run of the
commit shares: a cancelled duplicate's EXIT trap there removed `PKG/.lake/packages` under a live Lean audit (proofs, 3 Oct 2026).
`--cwd clone` (`clone/SUB`: SUB inside it) runs it in a `git clone --shared` of the source root instead, which borrows the
source's objects and so costs one checkout (about a second for the repository), detached at the source root's HEAD; on a machine
that must be the commit the run was shipped for.  The clone lives outside the run dir, so custody never reads it:
<root>/scratch/<run id>/tree beside src/ on a machine, a temporary directory locally.  The runner removes it once the workload has
exited, however it ended.  A source root that is not a git checkout's top level (a tree the archive stream shipped has no .git)
is refused, never guessed at.
"""
from __future__ import annotations

import errno
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CWD_CLONE = "clone"


class Refused(Exception):
    """`--cwd clone` cannot give this run a clone; the message says why."""


def sub(cwd: str | None) -> str | None:
    """SUB of `--cwd clone/SUB` ('' for `--cwd clone`), None for any other --cwd."""
    if cwd == CWD_CLONE:
        return ""
    if cwd and cwd.startswith(CWD_CLONE + "/"):
        return cwd[len(CWD_CLONE) + 1:].strip("/")
    return None


def home(runs_root: Path, run_id: str, *, machine: bool) -> Path:
    """The directory that holds the run's clone (its `tree`), created: <root>/scratch/<run id>/ on a machine, beside src/ and the
    run dir, a fresh temporary directory locally."""
    if not machine:
        return Path(tempfile.mkdtemp(prefix=f"research-{run_id}-"))
    d = Path(runs_root).parent / "scratch" / run_id
    try:
        d.mkdir(parents=True)
    except FileExistsError:
        raise Refused(f"{d} exists already; a run's clone directory is its own") from None
    return d


def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _why(r: subprocess.CompletedProcess) -> str:
    return (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]


def make(source_root: Path, dest: Path, *, commit: str | None = None, clean: bool = False) -> str:
    """Clone `source_root` into `dest` (absent), detached at its HEAD, and return that commit.  `commit`: the commit the run is for
    (on a machine, the shipped one), which HEAD must be.  `clean`: refuse a source root with uncommitted or untracked files, which a
    clone of HEAD would not carry (locally; a machine's shared tree may hold other runs' files, which is what the clone avoids).
    The clone must come out a clean checkout of that commit.  Refused on any failure, with `dest` removed."""
    src = Path(source_root)
    top = _git(["rev-parse", "--show-toplevel"], src)
    if top.returncode or os.path.realpath(top.stdout.strip()) != os.path.realpath(src):
        where = f"it is inside {top.stdout.strip()}" if top.returncode == 0 else _why(top)
        raise Refused(f"{src} is not the top level of a git checkout ({where}), so there is nothing to clone (a tree the archive "
                      f"stream shipped has no .git); pass --cwd source to run in it as it is")
    head = _git(["rev-parse", "--verify", "-q", "HEAD^{commit}"], src).stdout.strip()
    if not head:
        raise Refused(f"{src} has no commit at HEAD")
    if commit and head != commit:
        raise Refused(f"{src} is at {head}, not {commit}, the commit the run was shipped for")
    if clean:
        st = _git(["status", "--porcelain"], src)
        if st.returncode or st.stdout.strip():
            raise Refused(f"{src} has uncommitted or untracked files, which a clone of HEAD would not carry: commit them, or pass "
                          f"--cwd source to run in the checkout as it is")
    try:
        r = _git(["clone", "-q", "--shared", "--no-checkout", str(src), str(dest)])
        if r.returncode == 0:
            r = _git(["-c", "advice.detachedHead=false", "checkout", "-q", "--detach", head], dest)
        if r.returncode:
            raise Refused(f"git could not clone {src} into {dest}: {_why(r)}")
        got = _git(["rev-parse", "HEAD"], dest).stdout.strip()
        st = _git(["status", "--porcelain"], dest)
        if got != head or st.returncode or st.stdout.strip():
            raise Refused(f"the clone in {dest} is not a clean checkout of {head} (HEAD {got or 'unknown'}"
                          f"{', with changes' if st.stdout.strip() else ''})")
    except Refused:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return head


#: rmtree passes over a clone that is still filling (ENOTEMPTY): a process that has just been SIGKILLed may finish one more write
REMOVE_ATTEMPTS = 3


def remove(path: Path, attempts: int = REMOVE_ATTEMPTS) -> str | None:
    """Remove a run's clone directory: None once it is gone, else why not (the caller reports it; it never fails a run)."""
    why = None
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            why = f"{type(e).__name__}: {e}"
            if e.errno != errno.ENOTEMPTY:
                return why
        if not os.path.lexists(path):
            return None
    return f"{why or 'it kept filling'} after {attempts} attempts"
