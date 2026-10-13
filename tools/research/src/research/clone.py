"""`research run --cwd clone`: the workload's own checkout of the source root.

`--cwd source` runs a workload in the source root, which on a machine is the shipped tree <root>/src/<sha>/ that every run of the
commit shares: a cancelled duplicate's EXIT trap there removed `PKG/.lake/packages` under a live Lean audit (proofs, 3 Oct 2026).
`--cwd clone` (`clone/SUB`: SUB inside it) runs it in a `git clone --shared` of the source root instead, which borrows the
source's objects and so costs one checkout (about a second for the repository), detached at the source root's HEAD; on a machine
that must be the commit the run was shipped for.  The clone lives outside the run dir, so custody never reads it:
<root>/scratch/<run id>/tree beside src/ on a machine, a temporary directory locally.  The runner removes it once the workload has
exited, however it ended; on a machine, one it could not remove goes when the next clone is made there (`sweep`).  A source root that is not a git checkout's top level (a tree the archive stream shipped has no .git)
is refused, never guessed at.
"""
from __future__ import annotations

import errno
import functools
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

#: git's merge drivers for `lean-audit.json`, `.lean` files and the fixture map (`.gitattributes`), set in every run clone and
#: for every git a gate runs (research.gates): a clone's config holds it only after `merge.py --install`, and without it git
#: merges a record line by line (the queue's own clone, a fresh VM's checkout); a tree from before the tools/verity move has it
#: at tools/lean/merge.py.  Here, not in gates, because the suites that read clone.py (vllm's, evaluate's) don't read gates.py
RECORD_DRIVER = {"merge.lean-audit.name": "lean-audit.json: each side's guarantees, reads and dependencies (tools/verity/lean/merge.py)",
                 "merge.lean-audit.driver": 'f=tools/verity/lean/merge.py; [ -f "$f" ] || f=tools/lean/merge.py; python3 "$f" %O %A %B %P',
                 "merge.lean-imports.name": ".lean: each side's imports (tools/verity/lean/merge.py)",
                 "merge.lean-imports.driver": "python3 tools/verity/lean/merge.py --lean %O %A %B %P",
                 "merge.fixture-map.name": "fixtures/artifacts.json: each side's fixtures (tools/verity/check/merge_map.py)",
                 "merge.fixture-map.driver": "python3 tools/verity/check/merge_map.py %O %A %B %P"}


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
    run dir, a fresh temporary directory locally.  On a machine it first removes the clones finished runs left there (`sweep`)."""
    def cannot(e: OSError) -> Refused:
        return Refused(f"could not create a directory for the clone ({type(e).__name__}: {e}); pass --cwd source to run in the "
                       f"source root as it is")

    if not machine:
        try:
            return Path(tempfile.mkdtemp(prefix=f"research-{run_id}-"))
        except OSError as e:
            raise cannot(e) from None
    d = Path(runs_root).parent / "scratch" / run_id
    sweep(Path(runs_root), d.parent)
    try:
        d.mkdir(parents=True)
    except FileExistsError:
        raise Refused(f"{d} exists already; a run's clone directory is its own") from None
    except OSError as e:
        raise cannot(e) from None
    return d


def sweep(runs_root: Path, scratch: Path) -> None:
    """Remove each scratch/<run id> whose run is over (`_over`): a runner that could not remove its clone, or was killed before it
    did, leaves it on the machine's disk, and nothing else ever removes it.  A live run's clone is never touched.  Each removal,
    and each failure, is one line on stderr."""
    try:
        entries = list(os.scandir(scratch))
    except OSError:
        return
    for e in entries:
        try:
            if e.is_symlink() or not e.is_dir(follow_symlinks=False) or not _over(runs_root / e.name):
                continue
        except OSError:
            continue
        why = remove(Path(e.path))
        print(f"research: --cwd clone: removed {e.path}, the clone of finished run {e.name}" if why is None else
              f"research: --cwd clone: could not remove {e.path}, the clone of finished run {e.name} ({why}); remove it by hand",
              file=sys.stderr)


def _over(run_dir: Path) -> bool:
    """The run is over: its runner (launch.json runner_pid) has exited, and it refused (refused.json) or its status.json is final
    with nothing left in the workload's process group (its pgid).  Anything unreadable, or a process that may still be the run's
    (a recycled pid included), is not over."""
    from .runs import TERMINAL
    from .telemetry.procs import live_group, running

    try:
        if running(int(json.loads((run_dir / "launch.json").read_text())["runner_pid"])):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if (run_dir / "refused.json").exists():
        return True
    try:
        st = json.loads((run_dir / "status.json").read_text())
        return st["state"] in TERMINAL and not live_group(int(st["pgid"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False


@functools.lru_cache(maxsize=1)
def _repo_env_vars() -> frozenset[str]:
    """The variables that point git at a repository (GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, ...): inherited from the caller, they
    would make the clone's commands act on that repository instead (a `checkout --detach` detached the source's HEAD)."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return frozenset(subprocess.run(["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, env=clean).stdout.split())


def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    try:
        env = {k: v for k, v in os.environ.items() if k not in _repo_env_vars()}
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    except OSError as e:
        raise Refused(f"could not run git {args[0]} in {cwd or os.getcwd()} ({type(e).__name__}: {e})") from None


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
        st = _git(["status", "--porcelain", "--untracked-files=all"], src)
        if st.returncode or st.stdout.strip():
            raise Refused(f"{src} has uncommitted or untracked files, which a clone of HEAD would not carry: commit them, or pass "
                          f"--cwd source to run in the checkout as it is")
    try:
        r = _git(["clone", "-q", "--shared", "--no-checkout", str(src), str(dest)])
        if r.returncode == 0:
            r = _git(["-c", "advice.detachedHead=false", "checkout", "-q", "--detach", head], dest)
        for k, v in RECORD_DRIVER.items():
            r = r if r.returncode else _git(["config", k, v], dest)
        if r.returncode:
            raise Refused(f"git could not clone {src} into {dest}: {_why(r)}")
        got = _git(["rev-parse", "HEAD"], dest).stdout.strip()
        st = _git(["status", "--porcelain", "--untracked-files=all"], dest)
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
    """Remove a run's clone directory: None once it is gone, else why not (the caller reports it; it never fails a run).  A
    directory the workload left read-only is made writable first (`_Unprotect`); a symlink is never followed."""
    if os.path.islink(path):
        return "it is a symbolic link"
    why = None
    for _ in range(attempts):
        try:
            _rmtree(path, _Unprotect(path))
        except FileNotFoundError:
            pass
        except OSError as e:
            why = f"{type(e).__name__}: {e}"
            if e.errno != errno.ENOTEMPTY:
                return why
        if not os.path.lexists(path):
            return None
    return f"{why or 'it kept filling'} after {attempts} attempts"


def _rmtree(path: Path | str, hook: _Unprotect) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=hook)
    else:
        shutil.rmtree(path, onerror=lambda func, p, exc_info: hook(func, p, exc_info[1]))


class _Unprotect:
    """rmtree's error hook for a tree whose workload left directories read-only (a 0555 build output, a module cache, an
    unreadable 0000 directory): on a PermissionError it adds u+rwx to the entry's directory and to the entry itself when that is a
    directory, inside the tree only and never through a symlink, then removes the entry again.  An entry already gone is fine;
    any other error, or a second PermissionError on one path, is raised."""

    def __init__(self, root: Path | str):
        self.root = os.path.abspath(root)
        self.seen: set[str] = set()

    def __call__(self, func: object, path: str | os.PathLike[str], exc: BaseException) -> None:
        path = os.path.abspath(path)
        if isinstance(exc, FileNotFoundError):
            return
        if not isinstance(exc, PermissionError) or path in self.seen:
            raise exc
        self.seen.add(path)
        for d in (os.path.dirname(path), path):
            if (d == self.root or d.startswith(self.root + os.sep)) and not os.path.islink(d) and os.path.isdir(d):
                os.chmod(d, stat.S_IMODE(os.lstat(d).st_mode) | stat.S_IRWXU)
        if os.path.isdir(path) and not os.path.islink(path):
            _rmtree(path, self)
        else:
            os.unlink(path)
