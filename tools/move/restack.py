#!/usr/bin/env python3
"""`tools/move/restack.py BRANCH... [--move-commit SHA... [--script S [--script-arg ARG]...]] [--onto REF] [--main REF] [--push-to NEWBRANCH] [--kept] [--dry-run] [--json]`: carry an open
branch across a generated layout move by running the move's own script on it, instead of rebasing it.

With no `--move-commit`, the moves are every commit `--main` (default origin/main) has and the branch lacks whose message
has a move line, oldest first: `Move: MAP` runs `tools/move/rename.py --map MAP`, as drive and `stage.py` write it, and
`Layout-move: MAP` (with `--no-lock` when the move ran so) runs `tools/move/layout.py --map MAP`, as `stage.py` writes it.
The branch crosses each in turn as below, the last merging `--onto` (default `--main`), so one command carries it across a
window's moves; one of them it is based after may be skipped (below). A `--move-commit` without `--script` runs its own
move line's script, else `layout.py`; with `--script`, that script and its `--script-arg`s.

A layout move is one commit M that `python3 tools/move/layout.py` (with its map, `tools/move/layout_map.toml`) wrote by running at
the root of a checkout of its parent B. Because the move is a deterministic, idempotent codemod, a branch is migrated by the
same script rather than by rename detection:

1. merge B into the branch (skipped when the branch already contains B). A conflict here is an ordinary one with the tree
   before the move: the owner merges main first, as without the move.
2. check that merge out in a scratch worktree, run B's `tools/move/layout.py` at its root, and commit what it changed. Files
   the branch added are moved and rewritten too, which rename detection can't do. Where the branch edited `tools/move/`
   itself, B's copy runs and the branch's is put back before the commit. B's script runs with no limit on the text it
   reads (its `TEXT_LIMIT`), so a file the move rewrote is rewritten here too when the branch grew it past the limit.
3. merge `--onto` (default M; the moved main once M has landed, which must contain M), with M as the merge base. M is the
   script run on B, so against it the branch's side holds only the branch's own changes, in moved paths and names, and the
   other side only what landed after the move: what conflicts here is real content. (Against B instead, a line the script
   rewrote next to one the branch edited would conflict.)
4. run the result's own `tools/verity/check/pins.py --regen` at its root, which runs each stale pin's command, and commit what it
   rewrote (`--no-pins` skips it). A result with a pin still stale, its command failed, is `pins-stale`.

Before the steps, M is checked to be what the script writes on B, by running it there (`check_generated`): as long as step 2,
and the same for every branch. With `--kept`, a check that passed is kept as a file under
`~/.cache/verity-check/restack-generated/`, in the check's cache, and one kept there is read: on one machine the first
restack across a move pays for the check and the rest don't. The file's key is everything that decides the check
(`generated_key`); a file that is unreadable, malformed or of another key counts for nothing, and the script runs.

A branch that already contains M skips to step 3. A branch that doesn't but shares a commit after B with `--onto` or `--main`
(default origin/main) is `based-after` M: step 3 would bring main's commits since B in a second time and still report
clean (duplicate imports and lock entries), so M's script never runs on it. A `--move-commit` it is based after refuses
it, and so does the last of the moves found on `--main`: there is no result, and its owner merges main instead (`git merge
origin/main`); the restack is for branches based at or before B. An earlier one of the moves found is skipped where a
later one's base contains it, and named (`skipped`, and a line of the output): that later move's step 1 merges it in on
main's side, as its owner's merge of main would, so M's script is not run on the branch's own lines, and its owner looks
for M's map's old names in what the branch adds. A carry that skipped a move never ends `clean`: it ends `clean-skipped`
(or `locks-stale`), pushes nothing and exits 1, and its output gives the command that crosses the same moves named, with
`--onto` (`confirming`). A named move is never skipped, so that command builds the same result and reports it `clean`:
running it is the owner saying the skip is right. Where no later move's base contains it, no merge of a base brings M in,
and the branch is refused as above. The branch, its ref and the caller's checkout are never changed: the
commits go to the object store, and the scratch worktree is removed. Each branch's outcome is `clean` (with the result's
id), `up-to-date`, `clean-skipped` (the result's id; a move was skipped), `conflicted` (the step and the files), `no-row`
(the branch holds tracked files under a path the move's map retires, and no row moves them: each file, and the fix, a row
for it in the rows file the branch commits beside the map, `NAME.rows.toml`, which `layout.py` reads after the map and
deletes, so no result holds one), `script-failed` (the script's exit code and the tail of its output), `locks-stale` (the
last merge's only conflicts were Lean locks, `lean-audit.json`: the result takes `--onto`'s,
and its owner regenerates them with `check.py --update`; `--push-to` pushes it; a pinned `TRUSTED.sha256` in
conflict there likewise takes `--onto`'s and the pins step regenerates it) or `pins-stale` (the result's id, and the tail of `--regen`'s output, which names each stale pin's command). `plain` beside it is what an ordinary `git merge` of the branch into `--onto` gives, for comparison. `--push-to`
pushes a clean result of one branch to a new branch on `--remote`, never one that skipped a move, and refuses a name that
already exists there. Exit 0 when every branch is clean or up to date, 1 otherwise.

Before any branch, a clone that defines no merge driver for one `.gitattributes` names gets them registered (`DRIVERS`, as
`research`'s own clones have them), since without them a lock or a Lean import list merges as text and stops a branch its
owner's merge would carry; when `--python` can't register them, nothing runs, the refusal names the command, and the exit is 2.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

SCRIPT = "tools/move/layout.py"
#: the arguments the move's script runs with (`--script-arg`; rename.py takes its map as `--map`)
SCRIPT_ARGS: tuple[str, ...] = ()
#: what a script takes on a branch's tree rather than main's: layout.py notes the branch's own references to a directory the map
#: splits instead of failing on them (they're its owner's to check, and the move's commit is the same either way)
BRANCH_ARGS = {"tools/move/layout.py": ("--report-splits",)}
#: layout.py's exit when tracked files under a path its map retires have no row (its `NO_ROW`): its stderr then holds the
#: fix in its first line and each file on a line of its own, indented by two spaces
NO_ROW = 3
#: what a map's rows file is named by, beside the map (layout.py's `ROWS_SUFFIX`): the script deletes the one it read
ROWS_SUFFIX = ".rows.toml"
SCRIPT_DIR = "tools/move"
#: the mover's limit on the text files it reads, in the file every move script reads it from. A crossing runs the script with
#: none: a file under the limit on the move's base was rewritten by the move, and the branch's copy, grown past it, must be
#: too (a Lean lock that crossed with every old path in it). `check_generated` holds the script so run to the move itself
LIMIT_IN, LIMIT, NO_LIMIT = "tools/move/layout.py", re.compile(rb"^TEXT_LIMIT = .*$", re.M), b"TEXT_LIMIT = 1 << 62"
TRAILER_MOVE, TRAILER_OF = "Restack-move", "Restack-of"
#: a move commit's line naming its map, and so its script (`stage.py` writes both kinds, drive's queue `Move:`)
MOVE_LINE = re.compile(r"^(Move|Layout-move): (tools/move/[\w.-]+\.toml)((?: --no-lock)?)$", re.M)
LINE_SCRIPT = {"Move": "tools/move/rename.py", "Layout-move": "tools/move/layout.py"}
#: the result's own list of pins, whose `--regen` runs every stale pin's command
PINS = "tools/verity/check/pins.py"
#: a Lean package's lock: where both sides changed it, the result takes `--onto`'s and its owner regenerates it
LOCK = "lean-audit.json"
#: a pinned digest list, which the result's own pins step regenerates: where both sides changed it, the result takes
#: `--onto`'s too
PINNED = "TRUSTED.sha256"
#: where `--kept` keeps the moves `check_generated` found generated, a file a key (`kept_path`): in the check's cache, at its
#: default place (`tools/verity/check/common.cache_dir`). On a node that is the one directory every run shares: a run's
#: clone, with its `.git`, is removed when the run ends, so a file kept there would never be read
KEPT_DIR = "~/.cache/verity-check/restack-generated"
#: whether `check_generated` reads and writes `KEPT_DIR` (`--kept`). Off, as in drive's jobs and in every test that reaches
#: this file, it reads nothing of the machine's and leaves nothing on it: the suites give each test a `$CHECK_CACHE` of its
#: own, which this file doesn't read
KEEP = False
#: the clone's worktree lock, in its git directory, under the name `research.queue.WORKTREES_LOCK` gives it: a scratch
#: worktree is added and forgotten holding it alone, and every fetch the queue runs holds it shared
WORKTREES_LOCK = "research-worktrees.lock"
#: the options the command that confirms a skip takes from its run, where given (`confirming`): every one that names no move
CARRIED = ("main", "python", "push_to", "remote", "kept", "dry_run", "no_pins", "json")
#: what registers in a clone the merge drivers `.gitattributes` names, as `research`'s own clones have them
#: (`research.clone.set_drivers`): without them a lock, a Lean import list or a test list merges as text, and a branch stops
#: at a merge its owner's clone would make clean
DRIVERS = "import pathlib, research.clone as c; c.set_drivers(pathlib.Path.cwd())"


class ScriptFailed(Exception):
    pass


class NoRow(ScriptFailed):
    """layout.py stopped on tracked files with no row (`NO_ROW`): what it said, and the files it named."""

    def __init__(self, said: str):
        super().__init__(said)
        self.files = [line[2:] for line in said.splitlines()[1:] if line.startswith("  ")]


def git(*args: str, cwd: str | None = None, input: bytes | None = None, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=cwd, input=input, capture_output=True)
    if p.returncode not in ok:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.decode(errors='replace').strip()}")
    return p


@contextlib.contextmanager
def worktrees_locked(cwd: str | None):
    """Hold the clone's worktree lock alone. `git worktree add` writes a placeholder HEAD before the real one, and a fetch in
    the same clone reads every worktree's HEAD: one that read the placeholder fails (`bad object worktrees/restack-…/HEAD`)."""
    with open(os.path.join(cwd or ".", out("rev-parse", "--git-common-dir", cwd=cwd), WORKTREES_LOCK), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def add_worktree(wt: str, commit: str, cwd: str | None) -> None:
    """Check ``commit`` out at ``wt`` as a scratch worktree: registered under the clone's worktree lock with no checkout, so
    its HEAD is the real one before the lock goes, and filled after."""
    with worktrees_locked(cwd):
        git("worktree", "add", "--no-checkout", "--detach", "--quiet", wt, commit, cwd=cwd)
    git("reset", "--hard", "--quiet", cwd=wt)


def drop_worktree(wt: str, cwd: str | None) -> None:
    """Delete scratch worktree ``wt``, then have the clone forget it under its worktree lock. Its result is already read, so
    nothing here fails a restack: a file the script's tools left read-only goes too."""

    def writable(func, path, _exc):
        # Only the worktree's own directories change mode: a file in it may be a symlink or a hard link to one outside,
        # and deleting it needs only its directory's permission.
        os.chmod(os.path.dirname(path) or ".", 0o755)
        if os.path.isdir(path) and not os.path.islink(path):
            os.chmod(path, 0o755)
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                func(path)
            except OSError as e:
                print(f"restack.py: left {path} in the scratch worktree: {e}", file=sys.stderr)

    if os.path.lexists(wt):
        shutil.rmtree(wt, **({"onexc": writable} if sys.version_info >= (3, 12) else {"onerror": writable}))
    with worktrees_locked(cwd):
        git("worktree", "prune", cwd=cwd, ok=tuple(range(256)))


def out(*args: str, cwd: str | None = None) -> str:
    return git(*args, cwd=cwd).stdout.decode().strip()


def missing_drivers(cwd: str | None = None) -> list[str]:
    """The merge drivers the checkout's `.gitattributes` names (`merge=NAME`) that its clone's config defines no
    `merge.NAME.driver` for."""
    attrs, names = os.path.join(out("rev-parse", "--show-toplevel", cwd=cwd), ".gitattributes"), []
    if os.path.isfile(attrs):
        with open(attrs) as f:
            names = sorted(set(re.findall(r"\bmerge=([\w.-]+)", f.read())))
    have = git("config", "--get-regexp", r"^merge\..*\.driver$", cwd=cwd, ok=(0, 1)).stdout.decode().splitlines()
    got = {key.split(".")[1] for key, _, _ in (line.partition(" ") for line in have) if key.count(".") == 2}
    return [n for n in names if n not in got]


def drivers(python: str, cwd: str | None = None) -> str | None:
    """None once the clone defines every merge driver `.gitattributes` names, registering those it lacks with ``python``
    (`DRIVERS`); else why not, with the command that registers them."""
    if not missing_drivers(cwd):
        return None
    r = subprocess.run([python, "-c", DRIVERS], cwd=out("rev-parse", "--show-toplevel", cwd=cwd), capture_output=True, text=True)
    if not (lacking := missing_drivers(cwd)):
        return None
    said = ((r.stderr or r.stdout).strip().splitlines() or [f"exit {r.returncode}"])[-1]
    return (f"this clone defines no merge driver for {', '.join(lacking)} (.gitattributes), so a lock, a Lean import list or a "
            f"test list would merge as text, and {python} couldn't register them ({said}): run `uv run python -c "
            f"\"{DRIVERS}\"` at the repository's root, then this again")


def rev(ref: str, cwd: str | None) -> str:
    return out("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=cwd)


def contains(ancestor: str, descendant: str, cwd: str | None) -> bool:
    return git("merge-base", "--is-ancestor", ancestor, descendant, cwd=cwd, ok=(0, 1)).returncode == 0


def merge_trees(ours: str, theirs: str, cwd: str | None, base: str | None = None) -> tuple[str, list[str]]:
    """`git merge-tree` (ort, with rename detection): (tree, conflicted paths)."""
    pin = [f"--merge-base={base}"] if base else []
    p = git("merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", *pin, ours, theirs, cwd=cwd, ok=(0, 1))
    fields = [s.decode() for s in p.stdout.split(b"\0") if s]
    return fields[0], sorted(set(fields[1:])) if p.returncode else []


def with_blobs_of(tree: str, commit: str, files: list[str], cwd: str | None) -> str:
    """``tree`` with each of ``files`` as ``commit`` has it, in a scratch index."""
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "GIT_INDEX_FILE": os.path.join(d, "index")}
        subprocess.run(["git", "read-tree", tree], cwd=cwd, env=env, check=True)
        for f in files:
            mode, _, oid = out("ls-tree", commit, "--", f, cwd=cwd).split("\t", 1)[0].split(" ")
            subprocess.run(["git", "update-index", "--cacheinfo", f"{mode},{oid},{f}"], cwd=cwd, env=env, check=True)
        return subprocess.run(["git", "write-tree"], cwd=cwd, env=env, check=True, capture_output=True,
                              text=True).stdout.strip()


def commit_tree(tree: str, parents: list[str], message: str, cwd: str | None) -> str:
    args = ["commit-tree", tree]
    for p in parents:
        args += ["-p", p]
    return git(*args, "-F", "-", cwd=cwd, input=message.encode()).stdout.decode().strip()


def tree_of(commit: str, cwd: str | None) -> str:
    return out("rev-parse", f"{commit}^{{tree}}", cwd=cwd)


def blobs_under(commit: str, prefix: str, cwd: str | None) -> dict[str, str]:
    """path -> "mode blob" for every file under `prefix` in `commit`."""
    p = git("ls-tree", "-r", "-z", "--full-tree", commit, "--", prefix, cwd=cwd)
    entries = {}
    for line in (s.decode() for s in p.stdout.split(b"\0") if s):
        meta, path = line.split("\t", 1)
        mode, _, oid = meta.split(" ")
        entries[path] = f"{mode} {oid}"
    return entries


def move_base(move: str, cwd: str | None) -> str:
    parents = out("rev-list", "--parents", "-n", "1", move, cwd=cwd).split()[1:]
    if len(parents) != 1:
        raise SystemExit(f"--move-commit {move[:12]} has {len(parents)} parents; a generated move is one commit on its base "
                         "(pass --base)")
    return parents[0]


def uv_pin(base: str, cwd: str | None) -> tuple[str | None, str | None, str | None]:
    """(the uv version `base`'s `pod_setup.sh` pins, the uv on the PATH, that uv's version), each None where there is none."""
    found = git("show", f"{base}:tools/check/pod_setup.sh", cwd=cwd, ok=(0, 128)).stdout.decode() or \
        git("show", f"{base}:tools/verity/check/pod_setup.sh", cwd=cwd, ok=(0, 128)).stdout.decode()
    m = re.search(r"^UV_VERSION=\$\{UV_VERSION:-([^}]+)\}", found, re.M)
    real = shutil.which("uv")
    got = subprocess.run([real, "--version"], capture_output=True, text=True).stdout.split()[1:2] if real else []
    return (m.group(1) if m else None), real, (got[0] if got else None)


def pinned_uv_env(base: str, tmp: str, cwd: str | None) -> dict[str, str] | None:
    """The environment the move's script runs in: this one, unless the uv on the PATH isn't the version `base`'s
    `tools/check/pod_setup.sh` pins, which layout.py refuses to `uv lock` with. Then a `uv` in `tmp` that runs the pinned
    version through `uv tool run` goes first on the PATH, so the restack doesn't depend on which uv the caller has."""
    pin, real, got = uv_pin(base, cwd)
    if not pin or not real or got == pin:
        return None
    shim = os.path.join(tmp, "pinned-uv")
    os.makedirs(shim, exist_ok=True)
    with open(os.path.join(shim, "uv"), "w") as f:
        f.write(f'#!/bin/sh\nexec {shlex.quote(real)} tool run --quiet uv@{shlex.quote(pin)} "$@"\n')
    os.chmod(os.path.join(shim, "uv"), 0o755)
    return {**os.environ, "PATH": shim + os.pathsep + os.environ.get("PATH", "")}


def swap(path: str, old: bytes, new: bytes) -> None:
    with open(path, "rb") as f:
        text = f.read()
    with open(path, "wb") as f:
        f.write(text.replace(old, new))


def run_script(merged: str, base: str, python: str, cwd: str | None) -> tuple[str, list[str], list[str]]:
    """Check `merged` out in a scratch worktree, run `base`'s layout script at its root with no limit on the text it reads
    (`LIMIT`), and return the tree it leaves (every change staged, ignored files left out), the `tools/move/` paths whose
    branch copy was put back, and the tail of the script's output. A rows file the script deleted is spent, and the branch's
    copy of it is not put back."""
    script, mine = blobs_under(base, SCRIPT_DIR, cwd), blobs_under(merged, SCRIPT_DIR, cwd)
    if SCRIPT not in script:
        raise SystemExit(f"the move's base {base[:12]} has no {SCRIPT}")
    restored = sorted(p for p in script.keys() | mine.keys() if script.get(p) != mine.get(p))
    limit = LIMIT.search(git("show", f"{base}:{LIMIT_IN}", cwd=cwd, ok=(0, 128)).stdout)
    with tempfile.TemporaryDirectory(prefix="restack-", ignore_cleanup_errors=True) as tmp:
        wt = os.path.join(tmp, os.path.basename(tmp))  # a unique worktree name, so restacks can run side by side
        add_worktree(wt, merged, cwd)
        try:
            if restored or limit:
                # committed, so that the script sees a clean checkout (layout.py refuses a dirty one)
                git("checkout", base, "--", SCRIPT_DIR, cwd=wt)
                if limit:
                    swap(os.path.join(wt, LIMIT_IN), limit[0], NO_LIMIT)
                    git("add", "--", LIMIT_IN, cwd=wt)
                overlay = commit_tree(out("write-tree", cwd=wt), [merged], f"restack: {SCRIPT_DIR}/ as at {base}\n", wt)
                git("reset", "-q", "--soft", overlay, cwd=wt)
            p = subprocess.run([python, SCRIPT, *BRANCH_ARGS.get(SCRIPT, ()), *SCRIPT_ARGS], cwd=wt, capture_output=True, env=pinned_uv_env(base, tmp, cwd))
            output = (p.stdout + p.stderr).decode(errors="replace").strip().splitlines()[-15:]
            if p.returncode == NO_ROW and SCRIPT == LINE_SCRIPT["Layout-move"]:
                raise NoRow(p.stderr.decode(errors="replace").strip())
            if p.returncode:
                raise ScriptFailed(f"exit {p.returncode}: " + "\n".join(output))
            if limit and os.path.exists(os.path.join(wt, LIMIT_IN)):
                swap(os.path.join(wt, LIMIT_IN), NO_LIMIT, limit[0])    # the script's file as it left it, with its limit again
            git("add", "-A", cwd=wt)
            restored = [f for f in restored if not f.endswith(ROWS_SUFFIX) or os.path.lexists(os.path.join(wt, f))]
            for path in restored:
                if path in mine:
                    git("checkout", merged, "--", path, cwd=wt)
                else:
                    git("rm", "-q", "-f", "--ignore-unmatch", "--", path, cwd=wt)
            return out("write-tree", cwd=wt), restored, output
        finally:
            drop_worktree(wt, cwd)


def regen_pins(result: str, python: str, cwd: str | None) -> tuple[str | None, list[str], bool]:
    """Check `result` out in a scratch worktree and run its own `pins.py --regen` at the root: (the tree it leaves, or None when
    it changed nothing, the tail of its output, whether every pin is fresh after). A tree without the list has no pins to run."""
    if git("cat-file", "-e", f"{result}:{PINS}", cwd=cwd, ok=(0, 128)).returncode:
        return None, [], True
    with tempfile.TemporaryDirectory(prefix="restack-pins-", ignore_cleanup_errors=True) as tmp:
        wt = os.path.join(tmp, os.path.basename(tmp))
        add_worktree(wt, result, cwd)
        try:
            p = subprocess.run([python, PINS, "--regen"], cwd=wt, capture_output=True)
            output = (p.stdout + p.stderr).decode(errors="replace").strip().splitlines()[-15:]
            git("add", "-A", cwd=wt)
            tree = out("write-tree", cwd=wt)
            return (None if tree == tree_of(result, cwd) else tree), output, p.returncode == 0
        finally:
            drop_worktree(wt, cwd)


def with_pins(r: dict, result: str, python: str, cwd: str | None, pins: bool) -> dict:
    """``r`` as clean at ``result``, after ``result``'s stale pins are regenerated by their commands (`pins-stale` when one
    still is, with the commands' output)."""
    if not pins:
        return {**r, "status": "clean", "result": result}
    tree, output, fresh = regen_pins(result, python, cwd)
    if tree is not None:
        result = commit_tree(tree, [result], f"restack: regenerate stale pins ({PINS} --regen)\n", cwd)
        r["steps"].append({"step": "pins", "commit": result, "output": output})
    if not fresh:
        return {**r, "status": "pins-stale", "result": result, "error": "\n".join(output)}
    return {**r, "status": "clean", "result": result}


_GENERATED: dict[tuple, str] = {}


def generated_key(move: str, base: str, python: str, cwd: str | None) -> dict:
    """What decides the tree the move's script writes on `base`, and so whether `move` is it, as a kept check's key: the two
    commits (the script's own file and its map are `base`'s), the script and every argument it runs with, the interpreter
    by its path and by the version it reports, the uv the script runs under (`pinned_uv_env`), and this file, whose
    `run_script` makes the tree."""
    pin, real, got = uv_pin(base, cwd)
    with open(__file__, "rb") as f:
        me = hashlib.sha256(f.read()).hexdigest()
    version = subprocess.run([python, "-c", "import sys; print(sys.version)"], capture_output=True, text=True, check=True)
    return {"move": move, "base": base, "script": SCRIPT, "args": [*BRANCH_ARGS.get(SCRIPT, ()), *SCRIPT_ARGS],
            "python": python, "interpreter": version.stdout.strip(), "uv": pin if pin and real else got, "restack": me}


def kept_path(key: dict) -> str:
    """The file of the check under `key`, in `KEPT_DIR`."""
    name = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest() + ".json"
    return os.path.join(os.path.expanduser(KEPT_DIR), name)


def kept_generated(key: dict, tree: str) -> bool:
    """Whether the kept check under `key` records `tree`. A file that is unreadable, malformed, of another key or of another
    tree counts for nothing: it is named, and the script runs."""
    path = kept_path(key)
    if not os.path.lexists(path):
        return False
    try:
        with open(path) as f:
            kept = json.load(f)
    except (OSError, ValueError) as e:
        kept = f"unreadable: {e}"
    if kept == {"key": key, "tree": tree}:
        return True
    why = kept if isinstance(kept, str) else "another key or tree"
    print(f"restack.py: {path} is not this check's kept result ({why}): the script runs", file=sys.stderr)
    return False


def keep_generated(key: dict, tree: str) -> None:
    """Keep a check that passed. Where the file can't be written nothing is kept, and the next restack runs the script again."""
    path, tmp = kept_path(key), None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".writing-")
        with os.fdopen(fd, "w") as f:
            json.dump({"key": key, "tree": tree}, f, indent=1)
        os.replace(tmp, path)  # whole or absent: a restack beside this one never reads half a file
    except OSError as e:
        print(f"restack.py: the check of {key['move'][:12]} isn't kept in {os.path.dirname(path)}: {e}", file=sys.stderr)
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def check_generated(move: str, base: str, python: str, cwd: str | None) -> None:
    """Refuse a `move` whose tree isn't what `base`'s layout script writes: a later commit (the move branch's head, with its hand
    fixups) taken as the move would make the third merge undo them, while every branch still reported clean.

    With `KEEP`, a check that passed is kept (`kept_path`) under everything that decides it (`generated_key`), so a later
    restack of another branch across the same move, in another process, doesn't run the script on the base again. Only a
    pass is kept, and a kept file counts only when it records this very key and the move's own tree: anything else runs
    the script."""
    key = (move, base, python, SCRIPT, SCRIPT_ARGS)
    if key not in _GENERATED:
        want, full = tree_of(move, cwd), generated_key(move, base, python, cwd) if KEEP else None
        if full and kept_generated(full, want):
            print(f"restack.py: {move[:12]} is what {SCRIPT} writes on {base[:12]}, as checked before ({kept_path(full)})",
                  file=sys.stderr)
            _GENERATED[key] = want
        else:
            try:
                _GENERATED[key] = run_script(base, base, python, cwd)[0]
            except ScriptFailed as e:
                raise SystemExit(f"{SCRIPT} fails on the move's base {base[:12]}: {e}") from None
            if full and _GENERATED[key] == want:
                keep_generated(full, want)
    if _GENERATED[key] != (want := tree_of(move, cwd)):
        differ = out("diff", "--name-only", "--no-renames", want, _GENERATED[key], cwd=cwd).splitlines()
        raise SystemExit(f"--move-commit {move[:12]} isn't what {SCRIPT} writes on {base[:12]}: pass the commit the script "
                         "generated, and a later commit on the move's branch as --onto (the script runs here with no limit "
                         f"on the text it reads: a text file of {base[:12]} over the script's own limit, which the move left "
                         f"as it was, is one more cause). The {len(differ)} paths that differ"
                         + (", the first 20" if len(differ) > 20 else "") + ":\n  " + "\n  ".join(differ[:20]))


def based_after(head: str, base: str, refs: list[str], cwd: str | None) -> str | None:
    """A commit ``head`` shares with one of ``refs`` that ``base`` doesn't contain, or None: the branch was based (or merged
    main) after the move's base, so the last merge, with the move as its base, would bring those commits in a second time."""
    for ref in refs:
        mbs = git("merge-base", "--all", head, ref, cwd=cwd, ok=(0, 1)).stdout.decode().split()
        if late := next((mb for mb in mbs if not contains(mb, base, cwd)), None):
            return late
    return None


def restack(branch: str, move: str, onto: str | None = None, base: str | None = None, python: str = sys.executable,
            cwd: str | None = None, pins: bool = True, main: str | None = None) -> dict:
    head, move = rev(branch, cwd), rev(move, cwd)
    base = rev(base, cwd) if base else move_base(move, cwd)
    onto = rev(onto, cwd) if onto else move
    if not contains(move, onto, cwd):
        raise SystemExit(f"--onto {onto[:12]} doesn't contain the move {move[:12]}")
    check_generated(move, base, python, cwd)
    r = {"branch": branch, "head": head, "move": move, "base": base, "onto": onto, "steps": []}
    _, plain = merge_trees(head, onto, cwd)
    r["plain"] = {"status": "conflicted" if plain else "clean", "files": plain}

    if contains(onto, head, cwd):
        return {**r, "status": "up-to-date", "result": head}
    main_sha = git("rev-parse", "--verify", "-q", f"{main}^{{commit}}", cwd=cwd, ok=(0, 1)).stdout.decode().strip() if main else ""
    if not contains(move, head, cwd) and (late := based_after(head, base, [onto, *([main_sha] if main_sha else [])], cwd)):
        return {**r, "status": "based-after", "error": (
            f"{branch} has {late[:12]}, which came after the move's base {base[:12]}: restacking it would bring main's commits "
            f"since then in a second time, and report clean (duplicate imports and lock entries). Merge instead: "
            f"`git merge {main or onto[:12]}` once the move has landed, and resolve what that leaves; restack.py is only for "
            f"branches based at or before {base[:12]}")}
    tip, ran = head, False
    if not contains(move, head, cwd):
        if not contains(base, head, cwd):
            tree, conflicts = merge_trees(head, base, cwd)
            if conflicts:
                return {**r, "status": "conflicted", "step": "base", "files": conflicts}
            tip = commit_tree(tree, [head, base], f"Merge {base[:12]}, the layout move's base, into {branch}\n", cwd)
            r["steps"].append({"step": "base", "commit": tip})
        try:
            tree, restored, output = run_script(tip, base, python, cwd)
        except NoRow as e:
            return {**r, "status": "no-row", "step": "script", "files": e.files, "error": str(e)}
        except ScriptFailed as e:
            return {**r, "status": "script-failed", "step": "script", "error": str(e)}
        ran = True
        if tree != tree_of(tip, cwd):
            msg = (f"restack: rerun {shlex.join([SCRIPT, *SCRIPT_ARGS])} from {move[:12]} on {branch}\n\n"
                   f"The layout move's codemod, run at the root of this tree as {move[:12]} was made from {base[:12]}.\n\n"
                   f"{TRAILER_MOVE}: {move}\n{TRAILER_OF}: {head}\n")
            tip = commit_tree(tree, [tip], msg, cwd)
            r["steps"].append({"step": "script", "commit": tip, "restored": restored, "output": output})
    if contains(onto, tip, cwd):
        return with_pins(r, tip, python, cwd, pins)
    tree, conflicts = merge_trees(tip, onto, cwd, base=move if ran else None)
    locks = [f for f in conflicts if f.rsplit("/", 1)[-1] == LOCK]
    pinned = [f for f in conflicts if f.rsplit("/", 1)[-1] == PINNED]
    if len(locks) + len(pinned) < len(conflicts):
        return {**r, "status": "conflicted", "step": "moved", "files": conflicts}
    if conflicts:
        tree = with_blobs_of(tree, onto, conflicts, cwd)
    result = commit_tree(tree, [tip, onto], f"Merge {onto[:12]} (the layout move) into {branch}, restacked\n"
                         + (f"\nThe Lean locks take {onto[:12]}'s side, to be regenerated (`check.py --update`): "
                            f"{', '.join(locks)}\n" if locks else "")
                         + (f"\nThe pinned digests take {onto[:12]}'s side, regenerated by the pins step: "
                            f"{', '.join(pinned)}\n" if pinned else ""), cwd)
    r["steps"].append({"step": "moved", "commit": result})
    r = with_pins(r, result, python, cwd, pins)
    if locks and r["status"] == "clean":
        return {**r, "status": "locks-stale", "files": locks,
                "error": f"the Lean locks {', '.join(locks)} take {onto[:12]}'s side: regenerate them on the result "
                         "(tools/verity/lean/check.py --update) before marking it"}
    return r


def move_line(commit: str, cwd: str | None) -> tuple[str, tuple[str, ...]] | None:
    """The script and arguments ``commit``'s move line names, or None when it has none."""
    m = MOVE_LINE.search(out("log", "-1", "--format=%B", commit, cwd=cwd))
    return (LINE_SCRIPT[m[1]], ("--map", m[2], *m[3].split())) if m else None


def moves_on(ref: str, head: str, cwd: str | None) -> list[tuple[str, str, tuple[str, ...]]]:
    """The moves ``ref`` has and ``head`` lacks, oldest first: (commit, script, arguments) of each commit with a move line."""
    log = out("log", "--topo-order", "--reverse", "--no-merges", "--format=%H%x00%B%x1e", ref, f"^{head}", cwd=cwd)
    found = []
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        if m := MOVE_LINE.search(body):
            found.append((commit, LINE_SCRIPT[m[1]], ("--map", m[2], *m[3].split())))
    return found


def carry(branch: str, moves: list[tuple[str, str, tuple[str, ...]]], onto: str | None = None, base: str | None = None,
          python: str = sys.executable, cwd: str | None = None, pins: bool = True, main: str | None = None,
          found: bool = False) -> dict:
    """``branch`` restacked across each of ``moves`` in order, each with its own script and arguments, the last merging
    ``onto``: the last step's outcome (`restack`'s), or the first that isn't clean, with `crossed`, the moves it crossed.
    One that skipped a move is `clean-skipped` where it would be `clean` or `up-to-date`.

    With ``found`` (the moves are `moves_on`'s, none of them named), a move the branch is `based-after` is skipped, and
    named in `skipped`, where a later move's base contains it: crossing that later move starts by merging its base, which
    is the merge of main the refusal asks of the owner, up to that base. A skipped move is then in the base of a move that
    is crossed, or the carry stops before a result: the later move is crossed, or skipped under the same rule, or stops it.
    The last move has no later one, so it still refuses."""
    global SCRIPT, SCRIPT_ARGS
    head, tip, crossed, skipped, locks = rev(branch, cwd), branch, [], [], []
    if not moves:
        return {"branch": branch, "head": head, "crossed": [], "skipped": [], "status": "no-moves",
                "plain": {"status": "clean", "files": []}, "error": f"no move to cross: merge {main or onto} as usual"}
    for i, (move, script, args) in enumerate(moves):
        SCRIPT, SCRIPT_ARGS = script, tuple(args)
        last = i == len(moves) - 1
        r = restack(tip, move, onto if last else None, base if len(moves) == 1 else None, python, cwd,
                    pins=pins and last, main=main)
        r = {**r, "branch": branch, "head": head, "crossed": crossed, "skipped": skipped}
        if found and r["status"] == "based-after" and any(contains(move, move_base(m, cwd), cwd) for m, *_ in moves[i + 1:]):
            skipped = [*skipped, {"move": r["move"], "script": script, "args": list(args)}]
            continue
        if r["status"] not in ("clean", "up-to-date", "locks-stale"):
            return r
        crossed = [*crossed, move]
        locks += [f for f in r.get("files", []) if r["status"] == "locks-stale" and f not in locks]
        if last:
            break
        tip = r["result"]
    if locks and r["status"] != "locks-stale":
        r = {**r, "status": "locks-stale", "files": locks,
             "error": f"the Lean locks {', '.join(locks)} took the moves' side: regenerate them on the result "
                      "(tools/verity/lean/check.py --update) before marking it"}
    if skipped and r["status"] in ("clean", "up-to-date"):
        r = {**r, "status": "clean-skipped"}
    return {**r, "crossed": crossed, "skipped": skipped}


def confirming(r: dict, a: argparse.Namespace, ap: argparse.ArgumentParser) -> list[str]:
    """The command that carries ``r``'s branch across the moves ``r`` crossed, each named, onto what ``r`` merged last, with
    this run's other options as given. A named move is never skipped, so it does what ``r``'s carry did after its skips and
    reports it with none: running it is the owner saying the skip was right."""
    named = [x for m in r["crossed"] for x in ("--move-commit", m)]
    rest: list[str] = []
    for k in CARRIED:
        if (v := getattr(a, k)) != ap.get_default(k):
            rest += [f"--{k.replace('_', '-')}", *([] if v is True else [v])]
    return ["python3", f"{SCRIPT_DIR}/restack.py", r["branch"], *named, "--onto", r["onto"], *rest]


def push(result: str, new: str, remote: str, cwd: str | None) -> None:
    if git("ls-remote", "--exit-code", remote, f"refs/heads/{new}", cwd=cwd, ok=(0, 2)).returncode == 0:
        raise SystemExit(f"--push-to {new}: {remote} already has that branch; nothing pushed")
    git("push", remote, f"{result}:refs/heads/{new}", cwd=cwd)


def line(r: dict) -> str:
    plain = f"plain merge {r['plain']['status']}" + (f" ({len(r['plain']['files'])})" if r["plain"]["files"] else "")
    if len(r.get("crossed", [])) > 1 or r["status"] not in ("clean", "up-to-date", "no-moves") and r.get("crossed"):
        plain += (f"; crossed {len(r['crossed'])} move{'s' if len(r['crossed']) != 1 else ''}: "
                  f"{', '.join(c[:12] for c in r['crossed'])}")
        if r["status"] not in ("clean", "clean-skipped", "up-to-date", "locks-stale"):
            plain += f"; stopped at the move {r['move'][:12]}"
    skipped = "".join(
        f"\n    skipped the move {s['move'][:12]} ({shlex.join([s['script'], *s['args']])}): {r['branch']} is based after its "
        "base, so its script was NOT run on the branch's own lines, and the base merge of a later move carries it. Look for "
        "that map's old names in what the branch adds." for s in r.get("skipped", []))
    if "confirm" in r:
        skipped += ("\n    with a move skipped nothing is pushed and the exit is 1. If the skip is right, say so by crossing "
                    f"with the moves named, which skips none:\n    {shlex.join(r['confirm'])}")
    if r["status"] in ("clean", "clean-skipped", "up-to-date"):
        return f"{r['branch']}: {r['status']} {r['result'][:12]}  [{plain}]" + skipped
    if r["status"] in ("script-failed", "no-row", "pins-stale", "locks-stale", "based-after", "no-moves"):
        at = f" {r['result'][:12]}" if r["status"] not in ("script-failed", "no-row", "based-after", "no-moves") else ""
        return f"{r['branch']}: {r['status']}{at}  [{plain}]\n    " + r["error"].replace("\n", "\n    ") + skipped
    return (f"{r['branch']}: conflicted at {r['step']} ({len(r['files'])})  [{plain}]\n"
            + "".join(f"    {f}\n" for f in r["files"]) + skipped.lstrip("\n"))


def main(argv: list[str] | None = None) -> int:
    global SCRIPT, SCRIPT_ARGS, KEEP
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("branches", nargs="+", metavar="BRANCH")
    ap.add_argument("--move-commit", action="append", default=[], metavar="SHA",
                    help="a move's commit, repeated in order (default: each move --main has and the branch lacks, by its "
                         "move line)")
    ap.add_argument("--onto", metavar="REF",
                    help="what to merge last: the moved main (default: the last --move-commit, else --main)")
    ap.add_argument("--base", metavar="REF", help="the tree one --move-commit ran on (default: its parent)")
    ap.add_argument("--main", default="origin/main", metavar="REF",
                    help="main, when it resolves: a branch that shares a commit after a move's base with it or --onto is based "
                         "after that move, and refused; where the move is one found here, and a later one's base contains "
                         "it, it is skipped and named instead, and the result is clean-skipped (default origin/main)")
    ap.add_argument("--python", default=sys.executable, help="the interpreter that runs the move's script")
    ap.add_argument("--script", help=f"one --move-commit's script, under {SCRIPT_DIR}/ (default: its move line's, else "
                                     f"{SCRIPT}; a rename move's is tools/move/rename.py)")
    ap.add_argument("--script-arg", action="append", default=[], metavar="ARG",
                    help="an argument for the script, repeated in order; one that starts with - as --script-arg=ARG "
                         "(a rename move: --script tools/move/rename.py --script-arg=--map --script-arg tools/move/MAP.toml)")
    ap.add_argument("--push-to", metavar="NEWBRANCH",
                    help="push the clean result to this new branch; never one that skipped a move")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--kept", action="store_true",
                    help=f"keep each move's check, once it passed, under {KEPT_DIR}/, and read one kept there: on one "
                         "machine only the first restack across a move runs the script on the move's base")
    ap.add_argument("--dry-run", action="store_true", help="report only; never push")
    ap.add_argument("--no-pins", action="store_true", help=f"don't run the result's {PINS} --regen")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.script is not None and not a.script.startswith(SCRIPT_DIR + "/"):
        ap.error(f"--script: a script under {SCRIPT_DIR}/")
    if (a.script or a.script_arg or a.base) and len(a.move_commit) != 1:
        ap.error("--script, --script-arg and --base are one --move-commit's")
    if a.push_to and len(a.branches) != 1:
        ap.error("--push-to takes one branch")
    KEEP = a.kept
    if why := drivers(a.python):
        print(f"restack: {why}", file=sys.stderr)
        return 2
    named = [(m, a.script or SCRIPT, tuple(a.script_arg)) if a.script or a.script_arg
             else (m, *(move_line(m, None) or (SCRIPT, ()))) for m in a.move_commit]
    results = []
    for b in a.branches:
        moves = named or moves_on(a.main, b, None)
        r = carry(b, moves, a.onto or (None if named else a.main), a.base, a.python, pins=not a.no_pins, main=a.main,
                  found=not named)
        if r["skipped"] and r["status"] in ("clean-skipped", "locks-stale"):
            r["confirm"] = confirming(r, a, ap)
        results.append(r)
        if not a.json:
            print(line(r), flush=True)
    if a.push_to and not a.dry_run:
        r = results[0]
        if r["status"] not in ("clean", "clean-skipped", "locks-stale"):
            print(f"--push-to {a.push_to}: {r['branch']} is {r['status']}; nothing pushed", file=sys.stderr)
        elif r["skipped"]:
            print(f"--push-to {a.push_to}: {r['branch']} skipped the move{'s' if len(r['skipped']) != 1 else ''} "
                  f"{', '.join(s['move'][:12] for s in r['skipped'])}, whose script did not run on its own lines; nothing "
                  f"pushed. If the skip is right, cross with the moves named: {shlex.join(r['confirm'])}", file=sys.stderr)
        else:
            push(r["result"], a.push_to, a.remote, None)
            r["pushed"] = f"{a.remote}/{a.push_to}"
            if not a.json:
                print(f"pushed {r['result'][:12]} to {a.remote} {a.push_to}")
    if a.json:
        print(json.dumps(results, indent=2))
    return 0 if all(r["status"] in ("clean", "up-to-date", "no-moves") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
