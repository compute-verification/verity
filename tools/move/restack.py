#!/usr/bin/env python3
"""`tools/move/restack.py BRANCH... --move-commit SHA [--script S [--script-arg ARG]...] [--onto REF] [--main REF] [--push-to NEWBRANCH] [--dry-run] [--json]`: carry an open
branch across a generated layout move by running the move's own script on it, instead of rebasing it.

A layout move is one commit M that `python3 tools/move/layout.py` (with its map, `tools/move/layout_map.toml`) wrote by running at
the root of a checkout of its parent B. Because the move is a deterministic, idempotent codemod, a branch is migrated by the
same script rather than by rename detection:

1. merge B into the branch (skipped when the branch already contains B). A conflict here is an ordinary one with the tree
   before the move: the owner merges main first, as without the move.
2. check that merge out in a scratch worktree, run B's `tools/move/layout.py` at its root, and commit what it changed. Files
   the branch added are moved and rewritten too, which rename detection can't do. Where the branch edited `tools/move/`
   itself, B's copy runs and the branch's is put back before the commit.
3. merge `--onto` (default M; the moved main once M has landed, which must contain M), with M as the merge base. M is the
   script run on B, so against it the branch's side holds only the branch's own changes, in moved paths and names, and the
   other side only what landed after the move: what conflicts here is real content. (Against B instead, a line the script
   rewrote next to one the branch edited would conflict.)
4. run the result's own `tools/verity/check/pins.py --regen` at its root, which runs each stale pin's command, and commit what it
   rewrote (`--no-pins` skips it). A result with a pin still stale, its command failed, is `pins-stale`.

A branch that already contains M skips to step 3. A branch that doesn't but shares a commit after B with `--onto` or `--main`
(default origin/main) is `based-after`, and nothing runs: step 3 would bring main's commits since B in a second time and still
report clean (duplicate imports and lock entries). Its owner merges main instead (`git merge origin/main`); the
restack is for branches based at or before B. The branch, its ref and the caller's checkout are never changed: the
commits go to the object store, and the scratch worktree is removed. Each branch's outcome is `clean` (with the result's
id), `up-to-date`, `conflicted` (the step and the files), `script-failed` (the script's exit code and the tail of its
output), `locks-stale` (the last merge's only conflicts were Lean locks, `lean-audit.json`: the result takes `--onto`'s,
and its owner regenerates them with `check.py --update`; `--push-to` pushes it; a pinned `TRUSTED.sha256` in
conflict there likewise takes `--onto`'s and the pins step regenerates it) or `pins-stale` (the result's id, and the tail of `--regen`'s output, which names each stale pin's command). `plain` beside it is what an ordinary `git merge` of the branch into `--onto` gives, for comparison. `--push-to`
pushes a clean result of one branch to a new branch on `--remote`, and refuses a name that already exists there. Exit 0 when
every branch is clean or up to date, 1 otherwise.
"""
from __future__ import annotations

import argparse
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
SCRIPT_DIR = "tools/move"
TRAILER_MOVE, TRAILER_OF = "Restack-move", "Restack-of"
#: the result's own list of pins, whose `--regen` runs every stale pin's command
PINS = "tools/verity/check/pins.py"
#: a Lean package's lock: where both sides changed it, the result takes `--onto`'s and its owner regenerates it
LOCK = "lean-audit.json"
#: a pinned digest list, which the result's own pins step regenerates: where both sides changed it, the result takes
#: `--onto`'s too
PINNED = "TRUSTED.sha256"


class ScriptFailed(Exception):
    pass


def git(*args: str, cwd: str | None = None, input: bytes | None = None, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=cwd, input=input, capture_output=True)
    if p.returncode not in ok:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.decode(errors='replace').strip()}")
    return p


def out(*args: str, cwd: str | None = None) -> str:
    return git(*args, cwd=cwd).stdout.decode().strip()


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


def pinned_uv_env(base: str, tmp: str, cwd: str | None) -> dict[str, str] | None:
    """The environment the move's script runs in: this one, unless the uv on the PATH isn't the version `base`'s
    `tools/check/pod_setup.sh` pins, which layout.py refuses to `uv lock` with. Then a `uv` in `tmp` that runs the pinned
    version through `uv tool run` goes first on the PATH, so the restack doesn't depend on which uv the caller has."""
    found = git("show", f"{base}:tools/check/pod_setup.sh", cwd=cwd, ok=(0, 128)).stdout.decode() or \
        git("show", f"{base}:tools/verity/check/pod_setup.sh", cwd=cwd, ok=(0, 128)).stdout.decode()
    m = re.search(r"^UV_VERSION=\$\{UV_VERSION:-([^}]+)\}", found, re.M)
    real = shutil.which("uv")
    if not m or not real:
        return None
    got = subprocess.run([real, "--version"], capture_output=True, text=True).stdout.split()[1:2]
    if got == [m.group(1)]:
        return None
    shim = os.path.join(tmp, "pinned-uv")
    os.makedirs(shim, exist_ok=True)
    with open(os.path.join(shim, "uv"), "w") as f:
        f.write(f'#!/bin/sh\nexec {shlex.quote(real)} tool run --quiet uv@{shlex.quote(m.group(1))} "$@"\n')
    os.chmod(os.path.join(shim, "uv"), 0o755)
    return {**os.environ, "PATH": shim + os.pathsep + os.environ.get("PATH", "")}


def run_script(merged: str, base: str, python: str, cwd: str | None) -> tuple[str, list[str], list[str]]:
    """Check `merged` out in a scratch worktree, run `base`'s layout script at its root, and return the tree it leaves (every
    change staged, ignored files left out), the `tools/move/` paths whose branch copy was put back, and the tail of the
    script's output."""
    script, mine = blobs_under(base, SCRIPT_DIR, cwd), blobs_under(merged, SCRIPT_DIR, cwd)
    if SCRIPT not in script:
        raise SystemExit(f"the move's base {base[:12]} has no {SCRIPT}")
    restored = sorted(p for p in script.keys() | mine.keys() if script.get(p) != mine.get(p))
    with tempfile.TemporaryDirectory(prefix="restack-") as tmp:
        wt = os.path.join(tmp, os.path.basename(tmp))  # a unique worktree name, so restacks can run side by side
        git("worktree", "add", "--detach", "--quiet", wt, merged, cwd=cwd)
        try:
            if restored:
                # committed, so that the script sees a clean checkout (layout.py refuses a dirty one)
                git("checkout", base, "--", SCRIPT_DIR, cwd=wt)
                overlay = commit_tree(out("write-tree", cwd=wt), [merged], f"restack: {SCRIPT_DIR}/ as at {base}\n", wt)
                git("reset", "-q", "--soft", overlay, cwd=wt)
            p = subprocess.run([python, SCRIPT, *BRANCH_ARGS.get(SCRIPT, ()), *SCRIPT_ARGS], cwd=wt, capture_output=True, env=pinned_uv_env(base, tmp, cwd))
            output = (p.stdout + p.stderr).decode(errors="replace").strip().splitlines()[-15:]
            if p.returncode:
                raise ScriptFailed(f"exit {p.returncode}: " + "\n".join(output))
            git("add", "-A", cwd=wt)
            for path in restored:
                if path in mine:
                    git("checkout", merged, "--", path, cwd=wt)
                else:
                    git("rm", "-q", "-f", "--ignore-unmatch", "--", path, cwd=wt)
            return out("write-tree", cwd=wt), restored, output
        finally:
            git("worktree", "remove", "--force", wt, cwd=cwd, ok=(0, 128))
            git("worktree", "prune", cwd=cwd)


def regen_pins(result: str, python: str, cwd: str | None) -> tuple[str | None, list[str], bool]:
    """Check `result` out in a scratch worktree and run its own `pins.py --regen` at the root: (the tree it leaves, or None when
    it changed nothing, the tail of its output, whether every pin is fresh after). A tree without the list has no pins to run."""
    if git("cat-file", "-e", f"{result}:{PINS}", cwd=cwd, ok=(0, 128)).returncode:
        return None, [], True
    with tempfile.TemporaryDirectory(prefix="restack-pins-") as tmp:
        wt = os.path.join(tmp, os.path.basename(tmp))
        git("worktree", "add", "--detach", "--quiet", wt, result, cwd=cwd)
        try:
            p = subprocess.run([python, PINS, "--regen"], cwd=wt, capture_output=True)
            output = (p.stdout + p.stderr).decode(errors="replace").strip().splitlines()[-15:]
            git("add", "-A", cwd=wt)
            tree = out("write-tree", cwd=wt)
            return (None if tree == tree_of(result, cwd) else tree), output, p.returncode == 0
        finally:
            git("worktree", "remove", "--force", wt, cwd=cwd, ok=(0, 128))
            git("worktree", "prune", cwd=cwd)


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


def check_generated(move: str, base: str, python: str, cwd: str | None) -> None:
    """Refuse a `move` whose tree isn't what `base`'s layout script writes: a later commit (the move branch's head, with its hand
    fixups) taken as the move would make the third merge undo them, while every branch still reported clean."""
    key = (move, base, python, SCRIPT, SCRIPT_ARGS)
    if key not in _GENERATED:
        try:
            _GENERATED[key] = run_script(base, base, python, cwd)[0]
        except ScriptFailed as e:
            raise SystemExit(f"{SCRIPT} fails on the move's base {base[:12]}: {e}") from None
    if _GENERATED[key] != tree_of(move, cwd):
        raise SystemExit(f"--move-commit {move[:12]} isn't what {SCRIPT} writes on {base[:12]}: pass the commit the script "
                         "generated, and a later commit on the move's branch as --onto")


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


def push(result: str, new: str, remote: str, cwd: str | None) -> None:
    if git("ls-remote", "--exit-code", remote, f"refs/heads/{new}", cwd=cwd, ok=(0, 2)).returncode == 0:
        raise SystemExit(f"--push-to {new}: {remote} already has that branch; nothing pushed")
    git("push", remote, f"{result}:refs/heads/{new}", cwd=cwd)


def line(r: dict) -> str:
    plain = f"plain merge {r['plain']['status']}" + (f" ({len(r['plain']['files'])})" if r["plain"]["files"] else "")
    if r["status"] in ("clean", "up-to-date"):
        return f"{r['branch']}: {r['status']} {r['result'][:12]}  [{plain}]"
    if r["status"] in ("script-failed", "pins-stale", "locks-stale", "based-after"):
        at = f" {r['result'][:12]}" if r["status"] not in ("script-failed", "based-after") else ""
        return f"{r['branch']}: {r['status']}{at}  [{plain}]\n    " + r["error"].replace("\n", "\n    ")
    return f"{r['branch']}: conflicted at {r['step']} ({len(r['files'])})  [{plain}]\n" + "".join(f"    {f}\n" for f in r["files"])


def main(argv: list[str] | None = None) -> int:
    global SCRIPT, SCRIPT_ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("branches", nargs="+", metavar="BRANCH")
    ap.add_argument("--move-commit", required=True, metavar="SHA", help="the commit layout.py generated")
    ap.add_argument("--onto", metavar="REF", help="what to merge last: the moved main (default: the move commit)")
    ap.add_argument("--base", metavar="REF", help="the tree the move ran on (default: the move commit's parent)")
    ap.add_argument("--main", default="origin/main", metavar="REF",
                    help="main, when it resolves: a branch that shares a commit after the move's base with it or --onto is "
                         "refused (default origin/main)")
    ap.add_argument("--python", default=sys.executable, help="the interpreter that runs layout.py")
    ap.add_argument("--script", default=SCRIPT, help=f"the move's script, under {SCRIPT_DIR}/ (default {SCRIPT}; the Lean "
                                                    "move's is tools/move/lean.py, the rename move's tools/move/rename.py)")
    ap.add_argument("--script-arg", action="append", default=[], metavar="ARG",
                    help="an argument for the script, repeated in order; one that starts with - as --script-arg=ARG "
                         "(the rename move: --script tools/move/rename.py --script-arg=--map --script-arg tools/move/security_map.toml)")
    ap.add_argument("--push-to", metavar="NEWBRANCH", help="push the clean result to this new branch")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--dry-run", action="store_true", help="report only; never push")
    ap.add_argument("--no-pins", action="store_true", help=f"don't run the result's {PINS} --regen")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not a.script.startswith(SCRIPT_DIR + "/"):
        ap.error(f"--script: a script under {SCRIPT_DIR}/")
    SCRIPT, SCRIPT_ARGS = a.script, tuple(a.script_arg)
    if a.push_to and len(a.branches) != 1:
        ap.error("--push-to takes one branch")
    results = []
    for b in a.branches:
        r = restack(b, a.move_commit, a.onto, a.base, a.python, pins=not a.no_pins, main=a.main)
        results.append(r)
        if not a.json:
            print(line(r), flush=True)
    if a.push_to and not a.dry_run:
        r = results[0]
        if r["status"] not in ("clean", "locks-stale"):
            print(f"--push-to {a.push_to}: {r['branch']} is {r['status']}; nothing pushed", file=sys.stderr)
        else:
            push(r["result"], a.push_to, a.remote, None)
            r["pushed"] = f"{a.remote}/{a.push_to}"
            if not a.json:
                print(f"pushed {r['result'][:12]} to {a.remote} {a.push_to}")
    if a.json:
        print(json.dumps(results, indent=2))
    return 0 if all(r["status"] in ("clean", "up-to-date") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
