#!/usr/bin/env python3
"""`tools/verity/check/after_land.py COMMIT`: what `research merge` starts after a landing of main (`after_land` in `tool.py`).

First the full `check` that trails main (Daniel, 7 Oct: a change lands on a targeted check, and one full check of main's newest
tip follows every landing; when it fails, the lander reverts the merge that broke it unless its owner fixes it within 30
minutes): `check.py --record --on vy-nebius-1 --agreement` on COMMIT, in a worktree of its own, unless one of COMMIT or a later
main has been started already. `TRAILING_REF` on origin is the main whose full check was started last, moved with a lease, so
two landers start one; a landing that finds it at COMMIT or past it starts none, since that check covers this change too.

Then the vLLM reference regression's rebuilds (`post_train.py COMMIT`), for a landing whose `Check:` trailer names the full check
it landed on; a landing on a targeted check (`Checked:`) has none, and the trailing check's T0 tier is its regression.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MACHINE = "vy-nebius-1"
#: origin's ref naming the main whose trailing full check was started last
TRAILING_REF = "refs/research/trailing-check"
CHECK_RE = re.compile(r"^Check: r20\d{6}-\d{6}-[0-9a-f]{4}$", re.M)


def git(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def claim(commit: str, remote: str = "origin", cwd: Path = ROOT) -> str | None:
    """Move ``remote``'s `TRAILING_REF` to ``commit`` and return None, or say why no trailing check starts: one of ``commit`` or a
    later main was started already, or another lander moved the ref first."""
    r = git("ls-remote", remote, TRAILING_REF, cwd=cwd)
    if r.returncode:
        return f"{remote}'s {TRAILING_REF} can't be read ({(r.stderr or r.stdout).strip()[-200:]})"
    mark = r.stdout.split()[0] if r.stdout.strip() else ""
    if mark:
        git("fetch", "-q", remote, TRAILING_REF, cwd=cwd)
        if git("merge-base", "--is-ancestor", commit, mark, cwd=cwd).returncode == 0:
            return f"the full check of {mark[:12]} was started already, and it covers {commit[:12]}"
    p = git("push", "-q", "--no-verify", f"--force-with-lease={TRAILING_REF}:{mark}", remote, f"{commit}:{TRAILING_REF}", cwd=cwd)
    if p.returncode:
        return f"{remote}'s {TRAILING_REF} moved while this read it ({(p.stderr or p.stdout).strip()[-200:]}): another lander started one"
    return None


def trailing_check(commit: str, run=subprocess.run, cwd: Path = ROOT) -> int:
    """Start the full check of ``commit`` on `MACHINE` from a worktree of it (``run``: how `check.py --record` is run); its rc."""
    tree = Path(tempfile.mkdtemp(prefix=f"after-land-{commit[:12]}-"))
    try:
        r = git("worktree", "add", "-q", "--detach", str(tree), commit, cwd=cwd)
        if r.returncode:
            print(f"after-land: no worktree of {commit[:12]}: {(r.stderr or r.stdout).strip()}", file=sys.stderr)
            return 1
        # this checkout's environment, not a sync of the worktree's own: `--record --on` only launches the run
        cmd = [sys.executable, "tools/verity/check/check.py", "--record", "--on", MACHINE, "--agreement", "--base", commit]
        print(f"after-land: the full check of {commit[:12]} trails main: {' '.join(cmd)}", flush=True)
        return run(cmd, cwd=tree).returncode
    finally:
        git("worktree", "remove", "--force", str(tree), cwd=cwd)
        shutil.rmtree(tree, ignore_errors=True)


def main(argv: list[str] | None = None, run=subprocess.run, cwd: Path = ROOT, remote: str = "origin") -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    r = git("rev-parse", "--verify", f"{argv[0]}^{{commit}}", cwd=cwd)
    if r.returncode:
        print(f"after-land: no commit {argv[0]}", file=sys.stderr)
        return 2
    commit = r.stdout.strip()
    rc = 0
    if why := claim(commit, remote, cwd):
        print(f"after-land: no trailing check: {why}", flush=True)
    elif trailing_check(commit, run, cwd):
        print(f"after-land: the full check of {commit[:12]} didn't start; start it by hand: `check.py --record --on {MACHINE} "
              f"--agreement` in a checkout of it", file=sys.stderr)
        rc = 1
    if CHECK_RE.search(git("log", "-1", "--format=%B", commit, cwd=cwd).stdout):
        rc |= run([sys.executable, str(cwd / "tools" / "verity" / "check" / "post_train.py"), commit], cwd=cwd).returncode
    else:
        print(f"after-land: {commit[:12]} landed on a targeted check (no `Check:` trailer): no regression rebuilds", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
