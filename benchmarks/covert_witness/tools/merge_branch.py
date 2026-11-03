#!/usr/bin/env python3
"""The integrator's merge of a branch into covert-witness's head, where te_core changes only as the toolsmith's VERSION.

    python3 benchmarks/covert_witness/tools/merge_branch.py BRANCH [--at REV] [-m WHAT]   # merge and commit
    python3 benchmarks/covert_witness/tools/merge_branch.py BRANCH [--at REV] --check     # only say whether it may

It fetches BRANCH from origin and merges REV (BRANCH's head, or a commit on it) into the checked-out branch, --no-ff.
Before touching the tree it refuses:

- any branch but the toolsmith's (`TOOLSMITH`) whose commits since the merge base change a file under te_core/: a
  te_core change reaches covert-witness only as a VERSION of the toolsmith's, so the toolsmith carries it;
- a toolsmith's commit that moves products.CAPTURE off `FROZEN` or moves any `*_FROM` gate in te_core's code: the
  capture is frozen through the finals, and a moved gate makes a 4.12.9 trace read stale;
- a toolsmith's commit that changes the capture's code (te_core/capture/, its tests and docs aside) without
  `--capture-ruled WHY`: a trace made after the merge could read differently from one made before under the same
  products.CAPTURE, so it merges only once the coordinator has ruled the change may keep the label (WHY, which the
  merge's commit message carries).

A merge of the toolsmith's leaves te_core exactly as REV has it (a VERSION is the toolsmith's whole te_core, so a
te_core conflict resolves to theirs), and any other merge leaves it as it was. A conflict outside te_core stops before
the commit, for you to resolve and commit. The refusal names the files or the gates, and what to do instead. A gate
check sees constants only: read a VERSION's diff for a new path to 'retrace' as well.

Stdlib only; git on the PATH.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TE_CORE = "benchmarks/covert_witness/te_core"
TOOLSMITH = "cursor/te-core-dependence-4207"
#: the capture version every trace through the finals has; a VERSION that moves it is refused
FROZEN = "4.12.9"
CAPTURE = f"{TE_CORE}/capture/products.py:CAPTURE"
GATES = r"^(CAPTURE|CAPTURE_FINAL|[A-Z_]+_FROM) *= "


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, encoding="utf-8")
    if check and r.returncode:
        raise SystemExit(f"merge_branch: git {' '.join(args[:3])} failed ({r.returncode}): {r.stderr.strip()[-1000:]}")
    return r


def gates(repo: Path, rev: str) -> dict[str, str]:
    """te_core's capture version and version gates at `rev`, tests aside: `path:NAME` to the value as written."""
    r = git(repo, "grep", "-E", GATES, rev, "--", TE_CORE, ":!*/tests/*", check=False)
    if r.returncode not in (0, 1):
        raise SystemExit(f"merge_branch: git grep failed at {rev}: {r.stderr.strip()[-1000:]}")
    out = {}
    for line in r.stdout.splitlines():
        path, text = line.removeprefix(f"{rev}:").split(":", 1)
        name, value = text.split("=", 1)
        out[f"{path}:{name.strip()}"] = value.split("#", 1)[0].strip()
    return out


def refusal(repo: Path, branch: str, rev: str, ruled: str = "") -> str | None:
    """Why `rev` on `branch` may not merge into HEAD, or None; `ruled` is the coordinator's ruling that a change to the
    capture's code may keep products.CAPTURE."""
    base = git(repo, "merge-base", "HEAD", rev).stdout.strip()
    changed = git(repo, "diff", "--name-only", base, rev, "--", TE_CORE).stdout.split()
    if not changed:
        return None
    at = f"{branch} at {rev[:10]}"
    if branch != TOOLSMITH:
        shown = ", ".join(p.removeprefix(f"{TE_CORE}/") for p in changed[:6]) + (" ..." if len(changed) > 6 else "")
        return (f"{at} changes te_core ({shown}). te_core changes reach covert-witness only as a VERSION of the "
                f"toolsmith's ({TOOLSMITH}): ask the toolsmith to carry them, then merge that VERSION; where they are "
                f"already the toolsmith's, merge the toolsmith's branch first")
    now, then = gates(repo, "HEAD"), gates(repo, rev)
    if then.get(CAPTURE) != f'"{FROZEN}"':
        return (f"{at} moves products.CAPTURE to {then.get(CAPTURE, 'nothing')}; the capture is frozen at {FROZEN} "
                f"through the finals, so post why it is refused and ask the toolsmith for a VERSION that keeps it")
    moved = sorted(k for k in now.keys() | then.keys() if now.get(k) != then.get(k))
    if moved:
        shown = "; ".join(f"{k.removeprefix(f'{TE_CORE}/')} {now.get(k, 'absent')} -> {then.get(k, 'absent')}"
                          for k in moved)
        return (f"{at} moves te_core's version gates ({shown}), so a 4.12.9 trace would read stale; the capture is "
                f"frozen at {FROZEN} through the finals, so post why it is refused and ask the toolsmith for a VERSION "
                f"that keeps them")
    code = [p.removeprefix(f"{TE_CORE}/") for p in changed
            if p.startswith(f"{TE_CORE}/capture/") and "/tests/" not in p and not p.endswith(".md")]
    if code and not ruled:
        return (f"{at} changes the capture's code ({', '.join(code)}) under products.CAPTURE {FROZEN}, so a trace "
                f"made after the merge could read differently from one made before under the same label; the capture "
                f"is frozen through the finals, so post why it is refused, or, once the coordinator rules the change "
                f"may keep the label, merge with --capture-ruled WHY")
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("branch", help="the branch on origin to merge, e.g. cursor/te-core-dependence-4207")
    ap.add_argument("--at", help="a commit on BRANCH to merge (default: its head)")
    ap.add_argument("-m", dest="what", default="", help="what the merge brings, for the commit message")
    ap.add_argument("--check", action="store_true", help="say whether it may merge, and change nothing")
    ap.add_argument("--capture-ruled", metavar="WHY", default="",
                    help="the coordinator's ruling that the VERSION's change to the capture's code keeps its label")
    a = ap.parse_args(argv)
    repo = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.strip())
    if git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise SystemExit("merge_branch: the worktree has uncommitted changes (or a merge in progress); commit or "
                         "drop them first")
    git(repo, "fetch", "-q", "origin", a.branch)
    tip = f"refs/remotes/origin/{a.branch}"
    rev = git(repo, "rev-parse", "--verify", f"{a.at or tip}^{{commit}}").stdout.strip()
    if git(repo, "merge-base", "--is-ancestor", rev, tip, check=False).returncode:
        raise SystemExit(f"merge_branch: {a.at} is not on {a.branch}")
    if not git(repo, "merge-base", "--is-ancestor", rev, "HEAD", check=False).returncode:
        print(f"merge_branch: {a.branch} at {rev[:10]} is already merged")
        return 0
    why = refusal(repo, a.branch, rev, a.capture_ruled)
    if why:
        print(f"merge_branch: refused: {why}", file=sys.stderr)
        return 1
    if a.check:
        print(f"merge_branch: {a.branch} at {rev[:10]} may merge")
        return 0
    r = git(repo, "merge", "--no-ff", "--no-commit", rev, check=False)
    if git(repo, "rev-parse", "-q", "--verify", "MERGE_HEAD", check=False).returncode:
        raise SystemExit(f"merge_branch: git merge {rev[:10]} failed ({r.returncode}): {r.stderr.strip()[-1000:]}")
    if a.branch == TOOLSMITH:
        git(repo, "restore", f"--source={rev}", "--staged", "--worktree", "--", TE_CORE)
    want = rev if a.branch == TOOLSMITH else "HEAD"
    if git(repo, "diff", "--cached", "--quiet", want, "--", TE_CORE, check=False).returncode:
        git(repo, "merge", "--abort")
        raise SystemExit(f"merge_branch: te_core after the merge is not {want}'s; merge aborted")
    conflicts = git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    if conflicts:
        print(f"merge_branch: conflicts outside te_core ({', '.join(conflicts)}): resolve them, `git add` them and "
              f"`git commit`; te_core is already right", file=sys.stderr)
        return 1
    notes = [n for n in (a.what, a.capture_ruled and f"its capture code change ruled: {a.capture_ruled}") if n]
    git(repo, "commit", "-q", "-m", f"Merge {a.branch} at {rev[:10]}" + (f" ({'; '.join(notes)})" if notes else ""))
    merged = git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    print(f"merge_branch: merged {a.branch} at {rev[:10]} as {merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
