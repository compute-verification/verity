"""`research merge REF [--into BRANCH] [--tool check] [-m MESSAGE] [--dry-run]`: the merge gate.

Run in the checkout of the target branch.  It merges the commit REF names (`git merge --no-ff`) only if the evidence store holds
a passing attempt of the tool (default `check`) for that exact commit: `state` done, rc 0, source commit REF's, tree clean.  With
a usable remote the catalog's attempts are refreshed first, so an attempt recorded on another machine counts.  Refused as well: a
target checkout that is not on BRANCH or has tracked changes, and a REF the target has moved past (the target's tip must be an
ancestor of REF), since then the merged tree is not the one that was checked; merge the target into the branch and check the new
tip.  The merge commit carries a `Check: <attempt id>` trailer.  Exit 0 merged (or would, with --dry-run), 1 refused, 2 usage.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class Refused(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check and r.returncode:
        raise Refused(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r


def passing(store, commit: str, tool: str = "check") -> list[str]:
    """Ids of the attempts of ``tool`` that passed on the clean tree of ``commit``, newest first."""
    rows = store.index.sql("SELECT id FROM attempts WHERE tool = ? AND source_commit = ? AND source_dirty = 0 AND state = 'done' "
                           "AND rc = 0 ORDER BY start_utc DESC", (tool, commit))
    return [r[0] for r in rows]


def gate(repo: Path, ref: str, into: str, store, tool: str = "check", log=print) -> tuple[str, str]:
    """(commit, attempt id) when ``ref`` may be merged into ``into``; `Refused` naming why otherwise."""
    commit = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
    head = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    if head != into:
        raise Refused(f"the checkout is on {head or 'a detached HEAD'}, not {into}")
    if _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise Refused(f"the {into} checkout has uncommitted changes")
    if _git(repo, "merge-base", "--is-ancestor", "HEAD", commit, check=False).returncode:
        raise Refused(f"{into} has moved past {commit[:12]}: merge {into} into the branch, then record `{tool}` on its new tip")
    if store.remote is not None:
        try:
            store.refresh(attempts_only=True)
        except Exception as e:  # noqa: BLE001  (the local catalog still answers; say it may be behind)
            log(f"research merge: refresh from the remote failed ({type(e).__name__}: {e}); using the local catalog")
    ids = passing(store, commit, tool)
    if not ids:
        rows = store.index.sql("SELECT id, state, rc, source_dirty FROM attempts WHERE tool = ? AND source_commit = ?", (tool, commit))
        seen = "; ".join(f"{r[0]} {r[1]} rc={r[2]}{' dirty' if r[3] else ''}" for r in rows) or "none"
        raise Refused(f"no passing `{tool}` attempt for {commit} in the evidence store (attempts of it: {seen}); "
                      f"record one: `uv run python tools/check/check.py --record` on that commit")
    return commit, ids[0]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research merge", description=__doc__.split("\n\n")[1])
    ap.add_argument("ref", help="the commit to merge (a branch, tag or sha)")
    ap.add_argument("--into", default="main", help="the branch checked out here that receives the merge (default main)")
    ap.add_argument("--tool", default="check", help="the tool whose passing attempt the merge requires (default check)")
    ap.add_argument("-m", "--message", default=None, help="the merge commit's message (default `Merge REF (sha)`)")
    ap.add_argument("--dry-run", action="store_true", help="check the gate, merge nothing")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    a = ap.parse_args(argv)
    from .store.local import LocalStore

    repo = Path.cwd()
    try:
        commit, attempt = gate(repo, a.ref, a.into, LocalStore(a.store), a.tool)
    except Refused as e:
        print(f"research merge: refused: {e}", file=sys.stderr)
        return 1
    if a.dry_run:
        print(f"research merge: {a.ref} ({commit[:12]}) may be merged into {a.into}: `{a.tool}` passed in {attempt}")
        return 0
    msg = (a.message or f"Merge {a.ref} ({commit[:8]})") + f"\n\nCheck: {attempt}"
    r = _git(repo, "merge", "--no-ff", "-m", msg, commit, check=False)
    if r.returncode:
        print(f"research merge: git merge failed: {(r.stderr or r.stdout).strip()}", file=sys.stderr)
        return 1
    same = _git(repo, "rev-parse", "HEAD^{tree}").stdout == _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout
    print(f"research merge: merged {a.ref} ({commit[:12]}) into {a.into}; `{a.tool}` passed in {attempt}"
          + ("" if same else "; the merged tree differs from the checked one"))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
