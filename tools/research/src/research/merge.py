"""`research merge REF [--into BRANCH] [--tool check] [-m MESSAGE] [--dry-run]`: the merge gate.
`research merge --train A B C [--on MACHINE --project P] [--command CMD]`: several refs as one checked candidate (`train`).

Run in the checkout of the target branch.  It merges the commit REF names (`git merge --no-ff`) only if the evidence store holds
a passing attempt of the tool (default `check`) for that exact commit: `state` done, rc 0, source commit REF's, tree clean.  With
a usable remote the catalog's attempts are refreshed first, so an attempt recorded on another machine counts.  Refused as well: a
target checkout that is not on BRANCH or has tracked changes, and a REF the target has moved past (the target's tip must be an
ancestor of REF), since then the merged tree is not the one that was checked; merge the target into the branch and check the new
tip.  The merge commit carries a `Check: <attempt id>` trailer.  Exit 0 merged (or would, with --dry-run), 1 refused, 2 usage.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
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


def _record(tree: Path, tool: str, command: list[str], on: str | None, project: str | None, log=print) -> str:
    """Record ``tool`` on the commit checked out in ``tree`` (``research run --tool``; ``--on`` a machine: launched there and
    polled until it ends); returns the run id."""
    from . import runs as R

    rid = R.new_run_id()
    base = [sys.executable, "-m", "research", "run", "--id", rid, "--tool", tool, "--cwd", "source",
            *(["--project", project] if project else [])]
    if on:
        r = subprocess.run([*base, "--on", on, "--source", str(tree), "--", *command], cwd=tree, capture_output=True, text=True)
        if r.returncode:
            raise Refused(f"`research run --on {on}` failed: {(r.stderr or r.stdout).strip()[-400:]}")
        status = R.runs_root(None) / rid / "status.json"
        while True:
            time.sleep(POLL_S)
            subprocess.run([sys.executable, "-m", "research", "fetch", rid], cwd=tree, capture_output=True)
            try:
                if json.loads(status.read_text()).get("state") in ("done", "failed", "cancelled"):
                    break
            except (OSError, ValueError):
                pass
    else:
        subprocess.run([*base, "--", *command], cwd=tree)
    log(f"research merge: `{tool}` recorded as {rid}")
    return rid


POLL_S = 20


def train(repo: Path, refs: list[str], into: str, store, tool: str, command: list[str], on: str | None = None,
          project: str | None = None, log=print, spec: str | None = None) -> tuple[str | None, str | None]:
    """Merge ``refs`` in order onto ``into``'s tip as one candidate (a chain of `--no-ff` merges in a scratch worktree), record
    `tool` on it once and fast-forward ``into`` to it when it passes: the tree that lands is the one checked.  When it fails,
    bisect the chain's prefixes (each an exact commit, checked the same way) to the first ref whose merge makes it fail, and
    fast-forward to the longest prefix that passed.  ``tool`` is the tool's name, ``spec`` how `research run --tool` loads it
    (default the name).  Returns (the commit ``into`` now has, when it moved; the offending ref)."""
    head = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    if head != into or _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise Refused(f"the checkout must be a clean {into}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tree = Path(tempfile.mkdtemp(prefix="research-train-"))
    branch = f"research-train/{stamp}"
    _git(repo, "worktree", "add", "-q", "-b", branch, str(tree), "HEAD")
    try:
        chain = []
        for ref in refs:
            sha = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
            r = _git(tree, "merge", "--no-ff", "-m", f"Merge {ref} ({sha[:8]})\n\nTrain: {stamp}", sha, check=False)
            if r.returncode:
                _git(tree, "merge", "--abort", check=False)
                raise Refused(f"{ref} does not merge onto the train so far: {(r.stderr or r.stdout).strip()[-300:]}")
            chain.append(_git(tree, "rev-parse", "HEAD").stdout.strip())

        def passes(k: int) -> bool:
            _git(tree, "checkout", "-q", "--detach", chain[k - 1])
            _record(tree, spec or tool, command, on, project, log)
            if store.remote is not None:
                store.refresh(attempts_only=True)
            ok = bool(passing(store, chain[k - 1], tool))
            log(f"research merge: train prefix {k}/{len(chain)} ({chain[k - 1][:12]}): `{tool}` {'passed' if ok else 'FAILED'}")
            return ok

        lo, hi, bad = 0, len(chain), None
        if not passes(hi):
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if passes(mid) else (lo, mid)
            bad = refs[hi - 1]
        land = chain[hi - 1] if bad is None else (chain[lo - 1] if lo else None)
        if land is not None:
            gate(repo, land, into, store, tool, log)
            _git(repo, "merge", "--ff-only", "-q", land)
        return land, bad
    finally:
        _git(repo, "worktree", "remove", "--force", str(tree), check=False)
        _git(repo, "branch", "-D", branch, check=False)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research merge", description=__doc__.split("\n\n")[1])
    ap.add_argument("ref", nargs="+", help="the commit to merge (a branch, tag or sha); with --train, several, in merge order")
    ap.add_argument("--train", action="store_true", help="merge the refs as one candidate, record the tool on it, fast-forward on "
                    "a pass, bisect to the offending ref on a failure")
    ap.add_argument("--on", default=None, metavar="MACHINE", help="--train: record the tool on this machine (research run --on)")
    ap.add_argument("--project", default=None, help="--train: the project the run belongs to (required with --on)")
    ap.add_argument("--command", default=None, help="--train: the workload command (default the tool declaration's)")
    ap.add_argument("--into", default="main", help="the branch checked out here that receives the merge (default main)")
    ap.add_argument("--tool", default="check", help="the tool whose passing attempt the merge requires (default check)")
    ap.add_argument("-m", "--message", default=None, help="the merge commit's message (default `Merge REF (sha)`)")
    ap.add_argument("--dry-run", action="store_true", help="check the gate, merge nothing")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    a = ap.parse_args(argv)
    from .store.local import LocalStore

    repo = Path.cwd()
    if a.train:
        from .store import tools_registry

        decl = tools_registry.load(a.tool, repo)
        command = shlex.split(a.command) if a.command else list(decl.command)
        try:
            land, bad = train(repo, a.ref, a.into, LocalStore(a.store), decl.name, command, a.on, a.project, spec=a.tool)
        except Refused as e:
            print(f"research merge: refused: {e}", file=sys.stderr)
            return 1
        moved = f"{a.into} fast-forwarded to {land[:12]}" if land else f"{a.into} unchanged"
        print(f"research merge: train {' '.join(a.ref)}: " + (f"`{decl.name}` passed; {moved}" if bad is None else f"{bad} breaks it; {moved}"))
        return 0 if bad is None else 1
    if len(a.ref) != 1:
        ap.error("several refs need --train")
    a.ref = a.ref[0]
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
