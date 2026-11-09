"""Verdict carry (drive.toml's `verdict_carry`, `drive.Driver._carry_verdicts`): whether a PR's new head changes nothing a
reviewer saw at an older head with a land verdict, beyond merges of commits the queue trusts, so that verdict holds at the new
head with no reviewer.

The PR's own line is its first-parent line from ``new`` back to ``old`` (``git rev-list --first-parent old..new``); what its
merges bring in from off that line is other work.  ``new`` carries the verdict of ``old`` when ``old`` is an ancestor of ``new``
and each commit of that line is one of:

- landed (`trusted` with no merge: on `checked` or `main`);
- a merge of two parents whose other parent is trusted (`trusted(parent, merge)`: landed, or another PR's current marked head
  with a land verdict and no hold, which the queue's containment lands first, `Queue.entries`), and whose tree is what the queue's
  merge of its parents gives (`queue.merged_tree`: its first parent's merge drivers, each conflicted pin rebuilt) but in
  generated content;
- a commit that changes only generated content.

Generated content is each pin of the pins tool (`queue.pins_at` at ``new``), which `pins.py --stale` holds to the tree, and
each lock's records (`queue.LEAN_LOCK`: `lock_policy`), which a records run regenerates and its PR carries.  The rest
of a lock is policy a person writes and the Lean check keeps as written, so a lock that differs in it, or that the queue's
merge leaves conflicted (the lock's merge driver refuses only over policy), is the PR's own change.  A merge's textual result
can still be wrong where two changes interact; that is the risk `next` takes for every pair of members it merges, and the same
Lean check and full check catch it.

No other PR's verdict vouches for a commit of the PR's own line (a PR stacked on this one was reviewed over it, not for it),
nor for an older head of that PR.

``python -m research.carry OLD NEW [--repo DIR] [--landed REF ...] [--head REF ...]`` prints the verdict as JSON and exits 0
when it carries: each ``--landed`` ref and its ancestors landed (`origin/checked` and `origin/main` when none is given), each
``--head`` another PR's marked head with a land verdict and no hold."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from . import queue as Q

#: what `delta` says of a pair: the first carries the verdict, the rest name why not
CARRIES = ("merge-only",)
KINDS = CARRIES + ("missing", "rebased", "authored", "untrusted-parent", "hand-resolved", "edited-merge", "octopus")


@dataclass(frozen=True)
class Delta:
    kind: str
    why: str = ""
    commits: int = 0
    #: the files whose change the carry left to the checks (generated content), and the trusted parents it merged
    generated: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()

    @property
    def carries(self) -> bool:
        return self.kind in CARRIES

    def as_dict(self) -> dict:
        return {"kind": self.kind, "carries": self.carries, "why": self.why, "commits": self.commits,
                "generated": list(self.generated), "parents": list(self.parents)}


def lock_policy(text: bytes | None) -> dict | None:
    """What of a lock a person writes, which a Lean check keeps as written: all of it but its records, which `check.py
    --regenerate` rewrites from the build (`check.py`'s `regenerated`): `reads`, each guarantee's entry but its `owner`, and
    each assumption's `digest`.  The names of the guarantees and assumptions stay: they decide what is locked and which
    hypotheses count as named.  None for no lock, or one whose JSON isn't an object; a lock that isn't JSON raises ValueError."""
    doc = json.loads(text) if text is not None else None
    if not isinstance(doc, dict):
        return None

    def kept(k: str, keep: Callable[[str], bool]) -> object:
        v = doc[k]
        if not isinstance(v, dict):
            return v
        return {n: {f: x for f, x in r.items() if keep(f)} if isinstance(r, dict) else r for n, r in v.items()}

    out = {k: v for k, v in doc.items() if k != "reads"}
    if "guarantees" in doc:
        out["guarantees"] = kept("guarantees", lambda f: f == "owner")
    if "assumptions" in doc:
        out["assumptions"] = kept("assumptions", lambda f: f != "digest")
    return out


def _ok(repo: Path, *args: str) -> bool:
    return Q._git(repo, *args, check=False).returncode == 0


def _differ(repo: Path, a: str, b: str) -> list[str]:
    return Q._out(repo, "diff", "--name-only", "--no-renames", a, b).splitlines()


def _some(files: Iterable[str]) -> str:
    files = sorted(files)
    return f"{', '.join(files[:5])}{' ...' if len(files) > 5 else ''}"


def _judged(repo: Path, want: str, got: str, files: list[str], pins: set[str]) -> tuple[list[str], list[str]]:
    """(the PR's own changes, the generated ones) among ``files``, those commit ``got`` changes against ``want`` (a commit or
    a tree): a pin is generated, and so is a lock with the same policy (`lock_policy`) on both sides, each JSON."""
    locks = [f for f in files if PurePosixPath(f).name == Q.LEAN_LOCK and f not in pins]
    blobs = Q._blobs(repo, [f"{t}:{f}" for t in (want, got) for f in locks]) if locks else {}

    def generated(f: str) -> bool:
        if f in pins:
            return True
        if f not in locks:
            return False
        try:
            was, now = (lock_policy(blobs.get(f"{t}:{f}")) for t in (want, got))
        except ValueError:
            return False
        return was is not None and was == now

    gen = [f for f in files if generated(f)]
    return [f for f in files if f not in gen], gen


def delta(repo: Path, old: str, new: str, trusted: Callable[[str, str | None], bool]) -> Delta:
    """`Delta` of ``new`` over ``old`` (the module's docstring).  ``trusted(commit, merge)``: whether ``commit`` may come in
    unreviewed, as a parent ``merge`` brings in, or with ``merge`` None as a commit of the PR's own line (landed only)."""
    repo = Path(repo)
    for c in (old, new):
        if not _ok(repo, "cat-file", "-e", f"{c}^{{commit}}"):
            return Delta("missing", f"{c[:12]} isn't in this clone")
    if not _ok(repo, "merge-base", "--is-ancestor", old, new):
        return Delta("rebased", f"{old[:12]} isn't an ancestor of {new[:12]}")
    pins = set(Q.pins_at(repo, new))
    rows = [ln.split() for ln in Q._out(repo, "rev-list", "--first-parent", "--parents", f"{old}..{new}").splitlines() if ln]
    left: set[str] = set()
    merged: list[str] = []
    own: dict[str, set[str]] = {"authored": set(), "hand-resolved": set(), "edited-merge": set()}
    first: dict[str, str] = {}
    for c, *ps in rows:
        if trusted(c, None):
            continue
        if len(ps) > 2:
            return Delta("octopus", f"{c[:12]} merges {len(ps)} parents", len(rows))
        off = [p for p in ps[1:] if not _ok(repo, "merge-base", "--is-ancestor", p, ps[0])]
        if bad := next((p for p in off if not trusted(p, c)), None):
            return Delta("untrusted-parent", f"{c[:12]} merges {bad[:12]}, which is neither landed nor another PR's marked head "
                         "with a land verdict and no hold", len(rows))
        merged += off
        hand: list[str] = []
        if len(ps) == 1:
            want, files, kind = ps[0], _differ(repo, ps[0], c), "authored"
        else:
            tree, conflicted, r = Q.merged_tree(repo, ps[0], ps[1])
            if tree is None:
                if r.returncode != 1:
                    return Delta("hand-resolved", f"{c[:12]}: git merge-tree exited {r.returncode}", len(rows))
                if hand := [f for f in conflicted if f not in pins]:
                    own["hand-resolved"].update(hand)
                    first.setdefault("hand-resolved", f"{c[:12]} resolves conflicts by hand in {_some(hand)}")
                tree = r.stdout.split()[0]
            want, files, kind = tree, [f for f in _differ(repo, tree, c) if f not in hand], "edited-merge"
        mine, gen = _judged(repo, want, c, files, pins)
        if mine:
            own[kind].update(mine)
            first.setdefault(kind, f"{c[:12]} changes {_some(mine)}" if kind == "authored" else
                             f"{c[:12]} differs from the queue's merge of its parents in {_some(mine)}")
        left.update(gen)
    parents = tuple(dict.fromkeys(merged))
    done = dict(commits=len(rows), generated=tuple(sorted(left)), parents=parents)
    if kind := next((k for k in own if own[k]), None):
        return Delta(kind, first[kind], **done)
    why = (f"its {len(rows)} commit(s) since only merge {_some(p[:12] for p in parents)} as the queue would" if parents else
           f"its {len(rows)} commit(s) since change only generated content")
    return Delta("merge-only", why + (f", leaving {_some(left)} to the checks" if left else ""), **done)


def trust(repo: Path, landed: Iterable[str], heads: Iterable[str] = ()) -> Callable[[str, str | None], bool]:
    """``trusted`` for `delta`: a commit that is one of ``landed`` (the tips of `checked` and `main`) or an ancestor of one, and,
    as a parent a merge brings in, also one of ``heads`` (other PRs' current marked heads with a land verdict and no hold)."""
    landed, heads = [t for t in landed if t], {h for h in heads if h}

    def trusted(c: str, merge: str | None = None) -> bool:
        if any(c == t or _ok(repo, "merge-base", "--is-ancestor", c, t) for t in landed):
            return True
        return merge is not None and c in heads

    return trusted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research.carry", description=__doc__.split("\n\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--landed", action="append", default=[], help="a landed ref (default: origin/checked and origin/main)")
    ap.add_argument("--head", action="append", default=[], help="another PR's marked head with a land verdict and no hold")
    ns = ap.parse_args(argv)
    repo = Path(ns.repo)

    def sha(ref: str) -> str:
        return Q._out(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")

    trusted = trust(repo, [sha(t) for t in ns.landed or ["origin/checked", "origin/main"]], [sha(h) for h in ns.head])
    d = delta(repo, sha(ns.old), sha(ns.new), trusted)
    json.dump(d.as_dict(), sys.stdout, indent=1)
    print()
    return 0 if d.carries else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Q.QueueError, subprocess.CalledProcessError) as e:
        sys.exit(f"research.carry: {e}")
