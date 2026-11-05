"""Verdict carry-forward: whether a PR's new head changes nothing a reviewer saw at an older head that has a land verdict, beyond
merges of commits the queue already trusts, so the old verdict holds at the new head with no reviewer.

The PR's own commits are its first-parent line from ``new`` back to ``old`` (``git rev-list --first-parent old..new``); what
they merge in from off that line is other work.  A new head ``new`` carries the verdict of ``old`` when ``old`` is an ancestor
of ``new`` and each commit of that line is one of:

- landed (`trusted` with no merge: an ancestor of the landing branch);
- a merge whose other parents are each trusted (`trusted(parent, merge)`: landed, or in another PR's marked head with a land
  verdict that doesn't itself contain ``merge``, so that PR's own verdict covers what it brings and drive lands it first), and
  whose tree is what the queue's own merge of its parents gives (`queue.merged_tree`: the merge drivers of its first parent,
  each pin rebuilt) in every file but the generated ones;
- a commit that changes only generated files (a lock's records regenerated, a pin rebuilt).

No other PR's verdict vouches for a commit of the PR's own line: a PR stacked on this one was reviewed over it, not for it.

With ``lowrisk`` given, a delta whose own commits and hand resolutions change only files the low-risk lane lands with no
verdict at all (drive.toml's `lowrisk_excludes`: docs, benchmarks, most tests) carries it too: drive would land that change on
its ready mark alone as a PR of its own.

Generated files are each Lake package's lock (`queue.LEAN_LOCK`) and the pins tool's `PINS` (`queue.pins_at`): a Lean change's
lock waits for drive's Lean check, which regenerates its records and fails on a difference, and `pins.py --stale` holds each pin
to the tree, so neither needs a reviewer's eye.  A merge's textual result can still be wrong where two changes interact; that is
the risk `next` already takes for every pair of members it merges, and the same Lean check and full check catch it.

``python -m research.carry OLD NEW [--repo DIR] [--landed REF ...] [--head REF ...]`` prints the verdict as JSON, each
``--landed`` ref and its ancestors landed, each ``--head`` another PR's marked head with a land verdict."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from . import queue as Q

#: what `delta` says of a pair: the first two carry the verdict, the rest name why not
CARRIES = ("same-tree", "merge-only", "lowrisk-delta")
KINDS = CARRIES + ("missing", "rebased", "authored", "untrusted-parent", "hand-resolved", "edited-merge", "octopus")


@dataclass(frozen=True)
class Delta:
    kind: str
    why: str = ""
    commits: int = 0
    #: the files whose change the carry left to the checks (generated ones), and the trusted parents it merged
    generated: tuple[str, ...] = ()
    parents: tuple[str, ...] = field(default=())

    @property
    def carries(self) -> bool:
        return self.kind in CARRIES

    def as_dict(self) -> dict:
        return {"kind": self.kind, "carries": self.carries, "why": self.why, "commits": self.commits,
                "generated": list(self.generated), "parents": list(self.parents)}


def generated_at(repo: Path, ref: str) -> Callable[[str], bool]:
    """Whether a path is generated at ``ref``: a lock, or a pin of the pins tool there."""
    pins = set(Q.pins_at(repo, ref))
    return lambda f: PurePosixPath(f).name == Q.LEAN_LOCK or f in pins


def _ok(repo: Path, *args: str) -> bool:
    return Q._git(repo, *args, check=False).returncode == 0


def _tree(repo: Path, commit: str) -> str:
    return Q._out(repo, "rev-parse", f"{commit}^{{tree}}")


def _differ(repo: Path, a: str, b: str) -> list[str]:
    return Q._out(repo, "diff", "--name-only", "--no-renames", a, b).split()


def _some(files: Iterable[str]) -> str:
    files = sorted(files)
    return f"{', '.join(files[:5])}{' ...' if len(files) > 5 else ''}"


def delta(repo: Path, old: str, new: str, trusted: Callable[[str, str | None], bool],
          generated: Callable[[str], bool] | None = None, lowrisk: Callable[[list[str]], bool] | None = None) -> Delta:
    """`Delta` of ``new`` over ``old`` (see the module's docstring).  ``trusted(commit, merge)``: whether ``commit`` may come
    in unreviewed, as a parent ``merge`` brings in, or with ``merge`` None as a commit of the PR's own line (landed only);
    ``generated(path)``: whether a path's content is left to the checks (`generated_at` ``new`` when None).
    ``lowrisk(files)``, when given: whether the files the new head's own commits and hand resolutions change would land with
    no verdict at all (`queue.lowrisk` over drive.toml's `lowrisk_excludes`), so that delta carries the verdict too
    (`lowrisk-delta`)."""
    repo = Path(repo)
    for c in (old, new):
        if not _ok(repo, "cat-file", "-e", f"{c}^{{commit}}"):
            return Delta("missing", f"{c[:12]} isn't in this clone")
    if _tree(repo, old) == _tree(repo, new):
        return Delta("same-tree")
    if not _ok(repo, "merge-base", "--is-ancestor", old, new):
        return Delta("rebased", f"{old[:12]} isn't an ancestor of {new[:12]}")
    gen = generated or generated_at(repo, new)
    rows = [ln.split() for ln in Q._out(repo, "rev-list", "--first-parent", "--parents", f"{old}..{new}").splitlines() if ln]
    left: set[str] = set()
    merged: list[str] = []
    own: dict[str, set[str]] = {"authored": set(), "hand-resolved": set(), "edited-merge": set()}
    first: dict[str, str] = {}
    for c, *ps in rows:
        if trusted(c, None):
            continue
        off = [p for p in ps[1:] if not _ok(repo, "merge-base", "--is-ancestor", p, ps[0])]
        for p in off:
            if not trusted(p, c):
                return Delta("untrusted-parent", f"{c[:12]} merges {p[:12]}, which is neither landed nor in another PR's "
                             "marked head with a land verdict", len(rows))
        merged += off
        if len(ps) > 2:
            return Delta("octopus", f"{c[:12]} merges {len(ps)} parents", len(rows))
        if len(ps) == 1:
            files = _differ(repo, ps[0], c)
            if mine := [f for f in files if not gen(f)]:
                own["authored"].update(mine)
                first.setdefault("authored", f"{c[:12]} changes {_some(mine)}")
            left.update(f for f in files if gen(f))
            continue
        tree, conflicted, r = Q.merged_tree(repo, ps[0], ps[1])
        if tree is None:
            if r.returncode != 1:
                return Delta("hand-resolved", f"{c[:12]}: git merge-tree exited {r.returncode}", len(rows))
            if hand := [f for f in conflicted if not gen(f)]:
                own["hand-resolved"].update(hand)
                first.setdefault("hand-resolved", f"{c[:12]} resolves conflicts by hand in {_some(hand)}")
            tree = r.stdout.split()[0]
        files = _differ(repo, tree, c)
        if edited := [f for f in files if not gen(f) and f not in own["hand-resolved"]]:
            own["edited-merge"].update(edited)
            first.setdefault("edited-merge", f"{c[:12]} differs from the queue's merge of its parents in {_some(edited)}")
        left.update(f for f in files if gen(f))
    done = dict(commits=len(rows), generated=tuple(sorted(left)), parents=tuple(dict.fromkeys(merged)))
    mine = sorted(set().union(*own.values()))
    if not mine:
        return Delta("merge-only", **done)
    if lowrisk is not None and lowrisk(mine):
        return Delta("lowrisk-delta", f"its own change is low-risk: {_some(mine)}", **done)
    kind = next(k for k in own if own[k])
    return Delta(kind, first[kind], **done)


def trust(repo: Path, landed: Iterable[str], heads: Iterable[str] = ()) -> Callable[[str, str | None], bool]:
    """``trusted`` for `delta`: a commit that is one of ``landed`` (the landing branch's tip) or an ancestor of one, and, as a
    parent a merge brings in, also one in a head of ``heads`` (other PRs' marked heads with a land verdict) that doesn't
    contain that merge."""
    landed, heads = [t for t in landed if t], [h for h in heads if h]

    def within(c: str, t: str) -> bool:
        return c == t or _ok(repo, "merge-base", "--is-ancestor", c, t)

    def trusted(c: str, merge: str | None = None) -> bool:
        if any(within(c, t) for t in landed):
            return True
        return merge is not None and any(within(c, h) and not within(merge, h) for h in heads)

    return trusted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research.carry", description=__doc__.split("\n\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--landed", action="append", default=[], help="a landed ref (default: origin/checked)")
    ap.add_argument("--head", action="append", default=[], help="another PR's marked head with a land verdict")
    ns = ap.parse_args(argv)
    repo = Path(ns.repo)
    landed = [Q._out(repo, "rev-parse", t) for t in (ns.landed or ["origin/checked"])]
    d = delta(repo, ns.old, ns.new, trust(repo, landed, [Q._out(repo, "rev-parse", h) for h in ns.head]))
    json.dump(d.as_dict(), sys.stdout, indent=1)
    print()
    return 0 if d.carries else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Q.QueueError, subprocess.CalledProcessError) as e:
        sys.exit(f"research.carry: {e}")
