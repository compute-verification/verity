"""Verdict carry (drive.toml's `verdict_carry`, `Landing._carried_from`): whether a PR's new head changes nothing a
reviewer saw at an older head with a land verdict, beyond merges of commits the queue trusts, so that verdict holds at the new
head with no reviewer.

The PR's own line is its first-parent line from ``new`` back to ``old`` (``git rev-list --first-parent old..new``); what its
merges bring in from off that line is other work.  ``new`` carries the verdict of ``old`` when ``old`` is an ancestor of ``new``
and each commit of that line is one of:

- landed (`trusted` with no merge: on `main`);
- a merge of two parents whose other parent is trusted (`trusted(parent, merge)`: landed, another PR's current marked head
  with a land verdict and no hold, which drive's containment lands first, `Queue.entries`, or any open PR's head, current or
  older, with a land verdict of its own at that head and no hold there, read through the queue's labels: what a reviewer
  passed as that PR's change), and whose tree is what the queue's
  merge of its parents gives (`queue.merged_tree`: its first parent's merge drivers, each conflicted pin rebuilt) but in
  generated content;
- a commit that changes only generated content;
- a restack's commit across a move (`tools/move/restack.py`: one parent, a `Restack-move:` trailer naming a landed move)
  that changes no file of either head's own change against `main` (`queue.own_files` at the first landed tip `trust` names),
  as the commit that adds the map a move runs by does, which the merge above it then holds as the move has it: the crossing
  changes no line of the PR's own change, so nothing reruns.  The merge above such a commit is the queue's merge of its
  parents from their own merge base.  Given ``restack``, a move another commit of the line reruns is judged by the rerun
  below, every commit of it;
- given a landed `tools/move/restack.py` (``restack``), a restack's commit across a move, whose `Restack-move:` trailer names
  a landed move, and whose tree is that move's script rerun on its parent (`moved_tree`), with the map the move adds lent to
  the script as restack.py lends it, or is what restack.py brings onto its rerun of the change the move made beside its
  script, its `hand` paths and the map it adds (`hand_tree`), but in generated content; the merge above it is the queue's
  merge of its parents with the move for their merge base, as the restack's own merge is.  A move's
  script runs a tree's own code, so it reruns only once every other commit of the line passed, on trees they vouch for.

Generated content is each pin of the pins tool (`queue.pins_at` at ``new``), which `pins.py --stale` holds to the tree, and
each lock's records (`queue.LEAN_LOCK`: `lock_policy`), which a Lean check regenerates and drive commits as it lands.  The rest
of a lock is policy a person writes and the Lean check keeps as written, so a lock that differs in it, or that the queue's
merge leaves conflicted (the lock's merge driver refuses only over policy), is the PR's own change.  A merge's textual result
can still be wrong where two changes interact; that is the risk a candidate takes for every pair of members it merges, and the same
Lean check and full check catch it.

No other PR's verdict vouches for a commit of the PR's own line (a PR stacked on this one was reviewed over it, not for it),
nor for an older head of that PR.  A hold at a later head of the PR voids an older head's land for whatever carries from it
(`voided`).

``python -m research.carry OLD NEW [--repo DIR] [--landed REF ...] [--head REF ...]`` prints the verdict as JSON and exits 0
when it carries: each ``--landed`` ref and its ancestors landed (`origin/main` when none is given), each
``--head`` another PR's head with a land verdict at it and no hold, and the first landed ref holding `tools/move/restack.py`
reruns the moves."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from . import queue as Q

#: what `delta` says of a pair: the first carries the verdict, the rest name why not
CARRIES = ("merge-only",)
KINDS = CARRIES + ("missing", "rebased", "authored", "untrusted-parent", "hand-resolved", "edited-merge", "octopus")
#: a restack's commit across a move (`tools/move/restack.py`): the move it reran on its parent
RESTACK_MOVE_RE = re.compile(r"^Restack-move: ([0-9a-f]{40,64})$", re.M)
#: `moved_tree`'s program: with `tools/move/restack.py` from the directory argv[1], runs the move argv[3] on commit argv[2] (on
#: the move's base where it is empty) as a restack runs one on a branch (`run_script`: its move line's script and arguments,
#: its parent's `tools/move/`, and the map the move adds where its base lacks it, `own_map`, in a scratch worktree), and
#: prints the tree it leaves (`MOVED_RE`)
_MOVED = ("import sys; sys.path.insert(0, sys.argv[1]); import restack as R\n"
          "R.SCRIPT, R.SCRIPT_ARGS = R.move_line(sys.argv[3], None) or sys.exit(f'{sys.argv[3][:12]} has no move line')\n"
          "base = R.move_base(sys.argv[3], None)\n"
          "own = R.own_map(sys.argv[3], base, R.SCRIPT_ARGS, None) if hasattr(R, 'own_map') else None\n"
          "print('moved', R.run_script(sys.argv[2] or base, base, sys.executable, None,\n"
          "                            *([(sys.argv[3], own)] if own else []))[0])\n")
MOVED_RE = re.compile(r"^moved ([0-9a-f]{40,64})$", re.M)
#: how long one move's script runs at most on one commit (`moved_tree`): a layout map on the whole tree, `uv lock` with it
MOVED_TIMEOUT_S = 1800.0
#: what a move's script writes on the move's base (`hand_tree`), by the clone, the move and the `restack.py` that ran it
_ON_BASE: dict[tuple[str, str, str], tuple[str | None, str]] = {}


@dataclass(frozen=True)
class Delta:
    kind: str
    why: str = ""
    commits: int = 0
    #: the files whose change the carry left to the checks (generated content), the trusted parents it merged, and the moves
    #: whose script it reran
    generated: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    moves: tuple[str, ...] = ()
    #: the moves of the restack's commits that changed no file of either head's own change
    crossed: tuple[str, ...] = ()

    @property
    def carries(self) -> bool:
        return self.kind in CARRIES

    def as_dict(self) -> dict:
        return {"kind": self.kind, "carries": self.carries, "why": self.why, "commits": self.commits,
                "generated": list(self.generated), "parents": list(self.parents), "moves": list(self.moves),
                "crossed": list(self.crossed)}


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


def moved_tree(repo: Path, commit: str, move: str, restack: Path, timeout: float = MOVED_TIMEOUT_S) -> tuple[str | None, str]:
    """(the tree ``commit`` (the move's base where it is '') becomes once the move ``move`` runs on it as
    `tools/move/restack.py`, from the directory ``restack``, runs it on a branch (`_MOVED`), ''), or (None, why it couldn't).  It runs in a process group of its own, killed whole once
    ``timeout`` seconds pass (`queue.run_group`), with its scratch worktree in a temporary directory, which goes once it has
    run, and with it the clone's record of a worktree a killed run left, forgotten under restack.py's lock
    (`queue.WORKTREES_LOCK`).  The tree's own code runs there but needs neither the queue's credentials nor the store's keys,
    which drive and the loop hold (`queue.tree_env`), so the run gets none, and `gh` a config that isn't there."""
    with tempfile.TemporaryDirectory(prefix="research-carry-", ignore_cleanup_errors=True) as d:
        env = Q.tree_env(Q._script_env())
        rc, out = Q.run_group([sys.executable, "-c", _MOVED, str(restack), commit, move], repo, timeout,
                              env={**env, "TMPDIR": d, "GH_CONFIG_DIR": str(Path(d, "no-gh"))})
        left = any(Path(d).iterdir())
    if left:
        with open(Path(repo, Q._out(repo, "rev-parse", "--git-common-dir")) / Q.WORKTREES_LOCK, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            Q._git(repo, "worktree", "prune", check=False)
    if rc is None:
        return None, f"ran over {timeout:.0f} s"
    got = MOVED_RE.findall(out)
    return (got[-1], "") if rc == 0 and got else (None, f"exit {rc}: {' | '.join(out.strip().splitlines()[-3:])}")


def hand_tree(repo: Path, commit: str, move: str, restack: Path) -> tuple[str | None, str]:
    """(the tree `tools/move/restack.py` leaves once it brings onto ``commit`` (its rerun of ``move``) the change ``move`` made
    beside its script, its `hand` paths and the map it adds: the queue's merge of ``commit`` and ``move`` over what the
    script writes on the move's base (`moved_tree`, run once a move and `restack.py`), as restack.py merges them, ''), or
    (None, why it couldn't)."""
    key = (str(repo), move, hashlib.sha256((restack / "restack.py").read_bytes()).hexdigest())
    if key not in _ON_BASE:
        _ON_BASE[key] = moved_tree(repo, "", move, restack)
    tree, why = _ON_BASE[key]
    if tree is None:
        return None, why
    if tree == Q._out(repo, "rev-parse", f"{move}^{{tree}}"):
        return None, f"{move[:12]} made nothing beside its script"
    base = Q._out(repo, "-c", "user.name=research.carry", "-c", "user.email=carry@localhost", "commit-tree", tree, "-m",
                  f"what {move[:12]}'s script writes on its base")
    merged, conflicted, r = Q.merged_tree(repo, commit, move, merge_base=base)
    return (merged, "") if merged else (None, f"it conflicts with the move in {_some(conflicted)}" if conflicted
                                       else f"git merge-tree exited {r.returncode}")


def _reruns(repo: Path, old: str, new: str, trusted: Callable[[str, str | None], bool]) -> dict[str, str]:
    """{commit: the move it reran} of each restack's commit on the PR's own line: one parent, and one `Restack-move:` trailer,
    naming a landed move."""
    log = Q._out(repo, "log", "--first-parent", "--no-merges", "--format=%H%x00%B%x1e", f"{old}..{new}")
    found = {}
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        c, body = rec.split("\x00", 1)
        if len(moves := RESTACK_MOVE_RE.findall(body)) == 1 and trusted(moves[0], None):
            found[c] = moves[0]
    return found


def delta(repo: Path, old: str, new: str, trusted: Callable[[str, str | None], bool], restack: str | None = None) -> Delta:
    """`Delta` of ``new`` over ``old`` (the module's docstring).  ``trusted(commit, merge)``: whether ``commit`` may come in
    unreviewed, as a parent ``merge`` brings in, or with ``merge`` None as a commit of the PR's own line (landed only).
    ``restack``: a landed commit whose `tools/move/restack.py` reruns the move a restack's commit names; without it, such a
    commit is the PR's own unless it changes no file of either head's own change against the first landed tip ``trusted``
    names (`Trusted.landed`)."""
    repo = Path(repo)
    for c in (old, new):
        if not _ok(repo, "cat-file", "-e", f"{c}^{{commit}}"):
            return Delta("missing", f"{c[:12]} isn't in this clone")
    if not _ok(repo, "merge-base", "--is-ancestor", old, new):
        return Delta("rebased", f"{old[:12]} isn't an ancestor of {new[:12]}")
    pins = set(Q.pins_at(repo, new))
    rows = [ln.split() for ln in Q._out(repo, "rev-list", "--first-parent", "--parents", f"{old}..{new}").splitlines() if ln]
    crossings = _reruns(repo, old, new, trusted)
    main = next(iter(getattr(trusted, "landed", ())), None)
    owned = Q.own_files(repo, main, old) | Q.own_files(repo, main, new) if main and crossings else frozenset()
    crossed = {c: m for c, m in crossings.items() if main and not owned.intersection(_differ(repo, f"{c}^", c))}
    if restack:
        rerun = {m for c, m in crossings.items() if c not in crossed}
        crossed = {c: m for c, m in crossed.items() if m not in rerun}
    reruns = {c: m for c, m in crossings.items() if c not in crossed} if restack else {}
    left: set[str] = set()
    merged: list[str] = []
    moved: list[tuple[str, str]] = []
    own: dict[str, set[str]] = {"authored": set(), "hand-resolved": set(), "edited-merge": set()}
    first: dict[str, str] = {}
    for c, *ps in rows:
        if trusted(c, None):
            continue
        if len(ps) > 2:
            return Delta("octopus", f"{c[:12]} merges {len(ps)} parents", len(rows))
        off = [p for p in ps[1:] if not _ok(repo, "merge-base", "--is-ancestor", p, ps[0])]
        if bad := next((p for p in off if not trusted(p, c)), None):
            return Delta("untrusted-parent", f"{c[:12]} merges {bad[:12]}, which is neither landed nor a PR's head with a land "
                         "verdict at it and no hold", len(rows))
        merged += off
        hand: list[str] = []
        if len(ps) == 1 and c in crossed:
            continue
        if len(ps) == 1 and c in reruns:
            moved.append((c, ps[0]))
            continue
        if len(ps) == 1:
            want, files, kind = ps[0], _differ(repo, ps[0], c), "authored"
        else:
            move = reruns.get(ps[0])
            at = move if move and _ok(repo, "merge-base", "--is-ancestor", move, ps[1]) else None
            tree, conflicted, r = Q.merged_tree(repo, ps[0], ps[1], merge_base=at)
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
    moves: list[str] = []
    if moved and not any(own.values()):
        with tempfile.TemporaryDirectory(prefix="research-carry-", ignore_cleanup_errors=True) as d:
            tools = Q.write_scripts(repo, Q.restack_files(repo, restack), Path(d)) / Q.MOVE_DIR
            for c, p in reversed(moved):
                tree, why = moved_tree(repo, p, reruns[c], tools)
                mine, gen = _judged(repo, tree, c, _differ(repo, tree, c), pins) if tree else (_differ(repo, p, c), [])
                beside = bool(mine and tree and (hand := hand_tree(repo, p, reruns[c], tools)[0]))
                if beside:
                    mine, gen = _judged(repo, hand, c, _differ(repo, hand, c), pins)
                if mine:
                    own["authored"].update(mine)
                    first["authored"] = (f"{c[:12]} changes {_some(mine)} beyond the move {reruns[c][:12]}'s script"
                                         + (" and what it made beside it" if beside else "") if tree
                                         else f"{c[:12]} names the move {reruns[c][:12]}, whose script didn't rerun: {why}")
                    break
                left.update(gen)
                if reruns[c] not in moves:
                    moves.append(reruns[c])
    parents = tuple(dict.fromkeys(merged))
    across = tuple(dict.fromkeys(m for c, *_ in reversed(rows) if (m := crossed.get(c))))
    done = dict(commits=len(rows), generated=tuple(sorted(left)), parents=parents, moves=tuple(moves), crossed=across)
    if kind := next((k for k in own if own[k]), None):
        return Delta(kind, first[kind], **done)
    did = ([f"merge {_some(p[:12] for p in parents)} as the queue would"] if parents else []) + \
        ([f"rerun the move {_some(m[:12] for m in moves)} as a restack does"] if moves else []) + \
        ([f"cross the move {_some(m[:12] for m in across)} as a restack does, outside both heads' own change"]
         if across else [])
    why = (f"its {len(rows)} commit(s) since only {' and '.join(did)}" if did else
           f"its {len(rows)} commit(s) since change only generated content")
    return Delta("merge-only", why + (f", leaving {_some(left)} to the checks" if left else ""), **done)


@dataclass(frozen=True)
class Trusted:
    """``trusted`` for `delta` (`trust`), with the landed tips it trusts, `main`'s first, which a restack's commit's own-change
    test reads."""
    repo: Path
    landed: tuple[str, ...]
    heads: frozenset[str] = frozenset()
    lands: Callable[[str], bool] | None = None

    def __call__(self, c: str, merge: str | None = None) -> bool:
        if any(c == t or _ok(self.repo, "merge-base", "--is-ancestor", c, t) for t in self.landed):
            return True
        return merge is not None and (c in self.heads or (self.lands is not None and self.lands(c)))


def voided(repo: Path, head: str, held: Iterable[str]) -> str | None:
    """The first of ``held`` (heads of ``head``'s PR whose verdicts a hold keeps) that is a later head than ``head``: any but
    ``head`` and its ancestors (a head pushed on it, or one that took its place), and one this clone lacks.  A fault found
    there may be one ``head`` holds, so such a hold voids ``head``'s land for whatever carries from it, however old the hold;
    a hold at an ancestor is what ``head`` answered.  None when no hold voids it."""
    return next((h for h in sorted(set(held)) if h != head and not _ok(repo, "merge-base", "--is-ancestor", h, head)), None)


def trust(repo: Path, landed: Iterable[str], heads: Iterable[str] = (),
          lands: Callable[[str], bool] | None = None) -> Trusted:
    """``trusted`` for `delta`: a commit that is one of ``landed`` (`main`'s tip) or an ancestor of one, and,
    as a parent a merge brings in, also one of ``heads`` (other PRs' current marked heads with a land verdict and no hold) or
    one ``lands`` says has a land verdict at it and no hold, its own PR's (asked only of a parent nothing else trusts)."""
    return Trusted(Path(repo), tuple(t for t in landed if t), frozenset(h for h in heads if h), lands)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research.carry", description=__doc__.split("\n\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--landed", action="append", default=[], help="a landed ref (default: origin/main)")
    ap.add_argument("--head", action="append", default=[], help="another PR's head with a land verdict at it and no hold")
    ns = ap.parse_args(argv)
    repo = Path(ns.repo)

    def sha(ref: str) -> str:
        return Q._out(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")

    landed = [sha(t) for t in ns.landed or ["origin/main"]]
    restack = next((t for t in landed if Q.scripts_at(repo, t, [Q.RESTACK])), None)
    d = delta(repo, sha(ns.old), sha(ns.new), trust(repo, landed, [sha(h) for h in ns.head]), restack)
    json.dump(d.as_dict(), sys.stdout, indent=1)
    print()
    return 0 if d.carries else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Q.QueueError, subprocess.CalledProcessError) as e:
        sys.exit(f"research.carry: {e}")
