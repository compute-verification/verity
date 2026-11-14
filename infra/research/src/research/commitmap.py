"""Commit maps: after a history rewrite, a record keyed by an old commit is read for its image (fixture process plan §4.2).

A `commit-map/v1` artifact's payload is one line per rewritten commit, `<old> <new> <same-tree>` (`build` writes it from
git-filter-repo's `filter-repo/commit-map` and the two mirrors); `same-tree` is 1 when the rewrite left that commit's tree as it
was.  Its meta names the rewrite (`from`, `to`: the old and new `main`; `tool`; `args`).  A map applies once the latest
`commit_map` label on it by an acceptor (`ACCEPTORS`) is `accepted`, and stops applying when a later one is `retracted`.

    Maps.load(store)                   every map that applies, merged
    maps.new(old) / maps.olds(new)     the image of an old commit / the old commits a new one is the image of
    maps.equivalent(commit)            [commit, *old commits it is the image of with the same tree]: what a record keyed by a commit
                                       (an attempt's source commit, a grant's `pr:{n}@{sha}`) may be read for.  The map says which old
                                       commit to look under; the unchanged tree is why the record still holds
    maps.old_history(repo, ref, base)  commits of `ref` not in `base` that a rewrite replaced: a branch that merged old history brings
                                       back every old commit, and every blob the rewrite stripped

    research git old2new SHA... [--map FILE]           the image of each old commit (abbreviations of 7+ hex digits resolve)
    research git remap [BRANCH...] [--map FILE] [--dry-run]
                                                       a clone's local work moved onto the rewritten history: each local commit is
                                                       re-created with its old parents replaced by their images (tree, message,
                                                       author and committer unchanged; signatures dropped)
    research git check [REF] [--base REF] [--map FILE] exit 1 when REF holds a commit a rewrite replaced
    research git build PRE POST [--commit-map FILE]    print the `commit-map/v1` payload from the pre-rewrite and rewritten mirrors

Without `--map`, the maps come from the evidence store (`research data refresh` first on a machine that has not seen one).  With
no map at all, every function answers as if nothing had been rewritten.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .gates import git_env

KIND = "commit-map/v1"
LABEL = "commit_map"
ACCEPTED, RETRACTED = "accepted", "retracted"
#: Who may make a map apply: the rewrite's approver.  A map changes what the merge gate accepts, so no lane accepts its own.
ACCEPTORS = ("daniel",)
_SHA = re.compile(r"[0-9a-f]{40}")


def _git(repo: Path, *args: str, stdin: str | None = None, check: bool = True, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, text=True, check=check, env=env).stdout


def parse(text: str) -> dict[str, tuple[str, bool]]:
    """`{old: (new, same tree)}` from a `commit-map/v1` payload; filter-repo's own two-column file reads with same-tree unknown (False)."""
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 2 and _SHA.fullmatch(f[0]) and _SHA.fullmatch(f[1]) and set(f[1]) != {"0"}:
            out[f[0]] = (f[1], len(f) > 2 and f[2] == "1")
    return out


@dataclass
class Maps:
    """The merged maps that apply; empty when there is none."""

    pairs: dict[str, tuple[str, bool]] = field(default_factory=dict)
    arts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.inverse: dict[str, list[str]] = {}
        for o, (n, _) in self.pairs.items():
            if o != n:
                self.inverse.setdefault(n, []).append(o)

    @classmethod
    def load(cls, store) -> "Maps":
        pairs, arts = {}, []
        for row in store.index.select_artifacts(kind=KIND):
            art = row["id"]
            verdicts = [lab for lab in store.labels(art, key=LABEL) if lab.by in ACCEPTORS]
            if not verdicts or verdicts[-1].value != ACCEPTED:
                continue
            pairs.update(parse(Path(store.fetch(art)).read_text(encoding="utf-8")))
            arts.append(art)
        return cls(pairs, arts)

    @classmethod
    def from_file(cls, path: Path | str) -> "Maps":
        return cls(parse(Path(path).read_text(encoding="utf-8")), [str(path)])

    def __bool__(self) -> bool:
        return bool(self.pairs)

    def new(self, old: str) -> str | None:
        hit = self.pairs.get(old)
        return hit[0] if hit else None

    def olds(self, new: str) -> list[str]:
        return list(self.inverse.get(new, []))

    def equivalent(self, commit: str) -> list[str]:
        return [commit] + [o for o in self.olds(commit) if self.pairs[o][1]]

    def replaced(self) -> set[str]:
        return {o for o, (n, _) in self.pairs.items() if o != n}

    def old_history(self, repo: Path, ref: str, base: str | None = None) -> list[str]:
        if not self.pairs:
            return []
        revs = _git(repo, "rev-list", ref, *(["--not", base] if base else []), env=git_env()).split()
        gone = self.replaced()
        return [c for c in revs if c in gone]

    def resolve(self, sha: str) -> str | None:
        """An old commit named by 7 or more hex digits, when exactly one matches."""
        if sha in self.pairs:
            return sha
        hits = [o for o in self.pairs if o.startswith(sha)] if len(sha) >= 7 else []
        return hits[0] if len(hits) == 1 else None


def build(pre: Path, post: Path, commit_map: Path | None = None) -> str:
    """The `commit-map/v1` payload: every line of filter-repo's commit map, with whether the commit kept its tree."""
    pairs = parse((commit_map or post / "filter-repo" / "commit-map").read_text(encoding="utf-8"))

    def trees(repo: Path, commits: list[str]) -> list[str]:
        out = _git(repo, "cat-file", "--batch-check=%(objectname)", stdin="".join(f"{c}^{{tree}}\n" for c in commits))
        return out.split("\n")[:len(commits)]

    olds = sorted(pairs)
    before, after = trees(pre, olds), trees(post, [pairs[o][0] for o in olds])
    return "".join(f"{o} {pairs[o][0]} {int(b == a and ' ' not in b)}\n" for o, b, a in zip(olds, before, after))


# ---- remap: a clone's local commits onto the rewritten history ---------------------------------------------------------------------

def _recreate(repo: Path, commit: str, parents: list[str]) -> str:
    """`commit` with its parent lines replaced (and any signature dropped): the same tree, message, author and committer."""
    raw = subprocess.run(["git", "-C", str(repo), "cat-file", "commit", commit], capture_output=True, check=True).stdout
    head, _, msg = raw.partition(b"\n\n")
    lines, skip = [], False
    for ln in head.split(b"\n"):
        if skip and ln.startswith(b" "):
            continue
        skip = ln.startswith((b"gpgsig ", b"gpgsig-sha256 "))
        if skip or ln.startswith(b"parent "):
            continue
        lines.append(ln)
    lines[1:1] = [b"parent " + p.encode() for p in parents]            # after the tree line, as git writes it
    body = b"\n".join(lines) + b"\n\n" + msg
    return subprocess.run(["git", "-C", str(repo), "hash-object", "-t", "commit", "-w", "--stdin"], input=body,
                          capture_output=True, check=True).stdout.decode().strip()


def remap(repo: Path, maps: Maps, branches: list[str] | None = None, *, dry_run: bool = False, log=print) -> list[dict]:
    """Move each local branch (default: all) onto the rewritten history.  A commit the map knows becomes its image; every other commit
    reachable from the branch and not already in the rewritten history is re-created over its parents' images.  A branch whose history
    holds no replaced commit is left alone.  The images must be present: `git fetch` the rewritten refs first."""
    repo = Path(repo)
    current = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).strip()
    names = branches or _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/").split()
    gone = maps.replaced()
    done: dict[str, str] = {}
    out = []
    for b in names:
        tip = _git(repo, "rev-parse", "--verify", f"refs/heads/{b}").strip()
        todo = [ln.split() for ln in _git(repo, "rev-list", "--topo-order", "--reverse", "--parents", tip).splitlines()]
        if not any(c in gone for c, *_ in todo):
            out.append({"branch": b, "tip": tip, "new": tip, "recreated": 0, "moved": False})
            continue
        made = 0
        for c, *parents in todo:
            if c in done:
                continue
            if c in maps.pairs:
                done[c] = maps.pairs[c][0]
                continue
            new_parents = [done.get(p, p) for p in parents]
            if new_parents == parents:
                done[c] = c
                continue
            if dry_run:
                done[c] = c + "*"                                        # a placeholder: its children count as re-created too
                made += 1
                continue
            missing = [p for p in new_parents if _git(repo, "cat-file", "-t", p, check=False).strip() != "commit"]
            if missing:
                raise SystemExit(f"research git remap: {b}: the rewritten commit {missing[0][:12]} is not here; `git fetch` the rewritten history first")
            done[c] = _recreate(repo, c, new_parents)
            made += 1
        new = done[tip].rstrip("*")
        out.append({"branch": b, "tip": tip, "new": new, "recreated": made, "moved": True})
        log(f"{b}: {tip[:12]} -> {'(dry run)' if dry_run else new[:12]}, {made} local commit(s) re-created")
        if dry_run:
            continue
        if b == current and _git(repo, "rev-parse", f"{tip}^{{tree}}") != _git(repo, "rev-parse", f"{new}^{{tree}}"):
            _git(repo, "reset", "-q", "--keep", new)
        else:
            _git(repo, "update-ref", "-m", "research git remap", f"refs/heads/{b}", new, tip)
    return out


# ---- the command line -------------------------------------------------------------------------------------------------------------

def _maps(path: str | None, store_root: str | None) -> Maps:
    if path:
        return Maps.from_file(path)
    from .store.local import LocalStore  # noqa: PLC0415

    return Maps.load(LocalStore(store_root))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research git", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("old2new", "remap", "check"):
        p = sub.add_parser(name)
        p.add_argument("--map", default=None, metavar="FILE", help="a commit-map/v1 payload (or filter-repo's commit-map); default: the store's accepted maps")
        p.add_argument("--store", default=None, metavar="ROOT")
        p.add_argument("--repo", default=".", metavar="DIR")
    sub.choices["old2new"].add_argument("shas", nargs="+")
    sub.choices["remap"].add_argument("branches", nargs="*")
    sub.choices["remap"].add_argument("--dry-run", action="store_true")
    sub.choices["check"].add_argument("ref", nargs="?", default="HEAD")
    sub.choices["check"].add_argument("--base", default=None, help="only the commits of REF not in BASE")
    b = sub.add_parser("build")
    b.add_argument("pre", type=Path)
    b.add_argument("post", type=Path)
    b.add_argument("--commit-map", type=Path, default=None, help="default: POST/filter-repo/commit-map")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        sys.stdout.write(build(a.pre, a.post, a.commit_map))
        return 0
    maps = _maps(a.map, a.store)
    if not maps:
        print("research git: no commit map applies (none accepted in the store, and no --map)", file=sys.stderr)
    if a.cmd == "old2new":
        rc = 0
        for s in a.shas:
            old = maps.resolve(s.lower())
            if old is None:
                print(f"{s} ?")
                rc = 1
            else:
                print(f"{old} {maps.new(old)}")
        return rc
    if a.cmd == "check":
        hits = maps.old_history(Path(a.repo), a.ref, a.base)
        if hits:
            print(f"research git check: {a.ref} holds {len(hits)} commit(s) a history rewrite replaced (e.g. {hits[0][:12]}); "
                  f"move the work with `research git remap`, never merge old history", file=sys.stderr)
            return 1
        return 0
    remap(Path(a.repo), maps, a.branches or None, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
