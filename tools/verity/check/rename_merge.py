#!/usr/bin/env python3
"""`tools/verity/check/rename_merge.py TIP_REF PR_HEAD -m MSG [--map OLD=NEW ...] [--worktree DIR] [--json]`: merge a PR onto a train
tip that has moved files the PR still edits under their old paths.

ci resolves rename-only conflicts in the train tip's merge commit and names each resolution in the handoff; conflicts that need
judgment still go to the PR's owner (Daniel, 4 Oct).  This is that resolution, mechanical and refusing everything else.

The rename map is the tip's moves since merge-base(TIP, PR): its 100%-similar file renames, and the directory renames they imply.
A directory `old` renamed to `new` is inferred from a file renamed `old/rest -> new/rest` (and up the tree while the last
components agree), and accepted only when the tip has no file left under `old` and every rename out of it agrees on `new`.  A
directory whose renames disagree is ambiguous.  `--map OLD=NEW` adds a file (when OLD is a file at the merge base) or directory
rename and wins over an inferred one.

Each path the PR changed is mapped (exact file renames first, then the longest renamed directory).  When none maps, this is an
ordinary `git merge-tree`.  Otherwise the PR's changes at unmapped paths merge as usual, and each change at a mapped path is
applied at its new path as a three-way merge of the file (`git merge-file`: base = the merge base's file at the old path, ours =
the tip's at the new path, theirs = the PR's at the old path), which is what `git apply --3way` does with a patch from the merge
base to the PR, without its path quoting.  A file the PR added under a renamed directory lands in the new directory; a file the
PR deleted is deleted at the new path if the tip moved it unchanged.

Any of these refuses, naming the paths, and commits nothing: a conflict in the ordinary merge (both sides changed the same lines,
a file one side deleted and the other changed), a three-way merge of a mapped file that leaves conflict markers, a mapped path
the tip has no file at, an added file whose new path the tip already has, a deletion of a file the tip changed, and a file added
under an ambiguous directory.  Exit 1.

On success the merge commit (parents TIP and PR_HEAD; MSG and one `Renamed-by-ci: old -> new` trailer per resolved file) is
written to the object store and its id printed, with one line per resolution for the handoff (`--json`: the same as an object).
Nothing in the caller's working tree, index or refs changes, unless `--worktree DIR` names a clean checkout of TIP, which is then
fast-forwarded to the merge.
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import subprocess
import sys
import tempfile

TRAILER = "Renamed-by-ci"


class Refused(Exception):
    def __init__(self, refusals: list[tuple[str, str]]):
        super().__init__("; ".join(f"{p}: {r}" for p, r in refusals))
        self.refusals = refusals


def git(*args: str, cwd: str | None = None, env: dict | None = None, input: bytes | None = None, ok: tuple[int, ...] = (0,)
        ) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *args], cwd=cwd, env={**os.environ, **(env or {})}, input=input, capture_output=True)
    if p.returncode not in ok:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.decode(errors='replace').strip()}")
    return p


def out(*args: str, cwd: str | None = None) -> str:
    return git(*args, cwd=cwd).stdout.decode().strip()


def _z(raw: bytes) -> list[str]:
    return [s.decode() for s in raw.split(b"\0") if s]


def tree_entries(rev: str, cwd: str | None) -> dict[str, tuple[str, str]]:
    """path -> (mode, blob) for every file (and submodule) in `rev`'s tree."""
    entries = {}
    for line in _z(git("ls-tree", "-r", "-z", "--full-tree", rev, cwd=cwd).stdout):
        meta, path = line.split("\t", 1)
        mode, _, oid = meta.split(" ")
        entries[path] = (mode, oid)
    return entries


def tip_moves(base: str, tip: str, cwd: str | None) -> tuple[dict[str, str], set[str]]:
    """The tip's 100%-similar renames since `base` (old -> new), and the paths it deleted outright."""
    fields = _z(git("diff", "-z", "--name-status", "--find-renames=100%", "--diff-filter=RD", base, tip, cwd=cwd).stdout)
    renames, deleted, i = {}, set(), 0
    while i < len(fields):
        status = fields[i]
        if status.startswith("R"):
            renames[fields[i + 1]] = fields[i + 2]
            i += 3
        else:
            deleted.add(fields[i + 1])
            i += 2
    return renames, deleted


def _under(path: str, d: str) -> bool:
    return path.startswith(d + "/")


def directory_renames(renames: dict[str, str], tip_paths: set[str]) -> tuple[dict[str, str], set[str]]:
    """Directory renames the file renames imply (old dir -> new dir), and the old directories whose renames disagree."""
    candidates: dict[str, set[str]] = {}
    for old, new in renames.items():
        od, nd = posixpath.dirname(old), posixpath.dirname(new)
        while od and nd and od != nd:
            candidates.setdefault(od, set()).add(nd)
            if posixpath.basename(od) != posixpath.basename(nd):
                break
            od, nd = posixpath.dirname(od), posixpath.dirname(nd)
    tip_dirs = {posixpath.dirname(p) for p in tip_paths}
    tip_dirs |= {d for p in tip_dirs for d in _ancestors(p)}
    accepted, ambiguous = {}, set()
    for od, targets in candidates.items():
        if od in tip_dirs:
            continue
        if len(targets) == 1:
            accepted[od] = next(iter(targets))
        else:
            ambiguous.add(od)
    return accepted, ambiguous


def _ancestors(d: str) -> list[str]:
    out = []
    while d:
        out.append(d)
        d = posixpath.dirname(d)
    return out


class RenameMap:
    def __init__(self, files: dict[str, str], dirs: dict[str, str], ambiguous: set[str]):
        self.files, self.dirs, self.ambiguous = files, dirs, ambiguous

    def map(self, path: str) -> tuple[str | None, str | None]:
        """(new path or None, the ambiguous directory that blocks it or None)."""
        if path in self.files:
            return self.files[path], None
        for d in sorted(self.dirs, key=len, reverse=True):
            if _under(path, d):
                return self.dirs[d] + path[len(d):], None
        for d in sorted(self.ambiguous, key=len, reverse=True):
            if _under(path, d):
                return None, d
        return None, None


def rename_map(base: str, tip: str, extra: list[tuple[str, str]], base_paths: set[str], tip_paths: set[str], cwd: str | None
               ) -> RenameMap:
    files, _ = tip_moves(base, tip, cwd)
    dirs, ambiguous = directory_renames(files, tip_paths)
    for old, new in extra:
        old, new = old.strip("/"), new.strip("/")
        if old in base_paths:
            files[old] = new
        else:
            dirs[old] = new
            ambiguous.discard(old)
    return RenameMap(files, dirs, ambiguous)


def pr_changes(base: str, pr: str, cwd: str | None) -> dict[str, str]:
    """path -> A|M|D|T for every path the PR changed since `base`, without rename detection (a PR's own move is a D and an A)."""
    fields = _z(git("diff", "-z", "--name-status", "--no-renames", base, pr, cwd=cwd).stdout)
    return {fields[i + 1]: fields[i][0] for i in range(0, len(fields), 2)}


class Index:
    """A temporary index: read a tree, edit entries, write a tree. Touches no working tree."""

    def __init__(self, tree: str, tmp: str, name: str, cwd: str | None):
        self.env, self.cwd = {"GIT_INDEX_FILE": os.path.join(tmp, name)}, cwd
        git("read-tree", tree, cwd=cwd, env=self.env)

    def put(self, path: str, mode: str, blob: str) -> None:
        git("update-index", "--add", "--cacheinfo", f"{mode},{blob},{path}", cwd=self.cwd, env=self.env)

    def drop(self, path: str) -> None:
        git("update-index", "--force-remove", "--", path, cwd=self.cwd, env=self.env)

    def write(self) -> str:
        return git("write-tree", cwd=self.cwd, env=self.env).stdout.decode().strip()


def merge_trees(base: str, ours: str, theirs: str, cwd: str | None) -> tuple[str, list[str]]:
    """`git merge-tree`: (tree, conflicted paths)."""
    p = git("merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", f"--merge-base={base}", ours, theirs,
            cwd=cwd, ok=(0, 1))
    fields = _z(p.stdout)
    return fields[0], sorted(set(fields[1:])) if p.returncode else []


def three_way(base: tuple[str, str], ours: tuple[str, str], theirs: tuple[str, str], tmp: str, cwd: str | None
              ) -> tuple[str, str] | None:
    """The merged (mode, blob) of one file, or None when it conflicts."""
    if ours == base or ours == theirs:
        return theirs
    if theirs == base:
        return ours
    modes = {base[0], ours[0], theirs[0]}
    mode = ours[0] if theirs[0] == base[0] else theirs[0] if ours[0] == base[0] else None
    if mode is None or len(modes - {"100644", "100755"}):
        return None
    files = []
    for i, (_, blob) in enumerate((ours, base, theirs)):
        f = os.path.join(tmp, f"merge-file-{i}")
        with open(f, "wb") as h:
            h.write(git("cat-file", "blob", blob, cwd=cwd).stdout)
        files.append(f)
    p = subprocess.run(["git", "merge-file", "-p", "--quiet", *files], cwd=cwd, capture_output=True)
    if p.returncode != 0:
        return None
    return mode, git("hash-object", "-w", "--stdin", cwd=cwd, input=p.stdout).stdout.decode().strip()


def rename_merge(tip_ref: str, pr_ref: str, message: str, extra: list[tuple[str, str]] = (), cwd: str | None = None) -> dict:
    tip = out("rev-parse", "--verify", f"{tip_ref}^{{commit}}", cwd=cwd)
    pr = out("rev-parse", "--verify", f"{pr_ref}^{{commit}}", cwd=cwd)
    base = out("merge-base", tip, pr, cwd=cwd)
    base_t, tip_t, pr_t = (tree_entries(r, cwd) for r in (base, tip, pr))
    rmap = rename_map(base, tip, list(extra), set(base_t), set(tip_t), cwd)
    changes = pr_changes(base, pr, cwd)

    mapped: dict[str, str] = {}
    refusals: list[tuple[str, str]] = []
    for path in sorted(changes):
        new, blocked = rmap.map(path)
        if blocked is not None and changes[path] == "A":
            refusals.append((path, f"added under {blocked}/, which the tip moved to more than one directory"))
        elif new is not None and new != path:
            mapped[path] = new

    with tempfile.TemporaryDirectory(prefix="rename-merge-") as tmp:
        theirs = pr
        if mapped:
            ix = Index(pr, tmp, "pr", cwd)
            for old in mapped:
                if old in base_t:
                    ix.put(old, *base_t[old])
                else:
                    ix.drop(old)
            theirs = out("commit-tree", ix.write(), "-p", base, "-m", "rename_merge: the PR without its changes at moved paths",
                         cwd=cwd)
        tree, conflicts = merge_trees(base, tip, theirs, cwd)
        refusals += [(p, "conflicts in the merge (not a rename: the owner resolves it)") for p in conflicts]

        resolutions = []
        if mapped and not conflicts:
            merged = tree_entries(tree, cwd)
            ix = Index(tree, tmp, "merge", cwd)
            for old, new in mapped.items():
                kind = changes[old]
                if kind == "A":
                    if new in merged:
                        refusals.append((old, f"added under a moved directory, but the tip already has {new}"))
                        continue
                    ix.put(new, *pr_t[old])
                    resolutions.append({"old": old, "new": new, "change": "added"})
                elif new not in merged:
                    refusals.append((old, f"the PR changed it, but the tip has no {new} (deleted after the move?)"))
                elif kind == "D":
                    if merged[new] != base_t[old]:
                        refusals.append((old, f"the PR deletes it, and the tip moved it to {new} and changed it"))
                        continue
                    ix.drop(new)
                    resolutions.append({"old": old, "new": new, "change": "deleted"})
                else:
                    result = three_way(base_t[old], merged[new], pr_t[old], tmp, cwd)
                    if result is None:
                        refusals.append((old, f"the PR's change does not apply cleanly at {new} (both sides changed the same lines)"))
                        continue
                    ix.put(new, *result)
                    resolutions.append({"old": old, "new": new, "change": "edited"})
            tree = ix.write()
    if refusals:
        raise Refused(refusals)

    trailers = "".join(f"{TRAILER}: {r['old']} -> {r['new']}\n" for r in resolutions)
    full = message.rstrip("\n") + "\n" + (f"\n{trailers}" if trailers else "")
    commit = git("commit-tree", tree, "-p", tip, "-p", pr, "-F", "-", cwd=cwd, input=full.encode()).stdout.decode().strip()
    return {"status": "merged", "commit": commit, "tree": tree, "tip": tip, "pr": pr, "merge_base": base,
            "resolutions": resolutions}


def summary(result: dict) -> str:
    lines = [f"merged {result['pr'][:12]} onto {result['tip'][:12]}: {result['commit']}"]
    lines += [f"{TRAILER}: {r['old']} -> {r['new']} ({r['change']})" for r in result["resolutions"]]
    return "\n".join(lines)


def fast_forward(worktree: str, tip: str, commit: str) -> None:
    head = out("rev-parse", "HEAD", cwd=worktree)
    if head != tip:
        raise SystemExit(f"--worktree {worktree}: HEAD is {head[:12]}, not the tip {tip[:12]}; nothing changed")
    if out("status", "--porcelain", "--untracked-files=no", cwd=worktree):
        raise SystemExit(f"--worktree {worktree}: uncommitted changes; nothing changed")
    git("merge", "--ff-only", "-q", commit, cwd=worktree)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tip")
    ap.add_argument("pr")
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW", help="a file or directory rename to apply too")
    ap.add_argument("--worktree", metavar="DIR", help="a clean checkout of TIP to fast-forward to the merge")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    extra = []
    for m in a.map:
        old, sep, new = m.partition("=")
        if not sep or not old or not new:
            ap.error(f"--map {m!r}: expected OLD=NEW")
        extra.append((old, new))
    try:
        result = rename_merge(a.tip, a.pr, a.message, extra)
    except Refused as e:
        if a.json:
            print(json.dumps({"status": "refused", "refusals": [{"path": p, "reason": r} for p, r in e.refusals]}, indent=2))
        else:
            print(f"refused, nothing committed ({len(e.refusals)} path(s) for the owner):", file=sys.stderr)
            for p, r in e.refusals:
                print(f"  {p}: {r}", file=sys.stderr)
        return 1
    if a.worktree:
        fast_forward(a.worktree, result["tip"], result["commit"])
    print(json.dumps(result, indent=2) if a.json else summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
