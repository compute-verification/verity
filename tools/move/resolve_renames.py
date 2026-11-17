#!/usr/bin/env python3
"""Resolve the conflicts of a merge of moved code into a PR's head that come from a move's rename
against the PR's edit, and only those; leave every other conflict to the owner.

Run at the root of a checkout whose tools/move/ is the moved tree's, naming the merge's checkout:

    python3 tools/move/resolve_renames.py DIR OTHER [--head HEAD] [--extra-map COMMIT:PATH|FILE ...]

DIR is a checkout stopped in `git merge OTHER` (OTHER: `main` after the moves, or the move base restack.py named); HEAD
(default DIR's HEAD) is the PR's head. The moves are those OTHER has and HEAD lacks (`restack.moves_on`); REWRITE is their
maps applied in order (`stale_names.LayoutNames`, `stale_names.WordNames`): what each move's script would have done to the
text, short of layout.py's parsers. Each `--extra-map` is applied after them: a map no move line names. A lock
(lean-audit.json) is never merged: it is listed, to regenerate (`check.py --update`).

For each conflicted file F (at its new path T, where the maps move it, or where OTHER turned F into a link to T):
merge three ways REWRITE(F at HEAD), REWRITE(F at the merge base) and T at OTHER (the first two as git's merge staged
them, so a rename git found and a criss-cross's virtual base are git's own), as restack.py merges the PR's moved
tree with the move commit as base. A move that only renamed against the PR's edit leaves no conflict there, and T is
written and added; F's old path is put back as OTHER has it (a link, a shim, or nothing). Where that merge still conflicts,
the conflict is real content (the PR's edit against another change on OTHER's side): T is written with diff3 markers,
left unadded, and each hunk's first line is printed for the owner. A modify/delete the maps don't explain is the owner's,
and so is a binary file (a NUL byte on any side) or one `git merge-file` fails on: it is left as git's merge left it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stale_names as S


def git(d, *a, ok=(0,)):
    p = subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, errors="surrogateescape")
    if p.returncode not in ok:
        raise SystemExit(f"git {' '.join(a[:3])}: {p.stderr.strip()[-300:]}")
    return p


def rewrite(maps, text: str, path: str) -> str:
    for _, _, nm in maps:
        text = nm.sub(text, path)
    return text


def new_path(maps, p: str) -> str:
    for _, _, nm in maps:
        p = nm.path(p)
    return p


def merge3(ours: str, base: str, theirs: str) -> tuple[str, int, list[str], str]:
    with tempfile.TemporaryDirectory() as t:
        names = []
        for k, v in (("ours", ours), ("base", base), ("theirs", theirs)):
            names.append(os.path.join(t, k))
            with open(names[-1], "w", encoding="utf-8", errors="surrogateescape") as h:
                h.write(v)
        p = subprocess.run(["git", "merge-file", "-p", "--diff3", "-L", "ours (the PR, renamed)", "-L", "base (renamed)",
                            "-L", "theirs", *names], capture_output=True, text=True, errors="surrogateescape")
    left, lines = [], p.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("<<<<<<< "):
            left.append(next((x for x in lines[i + 1:] if not x.startswith("|||||||")), "")[:100])
    return p.stdout, p.returncode, left, p.stderr.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("other")
    ap.add_argument("--head")
    ap.add_argument("--extra-map", action="append", default=[], metavar="COMMIT:PATH",
                    help="a map applied after the moves' own: a move whose commit has no move line")
    a = ap.parse_args()
    d = a.dir
    head = git(d, "rev-parse", a.head or "HEAD").stdout.strip()
    other = git(d, "rev-parse", a.other).stdout.strip()
    bases = git(d, "merge-base", "--all", head, other).stdout.split()
    mb = bases[0]
    for b in bases[1:]:                     # a criss-cross: the virtual base git's own merge reads, its tree
        mb = git(d, "merge-tree", "--write-tree", mb, b, ok=(0, 1)).stdout.split("\n", 1)[0]
    maps = S.move_maps(other, head)
    for x in a.extra_map:
        c, path = x.split(":", 1) if not os.path.exists(x) else ("file", x)
        maps.append((c[:12], path, S.load(S.tomllib.loads(open(x).read() if c == "file" else S.show(x)))))
    status = [s for s in git(d, "status", "--porcelain=v1", "-z").stdout.split("\0") if s]
    unmerged = sorted({s[3:] for s in status if s[:2] in ("UU", "AA", "UA", "AU", "UD", "DU")})
    junk = [f for f in unmerged if "~" in f.rsplit("/", 1)[-1]]
    report = []
    for f in (f for f in unmerged if f not in junk):
        if f.rsplit("/", 1)[-1] == "lean-audit.json":
            report.append((f, "a lock: regenerate it (check.py --update), never merge it by hand", []))
            continue
        mode = (git(d, "ls-tree", other, "--", f).stdout.split(" ") or [""])[0]
        moved = new_path(maps, f)
        if mode == "120000":
            to = os.path.normpath(os.path.join(os.path.dirname(f), git(d, "cat-file", "-p", f"{other}:{f}").stdout))
        else:
            to = moved if moved != f and git(d, "cat-file", "-e", f"{other}:{moved}", ok=(0, 128)).returncode == 0 else f
        stage = {ln.split()[2]: ln.split()[1] for ln in git(d, "ls-files", "-u", "--", f).stdout.splitlines()}
        theirs = git(d, "show", f"{other}:{to}", ok=(0, 128))
        ours = git(d, "cat-file", "blob", stage["2"]) if "2" in stage else git(d, "show", f"{head}:{f}", ok=(0, 128))
        if theirs.returncode or ours.returncode:
            report.append((f, "the owner's: deleted on one side and not moved by the maps", []))
            continue
        base = (git(d, "cat-file", "blob", stage["1"]) if "1" in stage
                else git(d, "show", f"{mb}:{f}", ok=(0, 128))).stdout
        if "\0" in ours.stdout or "\0" in base or "\0" in theirs.stdout:
            report.append((f, "the owner's: a binary file, which no map rewrites and git can't merge", []))
            continue
        text, rc, left, err = merge3(rewrite(maps, ours.stdout, to), rewrite(maps, base, to), theirs.stdout)
        if not 0 <= rc <= 127:
            report.append((f, f"the owner's: git merge-file failed: {err.splitlines()[-1] if err else f'exit {rc}'}", []))
            continue
        with open(os.path.join(d, to), "w", encoding="utf-8", errors="surrogateescape") as h:
            h.write(text)
        if to != f:
            for j in [x for x in junk if x.startswith(f + "~")]:
                git(d, "rm", "-q", "--cached", "--ignore-unmatch", "--", j)
                if os.path.lexists(os.path.join(d, j)):
                    os.remove(os.path.join(d, j))
            if mode:
                git(d, "checkout", other, "--", f)
            else:
                git(d, "rm", "-q", "--cached", "--ignore-unmatch", "--", f)
                if os.path.lexists(os.path.join(d, f)):
                    os.remove(os.path.join(d, f))
        if not left:
            git(d, "add", "-A", "--", *[p for p in dict.fromkeys([to, f]) if p == to or os.path.lexists(os.path.join(d, p))])
        report.append((f if to == f else f"{f} -> {to}", "resolved" if not left else f"{len(left)} hunks left", left))
    for f, what, left in report:
        print(f"{f}: {what}" + "".join(f"\n    left: {x}" for x in left))
    return 1 if any(w != "resolved" for _, w, _ in report) else 0


if __name__ == "__main__":
    sys.exit(main())
