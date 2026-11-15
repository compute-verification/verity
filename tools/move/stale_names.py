#!/usr/bin/env python3
"""After a plain merge of `main` (the moved tree) into a PR's head, list the PR's own lines that still
spell a name one of the moves renames, and the files it adds at a path a move moves.

Run at the root of a checkout whose tools/move/ is the moved tree's (rename.py's `Words` and restack.py's readers):

    python3 tools/move/stale_names.py RESULT [--onto MAIN] [--head HEAD] [--json]
    python3 tools/move/stale_names.py --own BASE..HEAD --map COMMIT:PATH ... [--json]

The first form: RESULT is the merge; MAIN (default RESULT^2, the side merged in) holds the moves; HEAD (default RESULT^1)
is the PR's head before the merge. The moves are those MAIN has and HEAD lacks (`restack.moves_on`), oldest first, each
map read at its move commit (else its parent), as the move ran it. The own lines are what RESULT adds against MAIN
(`restack.own_lines`). Each file holding own lines is put through every move's map in order, the way its script would
have rewritten it had it run on the PR: a rename map by rename.py's own `Words.sub` (its [keep], [keep_in], [[words_in]],
`scope = "prose"` and [rewrite] exclude as the move had them), a layout map by its `paths` and `modules` read as literal
names, bounded as a path or a dotted name. An own line whose text then differs is listed, with each move that changed it.
So is an added file whose path the maps move. Each `--map` is applied after the moves' own: a map no move line names
(one not yet run, or a move commit that neither its message nor move_lines.toml gives a line).

The second form counts without a merge: own lines are what HEAD adds against BASE (a PR's own diff), the maps those named
(a map not yet on any commit: `--map FILE`), each applied once in the order given.

What it cannot see: layout.py's other rewrites (relative imports, `Path(__file__)` joins, shell anchors, Lake and
pyproject member paths), which only its own run makes; a name built from parts (`"verity/" + x`); a line the PR changed
on both sides that the merge resolved; a rename a commit made by hand, with no map (only a build finds those); a name the PR uses that a move deleted (retired) rather than renamed; and a
meaning, not a spelling (a renamed concept described in other words). A listed line may also be right as it stands (a
quoted old name, a format id the map would rename only in its own files): the owner decides each.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import restack as R
import rename as N

SKIP_DIRS = ("tools/move/",)


def show(spec: str) -> str | None:
    p = subprocess.run(["git", "show", spec], capture_output=True)
    return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None


def _glob(path: str, globs) -> bool:
    return any(fnmatch.fnmatchcase(path, g) or (g.startswith("**/") and fnmatch.fnmatchcase(path, g[3:])) for g in globs)


class LayoutNames:
    """A layout map's `paths` and `modules`, read as literal names: what the move renames in text, short of its parsers."""

    def __init__(self, data: dict):
        self.exclude = data.get("rewrite", {}).get("exclude", [])
        self.paths = {k: v for k, v in data.get("paths", {}).items() if k and v and k != v}
        self.modules = {k: v for k, v in data.get("modules", {}).items() if k and v and k != v}
        self.path_new = {o.rstrip("/"): n.rstrip("/") for o, n in self.paths.items()}
        self.path_olds = sorted(self.path_new, key=len, reverse=True)
        self.mod_olds = sorted(self.modules, key=len, reverse=True)
        self._re: dict[tuple, re.Pattern] = {}

    def _pattern(self, olds: tuple, kind: str) -> re.Pattern:
        if (olds, kind) not in self._re:
            if kind == "path":
                body = "|".join(rf"(?<![\w./-]){re.escape(o)}(?=/|[^\w.-]|\.(?!\w)|$)" for o in olds)
            else:
                body = "|".join(rf"(?<![\w.]){re.escape(m)}(?![\w])" for m in olds)
            self._re[olds, kind] = re.compile(body)
        return self._re[olds, kind]

    def sub(self, text: str, path: str) -> str:
        if _glob(path, self.exclude):
            return text
        if olds := tuple(o for o in self.path_olds if o in text):
            text = self._pattern(olds, "path").sub(lambda m: self.path_new[m.group(0)], text)
        if olds := tuple(o for o in self.mod_olds if o in text):
            text = self._pattern(olds, "module").sub(lambda m: self.modules[m.group(0)], text)
        return text

    def path(self, p: str) -> str:
        if p in self.paths:
            return self.paths[p]
        for old in sorted((o for o in self.paths if o.endswith("/")), key=len, reverse=True):
            if p.startswith(old):
                return self.paths[old] + p[len(old):]
        return p


class WordNames:
    def __init__(self, data: dict):
        self.w = N.Words(data)

    def sub(self, text: str, path: str) -> str:
        return text if self.w.excluded(path) else self.w.sub(text, path)

    def path(self, p: str) -> str:
        return p if self.w.excluded(p) else self.w.path(p)


def load(data: dict):
    if "words" in data or "words_in" in data:
        return WordNames({"words": {}, **data})
    return LayoutNames(data)


def move_maps(onto: str, head: str, moves=None) -> list[tuple[str, str, object]]:
    out = []
    for m, script, args in (moves if moves is not None else R.moves_on(onto, head, None)):
        path = R.map_of(args) or R.LAYOUT_MAP
        text = show(f"{m}:{path}") or show(f"{m}^1:{path}")
        if text is None:
            print(f"warning: {m[:12]} has no {path}", file=sys.stderr)
            continue
        out.append((m[:12], path, load(tomllib.loads(text))))
    return out


def own_lines(a: str, b: str) -> dict[str, list[tuple[int, str]]]:
    files: dict[str, list[tuple[int, str]]] = {}
    for path, n, text in R.own_lines(a, b, None):
        files.setdefault(path, []).append((n, text))
    return files


def added_files(a: str, b: str) -> list[str]:
    p = subprocess.run(["git", "diff", "--name-only", "--diff-filter=A", "--no-renames", a, b], capture_output=True, text=True)
    return p.stdout.split()


def check(result: str, base: str, maps: list[tuple[str, str, object]]) -> dict:
    files, lines, paths = own_lines(base, result), [], []
    for path, own in files.items():
        if path.startswith(SKIP_DIRS) or path.rsplit("/", 1)[-1] == R.LOCK:
            continue
        text = show(f"{result}:{path}")
        if text is None:
            continue
        before = text.splitlines()
        cur, changed_by = text, {}
        for mid, mpath, nm in maps:
            new = nm.sub(cur, path)
            if new != cur:
                a, b = cur.splitlines(), new.splitlines()
                if len(a) == len(b):
                    for i, (x, y) in enumerate(zip(a, b)):
                        if x != y:
                            changed_by.setdefault(i + 1, []).append(mid)
                else:
                    changed_by.setdefault(0, []).append(mid)
                cur = new
        after = cur.splitlines()
        whole = 0 in changed_by
        for n, t in own:
            if n - 1 >= len(before):
                continue
            if whole:
                fin = t
                for _, _, nm in maps:
                    fin = nm.sub(fin, path)
                if fin != t:
                    lines.append({"file": path, "line": n, "text": t.strip()[:200], "would_be": fin.strip()[:200],
                                  "moves": changed_by[0], "old_full": t, "new_full": fin})
            elif n in changed_by and len(after) == len(before):
                lines.append({"file": path, "line": n, "text": t.strip()[:200], "would_be": after[n - 1].strip()[:200],
                              "moves": changed_by[n], "old_full": before[n - 1], "new_full": after[n - 1]})
    for f in added_files(base, result):
        if f.startswith(SKIP_DIRS):
            continue
        p = f
        for _, _, nm in maps:
            p = nm.path(p)
        if p != f:
            paths.append({"file": f, "would_be": p})
    return {"lines": lines, "paths": paths, "files_read": len(files), "maps": len(maps)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", nargs="?")
    ap.add_argument("--onto")
    ap.add_argument("--head")
    ap.add_argument("--own", help="BASE..HEAD: count a PR's own lines against the named maps, without a merge")
    ap.add_argument("--map", action="append", default=[], help="COMMIT:PATH or a file")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--shown", type=int, default=40)
    ap.add_argument("--write", metavar="DIR", help="a checkout of RESULT: rewrite each listed own line there as the maps "
                    "would have (not the `stale-words` ones, which need a reading), and move each listed added file")
    a = ap.parse_args()
    if a.own:
        base, head = a.own.split("..")
        maps = []
        for spec in a.map:
            text = show(spec) if ":" in spec else open(spec).read()
            maps.append((spec.rsplit("/", 1)[-1], spec, load(tomllib.loads(text))))
        r = check(head, base, maps)
    else:
        if not a.result:
            ap.error("name RESULT, the merge (or --own BASE..HEAD)")
        if not a.onto and subprocess.run(["git", "rev-parse", "-q", "--verify", f"{a.result}^2"],
                                         capture_output=True).returncode:
            raise SystemExit(f"{a.result} is no merge: commit the merge first (git commit --no-edit, once no conflict "
                             "is left), or name --onto and --head")
        onto = R.rev(a.onto or f"{a.result}^2", None)
        head = R.rev(a.head or f"{a.result}^1", None)
        result = R.rev(a.result, None)
        moves = R.moves_on(onto, head, None)
        extra = [(spec.rsplit("/", 1)[-1], spec, load(tomllib.loads(show(spec) if ":" in spec else open(spec).read())))
                 for spec in a.map]
        r = check(result, onto, move_maps(onto, head, moves) + extra)
        r["scoped"] = [{**w, "move": w["move"][:12]} for w in R.stale_words(result, onto, moves, None)]
    if a.write:
        for f in {x["file"] for x in r["lines"]}:
            p = os.path.join(a.write, f)
            text = open(p, encoding="utf-8").read().split("\n")
            for x in r["lines"]:
                if x["file"] == f and text[x["line"] - 1] == x["old_full"]:
                    text[x["line"] - 1] = x["new_full"]
            open(p, "w", encoding="utf-8").write("\n".join(text))
        for x in r["paths"]:
            os.makedirs(os.path.dirname(os.path.join(a.write, x["would_be"])), exist_ok=True)
            subprocess.run(["git", "-C", a.write, "mv", x["file"], x["would_be"]], check=True)
    if a.json:
        print(json.dumps(r))
    else:
        print(f"{len(r['lines'])} own lines still spell a name the maps rename; {len(r['paths'])} added files at a path "
              f"they move ({r['maps']} maps, {r['files_read']} files with own lines)")
        for x in r["paths"][:a.shown]:
            print(f"  file {x['file']} -> {x['would_be']}")
        for x in r["lines"][:a.shown]:
            print(f"  {x['file']}:{x['line']} ({', '.join(x['moves'][:3])}): {x['text'][:110]}\n      -> {x['would_be'][:110]}")
        if len(r["lines"]) > a.shown:
            print(f"  and {len(r['lines']) - a.shown} more (--json lists every one)")
        if r.get("scoped"):
            print(f"{len(r['scoped'])} own lines spell a word a move renames only in other files (restack.py's "
                  "`stale-words`): rename each that means the renamed name")
            for x in r["scoped"][:a.shown]:
                print(f"  {x['file']}:{x['line']}: {x['word']} -> {x['new']} ({x['map']})")
    return 1 if r["lines"] or r["paths"] or r.get("scoped") else 0


if __name__ == "__main__":
    sys.exit(main())
