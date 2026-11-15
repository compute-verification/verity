#!/usr/bin/env python3
"""`tools/move/gone_dirs.py MAP... [--ref REF] [--deleted FILE...] [--json FILE] [--all]`: the references that name a
directory no longer there.

Reads the maps' rows (`layout.py` maps, applied in the order given) and REF's tree (default `origin/main`), moves
nothing, and counts, in every tracked text file the maps rewrite, each root-relative path that names a directory the
maps empty and that no map rewrites. `layout.py` looks only for paths that begin with a row's key or a `[tokens]` key,
so a directory emptied by rows for the files and directories inside it, with no row and no `[tokens]` row of its own, is
neither rewritten nor noted: a suite's declared input, a boundary's glob or a sentence goes on naming it.

A reference counts when it is the directory itself or a path below it that names nothing in REF's tree (a glob, a file
that is not there); a path to something inside it is that thing's reference, which its own row rewrites. It exits 1
when the count is above zero, and prints for each directory its count, where its files go and each `file:line`.
It also fails a `[tokens]` row whose new directory holds nothing after the maps.

`--deleted FILE...` names files the maps leave that a commit after them deletes (the `__init__.py` of a package level
the maps empty), by their paths today: a directory that holds nothing else is then gone too, and the count is the one
that holds once that commit has landed.

Not seen: a directory named with no `/` (one segment), a path relative to the file that holds it, a path put together
at run time, and a module name (`[modules]` rewrites those)."""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout as L
from moved import Moved

#: what may follow a directory's name in a path below it: names, globs, brace lists and shell or template variables
REST = r"(?:/(?:[\w.\-*?$]|\{[^{}\s]*\}|\[[^\[\]\s]*\])*)*"


def check(m: Moved) -> dict:
    gone = sorted(d for d in m.old_dirs - m.new_dirs if m.directory(d) is None)
    one_segment = [d for d in gone if "/" not in d]
    gone = [d for d in gone if "/" in d]
    refs: dict[str, list[str]] = defaultdict(list)
    if gone:
        names = "|".join(re.escape(d) for d in sorted(gone, key=len, reverse=True))
        pat = re.compile(rf"(?<![\w./-])(?:\./)?({names})(?![\w-])(?!\.\w)({REST})")
        for p in m.files:
            if m.excluded(p) or p == L.MOVES or (text := m.read(p)) is None or len(text) > L.TEXT_LIMIT:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                for x in pat.finditer(line):
                    first = x.group(2).strip("/").split("/")[0].rstrip(".")        # a sentence's full stop is not the name's
                    below = f"{x.group(1)}/{first}"
                    if first and (below in m.old_dirs or below in m.tracked):
                        continue                        # a path to something inside it: that thing's own reference
                    refs[x.group(1)].append(f"{p}:{n}")
    goes = {}
    for d in gone:
        tops = Counter()
        for f in m.files:
            if f.startswith(d + "/"):
                rest = f[len(d):]
                new = m.new[f]
                tops[new[: -len(rest)] if new.endswith(rest) else posixpath.dirname(new)] += 1
        goes[d] = [f"{t} ({n})" for t, n in tops.most_common()]
    empty = sorted({f"{o}/ = {n}/" for lay in m.layouts for o, n in lay.tokens.items()
                    if o in m.old_dirs and n not in m.new_dirs})
    return {"count": sum(len(v) for v in refs.values()),
            "directories": [{"directory": d, "references": len(refs[d]), "files_go_to": goes[d], "where": refs[d]}
                            for d in sorted(refs, key=lambda d: (-len(refs[d]), d))],
            "gone_unreferenced": [d for d in gone if d not in refs], "one_segment": one_segment, "tokens_to_nothing": empty}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("maps", nargs="+", help="layout.py maps, in the order they run")
    ap.add_argument("--ref", default="origin/main", help="the commit whose tree the maps move (default: origin/main)")
    ap.add_argument("--deleted", nargs="*", default=[], metavar="FILE", help="files a commit after the maps deletes")
    ap.add_argument("--json", type=Path, metavar="FILE", help="write the count and every reference here")
    ap.add_argument("--all", action="store_true", help="print every file:line (default: five a directory)")
    a = ap.parse_args(argv)
    m = Moved(a.maps, a.ref, deleted=tuple(a.deleted))
    if unknown := sorted(set(a.deleted) - m.tracked):
        sys.exit(f"--deleted names files {m.ref[:12]} does not hold: {', '.join(unknown)}")
    try:
        r = check(m)
    finally:
        m.close()
    if a.json:
        a.json.write_text(json.dumps({"ref": m.ref, "maps": a.maps, "deleted": a.deleted, **r}, indent=1) + "\n")
    print(f"gone directories of {len(a.maps)} map(s) on {m.ref[:12]}"
          + (f", {len(a.deleted)} more files deleted" if a.deleted else "")
          + f": {r['count']} references name {len(r['directories'])} directories that are gone and that no row or [tokens] row "
          f"rewrites")
    for d in r["directories"]:
        print(f"  {d['directory']}/: {d['references']} references; its files go to {', '.join(d['files_go_to'][:4])}"
              + (f" and {len(d['files_go_to']) - 4} more" if len(d["files_go_to"]) > 4 else ""))
        for w in d["where"] if a.all else d["where"][:5]:
            print(f"    {w}")
        if not a.all and len(d["where"]) > 5:
            print(f"    and {len(d['where']) - 5} more (--all or --json lists them)")
    for t in r["tokens_to_nothing"]:
        print(f"  [tokens] {t}: the new directory holds nothing after the maps")
    if r["one_segment"]:
        print(f"  not counted, one segment: {', '.join(r['one_segment'])}")
    return 1 if r["count"] or r["tokens_to_nothing"] else 0


if __name__ == "__main__":
    sys.exit(main())
