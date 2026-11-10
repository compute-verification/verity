#!/usr/bin/env python3
"""`tools/move/refused_lines.py BASE HEAD [--map OLD=NEW]... [--json FILE]`: every line of BASE..HEAD that `move_check.py`
doesn't take as part of a pure move, for a reviewer who reads them by hand.

`move_check.py` stops at each path's first offending hunk, which is enough to refuse. A move staged through its refusals
(`stage.py --through-refusals`) lands on its own full check instead, and its reviewer needs all of them: this runs
`move_check.py`'s own `certify` with the one function that stops at the first hunk made to return every one. Nothing else
of the judgement changes, so a path it lists is a path `move_check.py` refuses, and a certified range lists nothing.

It prints one block a path (its reason, then each hunk's lines) and exits 1 when there is any, 0 when there is none.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("move_check", HERE.parents[1] / "tools/verity/check/move_check.py")
MC = importlib.util.module_from_spec(spec)
sys.modules["move_check"] = MC
spec.loader.exec_module(MC)


def every_unexplained(a, b, na, nb, pair=None, line=None, added=None, block=None, left=None):
    """`move_check.first_unexplained`, but every hunk it would stop at, one after another."""
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, na, nb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if pair and tag == "replace" and i2 - i1 == j2 - j1 and all(
                pair(x, y) or (left and left(i, y)) for i, x, y in zip(range(i1, i2), na[i1:i2], nb[j1:j2])):
            continue
        if block and tag == "replace" and block(na[i1:i2], nb[j1:j2]):
            continue
        if line and all(line(x) for x in na[i1:i2] + nb[j1:j2]):
            continue
        if added and tag == "insert" and all(added(y) for y in nb[j1:j2]):
            continue
        if pair and tag == "replace" and i2 - i1 == j2 - j1:
            # line for line: only the lines of the hunk that aren't explained
            for k, (x, y) in enumerate(zip(na[i1:i2], nb[j1:j2])):
                if not (pair(x, y) or (left and left(i1 + k, y))):
                    out.append(MC.hunk(a, b, i1 + k, i1 + k + 1, j1 + k, j1 + k + 1))
            continue
        out.append(MC.hunk(a, b, i1, i2, j1, j2))
    return "\n".join(out) or None


def refused(base: str, head: str, extra: dict[str, str] | None = None, cwd: str | None = None) -> list[dict]:
    """`move_check.certify`'s refusals for the range, each path with every hunk it refuses."""
    first, lines = MC.first_unexplained, MC.HUNK_LINES
    MC.first_unexplained, MC.HUNK_LINES = every_unexplained, 10 ** 6
    try:
        return MC.certify(base, head, extra, cwd=cwd)["refusals"]
    finally:
        MC.first_unexplained, MC.HUNK_LINES = first, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/move/refused_lines.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--map", dest="extra", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--json", type=Path, metavar="FILE", help="write the refusals here too")
    a = ap.parse_args(argv)
    extra = dict(x.split("=", 1) for x in a.extra)
    found = refused(a.base, a.head, extra)
    if a.json:
        a.json.write_text(json.dumps({"base": a.base, "head": a.head, "refusals": found}, indent=1) + "\n")
    for f in found:
        old = f" (was {f['old']})" if f.get("old") else ""
        print(f"{f['path']}{old}: {f['reason']}")
        for x in (f.get("hunk") or "").splitlines():
            print("    " + x)
    n = sum((f.get("hunk") or "").count("\n@@") + bool(f.get("hunk")) for f in found)
    print(f"{len(found)} paths, {n} hunks refused in {a.base[:12]}..{a.head[:12]}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
