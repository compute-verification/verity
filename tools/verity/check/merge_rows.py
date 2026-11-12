#!/usr/bin/env python3
"""`merge_rows.py BASE OURS THEIRS [PATH]`: git's merge driver for Markdown (`.gitattributes`: `merge=markdown-rows`), so two
PRs that each add or change a row of one table merge.

Git's own merge runs first (`git merge-file`), and only a conflict all of whose sides are rows of a table (lines that
start and end with `|`) is merged, row by row under each row's first cell: a row either side added, changed or removed takes
that side's change, in ours' order with each of theirs' new rows below the row it followed (in the place of the row it
renames, where theirs renamed one).  Everything else stays git's conflict: a row both sides changed differently, or that one
removed and the other changed, a row both dropped while each brings a new one (each may have rewritten it under a new
first cell), a first cell that a side holds twice, and any conflict with a line that is not a row (prose, a list, a
heading).  Exit 0: merged, written to OURS.  Exit 1: conflicts left, OURS holding git's text with their markers.  A merged
table is a candidate: a test that holds a table to the tree (`tests/test_benchmarks_index.py`) runs on it as on any other.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROW = re.compile(r"\s*\|.*\|\s*\n")
LABELS = ("ours", "base", "theirs")


def key(row: str) -> str:
    """A row's first cell."""
    return row.strip()[1:].split("|", 1)[0].strip()


def rows(o: list[str], a: list[str], b: list[str]) -> list[str] | None:
    """The rows of a conflict whose sides are ``o`` (the base's lines), ``a`` (ours') and ``b`` (theirs'), merged under each
    row's first cell, or None to leave the conflict to git."""
    sides = []
    for side in (o, a, b):
        cells = {key(s): s for s in side}
        if len(cells) != len(side) or not all(ROW.fullmatch(s) for s in side):
            return None
        sides.append(cells)
    base, ours, theirs = sides
    # a row both sides dropped while each brings one the other lacks: each may have rewritten it under a new first cell
    if (any(k not in ours and k not in theirs for k in base) and any(k not in base and k not in theirs for k in ours)
            and any(k not in base and k not in ours for k in theirs)):
        return None
    out = {}
    for k in [*ours, *(k for k in theirs if k not in ours)]:
        x, y, z = base.get(k), ours.get(k), theirs.get(k)
        if y != z and x not in (y, z):
            return None
        if (row := y if z == x else z) is not None:
            out[k] = row
    # ours' rows, those theirs removed still among them to place theirs' new rows by
    order, old, new = list(ours), list(base), list(theirs)
    for i, k in enumerate(new):
        if k in out and k not in ours:
            renamed = len(old) == len(new) and old[i] not in theirs and old[i] in order
            above = old[i] if renamed else next((j for j in reversed(new[:i]) if j in order), None)
            below = next((j for j in new[i + 1:] if j in order), None)
            at = order.index(above) + 1 if above is not None else order.index(below) if below is not None else len(order)
            order.insert(at, k)
    return [out[k] for k in order if k in out]


def merge(base: Path, ours: Path, theirs: Path) -> tuple[str, int]:
    """(the merged text, the number of conflicts left): git's own merge with each conflict of rows merged (`rows`).  When
    the conflicts found are not the ones git counts (a marker written another way), the text is git's, with all of them."""
    p = subprocess.run(["git", "merge-file", "-p", "--diff3", *(x for label in LABELS for x in ("-L", label)), str(ours),
                        str(base), str(theirs)], capture_output=True)
    if p.returncode < 0 or p.returncode > 127:
        raise ValueError(f"git merge-file failed: {p.stderr.decode(errors='replace').strip()}")
    text = p.stdout.decode("utf-8")
    out, left, found, lines, n = [], 0, 0, re.split(r"(?<=\n)", text), 0
    while n < len(lines):
        if lines[n] != "<<<<<<< ours\n":
            out.append(lines[n])
            n += 1
            continue
        try:
            mid = lines.index("||||||| base\n", n)
            sep = lines.index("=======\n", mid)
            end = lines.index(">>>>>>> theirs\n", sep)
        except ValueError:
            return text, max(p.returncode, 1)
        merged = rows(lines[mid + 1:sep], lines[n + 1:mid], lines[sep + 1:end])
        if merged is None:
            out += lines[n:end + 1]
            left += 1
        else:
            out += merged
        found, n = found + 1, end + 1
    return ("".join(out), left) if found == p.returncode else (text, p.returncode)


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    base, ours, theirs = (Path(p) for p in argv[:3])
    path = argv[3] if len(argv) == 4 else ours
    try:
        text, left = merge(base, ours, theirs)
    except (ValueError, OSError) as e:
        print(f"merge_rows.py: {path}: left as it was: {e}", file=sys.stderr)
        return 1
    ours.write_bytes(text.encode("utf-8"))
    if left:
        print(f"merge_rows.py: {path}: {left} conflict(s) left: not rows of one table that each side changed apart",
              file=sys.stderr)
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
