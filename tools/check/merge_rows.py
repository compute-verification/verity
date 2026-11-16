#!/usr/bin/env python3
"""`merge_rows.py BASE OURS THEIRS [PATH]`: git's merge driver for Markdown (`.gitattributes`: `merge=markdown-rows`), so two
PRs that each add or change a row of one table merge.

Git's own merge runs first (`git merge-file`), and only a conflict all of whose sides are rows of a table (lines that start
and end with `|`) is merged, row by row under each row's first cell: a row either side added, changed or removed takes that
side's change, in ours' order with each of theirs' new rows below the row it followed (in the place of the row it renames,
where theirs renamed one).  Everything else stays git's conflict: a row both sides changed differently, or that one removed
and the other changed, rows the two sides both hold and hold in different orders (one side moved a row, or each moved rows
its own way: a move is a change, and taking one side's order would drop the other's untold; a move both made the same way
merges), a row one side moved and the other removed, a conflict that holds half of a move (git shows a row moved out of the
conflict's lines, or into them, as removed in one place and added in another: a row a side adds that the base's file holds
elsewhere, in any table, or drops and still holds elsewhere), a row both dropped while each brings a new one (each may have
rewritten it under a new first cell), a first cell that a side holds twice or that is elsewhere in the table too (the table
would hold it twice), a conflict that holds the row under a table's header (one side changed the table's columns, or each
added a table), and any conflict with a line that is not a row (prose, a list, a heading).
Exit 0: merged, written to OURS.  Exit 1: conflicts left, OURS holding git's text with their markers.  A merged
table is a candidate: a test that holds a table to the tree (`tests/test_benchmarks_index.py`) runs on it as on any other.

It does not judge where a row is: lines inside a fenced code block that start and end with `|` merge as rows too.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROW = re.compile(r"\s*\|.*\|\s*\n")
#: the row under a table's header (`| --- | :-: |`): a conflict holding one changes a table's columns, or adds a table
DELIMITER = re.compile(r"\s*\|(\s*:?-+:?\s*\|)+\s*\n")
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
        if len(cells) != len(side) or not all(ROW.fullmatch(s) and not DELIMITER.fullmatch(s) for s in side):
            return None
        sides.append(cells)
    base, ours, theirs = sides
    # a move is a change, and the result is in ours' order: the rows both sides hold must be in one order on both (a row
    # theirs alone moved would stay where ours has it; one ours alone moved would be kept, and lost when the same two
    # changes merge on the other branch), and a row one side moved must not be one the other removed
    if [k for k in ours if k in theirs] != [k for k in theirs if k in ours]:
        return None
    for one, other in ((ours, theirs), (theirs, ours)):
        was, now = [k for k in base if k in one], [k for k in one if k in base]
        if was != now and [k for k in was if k in other] == [k for k in now if k in other]:
            return None
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


def cells(lines: list[str]) -> set[str]:
    """The first cell of each of ``lines`` that is a row."""
    return {key(s) for s in lines if ROW.fullmatch(s if s.endswith("\n") else s + "\n")}


def moved(o: list[str], side: list[str], base: set[str], whole: set[str]) -> bool:
    """Whether a conflict holds half of a row ``side`` moved, which git shows as removed in one place and added in another:
    a row it adds to the base's ``o`` that the base's file holds elsewhere (``base``, its rows' first cells), or one it
    drops that its own file still holds (``whole``)."""
    was, now = cells(o), cells(side)
    return bool((now - was) & base or (was - now) & whole)


def beside(parts: list, i: int) -> set[str]:
    """The first cells of the rows of the table that ``parts[i]``, a conflict, lies in, outside it: those of the run of rows
    on each side of it, with every side's rows of another conflict in that run."""
    found = set()
    for step in (-1, 1):
        j = i + step
        while 0 <= j < len(parts) and (isinstance(parts[j], tuple) or ROW.fullmatch(parts[j])):
            found |= cells([parts[j]] if isinstance(parts[j], str) else [s for side in parts[j] for s in side])
            j += step
    return found


def merge(base: Path, ours: Path, theirs: Path) -> tuple[str, int]:
    """(the merged text, the number of conflicts left): git's own merge with each conflict of rows merged (`rows`), unless
    it holds half of a move (`moved`) or a first cell of its rows is elsewhere in its table too (`beside`: a row the table
    would hold twice).  A conflict sees only its own lines, so it can't weigh the other half.  When the conflicts found are
    not the ones git counts (a marker written another way), the text is git's, with all of them."""
    p = subprocess.run(["git", "merge-file", "-p", "--diff3", *(x for label in LABELS for x in ("-L", label)), str(ours),
                        str(base), str(theirs)], capture_output=True)
    if p.returncode < 0 or p.returncode > 127:
        raise ValueError(f"git merge-file failed: {p.stderr.decode(errors='replace').strip()}")
    text = p.stdout.decode("utf-8")
    in_base, in_ours, in_theirs = (cells(re.split(r"(?<=\n)", f.read_text(encoding="utf-8"))) for f in (base, ours, theirs))
    # a line outside a conflict, or a conflict's lines: (ours', the base's, theirs')
    parts, lines, n = [], re.split(r"(?<=\n)", text), 0
    while n < len(lines):
        if lines[n] != "<<<<<<< ours\n":
            parts.append(lines[n])
            n += 1
            continue
        try:
            mid = lines.index("||||||| base\n", n)
            sep = lines.index("=======\n", mid)
            end = lines.index(">>>>>>> theirs\n", sep)
        except ValueError:
            return text, max(p.returncode, 1)
        parts.append((lines[n + 1:mid], lines[mid + 1:sep], lines[sep + 1:end]))
        n = end + 1
    if sum(isinstance(part, tuple) for part in parts) != p.returncode:
        return text, p.returncode
    out, left = [], 0
    for i, part in enumerate(parts):
        if isinstance(part, str):
            out.append(part)
            continue
        a, o, b = part
        merged = rows(o, a, b)
        if (merged is None or moved(o, a, in_base, in_ours) or moved(o, b, in_base, in_theirs)
                or cells([s for side in part for s in side]) & beside(parts, i)):
            out += ["<<<<<<< ours\n", *a, "||||||| base\n", *o, "=======\n", *b, ">>>>>>> theirs\n"]
            left += 1
        else:
            out += merged
    return "".join(out), left


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
