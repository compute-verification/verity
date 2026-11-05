"""`benchmarks/README.md` indexes `benchmarks/`: every directory under it has one row, every row names one, and each row's
status is on the README's scale. A whole-tree invariant, since a new directory anywhere under `benchmarks/` must rerun it."""
from __future__ import annotations

import re

from test_repository import ROOT, _tracked

INDEX = "benchmarks/README.md"
#: The README's status scale.
STATUSES = frozenset({"maintained", "frozen", "data"})
_ROW = re.compile(r"^\| `([^`/]+)/` \| .+ \| ([a-z ]+) \|$", re.M)


def _rows() -> list[tuple[str, str]]:
    return _ROW.findall((ROOT / INDEX).read_text(encoding="utf-8"))


def test_every_benchmarks_directory_has_one_row_and_every_row_names_one() -> None:
    names = [name for name, _ in _rows()]
    directories = {p.split("/")[1] for p in _tracked() if p.startswith("benchmarks/") and p.count("/") >= 2}
    repeated = sorted({n for n in names if names.count(n) > 1})
    missing = sorted(directories - set(names))
    stale = sorted(set(names) - directories)
    assert not (repeated or missing or stale), (
        f"{INDEX} must have exactly one row per directory under benchmarks/, "
        "`| `NAME/` | what it measures | status |` in the group of what it measures:\n"
        + "".join(f"  add a row for benchmarks/{n}/\n" for n in missing)
        + "".join(f"  remove or rename the row for benchmarks/{n}/, which no longer exists\n" for n in stale)
        + "".join(f"  keep one row for benchmarks/{n}/\n" for n in repeated))


def test_every_row_has_a_status_on_the_scale() -> None:
    off = sorted(f"{n}/: {s!r}" for n, s in _rows() if s not in STATUSES)
    assert not off, f"statuses outside {INDEX}'s scale ({', '.join(sorted(STATUSES))}):\n" + "\n".join(off)
