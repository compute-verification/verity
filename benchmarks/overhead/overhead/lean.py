"""Where a citation stands, read from the locks: what a bound's or a cut's status is, rather than what it says.

`status(name)` is `proved` when a lock lists a proof of the statement `name` (a guarantee whose signature ends in
`: name`), `assumption` when a lock lists `name` as a named assumption, `draft` for a paper statement (`draft:` ids, not
yet in Lean), `evidence` for a run or a finding (`art:`, `note:`), and `absent` for a Lean name no lock lists. An
argument is only as good as its weakest citation (`weakest`; evidence doesn't weaken it).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCKS = (ROOT / "Security" / "lean-audit.json", ROOT / "Security" / "Properties" / "lean-audit.json")

ORDER = ("proved", "assumption", "draft", "absent")


@cache
def _locks() -> tuple[frozenset[str], frozenset[str]]:
    proved: set[str] = set()
    assumed: set[str] = set()
    for path in LOCKS:
        lock = json.loads(path.read_text())
        for entry in lock["guarantees"].values():
            proved.update(s.rsplit(" : ", 1)[-1] for s in entry["signature"])
        assumed.update(lock["assumptions"])
    return frozenset(proved), frozenset(assumed)


def status(name: str) -> str:
    if name.startswith("draft:"):
        return "draft"
    if name.startswith(("art:", "note:", "ruling:")):
        return "evidence"
    proved, assumed = _locks()
    if name in proved:
        return "proved"
    if name in assumed:
        return "assumption"
    return "absent"


def weakest(names: Iterable[str]) -> str:
    """`proved`, `conditional` (proved under named assumptions), `draft` or `absent`: the weakest of `names`."""
    s = max((s for s in map(status, names) if s != "evidence"), key=ORDER.index, default="proved")
    return "conditional" if s == "assumption" else s
