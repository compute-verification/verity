"""Where a citation stands, read from the locks: what a bound's or a cut's status is, rather than what it says.

`status(name)` is `proved` when a lock lists a proof of the statement `name` (a guarantee whose signature ends in
`: name`), `assumption` when a lock lists `name` as a premise (a named assumption, an open one or a domain hypothesis)
or holds it under `reads` in an assumptions module (one with an `Assumptions` part, which statements read but none takes),
`draft` for a paper statement (`draft:` ids, not yet in Lean), `evidence` for a run or a finding (`art:`, `note:`), and
`absent` for a Lean name no lock lists. An argument is only as good as its weakest citation (`weakest`; evidence doesn't
weaken it).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCKS = (ROOT / "security" / "lean-audit.json", ROOT / "security" / "Properties" / "lean-audit.json")

ORDER = ("proved", "assumption", "draft", "absent")


@cache
def read(locks: tuple[Path, ...] = LOCKS) -> tuple[frozenset[str], frozenset[str]]:
    """(the statements ``locks`` list a proof of, the names they list as premises), as `status` reads them."""
    proved: set[str] = set()
    assumed: set[str] = set()
    for path in locks:
        lock = json.loads(path.read_text())
        for entry in lock["guarantees"].values():
            proved.update(s.rsplit(" : ", 1)[-1] for s in entry.get("signature", ()))
        assumed.update(lock["assumptions"])
        assumed.update(lock.get("open") or {})
        assumed.update(lock.get("domain") or {})
        for module, group in (lock.get("reads") or {}).items():
            if "Assumptions" in module.split("."):
                assumed.update(group.get("definitions") or {})
    return frozenset(proved), frozenset(assumed)


def status(name: str) -> str:
    if name.startswith("draft:"):
        return "draft"
    if name.startswith(("art:", "note:", "ruling:")):
        return "evidence"
    proved, assumed = read()
    if name in proved:
        return "proved"
    if name in assumed:
        return "assumption"
    return "absent"


@cache
def premise_kind(name: str, locks: tuple[Path, ...] = LOCKS) -> str:
    """The lock section a premise is listed in: `assumptions` (a named assumption), `open` or `domain`; `absent` where
    no lock lists it."""
    for path in locks:
        lock = json.loads(path.read_text())
        for section in ("assumptions", "open", "domain"):
            if name in (lock.get(section) or {}):
                return section
    return "absent"


def weakest(names: Iterable[str]) -> str:
    """`proved`, `conditional` (proved under named assumptions), `draft` or `absent`: the weakest of `names`."""
    s = max((s for s in map(status, names) if s != "evidence"), key=ORDER.index, default="proved")
    return "conditional" if s == "assumption" else s
