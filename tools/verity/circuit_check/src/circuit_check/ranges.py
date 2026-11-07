"""A Definition registered over a range of its statics: a sidecar's ``RANGES``, a tuple of rules, each the family's id with
``lo..hi`` in place of each static it spans and the rest at the values the id spells
(``SparseMlaSplit_v1{T=1..2048,NS=1..28,NH=8,...}``).

circuit-check never checks a rule itself.  It checks the rule's spots, the targets of a run that the rule covers, which the
sidecar binds (`_model_roots` entries) so that ``--all`` and the sidecar's own run check them.  The report's ``ranges``
(`section`) gives, for each rule a target of the run falls in, its spots, the ones that failed, the ends of its ranges that no
spot reaches, and whether it holds: a spot at both ends of each range, and every spot passing.  An id a holding rule covers is
registered by range (`statuses`), which a reader keeps apart from one checked at its own statics.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import cache
from typing import Iterable

from . import cache as CC
from . import sidecars as SC
from .targets import _split

_ID = re.compile(r"([^{]+)\{(.*)\}")
_SPAN = re.compile(r"(-?\d+)\.\.(-?\d+)")


@dataclass(frozen=True)
class Rule:
    text: str
    family: str
    fixed: tuple[tuple[str, str], ...]
    over: tuple[tuple[str, int, int], ...]
    sidecar: str

    def at(self, did: str) -> dict[str, int] | None:
        """The ranged statics of Definition id ``did``, if this rule covers it; else None."""
        m = _ID.fullmatch(did.strip())
        if m is None or m[1] != self.family:
            return None
        got = _split(m[2])
        if got.keys() != {k for k, _ in self.fixed} | {k for k, _, _ in self.over}:
            return None
        if any(got[k] != v for k, v in self.fixed):
            return None
        try:
            vals = {k: int(got[k]) for k, _, _ in self.over}
        except ValueError:
            return None
        return vals if all(lo <= vals[k] <= hi for k, lo, hi in self.over) else None

    def covers(self, did: str) -> bool:
        return self.at(did) is not None

    def unreached(self, ids: Iterable[str]) -> dict[str, list[int]]:
        """{static: its range's ends that no id of ``ids`` this rule covers takes}, for each ranged static missing one."""
        seen = [v for v in map(self.at, ids) if v is not None]
        out = {k: [e for e in sorted({lo, hi}) if all(v[k] != e for v in seen)] for k, lo, hi in self.over}
        return {k: ends for k, ends in out.items() if ends}


def parse(text: str, sidecar: str = "") -> Rule:
    where = f"{sidecar}: " if sidecar else ""
    m = _ID.fullmatch(text.strip())
    if m is None:
        raise ValueError(f"{where}RANGES entry {text!r} is no id: write FAMILY_vN{{S=lo..hi,...}}")
    fixed, over = [], []
    for k, v in _split(m[2]).items():
        if s := _SPAN.fullmatch(v):
            lo, hi = int(s[1]), int(s[2])
            if lo > hi:
                raise ValueError(f"{where}RANGES entry {text!r}: {k}={v} is empty (write lo..hi with lo <= hi)")
            over.append((k, lo, hi))
        else:
            fixed.append((k, v))
    if not over:
        raise ValueError(f"{where}RANGES entry {text!r} spans no static: write a range as lo..hi, or bind the one id as a "
                         f"_model_roots entry")
    return Rule(text.strip(), m[1], tuple(fixed), tuple(over), sidecar)


@cache
def declared() -> tuple[Rule, ...]:
    """Every sidecar's ``RANGES``, read from its source as a literal tuple of strings, never by importing it; a family under two
    rules of different sidecars is an error."""
    out: list[Rule] = []
    with CC.internal():
        for files in SC.discover().values():
            if (p := files.get("py")) is None or "RANGES" not in (src := p.read_text()):
                continue
            for node in ast.parse(src).body:
                if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["RANGES"]:
                    out += [parse(text, SC.rel(p)) for text in ast.literal_eval(node.value)]
    owners: dict[str, str] = {}
    for r in out:
        if owners.setdefault(r.family, r.sidecar) != r.sidecar:
            raise ValueError(f"{r.family} has RANGES in both {owners[r.family]} and {r.sidecar}: its own module's sidecar holds them")
    return tuple(out)


def section(targets: list[dict], rules: Iterable[Rule] | None = None) -> list[dict]:
    """The report's ``ranges``: each rule some target falls in, with its spots (by id), those that failed, the ends of its ranges no
    spot reaches (``unreached``), and whether it holds."""
    out = []
    for rule in declared() if rules is None else rules:
        spots = [t for t in targets if rule.covers(str(t.get("target", "")))]
        if not spots:
            continue
        ids = [t["target"] for t in spots]
        failed = [t["target"] for t in spots if t.get("failures")]
        unreached = rule.unreached(ids)
        out.append({"rule": rule.text, "sidecar": rule.sidecar, "spots": ids, "failed": failed, "unreached": unreached,
                    "holds": not failed and not unreached})
    return out


def statuses(report: dict, ids: Iterable[str]) -> dict[str, str]:
    """How ``report`` registers each Definition id: ``checked`` (a target of it passes), ``failed`` (a target of it fails), ``by
    range`` (a rule of the report's ``ranges`` that holds covers it), ``range fails`` (only rules that don't hold cover it), else
    ``unregistered``."""
    own: dict[str, bool] = {}
    for t in report.get("targets", []):
        own[t.get("target")] = own.get(t.get("target"), False) or bool(t.get("failures"))
    rules = [(parse(r["rule"]), r["holds"]) for r in report.get("ranges", [])]
    out = {}
    for did in ids:
        if did in own:
            out[did] = "failed" if own[did] else "checked"
            continue
        held = [holds for rule, holds in rules if rule.covers(did)]
        out[did] = ("by range" if any(held) else "range fails") if held else "unregistered"
    return out


__all__ = ["Rule", "declared", "parse", "section", "statuses"]
