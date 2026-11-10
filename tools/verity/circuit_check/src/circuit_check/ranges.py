"""A Definition registered over a range of its statics: a sidecar's ``RANGES``, {rule: its spots}.  A rule is the family's id with
``lo..hi`` in place of each static it spans, or ``lo..hi step s`` for lo, lo + s, ..., hi (a grid of size classes,
``N=4096..131072 step 2048``), and the rest at the values the id spells (``SparseMlaSplit_v1{T=1..2048,NS=1..28,NH=8,...}``); a
spot gives each spanned static a value (``"T=2048,NS=28"``), and the spots reach both ends of each range (`parse` refuses a rule
whose spots don't).

circuit-check never checks a rule itself.  ``--ranges`` adds the rules' spots to a run's targets (`spots`), as named targets,
so a spot past `targets.MAX_GATES` is checked too (``--all`` keeps only its families' smaller bindings).  The report's ``ranges``
(`section`) gives, for each rule a spot of which the run checked, the spots it didn't check and those that failed: the rule
holds when the run checked every spot and each passed.  An id a holding rule covers is registered by range (`statuses`), which
a reader keeps apart from one checked at its own statics.
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
_SPAN = re.compile(r"(-?\d+)\.\.(-?\d+)(?: step (\d+))?")
_INT = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Rule:
    text: str
    family: str
    order: tuple[str, ...]
    fixed: tuple[tuple[str, str], ...]
    over: tuple[tuple[str, int, int, int], ...]
    spots: tuple[str, ...]
    sidecar: str

    def at(self, did: str) -> dict[str, int] | None:
        """The spanned statics of Definition id ``did``, if this rule covers it; else None."""
        m = _ID.fullmatch(did.strip())
        if m is None or m[1] != self.family:
            return None
        got = _split(m[2])
        if (tuple(got) != self.order or any(got[k] != v for k, v in self.fixed)
                or not all(_INT.fullmatch(got[k]) for k, *_ in self.over)):
            return None
        vals = {k: int(got[k]) for k, *_ in self.over}
        return vals if all(lo <= vals[k] <= hi and (vals[k] - lo) % step == 0 for k, lo, hi, step in self.over) else None

    def covers(self, did: str) -> bool:
        return self.at(did) is not None

    def unreached(self, ids: Iterable[str]) -> dict[str, list[int]]:
        """{static: its range's ends that no id of ``ids`` this rule covers takes}, for each spanned static missing one."""
        seen = [v for v in map(self.at, ids) if v is not None]
        out = {k: [e for e in sorted({lo, hi}) if all(v[k] != e for v in seen)] for k, lo, hi, _ in self.over}
        return {k: ends for k, ends in out.items() if ends}


def _rule(text: str, where: str = "") -> Rule:
    m = _ID.fullmatch(text.strip())
    if m is None:
        raise ValueError(f"{where}RANGES rule {text!r} is no id: write FAMILY_vN{{S=lo..hi,...}}")
    statics = _split(m[2])
    fixed, over = [], []
    for k, v in statics.items():
        if s := _SPAN.fullmatch(v):
            lo, hi, step = int(s[1]), int(s[2]), int(s[3] or 1)
            if lo > hi:
                raise ValueError(f"{where}RANGES rule {text!r}: {k}={v} is empty (write lo..hi with lo <= hi)")
            if step < 1 or (hi - lo) % step:
                raise ValueError(f"{where}RANGES rule {text!r}: {k}={v} doesn't reach hi from lo (write lo..hi step s with s >= 1 "
                                 f"dividing hi - lo)")
            over.append((k, lo, hi, step))
        else:
            fixed.append((k, v))
    if not over:
        raise ValueError(f"{where}RANGES rule {text!r} spans no static: write a range as lo..hi, or bind the one id as a "
                         f"_model_roots entry")
    return Rule(text.strip(), m[1], tuple(statics), tuple(fixed), tuple(over), (), where.rstrip(": "))


def parse(text: str, spots: Iterable[str], sidecar: str = "") -> Rule:
    """Rule ``text`` with ``spots``, each made the family's id at the rule's fixed statics."""
    where = f"{sidecar}: " if sidecar else ""
    rule = _rule(text, where)
    spanned = [k for k, *_ in rule.over]
    ids = []
    for spot in spots:
        got = _split(spot)
        if set(got) != set(spanned) or not all(_INT.fullmatch(got[k]) for k in spanned):
            raise ValueError(f"{where}RANGES rule {rule.text}: spot {spot!r} is not an integer for each of {spanned}")
        fixed = dict(rule.fixed)
        did = f"{rule.family}{{{','.join(f'{k}={int(got[k])}' if k in got else f'{k}={fixed[k]}' for k in rule.order)}}}"
        if not rule.covers(did):
            raise ValueError(f"{where}RANGES rule {rule.text}: spot {spot!r} is outside its ranges")
        if did in ids:
            raise ValueError(f"{where}RANGES rule {rule.text}: spot {spot!r} is listed twice")
        ids.append(did)
    rule = Rule(rule.text, rule.family, rule.order, rule.fixed, rule.over, tuple(ids), sidecar)
    if miss := rule.unreached(ids):
        raise ValueError(f"{where}RANGES rule {rule.text}: no spot at "
                         + ", ".join(f"{k}={e}" for k, ends in miss.items() for e in ends) + ": add one at each end of each range")
    return rule


@cache
def declared() -> tuple[Rule, ...]:
    """Every sidecar's ``RANGES``, read from its source as a literal dict, never by importing it; a family under two rules is an
    error."""
    out: list[Rule] = []
    with CC.internal():
        for files in SC.discover().values():
            if (p := files.get("py")) is None or "RANGES" not in (src := p.read_text()):
                continue
            for node in ast.parse(src).body:
                if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["RANGES"]:
                    out += [parse(text, spots, SC.rel(p)) for text, spots in ast.literal_eval(node.value).items()]
    seen: dict[str, str] = {}
    for r in out:
        if (other := seen.setdefault(r.family, r.text)) != r.text:
            raise ValueError(f"{r.family} has two RANGES rules, {other} and {r.text} ({r.sidecar}): make them one")
    return tuple(out)


def spots(families: Iterable[str] = ()) -> list[str]:
    """``--ranges``' targets: the spots of each declared rule of ``families`` (every rule's when empty)."""
    rules, want = declared(), list(families)
    if unknown := sorted(set(want) - {r.family for r in rules}):
        raise LookupError(f"no RANGES rule for {', '.join(unknown)} (the rules: {', '.join(r.family for r in rules) or 'none'})")
    return [s for r in rules if not want or r.family in want for s in r.spots]


def section(targets: list[dict], rules: Iterable[Rule] | None = None) -> list[dict]:
    """The report's ``ranges``: each rule a spot of which is among ``targets``, with its spots, those not among them
    (``missing``), those that failed, whether it holds, and {spot: the checks it passed without running} (``unchecked``: a
    warning of code ``unchecked``, such as the partition over its budget of a spot that is no Call, which fails nothing, as
    circuit-check's own verdict doesn't; a Call's fails, ``partition/unpartitioned``)."""
    out = []
    for rule in declared() if rules is None else rules:
        seen: dict[str, bool] = {}
        skipped: dict[str, set[str]] = {}
        for t in targets:
            if (did := t.get("target")) in rule.spots:
                seen[did] = seen.get(did, False) or bool(t.get("failures"))
                skipped.setdefault(did, set()).update(w["check"] for w in t.get("warnings", []) if w.get("code") == "unchecked")
        if not seen:
            continue
        missing = [s for s in rule.spots if s not in seen]
        failed = [s for s in rule.spots if seen.get(s)]
        out.append({"rule": rule.text, "sidecar": rule.sidecar, "spots": list(rule.spots), "missing": missing, "failed": failed,
                    "holds": not missing and not failed,
                    "unchecked": {s: sorted(skipped[s]) for s in rule.spots if skipped.get(s)}})
    return out


def statuses(report: dict, ids: Iterable[str]) -> dict[str, str]:
    """How ``report`` registers each Definition id: ``checked`` (a target of it passes), ``failed`` (a target of it fails), ``by
    range`` (a rule of the report's ``ranges`` that holds covers it), ``range fails`` (only rules that don't hold cover it), else
    ``unregistered``."""
    own: dict[str, bool] = {}
    for t in report.get("targets", []):
        own[t.get("target")] = own.get(t.get("target"), False) or bool(t.get("failures"))
    rules = [(_rule(r["rule"]), r["holds"]) for r in report.get("ranges", [])]
    out = {}
    for did in ids:
        if did in own:
            out[did] = "failed" if own[did] else "checked"
            continue
        held = [holds for rule, holds in rules if rule.covers(did)]
        out[did] = ("by range" if any(held) else "range fails") if held else "unregistered"
    return out


__all__ = ["Rule", "declared", "parse", "section", "spots", "statuses"]
