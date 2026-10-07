"""The draw: the verifier's own coins pick the units to prove, sent in the clear after registration and before the session.

    {"law": "subset", "population": N, "k": K, "units": [<ascending unit indices>]}
    {"law": "bernoulli", "population": N, "p": "NUM/DEN", "units": [...]}
    {"law": "stratified", "population": N, "k": K, "strata": [<stratum>, ...], "units": [...]}
    {"law": "work", "population": N, "k": K, "strata": [<work stratum>, ...], "units": [...], ["closure": [...]]}

This is the Lean verifier's `Flock.Draw` object and M0's `unit_draw`. The verifier samples it from the operating system's
randomness (Lean `flock-verify draw`, or M0's live `serve --draw`); nothing here samples, and no draw is derived from a
commitment. `check_draw` holds a draw to the registration and to the verifier's own law, and `check_proved` holds the
proved statement's units to the draw.

**The stratified law** (Daniel, Sep 27; `docs/sampling-strategies.md`): one stratum per template, a uniform k_s-subset drawn
without replacement inside each, independently.
* **Strata** are derived by each verifier from its own program and query (`partition.strata`): one per template descriptor
  id, its units as ascending, disjoint half-open ranges `[lo, hi)`, the strata in the order of their first unit. Defined for
  `Q_template_instance` and `Q_template_instances`; another query refuses it.
* **Sizes**: k_s = max(1, round(k n_s / n)), rounding half up in integers (`stratum_k`), so every template has a draw.
* **The law object** is `{"law": "stratified", "k": K, "strata": [{"template": <descriptor id>, "units": [[lo, hi], ...],
  "k": k_s}, ...]}`: the agreement's rule `stratified:K` expanded by `derive`. R1 holds the registered law to the verifier's
  own expansion, and the draw carries the law's keys plus `population` and `units`.

**The work law** (Daniel approved work-proportional sizing, Sep 28; the Lean verifier's `Flock.Draw.Law.work`, its bound
`FlockSoundness.Audit.Work.work_escape_le`): the stratified law's strata and sampler, with each stratum's draw sized by the
PoUW work its units do. The verifier's work table gives each template's work per unit, a natural number in any fixed unit,
and its floor f_s >= 1 (`work_table`). With W = sum_s w_s n_s >= 1, stratum s draws
k_s = min(n_s, max(f_s, ceil(K w_s n_s / W))) (`work_k`), and a work stratum is `{"template", "units", "work", "floor",
"k"}`. A work draw may carry a `closure`: units proved with the drawn ones, none of them drawn.
* **What the table credits.** A unit's work is what it is credited for, counted on the rows the Program declares, never on
  rows a kernel pads it with (the Pearl-C statement review, item 5). Filler is credited nothing: a unit of filler only, or
  one the prover declares uncredited before the draw, does work 0 (its floor still draws it for integrity, and missing it
  costs no work), and a unit mixing real and filler rows is credited its real cells only. The law weights by template, so
  every unit of a template must be credited the same work: a template whose units are padded differently is split before
  the law is derived, or credited 0. The escape bound needs each unit's inclusion probability to be at least K w_u / W
  over the credited W, which a per-template work that over-credits any unit breaks.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program

from . import partition as P
from .verdict import Verdict, _Collect

_STRATUM_KEYS = {"template", "units", "k"}
_WORK_STRATUM_KEYS = {"template", "units", "work", "floor", "k"}


def parse_law(text: str) -> dict[str, Any]:
    """`subset:K` or `bernoulli:NUM/DEN` (M0's `serve --draw` syntax), or `stratified:K` or `work:K` (rules, which `derive`
    expands over a program and query, and for `work` a work table), as a law object."""
    kind, _, arg = text.partition(":")
    if kind == "subset" and arg.isdigit():
        return {"law": "subset", "k": int(arg)}
    if kind in ("stratified", "work") and arg.isdigit() and int(arg) >= 1:
        return {"law": kind, "k": int(arg)}
    if kind == "bernoulli":
        num, _, den = arg.partition("/")
        if num.isdigit() and den.isdigit() and 0 < int(den) and int(num) <= int(den):
            return {"law": "bernoulli", "p": f"{int(num)}/{int(den)}"}
    raise ValueError(f"a law is subset:K, stratified:K, work:K or bernoulli:NUM/DEN, not {text!r}")


def stratum_k(k: int, size: int, population: int) -> int:
    """A stratum's draw under the stratified rule: max(1, round(k * size / population)), rounding half up."""
    return max(1, (2 * k * size + population) // (2 * population))


def work_k(k: int, work: int, size: int, total: int, floor: int) -> int:
    """A stratum's draw under the work rule (Lean `Flock.Draw.workK`): ceil(k * work * size / total), at least `floor` and
    at most `size`."""
    return min(size, max(floor, (k * work * size + total - 1) // total))


def work_table(table: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """The verifier's work table (Lean `workTableOfJson`): template descriptor id -> the credited work of one of its units
    (a natural number, floor one) or `{"work": w, "floor": f}` with f >= 1; as `{template: (work, floor)}`. Raises
    `ValueError` naming the first bad entry. See the module docstring for what a unit's work may count."""
    if not isinstance(table, Mapping):
        raise ValueError("a work table is an object from template id to work per unit")
    out: dict[str, tuple[int, int]] = {}
    for t, v in table.items():
        if not isinstance(t, str):
            raise ValueError("a work table is an object from template id to work per unit")
        if _nat(v):
            out[t] = (v, 1)
        elif isinstance(v, Mapping):
            if set(v) - {"work", "floor"} or not _nat(v.get("work")):
                raise ValueError(f"work table: template {t!r}: neither a work nor an object of work and floor")
            f = v.get("floor", 1)
            if not _nat(f):
                raise ValueError(f"work table: template {t!r}: neither a work nor an object of work and floor")
            if f < 1:
                raise ValueError(f"work table: template {t!r}: a floor is at least one")
            out[t] = (v["work"], f)
        else:
            raise ValueError(f"work table: template {t!r}: the work is not a natural number")
    return out


def _size(ranges: Sequence[Sequence[int]]) -> int:
    return sum(hi - lo for lo, hi in ranges)


def total_work(strata: Sequence[Mapping[str, Any]]) -> int:
    """A work law's population work, W = sum_s w_s n_s."""
    return sum(s["work"] * _size(s["units"]) for s in strata)


def derive(program: Program, query: Mapping[str, Any], law: Mapping[str, Any],
           table: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The verifier's law for its program and query: `stratified` (the rule, or an expanded law, whose strata are ignored)
    expands to the strata `partition.strata` derives, each with `stratum_k`; `work` expands to the same strata with the
    work table's works and floors (`work_table`), each with `work_k` (Lean `Flock.Draw.deriveWork`); `subset` and
    `bernoulli` are unchanged. Raises `partition.QueryRefused` when the query doesn't apply, and `ValueError` when k isn't
    in [1, population], a work law has no table, the table names no work for a template, or the population does none."""
    kind = law.get("law")
    if kind not in ("stratified", "work"):
        return dict(law)
    strata = P.strata(program, query)
    n = sum(hi - lo for _, rs in strata for lo, hi in rs)
    k = law.get("k")
    if not _int(k) or not 1 <= k <= n:
        raise ValueError(f"a {kind} law needs 1 <= k <= {n}, not {k!r}")
    if kind == "stratified":
        return {"law": "stratified", "k": k,
                "strata": [{"template": t, "units": [list(r) for r in rs], "k": stratum_k(k, _size(rs), n)}
                           for t, rs in strata]}
    if table is None:
        raise ValueError("a work law needs the verifier's work table")
    works = work_table(table)
    missing = [t for t, _ in strata if t not in works]
    if missing:
        raise ValueError(f"the work table names no work for template {missing[0]!r}")
    out = [{"template": t, "units": [list(r) for r in rs], "work": works[t][0], "floor": works[t][1]} for t, rs in strata]
    total = total_work(out)
    if total == 0:
        raise ValueError("a work law needs a population that does some work")
    for s in out:
        s["k"] = work_k(k, s["work"], _size(s["units"]), total, s["floor"])
    return {"law": "work", "k": k, "strata": out}


def law_of(draw: Mapping[str, Any]) -> dict[str, Any]:
    if draw.get("law") in ("stratified", "work"):
        return {"law": draw["law"], "k": draw.get("k"), "strata": draw.get("strata")}
    return {"law": "subset", "k": draw["k"]} if draw.get("law") == "subset" else {"law": "bernoulli", "p": draw.get("p")}


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _nat(x: Any) -> bool:
    return _int(x) and x >= 0


def same_law(a: Any, b: Any) -> bool:
    """Two law objects are the same canonical bytes (so `true` is not `1`, as the Lean verifier reads them)."""
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):
        return False


def strata_problem(law: Mapping[str, Any], population: int) -> str | None:
    """Why a stratified or work law object isn't well formed over `population` units, or None: each stratum is exactly
    {template, units, k} (a work stratum {template, units, work, floor, k}) with a distinct template; the ranges are
    ascending, disjoint and non-empty and tile [0, population); each stratum's k is the rule's (a work law's: a floor of
    at least one, a population that does some work, and each k `work_k` above its floor; Lean `workProblem`)."""
    strata, k, work = law.get("strata"), law.get("k"), law.get("law") == "work"
    if not isinstance(strata, list) or not strata or not _int(k) or not 1 <= k <= population:
        return "strata and 1 <= k <= population"
    keys = _WORK_STRATUM_KEYS if work else _STRATUM_KEYS
    total = 0
    if work:
        if not all(isinstance(s, Mapping) and set(s) == keys and _nat(s["work"]) and _nat(s["floor"]) and _ranges_ok(s["units"])
                   for s in strata):
            return "a work stratum is exactly {template, units, work, floor, k}, its work and floor natural numbers"
        total = total_work(strata)
        if total == 0:
            return "a work law's population does no work"
    at, seen = 0, set()
    ranges = []
    for s in strata:
        if not isinstance(s, Mapping) or set(s) != keys or not isinstance(s["template"], str):
            return "a stratum is exactly {template, units, work, floor, k}" if work else "a stratum is exactly {template, units, k}"
        if s["template"] in seen:
            return f"template {s['template']!r} twice"
        seen.add(s["template"])
        rs = s["units"]
        if not _ranges_ok(rs):
            return f"stratum {s['template']!r}: units are non-empty [lo, hi) ranges"
        if any(b[0] <= a[1] for a, b in zip(rs, rs[1:])):
            return f"stratum {s['template']!r}: ranges ascending, disjoint and not adjacent"
        if work and s["floor"] < 1:
            return f"stratum {s['template']!r}: a floor is at least one"
        rule = work_k(k, s["work"], _size(rs), total, s["floor"]) if work else stratum_k(k, _size(rs), population)
        if not _int(s["k"]) or s["k"] != rule:
            return f"stratum {s['template']!r}: k is not the {'work ' if work else ''}rule's"
        ranges += rs
    for lo, hi in sorted(ranges):
        if lo != at:
            return "the strata don't tile the population"
        at = hi
    return None if at == population else "the strata don't tile the population"


def _ranges_ok(rs: Any) -> bool:
    return isinstance(rs, list) and bool(rs) and all(isinstance(r, list) and len(r) == 2 and all(map(_int, r)) and r[0] < r[1]
                                                    for r in rs)


def proved(draw: Mapping[str, Any]) -> list[int]:
    """The units a draw proves: the drawn units and a work draw's closure, ascending (Lean `UnitDraw.proved`)."""
    return sorted([*draw["units"], *draw.get("closure", ())])


def check_draw(draw: Any, registration: Mapping[str, Any], law: Mapping[str, Any]) -> Verdict:
    """The draw is well formed, over the registered population, under the verifier's law (for `stratified` and `work`,
    the expanded law of `derive`)."""
    c = _Collect()
    if not isinstance(draw, Mapping) or draw.get("law") not in ("subset", "bernoulli", "stratified", "work"):
        c.fail("not-a-draw", draw)
        return c.verdict()
    keys = {"law", "population", "units", *{"subset": ["k"], "bernoulli": ["p"], "stratified": ["k", "strata"],
                                            "work": ["k", "strata"]}[draw["law"]]}
    if draw["law"] == "work" and "closure" in draw:
        keys.add("closure")
    if set(draw) != keys:
        c.fail("keys", sorted(set(draw) ^ keys))
        return c.verdict()
    n, units = draw["population"], draw["units"]
    if not same_law(law_of(draw), dict(law)):
        c.fail("law", {"drawn": law_of(draw), "verifier": dict(law)})
    if n != registration["population"]:
        c.fail("population", {"drawn": n, "registered": registration["population"]})
    if not isinstance(units, list) or not all(_int(u) for u in units):
        c.fail("units", units)
        return c.verdict()
    if any(b <= a for a, b in zip(units, units[1:])):
        c.fail("units-order", units[:16])
    if units and not (0 <= units[0] and units[-1] < n):
        c.fail("units-range", [u for u in units if not 0 <= u < n][:16])
    if draw["law"] == "subset" and len(units) != draw["k"]:
        c.fail("units-count", {"units": len(units), "k": draw["k"]})
    if draw["law"] == "bernoulli":
        num, _, den = str(draw["p"]).partition("/")
        if not (num.isdigit() and den.isdigit() and 0 < int(den) and Fraction(int(num), int(den)) <= 1):
            c.fail("p", draw["p"])
    if draw["law"] in ("stratified", "work"):
        problem = strata_problem(draw, n) if _int(n) else "population"
        if problem:
            c.fail("strata", problem)
        else:
            drawn = {s["template"]: sum(1 for u in units if any(lo <= u < hi for lo, hi in s["units"])) for s in draw["strata"]}
            wrong = {s["template"]: {"drawn": drawn[s["template"]], "k": s["k"]} for s in draw["strata"]
                     if drawn[s["template"]] != s["k"]}
            if wrong:
                c.fail("strata-count", wrong)
    if "closure" in draw:
        cs = draw["closure"]
        if (not isinstance(cs, list) or not cs or not all(_int(x) for x in cs) or any(b <= a for a, b in zip(cs, cs[1:]))
                or not (0 <= cs[0] and _int(n) and cs[-1] < n)):
            c.fail("closure", cs if isinstance(cs, list) else None)
        elif set(cs) & set(units):
            c.fail("closure-drawn", sorted(set(cs) & set(units))[:16])
    return c.verdict()


def check_proved(draw: Mapping[str, Any], proved_units: Sequence[int], instances: int) -> Verdict:
    """The proved statement's instances are exactly the units the draw proves (`proved`: the drawn units, and a work
    draw's closure), in ascending order."""
    c = _Collect()
    want = proved(draw)
    if instances != len(want):
        c.fail("instances", {"statement": instances, "drawn": len(want)})
    if list(proved_units) != want:
        c.fail("units", {"statement": list(proved_units)[:16], "drawn": want[:16]})
    return c.verdict()


__all__ = ["check_draw", "check_proved", "derive", "law_of", "parse_law", "proved", "same_law", "strata_problem",
           "stratum_k", "total_work", "work_k", "work_table"]
