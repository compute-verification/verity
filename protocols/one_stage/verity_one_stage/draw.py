"""The draw: the verifier's own coins pick the units to prove, sent in the clear after registration and before the session.

    {"law": "subset", "population": N, "k": K, "units": [<ascending unit indices>]}
    {"law": "bernoulli", "population": N, "p": "NUM/DEN", "units": [...]}
    {"law": "stratified", "population": N, "k": K, "strata": [<stratum>, ...], "units": [...]}

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
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

from verity.ir.program import Program

from . import partition as P
from .verdict import Verdict, _Collect

_STRATUM_KEYS = {"template", "units", "k"}


def parse_law(text: str) -> dict[str, Any]:
    """`subset:K` or `bernoulli:NUM/DEN` (M0's `serve --draw` syntax), or `stratified:K` (the rule, which `derive` expands
    over a program and query), as a law object."""
    kind, _, arg = text.partition(":")
    if kind == "subset" and arg.isdigit():
        return {"law": "subset", "k": int(arg)}
    if kind == "stratified" and arg.isdigit() and int(arg) >= 1:
        return {"law": "stratified", "k": int(arg)}
    if kind == "bernoulli":
        num, _, den = arg.partition("/")
        if num.isdigit() and den.isdigit() and 0 < int(den) and int(num) <= int(den):
            return {"law": "bernoulli", "p": f"{int(num)}/{int(den)}"}
    raise ValueError(f"a law is subset:K, stratified:K or bernoulli:NUM/DEN, not {text!r}")


def stratum_k(k: int, size: int, population: int) -> int:
    """A stratum's draw under the stratified rule: max(1, round(k * size / population)), rounding half up."""
    return max(1, (2 * k * size + population) // (2 * population))


def derive(program: Program, query: Mapping[str, Any], law: Mapping[str, Any]) -> dict[str, Any]:
    """The verifier's law for its program and query: `stratified` (the rule, or an expanded law, whose strata are ignored)
    expands to the strata `partition.strata` derives, each with `stratum_k`; `subset` and `bernoulli` are unchanged.
    Raises `partition.QueryRefused` when the query doesn't apply, and `ValueError` when k isn't in [1, population]."""
    if law.get("law") != "stratified":
        return dict(law)
    strata = P.strata(program, query)
    n = sum(hi - lo for _, rs in strata for lo, hi in rs)
    k = law.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
        raise ValueError(f"a stratified law needs 1 <= k <= {n}, not {k!r}")
    return {"law": "stratified", "k": k,
            "strata": [{"template": t, "units": [list(r) for r in rs], "k": stratum_k(k, sum(hi - lo for lo, hi in rs), n)}
                       for t, rs in strata]}


def law_of(draw: Mapping[str, Any]) -> dict[str, Any]:
    if draw.get("law") == "stratified":
        return {"law": "stratified", "k": draw.get("k"), "strata": draw.get("strata")}
    return {"law": "subset", "k": draw["k"]} if draw.get("law") == "subset" else {"law": "bernoulli", "p": draw.get("p")}


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def strata_problem(law: Mapping[str, Any], population: int) -> str | None:
    """Why a stratified law object isn't well formed over `population` units, or None: each stratum is exactly
    {template, units, k} with a distinct template; the ranges are ascending, disjoint and non-empty and tile
    [0, population); each stratum's k is the rule's."""
    strata, k = law.get("strata"), law.get("k")
    if not isinstance(strata, list) or not strata or not _int(k) or not 1 <= k <= population:
        return "strata and 1 <= k <= population"
    at, seen = 0, set()
    ranges = []
    for s in strata:
        if not isinstance(s, Mapping) or set(s) != _STRATUM_KEYS or not isinstance(s["template"], str):
            return "a stratum is exactly {template, units, k}"
        if s["template"] in seen:
            return f"template {s['template']!r} twice"
        seen.add(s["template"])
        rs = s["units"]
        if not isinstance(rs, list) or not rs or not all(isinstance(r, list) and len(r) == 2 and all(map(_int, r)) and r[0] < r[1]
                                                        for r in rs):
            return f"stratum {s['template']!r}: units are non-empty [lo, hi) ranges"
        if any(b[0] <= a[1] for a, b in zip(rs, rs[1:])):
            return f"stratum {s['template']!r}: ranges ascending, disjoint and not adjacent"
        if s["k"] != stratum_k(k, sum(hi - lo for lo, hi in rs), population):
            return f"stratum {s['template']!r}: k is not the rule's"
        ranges += rs
    for lo, hi in sorted(ranges):
        if lo != at:
            return "the strata don't tile the population"
        at = hi
    return None if at == population else "the strata don't tile the population"


def check_draw(draw: Any, registration: Mapping[str, Any], law: Mapping[str, Any]) -> Verdict:
    """The draw is well formed, over the registered population, under the verifier's law (for `stratified`, the expanded
    law of `derive`)."""
    c = _Collect()
    if not isinstance(draw, Mapping) or draw.get("law") not in ("subset", "bernoulli", "stratified"):
        c.fail("not-a-draw", draw)
        return c.verdict()
    keys = {"law", "population", "units", *{"subset": ["k"], "bernoulli": ["p"], "stratified": ["k", "strata"]}[draw["law"]]}
    if set(draw) != keys:
        c.fail("keys", sorted(set(draw) ^ keys))
        return c.verdict()
    n, units = draw["population"], draw["units"]
    if law_of(draw) != dict(law):
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
    if draw["law"] == "stratified":
        problem = strata_problem(draw, n) if _int(n) else "population"
        if problem:
            c.fail("strata", problem)
        else:
            drawn = {s["template"]: sum(1 for u in units if any(lo <= u < hi for lo, hi in s["units"])) for s in draw["strata"]}
            wrong = {s["template"]: {"drawn": drawn[s["template"]], "k": s["k"]} for s in draw["strata"]
                     if drawn[s["template"]] != s["k"]}
            if wrong:
                c.fail("strata-count", wrong)
    return c.verdict()


def check_proved(draw: Mapping[str, Any], proved_units: Sequence[int], instances: int) -> Verdict:
    """The proved statement's instances are exactly the drawn units, in draw order."""
    c = _Collect()
    if instances != len(draw["units"]):
        c.fail("instances", {"statement": instances, "drawn": len(draw["units"])})
    if list(proved_units) != list(draw["units"]):
        c.fail("units", {"statement": list(proved_units)[:16], "drawn": list(draw["units"])[:16]})
    return c.verdict()


__all__ = ["check_draw", "check_proved", "derive", "law_of", "parse_law", "strata_problem", "stratum_k"]
