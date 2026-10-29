"""The draw: the verifier's own coins pick the proof blocks to prove, sent in the clear after registration and before the session.

    {"law": "subset", "population": N, "k": K, "units": [<ascending proof block indices>]}
    {"law": "bernoulli", "population": N, "p": "NUM/DEN", "units": [...]}
    {"law": "stratified", "population": N, "k": K, "strata": [<stratum>, ...], "units": [...]}
    {"law": "work", "population": N, "k": K, "strata": [<work stratum>, ...], "units": [...], ["closure": [...]]}

This is the Lean verifier's `Flock.Draw` object and `flock-circuit`'s `unit_draw`. The verifier samples it from the operating system's
randomness (Lean `flock-verify draw`, or `flock-circuit serve --draw`, live); nothing here samples, and no draw is derived from a
commitment. `check_draw` holds a draw to the registration and to the verifier's own sampling design, and `check_proved` holds the
proved statement's proof blocks to the draw.

**The stratified sampling design**: one stratum per template, a uniform k_s-subset drawn
without replacement inside each, independently.
* **Strata** are derived by each verifier from its own program and query (`partition.strata`): one per template descriptor
  id, its proof blocks as ascending, disjoint half-open ranges `[lo, hi)`, the strata in the order of their first proof block. Defined for
  `Q_template_instance` and `Q_template_instances`; another query refuses it.
* **Sizes**: k_s = max(1, round(k n_s / n)), rounding half up in integers (`stratum_k`), so every template has a draw.
* **The sampling design object** is `{"law": "stratified", "k": K, "strata": [{"template": <descriptor id>, "units": [[lo, hi], ...],
  "k": k_s}, ...]}`: the agreement's rule `stratified:K` expanded by `derive`. R1 holds the registered sampling design to the verifier's
  own expansion, and the draw carries the sampling design's keys plus `population` and `units`.

**The work sampling design** (the Lean verifier's `Flock.Draw.Law.work`, its bound
`FlockSoundness.Audit.Work.work_escape_le`): the stratified sampling design's strata and sampler, with each stratum's draw sized by the
PoUW work its proof blocks do. The verifier's work table gives each template's work per proof block, a natural number in any fixed unit,
and its floor f_s >= 1 (`work_table`). With W = sum_s w_s n_s >= 1, stratum s draws
k_s = min(n_s, max(f_s, ceil(K w_s n_s / W))) (`work_k`), and a work stratum is `{"template", "units", "work", "floor",
"k"}`. A work draw may carry a `closure`: proof blocks proved with the drawn ones, none of them drawn.
* **What the table credits.** A proof block's work is what it is credited for, counted on the rows the Program declares, never on
  rows a kernel pads it with (the Pearl-C statement review, item 5). Filler is credited nothing: a proof block of filler only, or
  one the prover declares uncredited before the draw, does work 0 (its floor still draws it for integrity, and missing it
  costs no work), and a proof block mixing real and filler rows is credited its real cells only. The sampling design weights by template, so
  every proof block of a template must be credited the same work: a template whose proof blocks are padded differently is split before
  the sampling design is derived, or credited 0. The acceptance-probability bound needs each proof block's inclusion probability to be at least K w_u / W
  over the credited W, which a per-template work that over-credits any unit breaks.

**The rate sampling design** (`parse_rate`, the Lean's `Select.parseRate`): a selection rule's `rate:NUM/DEN` or `rate:NUM/DEN:T` has no
sampling design object of its own. Over a registration's N proof blocks it is `subset:K` at K = `rate_count`, ceil(q N) for q = NUM/DEN
rounded up to a multiple of the T tables, so its draw object is a subset's. Under a fixed `subset:K` a proof block's inclusion
K/N falls as the prover pads N; under a rate it stays at least q.

**Several tables** (`--session-tables J`) prove one draw split into J equal consecutive parts, so the draw's count must be
fixed and a multiple of J (`tables_problem`, Lean `Draw.tablesWhy`): a subset draw's k, or a stratified draw's sum of
k_s. A stratified draw is held to the verifier's sampling design once, for the whole session, before the split.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

from verity.core.protocols.profile import work_k
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program

from . import partition as P
from .verdict import Verdict, _Collect

_STRATUM_KEYS = {"template", "units", "k"}
_WORK_STRATUM_KEYS = {"template", "units", "work", "floor", "k"}


def digits(s: str) -> bool:
    """One or more ASCII digits, as the Lean verifier reads a number (`str.isdigit` alone admits other scripts' digits)."""
    return s.isascii() and s.isdigit()


def parse_law(text: str) -> dict[str, Any]:
    """`subset:K` or `bernoulli:NUM/DEN` (`flock-circuit serve --draw`'s syntax), or `stratified:K` or `work:K` (rules, which `derive`
    expands over a program and query, and for `work` a work table), as a sampling design object; numbers in ASCII digits, `K >= 1`.
    A design outside the sampling claims' scope (`Claims.Sampling.OneStage.WellFormed`) is refused: `subset:0` draws
    nothing, so its `P_a` is no miss probability."""
    kind, _, arg = text.partition(":")
    if kind == "subset" and digits(arg) and int(arg) >= 1:
        return {"law": "subset", "k": int(arg)}
    if kind in ("stratified", "work") and digits(arg) and int(arg) >= 1:
        return {"law": kind, "k": int(arg)}
    if kind == "bernoulli":
        num, _, den = arg.partition("/")
        if digits(num) and digits(den) and 0 < int(den) and int(num) <= int(den):
            return {"law": "bernoulli", "p": f"{int(num)}/{int(den)}"}
    raise ValueError(f"a law is subset:K, stratified:K, work:K or bernoulli:NUM/DEN, not {text!r}")


def parse_rate(text: Any) -> tuple[int, int, int] | None:
    """The rate sampling design `rate:NUM/DEN` or `rate:NUM/DEN:T` as `(NUM, DEN, T)`: a public rate NUM/DEN with 0 < NUM <= DEN over
    T >= 1 tables (one when it names none), numbers in ASCII digits; None for any other text. A selection rule states it
    (`service.Law`), and over a registration's N proof blocks it is `subset:K` at K = `rate_count` (`service.selection_law`)."""
    if not isinstance(text, str):
        return None
    kind, _, arg = text.partition(":")
    q, sep, t = arg.partition(":")
    num, _, den = q.partition("/")
    if not sep:
        tables = 1
    elif digits(t):
        tables = int(t)
    else:
        return None
    if kind == "rate" and digits(num) and digits(den) and 0 < int(num) <= int(den) and tables >= 1:
        return int(num), int(den), tables
    return None


def rate_count(num: int, den: int, tables: int, n: int) -> int:
    """The rate sampling design's draw over n proof blocks: ceil(NUM n / DEN) rounded up to a multiple of `tables`, so each of the session's
    tables proves the same count and every proof block is drawn with probability K/n >= NUM/DEN, whatever n the prover registers.
    The Lean's `SampledProofs.Rate.count` is this."""
    return tables * -(-(num * n) // (den * tables))


def stratum_k(k: int, size: int, population: int) -> int:
    """A stratum's draw under the stratified rule: max(1, round(k * size / population)), rounding half up."""
    return max(1, (2 * k * size + population) // (2 * population))


def work_table(table: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """The verifier's work table (Lean `workTableOfJson`): template descriptor id -> the credited work of one of its proof blocks
    (a natural number, floor one) or `{"work": w, "floor": f}` with f >= 1; as `{template: (work, floor)}`. Raises
    `ValueError` naming the first bad entry. See the module docstring for what a proof block's work may count."""
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
    """A work sampling design's population work, W = sum_s w_s n_s."""
    return sum(s["work"] * _size(s["units"]) for s in strata)


def derive(program: Program, query: Mapping[str, Any], law: Mapping[str, Any],
           table: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The verifier's sampling design for its program and query: `stratified` (the rule, or an expanded sampling design, whose strata are ignored)
    expands to the strata `partition.strata` derives, each with `stratum_k`; `work` expands to the same strata with the
    work table's works and floors (`work_table`), each with `work_k` (Lean `Flock.Draw.deriveWork`); `subset` and
    `bernoulli` are unchanged. Raises `partition.QueryRefused` when the query doesn't apply, and `ValueError` when k isn't
    in [1, population], a work sampling design has no table, the table names no work for a template, or the population does none."""
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
    """Two sampling design objects are the same canonical bytes (so `true` is not `1`, as the Lean verifier reads them)."""
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):
        return False


def strata_problem(law: Mapping[str, Any], population: int) -> str | None:
    """Why a stratified or work sampling design object isn't well formed over `population` proof blocks, or None: each stratum is exactly
    {template, proof blocks, k} (a work stratum {template, proof blocks, work, floor, k}) with a distinct template; the ranges are
    ascending, disjoint and non-empty and tile [0, population); each stratum's k is the rule's (a work sampling design's: a floor of
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


def fixed_count(law: Mapping[str, Any]) -> int | None:
    """The proof blocks every draw under `law` has, when that count is fixed: `k` for `subset` and the strata's sum of k_s for
    `stratified`; None for `bernoulli` (its count is random) and `work` (its closure varies). Lean `Draw.Law.fixedCount`."""
    if law.get("law") == "subset":
        return law.get("k")
    if law.get("law") == "stratified" and isinstance(law.get("strata"), list):
        return sum(s.get("k", 0) for s in law["strata"] if isinstance(s, Mapping))
    return None


#: Lean `Draw.TABLES_LAW`, the refusal of a sampling design several tables cannot split
TABLES_LAW = ("tables: several tables need a subset or stratified draw, whose unit count is fixed (a Bernoulli draw's "
              "count is random, so the tables could not split it evenly, and a work draw's closure is not split)")


def tables_problem(law: Mapping[str, Any], tables: int) -> str | None:
    """Why one draw under `law` can't be split into `tables` equal tables, as the Lean verifier's `Stmt.setupTables`
    splits a session's draw, or None: one table takes any sampling design; several take a sampling design of fixed count (`fixed_count`) that is
    positive and a multiple of `tables`. Lean `Draw.tablesWhy`, which `flock-verify draw --tables J` refuses by before
    any coin."""
    if tables <= 1:
        return None
    n = fixed_count(law)
    if not _int(n):
        return TABLES_LAW
    if n == 0 or n % tables:
        return f"tables: {n} units do not split into {tables} equal tables"
    return None


def proved(draw: Mapping[str, Any]) -> list[int]:
    """The proof blocks a draw proves: the drawn proof blocks and a work draw's closure, ascending (Lean `UnitDraw.proved`)."""
    return sorted([*draw["units"], *draw.get("closure", ())])


def check_draw(draw: Any, registration: Mapping[str, Any], law: Mapping[str, Any], tables: int = 1) -> Verdict:
    """The draw is well formed, over the registered population, under the verifier's sampling design (for `stratified` and `work`,
    the expanded sampling design of `derive`), and a session of `tables` tables can split it (`tables_problem`)."""
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
    if (why := tables_problem(law_of(draw), tables)) is not None:
        c.fail("tables", why)
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
        if not (digits(num) and digits(den) and 0 < int(den) and Fraction(int(num), int(den)) <= 1):
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
    """The proved statement's instances are exactly the proof blocks the draw proves (`proved`: the drawn proof blocks, and a work
    draw's closure), in ascending order."""
    c = _Collect()
    want = proved(draw)
    if instances != len(want):
        c.fail("instances", {"statement": instances, "drawn": len(want)})
    if list(proved_units) != want:
        c.fail("units", {"statement": list(proved_units)[:16], "drawn": want[:16]})
    return c.verdict()


__all__ = ["TABLES_LAW", "check_draw", "check_proved", "derive", "digits", "fixed_count", "law_of", "parse_law", "parse_rate",
           "proved", "rate_count", "same_law", "strata_problem", "stratum_k", "tables_problem", "total_work", "work_k",
           "work_table"]
