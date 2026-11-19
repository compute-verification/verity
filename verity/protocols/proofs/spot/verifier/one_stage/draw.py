"""The sample: the challenger's coins pick the proof blocks to prove, sent in the clear after docket and before the session.

    {"law": "subset", "population": N, "k": K, "units": [<ascending proof block indices>]}
    {"law": "bernoulli", "population": N, "p": "NUM/DEN", "units": [...]}
    {"law": "stratified", "population": N, "k": K, "strata": [<stratum>, ...], "units": [...]}
    {"law": "work", "population": N, "k": K, "strata": [<work stratum>, ...], "units": [...], ["closure": [...]]}

This is the Lean verifier's `Flock.Draw` object and `flock-circuit`'s `unit_draw`. A stream samples it from the
auditor's challenger's coins (`service.Stream.select`, Lean `Select.select`); `verity-verify-proof draw` reads them too
(`--stream`, `--stream-file`), and without them refuses any sample that reads coins: the verifier has no randomness of
its own. Nothing here samples, and no sample is derived from a commitment. `check_draw` holds a sample to the
docket and to the verifier's own sampling design, and `check_proved` holds the proved statement's proof blocks to
the sample.

**The stratified sampling design**: one stratum per template, a uniform k_s-subset sampled
without replacement inside each, independently.
* **Strata** are derived by each verifier from its own program and query (`partition.strata`): one per template descriptor
  id, its proof blocks as ascending, disjoint half-open ranges `[lo, hi)`, the strata in the order of their first proof block. Defined for
  `Q_template_instance` and `Q_template_instances`; another query refuses it.
* **Sizes**: k_s = max(1, round(k n_s / n)), rounding half up in integers (`stratum_k`), so every template has a sample.
* **The sampling design object** is `{"law": "stratified", "k": K, "strata": [{"template": <descriptor id>, "units": [[lo, hi], ...],
  "k": k_s}, ...]}`: the sampling design `{"law": "stratified", "k": K}` expanded by `derive`. R1 holds the registered sampling design to the verifier's
  own expansion, and the sample carries the sampling design's keys plus `population` and `units`.

**The work sampling design** (the Lean verifier's `Flock.Draw.SamplingDesign.work`, its bound
`Spot.Sampling.Work.work_escape_le`): the stratified sampling design's strata and sampler, with each stratum's sample sized by the
PoUW work its proof blocks do. The verifier's work table gives each template's work per proof block, a natural number in any fixed unit,
and its floor f_s >= 1 (`work_table`). With W = sum_s w_s n_s >= 1, stratum s samples
k_s = min(n_s, max(f_s, ceil(K w_s n_s / W))) (`work_k`), and a work stratum is `{"template", "units", "work", "floor",
"k"}`. A work sample may carry a `closure`: proof blocks proved with the sampled ones, none of them sampled.
* **What the table credits.** A proof block's work is what it is credited for, counted on the rows the Program declares, never on
  rows a kernel pads it with (the Pearl-C statement review, item 5). Filler is credited nothing: a proof block of filler only, or
  one the prover declares uncredited before the sample, does work 0 (its floor still samples it for integrity, and missing it
  costs no work), and a proof block mixing real and filler rows is credited its real cells only. The sampling design weights by template, so
  every proof block of a template must be credited the same work: a template whose proof blocks are padded differently is split before
  the sampling design is derived, or credited 0. The acceptance-probability bound needs each proof block's inclusion probability to be at least K w_u / W
  over the credited W, which a per-template work that over-credits any unit breaks.

**The rate sampling designs** (`randomness.parse_rate` and `randomness.parse_strata_rate`, the Lean's `Select.parseRate` and `Select.parseStrataRate`): a
selection rule states its sampling design as a rate q = NUM/DEN, which has no sampling design object of its own. Over a docket's
N proof blocks `rate:NUM/DEN` or `rate:NUM/DEN:T` is `subset:K` at K = `rate_count`, ceil(q N) rounded up to a multiple of the T
tables. `stratified-rate:NUM/DEN` is the stratified sampling design at the K its strata need (`strat_k`: the most any stratum of
n_s proof blocks needs for `stratum_k` to reach ceil(q n_s), at most N), so every stratum samples at least q of its proof blocks
however many the prover registers in it or elsewhere (Lean `Rate.stratK_ge`). `work-rate:NUM/DEN` is the work sampling design at K =
ceil(q N) over a table the verifier derives (`verity.commitments.rate`'s `rate_table`), the verifier's work table's work with each stratum's floor raised to
ceil(q n_s) (`rate_floor`), and since k_s >= its floor, every stratum samples at least q of its proof blocks too (Lean
`Rate.workK_ge`). A fixed count, `subset:K`, `stratified:K` or `work:K`, is refused as a rule: a proof block's inclusion K/N falls
as the prover pads the N it registers.

**Several tables** (`--session-tables J`) prove one sample split into J equal consecutive parts, so the sample's count must be
fixed and a multiple of J (`tables_problem`, Lean `Draw.tablesWhy`): a subset sample's k, or a stratified sample's sum of
k_s. A stratified sample is held to the verifier's sampling design once, for the whole session, before the split.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

from verity.circuits.codec import canonical_json
from verity.circuits.program import Program
from verity.commitments.randomness import digits, parse_rate, parse_strata_rate
from verity.commitments.rate import _int, _nat, _size, rate_count, work_table

from . import partition as P
from .verdict import Verdict, _Collect

_STRATUM_KEYS = {"template", "units", "k"}
_WORK_STRATUM_KEYS = {"template", "units", "work", "floor", "k"}


def rate_need(num: int, den: int, size: int, n: int) -> int:
    """The least K >= 1 at which the stratified sampling design samples ceil(NUM size / DEN) of a stratum of `size` of `n` proof
    blocks (`stratum_k`): ceil((2 c n - n) / (2 size)) at c = that count, and 1 where c <= 1. The Lean's `Spot.Rate.need` is
    this."""
    c = rate_count(num, den, 1, size)
    return 1 if c <= 1 else (2 * (c * n) - n + 2 * size - 1) // (2 * size)


def strat_k(num: int, den: int, sizes: Sequence[int], n: int) -> int:
    """The stratified rate's K over strata of `sizes` of `n` proof blocks: the most any stratum needs (`rate_need`), at most `n`,
    at which each stratum samples all of its proof blocks. The Lean's `Spot.Rate.stratK` is this, and `Rate.stratK_ge` the bound."""
    return min(n, max([1, *(rate_need(num, den, s, n) for s in sizes)]))


def strata_sizes(program: Program, query: Mapping[str, Any]) -> list[int]:
    """The proof blocks of each stratum of the program under the query (`partition.strata`), which the stratified rate's K reads
    (`strat_k`). Raises `partition.QueryRefused` when the query doesn't apply, where the Lean's `Select.strataSizes`, this
    elsewhere, is empty and `Rule.over` refuses a stratified rate (`service.selection_sizes`)."""
    return [_size(rs) for _, rs in P.strata(program, query)]


def stratum_k(k: int, size: int, population: int) -> int:
    """A stratum's sample under the stratified rule: max(1, round(k * size / population)), rounding half up."""
    return max(1, (2 * k * size + population) // (2 * population))


def total_work(strata: Sequence[Mapping[str, Any]]) -> int:
    """A work sampling design's population work, W = sum_s w_s n_s."""
    return sum(s["work"] * _size(s["units"]) for s in strata)


def work_k(k: int, work: int, size: int, total: int, floor: int) -> int:
    """A stratum's sample size under the work rule (Lean `Flock.Draw.workK`): ceil(k * work * size / total), at least
    `floor` and at most `size`."""
    return min(size, max(floor, (k * work * size + total - 1) // total))


def derive(program: Program, query: Mapping[str, Any], sampling_design: Mapping[str, Any],
           table: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The verifier's sampling design for its program and query: `stratified` (the rule, or an expanded sampling design, whose strata are ignored)
    expands to the strata `partition.strata` derives, each with `stratum_k`; `work` expands to the same strata with the
    work table's works and floors (`work_table`), each with `work_k` (Lean `Flock.Draw.deriveWork`); `subset` and
    `bernoulli` are unchanged. Raises `partition.QueryRefused` when the query doesn't apply, and `ValueError` when k isn't
    in [1, population], a work sampling design has no table, the table names no work for a template, or the population does none."""
    kind = sampling_design.get("law")
    if kind not in ("stratified", "work"):
        return dict(sampling_design)
    strata = P.strata(program, query)
    n = sum(hi - lo for _, rs in strata for lo, hi in rs)
    k = sampling_design.get("k")
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


def amount_of(table: Mapping[str, tuple[int, int]], template: str) -> int:
    """The amount of each proof block of `template` in an amount table (`work_table`'s, its work; the floor is not read).
    Raises `ValueError` for a template the table does not name. Lean `Flock.Draw.amountOf`."""
    if template not in table:
        raise ValueError(f"the amount table names no amount for template '{template}'")
    return table[template][0]


def meets_at(num: int, den: int, n: int, c: int, k: int) -> bool:
    """Whether a uniform k-subset of n proof blocks, each of amount at most c, meets the rate NUM/DEN: k = n or
    k DEN >= n c NUM. Lean `Flock.Draw.meetsAt`."""
    return k == n or n * c * num <= k * den


def _rate_miss(num: int, den: int, what: str, n: int, c: int, k: int) -> None:
    if not meets_at(num, den, n, c, k):
        raise ValueError(f"{what} misses the rate {num}/{den}: k * DEN = {k * den} < n * c * NUM = {n * c * num} "
                         f"(n = {n}, c = {c}, k = {k})")


def meets_rate(num: int, den: int, table: Mapping[str, tuple[int, int]], strata: Sequence[tuple[str, Sequence[Sequence[int]]]],
               sampling_design: Mapping[str, Any]) -> None:
    """Whether the expanded sampling design over the population's strata (`partition.strata`) meets the rate NUM/DEN per
    unit of the amount table's amount (`work_table`'s output); raises `ValueError` saying why not. Lean
    `Flock.Draw.meetsRate`, whose docstring has the rule and what it gives; the messages are its."""
    if den == 0:
        raise ValueError(f"a rate needs DEN > 0, not {num}/{den}")
    pop = [(t, [list(r) for r in rs], amount_of(table, t)) for t, rs in strata]
    n = sum(_size(rs) for _, rs, _ in pop)
    c = max((a for _, _, a in pop), default=0)
    law = sampling_design.get("law")
    if law == "subset":
        k = sampling_design["k"]
        if k > n:
            raise ValueError(f"a subset design of {k} exceeds the population of {n}")
        _rate_miss(num, den, "the pooled subset", n, c, k)
    elif law == "bernoulli":
        a_, _, b_ = str(sampling_design.get("p")).partition("/")
        if not (digits(a_) and digits(b_)):
            raise ValueError(f"a bernoulli design's p is NUM/DEN, not {sampling_design.get('p')!r}")
        a, b = int(a_), int(b_)
        if b == 0 or a > b:
            raise ValueError(f"a bernoulli design's p = {a}/{b} is not a probability")
        if not (a == b or b * c * num <= a * den):
            raise ValueError(f"the bernoulli design misses the rate {num}/{den}: a * DEN = {a * den} < b * c * NUM = "
                             f"{b * c * num} (p = {a}/{b}, c = {c})")
    elif law in ("stratified", "work"):
        ss = sampling_design["strata"]
        if len(ss) != len(pop):
            raise ValueError(f"the design has {len(ss)} strata, not the population's {len(pop)} templates")
        seen: set[str] = set()
        for s in ss:
            t = s["template"]
            if t in seen:
                raise ValueError(f"the design names template '{t}' twice")
            seen.add(t)
            mine = next((p for p in pop if p[0] == t), None)
            if mine is None:
                raise ValueError(f"the design's stratum '{t}' is no template of the population")
            units = [list(r) for r in s["units"]]
            if units != mine[1]:
                raise ValueError(f"stratum '{t}': its units are not the population's units of its template")
            size = _size(units)
            if s["k"] > size:
                raise ValueError(f"stratum '{t}': k = {s['k']} exceeds its {size} proof blocks")
            _rate_miss(num, den, f"stratum '{t}'", size, mine[2], s["k"])
    else:
        raise ValueError(f"a {law!r} design is not subset, bernoulli, stratified or work")


def cheapest_for(num: int, den: int, table: Mapping[str, tuple[int, int]],
                 strata: Sequence[tuple[str, Sequence[Sequence[int]]]]) -> list[dict[str, Any]]:
    """The prover's cheapest strata at the rate NUM/DEN: each of the population's strata with the least k_s that meets it,
    min(n_s, ceil(n_s c_s NUM / DEN)), as `{template, units, k}`. Refused as `meets_rate` refuses. Lean
    `Flock.Draw.cheapestFor`."""
    if den == 0:
        raise ValueError(f"a rate needs DEN > 0, not {num}/{den}")
    out = []
    for t, rs in strata:
        c, n = amount_of(table, t), _size(rs)
        out.append({"template": t, "units": [list(r) for r in rs], "k": min(n, (n * c * num + den - 1) // den)})
    return out


def rule_rate(text: Any) -> tuple[int, int] | None:
    """The rate NUM/DEN under which R1 also agrees to a registered law that meets it (`agreed`): `rate:NUM/DEN` over one
    table, or `stratified-rate:NUM/DEN`; None for any other law. Lean `OneStage.rateOf`."""
    if not isinstance(text, str):
        return None
    pooled = parse_rate(text)
    if pooled is not None:
        return (pooled[0], pooled[1]) if pooled[2] == 1 else None
    strata = parse_strata_rate(text)
    return (strata[1], strata[2]) if strata is not None and not strata[0] else None


def rate_agreed(num: int, den: int, strata: Sequence[tuple[str, Sequence[Sequence[int]]]], mine: Mapping[str, Any],
                law: Any) -> dict[str, Any]:
    """Under a rate rule NUM/DEN, the registered sampling design `law` when it is the rule's expansion `mine`'s law (subset or
    stratified) and meets the rate over the population's `strata` at amount 1 a proof block (`meets_rate`), so it covers
    every stratum of the population exactly; raises `ValueError` saying why not. A stratified `law`'s `k` is the sum of its
    strata's `k_s`, so a law other than the rule's expansion has one registered form. Lean `Spot.Draw.rateAgreed`."""
    kind = mine.get("law") if isinstance(mine, Mapping) else None
    if kind not in ("subset", "stratified") or not isinstance(law, Mapping) or law.get("law") != kind:
        raise ValueError("rate: the registered law is not the rule's, subset or stratified")
    n = sum(_size(rs) for _, rs in strata)
    if kind == "subset":
        if sorted(law) != ["k", "law"]:
            raise ValueError("rate: a subset law is exactly {law, k}")
        if not _nat(law["k"]):
            raise ValueError("rate: a subset law's k is a natural number")
    else:
        if sorted(law) != ["k", "law", "strata"]:
            raise ValueError("rate: a stratified law is exactly {law, k, strata}")
        why = strata_problem(law, n)
        if why is not None:
            raise ValueError(f"rate: {why}")
        total = sum(s["k"] for s in law["strata"])
        if law["k"] != total:
            raise ValueError(f"rate: the stratified law's k = {law['k']} is not the sum of its strata's k_s = {total}")
    meets_rate(num, den, {t: (1, 1) for t, _ in strata}, strata, law)
    return dict(law)


def agreed(program: Program, query: Mapping[str, Any], sampling_design: Mapping[str, Any],
           table: Mapping[str, Any] | None, rate: tuple[int, int] | None, law: Any) -> dict[str, Any]:
    """The sampling design the verifier agrees to for the registered `law`: `derive`'s expansion, or under a rate rule
    (`rate` = (NUM, DEN)) a different `law` that meets it (`rate_agreed`). Raises as they do. Lean `Spot.Draw.agreed`."""
    mine = derive(program, query, sampling_design, table)
    if rate is None or same_design(law, mine):
        return mine
    return rate_agreed(rate[0], rate[1], P.strata(program, query), mine, law)


def sampling_design_of(draw: Mapping[str, Any]) -> dict[str, Any]:
    if draw.get("law") in ("stratified", "work"):
        return {"law": draw["law"], "k": draw.get("k"), "strata": draw.get("strata")}
    return {"law": "subset", "k": draw["k"]} if draw.get("law") == "subset" else {"law": "bernoulli", "p": draw.get("p")}


def same_design(a: Any, b: Any) -> bool:
    """Two sampling design objects are the same canonical bytes (so `true` is not `1`, as the Lean verifier reads them)."""
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):
        return False


def strata_problem(sampling_design: Mapping[str, Any], population: int) -> str | None:
    """Why a stratified or work sampling design object isn't well formed over `population` proof blocks, or None: each stratum is exactly
    {template, proof blocks, k} (a work stratum {template, proof blocks, work, floor, k}) with a distinct template; the ranges are
    ascending, disjoint and non-empty and tile [0, population); each stratum's k is in [1, its size] (a work sampling design's: a floor
    of at least one, a population that does some work, and each k `work_k` above its floor; Lean `workProblem`). Which k_s the
    verifier agrees to is `agreed`'s."""
    strata, k, work = sampling_design.get("strata"), sampling_design.get("k"), sampling_design.get("law") == "work"
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
        if work:
            if not _int(s["k"]) or s["k"] != work_k(k, s["work"], _size(rs), total, s["floor"]):
                return f"stratum {s['template']!r}: k is not the work rule's"
        elif not _int(s["k"]) or not 1 <= s["k"] <= _size(rs):
            return f"stratum {s['template']!r}: k is not in [1, {_size(rs)}]"
        ranges += rs
    for lo, hi in sorted(ranges):
        if lo != at:
            return "the strata don't tile the population"
        at = hi
    return None if at == population else "the strata don't tile the population"


def _ranges_ok(rs: Any) -> bool:
    return isinstance(rs, list) and bool(rs) and all(isinstance(r, list) and len(r) == 2 and all(map(_int, r)) and r[0] < r[1]
                                                    for r in rs)


def fixed_count(sampling_design: Mapping[str, Any]) -> int | None:
    """The proof blocks every sample under `sampling_design` has, when that count is fixed: `k` for `subset` and the strata's sum of k_s for
    `stratified`; None for `bernoulli` (its count is random) and `work` (its closure varies). Lean `Draw.SamplingDesign.fixedCount`."""
    if sampling_design.get("law") == "subset":
        return sampling_design.get("k")
    if sampling_design.get("law") == "stratified" and isinstance(sampling_design.get("strata"), list):
        return sum(s.get("k", 0) for s in sampling_design["strata"] if isinstance(s, Mapping))
    return None


#: Lean `Draw.TABLES_SAMPLING_DESIGN`, the refusal of a sampling design several tables cannot split
TABLES_SAMPLING_DESIGN = ("tables: several tables need a subset or stratified draw, whose unit count is fixed (a Bernoulli draw's "
              "count is random, so the tables could not split it evenly, and a work draw's closure is not split)")


def tables_problem(sampling_design: Mapping[str, Any], tables: int) -> str | None:
    """Why one sample under `sampling_design` can't be split into `tables` equal tables, as the Lean verifier's `Stmt.setupTables`
    splits a session's sample, or None: one table takes any sampling design; several take a sampling design of fixed count (`fixed_count`) that is
    positive and a multiple of `tables`. Lean `Draw.tablesWhy`, which `verity-verify-proof draw --tables J` refuses by before
    any coin."""
    if tables <= 1:
        return None
    n = fixed_count(sampling_design)
    if not _int(n):
        return TABLES_SAMPLING_DESIGN
    if n == 0 or n % tables:
        return f"tables: {n} units do not split into {tables} equal tables"
    return None


def proved(draw: Mapping[str, Any]) -> list[int]:
    """The proof blocks a sample proves: the sampled proof blocks and a work sample's closure, ascending (Lean `UnitSample.proved`)."""
    return sorted([*draw["units"], *draw.get("closure", ())])


def check_draw(draw: Any, docket: Mapping[str, Any], sampling_design: Mapping[str, Any], tables: int = 1) -> Verdict:
    """The sample is well formed, over the registered population, under the verifier's sampling design (for `stratified` and `work`,
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
    if (why := tables_problem(sampling_design_of(draw), tables)) is not None:
        c.fail("tables", why)
    if not same_design(sampling_design_of(draw), dict(sampling_design)):
        c.fail("law", {"drawn": sampling_design_of(draw), "verifier": dict(sampling_design)})
    if n != docket["population"]:
        c.fail("population", {"drawn": n, "registered": docket["population"]})
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
    """The proved statement's instances are exactly the proof blocks the sample proves (`proved`: the sampled proof blocks, and a work
    sample's closure), in ascending order."""
    c = _Collect()
    want = proved(draw)
    if instances != len(want):
        c.fail("instances", {"statement": instances, "drawn": len(want)})
    if list(proved_units) != want:
        c.fail("units", {"statement": list(proved_units)[:16], "drawn": want[:16]})
    return c.verdict()


__all__ = ["TABLES_SAMPLING_DESIGN", "amount_of", "cheapest_for", "check_draw", "check_proved", "derive", "fixed_count",
           "meets_at", "meets_rate", "sampling_design_of", "proved", "rate_count", "same_design", "strata_problem",
           "stratum_k", "tables_problem", "total_work", "work_k", "work_table"]
