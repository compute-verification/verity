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

import re
from fractions import Fraction
from typing import Any, Mapping, Sequence

from verity.circuits.codec import canonical_json
from verity.circuits.program import Program
from verity.commitments.randomness import digits
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


def rate_demand(num: int, den: int, amount: int, size: int) -> int:
    """One rated entry's demand on a stratum of `size` proof blocks of amount `amount` (Lean `Flock.Draw.rateDemand`):
    ceil(size * amount * num / den), at least one and at most `size`."""
    return min(size, max(1, (size * amount * num + den - 1) // den))


def rate_k(rates: Sequence[tuple[str, int, int]], amounts: Sequence[int], size: int) -> int:
    """A stratum's draw under the rated rule (Lean `Flock.Draw.rateK`): the largest demand over the entries
    `(table, num, den)`, each at the stratum's amount in its table."""
    return max([0, *(rate_demand(num, den, a, size) for (_, num, den), a in zip(rates, amounts))])


def meets_at(num: int, den: int, n: int, c: int, k: int) -> bool:
    """A uniform k-subset of n proof blocks, each of amount c, meets the rate num/den (Lean `Flock.Draw.meetsAt`)."""
    return k == n or n * c * num <= k * den


def rate_miss(rates: Sequence[tuple[str, int, int]], s: Mapping[str, Any]) -> str | None:
    """Why rated stratum `s` misses a rate of `rates`, or None (Lean `Flock.Draw.rateMiss`)."""
    t, amounts, k, n = s["template"], s["amounts"], s["k"], _size(s["units"])
    if len(amounts) != len(rates):
        return f"stratum '{t}': {len(amounts)} amounts, not one for each of the {len(rates)} rates"
    if k > n:
        return f"stratum '{t}': k = {k} exceeds its {n} proof blocks"
    if k == 0 and n > 0:
        return f"stratum '{t}': k = 0, and every stratum draws at least one"
    for (table, num, den), c in zip(rates, amounts):
        if not meets_at(num, den, n, c, k):
            return (f"stratum '{t}' misses the rate {num}/{den} of table '{table}': "
                    f"k * DEN = {k * den} < n * c * NUM = {n * c * num} (n = {n}, c = {c}, k = {k})")
    return None


def meets_rate(rates: Sequence[tuple[str, int, int]], strata: Sequence[Mapping[str, Any]]) -> str | None:
    """Why rated strata miss an entry's rate, or None (Lean `Flock.Draw.meetsRate`): at least one entry, each DEN
    positive, and every stratum meeting every entry (`rate_miss`)."""
    if not rates:
        return "a rated law needs at least one rate"
    for table, num, den in rates:
        if den == 0:
            return f"table '{table}': a rate needs DEN > 0, not {num}/{den}"
    return next((why for s in strata if (why := rate_miss(rates, s)) is not None), None)


#: Lean `Spot.Draw.RATED_TABLES`: the amount tables a rated law's entry names
RATED_TABLES = ("plain", "work")
_NUM = re.compile(r"[0-9]+")


def _rate_entry(e: Any) -> tuple[str, int, int]:
    """A rated law's entry `{table, rate}` (Lean `Flock.Draw.rateEntryOfJson`)."""
    bad = "a rated law's entry is exactly {table, rate}, its rate NUM/DEN"
    if not isinstance(e, Mapping) or not isinstance(e.get("table"), str) or not isinstance(e.get("rate"), str) or len(e) != 2:
        raise ValueError(bad)
    parts = e["rate"].split("/")
    if len(parts) != 2 or not all(_NUM.fullmatch(x) for x in parts):
        raise ValueError(bad)
    return e["table"], int(parts[0]), int(parts[1])


def _rate_json(e: tuple[str, int, int]) -> dict[str, Any]:
    return {"rate": f"{e[1]}/{e[2]}", "table": e[0]}


def rated_entries(law: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """A rated law's entries (Lean `Spot.Draw.ratedEntries`): each `{table, rate}`, its table one of `RATED_TABLES`, no
    table twice. Raises `ValueError`."""
    rates = law.get("rates")
    if not isinstance(rates, list):
        raise ValueError("a rated law's rates are a list")
    es = [_rate_entry(e) for e in rates]
    for i, (table, _, _) in enumerate(es):
        if table not in RATED_TABLES:
            raise ValueError(f"a rated law's table is plain or work, not '{table}'")
        if any(u == table for u, _, _ in es[:i]):
            raise ValueError(f"a rated law names table '{table}' twice")
    return es


def rated_strata(rates: Sequence[tuple[str, int, int]], works: Mapping[str, tuple[int, int]],
                 strata: Sequence[tuple[str, Sequence[Sequence[int]]]]) -> list[dict[str, Any]]:
    """The rated law's strata at their default k_s (Lean `Flock.Draw.ratedStrata`): each stratum's amount in each entry's
    table, its template's work for `work` and one a proof block otherwise, and `rate_k`."""
    out = []
    for t, rs in strata:
        amounts = [(works[t][0] if t in works else 0) if table == "work" else 1 for table, _, _ in rates]
        out.append({"template": t, "units": [list(r) for r in rs], "amounts": amounts, "k": rate_k(rates, amounts, _size(rs))})
    return out


def rated_why(rates: Sequence[tuple[str, int, int]], strata: Sequence[tuple[str, Any]],
              works: Mapping[str, Any]) -> str | None:
    """Why the entries have no default over the strata and work table, or None (Lean `Flock.Draw.ratedWhy`)."""
    if not rates:
        return "a rated law needs at least one rate"
    for table, num, den in rates:
        if den == 0:
            return f"table '{table}': a rate needs DEN > 0, not {num}/{den}"
    if any(table == "work" for table, _, _ in rates):
        missing = [t for t, _ in strata if t not in works]
        if missing:
            return f"the work table names no work for template '{missing[0]}'"
    return None


def derive(program: Program, query: Mapping[str, Any], sampling_design: Mapping[str, Any],
           table: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The verifier's sampling design for its program and query: `stratified` (the rule, or an expanded sampling design, whose strata are ignored)
    expands to the strata `partition.strata` derives, each with `stratum_k`; `work` expands to the same strata with the
    work table's works and floors (`work_table`), each with `work_k` (Lean `Flock.Draw.deriveWork`); `subset` and
    `bernoulli` are unchanged. Raises `partition.QueryRefused` when the query doesn't apply, and `ValueError` when k isn't
    in [1, population], a work sampling design has no table, the table names no work for a template, or the population does none."""
    kind = sampling_design.get("law")
    if kind not in ("stratified", "work", "rated"):
        return dict(sampling_design)
    strata = P.strata(program, query)
    if kind == "rated":
        rates = rated_entries(sampling_design)
        works: Mapping[str, Any] = {}
        if any(table == "work" for table, _, _ in rates):
            if table is None:
                raise ValueError("a rated law's work rate needs the verifier's work table")
            works = work_table(table)
        if (why := rated_why(rates, strata, works)) is not None:
            raise ValueError(why)
        return {"law": "rated", "rates": [_rate_json(e) for e in rates], "strata": rated_strata(rates, works, strata)}
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


#: Lean `Spot.Draw.COUNTS_AWAIT_THEIR_CLAIM`: the refusal of a rated design whose counts are not the default's
COUNTS_AWAIT_THEIR_CLAIM = ("the rated design's counts are not the default's (rateK): a design's own counts wait on the "
                            "claim that reads them")


def agreed(program: Program, query: Mapping[str, Any], sampling_design: Mapping[str, Any],
           table: Mapping[str, Any] | None, registered: Any) -> dict[str, Any]:
    """The one design R1, the sample's coins and its draw read (Lean `Spot.Draw.agreed`): under a rated law, `registered`
    when it is `derive`'s rated expansion at its own k_s, byte for byte, those k_s meet every rate with the floor
    (`meets_rate`), and they are the default's (`COUNTS_AWAIT_THEIR_CLAIM`); else `derive`'s expansion, which the caller
    holds `registered` to. Raises `ValueError`."""
    if sampling_design.get("law") != "rated":
        return derive(program, query, sampling_design, table)
    mine = derive(program, query, sampling_design, table)
    ss = registered.get("strata") if isinstance(registered, Mapping) else None
    if not isinstance(ss, list):
        raise ValueError("a rated design's strata are a list")
    if len(ss) != len(mine["strata"]):
        raise ValueError(f"the rated design has {len(ss)} strata, not the population's {len(mine['strata'])}")
    ks = [s.get("k") if isinstance(s, Mapping) else None for s in ss]
    if not all(_nat(k) for k in ks):
        raise ValueError("a rated stratum's k is a natural number")
    at = [{**m, "k": k} for m, k in zip(mine["strata"], ks)]
    if not same_design(registered, {"law": "rated", "rates": mine["rates"], "strata": at}):
        raise ValueError("the rated design is not the population's strata at the verifier's rates and amounts")
    if (why := meets_rate(rated_entries(sampling_design), at)) is not None:
        raise ValueError(why)
    if at != mine["strata"]:
        raise ValueError(COUNTS_AWAIT_THEIR_CLAIM)
    return dict(registered)


def sampling_design_of(draw: Mapping[str, Any]) -> dict[str, Any]:
    if draw.get("law") in ("stratified", "work"):
        return {"law": draw["law"], "k": draw.get("k"), "strata": draw.get("strata")}
    if draw.get("law") == "rated":
        return {"law": "rated", "rates": draw.get("rates"), "strata": draw.get("strata")}
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
    ascending, disjoint and non-empty and tile [0, population); each stratum's k is the rule's (a work sampling design's: a floor of
    at least one, a population that does some work, and each k `work_k` above its floor; Lean `workProblem`)."""
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
        rule = work_k(k, s["work"], _size(rs), total, s["floor"]) if work else stratum_k(k, _size(rs), population)
        if not _int(s["k"]) or s["k"] != rule:
            return f"stratum {s['template']!r}: k is not the {'work ' if work else ''}rule's"
        ranges += rs
    for lo, hi in sorted(ranges):
        if lo != at:
            return "the strata don't tile the population"
        at = hi
    return None if at == population else "the strata don't tile the population"


def _rated_stratum(s: Any) -> dict[str, Any]:
    """A rated stratum `{template, units, amounts, k}` (Lean `Flock.Draw.ratedStratumOfJson`). Raises `ValueError`."""
    bad = "U2: a rated stratum is exactly {template, units, amounts, k}"
    if not isinstance(s, Mapping) or not isinstance(s.get("amounts"), list) or len(s) != 4:
        raise ValueError(bad)
    if not all(_nat(a) for a in s["amounts"]):
        raise ValueError("U2: a rated stratum's amount: not a natural number")
    if not isinstance(s.get("template"), str):
        raise ValueError("U2: a stratum is exactly {template, units, k}")
    t = s["template"]
    if not _nat(s.get("k")):
        raise ValueError("U2: a stratum's k: not a natural number")
    rs = s.get("units")
    if not isinstance(rs, list):
        raise ValueError(f"U2: stratum '{t}': units are non-empty [lo, hi) ranges")
    for r in rs:
        if not isinstance(r, list) or len(r) != 2:
            raise ValueError(f"U2: stratum '{t}': units are non-empty [lo, hi) ranges")
        for b in r:
            if not _nat(b):
                raise ValueError(f"U2: stratum '{t}': a range bound: not a natural number")
        if r[0] >= r[1]:
            raise ValueError(f"U2: stratum '{t}': units are non-empty [lo, hi) ranges")
    if not rs:
        raise ValueError(f"U2: stratum '{t}': units are non-empty [lo, hi) ranges")
    if any(b[0] <= a[1] for a, b in zip(rs, rs[1:])):
        raise ValueError(f"U2: stratum '{t}': ranges ascending, disjoint and not adjacent")
    return dict(s)


def rated_problem(sampling_design: Mapping[str, Any], population: int) -> str | None:
    """Why a rated sampling design isn't well formed over `population` proof blocks, or None (Lean `Spot.Draw.ratedMessage`,
    `Flock.Draw.ratedProblem`): its rates and strata parse, distinct templates tile the population, and every rate is
    met with the floor (`meets_rate`). The k_s are the registered ones, not a rule's."""
    rs, ss = sampling_design.get("rates"), sampling_design.get("strata")
    if not isinstance(rs, list) or not isinstance(ss, list):
        return "a rated law's rates and strata are lists"
    try:
        rates = [_rate_entry(e) for e in rs]
        strata = [_rated_stratum(s) for s in ss]
    except ValueError as e:
        return str(e)
    if population < 0:
        return "population"
    if not strata:
        return "U2: a rated law's strata"
    seen: set[str] = set()
    for s in strata:
        if s["template"] in seen:
            return f"U2: template '{s['template']}' twice"
        seen.add(s["template"])
    if (why := meets_rate(rates, strata)) is not None:
        return f"U2: {why}"
    at = 0
    for lo, hi in sorted(r for s in strata for r in s["units"]):
        if lo != at:
            return "U2: the strata don't tile the population"
        at = hi
    return None if at == population else "U2: the strata don't tile the population"


def _ranges_ok(rs: Any) -> bool:
    return isinstance(rs, list) and bool(rs) and all(isinstance(r, list) and len(r) == 2 and all(map(_int, r)) and r[0] < r[1]
                                                    for r in rs)


def fixed_count(sampling_design: Mapping[str, Any]) -> int | None:
    """The proof blocks every sample under `sampling_design` has, when that count is fixed: `k` for `subset` and the strata's sum of k_s for
    `stratified`; None for `bernoulli` (its count is random) and `work` (its closure varies). Lean `Draw.SamplingDesign.fixedCount`."""
    if sampling_design.get("law") == "subset":
        return sampling_design.get("k")
    if sampling_design.get("law") in ("stratified", "rated") and isinstance(sampling_design.get("strata"), list):
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
    if not isinstance(draw, Mapping) or draw.get("law") not in ("subset", "bernoulli", "stratified", "work", "rated"):
        c.fail("not-a-draw", draw)
        return c.verdict()
    keys = {"law", "population", "units", *{"subset": ["k"], "bernoulli": ["p"], "stratified": ["k", "strata"],
                                            "work": ["k", "strata"], "rated": ["rates", "strata"]}[draw["law"]]}
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
    if draw["law"] in ("stratified", "work", "rated"):
        problem = ((rated_problem if draw["law"] == "rated" else strata_problem)(draw, n) if _int(n) else "population")
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


__all__ = ["TABLES_SAMPLING_DESIGN", "check_draw", "check_proved", "derive", "fixed_count", "sampling_design_of",
           "proved", "rate_count", "same_design", "strata_problem", "stratum_k", "tables_problem", "total_work", "work_k",
           "work_table"]
