"""`verity bench prove`'s census table (`prove_records`), for a `steady_only` workload (`prove_common.Workload`): its
census of record read alone, request by request, and why its totals are refused.

The table gives each block type's proof blocks in each request and summed over them (the row as served), its AND
gates where the census gives them (a block's, times its blocks), and the steady-state row that rates it (`rated`): a
measurement at the stated prover count, timed at the stated host cores or more, the rows `prove_flock.Flock` prices a
type from. A gap, a Definition whose served rows no block of the census's cut holds, is a row of its own with its rows
and why: it is never counted as zero blocks. The text (`render`) prints a run of one family's Definitions that differ in
one static and in nothing else it shows as one line (`runs`); the stored table keeps every Definition.

The totals are refused, each reason named (`refusal`): a block type with blocks and no steady-state rate (under the
cut of record, where one is labelled), a gap, and each other slot the totals read that isn't labelled, with the label
that would fill it. A refused table records no measurement, and the command exits 1.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

from verity_bench import prove as P
from verity_bench import prove_flock as PF
from verity_bench.prove_common import (
    CENSUS,
    CUT,
    LEAN,
    SAMPLING,
    SINGLE,
    STEADY,
    WARMED,
    Workload,
    _label,
    _superseded,
    card,
    of_record,
)
from verity_bench.prove_store import census_docs, steady_of_record
from verity_bench.served_table import RUN_OF_RECORD, records

#: the census table a run's result keeps beside its measurements, relative to the run's directory
CENSUS_TABLE = "out/prove-census.json"
#: the slots of one target the totals read beside the census, each as a refusal names it
ONE_OF_RECORD = {CUT: "cut", SINGLE: "single-session card", SAMPLING: "sampling plan"}
#: the slots of any number of runs the totals read, each as a refusal names it
ANY_OF_RECORD = {LEAN: "Lean verifier run", WARMED: "warm-session run"}


def rated(steady: Sequence[PF.Steady], provers: int, cores: int) -> dict[str, PF.Steady]:
    """Each Definition's steady-state rate, by its id: a measured row at `provers` provers timed at `cores` cores or
    more, the newest run's."""

    out: dict[str, PF.Steady] = {}
    for s in sorted(steady, key=lambda s: s.run, reverse=True):
        if s.measured and s.provers == provers and s.cores >= cores:
            out.setdefault(s.definition, s)
    return out


def table(docs: Sequence[tuple[str | None, Mapping[str, Any]]], rates: Mapping[str, PF.Steady]) -> dict:
    """The census table of each request's `census.json` (a request named None is `request`): its requests' tokens and
    proof blocks, a row a block type with blocks and a row a gap with rows, the most first."""

    names = [r or "request" for r, _ in docs]
    found: dict[str, dict[str, dict]] = {"definitions": {}, "gaps": {}}
    for name, (_, doc) in zip(names, docs, strict=True):
        for part, rows in found.items():
            for spec, d in doc.get(part, {}).items():
                row = rows.setdefault(spec, {"block": spec, "k_log": d.get("k_log"), "ands": d.get("ands"),
                                             "why": d.get("why"), "requests": {}})
                row["requests"][name] = d["prefill"] + d["decode"]
    for rows in found.values():
        for row in rows.values():
            row["requests"] = {n: row["requests"].get(n, 0) for n in names}
            row["total"] = sum(row["requests"].values())
    blocks = sorted((r for r in found["definitions"].values() if r["total"]), key=lambda r: (-r["total"], r["block"]))
    for row in blocks:
        del row["why"]
        row["gates"] = None if row["ands"] is None else row["ands"] * row["total"]
        s = rates.get(P.definition_id(row["block"]))
        row["rate"] = None if s is None else {"record": s.record, "type": s.type, "provers": s.provers, "cores": s.cores}
    gaps = sorted((r for r in found["gaps"].values() if r["total"]), key=lambda r: (-r["total"], r["block"]))
    for row in gaps:
        del row["k_log"]
    return {"requests": {n: dict(doc["tokens"]) for n, (_, doc) in zip(names, docs, strict=True)},
            "blocks": {n: sum(r["requests"][n] for r in blocks) for n in names}, "rows": blocks, "gaps": gaps}


def slots(store: Any, w: Workload, system: str, sampling: str | None) -> list[dict]:
    """Each slot beside the census and the steady-state record that the totals read and isn't labelled, with why and
    the label that fills it: the cut, the single-session card, SPOT's sampling plan, the Lean verifier's runs and the
    warm-session runs."""

    out = []
    for slot, what in ONE_OF_RECORD.items():
        if slot == SAMPLING and system != "spot":
            continue
        try:
            of_record(store, w.record(slot), what, sampling if slot == SAMPLING else None)
        except ValueError as e:
            out.append({"slot": slot, "record": w.record(slot), "why": str(e)})
    for slot, what in ANY_OF_RECORD.items():
        if not any(not _superseded(store, t) for t in records(store).get(w.record(slot), [])):
            out.append({"slot": slot, "record": w.record(slot),
                        "why": (f"no {what} of record: no run labelled `{RUN_OF_RECORD}={w.record(slot)}` that is not "
                                f"superseded (label one: {_label(w.record(slot))})")})
    return out


def refusal(census: P.Census, found: Mapping[str, Any], rates: Mapping[str, PF.Steady],
            missing: Sequence[dict]) -> dict | None:
    """Why the totals of `census` (the census of record, under its cut where one is labelled) aren't given: each block
    type with blocks and no rate, each gap of the census table `found` and each slot `missing`; None where none."""

    no_rate = sorted(({"block": b.name, "per_token": b.per_token} for b in census.blocks
                      if P.definition_id(b.name) not in rates), key=lambda r: (-r["per_token"], r["block"]))
    gaps = [{"block": g["block"], "rows": g["total"], "why": g["why"]} for g in found["gaps"]]
    if not (no_rate or gaps or missing):
        return None
    return {"no_rate": no_rate, "gaps": gaps, "slots": list(missing)}


def read(store: Any, w: Workload, system: str, sampling: str | None, provers: int, cores: int) -> dict:
    """The workload's census table (`table`), what holds it, the steady-state rows left out with why, and the totals'
    refusal (`refusal`, None where they can be given)."""

    art, manifest, docs = census_docs(store, w)
    steady, _, left_out = steady_of_record(store, w)
    rates = rated(steady, provers, cores)
    whole = P.Census.of(docs[0][1] if len(docs) == 1 else P.summed([d for _, d in docs]))
    missing = slots(store, w, system, sampling)
    if not any(m["slot"] == CUT for m in missing):
        try:
            whole = whole.recut(card(store, of_record(store, w.record(CUT), "cut"), "partition", w))
        except ValueError as e:
            missing.insert(0, {"slot": CUT, "record": w.record(CUT), "why": str(e)})
    found = table(docs, rates) | {"art": art, "record": w.record(CENSUS), "run": manifest.meta.get("run"),
                                  "steady": w.record(STEADY), "provers": provers, "cores": cores}
    return {"census": found, "refused": left_out, "refusal": refusal(whole, found, rates, missing)}


# ---- the text ----------------------------------------------------------------------------------------------------------------

#: a reason's estimated ANDs, which a run of Definitions states in its ANDs column instead
_AT_ANDS = re.compile(r" at [\d,]+ ANDs$")


def _count(x: int | None) -> str:
    return "—" if x is None else f"{x:,}"


def _record(target: str) -> str:
    return target[:16] if target.startswith("art:") else target


def _plural(n: float, one: str, many: str | None = None) -> str:
    return one if n == 1 else many or f"{one}s"


def _statics(spec: str) -> tuple[str, list[tuple[str, str]]] | None:
    """A Definition's family and its statics in order, each a name and its value as the spec spells them; None where the
    spec isn't `FAMILY{NAME=VALUE,...}`."""

    family, brace, body = spec.partition("{")
    if not brace or not body.endswith("}"):
        return None
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(body[:-1]):
        depth += (ch in "{[(") - (ch in "}])")
        if ch == "," and not depth:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:-1])
    statics = [p.partition("=") for p in parts]
    return None if any(not eq for _, eq, _ in statics) else (family, [(k, v) for k, _, v in statics])


def runs(rows: Sequence[Mapping[str, Any]], kind: Callable[[Mapping[str, Any]], Hashable]) -> list[dict]:
    """`rows` (each a Definition by its spec, `block`) as the text prints them: each run of Definitions of one family
    whose statics differ only in one integer static, consecutive in it and of one `kind`, is one (`block` its spec with
    that static as `LO..HI`, `of` its rows in that static's order); a Definition in no such run is one alone. A
    family's static is the one that makes the fewest runs. Every row is in exactly one."""

    out: list[dict] = []
    families: dict[tuple, list[tuple[Mapping[str, Any], dict[str, str]]]] = {}
    for r in rows:
        if (s := _statics(r["block"])) is None:
            out.append({"block": r["block"], "of": [r]})
        else:
            families.setdefault((s[0], tuple(k for k, _ in s[1])), []).append((r, dict(s[1])))
    for (family, keys), members in families.items():
        best: tuple[str | None, list[list[tuple[int, Mapping[str, Any], dict[str, str]]]]] = (
            None, [[(0, r, st)] for r, st in members])
        for key in keys:
            if not all(re.fullmatch(r"-?\d+", st[key]) for _, st in members):
                continue
            held: dict[tuple, list[tuple[int, Mapping[str, Any], dict[str, str]]]] = {}
            for r, st in members:
                held.setdefault((tuple(v for k, v in st.items() if k != key), kind(r)), []).append((int(st[key]), r, st))
            found: list[list[tuple[int, Mapping[str, Any], dict[str, str]]]] = []
            for xs in held.values():
                xs.sort(key=lambda x: x[0])
                found.append([xs[0]])
                for x in xs[1:]:
                    if x[0] == found[-1][-1][0] + 1:
                        found[-1].append(x)
                    else:
                        found.append([x])
            if len(found) < len(best[1]):
                best = (key, found)
        key, found = best
        for run in found:
            if len(run) == 1:
                out.append({"block": run[0][1]["block"], "of": [run[0][1]]})
                continue
            spelled = ",".join(f"{k}={run[0][0]}..{run[-1][0]}" if k == key else f"{k}={v}" for k, v in run[0][2].items())
            out.append({"block": f"{family}{{{spelled}}}", "of": [r for _, r, _ in run]})
    return out


def _pattern(r: Mapping[str, Any]) -> tuple[bool, ...]:
    return tuple(bool(n) for n in r["requests"].values())


def _reason(why: str | None) -> str:
    return _AT_ANDS.sub(" at its ANDs", why or "—")


def printed(rows: Sequence[Mapping[str, Any]], kind: Callable[[Mapping[str, Any]], Hashable]) -> list[dict]:
    """The lines of a census table's rows (`table`'s `rows` or `gaps`) as the text prints them (`runs`), the most served
    rows first, each with its Definitions' count and their summed rows in each request and in all."""

    out = []
    for u in runs(rows, kind):
        of = u["of"]
        out.append(u | {"definitions": len(of), "total": sum(r["total"] for r in of),
                        "requests": {n: sum(r["requests"][n] for r in of) for n in of[0]["requests"]}})
    return sorted(out, key=lambda u: (-u["total"], u["block"]))


def unrated(rows: Sequence[Mapping[str, Any]], no_rate: Sequence[Mapping[str, Any]]) -> list[tuple[float, dict]]:
    """The refusal's block types with no rate (`refusal`'s `no_rate`) in the block-type table's runs (`rows`, `runs` by
    `_block_kind`), each with its summed proof blocks a served token, the most first."""

    kind = {r["block"]: _block_kind(r) for r in rows}
    found = [(sum(r["per_token"] for r in u["of"]), u) for u in runs(no_rate, lambda r: kind.get(r["block"]))]
    return sorted(found, key=lambda xu: (-xu[0], xu[1]["block"]))


def _block_kind(r: Mapping[str, Any]) -> Hashable:
    return r["k_log"], json.dumps(r["rate"], sort_keys=True), _pattern(r)


def _gap_kind(r: Mapping[str, Any]) -> Hashable:
    return _reason(r["why"]), _pattern(r)


def _ands(of: Sequence[Mapping[str, Any]]) -> str:
    xs = [r["ands"] for r in of]
    if any(x is None for x in xs):
        return "—"
    return _count(xs[0]) if min(xs) == max(xs) else f"{min(xs):,} to {max(xs):,}"


def render(doc: Mapping[str, Any]) -> list[str]:
    """The census table as Markdown lines, then the totals' refusal where they are refused."""

    from verity.commitments.periods.interface import K_MAX

    c = doc["census"]
    names = list(c["requests"])
    head = " | ".join(names)
    why = doc["refusal"]
    lines = ["## The census of record", "",
             (f"{c['art']} (`{RUN_OF_RECORD}={c['record']}`, run {c['run']}), {len(names)} "
              f"request{'' if len(names) == 1 else 's'}, summed: the row as served.")]
    if why is not None and doc.get("system") == "spot":
        lines += ["", ("System `spot`: only the refusal's `spot-sampling` slot, SPOT's sampling plan of record, is "
                       "SPOT's; nothing else here is system-specific yet.")]
    elif why is not None and doc.get("system"):
        lines += ["", (f"System `{doc['system']}`: nothing here is system-specific yet; `spot` adds only the refusal's "
                       "`spot-sampling` slot.")]
    lines += ["", "Each request's prompt (prefill) and decoded tokens, and its proof blocks:", "",
              "| request | prefill tokens | decode tokens | proof blocks |", "| --- | --- | --- | --- |"]
    lines += [f"| {n} | {_count(t.get('prefill'))} | {_count(t.get('decode'))} | {_count(c['blocks'][n])} |"
              for n, t in c["requests"].items()]
    lines += [f"| summed | {_count(sum(t.get('prefill', 0) for t in c['requests'].values()))} | "
              f"{_count(sum(t.get('decode', 0) for t in c['requests'].values()))} | {_count(sum(c['blocks'].values()))} |",
              "", ("Proof blocks a block type, in each request and summed. A line is one block type, or a run of block "
                   "types of one family that differ only in one static, consecutive in it (`T=1..8`: eight block types), "
                   "with one `k_log`, one steady-state rate and served rows in the same requests; a run's counts are its "
                   "block types' sums, and `block types` says how many it holds."), "",
              f"- `k_log`: a proof block's size, log2 of its witness bits; a block may be at most 2^{K_MAX}.",
              ("- `ANDs a block`: the census's estimate of a block's AND gates, not a staged statement's count (on a "
               "run, its least to its most)."),
              "- `AND gates`: the census's estimate of all its proof blocks' AND gates, `ANDs a block` times its blocks.",
              (f"- `steady-state rate`: the steady-state measurement of record that rates the block type at "
               f"{c['provers']} {_plural(c['provers'], 'prover')} and {c['cores']} CPU cores a card or more, by its staged "
               "type, provers, cores and record (its rate is prover GPU-seconds a statement of that type); `none` where "
               "no measurement rates it."), "",
              f"| block type | block types | k_log | ANDs a block | {head} | summed | AND gates | steady-state rate |",
              "| --- | --- | --- | --- | " + "--- | " * len(names) + "--- | --- | --- |"]
    for u in printed(c["rows"], _block_kind):
        r, of = u["of"][0], u["of"]
        rate = ("none" if r["rate"] is None else
                f"`{r['rate']['type']}` at {r['rate']['provers']} prover(s), {r['rate']['cores']} cores "
                f"({_record(r['rate']['record'])})")
        gates = None if any(x["gates"] is None for x in of) else sum(x["gates"] for x in of)
        lines.append(f"| `{u['block']}` | {u['definitions']} | {'—' if r['k_log'] is None else r['k_log']} | "
                     f"{_ands(of)} | " + " | ".join(_count(u["requests"][n]) for n in names)
                     + f" | {_count(u['total'])} | {_count(gates)} | {rate} |")
    rated_rows = [r for r in c["rows"] if r["rate"] is not None]
    lines += ["", (f"Rated: {len(rated_rows)} of {len(c['rows'])} block types, "
                   f"{sum(r['total'] for r in rated_rows):,} of {sum(r['total'] for r in c['rows']):,} proof blocks.")]
    if c["gaps"]:
        lines += ["", ("Gaps: Definitions whose served rows no block of the census's cut holds, counted in no proof "
                       "blocks above. A line is one Definition or a run of them as above (`Definitions` says how many), "
                       "with its served rows in each request and summed. A reason is one of two measures, each against "
                       f"the 2^{K_MAX} limit and tried in this order:"), "",
                  ("- `its rows alone need a 2^N block`: the block that hashing the Definition's inputs and outputs "
                   "needs, a lower bound whatever its ANDs: its block is 2^N or more."),
                  ("- `a 2^N block at X ANDs`: the block at the census's estimate of its ANDs (`at its ANDs` on a run: "
                   "its `ANDs a served row`), given only where its inputs and outputs alone fit."), "",
                  ("So a run at 2^28 by its rows beside one at 2^29 by its ANDs is two measures, not a smaller block for "
                   "the larger Definition."), "",
                  f"| Definition | Definitions | why | ANDs a served row | {head} | summed served rows |",
                  "| --- | --- | --- | --- | " + "--- | " * len(names) + "--- |"]
        for u in printed(c["gaps"], _gap_kind):
            of = u["of"]
            reason = (of[0]["why"] or "—") if len(of) == 1 else _reason(of[0]["why"])
            lines.append(f"| `{u['block']}` | {u['definitions']} | {reason} | {_ands(of)} | "
                         + " | ".join(_count(u["requests"][n]) for n in names) + f" | {_count(u['total'])} |")
    else:
        lines += ["", "Gaps: none."]
    if why is not None:
        lines += ["", "## Prover GPUs per serving GPU: refused", "",
                  "No total is given while any of these stands:", ""]
        if why["no_rate"]:
            lines.append(f"- {len(why['no_rate'])} block type(s) with blocks and no steady-state rate at {c['provers']} "
                         f"prover(s) and {c['cores']} cores or more (label each run of its rate: {_label(c['steady'])}), "
                         "each with its proof blocks a served token, a served token being a prompt or a decoded token "
                         "(as GLM-5.3's table counts it), and runs as above:")
            for x, u in unrated(c["rows"], why["no_rate"]):
                n = len(u["of"])
                lines.append(f"  - `{u['block']}`{'' if n == 1 else f' ({n} block types)'}, "
                             f"{x:.4g} {_plural(x, 'block')} a served token")
        if why["gaps"]:
            kinds: dict[tuple[str, str], list[int]] = {}
            for g in why["gaps"]:
                k = kinds.setdefault((g["block"].partition("{")[0], _reason(g["why"])), [0, 0])
                k[0] += 1
                k[1] += g["rows"]
            lines.append(f"- {len(why['gaps'])} {_plural(len(why['gaps']), 'gap')}, Definitions with "
                         f"{sum(g['rows'] for g in why['gaps']):,} served rows no block holds, by family and reason; "
                         "each is named, with its served rows and why, in the gaps table above and in the census table "
                         f"kept as `{CENSUS_TABLE}` under a research run (`refusal.gaps`; `--json` prints it):")
            lines += [f"  - `{f}`, {r}: {n} {_plural(n, 'Definition')}, {rows:,} served rows"
                      for (f, r), (n, rows) in sorted(kinds.items(), key=lambda x: (-x[1][1], x[0]))]
        lines += [f"- {m['slot']} (`{m['record']}`): {m['why']}" for m in why["slots"]]
    if doc.get("refused") and why is not None:
        lines += ["", "Refused (not read):", ""]
        lines += [f"- {x['target']} (`{x['record']}`): {x['why']}" for x in doc["refused"]]
    return lines
