"""`verity bench prove`'s census table (`prove_records`), for a `steady_only` workload (`prove_common.Workload`): its
census of record read alone, request by request, and why its totals are refused.

The table gives each block type's proof blocks in each request and summed over them (the row as served), its AND
gates where the census gives them (a block's, times its blocks), and the steady-state row that rates it (`rated`): a
measurement at the stated prover count, timed at the stated host cores or more, the rows `prove_flock.Flock` prices a
type from. A gap, a Definition whose served rows no block of the census's cut holds, is a row of its own with its rows
and why: it is never counted as zero blocks.

The totals are refused, each reason named (`refusal`): a block type with blocks and no steady-state rate (under the
cut of record, where one is labelled), a gap, and each other slot the totals read that isn't labelled, with the label
that would fill it. A refused table records no measurement, and the command exits 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
                        "why": (f"no {what} of record: nothing in the store carries `{RUN_OF_RECORD}={w.record(slot)}` "
                                f"and no `superseded_by` (label one: {_label(w.record(slot))})")})
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


def _count(x: int | None) -> str:
    return "—" if x is None else f"{x:,}"


def _record(target: str) -> str:
    return target[:16] if target.startswith("art:") else target


def render(doc: Mapping[str, Any]) -> list[str]:
    """The census table as Markdown lines, then the totals' refusal where they are refused."""

    c = doc["census"]
    names = list(c["requests"])
    head = " | ".join(names)
    lines = ["## The census of record", "",
             (f"{c['art']} (`{RUN_OF_RECORD}={c['record']}`, run {c['run']}), {len(names)} "
              f"request{'' if len(names) == 1 else 's'}, summed: the row as served."), "",
             "| request | prefill tokens | decode tokens | proof blocks |", "| --- | --- | --- | --- |"]
    lines += [f"| {n} | {_count(t.get('prefill'))} | {_count(t.get('decode'))} | {_count(c['blocks'][n])} |"
              for n, t in c["requests"].items()]
    lines += [f"| summed | {_count(sum(t.get('prefill', 0) for t in c['requests'].values()))} | "
              f"{_count(sum(t.get('decode', 0) for t in c['requests'].values()))} | {_count(sum(c['blocks'].values()))} |",
              "", (f"Proof blocks a block type, each request's and summed; AND gates where the census gives a block's; "
                   f"the steady-state rate of record at {c['provers']} prover{'' if c['provers'] == 1 else 's'} and "
                   f"{c['cores']} CPU cores a card or more:"), "",
              f"| block type | k_log | ANDs a block | {head} | summed | AND gates | steady-state rate |",
              "| --- | --- | --- | " + "--- | " * len(names) + "--- | --- | --- |"]
    for r in c["rows"]:
        rate = ("none" if r["rate"] is None else
                f"`{r['rate']['type']}` at {r['rate']['provers']} prover(s), {r['rate']['cores']} cores "
                f"({_record(r['rate']['record'])})")
        lines.append(f"| `{r['block']}` | {'—' if r['k_log'] is None else r['k_log']} | {_count(r['ands'])} | "
                     + " | ".join(_count(r["requests"][n]) for n in names)
                     + f" | {_count(r['total'])} | {'—' if r['gates'] is None else format(r['gates'], '.3e')} | {rate} |")
    rated_rows = [r for r in c["rows"] if r["rate"] is not None]
    lines += ["", (f"Rated: {len(rated_rows)} of {len(c['rows'])} block types, "
                   f"{sum(r['total'] for r in rated_rows):,} of {sum(r['total'] for r in c['rows']):,} proof blocks.")]
    if c["gaps"]:
        lines += ["", "Gaps: served rows no block of the census's cut holds, a row each, in no count above:", "",
                  f"| Definition | why | ANDs a row | {head} | summed rows |",
                  "| --- | --- | --- | " + "--- | " * len(names) + "--- |"]
        lines += [f"| `{g['block']}` | {g['why'] or '—'} | {_count(g['ands'])} | "
                  + " | ".join(_count(g["requests"][n]) for n in names) + f" | {_count(g['total'])} |" for g in c["gaps"]]
    else:
        lines += ["", "Gaps: none."]
    if (why := doc["refusal"]) is not None:
        lines += ["", "## Prover GPUs per serving GPU: refused", "",
                  "No total is given while any of these stands:", ""]
        if why["no_rate"]:
            lines.append(f"- {len(why['no_rate'])} block type(s) with blocks and no steady-state rate at {c['provers']} "
                         f"prover(s) and {c['cores']} cores or more (label each run of its rate: {_label(c['steady'])}):")
            lines += [f"  - `{r['block']}`, {r['per_token']:.4g} blocks a served token" for r in why["no_rate"]]
        if why["gaps"]:
            lines.append(f"- {len(why['gaps'])} gap(s), served rows no block holds, each with its rows and why in the "
                         "gaps above: " + ", ".join(f"`{g['block']}`" for g in why["gaps"]))
        lines += [f"- {m['slot']} (`{m['record']}`): {m['why']}" for m in why["slots"]]
    if doc.get("refused") and why is not None:
        lines += ["", "Refused (not read):", ""]
        lines += [f"- {x['target']} (`{x['record']}`): {x['why']}" for x in doc["refused"]]
    return lines
