"""Advice v1's bits by class over trace-corpus traces: each window's charge split by field, the prefix-cache lookups
beside it, and whether the string decodes back to the trace.

The split is `verity_vllm.pipeline.advice_code.by_class` (`bits_by_class`: a window's parts sum to
`codec.bits_by_window`'s charge for it; `closed`: the fields the registration holds at 0, each with its rule and the
rule's line). This module reads a trace as `v1.py` does, encodes it under the uniform book of the table's caps
(`v1.TABLE_CAPS`; under the certified tick `v1.TICK_TABLE_CAPS`, which are the code's own and code nothing), and
reports per window:
- the charge and its parts by class (`by_class.CLASSES`), and the two classes with no field (`by_class.NO_FIELD`);
- the window's length terms;
- the replay's prefix-cache lookups (`codec.Lookup`: every candidate its scheduler read, admitted then or not, a
  preempted request's when it resumes among them), the ones that hit, the admissions (each request's first row), the
  ones that started from a hit, and those hits' blocks, each at A5's window of its step (`codec.charged`).

Per trace it reports the same sums, the engine's own record of the hits it admitted from (the trace's `cache_hit`), and
the round trip (`decode(encode(a)) == a`, the streams too). Where the decoder refuses the string or decodes it to
another advice, the report says which, and gives the first step at which the replay the decoder runs differs from the
engine's schedule, with its window.

    python by_class.py LABEL=DIR ... --stops IDS [--T 600 --ingress client] --out REPORT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verity_vllm.pipeline.advice_code import by_class as BC
from verity_vllm.pipeline.advice_code.codec import (
    AdviceCodeError, arrival_order, bits_by_window, charged, decode_full, encode, lengths, Record, Registered, replay_record,
    streams, windows,
)

HERE = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


V, R, CV = _load("v1"), _load("replay"), _load("code_v1")

COUNTS = ("length_terms", "lookups", "lookups_hit", "admissions", "admissions_hit", "hit_blocks")


def registration(trace: Any, stop_ids: Sequence[int], T: int) -> Registered:
    """The trace's registration: under the certified tick, its tick and the code's own book (`codec.TICK_CAPS`'s, the
    table's); otherwise the uniform book of the table's caps."""
    if trace.releases is not None:
        return Registered(trace.engine, tuple(stop_ids), T=T, tick=trace.releases.tick)
    return Registered(trace.engine, tuple(stop_ids), T=T, book=CV.Book.uniform(V.TABLE_CAPS))


def replayed(advice: Any, record: Record, reg: Registered) -> tuple[Any, list[int], list[list[Any]], list[int]]:
    """The replay v1 encodes the advice under (`codec.replay_record`, each request arrived from its first row's step
    on), the arrival order, each step's lookups, and A5's window of each step (`codec.charged`)."""
    order = arrival_order(advice, record, reg.row_order)
    index = {record.requests[k].id: e for e, k in enumerate(order)}
    first = [-1] * len(order)
    for s, st in enumerate(advice.steps):
        for x in st.rows:
            e = index[advice.requests[x.request].id]
            if first[e] < 0:
                first[e] = s
    rp, _, looks = replay_record(record, reg, reg.engine, order, lambda i, k: k >= first[i])
    return rp, order, looks, charged(record, reg, order, rp, windows(record, reg.T))[0]


def classes(reg: Registered) -> dict[str, dict[str, Any]]:
    """Each class's standing under the registration: `charged`; `0 by construction`, with the rules that hold each of its
    fields at 0; or `no field`, with the rules that say what the code does instead. Each rule has its file and line."""
    def cite(rules: Sequence[str]) -> list[dict[str, Any]]:
        out = []
        for name in dict.fromkeys(rules):
            file, line = BC.where(name)
            out.append({"rule": name, "why": BC.RULES[name].why, "file": f"advice_code/{file}", "line": line})
        return out

    shut = BC.closed(reg)
    out: dict[str, dict[str, Any]] = {}
    for name, fields in BC.CLASSES.items():
        if all(f in shut for f in fields):
            out[name] = {"standing": "0 by construction", "rules": cite([r for f in fields for r in shut[f]])}
        else:
            out[name] = {"standing": "charged", "rules": []}
    for name, rules in BC.NO_FIELD.items():
        out[name] = {"standing": "no field", "rules": cite(rules)}
    return out


def report(label: str, path: str, stop_ids: Sequence[int], T: int = 600, ingress: str = "client") -> dict[str, Any]:
    """One trace-corpus trace's report (the module's docstring)."""
    tr, rec, adv = V._trace(path, stop_ids, ingress)
    return split(label, tr, rec, adv, registration(tr, stop_ids, T))


def split(label: str, trace: Any, record: Record, advice: Any, reg: Registered) -> dict[str, Any]:
    """`report` of a trace already read, under the registration `reg`."""
    data = encode(advice, record, reg, check=False)
    refused = None
    try:
        got = decode_full(data, record, reg)
        back = got.advice == advice and got.streams == streams(advice, record)
    except AdviceCodeError as e:
        back, refused = False, str(e)
    charge, parts = bits_by_window(data, record, reg), BC.bits_by_class(data, record, reg)
    rp, order, looks, at = replayed(advice, record, reg)
    B = reg.engine.block_size
    rows = [{"window": w, "charge": charge[w], "fields": parts[w],
             "classes": {c: sum(parts[w][f] for f in fs) for c, fs in BC.CLASSES.items()}, **dict.fromkeys(COUNTS, 0)}
            for w in sorted(charge)]
    for r, ks in zip(rows, lengths(record, reg)):
        if sum(r["fields"].values()) != r["charge"]:
            raise ValueError(f"window {r['window']}: its parts sum to {sum(r['fields'].values())} bits of {r['charge']}")
        r["length_terms"] = len(ks)
    seen: set[int] = set()
    for s, (st, ls) in enumerate(zip(rp.steps, looks)):
        r = rows[at[s]]
        r["lookups"] += len(ls)
        r["lookups_hit"] += sum(1 for x in ls if x.hit)
        for i, start, _ in st.rows:
            if i not in seen:
                seen.add(i)
                r["admissions"] += 1
                r["admissions_hit"] += start > 0
                r["hit_blocks"] += start // B
    index = {record.requests[k].id: e for e, k in enumerate(order)}
    theirs = [[(index[advice.requests[x.request].id], x.start, x.n) for x in st.rows] for st in advice.steps]
    m = R.compare(rp.steps, [R.Step(rows=r) for r in theirs])
    differs = None
    if m.first_mismatch is not None:
        differs = {"step": m.first_mismatch, "window": at[m.first_mismatch] if m.first_mismatch < len(at) else len(rows) - 1,
                   "matched": m.matched, "steps": m.steps}
    hits = [k for st in trace.steps for k in st.cache_hit.values() if k]
    return {"trace": label, "T": reg.T, "tick": None if reg.tick is None else asdict(reg.tick),
            "prefix_caching": reg.engine.enable_prefix_caching, "requests": len(record.requests), "steps": len(rp.steps),
            "windows": len(rows), "bits": sum(charge.values()), "bytes": len(data), "book": V.book_of(reg).digest(),
            "round_trip": back, "refused": refused, "first_step_that_differs": differs,
            "classes": classes(reg), "by_window": rows,
            "total": {"charge": sum(charge.values()),
                      "classes": {c: sum(r["classes"][c] for r in rows) for c in BC.CLASSES},
                      "fields": {f: sum(r["fields"][f] for r in rows) for f in BC.FIELDS},
                      **{c: sum(r[c] for r in rows) for c in COUNTS}},
            "engine_hits": {"admissions_hit": len(hits), "hit_blocks": sum(k // B for k in hits), "hit_tokens": sum(hits)}}


def _share(part: int, whole: int) -> str:
    return f"{part:,} of {whole:,}" + (f" ({100 * part / whole:.1f}%)" if whole else "")


def render(row: dict[str, Any]) -> str:
    """One trace's report as Markdown: the table by window and class, the lookups, and each class's standing."""
    if "error" in row:
        return f"**{row['trace']}**: error: {row['error']}\n"
    names = list(BC.CLASSES)
    tick = "under the certified tick" if row["tick"] else "admitted on arrival"
    cache = "on" if row["prefix_caching"] else "off"
    t = row["total"]
    out = [f"**{row['trace']}**: {row['requests']:,} requests, {row['steps']:,} steps, {tick}, the prefix cache {cache}; "
           f"{row['windows']} windows of T = {row['T']} buckets; {row['bits']:,} bits.", "",
           "| window | charge | " + " | ".join(names) + " | " + " | ".join(BC.NO_FIELD) + " | length terms | lookups | "
           "of them hit | admissions | of them from a hit | their hit blocks |",
           "|---|" + "---|" * (len(names) + len(BC.NO_FIELD) + 6)]

    def line(head: str, r: dict[str, Any]) -> str:
        cells = [f"{r['classes'][c]:,}" for c in names] + ["no field"] * len(BC.NO_FIELD)
        return (f"| {head} | {r['charge']:,} | " + " | ".join(cells) + f" | {r['length_terms']:,} | {r['lookups']:,} | "
                f"{r['lookups_hit']:,} | {r['admissions']:,} | {r['admissions_hit']:,} | {r['hit_blocks']:,} |")

    out += [line(str(r["window"]), r) for r in row["by_window"]]
    out += [line("all", t), ""]
    f = t["fields"]
    out.append(f"The lengths are {f['output_len']:,} bits of output_len and {f['finish']:,} of finish; the admission is "
               f"{f['decisions']:,} bits of decisions and {f['order']:,} of order entries.")
    e = row["engine_hits"]
    out.append(f"Lookups that hit: {_share(t['lookups_hit'], t['lookups'])}. Admissions from a hit: "
               f"{_share(t['admissions_hit'], t['admissions'])}, {t['hit_blocks']:,} blocks; the engine's own record: "
               f"{e['admissions_hit']:,} admissions, {e['hit_blocks']:,} blocks.")
    d = row["first_step_that_differs"]
    back = "yes." if row["round_trip"] else "NO."
    if row["refused"] is not None:
        back += f" The decoder refuses the string: {row['refused']}."
    elif not row["round_trip"]:
        back += " The string decodes to another advice."
    if d is not None:
        back += (f" The replay's schedule differs from the engine's from step {d['step']}, in window {d['window']} "
                 f"({d['matched']:,} of {d['steps']:,} steps match).")
    out.append("decode(encode(a)) == a: " + back)
    out.append("")
    for name, c in row["classes"].items():
        why = "; ".join(f"{r['why']} (`{r['file']}:{r['line']}`)" for r in c["rules"])
        out.append(f"- {name}: {c['standing']}" + (f": {why}." if why else "."))
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Advice v1's bits by class and window over trace-corpus traces, with the "
                                             "prefix-cache lookups and the round trip.")
    ap.add_argument("traces", nargs="+", help="LABEL=DIR: a trace's directory (steps.jsonl, requests.jsonl, meta.json, "
                                              "server.log)")
    ap.add_argument("--stops", default="128001", help="the registered stop ids, comma-separated, EOS first")
    ap.add_argument("--T", type=int, default=600, help="the window, in the trace's certified buckets: its tick's τ_b, "
                                                          "100 ms without the tick")
    ap.add_argument("--ingress", choices=V.INGRESS, default="client", help="the certified order's stand-in, as v1.py's")
    ap.add_argument("--out", type=Path, required=True, help="the report's JSON")
    a = ap.parse_args(argv)
    stops = tuple(int(s) for s in a.stops.split(","))
    rows = [V._guarded(report, *t.split("=", 1), stops, a.T, a.ingress) for t in a.traces]
    a.out.write_text(json.dumps({"T": a.T, "stops": stops, "ingress": a.ingress, "rows": rows}, indent=1) + "\n")
    print("\n".join(render(r) for r in rows), end="")
    return 0 if all("error" not in r and r["round_trip"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
