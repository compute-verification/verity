"""Advice v1's bits by class over trace-corpus traces: each window's charge split by class, each class's events
beside its bits, and whether the string decodes back to the trace.

The split is `verity_vllm.pipeline.advice_code.by_class` (`bits_by_class`: a window's parts sum to
`codec.bits_by_window`'s charge for it; `closed`: the fields the registration holds at 0, each with its rule and the
rule's line). This module reads a trace as `v1.py` does, encodes it under the uniform book of the table's caps
(`v1.TABLE_CAPS`; under the certified tick `v1.TICK_TABLE_CAPS`, which are the code's own and code nothing), and
reports per window the charge, its parts by class (`by_class.CLASSES`), the two classes with no field
(`by_class.NO_FIELD`), and the events (`EVENTS`) that say whether each class's case arose in the window at all: a
class with 0 bits and none of its events (`OF`) reads "not exercised", since its 0 then says nothing.

**The events**, each at A5's window (`codec.charged`) of the step, arrival or length term it belongs to:
- from the replay's lookups (`codec.Lookup`: every candidate its scheduler read, admitted then or not, a preempted
  request's when it resumes among them): the lookups, the ones that hit, the admissions (each request's first row),
  the ones that started from a hit and those hits' blocks;
- from the replay's block pool and scheduler (`by_class.observing`): the cached blocks the pool evicted, the admissions
  whose lookup an eviction cut short and the blocks they lost, the requests a step left waiting (and those it left
  waiting on a full batch, `max_num_seqs` running or the step's token budget spent), the steps that left one waiting
  by what held it, the preemptions and the resumptions;
- from the certified record: the requests released and the distinct release buckets they fall in (a request's
  last-frame mark, which under the certified tick is what admits it), and how each request ended (`stop` or
  `length`), at its length term's window.
A window's requests that waited and its release buckets are distinct counts, so the trace's are counted over the
trace and are not the windows' sums.

**Aborts** (`aborts`). The code has no field for an abort: the reader refuses a trace that holds one, so a trace that
has a table holds none. What the trace's directory records of them is read without the reader and reported either
way: the requests the engine freed by finish reason, the requests released that the engine never freed, and the
connections the network certifier failed, by reason and by whether their request had been released. A trace the reader
refuses is one row: the refusal and those counts.

Per trace it also reports the engine's own record of the hits it admitted from (the trace's `cache_hit`) and the round
trip (`decode(encode(a)) == a`, the streams too). Where the decoder refuses the string or decodes it to another advice,
the report says which, and gives the first step at which the replay the decoder runs differs from the engine's
schedule, with its window.

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

#: a window's counts of events (the module's docstring)
EVENTS = ("lookups", "lookups_hit", "admissions", "admissions_hit", "hit_blocks", "evicted_blocks", "admissions_cut",
          "cut_blocks", "waited", "waited_full", "preemptions", "resumptions", "released", "release_buckets", "ended_stop",
          "ended_length", "lagged", "length_terms", "segments")

#: each class's events: the class is exercised in a window where any of them is above 0
OF = {"hits": ("lookups_hit", "admissions_cut"), "admission": ("waited", "preemptions"), "aborts": (),
      "release buckets": ("release_buckets",), "terminator": ("ended_stop", "ended_length"),
      "output lengths": ("length_terms",), "resumes and lag": ("resumptions", "lagged"), "segment end": ("segments",)}

#: a batch is full where one of these holds a step's waiting requests (`by_class.WHY`)
FULL = ("max_num_seqs", "tokens")


def registration(trace: Any, stop_ids: Sequence[int], T: int) -> Registered:
    """The trace's registration: under the certified tick, its tick and the code's own book (`codec.TICK_CAPS`'s, the
    table's); otherwise the uniform book of the table's caps."""
    if trace.releases is not None:
        return Registered(trace.engine, tuple(stop_ids), T=T, tick=trace.releases.tick)
    return Registered(trace.engine, tuple(stop_ids), T=T, book=CV.Book.uniform(V.TABLE_CAPS))


def replayed(advice: Any, record: Record, reg: Registered) -> tuple[Any, list[int], list[list[Any]], list[BC.Seen], list[int], list[int]]:
    """The replay v1 encodes the advice under (`codec.replay_record`, each request arrived from its first row's step
    on), the arrival order, each step's lookups and what its scheduler and pool did (`by_class.Seen`), and A5's window
    of each step and of each arrival (`codec.charged`)."""
    order = arrival_order(advice, record, reg.row_order)
    index = {record.requests[k].id: e for e, k in enumerate(order)}
    first = [-1] * len(order)
    for s, st in enumerate(advice.steps):
        for x in st.rows:
            e = index[advice.requests[x.request].id]
            if first[e] < 0:
                first[e] = s
    with BC.observing() as seen:
        rp, _, looks = replay_record(record, reg, reg.engine, order, lambda i, k: k >= first[i])
    if len(seen) != len(rp.steps):
        raise ValueError(f"{len(seen)} observed steps for the replay's {len(rp.steps)}")
    return rp, order, looks, seen, *charged(record, reg, order, rp, windows(record, reg.T))


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
    for name, rules in BC.no_field(reg).items():
        out[name] = {"standing": "no field", "rules": cite(rules)}
    return out


def aborts(d: Path) -> dict[str, Any]:
    """What a trace's directory records of aborts, read without the reader that refuses them: the requests the engine
    freed by finish reason and those it ended on neither a stop id nor its length; where the directory has the
    client's rows (one a released request), the released requests the engine never freed (dropped before it, or
    unfinished); and where it has the network certifier's log, the connections it failed, by reason, apart for those
    whose request it had released: failing one closes its connection to the server, which is how a request the engine
    holds is aborted."""
    d = Path(d)
    reqs = R._jsonl(d / "requests.jsonl")
    finishes: dict[str, int] = {}
    for r in reqs:
        finishes[str(r["finish_reason"])] = finishes.get(str(r["finish_reason"]), 0) + 1
    out: dict[str, Any] = {"freed": len(reqs), "finishes": dict(sorted(finishes.items())),
                           "aborted": sum(n for k, n in finishes.items() if k not in ("stop", "length")),
                           "released_never_freed": None, "certifier_failed": None}
    if (d / "client.jsonl").exists():
        freed = {r["id"] for r in reqs}
        out["released_never_freed"] = sum(1 for r in R._jsonl(d / "client.jsonl") if r["id"] not in freed)
    if (d / "warden" / "warden.json").exists():
        failed: dict[str, dict[str, int]] = {"released": {}, "unreleased": {}}
        for c in json.loads((d / "warden" / "warden.json").read_text())["connections"]:
            if c.get("failed"):
                by = failed["unreleased" if c.get("release") is None else "released"]
                by[c["failed"]] = by.get(c["failed"], 0) + 1
        out["certifier_failed"] = failed
    return out


def report(label: str, path: str, stop_ids: Sequence[int], T: int = 600, ingress: str = "client") -> dict[str, Any]:
    """One trace-corpus trace's report (the module's docstring); for a trace the reader or the code refuses, the
    refusal (`error`) and the directory's aborts."""
    seen = aborts(Path(path))
    try:
        tr, rec, adv = V._trace(path, stop_ids, ingress)
        row = split(label, tr, rec, adv, registration(tr, stop_ids, T))
    except (ValueError, RuntimeError) as e:
        row = {"trace": label, "error": str(e)}
    return {**row, "aborts": seen}


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
    rp, order, looks, seen, at, arrived = replayed(advice, record, reg)
    B = reg.engine.block_size
    rows = [{"window": w, "charge": charge[w], "fields": parts[w],
             "classes": {c: sum(parts[w][f] for f in fs) for c, fs in BC.CLASSES.items()},
             "events": dict.fromkeys(EVENTS, 0), "steps_waiting": dict.fromkeys(BC.WHY, 0)} for w in sorted(charge)]
    total = {"events": dict.fromkeys(EVENTS, 0), "steps_waiting": dict.fromkeys(BC.WHY, 0)}
    waited: list[tuple[set[int], set[int]]] = [(set(), set()) for _ in (*rows, None)]
    buckets: list[set[int]] = [set() for _ in (*rows, None)]
    index = {record.requests[k].id: e for e, k in enumerate(order)}
    for r, ks in zip(rows, lengths(record, reg)):
        if sum(r["fields"].values()) != r["charge"]:
            raise ValueError(f"window {r['window']}: its parts sum to {sum(r['fields'].values())} bits of {r['charge']}")
        ev = r["events"]
        ev["length_terms"] = len(ks)
        ev["segments"] = int(any(r["fields"][f] for f in BC.SYMBOLS.values()))
        for k in ks:
            stop = rp.finish[index[record.requests[k].id]] == "stop"
            ev["ended_stop" if stop else "ended_length"] += 1
            ev["lagged"] += stop and reg.engine.async_scheduling
    for e, k in enumerate(order):
        rows[arrived[e]]["events"]["released"] += 1
        for into in (buckets[arrived[e]], buckets[-1]):
            into.add(record.requests[k].end)
    admitted: set[int] = set()
    for s, (st, ls, sn) in enumerate(zip(rp.steps, looks, seen)):
        r = rows[at[s]]
        ev = r["events"]
        ev["lookups"] += len(ls)
        ev["lookups_hit"] += sum(1 for x in ls if x.hit)
        for i, start, _ in st.rows:
            if i not in admitted:
                admitted.add(i)
                ev["admissions"] += 1
                ev["admissions_hit"] += start > 0
                ev["hit_blocks"] += start // B
        new = set(st.new)
        ev["evicted_blocks"] += sn.evicted
        ev["admissions_cut"] += sum(1 for i, _ in sn.cut if i in new)
        ev["cut_blocks"] += sum(k for i, k in sn.cut if i in new)
        ev["preemptions"] += len(sn.preempted)
        ev["resumptions"] += len(sn.resumed)
        if sn.why is not None:
            for by in (r, total):
                by["steps_waiting"][sn.why] += 1
            for any_, full in (waited[at[s]], waited[-1]):
                any_.update(sn.waiting)
                if sn.why in FULL:
                    full.update(sn.waiting)
    for r, (any_, full), bs in zip(rows, waited, buckets):
        r["events"].update(waited=len(any_), waited_full=len(full), release_buckets=len(bs))
    total["events"] = {e: sum(r["events"][e] for r in rows) for e in EVENTS}
    total["events"].update(waited=len(waited[-1][0]), waited_full=len(waited[-1][1]), release_buckets=len(buckets[-1]))
    theirs = [[(index[advice.requests[x.request].id], x.start, x.n) for x in st.rows] for st in advice.steps]
    m = R.compare(rp.steps, [R.Step(rows=r) for r in theirs])
    differs = None
    if m.first_mismatch is not None:
        differs = {"step": m.first_mismatch, "window": at[m.first_mismatch] if m.first_mismatch < len(at) else len(rows) - 1,
                   "matched": m.matched, "steps": m.steps}
    hits = [k for st in trace.steps for k in st.cache_hit.values() if k]
    e = reg.engine
    return {"trace": label, "T": reg.T, "tick": None if reg.tick is None else asdict(reg.tick),
            "prefix_caching": e.enable_prefix_caching, "requests": len(record.requests), "steps": len(rp.steps),
            "engine": {"num_gpu_blocks": e.num_gpu_blocks, "block_size": B, "max_num_seqs": e.max_num_seqs,
                       "max_num_batched_tokens": e.max_num_batched_tokens, "async_scheduling": e.async_scheduling},
            "windows": len(rows), "bits": sum(charge.values()), "bytes": len(data), "book": V.book_of(reg).digest(),
            "round_trip": back, "refused": refused, "first_step_that_differs": differs,
            "classes": classes(reg), "by_window": rows,
            "total": {"charge": sum(charge.values()),
                      "classes": {c: sum(r["classes"][c] for r in rows) for c in BC.CLASSES},
                      "fields": {f: sum(r["fields"][f] for r in rows) for f in BC.FIELDS}, **total},
            "engine_hits": {"admissions_hit": len(hits), "hit_blocks": sum(k // B for k in hits), "hit_tokens": sum(hits)}}


def _share(part: int, whole: int) -> str:
    return f"{part:,} of {whole:,}" + (f" ({100 * part / whole:.1f}%)" if whole else "")


#: the table's columns: each class, then its events' counts (`EVENTS`' names and their headings)
COLUMNS = (("hits", (("lookups_hit", "lookups that hit"), ("admissions_cut", "admissions an eviction cut"))),
           ("admission", (("waited", "requests left waiting"), ("waited_full", "of them on a full batch"),
                          ("preemptions", "preemptions"))),
           ("aborts", ()),
           ("release buckets", (("released", "requests released"), ("release_buckets", "their buckets"))),
           ("terminator", (("ended_stop", "ended `stop`"), ("ended_length", "ended `length`"))),
           ("output lengths", (("length_terms", "length terms"),)),
           ("resumes and lag", (("resumptions", "resumptions"), ("lagged", "ended `stop` under async scheduling"))),
           ("segment end", ()))


def _cell(name: str, r: dict[str, Any]) -> str:
    """A class's cell in a window's row: its bits, `no field`, or `not exercised` where it has no bits and none of its
    events happened."""
    bits = r["classes"].get(name)
    if not bits and not any(r["events"][e] for e in OF[name]):
        return "not exercised"
    return "no field" if bits is None else f"{bits:,}"


def _aborts(a: dict[str, Any]) -> str:
    """A trace's aborts (`aborts`), in a sentence."""
    ends = ", ".join(f"{n:,} `{k}`" for k, n in a["finishes"].items()) or "none"
    out = f"the engine freed {a['freed']:,} requests ({ends}), {a['aborted']:,} of them aborted"
    if a["released_never_freed"] is not None:
        out += f"; {a['released_never_freed']:,} released requests it never freed"
    if a["certifier_failed"] is not None:
        def by(kind: str) -> str:
            return ", ".join(f"{n:,} `{k}`" for k, n in sorted(a["certifier_failed"][kind].items())) or "none"
        out += (f"; the network certifier failed connections after their request's release: {by('released')}; and before "
                f"it: {by('unreleased')}")
    return out


def render(row: dict[str, Any], standing: bool = True) -> str:
    """One trace's report as Markdown: the table by window, each class's bits with its events' counts beside them, the
    pool's and the scheduler's other counts, the round trip, the aborts and, with ``standing``, each class's standing
    with its rules. A trace that was refused is its refusal and its aborts."""
    if "error" in row:
        out = f"**{row['trace']}**: refused, so it has no table: {row['error']}\n"
        if row.get("aborts") is not None:
            out += f"\nAborts in its record: {_aborts(row['aborts'])}.\n"
        return out
    tick = "under the certified tick" if row["tick"] else "admitted on arrival"
    cache = "on" if row["prefix_caching"] else "off"
    t, g = row["total"], row["engine"]
    columns = ["window", "charge"] + [h for name, events in COLUMNS for h in (name, *(head for _, head in events))]
    out = [f"**{row['trace']}**: {row['requests']:,} requests, {row['steps']:,} steps, {tick}, the prefix cache {cache}; "
           f"a pool of {g['num_gpu_blocks']:,} blocks of {g['block_size']} tokens, at most {g['max_num_seqs']} requests "
           f"and {g['max_num_batched_tokens']:,} tokens a step; {row['windows']} window{'s' if row['windows'] != 1 else ''} "
           f"of T = {row['T']} buckets; {row['bits']:,} bits.", "",
           "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]

    def line(head: str, r: dict[str, Any]) -> str:
        cells = [c for name, events in COLUMNS for c in (_cell(name, r), *(f"{r['events'][e]:,}" for e, _ in events))]
        return f"| {head} | {r['charge']:,} | " + " | ".join(cells) + " |"

    out += [line(str(r["window"]), r) for r in row["by_window"]]
    out += [line("all", t), ""]
    f, ev, e = t["fields"], t["events"], row["engine_hits"]
    out.append("A class reads \"not exercised\" in a window where it has no bits and none of its events happened; "
               "\"all\" counts the requests left waiting and the release buckets over the trace.")
    out.append(f"The admission is {f['decisions']:,} bits of decisions and {f['order']:,} of order entries.")
    off = "" if row["prefix_caching"] else " (the prefix cache is off: a lookup returns nothing)"
    out.append(f"Lookups that hit: {_share(ev['lookups_hit'], ev['lookups'])}{off}. Admissions from a hit: "
               f"{_share(ev['admissions_hit'], ev['admissions'])}, {ev['hit_blocks']:,} blocks; the engine's own record: "
               f"{e['admissions_hit']:,} admissions, {e['hit_blocks']:,} blocks.")
    out.append(f"The pool evicted {ev['evicted_blocks']:,} cached blocks; {ev['admissions_cut']:,} admissions' lookups "
               f"ended at an evicted block, {ev['cut_blocks']:,} blocks they would have hit.")
    held = ", ".join(f"{n:,} by `{why}`" for why, n in t["steps_waiting"].items() if n) or "none"
    out.append(f"Steps that left a request waiting: {held}.")
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
    if row.get("aborts") is not None:
        out.append(f"Aborts in its record: {_aborts(row['aborts'])}.")
    if standing:
        out.append("")
        for name, c in row["classes"].items():
            why = "; ".join(f"{r['why']} (`{r['file']}:{r['line']}`)" for r in c["rules"])
            out.append(f"- {name}: {c['standing']}" + (f": {why}." if why else "."))
    return "\n".join(out) + "\n"


def render_all(rows: Sequence[dict[str, Any]]) -> str:
    """Every trace's report; the classes' standing once for each run of traces that share it, after the last of them."""
    out = []
    for k, r in enumerate(rows):
        last = k + 1 == len(rows) or rows[k + 1].get("classes") != r.get("classes")
        out.append(render(r, standing=last))
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Advice v1's bits by class and window over trace-corpus traces, with each "
                                             "class's events, the aborts and the round trip.")
    ap.add_argument("traces", nargs="+", help="LABEL=DIR: a trace's directory (steps.jsonl, requests.jsonl, meta.json, "
                                              "server.log)")
    ap.add_argument("--stops", default="128001", help="the registered stop ids, comma-separated, EOS first")
    ap.add_argument("--T", type=int, default=600, help="the window, in the trace's certified buckets: its tick's τ_b, "
                                                          "100 ms without the tick")
    ap.add_argument("--ingress", choices=V.INGRESS, default="client", help="the certified order's stand-in, as v1.py's")
    ap.add_argument("--out", type=Path, required=True, help="the report's JSON")
    a = ap.parse_args(argv)
    stops = tuple(int(s) for s in a.stops.split(","))
    rows = [report(*t.split("=", 1), stops, a.T, a.ingress) for t in a.traces]
    a.out.write_text(json.dumps({"T": a.T, "stops": stops, "ingress": a.ingress, "rows": rows}, indent=1) + "\n")
    print(render_all(rows), end="")
    return 0 if all("error" not in r and r["round_trip"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
