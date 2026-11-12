"""Advice v1's bits by class over trace-corpus traces: each window's charge split by class, each class's events
beside its bits, and whether the string decodes back to the trace.

The split is `verity_vllm.pipeline.advice_code.by_class` (`bits_by_class`: a window's parts sum to
`codec.bits_by_window`'s charge for it; `closed`: the fields the registration holds at 0, each with its rule and the
rule's line). This module reads a trace as `v1.py` does, encodes it under the uniform book of the table's bounds
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
has a table holds none and its count of aborted requests is 0 in every window. What the trace's directory records of
them is read without the reader and reported either way: the requests the engine freed by finish reason, the requests
released that the engine never freed, the connections the network certifier failed, by reason and by whether their
request had been released, and, for a trace served under the certified tick, the aborted requests by the window their
release falls in. A trace the reader refuses is one row: the refusal and those counts, and that the run shows the
advice of none of that record's windows.

**The lengths** (`length_terms`). While a registration sends them, the lengths are the charge a tick-served trace has, and
they are the load's and not the serve's: a request's output length is sent in ⌈log2(`max_tokens` + 1)⌉ bits and how it
ended in 1. So the report prints that rule with its line, the load's `max_tokens` (its range and mean), the mean width
and the bits a request, and what the charge comes to over a week, a 30-day month and a year at this serve's own rate
of releases, as a scaled figure.

**What holds the charge.** The table is the bits charged by the registered code: `decode`, the registered codebook
and ℓ (`codec.Registered`) are Python's. ℓ bounds the whole string, every window's charge together, and is printed
beside each.

**The serve** (`serve_of`), read from the directory without the reader: the served model's name, its cards, the vLLM
version the server's log prints, where the network certifier ran, and whether the server had a draft. The reader
takes no draft from the server's arguments, so its replay of such a serve would be of another one: the trace is
refused, and a trace that has a table was served with the draft off, which its first line says. With
``--beside LABEL=DIR`` a second serve's trace is read the same way and set beside each trace that has a table
(`beside`): each class the trace did not exercise, with the other serve's counts of its events; each event of the
classes it did exercise that happened in the other serve only; then what the two serves share and where they
differ. So a class "not exercised" in one serve is read beside the serve that exercised it.

Per trace it also reports the engine's own record of the hits it admitted from (the trace's `cache_hit`) and the round
trip (`decode(encode(a)) == a`, the streams too). Where the decoder refuses the string or decodes it to another advice,
the report says which, and gives the first step at which the replay the decoder runs differs from the engine's
schedule, with its window.

    python by_class.py LABEL=DIR ... --stops IDS [--T 600 --ingress client] [--beside LABEL=DIR ...] --out REPORT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verity_vllm.pipeline.advice_code import by_class as BC
from verity_vllm.pipeline.advice_code.codec import (
    AdviceCodeError, arrival_order, bits_by_window, charged, decode_full, encode, length_widths, lengths, Record, Registered,
    replay_record, streams, windows,
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
          "ended_length", "lagged", "length_terms", "segments", "aborted")

#: each class's events: the class is exercised in a window where any of them is above 0
OF = {"hits": ("lookups_hit", "admissions_cut"), "admission": ("waited", "preemptions"), "aborts": ("aborted",),
      "release buckets": ("release_buckets",), "terminator": ("ended_stop", "ended_length"),
      "output lengths": ("length_terms",), "resumes and lag": ("resumptions", "lagged"), "segment end": ("segments",)}

#: a batch is full where one of these holds a step's waiting requests (`by_class.WHY`)
FULL = ("max_num_seqs", "tokens")


def registration(trace: Any, stop_ids: Sequence[int], T: int) -> Registered:
    """The trace's registration: under the certified tick, its tick and the code's own book (`codec.TICK_CAPS`'s, the
    table's); otherwise the uniform book of the table's bounds."""
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


def aborts(d: Path, T: int = 600) -> dict[str, Any]:
    """What a trace's directory records of aborts, read without the reader that refuses them: the requests the engine
    freed by finish reason and those it ended on neither a stop id nor its length; where every request has its
    certified release bucket (a trace served under the certified tick), the windows of `T` buckets that hold a
    release and the aborted requests in each that has one, a request's window counted from the first release as the
    reader counts it; where the directory has the client's rows (one a released request), the released requests the
    engine never freed (dropped before it, or unfinished); and where it has the network certifier's log, the
    connections it failed, by reason, apart for those whose request it had released: failing one closes its
    connection to the server, which is how a request the engine holds is aborted."""
    d = Path(d)
    reqs = R._jsonl(d / "requests.jsonl")
    finishes: dict[str, int] = {}
    for r in reqs:
        finishes[str(r["finish_reason"])] = finishes.get(str(r["finish_reason"]), 0) + 1
    out: dict[str, Any] = {"freed": len(reqs), "finishes": dict(sorted(finishes.items())),
                           "aborted": sum(n for k, n in finishes.items() if k not in ("stop", "length")),
                           "windows_released": None, "by_window": None,
                           "released_never_freed": None, "certifier_failed": None}
    if reqs and all(type(r.get("bucket")) is int for r in reqs):
        base = min(r["bucket"] for r in reqs)
        by: dict[int, int] = {}
        for r in reqs:
            w = (r["bucket"] - base) // T
            by[w] = by.get(w, 0) + (r["finish_reason"] not in ("stop", "length"))
        out["windows_released"] = len(by)
        out["by_window"] = [{"window": w, "aborted": n} for w, n in sorted(by.items()) if n]
    if (d / "client.jsonl").exists():
        freed = {r["id"] for r in reqs}
        out["released_never_freed"] = sum(1 for r in R._jsonl(d / "client.jsonl") if r["id"] not in freed)
    log = d / "warden" / "warden.json"
    if log.exists():
        failed: dict[str, dict[str, int]] = {"released": {}, "unreleased": {}}
        for c in json.loads(log.read_text())["connections"]:
            if c.get("failed"):
                by = failed["unreleased" if c.get("release") is None else "released"]
                by[c["failed"]] = by.get(c["failed"], 0) + 1
        out["certifier_failed"] = failed
    return out


def serve_of(d: Path) -> dict[str, Any]:
    """What a trace's directory says of its serve, read without the reader: the served model's name, the cards (the
    tensor-parallel size, 1 where the server names none), whether the server had a draft (an argument that begins
    ``--speculative``), the vLLM version its log prints, and where the network certifier ran (the note meta.json's
    tick has of it); None for what the directory doesn't say."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    argv = [str(a) for a in meta["server_argv"]]

    def opt(name: str) -> str | None:
        return argv[argv.index(name) + 1] if name in argv[:-1] else None

    log = d / "server.log"
    version = re.search(r"\bversion (\d+\.\d+[\w.+-]*)", log.read_text(errors="replace")) if log.exists() else None
    return {"model": opt("--served-model-name"), "cards": int(opt("--tensor-parallel-size") or 1),
            "draft": any(a.startswith("--speculative") for a in argv), "vllm": version.group(1) if version else None,
            "certifier": (meta.get("tick") or {}).get("where")}


def report(label: str, path: str, stop_ids: Sequence[int], T: int = 600, ingress: str = "client") -> dict[str, Any]:
    """One trace-corpus trace's report (the module's docstring); for a trace the reader or the code refuses, or one
    served with a draft, the refusal (`error`); either way what the directory says of the serve (`serve`) and of its
    aborts."""
    seen, serve = aborts(Path(path), T), serve_of(Path(path))
    try:
        if serve["draft"]:
            raise ValueError("the server had a draft, which the reader doesn't take from its arguments: the replay "
                             "would be of a serve without one")
        tr, rec, adv = V._trace(path, stop_ids, ingress)
        row = split(label, tr, rec, adv, registration(tr, stop_ids, T))
    except (ValueError, RuntimeError) as e:
        row = {"trace": label, "error": str(e), "T": T}
    return {**row, "serve": serve, "aborts": seen}


#: the monitoring periods a charge is scaled to, in seconds
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}


def length_terms(record: Record, reg: Registered, bucket_s: float) -> dict[str, Any]:
    """The lengths' charge as its rule (`by_class.RULES`' "length width", with its line): over the requests whose
    lengths the registration sends, their `max_tokens` (least, greatest and mean), the widths their output lengths
    are sent in, and the terms' bits, each a width and the 1 bit of how the request ended. And the seconds from the
    record's first release to its last, the rate a charge is scaled by (`PERIODS`)."""
    ws = [n for w in length_widths(record, reg) for n in w[0::2]]
    ks = [k for w in lengths(record, reg) for k in w]
    mt = [record.requests[k].max_tokens for k in ks]
    file, line = BC.where("length width")
    ends = [q.end for q in record.requests]
    span = (max(ends) - min(ends) + 1) * bucket_s
    return {"rule": {"rule": "length width", "why": BC.RULES["length width"].why, "file": f"advice_code/{file}", "line": line},
            "terms": len(ws), "bits": sum(ws) + len(ws),
            "max_tokens": {"min": min(mt), "max": max(mt), "mean": sum(mt) / len(mt)} if mt else None,
            "width": {"min": min(ws), "max": max(ws), "mean": sum(ws) / len(ws)} if ws else None,
            "release_span_s": span}


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
    terms = length_terms(record, reg, V._bucket_s(trace))
    bits = sum(charge.values())
    return {"trace": label, "T": reg.T, "tick": None if reg.tick is None else asdict(reg.tick), "ell": reg.cap_bits,
            "lengths": terms, "scaled": {name: round(bits * s / terms["release_span_s"]) for name, s in PERIODS.items()},
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
           ("aborts", (("aborted", "aborted requests"),)),
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


def _unshown(a: dict[str, Any], T: int) -> str:
    """What a refused trace's aborts leave unshown: its aborted requests by window, and that the run shows no window's
    advice."""
    if a["by_window"] is None:
        return (f"Aborted requests: {a['aborted']:,}; the record has no certified release buckets, so they are not "
                f"counted by window. The record is refused whole: this run does not show its advice.")
    n = a["windows_released"]
    which = ", ".join(f"window {w['window']}: {w['aborted']:,}" for w in a["by_window"]) or "none in any window"
    why = ("The code has no field for an abort and the reader refuses the whole record for one" if a["aborted"] else
           "The record is refused whole")
    return (f"Aborted requests: {a['aborted']:,}, by the window of T = {T} buckets their release falls in: {which}. {why}: "
            f"this run does not show the advice of {'its one window' if n == 1 else f'any of its {n} windows'} that "
            f"hold{'s' if n == 1 else ''} a release.")


#: each event's heading and its class: the table's (`COLUMNS`), and the segment end's, which has no column
HEADS = {"segments": "windows with a segment", **{e: head for _, events in COLUMNS for e, head in events}}
CLASS_OF = {"segments": "segment end", **{e: name for name, events in COLUMNS for e, _ in events}}


def _facts(row: dict[str, Any]) -> dict[str, str]:
    """What a report says of its serve, each fact as the text two serves are compared in: the directory's own
    (`serve_of`) and, for a trace with a table, the registration's."""
    s = row["serve"]
    out = {"model": s["model"] or "not named", "cards": str(s["cards"]), "vLLM": s["vllm"] or "not in the server's log",
           "the draft": "on" if s["draft"] else "off", "the network certifier": s["certifier"] or "not said"}
    if "error" not in row:
        g, t = row["engine"], row["tick"]
        tick = "on arrival" if t is None else (f"under the certified tick (τ_b = {t['bucket_us']:,}, τ_sat = {t['sat_us']:,}, "
                                               f"τ_free = {t['free_us']:,} µs)")
        out |= {"the pool": f"{g['num_gpu_blocks']:,} blocks of {g['block_size']} tokens",
                "requests a step": f"at most {g['max_num_seqs']:,}", "tokens a step": f"at most {g['max_num_batched_tokens']:,}",
                "admission": tick, "the prefix cache": "on" if row["prefix_caching"] else "off",
                "the registered codebook": f"`{row['book'][:16]}`", "ℓ": f"{row['ell']:,} bits",
                "T": f"{row['T']} buckets"}
    return out


def beside(row: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """A trace's report (one with a table) set beside another serve's (`other`, a `report`; a refused one has only its
    aborts and its directory's facts): each class not exercised over the trace, with the other's counts of its events
    (None where the other has no table to count them in); each event of a class the trace did exercise that didn't
    happen in it and did in the other, with the other's count; and the serves' facts (`_facts`) that are the same and
    those that differ."""
    table = "error" not in other

    def there(e: str) -> int | None:
        if e == "aborted":
            return other["aborts"]["aborted"]
        return other["total"]["events"][e] if table else None

    here = row["total"]["events"]
    quiet = {name: {e: there(e) for e in OF[name]} for name in OF if _cell(name, row["total"]) == "not exercised"}
    events = {e: there(e) for e in HEADS if CLASS_OF[e] not in quiet and not here[e] and there(e)}
    mine, theirs = _facts(row), _facts(other)
    both = [k for k in mine if k in theirs]
    return {"trace": other["trace"], "refused": other.get("error"), "not_exercised": quiet, "events": events,
            "same": {k: mine[k] for k in both if mine[k] == theirs[k]},
            "differs": {k: [mine[k], theirs[k]] for k in both if mine[k] != theirs[k]}}


def _beside(b: dict[str, Any]) -> list[str]:
    """`beside`, as Markdown lines."""
    label = f"`{b['trace']}`"
    out = [f"Beside {label}" + (", which is refused, so that only its aborts and its directory's facts are read"
                                if b["refused"] is not None else "") + ":", ""]
    for name, events in b["not_exercised"].items():
        seen = ", ".join(f"{HEADS[e]} {n:,}" for e, n in events.items() if n)
        unread = any(n is None for n in events.values())
        out.append(f"- {name}: not exercised here; " + (f"in {label}: {seen}." if seen else
                                                       f"{label} has no table to say." if unread else
                                                       f"not exercised in {label} either."))
    if b["events"]:
        out.append(f"- in the classes exercised here, none here and some in {label}: "
                   + ", ".join(f"{HEADS[e]} {n:,}" for e, n in b["events"].items()) + ".")
    if b["same"]:
        out.append("- the same in both serves: " + "; ".join(f"{k} {v}" for k, v in b["same"].items()) + ".")
    if b["differs"]:
        out.append("- differing: " + "; ".join(f"{k}: {x} here, {y} in {label}" for k, (x, y) in b["differs"].items()) + ".")
    return out


def render(row: dict[str, Any], standing: bool = True) -> str:
    """One trace's report as Markdown: the table by window, each class's bits with its events' counts beside them and ℓ
    beside each charge, what holds the charge, the lengths' rule and the charge scaled to the monitoring periods, the
    pool's and the scheduler's other counts, the round trip, the aborts, each serve it is set beside and, with
    ``standing``, each class's standing with its rules. A trace that was refused is its refusal, its aborts and what
    of the run they leave unshown."""
    if "error" in row:
        out = f"**{row['trace']}**: refused, so it has no table: {row['error']}\n"
        if row.get("aborts") is not None:
            out += f"\nAborts in its record: {_aborts(row['aborts'])}.\n\n{_unshown(row['aborts'], row['T'])}\n"
        return out
    tick = "under the certified tick" if row["tick"] else "admitted on arrival"
    cache = "on" if row["prefix_caching"] else "off"
    t, g = row["total"], row["engine"]
    columns = ["window", "charge", "ℓ"] + [h for name, events in COLUMNS for h in (name, *(head for _, head in events))]
    out = [f"**{row['trace']}**: {row['requests']:,} requests, {row['steps']:,} steps, {tick}, served with the draft off, "
           f"the prefix cache {cache}; "
           f"a pool of {g['num_gpu_blocks']:,} blocks of {g['block_size']} tokens, at most {g['max_num_seqs']} requests "
           f"and {g['max_num_batched_tokens']:,} tokens a step; {row['windows']} window{'s' if row['windows'] != 1 else ''} "
           f"of T = {row['T']} buckets; {row['bits']:,} bits charged by the registered code.", "",
           "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]

    def line(head: str, r: dict[str, Any]) -> str:
        cells = [c for name, events in COLUMNS for c in (_cell(name, r), *(f"{r['events'][e]:,}" for e, _ in events))]
        return f"| {head} | {r['charge']:,} | {row['ell']:,} | " + " | ".join(cells) + " |"

    out += [line(str(r["window"]), r) for r in row["by_window"]]
    out += [line("all", t), ""]
    f, ev, e = t["fields"], t["events"], row["engine_hits"]
    out.append("A class reads \"not exercised\" in a window where it has no bits and none of its events happened; "
               "\"all\" counts the requests left waiting and the release buckets over the trace.")
    out.append(f"The charge is the bits charged by the registered code: `decode`, the registered codebook and ℓ are held "
               f"by Python. ℓ = {row['ell']:,} bits bounds the whole string, every window's charge together.")
    n = row["lengths"]
    if n["terms"]:
        r, mt, w = n["rule"], n["max_tokens"], n["width"]
        out.append(f"The lengths: {r['why']} (`{r['file']}:{r['line']}`). The load's `max_tokens` runs from {mt['min']:,} "
                   f"to {mt['max']:,}, mean {mt['mean']:,.1f}; the widths from {w['min']} to {w['max']} bits, mean "
                   f"{w['mean']:.3f}; {n['terms']:,} terms are {n['bits']:,} bits, {n['bits'] / n['terms']:.3f} a request.")
    else:
        out.append("The lengths: the registration sends none.")
    k = row["scaled"]
    out.append(f"Scaled, not measured: at this serve's own rate, its {row['bits']:,} bits over the "
               f"{n['release_span_s']:,.1f} s from its first release to its last, the charge is {k['week']:,} bits a week, "
               f"{k['month']:,} a month of 30 days and {k['year']:,} a year.")
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
    for b in row.get("beside", ()):
        out += ["", *_beside(b)]
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
    ap.add_argument("--beside", action="append", default=[], metavar="LABEL=DIR",
                    help="another serve's trace, read the same way and set beside each trace that has a table: what "
                         "that serve exercised and this one did not, and where the two serves differ")
    ap.add_argument("--out", type=Path, required=True, help="the report's JSON")
    a = ap.parse_args(argv)
    stops = tuple(int(s) for s in a.stops.split(","))
    rows = [report(*t.split("=", 1), stops, a.T, a.ingress) for t in a.traces]
    others = [report(*t.split("=", 1), stops, a.T, a.ingress) for t in a.beside]
    for r in rows:
        if "error" not in r and others:
            r["beside"] = [beside(r, o) for o in others]
    doc = {"T": a.T, "stops": stops, "ingress": a.ingress, "rows": rows, "beside": others}
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(render_all(rows), end="")
    return 0 if all("error" not in r and r["round_trip"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
