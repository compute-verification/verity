"""Advice v1 measured over the trace corpus: each trace's v1 bits per certifier window under a book fit on the other
traces.

The codec is `verity_vllm.pipeline.advice_code.codec` (`encode`, `decode`, `bits_by_window`; its docstring states what
is certified, replayed, derived and sent, the windows and the string), re-exported here for the modules and tests that
load this one by path. This module adds a trace's certified record (`record_of`), its I1 v0 advice (`advice_of`) and,
for a trace served with a draft, which no I1 v0 says, its schedule (`schedule_of`, `schedule_code`'s),
a trace's residual windows (`windows_of`) and the books held out from it (`held_out_books`), and the table
(`measure`, `render`):

    python v1.py LABEL=DIR ... --stops IDS [--T 600 --ingress client --jobs N] --out TABLE.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypeVar

from verity.protocols.certifier.verifier import advice as I1
from verity_vllm.pipeline.advice_code.codec import (
    advice_of_replay as advice_of_replay, AdviceCodeError, arrival_order as arrival_order, bits_by_window, book_of, CAPS as CAPS,
    charged as charged, check_record as check_record, Decision as Decision, decode as decode,
    decode_bits as decode_bits, decode_full, encode, kept_terms as kept_terms, length_tails,
    length_widths as length_widths, Record, Registered, release_order as release_order,
    replay_record as replay_record, Request, residual, streams, string_of, TICK_CAPS as TICK_CAPS, windows,
)
from verity_vllm.pipeline.advice_code.schedule_code import Schedule

HERE = Path(__file__).resolve().parent
T = TypeVar("T")


def _load(name: str) -> Any:
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


R, C, D, CV = _load("replay"), _load("code"), _load("derived"), _load("code_v1")


# --- from a trace ------------------------------------------------------------------------------------------------------

def record_of(trace: Any, stop_ids: Sequence[int], *, bucket_s: float = 0.1) -> Record:
    """A trace's certified record (`replay.Trace`, clocked, with each request's tokens): each request's certified
    ingress bucket standing for its release a, its ingress time then its engine index for its ordinal, so that the
    record is in release order, and the trace's release buckets. A run under the certified tick (`Trace.releases`) has
    its releases certified: each request's `end` is its release bucket and its ordinal its release number, and its
    outputs' buckets are on the certifier run's grid, none before its release; all count from the run's first release
    bucket."""
    if not trace.clocked():
        raise AdviceCodeError("the trace has no clock, so no certified buckets")
    rel = trace.releases
    if rel is None:
        cert = trace.certified(bucket_s)
        order = sorted(range(len(trace.requests)), key=lambda k: (cert.ingress[k], trace.ingress_t[k], k))
        end, release = cert.ingress, cert.release
    else:
        if rel.tick.bucket_us != round(bucket_s * 1e6):
            raise AdviceCodeError(f"the certified tick's bucket is {rel.tick.bucket_us} µs and the record's "
                                  f"{bucket_s} s")
        if len(set(rel.n)) != len(rel.n):
            raise AdviceCodeError("two requests share a release number")
        base = min(rel.bucket)
        order = sorted(range(len(trace.requests)), key=lambda k: rel.n[k])
        end = [a - base for a in rel.bucket]
        release = [[max(a, int((t - rel.origin_unix) * 1e6 // rel.tick.bucket_us) - base) for t in ts]
                   for a, ts in zip(end, trace.output_times())]
    out = []
    for k in order:
        io = trace.requests[k]
        if io.tokens is None:
            raise AdviceCodeError(f"request {trace.ids[k]}: the trace has no tokens")
        output = tuple(io.tokens[io.prompt_len:io.prompt_len + io.output_len])
        if (output[-1] in stop_ids) != io.eos:
            raise AdviceCodeError(f"request {trace.ids[k]}: stopped {'on' if io.eos else 'without'} a stop id, its "
                                  f"last id {output[-1]}")
        out.append(Request(trace.ids[k], tuple(io.tokens[:io.prompt_len]), io.max_tokens, output, end[k],
                           tuple(release[k])))
    return Record(tuple(out))


def advice_of(trace: Any, record: Record) -> I1.Advice:
    """The trace as I1 v0: its rows in the order its runner ran them, I1's numbering by first row, each admitting row's
    hit its start and its sources the canonical ones, and each kept output released in the record's bucket. The trace's
    requests are the record's, by id."""
    steps_rows = [list(st.rows) for st in trace.steps]
    num: dict[int, int] = {}
    order: list[int] = []
    for rows in steps_rows:
        for i, _, _ in rows:
            if i not in num:
                num[i] = len(order)
                order.append(i)
    by_id = {q.id: q for q in record.requests}
    reqs = [by_id[trace.ids[i]] for i in order]
    requests = tuple(I1.RequestAdvice(q.id, len(q.prompt), len(q.output), "eos" if trace.requests[i].eos else "length")
                     for i, q in zip(order, reqs))
    prefixes = D.Prefixes([q.prompt + q.output for q in reqs], trace.engine.block_size)
    seen: set[int] = set()
    kept = [0] * len(order)
    steps, release = [], []
    for st, rows in zip(trace.steps, steps_rows):
        for i, start, n in rows:
            prefixes.add(num[i], start, start + n)
        out = []
        for i, start, n in rows:
            j, q = num[i], requests[num[i]]
            back = i in st.resumed
            hit = start if j not in seen or back else 0
            seen.add(j)
            out.append(I1.Row(j, "decode" if n == 1 and start >= q.prompt_len else "prefill", start, n, hit, back,
                              prefixes.sources(j, hit)))
            if start + n == q.prompt_len + kept[j] and kept[j] < q.output_len:
                release.append(reqs[j].release[kept[j]])
                kept[j] += 1
        steps.append(I1.StepAdvice(tuple(out)))
    return I1.Advice(requests, tuple(steps), tuple(release))


def schedule_of(trace: Any, record: Record) -> Schedule:
    """The trace as v1's schedule: the order its engine took the requests in and each step's rows, by the record's
    ordinals, with the drafts each request's emitting rows kept, one 0 an output for a trace served without a draft.
    The trace's requests are the record's, by id."""
    ordinal = {q.id: k for k, q in enumerate(record.requests)}
    arrival = [ordinal[i] for i in trace.ids]
    kept = {k: tuple(io.kept) if io.kept is not None else (0,) * io.output_len for k, io in zip(arrival, trace.requests)}
    return Schedule(tuple(arrival), tuple(tuple((arrival[i], start, n) for i, start, n in st.rows) for st in trace.steps),
                    tuple(kept[k] for k in range(len(arrival))))


# --- the corpus table --------------------------------------------------------------------------------------------------

INGRESS = ("client", "warden")


def _trace(path: str, stop_ids: Sequence[int], ingress: str = "client") -> tuple[Any, Record, I1.Advice]:
    """A trace-corpus trace, its certified record and its I1 v0 advice; a tick-served trace's record is on its own
    certified tick's buckets, 100 ms buckets otherwise."""
    tr = R.read_corpus_trace(path, ingress=ingress)
    rec = record_of(tr, stop_ids, bucket_s=_bucket_s(tr))
    return tr, rec, advice_of(tr, rec)


def _tick(trace: Any) -> Any:
    """The certified tick a trace's engine admitted by, None for one that admitted on arrival."""
    return None if trace.releases is None else trace.releases.tick


def _bucket_s(trace: Any) -> float:
    """A trace's certified bucket, in seconds: its certified tick's, 100 ms without the tick."""
    return 0.1 if trace.releases is None else trace.releases.tick.bucket_us / 1e6


def served(trace: Any, record: Record, advice: I1.Advice, reg: Registered) -> dict[str, Any]:
    """What a table row says of the serve its charge is of, so that the row alone reports an execution phase's advice:
    the window in seconds (``reg.T`` of the trace's buckets); whether the engine cached prefixes, and the rows that
    began from a cached prefix with the tokens they skipped, which no charge shows where no hit is sent; and each
    window's length terms' bits, the same under every book: a window's charge less them is its segment's."""
    hits = [row.cache_hit for st in advice.steps for row in st.rows if row.cache_hit]
    return {"window_s": round(reg.T * _bucket_s(trace), 6), "prefix_caching": bool(trace.engine.enable_prefix_caching),
            "cache_hit_rows": len(hits), "cache_hit_tokens": sum(hits),
            "length_by_window": {w: sum(n for _v, n in t) for w, t in enumerate(length_tails(record, reg))}}


def windows_of(label: str, path: str, stop_ids: Sequence[int], T: int = 600, ingress: str = "client") -> dict[str, Any]:
    """One trace-corpus trace's residual windows (`residual`), as JSON: what the other traces' books are fit on."""
    tr, rec, adv = _trace(path, stop_ids, ingress)
    wins = residual(adv, rec, Registered(tr.engine, tuple(stop_ids), T=T, tick=_tick(tr)))
    return {"trace": label, "tick": tr.releases is not None, "windows": [asdict(w) for w in wins]}


#: the table's registered caps: a window of a full trace sends more decisions and order entries than `code_v1.Caps`'s
#: defaults, and like v1's own (`CAPS`) no hit, resume or lag bit
TABLE_CAPS = replace(CAPS, arrived=1 << 20, order=1 << 16)
#: and under the certified tick
TICK_TABLE_CAPS = replace(TABLE_CAPS, arrived=0, order=0)


def measure(label: str, path: str, stop_ids: Sequence[int], T: int = 600, ingress: str = "client",
            book: dict | None = None) -> dict[str, Any]:
    """One trace-corpus trace (`replay.read_corpus_trace`; `ingress` the certified order's stand-in: the client's send,
    or on a `.released` view the warden's release) through v1 under `book` (a `code_v1.Book`'s JSON, fit without this
    trace) and under the uniform book: each string's charge per window, the order's part of it and the timing's (the
    decisions' and the order's: the string's bits less those of its windows without them), and whether it decodes back
    to the trace's I1 v0; the residual's decisions and order entries; the length terms' bits, which every string holds
    while the registration sends them (`length_bits`: under the certified tick, the whole charge); the replay's ideal
    order bits (`replay.order_bits`); and I1 v0's own charge (`advice.bits_by_window`) for that advice. A trace served
    under the certified tick is registered with its tick (`tick`: its τ)."""
    tr, rec, adv = _trace(path, stop_ids, ingress)
    tick = _tick(tr)
    row: dict[str, Any] = {"trace": label, "ingress": ingress, "tick": None if tick is None else asdict(tick),
                           **served(tr, rec, adv, Registered(tr.engine, tuple(stop_ids), T=T, tick=tick)),
                           "requests": len(rec.requests), "steps": len(adv.steps), "windows": windows(rec, T),
                           "v0": sum(I1.bits_by_window(adv, T).values())}
    reg = Registered(tr.engine, tuple(stop_ids), T=T, tick=tick)
    wins, tails = residual(adv, rec, reg), length_tails(rec, reg)
    uniform = CV.Book.uniform(TABLE_CAPS if tick is None else TICK_TABLE_CAPS)
    books = {"held_out": CV.Book.from_json(book) if book is not None else uniform, "uniform": uniform}
    for name, b in books.items():
        reg = Registered(tr.engine, tuple(stop_ids), T=T, book=b, tick=tick)
        t0 = time.monotonic()
        data = encode(adv, rec, reg, check=False)
        t1 = time.monotonic()
        got = decode_full(data, rec, reg)
        by = bits_by_window(data, rec, reg)
        unordered = sum(string_of([CV.Window(w.arrived) for w in wins], reg, tails)[1])
        untimed = sum(string_of([CV.Window() for _ in wins], reg, tails)[1])
        row[name] = {"bits": sum(by.values()), "by_window": by, "max_window": max(by.values()),
                     "order_bits": sum(by.values()) - unordered, "timing_bits": sum(by.values()) - untimed,
                     "bytes": len(data), "round_trip": got.advice == adv and got.streams == streams(adv, rec),
                     "book": book_of(reg).digest(), "encode_s": round(t1 - t0, 1),
                     "decode_s": round(time.monotonic() - t1, 1)}
    row["decisions"] = sum(len(w.arrived) for w in wins)
    row["order_entries"] = sum(len(w.order) for w in wins)
    row["inversions"] = sum(idx[0] for w in wins for _, idx in w.order)
    row["order_bits_replay"] = round(R.order_bits(tr), 2)
    row["length_bits"] = sum(n for t in tails for _v, n in t)
    return row


def render(rows: Sequence[dict[str, Any]], T: int) -> str:
    out = [f"| trace | requests | windows (T = {T}) | I1 v0 | v1 charged, held-out book (total; per window) | of it the "
           "order | of it the timing (decisions and order) | of it the length terms | order entries (inversions) | the "
           "replay's ideal order bits | decisions | v1, uniform book (of it the order) | decode(encode(a)) == a |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['trace']} | error: {r['error']} ||||||||||||")
            continue
        h, u = r["held_out"], r["uniform"]
        per = ", ".join(f"{h['by_window'][w]:,}" for w in sorted(h["by_window"]))
        label = f"{r['trace']} (certified tick)" if r.get("tick") else r["trace"]
        out.append(f"| {label} | {r['requests']} | {r['windows']} | {r['v0']:,} | {h['bits']:,}; {per} | "
                   f"{h['order_bits']:,} | {h['timing_bits']:,} | {r['length_bits']:,} | {r['order_entries']:,} "
                   f"({r['inversions']:,}) | "
                   f"{r['order_bits_replay']:,} | {r['decisions']:,} | "
                   f"{u['bits']:,} ({u['order_bits']:,}) | {h['round_trip'] and u['round_trip']} |")
    return "\n".join(out) + "\n"


def _guarded(f: Callable[..., dict[str, Any]], *job: Any) -> dict[str, Any]:
    try:
        return f(*job)
    except (ValueError, RuntimeError) as e:
        return {"trace": job[0], "error": str(e)}


def _windows_job(job: tuple) -> dict[str, Any]:
    return _guarded(windows_of, *job)


def _measure_job(job: tuple) -> dict[str, Any]:
    return _guarded(measure, *job)


def held_out_books(residuals: Sequence[dict[str, Any]]) -> list[dict | None]:
    """For each trace, the book `code_v1.fit_book` fits on every other trace's windows (None where it has none), as a
    registration would fix it from history before the run; for a trace under the certified tick, where no window's
    segment codes anything, the uniform book under `TICK_TABLE_CAPS`."""
    runs = [None if "error" in r else [CV.Window(**w) for w in r["windows"]] for r in residuals]
    out: list[dict | None] = []
    for k, mine in enumerate(runs):
        others = [w for j, w in enumerate(runs) if j != k and w is not None]
        if mine is None:
            out.append(None)
        elif residuals[k].get("tick"):
            out.append(CV.Book.uniform(TICK_TABLE_CAPS).to_json())
        else:
            out.append(CV.fit_book(others, TABLE_CAPS).to_json())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Advice v1's bits per window over trace-corpus traces, with its round "
                                             "trip, against I1 v0's charge; each trace's book fit on the others.")
    ap.add_argument("traces", nargs="+", help="LABEL=DIR: a trace's directory (steps.jsonl, requests.jsonl, meta.json, "
                                              "server.log)")
    ap.add_argument("--stops", default="128001", help="the registered stop ids, comma-separated, EOS first")
    ap.add_argument("--T", type=int, default=600, help="the window, in the trace's certified buckets: its tick's τ_b, "
                                                          "100 ms without the tick")
    ap.add_argument("--ingress", choices=INGRESS, default="client",
                    help="the certified order's stand-in: the client's send (client; on a .released view, the warden's "
                         "release), or the engine's own arrival stamp (warden: no deviations)")
    ap.add_argument("--jobs", type=int, default=0, help="processes (default: one a trace, up to the cores)")
    ap.add_argument("--out", type=Path, required=True, help="the table's JSON")
    a = ap.parse_args(argv)
    stops = tuple(int(s) for s in a.stops.split(","))
    jobs = [(t.split("=", 1)[0], t.split("=", 1)[1], stops, a.T, a.ingress) for t in a.traces]
    workers = a.jobs or min(len(jobs), os.cpu_count() or 1)

    def each(f: Callable[[tuple], dict[str, Any]], js: Sequence[tuple]) -> list[dict[str, Any]]:
        if workers == 1:
            return [f(j) for j in js]
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(f, js))

    books = held_out_books(each(_windows_job, jobs))
    rows = each(_measure_job, [(*j, b) for j, b in zip(jobs, books)])
    a.out.write_text(json.dumps({"T": a.T, "stops": stops, "ingress": a.ingress, "rows": rows}, indent=1) + "\n")
    print(render(rows, a.T), end="")
    return 0 if all("error" not in r and r["held_out"]["round_trip"] and r["uniform"]["round_trip"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
