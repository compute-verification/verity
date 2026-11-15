"""Advice v0's residual under the derived rules, measured.

The rules and the residual are `verity_vllm.pipeline.advice_code.derived` (`Config`, `Prefixes`, the walk `walk_steps`, the
no-eviction and pool bounds, the tie; `residual`, the encoder, and `reconstruct` and `rebuild`, the compile slice's proof
block; `charge` over `i1_records`, here `v0_records`, or `derived_records`), re-exported here for the modules and tests
that load this one by path. This module reads traces into advice v0 (`load_trace`, `load_corpus_trace`, `advice_of`). `FIELDS` is the table of
every I1 field: its verdict, derivation, checker and cost.

`bits_table` measures v0 (I1's own `bits_by_window`) and the residual per certifier window, the residual at its own least
widths and charged by I1's rule (`charge`), after checking that `rebuild(residual(a)) == a`.

    python derived.py table --trace TRACE.json[.gz] ... --out TABLE.json [--T 600 --bucket-ms 100]

A `--trace` that is a directory is one of the trace corpus's (`load_corpus_trace`; `--block-size`, `--eos`,
`--stop-ids`, `--row-order`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

from verity.protocols.resource_accounting.network.verifier import advice as I1
from verity.protocols.resource_accounting.network.verifier.advice import Advice, RequestAdvice, Row, Source, StepAdvice
from verity_vllm.pipeline.advice_code.derived import (
    CertifiedRequest, charge, Config as Config, config_of, derived_records as derived_records, earliest_windows, FORCED, Holdings,
    pool_bound as pool_bound, PREFIX_ROOT as PREFIX_ROOT, Prefixes, rebuild, reconstruct, Records as Records,
    Residual as Residual, residual, stop_rule as stop_rule, tie_fields as tie_fields, TOKEN_BYTES as TOKEN_BYTES,
    unit_key as unit_key, i1_records as v0_records, Walk as Walk, walk_steps as walk_steps,
)


# --- traces ----------------------------------------------------------------------------------------------------------

def load_trace(path: str | Path) -> dict:
    """A step trace (`sched_trace`'s format, `enc-derived/sched-trace/v2`, optionally gzipped): `config` (`Config`'s
    fields), `requests` in certified-ingress order (`id`, `prompt_ids`, `max_tokens`, `output_ids`, `finish`), and `steps`
    (`t_end_s`, `rows` in scheduling order: `request` (its ingress index), `start`, `n`, `samples`, `new`, `resumed`, and
    on a row that admits its request with a hit, `sources`: [ingress index, tokens] runs)."""
    p = Path(path)
    raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    return json.loads(raw)


def holdings_of(trace: dict) -> Holdings:
    return Holdings(requests=tuple(CertifiedRequest(r["id"], tuple(r["prompt_ids"]), r["max_tokens"]) for r in trace["requests"]),
                    outputs=tuple(tuple(r["output_ids"]) for r in trace["requests"]))


def _jsonl(path: Path) -> list[dict]:
    """A JSON-lines file's records (or its `.gz`'s), less an unterminated last line (a writer killed mid-line)."""
    gz = path.with_name(path.name + ".gz")
    raw = path.read_bytes() if path.exists() or not gz.exists() else gzip.decompress(gz.read_bytes())
    lines = raw.decode("utf-8").split("\n")
    return [json.loads(line) for line in lines[:-1] if line.strip()]


def _flag(argv: Sequence[str], name: str, default: int | None = None) -> int | None:
    return int(argv[argv.index(name) + 1]) if name in argv else default


def load_corpus_trace(d: str | Path, *, block_size: int = 16, eos_token_id: int = 128001,
                      stop_token_ids: Sequence[int] = (), num_gpu_blocks: int | None = None,
                      row_order: str = "batch") -> dict:
    """One trace of the trace corpus (`benchmarks/network_e2e/traces`, `vllm-trace-corpus/v1`: a directory with the
    joined `steps.jsonl` and `requests.jsonl`, `meta.json` and, when kept, `server.log`) in `load_trace`'s format.

    The registered config is the server's argv; the block size and the stop ids are the model's, which the corpus doesn't
    record, so they are parameters, and the walk's tie refuses a stop id they miss. The pool's size, unless given, is
    `server.log`'s "GPU KV cache size" in blocks: vLLM logs the pool's blocks over the blocks a max_model_len request
    takes, times max_model_len, which is never more than the pool, so the bound stays sound. Without either it is 0, so
    the pool bound fails and the resumes are sent.

    A step's rows are in the model runner's batch order (`row_order` "batch", I1's) or in vLLM's scheduling order
    ("scheduling": running, then resumed, then new; the hook gives each list in its loop's order). The requests are in
    the engine's arrival order, which stands for the certified ingress order. The corpus doesn't record whose KV a hit
    reads; `advice_of` names the canonical sources."""
    d = Path(d)
    argv = json.loads((d / "meta.json").read_text()).get("server_argv") or []
    budget = _flag(argv, "--max-num-batched-tokens")
    if budget is None:
        raise ValueError(f"{d}: the server's argv has no --max-num-batched-tokens, the token budget the walk needs")
    if num_gpu_blocks is None:
        log = d / "server.log"
        found = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log.read_text(errors="replace")) if log.exists() else []
        num_gpu_blocks = int(found[-1].replace(",", "")) // block_size if found else 0
    config = {"block_size": _flag(argv, "--block-size", block_size), "num_gpu_blocks": num_gpu_blocks,
              "max_num_batched_tokens": budget, "max_model_len": _flag(argv, "--max-model-len"),
              "max_num_seqs": _flag(argv, "--max-num-seqs", 256),
              "long_prefill_token_threshold": _flag(argv, "--long-prefill-token-threshold", 0),
              "scheduler": "async" if "--async-scheduling" in argv else "sync",
              "enable_prefix_caching": "--no-enable-prefix-caching" not in argv,
              "eos_token_id": eos_token_id, "stop_token_ids": list(stop_token_ids), "row_order": row_order}
    if "--async-scheduling" not in argv and "--no-async-scheduling" not in argv:
        raise ValueError(f"{d}: the server's argv doesn't say whether it schedules asynchronously")
    if config["max_model_len"] is None:
        raise ValueError(f"{d}: the server's argv has no --max-model-len")
    bs = config["block_size"]
    requests = []
    for q in sorted(_jsonl(d / "requests.jsonl"), key=lambda q: q.get("arrival", 0.0)):
        if q["finish_reason"] not in ("stop", "length"):
            raise ValueError(f"{d}: request {q['id']} finished {q['finish_reason']!r}, which the walk can't derive")
        requests.append({"id": q["id"], "prompt_ids": q["prompt_ids"], "max_tokens": q["max_tokens"],
                         "output_ids": q["output_ids"], "finish": "eos" if q["finish_reason"] == "stop" else "length"})
    index = {q["id"]: t for t, q in enumerate(requests)}
    seen: set[int] = set()
    steps, t0 = [], None
    for s in _jsonl(d / "steps.jsonl"):
        if not s["total_num_scheduled_tokens"]:
            continue
        t0 = s["t"] if t0 is None else t0
        running = [r for r in s["cached"] if not r["resumed"]]
        resumed = [r for r in s["cached"] + s["new"] if r["resumed"]]
        new = [r for r in s["new"] if not r["resumed"]]
        rows = {}
        for r in running + resumed + new:
            if not r["num_scheduled_tokens"]:
                continue
            if r["id"] not in index:
                raise ValueError(f"{d}: step {s['step']} schedules {r['id']}, which the engine never freed")
            t, start, n = index[r["id"]], r["num_computed_tokens"], r["num_scheduled_tokens"]
            row = {"request": t, "start": start, "n": n, "new": t not in seen, "resumed": bool(r["resumed"]),
                   "samples": start + n >= len(requests[t]["prompt_ids"]) + r.get("num_output_tokens", 0)}
            if (row["new"] or row["resumed"]) and start % bs:
                raise ValueError(f"{d}: step {s['step']}: {r['id']}'s hit of {start} isn't whole {bs}-token blocks")
            seen.add(t)
            rows[r["id"]] = row
        if row_order == "batch":
            if sorted(s.get("batch") or ()) != sorted(rows):
                raise ValueError(f"{d}: step {s['step']}'s batch order doesn't hold its scheduled requests")
            ordered = [rows[i] for i in s["batch"]]
        else:
            ordered = list(rows.values())
        steps.append({"t_end_s": round(s.get("t_output", s["t"]) - t0, 6), "rows": ordered})
    return {"kind": "enc-derived/sched-trace/v2", "source": str(d), "config": config, "requests": requests, "steps": steps}


def advice_of(trace: dict, bucket_ms: float = 100.0) -> tuple[Advice, tuple[int, ...]]:
    """I1 from a trace, and each I1 request's ingress index. A row admits its request when it is its first or resumes it;
    such a row's `cache_hit` is its start, and its sources the canonical ones (`Prefixes`), whichever block vLLM read.
    Each kept output is released in the bucket its step ends in (unshaped)."""
    order: list[int] = []
    num: dict[int, int] = {}
    for s in trace["steps"]:
        for r in s["rows"]:
            if r["request"] not in num:
                num[r["request"]] = len(order)
                order.append(r["request"])
    reqs = trace["requests"]
    requests = tuple(RequestAdvice(reqs[t]["id"], len(reqs[t]["prompt_ids"]), len(reqs[t]["output_ids"]), reqs[t]["finish"])
                     for t in order)
    prefixes = Prefixes([tuple(reqs[t]["prompt_ids"]) + tuple(reqs[t]["output_ids"]) for t in order], config_of(trace).unit)
    seen: set[int] = set()
    kept = [0] * len(order)
    steps, release = [], []
    for s in trace["steps"]:
        rows = []
        bucket = int(s["t_end_s"] * 1000.0 // bucket_ms)
        for r in s["rows"]:
            prefixes.add(num[r["request"]], r["start"], r["start"] + r["n"])
        for r in s["rows"]:
            i = num[r["request"]]
            admits = i not in seen or r["resumed"]
            seen.add(i)
            hit = r["start"] if admits else 0
            rows.append(Row(i, "decode" if r["n"] == 1 and r["start"] >= requests[i].prompt_len else "prefill", r["start"], r["n"],
                            hit, bool(r["resumed"]), prefixes.sources(i, hit)))
            if r["samples"] and kept[i] < requests[i].output_len:
                kept[i] += 1
                release.append(bucket)
        steps.append(StepAdvice(tuple(rows)))
    return Advice(requests, tuple(steps), tuple(release)), tuple(order)


# --- the derivations -------------------------------------------------------------------------------------------------


def tokens_of(held: Holdings, order: Sequence[int], prompt_len: Sequence[int]) -> list[tuple[int, ...]]:
    """Each I1 request's tokens, from certified request `order[i]`: its prompt ids, then its committed sampled ids."""
    R = len(prompt_len)
    if len(held.requests) != R or sorted(order) != list(range(R)):
        raise ValueError(f"the advice's {R} requests aren't the {len(held.requests)} certified ones, each once")
    for i, (t, P) in enumerate(zip(order, prompt_len)):
        if len(held.requests[t].prompt_ids) != P:
            raise ValueError(f"request {i}: prompt_len {P}, its certified prompt has {len(held.requests[t].prompt_ids)} ids")
    return [tuple(held.requests[t].prompt_ids) + tuple(held.outputs[t]) for t in order]


def canonical_sources(advice: Advice, tokens: Sequence[Sequence[int]], unit: int) -> Advice:
    """`advice` with each hit's sources the canonical ones (`Prefixes`, in hit units of `unit` tokens) over each I1
    request's `tokens`."""
    prefixes, steps = Prefixes(tokens, unit), []
    for st in advice.steps:
        for r in st.rows:
            prefixes.add(r.request, r.start, r.start + r.n)
        steps.append(StepAdvice(tuple(Row(r.request, r.kind, r.start, r.n, r.cache_hit, r.preempted,
                                          prefixes.sources(r.request, r.cache_hit)) for r in st.rows)))
    return Advice(advice.requests, tuple(steps), advice.release)


# --- bits ------------------------------------------------------------------------------------------------------------



def bits_table(trace: dict, *, T: int = 600, bucket_ms: float = 100.0) -> dict:
    """Per window: v0's bits (I1's `bits_by_window`) and the residual's, unshaped (release kept) and shaped (release
    derived), by field, after checking that the residual rebuilds v0 exactly."""
    cfg = config_of(trace)
    adv, order = advice_of(trace, bucket_ms)
    held = holdings_of(trace)
    res = residual(adv, cfg, held, order=order)
    shaped = residual(adv, cfg, held, order=order, shaped=True)
    if rebuild(res, cfg, held, ()) != adv or rebuild(shaped, cfg, held, adv.release) != adv:
        raise AssertionError("the residual doesn't rebuild v0 on this trace")
    v0 = charge(adv, T, v0_records(adv))
    if {w: sum(d.values()) for w, d in v0.items()} != I1.bits_by_window(adv, T):
        raise AssertionError("v0_records doesn't charge as I1's bits_by_window")
    walk = reconstruct(res, cfg, held)
    seen: set[int] = set()
    admitting = new_steps = 0
    for st in adv.steps:
        new_steps += any(r.request not in seen for r in st.rows)
        for r in st.rows:
            admitting += r.request not in seen or r.preempted
            seen.add(r.request)
    num = {t: i for i, t in enumerate(order)}
    vllm_sources = [(tuple(Source(num[a], n) for a, n in r["sources"]), adv.steps[s].rows[j].sources)
                    for s, st in enumerate(trace["steps"]) for j, r in enumerate(st["rows"]) if "sources" in r]
    dv, ds = charge(adv, T, derived_records(res, walk, cfg)), charge(adv, T, derived_records(shaped, walk, cfg))
    earliest, first = earliest_windows(adv, T)
    windows = []
    for w in sorted(v0):
        steps = [s for s, x in enumerate(earliest) if x == w]
        windows.append({"window": w, "steps": len(steps), "rows": sum(len(adv.steps[s].rows) for s in steps),
                        "requests": sum(1 for f in first if earliest[f] == w),
                        "outputs": sum(1 for b in adv.release if b // T == w),
                        "v0_bits": sum(v0[w].values()), "derived_bits": sum(dv.get(w, {}).values()),
                        "derived_shaped_bits": sum(ds.get(w, {}).values()),
                        "forced_bits": sum(dv.get(w, {}).get(f, 0) for f in FORCED),
                        "v0_by_field": v0[w], "derived_by_field": dv.get(w, {})})
    totals = {k: sum(r[k] for r in windows) for k in ("steps", "rows", "requests", "outputs", "v0_bits", "derived_bits",
                                                      "derived_shaped_bits", "forced_bits")}
    by_field: dict[str, dict[str, int]] = {"v0": {}, "derived": {}}
    for r in windows:
        for side, key in (("v0", "v0_by_field"), ("derived", "derived_by_field")):
            for f, b in r[key].items():
                by_field[side][f] = by_field[side].get(f, 0) + b
    return {"kind": "enc-derived/bits/v2", "config": asdict(cfg), "T": T, "bucket_ms": bucket_ms,
            "pool_bound": res.resumes is None, "new_steps": new_steps, "order_sent": len(res.order),
            "explicit_n": len(res.ns),
            "admitting_rows": admitting, "hits": sum(r.cache_hit > 0 for st in adv.steps for r in st.rows),
            "hits_sent": len(res.hits), "hits_derived": admitting - len(res.hits),
            "sources": sum(len(r.sources) for st in adv.steps for r in st.rows),
            "vllm_sources": len(vllm_sources), "vllm_sources_not_earliest": sum(a != b for a, b in vllm_sources),
            "resumed_rows": sum(r.preempted for st in adv.steps for r in st.rows),
            "lag_asked": len(walk.lag_requests), "lag_set": sum(res.lag or ()),
            "finish": {f: sum(q.finish == f for q in adv.requests) for f in ("eos", "length")},
            "windows": windows, "totals": totals, "by_field": by_field}


# --- the field table -------------------------------------------------------------------------------------------------

FIELDS = [
    {"field": "requests[i].id", "verdict": "derived", "bits": "0",
     "from": "certified ingress order, the walk's request order", "derivation": "request i is certified request "
     "order[i] (the walk's); its id is that request's `user`", "checker": "tie",
     "cost": "0: the tie addresses the certified request by index, with no id to compare"},
    {"field": "order (new)", "verdict": "derived; a step's sent sparse where the prediction misses",
     "bits": "0 / a count, and w_step + w_req a new request of each step sent (v0: none, the id says it)",
     "from": "the certified admission order (the network certifier's public schedule), rows, n",
     "derivation": "FCFS admits a step's new requests from the waiting queue's head: the lowest ingress indices not yet "
     "admitted, in arrival order. I1 numbers them by first row: in scheduling order, arrival order; in batch order, Model "
     "Runner V2's stable sort of the step by scheduled tokens, so by n and, at equal n, by arrival",
     "checker": "compile unit (it derives n in arrival order anyway); a sent step equal to the prediction is refused, "
                "and the tie uses each certified request once", "cost": "0"},
    {"field": "prompt_len", "verdict": "stays, forced", "bits": "w_len a request",
     "from": "certified request's id count (witness)", "derivation": "len(prompt ids); the Programs read it (a final prefill "
     "chunk is P - start), so it is public", "checker": "tie",
     "cost": "a 16-bit equality of the certified count with a public value: 15 ANDs a request"},
    {"field": "output_len", "verdict": "derived", "bits": "0", "from": "rows (public); forced by the committed sampled ids",
     "derivation": "the request's sampling rows (async: less the lag bit)",
     "checker": "compiler; tie",
     "cost": "0 in the clear; the tie's stop rule: is_stop on each committed id (3 stop ids of 17 bits: 3*16 + 2 = 50 ANDs), "
             "L <= max_tokens and L == max_tokens (16 + 15): 50 L + 31 ANDs a request"},
    {"field": "finish", "verdict": "derived", "bits": "0", "from": "committed sampled ids, certified max_tokens",
     "derivation": "eos iff the last committed id is a stop id, else length (max_tokens or the model's length)",
     "checker": "tie", "cost": "0 beyond the stop rule"},
    {"field": "lag (new, async only)", "verdict": "forced", "bits": "1 a request whose last two sampling rows are in consecutive steps",
     "from": "committed sampled ids", "derivation": "the last sampling row ran after the stop: its sample is dropped",
     "checker": "tie (a lag follows a stop id)", "cost": "0 beyond the stop rule"},
    {"field": "rows[k].request", "verdict": "stays", "bits": "w_req a row, and w_req a step for its row count",
     "from": "the scheduler's choices", "derivation": "none here: which requests each step runs, in scheduling order, "
     "depends on the arrival times, the timing channel (the replay's piece)",
     "checker": "compiler (the Programs' shapes)", "cost": "0"},
    {"field": "start", "verdict": "derived", "bits": "0", "from": "rows' history, cache_hit",
     "derivation": "the request's previous row's end; an admitting row's (its first, or a resumed one) is its hit",
     "checker": "compile unit (a derived hit makes it witness-dependent)", "cost": "0"},
    {"field": "n", "verdict": "derived", "bits": "0 (w_len a non-decode row when the scheduling order is unknown)",
     "from": "rows' history, prompt_len, max_num_batched_tokens, max_model_len, the certified ingress order",
     "derivation": "min(K - start, the step's budget left, the long-prefill threshold; a running row's max_model_len - 1 - "
     "start), K = prompt_len + sampling rows so far, the budget taken in vLLM's scheduling order (in batch order: running "
     "rows by admission, then resumed, then new ones in arrival order, so only the last arrival is cut short)",
     "checker": "compile unit", "cost": "0"},
    {"field": "kind", "verdict": "derived", "bits": "0", "from": "start, n, prompt_len",
     "derivation": "decode iff n = 1 and start >= prompt_len (I1's rule)", "checker": "compile unit", "cost": "0"},
    {"field": "cache_hit", "verdict": "canonical while nothing can have been evicted; else sent sparse (hits)",
     "bits": "0 / 1 an admitting row + w_hit (hit units) a hit (I1: w_len every row)",
     "from": "every request's tokens (witness), the rows scheduled before it, num_gpu_blocks",
     "derivation": "while the blocks of every request admitted so far, counted unshared from its prompt and kept outputs, "
     "fit num_gpu_blocks - 1 (vLLM hands out unused and uncached free blocks before evicting a cached one), in sync mode "
     "with the scheduling order known: the longest prefix, at most K - 1 tokens in whole hit units (the block, or a hybrid "
     "model's finer hash_block_size), each of whose units a row scheduled before it computed from the same tokens; "
     "past the bound, which cached blocks LRU eviction left is vLLM's block pool over witness, so the hit is sent",
     "checker": "compile unit: the derived hit, and a hit of any other length, shorter included, is refused; none yet "
                "for a sent one", "cost": "a token comparison per hit unit of each admitting row"},
    {"field": "sources", "verdict": "canonical", "bits": "0 (I1: a count, and w_req + w_len a run)",
     "from": "every request's tokens (witness), rows (which request computed KV where by each step)",
     "derivation": "hit unit by hit unit, the earliest-admitted (lowest-numbered) request that has computed the whole "
     "unit by this step (I1's rule) from the same tokens through it, in runs; looked up by a SHA-512 chain over whole "
     "units, every match checked on the tokens",
     "checker": "compile unit (it holds every request's tokens and checks their equality anyway); any other source is "
                "refused", "cost": "the token equality it checks for the KV read, and the lowest number among them"},
    {"field": "preempted", "verdict": "derived under the pool bound; else sent sparse (resumes)",
     "bits": "0 / a count and w_res a resumed row (I1: 1 every row)",
     "from": "rows, num_gpu_blocks", "derivation": "pool_bound: no allocation can fail, so nothing is preempted; "
     "else which rows resume is FCFS's pop of the last running request when the pool runs out, which needs the shared "
     "blocks (witness), so it is sent", "checker": "verifier (the bound, in the clear); none yet for the sent resumes",
     "cost": "0 (in the clear)"},
    {"field": "release", "verdict": "stays (unshaped); derived (shaped)", "bits": "w_rel a kept output / 0",
     "from": "unshaped: the egress head (witness), whose choices are the timing channel; shaped: the ingress anchors, the "
             "output lengths and the declared sync", "derivation": "shaped: served.derive with per-request output lengths",
     "checker": "none yet: no N2 unit holds a shaped egress's counts to the advice",
     "cost": "shaped: served.derive's circuit (served.size: 678,795 ANDs at T = 600, FIXLEN)"},
]


def render(table: dict) -> str:
    lines = ["| window | steps | rows | requests | outputs | v0 bits | derived | derived, shaped | of which forced | derived / v0 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table["windows"] + [dict(table["totals"], window="all")]:
        lines.append(f"| {r['window']} | {r['steps']} | {r['rows']} | {r['requests']} | {r['outputs']} | {r['v0_bits']:,} | "
                     f"{r['derived_bits']:,} | {r['derived_shaped_bits']:,} | {r['forced_bits']:,} | "
                     f"{r['derived_bits'] / max(r['v0_bits'], 1):.3f} |")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="bits per window, v0 against derived")
    t.add_argument("--trace", required=True, type=Path, nargs="+")
    t.add_argument("--out", required=True, type=Path)
    t.add_argument("--T", type=int, default=600, help="buckets a certifier window")
    t.add_argument("--bucket-ms", type=float, default=100.0)
    c = t.add_argument_group("a trace-corpus trace directory (`load_corpus_trace`)")
    c.add_argument("--block-size", type=int, default=16)
    c.add_argument("--eos", type=int, default=128001)
    c.add_argument("--stop-ids", type=int, nargs="*", default=())
    c.add_argument("--row-order", choices=("batch", "scheduling"), default="batch")
    a = ap.parse_args(list(argv) if argv is not None else None)
    tables = {}
    for p in a.trace:
        trace = load_corpus_trace(p, block_size=a.block_size, eos_token_id=a.eos, stop_token_ids=a.stop_ids,
                                  row_order=a.row_order) if p.is_dir() else load_trace(p)
        tables[p.name] = bits_table(trace, T=a.T, bucket_ms=a.bucket_ms)
        print(f"## {p.name}\n\n{render(tables[p.name])}\n")
    a.out.write_text(json.dumps({"kind": "enc-derived/tables/v2", "fields": FIELDS, "traces": tables}, indent=1))


if __name__ == "__main__":
    main()
