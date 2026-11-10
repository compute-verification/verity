"""CompileSlice_v4's pool calls (`compile_slice_v4.pool_step`) against the replay over whole traces.

`check_trace_v4` replays a trace as ReplayStep_v1 sees each unit (`replay_vectors.vectors`) and runs each unit's calls in
vLLM's order as CompileSlice_v4 makes them, on one CacheStep_v3 stream (`cache_step_v3.Stream`) from the empty pool over
the whole trace, every request numbered by its id and its block keys printed into RK: each hit, evictable count and
return a call derives is `BlockPool`'s.

    python3 compile_slice_pool.py tests/fixtures/async-prefix-cache.jsonl tests/fixtures/async-crush.json.gz \\
        tests/fixtures/sync-crush.json.gz [--R 16 --W 64 --A 16] [--out REPORT.json]
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as RP  # noqa: E402
import replay_vectors as RV  # noqa: E402

from verity.core.protocols.certifier import compile_slice as CS
from verity.ml.protocols.certifier import cache_step_v3 as V3S, compile_slice_v4 as V4


def replay_input(path: Path) -> tuple[RP.Engine, list[RP.RequestIO], list[int], list[list[tuple[int, int, int]]]]:
    """A step trace (`replay.read_any`), or a derived-format trace (`*.json.gz`, `derived.load_trace`'s), as replay input:
    its engine, requests, the step each is admitted before, and its own rows. A derived-format request is admitted before
    the first step to end after its arrival, or at the step it is first scheduled in, if that is sooner (an idle engine
    starts its next step on the arrival)."""
    if path.suffix != ".gz":
        tr = RP.read_any(path)
        return tr.engine, list(tr.requests), list(tr.before_step), [list(s.rows) for s in tr.steps]
    t = json.loads(gzip.open(path).read())
    c = t["config"]
    eng = RP.Engine(max_num_seqs=c["max_num_seqs"], max_num_batched_tokens=c["max_num_batched_tokens"],
                    max_model_len=c["max_model_len"], num_gpu_blocks=c["num_gpu_blocks"], block_size=c["block_size"],
                    long_prefill_token_threshold=c["long_prefill_token_threshold"],
                    enable_prefix_caching=c["enable_prefix_caching"], async_scheduling=c["scheduler"] == "async",
                    watermark=c["watermark"])
    rows = [[(x["request"], x["start"], x["n"]) for x in s["rows"]] for s in t["steps"]]
    first: dict[int, int] = {}
    for k, st in enumerate(rows):
        for i, _, _ in st:
            first.setdefault(i, k)
    ends = [s["t_end_s"] for s in t["steps"]]
    reqs, before = [], []
    for i, r in enumerate(t["requests"]):
        out = r["output_ids"]
        reqs.append(RP.RequestIO(prompt_len=len(r["prompt_ids"]), max_tokens=r["max_tokens"], output_len=len(out),
                                 eos=r["finish"] == "eos", tokens=tuple(r["prompt_ids"]) + tuple(out)))
        before.append(min(bisect.bisect_left(ends, r["arrival_s"]) + 1, first[i]))
    return eng, reqs, before, rows


def trace_of(path: Path) -> tuple[RP.Trace, dict[int, tuple[int, ...]]]:
    """A step trace or a derived-format one (`replay_input`) as a replay `Trace`, and its requests' tokens."""
    engine, requests, before, rows = replay_input(path)
    n = len(requests)
    tr = RP.Trace(engine, [str(i) for i in range(n)], requests, before, [RP.Step(rows=list(r)) for r in rows],
                  [True] * len(rows), [None] * n, [None] * len(rows))
    return tr, {i: tuple(r.tokens) for i, r in enumerate(requests)}


def check_trace_v4(path: Path, *, R: int, W: int, A: int) -> dict[str, Any]:
    tr, tokens = trace_of(path)
    if not tr.engine.enable_prefix_caching:
        return {"skipped": "no prefix caching"}
    st, vs = RV.vectors(tr, R=R, W=W, A=A)
    ids = sorted(tokens)
    CTW = max(max(max(t) for t in tokens.values()).bit_length(), 1)
    S = SimpleNamespace(**st, CU=-(-st["L"] // st["B"]), CTW=CTW, K=len(ids), NS=1 << 40, NM=1)
    sl = CS.Slice((), (), 0, tuple(tokens[i] for i in ids))
    S3 = V4.stream_statics(S)
    stream = V3S.Stream(S3, V3S.empty_pool(S3, V4.rk_keys(S, sl, V4.COINS[2])), 1, S.NB - 1)
    rnum = {i: k for k, i in enumerate(ids)}
    nc, fl, counts = (0,) * R, (0,) * R, Counter()
    for k, (state, inp, out) in enumerate(vs):
        if not RV.fits(st, state, inp, out):
            counts["out_of_class_at"] = k
            break
        where = f"{path.name} unit {k}"
        got = V4.pool_step(S, stream, state, inp, nc, fl, rnum)
        for kind, i in got.order:
            if kind == "c" and (got.hit[i], got.ev[i]) != (inp.hit[i], inp.evict[i]):
                raise AssertionError(f"{where}: candidate {i}'s hit and evictable count {got.hit[i], got.ev[i]}, "
                                     f"BlockPool's {inp.hit[i], inp.evict[i]}")
            if kind == "p" and got.ret[i] != inp.ret[i]:
                raise AssertionError(f"{where}: slot {i}'s returns {got.ret[i]}, BlockPool's {inp.ret[i]}")
        counts["hits"] += sum(got.hit[c] > 0 for kind, c in got.order if kind == "c")
        counts["lookups"] += sum(kind == "c" for kind, _ in got.order)
        counts["frees"] += sum(kind == "p" for kind, _ in got.order)
        counts["units"] += 1
        nc, fl = got.nc, got.fl
    counts.update(calls=len(stream.records), slots=len(stream.slots), memos=stream.memos)
    if stream.refusals:
        raise AssertionError(f"{path.name}: the stream refused {stream.refusals[:3]}")
    return dict(counts)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--A", type=int, default=16)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    report = {}
    for p in a.traces:
        report[p.name] = check_trace_v4(p, R=a.R, W=a.W, A=a.A)
        print(p.name, json.dumps(report[p.name], sort_keys=True), flush=True)
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
