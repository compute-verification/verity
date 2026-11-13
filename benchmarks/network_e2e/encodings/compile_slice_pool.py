"""CompileSlice_v4's pool calls (`compile_slice_v4.pool_step`) against the replay over whole traces.

`check_trace_v4` replays a trace as ReplayStep_v1 sees each unit (`replay_vectors.vectors`) and runs each unit's calls in
vLLM's order as CompileSlice_v4 makes them, on one CacheStep_v3 stream (`cache_step_v3.Stream`) from the empty pool over
the whole trace, every request numbered by its id and its block keys printed into RK: each hit, evictable count and
return a call derives is `BlockPool`'s (`BlockPool.hit`'s and `BlockPool.free_request`'s, which ReplayStep_v1 reads). It
also holds `pool_step`'s copy of ReplayStep_v1's wires (`compile_slice_v4.tap`) to the step's rows.

    python3 compile_slice_pool.py tests/fixtures/async-prefix-cache.jsonl tests/fixtures/async-crush.json.gz \\
        tests/fixtures/sync-crush.json.gz [--R 16 --W 64 --A 16] [--out REPORT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as RP
import replay_input as RI
import replay_vectors as RV

from verity.protocols.accounting.certifier.verifier import compile_slice as CS
from verity.protocols.accounting.certifier.prover import cache_step_v3 as V3S, compile_slice_v4 as V4


def trace_of(path: Path) -> tuple[RP.Trace, dict[int, tuple[int, ...]]]:
    """A step trace or a derived-format one (`replay_input.replay_input`) as a replay `Trace`, and its requests'
    tokens."""
    inp = RI.replay_input(path)
    n = len(inp.requests)
    tr = RP.Trace(inp.engine, [str(i) for i in range(n)], inp.requests, inp.before, [RP.Step(rows=list(r)) for r in inp.rows],
                  [True] * len(inp.rows), [None] * n, [None] * len(inp.rows))
    return tr, {i: tuple(r.tokens) for i, r in enumerate(inp.requests)}


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
        mine = sorted((p, s["id"], s["comp"], s["n"]) for p, s in enumerate(got.tap.post) if s is not None and s["n"])
        if mine != sorted((p, i, start, n) for p, i, _kind, start, n in out.rows):
            raise AssertionError(f"{where}: the tap's rows {mine}, ReplayStep_v1's {out.rows}")
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
