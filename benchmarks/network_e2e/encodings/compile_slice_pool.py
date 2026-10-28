"""CompileSlice_v2's pool calls (`compile_slice_v2.pool_step`) against the replay over whole traces.

`check_trace` replays a trace as ReplayStep_v1 sees each unit (`replay_vectors.vectors`) and runs each unit's
CacheStep_v2 calls from the empty pool, every lookup's hit and evictable count, every free's returns and every compute's
threshold derived from the pool (the prover's run). It holds them to `_KV`'s: each hit and evictable count ReplayStep_v1
reads (`_KV.hit`'s), each slot's returns (`_KV.free_request`'s), and at every unit the state's free count to the pool's
hashless free count plus its queued copies, the equality CompileSlice_v2's ``cache`` checks. It also holds `pool_step`'s
copy of ReplayStep_v1's wires (`compile_slice_v2.tap`) to the step's rows, and the calls the slice checks (the
witness the prover's run derived) to the prover's pool.

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
import cache_pool_vectors as CPV  # noqa: E402
import replay as RP  # noqa: E402
import replay_vectors as RV  # noqa: E402

from verity.core.protocols.certifier import compile_slice_v2 as V2
from verity.core.protocols.certifier import replay_step as RS


def trace_of(path: Path) -> tuple[RP.Trace, dict[int, tuple[int, ...]]]:
    """A step trace or a derived-format one (`cache_pool_vectors.replay_input`) as a replay `Trace`, and its requests'
    tokens."""
    inp = CPV.replay_input(path)
    n = len(inp.requests)
    tr = RP.Trace(inp.engine, [str(i) for i in range(n)], inp.requests, inp.before, [RP.Step(rows=list(r)) for r in inp.rows],
                  [True] * len(inp.rows), [None] * n, [None] * len(inp.rows))
    return tr, {i: tuple(r.tokens) for i, r in enumerate(inp.requests)}


def statics(st: dict, tokens: dict[int, Sequence[int]], *, CD: int = 4, SW: int = 20) -> SimpleNamespace:
    """ReplayStep_v1's statics `st` with the pool's: every block an entry (CN = NB − 1), every unit of max_model_len in
    ``cert`` (CU), the trace's token width and CD copies an entry."""
    CU = -(-st["L"] // st["B"])
    CTW = max(max(max(t) for t in tokens.values()).bit_length(), 1)
    return SimpleNamespace(**st, CN=st["NB"] - 1, CTW=CTW, CU=CU, CD=CD, SW=SW)


def check_trace(path: Path, *, R: int, W: int, A: int, CD: int = 4, SW: int = 20) -> dict[str, Any]:
    tr, tokens = trace_of(path)
    if not tr.engine.enable_prefix_caching:
        return {"skipped": "no prefix caching"}
    st, vs = RV.vectors(tr, R=R, W=W, A=A)
    S = statics(st, tokens, CD=CD, SW=SW)
    pool, counts = V2.empty(S), Counter()
    for k, (state, inp, out) in enumerate(vs):
        if not RV.fits(st, state, inp, out):
            counts["out_of_class_at"] = k
            break
        where = f"{path.name} unit {k}"
        if state.free != pool.pool.hf + V2.queued(pool.pool):
            raise AssertionError(f"{where}: free {state.free}, the pool's {pool.pool.hf} + {V2.queued(pool.pool)}")
        t = V2.tap(S, state, inp)
        mine = sorted((p, s["id"], s["comp"], s["n"]) for p, s in enumerate(t.post) if s is not None and s["n"])
        if mine != sorted((p, i, start, n) for p, i, _kind, start, n in out.rows):
            raise AssertionError(f"{where}: the tap's rows {mine}, ReplayStep_v1's {out.rows}")
        got = V2.pool_step(S, pool, state, inp, out, tokens, derive=True)
        for c in got.looked:
            if (got.hit[c], got.ev[c]) != (inp.hit[c], inp.evict[c]):
                raise AssertionError(f"{where}: candidate {c}'s hit and evictable count {got.hit[c], got.ev[c]}, "
                                     f"_KV's {inp.hit[c], inp.evict[c]}")
        if tuple(got.ret) != tuple(inp.ret):
            raise AssertionError(f"{where}: returns {got.ret}, _KV's {inp.ret}")
        if out.state.free != got.pool.pool.hf + V2.queued(got.pool.pool):
            raise AssertionError(f"{where}: next free {out.state.free}, the pool's "
                                 f"{got.pool.pool.hf} + {V2.queued(got.pool.pool)}")
        honest = V2.pool_step(S, pool, state, RS.Inputs(inp.arrivals, inp.bits, got.hit, got.ev, got.ret), out, tokens,
                              got.tau)
        if honest.pool != got.pool:
            raise AssertionError(f"{where}: the checked calls' pool isn't the prover's")
        for kind, op in got.calls:
            counts[kind] += 1
            counts["evicting computes"] += kind == "compute" and op.tau > 0
            counts["shared frees"] += kind == "free" and op.ret < op.held
        counts["hits"] += sum(got.hit[c] > 0 for c in got.looked)
        counts["entries_max"] = max(counts["entries_max"], sum(x is not None for x in got.pool.pool.table))
        counts["copies_max"] = max(counts["copies_max"], max((len(x.copies) for x in got.pool.pool.table if x), default=0))
        counts["clock"] = got.pool.pool.clock
        counts["units"] += 1
        pool = got.pool
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
        report[p.name] = check_trace(p, R=a.R, W=a.W, A=a.A)
        print(p.name, json.dumps(report[p.name], sort_keys=True), flush=True)
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
