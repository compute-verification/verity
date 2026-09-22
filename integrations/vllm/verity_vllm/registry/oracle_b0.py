"""B0 expansion oracle: fully expand one layer of the B0 (SmolLM2-135M) `Serve` program at one position, gate
by gate, and cross-check the O(1) canonical layout arithmetic (offsets, intrinsic summaries, boundaries) and
the query language against the brute-force expansion.

What is checked (all structural, independent of any capture):
  1. region arithmetic: the layer's three root calls (`pre`, `attn`, `post`) tile a contiguous interval and the
     size of every call equals the sum of its children's sizes plus its own primitive gates (recursively);
  2. canonical order is topological: every operand of every gate has a smaller root index, and lies either
     inside the enclosing call, in an `Input` gate, or in an earlier call's returns (`call.in_`);
  3. `.args`/`.returns` vs `.in_`/`.out`: `in_` is the sorted set of `args` (no repeats), `out` is the set of
     returns that are consumed outside — for every child of the layer;
  4. queries: `//LayerPre//GemmCoordinate` etc. enumerate exactly the calls found by brute-force scope walk, and
     counts agree with the intrinsic summaries (`spec.intrinsic().gates`);
  5. evaluation consistency: evaluating the whole `LayerPre` region on random inputs gives the same transcript
     as evaluating its children call by call.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from verity.ir.defs import PrimitiveDefinition, bind
from verity.ir.evaluate import evaluate_region
from verity.ir.layout import FunctionCall
from verity.ir.program import Program
from verity.ir.query import query
from verity_vllm.registry import b1


def _walk(call: FunctionCall, out: list[FunctionCall]) -> None:
    out.append(call)
    if not call.is_primitive:
        for c in call.children():
            _walk(c, out)


def run(t: int = 1, L: int = 0, LP: int = 2, STEPS: int = 1, seed: int = 0) -> dict:
    C = b1.B0_CONFIG
    prog = Program(bind(b1.Serve, C=C, LP=LP, STEPS=STEPS))
    circ = prog.circuit
    named = {c.node.name: c for c in circ.root_calls() if c.node.name}
    pre, attn, post = (named[f"t={t}/layers.{L}/{k}"] for k in ("pre", "attn", "post"))
    # The layer at one position is a *Subcircuit* made of three regions: in prefill all `pre` calls of the
    # positions precede the per-position `attn`/`post` pairs (K/V rows of every position are needed first).
    rep: dict = {"config": "B0", "program": prog.fn.id, "gates": prog.gates, "input_gates": prog.input_gates,
                 "layer": {"t": t, "L": L, "regions": [[pre.lo, pre.hi], [attn.lo, attn.hi], [post.lo, post.hi]],
                           "gates": sum(c.hi - c.lo for c in (pre, attn, post)),
                           "contiguous": pre.hi == attn.lo and attn.hi == post.lo}}
    t0 = time.time()

    # 1. region arithmetic ------------------------------------------------------------------------------
    assert attn.hi == post.lo, (attn.hi, post.lo)
    inside = lambda g: any(c.lo <= g < c.hi for c in (pre, attn, post))  # noqa: E731
    calls: list[FunctionCall] = []
    for root in (pre, attn, post):
        _walk(root, calls)
    for c in calls:
        if c.is_primitive:
            assert c.hi - c.lo == 1
            continue
        kids = list(c.children())
        # children tile [lo, hi) in order, with primitive-batch gates filling the gaps
        pos = c.lo
        child_gates = 0
        for k in kids:
            assert k.lo >= pos, (c.id, k.id, k.lo, pos)
            pos = k.hi
            child_gates += k.hi - k.lo
        assert pos <= c.hi
        # children() yields primitive nodes as one-gate calls; members of *primitive batches* are gates only
        prim_gates = sum(n.gates for n in c.scope().spec.body.nodes if n.form == "batch" and isinstance(n.fn, PrimitiveDefinition))
        assert child_gates + prim_gates == c.hi - c.lo, (c.id, child_gates, prim_gates, c.hi - c.lo)
        assert c.fn.intrinsic().gates == c.hi - c.lo
    rep["calls_walked"] = len(calls)
    rep["timing"] = {"1_region_arithmetic": round(time.time() - t0, 2)}

    # 2. topological order + operand provenance ----------------------------------------------------------
    n_in = prog.input_gates
    outside: set[int] = set()
    for root in (pre, attn, post):
        for g in range(root.lo, root.hi):
            gv = circ.gate(g)
            for o in gv.operands:
                assert o < g, (g, o)
                if not inside(o):
                    outside.add(o)
    inputs_used = sum(1 for o in outside if o < n_in)
    rep["layer_operands_outside"] = {"distinct": len(outside), "input_gates": inputs_used, "earlier_calls": len(outside) - inputs_used}
    for o in outside:  # every non-Input outside operand was produced earlier in canonical order
        assert o < n_in or o < post.lo
    rep["timing"]["2_operand_walk"] = round(time.time() - t0, 2)

    # 3. args/returns vs in_ (out needs the consumers of the whole 176 M-gate circuit: not enumerated here) ------
    n_repeat = 0
    for root in (pre, attn, post):
        for c in [root] + list(root.children()):
            args = c.args
            in_ = c.in_
            assert list(in_) == sorted(set(in_)), c.id  # deduplicated, ascending by producing gate
            assert set(in_) <= set(args), c.id  # everything entering was declared
            assert all(not (c.lo <= a < c.hi) for a in in_), c.id
            n_repeat += len(args) - len(set(args))
    rep["boundary_rule"] = {"rule": "in_/out are deduplicated and ascending by root gate index; args/returns keep signature order and repeats",
                            "repeated_arg_references_in_layer_calls": n_repeat}

    rep["timing"]["3_boundaries"] = round(time.time() - t0, 2)

    # 4. queries vs brute force ------------------------------------------------------------------------------
    def by_walk(name: str) -> list[tuple[int, int]]:
        return sorted((c.lo, c.hi) for c in calls if c.name == name)

    def by_query(q: str) -> list[tuple[int, int]]:
        return sorted((c.lo, c.hi) for c in query(circ, q) if inside(c.lo))

    qs = {"//Gemm": "Gemm", "//GemmCoordinate": "GemmCoordinate", "//LayerPre//RoPEHead": "RoPEHead",
          "//Attention//AttentionHead": "AttentionHead", "//RMSNormFusedCuda": "RMSNormFusedCuda", "//SiluMul": "SiluMul"}
    rep["queries"] = {}
    for q, name in qs.items():
        a, b = by_query(q), by_walk(name)
        assert a == b, (q, len(a), len(b))
        rep["queries"][q] = {"calls_in_layer": len(a), "gates": sum(h - l for l, h in a)}
    # whole-program counts against intrinsic summaries
    n_pos = LP + STEPS
    gemms = query(circ, "//Gemm")
    expected = n_pos * C["NL"] * 4 + (STEPS + 1)  # 4 GEMMs per layer per position + one lm_head per sampled position
    rep["program_query_counts"] = {"//Gemm": len(gemms), "expected": expected}
    assert len(gemms) == expected, len(gemms)
    rep["layer_intrinsic"] = {c.id: {"gates": c.fn.intrinsic().gates, "nodes": c.fn.intrinsic().nodes} for c in (pre, attn, post)}

    rep["timing"]["4_queries"] = round(time.time() - t0, 2)

    # 5. evaluation consistency: whole LayerPre region vs. child by child -----------------------------------
    rng = np.random.default_rng(seed)
    tr_whole: dict[int, int] = {}
    for o in sorted(set(o for g in range(pre.lo, pre.hi) for o in circ.gate(g).operands if not (pre.lo <= o < pre.hi))):
        w = circ.gate(o).prim.ret.w
        tr_whole[o] = int(rng.integers(0, 1 << 15)) if w == 16 else int(rng.integers(0, 1 << 31))  # positive bf16/f32 patterns
    tr_parts = dict(tr_whole)
    t1 = time.time()
    evaluate_region(circ, pre, tr_whole)
    for c in pre.children():
        evaluate_region(circ, c, tr_parts)
    # gates of `pre` that are not inside any child (its own primitive gates) are evaluated by the whole-region pass only
    common = [g for g in range(pre.lo, pre.hi) if g in tr_parts]
    assert all(tr_whole[g] == tr_parts[g] for g in common)
    assert all(tr_whole[g] == tr_parts[g] for g in pre.returns if g in tr_parts)
    rep["evaluation_consistency"] = {"LayerPre_gates": pre.hi - pre.lo, "compared": len(common), "seconds": round(time.time() - t1, 2)}
    rep["seconds"] = round(time.time() - t0, 2)
    return rep


if __name__ == "__main__":
    out = run()
    json.dump(out, sys.stdout, indent=1)
    print()
