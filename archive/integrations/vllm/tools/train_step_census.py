"""One AdamW step of the lf model through the training-step frontend (`program.frontend.train_step`), measured:

- what the default rules and the training rules refuse: the strict derivation's refusal, then a survey that stands a run input in
  for every refused node (`step_report(..., survey=True)`), its holes by op and reason; each derivation's CPU time;
- the survey's census (`verity.ml.circuits.census`) of the step by class, in word gates and in Boolean gates and ANDs:
  each primitive at its Boolean version's counts (`definitions.boolean.boolean_version`, `evaluation.bits.compile`), since the
  reference composites (`LinearRef`, `DotRef`, ...) have no Boolean version of their own; primitives with none are listed;
- te_core's count of the same step: `te_core.flops.Counter`'s MAC rules applied to the step's graph (matmuls m·k·n, one MAC per
  output element of mul, div, addcmul, addcdiv, lerp and a scaled add, and an embedding lookup at its gather's multiplexers per
  word, `gather_selects`), priced as `te_core.gates` prices them: an f32 op's MAC at F32Fma_v4 (the Boolean version of
  `F32Fma_v2`, at its counts as `boolean_prices` computes them), a gather's at one word multiplexer (3 gates a bit, one of them an
  AND: 96 gates for an f32 word).

    python tools/train_step_census.py --size s --batch 2 --out OUT    # from integrations/vllm
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.program.train_step_lf import ADAMW, CTX, VOCAB, loss_fn, meta_step
from verity.ml.definitions.registry.prims import F32FmaV2
from verity_vllm.program.frontend.rules import DEFAULT_RULES
from verity_vllm.program.frontend.rules.training import training_rules
from verity_vllm.program.frontend.train_step import AdamW, StepSurvey, export_step, step_report

#: te_core.opnames's sets (te_core.flops counts these at one MAC per output element)
ELEMENTWISE = {"mul", "div", "addcmul", "addcdiv", "lerp"}
SCALED = {"add", "sub"}
#: te_core.gates's f32 MAC, the Boolean version of `F32Fma_v2`
F32_MAC = "F32Fma_v4"
#: te_core.gates's price of a gather's MAC, one word multiplexer y ⊕ (b ∧ (x ⊕ y)): gates and ANDs per bit, of an f32 word
MUX_GATES_PER_BIT, MUX_ANDS_PER_BIT, WORD_BITS = 3, 1, 32
#: te_core.flops's gather: a 32-bit token, chunks of 1024 words by its low 10 bits
GATHER_INDEX_BITS, GATHER_CHUNK_BITS = 32, 10


def _mux_tree(n: int, k: int) -> int:
    out = 0
    for _ in range(k):
        n = (n + n % 2) // 2
        out += n
    return out


def gather_selects(n: int) -> int:
    """te_core.flops.gather_selects: the multiplexers per word a gather over n words spends."""
    chunk = 1 << GATHER_CHUNK_BITS
    if n <= chunk:
        return _mux_tree(n, GATHER_INDEX_BITS)
    full, r = divmod(n, chunk)
    return (full * _mux_tree(chunk, GATHER_CHUNK_BITS) + (_mux_tree(r, GATHER_CHUNK_BITS) if r else 0)
            + _mux_tree(full + (1 if r else 0), GATHER_INDEX_BITS - GATHER_CHUNK_BITS))


def survey(gm_ex, model, batch, opt, rules, label: str) -> tuple[dict, StepSurvey]:
    t0 = time.process_time()
    strict = step_report(model, loss_fn, opt, batch, rules=rules, exported=gm_ex)
    t1 = time.process_time()
    s = step_report(model, loss_fn, opt, batch, rules=rules, survey=True, exported=gm_ex)
    t2 = time.process_time()
    by = collections.Counter((h["op"], h["reason"]) for h in s.holes)
    out = {"rules": label, "strict_cpu_s": t1 - t0, "survey_cpu_s": t2 - t1,
           "strict": "Program" if strict.unsupported is None else strict.unsupported.to_json(),
           "refused": None if s.refused is None else s.refused.to_json(),
           "holes": len(s.holes), "by_op": [{"op": op, "reason": r, "n": n,
                                             "hint": next(h["hint"] for h in s.holes if h["op"] == op and h["reason"] == r)[:400]}
                                            for (op, r), n in sorted(by.items())]}
    print(json.dumps({k: v for k, v in out.items() if k != "by_op"}, default=str), flush=True)
    for row in out["by_op"]:
        print(f"  {row['n']:5d} {row['op']} {row['reason']}", flush=True)
    return out, s


def boolean_prices(prims: dict) -> dict[str, dict]:
    from verity.ml.circuits.evaluation.bits import compile as compile_bits
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.definitions.boolean import NoBooleanVersion, boolean_version

    out = {}
    for pid, fn in sorted(prims.items()):
        try:
            bv = boolean_version(fn)
        except NoBooleanVersion as e:
            out[pid] = {"boolean": None, "why": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        c = compile_bits(bv).counts
        a, x, n = c.get("and", 0), c.get("xor", 0), c.get("not", 0)
        out[pid] = {"boolean": descriptor_id(bv), "and": a, "xor": x, "not": n, "gates": a + x + n}
    return out


def priced(cen: dict, prices: dict) -> dict:
    """Each root callee class's word gates, Boolean gates and ANDs (its primitives at their prices, times its instances under the
    root), and the word gates of primitives without a Boolean version."""
    rows, unpriced = [], collections.Counter()
    for cid in cen["callees"]:
        cl = cen["classes"][cid]
        if cl["kind"] == "input":
            continue
        inst = sum(n for c, n in zip(cen["nodes"]["callee"], cen["nodes"]["n"]) if cen["callees"][c] == cid)
        g = a = 0
        for pid, k in cl["primitives"].items():
            p = prices.get(pid, {})
            if p.get("boolean") is None:
                unpriced[pid] += inst * k
                continue
            g += inst * k * p["gates"]
            a += inst * k * p["and"]
        rows.append({"class": cid, "instances": inst, "word_gates": inst * cl["gates"], "gates": g, "ands": a})
    rows.sort(key=lambda r: -r["gates"])
    return {"classes": rows, "gates": sum(r["gates"] for r in rows), "ands": sum(r["ands"] for r in rows),
            "word_gates": sum(r["word_gates"] for r in rows), "unpriced_word_gates": dict(unpriced)}


def te_core_count(gm) -> dict:
    """te_core.flops.Counter's MACs for the step's graph by its shape rules, by op, and their gates at te_core.gates's prices."""
    from torch.utils import flop_counter

    by = collections.Counter()

    def val(a):
        return a.meta.get("val") if isinstance(a, torch.fx.Node) else a

    for n in gm.graph.nodes:
        if n.op != "call_function" or not isinstance(n.target, torch._ops.OpOverload):
            continue
        p = n.target.overloadpacket
        name = p.__name__
        args = [val(a) for a in n.args]
        out = n.meta.get("val")
        if name == "mm":
            (m, k), (_, nn) = args[0].shape, args[1].shape
            by["matmul " + name] += m * k * nn
        elif name == "bmm":
            (b, m, k), (_, _, nn) = args[0].shape, args[1].shape
            by["matmul " + name] += b * m * k * nn
        elif p in flop_counter.flop_registry:
            by["matmul " + name] += int(flop_counter.flop_registry[p](*args, **n.kwargs, out_val=out)) // 2
        elif name == "embedding":
            by["gather embedding"] += gather_selects(args[0].shape[0]) * out.numel()
        elif name in ELEMENTWISE or (name in SCALED and n.kwargs.get("alpha", 1) != 1):
            by["elementwise " + name] += out.numel()
    macs = sum(by.values())
    gather = sum(n for k, n in by.items() if k.startswith("gather "))
    fma = boolean_prices({F32FmaV2.id: F32FmaV2})[F32FmaV2.id]
    if fma["boolean"] != F32_MAC:
        raise SystemExit(f"te_core.gates prices an f32 MAC at {F32_MAC}, but {F32FmaV2.id}'s Boolean version is {fma}")
    per = {"f32": fma["gates"], "gather": MUX_GATES_PER_BIT * WORD_BITS}
    return {"macs": macs, "by_op": dict(sorted(by.items())), "gates_per_mac": per, "f32_mac": fma,
            "gates": (macs - gather) * per["f32"] + gather * per["gather"],
            "ands": (macs - gather) * fma["and"] + gather * MUX_ANDS_PER_BIT * WORD_BITS,
            "rule": "te_core.flops.Counter's MAC rules on the step's ATen graph; every op f32, priced at F32Fma_v4 "
                    "(te_core.gates's gemm/f32 and elementwise/f32), a gather's MAC at one f32 word multiplexer (its gather)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", default="s")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--rules", default="default,training")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    model, batch = meta_step(a.size, a.batch, a.ctx, a.vocab)
    opt = AdamW(**ADAMW)
    t0 = time.process_time()
    ex = export_step(model, loss_fn, opt, batch)
    rec = {"size": a.size, "batch": a.batch, "ctx": a.ctx, "vocab": a.vocab, "adamw": ADAMW,
           "export_cpu_s": time.process_time() - t0, "graph_nodes": len(ex[0].graph.nodes),
           "params": sum(p.numel() for p in model.parameters()), "torch": torch.__version__}
    print(json.dumps(rec), flush=True)
    rec["te_core"] = te_core_count(ex[0])
    print("te_core", json.dumps({k: v for k, v in rec["te_core"].items() if k != "by_op"}), flush=True)
    s = None
    for label in a.rules.split(","):
        rs = DEFAULT_RULES if label == "default" else training_rules()
        rec[label], sv = survey(ex, model, batch, opt, rs, label)
        if label == "training":
            s = sv
        (a.out / "census.json").write_text(json.dumps(rec, indent=1, default=str))
    if s is not None and s.census is not None:
        cen = s.census
        prices = boolean_prices(s.primitives)
        rec["prices"] = prices
        rec["census"] = {"totals": {k: cen["totals"][k] for k in ("gates", "computed_gates", "inputs", "values",
                                                                  "value_bits", "classes")},
                         "priced": priced(cen, prices)}
        rec["theta"] = {"names": len(s.theta["names"]), "input_nodes": s.theta["input_nodes"], "bits": s.theta["bits"]}
        pr = rec["census"]["priced"]
        print("census", json.dumps({"word_gates": pr["word_gates"], "gates": pr["gates"], "ands": pr["ands"],
                                    "unpriced_word_gates": pr["unpriced_word_gates"]}), flush=True)
        for r in pr["classes"][:25]:
            print(f"  {r['gates']:>22,} gates {r['ands']:>22,} ands {r['instances']:>12,} x {r['class'][:90]}", flush=True)
    (a.out / "census.json").write_text(json.dumps(rec, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
