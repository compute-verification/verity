"""CMP-64 (measurement only, scratch branch): the bits an instance of each Definition in PoUW's served Programs commits,
at the served statics, under the rule `template_widths.py` (CIR-57) counts circuit-check's statics by.

    PYTHONPATH=<tree>:<tree>/integrations/vllm python benchmarks/one_stage/served_pouw_widths.py --out OUT.json \\
      [--shapes 4096x6144,4096x4096,4096x28672,14336x4096] [--rows 64,256]

The served statics are #2615's: vLLM serves PoUW H1 on the eager int8 row (`pouw_h1_gpu.py`, Llama-3.1-8B-INT8), each
linear's call (k, n) at T rows compiled to `SCHEME` with `TILE` = 32 x 32 (`pouw_h1.compiled(k, n, T)`). For every
Definition the compiled Program instantiates: its instances, `input_bits` (its body's Input gates' widths, batch and scan
multiplied), `sig_out_bits` (its signature's return type's widths), and, where `--describe-s` allows, CIR-57's exact
`bits` (Input bits plus each distinct non-Input returned gate's width).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import template_widths as TW  # noqa: E402

from verity.ml.circuits.defs import PrimitiveDefinition  # noqa: E402
from verity.ml.circuits.program import is_input  # noqa: E402
from verity.ml.circuits.types import Array, Record, Tuple, Value  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402

LIMIT = 262_144


def tbits(t) -> int:
    if isinstance(t, Value):
        return t.w
    if isinstance(t, Array):
        return t.n * tbits(t.elem)
    if isinstance(t, Tuple):
        return sum(tbits(f) for f in t.fields)
    if isinstance(t, Record):
        return sum(tbits(f) for _, f in t.fields)
    raise TypeError(t)


def instances(root) -> tuple[Counter, dict]:
    """Each composite Definition's instances in one root, by id (batch and scan multiplied)."""
    per: dict[str, Counter] = {}
    fns: dict[str, object] = {}

    def kids(fn) -> Counter:
        if fn.id in per:
            return per[fn.id]
        c: Counter = Counter()
        for nd in fn.body.nodes:
            if isinstance(nd.fn, PrimitiveDefinition) or is_input(nd.fn):
                continue
            fns.setdefault(nd.fn.id, nd.fn)
            c[nd.fn.id] += TW.K.mult(nd)
        per[fn.id] = c
        for k in c:
            kids(fns[k])
        return c

    fns[root.id] = root
    kids(root)
    order, seen = [], set()

    def topo(fid):
        if fid in seen:
            return
        seen.add(fid)
        for k in per[fid]:
            topo(k)
        order.append(fid)

    topo(root.id)
    total: Counter = Counter({root.id: 1})
    for fid in reversed(order):
        for k, n in per[fid].items():
            total[k] += total[fid] * n
    return total, fns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shapes", default="4096x6144,4096x4096,4096x28672,14336x4096")
    ap.add_argument("--rows", default="64,256")
    ap.add_argument("--describe-s", type=float, default=300)
    a = ap.parse_args()
    out = {"scheme": PH.SCHEME, "tile": list(PH.TILE), "limit": LIMIT, "shapes": {}}
    memo: dict = {}
    for t in map(int, a.rows.split(",")):
        for kn in a.shapes.split(","):
            k, n = map(int, kn.split("x"))
            t0 = time.time()
            c = PH.compiled(k, n, t)
            key = f"T{t}xk{k}xn{n}"
            rec = {"k": k, "n": n, "rows": t, "units": len(c.units), "compile_s": round(time.time() - t0, 1), "defs": []}
            total, fns = instances(c.program.fn)
            for fid, cnt in sorted(total.items(), key=lambda kv: kv[0]):
                fn = fns[fid]
                d = {"id": fid, "family": TW.K.family(fid), "instances": cnt, "gates": fn.gates,
                     "input_bits": TW.input_bits(fn, memo), "sig_out_bits": tbits(fn.signature()[1])}
                d["upper_bits"] = d["input_bits"] + d["sig_out_bits"]
                if d["upper_bits"] > 16:
                    try:
                        with TW.budget(a.describe_s):
                            x = TW.describe(fn, memo)
                        d.update(bits=x["bits"], returned_bits=x["returned_bits"], returned_gates=x["returned_gates"])
                    except (TW.OverBudget, MemoryError, RecursionError) as e:
                        d["describe"] = f"{type(e).__name__}: {e}"
                rec["defs"].append(d)
            out["shapes"][key] = rec
            over = [d for d in rec["defs"] if d.get("bits", d["upper_bits"]) > LIMIT]
            print(key, "units", rec["units"], "defs", len(rec["defs"]), "over", len(over), "s", round(time.time() - t0, 1),
                  flush=True)
            for d in sorted(over, key=lambda d: -d.get("bits", d["upper_bits"])):
                print("  ", d.get("bits"), d["upper_bits"], d["instances"], d["id"][:160], flush=True)
            a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
