"""Scratch (never lands): part1b's Gemm_v2 shapes through `class_statement.unit_lowering`, the builder `UnitStager` writes
a served unit file with (no memo of ANDs across activations), beside part1b's `ir_lower.lowering` (with one).

Each parameter is one row, as `benchmarks/unit_block/run.py` stages a unit. m = rows past the input columns.
"""
import json
import os
import time

from verity.circuits.defs import bind
from verity.definitions.ops import gemm as G
from verity.definitions.ops import prims as P
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL

out = {}
for K, N in ((16, 1), (64, 1), (64, 8), (256, 4), (2048, 1)):
    t0 = time.time()
    D = bind(G.Gemm, K=K, N=N, DOT=P.HopperBF16WgmmaDot16)
    params = D.signature()[0]
    low, _w = CS.unit_lowering(D, f"gemm-{K}-{N}", [(p, [p]) for p, _t in params])
    lay = low.layout
    in_rows = lay.in_words * IL.WORD
    row = {"K": K, "N": N, "lay_counts": lay.counts, "in_rows": in_rows, "useful": lay.useful,
           "m_unit": lay.counts["rows"] - in_rows, "s": round(time.time() - t0, 1)}
    out[f"{K}x{N}"] = row
    print(json.dumps(row), flush=True)

with open(os.path.join(os.environ.get("OUT", "."), "part1d.json"), "w") as f:
    json.dump(out, f, indent=1)
