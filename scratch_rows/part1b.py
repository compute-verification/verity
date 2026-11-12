"""Scratch (never lands): a served-shape GEMM row lowered to its committed unit file, beside the AND column.

Gemm_v2{K, N, DOT=HopperBF16WgmmaDot16_v1}, the template every served linear binds: the Flock lowering's units, live ANDs and
the unit's computed rows m, against N * (K/16) dot steps at the Definition's 7,872 ANDs plus each coordinate's F2fpBf16.
"""
import json
import os
import time

from verity.circuits.defs import bind
from verity.definitions.ops import gemm as G
from verity.definitions.ops import prims as P
from verity_flock import ir_lower as IL

out = {}
for K, N in ((16, 1), (64, 1), (64, 8), (256, 4), (2048, 1)):
    t0 = time.time()
    D = bind(G.Gemm, K=K, N=N, DOT=P.HopperBF16WgmmaDot16)
    row = {"K": K, "N": N, "definition_ands": IL.definition_ands(D)}
    low = IL.lowering(D, f"gemm-{K}-{N}")
    lay = low.layout
    in_rows = lay.in_words * IL.WORD
    row |= {"units": low.units.n, "cut_words": len(low.units.cut), "tail": len(low.units.tail),
            "lay_counts": lay.counts, "in_rows": in_rows, "m_unit": lay.counts["rows"] - in_rows,
            "rows_instance": low.units.n * (lay.counts["rows"] - in_rows),
            "ands_instance": low.units.n * lay.counts["ands"],
            "dot_column": N * (K // 16) * 7872, "s": round(time.time() - t0, 1)}
    out[f"{K}x{N}"] = row
    print(json.dumps(row), flush=True)

with open(os.path.join(os.environ.get("OUT", "."), "part1b.json"), "w") as f:
    json.dump(out, f, indent=1)
