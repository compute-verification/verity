"""Scratch (never lands): the calls of part1b's Gemm_v2 unit, each lowered alone: its dot and its F2fpBf16, each wrapped
in a one-call composite (a standalone program needs one call), with pins (`definition_ands`) and computed rows m, so
the template's m sits beside the sum of its calls' m."""
import json
import os

from verity.circuits.defs import composite
from verity.circuits.types import Array
from verity.definitions.ops import prims as P
from verity_flock import ir_lower as IL


@composite("ScratchRowsDot", 1, [], lambda S: ((("acc", P.F32), ("a", Array(16, P.BF16)), ("b", Array(16, P.BF16))),
                                                P.F32))
def ScratchRowsDot(B, S, acc, a, b):
    """One HopperBF16WgmmaDot16_v1 call."""
    return B.call(P.HopperBF16WgmmaDot16, acc, a, b)


@composite("ScratchRowsF2fp", 1, [], lambda S: ((("a", P.F32),), P.BF16))
def ScratchRowsF2fp(B, S, a):
    """One F2fpBf16_v1 call."""
    return B.call(P.F2fpBf16, a)


out = {}
for D in (ScratchRowsDot, ScratchRowsF2fp):
    low = IL.lowering(D, f"alone-{D.name}")
    lay = low.layout
    in_rows = lay.in_words * IL.WORD
    row = {"definition_ands": IL.definition_ands(D), "units": low.units.n, "lay_counts": lay.counts,
           "in_rows": in_rows, "m_unit": lay.counts["rows"] - in_rows}
    out[D.name] = row
    print(D.name, json.dumps(row), flush=True)
with open(os.path.join(os.environ.get("OUT", "."), "part1c.json"), "w") as f:
    json.dump(out, f, indent=1)
