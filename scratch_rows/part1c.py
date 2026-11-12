"""Scratch (never lands): the calls of part1b's Gemm_v2 unit, each lowered alone: its dot and its F2fpBf16, with their
names, pins (`definition_ands`) and computed rows m, so the template's m sits beside the sum of its calls' m."""
import json
import os

from verity.definitions.ops import prims as P
from verity_flock import ir_lower as IL

out = {}
for D in (P.HopperBF16WgmmaDot16, P.F2fpBf16):
    low = IL.lowering(D, f"alone-{D.name}")
    lay = low.layout
    in_rows = lay.in_words * IL.WORD
    row = {"name": getattr(D, "id", None) or D.name, "definition_ands": IL.definition_ands(D), "units": low.units.n,
           "lay_counts": lay.counts, "in_rows": in_rows, "m_unit": lay.counts["rows"] - in_rows}
    out[row["name"]] = row
    print(json.dumps(row), flush=True)
with open(os.path.join(os.environ.get("OUT", "."), "part1c.json"), "w") as f:
    json.dump(out, f, indent=1)
