"""Scratch (never lands): each training_slowdown.gates Definition's AND count beside the rows of its committed unit file.

For each Definition: compile(D)'s (AND, XOR, NOT), ir_lower.definition_ands (read off, never lowered), the Flock lowering's
live ANDs, asserts, table-read rows, the laid-out unit's rows, input rows and computed rows m = rows - input rows (what
`FilePart.ofCircuit`'s m counts a unit), units an instance and cut words (a native tail's words have no rows).
"""
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "training_slowdown"))

from training_slowdown.gates import DEFINITIONS, PYTHON_NAMES  # noqa: E402
from verity.circuits.evaluation.bits import compile as compile_bits  # noqa: E402
from verity_flock import ir_lower as IL  # noqa: E402

out = {}
for name, (module, a, x, n) in DEFINITIONS.items():
    t0 = time.time()
    D = getattr(importlib.import_module(f"verity.{module}"), PYTHON_NAMES[name])
    c = compile_bits(D).counts
    row = {"pinned": [a, x, n], "compile": [c.get("and", 0), c.get("xor", 0), c.get("not", 0)]}
    try:
        row["definition_ands"] = IL.definition_ands(D)
    except Exception as e:  # noqa: BLE001
        row["definition_ands"] = f"{type(e).__name__}: {e}"[:300]
    try:
        low = IL.lowering(D, name)
        lay = low.layout
        in_rows = lay.in_words * IL.WORD
        row |= {"units": low.units.n, "cut_words": len(low.units.cut), "tail": bool(low.units.tail),
                "lay_counts": lay.counts, "in_rows": in_rows, "m_unit": lay.counts["rows"] - in_rows,
                "out_bits": sum(lay.out_bits), "in_bits": sum(lay.in_bits),
                "circuit_ands": low.circuit.counts().get("ands") if low.circuit is not None else None}
    except Exception as e:  # noqa: BLE001
        row["lowering"] = f"{type(e).__name__}: {e}"[:500]
    row["s"] = round(time.time() - t0, 1)
    out[name] = row
    print(name, json.dumps(row, default=str), flush=True)

dst = os.environ.get("OUT", ".")
with open(os.path.join(dst, "part1.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
