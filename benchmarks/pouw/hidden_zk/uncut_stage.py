"""uncut_stage.py SET N OUT: stage a template's input set with its whole coordinate as one unit (``ir_lower.lowering``'s default
``units``, no cut: every connected component one unit, so a GEMM coordinate's chained accumulator stays inside the unit, with no
tail stage and no wired cut port), in place of the template's own ``circuit_lowering`` (for gemm-coordinate, ``tc_units``'
one unit per k-step). Writes ``OUT/circuit.txt``, ``inst-N.bin`` and ``pub-N.bin`` as ``python -m verity_flock.circuit`` does,
and prints the pin and the unit's rows, nnz and k_log as JSON."""
import json
import sys
from pathlib import Path

from verity_flock import circuit as FC
from verity_flock import ir_lower as IL
from verity_numerical.bench.input_sets import InputSet


def main() -> int:
    path, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    s = InputSet.open(path)
    sub = s.subcircuit
    low = IL.lowering(sub.definition(), f"{sub.id}/whole")
    if low.units.cut or low.units.n != 1:
        raise SystemExit(f"{sub.id}: the whole lowering is {low.units.n} units, cut {bool(low.units.cut)}: not one uncut unit")
    FC.stage(low, s, 0, n, out)
    text = (out / "circuit.txt").read_text()
    meta = json.loads(text.split("\n", 2)[1][5:])
    print(json.dumps({"subcircuit": sub.id, "unit": low.name, "pin": FC.digest(text), "useful": low.layout.useful,
                      "counts": getattr(low.layout, "counts", None), "unit_log": meta["unit_log"], "k_log": meta["k_log"],
                      "per_vu": meta["per_vu"], "vus_per_block": meta["vus_per_block"], "circuit_bytes": len(text)}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
