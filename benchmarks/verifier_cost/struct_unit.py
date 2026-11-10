"""The H1 tile unit (``benchmarks/pouw/h1_prove/h1unit.Shape``) through `verity_flock.struct_lower`, for ``struct_tmpl.sh``.

    python benchmarks/verifier_cost/struct_unit.py K [R C] [--support S] [--first-appearance]
        [--scattered-returns] [--out UNIT.tmpl] [--flat | --check]

prints ``STRUCT {json}`` (the unrolled totals, multiplied through the calls, and the template's size) and writes the template,
with each Definition's name in ``UNIT.tmpl.names.json``; ``--flat`` counts the unrolled lowering instead, and ``--check`` runs
`struct_lower.check`.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from verity_flock import struct_lower as SL
from verity_flock import tmpl as T

H1 = Path(__file__).resolve().parents[1] / "pouw" / "h1_prove"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="struct_unit")
    ap.add_argument("shape", nargs="+", type=int)
    ap.add_argument("--support", type=int, default=SL.SUPPORT)
    ap.add_argument("--first-appearance", action="store_true", help="number a callee's inputs as its arguments name them")
    ap.add_argument("--scattered-returns", action="store_true", help="return forms over a callee's columns where they lie")
    ap.add_argument("--out")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    sys.path[:0] = [str(H1), str(H1.parent / "hidden_zk")]
    import h1unit as U

    shape = U.Shape(*a.shape)
    t0 = time.time()
    if a.flat:
        out = SL.flat(shape.unit())
    elif a.check:
        out = {"check": SL.check(shape.unit(), a.support, column_order=not a.first_appearance, gather=not a.scattered_returns)}
    else:
        t, lw, top = SL.lower(shape.unit(), a.support, f"h1-struct-{shape.k}", column_order=not a.first_appearance,
                              gather=not a.scattered_returns)
        size = None
        if a.out:
            text = T.write(t)
            Path(a.out).write_text(text)
            size = len(text)
            Path(a.out + ".names.json").write_text(json.dumps([s.key[0] for s in lw.order] + ["(unit)"]))
        out = SL.stats(t, lw, top, size) | {"support": a.support, "column_order": not a.first_appearance,
                                            "gather": not a.scattered_returns}
    out |= {"k": shape.k, "tile": [shape.r, shape.c], "seconds": round(time.time() - t0, 1),
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    print("STRUCT " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
