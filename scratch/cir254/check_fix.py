"""CIR-254: a unit lowered with a `fixes.py` change computes what it computed without: both lowerings' laid-out rows run on
the same random inputs (`Lowering.evaluate`), every row satisfied and the outputs equal.

    python check_fix.py CASE FIX[,FIX] [INSTANCES]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.setrecursionlimit(100000)


def lowering(case: str):
    from stage_profile import unit
    from verity.commitments.windows import interface as IF
    from verity_flock import class_statement as CS

    fn = unit(case)
    in_rows, _out = IF.unit_rows(fn)
    g = {nm: k for k, (nm, _ty) in enumerate(fn.signature()[0])}
    low, _words = CS.unit_lowering(fn, "unit/serving-rows-cap", [(f"in{g[p]}", ps) for p, _s, _b, ps in in_rows])
    return low


def main(case: str, fix: str, instances: int = 16) -> int:
    import fixes

    plain = lowering(case)
    n_in = len(plain.units.in_widths)
    x = np.random.default_rng(254).integers(0, 1 << 16, size=(instances, n_in), dtype=np.uint64)
    x[0] = 0
    x[1] = 0xFFFF
    want, _, ok0 = plain.evaluate(x)
    fixes.apply(fix, {})
    fixed = lowering(case)
    got, _, ok1 = fixed.evaluate(x)
    full = (1 << instances) - 1
    rec = {"case": case, "fix": fix, "instances": instances, "in_words": n_in,
           "plain": {k: int(plain.layout.counts[k]) for k in ("ands", "rows", "nnz_a", "nnz_b")},
           "fixed": {k: int(fixed.layout.counts[k]) for k in ("ands", "rows", "nnz_a", "nnz_b")},
           "rows_satisfied": [int(ok0) == full, int(ok1) == full], "outputs_equal": bool(np.array_equal(want, got)),
           "outputs_nonzero": int(np.count_nonzero(want))}
    print("CIR254 check", json.dumps(rec), flush=True)
    return 0 if rec["outputs_equal"] and all(rec["rows_satisfied"]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], *(int(v) for v in sys.argv[3:4])))
