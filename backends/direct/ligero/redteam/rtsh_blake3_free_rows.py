"""red-team-standard-hash: single-row freedom scan of the keyed-BLAKE3 leaf gadget (``leaf/blake3.py``).

For each column shape (word bits, blocks per column), build the one-role column system of ``leaf/blake3_test.py``,
run the honest program over one VU row (``steps`` columns), and ask for every internal row ``r`` (every row but the
operand words and the carried state, which the relation's bit decode and the chain test bind) and every column ``j``
whether a perturbation ``W[r, j] += delta`` leaves every constraint satisfied.  A linear constraint with a non-zero
coefficient on ``r`` changes by ``coef * delta``; a quadratic ``a b = c`` changes by
``delta (alpha b + beta a - gamma) + alpha beta delta^2`` (evaluated exactly, per column).  A row free at some column is
a witness cell the digest does not pin (a candidate forgery); also listed: rows no constraint touches.
Also checked: the digest expressions at the last column equal ``native`` (the honest CVs) and fold to the reference
keyed-BLAKE3 digest.

    python -m backends.direct.ligero.redteam.rtsh_blake3_free_rows [--shapes 8:0.5,8:1,16:1,8:2] [--out DIR]

Exit 0 iff no internal row is free at any column, every internal row is touched by a constraint, the digests match, and
the positive control (one addition's bit decomposition dropped) does show free rows."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..compile import P
from ..leaf import blake3 as L
from ..leaf.blake3_test import _column_system, _eval, _one_row_inputs, failing_constraints, run_program_np

DELTAS = (1, P - 1, 1 << 16, 2)


def scan(word_bits: int, nblk: float, seed: int = 7, drop: str | None = None) -> dict:
    t0 = time.perf_counter()
    role = 1
    leaf, sys, words, carry, digest, carry_out = _column_system(word_bits, role, nblk)
    rows, steps, car, pub, hints = _one_row_inputs(leaf, sys, word_bits, role, seed, nblk)
    W = run_program_np(sys, pub, hints)
    bad = failing_constraints(sys, W, pub)
    dropped = None
    if drop is not None:                                   # positive control: remove one constraint, expect free rows
        dropped = next(c.name for c in sys.linear if c.name.endswith(drop))
        sys.linear = [c for c in sys.linear if c.name != dropped]
    dg = np.array([int(_eval(e, W)[steps - 1]) for e in digest])
    digest_ok = (np.array_equal(dg, leaf.native(rows, word_bits=word_bits, role=role)[0])
                 and leaf.leaf_bytes(dg) == leaf.reference_digest(rows, word_bits=word_bits, role=role)[0])
    inputs = {next(iter(e.lin)) for e in list(words) + list(carry)}
    lin_rows: set[int] = set()
    for c in sys.linear:
        lin_rows.update(j for j, v in c.lin.items() if v % P)
    quad_of: dict[int, list[int]] = defaultdict(list)
    for q, Q in enumerate(sys.quadratic):
        for j in set(Q.a.lin) | set(Q.b.lin) | set(Q.c.lin):
            quad_of[j].append(q)
    internal = [r for r in range(sys.m) if r not in inputs]
    untouched = [r for r in internal if r not in lin_rows and r not in quad_of]
    cache: dict[int, tuple] = {}

    def ab(q: int):
        if q not in cache:
            Q = sys.quadratic[q]
            cache[q] = (_eval(Q.a, W), _eval(Q.b, W))
        return cache[q]

    free = []
    for r in internal:
        if r in lin_rows or r not in quad_of:
            continue                                       # a linear coefficient pins the row; untouched listed above
        for d in DELTAS:
            ok_cols = np.ones(steps, dtype=bool)
            for q in quad_of[r]:
                Q = sys.quadratic[q]
                al, be, ga = Q.a.lin.get(r, 0) % P, Q.b.lin.get(r, 0) % P, Q.c.lin.get(r, 0) % P
                a, b = ab(q)
                ch = (d * ((al * b + be * a - ga) % P) + (al * be % P) * (d * d % P)) % P
                ok_cols &= ch == 0
                if not ok_cols.any():
                    break
            if ok_cols.any():
                free.append({"row": r, "name": sys.rows[r].name, "kind": sys.rows[r].kind, "delta": d,
                             "columns": np.nonzero(ok_cols)[0].tolist()})
                break
    return {"word_bits": word_bits, "dropped": dropped, "blocks_per_column": nblk, "steps": steps, "rows": sys.m, "internal": len(internal),
            "linear": len(sys.linear), "quadratic": len(sys.quadratic), "honest_failing": bad[:5], "digest_ok": bool(digest_ok),
            "rows_in_linear": len(lin_rows & set(internal)), "rows_quadratic_only": sum(1 for r in internal if r not in lin_rows and r in quad_of),
            "untouched": [sys.rows[r].name for r in untouched][:20], "n_untouched": len(untouched),
            "free": free[:20], "n_free": len(free), "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", default="8:0.5,8:1,16:1,8:2", help="word_bits:blocks_per_column, comma-separated")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = []
    for s in a.shapes.split(","):
        wb, nb = s.split(":")
        r = scan(int(wb), float(nb) if "." in nb else int(nb))
        print(json.dumps({k: v for k, v in r.items() if k not in ("free", "untouched")}), flush=True)
        for f in r["free"]:
            print("  FREE", json.dumps(f), flush=True)
        for n in r["untouched"]:
            print("  UNTOUCHED", n, flush=True)
        res.append(r)
    ctl = scan(8, 0.5, drop=".r3.g5.a2.lo.decomp")
    print("control", json.dumps({k: ctl[k] for k in ("dropped", "n_free", "n_untouched")}), json.dumps(ctl["free"][:3]), flush=True)
    res_ok = all(not r["honest_failing"] and r["digest_ok"] and r["n_free"] == 0 and r["n_untouched"] == 0 for r in res)
    ok = res_ok and ctl["n_free"] > 0
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "free_rows.json").write_text(json.dumps({"ok": ok, "shapes": res, "control": ctl}, indent=1) + "\n")
    print("no free row" if ok else "FREE OR UNTOUCHED ROWS FOUND (or honest witness failing)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
