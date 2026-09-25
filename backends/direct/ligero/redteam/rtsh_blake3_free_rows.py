"""red-team-standard-hash: mutate-and-recompute determinism scan of a leaf gadget (``leaf/blake3.py``, ``leaf/sha256.py``).

For each column shape (word bits, blocks per column), build the one-role column system of ``leaf/blake3_test.py`` and
the honest inputs of one VU row (operand words + carried state: the relation's bit decode and the chain test bind those).
Then, for every row the program computes (``prod`` / ``bits`` outputs), override its value at one column (``1 - v`` and
``v + 1``), recompute every later program op from the overridden value (the honest prover's code downstream), and check
every constraint at that column.  If all hold, the constraints do not pin that row given the inputs: a second witness for
the same operand words (a candidate forgery of the digest).  The system has no cross-column constraint (the carries are
inputs), so each program run tests one mutation per column over ``--width`` tiled copies of the row's columns, at two
column offsets per row.

The positive control drops one addition's low-limb bit decomposition and must show free rows.  Also checked: the honest
witness satisfies every constraint, and its digest at the last column equals ``native`` and folds to the leaf's reference
digest (keyed BLAKE3 / ``sha256_row_digest``).

    python -m backends.direct.ligero.redteam.rtsh_blake3_free_rows [--leaf blake3|sha256] [--shapes 8:0.5,8:1,16:1,8:2]
        [--control-shape 8:0.5] [--width 1536] [--out DIR]

Exit 0 iff no computed row is free for any shape, the honest digests match, and the control shows free rows."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..compile import LigeroCtx, P
from ..leaf.blake3_test import _column_system, _eval, _one_row_inputs


def _column(leaf_name: str, word_bits: int, nblk: float, seed: int):
    """One column system of one role (x rows) and the honest hints of one VU row over its ``steps`` columns."""
    role = 1
    if leaf_name == "blake3":
        leaf, sys, words, carry, digest, carry_out = _column_system(word_bits, role, nblk)
        rows, steps, _car, _pub, hints = _one_row_inputs(leaf, sys, word_bits, role, seed, nblk)
        return leaf, sys, words, carry, digest, rows, steps, hints
    from ..leaf import sha256 as S
    from ..relations import FP8_ADA_RELATION
    leaf = S.Sha256Leaf()
    ctx = LigeroCtx(FP8_ADA_RELATION.params)
    ctx.scope = "t"
    k = int(nblk * 8 * S.BLOCK_LEN) // word_bits
    words = [ctx.hint(f"w[{i}]") for i in range(k)]
    carry = [ctx.hint(f"carry[{i}]") for i in range(S.CARRY_ELEMS)]
    digest, _ = leaf.gadget(ctx, words, word_bits=word_bits, role=role, row_words=S.K_VU, carry_in=carry)
    steps = S.K_VU // k
    rows = np.random.default_rng(seed).integers(0, 1 << word_bits, size=(1, S.K_VU), dtype=np.int64)
    car = leaf.carry_values(rows, word_bits=word_bits, role=role, steps=steps)
    hints = {f"w[{i}]": rows[0].reshape(steps, k)[:, i] for i in range(k)}
    hints |= {f"carry[{i}]": car[0, :, i] for i in range(S.CARRY_ELEMS)}
    return leaf, ctx.sys, words, carry, digest, rows, steps, hints


def _run(sys, hints: dict, width: int, mut: dict[int, np.ndarray], alt: str) -> np.ndarray:
    """``run_program_np`` over ``width`` columns; row ``r`` is overridden at the columns ``mut[r]`` as soon as it is
    computed (so every later op sees the override)."""
    W = np.zeros((sys.m, width), dtype=np.int64)

    def apply(r: int) -> None:
        cols = mut.get(r)
        if cols is not None:
            v = W[r, cols]
            W[r, cols] = (1 - v) % P if alt == "flip" else (v + 1) % P

    for op in sys.program:
        if op[0] == "hint":
            W[op[1]] = np.asarray(hints[op[2]], dtype=np.int64) % P
        elif op[0] == "prod":
            W[op[1]] = (_eval(op[2], W) * _eval(op[3], W)) % P
            apply(op[1])
        elif op[0] == "bits":
            _, row0, nbits, e = op
            x = _eval(e, W)
            for i in range(nbits):
                W[row0 + i] = (x >> i) & 1
                apply(row0 + i)
        elif op[0] == "sels":
            _, row0, los, his, e = op
            x = _eval(e, W)
            x = np.where(x > P // 2, x - P, x)
            for i, (lo, hi) in enumerate(zip(los, his)):
                W[row0 + i] = ((lo <= x) & (x <= hi)).astype(np.int64)
                apply(row0 + i)
        else:
            raise ValueError(op[0])
    return W


def _ok_columns(sys, W: np.ndarray) -> np.ndarray:
    ok = np.ones(W.shape[1], dtype=bool)
    for c in sys.linear:
        acc = np.full(W.shape[1], c.c % P, dtype=np.int64)
        for j, v in c.lin.items():
            acc = (acc + (v % P) * W[j]) % P
        ok &= acc == 0
    for q in sys.quadratic:
        ok &= (_eval(q.a, W) * _eval(q.b, W) - _eval(q.c, W)) % P == 0
    return ok


def scan(leaf_name: str, word_bits: int, nblk: float, width: int, seed: int = 7, control: bool = False) -> dict:
    t0 = time.perf_counter()
    role = 1
    leaf, sys, words, carry, digest, rows, steps, hints = _column(leaf_name, word_bits, nblk, seed)
    W0 = _run(sys, hints, steps, {}, "flip")
    honest_ok = bool(_ok_columns(sys, W0).all())
    dg = np.array([int(_eval(e, W0)[steps - 1]) for e in digest])
    digest_ok = (np.array_equal(dg, leaf.native(rows, word_bits=word_bits, role=role)[0])
                 and leaf.leaf_bytes(dg) == leaf.reference_digest(rows, word_bits=word_bits, role=role)[0])
    dropped = None
    if control:                                           # positive control: drop the median addition's low-limb decomposition
        cands = [c.name for c in sys.linear if c.name.endswith("lo.decomp")]
        dropped = cands[len(cands) // 2]
        sys.linear = [c for c in sys.linear if c.name != dropped]
    width = max(steps, width - width % steps)
    tiled = {n: np.tile(np.asarray(v, dtype=np.int64), width // steps) for n, v in hints.items()}
    computed = []
    for op in sys.program:
        if op[0] == "prod":
            computed.append(op[1])
        elif op[0] == "bits":
            computed += list(range(op[1], op[1] + op[2]))
        elif op[0] == "sels":
            computed += list(range(op[1], op[1] + len(op[2])))
    free: dict[int, dict] = {}
    runs = 0
    for off in (0, steps // 2 + 1):
        for alt in ("flip", "plus1"):
            for b0 in range(0, len(computed), width):
                batch = computed[b0:b0 + width]
                cols = (np.arange(len(batch)) + off) % width
                mut = {r: np.array([c]) for r, c in zip(batch, cols)}
                W = _run(sys, tiled, width, mut, alt)
                ok = _ok_columns(sys, W)
                runs += 1
                for r, c in zip(batch, cols):
                    if ok[c] and r not in free:
                        free[r] = {"row": r, "name": sys.rows[r].name, "kind": sys.rows[r].kind, "alt": alt,
                                   "column_in_vu": int(c % steps)}
    return {"leaf": leaf_name, "word_bits": word_bits, "blocks_per_column": nblk, "steps": steps, "dropped": dropped, "rows": sys.m,
            "computed_rows": len(computed), "linear": len(sys.linear), "quadratic": len(sys.quadratic), "width": width,
            "program_runs": runs, "mutations": 4 * len(computed), "honest_ok": honest_ok, "digest_ok": bool(digest_ok),
            "n_free": len(free), "free": list(free.values())[:20], "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", default="8:0.5,8:1,16:1,8:2", help="word_bits:blocks_per_column, comma-separated")
    ap.add_argument("--width", type=int, default=1536, help="columns per program run (mutations tested at once)")
    ap.add_argument("--leaf", default="blake3", choices=("blake3", "sha256"))
    ap.add_argument("--control-shape", default="8:0.5", help="the positive control's word_bits:blocks_per_column")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    shape = lambda t: (int(t.split(":")[0]), float(t.split(":")[1]) if "." in t.split(":")[1] else int(t.split(":")[1]))  # noqa: E731
    ctl = scan(a.leaf, *shape(a.control_shape), a.width, control=True)
    print("control", json.dumps({k: v for k, v in ctl.items() if k != "free"}), json.dumps(ctl["free"][:3]), flush=True)
    res = []
    for s in a.shapes.split(","):
        r = scan(a.leaf, *shape(s), a.width)
        print(json.dumps({k: v for k, v in r.items() if k != "free"}), flush=True)
        for f in r["free"]:
            print("  FREE", json.dumps(f), flush=True)
        res.append(r)
    ok = (all(r["honest_ok"] and r["digest_ok"] and r["n_free"] == 0 for r in res)
          and ctl["n_free"] > 0 and ctl["honest_ok"])
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "free_rows.json").write_text(json.dumps({"ok": ok, "shapes": res, "control": ctl}, indent=1) + "\n")
    print("no free row (control shows free rows)" if ok else "FREE ROWS FOUND, or the control found none, or an honest witness failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
