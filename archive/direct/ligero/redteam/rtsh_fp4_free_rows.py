"""red-team-standard-hash-2: mutate-and-recompute determinism scan of the composed ``fp4-nvf4+poseidon2`` column system
(``hashchain.compose(fp4.hashed.FP4_HASHED)``: the NVFP4 unit, the FP4Format operand decode on 4-bit elements and the
Poseidon2 sponge on 24-bit lanes), torch-free.

The inputs of a column are the committed operand bits (``a[i].b<j>`` / ``b[i].b<j>``), the carried sponge capacity
(``a.cap[x]`` / ``b.cap[x]``), the chain-linked accumulator rows (``sys.chain["c"]``) and the public pins; the chain
test and the sponge bind those across columns.  Every other row -- the program's ``prod`` / ``bits`` / ``sels`` / ``inv``
outputs AND the prover-chosen hint rows (the 141 decode pins turned hints, the unit's own hints) -- is overridden at one
column (``1 - v`` and ``v + 1``), every later op is recomputed from it, and every constraint is checked at that column.
A row whose override satisfies every constraint is *free*.  It is *effective* when the override changes a chain-linked
output (``sys.chain["y"]``: the accumulator words and the sponge's outgoing capacity, which carries the digest): the same
inputs with another consequence, a candidate forgery.  A free row that leaves every output unchanged (an ``inv`` of a
zero, a shift amount applied to a zero value) is a second witness for the same statement, not a soundness break.

Two passes per family: the full system, and the decode + sponge layer alone (``layer``: its ``hash.*`` constraints,
the 141 converted pins and the layer's rows mutated; effective = a converted pin or a chain-linked output changes).
Positive controls: one sponge constraint dropped (the median ``hash.*`` quadratic of the Poseidon2 rounds; full pass)
and one decode constraint dropped (``hash.a[3].n``; layer pass); each must show an effective free row.  Also checked: the honest witness
satisfies every constraint, its published digests equal the committer's (``leaf.native``) and the system's id is the
Rust pin (``hashed_sys_id`` 8c6d260c...).

    python -m archive.direct.ligero.redteam.rtsh_fp4_free_rows [--families random,zero-scales,...] [--copies 64] [--out DIR]

Exit 0 iff no effective free row in any family, the honest checks pass, and both controls show effective free rows."""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from pathlib import Path

import numpy as np

from .. import hashchain
from .. import poseidon2 as h2
from ..compile import P
from ..fp4.hashed import FP4_HASHED, PIN_NAMES, hashed_hints_np, system_id_np
from ..fp4.relation import STEPS, WORDS
from ..fp4.witness import MODEL, check_constraints_np, eval_expr_np, hints_fp4, random_units

PINNED_SYS_ID = "8c6d260c675cad40ae7263db08974f1842d04997f344cb4f9e52821e70bb25f5"
INPUT = re.compile(r"^[ab]\[\d+\]\.b\d$|^[ab]\.cap\[\d+\]$")
FAMILIES = ("random", "zero-scales", "subnormal-scales", "top-scales", "sparse", "all-zero-codes", "max-codes", "cancel")


def _vu(family: str, seed: int):
    rng = np.random.default_rng(seed)
    _, A, B = random_units(rng, STEPS, family)
    c_in, c = np.zeros(STEPS, dtype=np.int64), 0
    for s in range(STEPS):
        c_in[s] = c
        c = MODEL.step(c, [int(x) for x in A[s]], [int(x) for x in B[s]])
    return A, B, c_in


def _honest(hr, family: str, seed: int):
    a, b, c_in = _vu(family, seed)
    base = hints_fp4(dataclasses.replace(hr.sys, hints=hr.sys.hints[: hr.n_base_hints]), c_in, a, b)
    extra, digests, pub = hashed_hints_np(hr, a, b, 1)
    return a, b, np.concatenate([base, extra], 0), pub, digests


def _run(sys, pub: dict, hints: np.ndarray, mut: dict[int, np.ndarray], alt: str) -> np.ndarray:
    """``build_witness_np`` with row ``r`` overridden at the columns ``mut[r]`` as soon as it is written."""
    l = hints.shape[1]
    W = np.zeros((sys.m, l), dtype=np.int64)
    hpos = {n: i for i, (_, n) in enumerate(sys.hints)}

    def apply(r: int) -> None:
        cols = mut.get(r)
        if cols is not None:
            v = W[r, cols]
            W[r, cols] = (1 - v) % P if alt == "flip" else (v + 1) % P

    for op in sys.program:
        kind = op[0]
        if kind == "pin":
            W[op[1]] = np.asarray(pub[op[2]], dtype=np.int64) % P
        elif kind == "hint":
            W[op[1]] = hints[hpos[op[2]]] % P
            apply(op[1])
        elif kind == "prod":
            W[op[1]] = (eval_expr_np(op[2], W) * eval_expr_np(op[3], W)) % P
            apply(op[1])
        elif kind == "bits":
            _, row0, nbits, e = op
            x = eval_expr_np(e, W)
            for i in range(nbits):
                W[row0 + i] = (x >> i) & 1
                apply(row0 + i)
        elif kind == "sels":
            _, row0, k0s, k1s, key = op
            x = eval_expr_np(key, W)
            x = np.where(x > P // 2, x - P, x)
            for i, (k0, k1) in enumerate(zip(k0s, k1s)):
                W[row0 + i] = ((x >= k0) & (x <= k1)).astype(np.int64)
                apply(row0 + i)
        elif kind == "inv":
            x = eval_expr_np(op[2], W)
            W[op[1]] = np.asarray([pow(int(v), P - 2, P) for v in x], dtype=np.int64)
            apply(op[1])
        else:
            raise ValueError(kind)
    return W


def _ok_columns(sys, W: np.ndarray, pub: dict, only=None) -> np.ndarray:
    ok = np.ones(W.shape[1], dtype=bool)
    for c in sys.linear:
        if only is not None and not only(c.name):
            continue
        acc = np.full(W.shape[1], c.c % P, dtype=np.int64)
        for j, v in c.lin.items():
            acc = (acc + (v % P) * W[j]) % P
        tgt = np.asarray(pub[c.pub], dtype=np.int64) % P if c.pub else 0
        ok &= acc == tgt
    for q in sys.quadratic:
        if only is not None and not only(q.name):
            continue
        ok &= (eval_expr_np(q.a, W) * eval_expr_np(q.b, W) - eval_expr_np(q.c, W)) % P == 0
    return ok


def mutable_rows(sys) -> list[int]:
    fixed = {r for r, _ in sys.pins} | set(sys.chain["c"]) | {r for r, n in sys.hints if INPUT.match(n)}
    rows = []
    for op in sys.program:
        if op[0] in ("hint", "prod", "inv"):
            rows.append(op[1])
        elif op[0] == "bits":
            rows += list(range(op[1], op[1] + op[2]))
        elif op[0] == "sels":
            rows += list(range(op[1], op[1] + len(op[2])))
    return [r for r in rows if r not in fixed]


def scan(family: str, copies: int, seed: int = 7, drop: str | None = None, layer: bool = False) -> dict:
    """``layer``: the decode + sponge layer alone -- only its own (``hash.*``) constraints are checked, the mutated rows
    are the 141 converted pins (hint rows) and the layer's computed rows; effective = a converted pin or a chain-linked
    output changes.  The full-system scan cannot see a pin the decode leaves free (the unit's hints would have to be
    regenerated for the forged pin), so this is where the decode control must show up."""
    t0 = time.perf_counter()
    hr = hashchain.compose(FP4_HASHED)
    sys = hr.sys
    sys_id = system_id_np(sys).hex()
    a, b, hints, pub, digests = _honest(hr, family, seed)
    W0 = _run(sys, pub, hints, {}, "flip")
    honest_ok = check_constraints_np(sys, W0, pub) == []
    da = hr.leaf.native(a.reshape(1, -1), word_bits=4, role=h2.ROLE_X)
    db = hr.leaf.native(b.reshape(1, -1), word_bits=4, role=h2.ROLE_W)
    digest_ok = bool(np.array_equal(digests[:, :8], da) and np.array_equal(digests[:, 8:], db))
    dropped = None
    if drop == "decode":
        dropped = "hash.a[3].n"
        sys.linear = [c for c in sys.linear if c.name != dropped]
    elif drop == "sponge":
        m1 = hr.census.get("rows_base", 0) + hr.census.get("rows_private_operands", 0)
        cands = [q for q in sys.quadratic if q.name.startswith("hash.") and min(q.a.lin or {10**9: 0}) >= m1]
        dropped = cands[len(cands) // 2].name
        sys.quadratic = [q for q in sys.quadratic if q.name != dropped]
    width = STEPS * copies
    tile = lambda x: np.tile(np.asarray(x, dtype=np.int64), (1, copies)) if np.ndim(x) == 2 else np.tile(np.asarray(x, dtype=np.int64), copies)  # noqa: E731
    th, tp = tile(hints), {k: tile(v) for k, v in pub.items()}
    Wt = np.tile(W0, (1, copies))
    Yt = np.stack([eval_expr_np(e, Wt) for e in sys.chain["y"]])
    rows = mutable_rows(sys)
    only = None
    pin_rows = {r for r, n in sys.hints if n in set(PIN_NAMES)}
    if layer:
        base_m = hr.census["rows_base"]
        rows = [r for r in rows if r in pin_rows or r >= base_m]
        only = lambda name: name.startswith("hash.") or name.startswith("pin:hash")  # noqa: E731
    free: dict[int, dict] = {}
    runs = 0
    for off in (0, STEPS // 2 + 1):
        for alt in ("flip", "plus1"):
            for b0 in range(0, len(rows), width):
                batch = rows[b0:b0 + width]
                cols = (np.arange(len(batch)) + off) % width
                W = _run(sys, tp, th, {r: np.array([c]) for r, c in zip(batch, cols)}, alt)
                ok = _ok_columns(sys, W, tp, only)
                Y = np.stack([eval_expr_np(e, W) for e in sys.chain["y"]])
                runs += 1
                for r, c in zip(batch, cols):
                    if not ok[c]:
                        continue
                    changed = np.nonzero(W[:, c] != Wt[:, c])[0]
                    others = [int(x) for x in changed if x != r]
                    y_changed = bool((Y[:, c] != Yt[:, c]).any())
                    eff = y_changed or bool(layer and ({r, *others} & pin_rows))
                    prev = free.get(r)
                    if prev is None or (eff and not prev["effective"]):
                        free[r] = {"row": r, "name": sys.rows[r].name, "kind": sys.rows[r].kind, "alt": alt,
                                   "column_in_vu": int(c % STEPS), "effective": eff, "y_changed": y_changed,
                                   "others_changed": [sys.rows[x].name for x in others[:4]]}
    eff = [f for f in free.values() if f["effective"]]
    return {"family": family, "layer": layer, "sys_id": sys_id, "sys_pinned": sys_id == PINNED_SYS_ID, "dropped": dropped, "rows": sys.m,
            "mutable_rows": len(rows), "linear": len(sys.linear), "quadratic": len(sys.quadratic), "width": width,
            "program_runs": runs, "mutations": 4 * len(rows), "honest_ok": honest_ok, "digest_ok": digest_ok,
            "n_free": len(free), "n_effective": len(eff), "effective": eff[:20],
            "y_invariant": sorted({re.sub(r"\[\d+\]", "[*]", f["name"]) for f in free.values() if not f["effective"]})[:20],
            "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--copies", type=int, default=64, help="tiled copies of the 24-column VU per program run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctls = [scan("random", a.copies, drop="sponge"), scan("random", a.copies, drop="decode", layer=True)]
    for c in ctls:
        print("control", json.dumps({k: v for k, v in c.items() if k != "effective"}), json.dumps(c["effective"][:2]), flush=True)
    res = []
    for fam in a.families.split(","):
        for layer in (False, True):
            r = scan(fam, a.copies, layer=layer)
            print(json.dumps({k: v for k, v in r.items() if k != "effective"}), flush=True)
            for f in r["effective"]:
                print("  EFFECTIVE FREE", json.dumps(f), flush=True)
            res.append(r)
    ok = (all(r["honest_ok"] and r["digest_ok"] and r["sys_pinned"] and r["n_effective"] == 0 for r in res)
          and all(c["n_effective"] > 0 and c["honest_ok"] for c in ctls))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "free_rows.json").write_text(json.dumps({"ok": ok, "families": res, "controls": ctls}, indent=1) + "\n")
    print("no effective free row (both controls show one)" if ok else "EFFECTIVE FREE ROWS, or a control found none, or an honest check failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
