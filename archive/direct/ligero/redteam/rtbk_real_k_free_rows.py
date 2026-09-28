"""red-team-bligero-real-k: mutate-and-recompute determinism scan of the leaf gadgets at vLLM's reduction lengths.

The real-K statements (``relations._at_k``: ``<fold>-k<K>+blake3-xob`` / ``+sha256``) run the x4 column gadget over rows of
K = 2048 / 8192 words.  A BLAKE3 row of more than 3 chunks is framed by ``Blake3Leaf.for_row`` (one held CV slot per chunk,
a wider position counter, header ``n_chunks + 64 role``); the SHA-256 leaf keeps its gadget with the K-dependent midstate and
header.  For each (word bits, blocks per column, K): build the one-role column system at that row length, run the honest
program over one row, then for every computed row (``prod`` / ``bits`` / ``sels`` / ``xadd`` outputs) override its value at
one column (``1 - v`` and ``v + 1``), recompute every later op, and check every constraint at that column.  A row that
survives is free given the inputs: a second witness for the same operand words and carries.

Controls (each must show free rows): the leaf's arithmetic control (one addition's low-limb identity dropped), and two
sized-frame controls -- the position counter's bit decomposition dropped (the layout logic that picks chunk start / end,
chunk counter and hold slots) and one hold-slot update product's constraint dropped.  Also checked per shape: the honest
witness satisfies every constraint, the digest at the last column equals ``native`` and folds to the ``blake3`` package's
keyed digest (``hashlib`` SHA-256 of prefix || row for sha256), the header is the sized one, and ``for_row`` sizes as Rust's
``leaf::sized`` does (4 / 8 / 16 slots, the registry frame at <= 3 chunks).

    python -m archive.direct.ligero.redteam.rtbk_real_k_free_rows --leaf blake3-xob --shapes 16:2:2048,16:2:8192 --out DIR

Exit 0 iff no computed row is free at any shape, every honest check holds and every control shows free rows."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..compile import LigeroCtx, P
from ..leaf.blake3_test import _eval
from ..relations import FP8_ADA_RELATION


def _leaf(name: str):
    if name == "blake3-xob":
        from ..leaf.blake3_xob import Blake3XobLeaf
        return Blake3XobLeaf()
    if name == "blake3":
        from ..leaf.blake3 import Blake3Leaf
        return Blake3Leaf()
    from ..leaf.sha256 import Sha256Leaf
    return Sha256Leaf()


def _column(leaf_name: str, word_bits: int, nblk: int, K: int, seed: int, role: int = 1):
    """One column system of one role at row length ``K`` (the leaf sized as ``hashchain.compose`` sizes it) and the honest
    hints of one row over its ``K / k`` columns."""
    from ..leaf import registry
    leaf = registry.sized(_leaf(leaf_name), K, word_bits)
    ctx = LigeroCtx(FP8_ADA_RELATION.params)
    ctx.scope = "t"
    k = nblk * 8 * 64 // word_bits
    words = [ctx.hint(f"w[{i}]") for i in range(k)]
    carry = [ctx.hint(f"carry[{i}]") for i in range(leaf.carry_elems)]
    digest, carry_out = leaf.gadget(ctx, words, word_bits=word_bits, role=role, row_words=K, carry_in=carry)
    steps = K // k
    rows = np.random.default_rng(seed).integers(0, 1 << word_bits, size=(1, K), dtype=np.int64)
    car = leaf.carry_values(rows, word_bits=word_bits, role=role, steps=steps)
    hints = {f"w[{i}]": rows[0].reshape(steps, k)[:, i] for i in range(k)}
    hints |= {f"carry[{i}]": car[0, :, i] for i in range(leaf.carry_elems)}
    return leaf, ctx.sys, digest, carry_out, rows, steps, car, hints


def _run(sys, hints: dict, width: int, mut: dict[int, np.ndarray], alt: str) -> np.ndarray:
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
        elif op[0] == "xadd":
            from ..leaf.blake3_xob import run_xadd
            _, row0, lo, hi, olo, ohi = op
            W[row0:row0 + 32] = run_xadd(_eval(lo, W), _eval(hi, W), _eval(olo, W), _eval(ohi, W))
            for i in range(32):
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


def _reference(leaf_name: str, rows: np.ndarray, word_bits: int, role: int) -> bytes:
    """The leaf value from outside the backend: the ``blake3`` package's keyed digest, or core's SHA-256 row digest."""
    from ..leaf.blake3 import key_for, row_bytes
    if leaf_name.startswith("blake3"):
        import blake3
        return blake3.blake3(bytes(row_bytes(rows, word_bits)[0]), key=key_for(role)).digest()
    import hashlib

    from verity.ml.commitments.rowleaf import sha256_row_prefix
    return hashlib.sha256(sha256_row_prefix(role, word_bits, rows.shape[1]) + bytes(row_bytes(rows, word_bits)[0])).digest()


def _drop(sys, control: str, leaf_name: str) -> str:
    if control == "arith":
        suffix = ("a1.lo", "a2.lo", "c1.lo", "c2.lo") if leaf_name == "blake3-xob" else ("lo.decomp",)
        cands = [c.name for c in sys.linear if c.name.endswith(suffix)]
        name = cands[len(cands) // 2]
        sys.linear = [c for c in sys.linear if c.name != name]
        return name
    if control == "pos":
        name = next(c.name for c in sys.linear if c.name.endswith(".pos.decomp"))
        sys.linear = [c for c in sys.linear if c.name != name]
        return name
    if control == "hold":
        cands = [q.name for q in sys.quadratic if ".hold" in q.name]
        name = cands[len(cands) // 2]
        sys.quadratic = [q for q in sys.quadratic if q.name != name]
        return name
    raise ValueError(control)


def scan(leaf_name: str, word_bits: int, nblk: int, K: int, width: int, seed: int = 7, control: str | None = None) -> dict:
    t0 = time.perf_counter()
    role = 1
    leaf, sys, digest, carry_out, rows, steps, car, hints = _column(leaf_name, word_bits, nblk, K, seed + K + word_bits, role)
    W0 = _run(sys, hints, steps, {}, "flip")
    honest_ok = bool(_ok_columns(sys, W0).all())
    outs = np.stack([_eval(e, W0) for e in carry_out])
    chain_ok = bool(np.array_equal(outs[:, :-1], car[0, 1:, :].T))
    dg = np.array([int(_eval(e, W0)[steps - 1]) for e in digest])
    native = leaf.native(rows, word_bits=word_bits, role=role)[0]
    ref = _reference(leaf_name, rows, word_bits, role)
    digest_ok = bool(np.array_equal(dg, native) and leaf.leaf_bytes(dg) == ref)
    ncm = getattr(leaf, "n_chunks_max", None)
    if leaf_name.startswith("blake3"):
        from ..leaf.blake3 import header, n_chunks
        nch = n_chunks(word_bits, K)
        rust_ncm = nch if nch > 3 else 3                  # leaf::sized: BLAKE3_SIZED rows 4 / 8 / 16, the registry frame at <= 3
        frame_ok = bool(ncm == rust_ncm and dg[0] == header(nch, role, ncm) and leaf.digest_elems == 1 + 16 * ncm)
    else:
        from ..leaf.sha256 import header as sha_header
        frame_ok = bool(dg[0] == sha_header(role, word_bits, K))
    dropped = _drop(sys, control, leaf_name) if control else None
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
        elif op[0] == "xadd":
            computed += list(range(op[1], op[1] + 32))
    free: dict[int, dict] = {}
    runs = 0
    # column offsets: 0 and mid-row, plus the last column (the digest's) -- a sized frame's chunk ends and holds sit at
    # columns 8c + 7, so offsets 7 and steps - 1 put mutations on them
    offs = sorted({0, 7 % steps, steps // 2 + 1, steps - 1})
    for off in offs:
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
    return {"leaf": leaf_name, "word_bits": word_bits, "blocks_per_column": nblk, "K": K, "steps": steps, "n_chunks_max": ncm,
            "dropped": dropped, "rows": sys.m, "computed_rows": len(computed), "linear": len(sys.linear),
            "quadratic": len(sys.quadratic), "width": width, "offsets": offs, "program_runs": runs,
            "mutations": 2 * len(offs) * len(computed), "honest_ok": honest_ok, "chain_ok": chain_ok, "digest_ok": digest_ok,
            "frame_ok": frame_ok, "n_free": len(free), "free": list(free.values())[:20],
            "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leaf", default="blake3-xob", choices=("blake3", "blake3-xob", "sha256"))
    ap.add_argument("--shapes", default="16:2:2048,16:2:8192,8:2:2048,8:2:8192", help="word_bits:blocks_per_column:K, comma-separated")
    ap.add_argument("--controls", default="arith,pos,hold", help="controls to run at --control-shape (sha256: arith only)")
    ap.add_argument("--control-shape", default="16:2:2048")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    shape = lambda t: tuple(int(x) for x in t.split(":"))  # noqa: E731
    ctls = []
    for c in [c for c in a.controls.split(",") if c]:
        if a.leaf == "sha256" and c != "arith":
            continue
        r = scan(a.leaf, *shape(a.control_shape), a.width, control=c)
        print("control", c, json.dumps({k: v for k, v in r.items() if k != "free"}), json.dumps(r["free"][:2]), flush=True)
        ctls.append(r)
    res = []
    for s in a.shapes.split(","):
        r = scan(a.leaf, *shape(s), a.width)
        print(json.dumps({k: v for k, v in r.items() if k != "free"}), flush=True)
        for f in r["free"]:
            print("  FREE", json.dumps(f), flush=True)
        res.append(r)
    ok = (all(r["honest_ok"] and r["chain_ok"] and r["digest_ok"] and r["frame_ok"] and r["n_free"] == 0 for r in res)
          and all(c["n_free"] > 0 and c["honest_ok"] for c in ctls))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"free_rows_{a.leaf}.json").write_text(json.dumps({"ok": ok, "shapes": res, "controls": ctls}, indent=1) + "\n")
    print("no free row; every control shows free rows" if ok else "FREE ROWS FOUND, a control found none, or an honest check failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
