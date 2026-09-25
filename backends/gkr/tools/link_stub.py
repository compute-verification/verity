"""The A-GKR side of a binary-field hash link (hash-proving survey §3.8, §4.3 (a)) as a stub circuit: per unit, the
operand words it already holds plus their bit decomposition in k-bit limbs, which a link would open against the
binary-field proof's committed bits.  Only the prime-field cost is modelled; the cross-field check itself is not built.

Per unit (default: the BF16 VU unit, 32 operand words of 16 bits = 512 bits):

  columns   one, the words, and 512 / k limbs
  asserts   ∏_{j<2^k} (v - j) = 0 per limb (a product tree), and word - Σ_i limb_i 2^{k i} = 0 per word

    python tools/link_stub.py --out DIR --units B [--k K] [--words 32] [--bits 16] [--seed S]

writes ``DIR/circuit.txt``, ``DIR/witness.bin`` (B rows), ``DIR/neg.bin`` (row 0 with limb 0 of word 0 out of range)
and ``DIR/manifest.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sha256_flat import Builder, P, konst, write_witness


def build(words: int, bits: int, k: int) -> tuple[Builder, dict]:
    if bits % k:
        raise SystemExit(f"link_stub: k = {k} does not divide {bits}")
    b = Builder()
    b.col("one")
    ws = [b.col(f"x{i}") for i in range(words)]
    for i, w in enumerate(ws):
        limbs = [b.col(f"x{i}.{j}") for j in range(bits // k)]
        acc = konst(0)
        for j, v in enumerate(limbs):
            fs = [v - konst(t) for t in range(1 << k)]
            while len(fs) > 1:
                fs = [b.mul(fs[m], fs[m + 1]) if m + 1 < len(fs) else fs[m] for m in range(0, len(fs), 2)]
            b.assert_zero(f"x{i}.{j}.range", fs[0])
            acc = acc + v.scale(1 << (k * j))
        b.assert_zero(f"x{i}.recompose", w - acc)
    depth = max(d for d, _, _ in b.prods)
    return b, {"ncols": len(b.cols), "limbs": words * bits // k, "operand_bits": words * bits, "k": k, "products": len(b.prods),
               "nwires": len(b.cols) + len(b.prods), "asserts": len(b.asserts), "max_depth": depth}


def extend_unit(text: str, cols: list[int], bits: int = 16) -> str:
    """``text`` (a unit circuit) with the bits of its columns ``cols`` committed in-unit: ``bits`` new input columns per
    column after every existing wire (inputs are the non-product wires in order, so nothing is renumbered), each with
    b (b - 1) = 0, and ``col - Σ_i 2^i b_i = 0``.  The witness gains the bits LSB first, column after column, at the end
    of each row (``unit_bits``)."""
    lines = text.splitlines()
    head = {t[0]: int(t[1]) for t in (l.split() for l in lines) if t and t[0] in ("ncols", "nwires")}
    nc, nw = head["ncols"], head["nwires"]
    prods = {int(l.split()[1]) for l in lines if l.startswith("prod ")}
    colwire, c = {}, 0
    for w in range(nw):
        if w not in prods:
            colwire[c] = w
            c += 1
    nb = len(cols) * bits
    out = [f"ncols {nc + nb}" if l.startswith("ncols ") else f"nwires {nw + 2 * nb}" if l.startswith("nwires ") else l for l in lines]
    for j, col in enumerate(cols):
        terms = [f"{colwire[col]} 1"]
        for i in range(bits):
            b = j * bits + i
            out.append(f"col {nc + b} link.b{col}.{i}")
            out.append(f"prod {nw + nb + b} 1 1 {nw + b} 1 0 1 {nw + b} 1 {P - 1}")
            out.append(f"assert link.bool{col}.{i} 1 {nw + nb + b} 1 0")
            terms.append(f"{nw + b} {(P - (1 << i)) % P}")
        out.append(f"assert link.rec{col} {len(terms)} {' '.join(terms)} 0")
    return "\n".join(out) + "\n"


def unit_bits(rows: np.ndarray, cols: list[int], bits: int = 16) -> np.ndarray:
    """The ``extend_unit`` witness columns of unit rows ``rows`` (values of ``cols`` below 2^bits)."""
    v = rows[:, cols].astype(np.uint64)
    return ((v[:, :, None] >> np.arange(bits, dtype=np.uint64)) & 1).reshape(rows.shape[0], -1)


def witness(units: int, words: int, bits: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 1 << bits, size=(units, words), dtype=np.uint64)
    cols = [np.ones((units, 1), dtype=np.uint64), x]
    for i in range(words):
        cols.append(np.stack([(x[:, i] >> np.uint64(k * j)) & np.uint64((1 << k) - 1) for j in range(bits // k)], 1))
    return np.concatenate(cols, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python tools/link_stub.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--units", type=int, default=64)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--words", type=int, default=32)
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    b, info = build(args.words, args.bits, args.k)
    text = b.text()
    (args.out / "circuit.txt").write_text(text)
    rows = witness(args.units, args.words, args.bits, args.k, args.seed)
    if rows.shape[1] != info["ncols"]:
        raise SystemExit(f"link_stub: {rows.shape[1]} witness columns, the circuit has {info['ncols']}")
    write_witness(args.out / "witness.bin", rows)
    neg = rows[:1].copy()
    col = b.cols.index("x0.0")
    neg[0, col] = (neg[0, col] + (1 << args.k)) % P
    neg[0, b.cols.index("x0")] = (neg[0, b.cols.index("x0")] + (1 << args.k)) % P
    write_witness(args.out / "neg.bin", neg)
    info.update({"units": args.units, "words": args.words, "bits": args.bits, "seed": args.seed,
                 "circuit_sha256": hashlib.sha256(text.encode()).hexdigest(),
                 "neg": f"row 0, limb x0.0 raised by 2^k with x0 kept consistent (range check only)"})
    (args.out / "manifest.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
