"""One SHA-256 compression as an A-GKR circuit in Longfellow's flat layout (``lib/circuits/sha/flatsha256_circuit.h``,
paper §4.2), over BabyBear: a measurement spike for the in-field route of the hash-proving survey (§3.3, §4.3 (b)).

Committed per compression (one row = one compression, data-parallel over rows):

  the chaining input H (8 words) and the message block W0..W15 (16 words), and every addition output: W16..W63,
  each round's new a and new e, and the output H' = H + final state (48 + 128 + 8 = 184 words, 5,888 bits),

all as single bits (no packing).  XOR, rotations, Σ, σ, Ch and Maj are products of committed bits inside the GKR
layers; nothing about them is committed.  A t-term addition modulo 2^32 whose committed output is r is checked in
16-bit halves, because BabyBear's p < 2^32 cannot hold Longfellow's single-check ∏_i (s - r - 2^32 i):

  d_lo = Σ_terms lo16 - r_lo,  ∏_{j<t} (d_lo - 2^16 j) = 0;   c = d_lo / 2^16;
  d_hi = Σ_terms hi16 + c - r_hi,  ∏_{j<t} (d_hi - 2^16 j) = 0

(every quantity is below t·2^16 + t < p, so the products vanish exactly when the integer identity holds).  Every
committed bit gets b (b - 1) = 0.

    python tools/sha256_flat.py --out DIR --blocks B [--seed S]

writes ``DIR/circuit.txt`` (``circuit.rs`` text), ``DIR/witness.bin`` (B rows, ``circuit.rs::read_witness``), ``DIR/neg.bin``
(row 0 with one committed bit of new a flipped) and ``DIR/manifest.json`` (the counts).  Each row hashes a random
55-byte message from the standard IV and is checked against ``hashlib``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

P = 2**31 - 2**27 + 1
INV16 = pow(1 << 16, P - 2, P)
IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01,
    0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA, 0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147,
    0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070, 0x19A4C116, 0x1E376C08,
    0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]


class Lin:
    """A linear form over wires, coefficients mod P."""

    __slots__ = ("t", "k")

    def __init__(self, t: dict | None = None, k: int = 0):
        self.t = t or {}
        self.k = k % P

    def __add__(self, o: "Lin") -> "Lin":
        t = dict(self.t)
        for w, c in o.t.items():
            v = (t.get(w, 0) + c) % P
            if v:
                t[w] = v
            else:
                t.pop(w, None)
        return Lin(t, self.k + o.k)

    def scale(self, c: int) -> "Lin":
        c %= P
        return Lin({w: v * c % P for w, v in self.t.items()} if c else {}, self.k * c)

    def __sub__(self, o: "Lin") -> "Lin":
        return self + o.scale(-1)

    def const(self) -> bool:
        return not self.t

    def text(self) -> str:
        return " ".join([str(len(self.t))] + [f"{w} {c}" for w, c in sorted(self.t.items())] + [str(self.k)])


def konst(v: int) -> Lin:
    return Lin({}, v)


class Builder:
    def __init__(self) -> None:
        self.cols: list[str] = []
        self.prods: list[tuple[int, Lin, Lin]] = []    # (depth, a, b) in wire order after the inputs
        self.asserts: list[tuple[str, Lin]] = []

    def col(self, name: str) -> Lin:
        self.cols.append(name)
        return Lin({("c", len(self.cols) - 1): 1})

    def depth(self, l: Lin) -> int:
        return max((self.prods[w[1]][0] if w[0] == "p" else 0 for w in l.t), default=0)

    def mul(self, a: Lin, b: Lin) -> Lin:
        if a.const():
            return b.scale(a.k)
        if b.const():
            return a.scale(b.k)
        self.prods.append((1 + max(self.depth(a), self.depth(b)), a, b))
        return Lin({("p", len(self.prods) - 1): 1})

    def assert_zero(self, name: str, l: Lin) -> None:
        self.asserts.append((name, l))

    def text(self) -> str:
        nc = len(self.cols)
        idx = lambda w: w[1] if w[0] == "c" else nc + w[1]     # inputs are wires 0..nc-1, products after them
        relabel = lambda l: Lin({idx(w): c for w, c in l.t.items()}, l.k)
        out = [f"ncols {nc}", f"nwires {nc + len(self.prods)}", f"one {self.cols.index('one')}"]
        out += [f"col {i} {n}" for i, n in enumerate(self.cols)]
        out += [f"prod {nc + i} {d} {relabel(a).text()} {relabel(b).text()}" for i, (d, a, b) in enumerate(self.prods)]
        out += [f"assert {n} {relabel(l).text()}" for n, l in self.asserts]
        return "\n".join(out) + "\n"


def xor(b: Builder, x: Lin, y: Lin) -> Lin:
    return x + y - b.mul(x, y).scale(2)


def xor3(b: Builder, x: Lin, y: Lin, z: Lin) -> Lin:
    return xor(b, xor(b, x, y), z)


Word = list  # 32 bit-Lins, LSB first


def rotr(w: Word, n: int) -> Word:
    return [w[(i + n) % 32] for i in range(32)]


def shr(w: Word, n: int) -> Word:
    return [w[i + n] if i + n < 32 else konst(0) for i in range(32)]


def big_sigma(b: Builder, w: Word, r: tuple[int, int, int]) -> Word:
    x, y, z = rotr(w, r[0]), rotr(w, r[1]), rotr(w, r[2])
    return [xor3(b, x[i], y[i], z[i]) for i in range(32)]


def small_sigma(b: Builder, w: Word, r: tuple[int, int], s: int) -> Word:
    x, y, z = rotr(w, r[0]), rotr(w, r[1]), shr(w, s)
    return [xor3(b, x[i], y[i], z[i]) if not z[i].const() else xor(b, x[i], y[i]) for i in range(32)]


def ch(b: Builder, e: Word, f: Word, g: Word) -> Word:
    return [b.mul(e[i], f[i] - g[i]) + g[i] for i in range(32)]


def maj(b: Builder, x: Word, y: Word, z: Word) -> Word:
    return [b.mul(x[i], y[i]) + b.mul(z[i], xor(b, x[i], y[i])) for i in range(32)]


def const_word(v: int) -> Word:
    return [konst((v >> i) & 1) for i in range(32)]


def half(w: Word, hi: bool) -> Lin:
    acc = konst(0)
    for i in range(16):
        acc = acc + w[16 * hi + i].scale(1 << i)
    return acc


def vanish(b: Builder, name: str, d: Lin, t: int) -> None:
    """∏_{j<t} (d - 2^16 j) = 0 as a product tree."""
    fs = [d - konst(j << 16) for j in range(t)]
    while len(fs) > 1:
        fs = [b.mul(fs[i], fs[i + 1]) if i + 1 < len(fs) else fs[i] for i in range(0, len(fs), 2)]
    b.assert_zero(name, fs[0])


def add_check(b: Builder, name: str, terms: list[Word], r: Word) -> None:
    t = len(terms)
    d_lo = konst(0)
    for w in terms:
        d_lo = d_lo + half(w, False)
    d_lo = d_lo - half(r, False)
    vanish(b, f"{name}.lo", d_lo, t)
    d_hi = d_lo.scale(INV16)
    for w in terms:
        d_hi = d_hi + half(w, True)
    d_hi = d_hi - half(r, True)
    vanish(b, f"{name}.hi", d_hi, t)


def build() -> tuple[Builder, dict]:
    b = Builder()
    b.col("one")
    committed: list[Lin] = []

    def word(name: str) -> Word:
        w = [b.col(f"{name}.{i}") for i in range(32)]
        committed.extend(w)
        return w

    H = [word(f"h{i}") for i in range(8)]
    W = [word(f"w{t}") for t in range(16)]
    W += [word(f"w{t}") for t in range(16, 64)]
    A = {0: H[0], -1: H[1], -2: H[2], -3: H[3]}
    E = {0: H[4], -1: H[5], -2: H[6], -3: H[7]}
    for t in range(1, 65):
        A[t] = word(f"a{t}")
        E[t] = word(f"e{t}")
    OUT = [word(f"o{i}") for i in range(8)]
    for t in range(16, 64):
        add_check(b, f"w{t}", [small_sigma(b, W[t - 2], (17, 19), 10), W[t - 7], small_sigma(b, W[t - 15], (7, 18), 3), W[t - 16]], W[t])
    for t in range(64):
        a, bb, c, d = A[t], A[t - 1], A[t - 2], A[t - 3]
        e, f, g, h = E[t], E[t - 1], E[t - 2], E[t - 3]
        t1 = [h, big_sigma(b, e, (6, 11, 25)), ch(b, e, f, g), const_word(K[t]), W[t]]
        add_check(b, f"e{t + 1}", [d] + t1, E[t + 1])
        add_check(b, f"a{t + 1}", t1 + [big_sigma(b, a, (2, 13, 22)), maj(b, a, bb, c)], A[t + 1])
    final = [A[64], A[63], A[62], A[61], E[64], E[63], E[62], E[61]]
    for i in range(8):
        add_check(b, f"o{i}", [H[i], final[i]], OUT[i])
    for k, x in enumerate(committed):
        b.assert_zero(f"bit{k}", b.mul(x, x - konst(1)))
    depth = max(d for d, _, _ in b.prods)
    return b, {"ncols": len(b.cols), "committed_bits": len(committed), "products": len(b.prods), "nwires": len(b.cols) + len(b.prods),
               "asserts": len(b.asserts), "max_depth": depth}


# --- witness -----------------------------------------------------------------------------------------------------

def _rotr(x, n):
    return ((x >> np.uint32(n)) | (x << np.uint32(32 - n))) & np.uint32(0xFFFFFFFF)


def compress(h: np.ndarray, m: np.ndarray) -> dict:
    """``h`` (B, 8), ``m`` (B, 16) uint32 -> every committed word, (B, ·) uint32."""
    w = [m[:, t] for t in range(16)]
    for t in range(16, 64):
        s0 = _rotr(w[t - 15], 7) ^ _rotr(w[t - 15], 18) ^ (w[t - 15] >> np.uint32(3))
        s1 = _rotr(w[t - 2], 17) ^ _rotr(w[t - 2], 19) ^ (w[t - 2] >> np.uint32(10))
        w.append((w[t - 16].astype(np.uint64) + s0 + w[t - 7] + s1).astype(np.uint32))
    A = {0: h[:, 0], -1: h[:, 1], -2: h[:, 2], -3: h[:, 3]}
    E = {0: h[:, 4], -1: h[:, 5], -2: h[:, 6], -3: h[:, 7]}
    for t in range(64):
        a, b_, c, d, e, f, g, hh = A[t], A[t - 1], A[t - 2], A[t - 3], E[t], E[t - 1], E[t - 2], E[t - 3]
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        chv = (e & f) ^ (~e & g)
        t1 = hh.astype(np.uint64) + S1 + chv + np.uint64(K[t]) + w[t]
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        mj = (a & b_) ^ (a & c) ^ (b_ & c)
        E[t + 1] = (d.astype(np.uint64) + t1).astype(np.uint32)
        A[t + 1] = (t1 + S0 + mj).astype(np.uint32)
    fin = [A[64], A[63], A[62], A[61], E[64], E[63], E[62], E[61]]
    out = [(h[:, i].astype(np.uint64) + fin[i]).astype(np.uint32) for i in range(8)]
    return {"h": [h[:, i] for i in range(8)], "w": w, "a": [A[t] for t in range(1, 65)], "e": [E[t] for t in range(1, 65)], "o": out}


def witness(nblocks: int, seed: int, ncols: int) -> tuple[np.ndarray, list[bytes]]:
    rng = np.random.default_rng(seed)
    msgs = [rng.bytes(55) for _ in range(nblocks)]
    blocks = np.array([np.frombuffer(m + b"\x80" + struct.pack(">Q", 8 * 55), dtype=">u4") for m in msgs], dtype=np.uint32)
    h = np.tile(np.array(IV, dtype=np.uint32), (nblocks, 1))
    wd = compress(h, blocks)
    words = wd["h"] + wd["w"]
    for t in range(64):
        words += [wd["a"][t], wd["e"][t]]
    words += wd["o"]
    for r in range(nblocks):
        got = b"".join(int(wd["o"][i][r]).to_bytes(4, "big") for i in range(8))
        if got != hashlib.sha256(msgs[r]).digest():
            raise SystemExit(f"sha256_flat: row {r} is not hashlib's digest")
    ws = np.stack(words, 1)                                       # (B, 184 + 24)
    bits = ((ws[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1).reshape(nblocks, -1).astype(np.uint64)
    out = np.concatenate([np.ones((nblocks, 1), dtype=np.uint64), bits], 1)
    if out.shape[1] != ncols:
        raise SystemExit(f"sha256_flat: {out.shape[1]} witness columns, the circuit has {ncols}")
    return out, msgs


def write_witness(path: Path, rows: np.ndarray) -> None:
    n, m = rows.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", n, m))
        f.write(np.ascontiguousarray(rows, dtype="<u8").tobytes())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python tools/sha256_flat.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    b, info = build()
    text = b.text()
    (args.out / "circuit.txt").write_text(text)
    rows, _ = witness(args.blocks, args.seed, info["ncols"])
    write_witness(args.out / "witness.bin", rows)
    neg = rows[:1].copy()
    col = b.cols.index("a1.5")
    neg[0, col] ^= 1
    write_witness(args.out / "neg.bin", neg)
    info.update({"blocks": args.blocks, "seed": args.seed, "circuit_sha256": hashlib.sha256(text.encode()).hexdigest(),
                 "neg": f"row 0, column {col} (a1.5) flipped", "layout": "Longfellow flatsha256, 16-bit-half adder checks, k = 1"})
    (args.out / "manifest.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
