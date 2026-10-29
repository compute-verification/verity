"""Which registers of a SHA-512 program carry or encode a committed 32-bit word, with the program written in 32-bit ops.

The question is whether hashing a committed row forces each committed word into a register of its own, one the hash's
own registers can't stand in for. Each variant hashes `PREFIX ‖ row` (a constant 128-byte prefix block, so a
constant midstate, then `BLOCKS` data blocks of 32 committed words and the padding block) as a straight-line program
of 32-bit register writes: a committed word as the `wgmma` that writes it (its accumulator `init` any earlier
register), a 64-bit add as two half adds with the carry in a flag, a rotation or a shift as one funnel shift a half,
`Ch`, `Maj` and a three-way XOR as one LOP3 a half, a byte swap as one PRMT. The digest is checked against `hashlib`
on every row.

Over `ROWS` random rows (the grid), a register is a *carrier* of word `w` when `reg − w` is constant mod 2³², and an
*encoder* when it is a function of `w` alone (unchanged when every other word is redrawn) and injective in `w`.
Registers constant over the grid are free and skipped; the counts give, a variant and a block, the registers written
and those that move over the grid (*live*), with the live registers distinct on the grid. Only a live register can be
charged, so a write that isn't live (the prefix block's, the padding block's schedule) is honest's to fold.

* `le`: frame-v3's row order. Each word is written little-endian, so SHA-512 reads `bswap32(w)`.
* `be`: the same with each word written big-endian, so SHA-512 reads `w` itself.
* `be-fold`: `be`, with every data block's round-0 message word `M_0` folded: the `wgmma`s write `w + P`, with `P`
  round 0's sum `h + Σ1(e) + Ch(e, f, g) + K_0`, so its outputs are round 0's `T1` halves (the high half after one
  carry add), and `W_16` is `(σ1(W_14) + W_9 + σ0(W_1) − P) + T1`. `M_0` is never written.

It also checks the byte-swap facts the Lean states (`bswap32_shift_not_const`, `bswap32_not_carrier`) on random
words, and which of `M_0..M_15` each use of the schedule and the rounds reads.

Exit 0 iff every check passes. `--json OUT` writes the counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys

import numpy as np

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
ROWS = 48
BLOCKS = 2
PREFIX = bytes(range(128))

K = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538,
    0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe,
    0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2, 0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235,
    0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5, 0x983e5152ee66dfab,
    0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
    0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df, 0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218,
    0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8, 0x19a4c116b8d2d0c8, 0x1e376c085141ab53,
    0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b, 0xca273eceea26619c,
    0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6,
    0x113f9804bef90dae, 0x1b710b35131c471b, 0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
]
IV = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
]


def bswap32(x: int) -> int:
    return int.from_bytes(x.to_bytes(4, "little"), "big")


class Prog:
    """A straight-line program of 32-bit register writes; a 64-bit value is a pair (hi, lo) of register values."""

    def __init__(self) -> None:
        self.regs: list[int] = []
        self.labels: list[str] = []

    def w(self, v: int, label: str) -> int:
        self.regs.append(v & M32)
        self.labels.append(label)
        return v & M32

    def add(self, a, b, label):
        lo = a[1] + b[1]
        return (self.w(a[0] + b[0] + (lo >> 32), label + ".hi"), self.w(lo, label + ".lo"))

    def sub(self, a, b, label):
        lo = a[1] - b[1]
        return (self.w(a[0] - b[0] - (1 if lo < 0 else 0), label + ".hi"), self.w(lo, label + ".lo"))

    def rotr(self, x, n, label):
        v = (x[0] << 32) | x[1]
        r = ((v >> n) | (v << (64 - n))) & M64
        return (self.w(r >> 32, label + ".hi"), self.w(r, label + ".lo"))

    def shr(self, x, n, label):
        v = ((x[0] << 32) | x[1]) >> n
        return (self.w(v >> 32, label + ".hi"), self.w(v, label + ".lo"))

    def lop3(self, f, a, b, c, label):
        return (self.w(f(a[0], b[0], c[0]), label + ".hi"), self.w(f(a[1], b[1], c[1]), label + ".lo"))

    def xor3(self, a, b, c, label):
        return self.lop3(lambda x, y, z: x ^ y ^ z, a, b, c, label)


def const(v: int):
    return (v >> 32, v & M32)


def big(x) -> int:
    return (x[0] << 32) | x[1]


def compress(p: Prog, H, Wm, tag: str, fold: bool):
    """One compression on the chaining value `H` (pairs) and message words `Wm`: 16 entries, each a pair, or for a
    folded `M_0` the tuple ("fold", w_hi, w_lo) of the raw committed words the `wgmma`s take."""
    a, b, c, d, e, f, g, h = H
    S1 = p.xor3(p.rotr(e, 14, f"{tag}r0.S1a"), p.rotr(e, 18, f"{tag}r0.S1b"), p.rotr(e, 41, f"{tag}r0.S1c"), f"{tag}r0.S1")
    ch = p.lop3(lambda x, y, z: (x & y) ^ (~x & z), e, f, g, f"{tag}r0.ch")
    P = p.add(p.add(p.add(h, S1, f"{tag}r0.p1"), ch, f"{tag}r0.p2"), const(K[0]), f"{tag}r0.P")
    W = list(Wm)
    if fold:
        _, whi, wlo = W[0]
        lo = p.w(wlo + P[1], f"{tag}r0.T1.lo=wgmma(w_lo,init=P.lo)")
        hip = p.w(whi + P[0], f"{tag}r0.T1.hi'=wgmma(w_hi,init=P.hi)")
        T1 = (p.w(hip + (1 if lo < P[1] else 0), f"{tag}r0.T1.hi"), lo)
    else:
        T1 = p.add(P, W[0], f"{tag}r0.T1")
    for t in range(16, 80):
        s0 = p.xor3(p.rotr(W[t - 15], 1, f"{tag}s{t}.s0a"), p.rotr(W[t - 15], 8, f"{tag}s{t}.s0b"),
                    p.shr(W[t - 15], 7, f"{tag}s{t}.s0c"), f"{tag}s{t}.s0")
        s1 = p.xor3(p.rotr(W[t - 2], 19, f"{tag}s{t}.s1a"), p.rotr(W[t - 2], 61, f"{tag}s{t}.s1b"),
                    p.shr(W[t - 2], 6, f"{tag}s{t}.s1c"), f"{tag}s{t}.s1")
        X = p.add(p.add(s1, W[t - 7], f"{tag}s{t}.x1"), s0, f"{tag}s{t}.x2")
        if fold and t == 16:
            W.append(p.add(p.sub(X, P, f"{tag}s16.xp"), T1, f"{tag}W16"))
        else:
            W.append(p.add(X, W[t - 16], f"{tag}W{t}"))
    for t in range(80):
        if t > 0:
            S1 = p.xor3(p.rotr(e, 14, f"{tag}r{t}.S1a"), p.rotr(e, 18, f"{tag}r{t}.S1b"), p.rotr(e, 41, f"{tag}r{t}.S1c"),
                        f"{tag}r{t}.S1")
            ch = p.lop3(lambda x, y, z: (x & y) ^ (~x & z), e, f, g, f"{tag}r{t}.ch")
            P = p.add(p.add(p.add(h, S1, f"{tag}r{t}.p1"), ch, f"{tag}r{t}.p2"), const(K[t]), f"{tag}r{t}.P")
            T1 = p.add(P, W[t], f"{tag}r{t}.T1")
        S0 = p.xor3(p.rotr(a, 28, f"{tag}r{t}.S0a"), p.rotr(a, 34, f"{tag}r{t}.S0b"), p.rotr(a, 39, f"{tag}r{t}.S0c"),
                    f"{tag}r{t}.S0")
        mj = p.lop3(lambda x, y, z: (x & y) ^ (x & z) ^ (y & z), a, b, c, f"{tag}r{t}.maj")
        T2 = p.add(S0, mj, f"{tag}r{t}.T2")
        h, g, f = g, f, e
        e = p.add(d, T1, f"{tag}r{t}.e")
        d, c, b = c, b, a
        a = p.add(T1, T2, f"{tag}r{t}.a")
    return [p.add(x, y, f"{tag}H{i}") for i, (x, y) in enumerate(zip(H, [a, b, c, d, e, f, g, h]))]


def run(words: list[int], variant: str) -> tuple[Prog, bytes, bytes]:
    """Hash PREFIX ‖ row for one row of committed words; return the program, its digest and hashlib's."""
    order = "little" if variant == "le" else "big"
    row = b"".join(x.to_bytes(4, order) for x in words)
    msg = PREFIX + row
    pad = msg + b"\x80" + b"\x00" * ((112 - (len(msg) + 1) % 128) % 128) + (8 * len(msg)).to_bytes(16, "big")
    blocks = [pad[i:i + 128] for i in range(0, len(pad), 128)]
    p = Prog()
    H = [const(v) for v in IV]
    for bi, blk in enumerate(blocks):
        data = 1 <= bi <= BLOCKS
        Wm = []
        for t in range(16):
            if data:
                wi = (bi - 1) * 32 + 2 * t
                w_hi, w_lo = words[wi], words[wi + 1]
                if variant == "be-fold" and t == 0:
                    Wm.append(("fold", w_hi, w_lo))
                    continue
                hi = p.w(w_hi, f"b{bi}.w{wi}=wgmma")
                lo = p.w(w_lo, f"b{bi}.w{wi + 1}=wgmma")
                if variant == "le":
                    hi = p.w(bswap32(hi), f"b{bi}.M{t}.hi=prmt(w{wi})")
                    lo = p.w(bswap32(lo), f"b{bi}.M{t}.lo=prmt(w{wi + 1})")
                Wm.append((hi, lo))
            else:
                v = int.from_bytes(blk[8 * t:8 * t + 8], "big")
                Wm.append(const(v))
        H = compress(p, H, Wm, f"b{bi}.", fold=data and variant == "be-fold")
    digest = b"".join(big(x).to_bytes(8, "big") for x in H)
    return p, digest, hashlib.sha512(msg).digest()


def regs(rows: list[list[int]], variant: str) -> np.ndarray:
    return np.array([run(r, variant)[0].regs for r in rows], dtype=np.uint64)


def analyse(variant: str, rng: random.Random) -> dict:
    """Carriers and encoders of every committed word. The grid is `ROWS` random rows and three boundary rows (all
    words 0, all 2³² − 1, alternating), so a carry that random rows rarely see is seen. An encoder is unchanged when
    every other word is redrawn, and changes when any one bit of its word flips (on two rows), so a shift that drops
    bits or a select between constants that agree on a bit is no encoder."""
    nw = 32 * BLOCKS
    base = [[0] * nw, [M32] * nw, [0x5555AAAA if i % 2 else 0xAAAA5555 for i in range(nw)]]
    base += [[rng.getrandbits(32) for _ in range(nw)] for _ in range(ROWS)]
    progs = [run(r, variant) for r in base]
    ok = all(d == ref for _, d, ref in progs)
    labels = progs[0][0].labels
    V = np.array([p.regs for p, _, _ in progs], dtype=np.uint64)
    Wv = np.array(base, dtype=np.uint64)
    live = ~np.all(V == V[0], axis=0)
    carriers: dict[int, list[str]] = {}
    for j in range(nw):
        diff = (V - Wv[:, j:j + 1]) & M32
        hit = live & np.all(diff == diff[0], axis=0)
        carriers[j] = [labels[i] for i in np.nonzero(hit)[0]]
    encoders: dict[int, list[str]] = {}
    flip_base = base[3:5]
    Vf0 = V[3:5]
    for j in range(nw):
        pert = []
        for r in base:
            q = [rng.getrandbits(32) for _ in range(nw)]
            q[j] = r[j]
            pert.append(q)
        same = np.all(regs(pert, variant) == V, axis=0)
        flips = [r[:j] + [r[j] ^ (1 << b)] + r[j + 1:] for r in flip_base for b in range(32)]
        Vf = regs(flips, variant).reshape(len(flip_base), 32, -1)
        moves = np.all(Vf != Vf0[:, None, :], axis=(0, 1))
        hit = live & same & moves
        encoders[j] = [labels[i] for i in np.nonzero(hit)[0]]
    blk = [L.split(".", 1)[0] for L in labels]
    per_block = {}
    for b in sorted(set(blk), key=lambda s: int(s[1:])):
        idx = np.array([i for i, x in enumerate(blk) if x == b])
        per_block[b] = {"writes": len(idx), "live": int(live[idx].sum())}
    distinct = len({V[:, i].tobytes() for i in np.nonzero(live)[0]})
    return {"variant": variant, "digest_ok": ok, "registers": len(labels), "live": int(live.sum()),
            "distinct_live": distinct, "per_block": per_block, "carriers": carriers, "encoders": encoders}


def uses() -> dict[int, list[str]]:
    """Which uses read each message word M_0..M_15: the round's add, sigma0 (W_{t-15}), sigma1 (W_{t-2}), and the
    schedule's adds W_{t-7} and W_{t-16}."""
    u: dict[int, set[str]] = {t: {"round"} for t in range(16)}
    for t in range(16, 80):
        for src, kind in ((t - 15, "sigma0"), (t - 2, "sigma1"), (t - 7, "add7"), (t - 16, "add16")):
            if src < 16:
                u[src].add(kind)
    return {t: sorted(v) for t, v in u.items()}


def bswap_checks(rng: random.Random) -> dict:
    bad_shift = 0
    for _ in range(200000):
        s = rng.randrange(1, 1 << 32)
        k0 = (bswap32(s) - bswap32(0)) & M32
        k1 = (bswap32((0xFFFFFFFF + s) & M32) - bswap32(0xFFFFFFFF)) & M32
        bad_shift += k0 == k1
    words = [rng.getrandbits(32) for _ in range(1000)]
    carrier = len({(bswap32(w) - w) & M32 for w in words}) == 1
    return {"shift_survives_swap_up_to_constant": bad_shift, "swap_is_carrier": carrier,
            "swap_involution": all(bswap32(bswap32(w)) == w for w in words)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", metavar="OUT")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = {"rows": ROWS, "blocks": BLOCKS, "uses": uses(), "bswap": bswap_checks(rng)}
    checks = []
    u = out["uses"]
    checks.append(("M_0 is read only additively", u[0] == ["add16", "round"]))
    checks.append(("M_1..M_15 are read by sigma0", all("sigma0" in u[t] for t in range(1, 16))))
    checks.append(("a carrier never survives the swap up to a constant", out["bswap"]["shift_survives_swap_up_to_constant"] == 0))
    checks.append(("the swapped word is no carrier", not out["bswap"]["swap_is_carrier"]))
    checks.append(("the swap is an involution", out["bswap"]["swap_involution"]))
    res = {}
    for v in ("le", "be", "be-fold"):
        r = analyse(v, rng)
        res[v] = r
        checks.append((f"{v}: digest equals hashlib on every row", r["digest_ok"]))
    nw = 32 * BLOCKS
    m0 = [w for b in range(BLOCKS) for w in (32 * b, 32 * b + 1)]
    for v in ("le", "be"):
        checks.append((f"{v}: every word is carried by its own wgmma register",
                       all(any("=wgmma" in L for L in res[v]["carriers"][j]) for j in range(nw))))
    checks.append(("le: no SHA-512 register carries a word",
                   all(all("=wgmma" in L for L in res["le"]["carriers"][j]) for j in range(nw))))
    checks.append(("le: the PRMT of each word encodes it",
                   all(any("prmt" in L for L in res["le"]["encoders"][j]) for j in range(nw))))
    first_lo = [1]
    checks.append(("be: in the first data block, round 0's lo registers carry M_0's low word",
                   all(len([L for L in res["be"]["carriers"][j] if "=wgmma" not in L]) >= 3 for j in first_lo)))
    checks.append(("be: M_0's high word has no SHA-512 carrier (the low half's carry moves)",
                   all("=wgmma" in L for L in res["be"]["carriers"][0])))
    checks.append(("be: only M_0's low word of the first data block has SHA-512 carriers",
                   all(all("=wgmma" in L for L in res["be"]["carriers"][j]) for j in range(nw) if j != 1)))
    checks.append(("le: in the first data block, round 0's lo registers encode M_0's low word but carry nothing",
                   all(len([L for L in res["le"]["encoders"][j] if "=wgmma" not in L and "prmt" not in L]) >= 3
                       for j in first_lo)))
    later = [w for w in m0 if w >= 32]
    checks.append(("be-fold: M_0's words of a later block have no carrier and no encoder",
                   all(not res["be-fold"]["carriers"][j] and not res["be-fold"]["encoders"][j] for j in later)))
    checks.append(("be-fold: every other word keeps its wgmma register",
                   all(any("=wgmma" in L for L in res["be-fold"]["carriers"][j]) for j in range(nw) if j not in m0)))
    checks.append(("the prefix block's registers are constant on the grid",
                   all(res[v]["per_block"]["b0"]["live"] == 0 for v in res)))
    out["writes"] = {v: res[v]["registers"] for v in res}
    out["live"] = {v: res[v]["live"] for v in res}
    out["distinct_live"] = {v: res[v]["distinct_live"] for v in res}
    out["per_block"] = {v: res[v]["per_block"] for v in res}
    out["carriers"] = {v: {j: res[v]["carriers"][j] for j in sorted(set(m0) | {2, 33})} for v in res}
    out["encoders"] = {v: {j: res[v]["encoders"][j] for j in sorted(set(m0) | {2, 33})} for v in res}
    out["sha_carriers"] = {v: sum(len([L for L in res[v]["carriers"][j] if "=wgmma" not in L]) for j in range(nw))
                           for v in res}
    out["sha_encoders"] = {v: sum(len([L for L in res[v]["encoders"][j] if "=wgmma" not in L and "prmt" not in L])
                                  for j in range(nw)) for v in res}
    out["checks"] = [{"check": n, "ok": bool(ok)} for n, ok in checks]
    for n, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {n}")
    print(json.dumps({k: out[k] for k in ("writes", "live", "distinct_live", "per_block", "sha_carriers",
                                          "sha_encoders")}, indent=1))
    for v in res:
        for j in sorted(set(m0) | {2, 33}):
            print(f"{v:8s} word {j:2d}: carriers {res[v]['carriers'][j]}  encoders {res[v]['encoders'][j]}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
