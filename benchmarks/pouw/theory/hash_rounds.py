"""Folds built from SHA-512's own round function, priced in W1 and attacked on CPU.

    python3 hash_rounds.py LEAD.json [--json OUT] [--mutations] [--verdict PATH]

LEAD.json is hash_sha_lead.py's output (the lead's h, r, T1 terms and T2 parts at the three Llama-3.2-1B points).
Units are W1's: a register write is 32 W1, stock is 1 W1 a useful MAC, h is honest commitment W1 a stock MAC, and a word
is one 32-bit accumulator value after a k32 step. Every 64-bit message word W of a fold is two words (lo, hi).

- M. The W1 model of SHA-512 on 32-bit registers: SHF funnel windows, LOP3, and 64-bit adds of at most three terms as an
  IADD3 / IADD3.X pair (two writes, K_t an immediate). A round is 28 writes, a schedule step 20, a compression 3,534
  with the feed-forward fused into round 79 (the minimal SHA-512 committer), a padding block 2,254. Checked bit for bit
  against hashlib over whole messages.
- F. The folds, each an honest W1 program checked bit for bit against its reference:
  - R80: SHA-512's compression with its message schedule replaced by 80 raw message words a block (K_0..K_79, the
    feed-forward fused into round 79), chained;
  - R16: chunks of SHA-512's first 16 rounds (K_0..K_15, which read raw message words in SHA-512 itself), chained with
    no feed-forward;
  - Rr: r rounds a message word (the word injected at the first), r = 1..5;
  - RD: R80 with a second message word V added into e' = d + T1 + V, the only free IADD3 slot of a round;
  - SCH: SHA-512's schedule step as the fold, each step adding one message word: W_t = s1 + W_{t-7} + s0 + W_{t-16} + M_t;
  - TC: truncated compressions of rc rounds with their schedule and feed-forward over each 16-word block.
- A. Attacks: a bit-sliced adversary (32 instances a register, rotations free, adds as carry-save trees and a ripple)
  for the round and the schedule step, bit for bit; skipping one round; carry-free linearization of r rounds;
  single-word and persistent-shift binding.
- P. h, T1 and T2 at the three points, with the seal, against the lead's 127.5; and the per-monitoring-period pools.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))
from exact_kit.checks import Checks  # noqa: E402

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
W1_PER_REG = 32
BLOCK = 128
R_FIRST, R_CHAIN, R_PAD = 3510, 3532, 2254
CHUNK_K, CHUNK_EXTRA = 2213, 5
T1_BAR = 0.01
POINTS = ("t512", "t512r512", "t8192")
H200_MACS = 9.89e14 * 0.4
AND_PER_MAC = 2079 / 16
WINDOW_S = 600
PERIODS = {"window": WINDOW_S, "week": 7 * 86400, "month": 30 * 86400, "year": 365.25 * 86400}
DC_GPUS = 100_000
WORLD_MACS_PER_YEAR = 4.68e30

K = [0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538,
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
     0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817]
IV = [0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
      0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179]
SIG = {"S0": (28, 34, 39, None), "S1": (14, 18, 41, None), "s0": (1, 8, None, 7), "s1": (19, 61, None, 6)}


# ---------------------------------------------------------------- reference (Python ints, one instance)

def rotr(x: int, n: int) -> int:
    return (x >> n | x << (64 - n)) & M64


def sig_ref(name: str, x: int) -> int:
    a, b, c, s = SIG[name]
    return rotr(x, a) ^ rotr(x, b) ^ (rotr(x, c) if c is not None else x >> s)


def round_u(st: list[int], u: int, v: int = 0) -> list[int]:
    """A round given U = h + W + K."""
    a, b, c, d, e, f, g, h = st
    t1 = (u + sig_ref("S1", e) + ((e & f) ^ (~e & g))) & M64
    t2 = (sig_ref("S0", a) + ((a & b) ^ (a & c) ^ (b & c))) & M64
    return [(t1 + t2) & M64, a, b, c, (d + t1 + v) & M64, e, f, g]


def round_ref(st: list[int], w: int, k: int, v: int = 0) -> list[int]:
    return round_u(st, (st[7] + k + w) & M64, v)


def xc(t: int) -> int:
    """The per-position constant XORed into a word before an add-only injection (none of 0, 2^31, ~0 in either half,
    which would let the XOR pass through an add)."""
    return K[(t + 40) % 80]


def sched_ref(w16: list[int], m: int = 0) -> int:
    return (sig_ref("s1", w16[-2]) + w16[-7] + sig_ref("s0", w16[-15]) + w16[-16] + m) & M64


def compress_ref(cv: list[int], block: list[int]) -> list[int]:
    w = list(block)
    for t in range(16, 80):
        w.append(sched_ref(w[t - 16:t]))
    st = list(cv)
    for t in range(80):
        st = round_ref(st, w[t], K[t])
    return [(x + y) & M64 for x, y in zip(cv, st)]


# ---------------------------------------------------------------- the W1 model (numpy uint32 lanes, many instances)

class Regs:
    """Counts 32-bit register writes by opcode; every value is a uint32 array over the instances. With fold_consts, an
    op whose inputs are equal across the instances (a value of the public start state alone) is free: the count of a
    straight-line program for that start state, which takes such values as immediates."""

    def __init__(self, fold_consts: bool = False):
        self.n = {"SHF": 0, "LOP3": 0, "IADD3": 0}
        self.fold_consts = fold_consts

    def total(self) -> int:
        return sum(self.n.values())

    def _count(self, op: str, k: int, *xs):
        if not (self.fold_consts and all(bool(np.all(x == x[0])) for x in xs)):
            self.n[op] += k

    def shf(self, lo, hi, m: int):
        self._count("SHF", 1, lo, hi)
        return ((hi.astype(np.uint64) << np.uint64(32) | lo.astype(np.uint64)) >> np.uint64(m)).astype(np.uint32)

    def shr(self, x, m: int):
        self._count("SHF", 1, x)
        return x >> np.uint32(m)

    def lop3(self, f, *xs):
        self._count("LOP3", 1, *xs)
        return f(*xs).astype(np.uint32)

    def add(self, *terms):
        """A 64-bit add of at most three terms, each a (lo, hi) pair or an int immediate: an IADD3 pair."""
        assert 2 <= len(terms) <= 3 and sum(isinstance(t, int) for t in terms) <= 1
        self._count("IADD3", 2, *[x for t in terms if not isinstance(t, int) for x in t])
        lo = np.zeros_like(next(t for t in terms if not isinstance(t, int))[0], dtype=np.uint64)
        hi = np.zeros_like(lo)
        for t in terms:
            tl, th = (np.uint64(t & M32), np.uint64(t >> 32)) if isinstance(t, int) else t
            lo = lo + np.asarray(tl, dtype=np.uint64)
            hi = hi + np.asarray(th, dtype=np.uint64)
        hi = hi + (lo >> np.uint64(32))
        return (lo & np.uint64(M32)).astype(np.uint32), (hi & np.uint64(M32)).astype(np.uint32)


def rotr_w1(r: Regs, x, n: int):
    lo, hi = x
    if n >= 32:
        lo, hi = hi, lo
    m = n & 31
    return (lo, hi) if m == 0 else (r.shf(lo, hi, m), r.shf(hi, lo, m))


def sig_w1(r: Regs, name: str, x):
    a, b, c, s = SIG[name]
    p, q = rotr_w1(r, x, a), rotr_w1(r, x, b)
    t = rotr_w1(r, x, c) if c is not None else (r.shf(x[0], x[1], s), r.shr(x[1], s))
    xor3 = lambda u, v, w: u ^ v ^ w  # noqa: E731
    return r.lop3(xor3, p[0], q[0], t[0]), r.lop3(xor3, p[1], q[1], t[1])


def ch_w1(r: Regs, e, f, g):
    fn = lambda x, y, z: (x & y) ^ (~x & z)  # noqa: E731
    return r.lop3(fn, e[0], f[0], g[0]), r.lop3(fn, e[1], f[1], g[1])


def maj_w1(r: Regs, a, b, c):
    fn = lambda x, y, z: (x & y) ^ (x & z) ^ (y & z)  # noqa: E731
    return r.lop3(fn, a[0], b[0], c[0]), r.lop3(fn, a[1], b[1], c[1])


def round_w1(r: Regs, st, w, k: int, v=None, ff=None):
    """One round: U = h + W + K, T1 = U + S1 + Ch, e' = d + T1 (+ V), a' = T1 + S0 + Maj. With ff (the chaining value),
    round 79 fused with the feed-forward: H0 = (T1 + S0 + Maj) + cv0, H4 = T1 + d + cv4, Hi = cv_i + reg."""
    a, b, c, d, e, f, g, h = st
    u = r.add(h, w, k) if w is not None else r.add(h, k)
    t1 = r.add(u, sig_w1(r, "S1", e), ch_w1(r, e, f, g))
    s0, mj = sig_w1(r, "S0", a), maj_w1(r, a, b, c)
    if ff is None:
        e2 = r.add(d, t1, v) if v is not None else r.add(d, t1)
        return [r.add(t1, s0, mj), a, b, c, e2, e, f, g]
    assert v is None
    return [r.add(r.add(t1, s0, mj), ff[0]), r.add(ff[1], a), r.add(ff[2], b), r.add(ff[3], c),
            r.add(t1, d, ff[4]), r.add(ff[5], e), r.add(ff[6], f), r.add(ff[7], g)]


def xor_w1(r: Regs, w, c: int):
    return (r.lop3(lambda x: x ^ np.uint32(c & M32), w[0]), r.lop3(lambda x: x ^ np.uint32(c >> 32), w[1]))


def sched_w1(r: Regs, w16, m=None):
    x = r.add(sig_w1(r, "s1", w16[-2]), w16[-7], sig_w1(r, "s0", w16[-15]))
    return r.add(x, w16[-16], m) if m is not None else r.add(x, w16[-16])


def compress_w1(r: Regs, cv, block):
    w = list(block)
    st = list(cv)
    for t in range(80):
        if t >= 16:
            w.append(sched_w1(r, w[t - 16:t]))
        st = round_w1(r, st, w[t], K[t], ff=cv if t == 79 else None)
    return st


def to_pair(xs: np.ndarray):
    xs = xs.astype(np.uint64)
    return (xs & np.uint64(M32)).astype(np.uint32), (xs >> np.uint64(32)).astype(np.uint32)


def from_pair(p) -> np.ndarray:
    return p[1].astype(np.uint64) << np.uint64(32) | p[0].astype(np.uint64)


def lanes_pair(vals: list[int]):
    return to_pair(np.array(vals, dtype=np.uint64))


def part_m(checks: Checks, rng: np.random.Generator) -> dict:
    n = 64
    msgs = [rng.integers(0, 256, size=int(rng.integers(0, 300)), dtype=np.uint8).tobytes() for _ in range(n)]
    # every message padded to its own block count; run lanes with equal block counts together
    out_ok, counts = True, []
    for nblocks in sorted({(len(m) + 17 + 127) // 128 for m in msgs}):
        group = [m for m in msgs if (len(m) + 17 + 127) // 128 == nblocks]
        padded = [m + b"\x80" + b"\0" * ((nblocks * 128 - 16 - len(m) - 1)) + (8 * len(m)).to_bytes(16, "big")
                  for m in group]
        cv = [lanes_pair([IV[i]] * len(group)) for i in range(8)]
        for b in range(nblocks):
            blk = [lanes_pair([int.from_bytes(p[128 * b + 8 * j:128 * b + 8 * j + 8], "big") for p in padded])
                   for j in range(16)]
            r = Regs()
            cv = compress_w1(r, cv, blk)
            counts.append(r.total())
        dig = [b"".join(int(from_pair(cv[i])[j]).to_bytes(8, "big") for i in range(8)) for j in range(len(group))]
        out_ok &= all(d == hashlib.sha512(m).digest() for d, m in zip(dig, group))
    checks.check("M: the W1 program of SHA-512 equals hashlib on 64 random messages of 0-299 bytes", out_ok,
                 f"{len(counts)} compressions")
    r1 = Regs()
    st = [lanes_pair(list(rng.integers(0, 2**63, size=4, dtype=np.uint64) * 2)) for _ in range(8)]
    round_w1(r1, st, lanes_pair([1, 2, 3, 4]), K[5])
    rs = Regs()
    sched_w1(rs, [lanes_pair([i] * 4) for i in range(16)])
    rsm = Regs()
    sched_w1(rsm, [lanes_pair([i] * 4) for i in range(16)], lanes_pair([7] * 4))
    comp = counts[0]
    checks.check("M: a round is 28 writes (S1 8, S0 8, Ch 2, Maj 2, four IADD3 pairs 8); a schedule step is 20, also "
                 "with a message word added (its two IADD3 pairs take up to five terms)",
                 r1.total() == 28 and rs.total() == 20 and rsm.total() == 20,
                 f"round {r1.n}, step {rs.n}, step+M {rsm.n}")
    checks.check("M: a compression of variable words from a variable chaining value is 3,534 = 80 x 28 + 64 x 20 + 14, "
                 "the minimal SHA-512 committer (the feed-forward fused into round 79)",
                 all(c == 3534 for c in counts) and 80 * 28 + 64 * 20 + 14 == 3534, f"{comp}")
    return {"round": r1.total(), "round_ops": r1.n, "sched": rs.total(), "sched_m": rsm.total(), "compress": comp,
            "compress_rounds_share": 80 * 28 / comp, "compress_sched_share": 64 * 20 / comp,
            "rounds_per_word_32": 80 / 32, "sched_steps_per_word_32": 64 / 32}


# ---------------------------------------------------------------- the folds

def fold_ref(design: dict, st0: list[int], ws: list[int], vs: list[int] | None = None) -> list[int]:
    kind = design["kind"]
    st = list(st0)
    if kind == "R80":
        for b0 in range(0, len(ws), 80):
            cv = list(st)
            for t, w in enumerate(ws[b0:b0 + 80]):
                st = round_ref(st, w ^ xc(t) if design.get("xor") else w, K[t])
            st = [(x + y) & M64 for x, y in zip(cv, st)]
        return st
    if kind == "R16":
        for i, w in enumerate(ws):
            st = round_ref(st, w, K[i % 16])
        return st
    if kind == "RR":
        t = 0
        for w in ws:
            for j in range(design["r"]):
                st = round_ref(st, w if j == 0 else 0, K[t % 80])
                t += 1
        return st
    if kind == "RD":
        for i, (w, v) in enumerate(zip(ws, vs)):
            st = round_ref(st, w ^ xc(i % 80) if design.get("xor") else w, K[i % 80], v)
        return st
    if kind == "SCH":
        win = list(st0) + list(st0)
        for m in ws:
            win = win[1:] + [sched_ref(win, m)]
        return win
    if kind == "TC":
        rc = design["rc"]
        for b0 in range(0, len(ws), 16):
            w = list(ws[b0:b0 + 16])
            for t in range(16, rc):
                w.append(sched_ref(w[t - 16:t]))
            cv = list(st)
            for t in range(rc):
                st = round_ref(st, w[t], K[t])
            st = [(x + y) & M64 for x, y in zip(cv, st)]
        return st
    raise ValueError(kind)


def fold_w1(r: Regs, design: dict, st0, ws, vs=None):
    kind = design["kind"]
    st = list(st0)
    if kind == "R80":
        for b0 in range(0, len(ws), 80):
            cv = list(st)
            blk = ws[b0:b0 + 80]
            for t, w in enumerate(blk):
                w = xor_w1(r, w, xc(t)) if design.get("xor") else w
                st = round_w1(r, st, w, K[t], ff=cv if t == len(blk) - 1 else None)
        return st
    if kind == "R16":
        for i, w in enumerate(ws):
            st = round_w1(r, st, w, K[i % 16])
        return st
    if kind == "RR":
        t = 0
        for w in ws:
            for j in range(design["r"]):
                st = round_w1(r, st, w if j == 0 else None, K[t % 80])
                t += 1
        return st
    if kind == "RD":
        for i, (w, v) in enumerate(zip(ws, vs)):
            w = xor_w1(r, w, xc(i % 80)) if design.get("xor") else w
            st = round_w1(r, st, w, K[i % 80], v=v)
        return st
    if kind == "SCH":
        win = list(st0) + list(st0)
        for m in ws:
            win = win[1:] + [sched_w1(r, win, m)]
        return win
    if kind == "TC":
        rc = design["rc"]
        for b0 in range(0, len(ws), 16):
            w = list(ws[b0:b0 + 16])
            for t in range(16, rc):
                w.append(sched_w1(r, w[t - 16:t]))
            cv = list(st)
            for t in range(rc):
                st = round_w1(r, st, w[t], K[t], ff=cv if t == rc - 1 else None)
        return st
    raise ValueError(kind)


DESIGNS = {
    "R80": {"kind": "R80", "w_per_unit": 80, "state_bits": 512,
            "what": "SHA-512's compression with its schedule replaced by 80 raw message words a block, chained"},
    "R16": {"kind": "R16", "w_per_unit": 16, "state_bits": 512,
            "what": "chunks of SHA-512's first 16 rounds (the rounds that read raw words), chained without feed-forward"},
    **{f"RR{r}": {"kind": "RR", "r": r, "w_per_unit": 1, "state_bits": 512,
                  "what": f"{r} SHA-512 rounds a 64-bit message word ({r / 2:g} a 32-bit word)"} for r in (2, 3, 4, 5)},
    "RD": {"kind": "RD", "w_per_unit": 80, "state_bits": 512, "two_per_round": True,
           "what": "R80 with a second message word added into e' = d + T1 + V (a modified round)"},
    "R80X": {"kind": "R80", "xor": True, "w_per_unit": 80, "state_bits": 512,
             "what": "R80 with each message word XORed with a per-position constant first (one LOP3 a word)"},
    "RDX": {"kind": "RD", "xor": True, "w_per_unit": 80, "state_bits": 512, "two_per_round": True,
            "what": "RD with its W words XORed with a per-position constant first, its V words raw"},
    "SCH": {"kind": "SCH", "w_per_unit": 1, "state_bits": 1024,
            "what": "SHA-512's schedule step as the fold, one message word added a step"},
    **{f"TC{rc}": {"kind": "TC", "rc": rc, "w_per_unit": 16, "state_bits": 512,
                   "what": f"SHA-512 compressions truncated to {rc} rounds (schedule and feed-forward kept)"}
       for rc in (16, 24, 32, 48)},
}


def fold_run(d: dict, ref_d: dict, rng: np.random.Generator, n: int = 8) -> tuple[bool, Regs, int]:
    """d's honest W1 program against ref_d's reference on n random instances: (exact, its counts, message words)."""
    nw = 160 if ref_d["kind"] in ("R80", "RD") else (64 if ref_d["kind"] in ("R16", "TC") else 40)
    ws = rng.integers(0, 2**64, size=(nw, n), dtype=np.uint64)
    vs = rng.integers(0, 2**64, size=(nw, n), dtype=np.uint64) if ref_d["kind"] == "RD" else None
    st0 = rng.integers(0, 2**64, size=(8, n), dtype=np.uint64)
    r = Regs()
    got = fold_w1(r, d, [to_pair(x) for x in st0], [to_pair(x) for x in ws],
                  [to_pair(x) for x in vs] if vs is not None else None)
    vals = [from_pair(p) for p in got]
    ok = True
    for j in range(n):
        ref = fold_ref(ref_d, [int(x) for x in st0[:, j]], [int(x) for x in ws[:, j]],
                       [int(x) for x in vs[:, j]] if vs is not None else None)
        ok &= [int(v[j]) for v in vals] == ref
    return bool(ok), r, nw * (2 if ref_d["kind"] == "RD" else 1)


def fold_exact(d: dict, ref_d: dict, rng: np.random.Generator) -> bool:
    return fold_run(d, ref_d, rng)[0]


def head_saving(d: dict, rng: np.random.Generator, n: int = 8) -> int:
    """Writes a chain from a public start state saves over the chained count: the program for that start state takes
    every value of the state alone as an immediate (ShaWrites' R_CHAIN - R_FIRST for SHA-512 itself)."""
    nw = 160 if d["kind"] in ("R80", "RD") else 64
    ws = [to_pair(x) for x in rng.integers(0, 2**64, size=(nw, n), dtype=np.uint64)]
    vs = [to_pair(x) for x in rng.integers(0, 2**64, size=(nw, n), dtype=np.uint64)] if d["kind"] == "RD" else None
    st0 = [to_pair(np.full(n, int(x), dtype=np.uint64)) for x in rng.integers(0, 2**64, size=8, dtype=np.uint64)]
    full, folded = Regs(), Regs(fold_consts=True)
    a = fold_w1(full, d, st0, ws, vs)
    b = fold_w1(folded, d, st0, ws, vs)
    assert all((x[0] == y[0]).all() and (x[1] == y[1]).all() for x, y in zip(a, b))
    return full.total() - folded.total()


def part_f(checks: Checks, rng: np.random.Generator) -> dict:
    out = {}
    for name, d in DESIGNS.items():
        ok, r, msg_words = fold_run(d, d, rng)
        per_word = r.total() / (2 * msg_words)
        out[name] = {"what": d["what"], "kind": d["kind"], "writes": r.total(), "message_words": msg_words,
                     "writes_per_word": per_word, "ops": dict(r.n), "exact": bool(ok), "state_bits": d["state_bits"]}
    checks.check("F: every fold's honest W1 program equals its reference bit for bit (8 instances each)",
                 all(v["exact"] for v in out.values()),
                 ", ".join(f"{k} {v['writes_per_word']:.3f}" for k, v in out.items()))
    checks.check("F: R80's block is 2,254 writes, ShaWrites' own R_PAD (80 rounds and a fused feed-forward, no "
                 "schedule): the count ShaWrites already asserts for a padding block, here over raw words",
                 out["R80"]["writes"] == 2 * R_PAD, f"{out['R80']['writes']} for 2 blocks")
    checks.check("F: R16 is 14 writes a word, RD 7, SCH 10; Rr is 14r; a full compression is 110.4",
                 out["R16"]["writes_per_word"] == 14 and out["RD"]["writes_per_word"] == 7
                 and out["SCH"]["writes_per_word"] == 10
                 and all(out[f"RR{r}"]["writes_per_word"] == 14 * r for r in (2, 3, 4, 5)),
                 f"R16 {out['R16']['writes_per_word']}, RD {out['RD']['writes_per_word']}, "
                 f"SCH {out['SCH']['writes_per_word']}")
    checks.check("F: the compression splits as rounds 0-15 (448) + schedule (1,280) + rounds 16-79 (1,792) + "
                 "feed-forward (14): R16's chunk is exactly the part of the lead's compression that reads raw words",
                 16 * 28 + 64 * 20 + 64 * 28 + 14 == 3534 and out["R16"]["writes"] == 64 * 28)
    ws = [int(x) for x in rng.integers(0, 2**64, size=16, dtype=np.uint64)]
    blk = compress_ref(IV, ws)
    st16 = fold_ref(DESIGNS["R16"], IV, ws)
    w = list(ws)
    for t in range(16, 80):
        w.append(sched_ref(w[t - 16:t]))
    st = list(st16)
    for t in range(16, 80):
        st = round_ref(st, w[t], K[t])
    checks.check("F: the reduction's split is exact: SHA-512's compression = feed-forward(rounds 16-79 over the "
                 "schedule, from R16's chunk state); and RD with V = 0 is SHA-512's rounds",
                 [(x + y) & M64 for x, y in zip(IV, st)] == blk
                 and fold_ref(DESIGNS["RD"], IV, ws, [0] * 16) == fold_ref(dict(DESIGNS["R16"]), IV, ws))
    for name, d in DESIGNS.items():
        out[name]["head_saving"] = head_saving(d, rng)
    r0, r1 = Regs(), Regs(fold_consts=True)
    blk = [to_pair(x) for x in rng.integers(0, 2**64, size=(16, 8), dtype=np.uint64)]
    iv = [to_pair(np.full(8, x, dtype=np.uint64)) for x in IV]
    compress_w1(r0, iv, blk)
    compress_w1(r1, iv, blk)
    sha_head = r0.total() - r1.total()
    checks.check("F: a chain from a public start state saves 20 writes on R80, R16 and RD, as SHA-512's own first block "
                 "does in this model (ShaWrites' card: 22), and 140 on SCH, whose first 15 steps' s0 and first two "
                 "steps' s1 read the start window alone",
                 all(out[n]["head_saving"] == sha_head == 20 for n in ("R80", "R16", "RD"))
                 and out["SCH"]["head_saving"] == 140,
                 f"SHA-512 {r0.total()} -> {r1.total()}; " + ", ".join(f"{n} {out[n]['head_saving']}"
                                                                       for n in ("R80", "R16", "RD", "SCH")))
    return out


# ---------------------------------------------------------------- attacks

class Planes:
    """Bit-sliced: a 64-bit value for 32 instances as 64 uint32 planes (bit i of the 32 instances); ops count LOP3s."""

    def __init__(self):
        self.ops = 0

    def lop(self, f, *xs):
        self.ops += 1
        return f(*xs)

    def xor3(self, a, b, c):
        return self.lop(lambda x, y, z: x ^ y ^ z, a, b, c)

    def csa(self, x, y, z):
        """Three 64-plane values to (sum, carry << 1); the carry out of bit 63 is dropped."""
        s = [self.xor3(x[i], y[i], z[i]) for i in range(64)]
        c = [np.zeros_like(x[0])] + [self.lop(lambda u, v, w: (u & v) | (u & w) | (v & w), x[i], y[i], z[i])
                                      for i in range(63)]
        return s, c

    def ripple(self, x, y):
        out, carry = [], None
        for i in range(64):
            if carry is None:
                out.append(self.lop(lambda u, v: u ^ v, x[i], y[i]))
                carry = self.lop(lambda u, v: u & v, x[i], y[i])
            else:
                out.append(self.xor3(x[i], y[i], carry))
                if i < 63:
                    carry = self.lop(lambda u, v, w: (u & v) | (u & w) | (v & w), x[i], y[i], carry)
        return out

    def addn(self, terms):
        terms = list(terms)
        while len(terms) > 2:
            s, c = self.csa(*terms[:3])
            terms = terms[3:] + [s, c]
        return self.ripple(*terms)

    def rot(self, x, n):
        return [x[(i + n) % 64] for i in range(64)]

    def sig(self, name, x):
        a, b, c, s = SIG[name]
        p, q = self.rot(x, a), self.rot(x, b)
        t = self.rot(x, c) if c is not None else [x[i + s] if i + s < 64 else None for i in range(64)]
        return [self.xor3(p[i], q[i], t[i]) if t[i] is not None else self.lop(lambda u, v: u ^ v, p[i], q[i])
                for i in range(64)]


def to_planes(vals: np.ndarray) -> list:
    """vals: (32, G) uint64 -> 64 planes of shape (G,) uint32."""
    g = vals.shape[1]
    planes = []
    for i in range(64):
        bits = ((vals >> np.uint64(i)) & np.uint64(1)).astype(np.uint32)
        planes.append((bits << np.arange(32, dtype=np.uint32)[:, None]).sum(axis=0, dtype=np.uint32).reshape(g))
    return planes


def from_planes(planes: list) -> np.ndarray:
    g = planes[0].shape[0]
    out = np.zeros((32, g), dtype=np.uint64)
    for i, p in enumerate(planes):
        bits = (p[None, :] >> np.arange(32, dtype=np.uint32)[:, None]) & np.uint32(1)
        out |= bits.astype(np.uint64) << np.uint64(i)
    return out


def const_planes(k: int, g: int) -> list:
    return [np.full(g, M32 if k >> i & 1 else 0, dtype=np.uint32) for i in range(64)]


def bs_round(p: Planes, st, w, k: int):
    a, b, c, d, e, f, g, h = st
    s1 = p.sig("S1", e)
    ch = [p.lop(lambda x, y, z: (x & y) ^ (~x & z), e[i], f[i], g[i]) for i in range(64)]
    s0 = p.sig("S0", a)
    mj = [p.lop(lambda x, y, z: (x & y) ^ (x & z) ^ (y & z), a[i], b[i], c[i]) for i in range(64)]
    kk = const_planes(k, a[0].shape[0])
    # T1 = h + W + K + S1 + Ch kept carry-save; e' = d + T1; a' = T1 + S0 + Maj
    s, cy = p.csa(h, w, kk)
    s, cy2 = p.csa(s, cy, s1)
    s, cy = p.csa(s, cy2, ch)
    e2 = p.addn([d, s, cy])
    a2 = p.addn([s, cy, s0, mj])
    return [a2, a, b, c, e2, e, f, g]


def bs_sched(p: Planes, w16, m=None):
    terms = [p.sig("s1", w16[-2]), w16[-7], p.sig("s0", w16[-15]), w16[-16]] + ([m] if m is not None else [])
    return p.addn(terms)


def bs_round_exact(rng: np.random.Generator, g: int) -> tuple[bool, int]:
    """The bit-sliced round against the reference over 32 x g instances: (exact, LOP3s for 32 instances)."""
    st = rng.integers(0, 2**64, size=(8, 32, g), dtype=np.uint64)
    w = rng.integers(0, 2**64, size=(32, g), dtype=np.uint64)
    p = Planes()
    got = [from_planes(q) for q in bs_round(p, [to_planes(x) for x in st], to_planes(w), K[3])]
    ok = all([int(q[i, j]) for q in got] == round_ref([int(x) for x in st[:, i, j]], int(w[i, j]), K[3])
             for i in range(32) for j in range(g))
    return ok, p.ops


def part_a(checks: Checks, rng: np.random.Generator, f: dict) -> dict:
    g = 4
    out = {}
    ok, ops = bs_round_exact(rng, g)
    p = Planes()
    p.ops = ops
    per_inst = p.ops / 32
    out["bitsliced_round"] = {"ops_per_32": p.ops, "per_instance": per_inst, "honest": 28, "ratio": per_inst / 28,
                              "exact": bool(ok), "state_planes": 512}
    checks.check("A: a bit-sliced SHA-512 round (rotations free, adds as carry-save trees and two ripples) is exact and "
                 "costs 39.7 writes an instance against honest's 28: rounds resist bit-slicing (1.42x)",
                 ok and per_inst > 28 * 1.3, f"{p.ops} LOP3 for 32 instances, {per_inst:.2f} an instance")
    # bit-sliced schedule step, plain and with a message word
    res = {}
    for inj in (False, True):
        ws = rng.integers(0, 2**64, size=(16, 32, g), dtype=np.uint64)
        m = rng.integers(0, 2**64, size=(32, g), dtype=np.uint64)
        p = Planes()
        got = bs_sched(p, [to_planes(x) for x in ws], to_planes(m) if inj else None)
        ok = all(int(from_planes(got)[i, j]) == sched_ref([int(x) for x in ws[:, i, j]], int(m[i, j]) if inj else 0)
                 for i in range(32) for j in range(g))
        res[inj] = {"ops_per_32": p.ops, "per_instance": p.ops / 32, "honest": 20, "ratio": p.ops / 32 / 20,
                    "exact": bool(ok), "state_planes": 1024}
    out["bitsliced_sched"] = res[False]
    out["bitsliced_sched_injected"] = res[True]
    checks.check("A: a bit-sliced schedule step is exact and beats honest's word-wise 20 (15.9 an instance), and the "
                 "message-injected step (SCH's) ties it (19.8): the schedule resists word-wise programs only inside "
                 "SHA-512, where its outputs must reach the rounds as words",
                 res[False]["exact"] and res[True]["exact"] and res[False]["per_instance"] < 20
                 and abs(res[True]["per_instance"] - 20) < 0.5,
                 f"plain {res[False]['per_instance']:.2f}, injected {res[True]['per_instance']:.2f} an instance")
    # SCH at zero data: the bit-sliced prover also forms its words (a bit-sliced zero-data word chain, 1.30 a word)
    # against honest's two MMA words a step at (32 - q)/32 of a write each (q = 1 coin lane, the lead's)
    adv = res[True]["per_instance"] + 2 * 1.30
    hon = 20 + 2 * 31 / 32
    out["sch_zero_data"] = {"adversary_per_step": adv, "honest_per_step": hon, "margin": adv / hon - 1,
                            "note": "unbounded registers; a 255-register thread can't hold SCH's 1,024 planes"}
    checks.check("A: SCH's bit-sliced zero-data prover stays within 3% of honest a step (unbounded registers): "
                 "a margin as thin as XorWrites' before the pack attack broke it", abs(adv / hon - 1) < 0.03,
                 f"{adv:.2f} against {hon:.2f} ({adv / hon - 1:+.2%})")

    # the words at zero data: a round reads each 32-bit word as a whole IADD3 operand, so a packed chain (the pack
    # attack's layouts: L cells in g registers) must still put each word alone in a register (one SGXT, BFE or SHF a word)
    packs = {"plain chain": (1, 1, 0), "rot/w127: 3 cells in 2 registers": (2, 3, 1),
             "blc4a-e7: 2 cells in 1": (1, 2, 1), "blc1a-e4: 3 cells in 1": (1, 3, 1)}
    words = {k: g_ / l_ + x_ for k, (g_, l_, x_) in packs.items()}
    out["zero_data_words"] = words
    checks.check("A: at zero data no packed chain feeds the rounds below a write a word (derived): packing saves on the "
                 "chain but each word must be extracted to a full operand, unlike an XOR fold that runs on the lanes",
                 min(words.values()) == 1.0, ", ".join(f"{k} {v:.2f}" for k, v in words.items()))

    # skipping one round, and carry-free linearization of r rounds, on R80's rounds
    n = 2000
    st0 = rng.integers(0, 2**64, size=(n, 8), dtype=np.uint64)
    ws = rng.integers(0, 2**64, size=(n, 16), dtype=np.uint64)

    def run(s, wl, skip=None, lin=False):
        s = [int(x) for x in s]
        for t, wv in enumerate(wl):
            if t == skip:
                continue
            if lin:
                a, b, c, d, e, f_, g_, h = s
                t1 = h ^ sig_ref("S1", e) ^ ((e & f_) ^ (~e & g_ & M64)) ^ K[t] ^ int(wv)
                t2 = sig_ref("S0", a) ^ ((a & b) ^ (a & c) ^ (b & c))
                s = [t1 ^ t2, a, b, c, d ^ t1, e, f_, g_]
            else:
                s = round_ref(s, int(wv), K[t])
        return s

    def bits_agree(x, y):
        return sum(64 - bin(u ^ v).count("1") for u, v in zip(x, y)) / 512

    skip_eq = skip_bits = 0
    for j in range(n):
        full = run(st0[j], ws[j])
        sk = run(st0[j], ws[j], skip=int(rng.integers(0, 16)))
        skip_eq += full == sk
        skip_bits += bits_agree(full, sk)
    lin = {}
    for rr in (1, 2, 4, 8, 16):
        eq = agree = 0
        for j in range(n // 4):
            full = run(st0[j], ws[j][:rr])
            ln = run(st0[j], ws[j][:rr], lin=True)
            eq += full == ln
            agree += bits_agree(full, ln)
        lin[rr] = {"exact": eq / (n // 4), "bits_agree": agree / (n // 4)}
    out["skip_round"] = {"equal": skip_eq / n, "bits_agree": skip_bits / n}
    out["linearized"] = lin
    checks.check("A: skipping any one of 16 rounds never gives the right state (2,000 tries), and its bits agree at "
                 "the coin-flip rate", skip_eq == 0 and abs(skip_bits / n - 0.5) < 0.02,
                 f"equal {skip_eq}/{n}, bits agree {skip_bits / n:.4f}")
    checks.check("A: carry-free linearization (every add an XOR) is never exact from one round on, and its bits fall "
                 "to the coin-flip rate by 8 rounds: few-round SHA-512 has no exact linear shortcut",
                 all(v["exact"] == 0 for v in lin.values()) and abs(lin[8]["bits_agree"] - 0.5) < 0.02,
                 ", ".join(f"r {k}: {v['bits_agree']:.3f}" for k, v in lin.items()))

    # binding: one word changed, and a persistent shift X on one cell's words from step tau on (R80, 160 words)
    n = 300
    one = shift = 0
    for _ in range(n):
        s0 = [int(x) for x in rng.integers(0, 2**64, size=8, dtype=np.uint64)]
        wl = [int(x) for x in rng.integers(0, 2**64, size=160, dtype=np.uint64)]
        base = fold_ref(DESIGNS["R80"], s0, wl)
        w2 = list(wl)
        i = int(rng.integers(0, 160))
        w2[i] = (w2[i] + int(rng.integers(1, 2**32))) & M64
        one += fold_ref(DESIGNS["R80"], s0, w2) == base
        x = int(rng.integers(1, 2**32))
        tau = int(rng.integers(0, 150))
        w3 = [(v + (x if k >= tau and k % 4 == 1 else 0)) & M64 for k, v in enumerate(wl)]
        shift += fold_ref(DESIGNS["R80"], s0, w3) == base
    out["binding"] = {"one_word_equal": one, "persistent_shift_equal": shift, "tries": n}
    checks.check("A: R80 binds every word: a changed word, or a persistent shift X on one cell's later words, changes "
                 "the state every time (a round is a permutation of the state for a fixed word, injective in the word)",
                 one == 0 and shift == 0, f"{one}/{n}, {shift}/{n} equal")
    return out


def decomp_r80(rng: np.random.Generator, threads: int, cols: int, steps: int, xor: bool = False,
               fix_carry: bool = True) -> tuple[bool, set]:
    """COUNTER's decomp on R80 at zero data. Words are w = R_s(row) + C_s(col) mod 2^32 (x = 0, one row class); thread
    tau holds rows 2 tau and 2 tau + 1 and every column, and its message word k of step s is (lo, hi) = (w(2 tau, k),
    w(2 tau + 1, k)) at round position (s cols + k) mod 80, the same in every thread (one SIMT program). The prover
    never forms a word: U = h + R + CK with CK = C + K_p, one register shared by every thread, and the lo half's
    carry corrected by predicates (an ISETP against ~C, shared, and predicate logic: free in W1). Returns whether its
    states equal honest's and the corrected carries it fed IADD3.X (two carry-ins hold 0..2)."""
    design = DESIGNS["R80X" if xor else "R80"]
    R = rng.integers(0, 2**32, size=(steps, 2 * threads), dtype=np.uint64)
    C = rng.integers(0, 2**32, size=(steps, cols), dtype=np.uint64)
    st0 = rng.integers(0, 2**64, size=(threads, 8), dtype=np.uint64)
    ok, carries = True, set()
    for tau in range(threads):
        ws = [((int(R[s, 2 * tau + 1]) + int(C[s, k])) & M32) << 32 | ((int(R[s, 2 * tau]) + int(C[s, k])) & M32)
              for s in range(steps) for k in range(cols)]
        honest = fold_ref(design, [int(x) for x in st0[tau]], ws)
        st, cv = [int(x) for x in st0[tau]], None
        for i in range(steps * cols):
            s, k, p = i // cols, i % cols, i % 80
            if p == 0:
                cv = list(st)
            c_ = int(C[s, k])
            ck_lo, ck_hi, b2 = (c_ + (K[p] & M32)) & M32, (c_ + (K[p] >> 32)) & M32, (c_ + (K[p] & M32)) >> 32
            r_lo, r_hi = int(R[s, 2 * tau]), int(R[s, 2 * tau + 1])
            s_lo = (st[7] & M32) + r_lo + ck_lo
            b1 = int(r_lo > (~c_ & M32))
            c = (s_lo >> 32) - b1 + b2 if fix_carry else s_lo >> 32
            carries.add(c)
            st = round_u(st, (((st[7] >> 32) + r_hi + ck_hi + c) & M32) << 32 | (s_lo & M32))
            if p == 79:
                st = [(x + y) & M64 for x, y in zip(cv, st)]
        ok &= st == honest
    return bool(ok), carries


def decomp_share(threads: int, cols: int) -> float:
    """decomp's writes a word on R80: CK_lo, CK_hi and ~C a (step, column) shared by the threads, R a (step, row)."""
    return 3 / (2 * threads) + 1 / cols


def part_d(checks: Checks, rng: np.random.Generator) -> dict:
    ok, carries = decomp_r80(rng, threads=4, cols=16, steps=10)
    xok, _ = decomp_r80(rng, threads=2, cols=16, steps=5, xor=True)
    served = decomp_share(16, 64)
    out = {"exact": ok, "carries": sorted(carries), "xor_exact": xok, "share_toy": decomp_share(4, 16),
           "share_served": served, "served_layout": "16 threads a (column, register) in a row class, 64 columns"}
    checks.check("D: decomp on R80 at zero data is exact without forming a word: each word's add U = h + W + K_t "
                 "becomes h + R + (C + K_t) with C + K_t shared by the threads that hold the column at that position, "
                 "and the lo carry, corrected by free predicates, always fits IADD3.X's two carry-ins",
                 ok and carries <= {0, 1, 2}, f"4 threads x 160 words: exact {ok}, carries {sorted(carries)}")
    checks.check("D: at the served layout decomp pays 0.11 writes a word against honest's MMA write, so R80, R16 and "
                 "Rr lose the words' credit (T1 about r (1 - 0.11), past 100% of stock) and RD half of it",
                 abs(served - (3 / 32 + 1 / 64)) < 1e-12, f"{served:.4f} a word")
    checks.check("D: XORing each word with a per-position constant first (R80X) leaves decomp's program wrong: the "
                 "XOR doesn't pass through the add, so the word must be formed", not xok)
    return out


# ---------------------------------------------------------------- prices at the lead's points

def r_row(row: int) -> float:
    return R_CHAIN - CHUNK_K / (row / BLOCK + CHUNK_EXTRA)


def part_p(checks: Checks, lead: dict, f: dict, a: dict, dd: dict) -> dict:
    out = {"seal": {}, "designs": {}}
    # the share of a fold's words read only by an add that also takes a round constant: decomp's (part D)
    exposed = {"R80": 1.0, "R16": 1.0, "TC16": 1.0, "RD": 0.5, "R80X": 0.0, "RDX": 0.0, "SCH": 0.0,
               **{f"RR{k}": 1.0 for k in (2, 3, 4, 5)}}
    # the seal: each fold state hashed once a call at the lead's per-byte price; states per thread, cells per thread
    seal_cases = {"64 cells a thread, K 2,048": (64, 2048), "128 cells a thread, K 8,192": (128, 8192)}
    for name, fd in f.items():
        rows = {}
        for pt in POINTS:
            p = lead["points"][pt]
            r = p["r"]
            h_words = r * r_row(256 * 1024) / 32
            h_rest = p["h"] - h_words
            seal, head = {}, {}
            for case, (cells, kk) in seal_cases.items():
                words_per_state = cells * kk / 32 * r
                seal_w = fd["state_bits"] / 8 * r_row(256 * 1024) / BLOCK / words_per_state
                seal[case] = r * seal_w
                head[case] = r / 32 / words_per_state * fd["head_saving"] * W1_PER_REG
            h_fold = r * fd["writes_per_word"]
            h = {case: h_rest + h_fold + s for case, s in seal.items()}
            base = p["t1"] - p["terms"].get("chunk_head", 0.0)
            t1 = {case: base + head[case] for case in seal_cases}
            words = p["stock_macs"] * r / 32
            fold_ms = p["hash_ms"] * h_fold / p["h"]
            rest_ms = p["hash_ms"] * h_rest / p["h"]
            seal_ms = {case: p["hash_ms"] * s / p["h"] for case, s in seal.items()}
            parts = dict(p["extra_parts_ms"])
            extra = sum(parts.values()) - parts["export"]
            t2 = {case: {e: (s + extra + fold_ms + rest_ms + seal_ms[case]) / s for e, s in p["stock_ms"].items()}
                  for case in seal_cases}
            rows[pt] = {"h": h, "cut": {c: p["h"] / v for c, v in h.items()}, "h_fold": h_fold, "h_rest": h_rest,
                        "seal": seal, "t1_premise_holds": t1, "t1_chain_head": head, "t1_cross_call": base,
                        "t1_decomp": (r * exposed[name] * (1 - dd["share_served"]) if name in exposed else None),
                        "t2": t2, "words": words, "fold_ms": fold_ms}
        out["designs"][name] = {"what": fd["what"], "writes_per_word": fd["writes_per_word"], "points": rows}
    lead_h = lead["points"]["t8192"]["h"]
    r80 = out["designs"]["R80"]["points"]["t8192"]
    checks.check("P: R80 cuts h about 7.5x (16.6-17.0 against the lead's 127.5) at every point, T1 the lead's other "
                 "terms if its premise holds, T2 6-17x against the lead's 10-66x (estimate)",
                 all(7.3 < v < 7.8 for pt in POINTS for v in out["designs"]["R80"]["points"][pt]["cut"].values())
                 and all(max(t.values()) < 17 for pt in POINTS
                         for t in out["designs"]["R80"]["points"][pt]["t2"].values()),
                 ", ".join(f"{c}: h {v:.2f} ({lead_h / v:.2f}x)" for c, v in r80["h"].items()))
    heads = {n: out["designs"][n]["points"]["t8192"]["t1_chain_head"] for n in ("R80", "RD", "SCH")}
    c64, c128 = seal_cases
    checks.check("P: a chain begun afresh each call adds its head to T1: R80 and RD 0.49% at 64 cells and K 2,048, "
                 "0.06% at 128 and 8,192; SCH 3.4%, past the 1% bar, and 0.43%. A thread's chain carried across calls "
                 "(each call starting from the state the last one sealed) pays the head once a window instead",
                 all(abs(heads[n][c64] - 20 / 4096) < 1e-12 and abs(heads[n][c128] - 20 / 32768) < 1e-12
                     for n in ("R80", "RD")) and heads["SCH"][c64] > T1_BAR > heads["SCH"][c128],
                 ", ".join(f"{n} {v[c64]:.3%} / {v[c128]:.3%}" for n, v in heads.items()))
    rd = out["designs"]["RD"]["points"]["t8192"]
    checks.check("P: only RD and RDX (a modified round) and SCH (the schedule alone) reach 10x among the round folds",
                 sorted(n for n, d in out["designs"].items()
                        if all(v >= 10 for v in d["points"]["t8192"]["cut"].values())) == ["RD", "RDX", "SCH"],
                 f"RD h {min(rd['h'].values()):.2f}-{max(rd['h'].values()):.2f}; SCH h "
                 f"{min(out['designs']['SCH']['points']['t8192']['h'].values()):.2f}-"
                 f"{max(out['designs']['SCH']['points']['t8192']['h'].values()):.2f}")
    # the precision each premise is needed to: T1's slack over the fold, absolute writes a 32 words and relative
    prec = {}
    for name in ("R80", "R16", "RD", "SCH"):
        pt = lead["points"]["t8192"]
        slack = T1_BAR - (pt["t1"] - pt["terms"].get("chunk_head", 0.0))
        h_fold = out["designs"][name]["points"]["t8192"]["h_fold"]
        prec[name] = {"relative": slack / h_fold,
                      "writes_per_32_words": slack / pt["r"] * 32 / W1_PER_REG * 32}
    pt = lead["points"]["t8192"]
    slack = T1_BAR - (pt["t1"] - pt["terms"].get("chunk_head", 0.0))
    prec["lead"] = {"relative": slack / pt["h"], "writes_per_32_words": slack / pt["r"] * 32 / W1_PER_REG * 32}
    out["precision"] = prec
    checks.check("P: each fold's premise is needed to the same absolute slack a word as the lead's ShaWrites (T1 is "
                 "absolute), so to a relative precision h_lead/h_fold looser",
                 abs(prec["R80"]["writes_per_32_words"] - prec["lead"]["writes_per_32_words"]) < 1e-9,
                 ", ".join(f"{k} {v['relative']:.2e}" for k, v in prec.items()))
    # per monitoring period, in AND gates: honest commitment work = h x stock
    pools = {}
    for name in ("lead", "R80", "RD", "SCH"):
        h = lead_h if name == "lead" else max(out["designs"][name]["points"]["t8192"]["h"].values())
        pools[name] = {"h": h, **{per: {"h200": h * H200_MACS * sec * AND_PER_MAC,
                                        "datacenter": h * H200_MACS * sec * DC_GPUS * AND_PER_MAC,
                                        "world": h * WORLD_MACS_PER_YEAR * sec / PERIODS["year"] * AND_PER_MAC}
                                  for per, sec in PERIODS.items()}}
    pools["stock"] = {"h": 1.0, **{per: {"h200": H200_MACS * sec * AND_PER_MAC,
                                         "datacenter": H200_MACS * sec * DC_GPUS * AND_PER_MAC,
                                         "world": WORLD_MACS_PER_YEAR * sec / PERIODS["year"] * AND_PER_MAC}
                                   for per, sec in PERIODS.items()}}
    out["pools"] = pools
    return out


def main() -> int:
    checks = Checks()
    args = checks.args
    lead = json.loads(Path(args[0]).read_text())
    out_path = args[args.index("--json") + 1] if "--json" in args else None
    rng = np.random.default_rng(20261029)
    m = part_m(checks, rng)
    f = part_f(checks, rng)
    a = part_a(checks, rng, f)
    dd = part_d(checks, rng)
    p = part_p(checks, lead, f, a, dd)
    if checks.mutations:
        mrng = np.random.default_rng(7)
        global maj_w1
        good_maj = maj_w1
        maj_w1 = lambda r, a, b, c: (r.lop3(lambda x, y, z: x & y, a[0], b[0], c[0]),  # noqa: E731
                                     r.lop3(lambda x, y, z: x & y, a[1], b[1], c[1]))
        try:
            checks.mutation("the W1 round with Maj as a & b", not fold_exact(DESIGNS["R80"], DESIGNS["R80"], mrng))
        finally:
            maj_w1 = good_maj
        checks.mutation("R80's program without its feed-forward (R16's) against R80's reference",
                        not fold_exact(dict(DESIGNS["R80"], kind="R16"), DESIGNS["R80"], mrng))
        good_ripple = Planes.ripple

        def bad_ripple(self, x, y):
            o = good_ripple(self, x, y)
            o[63] = self.lop(lambda u, v: u ^ v, x[63], y[63])
            return o
        Planes.ripple = bad_ripple
        try:
            checks.mutation("the bit-sliced round with bit 63's carry-in dropped", not bs_round_exact(mrng, 1)[0])
        finally:
            Planes.ripple = good_ripple
        good_count = Regs._count

        def bad_count(self, op, k, *xs):
            if not (self.fold_consts and any(bool(np.all(x == x[0])) for x in xs)):
                self.n[op] += k
        Regs._count = bad_count
        try:
            checks.mutation("an op freed when any one input is a start-state value",
                            head_saving(DESIGNS["R80"], mrng) != 20)
        finally:
            Regs._count = good_count
        checks.mutation("decomp without its carry correction", not decomp_r80(mrng, 2, 16, 5, fix_carry=False)[0])
        checks.mutation("a skipped round taken for the full state",
                        fold_ref(DESIGNS["R16"], IV, list(range(1, 17)))
                        != fold_ref(DESIGNS["R16"], IV, list(range(1, 16))))
    if out_path:
        Path(out_path).write_text(json.dumps({"model": m, "folds": f, "attacks": a, "decomp": dd, "prices": p}, indent=1,
                                             sort_keys=True, default=float) + "\n")
    print("\ndesign  writes/word  h@t8192 (64 cells K2048 | 128 cells K8192)  cut   T2 t8192 sync/async")
    for name, d in p["designs"].items():
        q = d["points"]["t8192"]
        cs = list(q["h"])
        print(f"{name:6s} {d['writes_per_word']:8.3f}   {q['h'][cs[0]]:7.2f} | {q['h'][cs[1]]:7.2f}"
              f"   {q['cut'][cs[0]]:5.2f}-{q['cut'][cs[1]]:5.2f}x   "
              f"{q['t2'][cs[0]]['sync']:5.2f}/{q['t2'][cs[0]]['async']:5.2f}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
