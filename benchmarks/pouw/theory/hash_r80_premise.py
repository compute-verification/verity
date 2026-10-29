"""ShaWrites-R80's falsification test, and the round function's cap on what a fold inside it can absorb.

    python3 hash_r80_premise.py [--trials N] [--blocks B] [--bound R] [--eps E] [--mutations] [--verdict PATH]

**The premise.** ShaWrites-R80(R, eps): no W1 program that writes fewer than R 32-bit registers a block computes R80
(hash_rounds.py: SHA-512's 80 rounds over 80 free 64-bit words, K_t at round t, the feed-forward added) on B chained
blocks correctly on more than 1 - eps of the protocol's inputs, for any data the prover chooses. Honest R80 writes 2,254
a block. With SHA-512's schedule (64 steps of 20 writes) in front of it, an R80 program computes SHA-512's compression,
so ShaWrites at R_SHA registers a compression gives R = R_SHA - 1,280 for programs correct on schedule-consistent
words; on the protocol's words, which aren't, it is a premise of its own.

**The test** (`falsify`): a candidate is a W1 program on `Machine`, a function of register handles that calls only the
machine's instructions (IADD3 with its carries, LOP3, SHF, IMAD, IMAD.HI, PRMT, SEL; ISETP's predicates and immediates
free), so its count is the machine's. It runs on N random instances of a distribution (uniform words, or each 32-bit
half in a 2^20 span around 0, the zero-data words' range) and is compared bit for bit with `r80_np`. The premise is
falsified when the candidate writes fewer than R a block and its correct fraction, less Hoeffding's margin at
confidence 1 - 2^-20 (0.0073 at the default 2^17 instances), is at least 1 - eps. Controls: honest R80 on the machine writes 2,254 a block and is exact; a
candidate one round short is cheap and wrong; with R one above honest's count, honest's program falsifies it.

**The cap** (part K): the round reads its word only through U = h + W + K_t, so a round with words in both W_t and the
K_t slot is a round of their sum (checked), a fold that puts words there collides at (w + d, k - d) (checked), and a
fold inside the unmodified round absorbs at most 64 bits a round of 28 writes: at least 14 writes a 32-bit word, 15
with the XOR that `decomp` requires.

Stdlib, numpy and exact_kit.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_kit.checks import Checks
from hash_rounds import DESIGNS, K, fold_ref, round_ref

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
HONEST = 2254
SCHED = 64 * 20
DELTA = 2.0 ** -20


class Reg:
    __slots__ = ("i",)

    def __init__(self, i: int):
        self.i = i


class Pred:
    __slots__ = ("i",)

    def __init__(self, i: int):
        self.i = i


class Machine:
    """W1's integer core on 32-bit registers over n instances. A candidate holds only handles (Reg, Pred) and Python
    ints (immediates); every instruction that writes a register counts one write; predicates and immediates are free."""

    def __init__(self, n: int):
        self.n = n
        self._r: list[np.ndarray] = []
        self._p: list[np.ndarray] = []
        self.writes = 0

    def _val(self, a) -> np.ndarray:
        if isinstance(a, Reg):
            return self._r[a.i].astype(np.uint64)
        return np.full(self.n, a & M32, np.uint64)

    def _reg(self, v: np.ndarray, count: bool = True) -> Reg:
        self.writes += count
        self._r.append((v & np.uint64(M32)).astype(np.uint32))
        return Reg(len(self._r) - 1)

    def _pred(self, v: np.ndarray) -> Pred:
        self._p.append(v.astype(bool))
        return Pred(len(self._p) - 1)

    def load(self, v: np.ndarray) -> Reg:
        return self._reg(v.astype(np.uint64), count=False)

    def read(self, a: Reg) -> np.ndarray:
        return self._r[a.i]

    def iadd3(self, a, b, c=0, cin=(), neg=()) -> tuple[Reg, tuple[Pred, Pred]]:
        """a + b + c + carry-ins (each operand negated if its index is in neg); the carry-out as two predicates."""
        vals = [self._val(x) for x in (a, b, c)]
        s = np.zeros(self.n, np.int64)
        for k, v in enumerate(vals):
            s = s + (-v.astype(np.int64) if k in neg else v.astype(np.int64))
        for p in cin:
            s = s + self._p[p.i].astype(np.int64)
        carry = s >> 32
        return self._reg((s & M32).astype(np.uint64)), (self._pred(carry >= 1), self._pred(carry >= 2))

    def lop3(self, a, b, c, lut: int) -> Reg:
        x, y, z = (self._val(v) for v in (a, b, c))
        out = np.zeros(self.n, np.uint64)
        for k in range(8):
            if lut >> k & 1:
                m = (x if k & 4 else ~x) & (y if k & 2 else ~y) & (z if k & 1 else ~z)
                out |= m
        return self._reg(out)

    def shf(self, lo, hi, s: int) -> Reg:
        """The funnel (hi:lo) >> s, low 32 bits."""
        return self._reg((self._val(hi) << np.uint64(32) | self._val(lo)) >> np.uint64(s))

    def imad(self, a, b, c) -> Reg:
        return self._reg(self._val(a) * self._val(b) + self._val(c))

    def imad_hi(self, a, b, c) -> Reg:
        p = self._val(a).astype(object) * self._val(b).astype(object)
        return self._reg(np.array([int(x) >> 32 for x in p], np.uint64) + self._val(c))

    def prmt(self, a, b, sel) -> Reg:
        x = self._val(b) << np.uint64(32) | self._val(a)
        s = self._val(sel)
        out = np.zeros(self.n, np.uint64)
        for k in range(4):
            idx = (s >> np.uint64(4 * k)) & np.uint64(7)
            out |= ((x >> (idx * np.uint64(8))) & np.uint64(0xFF)) << np.uint64(8 * k)
        return self._reg(out)

    def sel(self, a, b, p: Pred) -> Reg:
        return self._reg(np.where(self._p[p.i], self._val(a), self._val(b)))

    def isetp(self, a, b, op: str) -> Pred:
        x, y = self._val(a), self._val(b)
        return self._pred({"eq": x == y, "ne": x != y, "ltu": x < y, "geu": x >= y}[op])


# ---------------------------------------------------------------- R80 on the machine (honest) and in numpy (reference)

LUT_XOR3, LUT_CH, LUT_MAJ = 0x96, 0xCA, 0xE8
SIG = {"S0": (28, 34, 39), "S1": (14, 18, 41)}


def add64(m: Machine, *terms):
    """A 64-bit add of two or three (lo, hi) terms or 64-bit immediates: IADD3, IADD3.X."""
    lo = [t[0] if isinstance(t, tuple) else t & M32 for t in terms] + [0] * (3 - len(terms))
    hi = [t[1] if isinstance(t, tuple) else t >> 32 for t in terms] + [0] * (3 - len(terms))
    rlo, cy = m.iadd3(*lo)
    rhi, _ = m.iadd3(*hi, cin=cy)
    return rlo, rhi


def rotr64(m: Machine, x, n: int):
    lo, hi = x if n < 32 else (x[1], x[0])
    s = n & 31
    return (m.shf(lo, hi, s), m.shf(hi, lo, s)) if s else (lo, hi)


def sig64(m: Machine, name: str, x):
    p, q, r = (rotr64(m, x, s) for s in SIG[name])
    return m.lop3(p[0], q[0], r[0], LUT_XOR3), m.lop3(p[1], q[1], r[1], LUT_XOR3)


def honest_r80(m: Machine, st0, ws, rounds: int = 80):
    """Honest R80 on the machine: 28 writes a round, the feed-forward fused into round 79 (2,254 a block)."""
    st = list(st0)
    for b0 in range(0, len(ws), 80):
        cv = list(st)
        for t in range(rounds):
            a, b, c, d, e, f, g, h = st
            u = add64(m, h, ws[b0 + t], K[t])
            ch = (m.lop3(e[0], f[0], g[0], LUT_CH), m.lop3(e[1], f[1], g[1], LUT_CH))
            t1 = add64(m, u, sig64(m, "S1", e), ch)
            mj = (m.lop3(a[0], b[0], c[0], LUT_MAJ), m.lop3(a[1], b[1], c[1], LUT_MAJ))
            s0 = sig64(m, "S0", a)
            if t < 79:
                st = [add64(m, t1, s0, mj), a, b, c, add64(m, d, t1), e, f, g]
            else:
                st = [add64(m, add64(m, t1, s0, mj), cv[0]), add64(m, cv[1], a), add64(m, cv[2], b),
                      add64(m, cv[3], c), add64(m, t1, d, cv[4]), add64(m, cv[5], e), add64(m, cv[6], f),
                      add64(m, cv[7], g)]
    return st


def _rotr(x, n):
    return (x >> np.uint64(n)) | (x << np.uint64(64 - n))


def r80_np(st0: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """R80 in numpy: st0 (8, n), ws (80 B, n), uint64."""
    st = [x.copy() for x in st0]
    with np.errstate(over="ignore"):
        for b0 in range(0, len(ws), 80):
            cv = [x.copy() for x in st]
            for t in range(80):
                a, b, c, d, e, f, g, h = st
                t1 = h + ws[b0 + t] + np.uint64(K[t]) + (_rotr(e, 14) ^ _rotr(e, 18) ^ _rotr(e, 41)) + ((e & f) ^ (~e & g))
                t2 = (_rotr(a, 28) ^ _rotr(a, 34) ^ _rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c))
                st = [t1 + t2, a, b, c, d + t1, e, f, g]
            st = [x + y for x, y in zip(cv, st)]
    return np.stack(st)


def draw(rng: np.random.Generator, dist: str, blocks: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    st0 = rng.integers(0, 2**64, (8, n), dtype=np.uint64)
    if dist == "uniform":
        return st0, rng.integers(0, 2**64, (80 * blocks, n), dtype=np.uint64)
    half = (rng.integers(-(1 << 19), 1 << 19, (2, 80 * blocks, n)) & M32).astype(np.uint64)
    return st0, half[1] << np.uint64(32) | half[0]


def falsify(candidate, rng: np.random.Generator, *, dist: str = "zero-data", blocks: int = 1, trials: int = 4096,
            bound: int = HONEST, eps: float = 0.01) -> dict:
    """Runs candidate(machine, st0, ws) -> 8 (lo, hi) handle pairs; the premise ShaWrites-R80(bound, eps) is falsified
    when it writes fewer than bound a block and is right, at confidence 1 - DELTA, on at least 1 - eps of instances."""
    st0, ws = draw(rng, dist, blocks, trials)
    m = Machine(trials)
    pair = lambda x: (m.load(x & np.uint64(M32)), m.load(x >> np.uint64(32)))  # noqa: E731
    out = candidate(m, [pair(x) for x in st0], [pair(x) for x in ws])
    got = np.stack([m.read(lo).astype(np.uint64) | m.read(hi).astype(np.uint64) << np.uint64(32) for lo, hi in out])
    right = float(np.mean(np.all(got == r80_np(st0, ws), axis=0)))
    lcb = right - math.sqrt(math.log(1 / DELTA) / (2 * trials))
    return {"writes_a_block": m.writes / blocks, "right": right, "lcb": lcb,
            "falsified": m.writes < bound * blocks and lcb >= 1 - eps}


# ---------------------------------------------------------------- main

def main() -> int:
    checks = Checks()
    a = checks.args

    def opt(name: str, default):
        return type(default)(a[a.index(name) + 1]) if name in a else default
    trials, blocks, bound, eps = opt("--trials", 1 << 17), opt("--blocks", 2), opt("--bound", HONEST), opt("--eps", 0.01)
    rng = np.random.default_rng(20261029)

    # the reference against hash_rounds' R80, instance by instance
    st0, ws = draw(rng, "uniform", 2, 4)
    ref = r80_np(st0, ws)
    same = all([int(x) for x in ref[:, j]] == fold_ref(DESIGNS["R80"], [int(x) for x in st0[:, j]],
                                                        [int(x) for x in ws[:, j]]) for j in range(4))
    checks.check("reference: r80_np equals hash_rounds' R80 bit for bit on two chained blocks", same)

    res = {d: falsify(honest_r80, rng, dist=d, blocks=blocks, trials=trials, bound=bound, eps=eps)
           for d in ("uniform", "zero-data")}
    checks.check(f"S: honest R80 on the machine writes {HONEST:,} a block and is exact on both distributions, so it "
                 f"doesn't falsify ShaWrites-R80({bound:,}, {eps:g})",
                 all(r["writes_a_block"] == HONEST and r["right"] == 1.0 and not r["falsified"] for r in res.values()),
                 str(res))
    short = falsify(lambda m, s, w: honest_r80(m, s, w, rounds=79), rng, blocks=1, trials=trials, bound=bound, eps=eps)
    checks.check("S controls: a candidate one round short writes less and is wrong, so it doesn't falsify",
                 short["writes_a_block"] < HONEST and short["right"] == 0.0 and not short["falsified"], str(short))
    above = falsify(honest_r80, rng, blocks=1, trials=4096, bound=HONEST + 1, eps=0.1)
    checks.check("S controls: at a bound one above honest's count and eps 0.1, honest's own program falsifies the "
                 "premise (the test fires when a program beats its bound)", above["falsified"], str(above))
    checks.check(f"S: ShaWrites at R_SHA registers a compression gives ShaWrites-R80 at R_SHA - {SCHED:,} for programs "
                 "right on schedule-consistent words (the schedule is 64 steps of 20 writes): 3,510 gives 2,230 a block, "
                 f"{HONEST - (3510 - SCHED)} under honest R80's {HONEST:,}", 3510 - SCHED == 2230)

    # part K: the round reads its word only through U = h + W + K.
    ok_wk, ok_col = True, True
    for _ in range(64):
        st = [int(x) for x in rng.integers(0, 2**64, 8, dtype=np.uint64)]
        w, k, d = (int(x) for x in rng.integers(0, 2**64, 3, dtype=np.uint64))
        ok_wk &= round_ref(st, w, k) == round_ref(st, (w + k) & M64, 0) == round_ref(st, 0, (w + k) & M64)
        ok_col &= round_ref(st, (w + d) & M64, (k - d) & M64) == round_ref(st, w, k)
    checks.check("K: a round with a word in W_t and one in the K_t slot equals a round of their sum, so the K slot adds "
                 "no bits", ok_wk)
    checks.check("K: a fold with words in both slots collides at (w + d, k - d) for every d: it doesn't bind them", ok_col)
    m = Machine(1)
    honest_r80(m, [(m.load(np.zeros(1, np.uint64)), m.load(np.zeros(1, np.uint64)))] * 8,
               [(m.load(np.zeros(1, np.uint64)), m.load(np.zeros(1, np.uint64)))] * 80, rounds=79)
    per_round = m.writes / 79
    checks.check("K: the round is 28 writes and absorbs at most 64 bits, so a fold inside the unmodified round is at "
                 "least 14 writes a 32-bit word, 15 with decomp's XOR", per_round == 28, f"{per_round:g} a round")

    if checks.mutations:
        with checks.mutant("the round reads W and K separately (U = h + W ^ K)"):
            st = [int(x) for x in rng.integers(0, 2**64, 8, dtype=np.uint64)]
            w, k = (int(x) for x in rng.integers(0, 2**64, 2, dtype=np.uint64))
            checks.check("K slot adds no bits", round_ref(st, w ^ k, 0) == round_ref(st, (w + k) & M64, 0))
        with checks.mutant("the honest program drops a carry-in"):
            def drop(m, s, w):
                orig = m.iadd3
                calls = [0]

                def iadd3(*args, **kw):
                    calls[0] += 1
                    if calls[0] == 8:
                        kw["cin"] = ()
                    return orig(*args, **kw)
                m.iadd3 = iadd3
                return honest_r80(m, s, w)
            r = falsify(drop, rng, blocks=1, trials=512)
            checks.check("honest is exact", r["right"] == 1.0)
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
