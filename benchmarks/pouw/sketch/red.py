"""SI-RED: the cheapest prover for a fold or a MAC function over the accumulator words, exact, priced in W1 writes.

A design is a block function S = P(w_1, …, w_T; k) over T consecutive words of one accumulator, its coins k drawn
live and shared by the call, written once in Python over ring values. The same function runs on numpy arrays (`Val`)
and on trace nodes (`Sym`), and the trace gives the design's structure and its price:

- the op set: a polynomial over its ring (+, −, ×, constant shifts, a word embedded into F_p or Z/2⁶⁴), lane-wise
  (⊕, ∧, ∨, ¬ only), or a T-function (no right shift, rotation or reduction mod p);
- its degree in the words, and the op each word first meets (`entry`);
- its W1 price, with IADD3, IMAD and LOP3 merging and each add or XOR tree's operands regrouped by what they depend on
  (a row, a column class, a cell's step-invariant offset, or the cell's step), each group pre-summed where it is shared.

`hw` designs are MAC functions: the chip applies P inside the MMA path, so honest pays only the MMA, and a cheater runs
the MMA too or emulates P in software at the trace's price.

Words (`family`): the lead's zero-data words w_t = x(i, j) + R_t(i) + C_t(class(i), j), as `nh_attack` builds them
(COUNTER's `lead:rot` and NOISE's lead words), COUNTER's with x = 0 (`counter_x0`: the per-cell offset is the prover's
when it is data rather than a live coin), the lead's words with random data, and on-chip noise that no split
reproduces: `operand` (each step's increment a GEMM of fresh s8 noise operands, k 32) and `cell` (an i.i.d. 32-bit
increment a cell a step). On-chip noise rides the data lanes, so its stock is the whole MMA. A `_inc` family feeds the
k-steps' increments c_t in place of the words, for a MAC function acc ← F(acc, c_t; k) applied every k-step.

Attacks, each computing S itself and checked bit for bit against honest's:

- `fuse` (shortcut 2): no word is formed; each enters its consumer as x + R_t + C_t, merged into the add tree there.
  `mix` fuses only the steps where that saves writes (a block's first rounds from coin state, the edge attack) and
  forms the rest.
- `decomp` (shortcuts 1, 3 and 6): for a polynomial design of degree d ≤ DMAX, P's Newton expansion around the row's
  walk, S = Σ_e C(x, e)·Σ_β C(C, β)·Δ₁^e Δ^β P(R), β a multiset of steps and C(·, ·) binomials. The β = ∅ terms are
  row-only, a term whose coefficient is the same on every row is column-only, and the rest are an inner product a cell:
  IMMA digit chains mod 2³² (`nh_attack.imma`), and 16 u8 digit-pair chains mod p. C(x, e) is step-invariant, priced
  every S and once a call (`_once`). Row coefficients are priced as black-box evaluations of P, one a point, so the
  price is an upper bound on the cheapest decomposition.
- `pack` (shortcut 4): a lane-wise design run on b-bit lanes of one register, b the block's signed word width, and the
  bits above the lane from the words' sign bits, bitsliced 32 cells a register, once per distinct coin-bit column.
- `tfunc` (shortcut 5): flagged from the op set and verified (S mod 2^b from the words mod 2^b); not priced.

Prices are W1 register writes an S, as `nh_attack`'s: honest pays an MMA a word plus the trace (only the MMA when
`hw`); stock is the data lanes' share of the MMAs. Credit is C₀ − H accounting's, everything honest does, so T1 fails
when `gap_claimed` = (honest − cheapest)/stock is over 1%; `gap_share` = (stock − cheapest)/stock over 1% fails even a
credit of the stock alone. The S's seal is common to every program and left out.

Run (from this directory, COUNTER's `hash_counter.py` and NOISE's `commit_space.py`, `noise_space.py` and
`noise_sweep.py` in DIR): `uv run --no-project --with numpy --with-editable ../exact_kit python -B red.py OUT.json
--lib=DIR [--designs=a,b] [--mutations]`.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from collections import Counter

import numpy as np

import nh_attack as nh
from exact_kit.checks import Checks

M32 = 1 << 32
M64 = 1 << 64
P31 = (1 << 31) - 1
CALL_M = CALL_N = 4096
T1 = 0.01
DMAX = 5
TILE = 64
PRG_WORD = 14
GFMUL_SW = 24
COST = {32: {"add": 1, "mul": 1, "bit": 1, "shift": 1, "embed": 2, "gfmul": GFMUL_SW},
        64: {"add": 2, "mul": 4, "bit": 2, "shift": 2, "embed": 1},
        "p": {"add": 2, "mul": 5, "bit": 99, "shift": 99, "embed": 2}}
POLY_OPS = {"add", "sub", "neg", "mul", "shl", "embed", "to64"}
BIT_OPS = {"xor", "and", "or", "not"}
TFUNC_OPS = POLY_OPS - {"embed"} | BIT_OPS


def ring(mod):
    return 32 if mod == M32 else 64 if mod == M64 else "p"


def _red(a, mod):
    if mod == M64:
        return a
    if mod & (mod - 1) == 0:
        return a & np.uint64(mod - 1)
    return a % np.uint64(mod)


class Val:
    """A ring value: a numpy uint64 array mod `mod` (2^b, b ≤ 64, or a prime below 2³²)."""

    def __init__(self, a, mod):
        self.mod = mod
        self.a = _red(np.asarray(a, dtype=np.uint64), mod)

    @staticmethod
    def signed(v, mod):
        v = np.asarray(v, dtype=np.int64)
        return Val(v.astype(np.uint64) if mod & (mod - 1) == 0 else (v % mod).astype(np.uint64), mod)

    def _o(self, o):
        if isinstance(o, Val):
            if o.mod != self.mod:
                raise ValueError(f"ring mismatch {o.mod} vs {self.mod}")
            return o.a
        return np.uint64(int(o) % self.mod)

    def __add__(self, o):
        return Val(self.a + self._o(o), self.mod)

    __radd__ = __add__

    def __sub__(self, o):
        b = self._o(o)
        return Val(self.a - b if self.mod == M64 else self.a + (np.uint64(self.mod) - b), self.mod)

    def __rsub__(self, o):
        return Val(np.zeros_like(self.a) + self._o(o), self.mod) - self

    def __mul__(self, o):
        return Val(self.a * self._o(o), self.mod)

    __rmul__ = __mul__

    def __neg__(self):
        return 0 - self

    def __xor__(self, o):
        return Val(self.a ^ self._o(o), self.mod)

    __rxor__ = __xor__

    def __and__(self, o):
        return Val(self.a & self._o(o), self.mod)

    __rand__ = __and__

    def __or__(self, o):
        return Val(self.a | self._o(o), self.mod)

    __ror__ = __or__

    def __invert__(self):
        return Val(self.a ^ np.uint64(self.mod - 1), self.mod)

    def __lshift__(self, k):
        return Val(self.a << np.uint64(k), self.mod)

    def __rshift__(self, k):
        return Val(self.a >> np.uint64(k), self.mod)

    def rotl(self, r, w):
        return Val((self.a << np.uint64(r)) | (self.a >> np.uint64(w - r)), self.mod)

    def embed(self, p):
        s = self.a.astype(np.int64)
        return Val.signed(np.where(s >= 1 << 31, s - (1 << 32), s), p)

    def embedu(self, p):
        return Val(self.a % np.uint64(p), p)

    def lift(self):
        return Val(self.a, M32)

    def to64(self):
        s = self.a.astype(np.int64)
        return Val.signed(np.where(s >= 1 << 31, s - (1 << 32), s), M64)

    def cat(self, lo):
        return Val((self.a << np.uint64(32)) | lo.a, M64)

    def gfmul(self, o):
        """Carry-less product mod x³² + x⁷ + x³ + x² + 1 (GF(2³²))."""
        a, b = self.a, self._o(o)
        lo = np.zeros_like(a + b)
        hi = np.zeros_like(lo)
        for i in range(32):
            bit = (b >> np.uint64(i)) & np.uint64(1)
            m = np.uint64(0) - bit
            lo ^= (a << np.uint64(i)) & m & np.uint64(M32 - 1)
            hi ^= ((a >> np.uint64(32 - i)) if i else np.zeros_like(a)) & m
        for i in range(31, -1, -1):
            bit = (hi >> np.uint64(i)) & np.uint64(1)
            m = np.uint64(0) - bit
            red = np.uint64(0b10001101) << np.uint64(i)
            lo ^= red & m & np.uint64(M32 - 1)
            hi ^= (red >> np.uint64(32)) & m
        return Val(lo, M32)


class Trace:
    """A straight-line program over one block: each node an op, its args (node ids or constants), ring, class and
    degree in the words. Classes: const, glob (coins), row, col, static (x, a cell's step-invariant offset), cell."""

    def __init__(self, dist64=False):
        self.op, self.args, self.mod, self.cls, self.deg = [], [], [], [], []
        self.dist64 = dist64

    def leaf(self, op, mod, cls, deg):
        self.op.append(op), self.args.append([]), self.mod.append(mod), self.cls.append(cls), self.deg.append(deg)
        return Sym(self, len(self.op) - 1, mod)

    def make(self, op, args, mod):
        refs, cls, degs = [], "const", []
        for a in args:
            if isinstance(a, Sym):
                refs.append(("n", a.i))
                cls, d = join(cls, self.cls[a.i]), self.deg[a.i]
            else:
                refs.append(("c", int(a)))
                d = 0
            degs.append(d)
        if op in ("add", "sub"):
            deg = None if None in degs else max(degs)
        elif op == "mul":
            deg = None if None in degs else sum(degs)
        elif op in ("neg", "shl"):
            deg = degs[0]
        elif op in ("embed", "to64"):
            deg = degs[0] if degs[0] in (0, 1) else None
        else:
            deg = 0 if all(d == 0 for d in degs) else None
        self.op.append(op), self.args.append(refs), self.mod.append(mod), self.cls.append(cls), self.deg.append(deg)
        return Sym(self, len(self.op) - 1, mod)


def join(a, b):
    if a == b:
        return a
    s = {a, b} - {"const"}
    if len(s) == 1:
        return s.pop()
    s -= {"glob"}
    return s.pop() if len(s) == 1 else "cell"


class Sym:
    __slots__ = ("tr", "i", "mod")

    def __init__(self, tr, i, mod):
        self.tr, self.i, self.mod = tr, i, mod

    def _b(self, op, o, swap=False):
        return self.tr.make(op, [o, self] if swap else [self, o], self.mod)

    def __add__(self, o):
        return self._b("add", o)

    def __radd__(self, o):
        return self._b("add", o, True)

    def __sub__(self, o):
        return self._b("sub", o)

    def __rsub__(self, o):
        return self._b("sub", o, True)

    def __mul__(self, o):
        return self._b("mul", o)

    def __rmul__(self, o):
        return self._b("mul", o, True)

    def __neg__(self):
        return self.tr.make("neg", [self], self.mod)

    def __xor__(self, o):
        return self._b("xor", o)

    __rxor__ = __xor__

    def __and__(self, o):
        return self._b("and", o)

    __rand__ = __and__

    def __or__(self, o):
        return self._b("or", o)

    __ror__ = __or__

    def __invert__(self):
        return self.tr.make("not", [self], self.mod)

    def __lshift__(self, k):
        return self.tr.make("shl", [self, k], self.mod)

    def __rshift__(self, k):
        return self.tr.make("shr", [self, k], self.mod)

    def rotl(self, r, w):
        return self.tr.make("rotl", [self, r], self.mod)

    def embed(self, p):
        return self.tr.make("embed", [self], p)

    def embedu(self, p):
        return self.tr.make("embedu", [self], p)

    def lift(self):
        return self.tr.make("lift", [self], M32)

    def to64(self):
        tr = self.tr
        if tr.dist64 and tr.op[self.i] == "add" and all(k == "n" for k, _ in tr.args[self.i]):
            a, b = (Sym(tr, j, M32) for _, j in tr.args[self.i])
            return a.to64() + b.to64()
        return tr.make("to64", [self], M64)

    def cat(self, lo):
        return self.tr.make("cat", [self, lo], M64)

    def gfmul(self, o):
        return self._b("gfmul", o)


def rotr(x, r, w):
    return x.rotl(w - r, w)


# ---------- designs

def chacha_qr(a, b, c, d):
    a = a + b
    d = (d ^ a).rotl(16, 32)
    c = c + d
    b = (b ^ c).rotl(12, 32)
    a = a + b
    d = (d ^ a).rotl(8, 32)
    c = c + d
    b = (b ^ c).rotl(7, 32)
    return a, b, c, d


def _icbrt(n):
    x = 1 << ((n.bit_length() + 2) // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            break
        x = y
    while (x + 1) ** 3 <= n:
        x += 1
    return x


def _primes(k):
    out, n = [], 2
    while len(out) < k:
        if all(n % p for p in out if p * p <= n):
            out.append(n)
        n += 1
    return out


SHA512_K = [_icbrt(p << 192) % M64 for p in _primes(80)]


def sha512_round(s, wt, kt):
    a, b, c, d, e, f, g, h = s
    t1 = h + (rotr(e, 14, 64) ^ rotr(e, 18, 64) ^ rotr(e, 41, 64)) + ((e & f) ^ (~e & g)) + kt + wt
    t2 = (rotr(a, 28, 64) ^ rotr(a, 34, 64) ^ rotr(a, 39, 64)) + ((a & b) ^ (a & c) ^ (b & c))
    return (t1 + t2, a, b, c, d + t1, e, f, g)


def brw(ms, tp):
    """Bernstein's BRW polynomial of ms, tp[k] = τ^(2^k)."""
    n = len(ms)
    if n == 0:
        return 0
    if n == 1:
        return ms[0]
    if n == 2:
        return ms[0] * tp[0] + ms[1]
    if n == 3:
        return (tp[0] + ms[0]) * (tp[1] + ms[1]) + ms[2]
    k = n.bit_length() - 1
    return brw(ms[:2 ** k - 1], tp) * (tp[k] + ms[2 ** k - 1]) + brw(ms[2 ** k:], tp)


def tau_pows(tau, n):
    out = [tau]
    for _ in range(max(1, n.bit_length())):
        out.append(out[-1] * out[-1])
    return out


def d_nh(W, k):
    S = 0
    for s in range(0, len(W) - 1, 2):
        S = W[s] * W[s + 1] + S
    return S


def d_poly(W, k):
    S = W[0]
    for w in W[1:]:
        S = S * k["tau"] + w
    return S


def d_xor(W, k):
    S = W[0]
    for w in W[1:]:
        S = S ^ w
    return S


def d_brw32(W, k):
    return brw(list(W), tau_pows(k["tau"], len(W)))


def d_brw31(W, k):
    return brw([w.embed(P31) for w in W], tau_pows(k["tau"], len(W)))


def d_sqchain31(W, k):
    S = k["s0"]
    for w in W:
        S = (S.lift() ^ w).embedu(P31)
        S = S * S
    return S


def d_sqadd31(W, k):
    S = k["s0"]
    for w in W:
        S = S + w.embed(P31)
        S = S * S
    return S


def d_pow4(W, k):
    S = 0
    for w in W:
        w2 = w * w
        S = w2 * w2 + S
    return S


def d_chacha(entry):
    def f(W, k):
        a, b, c, d = k["s0"], k["s1"], k["s2"], k["s3"]
        for w in W:
            a = a + w if entry == "add" else a ^ w
            a, b, c, d = chacha_qr(a, b, c, d)
        return d
    return f


def d_sha512(W, k):
    s = tuple(k[f"h{i}"] for i in range(8))
    for t, w in enumerate(W):
        s = sha512_round(s, w.to64(), SHA512_K[t % 80])
    return s[0]


def d_sha512x(W, k):
    """SI-THEORY's hardware fold: half a SHA-512 round a word, W_t = (w_2t ‖ w_2t+1) ⊕ c_t, then the round's add."""
    s = tuple(k[f"h{i}"] for i in range(8))
    for t in range(len(W) // 2):
        s = sha512_round(s, W[2 * t].cat(W[2 * t + 1]) ^ k[f"c{t}"], SHA512_K[t % 80])
    return s[0]


def d_brw32x(W, k):
    return brw([w ^ k[f"c{t}"] for t, w in enumerate(W)], tau_pows(k["tau"], len(W)))


def d_add(W, k):
    S = W[0]
    for w in W[1:]:
        S = S + w
    return S


def d_rho(kind):
    """acc ← ρ(acc) + c a k-step, the words the k-steps' increments c: `rot` rotl(acc, 13), `xs` acc ⊕ (acc ≫ 15)."""
    def f(W, k):
        S = k["s0"]
        for c in W:
            S = (S.rotl(13, 32) if kind == "rot" else S ^ (S >> 15)) + c
        return S
    return f


def d_poly31(W, k):
    S = W[0].embed(P31)
    for w in W[1:]:
        S = S * k["tau"] + w.embed(P31)
    return S


def d_gf32(W, k):
    S = W[0]
    for w in W[1:]:
        S = S.gfmul(k["tau"]) ^ w
    return S


def u32coin(g):
    return int(g.integers(0, M32))


def odd32(g):
    return int(g.integers(0, M32 // 2)) * 2 + 1


def fp(g):
    return int(g.integers(1, P31))


def u64coin(g):
    return int(g.integers(0, M64, dtype=np.uint64))


DESIGNS = {
    "nh": dict(fn=d_nh, coins={}, Ts=(4, 8, 16, 32, 64), what="NH over one accumulator's steps (control: nh_attack)"),
    "poly": dict(fn=d_poly, coins={"tau": (M32, odd32)}, Ts=(4, 16, 64), what="Horner Σ τ^(T−t) w_t mod 2³² (control: linear)"),
    "xor": dict(fn=d_xor, coins={}, Ts=(4, 16, 64), what="⊕ w_t (control: lane-wise)"),
    "brw32": dict(fn=d_brw32, coins={"tau": (M32, odd32)}, Ts=(3, 7, 15, 16, 31), what="BRW_τ over Z/2³²"),
    "brw31": dict(fn=d_brw31, coins={"tau": (P31, fp)}, Ts=(3, 7, 15, 16, 31), what="BRW_τ over F_(2³¹−1)"),
    "pow4": dict(fn=d_pow4, coins={}, Ts=(4, 16, 64), what="Σ w_t⁴ mod 2³² (PowCredit_4)"),
    "sqchain31": dict(fn=d_sqchain31, coins={"s0": (P31, fp)}, Ts=(4, 16, 64), what="S ← (S ⊕ w)² mod 2³¹−1 (SqChainWrites)"),
    "sqadd31": dict(fn=d_sqadd31, coins={"s0": (P31, fp)}, Ts=(1, 2, 4, 16), what="S ← (S + w)² mod 2³¹−1"),
    "hw_chacha_add": dict(fn=d_chacha("add"), hw=True, coins={f"s{i}": (M32, u32coin) for i in range(4)},
                          Ts=(4, 16, 64), what="MAC: a += c; ChaCha quarter-round a k-step"),
    "hw_chacha_xor": dict(fn=d_chacha("xor"), hw=True, coins={f"s{i}": (M32, u32coin) for i in range(4)},
                          Ts=(4, 16, 64), what="MAC: a ^= c; ChaCha quarter-round a k-step"),
    "hw_sha512": dict(fn=d_sha512, hw=True, coins={f"h{i}": (M64, u64coin) for i in range(8)}, Ts=(4, 16, 64),
                      what="MAC: one SHA-512 round a k-step, the word sign-extended as W_t"),
    "sha512": dict(fn=d_sha512, coins={f"h{i}": (M64, u64coin) for i in range(8)}, Ts=(4, 16, 64),
                   what="software: a SHA-512 round a word, W_t the word sign-extended (add entry)"),
    "sha512x": dict(fn=d_sha512x, coins={**{f"h{i}": (M64, u64coin) for i in range(8)},
                                         **{f"c{t}": (M64, u64coin) for t in range(32)}}, Ts=(4, 16, 64),
                    what="software: half a SHA-512 round a word, AddXor entry (R80X's rate)"),
    "hw_sha512x": dict(fn=d_sha512x, hw=True, coins={**{f"h{i}": (M64, u64coin) for i in range(8)},
                                                     **{f"c{t}": (M64, u64coin) for t in range(32)}}, Ts=(4, 16, 64),
                       what="MAC: SI-THEORY's fold, half a SHA-512 round a word, AddXor entry, closed"),
    "brw32x": dict(fn=d_brw32x, coins={"tau": (M32, odd32), **{f"c{t}": (M32, u32coin) for t in range(64)}},
                   Ts=(3, 7, 15, 16, 31), what="BRW_τ over Z/2³², each word XORed with a position constant first"),
    "hw_nh": dict(fn=d_nh, hw=True, coins={}, Ts=(16, 64), what="sketch unit: NH on the accumulator stream"),
    "hw_poly31": dict(fn=d_poly31, hw=True, coins={"tau": (P31, fp)}, Ts=(16, 64),
                      what="sketch unit: Horner mod 2³¹−1 (a field sketch)"),
    "hw_gf32": dict(fn=d_gf32, hw=True, coins={"tau": (M32, u32coin)}, Ts=(16, 64),
                    what="sketch unit: Horner in GF(2³²), XOR entry"),
    "hw_brw31": dict(fn=d_brw31, hw=True, coins={"tau": (P31, fp)}, Ts=(15, 16, 31), what="sketch unit: BRW mod 2³¹−1"),
    "hw_brw32": dict(fn=d_brw32, hw=True, coins={"tau": (M32, odd32)}, Ts=(15, 16, 31),
                     what="sketch unit: BRW over Z/2³² (SI-THEORY's thin in-unit fold)"),
    "hw_xor": dict(fn=d_xor, hw=True, coins={}, Ts=(16, 64), what="sketch unit: ⊕ of the words (thin, XOR3)"),
    "hw_add": dict(fn=d_add, hw=True, coins={}, Ts=(16, 64), what="sketch unit: Σ of the words mod 2³² (thin, IADD3)"),
    "hw_poly": dict(fn=d_poly, hw=True, coins={"tau": (M32, odd32)}, Ts=(16, 64),
                    what="MAC: acc ← τ·acc + c a k-step (linear ρ), on increments"),
    "hw_rho_rot": dict(fn=d_rho("rot"), hw=True, coins={"s0": (M32, u32coin)}, Ts=(16, 64),
                       what="MAC: acc ← rotl(acc, 13) + c a k-step, on increments"),
    "hw_rho_xs": dict(fn=d_rho("xs"), hw=True, coins={"s0": (M32, u32coin)}, Ts=(16, 64),
                      what="MAC: acc ← (acc ⊕ acc ≫ 15) + c a k-step, on increments"),
}


def draw_coins(design, seed):
    g = np.random.default_rng(10_000 + seed)
    return {name: (mod, f(g)) for name, (mod, f) in design["coins"].items()}


def coin_vals(coins, mod_override=None):
    return {n: Val(np.uint64(v % (mod_override or m)), mod_override or m) for n, (m, v) in coins.items()}


def coin_syms(tr, coins):
    return {n: tr.leaf("coin", m, "glob", 0) for n, (m, _) in coins.items()}


# ---------- structure and price

def trace(design, coins, T, fused, with_x=True):
    """The block's program. Honest words are MMA outputs (a `word` leaf each); fused words are x + R_t + C_t (R_t + C_t
    without x), and a sign extension of one distributes over the sum, which `fuse` checks never overflows. `fused` is
    a bool (every word or none) or the set of steps fused."""
    steps = set(range(T)) if fused is True else set(fused or ())
    tr = Trace(dist64=bool(steps))
    x = tr.leaf("x", M32, "static", 1) if steps and with_x else None
    W = []
    for t in range(T):
        if t in steps:
            r = tr.leaf("R", M32, "row", 1)
            W.append((x + r if x is not None else r) + tr.leaf("C", M32, "col", 1))
        else:
            W.append(tr.leaf("word", M32, "cell", 1))
    out = design["fn"](W, coin_syms(tr, coins))
    return tr, out


def reach(tr, root):
    seen, stack = set(), [root.i]
    while stack:
        i = stack.pop()
        if i not in seen:
            seen.add(i)
            stack += [a for k, a in tr.args[i] if k == "n"]
    return seen


def structure(tr, root):
    nodes = reach(tr, root)
    live = [i for i in nodes if tr.deg[i] != 0]
    ops = {tr.op[i] for i in live if tr.args[i]}
    entry = Counter()
    for i in nodes:
        for k, a in tr.args[i]:
            if k == "n" and tr.op[a] == "word":
                entry[tr.op[i]] += 1
    poly = ops <= POLY_OPS and tr.deg[root.i] is not None
    return {"ops": sorted(ops), "poly": poly, "degree": tr.deg[root.i] if poly else None,
            "lanewise": ops <= BIT_OPS, "tfunc": ops <= TFUNC_OPS and ring(tr.mod[root.i]) != "p",
            "entry": dict(entry), "mults": sum(1 for i in live if tr.op[i] in ("mul", "gfmul") and tr.deg[i] and
                                               all(tr.deg[a] for k, a in tr.args[i] if k == "n") and
                                               len([1 for k, a in tr.args[i] if k == "n"]) == 2)}


def price(tr, root, row_ops=2.0, col_ops=4.0):
    """W1 writes a block, by bucket: mma, cell, row, col, static. Add and XOR trees merge single-use children; a cell
    tree's operands are regrouped by class and each non-cell group pre-summed in its own bucket; cell-class products
    feeding an add tree are IMADs that absorb an operand each. Coin-only work is once a call and free."""
    nodes = reach(tr, root)
    cons = {i: [] for i in nodes}
    for i in nodes:
        for k, a in tr.args[i]:
            if k == "n":
                cons[a].append(i)
    b = Counter()
    absorbed = set()

    def bucket(c):
        return None if c in ("const", "glob") else c

    def kind(i):
        o = tr.op[i]
        if o in ("add", "sub", "neg"):
            return "add"
        if o in BIT_OPS:
            return "bit"
        return o

    def merged(i):
        return (len(cons[i]) == 1 and kind(i) in ("add", "bit") and kind(cons[i][0]) == kind(i)
                and tr.mod[cons[i][0]] == tr.mod[i] and ring(tr.mod[i]) != "p")

    def leaves(i, out, consts, ops):
        ops.add(tr.op[i])
        for k, a in tr.args[i]:
            if k == "c":
                consts.append(a)
            elif merged(a):
                leaves(a, out, consts, ops)
            else:
                out.append(a)

    for i in sorted(nodes, reverse=True):
        o, c, r = tr.op[i], tr.cls[i], ring(tr.mod[i])
        if o == "word":
            b["mma"] += 1
        elif o == "R":
            b["row"] += row_ops
        elif o == "C":
            b["col"] += col_ops
        if not tr.args[i] or bucket(c) is None or i in absorbed or merged(i):
            continue
        k = kind(i)
        if k in ("add", "bit") and r != "p":
            lv, consts, ops = [], [], set()
            leaves(i, lv, consts, ops)
            lv = list(dict.fromkeys(lv))
            cells = [a for a in lv if tr.cls[a] == "cell"]
            groups = Counter(tr.cls[a] for a in lv if tr.cls[a] != "cell")
            if consts:
                groups["glob"] += 1
            if c == "cell" and (k == "add" or ops == {"xor"}):
                if "glob" in groups and len(groups) > 1:
                    tgt = next(g for g in ("row", "col", "static") if g in groups)
                    groups[tgt] += groups.pop("glob")
                for g, cnt in groups.items():
                    if bucket(g):
                        b[g] += math.ceil((cnt - 1) / 2) * COST[r][k]
                L = len(cells) + len(groups)
            else:
                L = len(lv) + bool(consts)
            if k == "add":
                muls = [a for a in cells if tr.op[a] in ("mul", "shl") and len(cons[a]) == 1 and tr.cls[a] == c]
                j = min(len(muls), max(0, L - 1)) if L > 1 else 0
                absorbed.update(muls[:j])
                b[bucket(c)] += j * COST[r]["mul"] + max(0, math.ceil((L - j - 1) / 2)) * COST[r]["add"]
            else:
                b[bucket(c)] += max(1, math.ceil((L - 1) / 2)) * COST[r]["bit"]
            continue
        cost = {"add": COST[r]["add"], "mul": COST[r]["mul"], "shl": COST[r]["shift"], "shr": COST[r]["shift"],
                "rotl": COST[r]["shift"], "embed": COST[r]["embed"], "embedu": COST[r]["embed"], "to64": 1,
                "lift": 0, "cat": 0, "gfmul": COST[32]["gfmul"], "bit": COST[r]["bit"]}.get(k, 1)
        b[bucket(c)] += cost
    return dict(b)


def per_s(bk, rows_a_class, once=False, mma=True):
    return (bk.get("mma", 0) * mma + bk.get("cell", 0) + bk.get("row", 0) / CALL_N + bk.get("col", 0) / rows_a_class
            + (0 if once else bk.get("static", 0)))


# ---------- words

def signed32(a):
    a = np.asarray(a).astype(np.int64)
    return np.where(a >= 1 << 31, a - (1 << 32), a)


def increments(p):
    """The k-steps' increments in place of the accumulator words: c_t = w_t − w_(t−1), w_(−1) = x, so x drops out and
    the split is ΔR + ΔC."""
    def diff(a, first):
        return (a - np.concatenate([first, a[..., :-1]], axis=-1)) & np.uint64(M32 - 1)
    q = dict(p)
    x = p.get("d0", np.zeros(p["W"].shape[:2], np.uint64))
    q["W"] = diff(p["W"], x[:, :, None])
    if "R" in p:
        q["R"], q["C"] = diff(p["R"], np.zeros_like(p["R"][:, :1])), diff(p["C"], np.zeros_like(p["C"][..., :1]))
        q["d0"] = np.zeros_like(x)
        q["separable"] = p["separable"] and bool(np.array_equal(
            (q["R"][:, None, :] + q["C"][q["cls"]]) & np.uint64(M32 - 1), q["W"]))
    return q


def family(name, hc, cs, S, seed=0):
    if name.endswith("_inc"):
        p = increments(family(name[:-4], hc, cs, S, seed))
        p["family"] = name
        return p
    if name in ("counter", "counter_x0", "counter_random"):
        p = nh.counter_parts(hc, "lead:rot", "random" if name == "counter_random" else "zero", S, seed=seed)
        p["rows_a_class"] = TILE / len(p["C"])
        p["row_ops"], p["col_ops"] = 2.0, 4.0
        if name == "counter_x0":
            p["W"] = (p["W"] - p["d0"][:, :, None]) & np.uint64(M32 - 1)
            p["d0"] = np.zeros_like(p["d0"])
            p["separable"] = bool(np.array_equal((p["R"][:, None, :] + p["C"][p["cls"]]) & np.uint64(M32 - 1),
                                                 p["W"]))
    elif name == "noise":
        p = nh.noise_parts(cs, S, seed=seed)
        p["row_ops"], p["col_ops"] = 2.0, 4.0
    else:
        g = np.random.default_rng(seed + 77)
        m, n = 64, 32
        if name == "operand":
            A = g.integers(-128, 128, (S, m, 32))
            B = g.integers(-128, 128, (S, n, 32))
            inc = np.einsum("tik,tjk->ijt", A, B)
        else:
            inc = g.integers(0, M32, (m, n, S), dtype=np.uint64).astype(np.int64)
        W = np.cumsum(inc, axis=2) & (M32 - 1)
        p = {"W": nh.u32(W), "separable": False, "stock_frac": 1.0, "rows_a_class": CALL_M, "label": name,
             "emulate_word": 1.0 if name == "operand" else PRG_WORD}
    p["family"] = name
    return p


# ---------- attacks

def submultisets(beta):
    cnt = Counter(beta)
    keys = sorted(cnt)
    for choice in itertools.product(*[range(cnt[t] + 1) for t in keys]):
        gamma = tuple(t for t, c in zip(keys, choice) for _ in range(c))
        coef = 1
        for t, c in zip(keys, choice):
            coef *= (-1) ** (cnt[t] - c) * math.comb(cnt[t], c)
        yield gamma, coef


def binom_obj(v, k):
    out = np.ones(v.shape, dtype=object)
    vo = v.astype(object)
    for q in range(k):
        out = out * (vo - q)
    return out // math.factorial(k)


class Newton:
    """Δ₁^e Δ^β P at a set of row walks, from black-box evaluations of the block, cached by offset."""

    def __init__(self, design, coins, rows, mod, chunk=1500):
        self.design, self.cv, self.rows, self.mod, self.chunk = design, coin_vals(coins), rows, mod, chunk
        self.cache = {}

    def f(self, pts):
        T = pts.shape[-1]
        W = [Val(nh.u32(pts[:, t]), M32) for t in range(T)]
        out = self.design["fn"](W, self.cv)
        return np.broadcast_to(out.a if isinstance(out, Val) else np.uint64(out), (pts.shape[0],)).astype(np.int64)

    def need(self, keys):
        todo = [k for k in dict.fromkeys(keys) if k not in self.cache]
        nR, T = self.rows.shape
        for s in range(0, len(todo), self.chunk):
            part = todo[s:s + self.chunk]
            pts = np.repeat(self.rows[None], len(part), 0)
            for q, (gamma, e) in enumerate(part):
                for t in gamma:
                    pts[q, :, t] += 1
                pts[q] += e
            vals = self.f(pts.reshape(-1, T)).reshape(len(part), nR)
            for q, key in enumerate(part):
                self.cache[key] = vals[q]

    def h(self, e, beta):
        terms = [((gamma, kk), coef * (-1) ** (e - kk) * math.comb(e, kk))
                 for kk in range(e + 1) for gamma, coef in submultisets(beta)]
        self.need([k for k, _ in terms])
        acc = np.zeros(self.rows.shape[0], dtype=object)
        for key, c in terms:
            acc = acc + self.cache[key].astype(object) * c
        return np.array([int(v) % self.mod for v in acc], dtype=np.uint64), [k for k, _ in terms]


def decomp(design, coins, parts, T, deg, mod, fold_cost, rng, mutate=None):
    """The Newton decomposition over every block of the instance: (exact, price every S, price once a call, info)."""
    x = signed32(parts["d0"])
    R, C, cls = signed32(parts["R"]), signed32(parts["C"]), parts["cls"]
    m, n = x.shape
    Sn = R.shape[1]
    W = parts["W"]
    rpc = parts["rows_a_class"]
    exact, worst_every, worst_once, info = True, 0.0, 0.0, {}
    for t0 in range(0, Sn - T + 1, T):
        Rb, Cb = R[:, t0:t0 + T], C[:, :, t0:t0 + T]
        lo, hi = int(Rb.min()), int(Rb.max()) + 1
        rows = np.vstack([Rb, rng.integers(lo, hi, (8, T))])
        nw = Newton(design, coins, rows, mod)
        feats = {}
        level = [()]
        nz0 = {(): True}
        for a in range(1, deg + 1):
            nxt = []
            for beta in level:
                for t in range(beta[-1] if beta else 0, T):
                    nxt.append(beta + (t,))
            level = []
            for beta in nxt:
                h, keys = nw.h(0, beta)
                if np.any(h):
                    feats[(0, beta)] = (h, keys)
                    level.append(beta)
            if not level:
                break
        h0, k0 = nw.h(0, ())
        feats[(0, ())] = (h0, k0)
        for e in range(1, deg + 1):
            for (e1, beta) in [k for k in feats if k[0] == e - 1]:
                if len(beta) + e > deg:
                    continue
                h, keys = nw.h(e, beta)
                if np.any(h):
                    feats[(e, beta)] = (h, keys)
        if mutate == "coef":
            k = next(k for k in feats if k[1])
            feats[k] = (feats[k][0].copy(), feats[k][1])
            feats[k][0][0] = (feats[k][0][0] + np.uint64(1)) % np.uint64(mod)
        Cuse = Cb.copy()
        if mutate == "classes":
            Cuse[:] = Cuse[0]
        xuse = x ^ 1 if mutate == "x" else x
        S_att = np.zeros((m, n), dtype=np.uint64)
        cell_cost, static_cost, static_once, col_cost, keys_used = 0.0, 0.0, 0.0, 0.0, set()
        K_e = {}
        for e in range(deg + 1):
            mine = {beta: v for (ee, beta), v in feats.items() if ee == e}
            if not mine:
                continue
            for _, keys in mine.values():
                keys_used.update(keys)
            row = mine.get((), (np.zeros(len(rows), np.uint64), []))[0][:m]
            const = {b: v[0][0] for b, v in mine.items() if b and np.all(v[0] == v[0][0])}
            mixed = [b for b in mine if b and b not in const]
            G = {}
            for beta in list(const) + mixed:
                g = np.ones(Cuse.shape[:2], dtype=object)
                for t, kk in Counter(beta).items():
                    g = g * binom_obj(Cuse[:, :, t], kk)
                G[beta] = np.array([[int(v) % mod for v in r_] for r_ in g], dtype=np.uint64)
                col_cost += len(beta)
            colv = np.zeros(Cuse.shape[:2], dtype=np.uint64)
            for beta, hv in const.items():
                colv = (colv + G[beta] * np.uint64(hv)) % np.uint64(mod)
            col_cost += len(const)
            V = (row[:, None] + colv[cls]) % np.uint64(mod)
            chains = 0.0
            if mixed:
                H = np.stack([mine[b][0][:m] for b in mixed], 1)
                Gm = np.stack([G[b] for b in mixed], 2)
                for c in range(len(Cuse)):
                    rr = np.flatnonzero(cls == c)
                    if mod == M32:
                        part, cost = nh.imma(H[rr], Gm[c])
                    else:
                        part = ((H[rr][:, None, :] * Gm[c][None, :, :]) % np.uint64(mod)).sum(2) % np.uint64(mod)
                        K = len(mixed)
                        cost = min(16 * math.ceil(K / 32) + 17, 4 * K + 3)
                    chains = max(chains, cost)
                    V[rr] = (V[rr] + part) % np.uint64(mod)
            K_e[e] = len(mixed)
            terms = (np.any(row) and 1) + (len(const) > 0) + (len(mixed) > 0)
            add, mul = (1, 1) if mod == M32 else (2, 7)
            invariant = not mixed and not const and () in mine
            bx = binom_obj(xuse, e)
            bxm = np.array([[int(v) % mod for v in r_] for r_ in bx], dtype=np.uint64)
            S_att = (S_att + bxm * V) % np.uint64(mod)
            here = chains + max(0, math.ceil((terms - 1) / 2)) * add + (mul if e else add)
            if invariant and np.all(mine[()][0] == mine[()][0][0]):
                static_cost += here + (e - 1 if e > 1 else 0) * mul
            else:
                cell_cost += here
                static_cost += (e - 1 if e > 1 else 0) * mul
        honest_b = design["fn"]([Val(W[:, :, t0 + t], M32) for t in range(T)], coin_vals(coins))
        exact &= bool(np.array_equal(S_att, np.broadcast_to(honest_b.a, S_att.shape)))
        row_cost = len(keys_used) * fold_cost + T * parts["row_ops"]
        colc = col_cost + T * parts["col_ops"]
        every = cell_cost + static_cost + row_cost / CALL_N + colc / rpc
        once = cell_cost + row_cost / CALL_N + colc / rpc
        worst_every, worst_once = max(worst_every, every), max(worst_once, once)
        info = {"K_mixed": {str(k): v for k, v in K_e.items()}, "points": len(keys_used),
                "features": len(feats), "cell": round(cell_cost, 3), "static": round(static_cost, 3)}
    return exact, worst_every, worst_once, info


def fuse(design, coins, parts, T, rpc):
    """No word formed: the fused trace's price, and its S from x + R + C, checked against honest's words."""
    with_x = bool(np.any(parts["d0"]))
    tr, out = trace(design, coins, T, fused=True, with_x=with_x)
    bk = price(tr, out, parts["row_ops"], parts["col_ops"])
    exact = True
    if "to64" in tr.op:
        R, C = signed32(parts["R"])[:, None, :], signed32(parts["C"][parts["cls"]])
        exact = bool(np.array_equal(signed32(parts["d0"])[:, :, None] + R + C, signed32(parts["W"])))
    for t0 in range(0, parts["R"].shape[1] - T + 1, T):
        Ws = [Val(parts["d0"] + parts["R"][:, None, t0 + t] + parts["C"][parts["cls"], :, t0 + t], M32)
              for t in range(T)]
        a = design["fn"](Ws, coin_vals(coins))
        h = design["fn"]([Val(parts["W"][:, :, t0 + t], M32) for t in range(T)], coin_vals(coins))
        exact &= bool(np.array_equal(a.a, h.a))
    return exact, per_s(bk, rpc), per_s(bk, rpc, once=True), bk


def mix(design, coins, parts, T, rpc, exact):
    """Shortcut 2 word by word (the edge attack among them): fuse each step whose fusing alone saves writes, form the
    rest by MMA, and price the combined program. Its S is `fuse`'s, so it is exact when `fuse` is."""
    def cost(steps, once=True):
        tr, out = trace(design, coins, T, steps, with_x=bool(np.any(parts["d0"])))
        return per_s(price(tr, out, parts["row_ops"], parts["col_ops"]), rpc, once=once)
    base = cost(())
    steps = [t for t in range(T) if cost({t}) < base]
    return exact, cost(steps, once=False), cost(steps), steps


def pack(design, coins, W, T, narrow_by=0):
    """A lane-wise design on b-bit lanes, L = ⌊32g/b⌋ cells in g registers, and the bits above b bitsliced from the
    sign bits, 32 cells a register, once a distinct column of coin bits. A lane's word is formed biased by 2^(b−1), so
    no lane borrows, and a LOP3 a register-step removes the bias; in an XOR-only design the biases cancel and are
    corrected once at the end. Returns (exact, price an S, b, L)."""
    m, n, Sn = W.shape
    Wi = signed32(W)
    b = int(np.abs(Wi).max()).bit_length() + 1 - narrow_by
    cv = coin_vals(coins)
    tr, out = trace(design, coins, T, fused=False)
    step = price(tr, out).get("cell", 0) / T
    pats = {}
    for pos in range(b, 32):
        pats.setdefault(tuple((int(v.a) >> pos) & 1 for v in cv.values()), []).append(pos)
    form = 1 if structure(tr, out)["ops"] == ["xor"] else 2
    best = None
    for g in (1, 2):
        L = 32 * g // b
        if L >= 2:
            cost = g * T * (form + step) / L + 2 + len(pats) * T * (step + 1) / 32
            if best is None or cost < best[0]:
                best = (cost, g, L)
    if best is None:
        return None
    cost, g, L = best
    mod = M32 if g == 1 else M64
    lane = (1 << b) - 1
    rep = {k: Val(np.uint64(sum((int(v.a) & lane) << (a * b) for a in range(L))), mod) for k, v in cv.items()}
    exact = True
    for t0 in range(0, Sn - T + 1, T):
        blk = Wi[:, :, t0:t0 + T]
        ref = design["fn"]([Val(W[:, :, t0 + t], M32) for t in range(T)], cv).a
        got = np.zeros((m, n), dtype=np.uint64)
        for g0 in range(0, n, L):
            cols = list(range(g0, min(n, g0 + L)))
            packed = [Val(sum(((blk[:, j, t] & lane).astype(np.uint64) << np.uint64(a * b)) for a, j in enumerate(cols)),
                          mod) for t in range(T)]
            res = design["fn"](packed, rep).a
            for a, j in enumerate(cols):
                got[:, j] = (res >> np.uint64(a * b)) & np.uint64(lane)
        signs = ((blk >> (b - 1)) & 1).astype(np.uint64)
        for pat, positions in pats.items():
            ck = {k: Val(M32 - 1 if bit else 0, M32) for k, bit in zip(cv, pat)}
            res = design["fn"]([Val(signs[:, :, t] * np.uint64(M32 - 1), M32) for t in range(T)], ck).a & np.uint64(1)
            for pos in positions:
                got |= res << np.uint64(pos)
        exact &= bool(np.array_equal(got, ref))
    return exact, cost, b, L


def tfunc_check(design, coins, W, T, bits=16):
    cvb = {k: Val(np.uint64(v % (1 << bits)), 1 << bits) for k, (mod, v) in coins.items() if mod & (mod - 1) == 0}
    if len(cvb) != len(coins):
        return None
    ok = True
    for t0 in range(0, W.shape[2] - T + 1, T):
        lo = design["fn"]([Val(W[:, :, t0 + t] & np.uint64((1 << bits) - 1), 1 << bits) for t in range(T)], cvb).a
        hi = design["fn"]([Val(W[:, :, t0 + t], M32) for t in range(T)], coin_vals(coins)).a
        ok &= bool(np.array_equal(lo, hi & np.uint64((1 << bits) - 1)))
    return ok


def gaps(honest, cheapest, stock):
    return {"gap_claimed": round(max(0.0, honest - cheapest) / stock, 6),
            "gap_share": round(max(0.0, stock - cheapest) / stock, 6)}


def evaluate(name, design, parts, seed=0, rng=None):
    rows = []
    rng = rng or np.random.default_rng(seed)
    hw = design.get("hw", False)
    for T in design["Ts"]:
        if T > parts["W"].shape[2]:
            continue
        coins = draw_coins(design, seed)
        tr, out = trace(design, coins, T, fused=False)
        st = structure(tr, out)
        bk = price(tr, out)
        rpc = parts["rows_a_class"]
        sw = per_s(bk, rpc, mma=False)
        honest = T + (0 if hw else sw)
        stock = parts["stock_frac"] * T
        mod = tr.mod[out.i]
        row = {"design": name, "family": parts["family"], "T": T, "G": T, "honest": round(honest, 3),
               "stock": round(stock, 3), "software": round(sw, 3), "structure": st, "attacks": {}}
        cands, cands_once = [honest], [honest]
        if parts.get("separable"):
            ok, ev, on, bkf = fuse(design, coins, parts, T, rpc)
            row["attacks"]["fuse"] = {"exact": ok, "every": round(ev, 3), "once": round(on, 3)}
            cands.append(ev), cands_once.append(on)
            if not hw:
                ok, mev, mon, steps = mix(design, coins, parts, T, rpc, ok)
                if steps and len(steps) < T:
                    row["attacks"]["mix"] = {"exact": ok, "every": round(mev, 3), "once": round(mon, 3),
                                             "fused_steps": steps}
                    cands.append(mev), cands_once.append(mon)
            if st["poly"] and st["degree"] and st["degree"] <= DMAX:
                ok, ev, on, info = decomp(design, coins, parts, T, st["degree"], mod, sw, rng)
                row["attacks"]["decomp"] = {"exact": ok, "every": round(ev, 3), "once": round(on, 3), **info}
                cands.append(ev), cands_once.append(on)
        elif "emulate_word" in parts:
            em = T * parts["emulate_word"] + sw
            row["attacks"]["emulate"] = {"exact": True, "every": round(em, 3), "once": round(em, 3)}
            cands.append(em), cands_once.append(em)
        if st["lanewise"]:
            pk = pack(design, coins, parts["W"], T)
            if pk:
                row["attacks"]["pack"] = {"exact": pk[0], "every": round(pk[1], 3), "once": round(pk[1], 3),
                                          "b": pk[2], "L": pk[3]}
                cands.append(pk[1]), cands_once.append(pk[1])
        if st["tfunc"]:
            row["tfunc_verified"] = tfunc_check(design, coins, parts["W"], T)
        row["cheapest"] = round(min(cands), 3)
        row["cheapest_once"] = round(min(cands_once), 3)
        row.update(gaps(honest, min(cands), stock))
        row["if_d0_once"] = gaps(honest, min(cands_once), stock)
        rows.append(row)
    return rows


def frontier(rows):
    out = {}
    for r in rows:
        key = f"{r['design']} {r['family']}"
        out.setdefault(key, {"T1_holds_to_G": 0, "first_G_T1_fails": None, "stock_only_holds_to_G": 0,
                             "_broken": False, "_sbroken": False})
        f = out[key]
        if r["if_d0_once"]["gap_claimed"] > T1:
            f["_broken"] = True
            f["first_G_T1_fails"] = f["first_G_T1_fails"] or r["G"]
        elif not f["_broken"]:
            f["T1_holds_to_G"] = r["G"]
        if r["if_d0_once"]["gap_share"] > T1:
            f["_sbroken"] = True
        elif not f["_sbroken"]:
            f["stock_only_holds_to_G"] = r["G"]
    for f in out.values():
        f.pop("_broken"), f.pop("_sbroken")
    return out


def brw_ref(ms, tau, mod):
    """BRW from Bernstein's definition over Python integers."""
    def go(v):
        n = len(v)
        if n == 0:
            return 0
        if n == 1:
            return v[0] % mod
        if n == 2:
            return (v[0] * tau + v[1]) % mod
        if n == 3:
            return ((tau + v[0]) * (tau * tau + v[1]) + v[2]) % mod
        k = n.bit_length() - 1
        return (go(v[:2 ** k - 1]) * (pow(tau, 2 ** k, mod) + v[2 ** k - 1]) + go(v[2 ** k:])) % mod
    return go(ms)


def main() -> int:
    np.seterr(over="ignore")
    checks = Checks()
    opts = dict(a[2:].split("=", 1) for a in checks.args if a.startswith("--") and "=" in a)
    args = [a for a in checks.args if not a.startswith("--")]
    out_path = args[0] if args else "red.json"
    sys.path.insert(0, os.path.abspath(opts.get("lib", ".")))
    import commit_space as cs
    import hash_counter as hc

    S = int(opts.get("steps", 64))
    names = opts["designs"].split(",") if "designs" in opts else list(DESIGNS)
    fams = opts.get("families", "counter,counter_x0,noise,counter_random,operand,cell,counter_inc,noise_inc,"
                                "operand_inc").split(",")
    parts = {f: family(f, hc, cs, S) for f in fams}
    for f, p in parts.items():
        if f in ("counter", "counter_x0", "noise", "counter_inc", "noise_inc"):
            checks.check(f"{f}: the split reproduces the words", p["separable"])

    g = np.random.default_rng(5)
    for mod, tau in ((M32, 0x9E3779B1), (P31, 123456789)):
        for n in (1, 2, 3, 4, 7, 9, 15, 16, 31):
            ms = [int(v) for v in g.integers(0, P31, n)]
            tp = tau_pows(Val(np.uint64(tau), mod), n)
            got = int(brw([Val(np.uint64(v), mod) for v in ms], tp).a) if n else 0
            if got != brw_ref(ms, tau, mod):
                checks.check(f"BRW n {n} mod {mod} matches Bernstein's definition", False)
                break
        else:
            checks.check(f"BRW matches Bernstein's definition at n 1–31 mod {'2³²' if mod == M32 else '2³¹−1'}", True)
    for T in (3, 7, 15, 16, 31):
        tr, out = trace(DESIGNS["brw32"], draw_coins(DESIGNS["brw32"], 0), T, False)
        st = structure(tr, out)
        if st["mults"] != T // 2 or st["degree"] != T.bit_length():
            checks.check(f"BRW T {T}: ⌊T/2⌋ word products and degree", False, st)
            break
    else:
        checks.check("BRW over T 3–31 words: ⌊T/2⌋ products of words, degree ⌊lg T⌋ + 1", True)
    checks.check("SHA-512's first round constant", SHA512_K[0] == 0x428A2F98D728AE22, hex(SHA512_K[0]))
    gm = Val(np.uint64(0x80000000), M32).gfmul(Val(np.uint64(2), M32))
    checks.check("GF(2³²): x³¹·x = x⁷ + x³ + x² + 1", int(gm.a) == 0b10001101, hex(int(gm.a)))

    rows = []
    for name in names:
        for f, p in parts.items():
            rs = evaluate(name, DESIGNS[name], p)
            rows += rs
            for r in rs:
                a = {k: v["once"] for k, v in r["attacks"].items()}
                print(f"{name:14s} {f:15s} T{r['T']:>3} honest {r['honest']:8.2f} stock {r['stock']:7.2f} "
                      f"cheapest {r['cheapest_once']:8.2f} share {r['if_d0_once']['gap_share']:.3f} "
                      f"claimed {r['if_d0_once']['gap_claimed']:.3f} {a} deg {r['structure']['degree']} "
                      f"entry {r['structure']['entry']}", flush=True)
    bad = [(r["design"], r["family"], r["T"], k) for r in rows for k, v in r["attacks"].items() if not v["exact"]]
    checks.check("every attack's S matches honest's bit for bit", not bad, bad[:5] or len(rows))
    summary = frontier(rows)

    if "nh" in names and "counter" in parts:
        p = parts["counter"]
        for T in (16, 64):
            if T <= S:
                ref = nh.honest_nh(p["W"], T)
                hand = nh.attack_nh(p, T, d0_free=True)
                gen = next(r for r in rows if r["design"] == "nh" and r["family"] == "counter" and r["T"] == T)
                checks.check(f"nh T {T}: nh_attack's hand decomposition is exact, and the generic one prices within "
                             f"2× of it", np.array_equal(hand[0], ref) and gen["attacks"]["decomp"]["once"]
                             <= 2 * hand[1]["total"] + 1,
                             f"generic {gen['attacks']['decomp']['once']}, hand {round(hand[1]['total'], 3)}")
    if "xor" in names and "counter" in parts:
        xs = [r for r in rows if r["design"] == "xor" and r["family"] == "counter"]
        checks.check("control xor: pack runs on the lead's words, exact", xs and all(
            r["attacks"].get("pack", {}).get("exact") for r in xs), [r["attacks"].get("pack") for r in xs])
    for ctl, fam in (("poly", "counter"), ("nh", "counter")):
        if ctl in names and fam in parts:
            big = [r for r in rows if r["design"] == ctl and r["family"] == fam and r["T"] == max(DESIGNS[ctl]["Ts"])]
            checks.check(f"control {ctl}: folds on the lead's words at G {max(DESIGNS[ctl]['Ts'])} (gap_share > 1%)",
                         big and big[0]["if_d0_once"]["gap_share"] > T1,
                         big and {k: v["once"] for k, v in big[0]["attacks"].items()})

    if checks.mutations:
        p = parts.get("counter")
        dz = DESIGNS["brw32"]
        coins = draw_coins(dz, 0)
        tr, out = trace(dz, coins, 7, False)
        sw = per_s(price(tr, out), p["rows_a_class"], mma=False)
        for mut, what in (("classes", "every row class given class 0's columns"), ("x", "x off in bit 0"),
                          ("coef", "one mixed coefficient off by one")):
            ok = decomp(dz, coins, p, 7, 3, M32, sw, np.random.default_rng(1), mutate=mut)[0]
            checks.mutation(f"BRW decomposition with {what}", not ok)
        bad = dict(coins, tau=(M32, coins["tau"][1] + 2))
        W = p["W"]
        a = dz["fn"]([Val(W[:, :, t], M32) for t in range(7)], coin_vals(bad)).a
        b = dz["fn"]([Val(W[:, :, t], M32) for t in range(7)], coin_vals(coins)).a
        checks.mutation("BRW with τ off by two", not np.array_equal(a, b))
        pk = pack(DESIGNS["xor"], {}, p["W"], 16, narrow_by=2)
        checks.mutation("pack with lanes two bits too narrow", pk is None or not pk[0])
        sz = DESIGNS["sha512"]
        ok = fuse(sz, draw_coins(sz, 0), dict(p, d0=np.zeros_like(p["d0"])), 4, p["rows_a_class"])[0]
        checks.mutation("fuse on the lead's words with the per-cell offset x dropped", not ok)

    json.dump({"schema": "si-red/v1", "steps": S, "rows": rows, "frontier": summary}, open(out_path, "w"), indent=1,
              default=str)
    for k, v in summary.items():
        print(f"frontier {k:32s} {v}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
