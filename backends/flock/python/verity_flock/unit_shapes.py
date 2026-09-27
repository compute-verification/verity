"""The proof-unit shape census: every proof unit of the no-recompute partition (``verity_vllm.query.word``,
``Q_word_v1{X=16,W=32}``) of a program graph, measured as one Flock unit table, and the cost of padding them all to one
public shape S = (R, Z, D).

Per unit (one representative per class, classes as ``word.unit_rule`` groups them plus the primitive multiset and the
values read):

* **R**: rows of its ``flock-ir-unit/v2`` layout (:func:`ir_lower.layout`: the input region, one row per live AND, the
  output region, the constant row), padded to the next power of two with at least one spare row;
* **Z**: nonzeros of its constraint matrices A and B, the input identity rows and the constant row excluded (the private
  nonzeros the recursion's matrix fold reads; the RoPE pair unit has 157,841);
* **D**: committed values it reads or writes (serving-commitment openings); a gather is a read of the one gathered value at
  a private position, not of its whole table;
* **l**: MUFU table lookups, modelled as public lookup slots (PR #83's ``LookupNet``) joined to the unit by glue: the table
  is not in Z.

The unit circuit is composed from per-primitive pieces (:data:`ir_lower.PIECES`, plus the :data:`SURROGATES` below for
primitives C-Flock has no piece for yet, flagged ``estimated``), tracking only the size of each wire's linear form: a
piece's rows cost their own entries plus, for every reference to an input bit, the size of that bit's form in the unit;
a form past :data:`REFRESH` terms is committed. This equals the exact composition up to XOR cancellations across pieces
(:meth:`CallUnits.exact` checks it on small units).

:func:`pack_best` splits a unit that does not fit a shape: its computing gates in a topological order (depth-first from the
outputs, or oldest-operand-first), greedily, every value crossing between pieces committed (an extra serving commitment,
e.g. a GEMM's FP32 carry at a K-chunk boundary). :func:`assess` prices one shape over a program's units.

    python -m verity_flock.unit_shapes --program-graphs DIR --out OUT_DIR [--pickle] [--rows ROW ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fp
from . import ir_lower as IL
from . import unit as TCU
from .gf2 import NOT, OR, ONE, ZERO, Circuit, add, and_reduce, const_bits, multiply, mux, or_reduce, sub

X_BITS, W_BITS = 16, 32
WORD = IL.WORD

# --- surrogate pieces (estimates) -------------------------------------------------------------------------------------


def _key(x):
    s = x[31]
    return [b ^ s for b in x[:31]] + [NOT(s)]


def _is_nan(C, x):
    return C.AND(and_reduce(C, x[23:31]), or_reduce(C, x[:23]))


def _f32_lt(C, a, b):
    return sub(C, _key(a), _key(b), 32)[1]


def _f32_max(C, a, b):
    pick_b = C.AND(_f32_lt(C, a, b), NOT(_is_nan(C, b)))
    return mux(C, pick_b, b, a)


def _f32_min(C, a, b):
    return _f32_max(C, b, a)


def _eq(C, a, b):
    return [NOT(or_reduce(C, [x ^ y for x, y in zip(a, b)]))]


def _f32_sat(C, a):
    one = const_bits(0x3F800000, 32)
    lo = mux(C, C.AND(a[31], NOT(_is_nan(C, a))), const_bits(0, 32), a)
    return mux(C, _f32_lt(C, one, lo), one, lo)


def _i32_le(C, a, b):
    d, borrow = sub(C, b + [b[31]], a + [a[31]], 33)
    return [NOT(d[32])]


def _f32_div(C, a, b):
    """IEEE divide surrogate: 26 restoring steps on the significands, exponent difference, sign; no rounding logic."""
    if all(w.is_const() for w in b):
        try:
            return IL._f32_div(C, a, b)
        except NotImplementedError:
            pass
    ma, mb = a[:23] + [ONE], b[:23] + [ONE]
    r, q = ma + [ZERO], []
    for _ in range(26):
        d, borrow = sub(C, r, mb + [ZERO], 25)
        q.append(NOT(borrow))
        r = [ZERO] + mux(C, NOT(borrow), d, r)[:24]
    e = sub(C, a[23:31] + [ZERO], b[23:31] + [ZERO], 9)[0]
    return list(reversed(q))[:23] + e[:8] + [a[31] ^ b[31]]


def _mufu(index_bits: int):
    """A MUFU op: the index is wiring of the operand, the table value comes from a public lookup slot (glue), the post
    stage adjusts the exponent and selects the special cases (``tail.rsq_post``'s shape)."""
    def piece(C, x):
        v = C.input(f"lookup{len(C.inputs)}", 31)
        C.lookups.append((index_bits, x[:index_bits]))
        e = sub(C, list(v[23:31]) + [ZERO], x[23:31] + [ZERO], 9)[0]
        body = list(v[:23]) + e[:8] + [ZERO]
        s, e_max, e_zero = x[31], and_reduce(C, x[23:31]), NOT(or_reduce(C, x[23:31]))
        out = mux(C, e_zero, const_bits(0x7F800000, 31) + [s], body)
        out = mux(C, e_max, const_bits(0x7FFFFFFF, 32), out)
        return mux(C, s, const_bits(0x7FFFFFFF, 32), out)
    return piece


def _ex2_pre(C, x):
    m = x[:23] + [ONE]
    for i, b in enumerate(x[23:28]):
        m = mux(C, b, [ZERO] * (1 << i) + m[: 24 - (1 << i)], m)
    return m[:23] + x[23:32]


def _mufu_ex2(C, x):
    return _mufu(23)(C, _ex2_pre(C, x))


def _div_scale_a(C, a, b):
    big = _f32_lt(C, const_bits(0x7E800000, 32), b[:31] + [ZERO])
    tiny = _f32_lt(C, b[:31] + [ZERO], const_bits(0x00800000, 32))
    ea = a[23:31]
    down = sub(C, ea, const_bits(2, 8), 8)[0]
    up = add(C, ea, const_bits(24, 8), width=8)
    e = mux(C, big, down, mux(C, tiny, up, ea))
    return a[:23] + e + [a[31]]


def _inv_sum(C, s):
    r = _mufu(23)(C, s)
    zero_or_nan = OR(C, NOT(or_reduce(C, s[:31])), _is_nan(C, s))
    return mux(C, zero_or_nan, const_bits(0x3F800000, 32), fp.f32_fma(C, r, s, r))


def _guard_neg_inf(C, m):
    return [C.AND(NOT(_eq(C, m, const_bits(0xFF800000, 32))[0]), b) for b in m]


def _tanh(C, x):
    """RN32(tanh(x)): of a bf16-widened operand (low 16 bits constant) a 16-bit table lookup, as SiLU's; else 8 FMAs."""
    if all(b.is_const() for b in x[:16]):
        return fp.lookup(C, list(x[16:]), [v | (v ^ 0x5A5A) << 16 for v in IL.silu_table()], 32)
    y = x
    for _ in range(8):
        y = fp.f32_fma(C, y, x, const_bits(0x3F000000, 32))
    return y


def _gelu_tanh_mul(C, g, u):
    """bf16(gelu_tanh(g)) is a function of the 16-bit g: lowered as SiLU·mul is (a 16-bit table, then the multiply)."""
    return IL._silu_mul(C, g, u)


def _to_e4m3_sat(C, x):
    m, e, s = x[:23], x[23:31], x[31]
    r = add(C, m[20:23], [ZERO], cin=OR(C, m[19], or_reduce(C, m[:19])), width=4)
    eb = sub(C, e, const_bits(120, 8), 8)[0]
    over = _f32_lt(C, const_bits(0x43E00000, 32), x[:31] + [ZERO])
    out = r[:3] + eb[:4] + [s]
    return mux(C, over, const_bits(0x7E, 7) + [s], out)


def _e4m3_dot32(C, acc, *ab):
    """Hopper QGMMA k32 step, finite-only (``unit.unit`` on E4M3, groups (32,), W 14, F -139): the total-semantics special
    handling is left out (about 5 % of a BF16 step)."""
    fmt, W, F = TCU.E4M3, 14, -139
    a, b = ab[:32], ab[32:]
    rescale = W - 24
    prods = []
    for x, y in zip(a, b):
        sa, ea, siga, nza = fp._tc_operand(C, fmt, x)
        sb, eb, sigb, nzb = fp._tc_operand(C, fmt, y)
        S = add(C, ea, eb, width=fmt.exp_bits + 1)
        S = add(C, S, const_bits((fmt.offset + TCU.S_BIAS) % 1024, 10), width=10)
        mag = multiply(C, siga, sigb)[: 2 * fmt.sig_bits]
        sh = fmt.product_shift + rescale
        prods.append((sa ^ sb, S, C.AND(nza, nzb), ([ZERO] * sh + mag) if sh >= 0 else mag[-sh:]))
    cm, cf, cs = acc[:23], acc[23:31], acc[31]
    hc = or_reduce(C, cf)
    cf_eff = [cf[0] ^ C.AND(NOT(cf[0]), NOT(hc))] + cf[1:]
    mag24 = cm + [hc]
    state = (cs, add(C, cf_eff, const_bits(127, 10), width=10), OR(C, hc, or_reduce(C, cm)),
             ([ZERO] * rescale + mag24) if rescale >= 0 else mag24[-rescale:])
    s_o, S_o, nz_o, m24 = TCU.group_sum(C, [state] + prods, W, F)
    field = add(C, S_o[:8], const_bits((-128) % 256, 8), cin=m24[23], width=8)
    return m24[:23] + [C.AND(nz_o, x) for x in field] + [C.AND(s_o, nz_o)]


def _philox(C, words, rounds=10):
    """Philox4x32's shape: per round two 32x32 -> 64-bit products by the round constants and the XORs."""
    x = list(words)
    for _ in range(rounds):
        for i, m in ((0, 0xD2511F53), (2, 0xCD9E8D57)):
            p = multiply(C, x[i], const_bits(m, 32))
            x[i], x[i + 1] = [a ^ b for a, b in zip(p[32:64], x[i + 1])], p[:32]
    return x


def _gumbel_key(C, seed, pos):
    return _philox(C, [seed[:32], seed[32:], pos[:32], pos[32:]])[0]


def _gumbel_lane(C, key, v):
    u = _philox(C, [key, v, const_bits(0, 32), const_bits(0, 32)])[0]
    y = u
    for _ in range(16):
        y = fp.f32_fma(C, y, u, const_bits(0x3F000000, 32))
    return y


#: primitive -> (piece, note); estimates, not the primitive's exact circuit (the Hopper BF16 step is exact: PR #84)
SURROGATES = {
    "HopperBF16WgmmaDot16_v1": (lambda C, acc, *ab: fp.tc_dot16(C, acc, list(ab[:16]), list(ab[16:]), groups=(16,), W=26, F=-133), "exact"),
    "HopperE4m3QgmmaDot32_v1": (_e4m3_dot32, "finite-only k32 step"),
    "F32MulFtz_v1": (fp.f32_mul, "as F32Mul"), "F32AddFtz_v1": (fp.f32_add, "as F32Add"),
    "F32SubFtz_v1": (fp.f32_sub, "as F32Sub"), "F32Sub_v1": (fp.f32_sub, "exact piece"),
    "F32Neg_v1": (lambda C, a: fp.f32_neg(a), "exact"), "F32Fabs_v1": (lambda C, a: a[:31] + [ZERO], "exact"),
    "F32BitsShl23_v1": (lambda C, a: [ZERO] * 23 + a[:9], "exact"),
    "F32FmaFtz_v1": (fp.f32_fma, "as F32Fma"), "F32FmaRm_v1": (fp.f32_fma, "as F32Fma"),
    "F32FmaSubFtz_v1": (lambda C, a, b, c: fp.f32_fma(C, a, b, fp.f32_neg(c)), "as F32Fma"),
    "F32Max_v1": (_f32_max, "compare + select"), "F32Fmaxf_v1": (_f32_max, "compare + select"),
    "F32Fminf_v1": (_f32_min, "compare + select"),
    "F32GtStrict_v1": (lambda C, a, b: [_f32_lt(C, b, a)], "compare"),
    "Bf16GtStrict_v1": (lambda C, a, b: [_f32_lt(C, [ZERO] * 16 + b, [ZERO] * 16 + a)], "compare"),
    "F32Eq_v1": (lambda C, a, b: _eq(C, a, b), "equality"), "I32Eq_v1": (lambda C, a, b: _eq(C, a, b), "exact"),
    "I32Le_v1": (_i32_le, "exact"), "I32Add_v1": (lambda C, a, b: add(C, a, b, width=32), "exact"),
    "F32IsFinite_v1": (lambda C, a: [NOT(and_reduce(C, a[23:31]))], "exact"), "F32Sat_v1": (_f32_sat, "compare + select"),
    "SelectF32_v1": (lambda C, c, a, b: mux(C, c[0], a, b), "exact"), "SelectBf16_v1": (lambda C, c, a, b: mux(C, c[0], a, b), "exact"),
    "SelectI32_v1": (lambda C, c, a, b: mux(C, c[0], a, b), "exact"),
    "BitNot_v1": (lambda C, a: [NOT(x) for x in a], "exact"), "BitOr_v1": (lambda C, a, b: [OR(C, x, y) for x, y in zip(a, b)], "exact"),
    "F32Div_v1": (_f32_div, "restoring division"), "DivFullScaleA_v1": (_div_scale_a, "range tests + exponent"),
    "MufuEx2Ftz_v1": (_mufu_ex2, "lookup slot (ex2, 23 b) + shift + post"), "MufuTanh_v1": (_mufu(23), "lookup slot + post"),
    "MufuSqrtFtz_v1": (_mufu(24), "lookup slot (sqrt, 24 b) + post"), "RsqrtApprox_v1": (_mufu(24), "lookup slot (rsq, 24 b) + post"),
    "DivFullRcp_v1": (_mufu(23), "lookup slot (rcp, 23 b) + post"),
    "Fa2InvSum_v1": (_inv_sum, "rcp lookup + fma"), "Fa3InvSum_v1": (_inv_sum, "rcp lookup + fma"),
    "GuardNegInfZero_v1": (_guard_neg_inf, "exact"),
    "TanhF32Rn_v1": (_tanh, "a 16-bit table of a bf16 operand (SiLU's shape), else 8 fma"),
    "GeluTanhMulBf16_v1": (_gelu_tanh_mul, "SiLU·mul's shape: a 16-bit table, then the multiply"),
    "F32ToE4m3Sat_v1": (_to_e4m3_sat, "round + saturate"),
    "GumbelStreamKey_v1": (_gumbel_key, "Philox 10 rounds"), "GumbelNoiseLane_v1": (_gumbel_lane, "Philox + 16 fma for two logs"),
}

#: primitives priced without building a circuit: (ANDs as a function of the input bits, nonzeros per AND, note)
FIXED_COST = {
    "TopPMaskWordx": (lambda nin: 40 * nin, 20, "top-p threshold search: ~40 ANDs per input bit (order of magnitude)"),
    "BitAtx": (lambda nin: 2 * nin, 3, "a word-wide mux by index"),
}


def _is_gather(pid: str) -> bool:
    return pid.startswith(("GatherBf16x", "GatherF32x", "LGatherRow"))


# --- piece models ------------------------------------------------------------------------------------------------------

@dataclass
class Piece:
    """One primitive lowered at one constant-bit pattern: ``ands`` live ANDs; ``fixed`` AND-row entries on AND columns or the
    constant; ``in_ref[i]`` AND-row references to variable input bit i; per output bit ``out_const`` (-1 variable, else its
    value), ``out_fixed`` and ``out_ref`` (CSR over variable input bits); ``lookups`` its MUFU lookup slots."""

    ands: int
    fixed: int
    in_ref: np.ndarray
    out_const: np.ndarray
    out_fixed: np.ndarray
    out_ptr: np.ndarray
    out_idx: np.ndarray
    lookups: list = field(default_factory=list)
    extra_in: int = 0
    estimate: str | None = None


def _fixed_cost(pid: str):
    return next((v for k, v in FIXED_COST.items() if pid.startswith(k)), None)


class Pieces:
    """Piece models by (primitive, constant-bit pattern of its operands); past CAP patterns of one primitive, its
    all-variable model stands in (constant bits then count as forms of size 0 or 1)."""

    CAP = 12

    def __init__(self):
        self.cache: dict = {}
        self.patterns: Counter = Counter()
        self.notes: dict[str, str] = {}

    @staticmethod
    def piece_fn(pid: str):
        if pid in SURROGATES:
            fn, note = SURROGATES[pid]
            return fn, (None if note in ("exact", "exact piece") else note)
        if pid in IL.PIECES:
            return IL.PIECES[pid], None
        return None, None

    def get(self, prim, pattern: tuple) -> tuple[Piece, bool]:
        """(model, whether it is the all-variable model standing in for ``pattern``)."""
        pid = prim.id
        if _fixed_cost(pid) is not None:
            nvar = sum(len(w) for w in pattern)
            key = (pid, nvar)
            if key not in self.cache:
                self.cache[key] = self._fixed(prim, nvar)
            return self.cache[key], True
        key = (pid, pattern)
        if key in self.cache:
            return self.cache[key], False
        var = tuple(tuple(None for _ in w) for w in pattern)
        if self.patterns[pid] >= self.CAP and var != pattern:
            if (pid, var) not in self.cache:
                self.cache[(pid, var)] = self._build(prim, var)
            return self.cache[(pid, var)], True
        self.patterns[pid] += 1
        self.cache[key] = p = self._build(prim, pattern)
        return p, False

    def _fixed(self, prim, nvar: int) -> Piece:
        f, rho, note = _fixed_cost(prim.id)
        a, wout = int(f(nvar)), prim.ret.w
        self.notes[prim.id.rstrip("0123456789_v")] = note
        return Piece(a, int(rho * a), np.full(nvar, 2, np.int64), np.full(wout, -1, np.int64), np.ones(wout, np.int64),
                     np.zeros(wout + 1, np.int64), np.zeros(0, np.int64), estimate=note)

    def _build(self, prim, pattern) -> Piece:
        pid = prim.id
        nvar = sum(1 for w in pattern for b in w if b is None)
        wout = prim.ret.w
        fn, note = self.piece_fn(pid)
        if fn is None:
            raise IL.NoPiece(f"no piece or surrogate for {pid}")
        if note:
            self.notes[pid] = note
        C = Circuit()
        C.lookups = []
        ops, pos = [], {}
        for j, w in enumerate(pattern):
            word = []
            for b in w:
                if b is None:
                    (x,) = C.input(f"v{len(pos)}", 1)
                    (v,) = tuple(x.s)
                    pos[v] = len(pos)
                    word.append(x)
                else:
                    word.append(ONE if b else ZERO)
            ops.append(word)
        out = fn(C, *ops)
        assert len(out) == wout, (pid, len(out), wout)
        C.output("out", out)
        live = C.live()
        extra_in = sum(1 for v in live if C.kind[v] == "in" and v not in pos)
        in_ref = np.zeros(nvar, np.int64)
        fixed_n, ands = 0, 0
        for v in live:
            if C.kind[v] != "and":
                continue
            ands += 1
            for f in C.data[v]:
                fixed_n += f.c
                for s in f.s:
                    if s in pos:
                        in_ref[pos[s]] += 1
                    else:
                        fixed_n += 1
        oc, of, ptr, idx = [], [], [0], []
        for w in out:
            if not w.s:
                oc.append(w.c)
                of.append(0)
            else:
                oc.append(-1)
                of.append(w.c + sum(1 for s in w.s if s not in pos))
                idx.extend(pos[s] for s in w.s if s in pos)
            ptr.append(len(idx))
        return Piece(ands, fixed_n, in_ref, np.asarray(oc, np.int64), np.asarray(of, np.int64), np.asarray(ptr, np.int64),
                     np.asarray(idx, np.int64), [b for b, _ in C.lookups], extra_in, note)


PIECES = Pieces()


def _apply(p: Piece, sizes: np.ndarray):
    """(Z of the piece's AND rows, output sizes, output constants) for variable-input form sizes ``sizes``."""
    z = p.fixed + int(p.in_ref @ sizes) if len(sizes) else p.fixed
    if len(p.out_idx):
        s = np.add.reduceat(np.append(sizes[p.out_idx], 0), p.out_ptr[:-1])
        s[p.out_ptr[1:] == p.out_ptr[:-1]] = 0
    else:
        s = np.zeros(len(p.out_fixed), np.int64)
    return z, p.out_fixed + s, p.out_const


# --- layout arithmetic -------------------------------------------------------------------------------------------------

def region(bits: int) -> int:
    """Rows of one port group: a power-of-two number of 128-bit words."""
    return WORD * pow2(max(1, math.ceil(bits / WORD)))


def layout_rows(in_bits: int, ands: int, out_bits: int) -> int:
    """Rows of a ``flock-ir-unit/v2`` layout with one input group and one output group (:func:`ir_lower.layout`)."""
    r = region(in_bits) + ands
    o = region(out_bits)
    r = -(-r // o) * o
    return r + o + 1


def pow2_slot(rows: int) -> int:
    return 1 << max(7, rows.bit_length())


def pow2(x: int) -> int:
    return 1 << max(0, (int(x) - 1).bit_length())


# --- a unit, composed --------------------------------------------------------------------------------------------------

@dataclass
class UnitShape:
    kind: str
    head: str | None
    scope: str | None
    word_gates: int
    out_bits: int
    out_gates: int
    count: int
    rows: int = 0
    ands: int = 0
    z: int = 0
    d_in: int = 0
    d_out: int = 0
    in_bits: int = 0
    lookups: list = field(default_factory=list)
    prims: dict = field(default_factory=dict)
    estimated: list = field(default_factory=list)
    gathers: int = 0
    seq: dict | None = None

    @property
    def d(self) -> int:
        return self.d_in + self.d_out

    @property
    def R(self) -> int:
        return pow2_slot(self.rows)

    def record(self) -> dict:
        r = {k: getattr(self, k) for k in ("kind", "head", "scope", "word_gates", "out_bits", "out_gates", "count", "rows", "ands",
                                           "z", "d_in", "d_out", "in_bits", "gathers")}
        r.update(R=self.R, D=self.d, lookups=len(self.lookups), lookup_bits=sorted(set(self.lookups)), prims=self.prims,
                 estimated=self.estimated)
        return r


WIDE_FAN_IN = 4096
#: a wire whose linear form has more terms than this is committed (one copy row), as a lowering does for a long select or
#: add chain; no unit checked against :meth:`CallUnits.exact` reaches it
REFRESH = 64
#: a unit's representative keeps its gate sequence (for :func:`pack`) when it might not fit the smallest shape assessed
SEQ_IF = (1 << 12, 1 << 16, 16)


class CallUnits:
    """One Call specialization's partition (``word.Graph`` + ``word.units``) with the circuit access composition needs."""

    def __init__(self, fn):
        from verity_vllm.query import word as WQ

        self.fn = fn
        self.G = G = WQ.Graph(fn)
        self.R = WQ.units(G, X_BITS, W_BITS)
        self.c, self.call = WQ._call_scope(fn)
        self.lo, self.n = self.call.offset, fn.gates
        self.kept = np.full(self.n, -1, np.int64)
        self.kept[G.keep] = np.arange(G.n)
        self.input_pos = {int(r): k for k, r in enumerate(G.input_ids)}
        owner = self.R["owner"]
        b = np.zeros(G.n, bool)
        cross = owner[G.src] != owner[G.dst]
        b[G.src[cross]] = True
        b[G.outputs] = True
        self.boundary = b
        o = np.argsort(G.dst, kind="stable")
        self.by_dst = (G.src[o], np.searchsorted(G.dst[o], np.arange(G.n + 1)))

    def classes(self) -> list[tuple[tuple, int, np.ndarray]]:
        """(class key, units per Call, member gates of the representative) per class: ``word.unit_rule``'s key plus the
        unit's primitive multiset (a hash) and the values it reads."""
        G, R = self.G, self.R
        owner = R["owner"]
        U = len(R["kinds"])
        if not U:
            return []
        h = np.random.default_rng(7).integers(1, 2**62, size=len(G.prim_names) + 1, dtype=np.int64)
        sig = np.zeros(U, np.int64)
        np.add.at(sig, owner, h[G.prim])
        cross = owner[G.src] != owner[G.dst]
        d_in = np.zeros(U, np.int64)
        if cross.any():
            pairs = np.unique(np.stack([owner[G.dst[cross]], G.src[cross]], 1), axis=0)
            d_in += np.bincount(pairs[:, 0], minlength=U)
        if G.input_reads:
            ir = np.asarray(G.input_reads, np.int64)
            ip = np.unique(np.stack([owner[ir[:, 1]], ir[:, 0]], 1), axis=0)
            d_in += np.bincount(ip[:, 0], minlength=U)
        kinds = np.array([("output", "committed", "dead").index(k) for k in R["kinds"]], np.int64)
        cols = np.stack([kinds, R["head_prim"], R["head_scope"], R["gates"], R["out_bits"], R["out_gates"], sig, d_in], 1)
        keys, first, count = np.unique(cols, axis=0, return_index=True, return_counts=True)
        order = np.argsort(owner, kind="stable")
        starts = np.searchsorted(owner[order], np.arange(U + 1))
        return [(tuple(k), int(c), order[starts[u]:starts[u + 1]]) for k, u, c in zip(keys.tolist(), first.tolist(), count.tolist())]

    def compose(self, key: tuple, count: int, members: np.ndarray) -> UnitShape:
        G, c, lo, n = self.G, self.c, self.lo, self.n
        kd, hp, hs, ng, ob, og = key[:6]
        us = UnitShape(("output", "committed", "dead")[kd], G.prim_names.get(hp), G.scope_names.get(hs), ng, ob, og, count)
        mset = set(G.keep[members].tolist())
        lowering: set[int] = set(mset)
        stack = list(mset)
        while stack:
            j = stack.pop()
            g = c.gate(lo + j)
            if _is_gather(g.prim.id):
                continue
            for o in g.operands:
                l = o - lo
                if 0 <= l < n and self.kept[l] < 0 and l not in lowering:
                    lowering.add(l)
                    stack.append(l)
        state: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        inputs: dict = {}
        z = ands = glue = 0
        lookups: list = []
        prims: Counter = Counter()
        est: set = set()
        per_gate: dict[int, tuple[int, int]] = {}

        def value(o: int):
            if o not in state:
                w = c.gate(o).prim.ret.w
                inputs[o] = w
                state[o] = (np.ones(w, np.int64), np.full(w, -1, np.int64))
            return state[o]

        def const(j, w, v):
            state[lo + j] = (np.zeros(w, np.int64), np.array([(v >> i) & 1 for i in range(w)], np.int64))

        for j in sorted(lowering):
            g = c.gate(lo + j)
            prim, pid, w = g.prim, g.prim.id, g.prim.ret.w
            if not prim.params:
                const(j, w, int(prim.evaluate()))
                continue
            if _is_gather(pid):
                if lo + j not in state:
                    idx = g.operands[0]
                    if idx not in state:
                        value(idx)
                inputs[("gather", lo + j)] = w
                state[lo + j] = (np.ones(w, np.int64), np.full(w, -1, np.int64))
                us.gathers += 1
                prims[pid] += 1
                per_gate[j] = (0, 0)
                continue
            if len(g.operands) > WIDE_FAN_IN and _fixed_cost(pid) is None:
                raise IL.NoPiece(f"{pid}: a wide gate that is neither a gather nor priced")
            ops = [value(o) for o in g.operands]
            if pid.startswith("BitAtx") and (ops[1][1] >= 0).all():
                k = sum(int(b) << i for i, b in enumerate(ops[1][1]))
                k = k - (1 << 32) if k & 0x80000000 else k
                ws, wc = ops[0]
                state[lo + j] = (ws[k:k + 1].copy(), wc[k:k + 1].copy()) if 0 <= k < len(ws) else (np.zeros(1, np.int64), np.zeros(1, np.int64))
                prims[pid] += 1
                continue
            if all((cv >= 0).all() for _s, cv in ops):
                const(j, w, int(prim.evaluate(*[sum(int(b) << i for i, b in enumerate(cv)) for _s, cv in ops])))
                continue
            pattern = tuple(tuple(None if b < 0 else int(b) for b in cv) for _s, cv in ops)
            p, all_var = PIECES.get(prim, pattern)
            if all_var:
                sizes = np.concatenate([np.where(cv < 0, s, cv) for s, cv in ops])
            else:
                sizes = np.concatenate([s[cv < 0] for s, cv in ops])
            zi, osz, oc = _apply(p, sizes)
            lk = sum(p.lookups)
            big = (osz > REFRESH) & (oc < 0)
            nb = int(big.sum())
            if nb:
                zi += int(osz[big].sum()) + nb
                osz = np.where(big, 1, osz)
            gz, gr = zi + p.extra_in + 2 * lk, p.ands + p.extra_in + lk + nb
            glue += nb
            z += gz
            ands += p.ands
            glue += p.extra_in + lk
            lookups += p.lookups
            if p.estimate:
                est.add(pid)
            prims[pid] += 1
            per_gate[j] = (gr, gz)
            state[lo + j] = (osz, oc)
        out_bits = 0
        outs = [int(G.keep[i]) for i in members.tolist() if self.boundary[i]]
        for j in outs:
            s, cv = state[lo + j]
            z += int(np.where(cv < 0, s, cv).sum()) + len(s)
            out_bits += len(s)
        us.in_bits, us.d_in, us.d_out = sum(inputs.values()), len(inputs), len(outs)
        us.ands, us.z, us.lookups, us.prims, us.estimated = ands, z, lookups, dict(prims.most_common()), sorted(est)
        us.rows = layout_rows(us.in_bits, ands + glue, out_bits)
        us._per_gate = per_gate
        return us

    def exact(self, members: np.ndarray) -> dict:
        """The unit lowered into one circuit (the same pieces and surrogates) and laid out by :func:`ir_lower.layout`: its
        rows and Z, to check :meth:`compose`. Only for units without gathers, lookups or priced primitives."""
        G, c, lo, n = self.G, self.c, self.lo, self.n
        mset = set(G.keep[members].tolist())
        lowering, stack = set(mset), list(mset)
        while stack:
            j = stack.pop()
            for o in c.gate(lo + j).operands:
                l = o - lo
                if 0 <= l < n and self.kept[l] < 0 and l not in lowering:
                    lowering.add(l)
                    stack.append(l)
        C = Circuit()
        C.lookups = []
        wires: dict[int, list] = {}
        names = []
        for j in sorted(lowering):
            for o in c.gate(lo + j).operands:
                l = o - lo
                if o not in wires and not (0 <= l < n and l in lowering):
                    names.append(f"in{len(names)}")
                    wires[o] = C.input(names[-1], c.gate(o).prim.ret.w)
        gates = [c.gate(lo + j) for j in sorted(lowering)]
        pieces = {**{k: v for k, (v, _n) in SURROGATES.items()}, **IL.PIECES}
        IL.lower_gates(C, gates, wires, pieces)
        outs = []
        for i in members.tolist():
            if self.boundary[i]:
                outs.append(f"out{len(outs)}")
                C.output(outs[-1], wires[lo + int(G.keep[i])])
        assert not C.lookups
        lay = IL.layout(C, names, outs)
        ib = sum(len(C.inputs[nm]) for nm in names)
        return {"rows": lay.useful, "ands": lay.counts["ands"], "z": lay.counts["nnz_a"] + lay.counts["nnz_b"] - 2 * ib - 2}

    def _src(self, o: int, mem: set):
        """A read of root gate ``o``: ('dep', member) inside the unit, else the committed value's key and width."""
        l = o - self.lo
        if 0 <= l < self.n and self.kept[l] >= 0:
            i = int(self.kept[l])
            return ("dep", i) if i in mem else (f"g{i}", int(self.G.width[i]))
        if 0 <= l < self.n:
            return None
        return (f"i{self.input_pos.get(o, o)}", int(self.c.gate(o).prim.ret.w))

    def seq(self, members: np.ndarray, per_gate: dict) -> dict:
        """The unit's computing gates in depth-first post-order from its outputs, with per gate its rows and Z (from the
        composition), the committed values it reads, its unit-internal operands and consumers: :func:`pack`'s input."""
        G, c, lo = self.G, self.c, self.lo
        mem = set(members.tolist())
        s, st = self.by_dst
        deps: dict[int, list[int]] = {}
        ext: dict[int, list[tuple]] = {}
        for m in members.tolist():
            g = c.gate(lo + int(G.keep[m]))
            if _is_gather(g.prim.id):
                r = self._src(g.operands[0], mem)
                deps[m] = [r[1]] if r and r[0] == "dep" else []
                ext[m] = ([r] if r and r[0] != "dep" else []) + [(f"gather{m}", int(G.width[m]))]
                continue
            deps[m] = [int(a) for a in s[st[m]:st[m + 1]].tolist() if int(a) in mem]
            ext[m] = [(f"g{int(a)}", int(G.width[a])) for a in s[st[m]:st[m + 1]].tolist() if int(a) not in mem]
        for inp, b in G.input_reads:
            if b in mem and not _is_gather(c.gate(lo + int(G.keep[b])).prim.id):
                ext[b].append((f"i{inp}", int(c.gate(G.input_ids[inp]).prim.ret.w)))
        cons = Counter(a for b in deps for a in deps[b])
        order, seen = [], set()
        for r in [i for i in members.tolist() if self.boundary[i]] + members.tolist():
            if r in seen:
                continue
            stack = [(r, False)]
            while stack:
                v, done = stack.pop()
                if done:
                    order.append(v)
                    continue
                if v in seen:
                    continue
                seen.add(v)
                stack.append((v, True))
                stack.extend((d, False) for d in reversed(deps[v]) if d not in seen)
        pos = {v: k for k, v in enumerate(order)}
        pg = [per_gate.get(int(G.keep[v]), (0, 0)) for v in order]
        return {"rows": [a for a, _ in pg], "z": [b for _, b in pg], "width": [int(G.width[v]) for v in order],
                "deps": [[pos[d] for d in deps[v]] for v in order], "ext": [ext[v] for v in order],
                "cons": [cons.get(v, 0) for v in order], "is_out": [bool(self.boundary[v]) for v in order]}


def primitive_unit(fn) -> UnitShape:
    """A Call that is one primitive: one unit reading its parameters."""
    leaves = [w for _n, t in fn.params for w in t.leaf_widths()]
    us = UnitShape("output", fn.id, fn.id, 1, fn.ret.w, 1, 1, prims={fn.id: 1})
    if _is_gather(fn.id):
        us.in_bits, us.d_in, us.d_out, us.gathers = leaves[0] + fn.ret.w, 2, 1, 1
        us.rows, us.z = layout_rows(us.in_bits, 0, fn.ret.w), 2 * fn.ret.w
        return us
    p, _ = PIECES.get(fn, tuple(tuple(None for _ in range(w)) for w in leaves))
    zi, osz, oc = _apply(p, np.ones(sum(leaves), np.int64))
    lk = sum(p.lookups)
    us.ands = p.ands
    us.z = zi + p.extra_in + 2 * lk + int(np.where(oc < 0, osz, oc).sum()) + len(osz)
    us.in_bits, us.d_in, us.d_out = sum(leaves), len(leaves), 1
    us.lookups, us.estimated = p.lookups, ([fn.id] if p.estimate else [])
    us.rows = layout_rows(us.in_bits, p.ands + p.extra_in + lk, fn.ret.w)
    return us


def _needs_seq(u: UnitShape) -> bool:
    return u.rows > SEQ_IF[0] or u.z > SEQ_IF[1] or u.d > SEQ_IF[2]


def spec_units(fn, memo: dict, log=None) -> list[tuple[UnitShape, int]]:
    """Every unit class of one Call specialization as (shape, units per Call), memoised by specialization id; a separable
    Definition (a Gemm's coordinates, attention's heads) is its members' classes."""
    from verity.ir.defs import PrimitiveDefinition
    from verity_vllm.query import word as WQ

    if fn.id in memo:
        return memo[fn.id]
    if isinstance(fn, PrimitiveDefinition):
        res = [] if (not fn.params or fn.id in WQ.WIRING) else [(primitive_unit(fn), 1)]
    elif (sep := WQ._separable(fn)) is not None:
        agg: dict[int, list] = {}
        for cfn, m in sep:
            for u, k in spec_units(cfn, memo, log):
                agg.setdefault(id(u), [u, 0])[1] += k * m
        res = [(u, k) for u, k in agg.values()]
    else:
        t = time.time()
        cu = CallUnits(fn)
        res = []
        for key, cnt, mem in cu.classes():
            u = cu.compose(key, cnt, mem)
            if _needs_seq(u):
                u.seq = cu.seq(mem, u._per_gate)
            del u._per_gate
            res.append((u, cnt))
        if log and time.time() - t > 5:
            log(f"    {fn.id[:90]}: {len(res)} classes, {sum(k for _u, k in res)} units, {time.time() - t:.1f}s")
    memo[fn.id] = res
    return res


# --- splitting ----------------------------------------------------------------------------------------------------------

def pack(seq: dict, R: int, Z: int, D: int) -> dict:
    """Greedy split of one unit into pieces fitting (R, Z, D), over its gates in ``seq`` order: ``pieces``; ``extra``, the
    values committed between pieces (new serving commitments: returned values are committed already); ``oversize``, gates
    that alone exceed the shape; per piece its rows, Z and D."""
    n = len(seq["rows"])
    piece_of = np.full(n, -1, np.int64)
    rem = np.zeros(n, np.int64)
    pieces: list[dict] = []
    crossing: set[int] = set()
    oversize = 0

    def fresh():
        return {"reads": set(), "in_bits": 0, "rows": 0, "z": 0, "writes": 0, "out_bits": 0, "gates": 0}

    def trial(P, i):
        cur = len(pieces)
        reads, in_bits, dw, dob = set(), 0, 0, 0
        for k, w in seq["ext"][i]:
            if k not in P["reads"] and k not in reads:
                reads.add(k)
                in_bits += w
        here = Counter()
        for d in seq["deps"][i]:
            if piece_of[d] == cur:
                here[d] += 1
            elif f"x{d}" not in P["reads"] and f"x{d}" not in reads:
                reads.add(f"x{d}")
                in_bits += seq["width"][d]
        for d, k in here.items():
            if rem[d] - k == 0 and not seq["is_out"][d]:
                dw -= 1
                dob -= seq["width"][d]
        if seq["cons"][i] > 0 or seq["is_out"][i]:
            dw += 1
            dob += seq["width"][i]
        ob = P["out_bits"] + dob
        rows = layout_rows(P["in_bits"] + in_bits, P["rows"] + seq["rows"][i], max(ob, 1))
        return rows, P["z"] + seq["z"][i] + 2 * ob, len(P["reads"]) + len(reads) + P["writes"] + dw, reads, in_bits, here, dw, dob

    def fits(rows, zz, dd):
        return pow2_slot(rows) <= R and zz <= Z and dd <= D

    P = fresh()
    for i in range(n):
        t = trial(P, i)
        if P["gates"] and not fits(*t[:3]):
            pieces.append(P)
            P = fresh()
            t = trial(P, i)
        _rows, _z, _d, reads, in_bits, here, dw, dob = t
        if not fits(*t[:3]):
            oversize += 1
        for d in seq["deps"][i]:
            if piece_of[d] != len(pieces) and not seq["is_out"][d]:
                crossing.add(d)
        for d, k in here.items():
            rem[d] -= k
        P["reads"] |= reads
        P["in_bits"] += in_bits
        P["rows"] += seq["rows"][i]
        P["z"] += seq["z"][i]
        P["writes"] += dw
        P["out_bits"] += dob
        P["gates"] += 1
        piece_of[i] = len(pieces)
        rem[i] = seq["cons"][i]
    pieces.append(P)
    each = []
    for P in pieces:
        ob = max(P["out_bits"], 1)
        each.append({"rows": layout_rows(P["in_bits"], P["rows"], ob), "z": P["z"] + 2 * ob, "d": len(P["reads"]) + P["writes"]})
    return {"pieces": len(each), "extra": len(crossing), "oversize": oversize, "each": each}


def fifo_order(seq: dict) -> dict:
    """``seq`` re-ordered topologically, oldest operand first (ties in ``seq``'s own order): interleaves the chains of a scan
    (an argmax's running maximum and index) that the depth-first order emits one after the other."""
    import heapq

    n = len(seq["rows"])
    cons: list[list[int]] = [[] for _ in range(n)]
    need = [len(set(d)) for d in seq["deps"]]
    for i, ds in enumerate(seq["deps"]):
        for d in set(ds):
            cons[d].append(i)
    t = [0] * n
    heap = [(-1, i) for i in range(n) if not need[i]]
    heapq.heapify(heap)
    order = []
    while heap:
        _p, i = heapq.heappop(heap)
        t[i] = len(order)
        order.append(i)
        for c in cons[i]:
            need[c] -= 1
            if not need[c]:
                heapq.heappush(heap, (min(t[d] for d in seq["deps"][c]), c))
    pos = {v: k for k, v in enumerate(order)}
    return {k: [seq[k][v] for v in order] for k in ("rows", "z", "width", "ext", "cons", "is_out")} | {
        "deps": [[pos[d] for d in seq["deps"][v]] for v in order]}


def pack_best(seq: dict, R: int, Z: int, D: int) -> dict:
    """:func:`pack` over the depth-first order and over :func:`fifo_order`, whichever makes fewer pieces plus commitments."""
    a = pack(seq, R, Z, D)
    if a["pieces"] == 1:
        return a
    b = pack(fifo_order(seq), R, Z, D)
    return min((a, b), key=lambda p: (p["oversize"], p["pieces"] + p["extra"]))


# --- program graphs ------------------------------------------------------------------------------------------------------

#: a varying static (attention's key count T) with more distinct values than EXACT_T is evaluated once per stratum of STRATUM
#: consecutive values, at the stratum's calls-weighted mean (unit counts linear in T are then exact in total)
STRATUM = 32
EXACT_T = 320


def t_points(calls: dict[int, int]) -> list[tuple[int, int]]:
    vals = sorted(calls)
    if len(vals) <= EXACT_T:
        return [(v, calls[v]) for v in vals]
    out = []
    for s in sorted({v // STRATUM for v in vals}):
        vs = [v for v in vals if v // STRATUM == s]
        c = sum(calls[v] for v in vs)
        out.append((int(round(sum(v * calls[v] for v in vs) / c)), c))
    return out


def program_units(pg_path: Path, memo: dict, log=None) -> dict:
    """Every unit class of a program graph with its total count over the row's Calls (and per template)."""
    from verity_vllm.query import word as WQ

    g = json.loads(Path(pg_path).read_text())
    rows: dict[int, list] = {}
    groups = []
    for x in g["groups"]:
        if x["definition"].startswith("Const"):
            continue
        pts = [(None, x["calls"])]
        if x.get("varying") and x.get("varying_calls"):
            pts = t_points({int(k): v for k, v in x["varying_calls"].items()})
        n_units = 0
        for t, calls in pts:
            st = dict(x["statics"])
            if t is not None:
                st[next(iter(x["varying"]))] = t
            for u, k in spec_units(WQ.specialization(x["definition"], st), memo, log):
                e = rows.setdefault(id(u), [u, 0, Counter()])
                e[1] += k * calls
                e[2][x.get("template") or x["definition"]] += k * calls
                n_units += k * calls
        groups.append({"id": x["id"], "definition": x["definition"], "template": x.get("template"), "calls": x["calls"],
                       "units": n_units, "stored_units": (x.get("q_word_v1") or {}).get("units")})
        if log:
            log(f"  {x['id']} {x['definition']} {len(pts)} point(s): {n_units} units")
    return {"classes": list(rows.values()), "groups": groups, "stored_total_units": g.get("q_word_v1", {}).get("units")}


# --- shapes --------------------------------------------------------------------------------------------------------------

def assess(classes: list, S: tuple[int, int, int], cache: dict) -> dict:
    """One program's units under the shape S = (R, Z, D): the unit count n, the padding waste per dimension (the fraction of
    the n padded units' rows / nonzeros / boundary values that is padding), the units split and the values their splits
    commit."""
    R, Z, D = S
    n = split = extra = oversize = 0
    used = np.zeros(3)
    for u, cnt, *_ in classes:
        if u.R <= R and u.z <= Z and u.d <= D:
            n += cnt
            used += cnt * np.array([u.rows, u.z, u.d], float)
            continue
        key = (id(u), S)
        if key not in cache:
            cache[key] = pack(u.seq, R, Z, D) if u.seq else {"pieces": 1, "extra": 0, "oversize": 1,
                                                            "each": [{"rows": u.rows, "z": u.z, "d": u.d}]}
        p = cache[key]
        n += cnt * p["pieces"]
        split += cnt * (p["pieces"] > 1)
        extra += cnt * p["extra"]
        oversize += cnt * p["oversize"]
        used += cnt * np.array([sum(e["rows"] for e in p["each"]), sum(e["z"] for e in p["each"]), sum(e["d"] for e in p["each"])], float)
    waste = (1 - used / (n * np.array([R, Z, D], float))).tolist() if n else [None] * 3
    return {"S": list(S), "n": n, "split_units": split, "extra_commitments": extra, "oversize_gates": oversize,
            "used": used.tolist(), "waste_rows": waste[0], "waste_z": waste[1], "waste_d": waste[2]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--program-graphs", required=True, type=Path)
    ap.add_argument("--rows", nargs="*", help="row key prefixes (default: every *.program.json)")
    ap.add_argument("--out", required=True, type=Path, help="directory: <row>.json per row (a row already there is skipped)")
    ap.add_argument("--pickle", action="store_true", help="also <row>.pkl: the classes with their gate sequences (for assess)")
    a = ap.parse_args(argv)
    files = sorted(a.program_graphs.glob("*.program.json"))
    if a.rows:
        files = [f for f in files if any(f.name.startswith(r) for r in a.rows)]
    a.out.mkdir(parents=True, exist_ok=True)
    memo: dict = {}
    for f in files:
        row = f.name[: -len(".program.json")]
        if (a.out / f"{row}.json").exists():
            continue
        t = time.time()
        print(f.name, file=sys.stderr, flush=True)
        pu = program_units(f, memo, log=lambda s: print(s, file=sys.stderr, flush=True))
        if a.pickle:
            import pickle
            (a.out / f"{row}.pkl").write_bytes(pickle.dumps(pu["classes"], protocol=5))
        (a.out / f"{row}.json").write_text(json.dumps({
            "schema": "verity-unit-shape-census/v1", "query": "Q_word_v1{X=16,W=32,R=no-recompute}", "row": row,
            "classes": [dict(u.record(), count=cnt, templates=dict(tp)) for u, cnt, tp in pu["classes"]],
            "groups": pu["groups"], "stored_total_units": pu["stored_total_units"], "surrogates": PIECES.notes}, indent=1))
        print(f"  {time.time() - t:.0f}s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    from verity_flock.unit_shapes import main as _main  # so pickled classes name this module, not __main__

    sys.exit(_main())
