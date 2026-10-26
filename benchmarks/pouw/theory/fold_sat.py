"""FOLD-SAT: no program of k - 1 instructions computes a receipt fold step, by SMT over exact 32-bit SASS semantics.

The targets are `verity/core/protocols/pouw/circuit/segment.py`'s folds with their real keys:

- li's half step (`_li_absorb`, chain 0, h = 0): y' = ((y ^ w0 ^ LK) * LA + w1) mod 2^32, two committed words;
- duo's entry step (`_duo_absorb`, t = 0, e = 0): u = y ^ p ^ KK, y' = u KA + z, z' = rotl(u, ROT) + y';
- fx1's (`_fx1_absorb`, q = 0): duo's step on p * fx1_key(0);
- the split forms (ABSORB in cost form): each committed word arrives as two or three 32-bit summands.

A program is k slots of straight-line SASS over 32-bit registers. Each slot picks an opcode, four operands (an input,
RZ, an earlier slot's output or high half, or its own 32-bit immediate) and its opcode's parameters, all symbolic: the
LOP3 LUT ranges over all 256 tables, shift counts over 0..63, immediates over all 2^32 values. Every target output must
equal some register of the program. `synth` asks z3 for such a program agreeing with the target on a finite set of
inputs: UNSAT is a proof that no k-slot program exists (a program that agrees everywhere agrees on the set), and since a
shorter program pads to k slots with unused ones, UNSAT at k covers every length up to k. SAT gives a candidate, which
`verify` checks for all inputs; a counterexample joins the set (CEGIS).

The instruction models are supersets of the real encodings, so a lower bound here holds for the card: an immediate may
stand in any operand (SASS allows one, usually in b), each IADD3 operand may be negated, and IADD64 adds two register
pairs in one slot (SASS needs IADD3 + IADD3.X). Semantics, per the PTX ISA (lop3, add/sub, mad.lo/.hi/.wide, shf.l/.r
.wrap/.clamp, prmt default mode, min/max, popc, clz, brev, bmsk, szext, selp) and their SASS forms (LOP3.LUT, IADD3,
IMAD, IMAD.HI, IMAD.WIDE, SHF, LEA/LEA.HI, PRMT, IMNMX, POPC, FLO, BREV, BMSK, SGXT, SEL, MOV):

  LOP3(a, b, c; lut)        bitwise lut[4 a + 2 b + c]
  IADD3(a, b, c; signs)     (+/-a) + (+/-b) + (+/-c)
  IMAD(a, b, c; neg)        lo(a b) +/- c
  IMAD.HI(a, b, c; s)       hi(a b) + c, signed or unsigned product
  IMAD.WIDE(a, b, c, d; s)  (lo, hi) of a b + {d:c}, signed or unsigned product (two registers)
  IADD64(a, b, c, d)        (lo, hi) of {b:a} + {d:c} (two registers)
  SHF(a, b, c; dir, s, n)   funnel: lo({b:a} >> n) or hi({b:a} << n); n is the immediate (0..63) or register c
                            wrapped (& 31, & 63) or clamped (min(c, 32), min(c, 64)); right shifts may fill from b's sign
  LEA(a, c; n)              (a << n) + c
  LEA.HI(a, b, c; n)        lo({b:a} >> (32 - n)) + c
  PRMT(a, b, c; mode)       default: byte i = byte (c >> 4 i) & 7 of {b:a}, replicated sign if (c >> 4 i) & 8; or
                            f4e, b4e, rc8, ecl, ecr, rc16 on c[1:0]
  IMNMX(a, b; s, max)       min or max, signed or unsigned
  UNARY(a; f)               POPC, CLZ, BREV, NOT, IABS, SGXT(a, n), BMSK(a, n), FLO.U32 (the highest set bit's index,
                            all ones for zero)
  SEL(a, b, c, d; cmp)      ISETP + SEL fused in one slot: (a cmp b) ? c : d, six comparisons, signed or unsigned
  IDP.4A(a, b, c; sa, sb)   c + the four byte products of a and b, each side signed or not

This covers every integer lane function of MODEL-SPEC's instruction set (`Security/Definitions/Core/Accel/Isa.lean`'s
`Lane`, branch cursor/compute-research-accel-model-spec-ea24 at 738a30c30f): iadd3, mad, madWide, lop3, shf, prmt, popc,
flo, brev, setp, sel, min, max, lea and dp4a, each as a special case of a slot here. Its FP32 lane functions (ffma, fadd,
fmul) are not modelled.

MOV with an immediate is an operand; IADD (32) is IADD3 with RZ. Not modelled, so not covered: the FP pipes (FFMA,
FADD, FMUL, HFMA2, DFMA on integer bit patterns), IDP.2A, memory (table lookups through LDS or constant memory),
shuffles, a predicate read by more than one instruction (predicated execution beyond one fused SEL), and carry chains
other than IADD64 (IADD3.X and IMAD.X with a three-input low half). A bound from this file is a bound over the
modelled set.

Run: `python fold_sat.py cell TARGET K [--vectors N] [--timeout S]`, or `python fold_sat.py sweep --out DIR`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import z3

from verity.core.protocols.pouw.circuit import segment as S

W = 32
M32 = (1 << W) - 1

LI_K, LI_K2 = S.LK[0]
LI_A, LI_A2 = S.LA[0]
DUO_K, DUO_A, DUO_R = S.KK[0], S.KA[0], S.ROT[0]
FX1_K = S.fx1_key(0)
rotl = S.rotl


# ---- targets --------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    honest: int          # the honest kernel's instruction count for this step
    words: int           # committed words the step absorbs
    honest_ops: str      # the honest program, for the report

    def eval(self, xs: list[int]) -> list[int]:
        return [v & M32 for v in TARGET_FNS[self.name](*xs)]

    def sym(self, xs):
        return TARGET_FNS[self.name](*xs)


def _li(y, w0, w1, K=None, A=None):
    K = LI_K if K is None else K
    A = LI_A if A is None else A
    return [(y ^ w0 ^ K) * A + w1]


def _rotl_any(x, r):
    if isinstance(x, int):
        return rotl(x, r)
    return z3.RotateLeft(x, r)


def _duo(y, z, p):
    u = y ^ p ^ DUO_K
    if isinstance(u, int):
        u &= M32
    y2 = u * DUO_A + z
    return [y2, _rotl_any(u, DUO_R) + y2]


def _fx1(y, z, p):
    return _duo(y, z, p * FX1_K)


TARGET_FNS = {
    "li": _li,
    "li_split2": lambda y, a0, b0, a1, b1: _li(y, a0 + b0, a1 + b1),
    "li_split_w0": lambda y, a0, b0, w1: _li(y, a0 + b0, w1),
    "li_split_w1": lambda y, w0, a1, b1: _li(y, w0, a1 + b1),
    "li_split3": lambda y, a0, b0, c0, a1, b1, c1: _li(y, a0 + b0 + c0, a1 + b1 + c1),
    "li2": lambda y, w0, w1, w2, w3: _li(_li(y, w0, w1)[0], w2, w3, LI_K2, LI_A2),
    "duo": _duo,
    "duo_split2": lambda y, z, a, b: _duo(y, z, a + b),
    "duo_split3": lambda y, z, a, b, c: _duo(y, z, a + b + c),
    "fx1": _fx1,
}

TARGETS = {
    "li": Target("li", ("y", "w0", "w1"), ("y'",), 2, 2, "LOP3 u = y^w0^K; IMAD y' = u*A + w1"),
    "li_split_w0": Target("li_split_w0", ("y", "a0", "b0", "w1"), ("y'",), 3, 2, "IADD3 w0 = a0+b0; LOP3; IMAD"),
    "li_split_w1": Target("li_split_w1", ("y", "w0", "a1", "b1"), ("y'",), 3, 2, "LOP3; IMAD u*A + a1; IADD3 + b1"),
    "li_split2": Target("li_split2", ("y", "a0", "b0", "a1", "b1"), ("y'",), 4, 2,
                        "IADD3 w0 = a0+b0; LOP3; IMAD u*A + a1; IADD3 + b1"),
    "li_split3": Target("li_split3", ("y", "a0", "b0", "c0", "a1", "b1", "c1"), ("y'",), 4, 2,
                        "IADD3 w0 = a0+b0+c0; LOP3; IADD3 w1 = a1+b1+c1; IMAD"),
    "li2": Target("li2", ("y", "w0", "w1", "w2", "w3"), ("y''",), 4, 4, "two li half steps"),
    "duo": Target("duo", ("y", "z", "p"), ("y'", "z'"), 3, 1, "LOP3 u; IMAD y' = u*KA + z; LEA.HI z' = rotl(u,R) + y'"),
    "duo_split2": Target("duo_split2", ("y", "z", "a", "b"), ("y'", "z'"), 4, 1, "IADD3 p = a+b; then duo's 3"),
    "duo_split3": Target("duo_split3", ("y", "z", "a", "b", "c"), ("y'", "z'"), 4, 1, "IADD3 p = a+b+c; then duo's 3"),
    "fx1": Target("fx1", ("y", "z", "p"), ("y'", "z'"), 4, 1, "IMAD p*k; then duo's 3"),
}


# ---- the instruction models (z3, symbolic parameters) ---------------------------------------------------------------------

OPS = ("LOP3", "IADD3", "IMAD", "IMAD.HI", "IMAD.WIDE", "IADD64", "SHF", "LEA", "LEA.HI", "PRMT", "IMNMX", "UNARY", "SEL",
       "IDP.4A")
TWO_OUT = {"IMAD.WIDE", "IADD64"}
PIPE = {"LOP3": "alu", "IADD3": "alu", "IMAD": "imad", "IMAD.HI": "imad", "IMAD.WIDE": "imad", "IADD64": "alu",
        "SHF": "alu", "LEA": "alu", "LEA.HI": "alu", "PRMT": "alu", "IMNMX": "alu", "UNARY": "alu", "SEL": "alu",
        "IDP.4A": "imad"}
UNARY_FNS = ("POPC", "CLZ", "BREV", "NOT", "IABS", "SGXT", "BMSK", "FLO")
CMPS = ("LT", "LE", "EQ", "NE", "GT", "GE")


def _bv(v, w=W):
    return z3.BitVecVal(v, w)


def _lop3(a, b, c, lut):
    out = _bv(0)
    for i in range(8):
        ta = a if (i >> 2) & 1 else ~a
        tb = b if (i >> 1) & 1 else ~b
        tc = c if i & 1 else ~c
        bit = z3.Extract(i, i, lut)
        out = out | (z3.SignExt(W - 1, bit) & ta & tb & tc)
    return out


def _neg_if(x, s):
    return z3.If(s, -x, x)


def _prod(a, b):
    """The unsigned and signed 64-bit products of a and b, from one multiplier."""
    pu = z3.ZeroExt(W, a) * z3.ZeroExt(W, b)
    hi_u = z3.Extract(2 * W - 1, W, pu)
    lo = z3.Extract(W - 1, 0, pu)
    hi_s = hi_u - z3.If(a < 0, b, _bv(0)) - z3.If(b < 0, a, _bv(0))
    return lo, hi_u, hi_s


def _n64(n):
    return n if n.size() == 2 * W else z3.ZeroExt(2 * W - n.size(), n)


def _funnel_r(lo, hi, n, arith):
    """lo({hi:lo} >> n), with an arithmetic fill when ``arith``; n of any width, a count of 64 or more shifts out."""
    x = z3.Concat(hi, lo)
    return z3.Extract(W - 1, 0, z3.If(arith, x >> _n64(n), z3.LShR(x, _n64(n))))


def _funnel_l(lo, hi, n):
    """hi({hi:lo} << n), n of any width."""
    x = z3.Concat(hi, lo)
    return z3.Extract(2 * W - 1, W, x << _n64(n))


def _shf_count(c, fn, n_imm):
    """SHF's count: the immediate, or register c wrapped (& 31, & 63) or clamped (min(c, 32), min(c, 64))."""
    c64 = z3.ZeroExt(W, c)
    cases = [z3.ZeroExt(2 * W - 6, n_imm), c64 & 31, c64 & 63,
             z3.If(z3.ULT(c, 32), c64, z3.BitVecVal(32, 2 * W)), z3.If(z3.ULT(c, 64), c64, z3.BitVecVal(64, 2 * W))]
    v = cases[0]
    for k in range(1, len(cases)):
        v = z3.If(fn == k, cases[k], v)
    return v


def _clz(a):
    out = _bv(W)
    for i in range(W):
        out = z3.If(z3.Extract(i, i, a) == 1, _bv(W - 1 - i), out)
    return out


def _popc(a):
    return sum((z3.ZeroExt(W - 1, z3.Extract(i, i, a)) for i in range(W)), _bv(0))


def _brev(a):
    return z3.Concat(*[z3.Extract(i, i, a) for i in range(W)])


#: PRMT's special modes (PTX prmt .f4e .b4e .rc8 .ecl .ecr .rc16): per c[1:0], the source byte of d's bytes 3..0
PRMT_MODES = {
    "f4e": ("3210", "4321", "5432", "6543"), "b4e": ("5670", "6701", "7012", "0123"),
    "rc8": ("0000", "1111", "2222", "3333"), "ecl": ("3210", "3211", "3222", "3333"),
    "ecr": ("0000", "1110", "2210", "3210"), "rc16": ("1010", "3232", "1010", "3232"),
}


def _prmt(a, b, c, fn=None):
    """PRMT: default mode (fn 0) or a special mode (fn 1..6, PRMT_MODES' order)."""
    x = z3.Concat(b, a)
    bytes_ = [z3.Extract(8 * i + 7, 8 * i, x) for i in range(8)]
    out = []
    for i in range(4):
        nib = z3.Extract(4 * i + 3, 4 * i, c)
        sel = bytes_[0]
        for j in range(1, 8):
            sel = z3.If(z3.Extract(2, 0, nib) == j, bytes_[j], sel)
        sign = z3.If(z3.Extract(7, 7, sel) == 1, z3.BitVecVal(0xFF, 8), z3.BitVecVal(0, 8))
        out.append(z3.If(z3.Extract(3, 3, nib) == 1, sign, sel))
    v = z3.Concat(*reversed(out))
    if fn is None:
        return v
    c2 = z3.Extract(1, 0, c)
    for m, table in enumerate(PRMT_MODES.values(), start=1):
        mv = None
        for s, digits in enumerate(table):
            word = z3.Concat(*[bytes_[int(ch)] for ch in digits])
            mv = word if mv is None else z3.If(c2 == s, word, mv)
        v = z3.If(fn == m, mv, v)
    return v


class Slot:
    """One instruction slot: symbolic opcode, operand selectors, immediates and parameters."""

    def __init__(self, i: int, n_avail: int, ops: tuple[str, ...]):
        p = f"s{i}_"
        self.i, self.ops = i, ops
        self.op = z3.Int(p + "op")
        self.sel = [z3.Int(p + f"sel{j}") for j in range(4)]
        self.imm = [z3.BitVec(p + f"imm{j}", W) for j in range(4)]
        self.lut = z3.BitVec(p + "lut", 8)
        self.sg = [z3.Bool(p + f"neg{j}") for j in range(3)]
        self.signed = z3.Bool(p + "signed")
        self.mode = z3.BitVec(p + "mode", 3)   # SHF: bit0 left, bit1 arith, bit2 count from c; IMNMX: bit0 max
        self.n = z3.BitVec(p + "n", 6)
        self.fn = z3.BitVec(p + "fn", 3)
        self.n_avail = n_avail

    def constraints(self):
        cs = [self.op >= 0, self.op < len(self.ops)]
        for s in self.sel:
            cs += [s >= 0, s <= self.n_avail]   # n_avail = the slot's immediate
        return cs

    def operands(self, avail):
        out = []
        for j, s in enumerate(self.sel):
            v = self.imm[j]
            for k in range(len(avail) - 1, -1, -1):
                v = z3.If(s == k, avail[k], v)
            out.append(v)
        return out

    def outputs(self, avail):
        """(lo, hi) of this slot on one input vector; hi is 0 for one-register ops."""
        a, b, c, d = self.operands(avail)
        lo_p, hi_u, hi_s = _prod(a, b) if {"IMAD", "IMAD.HI", "IMAD.WIDE"} & set(self.ops) else (None, None, None)
        res = {}
        for name in self.ops:
            if name == "LOP3":
                res[name] = (_lop3(a, b, c, self.lut), _bv(0))
            elif name == "IADD3":
                res[name] = (_neg_if(a, self.sg[0]) + _neg_if(b, self.sg[1]) + _neg_if(c, self.sg[2]), _bv(0))
            elif name == "IMAD":
                res[name] = (lo_p + _neg_if(c, self.sg[2]), _bv(0))
            elif name == "IMAD.HI":
                res[name] = (z3.If(self.signed, hi_s, hi_u) + c, _bv(0))
            elif name == "IMAD.WIDE":
                cd = z3.Concat(d, c)
                prod = z3.Concat(z3.If(self.signed, hi_s, hi_u), lo_p)
                s = prod + cd
                res[name] = (z3.Extract(W - 1, 0, s), z3.Extract(2 * W - 1, W, s))
            elif name == "IADD64":
                s = z3.Concat(b, a) + z3.Concat(d, c)
                res[name] = (z3.Extract(W - 1, 0, s), z3.Extract(2 * W - 1, W, s))
            elif name == "SHF":
                left = z3.Extract(0, 0, self.mode) == 1
                arith = z3.Extract(1, 1, self.mode) == 1
                n = _shf_count(c, self.fn, self.n)
                res[name] = (z3.If(left, _funnel_l(a, b, n), _funnel_r(a, b, n, arith)), _bv(0))
            elif name == "LEA":
                res[name] = ((a << z3.ZeroExt(W - 6, self.n)) + c, _bv(0))
            elif name == "LEA.HI":
                res[name] = (_funnel_l(a, b, self.n) + c, _bv(0))
            elif name == "PRMT":
                res[name] = (_prmt(a, b, c, self.fn), _bv(0))
            elif name == "IDP.4A":
                sa = z3.Extract(0, 0, self.mode) == 1
                sb = z3.Extract(1, 1, self.mode) == 1
                acc = c
                for i in range(4):
                    xa, xb = z3.Extract(8 * i + 7, 8 * i, a), z3.Extract(8 * i + 7, 8 * i, b)
                    acc = acc + z3.If(sa, z3.SignExt(24, xa), z3.ZeroExt(24, xa)) * \
                        z3.If(sb, z3.SignExt(24, xb), z3.ZeroExt(24, xb))
                res[name] = (acc, _bv(0))
            elif name == "IMNMX":
                mx = z3.Extract(0, 0, self.mode) == 1
                lt = z3.If(self.signed, a < b, z3.ULT(a, b))
                res[name] = (z3.If(lt != mx, a, b), _bv(0))
            elif name == "UNARY":
                n32 = z3.ZeroExt(W - 6, self.n)
                fns = [_popc(a), _clz(a), _brev(a), ~a, z3.If(a < 0, -a, a),
                       z3.If(self.n == 0, a, (a << (W - n32)) >> (W - n32)),   # SGXT: sign-extend the low n bits
                       z3.If(z3.UGE(n32, W), _bv(M32), (_bv(1) << n32) - 1),   # BMSK: n ones
                       z3.If(a == 0, _bv(M32), _bv(W - 1) - _clz(a))]          # FLO.U32: the highest set bit's index
                v = fns[0]
                for k in range(1, len(fns)):
                    v = z3.If(self.fn == k, fns[k], v)
                res[name] = (v, _bv(0))
            elif name == "SEL":
                cmps = [z3.If(self.signed, a < b, z3.ULT(a, b)), z3.If(self.signed, a <= b, z3.ULE(a, b)), a == b,
                        a != b, z3.If(self.signed, a > b, z3.UGT(a, b)), z3.If(self.signed, a >= b, z3.UGE(a, b))]
                p = cmps[0]
                for k in range(1, 6):
                    p = z3.If(self.fn == k, cmps[k], p)
                res[name] = (z3.If(p, c, d), _bv(0))
        lo, hi = res[self.ops[0]]
        for k in range(1, len(self.ops)):
            lo = z3.If(self.op == k, res[self.ops[k]][0], lo)
            hi = z3.If(self.op == k, res[self.ops[k]][1], hi)
        return lo, hi


def build(target: Target, k: int, ops: tuple[str, ...] = OPS):
    """The k-slot program's symbolic structure and a function from an input vector to (target constraints)."""
    n_in = len(target.inputs)
    slots = [Slot(i, n_in + 1 + 2 * i, ops) for i in range(k)]
    n_regs = n_in + 1 + 2 * k
    osel = [z3.Int(f"out{j}") for j in range(len(target.outputs))]
    structure = [c for s in slots for c in s.constraints()]
    structure += [z3.And(o >= 0, o < n_regs) for o in osel]
    # symmetry: a one-register slot's high half is never read; an output is never an input or RZ (targets aren't inputs)
    return slots, osel, structure


def run_vector(target: Target, slots, osel, xs: list):
    """Constraints that the program, on inputs xs (ints or z3 terms), produces target(xs)."""
    avail = [x if not isinstance(x, int) else _bv(x) for x in xs] + [_bv(0)]
    for s in slots:
        lo, hi = s.outputs(avail)
        avail = avail + [lo, hi]
    want = target.sym([x if not isinstance(x, int) else _bv(x) for x in xs])
    cs = []
    for o, w in zip(osel, want):
        v = avail[0]
        for k in range(1, len(avail)):
            v = z3.If(o == k, avail[k], v)
        cs.append(v == w)
    return cs


def _vectors(target: Target, n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    m = len(target.inputs)
    vs = [[0] * m, [M32] * m]
    while len(vs) < n:
        vs.append([rng.getrandbits(W) for _ in range(m)])
    return vs[:n]


def _fix(slots, osel, model):
    return [(model.eval(v, model_completion=True)) for s in slots
            for v in [s.op, *s.sel, *s.imm, s.lut, *s.sg, s.signed, s.mode, s.n, s.fn]] + \
           [model.eval(o, model_completion=True) for o in osel]


def _pin(slots, osel, model):
    vs = [v for s in slots for v in [s.op, *s.sel, *s.imm, s.lut, *s.sg, s.signed, s.mode, s.n, s.fn]] + list(osel)
    return [v == model.eval(v, model_completion=True) for v in vs]


def describe(slots, osel, model, target: Target, ops=OPS) -> list[str]:
    lines = []
    names = list(target.inputs) + ["RZ"]
    for s in slots:
        op = ops[model.eval(s.op, model_completion=True).as_long()]
        args = []
        for j in range(4):
            sv = model.eval(s.sel[j], model_completion=True).as_long()
            args.append(names[sv] if sv < len(names) else hex(model.eval(s.imm[j], model_completion=True).as_long()))
        extra = {"lut": hex(model.eval(s.lut, model_completion=True).as_long()),
                 "n": model.eval(s.n, model_completion=True).as_long(),
                 "mode": model.eval(s.mode, model_completion=True).as_long(),
                 "signed": z3.is_true(model.eval(s.signed, model_completion=True)),
                 "neg": [z3.is_true(model.eval(g, model_completion=True)) for g in s.sg],
                 "fn": model.eval(s.fn, model_completion=True).as_long()}
        lines.append(f"r{s.i} = {op}({', '.join(args)}) {extra}")
        names += [f"r{s.i}", f"r{s.i}.hi"]
    outs = [names[model.eval(o, model_completion=True).as_long()] for o in osel]
    lines.append("out = " + ", ".join(outs))
    return lines


def synth(target: Target, k: int, *, vectors: int = 4, seed: int = 1, timeout_s: int | None = None, max_iter: int = 40,
          ops: tuple[str, ...] = OPS) -> dict:
    """CEGIS for a k-slot program computing ``target``. Returns the verdict: unsat (a proof over the modelled set), sat
    (a program verified for all inputs) or unknown (timeout)."""
    slots, osel, structure = build(target, k, ops)
    s = z3.Solver()
    if timeout_s:
        s.set("timeout", timeout_s * 1000)
    s.add(structure)
    xs_list = _vectors(target, vectors, seed)
    for xs in xs_list:
        s.add(run_vector(target, slots, osel, xs))
    for it in range(max_iter):
        r = s.check()
        if r == z3.unsat:
            return {"target": target.name, "k": k, "verdict": "unsat", "vectors": len(xs_list), "iterations": it,
                    "ops": list(ops)}
        if r == z3.unknown:
            return {"target": target.name, "k": k, "verdict": "unknown", "vectors": len(xs_list), "iterations": it,
                    "reason": s.reason_unknown(), "ops": list(ops)}
        m = s.model()
        cex = verify(target, slots, osel, m, ops, timeout_s)
        if cex is None:
            return {"target": target.name, "k": k, "verdict": "sat", "vectors": len(xs_list), "iterations": it,
                    "program": describe(slots, osel, m, target, ops), "ops": list(ops)}
        xs_list.append(cex)
        s.add(run_vector(target, slots, osel, cex))
    return {"target": target.name, "k": k, "verdict": "unknown", "reason": "max_iter", "vectors": len(xs_list),
            "ops": list(ops)}


def verify(target: Target, slots, osel, model, ops, timeout_s: int | None):
    """None if the model's concrete program computes ``target`` on every input, else a counterexample input."""
    xs = [z3.BitVec(f"x_{n}", W) for n in target.inputs]
    v = z3.Solver()
    if timeout_s:
        v.set("timeout", timeout_s * 1000)
    v.add(_pin(slots, osel, model))
    v.add(z3.Not(z3.And(run_vector(target, slots, osel, xs))))
    r = v.check()
    if r == z3.unsat:
        return None
    if r == z3.unknown:
        raise RuntimeError(f"verify undecided for {target.name}: {v.reason_unknown()}")
    m = v.model()
    return [m.eval(x, model_completion=True).as_long() for x in xs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("target", choices=sorted(TARGETS))
    c.add_argument("k", type=int)
    c.add_argument("--vectors", type=int, default=4)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--timeout", type=int, default=600)
    c.add_argument("--ops", default=",".join(OPS))
    sw = sub.add_parser("sweep")
    sw.add_argument("--out", required=True)
    sw.add_argument("--cells", required=True, help="TARGET:K[:MENU],... with MENU all (default), alu or imad")
    sw.add_argument("--procs", type=int, default=4)
    sw.add_argument("--timeout", type=int, default=3600)
    sw.add_argument("--vectors", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "cell":
        ops = tuple(a.ops.split(","))
        r = synth(TARGETS[a.target], a.k, vectors=a.vectors, seed=a.seed, timeout_s=a.timeout, ops=ops)
        print(json.dumps(r, indent=1))
        return 0
    cells = [(c.split(":") + ["all"])[:3] for c in a.cells.split(",")]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(a.procs) as ex:
        futs = [ex.submit(_cell, t, int(k), a.vectors, a.timeout, MENUS[m]) for t, k, m in cells]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(json.dumps(r), flush=True)
            (out / "fold_sat.json").write_text(json.dumps(results, indent=1))
    return 0


#: sweep menus: every modelled op, the ALU pipe's alone, the IMAD pipe's alone
MENUS = {"all": OPS, "alu": tuple(o for o in OPS if PIPE[o] == "alu"),
         "imad": tuple(o for o in OPS if PIPE[o] == "imad")}


def _cell(t, k, vectors, timeout, ops=OPS):
    return synth(TARGETS[t], k, vectors=vectors, timeout_s=timeout, ops=ops)


if __name__ == "__main__":
    sys.exit(main())
