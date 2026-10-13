"""`dsa`'s DeepSeek Sparse Attention indexer on bits: version 2 of each Definition, its word body line for line over Boolean
Definitions (the bodies are `dsa`'s, written once over `dsa.Ops`), so each one's word view is its `_v1` at the same statics; the
logits' version 4, whose word view is `_v3`.

    U32And_v2, I32Sub_v2, I32ToF32_v2, F32CeilFtz_v2     traced: and.b32, sub.s32, cvt.rn.f32.s32, cvt.rpi.ftz.f32.f32
    DsaUe8m0Scale_v2, DsaUe8m0Quant_v2{N}
    DsaIndexerKRow_v2{D,ROPE,EPS}, DsaIndexerQHead_v2{D,ROPE,NH}, DsaIndexerQ_v2{H,D,ROPE}
    DsaMqaLogit_v4{H,D}, DsaMqaLogits_v4{N,H,D}          the H200's (`DsaMqaLogit_v3`); _v2 is the refuted `_v1` on bits
    DsaTopKShortRow_v2{TOPK}                            traced at its TOPK: slot i's bits are i's, each one bit `len <= i`
    DsaTopKSelection_v2{TOPK,N}                         traced at its statics: a decoder per slot, unsigned keys
    DsaTopKSelection_v4{TOPK,N,C}                       DsaTopKSelection_v3{TOPK,N}, traced: DsaTopKChunk_v1{TOPK,N,C,H}'s check
                                                        per chunk of C positions, then DsaTopKFinal_v1{TOPK,N,C}'s; at the served
                                                        statics a Program calls those at its root (`topk_selection_calls`)

The callees are core's Boolean FP32 pieces (`verity.ml.circuits.ops.boolean`'s `elementwise`, `attention`, `cast`, `mufu`, `scalar`) and
`gemm.HopperE4m3QgmmaDot32_v2`; a constant word is constant bits and `-x` one NOT gate on x's sign bit (`F32Neg_v1`).  Each callee is
its word Definition on every word, NaN payloads included.  No table or lookup gate.

Each composite is one traced circuit (`flat`): its body over those callees, compiled (`verity.ml.circuits.evaluation.bits.compile`) and its
gates traced again (`trace.trace`), so a constant operand is folded and a value its calls share (the log2f polynomial's ``t``,
the scale every quantized element divides by, a row's mean) is computed once (`Q_word`'s no-recompute rule).
"""
from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from functools import reduce

import numpy as np

from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.ops.boolean import cast as BC
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp as FP
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.circuits.boolean.forms import ONE, ZERO, NOT, OR, add, and_reduce, compress_columns, const_bits, mux, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry import sampling_boolean as SB
from verity_vllm.program.registry.boolean_attention import _Bits
from verity_vllm.program.registry.topp_word_gates import I32ToF32 as I32ToF32V1

BIT = BL.BIT
BF16 = Array(16, BIT)
F32 = I32 = Array(32, BIT)
E4M3 = Array(8, BIT)
_AB = (("a", I32), ("b", I32))

U32AndV2 = definition("U32And", 2, lambda C, a, b: [C.AND(x, y) for x, y in zip(a, b)], _AB, I32, word=D.U32And,
                      doc="and.b32 on bits: U32And_v1")
I32SubV2 = definition("I32Sub", 2, lambda C, a, b: sub(C, a, b, 32)[0], _AB, I32, word=D.I32Sub,
                      doc="sub.s32 on bits: I32Sub_v1, a - b modulo 2^32")


def i32_to_f32(C, a):
    """`cvt.rn.f32.s32`: |a| (2^31 for INT_MIN) rounded to f32 at exponent 158 by `fp.round_f32`, the sign a's."""
    return FP.round_f32(C, a[31], const_bits(158, 9), 0, FP.neg_if(C, list(a), a[31]), ZERO, underflow=False)


I32ToF32V2 = definition("I32ToF32", 2, i32_to_f32, (("a", I32),), F32, word=I32ToF32V1,
                        doc="cvt.rn.f32.s32 on bits: I32ToF32_v1")


def f32_ceil_ftz(C, x):
    """`cvt.rpi.ftz.f32.f32` (`dsa.f32_ceil_ftz`): for 127 <= e < 150 the fraction mask is the 150 - e low bits (`fp.shl` of ones),
    the result x with them cleared, plus one past them when x is positive and any was set; below, +1.0 for a positive normal and
    the zero of the sign otherwise; from 150 x itself, a NaN 0x7FFFFFFF."""
    m, e, s = list(x[:23]), list(x[23:31]), x[31]
    nan = C.AND(and_reduce(C, e), or_reduce(C, m))
    lt127 = sub(C, e, const_bits(127, 8), 8)[1]
    lt150 = sub(C, e, const_bits(150, 8), 8)[1]
    keep = FP.shl(C, [ONE] * 23, sub(C, const_bits(150, 8), e, 8)[0][:5])
    frac = or_reduce(C, [C.AND(b, NOT(k)) for b, k in zip(m, keep)])
    trunc = [C.AND(b, k) for b, k in zip(m, keep)] + e
    up = add(C, [OR(C, b, NOT(k)) for b, k in zip(m, keep)] + e, [ONE], width=31)
    reg = mux(C, C.AND(NOT(s), frac), up, trunc)
    one = C.AND(NOT(s), or_reduce(C, e))
    small = [one if b.c else ZERO for b in const_bits(0x3F800000, 31)]
    out = mux(C, lt127, small, mux(C, lt150, reg, list(x[:31]))) + [s]
    return mux(C, nan, const_bits(0x7FFFFFFF, 32), out)


F32CeilFtzV2 = definition("F32CeilFtz", 2, f32_ceil_ftz, (("x", F32),), F32, word=D.F32CeilFtz,
                          doc="cvt.rpi.ftz.f32.f32 on bits: F32CeilFtz_v1")

BITS = dict(cvt=BC.Bf16ToF32, add=EW.F32Add, mul=EW.F32Mul, fma=EW.F32Fma, fma_ftz=BA.F32FmaFtz, div=EW.F32Div, rcp=BM.DivFullRcp,
            scale_a=BM.DivFullScaleA, rsqrt=BM.RsqrtApprox, ex2=BM.MufuEx2Ftz, fabs=SC.F32Fabs, fmaxf=SC.F32Fmaxf,
            e4m3=SC.F32ToE4m3Sat, iadd=EW.I32Add, iand=U32AndV2, isub=I32SubV2, i2f=I32ToF32V2, ceil=F32CeilFtzV2,
            isel=SC.SelectI32, dot=BG.HopperE4m3QgmmaDot32)


def bit_ops(B) -> D.Ops:
    c = _Bits(B)
    return D.Ops(B, BITS, lambda v: c.word(v, 32),
                 lambda x: Coll(F32, concat([x.refs.slice(0, 31), B.call(BL.NOT, x[31]).refs])))


def replay(c: BS.Compiled, C, args: list) -> list:
    """The compiled body ``c``'s gates on ``C`` over the forms ``args`` (its parameter leaves), its calls inlined: its return's
    forms."""
    W = [ZERO, ONE, *args] + [None] * (c.size - 2 - len(args))
    for step in c.steps:
        if step[0] == "gates":
            for kind, out, a, b in step[1]:
                for o, x, y in zip(out.tolist(), a.tolist(), b.tolist()):
                    W[o] = C.AND(W[x], W[y]) if kind == 0 else W[x] ^ W[y] if kind == 1 else NOT(W[x])
        elif step[0] == "call":
            _, child, members, base, w = step
            for m, row in enumerate(members.tolist()):
                W[base + m * w:base + (m + 1) * w] = replay(child, C, [W[i] for i in row])
        else:
            raise TypeError(f"{c.target}: a scan is not inlined")
    return [W[i] for i in c.ret_idx.tolist()]


def flat(name: str, version: int, statics: Sequence[str], signature: Callable, body: Callable, word,
         lifted: Sequence[str] = ()) -> CompositeDefinition:
    """The Boolean Definition ``name_v{version}{statics}`` whose circuit is ``body`` (over Boolean Definitions, as `EW.on_bits`
    takes it, its ``lifted`` statics Boolean Definitions read as their word views) compiled and traced as one circuit per binding
    (`EW.specialized`), its word view ``word`` at the same statics."""
    inner = EW.on_bits(name, version, statics, signature, body, word, register=False, lifted=lifted)

    def build(S):
        c = BS.compile(bind(inner, **{k: getattr(S, k) for k in statics}))
        widths = BL.elements(signature(S)[1])
        ends = [sum(widths[:i + 1]) for i in range(len(widths))]

        def run(C, *elements):
            out = replay(c, C, [f for e in elements for f in e])
            return [out[e - w:e] for w, e in zip(widths, ends)]
        return run

    return EW.specialized(name, version, statics, build, signature, doc=body.__doc__ or "", word=inner.word)


def _scale(B, S, amax):
    """`DsaUe8m0Scale_v1`'s body on bits."""
    return D.ue8m0_scale(bit_ops(B), amax)


DsaUe8m0ScaleV2 = flat("DsaUe8m0Scale", 2, [], lambda S: ((("amax", F32),), F32), _scale, D.DsaUe8m0Scale)


def _quant(B, S, x):
    """`DsaUe8m0Quant_v1`'s body on bits."""
    return D.ue8m0_quant(bit_ops(B), [x[i] for i in range(S.N)], DsaUe8m0ScaleV2, E4M3)


DsaUe8m0QuantV2 = flat("DsaUe8m0Quant", 2, ["N"], lambda S: ((("x", Array(S.N, F32)),), Tuple(Array(S.N, E4M3), F32)), _quant,
                             D.DsaUe8m0Quant)


def _k_row(B, S, k, w, b, cs):
    """`DsaIndexerKRow_v1`'s body on bits."""
    return D.k_slot(bit_ops(B), S, k, w, b, cs, bind(DsaUe8m0QuantV2, N=S.D), F32)


DsaIndexerKRowV2 = flat(
    "DsaIndexerKRow", 2, ["D", "ROPE", "EPS"],
    lambda S: ((("k", Array(S.D, BF16)), ("w", Array(S.D, F32)), ("b", Array(S.D, F32)), ("cs", Array(S.ROPE, BF16))),
               Tuple(Array(S.D, E4M3), F32)),
    _k_row, D.DsaIndexerKRow)


def _q_head(B, S, q, cs, w):
    """`DsaIndexerQHead_v1`'s body on bits."""
    return D.q_head(bit_ops(B), S, q, cs, w, bind(DsaUe8m0QuantV2, N=S.D), F32)


DsaIndexerQHeadV2 = flat(
    "DsaIndexerQHead", 2, ["D", "ROPE", "NH"],
    lambda S: ((("q", Array(S.D, BF16)), ("cs", Array(S.ROPE, BF16)), ("w", BF16)), Tuple(Array(S.D, E4M3), F32)), _q_head,
    D.DsaIndexerQHead)


def _q(B, S, q, cs, w):
    """`DsaIndexerQ_v1`'s body on bits."""
    return D.heads(B, bind(DsaIndexerQHeadV2, D=S.D, ROPE=S.ROPE, NH=S.H), S.H, S.D, q, cs, w, E4M3, F32)


DsaIndexerQV2 = flat(
    "DsaIndexerQ", 2, ["H", "D", "ROPE"],
    lambda S: ((("q", Array(S.H, Array(S.D, BF16))), ("cs", Array(S.ROPE, BF16)), ("w", Array(S.H, BF16))),
               Tuple(Array(S.H, Array(S.D, E4M3)), Array(S.H, F32))),
    _q, D.DsaIndexerQ)


def _logit(B, S, q, w, k, ks):
    """`DsaMqaLogit_v1`'s body on bits."""
    return D.mqa_logit(bit_ops(B), S.H, S.D, q, w, k, ks)


DsaMqaLogitV2 = flat(
    "DsaMqaLogit", 2, ["H", "D"],
    lambda S: ((("q", Array(S.H, Array(S.D, E4M3))), ("w", Array(S.H, F32)), ("k", Array(S.D, E4M3)), ("ks", F32)), F32),
    _logit, D.DsaMqaLogit)


def _logits(B, S, q, w, k, ks):
    """`DsaMqaLogits_v1`'s body on bits."""
    return B.batch(bind(DsaMqaLogitV2, H=S.H, D=S.D), q, w, k, ks, axes=(None, None, 0, 0))


DsaMqaLogitsV2 = flat(
    "DsaMqaLogits", 2, ["N", "H", "D"],
    lambda S: ((("q", Array(S.H, Array(S.D, E4M3))), ("w", Array(S.H, F32)), ("k", Array(S.N, Array(S.D, E4M3))),
                ("ks", Array(S.N, F32))), Array(S.N, F32)),
    _logits, D.DsaMqaLogits)


def _logit_contracted(B, S, q, w, k, ks):
    """`DsaMqaLogit_v3`'s body on bits."""
    return D.mqa_logit(bit_ops(B), S.H, S.D, q, w, k, ks, contracted=True)


DsaMqaLogitV4 = flat(
    "DsaMqaLogit", 4, ["H", "D"],
    lambda S: ((("q", Array(S.H, Array(S.D, E4M3))), ("w", Array(S.H, F32)), ("k", Array(S.D, E4M3)), ("ks", F32)), F32),
    _logit_contracted, D.DsaMqaLogitV3)


def _logits_contracted(B, S, q, w, k, ks):
    """`DsaMqaLogits_v3`'s body on bits."""
    return B.batch(bind(DsaMqaLogitV4, H=S.H, D=S.D), q, w, k, ks, axes=(None, None, 0, 0))


DsaMqaLogitsV4 = flat(
    "DsaMqaLogits", 4, ["N", "H", "D"],
    lambda S: ((("q", Array(S.H, Array(S.D, E4M3))), ("w", Array(S.H, F32)), ("k", Array(S.N, Array(S.D, E4M3))),
                ("ks", Array(S.N, F32))), Array(S.N, F32)),
    _logits_contracted, D.DsaMqaLogitsV3)


def _topk(S):
    """`DsaTopKShortRow_v1{TOPK}` traced: `len <= i` (signed: the borrow of `(i ^ 2^31) - (len ^ 2^31)` negated) is one bit per
    slot, and slot i is i's bits each ORed with it (-1 when set)."""
    def build(C, n):
        flip = list(n[:31]) + [NOT(n[31])]
        out = []
        for i in range(S.TOPK):
            ci = const_bits(i ^ (1 << 31), 32)
            le = NOT(sub(C, ci, flip, 32)[1])
            out.append([ONE if b else le for b in ((i >> j) & 1 for j in range(32))])
        return out
    return build


DsaTopKShortRowV2 = EW.specialized("DsaTopKShortRow", 2, ["TOPK"], _topk, lambda S: ((("len", I32),), Array(S.TOPK, I32)),
                                   doc=_topk.__doc__, word=lambda b: bind(D.DsaTopKShortRow, TOPK=b["TOPK"]))


def decode(C, bits: list, n: int) -> list:
    """The first n lines of the decoder of the unsigned index ``bits`` (n <= 2^len(bits)): line j is `bits == j`, one AND of a line
    of the low half's decoder and one of the high half's."""
    if len(bits) == 1:
        return [NOT(bits[0]), bits[0]][:n]
    h = len(bits) // 2
    lo, hi = decode(C, bits[:h], min(n, 1 << h)), decode(C, bits[h:], -(-n >> h))
    return [C.AND(hi[j >> h], lo[j & ((1 << h) - 1)]) for j in range(n)]


def _selection(S):
    """`DsaTopKSelection_v1{TOPK, N}` traced.  `len <= j` is `_topk`'s comparison; keys compare unsigned as `order_key ^ 2^31`
    (`convert_to_uint32_v2`: bit 31 the sign's NOT, bit b < 31 x_b ^ x_31), the sentinels INT_MAX and INT_MIN all ones and all
    zeros; `idx < len` unsigned is the range check (len > TOPK > 0 wherever it is read).  Slot i's membership lines decode its low
    ceil(log2 N) bits (an index in range has no other) and position j's membership is their XOR over the slots: with every index
    in range, TOPK odd positions means TOPK distinct indices, as TOPK members by the OR does, so the bit is v1's.  The count is
    `compress_columns`."""
    D.check_selection_statics(S.TOPK, S.N)
    T, N = S.TOPK, S.N
    p = (N - 1).bit_length()

    def le_u(C, a, b):
        return NOT(sub(C, b, a, 32)[1])

    def build(C, *els):
        x, n, idx = els[:N], els[N], els[N + 1:]
        flip = list(n[:31]) + [NOT(n[31])]
        le = [NOT(sub(C, const_bits(j ^ (1 << 31), 32), flip, 32)[1]) for j in range(N + 1)]
        short = and_reduce(C, [b if (i >> k) & 1 else NOT(b ^ le[i]) for i in range(T) for k, b in enumerate(idx[i])])
        inside = and_reduce(C, [sub(C, idx[i], n, 32)[1] for i in range(T)])
        lines = [decode(C, list(idx[i][:p]), N) for i in range(T)]
        member = [reduce(operator.xor, (lines[i][j] for i in range(T))) for j in range(N)]
        count = compress_columns(C, [member])
        tight = and_reduce(C, [c if (T >> k) & 1 else NOT(c) for k, c in enumerate(count)])
        key = [[b ^ v[31] for b in v[:31]] + [NOT(v[31])] for v in x]
        least = D.pairwise([mux(C, member[j], key[j], const_bits(D.M32, 32)) for j in range(N)],
                           lambda a, b: mux(C, le_u(C, a, b), a, b))
        most = D.pairwise([mux(C, OR(C, member[j], le[j]), const_bits(0, 32), key[j]) for j in range(N)],
                          lambda a, b: mux(C, le_u(C, a, b), b, a))
        long_ok = and_reduce(C, [inside, tight, le_u(C, most, least), le[N]])
        return [OR(C, C.AND(le[T], short), C.AND(NOT(le[T]), long_ok))]
    return build


DsaTopKSelectionV2 = EW.specialized(
    "DsaTopKSelection", 2, ["TOPK", "N"], _selection,
    lambda S: ((("logits", Array(S.N, F32)), ("len", I32), ("idx", Array(S.TOPK, I32))), BIT),
    doc=_selection.__doc__, word=lambda b: bind(D.DsaTopKSelection, TOPK=b["TOPK"], N=b["N"]))


# --- DsaTopKSelection_v4: DsaTopKSelection_v3 chunk by chunk ----------------------------------------------------------------------

#: the positions of one chunk at the served statics: a chunk's unit reads 2^14 logits, whose row hashes fit C-Flock's 2^27 block
CHUNK = 1 << 14
#: the form size past which a comparator's swap (four ANDs for its one) or a move (one more) writes fresh bits, so no form, and
#: no matrix row, grows with the network
FORM_BOUND = 8


def _or(C, a, b):
    """a OR b as one fresh bit (`forms.OR`'s form also carries a and b)."""
    return NOT(C.AND(NOT(a), NOT(b)))


def _maj(C, x, y, c):
    """maj(x, y, c) in one AND whose form is the smaller of x and y and that bit, so a carry chain's forms stay bounded
    (`forms.maj` keeps the carry's, which grows along the chain); with a constant operand, one fresh bit."""
    for a, b, k in ((x, y, c), (x, c, y), (y, c, x)):
        if k.is_const():
            return _or(C, a, b) if k.c else C.AND(a, b)
    p, q = (x, y) if len(x.s) <= len(y.s) else (y, x)
    return p ^ C.AND(p ^ q, p ^ c)


def _carry(C, a, b, cin):
    """The carry out of a + b + cin, unsigned, a and b of one width."""
    for x, y in zip(a, b, strict=True):
        cin = _maj(C, x, y, cin)
    return cin


def gt_u(C, a, b):
    """a > b unsigned: the carry out of a + ~b."""
    return _carry(C, a, [NOT(y) for y in b], ZERO)


def _plus(C, a, k: int) -> list:
    """(a + k) mod 2^len(a) for the constant k, each carry one fresh bit."""
    out, c = [], ZERO
    for i, x in enumerate(a):
        kb = ONE if (k >> i) & 1 else ZERO
        out.append(x ^ kb ^ c)
        if i + 1 < len(a):
            c = _maj(C, x, kb, c)
    return out


def _eq(C, a, b):
    return and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b, strict=True)])


def _key(v):
    """`order_key ^ 2^31` on bits (`convert_to_uint32_v2`): bit 31 the sign's NOT, bit b < 31 x_b ^ x_31."""
    return [b ^ v[31] for b in v[:31]] + [NOT(v[31])]


def merge_sort_layers(n: int) -> list[list[tuple[int, int]]]:
    """Batcher's odd-even merge sort on n keys as layers of disjoint comparators (a, b), a < b: the network on the next power of
    two without the comparators that reach position n or past it, which meet only the +inf padding (it never moves)."""
    layers = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            layer = [(i + j, i + j + k) for j in range(k % p, n - k, 2 * k) for i in range(min(k, n - j - k))
                     if (i + j) // (2 * p) == (i + j + k) // (2 * p)]
            if layer:
                layers.append(layer)
            k //= 2
        p *= 2
    return layers


def _swap(C, s, a, b):
    """(s ? b : a, s ? a : b) for one bit: one AND, each output a's or b's form and that bit, or past `FORM_BOUND` four fresh ANDs."""
    if len(a.s) + len(b.s) <= FORM_BOUND:
        m = C.AND(s, a ^ b)
        return a ^ m, b ^ m
    return C.AND(NOT(s), a) ^ C.AND(s, b), C.AND(s, a) ^ C.AND(NOT(s), b)


def sort_keys(C, keys: list) -> list:
    """``keys`` (unsigned, LSB first, one width) in ascending order by `merge_sort_layers`."""
    keys = list(keys)
    for layer in merge_sort_layers(len(keys)):
        for a, b in layer:
            s = gt_u(C, keys[a], keys[b])
            keys[a], keys[b] = map(list, zip(*(_swap(C, s, x, y) for x, y in zip(keys[a], keys[b]))))
    return keys


def therm(C, x: list, n: int) -> list:
    """[x > j] for j < n, x unsigned (LSB first): by its top bit, the rest's line ORed (j below the top bit's weight) or ANDed (at
    or above it), each line one fresh bit."""
    if n <= 0:
        return []
    if not x:
        return [ZERO] * n
    top, rest = x[-1], x[:-1]
    h = 1 << len(rest)
    sub = therm(C, rest, max(min(n, h), n - h))
    return [_or(C, top, t) for t in sub[:min(n, h)]] + [C.AND(top, t) for t in sub[:max(0, n - h)]]


def below(C, n, base: int, W: int) -> list:
    """[base + j < len] for j < W, len signed: r = len - base in 33 bits; r >= 2^w (w the bits of W - 1) is every line, and
    0 <= r < 2^w the thermometer of r's low w bits."""
    w = (W - 1).bit_length()
    r = _plus(C, list(n) + [n[31]], (-base) % (1 << 33))
    pos = NOT(r[32])
    hi = NOT(and_reduce(C, [NOT(b) for b in r[w:32]]))
    every, low = C.AND(pos, hi), C.AND(pos, NOT(hi))
    return [every ^ C.AND(low, t) for t in therm(C, r[:w], W)]


def expand(C, valid: list, off: list, W: int) -> list:
    """The membership bits of W positions from the sorted slots (``valid`` first, ``off`` strictly increasing over them): slot q
    carries d = off - q (zero when not valid) and moves right by 2^k when bit k of d is set, k from the top down.  The moves
    keep every slot's position below the next's (d does not decrease), so no two meet: a position's valid bit is the one staying
    (`AND(~b, V)`, fresh) XOR the one arriving (its bit b), and each payload bit the stayer's (its form and one AND, or past
    `FORM_BOUND` fresh) XOR the arrival's (`AND(b, bit)`, the bit the mover leaves behind)."""
    w = (W - 1).bit_length()
    m = len(valid)
    V = [valid[q] if q < m else ZERO for q in range(W)]
    A = [[C.AND(valid[q], d) for d in _plus(C, off[q], -q % (1 << w))] if q < m else [ZERO] * w for q in range(W)]
    for k in reversed(range(w)):
        step = 1 << k
        b = [a[k] for a in A]
        mv = [[C.AND(b[q], A[q][i]) for i in range(k)] for q in range(W)]
        nV, nA = [], []
        for q in range(W):
            src = q - step
            nV.append(C.AND(NOT(b[q]), V[q]) ^ (b[src] if src >= 0 else ZERO))
            stay = [C.AND(NOT(b[q]), a) if len(a.s) > FORM_BOUND else a ^ t for a, t in zip(A[q][:k], mv[q])]
            nA.append([s ^ mv[src][i] if src >= 0 else s for i, s in enumerate(stay)])
        V, A = nV, nA
    return V


def check_chunk_statics(TOPK: int, N: int, C: int, H: int | None = None) -> None:
    D.check_selection_statics(TOPK, N)
    if not (C > 0 and C & (C - 1) == 0):
        raise ValueError(f"DsaTopKSelection_v4: a chunk spans a power of two positions (C={C})")
    if H is not None and not 0 <= H < -(-N // C):
        raise ValueError(f"DsaTopKSelection_v4: the row of {N} positions has chunks 0..{-(-N // C) - 1} of {C} (H={H})")


def chunk_width(S) -> int:
    return min(S.C, S.N - S.H * S.C)


def topk_chunk(C, S, *els):
    """`DsaTopKChunk_v1{TOPK,N,C,H}`'s bit (`DsaTopKChunk`)."""
    check_chunk_statics(S.TOPK, S.N, S.C, S.H)
    T, W, base = S.TOPK, chunk_width(S), S.H * S.C
    w = (W - 1).bit_length()
    x, n, idx, thr = els[:W], els[W], els[W + 1:W + 1 + T], els[W + 1 + T]
    keys = []
    for v in idx:
        hit = [v[b] if (base >> b) & 1 else NOT(v[b]) for b in range(w, 32)]
        if W < 1 << w:
            hit.append(gt_u(C, const_bits(W, w), list(v[:w])))
        keys.append(list(v[:w]) + [NOT(and_reduce(C, hit))])
    srt = sort_keys(C, keys)
    valid, off = [NOT(k[w]) for k in srt], [k[:w] for k in srt]
    strict = and_reduce(C, [NOT(C.AND(valid[i + 1], _eq(C, off[i], off[i + 1]))) for i in range(min(T - 1, W))])
    member = expand(C, valid[:min(T, W)], off, W)
    t = [NOT(b) for b in _key(thr)]
    fits = [NOT(m ^ C.AND(lt, _carry(C, _key(x[j]), t, m))) for j, (m, lt) in enumerate(zip(member, below(C, n, base, W)))]
    return [C.AND(strict, and_reduce(C, fits))]


def topk_final(C, S, *els):
    """`DsaTopKFinal_v1{TOPK,N,C}`'s bit (`DsaTopKFinal`)."""
    check_chunk_statics(S.TOPK, S.N, S.C)
    T = S.TOPK
    ok, n, idx = els[0], els[1], els[2:]
    lines = below(C, n, 0, T + 1)
    short = and_reduce(C, [b if (i >> k) & 1 else b ^ lines[i] for i in range(T) for k, b in enumerate(idx[i])])
    inside = [gt_u(C, list(n), list(v)) for v in idx]
    long_ok = and_reduce(C, inside + [NOT(gt_u(C, list(n), const_bits(S.N, 32)))] + list(ok))
    return [long_ok ^ C.AND(NOT(lines[T]), short ^ long_ok)]


def chunk_ok(words, n: int, idx, thr, TOPK: int, N: int, C: int, H: int) -> bool:
    """`DsaTopKChunk_v1{TOPK,N,C,H}` in numpy: ``words`` the chunk's W f32 words, ``thr`` an f32 word (float32, or bits)."""
    check_chunk_statics(TOPK, N, C, H)
    base, W = H * C, min(C, N - H * C)
    u = np.asarray(idx).astype(np.int64).reshape(-1) & D.M32
    offs = u[(u >= base) & (u < base + W)] - base
    if np.unique(offs).size != offs.size:
        return False
    member = np.zeros(W, dtype=bool)
    member[offs] = True
    live = base + np.arange(W) < int(D._signed(n))
    key, t = D.order_keys(np.asarray(words).reshape(-1)), int(D.order_keys(np.asarray([thr]))[0])
    return bool(not (member & ~live).any() and (key[member] >= t).all() and (key[~member & live] <= t).all())


def final_ok(ok, n: int, idx, TOPK: int, N: int) -> bool:
    """`DsaTopKFinal_v1{TOPK,N,C}` in numpy, ``ok`` its chunks' bits."""
    D.check_selection_statics(TOPK, N)
    s, n = D._signed(idx).reshape(-1), int(D._signed(n))
    j = np.arange(TOPK)
    if n <= TOPK:
        return bool(np.array_equal(s, np.where(j < n, j, -1)))
    return bool(((s & D.M32) < n).all() and n <= N and np.asarray(ok).all())


def _chunk_sig(S):
    check_chunk_statics(S.TOPK, S.N, S.C, S.H)
    return (("x", Array(chunk_width(S), F32)), ("len", I32), ("idx", Array(S.TOPK, I32)), ("thr", F32)), BIT


def _row_sig(S):
    check_chunk_statics(S.TOPK, S.N, S.C)
    return (("logits", Array(S.N, F32)), ("len", I32), ("idx", Array(S.TOPK, I32)), ("thr", F32))


DsaTopKChunk = SB.traced_family(
    "DsaTopKChunk", 1, ("TOPK", "N", "C", "H"), _chunk_sig, topk_chunk,
    "Chunk H of `DsaTopKSelection_v3{TOPK,N}`'s key check: positions H*C + j, j < W = min(C, N - H*C), with x = logits[H*C : "
    "H*C + W].  An index is in the chunk when it lies in [H*C, H*C + W) (its bits from w = bits(W - 1) up are H*C's, its low w "
    "bits below W); the TOPK keys (low w bits, not-in-chunk on top) are sorted (`sort_keys`), and the bit is 1 when no two "
    "sorted in-chunk indices are equal and every position j fits the threshold: with its membership bit m (`expand`) and t = "
    "`order_key(thr)`, the carry out of key(x[j]) + ~t + m (key >= t for a member, key > t for a non-member) is m when H*C + j < "
    "len (`below`), and m is 0 past len.")
DsaTopKFinal = SB.traced_family(
    "DsaTopKFinal", 1, ("TOPK", "N", "C"),
    lambda S: ((("ok", Array(-(-S.N // S.C), BIT)), ("len", I32), ("idx", Array(S.TOPK, I32))), BIT), topk_final,
    "`DsaTopKSelection_v3{TOPK,N}`'s bit from its chunks' bits ``ok`` (`DsaTopKChunk_v1{TOPK,N,C,H}` for each H): for len <= "
    "TOPK the shortcut's words, else every index below len (unsigned, as len > TOPK > 0 there), len <= N and every chunk's bit.")


def topk_selection_calls(call, logits: Coll, n: Coll, idx: Coll, thr: Coll, TOPK: int, N: int, C: int = CHUNK) -> Coll:
    """``DsaTopKSelection_v4{TOPK,N,C}``'s checks as Calls: chunk H's `DsaTopKChunk_v1{TOPK,N,C,H}` on its logits and the row's
    len, idx and thr, then `DsaTopKFinal_v1` on their bits.  ``call(fn, *args)`` emits one (``Builder.call``).  At a Program's
    root each is a unit of its own under `Q_word` and `Q_call`, each recomputing what it reads of idx, len and thr; one Call of
    v4 is one unit (its single output bit reads every gate), past `K_MAX` at the served statics.  Returns the bit."""
    K = -(-N // C)
    ok = [call(bind(DsaTopKChunk, TOPK=TOPK, N=N, C=C, H=h), logits[h * C:min(N, (h + 1) * C)], n, idx, thr) for h in range(K)]
    return call(bind(DsaTopKFinal, TOPK=TOPK, N=N, C=C), Coll(Array(K, BIT), concat([o.refs for o in ok])), n, idx)


def topk_selection(C, S, *els):
    """`DsaTopKSelection_v4{TOPK,N,C}`'s bit: each chunk's `topk_chunk` and then `topk_final` on one builder, so what two of them
    compute alike (idx's NOTs, the in-chunk tests' common ANDs, key(thr), the first sort layer's low carries) is one gate."""
    check_chunk_statics(S.TOPK, S.N, S.C)
    T, N, W = S.TOPK, S.N, S.C
    x, n, idx, thr = els[:N], els[N], els[N + 1:N + 1 + T], els[N + 1 + T]
    ok = [topk_chunk(C, bind(DsaTopKChunk, TOPK=T, N=N, C=W, H=h).S(), *x[h * W:(h + 1) * W], n, *idx, thr)[0]
          for h in range(-(-N // W))]
    return topk_final(C, S, ok, n, *idx)


def _with_word(d: CompositeDefinition, word: Callable, word_statics: dict) -> CompositeDefinition:
    """``d`` with its word Definition and the statics that word has no counterpart for (`program.boolean`), as
    `EW.specialized`'s ``word``."""
    d.word = word  # type: ignore[attr-defined]
    d.word_statics = word_statics  # type: ignore[attr-defined]
    return d


DsaTopKSelectionV4 = _with_word(SB.traced_family(
    "DsaTopKSelection", 4, ("TOPK", "N", "C"), lambda S: (_row_sig(S), BIT), topk_selection,
    "`DsaTopKSelection_v3{TOPK,N}` on bits in chunks of C positions: `DsaTopKChunk_v1{TOPK,N,C,H}`'s check for each chunk H, "
    "then `DsaTopKFinal_v1{TOPK,N,C}`'s, traced as one circuit (`topk_selection`).  A Program at the served statics calls those "
    "Definitions at its root instead (`topk_selection_calls`), a unit each."),
    lambda b: bind(D.DsaTopKSelectionV3, TOPK=b["TOPK"], N=b["N"]), {"C": CHUNK})

BOOLEAN_DSA = [U32AndV2, I32SubV2, I32ToF32V2, F32CeilFtzV2, DsaUe8m0ScaleV2, DsaUe8m0QuantV2, DsaIndexerKRowV2, DsaIndexerQHeadV2,
               DsaIndexerQV2, DsaMqaLogitV2, DsaMqaLogitsV2, DsaMqaLogitV4, DsaMqaLogitsV4, DsaTopKShortRowV2, DsaTopKSelectionV2,
               DsaTopKChunk, DsaTopKFinal, DsaTopKSelectionV4]
