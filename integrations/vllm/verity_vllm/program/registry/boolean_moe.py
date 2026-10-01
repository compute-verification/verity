"""`registry.moe`'s element-wise Definitions as Boolean Definitions: the next version of each word Definition, the same function on
bits (I32 = F32 = Array<32, Value<1>>, BF16 = Array<16, Value<1>>).

    MoeSumCoordinate_v2{TOPK}             (col: Array<TOPK, BF16>) -> BF16                    word view: MoeSumCoordinate_v1{TOPK}
    MoeSum_v2{TOPK,H}                     (y: Array<TOPK, Array<H, BF16>>) -> Array<H, BF16>   word view: MoeSum_v1{TOPK,H}
    NvExpf_v2                             (x: F32) -> F32                                      word view: NvExpf_v1
    MoeRouterProbs_v2{E,VPT}              (logits: Array<E, BF16>) -> Array<E, F32>            word view: MoeRouterProbs_v1{E,VPT}
    MoeRouterTopKOrdered_v2{E,TOPK,VPT}   (logits: Array<E, BF16>) -> (Array<TOPK, F32>, Array<TOPK, I32>)
                                                                                               word view: MoeRouterTopKOrdered_v1{E,TOPK,VPT}

`MoeSumCoordinate_v2{TOPK}` is traced for each TOPK (`elementwise.specialized`) from ``bf16_rn(((0 + f32(col[0])) + f32(col[1])) +
...)`` through `verity.ml.boolean.fp`, the first add written as what it does (``+0 + x`` is x but for -0, which becomes +0). Its
adds make the canonical NaN where `F32Add_v1` keeps an operand's payload; both are NaN on the same inputs, and the cast ends every
NaN in F2FP's 0x7FFF, so the word is `MoeSumCoordinate_v1`'s. `MoeSum_v2` is `MoeSum_v1`'s body on bits: one coordinate per column of the
TOPK expert outputs.

The router reads each value in one traced circuit, so that no value is computed twice (`Q_word`'s no-recompute rule): the
word body's rounds each re-read the probabilities, which on bits would decode each probability once per round. Its Calls are
cut only at the MUFU, a sub-Call to `verity.ml.boolean.mufu.MufuEx2Ftz_v2`, which `NvExpf_v1`'s ``ex2.approx.ftz.f32`` is:

    NvExpf_v2                  `NvExpfIn_v1` (x -> the ex2 operand f and the low byte L of the fma.rm magic number jb), then
                               `MufuEx2Ftz_v2` (f), then `NvExpfOut_v1` (``e * 2^(L - 127)``, F32Mul_v1's NaN words)
    MoeRouterProbs_v2          `MoeRouterShift_v1` (``x_i - max``: the f32 widening, the kernel's fmaxf reduction geometry and
                               F32Sub_v1), a batch of E `NvExpf_v2`, then `MoeRouterNormalize_v1` (the sums in the same
                               geometry, ``rcp.rn``, the products, NaN/Inf -> 0)
    MoeRouterTopKOrdered_v2    `MoeRouterProbs_v2`, then `MoeRouterSelect_v1` (the TOPK rounds of `MoeRouterFirst_v1`,
                               `MoeRouterNext_v1` and `MoeRouterPick_v1`, on every input word, NaN, -0 and -inf included)

A NaN's payload inside the probabilities never reaches them (an exponential or a reciprocal that is NaN makes every product NaN,
and NaN or Inf is 0), so `MoeRouterShift_v1` and `MoeRouterNormalize_v1` make canonical NaNs where the word body keeps the GPU's
0x7FFFFFFF; `NvExpf_v2` is `NvExpf_v1` on every word. `MoeRouterSelect_v1` compares f32 values by their place in f32's order (an
unsigned 32-bit key, +0 and -0 at one place): `F32GtStrict_v1`, `F32Eq_v1` and the kernel's tie on the lower expert id are one
carry chain each. No sub-Call is a word Definition.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import bind
from verity.ir.types import Array, Tuple
from verity.ml.boolean import elementwise as EW
from verity.ml.boolean import fp
from verity.ml.boolean.forms import NOT, OR, ONE, ZERO, add, and_reduce, const_bits, mux, or_reduce, shr, sub
from verity.ml.boolean.mufu import MufuEx2Ftz
from verity.ml.boolean.trace import definition
from verity_vllm.program.registry import moe as M
from verity_vllm.program.registry import prims as P

__all__ = ["MoeRouterNormalize", "MoeRouterProbs", "MoeRouterSelect", "MoeRouterShift", "MoeRouterTopKOrdered", "MoeSum",
           "MoeSumCoordinate", "NvExpf", "NvExpfIn", "NvExpfOut", "jb_low_byte", "moe_sum", "nv_expf_in",
           "nv_expf_out"]

BF16 = Array(16, BL.BIT)
F32 = I32 = Array(32, BL.BIT)


def _plus_zero(C, w):
    """``+0 + w`` for a word w of any float width, but a NaN kept as it is: w, with -0 made +0."""
    return list(w[:-1]) + [C.AND(w[-1], or_reduce(C, w[:-1]))]


def moe_sum(C, *col):
    """`MoeSumCoordinate_v1`: ``bf16_rn(sum_k f32(col[k]))``, ascending k from +0, every NaN 0x7FFF."""
    acc = fp.bf16_to_f32(_plus_zero(C, col[0]))
    for c in col[1:]:
        acc = fp.f32_add(C, acc, fp.bf16_to_f32(c))
    return fp.f32_to_bf16(C, acc, P.BF16_NAN_F2FP)


MoeSumCoordinate = EW.specialized("MoeSumCoordinate", 2, ["TOPK"], lambda S: moe_sum,
                                  lambda S: ((("col", Array(S.TOPK, BF16)),), BF16),
                                  doc="moe_sum_vec_kernel's coordinate on bits: MoeSumCoordinate_v1, bf16_rn(sum_k f32(col[k])), "
                                      "NaN 0x7FFF", word=lambda b: bind(M.MoeSumCoordinate, TOPK=b["TOPK"]))


def _moe_sum(B, S, y):
    """`out[h] = MoeSumCoordinate_v2{TOPK}(y[:, h])` on bits."""
    return B.batch(bind(MoeSumCoordinate, TOPK=S.TOPK), y, axes=(1,))


MoeSum = EW.on_bits("MoeSum", 2, ("TOPK", "H"), lambda S: ((("y", Array(S.TOPK, Array(S.H, BF16))),), Array(S.H, BF16)),
                    _moe_sum, M.MoeSum)


# --- the router's exponential --------------------------------------------------------------------------------------------------

def _f32(x: float) -> list:
    return const_bits(P.bits_of(x), 32)


def _nan(C, w):
    return C.AND(and_reduce(C, w[23:31]), or_reduce(C, w[:23]))


def _sat(C, t):
    """`F32Sat_v1` (``cvt.sat.f32``): t clamped to [+0, 1.0], NaN and -0 to +0."""
    zero = OR(C, t[31], _nan(C, t))
    _, below = sub(C, t[:31], const_bits(0x3F800000, 31), 31)
    keep, one = C.AND(below, NOT(zero)), C.AND(NOT(below), NOT(zero))
    return [C.AND(keep, b) ^ (one if (0x3F800000 >> i) & 1 else ZERO) for i, b in enumerate(t[:31])] + [ZERO]


def jb_low_byte(C, s):
    """The low byte of ``fma.rm(s, 252, 12582913)`` for s in [+0, 1]: ``1 + floor(s * 252)``. With ``s = sig * 2^(e - 150)``,
    ``floor(s * 252)`` is the top 9 bits of the 32-bit ``sig * 252 = sig * 256 - sig * 4`` shifted right by ``127 - e``, and 0
    below e = 119."""
    e = list(s[23:31])
    sig = list(s[:23]) + [or_reduce(C, e)]
    P, _ = sub(C, [ZERO] * 8 + sig, [ZERO] * 2 + sig, 32)
    _, below = sub(C, e, const_bits(119, 8), 8)
    k = shr(C, P[23:], [NOT(b) for b in e[:4]], zero_if=below)
    return add(C, k[:8], [], cin=ONE, width=8)


def nv_expf_in(C, x):
    """`NvExpf_v1` up to its ``ex2.approx.ftz.f32``: the ex2 operand ``f = x*log2e - j`` (two fma.rn, log2e's hi/lo split) and the
    low byte L of ``jb = fma.rm(sat(x * log2e/252 + 0.5), 252, 12582913)``, the exponent field `F32BitsShl23_v1` makes 2^j of.

    sat is in [+0, 1], so jb is ``12582913 + floor(sat * 252)``, in [2^23, 2^24) with mantissa field ``0x400000 + L``, L in
    [1, 253] (`jb_low_byte`): jb's bit 8 (the scale's sign) is 0 and ``j = jb - 12583039`` is ``L - 127`` exactly, made here
    from L (``-j`` is -0 at j = +0, as `F32Neg_v1` makes it)."""
    t = fp.f32_fma(C, x, const_bits(0x3BBB989D, 32), _f32(0.5))
    L = jb_low_byte(C, _sat(C, t))
    v, _ = sub(C, const_bits(127, 8), L, 8)
    s = v[7]
    neg_j = fp.round_f32(C, s, const_bits(133, 9), 0, add(C, [b ^ s for b in v[:7]], [s], width=7), ONE, underflow=False)
    f = fp.f32_fma(C, x, const_bits(0x3FB8AA3B, 32), neg_j)
    f = fp.f32_fma(C, x, const_bits(0x32A57060, 32), f)
    return [f, L]


def nv_expf_out(C, e, L):
    """``F32Mul_v1(e, F32BitsShl23_v1(jb))`` from jb's low byte L (jb's bit 8 is 0): e scaled by 2^(L - 127), F32Mul_v1's NaN
    words."""
    return fp.f32_mul(C, e, [ZERO] * 23 + list(L) + [ZERO], ir_nan=True)


NvExpfIn = definition("NvExpfIn", 1, nv_expf_in, (("x", F32),), Tuple(F32, Array(8, BL.BIT)),
                      doc="NvExpf_v1 up to its MUFU.EX2: the ex2 operand and the low byte of the fma.rm magic number")
NvExpfOut = definition("NvExpfOut", 1, nv_expf_out, (("e", F32), ("L", Array(8, BL.BIT))), F32,
                       doc="NvExpf_v1 after its MUFU.EX2: e * 2^(L - 127), F32Mul_v1's NaN words")


def _nv_expf(B, S, x):
    """CUDA's `expf(x)` (`NvExpf_v1`) on bits: the range reduction, `MufuEx2Ftz_v2`, the scaling by 2^j."""
    fj = B.call(NvExpfIn, x)
    return B.call(NvExpfOut, B.call(MufuEx2Ftz, fj[0]), fj[1])


NvExpf = EW.on_bits("NvExpf", 2, (), lambda S: ((("x", F32),), F32), _nv_expf, M.NvExpf)


# --- the router's probabilities ------------------------------------------------------------------------------------------------

def _fmaxf(C, a, b):
    """`F32Fmaxf_v1`: the larger by f32's order, +0 over -0; a NaN operand yields the other, two NaNs 0x7FFFFFFF."""
    na, nb = _nan(C, a), _nan(C, b)
    take_a = _less(C, _key(C, b), _key(C, a))
    zeros = C.AND(NOT(or_reduce(C, a[:31])), NOT(or_reduce(C, b[:31])))
    m = mux(C, take_a, a, b)
    m = m[:31] + [m[31] ^ C.AND(zeros, m[31] ^ C.AND(a[31], b[31]))]
    m = mux(C, nb, a, mux(C, na, b, m))
    return mux(C, C.AND(na, nb), const_bits(fp.F32_NAN_PTX, 32), m)


def _butterfly(C, op, vals: list) -> list:
    """`moe._butterfly` on bits: lane 0's value after the xor-T/2 .. 1 stages (the other lanes' gates are dead)."""
    T, mask = len(vals), len(vals) // 2
    while mask >= 1:
        vals = [op(C, vals[t], vals[t ^ mask]) for t in range(T)]
        mask //= 2
    return vals[0]


def _threads(S, vals: list) -> list[list]:
    if S.E % S.VPT:
        raise ValueError("E must be a multiple of VPT")
    return [vals[t * S.VPT:(t + 1) * S.VPT] for t in range(S.E // S.VPT)]


def router_shift(S):
    """``F32Sub_v1(f32(logits_i), max)``, the max `F32Fmaxf_v1` in the kernel's geometry (each thread's VPT values in order,
    then the warp butterfly), every NaN canonical."""
    def build(C, *logits):
        x = [fp.bf16_to_f32(h) for h in logits]
        tmax = []
        for vs in _threads(S, x):
            m = vs[0]
            for v in vs[1:]:
                m = _fmaxf(C, m, v)
            tmax.append(m)
        rmax = _butterfly(C, _fmaxf, tmax)
        return [fp.f32_sub(C, v, rmax) for v in x]
    return build


def router_normalize(S):
    """``isfinite(q_i) ? q_i : 0`` for ``q_i = ex_i * (1 / sum)``, the sum F32Add_v1's in the kernel's geometry from +0 and
    ``1 / sum`` `F32Div_v1`'s, every NaN canonical (the products of a NaN are NaN, and then 0)."""
    def build(C, *ex):
        tsum = []
        for vs in _threads(S, list(ex)):
            s = _plus_zero(C, vs[0])
            for v in vs[1:]:
                s = fp.f32_add(C, s, v)
            tsum.append(s)
        rcp = fp.f32_div(C, _f32(1.0), _butterfly(C, fp.f32_add, tsum))
        out = []
        for v in ex:
            q = fp.f32_mul(C, v, rcp)
            finite = NOT(and_reduce(C, q[23:31]))
            out.append([C.AND(finite, b) for b in q])
        return out
    return build


MoeRouterShift = EW.specialized("MoeRouterShift", 1, ["E", "VPT"], router_shift,
                                lambda S: ((("logits", Array(S.E, BF16)),), Array(S.E, F32)),
                                doc="topk_softmax's x_i - max on bits, canonical NaNs")
MoeRouterNormalize = EW.specialized("MoeRouterNormalize", 1, ["E", "VPT"], router_normalize,
                                    lambda S: ((("ex", Array(S.E, F32)),), Array(S.E, F32)),
                                    doc="topk_softmax's ex_i * (1 / sum), NaN/Inf -> 0, on bits")


def _probs(B, S, logits):
    """`MoeRouterProbs_v1` on bits: `MoeRouterShift_v1`, E `NvExpf_v2`, `MoeRouterNormalize_v1`."""
    d = B.call(bind(MoeRouterShift, E=S.E, VPT=S.VPT), logits)
    return B.call(bind(MoeRouterNormalize, E=S.E, VPT=S.VPT), B.batch(NvExpf, d, axes=(0,)))


MoeRouterProbs = EW.on_bits("MoeRouterProbs", 2, ("E", "VPT"), lambda S: ((("logits", Array(S.E, BF16)),), Array(S.E, F32)),
                            _probs, M.MoeRouterProbs)


# --- the router's selection ----------------------------------------------------------------------------------------------------

def _key(C, w):
    """w's place in f32's order as an unsigned 32-bit number, +0 and -0 at one place (a NaN's is no place)."""
    neg = C.AND(w[31], or_reduce(C, w[:31]))
    return [b ^ neg for b in w[:31]] + [NOT(neg)]


def _key_of(bits: int) -> int:
    """`_key` of a constant word."""
    return ~bits & 0xFFFFFFFF if bits >> 31 and bits & 0x7FFFFFFF else (bits | 0x80000000) & 0xFFFFFFFF


def _less(C, a, b, tie=ZERO):
    """``a < b`` as unsigned numbers, or ``a == b`` and ``tie``: the borrow of ``a - b - tie``."""
    return NOT(add(C, a, [NOT(x) for x in b], cin=NOT(tie), width=len(a), carry_out=True)[-1])


def router_select(S):
    """`MoeRouterTopKOrdered_v1`'s rounds on its probabilities p: ``ids[0] = MoeRouterFirst_v1(p)``, ``ids[k] =
    MoeRouterNext_v1(p, w[k-1], ids[k-1])``, ``w[k] = MoeRouterPick_v1(p, ids[k])``, for every p."""
    E, nb = S.E, max(1, (S.E - 1).bit_length())
    masked = const_bits(_key_of(P.bits_of(-10000.0)), 32)

    def build(C, *p):
        def argmax(keys, ok):
            """The first strict maximum's index (`RouterArgmaxStep_v1`'s scan), entry j taking part when ``ok[j]``."""
            best, idx = keys[0], [ZERO] * nb
            for j in range(1, E):
                gt = C.AND(ok[j], _less(C, best, keys[j]))
                best, idx = mux(C, gt, keys[j], best), mux(C, gt, const_bits(j, nb), idx)
            return idx

        def pick(idx):
            """``p[idx]``, or p[0] past the end (`MoeRouterPick_v1`)."""
            rows = list(p) + [p[0]] * ((1 << nb) - E)
            for b in idx:
                rows = [mux(C, b, rows[i + 1], rows[i]) for i in range(0, len(rows), 2)]
            return rows[0]

        nan = [_nan(C, v) for v in p]
        key = [_key(C, v) for v in p]
        ok = [NOT(n) for n in nan]
        first = [OR(C, nan[0], b) for b in key[0]]
        ids = [argmax([first] + key[1:], ok)]
        sel = [pick(ids[0])]
        for _ in range(1, S.TOPK):
            w, e = sel[-1], ids[-1]
            kw, okw = _key(C, w), NOT(_nan(C, w))
            q = []
            for i in range(E):
                after = C.AND(C.AND(okw, ok[i]), _less(C, key[i], kw, _less(C, e, const_bits(i, nb))))
                q.append(mux(C, after, key[i], masked))
            ids.append(argmax(q, [ONE] * E))
            sel.append(pick(ids[-1]))
        return sel + [i + [ZERO] * (32 - nb) for i in ids]

    return build


MoeRouterSelect = EW.specialized("MoeRouterSelect", 1, ["E", "TOPK"], router_select,
                                 lambda S: ((("p", Array(S.E, F32)),), Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))),
                                 doc="topk_softmax's TOPK selection rounds on bits (MoeRouterFirst_v1, MoeRouterNext_v1, "
                                     "MoeRouterPick_v1)")


def _topk(B, S, logits):
    """`MoeRouterTopKOrdered_v1` on bits: `MoeRouterProbs_v2`, then `MoeRouterSelect_v1`."""
    p = B.call(bind(MoeRouterProbs, E=S.E, VPT=S.VPT), logits)
    return B.call(bind(MoeRouterSelect, E=S.E, TOPK=S.TOPK), p)


MoeRouterTopKOrdered = EW.on_bits("MoeRouterTopKOrdered", 2, ("E", "TOPK", "VPT"),
                                  lambda S: ((("logits", Array(S.E, BF16)),), Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))),
                                  _topk, M.MoeRouterTopKOrdered)
