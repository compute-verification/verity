"""`mtp_draft`'s Definitions on bits, each written as Calls, so that a description does not grow with the row:

    MtpEhZero_v2{H}                          (pos: i64, e: H bf16) -> H bf16    `I64Le_v2`, `BitAnd_v2` and a batch of `SelectBf16_v2`
    TritonBlockRMSNormMeanRsqrt_v1{N, EPS}   (sum: f32) -> f32                  rsqrt.approx.ftz(fma(DivFullScaleA(sum, N), DivFullRcp(N),
                                                                                eps)), N and eps folded
    TritonBlockRMSNormSumSq_v2{N, BLOCK, WARPS, EPS}   (x: N bf16) -> f32       the scale: `boolean_dsa_norm`'s squares over the 32 WARPS
                                                                                threads, `F32Add_v3` trees, then the epilogue above
    MtpEhNorm_v2{H, BLOCK, WARPS, EPS}       (pos, e, p, ew, hw) -> 2H bf16      the select, then each half's scale and H / 64 of
                                                                                `boolean_dsa_norm.DsaRMSNormApply_v3{G=64}`, the cut

The scale is `boolean_dsa_norm.DsaRMSNormSumSq_v3`'s form on a masked block: register k of every thread (`dsa_norm.layout` over
BLOCK) is one row of the batch that takes it, `DsaRMSNormSquares` for the first two and `DsaRMSNormSquareFma` for each next one, a
masked register's operand the bf16 word +0 (its f32 word +0, as the masked load leaves), then the warps' butterflies and their
halving tree in `dsa_norm.rms_scale`'s order.  A register must be live in every thread or masked in every thread (N a multiple of
32 WARPS times a thread's run of V = min(BLOCK / (32 WARPS), 8) elements): GLM-5.3's 6144 in a block of 8192 on 4 warps is, eight
runs of 1024 of which the last two are masked.  Each word view is `mtp_draft`'s `_v1` at the same statics.
"""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.defs import bind
from verity.circuits.refs import Coll, concat, strided_view
from verity.circuits.types import Array
from verity.definitions.boolean import boolean_version
from verity.definitions.ops.boolean import elementwise as EW
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import BitAnd, SelectBf16
from verity.definitions.registry import boolean_dsa_norm as BDN
from verity.definitions.registry import dsa as D
from verity.definitions.registry import dsa_norm as DN
from verity.definitions.registry import mtp_draft as MD
from verity.definitions.registry.boolean_dsa import F32, bit_ops
from verity.definitions.registry.torch_words import I64Le

BF16, I64 = Array(16, BL.BIT), Array(64, BL.BIT)


def _zero(B, S, pos, e):
    """`MtpEhZero_v1` on bits: e with every word +0 where the whole int64 pos is 0 (signed `pos <= 0` and `0 <= pos`)."""
    zero = B.call(boolean_version(const(64, 0)))
    first = B.call(boolean_version(BitAnd), B.call(boolean_version(I64Le), pos, zero), B.call(boolean_version(I64Le), zero, pos))
    return B.batch(boolean_version(SelectBf16), first, B.call(boolean_version(const(16, 0))), e, axes=(None, None, 0))


MtpEhZeroV2 = EW.on_bits("MtpEhZero", 2, ["H"], lambda S: ((("pos", I64), ("e", Array(S.H, BF16))), Array(S.H, BF16)), _zero,
                         MD.MtpEhZero)


def _mean_rsqrt(B, S, total):
    """`dsa_norm.rms_scale`'s fused epilogue: ``rsqrt.approx.ftz(fma(DivFullScaleA(sum, N), DivFullRcp(N), eps))``, the words N and
    eps constants of the binding."""
    o = bit_ops(B)
    n = o.word(D.f32_bits(S.N))
    return o.rsqrt(o.fma(o.scale_a(total, n), o.rcp(n), o.word(D.f32_bits(S.EPS))))


TritonBlockRMSNormMeanRsqrt = BDN._piece("TritonBlockRMSNormMeanRsqrt", ("N", "EPS"), lambda S: ((("sum", F32),), F32), _mean_rsqrt)


def live_registers(N: int, BLOCK: int, WARPS: int) -> list[bool]:
    """Whether each register of a thread holds a live element, the same for every thread; else a ValueError."""
    regs = DN.layout(BLOCK, DN.WARP * WARPS)
    live = [regs[-1][k] < N for k in range(len(regs[0]))]
    if any((regs[0][k] < N) != live[k] for k in range(len(live))):
        raise ValueError(f"TritonBlockRMSNormSumSq_v2: N = {N} leaves a register of BLOCK = {BLOCK} on {WARPS} warps live in some "
                         "threads and masked in others; its Calls take a register of every thread as one row")
    return live


def _sum_sq(B, S, x):
    """`TritonBlockRMSNormSumSq_v1` on bits as Calls: register k of the 32 WARPS threads one row of a batch of
    `DsaRMSNormSquares` (k = 0, 1) or `DsaRMSNormSquareFma` (each next k, +0 where masked), the warps' butterflies and their
    halving tree of `F32Add_v3`, then `TritonBlockRMSNormMeanRsqrt{N, EPS}`."""
    DN.check_block(S.N, S.BLOCK, S.WARPS, S.EPS)
    threads = DN.WARP * S.WARPS
    live = live_registers(S.N, S.BLOCK, S.WARPS)
    regs = DN.layout(S.BLOCK, threads)
    step = regs[1][0] - regs[0][0]

    def register(k: int) -> Coll:
        """Register k of each thread: element ``regs[t][k]``, a thread's elements ``step`` apart from the next thread's."""
        return Coll(Array(threads, BF16), strided_view(x, 16 * regs[0][k], 16, 16 * step, 1, 16 * threads))
    s = B.batch(BDN.DsaRMSNormSquares, register(0), register(1), axes=(0, 0))
    plus0 = B.call(boolean_version(const(16, 0)))
    for k in range(2, len(live)):
        s = (B.batch(BDN.DsaRMSNormSquareFma, register(k), s, axes=(0, 0)) if live[k] else
             B.batch(BDN.DsaRMSNormSquareFma, plus0, s, axes=(None, 0)))
    add = bit_ops(B).add
    warps = [D.halving_tree([s[t] for t in range(DN.WARP * g, DN.WARP * (g + 1))], add) for g in range(S.WARPS)]
    return B.call(bind(TritonBlockRMSNormMeanRsqrt, N=S.N, EPS=S.EPS), D.halving_tree(warps, add))


TritonBlockRMSNormSumSqV2 = EW.on_bits("TritonBlockRMSNormSumSq", 2, ["N", "BLOCK", "WARPS", "EPS"],
                                       lambda S: ((("x", Array(S.N, BF16)),), F32), _sum_sq, MD.TritonBlockRMSNormSumSq)


def _sig(S):
    row = Array(S.H, BF16)
    return (("pos", I64), ("e", row), ("p", row), ("ew", row), ("hw", row)), Array(2 * S.H, BF16)


def _eh(B, S, pos, e, p, ew, hw):
    """`MtpEhNorm_v1` on bits as its cut (`mtp_draft.eh_cut`): `MtpEhZero_v2{H}` of e, then for each half its scale
    `TritonBlockRMSNormSumSq_v2` and a batch of `DsaRMSNormApply_v3{G=64}` over its H / 64 column groups and their weights."""
    G = MD.APPLY_G
    if S.H % G:
        raise ValueError(f"MtpEhNorm_v2: H = {S.H} is not whole apply units of {G}")
    groups = Array(S.H // G, Array(G, BF16))
    sumsq = bind(TritonBlockRMSNormSumSqV2, N=S.H, BLOCK=S.BLOCK, WARPS=S.WARPS, EPS=S.EPS)
    apply = bind(BDN.DsaRMSNormApplyV3, G=G)
    halves = []
    for x, w in ((B.call(bind(MtpEhZeroV2, H=S.H), pos, e), ew), (p, hw)):
        y = B.batch(apply, Coll(groups, x.refs), Coll(groups, w.refs), B.call(sumsq, x), axes=(0, 0, None))
        halves.append(y.refs)
    return Coll(Array(2 * S.H, BF16), concat(halves))


MtpEhNormV2 = EW.on_bits("MtpEhNorm", 2, ["H", "BLOCK", "WARPS", "EPS"], _sig, _eh, MD.MtpEhNorm)
