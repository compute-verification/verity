"""`mhc`'s kernels on bits: version 2 of each Definition one kernel launch computes, its word body line for line over Boolean
Definitions (the bodies are `mhc`'s, written once over `mhc.Ops`), so each one's word view is its `_v1` at the same statics.

    HopperTf32WgmmaDot8_v2                      traced: `fp.tc_dot_tf32` on each f32 operand's top 19 bits, the candidate's
                                                groups, adder width and floor
    MhcSigmoid_v2, MhcSinkhorn_v2{HC,REPEAT,EPS}, MhcMixes_v2{HC,NS,K,REPEAT,EPS,HC_EPS}
    MhcPreNorm_v2{HC,H,TN,EPS}, MhcPreNormBroadcast_v2{HC,H,TN,EPS}, MhcPost_v2{HC,N}, MhcFusedSplit_v2{HC,N,OUT,TH}
    MhcPrenormTf32Split_v2{K,OUT}, HcHead_v2{HC,H,TH,EPS,HC_EPS}
                                                each one traced circuit (`boolean_dsa.flat`)
    MhcPreBigFuse_v2{HC,H,NS,TN,REPEAT,EPS,HC_EPS}, MhcPreBigFuseBroadcast_v2{...}, MhcPrenormTf32_v2{K,NS,OUT}
                                                Calls of those (`EW.on_bits`), which read disjoint operands

The callees are core's Boolean FP32 pieces (`elementwise`, `cast`, `mufu`, `scalar`), `gemm.F2fpBf16_v2`, `boolean_moe.NvExpf_v2`
and the TF32 step above; a constant word is constant bits and `-x` one NOT gate on x's sign bit (`F32Neg_v1`).  Each callee is its
word Definition on every word, NaN payloads included.  A traced circuit computes no value twice: two pieces that decode one operand
(a Sinkhorn row's sum, the post's `comb`, a TF32 operand of every output) share its gates.

`MhcFusedPostPre_v1`, `MhcPostPreTf32_v1` and `MhcPreBroadcast_v1` have no version here: each is two or three kernel launches,
the Calls above, and its smallest binding on bits (H 16 or 32 at HC 2) is over circuit-check's partition budget.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.silicon import fp as FP
from verity.ml.circuits.boolean.trace import definition
from verity_vllm.program.registry import boolean_dsa as BD
from verity_vllm.program.registry import boolean_moe as BMOE
from verity_vllm.program.registry import mhc as M
from verity_vllm.program.registry.boolean_attention import _Bits

BIT = BL.BIT
BF16 = Array(16, BIT)
F32 = Array(32, BIT)


def bit_ops(B) -> M.Ops:
    c = _Bits(B)
    return M.Ops(B, BITS, lambda v: c.word(v, 32),
                 lambda x: Coll(F32, concat([x.refs.slice(0, 31), B.call(BL.NOT, x[31]).refs])))


HopperTf32WgmmaDot8V2 = definition(
    "HopperTf32WgmmaDot8", 2,
    lambda C, acc, *ab: FP.tc_dot_tf32(C, acc, [x[13:] for x in ab[:8]], [y[13:] for y in ab[8:16]], M.TF32_GROUPS, M.TF32_WIDTH,
                                       M.TF32_FLOOR),
    (("acc", F32), ("a", Array(8, F32)), ("b", Array(8, F32))), F32, word=M.HopperTf32WgmmaDot8,
    doc="the candidate wgmma.m64nNk8 tf32 step on bits: HopperTf32WgmmaDot8_v1")


def _sigmoid(B, S, z):
    """`MhcSigmoid_v1`'s body on bits."""
    return M.sigmoid(bit_ops(B), z)


MhcSigmoidV2 = BD.flat("MhcSigmoid", 2, [], lambda S: ((("z", F32),), F32), _sigmoid, M.MhcSigmoid)


def _sinkhorn(B, S, x):
    """`MhcSinkhorn_v1`'s body on bits."""
    return M._sinkhorn(bit_ops(B), S, x)


MhcSinkhornV2 = BD.flat("MhcSinkhorn", 2, ["HC", "REPEAT", "EPS"], lambda S: M._sig_sinkhorn(S, F32), _sinkhorn, M.MhcSinkhorn)


def _mixes(B, S, yp, rp, scale, base):
    """`MhcMixes_v1`'s body on bits."""
    return M._mixes(bit_ops(B), S, yp, rp, scale, base)


MhcMixesV2 = BD.flat("MhcMixes", 2, ["HC", "NS", "K", "REPEAT", "EPS", "HC_EPS"], lambda S: M._sig_mixes(S, F32), _mixes,
                     M.MhcMixes)


def _norm(B, S, residual, pre, w):
    """`MhcPreNorm_v1`'s body on bits."""
    return M._norm(bit_ops(B), S, residual, pre, w)


def _norm_b(B, S, x, pre, w):
    """`MhcPreNormBroadcast_v1`'s body on bits."""
    return M._norm_b(bit_ops(B), S, x, pre, w)


MhcPreNormV2 = BD.flat("MhcPreNorm", 2, ["HC", "H", "TN", "EPS"], lambda S: M._sig_norm(S, BF16, F32), _norm, M.MhcPreNorm)
MhcPreNormBroadcastV2 = BD.flat("MhcPreNormBroadcast", 2, ["HC", "H", "TN", "EPS"], lambda S: M._sig_norm_b(S, BF16, F32),
                                _norm_b, M.MhcPreNormBroadcast)


def _fuse(B, S, yp, rp, scale, base, residual, w):
    """`MhcPreBigFuse_v1`'s body on bits."""
    return M.big_fuse(bit_ops(B), S, yp, rp, scale, base, residual, w, False)


def _fuse_b(B, S, yp, rp, scale, base, x, w):
    """`MhcPreBigFuseBroadcast_v1`'s body on bits."""
    return M.big_fuse(bit_ops(B), S, yp, rp, scale, base, x, w, True)


MhcPreBigFuseV2 = EW.on_bits("MhcPreBigFuse", 2, M._FUSE, lambda S: M._sig_fuse(S, BF16, F32), _fuse, M.MhcPreBigFuse)
MhcPreBigFuseBroadcastV2 = EW.on_bits("MhcPreBigFuseBroadcast", 2, M._FUSE, lambda S: M._sig_fuse_b(S, BF16, F32), _fuse_b,
                                      M.MhcPreBigFuseBroadcast)


def _post(B, S, x, residual, post, comb):
    """`MhcPost_v1`'s body on bits."""
    return M._post(bit_ops(B), S, x, residual, post, comb)


MhcPostV2 = BD.flat("MhcPost", 2, ["HC", "N"], lambda S: M._sig_post(S, BF16, F32), _post, M.MhcPost)


def _split(B, S, x, residual, post, comb, fn):
    """`MhcFusedSplit_v1`'s body on bits."""
    return M.fused_split(bit_ops(B), S, x, residual, post, comb, fn)


MhcFusedSplitV2 = BD.flat("MhcFusedSplit", 2, ["HC", "N", "OUT", "TH"], lambda S: M._sig_split(S, BF16, F32), _split,
                          M.MhcFusedSplit)


def _tf32_split(B, S, a, b):
    """`MhcPrenormTf32Split_v1`'s body on bits."""
    return M.tf32_split(bit_ops(B), S, a, b)


MhcPrenormTf32SplitV2 = BD.flat("MhcPrenormTf32Split", 2, ["K", "OUT"], lambda S: M._sig_tf32_split(S, BF16, F32), _tf32_split,
                                M.MhcPrenormTf32Split)


def _tf32(B, S, a, b):
    """`MhcPrenormTf32_v1`'s body on bits."""
    return M.prenorm_tf32(bit_ops(B), S, a, b)


MhcPrenormTf32V2 = EW.on_bits("MhcPrenormTf32", 2, ["K", "NS", "OUT"], lambda S: M._sig_tf32(S, BF16, F32), _tf32, M.MhcPrenormTf32)


def _head(B, S, residual, fn, scale, base):
    """`HcHead_v1`'s body on bits."""
    return M.hc_head(bit_ops(B), S, residual, fn, scale, base)


HcHeadV2 = BD.flat("HcHead", 2, ["HC", "H", "TH", "EPS", "HC_EPS"], lambda S: M._sig_head(S, BF16, F32), _head, M.HcHead)

BITS = dict(BD.BITS, expf=BMOE.NvExpf, bf16=BG.F2fpBf16, sigmoid=MhcSigmoidV2, tf32=HopperTf32WgmmaDot8V2, MhcSinkhorn=MhcSinkhornV2,
            MhcMixes=MhcMixesV2, MhcPreNorm=MhcPreNormV2, MhcPreNormBroadcast=MhcPreNormBroadcastV2, MhcPreBigFuse=MhcPreBigFuseV2,
            MhcPreBigFuseBroadcast=MhcPreBigFuseBroadcastV2, MhcPost=MhcPostV2, MhcFusedSplit=MhcFusedSplitV2,
            MhcPrenormTf32Split=MhcPrenormTf32SplitV2, MhcPrenormTf32=MhcPrenormTf32V2)

BOOLEAN_MHC = [HopperTf32WgmmaDot8V2, MhcSigmoidV2, MhcSinkhornV2, MhcMixesV2, MhcPreNormV2, MhcPreNormBroadcastV2, MhcPreBigFuseV2,
               MhcPreBigFuseBroadcastV2, MhcPostV2, MhcFusedSplitV2, MhcPrenormTf32SplitV2, MhcPrenormTf32V2, HcHeadV2]
