"""`mhc`'s hyper-connections on bits: version 2 of each Definition, its word body line for line over Boolean Definitions (the bodies
are `mhc`'s, written once over `mhc.Ops`), so each one's word view is its `_v1` at the same statics.

    HopperTf32WgmmaDot8_v2                      traced: `fp.tc_dot_tf32` on each f32 operand's top 19 bits, the candidate's
                                                groups, adder width and floor
    MhcSigmoid_v2                               one traced circuit (`boolean_dsa.flat`)
    MhcSinkhorn_v2{HC,REPEAT,EPS}, MhcMixes_v2{HC,NS,K,REPEAT,EPS,HC_EPS}
    MhcPreNorm_v2{HC,H,TN,EPS}, MhcPreNormBroadcast_v2{HC,H,TN,EPS}
    MhcPreBigFuse_v2{HC,H,NS,TN,REPEAT,EPS,HC_EPS}, MhcPreBigFuseBroadcast_v2{...}
    MhcPost_v2{HC,N}, MhcFusedSplit_v2{HC,N,OUT,TH}, MhcFusedPostPre_v2{HC,H,NS,TH,TN,REPEAT,EPS,HC_EPS}
    MhcPrenormTf32Split_v2{K,OUT}, MhcPrenormTf32_v2{K,NS,OUT}, MhcPostPreTf32_v2{...}, MhcPreBroadcast_v2{...}
    HcHead_v2{HC,H,TH,EPS,HC_EPS}

The callees are core's Boolean FP32 pieces (`elementwise`, `cast`, `mufu`, `scalar`), `gemm.F2fpBf16_v2`, `boolean_moe.NvExpf_v2`
and the TF32 step above; a constant word is constant bits and `-x` one NOT gate on x's sign bit (`F32Neg_v1`).  Each callee is its
word Definition on every word, NaN payloads included.  `MhcSigmoid_v2` is traced as one circuit; the rest are Calls of those
pieces (`EW.on_bits`), as their word bodies are.
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


MhcSinkhornV2 = EW.on_bits("MhcSinkhorn", 2, ["HC", "REPEAT", "EPS"], lambda S: M._sig_sinkhorn(S, F32), _sinkhorn, M.MhcSinkhorn)


def _mixes(B, S, yp, rp, scale, base):
    """`MhcMixes_v1`'s body on bits."""
    return M._mixes(bit_ops(B), S, yp, rp, scale, base)


MhcMixesV2 = EW.on_bits("MhcMixes", 2, ["HC", "NS", "K", "REPEAT", "EPS", "HC_EPS"], lambda S: M._sig_mixes(S, F32), _mixes,
                        M.MhcMixes)


def _norm(B, S, residual, pre, w):
    """`MhcPreNorm_v1`'s body on bits."""
    return M._norm(bit_ops(B), S, residual, pre, w)


def _norm_b(B, S, x, pre, w):
    """`MhcPreNormBroadcast_v1`'s body on bits."""
    return M._norm_b(bit_ops(B), S, x, pre, w)


MhcPreNormV2 = EW.on_bits("MhcPreNorm", 2, ["HC", "H", "TN", "EPS"], lambda S: M._sig_norm(S, BF16, F32), _norm, M.MhcPreNorm)
MhcPreNormBroadcastV2 = EW.on_bits("MhcPreNormBroadcast", 2, ["HC", "H", "TN", "EPS"], lambda S: M._sig_norm_b(S, BF16, F32),
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


MhcPostV2 = EW.on_bits("MhcPost", 2, ["HC", "N"], lambda S: M._sig_post(S, BF16, F32), _post, M.MhcPost)


def _split(B, S, x, residual, post, comb, fn):
    """`MhcFusedSplit_v1`'s body on bits."""
    return M.fused_split(bit_ops(B), S, x, residual, post, comb, fn)


MhcFusedSplitV2 = EW.on_bits("MhcFusedSplit", 2, ["HC", "N", "OUT", "TH"], lambda S: M._sig_split(S, BF16, F32), _split,
                             M.MhcFusedSplit)


def _fused(B, S, x, residual, post, comb, fn, scale, base, w):
    """`MhcFusedPostPre_v1`'s body on bits."""
    return M.fused_post_pre(bit_ops(B), S, x, residual, post, comb, fn, scale, base, w)


MhcFusedPostPreV2 = EW.on_bits("MhcFusedPostPre", 2, ["HC", "H", "NS", "TH", "TN", "REPEAT", "EPS", "HC_EPS"],
                               lambda S: M._sig_layer(S, BF16, F32), _fused, M.MhcFusedPostPre)


def _tf32_split(B, S, a, b):
    """`MhcPrenormTf32Split_v1`'s body on bits."""
    return M.tf32_split(bit_ops(B), S, a, b)


MhcPrenormTf32SplitV2 = EW.on_bits("MhcPrenormTf32Split", 2, ["K", "OUT"], lambda S: M._sig_tf32_split(S, BF16, F32), _tf32_split,
                                   M.MhcPrenormTf32Split)


def _tf32(B, S, a, b):
    """`MhcPrenormTf32_v1`'s body on bits."""
    return M.prenorm_tf32(bit_ops(B), S, a, b)


MhcPrenormTf32V2 = EW.on_bits("MhcPrenormTf32", 2, ["K", "NS", "OUT"], lambda S: M._sig_tf32(S, BF16, F32), _tf32, M.MhcPrenormTf32)


def _post_pre(B, S, x, residual, post, comb, fn, scale, base, w):
    """`MhcPostPreTf32_v1`'s body on bits."""
    return M.post_pre_tf32(bit_ops(B), S, x, residual, post, comb, fn, scale, base, w)


MhcPostPreTf32V2 = EW.on_bits("MhcPostPreTf32", 2, M._FUSE, lambda S: M._sig_layer(S, BF16, F32), _post_pre, M.MhcPostPreTf32)


def _broadcast(B, S, x, fn, scale, base, w):
    """`MhcPreBroadcast_v1`'s body on bits."""
    return M.pre_broadcast(bit_ops(B), S, x, fn, scale, base, w)


MhcPreBroadcastV2 = EW.on_bits("MhcPreBroadcast", 2, M._FUSE, lambda S: M._sig_broadcast(S, BF16, F32), _broadcast,
                               M.MhcPreBroadcast)


def _head(B, S, residual, fn, scale, base):
    """`HcHead_v1`'s body on bits."""
    return M.hc_head(bit_ops(B), S, residual, fn, scale, base)


HcHeadV2 = EW.on_bits("HcHead", 2, ["HC", "H", "TH", "EPS", "HC_EPS"], lambda S: M._sig_head(S, BF16, F32), _head, M.HcHead)

BITS = dict(BD.BITS, expf=BMOE.NvExpf, bf16=BG.F2fpBf16, sigmoid=MhcSigmoidV2, tf32=HopperTf32WgmmaDot8V2, MhcSinkhorn=MhcSinkhornV2,
            MhcMixes=MhcMixesV2, MhcPreNorm=MhcPreNormV2, MhcPreNormBroadcast=MhcPreNormBroadcastV2, MhcPreBigFuse=MhcPreBigFuseV2,
            MhcPreBigFuseBroadcast=MhcPreBigFuseBroadcastV2, MhcPost=MhcPostV2, MhcFusedSplit=MhcFusedSplitV2,
            MhcPrenormTf32Split=MhcPrenormTf32SplitV2, MhcPrenormTf32=MhcPrenormTf32V2)

BOOLEAN_MHC = [HopperTf32WgmmaDot8V2, MhcSigmoidV2, MhcSinkhornV2, MhcMixesV2, MhcPreNormV2, MhcPreNormBroadcastV2, MhcPreBigFuseV2,
               MhcPreBigFuseBroadcastV2, MhcPostV2, MhcFusedSplitV2, MhcFusedPostPreV2, MhcPrenormTf32SplitV2, MhcPrenormTf32V2,
               MhcPostPreTf32V2, MhcPreBroadcastV2, HcHeadV2]
