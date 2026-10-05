"""`fp8_moe`'s block-scaled FP8 expert GEMMs on bits: version 2 of each, its word body line for line over Boolean Definitions, so each
one's word view is its `_v1` composite at the same statics (`elementwise.on_bits`).

    MoeExpertFp8BlockCoordinate_v2{K,G,KB,SWAP}, MoeExpertFp8BlockCoordinateW_v2{K,G,KB,SWAP}
    MoeExpertFp8BlockGemm_v2{K,N,G,KB,SWAP}, MoeExpertFp8BlockGemmW_v2{K,N,G,KB,SWAP}

    SiluMulFp8BlockQuantResult_v2, SiluMulFp8BlockQuantGroup_v2{G}, SiluMulFp8BlockQuant_v2{I,G}

The callees are `verity.ml.boolean.gemm`'s `HopperE4m3QgmmaDot32_v2` and `F2fpBf16_v2`, `verity.ml.boolean.scalar`'s FP32 pieces,
`boolean_moe.NvExpf_v2` and `boolean_fp8.Fp8RowScale_v2`; the constant words are constant bits and `-g` is one NOT gate on g's sign bit
(`F32Neg_v1`).  Each callee is its word Definition on every word, NaN payloads included.  No table or lookup gate.
"""
from __future__ import annotations

import numpy as np

from verity.primitives.circuits.ir import boolean as BL
from verity.primitives.circuits.ir.defs import bind
from verity.primitives.circuits.ir.refs import Coll, concat, tuple_of
from verity.primitives.circuits.ir.types import Array, Tuple
from verity.ml.boolean import elementwise as EW
from verity.ml.boolean import gemm as BG
from verity.ml.boolean import scalar as SC
from verity_vllm.program.registry import boolean_moe as BMOE
from verity_vllm.program.registry import fp8_moe as M
from verity_vllm.program.registry.boolean_attention import _Bits
from verity_vllm.program.registry.boolean_fp8 import F448, F_NEG448, Fp8RowScaleV2
from verity_vllm.program.registry.fp8 import weight_blocks
from verity_vllm.program.registry.prims import bits_of

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
E4M3 = Array(8, BL.BIT)


def _acc(B, S, xq, sx, wrow, sw):
    c = _Bits(B)
    return M.triton_block_acc(B, S.K, S.G, S.KB, bool(S.SWAP), xq, sx, wrow, sw, lambda: c.word(0, 32), BG.HopperE4m3QgmmaDot32,
                              SC.F32Mul, SC.F32Fma)


def _coordinate(B, S, xq, sx, wrow, sw):
    """`MoeExpertFp8BlockCoordinate_v1`'s body on bits."""
    return B.call(BG.F2fpBf16, _acc(B, S, xq, sx, wrow, sw))


MoeExpertFp8BlockCoordinateV2 = EW.on_bits(
    "MoeExpertFp8BlockCoordinate", 2, ["K", "G", "KB", "SWAP"],
    lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("wrow", Array(S.K, E4M3)), ("sw", Array(S.K // S.G, F32))),
               BF16),
    _coordinate, M.MoeExpertFp8BlockCoordinate)


def _coordinate_w(B, S, xq, sx, g, wrow, sw):
    """`MoeExpertFp8BlockCoordinateW_v1`'s body on bits: the accumulator times the routed weight, then the cast."""
    return B.call(BG.F2fpBf16, B.call(SC.F32Mul, _acc(B, S, xq, sx, wrow, sw), g))


MoeExpertFp8BlockCoordinateWV2 = EW.on_bits(
    "MoeExpertFp8BlockCoordinateW", 2, ["K", "G", "KB", "SWAP"],
    lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("g", F32), ("wrow", Array(S.K, E4M3)),
                ("sw", Array(S.K // S.G, F32))), BF16),
    _coordinate_w, M.MoeExpertFp8BlockCoordinateW)


def _gemm(B, S, xq, sx, w, ws):
    """`MoeExpertFp8BlockGemm_v1`'s body on bits."""
    coord = bind(MoeExpertFp8BlockCoordinateV2, K=S.K, G=S.G, KB=S.KB, SWAP=S.SWAP)
    return M.rows(B, coord, S.N, S.G, w, ws, (xq, sx), BF16)


MoeExpertFp8BlockGemmV2 = EW.on_bits(
    "MoeExpertFp8BlockGemm", 2, ["K", "N", "G", "KB", "SWAP"],
    lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("w", Array(S.N, Array(S.K, E4M3))),
                ("ws", Array(weight_blocks(S.N, S.G), Array(S.K // S.G, F32)))), Array(S.N, BF16)),
    _gemm, M.MoeExpertFp8BlockGemm)


def _gemm_w(B, S, xq, sx, g, w, ws):
    """`MoeExpertFp8BlockGemmW_v1`'s body on bits."""
    coord = bind(MoeExpertFp8BlockCoordinateWV2, K=S.K, G=S.G, KB=S.KB, SWAP=S.SWAP)
    return M.rows(B, coord, S.N, S.G, w, ws, (xq, sx, g), BF16)


MoeExpertFp8BlockGemmWV2 = EW.on_bits(
    "MoeExpertFp8BlockGemmW", 2, ["K", "N", "G", "KB", "SWAP"],
    lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("g", F32), ("w", Array(S.N, Array(S.K, E4M3))),
                ("ws", Array(weight_blocks(S.N, S.G), Array(S.K // S.G, F32)))), Array(S.N, BF16)),
    _gemm_w, M.MoeExpertFp8BlockGemmW)


F1 = bits_of(np.float32(1.0))


def _result(B, S, gate, up):
    """`SiluMulFp8BlockQuantResult_v1`'s body on bits."""
    c = _Bits(B)
    g = B.call(SC.Bf16ToF32, gate)
    neg = Coll(F32, concat([g.refs.slice(0, 31), B.call(BL.NOT, g[31]).refs]))
    one = c.word(F1, 32)
    sig = B.call(SC.F32Div, one, B.call(SC.F32Add, one, B.call(BMOE.NvExpf, neg)))
    return B.call(SC.F32Mul, B.call(SC.F32Mul, g, sig), B.call(SC.Bf16ToF32, up))


SiluMulFp8BlockQuantResultV2 = EW.on_bits("SiluMulFp8BlockQuantResult", 2, [], lambda S: ((("gate", BF16), ("up", BF16)), F32), _result,
                                          M.SiluMulFp8BlockQuantResult)


def _act_group(B, S, gate, up):
    """`SiluMulFp8BlockQuantGroup_v1`'s body on bits."""
    M.check_act_statics(S.G, S.G)
    c = _Bits(B)
    r = [B.call(SiluMulFp8BlockQuantResultV2, gate[e], up[e]) for e in range(S.G)]
    s = B.call(Fp8RowScaleV2, M.max_tree(B, [B.call(SC.F32Fabs, x) for x in r], SC.F32Fmaxf))
    c448, cneg = c.word(F448, 32), c.word(F_NEG448, 32)
    q = [B.call(SC.F32ToE4m3Sat, B.call(SC.F32Fmaxf, cneg, B.call(SC.F32Fminf, B.call(SC.F32Div, x, s), c448))) for x in r]
    return tuple_of(Coll(Array(S.G, E4M3), concat([w.refs for w in q])), s)


SiluMulFp8BlockQuantGroupV2 = EW.on_bits(
    "SiluMulFp8BlockQuantGroup", 2, ["G"], lambda S: ((("gate", Array(S.G, BF16)), ("up", Array(S.G, BF16))), Tuple(Array(S.G, E4M3), F32)),
    _act_group, M.SiluMulFp8BlockQuantGroup)


def _act(B, S, gu):
    """`SiluMulFp8BlockQuant_v1`'s body on bits."""
    return M.act_row(B, bind(SiluMulFp8BlockQuantGroupV2, G=S.G), S.I, S.G, gu, E4M3, F32)


SiluMulFp8BlockQuantV2 = EW.on_bits(
    "SiluMulFp8BlockQuant", 2, ["I", "G"], lambda S: ((("gu", Array(2 * S.I, BF16)),), Tuple(Array(S.I, E4M3), Array(S.I // S.G, F32))),
    _act, M.SiluMulFp8BlockQuant)
