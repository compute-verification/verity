"""`fp8_moe`'s block-scaled FP8 expert GEMMs on bits: version 2 of each, its word body line for line over Boolean Definitions, so each
one's word view is its `_v1` composite at the same statics (`elementwise.on_bits`).

    MoeExpertFp8BlockCoordinate_v2{K,G,KB,SWAP}, MoeExpertFp8BlockCoordinateW_v2{K,G,KB,SWAP}
    MoeExpertFp8BlockGemm_v2{K,N,G,KB,SWAP}, MoeExpertFp8BlockGemmW_v2{K,N,G,KB,SWAP}

The callees are `verity.ml.boolean.gemm`'s `HopperE4m3QgmmaDot32_v2` and `F2fpBf16_v2` and `verity.ml.boolean.scalar`'s `F32Mul` and
`F32Fma`; the +0 words are constant bits.  No table or lookup gate.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import bind
from verity.ir.types import Array
from verity.ml.boolean import elementwise as EW
from verity.ml.boolean import gemm as BG
from verity.ml.boolean import scalar as SC
from verity_vllm.program.registry import fp8_moe as M
from verity_vllm.program.registry.boolean_attention import _Bits
from verity_vllm.program.registry.fp8 import weight_blocks

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
