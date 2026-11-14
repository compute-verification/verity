"""`dsa_ds_mla`'s cache write on bits, each body `dsa_ds_mla`'s over `boolean_dsa.bit_ops` (core's Boolean FP32 pieces: `F32Fabs`,
`F32Fmaxf`, `F32Mul`, `DivFullScaleA`, `DivFullRcp`, `F32ToE4m3Sat`, `Bf16ToF32`), so each one's word view is its `_v1` at the same
statics.  `DsMlaFp8Tile_v2{T}` is one traced circuit (`boolean_dsa.flat`: the reciprocal of the scale computed once, as every
value its calls share); `DsaLatentRowDsMla_v2{KVL, R, TILE, EPS}` is a composite of `boolean_dsa_norm.DsaRMSNormSumSq_v2`, the
products and `F2fpBf16_v2` over bits, `boolean_rope.RoPEHeadGptJ_v3` and `DsMlaFp8Tile_v2`; `DsaLatentRowDsMlaTile_v2{TILE}`, the row's step for one tile given its scale,
is a composite of the products, `F2fpBf16_v2` and `DsMlaFp8Tile_v2`."""
from __future__ import annotations

from verity.circuits.defs import bind
from verity.circuits.types import Array, Tuple
from verity.circuits.ops.boolean import elementwise as EW
from verity.circuits.ops.boolean import gemm as BG
from verity_vllm.definitions.registry import boolean_rope as BR
from verity_vllm.definitions.registry import dsa_ds_mla as M
from verity_vllm.definitions.registry.boolean_dsa import E4M3, F32, bit_ops, flat
from verity_vllm.definitions.registry.boolean_dsa_norm import BF16, DsaRMSNormSumSqV2


def _tile(B, S, x):
    """`DsMlaFp8Tile_v1`'s body on bits."""
    return M.tile_quant(bit_ops(B), [x[i] for i in range(S.T)], E4M3)


DsMlaFp8TileV2 = flat("DsMlaFp8Tile", 2, ["T"], lambda S: ((("x", Array(S.T, F32)),), Tuple(Array(S.T, E4M3), F32)), _tile,
                      M.DsMlaFp8Tile)


def _row(B, S, kv, w, cs):
    """`DsaLatentRowDsMla_v1`'s body on bits."""
    return M.latent_row(bit_ops(B), S, kv, w, cs, BG.F2fpBf16, bind(DsaRMSNormSumSqV2, N=S.KVL, EPS=S.EPS),
                        bind(BR.RoPEHeadGptJ, D=S.R), bind(DsMlaFp8TileV2, T=S.TILE), BF16, F32, E4M3)


DsaLatentRowDsMlaV2 = EW.on_bits(
    "DsaLatentRowDsMla", 2, ["KVL", "R", "TILE", "EPS"],
    lambda S: ((("kv", Array(S.KVL + S.R, BF16)), ("w", Array(S.KVL, BF16)), ("cs", Array(S.R, BF16))),
               Tuple(Array(S.KVL + S.R, BF16), Array(S.KVL, E4M3), Array(S.KVL // S.TILE, F32))),
    _row, M.DsaLatentRowDsMla)



def _tile_part(B, S, x, w, r):
    """`DsaLatentRowDsMlaTile_v1`'s body on bits."""
    return M.tile_part(bit_ops(B), S, x, w, r, BG.F2fpBf16, bind(DsMlaFp8TileV2, T=S.TILE), BF16, F32, E4M3)


DsaLatentRowDsMlaTileV2 = EW.on_bits(
    "DsaLatentRowDsMlaTile", 2, ["TILE"],
    lambda S: ((("x", Array(S.TILE, BF16)), ("w", Array(S.TILE, BF16)), ("r", F32)), Tuple(Array(S.TILE, BF16), Array(S.TILE, E4M3), F32)),
    _tile_part, M.DsaLatentRowDsMlaTile)

BOOLEAN_DSA_DS_MLA = [DsMlaFp8TileV2, DsaLatentRowDsMlaV2, DsaLatentRowDsMlaTileV2]
