"""`dsa_ds_mla`'s cache write on bits, each body `dsa_ds_mla`'s over `boolean_dsa.bit_ops` (core's Boolean FP32 pieces: `F32Fabs`,
`F32Fmaxf`, `F32Mul`, `DivFullScaleA`, `DivFullRcp`, `F32ToE4m3Sat`, `Bf16ToF32`), so each one's word view is its `_v1` at the same
statics.  `DsMlaFp8Tile_v2{T}` is one traced circuit (`boolean_dsa.flat`: the reciprocal of the scale computed once, as every
value its calls share); `DsaLatentRowDsMla_v2{KVL, R, TILE, EPS}` is a composite of `boolean_dsa_norm.DsaRMSNormSumSq_v2`, the
products and `F2fpBf16_v2` over bits, `boolean_rope.RoPEHeadGptJ_v3` and `DsMlaFp8Tile_v2`."""
from __future__ import annotations

from verity.circuits.defs import bind
from verity.circuits.types import Array, Tuple
from verity.definitions.ops.boolean import elementwise as EW
from verity.definitions.ops.boolean import gemm as BG
from verity.definitions.registry import boolean_rope as BR
from verity.definitions.registry import dsa_ds_mla as M
from verity.definitions.registry.boolean_dsa import E4M3, F32, bit_ops, flat
from verity.definitions.registry.boolean_dsa_norm import BF16, DsaRMSNormSumSqV2


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

BOOLEAN_DSA_DS_MLA = [DsMlaFp8TileV2, DsaLatentRowDsMlaV2]
