"""[GLM-5.3] Program 1's row of `_fused_norm_rope_kernel` under `--kv-cache-dtype fp8`, the cache layout `fp8_ds_mla` (vLLM d9105ea80
`models/deepseek_v32/common/kernels.py`): kv_a_layernorm over kv_c [512], the interleaved RoPE of k_pe [64], kv_c_out and k_pe_out
stored (:240-250; `attention.py:338-366` always passes both), then the cache row (:263-291), 656 bytes:

    bytes   0 .. 511   e4m3 of y / scale, y the norm's f32 output, one scale per tile of 128     (:277-282)
    bytes 512 .. 527   the four f32 scales                                                       (:284-287)
    bytes 528 .. 655   the 64 roped values as bf16, the same words as k_pe_out                   (:288-290)

The quantisation, `tile_scale = tl.maximum(tile_amax * (1.0 / 448.0), 1.1754944e-38)` and `(kv_2d / tile_scale).to(tl.float8e4nv)`
(:277-281), reads the f32 registers that kv_c_out rounds to bf16 (`_rms_norm`'s `(x * rrms) * w`, :25-31, 219), before any
rounding.  "The PTX" is Triton's sm_90a output for the two specialisations the H200 serves under this arm (`.reqntid 128`, thread t
holding elements 4t .. 4t+3 of kv_c by `ld.global.v2.b32`, so tile j is warp j): the shared layers' (no indexer; lines below) and
the indexer layers' (`HAS_INDEXER`, the same instructions at its lines 593-846).  Program 1's norm and RoPE are
`dsa_norm.DsaRMSNorm_v1`'s and `boolean_rope.RoPEHeadGptJ_v2`'s lines; then, line by line:

    230-247       y_i       mul.f32(mul.f32(r, x_i), cvt.f32.bf16(w_i)), r = rsqrt.approx.ftz(eps + div.full(sum, 512)),
                            four registers %r4 .. %r7 a thread
    329-332       kv_c_out  cvt.rn.bf16x2.f32 of y, st.global.v2.b32                    (:240)
    348, 356      k_pe_out  st.global.b16 of r1, r2 as bf16                             (:244-250)
    390-393       |y_i|     abs.f32                                                     (:277)
    396-398       a_t       max.f32 chain over the thread's four
    401-428       amax      xor-butterfly 16, 8, 4, 2, 1 of max.f32 (`shfl.sync.bfly`), one warp a tile
    431           p         mul.f32(amax, 0f3B124925)                                   f32(1/448), the nearest f32, above 1/448
    433           scale     max.f32(p, 0f00800000)                                      FLT_MIN (:280)
    435-438       y_i / s   div.full.f32(y_i, scale)                                    (:281)
    441, 445      q         cvt.rn.satfinite.e4m3x2.f32 (hi, lo) = (y_{4t+1}/s, y_{4t}/s), each pair's second source the low byte
    455           bytes     st.global.b32 at byte 4t of the row                         (:282)
    468-479       scales    st.shared then ld.shared, threads 0-3 store tile t's scale at byte 512 + 4t   (:284-287)
    493, 499      rope      st.global.b16 of the k_pe_out registers %rs48, %rs49 at 528 + 4j and + 2   (:288-290)

`max.f32` is `F32Fmaxf_v1` (a NaN operand yields the other, so a NaN in a tile is dropped and an all-NaN tile's amax is NaN, whose
scale is the floor); over words whose sign `abs.f32` cleared it is order-free, so the body's tree is the chain-then-butterfly's word.
`div.full.f32` is `F32Mul(DivFullScaleA(y, s), DivFullRcp(s))` (as `DsaRMSNorm_v1`'s mean), the reciprocal once a tile; and
`cvt.rn.satfinite.e4m3x2.f32` is `F32ToE4m3Sat_v1` on each value.  What the corners give, from those words:

- the scale is in [FLT_MIN, +inf], never subnormal: a tile whose every y is a zero or a NaN, or whose amax * f32(1/448) rounds
  below FLT_MIN, takes FLT_MIN;
- f32(1/448) exceeds 1/448, so a finite tile's largest |y| / scale is at most 448 and its byte 0x7E or 0xFE; no finite value
  saturates;
- a NaN y is 0x7F; an infinite y makes the scale +inf, its own byte 0x7F (inf / inf) and every finite one a zero of its sign;
- y = -0 gives 0x80.

Registered: `DsMlaFp8Tile_v1{T}` (x: T f32) -> (q: T e4m3, scale: f32), one tile; and `DsaLatentRowDsMla_v1{KVL, R, TILE, EPS}` (kv:
KVL + R bf16, w: KVL bf16, cs: R bf16) -> (latent: KVL + R bf16, nope: KVL e4m3, scale: KVL / TILE f32), program 1's whole row:
latent is `dsa_norm.DsaLatentRow_v1{KVL, R, EPS}`'s words ([kv_c_out | k_pe_out]), and the cache row is nope, scale, latent[KVL:].
The norm's f32 output is no output: it is stated once, inside the row, and both kv_c_out and the tiles read it.  GLM-5.3's row is
KVL = 512, R = 64, TILE = 128, EPS = 1e-5 (the wrapper asserts kv_dim 512, :418-607).
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.circuits.refs import Coll, concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import dsa as D
from verity.definitions.registry import dsa_norm as DN
from verity.definitions.registry import prims as P
from verity.definitions.registry.boolean_rope import RoPEHeadGptJV2
from verity.definitions.registry.fp8 import E4M3
from verity.definitions.registry.prims import BF16, F32

WARP = 32
#: f32(1 / 448), the nearest f32 (0.00223214295...), and FLT_MIN (1.1754944e-38 as f32), the PTX's two immediates
F_INV448, F_FLT_MIN = 0x3B124925, 0x00800000

_CONF = ("model-untested: read from the Triton kernel and its PTX at the commits the module docstring pins; "
         "adapters/vllm/tests/properties/glm53_ds_mla_write_capture_gpu.py is its hardware check")


def check_tile(T: int) -> None:
    if not (T >= WARP and T & (T - 1) == 0):
        raise ValueError(f"DsMlaFp8Tile: a tile is a whole warp's elements, a power of two from {WARP} (T={T})")


def check_row(KVL: int, R: int, TILE: int, EPS: float) -> None:
    DN.check_statics(KVL, EPS)
    check_tile(TILE)
    if not (KVL % TILE == 0 and R > 0 and R % 2 == 0):
        raise ValueError(f"DsaLatentRowDsMla: whole tiles of the norm's row and an even rotary width (KVL={KVL}, R={R}, TILE={TILE})")


def tile_quant(o: D.Ops, x: list, e4m3=E4M3):
    """`DsMlaFp8Tile`'s body over ``o``'s steps: amax as a warp takes it (each lane's chain over its T / 32 contiguous words, then
    the butterfly), scale = fmaxf(amax * f32(1/448), FLT_MIN), q[i] = e4m3(div.full(x[i], scale))."""
    T = len(x)
    check_tile(T)
    E = T // WARP
    lanes = []
    for t in range(WARP):
        a = o.fabs(x[E * t])
        for v in x[E * t + 1:E * t + E]:
            a = o.fmaxf(a, o.fabs(v))
        lanes.append(a)
    scale = o.fmaxf(o.mul(D.halving_tree(lanes, o.fmaxf), o.word(F_INV448)), o.word(F_FLT_MIN))
    rcp = o.rcp(scale)
    q = [o.e4m3(o.mul(o.scale_a(v, scale), rcp)) for v in x]
    return tuple_of(Coll(Array(T, e4m3), concat([w.refs for w in q])), scale)


@composite("DsMlaFp8Tile", 1, ["T"], lambda S: ((("x", Array(S.T, F32)),), Tuple(Array(S.T, E4M3), F32)), conformance=_CONF)
def DsMlaFp8Tile(B, S, x):
    """One tile of `fp8_ds_mla`'s cache write (kernels.py:277-281; the module docstring has the PTX): amax the `F32Fmaxf` of the
    `F32Fabs` of the T words, scale = `F32Fmaxf(F32Mul(amax, f32(1/448)), FLT_MIN)`, q[i] = `F32ToE4m3Sat(F32Mul(DivFullScaleA(x[i],
    scale), DivFullRcp(scale)))`."""
    return tile_quant(D.word_ops(B), [x[i] for i in range(S.T)])


def latent_row(o: D.Ops, S, kv, w, cs, to_bf16, sumsq, rope, tile, t=BF16, f32=F32, e4m3=E4M3):
    """`DsaLatentRowDsMla`'s body over ``o``'s steps: r a Call of ``sumsq``, y_i = r * x_i * f32(w_i) once, kv_c_out its Calls of
    ``to_bf16``, k_pe_out a Call of ``rope``, and each TILE of y a Call of ``tile``."""
    check_row(S.KVL, S.R, S.TILE, S.EPS)
    r = o.B.call(sumsq, kv[0:S.KVL])
    y = [o.mul(o.mul(r, o.cvt(kv[i])), o.cvt(w[i])) for i in range(S.KVL)]
    kvc = [o.B.call(to_bf16, v) for v in y]
    kpe = o.B.call(rope, kv[S.KVL:S.KVL + S.R], cs)
    tiles = [o.B.call(tile, Coll(Array(S.TILE, f32), concat([v.refs for v in y[j * S.TILE:(j + 1) * S.TILE]])))
             for j in range(S.KVL // S.TILE)]
    return tuple_of(Coll(Array(S.KVL + S.R, t), concat([*(v.refs for v in kvc), kpe.refs])),
                    Coll(Array(S.KVL, e4m3), concat([q[0].refs for q in tiles])),
                    Coll(Array(S.KVL // S.TILE, f32), concat([q[1].refs for q in tiles])))


def _sig_row(S):
    return ((("kv", Array(S.KVL + S.R, BF16)), ("w", Array(S.KVL, BF16)), ("cs", Array(S.R, BF16))),
            Tuple(Array(S.KVL + S.R, BF16), Array(S.KVL, E4M3), Array(S.KVL // S.TILE, F32)))


@composite("DsaLatentRowDsMla", 1, ["KVL", "R", "TILE", "EPS"], _sig_row, conformance=_CONF)
def DsaLatentRowDsMla(B, S, kv, w, cs):
    """One token's program-1 row under `fp8_ds_mla`: r = `DsaRMSNormSumSq_v1{KVL, EPS}` of kv's first KVL words, y_i = `F32Mul(F32Mul(r,
    f32(kv_i)), f32(w_i))`; latent = [`F2fpBf16`(y) | `RoPEHeadGptJ_v2{R}`(kv's last R, cs)], `DsaLatentRow_v1`'s words (kv_c_out
    and k_pe_out); nope and scale `DsMlaFp8Tile_v1{TILE}` of each TILE of y.  The cache row is nope, scale, latent[KVL:]."""
    return latent_row(D.word_ops(B), S, kv, w, cs, P.F2fpBf16, bind(DN.DsaRMSNormSumSq, N=S.KVL, EPS=S.EPS),
                      bind(RoPEHeadGptJV2, D=S.R), bind(DsMlaFp8Tile, T=S.TILE))


DSA_DS_MLA_COMPOSITES = [DsMlaFp8Tile, DsaLatentRowDsMla]
