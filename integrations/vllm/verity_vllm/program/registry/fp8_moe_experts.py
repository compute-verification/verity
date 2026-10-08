"""[Qwen3-235B-A22B fp8] `fp8_moe`'s expert GEMMs on one (token, slot) pair with the expert index as an I32 value operand and the
whole expert stack as the weight operand, the rows gathered inside the body (`moe.MoeExpertGemm`'s construction): what the fused-MoE
derive rule and fold scope bind, since a pair's expert is data (the router's output), not structure.

    GatherE4m3x{E}_v1                       (e: i32, column: Array<E, e4m3>) -> e4m3 = column[e]; e outside [0, E) -> 0x7F (NaN)
    GatherF32x{E}_v1                        (e: i32, column: Array<E, f32>) -> f32 = column[e]; e outside [0, E) -> 0x7FC00000 (NaN): `ref_prims.gather_f32`, main's
    MoeExpertFp8Row{E,K}                    (e, slab: [E][K] e4m3) -> [K] e4m3 = slab[e]
    MoeExpertFp8ScaleRow{E,K}               (e, slab: [E][K] f32) -> [K] f32 = slab[e]
    MoeExpertFp8Coordinate{E,K,G,KB,SWAP,DOT}    (xq, sx, e, slab, sw) -> bf16: `MoeExpertFp8BlockCoordinate_v3` over slab[e]
    MoeExpertFp8CoordinateW{E,K,G,KB,SWAP,DOT}   (xq, sx, e, g, slab, sw) -> bf16: `MoeExpertFp8BlockCoordinateW_v3` over slab[e]
    MoeExpertFp8Gemm{E,K,N,G,KB,SWAP,DOT}        (xq, sx, e, w: [E][N][K] e4m3, ws: [E][cdiv(N, G)][K/G] f32) -> [N] bf16
    MoeExpertFp8GemmW{E,K,N,G,KB,SWAP,DOT}       (xq, sx, e, g, w, ws) -> [N] bf16

`MoeExpertFp8Gemm(W)_v1` at expert e is `MoeExpertFp8BlockGemm(W)_v3` over (w[e], ws[e]): `w` / `ws` are a layer's `w13` / `w13_ws` or
`w2` / `w2_ws` (`w13_weight` / `w13_weight_scale_inv`, the latter on the rank's refined G-block grid), `xq` / `sx` the pair's quantised
activation and its group scales, `g` its f32 routed weight.  Each weight block's scale row is gathered once and read by the block's
coordinates.  The gathers are total: a routed id is always in range, and an id outside it selects NaN words.
"""
from __future__ import annotations

from verity.ml.circuits.defs import PrimitiveDefinition, bind, composite
from verity.ml.circuits.refs import Coll, concat, strided_view
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import fp8_moe as FM
from verity_vllm.program.registry import ref_prims as RP
from verity_vllm.program.registry.fp8 import E4M3, E4M3_NAN, weight_blocks
from verity_vllm.program.registry.prims import BF16, F32, I32

F32_NAN_FILL = 0x7FC00000

_GATHERS: dict[tuple[str, int], PrimitiveDefinition] = {}


def _gather(name: str, vocab: int, word, fill: int) -> PrimitiveDefinition:
    p = _GATHERS.get((name, vocab))
    if p is None:
        def ev(tok: int, *col: int, _V=vocab) -> int:
            t = tok - (1 << 32) if tok & 0x80000000 else tok
            return col[t] if 0 <= t < _V else fill

        def ev_lazy(get, _V=vocab) -> int:
            tok = get(0)
            t = tok - (1 << 32) if tok & 0x80000000 else tok
            return get(1 + t) if 0 <= t < _V else fill

        p = PrimitiveDefinition(f"{name}x{vocab}", 1, (("token", I32), ("column", Array(vocab, word))), word, ev,
                                doc=f"`{name}x{vocab}`: (token: i32, column: Array<{vocab}, word>) -> column[token]; a token outside "
                                    f"[0, {vocab}) -> {fill:#x} (a total extension)", conformance="exact-model-tested")
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        _GATHERS[(name, vocab)] = p
    return p


def gather_e4m3(vocab: int) -> PrimitiveDefinition:
    """`GatherE4m3x{V}_v1`: column[token] of V e4m3 words, else 0x7F."""
    return _gather("GatherE4m3", vocab, E4M3, E4M3_NAN)


def gather_f32(vocab: int) -> PrimitiveDefinition:
    """`GatherF32x{V}_v1`: column[token] of V f32 words, else 0x7FC00000. It is `ref_prims.gather_f32`'s Definition itself,
    since a registry holds one Definition per id."""
    return RP.gather_f32(vocab)


def columns(slab: Coll, E: int, K: int, word) -> Coll:
    """`cols[k][e] = slab[e][k]` of a row-major [E][K] slab of `word`s (one leaf, or a word of bits): no gates."""
    L = word.leaves
    if L == 1:
        return Coll(Array(K, Array(E, word)), strided_view(slab, 0, E, 1, K, K * E))
    return Coll(Array(K, Array(E, word)), concat([strided_view(slab, L * k, L, L * K, 1, L * E) for k in range(K)]))


def row_sig(word, i32=I32):
    return lambda S: ((("e", i32), ("slab", Array(S.E, Array(S.K, word)))), Array(S.K, word))


@composite("MoeExpertFp8Row", 1, ["E", "K"], row_sig(E4M3), conformance="exact-model-tested")
def MoeExpertFp8Row(B, S, e, slab):
    """`wrow[k] = slab[e][k]`: one `GatherE4m3x{E}_v1` per k over the E experts' words (`moe.MoeExpertRow_v1`'s construction)."""
    return B.batch(gather_e4m3(S.E), e, columns(slab, S.E, S.K, E4M3), axes=(None, 0))


@composite("MoeExpertFp8ScaleRow", 1, ["E", "K"], row_sig(F32), conformance="exact-model-tested")
def MoeExpertFp8ScaleRow(B, S, e, slab):
    """`sw[k] = slab[e][k]`: one `GatherF32x{E}_v1` per k, a weight block's K/G scales of expert e."""
    return B.batch(gather_f32(S.E), e, columns(slab, S.E, S.K, F32), axes=(None, 0))


STATICS = ["E", "K", "G", "KB", "SWAP", "DOT"]
GEMM_STATICS = ["E", "K", "N", "G", "KB", "SWAP", "DOT"]


def block(S) -> dict:
    return {"K": S.K, "G": S.G, "KB": S.KB, "SWAP": S.SWAP, "DOT": S.DOT}


def coordinate_sig(routed: bool, e4m3=E4M3, f32=F32, i32=I32, bf16=BF16):
    g = (("g", f32),) if routed else ()
    return lambda S: ((("xq", Array(S.K, e4m3)), ("sx", Array(S.K // S.G, f32)), ("e", i32)) + g
                      + (("slab", Array(S.E, Array(S.K, e4m3))), ("sw", Array(S.K // S.G, f32))), bf16)


def gemm_sig(routed: bool, e4m3=E4M3, f32=F32, i32=I32, bf16=BF16):
    g = (("g", f32),) if routed else ()
    return lambda S: ((("xq", Array(S.K, e4m3)), ("sx", Array(S.K // S.G, f32)), ("e", i32)) + g
                      + (("w", Array(S.E, Array(S.N, Array(S.K, e4m3)))),
                         ("ws", Array(S.E, Array(weight_blocks(S.N, S.G), Array(S.K // S.G, f32))))), Array(S.N, bf16))


@composite("MoeExpertFp8Coordinate", 1, STATICS, coordinate_sig(False), conformance=FM.DOT_CONF)
def MoeExpertFp8Coordinate(B, S, xq, sx, e, slab, sw):
    """One output word of the w13 launch of pair (t, k): `MoeExpertFp8BlockCoordinate_v3`(xq, sx, slab[e], sw), `slab` the column's
    weight rows of every expert and `sw` its weight block's scales of expert e."""
    wrow = B.call(bind(MoeExpertFp8Row, E=S.E, K=S.K), e, slab)
    return B.call(bind(FM.MoeExpertFp8BlockCoordinateV3, **block(S)), xq, sx, wrow, sw)


@composite("MoeExpertFp8CoordinateW", 1, STATICS, coordinate_sig(True), conformance=FM.DOT_CONF)
def MoeExpertFp8CoordinateW(B, S, xq, sx, e, g, slab, sw):
    """One output word of the w2 launch of pair (t, k): `MoeExpertFp8BlockCoordinateW_v3`(xq, sx, g, slab[e], sw)."""
    wrow = B.call(bind(MoeExpertFp8Row, E=S.E, K=S.K), e, slab)
    return B.call(bind(FM.MoeExpertFp8BlockCoordinateWV3, **block(S)), xq, sx, g, wrow, sw)


def pair_rows(B, S, coord, srow, lead: tuple, e: Coll, w: Coll, ws: Coll, word=BF16, f32=F32, e4m3=E4M3) -> Coll:
    """The N coordinates of one pair: per weight block nb, its scale row of expert e gathered once (`srow`), then `coord` over the
    block's columns, column n reading `lead` (e among them) whole, the [E][K] slab `w[:, n, :]` and that scale row."""
    E, K, N, G = S.E, S.K, S.N, S.G
    NB, KG = weight_blocks(N, G), K // G
    Lw, Ls = e4m3.leaves, f32.leaves
    parts = []
    for nb in range(NB):
        sw = B.call(srow, e, Coll(Array(E, Array(KG, f32)), strided_view(ws, Ls * nb * KG, Ls * KG, Ls * NB * KG, 1, Ls * E * KG)))
        n0, n1 = nb * G, min((nb + 1) * G, N)
        slabs = Coll(Array(n1 - n0, Array(E, Array(K, e4m3))),
                     concat([strided_view(w, Lw * n * K, Lw * K, Lw * N * K, 1, Lw * E * K) for n in range(n0, n1)]))
        parts.append(B.batch(coord, *lead, slabs, sw, axes=(None,) * len(lead) + (0, None)))
    return Coll(Array(N, word), concat([p.refs for p in parts]))


@composite("MoeExpertFp8Gemm", 1, GEMM_STATICS, gemm_sig(False), conformance=FM.DOT_CONF)
def MoeExpertFp8Gemm(B, S, xq, sx, e, w, ws):
    """`fused_moe_kernel`'s block-FP8 w13 launch (MUL_ROUTED_WEIGHT False) on one (token, slot) pair routed to expert e:
    `MoeExpertFp8BlockGemm_v3`(xq, sx, w[e], ws[e]), the rows of expert e gathered inside."""
    coord = bind(MoeExpertFp8Coordinate, E=S.E, **block(S))
    return pair_rows(B, S, coord, bind(MoeExpertFp8ScaleRow, E=S.E, K=S.K // S.G), (xq, sx, e), e, w, ws)


@composite("MoeExpertFp8GemmW", 1, GEMM_STATICS, gemm_sig(True), conformance=FM.DOT_CONF)
def MoeExpertFp8GemmW(B, S, xq, sx, e, g, w, ws):
    """The block-FP8 w2 launch (MUL_ROUTED_WEIGHT True) on one pair: `MoeExpertFp8BlockGemmW_v3`(xq, sx, g, w[e], ws[e])."""
    coord = bind(MoeExpertFp8CoordinateW, E=S.E, **block(S))
    return pair_rows(B, S, coord, bind(MoeExpertFp8ScaleRow, E=S.E, K=S.K // S.G), (xq, sx, e, g), e, w, ws)
