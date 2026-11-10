"""`fp8_moe`'s block-scaled FP8 expert GEMMs on bits: version 2 of each, its word body line for line over Boolean Definitions, so each
one's word view is its `_v1` composite at the same statics (`elementwise.on_bits`).

    MoeExpertFp8BlockCoordinate_v2{K,G,KB,SWAP}, MoeExpertFp8BlockCoordinateW_v2{K,G,KB,SWAP}
    MoeExpertFp8BlockGemm_v2{K,N,G,KB,SWAP}, MoeExpertFp8BlockGemmW_v2{K,N,G,KB,SWAP}

    SiluMulFp8BlockQuantResult_v2, SiluMulFp8BlockQuantGroup_v2{G}, SiluMulFp8BlockQuant_v2{I,G}

    MoeExpertFp8BlockCoordinate(W)_v4{K,G,KB,SWAP,DOT}, MoeExpertFp8BlockGemm(W)_v4{K,N,G,KB,SWAP,DOT}
                    word view `_v3`, `DOT` a Boolean e4m3 k32 step read as its word view (`BlackwellE4m3QmmaDot32_v2`)
    MoeExpertFp8ReadGemm(W)_v1{READ,K,N,G,KB,SWAP,DOT}
                    the `_v4` GEMM over the N rows `READ` opens: (tok, root, domain, row, salt, path, xq, sx, [g,] ws) -> (y, ok)
    GatherE4m3x{E}_v2, GatherF32x{E}_v2, MoeExpertFp8Row_v2{E,K}, MoeExpertFp8ScaleRow_v2{E,K},
    MoeExpertFp8Coordinate(W)_v2{E,K,G,KB,SWAP,DOT}, MoeExpertFp8Gemm(W)_v2{E,K,N,G,KB,SWAP,DOT},
    MoeExpertFp8CoordinateChunk_v2{E,K,G,KB,SWAP,DOT,FIRST,LAST}
                    word view `fp8_moe_experts`' `_v1`; each gather `boolean_gather`'s multiplexer tree over the id's 32 bits
                    (`GatherE4m3x{E}_v2` for E <= 1024; `GatherF32x{E}_v2` is `boolean_gather.gather_f32`'s at every E)

The callees are `verity.definitions.ops.boolean.gemm`'s `HopperE4m3QgmmaDot32_v2` and `F2fpBf16_v2`, `verity.definitions.ops.boolean.scalar`'s FP32 pieces,
`boolean_moe.NvExpf_v2` and `boolean_fp8.Fp8RowScale_v2`; the constant words are constant bits and `-g` is one NOT gate on g's sign bit
(`F32Neg_v1`).  Each callee is its word Definition on every word, NaN payloads included.  No table or lookup gate.
"""
from __future__ import annotations

from typing import Any

from verity.circuits import boolean as BL
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll, concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.ops.boolean import elementwise as EW
from verity.definitions.ops.boolean import gemm as BG
from verity.definitions.ops.boolean import scalar as SC
from verity.circuits.boolean.gather import gather
from verity.circuits.boolean.trace import definition
from verity.definitions.registry import boolean_gather as BGA
from verity.definitions.registry import boolean_moe as BMOE
from verity.definitions.registry import fp8 as F8
from verity.definitions.registry import fp8_moe as M
from verity.definitions.registry import fp8_moe_experts as X
from verity.definitions.registry.boolean_attention import _Bits
from verity.definitions.registry.boolean_fp8 import F448, F_NEG448, Fp8RowScaleV2
from verity.definitions.registry.fp8 import weight_blocks
from verity.definitions.registry.prims import bits_of

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
E4M3 = Array(8, BL.BIT)
I32 = Array(32, BL.BIT)


def _acc(B, S, xq, sx, wrow, sw, dot=BG.HopperE4m3QgmmaDot32):
    c = _Bits(B)
    return M.triton_block_acc(B, S.K, S.G, S.KB, S.SWAP, xq, sx, wrow, sw, lambda: c.word(0, 32), dot, SC.F32Mul, SC.F32Fma)


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


def _coordinate_dot(B, S, xq, sx, wrow, sw):
    """`MoeExpertFp8BlockCoordinate_v3`'s body on bits, `DOT` a Boolean e4m3 k32 step."""
    return B.call(BG.F2fpBf16, _acc(B, S, xq, sx, wrow, sw, S.DOT))


def _coordinate_w_dot(B, S, xq, sx, g, wrow, sw):
    """`MoeExpertFp8BlockCoordinateW_v3`'s body on bits."""
    return B.call(BG.F2fpBf16, B.call(SC.F32Mul, _acc(B, S, xq, sx, wrow, sw, S.DOT), g))


_DOT_BLOCK = ["K", "G", "KB", "SWAP", "DOT"]
MoeExpertFp8BlockCoordinateV4 = EW.on_bits("MoeExpertFp8BlockCoordinate", 4, _DOT_BLOCK, M.coordinate_sig(False, E4M3, F32, BF16),
                                           _coordinate_dot, M.MoeExpertFp8BlockCoordinateV3, lifted=["DOT"])
MoeExpertFp8BlockCoordinateWV4 = EW.on_bits("MoeExpertFp8BlockCoordinateW", 4, _DOT_BLOCK, M.coordinate_sig(True, E4M3, F32, BF16),
                                            _coordinate_w_dot, M.MoeExpertFp8BlockCoordinateWV3, lifted=["DOT"])


def _gemm_dot(B, S, xq, sx, w, ws):
    """`MoeExpertFp8BlockGemm_v3`'s body on bits."""
    coord = bind(MoeExpertFp8BlockCoordinateV4, K=S.K, G=S.G, KB=S.KB, SWAP=S.SWAP, DOT=S.DOT)
    return M.rows(B, coord, S.N, S.G, w, ws, (xq, sx), BF16)


def _gemm_w_dot(B, S, xq, sx, g, w, ws):
    """`MoeExpertFp8BlockGemmW_v3`'s body on bits."""
    coord = bind(MoeExpertFp8BlockCoordinateWV4, K=S.K, G=S.G, KB=S.KB, SWAP=S.SWAP, DOT=S.DOT)
    return M.rows(B, coord, S.N, S.G, w, ws, (xq, sx, g), BF16)


MoeExpertFp8BlockGemmV4 = EW.on_bits("MoeExpertFp8BlockGemm", 4, ["K", "N", "G", "KB", "SWAP", "DOT"], M.gemm_sig(False, E4M3, F32, BF16),
                                     _gemm_dot, M.MoeExpertFp8BlockGemmV3, lifted=["DOT"])
MoeExpertFp8BlockGemmWV4 = EW.on_bits("MoeExpertFp8BlockGemmW", 4, ["K", "N", "G", "KB", "SWAP", "DOT"], M.gemm_sig(True, E4M3, F32, BF16),
                                      _gemm_w_dot, M.MoeExpertFp8BlockGemmWV3, lifted=["DOT"])


# --- a pair's GEMM over the weight rows it reads itself: one proof block opens a block of its expert's rows from a table
# registered one expert a row and multiplies by them, so the rows are no block's input or output.  `READ` is the read,
# `commitments.gates.embedding`'s `EmbeddingRead_v1{V, N K / 2}` where served: a static, as `DOT` is, so this module imports
# no commitment code.

#: a read's ports: its hidden key, the table's root and domain id, and the opening (the row, its salt, its path)
READ_PORTS = ("tok", "root", "domain", "row", "salt", "path")


def _read_ports(S) -> tuple:
    """`READ`'s parameters, refused (ValueError) unless it is a read of the block's N K / 2 u16 words: `READ_PORTS` in, (h, ok)
    out, h the block's N rows of K e4m3 bytes (a word's low byte first) and ok its verdict."""
    if (S.N * S.K) % 2:
        raise ValueError(f"a block of N = {S.N} rows of K = {S.K} e4m3 bytes is no whole number of u16 words")
    params, ret = S.READ.signature()
    want = Tuple(Array(S.N * S.K // 2, BF16), BL.BIT)
    if tuple(n for n, _ in params) != READ_PORTS or ret != want:
        raise ValueError(f"READ = {S.READ.id} is not a read of N K / 2 = {S.N * S.K // 2} u16 words: it takes "
                         f"{[n for n, _ in params]} and returns {ret!r}; bind EmbeddingRead_v1{{V, H = N K / 2}}")
    return tuple(params)


def _read_gemm_sig(routed: bool):
    def sig(S):
        gemm = tuple(p for p in M.gemm_sig(routed, E4M3, F32, BF16)(S)[0] if p[0] != "w")
        return _read_ports(S) + gemm, Tuple(Array(S.N, BF16), BL.BIT)
    return sig


def _read_gemm(routed: bool):
    def body(B, S, tok, root, domain, row, salt, path, *operands):
        h, ok = B.call(S.READ, tok, root, domain, row, salt, path)
        *lead, ws = operands
        gemm = bind(MoeExpertFp8BlockGemmWV4 if routed else MoeExpertFp8BlockGemmV4, K=S.K, N=S.N, G=S.G, KB=S.KB, SWAP=S.SWAP,
                    DOT=S.DOT)
        return tuple_of(B.call(gemm, *lead, h.reshape(Array(S.N, Array(S.K, E4M3))), ws), ok)
    return body


_READ_CONF = ("exact-model-tested as its two parts, whose composition it is (the vLLM tests' program/test_fp8_moe_per_expert.py): "
              "READ (EmbeddingRead_v1: Embedding_v1's row on a registered table) and MoeExpertFp8BlockGemm(W)_v4")
_READ_GEMM = ["READ", "K", "N", "G", "KB", "SWAP", "DOT"]
MoeExpertFp8ReadGemmV1 = CompositeDefinition(
    "MoeExpertFp8ReadGemm", 1, _READ_GEMM, _read_gemm_sig(False), _read_gemm(False), conformance=_READ_CONF,
    doc="MoeExpertFp8BlockGemm_v4{K, N, G, KB, SWAP, DOT} over the N rows READ opens at tok: (y, ok), ok READ's verdict; the "
        "ports READ's (tok, root, domain, row, salt, path), then xq, sx, ws")
MoeExpertFp8ReadGemmWV1 = CompositeDefinition(
    "MoeExpertFp8ReadGemmW", 1, _READ_GEMM, _read_gemm_sig(True), _read_gemm(True), conformance=_READ_CONF,
    doc="MoeExpertFp8BlockGemmW_v4{K, N, G, KB, SWAP, DOT} over the N rows READ opens at tok: (y, ok), ok READ's verdict; the "
        "ports READ's (tok, root, domain, row, salt, path), then xq, sx, g, ws")


# --- the expert-indexed pair forms (`fp8_moe_experts`) on bits: version 2 of each, the gathers the multiplexer tree over the id's bits

_GATHERS: dict[tuple[str, int], Any] = {}


def _gather(name: str, vocab: int, word, fill: int, word_def) -> Any:
    if not 1 <= vocab <= BGA.CHUNK:
        raise ValueError(f"an expert gather on bits selects from 1 to {BGA.CHUNK} experts, not {vocab}")
    d = _GATHERS.get((name, vocab))
    if d is None:
        d = _GATHERS[(name, vocab)] = definition(
            f"{name}x{vocab}", 2, lambda C, idx, *column: gather(C, idx, list(column), fill), (("token", I32), ("column", Array(vocab, word))),
            word, word=word_def, doc=f"{name}x{vocab}_v1 on bits: column[token] for 0 <= token < {vocab} (i32), else {fill:#x}; the "
                                     "multiplexer tree over the token's bits")
    return d


def gather_e4m3(vocab: int) -> Any:
    """`GatherE4m3x{V}_v2`: `GatherE4m3x{V}_v1` on bits."""
    return _gather("GatherE4m3", vocab, E4M3, F8.E4M3_NAN, X.gather_e4m3(vocab))


def gather_f32(vocab: int) -> Any:
    """`GatherF32x{V}_v2`: `GatherF32x{V}_v1` on bits, `boolean_gather`'s one Definition at every V."""
    return BGA.gather_f32(vocab)


def _row(B, S, e, slab):
    """`MoeExpertFp8Row_v1`'s body on bits: one `GatherE4m3x{E}_v2` per k."""
    return B.batch(gather_e4m3(S.E), e, X.columns(slab, S.E, S.K, E4M3), axes=(None, 0))


def _scale_row(B, S, e, slab):
    """`MoeExpertFp8ScaleRow_v1`'s body on bits: one `GatherF32x{E}_v2` per k."""
    return B.batch(gather_f32(S.E), e, X.columns(slab, S.E, S.K, F32), axes=(None, 0))


MoeExpertFp8RowV2 = EW.on_bits("MoeExpertFp8Row", 2, ["E", "K"], X.row_sig(E4M3, I32), _row, X.MoeExpertFp8Row)
MoeExpertFp8ScaleRowV2 = EW.on_bits("MoeExpertFp8ScaleRow", 2, ["E", "K"], X.row_sig(F32, I32), _scale_row, X.MoeExpertFp8ScaleRow)


def _pair_coordinate(B, S, xq, sx, e, slab, sw):
    """`MoeExpertFp8Coordinate_v1`'s body on bits."""
    wrow = B.call(bind(MoeExpertFp8RowV2, E=S.E, K=S.K), e, slab)
    return B.call(bind(MoeExpertFp8BlockCoordinateV4, **X.block(S)), xq, sx, wrow, sw)


def _pair_coordinate_w(B, S, xq, sx, e, g, slab, sw):
    """`MoeExpertFp8CoordinateW_v1`'s body on bits."""
    wrow = B.call(bind(MoeExpertFp8RowV2, E=S.E, K=S.K), e, slab)
    return B.call(bind(MoeExpertFp8BlockCoordinateWV4, **X.block(S)), xq, sx, g, wrow, sw)


MoeExpertFp8CoordinateV2 = EW.on_bits("MoeExpertFp8Coordinate", 2, X.STATICS, X.coordinate_sig(False, E4M3, F32, I32, BF16),
                                      _pair_coordinate, X.MoeExpertFp8Coordinate, lifted=["DOT"])
MoeExpertFp8CoordinateWV2 = EW.on_bits("MoeExpertFp8CoordinateW", 2, X.STATICS, X.coordinate_sig(True, E4M3, F32, I32, BF16),
                                       _pair_coordinate_w, X.MoeExpertFp8CoordinateW, lifted=["DOT"])


def _pair_gemm(B, S, xq, sx, e, w, ws):
    """`MoeExpertFp8Gemm_v1`'s body on bits."""
    coord = bind(MoeExpertFp8CoordinateV2, E=S.E, **X.block(S))
    return X.pair_rows(B, S, coord, bind(MoeExpertFp8ScaleRowV2, E=S.E, K=S.K // S.G), (xq, sx, e), e, w, ws, BF16, F32, E4M3)


def _pair_gemm_w(B, S, xq, sx, e, g, w, ws):
    """`MoeExpertFp8GemmW_v1`'s body on bits."""
    coord = bind(MoeExpertFp8CoordinateWV2, E=S.E, **X.block(S))
    return X.pair_rows(B, S, coord, bind(MoeExpertFp8ScaleRowV2, E=S.E, K=S.K // S.G), (xq, sx, e, g), e, w, ws, BF16, F32, E4M3)


MoeExpertFp8GemmV2 = EW.on_bits("MoeExpertFp8Gemm", 2, X.GEMM_STATICS, X.gemm_sig(False, E4M3, F32, I32, BF16), _pair_gemm,
                                X.MoeExpertFp8Gemm, lifted=["DOT"])
MoeExpertFp8GemmWV2 = EW.on_bits("MoeExpertFp8GemmW", 2, X.GEMM_STATICS, X.gemm_sig(True, E4M3, F32, I32, BF16), _pair_gemm_w,
                                 X.MoeExpertFp8GemmW, lifted=["DOT"])


def _chunk(B, S, xq, sx, e, *rest):
    """`MoeExpertFp8CoordinateChunk_v1`'s body on bits."""
    c = _Bits(B)
    return X.chunk(B, S, xq, sx, e, rest, bind(MoeExpertFp8RowV2, E=S.E, K=S.K), bind(MoeExpertFp8ScaleRowV2, E=S.E, K=S.K // S.G),
                   lambda: c.word(0, 32), SC.F32Mul, SC.F32Fma, BG.F2fpBf16)


MoeExpertFp8CoordinateChunkV2 = EW.on_bits("MoeExpertFp8CoordinateChunk", 2, X.CHUNK_STATICS, X.chunk_sig(E4M3, F32, I32, BF16), _chunk,
                                           X.MoeExpertFp8CoordinateChunk, lifted=["DOT"])


F1 = bits_of(1.0)


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
