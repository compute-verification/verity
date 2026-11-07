"""A MoE expert GEMM over weight rows it reads itself: one (token, slot) pair's block-FP8 GEMM over a block of N rows of the
pair's expert, the rows opened inside the same proof block from a table registered one expert a row, so the rows are never a
proof block's input or output and never committed apart from the table.

    MoeExpertFp8ReadGemm_v1{V, K, N, G, KB, SWAP, DOT}
        (tok, root, domain, row, salt, path, xq, sx, ws) -> Tuple<Array<N, BF16>, Value<1>>                      (y, ok)
    MoeExpertFp8ReadGemmW_v1{V, K, N, G, KB, SWAP, DOT}
        (tok, root, domain, row, salt, path, xq, sx, g, ws) -> Tuple<Array<N, BF16>, Value<1>>                   (y, ok)

        tok, root, domain, row, salt, path   `EmbeddingRead_v1{V, H}`'s, H = N K / 2: row is the block's N rows of K e4m3
                                             bytes as H u16 words (each word two bytes, little-endian)
        xq, sx, [g,] ws                      `MoeExpertFp8BlockGemm(W)_v4{K, N, G, KB, SWAP, DOT}`'s, without w

**What it computes**, by its body:

    (h, ok) = EmbeddingRead_v1{V, H}(tok, root, domain, row, salt, path)
    y       = MoeExpertFp8BlockGemm(W)_v4{K, N, G, KB, SWAP, DOT}(xq, sx, [g,] w = h, ws)

h's H words are the GEMM's N x K e4m3 bytes in the same bit order (a word's low byte first), so ``w`` is h's bits as they are.
When ok = 1 and tok is in [0, V), y is the GEMM over the rows registered at position tok under ``root`` and ``domain``
(`EmbeddingRead_v1`'s guarantee); the unit publishes ``ok`` as the read does. ``ws``, the block's G-block scale row, stays a
port: a scale row feeds G / N blocks, so it is read once, by a unit of its own (`EmbeddingRead_v1{V, 2 K / G}`), and wired in.

The ports and outputs keep the read's names and places (``tok`` .. ``path`` first, ``o0`` the row's image and ``o1`` the
verdict), so a table's root and domain reads, its opening at ``tok`` and its ``o1`` verdict are declared on this unit as on
the read's.

**Cost** is its parts': `EmbeddingRead_v1{V, H}` (row digest nine compressions an 8,192-bit segment and s / 2 + 1 for s
segments, the hm96 leaf four, the read's leaf two, three a tree level) and the GEMM. At GLM-5.3's TP8 statics (V = 256 experts,
G = KB = 128) w13's unit (K = 6144, N = 1, H = 3072) and w2's (K = 256, N = 32, H = 4096) each fit one Flock block.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments.gates import embedding as EMB
from verity.ml.definitions.registry import boolean_fp8_moe as BM
from verity.ml.definitions.registry import fp8_moe as M

__all__ = ["MoeExpertFp8ReadGemm", "MoeExpertFp8ReadGemmW", "words"]

E4M3 = Array(8, BL.BIT)
STATICS = ("V", "K", "N", "G", "KB", "SWAP", "DOT")


def words(K: int, N: int) -> int:
    """H, the u16 words of a block of N rows of K e4m3 bytes."""
    if (N * K) % 2:
        raise ValueError(f"a block of N = {N} rows of K = {K} e4m3 bytes is no whole number of u16 words")
    return N * K // 2


def _gemm(routed: bool, S):
    return bind(BM.MoeExpertFp8BlockGemmWV4 if routed else BM.MoeExpertFp8BlockGemmV4, K=S.K, N=S.N, G=S.G, KB=S.KB,
                SWAP=S.SWAP, DOT=S.DOT)


def _signature(routed: bool):
    def sig(S):
        read = EMB.read(S.V, words(S.K, S.N)).signature()[0]
        gemm = [p for p in M.gemm_sig(routed, E4M3, BM.F32, BM.BF16)(S)[0] if p[0] != "w"]
        return (*read, *gemm), Tuple(Array(S.N, BM.BF16), BL.BIT)
    return sig


def _body(routed: bool):
    def body(B, S, tok, root, domain, row, salt, path, *operands):
        h, ok = B.call(EMB.read(S.V, words(S.K, S.N)), tok, root, domain, row, salt, path)
        *lead, ws = operands
        y = B.call(_gemm(routed, S), *lead, h.reshape(Array(S.N, Array(S.K, E4M3))), ws)
        return tuple_of(y, ok)
    return body


_CONF = ("exact-model-tested as its two parts, whose composition it is (tests/program/test_fp8_moe_per_expert.py): "
         "EmbeddingRead_v1 (Embedding_v1's row on a registered table) and MoeExpertFp8BlockGemm(W)_v4 (registry.fp8_moe's "
         "capture of fused_moe_kernel)")

MoeExpertFp8ReadGemm = CompositeDefinition(
    "MoeExpertFp8ReadGemm", 1, STATICS, _signature(False), _body(False), conformance=_CONF,
    doc="MoeExpertFp8BlockGemm_v4{K, N, G, KB, SWAP, DOT} over the N rows EmbeddingRead_v1{V, N K / 2} opens at tok: (y, ok), "
        "ok the read's verdict")
MoeExpertFp8ReadGemmW = CompositeDefinition(
    "MoeExpertFp8ReadGemmW", 1, STATICS, _signature(True), _body(True), conformance=_CONF,
    doc="MoeExpertFp8BlockGemmW_v4{K, N, G, KB, SWAP, DOT} over the N rows EmbeddingRead_v1{V, N K / 2} opens at tok: (y, ok), "
        "ok the read's verdict")
