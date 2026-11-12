"""`verity.definitions.registry.boolean_gather`: the Boolean vocabulary gathers, embedding and bit selection agree with
`GatherBf16x{V}_v1`, `GatherF32x{V}_v1`, `Embedding_v1` and `BitAtx{W}_v1` on edge and random tokens (negative, past the
vocabulary, at every chunk boundary), are pinned by gate count and program digest, and cut under `Q_word_v1{X=16,W=32}` into one
unit of 16 output bits per gather, with no gate two gathers compute. `GatherF32x{V}_v2` is one Definition, the word family's
Boolean version made on demand and `boolean_fp8_moe`'s expert gather alike. `GatherF32Chunked_v1{V, W}` is `GatherF32x{V}_v1`
and `GatherF32x{V}_v2` at every token, V-chunk boundaries and the wrap past i32's extremes included, and `W_MAX` is where
`circuit.compose`'s block of 2^27 bits ends. The gathers over more than 1024 words are trees of Calls whose chunk Definition
takes seconds to compile, so their tests are slow; the quick ones cover the multiplexer tree traced whole."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_flock import class_statement as CS
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.circuits import evaluation as E
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import standalone_call
from verity.circuits import boolean as BL
from verity.circuits import codec
from verity.circuits.defs import REGISTRY, bind
from verity.circuits.boolean.trace import word_of
from verity.definitions import boolean as BB
from verity.definitions.registry import boolean_fp8_moe as BFM
from verity.definitions.registry import boolean_gather as BGA
from verity.definitions.registry import prims as P
from verity.definitions.registry import ref_prims as RP
from verity.definitions import word as W

SERVED_VOCAB = 49152
#: the training step's vocabulary, over which it gathers an f32 word (`GatherF32x50304_v1`)
TRAINING_VOCAB = 50304
#: Llama 3's vocabulary, three V-chunks of at most `BGA.W_MAX` words
LLAMA_VOCAB = 128256

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "GatherBf16x64_v2": (1424, 2432, 390, "2e1cfba6e2a01125cdf2ea16da210c91f6f71d08c3719d370ad85ca447aec674b948e4317e5398c23366f9ead77299041628401a467b83fd720d5864c10d9f16"),
    "GatherBf16x76u10_v1": (1296, 2496, 90, "8e52f98031c197156ad9c2b522590e891442fe20131a156b01239f595ce6e752b84477a49d1a09af9846985dc76b0adbce36ffee7314e6cf5987cb897f40be92"),
    "GatherBf16x2u22_v1": (352, 368, 315, "644d53b8b1d12a48010d7b92b906f1fb59aa4b73d64110d72bde8c2e28c9c03a922c49dcd968e8ef08b1c8606bc6894a8731a970f5d21c36f5adc5a34cd36a0e"),
    "Embedding_v2{V=64,H=16}": (22784, 38912, 6240, "dc6159de5ce51bacbf41b8e34ba63754be24b70569bfa501765e6396cabb66476f1f5936e97870fc115cf1a984ffe51c8cf384c3e64547dc2bfcf19b45117b58"),
    "GatherBf16x1024u10_v1": (16368, 32736, 0, "342ae6660d384c030788a4e8d1acd46784bb40d4bd1c49f027378765bc35c19036ff5ec3d241ca671d94f34375f48902539942fe64097d8ac4d2d1cb44796b6b"),
    "GatherBf16x1100_v2": (18016, 35600, 405, "13b5d96e04c3c84641ee9330dd9e2d1a14224fc79cf0603aa9216b8e135f1db500f807f3f1a5568c2eec14653871caac90ff709e1fd3b400739046ae18e28d77"),
    "Embedding_v2{V=1100,H=2}": (36032, 71200, 810, "01796fa581b87c15391b8bc98d1789432a6a82b1e928435b46853c81bd15346ebc0d78acaa6db9fdc2baadecb90849af27852c9de053dad3899c6c369d8c0387"),
    "GatherF32x64_v2": (2848, 4864, 234, "3fcbb5a9e4b1b6f77b3ba6d123d85521fdc68a1876aa53ce3171a6c2d3b0dc7796c9342d679fbf5994a69031a24a73455efde8634e9edebab5b4ed60313761e1"),
    "GatherF32x128u10_v1": (4160, 8224, 27, "b68e161389f1e1a9badaaf275fc86ceb3d9655d2e26240fbfcf07799792c85c5af8e37078faf796f579e5066c70409012659239d414aff4bf5e6b82bd20c1a4b"),
    "GatherF32x50u22_v1": (2176, 3744, 171, "5f4eac65d7ed4d1b6c9a45187657be4ee9d0eeaab5d3d15767ba1b861e8d1f80d6c575761f5f91178a4aacce2a8f0b9c8390dbba6060858e654b975ddc6d9ccd"),
    "GatherF32x1024u10_v1": (32736, 65472, 0, "ef4a2799247f2f245f1e83882e6f0dd57504ebb379dfb4faea4e340b5adbe2a5c520f83f4f609c07f8b2772567e46bbc56c56d0a761c776fa3a4263b7383dd4c"),
    "GatherF32Chunked_v1{V=100,W=40}": (6107, 9392, 915, "6fe31af10a432b992d216814a72bee4f2e149877d334ce0bd523a711a43aab17f2d906261d309564a517336443a1af1edb9f79ca34419f1b0d9b9c0517c2b864"),
    "GatherChunkOffsets_v1{V=100,W=40}": (59, 176, 87, "e56edfaee0072829047c7dc6d3662c832e50f670f64b993f305b616e5016b3a0c91ea586da4e228ed4b4e11ebe8b7908160c24a88cf972a76ea5d78a1686781c"),
    "GatherChunkOffsets_v1{V=50304,W=47163}": (24, 70, 43, "eba188172a6ea47375f882c7daa9f95b87adefae31d8301336102459e16fd6b95f332498f846731b2375a18151fa66edb217bb2b7db4e4a1a0660ace7ffe34d6"),
    "GatherChunkOffsets_v1{V=128256,W=47163}": (43, 123, 58, "30fb2c313cc16dc67a7adb59cd0d9f44735556a1afa14def8eab66c1159bfbdb6eadb4fa98420993e15a783952026c09647c81068dfb04c7bd06a79aaf35c0f2"),
    "GatherChunkXor_v1{C=2}": (0, 32, 9, "5ee5fbc85131ed0aa8a444144f10d3ef0f94ce71bf24980619d937997f5164a138d4da617e1f14cb17d6ff15a9a3c49b6801e9ded2ff8f909802103358c4364e"),
    "GatherChunkXor_v1{C=3}": (0, 64, 0, "78846407d8cee1b0fadc2644b0187afa3b25476d9a7f9f6d34072865c835800bec57a1f4a3eb5dbf0e0d4f3bb1d4fa8fcf1d0e844c380508a1d0a5d149f35b42"),
    "GatherF32x576u10_v1": (18496, 36896, 27, "41263c18f52bbbc0dc0c90ad8903f8180d543236eedb0714253ddf7b98cb75872b6261b6fce67116f9fbc386a226d22c33ecd6268fea148de8f250593d8be224"),
    "GatherF32x25u22_v1": (1408, 2176, 180, "8058a76e03075fcfff33325f2b5d2869c9eeab3114dff9ba1a15077cba71cc0b0963677b1269053eb5a3b0585b8751fb318032c37e5430198dbfd00c6254ff7b"),
    "GatherF32x768u10_v1": (24576, 49120, 9, "96be69465c6be5897f8834275589041120e5748474f9ffe5c4ea131f356c5f75349b046ac8a402b6e4d81ba6a0ec3c89adf6df05dfcb9404ea1393eea7aaac4f"),
    "GatherF32x42u22_v1": (1920, 3232, 171, "31b0ed477c18305eb76a24e7ca54a5a841d33519e20dfbced8cf0ea2b6db9134406feaff2f036564f20624f8fbf71a1ddf4956251d85db01fa048ba9575df5b6"),
    "BitAtx32_v2": (58, 89, 0, "fb8ec4e743bf5393db3093af9048767760a22c67808dd984dc9439415d269cfa6dc1cdbeaee5dfb4f5343cf60ac71830a41ac44fbf6153febf8d229aa6db5c06"),
}
QUICK = [bind(BGA.gather_bf16(64)), bind(BGA.gather_index(76, 10)), bind(BGA.gather_index(2, 22)), bind(BGA.Embedding, V=64, H=16),
         bind(BGA.gather_f32(64)), bind(BGA.gather_index(128, 10, "F32")), bind(BGA.gather_index(50, 22, "F32")),
         bind(BGA.GatherF32Chunked, V=100, W=40), bind(BGA.GatherChunkOffsets, V=100, W=40),
         bind(BGA.GatherChunkOffsets, V=TRAINING_VOCAB, W=BGA.W_MAX), bind(BGA.GatherChunkOffsets, V=LLAMA_VOCAB, W=BGA.W_MAX),
         bind(BGA.GatherChunkXor, C=2), bind(BGA.GatherChunkXor, C=3),
         bind(BGA.gather_index(576, 10, "F32")), bind(BGA.gather_index(25, 22, "F32")), bind(BGA.gather_index(768, 10, "F32")),
         bind(BGA.gather_index(42, 22, "F32")), bind(BGA.bit_at(32))]
TREES = [bind(BGA.gather_index(1024, 10)), bind(BGA.gather_bf16(1100)), bind(BGA.Embedding, V=1100, H=2),
         bind(BGA.gather_index(1024, 10, "F32"))]


def _pinned(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


@pytest.mark.parametrize("target", QUICK, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    _pinned(target)


@pytest.mark.parametrize("target", TREES, ids=lambda t: t.id)
def test_pinned_counts_and_digests_of_the_trees(target) -> None:
    _pinned(target)


def _tokens(V: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """Every edge of the range check and the tree (0, V - 1, V, the chunk boundaries, i32's extremes, negatives), then ``n``
    tokens in [0, V) and ``n // 4`` over all of u32."""
    edge = [0, 1, V - 1, V, V + 1, 1023, 1024, 1025, 2047, 2048, (1 << 16) - 1, 1 << 16, (1 << 22) - 1, 1 << 22, (1 << 31) - 1,
            1 << 31, (1 << 32) - 1, -V, -V - 1, -1024]
    edge += [m * 1024 + d for m in range(V // 1024 + 2) for d in (-1, 0, 1)]
    edge += [(1 << b) + d for b in range(10, 32) for d in (0, V % 1024)]
    return np.array([e % (1 << 32) for e in edge] + list(rng.integers(0, V, n)) + list(rng.integers(0, 1 << 32, n // 4)),
                    np.uint32)


def _agree(target, *inputs) -> None:
    got = BS.evaluate_words(target, *inputs).astype(np.uint64)
    ref = E.evaluate_batch(word_of(target), *inputs).astype(np.uint64)
    n = len(inputs[0])
    bad = np.flatnonzero(got.reshape(n, -1) != ref.reshape(n, -1))
    assert not bad.size, f"{target.id}: {bad.size} instances differ from {word_of(target).id}, first {bad[:5]}"


def _agree_gather(V: int, seed: int, n: int) -> None:
    g = bind(BGA.gather_bf16(V))
    assert word_of(g) is P.gather_bf16(V)
    rng = np.random.default_rng(seed)
    tok = _tokens(V, rng, n)
    col = rng.integers(0, 1 << 16, (len(tok), V), dtype=np.uint64).astype(np.uint16)
    _agree(g, tok, col)


@pytest.mark.parametrize("V", [1, 2, 3, 64, 300])
def test_the_gather_agrees_with_gather_bf16_v1(V: int) -> None:
    _agree_gather(V, V, 400)


@pytest.mark.parametrize("V", [1024, 1100, 2048, 3000])
def test_the_tree_of_calls_agrees_with_gather_bf16_v1(V: int) -> None:
    _agree_gather(V, V, 400)


def _agree_f32(V: int, seed: int, n: int) -> None:
    g = bind(BGA.gather_f32(V))
    assert word_of(g) is RP.gather_f32(V)
    rng = np.random.default_rng(seed)
    tok = _tokens(V, rng, n)
    col = rng.integers(0, 1 << 32, (len(tok), V), dtype=np.uint64)
    _agree(g, tok, col)
    past = np.array([V, (1 << 32) - 1], np.uint32)
    assert BS.evaluate_words(g, past, col[:2]).reshape(-1).tolist() == [BGA.F32_FILL] * 2


@pytest.mark.parametrize("V", [1, 3, 64])
def test_the_f32_gather_agrees_with_gather_f32_v1(V: int) -> None:
    _agree_f32(V, V, 400)


@pytest.mark.slow
@pytest.mark.parametrize("V", [1024, 1100, 3000])
def test_the_f32_tree_of_calls_agrees_with_gather_f32_v1(V: int) -> None:
    _agree_f32(V, V, 400)


def test_the_f32_gather_is_one_definition_for_the_word_family_and_the_expert_gathers() -> None:
    """`GatherF32x{V}_v1`'s Boolean version, made on demand by id, is `boolean_gather.gather_f32`'s, which `boolean_fp8_moe`'s
    expert gather returns too, whole (V <= 1024) and as a tree of Calls."""
    for V in (3, 1100, TRAINING_VOCAB):
        g = BB.boolean_version(RP.gather_f32(V))
        assert g.id == f"GatherF32x{V}_v2" and g.defn is BGA.gather_f32(V) is BFM.gather_f32(V)


def test_the_training_vocabulary_f32_gather_is_pinned() -> None:
    g = bind(BGA.gather_f32(TRAINING_VOCAB))
    assert BL.is_boolean(g)
    assert BS.compile(g).counts == {"and": 1610400, "xor": 3220096, "not": 198}
    assert [(n.fn.id, n.gates) for n in g.body.nodes] == [("GatherF32x1024u10_v1", 49 * 98208), ("GatherF32x128u10_v1", 12411),
                                                          ("GatherF32x50u22_v1", 6091)]


@pytest.mark.slow
def test_the_training_vocabulary_f32_gather_agrees_with_gather_f32x50304_v1() -> None:
    _agree_f32(TRAINING_VOCAB, 7, 24)


def test_the_v_chunks_are_the_fewest_of_at_most_w_words_as_even_as_whole_words_allow() -> None:
    assert BGA.chunk_widths(TRAINING_VOCAB, BGA.W_MAX) == [25152] * 2
    assert BGA.chunk_widths(LLAMA_VOCAB, BGA.W_MAX) == [42752] * 3
    assert BGA.chunk_widths(BGA.W_MAX, BGA.W_MAX) == [BGA.W_MAX] and BGA.chunk_widths(BGA.W_MAX + 1, BGA.W_MAX) == [23582] * 2
    assert BGA.chunk_widths(100, 40) == [34, 33, 33] and BGA.chunk_widths(5, 2) == [2, 2, 1]
    for V, w in ((0, 4), (4, 0), ((1 << 31) + 1, BGA.W_MAX)):
        with pytest.raises(ValueError, match="V-chunks of at most"):
            BGA.chunk_widths(V, w)
    with pytest.raises(ValueError, match="no offsets"):
        bind(BGA.GatherChunkOffsets, V=64, W=64).signature()


def _starts(V: int, w: int) -> list[int]:
    widths = BGA.chunk_widths(V, w)
    return [sum(widths[:j]) for j in range(len(widths) + 1)]


def _chunk_edges(V: int, w: int) -> list[int]:
    """Each V-chunk's first and last word and their neighbours, the column's ends, i32's extremes, and the negative tokens that
    wrap to the largest i32s once the last V-chunk's start is subtracted."""
    s = _starts(V, w)
    return ([x + d for x in s for d in (-1, 0, 1)] + [V - 1, -1, -V, (1 << 31) - 1, -(1 << 31), -(1 << 31) + 1, (1 << 31) - V]
            + [-(1 << 31) + s[-2] + d for d in (-1, 0)])


def _u32(tokens) -> np.ndarray:
    return np.array([t % (1 << 32) for t in tokens], np.uint32)


@pytest.mark.parametrize(("V", "w"), [(1, 1), (5, 2), (7, 3), (9, 4), (64, 64), (65, 64), (100, 25), (100, 40)])
def test_the_chunked_gather_is_the_whole_gather_at_every_token(V: int, w: int) -> None:
    """Every token in [-3, V + 3] and every edge (`_chunk_edges`), against `GatherF32x{V}_v1` and `GatherF32x{V}_v2`: one V-chunk
    (V <= w), uneven widths, a V-chunk of one word, and an even and an odd number of V-chunks, so a sum with and without the
    extra fill."""
    f = bind(BGA.GatherF32Chunked, V=V, W=w)
    assert word_of(f) is RP.gather_f32(V) and BL.is_boolean(f)
    first = f"GatherChunkOffsets_v1{{V={V},W={w}}}" if len(BGA.chunk_widths(V, w)) > 1 else f"GatherF32x{V}_v2"
    assert f.body.nodes[0].fn.id == first
    tok = _u32(list(range(-3, V + 4)) + _chunk_edges(V, w))
    _agree_chunked(f, tok, np.random.default_rng(V * w).integers(0, 1 << 32, (len(tok), V), dtype=np.uint64))


def _agree_chunked(f, tok: np.ndarray, col: np.ndarray) -> None:
    """``f`` is `GatherF32x{V}_v1` and `GatherF32x{V}_v2` at ``tok``, and a token outside [0, V) selects 0x7FC00000."""
    V = f.bindings["V"]
    _agree(f, tok, col)
    got = BS.evaluate_words(f, tok, col).reshape(-1)
    assert np.array_equal(got, BS.evaluate_words(bind(BGA.gather_f32(V)), tok, col).reshape(-1))
    off = tok >= V
    assert off.any() and (got[off] == BGA.F32_FILL).all()


@pytest.mark.slow
@pytest.mark.parametrize("w", [2048, 1500])
def test_the_chunked_gather_over_trees_of_calls_is_the_whole_gather(w: int) -> None:
    """Two V-chunks of 2,048 words and three of 1,366 and 1,365, each a tree of Calls, the latter with a partial last chunk: the
    trees' edges and the V-chunks', and random tokens."""
    rng = np.random.default_rng(w)
    tok = np.concatenate([_tokens(4096, rng, 200), _u32(_chunk_edges(4096, w))])
    _agree_chunked(bind(BGA.GatherF32Chunked, V=4096, W=w), tok, rng.integers(0, 1 << 32, (len(tok), 4096), dtype=np.uint64))


#: V -> the chunked gather's body (its nodes' ids and gates), a V-chunk's body, and the chunked gather's ANDs
CHUNKED = {
    TRAINING_VOCAB: ([("GatherChunkOffsets_v1{V=50304,W=47163}", 137), ("GatherF32x25152_v2", 4832350), ("GatherChunkXor_v1{C=2}", 41)],
                     [("GatherF32x1024u10_v1", 2356992), ("GatherF32x576u10_v1", 55419), ("GatherF32x25u22_v1", 3764)], 1611160),
    LLAMA_VOCAB: ([("GatherChunkOffsets_v1{V=128256,W=47163}", 224), ("GatherF32x42752_v2", 12316668), ("GatherChunkXor_v1{C=3}", 64)],
                  [("GatherF32x1024u10_v1", 4026528), ("GatherF32x768u10_v1", 73705), ("GatherF32x42u22_v1", 5323)], 4106059),
}


@pytest.mark.parametrize("V", sorted(CHUNKED))
def test_the_vocabularies_chunked_gathers_are_pinned(V: int) -> None:
    """The training vocabulary in two V-chunks of 25,152 words and Llama 3's in three of 42,752, each a batch of one gather
    behind offsets of 24 and 43 ANDs, then a two- and a three-word XOR of no AND."""
    nodes, chunk, _ = CHUNKED[V]
    g = bind(BGA.GatherF32Chunked, V=V, W=BGA.W_MAX)
    assert BL.is_boolean(g) and word_of(g) is RP.gather_f32(V)
    assert [(n.fn.id, n.gates) for n in g.body.nodes] == nodes
    assert [(n.fn.id, n.gates) for n in g.body.nodes[1].fn.body.nodes] == chunk


@pytest.mark.slow
@pytest.mark.parametrize("V", sorted(CHUNKED))
def test_the_vocabularies_chunked_gathers_are_the_whole_gather(V: int) -> None:
    """At every V-chunk edge (`_chunk_edges`), past the column and at random tokens in it, against `GatherF32x{V}_v1` and
    `GatherF32x{V}_v2`. The pinned bodies hold the shapes; only evaluating the full circuits (millions of gates, a minute or
    more each) shows that the offsets at these starts route each token to its own V-chunk's word."""
    g = bind(BGA.GatherF32Chunked, V=V, W=BGA.W_MAX)
    assert BS.compile(g).counts["and"] == CHUNKED[V][2]
    rng = np.random.default_rng(V)
    tok = np.concatenate([_u32(_chunk_edges(V, BGA.W_MAX) + [V, V + 1, 1 << 31, (1 << 32) - 1]), rng.integers(0, V, 16).astype(np.uint32)])
    _agree_chunked(g, tok, rng.integers(0, 1 << 32, (len(tok), V), dtype=np.uint64))


def _block(d) -> dict:
    """``d`` as one Flock block, as `benchmarks/unit_block` composes it (each parameter one ``hm96-sha512/row/v2`` bit row and
    its return one more): its META."""
    return CS.UnitStager(d, f"unit-block/{d.name}", [(p, V2, t.leaves, [p]) for p, t in d.signature()[0]], [(V2, (0,))]).meta


@pytest.mark.slow
def test_w_max_is_the_most_words_whose_gather_and_its_rows_openings_fit_one_block() -> None:
    """`GatherF32x{W_MAX}_v2` composes into exactly 2^27 bits, its unit in a 2^22 slot and the column row's 1,473 compressions
    in 491 sha512x3 slots, and one more word needs a 492nd slot and a block of 2^28, which `compose` refuses. Only `compose`
    itself shows that edge (two builds, about a minute each): the module doc's arithmetic is its layout's."""
    m = _block(bind(BGA.gather_f32(BGA.W_MAX)))
    assert (m["k_log"], m["unit_log"], m["per_vu"]["sha512x3"]) == (27, 22, 491)
    assert max((r["pos0"] + r["count"]) << r["slot_log"] for r in m["ranges"]) == 1 << 27
    with pytest.raises(ValueError, match=r"a block of 2\^28 bits"):
        _block(bind(BGA.gather_f32(BGA.W_MAX + 1)))


def test_the_served_gather_is_pinned_and_agrees_with_gather_bf16x49152_v1() -> None:
    g = bind(BGA.gather_bf16(SERVED_VOCAB))
    assert BL.is_boolean(g)
    assert BS.compile(g).counts == {"and": 786688, "xor": 1573104, "not": 255}
    assert [(n.fn.id, n.gates) for n in g.body.nodes] == [("GatherBf16x1024u10_v1", 48 * 49104), ("GatherBf16x48u22_v1", 3055)]
    _agree_gather(SERVED_VOCAB, 7, 24)


@pytest.mark.parametrize(("V", "H"), [(64, 16), (5, 3)])
def test_the_embedding_agrees_with_embedding_v1(V: int, H: int) -> None:
    e = bind(BGA.Embedding, V=V, H=H)
    assert word_of(e).id == f"Embedding_v1{{V={V},H={H}}}"
    rng = np.random.default_rng(V * H)
    tok = _tokens(V, rng, 100)
    _agree(e, tok, rng.integers(0, 1 << 16, (len(tok), V, H), dtype=np.uint64).astype(np.uint16))


@pytest.mark.parametrize("width", [1, 5, 32])
def test_bit_at_agrees_with_bit_at_v1(width: int) -> None:
    """Every index from -3 to W + 2, i32's extremes, then in-range and arbitrary indices over random words."""
    b = bind(BGA.bit_at(width))
    assert word_of(b).id == f"BitAtx{width}_v1" and BGA.bit_at(width) is BGA.bit_at(width)
    rng = np.random.default_rng(width)
    edge = list(range(-3, width + 3)) + [(1 << 31) - 1, 1 << 31, (1 << 32) - 1]
    i = np.array([e % (1 << 32) for e in edge] + list(rng.integers(0, width, 300)) + list(rng.integers(0, 1 << 32, 100)),
                 np.uint32)
    _agree(b, rng.integers(0, 1 << width, len(i), dtype=np.uint64), i)


def test_the_embedding_over_a_tree_agrees_with_embedding_v1() -> None:
    e = bind(BGA.Embedding, V=1100, H=3)
    rng = np.random.default_rng(11)
    tok = _tokens(1100, rng, 60)
    _agree(e, tok, rng.integers(0, 1 << 16, (len(tok), 1100, 3), dtype=np.uint64).astype(np.uint16))


def _cut(fn, units: int) -> None:
    rule = W.unit_rule(fn, 16, 32, memo={})
    assert rule["violations"] == []
    assert (rule["units_per_call"], rule["max_out_bits"], rule["committed_interior_words"]) == (units, 16, 0)
    G = W.Graph(fn)
    R = W.units(G, 16, 32)
    assert (len(R["kinds"]), R["cut"].codes, R["cut"].detail["redundant_gates"], len(G.recomputed)) == (units, (), 0, 0)


def test_the_embedding_cuts_into_one_word_per_coordinate() -> None:
    """The embedding's body is separable (its gathers read only its parameters and it returns them all), so `Q_word_v1` cuts it
    gather by gather; no gate of one gather is another's, so the whole graph's cut agrees and recomputes nothing."""
    _cut(bind(BGA.Embedding, V=64, H=16), 16)


def test_a_tree_of_calls_cuts_into_one_word() -> None:
    """The chunk gathers' words are read only by the gather above them, which returns 16 bits: one output unit, nothing committed
    inside (`circuit-check` holds `Embedding_v2{V=1100,H=2}` over two such trees to the same)."""
    _cut(bind(BGA.gather_bf16(1100)), 1)


def test_what_a_gather_cannot_select_is_refused() -> None:
    for n, k in ((0, 10), (1025, 10), (5, 2)):
        with pytest.raises(ValueError, match="selects from"):
            BGA.gather_index(n, k)
    with pytest.raises(ValueError, match="at least one word"):
        BGA.gather_bf16(0)
    with pytest.raises(ValueError, match="at least one bit"):
        BGA.bit_at(0)
