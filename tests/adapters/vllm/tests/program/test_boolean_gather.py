"""`verity_vllm.definitions.registry.boolean_gather`: the Boolean vocabulary gathers, embedding and bit selection agree with
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
from verity.circuits import boolean as BL
from verity.circuits import codec
from verity.circuits import evaluation as E
from verity.circuits import word as W
from verity.circuits.boolean.trace import word_of
from verity.circuits.defs import REGISTRY, bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import standalone_call
from verity.circuits.ops.boolean import versions as BB
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import class_statement as CS

from verity_vllm.definitions.registry import boolean_fp8_moe as BFM
from verity_vllm.definitions.registry import boolean_gather as BGA
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry import ref_prims as RP

SERVED_VOCAB = 49152
#: the training step's vocabulary, over which it gathers an f32 word (`GatherF32x50304_v1`)
TRAINING_VOCAB = 50304
#: Llama 3's vocabulary, three V-chunks of at most `BGA.W_MAX` words
LLAMA_VOCAB = 128256

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "GatherBf16x64_v2": (1424, 2432, 390, "4257b4ceb60eb8ac8ad86c930772f258d6c8be2ecff6c488a8a66e87d87edb48019ccc758fb21ae1c15b9227a606aca36646f8e7f65c214918994d2210db6c2d"),
    "GatherBf16x76u10_v1": (1296, 2496, 90, "85a48cd09e77f738396b81bbd24ba6c395bf51a6d25c6ada0f84bacc9dd5478ab6e3f819633278106d4b70a462ed10bb16e9e2db9eb8f195e2c60b9222616b94"),
    "GatherBf16x2u22_v1": (352, 368, 315, "fef8f03d2f7e0474c97e7d9c452796829761739d8d6eac135340dc9b2337789e7163f201367456013a94a61de5ff8e509acc5f1b15b4289da70c33b1205631b3"),
    "Embedding_v2{V=64,H=16}": (22784, 38912, 6240, "e980a265b3dc0ebc5a5fe64177d0e5803e174d75a0812732459d1af44b717cdef082a95341921a3a6f865129c4de7d5617d8523e79630b73641defbd614c14e4"),
    "GatherBf16x1024u10_v1": (16368, 32736, 0, "19c602222b1016c2f098e6ad7540b9a68d55eeeb76c94380a297a6d91dae47fde9cbb43d232e4b195ca0024ef974eee719ae351f84b7f7f1015f1bf6c356f418"),
    "GatherBf16x1100_v2": (18016, 35600, 405, "5b33e21c820f248d993621d1153fee0a17bb670fd97fd17d376ce26bd30eebbe66d561d79046a16d7a42b1f8683a97d2ee97a242b24d7948f7551c468838eafe"),
    "Embedding_v2{V=1100,H=2}": (36032, 71200, 810, "2f5a4c77a173374dc4628f004b5188f00b9aecec43743809be11e1fa0bf043917748e9b264747ebf0cda6bbfd57feecc988becb216f11ef00e7cd68665f2f929"),
    "GatherF32x64_v2": (2848, 4864, 234, "f8abdf3b793a789cc399f04e871b947ef0aed52bece81f6fd58ea21a6a3b7353858f973774dd173d8d96b4eee3f0b5b3e2d5cee829718255b64276bef0d4e865"),
    "GatherF32x128u10_v1": (4160, 8224, 27, "d5e74f36e72a8c2e893121ee54bfc86d2e497c2311ab0ca79d7c338c29e890bac113d180443a145d11ee80c6b23aa271f9449d82ad599e1100a190f2fd757725"),
    "GatherF32x50u22_v1": (2176, 3744, 171, "5be56dfe0170033bea30af09120636eabcd084f286464672e56e5b0886bd68d89c58e67491041162749b49af223203139b99a599a8da442687510513d7b27e56"),
    "GatherF32x1024u10_v1": (32736, 65472, 0, "43cb7a8bba1eae3790dc84e3926e891fcfa2906c2997c8f4576e9ea29e5538bc4946fbebeb09ebece19e41348d66f7c97823957076e645671014e656794ddbe9"),
    "GatherF32Chunked_v1{V=100,W=40}": (6107, 9392, 915, "d40a5f8ba12ba89ebd41bc00691a3e48097cf243d35f4dd3e2a8b29a43c50052a550b1b17cd7d1542e3b1c1d122aff4e6ce583442115fe8eb854875c0cbf975c"),
    "GatherChunkOffsets_v1{V=100,W=40}": (59, 176, 87, "af3b638af219e1a6bdd8b58880452a5b4dae9f2e2c764d409c142d6fea4fec03cd7699007abb8605fd12c30ed0f068fd435a92db6549625b9431322f22de9f4b"),
    "GatherChunkOffsets_v1{V=50304,W=47163}": (24, 70, 43, "858ef94b726b375981dd277087a93dc81b2dc1ac906e0fb96555bf7619b22a74c5118b22c9a8e9408303203bfb638dc20bcba9b26412a48f37d6b7a41a9828e6"),
    "GatherChunkOffsets_v1{V=128256,W=47163}": (43, 123, 58, "d6c397e2cc193f247cc0251bae58f8e54632bd9692a58c5f991de6282ba0296f6349a0499d825b1c5b1e625b383a0c15fe0d2482ce38fcf908b6fa70573e3caf"),
    "GatherChunkXor_v1{C=2}": (0, 32, 9, "3dc042098d7a71558d4a421b581c13fa3fd1c8bd893b50a371132ec40f8ba6df425318264ed95bb020a164b69aca182f1b25ca266509c5f91da30b17b87e6b1e"),
    "GatherChunkXor_v1{C=3}": (0, 64, 0, "9d365b8b6104de25561c1a086085784350bc7e94e30e3ebe32ef0e932cf673e6ef8148086867238bff271faae3b144638a342e023cd071d9f7ee0b88e8f3ae8f"),
    "GatherF32x576u10_v1": (18496, 36896, 27, "6828395f74508229bd708b03e1f7f41983c3dc108608a6031421796b4307a1d99a0042b728dc46c03ed88ffcf52325cc6c202c7ae6a077cf732512a44f11ea2d"),
    "GatherF32x25u22_v1": (1408, 2176, 180, "209fc9f6bb783c9e300205400f66f7ff7d7b0c3ef03ff8956e77fce02bcc921fcf8bb6f42689d18fd292c8f7fcda82b4bc3a6fcc73869f2a2811da5455b9f777"),
    "GatherF32x768u10_v1": (24576, 49120, 9, "e84e2f5e70eadffc3806de29efb62065e029dcabe0fd6530bb62247f35ab7e6ede4d9da491eb631ece94668986ed2ba804e6a922fbe0519f798ab5a304675352"),
    "GatherF32x42u22_v1": (1920, 3232, 171, "9aafd9116e0323173907237d743e19944aa59ba0e60676ca88e7c5d9f66be88ec695fa4ef7364fa616b0bad28e99e4c734cbe1f450f3dd89ee3b17db8736f991"),
    "BitAtx32_v2": (58, 89, 0, "bcafb75409a723b3bf77d5b2c0f0d5c4f0614fdc9a9c1eccc98943cdf5533bc93d7490cc8475b42e2d0e2e1fab0e60bfd73c73871bd5ad6422a180fad5b81117"),
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
    """``d`` as one Flock block (each parameter one ``hm96-sha512/row/v2`` bit row and
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
