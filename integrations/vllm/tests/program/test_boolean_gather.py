"""`verity_vllm.program.registry.boolean_gather`: the Boolean vocabulary gathers and embedding agree with `GatherBf16x{V}_v1` and
`Embedding_v1` on edge and random tokens (negative, past the vocabulary, at every chunk boundary), are pinned by gate count and
program digest, and cut under `Q_word_v1{X=16,W=32}` into one unit of 16 output bits per gather, with no gate two gathers compute.
The gathers over more than 1024 words are trees of Calls whose chunk Definition takes seconds to compile, so their tests are
slow; the quick ones cover the multiplexer tree traced whole."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.boolean.trace import word_of
from verity_vllm.program.registry import boolean_gather as BGA
from verity_vllm.program.registry import prims as P
from verity_vllm.query import word as W

SERVED_VOCAB = 49152

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "GatherBf16x64_v2": (1424, 2432, 390, "2e1cfba6e2a01125cdf2ea16da210c91f6f71d08c3719d370ad85ca447aec674b948e4317e5398c23366f9ead77299041628401a467b83fd720d5864c10d9f16"),
    "GatherBf16x76u10_v1": (1296, 2496, 90, "8e52f98031c197156ad9c2b522590e891442fe20131a156b01239f595ce6e752b84477a49d1a09af9846985dc76b0adbce36ffee7314e6cf5987cb897f40be92"),
    "GatherBf16x2u22_v1": (352, 368, 315, "644d53b8b1d12a48010d7b92b906f1fb59aa4b73d64110d72bde8c2e28c9c03a922c49dcd968e8ef08b1c8606bc6894a8731a970f5d21c36f5adc5a34cd36a0e"),
    "Embedding_v2{V=64,H=16}": (22784, 38912, 6240, "dc6159de5ce51bacbf41b8e34ba63754be24b70569bfa501765e6396cabb66476f1f5936e97870fc115cf1a984ffe51c8cf384c3e64547dc2bfcf19b45117b58"),
    "GatherBf16x1024u10_v1": (16368, 32736, 0, "342ae6660d384c030788a4e8d1acd46784bb40d4bd1c49f027378765bc35c19036ff5ec3d241ca671d94f34375f48902539942fe64097d8ac4d2d1cb44796b6b"),
    "GatherBf16x1100_v2": (18016, 35600, 405, "13b5d96e04c3c84641ee9330dd9e2d1a14224fc79cf0603aa9216b8e135f1db500f807f3f1a5568c2eec14653871caac90ff709e1fd3b400739046ae18e28d77"),
    "Embedding_v2{V=1100,H=2}": (36032, 71200, 810, "01796fa581b87c15391b8bc98d1789432a6a82b1e928435b46853c81bd15346ebc0d78acaa6db9fdc2baadecb90849af27852c9de053dad3899c6c369d8c0387"),
}
QUICK = [bind(BGA.gather_bf16(64)), bind(BGA.gather_index(76, 10)), bind(BGA.gather_index(2, 22)), bind(BGA.Embedding, V=64, H=16)]
TREES = [bind(BGA.gather_index(1024, 10)), bind(BGA.gather_bf16(1100)), bind(BGA.Embedding, V=1100, H=2)]


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


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.parametrize("V", [1024, 1100, 2048, 3000])
def test_the_tree_of_calls_agrees_with_gather_bf16_v1(V: int) -> None:
    _agree_gather(V, V, 400)


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
