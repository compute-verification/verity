"""`boolean_attention`: the v6 chain (`AttnBlock_v6` / `AttentionHead_v6` / `Attention_v6`), the v7 chain on both of its
profiles (FA2 on mma.sync, FA3 on wgmma), the v8 chain at the served FA2 binding, the v9 chain at FA3's and `Fa3InvSum_v2` are
Boolean, agree with their word views (the v3, v2, v5 and v4 chains and `Fa3InvSum_v1`) on random words and on words that reach
every guard (a zero query, signed zeros, subnormals, infinities, NaNs, scores that overflow), and are pinned by gate count and
program digest."""
from __future__ import annotations

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.batch import type_shape
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import bind
from verity.ml.boolean import trace as TR
from verity_vllm.program.registry import boolean_attention as BAt

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "AttnBlock_v6{D=16,NVIS=5,FIRST=True,BN=16}":
        (208773, 405521, 20684, "aa9a44d89242606c9e0bbe11b21020face4ad38019c26af779aa7403b08df976"),
    "AttnBlock_v6{D=16,NVIS=16,FIRST=False,BN=16}":
        (440645, 860928, 39676, "8c40e141a07ac1b90952c6b0acf7fad468482596d59d34acf0fb21cab1ae6d16"),
    "AttentionHead_v6{T=20,D=16,BN=16}":
        (679680, 1323168, 61457, "6dbcba90ccba5b0e33fcfff7254a34fef29c671ee42c754b2addaeb8755db868"),
    "Attention_v6{T=20,NH=4,KVH=2,D=16,BN=16}":
        (2718720, 5292672, 245828, "a808120f183c73409b306fe87ebdd2180b8235aa6c85e5e75465c9a5a2f3313e"),
    "Fa3InvSum_v2": (947, 5695, 71, "c0ce2f7ff45cfa8da67b76e115100b9b34b691166dbb6c20689f2d6162b447bb"),
    "AttnBlock_v7{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2}":
        (428229, 839552, 36508, "64c27b5a8cc8387141408ca780ee0da8e1d62119a38c27d52c1a37dfea0332ea"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=AmpereBF16TcDot16_v3,INV=Fa2InvSum_v2}":
        (679680, 1323168, 61457, "434ce1efc9d0633da9094bcd9d00c873c18abc975c40bbe5b6986822a47e4f80"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (659504, 1288432, 56302, "4e58aef2275351298fa711950888b1d07fc50d72f76c5e62d6a51b33e45ddd0a"),
    "Attention_v7{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (2638016, 5153728, 225208, "7e98b2335269aeb4fe3ebc80427d7995aa116540a7014abea5517a64e54e7b8e"),
    "AttnBlock_v8{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2,CHECK=False}":
        (428229, 839552, 36508, "ca47a0cd1099ca8c223a0e85a00ad393616e22532d00e9f34034ae6d389a99b0"),
    "AttentionHead_v8{T=40,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=1}":
        (1145961, 2245710, 97955, "50f5098203a04e2196cbba1fcd9e66d0e7b7c7d958857d64d894f90f16469ecd"),
    "Attention_v8{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=0}":
        (2638016, 5153728, 225236, "eb4a392c51e80c3cc829c1bc7e3a03e68ceeadef2d7087f3a1ce6c855463fcac"),
    "AttnBlock_v9{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2,CHECK=False}":
        (428166, 839552, 36484, "d6f50a22f6d994954154ae81be103bae9ba467b022e8a669220cdf4cf9e73c05"),
    "AttentionHead_v9{T=40,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2,MASKED_FROM=2}":
        (1145835, 2245710, 97900, "e7f7a7130955cb45594def50de48b3862ec7e8d3e979169c98531fd8f1867b48"),
    "Attention_v9{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2,MASKED_FROM=1}":
        (2637764, 5153728, 225112, "aad5a4ddf6846ee43fb86eb491d2303eb7cc2435d98887ce7dba1633412cfbee"),
}

FA2, FA3 = BAt.ATTENTION_FA2, BAt.ATTENTION_FA3
#: the served FA2 row's `Attention_v5` binding (`kernels.rows`): the wgmma step and FA2's inverse sum
FA2_CHECK = {"DOT": FA3["DOT"], "INV": FA2["INV"]}
PINNED = [bind(BAt.AttnBlockV6, D=16, NVIS=5, FIRST=True, BN=16), bind(BAt.AttnBlockV6, D=16, NVIS=16, FIRST=False, BN=16),
          bind(BAt.AttentionHeadV6, T=20, D=16, BN=16), bind(BAt.AttentionV6, T=20, NH=4, KVH=2, D=16, BN=16), bind(BAt.Fa3InvSum),
          bind(BAt.AttnBlockV7, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"]), bind(BAt.AttentionHeadV7, T=20, D=16, BN=16, **FA2),
          bind(BAt.AttentionHeadV7, T=20, D=16, BN=16, **FA3), bind(BAt.AttentionV7, T=20, NH=4, KVH=2, D=16, BN=16, **FA3),
          bind(BAt.AttnBlockV8, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"], CHECK=False),
          bind(BAt.AttentionHeadV8, T=40, D=16, BN=16, MASKED_FROM=1, **FA2_CHECK),
          bind(BAt.AttentionV8, T=20, NH=4, KVH=2, D=16, BN=16, MASKED_FROM=0, **FA2_CHECK),
          bind(BAt.AttnBlockV9, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"], CHECK=False),
          bind(BAt.AttentionHeadV9, T=40, D=16, BN=16, MASKED_FROM=2, **FA3),
          bind(BAt.AttentionV9, T=20, NH=4, KVH=2, D=16, BN=16, MASKED_FROM=1, **FA3)]

#: one block per mask shape and per state; one and two blocks, a partial last block, BN = 32; GQA with two KV heads; v7 on both
#: profiles; v8 and v9 with an unguarded block, a head with an unguarded block, and GQA
AGREEMENT = PINNED[:2] + [bind(BAt.AttnBlockV6, D=16, NVIS=9, FIRST=False, BN=32), bind(BAt.AttentionHeadV6, T=5, D=16, BN=16),
                          bind(BAt.AttentionHeadV6, T=16, D=16, BN=16), bind(BAt.AttentionHeadV6, T=40, D=16, BN=32),
                          bind(BAt.AttentionV6, T=20, NH=4, KVH=2, D=16, BN=16)] + PINNED[5:] + [
    bind(BAt.AttentionHeadV7, T=40, D=16, BN=32, **FA3)]

#: bf16 words that reach FA2's guards: signed zeros, subnormals, the normal range's ends, infinities, NaNs, 2^63
SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x5F00, 0xDF00],
                    np.uint16)


def _words(rng, shape, scale: float, width: int) -> np.ndarray:
    x = (rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32)
    return x if width == 32 else (x >> 16).astype(np.uint16)


def _inputs(word, n: int, seed: int) -> list[np.ndarray]:
    """``n`` instances per parameter of ``word``: the first third gaussian at softmax's working scales, the second with special
    words planted at random positions, the last with a few entries near 2^63 (scores that overflow to infinity); the first
    instance's query is zero. The f32 state of a block (m, l, o) is gaussian, l's sums positive."""
    rng = np.random.default_rng(seed)
    third = n // 3
    out = []
    for j, (name, t) in enumerate(word.signature()[0]):
        shape, width = (n, *type_shape(t)), max(t.width_set())
        if width == 32:
            a = _words(rng, shape, 4.0, 32)
            if name == "l_prev":
                a &= np.uint32(0x7FFFFFFF)
            out.append(a)
            continue
        a = _words(rng, shape, 2.0 if j == 0 else 1.0, 16)
        mid, end = a[third:2 * third], a[2 * third:]
        m = rng.random(mid.shape) < 0.05
        mid[m] = rng.choice(SPECIALS, int(m.sum()))
        m = rng.random(end.shape) < 0.02
        end[m] = (rng.integers(0x5E00, 0x5F80, int(m.sum())) | (rng.integers(0, 2, int(m.sum())) << 15)).astype(np.uint16)
        if j == 0:
            a[0] = 0
        out.append(a)
    return out


@pytest.mark.parametrize("target", PINNED, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)) == digest


def test_fa3_inv_sum_agrees_with_fa3_inv_sum_v1() -> None:
    """On random words, the special words (signed zeros, infinities, NaNs, subnormals) and every mantissa at exponent 0x7F."""
    target = bind(BAt.Fa3InvSum)
    words = np.concatenate([np.random.default_rng(20).integers(0, 1 << 32, 200_000, dtype=np.uint64).astype(np.uint32),
                            np.array([0, 0x80000000, 0x7FC00000, 0xFFC00001, 0x7F800000, 0xFF800000, 1, 0x00800000], np.uint32),
                            np.arange(1 << 23, dtype=np.uint32) | np.uint32(0x3F800000)])
    got = np.asarray(BS.evaluate_words(target, words)).astype(np.uint64)
    assert np.array_equal(got, np.asarray(E.evaluate_batch(TR.word_of(target), words)).astype(np.uint64))


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k in ("DOT", "INV") else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, {6: 3, 7: 2, 8: 5, 9: 4}[target.defn.version], lifted)
    inputs = _inputs(word, 96, sum(map(ord, target.id)))
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64)
    ref = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64)
    bad = np.flatnonzero((got != ref).reshape(len(got), -1).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} instances differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("block", [BAt.AttnBlockV8, BAt.AttnBlockV9], ids=lambda d: d.id)
def test_an_unguarded_block_rescales_by_the_running_max_as_it_is(block) -> None:
    """A block entered with the running max at -inf and scores of -inf only (an infinite query against negative keys) keeps it at
    -inf: `AttnBlock_v8{CHECK=False}` rescales by exp2(-inf - -inf), a NaN, as `AttnBlock_v5{CHECK=False}` does, and with `CHECK`
    by exp2(-inf - 0) = 0, so the static is read. `AttnBlock_v9{CHECK=False}` also scales the scores by the unguarded max, as
    `AttnBlock_v4{CHECK=False}` does, which a block guarding it as v8 does would not match."""
    blocks = [bind(block, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"], CHECK=c) for c in (False, True)]
    inputs = _inputs(TR.word_of(blocks[0]), 12, 85)
    q, k, m_prev = inputs[0], inputs[1], inputs[3]
    q[:6] = 0
    q[:6, 0] = 0x7F80
    k[:6] = 0
    k[:6, :, 0] = 0xBF80
    m_prev[:6] = 0xFF800000
    outs = []
    for target in blocks:
        got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64)
        assert np.array_equal(got, np.asarray(E.evaluate_batch(TR.word_of(target), *inputs)).astype(np.uint64)), target.id
        outs.append(got.reshape(len(got), -1))
    assert (outs[0][:6] != outs[1][:6]).any(axis=1).all()
