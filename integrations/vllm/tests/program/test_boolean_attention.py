"""`boolean_attention`: the v6 chain (`AttnBlock_v6` / `AttentionHead_v6` / `Attention_v6`), the v7 chain on both of its
profiles (FA2 on mma.sync, FA3 on wgmma), the v8 chain at the served FA2 binding, the v9 chain at FA3's and `Fa3InvSum_v2` are
Boolean, agree with their word views (the v3, v2, v5 and v4 chains and `Fa3InvSum_v1`) on random words and on words that reach
every guard (a zero query, signed zeros, subnormals, infinities, NaNs, scores that overflow), and are pinned by gate count and
program digest."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.batch import type_shape
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import bind
from verity.ml.circuits.boolean import trace as TR
from verity_vllm.program import boolean as SW
from verity_vllm.program.registry import boolean_attention as BAt

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "AttnBlock_v6{D=16,NVIS=5,FIRST=True,BN=16}":
        (207368, 401776, 20609, "ab2416cc9c468378bdd59613e568da63db6f797066618364e182247cf70289d6"),
    "AttnBlock_v6{D=16,NVIS=16,FIRST=False,BN=16}":
        (435868, 848195, 39421, "29d550d42be678bb90f181253d042b4b0a25e34f9df2aae04c0fbb1c716c4fbb"),
    "AttentionHead_v6{T=20,D=16,BN=16}":
        (673779, 1306610, 61173, "d395754e7904c882aca9ce74741b08ff0d7588621e15c7531ec07e13b9e826f7"),
    "Attention_v6{T=20,NH=4,KVH=2,D=16,BN=16}":
        (2695116, 5226440, 244692, "756f612ee6c53ec47dab52cc696841097d000045ea24c186adf944b04b7dc6ab"),
    "Fa3InvSum_v2": (947, 4866, 102, "47bce9f5362663c4900c1cb1161e72c82290cba29067319c84c833866d58ac1a"),
    "AttnBlock_v7{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2}":
        (423452, 826819, 36253, "f368b54a12412459aab96880cc583052ed0106a59d74b29b831c1a249c9ef7cb"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=AmpereBF16TcDot16_v3,INV=Fa2InvSum_v2}":
        (673779, 1306610, 61173, "1d5dcacaed0741eb24c0a81d012bdffa83a3de4997e23ace8f9b63f9aa221331"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (653603, 1271874, 56018, "b93597b1ddc67975a6ca60a168370e6530ed2d31baeffa872f4963894d6c15d6"),
    "Attention_v7{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (2614412, 5087496, 224072, "64b3e252a8279e2c66736791a7d8ab109868af1843941878c9d0412ee72d9e48"),
    "AttnBlock_v8{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2,CHECK=False}":
        (423452, 826819, 36253, "42aeb37ee1c67679030a82f40597352cf8d6beb1edcd05d30eeccc54f041a316"),
    "AttentionHead_v8{T=40,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=1}":
        (1134159, 2213423, 97356, "11355fbfd9d1b67e9f2c8c4a83f7b6ffd0f68d3735321fec0a8cb6fecb88009c"),
    "Attention_v8{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=0}":
        (2614412, 5087496, 224100, "d5fa69f89e7905c7ea19ad232df1442652329ceb0a776f3c07f926449b09a41b"),
    "AttnBlock_v9{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2,CHECK=False}":
        (423389, 826819, 36229, "d3d95e63b62cd24677de54e94d3cc2dc5ba9d4bb875c6d8098e95c13a8515fb4"),
    "AttentionHead_v9{T=40,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2,MASKED_FROM=2}":
        (1134033, 2213423, 97301, "dc0c870d652b50b92f6a4b67cf942805a2d905c82b19830e12836a8868e2026a"),
    "Attention_v9{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2,MASKED_FROM=1}":
        (2614160, 5087496, 223976, "8f8ef0b7049dc2d38a587748246df2410e5e80b439a5ada9c45026d2035d4cad"),
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


@pytest.mark.parametrize("target", PINNED, ids=lambda t: t.id)
def test_the_switch_maps_its_word_view_to_it(target) -> None:
    """`boolean_version` of each word chain is the Boolean chain whose word view it is: `Attention_v4` (FA3's per-iteration
    `Check_inf`) to v9, `Attention_v5` (FA2's) to v8, v2 and v3 to v7 and v6."""
    assert SW.boolean_version(TR.word_of(target)).id == target.id


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
