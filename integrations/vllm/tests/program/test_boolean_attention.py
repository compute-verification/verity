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
        (207368, 401776, 20609, "8bf294c4e2261155f51d7813250f7ce10876b4ba2dc45d58fc8eedd422f7d5e2e1226a96a6039074f60b551f4d9415a593f67ba2f223c300a83293b8ce0c9952"),
    "AttnBlock_v6{D=16,NVIS=16,FIRST=False,BN=16}":
        (435868, 848195, 39421, "19a68a87889fefbf58241be6573f710721c42fb4a6b318f86259f38e6a722f37e57d587ce00c4cee995f47162edde9c3aa632d6e4d47bc62d23137e7784034cb"),
    "AttentionHead_v6{T=20,D=16,BN=16}":
        (673779, 1306610, 61173, "967e5dd8d1832716dc26c8b6f66fbffb012fb2914f20f3791fe22b64f19e1f7806863cc9b7e74c67a0738d78d12c7a1664fa0b9a3393e98ab9d3defbd646ae33"),
    "Attention_v6{T=20,NH=4,KVH=2,D=16,BN=16}":
        (2695116, 5226440, 244692, "27a8d84292ec0b928a4e06cb6390aa8416d97a5ce73cf01a5a4245b62fbb3324bb3b62902616a04d97bd3c2ed7ab378ba3f55753605c1633072f701cfd619094"),
    "Fa3InvSum_v2": (947, 4866, 102, "017c77eb48a6d4ed9be9ddaa291fa4c3ae280333540ed06045fef47238963ed59268c32fb069f62910b83e099df8eded6bba977ae659d6b42280ca8d8d653c5d"),
    "AttnBlock_v7{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2}":
        (423452, 826819, 36253, "d92cfd5e65517fc26f16b0a3b6bb7299cfc03700d6779850414fe1c6d74af9429b77fddb36bdd7f2ae7d1f2ffb9fcf65d894404b4e4ae4c7ab0b1a0f25f93373"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=AmpereBF16TcDot16_v3,INV=Fa2InvSum_v2}":
        (673779, 1306610, 61173, "fd36c8a200873698b9e36b1e397f4325e89fc06a730d1d7a1ae04a69cb2cf135d81ea2d26f0234197c1750a3c345cacf3cebbe9bb31f85053071ef07ef4e493d"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (653603, 1271874, 56018, "6f6c1e3b711a9a9c8945f177798a73a507bb950dc7a944e2a57c5c18f7bedf5109d8d4b8991242eb5f172499ea35f24fcf9b210ba2986d0c5a0860aec1aca844"),
    "Attention_v7{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (2614412, 5087496, 224072, "ffe2f2a05f7fe9100bfd69fcc393c1862f370134702fb1bdd967c4f2310ba33502fc588754b56145b4602099ccef5230b1aa90f24b029bfc152caa1e545857d1"),
    "AttnBlock_v8{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2,CHECK=False}":
        (423452, 826819, 36253, "f1ecfec2525794dd9755b9e409d764c99402dd1778d8cdfedaadf4cb06fa23b045c894204bedea2313090e499c2b3b534762ea316dc93d3d5430c18cea9a2ec9"),
    "AttentionHead_v8{T=40,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=1}":
        (1134159, 2213423, 97356, "07eaeb2ecd4ff38c7865ba7946c083432363e8a04e681c67fe22dfa456cca6c108e94e70d36363929227c17bb352cac6282028f3e8164b11609bb20ad756f643"),
    "Attention_v8{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=0}":
        (2614412, 5087496, 224100, "881930b20185c6ead14b5b25f3b383315371b79526e6d356ad1813add166397515e4f0d503f6d1fc5e6a29f33d4e4231af2a0a9ea24248f4767214a4b6e0f3f2"),
    "AttnBlock_v9{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2,CHECK=False}":
        (423389, 826819, 36229, "983b119dad8d4fa90b1eb2c2426dca506af42694a2b22b83009a7241b925b6fff0645d436669de29ceedec9a3c0b77845a30c1b8c5b82cccb219d95c65ad74ae"),
    "AttentionHead_v9{T=40,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2,MASKED_FROM=2}":
        (1134033, 2213423, 97301, "71d4680190e38a0736b8a2ab86b97c0cf501fc65ad5508c36bd6d57376250bbbbff785ceac056c7153e9fced9ba01a712d887d698d28ffcbfeda9f06f30d99cb"),
    "Attention_v9{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2,MASKED_FROM=1}":
        (2614160, 5087496, 223976, "d9481cd9ea4938b7941f85c0b993ec08241739f47a661f8257362f9d772c7075ca18f83c6390fae02c19470eec60bcc61bbcebf2d6362c16c7354c5f2e06f0e9"),
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
