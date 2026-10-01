"""`boolean_attention`: the v6 chain (`AttnBlock_v6` / `AttentionHead_v6` / `Attention_v6`), the v7 chain on both of its
profiles (FA2 on mma.sync, FA3 on wgmma) and `Fa3InvSum_v2` are Boolean, agree with their word views (the v3 and v2 chains
and `Fa3InvSum_v1`) on random words and on words that reach every guard (a zero query, signed zeros, subnormals, infinities,
NaNs, scores that overflow), and are pinned by gate count and program digest."""
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
        (207368, 403631, 20499, "c1a54c0306b6cbee96127cac5e03778a1610558f50b5fe7dbf46e0b67dc5946d"),
    "AttnBlock_v6{D=16,NVIS=16,FIRST=False,BN=16}":
        (435868, 854502, 39047, "7e7dc2cd84c65554314bba82f33d37190e4000651b149945352f0112036dfc5f"),
    "AttentionHead_v6{T=20,D=16,BN=16}":
        (673779, 1315230, 60680, "891b3b7ff3ca4f4825dc567b89dbee7b8c91918e8ec090ec880ce704c67b8034"),
    "Attention_v6{T=20,NH=4,KVH=2,D=16,BN=16}":
        (2695116, 5260920, 242720, "681f5aba87f876ab212b2c3512594ec43bbe746b260383c169f4bc6c1b9ce367"),
    "Fa3InvSum_v2": (947, 5695, 71, "c0ce2f7ff45cfa8da67b76e115100b9b34b691166dbb6c20689f2d6162b447bb"),
    "AttnBlock_v7{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2}":
        (423452, 833126, 35879, "d30dd74fb75f08920c50293524f595e2dc3a34320a2ed97843b80c67b7397b9c"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=AmpereBF16TcDot16_v3,INV=Fa2InvSum_v2}":
        (673779, 1315230, 60680, "718f7224157266ab29f9d28d61b4a74fcf2931a0c722830910d2205463fda2eb"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (653603, 1280494, 55525, "ff67ed4f41fab2779a96c2235e6c93e4754751b6604977838b4be8bfc0d800fd"),
    "Attention_v7{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (2614412, 5121976, 222100, "adb4aaa6d94261c4ef500e51635be25fa747476de1171523e08f4869775398a8"),
}

FA2, FA3 = BAt.ATTENTION_FA2, BAt.ATTENTION_FA3
PINNED = [bind(BAt.AttnBlockV6, D=16, NVIS=5, FIRST=True, BN=16), bind(BAt.AttnBlockV6, D=16, NVIS=16, FIRST=False, BN=16),
          bind(BAt.AttentionHeadV6, T=20, D=16, BN=16), bind(BAt.AttentionV6, T=20, NH=4, KVH=2, D=16, BN=16), bind(BAt.Fa3InvSum),
          bind(BAt.AttnBlockV7, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"]), bind(BAt.AttentionHeadV7, T=20, D=16, BN=16, **FA2),
          bind(BAt.AttentionHeadV7, T=20, D=16, BN=16, **FA3), bind(BAt.AttentionV7, T=20, NH=4, KVH=2, D=16, BN=16, **FA3)]

#: one block per mask shape and per state; one and two blocks, a partial last block, BN = 32; GQA with two KV heads; v7 on both
#: profiles
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
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 3 if target.defn.version == 6 else 2, lifted)
    inputs = _inputs(word, 96, sum(map(ord, target.id)))
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64)
    ref = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64)
    bad = np.flatnonzero((got != ref).reshape(len(got), -1).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} instances differ from {word.id}, first {bad[:5]}"
