"""The primitives against `docs/p3-instantiation.md` §1, recomputed from the spec's text where it gives byte formats."""

from __future__ import annotations

import hashlib
import random

import pytest

from verity.core.protocols.pous.oracle import Tweak, run
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.primitives import (CAPACITY_BITS, ArxButterfly, EvenMansourArx, Feistel10, FeistelP, FeistelPi,
                                    IdealPermutation, IdealTweakablePermutation, MiyaguchiPreneel, OverwriteChain,
                                    XorSponge, bit_diffusion_stages, diffusion_stages, key_program, key_query,
                                    key_rounds, Primitives, reference)

PARAMS = Params(16, 32, 12)
RNG = random.Random(7)


def test_feistel_round_is_the_spec():
    f = Feistel10(8, 0x50)
    x = bytes(range(8))
    t = b"T" * 35
    left, right = x[:4], x[4:]
    for i in range(1, 11):
        fi = hashlib.shake_256(bytes([0x50, i]) + t + right).digest(4)
        left, right = right, bytes(a ^ b for a, b in zip(left, fi))
    assert f.forward(int.from_bytes(x, "little"), t) == int.from_bytes(left + right, "little")


def test_permutations_invert():
    t = Tweak(RNG.getrandbits(256), 1, 3)
    for perm in (FeistelPi(PARAMS.m + CAPACITY_BITS), ArxButterfly(PARAMS.m + CAPACITY_BITS),
                 IdealPermutation(PARAMS.m + CAPACITY_BITS, b"s")):
        x = RNG.getrandbits(perm.width)
        assert perm.inverse(perm.forward(x)) == x and perm.forward(perm.inverse(x)) == x
    for perm in (FeistelP(PARAMS.m), EvenMansourArx(PARAMS.m), IdealTweakablePermutation(PARAMS.m, b"s")):
        x = RNG.getrandbits(perm.width)
        assert perm.inverse(t, perm.forward(t, x)) == x and perm.forward(t, perm.inverse(t, x)) == x


@pytest.mark.parametrize("perm", [FeistelP(PARAMS.m), EvenMansourArx(PARAMS.m)], ids=lambda p: p.name)
def test_tweak_reaches_every_call(perm):
    x = RNG.getrandbits(PARAMS.m)
    base = Tweak(5, 0, 7)
    outs = {perm.forward(t, x) for t in (base, Tweak(6, 0, 7), Tweak(5, 1, 7), Tweak(5, 0, 8), Tweak(5 | 1 << 200, 0, 7))}
    assert len(outs) == 5
    assert base.to_bytes() == (5).to_bytes(32, "little") + b"\x00" + (7).to_bytes(2, "little")


def _spec_inputs(v: int):
    mb = PARAMS.m // 8
    t, layer = RNG.getrandbits(256), 1
    labels = {u: RNG.getrandbits(PARAMS.m) for u in PARAMS.parents(v)}
    head = b"pous-p3-v1-H" + t.to_bytes(32, "little") + bytes([layer]) + v.to_bytes(2, "little") + bytes([len(labels)])
    parents = [labels[u].to_bytes(mb, "little") for u in sorted(labels, reverse=True)]
    return t, layer, labels, head, parents, b"\x01" + bytes(mb - 2) + b"\x80"


def test_overwrite_chain_is_the_spec():
    """H recomputed from the spec: h_0 the header zero-filled to m + 64 B, then (r, h) <- Π₂(B ‖ h) for the parents
    newest first and the pad, the key the pad call's r."""
    prims = reference(PARAMS)
    mb = PARAMS.m // 8
    assert prims.h.pi2.width == 2 * PARAMS.m + 512 and prims.h.pi2.core.nbytes == 2 * mb + 64
    for v in (0, 5, 13):
        t, layer, labels, head, parents, pad = _spec_inputs(v)
        h = head + bytes(mb + 64 - len(head))
        for b in parents + [pad]:
            out = prims.h.pi2.forward(int.from_bytes(b + h, "little")).to_bytes(2 * mb + 64, "little")
            r, h = out[:mb], out[mb:]
        key, cost = run(key_program(prims.h, key_query(PARAMS, t, layer, v, labels)), prims)
        assert key == int.from_bytes(r, "little")
        assert cost.depth == len(parents) + 1 and cost.kinds == {"PiQuery": len(parents) + 1}


@pytest.mark.parametrize("mode", [XorSponge, MiyaguchiPreneel])
def test_broken_modes_are_as_described(mode):
    """The broken modes, recomputed: zero IV, the header, the parents newest first and the pad as rate blocks."""
    h = mode(PARAMS, FeistelPi(PARAMS.m + CAPACITY_BITS))
    mb = PARAMS.m // 8
    t, layer, labels, head, parents, pad = _spec_inputs(13)
    state = 0
    for b in [head + bytes(mb - len(head))] + parents + [pad]:
        x = state ^ int.from_bytes(b, "little")
        state = h.pi.forward(x) ^ (x if mode is MiyaguchiPreneel else 0)
    key, cost = run(key_program(h, key_query(PARAMS, t, layer, 13, labels)), Primitives(h, FeistelP(PARAMS.m)))
    assert key == state & ((1 << PARAMS.m) - 1) and cost.depth == 14


def test_key_rounds():
    """13 calls with every parent held; an edge from the oldest parent leaves 3 calls per level: Π₂(oldest), Π₂(pad), P
    (instantiation §1)."""
    h = reference(PARAMS).h
    par = list(PARAMS.parents(13))
    assert key_rounds(h, par, {u: 0 for u in par}) == 13
    late_oldest = {u: 0 for u in par} | {1: 100}
    assert key_rounds(h, par, late_oldest) == 102
    late_newest = {u: 0 for u in par} | {12: 100}
    assert key_rounds(h, par, late_newest) == 113


def test_modes_reject_mismatched_widths():
    with pytest.raises(ValueError):
        OverwriteChain(PARAMS, FeistelPi(PARAMS.m + 512))
    with pytest.raises(ValueError):
        XorSponge(Params(4, 64, 1), FeistelPi(256 + 512))
    with pytest.raises(ValueError):
        XorSponge(PARAMS, FeistelPi(PARAMS.m + 256))


@pytest.mark.parametrize("words", [2, 3, 5, 8, 20, 33, 144])
def test_arx_matches_its_specification(words):
    perm = ArxButterfly(32 * words)
    for _ in range(3):
        x = RNG.getrandbits(32 * words)
        assert perm.forward(x) == perm.forward_words(x)
        assert perm.inverse(perm.forward(x)) == x


def test_arx_diffusion():
    """Twice the bit-level count, at least twice word-level + 12: 40 stages at 256 words, where the flip test finds
    the ideal never-flipped fraction from 16 stages (the unrotated §2c butterfly needed 20)."""
    assert diffusion_stages(1 << 14) == 14 and diffusion_stages(16390) == 29
    assert bit_diffusion_stages("arx-pi/heuristic-v2", 20) == max(
        __import__("verity.core.protocols.pous.primitives", fromlist=["_bit_reach"])._bit_reach("arx-pi/heuristic-v2", 20, b, 200)
        for b in range(640)), "the sampled bits find the exact count"
    assert ArxButterfly(32 * 256).stages == 40


def test_arx_rotates_word_zero():
    """Word 0 of the §2c butterfly was never rotated, so its MSB difference passed with probability 1."""
    perm = ArxButterfly(32 * 16)
    diffs = {perm.forward(x) ^ perm.forward(x ^ (1 << 31)) for x in (RNG.getrandbits(512) for _ in range(4))}
    assert len(diffs) == 4
    two = ArxButterfly(64)
    st = __import__("verity.core.protocols.pous.primitives", fromlist=["_schedule"])._schedule(two.name, 2, two.stages)[0]
    a, c = 0x80000000, 0
    rotl = lambda v, r: ((v << r) | (v >> (32 - r))) & 0xFFFFFFFF  # noqa: E731
    assert st.rotl(a + c, st.rot_a) == rotl(a, st.rot_a) != a


def test_ideal_is_a_consistent_permutation():
    perm = IdealPermutation(4, b"seed")
    images = [perm.forward(x) for x in range(16)]
    assert sorted(images) == list(range(16)) and all(perm.inverse(y) == x for x, y in enumerate(images))
