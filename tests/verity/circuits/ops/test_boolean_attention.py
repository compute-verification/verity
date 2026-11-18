"""`verity.circuits.ops.boolean.attention`: the Boolean softmax operations agree with their word primitives (`trace.word_of`) on every
pair of special words and on random words (the inverse sum also on every mantissa), carry exactly their builders' live ANDs, and are pinned by gate count and program
digest."""
from __future__ import annotations

import itertools
import zlib

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.evaluation import bits as BS
from verity.circuits.defs import bind
from verity.circuits.ops.boolean import attention as BA
from verity.circuits.boolean import trace as TR

from _boolean_checks import assert_live_ands, assert_pinned

DEFS = [BA.F32AddFtz, BA.F32SubFtz, BA.F32MulFtz, BA.F32FmaFtz, BA.F32FmaSubFtz, BA.F32Max, BA.GuardNegInfZero, BA.Fa2InvSum]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "F32AddFtz_v3": (888, 1409, 115, "6249fa133e755d4ae055ea741935b6c04b30d5d1bf92dc0e9e876a36e65384f1fc7677e6349402bc0392203676f88794facaf943031c58a1e7e95b992ec1c70f"),
    "F32SubFtz_v3": (952, 1473, 135, "216630fd2aec02c0f8f14dc33c9638ea662e1631c98d7db5e184d4d4b6b55a14f65481a62bf4217e2a0ac6c7ea4cec60d56b2a16fabe498e1b366d83572b1f66"),
    "F32MulFtz_v3": (2506, 4664, 110, "4a409c7e4c410a7e0c5769a7454ab6b434b607a8cc0ccaf82c8082db9cf6e4267016f1b594c8ef6c2b49185a5540cf36f1e3ed297548e0b62c79bfe7c8ecb87f"),
    "F32FmaFtz_v3": (4584, 7877, 267, "63437b3498c1efb0ca92e071b48cbbf19cf994f2a4ab9de74bd79c88147fddade739e391b9f4fce219e78ca210527324ae833c2e1e36c63b90c8c8077b100ede"),
    "F32FmaSubFtz_v3": (4584, 7877, 268, "2a5f99b1c5692427ae080ecb8f721c0c773cc44c49bfc6b558dbaa1de700dfd8bf3aff3b7c70c73eaf19e2f14d68de2aef23a7ca148e103789cdc16a1268bf56"),
    "F32Max_v2": (237, 408, 70, "bbbfa10d3c83e75f521a1c3eeeb6ec1a82c8f1bda0442992896795de398c2315a769fa47dc9de45a6574bc8f6810d275a31625fd74d0f1d8b57a0eaacafdb872"),
    "GuardNegInfZero_v2": (63, 0, 24, "abd74e62250c08df082ead10b266bebec23c6e2501c3a7a72132f35ce62434c212c3bd392e80dde2f3e13db502c0e69b529b8726f9db1d300170363a0d6e4899"),
    "Fa2InvSum_v2": (947, 4866, 109, "abe5979ff7ca217138bcdf5ef26cb19f2adaa3efee04a43e5f48599b89085a0f8a8d9d49e6d74fa2396404dfc8bbd254c81df9a5ef8229fe72ddcdaac1a32cec"),
}

#: signed zeros, subnormals (the FTZ boundary), the normal range's ends, ones, infinities, NaNs of both signs and kinds, and
#: words near FA2's softmax values (log2(e) / sqrt(d) scales, exp2 arguments)
_MAGNITUDES = [0x00000000, 0x00000001, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, 0x3F7FFFFF, 0x3F800000, 0x3F800001,
               0x3FB8AA3B, 0x3E3504F3, 0x4B000000, 0x4B800000, 0x7E800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000,
               0x7FFFFFFF, 0x42FE0000, 0x33800000, 0x0C000000]
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x80000000)], np.uint32)


def _inputs(n_params: int, rng) -> list[np.ndarray]:
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=n_params)), np.int64).T
    rand = rng.integers(0, 1 << 32, (n_params, 60_000), dtype=np.uint64).astype(np.uint32)
    near = (rng.integers(0x3C000000, 0x44000000, (n_params, 20_000), dtype=np.uint64)
            | (rng.integers(0, 2, (n_params, 20_000), dtype=np.uint64) << np.uint64(31))).astype(np.uint32)
    return [np.concatenate([SPECIALS[g], r, x]) for g, r, x in zip(grid, rand, near)]


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_pinned_counts_and_digests(d) -> None:
    assert_pinned(bind(d), PINS)


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_the_ands_are_the_builders_live_ands(d) -> None:
    assert_live_ands(d, PINS)


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_agrees_with_its_word_primitive(d) -> None:
    target = bind(d)
    word = TR.word_of(target)
    assert (word.name, word.version) == (d.name, 1)
    inputs = _inputs(len(d.signature(None)[0]), np.random.default_rng(zlib.crc32(d.name.encode())))
    if d is BA.Fa2InvSum:
        inputs = [np.concatenate([inputs[0], np.arange(1 << 23, dtype=np.uint32) | np.uint32(0x3F800000)])]
    got = BS.evaluate_words(target, *inputs).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ from {word.id}, first {bad[:5]}"

