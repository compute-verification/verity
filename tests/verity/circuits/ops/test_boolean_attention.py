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
    "F32AddFtz_v3": (888, 1409, 115, "4d7178b9592838e5963a059af872ff128caf957f4ce8c40b64c4871524811c4829cc2fe06e62eaa3c28123bc2d7a9bbbe0e3a5e7e9d82269ae2744ad83112aa1"),
    "F32SubFtz_v3": (952, 1473, 135, "500e8ee268661d24025ac3fb7f71b18ae0332fd14b7bc8a8d98db4aeb5e70b6a7319c96803be4fd2645fcc198a460850a61e4af70cb54e6ae0d4fde36f85558d"),
    "F32MulFtz_v3": (2506, 4664, 110, "c1aaab188dc81b0f14802ee806cdb72c541402128133da4ef6efd6176bc4e20458be955682cbe233ff1f9c129a4540ac009e54cdd92fc3b876b92e6b4476d6d6"),
    "F32FmaFtz_v3": (4584, 7877, 267, "b50afa885e1512ee104a2b7535846ebe27965cf2d0c1628d95700f5ed8839886882f7c684378b7cc42f2431b36da87c37854b66e6ade913afa0f1da9dfcba80a"),
    "F32FmaSubFtz_v3": (4584, 7877, 268, "3f8d6182d654da428f4e44b7333e381938c0ee58daa0d13d9e5a0d5d467b45767ad52c262dba91a6adf243e722d3dcc5b5d1e1e6dd6e3be62d742cd259d366c3"),
    "F32Max_v2": (237, 408, 70, "6ddefc0b97d536fcfee42267e5f2984d73918ee73e1aa4a229cfe3908ad8894d59d5bb2c9cb79e2cb2d072cc8274ded0a2c48a5d592d2fcf10a5886e9546d510"),
    "GuardNegInfZero_v2": (63, 0, 24, "19e9a9b0f68d142c2886f271d9ed1b57e9e0388df2826e0f9d105286f0464ddf62c194894c686736529d92d7f8714b2333fef6b9e2d0ac5522861086e805255f"),
    "Fa2InvSum_v2": (947, 4866, 109, "f4714f36a41c9a23919cfb9f2c5e03ca89a80fc42f47ad4d6c0d30c6e82ec9eaa6fe918f5728548f7502df534193d9d76a27e2133009a9b7319b8e671f9b02f5"),
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

