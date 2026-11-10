"""`verity.definitions.registry.boolean_scalar`'s Boolean versions of `ref_prims`' `I64ToI32_v1`, `F32MaxRef_v1` and
`F32Sqrt_v1`: each is Boolean, names its word primitive, and agrees with it on edge and random words (both zeros, the
infinities, NaNs with either sign and payload, subnormals, every binade's ends, the i32 range's edges); their gate counts are
pinned. `F32Sqrt_v2` is held on every one of the 2^32 words by `benchmarks/f32_exhaustive/run.py boolean F32Sqrt`."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean.trace import word_of
from verity.circuits.evaluation import bits as BS
from verity.definitions.boolean import boolean_version
from verity.definitions.registry import boolean_scalar as BR
from verity.definitions.registry import ref_prims as RP

#: id -> (And, Xor, Not gates)
PINS = {"I64ToI32_v2": (63, 94, 3), "F32MaxRef_v2": (167, 371, 46), "F32Sqrt_v2": (967, 1714, 145)}
M32 = (1 << 32) - 1
#: both zeros, the smallest and largest subnormals and normals, ±1, the infinities, NaNs (quiet, signalling, both signs, all ones)
F32_EDGES = [0, 1 << 31, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
             0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, M32]


def _binades() -> list[int]:
    """Each exponent field's first and last word, both signs."""
    return [s | e << 23 | m for s in (0, 1 << 31) for e in range(256) for m in (0, 0x7FFFFF)]


def _agree(d, *cols) -> int:
    got = BS.evaluate_words(d, *[np.asarray(c, dtype=np.uint64).reshape(-1, 1) for c in cols]).reshape(-1).tolist()
    word = word_of(d)
    want = [word.evaluate(*map(int, a)) for a in zip(*cols)]
    bad = [i for i in range(len(want)) if got[i] != want[i]]
    assert not bad, [(tuple(hex(int(c[i])) for c in cols), hex(got[i]), hex(want[i])) for i in bad[:5]]
    return len(want)


@pytest.mark.parametrize("d", [BR.I64ToI32, BR.F32MaxRef, BR.F32Sqrt], ids=lambda d: d.id)
def test_each_is_boolean_names_its_word_and_is_its_boolean_version(d) -> None:
    word = getattr(RP, d.name)
    assert word_of(d) is word and d.id == word.id.replace("_v1", "_v2")
    assert BL.is_boolean(d.bind()) and boolean_version(word).id == d.bind().id
    assert BS.compile(d.bind()).counts == dict(zip(("and", "xor", "not"), PINS[d.id]))


def test_i64_to_i32_keeps_the_low_word_inside_the_i32_range_and_is_the_sentinel_outside() -> None:
    edges = [v + d for v in (0, 1 << 31, -(1 << 31), 1 << 32, -(1 << 32), 1 << 62, -(1 << 63)) for d in (-1, 0, 1)]
    rng = np.random.default_rng(20261103)
    t = [e % (1 << 64) for e in edges] + list(rng.integers(0, 1 << 63, 4096, dtype=np.uint64) << np.uint64(1)) \
        + list(rng.integers(-(1 << 31), 1 << 31, 4096).astype(np.int64).view(np.uint64))
    assert _agree(BR.I64ToI32, np.array(t, dtype=np.uint64)) == len(t)


def test_f32_max_ref_on_every_pair_of_edges_neighbours_and_random_pairs() -> None:
    """Every ordered pair of edge and binade-end words (ties of +0 and -0 keep a), each word against its neighbours, and random
    pairs over all words."""
    rng = np.random.default_rng(20261104)
    w = np.array(F32_EDGES + _binades()[::7], dtype=np.uint64)
    a, b = np.repeat(w, len(w)), np.tile(w, len(w))
    r = rng.integers(0, 1 << 32, 1 << 14, dtype=np.uint64)
    near = (r + rng.integers(-2, 3, len(r)).astype(np.uint64)) & np.uint64(M32)
    _agree(BR.F32MaxRef, np.concatenate([a, r, r]), np.concatenate([b, near, rng.permutation(r)]))


def test_f32_sqrt_on_edges_binade_ends_and_dense_random_words() -> None:
    rng = np.random.default_rng(20261105)
    x = np.concatenate([np.array(F32_EDGES + _binades(), dtype=np.uint64), rng.integers(0, 1 << 32, 1 << 15, dtype=np.uint64),
                        rng.integers(1, 1 << 23, 1 << 12, dtype=np.uint64), rng.integers(0x3F000000, 0x40800000, 1 << 12, dtype=np.uint64)])
    _agree(BR.F32Sqrt, x)
