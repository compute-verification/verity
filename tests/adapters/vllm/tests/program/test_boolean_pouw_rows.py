"""`pouw_rows`' quantizer and dot product on bits (`registry.boolean_pouw_rows`) are the Boolean versions a lookup finds for their
words and compute them: the max on special and random words, the clamp on every exponent's mantissa edges and halves and on
random words near the clamp (every word: `calibration/f32_exhaustive/run.py boolean NcpRneClampI7_v1`), and the dot product on
every int7 pair in one lane and on extreme and random rows."""
from __future__ import annotations

import itertools

import numpy as np
from verity.circuits.evaluation import bits as BS
from verity.circuits.ops.boolean import versions as SW

from verity_vllm.definitions.registry import boolean_pouw_rows as BP
from verity_vllm.definitions.registry import pouw_rows as PR

SPECIAL = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F000000, 0xBF000000, 0x3F800000,
           0x3FC00000, 0x40200000, 0x427C0000, 0x427E0000, 0x427DFFFF, 0x427E0001, 0x42800000, 0xC27E0000, 0xC27DFFFF,
           0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF]


def _on_bits(word, bits, *cols) -> list[int]:
    assert SW.boolean_version(word).defn is bits
    return np.asarray(BS.evaluate_words(SW.boolean_version(word), *cols)).reshape(-1).tolist()


def test_the_max_on_bits_is_its_word_on_special_and_random_words() -> None:
    rng = np.random.default_rng(1)
    pairs = list(itertools.product(SPECIAL, SPECIAL))
    pairs += [tuple(int(v) for v in rng.integers(0, 1 << 32, 2, dtype=np.uint64)) for _ in range(2000)]
    m, x = (np.array(c, np.uint64).reshape(-1, 1) for c in zip(*pairs))
    assert _on_bits(PR.NcpAbsMax, BP.NcpAbsMax, m, x) == [PR.NcpAbsMax.evaluate(*p) for p in pairs]


def test_the_clamp_on_bits_is_its_word_at_every_exponents_edges_and_near_the_clamp() -> None:
    rng = np.random.default_rng(2)
    mantissas = [0, 1, 2, 0x1FFFFF, 0x200000, 0x200001, 0x3FFFFF, 0x400000, 0x400001, 0x5FFFFF, 0x600000, 0x7FFFFE, 0x7FFFFF]
    words = [s << 31 | e << 23 | m for s in (0, 1) for e in range(256) for m in mantissas] + SPECIAL
    near = rng.integers(120, 136, 20000, dtype=np.uint64) << np.uint64(23) | rng.integers(0, 1 << 23, 20000, dtype=np.uint64)
    words += [int(w) | int(s) << 31 for w, s in zip(near, rng.integers(0, 2, 20000))]
    words += [int(v) for v in rng.integers(0, 1 << 32, 2000, dtype=np.uint64)]
    got = _on_bits(PR.NcpRneClampI7, BP.NcpRneClampI7, np.array(words, np.uint64).reshape(-1, 1))
    assert got == [PR.NcpRneClampI7.evaluate(w) for w in words]


def test_the_dot_product_on_bits_is_its_word_on_every_pair_in_a_lane_and_on_extreme_and_random_rows() -> None:
    rng = np.random.default_rng(3)
    n = 1 << 14
    a, b = rng.integers(0, 128, (n, 16), dtype=np.uint64), rng.integers(0, 128, (n, 16), dtype=np.uint64)
    a[:, 5], b[:, 5] = np.arange(n, dtype=np.uint64) >> np.uint64(7), np.arange(n, dtype=np.uint64) & np.uint64(127)
    extreme = np.array([0x40, 0x3F, 0x41, 0x7F, 0x00], np.uint64)
    a = np.concatenate([a, np.repeat(extreme, 5)[:, None].repeat(16, 1)])
    b = np.concatenate([b, np.tile(extreme, 5)[:, None].repeat(16, 1)])
    acc = np.concatenate([rng.integers(0, 1 << 32, n, dtype=np.uint64), np.array([0, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 1] * 5,
                                                                                       np.uint64)])
    got = _on_bits(PR.NcpDot16I7, BP.NcpDot16I7, acc.reshape(-1, 1), a, b)
    assert got == [PR.NcpDot16I7.evaluate(int(acc[i]), *map(int, a[i]), *map(int, b[i])) for i in range(len(acc))]
