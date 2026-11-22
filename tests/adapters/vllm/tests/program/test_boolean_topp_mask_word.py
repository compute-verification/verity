"""The top-p keep word on bits (`registry.boolean_topp_mask_word`): `TopPMaskWordx{V}_v2` is the Boolean version a lookup finds
for `TopPMaskWordx{V}_v1`, Boolean through every Call, and computes v1's keep bits (`sampling.topp_keep`) on bits: at V = 2 on
every row of logits from `ALPHABET` with every top_p of `PS` at every split count and four that no launch has, at V = 16
and 300 on random rows, and at the served vocabulary on random rows (opt-in: a recorded run)."""
from __future__ import annotations

import itertools
import os

import numpy as np
import pytest
from verity.circuits import boolean as BL
from verity.circuits.boolean.trace import word_of
from verity.circuits.evaluation import evaluate_bits
from verity.circuits.ops.boolean.versions import boolean_version

from verity_vllm.definitions.registry import boolean_topp_mask_word as BTM
from verity_vllm.definitions.registry import sampling as SM

f32 = np.float32
#: f32 logits: zeros, ties, a NaN of each sign, both infinities, the largest finite, a subnormal and an ordinary spread
ALPHABET = np.array([0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3F000000, 0x40400000, 0x7F800000, 0xFF800000,
                     0x7FC00000, 0xFFC00001, 0x7F7FFFFF, 0x00000001], np.uint32)
PS = np.array([0.0, 1e-30, 0.3, 0.5, 0.9, 0.99, 1.0, 1.5, np.nan, np.inf, -0.5], f32)
SPLITS = (*SM.SPLIT_COUNTS, 0, 3, 64, 0xFFFFFFFF)


def _bits(words: np.ndarray, w: int) -> np.ndarray:
    words = np.asarray(words, np.uint64)
    return ((words[..., None] >> np.arange(w, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


def _check(V: int, rows: list[tuple[np.ndarray, np.float32, int]]) -> None:
    body = boolean_version(SM.topp_mask_word(V))
    x = np.stack([r[0].view(np.uint32) for r in rows])
    p = np.array([r[1] for r in rows], f32).view(np.uint32)
    s = np.array([r[2] for r in rows], np.uint64)
    got = evaluate_bits(body, _bits(x, 32), _bits(p, 32), _bits(s, 32)).reshape(len(rows), V).astype(bool)
    for i, (xi, pi, si) in enumerate(rows):
        want = SM.topp_keep(xi.view(f32), float(pi), si)
        assert (got[i] == want).all(), (V, i, xi.view(np.uint32).tolist(), float(pi), si, np.flatnonzero(got[i] != want)[:8])


def _random_rows(V: int, n: int, seed: int) -> list[tuple[np.ndarray, np.float32, int]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x = (rng.standard_normal(V) * rng.choice([0.5, 3, 10])).astype(f32)
        if i % 5 == 1:
            x = np.round(x).astype(f32)
        if i % 7 == 2:
            x[rng.choice(V, 1)] = np.nan
        if i % 9 == 4:
            x[:] = 1.0
        rows.append((x, PS[i % len(PS)], SPLITS[i % len(SPLITS)]))
    return rows


def test_the_keep_words_boolean_version_is_the_family_member_and_boolean():
    body = boolean_version(SM.topp_mask_word(2))
    assert body.defn is BTM.topp_mask_word(2) and body.id == "TopPMaskWordx2_v2"
    assert word_of(body.defn) is SM.topp_mask_word(2) and BL.is_boolean(body)
    params, ret = SM.topp_mask_word(2).signature()
    assert body.signature() == (tuple((n, BL.bit_type(t)) for n, t in params), BL.bit_type(ret))


def test_on_bits_it_is_the_keep_word_at_every_split_count():
    _check(2, _random_rows(2, 2 * len(SPLITS), 1))


@pytest.mark.slow
def test_on_bits_it_is_the_keep_word_on_every_alphabet_row_at_v2():
    rows = [(np.array(xs, np.uint32).view(f32), p, s)
            for xs in itertools.product(ALPHABET, repeat=2) for p in PS for s in SPLITS]
    _check(2, rows)


@pytest.mark.slow
@pytest.mark.parametrize("V,n", [(16, 40), (300, 24)])
def test_on_bits_it_is_the_keep_word_on_random_rows(V, n):
    _check(V, _random_rows(V, n, V))


@pytest.mark.skipif(not os.environ.get("VERITY_VLLM_TOPP_SERVED"),
                    reason="VERITY_VLLM_TOPP_SERVED: about 3 h and 400 GB of bit-sliced evaluation, a recorded run")
def test_on_bits_it_is_the_keep_word_at_the_served_vocabulary():
    V = 154880
    rows = _random_rows(V, 4, V)
    _check(V, [(x, p, s) for (x, p, _), s in zip(rows, (32, 32, 1, 0))])
