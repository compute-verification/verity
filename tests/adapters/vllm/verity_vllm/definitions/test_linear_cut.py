"""`registry.linear_cut`: a cut linear's A row is x's BF16 words widened to FP32 exactly, at every R, on every encoding (NaNs,
infinities, subnormals and both zeros among them); its y unit hands on each committed word's high half in rows of W; each σ
is its word Definition on bits; and a binding that names no row of x or no rows of y is refused by name."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity_vllm.definitions.registry import linear_cut as LC
from verity_vllm.definitions.registry.prims import BF16

SPECIALS = [0x0000, 0x8000, 0x0001, 0x807F, 0x3F80, 0xBF80, 0x7F7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81]


def _bf16(n: int, k: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).integers(0, 1 << 16, size=(n, k), dtype=np.uint64)
    x[0, :min(k, len(SPECIALS))] = SPECIALS[:k]
    return x


def _bits(words: np.ndarray, width: int) -> np.ndarray:
    return ((words[..., None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8).reshape(len(words), -1)


def _words(bits: np.ndarray, n: int, width: int) -> np.ndarray:
    return (np.asarray(bits).reshape(n, -1, width).astype(np.uint64) << np.arange(width, dtype=np.uint64)).sum(-1)


@pytest.mark.parametrize("K,R", [(16, 1), (16, 4), (64, 16)])
def test_the_a_row_is_x_widened_exactly(K, R):
    x = _bf16(4, K, K + R)
    got = np.asarray(evaluate_batch(bind(LC.LinearARow, K=K, R=R), x.astype(np.uint16))).reshape(4, K)
    assert np.array_equal(got.astype(np.uint64), x << 16)
    bits = evaluate_batch(bind(LC.LinearARowV4, K=K, R=R), _bits(x, 16))
    assert np.array_equal(_words(bits, 4, 32), x << 16)


@pytest.mark.parametrize("W", [4, 12])
def test_the_y_unit_hands_on_each_committed_words_high_half_in_rows_of_w(W):
    """A committed y word is its BF16 widened, and the unit's output is the BF16: each word's high half, whatever its low,
    as N / W return fields of W words in order (so each is an output row of its own)."""
    y = _bf16(3, 12, 7)
    low = np.random.default_rng(8).integers(0, 1 << 16, size=y.shape, dtype=np.uint64)
    fn = bind(LC.LinearOut, N=12, W=W)
    assert [(f.n, f.elem) for f in fn.signature()[1].fields] == [(W, BF16)] * (12 // W)
    for words in (y << 16, (y << 16) | low):
        got = np.asarray(evaluate_batch(fn, words.astype(np.uint32)))
        assert np.array_equal(got.astype(np.uint64).reshape(3, 12), y)
        bits = evaluate_batch(bind(LC.LinearOutV4, N=12, W=W), _bits(words, 32))
        assert np.array_equal(_words(bits, 3, 16), y)


@pytest.mark.parametrize("fn,bad,why", [
    (LC.LinearARow, {"K": 8, "R": 3}, "R divides K"),
    (LC.LinearOut, {"N": 0, "W": 4}, "N=0: .* N and W are positive integers"),
    (LC.LinearOut, {"N": 8, "W": 0}, "W=0: .* N and W are positive integers"),
    (LC.LinearOut, {"N": 12, "W": 5}, "N=12, W=5: .* so W divides N"),
    (LC.LinearOutV4, {"N": 4, "W": True}, "W=True: .* N and W are positive integers"),
])
def test_a_binding_that_names_no_row_is_refused(fn, bad, why):
    with pytest.raises(ValueError, match=why):
        bind(fn, **bad)
