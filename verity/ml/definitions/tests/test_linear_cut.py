"""`registry.linear_cut`: a cut linear's A row is x's BF16 words widened to FP32 exactly, at every R, on every encoding (NaNs,
infinities, subnormals and both zeros among them); its y unit hands y on unchanged; each σ is its word Definition on bits;
and a binding that names no table or no row of x is refused by name."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.definitions.registry import linear_cut as LC

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
    got = np.asarray(evaluate_batch(bind(LC.LinearARow, K=K, R=R, ROUND=0), x.astype(np.uint16))).reshape(4, K)
    assert np.array_equal(got.astype(np.uint64), x << 16)
    bits = evaluate_batch(bind(LC.LinearARowV2, K=K, R=R, ROUND=3), _bits(x, 16))
    assert np.array_equal(_words(bits, 4, 32), x << 16)


def test_the_y_unit_hands_y_on():
    y = _bf16(3, 12, 7)
    assert np.array_equal(np.asarray(evaluate_batch(bind(LC.LinearOut, N=12, K=64, ROUND=1), y.astype(np.uint16))), y)
    assert np.array_equal(_words(evaluate_batch(bind(LC.LinearOutV2, N=12, K=64, ROUND=1), _bits(y, 16)), 3, 16), y)


def test_a_round_or_k_is_its_own_template():
    """ROUND and K key the table a unit reads, so units of two tables are two templates however alike their gates."""
    ids = {bind(LC.LinearOut, N=8, K=k, ROUND=s).id for k in (64, 128) for s in (0, 1)}
    assert len(ids) == 4
    assert bind(LC.LinearARow, K=8, R=1, ROUND=0).id != bind(LC.LinearARow, K=8, R=1, ROUND=1).id


@pytest.mark.parametrize("fn,bad,why", [
    (LC.LinearARow, {"K": 8, "R": 3, "ROUND": 0}, "R divides K"),
    (LC.LinearARow, {"K": 8, "R": 2, "ROUND": -1}, "a round is a natural number"),
    (LC.LinearOut, {"N": 0, "K": 8, "ROUND": 0}, "positive integers"),
    (LC.LinearOut, {"N": 4, "K": 8, "ROUND": True}, "a round is a natural number"),
    (LC.LinearOutV2, {"N": 4, "K": 0, "ROUND": 0}, "positive integers"),
])
def test_a_binding_that_names_no_table_or_row_is_refused(fn, bad, why):
    with pytest.raises(ValueError, match=why):
        bind(fn, **bad)
