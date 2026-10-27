"""`b1.TokenSelectChunk_v1{C}` and `b1.TokenSelectCombine_v1{V,C}`: greedy token selection cut along V, each chunk's partial (its
first maximum above -inf and that index in the chunk) reduced with logits[0] to `TokenSelect_v1{V}`'s token.  The composition
is the row's Definition word for word: at random widths and cuts over every bf16 word and over the words that decide an argmax
(NaNs of both signs and payloads, +-0, +-inf, ties), and at the served statics, GLM-5.3's V = 154880 in 32 chunks of 4840
(the fewest chunks of a whole width whose combine's index row passes one SHA-512 block, which C-Flock's circuit prover requires of a unit's rows)."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity_flock import ir_lower as IL
from verity_flock.gf2 import Circuit
from verity.ml.definitions.registry import b1

#: the words an argmax turns on: quiet NaNs of both signs, a signalling NaN, +0, -0, +inf, -inf, 1, -1, 2
EDGES = np.array([0x7FC0, 0xFFC0, 0x7F81, 0x0000, 0x8000, 0x7F80, 0xFF80, 0x3F80, 0xBF80, 0x4000], dtype=np.uint16)
SERVED_V, SERVED_C = 154880, 4840


def _ev(fn, *xs) -> np.ndarray:
    """``fn`` on rows of words, one row per leading index of each ``xs``."""
    return np.asarray(evaluate_batch(fn, *(np.asarray(x).astype(np.uint64) for x in xs)))


def _composed(x: np.ndarray, C: int) -> np.ndarray:
    """The token of each row of ``x`` by its V-chunks of C and their combine."""
    n, V = x.shape
    best, idx = [], []
    for lo, hi in b1.chunks(V, C):
        o = _ev(bind(b1.TokenSelectChunk, C=hi - lo), x[:, lo:hi]).reshape(n, 2)
        best.append(o[:, 0])
        idx.append(o[:, 1])
    return _ev(bind(b1.TokenSelectCombine, V=V, C=C), x[:, :1], np.stack(best, 1), np.stack(idx, 1)).reshape(n)


def _token_select(x: np.ndarray) -> np.ndarray:
    return _ev(bind(b1.TokenSelect, V=x.shape[1]), x).reshape(x.shape[0])


@pytest.mark.parametrize("seed", range(6))
def test_the_chunks_and_their_combine_are_token_select(seed: int) -> None:
    """At V from 2 (`TokenSelect_v1{V=1}` scans no step) and every cut, over random words and over the edge words, so NaNs
    lead rows and chunks, ties straddle chunk boundaries and a chunk holds no number above -inf."""
    rng = np.random.default_rng(seed)
    for V in (2, 3, 7, 16, 33, 64):
        for C in sorted({c for c in (1, 2, 3, V // 2, V - 1, V) if 1 <= c <= V}):
            x = np.concatenate([rng.choice(EDGES, (24, V)), rng.integers(0, 1 << 16, (8, V)).astype(np.uint16)])
            assert np.array_equal(_composed(x, C), _token_select(x)), (V, C)


def test_a_nan_first_logit_is_the_token_and_a_later_nan_never_is() -> None:
    nan, one, two = 0x7FC0, 0x3F80, 0x4000
    x = np.array([[nan, two, one, two], [one, nan, two, nan], [0xFF80, 0xFF80, nan, 0xFF80], [0x8000, 0x0000, nan, 0x8000]],
                 dtype=np.uint16)
    assert _token_select(x).tolist() == [0, 2, 0, 0]
    for C in (1, 2, 3):
        assert _composed(x, C).tolist() == [0, 2, 0, 0]


def test_the_chunk_step_is_argmax_step() -> None:
    """`TokenSelectChunkStep_v1` is `ArgmaxStep_v1` word for word: every pair of edge words as (x, best), at indices past the
    best's and before it, and random words."""
    rng = np.random.default_rng(3)
    x, best = (a.ravel() for a in np.meshgrid(EDGES, EDGES))
    x, best = np.concatenate([x, rng.integers(0, 1 << 16, 512)]), np.concatenate([best, rng.integers(0, 1 << 16, 512)])
    n = len(x)
    carry = np.stack([best, rng.integers(0, 1 << 31, n), rng.integers(0, 1 << 31, n)], 1)
    assert np.array_equal(_ev(bind(b1.TokenSelectChunkStep), carry, x[:, None]), _ev(bind(b1.ArgmaxStep), carry, x[:, None]))


def _flat(C: int) -> tuple[int, int, int]:
    """``TokenSelectChunk_v1{C}``'s flat lowering (`ir_lower`): its ANDs, their operands' bits (the AND rows' nonzeros) and the
    widest operand."""
    Cc = Circuit()
    IL.lower(Cc, bind(b1.TokenSelectChunk, C=C), [Cc.input(f"x{j}", 16) for j in range(C)])
    ops = [len(f.s) for k, d in zip(Cc.kind, Cc.data) if k == "and" for f in d]
    return len(ops) // 2, sum(ops), max(ops)


def test_the_chunk_lowers_linearly_in_its_width() -> None:
    """Each step's ANDs read the running (best, index) as forms of a few bits at every width, so the steps 130 .. 249, past the
    first at which the index's 8th bit is a form, each add the same ANDs and nonzeros, and no operand widens (at C = 4840 a
    step that kept the running best's form, as `ArgmaxStep_v1` does, built a 14 GB circuit in 35 min at 63 GB)."""
    (a1, z1, w1), (a2, z2, w2), (a3, z3, w3) = _flat(130), _flat(190), _flat(250)
    assert w1 == w2 == w3
    assert a3 - a2 == a2 - a1 > 0 and z3 - z2 == z2 - z1 > 0


def test_a_chunk_of_no_number_above_minus_inf_is_minus_inf_at_zero() -> None:
    x = np.array([[0xFF80, 0x7FC0, 0xFF80], [0x7FC0, 0xFFC0, 0x7F81], [0xFF80, 0x3F80, 0x3F80]], dtype=np.uint16)
    assert _ev(bind(b1.TokenSelectChunk, C=3), x).reshape(3, 2).tolist() == [[0xFF80, 0], [0xFF80, 0], [0x3F80, 1]]


@pytest.mark.slow
def test_the_served_row_is_its_chunks_and_their_combine() -> None:
    """GLM-5.3's logits row in 32 chunks of 4840: the combine's index row is 1024 bits, past one SHA-512 block (32 chunks is
    the fewest of a whole width: 28 is the least for that row, and V = 2^8 * 5 * 11^2)."""
    rng = np.random.default_rng(16)
    x = (rng.standard_normal((2, SERVED_V)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    x[1, rng.integers(1, SERVED_V, 64)] = 0x7FC0
    x[1, [SERVED_C - 1, SERVED_C, 7 * SERVED_C + 5]] = 0x7F00
    assert len(b1.chunks(SERVED_V, SERVED_C)) == 32
    got, want = _composed(x, SERVED_C), _token_select(x)
    assert np.array_equal(got, want) and int(got[1]) == SERVED_C - 1


def test_a_cut_wider_than_the_row_or_empty_is_refused() -> None:
    for V, C in ((8, 0), (8, 9)):
        with pytest.raises(ValueError, match=f"V-chunks of C = {C} logits of a row of V = {V}"):
            bind(b1.TokenSelectCombine, V=V, C=C).signature()
