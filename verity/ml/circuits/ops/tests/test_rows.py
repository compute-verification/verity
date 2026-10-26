"""`verity.ml.circuits.ops.rows` and `ops.boolean.rows`: the row units. The residual add is `Bf16AddF2fp_v1` per element on
every edge word; the argmax is the scan's first maximum (0 for a NaN logits[0], else the least index of the greatest non-NaN
logit, +0 equal to -0); and each Boolean form agrees with its word Definition (`word_of`) and its reference."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import fp32 as F
from verity.ml.circuits.ops import rows as W
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import rows as R

#: +-0, the least and greatest subnormals, the least normal, 1, the next word, 2, the greatest finite, +-inf, the F2FP and
#: torch NaNs, a signalling NaN and a negative NaN
EDGES = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x3F80, 0xBF80, 0x3F81, 0x4000, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80,
         0x7FFF, 0x7FC0, 0x7F81, 0xFFC0]
NAN = [w for w in EDGES if w & 0x7F80 == 0x7F80 and w & 0x7F]


def _words(rows) -> np.ndarray:
    return np.asarray(rows, np.uint64).astype(np.uint16)


def _agree(target, *inputs: np.ndarray) -> np.ndarray:
    """The Boolean target's words on ``inputs``, asserted equal to its word view's."""
    got = BS.evaluate_words(target, *inputs).astype(np.uint64)
    ref = E.evaluate_batch(TR.word_of(target), *inputs).astype(np.uint64)
    n = len(inputs[0])
    assert (got.reshape(n, -1) == ref.reshape(n, -1)).all(), target.id
    return ref.reshape(n, -1)


# ---- the residual add ----------------------------------------------------------------------------------------------------

def test_the_add_row_is_bf16addf2fp_per_element_on_every_edge_pair() -> None:
    a = [x for x in EDGES for _ in EDGES]
    b = [y for _ in EDGES for y in EDGES]
    want = W.add_row(a, b)
    assert want == [F.Bf16AddF2fp.evaluate(x, y) for x, y in zip(a, b)]
    N = len(EDGES)
    got = _agree(bind(R.Bf16AddRow, N=N), _words(a).reshape(N, N), _words(b).reshape(N, N))
    assert [int(w) for w in got.ravel()] == want
    pairs = dict(zip(zip(a, b), want))
    assert all(pairs[(x, y)] == 0x7FFF for x in NAN for y in EDGES)
    assert pairs[(0x7F80, 0xFF80)] == 0x7FFF and pairs[(0x7F7F, 0x7F7F)] == 0x7F80
    assert pairs[(0x0000, 0x8000)] == 0x0000 and pairs[(0x8000, 0x8000)] == 0x8000
    assert pairs[(0x0001, 0x0001)] == 0x0002 and pairs[(0x007F, 0x0001)] == 0x0080
    assert pairs[(0x3F80, 0x3F80)] == 0x4000 and pairs[(0x3F81, 0xBF80)] == 0x3C00


def test_the_add_row_on_bits_agrees_with_its_word_view_on_sampled_rows() -> None:
    rng = np.random.default_rng(20261025)
    for N in (1, 5):
        a, b = (rng.integers(0, 1 << 16, size=(256, N)).astype(np.uint16) for _ in range(2))
        got = _agree(bind(R.Bf16AddRow, N=N), a, b)
        assert [[int(w) for w in r] for r in got] == [W.add_row(list(map(int, x)), list(map(int, y))) for x, y in zip(a, b)]


def test_the_add_row_is_its_elements_adds() -> None:
    one = BS.compile(EW.Bf16AddF2fp).counts["and"]
    assert one == 717
    for N in (1, 3, 8):
        assert BS.compile(bind(R.Bf16AddRow, N=N)).counts["and"] == N * one
    params, ret = bind(R.Bf16AddRow, N=16384).signature()
    assert [x.leaves for _, x in params] == [262_144, 262_144] and ret.leaves == 262_144


# ---- the argmax ----------------------------------------------------------------------------------------------------------

def _value(w: int) -> float:
    return float(np.asarray([w << 16], np.uint32).view(np.float32)[0])


def _first_greatest(logits) -> int:
    """0 for a NaN logits[0], else the least index of the greatest non-NaN logit."""
    xs = [_value(w) for w in logits]
    if np.isnan(xs[0]):
        return 0
    top = max(x for x in xs if not np.isnan(x))
    return next(k for k, x in enumerate(xs) if x == top)


@pytest.mark.parametrize("logits, want", [
    ([0x3F80, 0x3F80], 0), ([0x0000, 0x8000], 0), ([0x8000, 0x0000], 0), ([0x7FFF, 0x7F80, 0x4000], 0),
    ([0x3F80, 0x7FC0, 0x4000], 2), ([0x3F80, 0x7FC0, 0x3F80], 0), ([0xFF80, 0xFF80, 0xFF80], 0), ([0xFF80, 0xFFC0, 0x3F80], 2),
    ([0x3F80, 0x7F80], 1), ([0x0001, 0x0002], 1), ([0xBF80, 0xC000], 0), ([0xC000, 0xBF80], 1), ([0x7F81, 0x7FFF], 0),
])
def test_the_argmax_takes_the_first_maximum_and_never_a_nan(logits, want) -> None:
    assert W.argmax(logits) == want == _first_greatest(logits)
    got = _agree(bind(R.Argmax, V=len(logits)), _words([logits]))
    assert int(got[0, 0]) == want


def test_the_argmax_on_bits_agrees_with_its_word_view_and_reference_over_the_byte_vocabulary() -> None:
    """V = 256 rows of uniform words, of a few values (ties everywhere), and of each with NaNs injected, logits[0] among them."""
    rng = np.random.default_rng(20261026)
    V, n = 256, 96
    rows = rng.integers(0, 1 << 16, size=(n, V)).astype(np.uint16)
    rows[n // 3:2 * n // 3] = rng.choice(np.asarray([0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80], np.uint16),
                                         size=(n // 3, V))
    nan = rng.random((n, V)) < 0.05
    nan[2 * n // 3:] |= rng.random((n - 2 * n // 3, V)) < 0.3
    nan[::7, 0] = True
    rows[nan] = rng.choice(np.asarray(NAN, np.uint16), size=int(nan.sum()))
    got = _agree(bind(R.Argmax, V=V), rows)
    want = [W.argmax([int(w) for w in r]) for r in rows]
    assert [int(g) for g in got[:, 0]] == want == [_first_greatest([int(w) for w in r]) for r in rows]
    assert len(set(want)) > 20 and want.count(0) >= n // 7


def test_the_argmax_costs_the_pinned_ands_and_outputs_16_bits() -> None:
    t = bind(R.Argmax, V=256)
    assert BS.compile(t).counts["and"] == 15_557
    assert t.signature()[1].leaves == 16 and bind(W.Argmax, V=256).signature()[1] == W.INDEX


def test_malformed_bindings_are_refused() -> None:
    for D in (W.Argmax, R.Argmax):
        for V in (0, 1, (1 << 16) + 1, True):
            with pytest.raises(ValueError, match="2 to 65536 logits"):
                bind(D, V=V)
    for D in (W.Bf16AddRow, R.Bf16AddRow):
        for N in (0, -1, "2"):
            with pytest.raises(ValueError, match="positive int of elements"):
                bind(D, N=N)
