"""`verity.ml.circuits.ops.argmax` and `ops.boolean.argmax`: a row's chunk units and their combine are `TokenSelect_v1`'s greedy
argmax (here in numpy float32: the first index on ties, a NaN never taken, a NaN logits[0] never left), on random rows and on
rows sampled from a few words (ties across chunks, +0 against -0, -inf, ±inf and NaNs), at the demo's V = 256 in chunks of 128 and
on ragged rows; each Boolean unit computes its word Definition, the combine on any partials, in range or not; each unit's output
is one 16-bit word; and `bind` refuses statics outside the family."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import argmax as A
from verity.ml.circuits.ops.boolean import argmax as BA

#: signed zeros, the smallest subnormal, 1.0 and its neighbour, the largest finite, the infinities and two NaNs, each sign
SPECIALS = np.array([m | s for m in (0x0000, 0x0001, 0x3F80, 0x3F81, 0x7F7F, 0x7F80, 0x7FC0, 0x7F81) for s in (0, 0x8000)],
                    np.uint16)
FEW = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x4000, 0xFF80, 0x7F80, 0x7FC0], np.uint16)
SHAPES = [(256, 128), (200, 64), (7, 3), (5, 5), (4, 1)]


def _f32(xb: np.ndarray) -> np.ndarray:
    return (xb.astype(np.uint32) << 16).view(np.float32)


def _scan(rows: np.ndarray, best: np.ndarray, first: int) -> tuple[np.ndarray, np.ndarray]:
    """``TokenSelect_v1``'s scan from (best, 0) over columns ``first ..`` of the rows: (the best word, its column)."""
    v = _f32(rows)
    bv, at = _f32(best), np.zeros(len(rows), np.uint32)
    with np.errstate(invalid="ignore"):
        for j in range(first, rows.shape[1]):
            take = v[:, j] > bv
            best, bv, at = np.where(take, rows[:, j], best), np.where(take, v[:, j], bv), np.where(take, j, at)
    return best.astype(np.uint16), at


def token_select(rows: np.ndarray) -> np.ndarray:
    """The row's greedy argmax: from (logits[0], 0)."""
    return _scan(rows, rows[:, 0].copy(), 1)[1]


def chunk(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A chunk's (best, idx): from (-inf, 0)."""
    return _scan(rows, np.full(len(rows), A.NEG_INF_BF16, np.uint16), 0)


def rows(V: int, rng, n: int = 96) -> np.ndarray:
    """Random logits at two scales, uniform words, special words, and rows of a few words; then a row of -inf, one of NaNs, one
    whose maximum is logits[0] and recurs in every chunk, and one whose logits[0] is a NaN beside +inf."""
    normal = [(np.asarray(rng.normal(0, s, (n, V)), np.float32).view(np.uint32) >> 16).astype(np.uint16) for s in (1.0, 30.0)]
    words = [rng.integers(0, 1 << 16, (n, V)).astype(np.uint16), rng.choice(SPECIALS, (n, V)), rng.choice(FEW, (n, V))]
    edge = np.full((4, V), A.NEG_INF_BF16, np.uint16)
    edge[1] = 0x7FC0
    edge[2] = rng.choice(np.array([0x3F80, 0x0000, 0x8000], np.uint16), V)
    edge[2, ::max(1, V // 3)] = 0x4000
    edge[3] = 0x3F80
    edge[3, 0], edge[3, -1] = 0x7FC0, 0x7F80
    return np.concatenate(normal + words + [edge])


def _ev(target, *args) -> np.ndarray:
    """A word Definition's outputs (`evaluate_batch`), or a Boolean one's through its word view (`evaluate_words`)."""
    ev = BS.evaluate_words if target.defn.id.endswith("_v2") else E.evaluate_batch
    return np.asarray(ev(target, *args)).reshape(-1).astype(np.uint16)


def _parts(x: np.ndarray, V: int, C: int, m) -> tuple[np.ndarray, np.ndarray]:
    """Each V-chunk's (best, idx) by module ``m``'s chunk Definitions, each bound at its chunk's width."""
    parts = [x[:, lo:lo + C] for lo in range(0, V, C)]
    best = [_ev(bind(m.ArgmaxChunkBest, C=p.shape[1]), p) for p in parts]
    idx = [_ev(bind(m.ArgmaxChunkIdx, C=p.shape[1]), p) for p in parts]
    return np.stack(best, axis=1), np.stack(idx, axis=1)


@pytest.mark.parametrize("V,C", SHAPES)
def test_the_word_chunks_and_combine_are_token_select(V: int, C: int) -> None:
    x = rows(V, np.random.default_rng(V * 7 + C))
    best, idx = _parts(x, V, C, A)
    for c, lo in enumerate(range(0, V, C)):
        want_best, want_idx = chunk(x[:, lo:lo + C])
        assert np.array_equal(best[:, c], want_best) and np.array_equal(idx[:, c], want_idx), c
    got = _ev(bind(A.ArgmaxCombine, V=V, C=C), x[:, 0], best, idx)
    assert np.array_equal(got, token_select(x)), np.flatnonzero(got != token_select(x))[:5]


@pytest.mark.parametrize("V,C", SHAPES)
def test_each_boolean_unit_computes_its_word_definition(V: int, C: int) -> None:
    """The chunks on the rows above, and the combine on them and on uniform partials (NaN bests, indices past C)."""
    rng = np.random.default_rng(V * 11 + C)
    x = rows(V, rng)
    best, idx = _parts(x, V, C, BA)
    want_best, want_idx = _parts(x, V, C, A)
    assert np.array_equal(best, want_best) and np.array_equal(idx, want_idx)
    n = best.shape[1]
    x0 = np.concatenate([x[:, 0], rng.integers(0, 1 << 16, len(x)).astype(np.uint16)])
    best = np.concatenate([best, rng.choice(SPECIALS, (len(x), n))])
    idx = np.concatenate([idx, rng.integers(0, 1 << 16, (len(x), n)).astype(np.uint16)])
    got = _ev(bind(BA.ArgmaxCombine, V=V, C=C), x0, best, idx)
    assert np.array_equal(got, _ev(bind(A.ArgmaxCombine, V=V, C=C), x0, best, idx))
    assert np.array_equal(got[:len(x)], token_select(x))


def test_the_index_words_select_and_add_wrap_at_16_bits() -> None:
    rng = np.random.default_rng(5)
    a, b = (rng.integers(0, 1 << 16, 4096).astype(np.uint16) for _ in range(2))
    a[:2], b[:2] = 0xFFFF, (1, 0xFFFF)
    c = rng.integers(0, 2, 4096).astype(np.uint8)
    add = (a.astype(np.uint32) + b) & 0xFFFF
    sel = np.where(c == 1, a, b)
    for defn, want, args in ((A.U16Add, add, (a, b)), (A.SelectU16, sel, (c, a, b))):
        assert np.array_equal(np.asarray(E.evaluate_batch(defn, *args)).reshape(-1), want)
    for defn, want, args in ((BA.U16Add, add, (a, b)), (BA.SelectU16, sel, (c, a, b))):
        assert np.array_equal(np.asarray(BS.evaluate_words(defn, *args)).reshape(-1), want)


def test_each_unit_outputs_one_16_bit_word() -> None:
    for d in (A, BA):
        for t in (bind(d.ArgmaxChunkBest, C=128), bind(d.ArgmaxChunkIdx, C=128), bind(d.ArgmaxCombine, V=256, C=128)):
            assert sum(t.signature()[1].leaf_widths()) == 16, t.id


@pytest.mark.parametrize("c", [0, (1 << 16) + 1, "128"])
def test_bind_refuses_a_chunk_width_outside_the_family(c) -> None:
    for m in (A, BA):
        for target in (lambda: bind(m.ArgmaxChunkBest, C=c), lambda: bind(m.ArgmaxChunkIdx, C=c),
                       lambda: bind(m.ArgmaxCombine, V=256, C=c)):
            with pytest.raises(ValueError, match="refuses"):
                target()


@pytest.mark.parametrize("v", [100, (1 << 16) + 1, "256"])
def test_bind_refuses_a_row_outside_the_family(v) -> None:
    for m in (A, BA):
        with pytest.raises(ValueError, match="refuses"):
            bind(m.ArgmaxCombine, V=v, C=128)
