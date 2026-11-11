"""`sparse_mla.SparseMlaHeadsBlock_v1`: one key block of NHB heads of a row in one unit is each head's own
`SparseMlaHeadBlock_v1` on the one ``kv``, field by field and word for word (Gaussian rows, and zeros, subnormals, infinities and
NaNs planted in the queries, the keys and the carried state), at small statics and at GLM-5.3's served ones; a row's heads run
through such units block after block, each unit's outputs the next one's state port for port, then each head's end, are the row
(`SparseMla_v1`); the Boolean version agrees with its word view."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import evaluate
from verity.definitions import boolean as SW
from verity.definitions.registry import boolean_sparse_mla as BM
from verity.definitions.registry import sparse_mla as M

GLM = M.SPARSE_MLA_GLM53
SMALL = dict(DR=16, DV=16, BN=16, DOT=GLM["DOT"], SCALE=0.0625)
SERVED = {k: GLM[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}
#: bf16 words at the k16 step's guards: signed zeros, subnormals, the normal range's ends, infinities, NaNs
BF16_SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1], np.uint16)
#: f32 words a carried state can hold: zeros, the smallest subnormal and normal, the largest finite, infinities, NaNs
F32_SPECIALS = np.array([0x00000000, 0x80000000, 0x00000001, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
                         0xFFC00001], np.uint32)


def _bf16(rng, shape) -> np.ndarray:
    scale = np.exp2(rng.integers(-4, 5, (shape[0],) + (1,) * (len(shape) - 1))).astype(np.float32)
    return ((rng.standard_normal(shape).astype(np.float32) * scale).view(np.uint32) >> 16).astype(np.uint16)


def _planted(rng, a: np.ndarray, specials: np.ndarray, p: float) -> np.ndarray:
    """``a`` with specials at random places in the second half of its instances."""
    half = a[len(a) // 2:]
    at = rng.random(half.shape) < p
    half[at] = rng.choice(specials, int(at.sum()))
    return a


def _inputs(n: int, NHB: int, NVIS: int, FIRST: bool, DR: int, DV: int, seed: int) -> list[np.ndarray]:
    """``n`` instances of a heads block's operands, each (n, NHB, ...) but the keys, which are (n, NVIS, DV + DR)."""
    rng = np.random.default_rng(seed)
    out = [_planted(rng, _bf16(rng, shape), BF16_SPECIALS, 0.02) for shape in ((n, NHB, DV), (n, NHB, DR), (n, NVIS, DV + DR))]
    if not FIRST:
        m = rng.standard_normal((n, NHB)).astype(np.float32).view(np.uint32)
        lanes = np.abs(rng.standard_normal((n, NHB, 4))).astype(np.float32).view(np.uint32)
        o = rng.standard_normal((n, NHB, DV)).astype(np.float32).view(np.uint32)
        out += [_planted(rng, x, F32_SPECIALS, 0.1) for x in (m, lanes, o)]
    return out


def _fields(words, widths: list[int]) -> list[np.ndarray]:
    """An evaluated batch's (n, leaves) words as its return fields, each (n, width)."""
    words = np.asarray(words).astype(np.uint64)
    assert words.shape[1] == sum(widths)
    return np.split(words, np.cumsum(widths)[:-1], axis=1)


def _held_to_single_blocks(st: dict, NHB: int, NVIS: int, FIRST: bool, n: int, seed: int) -> None:
    many = bind(M.SparseMlaHeadsBlock, NHB=NHB, NVIS=NVIS, FIRST=FIRST, **st)
    one = bind(M.SparseMlaHeadBlock, NVIS=NVIS, FIRST=FIRST, **st)
    qn, qr, kv, *state = _inputs(n, NHB, NVIS, FIRST, st["DR"], st["DV"], seed)
    widths = [1, 4, st["DV"]]
    got = _fields(E.evaluate_batch(many, qn, qr, kv, *state), [NHB * w for w in widths])
    for h in range(NHB):
        want = _fields(E.evaluate_batch(one, qn[:, h], qr[:, h], kv, *(x[:, h] for x in state)), widths)
        for field, (g, w, width) in enumerate(zip(got, want, widths, strict=True)):
            bad = np.flatnonzero((g[:, h * width:(h + 1) * width] != w).any(axis=1))
            assert not bad.size, f"{many.id}: head {h}'s field {field} differs from {one.id} in {bad.size} of {n}, first {bad[:5]}"


@pytest.mark.parametrize("NHB,NVIS,FIRST", [(2, 16, False), (2, 16, True), (2, 5, True), (3, 16, False), (1, 16, False),
                                            (8, 1, True)])
def test_a_heads_block_is_each_heads_own_block(NHB: int, NVIS: int, FIRST: bool) -> None:
    _held_to_single_blocks(SMALL, NHB, NVIS, FIRST, 48, 1000 * NHB + 2 * NVIS + FIRST)


@pytest.mark.parametrize("NVIS,FIRST", [(64, False), (64, True), (2, True)])
def test_two_heads_of_a_block_at_the_served_statics(NVIS: int, FIRST: bool) -> None:
    _held_to_single_blocks(SERVED, 2, NVIS, FIRST, 4, 7 + NVIS + FIRST)


@pytest.mark.parametrize("T,NH,NHB", [(37, 4, 2), (16, 2, 2), (33, 3, 3), (5, 4, 2)])
def test_a_rows_heads_through_their_heads_blocks_are_the_row(T: int, NH: int, NHB: int) -> None:
    """The NH heads in groups of NHB, each group's key blocks in FA3's order as one unit each, a unit's three outputs the next
    one's ``m_prev``, ``l_prev`` and ``o_prev`` as they are; then each head's end on its entries of the last unit's lanes and O."""
    DR, DV, BN = SMALL["DR"], SMALL["DV"], SMALL["BN"]
    rng = np.random.default_rng(T + 10 * NH)
    qn, qr, kv = _bf16(rng, (NH, DV)), _bf16(rng, (NH, DR)), _bf16(rng, (T, DV + DR))
    row = bind(M.SparseMla, T=T, NH=NH, INV=GLM["INV"], **SMALL)
    want = np.asarray(evaluate(row, qn.ravel().tolist(), qr.ravel().tolist(), kv.ravel().tolist()), dtype=np.uint32)
    end = bind(M.SparseMlaHeadEnd, DV=DV, INV=GLM["INV"])
    got = []
    for h0 in range(0, NH, NHB):
        state: list = []
        for j, (lo, hi) in enumerate(M.visits(T, BN)):
            unit = bind(M.SparseMlaHeadsBlock, NHB=NHB, NVIS=hi - lo, FIRST=j == 0, **SMALL)
            out = evaluate(unit, qn[h0:h0 + NHB].ravel().tolist(), qr[h0:h0 + NHB].ravel().tolist(), kv[lo:hi].ravel().tolist(),
                           *state)
            state = [list(out[:NHB]), list(out[NHB:5 * NHB]), list(out[5 * NHB:])]
        for h in range(NHB):
            got += list(evaluate(end, state[1][4 * h:4 * h + 4], state[2][DV * h:DV * (h + 1)]))
    assert np.array_equal(np.asarray(got, dtype=np.uint32), want)


def test_a_heads_block_takes_its_blocks_statics() -> None:
    unit = bind(M.SparseMlaHeadsBlock, NHB=2, NVIS=16, FIRST=False, **SMALL)
    assert [n for n, _ in unit.signature()[0]] == ["q_nope", "q_rope", "kv", "m_prev", "l_prev", "o_prev"]
    first = bind(M.SparseMlaHeadsBlock, NHB=2, NVIS=16, FIRST=True, **SMALL)
    assert [n for n, _ in first.signature()[0]] == ["q_nope", "q_rope", "kv"]
    with pytest.raises(ValueError, match="NHB=0 heads"):
        _ = bind(M.SparseMlaHeadsBlock, NHB=0, NVIS=16, FIRST=True, **SMALL).body
    with pytest.raises(ValueError, match="only the first visited block is partial"):
        _ = bind(M.SparseMlaHeadsBlock, NHB=2, NVIS=5, FIRST=False, **SMALL).body


_BITS = dict(NHB=2, DR=16, DV=16, BN=16, SCALE=0.0625, DOT=BM.SPARSE_MLA_GLM53["DOT"])
AGREEMENT = [bind(BM.SparseMlaHeadsBlockV2, NVIS=5, FIRST=True, **_BITS),
             bind(BM.SparseMlaHeadsBlockV2, NVIS=16, FIRST=False, **_BITS)]


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    """Each unit traces most of a million ANDs (minutes): what the word-level tests can't see is the Boolean unit's agreement
    with its word view on special words through two heads' padding and carried state, and that the two share one ``kv``."""
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k == "DOT" else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
    assert SW.boolean_version(word).id == target.id
    S = target.bindings
    n = 64
    inputs = _inputs(n, S["NHB"], S["NVIS"], S["FIRST"], S["DR"], S["DV"], sum(map(ord, target.id)))
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64).reshape(n, -1)
    want = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64).reshape(n, -1)
    assert got.shape == want.shape
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {n} instances differ from {word.id}, first {bad[:5]}"
