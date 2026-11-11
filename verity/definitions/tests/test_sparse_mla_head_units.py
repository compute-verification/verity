"""The units a head's key block can be proved as, each held to `SparseMlaHeadBlock_v1` word for word (Gaussian rows, and zeros,
subnormals, infinities and NaNs planted in the queries, the keys and the carried state), at small statics and at GLM-5.3's
served ones.

The block cut at P: `SparseMlaHeadScores_v1`'s running max and lanes are the block's, and `SparseMlaHeadValues_v1` on its bf16(P)
and rescale, over the DV / G column blocks, is the block's O; a head run through such units block after block, each unit's
outputs the next one's state, then its end on each column block, is the head (`SparseMlaHead_v1`).

Several heads in one unit: `SparseMlaHeadsBlock_v1` is each head's own block on the one ``kv``; a row's heads run through such
units, then each head's end, are the row (`SparseMla_v1`).

The Boolean versions agree with their word views."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.batch import type_shape
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


def _cut_is_the_block(st: dict, NVIS: int, FIRST: bool, G: int, n: int, seed: int) -> None:
    DR, DV = st["DR"], st["DV"]
    block = bind(M.SparseMlaHeadBlock, NVIS=NVIS, FIRST=FIRST, **st)
    scores = bind(M.SparseMlaHeadScores, NVIS=NVIS, FIRST=FIRST, **st)
    qn, qr, kv, *state = _inputs(n, 1, NVIS, FIRST, DR, DV, seed)
    qn, qr, state = qn[:, 0], qr[:, 0], [x[:, 0] for x in state]
    m, lanes, o = _fields(E.evaluate_batch(block, qn, qr, kv, *state), [1, 4, DV])
    got = _fields(E.evaluate_batch(scores, qn, qr, kv, *state[:2]), [1, 4, NVIS] + [1] * (not FIRST))
    for name, g, w in (("running max", got[0], m), ("lanes", got[1], lanes)):
        bad = np.flatnonzero((g != w).any(axis=1))
        assert not bad.size, f"{scores.id}: its {name} differ from {block.id}'s in {bad.size} of {n}, first {bad[:5]}"
    pb = got[2].astype(np.uint16)
    for c in range(DV // G):
        unit = bind(M.SparseMlaHeadValues, NVIS=NVIS, FIRST=FIRST, G=G, C=c, **{k: v for k, v in st.items() if k != "SCALE"})
        carried = () if FIRST else (got[3][:, 0].astype(np.uint32), state[2][:, c * G:(c + 1) * G])
        cols = np.asarray(E.evaluate_batch(unit, pb, kv, *carried)).astype(np.uint64).reshape(n, G)
        bad = np.flatnonzero((cols != o[:, c * G:(c + 1) * G]).any(axis=1))
        assert not bad.size, f"{unit.id}: its columns differ from {block.id}'s O in {bad.size} of {n}, first {bad[:5]}"


@pytest.mark.parametrize("NVIS,FIRST,G", [(16, False, 8), (16, False, 16), (16, True, 4), (5, True, 4), (1, True, 2),
                                          (16, False, 1)])
def test_scores_then_values_on_each_column_block_are_the_head_block(NVIS: int, FIRST: bool, G: int) -> None:
    _cut_is_the_block(SMALL, NVIS, FIRST, G, 48, 100 * NVIS + 10 * G + FIRST)


@pytest.mark.parametrize("NVIS,FIRST,G", [(64, False, 128), (64, True, 256), (2, True, 128)])
def test_the_cut_at_the_served_statics(NVIS: int, FIRST: bool, G: int) -> None:
    _cut_is_the_block(SERVED, NVIS, FIRST, G, 4, 11 + NVIS + FIRST)


@pytest.mark.parametrize("T,G", [(37, 4), (16, 16), (33, 8), (5, 2)])
def test_a_head_through_its_scores_and_values_units_is_the_head(T: int, G: int) -> None:
    """A head's key blocks in FA3's order, each as its scores unit and its DV / G values units: the scores' max and lanes the
    next scores' state, its bf16(P) and rescale every values unit's, a values unit's columns the next block's same unit's;
    then the head's end on each column block with the last lanes."""
    DR, DV, BN = SMALL["DR"], SMALL["DV"], SMALL["BN"]
    rng = np.random.default_rng(T + 100 * G)
    qn, qr, kv = _bf16(rng, (1, DV))[0].tolist(), _bf16(rng, (1, DR))[0].tolist(), _bf16(rng, (T, DV + DR))
    want = evaluate(bind(M.SparseMlaHead, T=T, INV=GLM["INV"], **SMALL), qn, qr, kv.ravel().tolist())
    cols = {k: v for k, v in SMALL.items() if k != "SCALE"}
    state: list = []
    o: list = [None] * (DV // G)
    for j, (lo, hi) in enumerate(M.visits(T, BN)):
        nvis, first, keys = hi - lo, j == 0, kv[lo:hi].ravel().tolist()
        out = evaluate(bind(M.SparseMlaHeadScores, NVIS=nvis, FIRST=first, **SMALL), qn, qr, keys, *state)
        state, pb, rescale = [out[0], list(out[1:5])], list(out[5:5 + nvis]), out[5 + nvis:]
        for c in range(DV // G):
            unit = bind(M.SparseMlaHeadValues, NVIS=nvis, FIRST=first, G=G, C=c, **cols)
            o[c] = list(evaluate(unit, pb, keys, *(() if first else (rescale[0], o[c]))))
    end = bind(M.SparseMlaHeadEnd, DV=G, INV=GLM["INV"])
    assert [w for c in range(DV // G) for w in evaluate(end, state[1], o[c])] == list(want)


def test_the_cuts_units_take_what_they_read() -> None:
    def names(fn) -> list[str]:
        return [n for n, _ in fn.signature()[0]]

    cols = {k: v for k, v in SMALL.items() if k != "SCALE"}
    assert names(bind(M.SparseMlaHeadScores, NVIS=16, FIRST=False, **SMALL)) == ["q_nope", "q_rope", "kv", "m_prev", "l_prev"]
    assert names(bind(M.SparseMlaHeadScores, NVIS=5, FIRST=True, **SMALL)) == ["q_nope", "q_rope", "kv"]
    assert names(bind(M.SparseMlaHeadValues, NVIS=16, FIRST=False, G=4, C=1, **cols)) == ["pb", "kv", "rescale", "o_prev"]
    assert names(bind(M.SparseMlaHeadValues, NVIS=5, FIRST=True, G=4, C=1, **cols)) == ["pb", "kv"]
    for unit in (bind(M.SparseMlaHeadScores, NVIS=5, FIRST=False, **SMALL),
                 bind(M.SparseMlaHeadValues, NVIS=5, FIRST=False, G=4, C=0, **cols)):
        with pytest.raises(ValueError, match="only the first visited block is partial"):
            _ = unit.body
    for G, C in ((3, 0), (4, 4), (4, -1), (0, 0)):
        with pytest.raises(ValueError, match="are not a block of DV=16"):
            _ = bind(M.SparseMlaHeadValues, NVIS=16, FIRST=True, G=G, C=C, **cols).body


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


_BITS = dict(DR=16, DV=16, BN=16, DOT=BM.SPARSE_MLA_GLM53["DOT"])
AGREEMENT = [bind(BM.SparseMlaHeadScoresV2, NVIS=5, FIRST=True, SCALE=0.0625, **_BITS),
             bind(BM.SparseMlaHeadScoresV2, NVIS=16, FIRST=False, SCALE=0.0625, **_BITS),
             bind(BM.SparseMlaHeadValuesV2, NVIS=5, FIRST=True, G=4, C=3, **_BITS),
             bind(BM.SparseMlaHeadValuesV2, NVIS=16, FIRST=False, G=8, C=1, **_BITS),
             bind(BM.SparseMlaHeadsBlockV2, NHB=2, NVIS=5, FIRST=True, SCALE=0.0625, **_BITS),
             bind(BM.SparseMlaHeadsBlockV2, NHB=2, NVIS=16, FIRST=False, SCALE=0.0625, **_BITS)]


def _operands(fn, n: int, seed: int) -> list[np.ndarray]:
    """``n`` instances of each of ``fn``'s parameters: Gaussian bf16 rows and f32 words, specials planted in half of them."""
    rng = np.random.default_rng(seed)
    out = []
    for _name, ty in fn.signature()[0]:
        shape = (n, *type_shape(ty))
        if set(ty.leaf_widths()) == {16}:
            out.append(_planted(rng, _bf16(rng, shape), BF16_SPECIALS, 0.02))
        else:
            out.append(_planted(rng, rng.standard_normal(shape).astype(np.float32).view(np.uint32), F32_SPECIALS, 0.1))
    return out


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    """The six units trace some millions of ANDs between them (minutes): what the word-level tests can't see is each Boolean
    unit's agreement with its word view on special words through the padding and the carried state, that a values unit reads
    its own columns of the keys, and that two heads share one ``kv``."""
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k == "DOT" else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
    assert SW.boolean_version(word).id == target.id
    n = 64
    inputs = _operands(target, n, sum(map(ord, target.id)))
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64).reshape(n, -1)
    want = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64).reshape(n, -1)
    assert got.shape == want.shape
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {n} instances differ from {word.id}, first {bad[:5]}"
