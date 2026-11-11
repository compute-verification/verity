"""A head's key block cut at P, each unit held to `SparseMlaHeadBlock_v1` word for word (Gaussian rows, and zeros, subnormals,
infinities and NaNs planted in the queries, the keys and the carried state), at small statics and at GLM-5.3's served ones:
`SparseMlaHeadScores_v1`'s running max and lanes are the block's, and `SparseMlaHeadValues_v1` on its bf16(P) and rescale, over
the DV / G column blocks, each on its own columns of the keys' kv_c, is the block's O; a head run through such units block
after block, each unit's outputs the next one's state, then its end on each column block, is the head (`SparseMlaHead_v1`).

The Boolean versions agree with their word views, and at the served statics each is one Flock block, read off its Calls."""
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


def _inputs(n: int, NVIS: int, FIRST: bool, DR: int, DV: int, seed: int) -> list[np.ndarray]:
    """``n`` instances of a head block's operands: the query's two parts, the block's keys and, past the first visited block,
    the carried max, lanes and O."""
    rng = np.random.default_rng(seed)
    out = [_planted(rng, _bf16(rng, shape), BF16_SPECIALS, 0.02) for shape in ((n, DV), (n, DR), (n, NVIS, DV + DR))]
    if not FIRST:
        m = rng.standard_normal(n).astype(np.float32).view(np.uint32)
        lanes = np.abs(rng.standard_normal((n, 4))).astype(np.float32).view(np.uint32)
        o = rng.standard_normal((n, DV)).astype(np.float32).view(np.uint32)
        out += [_planted(rng, x, F32_SPECIALS, 0.1) for x in (m, lanes, o)]
    return out


def _fields(words, widths: list[int]) -> list[np.ndarray]:
    """An evaluated batch's (n, leaves) words as its return fields, each (n, width)."""
    words = np.asarray(words).astype(np.uint64)
    assert words.shape[1] == sum(widths)
    return np.split(words, np.cumsum(widths)[:-1], axis=1)


def _cut_is_the_block(st: dict, NVIS: int, FIRST: bool, G: int, n: int, seed: int) -> None:
    DR, DV = st["DR"], st["DV"]
    block = bind(M.SparseMlaHeadBlock, NVIS=NVIS, FIRST=FIRST, **st)
    scores = bind(M.SparseMlaHeadScores, NVIS=NVIS, FIRST=FIRST, **st)
    qn, qr, kv, *state = _inputs(n, NVIS, FIRST, DR, DV, seed)
    m, lanes, o = _fields(E.evaluate_batch(block, qn, qr, kv, *state), [1, 4, DV])
    got = _fields(E.evaluate_batch(scores, qn, qr, kv, *state[:2]), [1, 4, NVIS] + [1] * (not FIRST))
    for name, g, w in (("running max", got[0], m), ("lanes", got[1], lanes)):
        bad = np.flatnonzero((g != w).any(axis=1))
        assert not bad.size, f"{scores.id}: its {name} differ from {block.id}'s in {bad.size} of {n}, first {bad[:5]}"
    pb = got[2].astype(np.uint16)
    unit = bind(M.SparseMlaHeadValues, NVIS=NVIS, FIRST=FIRST, BN=st["BN"], DOT=st["DOT"], G=G)
    for c in range(DV // G):
        at = slice(c * G, (c + 1) * G)  # a key row is [kv_c | k_pe]: the unit takes these words of each key, and no other
        carried = () if FIRST else (got[3][:, 0].astype(np.uint32), state[2][:, at])
        cols = np.asarray(E.evaluate_batch(unit, pb, np.ascontiguousarray(kv[:, :, at]), *carried)).astype(np.uint64).reshape(n, G)
        bad = np.flatnonzero((cols != o[:, at]).any(axis=1))
        assert not bad.size, (f"{unit.id} on columns {c * G} to {(c + 1) * G}: differs from {block.id}'s O in {bad.size} of {n}, "
                              f"first {bad[:5]}")


@pytest.mark.parametrize("NVIS,FIRST,G", [(16, False, 8), (16, False, 16), (16, True, 4), (5, True, 4), (1, True, 2),
                                          (16, False, 1)])
def test_scores_then_values_on_each_column_block_are_the_head_block(NVIS: int, FIRST: bool, G: int) -> None:
    _cut_is_the_block(SMALL, NVIS, FIRST, G, 48, 100 * NVIS + 10 * G + FIRST)


@pytest.mark.parametrize("NVIS,FIRST,G", [(64, False, 128), (64, True, 256), (2, True, 32)])
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
    state: list = []
    o: list = [None] * (DV // G)
    for j, (lo, hi) in enumerate(M.visits(T, BN)):
        nvis, first, keys = hi - lo, j == 0, kv[lo:hi].ravel().tolist()
        out = evaluate(bind(M.SparseMlaHeadScores, NVIS=nvis, FIRST=first, **SMALL), qn, qr, keys, *state)
        state, pb, rescale = [out[0], list(out[1:5])], list(out[5:5 + nvis]), out[5 + nvis:]
        unit = bind(M.SparseMlaHeadValues, NVIS=nvis, FIRST=first, BN=BN, DOT=SMALL["DOT"], G=G)
        for c in range(DV // G):
            own = kv[lo:hi, c * G:(c + 1) * G].ravel().tolist()
            o[c] = list(evaluate(unit, pb, own, *(() if first else (rescale[0], o[c]))))
    end = bind(M.SparseMlaHeadEnd, DV=G, INV=GLM["INV"])
    assert [w for c in range(DV // G) for w in evaluate(end, state[1], o[c])] == list(want)


def test_the_cuts_units_take_what_they_read() -> None:
    def params(fn) -> list[tuple[str, tuple[int, ...]]]:
        return [(n, type_shape(ty)) for n, ty in fn.signature()[0]]

    cols = dict(BN=16, DOT=SMALL["DOT"])
    assert params(bind(M.SparseMlaHeadScores, NVIS=16, FIRST=False, **SMALL)) == [
        ("q_nope", (16,)), ("q_rope", (16,)), ("kv", (16, 32)), ("m_prev", ()), ("l_prev", (4,))]
    assert params(bind(M.SparseMlaHeadScores, NVIS=5, FIRST=True, **SMALL)) == [
        ("q_nope", (16,)), ("q_rope", (16,)), ("kv", (5, 32))]
    # a values unit takes G words of each key, the columns of kv_c it reads, not the key rows
    assert params(bind(M.SparseMlaHeadValues, NVIS=16, FIRST=False, G=4, **cols)) == [
        ("pb", (16,)), ("kv", (16, 4)), ("rescale", ()), ("o_prev", (4,))]
    assert params(bind(M.SparseMlaHeadValues, NVIS=5, FIRST=True, G=4, **cols)) == [("pb", (5,)), ("kv", (5, 4))]
    for unit in (bind(M.SparseMlaHeadScores, NVIS=5, FIRST=False, **SMALL),
                 bind(M.SparseMlaHeadValues, NVIS=5, FIRST=False, G=4, **cols)):
        with pytest.raises(ValueError, match="only the first visited block is partial"):
            _ = unit.body
    for NVIS, BN, G in ((16, 16, 0), (17, 16, 4), (0, 16, 4), (8, 24, 4)):
        with pytest.raises(ValueError, match="SparseMlaHeadValues: BN="):
            _ = bind(M.SparseMlaHeadValues, NVIS=NVIS, FIRST=True, BN=BN, DOT=SMALL["DOT"], G=G).body


_BITS = dict(DR=16, DV=16, BN=16, DOT=BM.SPARSE_MLA_GLM53["DOT"])
_COLS = dict(BN=16, DOT=BM.SPARSE_MLA_GLM53["DOT"])
AGREEMENT = [bind(BM.SparseMlaHeadScoresV2, NVIS=5, FIRST=True, SCALE=0.0625, **_BITS),
             bind(BM.SparseMlaHeadScoresV2, NVIS=16, FIRST=False, SCALE=0.0625, **_BITS),
             bind(BM.SparseMlaHeadValuesV2, NVIS=5, FIRST=True, G=4, **_COLS),
             bind(BM.SparseMlaHeadValuesV2, NVIS=16, FIRST=True, G=16, **_COLS),
             bind(BM.SparseMlaHeadValuesV2, NVIS=16, FIRST=False, G=8, **_COLS)]


def _operands(fn, n: int, seed: int) -> list[np.ndarray]:
    """``n`` instances of each parameter of the word Definition ``fn``: Gaussian bf16 rows and f32 words, specials planted in
    half of them."""
    rng = np.random.default_rng(seed)
    out = []
    for _name, ty in fn.signature()[0]:
        shape = (n, *type_shape(ty))
        if set(ty.leaf_widths()) == {16}:
            out.append(_planted(rng, _bf16(rng, shape), BF16_SPECIALS, 0.02))
        else:
            out.append(_planted(rng, rng.standard_normal(shape).astype(np.float32).view(np.uint32), F32_SPECIALS, 0.1))
    return out


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    """What the word-level tests can't see: each Boolean unit's agreement with its word view on special words through the
    padding and the carried state, a values unit reading its own columns of the keys (in whole steps by their own gates,
    in a partial step beside zero rows), and the two written over the block's Calls."""
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k == "DOT" else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
    assert SW.boolean_version(word).id == target.id
    n = 64
    inputs = _operands(word, n, sum(map(ord, target.id)))
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64).reshape(n, -1)
    want = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64).reshape(n, -1)
    assert got.shape == want.shape
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {n} instances differ from {word.id}, first {bad[:5]}"


def test_the_served_units_are_one_flock_block_each() -> None:
    """At GLM-5.3's statics, read off the Boolean units' Calls: the scores of a 64-key block take the block's own 2^27; a
    values unit, which takes only its own columns of the keys, takes a quarter of that block on half of O's columns, and half
    as much again each time its columns halve, down to 32."""
    from verity_flock import circuit as FC

    S = {k: BM.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}
    block = FC.unit_block(bind(BM.SparseMlaHeadBlockV3, NVIS=64, FIRST=False, **S))
    for first in (False, True):
        scores = FC.unit_block(bind(BM.SparseMlaHeadScoresV2, NVIS=64, FIRST=first, **S))
        assert FC.past_block(scores) is None and scores["k_log"] == block["k_log"] == 27
        for g, k_log in ((256, 25 if first else 26), (128, 25), (64, 24), (32, 23)):
            values = FC.unit_block(bind(BM.SparseMlaHeadValuesV2, NVIS=64, FIRST=first, BN=S["BN"], DOT=S["DOT"], G=g))
            assert FC.past_block(values) is None and values["k_log"] == k_log, (first, g, values)
