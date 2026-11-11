"""CIR-242 scratch, not for main: `SparseMlaHeadScoresCols_v1` (a key block's scores with its keys in column-block ports) held to
`SparseMlaHeadScores_v1` on the concatenation of its ports, word for word (Gaussian rows, specials planted in the queries, the
keys and the carried state), at small statics and at GLM-5.3's served ones; its Boolean version to its word view; the column
block writer `SparseMlaHeadKeys_v1` a copy; and the variant's Flock block at the served statics."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.definitions import boolean as SW
from verity.definitions.registry import boolean_sparse_mla as BM
from verity.definitions.registry import sparse_mla as M
from verity.definitions.tests.test_sparse_mla_head_units import (BF16_SPECIALS, SERVED, SMALL, _bf16, _fields, _inputs, _operands,
                                                                   _planted)

#: the toy of `benchmarks/one_stage/cir242_window.py`: DR = 32, DV = 256, column blocks of 32
TOY = dict(DR=32, DV=256, BN=16, DOT=SMALL["DOT"], SCALE=0.0625)


def _cols(kv: np.ndarray, DV: int, G: int) -> list[np.ndarray]:
    """A batch of key blocks (n, NVIS, DV + DR) as the variant's ports: kv_c's DV / G column blocks, then k_pe."""
    return [np.ascontiguousarray(kv[:, :, c * G:(c + 1) * G]) for c in range(DV // G)] + [np.ascontiguousarray(kv[:, :, DV:])]


def _same(st: dict, NVIS: int, FIRST: bool, G: int, n: int, seed: int) -> None:
    DR, DV = st["DR"], st["DV"]
    scores = bind(M.SparseMlaHeadScores, NVIS=NVIS, FIRST=FIRST, **st)
    cols = bind(M.SparseMlaHeadScoresCols, NVIS=NVIS, FIRST=FIRST, G=G, **st)
    qn, qr, kv, *state = _inputs(n, NVIS, FIRST, DR, DV, seed)
    widths = [1, 4, NVIS] + [1] * (not FIRST)
    want = _fields(E.evaluate_batch(scores, qn, qr, kv, *state[:2]), widths)
    got = _fields(E.evaluate_batch(cols, qn, qr, *_cols(kv, DV, G), *state[:2]), widths)
    for name, g, w in zip(("running max", "lanes", "bf16(P)", "rescale"), got, want):
        bad = np.flatnonzero((g != w).any(axis=1))
        assert not bad.size, f"{cols.id}: its {name} differ from {scores.id}'s in {bad.size} of {n}, first {bad[:5]}"


@pytest.mark.parametrize("st,NVIS,FIRST,G,n", [(SMALL, 16, False, 16, 48), (SMALL, 5, True, 8, 48), (SMALL, 16, True, 4, 48),
                                               (TOY, 16, False, 32, 24), (TOY, 8, True, 32, 24)])
def test_the_variant_is_the_scores_unit_on_its_ports_side_by_side(st, NVIS, FIRST, G, n) -> None:
    _same(st, NVIS, FIRST, G, n, 7 * NVIS + G + FIRST)


@pytest.mark.parametrize("NVIS,FIRST", [(64, False), (64, True), (16, True)])
def test_the_variant_at_the_served_statics(NVIS, FIRST) -> None:
    _same(SERVED, NVIS, FIRST, SERVED["DR"], 4, 31 + NVIS + FIRST)


_BITS = dict(DR=16, DV=32, BN=16, DOT=BM.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)


@pytest.mark.parametrize("target", [bind(BM.SparseMlaHeadScoresColsV2, NVIS=5, FIRST=True, G=16, **_BITS),
                                    bind(BM.SparseMlaHeadScoresColsV2, NVIS=16, FIRST=False, G=16, **_BITS),
                                    bind(BM.SparseMlaHeadKeysV2, NVIS=16, W=16)], ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    assert SW.boolean_version(word).id == target.id
    n = 32
    inputs = _operands(word, n, sum(map(ord, target.id)))
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64).reshape(n, -1)
    want = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64).reshape(n, -1)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert got.shape == want.shape and not bad.size, f"{target.id}: {bad.size} of {n} instances differ from {word.id}"


def test_the_writer_is_a_copy() -> None:
    kv = _planted(np.random.default_rng(5), _bf16(np.random.default_rng(6), (8, 16, 32)), BF16_SPECIALS, 0.05)
    got = np.asarray(E.evaluate_batch(bind(M.SparseMlaHeadKeys, NVIS=16, W=32), kv)).reshape(kv.shape)
    assert np.array_equal(got, kv)


def test_the_variant_and_the_writer_at_the_served_statics_as_flock_blocks() -> None:
    """Read off the Boolean units' Calls: the variant takes the scores unit's block, 2^27."""
    from verity_flock import circuit as FC

    S = {k: BM.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}
    for first in (False, True):
        cols = FC.unit_block(bind(BM.SparseMlaHeadScoresColsV2, NVIS=64, FIRST=first, G=S["DR"], **S))
        scores = FC.unit_block(bind(BM.SparseMlaHeadScoresV2, NVIS=64, FIRST=first, **S))
        print({"first": first, "cols": cols, "scores": scores})
        assert FC.past_block(cols) is None and cols["k_log"] == scores["k_log"] == 27
