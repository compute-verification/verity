"""A key block's rows read from one committed row per column block: `SparseMlaHeadScoresCols_v1` (a key block's scores, its key
rows given in column blocks of G columns) held to `SparseMlaHeadScores_v1` on its blocks side by side, word for word (Gaussian
rows, specials planted in the queries, the keys and the carried state), at small statics and at GLM-5.3's served ones; the
column block's writer `SparseMlaHeadKeysWire_v1` the identity, at no AND, whose flat unit a returned bit other than its input's
leaves unsatisfied; the Boolean versions to their word views; and at the served statics each one Flock block."""
from __future__ import annotations

import re

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

#: DR = 32, DV = 64: key rows of three column blocks of 32, or six of 16
WIDE = dict(DR=32, DV=64, BN=16, DOT=SMALL["DOT"], SCALE=0.0625)


def _cols(kv: np.ndarray, G: int) -> list[np.ndarray]:
    """A batch of key blocks (n, NVIS, DV + DR) as the variant's ports: their rows' column blocks of G columns, in order."""
    return [np.ascontiguousarray(kv[:, :, c * G:(c + 1) * G]) for c in range(kv.shape[2] // G)]


def _same(st: dict, NVIS: int, FIRST: bool, G: int, n: int, seed: int) -> None:
    DR, DV = st["DR"], st["DV"]
    scores = bind(M.SparseMlaHeadScores, NVIS=NVIS, FIRST=FIRST, **st)
    cols = bind(M.SparseMlaHeadScoresCols, NVIS=NVIS, FIRST=FIRST, G=G, **st)
    qn, qr, kv, *state = _inputs(n, NVIS, FIRST, DR, DV, seed)
    widths = [1, 4, NVIS] + [1] * (not FIRST)
    want = _fields(E.evaluate_batch(scores, qn, qr, kv, *state[:2]), widths)
    got = _fields(E.evaluate_batch(cols, qn, qr, *_cols(kv, G), *state[:2]), widths)
    for name, g, w in zip(("running max", "lanes", "bf16(P)", "rescale"), got, want):
        bad = np.flatnonzero((g != w).any(axis=1))
        assert not bad.size, f"{cols.id}: its {name} differ from {scores.id}'s in {bad.size} of {n}, first {bad[:5]}"


@pytest.mark.parametrize("st,NVIS,FIRST,G", [(SMALL, 16, False, 16), (SMALL, 5, True, 8), (SMALL, 16, True, 4),
                                             (WIDE, 16, False, 32), (WIDE, 8, True, 32), (WIDE, 16, False, 16)])
def test_the_scores_on_column_blocks_are_the_scores_unit_on_them_side_by_side(st, NVIS, FIRST, G) -> None:
    _same(st, NVIS, FIRST, G, 48, 7 * NVIS + G + FIRST)


@pytest.mark.slow
@pytest.mark.parametrize("NVIS,FIRST", [(64, False), (64, True), (16, True)])
def test_the_scores_on_column_blocks_at_the_served_statics(NVIS, FIRST) -> None:
    """The served key rows' nine column blocks of 64 at GLM-5.3's statics, which the small statics' blocks don't reach."""
    _same(SERVED, NVIS, FIRST, 64, 4, 31 + NVIS + FIRST)


def test_column_blocks_that_do_not_tile_both_parts_of_a_key_row_are_refused() -> None:
    """A column block is G columns of kv_c or of k_pe, never both: G must divide DR as well as DV."""
    assert [n for n, _t in bind(M.SparseMlaHeadScoresCols, NVIS=5, FIRST=True, G=16, **WIDE).signature()[0]] == [
        "q_nope", "q_rope", "kv0", "kv1", "kv2", "kv3", "kv4", "kv5"]
    for G in (0, 64, 24):
        with pytest.raises(ValueError, match=f"not whole column blocks of G={G}"):
            bind(M.SparseMlaHeadScoresCols, NVIS=5, FIRST=True, G=G, **WIDE).signature()


_BITS = dict(DR=16, DV=32, BN=16, DOT=BM.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)


@pytest.mark.parametrize("target", [bind(BM.SparseMlaHeadScoresColsV2, NVIS=5, FIRST=True, G=16, **_BITS),
                                    bind(BM.SparseMlaHeadScoresColsV2, NVIS=16, FIRST=False, G=16, **_BITS),
                                    bind(BM.SparseMlaHeadKeysWireV2, NVIS=16, W=16, C=2)],
                         ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    assert SW.boolean_version(word).id == target.id
    n = 32
    inputs = _operands(word, n, sum(map(ord, target.id)))
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64).reshape(n, -1)
    want = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64).reshape(n, -1)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert got.shape == want.shape and not bad.size, f"{target.id}: {bad.size} of {n} instances differ from {word.id}"


def test_the_writer_returns_its_input_at_no_and_and_a_return_other_than_its_input_is_refused() -> None:
    """The writer is the identity on its words (word and Boolean), its gates its 2 NVIS W primitive calls (one a word on
    bits), none an AND, the same gates on every column block it names; and its flat unit (`class_statement.unit_lowering`, as `UnitStager` stages it), honest, satisfies every
    row, while a returned bit flipped after witness generation (`ir_lower.evaluate_layout`'s ``flip``) leaves its lane
    unsatisfied: a writer whose return is not its input is refused by its Definition."""
    from verity_flock import class_statement as CL
    from verity_flock import ir_lower as IL

    kv = _planted(np.random.default_rng(5), _bf16(np.random.default_rng(6), (8, 16, 32)), BF16_SPECIALS, 0.05)
    word, boo = bind(M.SparseMlaHeadKeysWire, NVIS=16, W=32, C=0), bind(BM.SparseMlaHeadKeysWireV2, NVIS=16, W=32, C=0)
    assert np.array_equal(np.asarray(E.evaluate_batch(word, kv)).reshape(kv.shape), kv)
    assert np.array_equal(np.asarray(BS.evaluate_words(boo, kv)).astype(kv.dtype).reshape(kv.shape), kv)
    assert (word.gates, boo.gates) == (2 * 16 * 32, 16 * 32)
    assert bind(BM.SparseMlaHeadKeysWireV2, NVIS=16, W=32, C=5).body.gates == boo.body.gates
    for d in (M.SparseMlaHeadKeysWire, BM.SparseMlaHeadKeysWireV2):
        with pytest.raises(ValueError, match=re.escape("SparseMlaHeadKeysWire: NVIS=2 and W=16 must be >= 1 and C=-1 >= 0")):
            bind(d, NVIS=2, W=16, C=-1).signature()
        low, _ow = CL.unit_lowering(bind(d, NVIS=2, W=16, C=1), "w", [("kv", ["kv"])])
        words = [[int(x)] for x in kv.reshape(-1)[:32]]
        outs, ok = IL.evaluate_layout(low.layout, words, 1)
        _o, flipped = IL.evaluate_layout(low.layout, words, 1, flip=(low.layout.out_port_cols[3], 0))
        assert (low.units.n, low.layout.counts["ands"], ok, [o[0] for o in outs], flipped) == (1, 0, 1, [w[0] for w in words], 0)


@pytest.mark.slow
def test_the_scores_on_column_blocks_and_the_writers_at_the_served_statics_are_one_flock_block_each() -> None:
    """Read off the Boolean units' Calls at GLM-5.3's statics: the scores on nine column blocks of 64 take the scores unit's
    block, 2^27; the writer of a column block of 64 of a 64-key block takes 2^24, of a 16-key block 2^22."""
    from verity_flock import circuit as FC

    S = {k: BM.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}
    for first in (False, True):
        cols = FC.unit_block(bind(BM.SparseMlaHeadScoresColsV2, NVIS=64, FIRST=first, G=64, **S))
        scores = FC.unit_block(bind(BM.SparseMlaHeadScoresV2, NVIS=64, FIRST=first, **S))
        assert FC.past_block(cols) is None and cols["k_log"] == scores["k_log"] == 27
    for nvis, k_log in ((64, 24), (16, 22)):
        keys = FC.unit_block(bind(BM.SparseMlaHeadKeysWireV2, NVIS=nvis, W=64, C=0))
        assert FC.past_block(keys) is None and keys["k_log"] == k_log, (nvis, keys)
