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
                                    bind(BM.SparseMlaHeadKeysV2, NVIS=16, W=16),
                                    bind(BM.SparseMlaHeadKeysWireV2, NVIS=16, W=16),
                                    bind(BM.SparseMlaHeadScoresKeysV2, NVIS=5, FIRST=True, G=16, **_BITS),
                                    bind(BM.SparseMlaHeadScoresKeysV2, NVIS=16, FIRST=False, G=16, **_BITS)],
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


def test_the_writer_is_a_copy() -> None:
    kv = _planted(np.random.default_rng(5), _bf16(np.random.default_rng(6), (8, 16, 32)), BF16_SPECIALS, 0.05)
    got = np.asarray(E.evaluate_batch(bind(M.SparseMlaHeadKeys, NVIS=16, W=32), kv)).reshape(kv.shape)
    assert np.array_equal(got, kv)


def test_the_wire_writer_is_the_identity_with_gates_at_no_and_and_binds_its_output_to_its_input() -> None:
    """Form (b3)'s writer: the identity on its words (word and Boolean), gates in the IR (the partition's `fn.gates`), none an
    AND, and its flat unit (`class_statement.unit_lowering`, as `UnitStager` stages it) of one unit, whose lane a flipped output
    bit leaves unsatisfied."""
    from verity_flock import class_statement as CL
    from verity_flock import ir_lower as IL

    kv = _planted(np.random.default_rng(5), _bf16(np.random.default_rng(6), (8, 16, 32)), BF16_SPECIALS, 0.05)
    word, boo = bind(M.SparseMlaHeadKeysWire, NVIS=16, W=32), bind(BM.SparseMlaHeadKeysWireV2, NVIS=16, W=32)
    assert np.array_equal(np.asarray(E.evaluate_batch(word, kv)).reshape(kv.shape), kv)
    assert np.array_equal(np.asarray(BS.evaluate_words(boo, kv)).astype(kv.dtype).reshape(kv.shape), kv)
    assert (word.gates, boo.gates) == (2 * 16 * 32, 16 * 32)
    for d in (M.SparseMlaHeadKeysWire, BM.SparseMlaHeadKeysWireV2):
        low, _ow = CL.unit_lowering(bind(d, NVIS=2, W=16), "w", [("kv", ["kv"])])
        words = [[int(x)] for x in kv.reshape(-1)[:32]]
        outs, ok = IL.evaluate_layout(low.layout, words, 1)
        _o, flipped = IL.evaluate_layout(low.layout, words, 1, flip=(low.layout.out_port_cols[3], 0))
        assert (low.units.n, low.layout.counts["ands"], ok, [o[0] for o in outs], flipped) == (1, 0, 1, [w[0] for w in words], 0)


@pytest.mark.parametrize("st,NVIS,FIRST,G", [(SMALL, 16, False, 16), (SMALL, 5, True, 8), (TOY, 16, False, 32),
                                             (TOY, 8, True, 32)])
def test_the_gated_writer_is_the_scores_unit_and_the_keys_it_read(st, NVIS, FIRST, G) -> None:
    DR, DV, n = st["DR"], st["DV"], 16
    scores = bind(M.SparseMlaHeadScores, NVIS=NVIS, FIRST=FIRST, **st)
    writer = bind(M.SparseMlaHeadScoresKeys, NVIS=NVIS, FIRST=FIRST, G=G, **st)
    qn, qr, kv, *state = _inputs(n, NVIS, FIRST, DR, DV, 11 * NVIS + G + FIRST)
    widths = [1, 4, NVIS] + [1] * (not FIRST)
    cols = [w * NVIS for w in [G] * (DV // G) + [DR]]
    want = np.asarray(E.evaluate_batch(scores, qn, qr, kv, *state[:2])).astype(np.uint64).reshape(n, -1)
    got = np.asarray(E.evaluate_batch(writer, qn, qr, kv, *state[:2])).astype(np.uint64).reshape(n, -1)
    lead = want.shape[1]
    assert got.shape[1] == lead + sum(cols), f"{writer.id}: {got.shape[1]} words, not {lead} + {sum(cols)}"
    assert np.array_equal(got[:, :lead], want), f"{writer.id}: its scores returns are not {scores.id}'s"
    want_cols = np.concatenate([c.reshape(n, -1) for c in _cols(np.asarray(kv).reshape(n, NVIS, DR + DV), DV, G)], axis=1)
    assert np.array_equal(got[:, lead:], want_cols.astype(np.uint64)), f"{writer.id}: its key returns are not its kv's columns"


def test_the_gated_writer_binds_its_returned_keys_to_its_input() -> None:
    """The flat unit (`class_statement.unit_lowering`) of the Boolean writer at small statics: honest inputs satisfy every row,
    and a returned key bit flipped after witness generation (`ir_lower.evaluate_layout`'s ``flip``) leaves the lane unsatisfied:
    the output row's copy row reads the input column."""
    from verity_flock import class_statement as CL
    from verity_flock import ir_lower as IL

    target = bind(BM.SparseMlaHeadScoresKeysV2, NVIS=2, FIRST=True, G=16, **_BITS)
    params = [nm for nm, _t in target.signature()[0]]
    low, out_words = CL.unit_lowering(target, "writer", [(p, [p]) for p in params])
    word = TR.word_of(target)
    inputs = _operands(word, 1, 3)
    words = np.concatenate([np.asarray(x).astype(np.uint64).reshape(-1) for x in inputs])
    lay = low.layout
    assert len(lay.in_bits) == words.size
    outs, ok = IL.evaluate_layout(lay, [[int(w)] for w in words], 1)
    leaves = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64).reshape(-1)
    widths = word.signature()[1].leaf_widths()
    want = [int(v) >> s & 0xFFFF for v, w in zip(leaves, widths, strict=True) for s in range(0, w, 16)]
    assert ok == 1 and [int(o[0]) for o in outs] == want
    lead = 2 + 8 + 2
    key_word = lead
    _o, ok_flip = IL.evaluate_layout(lay, [[int(w)] for w in words], 1, flip=(lay.out_port_cols[key_word], 0))
    assert ok_flip == 0, "a flipped returned key bit satisfies every row: the return is not bound to the input"
    _o, ok_score = IL.evaluate_layout(lay, [[int(w)] for w in words], 1, flip=(lay.out_port_cols[0], 0))
    assert ok_score == 0


def test_the_variant_and_the_writer_at_the_served_statics_as_flock_blocks() -> None:
    """Read off the Boolean units' Calls: the variant takes the scores unit's block, 2^27."""
    from verity_flock import circuit as FC

    S = {k: BM.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}
    for first in (False, True):
        cols = FC.unit_block(bind(BM.SparseMlaHeadScoresColsV2, NVIS=64, FIRST=first, G=S["DR"], **S))
        scores = FC.unit_block(bind(BM.SparseMlaHeadScoresV2, NVIS=64, FIRST=first, **S))
        print({"first": first, "cols": cols, "scores": scores})
        assert FC.past_block(cols) is None and cols["k_log"] == scores["k_log"] == 27
