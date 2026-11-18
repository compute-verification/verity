"""The fast implementation's compute-once functions (`verity_vllm.check.match.global_match_fast`) against the record's they
stand for, on views built by hand as a loader leaves them (the builders of `test_match_operand_reads.py`): the live cone
(`verity_vllm.check.match.dag_form.live_cone`), the canonical hash and the comparison of two views
(`verity_vllm.check.match.program_compare`), and an instance's operands relative to a step sequence.

Each gives the record's answer, and gives it once a view: a second ask is counted a hit and computes nothing.  The step-relative
operands are the view's own only where the sequence asked about is the view's steps and nothing is unfolded; unfolding, or over
another sequence, they are the record's function's, whose positions are places in that sequence."""
from __future__ import annotations

import pytest

from tests.check import test_match_operand_reads as reads
from verity_vllm.check.match import dag_form
from verity_vllm.check.match import global_match_fast as gmf
from verity_vllm.check.match import program_compare as C
from verity_vllm.check.match.instance_sequence import RootView
from verity_vllm.check.match.projection import Projection
from verity_vllm.program import compact as K

FORMS = pytest.mark.parametrize("compact", [False, True], ids=["record", "compact"])
GEMM = "Gemm_v1{M=1,N=8,K=8}"
TOK, WTS = 0, 1
INPUTS = {TOK: ("Input32_v1", 4), WTS: ("Input16_v1", 8)}
PROMPT, WEIGHTS = [("n", TOK, 0, 4)], [("n", WTS, 0, 8)]


@pytest.fixture(autouse=True)
def stats():
    """The fast path's counters, from nothing; no memo is left for the next test."""
    gmf.reset()
    yield gmf.stats
    gmf.reset()


def _view(inst: list[tuple], compact: bool) -> RootView:
    return reads._view(INPUTS, inst, compact=compact, weight_fields={WTS: "model.a.weight"})


# ------------------------------------------------------------------------------------------------ the live cone

def _three_steps(compact: bool) -> RootView:
    """Two engine steps and what follows the last select.  Instance 2 is read by nothing; instance 5 reads the last select."""
    return _view([(2, GEMM, [PROMPT, WEIGHTS]), (3, reads.SELECT, [[("n", 2, 0, 8)]]),
                  (4, GEMM, [PROMPT, WEIGHTS]), (5, GEMM, [[("n", 3, 0, 1)], WEIGHTS]), (6, reads.SELECT, [[("n", 5, 0, 8)]]),
                  (7, GEMM, [[("n", 6, 0, 1)], WEIGHTS])], compact)


@FORMS
def test_the_live_cone_is_the_last_selects_and_a_view_without_a_select_is_live_whole(compact):
    """What the last sampling event depends on, through the earlier selects it reads; an instance nothing on that path reads, and
    one after the last select, are outside it.  A view that holds no sampling event has no such cone: every instance is live."""
    P = _three_steps(compact)
    assert P.steps == [[0, 1], [2, 3, 4], [5]] and dag_form.live_cone(P) == {0, 1, 3, 4}
    no_select = _view([(2, GEMM, [PROMPT, WEIGHTS]), (3, GEMM, [[("n", 2, 0, 8)], WEIGHTS]), (4, GEMM, [PROMPT, WEIGHTS])], compact)
    assert no_select.steps == [[0, 1, 2]] and dag_form.live_cone(no_select) == {0, 1, 2}


@FORMS
def test_a_views_live_cone_is_computed_once_and_is_the_records(compact, stats):
    P, Q = _three_steps(compact), _three_steps(compact)
    first = gmf._live_cone_memo(P)
    assert first == dag_form.live_cone(P) and gmf._live_cone_memo(P) is first
    assert gmf._live_cone_memo(Q) == first and gmf._live_cone_memo(Q) is not first          # another view, its own cone
    assert (stats["live_cone_miss"], stats["live_cone_hit"]) == (2, 2)


# ------------------------------------------------------------------------------------------------ the comparison and the hash

R0_PROMPT, R1_PROMPT, FOLD_WTS = range(3)
FOLD_INPUTS = {R0_PROMPT: ("Input32_v1", 4), R1_PROMPT: ("Input32_v1", 4), FOLD_WTS: ("Input16_v1", 8)}


def _component(compact: bool) -> RootView:
    return _view([(2, reads.EMBED, [PROMPT, WEIGHTS]), (3, reads.SELECT, [[("n", 2, 0, 8)]])], compact)


def _requests(compact: bool) -> list[Projection]:
    """A fold of two requests of the component's shape, r0's instances before r1's, and each one's projection."""
    fold = reads._view(FOLD_INPUTS, [(3, reads.EMBED, [[("n", R0_PROMPT, 0, 4)], [("n", FOLD_WTS, 0, 8)]]), (4, reads.SELECT, [[("n", 3, 0, 8)]]),
                                     (5, reads.EMBED, [[("n", R1_PROMPT, 0, 4)], [("n", FOLD_WTS, 0, 8)]]), (6, reads.SELECT, [[("n", 5, 0, 8)]])],
                       compact=compact, oracle_wmap=[(0, 8, "model.a.weight")], _wlos=[0], oracle_wmap_overlaps=[], input_base={FOLD_WTS: 0})
    return [Projection(fold, f"fold[r{i}]", [2 * i, 2 * i + 1], tok_node=prompt, weights_node=FOLD_WTS, seed_node=None)
            for i, prompt in enumerate((R0_PROMPT, R1_PROMPT))]


@FORMS
def test_a_component_two_requests_are_compared_with_is_hashed_once_and_each_comparison_is_the_records(compact, stats):
    """The component is a plain view: its hash is kept, and the second request's comparison reads it.  A projection is hashed by
    the record's function each time, its cross-request reads being counted as it is read.  A pair compared before is not
    compared again: the document is the one object."""
    A, (P0, P1) = _component(compact), _requests(compact)
    first, second = gmf._compare_steps_memo(A, P0), gmf._compare_steps_memo(A, P1)
    for got, P in ((first, P0), (second, P1)):
        assert got == C.compare_steps(_component(compact), P)
        assert got["canonical_sha256"]["equal"] and got["totals"]["operands_different"] == 0
    assert first["canonical_sha256"]["a"] == second["canonical_sha256"]["a"]
    assert (stats["compare_steps_miss"], stats["canonical_hash_miss"], stats["canonical_hash_hit"], stats["canonical_hash_projection"]) == (2, 1, 1, 2)
    assert gmf._compare_steps_memo(A, P0) is first and stats["compare_steps_hit"] == 1
    assert (stats["canonical_hash_miss"], stats["canonical_hash_hit"], stats["canonical_hash_projection"]) == (1, 1, 2)


@FORMS
def test_a_views_hash_is_kept_by_whether_it_is_unfolded(compact, stats):
    """A padded view's hash unfolded is over the sequence without its zero and muxes; the two are kept apart, each computed once."""
    P = reads._mux_reader([("n", reads.M, 0, 2)], compact=compact)
    kept = [[0, 5, 6]]                                           # the embedding, the reader and the select: what unfolding keeps
    folded, unfolded = gmf._canonical_hash_memo(P, P.steps, False), gmf._canonical_hash_memo(P, kept, True)
    assert folded == C._canonical_hash(P, P.steps, False) != unfolded == C._canonical_hash(P, kept, True)
    assert gmf._canonical_hash_memo(P, P.steps, False) == folded and gmf._canonical_hash_memo(P, kept, True) == unfolded
    assert (stats["canonical_hash_miss"], stats["canonical_hash_hit"]) == (2, 2)


@FORMS
def test_unfolding_a_padded_program_against_a_flat_one_is_the_records_comparison(compact, stats):
    """The comparison `program-compare` asks for, through the fast path's functions: the unfolded side's operands are the
    record's function's, so the document is the record's, and a view's hash of that sequence is counted once."""
    def sides():
        return reads._mux_reader([("n", reads.M, 0, 2)], compact=compact), reads._flat_reader([("n", reads.E, 2, 4)], compact=False)
    got, record = gmf._compare_steps_memo(*sides(), unfold_a=True), C.compare_steps(*sides(), unfold_a=True)
    assert got == record and got["canonical_sha256"]["equal"] and got["totals"]["operands_different"] == 0
    assert (stats["compare_steps_miss"], stats["canonical_hash_miss"], stats["canonical_hash_hit"]) == (1, 2, 0)


@FORMS
@pytest.mark.parametrize("unfolded_first", [False, True], ids=["as it stands first", "unfolded first"])
def test_one_pair_compared_as_it_stands_and_unfolded_is_kept_as_two_comparisons(compact, unfolded_first, stats):
    """A comparison is kept by whether its first side is unfolded.  The padded Program differs from the flat one as it stands and
    equals it unfolded; each document is the record's, in whichever order the two are asked, and a second ask of either is a hit."""
    def sides():
        return reads._mux_reader([("n", reads.M, 0, 2)], compact=compact), reads._flat_reader([("n", reads.E, 2, 4)], compact=False)
    A, B = sides()
    got = {unfold: gmf._compare_steps_memo(A, B, unfold_a=unfold) for unfold in (unfolded_first, not unfolded_first)}
    for unfold in (False, True):
        assert got[unfold] == C.compare_steps(*sides(), unfold_a=unfold) and got[unfold]["canonical_sha256"]["equal"] is unfold
    assert (got[False]["steps"][0]["n_a"], got[True]["steps"][0]["n_a"]) == (7, 3)
    assert gmf._compare_steps_memo(A, B) is got[False] and gmf._compare_steps_memo(A, B, unfold_a=True) is got[True]
    # the flat side's hash is the same either way and is read the second time; the padded side's is computed for each
    assert (stats["compare_steps_miss"], stats["compare_steps_hit"], stats["canonical_hash_miss"], stats["canonical_hash_hit"]) == (2, 2, 3, 1)


# ------------------------------------------------------------------------------------------------ operands relative to a step sequence

READ, LAST = reads.READ - 2, reads.READ - 1                      # the padded step's reader and its select, as instances
KEPT = [[0, READ, LAST]]                                         # the step without its zero and muxes
STEP_RELATIVE = [   # (the sequence: None for the view's steps, unfolding, the reader's operand, the select's)
    pytest.param(None, False, [("N", 0, 2, 0, 2)], [("N", 0, READ, 0, 8)], id="the view's steps"),
    pytest.param(None, True, [("N", 0, 0, 2, 4)], [("N", 0, READ, 0, 8)], id="the view's steps, unfolding"),
    pytest.param(KEPT, True, [("N", 0, 0, 2, 4)], [("N", 0, 1, 0, 8)], id="what unfolding keeps, unfolding"),
    pytest.param(KEPT, False, [("N-PAD", 0, 2, 0, 2)], [("N", 0, 1, 0, 8)], id="what unfolding keeps, not unfolding"),
]


@FORMS
@pytest.mark.parametrize(("sequence", "unfold", "reader", "select"), STEP_RELATIVE)
def test_an_instances_operands_relative_to_a_sequence_are_the_records(compact, sequence, unfold, reader, select, stats):
    """The reader reads a mux, and the select reads the reader.  Over the view's steps and without unfolding the operands are the
    view's own, in its own form; in the three other cases they are the record's function's, in the record's form from a compact
    view too: a producer is named by its place in the sequence asked about, the mux's live operand when unfolding, and a
    producer the sequence does not hold is `N-PAD` under the view's own step and index."""
    def view():
        return reads._mux_reader([("n", reads.M, 0, 2)], compact=compact)
    P, R = view(), view()
    seqs, record_seqs = (P.steps, R.steps) if sequence is None else (sequence, sequence)
    for i, want in ((READ, reader), (LAST, select)):
        got = gmf._canon_step_relative_identity(P, i, seqs, unfold)
        assert got == C._canon_step_relative(R, i, record_seqs, unfold)
        assert [K.expand_mixed(arg) if compact and sequence is None and not unfold else list(arg) for arg in got] == [want]
    # whether the sequence is the view's steps is asked once a list, and never when unfolding
    assert stats["seqs_identity_checks"] == (0 if unfold else 1)


@FORMS
def test_a_sequence_equal_to_the_views_steps_in_another_list_is_the_views_own_too(compact, stats, monkeypatch):
    """The record's function is not asked: it would give the same operands, at the cost the view's own saves."""
    P = reads._mux_reader([("n", reads.M, 0, 2)], compact=compact)
    copy = [list(step) for step in P.steps]

    def the_records(*args):
        raise AssertionError(f"the record's function is asked about instance {args[1]}")
    monkeypatch.setattr(C, "_canon_step_relative", the_records)
    assert gmf._canon_step_relative_identity(P, READ, copy, False) == gmf._canon_step_relative_identity(P, READ, P.steps, False) == \
        [tuple(arg) for arg in P.canon(READ, False)]
    assert stats["seqs_identity_checks"] == 2                    # asked again when another list is handed
