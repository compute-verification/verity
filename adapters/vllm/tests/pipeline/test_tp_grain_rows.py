"""`verity_vllm.pipeline.tp.grain`: a collective site lowered at row grain, where each producer is a row whose own Definition is
its unit (a top-k sum's row, a fused call's residual or normed row): one combine unit a row for an elementwise all-reduce, the
fused call itself for its one token, no combine for a gather, and neither a gather of words nor a norm of the site's own.  Then
the sites `lower` refuses by name: a partial that is part of such a row, producers of two grains, rows of two widths, ranks
whose partials differ in units, a combine whose parts are not the ranks' rows.  Last, the width of a token select's group unit
and the chunk of a gathered row.  `lower` is given the runs a binding's site holds, written out here: no request is compiled."""
from __future__ import annotations

from typing import Any

import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.ops import gemm as GE
from verity.commitments.windows import interface as I
from verity.definitions.registry import b1_tp2 as B1T
from verity.definitions.registry import moe
from verity.definitions.registry import tp_allreduce as TA

from verity_vllm.pipeline.tp import grain as TG

N = 16
#: a producer of coordinate grain (its unit is one output word) and two of row grain (each its own unit)
GEMM = bind(GE.Gemm, K=N, N=N, **GE.GEMM_AMPERE)
SUM, SUM8 = bind(moe.MoeSum, TOPK=3, H=N), bind(moe.MoeSum, TOPK=3, H=8)
#: the fused calls, the first of two outputs (the residual row, then the normed row)
FUSED = bind(TA.AllReduceRmsNormMnnvl, WORLD=2, N=N, SHOT=1, CLUSTER=1, EPS=1e-5, FMA=False)
RESIDUAL = bind(TA.AllReduceResidualMnnvl, WORLD=2, N=N, SHOT=1)
REDUCE, REDUCE2, GATHER = bind(B1T.AllReduce, WORLD=2, N=N), bind(B1T.AllReduce2, N=N), bind(B1T.AllGather, WORLD=2, N=N)
WHERE = "w2: request 0's collective 3 [48, 64)"
READS = [(0, N)]


def _said(low: TG.Lowering) -> dict[str, Any]:
    """A lowering with each Definition by its id."""
    return {"grain": low.grain, "width": low.width, "positions": low.positions,
            "producers": [[(fn.id, out, n) for fn, out, n in rank] for rank in low.producers], "chunk": low.chunk,
            "gather": low.gather and low.gather.id, "combines": [(fn.id, n) for fn, n in low.combines], "parts": low.parts, "row": low.row,
            "output": low.output and low.output.id, "norm": low.norm and low.norm.id, "rows": low.rows, "tokens": low.tokens}


def _rows(producer: Any, out: str, positions: int, combines: list[tuple[str, int]]) -> dict[str, Any]:
    """What `_said` gives for two ranks' ``positions`` rows of ``producer``'s output ``out`` at row grain."""
    return {"grain": "row", "width": N, "positions": positions, "producers": [[(producer.id, out, 1)] * positions] * 2, "chunk": N,
            "gather": None, "combines": combines, "parts": "parts", "row": N, "output": None, "norm": None, "rows": positions, "tokens": positions}


# ------------------------------------------------------------------------------------------------------------------------ row grain


def test_an_all_reduce_of_rows_that_are_their_own_units_is_one_combine_unit_a_row():
    """Two rows a rank: the all-reduce at the row's width, twice, as one run; each row is read whole, so nothing gathers words."""
    low = TG.lower(REDUCE, [[(SUM, 0, N), (SUM, 0, N)]] * 2, READS, WHERE)
    assert _said(low) == _rows(SUM, "o0", 2, [("AllReduce_v2{WORLD=2,N=16}", 2)])
    assert descriptor_id(low.combines[0][0]) == descriptor_id(REDUCE)
    assert dict(I.layout(low.combines[0][0]).rows)[low.parts] == 2 * N


def test_the_two_rank_all_reduce_is_its_family_at_the_rows_width():
    low = TG.lower(REDUCE2, [[(SUM, 0, N)]] * 2, READS, WHERE)
    assert _said(low) == _rows(SUM, "o0", 1, [("AllReduce2_v1{N=16}", 1)])


def test_an_elementwise_all_reduce_over_rows_is_its_family_at_the_rows_width_whatever_its_own():
    """The 16-word all-reduce over one 16-word row a rank, then over one 8-word row a rank: the second site's unit is the
    family at eight words, not the first site's."""
    assert [(fn.id, n) for fn, n in TG.lower(REDUCE, [[(SUM, 0, N)]] * 2, READS, WHERE).combines] == [("AllReduce_v2{WORLD=2,N=16}", 1)]
    low = TG.lower(REDUCE, [[(SUM8, 0, 8)]] * 2, [(0, 8)], WHERE)
    assert (low.grain, low.width, low.chunk, low.row) == ("row", 8, 8, 8)
    assert [(fn.id, n) for fn, n in low.combines] == [("AllReduce_v2{WORLD=2,N=8}", 1)]


@pytest.mark.parametrize("fused", [FUSED, RESIDUAL], ids=["with the norm", "the residual alone"])
def test_a_fused_call_over_a_row_that_is_its_own_unit_is_the_call_itself(fused):
    """One token a call: the call is the unit, and its normed row is the call's own output, so the site has no norm unit."""
    low = TG.lower(fused, [[(SUM, 0, N)]] * 2, READS, WHERE)
    assert _said(low) == _rows(SUM, "o0", 1, [(fused.id, 1)]) and low.combines == ((fused, 1),)


@pytest.mark.parametrize("fused", [FUSED, RESIDUAL], ids=["with the norm", "the residual alone"])
def test_a_fused_call_over_two_rows_that_are_their_own_units_is_refused_by_name(fused):
    with pytest.raises(ValueError) as refused:
        TG.lower(fused, [[(SUM, 0, N), (SUM, 0, N)]] * 2, READS, WHERE)
    assert str(refused.value) == f"{WHERE}: {fused.id} has no row combine units: {fused.id}: one token a call, not 2 rows"


def test_the_grain_is_part_of_what_a_sites_combine_units_are_shared_by():
    """One fused call of one 16-word token, over words and over a row that is its own unit: the words' site has the call's row
    unit and its norm, the row's site the call itself, whichever is lowered first."""
    words = TG.lower(FUSED, [[(GEMM, 0, N)]] * 2, READS, WHERE)
    rows = TG.lower(FUSED, [[(SUM, 0, N)]] * 2, READS, WHERE)
    assert (rows.grain, rows.combines, rows.norm) == ("row", ((FUSED, 1),), None)
    assert (words.grain, [(fn.id, n) for fn, n in words.combines], words.norm and words.norm.id) == (
        "coordinate", [("AllReduceResidualMnnvlAt_v1{WORLD=2,N=16,C=16,AT=0,SHOT=1}", 1)], "RmsNormMnnvl_v1{N=16,SHOT=1,CLUSTER=1,EPS=1e-05,FMA=False}")
    assert (words.rows, words.tokens, words.chunk) == (rows.rows, rows.tokens, rows.chunk) == (1, 1, N)


def test_a_row_that_is_its_own_unit_is_one_combine_unit_however_its_output_is_read():
    """At coordinate grain a read of four words would make the token row four words; here the row is the producer's."""
    low = TG.lower(REDUCE, [[(SUM, 0, N)]] * 2, [(0, 4), (4, N)], WHERE)
    assert _said(low) == _rows(SUM, "o0", 1, [("AllReduce_v2{WORLD=2,N=16}", 1)])


def test_a_row_that_is_its_own_unit_is_one_combine_unit_however_wide_it_is():
    """128 words, one more than a gathered row holds: nothing is gathered at row grain, so the row is not cut into chunks."""
    wide = bind(moe.MoeSum, TOPK=3, H=128)
    low = TG.lower(bind(B1T.AllReduce, WORLD=2, N=128), [[(wide, 0, 128)]] * 2, [(0, 128)], WHERE)
    assert TG.chunk(128) == 64 and (low.grain, low.width, low.chunk, low.row, low.rows, low.tokens) == ("row", 128, 128, 128, 1, 1)
    assert [(fn.id, n) for fn, n in low.combines] == [("AllReduce_v2{WORLD=2,N=128}", 1)] and low.gather is None and low.output is None


def test_a_gather_of_rows_that_are_their_own_units_has_no_combine_unit():
    low = TG.lower(GATHER, [[(SUM, 0, N), (SUM, 0, N)]] * 2, READS, WHERE)
    assert _said(low) == _rows(SUM, "o0", 2, []) and low.combines == ()


@pytest.mark.parametrize("lo, out", [(0, "o0"), (N, "o1")], ids=["its first output", "its second output"])
def test_a_partial_that_is_one_whole_output_of_a_call_of_two_is_that_outputs_row(lo, out):
    low = TG.lower(REDUCE, [[(FUSED, lo, lo + N)]] * 2, READS, WHERE)
    assert _said(low) == _rows(FUSED, out, 1, [("AllReduce_v2{WORLD=2,N=16}", 1)])


def test_a_sites_combine_units_are_made_once_and_shared_by_the_sites_of_its_shape():
    first = TG.lower(REDUCE, [[(SUM, 0, N), (SUM, 0, N)]] * 2, READS, WHERE)
    again = TG.lower(REDUCE, [[(FUSED, N, 2 * N), (FUSED, 0, N)]] * 2, [(0, 8)], "another site")
    assert again.combines is first.combines
    assert TG.lower(REDUCE, [[(SUM, 0, N)]] * 2, READS, WHERE).combines is not first.combines        # one row, another shape


# ------------------------------------------------------------------------------------------------------- coordinate grain, beside it


def test_an_elementwise_all_reduce_of_words_is_its_family_at_the_chunk_once_a_chunk_as_one_run():
    """A 32-word partial of two Gemm rows whose output is read eight words at a time: the token row is eight words, each rank's
    words gathered into four rows of eight, and the all-reduce at eight words four times.  A row is one chunk, so nothing
    gathers the chunks."""
    low = TG.lower(bind(B1T.AllReduce, WORLD=2, N=2 * N), [[(GEMM, 0, N), (GEMM, 0, N)]] * 2, [(0, 8), (8, 16), (16, 24), (24, 32)], WHERE)
    assert _said(low) == {"grain": "coordinate", "width": 1, "positions": 32, "producers": [[("Gemm_v2{K=16,N=1,DOT=AmpereBF16TcDot16_v2}", "o0", N)] * 2] * 2,
                          "chunk": 8, "gather": "AllGather_v1{WORLD=8,N=1}", "combines": [("AllReduce_v2{WORLD=2,N=8}", 4)], "parts": "parts", "row": 8,
                          "output": None, "norm": None, "rows": 4, "tokens": 4}


@pytest.mark.parametrize("fused, norm", [
    (bind(TA.AllReduceRmsNormMnnvl, WORLD=2, N=256, SHOT=1, CLUSTER=1, EPS=1e-5, FMA=False), "RmsNormMnnvl_v1{N=256,SHOT=1,CLUSTER=1,EPS=1e-05,FMA=False}"),
    (bind(TA.AllReduceResidualMnnvl, WORLD=2, N=256, SHOT=1), None),
], ids=["with the norm", "the residual alone"])
def test_a_fused_call_of_a_256_word_token_over_words_is_four_chunk_units_each_a_run_of_its_own(fused, norm):
    """Four chunks of 64 words: each chunk's unit is the call's row unit at that chunk, so no two are one run; the chunks are
    gathered into the token's one row, which only the call with the norm then norms."""
    gemm = bind(GE.Gemm, K=N, N=256, **GE.GEMM_AMPERE)
    low = TG.lower(fused, [[(gemm, 0, 256)]] * 2, [(0, 256)], WHERE)
    assert _said(low) == {
        "grain": "coordinate", "width": 1, "positions": 256, "producers": [[("Gemm_v2{K=16,N=1,DOT=AmpereBF16TcDot16_v2}", "o0", 256)]] * 2,
        "chunk": 64, "gather": "AllGather_v1{WORLD=64,N=1}",
        "combines": [(f"AllReduceResidualMnnvlAt_v1{{WORLD=2,N=256,C=64,AT={c},SHOT=1}}", 1) for c in range(4)], "parts": "parts", "row": 256,
        "output": "AllGather_v1{WORLD=4,N=64}", "norm": norm, "rows": 4, "tokens": 1}


def test_part_of_a_row_of_words_is_that_many_units():
    """What row grain refuses, coordinate grain takes: half a Gemm row is eight of its coordinate units."""
    low = TG.lower(bind(B1T.AllReduce, WORLD=2, N=8), [[(GEMM, 8, N)]] * 2, [(0, 8)], WHERE)
    said = _said(low)
    assert (said["grain"], said["positions"], said["chunk"], said["combines"]) == ("coordinate", 8, 8, [("AllReduce_v2{WORLD=2,N=8}", 1)])
    assert said["producers"] == [[("Gemm_v2{K=16,N=1,DOT=AmpereBF16TcDot16_v2}", "o0", 8)]] * 2


# --------------------------------------------------------------------------------------------------------------------- the refusals


REDUCE3 = bind(B1T.AllReduce, WORLD=3, N=N)


@pytest.mark.parametrize("combine, own, why", [
    (REDUCE, [[(SUM, 0, N)], [(SUM, 0, 8)]],
     "rank 1's partial reads leaves [0, 8) of a MoeSum_v1{TOPK=3,H=16} node, neither its output words' units nor one whole output of it"),
    (REDUCE, [[(FUSED, 8, 24)], [(SUM, 0, 8)]],
     f"rank 0's partial reads leaves [8, 24) of a {FUSED.id} node, neither its output words' units nor one whole output of it"),
    (REDUCE, [[(SUM, 0, N)], [(SUM, 0, N), (FUSED, 0, 2 * N)]],
     f"rank 1's partial reads leaves [0, 32) of a {FUSED.id} node, neither its output words' units nor one whole output of it"),
    (REDUCE, [[(GEMM, 0, N)], [(SUM, 0, N)]], "its producers are of grains ['coordinate', 'row']: a combine's grain is its producers' one grain"),
    (REDUCE, [[(SUM, 0, N), (GEMM, 0, N)]] * 2, "its producers are of grains ['coordinate', 'row']: a combine's grain is its producers' one grain"),
    (REDUCE, [[(SUM, 0, N)], [(SUM8, 0, 8)]], "its producers' rows are [8, 16] words: a row-grain combine reads rows of one width"),
    (REDUCE, [[(SUM, 0, N)], [(SUM, 0, N), (SUM, 0, N)]], "its ranks' partials are [1, 2] units"),
    (REDUCE, [[(GEMM, 0, N)], [(GEMM, 0, 8)]], "its ranks' partials are [8, 16] units"),
    (REDUCE3, [[(SUM, 0, N)]] * 2, "AllReduce_v2{WORLD=3,N=16}'s parts is not 2 rows of 16 words"),
    (REDUCE, [[(SUM, 0, N)]] * 3, "AllReduce_v2{WORLD=2,N=16}'s parts is not 3 rows of 16 words"),
], ids=["part of a row", "across a call's two outputs", "both outputs of a call, in the second rank", "a rank of words and a rank of rows",
        "words and rows in one rank", "rows of two widths", "ranks of one row and two", "ranks of 16 words and 8",
        "a combine of more ranks than the site has", "a combine of fewer ranks than the site has"])
def test_a_site_no_grain_holds_for_is_refused_by_name(combine, own, why):
    with pytest.raises(ValueError) as refused:
        TG.lower(combine, own, READS, WHERE)
    assert str(refused.value) == f"{WHERE}: {why}"


@pytest.mark.parametrize("own, grain", [([[(SUM, 0, N)]] * 2, "row"), ([[(GEMM, 0, N)]] * 2, "coordinate")])
def test_a_combine_of_no_family_here_is_refused_with_the_site_the_combine_and_the_grain(own, grain):
    """The reason after them is `tp_combine.row_binding`'s own."""
    with pytest.raises(ValueError) as refused:
        TG.lower(bind(moe.MoeSum, TOPK=2, H=N), own, READS, WHERE)
    assert str(refused.value).startswith(f"{WHERE}: MoeSum_v1{{TOPK=2,H=16}} has no {grain} combine units: MoeSum_v1: ")
    assert isinstance(refused.value.__cause__, ValueError)


# ------------------------------------------------------------------------------------------------------------- a Definition's outputs


def test_a_definitions_outputs_are_its_fields_in_order_each_with_its_leaves():
    assert TG._fields(FUSED) == [("o0", 0, N), ("o1", N, 2 * N)] and TG._fields(SUM) == [("o0", 0, N)]
    assert TG._out(FUSED, 1) == ("o1", N, 2 * N) and TG._out(SUM, 0) == ("o0", 0, N)


@pytest.mark.parametrize("fn, f", [(FUSED, 2), (SUM, 1)])
def test_an_output_a_definition_does_not_have_is_refused_by_name(fn, f):
    with pytest.raises(ValueError) as refused:
        TG._out(fn, f)
    assert str(refused.value) == f"{fn.id} has no output {f}"


# ---------------------------------------------------------------------------------------- the token select's groups, and a row's chunk


@pytest.mark.parametrize("V, w, G, C", [
    (1024, 64, 16, 64), (512, 1, 16, 16), (48, 3, 16, 48), (48, 3, 8, 24), (152064, 64, 16, 1408),
    (127, 1, 1, 1), (128, 1, 1, 2), (254 * 127, 1, 1, 254),
    (255 * 127, 1, 1, None), (64 * 257, 64, 16, None), (7, 2, 16, None), (100, 64, 16, None),
], ids=["one row of 64", "a group of 16 words", "whole rows and whole groups", "the least of them", "a vocabulary: 108 units of 22 rows",
        "127 units is the most a select reads", "128 units are too many", "254 rows is the most a group unit reads",
        "255 rows are too many", "a prime count of rows", "fewer logits than a group", "rows that do not divide the logits"])
def test_a_token_selects_group_unit_reads_the_least_whole_groups_and_rows_that_fit_both_proof_blocks(V, w, G, C):
    """C logits a unit: the unit reads C / w rows and writes two (C / w + 2 rows), and the select over the V / C units reads each
    one's best and index and writes two (2 V / C + 2 rows); a proof block holds at most 256."""
    assert I.MAX_ROWS == 256 and TG.select_width(V, w, G) == C
    if C is not None:
        assert V % C == 0 and C % w == 0 and C % G == 0 and C // w + 2 <= 256 and 2 * (V // C) + 2 <= 256


@pytest.mark.parametrize("words, chunk", [(1, 1), (16, 16), (127, 127), (128, 64), (254, 127), (256, 64), (131, 1), (4096, 64), (381, 127)],
                         ids=["one word", "a short row", "the most", "one more than the most", "twice the most", "a hidden row of 256",
                              "a prime past the most", "a hidden row of 4096", "three times the most"])
def test_a_gathered_rows_chunk_is_the_rows_largest_divisor_of_at_most_127_words(words, chunk):
    assert TG.CHUNK_MAX == 127 and TG.chunk(words) == chunk


@pytest.mark.parametrize("combine, words, reads, row", [
    (REDUCE, 32, [(0, 8), (8, 32)], 8), (REDUCE, 32, [(0, 32)], 32), (REDUCE, 32, [], 32), (REDUCE, 32, [(0, 12), (12, 32)], 4),
    (REDUCE, 32, [(0, 8)], 8), (GATHER, 32, [(16, 32)], 16), (FUSED, 32, [(0, 8)], 32), (RESIDUAL, 32, [(0, 8)], 32),
], ids=["read in eights", "read whole", "not read", "read at a twelfth word", "its first eight words read", "a gather read by halves",
        "the fused call, however read", "the fused residual, however read"])
def test_a_sites_token_row_is_the_widest_row_every_read_of_its_output_is_whole_rows_of(combine, words, reads, row):
    assert TG.row(combine, words, reads) == row
