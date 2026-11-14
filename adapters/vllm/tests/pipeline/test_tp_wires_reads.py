"""`verity_vllm.pipeline.tp.wires`: the reads of a combine's output that a window wires or lists by name as free inputs, and the
windows `sites_circuit` refuses, over bindings written out by hand (`tp_binding.RequestBinding`: two ranks, each collective
one 16-word row a rank).  A site here is one producer a rank and its combine, so each case says which read of which site it
is about, and no request is compiled.

First a fused call that is its own unit, whose two outputs are two writer rows.  Then the reads the grains do not fit, each
left a free input and named: part of a writer row, a row into a port that is not whole rows of that width or from a word that
is no row's start, a port two reads cut two ways, a gather reading a combine, a producer's port its unit does not have; and a
call that reads two sites, one unit at the later.  Last the refusals: no binding, ranks the bindings do not have, and a
template that an earlier request's call cut otherwise than a later request's gather, combine or select needs it (a rank
Program's collective is always a site's combine, so only a binding written by hand has such a call)."""
from __future__ import annotations

from typing import Any

import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.ops import gemm as GE
from verity.definitions.registry import b1_tp2 as B1T
from verity.definitions.registry import dead_free as DF
from verity.definitions.registry import moe
from verity.definitions.registry import tp_allreduce as TA

from verity_vllm.pipeline.tp import wires as TW
from verity_vllm.program import tp_binding as TB

N, NAME = 16, "w2"
#: a producer whose unit is one output word (`WORD`, sixteen units a row), and one that is its own unit
GEMM, SUM = bind(GE.Gemm, K=N, N=N, **GE.GEMM_AMPERE), bind(moe.MoeSum, TOPK=3, H=N)
WORD = bind(GE.Gemm, K=N, N=1, **GE.GEMM_AMPERE)
FUSED = bind(TA.AllReduceRmsNormMnnvl, WORLD=2, N=N, SHOT=1, CLUSTER=1, EPS=1e-5, FMA=False)
REDUCE, GATHER = bind(B1T.AllReduce, WORLD=2, N=N), bind(B1T.AllGather, WORLD=2, N=N)
#: calls with one port: of 8 words, and of 16 words in sixteen rows
EIGHT, WORDS = bind(B1T.AllGather, WORLD=1, N=8), bind(B1T.AllGather, WORLD=N, N=1)
SELECT, CHOOSE = bind(DF.TokenSelectV4, V=2 * N), bind(DF.TokenSelectCombineV2, V=2 * N, C=N)
#: each rank's partial at a site of sums: the two units, and their wires into the combine's parts
SUMS = {(TW.PRODUCER, 0): [(SUM.id, 0, 1)], (TW.PRODUCER, 1): [(SUM.id, 1, 1)]}


def _binding(sites: list[TB.Site], fns: dict[int, Any], world: int = 2) -> TB.RequestBinding:
    """A request whose ranks' Programs have the same nodes: ``fns`` by node."""
    return TB.RequestBinding(world, tuple(f"rank {r}'s Program" for r in range(world)), tuple(sites), {r: fns for r in range(world)})


def _site(k: int, combine: int, producer: int, *reads: tuple[int, int, int, int, int], by_rank: tuple | None = None) -> TB.Site:
    """Collective ``k`` of both ranks: node ``combine`` over node ``producer``'s 16 words at each, its output read by ``reads`` at
    each rank (``by_rank``: each rank's own reads)."""
    return TB.Site(k, 0, N, ((combine, "parts"),) * 2, (((producer, 0, N),),) * 2, by_rank if by_rank is not None else (reads,) * 2)


def _units(sites: TW.Sites, request: int = 0, site: int = 0) -> dict[tuple[str, int], list[tuple[str, int, int]]]:
    """One site's units by (stage, rank), each range with its Definition by id."""
    fns = {descriptor_id(fn): fn.id for fn, _ in sites.circuit.templates}
    return {(stage, r): [(fns[t], at, n) for t, at, n in ranges] for (i, j, stage, r), ranges in sites.units.items() if (i, j) == (request, site)}


def _into(sites: TW.Sites, reader: Any) -> list[tuple]:
    """The wires into ``reader``'s units: (port, the writer by id, its output, the first reader unit, radix, base, step, cut)."""
    fns = {descriptor_id(fn): fn.id for fn, _ in sites.circuit.templates}
    return [(w.port, fns[w.writer], w.out, w.at, w.radix, w.base, w.step, w.cut) for w in sites.circuit.wires
            if w.reader == descriptor_id(reader)]


def _what(k: int, rank: int, node: int, fn: Any, port: str, hi: int = N) -> str:
    return f"{NAME}: request 0's collective {k} [0, {hi}): rank {rank}'s node {node} ({fn.id}) {port}"


# ------------------------------------------------------------------------------------------------ a fused call that is its own unit


def test_a_fused_call_that_is_its_own_unit_writes_its_residual_row_and_its_normed_row_and_each_read_is_wired_from_its_own():
    """Rank 0's next call reads the residual row, rank 1's the normed row: the call is the one unit of the site's combine, with
    no gather, output gather or norm of the site's own, and its two outputs are the site's two writer rows."""
    sites = TW.sites_circuit([_binding([_site(0, 1, 0, by_rank=(((2, 0, 0, 0, N),), ((2, 0, 0, N, 2 * N),)))], {0: SUM, 1: FUSED, 2: GEMM})], NAME)
    assert _units(sites) == {**SUMS, (TW.COMBINE, TW.ALL): [(FUSED.id, 0, 1)], (TW.CONSUMER, 0): [(GEMM.id, 0, 1)], (TW.CONSUMER, 1): [(GEMM.id, 1, 1)]}
    assert _into(sites, FUSED) == [("parts", SUM.id, "o0", 0, (1,), 0, (1,), (0, 1)), ("parts", SUM.id, "o0", 0, (1,), 1, (1,), (1, 1))]
    assert _into(sites, GEMM) == [("x", FUSED.id, "o0", 0, (1,), 0, (1,), (0, 1)), ("x", FUSED.id, "o1", 1, (1,), 0, (1,), (0, 1))]
    assert sites.unwired == () and sites.circuit.cuts == {descriptor_id(FUSED): {"parts": 2}}
    assert sites.consumers == {(0, r, 2): (sites.units[0, 0, TW.CONSUMER, r][0],) for r in (0, 1)}


@pytest.mark.parametrize("reads, rows", [(((2, 0, 0, 0, 2 * N),), ["o0", "o1"]), (((2, 0, N, 0, N), (2, 0, 0, N, 2 * N)), ["o1", "o0"])],
                         ids=["both rows in order", "the normed row first"])
def test_a_call_reading_both_rows_of_a_fused_call_has_its_port_cut_in_two_one_row_from_each_output(reads, rows):
    """Both ranks' calls read the one unit, so each row is one wire over the two reader units."""
    sites = TW.sites_circuit([_binding([_site(0, 1, 0, *reads)], {0: SUM, 1: FUSED, 2: GATHER})], NAME)
    assert _into(sites, GATHER) == [("parts", FUSED.id, out, 0, (2,), 0, (0,), (c, 1)) for c, out in enumerate(rows)]
    assert sites.unwired == () and sites.circuit.cuts[descriptor_id(GATHER)] == {"parts": 2}


def test_a_read_across_a_fused_calls_two_rows_into_a_port_neither_fits_is_listed_row_by_row():
    half = bind(moe.MoeSum, TOPK=3, H=8)
    sites = TW.sites_circuit([_binding([_site(0, 1, 0, (2, 0, 0, 0, 2 * N))], {0: SUM, 1: FUSED, 2: half})], NAME)
    assert sites.unwired == tuple(f"{_what(0, r, 2, half, 'y')} reads leaves [{lo}, {lo + N}): 16-word rows of {descriptor_id(FUSED)} into its 24-word y "
                                  f"from word {lo}" for r in (0, 1) for lo in (0, N))
    assert _into(sites, half) == []


# ---------------------------------------------------------------------------------------------------- reads the grains do not fit


@pytest.mark.parametrize("reader, read, port, why", [
    (GEMM, (2, 0, 0, 4, 12), "x", "[4, 12): part of 16-word rows of AllReduce_v2{WORLD=2,N=16}"),
    (GEMM, (2, 0, 0, 0, 8), "x", "[0, 8): part of 16-word rows of AllReduce_v2{WORLD=2,N=16}"),
    (GEMM, (2, 1, 8, 0, N), "w", "[0, 16): 16-word rows of AllReduce_v2{WORLD=2,N=16} into its 256-word w from word 8"),
    (bind(moe.MoeSum, TOPK=3, H=8), (2, 0, 0, 0, N), "y", "[0, 16): 16-word rows of AllReduce_v2{WORLD=2,N=16} into its 24-word y from word 0"),
], ids=["the middle of a row", "the first half of a row", "from a word that is no row's start", "a port that is not whole rows"])
def test_a_read_that_is_not_whole_writer_rows_into_whole_rows_of_its_port_is_a_free_input_listed_by_name(reader, read, port, why):
    """The site's combine is at row grain, so its output is one 16-word writer row whatever reads it.  The reading call is
    still a unit; nothing is wired into it."""
    sites = TW.sites_circuit([_binding([_site(0, 1, 0, read)], {0: SUM, 1: REDUCE, 2: reader})], NAME)
    assert sites.unwired == tuple(f"{_what(0, r, 2, reader, port)} reads leaves {why}" for r in (0, 1))
    assert _units(sites) == {**SUMS, (TW.COMBINE, TW.ALL): [(REDUCE.id, 0, 1)], (TW.CONSUMER, 0): [(reader.id, 0, 1)], (TW.CONSUMER, 1): [(reader.id, 1, 1)]}
    assert _into(sites, reader) == [] and descriptor_id(reader) not in sites.circuit.cuts


def test_a_row_into_a_port_from_a_rows_start_is_wired_where_the_same_row_from_another_word_is_not():
    """Rank 1's call reads the row into its 16-word port, rank 0's into the middle of its 256-word one."""
    sites = TW.sites_circuit([_binding([_site(0, 1, 0, by_rank=(((2, 1, 8, 0, N),), ((2, 0, 0, 0, N),)))], {0: SUM, 1: REDUCE, 2: GEMM})], NAME)
    assert _into(sites, GEMM) == [("x", REDUCE.id, "o0", 1, (1,), 0, (1,), (0, 1))]
    assert sites.unwired == (f"{_what(0, 0, 2, GEMM, 'w')} reads leaves [0, 16): 16-word rows of AllReduce_v2{{WORLD=2,N=16}} into its 256-word w from word 8",)


def _two_rows(*reads: tuple[int, int, int, int, int]) -> TB.RequestBinding:
    """One all-reduce of two rows a rank, each row its own unit (nodes 0 and 1), read by ``reads`` at each rank."""
    site = TB.Site(0, 0, 2 * N, ((2, "parts"),) * 2, (((0, 0, N), (1, 0, N)),) * 2, (reads,) * 2)
    return _binding([site], {0: SUM, 1: SUM, 2: REDUCE, 3: GEMM})


def test_a_site_of_two_rows_a_rank_is_one_run_of_two_combine_units_and_a_read_of_its_second_row_is_wired_from_the_second():
    sites = TW.sites_circuit([_two_rows((3, 0, 0, N, 2 * N))], NAME)
    assert _units(sites)[TW.COMBINE, TW.ALL] == [(REDUCE.id, 0, 2)] and _units(sites)[TW.PRODUCER, 1] == [(SUM.id, 2, 1), (SUM.id, 3, 1)]
    assert _into(sites, REDUCE) == [("parts", SUM.id, "o0", 0, (2,), 0, (1,), (0, 1)), ("parts", SUM.id, "o0", 0, (2,), 2, (1,), (1, 1))]
    assert _into(sites, GEMM) == [("x", REDUCE.id, "o0", 0, (2,), 1, (0,), (0, 1))] and sites.unwired == ()


def test_a_read_of_a_rows_width_that_lies_across_two_rows_is_part_of_each_and_listed():
    sites = TW.sites_circuit([_two_rows((3, 0, 0, 8, N + 8))], NAME)
    assert sites.unwired == tuple(f"{_what(0, r, 3, GEMM, 'x', 2 * N)} reads leaves [8, 24): part of 16-word rows of AllReduce_v2{{WORLD=2,N=16}}"
                                  for r in (0, 1))
    assert _into(sites, GEMM) == []


def _two_widths(first: Any, second: Any, at: int) -> TB.RequestBinding:
    """Two all-reduces, the first of rows that are their own units (one 16-word writer row), the second of words read by
    eights (two 8-word writer rows); node 5 reads the first's output and node 7 the second's, each whole, into argument
    ``at``.  With ``first`` and ``second`` fused calls, 5 and 7 are two more sites' combines, else each rank's two calls."""
    fns = {0: SUM, 1: REDUCE, 2: GEMM, 3: REDUCE, 4: SUM, 5: first, 6: SUM, 7: second, 8: EIGHT}
    sites = [_site(0, 1, 0, (5, at, 0, 0, N)), _site(1, 3, 2, (7, at, 0, 0, N), (8, 0, 0, 0, 8))]
    return _binding(sites + ([_site(2, 5, 4), _site(3, 7, 6)] if first is FUSED else []), fns)


def test_a_port_of_two_sites_combine_units_that_two_reads_cut_two_ways_keeps_the_first_cut_and_the_other_read_is_listed():
    """Two fused calls of one template, each its own unit: the first's residual is one 16-word row, the second's two 8-word
    rows.  Every unit of a template has one cut, so the second's residual is a free input."""
    sites = TW.sites_circuit([_two_widths(FUSED, FUSED, 1)], NAME)
    assert _units(sites, site=1)[TW.COMBINE, TW.ALL] == [("AllReduce_v2{WORLD=2,N=8}", 0, 2)]
    assert [_units(sites, site=j)[TW.COMBINE, TW.ALL] for j in (2, 3)] == [[(FUSED.id, 0, 1)], [(FUSED.id, 1, 1)]]
    assert [w for w in _into(sites, FUSED) if w[0] == "res"] == [("res", REDUCE.id, "o0", 0, (1, 1), 0, (0, 1), (0, 1))]
    why = f"8-word rows of AllReduce_v2{{WORLD=2,N=8}}, and {descriptor_id(FUSED)}'s res is cut 1 ways"
    assert sites.unwired == (f"{_what(1, 0, 7, FUSED, 'res')} reads leaves [0, 16): {why}",)
    assert sites.circuit.cuts[descriptor_id(FUSED)] == {"parts": 2}


def test_a_port_of_two_calls_of_one_definition_that_two_reads_cut_two_ways_keeps_the_first_cut_and_the_other_read_is_listed():
    """Each rank's first call reads one 16-word row into its 32-word port, its second two 8-word rows into the same port of
    the same template."""
    sites = TW.sites_circuit([_two_widths(GATHER, GATHER, 0)], NAME)
    assert _units(sites, site=0)[TW.CONSUMER, 0] == [(GATHER.id, 0, 1)] and _units(sites, site=1)[TW.CONSUMER, 0] == [(GATHER.id, 2, 1), (EIGHT.id, 0, 1)]
    assert _into(sites, GATHER) == [("parts", REDUCE.id, "o0", 0, (2,), 0, (0,), (0, 1))]
    assert _into(sites, EIGHT) == [("parts", "AllReduce_v2{WORLD=2,N=8}", "o0", 0, (2,), 0, (0,), (0, 1))]
    assert sites.unwired == tuple(f"{_what(1, r, 7, GATHER, 'parts')} reads leaves [0, 16): AllGather_v1{{WORLD=2,N=16}}'s parts is cut 2 ways, not 4"
                                  for r in (0, 1))
    assert sites.circuit.cuts[descriptor_id(GATHER)] == {"parts": 2}


def test_a_gather_that_reads_a_combines_output_has_no_unit_to_read_it_and_is_listed_once():
    """The reading node is the next site's combine, an all-gather, which has no combine unit.  It is named by its collective's
    key and by the place of the site it combines; the read is every rank's, listed at the first."""
    sites = TW.sites_circuit([_binding([_site(5, 1, 0, (3, 0, 0, 0, N)), _site(9, 3, 2)], {0: SUM, 1: REDUCE, 2: SUM, 3: GATHER})], NAME)
    assert sites.unwired == (f"{_what(5, 0, 3, GATHER, 'parts')} reads leaves [0, 16): no unit of collective 1 reads them",)
    assert _units(sites, site=1) == {(TW.PRODUCER, 0): [(SUM.id, 2, 1)], (TW.PRODUCER, 1): [(SUM.id, 3, 1)]} and sites.consumers == {}


@pytest.mark.parametrize("at, wired, why", [
    (0, [("x", REDUCE.id, "o0", 0, (2, N), 0, (0, 0), (0, 1))], None),
    (1, [], f"w reads leaves [0, 16): its producer unit {descriptor_id(WORD)} has no 256-word w"),
], ids=["its 16-word x", "its 256-word w"])
def test_a_later_sites_producer_reads_a_row_in_each_of_its_units_where_the_unit_has_the_port_at_the_calls_width(at, wired, why):
    """The producer is a Gemm, its units one a coordinate: each of the 32 reads the 16-word x whole, and has 16 words of the
    call's 256 of w."""
    sites = TW.sites_circuit([_binding([_site(0, 1, 0, (2, at, 0, 0, N)), _site(1, 3, 2)], {0: SUM, 1: REDUCE, 2: GEMM, 3: REDUCE})], NAME)
    assert _units(sites, site=1)[TW.PRODUCER, 0] == [(WORD.id, 0, N)] and _units(sites, site=1)[TW.PRODUCER, 1] == [(WORD.id, N, N)]
    assert _into(sites, WORD) == wired and sites.consumers == {}
    assert sites.unwired == (() if why is None else tuple(f"{_what(0, r, 2, GEMM, why)}" for r in (0, 1)))


def test_a_call_that_reads_two_sites_is_one_unit_placed_at_the_later_site_and_wired_from_both():
    """Each rank's gather (node 4) reads the first site's row into the first half of its port and the second site's into the
    second half."""
    sites = TW.sites_circuit([_binding([_site(0, 1, 0, (4, 0, 0, 0, N)), _site(1, 3, 2, (4, 0, N, 0, N))],
                                       {0: SUM, 1: REDUCE, 2: SUM, 3: REDUCE, 4: GATHER})], NAME)
    assert [stage for stage, _ in _units(sites, site=0)] == [TW.PRODUCER, TW.PRODUCER, TW.COMBINE]
    assert [_units(sites, site=1)[TW.CONSUMER, r] for r in (0, 1)] == [[(GATHER.id, 0, 1)], [(GATHER.id, 1, 1)]]
    assert _into(sites, GATHER) == [("parts", REDUCE.id, "o0", 0, (2,), c, (0,), (c, 1)) for c in (0, 1)]
    assert sites.unwired == () and sites.consumers == {(0, r, 4): (sites.units[0, 1, TW.CONSUMER, r][0],) for r in (0, 1)}


# ------------------------------------------------------------------------------------------------------------------- the refusals


@pytest.mark.parametrize("bindings", [[], [None], [None, None]], ids=["no request", "a request no rank ran", "two of them"])
def test_a_window_of_no_binding_is_refused_by_name(bindings):
    with pytest.raises(ValueError) as refused:
        TW.sites_circuit(bindings, NAME)
    assert str(refused.value) == "w2: no request's binding, so no collective site"


def _one(world: int = 2) -> TB.RequestBinding:
    reduce = bind(B1T.AllReduce, WORLD=world, N=N)
    return _binding([TB.Site(0, 0, N, ((1, "parts"),) * world, (((0, 0, N),),) * world, (((2, 0, 0, 0, N),),) * world)], {0: SUM, 1: reduce, 2: GEMM}, world)


@pytest.mark.parametrize("worlds, ranks, why", [
    ([2], [2], "ranks [2] of bindings of worlds [2]"), ([2], [0, 2], "ranks [0, 2] of bindings of worlds [2]"),
    ([2], [-1], "ranks [-1] of bindings of worlds [2]"), ([2], [], "ranks [] of bindings of worlds [2]"),
    ([2, 3], None, "ranks [0, 1] of bindings of worlds [2, 3]"), ([3, 2], [0], "ranks [0] of bindings of worlds [2, 3]"),
], ids=["a rank past the world", "one of the world and one past it", "a rank before the first", "no rank", "requests of two worlds",
        "requests of two worlds, one rank"])
def test_ranks_the_bindings_do_not_all_have_are_refused_by_name(worlds, ranks, why):
    with pytest.raises(ValueError) as refused:
        TW.sites_circuit([_one(w) for w in worlds], NAME, ranks)
    assert str(refused.value) == f"w2: {why}"


def test_a_request_no_rank_ran_has_no_unit_and_the_next_keeps_its_place():
    alone, after = TW.sites_circuit([_one()], NAME), TW.sites_circuit([None, _one(), None], NAME)
    assert sorted({i for i, *_ in after.units}) == [1] and list(after.consumers) == [(1, 0, 2), (1, 1, 2)]
    assert _units(after, request=1) == _units(alone) and after.circuit == alone.circuit and after.order == alone.order
    part = TW.sites_circuit([None, _binding([_site(0, 1, 0, (2, 0, 0, 4, 12))], {0: SUM, 1: REDUCE, 2: GEMM})], NAME)
    assert part.unwired[0].startswith("w2: request 1's collective 0 [0, 16): rank 0's node 2 ")


def test_ranks_named_twice_or_out_of_order_are_the_window_of_those_ranks():
    assert TW.sites_circuit([_one()], NAME, [1, 1, 0]) == TW.sites_circuit([_one()], NAME)
    one = TW.sites_circuit([_one()], NAME, [1, 1])
    assert _units(one) == {(TW.PRODUCER, 1): [(SUM.id, 0, 1)], (TW.COMBINE, TW.ALL): [(REDUCE.id, 0, 1)], (TW.CONSUMER, 1): [(GEMM.id, 0, 1)]}
    assert _into(one, REDUCE) == [("parts", SUM.id, "o0", 0, (1,), 0, (1,), (1, 1))]


#: by the later request's unit: an earlier request whose call (node 2) reads one writer row whole into a port of that unit's
#: template, the later request, the ways its unit cuts that port, and its refusal
ONE_ROW = {
    "gather": (_binding([_site(0, 1, 0, (2, 0, 0, 0, N))], {0: SUM, 1: REDUCE, 2: WORDS}), _binding([_site(0, 1, 0)], {0: GEMM, 1: REDUCE}), 16,
               "AllGather_v1{WORLD=16,N=1}'s parts is cut 1 ways, not 16"),
    "combine": (_binding([TB.Site(0, 0, 2 * N, ((1, "parts"),) * 2, (((0, 0, 2 * N),),) * 2, (((2, 0, 0, 0, 2 * N),),) * 2)],
                         {0: bind(moe.MoeSum, TOPK=3, H=2 * N), 1: bind(B1T.AllReduce, WORLD=2, N=2 * N), 2: REDUCE}),
                _binding([_site(0, 1, 0)], {0: SUM, 1: REDUCE}), 2, "AllReduce_v2{WORLD=2,N=16}'s parts is read otherwise than as 2 ranks' rows"),
    "select": (_binding([_site(0, 1, 0, (2, 1, 0, 0, 2))], {0: GEMM, 1: REDUCE, 2: CHOOSE}),
               _binding([_site(0, 1, 0, (2, 0, 0, 0, 2 * N))], {0: GEMM, 1: GATHER, 2: SELECT}), 2,
               "TokenSelectCombine_v2{V=32,C=16}'s best is cut 1 ways, not 2"),
}


@pytest.mark.parametrize("unit", ["gather", "combine", "select"])
def test_a_template_an_earlier_requests_call_read_as_one_row_cannot_be_a_later_requests_unit_that_reads_several(unit):
    """The later request needs the template's port in rows: sixteen words gathered into a row, the two ranks' partial rows of
    a combine, the best of each of a token select's two groups.  Each request alone is a window with every read wired.  The
    first cut of a port stands: with the requests the other way round the call's read is the one left a free input."""
    earlier, later, k, why = ONE_ROW[unit]
    assert TW.sites_circuit([earlier], NAME).unwired == () == TW.sites_circuit([later], NAME).unwired
    with pytest.raises(ValueError) as refused:
        TW.sites_circuit([earlier, later], NAME)
    assert str(refused.value) == f"w2: {why}"
    unwired = TW.sites_circuit([later, earlier], NAME).unwired
    assert len(unwired) == 2 and all(u.startswith("w2: request 1's collective 0 ") and u.endswith(f" is cut {k} ways, not 1") for u in unwired)
