"""verity_vllm.program.predict.node_window_join: the eight ranks' windows of GLM-5.3's request Program at TP8 and their join
(`rank_window`, `join`). Every rank's window builds (the all-reduce's own segment and the select's own lanes are the rank's,
`node_window_multimem.mine`); joined, every rank's units follow the rank before's, every peer input is wired from its
writer rank's own partial, every rank's prompt reads are its boundary reads (`node_window_reads`), and no input stays
free. `_slice` cuts a pattern's indices; `free_rows` and `ranks_of` write what proofs' readers take.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
from types import SimpleNamespace

import pytest
from verity.circuits.codec import canonical_json
from verity.commitments.periods import interface as IF
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_join as NJ
from verity_vllm.program.predict import node_window_dense as ND
from verity_vllm.program.predict import node_window_embed as NE
from verity_vllm.program.predict import node_window_multimem as NM
from verity_vllm.program.predict import node_window_reads as NR
from verity_vllm.program.predict import node_window_select as NSEL
from verity_vllm.program.predict import node_window_sparse as NS
from verity_vllm.program.predict.build import Unexpressible

from tests.program.predict.test_node_window import glm_request

FAMILIES = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NM.FAMILIES, **NE.FAMILIES, **NSEL.FAMILIES}
IDS, WORLD = 43, 8


def _sources(radix: tuple[int, ...], step: tuple[int, ...], base: int, i: int) -> int:
    out = base
    for r, s in zip(reversed(radix), reversed(step)):
        out, i = out + (i % r) * s, i // r
    return out


def test_slice():
    """Each cut of a pattern's indices [a, b) is patterns reading the same sources at the same indices, in order."""
    for radix, step in (((3,), (2,)), ((2, 3), (5, 1)), ((3, 1, 4), (-2, 7, 1)), ((2, 2, 2), (0, 3, 1))):
        n = math.prod(radix)
        for a, b in itertools.combinations(range(n + 1), 2):
            got = [(i0 + i, _sources(r, step[len(step) - len(r):], s, i))
                   for r, s, i0 in NJ._slice(radix, step, 9, a, b) for i in range(math.prod(r))]
            assert got == [(i, _sources(radix, step, 9, i)) for i in range(a, b)]


@pytest.mark.slow
def test_eight_ranks_joined():
    """GLM-5.3's first decoder layer at each of the eight ranks, its window, and the eight joined: each rank as many units
    of each template; the joined circuit passes `interface.Circuit`'s checks and keeps every rank's units, wires every peer
    input of every rank and leaves none unwired, takes each rank's boundary reads (its prompt's, and rank 0's served ids) at
    that rank's units, and leaves no input free; its ranks (`ranks_of`) are `interface.Ranks`, each window its rank's H, and
    fit the circuit (`interface.check_ranks`: each join wire's reader units one rank's), stated on it or checked alone.
    About 90 s: the only test that builds a rank other than 0 and the join."""
    ranks = [NJ.rank_window(r, glm_request(IDS, 1, rank=r).program.fn, FAMILIES, reads=NR.boundary_reads)
             for r in range(WORLD)]
    assert len({tuple(sorted(x.circuit.units.items())) for x in ranks}) == 1
    assert len({x.census["digest"] for x in ranks}) == WORLD
    j = NJ.join(ranks)
    assert isinstance(j.circuit, IF.Circuit) and not j.unwired
    assert j.circuit.units == {t: WORLD * n for t, n in ranks[0].circuit.units.items()}
    assert j.circuit.external == ranks[0].circuit.external
    peers = [(x.rank, f) for x in ranks for f in x.free if f.kind == "peer"]
    assert len(j.wired) == len(peers) and {(r, f.name) for r, f in j.wired} == {(r, f.name) for r, f in peers}
    assert j.free == () and j.circuit.ingress == ()
    assert [(r.reader, r.port, r.param, r.at) for r in j.circuit.boundary_reads] == [
        (r.reader, r.port, r.param, r.at + j.first[x.rank][r.reader]) for x in ranks for r in x.circuit.boundary_reads]
    served = [r for r in j.circuit.boundary_reads if r.param == NR.EGRESS]
    assert served and served == [r for r in ranks[0].circuit.boundary_reads if r.param == NR.EGRESS]
    stated = IF.Ranks.of_doc(NJ.ranks_of(ranks, j, 1))
    assert [r.window for r in stated.of] == [
        hashlib.sha512(canonical_json(IF.circuit_of(IF.descriptor_record(x.circuit, 1)))).hexdigest() for x in ranks]
    IF.check_ranks(j.circuit.layouts, stated, j.circuit.wires, j.circuit.registered)
    assert dataclasses.replace(j.circuit, ranks=stated).description()["ranks"] == stated.doc()
    assert all(j.first[r][t] == r * n for r in range(WORLD) for t, n in ranks[0].circuit.units.items())


def test_free_rows():
    """`free_rows` writes each free row as `interface.free_of` reads it: a peer row of a port the record cuts eight ways at
    that cut (``cut``), a whole weights port with none, half a port cut four ways as its two rows. Refused by name: the
    prompt's rows where no boundary read declares them, and rows that aren't whole rows of the record's cut."""
    def joined(*free):
        return NJ.Joined(SimpleNamespace(cuts={"A": {"p": 8}, "C": {"x": 4}}), {}, free, (), ())

    got = NJ.free_rows(joined(NW.Free("A", "p", 3, 5, "peer", "peers", "r1/peers_0", (2, 1, 8)),
                              NW.Free("B", "w", 0, 4, "param", NJ.WEIGHTS, f"r0/{NJ.WEIGHTS}"),
                              NW.Free("C", "x", 1, 2, "peer", "peers", "r2/peers_3", (0, 1, 2))))
    assert got == [{"reader": "A", "port": "p", "at": 3, "units": 5, "kind": "peer", "cut": [2, 1]},
                   {"reader": "B", "port": "w", "at": 0, "units": 4, "kind": "weights"},
                   {"reader": "C", "port": "x", "at": 1, "units": 2, "kind": "peer", "cut": [0, 2]}]
    assert [f.doc() for f in IF.free_of({"free": got})] == got
    with pytest.raises(Unexpressible, match="boundary read"):
        NJ.free_rows(joined(NW.Free("A", "tok", 0, 2, "param", NW.PROMPT, f"r0/{NW.PROMPT}")))
    with pytest.raises(Unexpressible, match="whole rows"):
        NJ.free_rows(joined(NW.Free("C", "x", 0, 1, "peer", "peers", "r1/peers_0", (1, 1, 3))))
