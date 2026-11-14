"""verity_vllm.program.predict.node_window_join: the eight ranks' windows of GLM-5.3's request Program at TP8 and their join
(`rank_window`, `join`). Every rank's window builds (the all-reduce's own segment and the select's own lanes are the rank's,
`node_window_multimem.mine`); joined, every rank's units follow the rank before's, every peer input is wired from its
writer rank's own partial, every rank's prompt reads are the request's ingress, and no input stays free. `_slice` cuts
a pattern's indices.
"""
from __future__ import annotations

import itertools
import math

import pytest
from verity.commitments.windows import interface as IF
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_join as NJ
from verity_vllm.program.predict import node_window_dense as ND
from verity_vllm.program.predict import node_window_embed as NE
from verity_vllm.program.predict import node_window_multimem as NM
from verity_vllm.program.predict import node_window_select as NSEL
from verity_vllm.program.predict import node_window_sparse as NS

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
    input of every rank and leaves none unwired, takes each rank's prompt reads as the request's ingress at that rank's
    units, and leaves no input free. About 90 s: the only test that builds a rank other than 0 and the join."""
    ranks = [NJ.rank_window(r, glm_request(IDS, 1, rank=r).program.fn, FAMILIES) for r in range(WORLD)]
    assert len({tuple(sorted(x.circuit.units.items())) for x in ranks}) == 1
    assert len({x.census["digest"] for x in ranks}) == WORLD
    j = NJ.join(ranks)
    assert isinstance(j.circuit, IF.Circuit) and not j.unwired
    assert j.circuit.units == {t: WORLD * n for t, n in ranks[0].circuit.units.items()}
    peers = [(x.rank, f) for x in ranks for f in x.free if f.kind == "peer"]
    assert len(j.wired) == len(peers) and {(r, f.name) for r, f in j.wired} == {(r, f.name) for r, f in peers}
    assert j.free == ()
    assert [(g.reader, g.at, g.units) for g in j.circuit.ingress] == [
        (g.reader, g.at + j.first[x.rank][g.reader], g.units) for x in ranks for g in x.circuit.ingress]
    assert {g.port for g in j.circuit.ingress} == {"tok"} and len(j.circuit.ingress) == WORLD
    assert all(j.first[r][t] == r * n for r in range(WORLD) for t, n in ranks[0].circuit.units.items())
