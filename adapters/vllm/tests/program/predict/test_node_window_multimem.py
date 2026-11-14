"""verity_vllm.program.predict.node_window_multimem: the symmetric-memory all-reduce (`AllReduceMultimem_v1`, the prefill's
past FlashInfer's 42 tokens) lowered into GLM-5.3's node window at TP8 rank 0, at a 43-id prompt, 2 outputs and one decoder
layer, beside the frame's and the dense families. At coordinate grain its units are `pipeline.tp.grain.lower`'s for the
site; the norms read its 192-column blocks through one more gather level. Rank 0's partial is row 0 of each combine's
``parts`` cut 8 ways and is wired (free where its producer, the embedding shard, has no lowering); the other ranks' rows
are free inputs listed by name (each site's own ``peers_*``, as the Program names them).
"""
from __future__ import annotations

import re
from typing import Any

import pytest
from verity_vllm.pipeline.tp import grain as GR
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_dense as ND
from verity_vllm.program.predict import node_window_multimem as NM

from tests.program.predict.test_node_window import glm_request
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.refs import runs
from verity.commitments.windows import interface as IF
from verity.definitions.registry import b1_tp2 as B1T

FAMILIES = {**NW.FAMILIES, **ND.FAMILIES, **NM.FAMILIES}
LP, WORLD, WIDTH, COLS = 43, 8, 6144, 192


@pytest.fixture(scope="module")
def program():
    return glm_request(LP, 1).program.fn


@pytest.fixture(scope="module")
def window(program):
    return NW.node_window(program, families=FAMILIES)


def _sites(w: NW.NodeWindow) -> list[tuple[int, Any]]:
    return [(j, n) for j, n in enumerate(w.fn.body.nodes) if NW._family(n) == NM.FAMILY]


def _peers(w: NW.NodeWindow, tid: str) -> set[tuple[int, str, tuple[int, int, int]]]:
    """Each of ``tid``'s allocations' rows r > 0 of ``parts``, as the Program names the input it reads: ``(first unit,
    input, (r, 1, WORLD))``."""
    out: set[tuple[int, str, tuple[int, int, int]]] = set()
    for at, _n, _f in w.allocs[tid]:
        node = w.fn.body.nodes[w.node_of[(tid, at)]]
        size = int(node.fn.bindings["SIZE"])
        for r in range(1, WORLD):
            ((space, idx, _lo, _hi),) = runs(node.args[0].refs.slice(r * size, (r + 1) * size))
            assert space != "n"
            out.add((at, w.params[idx], (r, 1, WORLD)))
    return out


def _pairs(circuit: IF.Circuit) -> set[tuple[str, str, str]]:
    return {(x.writer, x.reader, x.port) for x in circuit.wires}


def test_units_are_grain_lowers(window):
    """Each site's units are `grain.lower`'s (rank 0's words gathered in chunks of K, a combine a chunk, an output gather a
    token row) and one `AllGather_v1{WORLD=192/K, N=K}` a 192-column block; the norms' apply units read those blocks."""
    circuit, w = window
    sites = _sites(w)
    assert sites
    ids = {descriptor_id(fn): tid for tid, fn in w.b.fns.items()}
    combines: dict[str, int] = {}
    for j, node in sites:
        size = int(node.fn.bindings["SIZE"])
        low = GR.lower(node.fn, [NM._own(w, j, node, size, NM.mine(w, j, node, size, WORLD))] * WORLD, NM.reads(w, j), "test")
        assert low.grain == GR.COORD and (low.row, low.chunk) == (WIDTH, 96)
        (fn, n), = low.combines
        assert n == size // low.chunk and descriptor_id(low.output) in ids and descriptor_id(low.gather) in ids
        combines[ids[descriptor_id(fn)]] = combines.get(ids[descriptor_id(fn)], 0) + n
    assert combines == {t: circuit.units[t] for t in combines}
    (tc,) = combines
    assert re.fullmatch(r"AllReduceMultimem_v1\{WORLD=8,SIZE=96,.*\}", tc)
    block = ids[descriptor_id(bind(B1T.AllGather, WORLD=2, N=96))]
    gather, output = (ids[descriptor_id(bind(B1T.AllGather, WORLD=k, N=n))] for k, n in ((96, 1), (64, 96)))
    apply = [t for t in circuit.units if re.match(r"RMSNorm\w*Apply", t)]
    assert {(gather, tc, "parts"), (tc, output, "parts"), (tc, block, "parts")} <= _pairs(circuit)
    assert any((block, a, "x") in _pairs(circuit) for a in apply)
    assert all(x.cut == (0, 1) for x in circuit.wires if x.reader == tc)


def test_peers_free_by_name(window):
    """The 7 other ranks' rows of each combine's ``parts`` are free inputs named as the Program names the site's partials
    (``peers_*``), at their rows of the 8-way cut;
    rank 0's row is free only at the embedding's site, under the embedding shard (no lowering yet); nothing is free under
    the all-reduce, and it is lowered."""
    circuit, w = window
    tc = next(t for t in circuit.units if t.startswith("AllReduceMultimem_v1"))
    assert circuit.cuts[tc] == {"parts": WORLD}
    free = [f for f in w.free if f.tid == tc]
    peers = {(f.at, f.name, f.rows) for f in free if f.kind == "peer"}
    assert peers == _peers(w, tc) and all(name.startswith(NW.PEERS) for _a, name, _r in peers)
    assert {(f.family, f.rows) for f in free if f.kind != "peer"} == {("EmbeddingShard", (0, 1, WORLD))}
    census = w.census(circuit)
    assert NM.FAMILY not in census["unlowered"] and NM.FAMILY not in census["free"]
    for tid, fn in w.b.fns.items():
        IF.check_proof_block(fn, circuit.cuts.get(tid))


def test_stand_ins(program, window):
    """With ``stand_ins`` (the value check's) the window is the same, each free row still free."""
    circuit, w = window
    c2, w2 = NW.node_window(program, families=FAMILIES, stand_ins=True)
    assert w2.census(c2)["digest"] == w.census(circuit)["digest"]
    assert {(f.tid, f.port, f.at, f.rows) for f in w2.free} == {(f.tid, f.port, f.at, f.rows) for f in w.free}


def test_moe_row_grain():
    """Over an MoE layer the final all-reduce's own partial is the residual add's rows (row grain): 192-column blocks of
    each token row, one `AllReduceMultimem_v1{SIZE=192}` a block reading the residual add's block as its row 0."""
    circuit, w = NW.node_window(glm_request(LP, 1, moe=True).program.fn, families=FAMILIES)
    add = next(t for t in circuit.units if re.fullmatch(r"ResidualAdd_v1\{N=192\}", t))
    (tr,) = [t for t in circuit.units if re.fullmatch(r"AllReduceMultimem_v1\{WORLD=8,SIZE=192,.*\}", t)]
    assert circuit.units[tr] == LP * WIDTH // COLS
    assert [(x.writer, x.cut) for x in circuit.wires if x.reader == tr] == [(add, (0, 1))]
    assert {(f.at, f.name, f.rows) for f in w.free if f.tid == tr} == _peers(w, tr)
    census = w.census(circuit)
    assert NM.FAMILY not in census["unlowered"] and NM.FAMILY not in census["free"]
    for tid, fn in w.b.fns.items():
        IF.check_proof_block(fn, circuit.cuts.get(tid))
