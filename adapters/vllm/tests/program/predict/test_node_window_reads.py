"""verity_vllm.program.predict.node_window_reads: a node window's boundary reads, the two the class's Definition reads first,
in GLM-5.3's window at TP8 at a 32-id prompt, 2 outputs and one decoder layer: the prefill's embedding reads take the prompt's
ids (block b of token t at unit ``32 b + t``, id t at its leaves ``2 t`` and ``2 t + 1``), and rank 0's select combines
write the served ids, output k at leaves ``2 k`` and ``2 k + 1``. Another rank's window reads the prompt and writes none.
Built on them (`rank_window`), each rank's window with every family lowered leaves only its peer inputs free."""
from __future__ import annotations

import dataclasses

import pytest
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_embed as NE
from verity_vllm.program.predict import node_window_join as NJ
from verity_vllm.program.predict import node_window_reads as NR
from verity_vllm.program.predict.build import Unexpressible

from tests.program.predict.test_node_window import LP, glm_request
from tests.program.predict.test_node_window_ranks import FAMILIES as LOWERED
from tests.program.predict.test_node_window_select import FAMILIES as SELECT
from verity.commitments.periods import interface as IF

FAMILIES = {**SELECT, **NE.FAMILIES}
BLOCKS, OUTPUTS = 6144 // NE.WIDTH, 2
READ, COMBINE = "EmbeddingRead_v1{V=154880,H=192}", "GreedyCombine_v2{V=154880,N=7744}"


def _window(rank: int, stand_ins: bool = True):
    fn = glm_request(LP, 1, rank=rank).program.fn
    return NW.node_window(fn, families=FAMILIES, stand_ins=stand_ins)


@pytest.fixture(scope="module")
def rank0():
    return _window(0)


@pytest.fixture(scope="module")
def rank1():
    return _window(1)


def _source(read: IF.Read, unit: int) -> int:
    return int(read.source([unit - read.at])[0])


def test_the_prompt_read(rank0):
    """One read: unit ``32 b + t`` of the prefill's embedding reads takes the prompt's id t, ``radix [32, 32]``, ``step [0, 2]``;
    its units are the ones the window leaves free under the prompt, and the decode's reads take theirs by wire."""
    circuit, w = rank0
    (read,) = NR.prompt_reads(w)
    assert read.doc() == {"reader": READ, "port": "tok", "param": NR.PROMPT, "at": 0, "radix": [BLOCKS, LP], "base": 0,
                          "step": [0, 2]}
    assert all(_source(read, b * LP + t) == 2 * t for b in range(BLOCKS) for t in range(LP))
    assert {(f.tid, f.port, f.at, f.count) for f in w.free if f.name == NW.PROMPT} == {(READ, "tok", 0, BLOCKS * LP)}
    assert circuit.units[READ] == (LP + 1) * BLOCKS


def test_the_served_ids(rank0):
    """One read: the combine's ``o0`` of output k at leaf ``2 k`` of the return, ``radix [2]``, ``step [2]``."""
    _circuit, w = rank0
    (read,) = NR.served_reads(w, 0)
    assert read.doc() == {"reader": COMBINE, "port": "o0", "param": NR.EGRESS, "at": 0, "radix": [OUTPUTS], "base": 0,
                          "step": [2]}


def test_rank_0_alone_writes_the_served_ids(rank0, rank1):
    """Rank 1's window computes the same select (its combine's two units) and declares no read of it; its prompt read is
    rank 0's at its own units. Over the ranks' windows together each leaf of the return is written once."""
    (c1, w1), (c0, w0) = rank1, rank0
    assert c1.units[COMBINE] == c0.units[COMBINE] == OUTPUTS
    assert NR.served_reads(w1, 1) == []
    assert NR.prompt_reads(w1) == NR.prompt_reads(w0)
    reads = [(c0, NR.boundary_reads(w0, 0)), (c1, NR.boundary_reads(w1, 1))]
    NR.written_once(reads, 2 * OUTPUTS)
    with pytest.raises(ValueError, match="written by 2 reads"):
        NR.written_once([*reads, (c0, NR.served_reads(w0, 0))], 2 * OUTPUTS)
    with pytest.raises(ValueError, match="written by none"):
        NR.written_once(reads[1:], 2 * OUTPUTS)
    with pytest.raises(ValueError, match="past the return's"):
        NR.written_once(reads, 2 * OUTPUTS - 1)


@pytest.fixture(scope="module")
def ranks():
    return [NR.rank_window(r, glm_request(LP, 1, rank=r).program.fn, LOWERED) for r in (0, 1)]


def test_the_window_declares_them(ranks):
    """Each rank's window built on its boundary reads (`rank_window`; PRF-618: no ingress on the prompt's rows) refuses
    none: rank 0's takes the prompt's read and the served ids together as its ``boundary_reads``, rank 1's the prompt's
    alone, and over the two each leaf of the return is written once. At one layer with every family lowered, all either
    leaves free is the peer inputs, which `node_window_join.join` wires (`test_node_window_ranks`). With the served ids
    twice, rank 0's circuit is refused (`check_inputs`: no row declared twice)."""
    x0, x1 = ranks
    assert x0.circuit.ingress == x1.circuit.ingress == ()
    params = [[r.param for r in x.circuit.boundary_reads] for x in ranks]
    assert sorted(params[0]) == [NR.PROMPT, NR.EGRESS] and params[1] == [NR.PROMPT]
    assert x0.circuit.description()["boundary_reads"] == [r.doc() for r in x0.circuit.boundary_reads]
    assert x0.free and {f.kind for x in ranks for f in x.free} == {"peer"}
    NR.written_once([(x.circuit, x.circuit.boundary_reads) for x in ranks], 2 * OUTPUTS)
    served = [r for r in x0.circuit.boundary_reads if r.param == NR.EGRESS]
    with pytest.raises(ValueError, match="both fill rows"):
        dataclasses.replace(x0.circuit, boundary_reads=(*x0.circuit.boundary_reads, *served))


def test_reads_fill_a_run_together():
    """`node_window_join.unfilled`: a run of free rows is filled by whole-row reads of its port that hold each of its units,
    one read or several together; a gap, a read of part of a row, another port and a free run of part of a row leave it
    free."""
    run = NW.Free(READ, "tok", 4, 8, "param", NW.PROMPT, NW.PROMPT)

    def read(at, units, port="tok", cut=(0, 1)):
        return IF.Read(READ, port, NR.PROMPT, at, (units,), 0, (2,), cut)

    assert NJ.unfilled([run], [read(0, 6), read(6, 6)]) == ()
    assert NJ.unfilled([run], [read(4, 8)]) == ()
    assert NJ.unfilled([run], [read(0, 6), read(7, 6)]) == (run,)
    assert NJ.unfilled([run], [read(0, 6), read(6, 6, cut=(0, 2))]) == (run,)
    assert NJ.unfilled([run], [read(0, 12, port="ids")]) == (run,)
    part = dataclasses.replace(run, rows=(0, 1, 2))
    assert NJ.unfilled([part], [read(0, 12)]) == (part,)


@pytest.mark.slow
def test_a_prompt_of_two_prefill_steps():
    """At a prompt served in two prefill steps the prompt's reads coalesce into runs that cross the steps' runs of free
    rows, so no one read holds a step's run; together they hold both, and the window leaves only its peer inputs free.
    About 5 s: the only test that builds a window over more than one prefill step."""
    x = NR.rank_window(0, glm_request(2 * LP, 1, chunk=LP).program.fn, LOWERED)
    reads = [(r.at, r.reader_units) for r in x.circuit.boundary_reads if r.param == NR.PROMPT]
    assert len(reads) > 2 and reads[0] == (0, BLOCKS * LP - LP) and sum(n for _a, n in reads) == 2 * BLOCKS * LP
    assert {f.kind for f in x.free} == {"peer"}


def test_refused_by_name():
    """A window built without stand-ins keeps no rows for what it leaves free, so its prompt reads aren't known."""
    _circuit, w = _window(0, stand_ins=False)
    with pytest.raises(Unexpressible, match="stand_ins"):
        NR.prompt_reads(w)
