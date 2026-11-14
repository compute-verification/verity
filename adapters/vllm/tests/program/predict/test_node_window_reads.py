"""verity_vllm.program.predict.node_window_reads: a node window's boundary reads, the two the class's Definition reads first,
in GLM-5.3's window at TP8 at a 32-id prompt, 2 outputs and one decoder layer: the prefill's embedding reads take the prompt's
ids (block b of token t at unit ``32 b + t``, id t at its leaves ``2 t`` and ``2 t + 1``), and rank 0's select combines
write the served ids, output k at leaves ``2 k`` and ``2 k + 1``. Another rank's window reads the prompt and writes none."""
from __future__ import annotations

import dataclasses

import pytest
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_embed as NE
from verity_vllm.program.predict import node_window_reads as NR
from verity_vllm.program.predict.build import Unexpressible

from tests.program.predict.test_node_window import LP, glm_request
from tests.program.predict.test_node_window_select import FAMILIES as SELECT
from verity.commitments.windows import interface as IF

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


def test_the_window_declares_them(rank0):
    """Rank 0's circuit with its boundary reads passes `interface.Circuit`'s checks (`check_inputs`: each on its template's
    port and units, no row declared twice) and records them as its ``boundary_reads``; the served ids declared twice don't."""
    circuit, w = rank0
    reads = tuple(NR.boundary_reads(w, 0))
    assert dataclasses.replace(circuit, boundary_reads=reads).description()["boundary_reads"] == [r.doc() for r in reads]
    with pytest.raises(ValueError, match="both fill rows"):
        dataclasses.replace(circuit, boundary_reads=(*reads, *NR.served_reads(w, 0)))


def test_refused_by_name():
    """A window built without stand-ins keeps no rows for what it leaves free, so its prompt reads aren't known."""
    _circuit, w = _window(0, stand_ins=False)
    with pytest.raises(Unexpressible, match="stand_ins"):
        NR.prompt_reads(w)
