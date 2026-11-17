"""The embedding shard joined to the prompt's all-reduce in GLM-5.3's node window at TP8 rank 0, beside every lowered family
(`node_window`, `_dense`, `_sparse`, `_multimem`, `_embed`), at a 43-id prompt, 2 outputs and one decoder layer: the prefill's
1,376 read units (43 tokens x 32 column blocks) are row 0 of the all-reduce's combine units' ``parts``, read by wire, and
only the decoder's stop check and token selection stay unlowered.

The shard's rows are 192 words, wider than the all-reduce's 96-word chunks at coordinate grain, so `node_window_multimem`
combines that site as rows: one `AllReduceMultimem_v1{SIZE=192}` unit a column block of a token, rank 0's block by wire.
"""
from __future__ import annotations

import re

import pytest
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_dense as ND
from verity_vllm.program.predict import node_window_embed as NE
from verity_vllm.program.predict import node_window_multimem as NM
from verity_vllm.program.predict import node_window_sparse as NS

from tests.program.predict.test_node_window import glm_request

FAMILIES = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NM.FAMILIES, **NE.FAMILIES}
IDS, BLOCKS, WORLD = 43, 6144 // NE.WIDTH, 8
READ = r"EmbeddingRead_v1\{V=154880,H=192\}"


@pytest.mark.slow
def test_joined():
    """The prefill's read units are row 0 of the combine's ``parts`` by one wire, block b of token t at unit ``32 t + b``
    reading read unit ``t + 43 b``; the other ranks' rows are the only free rows there, the reads' ``o0`` free nowhere,
    and the unlowered kinds exactly the stop check and token selection. About 10 s: the only test that builds the window
    with all five family sets, which no family's own test does."""
    circuit, w = NW.node_window(glm_request(IDS, 1).program.fn, families=FAMILIES)
    assert w.census(circuit)["unlowered"] == {"StopCheck": 1, "TokenSelect": 2}
    (read,) = [t for t in circuit.units if re.fullmatch(READ, t)]
    wires = [x for x in circuit.wires if x.writer == read and x.reader.startswith("AllReduceMultimem_v1")]
    assert [(x.port, x.out, x.at, x.radix, x.base, x.step, x.cut[0]) for x in wires] == [
        ("parts", "o0", 0, (IDS, BLOCKS), 0, (1, IDS), 0)]
    combine = wires[0].reader
    assert circuit.units[combine] == IDS * BLOCKS and circuit.cuts[combine] == {"parts": WORLD}
    assert sorted((f.port, f.kind, f.rows, f.at, f.count) for f in w.free if f.tid == combine) == [
        ("parts", "peer", (k, 1, WORLD), 0, IDS * BLOCKS) for k in range(1, WORLD)]
    assert not [f for f in w.free if f.family == "EmbeddingShard"]
    assert {f.port for f in w.free if f.tid == read} == {"tok"}
