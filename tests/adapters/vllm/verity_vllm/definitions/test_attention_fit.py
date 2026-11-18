"""`Q_fit` v1 on `AttentionHead_v19` at Llama-3.2-1B's heads (D 64, kBlockN 192), its keys one input node registered a key a
row and its values one registered 16 columns a row: each node of a key block's softmax step writes rows of its own, so the
step does not fit one proof block, and a cut of it after its score dots reads no row past 111 B, so `build` refuses it."""
from __future__ import annotations

import functools

import pytest
from verity.circuits import fit as FT
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.parts import Part
from verity.circuits.program import Program
from verity.circuits.types import Array

from verity_vllm.definitions.registry import dead_free as DF
from verity_vllm.definitions.registry import hopper
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry.prims import BF16

D, BN, VS, T = 64, 192, 16, 200
REGISTERED = {1: D, 2: VS}


@functools.cache
def head() -> Program:
    fn = bind(DF.AttentionHeadV19, T=T, D=D, BN=BN, MASKED_FROM=0, DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum)
    params = (("q", Array(D, BF16)), ("k", Array(T, Array(D, BF16))), ("v", Array(T, Array(D, BF16))))
    return Program(CompositeDefinition("FitAttentionHead", 1, (), lambda S: (params, Array(D, BF16)),
                                       lambda B, S, q, k, v: B.call(fn, q, k, v), register=False))


def test_the_full_key_blocks_softmax_step_is_refused_unpinned():
    with pytest.raises(FT.Unfit) as e:
        FT.build(head(), registered=REGISTERED)
    assert e.value.code == "fit-pin" and "[[3, 0], [1, 0]]" in str(e.value)


def test_the_softmax_step_whole_is_past_the_rows():
    prog = head()
    fp, fit = FT.FitPartition(prog, FT.query()), FT.Fit(prog, REGISTERED)
    first = FT.part_of(prog, fp, {"path": [[3, 0]], "nodes": [0, 1]})
    fit.finish(first, fit.measure(first))
    blk = prog.circuit.root_scope.child(3, 0).child(1, 0)
    nodes = blk.spec.body.nodes
    dot = hopper.HopperBF16WgmmaDot16.id
    s = min(j for j, n in enumerate(nodes) if n.fn.id == dot)
    step = Part(blk, frozenset((j, m) for j in range(s) for m in range(nodes[j].n if nodes[j].form == "batch" else 1)))
    m = fit.measure(step)
    assert m.rows > FT.MAX_ROWS and m.pinned
