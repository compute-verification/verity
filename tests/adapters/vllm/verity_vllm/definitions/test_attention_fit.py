"""`Q_fit` v1 on FA2's head at Llama-3.2-1B's heads (D 64, kBlockN 192), its keys one input node registered a key a row and
its values one registered 16 columns a row. `AttentionHead_v21`, whose softmax step returns its small values as fields (the
max, the lane sums, the rescale factor, P in chunks of 48 keys), is cut into whole Calls of the head, each fitting a Flock
proof block with a pin, and each block restricted computes what C computes there. `AttentionHead_v19`'s key block writes a
row a node, so its softmax step does not fit one proof block and a cut of it reads no row past 111 B: `build` refuses it."""
from __future__ import annotations

import functools
import json
import random

import pytest
from verity.circuits import fit as FT
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import decode_program, descriptor_chunks
from verity.circuits.defs import CompositeDefinition, DefinitionTable, PrimitiveDefinition, bind
from verity.circuits.evaluation.reference import evaluate_region
from verity.circuits.parts import Part
from verity.circuits.program import Program, input_id
from verity.circuits.types import Array

from verity_vllm.definitions.registry import attention_key_units as AK
from verity_vllm.definitions.registry import dead_free as DF
from verity_vllm.definitions.registry import hopper
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry.prims import BF16

D, BN, VS, PK = 64, 192, 16, 48
REGISTERED = {1: D, 2: VS}
#: the class of record's most keys
MAX_T = 1087
DOT = hopper.HopperBF16WgmmaDot16


@functools.cache
def head(T: int, version: int = 21) -> Program:
    common = dict(T=T, D=D, BN=BN, MASKED_FROM=0, DOT=DOT, INV=P.Fa2InvSum)
    fn = bind(AK.AttentionHeadV21, PK=PK, VS=VS, **common) if version == 21 else bind(DF.AttentionHeadV19, **common)
    params = (("q", Array(D, BF16)), ("k", Array(T, Array(D, BF16))), ("v", Array(T, Array(D, BF16))))
    return Program(CompositeDefinition("FitAttentionHead", 1, (), lambda S: (params, Array(D, BF16)),
                                       lambda B, S, q, k, v: B.call(fn, q, k, v), register=False))


@functools.cache
def blocks(T: int) -> list:
    return FT.build(head(T), registered=REGISTERED)


def _check(T: int) -> None:
    prog = head(T)
    assert PO.verify(PO.build(prog, FT.query(), blocks(T)), prog, registered=REGISTERED).ok
    fp, fit = FT.FitPartition(prog, FT.query()), FT.Fit(prog, REGISTERED)
    n = len(prog.circuit.root_scope.child(3, 0).spec.body.nodes)
    at = 0
    for e in blocks(T):
        assert e["path"] == [[3, 0]] and e["nodes"][0] == at, e
        at = e["nodes"][1]
        p = FT.part_of(prog, fp, e)
        m = fit.measure(p)
        assert m.rows <= FT.MAX_ROWS and m.k <= FT.K_MAX and m.pinned, (e, m.rows, m.k)
        fit.finish(p, m)
    assert at == n


def _restricted_equal(T: int) -> None:
    prog = head(T)
    c = b"".join(descriptor_chunks(prog))
    obj = PO.build(prog, FT.query(), blocks(T))
    C, rng, t = prog.circuit, random.Random(T), {}
    for g in C.gates():
        if (inp := input_id(g.prim.id)) is not None:
            t[g.index] = rng.getrandbits(inp[1])
    evaluate_region(C, C[0:C.size], t)
    reg, seen = DefinitionTable(), set()

    def walk(fn) -> None:
        if fn.id in seen:
            return
        seen.add(fn.id)
        if isinstance(fn, PrimitiveDefinition):
            if input_id(fn.id) is None:
                reg.add(fn)
            return
        for nd in fn.body.nodes:
            walk(nd.fn)

    walk(prog.root)
    bits = R.Bits(R.program(c))
    for e in blocks(T):
        b = R.restrict(c, obj, e, bits, registered=REGISTERED)
        own = decode_program(json.loads(b.descriptor), reg).circuit
        mine = {g.index: t[b.to_c_gate(g.index)] for g in own.gates() if input_id(g.prim.id) is not None}
        evaluate_region(own, own[0:own.size], mine)
        assert [mine[x] for x in own.root_outputs()] == [t[g] for g in b.outputs], e


def test_the_grouped_head_is_cut_into_whole_calls_that_each_fit():
    """At 200 keys (a first block of 8, then one of 192): the first block's softmax step and P·V with the full block's
    softmax step, then its four P·V chunks, the last with the epilogue."""
    assert [e["nodes"] for e in blocks(200)] == [[0, 3], [3, 4], [4, 5], [5, 6], [6, 8]]
    _check(200)
    _restricted_equal(200)


@pytest.mark.slow
def test_the_class_of_records_most_keys():
    """At 1,087 keys (a first block of 127, then five of 192): every block whole Calls of the head, each fitting with a
    pin, and each restricted block equal to C there."""
    _check(MAX_T)
    _restricted_equal(MAX_T)


def test_v19s_softmax_step_is_refused_unpinned():
    with pytest.raises(FT.Unfit) as e:
        FT.build(head(200, 19), registered=REGISTERED)
    assert e.value.code == "fit-pin" and "[[3, 0], [1, 0]]" in str(e.value)


def test_v19s_softmax_step_whole_is_past_the_rows():
    prog = head(200, 19)
    fp, fit = FT.FitPartition(prog, FT.query()), FT.Fit(prog, REGISTERED)
    first = FT.part_of(prog, fp, {"path": [[3, 0]], "nodes": [0, 1]})
    fit.finish(first, fit.measure(first))
    blk = prog.circuit.root_scope.child(3, 0).child(1, 0)
    nodes = blk.spec.body.nodes
    s = min(j for j, n in enumerate(nodes) if n.fn.id == DOT.id)
    step = Part(blk, frozenset((j, m) for j in range(s) for m in range(nodes[j].n if nodes[j].form == "batch" else 1)))
    m = fit.measure(step)
    assert m.rows > FT.MAX_ROWS and m.pinned
