"""`Q_fit` v1 on `AttentionHead_v19` at Llama-3.2-1B's heads (D 64, kBlockN 192), its keys a row each and its values four
rows of 16 columns each, as the window writes them: every block fits a Flock proof block, each key block is cut into its
softmax step and then its P·V, a full block's in chunks of three 16-key k-steps, and each block restricted computes what C
computes there."""
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
from verity.circuits.program import Program, input_id
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array

from verity_vllm.definitions.registry import dead_free as DF
from verity_vllm.definitions.registry import hopper
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry.prims import BF16

D, BN, VS = 64, 192, 16
#: a P·V chunk of 48 keys, three k-steps
K_STEPS = 3
#: the class of record's most keys
MAX_T = 1087


@functools.cache
def head(T: int) -> Program:
    fn = bind(DF.AttentionHeadV19, T=T, D=D, BN=BN, MASKED_FROM=0, DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum)
    params = (("q", Array(D, BF16)),) + tuple((f"k{t}", Array(D, BF16)) for t in range(T)) + \
        tuple((f"v{t}_{s}", Array(VS, BF16)) for t in range(T) for s in range(D // VS))

    def body(B, S, q, *rows):
        k = Coll(Array(T, Array(D, BF16)), concat([x.refs for x in rows[:T]]))
        v = Coll(Array(T, Array(D, BF16)), concat([x.refs for x in rows[T:]]))
        return B.call(fn, q, k, v)

    return Program(CompositeDefinition("FitAttentionHead", 1, (), lambda S: (params, Array(D, BF16)), body, register=False))


@functools.cache
def blocks(T: int) -> list:
    return FT.build(head(T))


def _key_blocks(T: int) -> dict:
    """The entries inside each key block's activation, by its path."""
    out: dict = {}
    for e in blocks(T):
        if isinstance(e, dict) and len(e["path"]) == 2:
            out.setdefault(json.dumps(e["path"]), []).append(e)
    return out


def _check(T: int) -> None:
    prog = head(T)
    v = PO.verify(PO.build(prog, FT.query(), blocks(T)), prog)
    assert v.ok, v.codes
    fp, fit = FT.FitPartition(prog, FT.query()), FT.Fit(prog)
    for e in blocks(T):
        p = FT.part_of(prog, fp, e)
        rows, k, outs, widths = fit.measure(p)
        assert rows <= FT.MAX_ROWS and k <= FT.K_MAX
        fit.finish(p, outs, widths)
    dot = hopper.HopperBF16WgmmaDot16.id
    for path, es in _key_blocks(T).items():
        scope = prog.circuit.root_scope
        for k, m in json.loads(path):
            scope = scope.child(k, m)
        nodes = scope.spec.body.nodes
        s = min(j for j, n in enumerate(nodes) if n.fn.id == dot)
        assert all(n.fn.id == dot for n in nodes[s:]) and not any(n.fn.id == dot for n in nodes[:s])
        got = [e["nodes"] for e in es]
        assert got[0] == [0, s] and got[-1][1] == len(nodes)
        if len(nodes) - s == BN // 16:
            assert got[1:] == [[a, a + K_STEPS] for a in range(s, len(nodes), K_STEPS)], path


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
        b = R.restrict(c, obj, e, bits)
        own = decode_program(json.loads(b.descriptor), reg).circuit
        mine = {g.index: t[b.to_c_gate(g.index)] for g in own.gates() if input_id(g.prim.id) is not None}
        evaluate_region(own, own[0:own.size], mine)
        assert [mine[x] for x in own.root_outputs()] == [t[g] for g in b.outputs], e


def test_a_full_key_block_is_cut_as_its_softmax_step_and_its_PV_chunks():
    assert len(_key_blocks(200)) == 1 and len(next(iter(_key_blocks(200).values()))) == 1 + BN // 48
    _check(200)
    _restricted_equal(200)


@pytest.mark.slow
def test_the_class_of_records_most_keys_are_thirty_blocks_a_head():
    """At 1,087 keys (a first block of 127 keys, then five of 192): 29 blocks, each key block its softmax step and its P·V
    chunks, the short block's last chunk four k-steps, and each restricted block equal to C there; the 200-key case has no
    short first block cut."""
    assert len(blocks(MAX_T)) == 29
    assert sorted(len(es) for es in _key_blocks(MAX_T).values()) == [3, 5, 5, 5, 5, 5]
    _check(MAX_T)
    _restricted_equal(MAX_T)
