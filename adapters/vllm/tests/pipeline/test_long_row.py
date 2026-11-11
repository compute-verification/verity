"""`pipeline.long_row`, the check template of a window over one of DSA's long rows:

* at the served statics it is 44 units of the mask's cut, 8 positions, the merge, 64 chunks, 2,048 slots and 2,048 reads, 4,213
  units of 56 templates an instance, and its body calls exactly those (built without any callee's body);
* at small statics its body calls `units` in order, `boundary_rows.Layout` takes it inline over the window's class, and each
  unit reads the window's operands and the units before it as round 2 links and wires them (`second_stage`'s node links and
  node wires);
* where its index rows are `index_values` it returns `DsaTopKReadRows_v1`'s ok and its read units return that Definition's
  kv, so the check and the rows its reads commit are together `DsaTopKReadRows_v1`'s function (slow).
"""
from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from verity.protocols.spot.verifier import second_stage as SS
from verity.circuits.codec import definitions, descriptor_id
from verity.circuits.defs import Builder, CompositeDefinition, bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.refs import Affine, Coll, concat
from verity.circuits.types import Array
from verity.commitments.gates.embedding import read
from verity.definitions.registry import dsa_topk_reads as R
from verity_vllm.pipeline import long_row as LR

RUN = "0" * 128


class _Unlaid(Builder):
    """A body's nodes and wiring alone: no node's gate count, so no callee's body is built."""

    def _emit(self, node):
        self.nodes.append(node)
        self.offsets.append(0)
        return Coll(node.out, Affine("n", len(self.nodes) - 1, 0, node.out.leaves))


def _unlaid_nodes(t):
    params, ret = t.signature()
    B = _Unlaid(params, ret)
    B.finish(t.defn.body(B, t.S(), *(Coll(ty, Affine("p", i, 0, ty.leaves)) for i, (_, ty) in enumerate(params))))
    return B.nodes


def _members(nodes):
    return [descriptor_id(n.fn) for n in nodes for _ in range(n.n if n.form == "batch" else 1)]


def _small():
    return bind(LR.DsaTopKReadRowsCut, N=12, TOPK=2, BLOCK=4, READ=read(16, 1))


def test_the_served_row_is_4213_units_of_56_templates() -> None:
    t = LR.served()
    assert {k: t.bindings[k] for k in LR.SERVED} == {"N": 1 << 17, "TOPK": 2048, "BLOCK": 16384}
    assert descriptor_id(t.bindings["READ"]) == descriptor_id(read(1 << 17, 576))
    us = LR.units(t)
    by = Counter(d.base_id for _, d in us)
    assert by == {"DsaTopKDigitCount_v1": 32, "DsaTopKDigitPick_v1": 4, "DsaTopKMaskBlock_v1": 8, "DsaMaskPositions_v1": 8,
                  "DsaPositionsMerge_v1": 1, "DsaPositionsChunk_v1": 64, "DsaPositionsSlot_v1": 2048, "EmbeddingRead_v1": 2048}
    ids = Counter(descriptor_id(d) for _, d in us)
    assert (len(us), len(ids)) == (4213, 56)
    assert sorted(ids.values()) == [1] * 53 + [64, 2048, 2048]
    assert [(name, ty.n, ty.elem) for name, ty in t.signature()[0][-2:]] == [("c", 64, R.I32), ("i", R.CHUNK, R.I32)]
    assert LR.index_values(2048) == {"c": list(range(64)), "i": list(range(R.CHUNK))}
    nodes = _unlaid_nodes(t)
    assert {n.form for n in nodes} == {"call"}
    assert _members(nodes) == [descriptor_id(d) for _, d in us]


@pytest.mark.slow
def test_the_layout_takes_it_inline_and_round_two_links_and_wires_each_units_operands() -> None:
    """The check template registered inline at small statics, and round 2's own derivation run on the replay block it
    registers: every unit's operand gets a link or a wire. About a minute, nearly all of it building the template's body (its
    units' gates, the picks' most); no cheaper test runs `second_stage` on the registered descriptor, which round 2 reads."""
    t = _small()
    nodes = t.body.nodes
    assert _members(nodes) == [descriptor_id(d) for _, d in LR.units(t)]
    L = LR.layout(RUN, t)
    assert (L.per_prompt, L.per_output, L.prompt, L.outputs, L.n_rho) == (0, 1, 1 << 17, 2, 2)
    assert [descriptor_id(x) for x in L.templates] == list(dict.fromkeys(descriptor_id(n.fn) for n in nodes))
    assert L.provenance()["check"] == descriptor_id(t) and L.provenance()["inline"] is True
    defs = definitions(L.program)
    fn = defs[descriptor_id(L.replay)]
    cs = SS._callers_of(defs, fn)
    assert None not in cs.template
    assert all(SS._node_links(cs, fn, k, node) is not None for k, node in enumerate(fn.body.nodes))
    assert all(SS._node_wires(cs, fn, k, node) is not None for k, node in enumerate(fn.body.nodes))
    wires = {(x.template.partition("{")[0], x.port, x.writer.partition("{")[0], x.wport)
             for x, _per, _runs in SS.wire_runs(defs, descriptor_id(L.replay))}
    assert {("DsaPositionsChunk_v1", "slots", "DsaPositionsMerge_v1", "o0"),
            ("DsaPositionsSlot_v1", "slots", "DsaPositionsChunk_v1", "o0"),
            ("EmbeddingRead_v1", "tok", "DsaPositionsSlot_v1", "o0")} <= wires


def _i32s(values) -> list[int]:
    return [(v >> b) & 1 for v in values for b in range(32)]


def _random_args(rng: random.Random, t, rows: int) -> list[np.ndarray]:
    """Random bits for each of the long row's parameters, then the index rows as the verifier supplies them."""
    params = t.signature()[0]
    index = LR.index_values(t.bindings["TOPK"])
    return [np.asarray([_i32s(index[name]) if name in index else [rng.getrandbits(1) for _ in range(ty.leaves)]
                        for _ in range(rows)], dtype=np.uint8) for name, ty in params]


def _read_rows(t):
    """The rows the check's read units return, as a Definition of this test's: the check's own units (`long_row.read_calls`) on
    its parameters, returning each read's row in slot order where the check returns each read's ok. They are what an interior
    commits at the reads' output port."""
    rows = Array(t.bindings["TOPK"], t.bindings["READ"].signature()[1].fields[0])
    defn = CompositeDefinition("LongRowTestReadRows", 0, (), lambda S_: (t.signature()[0], rows),
                               lambda B, S_, *args: Coll(rows, concat([r[0].refs for r in LR.read_calls(B, t.S(), *args)])),
                               register=False)
    return bind(defn)


@pytest.mark.slow
def test_its_ok_and_its_reads_rows_are_the_long_rows_function() -> None:
    """The stand-in against `DsaTopKReadRows_v1` at the same small statics, its index rows `index_values`, on random operands
    and on scores with ties at the threshold at every n: the check returns that Definition's ok, and its read units' rows
    (`_read_rows`) are that Definition's kv, so the check with the rows its reads commit loses nothing of the Definition.
    The one test that evaluates the stand-in whole, which the counts and the layout don't."""
    t = _small()
    rng = random.Random("long-row-cut")
    args = _random_args(rng, t, 6)
    pool = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000]
    for i, n in enumerate((0, 1, 2, 7, 12, 1 << 20)):
        if i % 2:
            words = [rng.choice(pool) for _ in range(12)]
            args[0][i] = np.asarray(_i32s(words), dtype=np.uint8)
        args[1][i] = np.asarray(_i32s([n]), dtype=np.uint8)
    whole = bind(R.DsaTopKReadRows, **{k: t.bindings[k] for k in ("N", "TOPK", "BLOCK", "READ")})
    kv, ok = whole.signature()[1].fields
    assert t.signature()[1] == ok
    want = np.asarray(BS.evaluate_bits(whole, *args[:-2])).reshape(6, -1)
    assert want.shape[1] == kv.leaves + ok.leaves
    assert np.array_equal(np.asarray(BS.evaluate_bits(t, *args)).reshape(6, -1), want[:, kv.leaves:])
    assert np.array_equal(np.asarray(BS.evaluate_bits(_read_rows(t), *args)).reshape(6, -1), want[:, :kv.leaves])
