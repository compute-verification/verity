"""`pipeline.long_row`, the check template of a window over one of DSA's long rows:

* at the served statics it is 44 units of the mask's cut, 8 positions, the merge and 2,048 reads, 2,101 units of 54 templates an
  instance, and its body calls exactly those (built without any callee's body);
* at small statics its body calls `units` in order, `boundary_rows.Layout` takes it inline over the window's class, and each
  unit reads the window's operands as round 2 links them (`second_stage`'s node links);
* it computes `DsaTopKReadRows_v1`'s function (slow).
"""
from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from verity.core.protocols.spot import second_stage as SS
from verity.ml.circuits.codec import definitions, descriptor_id
from verity.ml.circuits.defs import Builder, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.refs import Affine, Coll
from verity.ml.commitments.gates.embedding import read
from verity.ml.definitions.registry import dsa_topk_reads as R
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


def test_the_served_row_is_2101_units_of_54_templates() -> None:
    t = LR.served()
    assert {k: t.bindings[k] for k in LR.SERVED} == {"N": 1 << 17, "TOPK": 2048, "BLOCK": 16384}
    assert descriptor_id(t.bindings["READ"]) == descriptor_id(read(1 << 17, 576))
    us = LR.units(t)
    by = Counter(d.base_id for _, d in us)
    assert by == {"DsaTopKDigitCount_v1": 32, "DsaTopKDigitPick_v1": 4, "DsaTopKMaskBlock_v1": 8, "DsaMaskPositions_v1": 8,
                  "DsaPositionsMerge_v1": 1, "EmbeddingRead_v1": 2048}
    ids = Counter(descriptor_id(d) for _, d in us)
    assert (len(us), len(ids)) == (2101, 54)
    assert sorted(ids.values()) == [1] * 53 + [2048]
    nodes = _unlaid_nodes(t)
    assert {n.form for n in nodes} <= {"call", "batch"}
    assert _members(nodes) == [descriptor_id(d) for _, d in us]


def test_the_layout_takes_it_inline_and_round_two_links_each_units_operands() -> None:
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


def _random_args(rng: random.Random, t, rows: int) -> list[np.ndarray]:
    params = t.signature()[0]
    return [np.asarray([[rng.getrandbits(1) for _ in range(ty.leaves)] for _ in range(rows)], dtype=np.uint8)
            for _, ty in params]


@pytest.mark.slow
def test_it_computes_the_long_rows_function() -> None:
    """The stand-in against `DsaTopKReadRows_v1` at the same small statics, on random operands and on scores with ties at the
    threshold at every n: the one test that evaluates the stand-in whole, which the counts and the layout don't."""
    t = _small()
    rng = random.Random("long-row-cut")
    args = _random_args(rng, t, 6)
    pool = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000]
    for i, n in enumerate((0, 1, 2, 7, 12, 1 << 20)):
        if i % 2:
            words = [rng.choice(pool) for _ in range(12)]
            args[0][i] = np.asarray([(w >> b) & 1 for w in words for b in range(32)], dtype=np.uint8)
        args[1][i] = np.asarray([(n >> b) & 1 for b in range(32)], dtype=np.uint8)
    whole = bind(R.DsaTopKReadRows, **{k: t.bindings[k] for k in ("N", "TOPK", "BLOCK", "READ")})
    assert np.array_equal(np.asarray(BS.evaluate_bits(t, *args)), np.asarray(BS.evaluate_bits(whole, *args)))
