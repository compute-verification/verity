"""`pipeline.long_row`, the check template of a window over one of DSA's long rows:

* at the served statics it is 44 units of the mask's cut, 8 positions, the merge, 64 chunks, 2,048 slots and 2,048 reads, 4,213
  units of 56 templates an instance, and its body calls exactly those (built without any callee's body);
* at small statics its body calls `units` in order, `boundary_rows.Layout` takes it inline over the window's class, and each
  unit reads the window's operands and the units before it as round 2 links and wires them (`second_stage`'s node links and
  node wires);
* where its index rows are `index_values` it computes `DsaTopKReadRows_v1`'s function (slow);
* its links (`links`) at the served statics are 87, of nine values, and at small statics they are the ones round 2 derives
  from the replay block (`interior_links.links`), under either naming of round 1's record, and its interior window
  (`window`) with them is the stream's interior program, a function of the replay sample's size (slow).
"""
from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from verity.core.protocols.spot import second_stage as SS
from verity.core.protocols.spot.one_stage import partition as P
from verity.ml.circuits.codec import definitions, descriptor_id
from verity.ml.circuits.defs import Builder, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.refs import Affine, Coll
from verity.ml.commitments.gates.embedding import read
from verity.ml.definitions.registry import dsa_topk_reads as R
from verity_vllm.pipeline import boundary_rows as BR, boundary_tables as BT, interior_links as IL, interior_rows as IR
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


@pytest.mark.slow
def test_it_computes_the_long_rows_function() -> None:
    """The stand-in against `DsaTopKReadRows_v1` at the same small statics, its index rows `index_values`, on random operands
    and on scores with ties at the threshold at every n: the one test that evaluates the stand-in whole, which the counts and
    the layout don't."""
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
    assert np.array_equal(np.asarray(BS.evaluate_bits(t, *args)), np.asarray(BS.evaluate_bits(whole, *args[:-2])))


PARAMS = ("scores", "n", "c", "i", "root", "domain", "row", "salt", "path")


def test_the_served_rows_links_are_87_of_nine_values() -> None:
    """At the served statics, two instances a block: 40 units read their block of the scores and n, the chunks ``c``, the
    slots ``i``, and the reads the table's root and domain and their own row, salt and path. Nothing is built: the
    links are stated from the statics, and the window's 56 members take each at a row of theirs (`interior_links.placed`)."""
    t = LR.served()
    links = LR.links(t, 2)
    assert len(links) == 2 * 40 + 2 + 5
    ms, cells = LR.window(t, 2)
    assert (len(ms), sum(cells)) == (56, 2 * 4213)
    placed = IL.placed(ms, cells, 3, links)
    assert len(placed) == 87 and all(len(at) == 3 * cells[k] for _x, k, at in placed)
    assert IL.span_values(links) == (
        ("scores", 16, (32768, 16)), ("n", 2, (2, 16)), ("c", 128, IL.INDEX), ("i", 64, IL.INDEX), ("root", 2, IL.DIGEST),
        ("domain", 2, IL.DIGEST), ("row", 4096, (576, 16)), ("salt", 4096, (96, 16)), ("path", 4096, (544, 16)))
    assert {x.value for x in links if isinstance(x, IL.Supply)} == {"c", "i", "root", "domain"}
    c, i = (next(x for x in links if x.value == v) for v in ("c", "i"))
    assert c.index == tuple(range(64)) * 2 and c.at == tuple(range(128))
    assert i.index == tuple(range(R.CHUNK)) * 2 and i.at[:34] == (*range(32), 0, 1) and i.at[2048:2050] == (32, 33)
    key = descriptor_id(BR.inlined(t, 2))
    assert [v for v, _per, _leaf in IL.span_values(LR.links(t, 2, tables=True))] == [f"{key}/{p}" for p in PARAMS]
    with pytest.raises(ValueError, match="keys each"):
        LR.links(bind(LR.DsaTopKReadRowsCut, N=(1 << 17) + 4096, TOPK=2048, BLOCK=16384, READ=read(1 << 17, 576)), 2)


@pytest.mark.slow
def test_its_links_are_the_ones_round_two_derives_and_its_window_takes_them() -> None:
    """`links` against round 2's own derivation on the registered replay block (`interior_links.links`), where round 1's
    record names its ports bare and where it names a history table, and the interior window of `window`'s members with
    those values against the stream's interior program at replay samples of one and two spans. About a minute, building
    the small template's body as the layout test does; nothing cheaper runs round 2's derivation, which reads the
    block's Program."""
    t = _small()
    L = LR.layout(RUN, t)
    key, q0 = descriptor_id(L.replay), L.n_rho
    ms, cells = LR.window(t, q0)
    assert [m.template for m in ms] == [descriptor_id(x) for x in L.templates] and sum(cells) == q0 * len(LR.units(t))
    for tables, ports in ((False, list(L.ports())), (True, [*L.ports(), BT.name(RUN, L.prompt, L.outputs, 0, 0)])):
        derived = IL.links(L.program, key, {"roots": [{"port": p} for p in ports]}, ms, cells, LR.supplied(t, q0))
        assert LR.links(t, q0, tables) == derived
    links = LR.links(t, q0)
    assert IL.span_values(links) == (
        ("scores", 6, (8, 16)), ("n", 2, (2, 16)), ("c", 2, IL.INDEX), ("i", 64, IL.INDEX), ("root", 2, IL.DIGEST),
        ("domain", 2, IL.DIGEST), ("row", 4, (1, 16)), ("salt", 4, (96, 16)), ("path", 4, (128, 16)))
    wires, digests = SS.replay_wires(L.program, key), []
    for q in (1, 2):
        values = IL.link_values(links, q)
        w = IR.interior_window(ms, cells, "00", list(range(q)), wires, values)
        assert [m.n for m in w.members] == [q * n for n in cells]
        want = P.mixed_population_program([(x, q * n) for x, n in zip(L.templates, cells, strict=True)], values=values)
        digests.append(P.program_digest(w.program))
        assert digests[-1] == P.program_digest(want)
    assert digests[0] != digests[1]
