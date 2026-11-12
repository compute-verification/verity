"""`pipeline.long_row`, the check template of a window over one of DSA's long rows:

* at the served statics it is 44 units of the mask's cut, 8 positions, the merge, 64 chunks, 2,048 slots and 2,048 reads, 4,213
  units of 56 templates an instance, and its body calls exactly those (built without any callee's body);
* at small statics its body calls `units` in order, `boundary_rows.Layout` takes it inline over the window's class, with its
  return, each read's ok, the boundary's third value, a row a read, and each unit reads the window's operands and the units
  before it as round 2 links and wires them (`second_stage`'s node links and node wires);
* where its index rows are `index_values` it returns `DsaTopKReadRows_v1`'s ok and its read units return that Definition's
  kv, so the check and the rows its reads commit are together `DsaTopKReadRows_v1`'s function (slow);
* its links (`links`) at the served statics are 87, of nine values, and at small statics they are the ones round 2 derives
  from the replay block (`interior_links.links`), under either naming of round 1's record, and its interior window
  (`window`) with them is the stream's interior program, a function of the replay sample's size (slow);
* a sampled span's units (`spans`) are one span a template, every unit of the mask's cut and of the index on rows its
  Definition returns its outputs on and every read an opening of the span's table at its slot's position; the reads' ok
  bits are what the check returns on the same operands and their rows `DsaTopKReadRows_v1`'s kv, and
  `interior_rows.interior` commits the spans under the block's wires and links, refusing a row that is not its writer's,
  its value's or its table's (slow).
"""
from __future__ import annotations

import functools
import random
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from verity.protocols.spot.verifier import second_stage as SS
from verity.protocols.spot.verifier.one_stage import partition as P
from verity.circuits.codec import definitions, descriptor_id
from verity.circuits.defs import Builder, CompositeDefinition, bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.refs import Affine, Coll, concat
from verity.circuits.types import Array, Tuple
from verity.commitments import history as HT
from verity.commitments.gates.embedding import read
from verity.definitions.registry import dsa_topk_reads as R
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


@functools.cache
def _laid():
    """`_small` and its layout, built once for the slow tests that read it."""
    t = _small()
    return t, LR.layout(RUN, t)


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
    registers: every unit's operand gets a link or a wire, and each read's ok a link to the return the class commits, a row
    a read. About a minute, nearly all of it building the template's body (its
    units' gates, the picks' most); no cheaper test runs `second_stage` on the registered descriptor, which round 2 reads."""
    t, L = _laid()
    nodes = t.body.nodes
    assert _members(nodes) == [descriptor_id(d) for _, d in LR.units(t)]
    assert (L.per_prompt, L.per_output, L.prompt, L.outputs, L.n_rho) == (0, 1, 1 << 17, 2, 2)
    assert [descriptor_id(x) for x in L.templates] == list(dict.fromkeys(descriptor_id(n.fn) for n in nodes))
    assert L.provenance()["check"] == descriptor_id(t) and L.provenance()["inline"] is True
    assert L.verdict_rows == (L.n_rho * t.bindings["TOPK"], 1) and L.values == (*BR.PORTS, BR.RETURN)
    assert L.ports()[BR.RETURN]["leaves"] == L.spans * L.n_rho * t.bindings["TOPK"]
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
    returns = [(x.template, x.port, per) for x, per, _ in SS.link_runs(defs, descriptor_id(L.replay)) if x.value == BR.RETURN]
    assert returns == [(descriptor_id(t.bindings["READ"]), "o1", L.verdict_rows[0])]


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
    t, L = _laid()
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
    wires, rows, digests = SS.replay_wires(L.program, key), SS.replay_rows(L.program, key), []
    with pytest.raises(ValueError, match="a writer of 2 units a block"):
        IR.interior_window(ms, cells, "00", [0], wires)
    for q in (1, 2):
        values = IL.link_values(links, q)
        w = IR.interior_window(ms, cells, "00", list(range(q)), wires, values, rows)
        assert [m.n for m in w.members] == [q * n for n in cells]
        want = P.mixed_population_program([(x, q * n) for x, n in zip(L.templates, cells, strict=True)], values=values)
        digests.append(P.program_digest(w.program))
        assert digests[-1] == P.program_digest(want)
    assert digests[0] != digests[1]


def _bits(a: np.ndarray) -> np.ndarray:
    """Rows of u16 words as rows of bits, each word's low bit first."""
    return np.unpackbits(np.ascontiguousarray(a, dtype="<u2").view(np.uint8).reshape(len(a), -1), axis=1, bitorder="little")


def _table(length: int, seed: int, name: str = "long-row-test", fill: tuple[int, int] | None = None) -> HT.HistoryTable:
    """A history table of 16 leaves holding ``length`` rows of one word, each under a salt of its own; ``fill``: its class
    (prompt, outputs), to whose rows it is filled (`boundary_tables.fill`)."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, 1 << 16, (length, 1), dtype=np.uint16)
    rows = BT.fill(rows, *fill, 16) if fill else rows
    tc = HT.HistoryTable("ab" * 64, name, 1, 16)
    tc.append(rows, [rng.bytes(192) for _ in rows])
    return tc


def _rows(seed: str, ns) -> list[tuple[list[int], int]]:
    """An instance a given n: 12 index scores (f32 words) and n, every second instance's scores ties at the threshold."""
    rng = random.Random(seed)
    pool = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000]
    return [([rng.choice(pool) if k % 2 else rng.getrandbits(32) for _ in range(12)], n) for k, n in enumerate(ns)]


def test_a_spans_units_are_each_its_definitions_value() -> None:
    """`spans` at small statics over three instances (random scores, ties at the threshold, n = 0): one span a template in
    `window`'s order, a template's units an instance its group, and each unit of the mask's cut and of the index takes rows
    on which its Definition returns its outputs, every word's bits past its type 0. Each read's key is its slot unit's
    position and its rows open the table there (`interior_links.opened`, the read's reference), past the table's length
    too."""
    t = _small()
    table = _table(9, 1)
    got = LR.spans(t, "r0", _rows("long-row-spans", (12, 7, 0)), table, "p" * 64)
    ms, cells = LR.window(t, 1)
    assert [(s.member.template, s.member.rows, s.member.outs, s.n, s.group) for s in got] == \
        [(m.template, m.rows, m.outs, 3 * n, n) for m, n in zip(ms, cells, strict=True)]
    assert all(s.entries == [[i * s.group, (i + 1) * s.group, "r0", i] for i in range(3)] for s in got)
    defs = {descriptor_id(d): d for _, d in LR.units(t)}
    for s in got[:-1]:
        params, ret = defs[s.member.template].signature()
        fields = ret.fields if isinstance(ret, Tuple) else (ret,)
        args = [_bits(s.data["rows"][p])[:, :ty.leaves] for (p, _w), (_n, ty) in zip(s.member.rows, params, strict=True)]
        outs = [_bits(s.data["outs"][p]) for p, _w in s.member.outs]
        want = np.asarray(BS.evaluate_bits(defs[s.member.template], *args)).reshape(s.n, -1)
        assert np.array_equal(np.concatenate([o[:, :f.leaves] for o, f in zip(outs, fields, strict=True)], axis=1), want)
        assert not any(o[:, f.leaves:].any() for o, f in zip(outs, fields, strict=True))
    slot, reads = got[-2], got[-1]
    assert reads.member.template == descriptor_id(t.bindings["READ"])
    assert np.array_equal(reads.data["rows"]["tok"], slot.data["outs"]["o0"])
    assert {int(k) for k in reads.data["rows"]["tok"][:, 0]} & set(range(9, 12))
    IL.opened(reads.member, reads.data, reads.n, [table], 16)
    with pytest.raises(ValueError, match="11 scores for a row of 12 keys"):
        LR.spans(t, "r0", [([0] * 11, 3)], table, "p" * 64)


@pytest.mark.slow
def test_a_sampled_spans_units_commit_as_its_interior() -> None:
    """Two sampled spans of a class of three (`spans`: one of two instances, one of one, which the interior pads to the
    block's two), each opening its own table as round 1 registered it: the reads' oks are what the check returns on the
    same operands at each instance and their rows `DsaTopKReadRows_v1`'s kv there (the one place a span's units are held
    to the whole check and to its Definition), and `interior_rows.interior` commits them under the replay block's wires and rows (`second_stage.replay_wires`,
    `replay_rows`) and `links`: its program is the stream's at two spans, it registers the block's nine parameters as
    round 2 names them, and a wired unit reads its own instance's writer (read j its slot j). It refuses a
    span whose wired row is not its writer's output, whose scores at one unit are not the block's another unit read, or
    whose read is not an opening of its table. On the layout the links test builds (`_laid`): nothing cheaper has the
    block's wires."""
    t, L = _laid()
    key, q0 = descriptor_id(L.replay), L.n_rho
    ms, cells = LR.window(t, q0)
    links, wires, block = LR.links(t, q0, tables=True), SS.replay_wires(L.program, key), SS.replay_rows(L.program, key)
    tables = [_table(n, seed, BT.name(RUN, L.prompt, L.outputs, slot, 0), (L.prompt, L.outputs))
              for slot, n, seed in ((0, 9, 1), (2, 5, 2))]
    rows = _rows("long-row-interior", (12, 7))
    sampled = [LR.spans(t, "r0", rows, tables[0], "p" * 64),
               LR.spans(t, "r2", _rows("long-row-padded", (5,)), tables[1], "p" * 64)]

    read = sampled[0][-1]
    assert read.n == len(rows) * 2
    index = LR.index_values(2)
    opening = {p: _bits(read.data["rows"][p]).reshape(len(rows), 2, -1) for p in IL.READ[1:]}
    args = {"scores": [_i32s(s) for s, _n in rows], "n": [_i32s([n]) for _s, n in rows],
            **{p: [_i32s(v)] * len(rows) for p, v in index.items()},
            **{p: opening[p][:, 0] for p in IL.TABLE}, **{p: opening[p].reshape(len(rows), -1) for p in IL.READ[3:]}}
    operands = [np.asarray(args[p], dtype=np.uint8) for p, _ty in t.signature()[0]]
    whole = bind(R.DsaTopKReadRows, **{k: t.bindings[k] for k in ("N", "TOPK", "BLOCK", "READ")})
    kv = whole.signature()[1].fields[0]
    assert np.array_equal(np.asarray(BS.evaluate_bits(t, *operands)).reshape(len(rows), -1),
                          _bits(read.data["outs"]["o1"])[:, 0].reshape(len(rows), -1))
    assert np.array_equal(np.asarray(BS.evaluate_bits(whole, *operands[:-2])).reshape(len(rows), -1)[:, :kv.leaves],
                          _bits(read.data["outs"]["o0"]).reshape(len(rows), -1))

    served = {"roots": [{"port": x.value, "root": x.root.hex(), "leaves": x.leaves, "domain": x.domain.domain_id.hex()}
                        for x in tables]}
    kw = {"row": "long-row-test", "salt_key": bytes(range(32)), "wires": wires, "rows": block, "tables": tables,
          "served": served, "positions": BT.class_rows(L.prompt, L.outputs, 16), "links": links}
    it = IR.interior(sampled, cells, "00" * 64, [0, 2], **kw)
    w = it.committed.window
    assert [(m.template, m.n) for m in w.members] == [(m.template, 2 * n) for m, n in zip(ms, cells, strict=True)]
    want = P.mixed_population_program([(x, 2 * n) for x, n in zip(L.templates, cells, strict=True)],
                                      values=IL.link_values(links, 2))
    assert P.program_digest(w.program) == P.program_digest(want)
    assert [e["value"] for e in it.record["registered"]] == [f"{key}/{p}" for p in PARAMS]
    assert it.published == {x.value: [bc.hex() for bc in x.bcs] for x in tables}
    reads = next(m for m in w.members if m.template == descriptor_id(t.bindings["READ"]))
    assert w.held_leaves(reads, "tok") == tuple(range(reads.n))
    pick = next(m for m in w.members if m.template.startswith("DsaTopKDigitPick"))
    assert w.held_leaves(pick, pick.rows[0][0]) == (0, 1, 2, 3)

    ids = [s.member.template for s in sampled[0]]

    def off(base: str, kind: str, port: str, unit: int = 0) -> list:
        k = next(i for i, x in enumerate(ids) if x.startswith(base))
        s = sampled[0][k]
        a = s.data[kind][port].copy()
        a[unit, 0] ^= 1
        return [(*sampled[0][:k], replace(s, data={**s.data, kind: {**s.data[kind], port: a}}), *sampled[0][k + 1:]),
                sampled[1]]

    merge = ms[ids.index(next(x for x in ids if x.startswith("DsaPositionsMerge")))]
    for base, kind, port, why in (
            ("DsaPositionsMerge", "rows", merge.rows[0][0], "units that write them"),
            ("DsaPositionsSlot", "outs", "o0", "units that write them"),
            ("DsaTopKDigitCount", "rows", "scores", "which another unit reads as another row"),
            ("EmbeddingRead", "rows", "row", "does not open its table there")):
        with pytest.raises(ValueError, match=why):
            IR.interior(off(base, kind, port), cells, "00" * 64, [0, 2], **kw)

