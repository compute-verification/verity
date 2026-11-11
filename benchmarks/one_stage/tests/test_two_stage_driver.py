"""A served window's two-stage audit as `two_stage.py` drives it: one size class's boundary registration
(`verity_vllm.pipeline.boundary_rows`) on its two-stage stream, sampled on a test's own challenger (`fake_challenger`);
the replay side's stream again from ``stage1.json`` alone, sampling the same spans on the bytes the first stage read; the
interior registration over the sampled spans' proof blocks and the second stage on the next challenger's coins; and what
`resume` refuses. A toy check template of one gate's worth."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.codec import definitions, descriptor_id
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat
from verity.circuits.types import Array, Value as Bits
from verity.protocols.spot.device import challenger as Ch
from verity.protocols.spot.verifier import second_stage as SS, service as S
from verity.protocols.spot.verifier.one_stage import partition as P, registered as RG, registration as R
from verity.commitments import history as HT
from verity.commitments.gates import embedding as E
from verity_vllm.pipeline import (boundary_rows as BR, boundary_tables as BT, chunk_rows as CR, interior_rows as IR,
                                  serving_rows as SRW)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "verity" / "protocols" / "spot" / "verifier" / "tests"))
TS = importlib.import_module("two_stage")
FC = importlib.import_module("fake_challenger")

XOR8 = PrimitiveDefinition("TwoStageDriverTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("TwoStageDriverTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
#: a check template whose body calls PAIR twice: inlined, its proof blocks are PAIR's instances
WRAP = bind(CompositeDefinition("TwoStageDriverTestWrap", 0, (),
                                lambda S_: ((("x", Array(2, Array(2, Bits(8)))),), Array(2, Bits(8))),
                                lambda B, S_, x: coll_concat(Array(2, Bits(8)), [B.call(PAIR, x[0]), B.call(PAIR, x[1])]),
                                register=False))
SAME8 = PrimitiveDefinition("TwoStageDriverTestSame8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(1), lambda a, b: int(a == b),
                            register=False)
#: a check template that returns a verdict, one bit
SAME = bind(CompositeDefinition("TwoStageDriverTestSame", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(1)),
                                lambda B, S_, x: B.call(SAME8, x[0], x[1]), register=False))
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 4, "min_tokens": 0,
          "ignore_eos": True, "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
N, Q, K, N_RHO = 6, 2, 1, 4
INTERIOR = (("x", BR.SCHEMA), ("out", BR.SCHEMA))
RULE = TS.Rule(f"rate:{Q}/{N}", K, INTERIOR, 60.0)


def _class(check=PAIR, inline: bool = False) -> BR.Committed:
    spans = [BR.Span(f"r{i}", (i, i + 1, i + 2), GREEDY, (7, 8, 9, i), "length") for i in range(N)]
    return BR.commit(spans, BR.Layout(check, 0, 1, 4, 4, N, BR.run_root("test-serve", 0), inline), 0)


def _interior(layout: BR.Layout, previous: str) -> tuple[dict, dict]:
    """The replay side's interior registration over the Q sampled spans' proof blocks (PAIR's instances), at stand-in
    roots."""
    m = Q * P.units(Program(layout.replay), layout.query)
    program, query = P.population_program(PAIR, m), P.template_instance_query(PAIR)
    part = P.partition(program, query)
    ports = {v: {"leaves": m, "domain": R.window_domain(part["program"], P.digest(part), v, s, m, layout.run)}
             for v, s in INTERIOR}
    roots = [{"port": v, "root": f"{i + 1:02x}" * 64, **ports[v]} for i, (v, _) in enumerate(INTERIOR)]
    rec = R.record(program, query, {"law": "subset", "k": m}, S.SCHEME, roots,
                   {"kind": "served", "run": layout.run, "row": 0, "interior": True}, previous=previous)
    return rec, ports


def test_the_replay_side_resumes_the_first_stage_and_samples_the_second():
    """The first stage samples Q of the class's N spans on the challenger's coins after the boundary registration;
    ``stage1.json`` alone gives the replay side the same registration and sample on the bytes it read; its interior
    registration is logged after it, and the second stage samples K proof blocks in each sampled span on the next
    challenger's coins, opened after the interior registration's digest."""
    c = _class()
    got = TS.first(c.public(), PAIR, RULE, FC.Challenger("first"))
    assert got.layout.n_rho == N_RHO and got.registration.record["population"] == N
    assert got.draw.draw["law"] == "subset" and len(got.draw.units) == Q and got.read
    stage1 = json.loads(json.dumps(got.stage1()))
    assert stage1["units"] == list(got.draw.units) and stage1["spec"] == got.stream.spec.digest()
    again = TS.resume(stage1, PAIR, FC.Challenger("second"))
    assert (again.registration.digest, again.draw.units) == (got.registration.digest, got.draw.units)
    ireg, second = again.second(*_interior(again.layout, again.registration.digest))
    assert ireg.order == again.registration.order and again.coins.live
    assert second.replay_sample == again.draw and len(second.units) == Q * K
    assert [u // N_RHO for u in second.units] == list(range(Q))


def test_an_inlined_checks_second_stage_samples_the_template_its_body_calls():
    """An inlined check (`commit.json`'s ``inline``): the class's stream checks the template WRAP's body calls, so the
    replay side resumes it from ``stage1.json`` alone and the second stage samples among PAIR's instances, two an
    instance of WRAP, in each sampled span."""
    got = TS.first(_class(WRAP, inline=True).public(), WRAP, RULE, FC.Challenger("first"))
    assert got.layout.inline and got.stream.spec.check.query == P.template_instance_query(PAIR)
    again = TS.resume(json.loads(json.dumps(got.stage1())), WRAP, FC.Challenger("second"))
    assert (again.registration.digest, again.draw.units) == (got.registration.digest, got.draw.units)
    _, second = again.second(*_interior(again.layout, again.registration.digest))
    assert len(second.units) == Q * K and [u // (2 * N_RHO) for u in second.units] == list(range(Q))


def test_resume_refuses_another_rule_registration_or_sample():
    got = TS.first(_class().public(), PAIR, RULE, FC.Challenger("first"))
    stage1 = json.loads(json.dumps(got.stage1()))
    with pytest.raises(ValueError, match="hand-off"):
        TS.resume({**stage1, "format": "other"}, PAIR)
    with pytest.raises(ValueError, match="Spec"):
        TS.resume({**stage1, "rule": TS.Rule(RULE.replay, K + 1, INTERIOR, 60.0).to_json()}, PAIR)
    with pytest.raises(ValueError, match="registers to another digest"):
        TS.resume({**stage1, "registration": "00" * 64}, PAIR)
    others = [u for u in range(N) if u not in got.draw.units][:Q]
    with pytest.raises(ValueError, match="other spans"):
        TS.resume({**stage1, "units": others}, PAIR)


def test_the_second_stages_coins_follow_the_whole_first_stage():
    """`Staged` reads the first stage's bytes again in order, commits only once all of them were read, and only to live
    coins of its own."""
    staged = TS.Staged(b"\x01\x02\x03")
    assert staged(2) == b"\x01\x02"
    with pytest.raises(Ch.CoinError, match="of which 2 were read again"):
        staged.commit(b"\x00" * 64)
    with pytest.raises(Ch.CoinError, match="past its bytes"):
        staged(2)
    assert staged(1) == b"\x03"
    with pytest.raises(Ch.CoinError, match="no live coins"):
        staged.commit(b"\x00" * 64)


def test_a_class_with_history_tables_declares_and_registers_them_beside_its_values():
    """A class whose spans' history tables are committed (`Layout.layers`): its Spec declares each table a served value
    of the history table's row schema after ``b.ingress`` and ``b.egress``, and its first stage registers their roots,
    from ``commit.json`` alone, before the coins commit."""
    spans = [BR.Span(f"r{i}", (i, i + 1, i + 2), GREEDY, (7, 8, 9, i), "length") for i in range(N)]
    layout = BR.Layout(PAIR, 0, 1, 4, 4, N, BR.run_root("test-serve", 0), layers=(0, 1), capacity=8)
    tables = {}
    for t in layout.tables:
        h = HT.HistoryTable(layout.partition["program"], t, BT.WORDS, layout.capacity)
        h.append(np.zeros((6, BT.WORDS), dtype=np.uint16))
        tables[t] = h.root
    pub = json.loads(json.dumps(BR.commit(spans, layout, 0, tables=tables).public()))
    got = TS.first(pub, PAIR, RULE, FC.Challenger("first"))
    served = [(v.name, v.schema) for v in got.stream.spec.values if v.phase == "serve"]
    assert served == [("b.ingress", BR.SCHEMA), ("b.egress", BR.SCHEMA), *((t, HT.SCHEMA) for t in layout.tables)]
    assert [r["port"] for r in got.record["roots"]] == [*BR.PORTS, *layout.tables]
    assert {r["port"]: r["root"] for r in got.record["roots"]} == {k: v["root"] for k, v in pub["values"].items()}
    assert got.draw.draw["law"] == "subset" and len(got.draw.units) == Q


@pytest.mark.parametrize("layers", [(), (0,)])
def test_a_class_of_verdicts_declares_and_registers_its_blocks_return(layers):
    """A class that commits its replay block's return (`Layout.verdicts`): its Spec declares the return a served value
    after ``b.ingress`` and ``b.egress``, named under its replay block and before the tables where the class has history
    tables; its first stage registers its root, from ``commit.json`` alone, before the coins commit and samples; and the
    replay side resumes from ``stage1.json`` alone."""
    spans = [BR.Span(f"r{i}", (i, i + 1, i + 2), GREEDY, (7, 8, 9, i), "length") for i in range(N)]
    layout = BR.Layout(SAME, 0, 1, 4, 4, N, BR.run_root("test-serve", 0), layers=layers, capacity=8, verdicts=True)
    tables = {}
    for t in layout.tables:
        h = HT.HistoryTable(layout.partition["program"], t, BT.WORDS, layout.capacity)
        h.append(np.zeros((6, BT.WORDS), dtype=np.uint16))
        tables[t] = h.root
    pub = json.loads(json.dumps(BR.commit(spans, layout, 0, tables=tables).public()))
    name = f"{P.template_id(layout.replay)}/{BR.RETURN}" if layers else BR.RETURN
    got = TS.first(pub, SAME, RULE, FC.Challenger("first"))
    served = [(v.name, v.schema) for v in got.stream.spec.values if v.phase == "serve"]
    assert served == [("b.ingress", BR.SCHEMA), ("b.egress", BR.SCHEMA), (name, BR.SCHEMA),
                      *((t, HT.SCHEMA) for t in layout.tables)]
    assert [(r["port"], r["leaves"]) for r in got.record["roots"]][:3] == [("b.ingress", N), ("b.egress", N), (name, N * N_RHO)]
    assert {r["port"]: r["root"] for r in got.record["roots"]} == {k: v["root"] for k, v in pub["values"].items()}
    assert got.draw.draw["law"] == "subset" and len(got.draw.units) == Q
    again = TS.resume(json.loads(json.dumps(got.stage1())), SAME, FC.Challenger("second"))
    assert (again.registration.digest, again.draw.units) == (got.registration.digest, got.draw.units)


def test_a_class_with_history_tables_gives_the_challengers_history_from_its_own_sample():
    """A class whose spans' tables an interior's reads open: its Spec declares the replay block's parameters its rule
    names (the reads' root and domain, which the verifier supplies, and their row, salt and path, which the replay gives)
    registered interior values, its stream takes them a sampled span and sizes them by the sample, and `First.history` names each sampled span's table at a layer from the first stage's own sample,
    under the class record's root for it, each holding the class's C strings, so `history.roots_of` recomputes the served
    root and domain from the strings the prover publishes."""
    spans = [BR.Span(f"r{i}", (i, i + 1, i + 2), GREEDY, (7, 8, 9, i), "length") for i in range(N)]
    layout = BR.Layout(PAIR, 0, 1, 4, 4, N, BR.run_root("test-serve", 0), layers=(0, 1), capacity=16)
    c = BT.class_rows(layout.prompt, layout.outputs, layout.capacity)
    hs = {}
    for t in layout.tables:
        hs[t] = HT.HistoryTable(layout.partition["program"], t, BT.WORDS, layout.capacity)
        hs[t].append(BT.fill(np.ones((3, BT.WORDS), dtype=np.uint16), layout.prompt, layout.outputs, layout.capacity))
    pub = json.loads(json.dumps(BR.commit(spans, layout, 0, tables={t: h.root for t, h in hs.items()}).public()))
    opening = tuple(IR.Operand("Read_v1", p, p, 2, (0, 1), w) for p, w in (("row", BT.WORDS), ("salt", 96), ("path", 128)))
    rule = replace(RULE, span_values=(("root", 1, IR.DIGEST), ("domain", 1, IR.DIGEST), *IR.span_values(opening)))
    assert TS.Rule.from_json(json.loads(json.dumps(rule.to_json()))) == rule
    got = TS.first(pub, PAIR, rule, FC.Challenger("first"))
    linked = [v for v in got.stream.spec.values if v.registered]
    assert [(v.name, v.phase) for v in linked] == [(p, "interior") for p in ("root", "domain", "row", "salt", "path")]
    assert got.stream.span_values == rule.span_values
    for q in (1, 3):
        sig = dict(got.stream.interior_program(q).fn.signature()[0])
        assert sig["root"] == sig["domain"] == Array(q, RG.DIGEST)
        assert [sig[x.value] for x in opening] == [Array(2 * q, Array(x.words, Bits(16))) for x in opening]
    assert not [v for v in TS.first(pub, PAIR, RULE, FC.Challenger("first")).stream.spec.values if v.registered]
    names = [BT.name(layout.run, layout.prompt, layout.outputs, u, 1) for u in got.draw.units]
    bodies = {"m0": b"the interior's public body"}
    hist = got.history(1, {n: [bc.hex() for bc in hs[n].bcs] for n in names}, bodies)
    assert [t["name"] for t in hist["tables"]] == names and c == 8
    assert all(t["runs"] == [{"start": 0, "count": c, "held": IR.HELD, "row": 0}] for t in hist["tables"])
    leaf = HT.leaf_layer({k: hashlib.sha512(b).hexdigest() for k, b in bodies.items()})
    assert HT.roots_of(leaf, hist) == {n: (hs[n].root, hs[n].domain.domain_id) for n in names}


@pytest.mark.slow
def test_round_2_reads_a_served_classs_tables_root_and_domain_under_its_replay_blocks_name():
    """Round 2's read check (`second_stage._read_ok`) on a table read's ``root`` and ``domain`` ports, from a record the
    serve wrote: a class committed as the engine core commits it (`boundary_rows.commit`, its ``commit.json`` read
    back), its check two reads of one table inlined, so its replay block is `Layout.replay`, the check's two instances a
    span. The first stage registers that record on the class's stream (`first`), whose ports are ``b.ingress``,
    ``b.egress`` and each span's table: a table's name has a ``/``, so round 2 names each link
    ``<replay block>/<parameter>``, the replay block's descriptor id and not the check's. The interior of the sampled
    spans' reads, committed with no test key, registers the reads' parameters under those names: ``root`` and ``domain``
    two rows a span, one an instance, each the entry the verifier's supply commits from round 1's own root for the
    span's table (`interior_rows.sources`, resolved through `First.history`). Round 2 takes each read of them at its
    instance's row, and refuses the same read under the check's id or the bare parameter. A class with no tables names
    no port with a ``/``. What a cheaper test cannot show: the names on the serve's own layout, whose table rows are the
    served 576 words."""
    reads, capacity, prompt, outputs = 2, 8, 4, 2
    rd = E.read(capacity, BT.WORDS)
    params, ret = rd.signature()
    check = bind(CompositeDefinition(
        "TwoStageDriverTestReads", 0, (),
        lambda S_: (tuple((n, t if n in IR.TABLE else Array(reads, t)) for n, t in params), Array(reads, ret)),
        lambda B, S_, *args: B.batch(rd, *args, axes=tuple(None if n in IR.TABLE else 0 for n, _t in params)),
        register=False))
    spans = [BR.Span(f"r{i}", (i, i + 1, i + 2), GREEDY, (7, i), "length") for i in range(N)]
    layout = BR.Layout(check, 0, 1, prompt, outputs, N, BR.run_root("test-serve", 0), inline=True, layers=(0,), capacity=capacity)
    key, tid, c = descriptor_id(layout.replay), P.template_id(rd), BT.class_rows(prompt, outputs, capacity)
    cells = P.units(Program(layout.replay), layout.query)
    assert (layout.n_rho, cells, c) == (2, 4, 6) and key != descriptor_id(check) and layout.templates == (rd,)
    rng = np.random.default_rng(0)
    hs = {}
    for t in layout.tables:
        hs[t] = HT.HistoryTable(layout.partition["program"], t, BT.WORDS, capacity)
        hs[t].append(BT.fill(rng.integers(0, 1 << 16, (3, BT.WORDS), dtype=np.uint16), prompt, outputs, capacity))
    member = IR.named(CR.unit_member(SRW.template(tid), cells))
    table = tuple(IR.Supply(tid, p, f"{key}/{p}", layout.n_rho, (0, 0, 1, 1)) for p in IR.TABLE)
    opening = tuple(IR.Operand(tid, p, f"{key}/{p}", cells, tuple(range(cells)), w) for p, w in member.rows
                    if p not in IR.TABLE)
    links = (*table, *opening)
    w = IR.interior_window(member, cells, "00", [0], values=IR.link_values(links, Q))
    rule = TS.Rule(f"rate:{Q}/{N}", K, tuple(v for m in w.members for v in TS.interior_values(m)), 60.0,
                   IR.span_values(links))
    pub = json.loads(json.dumps(BR.commit(spans, layout, 0, tables={t: h.root for t, h in hs.items()}).public()))
    got = TS.first(pub, check, rule, FC.Challenger("first"))
    assert got.stream.replay is not None and descriptor_id(got.layout.replay) == key
    assert [r["port"] for r in got.record["roots"]] == [*BR.PORTS, *layout.tables]
    named = SS._by_template(got.record)
    assert named and not any("/" in x for x in BR.PORTS)
    assert not SS._by_template(TS.first(_class().public(), PAIR, RULE, FC.Challenger("first")).record)

    defs = definitions(got.layout.program)
    cs = SS._callers_of(defs, defs[key])
    shares = [x for k, node in enumerate(defs[key].body.nodes) for x in SS._node_links(cs, defs[key], k, node)]
    runs = SS._gather(lambda x: (x.template, x.port), lambda a, b: a.value == b.value, cs.units, shares)
    assert SS.replay_links(got.layout.program, key, got.record) is None
    assert {(x.template, x.port, SS._port_of(named, key, x.value), per, tuple(SS.row_at(rs, j) for j in range(cells)))
            for x, per, rs in runs} == {(x.template, x.port, x.value, x.per, x.at) for x in links}

    names = [BT.name(layout.run, prompt, outputs, int(u), 0) for u in got.draw.units]
    sampled = [hs[n] for n in names]
    toks = ((1, 5, 0, 2), (0, 4, 3, 2))
    it = IR.interior([IR.Span(f"s{k}", member, IR.table_reads(t, toks[k]), [[0, cells]], "p" * 64)
                      for k, t in enumerate(sampled)], cells, got.registration.digest, got.draw.units,
                     row="two-stage-driver-test", tables=sampled, served=got.record, positions=c, links=links)
    entries = {e["value"]: e for e in it.record["registered"]}
    assert list(entries) == [x.value for x in links]
    com = it.committed
    hist = got.history(0, it.published, {m.key: com.public_body(k) for k, m in enumerate(com.window.members)})
    round1 = {r["port"]: r for r in got.record["roots"]}
    roots = HT.roots_of(it.leaf_layer, hist)
    assert roots == {n: (bytes.fromhex(round1[n]["root"]), bytes.fromhex(round1[n]["domain"])) for n in names}
    src = IR.sources(links, Q, names)
    assert src == {x.value: [[{"history": f"{n}/{x.port}"}] for n in names for _ in range(layout.n_rho)] for x in table}
    sup = IR.supplied(P.program_digest(com.window.program), table, Q, [t.entry() for t in sampled],
                      IR.link_schemas(member, table))
    for v in sup:
        assert {k: entries[v.value][k] for k in v.entry()} == v.entry()

    part = SS.Part(tid, 0, cells)
    own = SS.Own(tuple(got.draw.units), (part,), it.record, got.record, (),
                 links=tuple(SS.Link(x.template, x.port, SS._port_of(named, key, x.value)) for x, _per, _rs in runs),
                 rows=tuple(SS.Rows(x.template, x.port, per, rs) for x, per, rs in runs))
    for x in table:
        at = list(com.given[com.window.members[0].port(x.port)].positions)
        assert at == [0, 0, 1, 1, 2, 2, 3, 3]
        e = {"value": x.value, **{f: entries[x.value][f] for f in ("root", "domain", "leaves")}, "reads": at,
             "reads_from": 0}
        assert SS._read_ok(own, named, part, x.port, e)
        for bad in ({"value": f"{descriptor_id(check)}/{x.port}"}, {"value": x.port}, {"reads": [0, 1, 0, 1, 2, 2, 3, 3]}):
            assert not SS._read_ok(own, named, part, x.port, {**e, **bad})


def test_a_rule_round_trips_and_its_spec_holds_the_boundary_values_then_the_interior_ones():
    assert TS.Rule.from_json(json.loads(json.dumps(RULE.to_json()))) == RULE
    with pytest.raises(ValueError, match="rate law"):
        TS.Rule(f"subset:{Q}", K, INTERIOR, 60.0)
    spec = TS.stream_of(_class().layout, RULE, FC.Challenger()).spec
    assert [(v.name, v.phase) for v in spec.values] == [("b.ingress", "serve"), ("b.egress", "serve"),
                                                        ("x", "interior"), ("out", "interior")]
    assert spec.selection == S.TwoStage(_class().layout.grain, RULE.replay, K)
