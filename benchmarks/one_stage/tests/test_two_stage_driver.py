"""A served window's two-stage audit as `two_stage.py` drives it: one size class's boundary registration
(`verity_vllm.pipeline.boundary_rows`) on its two-stage stream, sampled on a test's own challenger (`fake_challenger`);
the replay side's stream again from ``stage1.json`` alone, sampling the same spans on the bytes the first stage read; the
interior registration over the sampled spans' proof blocks and the second stage on the next challenger's coins; and what
`resume` refuses. A toy check template of one gate's worth."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import shutil
import sys
import tempfile
import threading
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from verity.circuits.codec import definitions, descriptor_id
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat
from verity.circuits.types import Array, Value as Bits
from verity.protocols.spot.device import challenger as Ch, coins as CC
from verity.protocols.spot.verifier import second_stage as SS, service as S
from verity.protocols.spot.verifier.one_stage import partition as P, registered as RG, registration as R
from verity.commitments import history as HT
from verity.commitments.gates import embedding as E
from verity_vllm.pipeline import (boundary_rows as BR, boundary_stop as BS, boundary_tables as BT, chunk_rows as CR,
                                  interior_rows as IR, serving_rows as SRW)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "verity" / "protocols" / "spot" / "verifier" / "tests"))
TS = importlib.import_module("two_stage")
FC = importlib.import_module("fake_challenger")
AC = importlib.import_module("a5_coins")

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
#: PAIR with its operands the other way round: a second template of PAIR's ports
SWAP = bind(CompositeDefinition("TwoStageDriverTestSwap", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[1], x[0]), register=False))
#: a check template whose body calls two templates, PAIR twice and SWAP once: inlined, its proof blocks are both's
BOTH = bind(CompositeDefinition("TwoStageDriverTestBoth", 0, (),
                                lambda S_: ((("x", Array(3, Array(2, Bits(8)))),), Array(3, Bits(8))),
                                lambda B, S_, x: coll_concat(Array(3, Bits(8)), [B.call(PAIR, x[0]), B.call(PAIR, x[1]),
                                                                                 B.call(SWAP, x[2])]),
                                register=False))
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 4, "min_tokens": 0,
          "ignore_eos": True, "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
N, Q, K, N_RHO = 6, 2, 1, 4
INTERIOR = (("x", BR.SCHEMA), ("out", BR.SCHEMA))
RULE = TS.Rule(f"rate:{Q}/{N}", K, INTERIOR, 60.0)


def _class(check=PAIR, inline: bool = False) -> BR.Committed:
    spans = [BR.Span(f"r{i}", (i, i + 1, i + 2), GREEDY, (7, 8, 9, i), "length") for i in range(N)]
    return BR.commit(spans, BR.Layout(check, 0, 1, 4, 4, N, BR.run_root("test-serve", 0), inline), 0)


def _interior(layout: BR.Layout, previous: str, program: Program | None = None) -> tuple[dict, dict]:
    """The replay side's interior registration over the Q sampled spans' proof blocks (PAIR's instances, or
    ``program``'s, a window of templates'), at stand-in roots."""
    m = Q * P.units(Program(layout.replay), layout.query)
    program, query = program or P.population_program(PAIR, m), layout.query
    part = P.partition(program, query)
    ports = {v: {"leaves": m, "domain": R.window_domain(part["program"], P.digest(part), v, s, m, layout.run)}
             for v, s in INTERIOR}
    roots = [{"port": v, "root": f"{i + 1:02x}" * 64, **ports[v]} for i, (v, _) in enumerate(INTERIOR)]
    rec = R.record(program, query, {"law": "subset", "k": m}, S.SCHEME, roots,
                   {"kind": "served", "run": layout.run, "row": 0, "interior": True}, previous=previous)
    return rec, ports


def _samples_q(got: TS.First) -> bool:
    """The first stage sampled Q of the class's spans, a subset."""
    return got.sample.draw["law"] == "subset" and len(got.sample.units) == Q


def test_the_replay_side_resumes_the_first_stage_and_samples_the_second():
    """The first stage samples Q of the class's N spans on the challenger's coins after the boundary registration;
    ``stage1.json`` alone gives the replay side the same registration and sample on the bytes it read; its interior
    registration is logged after it, and the second stage samples K proof blocks in each sampled span on the next
    challenger's coins, opened after the interior registration's digest."""
    c = _class()
    got = TS.first(c.public(), PAIR, RULE, FC.Challenger("first"))
    assert got.layout.n_rho == N_RHO and got.registration.record["population"] == N
    assert _samples_q(got) and got.read
    handoff = json.loads(json.dumps(got.stage1()))
    assert handoff["units"] == list(got.sample.units) and handoff["spec"] == got.stream.spec.digest()
    again = TS.resume(handoff, PAIR, FC.Challenger("second"))
    assert (again.registration.digest, again.sample.units) == (got.registration.digest, got.sample.units)
    ireg, second = again.second(*_interior(again.layout, again.registration.digest))
    assert ireg.order == again.registration.order and again.coins.live
    assert second.replay_sample == again.sample and len(second.units) == Q * K
    assert [u // N_RHO for u in second.units] == list(range(Q))


def test_an_inlined_checks_second_stage_samples_the_template_its_body_calls():
    """An inlined check (`commit.json`'s ``inline``): the class's stream checks the template WRAP's body calls, so the
    replay side resumes it from ``stage1.json`` alone and the second stage samples among PAIR's instances, two an
    instance of WRAP, in each sampled span."""
    got = TS.first(_class(WRAP, inline=True).public(), WRAP, RULE, FC.Challenger("first"))
    assert got.layout.inline and got.stream.spec.check.query == P.template_instance_query(PAIR)
    again = TS.resume(json.loads(json.dumps(got.stage1())), WRAP, FC.Challenger("second"))
    assert (again.registration.digest, again.sample.units) == (got.registration.digest, got.sample.units)
    _, second = again.second(*_interior(again.layout, again.registration.digest))
    assert len(second.units) == Q * K and [u // (2 * N_RHO) for u in second.units] == list(range(Q))


def test_a_class_of_two_check_templates_samples_each_at_its_own_count(tmp_path, monkeypatch, capsys):
    """Counts by template (``--k COUNTS.json``, `Rule.k` a map): a class whose inlined check calls PAIR and SWAP samples
    three of a sampled span's eight PAIR instances and one of its four SWAP instances, the rule carried through
    the hand-off to the replay side's stream. A file is read alone or as a counts record's ``k``; one that names a
    template twice, gives a count that is no whole number of at least one, misses one of the check's templates or names
    another is refused by name, and the command refuses it, and a class that is not the check's, before it opens a
    challenger session. The command lays each class out under the verifier's own stop rule (``--stop``) as it holds
    the counts to it: a stop class's counts are held to its check's templates, and without the rule it is refused by
    name."""
    c, (pair, swap) = _class(BOTH, inline=True), (P.template_id(t) for t in (PAIR, SWAP))
    ks = {pair: 3, swap: 1}

    def file(text: str) -> str:
        (tmp_path / "k.json").write_text(text)
        return str(tmp_path / "k.json")
    assert TS.counts("2") == 2 and TS.counts(file(json.dumps(ks))) == ks
    assert TS.counts(file(json.dumps({"schema": TS.COUNTS, "sampled": Q, "k": ks}))) == ks
    rule = TS.Rule(RULE.replay, TS.counts(file(json.dumps(ks))), INTERIOR, 60.0)
    assert TS.Rule.from_json(json.loads(json.dumps(rule.to_json()))) == rule
    assert TS.counts_refused(ks, c.layout.templates) is None and TS.counts_refused(2, c.layout.templates) is None

    got = TS.first(c.public(), BOTH, rule, FC.Challenger("first"))
    assert got.stream.cells == (2 * N_RHO, N_RHO) and got.stream.counts == (3, 1)
    handoff = json.loads(json.dumps(got.stage1()))
    assert handoff["rule"]["k"] == ks
    again = TS.resume(handoff, BOTH, FC.Challenger("second"))
    assert again.stream.spec.selection == S.TwoStage(c.layout.grain, RULE.replay, ks)
    _, second = again.second(*_interior(again.layout, again.registration.digest, again.stream.interior_program(Q)))
    assert second.law == S.second_law(again.sample.units, (2 * N_RHO, N_RHO), (3, 1))
    pairs = [u for u in second.units if u < Q * 2 * N_RHO]
    assert sorted(u // (2 * N_RHO) for u in pairs) == [0, 0, 0, 1, 1, 1] and len(second.units) - len(pairs) == Q

    for text, why in (('{"%s": 3, "%s": 1, "%s": 2}' % (pair, swap, pair), f"names {pair} twice"),
                      (json.dumps({pair: 3, swap: 0}), "whole number of at least one"),
                      (json.dumps({pair: 3, swap: 1.5}), "whole number of at least one"),
                      (json.dumps({"schema": "other/v1", "k": ks}), "not a spot-two-stage-counts/v1 one"),
                      (json.dumps([3, 1]), "no counts file")):
        with pytest.raises(ValueError) as refused:
            TS.counts(file(text))
        assert why in str(refused.value)
    assert TS.counts_refused({pair: 3}, c.layout.templates) == (
        f"the counts name each of the check's templates once: missing {swap}")
    assert TS.counts_refused({**ks, "Other_v0{}": 1}, c.layout.templates) == (
        "the counts name each of the check's templates once: Other_v0{} is no template of the check")
    with pytest.raises(ValueError, match="name each of the check's templates once"):
        TS.first(c.public(), BOTH, TS.Rule(RULE.replay, {pair: 3}, INTERIOR, 60.0), FC.Challenger("first"))

    (tmp_path / "p4-o4").mkdir()
    (tmp_path / "p4-o4" / BR.COMMIT).write_text(json.dumps(c.public()))
    monkeypatch.setattr(TS, "_first_all", lambda *a, **k: pytest.fail("a refused counts file reached the first stage"))
    for check, text, why in (
            (BOTH, json.dumps({pair: 3}), f"class p4-o4: the counts name each of the check's templates once: missing {swap}"),
            (BOTH, '{"%s": 3, "%s": 1, "%s": 2}' % (pair, swap, swap), f"names {swap} twice"),
            (PAIR, json.dumps(ks), f"class p4-o4: a {BR.FORMAT} commitment of check {P.template_id(BOTH)}, not {pair}")):
        monkeypatch.setattr(TS, "template_fn", lambda t, check=check: check)
        with pytest.raises(SystemExit):
            TS.main(["first", "--boundary", str(tmp_path / "p4-o4"), "--check", P.template_id(BOTH), "--replay", RULE.replay,
                     "--k", file(text), "--challenger", "local", "--main", str(tmp_path / "main"), "--out", str(tmp_path)])
        assert why in capsys.readouterr().err

    (tmp_path / "stopped").mkdir()
    (tmp_path / "stopped" / BR.COMMIT).write_text(json.dumps(_stop_class().public()))
    monkeypatch.setattr(TS, "template_fn", lambda t: PAIR)

    def stopped(text: str, *given: str) -> int:
        return TS.main(["first", "--boundary", str(tmp_path / "stopped"), "--check", pair, "--replay", RULE.replay, "--k",
                        file(text), *given, "--challenger", "local", "--main", str(tmp_path / "main"), "--out", str(tmp_path)])
    held = ("--stop", json.dumps(STOP.to_json()))
    for given, why in ((held, f"class stopped: the counts name each of the check's templates once: {swap} is no template"),
                       ((), f"class stopped: a {BR.FORMAT} commitment under the stop rule")):
        with pytest.raises(SystemExit):
            stopped(json.dumps(ks), *given)
        assert why in capsys.readouterr().err
    reached: list = []
    monkeypatch.setattr(TS, "_first_all", lambda a, pubs, check, rule, window, stop: reached.append((a.k, stop)) or [
        {"commit": pubs[0]}])
    assert stopped(json.dumps({pair: 2}), *held) == 0 and reached == [({pair: 2}, STOP)]


def test_resume_refuses_another_rule_registration_or_sample():
    got = TS.first(_class().public(), PAIR, RULE, FC.Challenger("first"))
    stage1 = json.loads(json.dumps(got.stage1()))
    with pytest.raises(ValueError, match="hand-off"):
        TS.resume({**stage1, "format": "other"}, PAIR)
    with pytest.raises(ValueError, match="Spec"):
        TS.resume({**stage1, "rule": TS.Rule(RULE.replay, K + 1, INTERIOR, 60.0).to_json()}, PAIR)
    with pytest.raises(ValueError, match="registers to another digest"):
        TS.resume({**stage1, "registration": "00" * 64}, PAIR)
    others = [u for u in range(N) if u not in got.sample.units][:Q]
    with pytest.raises(ValueError, match="other spans"):
        TS.resume({**stage1, "units": others}, PAIR)


def test_a_hand_off_that_names_no_window_root_resumes_under_its_commitments_run():
    """The first stage registers under the window's open root, the commitment's ``run`` (`_first_all`), which the
    outcome holds its session's evidence to. A hand-off that names the root resumes under it; one that names none
    resumes under the commitment's ``run``, to the same registration and sample."""
    pub = _class().public()
    got = TS.first(pub, PAIR, RULE, FC.Challenger("first"), window_root=pub["run"])
    stage1 = json.loads(json.dumps(got.stage1()))
    assert "window_root" not in stage1 and TS.first_root(stage1) == pub["run"]
    assert TS.first_root({**stage1, "window_root": "ab" * 64}) == "ab" * 64
    again = TS.resume(stage1, PAIR, window_root=TS.first_root(stage1))
    assert again.registration.digest == got.registration.digest and again.sample.units == got.sample.units


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
    assert _samples_q(got)


@pytest.mark.parametrize("layers", [(), (0,)])
def test_a_class_of_verdicts_declares_and_registers_its_blocks_return(layers):
    """A class that commits its replay block's return (`Layout.verdicts`): its Spec declares the return a served value
    after ``b.ingress`` and ``b.egress``, named under its replay block and before the tables where the class has history
    tables; its first stage registers its root, from ``commit.json`` alone, before the coins commit and samples; and the
    replay side resumes from the first stage's hand-over alone."""
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
    assert _samples_q(got)
    again = TS.resume(json.loads(json.dumps(got.stage1())), SAME, FC.Challenger("second"))
    assert (again.registration.digest, again.sample.units) == (got.registration.digest, got.sample.units)


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
    names = [BT.name(layout.run, layout.prompt, layout.outputs, u, 1) for u in got.sample.units]
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

    names = [BT.name(layout.run, prompt, outputs, int(u), 0) for u in got.sample.units]
    sampled = [hs[n] for n in names]
    toks = ((1, 5, 0, 2), (0, 4, 3, 2))
    it = IR.interior([IR.Span(f"s{k}", member, IR.table_reads(t, toks[k]), [[0, cells]], "p" * 64)
                      for k, t in enumerate(sampled)], cells, got.registration.digest, got.sample.units,
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
    own = SS.Own(tuple(got.sample.units), (part,), it.record, got.record, (),
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
    assert TS.Rule.from_json({k: v for k, v in RULE.to_json().items() if k != "span_values"}) == RULE
    with pytest.raises(ValueError, match="rate law"):
        TS.Rule(f"subset:{Q}", K, INTERIOR, 60.0)
    spec = TS.stream_of(_class().layout, RULE, FC.Challenger()).spec
    assert [(v.name, v.phase) for v in spec.values] == [("b.ingress", "serve"), ("b.egress", "serve"),
                                                        ("x", "interior"), ("out", "interior")]
    assert spec.selection == S.TwoStage(_class().layout.grain, RULE.replay, K)


def test_a_class_under_a_stop_rule_registers_its_boundary_values_and_is_sampled_like_any_other():
    """A serve's stop rule cuts a class's boundary rows into `b.request`, `b.prompt` and `b.egress`: the class's Spec
    names those, its first stage registers their record and samples its spans, and its stop word, the stop record's
    port, is no value of this stream. Its spans set ``ignore_eos``, so the stop id one of them serves ends none."""
    stop = BS.Stop((3,), 64, 4)
    spans = [BR.Span(f"r{i}", (i, i + 1, i + 2), GREEDY, (7, 8, 9, i), "length") for i in range(N)]
    c = BR.commit(spans, BR.Layout(PAIR, 0, 1, 4, 4, N, BR.run_root("test-serve", 0), stop=stop), 0)
    assert list(c.public()["values"]) == [*BR.PARTS, BR.STOP]
    got = TS.first(c.public(), PAIR, RULE, FC.Challenger("first"), stop=stop)
    assert [v.name for v in got.stream.spec.values if v.phase == "serve"] == list(BR.PARTS)
    assert [r["port"] for r in got.registration.record["roots"]] == list(BR.PARTS)
    assert got.registration.record["population"] == N and len(got.sample.units) == Q


#: a serve's stop rule: one stop id, a model of 64 ids, responses of at most 4
STOP = BS.Stop((3,), 64, 4)
ROOT = FC.WINDOW["root"]


def _stop_class(held: bool = True) -> BR.Committed:
    """A class of N spans under `STOP` on the window `ROOT`: a stop class, each span ended at the stop id or at
    ``max_tokens``; or (not ``held``) one of aborted spans, which has no stop block."""
    sampling = {**GREEDY, "ignore_eos": False}
    ends = [((7, 8, 3), "stop"), ((7, 8, 9, 9), "length")] if held else [((7, 8), "abort")]
    spans = [BR.Span(f"r{i}", (i, i + 1, i + 2), sampling, *ends[i % len(ends)]) for i in range(N)]
    return BR.commit(spans, BR.Layout(PAIR, 0, 1, 4, 4 if held else 2, N, ROOT, stop=STOP, held=held), 0)


class OneSession(CC.ChallengerCoins):
    """One in-process challenger session of the window's family and ``ranks`` ranks' (`a5_coins.families`), as a
    window's first stage has: the session's client (`open`, `open_many`, `close`), whose signed release log holds every
    family's roots and releases in the order they were asked for; and coins that take the ranks' families in order, a
    commitment each."""

    def __init__(self, seed: str, ranks: int = 6, coins: int = 64) -> None:
        layout = AC.families(dict.fromkeys(range(ranks), coins))
        self.directory = Path(tempfile.mkdtemp(prefix="one-session-"))
        weakref.finalize(self, shutil.rmtree, self.directory, True)
        self.ledger = FC.Ledger(self.directory / "windows.jsonl", FC.at(0, 3))
        self.ledger.open("s-1", FC.WINDOW, withheld=True)
        self.live = Ch.LiveChallenger(FC.SCHEME, 0, layout, FC.entropy(f"{seed}/s-1"))
        self.session = FC.R.open_session(self.live, FC.pinned_key(), f"{FC.HOST}.s-1", FC.at(1, 2), self.directory,
                                         "s-1", withhold=True)
        (self.directory / FC.R.HELLOS).write_text("")
        self.commitment, self.families = self.live.commitment, self.live.families
        self._lines: list[str] = []
        self._clock, self._files = itertools.count(), None
        super().__init__(*(Ch.LiveRecord(self, AC.family(r)) for r in range(ranks)))

    def open_many(self, requests: list[tuple[str, int, bytes]]) -> list[Ch.LiveOpening]:
        out = []
        for label, index, after in requests:
            root = {"event": "root", "index": index, "label": label, "received_at": next(self._clock),
                    "root": bytes(after).hex()}
            out.append(self.live.open(label, index, after))
            release = {"after": bytes(after).hex(), "event": "release", "index": index, "label": label,
                       "released_at": next(self._clock)}
            self._lines += [FC.canonical_json_text(root), FC.canonical_json_text(release)]
        return out

    def open(self, label: str, index: int, after: bytes) -> Ch.LiveOpening:
        return self.open_many([(label, index, after)])[0]

    def close(self) -> None:
        pass

    def files(self) -> dict:
        """The session's published files under the verifier's key, the auditor's ledger and the host's pins; the first
        call closes the session."""
        if self._files is None:
            with open(self.directory / FC.R.RELEASES, "a", encoding="utf-8") as log:
                log.write("".join(line + "\n" for line in self._lines))
            self.session.lines.extend(self._lines)
            FC.R.close_session(self.session, None)
            self.ledger.end(self.session.id, self.session.directory, True)
            self._files = {"key": FC.pinned_key().to_json(), "ledger": self.ledger.lines, "main": FC.MAIN,
                           "pins": FC.HOST_PINS, "session": FC.R.session_files(self.directory)}
        return self._files

    def evidence(self, digest: str) -> dict:
        """`coins.coins_reason`'s evidence for the sample after ``digest``."""
        return {**self.files(), "segment": next(s for s in self.segments if s.after == bytes.fromhex(digest)).to_json()}


def _start(one: OneSession):
    """`a5_coins.start` in process: its ranks' turns on ``one``, opened and published as `start`'s thread does a served
    session's, and refused, as it refuses, when a rank ended before it planned."""
    def start(challenger: str, root: str, ranks: list[int], dest: Path, main: object, **kwargs: object):
        turns = AC.Draws(root, ranks)

        def finish() -> None:
            needs = turns.wait_planned()
            plan = {"needs": {str(r): n for r, n in sorted(needs.items())}}
            missing = sorted(set(ranks) - set(needs))
            if missing:
                why = f"rank(s) {missing} ended before planning their draw (refused before it, or failed)"
                turns.refuse(f"{why}: the root's one session was not opened")
                turns.publish({"error": f"the challenger's session was not opened: {why}", "plan": plan})
                return
            turns._client, turns._window = one, CC.ChallengerCoins(Ch.LiveRecord(one, AC.WINDOW_FAMILY))
            turns._pre = turns._window.commit(bytes.fromhex(root))
            turns._window(Ch.COIN_BYTES)
            turns._opened.set()
            turns.wait_closed()
            turns.publish({**one.files(), "segments": turns.released()["segments"], "trusted": {}, "plan": plan})
        finisher = threading.Thread(target=finish, daemon=True)
        finisher.start()
        return NS(shutdown=lambda: None), turns, finisher
    return start


def _args(tmp_path: Path, **more: object) -> NS:
    return NS(**{"challenger": "test", "out": str(tmp_path), "main": None, "ssh_key": None, "peer": None,
                 "challenger_key": None, "replay": f"rate:{Q}/{N}", "k": K, "proof_deadline": 60.0, "delta": 0.5,
                 "stop_design": None, **more})


def test_a_stop_classs_stop_blocks_register_after_its_boundary_record_and_are_sampled_on_a_stream_of_their_own():
    """After a stop class's first stage its stop record registers on a one-stage stream of the verifier's stop block
    whose log starts with the class's boundary record: one stop block a span, `b.stop` its one port, its inputs that
    record's `b.request` and `b.egress` as output values. It is sampled at a rate over the class's N stop blocks on
    coins opened after its own registration's digest. Refused: a boundary record that is another class's, another
    rule than the commitment's, and a sampling design that is a fixed count."""
    pub = json.loads(json.dumps(_stop_class().public()))
    got = TS.first(pub, PAIR, RULE, FC.Challenger("first"), stop=STOP)
    coins = FC.Challenger("stop")
    s = TS.stop_first(pub, PAIR, STOP, f"rate:{Q}/{N}", coins, got.record)
    assert s["record"]["previous"] == got.registration.digest and s["record"]["population"] == N
    assert [r["port"] for r in s["record"]["roots"]] == [BR.STOP]
    assert sorted(e["value"] for e in s["record"]["registered"]) == sorted(BS.STOP_READS)
    assert s["record"]["window"] == {**got.record["window"], "population": "stop"}
    assert s["sample"]["population"] == N and len(s["units"]) == Q and set(s["units"]) <= set(range(N))
    assert s["spec"] == TS.stop_spec(got.layout, f"rate:{Q}/{N}").digest() != got.stream.spec.digest()
    read, after = bytes.fromhex(s["read"]), bytes.fromhex(s["registration"])
    assert s["family"] == FC.LABEL
    assert read and CC.coins_reason(coins.evidence(s["registration"]), ROOT, FC.LABEL, after, read, AC.lms_verify) is None
    other = TS.first(json.loads(json.dumps(_stop_class().public())), PAIR, RULE, FC.Challenger("other"), stop=STOP)
    with pytest.raises(ValueError, match="not this commitment's values"):
        TS.stop_first(pub, PAIR, STOP, f"rate:{Q}/{N}", FC.Challenger("x"), other.record)
    with pytest.raises(ValueError, match="not the verifier's own"):
        TS.stop_first(pub, PAIR, BS.Stop((7,), 64, 4), f"rate:{Q}/{N}", FC.Challenger("y"), got.record)
    with pytest.raises(ValueError, match="a fixed count samples each proof block with K/N"):
        TS.stop_first(pub, PAIR, STOP, "subset:2", FC.Challenger("z"), got.record)


def test_a_stop_samples_order_is_read_from_the_challengers_signed_log():
    """`stop_order` on a session's published evidence: it holds when the log shows the boundary registration's coins,
    then the stop registration's, each in a family of its own, and the stop sample read exactly the latter. It names
    why otherwise: no published log; no family; a coin released after the stop registration left unread, or bytes that
    are not those coins'; the boundary registration's coins handed in as the stop sample's, or the stop sample's held
    to the boundary registration's family; another window's root; a log the challenger's close does not sign (a
    prover's own ordering of the same lines); the stop registration's coins released first; no coin after the boundary
    registration at all; and a stop sample whose family holds the boundary registration's coins before its own, the
    one family a prover that registers again after reading would use."""
    b, s = "b0" * 64, "c0" * 64

    def session(seed: str, *commitments: str) -> tuple[OneSession, bytes, dict[str, str]]:
        one, read, family = OneSession(seed), {}, {}
        for c in commitments:
            one.commit(bytes.fromhex(c))
            read[c], family[c] = one(2 * Ch.COIN_BYTES), one.label
        return one, read[s], family

    def order(evidence: dict | None, read: bytes, family: str | None, root: str = ROOT) -> str | None:
        return TS.stop_order(evidence, root, family, b, s, read, AC.lms_verify)

    one, read, family = session("order", b, s)
    assert family == {b: AC.family(0), s: AC.family(1)}
    ev = one.evidence(s)
    assert order(ev, read, family[s]) is None
    assert "published no signed log" in order(None, read, family[s])
    assert "no coins of a challenger's family" in order(ev, read, None)
    assert "a coin opened and not read" in order(ev, read[:Ch.COIN_BYTES], family[s])
    assert "are not its coins'" in order(ev, bytes(len(read)), family[s])
    assert f"not the challenger's {family[s]} family's" in order(one.evidence(b), read, family[s])
    assert f"not the challenger's {family[b]} family's" in order(ev, read, family[b])
    assert order(ev, read, family[s], "22" * 64) is not None
    log = ev["session"]["log"]
    assert len(log) == 9
    forged = {**ev, "session": {**ev["session"], "log": [log[0], *log[5:], *log[1:5]]}}
    assert "release log" in order(forged, read, family[s])
    early, read, family = session("early", s, b)
    assert "the boundary registration's coins come first" in order(early.evidence(s), read, family[s])
    alone, read, family = session("alone", s)
    assert "no coin released after the class's boundary registration" in order(alone.evidence(s), read, family[s])

    shared = FC.Challenger("shared")
    shared.commit(bytes.fromhex(b))
    shared(2 * Ch.COIN_BYTES)
    again = shared.record
    shared.record = NS(label=again.label, roots=[])
    segment = CC.ChallengerCoins.commit(shared, bytes.fromhex(s))
    shared.record, segment.start = again, len(again.roots)
    shared.sessions[bytes.fromhex(s)] = shared.sessions[bytes.fromhex(b)]
    read = shared(2 * Ch.COIN_BYTES)
    assert order(shared.evidence(s), read, FC.LABEL) == f"the stop sample's coins: {CC.FAMILY_ROOT}"


def test_each_stop_class_is_a_rank_more_of_the_windows_session_and_a_wrong_order_refuses_the_window(tmp_path, monkeypatch):
    """A window of a stop class and a class without a stop block, under the verifier's rule, on one session: three
    ranks, the stop class's stop blocks the third, registered and sampled after its first stage at the rate given (the
    first stage's when none is), and its order held by the session's signed log. A stop sample whose order does not
    hold refuses every class of the window, each by name, with no sample handed on. A stop class whose stop blocks
    register nothing is refused itself and the other class goes on. A stop class that cannot plan its stop sample (a
    sampling design that is no rate; a verifier holding no rule) refuses the window's whole first stage, and no
    session opens. The sessions here are what no cheaper test has: the ranks' turns on one session and its log's
    order."""
    pubs = [json.loads(json.dumps(_stop_class(held).public())) for held in (True, False)]

    def run(seed: str, stop: BS.Stop | None = STOP, **more: object) -> list[dict]:
        monkeypatch.setattr(AC, "start", _start(OneSession(seed)))
        return TS._first_all(_args(tmp_path, **more), pubs, PAIR, RULE, None, stop)

    stopped, plain = run("window", stop_design=f"rate:1/{N}")
    assert len(stopped["units"]) == len(plain["units"]) == Q and "stop" not in plain
    mine = stopped["stop"]
    assert len(mine["units"]) == 1 and mine["order"] == {"held": True, "reason": None}
    assert mine["record"]["previous"] == stopped["registration"] and mine["design"] == f"rate:1/{N}"
    assert sorted(stopped["session"]["plan"]["needs"]) == ["0", "1", "2"]
    received = [bytes.fromhex(json.loads(x)["root"]) for x in mine["evidence"]["session"]["log"] if '"event":"root"' in x]
    at = {name: [i for i, r in enumerate(received) if r == bytes.fromhex(d)]
          for name, d in (("boundary", stopped["registration"]), ("stop", mine["registration"]))}
    assert at["boundary"] and at["stop"] and at["boundary"][-1] < at["stop"][0]

    with monkeypatch.context() as m:
        m.setattr(TS, "stop_order", lambda *a: "the log shows otherwise")
        out = run("wrong")
    assert [o["refused"] for o in out] == [[TS.ORDER_REFUSED]] * 2 and not any("units" in o for o in out)
    assert all("class 0 (prompt 4, outputs 4): the log shows otherwise" in o["error"] for o in out)
    assert out[0]["stop"]["order"] == {"held": False, "reason": "the log shows otherwise"}
    assert out[0]["stop"]["design"] == f"rate:{Q}/{N}" and len(out[0]["stop"]["units"]) == Q

    def nothing(*a: object, **k: object) -> dict:
        raise ValueError("no stop record")
    with monkeypatch.context() as m:
        m.setattr(TS, "stop_first", nothing)
        stopped, plain = run("unregistered")
    assert stopped["refused"] == [TS.STOP_REFUSED] and "units" not in stopped and "no stop record" in stopped["error"]
    assert len(plain["units"]) == Q

    stopped, plain = run("no-rate", stop_design="subset:2")
    assert "units" not in stopped and "units" not in plain and "session was not opened" in plain["session"]["error"]
    assert "a fixed count samples" in stopped["error"] and "planned no sample" in stopped["stop"]["error"]
    assert "rank(s) [0, 2] ended before planning" in plain["error"]
    for o in run("no-rule", stop=None):
        assert "units" not in o and "not the verifier's own" in o["error"] and "was not opened" in o["session"]["error"]
