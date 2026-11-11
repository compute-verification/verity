"""A served window's two-stage audit as `two_stage.py` drives it: one size class's boundary registration
(`verity_vllm.pipeline.boundary_rows`) on its two-stage stream, sampled on a test's own challenger (`fake_challenger`);
the replay side's stream again from ``stage1.json`` alone, sampling the same spans on the bytes the first stage read; the
interior registration over the sampled spans' proof blocks and the second stage on the next challenger's coins; and what
`resume` refuses. A toy check template of one gate's worth."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat
from verity.circuits.types import Array, Value as Bits
from verity.protocols.spot.device import challenger as Ch
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import partition as P, registration as R
from verity_vllm.pipeline import boundary_rows as BR

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


def test_a_rule_round_trips_and_its_spec_holds_the_boundary_values_then_the_interior_ones():
    assert TS.Rule.from_json(json.loads(json.dumps(RULE.to_json()))) == RULE
    with pytest.raises(ValueError, match="rate law"):
        TS.Rule(f"subset:{Q}", K, INTERIOR, 60.0)
    spec = TS.stream_of(_class().layout, RULE, FC.Challenger()).spec
    assert [(v.name, v.phase) for v in spec.values] == [("b.ingress", "serve"), ("b.egress", "serve"),
                                                        ("x", "interior"), ("out", "interior")]
    assert spec.selection == S.TwoStage(_class().layout.grain, RULE.replay, K)
