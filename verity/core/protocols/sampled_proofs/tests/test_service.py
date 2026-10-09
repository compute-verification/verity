"""The proof service: the Spec's types, the public-items rule, and a Stream's register, select, deliver and outcome over
the one-stage audit, with a stub verifier, injected coins and an injected clock."""

from __future__ import annotations

import random

import pytest

import verity.ml.circuits.ops.prims  # noqa: F401  (the one-stage tests' primitives)
import verity.ml.circuits.ops.scalar  # noqa: F401
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.sampled_proofs import service as S, two_stage as T
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R

XOR8 = PrimitiveDefinition("ServiceTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("ServiceTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
OTHER = bind(CompositeDefinition("ServiceTestOther", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                 lambda B, S_, x: B.call(XOR8, x[1], x[0]), register=False))
QUERY = P.template_instance_query(PAIR)
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-service-test"
SESSION = "5e" * 64
DELTA = 2.0 ** -40
#: What one-stage's record shows besides the salted roots, declared as step 0 declares it.
STEP0 = tuple(S.Public(i, "structure", r) for i, r in (
    ("population", "N, the draw's range"), ("window", "provenance: served, on test prompts (step 0)"),
    ("leaves:x", "x's leaf count (step 0)"), ("domain:x", "x's served domain (step 0)"),
    ("leaves:out", "out's leaf count (step 0)"), ("domain:out", "out's served domain (step 0)")))
VALUES = (S.Value("x", SCHEMA, "serve"), S.Value("out", SCHEMA, "serve"))


def _spec(selection=S.Law("subset:2"), public=STEP0, proof_deadline=None, public_outputs=(), values=VALUES) -> S.Spec:
    return S.Spec(values, selection, S.Check(P.program_digest(Program(PAIR)), QUERY, public_outputs=public_outputs,
                                             proof_deadline=proof_deadline), public, b"service-test")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _stream(spec=None, budget=2, clock=None, coins=b"\x07", sampler=None) -> S.Stream:
    return S.Stream(spec or _spec(), PAIR, budget=budget, delta=DELTA, challenger="challenger-test",
                    coins=(lambda n: coins * n) if isinstance(coins, bytes) else coins, clock=clock or Clock(),
                    sampler=sampler)


def _ports(n: int) -> dict:
    part = P.partition(P.population_program(PAIR, n), QUERY)
    return {p: {"leaves": leaves, "domain": R.served_domain(part["program"], P.digest(part), p, SCHEMA, leaves, RUN)}
            for p, leaves in (("x", n), ("out", 2 * n))}


def _record(n: int = 6, law=None, row: int = 0, previous: str | None = None, work_table=None, **change) -> dict:
    ports = _ports(n)
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **ports[p]} for i, p in enumerate(ports)]
    rec = R.record(P.population_program(PAIR, n), QUERY, law or {"law": "subset", "k": 2}, S.SCHEME, roots,
                   {"kind": "served", "run": RUN, "row": row}, previous=previous, work_table=work_table)
    return {**rec, **change}


#: A two-stage Spec's values: the RU boundary `x`, committed by serve time, and the interior `out`, after the draw.
TWO = (S.Value("x", SCHEMA, "serve"), S.Value("out", SCHEMA, "interior"))
ROOTS = {"x": "01" * 64, "out": "02" * 64}


def _two_spec(replay: str = "subset:2", check="all", **kw) -> S.Spec:
    return _spec(S.TwoStage(S.Law(replay), check), values=TWO, **kw)


def _half(port: str, n: int, law=None, row: int = 0, previous: str | None = None, root: str | None = None,
          work_table=None) -> tuple[dict, dict]:
    """One stage's record and the challenger's copy of its port: the boundary's (`x`, on serve's window) or the
    interiors' (`out`, over the proved RUs, on a window of its own: R6 refuses a window logged before)."""
    ports = {port: _ports(n)[port]}
    roots = [{"port": port, "root": root or ROOTS[port], **ports[port]}]
    window = {"kind": "served", "run": RUN, "row": row, **({} if port == "x" else {"stage": "interior"})}
    rec = R.record(P.population_program(PAIR, n), QUERY, law or {"law": "subset", "k": 2}, S.SCHEME, roots, window,
                   previous=previous, work_table=work_table)
    return rec, ports


def _interiors(draw: S.Draw, row: int = 0, **kw) -> tuple[dict, dict]:
    n = len(draw.proved)
    return _half("out", n, {"law": "subset", "k": n}, row=row, previous=draw.registration.digest, **kw)


def _lean(accepted=True, sessions=(SESSION,), **extra) -> dict:
    return {"lean": {"of_record": True, "accepted": accepted, "digest": "dd" * 64, "sessions": list(sessions), **extra}}


def _round(stream: S.Stream, accepted=True, n: int = 6) -> S.Outcome:
    previous = R.digest(stream._entries[-1].registration.record) if stream._entries else None
    reg = stream.register(_record(n, row=len(stream._entries), previous=previous), _ports(n))
    draw = stream.select(reg)
    assert stream.deliver(draw, [SESSION]) is None
    return stream.outcome(draw, _lean(accepted), {"statement": "stub"})


def test_round_trip_register_select_deliver_outcome():
    stream = _stream()
    reg = stream.register(_record(), _ports(6))
    assert reg.order == 0 and reg.receipt["received"] == "0" and reg.digest == R.digest(_record())
    draw = stream.select(reg)
    assert draw.draw["law"] == "subset" and draw.draw["population"] == 6 and len(draw.units) == 2
    assert all(0 <= u < 6 for u in draw.units) and draw.units == sorted(draw.units)
    assert stream.deliver(draw, [SESSION]) is None
    out = stream.outcome(draw, _lean(), {"statement": "stub"})
    assert out.accepted and out.verdict.ok and out.verdict.codes == ()
    assert out.profile is not None and out.profile.bound() == 4 and out.exhaustive is None and out.terms == S.Terms(DELTA)
    assert stream.symbols == ("accept",) and stream.failures == 0 and not stream.stopped
    body = stream.records[0]
    assert S._sha512(body) == out.record and body["sessions"] == [SESSION] and body["registration"] == reg.digest
    assert body["audit"]["accepted"] and body["audit"]["integrity_profile"]["evidence"]["sessions"] == [SESSION]


def test_the_draw_is_the_challengers_coins_and_nothing_else():
    def drawn(coins):
        stream = _stream(coins=coins)
        return stream.select(stream.register(_record(), _ports(6))).units

    assert drawn(b"\x07") == drawn(b"\x07")
    assert len({tuple(drawn(bytes([b]))) for b in range(16)}) > 1


def test_all_proves_every_unit_and_gives_the_exhaustive_fact():
    stream = _stream(_spec(S.All()))
    reg = stream.register(_record(law={"law": "subset", "k": 6}), _ports(6))
    draw = stream.select(reg)
    assert draw.units == list(range(6))
    stream.deliver(draw, [SESSION])
    out = stream.outcome(draw, _lean(), {})
    assert out.accepted and out.exhaustive == S.Exhaustive(6) and out.profile is None and out.terms == S.Terms(None)


def test_the_rule_refuses_an_extra_field():
    assert "keys" in S.check_public(_spec(), {**_record(), "tiles": [2, 3]}).codes
    roots = [dict(r, call=237) for r in _record()["roots"]]
    v = S.check_public(_spec(), _record(roots=roots))
    assert "keys" in v.codes and "roots.x.call" in v.detail["keys"]
    with pytest.raises(S.Refused) as e:
        _stream().register({**_record(), "tiles": [2, 3]}, _ports(6))
    assert "keys" in e.value.verdict.codes


def test_the_rule_refuses_a_per_call_root_in_the_clear():
    rec = _record()
    per_call = [dict(rec["roots"][0], port="x/call-0"), dict(rec["roots"][0], port="x/call-1"), rec["roots"][1]]
    v = S.check_public(_spec(), {**rec, "roots": per_call})
    assert v.codes == ("roots",) and v.detail["roots"] == ["x/call-0", "x/call-1"]
    twice = [rec["roots"][0], dict(rec["roots"][0], root="ee" * 64), rec["roots"][1]]
    assert S.check_public(_spec(), {**rec, "roots": twice}).detail["roots"] == ["x"]


def test_the_rule_refuses_a_leaf_count_unless_declared():
    spec = _spec(public=tuple(p for p in STEP0 if p.item != "leaves:x"))
    v = S.check_public(spec, _record())
    assert v.codes == ("undeclared",) and v.detail["undeclared"] == ["leaves:x"]
    with pytest.raises(S.Refused) as e:
        _stream(spec).register(_record(), _ports(6))
    assert e.value.verdict.detail == {"undeclared": ["leaves:x"]}
    bare = _spec(public=())
    assert S.check_public(bare, _record()).detail["undeclared"] == ["population", "window", "domain:x", "leaves:x",
                                                                   "domain:out", "leaves:out"]


def test_the_rule_refuses_a_template_population_unless_declared():
    law = {"law": "stratified", "k": 2}
    rec = _record(law=law)
    assert rec["law"]["strata"][0]["template"] == PAIR.id
    v = S.check_public(_spec(S.Law("stratified:2")), rec)
    assert v.codes == ("undeclared",) and v.detail["undeclared"] == [f"population:{PAIR.id}"]
    declared = _spec(S.Law("stratified:2"), public=(*STEP0, S.Public(f"population:{PAIR.id}", "structure", "a stratum")))
    assert S.check_public(declared, rec).ok
    stream = _stream(declared)
    draw = stream.select(stream.register(rec, _ports(6)))
    assert draw.draw["law"] == "stratified" and len(draw.units) == 2


def test_the_rule_refuses_an_unsalted_digest():
    v = S.check_public(_spec(), _record(scheme="frame-v3-sha512/sha512"))
    assert v.codes == ("unsalted",)
    for schema in ("sha512/row/v2", "flock-leaf/sha512-unsalted", None):
        with pytest.raises(ValueError, match="hide"):
            S.Value("x", schema, "serve")
    with pytest.raises(ValueError, match="service's"):
        S.Value("W", SCHEMA, "epoch", registered=True)


def test_the_rule_refuses_a_public_kind_outside_the_four():
    for kind in ("secret", "value", "", None):
        with pytest.raises(ValueError, match="kind"):
            S.Public("N", kind, "why")
    with pytest.raises(ValueError, match="reason"):
        S.Public("N", "structure", " ")
    with pytest.raises(ValueError, match="source"):
        S.Public("N", "structure", "why", source="prover")


def test_the_rule_refuses_the_rest():
    assert S.check_public(_spec(), _record(query="00" * 64)).codes == ("query",)
    assert S.check_public(_spec(), _record(anchors=["w"])).detail["undeclared"] == ["anchors"]
    assert S.check_public(_spec(), _record(leaf_layer="ee" * 64)).detail["undeclared"] == ["leaf_layer"]
    reg = [{"value": "W", "root": "ab" * 64, "leaves": 4, "words": 4, "domain": "cd" * 64}]
    assert S.check_public(_spec(), _record(registered=reg)).detail["roots"] == ["W"]
    spec = _spec(values=(*VALUES, S.Value("W", None, "epoch", registered=True)),
                 public=(*STEP0, S.Public("W", "commitment", "the weights' salted root")))
    assert S.check_public(spec, _record(registered=reg)).detail["undeclared"] == ["domain:W", "leaves:W", "words:W"]
    assert S.check_public(_spec(), "a record").codes == ("keys",)


def test_a_spec_states_each_public_item_once_with_its_reason():
    with pytest.raises(ValueError, match="distinct"):
        _spec(public=(*STEP0, STEP0[0]))
    with pytest.raises(ValueError, match="commitment"):
        _spec(public=(*STEP0, S.Public("W", "commitment", "a root")))
    with pytest.raises(ValueError, match="window"):
        _spec(public=(*STEP0, S.Public("credited", "output", "credited work", source="window")))
    with pytest.raises(ValueError, match="output"):
        _spec(public_outputs=("credited",))
    with pytest.raises(ValueError, match="distinct"):
        _spec(values=(*VALUES, VALUES[0]))
    with pytest.raises(ValueError):
        S.Law("subset:two")
    with pytest.raises(ValueError, match="uniform"):
        S.Sequential("subset:4", "B")
    with pytest.raises(ValueError, match="positive"):
        _spec(proof_deadline=0)
    assert _spec().digest() == _spec().digest() != _spec(S.Law("subset:3")).digest()
    for values in (VALUES, tuple(S.Value(v.name, SCHEMA, "interior") for v in VALUES),
                   (*TWO, S.Value("answer", SCHEMA, "in-deadline"))):
        with pytest.raises(ValueError, match="interiors"):
            _spec(S.TwoStage(S.Law("subset:2"), "all"), values=values)
    with pytest.raises(ValueError, match="registered"):
        _spec(S.TwoStage(S.Law("subset:2"), "all"), values=(*TWO, S.Value("W", None, "interior", registered=True)))
    assert _two_spec().digest() != _spec().digest()


def test_a_stream_refuses_what_isnt_built_and_another_unit():
    with pytest.raises(ValueError, match="Sequential"):
        _stream(_spec(S.Sequential("uniform:4", "population")))
    with pytest.raises(ValueError, match="every proof unit"):
        _stream(_two_spec(check=2))
    with pytest.raises(ValueError, match="unit"):
        S.Stream(_spec(), OTHER, budget=1, delta=DELTA, challenger="c")
    for bad in ({"budget": 0}, {"budget": 1.0}, {"delta": 1}, {"delta": 1.5}):
        with pytest.raises(ValueError):
            S.Stream(_spec(), PAIR, **{"budget": 1, "delta": DELTA, "challenger": "c", **bad})


def test_one_stage_refusals_come_through_register():
    stream = _stream()
    with pytest.raises(S.Refused) as e:
        stream.register(_record(), {**_ports(6), "x": {**_ports(6)["x"], "leaves": 7}})
    assert "R3-leaves" in e.value.verdict.codes
    with pytest.raises(S.Refused) as e:
        stream.register(_record(law={"law": "subset", "k": 3}), _ports(6))
    assert "R1-law" in e.value.verdict.codes
    with pytest.raises(S.Refused) as e:
        stream.register(_record(population=0), _ports(6))
    assert stream.register(_record(), _ports(6)).order == 0


def test_no_retries_nothing_redraws_and_an_outcome_is_final():
    stream = _stream()
    reg = stream.register(_record(), _ports(6))
    draw = stream.select(reg)
    with pytest.raises(S.Refused) as e:
        stream.select(reg)
    assert e.value.verdict.codes == ("drawn",)
    with pytest.raises(S.Refused) as e:
        stream.register(_record(row=1, previous=reg.digest), _ports(6))
    assert e.value.verdict.codes == ("open",)
    stream.deliver(draw, [SESSION])
    with pytest.raises(S.Refused) as e:
        stream.deliver(draw, [SESSION])
    assert e.value.verdict.codes == ("delivered",)
    out = stream.outcome(draw, _lean(False), {})
    assert not out.accepted and out.verdict.codes == ("reject",) and out.profile is None
    for call in (lambda: stream.outcome(draw, _lean(True), {}), lambda: stream.deliver(draw, [SESSION])):
        with pytest.raises(S.Refused) as e:
            call()
        assert e.value.verdict.codes == ("final",)
    assert stream.symbols == ("reject",)


def test_the_outcome_is_pending_until_the_verifier_of_record_ran():
    stream = _stream()
    draw = stream.select(stream.register(_record(), _ports(6)))
    with pytest.raises(S.Pending):
        stream.outcome(draw, _lean(), {})
    stream.deliver(draw, [SESSION])
    for verdicts in (_lean(None), {"rust": {"of_record": False, "accepted": True}}):
        with pytest.raises(S.Pending):
            stream.outcome(draw, verdicts, {})
    assert stream.records == () and stream.outcome(draw, _lean(), {}).accepted


def test_the_outcome_binds_the_delivered_sessions():
    stream = _stream()
    draw = stream.select(stream.register(_record(), _ports(6)))
    for bad in ([], ["not a digest"], SESSION):
        with pytest.raises(S.Refused) as e:
            stream.deliver(draw, bad)
        assert e.value.verdict.codes == ("sessions",)
    stream.deliver(draw, [SESSION])
    for sessions in ((), ("ff" * 64,), (SESSION, SESSION)):
        with pytest.raises(S.Refused) as e:
            stream.outcome(draw, _lean(sessions=sessions), {})
        assert e.value.verdict.codes == ("sessions",)
    assert stream.outcome(draw, _lean(), {}).accepted


def test_a_late_proof_is_a_reject():
    clock = Clock()
    stream = _stream(_spec(proof_deadline=60.0), budget=3, clock=clock)
    draw = stream.select(stream.register(_record(), _ports(6)))
    clock.now = 60.5
    out = stream.deliver(draw, [SESSION])
    assert out is not None and not out.accepted and out.verdict.codes == ("late",) and out.profile is None
    with pytest.raises(S.Refused):
        stream.outcome(draw, _lean(), {})
    assert stream.symbols == ("late",) and stream.records[0]["audit"] is None

    reg = stream.register(_record(row=1, previous=draw.registration.digest), _ports(6))
    draw = stream.select(reg)
    clock.now = 100.0
    with pytest.raises(S.Pending):
        stream.outcome(draw, _lean(), {})
    clock.now = 120.6
    assert stream.outcome(draw, _lean(), {}).verdict.codes == ("late",)

    reg = stream.register(_record(row=2, previous=reg.digest), _ports(6))
    draw = stream.select(reg)
    clock.now += 60.0
    assert stream.deliver(draw, [SESSION]) is None and stream.outcome(draw, _lean(), {}).accepted
    assert stream.symbols == ("late", "late", "accept") and stream.failures == 2


def test_the_stream_stops_at_its_budgeth_failure():
    stream = _stream(budget=2)
    assert not _round(stream, accepted=False).accepted and not stream.stopped
    assert _round(stream).accepted and not stream.stopped
    assert not _round(stream, accepted=False).accepted
    assert stream.stopped and stream.failures == 2 and stream.symbols == ("reject", "accept", "reject")
    with pytest.raises(S.Refused) as e:
        _round(stream)
    assert e.value.verdict.codes == ("stopped",) and len(stream.records) == 3


def test_a_sampler_draws_instead_and_is_held_to_the_law():
    stream = _stream(sampler=lambda law, n: [n - 1, 0])
    assert stream.select(stream.register(_record(), _ports(6))).units == [0, 5]
    for bad in ([0, 6], [1], [2, 2]):
        stream = _stream(sampler=lambda law, n, bad=bad: bad)
        reg = stream.register(_record(), _ports(6))
        with pytest.raises(S.Refused):
            stream.select(reg)


def test_public_outputs_come_from_the_verifier_of_record():
    spec = _spec(public=(*STEP0, S.Public("credited", "output", "the credited work")), public_outputs=("credited",))
    stream = _stream(spec)
    draw = stream.select(stream.register(_record(), _ports(6)))
    stream.deliver(draw, [SESSION])
    for public in (None, {}, {"credited": 1, "other": 2}):
        with pytest.raises(S.Refused) as e:
            stream.outcome(draw, _lean(public=public), {})
        assert e.value.verdict.codes == ("public",)
    out = stream.outcome(draw, _lean(public={"credited": 12}), {})
    assert out.public == {"credited": 12} and stream.records[0]["public"] == {"credited": 12}


def test_a_call_names_its_own_streams_registration_and_draw():
    stream, other = _stream(), _stream()
    reg = other.register(_record(row=5), _ports(6))
    with pytest.raises(S.Refused) as e:
        stream.select(reg)
    assert e.value.verdict.codes == ("unknown",)
    mine = stream.register(_record(), _ports(6))
    draw = other.select(reg)
    with pytest.raises(S.Refused):
        stream.deliver(S.Draw(mine, draw.law, draw.draw), [SESSION])


def _refused(call) -> tuple[str, ...]:
    with pytest.raises(S.Refused) as e:
        call()
    return e.value.verdict.codes


def test_two_stage_round_trip_registers_the_interiors_between_the_draw_and_the_proof():
    stream = _stream(_two_spec())
    rec, ports = _half("x", 6)
    reg = stream.register(rec, ports)
    assert reg.order == 0 and reg.receipt["received"] == "0" and reg.digest == R.digest(rec)
    draw = stream.select(reg)
    assert len(draw.units) == 2 and draw.proved == draw.units
    inner, iports = _interiors(draw)
    ireg = stream.register_interiors(draw, inner, iports)
    assert ireg.order == 0 and ireg.receipt["received"] == "1" and ireg.digest == R.digest(inner)
    assert inner["population"] == 2 and inner["previous"] == reg.digest
    assert stream.deliver(draw, [SESSION]) is None
    out = stream.outcome(draw, _lean(), {"statement": "stub"}, session_roots=dict(ROOTS))
    assert out.accepted and out.verdict.codes == () and out.profile is not None and out.profile.bound() == 4
    assert out.exhaustive is None and stream.symbols == ("accept",)
    body = stream.records[0]
    assert S._sha512(body) == out.record and body["registration"] == reg.digest and body["audit"]["accepted"]
    assert body["interior"]["registration"] == ireg.digest and body["interior"]["receipt"] == S._sha512(dict(ireg.receipt))
    assert body["interior"]["audit"]["accepted"] and body["interior"]["audit"]["proved_units"] == [0, 1]
    # The log runs boundary, interiors, next boundary: the next registration chains to the interiors.
    assert _refused(lambda: stream.register(_half("x", 6, row=1, previous=reg.digest)[0], ports)) == ("R6-previous",)
    nxt = stream.register(_half("x", 6, row=1, previous=ireg.digest)[0], ports)
    assert nxt.order == 1 and nxt.receipt["received"] == "2"


def test_the_interiors_are_registered_once_after_select_before_deliver_over_the_proved_rus():
    stream = _stream(_two_spec())
    rec, ports = _half("x", 6)
    reg = stream.register(rec, ports)
    forged = S.Draw(reg, {"law": "subset", "k": 2}, {"law": "subset", "k": 2, "population": 6, "units": [0, 1]})
    assert _refused(lambda: stream.register_interiors(forged, *_interiors(forged))) == ("not-drawn",)
    draw = stream.select(reg)
    assert _refused(lambda: stream.deliver(draw, [SESSION])) == ("interior",)
    for n in (1, 3, 6):
        other = _half("out", n, {"law": "subset", "k": n}, previous=reg.digest)
        assert _refused(lambda: stream.register_interiors(draw, *other)) == ("interior-population",)
    assert _refused(lambda: stream.register_interiors(draw, *_half("x", 2, previous=reg.digest))) == ("roots",)
    assert _refused(lambda: stream.register_interiors(draw, *_half("out", 2, previous=None))) == ("R6-previous",)
    inner, iports = _interiors(draw)
    assert _refused(lambda: stream.register_interiors(draw, {**inner, "window": rec["window"]}, iports)) == ("R6-window",)
    assert "R3-leaves" in _refused(lambda: stream.register_interiors(draw, inner, {"out": {**iports["out"], "leaves": 5}}))
    ireg = stream.register_interiors(draw, inner, iports)
    assert _refused(lambda: stream.register_interiors(draw, inner, iports)) == ("interior-registered",)
    stream.deliver(draw, [SESSION])
    assert _refused(lambda: stream.register_interiors(draw, inner, iports)) == ("delivered",)
    stream.outcome(draw, _lean(), {})
    assert _refused(lambda: stream.register_interiors(draw, inner, iports)) == ("final",)
    assert stream.records[0]["interior"]["registration"] == ireg.digest and len(stream._log) == 2


def test_a_two_stage_registration_holds_each_stage_to_its_own_values():
    stream = _stream(_two_spec())
    assert _refused(lambda: stream.register(*_half("out", 6))) == ("roots",)
    assert _refused(lambda: stream.register(_record(), _ports(6))) == ("roots",)
    plain = _stream()
    draw = plain.select(plain.register(_record(), _ports(6)))
    assert _refused(lambda: plain.register_interiors(draw, *_interiors(draw))) == ("rule",)
    assert stream._log == [] and plain._log == [_record()]


def test_a_two_stage_outcome_accepts_only_when_both_audits_do():
    stream = _stream(_two_spec(), budget=3)

    def staged(row: int, previous: str | None) -> tuple[S.Draw, S.Registration]:
        reg = stream.register(_half("x", 6, row=row, previous=previous)[0], {"x": _ports(6)["x"]})
        draw = stream.select(reg)
        ireg = stream.register_interiors(draw, *_interiors(draw, row=row))
        stream.deliver(draw, [SESSION])
        return draw, ireg

    draw, ireg = staged(0, None)
    out = stream.outcome(draw, _lean(), {}, session_roots={**ROOTS, "out": "ee" * 64})
    why = out.verdict.detail["reject"]
    assert not out.accepted and out.verdict.codes == ("reject",) and why["checks"] == []
    assert why["interior"] == {"accepted": False, "checks": ["roots"]} and stream.records[0]["audit"]["accepted"]
    draw, ireg = staged(1, ireg.digest)
    out = stream.outcome(draw, _lean(), {}, session_roots={**ROOTS, "x": "ee" * 64})
    why = out.verdict.detail["reject"]
    assert not out.accepted and why["checks"] == ["roots"] and why["interior"] == {"accepted": True, "checks": []}
    draw, ireg = staged(2, ireg.digest)
    assert stream.outcome(draw, _lean(), {}, session_roots=dict(ROOTS)).accepted
    assert stream.symbols == ("reject", "reject", "accept")


def test_a_two_stage_draw_of_no_rus_has_no_interiors():
    stream = _stream(_two_spec("bernoulli:1/2"), sampler=lambda law, n: [])
    reg = stream.register(*_half("x", 6, {"law": "bernoulli", "p": "1/2"}))
    draw = stream.select(reg)
    assert draw.units == [] == draw.proved
    assert _refused(lambda: stream.register_interiors(draw, *_half("out", 1, {"law": "subset", "k": 1},
                                                                    previous=reg.digest))) == ("interior-empty",)
    assert stream.deliver(draw, [SESSION]) is None
    assert stream.outcome(draw, _lean(), {}).accepted and stream.records[0]["interior"] is None


def test_a_late_two_stage_proof_is_a_reject_with_or_without_interiors():
    clock = Clock()
    stream = _stream(_two_spec(proof_deadline=60.0), budget=3, clock=clock)
    reg = stream.register(*_half("x", 6))
    draw = stream.select(reg)
    clock.now = 30.0
    ireg = stream.register_interiors(draw, *_interiors(draw))
    clock.now = 60.5
    out = stream.deliver(draw, [SESSION])
    assert out is not None and out.verdict.codes == ("late",)
    assert stream.records[0]["interior"] == {"registration": ireg.digest, "receipt": S._sha512(dict(ireg.receipt)),
                                             "audit": None}
    draw = stream.select(stream.register(*_half("x", 6, row=1, previous=ireg.digest)))
    clock.now = 121.0
    assert stream.deliver(draw, [SESSION]).verdict.codes == ("late",)
    draw = stream.select(stream.register(*_half("x", 6, row=2, previous=draw.registration.digest)))
    clock.now = 182.0
    assert stream.outcome(draw, _lean(), {}).verdict.codes == ("late",)
    assert stream.symbols == ("late",) * 3 and [r["interior"] is None for r in stream.records] == [False, True, True]


WORK = {PAIR.id: 10}
STRATUM = S.Public(f"population:{PAIR.id}", "structure", "the work law's one stratum")
#: The challenger's closure map: each unit is proved with the next.
NEXT = {u: [(u + 1) % 6] for u in range(6)}


def test_a_work_draw_carries_its_closure_and_the_proof_proves_it():
    stream = _stream(_spec(S.Law("work:2"), public=(*STEP0, STRATUM)))
    reg = stream.register(_record(law={"law": "work", "k": 2}, work_table=WORK), _ports(6), work_table=WORK)
    draw = stream.select(reg, closure=NEXT)
    assert draw.draw["closure"] == T.closure_of(NEXT, draw.units) and draw.draw["closure"]
    assert D.check_draw(draw.draw, reg.record, draw.law).ok
    assert draw.proved == sorted({*draw.units, *((u + 1) % 6 for u in draw.units)})
    stream.deliver(draw, [SESSION])
    out = stream.outcome(draw, _lean(), {}, proved_units=draw.units)
    assert not out.accepted and out.verdict.detail["reject"]["checks"] == ["proved"]
    stream = _stream(_spec(S.Law("work:2"), public=(*STEP0, STRATUM)))
    draw = stream.select(stream.register(_record(law={"law": "work", "k": 2}, work_table=WORK), _ports(6),
                                         work_table=WORK), closure=NEXT)
    stream.deliver(draw, [SESSION])
    out = stream.outcome(draw, _lean(), {})
    assert out.accepted and stream.records[0]["audit"]["proved_units"] == draw.proved


def test_a_closure_widens_a_work_draw_only_and_stays_in_range():
    stream = _stream()
    reg = stream.register(_record(), _ports(6))
    assert _refused(lambda: stream.select(reg, closure=NEXT)) == ("closure",)
    reads: list[int] = []
    stream = _stream(_spec(S.Law("work:2"), public=(*STEP0, STRATUM)), coins=lambda n: reads.append(n) or b"\x07" * n)
    reg = stream.register(_record(law={"law": "work", "k": 2}, work_table=WORK), _ports(6), work_table=WORK)
    for bad in ({u: [6] for u in range(6)}, {6: [0]}, {u: [-1] for u in range(6)}, {u: ["1"] for u in range(6)},
                [[1]] * 6):
        assert _refused(lambda: stream.select(reg, closure=bad)) == ("closure",)
    assert reads == []
    assert "closure" not in stream.select(reg, closure={}).draw and reads


def test_two_stage_interiors_cover_a_work_draws_closure():
    stream = _stream(_two_spec("work:2", public=(*STEP0, STRATUM)))
    reg = stream.register(*_half("x", 6, {"law": "work", "k": 2}, work_table=WORK), work_table=WORK)
    draw = stream.select(reg, closure=NEXT)
    assert len(draw.proved) > len(draw.units)
    n = len(draw.units)
    assert _refused(lambda: stream.register_interiors(draw, *_half("out", n, {"law": "subset", "k": n},
                                                                    previous=reg.digest))) == ("interior-population",)
    ireg = stream.register_interiors(draw, *_interiors(draw))
    assert ireg.record["population"] == len(draw.proved)
    stream.deliver(draw, [SESSION])
    out = stream.outcome(draw, _lean(), {}, session_roots=dict(ROOTS))
    assert out.accepted and stream.records[0]["interior"]["audit"]["proved_units"] == list(range(len(draw.proved)))


@pytest.mark.parametrize("text", ["subset:2", "bernoulli:1/3", "stratified:2", "work:2"])
def test_the_default_draw_is_draw_first_on_the_challengers_bytes(text):
    """The draw is Lean's `drawWith` on the coins (`two_stage.draw_first`), so `flock-verify draw` on the same bytes
    draws the same units; a work draw's closure is the same too."""
    table = WORK if text.startswith("work") else None
    closure = NEXT if table else None
    for seed in range(8):
        stream = _stream(_spec(S.Law(text), public=(*STEP0, STRATUM)), coins=random.Random(seed).randbytes)
        reg = stream.register(_record(law=D.parse_law(text), work_table=table), _ports(6), work_table=table)
        draw = stream.select(reg, closure=closure)
        assert dict(draw.draw) == T.draw_first(T.OwnRandomness(random.Random(seed).randbytes), draw.law, 6, closure)
