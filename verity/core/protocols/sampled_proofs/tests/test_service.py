"""The proof system: the Spec's types, the public-items rule, and a Stream's register, select, deliver and outcome over
the one-stage audit, with a stub verifier, injected coins and an injected clock."""

from __future__ import annotations

import random

import pytest

import verity.ml.circuits.ops.prims  # noqa: F401  (the one-stage tests' primitives)
import verity.ml.circuits.ops.scalar  # noqa: F401
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.sampled_proofs import coins as T, service as S
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


def _spec(selection=S.Law("subset:2"), public=STEP0, proof_deadline=60.0, public_outputs=(), values=VALUES,
          registration_deadline=None) -> S.Spec:
    return S.Spec(values, selection, S.Check(P.program_digest(Program(PAIR)), QUERY, public_outputs=public_outputs,
                                             proof_deadline=proof_deadline, registration_deadline=registration_deadline),
                  public, b"service-test")


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
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, RUN)}
            for p, leaves in (("x", n), ("out", 2 * n))}


def _record(n: int = 6, law=None, row: int = 0, previous: str | None = None, work_table=None, **change) -> dict:
    ports = _ports(n)
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **ports[p]} for i, p in enumerate(ports)]
    rec = R.record(P.population_program(PAIR, n), QUERY, law or {"law": "subset", "k": 2}, S.SCHEME, roots,
                   {"kind": "served", "run": RUN, "row": row}, previous=previous, work_table=work_table)
    return {**rec, **change}


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


def test_a_provers_stream_binds_each_registration_to_itself_and_the_prover():
    """A stream given the identity its challenger registered the prover under names it in its digest and binds every
    registration to the stream and the prover (`Stream.binding`, `registration.check`'s): a record whose window names
    both runs round to an accepted outcome, by both verifiers, while the same record in another prover's
    stream, in the same window, is refused (`R6-stream`, `R6-prover`), as is one whose window names neither. A prover
    identity that is not lowercase hex bytes makes no stream."""
    def stream(prover):
        return S.Stream(_spec(), PAIR, budget=2, delta=DELTA, challenger="challenger-test", coins=lambda n: b"\x07" * n,
                        clock=Clock(), prover=prover)

    a, b, plain = stream("a0" * 32), stream("b0" * 32), _stream()
    assert len({a.digest, b.digest, plain.digest}) == 3 and plain.binding is None
    assert a.binding == {"stream": a.digest, "prover": "a0" * 32}
    assert a.digest == S._sha512({"format": S.STREAM, "spec": _spec().digest(), "budget": 2, "delta": DELTA,
                                  "challenger": "challenger-test", "prover": "a0" * 32})
    rec = _record()
    mine = {**rec, "window": {**rec["window"], **a.binding}}
    assert _refused(lambda: b.register(mine, _ports(6))) == ("R6-stream", "R6-prover")
    assert _refused(lambda: a.register(rec, _ports(6))) == ("R6-stream", "R6-prover")
    draw = a.select(a.register(mine, _ports(6)))
    assert a.deliver(draw, [SESSION]) is None
    out = a.outcome(draw, _lean(), {"statement": "stub"})
    assert out.accepted and out.verdict.ok and a.records[0]["stream"] == a.digest
    for bad in ("A0" * 32, "a0a", ""):
        with pytest.raises(ValueError):
            stream(bad)


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


def test_a_tree_over_roots_names_its_hiding_leaves_and_the_digest_binds_them():
    """A row-table window's `row-table` port is a tree over roots (`ROOTS`), hiding through its trees' leaves: its Value
    names their schema (`over`), a hiding one, and the Spec's digest binds it. A `ROOTS` value without `over`, an `over`
    that isn't hiding, and an `over` on any other value are refused."""
    table = S.Value("row-table", S.ROOTS, "serve", over="hm96-sha512/row-seg/v1")
    assert table.to_json()["over"] == "hm96-sha512/row-seg/v1" and "over" not in VALUES[0].to_json()
    digests = {_spec(values=(*VALUES, S.Value("row-table", S.ROOTS, "serve", over=o))).digest()
               for o in ("hm96-sha512/row-seg/v1", "hm96-sha512/row/v2")}
    assert len(digests | {_spec().digest()}) == 3
    for over in (None, "sha512/row/v2", "frame-v3-sha512/root/v1", ""):
        with pytest.raises(ValueError, match="over"):
            S.Value("row-table", S.ROOTS, "serve", over=over)
    with pytest.raises(ValueError, match="only a tree over roots"):
        S.Value("x", SCHEMA, "serve", over="hm96-sha512/row-seg/v1")
    with pytest.raises(ValueError, match="only a tree over roots"):
        S.Value("W", None, "epoch", registered=True, over="hm96-sha512/row-seg/v1")


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


def test_a_stream_refuses_what_isnt_built_and_another_unit():
    with pytest.raises(ValueError, match="Sequential"):
        _stream(_spec(S.Sequential("uniform:4", "population")))
    with pytest.raises(ValueError, match="unit"):
        S.Stream(_spec(), OTHER, budget=1, delta=DELTA, challenger="c")
    with pytest.raises(ValueError, match="proof deadline"):
        _stream(_spec(proof_deadline=None))
    for bad in ({"budget": 0}, {"budget": 1.0}, {"delta": 1}, {"delta": 1.5}):
        with pytest.raises(ValueError):
            S.Stream(_spec(), PAIR, **{"budget": 1, "delta": DELTA, "challenger": "c", **bad})


def test_a_two_stage_rule_is_refused_as_retired():
    """Two-stage sampling is retired: a Spec whose selection rule names it fails closed, saying so."""
    for rule in ({"rule": "two-stage", "replay": "subset:4", "check": "all"}, "two-stage:subset:4"):
        with pytest.raises(ValueError, match="two-stage sampling is retired") as e:
            _spec(rule)
        assert str(e.value) == S.RETIRED_TWO_STAGE
    assert not hasattr(S, "TwoStage")


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


def test_a_law_whose_count_varies_is_refused_at_register():
    """A Bernoulli sampling design at 0 < p < 1 draws a count that varies, so one registration's sessions would set up tables of
    different shapes: `register` refuses it (`shapes`) after R1-R7, and logs nothing. At p = 0 or 1 it registers."""
    for p in ("1/2", "2/3", "2/4"):
        stream = _stream(_spec(S.Law(f"bernoulli:{p}")))
        assert _refused(lambda: stream.register(_record(law={"law": "bernoulli", "p": p}), _ports(6))) == ("shapes",)
        assert "R1-law" in _refused(lambda: stream.register(_record(), _ports(6)))
        assert stream.log == () and stream.records == ()
    for p in ("0/1", "1/1", "3/3"):
        stream = _stream(_spec(S.Law(f"bernoulli:{p}")))
        assert stream.register(_record(law={"law": "bernoulli", "p": p}), _ports(6)).order == 0


def test_a_rate_law_keeps_every_units_inclusion_at_its_rate_however_the_prover_pads_n():
    """`rate:1/3:2` draws `subset:K` at K = ⌈N/3⌉ rounded up to a multiple of its 2 tables over each registration's N, so
    a registration padded tenfold (N = 60 for 6) draws K = 20 and every proof block is still drawn with probability K/N ≥ 1/3,
    where `subset:2` falls to 2/60. The draw, the Lean verifier's draw of record and the outcome's run (`verifier.run`,
    which an accept needs) are a subset's of K."""
    for n, k in ((6, 2), (60, 20)):
        stream = _stream(_spec(S.Law("rate:1/3:2")))
        assert S.selection_law(stream.spec.selection, n) == {"law": "subset", "k": k} and 3 * k >= n and k % 2 == 0
        draw = stream.select(stream.register(_record(n, law={"law": "subset", "k": k}), _ports(n)))
        assert (draw.draw["law"], draw.draw["k"], draw.draw["population"], len(draw.units)) == ("subset", k, n, k)
        assert stream.deliver(draw, [SESSION]) is None
        assert stream.outcome(draw, _lean(), {"statement": "stub"}).accepted
    fixed = _stream()
    assert len(fixed.select(fixed.register(_record(60), _ports(60))).units) == 2


def test_a_rate_count_its_tables_dont_divide_rounds_up_and_one_past_n_is_refused():
    """A rate's ⌈q·N⌉ rounds up to the next multiple of its tables: `rate:1/3:4` over 20 proof blocks draws 8 (not 7) and
    `rate:1/5:4` over 12 draws 4 (not 3), and a record that names the unrounded count fails R1. A count past N (`rate:9/10:4`
    over 6 proof blocks, 8) is refused at register (`rate`) before R1-R7 and logs nothing. A rate is 0 < NUM <= DEN over T >= 1
    tables, in ASCII digits."""
    for text, n, k in (("rate:1/3:4", 20, 8), ("rate:1/5:4", 12, 4)):
        stream = _stream(_spec(S.Law(text)))
        assert "R1-law" in _refused(lambda: stream.register(_record(n, law={"law": "subset", "k": k - 1}), _ports(n)))
        assert len(stream.select(stream.register(_record(n, law={"law": "subset", "k": k}), _ports(n))).units) == k
    stream = _stream(_spec(S.Law("rate:9/10:4")))
    with pytest.raises(S.Refused) as e:
        stream.register(_record(6, law={"law": "subset", "k": 6}), _ports(6))
    assert e.value.verdict.codes == ("rate",) and e.value.verdict.detail["rate"]["k"] == 8
    assert stream.log == () and stream.records == ()
    for bad in ("rate:0/3", "rate:4/3", "rate:1/0", "rate:1/2:0", "rate:1/2:", "rate:1/2:3:4", "rate:\u0661/2", "Rate:1/2"):
        with pytest.raises(ValueError, match="rate:NUM/DEN"):
            S.Law(bad)


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


def test_a_proof_that_never_arrives_ends_late():
    """A drawn proof block whose proof is never delivered is pending until the proof deadline and a reject (`late`) after it:
    no draw stays pending forever, and the stream receives the next registration only once it ends."""
    clock = Clock()
    stream = _stream(clock=clock)
    draw = stream.select(stream.register(_record(), _ports(6)))
    for now in (0.0, 30.0, 60.0):
        clock.now = now
        with pytest.raises(S.Pending, match="no proof delivered"):
            stream.outcome(draw, _lean(), {})
        assert _refused(lambda: stream.register(_record(row=1, previous=draw.registration.digest), _ports(6))) == ("open",)
    clock.now = 60.5
    out = stream.outcome(draw, _lean(), {})
    assert not out.accepted and out.verdict.codes == ("late",) and out.profile is None and stream.symbols == ("late",)
    assert stream.records[0]["sessions"] == [] and stream.records[0]["audit"] is None and stream.failures == 1
    assert _refused(lambda: stream.deliver(draw, [SESSION])) == ("final",)
    assert stream.register(_record(row=1, previous=draw.registration.digest), _ports(6)).order == 1


OUTCOME_KEYS = {"format", "stream", "order", "registration", "receipt", "draw", "sessions", "symbol", "public", "audit"}


def test_a_late_registration_is_refused_and_ends_the_epoch():
    clock = Clock()
    stream = _stream(_spec(registration_deadline=30.0), budget=3, clock=clock)
    clock.now = 5.0
    salt = stream.salt()
    clock.now = 35.5
    readings = {"salted_at": 5.0, "received_at": 35.5, "registration_deadline": 30.0}
    with pytest.raises(S.Refused) as e:
        stream.register({"not": "a record"}, _ports(6))
    assert e.value.verdict.codes == ("late-registration",) and e.value.verdict.detail == {"late-registration": readings}
    assert stream.ended and stream.log == () and stream._entries == [] and stream.symbols == () and not stream.stopped
    assert stream.epoch_log == ({"event": "salt", "salt": salt.hex(), "salted_at": 5.0},
                                {"event": "late-registration", "salted_at": 5.0, "received_at": 35.5})
    for call in (lambda: stream.register(_record(), _ports(6)), stream.ready):
        with pytest.raises(S.Refused) as e:
            call()
        assert e.value.verdict.codes == ("epoch-ended",) and e.value.verdict.detail == {"epoch-ended": readings}
    assert _refused(stream.salt) == ("salted",)
    assert len(stream.epoch_log) == 2 and stream.log == ()


def test_an_on_time_registration_carries_both_readings():
    clock = Clock()
    stream = _stream(_spec(registration_deadline=30.0), clock=clock)
    clock.now = 5.0
    salt = stream.salt()
    clock.now = 20.0
    assert "R3-leaves" in _refused(lambda: stream.register(_record(), {**_ports(6), "x": {**_ports(6)["x"], "leaves": 7}}))
    assert not stream.ended and len(stream.epoch_log) == 1
    clock.now = 21.0
    reg = stream.register(_record(), _ports(6))
    assert (reg.salted_at, reg.received_at) == (5.0, 21.0) and reg.order == 0
    assert reg.receipt == R.receipt(_record(), "challenger-test", received="0")
    assert stream.epoch_log == ({"event": "salt", "salt": salt.hex(), "salted_at": 5.0},
                                {"event": "registration", "order": 0, "registration": reg.digest, "salted_at": 5.0,
                                 "received_at": 21.0})
    draw = stream.select(reg)
    stream.deliver(draw, [SESSION])
    assert stream.outcome(draw, _lean(), {}).accepted and set(stream.records[0]) == OUTCOME_KEYS
    clock.now = 35.0
    nxt = stream.register(_record(row=1, previous=reg.digest), _ports(6))
    assert (nxt.salted_at, nxt.received_at) == (5.0, 35.0) and not stream.ended
    assert stream.epoch_log[-1] == {"event": "registration", "order": 1, "registration": nxt.digest, "salted_at": 5.0,
                                    "received_at": 35.0}


def test_a_registration_before_the_salt_is_late():
    stream = _stream(_spec(registration_deadline=30.0))
    readings = {"salted_at": None, "received_at": 0.0, "registration_deadline": 30.0}
    with pytest.raises(S.Refused) as e:
        stream.register(_record(), _ports(6))
    assert e.value.verdict.codes == ("late-registration",) and e.value.verdict.detail == {"late-registration": readings}
    assert stream.ended and stream.log == ()
    assert stream.epoch_log == ({"event": "late-registration", "salted_at": None, "received_at": 0.0},)
    with pytest.raises(S.Refused) as e:
        stream.salt()
    assert e.value.verdict.codes == ("epoch-ended",) and e.value.verdict.detail == {"epoch-ended": readings}
    assert _refused(lambda: stream.register(_record(), _ports(6))) == ("epoch-ended",) and len(stream.epoch_log) == 1


def test_the_salt_is_issued_once_from_the_challengers_coins():
    """The salt is the challenger's next `n` bytes, read before the draw's, and a second salt reads nothing."""
    reads: list[int] = []
    source = random.Random(3).randbytes
    clock = Clock()
    clock.now = 3.0
    stream = _stream(coins=lambda n: reads.append(n) or source(n), clock=clock)
    salt = stream.salt()
    expected = random.Random(3)
    assert salt == expected.randbytes(64) and reads == [64]
    assert stream.epoch_log == ({"event": "salt", "salt": salt.hex(), "salted_at": 3.0},)
    clock.now = 4.0
    with pytest.raises(S.Refused) as e:
        stream.salt()
    assert e.value.verdict.codes == ("salted",) and e.value.verdict.detail == {"salted": {"salted_at": 3.0}}
    assert reads == [64] and len(stream.epoch_log) == 1
    draw = stream.select(stream.register(_record(), _ports(6)))
    assert dict(draw.draw) == T.draw_first(T.OwnRandomness(expected.randbytes), draw.law, 6)
    assert _stream().salt(32) == b"\x07" * 32
    for bad in (0, -1, 1.5, True, "64"):
        with pytest.raises(ValueError, match="positive"):
            _stream().salt(bad)


def test_no_registration_deadline_changes_nothing():
    """Unset, the registration deadline reads no clock and logs nothing, salted or not; the Check states it either way."""
    reads: list[float] = []
    stream = _stream(clock=lambda: reads.append(0.0) or 0.0)
    reg = stream.register(_record(), _ports(6))
    assert (reg.salted_at, reg.received_at) == (None, None) and reads == [] and stream.epoch_log == ()
    draw = stream.select(reg)
    stream.deliver(draw, [SESSION])
    assert stream.outcome(draw, _lean(), {}).accepted
    clock = Clock()
    salted = _stream(clock=clock)
    salted.salt()
    clock.now = 1e6
    reg = salted.register(_record(), _ports(6))
    assert (reg.salted_at, reg.received_at) == (None, None) and not salted.ended and len(salted.epoch_log) == 1
    assert _spec().check.to_json()["registration_deadline"] is None
    assert _spec(registration_deadline=30.0).check.to_json()["registration_deadline"] == 30.0
    assert _spec(registration_deadline=30.0).digest() != _spec().digest() != _spec(proof_deadline=30.0).digest()
    for bad in (0, -1.0, True, "30", float("nan")):
        with pytest.raises(ValueError, match="registration deadline is a positive number"):
            _spec(registration_deadline=bad)


def test_a_salt_released_elsewhere_counts_the_deadline_from_its_release():
    """A level's stream created after the epoch's salt was released measures from that release, and reads no coins."""
    reads: list[int] = []
    clock = Clock()
    clock.now = 50.0
    stream = _stream(_spec(registration_deadline=30.0), clock=clock, coins=lambda n: reads.append(n) or b"\x07" * n)
    salt = b"\x5a" * 64
    assert stream.salt_released(salt, 20.0) is None and reads == []
    assert stream.epoch_log == ({"event": "salt", "salt": salt.hex(), "salted_at": 20.0, "external": True},)
    reg = stream.register(_record(), _ports(6))
    assert (reg.salted_at, reg.received_at) == (20.0, 50.0)
    assert stream.epoch_log[-1] == {"event": "registration", "order": 0, "registration": reg.digest, "salted_at": 20.0,
                                    "received_at": 50.0}
    draw = stream.select(reg)
    stream.deliver(draw, [SESSION])
    assert stream.outcome(draw, _lean(), {}).accepted
    clock.now = 50.5
    readings = {"salted_at": 20.0, "received_at": 50.5, "registration_deadline": 30.0}
    with pytest.raises(S.Refused) as e:
        stream.register(_record(row=1, previous=reg.digest), _ports(6))
    assert e.value.verdict.codes == ("late-registration",) and e.value.verdict.detail == {"late-registration": readings}
    assert stream.ended and stream.epoch_log[-1] == {"event": "late-registration", "salted_at": 20.0, "received_at": 50.5}
    assert _refused(lambda: stream.register(_record(row=1, previous=reg.digest), _ports(6))) == ("epoch-ended",)


def test_a_released_salt_is_never_in_the_future_and_is_the_epochs_one_salt():
    clock = Clock()
    clock.now = 10.0
    stream = _stream(_spec(registration_deadline=30.0), clock=clock)
    with pytest.raises(S.Refused) as e:
        stream.salt_released(b"\x01", 10.5)
    assert e.value.verdict.codes == ("released-later",)
    assert e.value.verdict.detail == {"released-later": {"released_at": 10.5, "now": 10.0}}
    assert stream.epoch_log == () and not stream.ended
    stream.salt_released(b"\x01", 10.0)
    for call in (stream.salt, lambda: stream.salt_released(b"\x02", 10.0)):
        with pytest.raises(S.Refused) as e:
            call()
        assert e.value.verdict.codes == ("salted",) and e.value.verdict.detail == {"salted": {"salted_at": 10.0}}
    assert len(stream.epoch_log) == 1
    issued = _stream(clock=clock)
    issued.salt()
    assert _refused(lambda: issued.salt_released(b"\x02", 5.0)) == ("salted",) and len(issued.epoch_log) == 1
    ended = _stream(_spec(registration_deadline=30.0), clock=clock)
    assert _refused(lambda: ended.register(_record(), _ports(6))) == ("late-registration",)
    assert _refused(lambda: ended.salt_released(b"\x01", 5.0)) == ("epoch-ended",) and len(ended.epoch_log) == 1
    for salt in (b"", "01", bytearray(b"\x01"), None):
        with pytest.raises(ValueError, match="non-empty bytes"):
            _stream().salt_released(salt, 0.0)
    for at in (float("nan"), float("inf"), float("-inf"), True, "0", None):
        with pytest.raises(ValueError, match="finite"):
            _stream().salt_released(b"\x01", at)


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


def test_each_outcome_is_the_lean_runs_from_the_receipt_through_the_verdict(monkeypatch):
    """A one-stage outcome on a draw from the coins is the verifier's run (`verifier.run`, the Lean's `Run.run`)
    from the registration, the bytes the draw read, the sessions and the verdicts: one run per outcome, agreeing on an
    accepted and a rejected round. A run the Python doesn't make alike (here, a draw said to read one byte more) rejects
    the round, its audit record failing `run`; a sampler's draw isn't run."""
    run, ran = S.V.run, []

    def spy(python, *a, **kw):
        ran.append(v := run(python, *a, **kw))
        return v

    monkeypatch.setattr(S.V, "run", spy)
    stream = _stream(budget=10)
    assert [_round(stream).accepted, _round(stream, accepted=False).accepted] == [True, False]
    assert len(ran) == 2 and all(v.ok and v.verifiers["of_record"] == "lean" for v in ran)
    monkeypatch.setattr(S.V, "run", lambda python, *a, **kw: spy({**python, "read": python["read"] + 1}, *a, **kw))
    out = _round(stream)
    assert not out.accepted and out.verdict.detail["reject"]["checks"] == ["run"] and ran[-1].codes == ("disagree",)
    body = stream.records[-1]["audit"]
    assert body["accepted"] is False and body["checks"]["run"]["codes"] == ["disagree"]
    assert set(body["checks"]["run"]["detail"]["disagree"]["fields"]) == {"read"}
    sampled = _stream(sampler=lambda law, n: [0, 1])
    assert _round(sampled).accepted and len(ran) == 3


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


WORK = {PAIR.id: 10}
STRATUM = S.Public(f"population:{PAIR.id}", "structure", "the work law's one stratum")
#: The challenger's closure map: each proof block is proved with the next.
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


@pytest.mark.parametrize("text", ["subset:2", "bernoulli:3/3", "stratified:2", "work:2"])
def test_the_default_draw_is_draw_first_on_the_challengers_bytes(text):
    """The draw is Lean's `drawWith` on the coins (`coins.draw_first`), so `flock-verify draw` on the same bytes
    draws the same proof blocks; a work draw's closure is the same too."""
    table = WORK if text.startswith("work") else None
    closure = NEXT if table else None
    for seed in range(8):
        stream = _stream(_spec(S.Law(text), public=(*STEP0, STRATUM)), coins=random.Random(seed).randbytes)
        reg = stream.register(_record(law=D.parse_law(text), work_table=table), _ports(6), work_table=table)
        draw = stream.select(reg, closure=closure)
        assert dict(draw.draw) == T.draw_first(T.OwnRandomness(random.Random(seed).randbytes), draw.law, 6, closure)
