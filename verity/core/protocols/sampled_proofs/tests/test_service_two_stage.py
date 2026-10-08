"""The two-stage law on the proof service's `Stream`: the four steps in order, the draw as the Lean verifier's from the
challenger's own bytes, the interiors held to the rule, one outcome record binding both registrations, both receipts and
the draw, and no retries. PoUW's example follows the architecture note: a uniform subset:K' over N equal tiles."""

from __future__ import annotations

import hashlib
import random

import pytest

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.profile import Subset
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R
from verity.core.protocols.sampled_proofs.two_stage import REPLAY, OwnRandomness, Stage, TwoStageLaw, draw_first

XOR8 = PrimitiveDefinition("TwoStageTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
TILE = bind(CompositeDefinition("TwoStageTestTile", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
QUERY = P.template_instance_query(TILE)
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-two-stage-test"
SESSION = "5e" * 64
DELTA = 2.0 ** -40
N, K = 8, 3
OUTCOME = "verity/proof-service/outcome/v0"
#: PoUW's replay unit is a tile: its boundary is A's rows, the tile digests and the per-call records, committed while
#: serving; its interiors are A', the ok rows, `rowk`'s carries and the cap rows, regenerated after the draw.
BOUNDARY = ("a_rows", "tile_digests", "call_records")
INTERIOR_VALUES = ("a_prime", "ok_rows", "rowk_carries", "cap_rows")
PUBLIC = (S.Public("population", "structure", "N, the draw's range; the interiors' count"),
          S.Public("window", "structure", "provenance: served, on test prompts (step 0)"),
          *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k} (step 0)") for v in BOUNDARY + INTERIOR_VALUES
            for k in ("leaves", "domain")))
VALUES = (*(S.Value(v, SCHEMA, "serve") for v in BOUNDARY), *(S.Value(v, SCHEMA, "interior") for v in INTERIOR_VALUES))


def _spec(selection=S.TwoStage(S.Law(f"subset:{K}"), "all"), values=VALUES, public=PUBLIC, proof_deadline=None) -> S.Spec:
    return S.Spec(values, selection, S.Check(P.program_sha512(Program(TILE)), QUERY, proof_deadline=proof_deadline),
                  public, b"two-stage-test")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _seeded(seed: int) -> OwnRandomness:
    return OwnRandomness(random.Random(seed).randbytes)


def _stream(spec=None, budget=2, seed=1, clock=None) -> S.Stream:
    return S.Stream(spec or _spec(), TILE, budget=budget, delta=DELTA, challenger="challenger-test",
                    coins=random.Random(seed).randbytes, clock=clock or Clock())


def _previous(stream: S.Stream) -> str | None:
    """The digest of the stream's last logged record: an audit's interiors when it has them, else its boundary."""
    return R.digest(stream._log[-1]) if stream._log else None


def _ports(n: int, names) -> dict:
    part = P.partition(P.population_program(TILE, n), QUERY)
    return {v: {"leaves": (i + 1) * n,
                "domain": R.served_domain(part["program"], P.digest(part), v, SCHEMA, (i + 1) * n, RUN)}
            for i, v in enumerate(names)}


def _record(n: int, names, law, window, previous=None, work_table=None, **change) -> dict:
    ports = _ports(n, names)
    roots = [{"port": v, "root": f"{i + 1:02x}" * 64, **ports[v]} for i, v in enumerate(names)]
    rec = R.record(P.population_program(TILE, n), QUERY, law, S.SCHEME, roots, window, previous=previous,
                   work_table=work_table)
    return {**rec, **change}


def _boundary(n=N, law=None, row=0, previous=None, work_table=None, **change) -> dict:
    return _record(n, BOUNDARY, law or {"law": "subset", "k": K}, {"kind": "served", "run": RUN, "row": row}, previous,
                   work_table, **change)


def _interior(n: int, previous: str | None, row=0, **change) -> dict:
    return _record(n, INTERIOR_VALUES, {"law": "subset", "k": n}, {"kind": "served", "run": RUN, "row": row,
                                                                   "stage": "interior"}, previous, **change)


def _roots(names, wrong: str | None = None) -> dict:
    return {v: ("ee" * 64 if v == wrong else f"{i + 1:02x}" * 64) for i, v in enumerate(names)}


def _lean(accepted=True, sessions=(SESSION,)) -> dict:
    return {"lean": {"of_record": True, "accepted": accepted, "digest": "dd" * 64, "sessions": list(sessions)}}


def _sha512(obj) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


def _audit(stream: S.Stream, accepted=True, row=0, session_roots=None, closure=None, **boundary) -> S.Outcome:
    reg = stream.register(_boundary(row=row, previous=_previous(stream), **boundary), _ports(N, BOUNDARY),
                          **({"work_table": boundary["work_table"]} if "work_table" in boundary else {}))
    draw = stream.select(reg, closure=closure)
    stream.register_interiors(draw, _interior(len(draw.proved), reg.digest, row), _ports(len(draw.proved), INTERIOR_VALUES))
    assert stream.deliver(draw, [SESSION]) is None
    return stream.outcome(draw, _lean(accepted), {"statement": "stub"}, session_roots=session_roots)


def test_the_pouw_example_runs_the_four_steps_and_one_record_binds_them():
    stream = _stream()
    reg = stream.register(_boundary(), _ports(N, BOUNDARY))
    assert reg.order == 0 and reg.receipt["received"] == "0" and reg.record["population"] == N
    draw = stream.select(reg)
    assert draw.units == draw_first(_seeded(1), {"law": "subset", "k": K}, N)["units"]
    assert len(draw.units) == K and draw.proved == draw.units and "closure" not in draw.draw
    inner = stream.register_interiors(draw, _interior(K, reg.digest), _ports(K, INTERIOR_VALUES))
    assert inner.order == 0 and inner.receipt["received"] == "1"
    assert inner.record["population"] == K and inner.record["law"] == {"law": "subset", "k": K}
    assert stream.deliver(draw, [SESSION]) is None
    out = stream.outcome(draw, _lean(), {"statement": "stub"},
                         session_roots={**_roots(BOUNDARY), **_roots(INTERIOR_VALUES)})
    assert out.accepted and out.verdict.codes == () and out.exhaustive is None and out.terms == S.Terms(DELTA)
    assert stream.symbols == ("accept",) and stream.failures == 0

    law = TwoStageLaw({"tile": Stage({"law": "subset", "k": K})}).profile("tile", n_r=N, delta=DELTA)
    assert out.profile.draws["unit"] == law.draws[REPLAY] == Subset(K) and out.profile.sizes == law.sizes == (N,)
    assert out.profile.bound() == law.bound()

    body = stream.records[0]
    assert _sha512(body) == out.record and body["format"] == OUTCOME and body["stream"] == stream.digest
    assert body["registration"] == reg.digest and body["receipt"] == _sha512(dict(reg.receipt))
    assert body["draw"] == _sha512(dict(draw.draw)) and body["sessions"] == [SESSION] and body["symbol"] == "accept"
    assert body["interior"]["registration"] == inner.digest and body["interior"]["receipt"] == _sha512(dict(inner.receipt))
    assert body["audit"]["accepted"] and body["interior"]["audit"]["accepted"]


def test_each_step_waits_for_the_one_before():
    stream = _stream()
    reg = stream.register(_boundary(), _ports(N, BOUNDARY))
    with pytest.raises(S.Refused) as e:
        stream.register(_boundary(row=1, previous=reg.digest), _ports(N, BOUNDARY))
    assert e.value.verdict.codes == ("open",)
    draw = stream.select(reg)
    with pytest.raises(S.Pending):
        stream.outcome(draw, _lean(), {})
    with pytest.raises(S.Refused) as e:
        stream.deliver(draw, [SESSION])
    assert e.value.verdict.codes == ("interior",)
    inner = stream.register_interiors(draw, _interior(K, reg.digest), _ports(K, INTERIOR_VALUES))
    with pytest.raises(S.Pending):
        stream.outcome(draw, _lean(), {})
    stream.deliver(draw, [SESSION])
    with pytest.raises(S.Pending):
        stream.outcome(draw, _lean(None), {})
    assert stream.outcome(draw, _lean(), {}).accepted
    with pytest.raises(S.Refused) as e:
        stream.register(_boundary(row=1, previous=reg.digest), _ports(N, BOUNDARY))
    assert "R6-previous" in e.value.verdict.codes
    assert stream.register(_boundary(row=1, previous=inner.digest), _ports(N, BOUNDARY)).order == 1


def test_the_interiors_are_the_proved_replay_units_held_to_the_rule():
    stream = _stream()
    reg = stream.register(_boundary(), _ports(N, BOUNDARY))
    draw = stream.select(reg)
    with pytest.raises(S.Refused) as e:
        stream.register_interiors(draw, _interior(K + 1, reg.digest), _ports(K + 1, INTERIOR_VALUES))
    assert e.value.verdict.codes == ("interior-population",)
    shown = _record(K, ("a_rows", *INTERIOR_VALUES), {"law": "subset", "k": K},
                    {"kind": "served", "run": RUN, "row": 0, "stage": "interior"}, reg.digest)
    with pytest.raises(S.Refused) as e:
        stream.register_interiors(draw, shown, _ports(K, ("a_rows", *INTERIOR_VALUES)))
    assert e.value.verdict.detail["roots"] == ["a_rows"]
    with pytest.raises(S.Refused) as e:
        stream.register_interiors(draw, {**_interior(K, reg.digest), "tiles": draw.units}, _ports(K, INTERIOR_VALUES))
    assert "keys" in e.value.verdict.codes
    with pytest.raises(S.Refused) as e:
        stream.register_interiors(draw, _interior(K, reg.digest),
                                  {**_ports(K, INTERIOR_VALUES), "ok_rows": {"leaves": 1, "domain": "00"}})
    assert "R3-leaves" in e.value.verdict.codes
    inner = stream.register_interiors(draw, _interior(K, reg.digest), _ports(K, INTERIOR_VALUES))
    with pytest.raises(S.Refused) as e:
        stream.register_interiors(draw, _interior(K, reg.digest), _ports(K, INTERIOR_VALUES))
    assert e.value.verdict.codes == ("interior-registered",)
    stream.deliver(draw, [SESSION])
    assert stream.outcome(draw, _lean(), {}).accepted and stream.records[0]["interior"]["registration"] == inner.digest


def test_a_work_first_stage_draws_with_the_challengers_closure():
    spec = _spec(S.TwoStage(S.Law("work:2"), "all"),
                 public=(*PUBLIC, S.Public(f"population:{TILE.id}", "structure", "the tiles' one stratum")))
    table = {TILE.id: 3}
    law = D.derive(P.population_program(TILE, N), QUERY, {"law": "work", "k": 2}, table)
    closure = {u: [(u + 1) % N] for u in range(N)}
    stream = _stream(spec, seed=5)
    reg = stream.register(_boundary(law={"law": "work", "k": 2}, work_table=table), _ports(N, BOUNDARY), work_table=table)
    draw = stream.select(reg, closure=closure)
    want = draw_first(_seeded(5), law, N, closure)
    assert draw.units == want["units"] and draw.draw["closure"] == want["closure"]
    assert draw.proved == sorted(want["units"] + want["closure"]) and len(draw.proved) > len(draw.units)
    with pytest.raises(S.Refused) as e:
        stream.register_interiors(draw, _interior(len(draw.units), reg.digest), _ports(len(draw.units), INTERIOR_VALUES))
    assert e.value.verdict.codes == ("interior-population",)
    n = len(draw.proved)
    stream.register_interiors(draw, _interior(n, reg.digest), _ports(n, INTERIOR_VALUES))
    stream.deliver(draw, [SESSION])
    out = stream.outcome(draw, _lean(), {})
    assert out.accepted and stream.records[0]["draw"] == _sha512(dict(draw.draw))
    stage = TwoStageLaw({"tile": Stage(law)}).profile("tile", n_r=N, delta=DELTA)
    assert out.profile.draws["unit"] == stage.draws[REPLAY] and out.profile.bound() == stage.bound()

    subset = _stream(seed=5)
    reg = subset.register(_boundary(), _ports(N, BOUNDARY))
    with pytest.raises(S.Refused) as e:
        subset.select(reg, closure=closure)
    assert e.value.verdict.codes == ("closure",)
    assert subset.select(reg).units == draw_first(_seeded(5), {"law": "subset", "k": K}, N)["units"]


def test_a_wrong_interior_or_boundary_rejects_the_audit_once():
    both = {**_roots(BOUNDARY), **_roots(INTERIOR_VALUES)}
    for wrong in ("cap_rows", "tile_digests"):
        stream = _stream(budget=3)
        out = _audit(stream, session_roots={**_roots(BOUNDARY, wrong), **_roots(INTERIOR_VALUES, wrong)})
        assert not out.accepted and out.verdict.codes == ("reject",) and out.profile is None
        assert stream.symbols == ("reject",) and stream.failures == 1
        why = out.verdict.detail["reject"]
        assert why["interior"]["accepted"] is (wrong != "cap_rows") and bool(why["checks"]) is (wrong == "tile_digests")
        assert _audit(stream, row=1, session_roots=both).accepted and stream.symbols == ("reject", "accept")
    stream = _stream()
    assert not _audit(stream, accepted=False).accepted and stream.failures == 1


def test_the_stream_stops_at_its_budgeth_failure():
    stream = _stream(budget=2)
    assert not _audit(stream, accepted=False).accepted and not stream.stopped
    assert _audit(stream, row=1).accepted
    assert not _audit(stream, accepted=False, row=2).accepted
    assert stream.stopped and stream.symbols == ("reject", "accept", "reject") and len(stream.records) == 3
    with pytest.raises(S.Refused) as e:
        _audit(stream, row=3)
    assert e.value.verdict.codes == ("stopped",)


def test_no_retries_nothing_redraws_and_an_outcome_is_final():
    stream = _stream()
    reg = stream.register(_boundary(), _ports(N, BOUNDARY))
    draw = stream.select(reg)
    with pytest.raises(S.Refused) as e:
        stream.select(reg)
    assert e.value.verdict.codes == ("drawn",)
    stream.register_interiors(draw, _interior(K, reg.digest), _ports(K, INTERIOR_VALUES))
    stream.deliver(draw, [SESSION])
    with pytest.raises(S.Refused) as e:
        stream.deliver(draw, [SESSION])
    assert e.value.verdict.codes == ("delivered",)
    assert not stream.outcome(draw, _lean(False), {}).accepted
    for call in (lambda: stream.outcome(draw, _lean(), {}), lambda: stream.deliver(draw, [SESSION]),
                 lambda: stream.register_interiors(draw, _interior(K, reg.digest), _ports(K, INTERIOR_VALUES))):
        with pytest.raises(S.Refused) as e:
            call()
        assert e.value.verdict.codes == ("final",)
    other = _stream()
    with pytest.raises(S.Refused) as e:
        other.register_interiors(draw, _interior(K, reg.digest), _ports(K, INTERIOR_VALUES))
    assert e.value.verdict.codes == ("unknown",)


def test_a_late_proof_or_missing_interiors_are_late():
    clock = Clock()
    stream = _stream(_spec(proof_deadline=60.0), budget=3, clock=clock)
    reg = stream.register(_boundary(), _ports(N, BOUNDARY))
    draw = stream.select(reg)
    clock.now = 40.0
    stream.register_interiors(draw, _interior(K, reg.digest), _ports(K, INTERIOR_VALUES))
    clock.now = 60.5
    out = stream.deliver(draw, [SESSION])
    assert out is not None and out.verdict.codes == ("late",) and stream.symbols == ("late",)
    assert stream.records[0]["interior"]["audit"] is None and stream.records[0]["symbol"] == "late"

    draw = stream.select(stream.register(_boundary(row=1, previous=_previous(stream)), _ports(N, BOUNDARY)))
    clock.now = 100.0
    with pytest.raises(S.Pending):
        stream.outcome(draw, _lean(), {})
    clock.now = 121.0
    out = stream.outcome(draw, _lean(), {})
    assert out.verdict.codes == ("late",) and stream.records[1]["interior"] is None

    reg = stream.register(_boundary(row=2, previous=_previous(stream)), _ports(N, BOUNDARY))
    draw = stream.select(reg)
    clock.now += 59.0
    stream.register_interiors(draw, _interior(K, reg.digest, row=2), _ports(K, INTERIOR_VALUES))
    assert stream.deliver(draw, [SESSION]) is None
    clock.now += 600.0
    assert stream.outcome(draw, _lean(), {}).accepted and stream.symbols == ("late", "late", "accept")


def test_a_draw_of_no_rus_has_no_interiors_and_its_proof_goes_to_the_boundary():
    law = {"law": "bernoulli", "p": "0/1"}
    stream = _stream(_spec(S.TwoStage(S.Law("bernoulli:0/1"), "all")))
    reg = stream.register(_boundary(law=law), _ports(N, BOUNDARY))
    draw = stream.select(reg)
    assert draw.units == draw.proved == []
    for record in (_interior(1, reg.digest), {**_interior(1, reg.digest), "population": 0}):
        with pytest.raises(S.Refused) as e:
            stream.register_interiors(draw, record, _ports(1, INTERIOR_VALUES))
        assert e.value.verdict.codes == ("interior-empty",)
    assert stream.deliver(draw, [SESSION]) is None
    out = stream.outcome(draw, _lean(), {}, session_roots={**_roots(BOUNDARY), **_roots(INTERIOR_VALUES)})
    assert out.accepted and stream.symbols == ("accept",) and stream.records[0]["interior"] is None
    assert stream.register(_boundary(law=law, row=1, previous=reg.digest), _ports(N, BOUNDARY)).order == 1


def test_what_a_two_stage_spec_and_stream_refuse():
    with pytest.raises(ValueError, match="count"):
        _stream(_spec(S.TwoStage(S.Law("subset:3"), 1)))
    with pytest.raises(ValueError, match="interiors"):
        _spec(values=VALUES[:3])
    with pytest.raises(ValueError, match="interiors"):
        _spec(values=VALUES[3:])
    with pytest.raises(ValueError, match="registered"):
        _spec(values=(*VALUES, S.Value("W", None, "interior", registered=True)))
    with pytest.raises(ValueError, match="interiors"):
        _spec(values=(*VALUES, S.Value("answers", SCHEMA, "in-deadline")))
    with pytest.raises(ValueError, match="window"):
        _stream(S.Spec(VALUES, S.TwoStage(S.Law("subset:3"), "all"),
                       S.Check(P.program_sha512(Program(TILE)), QUERY, window="ab" * 64), PUBLIC, b"w"))


def test_the_draw_is_the_challengers_own_randomness():
    def drawn(seed):
        stream = _stream(seed=seed)
        return stream.select(stream.register(_boundary(), _ports(N, BOUNDARY))).units

    assert drawn(3) == drawn(3)
    assert len({tuple(drawn(s)) for s in range(16)}) > 1
    stream = S.Stream(_spec(), TILE, budget=1, delta=DELTA, challenger="c")
    assert len(stream.select(stream.register(_boundary(), _ports(N, BOUNDARY))).units) == K
