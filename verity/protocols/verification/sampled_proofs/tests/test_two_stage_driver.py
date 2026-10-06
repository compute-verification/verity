"""The two-stage driver on the proof service's Streams: the four steps in order, the draw as the Lean verifier's from the
verifier's own bytes, the interiors held to the rule, one record binding both registrations, both receipts and the draw,
and no retries. PoUW's example follows the architecture note: a uniform subset:K' over N equal tiles."""

from __future__ import annotations

import hashlib
import random

import pytest

from verity.primitives.circuits.codec import canonical_json
from verity.primitives.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.primitives.circuits.program import Program
from verity.primitives.circuits.types import Array, Value as Bits
from verity.protocols.profile import Subset
from verity.protocols.verification.sampled_proofs import service as S
from verity.protocols.verification.sampled_proofs.one_stage import draw as D, partition as P, registration as R
from verity.protocols.verification.sampled_proofs.two_stage import REPLAY, OwnRandomness, Stage, TwoStageLaw, draw_first
from verity.protocols.verification.sampled_proofs.two_stage_driver import AUDIT, INTERIOR, TwoStageStream

XOR8 = PrimitiveDefinition("DriverTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
TILE = bind(CompositeDefinition("DriverTestTile", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
QUERY = P.template_instance_query(TILE)
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-driver-test"
SESSION = "5e" * 64
DELTA = 2.0 ** -40
N, K = 8, 3
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
                  public, b"driver-test")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _seeded(seed: int) -> OwnRandomness:
    return OwnRandomness(random.Random(seed).randbytes)


def _driver(spec=None, budget=2, seed=1, clock=None) -> TwoStageStream:
    return TwoStageStream(spec or _spec(), TILE, budget=budget, delta=DELTA, challenger="challenger-test",
                          randomness=_seeded(seed), clock=clock or Clock())


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


def _interior(n: int, row=0, **change) -> dict:
    return _record(n, INTERIOR_VALUES, {"law": "subset", "k": n}, {"kind": "served", "run": RUN, "row": row,
                                                                   "stage": "interior"}, **change)


def _roots(names, wrong: str | None = None) -> dict:
    return {v: ("ee" * 64 if v == wrong else f"{i + 1:02x}" * 64) for i, v in enumerate(names)}


def _lean(accepted=True, sessions=(SESSION,)) -> dict:
    return {"lean": {"of_record": True, "accepted": accepted, "digest": "dd" * 64, "sessions": list(sessions)}}


def _sha512(obj) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


def _audit(driver: TwoStageStream, accepted=True, row=0, session_roots=None, closure=None, **boundary) -> S.Outcome:
    previous = R.digest(driver._audits[-1].registration.record) if driver._audits else None
    reg = driver.register(_boundary(row=row, previous=previous, **boundary), _ports(N, BOUNDARY),
                          **({"work_table": boundary["work_table"]} if "work_table" in boundary else {}))
    draw = driver.select(reg, closure)
    driver.register_interiors(draw, _interior(len(draw.proved), row), _ports(len(draw.proved), INTERIOR_VALUES))
    assert driver.deliver(draw, [SESSION]) is None
    return driver.outcome(draw, _lean(accepted), {"statement": "stub"}, session_roots=session_roots)


def test_the_pouw_example_runs_the_four_steps_and_one_record_binds_them():
    driver = _driver()
    reg = driver.register(_boundary(), _ports(N, BOUNDARY))
    assert reg.order == 0 and reg.receipt["received"] == "0" and reg.record["population"] == N
    draw = driver.select(reg)
    assert draw.units == draw_first(_seeded(1), {"law": "subset", "k": K}, N)["units"] == draw.draw.units
    assert len(draw.units) == K and draw.proved == draw.units and "closure" not in draw.replay
    inner = driver.register_interiors(draw, _interior(K), _ports(K, INTERIOR_VALUES))
    assert inner.record["population"] == K and inner.record["law"] == {"law": "subset", "k": K}
    assert driver.deliver(draw, [SESSION]) is None
    out = driver.outcome(draw, _lean(), {"statement": "stub"},
                         session_roots={**_roots(BOUNDARY), **_roots(INTERIOR_VALUES)})
    assert out.accepted and out.verdict.codes == () and out.exhaustive is None and out.terms == S.Terms(DELTA)
    assert driver.symbols == ("accept",) and driver.failures == 0

    law = TwoStageLaw({"tile": Stage({"law": "subset", "k": K})}).profile("tile", n_r=N, delta=DELTA)
    assert out.profile.draws["unit"] == law.draws[REPLAY] == Subset(K) and out.profile.sizes == law.sizes == (N,)
    assert out.profile.bound() == law.bound()

    body = driver.records[0]
    assert _sha512(body) == out.record and body["format"] == AUDIT and body["stream"] == driver.digest
    assert body["boundary"]["registration"] == reg.record and body["boundary"]["receipt"] == reg.receipt
    assert body["interior"]["registration"] == inner.record and body["interior"]["receipt"] == inner.receipt
    assert body["draw"] == draw.replay and body["replay_units"] == draw.units and body["sessions"] == [SESSION]
    bound = driver.boundary_records[0]
    assert body["boundary"]["outcome"] == _sha512(bound) and bound["symbol"] == "accept"
    assert bound["audit"]["verdicts"][INTERIOR] == {"of_record": False, "accepted": True, "sessions": [SESSION],
                                                    "outcome": body["interior"]["outcome"]}


def test_each_step_waits_for_the_one_before():
    driver = _driver()
    reg = driver.register(_boundary(), _ports(N, BOUNDARY))
    with pytest.raises(S.Refused) as e:
        driver.register(_boundary(row=1, previous=reg.digest), _ports(N, BOUNDARY))
    assert e.value.verdict.codes == ("open",)
    draw = driver.select(reg)
    with pytest.raises(S.Pending):
        driver.outcome(draw, _lean(), {})
    with pytest.raises(S.Refused) as e:
        driver.deliver(draw, [SESSION])
    assert e.value.verdict.codes == ("interior",)
    driver.register_interiors(draw, _interior(K), _ports(K, INTERIOR_VALUES))
    with pytest.raises(S.Pending):
        driver.outcome(draw, _lean(), {})
    driver.deliver(draw, [SESSION])
    with pytest.raises(S.Pending):
        driver.outcome(draw, _lean(None), {})
    assert driver.outcome(draw, _lean(), {}).accepted
    assert driver.register(_boundary(row=1, previous=reg.digest), _ports(N, BOUNDARY)).order == 1


def test_the_interiors_are_the_proved_replay_units_held_to_the_rule():
    driver = _driver()
    draw = driver.select(driver.register(_boundary(), _ports(N, BOUNDARY)))
    with pytest.raises(S.Refused) as e:
        driver.register_interiors(draw, _interior(K + 1), _ports(K + 1, INTERIOR_VALUES))
    assert e.value.verdict.codes == ("interior-population",)
    shown = _record(K, ("a_rows", *INTERIOR_VALUES), {"law": "subset", "k": K}, {"kind": "served", "run": RUN, "row": 0})
    with pytest.raises(S.Refused) as e:
        driver.register_interiors(draw, shown, _ports(K, ("a_rows", *INTERIOR_VALUES)))
    assert e.value.verdict.detail["roots"] == ["a_rows"]
    with pytest.raises(S.Refused) as e:
        driver.register_interiors(draw, {**_interior(K), "tiles": draw.units}, _ports(K, INTERIOR_VALUES))
    assert "keys" in e.value.verdict.codes
    with pytest.raises(S.Refused) as e:
        driver.register_interiors(draw, _interior(K), {**_ports(K, INTERIOR_VALUES), "ok_rows": {"leaves": 1, "domain": "00"}})
    assert "R3-leaves" in e.value.verdict.codes
    inner = driver.register_interiors(draw, _interior(K), _ports(K, INTERIOR_VALUES))
    with pytest.raises(S.Refused) as e:
        driver.register_interiors(draw, _interior(K), _ports(K, INTERIOR_VALUES))
    assert e.value.verdict.codes == ("interior-registered",)
    driver.deliver(draw, [SESSION])
    assert driver.outcome(draw, _lean(), {}).accepted and driver.records[0]["interior"]["registration"] == inner.record


def test_a_work_first_stage_draws_with_the_verifiers_closure():
    spec = _spec(S.TwoStage(S.Law("work:2"), "all"),
                 public=(*PUBLIC, S.Public(f"population:{TILE.id}", "structure", "the tiles' one stratum")))
    table = {TILE.id: 3}
    law = D.derive(P.population_program(TILE, N), QUERY, {"law": "work", "k": 2}, table)
    closure = {u: [(u + 1) % N] for u in range(N)}
    driver = _driver(spec, seed=5)
    reg = driver.register(_boundary(law={"law": "work", "k": 2}, work_table=table), _ports(N, BOUNDARY), work_table=table)
    draw = driver.select(reg, closure)
    want = draw_first(_seeded(5), law, N, closure)
    assert draw.replay == want and draw.units == want["units"] and draw.draw.units == want["units"]
    assert draw.proved == sorted(want["units"] + want["closure"]) and len(draw.proved) > len(draw.units)
    assert "closure" not in draw.draw.draw
    with pytest.raises(S.Refused) as e:
        driver.register_interiors(draw, _interior(len(draw.units)), _ports(len(draw.units), INTERIOR_VALUES))
    assert e.value.verdict.codes == ("interior-population",)
    n = len(draw.proved)
    driver.register_interiors(draw, _interior(n), _ports(n, INTERIOR_VALUES))
    driver.deliver(draw, [SESSION])
    out = driver.outcome(draw, _lean(), {})
    assert out.accepted and driver.records[0]["draw"]["closure"] == want["closure"]
    assert driver.records[0]["replay_units"] == draw.proved
    stage = TwoStageLaw({"tile": Stage(law)}).profile("tile", n_r=N, delta=DELTA)
    assert out.profile.draws["unit"] == stage.draws[REPLAY] and out.profile.bound() == stage.bound()

    subset = _driver(seed=5)
    reg = subset.register(_boundary(), _ports(N, BOUNDARY))
    with pytest.raises(ValueError, match="work draw only"):
        subset.select(reg, closure)
    assert subset.select(reg).units == draw_first(_seeded(5), {"law": "subset", "k": K}, N)["units"]


def test_a_wrong_interior_or_boundary_rejects_the_audit_once():
    both = {**_roots(BOUNDARY), **_roots(INTERIOR_VALUES)}
    for wrong in ("cap_rows", "tile_digests"):
        driver = _driver(budget=3)
        out = _audit(driver, session_roots={**_roots(BOUNDARY, wrong), **_roots(INTERIOR_VALUES, wrong)})
        assert not out.accepted and out.verdict.codes == ("reject",) and out.profile is None
        assert driver.symbols == ("reject",) and driver.failures == 1
        verdict = driver.boundary_records[0]["audit"]["verdicts"][INTERIOR]
        assert verdict["accepted"] is (wrong != "cap_rows")
        assert _audit(driver, row=1, session_roots=both).accepted and driver.symbols == ("reject", "accept")
    driver = _driver()
    assert not _audit(driver, accepted=False).accepted and driver.failures == 1


def test_the_driver_stops_at_its_budgeth_failure():
    driver = _driver(budget=2)
    assert not _audit(driver, accepted=False).accepted and not driver.stopped
    assert _audit(driver, row=1).accepted
    assert not _audit(driver, accepted=False, row=2).accepted
    assert driver.stopped and driver.symbols == ("reject", "accept", "reject") and len(driver.records) == 3
    with pytest.raises(S.Refused) as e:
        _audit(driver, row=3)
    assert e.value.verdict.codes == ("stopped",)


def test_no_retries_nothing_redraws_and_an_outcome_is_final():
    driver = _driver()
    reg = driver.register(_boundary(), _ports(N, BOUNDARY))
    draw = driver.select(reg)
    with pytest.raises(S.Refused) as e:
        driver.select(reg)
    assert e.value.verdict.codes == ("drawn",) and driver._audits[0].draw == draw
    driver.register_interiors(draw, _interior(K), _ports(K, INTERIOR_VALUES))
    driver.deliver(draw, [SESSION])
    with pytest.raises(S.Refused) as e:
        driver.deliver(draw, [SESSION])
    assert e.value.verdict.codes == ("delivered",)
    with pytest.raises(S.Refused) as e:
        driver.outcome(draw, {**_lean(), INTERIOR: {"accepted": True}}, {})
    assert e.value.verdict.codes == ("verdicts",)
    assert not driver.outcome(draw, _lean(False), {}).accepted
    for call in (lambda: driver.outcome(draw, _lean(), {}), lambda: driver.deliver(draw, [SESSION]),
                 lambda: driver.register_interiors(draw, _interior(K), _ports(K, INTERIOR_VALUES))):
        with pytest.raises(S.Refused) as e:
            call()
        assert e.value.verdict.codes == ("final",)
    other = _driver()
    with pytest.raises(S.Refused) as e:
        other.register_interiors(draw, _interior(K), _ports(K, INTERIOR_VALUES))
    assert e.value.verdict.codes == ("unknown",)


def test_a_late_proof_or_missing_interiors_are_late():
    clock = Clock()
    driver = _driver(_spec(proof_deadline=60.0), budget=3, clock=clock)
    draw = driver.select(driver.register(_boundary(), _ports(N, BOUNDARY)))
    clock.now = 40.0
    driver.register_interiors(draw, _interior(K), _ports(K, INTERIOR_VALUES))
    clock.now = 60.5
    out = driver.deliver(draw, [SESSION])
    assert out is not None and out.verdict.codes == ("late",) and driver.symbols == ("late",)
    assert driver.records[0]["interior"]["outcome"] is None and driver.records[0]["symbol"] == "late"

    draw = driver.select(driver.register(_boundary(row=1, previous=draw.draw.registration.digest), _ports(N, BOUNDARY)))
    clock.now = 100.0
    with pytest.raises(S.Pending):
        driver.outcome(draw, _lean(), {})
    clock.now = 121.0
    out = driver.outcome(draw, _lean(), {})
    assert out.verdict.codes == ("late",) and driver.records[1]["interior"] is None

    draw = driver.select(driver.register(_boundary(row=2, previous=draw.draw.registration.digest), _ports(N, BOUNDARY)))
    clock.now += 59.0
    driver.register_interiors(draw, _interior(K, row=2), _ports(K, INTERIOR_VALUES))
    assert driver.deliver(draw, [SESSION]) is None
    clock.now += 600.0
    assert driver.outcome(draw, _lean(), {}).accepted and driver.symbols == ("late", "late", "accept")


def test_a_draw_of_no_rus_has_no_interiors_and_its_proof_goes_to_the_boundary():
    law = {"law": "bernoulli", "p": "0/1"}
    driver = _driver(_spec(S.TwoStage(S.Law("bernoulli:0/1"), "all")))
    reg = driver.register(_boundary(law=law), _ports(N, BOUNDARY))
    draw = driver.select(reg)
    assert draw.units == draw.proved == []
    for record in (_interior(1), {**_interior(1), "population": 0}):
        with pytest.raises(S.Refused) as e:
            driver.register_interiors(draw, record, _ports(1, INTERIOR_VALUES))
        assert e.value.verdict.codes == ("interior-population",)
    assert driver.deliver(draw, [SESSION]) is None
    out = driver.outcome(draw, _lean(), {}, session_roots={**_roots(BOUNDARY), **_roots(INTERIOR_VALUES)})
    assert out.accepted and driver.symbols == ("accept",)
    assert driver.records[0]["interior"] is None and driver.records[0]["replay_units"] == []
    assert INTERIOR not in driver.boundary_records[0]["audit"]["verdicts"]
    assert driver.register(_boundary(law=law, row=1, previous=reg.digest), _ports(N, BOUNDARY)).order == 1


def test_what_the_driver_refuses():
    for selection in (S.Law("subset:3"), S.All(), S.TwoStage(S.Law("subset:3"), 1)):
        with pytest.raises(ValueError):
            _driver(_spec(selection))
    with pytest.raises(ValueError, match="interiors"):
        _driver(_spec(values=VALUES[:3]))
    with pytest.raises(ValueError, match="interiors"):
        _driver(_spec(values=VALUES[3:]))
    with pytest.raises(ValueError, match="registered"):
        _driver(_spec(values=(*VALUES, S.Value("W", None, "interior", registered=True))))
    with pytest.raises(ValueError, match="deadline"):
        _driver(_spec(values=(*VALUES, S.Value("answers", SCHEMA, "in-deadline"))))
    with pytest.raises(ValueError, match="window"):
        _driver(S.Spec(VALUES, S.TwoStage(S.Law("subset:3"), "all"),
                       S.Check(P.program_sha512(Program(TILE)), QUERY, window="ab" * 64), PUBLIC, b"w"))


def test_the_draw_is_the_verifiers_own_randomness():
    def drawn(seed):
        driver = _driver(seed=seed)
        return driver.select(driver.register(_boundary(), _ports(N, BOUNDARY))).units

    assert drawn(3) == drawn(3)
    assert len({tuple(drawn(s)) for s in range(16)}) > 1
    driver = TwoStageStream(_spec(), TILE, budget=1, delta=DELTA, challenger="c")
    assert len(driver.select(driver.register(_boundary(), _ports(N, BOUNDARY))).units) == K
