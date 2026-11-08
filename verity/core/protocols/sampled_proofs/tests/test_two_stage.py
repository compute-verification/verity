"""A two-stage sampling design on the proof system's `Stream`: the boundary registration over replay blocks, the first
stage, the interior registration over the replay sample's proof blocks, the second stage on its own coins, one outcome
binding both registrations and both samples, each step by both verifiers; round 2's statement check of the second-stage
sessions (`second_stage`), which refuses the two-stage form while `second_stage.FORM` is off; its refusals; and its
profile at the footprint."""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb

import pytest

from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.profile import Bernoulli, Subset
from verity.core.protocols.sampled_proofs import second_stage as SS, service as S
from verity.core.protocols.sampled_proofs.one_stage import (draw as D, partition as P, registration as R,
                                                            verifier as V)

import fake_challenger as FC

XOR8 = PrimitiveDefinition("TwoStageTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("TwoStageTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
#: A replay block: three proof blocks, instances of PAIR.
BLOCK = bind(CompositeDefinition("TwoStageTestBlock", 0, (), lambda S_: ((("x", Array(3, Array(2, Bits(8)))),),
                                                                         Array(3, Bits(8))),
                                 lambda B, S_, x: B.batch(PAIR, x, axes=(0,)), register=False))
QUERY = P.template_instance_query(PAIR)
GRAIN = P.template_instance_query(BLOCK)
N_RHO = 3
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-two-stage-test"
SESSION = "5e" * 64
DELTA = 2.0 ** -40
N, Q, K = 4, 2, 1
BOUNDARY = ("bx", "bout")
INTERIOR = ("x", "out")
PUBLIC = (S.Public("population", "structure", "N, the sample's range"),
          S.Public("window", "structure", "provenance: served, on test prompts (step 0)"),
          *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k} (step 0)") for v in BOUNDARY + INTERIOR
            for k in ("leaves", "domain")))
VALUES = (*(S.Value(v, SCHEMA, "serve") for v in BOUNDARY), *(S.Value(v, SCHEMA, "interior") for v in INTERIOR))
RULE = S.TwoStage(GRAIN, f"subset:{Q}", K)
TEMPLATE, = (t for t, _ in P.strata(P.population_program(PAIR, Q * N_RHO), QUERY))
#: The design's one link: PAIR's session port `y`, of no interior root, reads the boundary's `bx`.
LINK = {"template": TEMPLATE, "port": "y", "value": "bx"}


def _spec(selection=RULE, values=VALUES, proof_deadline=60.0) -> S.Spec:
    return S.Spec(values, selection, S.Check(P.program_digest(Program(PAIR)), QUERY, proof_deadline=proof_deadline),
                  PUBLIC, b"two-stage-test")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _stream(spec=None, clock=None, **kwargs) -> S.Stream:
    return FC.Stream(spec or _spec(), PAIR, **{"replay": BLOCK, "budget": 2, "delta": DELTA,
                                               "challenger": "challenger-test", "clock": clock or Clock(), **kwargs})


def _ports(program: Program, query, names, n: int) -> dict:
    part = P.partition(program, query)
    return {v: {"leaves": n, "domain": R.window_domain(part["program"], P.digest(part), v, SCHEMA, n, RUN)} for v in names}


def _record(program: Program, query, names, n: int, law, window, previous) -> dict:
    ports = _ports(program, query, names, n)
    roots = [{"port": v, "root": f"{i + 1:02x}" * 64, **ports[v]} for i, v in enumerate(names)]
    return R.record(program, query, law, S.SCHEME, roots, window, previous=previous)


def _boundary(n=N, row=0, previous=None) -> tuple[dict, dict]:
    program = P.population_program(BLOCK, n)
    return (_record(program, GRAIN, BOUNDARY, n, {"law": "subset", "k": Q}, {"kind": "served", "run": RUN, "row": row},
                    previous), _ports(program, GRAIN, BOUNDARY, n))


def _interior(m: int, previous: str, row=0, **change) -> tuple[dict, dict]:
    program = P.population_program(PAIR, m)
    rec = _record(program, QUERY, INTERIOR, m, {"law": "subset", "k": m},
                  {"kind": "served", "run": RUN, "row": row, "interior": True}, previous)
    return {**rec, **change}, _ports(program, QUERY, INTERIOR, m)


def _lean(accepted=True, reads=None) -> dict:
    v = {"of_record": True, "accepted": accepted, "digest": "dd" * 64, "sessions": [SESSION]}
    return {"lean": v if reads is None else {**v, "registered": reads}}


def _roots(names) -> dict:
    return {v: f"{i + 1:02x}" * 64 for i, v in enumerate(names)}


def _refused(f) -> tuple[str, ...]:
    with pytest.raises(S.Refused) as e:
        f()
    return e.value.verdict.codes


def _stages(stream: S.Stream, row=0, links=None) -> tuple[S.Draw, S.Draw]:
    previous = R.digest(stream._log[-1]) if stream._log else None
    reg = stream.register(*_boundary(row=row, previous=previous))
    first = stream.select(reg)
    stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, row), links=links)
    return first, stream.select(stream._entries[reg.order].interior)


def _reads_of(statements) -> dict | None:
    """What a verifier of record reports the sessions read (`{port: {"value", "root"}}`): the statements' registered
    reads, None when none reads any."""
    out = {}
    for s in statements if isinstance(statements, list) else ():
        ports = s.get("registered", {}).get("ports") if isinstance(s, dict) and isinstance(s.get("registered"), dict) else None
        out |= {p: {"value": e["value"], "root": e["root"]} for p, e in (ports or {}).items()}
    return out or None


def _two_stage_outcome(stream: S.Stream, statements, links=(LINK,), reads=_reads_of) -> S.Outcome:
    """The run's outcome with the second-stage statements `statements` gives (a function of the honest prover's one,
    `fake_challenger.Stream.statements`, PAIR's, and the second stage's sample, or as given), the design's `links`, and
    the verifier of record reporting the reads `reads` gives of them."""
    _, second = _stages(stream, links=list(links))
    stream.deliver(second, [SESSION])
    if callable(statements):
        statements = statements(stream.statements(second)[0], second)
    return stream.outcome(second, _lean(reads=reads(statements)), {"statement": "stub"},
                          session_roots={**_roots(BOUNDARY), **_roots(INTERIOR)}, statements=statements)


def test_a_two_stage_stream_runs_both_stages_and_its_outcome_binds_both():
    """The boundary registration's population is the replay blocks; the first stage samples Q of them; the interior
    registration commits their Q·n_rho proof blocks after it, logged after the registration; the second stage samples k in
    each sampled replay block on the coins after the interior registration; and one outcome holds both audit records, each
    by both verifiers and the whole run by the Lean's. Round 2 holds the session's statement to the interior registration
    and the second stage's cells, and passes it; while the two-stage form is refused (`second_stage.FORM`) that refusal is
    the outcome's one failure, a reject. The profile an accepted one carries is at the footprint."""
    stream = _stream()
    assert stream.n_rho == N_RHO and stream._params["two_stage"] is True
    reg = stream.register(*_boundary())
    assert reg.record["population"] == N and reg.record["query"] == R.query_digest(GRAIN)
    first = stream.select(reg)
    assert first.draw["law"] == "subset" and len(first.units) == Q and first.replay_sample is None
    ireg = stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest))
    assert ireg.order == reg.order and ireg.receipt["received"] == "1" and stream.log[-1] == ireg.record
    second = stream.select(ireg)
    assert second.replay_sample == first and second.law == S.second_law(first.units, N_RHO, K)
    assert [u // N_RHO for u in second.units] == list(range(Q))
    assert stream.deliver(second, [SESSION]) is None
    statements = stream.statements(second)
    (template, _), = P.strata(P.population_program(PAIR, Q * N_RHO), QUERY)
    assert [(s["template"], s["draw"]["units"], s["header"]["units"]["indices"]) for s in statements] == \
        [(template, D.proved(second.draw), list(range(Q * N_RHO)))]
    out = stream.outcome(second, _lean(), {"statement": "stub"}, session_roots={**_roots(BOUNDARY), **_roots(INTERIOR)},
                         statements=statements)
    assert not SS.FORM and not out.accepted and out.profile is None and stream.symbols == ("reject",)
    assert out.verdict.detail["reject"] == {"checks": ["statements"], "verdicts": {"lean": True}, "replay_sample": []}

    body = stream.records[0]
    assert S._sha512(body) == out.record and body["registration"] == reg.digest
    assert body["draw"] == S._sha512(second.draw) and "run" not in body["audit"]["checks"]
    assert body["audit"]["checks"]["statements"] == {"ok": False, "codes": ["two-stage-form"],
                                                     "detail": {"two-stage-form": SS.FORM_WHY}}
    assert body["replay_sample"]["draw"] == S._sha512(first.draw) and body["replay_sample"]["audit"]["accepted"]
    assert body["interior"] == {"registration": ireg.digest, "receipt": S._sha512(ireg.receipt)}

    p = stream._footprint(stream._entries[reg.order])
    assert [lv.name for lv in p.levels] == [S.REPLAY, "unit"] and p.sizes == (N, N_RHO)
    assert p.draws == {S.REPLAY: Bernoulli(Fraction(Q, N)), "unit": Subset(K)} and dict(p.drawn) == {S.REPLAY: Q, "unit": Q * K}
    assert p.programs == (reg.record["program"], ireg.record["program"])
    for m in range(N_RHO + 1):
        assert p.accept(m) == 1 - Fraction(Q, N) + Fraction(Q, N) * Fraction(comb(N_RHO - m, K), comb(N_RHO, K))


def test_the_run_of_record_holds_the_form_switch(monkeypatch):
    """`second_stage.FORM` is the Lean's `SecondStage.FORM`, which the run of record reads: flipped in the Python alone,
    round 2's check passes and the Python would accept, and the run of record, still refusing the form, disagrees."""
    monkeypatch.setattr(SS, "FORM", True)
    stream = _stream()
    out = _two_stage_outcome(stream, lambda s, second: [s])
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["ok"] and checks["run"]["codes"] == [V.DISAGREE]
    assert {"interior", "symbol"} <= set(checks["run"]["detail"][V.DISAGREE]["fields"])


def _link(second: S.Draw, shift: int = 0) -> dict:
    """Registered reads under the linked port `y` of boundary port `bx`, one row a replay block: each slot's of its cell's
    replay block, or of the one `shift` on."""
    first = second.replay_sample
    bx, = (r for r in first.registration.record["roots"] if r["port"] == "bx")
    sample = D.proved(first.draw)
    return {"format": "verity/registered-reads/v1", "ports": {"y": {
        "value": "bx", "root": bx["root"], "domain": bx["domain"], "leaves": bx["leaves"], "reads_from": 0,
        "reads": [(sample[u // N_RHO] + shift) % N for u in range(Q * N_RHO)]}}}


def _own(second: S.Draw, shift: int = 0) -> dict:
    """Registered reads under PAIR's port `x` of the interior's own `x`, one row a slot: each slot's own, or the one
    `shift` on."""
    x, = (r for r in second.registration.record["roots"] if r["port"] == "x")
    return {"format": "verity/registered-reads/v1", "ports": {"x": {
        "value": "x", "root": x["root"], "domain": x["domain"], "leaves": x["leaves"], "reads_from": 0,
        "reads": [(u + shift) % (Q * N_RHO) for u in range(Q * N_RHO)]}}}


def _restated(**change):
    """The honest statement with `change`, each a field or a function of the honest field's value."""
    return lambda s, second: [{**s, **{k: f(s[k]) if callable(f) else f for k, f in change.items()}}]


def _header(**change):
    return _restated(header=lambda h: {**h, **change})


def _frame(key: str, f):
    return _restated(header=lambda h: {**h, "frame_v3": {**h["frame_v3"], key: f(h["frame_v3"][key])}})


@pytest.mark.parametrize("statements, codes", [
    (lambda s, second: [{**s, "registered": _link(second)}], ["two-stage-form"]),
    (lambda s, second: [{**s, "registered": _own(second)}], ["two-stage-form"]),
    (None, ["second-statements"]),
    ([], ["second-cover"]),
    (lambda s, second: [s, s], ["second-template"]),
    (_restated(template="TwoStageTestOther_v0"), ["second-template", "second-cover"]),
    (lambda s, second: [{k: v for k, v in s.items() if k != "registered"}], ["second-statement", "second-cover"]),
    (_restated(header=None), ["second-statement", "second-cover"]),
    (_header(instances=Q * N_RHO + 1), ["second-population"]),
    (_header(units={"indices": list(range(1, Q * N_RHO + 1))}), ["second-population"]),
    (_header(scheme="frame-v3-sha512/other"), ["second-roots"]),
    (_frame("roots", lambda r: {**r, "x": "00" * 64}), ["second-roots"]),
    (_frame("roots", lambda r: {k: v for k, v in r.items() if k != "out"}), ["second-roots"]),
    (_frame("domain_ids", lambda d: {**d, "out": d["x"]}), ["second-roots"]),
    (_restated(draw=lambda d: {**d, "units": [0, 3] if d["units"] != [0, 3] else [1, 4]}), ["second-sample"]),
    (_restated(draw=lambda d: {**d, "population": N * N_RHO}), ["second-sample"]),
    (lambda s, second: [{**s, "registered": _link(second, shift=1)}], ["second-read"]),
    (lambda s, second: [{**s, "registered": _own(second, shift=1)}], ["second-read"]),
    (lambda s, second: [{**s, "registered": {"ports": {"x": _link(second)["ports"]["y"]}}}], ["second-read"]),
    (lambda s, second: [{**s, "registered": {"ports": {"y": _own(second)["ports"]["x"]}}}], ["second-read"]),
])
def test_round_two_refuses_a_statement_the_run_did_not_make(statements, codes):
    """Round 2, by both verifiers, passes the run's own statements, its linked port reading the boundary at each slot's
    cell's replay block or its own port reading each slot's own rows included (the form's refusal their one failure),
    and refuses: none (`second-statements`), none for a template the second stage drew in (`second-cover`), a template
    twice or one of no part (`second-template`), one that isn't a statement (`second-statement`), and one that isn't the
    run's: another population of slots (`second-population`), a scheme, root or domain the interior registration's isn't
    or a port of it it neither roots nor reads (`second-roots`), another sample (`second-sample`), or a read of another
    cell's replay block, another slot's rows, or a port the design doesn't link or another port's own (`second-read`).
    Neither audit record refuses the reads of its own ports at their own roots, which round 2 holds."""
    stream = _stream()
    out = _two_stage_outcome(stream, statements)
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["codes"] == codes and "run" not in checks, checks["statements"]
    assert "registered" not in checks and stream.records[0]["replay_sample"]["audit"]["accepted"]


@pytest.mark.parametrize("links, reads, codes", [
    ((), _reads_of, ["second-read"]),
    ((LINK,), lambda ss: None, ["second-reported"]),
    ((LINK,), lambda ss: {"y": {"value": "bx", "root": "ee" * 64}}, ["second-reported"]),
    ((LINK,), lambda ss: {**_reads_of(ss), "z": {"value": "bout", "root": "02" * 64}}, ["second-reported"]),
])
def test_round_two_reads_only_what_the_design_links_and_the_verdicts_report(links, reads, codes):
    """A boundary port is read only by the session port the design's links name for it (`second-read`), and the reads
    the verifier of record reports are exactly the statements' (`second-reported`): none, another root, or one more."""
    stream = _stream()
    out = _two_stage_outcome(stream, lambda s, second: [{**s, "registered": _link(second)}], links, reads)
    assert not out.accepted and stream.records[0]["audit"]["checks"]["statements"]["codes"] == codes


def test_an_own_read_at_another_root_fails_the_interior_audit():
    """The interior audit takes a read of its own port only at that port's root: at another, its `registered` check
    fails beside round 2's (`second-read`)."""
    stream = _stream()
    out = _two_stage_outcome(stream, lambda s, second: [{**s, "registered": {"ports": {"x": {
        **_own(second)["ports"]["x"], "root": "ee" * 64}}}}])
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and not checks["registered"]["ok"] and checks["statements"]["codes"] == ["second-read"]


def test_a_rejected_verdict_rejects_and_names_both_records():
    stream = _stream()
    _, second = _stages(stream)
    stream.deliver(second, [SESSION])
    out = stream.outcome(second, _lean(accepted=False), {"statement": "stub"})
    assert not out.accepted and out.verdict.codes == ("reject",) and out.profile is None
    assert set(out.verdict.detail["reject"]) == {"checks", "verdicts", "replay_sample"}
    assert stream.records[0]["replay_sample"]["audit"]["accepted"] is False


def test_the_proof_deadline_counts_from_the_first_stage():
    clock = Clock()
    stream = _stream(clock=clock)
    _, second = _stages(stream)
    clock.now = 61.0
    late = stream.deliver(second, [SESSION])
    assert late is not None and late.verdict.codes == ("late",) and stream.symbols == ("late",)


def test_each_stage_is_once_and_in_order():
    """The interior registration follows the first stage and precedes the second, each once; the proof is the second
    stage's sample's; and a two-stage stream opens no session over several rounds."""
    stream = _stream()
    reg = stream.register(*_boundary())
    first = stream.select(reg)
    assert _refused(lambda: stream.deliver(first, [SESSION])) == ("second",)
    assert _refused(lambda: stream.session([first])) == ("rule",)
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO + 1, reg.digest))) == \
        ("interior-population",)
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO, "00" * 64))) == ("R6-previous",)
    ireg = stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest))
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, row=1))) == \
        ("interior-registered",)
    assert _refused(lambda: stream.select(reg)) == ("drawn",)
    second = stream.select(ireg)
    assert _refused(lambda: stream.select(ireg)) == ("drawn",)
    assert _refused(lambda: stream.deliver(first, [SESSION])) == ("not-drawn",)
    assert stream.deliver(second, [SESSION]) is None
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest))) == ("delivered",)


def test_a_one_stage_stream_has_no_interior_registration():
    spec = S.Spec(VALUES[:2], S.Law("subset:2"), S.Check(P.program_digest(Program(BLOCK)), GRAIN, proof_deadline=60.0),
                  PUBLIC, b"two-stage-test")
    stream = FC.Stream(spec, BLOCK, budget=2, delta=DELTA, challenger="challenger-test", clock=Clock())
    reg = stream.register(*_boundary())
    first = stream.select(reg)
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest))) == ("rule",)
    assert _refused(lambda: stream.outcome(first, _lean(), {"statement": "stub"}, statements=[])) == ("statements",)


def test_the_stream_holds_the_design_to_its_replay_block():
    one = bind(CompositeDefinition("TwoStageTestOne", 0, (), lambda S_: ((("x", Array(1, Array(2, Bits(8)))),),
                                                                       Array(1, Bits(8))),
                                   lambda B, S_, x: B.batch(PAIR, x, axes=(0,)), register=False))
    with pytest.raises(ValueError, match="replay="):
        _stream(replay=None)
    with pytest.raises(ValueError, match="its rule is 'law'"):
        _stream(spec=_spec(S.TwoStage(P.template_instance_query(one), f"subset:{Q}", K)), replay=one)
    with pytest.raises(ValueError, match="grain is not the template query"):
        _stream(spec=_spec(S.TwoStage(QUERY, f"subset:{Q}", K)))
    with pytest.raises(ValueError, match="would prove whole replay blocks"):
        _stream(spec=_spec(S.TwoStage(GRAIN, f"subset:{Q}", N_RHO)))
    with pytest.raises(ValueError, match="one round"):
        _stream(rounds=2)
    with pytest.raises(ValueError, match="only a two-stage stream"):
        FC.Stream(S.Spec(VALUES[:2], S.Law("subset:2"), _spec().check, PUBLIC, b"t"), PAIR, replay=BLOCK, budget=2,
                  delta=DELTA, challenger="challenger-test", clock=Clock())


def test_the_rule_and_the_spec_state_one_form():
    assert RULE.to_json() == {"rule": "two-stage", "grain": GRAIN, "replay": f"subset:{Q}", "k": K}
    assert S.TwoStage(GRAIN, "rate:1/2", 2).replay == "rate:1/2"
    for bad, match in (((dict(GRAIN, version=1), "subset:2", 1), "grain"), ((GRAIN, "stratified:2", 1), "replay law"),
                       ((GRAIN, "subset:2", 0), "k >= 1")):
        with pytest.raises(ValueError, match=match):
            S.TwoStage(*bad)
    for retired in (RULE.to_json(), {"rule": "two-stage", "replay": "subset:4", "check": "all"}, "two-stage:subset:4"):
        with pytest.raises(ValueError, match="a selection rule is All, Law, Sequential or TwoStage"):
            _spec(selection=retired)
    with pytest.raises(ValueError, match="interior values"):
        _spec(values=VALUES[:2])
    with pytest.raises(ValueError, match="one-stage selection has none"):
        _spec(selection=S.Law("subset:2"))
    assert S.selection_law(RULE, N) == {"law": "subset", "k": Q}
    assert S.boundary_values(VALUES) == VALUES[:2] and S.interior_values(VALUES) == VALUES[2:]


def test_the_verifier_of_record_draws_both_stages():
    """The verifier's executable (`Pieces.verifier`, `verifier.EXE`) samples a two-stage rule's first stage, `Q`
    of the `N` replay blocks, and its second, `k` of each sampled replay block's proof blocks, as the service does."""
    assert V.EXE == "sampled-proofs-verifier" and _spec().selection == RULE
    source = random.Random(1).randbytes(4096)
    first = {"rule": RULE.to_json(), "program": codec.encode_program(P.population_program(BLOCK, N)), "query": GRAIN,
             "work_table": None, "closure": None, "source": source.hex()}
    assert len(V.lean("select", first)["draw"]["units"]) == Q
    source = random.Random(2).randbytes(4096)
    second = {"sample": [0, 2], "n": N_RHO, "k": K, "source": source.hex()}
    drawn = V.lean("select_second", second)
    read = iter(source)
    assert drawn["draw"] == S.select_second([0, 2], N_RHO, K, lambda m: bytes(next(read) for _ in range(m)))
    assert sorted(u // N_RHO for u in drawn["draw"]["units"]) == [0, 1]


def test_the_second_stage_samples_k_in_each_sampled_replay_block():
    law = S.second_law([1, 3], 4, 2)
    assert law == {"law": "stratified", "k": 4, "strata": [{"template": "1", "units": [[0, 4]], "k": 2},
                                                          {"template": "3", "units": [[4, 8]], "k": 2}]}
    assert D.strata_problem(law, 8) is None
    obj = S.select_second([1, 3], 4, 2, random.Random(5).randbytes)
    assert obj["population"] == 8 and len(obj["units"]) == 4
    assert sorted(u // 4 for u in obj["units"]) == [0, 0, 1, 1]
    for sample, n, k, match in (([], 4, 1, "ascending"), ([2, 1], 4, 1, "ascending"), ([0], 1, 1, "'law'"),
                                ([0], 4, 4, "whole replay blocks"), ([0], 4, 0, "whole replay blocks")):
        with pytest.raises(ValueError, match=match):
            S.select_second(sample, n, k, lambda m: bytes(m))
