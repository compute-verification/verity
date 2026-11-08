"""A two-stage sampling design on the proof system's `Stream`: the boundary registration over replay blocks, the first
stage, the interior registration over the replay sample's proof blocks, the second stage on its own coins, one outcome
binding both registrations and both samples, each step by both verifiers; its refusals; and its profile at the
footprint."""

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
from verity.core.protocols.sampled_proofs import service as S
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


def _boundary(n=N, row=0, previous=None, q=Q) -> tuple[dict, dict]:
    program = P.population_program(BLOCK, n)
    return (_record(program, GRAIN, BOUNDARY, n, {"law": "subset", "k": q}, {"kind": "served", "run": RUN, "row": row},
                    previous), _ports(program, GRAIN, BOUNDARY, n))


def _interior(m: int, previous: str, row=0, **change) -> tuple[dict, dict]:
    program = P.population_program(PAIR, m)
    rec = _record(program, QUERY, INTERIOR, m, {"law": "subset", "k": m},
                  {"kind": "served", "run": RUN, "row": row, "interior": True}, previous)
    return {**rec, **change}, _ports(program, QUERY, INTERIOR, m)


def _lean(accepted=True) -> dict:
    return {"lean": {"of_record": True, "accepted": accepted, "digest": "dd" * 64, "sessions": [SESSION]}}


def _roots(names) -> dict:
    return {v: f"{i + 1:02x}" * 64 for i, v in enumerate(names)}


def _refused(f) -> tuple[str, ...]:
    with pytest.raises(S.Refused) as e:
        f()
    return e.value.verdict.codes


def _stages(stream: S.Stream, row=0) -> tuple[S.Draw, S.Draw]:
    previous = R.digest(stream._log[-1]) if stream._log else None
    reg = stream.register(*_boundary(row=row, previous=previous))
    first = stream.select(reg)
    stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, row))
    return first, stream.select(stream._entries[reg.order].interior)


def test_a_two_stage_stream_runs_both_stages_and_its_outcome_binds_both():
    """The boundary registration's population is the replay blocks; the first stage samples Q of them; the interior
    registration commits their Q·n_rho proof blocks after it, logged after the registration; the second stage samples k in
    each sampled replay block on the coins after the interior registration; and one accepted outcome holds both audit
    records, each by both verifiers and the whole run by the Lean's, with the profile at the footprint."""
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
    out = stream.outcome(second, _lean(), {"statement": "stub"}, session_roots={**_roots(BOUNDARY), **_roots(INTERIOR)})
    assert out.accepted and out.verdict.codes == () and out.terms == S.Terms(DELTA) and stream.symbols == ("accept",)

    body = stream.records[0]
    assert S._sha512(body) == out.record and body["registration"] == reg.digest
    assert body["draw"] == S._sha512(second.draw) and body["audit"]["accepted"]
    assert body["replay_sample"]["draw"] == S._sha512(first.draw) and body["replay_sample"]["audit"]["accepted"]
    assert body["interior"] == {"registration": ireg.digest, "receipt": S._sha512(ireg.receipt)}

    p = out.profile
    assert [lv.name for lv in p.levels] == [S.REPLAY, "unit"] and p.sizes == (N, N_RHO)
    assert p.draws == {S.REPLAY: Bernoulli(Fraction(Q, N)), "unit": Subset(K)} and dict(p.drawn) == {S.REPLAY: Q, "unit": Q * K}
    assert p.programs == (reg.record["program"], ireg.record["program"])
    for m in range(N_RHO + 1):
        assert p.accept(m) == 1 - Fraction(Q, N) + Fraction(Q, N) * Fraction(comb(N_RHO - m, K), comb(N_RHO, K))


def test_a_rate_first_stage_keeps_one_wrong_proof_blocks_escape_however_the_prover_pads_the_replay_blocks():
    """`rate:1/2` samples Q = ⌈N/2⌉ of each registration's N replay blocks, so a registration padded tenfold (N = 40 for
    4) samples Q = 20, and a wrong proof block escapes with probability 1 − (Q/N)·(k/n_rho) = 5/6 at both sizes. The second
    stage's k/n_rho is the replay Definition's, which the registration can't pad. Under `subset:2` the same padding raises
    that escape from 5/6 to 59/60."""
    def run(rule, n, q):
        stream = _stream(_spec(rule))
        assert S.selection_law(stream.spec.selection, n) == {"law": "subset", "k": q}
        reg = stream.register(*_boundary(n, q=q))
        first = stream.select(reg)
        assert len(first.units) == q
        stream.register_interiors(first, *_interior(q * N_RHO, reg.digest))
        second = stream.select(stream._entries[reg.order].interior)
        assert len(second.units) == q * K and stream.deliver(second, [SESSION]) is None
        out = stream.outcome(second, _lean(), {"statement": "stub"}, session_roots={**_roots(BOUNDARY), **_roots(INTERIOR)})
        assert out.accepted and out.profile.draws == {S.REPLAY: Bernoulli(Fraction(q, n)), "unit": Subset(K)}
        return out.profile.accept(1)

    rate = S.TwoStage(GRAIN, "rate:1/2", K)
    assert run(rate, 4, 2) == run(rate, 40, 20) == 1 - Fraction(1, 2) * Fraction(K, N_RHO) == Fraction(5, 6)
    assert run(RULE, 4, Q) == Fraction(5, 6) and run(RULE, 40, Q) == Fraction(59, 60)


@pytest.mark.parametrize("port", BOUNDARY)
def test_a_replay_block_is_held_to_the_boundary_values_committed_while_serving(port):
    """A replay block's starting state (`bx`) and its far edge (`bout`) are boundary values, committed in the registration
    before the first stage's coins. An outcome whose sessions opened either at another root rejects, the registration's
    audit record failing `roots` on that port, so a replay can't start from, or end at, a state other than the
    one committed while serving."""
    stream = _stream()
    _, second = _stages(stream)
    assert stream.deliver(second, [SESSION]) is None
    swapped = {**_roots(BOUNDARY), port: "ee" * 64, **_roots(INTERIOR)}
    out = stream.outcome(second, _lean(), {"statement": "stub"}, session_roots=swapped)
    assert not out.accepted and out.verdict.codes == ("reject",)
    assert "roots" in out.verdict.detail["reject"]["replay_sample"] and "roots" not in out.verdict.detail["reject"]["checks"]
    assert stream.records[0]["replay_sample"]["audit"]["checks"]["roots"]["detail"]["ports"] == [port]


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
