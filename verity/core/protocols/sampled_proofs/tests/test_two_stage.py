"""A two-stage sampling design on the proof system's `Stream`: the boundary registration over replay blocks, the first
stage, the interior registration over the replay sample's proof blocks, the second stage on its own coins, one outcome
binding both registrations and both samples, each step by both verifiers; replay blocks of one template and of two,
whose interior is template-major and whose second stage samples each cell; its refusals; and its profile at the
footprint."""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb

import pytest

from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value as Bits
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


def _boundary(n=N, row=0, previous=None) -> tuple[dict, dict]:
    program = P.population_program(BLOCK, n)
    return (_record(program, GRAIN, BOUNDARY, n, {"law": "subset", "k": Q}, {"kind": "served", "run": RUN, "row": row},
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
    assert stream.cells == (N_RHO,) and stream.counts == (K,)
    reg = stream.register(*_boundary())
    assert reg.record["population"] == N and reg.record["query"] == R.query_digest(GRAIN)
    first = stream.select(reg)
    assert first.draw["law"] == "subset" and len(first.units) == Q and first.replay_sample is None
    ireg = stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest))
    assert ireg.order == reg.order and ireg.receipt["received"] == "1" and stream.log[-1] == ireg.record
    second = stream.select(ireg)
    assert second.replay_sample == first and second.law == S.second_law(first.units, (N_RHO,), (K,))
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
    for k in (N_RHO, N_RHO + 1, {PAIR.id: N_RHO}):
        with pytest.raises(ValueError, match="whole replay blocks"):
            _stream(spec=_spec(S.TwoStage(GRAIN, f"subset:{Q}", k)))
    with pytest.raises(ValueError, match="name each of the check's templates once"):
        _stream(spec=_spec(S.TwoStage(GRAIN, f"subset:{Q}", {BLOCK.id: 1})))
    with pytest.raises(ValueError, match="one round"):
        _stream(rounds=2)
    with pytest.raises(ValueError, match="only a two-stage stream"):
        FC.Stream(S.Spec(VALUES[:2], S.Law("subset:2"), _spec().check, PUBLIC, b"t"), PAIR, replay=BLOCK, budget=2,
                  delta=DELTA, challenger="challenger-test", clock=Clock())


def test_the_rule_and_the_spec_state_one_form():
    assert RULE.to_json() == {"rule": "two-stage", "grain": GRAIN, "replay": f"subset:{Q}", "k": K}
    assert S.TwoStage(GRAIN, "rate:1/2", 2).replay == "rate:1/2"
    assert S.TwoStage(GRAIN, "subset:2", {PAIR.id: 2}).to_json()["k"] == {PAIR.id: 2}
    for bad, match in (((dict(GRAIN, version=1), "subset:2", 1), "grain"), ((GRAIN, "stratified:2", 1), "replay law"),
                       ((GRAIN, "subset:2", 0), "k >= 1"), ((GRAIN, "subset:2", True), "k >= 1"),
                       ((GRAIN, "subset:2", {}), "k >= 1"), ((GRAIN, "subset:2", {PAIR.id: 0}), "k >= 1"),
                       ((GRAIN, "subset:2", {PAIR.id: True}), "k >= 1")):
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
    for ns, ks, cell in (((N_RHO,), (K,), lambda u: u // N_RHO), ((2, 1), (1, 1), lambda u: (u // 2, 0) if u < 4 else
                                                                   (u - 4, 1))):
        source = random.Random(2).randbytes(4096)
        drawn = V.lean("select_second", {"sample": [0, 2], "ns": list(ns), "ks": list(ks), "source": source.hex()})
        read = iter(source)
        assert drawn["draw"] == S.select_second([0, 2], ns, ks, lambda m: bytes(next(read) for _ in range(m)))
        assert sorted({cell(u) for u in drawn["draw"]["units"]}) == sorted({cell(u) for u in range(2 * sum(ns))})


def test_the_second_stage_samples_k_in_each_cell():
    """One stratum per cell, template by template and in each by sampled replay block, tiling the interior in order,
    each of the work law's floor its count (cells' counts aren't in proportion to their sizes, as the stratified law's
    are); a cell smaller than the rule's count is proved whole, and counts proving every cell whole are refused."""
    law = S.second_law([1, 3], (4,), (2,))
    assert law == {"law": "work", "k": 1,
                   "strata": [{"template": "1/0", "units": [[0, 4]], "work": 1, "floor": 2, "k": 2},
                              {"template": "3/0", "units": [[4, 8]], "work": 1, "floor": 2, "k": 2}]}
    assert D.strata_problem(law, 8) is None
    obj = S.select_second([1, 3], (4,), (2,), random.Random(5).randbytes)
    assert obj["population"] == 8 and len(obj["units"]) == 4
    assert sorted(u // 4 for u in obj["units"]) == [0, 0, 1, 1]
    law = S.second_law([1, 3], (3, 1), (2, 1))
    assert [(s["template"], s["units"], s["k"]) for s in law["strata"]] == [
        ("1/0", [[0, 3]], 2), ("3/0", [[3, 6]], 2), ("1/1", [[6, 7]], 1), ("3/1", [[7, 8]], 1)]
    assert D.strata_problem(law, 8) is None
    obj = S.select_second([1, 3], (3, 1), (2, 1), random.Random(6).randbytes)
    assert obj["population"] == 8 and {6, 7} <= set(obj["units"]) and len(obj["units"]) == 6
    for ns, ks in (((10, 1), (1, 1)), ((1, 9, 2), (1, 3, 2)), ((5, 5), (1, 4))):
        law = S.second_law([0, 2, 5], ns, ks)
        assert D.strata_problem(law, 3 * sum(ns)) is None
        stratified = {"law": "stratified", "k": 3 * sum(ks),
                      "strata": [{v: s[v] for v in ("template", "units", "k")} for s in law["strata"]]}
        assert "k is not the rule's" in D.strata_problem(stratified, 3 * sum(ns))
        obj = S.select_second([0, 2, 5], ns, ks, random.Random(7).randbytes)
        assert len(obj["units"]) == 3 * sum(ks)
    assert S.second_counts(2, ["a", "b"], (3, 1)) == (2, 1) and S.second_counts({"b": 1, "a": 3}, ["a", "b"], (3, 2)) == (3, 1)
    with pytest.raises(ValueError, match="name each"):
        S.second_counts({"a": 1}, ["a", "b"], (3, 2))
    for sample, ns, ks, match in (([], (4,), (1,), "ascending"), ([2, 1], (4,), (1,), "ascending"),
                                  ([0], (1,), (1,), "'law'"), ([0], (), (), "one or more templates'"),
                                  ([0], (2, 0), (1, 1), "at least one proof block"), ([0], (2, 1), (1,), "a count each"),
                                  ([0], (4,), (4,), "whole replay blocks"), ([0], (3, 1), (3, 1), "whole replay blocks"),
                                  ([0], (4,), (0,), "1 to n_t"), ([0], (4, 2), (1, 3), "1 to n_t")):
        with pytest.raises(ValueError, match=match):
            S.select_second(sample, ns, ks, lambda m: bytes(m))


TRIPLE = bind(CompositeDefinition("TwoStageTestTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                  lambda B, S_, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
#: A replay block of two templates: two instances of PAIR, then one of TRIPLE.
MIXED = bind(CompositeDefinition("TwoStageTestMixed", 0, (),
                                 lambda S_: ((("x", Array(2, Array(2, Bits(8)))), ("y", Array(3, Bits(8)))),
                                             Tuple(Array(2, Bits(8)), Bits(8))),
                                 lambda B, S_, x, y: tuple_of(B.batch(PAIR, x, axes=(0,)), B.call(TRIPLE, y)),
                                 register=False))
WINDOW = (PAIR, TRIPLE)
WINDOW_QUERY = P.template_instances_query(WINDOW)
MIXED_GRAIN = P.template_instance_query(MIXED)


def _window_spec(k=K, query=WINDOW_QUERY, window=WINDOW) -> S.Spec:
    return S.Spec(VALUES, S.TwoStage(MIXED_GRAIN, f"subset:{Q}", k),
                  S.Check(tuple(P.program_digest(Program(t)) for t in window), query, proof_deadline=60.0), PUBLIC,
                  b"two-stage-window-test")


def _window_stream(spec=None, window=WINDOW, replay=MIXED) -> S.Stream:
    return FC.Stream(spec or _window_spec(), window, replay=replay, budget=2, delta=DELTA, challenger="challenger-test",
                     clock=Clock())


def _window_interior(program: Program, previous: str) -> tuple[dict, dict]:
    n = P.units(program, WINDOW_QUERY)
    return (_record(program, WINDOW_QUERY, INTERIOR, n, {"law": "subset", "k": n},
                    {"kind": "served", "run": RUN, "row": 0, "interior": True}, previous),
            _ports(program, WINDOW_QUERY, INTERIOR, n))


def test_a_replay_block_of_two_templates_runs_with_a_template_major_interior():
    """A replay block of PAIR x 2 and TRIPLE x 1: the boundary registration over N of them, the interior over the
    sample's Q·3 proof blocks template-major (PAIR's Q·2, then TRIPLE's Q), the second stage one stratum per cell, k of
    each PAIR cell and TRIPLE's one-unit cells whole, and an accepted outcome whose whole run is the Lean verifier's; its
    profile is `Subset(min k_t)` over T·n_max proof blocks a replay block, which bounds the cells' escape."""
    stream = _window_stream()
    assert stream.cells == (2, 1) and stream.counts == (1, 1) and stream.n_rho == 3
    boundary = P.population_program(MIXED, N)
    reg = stream.register(_record(boundary, MIXED_GRAIN, BOUNDARY, N, {"law": "subset", "k": Q},
                                  {"kind": "served", "run": RUN, "row": 0}, None),
                          _ports(boundary, MIXED_GRAIN, BOUNDARY, N))
    first = stream.select(reg)
    interior = P.mixed_population_program([(PAIR, 2 * Q), (TRIPLE, Q)])
    assert stream.interior_program(Q).root.id == interior.root.id
    ireg = stream.register_interiors(first, *_window_interior(interior, reg.digest))
    second = stream.select(ireg)
    assert second.law == S.second_law(first.units, (2, 1), (1, 1))
    assert [s["units"] for s in second.law["strata"]] == [[[0, 2]], [[2, 4]], [[4, 5]], [[5, 6]]]
    assert sorted(u // 2 for u in second.units if u < 4) == [0, 1] and {4, 5} <= set(second.units)
    assert stream.deliver(second, [SESSION]) is None
    out = stream.outcome(second, _lean(), {"statement": "stub"}, session_roots={**_roots(BOUNDARY), **_roots(INTERIOR)})
    assert out.accepted and "run" not in stream.records[0]["audit"]["checks"]
    p = out.profile
    assert p.sizes == (N, 4) and p.draws["unit"] == Subset(1) and dict(p.drawn) == {S.REPLAY: Q, "unit": Q * 2}
    for w in ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)):
        cells = Fraction(comb(2 - w[0], 1), comb(2, 1)) * Fraction(comb(1 - w[1], 1), comb(1, 1))
        assert cells <= Fraction(comb(4 - sum(w), 1), comb(4, 1))


def test_a_replay_block_of_two_templates_takes_a_count_per_template():
    """`k` per template: k_t of each cell of template t, capped at the cell's size; counts naming another template are
    refused, and so are counts proving every cell whole."""
    ids = [P.template_id(t) for t in WINDOW]
    stream = _window_stream(_window_spec({ids[0]: 1, ids[1]: 5}))
    assert stream.counts == (1, 1) and stream.spec.selection.to_json()["k"] == {ids[0]: 1, ids[1]: 5}
    with pytest.raises(ValueError, match="whole replay blocks"):
        _window_stream(_window_spec({ids[0]: 2, ids[1]: 1}))
    with pytest.raises(ValueError, match="name each of the check's templates once"):
        _window_stream(_window_spec({ids[0]: 1}))


def test_a_replay_block_of_two_templates_fits_the_check_in_order():
    """The check names the replay block's templates in the order of their first instance in it, each once: a check over
    them in the other order, or over a template the replay block lacks, is refused; and an interior registered in replay
    block order, not template-major, is refused against the challenger's own interior program."""
    with pytest.raises(ValueError, match="order of its first instance"):
        _window_stream(_window_spec(query=P.template_instances_query(WINDOW[::-1]), window=WINDOW[::-1]),
                       window=WINDOW[::-1])
    with pytest.raises(ValueError, match="doesn't apply to the replay block"):
        S.replay_cells(Program(BLOCK), WINDOW_QUERY)
    stream = _window_stream()
    boundary = P.population_program(MIXED, N)
    reg = stream.register(_record(boundary, MIXED_GRAIN, BOUNDARY, N, {"law": "subset", "k": Q},
                                  {"kind": "served", "run": RUN, "row": 0}, None),
                          _ports(boundary, MIXED_GRAIN, BOUNDARY, N))
    first = stream.select(reg)
    blocks = P.mixed_population_program([(PAIR, 2), (TRIPLE, 1)] * Q)
    assert "R2-program" in _refused(lambda: stream.register_interiors(first, *_window_interior(blocks, reg.digest)))
    assert not P.in_order(blocks, WINDOW_QUERY, 3 * Q)
