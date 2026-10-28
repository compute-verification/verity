"""The proof system over a window of templates (a TP rank's served window): one `Stream` whose check is each template's
digest and `Q_template_instances` in the window's order, each registration's population the templates' mixed population
at the challenger's own counts (`register(counts=)`), drawn once by `stratified:K`, a stratum per template each drawn at
least once, and its whole run held to the Lean verifier of record's (`Run.run`) as one template's is."""

from __future__ import annotations

import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R, verifier as V

from test_challenger_coins import assert_only_honest_coins_accepted, coin_outcomes
from test_remote import signer  # noqa: F401  (the challenger key's signer fixture)
from test_service import DELTA, PAIR, RUN, SCHEMA, SESSION, STEP0, VALUES, XOR8, Clock, _lean, _refused

TRIPLE = bind(CompositeDefinition("ServiceTestTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                  lambda B, S_, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
WINDOW = (PAIR, TRIPLE)
QUERY = P.template_instances_query(WINDOW)
#: PAIR's one instance is a stratum `stratified:3` draws round(3 / 10) = 0 of, so its floor draws it.
COUNTS = {P.template_id(PAIR): 1, P.template_id(TRIPLE): 9}
LAW = S.Law("stratified:3")
PUBLIC = (*STEP0, *(S.Public(f"population:{P.template_id(t)}", "structure", "a template's stratum") for t in WINDOW))


def _check(program=None, query=QUERY) -> S.Check:
    return S.Check(tuple(P.program_digest(Program(t)) for t in WINDOW) if program is None else program, query,
                   proof_deadline=60.0)


def _spec(check=None, selection=LAW) -> S.Spec:
    return S.Spec(VALUES, selection, check or _check(), PUBLIC, b"service-window-test")


def _stream(spec=None, coins=b"\x07", window=WINDOW, sampler=None) -> S.Stream:
    return S.Stream(spec or _spec(), window, budget=2, delta=DELTA, challenger="challenger-test",
                    coins=(lambda n: coins * n) if isinstance(coins, bytes) else coins, clock=Clock(), sampler=sampler)


def _program(counts=COUNTS, window=WINDOW):
    return P.mixed_population_program([(t, counts[P.template_id(t)]) for t in window])


def _ports(counts=COUNTS) -> dict:
    part = P.partition(_program(counts), QUERY)
    n = sum(counts.values())
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, RUN)}
            for p, leaves in (("x", n), ("out", 2 * n))}


def _record(counts=COUNTS, program=None, query=QUERY, law=None) -> dict:
    ports = _ports(counts)
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **ports[p]} for i, p in enumerate(ports)]
    return R.record(program or _program(counts), query, law or {"law": "stratified", "k": 3}, S.SCHEME, roots,
                    {"kind": "served", "run": RUN, "row": 0})


def test_a_windows_stream_takes_its_templates_digests_and_query_in_its_order():
    """A window's check names each template's digest, in the window's order, under `Q_template_instances` over them in
    the same order; a stream over the templates in another order, over one of them, or a check of one digest or a
    single template's query, is refused, as is a check of no digests or a digest twice."""
    assert _stream().templates == WINDOW
    for build in (lambda: _stream(window=WINDOW[::-1]), lambda: _stream(window=WINDOW[:1]),
                  lambda: _stream(_spec(_check(query=P.template_instances_query(WINDOW[::-1])))),
                  lambda: _stream(_spec(_check(query=P.template_instance_query(PAIR)))),
                  lambda: S.Stream(_spec(), PAIR, budget=2, delta=DELTA, challenger="c")):
        with pytest.raises(ValueError, match="challenger's unit|check's query"):
            build()
    for program in ((), (P.program_digest(Program(PAIR)),) * 2, ("AB" * 64,)):
        with pytest.raises(ValueError, match="window's"):
            _check(program)
    one = S.Check((P.program_digest(Program(PAIR)),), P.template_instances_query([PAIR]), proof_deadline=60.0)
    assert S.Stream(_spec(one), [PAIR], budget=2, delta=DELTA, challenger="c").templates == (PAIR,)
    assert _check().to_json()["program"] == [P.program_digest(Program(t)) for t in WINDOW]


def test_a_two_template_window_runs_to_an_accepted_outcome_by_the_lean_run():
    """A window of PAIR x 1 and TRIPLE x 9 under `stratified:3`: registered at the challenger's own counts, sampled once
    from its coins with each stratum drawn its k (PAIR's by its floor of one), and run to an accepted outcome, the
    verifier of record's whole run (`Run.run`, over the window's mixed population) agreeing."""
    stream = _stream()
    rec = _record()
    reg = stream.register(rec, _ports(), counts=COUNTS)
    law = stream._entries[reg.order].law
    assert [(s["template"], s["k"]) for s in law["strata"]] == [(P.template_id(PAIR), 1), (P.template_id(TRIPLE), 3)]
    draw = stream.select(reg)
    assert draw.draw["law"] == "stratified" and len(draw.units) == 4 and 0 in draw.units
    assert V.check_draw(draw.draw, rec, law).ok
    assert stream.deliver(draw, [SESSION]) is None
    out = stream.outcome(draw, _lean(), {"statement": "stub"})
    audit = stream.records[-1]["audit"]
    assert out.accepted and "run" not in audit["checks"] and stream._entries[reg.order].read


def test_a_registration_of_another_window_is_refused():
    """The challenger builds the window's program from its own counts, so a record over another split of the same N
    (`R2-program`), over the templates in the other order (`query`), or over one template's population, is refused, and
    so are counts that miss a template, name another, give one none or a boolean, number other than N, or come to a
    single proof block's stream (`population`)."""
    stream = _stream()
    other = {P.template_id(PAIR): 2, P.template_id(TRIPLE): 8}
    assert "R2-program" in _refused(lambda: stream.register(_record(other), _ports(), counts=COUNTS))
    assert "R2-program" in _refused(lambda: stream.register(_record(), _ports(), counts=other))
    flipped = P.mixed_population_program([(TRIPLE, 9), (PAIR, 1)])
    assert "query" in _refused(lambda: stream.register(
        _record(program=flipped, query=P.template_instances_query(WINDOW[::-1])), _ports(), counts=COUNTS))
    single = P.population_program(PAIR, 10)
    assert _refused(lambda: stream.register(_record(program=single, query=P.template_instance_query(PAIR)), _ports(),
                                            counts=COUNTS))
    pair, triple = P.template_id(PAIR), P.template_id(TRIPLE)
    for counts in (None, {pair: 10}, {pair: 1, triple: 9, "Other_v0": 1}, {pair: 0, triple: 10},
                   {pair: True, triple: 9}, {pair: 1, triple: 8}, [(pair, 1), (triple, 9)]):
        assert _refused(lambda: stream.register(_record(), _ports(), counts=counts)) == ("population",), counts
    from test_service import _ports as pair_ports, _record as pair_record, _stream as pair_stream
    assert _refused(lambda: pair_stream().register(pair_record(), pair_ports(6), counts={pair: 6})) == ("population",)
    reg = stream.register(_record(), _ports(), counts=COUNTS)
    assert reg.order == 0


def test_a_stratum_short_of_its_floor_is_refused():
    """A record whose sampling design samples PAIR's stratum none (below its floor of one) is refused (`R1-law`), and a sample short of
    a stratum, from a sampler that samples only TRIPLE's proof blocks, is refused at the select (`check_draw`), before any proof."""
    rec = _record()
    short = {**rec, "law": {**rec["law"], "strata": [{**rec["law"]["strata"][0], "k": 0}, *rec["law"]["strata"][1:]]}}
    assert "R1-law" in _refused(lambda: _stream().register(short, _ports(), counts=COUNTS))
    law = D.derive(_program(), QUERY, {"law": "stratified", "k": 3})
    assert not V.check_draw({"law": "stratified", "population": 10, "k": 3, "strata": law["strata"],
                             "units": [1, 2, 3, 4]}, rec, law).ok
    stream = _stream(sampler=lambda law, n: [1, 2, 3, 4])
    reg = stream.register(rec, _ports(), counts=COUNTS)
    assert _refused(lambda: stream.select(reg))


def test_a_windows_draw_from_the_challengers_coins_is_held_to_its_evidence_by_the_lean_run(
        tmp_path, signer, monkeypatch):  # noqa: F811
    """A window's stream on the auditor's challenger's coins: the honest draw's outcome is accepted by the Lean run,
    and each of `coin_outcomes`' negatives is a reject the Lean run refused at its select (`coins`)."""
    def register(coins, i, window_root):
        stream = _stream(coins=coins)
        counts = {P.template_id(PAIR): 1, P.template_id(TRIPLE): 9 + i}
        return stream, stream.register(_record(counts), _ports(counts), counts=counts, window_root=window_root)

    outcomes, _ = coin_outcomes(tmp_path, signer, monkeypatch, register, positions=64)
    assert_only_honest_coins_accepted(outcomes)
    assert outcomes["honest"][1].accepted
