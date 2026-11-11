"""The proof system over a window of templates (a TP rank's served window): one `Stream` whose check is each template's
digest and `Q_template_instances` in the window's order, each registration's population the templates' mixed population
at the challenger's own counts (`register(counts=)`), sampled once by `stratified:K`, a stratum per template each sampled at
least once, and its whole run held to the Lean verifier's (`Run.run`) as one template's is."""

from __future__ import annotations

import pytest

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import array_of, coll_concat
from verity.circuits.types import Array, Tuple, Value as Bits
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import draw as D, partition as P, registration as R, verifier as V

from test_challenger_coins import assert_only_honest_coins_accepted, coin_outcomes
from test_remote import signer  # noqa: F401  (the challenger key's signer fixture)
from test_service import DELTA, PAIR, RECORD_ITEMS, RUN, SCHEMA, SESSION, VALUES, XOR8, Clock, _lean, _refused

import fake_challenger as FC

TRIPLE = bind(CompositeDefinition("ServiceTestTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                  lambda B, S_, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
WINDOW = (PAIR, TRIPLE)
QUERY = P.template_instances_query(WINDOW)
#: PAIR's one instance and TRIPLE's nine, strata `stratified-rate:1/3` samples ⌈1/3⌉ = 1 and ⌈9/3⌉ = 3 of.
COUNTS = {P.template_id(PAIR): 1, P.template_id(TRIPLE): 9}
LAW = S.Law("stratified-rate:1/3")
PUBLIC = (*RECORD_ITEMS, *(S.Public(f"population:{P.template_id(t)}", "structure", "a template's stratum") for t in WINDOW))


def _check(program=None, query=QUERY, tied=()) -> S.Check:
    return S.Check(tuple(P.program_digest(Program(t)) for t in WINDOW) if program is None else program, query,
                   proof_deadline=60.0, tied=tied)


def _spec(check=None, selection=LAW) -> S.Spec:
    return S.Spec(VALUES, selection, check or _check(), PUBLIC, b"service-window-test")


def _stream(spec=None, coins=None, window=WINDOW, sampler=None) -> S.Stream:
    return FC.Stream(spec or _spec(), window, budget=2, delta=DELTA, challenger="challenger-test", coins=coins,
                     clock=Clock(), sampler=sampler)


def _program(counts=COUNTS, window=WINDOW, values=()):
    return P.mixed_population_program([(t, counts[P.template_id(t)]) for t in window], values=values)


def _gathered(counts=COUNTS) -> Program:
    """The window in another form than its templates' mixed population, as the network certifier's scan's generator form
    is: PAIR's instances read their bytes from one input they share, instance j rows j and j + 1 of ``t``."""
    n0, n1 = counts[P.template_id(PAIR)], counts[P.template_id(TRIPLE)]

    def signature(S_):
        return ((("t", Array(n0 + 1, Bits(8))), ("1.x", Array(n1, Array(3, Bits(8))))),
                Tuple(Array(n0, Bits(8)), Array(n1, Bits(8))))

    def body(B, S_, t, x):
        pairs = array_of([array_of([t[j], t[j + 1]]) for j in range(n0)])
        return coll_concat(signature(S_)[1], [B.batch(PAIR, pairs, axes=(0,)), B.batch(TRIPLE, x, axes=(0,))])
    return Program(bind(CompositeDefinition("ServiceTestGathered", 0, ("counts",), signature, body, register=False),
                        counts=(n0, n1)))


def _ports(counts=COUNTS, values=(), program=None) -> dict:
    part = P.partition(program or _program(counts, values=values), QUERY)
    n = sum(counts.values())
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, RUN)}
            for p, leaves in (("x", n), ("out", 2 * n))}


def _named(program: Program, query=QUERY) -> tuple[dict, dict]:
    """The sampling design `LAW` names over `program`'s population under `query`, and the work table it is expanded over."""
    law = S.selection_law(LAW, P.units(program, query), S.selection_sizes(LAW, program, query))
    return law, S.selection_table(LAW, program, query, None)


def _record(counts=COUNTS, program=None, query=QUERY, law=None, values=(), form=None, **kw) -> dict:
    """A registration of the window's mixed population, or of `program` with the mixed population's ports; of `form`
    with its own."""
    ports = _ports(counts, values, form)
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **ports[p]} for i, p in enumerate(ports)]
    program = form or program or _program(counts, values=values)
    named, table = _named(program, query)
    return R.record(program, query, law or named, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": 0},
                    work_table=None if law else table, **kw)


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
    assert FC.Stream(_spec(one), [PAIR], budget=2, delta=DELTA, challenger="c").templates == (PAIR,)
    assert _check().to_json()["program"] == [P.program_digest(Program(t)) for t in WINDOW]


def test_a_two_template_window_runs_to_an_accepted_outcome_by_the_lean_run():
    """A window of PAIR x 1 and TRIPLE x 9 under `stratified-rate:1/3`: registered at the challenger's own counts, sampled once
    from its coins by the stratified law at the K its strata need (`draw.strat_k`, 3), each stratum sampled its k (a third of
    its proof blocks, rounded up), and run to an accepted outcome, the
    verifier's whole run (`Run.run`, over the window's mixed population) agreeing."""
    stream = _stream()
    rec = _record()
    reg = stream.register(rec, _ports(), counts=COUNTS)
    law = stream._entries[reg.order].law
    assert [(s["template"], s["k"]) for s in law["strata"]] == [(P.template_id(PAIR), 1), (P.template_id(TRIPLE), 3)]
    draw = stream.select(reg)
    assert draw.draw["law"] == "stratified" and draw.draw["k"] == 3 and len(draw.units) == 4 and 0 in draw.units
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


def test_a_window_in_another_form_runs_on_the_challengers_own_copy_of_it():
    """A window registered in another form than its templates' mixed population (`_gathered`, as the network certifier's
    scan registers its generator form) runs to an accepted outcome by the Lean run on a stream given the challenger's
    own copy of that form (`Stream(program=)`), whose digest names it; a stream without it builds the mixed population and
    refuses the registration (`R2-program`, `R2-partition`), and the stream with it refuses a registration of the mixed
    population (`R2-program`) and counts that aren't its form's ranges (`population`). A form is a window's, of its
    templates, one range each, in its order: refused with one template, values or another order."""
    gathered = _gathered()
    rec, ports = _record(form=gathered), _ports(program=gathered)
    assert {"R2-program", "R2-partition"} <= set(_refused(lambda: _stream().register(rec, ports, counts=COUNTS)))
    stream = FC.Stream(_spec(), WINDOW, budget=2, delta=DELTA, challenger="challenger-test", clock=Clock(),
                       program=gathered)
    assert stream.digest != _stream().digest
    assert "R2-program" in _refused(lambda: stream.register(_record(), _ports(), counts=COUNTS))
    pair, triple = P.template_id(PAIR), P.template_id(TRIPLE)
    assert _refused(lambda: stream.register(rec, ports, counts={pair: 2, triple: 8})) == ("population",)
    reg = stream.register(rec, ports, counts=COUNTS)
    assert stream._entries[reg.order].program is gathered
    draw = stream.select(reg)
    assert len(draw.units) == 4 and 0 in draw.units and V.check_draw(draw.draw, rec, stream._entries[reg.order].law).ok
    assert stream.deliver(draw, [SESSION]) is None
    out = stream.outcome(draw, _lean(), {"statement": "stub"})
    assert out.accepted and "run" not in stream.records[-1]["audit"]["checks"]
    from test_service import _spec as pair_spec
    for unit, spec, kw, why in ((PAIR, pair_spec(), {}, "takes no values"),
                                (WINDOW, _spec(), {"values": (VALUE,)}, "takes no values"),
                                (WINDOW, _spec(), {"program": P.mixed_population_program([(TRIPLE, 9), (PAIR, 1)])},
                                 "one range a template")):
        with pytest.raises(ValueError, match=why):
            FC.Stream(spec, unit, budget=2, delta=DELTA, challenger="c", **{"program": gathered, **kw})


def test_a_tied_window_runs_to_an_accepted_outcome_by_the_lean_run():
    """The gathered form at two PAIR instances: instance 0 reads rows 0 and 1 of ``t`` and instance 1 rows 1 and 2, so
    two proof blocks read ``t``, a run Input, which R8 refuses unless the check names it tied. Named tied
    (`Check(tied=)`), the registration is let through on both verifiers and publishes ``t``, and the outcome is
    accepted with the whole run the Lean's (`Run.run`, which reads the tied names from its params). Named nothing, or
    the other parameter instead, both verifiers refuse the registration (``R8-shared-input``)."""
    counts = {P.template_id(PAIR): 2, P.template_id(TRIPLE): 8}
    gathered = _gathered(counts)
    rec, ports = _record(counts, form=gathered), _ports(counts, program=gathered)

    def stream(tied):
        return FC.Stream(_spec(_check(tied=tied)), WINDOW, budget=2, delta=DELTA, challenger="challenger-test",
                         clock=Clock(), program=gathered)

    tied = stream(("t",))
    reg = tied.register(rec, ports, counts=counts)
    assert reg.rests_on_period_check == ("t",)
    draw = tied.select(reg)
    assert tied.deliver(draw, [SESSION]) is None
    out = tied.outcome(draw, _lean(), {"statement": "stub"})
    assert out.accepted and "run" not in tied.records[-1]["audit"]["checks"] and tied._entries[reg.order].read
    for names in ((), ("1.x",)):
        with pytest.raises(S.Refused) as e:
            stream(names).register(rec, ports, counts=counts)
        assert e.value.verdict.codes == ("R8-shared-input",), names
        assert e.value.verdict.verifiers["lean"]["codes"] == ["R8-shared-input"], names


def test_a_stratum_short_of_its_floor_is_refused():
    """A record whose sampling design samples PAIR's stratum none (the stratified law samples each at least one) is refused
    (`R1-law`), and a sample short of a stratum, from a sampler that samples only TRIPLE's proof blocks, is refused at the
    select (`check_draw`), before any proof."""
    rec = _record()
    short = {**rec, "law": {**rec["law"], "strata": [{**rec["law"]["strata"][0], "k": 0}, *rec["law"]["strata"][1:]]}}
    assert "R1-law" in _refused(lambda: _stream().register(short, _ports(), counts=COUNTS))
    law = D.derive(_program(), QUERY, *_named(_program()))
    assert not V.check_draw({**law, "population": 10, "units": [1, 2, 3, 4]}, rec, law).ok
    stream = _stream(sampler=lambda law, n: [1, 2, 3, 4])
    reg = stream.register(rec, _ports(), counts=COUNTS)
    assert _refused(lambda: stream.select(reg))


#: a value the window's mixed population takes ahead of its templates' parameters: 8 leaves of 16 bits, read as rows of 4
VALUE = ("t", 8, 16)
VALUED = S.Spec((*VALUES, S.Value("t", None, "serve", registered=True)), LAW, _check(),
                (*PUBLIC, *(S.Public(f"{x}:t", "structure", "the value's layout") for x in ("leaves", "words", "domain"))),
                b"service-window-test")


def _earlier(row: int, registered=(), roots=()) -> dict:
    """A record of the prover's log before the stream: a registration of another Program (PAIR's two instances), in
    window ``row``."""
    prog, q = P.population_program(PAIR, 2), P.template_instance_query(PAIR)
    return R.record(prog, q, {"law": "subset", "k": 1}, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": row},
                    registered=registered)


def _reusing(log, entry, outputs):
    """A window's stream that takes `VALUE` with ``log`` before it, and its registration of the value at ``entry``
    with ``outputs``, chained after the log's last record: (the stream, the record, its call)."""
    stream = FC.Stream(VALUED, WINDOW, budget=2, delta=DELTA, challenger="challenger-test", clock=Clock(),
                       values=(VALUE,), log=log)
    rec = _record(values=(VALUE,), previous=R.digest(log[-1]) if log else None, registered=[{**entry, "value": "t"}])
    return stream, rec, lambda r=rec: stream.register(r, _ports(values=(VALUE,)), counts=COUNTS, registered={"t": 4},
                                                       outputs=outputs)


def test_a_window_reusing_an_earlier_records_value_runs_to_an_accepted_outcome_by_the_lean_run():
    """A window whose mixed population takes a value (`Stream(values=)`) that the prover's record before the stream
    registered first (`Stream(log=)`: a registration of another Program, the log's first record, never sampled): the
    window's registration registers it as that record's value (`register(outputs=)`: R7 takes its domain and root),
    chained right after it (R6), and is sampled and run to an accepted outcome, its session reading the value at that
    root, the Lean's whole run agreeing. Refused:
    the value at another root (``R7-held``), a registration chained after nothing (``R6-previous``), and the same
    registration on a stream with no log before it (``R6-previous``, ``R7-output``, ``R7-domain``)."""
    entry = {"value": "w/t", "root": "5a" * 64, "leaves": 2, "words": 4, "domain": "6b" * 64}
    w = _earlier(-1, registered=[entry])
    outputs = {"t": {"record": R.digest(w), "value": "w/t"}}
    stream, honest, register = _reusing([w], entry, outputs)
    reg = register()
    assert stream.log == (w, honest) and reg.order == 0 and reg.receipt["received"] == "1"
    assert stream.values == (VALUE,)
    draw = stream.select(reg)
    assert stream.deliver(draw, [SESSION]) is None
    out = stream.outcome(draw, _lean(registered={"0": {"value": "t", "root": entry["root"]}}), {"statement": "stub"})
    assert out.accepted and "run" not in stream.records[-1]["audit"]["checks"] and stream._entries[0].read
    assert stream.digest != _stream().digest
    _s, _r, register = _reusing([w], {**entry, "root": "00" * 64}, outputs)
    assert _refused(register) == ("R7-held",)
    _s, _r, register = _reusing([w], entry, outputs)
    assert _refused(lambda: register({**honest, "previous": None})) == ("R6-previous",)
    _s, _r, register = _reusing([], entry, outputs)
    assert set(_refused(lambda: register(honest))) == {"R6-previous", "R7-output", "R7-domain"}


def test_a_registration_reads_the_earlier_record_its_outputs_name_whatever_the_log():
    """`_earlier` gives a registration's checks the record each of its outputs names, here an earlier run's port, even
    where that record is neither the log's last, nor of the registration's window, nor any value's first registration:
    the verdict is the whole log's, accepted. A stream that kept no record's digest reads only the last, and refuses
    (``R7-output``, and ``R7-domain`` with no output to take the domain of)."""
    part = P.partition(P.population_program(PAIR, 2), P.template_instance_query(PAIR))
    port = {"port": "o", "root": "8d" * 64, "leaves": 2,
            "domain": R.window_domain(part["program"], P.digest(part), "o", SCHEMA, 2, RUN)}
    w, x = _earlier(-2, roots=[port]), _earlier(-1)
    outputs = {"t": {"record": R.digest(w), "port": "o"}}
    entry = {"value": "t", "root": port["root"], "leaves": 2, "words": 4, "domain": port["domain"]}
    stream, rec, register = _reusing([w, x], entry, outputs)
    assert stream._earlier(rec, None, outputs) == [w, x] and stream._earlier(rec) == [x]
    law, table = _named(_program(values=(VALUE,)))
    args = (rec, _program(values=(VALUE,)), QUERY, law, S.SCHEME, _ports(values=(VALUE,)), None, ())
    few = V.check(*args, stream._earlier(rec, None, outputs), None, table, {"t": 4}, outputs=outputs)
    assert few == V.check(*args, stream.log, None, table, {"t": 4}, outputs=outputs) and few.ok
    assert register().order == 0
    stream, rec, register = _reusing([w, x], entry, outputs)
    stream._digests.clear()
    assert set(_refused(register)) == {"R7-output", "R7-domain"}


def test_a_streams_values_and_earlier_log_are_refused_in_another_form():
    """Values are a window's (a mixed population of its templates, each value a new name of at least one leaf), and the
    earlier log is JSON records whose registered entries name their value, domain and root."""
    from test_service import _spec as pair_spec
    with pytest.raises(ValueError, match="one template's instances takes no values"):
        FC.Stream(pair_spec(), PAIR, budget=2, delta=DELTA, challenger="c", values=(VALUE,))
    for values in ((("t", 0, 16),), (("x", 8, 16), ("x", 8, 16)), (("0.x", 8, 16),)):
        with pytest.raises(ValueError, match="each \\(name, leaves, width\\)"):
            FC.Stream(_spec(), WINDOW, budget=2, delta=DELTA, challenger="c", values=values)
    for log in ([["not", "a", "record"]], [{"registered": [{"value": "t"}]}], [{"window": {1, 2}}]):
        with pytest.raises(ValueError, match="earlier log is the prover's records"):
            FC.Stream(_spec(), WINDOW, budget=2, delta=DELTA, challenger="c", log=log)


def test_a_windows_draw_from_the_challengers_coins_is_held_to_its_evidence_by_the_lean_run(
        tmp_path, signer, monkeypatch):  # noqa: F811
    """A window's stream on the auditor's challenger's coins: the honest sample's outcome is accepted by the Lean run,
    and each of `coin_outcomes`' negatives is a reject the Lean run refused at its select (`coins`)."""
    def register(coins, i, window_root):
        stream = _stream(coins=coins)
        counts = {P.template_id(PAIR): 1, P.template_id(TRIPLE): 9 + i}
        return stream, stream.register(_record(counts), _ports(counts), counts=counts, window_root=window_root)

    outcomes, _ = coin_outcomes(tmp_path, signer, monkeypatch, register, positions=64)
    assert_only_honest_coins_accepted(outcomes)
    assert outcomes["honest"][1].accepted
