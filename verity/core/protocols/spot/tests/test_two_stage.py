"""A two-stage sampling design on the proof system's `Stream`: the boundary registration over replay blocks, the first
stage, the interior registration over the replay sample's proof blocks, the second stage on its own coins, one outcome
binding both registrations and both samples, each step by both verifiers; replay blocks of one template and of two,
whose interior is template-major and whose second stage samples each cell; round 2's statement check of the
second-stage sessions (`second_stage`), which refuses the two-stage form while `second_stage.FORM` is off; its
refusals; and its profile at the footprint."""

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
from verity.core.protocols.spot import second_stage as SS, service as S
from verity.core.protocols.spot.one_stage import (draw as D, partition as P, registration as R,
                                                            verifier as V)
from verity.ml.commitments.window import window_binding

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
#: The boundary registration's ports, the replay block's `x` and output named by its template, which the design's links
#: (`second_stage.replay_links`) name.
BOUNDARY = BX, BOUT = tuple(f"{codec.descriptor_id(BLOCK)}/{p}" for p in ("x", "o0"))
INTERIOR = ("x", "out")


def _public(boundary: tuple[str, ...]) -> tuple[S.Public, ...]:
    return (S.Public("population", "structure", "N, the sample's range"),
            S.Public("window", "structure", "provenance: served, on test prompts (step 0)"),
            *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k} (step 0)") for v in boundary + INTERIOR
              for k in ("leaves", "domain")))


def _values(boundary: tuple[str, ...]) -> tuple[S.Value, ...]:
    return (*(S.Value(v, SCHEMA, "serve") for v in boundary), *(S.Value(v, SCHEMA, "interior") for v in INTERIOR))


PUBLIC, VALUES = _public(BOUNDARY), _values(BOUNDARY)
RULE = S.TwoStage(GRAIN, f"subset:{Q}", K)
TEMPLATE, = (t for t, _ in P.strata(P.population_program(PAIR, Q * N_RHO), QUERY))
#: The reference header of an honest statement in these streams, fixed before the run: PAIR's part's, its ports' schemas
#: and the set it was staged in, none of the run's fields (`fake_challenger.reference_header`).
REFERENCE = FC.reference_header(TEMPLATE, 0, Q * N_RHO, dict.fromkeys(INTERIOR, SCHEMA))


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


def _interior(m: int, previous: str, sample, row=0, **change) -> tuple[dict, dict]:
    """The interior registration's record over `m` proof blocks, its window naming the replay sample `sample` it lays out
    (`window.replay_sample`, the first stage's proved blocks in order where honest), and its ports."""
    program = P.population_program(PAIR, m)
    rec = _record(program, QUERY, INTERIOR, m, {"law": "subset", "k": m},
                  {"kind": "served", "run": RUN, "row": row, "interior": True, "replay_sample": sample}, previous)
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


def _stages(stream: S.Stream, row=0, sample=list) -> tuple[S.Draw, S.Draw]:
    """Both stages on `stream`, the interior record naming as its replay sample `sample` of the first stage's proved
    blocks (they themselves, in order, where honest)."""
    previous = R.digest(stream._log[-1]) if stream._log else None
    reg = stream.register(*_boundary(row=row, previous=previous))
    first = stream.select(reg)
    stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, sample(D.proved(first.draw)), row))
    return first, stream.select(stream._entries[reg.order].interior)


def _reads_of(statements) -> dict | None:
    """What a verifier of record reports the sessions read (`{<template>/<port>: {"value", "root"}}`): the statements'
    registered reads, None when none reads any."""
    out = {}
    for s in statements if isinstance(statements, list) else ():
        ports = s.get("registered", {}).get("ports") if isinstance(s, dict) and isinstance(s.get("registered"), dict) else None
        out |= {f"{s.get('template')}/{p}": {"value": e["value"], "root": e["root"]} for p, e in (ports or {}).items()}
    return out or None


def _two_stage_outcome(stream: S.Stream, statements, reads=_reads_of, sample=list, **params) -> S.Outcome:
    """The run's outcome with the second-stage statements `statements` gives (a function of the honest prover's one,
    `fake_challenger.Stream.statements`, PAIR's, and the second stage's sample, or as given), and the verifier of record
    reporting the reads `reads` gives of them; the interior record naming `sample` of the replay sample (`_stages`); the
    auditor's params `proofs`, if given, else the honest ones (`fake_challenger.Stream.proofs`)."""
    _, second = _stages(stream, sample=sample)
    stream.deliver(second, [SESSION])
    if callable(statements):
        statements = statements(stream.statements(second)[0], second)
    return stream.outcome(second, _lean(reads=reads(statements)), {"statement": "stub"},
                          session_roots={**_roots(BOUNDARY), **_roots(INTERIOR)}, statements=statements, **params)


def test_a_two_stage_stream_runs_both_stages_and_its_outcome_binds_both():
    """The boundary registration's population is the replay blocks; the first stage samples Q of them; the interior
    registration commits their Q·n_rho proof blocks after it, logged after the registration; the second stage samples k in
    each sampled replay block on the coins after the interior registration; and one outcome holds both audit records, each
    by both verifiers and the whole run by the Lean's. Round 2 holds the session's statement to the interior registration
    and the second stage's cells, its ports the replay block's Definition links (PAIR's `x` and `o0`) read at the
    boundary, and passes it; while the two-stage form is refused (`second_stage.FORM`) that refusal is the outcome's one
    failure, a reject. The profile an accepted one carries is at the footprint."""
    stream = _stream()
    assert stream.n_rho == N_RHO and stream._params["two_stage"] is True
    assert stream.cells == (N_RHO,) and stream.counts == (K,)
    reg = stream.register(*_boundary())
    assert reg.record["population"] == N and reg.record["query"] == R.query_digest(GRAIN)
    first = stream.select(reg)
    assert first.draw["law"] == "subset" and len(first.units) == Q and first.replay_sample is None
    ireg = stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, D.proved(first.draw)))
    assert ireg.order == reg.order and ireg.receipt["received"] == "1" and stream.log[-1] == ireg.record
    second = stream.select(ireg)
    assert second.replay_sample == first and second.law == S.second_law(first.units, (N_RHO,), (K,))
    assert [u // N_RHO for u in second.units] == list(range(Q))
    assert stream.deliver(second, [SESSION]) is None
    statements = stream.statements(second)
    (template, _), = P.strata(P.population_program(PAIR, Q * N_RHO), QUERY)
    assert [(s["template"], s["draw"]["units"], s["header"]["units"]["indices"]) for s in statements] == \
        [(template, D.proved(second.draw), list(range(Q * N_RHO)))]
    reads = statements[0]["registered"]["ports"]
    assert {p: (e["value"], e["reads"]) for p, e in reads.items()} == {
        p: (v, [D.proved(first.draw)[u // N_RHO] for u in range(Q * N_RHO)]) for p, v in (("x", BX), ("o0", BOUT))}
    out = stream.outcome(second, _lean(reads=_reads_of(statements)), {"statement": "stub"},
                         session_roots={**_roots(BOUNDARY), **_roots(INTERIOR)}, statements=statements)
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


def _linked(s: dict, shift: int = 0, **ports) -> dict:
    """Statement `s`'s registered reads, the honest prover's (PAIR's linked `x` and `o0` read at the boundary, one row a
    replay block, each slot its cell's), each read the replay block `shift` on, with `ports` more."""
    linked = {p: {**e, "reads": [(r + shift) % N for r in e["reads"]]} for p, e in s["registered"]["ports"].items()}
    return {**s["registered"], "ports": {**linked, **ports}}


def _own(second: S.Draw, shift: int = 0, port: str = "out") -> dict:
    """A registered read of the interior's own `port`, one row a slot: each slot's own, or the one `shift` on."""
    r, = (r for r in second.registration.record["roots"] if r["port"] == port)
    return {"value": port, "root": r["root"], "domain": r["domain"], "leaves": r["leaves"], "reads_from": 0,
            "reads": [(u + shift) % (Q * N_RHO) for u in range(Q * N_RHO)]}


def _restated(**change):
    """The honest statement with `change`, each a field or a function of the honest field's value."""
    return lambda s, second: [{**s, **{k: f(s[k]) if callable(f) else f for k, f in change.items()}}]


def _header(**change):
    return _restated(header=lambda h: {**h, **change})


def _frame(key: str, f):
    return _restated(header=lambda h: {**h, "frame_v3": {**h["frame_v3"], key: f(h["frame_v3"][key])}})


@pytest.mark.parametrize("statements, codes", [
    (lambda s, second: [s], ["two-stage-form"]),
    (lambda s, second: [{**s, "registered": _linked(s, out=_own(second))}], ["two-stage-form"]),
    (None, ["second-statements"]),
    ([], ["second-cover"]),
    (lambda s, second: [s, s], ["second-template"]),
    (_restated(template="TwoStageTestOther_v0"), ["second-template", "second-cover"]),
    (lambda s, second: [{k: v for k, v in s.items() if k != "registered"}], ["second-statement", "second-cover"]),
    (_restated(header=None), ["second-statement", "second-cover", "second-reported"]),
    (_header(instances=Q * N_RHO + 1), ["second-population"]),
    (_restated(header=lambda h: {**h, "units": {**h["units"], "indices": list(range(1, Q * N_RHO + 1))}}),
     ["second-population"]),
    (_header(scheme="frame-v3-sha512/other"), ["second-header", "second-roots"]),
    (_header(set="served/other/m0"), ["second-header"]),
    (_header(set=SS.set_name(RUN, 1)), ["second-header"]),
    (_frame("bindings", lambda b: {**b, "out": b["x"]}), ["second-header"]),
    (_frame("bindings", lambda b: {"x": b["x"]}), ["second-header"]),
    (_frame("schemas", lambda s: {**s, "out": "hm96-sha512/row/v1"}), ["second-header"]),
    (_header(registered={"x": {"value": BX, "positions": [0], "paths": [[0]]}}), ["two-stage-form"]),
    (_restated(header=lambda h: {**h, "units": {**h["units"], "blocks": [0]}}), ["two-stage-form"]),
    (_restated(header=lambda h: {**h, "units": {**h["units"], "units_per_vu": 2}}), ["second-header"]),
    (_restated(header=lambda h: {**h, "frame_v3": {**h["frame_v3"], "owner": "other"}}), ["second-header"]),
    (_frame("roots", lambda r: {**r, "out": "00" * 64}), ["second-roots"]),
    (_frame("roots", lambda r: {k: v for k, v in r.items() if k != "out"}), ["second-roots"]),
    (_frame("domain_ids", lambda d: {**d, "out": d["x"]}), ["second-roots"]),
    (_restated(draw=lambda d: {**d, "units": [0, 3] if d["units"] != [0, 3] else [1, 4]}), ["second-sample"]),
    (_restated(draw=lambda d: {**d, "population": N * N_RHO}), ["second-sample"]),
    (lambda s, second: [{**s, "registered": _linked(s, shift=1)}], ["second-read"]),
    (lambda s, second: [{**s, "registered": _linked(s, out=_own(second, shift=1))}], ["second-read"]),
    (lambda s, second: [{**s, "registered": _linked(s, out=s["registered"]["ports"]["x"])}], ["second-read"]),
    (lambda s, second: [{**s, "registered": _linked(s, y=_own(second))}], ["second-read"]),
    (lambda s, second: [{**s, "registered": _linked(s, x=_own(second, port="x"))}], ["second-read"]),
    (lambda s, second: [{**s, "registered": {"ports": {"x": s["registered"]["ports"]["x"]}}}], ["second-read"]),
    (_restated(registered=None), ["second-read"]),
])
def test_round_two_refuses_a_statement_the_run_did_not_make(statements, codes):
    """Round 2, by both verifiers, passes the run's own statements, its linked ports reading the boundary at each slot's
    cell's replay block, and its own `out` reading each slot's own rows or not (the form's refusal their one failure),
    and refuses: none (`second-statements`), none for a part's template (`second-cover`), a template
    twice or one of no part (`second-template`), one that isn't a statement (`second-statement`, and `second-reported`
    when its session's reads are reported, since no statement makes them), one whose header has a key its template's
    reference header lacks, at the top or in an object, or holds otherwise, or names another input set than the run's
    for its position or another binding than the run's for a port (`second-header`; the `registered` it names is the
    reads' business and its `units.blocks` the session verifier's, not the header's), and one that isn't the
    run's: another population of slots (`second-population`), a scheme, root or domain the interior registration's isn't
    or a port of it it neither roots nor reads (`second-roots`), another sample (`second-sample`), or a read of another
    cell's replay block, another slot's rows, a boundary port by a port no link names, another port's own, or a linked
    port's own, or a linked port not read (`second-read`). Neither audit record refuses the reads of its own ports at
    their own roots, which round 2 holds."""
    stream = _stream()
    out = _two_stage_outcome(stream, statements)
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["codes"] == codes and "run" not in checks, checks["statements"]
    assert "registered" not in checks and stream.records[0]["replay_sample"]["audit"]["accepted"]


@pytest.mark.parametrize("reads", [
    lambda ss: None,
    lambda ss: {**_reads_of(ss), "x": {"value": BX, "root": "ee" * 64}},
    lambda ss: {**_reads_of(ss), "z": {"value": "w", "root": "02" * 64}},
])
def test_round_two_holds_the_reads_to_what_the_verdicts_report(reads):
    """The reads the verifier of record reports are exactly the statements' (`second-reported`): none, another root, or
    one more."""
    stream = _stream()
    out = _two_stage_outcome(stream, lambda s, second: [s], reads)
    assert not out.accepted and stream.records[0]["audit"]["checks"]["statements"]["codes"] == ["second-reported"]


FV = REFERENCE["frame_v3"]


@pytest.mark.parametrize("header, codes", [
    (REFERENCE, ["two-stage-form"]),
    ({**REFERENCE, "instances": 7, "units": {**REFERENCE["units"], "indices": [9]},
      "frame_v3": {**FV, "roots": {"x": "00" * 64}, "domain_ids": {}}, "registered": {"x": {"value": "w"}}},
     ["two-stage-form"]),
    (FC.reference_header(TEMPLATE, 9, 9 + Q * N_RHO, dict.fromkeys(INTERIOR, SCHEMA)), ["two-stage-form"]),
    ({**REFERENCE, "set": "served/other/m0", "frame_v3": {**FV, "bindings": {"x": "11" * 64}}}, ["two-stage-form"]),
    ({**REFERENCE, "scheme": "frame-v3-sha512/other"}, ["second-header"]),
    ({**REFERENCE, "units": {**REFERENCE["units"], "units_per_vu": 2}}, ["second-header"]),
    ({**REFERENCE, "frame_v3": {**FV, "schemas": {**FV["schemas"], "out": "hm96-sha512/row/v1"}}}, ["second-header"]),
    ({**REFERENCE, "frame_v3": {**FV, "schemas": {**FV["schemas"], "y": SCHEMA}}}, ["second-header"]),
    ({**REFERENCE, "frame_v3": {k: v for k, v in FV.items() if k != "schemas"}}, ["second-header"]),
    ({"scheme": S.SCHEME}, ["second-header"]),
    (None, ["second-header"]),
])
def test_round_two_holds_each_header_to_its_templates_reference(header, codes):
    """`second-header`, by both verifiers: each statement's header is the reference header the auditor's params give its
    template (`template_circuits`) placed in the run (`second_stage.at_run`: its `set` the run's `served/<run>/m<k>`,
    its `frame_v3.bindings` each `frame_v3.schemas` port's binding in the run, both the verifier's), the two the same
    with the run's fields blanked (`second_stage.RUN_FIELDS`: `instances`, `units.indices`, `units.blocks`,
    `frame_v3.roots`, `frame_v3.domain_ids` and `registered`). The honest run passes unchanged, refused only on the form,
    under a reference fixed before the run that names none of the run's fields or names them otherwise, or was staged at
    other slots, or names another set and bindings, which the verifier replaces; a reference with another scheme, another
    `units.units_per_vu`, another schema, a port the interior record lacks or none of the ports, or a key the header
    lacks, is refused by name, and so is the statement when the params give no reference."""
    stream = _stream()
    out = _two_stage_outcome(stream, lambda s, second: [s], proofs=None if header is None else _circuits((PAIR,), header=header))
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["codes"] == codes and "run" not in checks, checks["statements"]


#: The interior's ports as a session's header shares them (``shared_rows``), a table of one row a slot each.
SHARED = dict.fromkeys(INTERIOR, Q * N_RHO)


def _sharing(registered=None):
    """The honest statement, its header sharing rows at the interior's ports (`SHARED`), with `registered` its reads, a
    function of the honest statement and the second stage, if given."""
    return lambda s, second: [{**s, "header": {**s["header"], "shared_rows": SHARED},
                               **({} if registered is None else {"registered": registered(s, second)})}]


@pytest.mark.parametrize("statements, codes", [
    (_sharing(lambda s, second: _linked(s, out=_own(second))), ["two-stage-form"]),
    (_sharing(), ["second-refs"]),
    (_sharing(lambda s, second: None), ["second-read", "second-refs"]),
    (_sharing(lambda s, second: _linked(s, out=_own(second, shift=1))), ["second-read"]),
    (lambda s, second: [{**s, "registered": _linked(s, out=_own(second))}], ["second-header"]),
])
def test_round_two_holds_each_port_a_header_shares_rows_at_to_a_registered_read(statements, codes):
    """`second-refs`, by both verifiers, under a reference whose sessions share rows at the interior's ports
    (``shared_rows``: tables whose rows an instance reads by the refs in its public file's body). The run's statement
    reading its linked `x` at the boundary and its own `out` at each slot's own rows passes, refused only on the form.
    One leaving the shared `out` unread is refused: the session's verifier would only bound its refs by their table, so
    one could point a drawn unit at another unit's rows. So is one reading nothing registered. A shared port read at
    another slot's rows, where the verifier would hold the ref, is `second-read`'s, and a header sharing no rows where
    its reference does is `second-header`'s."""
    stream = _stream()
    out = _two_stage_outcome(stream, statements, proofs=_circuits((PAIR,), header={**REFERENCE, "shared_rows": SHARED}))
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["codes"] == codes and "run" not in checks, checks["statements"]


def test_round_two_places_the_reference_in_the_run_as_the_served_path_names_it():
    """The honest header is the served path's (`flock_inputs.flock_input_files`): its template's reference, the input
    set `served/<run>/m<k>` of the interior's `k`-th member, `<run>` the interior record's `window.run`, and each port's
    binding the window-domain rule's over the interior record's program, partition and that port's leaves
    (`rows.Binding.binding`, whose domain is the one the record registers for the port). The verifier recomputes both
    from the interior record alone (`second_stage.at_run`), so the header is the reference so placed but for the run's
    fields, and passes, refused only on the form."""
    stream = _stream()
    _, second = _stages(stream)
    rec, (s,) = second.registration.record, stream.statements(second)
    h, n = s["header"], Q * N_RHO
    assert h["set"] == SS.set_name(RUN, 0) == f"served/{RUN}/m0" and rec["window"]["run"] == RUN
    assert h["frame_v3"]["bindings"] == {
        p: window_binding(rec["program"], rec["partition"], p, SCHEMA, n, RUN).hex() for p in INTERIOR}
    assert h["frame_v3"]["domain_ids"] == {
        p: R.window_domain(rec["program"], rec["partition"], p, SCHEMA, n, RUN) for p in INTERIOR}
    own = SS.Own((), None, rec, None, (), refs=((TEMPLATE, REFERENCE),))
    assert {p: SS.port_binding(rec, p, SCHEMA, RUN) for p in INTERIOR} == h["frame_v3"]["bindings"]
    assert SS.blank_run(SS.at_run(own, False, 0, TEMPLATE, REFERENCE)) == SS.blank_run(h)
    assert SS.at_run(own, False, 1, TEMPLATE, REFERENCE)["set"] == f"served/{RUN}/m1"
    assert SS.at_run(SS.Own((), None, {**rec, "window": {"kind": "served"}}, None, ()), False, 0, TEMPLATE,
                     REFERENCE) is None
    assert stream.deliver(second, [SESSION]) is None
    out = stream.outcome(second, _lean(reads=_reads_of([s])), {"statement": "stub"},
                         session_roots={**_roots(BOUNDARY), **_roots(INTERIOR)}, statements=[s])
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["codes"] == ["two-stage-form"] and "run" not in checks


@pytest.mark.parametrize("sample, statements, codes", [
    (list, lambda s, second: [s], ["two-stage-form"]),
    (lambda u: u[::-1], lambda s, second: [s], ["second-interior-order"]),
    (lambda u: u[:-1], lambda s, second: [s], ["second-interior-order"]),
    (lambda u: [*u, u[0]], lambda s, second: [s], ["second-interior-order"]),
    (lambda u: [str(x) for x in u], lambda s, second: [s], ["second-interior-order"]),
    (lambda u: None, lambda s, second: [s], ["second-interior-order"]),
    (lambda u: u[::-1], None, ["second-interior-order", "second-statements"]),
    (lambda u: u[::-1], lambda s, second: [{**s, "header": {**s["header"], "set": "served/other/m0"}}],
     ["second-interior-order", "second-header"]),
])
def test_round_two_holds_the_interior_record_to_the_replay_sample_in_order(sample, statements, codes):
    """`second-interior-order`, by both verifiers: the interior record names the replay sample it lays out
    (`window.replay_sample`, which the proof server writes), and round 2 takes it only as the verifier's own, the first
    stage's proved blocks in their order, so block `S[i]`'s cell is `off_t + i·n_t` by the record and not by convention.
    Reversed, short, long, as strings or missing, it is refused by name, before any statement is read and beside
    whatever else fails."""
    stream = _stream()
    out = _two_stage_outcome(stream, statements, sample=sample)
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["codes"] == codes and "run" not in checks, checks["statements"]
    if "second-interior-order" in codes:
        assert checks["statements"]["detail"]["second-interior-order"] == SS.INTERIOR_ORDER_WHY


def test_round_two_derives_the_links_from_the_replay_block(monkeypatch):
    """The design's links are the verifier's own, derived from the replay block's Definition, and the Stream takes none
    from its caller: BLOCK's PAIR batch reads its `x` and it returns PAIR's output, so PAIR's `x` and `o0` are the
    boundary's ports of them (`second_stage.replay_links`). A replay block they can't be derived from is refused
    (`second-links`); the run of record, deriving its own, then disagrees."""
    stream = _stream()
    first, _ = _stages(stream)
    assert SS.replay_links(P.population_program(BLOCK, N), codec.descriptor_id(BLOCK), first.registration.record) == \
        (SS.Link(TEMPLATE, "x", BX), SS.Link(TEMPLATE, "o0", BOUT))
    with pytest.raises(TypeError):
        stream.register_interiors(first, *_interior(Q * N_RHO, first.registration.digest, D.proved(first.draw)), links=[])
    monkeypatch.setattr(SS, "replay_links", lambda *a: None)
    stream = _stream()
    out = _two_stage_outcome(stream, lambda s, second: [s])
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["codes"] == ["second-links"]
    assert checks["statements"]["detail"]["second-links"] == SS.LINKS_WHY and checks["run"]["codes"] == [V.DISAGREE]


def test_an_own_read_at_another_root_fails_the_interior_audit():
    """The interior audit takes a read of its own port only at that port's root: at another, its `registered` check
    fails beside round 2's (`second-read`)."""
    stream = _stream()
    out = _two_stage_outcome(stream, lambda s, second: [{**s, "registered": _linked(s, out={
        **_own(second), "root": "ee" * 64})}])
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
    sample = D.proved(first.draw)
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO + 1, reg.digest, sample))) == \
        ("interior-population",)
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO, "00" * 64, sample))) == \
        ("R6-previous",)
    ireg = stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, sample))
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, sample, row=1))) == \
        ("interior-registered",)
    assert _refused(lambda: stream.select(reg)) == ("drawn",)
    second = stream.select(ireg)
    assert _refused(lambda: stream.select(ireg)) == ("drawn",)
    assert _refused(lambda: stream.deliver(first, [SESSION])) == ("not-drawn",)
    assert stream.deliver(second, [SESSION]) is None
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, sample))) == \
        ("delivered",)


def test_a_one_stage_stream_has_no_interior_registration():
    spec = S.Spec(VALUES[:2], S.Law("subset:2"), S.Check(P.program_digest(Program(BLOCK)), GRAIN, proof_deadline=60.0),
                  PUBLIC, b"two-stage-test")
    stream = FC.Stream(spec, BLOCK, budget=2, delta=DELTA, challenger="challenger-test", clock=Clock())
    reg = stream.register(*_boundary())
    first = stream.select(reg)
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, D.proved(first.draw)))) == \
        ("rule",)
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
#: MIXED's boundary ports, its `x` and `y` and its two outputs named by its template: PAIR's `x` and `o0` are its `x`
#: and `o0`, TRIPLE's its `y` and `o1` (`second_stage.replay_links`).
MIXED_BOUNDARY = tuple(f"{codec.descriptor_id(MIXED)}/{p}" for p in ("x", "y", "o0", "o1"))


def _window_spec(k=K, query=WINDOW_QUERY, window=WINDOW) -> S.Spec:
    return S.Spec(_values(MIXED_BOUNDARY), S.TwoStage(MIXED_GRAIN, f"subset:{Q}", k),
                  S.Check(tuple(P.program_digest(Program(t)) for t in window), query, proof_deadline=60.0),
                  _public(MIXED_BOUNDARY), b"two-stage-window-test")


def _window_stream(spec=None, window=WINDOW, replay=MIXED) -> S.Stream:
    return FC.Stream(spec or _window_spec(), window, replay=replay, budget=2, delta=DELTA, challenger="challenger-test",
                     clock=Clock())


def _window_boundary(stream: S.Stream) -> S.WindowCommitmentRecord:
    boundary = P.population_program(MIXED, N)
    return stream.register(_record(boundary, MIXED_GRAIN, MIXED_BOUNDARY, N, {"law": "subset", "k": Q},
                                   {"kind": "served", "run": RUN, "row": 0}, None),
                           _ports(boundary, MIXED_GRAIN, MIXED_BOUNDARY, N))


def _window_interior(program: Program, previous: str, first: S.Draw) -> tuple[dict, dict]:
    n = P.units(program, WINDOW_QUERY)
    return (_record(program, WINDOW_QUERY, INTERIOR, n, {"law": "subset", "k": n},
                    {"kind": "served", "run": RUN, "row": 0, "interior": True, "replay_sample": D.proved(first.draw)},
                    previous),
            _ports(program, WINDOW_QUERY, INTERIOR, n))


def _window_stages(stream: S.Stream) -> S.Draw:
    """The window's two stages on `stream`: MIXED's boundary registration, its replay sample, the interior registration
    over the sample's proof blocks template-major, and the second stage, delivered."""
    reg = _window_boundary(stream)
    first = stream.select(reg)
    ireg = stream.register_interiors(first, *_window_interior(stream.interior_program(Q), reg.digest, first))
    second = stream.select(ireg)
    assert stream.deliver(second, [SESSION]) is None
    return second


def test_a_replay_block_of_two_templates_runs_with_a_template_major_interior():
    """A replay block of PAIR x 2 and TRIPLE x 1: the boundary registration over N of them, the interior over the
    sample's Q·3 proof blocks template-major (PAIR's Q·2, then TRIPLE's Q), the second stage one stratum per cell, k of
    each PAIR cell and TRIPLE's one-unit cells whole, and an outcome whose whole run is the Lean verifier's. Round 2
    links each template's `x` and output to the boundary ports of the parameter it reads and the field it returns
    (PAIR's MIXED's `x` and `o0`, TRIPLE's its `y` and `o1`), and passes the sessions reading them there, refused only
    on the form; its profile is `Subset(min k_t)` over T·n_max proof blocks a replay block, which bounds the cells'
    escape."""
    stream = _window_stream()
    assert stream.cells == (2, 1) and stream.counts == (1, 1) and stream.n_rho == 3
    reg = _window_boundary(stream)
    first = stream.select(reg)
    interior = P.mixed_population_program([(PAIR, 2 * Q), (TRIPLE, Q)])
    assert stream.interior_program(Q).root.id == interior.root.id
    ireg = stream.register_interiors(first, *_window_interior(interior, reg.digest, first))
    second = stream.select(ireg)
    assert second.law == S.second_law(first.units, (2, 1), (1, 1))
    assert [s["units"] for s in second.law["strata"]] == [[[0, 2]], [[2, 4]], [[4, 5]], [[5, 6]]]
    assert sorted(u // 2 for u in second.units if u < 4) == [0, 1] and {4, 5} <= set(second.units)
    assert stream.deliver(second, [SESSION]) is None
    (pair, triple), (x, y, o0, o1) = (codec.descriptor_id(t) for t in WINDOW), MIXED_BOUNDARY
    assert SS.replay_links(P.population_program(MIXED, N), codec.descriptor_id(MIXED), reg.record) == (
        SS.Link(pair, "x", x), SS.Link(triple, "x", y), SS.Link(pair, "o0", o0), SS.Link(triple, "o0", o1))
    statements = stream.statements(second)
    assert [(s["template"], {p: e["value"] for p, e in s["registered"]["ports"].items()}) for s in statements] == [
        (pair, {"x": x, "o0": o0}), (triple, {"x": y, "o0": o1})]
    out = stream.outcome(second, _lean(reads=_reads_of(statements)), {"statement": "stub"},
                         session_roots={**_roots(MIXED_BOUNDARY), **_roots(INTERIOR)}, statements=statements)
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["codes"] == ["two-stage-form"] and "run" not in checks, checks
    p = stream._footprint(stream._entries[reg.order])
    assert p.sizes == (N, 4) and p.draws["unit"] == Subset(1) and dict(p.drawn) == {S.REPLAY: Q, "unit": Q * 2}
    for w in ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)):
        cells = Fraction(comb(2 - w[0], 1), comb(2, 1)) * Fraction(comb(1 - w[1], 1), comb(1, 1))
        assert cells <= Fraction(comb(4 - sum(w), 1), comb(4, 1))


def test_round_two_refuses_statements_out_of_the_parts_order():
    """Round 2 reads the statements by position, one for each part in the parts' order, the interior program's strata's
    (PAIR's, then TRIPLE's): the honest statements swapped are refused by name (`second-order`), at the first statement
    not its position's part's, and the run of record refuses them alike."""
    stream = _window_stream()
    second = _window_stages(stream)
    statements = stream.statements(second)[::-1]
    out = stream.outcome(second, _lean(reads=_reads_of(statements)), {"statement": "stub"},
                         session_roots={**_roots(MIXED_BOUNDARY), **_roots(INTERIOR)}, statements=statements)
    checks = stream.records[0]["audit"]["checks"]
    assert not out.accepted and checks["statements"]["codes"] == ["second-order"] and "run" not in checks, checks
    assert checks["statements"]["detail"]["second-order"] == {"statement": 0, "template": codec.descriptor_id(TRIPLE)}


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
    reg = _window_boundary(stream)
    first = stream.select(reg)
    blocks = P.mixed_population_program([(PAIR, 2), (TRIPLE, 1)] * Q)
    assert "R2-program" in _refused(lambda: stream.register_interiors(first, *_window_interior(blocks, reg.digest, first)))
    assert not P.in_order(blocks, WINDOW_QUERY, 3 * Q)


def _circuits(window=WINDOW, files=("ab" * 64, "cd" * 64), header=REFERENCE) -> dict:
    return {"templates": [{"template": P.template_id(t), "circuit": c, "header": header} for t, c in zip(window, files)]}


def test_the_params_name_each_templates_circuit_file_in_the_checks_order():
    """`template_circuits`: the auditor's params name each of the check's templates' circuit files by SHA-512 and its
    statements' reference header, in the check's order, as (template, circuit, header) triples, for one template's
    check and a window's; none names no file; any other form, a header that isn't an object, and a list that isn't the
    check's templates once each in its order, is refused."""
    ids = [P.template_id(t) for t in WINDOW]
    assert S.template_circuits(_circuits(), WINDOW_QUERY) == ((ids[0], "ab" * 64, REFERENCE), (ids[1], "cd" * 64, REFERENCE))
    assert S.template_circuits(_circuits((PAIR,)), QUERY) == ((ids[0], "ab" * 64, REFERENCE),)
    assert S.template_circuits(None, WINDOW_QUERY) == ()
    entries = _circuits()["templates"]
    for proofs, match in (({"templates": entries[::-1]}, "not the check's templates"),
                          ({"templates": entries[:1]}, "not the check's templates"),
                          ({"templates": [*entries, entries[0]]}, "not the check's templates"),
                          ({"templates": [{**entries[0], "circuit": "AB" * 64}, entries[1]]}, "lowercase hex"),
                          ({"templates": [{**entries[0], "circuit": "ab" * 64 + "\n"}, entries[1]]}, "lowercase hex"),
                          ({"templates": [{**entries[0], "kind": "flat"}, entries[1]]}, "lowercase hex"),
                          ({"templates": [{**entries[0], "header": None}, entries[1]]}, "each header an object"),
                          ({"templates": [{**entries[0], "header": []}, entries[1]]}, "each header an object"),
                          ({"templates": [{k: v for k, v in entries[0].items() if k != "header"}, entries[1]]},
                           "each header an object"),
                          ({"templates": entries, "circuit": "ab" * 64}, "proofs are"), ({}, "proofs are")):
        with pytest.raises(ValueError, match=match):
            S.template_circuits(proofs, WINDOW_QUERY)
    with pytest.raises(ValueError, match="a template query"):
        S.template_circuits(_circuits(), {"name": "Q_word", "version": 1, "params": {"X": 16, "W": 32}})


def test_an_outcome_carries_the_params_to_the_run_of_record(monkeypatch):
    """The auditor's params reach the run of record under the interior's params, where the Lean verifier reads them
    (`Run.templateCircuits`): an outcome naming each template's file is the run of record's too, its run carrying them
    (refused by round 2, as `test_a_replay_block_of_two_templates_runs_with_a_template_major_interior` says); a run whose
    params the Lean refuses fails `run`; params `template_circuits` refuses are refused before anything is decided, and a
    one-stage stream takes none."""
    runs, lean = [], V.lean
    tampered = {"templates": _circuits()["templates"][::-1]}

    def spy(piece, args, tamper=False):
        if piece == "run":
            runs.append(args)
            if tamper:
                args["interior"]["params"]["proofs"] = tampered
        return lean(piece, args)

    for tamper in (False, True):
        monkeypatch.setattr(V, "lean", lambda piece, args, tamper=tamper: spy(piece, args, tamper))
        stream = _window_stream()
        second = _window_stages(stream)
        assert _refused(lambda: stream.outcome(second, _lean(), {"statement": "stub"}, proofs=tampered)) == ("proofs",)
        assert _refused(lambda: stream.outcome(second, _lean(), {"statement": "stub"}, proofs={})) == ("proofs",)
        assert not runs
        statements = stream.statements(second)
        out = stream.outcome(second, _lean(reads=_reads_of(statements)), {"statement": "stub"},
                             session_roots={**_roots(MIXED_BOUNDARY), **_roots(INTERIOR)}, statements=statements,
                             proofs=_circuits())
        assert runs.pop()["interior"]["params"]["proofs"] == (tampered if tamper else _circuits()) and not runs
        assert not out.accepted and ("run" in stream.records[0]["audit"]["checks"]) is tamper
    stream = FC.Stream(S.Spec(VALUES[:2], S.Law("subset:2"), S.Check(P.program_digest(Program(BLOCK)), GRAIN,
                                                                      proof_deadline=60.0), PUBLIC, b"two-stage-test"),
                       BLOCK, budget=2, delta=DELTA, challenger="challenger-test", clock=Clock())
    draw = stream.select(stream.register(*_boundary()))
    stream.deliver(draw, [SESSION])
    assert _refused(lambda: stream.outcome(draw, _lean(), {"statement": "stub"}, proofs=_circuits())) == ("proofs",)
