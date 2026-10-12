"""The one-stage audit's verifier of record (`one_stage.verifier`): the Lean and the Python run on every check, both
outcomes are recorded beside the one of record, a disagreement is refused naming both, Lean that can't run fails closed,
and `RECORD_VERDICT` switches the record back to Python in one line."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import verity_difftest
from test_service import DELTA, PAIR, QUERY, SESSION, _lean, _ports, _record, _spec, _stream
from verity.ml.circuits.codec import canonical_json
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as D, partition as P, registration as R, verifier as V
from verity.core.protocols.sampled_proofs.one_stage.verdict import Verdict


def _drawn():
    stream = _stream()
    reg = stream.register(_record(), _ports(6))
    return stream, reg, stream.select(reg)


def _args(reg, draw, **change) -> dict:
    return {"registration": reg.record, "receipt": reg.receipt, "draw": draw.draw, "law": draw.law,
            "proved_units": list(draw.units), "instances": len(draw.units), "statement": {"statement": "stub"},
            "verdicts": _lean(), "delta": DELTA, "sessions": [SESSION], **change}


def _python_record(a: dict) -> dict:
    return audit.audit_record(*a.values(), session_registered=audit.session_registered(a["verdicts"], a["registration"]))


def _public(rec: dict) -> tuple:
    return (S.SCHEME, {r["port"]: r["root"] for r in rec["roots"]}, {r["port"]: r["domain"] for r in rec["roots"]},
            rec["population"])


def _rule(spec: S.Spec, rec) -> tuple[Verdict, Verdict]:
    """The public-items rule by the verifier of record, and the service's Python."""
    py = S.check_public(spec, rec)
    return V.check_public(py, rec, S.SCHEME, spec.check.query, [(v.name, v.registered) for v in spec.values],
                          [p.item for p in spec.public]), py


def _held(verdicts: dict, outputs: tuple = ()) -> tuple[Verdict, Verdict]:
    """The outcome's hold on the verdicts by the verifier of record, and the service's Python."""
    py = S.check_verdicts(verdicts, [SESSION], outputs)
    return V.check_verdicts(py, verdicts, [SESSION], outputs), py


def test_every_check_runs_both_and_records_them():
    _, reg, draw = _drawn()
    program, law = P.population_program(PAIR, 6), {"law": "subset", "k": 2}
    for v, py in ((V.check(_record(), program, QUERY, law, S.SCHEME, _ports(6)),
                   R.check(_record(), program, QUERY, law, S.SCHEME, _ports(6))),
                  (V.check(_record(), P.population_program(PAIR, 5), QUERY, law, S.SCHEME, _ports(6)),
                   R.check(_record(), P.population_program(PAIR, 5), QUERY, law, S.SCHEME, _ports(6))),
                  (V.check_draw(draw.draw, reg.record, draw.law), D.check_draw(draw.draw, reg.record, draw.law)),
                  (V.check_draw({**draw.draw, "units": [0, 6]}, reg.record, draw.law),
                   D.check_draw({**draw.draw, "units": [0, 6]}, reg.record, draw.law)),
                  (V.check_proved(draw.draw, draw.units, 2), D.check_proved(draw.draw, draw.units, 2)),
                  (V.check_proved(draw.draw, draw.units[:1], 2), D.check_proved(draw.draw, draw.units[:1], 2)),
                  (V.matches_public(reg.record, *_public(reg.record)), R.matches_public(reg.record, *_public(reg.record))),
                  (V.matches_public(reg.record, "frame-v3/blake3", *_public(reg.record)[1:]),
                   R.matches_public(reg.record, "frame-v3/blake3", *_public(reg.record)[1:])),
                  _rule(_spec(), _record()), _rule(_spec(), {**_record(), "tiles": [2, 3]}),
                  _rule(_spec(public=()), _record()), _rule(_spec(), "a record"),
                  _held(_lean()), _held(_lean(None)), _held(_lean(sessions=())), _held(_lean(public={"n": 1}), ("n",)),
                  _held({**_lean(public={"n": 1}), "b": _lean(public={"n": 2})["lean"]}, ("n",))):
        side = {"ok": py.ok, "codes": list(py.codes)}
        assert v == py and v.verifiers == {"of_record": "lean", "lean": side, "python": side}
    rec = V.audit_record(**_args(reg, draw))
    assert rec["verifiers"] == {"of_record": "lean", "lean": {"accepted": True, "failed": []},
                                "python": {"accepted": True, "failed": []}, "agree": True}
    assert canonical_json({k: x for k, x in rec.items() if k != "verifiers"}) == canonical_json(_python_record(_args(reg, draw)))
    bad = V.audit_record(**_args(reg, draw, session_roots={"x": "00" * 64}))
    assert not bad["accepted"] and bad["verifiers"]["agree"] and bad["verifiers"]["lean"]["failed"] == ["roots"]


def _lean_answering(monkeypatch, fn: str, answer):
    """The Lean, answering `fn` with `answer(its own answer)`."""
    real = V.lean
    monkeypatch.setattr(V, "lean", lambda f, value: answer(real(f, value)) if f == fn else real(f, value))


def test_a_disagreement_is_refused_naming_both(monkeypatch):
    stream, reg, draw = _drawn()
    _lean_answering(monkeypatch, "registration", lambda got: {"codes": ["R1-format"]})
    v = V.check(_record(), P.population_program(PAIR, 6), QUERY, {"law": "subset", "k": 2}, S.SCHEME, _ports(6))
    assert v == Verdict(False, (V.DISAGREE,), {V.DISAGREE: {"lean": {"ok": False, "codes": ["R1-format"]},
                                                         "python": {"ok": True, "codes": [], "detail": {}}}})
    assert v.verifiers["lean"] == {"ok": False, "codes": ["R1-format"]} and v.verifiers["python"] == {"ok": True, "codes": []}
    with pytest.raises(S.Refused) as e:
        _stream().register(_record(), _ports(6))
    assert e.value.verdict.codes == (V.DISAGREE,)

    monkeypatch.setattr(V.D, "check_draw", lambda *a: Verdict(False, ("p",), {"p": "injected"}))
    assert V.check_draw(draw.draw, reg.record, draw.law).codes == (V.DISAGREE,)

    def other_digest(text):
        rec = json.loads(text)
        return json.dumps({**rec, "registration_digest": "00" * 64})
    _lean_answering(monkeypatch, "record", other_digest)
    rec = V.audit_record(**_args(reg, draw))
    assert not rec["accepted"] and rec["profile"] is None and rec["integrity_profile"] is None and rec["wrong_units"] is None
    assert rec["checks"]["verifiers"]["codes"] == [V.DISAGREE] and rec["verifiers"]["agree"] is False
    detail = rec["checks"]["verifiers"]["detail"]
    assert detail["fields"] == ["registration_digest"] and detail["lean"] == {"registration_digest": "00" * 64}
    assert detail["python"] == {"registration_digest": R.digest(reg.record)}
    stream.deliver(draw, [SESSION])
    out = stream.outcome(draw, _lean(), {"statement": "stub"})
    assert not out.accepted and out.verdict.detail["reject"]["checks"] == ["verifiers"]


def test_a_public_items_disagreement_refuses_the_registration(monkeypatch):
    _lean_answering(monkeypatch, "check_public", lambda got: {**got, "ok": False, "codes": ["undeclared"],
                                                              "detail": {"undeclared": ["leaves:x"]}})
    with pytest.raises(S.Refused) as e:
        _stream().register(_record(), _ports(6))
    v = e.value.verdict
    assert v.codes == (V.DISAGREE,) and v.detail[V.DISAGREE]["lean"]["codes"] == ["undeclared"]
    assert v.verifiers["lean"] == {"ok": False, "codes": ["undeclared"]} and v.verifiers["python"] == {"ok": True, "codes": []}


def test_an_outcome_whose_verdicts_disagree_is_refused(monkeypatch):
    stream, _, draw = _drawn()
    stream.deliver(draw, [SESSION])
    _lean_answering(monkeypatch, "check_verdicts", lambda got: {"ok": False, "codes": ["pending"],
                                                                "detail": {"pending": "the verifier of record hasn't run"}})
    with pytest.raises(S.Refused) as e:
        stream.outcome(draw, _lean(), {"statement": "stub"})
    assert e.value.verdict.codes == (V.DISAGREE,) and e.value.verdict.verifiers["python"] == {"ok": True, "codes": []}
    monkeypatch.undo()
    assert stream.outcome(draw, _lean(), {"statement": "stub"}).accepted


def test_lean_that_cannot_run_fails_closed(monkeypatch):
    _, reg, draw = _drawn()

    def broken(fn, value):
        raise V.LeanError("lake is not on the PATH")
    monkeypatch.setattr(V._server, "call", broken)
    v = V.check_draw(draw.draw, reg.record, draw.law)
    assert v.codes == (V.DISAGREE,) and v.verifiers["lean"] == {"refused": "lake is not on the PATH"}
    rec = V.audit_record(**_args(reg, draw))
    assert not rec["accepted"] and rec["verifiers"]["lean"] == {"refused": "lake is not on the PATH"}
    assert rec["checks"]["verifiers"]["detail"]["lean"] == {"refused": "lake is not on the PATH"}


def test_floats_reach_the_lean_as_python_wrote_them():
    """A float of 10^16 or more is sent with its digits (`Json.parse` would read Python's `1e+22` as an integer), so the
    Lean writes back what Python does; one it can't read back (-0.0) makes the two disagree."""
    assert V._dumps({"a": 1e22, "b": [2.5, 1e16, 3]}) == '{"a":10000000000000000000000.0,"b":[2.5,10000000000000000.0,3]}'
    _, reg, draw = _drawn()
    floats = {"big": 1e22, "small": 1e-7, "third": 1 / 3, "huge": 1.7976931348623157e308, "tiny": 5e-324}
    for delta in (DELTA, 0.05, 0.1):
        assert V.audit_record(**_args(reg, draw, statement=floats, delta=delta))["verifiers"]["agree"]
    assert not V.audit_record(**_args(reg, draw, statement={"x": -0.0}))["verifiers"]["agree"]


def test_the_switch_is_one_line(monkeypatch):
    lines = [line for line in Path(V.__file__).read_text().splitlines() if re.match(r"RECORD_VERDICT\b", line)]
    assert lines == ['RECORD_VERDICT = "lean"'] and V.RECORD_VERDICT == "lean"
    _, reg, draw = _drawn()
    monkeypatch.setattr(V, "RECORD_VERDICT", "python")
    v = V.check(_record(), P.population_program(PAIR, 5), QUERY, {"law": "subset", "k": 2}, S.SCHEME, _ports(6))
    py = R.check(_record(), P.population_program(PAIR, 5), QUERY, {"law": "subset", "k": 2}, S.SCHEME, _ports(6))
    assert v == py and v.verifiers["of_record"] == "python" and v.verifiers["lean"] == v.verifiers["python"]
    rec = V.audit_record(**_args(reg, draw))
    assert rec["verifiers"]["of_record"] == "python" and rec["verifiers"]["agree"]
    assert {k: x for k, x in rec.items() if k != "verifiers"} == _python_record(_args(reg, draw))
    _lean_answering(monkeypatch, "check_draw", lambda got: {**got, "ok": False, "codes": ["p"]})
    assert V.check_draw(draw.draw, reg.record, draw.law).codes == (V.DISAGREE,)


def test_the_build_convention_is_the_difftest_harness():
    assert V.BUILD_LOCK == verity_difftest.BUILD_LOCK
    assert f'name = "{V.EXE}"' in (V.PACKAGE / "lakefile.toml").read_text()
