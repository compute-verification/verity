"""The one-stage audit's verifier (`one_stage.verifier`): the Lean and the Python run on every check, both
outcomes are recorded beside the one of record, a disagreement is refused naming both, Lean that can't run fails closed,
and `RECORD_VERDICT` switches the record back to Python in one line."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest

import verity_difftest
from test_service import DELTA, OTHER, PAIR, QUERY, RUN, SCHEMA, SESSION, _lean, _ports, _record, _spec, _stream
from verity.circuits.codec import canonical_json
from verity.protocols.proofs.spot.device import challenger as C, coins as T
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import (audit, draw as D, hello as H, link_points as LP, partition as P,
                                                            registration as R, verifier as V)
from verity.protocols.proofs.spot.verifier.one_stage.verdict import Verdict


def _drawn():
    stream = _stream()
    reg = stream.register(_record(), _ports(6))
    return stream, reg, stream.select(reg)


def _args(reg, draw, **change) -> dict:
    return {"registration": reg.record, "receipt": reg.receipt, "draw": draw.draw, "law": draw.law,
            "proved_units": list(draw.units), "instances": len(draw.units), "statement": {"statement": "stub"},
            "verdicts": _lean(), "delta": DELTA, "sessions": [SESSION], **change}


def _python_record(a: dict) -> dict:
    opened = audit.less_own_reads(audit.session_registered(a["verdicts"], a["registration"]), a["registration"])
    return audit.audit_record(*a.values(), session_registered=opened)


def _public(rec: dict) -> tuple:
    return (S.SCHEME, {r["port"]: r["root"] for r in rec["roots"]}, {r["port"]: r["domain"] for r in rec["roots"]},
            rec["population"])


def _rule(spec: S.Spec, rec) -> tuple[Verdict, Verdict]:
    """The public-items rule by the verifier, and the service's Python."""
    py = S.check_public(spec, rec)
    return V.check_public(py, rec, S.SCHEME, spec.check.query, [(v.name, v.registered) for v in spec.values],
                          [p.item for p in spec.public]), py


def _held(verdicts: dict, outputs: tuple = ()) -> tuple[Verdict, Verdict]:
    """The outcome's hold on the verdicts by the verifier, and the service's Python."""
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


def test_the_declared_day_is_held_by_both():
    """R2's declared day (`registration._declared`, `VStar.Recursive.Caller.declaredCode`) by the verifier: the day's own program,
    another day's, a day with no declaration, an empty log and a day declared twice give the same codes in the Lean and the
    Python; and the Lean refuses the parameters the Python refuses (`registration._declarations`): a day without
    declarations or the other way round, a day that isn't a natural number, and a declaration not of the view's form."""
    rec, program, law = _record(), P.population_program(PAIR, 6), {"law": "subset", "k": 2}
    mine = {"blocks": [PAIR.id], "day": 0, "program": rec["program"]}
    other = {"blocks": [], "day": 1, "program": "0" * 128}
    for day, declared, codes in ((0, [mine, other], ()), (1, [mine, other], ("R2-declared",)),
                                 (2, [mine, other], ("R2-undeclared",)), (0, [], ("R2-undeclared",)),
                                 (0, [mine, {**other, "day": 0}], ("R2-declared-twice",))):
        v = V.check(rec, program, QUERY, law, S.SCHEME, _ports(6), day=day, declarations=declared)
        side = {"ok": not codes, "codes": list(codes)}
        assert v.codes == codes and v.verifiers == {"of_record": "lean", "lean": side, "python": side}
    params = V._params(program, QUERY, law, S.SCHEME, _ports(6), None, (), (), None, None, None)
    assert V.lean("registration", {"params": params, "record": rec})["codes"] == []
    for change in ({"day": 0}, {"declarations": [mine]}, {"day": -1, "declarations": [mine]},
                   {"day": True, "declarations": [mine]}, {"day": 0, "declarations": None},
                   {"day": 0, "declarations": [{"day": 0, "program": rec["program"]}]},
                   {"day": 0, "declarations": [{**mine, "program": 7}]}):
        assert "refused" in V.lean("registration", {"params": {**params, **change}, "record": rec}), change
        with pytest.raises(ValueError, match="declar"):
            R.check(rec, program, QUERY, law, S.SCHEME, _ports(6), day=change.get("day"),
                    declarations=change.get("declarations"))


def _window(design: dict, counts=(2, 18)) -> tuple:
    """A registration under `design` of a window of two templates, `counts` proof blocks of PAIR then of OTHER: the
    record, its program, its query and its ports."""
    program = P.mixed_population_program(list(zip((PAIR, OTHER), counts)))
    query = P.template_instances_query((PAIR, OTHER))
    part, n = P.partition(program, query), sum(counts)
    ports = {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, RUN)}
             for p, leaves in (("x", n), ("out", 2 * n))}
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **ports[p]} for i, p in enumerate(ports)]
    return (R.record(program, query, design, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": 0}), program, query,
            ports)


def test_the_per_member_forms_link_points_are_held_by_both():
    """`link-points` by the verifier (`registration.check`'s ``members``, `Registration.linkRefused`), after R8, on a
    window of two members at one unit a proof block, PAIR's proof blocks then OTHER's: two and eighteen, or ten each.
    A row is a sampling design with each member's fewest and most sampled proof blocks over the samples that end, the
    members a sample that ends misses (fewest none) and the members whose fewest and most open different link points
    (3 up to eight proof blocks, 4 up to sixteen, 5 past). The Lean and the Python both refuse the registration, for
    `link-points` alone and naming the members of either list, and pass it where both lists are empty:

    * a stratum inside each member (`stratified:10`), or a subset that leaves out one proof block (`subset:19`): no
      sample misses a member, each has one count, and both pass;
    * `subset:9` over two and eighteen misses PAIR's member and holds seven to nine of OTHER's, 3 link points or 4;
    * `subset:2` over ten and ten misses either member and gives each one count: link points alone pass it;
    * `subset:17` over ten and ten holds seven to ten of each, 3 link points or 4, and misses neither;
    * a Bernoulli sampling design of rate 0 samples nothing, so each member's fewest and most are none: link points
      alone pass it.

    With no members, the one-statement form's, both pass every record, and a member with a unit past the population
    is refused by both under each."""
    rows = [((2, 18), {"law": "stratified", "k": 10}, [(1, 1), (9, 9)], [], []),
            ((10, 10), {"law": "stratified", "k": 10}, [(5, 5), (5, 5)], [], []),
            ((2, 18), {"law": "subset", "k": 19}, [(1, 2), (17, 18)], [], []),
            ((2, 18), {"law": "subset", "k": 9}, [(0, 2), (7, 9)], [0], [1]),
            ((10, 10), {"law": "subset", "k": 2}, [(0, 2), (0, 2)], [0, 1], []),
            ((10, 10), {"law": "subset", "k": 17}, [(7, 10), (7, 10)], [], [0, 1]),
            ((2, 18), {"law": "bernoulli", "p": "0/1"}, [(0, 0), (0, 0)], [0, 1], [])]
    for counts, design, bounds, missed, differ in rows:
        rec, program, query, ports = _window(design, counts)
        n = sum(counts)
        members, past = [([[0, counts[0]]], 1), ([[counts[0], n]], 1)], [([[0, counts[0]]], 1), ([[counts[0], n + 1]], 1)]
        assert LP.bounds(*LP.law_of(program, query, design), members) == bounds, design
        assert [i for i, (lo, _) in enumerate(bounds) if lo == 0] == missed, design
        assert [i for i, (lo, hi) in enumerate(bounds) if LP.nbl(1, lo) != LP.nbl(1, hi)] == differ, design
        refused = sorted({*missed, *differ})
        codes = ("link-points",) if refused else ()
        for ms, want in ((members, codes), ((), ()), (past, ("link-points",))):
            v = V.check(rec, program, query, design, S.SCHEME, ports, members=ms)
            side = {"ok": not want, "codes": list(want)}
            assert v.codes == want and v.verifiers == {"of_record": "lean", "lean": side, "python": side}, (design, ms)
        py = R.check(rec, program, query, design, S.SCHEME, ports, members=members)
        assert py.codes == codes and py.detail == ({"link-points": {"members": refused}} if refused else {}), design
        if design["law"] == "stratified":
            assert [s["k"] for s in rec["law"]["strata"]] == [lo for lo, _ in bounds]


def test_the_record_reads_each_ports_last_root_and_its_own_values_as_the_python_does():
    """`audit_record`'s sessions' roots (`Record.checks`) and the registered reads it keeps (`Record.ownRegistered`), which
    the Lean compiles on a hash map of the registration's roots, against the Python's whole: the sessions' roots all
    right, one wrong, one at a port the registration doesn't name, and a port registered twice, read at its last root;
    and verdicts reporting reads of the registration's own port and of a value it doesn't name, of which the value's is
    dropped and the port's taken at that port's root (`Record.lessOwnReads`) and kept, refused, at another."""
    _, reg, draw = _drawn()
    rec = reg.record
    roots = {r["port"]: r["root"] for r in rec["roots"]}
    x = rec["roots"][0]["port"]
    twice = {**rec, "roots": [*rec["roots"], {**rec["roots"][0], "root": "77" * 64}]}
    reads = lambda root: _lean(registered={"0": {"value": x, "root": root},  # noqa: E731
                                           "1": {"value": "elsewhere", "root": "00" * 64}})
    for registration, given, verdicts, failed in (
            (rec, roots, _lean(), []), (rec, {**roots, x: "77" * 64}, _lean(), ["roots"]),
            (rec, {**roots, "elsewhere": "00" * 64}, _lean(), ["roots"]),
            (twice, {**roots, x: "77" * 64}, _lean(), ["receipt"]), (twice, roots, _lean(), ["receipt", "roots"]),
            (rec, roots, reads(roots[x]), []), (rec, roots, reads("77" * 64), ["registered"])):
        a = {**_args(reg, draw, registration=registration, verdicts=verdicts), "session_roots": given}
        got = V.audit_record(**a)
        assert got["verifiers"]["agree"] and got["verifiers"]["lean"]["failed"] == failed, got["verifiers"]
        opened = audit.less_own_reads(audit.session_registered(verdicts, registration), registration)
        py = audit.audit_record(**a, session_registered=opened)
        assert canonical_json({k: v for k, v in got.items() if k != "verifiers"}) == canonical_json(py)
    assert got["checks"]["registered"]["detail"]["opened"] == [x]


def _lean_answering(monkeypatch, fn: str, answer):
    """The Lean, answering `fn` with `answer(its own answer)`."""
    real = V.lean
    monkeypatch.setattr(V, "lean", lambda f, value: answer(real(f, value)) if f == fn else real(f, value))


def test_a_disagreement_is_refused_naming_both(monkeypatch):
    stream, reg, draw = _drawn()
    _lean_answering(monkeypatch, "registration", lambda got: {"codes": ["R1-format"]})
    v = V.check(_record(), P.population_program(PAIR, 6), QUERY, {"law": "subset", "k": 2}, S.SCHEME, _ports(6))
    assert v == Verdict(False, (V.DISAGREE,), {V.DISAGREE: {"lean": {"ok": False, "codes": ["R1-format"]},
                                                         "python": {"ok": True, "codes": [], "detail": {},
                                                                    "fanout": 1}}})
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


def test_the_draw_of_record_is_the_leans_on_the_bytes_the_python_read(monkeypatch):
    seen = []
    real = V.lean
    monkeypatch.setattr(V, "lean", lambda f, value: (seen.append(value) if f == "select" else None) or real(f, value))
    stream, reg, draw = _drawn()
    (asked,) = seen
    assert asked["rule"] == {"rule": "law", "law": "rate:1/3"} and asked["query"] == dict(QUERY)
    got = real("select", asked)
    assert draw.draw == got["draw"] and got["read"] == len(bytes.fromhex(asked["source"])) > 0
    for rule, law in ((S.All(), {"law": "subset", "k": 6}), (S.Law("rate:1/2"), {"law": "subset", "k": 3})):
        seen.clear()
        s = _stream(_spec(rule))
        d = s.select(s.register(_record(law=law), _ports(6)))
        (asked,) = seen
        got = real("select", asked)
        assert d.draw == got["draw"] and got["read"] == len(bytes.fromhex(asked["source"]))
        assert (got["read"] == 0) == isinstance(rule, S.All)
    _lean_answering(monkeypatch, "select", lambda got: {**got, "draw": {**got["draw"], "units": got["draw"]["units"][::-1]}})
    stream = _stream()
    with pytest.raises(S.Refused) as e:
        stream.select(stream.register(_record(), _ports(6)))
    assert e.value.verdict.codes == (V.DISAGREE,) and e.value.verdict.verifiers["python"]["read"] > 0
    _lean_answering(monkeypatch, "select", lambda got: {**got, "read": got["read"] + 1})
    stream = _stream()
    with pytest.raises(S.Refused) as e:
        stream.select(stream.register(_record(), _ports(6)))
    assert e.value.verdict.codes == (V.DISAGREE,)


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


def test_the_runs_hello_is_checked_by_both(monkeypatch):
    """The run's Hello on its record: one tree over the selection stream and each slot's, its key unused by earlier runs.
    A reused tree is refused `fresh` by both, and a Lean that answers otherwise makes the two disagree."""
    nonce, key, root = "ab" * 32, "cd" * 255 + "0e", "ef" * 64
    spec = {"block": 3, "rounds_log": 2, "scheme": H.SCHEME, "streams": H.run_streams(["selection"], 2, ["session"])}
    own = {"profile": "fast100", "coins": {**spec, "nonce": ""}}
    received = canonical_json({**own, "coins": {**spec, "nonce": nonce}}).decode()
    tree = {"key": key, "nonce": nonce, "root": root, "spec": spec}
    run = (own, received, tree)
    v = V.hello(*run, [], ["selection"], 2, ["session"])
    assert v.ok and v.verifiers["lean"] == v.verifiers["python"] == {"ok": True, "codes": []}
    assert spec["streams"] == ["selection", "slot/0/session", "slot/1/session"]
    v = V.hello(*run, [key], ["selection"], 2, ["session"])
    assert v.codes == ("fresh",) and v.verifiers["lean"] == {"ok": False, "codes": ["fresh"]}
    assert V.hello(*run, [], ["selection"], 3, ["session"]).codes == ("streams",)
    _lean_answering(monkeypatch, "hello", lambda got: {**got, "ok": True, "codes": []})
    assert V.hello(*run, [key], ["selection"], 2, ["session"]).codes == (V.DISAGREE,)


def test_the_leans_coins_are_a_challengers_entropy_and_never_a_streams(monkeypatch):
    """The Lean verifier's `IO.getRandomBytes` (A3, `verifier.coins`) is a challenger's entropy, never a stream's own
    coins: a stream given it, or no coins, refuses (`NO_OWN_COINS`). A challenger on it commits to its coins, and a
    stream sampling from that challenger has the Lean's sample of record on coins it released after the registration."""
    a, b = V.coins(64), V.coins(64)
    assert len(a) == len(b) == 64 and a != b and V.coins(0) == b""
    for coins in (None, V.coins):
        with pytest.raises(ValueError) as e:
            S.Stream(_spec(), PAIR, budget=1, delta=DELTA, challenger="c", coins=coins)
        assert str(e.value) == S.NO_OWN_COINS
    selected = []
    real = V._server.call

    def spy(fn, value):
        if fn == "select":
            selected.append(bytes.fromhex(value["source"]))
        return real(fn, value)
    monkeypatch.setattr(V._server, "call", spy)
    challenger = C.LiveChallenger("verity/test/coins/v1", 0, [("positions", 8)], V.coins)
    coins = T.ChallengerCoins(C.LiveRecord(challenger, "positions", itertools.count().__next__))
    stream = S.Stream(_spec(), PAIR, budget=1, delta=DELTA, challenger="c", coins=coins)
    reg = stream.register(_record(), _ports(6))
    segment = coins.commit(bytes.fromhex(reg.digest))
    draw = stream.select(reg)
    released = b"".join(o.coin for o in segment.openings)
    assert len(draw.units) == 2 and len(selected) == 1 and 0 < len(selected[0]) and released.startswith(selected[0])


def test_coins_the_lean_cannot_give_are_no_coins(monkeypatch):
    for n in (-1, V.COINS_MAX + 1, 2.0, True):
        with pytest.raises(ValueError, match="coins"):
            V.coins(n)
    assert V.lean("coins", "64") == {"refused": "coins: the input is a number of bytes"}
    assert V.lean("coins", V.COINS_MAX + 1) == {"refused": f"coins: {V.COINS_MAX + 1} bytes in one request, past "
                                                           f"{V.COINS_MAX}"}
    for answer in (("reject", "no"), ("output", "zz"), ("output", "00" * 7)):
        monkeypatch.setattr(V._server, "call", lambda fn, value, answer=answer: answer)
        with pytest.raises(V.LeanError):
            V.coins(8)

    def broken(fn, value):
        raise V.LeanError("lake is not on the PATH")
    monkeypatch.setattr(V._server, "call", broken)
    with pytest.raises(V.LeanError, match="lake"):
        V.coins(8)


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
    """A float of 10^16 or more is sent with its digits (`Flock.JsonParse.parse` would read Python's `1e+22` as an
    integer), so the Lean writes back what Python does; one it can't read back (-0.0) makes the two disagree."""
    assert V._dumps({"a": 1e22, "b": [2.5, 1e16, 3]}) == '{"a":10000000000000000000000.0,"b":[2.5,10000000000000000.0,3]}'
    _, reg, draw = _drawn()
    floats = {"big": 1e22, "small": 1e-7, "third": 1 / 3, "huge": 1.7976931348623157e308, "tiny": 5e-324}
    for delta in (DELTA, 0.05, 0.1):
        assert V.audit_record(**_args(reg, draw, statement=floats, delta=delta))["verifiers"]["agree"]
    assert not V.audit_record(**_args(reg, draw, statement={"x": -0.0}))["verifiers"]["agree"]


def test_a_stream_rules_disagreement_stops_the_stream(monkeypatch):
    """A stream's own rules decide each op by both, on what they hold of the stream that the op reads: where the Lean
    decides otherwise, or holds another state after it, the op is refused (`disagree`, naming both), and so is every op
    after it."""
    stream, reg, draw = _drawn()
    assert stream._rule_entries == [{"order": 0, "registration": S._registration_token(reg), "draw": S._draw_token(draw),
                                     "proved": len(draw.proved), "drawn_at": stream._entries[0].drawn_at,
                                     "delivered": False, "session": None, "final": False}]
    _lean_answering(monkeypatch, "stream", lambda got: {**got, "decision": {"ok": "late"}})
    with pytest.raises(S.Refused) as e:
        stream.deliver(draw, [SESSION])
    v = e.value.verdict
    assert v.codes == (V.DISAGREE,) and v.detail[V.DISAGREE]["lean"]["decision"] == {"ok": "late"}
    assert v.verifiers["python"]["decision"] == {"unchecked": "now"}
    monkeypatch.undo()
    for op in (lambda: stream.deliver(draw, [SESSION]), stream.ready):
        with pytest.raises(S.Refused) as e:
            op()
        assert e.value.verdict is v
    stream = _stream()
    _lean_answering(monkeypatch, "stream", lambda got: {**got, "state": got["state"].replace('"failures":0', '"failures":1')})
    with pytest.raises(S.Refused) as e:
        stream.register(_record(), _ports(6))
    assert e.value.verdict.codes == (V.DISAGREE,) and stream.failures == 0 and stream.log == ()


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
