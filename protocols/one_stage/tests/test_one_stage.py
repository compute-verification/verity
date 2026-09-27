"""One-stage audit: the demo partition query, the registration record, the draw and proved-unit checks, the profile."""

from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest

from verity.ir.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ir.program import Program
from verity.ir.types import Array, Value
from verity_one_stage import audit, draw as D, partition as P, registration as R

XOR8 = PrimitiveDefinition("OneStageTestXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("OneStageTestPair", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, x[0], x[1]), register=False))
ROOTS = {"x": "ab" * 64, "out": "cd" * 64}
LEAVES = {"x": 6, "out": 6}


def _registration(n: int = 6, **kw):
    prog = P.population_program(PAIR, n)
    q = P.template_instance_query(PAIR.id)
    return prog, q, R.record("frame-v3-sha512/hm96-sha512", ROOTS, LEAVES, prog, q, registered_utc="2026-09-27T05:40:00Z", **kw)


def test_the_query_makes_each_instance_one_unit():
    prog, q, _ = _registration(6)
    assert P.units(prog, q) == 6
    assert P.units(P.population_program(PAIR, 1), q) == 1


def test_the_query_refuses_what_it_does_not_describe():
    prog, q, _ = _registration(3)
    for bad in ({**q, "template": "Other_v1"}, {**q, "version": 1}, {**q, "width_rule": [16, 32, 0]}, {**q, "name": "Q_word"}):
        with pytest.raises(P.QueryRefused):
            P.units(prog, bad)
    other = CompositeDefinition("OneStageTestMixed", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, B.call(PAIR, x), x[0]), register=False)
    with pytest.raises(P.QueryRefused):
        P.units(Program(bind(other)), q)


def test_partition_digest_is_the_named_query_over_the_program():
    prog, q, _ = _registration(6)
    obj = P.partition(prog, q)
    assert set(obj) == {"program", "query"} and len(obj["program"]) == 128
    assert obj["query"] == {"name": "Q_template_instance", "version": 0, "template": PAIR.id, "width_rule": None}
    assert P.digest(obj) == P.digest(dict(reversed(list(obj.items()))))
    assert P.digest(obj) != P.digest(P.partition(P.population_program(PAIR, 7), q))


def test_registration_is_checked_by_content():
    prog, q, rec = _registration(6)
    assert R.check(rec, prog, q).ok
    assert R.digest(rec) == R.digest(dict(rec)) and len(R.digest(rec)) == 128
    for key, value, code in (("population", 7, "population"), ("partition_digest", "0" * 128, "partition-digest"),
                             ("leaves", {"x": 6}, "leaves"), ("format", "x", "format")):
        assert code in R.check({**rec, key: value}, prog, q).codes
    assert "partition" in R.check(rec, P.population_program(PAIR, 5), q).codes
    assert "keys" in R.check({**rec, "extra": 1}, prog, q).codes


def test_registration_holds_the_public_file():
    _, _, rec = _registration(6)
    assert R.matches_public(rec, "frame-v3-sha512/hm96-sha512", ROOTS, 6).ok
    assert R.matches_public(rec, "frame-v3-sha512/hm96-sha512", {**ROOTS, "x": "ef" * 64}, 6).codes == ("roots",)
    assert R.matches_public(rec, "frame-v3/blake3", ROOTS, 5).codes == ("scheme", "population")


def test_stand_in_is_recorded():
    _, _, rec = _registration(6, stand_in={"of": "vllm-v1 run root", "root": "7a" * 32})
    assert rec["commitment"]["stand_in"]["of"] == "vllm-v1 run root"


def test_draw_checks():
    _, _, rec = _registration(6)
    law = D.parse_law("subset:2")
    good = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    assert D.check_draw(good, rec, law).ok
    assert D.check_draw({**good, "units": [4, 1]}, rec, law).codes == ("units-order",)
    assert D.check_draw({**good, "units": [1, 6]}, rec, law).codes == ("units-range",)
    assert D.check_draw({**good, "units": [1]}, rec, law).codes == ("units-count",)
    assert D.check_draw({**good, "population": 5}, rec, law).codes == ("population",)
    assert D.check_draw(good, rec, D.parse_law("subset:3")).codes == ("law",)
    assert D.check_draw(None, rec, law).codes == ("not-a-draw",)
    assert D.check_draw({"law": "bernoulli", "population": 6, "p": "1/2", "units": [0, 5]}, rec, D.parse_law("bernoulli:1/2")).ok
    for bad in ("subset", "subset:x", "bernoulli:3/2", "beta:1"):
        with pytest.raises(ValueError):
            D.parse_law(bad)


def test_proved_units_are_the_draw():
    good = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    assert D.check_proved(good, [1, 4], 2).ok
    assert D.check_proved(good, [1, 5], 2).codes == ("units",)
    assert D.check_proved(good, [1, 4, 5], 3).codes == ("instances", "units")


def test_profile_counts_wrong_units():
    _, _, rec = _registration(6)
    d = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    p = audit.profile(rec, d, 0.25)
    assert p.sizes == (6,) and p.drawn == {"unit": 2} and p.levels[0].query == rec["partition_digest"]
    assert p.accept(1) == Fraction(comb(5, 2), comb(6, 2)) and p.accept(5) == 0
    assert 0 < audit.wrong_units_bound(p) < 6


def test_audit_record_needs_every_check_and_verdict():
    _, _, rec = _registration(6)
    d = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    law = D.parse_law("subset:2")
    ok = audit.audit_record(rec, d, law, [1, 4], 2, {"circuit_sha512": "c8"}, {"m0": {"accepted": True}}, 0.25)
    assert ok["accepted"] and ok["complete"] and ok["profile"]["levels"][0]["draw"] == {"subset": 2}
    assert ok["proved_units"] == [1, 4] and ok["law"] == law and ok["wrong_units"]["of"] == 6
    part = audit.audit_record(rec, d, law, [1, 4], 2, {}, {"m0": {"accepted": True}, "lean": {"accepted": None}}, 0.25)
    assert part["accepted"] and not part["complete"]
    for verdicts, proved in (({"m0": {"accepted": False}}, [1, 4]), ({"m0": {"accepted": True}}, [1, 5]), ({}, [1, 4])):
        r = audit.audit_record(rec, d, law, proved, 2, {}, verdicts, 0.25)
        assert not r["accepted"] and r["profile"] is None
