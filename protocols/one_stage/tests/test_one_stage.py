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
SCHEME = "frame-v3-sha512/hm96-sha512"
LAW = {"law": "subset", "k": 2}
ROOTS = {"x": "ab" * 64, "out": "cd" * 64}
SERVED = {"kind": "served", "row": 101, "run": "r-test"}


def _ports(prog, q, n: int) -> dict:
    part = P.partition(prog, q)
    return {p: {"leaves": leaves, "domain": R.served_domain(part["program"], P.digest(part), p, schema, leaves, "r-test")}
            for p, schema, leaves in (("x", "hm96-sha512/row/v1", n), ("out", "u16", 2 * n))}


def _registration(n: int = 6, window=SERVED):
    prog = P.population_program(PAIR, n)
    q = P.template_instance_query(PAIR.id)
    ports = _ports(prog, q, n)
    roots = [{"port": p, "root": ROOTS[p], **ports[p]} for p in ports]
    return prog, q, R.record(prog, q, LAW, SCHEME, roots, window, leaf_layer="ee" * 64)


def _check(rec, prog, q, **kw):
    n = P.units(prog, q)
    args = {"law": LAW, "scheme": SCHEME, "ports": _ports(prog, q, n), "leaf_layer": "ee" * 64, **kw}
    return R.check(rec, prog, q, **args)


def test_the_query_makes_each_instance_one_unit():
    prog, q, _ = _registration(6)
    assert P.units(prog, q) == 6
    assert P.units(P.population_program(PAIR, 1), q) == 1


def test_the_query_refuses_what_it_does_not_describe():
    prog, q, _ = _registration(3)
    for bad in ({**q, "params": {"template": "Other_v1"}}, {**q, "version": 1}, {**q, "params": {**q["params"], "X": 16}},
                {**q, "name": "Q_word"}, {**q, "width_rule": None}, {"name": q["name"], "version": 0, "template": PAIR.id}):
        with pytest.raises(P.QueryRefused):
            P.units(prog, bad)
    other = CompositeDefinition("OneStageTestMixed", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, B.call(PAIR, x), x[0]), register=False)
    with pytest.raises(P.QueryRefused):
        P.units(Program(bind(other)), q)


XOR3 = bind(CompositeDefinition("OneStageTestTriple", 0, (), lambda S: ((("x", Array(3, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))


def test_several_templates_one_population():
    prog = P.mixed_population_program([(PAIR, 3), (XOR3, 5)])
    q = P.template_instances_query([PAIR.id, XOR3.id])
    assert P.ranges(prog, q) == [(PAIR.id, 0, 3), (XOR3.id, 3, 5)] and P.units(prog, q) == 8
    assert P.units(prog, P.template_instances_query([XOR3.id, PAIR.id])) == 8
    for bad in (P.template_instances_query([PAIR.id]), P.template_instances_query([PAIR.id, XOR3.id, "Other_v1"]),
                P.template_instances_query([PAIR.id, PAIR.id, XOR3.id]), P.template_instance_query(PAIR.id)):
        with pytest.raises(P.QueryRefused):
            P.units(prog, bad)
    assert P.digest(P.partition(prog, q)) != P.digest(P.partition(P.mixed_population_program([(PAIR, 3), (XOR3, 4)]), q))


def test_partition_digest_is_the_named_query_over_the_program():
    prog, q, _ = _registration(6)
    obj = P.partition(prog, q)
    assert set(obj) == {"format", "program", "query"} and obj["format"] == "verity/partition/v1" and len(obj["program"]) == 128
    assert obj["query"] == {"name": "Q_template_instance", "version": 0, "params": {"template": PAIR.id}}
    assert P.digest(obj) == P.digest(dict(reversed(list(obj.items()))))
    assert P.digest(obj) != P.digest(P.partition(P.population_program(PAIR, 7), q))


def test_registration_v1_passes_r1_to_r6():
    prog, q, rec = _registration(6)
    assert rec["format"] == "verity/registration/v1" and rec["protocol"] == "verity/sampled-proofs/one-stage/v1"
    assert rec["partition"] == P.digest(P.partition(prog, q)) and rec["query"] == R.query_digest(q) and rec["population"] == 6
    assert _check(rec, prog, q).ok
    assert len(R.digest(rec)) == 128 and R.digest(rec) == R.digest(dict(rec))


def test_registration_refusals():
    prog, q, rec = _registration(6)
    cases = [({"format": "verity/registration/v0"}, "R1-protocol"), ({"law": {"law": "subset", "k": 3}}, "R1-law"),
             ({"window": {"kind": "guess"}}, "window"), ({"program": "0" * 128}, "R2-program"),
             ({"query": "0" * 128}, "R2-query"), ({"partition": "0" * 128}, "R2-partition"), ({"population": 7}, "R2-population"),
             ({"scheme": "frame-v3/blake3"}, "R3-scheme"), ({"roots": rec["roots"][:1]}, "R3-ports"),
             ({"roots": [dict(rec["roots"][0], leaves=5), rec["roots"][1]]}, "R3-leaves"),
             ({"roots": [dict(rec["roots"][0], domain="00" * 64), rec["roots"][1]]}, "R3-domain"),
             ({"leaf_layer": "ff" * 64}, "R4-leaf-layer"), ({"anchors": ["weights"]}, "R5-anchors"),
             ({"previous": "11" * 64}, "R6-previous")]
    for change, code in cases:
        assert code in _check({**rec, **change}, prog, q).codes, code
    assert "R2-partition" in _check(rec, P.population_program(PAIR, 5), q).codes
    assert "keys" in _check({**rec, "extra": 1}, prog, q).codes
    assert _check({**rec, "previous": R.digest(rec)}, prog, q, log=[rec]).codes == ("R6-window",)


def test_served_domains_bind_program_partition_port_schema_leaves_and_run():
    base = ("a" * 128, "b" * 128, "x", "hm96-sha512/row/v1", 6, "r1")
    d = R.served_domain(*base)
    assert len(d) == 128
    for i, other in ((0, "c" * 128), (1, "c" * 128), (2, "cs"), (3, "u16"), (4, 7), (5, "r2")):
        assert R.served_domain(*base[:i], other, *base[i + 1:]) != d


def test_receipt_names_the_record():
    _, _, rec = _registration(6)
    r = R.receipt(rec, "verifier-test", received="2026-09-27T06:00:00Z")
    assert r == {"format": "verity/registration-receipt/v1", "registration": R.digest(rec), "verifier": "verifier-test",
                 "received": "2026-09-27T06:00:00Z"}


def test_registration_holds_the_public_file():
    _, _, rec = _registration(6)
    doms = {r["port"]: r["domain"] for r in rec["roots"]}
    assert R.matches_public(rec, SCHEME, ROOTS, doms, 6).ok
    assert R.matches_public(rec, SCHEME, {**ROOTS, "x": "ef" * 64}, doms, 6).codes == ("roots",)
    assert R.matches_public(rec, SCHEME, ROOTS, {**doms, "out": "03" * 64}, 6).codes == ("domains",)
    assert R.matches_public(rec, "frame-v3/blake3", ROOTS, doms, 5).codes == ("scheme", "population")


def test_the_window_says_served_or_stand_in():
    prog, q, rec = _registration(6, window={"kind": "stand-in", "of": "vllm-v1 run root", "root": "7a" * 32})
    assert rec["window"]["kind"] == "stand-in" and _check(rec, prog, q).ok


def test_draw_checks():
    _, _, rec = _registration(6)
    law = D.parse_law("subset:2")
    assert law == LAW
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
    assert p.sizes == (6,) and p.drawn == {"unit": 2} and p.levels[0].query == rec["partition"]
    assert p.accept(1) == Fraction(comb(5, 2), comb(6, 2)) and p.accept(5) == 0
    assert 0 < audit.wrong_units_bound(p) < 6


def test_audit_record_needs_every_check_and_verdict():
    _, _, rec = _registration(6)
    d = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    law = D.parse_law("subset:2")
    rcpt = R.receipt(rec, "verifier-test")
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64}
    ok = audit.audit_record(rec, rcpt, d, law, [1, 4], 2, {"circuit_sha512": "c8"}, {"lean": lean}, 0.25, sessions=["aa" * 64])
    assert ok["accepted"] and ok["complete"] and ok["profile"]["levels"][0]["draw"] == {"subset": 2}
    assert ok["proved_units"] == [1, 4] and ok["law"] == law and ok["wrong_units"]["of"] == 6
    ip = ok["integrity_profile"]
    assert ip["format"] == "verity/integrity-profile/v1" and ip["drawn"] == [1, 4] and ip["registration"] == R.digest(rec)
    assert ip["partition"] == rec["partition"] and ip["evidence"]["verdict"] == "dd" * 64 and ip["evidence"]["sessions"] == ["aa" * 64]
    part = audit.audit_record(rec, rcpt, d, law, [1, 4], 2, {}, {"m0": {"accepted": True}, "lean": {"accepted": None}}, 0.25)
    assert part["accepted"] and not part["complete"] and part["integrity_profile"]["evidence"]["verdict"] is None
    bad_receipt = dict(rcpt, registration="00" * 64)
    for r_, verdicts, proved in ((rcpt, {"m0": {"accepted": False}}, [1, 4]), (rcpt, {"m0": {"accepted": True}}, [1, 5]),
                                 (rcpt, {}, [1, 4]), (bad_receipt, {"m0": {"accepted": True}}, [1, 4])):
        r = audit.audit_record(rec, r_, d, law, proved, 2, {}, verdicts, 0.25)
        assert not r["accepted"] and r["profile"] is None and r["integrity_profile"] is None


def test_a_record_over_an_explicit_partition_object():
    """A record made before the canonical envelope names its own object; the verifier checks it against that object,
    whose program must still be the verifier's."""
    prog = P.population_program(PAIR, 6)
    q = P.template_instance_query(PAIR.id)
    old = {"program": P.program_sha512(prog), "query": {"name": q["name"], "version": 0, "template": PAIR.id, "width_rule": None}}
    ports = _ports(prog, q, 6)
    roots = [{"port": p, "root": ROOTS[p], **ports[p]} for p in ports]
    rec = R.record(prog, q, LAW, SCHEME, roots, SERVED, leaf_layer="ee" * 64, partition=old)
    assert rec["partition"] == P.digest(old) != P.digest(P.partition(prog, q))
    assert "R2-partition" in _check(rec, prog, q).codes
    assert R.check(rec, prog, q, LAW, SCHEME, ports, leaf_layer="ee" * 64, partition=old).ok
    assert "R2-program" in R.check(rec, prog, q, LAW, SCHEME, ports, leaf_layer="ee" * 64, partition={**old, "program": "0" * 128}).codes


def test_the_wrong_unit_bound_matches_worst_case_and_scales():
    """The direct Subset(k) bound equals the generic worst_case where that one can run, and serves N = 183,680."""
    from verity.ir.partition import Level
    from verity.proofs.profile import IntegrityProfile, Subset
    for n, k, delta in ((1024, 256, 2.0 ** -20), (1024, 16, 0.01), (344, 128, 2.0 ** -20), (6, 2, 0.25), (50, 50, 0.01)):
        p = IntegrityProfile((Level("unit"),), (n,), {"unit": Subset(k)}, delta)
        assert abs(audit.wrong_units_bound(p) - p.worst_case(lambda m: float(m))) < 1e-6, (n, k, delta)
    big = IntegrityProfile((Level("unit"),), (183680,), {"unit": Subset(256)}, 2.0 ** -20)
    b = audit.wrong_units_bound(big)
    assert 9600 < b < 9750   # (1 - f)^256 = 2^-20 gives f ~ 5.27%, ~9,680 of 183,680
