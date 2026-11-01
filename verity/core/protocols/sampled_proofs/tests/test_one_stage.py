"""One-stage audit: the demo partition query, the registration record, the sample and proved-block checks, the profile."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from itertools import islice
from math import comb
from pathlib import Path

import pytest

import verity.ml.circuits.ops.prims  # noqa: F401  (with verity.ml.circuits.ops.scalar, the primitives partition_vectors.json's descriptor names)
import verity.ml.circuits.ops.scalar  # noqa: F401
from verity.ml.circuits import codec, partition_object as PO
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, Registry, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as D, partition as P, registration as R

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
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, schema, leaves, "r-test")}
            for p, schema, leaves in (("x", "hm96-sha512/row/v1", n), ("out", "u16", 2 * n))}


def _registration(n: int = 6, window=SERVED, version: int = 1):
    prog = P.population_program(PAIR, n, version=version)
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


XOR64 = PrimitiveDefinition("OneStageTestXor64", 1, (("a", Value(64)), ("b", Value(64))), Value(64),
                            lambda a, b: a ^ b, register=False)
#: A template of one batch of `N` 64-bit XORs: an instance returns `64 N` bits.
WIDE = CompositeDefinition("OneStageTestWide", 0, ("N",),
                           lambda S: ((("x", Array(S.N, Value(64))), ("y", Array(S.N, Value(64)))), Array(S.N, Value(64))),
                           lambda B, S, x, y: B.batch(XOR64, x, y, axes=(0, 0)), register=False)


def test_the_size_classes_are_the_powers_of_two_from_16_to_the_widest():
    assert P.UNIT_CLASSES == tuple(16 * 2 ** i for i in range(15)) and P.UNIT_CLASSES[-1] == P.UNIT_CLASS_MAX == 262_144
    assert [P.unit_class(b) for b in (0, 16, 17, 24, 1024, 1025, 262_144, 262_145)] == \
        [16, 16, 32, 32, 1024, 2048, 262_144, None]


def test_a_unit_wider_than_the_widest_class_or_a_template_of_no_gate_is_refused():
    """Each proof block's committed bits fit a public size class, the least that holds them, and a template has a gate: a
    template instance of 262,144 bits (the widest class, `Gemm_v2{K=2048,N=16384}`'s in the measured widths) is taken, one
    of 262,208 bits and one of no gate are refused; a Call whose first proof block at `X = 16` returns 16 bits and holds a
    constant a later proof block reads commits 24 bits, class 32, and is taken, as at `X = 8`, where that constant is in no
    proof block."""
    v8x3 = Tuple(Value(8), Value(8), Value(8))
    const = PrimitiveDefinition("OneStageTestConst8", 1, (), Value(8), lambda: 0x5A, register=False)
    empty = bind(CompositeDefinition("OneStageTestEmpty", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                     lambda B, S, x: x[0], register=False))
    widest, over = bind(WIDE, N=4096), bind(WIDE, N=4097)
    assert P.units(P.population_program(widest, 2), P.template_instance_query(widest.id)) == 2
    for t, why in ((over, "commits 262208 bits, more than the widest class, 262144"), (empty, "has no gate")):
        with pytest.raises(P.QueryRefused, match=why):
            P.units(P.population_program(t, 2), P.template_instance_query(t.id))

    def held(B, S, x):
        a = B.call(XOR8, x[0], x[1])
        return tuple_of(a, B.call(XOR8, a, x[1]), B.call(const))

    inner = bind(CompositeDefinition("OneStageTestHeld", 0, (), lambda S: ((("x", Array(2, Value(8))),), v8x3), held,
                                     register=False))

    def shared(B, S, x):
        s = B.call(inner, x)
        return tuple_of(s[0], s[1], B.call(XOR8, s[0], s[2]))

    call = bind(CompositeDefinition("OneStageTestShared", 0, (), lambda S: ((("x", Array(2, Value(8))),), v8x3), shared,
                                    register=False))
    prog = Program(bind(CompositeDefinition("OneStageTestSharedProgram", 0, (), lambda S: call.signature(),
                                            lambda B, S, x: B.call(call, x), register=False)))
    assert P.unit_gates(prog, PO.call_query(16)) == [(2, 5), (5, 6)]
    assert P.unit_gates(prog, PO.call_query(8)) == [(2, 3), (3, 4), (5, 6)]


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


# ---- the population programs' versions: v1 names its templates, v0 (kept for old records) its counts alone -----------------

EPS = bind(CompositeDefinition("OneStageTestEps", 0, ("EPS", "NAME"), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                               lambda B, S, x: B.call(XOR8, x[0], x[1]), register=False), EPS=1e-5, NAME='a"b')
#: the program and registration record digests of `_registration(6, version=0)`: what an old record holds
V0_PAIR_6 = ("179416c1470fbf1c1d679826635d18b475dcf5584b94a9249397c22c7c70f986"
             "23f87f33d710c1b58fe2f55824d85af4af52df20fe3f64231046f23549e84355")
V0_RECORD_6 = ("677e6e684f82c37bb8610bbbfb5732805fc2678b96140443b1f048cb6c35e3b5"
               "4ae03cca27ecd15b541e99f4769335f0aef7c726024f30fbbee1c35cdc764c53")
#: v1's program digests of `population_program(PAIR, 6)` and `mixed_population_program([(PAIR, 3), (XOR3, 5)])`
V1_PAIR_6 = ("e8e632dcd15c1ebfde2fa1250164377934de456593abe8d655f36953ffd6ec09"
             "3aea8ce599b6fe1fb99faaa4a4335e6d9e3c15ff4b79f853f709daf3fb84fccc")
V1_PAIR_3_XOR3_5 = ("dedeaac905807d8b5da6852a43f4295a99d2930029074edbf330484051a36ddc"
                    "01af2b4de0207fb2cc295498f99064fe08c39cd46aae77bd532989cbc509f29a")


def _xor8_registry() -> Registry:
    reg = Registry()
    reg.add(XOR8)
    return reg


def test_v1_names_the_templates_a_population_batches_and_v0_does_not():
    """Two populations of equal counts over different templates are different programs. v0's Definitions (and their roots)
    share one id all the same, which an id-keyed lookup would confuse; v1's ids name the templates."""
    def ids(p):
        return codec.descriptor_id(p.fn), p.root.id

    for version, shared in ((0, True), (1, False)):
        pairs = ((P.population_program(PAIR, 3, version=version), P.population_program(XOR3, 3, version=version)),
                 (P.mixed_population_program([(PAIR, 3), (XOR3, 5)], version=version),
                  P.mixed_population_program([(XOR3, 3), (PAIR, 5)], version=version)))
        for a, b in pairs:
            assert P.program_digest(a) != P.program_digest(b)
            assert (ids(a) == ids(b)) is shared, (version, ids(a), ids(b))
    assert codec.descriptor_id(P.population_program(PAIR, 3, version=0).fn) == "TemplateInstances_v0{N=3}"
    assert codec.descriptor_id(P.mixed_population_program([(PAIR, 3), (XOR3, 5)], version=0).fn) == \
        "TemplatePopulations_v0{counts=[3,5]}"
    assert codec.descriptor_id(P.population_program(PAIR, 3).fn) == 'TemplateInstances_v1{template="OneStageTestPair_v0",N=3}'
    assert codec.descriptor_id(P.mixed_population_program([(PAIR, 3), (XOR3, 5)]).fn) == \
        'TemplatePopulations_v1{templates=["OneStageTestPair_v0","OneStageTestTriple_v0"],counts=[3,5]}'
    for build in (lambda: P.population_program(PAIR, 3, version=2), lambda: P.mixed_population_program([(PAIR, 3)], version=2)):
        with pytest.raises(ValueError, match="no population program v2"):
            build()


def test_v1_digests_are_pinned():
    assert P.program_digest(P.population_program(PAIR, 6)) == V1_PAIR_6
    assert P.program_digest(P.mixed_population_program([(PAIR, 3), (XOR3, 5)])) == V1_PAIR_3_XOR3_5


def test_v1_names_a_template_by_its_descriptor_id_and_decodes_from_its_descriptor():
    """v1's record holds each template's descriptor id, so a float or string static is written canonically (not the readable
    `EPS=1e-05`). The decoder rebuilds either version's body from the descriptor's definitions, at the program's digest, ids
    and proof blocks."""
    prog = P.population_program(EPS, 2)
    assert prog.fn.bindings["template"] == codec.descriptor_id(EPS) != EPS.id
    assert codec.descriptor_id(prog.fn) == \
        'TemplateInstances_v1{template="OneStageTestEps_v0{EPS={\\"f64\\":\\"0x1.4f8b588e368f1p-17\\"},NAME=\\"a\\\\\\"b\\"}",N=2}'
    one, both = P.template_instance_query(EPS), P.template_instances_query([PAIR.id, EPS])
    cases = [(prog, one), (P.population_program(EPS, 2, version=0), one)]
    cases += [(P.mixed_population_program([(PAIR, 3), (EPS, 2)], version=v), both) for v in P.POPULATION_VERSIONS]
    for p, q in cases:
        back = codec.decode_program(json.loads(json.dumps(codec.encode_program(p))), _xor8_registry())
        assert (back.program_fn_id, back.root.id, codec.program_digest(back)) == \
            (codec.descriptor_id(p.fn), p.root.id, P.program_digest(p))
        assert P.ranges(back, q) == P.ranges(p, q) and P.digest(P.partition(back, q)) == P.digest(P.partition(p, q))


def test_an_old_v0_record_still_decodes_and_verifies():
    """A registration made with v0's program, at the digests an old record holds, passes R1-R6 against the verifier's v0
    copy, built or decoded from its descriptor; v1's program of the same population is another program, which it refuses."""
    old, q, rec = _registration(6, version=0)
    assert (rec["program"], R.digest(rec)) == (V0_PAIR_6, V0_RECORD_6)
    decoded = codec.decode_program(json.loads(json.dumps(codec.encode_program(old))), _xor8_registry())
    for prog in (old, decoded):
        assert _check(rec, prog, q).ok
    assert "R2-program" in _check(rec, P.population_program(PAIR, 6), q).codes


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


def test_a_root_with_a_trailing_newline_is_refused_at_r3():
    """`$` matches before a final newline, so a root is held to hex by `fullmatch`: Lean's `isHex` refuses the newline too."""
    prog, q, rec = _registration(6)
    roots = [dict(rec["roots"][0], root=rec["roots"][0]["root"] + "\n"), *rec["roots"][1:]]
    assert _check({**rec, "roots": roots}, prog, q).codes == ("R3-root",)


def test_registration_refusals():
    prog, q, rec = _registration(6)
    cases = [({"format": "verity/registration/v0"}, "R1-protocol"), ({"law": {"law": "subset", "k": 3}}, "R1-law"),
             ({"window": {"kind": "guess"}}, "window"), ({"program": "0" * 128}, "R2-program"),
             ({"query": "0" * 128}, "R2-query"), ({"partition": "0" * 128}, "R2-partition"), ({"population": 7}, "R2-population"),
             ({"scheme": "frame-v3/blake3"}, "R3-scheme"), ({"roots": rec["roots"][:1]}, "R3-ports"),
             ({"roots": [dict(rec["roots"][0], leaves=5), rec["roots"][1]]}, "R3-leaves"),
             ({"roots": [dict(rec["roots"][0], domain="00" * 64), rec["roots"][1]]}, "R3-domain"),
             ({"leaf_layer": "ff" * 64}, "R4-leaf-layer"), ({"leaf_layer": None}, "R4-leaf-layer"), ({"anchors": ["weights"]}, "R5-anchors"),
             ({"previous": "11" * 64}, "R6-previous")]
    for change, code in cases:
        assert code in _check({**rec, **change}, prog, q).codes, code
    assert "R2-partition" in _check(rec, P.population_program(PAIR, 5), q).codes
    assert "keys" in _check({**rec, "extra": 1}, prog, q).codes
    assert _check({**rec, "previous": R.digest(rec)}, prog, q, log=[rec]).codes == ("R6-window",)
    both = {"roots": [dict(rec["roots"][0], domain="00" * 64), dict(rec["roots"][1], leaves=5)]}
    ports = _ports(prog, q, P.units(prog, q))
    codes = {_check({**rec, **both}, prog, q, ports=dict(order)).codes for order in (ports.items(), reversed(ports.items()))}
    assert len(codes) == 1 and set(next(iter(codes))) == {"R3-domain", "R3-leaves"}


def test_window_domains_bind_program_partition_port_schema_leaves_and_run():
    from verity.ml.commitments.identity import identity_digest_sha512
    from verity.ml.commitments.indexed import RangeIndexedDomain
    from verity.ml.commitments.merkle import CommitmentDomain

    base = ("a" * 128, "b" * 128, "x", "hm96-sha512/row/v1", 6, "r1")
    d = R.window_domain(*base)
    binding = identity_digest_sha512("verity/one-stage/served-domain/v1", dict(zip(("program", "partition", "port", "schema", "leaves", "run"), base)))
    assert len(d) == 128 and d == CommitmentDomain(binding, -1, RangeIndexedDomain(0, 6), hash="sha512").domain_id.hex()
    for i, other in ((0, "c" * 128), (1, "c" * 128), (2, "cs"), (3, "u16"), (4, 7), (5, "r2")):
        assert R.window_domain(*base[:i], other, *base[i + 1:]) != d


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
    for bad in ("subset", "subset:x", "subset:0", "subset:00", "bernoulli:3/2", "bernoulli:1/0", "beta:1", "subset:\u0663",
                "work:\u00b2", "bernoulli:1/\u0662"):
        with pytest.raises(ValueError):
            D.parse_law(bad)
    for p in ("\u0661/\u0662", "\u00b9/2"):
        drawn = {"law": "bernoulli", "population": 6, "p": p, "units": [0, 5]}
        assert D.check_draw(drawn, rec, {"law": "bernoulli", "p": p}).codes == ("p",)


def test_the_rate_law_is_its_rate_rounded_up_to_its_tables():
    """`parse_rate` reads `rate:NUM/DEN[:T]` at 0 < NUM <= DEN and T >= 1 in ASCII digits, and `rate_count` is ⌈q·n⌉ for
    q = NUM/DEN rounded up to the next multiple of T and no further: q·n <= K < q·n + T (the Lean's `Rate.count_ge`)."""
    assert [D.parse_rate(t) for t in ("rate:1/3", "rate:1/3:4", "rate:01/02:003", "rate:5/5")] == [
        (1, 3, 1), (1, 3, 4), (1, 2, 3), (5, 5, 1)]
    for bad in ("rate:0/3", "rate:4/3", "rate:1/0", "rate:1/2:0", "rate:1/2:", "rate:1/2:3:4", "rate:1/2/3", "rate:\u0661/2",
                "rate:1/2:\u0663", "Rate:1/2", "rate:/2", "rate", "subset:3", "bernoulli:1/2", 5, None):
        assert D.parse_rate(bad) is None, bad
    for num, den, t in ((1, 3, 1), (1, 3, 4), (2, 7, 3), (5, 5, 2), (1, 100, 8)):
        for n in range(60):
            k = D.rate_count(num, den, t, n)
            assert k % t == 0 and num * n <= den * k < num * n + den * t, (num, den, t, n, k)
    assert (D.rate_count(1, 3, 4, 20), D.rate_count(1, 5, 4, 12), D.rate_count(1, 3, 2, 60)) == (8, 4, 20)


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
    assert 0 < p.bound() < 6


def _v0(registration, draw, delta):
    """The record's `profile` and `wrong_units` as `audit.py` computed them before the profile was one type (`main` at
    6746f408), kept as the oracle for the bytes recorded runs hold, but for a Bernoulli sampling design's bound: then the linear
    relaxation (`worst_case`), now the exact one the Lean computes (`Bernoulli.count_bound`, `Bound.bernoulliBound`)."""
    from verity.ml.circuits.partition import Level
    from verity.core.protocols.profile import Bernoulli, IntegrityProfile, Stratified, Stratum, Subset
    n, law = registration["population"], D.law_of(draw)
    core = (Stratified.uniform(n, law["k"]) if law["law"] == "subset" else Stratified(tuple(
        Stratum(s["template"], sum(hi - lo for lo, hi in s["units"]), s["k"]) for s in law["strata"]))
        if law["law"] == "stratified" else None)
    if draw["law"] == "stratified":
        drawn = {s["template"]: sum(1 for u in draw["units"] if any(lo <= u < hi for lo, hi in s["units"]))
                 for s in draw["strata"]}
        prof = {**core.to_json(), "delta": delta, "drawn": drawn}
        single = None
    else:
        single = IntegrityProfile((Level("unit", registration["partition"]),), (n,),
                                  {"unit": Subset(draw["k"]) if draw["law"] == "subset" else Bernoulli(Fraction(draw["p"]))},
                                  delta, programs=(registration["program"],), drawn={"unit": len(draw["units"])})
        prof = single.to_json()
    out = {"delta": delta, "of": n}
    rule = "except with probability delta, an accepted transcript has at most `bound` wrong units"
    if core is None:
        out["bound"], out["rule"] = Bernoulli(Fraction(draw["p"])).count_bound(n, delta), rule
    else:
        out["bound"], out["rule"] = core.count_bound(delta), rule
        if draw["law"] == "stratified":
            out["per_template"] = {s.name: core.stratum_bound(s.name, delta) for s in core.strata}
            out["rule"] += ", and at most `per_template[t]` of them in template t"
    return prof, out


def test_the_record_states_the_one_profile_in_its_recorded_bytes():
    """One profile type for every sampling design leaves the record's `profile` and `wrong_units` byte for byte as before (`_v0`)."""
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64}
    _, _, rec = _registration(6)
    d = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    r = audit.audit_record(rec, R.receipt(rec, "verifier-test"), d, LAW, [1, 4], 2, {}, {"lean": lean}, 0.25)
    assert json.dumps([r["profile"], r["wrong_units"]]) == json.dumps(list(_v0(rec, d, 0.25)))
    prog, q, rule, srec = _stratified_registration()
    law = D.derive(prog, q, rule)
    sd = {"law": "stratified", "population": 6, "k": 4, "strata": law["strata"], "units": [1, 3, 4, 5]}
    r = audit.audit_record(srec, R.receipt(srec, "verifier-test"), sd, law, [1, 3, 4, 5], 4, {}, {"lean": lean}, 0.25)
    assert json.dumps([r["profile"], r["wrong_units"]]) == json.dumps(list(_v0(srec, sd, 0.25)))
    for p in ("1/3", "1/2", "1"):
        bd = {"law": "bernoulli", "population": 6, "p": p, "units": [0, 2]}
        for delta in (0.25, 0.01):
            new = [audit.profile(rec, bd, delta).to_json(), audit.wrong_units(rec, bd, delta)]
            assert json.dumps(new) == json.dumps(list(_v0(rec, bd, delta))), (p, delta)
    with pytest.raises(ValueError):
        audit.profile(rec, {"law": "nested", "units": []}, 0.25)


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
    part = audit.audit_record(rec, rcpt, d, law, [1, 4], 2, {}, {"m0": {"accepted": True}, "lean": {**lean, "accepted": None}}, 0.25)
    assert not part["accepted"] and not part["complete"] and part["profile"] is None and part["integrity_profile"] is None
    bad_receipt = dict(rcpt, registration="00" * 64)
    for r_, verdicts, proved in ((rcpt, {"lean": lean, "m0": {"accepted": False}}, [1, 4]), (rcpt, {"lean": lean}, [1, 5]),
                                 (rcpt, {}, [1, 4]), (rcpt, {"m0": {"accepted": True}}, [1, 4]),
                                 (rcpt, {"lean": {**lean, "accepted": False}}, [1, 4]), (bad_receipt, {"lean": lean}, [1, 4])):
        r = audit.audit_record(rec, r_, d, law, proved, 2, {}, verdicts, 0.25)
        assert not r["accepted"] and r["profile"] is None and r["integrity_profile"] is None


def test_audit_record_holds_the_served_draw_and_the_sessions_roots():
    """The sessions must have been served exactly the verifier's sample, and each session's root must be the registered one."""
    _, _, rec = _registration(6)
    d = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64}
    args = (rec, R.receipt(rec, "verifier-test"), d, D.parse_law("subset:2"), [1, 4], 2, {}, {"lean": lean}, 0.25)
    assert audit.audit_record(*args, served=[4, 1], session_roots=dict(ROOTS))["accepted"]
    r = audit.audit_record(*args, served=[1, 3])
    assert not r["accepted"] and r["checks"]["served"]["codes"] == ["served"] and r["profile"] is None
    r = audit.audit_record(*args, session_roots={**ROOTS, "x": "ef" * 64})
    assert not r["accepted"] and r["checks"]["roots"]["detail"]["ports"] == ["x"] and r["profile"] is None
    for roots in ({}, {"y": ROOTS["x"]}):
        assert not audit.audit_record(*args, session_roots=roots)["accepted"]


def test_a_record_over_an_explicit_partition_object():
    """A record made before the canonical envelope names its own object; the verifier checks it against that object,
    whose program must still be the verifier's."""
    prog = P.population_program(PAIR, 6)
    q = P.template_instance_query(PAIR.id)
    old = {"program": P.program_digest(prog), "query": {"name": q["name"], "version": 0, "template": PAIR.id, "width_rule": None}}
    ports = _ports(prog, q, 6)
    roots = [{"port": p, "root": ROOTS[p], **ports[p]} for p in ports]
    rec = R.record(prog, q, LAW, SCHEME, roots, SERVED, leaf_layer="ee" * 64, partition=old)
    assert rec["partition"] == P.digest(old) != P.digest(P.partition(prog, q))
    assert "R2-partition" in _check(rec, prog, q).codes
    assert R.check(rec, prog, q, LAW, SCHEME, ports, leaf_layer="ee" * 64, partition=old).ok
    assert "R2-program" in R.check(rec, prog, q, LAW, SCHEME, ports, leaf_layer="ee" * 64, partition={**old, "program": "0" * 128}).codes


def test_the_wrong_unit_bound_matches_worst_case_and_scales():
    """The profile's Subset(k) bound is the exact count, the generic worst_case rounded down, and serves N = 183,680."""
    from verity.ml.circuits.partition import Level
    from verity.core.protocols.profile import IntegrityProfile, Subset
    for n, k, delta in ((1024, 256, 2.0 ** -20), (1024, 16, 0.01), (344, 128, 2.0 ** -20), (6, 2, 0.25), (50, 50, 0.01)):
        p = IntegrityProfile((Level("unit"),), (n,), {"unit": Subset(k)}, delta)
        assert p.bound() == int(p.worst_case(lambda m: float(m))), (n, k, delta)
    big = IntegrityProfile((Level("unit"),), (183680,), {"unit": Subset(256)}, 2.0 ** -20)
    assert 9600 < big.bound() < 9750   # (1 - f)^256 = 2^-20 gives f ~ 5.27%, ~9,680 of 183,680


# ---- the stratified sampling design: strata per template, k_s = max(1, round(k n_s / n)) ---------------------------------------

def _template(i: int):
    """A distinct one-input test template (its own descriptor id)."""
    return bind(CompositeDefinition(f"OneStageTestT{i}", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                    lambda B, S, x, i=i: B.call(XOR8, x[i % 2], x[(i + 1) % 2]), register=False))


def test_strata_group_units_by_template_in_order_of_first_unit():
    prog = P.mixed_population_program([(XOR3, 2), (PAIR, 3), (XOR3, 1)])
    q = P.template_instances_query([PAIR.id, XOR3.id])
    assert P.strata(prog, q) == [(XOR3.id, [[0, 2], [5, 6]]), (PAIR.id, [[2, 5]])]
    merged = P.mixed_population_program([(PAIR, 2), (PAIR, 3), (XOR3, 1)])
    assert P.strata(merged, q) == [(PAIR.id, [[0, 5]]), (XOR3.id, [[5, 6]])]
    assert P.strata(P.population_program(PAIR, 4), P.template_instance_query(PAIR.id)) == [(PAIR.id, [[0, 4]])]


def test_the_stratified_rule_rounds_half_up_with_a_floor_of_one():
    assert [D.stratum_k(5, s, 4) for s in (1, 2, 3, 4)] == [1, 3, 4, 5]        # 1.25, 2.5 -> 3, 3.75, 5
    assert D.stratum_k(1, 1, 100) == 1 and D.stratum_k(5, 3, 8) == 2 and D.stratum_k(4, 1, 8) == 1
    assert all(1 <= D.stratum_k(k, s, n) <= s for n in range(1, 30) for s in range(1, n + 1) for k in range(1, n + 1))


def test_derive_expands_the_rule_over_the_verifiers_program():
    prog = P.mixed_population_program([(XOR3, 2), (PAIR, 3), (XOR3, 1)])
    q = P.template_instances_query([PAIR.id, XOR3.id])
    law = D.derive(prog, q, D.parse_law("stratified:4"))
    assert law == {"law": "stratified", "k": 4, "strata": [{"template": XOR3.id, "units": [[0, 2], [5, 6]], "k": 2},
                                                          {"template": PAIR.id, "units": [[2, 5]], "k": 2}]}
    assert D.derive(prog, q, law) == law and D.derive(prog, q, LAW) == LAW and D.strata_problem(law, 6) is None
    for bad in ("stratified:0", "stratified:x"):
        with pytest.raises(ValueError):
            D.parse_law(bad)
    with pytest.raises(ValueError):
        D.derive(prog, q, {"law": "stratified", "k": 7})
    with pytest.raises(P.QueryRefused):
        D.derive(prog, P.template_instance_query(PAIR.id), law)


def _stratified_registration():
    prog = P.mixed_population_program([(XOR3, 2), (PAIR, 3), (XOR3, 1)])
    q = P.template_instances_query([PAIR.id, XOR3.id])
    ports = _ports(prog, q, 6)
    roots = [{"port": p, "root": ROOTS[p], **ports[p]} for p in ports]
    rule = D.parse_law("stratified:4")
    return prog, q, rule, R.record(prog, q, rule, SCHEME, roots, SERVED, leaf_layer="ee" * 64)


def test_r1_holds_the_registered_strata_to_the_verifiers_own():
    prog, q, rule, rec = _stratified_registration()
    assert rec["law"]["law"] == "stratified" and [s["k"] for s in rec["law"]["strata"]] == [2, 2]
    assert _check(rec, prog, q, law=rule).ok and _check(rec, prog, q, law=rec["law"]).ok
    strata = rec["law"]["strata"]
    tampered = [dict(rec["law"], strata=[{**strata[0], "k": 1}, strata[1]]),                          # a stratum's k
                dict(rec["law"], strata=[{**strata[0], "units": [[0, 3]]}, {**strata[1], "units": [[3, 6]]}]),  # its proof blocks
                dict(rec["law"], strata=list(reversed(strata))),                                       # their order
                dict(rec["law"], k=5), {"law": "subset", "k": 4}]
    for law in tampered:
        assert _check({**rec, "law": law}, prog, q, law=rule).codes == ("R1-law",), law
    assert _check(rec, prog, q, law=D.parse_law("stratified:3")).codes == ("R1-law",)
    assert _check({**rec, "law": {**rec["law"], "k": 4.0}}, prog, q, law=rule).codes == ("R1-law",)   # 4.0 is not 4
    two = D.derive(prog, q, D.parse_law("stratified:2"))                  # k_s = 1, 1
    rec2 = {**rec, "law": two}
    assert [s["k"] for s in two["strata"]] == [1, 1] and _check(rec2, prog, q, law=D.parse_law("stratified:2")).ok
    as_bool = dict(two, strata=[{**two["strata"][0], "k": True}, two["strata"][1]])   # `true` is not 1, as Lean reads it
    assert _check({**rec, "law": as_bool}, prog, q, law=D.parse_law("stratified:2")).codes == ("R1-law",)
    assert _check(rec, prog, q, law=LAW).codes == ("R1-law",)


def test_stratified_draw_checks():
    prog, q, rule, rec = _stratified_registration()
    law = D.derive(prog, q, rule)
    good = {"law": "stratified", "population": 6, "k": 4, "strata": law["strata"], "units": [1, 3, 4, 5]}
    assert D.check_draw(good, rec, law).ok and D.law_of(good) == law
    assert D.check_draw({**good, "units": [0, 1, 3, 4]}, rec, law).ok         # 2 and 2: another valid draw
    assert D.check_draw({**good, "units": [0, 1, 3, 5]}, rec, law).codes == ("strata-count",)
    assert D.check_draw({**good, "units": [1, 3, 4, 6]}, rec, law).codes == ("units-range", "strata-count")
    assert D.check_draw({**good, "k": 5}, rec, law).codes == ("law", "strata")
    assert D.check_draw(good, rec, rule).codes == ("law",)            # the rule unexpanded is not the verifier's sampling design
    two = D.derive(prog, q, D.parse_law("stratified:2"))                  # k_s = 1, 1
    ones = [{**two["strata"][0], "k": True}, two["strata"][1]]           # `true` is not 1, as the Lean verifier reads it
    assert D.strata_problem(two, 6) is None and D.strata_problem({"k": 2, "strata": ones}, 6) is not None
    d2 = {"law": "stratified", "population": 6, "k": 2, "strata": ones, "units": [0, 2]}
    assert D.check_draw(d2, rec, two).codes == ("law", "strata")
    assert D.check_draw({**good, "strata": [{**s, "k": s["k"] * 1.0} for s in law["strata"]]}, rec, law).codes[0] == "law"
    assert D.check_draw({k: v for k, v in good.items() if k != "strata"}, rec, law).codes == ("keys",)
    bad_strata = [{**law["strata"][0], "units": [[0, 2], [2, 3]]}, law["strata"][1]]   # adjacent ranges not merged
    assert "strata" in D.check_draw({**good, "strata": bad_strata}, rec, law).codes
    for problem in ([law["strata"][0]], [law["strata"][0], {**law["strata"][1], "template": XOR3.id}],
                    [{**law["strata"][0], "extra": 1}, law["strata"][1]]):
        assert D.strata_problem({"k": 4, "strata": problem}, 6) is not None


def test_one_stratified_draw_splits_over_several_tables():
    """A session of J tables splits one sample (Lean `Draw.tablesWhy`): a stratified sample whose sum of k_s is a positive
    multiple of J, as a subset sample's k must be; a Bernoulli or work sampling design never, whatever its count."""
    prog, q, rule, rec = _stratified_registration()
    law = D.derive(prog, q, rule)                                          # k_s = 2, 2
    good = {"law": "stratified", "population": 6, "k": 4, "strata": law["strata"], "units": [1, 3, 4, 5]}
    assert D.fixed_count(law) == 4
    for tables in (1, 2, 4):
        assert D.tables_problem(law, tables) is None and D.check_draw(good, rec, law, tables=tables).ok
    assert D.tables_problem(law, 3) == "tables: 4 units do not split into 3 equal tables"
    assert D.check_draw(good, rec, law, tables=3).codes == ("tables",)
    assert D.tables_problem({"law": "subset", "k": 4}, 2) is None and D.tables_problem({"law": "subset", "k": 3}, 2)
    assert D.tables_problem({"law": "bernoulli", "p": "1/1"}, 2) == D.TABLES_LAW
    assert D.tables_problem({"law": "work", "k": 4, "strata": law["strata"]}, 2) == D.TABLES_LAW
    assert D.tables_problem({"law": "bernoulli", "p": "1/2"}, 1) is None


def test_audit_record_under_the_stratified_law():
    """The bound is core's exact count_bound over the strata, each template's own bound is reported, and the integrity
    profile carries the whole sampling design."""
    from verity.core.protocols.profile import Stratified, Stratum
    prog, q, rule, rec = _stratified_registration()
    law = D.derive(prog, q, rule)
    d = {"law": "stratified", "population": 6, "k": 4, "strata": law["strata"], "units": [1, 3, 4, 5]}
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64}
    r = audit.audit_record(rec, R.receipt(rec, "verifier-test"), d, law, [1, 3, 4, 5], 4, {}, {"lean": lean}, 0.25)
    core = Stratified((Stratum(XOR3.id, 3, 2), Stratum(PAIR.id, 3, 2)))
    assert r["accepted"] and r["law"] == law and r["integrity_profile"]["law"] == law
    assert r["profile"] == {**core.to_json(), "delta": 0.25, "drawn": {XOR3.id: 2, PAIR.id: 2}}
    assert r["wrong_units"]["bound"] == core.count_bound(0.25) and r["wrong_units"]["of"] == 6
    assert r["wrong_units"]["per_template"] == {XOR3.id: core.stratum_bound(XOR3.id, 0.25), PAIR.id: core.stratum_bound(PAIR.id, 0.25)}
    wrong = audit.audit_record(rec, R.receipt(rec, "verifier-test"), {**d, "units": [0, 1, 3, 5]}, law, [0, 1, 3, 5], 4, {},
                               {"lean": lean}, 0.25)
    assert not wrong["accepted"] and wrong["checks"]["draw"]["codes"] == ["strata-count"] and wrong["profile"] is None


def test_the_subset_bound_is_now_the_exact_count():
    _, _, rec = _registration(6)
    d = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64}
    r = audit.audit_record(rec, R.receipt(rec, "verifier-test"), d, LAW, [1, 4], 2, {}, {"lean": lean}, 0.25)
    assert r["wrong_units"]["bound"] == max(K for K in range(7) if Fraction(comb(6 - K, 2), comb(6, 2)) >= Fraction(0.25))
    assert "per_template" not in r["wrong_units"]


def test_a4_under_the_stratified_rule():
    """A4 P6's population (six templates, 6,771,765 proof blocks) under `stratified:1024`: k_s = 1, 933, 2, 1, 1, 89, and at
    most 91,063 faulty proof blocks at 2^-20 (uniform subset:1024: 91,051), with a floor in every template."""
    sizes = (287, 6_171_648, 11_480, 287, 287, 587_776)
    temps = [_template(i) for i in range(6)]
    prog = P.mixed_population_program(list(zip(temps, sizes)))
    q = P.template_instances_query([t.id for t in temps])
    law = D.derive(prog, q, D.parse_law("stratified:1024"))
    assert [s["k"] for s in law["strata"]] == [1, 933, 2, 1, 1, 89]
    assert [s["units"] for s in law["strata"]][:2] == [[[0, 287]], [[287, 6_171_935]]]
    assert D.strata_problem(law, sum(sizes)) is None
    core = audit.core_law(law, sum(sizes))
    assert core.count_bound(2.0 ** -20) == 91_063 and core.stratum_bound(temps[0].id, 2.0 ** -20) == 286
    assert core.escape({temps[0].id: 287}) == 0
    reg = {"population": sum(sizes), "partition": "00" * 64, "program": "11" * 64}
    units = sorted(u for s in law["strata"] for u in islice((u for lo, hi in s["units"] for u in range(lo, hi)), s["k"]))
    d = {**law, "population": sum(sizes), "units": units}
    p = audit.profile(reg, d, 2.0 ** -20)
    assert p.strata() == core and p.drawn == {"unit": 1027}
    got = audit.wrong_units(reg, d, 2.0 ** -20)
    assert json.dumps(got) == json.dumps(_v0(reg, d, 2.0 ** -20)[1])
    assert got["bound"] == 91_063 and list(got["per_template"].values()) == [286, 91_016, 11_468, 286, 286, 84_773]


# ---- the partition is core's `verity/partition/v1` (`verity.ml.circuits.partition_object`), byte for byte ------------------------

CORE_VECTORS = Path(__file__).resolve().parents[4] / "ml" / "circuits" / "tests"
QWORD = json.loads((CORE_VECTORS / "partition_vectors.json").read_text())
TEMPLATE = json.loads((CORE_VECTORS / "template_instance_vectors.json").read_text())
V32 = Value(32)
#: the primitives template_instance_vectors.json's descriptors name (a partition reads their ids and types only)
TEMPLATE_REG = Registry()
for _name in ("TAdd32", "TMul32"):
    TEMPLATE_REG.add(PrimitiveDefinition(_name, 1, (("a", V32), ("b", V32)), V32, lambda a, b: (a + b) & 0xFFFFFFFF, register=False))


def _template_program(name: str):
    return codec.decode_program(json.loads(TEMPLATE["descriptors"][name]), TEMPLATE_REG)


def _one_stage(program, q) -> dict:
    obj = P.partition(program, q)
    return {"object": codec.canonical_json(obj), "digest": P.digest(obj), "program": P.program_digest(program),
            "population": P.units(program, q), "ranges": P.ranges(program, q)}


def _core(program, q) -> dict:
    obj, part = PO.build(program, q), PO.evaluate(program, q)
    return {"object": codec.canonical_json(obj), "digest": PO.digest(obj), "program": codec.program_digest(program),
            "population": part.population, "ranges": [(t, lo, n) for _k, t, lo, n in part.ranges]}


def test_the_partition_is_cores_on_the_pinned_q_word_vector():
    """One-stage names any `verity/partition/v1` object as core does, and applies the pinned `Q_word` vector as core does:
    its proof blocks, some committing up to `W = 32` bits, fit the public classes."""
    prog = codec.decode_program(json.loads(QWORD["descriptor"]), REGISTRY)
    q = QWORD["object"]["query"]
    obj = P.partition(prog, q)
    assert codec.canonical_json(obj) == codec.canonical_json(QWORD["object"]) == codec.canonical_json(PO.build(prog, q))
    assert P.program_digest(prog) == QWORD["program_digest"] == codec.program_digest(prog)
    assert P.digest(obj) == QWORD["object_digest"] == PO.digest(PO.build(prog, q))
    assert P.units(prog, q) == QWORD["population"] == PO.evaluate(prog, q).population


@pytest.mark.parametrize("n", range(len(TEMPLATE["cases"])), ids=[c[0] for c in TEMPLATE["cases"]])
def test_the_partition_is_cores_on_the_pinned_template_vectors(n):
    _label, name, q, want = TEMPLATE["cases"][n]
    prog = _template_program(name)
    pinned = {"object": codec.canonical_json(want["object"]), "digest": want["digest"], "program": want["object"]["program"],
              "population": want["population"], "ranges": [(t, lo, c) for _k, t, lo, c in want["ranges"]]}
    assert _one_stage(prog, q) == pinned == _core(prog, q)


def test_the_partition_is_cores_on_this_protocols_programs():
    temps = [_template(i) for i in range(6)]
    pair, both = P.template_instance_query(PAIR.id), P.template_instances_query([PAIR.id, XOR3.id])
    cases = [(P.population_program(PAIR, n), pair) for n in (1, 3, 5, 6, 7)]
    for members in ([(PAIR, 3), (XOR3, 5)], [(PAIR, 3), (XOR3, 4)], [(XOR3, 2), (PAIR, 3), (XOR3, 1)], [(PAIR, 2), (PAIR, 3), (XOR3, 1)]):
        prog = P.mixed_population_program(members)
        cases += [(prog, both), (prog, P.template_instances_query([XOR3.id, PAIR.id]))]
    sizes = (287, 6_171_648, 11_480, 287, 287, 587_776)
    cases.append((P.mixed_population_program(list(zip(temps, sizes))), P.template_instances_query(temps)))
    for prog, q in cases:
        assert _one_stage(prog, q) == _core(prog, q), (prog.root.id, q)


@pytest.mark.parametrize("n", range(len(TEMPLATE["refused"])), ids=[r[0] for r in TEMPLATE["refused"]])
def test_the_partition_refuses_what_core_refuses(n):
    _label, name, q, _why = TEMPLATE["refused"][n]
    with pytest.raises(P.QueryRefused):
        P.units(_template_program(name), q)


def test_a_program_that_is_not_topological_has_no_units():
    """Core's `order_violation`: the population program whose instances read their own outputs is no computation."""
    d = json.loads(TEMPLATE["descriptors"]["population"])
    d["definitions"][d["root"]]["body"]["nodes"][2]["args"][0]["refs"] = {"affine": ["n", 2, 0, 6]}
    looped = codec.decode_program(d, TEMPLATE_REG)
    q = PO.template_instance_query(TEMPLATE["template_ids"]["TScale descriptor id"])
    with pytest.raises(P.QueryRefused) as e:
        P.units(looped, q)
    assert [e.value.detail] == TEMPLATE["order_refused"][1]


@pytest.mark.parametrize("n", range(len(TEMPLATE["malformed"])), ids=[m[0] for m in TEMPLATE["malformed"]])
def test_a_malformed_query_has_no_units_and_no_digest(n):
    _label, q, codes = TEMPLATE["malformed"][n]
    prog = _template_program("population")
    with pytest.raises(P.QueryRefused) as e:
        P.units(prog, q)
    assert e.value.detail["codes"] == codes
    with pytest.raises(ValueError):
        P.digest(P.partition(prog, q))


def test_registration_check_refuses_a_query_core_does_not_digest():
    prog, q, rec = _registration(6)
    v = R.check(rec, prog, {**q, "width_rule": None}, LAW, SCHEME, _ports(prog, q, 6), leaf_layer="ee" * 64)
    assert {"R2-partition", "R2-query"} <= set(v.codes) and "query-keys" in v.detail["R2-partition"]["verifier"]


def test_a_pre_canonical_object_keeps_its_digest():
    """A record made before the canonical envelope names `{program, query}` (benchmarks' `--legacy-partition`); its digest is
    the envelope's rule over that object, as the roots registered with it bind."""
    prog = P.population_program(PAIR, 6)
    old = {"program": P.program_digest(prog), "query": {"name": "Q_template_instance", "version": 0, "template": PAIR.id, "width_rule": None}}
    assert P.digest(old) == hashlib.sha512(b"verity/partition/v1\x00" + codec.canonical_json(old)).hexdigest()


# --- the work sampling design (the sample sampling design's second version; Lean `Flock.Draw.Law.work`) -------------------------------------------------

def test_the_work_rule_sizes_by_work_above_the_floor():
    """`work_k` is Lean's `workK`: ceil(K w n_s / W), at least the floor and at most n_s; so every non-empty stratum samples at
    least min(n_s, f_s) (`floor_le_workK`), at least K w n_s / W or all (`workK_ge`), and at most K + sum_s f_s in all
    (`sum_workK_le`)."""
    assert D.work_k(4, 10, 3, 30, 1) == 3 and D.work_k(4, 0, 3, 30, 1) == 1 and D.work_k(4, 0, 3, 30, 2) == 2
    assert D.work_k(10, 1, 7, 70, 1) == 1 and D.work_k(10, 3, 7, 70, 1) == 3 and D.work_k(1, 1, 5, 1000, 9) == 5
    for works in ((0, 5), (1, 1), (2, 7, 0), (3, 0, 0, 1)):
        for sizes in ((1, 4), (3, 2), (2, 2, 5), (1, 1, 1, 9)):
            if len(sizes) != len(works):
                continue
            total = sum(w * n for w, n in zip(works, sizes))
            for k in range(1, sum(sizes) + 1):
                for floor in (1, 2):
                    ks = [D.work_k(k, w, n, total, floor) for w, n in zip(works, sizes)]
                    assert all(min(n, floor) <= x <= n for x, n in zip(ks, sizes))
                    assert all(x == n or x * total >= k * w * n for x, w, n in zip(ks, works, sizes))
                    assert sum(ks) <= k + floor * len(sizes)


def _work_program():
    prog = P.mixed_population_program([(XOR3, 2), (PAIR, 3), (XOR3, 1)])
    return prog, P.template_instances_query([PAIR.id, XOR3.id])


def test_derive_expands_the_work_rule_over_the_verifiers_table():
    """PAIR does the PoUW work; XOR3 stands for proof blocks credited nothing (dequantization, or filler: proof blocks of padded rows only,
    or declared uncredited), which draw their floor for integrity and add nothing to W."""
    prog, q = _work_program()
    table = {PAIR.id: 10, XOR3.id: 0}
    law = D.derive(prog, q, D.parse_law("work:4"), table)
    assert law == {"law": "work", "k": 4, "strata": [
        {"template": XOR3.id, "units": [[0, 2], [5, 6]], "work": 0, "floor": 1, "k": 1},
        {"template": PAIR.id, "units": [[2, 5]], "work": 10, "floor": 1, "k": 3}]}
    assert D.total_work(law["strata"]) == 30 and D.strata_problem(law, 6) is None
    assert D.derive(prog, q, law, table) == law
    floors = D.derive(prog, q, {"law": "work", "k": 2}, {PAIR.id: {"work": 10, "floor": 2}, XOR3.id: {"work": 0, "floor": 3}})
    assert [s["k"] for s in floors["strata"]] == [3, 2]            # the floors bind: 3 of XOR3's 3, 2 of PAIR's 3
    assert D.parse_law("work:5") == {"law": "work", "k": 5}
    with pytest.raises(ValueError, match="work table"):
        D.derive(prog, q, law)
    with pytest.raises(ValueError, match="names no work"):
        D.derive(prog, q, law, {PAIR.id: 10})
    with pytest.raises(ValueError, match="does some work"):
        D.derive(prog, q, law, {PAIR.id: 0, XOR3.id: 0})
    for bad in ({PAIR.id: -1, XOR3.id: 0}, {PAIR.id: True, XOR3.id: 0}, {PAIR.id: {"work": 1, "floor": 0}, XOR3.id: 0},
                {PAIR.id: {"work": 1, "extra": 1}, XOR3.id: 0}, {PAIR.id: 1.5, XOR3.id: 0}, [PAIR.id]):
        with pytest.raises(ValueError):
            D.work_table(bad)
    for bad in ("work:0", "work:x"):
        with pytest.raises(ValueError):
            D.parse_law(bad)


def _work_registration():
    prog, q = _work_program()
    ports = _ports(prog, q, 6)
    roots = [{"port": p, "root": ROOTS[p], **ports[p]} for p in ports]
    rule, table = D.parse_law("work:4"), {PAIR.id: 10, XOR3.id: 0}
    return prog, q, rule, table, R.record(prog, q, rule, SCHEME, roots, SERVED, leaf_layer="ee" * 64, work_table=table)


def test_r1_holds_a_work_law_to_the_verifiers_own_table():
    prog, q, rule, table, rec = _work_registration()
    assert rec["law"]["law"] == "work" and [s["k"] for s in rec["law"]["strata"]] == [1, 3]
    assert _check(rec, prog, q, law=rule, work_table=table).ok
    assert _check(rec, prog, q, law=rule, work_table={PAIR.id: 10, XOR3.id: 5}).codes == ("R1-law",)
    assert _check(rec, prog, q, law=rule).codes == ("R1-law",)                           # no table: the sampling design doesn't apply
    assert _check(rec, prog, q, law=D.parse_law("stratified:4")).codes == ("R1-law",)


def test_work_draw_checks_and_closures():
    prog, q, rule, table, rec = _work_registration()
    law = D.derive(prog, q, rule, table)
    good = {"law": "work", "population": 6, "k": 4, "strata": law["strata"], "units": [2, 3, 4, 5]}
    assert D.check_draw(good, rec, law).ok and D.law_of(good) == law and D.proved(good) == [2, 3, 4, 5]
    assert D.check_draw({**good, "units": [0, 2, 3, 4]}, rec, law).ok
    assert D.check_draw({**good, "units": [0, 2, 3, 5]}, rec, law).codes == ("strata-count",)
    closed = {**good, "units": [0, 2, 3, 4], "closure": [1, 5]}
    assert D.check_draw(closed, rec, law).ok and D.proved(closed) == [0, 1, 2, 3, 4, 5]
    assert D.check_proved(closed, [0, 1, 2, 3, 4, 5], 6).ok and D.check_proved(closed, [0, 2, 3, 4], 4).codes == ("instances", "units")
    assert D.check_draw({**closed, "closure": [1, 2]}, rec, law).codes == ("closure-drawn",)
    assert D.check_draw({**closed, "closure": []}, rec, law).codes == ("closure",)
    assert D.check_draw({**closed, "closure": [5, 1]}, rec, law).codes == ("closure",)
    stratified = D.derive(prog, q, D.parse_law("stratified:4"))
    assert D.check_draw({"law": "stratified", "population": 6, "k": 4, "strata": stratified["strata"], "units": [0, 2, 3, 5],
                         "closure": [1]}, rec, stratified).codes == ("keys",)           # a closure only on a work sample
    wrong_k = [law["strata"][0], {**law["strata"][1], "k": 2}]
    assert D.strata_problem({"law": "work", "k": 4, "strata": wrong_k}, 6) == f"stratum {PAIR.id!r}: k is not the work rule's"
    no_work = [{**s, "work": 0} for s in law["strata"]]
    assert D.strata_problem({"law": "work", "k": 4, "strata": no_work}, 6) == "a work law's population does no work"
    floor0 = [{**law["strata"][0], "floor": 0}, law["strata"][1]]
    assert D.strata_problem({"law": "work", "k": 4, "strata": floor0}, 6) == f"stratum {XOR3.id!r}: a floor is at least one"
    assert D.strata_problem({"law": "work", "k": 4, "strata": [dict(s, k=s["k"]) for s in stratified["strata"]]}, 6) is not None


def test_audit_record_under_the_work_law():
    """The work sampling design is the stratified sampling design at its sizes, so the profile is core's `Stratified` over the same strata at the
    work sizes; the zero-work stratum is bounded by its floor draw."""
    from verity.core.protocols.profile import Stratified, Stratum
    prog, q, rule, table, rec = _work_registration()
    law = D.derive(prog, q, rule, table)
    d = {"law": "work", "population": 6, "k": 4, "strata": law["strata"], "units": [0, 2, 3, 4]}
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64}
    r = audit.audit_record(rec, R.receipt(rec, "verifier-test"), d, law, [0, 2, 3, 4], 4, {}, {"lean": lean}, 0.25)
    core = Stratified((Stratum(XOR3.id, 3, 1), Stratum(PAIR.id, 3, 3)))
    assert r["accepted"] and r["law"] == law and r["integrity_profile"]["law"] == law
    assert r["profile"] == {**core.to_json(), "delta": 0.25, "drawn": {XOR3.id: 1, PAIR.id: 3}}
    assert r["wrong_units"]["per_template"] == {XOR3.id: core.stratum_bound(XOR3.id, 0.25), PAIR.id: 0}


def test_work_sizes_are_the_lean_verifiers():
    """The sizes `verity/ml/flock/tests/test_lean_verifier.py::test_work_draw_from_the_verifiers_own_program` pins for
    `flock-verify draw --work K`: strata of 3, 3 and 4 proof blocks doing 3, 1 and 0 (W = 12), and its floor cases."""
    sizes, works = (3, 3, 4), (3, 1, 0)
    total = sum(w * n for w, n in zip(works, sizes))
    for k, want in {1: [1, 1, 1], 2: [2, 1, 1], 4: [3, 1, 1], 10: [3, 3, 1]}.items():
        assert [D.work_k(k, w, n, total, 1) for w, n in zip(works, sizes)] == want, k
    for floors, want in (((1, 1, 2), [2, 1, 2]), ((1, 1, 9), [2, 1, 4]), ((1, 3, 1), [2, 3, 1])):
        assert [D.work_k(2, w, n, total, f) for w, n, f in zip(works, sizes, floors)] == want, floors
