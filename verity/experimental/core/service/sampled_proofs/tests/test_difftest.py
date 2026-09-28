"""Sampled proofs' one-stage audit in Lean (`core_sampled_proofs`) against `one_stage`'s Python: the whole protocol through
its own `serve` (`sampled-proofs-one-stage`, a `ProtocolPort`), and its pieces through `sampled-proofs-difftest` (`Port`s),
on drawn programs, laws, records, draws and verdicts, valid ones and one or two changes to each.

The reference is `one_stage` itself (`registration.check`, `draw.derive`, `check_draw`, `audit.audit_record`, `profile`,
`wrong_units`) behind the protocol's decoders, which this file writes as the Lean reads them (`OneStage.paramsOfJson`,
`commitmentOf`, `evidenceOf`), and for the `draw` piece `two_stage.draw_first` on `OwnRandomness` over the given bytes.
The verifier's program is looked up among this file's programs by its descriptor. Inputs
stay inside what Python reads as Lean does: integers that are not booleans where Python compares with `==`, ASCII digits
in a probability, and integer populations in a draw.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from fractions import Fraction
from math import gcd
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity_difftest import protocol as PR
from verity.experimental.protocols.profile import Stratified, Stratum
from verity.experimental.protocols.sampled_proofs.one_stage import audit, draw as DR, partition as P, registration as R
from verity.experimental.protocols.sampled_proofs.two_stage import OwnRandomness, draw_first
from verity.ml.circuits import codec, partition_object as PO
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.types import Array, Value

PACKAGE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

XOR8 = PrimitiveDefinition("CoreSampledProofsXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("CoreSampledProofsPair", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("CoreSampledProofsTriple", 0, (), lambda S: ((("x", Array(3, Value(8))),), Value(8)),
                                  lambda B, S, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
SCHEME = "frame-v3-sha512/hm96-sha512"
LEAF = "ee" * 64
ROOTS = {"out": "cd" * 64, "x": "ab" * 64}
WINDOW = {"kind": "served", "row": 101, "run": "r-test"}
EARLIER = {"kind": "served", "row": 100, "run": "r-test"}
RECEIVED = "2026-10-07T00:00:00Z"


def _json(value):
    """The value as Lean reads it: canonical JSON's, keys sorted."""
    return json.loads(D.canonical(value))


class Setup:
    """A verifier's program and query, with its descriptor and the ports its layout derives."""

    def __init__(self, program, query):
        self.program, self.query = program, query
        self.desc = _json(codec.encode_program(program))
        self.sha512 = PO.program_sha512(program)
        self.population = P.units(program, query)
        self.templates = [t for t, _ in P.strata(program, query)]
        part = P.partition(program, query)
        self.ports = {p: {"domain": R.served_domain(self.sha512, P.digest(part), p, schema, leaves, "r-test"), "leaves": leaves}
                      for p, schema, leaves in (("out", "u16", 2 * self.population),
                                                ("x", "hm96-sha512/row/v1", self.population))}


SETUPS = [Setup(P.population_program(PAIR, n), P.template_instance_query(PAIR.id)) for n in (1, 3, 6)] + [
    Setup(P.mixed_population_program([(PAIR, 2), (TRIPLE, 3)]), P.template_instances_query([PAIR.id, TRIPLE.id])),
    Setup(P.mixed_population_program([(PAIR, 1), (TRIPLE, 2), (PAIR, 2)]), P.template_instances_query([TRIPLE.id, PAIR.id]))]
PROGRAMS = {s.sha512: s.program for s in SETUPS}
PROGRAMS_TEMPLATES = [PAIR.id, TRIPLE.id]
#: Queries a verifier may hold: the setups' own, and ones core refuses or one-stage doesn't apply.
QUERIES = [s.query for s in SETUPS] + [
    {}, PO.query(), PO.query(version=2), {**PO.query(), "params": {"W": 32, "X": 70000}}, P.template_instance_query("Other_v1"),
    P.template_instances_query([PAIR.id]), {"name": "Q_template_instance", "version": 1, "params": {"template": PAIR.id}},
    {"name": "Nope", "version": 0, "params": {}}, {"name": "Q_template_instance", "version": 0, "params": {}}]


# --- the protocol's decoders, as `OneStage` reads them ---------------------------------------------------------------

PARAM_KEYS = {"anchors", "delta", "law", "leaf_layer", "log", "partition", "ports", "program", "query", "registered",
              "scheme", "work_table"}
EVIDENCE_KEYS = {"draw", "instances", "proved_units", "receipt", "served", "session_registered", "session_roots", "sessions",
                 "statement", "verdicts"}
RATIO = re.compile(r"(0|[1-9][0-9]*)(?:/(0|[1-9][0-9]*))?")
DIGITS = re.compile(r"[0-9]+")


def _nat(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _exactly(j, keys: set[str], what: str) -> None:
    if not isinstance(j, dict) or set(j) != keys:
        raise D.Reject(f"the keys of {what} are not {sorted(keys)}")


def _obj(j, what: str) -> dict:
    if not isinstance(j, dict):
        raise D.Reject(f"{what} is not an object")
    return j


def ratio(s) -> Fraction:
    """`Core.Ratio.ofString`: `n` or `n/d` in lowest terms, ASCII digits without leading zeros, `d >= 2`."""
    m = RATIO.fullmatch(s) if isinstance(s, str) else None
    if m is None:
        raise D.Reject(f"{s!r} is not a ratio")
    n, d = int(m[1]), (1 if m[2] is None else int(m[2]))
    if m[2] is not None and (d <= 1 or gcd(n, d) != 1):
        raise D.Reject(f"{s!r} is not in lowest terms")
    return Fraction(n, d)


def probability(p) -> bool:
    """`OneStage.probability?`: `NUM/DEN` in ASCII digits, `0 < DEN`, `NUM <= DEN`."""
    parts = p.split("/") if isinstance(p, str) else []
    return (len(parts) == 2 and all(DIGITS.fullmatch(x) for x in parts) and 0 < int(parts[1])
            and int(parts[0]) <= int(parts[1]))


def rule(law) -> dict:
    """`OneStage.rule`: the agreed law, `{"law": "bernoulli", "p"}` or `{"law": "subset" | "stratified" | "work", "k"}`."""
    kind = law.get("law") if isinstance(law, dict) else None
    if kind == "bernoulli":
        _exactly(law, {"law", "p"}, "the law")
        if not probability(law["p"]):
            raise D.Reject("the law's p is not NUM/DEN")
    elif kind in ("subset", "stratified", "work"):
        _exactly(law, {"k", "law"}, "the law")
        if not _nat(law["k"]):
            raise D.Reject("the law's k is not a natural number")
    else:
        raise D.Reject("the law is not subset, bernoulli, stratified or work")
    return law


def decode_params(j) -> dict:
    _exactly(j, PARAM_KEYS, "the parameters")
    desc = _obj(j["program"], "program")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    _obj(j["query"], "query")
    rule(j["law"])
    if j["work_table"] is not None:
        _obj(j["work_table"], "work_table")
    if not isinstance(j["scheme"], str):
        raise D.Reject("scheme is not a string")
    for p, v in _obj(j["ports"], "ports").items():
        _exactly(v, {"domain", "leaves"}, f"port {p}")
        if not _nat(v["leaves"]) or not isinstance(v["domain"], str):
            raise D.Reject(f"port {p}: leaves or domain")
    if j["leaf_layer"] is not None and not isinstance(j["leaf_layer"], str):
        raise D.Reject("leaf_layer is not a string or null")
    if not isinstance(j["anchors"], list) or not isinstance(j["log"], list) or not all(isinstance(e, dict) for e in j["log"]):
        raise D.Reject("anchors is not a list or log not a list of records")
    part = j["partition"]
    if part is not None and not (isinstance(part, dict) and isinstance(part.get("program"), str)
                                 and isinstance(part.get("query"), dict)):
        raise D.Reject("partition names no program digest or query")
    if j["registered"] is not None:
        raise D.Reject("registered values (R7) are not ported")
    delta = ratio(j["delta"])
    if not 0 < delta < 1:
        raise D.Reject("delta is not in (0, 1)")
    return {**j, "program": PROGRAMS[sha], "delta": delta}


def decode_commitment(b: bytes) -> tuple[dict, bytes]:
    rec = PR.canonical_json(b)
    if not isinstance(rec, dict):
        raise D.Reject("the registration record is not an object")
    return rec, bytes.fromhex(R.digest(rec))


def decode_evidence(b: bytes) -> dict:
    j = PR.canonical_json(b)
    _exactly(j, EVIDENCE_KEYS, "the evidence")
    nats = lambda x: isinstance(x, list) and all(map(_nat, x))  # noqa: E731
    if not (_nat(j["instances"]) and isinstance(j["verdicts"], dict) and all(isinstance(v, dict) for v in j["verdicts"].values())
            and isinstance(j["sessions"], list) and all(isinstance(s, str) for s in j["sessions"])
            and isinstance(j["receipt"], dict) and nats(j["proved_units"]) and isinstance(j["statement"], dict)
            and (j["served"] is None or nats(j["served"]))
            and all(j[k] is None or isinstance(j[k], dict) for k in ("session_roots", "session_registered"))):
        raise D.Reject("the evidence's fields")
    return j


def _registration(p: dict, rec) -> tuple[str, ...]:
    return R.check(rec, p["program"], p["query"], p["law"], p["scheme"], p["ports"], p["leaf_layer"], p["anchors"], p["log"],
                   p["partition"], p["work_table"]).codes


def check(p: dict, c: tuple[dict, bytes], e: dict) -> tuple:
    """The outcome `OneStage.check` gives, from `one_stage`'s own checks: reject on R1-R6, the draw's checks, the record's
    checks or a verdict that ran and did not accept; undecided when only the verifier of record is missing or has not run;
    accept exactly when `audit_record` accepts, with the drawn and proved units, the bounds and the profile."""
    rec, _ = c
    if _registration(p, rec):
        return "reject", "registration"
    mine = DR.derive(p["program"], p["query"], p["law"], p["work_table"])
    if not DR.check_draw(e["draw"], rec, mine).ok:
        return "reject", "draw"
    out = audit.audit_record(rec, e["receipt"], e["draw"], mine, e["proved_units"], e["instances"], e["statement"],
                             e["verdicts"], p["delta"], e["sessions"], e["served"], e["session_roots"], e["session_registered"])
    if not out["accepted"]:
        if not all(x["ok"] for x in out["checks"].values()):
            return "reject", "checks"
        ran = [v for v in e["verdicts"].values() if v.get("accepted") is not None]
        return ("undecided", "verifier of record") if all(v["accepted"] is True for v in ran) else ("reject", "verdict")
    profile = audit.profile(rec, e["draw"], p["delta"]).to_json()
    wrong = out["wrong_units"]
    public = {"bound": wrong["bound"] if isinstance(wrong["bound"], int) else None, "drawn": list(e["draw"]["units"]),
              "per_template": wrong.get("per_template"), "proved": DR.proved(e["draw"])}
    return "accept", public, {**profile, "delta": str(profile["delta"])}


def answer(case) -> dict:
    return PR.answer(R.PROTOCOL, _json(case), params=decode_params, commitment=decode_commitment, evidence=decode_evidence,
                     digest=lambda c: c[1], check=check)


# --- the pieces' references ------------------------------------------------------------------------------------------

def _decodes(f, x) -> bool:
    try:
        f(x)
    except D.Reject:
        return False
    return True


def decode(case):
    case = _json(case)
    if not isinstance(case, dict) or "params" not in case:
        raise D.Reject("the case")
    if not all(isinstance(case.get(k), str) for k in ("commitment", "evidence")):
        raise D.Reject("the case")
    c, e = PR.unhex(case["commitment"]), PR.unhex(case["evidence"])
    try:
        digest = decode_commitment(c)[1].hex()
    except D.Reject:
        digest = None
    return {"commitment": digest, "evidence": _decodes(decode_evidence, e), "params": _decodes(decode_params, case["params"])}


def _params_of(j) -> dict:
    return decode_params(j.get("params") if isinstance(j, dict) else None)


def partition(j):
    p = _params_of(_json(j))
    part = dict(p["partition"]) if p["partition"] is not None else P.partition(p["program"], p["query"])
    out: dict = {}
    try:
        out["digest"] = P.digest(part)
    except ValueError:
        out["digest"] = None
    try:
        out["population"] = P.units(p["program"], p["query"])
        out["strata"] = [[t, rs] for t, rs in P.strata(p["program"], p["query"])]
    except P.QueryRefused:
        out["population"] = out["strata"] = None
    try:
        out["law"] = DR.derive(p["program"], p["query"], p["law"], p["work_table"])
    except (P.QueryRefused, ValueError):
        out["law"] = None
    return out


def registration(j):
    j = _json(j)
    return {"codes": list(_registration(_params_of(j), j.get("record") if isinstance(j, dict) else None))}


def check_draw(j):
    j = _json(j)
    if not isinstance(j, dict) or not _int(j.get("population")) or not isinstance(j.get("law"), dict):
        raise D.Reject("population is not an integer or law not an object")
    return {"codes": list(DR.check_draw(j.get("draw"), {"population": j["population"]}, j["law"]).codes)}


def check_proved(j):
    j = _json(j)
    ints = lambda x: isinstance(x, list) and all(map(_int, x))  # noqa: E731
    d = j.get("draw") if isinstance(j, dict) else None
    if not (isinstance(d, dict) and ints(d.get("units")) and ("closure" not in d or ints(d["closure"]))
            and ints(j.get("proved_units")) and _int(j.get("instances"))):
        raise D.Reject("not a draw of integer units, proved units and an instance count")
    return {"codes": list(DR.check_proved(d, j["proved_units"], j["instances"]).codes)}


def draw(j):
    """`draw_first` on the source's bytes, read in order; it raises `ValueError` (and `TypeError` on an unhashable law
    name) where Lean refuses."""
    j = _json(j)
    if not isinstance(j, dict) or not isinstance(j.get("law"), dict):
        raise D.Reject("law is not an object")
    if not _nat(j.get("population")):
        raise D.Reject("population is not a natural number")
    if not isinstance(j.get("source"), str):
        raise D.Reject("source is not hex")
    src, at = PR.unhex(j["source"]), 0

    def source(n: int) -> bytes:
        nonlocal at
        at += n
        return src[at - n:at]
    try:
        return draw_first(OwnRandomness(source), j["law"], j["population"])
    except (ValueError, TypeError) as e:
        raise D.Reject(str(e)) from None


def bound(j):
    j = _json(j)
    ss = j.get("strata") if isinstance(j, dict) else None
    if not isinstance(ss, list):
        raise D.Reject("strata is not a list")
    for s in ss:
        _exactly(s, {"k", "name", "size"}, "a stratum")
        if not (isinstance(s["name"], str) and _nat(s["size"]) and _nat(s["k"])):
            raise D.Reject("a stratum's types")
    delta = ratio(j.get("delta"))
    try:
        law = Stratified(tuple(Stratum(s["name"], s["size"], s["k"]) for s in ss))
        return {"bound": law.count_bound(delta), "per_stratum": {s.name: law.stratum_bound(s.name, delta) for s in law.strata}}
    except ValueError as e:
        raise D.Reject(str(e)) from None


def _fixed_audit():
    """One valid audit whose verdicts the `verdict` piece replaces."""
    s = SETUPS[2]
    law = {"k": 2, "law": "subset"}
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports], WINDOW,
                   leaf_layer=LEAF)
    d = {"k": 2, "law": "subset", "population": s.population, "units": [1, 4]}
    return rec, R.receipt(rec, "verifier-test", received=RECEIVED), d, law


FIXED = _fixed_audit()


def verdict(j):
    j = _json(j)
    vs = j.get("verdicts") if isinstance(j, dict) else None
    if not isinstance(vs, dict) or not all(isinstance(v, dict) for v in vs.values()):
        raise D.Reject("verdicts is not an object of objects")
    rec, receipt, d, law = FIXED
    out = audit.audit_record(rec, receipt, d, law, d["units"], 2, {}, vs, Fraction(1, 1000))
    assert all(c["ok"] for c in out["checks"].values()), out["checks"]
    if out["accepted"]:
        return {"verdict": "accept"}
    ran = [v for v in vs.values() if v.get("accepted") is not None]
    return {"verdict": "undecided" if all(v["accepted"] is True for v in ran) else "reject"}


# --- strategies ------------------------------------------------------------------------------------------------------

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text("ab/1", max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text("abk", max_size=2), inner, max_size=3),
                    max_leaves=6)
WORK_ENTRY = st.one_of(st.integers(0, 4), st.fixed_dictionaries({"work": st.integers(0, 4)}),
                       st.fixed_dictionaries({"floor": st.integers(1, 3), "work": st.integers(0, 4)}))
BAD_WORK_ENTRY = st.sampled_from([-1, "1", {"work": -1}, {"floor": 0, "work": 1}, {"work": 1, "x": 1}, {"floor": 1}, True])


@st.composite
def laws(draw, n: int):
    kind = draw(st.sampled_from(["subset", "bernoulli", "stratified", "work"]))
    if kind == "bernoulli":
        den = draw(st.integers(1, 6))
        return {"law": "bernoulli", "p": f"{draw(st.integers(0, den))}/{den}"}
    return {"k": draw(st.integers(0 if kind == "subset" else 1, n)), "law": kind}


@st.composite
def work_tables(draw, s: Setup):
    """A work table naming every template, mostly; sometimes one missing, one extra, or a bad entry."""
    table = {t: draw(WORK_ENTRY) for t in s.templates}
    change = draw(st.integers(0, 9))
    if change == 0:
        table.pop(s.templates[0])
    elif change == 1:
        table["Other_v1"] = 1
    elif change == 2:
        table[s.templates[-1]] = draw(BAD_WORK_ENTRY)
    return table


def drawn(draw, law: dict, n: int) -> dict:
    """The verifier's draw under its expanded law."""
    kind = law["law"]
    if kind == "subset":
        return {"k": law["k"], "law": "subset", "population": n, "units": sorted(draw(st.permutations(range(n)))[:law["k"]])}
    if kind == "bernoulli":
        return {"law": "bernoulli", "p": law["p"], "population": n,
                "units": sorted(draw(st.sets(st.integers(0, n - 1), max_size=n)))}
    units = sorted(u for s in law["strata"] for u in draw(st.permutations([u for lo, hi in s["units"] for u in range(lo, hi)]))[:s["k"]])
    d = {"k": law["k"], "law": kind, "population": n, "strata": law["strata"], "units": units}
    rest = [u for u in range(n) if u not in units]
    if kind == "work" and rest and draw(st.booleans()):
        d["closure"] = sorted(draw(st.sets(st.sampled_from(rest), min_size=1)))
    return d


@st.composite
def audits(draw) -> tuple[dict, dict, dict]:
    """A valid audit: the verifier's parameters, the registration record and the evidence (verdicts mostly accepting)."""
    s = draw(st.sampled_from(SETUPS))
    n = s.population
    law = draw(laws(n))
    table = draw(work_tables(s)) if law["law"] == "work" else None
    try:
        mine = DR.derive(s.program, s.query, law, table)
    except ValueError:
        table = {t: 1 for t in s.templates}
        mine = DR.derive(s.program, s.query, law, table)
    pre = draw(st.booleans())
    part = {"program": s.sha512, "query": s.query} if pre else None
    log = []
    if draw(st.booleans()):
        log = [R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                        EARLIER, leaf_layer=LEAF, partition=part, work_table=table)]
    anchors = draw(st.sampled_from([[], ["weights"]]))
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports], WINDOW,
                   leaf_layer=LEAF, anchors=anchors, previous=R.digest(log[-1]) if log else None, partition=part,
                   work_table=table)
    assert rec["law"] == mine
    params = {"anchors": anchors, "delta": draw(st.sampled_from(["1/1000", "1/10", "1/2", "999/1000"])), "law": law,
              "leaf_layer": LEAF, "log": log, "partition": part, "ports": s.ports, "program": s.desc, "query": s.query,
              "registered": None, "scheme": SCHEME, "work_table": table}
    d = drawn(draw, mine, n)
    proved = DR.proved(d)
    verdicts = {"lean": {"accepted": True, "digest": "aa" * 64, "of_record": True}}
    if draw(st.booleans()):
        verdicts["upstream"] = {"accepted": draw(st.sampled_from([True, None])), "of_record": False}
    evidence = {"draw": d, "instances": len(proved), "proved_units": proved,
                "receipt": R.receipt(rec, "verifier-test", received=RECEIVED),
                "served": draw(st.sampled_from([None, list(d["units"])])),
                "session_registered": draw(st.sampled_from([None, {}])),
                "session_roots": draw(st.sampled_from([None, dict(ROOTS)])), "sessions": ["ab" * 64],
                "statement": {"instances": len(proved)}, "verdicts": verdicts}
    return _json(params), _json(rec), _json(evidence)


def _set(where: str, key: str, value):
    def change(a, x):
        a[where][key] = value(a, x) if callable(value) else value
    return change


def _roots(f):
    def change(a, x):
        a["record"]["roots"] = f(a["record"]["roots"], x)
    return change


def _draw(f):
    def change(a, x):
        a["evidence"]["draw"] = f(a["evidence"]["draw"], x)
    return change


def _other_setup(a, x):
    return SETUPS[x % len(SETUPS)].desc


#: One change each to the record (R1-R6), the parameters (decodable or not) and the evidence (the draw, the record's checks,
#: the verdicts, its decoding); `x` picks among variants.
RECORD = [
    _set("record", "format", lambda a, x: [R.FORMAT_V2, "verity/registration/v0"][x % 2]),
    _set("record", "protocol", "verity/sampled-proofs/two-stage/v1"),
    _set("record", "law", lambda a, x: {"k": x % 4, "law": "subset"}),
    _set("record", "window", lambda a, x: [{"kind": "guess"}, "served", {}, EARLIER][x % 4]),
    _set("record", "program", "0" * 128), _set("record", "query", "0" * 128), _set("record", "partition", "0" * 128),
    _set("record", "population", lambda a, x: a["record"]["population"] + 1 + x % 2),
    _set("record", "population", "6"),
    _set("record", "scheme", "frame-v3/blake3"),
    _roots(lambda rs, x: rs[:1]), _roots(lambda rs, x: []), _roots(lambda rs, x: rs + rs[:1]), _roots(lambda rs, x: "roots"),
    _roots(lambda rs, x: [{**rs[0], "leaves": rs[0]["leaves"] + 1}, *rs[1:]]),
    _roots(lambda rs, x: [{**rs[0], "domain": "00" * 64}, *rs[1:]]),
    _roots(lambda rs, x: [{**rs[0], "root": ["AB" * 64, 5, "", "xy"][x % 4]}, *rs[1:]]),
    _roots(lambda rs, x: [{k: v for k, v in rs[0].items() if k != "domain"}, *rs[1:]]),
    _roots(lambda rs, x: [{**rs[0], "port": "y"}, *rs[1:]]),
    _set("record", "leaf_layer", lambda a, x: [None, "ff" * 64][x % 2]),
    _set("record", "anchors", lambda a, x: [["weights"], [], ["prompt"]][x % 3]),
    _set("record", "previous", lambda a, x: ["11" * 64, None][x % 2]),
    _set("record", "extra", 1),
    lambda a, x: a["record"].pop(sorted(a["record"])[x % len(a["record"])]),
]
PARAMS = [
    _set("params", "delta", lambda a, x: ["2/4", "0", "1", "1/1", "3/2", "a", 0, "1/0", "01/2", "1/2 ", "1/1000"][x % 11]),
    _set("params", "registered", {"weights": 4}),
    _set("params", "law", lambda a, x: [{"law": "subset"}, {"k": -1, "law": "subset"}, {"law": "bernoulli", "p": "3/2"},
                                        {"law": "bernoulli", "p": 1}, {"k": 1, "law": "nope"}, {"k": 1, "law": "subset", "x": 1},
                                        "subset:2", {"law": "bernoulli", "p": "1/2/3"}, {"k": True, "law": "subset"},
                                        {"k": 2, "law": "subset"}, {"k": 1, "law": "stratified"}, {"k": 1, "law": "work"},
                                        {"law": "bernoulli", "p": "1/2"}][x % 13]),
    _set("params", "ports", lambda a, x: [{"x": {"domain": "d", "leaves": "6"}}, {}, 5, {"x": {"domain": "d", "leaves": 6, "y": 1}},
                                          {**a["params"]["ports"], "z": {"domain": "d", "leaves": 1}}][x % 5]),
    _set("params", "work_table", lambda a, x: [None, {}, 5, {t: 2 for t in PROGRAMS_TEMPLATES}][x % 4]),
    _set("params", "query", lambda a, x: QUERIES[x % len(QUERIES)]),
    _set("params", "program", lambda a, x: [_other_setup(a, x), {}, 5, "program"][x % 4]),
    _set("params", "partition", lambda a, x: [{"program": 5, "query": {}}, {"program": "p"}, None,
                                              P.partition(PROGRAMS[PO.program_sha512(a["params"]["program"])], a["params"]["query"]),
                                              {"program": "0" * 128, "query": a["params"]["query"]}][x % 5]),
    _set("params", "scheme", lambda a, x: [5, "frame-v3/blake3"][x % 2]),
    _set("params", "leaf_layer", lambda a, x: [5, None, "ff" * 64][x % 3]),
    _set("params", "anchors", lambda a, x: ["x", ["weights"], []][x % 3]),
    _set("params", "log", lambda a, x: [[5], [], [a["record"]]][x % 3]),
    _set("params", "extra", 1),
    lambda a, x: a["params"].pop(sorted(a["params"])[x % len(a["params"])]),
]


def _strip(d, x):
    return {k: v for k, v in d.items() if k != sorted(d)[x % len(d)]}


DRAW = [
    _draw(lambda d, x: {**d, "units": list(reversed(d["units"]))}),
    _draw(lambda d, x: {**d, "units": sorted(d["units"] + [d["population"] + x % 2])}),
    _draw(lambda d, x: {**d, "units": d["units"][:-1]}),
    _draw(lambda d, x: {**d, "units": [-1, *d["units"]]}),
    _draw(lambda d, x: {**d, "units": [*d["units"], *d["units"][-1:]]}),
    _draw(lambda d, x: {**d, "units": [["0", *d["units"]], "units", [None], [True, *d["units"]]][x % 4]}),
    _draw(lambda d, x: {**d, "population": d["population"] + 1}),
    _draw(lambda d, x: {**d, "k": d.get("k", 0) + 1} if "k" in d else {**d, "p": "0/1"}),
    _draw(lambda d, x: {**d, "law": ["subset", "bernoulli", "stratified", "work", "nope"][x % 5]}),
    _draw(lambda d, x: {**d, "extra": 1}),
    _draw(_strip),
    _draw(lambda d, x: {**d, "closure": [[], [d["units"][0]] if d["units"] else [0], [d["population"]], "c", [2, 1]][x % 5]}),
    _draw(lambda d, x: {**d, "p": ["1/2", "3/2", "a/2", "1/0", "1"][x % 5]} if d["law"] == "bernoulli" else d),
    _draw(lambda d, x: {**d, "strata": d["strata"][::-1]} if "strata" in d else d),
    _draw(lambda d, x: [None, 5, [], "draw"][x % 4]),
]
EVIDENCE = [
    _set("evidence", "receipt", lambda a, x: [{**a["evidence"]["receipt"], "registration": "00" * 64}, {}, 5][x % 3]),
    _set("evidence", "proved_units", lambda a, x: [a["evidence"]["proved_units"][:-1], a["evidence"]["proved_units"][::-1],
                                                   [-1], ["a"]][x % 4]),
    _set("evidence", "instances", lambda a, x: [a["evidence"]["instances"] + 1, -1, "1"][x % 3]),
    _set("evidence", "served", lambda a, x: [a["evidence"]["draw"]["units"][:-1], [a["evidence"]["draw"]["population"]],
                                             a["evidence"]["draw"]["units"][::-1], "s"][x % 4]),
    _set("evidence", "session_roots", lambda a, x: [{}, {"x": "00" * 64}, {"y": "ab" * 64}, {**ROOTS, "y": "ab" * 64},
                                                    {"x": ROOTS["x"]}, 5][x % 6]),
    _set("evidence", "session_registered", lambda a, x: [{"weights": "aa" * 64}, 5][x % 2]),
    _set("evidence", "sessions", lambda a, x: [[1], "s", []][x % 3]),
    _set("evidence", "statement", 5),
    lambda a, x: a["evidence"].pop(sorted(a["evidence"])[x % len(a["evidence"])]),
    _set("evidence", "extra", 1),
]
VERDICTS_OF = [_set("evidence", "verdicts", v) for v in (
    {"lean": {"accepted": True}}, {"lean": {"accepted": None, "of_record": True}}, {"lean": {"of_record": True}},
    {"lean": {"accepted": True, "of_record": True}, "other": {"accepted": False}},
    {"lean": {"accepted": True, "of_record": True}, "other": {"accepted": None}},
    {"lean": {"accepted": 1, "of_record": True}}, {"lean": {"accepted": False, "of_record": True}}, {},
    {"lean": "accepted"}, [], {"a": {"accepted": True, "of_record": True}, "b": {"accepted": None, "of_record": 1}})] + [
    _set("evidence", "verdicts", lambda a, x: {"lean": {"accepted": True, "of_record": [1, "x", [], 0, {}, [0], "", None][x % 8]}})]
GROUPS = [RECORD, PARAMS, DRAW, EVIDENCE, VERDICTS_OF]


@st.composite
def changed_audits(draw, groups=GROUPS, most: int = 2, least: int = 0) -> dict:
    """An audit as `{"params", "record", "evidence"}`, with `least` to `most` changes, each from one of `groups`; a change
    that doesn't apply to what an earlier one left (a field it removed, a list it replaced) is skipped."""
    params, rec, evidence = draw(audits())
    a = {"params": params, "record": rec, "evidence": evidence}
    for _ in range(draw(st.integers(least, most))):
        change = draw(st.sampled_from(draw(st.sampled_from(groups))))
        try:
            change(a, draw(st.integers(0, 63)))
        except (KeyError, TypeError, IndexError, ValueError, AttributeError):
            pass
    return _json(a)


def _hex(value) -> str:
    return D.canonical(value).encode().hex()


def _case(a: dict) -> dict:
    return {"commitment": _hex(a["record"]), "evidence": _hex(a["evidence"]), "params": a["params"]}


def _upper(field):
    return lambda case, x: case | {field: case[field].upper()}


#: One change to a case's framing: hex `Flock.unhex` refuses, a field missing or not a string, a commitment or evidence
#: that isn't canonical JSON or isn't an object.
FRAMING = [
    _upper("commitment"), _upper("evidence"),
    lambda case, x: case | {"evidence": case["evidence"] + "0"},
    lambda case, x: case | {"commitment": case["commitment"][:2] + " " + case["commitment"][2:]},
    lambda case, x: {k: v for k, v in case.items() if k != ["commitment", "evidence", "params"][x % 3]},
    lambda case, x: case | {["commitment", "evidence"][x % 2]: x},
    lambda case, x: case | {["commitment", "evidence"][x % 2]: json.dumps(json.loads(bytes.fromhex(case["commitment"]))).encode().hex()},
    lambda case, x: case | {"commitment": [b"[]", b"5", b'"r"', b"\xff", b""][x % 5].hex()},
    lambda case, x: case | {"evidence": [b"[]", b"{}", b'{"draw":1.5}', b"\xff"][x % 4].hex()},
]
CASES = st.one_of(
    changed_audits(most=0).map(_case), changed_audits([VERDICTS_OF], most=1, least=1).map(_case),
    changed_audits(most=1, least=1).map(_case), changed_audits(most=1, least=1).map(_case),
    changed_audits(most=2, least=2).map(_case),
    st.builds(lambda a, i, x: FRAMING[i](_case(a), x), changed_audits(most=0), st.integers(0, len(FRAMING) - 1),
              st.integers(0, 63)),
    JSON)

ONE_STAGE = D.ProtocolPort("sampled-proofs-one-stage", answer, CASES, package=PACKAGE)
server = D.server_fixture(ONE_STAGE)


def test_the_one_stage_audit_agrees(server):
    stats = D.run(ONE_STAGE, examples=600, server=server)
    assert stats.examples == 600 and min(stats.counts.get(v, 0) for v in ("accept", "reject", "undecided")) > 40, stats


def test_an_accepted_case_is_compared_whole(server):
    """One accepted audit of each law, field by field: the public outputs and the profile Python writes."""
    for kind, k in (("subset", 2), ("bernoulli", None), ("stratified", 3), ("work", 3)):
        s = SETUPS[3]
        law = {"law": "bernoulli", "p": "1/2"} if kind == "bernoulli" else {"k": k, "law": kind}
        table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}} if kind == "work" else None
        mine = DR.derive(s.program, s.query, law, table)
        rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                       WINDOW, leaf_layer=LEAF, work_table=table)
        units = {"subset": [0, 3], "bernoulli": [1, 2, 4]}.get(kind) or sorted(
            u for st_ in mine["strata"] for u in [u for lo, hi in st_["units"] for u in range(lo, hi)][:st_["k"]])
        d = {**{k_: v for k_, v in mine.items()}, "population": s.population, "units": units}
        proved = DR.proved(d)
        evidence = {"draw": d, "instances": len(proved), "proved_units": proved,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_registered": None,
                    "session_roots": None, "sessions": [], "statement": {},
                    "verdicts": {"lean": {"accepted": True, "of_record": True}}}
        params = {"anchors": [], "delta": "1/20", "law": law, "leaf_layer": LEAF, "log": [], "partition": None,
                  "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
                  "work_table": table}
        case = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        verdict_, line = D.replay(ONE_STAGE, case, server)
        assert verdict_ == "accept" and json.loads(line) == answer(case) | {"reason": None}, (kind, line)


PIECES = "sampled-proofs-difftest"


def _audit_inputs(f, groups, most=1):
    return changed_audits(groups, most).map(f)


DECODE = D.Port(PIECES, "decode", decode, st.one_of(
    changed_audits(most=1).map(_case),
    st.builds(lambda a, i, x: FRAMING[i](_case(a), x), changed_audits(most=0), st.integers(0, len(FRAMING) - 1),
              st.integers(0, 63)), JSON), package=PACKAGE)
PARTITION = D.Port(PIECES, "partition", partition, st.one_of(
    _audit_inputs(lambda a: {"params": a["params"]}, [PARAMS]),
    st.builds(lambda a, q, law, t, pre: {"params": a["params"] | {
        "query": q, "law": law, "work_table": t,
        "partition": {"program": PO.program_sha512(a["params"]["program"]), "query": q} if pre else None}},
        changed_audits(most=0), st.sampled_from(QUERIES), st.one_of(laws(6), st.just({"k": 1, "law": "stratified"})),
        st.one_of(st.none(), st.dictionaries(st.sampled_from(PROGRAMS_TEMPLATES + ["Other_v1"]), WORK_ENTRY | BAD_WORK_ENTRY,
                                             max_size=3)), st.booleans()),
    JSON), package=PACKAGE)
REGISTRATION = D.Port(PIECES, "registration", registration,
                      st.one_of(_audit_inputs(lambda a: {"params": a["params"], "record": a["record"]}, [RECORD, PARAMS], 2),
                                st.builds(lambda a, r: {"params": a["params"], "record": r}, changed_audits(most=0), JSON)),
                      package=PACKAGE)
CHECK_DRAW = D.Port(PIECES, "check_draw", check_draw, st.one_of(
    st.builds(lambda a, dn, other: {"draw": a["evidence"]["draw"], "law": other or a["record"]["law"],
                                    "population": a["record"]["population"] + dn},
              changed_audits([DRAW], 2), st.sampled_from([0, 0, 0, 1, -1]),
              st.sampled_from([None, None, None, {"k": 2, "law": "subset"}, {"law": "bernoulli", "p": "1/2"}])),
    st.builds(lambda d, n, law: {"draw": d, "law": law, "population": n}, JSON, st.integers(0, 6) | JSON,
              st.just({"k": 1, "law": "subset"}) | JSON)), package=PACKAGE)
CHECK_PROVED = D.Port(PIECES, "check_proved", check_proved, st.one_of(
    st.builds(lambda a, f, dn: {"draw": a["evidence"]["draw"], "instances": a["evidence"]["instances"] + dn,
                                "proved_units": f(a["evidence"]["proved_units"])},
              changed_audits(most=0), st.sampled_from([lambda u: u, lambda u: u, lambda u: u[:-1], lambda u: u[::-1],
                                                       lambda u: [*u, 99], lambda u: "u"]),
              st.sampled_from([0, 0, 1, -1])),
    st.builds(lambda d, p, i: {"draw": d, "instances": i, "proved_units": p}, JSON, JSON, JSON)), package=PACKAGE)
def _sources(least: int, most: int):
    """Sources of the verifier's own randomness, `least` to `most` bytes after some `ff` bytes or none: a word of them is
    rejected under any `n` but a power of two, so a stream runs out early and is extended and rerun."""
    random = st.builds(lambda seed, n: hashlib.shake_256(seed).hexdigest(n), st.binary(max_size=4), st.integers(least, most))
    return random | st.builds(lambda r, s: "ff" * r + s, st.integers(1, 400), random)


SOURCE = _sources(1200, 2400)
SHORT_SOURCE = _sources(0, 200)
SIMPLE_LAWS = st.builds(lambda k: {"k": k, "law": "subset"}, st.integers(0, 40)) | st.builds(
    lambda num, den: {"law": "bernoulli", "p": f"{num}/{den}"}, st.integers(0, 12), st.integers(1, 12))
LAW_CHANGES = [
    lambda law, x: law,
    lambda law, x: {**law, "k": law["k"] + x % 3 - 1} if "k" in law else {**law, "p": ["0/1", "1/1", "2/4", "3/2", "1/0", "1"][x % 6]},
    lambda law, x: {**law, "strata": law["strata"][:-1]},
    lambda law, x: {**law, "strata": [{**s, "k": s["k"] + 1} if i == x % len(law["strata"]) else s
                                      for i, s in enumerate(law["strata"])]},
    lambda law, x: {**law, "extra": 1},
    lambda law, x: {k: v for k, v in law.items() if k != "law"},
    lambda law, x: {**law, "law": ["subset", "bernoulli", "stratified", "work", [1]][x % 5]},
]


def _changed_law(law: dict, change, x: int) -> dict:
    try:
        return change(law, x)
    except (KeyError, ZeroDivisionError):
        return law


DRAW = D.Port(PIECES, "draw", draw, st.one_of(
    st.builds(lambda a, src: {"law": a["record"]["law"], "population": a["record"]["population"], "source": src},
              changed_audits(most=0), SOURCE),
    st.builds(lambda a, f, x, dn, src: {"law": _changed_law(a["record"]["law"], f, x),
                                        "population": a["record"]["population"] + dn, "source": src},
              changed_audits(most=0), st.sampled_from(LAW_CHANGES), st.integers(0, 63), st.sampled_from([0, 0, 0, 1, -1]),
              SOURCE),
    st.builds(lambda law, n, src: {"law": law, "population": n, "source": src}, SIMPLE_LAWS, st.integers(0, 40), SOURCE),
    st.builds(lambda law, n, src: {"law": law, "population": n, "source": src}, SIMPLE_LAWS | laws(6),
              st.integers(0, 40), SHORT_SOURCE),
    st.builds(lambda law, n, src: {"law": law, "population": n, "source": src}, JSON, st.integers(0, 6) | JSON,
              SHORT_SOURCE | st.sampled_from(["", "0", "zz", "AB", "ab cd"]) | JSON)), package=PACKAGE)
STRATA = st.lists(st.builds(lambda name, size, dk: {"k": max(0, size - dk), "name": name, "size": size},
                            st.sampled_from(["a", "b", "c", "unit"]), st.integers(0, 14), st.integers(0, 14)),
                  min_size=1, max_size=3, unique_by=lambda s: s["name"])
DELTA = st.builds(lambda d, n: str(Fraction(n, d)), st.integers(2, 1000), st.integers(1, 999)).filter(
    lambda s: 0 < Fraction(s) < 1)
LARGE_STRATA = st.lists(st.builds(lambda name, size, k: {"k": min(k, size), "name": name, "size": size},
                                  st.sampled_from(["a", "b", "c", "unit"]), st.integers(1, 300), st.integers(0, 40)),
                        min_size=1, max_size=3, unique_by=lambda s: s["name"])
BOUND = D.Port(PIECES, "bound", bound, st.one_of(
    st.builds(lambda ss, d: {"delta": d, "strata": ss}, STRATA, DELTA),
    st.builds(lambda ss, d: {"delta": d, "strata": ss}, LARGE_STRATA, DELTA),
    st.builds(lambda ss, d: {"delta": d, "strata": ss}, STRATA, st.sampled_from(["0", "1", "2/4", "1/1", "3/2", "a", "1/0"])),
    st.builds(lambda ss, d, i, bad: {"delta": d, "strata": [*ss[:i], bad, *ss[i:]]}, STRATA, DELTA, st.integers(0, 3),
              st.sampled_from([{"k": 1, "name": "", "size": 2}, {"k": 3, "name": "z", "size": 2}, {"k": 1, "name": "a", "size": 2},
                               {"k": True, "name": "z", "size": 2}, {"name": "z", "size": 2}, {"k": 1, "name": 5, "size": 2}])),
    st.builds(lambda d: {"delta": d, "strata": []}, DELTA), JSON), package=PACKAGE)
VERDICT_VALUE = st.sampled_from([True, False, None, 1, 0, "yes", [], {}, [0], ""])
VERDICTS = st.dictionaries(st.sampled_from(["lean", "upstream", "rust"]), st.one_of(
    st.fixed_dictionaries({"accepted": VERDICT_VALUE, "of_record": VERDICT_VALUE}),
    st.fixed_dictionaries({"accepted": VERDICT_VALUE}), st.fixed_dictionaries({"of_record": VERDICT_VALUE}),
    st.fixed_dictionaries({"accepted": st.just(True), "digest": st.just("aa" * 64), "of_record": st.just(True)})), max_size=3)
VERDICT = D.Port(PIECES, "verdict", verdict, st.one_of(
    st.builds(lambda vs: {"verdicts": vs}, VERDICTS), st.builds(lambda vs: {"verdicts": vs}, JSON), JSON), package=PACKAGE)

pieces = D.server_fixture(DECODE)


PIECE_RUNS = [(DECODE, 300), (PARTITION, 400), (REGISTRATION, 500), (CHECK_DRAW, 500), (DRAW, 500), (CHECK_PROVED, 300),
              (BOUND, 500), (VERDICT, 300)]


@pytest.mark.parametrize("port, examples", PIECE_RUNS, ids=[p.fn for p, _ in PIECE_RUNS])
def test_a_piece_agrees(pieces, port, examples):
    stats = D.run(port, examples=examples, server=pieces)
    assert stats.outputs > examples // 4 and stats.rejects > 0, stats
