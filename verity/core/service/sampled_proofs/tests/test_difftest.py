"""Sampled proofs' one-stage audit in Lean (`core_sampled_proofs`) against `one_stage`'s Python: the whole protocol through
its own `serve` (`sampled-proofs-one-stage`, a `ProtocolPort`), and its pieces through `sampled-proofs-difftest` (`Port`s),
on drawn programs, laws, records, draws and verdicts, valid ones and one or two changes to each.

The reference is `one_stage` itself (`registration.check`, `draw.derive`, `check_draw`, `audit.audit_record`, `profile`,
`wrong_units`, `registered.registered_entries` and `domain`) behind the protocol's decoders, which this file writes as the
Lean reads them (`OneStage.paramsOfJson`, `commitmentOf`, `evidenceOf`), for the `draw` piece `two_stage.draw_first` on
`OwnRandomness` over the given bytes, and for `check_public` and `check_verdicts` the proof service's own
(`service.check_public`, `service.check_verdicts`). Records are v1 and, on the programs with root parameters to register
(`REGISTERED_SETUPS`, 16- and 32-bit), mostly v2: the verifier expects values registered (`registered`, with its program
function's `parameters`), and the record (and an earlier one in the log) registers `Registered.commit`'s entries.
The verifier's program is looked up among this file's programs by its descriptor. Inputs
stay inside what Python reads as Lean does: integers that are not booleans where Python compares with `==`, ASCII digits
in a probability, and integer populations in a draw.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from fractions import Fraction
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity_difftest import protocol as PR
from verity.core.protocols.profile import Stratified, Stratum
from verity.core.protocols.sampled_proofs import service
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as DR, partition as P, registered as G, registration as R
from verity.core.protocols.sampled_proofs.two_stage import OwnRandomness, draw_first
from verity.ml.circuits import codec, partition_object as PO
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program, input_id
from verity.ml.circuits.types import Array, Record, Tuple, Value

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


def _registered_program(name: str, n: int, extra, registered) -> Program:
    """`n` instances of `PAIR` over `x`, beside root parameters `extra` the body does not read, those `registered` names
    bound at registration."""
    ((_, t),), ret = PAIR.signature()

    def signature(S):
        return (("x", Array(S.N, t)), *extra), Array(S.N, ret)

    def body(B, S, x, *_):
        return B.batch(PAIR, x, axes=(0,))

    return Program(bind(CompositeDefinition(name, 0, ("N",), signature, body, register=False), N=n), registered=registered)


class RegisteredSetup(Setup):
    """A verifier's program with root parameters to register: `values`, the `{parameter: words}` it may expect, each value
    committed once at each of its words (`Registered.commit`, salts fixed), and its signature as `parameters` names it."""

    def __init__(self, program, values):
        super().__init__(program, P.template_instance_query(PAIR.id))
        self.signature = _json([[name, t.to_json()] for name, t in program.fn.signature()[0]])
        self.values = values
        self.committed = {}
        for v, w in {(v, w) for vs in values for v, w in vs.items()}:
            n, bits = len(G.parameter_gates(program, v)), G.width(program, v)
            rows = [[(i * 40503 + 7) % (1 << bits) for i in range(r * w, (r + 1) * w)] for r in range(n // w)]
            salts = [hashlib.shake_256(f"{v}/{w}/{r}".encode()).digest(192) for r in range(n // w)]
            self.committed[v, w] = G.Registered.commit(self.sha512, v, rows, salts, bits)

    def entries(self, values) -> list[dict]:
        return [self.committed[v, w].entry() for v, w in (values or {}).items()]


#: Programs whose root parameters a verifier may expect registered: 16-bit ones (`W`, and `D` of 12-bit leaves) and ones
#: `registered_entries` refuses (`E` has no gates, `M` two widths, `V` is 40 bits), and 32-bit ones (`T`, and `U` of
#: 20-bit leaves); `x`, bound per run, reads as a 16-bit value too.
REGISTERED_SETUPS = [
    RegisteredSetup(_registered_program("CoreSampledProofsRegistered16", 3, (
        ("W", Array(6, Value(16))), ("D", Array(4, Value(12))), ("E", Array(0, Value(16))), ("M", Tuple(Value(8), Value(16))),
        ("V", Value(40))), ("W", "D")), [{"W": 2}, {"D": 1, "W": 3}, {"D": 4}, {"W": 6, "x": 2}]),
    RegisteredSetup(_registered_program("CoreSampledProofsRegistered32", 2, (
        ("T", Array(4, Value(32))), ("U", Record(("a", Array(3, Value(20))),))), ("T", "U")),
        [{"T": 2}, {"T": 1, "U": 3}, {"U": 1}, {"T": 4, "x": 1}])]
PROGRAMS = {s.sha512: s.program for s in SETUPS + REGISTERED_SETUPS}
PROGRAMS_TEMPLATES = [PAIR.id, TRIPLE.id]
#: Queries a verifier may hold: the setups' own, and ones core refuses or one-stage doesn't apply.
QUERIES = [s.query for s in SETUPS] + [
    {}, PO.query(), PO.query(version=2), {**PO.query(), "params": {"W": 32, "X": 70000}}, P.template_instance_query("Other_v1"),
    P.template_instances_query([PAIR.id]), {"name": "Q_template_instance", "version": 1, "params": {"template": PAIR.id}},
    {"name": "Nope", "version": 0, "params": {}}, {"name": "Q_template_instance", "version": 0, "params": {}}]


# --- the protocol's decoders, as `OneStage` reads them ---------------------------------------------------------------

PARAM_KEYS = {"anchors", "delta", "law", "leaf_layer", "log", "partition", "ports", "program", "query", "registered",
              "scheme", "work_table"}
EVIDENCE_KEYS = {"draw", "instances", "proved_units", "receipt", "served", "session_roots", "sessions",
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


def _type(j):
    """`Ty.ofJson`: a type exactly as `to_json` writes it."""
    tag = j[0] if isinstance(j, list) and j else None
    if tag == "Value" and len(j) == 2 and _nat(j[1]) and j[1] > 0:
        return Value(j[1])
    if tag == "Array" and len(j) == 3 and _nat(j[1]):
        return Array(j[1], _type(j[2]))
    if tag == "Tuple" and len(j) == 2 and isinstance(j[1], list):
        return Tuple(*map(_type, j[1]))
    if (tag == "Record" and len(j) == 2 and isinstance(j[1], list)
            and all(isinstance(f, list) and len(f) == 2 and isinstance(f[0], str) for f in j[1])
            and len({k for k, _ in j[1]}) == len(j[1])):
        return Record(*[(k, _type(t)) for k, t in j[1]])
    raise D.Reject(f"type {j!r}")


def _lower(t) -> list[tuple[int, int]]:
    """`program._lower_input`'s input nodes, `(width, n)` each."""
    w = t.homogeneous_width()
    if w is not None:
        return [(w, t.leaves)]
    fields = [t.elem] * t.n if isinstance(t, Array) else [f[1] if isinstance(t, Record) else f for f in t.fields]
    return [node for f in fields for node in _lower(f)]


def _parameters(desc: dict, program: Program, j) -> None:
    """`Registered.params`: the program function's `[[name, type], ...]`, distinct names whose types lower to exactly the
    root body's input nodes. The strategies draw no signature but the program's own that lowers so, which this file would
    have no program for."""
    if not (isinstance(j, list) and all(isinstance(p, list) and len(p) == 2 and isinstance(p[0], str) for p in j)):
        raise D.Reject("parameters is not a list of [name, type]")
    types = [_type(t) for _, t in j]
    if len({k for k, _ in j}) != len(j):
        raise D.Reject("parameters: a name twice")
    callees, root = desc["callees"], desc["definitions"][desc["root"]]["body"]["nodes"]
    nodes = [(input_id(callees[x["fn"]])[1], x["n"]) for x in root if input_id(callees[x["fn"]])]
    if [node for t in types for node in _lower(t)] != nodes:
        raise D.Reject("parameters: they do not lower to the program's input nodes")
    if j != _json([[name, t.to_json()] for name, t in program.fn.signature()[0]]):
        raise RuntimeError(f"the strategy drew a signature this file has no program for: {j}")


def _log_record(e) -> None:
    """`OneStage.logRecord`: an object whose `registered`, when truthy, lists objects with a string `value` and `root`."""
    r = e.get("registered") if isinstance(e, dict) else None
    if not isinstance(e, dict) or (r and not (isinstance(r, list) and all(
            isinstance(x, dict) and isinstance(x.get("value"), str) and isinstance(x.get("root"), str) for x in r))):
        raise D.Reject("a record of the log")


def decode_params(j) -> dict:
    v2 = isinstance(j, dict) and j.get("registered") is not None
    _exactly(j, PARAM_KEYS | {"parameters"} if v2 else PARAM_KEYS, "the parameters")
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
    if not isinstance(j["anchors"], list) or not isinstance(j["log"], list):
        raise D.Reject("anchors is not a list or log not a list of records")
    for e in j["log"]:
        _log_record(e)
    part = j["partition"]
    if part is not None and not (isinstance(part, dict) and isinstance(part.get("program"), str)
                                 and isinstance(part.get("query"), dict)):
        raise D.Reject("partition names no program digest or query")
    if v2:
        if not all(map(_nat, _obj(j["registered"], "registered").values())):
            raise D.Reject("registered: words that are not a natural number")
        _parameters(desc, PROGRAMS[sha], j["parameters"])
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
            and (j["served"] is None or nats(j["served"])) and (j["session_roots"] is None or isinstance(j["session_roots"], dict))):
        raise D.Reject("the evidence's fields")
    _registered_reads(j["verdicts"])
    return j


def _registered_reads(verdicts: dict) -> None:
    """`OneStage.evidenceOf`'s rule on the verdicts: a verifier of record that accepted reports its registered reads as
    objects with a string value and a root."""
    for v in verdicts.values():
        if v.get("of_record") and v.get("accepted") is True and isinstance(v.get("registered"), dict) and not all(
                isinstance(e, dict) and isinstance(e.get("value"), str) and "root" in e for e in v["registered"].values()):
            raise D.Reject("a verifier of record reports registered reads that are not objects with a string value and a root")


def _registration(p: dict, rec) -> tuple[str, ...]:
    return R.check(rec, p["program"], p["query"], p["law"], p["scheme"], p["ports"], p["leaf_layer"], p["anchors"], p["log"],
                   p["partition"], p["work_table"], p["registered"]).codes


def check(p: dict, c: tuple[dict, bytes], e: dict) -> tuple:
    """The outcome `OneStage.check` gives, from `one_stage`'s own checks: reject on R1-R7, the draw's checks, the record's
    checks or a verdict that ran and did not accept; undecided when only the verifier of record is missing or has not run;
    accept exactly when `audit_record` accepts, with the drawn and proved units, the bounds and the profile."""
    rec, _ = c
    if _registration(p, rec):
        return "reject", "registration"
    mine = DR.derive(p["program"], p["query"], p["law"], p["work_table"])
    if not DR.check_draw(e["draw"], rec, mine).ok:
        return "reject", "draw"
    out = audit.audit_record(rec, e["receipt"], e["draw"], mine, e["proved_units"], e["instances"], e["statement"],
                             e["verdicts"], p["delta"], e["sessions"], e["served"], e["session_roots"],
                             audit.session_registered(e["verdicts"], rec))
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


def registered(j):
    p = _params_of(_json(j))
    try:
        return G.registered_entries(p["program"], p["registered"] or {})
    except ValueError:
        return None


def domain(j):
    j = _json(j)
    if not (isinstance(j, dict) and isinstance(j.get("program"), str) and isinstance(j.get("value"), str)
            and all(_nat(j.get(k)) for k in ("words", "leaves", "word_bits"))):
        raise D.Reject("program or value is not a string, or words, leaves or word_bits not a natural number")
    try:
        return G.domain(j["program"], j["value"], j["words"], j["leaves"], j["word_bits"]).domain_id.hex()
    except ValueError as e:
        raise D.Reject(str(e)) from None


def check_draw(j):
    j = _json(j)
    if not isinstance(j, dict) or not _int(j.get("population")) or not isinstance(j.get("law"), dict):
        raise D.Reject("population is not an integer or law not an object")
    return _json(DR.check_draw(j.get("draw"), {"population": j["population"]}, j["law"]).to_json())


def check_proved(j):
    j = _json(j)
    ints = lambda x: isinstance(x, list) and all(map(_int, x))  # noqa: E731
    d = j.get("draw") if isinstance(j, dict) else None
    if not (isinstance(d, dict) and ints(d.get("units")) and ("closure" not in d or ints(d["closure"]))
            and ints(j.get("proved_units")) and _int(j.get("instances"))):
        raise D.Reject("not a draw of integer units, proved units and an instance count")
    return _json(DR.check_proved(d, j["proved_units"], j["instances"]).to_json())


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


def matches_public(j):
    j = _json(j)
    rec = j.get("record") if isinstance(j, dict) else None
    roots = rec.get("roots") if isinstance(rec, dict) else None
    if not (isinstance(roots, list) and all(isinstance(r, dict) and all(isinstance(r.get(k), str) for k in ("port", "root", "domain"))
                                            for r in roots)):
        raise D.Reject("a registered root is not an object with a string port, root and domain")
    if "scheme" not in rec or not _int(rec.get("population")):
        raise D.Reject("the record has no scheme or no integer population")
    strings = lambda x: isinstance(x, dict) and all(isinstance(v, str) for v in x.values())  # noqa: E731
    if not (isinstance(j.get("scheme"), str) and strings(j.get("roots")) and strings(j.get("domains")) and _int(j.get("population"))):
        raise D.Reject("scheme is not a string, roots or domains not an object of strings, or population not an integer")
    return _json(R.matches_public(rec, j["scheme"], j["roots"], j["domains"], j["population"]).to_json())


def session_registered(j):
    j = _json(j)
    vs = j.get("verdicts") if isinstance(j, dict) else None
    if not isinstance(vs, dict) or not all(isinstance(v, dict) for v in vs.values()):
        raise D.Reject("verdicts is not an object of objects")
    _registered_reads(vs)
    reg = j.get("registration")
    if "registration" in j:
        if not (isinstance(reg, dict) and isinstance(reg.get("roots"), list)
                and all(isinstance(r, dict) and isinstance(r.get("port"), str) for r in reg["roots"])):
            raise D.Reject("a registered root has no string port")
        if "registered" in reg and not (isinstance(reg["registered"], list) and all(
                isinstance(e, dict) and isinstance(e.get("value"), str) for e in reg["registered"])):
            raise D.Reject("a registered value has no string value")
    return {"registered": audit.session_registered(vs, reg)}


def check_public(j):
    """The service's public-items rule (`service.check_public`) on `{"record", "scheme", "query", "values", "public"}`, over a
    Spec holding only what the rule reads, with `scheme` the service's. Rejected where the Lean refuses: a malformed scheme,
    query, value or declared item, and where Python raises or would write a `repr` (an entry named by a list or an object,
    a stratum's template one)."""
    j = _json(j)
    if not (isinstance(j, dict) and isinstance(j.get("scheme"), str) and isinstance(j.get("query"), dict)):
        raise D.Reject("scheme is not a string or query not an object")
    vs, ps, rec = j.get("values"), j.get("public"), j.get("record")
    if not (isinstance(vs, list) and all(isinstance(v, dict) and isinstance(v.get("name"), str)
                                         and isinstance(v.get("registered"), bool) for v in vs)):
        raise D.Reject("a value is not a string name with a boolean")
    if not (isinstance(ps, list) and all(isinstance(i, str) for i in ps)):
        raise D.Reject("a declared item is not a string")
    if isinstance(rec, dict):
        law = rec.get("law")
        strata = law.get("strata") if isinstance(law, dict) else None
        if isinstance(strata, list) and any(isinstance(t.get("template") if isinstance(t, dict) else t, (list, dict))
                                            for t in strata):
            raise D.Reject("a stratum's template is a list or an object")
        for field_, key in (("roots", "port"), ("registered", "value")):
            es = rec.get(field_) if isinstance(rec.get(field_), list) else [rec.get(field_)]
            if field_ in rec and any(isinstance(e, dict) and isinstance(e.get(key), (list, dict)) for e in es):
                raise D.Reject("an entry named by a list or an object")
    spec = SimpleNamespace(check=SimpleNamespace(query=j["query"]), public=[SimpleNamespace(item=i) for i in ps])
    values = [SimpleNamespace(name=v["name"], registered=v["registered"]) for v in vs]
    with mock.patch.object(service, "SCHEME", j["scheme"]):
        return _json(service.check_public(spec, rec, values).to_json())


def check_verdicts(j):
    """The proof service's hold on an outcome's verdicts (`service.check_verdicts`) on `{"verdicts", "sessions",
    "outputs"}`. Rejected where the Lean refuses: malformed verdicts, sessions or outputs, and, once none is pending, a
    verdict of record's sessions that are neither a list nor empty (whose characters or keys Python's `list` would read)."""
    j = _json(j)
    vs, ss, outputs = (j.get(k) for k in ("verdicts", "sessions", "outputs")) if isinstance(j, dict) else (None, None, None)
    if not (isinstance(vs, dict) and all(isinstance(v, dict) for v in vs.values())):
        raise D.Reject("verdicts is not an object of objects")
    if not (isinstance(ss, list) and all(isinstance(x, str) for x in ss)):
        raise D.Reject("sessions is not a list of strings")
    if not (isinstance(outputs, list) and all(isinstance(o, str) for o in outputs) and len(set(outputs)) == len(outputs)):
        raise D.Reject("outputs are not distinct strings")
    of_record = [v for v in vs.values() if v.get("of_record")]
    pending = not of_record or any(v.get("accepted") is None for v in of_record)
    if not pending and any(v.get("sessions") and not isinstance(v["sessions"], list) for v in of_record):
        raise D.Reject("a verdict of record's sessions are not a list")
    return _json(service.check_verdicts(vs, ss, outputs).to_json())


RECORD_INPUT_KEYS = {"delta", "draw", "instances", "law", "proved_units", "receipt", "registration", "served", "session_roots",
                     "sessions", "statement", "verdicts"}
REGISTRATION_KEYS = ("anchors", "law", "partition", "population", "program", "protocol", "query", "roots", "scheme")


def _record_input(j) -> Fraction:
    """`Record.inputOf`'s domain, where `audit_record` reads its inputs as the Lean does; δ."""
    ints = lambda x: isinstance(x, list) and all(map(_int, x))  # noqa: E731
    _exactly(j, RECORD_INPUT_KEYS, "the record's input")
    reg, draw_ = j["registration"], j["draw"]
    if not (isinstance(reg, dict) and all(k in reg for k in REGISTRATION_KEYS) and _int(reg["population"])
            and isinstance(reg["anchors"], list) and isinstance(reg["roots"], list)
            and all(isinstance(r, dict) and isinstance(r.get("port"), str) and "root" in r and "leaves" in r for r in reg["roots"])
            and ("registered" not in reg or isinstance(reg["registered"], list) and all(
                isinstance(e, dict) and isinstance(e.get("value"), str) and all(k in e for k in ("root", "leaves", "domain"))
                for e in reg["registered"]))):
        raise D.Reject("the registration")
    if not (isinstance(j["receipt"], dict) and isinstance(draw_, dict) and ints(draw_.get("units"))
            and ("closure" not in draw_ or ints(draw_["closure"])) and ("population" not in draw_ or _int(draw_["population"]))
            and (draw_.get("law") != "subset" or "k" in draw_) and isinstance(j["law"], dict)):
        raise D.Reject("the receipt, the draw or the law")
    if not (ints(j["proved_units"]) and _int(j["instances"]) and isinstance(j["statement"], dict)
            and isinstance(j["verdicts"], dict) and all(isinstance(v, dict) for v in j["verdicts"].values())
            and isinstance(j["sessions"], list) and all(isinstance(x, str) for x in j["sessions"])
            and (j["served"] is None or ints(j["served"])) and (j["session_roots"] is None or isinstance(j["session_roots"], dict))):
        raise D.Reject("the evidence")
    _registered_reads(j["verdicts"])
    delta = ratio(j["delta"])
    if not 0 < delta < 1:
        raise D.Reject("delta is not in (0, 1)")
    return delta


def audit_record_of(j) -> dict:
    """`audit_record` on the record piece's input, δ the float it names (`Fraction(delta)` is one), the registered roots
    the verdicts' own for the registration's values, and Bernoulli's float bound null, which the Lean leaves to Python."""
    delta = _record_input(j)
    out = audit.audit_record(j["registration"], j["receipt"], j["draw"], j["law"], j["proved_units"], j["instances"],
                             j["statement"], j["verdicts"], float(delta), j["sessions"], j["served"], j["session_roots"],
                             audit.session_registered(j["verdicts"], j["registration"]))
    if j["draw"].get("law") == "bernoulli" and out["wrong_units"]:
        out["wrong_units"]["bound"] = None
    return out


def record(j):
    return codec.canonical_json(audit_record_of(_json(j))).decode()


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


def reads(rec: dict) -> dict:
    """A Lean verdict's `registered`: per port of the statement, the value it reads and the root it opened against (the
    record's)."""
    return {f"{e['value']}.{i}": {"root": e["root"], "value": e["value"]} for e in rec["registered"] for i in range(2)}


@st.composite
def audits(draw, setups=(*SETUPS, *REGISTERED_SETUPS)) -> tuple[dict, dict, dict]:
    """A valid audit: the verifier's parameters, the registration record and the evidence (verdicts mostly accepting). On a
    program with values to register it is mostly v2: the verifier expects some registered, the record registers them, the
    log's earlier record (if any) some of them under the same roots, and the verdict of record reports the sessions'
    reads opened against those."""
    s = draw(st.sampled_from(setups))
    values = draw(st.sampled_from([*s.values, None, *s.values, {}])) if isinstance(s, RegisteredSetup) else None
    held = draw(st.sampled_from([values, dict(list(values.items())[:1]), None])) if values else None
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
                        EARLIER, leaf_layer=LEAF, partition=part, work_table=table,
                        registered=s.entries(held) if held else ())]
    anchors = draw(st.sampled_from([[], ["weights"]]))
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports], WINDOW,
                   leaf_layer=LEAF, anchors=anchors, previous=R.digest(log[-1]) if log else None, partition=part,
                   work_table=table, registered=s.entries(values) if values else ())
    assert rec["law"] == mine
    params = {"anchors": anchors, "delta": draw(st.sampled_from(["1/1000", "1/10", "1/2", "999/1000"])), "law": law,
              "leaf_layer": LEAF, "log": log, "partition": part, "ports": s.ports, "program": s.desc, "query": s.query,
              "registered": values, "scheme": SCHEME, "work_table": table}
    if values is not None:
        params["parameters"] = s.signature
    d = drawn(draw, mine, n)
    proved = DR.proved(d)
    verdicts = {"lean": {"accepted": True, "digest": "aa" * 64, "of_record": True}}
    if values:
        verdicts["lean"]["registered"] = reads(rec)
    elif draw(st.booleans()):
        verdicts["lean"]["registered"] = {}
    if draw(st.booleans()):
        verdicts["upstream"] = {"accepted": draw(st.sampled_from([True, None])), "of_record": False}
    evidence = {"draw": d, "instances": len(proved), "proved_units": proved,
                "receipt": R.receipt(rec, "verifier-test", received=RECEIVED),
                "served": draw(st.sampled_from([None, list(d["units"])])),
                "session_roots": draw(st.sampled_from([None, dict(ROOTS)])), "sessions": ["ab" * 64],
                "statement": {"instances": len(proved)}, "verdicts": verdicts}
    return _json(params), _json(rec), _json(evidence)


def _registered(f):
    """A change to the registered reads the verdict of record reports (`f(reads or None, x)`)."""
    def change(a, x):
        lean = a["evidence"]["verdicts"]["lean"]
        lean["registered"] = f(lean.get("registered"), x)
    return change


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


#: One change each to the record (R1-R6; R7's are `ENTRIES`), the parameters (decodable or not) and the evidence (the draw, the record's checks,
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
    _roots(lambda rs, x: [{**rs[0], "root": ["AB" * 64, 5, "", "xy", rs[0]["root"] + "\n"][x % 5]}, *rs[1:]]),
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
DRAW_CHANGES = DRAW  # `DRAW` is the draw piece's port below
EVIDENCE = [
    _set("evidence", "receipt", lambda a, x: [{**a["evidence"]["receipt"], "registration": "00" * 64}, {}, 5][x % 3]),
    _set("evidence", "proved_units", lambda a, x: [a["evidence"]["proved_units"][:-1], a["evidence"]["proved_units"][::-1],
                                                   [-1], ["a"]][x % 4]),
    _set("evidence", "instances", lambda a, x: [a["evidence"]["instances"] + 1, -1, "1"][x % 3]),
    _set("evidence", "served", lambda a, x: [a["evidence"]["draw"]["units"][:-1], [a["evidence"]["draw"]["population"]],
                                             a["evidence"]["draw"]["units"][::-1], "s"][x % 4]),
    _set("evidence", "session_roots", lambda a, x: [{}, {"x": "00" * 64}, {"y": "ab" * 64}, {**ROOTS, "y": "ab" * 64},
                                                    {"x": ROOTS["x"]}, 5][x % 6]),
    _registered(lambda r, x: [{"w.0": {"root": "aa" * 64, "value": "weights"}}, 5, {"w.0": 5}, {"w.0": {"value": "weights"}},
                              {"w.0": {"root": "aa" * 64, "value": 1}}][x % 5]),
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


def _entries(f):
    def change(a, x):
        a["record"]["registered"] = f(a["record"]["registered"], x)
    return change


def _entry(f):
    """A change to the record's `x`-th registered entry."""
    return _entries(lambda es, x: [f(e, x) if i == x % len(es) else e for i, e in enumerate(es)])


def _held(a, x):
    """The log's last record registered its values under other roots (and the record's `previous` follows it)."""
    last = a["params"]["log"][-1]
    last = {**last, "registered": [{**e, "root": "12" * 64} for e in last["registered"]]}
    a["params"]["log"] = [*a["params"]["log"][:-1], last]
    a["record"]["previous"] = R.digest(last)


def _signature(sig: list, x: int):
    """A signature `Registered.params` refuses: not a list, a parameter dropped, added, retyped or named twice, a malformed
    type or pair, or another program's."""
    t = [["Value", 0], ["Value", 8, 1], ["Nope"], [], ["Record", [["a", ["Value", 8]], ["a", ["Value", 8]]]],
         ["Array", -1, ["Value", 8]], ["Array", True, ["Value", 8]], ["Tuple", ["Value", 8]], ["Record", [[5, ["Value", 8]]]],
         ["Value", 7], ["Array", 6, ["Value", 32]]][x % 11]
    i = x % len(sig)
    return [
        "parameters", sig[:-1], [*sig, ["Z", ["Value", 8]]], [*sig[:i], [sig[i][0], t], *sig[i + 1:]],
        [sig[0], [sig[0][0], sig[1][1]], *sig[2:]], [*sig[:i], [sig[i][0]], *sig[i + 1:]], [*sig[:i], [5, sig[i][1]], *sig[i + 1:]],
        REGISTERED_SETUPS[x % 2].signature if REGISTERED_SETUPS[x % 2].signature != sig else sig[1:],
        _json([[n, t_.to_json()] for n, t_ in SETUPS[0].program.fn.signature()[0]])][x % 9]


#: One change each to a v2 record's registered entries (R7), the verifier's registered values, signature or log, and the
#: roots the sessions opened registered reads against.
ENTRIES = [
    _entries(lambda es, x: ["registered", None, {}, [5], [[]]][x % 5]),
    _entry(lambda e, x: {k: v for k, v in e.items() if k != ["domain", "root", "value", "leaves", "words"][x % 5]}),
    _entry(lambda e, x: {**e, "extra": 1}),
    _entries(lambda es, x: es[:-1]), _entries(lambda es, x: es + es[:1]), _entries(lambda es, x: es[::-1]),
    _entry(lambda e, x: {**e, "value": ["nope", 5, "w", None, True, "x"][x % 6]}),
    _entry(lambda e, x: {**e, "root": ["AB" * 64, "ab" * 63, "ab" * 65, 5, "", "zz" * 64, None, "a" * 128,
                                       "a" * 127 + "\n"][x % 9]}),
    _entry(lambda e, x: {**e, "root": "12" * 64}),
    _entry(lambda e, x: {**e, "leaves": [e["leaves"] + 1, True, str(e["leaves"]), None, e["leaves"] - 1, 1][x % 6]}),
    _entry(lambda e, x: {**e, "words": [e["words"] + 1, True, str(e["words"]), 0, 1][x % 5]}),
    _entry(lambda e, x: {**e, "domain": ["00" * 64, e["domain"].upper(), 5, None][x % 4]}),
    _entry(lambda e, x: {**e, "word_bits": [16, 32, 24, True, "32", None, 0, 64][x % 8]}),
    _entry(lambda e, x: {k: v for k, v in e.items() if k != "word_bits"}),
    _set("record", "format", lambda a, x: [R.FORMAT, "verity/registration/v3"][x % 2]),
    lambda a, x: a["record"].pop("registered"),
    _held,
]
REGISTERED_PARAMS = [
    _set("params", "registered", lambda a, x: [{"W": 4}, {"W": 0}, {"E": 1}, {"M": 1}, {"V": 1}, {"nope": 1}, {"T": 3},
                                               {"U": 2}, {"W": -1}, {"W": True}, {"W": "2"}, [], "W", {}, {"T": 2}][x % 15]),
    _set("params", "registered", lambda a, x: {**a["params"]["registered"],
                                               **[{"D": 1}, {"x": 1}, {"T": 4}, {"U": 3}, {"W": 6}, {"x": 3}][x % 6]}),
    lambda a, x: (a["params"].pop("parameters"), a["params"].update(registered=None)),
    lambda a, x: a["params"].pop("parameters"),
    _set("params", "parameters", lambda a, x: _signature(a["params"]["parameters"], x)),
    _set("params", "log", lambda a, x: [{**a["record"], "registered": [
        "x", [5], [{"value": 5, "root": "r"}], [{"value": "W"}], {"W": "r"}, 1, [], 0, "", None, [{"root": 1, "value": "W"}],
        [{"root": "r", "value": "W", "x": 1}]][x % 12]}]),
    _held,
]
def _other_reads(r: dict, x: int) -> dict:
    """The verdict of record's registered reads, changed: one port's root another (so its value's two ports disagree), every
    root upper case, a value missing or added (one the record names as a root port, or one it doesn't name), only one
    port's, none."""
    ports = sorted(r)
    p = ports[x % len(ports)]
    v = r[p]["value"]
    return [{**r, p: {**r[p], "root": "12" * 64}}, {k: e for k, e in r.items() if e["value"] != v}, {},
            {**r, "extra.0": {"root": "ab" * 64, "value": "extra"}}, {**r, "x.0": {"root": ROOTS["x"], "value": "x"}},
            {k: {**e, "root": e["root"].upper()} for k, e in r.items()}, {p: r[p]}][x % 7]


REGISTERED_EVIDENCE = [
    _registered(lambda r, x: _other_reads(r, x) if r else [None, {}][x % 2]),
    _registered(lambda r, x: None),
    _set("evidence", "verdicts", lambda a, x: {**a["evidence"]["verdicts"], "lean-2": {
        **a["evidence"]["verdicts"]["lean"], "registered": _other_reads(a["evidence"]["verdicts"]["lean"].get("registered") or {
            "W.0": {"root": "ab" * 64, "value": "W"}}, x)}}),
]
GROUPS = [RECORD, PARAMS, DRAW, EVIDENCE, VERDICTS_OF, ENTRIES, REGISTERED_PARAMS, REGISTERED_EVIDENCE]


@st.composite
def changed_audits(draw, groups=GROUPS, most: int = 2, least: int = 0, setups=(*SETUPS, *REGISTERED_SETUPS)) -> dict:
    """An audit as `{"params", "record", "evidence"}`, with `least` to `most` changes, each from one of `groups`; a change
    that doesn't apply to what an earlier one left (a field it removed, a list it replaced) is skipped."""
    params, rec, evidence = draw(audits(setups))
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
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                    "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
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
_REGISTRATION_OF = lambda a: {"params": a["params"], "record": a["record"]}  # noqa: E731
REGISTRATION = D.Port(PIECES, "registration", registration, st.one_of(
    _audit_inputs(_REGISTRATION_OF, [RECORD, PARAMS, ENTRIES, REGISTERED_PARAMS], 2),
    changed_audits([ENTRIES], 1, 1, REGISTERED_SETUPS).map(_REGISTRATION_OF),
    changed_audits([REGISTERED_PARAMS], 1, 1, REGISTERED_SETUPS).map(_REGISTRATION_OF),
    changed_audits([ENTRIES, RECORD], 2, 2, REGISTERED_SETUPS).map(_REGISTRATION_OF),
    st.builds(lambda a, r: {"params": a["params"], "record": r}, changed_audits(most=0), JSON)), package=PACKAGE)


@st.composite
def registered_values(draw) -> dict:
    """The verifier's parameters expecting values registered: a v2 audit's with one change, or any `{parameter: words}` of
    its program's parameters (and one it lacks)."""
    a = draw(changed_audits([REGISTERED_PARAMS, PARAMS], 1, 0, REGISTERED_SETUPS))
    if "parameters" in a["params"] and draw(st.booleans()):
        names = [p[0] for p in a["params"]["parameters"] if isinstance(p, list) and p and isinstance(p[0], str)]
        a["params"]["registered"] = draw(st.dictionaries(st.sampled_from([*names, "nope"]), st.integers(0, 7), max_size=3))
    return {"params": a["params"]}


REGISTERED_ENTRIES = D.Port(PIECES, "registered", registered, st.one_of(
    registered_values(), _audit_inputs(lambda a: {"params": a["params"]}, [PARAMS]), JSON), package=PACKAGE)
TEXT = st.text(max_size=4) | st.sampled_from(["W", "x", "é", "\x00", "\U0001f600", '"'])
def _domains(widths):
    return st.builds(lambda program, value, words, leaves, b: {"leaves": leaves, "program": program, "value": value,
                                                               "word_bits": b, "words": words},
                     st.sampled_from([s.sha512 for s in REGISTERED_SETUPS]) | TEXT, TEXT, st.integers(0, 70),
                     st.integers(0, 70) | st.integers(0, 1 << 40), st.sampled_from(widths))


DOMAIN = D.Port(PIECES, "domain", domain, st.one_of(_domains([16, 32]), st.one_of(
    _domains([0, 8, 24, 48]),
    st.builds(lambda k, v: {"leaves": 3, "program": "p", "value": "W", "word_bits": 16, "words": 2, k: v},
              st.sampled_from(["leaves", "program", "value", "word_bits", "words"]), JSON), JSON)), package=PACKAGE)
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



def _public(a: dict, change, x: int) -> dict:
    """`matches_public`'s input from an audit: its record, and the verifier's public file read off it, maybe changed."""
    rec = a["record"]
    roots = rec.get("roots") if isinstance(rec.get("roots"), list) else []
    ok = all(isinstance(r, dict) and "port" in r for r in roots)
    pub = {"domains": {r["port"]: r.get("domain") for r in roots} if ok else {}, "population": rec.get("population"),
           "record": rec, "roots": {r["port"]: r.get("root") for r in roots} if ok else {}, "scheme": rec.get("scheme")}
    return change(pub, x)


PUBLIC_CHANGES = [
    lambda p, x: p,
    lambda p, x: {**p, "roots": {**p["roots"], sorted(p["roots"] or ["x"])[0]: "00" * 64}},
    lambda p, x: {**p, "domains": {**p["domains"], sorted(p["domains"] or ["x"])[0]: "00" * 64}},
    lambda p, x: {**p, "domains": {**p["domains"], "z": "00" * 64}},
    lambda p, x: {**p, "roots": {**p["roots"], "z": "00" * 64}},
    lambda p, x: {**p, "roots": dict(list(p["roots"].items())[:-1])},
    lambda p, x: {**p, "scheme": ["frame-v3/blake3", 5][x % 2]},
    lambda p, x: {**p, "population": [p["population"] + 1 if _int(p["population"]) else 1, "6", True][x % 3]},
    lambda p, x: {**p, ["roots", "domains"][x % 2]: [5, {"x": 5}, {"x": None}][x % 3]},
    lambda p, x: {k: v for k, v in p.items() if k != sorted(p)[x % 5]},
]
MATCHES_PUBLIC = D.Port(PIECES, "matches_public", matches_public, st.one_of(
    st.builds(_public, changed_audits(most=0), st.sampled_from(PUBLIC_CHANGES), st.integers(0, 63)),
    st.builds(_public, changed_audits([RECORD], most=1, least=1), st.sampled_from(PUBLIC_CHANGES[:2]), st.integers(0, 63)),
    st.builds(lambda a, x: _public(a, PUBLIC_CHANGES[0], x), changed_audits([ENTRIES], 1, 1, REGISTERED_SETUPS), st.integers(0, 3)),
    JSON), package=PACKAGE)


def _public_items(a: dict, change, x: int) -> dict:
    """`check_public`'s input from an audit: its record, the check's query, the verifier's ports and registered values as
    the Spec's values, and every item they and the templates may show declared; maybe changed."""
    p = a["params"]
    values = [{"name": n, "registered": False} for n in sorted(p["ports"])] + [
        {"name": n, "registered": True} for n in sorted(p.get("registered") or {})]
    public = ["population", "window", "leaf_layer", "anchors", *(f"population:{t}" for t in PROGRAMS_TEMPLATES),
              *(f"{k}:{v['name']}" for v in values for k in ("leaves", "domain", *(("words", "word_bits") if v["registered"] else ())))]
    return change({"public": public, "query": p["query"], "record": a["record"], "scheme": SCHEME, "values": values}, x)


def _record_with(c: dict, **fields) -> dict:
    return {**c, "record": {**c["record"], **fields}}


#: One change each to the rule's input: a declared item or every one dropped, a value dropped, added, twice or the other
#: kind, another query or scheme, a record that is not one, a stratified law's templates, and entries the rule names.
PUBLIC_ITEMS = [
    lambda c, x: c,
    lambda c, x: {**c, "public": [i for n, i in enumerate(c["public"]) if n != x % len(c["public"])]},
    lambda c, x: {**c, "public": []},
    lambda c, x: {**c, "values": [v for n, v in enumerate(c["values"]) if n != x % len(c["values"])]},
    lambda c, x: {**c, "values": [*c["values"], {"name": ["z", "W", "x"][x % 3], "registered": x % 2 == 0}]},
    lambda c, x: {**c, "values": [*c["values"], c["values"][x % len(c["values"])]]},
    lambda c, x: {**c, "values": [{**v, "registered": not v["registered"]} if n == x % len(c["values"]) else v
                                  for n, v in enumerate(c["values"])]},
    lambda c, x: {**c, "query": QUERIES[x % len(QUERIES)]},
    lambda c, x: {**c, "scheme": "frame-v3/blake3"},
    lambda c, x: {**c, "record": [None, "a record", [], 5][x % 4]},
    lambda c, x: _record_with(c, law={"law": "stratified", "strata": [
        {"k": 1, "template": [PAIR.id, None, 5, True, "x", -2, [1], {"a": 1}, 0][x % 9]}, [TRIPLE.id, None, False, 7][x % 4]]}),
    lambda c, x: _record_with(c, law={"law": "stratified", "strata": "s"}),
    lambda c, x: _record_with(c, roots=[[{"port": [1]}, {"port": {"a": 1}}, {"port": 5}, {"port": None}, {"root": "r"}, "x",
                                         {"port": "x", "call": 1, "root": "r"}][x % 7], *c["record"]["roots"]]),
    lambda c, x: _record_with(c, registered=[{"value": [["W"], {}, 1, None, "W", "out"][x % 6], "leaves": 1, "root": "r"}]),
    lambda c, x: _record_with(c, roots={"port": "x", "root": "r"}),
]
CHECK_PUBLIC = D.Port(PIECES, "check_public", check_public, st.one_of(
    st.builds(_public_items, changed_audits(most=0), st.sampled_from(PUBLIC_ITEMS), st.integers(0, 63)),
    st.builds(_public_items, changed_audits([RECORD], most=1, least=1), st.sampled_from(PUBLIC_ITEMS[:3]), st.integers(0, 63)),
    st.builds(_public_items, changed_audits([ENTRIES], 1, 1, REGISTERED_SETUPS), st.sampled_from(PUBLIC_ITEMS[:3]),
              st.integers(0, 63)),
    st.builds(lambda c, k: {kk: v for kk, v in _public_items(c, PUBLIC_ITEMS[0], 0).items() if kk != k}, changed_audits(most=0),
              st.sampled_from(["public", "query", "record", "scheme", "values"])),
    JSON), package=PACKAGE)


OUTCOME_SESSIONS = ["5e" * 64, "a1" * 64]


def _outcome_verdicts(n: int, accepted: list, outputs: list, change, x: int) -> dict:
    """`check_verdicts`' input: `n` verdicts of record, each about the delivered sessions and reporting the public outputs
    alike, maybe changed."""
    delivered = OUTCOME_SESSIONS[:1 + x % 2]
    vs = {name: {"accepted": accepted[i % len(accepted)], "of_record": True, "public": {o: 12 for o in outputs},
                 "sessions": list(delivered)} for i, name in enumerate(["lean", "rust", "upstream"][:n])}
    return change({"outputs": outputs, "sessions": delivered, "verdicts": vs}, x)


def _one_verdict(f):
    """A change to the `x`-th verdict."""
    def change(c, x):
        name = sorted(c["verdicts"])[x % len(c["verdicts"])]
        return {**c, "verdicts": {**c["verdicts"], name: f(c["verdicts"][name], c, x)}}
    return change


#: One change each to the verdicts' share of an outcome: a verdict's sessions, public outputs, `of_record` or `accepted`,
#: a verdict that is not of record (whose sessions nobody reads), none, the outputs and the delivered sessions.
OUTCOME_VERDICTS = [
    lambda c, x: c,
    _one_verdict(lambda v, c, x: {**v, "sessions": [[], None, "s", {OUTCOME_SESSIONS[0]: 1}, 5, [OUTCOME_SESSIONS[1]],
                                                    c["sessions"][::-1], c["sessions"] * 2, [True]][x % 9]}),
    _one_verdict(lambda v, c, x: {**v, "public": [{o: 13 for o in c["outputs"]}, {}, None, "p", {**v["public"], "z": 1},
                                                  [1], {o: [12] for o in c["outputs"]}][x % 7]}),
    _one_verdict(lambda v, c, x: {**v, "of_record": [False, 0, "", None, [], 1, "yes"][x % 7]}),
    _one_verdict(lambda v, c, x: {**v, "accepted": [None, False, 1, "no"][x % 4]}),
    _one_verdict(lambda v, c, x: {k: e for k, e in v.items() if k != sorted(v)[x % len(v)]}),
    lambda c, x: {**c, "verdicts": {}},
    lambda c, x: {**c, "verdicts": {**c["verdicts"], "other": {"accepted": True, "sessions": 5, "public": None}}},
    lambda c, x: {**c, "outputs": [[], ["credited", "x"], ["z"], ["credited", "credited"]][x % 4]},
    lambda c, x: {**c, "sessions": [[], OUTCOME_SESSIONS[::-1], [OUTCOME_SESSIONS[1]], [5]][x % 4]},
]
CHECK_VERDICTS = D.Port(PIECES, "check_verdicts", check_verdicts, st.one_of(
    st.builds(_outcome_verdicts, st.integers(1, 3), st.lists(st.sampled_from([True, False]), min_size=1, max_size=3),
              st.sampled_from([[], ["credited"], ["credited", "x"]]), st.sampled_from(OUTCOME_VERDICTS), st.integers(0, 63)),
    st.builds(lambda vs, ss, o: {"outputs": o, "sessions": ss, "verdicts": vs}, VERDICTS,
              st.sampled_from([OUTCOME_SESSIONS[:1], []]), st.sampled_from([[], ["credited"]])),
    JSON), package=PACKAGE)


def _reads_input(a: dict, with_registration: bool) -> dict:
    out = {"verdicts": a["evidence"].get("verdicts")}
    return out | {"registration": a["record"]} if with_registration else out


SESSION_REGISTERED = D.Port(PIECES, "session_registered", session_registered, st.one_of(
    st.builds(_reads_input, changed_audits([REGISTERED_EVIDENCE, VERDICTS_OF], 1, 0, REGISTERED_SETUPS), st.booleans()),
    st.builds(_reads_input, changed_audits([EVIDENCE, REGISTERED_EVIDENCE, ENTRIES, RECORD], 1, 1), st.booleans()),
    st.builds(lambda vs, r: {"registration": r, "verdicts": vs}, VERDICTS, JSON),
    st.builds(lambda vs: {"verdicts": vs}, JSON), JSON), package=PACKAGE)


def _record_case(a: dict, delta: str) -> dict:
    """`audit_record`'s arguments from an audit: the verifier's derived law (or the record's, when the parameters don't
    derive one), and the evidence, with δ."""
    p, e = a["params"], a["evidence"]
    try:
        law = DR.derive(PROGRAMS[PO.program_sha512(p["program"])], p["query"], p["law"], p["work_table"])
    except Exception:  # noqa: BLE001 - a changed parameter the verifier would refuse: the record's own law stands in
        law = a["record"].get("law") if isinstance(a["record"], dict) else None
    out = {"delta": delta, "law": law, "registration": a["record"]}
    return out | {k: e.get(k) for k in ("draw", "instances", "proved_units", "receipt", "served", "session_roots", "sessions",
                                         "statement", "verdicts")}


#: δ as the exact ratio of a double (`str(Fraction(delta))`), as the switch passes it.
EXACT_DELTA = st.floats(2 ** -40, 1 - 2 ** -40).map(lambda f: str(Fraction(f))) | st.sampled_from(
    [str(Fraction(f)) for f in (0.5, 0.25, 0.1, 0.05, 0.01, 1e-6, 2 ** -20, 0.999)])
RECORD_PORT = D.Port(PIECES, "record", record, st.one_of(
    st.builds(_record_case, changed_audits(most=0), EXACT_DELTA),
    st.builds(_record_case, changed_audits([EVIDENCE, DRAW_CHANGES, VERDICTS_OF, REGISTERED_EVIDENCE], 1, 1), EXACT_DELTA),
    st.builds(_record_case, changed_audits([RECORD, ENTRIES, REGISTERED_EVIDENCE], 1, 1, REGISTERED_SETUPS), EXACT_DELTA),
    st.builds(_record_case, changed_audits(most=2, least=2), EXACT_DELTA),
    st.builds(_record_case, changed_audits(most=0), st.sampled_from(["0", "1", "3/2", "a", "1/0", "2/4", 5, None])),
    st.builds(lambda a, k: {kk: v for kk, v in _record_case(a, "1/4").items() if kk != k}, changed_audits(most=0),
              st.sampled_from(sorted(RECORD_INPUT_KEYS))),
    JSON), package=PACKAGE)

pieces = D.server_fixture(DECODE)


PIECE_RUNS = [(DECODE, 300), (PARTITION, 400), (REGISTRATION, 700), (REGISTERED_ENTRIES, 300), (DOMAIN, 300),
              (CHECK_DRAW, 500), (DRAW, 500), (CHECK_PROVED, 300), (BOUND, 500), (VERDICT, 300), (MATCHES_PUBLIC, 400),
              (SESSION_REGISTERED, 400), (RECORD_PORT, 600), (CHECK_PUBLIC, 500), (CHECK_VERDICTS, 400)]


@pytest.mark.parametrize("port, examples", PIECE_RUNS, ids=[p.fn for p, _ in PIECE_RUNS])
def test_a_piece_agrees(pieces, port, examples):
    stats = D.run(port, examples=examples, server=pieces)
    assert stats.outputs > examples // 4 and stats.rejects > 0, stats


R7_CODES = {"R7-registered", "R7-values", "R7-root", "R7-leaves", "R7-words", "R7-domain", "R7-word_bits", "R7-held"}


def _v2_audit(s: RegisteredSetup, values: dict) -> dict:
    """One valid v2 audit of `s` expecting `values` registered, the log's earlier record registering them under the same
    roots."""
    law = {"k": 1, "law": "subset"}
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    earlier = R.record(s.program, s.query, law, SCHEME, roots, EARLIER, leaf_layer=LEAF, registered=s.entries(values))
    rec = R.record(s.program, s.query, law, SCHEME, roots, WINDOW, leaf_layer=LEAF, previous=R.digest(earlier),
                   registered=s.entries(values))
    params = {"anchors": [], "delta": "1/10", "law": law, "leaf_layer": LEAF, "log": [earlier], "parameters": s.signature,
              "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": values,
              "scheme": SCHEME, "work_table": None}
    evidence = {"draw": {**law, "population": s.population, "units": [0]}, "instances": 1, "proved_units": [0],
                "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                "sessions": [], "statement": {},
                "verdicts": {"lean": {"accepted": True, "of_record": True, "registered": reads(rec)}}}
    return _json({"evidence": evidence, "params": params, "record": rec})


def test_every_r7_code_is_reached(pieces):
    """The record's and the parameters' registered changes on v2 audits of both programs, Lean and Python agreeing on each:
    together they reach every R7 code, which the drawn runs only probably do, and the unchanged audits none."""
    seen: set[str] = set()
    for s in REGISTERED_SETUPS:
        for values in s.values[:2]:
            a = _v2_audit(s, values)
            assert D.replay(REGISTRATION, _REGISTRATION_OF(a), pieces) == ("output", D.canonical({"codes": []}))
            for change, x in ((c, x) for c in ENTRIES + REGISTERED_PARAMS for x in range(8)):
                b = copy.deepcopy(a)
                try:
                    change(b, x)
                except (KeyError, TypeError, IndexError, ValueError, AttributeError):
                    continue
                kind, out = D.replay(REGISTRATION, _json(_REGISTRATION_OF(b)), pieces)
                seen |= set(json.loads(out)["codes"]) if kind == "output" else set()
    assert R7_CODES <= seen, sorted(R7_CODES - seen)


def test_a_root_with_a_trailing_newline_is_refused(pieces):
    """A registered root and a registered value's 128-character root (127 hex digits and a newline), each with a newline
    last, are refused by Lean and Python alike: Python's `$` matches before a final newline, which `fullmatch` doesn't."""
    s = REGISTERED_SETUPS[0]
    a = _v2_audit(s, s.values[1])
    r3, r7 = copy.deepcopy(a), copy.deepcopy(a)
    r3["record"]["roots"][0]["root"] += "\n"
    r7["record"]["registered"][0]["root"] = r7["record"]["registered"][0]["root"][:127] + "\n"
    for b, codes in ((r3, ["R3-root"]), (r7, ["R7-root", "R7-held"])):
        assert D.replay(REGISTRATION, _json(_REGISTRATION_OF(b)), pieces) == ("output", D.canonical({"codes": codes}))


def test_every_public_items_code_is_reached(pieces):
    """Each change to the rule's input on a v1 audit and a v2 one, Lean and Python agreeing on each: together they reach
    every code of `check_public` and a reject, which the drawn runs only probably do, and the unchanged audits none."""
    s = REGISTERED_SETUPS[0]
    v1 = {"params": {"ports": SETUPS[2].ports, "query": SETUPS[2].query}, "record": FIXED[0]}
    seen: set[str] = set()
    for a in (_json(v1), _v2_audit(s, s.values[1])):
        assert D.replay(CHECK_PUBLIC, _public_items(a, PUBLIC_ITEMS[0], 0), pieces) == (
            "output", D.canonical({"codes": [], "detail": {}, "ok": True}))
        for change, x in ((c, x) for c in PUBLIC_ITEMS for x in range(9)):
            kind, out = D.replay(CHECK_PUBLIC, _public_items(copy.deepcopy(a), change, x), pieces)
            seen |= set(json.loads(out)["codes"]) if kind == "output" else {kind}
    assert seen == {"keys", "unsalted", "query", "roots", "undeclared", "reject"}, sorted(seen)


def test_a_v2_audit_reads_the_sessions_registered_roots(server):
    """A v2 audit of each program accepts when the verifier of record reports the sessions' registered reads opened against
    the registered roots (and a value the registration doesn't name is not its to read), and rejects them opened against
    another root, a value's reads missing, a root port's value read as registered, or none reported."""
    for s in REGISTERED_SETUPS:
        a = _v2_audit(s, s.values[1])
        opened = a["evidence"]["verdicts"]["lean"]["registered"]
        p = sorted(opened)[0]
        v = opened[p]["value"]
        for got, want in ((opened, "accept"), ({**opened, "extra.0": {"root": "ab" * 64, "value": "extra"}}, "accept"),
                          ({**opened, p: {**opened[p], "root": "12" * 64}}, "reject"),
                          ({k: e for k, e in opened.items() if e["value"] != v}, "reject"),
                          ({**opened, "x.0": {"root": ROOTS["x"], "value": "x"}}, "reject"), ({}, "reject"), (None, "reject")):
            lean = {**a["evidence"]["verdicts"]["lean"], "registered": got}
            case = _json(_case({**a, "evidence": {**a["evidence"], "verdicts": {"lean": lean}}}))
            assert D.replay(ONE_STAGE, case, server)[0] == want, (s.program.fn.id, got)


def _accepted_record_case(kind: str, delta: float) -> dict:
    s = SETUPS[3]
    law = {"law": "bernoulli", "p": "1/2"} if kind == "bernoulli" else {"k": 3, "law": kind}
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}} if kind == "work" else None
    mine = DR.derive(s.program, s.query, law, table)
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, work_table=table)
    units = {"subset": [0, 3, 4], "bernoulli": [1, 2, 4]}.get(kind) or sorted(
        u for st_ in mine["strata"] for u in [u for lo, hi in st_["units"] for u in range(lo, hi)][:st_["k"]])
    d = {**mine, "population": s.population, "units": units}
    if kind == "work":
        d["closure"] = [u for u in range(s.population) if u not in units][:1]
    proved = DR.proved(d)
    return _json({"delta": str(Fraction(delta)), "draw": d, "instances": len(proved), "law": mine, "proved_units": proved,
                  "receipt": R.receipt(rec, "v", received=RECEIVED), "registration": rec, "served": list(units),
                  "session_roots": dict(ROOTS), "sessions": ["ab" * 64, "cd" * 64], "statement": {"instances": len(proved)},
                  "verdicts": {"lean": {"accepted": True, "digest": "aa" * 64, "of_record": True},
                               "upstream": {"accepted": True, "of_record": False}}})


def test_the_record_is_compared_field_by_field(pieces):
    """The whole audit record, field by field, on an accepted audit of each law (with a work law's closure), the same audit
    rejected by its served draw and its sessions' roots, and an accepted v2 audit: every field the Lean writes is
    `audit_record`'s, Bernoulli's float bound aside (null in the Lean, a number in Python)."""
    cases = [_accepted_record_case(kind, delta) for kind in ("subset", "bernoulli", "stratified", "work")
             for delta in (0.05, 0.1, 2 ** -20)]
    rejected = _accepted_record_case("subset", 0.25)
    cases.append({**rejected, "served": rejected["served"][:-1], "session_roots": {**ROOTS, "x": "00" * 64}})
    v2 = _v2_audit(REGISTERED_SETUPS[0], REGISTERED_SETUPS[0].values[1])
    cases.append(_record_case(v2, str(Fraction(0.1))))
    for case in cases:
        kind, line = D.replay(RECORD_PORT, case, pieces)
        assert kind == "output", line
        lean, python = json.loads(json.loads(line)), audit_record_of(copy.deepcopy(case))
        assert set(lean) == set(python)
        for field in sorted(python):
            assert codec.canonical_json(lean[field]) == codec.canonical_json(python[field]), (case["draw"]["law"], field)
        if case["draw"]["law"] == "bernoulli":
            whole = audit.audit_record(*(case[k] for k in ("registration", "receipt", "draw", "law", "proved_units",
                                                           "instances", "statement", "verdicts")), float(Fraction(case["delta"])))
            assert lean["wrong_units"]["bound"] is None and isinstance(whole["wrong_units"]["bound"], int | float)
    assert [json.loads(json.loads(D.replay(RECORD_PORT, c, pieces)[1]))["accepted"] for c in cases] == [True] * 12 + [False, True]
