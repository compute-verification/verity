"""Sampled proofs' one-stage audit in Lean (`core_sampled_proofs`) against `one_stage`'s Python: the whole protocol through
its own `serve` (`sampled-proofs-one-stage`, a `ProtocolPort`), and its pieces through `sampled-proofs-difftest` (`Port`s),
on drawn programs, laws, records, draws and verdicts, valid ones and one or two changes to each.

The reference is `one_stage` itself (`registration.check`, `draw.derive`, `check_draw`, `audit.audit_record`, `profile`,
`wrong_units`, `registered.registered_entries` and `domain`) behind the protocol's decoders, which this file writes as the
Lean reads them (`OneStage.paramsOfJson`, `commitmentOf`, `evidenceOf`), for the `draw` piece `two_stage.draw_first` on
`OwnRandomness` over the given bytes, for the `select` piece the challenger's draw from a Spec's selection rule on
(`service.selection_law`, `draw.derive` and `service.select_draw` on fixed coins, with the bytes it read), for the
`slots` piece that draw in the claims' slots, read off `partition.strata`, `partition.unit_gates` and the root's input
nodes (`slots`), for `check_public` and `check_verdicts` the proof service's own (`service.check_public`,
`service.check_verdicts`), and for the `run` piece (a registration's whole run, `Run.run`) those references in a
`Stream`'s order. The `partition`, `select` and `slots` pieces also run on random partitions: mixed population
programs of random members of three templates, and rarely of `FORK` and of `IDENTITY`, of no gate, which one-stage
refuses, under template queries over them, and programs of random root Calls (`call_program`) under `Q_word` and `Q_call`, a 32-bit
Call among them, which one-stage refuses at either, and `SHARED`, whose `Q_call` unit at `X = 16` holds a constant
another unit reads. Records are v1 and, on the programs with root parameters to register
(`REGISTERED_SETUPS`, 16- and 32-bit), mostly v2: the verifier expects values registered (`registered`, with its program
function's `parameters`), and the record (and an earlier one in the log) registers `Registered.commit`'s entries.
The verifier's program is looked up among this file's programs by its descriptor. Inputs
stay inside what Python reads as Lean does: integers that are not booleans where Python compares with `==`, and integer
populations in a draw.
"""

from __future__ import annotations

import copy
import functools
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
from verity.core.protocols.sampled_proofs.one_stage import (audit, draw as DR, hello as H, partition as P, registered as G,
                                                            registration as R)
from verity.core.protocols.sampled_proofs.two_stage import (OwnRandomness, ReplayUnit, Stage, TwoStageLaw, draw_first,
                                                            select_proof_units, select_replay_units)
from verity.ml.circuits import codec, partition_object as PO
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program, input_id, is_input
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Record, Tuple, Value

PACKAGE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

XOR8 = PrimitiveDefinition("CoreSampledProofsXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("CoreSampledProofsPair", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("CoreSampledProofsTriple", 0, (), lambda S: ((("x", Array(3, Value(8))),), Value(8)),
                                  lambda B, S, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
QUAD = bind(CompositeDefinition("CoreSampledProofsQuad", 0, (), lambda S: ((("x", Array(4, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), B.call(XOR8, x[2], x[3])),
                                register=False))
TEMPLATES = [PAIR, TRIPLE, QUAD]
XOR32 = PrimitiveDefinition("CoreSampledProofsXor32", 1, (("a", Value(32)), ("b", Value(32))), Value(32),
                            lambda a, b: a ^ b, register=False)


def _fork(B, S, x):
    a = B.call(XOR8, x[0], x[1])
    first, last = B.call(XOR8, a, x[2]), B.call(XOR8, a, x[3])
    return tuple_of(first, B.call(XOR8, x[2], x[3]), last)


#: Three outputs, two units at `Q_word`'s `X = 16`: the first two, whose gates the last one's interleaves, and the last;
#: the interior value both read is committed, a unit of its own.
FORK = bind(CompositeDefinition("CoreSampledProofsFork", 0, (),
                                lambda S: ((("x", Array(4, Value(8))),), Tuple(Value(8), Value(8), Value(8))), _fork,
                                register=False))
#: A template of no gate, which one-stage refuses.
IDENTITY = bind(CompositeDefinition("CoreSampledProofsIdentity", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                    lambda B, S, x: x[0], register=False))
#: The templates a population's members are instances of: `TEMPLATES`, `FORK` and `IDENTITY`.
MEMBERS = TEMPLATES + [FORK, IDENTITY]
CONST8 = PrimitiveDefinition("CoreSampledProofsConst8", 1, (), Value(8), lambda: 0x5A, register=False)


def _held(B, S, x):
    a = B.call(XOR8, x[0], x[1])
    return tuple_of(a, B.call(XOR8, a, x[1]), B.call(CONST8))


def _shared(B, S, x):
    s = B.call(HELD, x)
    return tuple_of(s[0], s[1], B.call(XOR8, s[0], s[2]))


HELD = bind(CompositeDefinition("CoreSampledProofsHeld", 0, (),
                                lambda S: ((("x", Array(2, Value(8))),), Tuple(Value(8), Value(8), Value(8))), _held,
                                register=False))
#: A Call whose first `Q_call` unit at `X = 16`, `HELD`, returns 16 bits and holds a constant the last gate reads: 24
#: committed bits, which `Q_call`'s width rule doesn't count; at `X = 8` the constant is in no unit.
SHARED = bind(CompositeDefinition("CoreSampledProofsShared", 0, (),
                                  lambda S: ((("x", Array(2, Value(8))),), Tuple(Value(8), Value(8), Value(8))), _shared,
                                  register=False))
#: The Calls a `call_program` may make: the templates, the fork, the 8- and 32-bit primitives and the shared constant's.
CALLEES = [PAIR, TRIPLE, FORK, XOR8, XOR32, SHARED]
CALLEE_IDS = [codec.descriptor_id(f) for f in CALLEES]
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


def call_program(members) -> Program:
    """A root of one Call per member, an index into `CALLEES`, each over a parameter of its own (a primitive's two values
    as an array of two), returning their results: what `Q_word` and `Q_call` cut, Call by Call."""
    def param(f):
        return Array(2, f.signature()[0][0][1]) if isinstance(f, PrimitiveDefinition) else f.signature()[0][0][1]

    def call(B, f, x):
        return B.call(f, x[0], x[1]) if isinstance(f, PrimitiveDefinition) else B.call(f, x)

    fns = [CALLEES[m] for m in members]
    defn = CompositeDefinition(f"CoreSampledProofsCalls{''.join(map(str, members))}", 0, (),
                               lambda S: (tuple((f"p{i}", param(f)) for i, f in enumerate(fns)), Tuple(*(f.signature()[1] for f in fns))),
                               lambda B, S, *xs: tuple_of(*(call(B, f, x) for f, x in zip(fns, xs))), register=False)
    return Program(bind(defn))


SETUPS = [Setup(P.population_program(PAIR, n), P.template_instance_query(PAIR.id)) for n in (1, 3, 6)] + [
    Setup(P.mixed_population_program([(PAIR, 2), (TRIPLE, 3)]), P.template_instances_query([PAIR.id, TRIPLE.id])),
    Setup(P.mixed_population_program([(PAIR, 1), (TRIPLE, 2), (PAIR, 2)]), P.template_instances_query([TRIPLE.id, PAIR.id])),
    Setup(call_program((0, 2, 0, 3)), PO.query()), Setup(call_program((1, 2, 3)), PO.call_query(8))]


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
#: The strata a work table may name: the templates' and the Calls'.
WORK_IDS = list(dict.fromkeys([t.id for t in TEMPLATES] + CALLEE_IDS))
#: `Q_word` and `Q_call` as one-stage applies them, `X = 16` and `X <= 16`, and at the widths it refuses.
CALL_QUERIES = [PO.query(), PO.query(W=16), PO.call_query(16), PO.call_query(8), PO.call_query(4), PO.call_query(32),
                PO.query(X=8), PO.query(version=2)]
#: Queries a verifier may hold: the setups' own, `CALL_QUERIES`, and ones core refuses or one-stage doesn't apply.
QUERIES = [s.query for s in SETUPS] + CALL_QUERIES + [
    {}, {**PO.query(), "params": {"W": 32, "X": 70000}}, P.template_instance_query("Other_v1"),
    P.template_instances_query([PAIR.id]), {"name": "Q_template_instance", "version": 1, "params": {"template": PAIR.id}},
    {"name": "Nope", "version": 0, "params": {}}, {"name": "Q_template_instance", "version": 0, "params": {}}]


@functools.cache
def _population(members: tuple[tuple[int, int], ...]) -> str:
    """The canonical descriptor of the mixed population of `members`, `(template, instances)` pairs with the template an
    index into `MEMBERS`, its program put among `PROGRAMS`."""
    program = P.mixed_population_program([(MEMBERS[t], n) for t, n in members])
    PROGRAMS[PO.program_sha512(program)] = program
    return D.canonical(codec.encode_program(program))


@functools.cache
def _interior_setup(n: int, query: str) -> Setup:
    """`PAIR`'s population over K = `n` proved RUs under the run's `query` (canonical JSON), its program put among
    `PROGRAMS`, with the ports of the values a two-stage Spec commits after the draw (`y` and `z`, which no boundary
    has)."""
    s = Setup(P.population_program(PAIR, n), json.loads(query))
    PROGRAMS[s.sha512] = s.program
    part = P.partition(s.program, s.query)
    s.ports = {q: {"domain": R.served_domain(s.sha512, P.digest(part), q, schema, leaves * n, "r-test"), "leaves": leaves * n}
               for q, schema, leaves in (("y", "u16", 2), ("z", "hm96-sha512/row/v1", 1))}
    return s


@functools.cache
def _calls(members: tuple[int, ...]) -> str:
    """The canonical descriptor of `call_program(members)`, its program put among `PROGRAMS`."""
    program = call_program(members)
    PROGRAMS[PO.program_sha512(program)] = program
    return D.canonical(codec.encode_program(program))


@st.composite
def partitions(draw) -> tuple[dict, dict, int]:
    """A random partition P = Q(C): a mixed population program of one to four members, each some instances of one of
    `MEMBERS` (rarely of `FORK` or `IDENTITY`; one template's members apart, adjacent or alone), and a template query over
    it, mostly one that applies (the members' templates, in any order); or, a third of the time, a program of one to five
    root Calls of `CALLEES` (rarely the 32-bit one or `SHARED`) under `Q_word` or `Q_call`, mostly at a width one-stage
    applies. The program's descriptor, the query and the population, 0 where the query doesn't apply."""
    if draw(st.integers(0, 2)) == 0:
        members = tuple(draw(st.lists(st.sampled_from([0, 1, 2, 3] * 6 + [4, 4, 5]), min_size=1, max_size=5)))
        query = draw(st.sampled_from(CALL_QUERIES[:4] * 3 + CALL_QUERIES[4:]) | st.sampled_from(QUERIES))
        desc = json.loads(_calls(members))
        try:
            n = P.units(PROGRAMS[PO.program_sha512(desc)], query)
        except (P.QueryRefused, ValueError):
            n = 0
        return desc, _json(query), n
    members = tuple(draw(st.lists(st.tuples(st.sampled_from([0, 1, 2] * 10 + [3, 4]), st.integers(1, 6)), min_size=1,
                                  max_size=4)))
    ids = [t.id for t in MEMBERS]
    kind = draw(st.integers(0, 9))
    if kind < 8:
        templates = draw(st.permutations(list(dict.fromkeys(MEMBERS[t].id for t, _ in members))))
        one = len(templates) == 1 and draw(st.booleans())
        query = P.template_instance_query(templates[0]) if one else P.template_instances_query(templates)
    elif kind == 8:
        query = P.template_instances_query(draw(st.lists(st.sampled_from(ids), min_size=1, unique=True)))
    else:
        query = draw(st.sampled_from(QUERIES))
    desc = json.loads(_population(members))
    try:
        n = P.units(PROGRAMS[PO.program_sha512(desc)], query)
    except (P.QueryRefused, ValueError):
        n = 0
    return desc, _json(query), n


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


def _rule(j) -> service.Selection:
    """`Select.Rule.ofJson`: a selection rule as `to_json` writes it, exactly its keys, of the kinds a `Stream` runs (a
    two-stage rule at `check="all"`)."""
    kind = j.get("rule") if isinstance(j, dict) else None
    try:
        if kind == "all" and set(j) == {"rule"}:
            return service.All()
        if kind == "law" and set(j) == {"law", "rule"} and isinstance(j["law"], str):
            return service.Law(j["law"])
        if kind == "two-stage" and set(j) == {"check", "replay", "rule"} and isinstance(j["replay"], str):
            rule = service.TwoStage(service.Law(j["replay"]), j["check"])
            if rule.check == "all":
                return rule
    except ValueError as e:
        raise D.Reject(str(e)) from None
    raise D.Reject("not a selection rule the stream runs")


def select(j):
    """The challenger's draw as `Stream.register` and `Stream.select` make it: `service.selection_law` over the
    population of the program under the query, `draw.derive`, `service.select_draw` on the source's bytes read in order
    (the closure map `[[unit, [units]], ...]` as the `{unit: [units]}` it names, each unit once), then `draw.check_draw`;
    with the number of bytes read."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    rule = _rule(j.get("rule"))
    desc, query = _obj(j.get("program"), "program"), _obj(j.get("query"), "query")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    closure = j.get("closure")
    if closure is not None:
        if not (isinstance(closure, list) and all(isinstance(p, list) and len(p) == 2 and _int(p[0]) and isinstance(p[1], list)
                                                  and all(map(_int, p[1])) for p in closure)):
            raise D.Reject("a closure map is a list of [unit, [units]], integers")
        if len({u for u, _ in closure}) != len(closure):
            raise D.Reject("a closure map names a unit once")
        closure = {u: cs for u, cs in closure}
    if not isinstance(j.get("source"), str):
        raise D.Reject("source is not hex")
    src, at = PR.unhex(j["source"]), 0

    def coins(n: int) -> bytes:
        nonlocal at
        at += n
        return src[at - n:at]
    program = PROGRAMS[sha]
    try:
        n = P.units(program, query)
        law = DR.derive(program, query, service.selection_law(rule, n), j.get("work_table"))
        obj = service.select_draw(rule, law, n, coins, closure=closure)
    except (P.QueryRefused, ValueError, service.Refused) as e:
        raise D.Reject(str(e)) from None
    if not DR.check_draw(obj, {"population": n}, law).ok:
        raise D.Reject("the draw is not well formed under the law")
    return {"draw": obj, "read": at}


def slots(j):
    """`Slots.select`: the claims' partition over the select piece's program and query, two classes, and its draw's
    checked proof units. The population's strata are `partition.strata`'s, each unit a replay unit of one proof unit, its
    gates (`partition.unit_gates`); the input class is one stratum of every gate of the root's input nodes
    (`program.is_input`), each its own replay unit. A replay unit's slot is its index in that order, and the checked proof
    units are `(slot, 0)` of each unit the draw proves (`draw.proved`) and of each input gate, in slot order."""
    out = select(j)
    j = _json(j)
    program = PROGRAMS[PO.program_sha512(j["program"])]
    gates = P.unit_gates(program, j["query"])
    pop = [[{"pus": [list(gates[u])], "unit": u} for lo, hi in rs for u in range(lo, hi)]
           for _t, rs in P.strata(program, j["query"])]
    body = program.root.body
    inputs = [{"pus": [[g, g + 1]], "unit": None} for k, node in enumerate(body.nodes) if is_input(node.fn)
              for g in range(body.offsets[k], body.offsets[k] + (node.n if node.form in ("batch", "scan") else 1) * node.fn.gates)]
    classes = [pop, [inputs] if inputs else []]
    proved = set(DR.proved(out["draw"]))
    rus = [r for c in classes for s in c for r in s]
    return {"checked": [[v, 0] for v, r in enumerate(rus) if r["unit"] is None or r["unit"] in proved], "classes": classes}


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


def _closure_map(closure):
    if closure is None:
        return None
    if not (isinstance(closure, list) and all(isinstance(p, list) and len(p) == 2 and _int(p[0]) and isinstance(p[1], list)
                                              and all(map(_int, p[1])) for p in closure)):
        raise D.Reject("a closure map is a list of [unit, [units]], integers")
    if len({u for u, _ in closure}) != len(closure):
        raise D.Reject("a closure map names a unit once")
    return closure


def _values_of(vs) -> list:
    if not (isinstance(vs, list) and all(isinstance(v, dict) and isinstance(v.get("name"), str)
                                         and isinstance(v.get("registered"), bool) for v in vs)):
        raise D.Reject("a value is not a string name with a boolean")
    return vs


def _receipt(rec, challenger: str, received: str) -> dict:
    return {"format": R.RECEIPT, "registration": R.digest(rec), "verifier": challenger, "received": received}


def run(j):
    """`Run.run`: a `Stream`'s run of one registration from its public check through its verdict, the pieces above in its
    order. `register`: `check_public` on the boundary values and declared items, the population, R1-R7 under
    `service.selection_law` over it (a rate law's count of none or more than N: `rate`), a law whose drawn count varies
    (`service.count_varies`: `shapes`), and the
    receipt; `select`: the draw from the source's bytes (`select`); `interior`
    (`register_interiors`): a two-stage rule that drew some RU, `check_public` on the interior values, the K proved RUs'
    population, R1-R7 on the interiors' copies under the run's query and `subset:K`, and the receipt; `deliver`: one or
    more sessions, each matching `service._DIGEST`, and the interiors registered first when a two-stage rule drew some RU;
    `outcome`: `check_verdicts`, then the audit record (`record`) over the draw's proved units unless given, with the
    interiors' over theirs and the sessions' roots split between the two by the interior values' names, their acceptance
    the symbol and the first verdict of record's public outputs shown on an accept. A step that refuses gives `{"step",
    "codes"}`; inputs the Lean decodes before the run (`Pieces.run`) are refused up front."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    p = _params_of(j)
    rule = _rule(j.get("rule"))
    vs, ps = _values_of(j.get("values")), j.get("public")
    ivs = [] if j.get("interior_values") is None else _values_of(j["interior_values"])
    ir = j.get("interior")
    if ir is not None:
        if not (isinstance(ir, dict) and isinstance(ir.get("received"), str)):
            raise D.Reject("the interior's received is not a string")
        ip = _params_of(ir)
    if not (isinstance(ps, list) and all(isinstance(i, str) for i in ps)):
        raise D.Reject("a declared item is not a string")
    if not (isinstance(j.get("challenger"), str) and isinstance(j.get("received"), str)):
        raise D.Reject("challenger or received is not a string")
    closure = _closure_map(j.get("closure"))
    sessions, verdicts, outputs = j.get("sessions"), j.get("verdicts"), j.get("outputs")
    if not (isinstance(sessions, list) and isinstance(verdicts, dict) and all(isinstance(v, dict) for v in verdicts.values())):
        raise D.Reject("sessions is not a list, or verdicts not an object of objects")
    if not (isinstance(outputs, list) and all(isinstance(o, str) for o in outputs) and len(set(outputs)) == len(outputs)):
        raise D.Reject("outputs are not distinct strings")
    proved, instances = j.get("proved_units"), j.get("instances")
    if not (proved is None or isinstance(proved, list) and all(map(_int, proved))) or not (instances is None or _int(instances)):
        raise D.Reject("proved_units is not a list of integers, or instances not an integer")
    if not isinstance(j.get("source"), str):
        raise D.Reject("source is not hex")
    PR.unhex(j["source"])
    rec = j.get("record")
    pub = check_public({"public": ps, "query": j["params"]["query"], "record": rec, "scheme": p["scheme"], "values": vs})
    if pub["codes"]:
        return {"codes": pub["codes"], "step": "register"}
    n = rec.get("population") if isinstance(rec, dict) else None
    if not (_int(n) and n >= 1):
        return {"codes": ["population"], "step": "register"}
    try:
        law = service.selection_law(rule, n)
    except service.Refused as e:
        return {"codes": list(e.verdict.codes), "step": "register"}
    except ValueError as e:
        raise D.Reject(str(e)) from None
    codes = _registration({**p, "law": law}, rec)
    if codes:
        return {"codes": list(codes), "step": "register"}
    if service.count_varies(law):
        return {"codes": ["shapes"], "step": "register"}
    receipt = _receipt(rec, j["challenger"], j["received"])
    try:
        derived = DR.derive(p["program"], p["query"], law, p["work_table"])
    except (P.QueryRefused, ValueError) as e:
        raise D.Reject(str(e)) from None
    drawn = select({"closure": closure, "program": j["params"]["program"], "query": j["params"]["query"], "rule": j["rule"],
                    "source": j["source"], "work_table": j["params"].get("work_table")})
    codes = DR.check_draw(drawn["draw"], rec, derived).codes
    if codes:
        return {"codes": list(codes), "step": "select"}
    size, staged = len(DR.proved(drawn["draw"])), isinstance(rule, service.TwoStage)
    inner = None
    if ir is not None:
        if not staged:
            return {"codes": ["rule"], "step": "interior"}
        if not size:
            return {"codes": ["interior-empty"], "step": "interior"}
        irec = ir.get("record")
        pub = check_public({"public": ps, "query": j["params"]["query"], "record": irec, "scheme": p["scheme"], "values": ivs})
        if pub["codes"]:
            return {"codes": pub["codes"], "step": "interior"}
        if not (isinstance(irec, dict) and _int(irec.get("population")) and irec["population"] == size):
            return {"codes": ["interior-population"], "step": "interior"}
        ilaw = {"k": size, "law": "subset"}
        codes = _registration({**ip, "law": ilaw, "query": p["query"]}, irec)
        if codes:
            return {"codes": list(codes), "step": "interior"}
        inner = (irec, _receipt(irec, j["challenger"], ir["received"]), ilaw,
                 {**ilaw, "population": size, "units": list(range(size))})
    if not sessions or not all(isinstance(s, str) and service._DIGEST.match(s) for s in sessions):
        return {"codes": ["sessions"], "step": "deliver"}
    if staged and size and inner is None:
        return {"codes": ["interior"], "step": "deliver"}
    held = check_verdicts({"outputs": outputs, "sessions": sessions, "verdicts": verdicts})
    if held["codes"]:
        return {"codes": held["codes"], "step": "outcome"}
    of_record = sorted(k for k in verdicts if verdicts[k].get("of_record"))
    shown = dict(verdicts[of_record[0]]["public"]) if outputs else {}
    proved = proved if proved is not None else DR.proved(drawn["draw"])
    instances = len(proved) if instances is None else instances
    roots, inner_roots = j.get("session_roots"), None
    if staged and roots is not None:
        if not isinstance(roots, dict):
            raise D.Reject("session_roots is not an object")
        names = {v["name"] for v in ivs}
        roots, inner_roots = ({q: r for q, r in roots.items() if (q in names) is side} for side in (False, True))
    common = {"delta": j.get("delta"), "instances": instances, "sessions": sessions, "statement": j.get("statement"),
              "verdicts": verdicts}
    out = audit_record_of({**common, "draw": drawn["draw"], "law": derived, "proved_units": proved, "receipt": receipt,
                           "registration": rec, "served": j.get("served"), "session_roots": roots})
    accepted = out["accepted"] is True
    interior = None
    if inner is not None:
        irec, ireceipt, ilaw, idraw = inner
        iout = audit_record_of({**common, "draw": idraw, "law": ilaw, "proved_units": list(range(size)), "receipt": ireceipt,
                                "registration": irec, "served": None, "session_roots": inner_roots})
        accepted = accepted and iout["accepted"] is True
        interior = {"audit": codec.canonical_json(iout).decode(), "receipt": ireceipt}
    return {"audit": codec.canonical_json(out).decode(), "draw": drawn["draw"], "interior": interior,
            "public": shown if accepted else {}, "read": drawn["read"], "receipt": receipt,
            "symbol": "accept" if accepted else "reject"}


def _two_stage_law(j) -> TwoStageLaw:
    """`TwoStage.lawOfJson`: `{class: stage}` as `TwoStageLaw.to_json` writes it, each stage exactly `{"p", "k"}` (`p`
    NUM/DEN or NUM) or `{"law", "k"}` (a one-stage law object), `k` null or a natural number, and refused where `Stage`
    refuses it."""
    if not isinstance(j, dict):
        raise D.Reject("a two-stage law is an object of stages")
    stages = {}
    for cls, s in j.items():
        if not (isinstance(s, dict) and set(s) in ({"k", "p"}, {"k", "law"})):
            raise D.Reject("a stage is {p, k} or {law, k}")
        if not (s["k"] is None or _nat(s["k"])):
            raise D.Reject("a stage needs k >= 0")
        if "p" in s:
            num, sep, den = s["p"].partition("/") if isinstance(s["p"], str) else ("", "", "")
            if not (DR.digits(num) and (not sep or DR.digits(den) and int(den) > 0)):
                raise D.Reject("a stage's p is NUM/DEN")
            first = Fraction(int(num), int(den) if sep else 1)
        elif isinstance(s["law"], dict):
            first = s["law"]
        else:
            raise D.Reject("a stage's law is a one-stage law object")
        try:
            stages[cls] = Stage(first, s["k"])
        except (ValueError, TypeError) as e:
            raise D.Reject(str(e)) from None
    return TwoStageLaw(stages)


def _rus(j) -> list[ReplayUnit]:
    if not (isinstance(j, list) and all(isinstance(u, list) and len(u) == 3 and _int(u[0]) and isinstance(u[1], str)
                                        and _nat(u[2]) for u in j)):
        raise D.Reject("replay units are a list of [index, class, n_v]")
    return [ReplayUnit(*u) for u in j]


def _two_stage_draw(j, units: str, sample):
    """`sample` on the case's law and RUs, the source's bytes read in order (`OwnRandomness` raises `ValueError` when they
    run out), with the number read; it raises `ValueError` and `KeyError` where Lean refuses."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    law, rus = _two_stage_law(j.get("law")), _rus(j.get(units))
    if not isinstance(j.get("source"), str):
        raise D.Reject("source is not hex")
    src, at = PR.unhex(j["source"]), 0

    def coins(n: int) -> bytes:
        nonlocal at
        b = src[at:at + n]
        at += len(b)
        return b
    try:
        out = sample(OwnRandomness(coins), law, rus)
    except (ValueError, KeyError) as e:
        raise D.Reject(str(e)) from None
    return out, at


def replay_units(j):
    """`select_replay_units`: the drawn RUs' indices in the given order, and the bytes read."""
    out, read = _two_stage_draw(j, "units", select_replay_units)
    return {"read": read, "units": [u.index for u in out]}


def proof_units(j):
    """`select_proof_units`: each selected RU's index and the positions checked in it, in order, and the bytes read."""
    out, read = _two_stage_draw(j, "selected", select_proof_units)
    return {"read": read, "units": [[i, list(ps)] for i, ps in out.items()]}


def hello(j):
    """The run's Hello on its record (`hello.check_run`) on `{"own", "hello", "coin_tree", "seen", "selection", "slots",
    "session"}`. Rejected where the Lean refuses: an own Hello that isn't an object with an object `coins`, a Hello that
    isn't a string, `seen`, `selection` or `session` not lists of strings, and `slots` not a natural number."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("the run is not an object")
    own, received, slots = j.get("own"), j.get("hello"), j.get("slots")
    if not (isinstance(own, dict) and isinstance(own.get("coins"), dict)):
        raise D.Reject("the verifier's own Hello is not an object with an object coins")
    if not isinstance(received, str):
        raise D.Reject("the Hello is not a string")
    seen, selection, session = (j.get(k) for k in ("seen", "selection", "session"))
    if not all(isinstance(x, list) and all(isinstance(s, str) for s in x) for x in (seen, selection, session)):
        raise D.Reject("seen, selection or session is not a list of strings")
    if not (type(slots) is int and slots >= 0):
        raise D.Reject("slots is not a natural number")
    return _json(H.check_run(own, received, j.get("coin_tree"), seen, selection, slots, session).to_json())


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
    st.builds(lambda a, pq, law, t, pre: {"params": a["params"] | {
        "program": pq[0], "query": pq[1], "law": law, "work_table": t,
        "partition": {"program": PO.program_sha512(pq[0]), "query": pq[1]} if pre else None}},
        changed_audits(most=0, setups=SETUPS), partitions(), laws(24),
        st.one_of(st.none(), st.fixed_dictionaries({t: WORK_ENTRY for t in WORK_IDS})), st.booleans()),
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
#: Law texts `draw.parse_law` or `draw.parse_rate` refuses or reads oddly: digits of other scripts, signs, spaces, a
#: second separator, leading zeros, a missing or zero argument.
ODD_LAW_TEXTS = ["subset:", "subset", "", "subset:\u0663", "work:\u00b2", "stratified:\uff11", "bernoulli:1/\u0662",
                 "bernoulli:\u00b9/2", "bernoulli:01/04", "bernoulli:0/0", "bernoulli:3/2", "bernoulli:1", "bernoulli:1/2/3",
                 "subset:1:2", "subset:+1", "subset: 1", "subset:-1", "stratified:0", "subset:0", "subset:00", "work:007",
                 "subset:007", "nope:1",
                 "Subset:1", "bernoulli:", "bernoulli:/2", "rate:", "rate", "rate:0/3", "rate:4/3", "rate:1/0",
                 "rate:1/2:0", "rate:1/2:", "rate:1/2:3:4", "rate:1/2/3", "rate:\u0661/2", "rate:1/2:\u0663",
                 "rate:01/02:003", "Rate:1/2", "rate:/2", "rate:1/2: 3", "rate:+1/2", "rate:1:2"]
#: A rate law's text, `rate:NUM/DEN[:T]`, the tables written or not.
RATE_TEXTS = st.builds(lambda num, den, t: f"rate:{num}/{den}" + ("" if t is None else f":{t}"),
                       st.integers(0, 12), st.integers(0, 12), st.none() | st.integers(0, 6))
LAW_TEXTS = st.one_of(
    st.builds(lambda kind, k: f"{kind}:{k}", st.sampled_from(["subset", "stratified", "work"]), st.integers(0, 30)),
    st.builds(lambda num, den: f"bernoulli:{num}/{den}", st.integers(0, 12), st.integers(0, 12)),
    RATE_TEXTS, st.sampled_from(ODD_LAW_TEXTS))
#: Rules a stream doesn't run, or that aren't rules as `to_json` writes them.
ODD_RULES = [{"check": 2, "replay": "subset:1", "rule": "two-stage"}, {"check": "some", "replay": "subset:1", "rule": "two-stage"},
             {"check": True, "replay": "subset:1", "rule": "two-stage"}, {"replay": "subset:1", "rule": "two-stage"},
             {"law": "uniform:3", "population": "n", "rule": "sequential"}, {"rule": "all", "x": 1}, {"law": 5, "rule": "law"},
             {"rule": "law"}, {"law": "subset:1", "rule": "law", "x": 1}, {"law": ["subset:1"], "rule": "law"},
             {"check": "all", "replay": 1, "rule": "two-stage"}, {"rule": "All"}, {"rule": None}, {}]


@st.composite
def closure_maps(draw, n: int) -> list:
    """A closure map over [0, n) as `[[unit, [units]], ...]`, each unit once, sometimes with a unit outside it."""
    units = st.integers(0, max(n - 1, 0))
    pairs = draw(st.lists(st.tuples(units, st.lists(units, min_size=1, max_size=4)), min_size=1, max_size=n + 1,
                          unique_by=lambda p: p[0]))
    out = [[u, cs] for u, cs in pairs]
    if draw(st.integers(0, 7)) == 7:
        out.append([draw(st.sampled_from([n, -1, n + 3])), [0]] if draw(st.booleans()) else [n + 9, [draw(st.sampled_from([n, -1]))]])
    return out


@st.composite
def selections(draw) -> dict:
    """The select piece's input on a random partition: a rule of each kind the stream runs (its k within the population,
    a rate's mostly, its tables sometimes more than its count fits), a two-stage rule's replay law as often as a law,
    the work table naming every template for a work law (sometimes with a template's work 0, or a bad entry), mostly a
    closure map for a work law, and the coins."""
    desc, query, n = draw(partitions())
    kind = draw(st.sampled_from(["work", "stratified", "subset", "bernoulli", "rate", "work", "all"]))
    if kind == "all":
        rule = {"rule": "all"}
    else:
        den = draw(st.integers(1, 12))
        if kind == "bernoulli":
            text = f"bernoulli:{draw(st.integers(0, den))}/{den}"
        elif kind == "rate":
            tables = draw(st.none() | st.integers(1, 4))
            text = f"rate:{draw(st.integers(1, den))}/{den}" + ("" if tables is None else f":{tables}")
        else:
            text = f"{kind}:{draw(st.integers(1, max(n, 1)))}"
        rule = draw(st.sampled_from([{"law": text, "rule": "law"}, {"check": "all", "replay": text, "rule": "two-stage"}]))
    table = None
    if kind == "work":
        table = {t: draw(st.integers(1, 4) | WORK_ENTRY) for t in WORK_IDS}
        if draw(st.integers(0, 9)) == 9:
            table[draw(st.sampled_from(WORK_IDS))] = draw(BAD_WORK_ENTRY)
    closure = draw(closure_maps(n)) if kind == "work" and draw(st.integers(0, 3)) < 3 else None
    return {"closure": closure, "program": desc, "query": query, "rule": rule, "source": draw(SOURCE), "work_table": table}


def _with(field: str, values):
    return lambda c, v: {**c, field: v}


#: One change each to the select piece's input: the rule (a law text of any form, a rule the stream doesn't run, or
#: junk), the work table (none, empty, junk), the closure map (any law's, malformed, a unit twice), the query, the
#: program, the coins (short, junk) or a field missing.
SELECT_CHANGES = [
    (_with("rule", None), st.one_of(LAW_TEXTS.map(lambda t: {"law": t, "rule": "law"}),
                                    LAW_TEXTS.map(lambda t: {"check": "all", "replay": t, "rule": "two-stage"}))),
    (_with("rule", None), st.sampled_from(ODD_RULES) | JSON),
    (_with("work_table", None), st.sampled_from([None, {}, 5, [], {PAIR.id: 1}]) | JSON),
    (_with("closure", None), st.sampled_from([[], [[0, []]], [[0, [1]], [0, [2]]], [[0, 1]], [[0]], [["0", [1]]], [[0, [True]]],
                                              [[True, [0]]], {"0": [1]}, "c", 5])),
    (_with("closure", None), st.integers(0, 6).flatmap(closure_maps)),
    (_with("query", None), st.sampled_from(QUERIES) | JSON),
    (_with("program", None), st.sampled_from([{}, 5, "program", [], SETUPS[0].desc])),
    (_with("source", None), SHORT_SOURCE | st.sampled_from(["", "0", "zz", "AB", "ab cd", 5, None])),
    (lambda c, k: {kk: v for kk, v in c.items() if kk != k},
     st.sampled_from(["closure", "program", "query", "rule", "source", "work_table"])),
]


@st.composite
def changed_selections(draw, most: int = 2) -> dict:
    c = draw(selections())
    for _ in range(draw(st.integers(1, most))):
        change, values = draw(st.sampled_from(SELECT_CHANGES))
        c = change(c, draw(values))
    return c


SELECT = D.Port(PIECES, "select", select, st.one_of(selections(), selections(), changed_selections(1),
                                                     changed_selections(2), JSON), package=PACKAGE)
SLOTS = D.Port(PIECES, "slots", slots, st.one_of(selections(), selections(), changed_selections(1), JSON), package=PACKAGE)
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

RUN_SESSIONS = ["5e" * 64, "a1" * 64]


def _law_text(law) -> str | None:
    if not isinstance(law, dict):
        return None
    return f"bernoulli:{law.get('p')}" if law.get("law") == "bernoulli" else f"{law.get('law')}:{law.get('k')}"


def _run_case(a: dict, source: str, sessions: int, outputs: list, two_stage: bool) -> dict:
    """`run`'s input from an audit: the rule naming the verifier's law (or its replay law, two-stage), every item
    `_public_items` declares, the challenger's name and place, the coins, and the evidence's statement and verdicts, each
    verdict of record about the delivered sessions and reporting the public outputs."""
    p, e = a["params"], a["evidence"]
    text = _law_text(p.get("law"))
    rule = ({"rule": "all"} if text is None else
            {"check": "all", "replay": text, "rule": "two-stage"} if two_stage else {"law": text, "rule": "law"})
    try:
        pub = _public_items(a, PUBLIC_ITEMS[0], 0)
    except (AttributeError, KeyError, TypeError):
        pub = {"public": [], "values": []}
    delivered = RUN_SESSIONS[:sessions]
    verdicts = e.get("verdicts")
    if isinstance(verdicts, dict) and all(isinstance(v, dict) for v in verdicts.values()):
        verdicts = {k: {**v, "public": {o: 12 for o in outputs}, "sessions": delivered} if v.get("of_record") else v
                    for k, v in verdicts.items()}
    return {"challenger": "verifier-test", "closure": None, "delta": "1/4", "instances": None, "outputs": outputs,
            "params": p, "proved_units": None, "public": pub["public"], "received": RECEIVED, "record": a["record"],
            "rule": rule, "served": None, "session_roots": e.get("session_roots"), "sessions": delivered,
            "source": source, "statement": e.get("statement"), "values": pub["values"], "verdicts": verdicts}


INTERIOR_ROOTS = {"y": "c1" * 64, "z": "c2" * 64}


def _with_interior(c: dict) -> dict:
    """A two-stage run's case with the interiors `Stream.register_interiors` takes for the draw its coins make: the K proved
    RUs' record under `subset:K` after the boundary's (its `previous`, and a window of its own), on copies of their own,
    their values `y` and `z` declared and the sessions' roots holding theirs. A case whose draw the reference refuses, or
    that draws no RU, is unchanged."""
    if not isinstance(c["record"], dict):
        return c
    try:
        drawn = select({"closure": c["closure"], "program": c["params"]["program"], "query": c["params"]["query"],
                        "rule": c["rule"], "source": c["source"], "work_table": c["params"].get("work_table")})
    except D.Reject:
        return c
    n = len(DR.proved(drawn["draw"]))
    if not n:
        return c
    p = c["params"]
    try:
        s = _interior_setup(n, D.canonical(p["query"]))
    except (P.QueryRefused, ValueError):
        return c
    law = {"k": n, "law": "subset"}
    if not (isinstance(p.get("log"), list) and isinstance(p.get("anchors"), list)):
        return c
    try:
        rec = R.record(s.program, s.query, law, SCHEME, [{"port": q, "root": INTERIOR_ROOTS[q], **s.ports[q]} for q in s.ports],
                       {**WINDOW, "stage": "interior"}, leaf_layer=p.get("leaf_layer"), anchors=p["anchors"],
                       previous=R.digest(c["record"]))
    except (TypeError, ValueError):
        return c
    params = {**{k: v for k, v in p.items() if k != "parameters"}, "law": law, "log": [*p["log"], c["record"]],
              "partition": None, "ports": s.ports, "program": s.desc, "registered": None, "work_table": None}
    roots = c["session_roots"]
    return _json({**c, "interior": {"params": params, "received": "1", "record": rec},
                  "interior_values": [{"name": q, "registered": False} for q in s.ports],
                  "public": [*c["public"], *(f"{k}:{q}" for q in s.ports for k in ("leaves", "domain"))],
                  "session_roots": {**roots, **INTERIOR_ROOTS} if isinstance(roots, dict) else roots})


def _interior_change(f):
    """A change to the case's interiors, where it has them as `_with_interior` makes them."""
    def apply(c, x):
        ir = c.get("interior")
        if not (isinstance(ir, dict) and isinstance(ir.get("record"), dict) and isinstance(ir.get("params"), dict)):
            return c
        return {**c, "interior": f(ir, x)}
    return apply


def _interior_record(f):
    return _interior_change(lambda ir, x: {**ir, "record": f(ir["record"], x)})


def _interior_leaves(ir, x):
    ports = ir["params"]["ports"]
    return {**ir, "params": {**ir["params"], "ports": {**ports, "y": {**ports["y"], "leaves": ports["y"]["leaves"] + 1}}}}


def _public_change(change):
    """One of `PUBLIC_ITEMS`' changes to the run's values and declared items."""
    def apply(c, x):
        if not (c.get("public") and c.get("values")):
            return c
        changed = change({"public": c["public"], "values": c["values"]}, x)
        return {**c, "public": changed["public"], "values": changed["values"]}
    return apply


def _verdict_change(f):
    """A change to the `x`-th verdict of the run, as `_one_verdict` makes one."""
    def apply(c, x):
        if not (isinstance(c["verdicts"], dict) and c["verdicts"]):
            return c
        return _one_verdict(f)(c, x)
    return apply


#: One change each to a two-stage run's interiors: none or junk, the rule made its replay law's, their population, R6's
#: previous or window, a root dropped, a record that is not one, a port's leaves, the receipt's place, the interior
#: values, or the sessions' roots.
INTERIOR_CHANGES = [
    (_with("interior", None), st.sampled_from([None, 5, [], {}, {"received": "1"}, {"params": None, "received": "1"}])),
    (lambda c, x: {**c, "rule": {"law": c["rule"]["replay"], "rule": "law"}} if c["rule"].get("rule") == "two-stage" else c,
     st.just(0)),
    (_interior_record(lambda r, x: {**r, "population": [r["population"] + 1, True, 0, "1", None][x % 5]}),
     st.integers(0, 63)),
    (_interior_record(lambda r, x: {**r, ["previous", "window"][x % 2]: ["11" * 64, WINDOW][x % 2]}), st.integers(0, 63)),
    (_interior_record(lambda r, x: {**r, "roots": r["roots"][1:]}), st.integers(0, 63)),
    (_interior_record(lambda r, x: [None, "a record", 5][x % 3]), st.integers(0, 63)),
    (_interior_change(_interior_leaves), st.integers(0, 63)),
    (_interior_change(lambda ir, x: {**ir, "received": [5, None, ""][x % 3]}), st.integers(0, 63)),
    (_with("interior_values", None), st.sampled_from([None, [], [{"name": "y", "registered": False}],
                                                      [{"name": "x", "registered": False}, {"name": "y", "registered": False},
                                                       {"name": "z", "registered": False}], [5], "v"])),
    (_with("session_roots", None), st.sampled_from([dict(ROOTS), {**ROOTS, **INTERIOR_ROOTS, "y": "00" * 64},
                                                    {**ROOTS, "z": INTERIOR_ROOTS["z"]}, None])),
]

def _as_rate(c: dict, x: int) -> dict:
    """The run's `subset:K` rule (a two-stage rule's replay) as a rate law over the record's N: `rate:K/N` over one table
    or over a divisor of K, which is `subset:K` again, over tables K doesn't divide, which rounds K up past the record's
    law (R1) or past N, or over N + 1 tables, past N (`rate`). Any other rule is unchanged."""
    rule, rec = c.get("rule"), c.get("record")
    if not isinstance(rule, dict):
        return c
    field = "replay" if rule.get("rule") == "two-stage" else "law"
    m = re.fullmatch(r"subset:([0-9]+)", str(rule.get(field)))
    n = rec.get("population") if isinstance(rec, dict) else None
    if m is None or not (type(n) is int and 1 <= int(m[1]) <= n):
        return c
    k = int(m[1])
    divisors = [t for t in range(1, k + 1) if k % t == 0]
    tables = [None, 1, divisors[x % len(divisors)], n + 1, 2 + x % 5][x % 5]
    return {**c, "rule": {**rule, field: f"rate:{k}/{n}" + ("" if tables is None else f":{tables}")}}


#: One change each to the run's own inputs: the rule (a law text of any form, its subset law as a rate law, a rule the
#: stream doesn't run, or junk), the declared items and values, the challenger's name and place, the closure map, the
#: coins, the sessions, a verdict, the outputs, the statement, δ, the proved units and instances, the served units, the
#: session roots, or a field missing.
RUN_CHANGES = [
    (_with("rule", None), st.one_of(LAW_TEXTS.map(lambda t: {"law": t, "rule": "law"}), st.just({"rule": "all"}),
                                    st.sampled_from(ODD_RULES) | JSON)),
    (_as_rate, st.integers(0, 63)),
    *((lambda c, x, f=f: _public_change(f)(c, x), st.integers(0, 63)) for f in PUBLIC_ITEMS[1:7]),
    (_with("challenger", None), st.sampled_from([5, None, "other-verifier", ""])),
    (_with("received", None), st.sampled_from([5, None, "", RECEIVED + "Z"])),
    (_with("closure", None), st.sampled_from([[], [[0, []]], [[0, [1]], [0, [2]]], [[0, 1]], [["0", [1]]], {"0": [1]}, 5])),
    (_with("closure", None), st.integers(0, 6).flatmap(closure_maps)),
    (_with("source", None), SHORT_SOURCE | st.sampled_from(["", "0", "zz", "AB", 5, None])),
    (_with("sessions", None), st.sampled_from([[], ["zz"], ["AB" * 64], [RUN_SESSIONS[0] + "\n"], [5], "s", RUN_SESSIONS[::-1],
                                               [RUN_SESSIONS[0]] * 2, {"s": 1}])),
    (_verdict_change(lambda v, c, x: {**v, "accepted": [False, None, 1][x % 3]}), st.integers(0, 63)),
    (_verdict_change(lambda v, c, x: {**v, "of_record": [False, None, 1][x % 3]}), st.integers(0, 63)),
    (_verdict_change(lambda v, c, x: {**v, "sessions": [[], None, "s", RUN_SESSIONS[1:]][x % 4]}), st.integers(0, 63)),
    (_verdict_change(lambda v, c, x: {**v, "public": [{}, None, {"credited": 13}, {"credited": 12, "z": 1}][x % 4]}),
     st.integers(0, 63)),
    (_with("verdicts", None), st.sampled_from([{}, [], 5, {"lean": 5}])),
    (_with("outputs", None), st.sampled_from([["credited"], ["credited", "credited"], [5], "o", ["z"]])),
    (_with("statement", None), st.sampled_from([None, 5, [], {}, {"instances": 1}])),
    (_with("delta", None), st.sampled_from(["0", "1", "1/0", "a", 5, None, "1/2", "999/1000"])),
    (_with("proved_units", None), st.sampled_from([[], [0], [0, 1, 2], "u", [True], [0, 0]])),
    (_with("instances", None), st.sampled_from([0, 1, 5, "1", True, -1])),
    (_with("served", None), st.sampled_from([[], [0], [0, 1], "s", [True]])),
    (_with("session_roots", None), st.sampled_from([{}, {"x": "00" * 64}, dict(ROOTS), {**ROOTS, **INTERIOR_ROOTS},
                                                    {**ROOTS, **INTERIOR_ROOTS, "y": "00" * 64}, 5, None])),
    *INTERIOR_CHANGES,
    (lambda c, k: {kk: v for kk, v in c.items() if kk != k},
     st.sampled_from(["challenger", "closure", "delta", "instances", "interior", "interior_values", "outputs", "params",
                      "proved_units", "public", "received", "record", "rule", "served", "session_roots", "sessions",
                      "source", "statement", "values", "verdicts"])),
]


@st.composite
def runs(draw, groups=(), most: int = 0, changes: int = 0, staged: bool = False) -> dict:
    """A run's input from an audit with up to `most` changes from `groups`, then `changes` of the run's own; `staged`, a
    two-stage run with its interiors and `changes` of theirs."""
    a = draw(changed_audits(list(groups), most=most, least=min(most, 1))) if groups else draw(changed_audits(most=0))
    c = _run_case(a, draw(SOURCE), draw(st.integers(1, 2)), draw(st.sampled_from([[], [], ["credited"]])),
                  staged or draw(st.integers(0, 3)) == 0)
    if c["rule"].get("rule") == "two-stage" and (staged or draw(st.integers(0, 3))):
        c = _with_interior(c)
    for _ in range(changes):
        change, values = draw(st.sampled_from(INTERIOR_CHANGES if staged else RUN_CHANGES))
        c = change(c, draw(values))
    return c


RUN = D.Port(PIECES, "run", run, st.one_of(
    runs(), runs(), runs(changes=1), runs(changes=1), runs(changes=2), runs(staged=True, changes=1),
    runs(staged=True, changes=1), st.builds(_as_rate, runs(), st.integers(0, 63)),
    runs([RECORD], 1), runs([PARAMS], 1), runs([ENTRIES, REGISTERED_PARAMS], 1), runs([EVIDENCE, VERDICTS_OF], 1),
    JSON), package=PACKAGE)

RU_CLASSES = ["decode", "prefill", "tile"]


@functools.cache
def _expanded(kind: str, a: int, b: int, k: int) -> dict | None:
    """A `stratified` or `work` law of `k` expanded over `a` PAIRs then `b` TRIPLEs (`draw.derive`), or None where it
    refuses."""
    members = [(PAIR, a), (TRIPLE, b)] if b else [(PAIR, a)]
    query = P.template_instances_query([t.id for t, _ in members])
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}} if b else {PAIR.id: 2}
    try:
        return DR.derive(P.mixed_population_program(members), query, {"k": k, "law": kind}, table if kind == "work" else None)
    except (P.QueryRefused, ValueError):
        return None


@st.composite
def two_stage_stages(draw, n: int, bad: bool):
    """A class of `n` RUs' stage: a probability, or a one-stage law over them (a subset, a Bernoulli law, or a stratified
    or work law expanded over two templates), mostly checking every proof unit; with `bad`, one the stage refuses (`k`,
    `p` or the law) or junk."""
    kind = draw(st.integers(0, 6))
    k = draw(st.sampled_from([-1, True, "1"]) if bad and kind % 3 == 0 else st.sampled_from([None, None, 0, 1, 2, 3, 7]))
    if kind < 3:
        den = draw(st.integers(1, 9))
        p = draw(st.sampled_from(["5/4", "1/0", "a", "1/2/3", "01/2", "", "/2", "1/", 5]) if bad and kind == 1 else
                 st.sampled_from([f"{draw(st.integers(0, den))}/{den}", "0", "1", f"{2 * den}/{4 * den}"]))
        return {"k": k, "p": p}
    if kind < 6:
        a = draw(st.integers(1, max(n, 1)))
        law = draw(st.sampled_from([{"k": draw(st.integers(0, n + 1 if bad else n)), "law": "subset"},
                                    {"law": "bernoulli", "p": f"{draw(st.integers(0, 4))}/4"},
                                    _expanded("stratified", a, max(n - a, 0), draw(st.integers(1, max(n, 1)))),
                                    _expanded("work", a, max(n - a, 0), draw(st.integers(1, max(n, 1))))]))
        law = law or {"k": min(1, n), "law": "subset"}
        return {"k": k if law["law"] == "bernoulli" or bad and kind == 3 else None, "law": law}
    return draw(st.just({"k": None, "law": {"k": 1, "law": "subset", "x": 1}}) | JSON) if bad else {"k": None, "p": "1/2"}


@st.composite
def two_stage_cases(draw, units: str) -> dict:
    """RUs of up to three classes in a shuffled order, indices distinct but sometimes repeated, with each class's stage
    and the coins; sometimes one change: a stage refused, a class without one, a law or RUs that aren't, or short coins."""
    n = draw(st.integers(0, 12))
    order = draw(st.permutations(range(n)))
    change = draw(st.integers(0, 15))
    rus = [[draw(st.integers(-2, 40)) if change == 0 else 3 * i - 5, draw(st.sampled_from(RU_CLASSES)),
            draw(st.integers(0, 6))] for i in order]
    sizes = {c: sum(u[1] == c for u in rus) for c in RU_CLASSES}
    bad = draw(st.sampled_from(RU_CLASSES)) if change == 1 else None
    law = {c: draw(two_stage_stages(sizes[c], c == bad)) for c in RU_CLASSES if sizes[c] or draw(st.booleans())}
    if rus and change == 2:
        law.pop(rus[0][1], None)
    if change == 3:
        law = draw(JSON)
    if change == 4:
        rus = draw(st.sampled_from([[[0, "decode", -1]], [[0, 5, 1]], [[True, "decode", 1]], [[0, "decode"]], "u", None]))
    return {"law": law, "source": draw(SHORT_SOURCE if change == 5 else SOURCE), units: rus}


REPLAY_UNITS = D.Port(PIECES, "replay_units", replay_units, st.one_of(*[two_stage_cases("units")] * 5, JSON),
                      package=PACKAGE)
PROOF_UNITS = D.Port(PIECES, "proof_units", proof_units, st.one_of(*[two_stage_cases("selected")] * 5, JSON),
                     package=PACKAGE)


def _key(tag: bytes) -> str:
    """A 256-byte key with bit 2047 zero."""
    h = hashlib.shake_256(tag).hexdigest(256)
    return h[:510] + "0" + h[511]


RUN_NONCE = hashlib.sha256(b"run nonce").hexdigest()
RUN_KEY = _key(b"run key")
RUN_ROOT = hashlib.sha512(b"run root").hexdigest()


def run_hello(selection=("selection",), slots=2, session=("session", "t/rep0", "t/rep1"), streams=None) -> dict:
    """A run's Hello on its record: the verifier's own Hello, the prover's (the nonce put in), the record's tree."""
    spec = {"block": 3, "rounds_log": 2, "scheme": H.SCHEME,
            "streams": H.run_streams(selection, slots, session) if streams is None else list(streams)}
    own = {"digests": "sha512", "profile": "fast100", "slots": slots, "coins": {**spec, "nonce": ""}}
    return {"own": own, "hello": D.canonical({**own, "coins": {**spec, "nonce": RUN_NONCE}}),
            "coin_tree": {"key": RUN_KEY, "nonce": RUN_NONCE, "root": RUN_ROOT, "spec": spec}, "seen": [],
            "selection": list(selection), "slots": slots, "session": list(session)}


def _hello_as(c: dict, own: dict) -> dict:
    """`c` with the verifier's own Hello `own`, and the prover's Hello its with the nonce."""
    return {**c, "own": own, "hello": D.canonical({**own, "coins": {**own["coins"], "nonce": RUN_NONCE}})}


def _tree(c: dict, **change) -> dict:
    return {**c, "coin_tree": {**c["coin_tree"], **change}}


#: One change each to a run's Hello on its record: the tree reused, the Hello tampered, the streams not the run's, the
#: record's tree not the Hello's, and inputs the Lean refuses.
HELLO_CHANGES = [
    lambda c, x: c,
    lambda c, x: {**c, "seen": [_key(b"earlier"), RUN_KEY][x % 2:]},
    lambda c, x: {**c, "seen": [_key(b"earlier %d" % i) for i in range(x % 3)]},
    lambda c, x: {**c, "hello": c["hello"].replace(RUN_NONCE, [RUN_NONCE.upper(), RUN_NONCE[:-2], "1" * 64][x % 3])},
    lambda c, x: {**c, "hello": [json.dumps(json.loads(c["hello"])), c["hello"] + " ", " " + c["hello"],
                                 c["hello"].replace("fast100", "fast101"), "x", "", "[]", "null"][x % 8]},
    lambda c, x: _hello_as(c, {**c["own"], "coins": {**c["own"]["coins"], "scheme": ["coin-tree/v1", 1, None][x % 3]}}),
    lambda c, x: _hello_as(c, {**c["own"], "coins": {k: v for k, v in c["own"]["coins"].items()
                                                     if k != ["streams", "rounds_log", "block"][x % 3]}}),
    lambda c, x: _hello_as(c, {**c["own"], "coins": {**c["own"]["coins"], "rounds_log": [-1, True, "2"][x % 3]}}),
    lambda c, x: {**c, "selection": [[], ["other"], c["selection"] * 2][x % 3]},
    lambda c, x: {**c, "slots": c["slots"] + 1 + x % 2},
    lambda c, x: {**c, "session": c["session"][::-1] if len(set(c["session"])) > 1 else c["session"] + ["x"]},
    lambda c, x: _tree(c, nonce=["11" * 32, None, RUN_NONCE.upper()][x % 3]),
    lambda c, x: _tree(c, key=[RUN_KEY[:-2] + "a5", RUN_KEY.upper(), RUN_KEY[:-2], 7][x % 4]),
    lambda c, x: _tree(c, root=[RUN_ROOT[:-2], RUN_ROOT.upper(), None][x % 3]),
    lambda c, x: _tree(c, spec=[{**c["coin_tree"]["spec"], "block": 4}, None,
                                {**c["coin_tree"]["spec"], "streams": c["coin_tree"]["spec"]["streams"][::-1] + ["z"]}][x % 3]),
    lambda c, x: _tree(c, answer="00"),
    lambda c, x: {**c, "coin_tree": [None, [], "tree"][x % 3]},
    lambda c, x: {**c, ["own", "hello", "seen", "selection", "session", "slots"][x % 6]:
                  [None, {"coins": []}, 5, ["a", 1], "s", -1][x % 6]},
]
STREAM_NAMES = st.lists(st.sampled_from(["session", "t/rep0", "t/rep1", "nl/rep0", "lin"]), unique=True, max_size=4)
RUN_HELLOS = st.builds(lambda sel, slots, ses, change, x: HELLO_CHANGES[change](run_hello(sel, slots, ses), x),
                       st.sampled_from([("selection",), ("selection", "selection/1"), ("draw",)]), st.integers(0, 3),
                       STREAM_NAMES, st.integers(0, len(HELLO_CHANGES) - 1), st.integers(0, 63))
HELLO = D.Port(PIECES, "hello", hello, st.one_of(RUN_HELLOS, RUN_HELLOS, JSON), package=PACKAGE)

pieces = D.server_fixture(DECODE)


PIECE_RUNS = [(DECODE, 300), (PARTITION, 500), (REGISTRATION, 700), (REGISTERED_ENTRIES, 300), (DOMAIN, 300),
              (CHECK_DRAW, 500), (DRAW, 500), (SELECT, 600), (SLOTS, 400), (CHECK_PROVED, 300), (BOUND, 500), (VERDICT, 300),
              (MATCHES_PUBLIC, 400), (SESSION_REGISTERED, 400), (RECORD_PORT, 600), (CHECK_PUBLIC, 500), (CHECK_VERDICTS, 400),
              (RUN, 500), (REPLAY_UNITS, 500), (PROOF_UNITS, 400), (HELLO, 500)]


@pytest.mark.parametrize("port, examples", PIECE_RUNS, ids=[p.fn for p, _ in PIECE_RUNS])
def test_a_piece_agrees(pieces, port, examples):
    stats = D.run(port, examples=examples, server=pieces)
    assert stats.outputs > examples // 4 and stats.rejects > 0, stats


@pytest.mark.parametrize("name, change, code", [
    ("a good run", lambda c: c, None),
    ("a reused tree", lambda c: {**c, "seen": [_key(b"earlier"), RUN_KEY]}, "fresh"),
    ("a nonce not lowercase hex", lambda c: {**c, "hello": c["hello"].replace(RUN_NONCE, RUN_NONCE.upper())}, "hello"),
    ("a Hello whose nonce the tree is not under", lambda c: {**c, "hello": c["hello"].replace(RUN_NONCE, "1" * 64)},
     "coin_tree"),
    ("a tampered parameter", lambda c: {**c, "hello": c["hello"].replace("fast100", "fast101")}, "hello"),
    ("the selection not first", lambda c: {**c, "selection": ["t/rep0"]}, "streams"),
    ("a slot short", lambda c: {**c, "slots": 3}, "streams"),
    ("the slots out of order", lambda c: _hello_as(c, {**c["own"], "coins": {**c["own"]["coins"], "streams": [
        "selection", *(H.slot_stream(i, s) for i in (1, 0) for s in ("session", "t/rep0", "t/rep1"))]}}), "streams"),
    ("the record's tree not the Hello's", lambda c: _tree(c, nonce="11" * 32), "coin_tree"),
    ("a key with bit 2047 set", lambda c: _tree(c, key=RUN_KEY[:-2] + "a5"), "coin_tree"),
    ("a tampered spec", lambda c: _tree(c, spec={**c["coin_tree"]["spec"], "streams": ["selection"]}), "coin_tree"),
])
def test_the_runs_hello_and_its_one_tree(pieces, name, change, code):
    """The run's Hello on its record, Lean's verdict the Python's whole: a good run, then one named change each, refused
    with its code."""
    case = change(run_hello())
    out = hello(copy.deepcopy(case))
    assert D.replay(HELLO, case, pieces) == ("output", D.canonical(out)), name
    assert out["codes"] == ([] if code is None else [code]) and out["ok"] == (code is None), (name, out)


def test_the_two_stage_samplers_read_the_coins_in_order(pieces):
    """Both stages on one stream of coins, Lean's the Python's whole: RUs in a shuffled order, a class drawn at a rate RU by
    RU (and one at p = 1, which reads nothing) around two classes drawn together under a stratified and a work law when
    their first RU comes up, then the drawn RUs' proof units on the bytes after them, a `k`-subset of the larger RUs and
    every position of the rest."""
    law = {"decode": {"k": 2, "p": "1/2"}, "prefill": {"k": None, "law": _expanded("stratified", 3, 2, 2)},
           "tile": {"k": None, "law": _expanded("work", 2, 2, 2)}, "cap": {"k": 1, "p": "1"}}
    rus = [[11, "decode", 5], [7, "prefill", 3], [3, "tile", 4], [1, "decode", 1], [9, "prefill", 2], [0, "cap", 3],
           [4, "tile", 2], [2, "prefill", 1], [8, "decode", 3], [6, "tile", 6], [5, "prefill", 4], [10, "prefill", 2],
           [12, "tile", 1], [13, "decode", 4]]
    source = hashlib.shake_256(b"two-stage").hexdigest(800)
    case = {"law": law, "source": source, "units": rus}
    first = replay_units(copy.deepcopy(case))
    assert D.replay(REPLAY_UNITS, case, pieces) == ("output", D.canonical(first))
    drawn = [u for u in rus if u[0] in first["units"]]
    assert [u[0] for u in drawn] == first["units"] and {"cap", "prefill", "tile"} <= {u[1] for u in drawn}
    second = {"law": law, "selected": drawn, "source": source[2 * first["read"]:]}
    picks = proof_units(copy.deepcopy(second))
    assert D.replay(PROOF_UNITS, second, pieces) == ("output", D.canonical(picks))
    assert picks["read"] > 0 and all(len(ps) == (min(2 if c == "decode" else 1, n) if c in ("decode", "cap") else n)
                                     for (i, ps), (_, c, n) in zip(picks["units"], drawn))


def test_the_select_piece_draws_every_rule(pieces):
    """One draw under each rule the stream runs, on a partition whose `PAIR` stratum is two ranges apart, Lean's the
    Python's whole: `All` every unit, reading no coins, and each law's draw (a work law's with its closure) reading them,
    each stratum's count its k, which the drawn runs reach only probably, and a rate law's a subset's of its count over
    the 10 units. Then the same with every non-ASCII digit `str.isdigit` admits in place of an ASCII one, and a rate
    whose count passes the population, which both refuse."""
    desc = json.loads(_population(((0, 2), (1, 3), (0, 4), (2, 1))))
    query = _json(P.template_instances_query([t.id for t in TEMPLATES]))
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 2, "work": 1}, QUAD.id: 0}
    closure = [[u, [(u + 1) % 10, (u + 5) % 10]] for u in range(10)]
    source = hashlib.shake_256(b"select").hexdigest(1500)
    rules = [({"rule": "all"}, None), ({"law": "subset:4", "rule": "law"}, None), ({"law": "bernoulli:1/3", "rule": "law"}, None),
             ({"law": "stratified:5", "rule": "law"}, None), ({"law": "work:4", "rule": "law"}, closure),
             ({"check": "all", "replay": "stratified:5", "rule": "two-stage"}, None),
             ({"check": "all", "replay": "work:4", "rule": "two-stage"}, closure),
             ({"law": "rate:1/3:2", "rule": "law"}, None), ({"check": "all", "replay": "rate:1/4:3", "rule": "two-stage"}, None)]
    rates = {"rate:1/3:2": 4, "rate:1/4:3": 3}                       # 2·⌈10/6⌉ and 3·⌈10/12⌉
    for rule, cm in rules:
        case = {"closure": cm, "program": desc, "query": query, "rule": rule, "source": source,
                "work_table": table if "work" in str(rule) else None}
        out = select(copy.deepcopy(case))
        assert D.replay(SELECT, case, pieces) == ("output", D.canonical(out)), rule
        assert (out["read"] == 0) == (rule["rule"] == "all") and bool(cm) == ("closure" in out["draw"]), (rule, out)
        for s in out["draw"].get("strata", []):
            assert sum(any(lo <= u < hi for lo, hi in s["units"]) for u in out["draw"]["units"]) == s["k"] >= 1
        if (k := rates.get(rule.get("law", rule.get("replay")))) is not None:
            assert (out["draw"]["law"], out["draw"]["k"], len(out["draw"]["units"])) == ("subset", k, k), (rule, out)
    for text in ("subset:\u0663", "stratified:\u0665", "work:\u0664", "bernoulli:\u0661/3", "bernoulli:1/\u0663",
                 "rate:\u0661/3", "rate:1/3:\u0662", "rate:9/10:4"):
        case = {"closure": None, "program": desc, "query": query, "rule": {"law": text, "rule": "law"}, "source": source,
                "work_table": table}
        assert D.replay(SELECT, case, pieces)[0] == "reject", text
        with pytest.raises(D.Reject):
            select(case)


def test_the_slots_follow_the_strata(pieces):
    """On the partition whose `PAIR` stratum is two ranges apart, Lean's slots are the Python's whole: the population's
    slots run through `PAIR`'s units, then `TRIPLE`'s, then `QUAD`'s, so units 2 to 8 move; the replay units' gates and
    the input class's are every gate of the program once; and the checked proof units are the drawn units' slots and
    every input slot, in slot order."""
    desc = json.loads(_population(((0, 2), (1, 3), (0, 4), (2, 1))))
    query = _json(P.template_instances_query([t.id for t in TEMPLATES]))
    case = {"closure": None, "program": desc, "query": query, "rule": {"law": "subset:4", "rule": "law"},
            "source": hashlib.shake_256(b"slots").hexdigest(400), "work_table": None}
    out = slots(copy.deepcopy(case))
    assert D.replay(SLOTS, case, pieces) == ("output", D.canonical(out))
    assert [[r["unit"] for r in s] for s in out["classes"][0]] == [[0, 1, 5, 6, 7, 8], [2, 3, 4], [9]]
    rus = [r for c in out["classes"] for s in c for r in s]
    gates = sorted(g for r in rus for lo, hi in r["pus"] for g in range(lo, hi))
    assert gates == list(range(PROGRAMS[PO.program_sha512(desc)].circuit.size))
    units = [r["unit"] for r in rus]
    drawn = [units.index(u) for u in select(case)["draw"]["units"]]
    assert out["checked"] == [[v, 0] for v in sorted(drawn + [v for v, u in enumerate(units) if u is None])]


def test_the_call_queries_cut_call_by_call(pieces):
    """On a program of root Calls, Lean's slots are the Python's whole under `Q_word` at `X = 16` and `Q_call` at `X = 8`:
    the strata are the Calls' Definitions, `FORK`'s `Q_word` units two output units and its committed interior value, the
    first's gate span holding the second's; and both refuse the widths one-stage doesn't apply, and a Call whose unit
    commits 32 bits, which `Q_word`'s `W = 32` admits."""
    desc = json.loads(_calls((0, 2, 0, 3)))
    case = {"closure": None, "program": desc, "rule": {"law": "subset:2", "rule": "law"},
            "source": hashlib.shake_256(b"calls").hexdigest(400), "work_table": None}
    word = slots({**case, "query": _json(PO.query())})
    assert D.replay(SLOTS, {**case, "query": _json(PO.query())}, pieces) == ("output", D.canonical(word))
    assert [[r["unit"] for r in s] for s in word["classes"][0]] == [[0, 4], [1, 2, 3], [5]]
    (first,), (second,) = word["classes"][0][1][0]["pus"], word["classes"][0][1][1]["pus"]
    assert first[0] < second[0] < second[1] < first[1]
    call = slots({**case, "query": _json(PO.call_query(8))})
    assert D.replay(SLOTS, {**case, "query": _json(PO.call_query(8))}, pieces) == ("output", D.canonical(call))
    assert [[r["unit"] for r in s] for s in call["classes"][0]] == [[0, 5], [1, 2, 3, 4], [6]]
    wide = json.loads(_calls((0, 4)))
    for program, query in [(desc, PO.call_query(4)), (desc, PO.call_query(32)), (desc, PO.query(X=8)),
                           (desc, PO.query(version=2)), (wide, PO.query()), (wide, PO.call_query(16))]:
        refused = {**case, "program": program, "query": _json(query)}
        assert D.replay(SLOTS, refused, pieces)[0] == "reject", query
        with pytest.raises(D.Reject):
            slots(copy.deepcopy(refused))


def test_a_q_call_unit_wider_than_the_limit_or_a_template_of_no_gate_is_refused(pieces):
    """Lean refuses as the Python does a population with an instance of `IDENTITY`, of no gate, and `SHARED` under
    `Q_call` at `X = 16`, whose first unit passes `Q_call`'s width rule with 16 returned bits while it holds a constant a
    later unit reads; at `X = 8`, where that constant is in no unit, both take the program and agree on its slots."""
    case = {"closure": None, "rule": {"law": "subset:1", "rule": "law"},
            "source": hashlib.shake_256(b"refused").hexdigest(400), "work_table": None}
    shared = json.loads(_calls((0, 5)))
    for program, query in [(json.loads(_population(((0, 1), (4, 2)))), P.template_instances_query([PAIR.id, IDENTITY.id])),
                           (shared, PO.call_query(16))]:
        refused = {**case, "program": program, "query": _json(query)}
        assert D.replay(SLOTS, refused, pieces)[0] == "reject", query
        with pytest.raises(D.Reject):
            slots(copy.deepcopy(refused))
    taken = {**case, "program": shared, "query": _json(PO.call_query(8))}
    assert D.replay(SLOTS, taken, pieces) == ("output", D.canonical(slots(copy.deepcopy(taken))))


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


def _v1_audit(kind: str, s: Setup = SETUPS[3], p: str = "1/1") -> dict:
    """One valid v1 audit of `s` under a law of `kind` (a Bernoulli one at `p`, which a run refuses at 0 < p < 1), its
    verdicts accepting."""
    law = {"law": "bernoulli", "p": p} if kind == "bernoulli" else {"k": 3, "law": kind}
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}}
    table = {t: w for t, w in table.items() if t in s.templates} if kind == "work" else None
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, work_table=table)
    params = {"anchors": [], "delta": "1/10", "law": law, "leaf_layer": LEAF, "log": [], "partition": None, "ports": s.ports,
              "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME, "work_table": table}
    evidence = {"session_roots": dict(ROOTS), "statement": {"instances": 3},
                "verdicts": {"lean": {"accepted": True, "digest": "aa" * 64, "of_record": True},
                             "upstream": {"accepted": True, "of_record": False}}}
    return _json({"evidence": evidence, "params": params, "record": rec})


def test_the_run_reaches_every_step(pieces):
    """A registration's run under each law, one-stage and (one unit's population, with its interiors) two-stage, and a v2
    registration's, Lean's the Python's whole: each accepts, on the draw from the coins, showing the verdicts' public
    outputs. The same runs refused at each step (a
    declared item missing, a record R1-R7 refuses, no session, a verdict of record about other sessions) and rejected (a
    verdict that ran and did not accept), which the drawn runs reach only probably."""
    source = hashlib.shake_256(b"run").hexdigest(600)
    v2 = _v2_audit(REGISTERED_SETUPS[0], REGISTERED_SETUPS[0].values[1])
    v2["evidence"]["verdicts"]["upstream"] = {"accepted": True, "of_record": False}
    seen = []
    laws = ("subset", "bernoulli", "stratified", "work")
    for a, stages in [*((_v1_audit(k), (False,)) for k in laws), *((_v1_audit(k, SETUPS[2]), (False, True)) for k in laws),
                      (v2, (False, True))]:
        for two_stage in stages:
            case = _run_case(a, source, 2, ["credited"], two_stage)
            if two_stage:
                case = _with_interior(case)
            other = {**case["verdicts"]["lean"], "sessions": RUN_SESSIONS[::-1]}
            for changed in (case, {**case, "public": case["public"][1:]},
                            {**case, "record": {**case["record"], "previous": "11" * 64}}, {**case, "sessions": []},
                            {**case, "verdicts": {**case["verdicts"], "lean": other}},
                            {**case, "verdicts": {**case["verdicts"], "upstream": {"accepted": False, "of_record": False}}}):
                out = run(copy.deepcopy(changed))
                assert D.replay(RUN, changed, pieces) == ("output", D.canonical(out)), (a["params"]["law"], out)
                seen.append(out.get("step") or out["symbol"])
            assert run(copy.deepcopy(case))["public"] == {"credited": 12} and run(copy.deepcopy(case))["read"] > 0
    assert seen == ["accept", "register", "register", "deliver", "outcome", "reject"] * 14


def test_the_run_holds_the_interiors(pieces):
    """A two-stage run's interiors, Lean's the Python's whole: registered after the draw, they accept with the boundary and
    their audit record is the run's. They are refused at their own step (under a one-stage rule, a population other than
    the proved RUs', a value missing, R6's previous), the proof is refused without them, and a wrong root of theirs
    rejects; the drawn runs reach each only probably."""
    source = hashlib.shake_256(b"run").hexdigest(600)
    seen = []
    for kind in ("subset", "stratified", "work"):
        case = _with_interior(_run_case(_v1_audit(kind, SETUPS[2]), source, 2, ["credited"], True))
        ir, rec = case["interior"], case["interior"]["record"]
        one = _run_case(_v1_audit(kind, SETUPS[2]), source, 2, ["credited"], False)
        for changed in (case, {**one, "interior": ir, "interior_values": case["interior_values"]},
                        {**case, "interior": {**ir, "record": {**rec, "population": rec["population"] + 1}}},
                        {**case, "interior_values": case["interior_values"][1:]},
                        {**case, "interior": {**ir, "record": {**rec, "previous": "11" * 64}}},
                        {**case, "interior": None},
                        {**case, "session_roots": {**case["session_roots"], "y": "00" * 64}}):
            out = run(copy.deepcopy(changed))
            assert D.replay(RUN, changed, pieces) == ("output", D.canonical(out)), (kind, out)
            seen.append(out.get("step") or out["symbol"])
        assert json.loads(run(copy.deepcopy(case))["interior"]["audit"])["accepted"] is True
    assert seen == ["accept", "interior", "interior", "interior", "interior", "deliver", "reject"] * 3
    none = _run_case(_v1_audit("bernoulli", SETUPS[2], "0/1"), source, 2, ["credited"], True)
    for changed in (none, {**none, "interior": ir, "interior_values": case["interior_values"]}):
        out = run(copy.deepcopy(changed))
        assert D.replay(RUN, changed, pieces) == ("output", D.canonical(out))
        seen.append(out.get("codes") or out["symbol"])
    assert seen[-2:] == ["accept", ["interior-empty"]]


def test_the_run_refuses_a_count_that_varies(pieces):
    """A registration whose law draws a count that varies, a Bernoulli one at 0 < p < 1, one-stage or as a two-stage
    rule's replay, is refused at its registration (`shapes`), Lean's the Python's whole: its sessions would set up tables
    of different shapes. At p = 0 or 1, and under subset, stratified and work, the run goes on; R1-R7 come first."""
    source = hashlib.shake_256(b"run").hexdigest(600)
    seen = []
    for s, p, two_stage in [(SETUPS[3], "1/2", False), (SETUPS[3], "2/3", False), (SETUPS[3], "2/4", False),
                            (SETUPS[2], "1/2", True), (SETUPS[3], "0/1", False), (SETUPS[3], "1/1", False),
                            (SETUPS[3], "3/3", False), (SETUPS[2], "0/1", True)]:
        case = _run_case(_v1_audit("bernoulli", s, p), source, 2, ["credited"], two_stage)
        for changed in (case, {**case, "record": {**case["record"], "previous": "11" * 64}}):
            out = run(copy.deepcopy(changed))
            assert D.replay(RUN, changed, pieces) == ("output", D.canonical(out)), (p, out)
            seen.append((p, out.get("codes") or out["symbol"]))
    assert [c for _, c in seen[0:8:2]] == [["shapes"]] * 4
    assert all(c != ["shapes"] for _, c in seen[8::2]) and all(c != ["shapes"] for _, c in seen[1::2])
    for kind in ("subset", "stratified", "work"):
        out = run(copy.deepcopy(_run_case(_v1_audit(kind), source, 2, ["credited"], False)))
        assert out["symbol"] == "accept"
    assert service.count_varies({"law": "bernoulli", "p": "1/2"})
    assert not any(service.count_varies(law) for law in ({"law": "bernoulli", "p": "0/1"}, {"law": "bernoulli", "p": "1/1"},
                                                         {"k": 2, "law": "subset"}, {"k": 2, "law": "work"}))
