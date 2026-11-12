"""`one_stage`'s Python held to sampled proofs' one-stage audit in Lean (`core_spot`), its definition and verifier (`one_stage.verifier.RECORD_VERDICT`): the whole protocol through its own `serve` (`sampled-proofs-one-stage`, a
`ProtocolPort`), and its pieces through `sampled-proofs-difftest` (`Port`s), on sampled programs, laws, records, samples and
verdicts, valid ones and one or two changes to each. A disagreement is the Python's to fix, unless it finds the Lean wrong.

The Python held to it is `one_stage` itself (`registration.check`, `draw.derive`, `check_draw`, `audit.audit_record`, `profile`,
`wrong_units`, `registered.registered_entries` and `domain`) behind the protocol's decoders, which this file writes as the
Lean reads them (`OneStage.paramsOfJson`, `commitmentOf`, `evidenceOf`), for the `draw` piece `coins.draw_first` on
`OwnRandomness` over the given bytes, for the `select` piece the challenger's sample from a Spec's selection rule on
(`service.selection_law`, `draw.derive` and `service.select_draw` on fixed coins, with the bytes it read), for the
`slots` piece that sample in the claims' slots, read off `partition.strata`, `partition.unit_gates` and the root's input
nodes (`slots`), for the `hidden_offsets`, `hidden_locate` and `hidden_witness` pieces `hidden_draw` on honest and
arbitrary calls tables, for `check_public` and `check_verdicts` the proof system's own (`service.check_public`,
`service.check_verdicts`), for the `second_statements` piece round 2's statement check (`second_stage.check`) on the
second-stage sessions' statements over one and two templates' interiors, for the `fits` piece v2's refusal of a length
a session's statement can't hold (`proofs.fits_why`, `proofs.draw_fits_why`) at small limits, and for the `run` piece
(a registration's whole run, `Run.run`) those references in a `Stream`'s order.
The `partition`, `select` and `slots` pieces also run on random partitions: mixed population
programs of random members of three templates, and rarely of `FORK` and of `IDENTITY`, of no gate, which one-stage
refuses, under template queries over them, and programs of random root Calls (`call_program`) under `Q_word` and `Q_call`, a 32-bit
Call among them, whose unit `Q_word` takes in the 32-bit class and `Q_call` at `X = 16` refuses, and `SHARED`, whose
`Q_call` unit at `X = 16` holds a constant another unit reads. Records are v1 and, on the programs with root parameters to register
(`REGISTERED_SETUPS`, 16- and 32-bit), mostly v2: the verifier expects values registered (`registered`, with its program
function's `parameters`), and the record (and an earlier one in the log) registers `Registered.commit`'s entries; on
the programs with a pair of digests to fill (`SUPPLIED_SETUPS`, 32- and 1-bit leaves, or a digest a row in a bit-row
format) the verifier fills it itself from
the record, the log's positions and its coin bound to its binding and the record's window (`supplied`,
`registered.supplied_root`), which the record registers as it fills it, and it may pin a value's root (`pinned`, R7-held)
or take W as an output of the log's earlier record (`outputs`: W's entry is that record's port `EARLIER_OUT`, its domain
and root, or that record's registered value `EARLIER_VALUE`, its domain and root at W's leaves, words and word bits,
R7-output), and be bound to a stream and a prover (`binding`, which the record's window names: R6-stream,
R6-prover). A verifier may name its window's open root (`window_root`, `WROOT`) and the values it holds across windows
(`held`): the record registers every other value it doesn't fill, pin or take as an output under `WROOT`'s domain, and
the log's earlier record under another window's (`EARLIER_WROOT`), its own or none. Such a verifier may read its
window's seed salt from its challenger (`seed_salt`, `SEED_SALT`), which the record's window names (R6-seed-salt).
The verifier's program is looked up among this file's programs by its descriptor. Inputs
stay inside what Python reads as Lean does: integers that are not booleans where Python compares with `==`, and integer
populations in a sample.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import random
import re
import shutil
from fractions import Fraction
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from hypothesis import strategies as st

import settings
import verity_difftest as D
from verity_difftest import protocol as PR
from verity.protocols.pouw.verifier import audit as PA
from verity.protocols.pouw.verifier.protocol import tagged
from verity.protocols.spot.verifier.profile import Bernoulli, Stratified, Stratum
from verity.protocols.spot.device import challenger as CH
from verity.protocols.spot.verifier import schedule as SCH, second_stage as SS, service
from verity.protocols.spot.verifier.one_stage import (audit, draw as DR, hello as H, hidden_draw as HD,
                                                            partition as P, produced as PD, proofs as PF,
                                                            registered as G, registration as R)
from verity.protocols.spot.device.challenger import COIN_BYTES, NO_CLOSE, NO_TREE, ChallengerKey, CoinError
from verity.protocols.spot.device.coins import (LIMITS, MAX_PATH, OwnRandomness, coins_reason, coins_within,
                                                        draw_coins, draw_first)
from verity.protocols.spot.verifier.one_stage.proofs import order_why
from verity.protocols.spot.verifier.one_stage.verdict import Verdict
from verity.circuits import codec, partition_object as PO
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program, input_id, is_input
from verity.circuits.refs import array_of, record_of, tuple_of
from verity.circuits.types import Array, RecordType, Tuple, Value
from verity.commitments.window import window_binding

from test_challenger_difftest import DEPLOYED, DRAW_MUTATIONS, ORDERS, Ch, Drawn, _flip, _mutated, lms_verify

PACKAGE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

XOR8 = PrimitiveDefinition("CoreSpotXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("CoreSpotPair", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("CoreSpotTriple", 0, (), lambda S: ((("x", Array(3, Value(8))),), Value(8)),
                                  lambda B, S, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
QUAD = bind(CompositeDefinition("CoreSpotQuad", 0, (), lambda S: ((("x", Array(4, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), B.call(XOR8, x[2], x[3])),
                                register=False))
TEMPLATES = [PAIR, TRIPLE, QUAD]
#: A two-stage design's replay block: three proof blocks, instances of `PAIR`.
BLOCK = bind(CompositeDefinition("CoreSpotBlock", 0, (), lambda S: ((("x", Array(3, Array(2, Value(8)))),),
                                                                         Array(3, Value(8))),
                                 lambda B, S, x: B.batch(PAIR, x, axes=(0,)), register=False))
XOR32 = PrimitiveDefinition("CoreSpotXor32", 1, (("a", Value(32)), ("b", Value(32))), Value(32),
                            lambda a, b: a ^ b, register=False)


def _fork(B, S, x):
    a = B.call(XOR8, x[0], x[1])
    first, last = B.call(XOR8, a, x[2]), B.call(XOR8, a, x[3])
    return tuple_of(first, B.call(XOR8, x[2], x[3]), last)


#: Three outputs, two units at `Q_word`'s `X = 16`: the first two, whose gates the last one's interleaves, and the last;
#: the interior value both read is committed, a unit of its own.
FORK = bind(CompositeDefinition("CoreSpotFork", 0, (),
                                lambda S: ((("x", Array(4, Value(8))),), Tuple(Value(8), Value(8), Value(8))), _fork,
                                register=False))
#: A template of no gate, which one-stage refuses.
IDENTITY = bind(CompositeDefinition("CoreSpotIdentity", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                    lambda B, S, x: x[0], register=False))
#: The templates a population's members are instances of: `TEMPLATES`, `FORK` and `IDENTITY`.
MEMBERS = TEMPLATES + [FORK, IDENTITY]
CONST8 = PrimitiveDefinition("CoreSpotConst8", 1, (), Value(8), lambda: 0x5A, register=False)


def _held(B, S, x):
    a = B.call(XOR8, x[0], x[1])
    return tuple_of(a, B.call(XOR8, a, x[1]), B.call(CONST8))


def _shared(B, S, x):
    s = B.call(HELD, x)
    return tuple_of(s[0], s[1], B.call(XOR8, s[0], s[2]))


HELD = bind(CompositeDefinition("CoreSpotHeld", 0, (),
                                lambda S: ((("x", Array(2, Value(8))),), Tuple(Value(8), Value(8), Value(8))), _held,
                                register=False))
#: A Call whose first `Q_call` unit at `X = 16`, `HELD`, returns 16 bits and holds a constant the last gate reads: 24
#: committed bits, in the 32-bit class, which `Q_call`'s width rule doesn't count; at `X = 8` the constant is in no unit.
SHARED = bind(CompositeDefinition("CoreSpotShared", 0, (),
                                  lambda S: ((("x", Array(2, Value(8))),), Tuple(Value(8), Value(8), Value(8))), _shared,
                                  register=False))
#: The Calls a `call_program` may make: the templates, the fork, the 8- and 32-bit primitives and the shared constant's.
CALLEES = [PAIR, TRIPLE, FORK, XOR8, XOR32, SHARED]
CALLEE_IDS = [codec.descriptor_id(f) for f in CALLEES]
SCHEME = "frame-v3-sha512/hm96-sha512"
LEAF = "ee" * 64
ROOTS = {"out": "cd" * 64, "x": "ab" * 64}
#: the verifier's coin a window salt binds
COIN = "c0" * 64
#: the verifier's binding (its stream's digest and the prover's registered identity), which a bound record's window names
BINDING = {"stream": "3c" * 64, "prover": "d0" * 32}
WINDOW = {"kind": "served", "row": 101, "run": "r-test"}
EARLIER = {"kind": "served", "row": 100, "run": "r-test"}
#: the earlier record's output port a verifier's `outputs` may name as W's commitment, and its domain
EARLIER_OUT = "earlier.out"
EARLIER_OUT_DOMAIN = "dd" * 64
#: the earlier record's registered value a verifier's `outputs` may reuse as W's commitment, under that domain
EARLIER_VALUE = "earlier.W"
#: the window's open root a verifier may name, and an earlier window's
WROOT = "a7" * 64
EARLIER_WROOT = "e0" * 64
#: the window's seed salt a windowed verifier may read from its challenger, which a seeded record's window names, and
#: another
SEED_SALT = "5a" * 64
OTHER_SEED_SALT = "5b" * 64
RECEIVED = "2026-10-07T00:00:00Z"


def _json(value):
    """The value as Lean reads it: canonical JSON's, keys sorted."""
    return json.loads(D.canonical(value))


class Setup:
    """A verifier's program and query, with its descriptor and the ports its layout derives, each `rows` rows a unit."""

    def __init__(self, program, query, names=("out", "x"), rows=(2, 1)):
        self.program, self.query = program, query
        self.desc = _json(codec.encode_program(program))
        self.sha512 = PO.program_sha512(program)
        self.population = P.units(program, query)
        self.templates = [t for t, _ in P.strata(program, query)]
        part = P.partition(program, query)
        self.ports = {p: {"domain": R.window_domain(self.sha512, P.digest(part), p, schema, leaves, "r-test"), "leaves": leaves}
                      for p, schema, leaves in ((names[0], "u16", rows[0] * self.population),
                                                (names[1], "hm96-sha512/row/v1", rows[1] * self.population))}


def call_program(members) -> Program:
    """A root of one Call per member, an index into `CALLEES`, each over a parameter of its own (a primitive's two values
    as an array of two), returning their results: what `Q_word` and `Q_call` cut, Call by Call."""
    def param(f):
        return Array(2, f.signature()[0][0][1]) if isinstance(f, PrimitiveDefinition) else f.signature()[0][0][1]

    def call(B, f, x):
        return B.call(f, x[0], x[1]) if isinstance(f, PrimitiveDefinition) else B.call(f, x)

    fns = [CALLEES[m] for m in members]
    defn = CompositeDefinition(f"CoreSpotCalls{''.join(map(str, members))}", 0, (),
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
    committed once at each of its words (`Registered.commit`, salts fixed) in its row format (`formats`, `{parameter:
    schema}`, `row/v1` where it names none), and its signature as `parameters` names it."""

    def __init__(self, program, values, formats=None):
        super().__init__(program, P.template_instance_query(PAIR.id))
        self.signature = _json([[name, t.to_json()] for name, t in program.fn.signature()[0]])
        self.values = values
        self.formats = dict(formats or {})
        self.committed = {}
        self._windowed = {}
        for v, w in {(v, w) for vs in values for v, w in vs.items()}:
            self.committed[v, w] = G.Registered.commit(self.sha512, v, *self._rows(v, w))

    def _rows(self, v: str, w: int) -> tuple:
        """`Registered.commit`'s rows, salts, width and row format for `v` at `w` words."""
        schema = self.formats.get(v, G.SCHEMA)
        n, bits = len(G.parameter_gates(self.program, v)), G.width(self.program, v)
        k = G.row_leaves(self.program, v, w, schema)
        rows = [[(i * 40503 + 7) % (1 << bits) for i in range(r * w, (r + 1) * w)] for r in range(n // k)]
        return rows, [hashlib.shake_256(f"{v}/{w}/{r}".encode()).digest(192) for r in range(n // k)], bits, schema

    def entries(self, values, window: str | None = None, fixed=()) -> list[dict]:
        """`values`' entries: each one's commitment, under `window`'s domain unless it is in `fixed`."""
        out = []
        for v, w in (values or {}).items():
            if window is None or v in fixed:
                out.append(self.committed[v, w].entry())
                continue
            if (v, w, window) not in self._windowed:
                self._windowed[v, w, window] = G.Registered.commit(self.sha512, v, *self._rows(v, w), window=window)
            out.append(self._windowed[v, w, window].entry())
        return out

    def supplied(self, values) -> dict:
        """Of `values`, the ones the verifier fills itself, with their sources."""
        return {}


class SuppliedSetup(RegisteredSetup):
    """A `RegisteredSetup` whose verifier fills some values itself: `sup`, `{value: rows of sources}`
    (`registered.sources`), each expected registered with the others in some of `values`, in its row format of
    `formats` (`row/v1` where it names none)."""

    def __init__(self, program, values, sup, formats=None):
        super().__init__(program, values, formats)
        self.sup = sup

    def supplied(self, values) -> dict:
        return {v: self.sup[v] for v in values or {} if v in self.sup}


#: Programs whose root parameters a verifier may expect registered: 16-bit ones (`W`, and `D` of 12-bit leaves) and ones
#: `registered_entries` refuses (`E` has no gates, `M` two widths, `V` is 40 bits), 32-bit ones (`T`, and `U` of
#: 20-bit leaves), and bit rows (`A`, 8-bit leaves two to a u16 word in `row/v2` rows, as H1's A, and `B` in
#: `row-seg/v1` rows); `x`, bound per run, reads as a 16-bit value too.
REGISTERED_SETUPS = [
    RegisteredSetup(_registered_program("CoreSpotRegistered16", 3, (
        ("W", Array(6, Value(16))), ("D", Array(4, Value(12))), ("E", Array(0, Value(16))), ("M", Tuple(Value(8), Value(16))),
        ("V", Value(40))), ("W", "D")), [{"W": 2}, {"D": 1, "W": 3}, {"D": 4}, {"W": 6, "x": 2}]),
    RegisteredSetup(_registered_program("CoreSpotRegistered32", 2, (
        ("T", Array(4, Value(32))), ("U", RecordType(("a", Array(3, Value(20))),))), ("T", "U")),
        [{"T": 2}, {"T": 1, "U": 3}, {"U": 1}, {"T": 4, "x": 1}]),
    RegisteredSetup(_registered_program("CoreSpotRegisteredBits", 2, (
        ("A", Array(8, Value(8))), ("B", Array(4, Value(16)))), ("A", "B")),
        [{"A": 2}, {"A": 1, "B": 2}, {"B": 4}, {"A": 4, "x": 2}], {"A": G.SCHEMA_V2, "B": G.SCHEMA_SEG})]
#: Programs with a value the verifier fills itself, as PoUW's `pouw.roots`: `S`, each row a pair of digests of 32-bit
#: leaves (rows of x's root and W's registered root, and W's and out's) or of 1-bit leaves (out's root and bytes of its own),
#: or, in a bit-row format, one `Digest` a row in `row/v2` (out's root, then bytes of its own: 32 u16 words a row) or
#: one digest of 1-bit leaves a row in `row-seg/v1` (x's root, then bytes of its own).
SUPPLIED_SETUPS = [
    SuppliedSetup(_registered_program("CoreSpotSupplied32", 3, (
        ("S", Array(2, Array(2, Array(16, Value(32))))), ("W", Array(6, Value(16)))), ("S", "W")),
        [{"S": 32, "W": 2}, {"W": 3, "S": 32}, {"W": 2}],
        {"S": [[{"port": "x"}, {"value": "W"}], [{"value": "W"}, {"port": "out"}]]}),
    SuppliedSetup(_registered_program("CoreSpotSupplied1", 2, (
        ("S", Array(1, Array(2, Array(512, Value(1))))),), ("S",)),
        [{"S": 1024}, {"x": 2, "S": 1024}], {"S": [[{"port": "out"}, {"bytes": "5a" * 64}]]}),
    SuppliedSetup(_registered_program("CoreSpotSuppliedPW", 3, (
        ("S", Array(2, Array(2, Array(16, Value(32))))), ("W", Array(6, Value(16)))), ("S", "W")),
        [{"S": 32, "W": 2}, {"W": 3, "S": 32}],
        {"S": [[{"window": COIN}] + [{"position": "0"}] * 8, [{"position": "2"}] * 8 + [{"window": COIN}]]}),
    SuppliedSetup(_registered_program("CoreSpotSuppliedDigestV2", 2, (
        ("S", Array(2, Array(8, Array(64, Value(1))))),), ("S",)),
        [{"S": 32}, {"x": 2, "S": 32}], {"S": [[{"port": "out"}], [{"bytes": "5a" * 64}]]}, {"S": G.SCHEMA_V2}),
    SuppliedSetup(_registered_program("CoreSpotSuppliedBitsSeg", 2, (
        ("S", Array(2, Array(512, Value(1)))),), ("S",)),
        [{"S": 32}, {"x": 2, "S": 32}], {"S": [[{"port": "x"}], [{"bytes": "a5" * 64}]]}, {"S": G.SCHEMA_SEG})]
REGISTERED_SETUPS += SUPPLIED_SETUPS
#: A program whose window opens a pool from `pin` and leaves it in `pout`, each one SHA-512 digest of 32-bit leaves, and
#: the empty pool's root a chain restarts from.
POOLED = RegisteredSetup(_registered_program("CoreSpotPooled", 2, (
    ("pin", Array(1, Array(16, Value(32)))), ("pout", Array(1, Array(16, Value(32))))), ("pin", "pout")),
    [{"pin": 16, "pout": 16}])
#: `POOLED` whose window also reads its certified arrivals `ing` (four slots of a valid bit, an id and a length) and its
#: rows `req`, which no unit reads.
CONTINUED = RegisteredSetup(_registered_program("CoreSpotContinued", 2, (
    ("ing", Array(4, Array(3, Value(16)))), ("req", Array(4, Array(12, Value(16)))),
    ("pin", Array(1, Array(16, Value(32)))), ("pout", Array(1, Array(16, Value(32))))), ("ing", "req", "pin", "pout")),
    [{"ing": 3, "req": 12, "pin": 16, "pout": 16}])
GENESIS = "9e" * 64
#: A two-stage design's two registrations: the boundary one over four replay blocks under the grain, its ports its own,
#: and the interior one over a replay sample of two of them, six proof blocks under the check's query.
#: The boundary's ports: `BLOCK`'s own, under its descriptor id, which the links round 2 derives from it read, each its
#: three PAIR instances' rows a replay block, of PAIR's `x` and output each.
BX, BOUT = (f"{codec.descriptor_id(BLOCK)}/{p}" for p in ("x", "o0"))
BOUNDARY = Setup(P.population_program(BLOCK, 4), P.template_instance_query(BLOCK.id), names=(BOUT, BX), rows=(3, 3))
INTERIOR = Setup(P.population_program(PAIR, 6), P.template_instance_query(PAIR.id))
PROGRAMS = {s.sha512: s.program for s in SETUPS + REGISTERED_SETUPS + [POOLED, CONTINUED, BOUNDARY, INTERIOR]}
PROGRAMS[PO.program_sha512(Program(BLOCK))] = Program(BLOCK)
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
HEX_DIGITS = re.compile(r"[0-9a-f]*")


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
        return RecordType(*[(k, _type(t)) for k, t in j[1]])
    raise D.Reject(f"type {j!r}")


def _lower(t) -> list[tuple[int, int]]:
    """`program._lower_input`'s input nodes, `(width, n)` each."""
    w = t.homogeneous_width()
    if w is not None:
        return [(w, t.leaves)]
    fields = [t.elem] * t.n if isinstance(t, Array) else [f[1] if isinstance(t, RecordType) else f for f in t.fields]
    return [node for f in fields for node in _lower(f)]


def _parameters(desc: dict, program: Program, j) -> None:
    """`Registered.params`: the program function's `[[name, type], ...]`, distinct names whose types lower to exactly the
    root body's input nodes. The strategies sample no signature but the program's own that lowers so, which this file would
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
    """`OneStage.logRecord`: an object whose `registered`, when truthy, lists objects with a string `value`, `root` and
    `domain`."""
    r = e.get("registered") if isinstance(e, dict) else None
    if not isinstance(e, dict) or (r and not (isinstance(r, list) and all(
            isinstance(x, dict) and all(isinstance(x.get(k), str) for k in ("value", "root", "domain")) for x in r))):
        raise D.Reject("a record of the log")


def decode_params(j) -> dict:
    v2 = isinstance(j, dict) and j.get("registered") is not None
    tied = isinstance(j, dict) and "tied" in j
    keys = PARAM_KEYS | {"parameters"} if v2 or tied else PARAM_KEYS
    keys = keys | {"tied"} if tied else keys
    keys = keys | {"supplied"} if isinstance(j, dict) and "supplied" in j else keys
    keys = keys | {"pinned"} if isinstance(j, dict) and "pinned" in j else keys
    keys = keys | {"outputs"} if isinstance(j, dict) and "outputs" in j else keys
    keys = keys | {k for k in ("row_schemas", "window_root", "held", "challenger", "seed_salt", "pool")
                   if isinstance(j, dict) and k in j}
    _exactly(j, keys | {"binding"} if isinstance(j, dict) and "binding" in j else keys, "the parameters")
    desc = _obj(j["program"], "program")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    _obj(j["query"], "query")
    if j["work_table"] is not None:
        _obj(j["work_table"], "work_table")
    try:
        law, table = service.params_law(j["law"], PROGRAMS[sha], j["query"], j["work_table"])
    except ValueError as e:
        raise D.Reject(str(e)) from None
    rule(law)
    j = {**j, "law": law, "work_table": table}
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
    if v2 and not all(map(_nat, _obj(j["registered"], "registered").values())):
        raise D.Reject("registered: words that are not a natural number")
    if v2 or tied:
        _parameters(desc, PROGRAMS[sha], j["parameters"])
    if tied and not (j["tied"] is None or isinstance(j["tied"], list) and all(isinstance(t, str) for t in j["tied"])):
        raise D.Reject("tied values are a list of the program's parameters")
    raw = j
    if "pool" in j:
        try:
            pool = R._pool(j["pool"], j["registered"] or {}, j.get("supplied"), j.get("pinned"), j.get("outputs"),
                           j.get("row_schemas"))
        except ValueError as e:
            raise D.Reject(str(e)) from None
        compiled = R.pool_values(pool, PROGRAMS[sha], j.get("supplied"), j.get("outputs"), j.get("row_schemas"), j["log"])
        j = {**j, **{k: x for k, x in zip(("supplied", "outputs", "row_schemas"), compiled) if x is not None}}
    if "supplied" in j:
        if not j["registered"]:
            raise D.Reject("supplied values are registered values: the verifier expects none registered")
        try:
            G.sources(j["supplied"])
        except ValueError as e:
            raise D.Reject(str(e)) from None
    if "pinned" in j:
        if not isinstance(j["pinned"], dict):
            raise D.Reject("pinned roots are not an object")
        try:
            R._pins(j["pinned"], j["registered"] or {}, G.sources(j.get("supplied", {})))
        except ValueError as e:
            raise D.Reject(str(e)) from None
    if "outputs" in j:
        if not isinstance(j["outputs"], dict):
            raise D.Reject("output values are not an object")
        try:
            R._outputs(j["outputs"], j["registered"] or {}, G.sources(j.get("supplied", {})), j.get("pinned") or {})
        except ValueError as e:
            raise D.Reject(str(e)) from None
    if "row_schemas" in j:
        if not isinstance(j["row_schemas"], dict):
            raise D.Reject("row formats are not an object")
        try:
            R._schemas(j["row_schemas"], j["registered"] or {})
        except ValueError as e:
            raise D.Reject(str(e)) from None
    if "binding" in j:
        if not isinstance(j["binding"], dict):
            raise D.Reject("the binding is not an object")
        try:
            R._binding(j["binding"])
        except ValueError as e:
            raise D.Reject(str(e)) from None
    try:
        window, fixed = R._window(j.get("window_root"), j.get("held"), j["registered"] or {},
                                  G.sources(j.get("supplied", {})), j.get("pinned") or {}, j.get("outputs") or {})
    except ValueError as e:
        raise D.Reject(str(e)) from None
    try:
        R._seed_salt(j.get("seed_salt"), window)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    if "challenger" in j:
        _challenger(j["challenger"], window)
    delta = ratio(j["delta"])
    if not 0 < delta < 1:
        raise D.Reject("delta is not in (0, 1)")
    if R.unwindowed(PROGRAMS[sha], j["registered"] or {}, j.get("row_schemas"), window, fixed, j["log"]):
        raise D.Reject("a value it doesn't hold across windows that the log registered, and no window root")
    return {**raw, "program": PROGRAMS[sha], "delta": delta}


def _challenger(ch, window) -> ChallengerKey:
    """`OneStage.paramsOfJson`'s `challenger`: exactly `{key, label}`, its key `ChallengerKey.ofJson`'s, its label a string,
    and a window root to read the coins on."""
    _exactly(ch, {"key", "label"}, "the challenger")
    if not isinstance(ch["label"], str):
        raise D.Reject("the challenger's label is not a string")
    try:
        key = ChallengerKey.from_json(ch["key"])
    except (CoinError, ValueError, TypeError, KeyError) as e:
        raise D.Reject(f"the challenger's key: {e}") from None
    if window is None:
        raise D.Reject("a draw from the challenger's coins needs the window's open root (window_root)")
    return key


def decode_commitment(b: bytes) -> tuple[dict, bytes]:
    rec = PR.canonical_json(b)
    if not isinstance(rec, dict):
        raise D.Reject("the registration record is not an object")
    return rec, bytes.fromhex(R.digest(rec))


def decode_evidence(b: bytes) -> dict:
    j = PR.canonical_json(b)
    _exactly(j, EVIDENCE_KEYS | {"coins"} if isinstance(j, dict) and "coins" in j else EVIDENCE_KEYS, "the evidence")
    nats = lambda x: isinstance(x, list) and all(map(_nat, x))  # noqa: E731
    if not (_nat(j["instances"]) and isinstance(j["verdicts"], dict) and all(isinstance(v, dict) for v in j["verdicts"].values())
            and isinstance(j["sessions"], list) and all(isinstance(s, str) for s in j["sessions"])
            and isinstance(j["receipt"], dict) and nats(j["proved_units"]) and isinstance(j["statement"], dict)
            and (j["served"] is None or nats(j["served"])) and (j["session_roots"] is None or isinstance(j["session_roots"], dict))):
        raise D.Reject("the evidence's fields")
    _registered_reads(j["verdicts"])
    return j


def _registered_reads(verdicts: dict) -> None:
    """`OneStage.evidenceOf`'s rule on the verdicts: the verifier that accepted reports its registered reads as
    objects with a string value and a root."""
    for v in verdicts.values():
        if v.get("of_record") and v.get("accepted") is True and isinstance(v.get("registered"), dict) and not all(
                isinstance(e, dict) and isinstance(e.get("value"), str) and "root" in e for e in v["registered"].values()):
            raise D.Reject("a verifier of record reports registered reads that are not objects with a string value and a root")


def _registration_verdict(p: dict, rec) -> Verdict:
    return R.check(rec, p["program"], p["query"], p["law"], p["scheme"], p["ports"], p["leaf_layer"], p["anchors"], p["log"],
                   p["partition"], p["work_table"], p["registered"], p.get("supplied"), p.get("pinned"),
                   p.get("binding"), p.get("outputs"), p.get("row_schemas"), p.get("window_root"), p.get("held"),
                   seed_salt=p.get("seed_salt"), pool=p.get("pool"), tied=p.get("tied"))


def _registration(p: dict, rec) -> tuple[str, ...]:
    return _registration_verdict(p, rec).codes


def _coins_of(p: dict, c: tuple[dict, bytes], e: dict) -> bytes | None:
    """`OneStage.coinsOf`: with the verifier's challenger, the coins `coins.draw_coins` reads from the evidence's `coins`
    under its key, on the verifier's window, of its family, after the registration's digest; none without one, when the
    evidence carries no coins. Raises `CoinError`."""
    ch = p.get("challenger")
    if ch is None:
        if e.get("coins") is not None:
            raise CoinError("the evidence carries coins, and the verifier names no challenger they come from")
        return None
    return draw_coins(ChallengerKey.from_json(ch["key"]), p["window_root"], ch["label"], c[1], e.get("coins"), lms_verify)


def _coins_problem(rec: dict, mine: dict, draw: dict, src: bytes) -> bool | ValueError:
    """`OneStage.coinsReason`: the sample's units are not the ones `src` samples under the sampling design (`coins.draw_first` read in
    order, its refusal the problem, and a Bernoulli sampling design refused, `OneStage.coveredDesign`), or a coin opened was not
    read."""
    n = rec.get("population")
    if not _nat(n):
        return True
    if mine.get("law") == "bernoulli":
        return ValueError("the law is bernoulli, which the sampled audit's soundness statement doesn't cover")
    read = 0

    def source(k: int) -> bytes:
        nonlocal read
        b = src[read:read + k]
        read += len(b)
        return b
    try:
        units = draw_first(OwnRandomness(source), mine, n)["units"]
    except ValueError as refused:
        return refused
    drawn = [u for u in draw["units"] if _nat(u)] if isinstance(draw.get("units"), list) else []
    return list(units) != drawn or -(-read // COIN_BYTES) != len(src) // COIN_BYTES


def check(p: dict, c: tuple[dict, bytes], e: dict) -> tuple:
    """The outcome `OneStage.check` gives, from `one_stage`'s own checks: reject when the sample's coins don't read
    (`_coins_of`), on R1-R7, the sample's checks (with coins, that they sample it: `_coins_problem`), the record's checks or a
    verdict that ran and did not accept; undecided when only the verifier is missing or has not run; accept
    exactly when `audit_record` accepts, with the sampled and proved units, the bounds and the profile."""
    rec, _ = c
    try:
        coins = _coins_of(p, c, e)
    except CoinError:
        return "reject", "coins"
    if _registration(p, rec):
        return "reject", "registration"
    mine = DR.derive(p["program"], p["query"], p["law"], p["work_table"])
    if not DR.check_draw(e["draw"], rec, mine).ok:
        return "reject", "draw"
    if coins is not None and _coins_problem(rec, mine, e["draw"], coins):
        return "reject", "coins"
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
    public = {"bound": wrong["bound"], "drawn": list(e["draw"]["units"]),
              "per_template": wrong.get("per_template"), "proved": DR.proved(e["draw"])}
    return "accept", public, {**profile, "delta": str(profile["delta"])}


def covered_params(j) -> dict:
    """`OneStage.paramsOfJson`: `decode_params` (`paramsParsed`), and every bernoulli law refused, at any p
    (`service.bernoulli`)."""
    p = decode_params(j)
    if service.bernoulli(p["law"]):
        raise D.Reject(service.BERNOULLI)
    return p


def covered_evidence(b: bytes) -> dict:
    """`OneStage.evidenceOf`: `decode_evidence` (`evidenceParsed`), and a sample that carries a closure refused."""
    e = decode_evidence(b)
    if isinstance(e["draw"], dict) and "closure" in e["draw"]:
        raise D.Reject("the draw carries a closure, which the sampled audit's soundness statement doesn't cover")
    return e


def answer(case) -> dict:
    return PR.answer(R.PROTOCOL, _json(case), params=covered_params, commitment=decode_commitment,
                     evidence=covered_evidence, digest=lambda c: c[1], check=check)


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
    v = _registration_verdict(_params_of(j), j.get("record") if isinstance(j, dict) else None)
    return {"codes": list(v.codes), **({} if v.fanout is None else {"fanout": v.fanout}),
            **({"rests_on_period_check": list(v.rests_on_period_check)} if v.rests_on_period_check else {})}


def registered(j):
    p = _params_of(_json(j))
    fixed = [*(p.get("held") or ()), *(p.get("pinned") or {}), *G.sources(p.get("supplied", {})), *(p.get("outputs") or {})]
    try:
        return G.registered_entries(p["program"], p["registered"] or {}, p.get("row_schemas"), p.get("window_root"), fixed)
    except ValueError:
        return None


def domain(j):
    j = _json(j)
    if not (isinstance(j, dict) and isinstance(j.get("program"), str) and isinstance(j.get("value"), str)
            and all(_nat(j.get(k)) for k in ("words", "leaves", "word_bits"))):
        raise D.Reject("program or value is not a string, or words, leaves or word_bits not a natural number")
    for k in ("window", "produced"):
        if k in j and j[k] is None:
            raise D.Reject(f"the {k} is not 64 bytes of hex")
    try:
        return G.domain(j["program"], j["value"], j["words"], j["leaves"], j["word_bits"], j.get("window"),
                        j.get("produced")).domain_id.hex()
    except ValueError as e:
        raise D.Reject(str(e)) from None


def produced(j):
    j = _json(j)
    if not (isinstance(j, dict) and isinstance(j.get("registration"), str) and isinstance(j.get("draw"), str)):
        raise D.Reject("registration or draw is not a string")
    mine, opened = j.get("mine"), j.get("opened")
    if not isinstance(mine, dict) or not all(
            isinstance(e, dict) and all(_nat(e.get(k)) for k in ("leaves", "words", "word_bits"))
            and isinstance(e.get("domain"), str) for e in mine.values()):
        raise D.Reject("mine is not {value: leaves, words and word_bits and a string domain}")
    if opened is not None and not isinstance(opened, dict):
        raise D.Reject("opened is not an object")
    known, verdicts = ({} if j.get(k) is None else j[k] for k in ("known", "verdicts"))
    if not isinstance(known, dict) or not all(
            isinstance(rows, list) and all(isinstance(r, list) and len(r) == 2 and _nat(r[0]) and isinstance(r[1], str)
                                           for r in rows) and len({r[0] for r in rows}) == len(rows)
            for rows in known.values()):
        raise D.Reject("known is not {value: [[position, b || c] ...]} with distinct natural positions")
    if not isinstance(verdicts, dict):
        raise D.Reject("verdicts is not an object")
    codes = list(PD.check(j.get("record"), j["registration"], j["draw"], mine))
    if codes:
        return {"codes": codes, "opened": None, "shared": None}
    return {"codes": codes, "opened": list(PD.opened(j["record"], opened)),
            "shared": list(PD.shared(PD.wanted(j["record"], known), verdicts))}


def known_port(j):
    j = _json(j)
    keys = ("program", "value", "salt", "registration")
    if not (isinstance(j, dict) and all(isinstance(j.get(k), str) for k in keys)):
        raise D.Reject("program, value, salt or registration is not a string")
    if not (isinstance(j.get("row"), list) and all(map(_nat, j["row"]))) or not _nat(j.get("word_bits")):
        raise D.Reject("row is not natural numbers, or word_bits not a natural number")
    if not re.fullmatch(r"(?:[0-9a-f]{2})*", j["salt"]):
        raise D.Reject("the salt is not lowercase hex")
    try:
        port, (_, bc) = PD.known_port(j["program"], j["value"], j["row"], bytes.fromhex(j["salt"]), j["registration"],
                                      j["word_bits"])
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return {"root": port["root"], "bc": bc}


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
    """`Select.Rule.ofJson`: a selection rule as `to_json` writes it, exactly its keys, of the kinds a `Stream` runs."""
    kind = j.get("rule") if isinstance(j, dict) else None
    try:
        if kind == "all" and set(j) == {"rule"}:
            return service.All()
        if kind == "law" and set(j) == {"law", "rule"} and isinstance(j["law"], str):
            return service.Law(j["law"])
        if (kind == "two-stage" and set(j) == {"grain", "k", "replay", "rule"} and isinstance(j["replay"], str)
                and (_nat(j["k"]) or isinstance(j["k"], dict)) and service._grain(j["grain"])):
            return service.TwoStage(j["grain"], j["replay"], j["k"])
    except ValueError as e:
        raise D.Reject(str(e)) from None
    raise D.Reject("not a selection rule the stream runs")


def select(j):
    """The challenger's sample as `Stream.register` and `Stream.select` make it: `service.selection_law` over the
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
        table = service.selection_table(rule, program, query, j.get("work_table"))
        law = DR.derive(program, query, service.selection_law(rule, n, service.selection_sizes(rule, program, query)), table)
        obj = service.select_draw(rule, law, n, coins, closure=closure)
    except (P.QueryRefused, ValueError, service.Refused) as e:
        raise D.Reject(str(e)) from None
    if not DR.check_draw(obj, {"population": n}, law).ok:
        raise D.Reject("the draw is not well formed under the law")
    return {"draw": obj, "read": at}


def select_second(j):
    """A two-stage design's second stage (`service.select_second`) over the replay sample `sample`, each template's
    `ns[t]` proof blocks in each replay block (a cell) and `ks[t]` sampled in each, on the source's bytes read in order;
    with the number of bytes read."""
    j = _json(j)
    if not (isinstance(j, dict) and isinstance(j.get("sample"), list) and all(map(_nat, j["sample"]))):
        raise D.Reject("sample is not a list of replay blocks")
    if not all(isinstance(j.get(f), list) and all(map(_nat, j[f])) for f in ("ns", "ks")):
        raise D.Reject("ns and ks are not lists of counts")
    if not isinstance(j.get("source"), str):
        raise D.Reject("source is not hex")
    src, at = PR.unhex(j["source"]), 0

    def coins(n: int) -> bytes:
        nonlocal at
        at += n
        return src[at - n:at]
    try:
        obj = service.select_second(j["sample"], j["ns"], j["ks"], coins)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return {"draw": obj, "read": at}


def template_circuits(j):
    """A two-stage run's params naming each template's circuit file and its statements' reference header and registered
    reads (`service.template_circuits`), in the order the check's template query names the templates; a missing `proofs`
    names none."""
    j = _json(j)
    proofs, query = (j.get("proofs"), j.get("query")) if isinstance(j, dict) else (None, None)
    try:
        named = service.template_circuits(proofs, query)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return {"circuits": [[t, c, h, r] for t, c, h, r in named]}


def slots(j):
    """`Slots.select`: the claims' partition over the select piece's program and query, two classes, and its sample's
    checked proof blocks. The population's strata are `partition.strata`'s, each unit a proof block, its gates
    (`partition.unit_gates`); the input class is one stratum of every gate of the root's input nodes (`program.is_input`),
    each its own proof block. A proof block's slot is its index in that order, and the checked proof blocks are
    `(slot, 0)` of each unit the sample proves (`draw.proved`) and of each input gate, in slot order."""
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
    blocks = [b for c in classes for s in c for b in s]
    return {"checked": [[v, 0] for v, b in enumerate(blocks) if b["unit"] is None or b["unit"] in proved], "classes": classes}


_LOWER_HEX = re.compile(r"(?:[0-9a-f]{2})*")


def _unhex(x) -> bytes:
    if not (isinstance(x, str) and _LOWER_HEX.fullmatch(x)):
        raise D.Reject("not lowercase hex of whole bytes")
    return bytes.fromhex(x)


def fits(j):
    """`OneStageZk.fitsWhy` and `drawFitsWhy` at ``limit`` (`proofs.fits_why`, `proofs.draw_fits_why`): the files hex,
    the tables ``[name, hex]`` pairs, the program hex or null, and the partition object, registered reads and draw null
    for none."""
    j = _json(j)
    if not (isinstance(j, dict) and _nat(j.get("limit"))):
        raise D.Reject("limit is not a natural number")
    circuit, public = _unhex(j.get("circuit")), _unhex(j.get("public"))
    ts = j.get("tables")
    if not (isinstance(ts, list) and all(isinstance(t, list) and len(t) == 2 and all(isinstance(x, str) for x in t)
                                         for t in ts)):
        raise D.Reject("tables is not a list of names and their hex")
    tables = [(name, _unhex(b)) for name, b in ts]
    program = None if j.get("program") is None else _unhex(j["program"])
    return {"inputs": PF.fits_why(j["limit"], circuit, public, tables, program, j.get("partition"), j.get("registered")),
            "draw": PF.draw_fits_why(j["limit"], j.get("draw"))}


def _public_units(public) -> list:
    """`FilePart.unitIndices`: the public file's ``units.indices`` (``public``, its bytes in hex), off its header as the
    Python verifier reads it (`proofs._header`: the first line, decoded, ``json.loads``)."""
    if not (isinstance(public, str) and len(public) % 2 == 0 and re.fullmatch(r"[0-9a-f]*", public)):
        raise D.Reject("public is not lowercase hex")
    try:
        head = json.loads(bytes.fromhex(public).split(b"\n", 1)[0].decode())
    except ValueError as e:
        raise D.Reject(f"the public file's header is not JSON: {e}") from None
    units = head.get("units") if isinstance(head, dict) else None
    indices = units.get("indices") if isinstance(units, dict) else None
    if not (isinstance(indices, list) and all(map(_nat, indices))):
        raise D.Reject("the public file's units.indices is not a list of natural numbers")
    return indices


def file_population(j):
    """`FilePart.why` (`proofs.population_why`): the registration's strata over the program and query
    (`partition.strata`), against a public file's units (`units.indices`): given, or read off the public file."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    desc = _obj(j.get("program"), "program")
    if j.get("public") is not None:
        indices = _public_units(j["public"])
    else:
        indices = j.get("indices")
        if not (isinstance(indices, list) and all(map(_nat, indices))):
            raise D.Reject("indices is not a list of natural numbers")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    try:
        strata = P.strata(PROGRAMS[sha], _obj(j.get("query"), "query"))
    except (P.QueryRefused, ValueError) as e:
        raise D.Reject(str(e)) from None
    return {"why": PF.population_why(indices, strata)}


def tables_held(j):
    """`OneStageZk.tablesPiece` (`proofs.tables_why`): v2's refusal of blobs under `digests` (even-length lowercase hex)
    whose pinned MUFU tables are not the lookups of a circuit whose META is ``{"lookups": lookups}`` (`proofs.lookups_of`;
    ``{}`` without ``lookups``)."""
    j = _json(j)
    ds = j.get("digests") if isinstance(j, dict) else None
    if not isinstance(ds, list):
        raise D.Reject("digests is not a list")
    if not all(isinstance(d, str) for d in ds):
        raise D.Reject("digests is not a list of hex strings")
    if not all(len(d) % 2 == 0 and HEX_DIGITS.fullmatch(d) for d in ds):
        raise D.Reject("not lowercase hex")
    try:
        lookups = PF.lookups_of({"lookups": j["lookups"]} if "lookups" in j else {})
    except ValueError as e:
        return {"why": str(e)}
    return {"why": PF.tables_why(lookups, ds)}


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


def bernoulli_bound(j):
    j = _json(j)
    _exactly(j, {"delta", "p", "population"}, "the input")
    if not _nat(j["population"]):
        raise D.Reject("population is not a natural number")
    p, delta = ratio(j["p"]), ratio(j["delta"])
    if not 0 <= p <= 1 or not 0 < delta < 1:
        raise D.Reject("p is not in [0, 1] or delta not in (0, 1)")
    return {"bound": Bernoulli(p).count_bound(j["population"], delta)}


def _hidden_width(j) -> int:
    if not _nat(j["width"]) or j["width"] < 1:
        raise D.Reject("width is not a positive integer")
    return j["width"]


def _hidden_nats(x, what: str) -> list[int]:
    if not isinstance(x, list) or not all(_nat(n) for n in x):
        raise D.Reject(f"{what} is not a list of natural numbers")
    return x


def _hidden_calls(x) -> list[HD.Call]:
    if not isinstance(x, list) or not all(isinstance(c, list) and len(c) == 2 and all(_nat(n) for n in c) for c in x):
        raise D.Reject("calls is not a list of [offset, rows] of natural numbers")
    return [HD.Call(*c) for c in x]


def _hidden_witness(w):
    return None if w is None else list(w)


def hidden_offsets(j):
    j = _json(j)
    _exactly(j, {"rows", "width"}, "the input")
    width, rows = _hidden_width(j), _hidden_nats(j["rows"], "rows")
    return {"offsets": HD.offsets(width, rows), "total": HD.total(width, rows)}


def hidden_locate(j):
    j = _json(j)
    _exactly(j, {"calls", "positions", "width"}, "the input")
    width, calls, positions = _hidden_width(j), _hidden_calls(j["calls"]), _hidden_nats(j["positions"], "positions")
    return {"located": [_hidden_witness(HD.locate(width, calls, r)) for r in positions]}


def hidden_witness(j):
    j = _json(j)
    _exactly(j, {"call", "calls", "position", "width"}, "the input")
    width, calls = _hidden_width(j), _hidden_calls(j["calls"])
    if not _nat(j["call"]) or not _nat(j["position"]):
        raise D.Reject("call or position is not a natural number")
    return {"witness": _hidden_witness(HD.witness(width, calls, j["call"], j["position"]))}


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
    """The proof system's hold on an outcome's verdicts (`service.check_verdicts`) on `{"verdicts", "sessions",
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
    """`RecordType.inputOf`'s domain, where `audit_record` reads its inputs as the Lean does, `own_reads` a boolean
    when given; δ."""
    ints = lambda x: isinstance(x, list) and all(map(_int, x))  # noqa: E731
    _exactly(j, RECORD_INPUT_KEYS | ({"own_reads"} if isinstance(j, dict) and "own_reads" in j else set()),
             "the record's input")
    if not isinstance(j.get("own_reads", False), bool):
        raise D.Reject("own_reads is not a boolean")
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
    """`audit_record` on the record piece's input, δ the float it names (`Fraction(delta)` is one), and the registered
    roots the verdicts' own for the registration's values, with `own_reads` less those of its own ports at their own
    roots (`Record.lessOwnReads`)."""
    delta = _record_input(j)
    opened = audit.session_registered(j["verdicts"], j["registration"])
    if j.get("own_reads") and opened is not None:
        roots = {r["port"]: r["root"] for r in j["registration"]["roots"]}
        opened = {v: r for v, r in opened.items() if not (v in roots and SS._jeq(roots[v], r))}
    return audit.audit_record(j["registration"], j["receipt"], j["draw"], j["law"], j["proved_units"], j["instances"],
                              j["statement"], j["verdicts"], float(delta), j["sessions"], j["served"], j["session_roots"],
                              opened)


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
    order. `register`: `check_public` on the values and declared items, the population, R1-R7 under
    `service.selection_law` over it (a rate law's count of none or more than N: `rate`), a bernoulli sampling design at
    any p (`service.bernoulli`: `bernoulli`), a program whose units read stratum by stratum are not its N in order
    (`partition.in_order`: `population-order`), and the receipt; `select`: a closure map refused (`closure`), then the
    sample from the source's bytes (`select`), the challenger's coins with their evidence when it read any (`_coins_hold`:
    code `coins`; a `coin_source` other than the challenger is refused up front: the verifier samples no coins of its
    own); `deliver`: one or more sessions, each matching
    `service._DIGEST`; `outcome`: `check_verdicts`, then the audit record (`record`) over the sample's proved units unless
    given, its acceptance the symbol and the first verdict of record's public outputs shown on an accept. A step that refuses gives `{"step",
    "codes"}`; inputs the Lean decodes before the run (`Pieces.run`) are refused up front. Under a two-stage rule the
    registration's query is its grain (`grain`), `select` is the first stage, then `interiors` and `second`
    (`_interior_steps`) before `deliver`, and the interior registration's audit record over the second stage's sample,
    with round 2's statement check of the interior's `statements` (`second_stage.check` under `second_stage.FORM`), is
    under `interior`, the symbol accepting when both do."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    p = _params_of(j)
    rule = _rule(j.get("rule"))
    vs, ps = _values_of(j.get("values")), j.get("public")
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
    if j.get("coin_source", "challenger") not in (None, "challenger"):
        raise D.Reject("the verifier draws no coins of its own")
    it = None if j.get("interior") is None else _interior_of(j["interior"])
    if isinstance(rule, service.TwoStage) and D.canonical(j["params"]["query"]) != D.canonical(rule.grain):
        return {"codes": ["grain"], "step": "register"}
    rec = j.get("record")
    pub = check_public({"public": ps, "query": j["params"]["query"], "record": rec, "scheme": p["scheme"], "values": vs})
    if pub["codes"]:
        return {"codes": pub["codes"], "step": "register"}
    n = rec.get("population") if isinstance(rec, dict) else None
    if not (_int(n) and n >= 1):
        return {"codes": ["population"], "step": "register"}
    try:
        law = service.selection_law(rule, n, service.selection_sizes(rule, p["program"], p["query"]))
        table = service.selection_table(rule, p["program"], p["query"], p["work_table"])
    except service.Refused as e:
        return {"codes": list(e.verdict.codes), "step": "register"}
    except ValueError as e:
        raise D.Reject(str(e)) from None
    codes = _registration({**p, "law": law, "work_table": table}, rec)
    if codes:
        return {"codes": list(codes), "step": "register"}
    if service.bernoulli(law):
        return {"codes": ["bernoulli"], "step": "register"}
    if not P.in_order(p["program"], p["query"], n):
        return {"codes": ["population-order"], "step": "register"}
    receipt = _receipt(rec, j["challenger"], j["received"])
    try:
        derived = DR.derive(p["program"], p["query"], law, table)
    except (P.QueryRefused, ValueError) as e:
        raise D.Reject(str(e)) from None
    if closure is not None:
        return {"codes": ["closure"], "step": "select"}
    drawn = select({"closure": closure, "program": j["params"]["program"], "query": j["params"]["query"], "rule": j["rule"],
                    "source": j["source"], "work_table": p["work_table"]})
    if not _coins_hold(j, rec, PR.unhex(j["source"])[:drawn["read"]]):
        return {"codes": ["coins"], "step": "select"}
    codes = DR.check_draw(drawn["draw"], rec, derived).codes
    if codes:
        return {"codes": list(codes), "step": "select"}
    inner = None
    if isinstance(rule, service.TwoStage):
        if it is None:
            return {"codes": ["interior"], "step": "interiors"}
        inner = _interior_steps(j, it, rule.k, DR.proved(drawn["draw"]))
        if "step" in inner:
            return inner
    elif it is not None:
        raise D.Reject("a one-stage run has no interior registration")
    if not sessions or not all(isinstance(s, str) and service._DIGEST.match(s) for s in sessions):
        return {"codes": ["sessions"], "step": "deliver"}
    held = check_verdicts({"outputs": outputs, "sessions": sessions, "verdicts": verdicts})
    if held["codes"]:
        return {"codes": held["codes"], "step": "outcome"}
    of_record = sorted(k for k in verdicts if verdicts[k].get("of_record"))
    shown = dict(verdicts[of_record[0]]["public"]) if outputs else {}
    proved = proved if proved is not None else DR.proved(drawn["draw"])
    instances = len(proved) if instances is None else instances
    out = audit_record_of({"delta": j.get("delta"), "draw": drawn["draw"], "instances": instances, "law": derived,
                           "proved_units": proved, "receipt": receipt, "registration": rec, "served": j.get("served"),
                           "session_roots": j.get("session_roots"), "sessions": sessions, "statement": j.get("statement"),
                           "verdicts": verdicts, "own_reads": inner is not None})
    accepted = out["accepted"] is True
    staged = {}
    if inner is not None:
        proved2 = it["proved_units"] if it["proved_units"] is not None else DR.proved(inner["draw"])
        out2 = audit_record_of({
            "delta": j.get("delta"), "draw": inner["draw"], "law": inner["law"], "proved_units": proved2,
            "instances": len(proved2) if it["instances"] is None else it["instances"], "receipt": inner["receipt"],
            "registration": it["record"], "served": it["served"], "session_roots": it["session_roots"],
            "sessions": sessions, "statement": j.get("statement"), "verdicts": verdicts, "own_reads": True})
        try:
            strata = P.strata(it["decoded"]["program"], it["decoded"]["query"])
        except (P.QueryRefused, ValueError) as e:
            raise D.Reject(str(e)) from None
        sample = DR.proved(drawn["draw"])
        links = SS.replay_links(p["program"], it["replay"]["program_fn"], rec)
        own = SS.Own(tuple(sample), SS.parts(len(sample), strata), it["record"], rec, tuple(DR.proved(inner["draw"])),
                     links, audit.session_registered(verdicts), SS.verified_of(verdicts), it["refs"], it["ref_reads"],
                     SS.replay_wires(p["program"], it["replay"]["program_fn"]),
                     SS.replay_rows(p["program"], it["replay"]["program_fn"]))
        out2 = SS.attach(out2, SS.check(SS.FORM, own, it["statements"]), SS.interior_reads(own, it["statements"] or ()))
        accepted = accepted and out2["accepted"] is True
        staged = {"interior": {"audit": codec.canonical_json(out2).decode(), "draw": inner["draw"], "read": inner["read"],
                               "receipt": inner["receipt"]}}
    return {"audit": codec.canonical_json(out).decode(), "draw": drawn["draw"],
            "public": shown if accepted else {}, "read": drawn["read"], "receipt": receipt,
            "symbol": "accept" if accepted else "reject", **staged}


def _interior_of(it) -> dict:
    """`Pieces.run`'s `interior`, decoded where the Lean decodes it: its params, the auditor's among them (`proofs`,
    `service.template_circuits`, whose reference headers and registered reads round 2 reads), one replay block's program
    object, its values, received, source, proved units, instances and statements, and refused where it names links,
    wires or rows; the rest as given."""
    if not (isinstance(it, dict) and isinstance(it.get("replay"), dict)):
        raise D.Reject("an interior's replay is a program object")
    if not isinstance(it.get("received"), str):
        raise D.Reject("an interior's received is a string")
    proved, instances = it.get("proved_units"), it.get("instances")
    if not (proved is None or isinstance(proved, list) and all(map(_int, proved))):
        raise D.Reject("an interior's proved_units is not a list of integers")
    if not (it.get("statements") is None or isinstance(it["statements"], list)):
        raise D.Reject("an interior's statements is a list")
    params = it.get("params")
    if not isinstance(params, dict):
        raise D.Reject("an interior's params are an object")
    decoded = decode_params({k: v for k, v in params.items() if k != "proofs"})
    try:
        named = service.template_circuits(params.get("proofs"), params["query"])
    except ValueError as e:
        raise D.Reject(str(e)) from None
    if not it["replay"]:
        raise D.Reject("an interior's replay is not a program descriptor")
    sha = PO.program_sha512(it["replay"])
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a replay block this file has no program for: {sha[:16]}")
    values = _values_of(it.get("values"))
    if not isinstance(it.get("source"), str):
        raise D.Reject("an interior's source is not hex")
    PR.unhex(it["source"])
    if not (instances is None or _int(instances)):
        raise D.Reject("an interior's instances is not an integer")
    if "links" in it:
        raise D.Reject("an interior names no links: the verifier derives them from the replay block's Definition")
    if "wires" in it:
        raise D.Reject("an interior names no wires: the verifier derives them from the replay block's Definition")
    if "rows" in it:
        raise D.Reject("an interior names no rows: the verifier derives them from the replay block's Definition")
    return {**it, "decoded": decoded, "replay_program": PROGRAMS[sha], "values": values, "proved_units": proved,
            "instances": instances, "served": it.get("served"), "session_roots": it.get("session_roots"),
            "record": it.get("record"), "coins": it.get("coins"), "statements": it.get("statements"),
            "refs": tuple((t, h) for t, _c, h, _r in named), "ref_reads": tuple((t, r) for t, _c, _h, r in named)}


def _interior_steps(j: dict, it: dict, k, sample: list[int]) -> dict:
    """`Run.interiorSteps`: the interior registration over the replay sample's proof blocks, in the boundary
    registration's window or the one the verifier computes (`Run.interiorWindowOk`), then the second stage on its own
    coins, the rule's counts `k` over the replay block's cells; its receipt, sample, law and bytes read, or the step's
    refusal."""
    p, rec = it["decoded"], it["record"]
    pub = check_public({"public": j["public"], "query": it["params"]["query"], "record": rec, "scheme": p["scheme"],
                        "values": it["values"]})
    if pub["codes"]:
        return {"codes": pub["codes"], "step": "interiors"}
    try:
        ns = service.replay_cells(it["replay_program"], p["query"])
        params = p["query"]["params"]
        ks = service.second_counts(k, params["templates"] if "templates" in params else [params["template"]], ns)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    m = len(sample) * sum(ns)
    population = rec.get("population") if isinstance(rec, dict) else None
    if not (_nat(population) and population == m):
        return {"codes": ["interior-population"], "step": "interiors"}
    window, boundary = it["params"].get("window_root"), j["params"].get("window_root")
    if window != boundary and (window is None or window != SS.interior_root(R.digest(j["record"]), sample)):
        return {"codes": ["interior-window"], "step": "interiors"}
    codes = _registration({**p, "law": {"law": "subset", "k": m}}, rec)
    if codes:
        return {"codes": list(codes), "step": "interiors"}
    if not P.in_order(p["program"], p["query"], m):
        return {"codes": ["population-order"], "step": "interiors"}
    receipt = _receipt(rec, j["challenger"], it["received"])
    src, at = PR.unhex(it["source"]), 0

    def coins(c: int) -> bytes:
        nonlocal at
        at += c
        return src[at - c:at]
    try:
        draw2 = service.select_second(sample, ns, ks, coins)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    if not _coins_hold(it, rec, src[:at]):
        return {"codes": ["coins"], "step": "second"}
    return {"draw": draw2, "law": service.second_law(sample, ns, ks), "read": at, "receipt": receipt}


def _coins_hold(j: dict, rec: dict, read: bytes) -> bool:
    """`Run.coinsHold`: a sample that read no coins carries no evidence; one that read coins read the challenger's and
    carries evidence that holds (`coins_reason`) for the verifier's window root, after the registration's digest."""
    if not read:
        return j.get("coins") is None
    window = j["params"].get("window_root")
    return window is not None and coins_reason(j.get("coins"), window, bytes.fromhex(R.digest(rec)), read,
                                               lms_verify) is None


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
def agreed_laws(draw, n: int):
    """One-stage's agreed law: mostly a rate as a selection rule states it (`OneStage.design`), a rate law's tables a divisor
    of `n` so its count fits, else a law object (`laws`)."""
    kind = draw(st.sampled_from(["rate", "stratified-rate", "work-rate", "rate", "stratified-rate", "work-rate", "object"]))
    if kind == "object":
        return draw(laws(n))
    den = draw(st.integers(1, 6))
    tables = draw(st.sampled_from([None, *(t for t in (1, 2, 3) if n % t == 0)])) if kind == "rate" else None
    return f"{kind}:{draw(st.integers(1, den))}/{den}" + ("" if tables is None else f":{tables}")


def _worked(law) -> bool:
    """Whether an agreed law is expanded over a work table: `work` or a work rate."""
    return (law.get("law") == "work") if isinstance(law, dict) else str(law).startswith("work-rate:")


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
    """The verifier's sample under its expanded law; a work sample's closure, which the audit refuses, one time in four."""
    kind = law["law"]
    if kind == "subset":
        return {"k": law["k"], "law": "subset", "population": n, "units": sorted(draw(st.permutations(range(n)))[:law["k"]])}
    if kind == "bernoulli":
        return {"law": "bernoulli", "p": law["p"], "population": n,
                "units": sorted(draw(st.sets(st.integers(0, n - 1), max_size=n)))}
    units = sorted(u for s in law["strata"] for u in draw(st.permutations([u for lo, hi in s["units"] for u in range(lo, hi)]))[:s["k"]])
    d = {"k": law["k"], "law": kind, "population": n, "strata": law["strata"], "units": units}
    rest = [u for u in range(n) if u not in units]
    if kind == "work" and rest and draw(st.integers(0, 3)) == 0:
        d["closure"] = sorted(draw(st.sets(st.sampled_from(rest), min_size=1)))
    return d


def _reads_window(supplied: dict) -> bool:
    """Whether a supplied value reads the window's salt, which no verifier without a binding fills."""
    return any(k == "window" for rows in G.sources(supplied).values() for row in rows for k, _ in row)


def _record_of(s: Setup, values, law: dict, window: dict, log=(), binding=None, output=None, window_root=None, fixed=(),
               **kw) -> dict:
    """`R.record` over `ROOTS` registering `values`: `s`'s commitments (W's under `output`'s root and domain when it is
    an earlier record's output, the rest under `window_root`'s domain but `fixed`'s), and each value the verifier fills
    itself as it fills it from that record, `log` and its `binding` (`registered.supplied_value`)."""
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    sup = s.supplied(values) if isinstance(s, RegisteredSetup) else {}
    entries = s.entries({v: w for v, w in values.items() if v not in sup}, window_root, fixed) if values else []
    entries = [{**e, **output} if output and e["value"] == "W" else e for e in entries]
    rec = R.record(s.program, s.query, law, SCHEME, roots, window, registered=entries, **kw)
    if not sup:
        return rec
    filled = [G.supplied_value(s.program, v, rows, rec, sorted(sup), list(log), binding=binding,
                               schema=s.formats.get(v, G.SCHEMA)).entry()
              for v, rows in G.sources(sup).items()]
    return R.record(s.program, s.query, law, SCHEME, roots, window, registered=entries + filled, **kw)


def reads(rec: dict) -> dict:
    """A Lean verdict's `registered`: per port of the statement, the value it reads and the root it opened against (the
    record's)."""
    return {f"{e['value']}.{i}": {"root": e["root"], "value": e["value"]} for e in rec["registered"] for i in range(2)}


@st.composite
def audits(draw, setups=(*SETUPS, *REGISTERED_SETUPS)) -> tuple[dict, dict, dict]:
    """A valid audit: the verifier's parameters, the registration record and the evidence (verdicts mostly accepting). On a
    program with values to register it is mostly v2: the verifier expects some registered, the record registers them, the
    log's earlier record (if any) some of them under the same roots, and the verdict of record reports the sessions'
    reads opened against those. Half the v2 verifiers name the window's open root (`WROOT`) and hold some values across
    windows: the record registers the others it neither fills, pins nor takes as an output under `WROOT`'s domain, and
    the log's record under an earlier window's, the same window's or none. Half of those read the window's seed salt
    (`SEED_SALT`), which the record's window names."""
    s = draw(st.sampled_from(setups))
    values = draw(st.sampled_from([*s.values, None, *s.values, {}])) if isinstance(s, RegisteredSetup) else None
    held = draw(st.sampled_from([values, dict(list(values.items())[:1]), None])) if values else None
    sup = s.supplied(values) if isinstance(s, RegisteredSetup) else {}
    windowed = bool(values) and draw(st.booleans())
    keep = draw(st.sampled_from([[], [v for v in values if v not in sup], [v for v in values if v == "W"]])) if windowed else []
    log_window = draw(st.sampled_from([EARLIER_WROOT, WROOT, None])) if windowed else None
    n = s.population
    law = draw(agreed_laws(n))
    table = draw(work_tables(s)) if _worked(law) else None
    try:
        named, expanded = service.params_law(law, s.program, s.query, table)
        mine = DR.derive(s.program, s.query, named, expanded)
    except ValueError:
        table = {t: 1 for t in s.templates}
        named, expanded = service.params_law(law, s.program, s.query, table)
        mine = DR.derive(s.program, s.query, named, expanded)
    pre = draw(st.booleans())
    part = {"program": s.sha512, "query": s.query} if pre else None
    log = []
    output = None
    earlier = draw(st.booleans())
    if earlier and held and not windowed:
        # without the window's root, a value the log registered is one the verifier holds across windows
        keep = [v for v in held if v not in sup]
    if earlier and values and "W" in values and "W" not in sup and draw(st.booleans()):
        # W is an output of the earlier run: its port, of W's rows, under a root and domain of its own
        output = {"root": "c1" * 64, "domain": EARLIER_OUT_DOMAIN}
    reuse = bool(output) and draw(st.booleans())
    pin = not output and bool(values) and "W" in values and "W" not in sup and draw(st.booleans())
    fixed = [*keep, *(["W"] if pin or output else []), *sup]
    if earlier:
        # the log's commitment of a supplied value is any other: it is the record's own, never held
        roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
        given = list(s.entries(held, log_window, fixed)) if held else []
        if reuse:
            # W reuses the earlier record's registered value: an entry of W's rows, words and word bits, under that root
            given.append({**s.committed["W", values["W"]].entry(), "value": EARLIER_VALUE, **output})
        elif output:
            roots.append({"port": EARLIER_OUT, "leaves": s.committed["W", values["W"]].entry()["leaves"], **output})
        log = [R.record(s.program, s.query, named, SCHEME, roots, EARLIER, leaf_layer=LEAF, partition=part,
                        work_table=expanded, registered=given)]
    anchors = draw(st.sampled_from([[], ["weights"]]))
    binding = BINDING if _reads_window(sup) or draw(st.booleans()) else None
    params = {"anchors": anchors, "delta": draw(st.sampled_from(["1/1000", "1/10", "1/2", "999/1000"])), "law": law,
              "leaf_layer": LEAF, "log": log, "partition": part, "ports": s.ports, "program": s.desc, "query": s.query,
              "registered": values, "scheme": SCHEME, "work_table": table}
    if values is not None:
        params["parameters"] = s.signature
    if isinstance(s, RegisteredSetup) and s.supplied(values):
        params["supplied"] = s.supplied(values)
    if output:
        kind = {"value": EARLIER_VALUE} if reuse else {"port": EARLIER_OUT}
        params["outputs"] = {"W": kind if not reuse and draw(st.booleans()) else {**kind, "record": R.digest(log[-1])}}
    if values and s.formats and draw(st.integers(0, 3)):
        # without the formats, R7 counts a packed value's rows as row/v1 rows and refuses it
        params["row_schemas"] = {v: f for v, f in s.formats.items() if v in values}
    if binding:
        params["binding"] = binding
    if windowed:
        params["window_root"] = WROOT
        if keep or draw(st.booleans()):
            params["held"] = keep
    elif keep:
        params["held"] = keep
    d = drawn(draw, mine, n)
    proved = DR.proved(d)
    verdicts = {"lean": {"accepted": True, "digest": "aa" * 64, "of_record": True}}
    none_registered = not values and draw(st.booleans())
    if draw(st.booleans()):
        verdicts["upstream"] = {"accepted": draw(st.sampled_from([True, None])), "of_record": False}
    served, session_roots = draw(st.sampled_from([None, list(d["units"])])), draw(st.sampled_from([None, dict(ROOTS)]))
    seeded = windowed and draw(st.booleans())
    window = {**WINDOW, **(binding or {}), **({R.SEED_SALT: SEED_SALT} if seeded else {})}
    rec = _record_of(s, values, named, window, log, binding, output=output,
                     window_root=WROOT if windowed else None, fixed=fixed, leaf_layer=LEAF, anchors=anchors,
                     previous=R.digest(log[-1]) if log else None, partition=part, work_table=expanded)
    assert rec["law"] == mine
    if pin:
        params["pinned"] = {"W": next(e["root"] for e in rec["registered"] if e["value"] == "W")}
    if seeded:
        params["seed_salt"] = SEED_SALT
    if values:
        verdicts["lean"]["registered"] = reads(rec)
    elif none_registered:
        verdicts["lean"]["registered"] = {}
    evidence = {"draw": d, "instances": len(proved), "proved_units": proved,
                "receipt": R.receipt(rec, "verifier-test", received=RECEIVED), "served": served,
                "session_roots": session_roots, "sessions": ["ab" * 64], "statement": {"instances": len(proved)},
                "verdicts": verdicts}
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


#: One change each to the record (R1-R6; R7's are `ENTRIES`), the parameters (decodable or not) and the evidence (the sample, the record's checks,
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
                                        {"law": "bernoulli", "p": "1/2"}, "rate:1/2", "rate:1/2:7", "stratified-rate:1/3",
                                        "work-rate:1/2", "work-rate:0/2", "bernoulli:1/2", "stratified-rate:2/1"][x % 20]),
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
DRAW_CHANGES = DRAW  # `DRAW` is the sample piece's port below
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


def _earlier_out(a: dict, x: int) -> None:
    """The earlier record's output port changed (its leaves, domain or root of another form or value), with `outputs`
    naming the changed record, so R7 reads the port itself."""
    a["params"]["log"] = [{**r, "roots": [
        {**e, ["leaves", "domain", "root"][x % 3]: [1, 1000, True, None, "x", -1][x // 3 % 6]} if e["port"] == EARLIER_OUT
        else e for e in r["roots"]]} for r in a["params"]["log"]]
    _repoint(a)


def _earlier_value(a: dict, x: int) -> None:
    """The earlier record's registered value `EARLIER_VALUE` changed (its leaves, words, word bits, domain, root or name,
    to another form or value, or its word bits left out), with `outputs` naming the changed record, so R7 reads the entry
    itself."""
    def change(e):
        if not (isinstance(e, dict) and e.get("value") == EARLIER_VALUE):
            return e
        if x % 7 == 6:
            return {k: v for k, v in e.items() if k != G.WIDTH_KEY}
        return {**e, ["leaves", "words", G.WIDTH_KEY, "domain", "root", "value"][x % 7]:
                [1, 1000, True, None, "x", -1, 16, 32, "W"][x // 7 % 9]}
    a["params"]["log"] = [{**r, "registered": [change(e) for e in r["registered"]]}
                          if isinstance(r.get("registered"), list) else r for r in a["params"]["log"]]
    _repoint(a)


def _repoint(a: dict) -> None:
    """`outputs`, if any, naming the log's last record in the form it had (a port, else a registered value; a port alone
    names the last record already)."""
    if "outputs" in a["params"]:
        o = a["params"]["outputs"]
        kind = {"value": EARLIER_VALUE} if isinstance(o, dict) and "value" in (o.get("W") or {}) else {"port": EARLIER_OUT}
        last = isinstance(o, dict) and isinstance(o.get("W"), dict) and set(o["W"]) == {"port"}
        a["params"]["outputs"] = {"W": kind if last else {**kind, "record": _earlier(a)}}


def _earlier(a: dict) -> str:
    """The digest of the log's last record, or of none."""
    return R.digest(a["params"]["log"][-1]) if a["params"]["log"] else "ee" * 64


def _registered(value):
    """The verifier's registered values changed, its held values the ones still among them (a held value it doesn't expect
    registered is a malformed parameter, which `registration._window` refuses whatever else changed)."""
    def change(a, x):
        r = a["params"]["registered"] = value(a, x)
        if "held" in a["params"]:
            a["params"]["held"] = [v for v in a["params"]["held"] if isinstance(r, dict) and v in r]
    return change


REGISTERED_PARAMS = [
    _registered(lambda a, x: [{"W": 4}, {"W": 0}, {"E": 1}, {"M": 1}, {"V": 1}, {"nope": 1}, {"T": 3},
                              {"U": 2}, {"W": -1}, {"W": True}, {"W": "2"}, [], "W", {}, {"T": 2}][x % 15]),
    _registered(lambda a, x: {**a["params"]["registered"],
                              **[{"D": 1}, {"x": 1}, {"T": 4}, {"U": 3}, {"W": 6}, {"x": 3}][x % 6]}),
    lambda a, x: (a["params"].pop("parameters"), a["params"].update(registered=None), a["params"].pop("held", None)),
    lambda a, x: a["params"].pop("parameters"),
    _set("params", "parameters", lambda a, x: _signature(a["params"]["parameters"], x)),
    _set("params", "log", lambda a, x: [{**a["record"], "registered": [
        "x", [5], [{"value": 5, "root": "r"}], [{"value": "W"}], {"W": "r"}, 1, [], 0, "", None, [{"root": 1, "value": "W"}],
        [{"root": "r", "value": "W", "x": 1}], [{"domain": "d", "root": "r", "value": "W", "x": 1}],
        [{"domain": 5, "root": "r", "value": "W"}]][x % 14]}]),
    _held,
    _set("params", "outputs", lambda a, x: [
        {"W": {"port": EARLIER_OUT, "record": "ee" * 64}}, {"W": {"port": "nope", "record": _earlier(a)}},
        {"W": {"port": "x", "record": _earlier(a)}}, {"W": {"port": EARLIER_OUT, "record": R.digest(a["record"])}},
        {"W": {"port": EARLIER_OUT, "record": _earlier(a)}}, {"S": {"port": EARLIER_OUT, "record": _earlier(a)}},
        {"nope": {"port": EARLIER_OUT, "record": _earlier(a)}}, {"W": {"port": EARLIER_OUT, "record": "EE" * 64}},
        {"W": {"port": EARLIER_OUT}}, {"W": {"port": 1, "record": _earlier(a)}},
        {"W": {"port": EARLIER_OUT, "record": _earlier(a), "x": 1}}, {"W": EARLIER_OUT}, ["W"], None, {},
        {"W": {"value": EARLIER_VALUE, "record": "ee" * 64}}, {"W": {"value": "nope", "record": _earlier(a)}},
        {"W": {"value": "W", "record": _earlier(a)}}, {"W": {"value": EARLIER_VALUE, "record": _earlier(a)}},
        {"W": {"value": EARLIER_VALUE, "record": R.digest(a["record"])}}, {"W": {"value": 1, "record": _earlier(a)}},
        {"W": {"value": EARLIER_VALUE, "port": EARLIER_OUT, "record": _earlier(a)}}, {"W": {"value": EARLIER_VALUE}},
        {"S": {"value": EARLIER_VALUE, "record": _earlier(a)}}, {"W": {"port": "nope"}}, {"W": {"port": 1}},
        {"S": {"port": EARLIER_OUT}}, {"W": {"port": "x"}}][x % 28]),
    lambda a, x: a["params"].pop("outputs"),
    _earlier_out,
    _earlier_value,
    _set("params", "row_schemas", lambda a, x: [
        {"A": G.SCHEMA}, {"A": G.SCHEMA_SEG}, {"B": G.SCHEMA_V2}, {"W": G.SCHEMA_V2}, {"T": G.SCHEMA_SEG}, {"x": G.SCHEMA_V2},
        {"S": G.SCHEMA_V2}, {"nope": G.SCHEMA_V2}, {"A": "u16"}, {"A": "hm96-sha512/row/v2 "}, {"A": 1}, ["A"], "A", None,
        {}][x % 15]),
    lambda a, x: a["params"].pop("row_schemas", None),
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
def _supplied(f):
    """A change to the values the verifier fills itself (`f(supplied or None, x)`)."""
    def change(a, x):
        a["params"]["supplied"] = f(a["params"].get("supplied"), x)
    return change


#: One change each to the values the verifier fills itself (their form, their sources, their rows, their names, or none)
#: and to the record's roots they read.
SUPPLIED = [
    _supplied(lambda s, x: [None, 5, [], "S", {"S": 5}, {"S": [5]}, {"S": [[5]]}, {"S": [[{}]]}, {"S": [[{"port": "x", "value": "W"}]]},
                            {"S": [[{"root": "x"}]]}, {"S": [[{"port": 1}]]}, {"S": [["x"]]}, {}][x % 13]),
    _supplied(lambda s, x: {**(s or {}), "S": [[{"port": ["y", "x", "out"][x % 3]}, {"value": ["W", "S", "nope", "x"][x // 3 % 4]}],
                                               [{"value": "W"}, {"port": "out"}]]}),
    _supplied(lambda s, x: {**(s or {}), "S": [[{"bytes": [ROOTS["x"], "AB" * 64, "", "abc", ROOTS["x"] + "\n", ROOTS["out"]][x % 6]},
                                                {"value": "W"}], [{"value": "W"}, {"port": "out"}]]}),
    _supplied(lambda s, x: {**(s or {}), ["D", "T", "nope", "x", "W"][x % 5]: [[{"port": "x"}]]}),
    _supplied(lambda s, x: {**(s or {}), "S": [[{"window": [COIN, "C0" * 64, "", "c0c", ROOTS["x"]][x % 5]}]
                                               + [{"position": ["0", "2", "3", "01", "-1", "", "x", "18446744073709551615"][x // 5 % 8]}] * 8] * 2}),
    _set("params", "log", lambda a, x: [{**r, "population": ["3", -1, True, None, 2 ** 40, 0][x % 6]} for r in a["params"]["log"]]),
    _set("params", "pinned", lambda a, x: [{"W": "12" * 64}, {"W": ROOTS["x"]}, {"S": "12" * 64}, {"nope": "12" * 64},
                                           {"W": "AB" * 64}, {"W": "ab"}, {"W": 1}, ["W"], None, {}][x % 10]),
    _supplied(lambda s, x: {k: rows[:-1] or rows + rows for k, rows in s.items()}),
    _supplied(lambda s, x: {k: [r[:-1] for r in rows] for k, rows in s.items()}),
    _supplied(lambda s, x: {k: [r + r[:1] for r in rows] for k, rows in s.items()}),
    _supplied(lambda s, x: {k: rows[::-1] for k, rows in s.items()}),
    lambda a, x: a["params"].pop("supplied"),
    _roots(lambda rs, x: [{**r, "root": "12" * 64} if r["port"] == ["x", "out"][x % 2] else r for r in rs]),
    _entries(lambda es, x: [{**e, "root": "12" * 64} if e["value"] == "W" else e for e in es]),
]


def _bound(f):
    """A change to the verifier's binding (`f(binding or None, x)`)."""
    def change(a, x):
        a["params"]["binding"] = f(a["params"].get("binding"), x)
    return change


def _replayed(a, x):
    """The record as another stream's (or prover's) verifier would see it replayed, relabeled to its binding."""
    k, other = [("stream", "3d" * 64), ("prover", "d1" * 32)][x % 2]
    a["params"]["binding"] = {**(a["params"].get("binding") or BINDING), k: other}
    a["record"]["window"] = {**a["record"]["window"], k: other}


#: One change each to the verifier's binding (another stream's or prover's, in another form, or none) and to the record's
#: window that names it, and the record replayed to another stream or prover.
BINDINGS = [
    _bound(lambda b, x: {**(b or BINDING), **[{"stream": "3d" * 64}, {"prover": "d1" * 32}][x % 2]}),
    _bound(lambda b, x: [None, 5, [], "s", {}, {"stream": "3c" * 64}, {**BINDING, "run": "r"}, {**BINDING, "stream": "3C" * 64},
                         {**BINDING, "stream": "3c" * 63}, {**BINDING, "prover": ""}, {**BINDING, "prover": "d0d"},
                         {**BINDING, "prover": 1}, {**BINDING, "stream": None}][x % 13]),
    lambda a, x: a["params"].pop("binding"),
    _set("record", "window", lambda a, x: {**a["record"]["window"], ["stream", "prover"][x % 2]:
                                           [BINDING["stream"].upper(), "d1" * 32, None, 5][x // 2 % 4]}),
    _set("record", "window", lambda a, x: {k: v for k, v in a["record"]["window"].items() if k != ["stream", "prover"][x % 2]}),
    _replayed,
]


def _replayed_window(a: dict, x: int) -> None:
    """The record's `x`-th entry under another window's domain: an earlier window's, none (held across windows) or this
    one's, its root unchanged."""
    es, rec = a["record"]["registered"], a["record"]
    i, window = x % len(es), [EARLIER_WROOT, None, WROOT][x // len(es) % 3]
    e = es[i]
    dom = G.domain(rec["program"], e["value"], e["words"], e["leaves"], e.get("word_bits", 16), window).domain_id.hex()
    rec["registered"] = [{**e, "domain": dom} if j == i else f for j, f in enumerate(es)]


def _same_window_log(a: dict, x: int) -> None:
    """The log's last record registered the record's values under the record's own domains (the same window), at other
    roots, so R7 holds them to those."""
    last = a["params"]["log"][-1]
    a["params"]["log"] = [*a["params"]["log"][:-1], {**last, "registered": [
        {**e, "root": "12" * 64} for e in a["record"]["registered"]][:1 + x % len(a["record"]["registered"])]}]
    a["record"]["previous"] = R.digest(a["params"]["log"][-1])


#: One change each to the window's open root the verifier names (another window's, in another form, or none), the values
#: it holds across windows, and the record's entries replayed from another window (an earlier window's domain or none).
WINDOWS = [
    _set("params", "window_root", lambda a, x: [EARLIER_WROOT, WROOT.upper(), WROOT[:-2], None, 5, "", WROOT + "\n",
                                                WROOT][x % 8]),
    lambda a, x: a["params"].pop("window_root"),
    _set("params", "held", lambda a, x: [[], ["nope"], "W", [5], None, sorted(a["params"]["registered"] or {}), ["W"],
                                         ["x"], {"W": 1}][x % 9]),
    lambda a, x: a["params"].pop("held"),
    _replayed_window,
    _same_window_log,
]


def _server_seeded(a: dict, x: int) -> None:
    """The record's window names a seed salt the verifier read none of: a salt the server chose."""
    a["record"]["window"] = {**a["record"]["window"], R.SEED_SALT: [SEED_SALT, OTHER_SEED_SALT][x % 2]}
    a["params"].pop("seed_salt", None)


#: One change each to the seed salt the verifier read from its challenger (another, in another form, or none), to the
#: record's window that names it (another, in another form, or none), and a salt the server chose.
SEED_SALTS = [
    _set("params", "seed_salt", lambda a, x: [OTHER_SEED_SALT, SEED_SALT.upper(), SEED_SALT[:-2], None, 5, "",
                                              SEED_SALT + "\n", SEED_SALT][x % 8]),
    lambda a, x: a["params"].pop("seed_salt"),
    _set("record", "window", lambda a, x: {**a["record"]["window"],
                                           R.SEED_SALT: [OTHER_SEED_SALT, SEED_SALT.upper(), None, 5][x % 4]}),
    _set("record", "window", lambda a, x: {k: v for k, v in a["record"]["window"].items() if k != R.SEED_SALT}),
    _server_seeded,
]


def _tied(a: dict, x: int) -> None:
    """The verifier's tied values, with the program's parameters `tied` names: none, its first parameter, every one, one
    it expects registered, one that isn't a parameter, a parameter twice, or another form."""
    p = a["params"]
    sig = p.get("parameters") or _json([[name, t.to_json()] for name, t in
                                         PROGRAMS[PO.program_sha512(p["program"])].fn.signature()[0]])
    names = [n for n, _ in sig]
    p["parameters"] = sig
    p["tied"] = [names[:1], names, sorted(p.get("registered") or {})[:1], ["nope"], [*names[:1], "nope"],
                 names[:1] * 2, [], None, "x", [5], {"x": 1}][x % 11]


#: One change each to the run Inputs the verifier names tied (`_tied`), or its program's parameters left out.
TIED = [_tied, _tied, lambda a, x: (_tied(a, x), a["params"].pop("parameters"))]
GROUPS = [RECORD, PARAMS, DRAW, EVIDENCE, VERDICTS_OF, ENTRIES, REGISTERED_PARAMS, REGISTERED_EVIDENCE, SUPPLIED, BINDINGS,
          WINDOWS, SEED_SALTS]


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
    changed_audits([VERDICTS_OF], most=1, least=1).map(_case),
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
    """One accepted audit of each sampling design the audit covers, field by field: the public outputs and the profile Python
    writes."""
    for law in ({"k": 2, "law": "subset"}, {"k": 3, "law": "stratified"}, {"k": 3, "law": "work"}, "rate:1/3",
                "stratified-rate:1/2", "work-rate:1/3"):
        s = SETUPS[3]
        table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}} if _worked(law) else None
        named, expanded = service.params_law(law, s.program, s.query, table)
        mine = DR.derive(s.program, s.query, named, expanded)
        rec = R.record(s.program, s.query, named, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                       WINDOW, leaf_layer=LEAF, work_table=expanded)
        units = [0, 3][:mine["k"]] if mine["law"] == "subset" else sorted(
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
        assert verdict_ == "accept" and json.loads(line) == answer(case) | {"reason": None}, (law, line)


ONE_STAGE_V2 = D.ProtocolPort("sampled-proofs-one-stage-zk", answer, CASES, package=PACKAGE)
server_v2 = D.server_fixture(ONE_STAGE_V2)


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _public_of(rec: dict, roots: dict | None = None, indices: list | None = None) -> bytes:
    """A public file whose header is `rec`'s as `registration.matches_public` reads one (`OneStageZk.publicHeld`): its
    scheme, its ``frame_v3`` roots (`roots` in their place) and their ``domain_ids``, its population as ``instances``,
    and its population's units in order as ``units.indices`` (`indices` in their place). No statement follows it, so a
    session set up on it is rejected."""
    reg = {r["port"]: r for r in rec["roots"]}
    fv = {"domain_ids": {p: r["domain"] for p, r in reg.items()}, "roots": roots or {p: r["root"] for p, r in reg.items()}}
    units = {"indices": list(range(rec["population"])) if indices is None else indices}
    header = {"frame_v3": fv, "instances": rec["population"], "scheme": rec["scheme"], "units": units}
    return json.dumps(header, sort_keys=True).encode() + b"\nnot a public file"


#: `OneStageZk.INDICES_WHY`
INDICES_WHY = ("public file: its units.indices are not the registration's population units in order "
               "(instance i is unit i)")


#: `OneStageZk.NO_CHALLENGER`
NO_CHALLENGER = ("the parameters name no challenger: the draw's coins are the auditor's challenger's signed release "
                 "(challenger, with window_root), never the prover's record")
#: `OneStageZk.POPULATION_ORDER`
POPULATION_ORDER = ("population-order: the program's proof blocks under the query are not the registration's "
                    "population units in order")
#: `OneStageZk.sessionTables`: the tables of the one session v2 reads, its statement's (`proofs.STATEMENT`) one
V2_TABLES = ["circuit"]
#: A stand-in circuit file: v2 reads its META alone, for the lookups it holds the blobs' tables to (`OneStageZk.tablesHeld`)
STAND_IN_CIRCUIT = b"not a circuit file\nMETA {}\n"


#: `OneStageZk.KEY_PARAMS`
KEY_PARAMS = (f"the challenger's key is not at the deployed LMS parameters (W = {DEPLOYED[0]}, H = {DEPLOYED[1]}): v2 "
              "reads the challenger's signatures only under those")


def decode_params_v2(j) -> dict:
    """`OneStageZk.paramsOfJson`'s refusals of v1's part of the parameters: v1's (`covered_params`,
    `OneStage.paramsOfJson`), then parameters naming no challenger (`NO_CHALLENGER`), or one whose key is at other LMS
    parameters than the deployed ones (`KEY_PARAMS`)."""
    rest = {k: v for k, v in j.items() if k != "proofs"}
    p = covered_params(rest)
    if "challenger" not in rest:
        raise D.Reject(NO_CHALLENGER)
    key = ChallengerKey.from_json(rest["challenger"]["key"])
    if (key.W, key.H) != DEPLOYED:
        raise D.Reject(KEY_PARAMS)
    return p


def _challenger_draw(rng: random.Random, s: Setup, law: dict, rec: dict, trees: int = 1,
                     keys: list[str] | None = None, seed: int = 0, order: str = "honest") -> tuple[Drawn, dict, dict]:
    """A sample under `law` from the coins of a challenger session its ledger opens withheld, under the key of `seed` at
    the deployed parameters, released after `rec`'s digest and read as `draw_first` reads them, the session's
    challenger sampling and opening `trees` coin trees too (`TreeSession`) on v2's one table (`OneStageZk.sessionTables`),
    their links opened after `keys` (that table when none), its log in `order` (`ORDERS`): the session, the sample, and the
    evidence's `coins` (the session's record without the key, which the verifier's parameters name)."""
    mine = DR.derive(s.program, s.query, law, None)

    def reads(coins) -> bytes:
        got = bytearray()

        def source(k: int) -> bytes:
            b = coins(k)
            got.extend(b)
            return b
        draw_first(OwnRandomness(source), mine, s.population)
        return bytes(got)
    d = Drawn(rng, before=True, after=bytes.fromhex(R.digest(rec)), reads=reads, trees=trees, tables=V2_TABLES,
              keys=keys, params=[DEPLOYED], seed=seed, order=order)
    at = [0]

    def source(k: int) -> bytes:
        at[0] += k
        return d.read[at[0] - k:at[0]]
    coins = {k: v for k, v in d.evidence(d.segment.to_json()).items() if k != "key"}
    return d, draw_first(OwnRandomness(source), mine, s.population), coins


def test_v2_reads_no_verdict_from_the_evidence(server, server_v2):
    """v2 (`Spot.OneStageZk`, `one_stage.proofs.check_v2`) is v1 with the verdicts its own run of the verifier
    on the proofs. An audit v1 accepts on the verdict its evidence hands in and its challenger's coins is, in v2: refused
    for naming verdicts; refused, before any tree is read, with them replaced by proofs naming no session or two (the
    sampled proof blocks are proved in one session, `oneSession`); rejected by the session's own verdict when its files are not
    a statement's; refused before the session is verified when its coins are not a tree the challenger logged or its
    link was opened after other keys than its one table (`LINK_TABLES`; the `root_B` length a statement fixes,
    `LINK_ROOT_LEN`, these files set up none to hold), or the public file is not the registration's
    (`publicHeld`, its codes Python's `matches_public`'s), or its ``units.indices`` are not the population's units in
    order (`indicesHeld`): a header rewritten after the sample, its pin with it, to credit an honest instance as the sampled
    proof block, or naming none; refused when a file it names is in no blob, as the public file the parameters name is when
    the blobs hold another, or its sessions are not the evidence's. The public file is the parameters', committed before
    any coin: evidence naming one of its own, or parameters naming none, are refused. `test_proofs.py` holds the
    verdicts to `verify`'s on a real session, and `test_challenger_difftest.py` the trees' check to its Python."""
    s = SETUPS[3]
    law = {"k": 2, "law": "subset"}
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF)
    rng = random.Random("v2-verdicts")
    circuit, public = STAND_IN_CIRCUIT, _public_of(rec)

    def audit(trees: int, keys: list[str] | None = None) -> tuple[Drawn, dict, dict, dict]:
        ch, d, coins = _challenger_draw(rng, s, law, rec, trees, keys)
        proved = DR.proved(d)
        evidence = {"coins": coins, "draw": d, "instances": len(proved), "proved_units": proved,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                    "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
        params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/20",
                  "law": law, "leaf_layer": LEAF, "log": [], "partition": P.partition(s.program, s.query),
                  "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
                  "window_root": ch.ledger.window["root"], "work_table": None}
        v1 = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        assert D.replay(ONE_STAGE, v1, server)[0] == "accept"
        base = {k: v for k, v in evidence.items() if k != "verdicts"}
        return ch, evidence, base, params | {"proofs": {"circuit": _sha(circuit), "public": _sha(public),
                                                        "registered": None}}

    ch, evidence, base, p2 = audit(1)
    record, stand_in = D.canonical(ch.trees[0].record).encode(), b"{}"
    blobs = [circuit.hex(), public.hex(), record.hex(), stand_in.hex()]

    def run(e, blobs_=blobs, p=p2):
        case = {"blobs": list(blobs_), "commitment": _hex(rec), "evidence": _hex(e), "params": p}
        out = server_v2.exchange(_json(case))
        return out["verdict"], out["reason"]

    def proofs(sessions, record_=record):
        return {"sessions": [{"name": n, "record": _sha(record_), "reps": {}} for n in sessions]}

    assert run(evidence | {"proofs": proofs([])}) == (
        "reject", "evidence: the evidence names verdicts: this audit's verdicts are its own run of the verifier on the "
                  "proofs")
    assert run(base | {"proofs": proofs([])}) == (
        "reject", "proofs: the evidence names 0 sessions: the drawn units are proved in one session")
    assert run(base | {"proofs": proofs(["s"]), "sessions": ["s"]}) == ("reject", "verdict flock-verify/s did not accept")
    assert run(base | {"proofs": proofs(["s"], stand_in), "sessions": ["s"]}) == (
        "reject", f"proofs: session s's coins are not the challenger's: {NO_TREE}")
    assert run(base | {"proofs": proofs(["s", "t"]), "sessions": ["s", "t"]}) == (
        "reject", "proofs: the evidence names 2 sessions: the drawn units are proved in one session")
    got = run(base | {"proofs": proofs(["s"]), "sessions": ["s"]}, blobs_=blobs[1:])
    assert got[0] == "reject" and got[1].startswith("proofs: circuit: no blob has the SHA-512"), got
    got = run(base | {"proofs": proofs(["s"]), "sessions": ["s"]}, blobs_=blobs[:2])
    assert got[0] == "reject" and got[1].startswith("proofs: session s's record: no blob has the SHA-512"), got
    assert run(base | {"proofs": proofs(["s"]), "sessions": ["t"]}) == (
        "reject", "evidence: proofs: the sessions are not the evidence's sessions, in order")
    got = run(base | {"proofs": proofs([])}, p={k: v for k, v in p2.items() if k != "proofs"})
    assert got[0] == "reject" and got[1].startswith("params: the parameters name no proofs"), got
    one = base | {"proofs": proofs(["s"]), "sessions": ["s"]}
    other = _public_of(rec) + b", another"
    got = run(one, p=p2 | {"proofs": {**p2["proofs"], "public": _sha(other)}})
    assert got[0] == "reject" and got[1].startswith("proofs: public file: no blob has the SHA-512"), got
    first = rec["roots"][0]["port"]
    roots = {r["port"]: r["root"] for r in rec["roots"]} | {first: "00" * 64}
    codes = R.matches_public(rec, SCHEME, roots, {p: s.ports[p]["domain"] for p in s.ports}, s.population).codes
    assert codes == ("roots",)
    wrong = _public_of(rec, roots)
    assert run(one, blobs_=[circuit.hex(), wrong.hex(), *blobs[2:]],
               p=p2 | {"proofs": {**p2["proofs"], "public": _sha(wrong)}}) == (
        "reject", f"proofs: public file: not the registration's: {', '.join(codes)}")
    n = rec["population"]
    head, body = public.split(b"\n", 1)
    unitless = json.dumps({k: v for k, v in json.loads(head).items() if k != "units"}, sort_keys=True).encode()
    relabelled = [_public_of(rec, indices=is_) for is_ in ([0, *range(2, n + 1)], [*range(1, n), 0], list(range(n - 1)),
                                                           list(range(n + 1)))]
    for f in (*relabelled, unitless + b"\n" + body):
        assert run(one, blobs_=[circuit.hex(), f.hex(), *blobs[2:]],
                   p=p2 | {"proofs": {**p2["proofs"], "public": _sha(f)}}) == ("reject", f"proofs: {INDICES_WHY}"), f
    assert run(base | {"proofs": {"public": _sha(public), **proofs(["s"])}, "sessions": ["s"]}) == (
        "reject", "evidence: the keys of the evidence's proofs are not sessions")
    got = run(one, p=p2 | {"proofs": {k: v for k, v in p2["proofs"].items() if k != "public"}})
    assert got == ("reject", "params: the keys of proofs are not circuit, public, registered"), got
    for keys in (["circuit", "z"], ["a", "circuit"], ["circuit-00"], []):
        ch_k, _, base_k, p_k = audit(1, keys)
        extra = D.canonical(ch_k.trees[0].record).encode()
        assert run(base_k | {"proofs": proofs(["s"], extra), "sessions": ["s"]},
                   blobs_=[circuit.hex(), public.hex(), extra.hex()], p=p_k) == (
            "reject", f"proofs: session s's coins are not the challenger's: {CH.LINK_TABLES}"), keys


def _audit_v1_accepts(s: Setup, seed: str, server) -> tuple:
    """An audit of `s` that v1 accepts on the verdict its evidence hands in and its challenger's coins, sampled under
    `seed`: the registration record, the challenger's session, v1's evidence without its ``sessions`` and
    ``verdicts``, and v1's parameters, which v2 reads with ``proofs``."""
    law = {"k": 2, "law": "subset"}
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF)
    ch, d, coins = _challenger_draw(random.Random(seed), s, law, rec)
    proved = DR.proved(d)
    base = {"coins": coins, "draw": d, "instances": len(proved), "proved_units": proved,
            "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
            "statement": {}}
    params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/20",
              "law": law, "leaf_layer": LEAF, "log": [], "partition": P.partition(s.program, s.query),
              "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
              "window_root": ch.ledger.window["root"], "work_table": None}
    v1 = base | {"sessions": [], "verdicts": {"lean": {"accepted": True, "of_record": True}}}
    assert D.replay(ONE_STAGE, _json({"commitment": _hex(rec), "evidence": _hex(v1), "params": params}),
                    server)[0] == "accept", s.templates
    return rec, ch, base, params


def test_v2_refuses_a_population_out_of_order(server, server_v2):
    """v2 refuses a registration whose program's proof blocks, read stratum by stratum, are not its population's units
    in order (`OneStageZk.popHeld`, `Slots.popOk`), before any session is verified, as the service's run refuses it at
    registration (`population-order`): the session's knowledge soundness reads slot `u` as proof block `u`. v1, whose
    R2 holds only the population's count, accepts the same audit on its verdicts. `SETUPS[4]` and `SETUPS[5]` are out
    of order (`test_the_run_refuses_a_population_out_of_order`); `SETUPS[3]`'s audit goes on to its session's
    verdict."""
    circuit = STAND_IN_CIRCUIT
    for s, ok in ((SETUPS[4], False), (SETUPS[5], False), (SETUPS[3], True)):
        assert P.in_order(s.program, s.query, s.population) is ok, s.templates
        rec, ch, base, params = _audit_v1_accepts(s, f"v2-order-{s.templates}", server)
        public, record = _public_of(rec), D.canonical(ch.trees[0].record).encode()
        v2 = base | {"proofs": {"sessions": [{"name": "s", "record": _sha(record), "reps": {}}]}, "sessions": ["s"]}
        case = {"blobs": [circuit.hex(), public.hex(), record.hex()], "commitment": _hex(rec), "evidence": _hex(v2),
                "params": params | {"proofs": {"circuit": _sha(circuit), "public": _sha(public), "registered": None}}}
        out = server_v2.exchange(_json(case))
        want = "verdict flock-verify/s did not accept" if ok else f"proofs: {POPULATION_ORDER}"
        assert (out["verdict"], out["reason"]) == ("reject", want), (s.templates, out)


def test_the_per_member_form_as_written_is_what_v2_parses_and_never_checks(tmp_path, server, server_v2):
    """`proofs.members_v2` writes v2's per-member form of ``proofs`` for a window of several members, and
    `OneStageZk`'s parsers read what it wrote (`proofs.members_parsed`, the executable's ``members``:
    `membersParamsOfJson`, `membersEvidenceOf`, `membersHeld`), on an audit v1 accepts: every member with its files'
    digests, its units and its ``vus_per_block``, and a statement for each member with a session, in the members'
    order. The command's line says ``parsed, not checked`` and names no verdict, and the audit refuses the same case
    for its keys, so no check of the form is reached. What the parsers refuse the command refuses with their reason:
    statements out of the members' order, a statement of two sessions or of a member the parameters don't name, v1's
    ``sessions`` other than the statements', and either file in the one statement's form."""
    rec, _, base, params = _audit_v1_accepts(SETUPS[3], "v2-members", server)

    def member(k: int, name: str, lo: int, hi: int, *sessions: str) -> PF.Member:
        at = tmp_path / f"m{k}"
        at.mkdir()
        (at / "circuit.txt").write_bytes(STAND_IN_CIRCUIT + name.encode())
        (at / "pub.bin").write_bytes(_public_of(rec, indices=list(range(lo, hi))))
        for x in sessions:
            (at / x).mkdir()
            (at / x / "session.json").write_text(json.dumps({"member": name, "session": x}))
            (at / x / "rep-0.bin").write_bytes(f"{name} {x}".encode())
        return PF.Member(name, PF.Proof(at / "circuit.txt", at / "pub.bin", tuple(at / x for x in sessions)),
                         PF.file_sha512(at / "circuit.txt"), PF.file_sha512(at / "pub.bin"), None, ((lo, hi),), 4)

    ms = [member(0, f"{PAIR.id}/c0", 0, 2, "s0"), member(1, f"{TRIPLE.id}/c0", 2, 4),
          member(2, f"{TRIPLE.id}/c1", 4, 5, "s1")]
    pm, pe, blobs = PF.members_v2(ms)
    names = [x["name"] for st_ in pe["statements"] for x in st_["sessions"]]
    assert names == ["s0", "s1"] and [st_["member"] for st_ in pe["statements"]] == [ms[0].name, ms[2].name]
    assert blobs == [f for m in (ms[0], ms[2]) for f in (m.proof.circuit, m.proof.public, m.proof.sessions[0] / "session.json",
                                                         m.proof.sessions[0] / "rep-0.bin")]
    p2, e2 = params | {"proofs": pm}, base | {"proofs": pe, "sessions": names}
    got, refused = PF.members_parsed(p2, e2)
    assert refused is None, refused
    assert got == {
        "members": sorted(({"circuit": m.circuit_sha512, "name": m.name, "public": m.public_sha512, "registered": False,
                            "units": [list(u) for u in m.units], "vus_per_block": 4} for m in ms), key=lambda m: m["name"]),
        "parsed": "parsed, not checked",
        "statements": [{"member": ms[0].name, "sessions": ["s0"]}, {"member": ms[2].name, "sessions": ["s1"]}]}
    assert PF.PARSED == got["parsed"] and not {"verdict", "accept", "accepted", "public", "profile"} & set(got)

    case = {"blobs": [f.read_bytes().hex() for f in blobs], "commitment": _hex(rec), "evidence": _hex(e2), "params": p2}
    out = server_v2.exchange(_json(case))
    assert (out["verdict"], out["reason"]) == ("reject", "params: the keys of proofs are not circuit, public, registered")

    def why(p: dict = p2, e: dict = e2) -> str:
        got, refused = PF.members_parsed(p, e)
        assert got is None, got
        return refused

    first, last = pe["statements"]
    assert why(e=e2 | {"proofs": {"statements": [last, first]}, "sessions": names[::-1]}).startswith(
        f"proofs: member {ms[0].name}: its statement follows a member's whose units start after its own")
    twice = {**first, "sessions": [*first["sessions"], {**first["sessions"][0], "name": "s2"}]}
    assert why(e=e2 | {"proofs": {"statements": [twice, last]}, "sessions": ["s0", "s2", "s1"]}).startswith(
        f"proofs: member {ms[0].name}: its statement has 2 sessions")
    assert why(e=e2 | {"proofs": {"statements": [{**first, "member": "x/c0"}, last]}}) == (
        "proofs: member x/c0: the parameters name no such member")
    assert why(e=e2 | {"proofs": {"statements": [first, {**last, "member": ms[0].name}]}}) == (
        f"proofs: member {ms[0].name}: a second statement: a member's units are one statement's")
    assert why(e=e2 | {"sessions": names[::-1]}) == (
        "evidence: proofs: the sessions are not the evidence's sessions, in order")
    one = {"sessions": [x for st_ in pe["statements"] for x in st_["sessions"]]}
    assert why(e=e2 | {"proofs": one}) == "evidence: the keys of the evidence's proofs are not statements"
    assert why(p=params | {"proofs": {"circuit": ms[0].circuit_sha512, "public": ms[0].public_sha512,
                                      "registered": None}}) == "params: the keys of proofs are not members"
    again = pm["members"] | {"z/c0": {**pm["members"][ms[0].name], "units": [[1, 2]]}}
    assert why(p=params | {"proofs": {"members": again}}) == (
        "params: proofs: unit 1 is in two members' units: a unit is one member's")
    with pytest.raises(ValueError, match="exactly one session"):
        PF.members_v2([member(3, f"{PAIR.id}/c1", 5, 6, "a", "b")])
    with pytest.raises(ValueError, match="two members have one name"):
        PF.members_v2([ms[0], ms[0]])


def test_v2_refuses_a_release_log_out_of_order(server, server_v2):
    """v2 holds its challenger session's release log to the sample, then the link, then the other rounds
    (`OneStageZk.orderHeld`, `Challenger.sampleLinkFirst`) before any session's record is read. Of three audits v1
    accepts on their verdict and their challenger's coins, their coin trees all as the challenger logged them, the
    honest log goes on to its session's verdict; one releasing a coin of the sample after a tree's link is opened
    (`sample-late`), and one whose first round opened is of another stream than the link's (`link-late`), are refused
    (`SAMPLE_LINK_FIRST`). The Python twins (`challenger.sample_link_first`, `one_stage.proofs.order_why`) agree;
    `test_challenger_difftest.py` holds the first to Lean's piece."""
    s = SETUPS[3]
    law = {"k": 2, "law": "subset"}
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF)
    circuit, public = STAND_IN_CIRCUIT, _public_of(rec)
    for order in ORDERS:
        ch, d, coins = _challenger_draw(random.Random(f"v2-order-{order}"), s, law, rec, 1, order=order)
        proved = DR.proved(d)
        evidence = {"coins": coins, "draw": d, "instances": len(proved), "proved_units": proved,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                    "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
        params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/20",
                  "law": law, "leaf_layer": LEAF, "log": [], "partition": P.partition(s.program, s.query),
                  "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
                  "window_root": ch.ledger.window["root"], "work_table": None}
        v1 = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        assert D.replay(ONE_STAGE, v1, server)[0] == "accept", order
        want = None if order == "honest" else f"coins: {Ch.SAMPLE_LINK_FIRST}"
        assert order_why(params, rec, evidence) == want, order
        assert Ch.sample_link_first("draw", bytes.fromhex(R.digest(rec)), ch.s.lines[1:]) == (want is None), order
        record = D.canonical(ch.trees[0].record).encode()
        proofs = {"sessions": [{"name": "s", "record": _sha(record), "reps": {}}]}
        base = {k: v for k, v in evidence.items() if k != "verdicts"}
        case = {"blobs": [circuit.hex(), public.hex(), record.hex()], "commitment": _hex(rec),
                "evidence": _hex(base | {"proofs": proofs, "sessions": ["s"]}),
                "params": params | {"proofs": {"circuit": _sha(circuit), "public": _sha(public), "registered": None}}}
        out = server_v2.exchange(_json(case))
        assert (out["verdict"], out["reason"]) == (
            "reject", "verdict flock-verify/s did not accept" if want is None else f"proofs: {want}"), order


def test_v2_reads_its_draw_from_the_challengers_coins(server_v2):
    """v2 reads its sample's coins as v1 does (`OneStageZk.coinsOf`, v1's `coinsOf`), and only from the auditor's
    challenger under a key at the deployed parameters: parameters naming none, one without a window (`NO_CHALLENGER`) or
    one whose key is at other parameters (`KEY_PARAMS`) are refused, as is evidence holding more than its limits
    (`Challenger.LIMITS`: an opening's path one digest past `MAX_PATH`, and the evidence at it read on), and evidence
    whose coins are not the challenger's signed release is rejected: none, the record carrying its own key, coins under
    another key, on another window, of another family or released after another digest, a session record or close whose
    signature was forged, each mutation of the segment, or a session the ledger doesn't open withheld. So is a session
    whose release log, which its coin trees are read from too, is not the whole log the challenger's signed close
    covers: no close, a log a line short of the count the close signs, or the log's lines in another order than its
    chain, each refused in Python's words. On the challenger's coins it goes on to its own verdicts, and a sample other
    than theirs is rejected after them (`checkBlobsAt`, v1's `checkAt`). Each step Lean rejects at is the Python
    reference's: `decode_params_v2`, `coins_within`, `_coins_of`, then v1's `check` with the session's verdict the
    reject its blobs give, its record on the coin tree the challenger sampled."""
    s, law = SETUPS[3], {"k": 3, "law": "stratified"}
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF)
    after = bytes.fromhex(R.digest(rec))
    circuit, public = STAND_IN_CIRCUIT, _public_of(rec)
    failed = {"flock-verify/s": {"accepted": False, "of_record": True}}

    def audit_of(d: Drawn, draw: dict, coins, **params) -> dict:
        proved = DR.proved(draw)
        record = D.canonical(d.trees[0].record).encode() if d.trees else b"{}"
        proofs = {"sessions": [{"name": "s", "record": _sha(record), "reps": {}}]}
        evidence = {"draw": draw, "instances": len(proved), "proved_units": proved, "proofs": proofs,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                    "sessions": ["s"], "statement": {}}
        if coins is not None:
            evidence["coins"] = coins
        p = {"anchors": [], "challenger": {"key": d.s.key.to_json(), "label": "draw"}, "delta": "1/20", "law": law,
             "leaf_layer": LEAF, "log": [], "partition": P.partition(s.program, s.query), "ports": s.ports,
             "program": s.desc, "proofs": {"circuit": _sha(circuit), "public": _sha(public), "registered": None},
             "query": s.query, "registered": None, "scheme": SCHEME, "window_root": d.ledger.window["root"],
             "work_table": None}
        return {"evidence": evidence, "params": {k: v for k, v in (p | params).items() if k not in params or v is not None},
                "record": record}

    def python(a: dict) -> str:
        try:
            p = decode_params_v2(a["params"])
        except D.Reject:
            return "params"
        if not coins_within(a["evidence"].get("coins")):
            return "evidence"
        e = {k: v for k, v in a["evidence"].items() if k != "proofs"} | {"verdicts": failed}
        try:
            _coins_of(p, (rec, after), e)
        except CoinError:
            return "coins"
        step = check(p, (rec, after), e)[1]
        return "draw" if step == "coins" else step

    def outcome(a: dict) -> dict:
        case = {"blobs": [circuit.hex(), public.hex(), a["record"].hex()], "commitment": _hex(rec),
                "evidence": _hex(a["evidence"]), "params": a["params"]}
        return server_v2.exchange(_json(case))

    def lean(a: dict) -> str:
        out = outcome(a)
        assert out["verdict"] == "reject", out
        step = next(step for prefix, step in (("params: ", "params"), ("evidence: ", "evidence"), ("coins: ", "coins"),
                                              ("draw: coins: ", "draw"), ("verdict flock-verify/s ", "verdict"),
                                              ("", out["reason"]))
                    if out["reason"].startswith(prefix))
        if step == "params" and "challenger" not in a["params"]:
            assert out["reason"] == f"params: {NO_CHALLENGER}", out
        if step == "params" and "challenger" in a["params"] and a["params"]["challenger"]["key"] != d.s.key.to_json():
            assert out["reason"] == f"params: {KEY_PARAMS}", out
        if step == "evidence":
            assert out["reason"] == f"evidence: {LIMITS.form()}", out
        return step

    def path_of(coins: dict, n: int) -> dict:
        """`coins` with its segment's first opening's path padded to `n` digests."""
        seg = coins["segment"]
        o = seg["openings"][0]
        o = {**o, "path": o["path"] + ["00" * 64] * (n - len(o["path"]))}
        return {**coins, "segment": {**seg, "openings": [o, *seg["openings"][1:]]}}

    def signed(coins: dict, part: str) -> dict:
        f = coins["session"][part]
        return {**coins, "session": {**coins["session"], part: {**f, "signature": _flip(f["signature"], 9)}}}

    def files(coins: dict, **f) -> dict:
        return {**coins, "session": {**coins["session"], **f}}

    def unclosed(coins: dict) -> dict[str, tuple[dict, str]]:
        """Session files whose release log isn't the one the challenger's signed close covers, each with Python's
        refusal."""
        log = coins["session"]["log"]
        assert len(log) >= 3 and log[1] != log[2]
        return {"no close": (files(coins, close=None), NO_CLOSE),
                "a log a line short of its close": (
                    files(coins, log=log[:-1]),
                    f"the challenger's release log has {len(log) - 1} lines, not the {len(log)} its close record signs"),
                "a log out of its close's chain": (files(coins, log=[log[0], log[2], log[1], *log[3:]]),
                                                   "the challenger's release log is not the one its close record signs")}

    for seed in range(3):
        rng = random.Random(f"one-stage-v2-coins/{seed}")
        d, draw, coins = _challenger_draw(rng, s, law, rec)
        honest = audit_of(d, draw, coins)
        assert (python(honest), lean(honest)) == ("verdict", "verdict")
        other = next(x for x in (_challenger_draw(rng, s, law, rec)[1] for _ in range(64)) if x["units"] != draw["units"])
        stranger = _challenger_draw(rng, s, law, rec, seed=1)[0]
        assert stranger.s.key != d.s.key
        elsewhere = Drawn(rng, reads=lambda c: c(COIN_BYTES), params=[DEPLOYED])
        bare = Drawn(rng, before=True, after=after, reads=lambda c: c(COIN_BYTES), withheld=False, params=[DEPLOYED])
        undeployed = {**d.s.key.to_json(), **rng.choice([{"H": 6}, {"W": 2}, {"W": 4, "H": 10}])}
        cases = {
            "no challenger": ("params", audit_of(d, draw, coins, challenger=None)),
            "no challenger, no coins": ("params", audit_of(d, draw, None, challenger=None)),
            "no window": ("params", audit_of(d, draw, coins, window_root=None)),
            "a key at other parameters": ("params", audit_of(d, draw, coins, challenger={"key": undeployed,
                                                                                        "label": "draw"})),
            "an opening's path past its limit": ("evidence", audit_of(d, draw, path_of(coins, MAX_PATH + 1))),
            "an opening's path at its limit": ("coins", audit_of(d, draw, path_of(coins, MAX_PATH))),
            "other draw": ("draw", audit_of(d, other, coins)),
            "no coins": ("coins", audit_of(d, draw, None)),
            "the record's own key": ("coins", audit_of(d, draw, d.evidence(d.segment.to_json()))),
            "another key": ("coins", audit_of(d, draw, coins, challenger={"key": stranger.s.key.to_json(),
                                                                             "label": "draw"})),
            "another window": ("coins", audit_of(d, draw, coins, window_root=d.other_window(rng))),
            "another family": ("coins", audit_of(d, draw, coins, challenger={"key": d.s.key.to_json(),
                                                                                "label": "other"})),
            "forged record": ("coins", audit_of(d, draw, signed(coins, "record"))),
            "forged close": ("coins", audit_of(d, draw, signed(coins, "close"))),
            "released after another digest": ("coins", audit_of(
                elsewhere, draw, {k: v for k, v in elsewhere.evidence(elsewhere.segment.to_json()).items() if k != "key"})),
            "not withheld": ("coins", audit_of(
                bare, draw, {k: v for k, v in bare.evidence(bare.segment.to_json()).items() if k != "key"})),
        } | {name: ("coins", audit_of(d, draw, {**coins, "segment": _mutated(rng, d, name)[0]}))
             for name in DRAW_MUTATIONS if name not in ("unread", "read-tampered", "other-after")}
        for name, (want, a) in cases.items():
            assert (python(a), lean(a)) == (want, want), (seed, name)
        for name, (c, why) in unclosed(coins).items():
            a = audit_of(d, draw, c)
            e = {k: v for k, v in a["evidence"].items() if k != "proofs"} | {"verdicts": failed}
            with pytest.raises(CoinError) as py:
                _coins_of(decode_params_v2(a["params"]), (rec, after), e)
            out = outcome(a)
            assert (str(py.value), out["verdict"], out["reason"]) == (why, "reject", f"coins: {why}"), (seed, name)


def committed_bits(j):
    """`Q_call` v1 at X over a program of root Calls (`partition_object.evaluate`): per distinct Definition, in Call order,
    its units' committed bits (`partition._committed_bits`), twice, as Lean gives them read off its layout and by the walk
    that layout replaced."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    desc, x = _obj(j.get("program"), "program"), j.get("X")
    if not (_int(x) and x >= 0):
        raise D.Reject("X is not a natural number")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    try:
        part = PO.evaluate(PROGRAMS[sha], PO.call_query(x))
    except ValueError as e:
        raise D.Reject(str(e)) from None
    cuts = {codec.descriptor_id(cut.fn): cut for _k, cut in part.calls}
    out = []
    for fid, cut in cuts.items():
        bits = P._committed_bits(cut.fn, [(lo, hi) for lo, hi, _path in cut.units])
        out.append([fid, bits, bits])
    return out


def unit_gates(j):
    """Each unit's root gates `[lo, hi)` over a program and query (`partition.unit_gates`), twice, as Lean gives them a
    template query's instances a range at a time and by the walk unit by unit that replaced (`Partition.unitGates`,
    `Partition.unitGatesRef`)."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    desc, query = _obj(j.get("program"), "program"), _obj(j.get("query"), "query")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    try:
        gates = [[lo, hi] for lo, hi in P.unit_gates(PROGRAMS[sha], query)]
    except (P.QueryRefused, ValueError) as e:
        raise D.Reject(str(e)) from None
    return [gates, gates]


def fanout(j):
    """A template query's fan-out and the run Input nodes outside `held`, the gates `[[first, count], ...]` of the
    parameters a verifier expects registered, its proof blocks don't each read alone over a program (`partition.fanout`,
    `partition.shared_inputs`; `Partition.fanout`, `Partition.sharedInputs`), refused for another query or one that
    doesn't apply."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    desc = _obj(j.get("program"), "program")
    held = [] if j.get("held") is None else j["held"]
    if not isinstance(held, list) or not all(isinstance(h, list) and len(h) == 2 and all(_nat(x) for x in h)
                                             for h in held):
        raise D.Reject("held is not a list of [first, count]")
    query = _obj(j.get("query"), "query")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    if not P.is_template_query(query):
        raise D.Reject("not a template query")
    try:
        return {"fanout": P.fanout(PROGRAMS[sha], query),
                "shared": P.shared_inputs(PROGRAMS[sha], query, [range(f, f + n) for f, n in held])}
    except (P.QueryRefused, ValueError) as e:
        raise D.Reject(str(e)) from None


@functools.cache
def _readers(rows: int, plan: tuple, registered: bool) -> Program:
    """A root over `x`, `rows` pairs of bytes (registered with `registered`), of one instance of `PAIR` per step of
    `plan`, `(form, members, sources)`: a call, a batch of `members` mapping its pairs (`map`) or broadcasting one
    (`broadcast`), each byte read by a source `(kind, a, b)` from `x[a][b]` (kind 0, or before any instance) or from an
    earlier instance's result (its `b`-th member's, of a batch), returning every instance's result. Put among
    `PROGRAMS`."""
    def body(B, S, x):
        outs: list = []
        for form, n, sources in plan:
            it = iter(sources)

            def byte():
                kind, a, b = next(it)
                if kind == 0 or not outs:
                    return x[a % rows][b % 2]
                f, v, m = outs[a % len(outs)]
                return v if f == "call" else v[b % m]
            if form == "call":
                outs.append(("call", B.call(PAIR, array_of([byte(), byte()])), 1))
            elif form == "map":
                outs.append(("batch", B.batch(PAIR, array_of([array_of([byte(), byte()]) for _ in range(n)]), axes=(0,)), n))
            else:
                outs.append(("batch", B.batch(PAIR, array_of([byte(), byte()]), axes=(None,), n=n), n))
        return tuple_of(*(v for _f, v, _n in outs))
    ret = Tuple(*(Value(8) if form == "call" else Array(n, Value(8)) for form, n, _s in plan))
    name = "CoreSpotReaders" + hashlib.sha256(repr((rows, plan)).encode()).hexdigest()[:12]
    program = Program(bind(CompositeDefinition(name, 0, (), lambda S: ((("x", Array(rows, Array(2, Value(8)))),), ret),
                                               body, register=False)), registered=("x",) if registered else ())
    PROGRAMS[PO.program_sha512(program)] = program
    return program


@st.composite
def fanout_cases(draw) -> dict:
    """A program and a query for the fan-out: a population program of `PAIR`, or a root over `x`, two to five pairs of
    bytes, a run Input or a registered one, of one to four instances of `PAIR` wired at random (`_readers`), under `PAIR`'s
    template query, or rarely a query that is not one or doesn't apply; with no `held`, the first parameter's gates, or
    gate intervals at random."""
    query = draw(st.sampled_from([P.template_instance_query(PAIR.id)] * 6 + [P.template_instances_query([PAIR.id]),
                                                                            P.template_instance_query(TRIPLE.id),
                                                                            PO.query()]))
    if draw(st.integers(0, 4)) == 0:
        program = P.population_program(PAIR, draw(st.integers(1, 6)))
        PROGRAMS[PO.program_sha512(program)] = program
    else:
        source = st.tuples(st.integers(0, 1), st.integers(0, 9), st.integers(0, 9))
        step = st.tuples(st.sampled_from(["call", "call", "map", "broadcast"]), st.integers(2, 3),
                         st.tuples(*[source] * 6))
        program = _readers(draw(st.integers(2, 5)), tuple(draw(st.lists(step, min_size=1, max_size=4))),
                           draw(st.integers(0, 3)) == 0)
    case = {"program": _json(codec.encode_program(program)), "query": _json(query)}
    held = draw(st.integers(0, 3))
    if held == 1:
        case["held"] = [[g.start, len(g)] for g in G.held_gates(program, [program.fn.signature()[0][0][0]])]
    elif held == 2:
        case["held"] = draw(st.lists(st.lists(st.integers(0, 48), min_size=2, max_size=2), max_size=2))
    return case


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
COMMITTED_BITS = D.Port(PIECES, "committed_bits", committed_bits, st.one_of(
    st.builds(lambda m, x: {"program": json.loads(_calls(tuple(m))), "X": x},
              st.lists(st.sampled_from(range(len(CALLEES))), min_size=1, max_size=5), st.sampled_from([4, 8, 16, 32])),
    JSON), package=PACKAGE)
UNIT_GATES = D.Port(PIECES, "unit_gates", unit_gates, st.one_of(
    st.builds(lambda c: {"program": c[0], "query": c[1]}, partitions()), JSON), package=PACKAGE)
FANOUT = D.Port(PIECES, "fanout", fanout, st.one_of(fanout_cases(), JSON), package=PACKAGE)
_X0, _X00 = ((0, 0, 0), (0, 0, 1)), ((0, 0, 0),) * 4
#: Programs R8 reads past a population's: two calls of `PAIR` on `x`'s first pair, a batch of three broadcasting it (a
#: run Input two proof blocks read), and a call reading the first's result whose result a batch of two maps (F 4).
READER_SETUPS = [Setup(_readers(2, (("call", 2, _X0 + _X00),) * 2, False), P.template_instance_query(PAIR.id)),
                 Setup(_readers(2, (("broadcast", 3, _X0 + _X00),), False), P.template_instance_query(PAIR.id)),
                 Setup(_readers(3, (("call", 2, _X0 + _X00), ("call", 2, ((1, 0, 0), (0, 1, 0)) + _X00),
                                    ("map", 2, ((1, 1, 0), (0, 2, 0), (0, 2, 1), (0, 1, 1), (0, 0, 0), (0, 0, 0)))),
                                False), P.template_instance_query(PAIR.id))]
_REGISTRATION_OF = lambda a: {"params": a["params"], "record": a["record"]}  # noqa: E731
REGISTRATION = D.Port(PIECES, "registration", registration, st.one_of(
    changed_audits(most=1, setups=READER_SETUPS).map(_REGISTRATION_OF),
    changed_audits([TIED], 1, 1, READER_SETUPS).map(_REGISTRATION_OF),
    changed_audits([TIED, REGISTERED_PARAMS], 2, 1, REGISTERED_SETUPS).map(_REGISTRATION_OF),
    _audit_inputs(_REGISTRATION_OF, [RECORD, PARAMS, ENTRIES, REGISTERED_PARAMS, SUPPLIED], 2),
    changed_audits([ENTRIES], 1, 1, REGISTERED_SETUPS).map(_REGISTRATION_OF),
    changed_audits([SUPPLIED], 1, 0, SUPPLIED_SETUPS).map(_REGISTRATION_OF),
    changed_audits([SUPPLIED, ENTRIES, RECORD], 2, 2, SUPPLIED_SETUPS).map(_REGISTRATION_OF),
    changed_audits([BINDINGS], 1, 1, SUPPLIED_SETUPS).map(_REGISTRATION_OF),
    changed_audits([WINDOWS], 1, 1, REGISTERED_SETUPS).map(_REGISTRATION_OF),
    changed_audits([WINDOWS, ENTRIES, REGISTERED_PARAMS], 2, 2, REGISTERED_SETUPS).map(_REGISTRATION_OF),
    changed_audits([SEED_SALTS], 1, 1, REGISTERED_SETUPS).map(_REGISTRATION_OF),
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
    registered_values(), changed_audits([WINDOWS], 1, 0, REGISTERED_SETUPS).map(lambda a: {"params": a["params"]}),
    _audit_inputs(lambda a: {"params": a["params"]}, [PARAMS]), JSON), package=PACKAGE)
TEXT = st.text(max_size=4) | st.sampled_from(["W", "x", "é", "\x00", "\U0001f600", '"'])
def _domains(widths):
    return st.builds(lambda program, value, words, leaves, b: {"leaves": leaves, "program": program, "value": value,
                                                               "word_bits": b, "words": words},
                     st.sampled_from([s.sha512 for s in REGISTERED_SETUPS]) | TEXT, TEXT, st.integers(0, 70),
                     st.integers(0, 70) | st.integers(0, 1 << 40), st.sampled_from(widths))


DOMAIN = D.Port(PIECES, "domain", domain, st.one_of(
    _domains([16, 32]), st.builds(lambda d, w: {**d, "window": w}, _domains([16, 32]), st.sampled_from([WROOT, EARLIER_WROOT])),
    st.builds(lambda d, r: {**d, "produced": r}, _domains([16, 32]), st.sampled_from([WROOT, "c3" * 64])),
    st.builds(lambda d, r, w: {**d, "produced": r, "window": w}, _domains([16, 32]),
              st.sampled_from([WROOT, WROOT.upper(), WROOT[:-2], None, 5]), st.sampled_from([WROOT, WROOT[:-2]])),
    _domains([16, 32]), st.one_of(st.builds(lambda d, w: {**d, "window": w}, _domains([16, 32]),
                        st.sampled_from([WROOT.upper(), WROOT[:-2], "", None, 5, WROOT + "\n"])),
    _domains([0, 8, 24, 48]),
    st.builds(lambda k, v: {"leaves": 3, "program": "p", "value": "W", "word_bits": 16, "words": 2, k: v},
              st.sampled_from(["leaves", "program", "value", "word_bits", "words"]), JSON), JSON)), package=PACKAGE)
PRODUCED_REG, PRODUCED_DRAW, PRODUCED_OTHER = "c3" * 64, "d4" * 64, "e0" * 64


def _bc(*parts) -> str:
    """A stand-in ``b || c``, lowercase hex of 128 bytes."""
    h = hashlib.sha512(repr(parts).encode()).hexdigest()
    return h + hashlib.sha512(h.encode()).hexdigest()


def _produced_case(names, sizes, roots) -> dict:
    """An honest produced record over values ``names`` (``sizes``: (leaves, words, word_bits) each), with the openings of
    every row and of rows 0 and 3 of a registered value ``A``, the verifier's own entries, a port it registers at known
    content (``ok``), the roots its verifiers opened and their verdicts, one per value and one of ``A`` and
    ``ok``, each opening its rows at the record's openings: `produced.check`, `produced.opened` and `produced.shared`
    all empty."""
    prog = REGISTERED_SETUPS[0].sha512
    mine = {v: {"leaves": n, "words": w, "word_bits": b, "domain": PD.domain(prog, v, w, n, b, PRODUCED_REG).domain_id.hex()}
            for v, (n, w, b) in zip(names, sizes)}
    es = [{"value": v, "root": r, "leaves": m["leaves"], "words": m["words"], "domain": m["domain"],
           **({} if m["word_bits"] == 16 else {"word_bits": m["word_bits"]})} for (v, m), r in zip(mine.items(), roots)]
    ops = {**{v: [(i, _bc(v, i)) for i in range(m["leaves"])] for v, m in mine.items()}, "A": [(0, _bc("A", 0)), (3, _bc("A", 3))]}
    known = {"ok": [[0, _bc("ok", 0)]]}

    def verdict(reads):
        return {"of_record": True, "accepted": True, "registered": {p: {"value": v, "root": "ab" * 64} for p, (v, _) in reads.items()},
                "openings": {p: [[i, bc] for i, bc in rows] for p, (_, rows) in reads.items()}}

    verdicts = {f"s{x}": verdict({"p": (v, ops[v][-1:])}) for x, v in enumerate(mine)}
    verdicts["sA"] = verdict({"a0": ("A", ops["A"]), "o0": ("ok", [(0, _bc("ok", 0))])})
    return {"record": PD.record(PRODUCED_REG, PRODUCED_DRAW, es, ops), "registration": PRODUCED_REG, "draw": PRODUCED_DRAW,
            "mine": mine, "opened": {e["value"]: e["root"] for e in es}, "known": known, "verdicts": verdicts}


def _entry(c, x):
    return c["record"]["produced"][x % len(c["record"]["produced"])]


def _opening(c, x):
    """A verdict's reported row: ``sA``'s ``a0`` row ``x % 2``."""
    return c["verdicts"]["sA"]["openings"]["a0"][x % 2]


#: one change each to an honest case: (name, change, the code it reaches)
PRODUCED_CHANGES = [
    ("keys", lambda c, x: c["record"].pop("draw"), "P-keys"),
    ("extra key", lambda c, x: c["record"].update(window=None), "P-keys"),
    ("format", lambda c, x: c["record"].update(format="verity/registration/produced/v0"), "P-format"),
    ("no openings", lambda c, x: c["record"].update(openings=[]), "P-openings"),
    ("a produced row's opening missing", lambda c, x: c["record"]["openings"][_entry(c, x)["value"]].pop(), "P-openings"),
    ("a produced value's openings missing", lambda c, x: c["record"]["openings"].pop(_entry(c, x)["value"]), "P-openings"),
    ("produced rows out of order", lambda c, x: c["record"]["openings"][_entry(c, x)["value"]].reverse(), "P-openings"),
    ("a position twice", lambda c, x: c["record"]["openings"]["A"].append(list(c["record"]["openings"]["A"][0])),
     "P-openings"),
    ("a negative position", lambda c, x: c["record"]["openings"]["A"][x % 2].__setitem__(0, -1), "P-openings"),
    ("an opening in capitals", lambda c, x: c["record"]["openings"]["A"][x % 2].__setitem__(1, _bc("A").upper()),
     "P-openings"),
    ("a short opening", lambda c, x: c["record"]["openings"]["A"][x % 2].__setitem__(1, _bc("A")[:-2]), "P-openings"),
    ("an opening not a pair", lambda c, x: c["record"]["openings"]["A"][x % 2].append(0), "P-openings"),
    ("a session opens another opening", lambda c, x: _opening(c, x).__setitem__(1, _bc("other")), "P-shared"),
    ("a session opens a row the record has no opening of", lambda c, x: _opening(c, x).__setitem__(0, 1), "P-shared"),
    ("a session reports no rows of a port", lambda c, x: c["verdicts"]["sA"]["openings"].pop("a0"), "P-shared"),
    ("a session reports no openings", lambda c, x: c["verdicts"]["sA"].pop("openings"), "P-shared"),
    ("a session reports an empty port", lambda c, x: c["verdicts"]["sA"]["openings"].update(a0=[]), "P-shared"),
    ("a session's ok is not the known row", lambda c, x: c["verdicts"]["sA"]["openings"]["o0"][0].__setitem__(1, _bc("ok", 1)),
     "P-shared"),
    ("a produced value's session opens another opening",
     lambda c, x: c["verdicts"]["s0"]["openings"]["p"][0].__setitem__(1, _bc("other")), "P-shared"),
    ("a rejected session's other opening", lambda c, x: [c["verdicts"]["sA"].update(accepted=False),
                                                         _opening(c, x).__setitem__(1, _bc("other"))], None),
    ("a session not of record", lambda c, x: [c["verdicts"]["sA"].update(of_record=False),
                                              _opening(c, x).__setitem__(1, _bc("other"))], None),
    ("a value the verifier holds no openings of", lambda c, x: c["verdicts"]["sA"]["registered"]["a0"].update(value="B"), None),
    ("the verifier's own openings of a value replace the record's", lambda c, x: c["known"].update(A=c["record"]["openings"]["A"][:1]), "P-shared"),
    ("registration", lambda c, x: c.update(registration=PRODUCED_OTHER), "P-registration"),
    ("draw", lambda c, x: c["record"].update(draw=PRODUCED_OTHER), "P-draw"),
    ("not a list", lambda c, x: c["record"].update(produced={}), "P-produced"),
    ("an entry's extra key", lambda c, x: _entry(c, x).update(row_schema="hm96-sha512/row/v2"), "P-produced"),
    ("a value twice", lambda c, x: c["record"]["produced"].append(dict(_entry(c, x))), "P-values"),
    ("a value missing", lambda c, x: c["record"]["produced"].pop(x % len(c["record"]["produced"])), "P-values"),
    ("a value not the verifier's", lambda c, x: _entry(c, x).update(value="other"), "P-values"),
    ("a value not a string", lambda c, x: _entry(c, x).update(value=x), "P-values"),
    ("a root in capitals", lambda c, x: _entry(c, x).update(root=_entry(c, x)["root"].upper()), "P-root"),
    ("a short root", lambda c, x: _entry(c, x).update(root=_entry(c, x)["root"][:-2]), "P-root"),
    ("leaves", lambda c, x: _entry(c, x).update(leaves=_entry(c, x)["leaves"] + 1), "P-leaves"),
    ("leaves as a bool", lambda c, x: _entry(c, x).update(leaves=True), "P-leaves"),
    ("words", lambda c, x: _entry(c, x).update(words=_entry(c, x)["words"] + 1), "P-words"),
    ("another registration's domain", lambda c, x: _entry(c, x).update(domain=PD.domain(
        REGISTERED_SETUPS[0].sha512, _entry(c, x)["value"], _entry(c, x)["words"], _entry(c, x)["leaves"],
        _entry(c, x).get("word_bits", 16), PRODUCED_OTHER).domain_id.hex()), "P-domain"),
    ("a registered value's domain", lambda c, x: _entry(c, x).update(domain=G.domain(
        REGISTERED_SETUPS[0].sha512, _entry(c, x)["value"], _entry(c, x)["words"], _entry(c, x)["leaves"],
        _entry(c, x).get("word_bits", 16)).domain_id.hex()), "P-domain"),
    ("16 named", lambda c, x: _entry(c, x).update(word_bits=16), "P-word_bits"),
    ("32 unnamed", lambda c, x: [m.update(word_bits=32) for m in c["mine"].values()], "P-word_bits"),
    ("a value opened at another root", lambda c, x: c["opened"].update({_entry(c, x)["value"]: PRODUCED_OTHER}),
     "P-opened"),
    ("a value not opened", lambda c, x: c["opened"].pop(_entry(c, x)["value"]), "P-opened"),
    ("a value opened under two roots", lambda c, x: c["opened"].update({_entry(c, x)["value"]: None}), "P-opened"),
    ("no verdict reports", lambda c, x: c.update(opened=None), "P-opened"),
    ("another value opened too", lambda c, x: c["opened"].update(other=PRODUCED_OTHER), None),
]
PRODUCED_NAMES = ["partial", "PouwUnitSeeds_v2", "s", "é"]


@st.composite
def produced_cases(draw) -> dict:
    names = draw(st.lists(st.sampled_from(PRODUCED_NAMES), min_size=1, max_size=3, unique=True))
    sizes = [(draw(st.integers(1, 40)), draw(st.integers(1, 9)), draw(st.sampled_from([16, 16, 32]))) for _ in names]
    c = _produced_case(names, sizes, [draw(st.sampled_from(["a1", "b2", "0f"])) * 64 for _ in names])
    for _ in range(draw(st.integers(0, 2))):
        _name, change, _code = draw(st.sampled_from(PRODUCED_CHANGES))
        try:
            change(c, draw(st.integers(0, 3)))
        except (KeyError, IndexError, ZeroDivisionError, AttributeError, TypeError):
            pass
    return c


PRODUCED = D.Port(PIECES, "produced", produced, st.one_of(
    produced_cases(), produced_cases(), st.builds(lambda c, x: {**c, "record": x}, produced_cases(), JSON),
    st.builds(lambda c, k, x: {**c, k: x}, produced_cases(),
              st.sampled_from(["mine", "opened", "registration", "draw", "known", "verdicts"]), JSON),
    st.builds(lambda c, x: {**c, "known": {"ok": x}}, produced_cases(), JSON), JSON), package=PACKAGE)
KNOWN_SALT = bytes(range(192)).hex()
KNOWN_PORT = D.Port(PIECES, "known_port", known_port, st.one_of(
    st.builds(lambda program, value, row, reg, b: {"program": program, "value": value, "row": row, "salt": KNOWN_SALT,
                                                   "registration": reg, "word_bits": b},
              st.sampled_from([s.sha512 for s in REGISTERED_SETUPS]) | TEXT, st.sampled_from(["ok", "pub3", "admit"]) | TEXT,
              st.lists(st.integers(0, 3) | st.integers(0, 65535), min_size=1, max_size=110), st.sampled_from([PRODUCED_REG, WROOT]),
              st.sampled_from([16, 16, 32])),
    st.builds(lambda row, b, salt, reg: {"program": "p", "value": "ok", "row": row, "salt": salt, "registration": reg,
                                         "word_bits": b},
              st.lists(st.integers(0, 1 << 33), max_size=3), st.sampled_from([0, 8, 16, 24, 32, 48]),
              st.sampled_from([KNOWN_SALT, KNOWN_SALT[:-2], KNOWN_SALT.upper(), KNOWN_SALT[:-1], "", "00" * 191 + "0g"]),
              st.sampled_from([PRODUCED_REG, PRODUCED_REG.upper(), PRODUCED_REG[:-2], ""])),
    st.builds(lambda k, v: {"program": "p", "value": "ok", "row": [1], "salt": KNOWN_SALT, "registration": PRODUCED_REG,
                            "word_bits": 16, k: v},
              st.sampled_from(["program", "value", "row", "salt", "registration", "word_bits"]), JSON), JSON), package=PACKAGE)
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
    """Byte sources (a challenger's coins), `least` to `most` bytes after some `ff` bytes or none: a word of them is
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
#: Law texts `draw.parse_design` or `draw.parse_rate` refuses or reads oddly: digits of other scripts, signs, spaces, a
#: second separator, leading zeros, a missing or zero argument.
ODD_LAW_TEXTS = ["subset:", "subset", "", "subset:\u0663", "work:\u00b2", "stratified:\uff11", "bernoulli:1/\u0662",
                 "bernoulli:\u00b9/2", "bernoulli:01/04", "bernoulli:0/0", "bernoulli:3/2", "bernoulli:1", "bernoulli:1/2/3",
                 "subset:1:2", "subset:+1", "subset: 1", "subset:-1", "stratified:0", "subset:0", "subset:00", "work:007",
                 "subset:007", "nope:1",
                 "Subset:1", "bernoulli:", "bernoulli:/2", "rate:", "rate", "rate:0/3", "rate:4/3", "rate:1/0",
                 "rate:1/2:0", "rate:1/2:", "rate:1/2:3:4", "rate:1/2/3", "rate:\u0661/2", "rate:1/2:\u0663",
                 "rate:01/02:003", "Rate:1/2", "rate:/2", "rate:1/2: 3", "rate:+1/2", "rate:1:2",
                 "stratified-rate:", "stratified-rate", "work-rate:0/3", "work-rate:4/3", "stratified-rate:1/0",
                 "stratified-rate:1/2:3", "work-rate:1/2/3", "work-rate:\u0661/2", "stratified-rate:1/\u0663",
                 "stratified-rate:01/02", "Work-rate:1/2", "work_rate:1/2", "stratified-rate:+1/2", "work-rate:1/ 2",
                 "stratified-rate:1", "work-rate:/2", "stratified:1/2", "work:1/2"]
#: A rate law's text, `rate:NUM/DEN[:T]`, the tables written or not.
RATE_TEXTS = st.builds(lambda num, den, t: f"rate:{num}/{den}" + ("" if t is None else f":{t}"),
                       st.integers(0, 12), st.integers(0, 12), st.none() | st.integers(0, 6))
#: A stratified or work rate law's text, `stratified-rate:NUM/DEN` or `work-rate:NUM/DEN`.
STRATA_RATE_TEXTS = st.builds(lambda kind, num, den: f"{kind}:{num}/{den}", st.sampled_from(["stratified-rate", "work-rate"]),
                              st.integers(0, 12), st.integers(0, 12))
LAW_TEXTS = st.one_of(
    st.builds(lambda kind, k: f"{kind}:{k}", st.sampled_from(["subset", "stratified", "work"]), st.integers(0, 30)),
    st.builds(lambda num, den: f"bernoulli:{num}/{den}", st.integers(0, 12), st.integers(0, 12)),
    RATE_TEXTS, STRATA_RATE_TEXTS, st.sampled_from(ODD_LAW_TEXTS))
#: Rules a stream doesn't run, or that aren't rules as `to_json` writes them.
ODD_RULES = [{"check": "all", "replay": "subset:1", "rule": "two-stage"},
             {"check": 2, "replay": "subset:1", "rule": "two-stage"}, {"check": "some", "replay": "subset:1", "rule": "two-stage"},
             {"check": True, "replay": "subset:1", "rule": "two-stage"}, {"replay": "subset:1", "rule": "two-stage"},
             {"law": "uniform:3", "population": "n", "rule": "sequential"}, {"rule": "all", "x": 1}, {"law": 5, "rule": "law"},
             {"rule": "law"}, {"law": "rate:1/2", "rule": "law", "x": 1}, {"law": ["rate:1/2"], "rule": "law"},
             {"check": "all", "replay": 1, "rule": "two-stage"}, {"rule": "All"}, {"rule": None}, {},
             *({"grain": P.template_instance_query(PAIR.id), "k": 1, "replay": "rate:1/2", "rule": "two-stage", **change}
               for change in ({"k": 0}, {"k": -1}, {"k": "1"}, {"k": True}, {"k": None}, {"k": {}}, {"k": [1]},
                              {"k": {PAIR.id: 0}}, {"k": {PAIR.id: True}}, {"k": {PAIR.id: "1"}}, {"replay": "bernoulli:1/2"},
                              {"replay": "stratified:2"}, {"replay": "work:2"}, {"replay": "subset:2"},
                              {"replay": "stratified-rate:1/2"}, {"replay": "work-rate:1/2"}, {"replay": 2},
                              {"grain": P.template_instances_query([PAIR.id])}, {"grain": PO.query()},
                              {"grain": {**P.template_instance_query(PAIR.id), "version": 1}},
                              {"grain": P.template_instance_query(PAIR.id) | {"x": 1}}, {"check": "all"}))]


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
    """The select piece's input on a random partition: a rule of each kind the stream runs (a rate's tables sometimes
    more than its count fits), now and then a fixed count, which it refuses, the work table naming every template for a
    work rate (sometimes with a template's work 0, or a bad entry), mostly a closure map for a work rate, and the coins."""
    desc, query, n = draw(partitions())
    kind = draw(st.sampled_from(["work-rate", "stratified-rate", "bernoulli", "rate", "work-rate", "all", "two-stage",
                                 "fixed"]))
    if kind == "all":
        rule = {"rule": "all"}
    elif kind == "two-stage":
        den = draw(st.integers(1, 12))
        replay = draw(st.sampled_from([f"rate:{draw(st.integers(1, den))}/{den}", f"rate:{draw(st.integers(1, den))}/{den}",
                                       f"subset:{draw(st.integers(1, max(n, 1)))}"]))
        grain = query if query.get("name") == "Q_template_instance" else P.template_instance_query(draw(st.sampled_from(WORK_IDS)))
        rule = {"grain": grain, "k": draw(st.integers(1, 3)), "replay": replay, "rule": "two-stage"}
    else:
        den = draw(st.integers(1, 12))
        if kind == "bernoulli":
            text = f"bernoulli:{draw(st.integers(0, den))}/{den}"
        elif kind == "rate":
            tables = draw(st.none() | st.integers(1, 4))
            text = f"rate:{draw(st.integers(1, den))}/{den}" + ("" if tables is None else f":{tables}")
        elif kind == "fixed":
            text = f"{draw(st.sampled_from(['subset', 'stratified', 'work']))}:{draw(st.integers(1, max(n, 1)))}"
        else:
            text = f"{kind}:{draw(st.integers(1, den))}/{den}"
        rule = {"law": text, "rule": "law"}
    table = None
    if kind in ("work-rate", "fixed"):
        table = {t: draw(st.integers(1, 4) | WORK_ENTRY) for t in WORK_IDS}
        if draw(st.integers(0, 9)) == 9:
            table[draw(st.sampled_from(WORK_IDS))] = draw(BAD_WORK_ENTRY)
    closure = draw(closure_maps(n)) if kind == "work-rate" and draw(st.integers(0, 3)) < 3 else None
    return {"closure": closure, "program": desc, "query": query, "rule": rule, "source": draw(SOURCE), "work_table": table}


def _with(field: str, values):
    return lambda c, v: {**c, field: v}


#: One change each to the select piece's input: the rule (a sampling design text of any form, a rule the stream doesn't run, or
#: junk), the work table (none, empty, junk), the closure map (any sampling design's, malformed, a unit twice), the query, the
#: program, the coins (short, junk) or a field missing.
SELECT_CHANGES = [
    (_with("rule", None), LAW_TEXTS.map(lambda t: {"law": t, "rule": "law"})),
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


@st.composite
def second_stages(draw) -> dict:
    """The select_second piece's input: a replay sample of up to four of eight replay blocks, cells of one to three
    templates of 1 to 5 proof blocks each (at least two in all), a count within [1, n_t] in each, not every cell whole,
    and the coins; or, a third of the time, one field changed to a form the second stage refuses (a sample out of order,
    twice, empty or not counts; no cells, a cell of none, one proof block in all, counts of another length; a count 0 or
    past its cell, every cell whole; short or junk coins) or missing."""
    sample = sorted(draw(st.sets(st.integers(0, 7), min_size=1, max_size=4)))
    ns = draw(st.lists(st.integers(1, 5), min_size=1, max_size=3).filter(lambda ns: sum(ns) >= 2))
    ks = draw(st.tuples(*(st.integers(1, n) for n in ns)).filter(lambda ks: list(ks) != ns))
    c = {"sample": sample, "ns": ns, "ks": list(ks), "source": draw(SOURCE)}
    if draw(st.integers(0, 2)) == 0:
        key, value = draw(st.sampled_from([
            ("sample", sample[::-1] if len(sample) > 1 else []), ("sample", sample + sample[-1:]), ("sample", []),
            ("sample", [-1, *sample]), ("sample", [True]), ("sample", "0"), ("ns", []), ("ns", [0, *ns[1:]]),
            ("ns", [1]), ("ns", [*ns, 2]), ("ns", "3"), ("ns", [True, *ns[1:]]), ("ks", [0, *ks[1:]]),
            ("ks", [ns[0] + 1, *ks[1:]]), ("ks", list(ns)), ("ks", list(ks[:-1])), ("ks", [True, *ks[1:]]),
            ("source", "zz"), ("source", ""), ("source", 5), (None, None)]))
        drop = draw(st.sampled_from(sorted(c)))
        c = {kk: v for kk, v in c.items() if kk != drop} if key is None else {**c, key: value}
    return c


SELECT_SECOND = D.Port(PIECES, "select_second", select_second, st.one_of(second_stages(), second_stages(), JSON),
                       package=PACKAGE)

_PARAMS_CHANGES = {
    "a template dropped": lambda p, q: ({"templates": p["templates"][:-1]}, q),
    "a template added": lambda p, q: ({"templates": [*p["templates"], {"template": "z/9", "circuit": "0" * 128,
                                                                        "header": {}, "registered": None}]}, q),
    "the templates reversed": lambda p, q: ({"templates": p["templates"][::-1]}, q),
    "a template twice": lambda p, q: ({"templates": [*p["templates"], p["templates"][0]]}, q),
    "a circuit in capitals": lambda p, q: ({"templates": [{**p["templates"][0], "circuit": "AB" * 64},
                                                          *p["templates"][1:]]}, q),
    "a circuit short": lambda p, q: ({"templates": [{**p["templates"][0], "circuit": "ab" * 63 + "a"},
                                                    *p["templates"][1:]]}, q),
    "a circuit and a newline": lambda p, q: ({"templates": [{**p["templates"][0], "circuit": "ab" * 64 + "\n"},
                                                            *p["templates"][1:]]}, q),
    "a circuit a number": lambda p, q: ({"templates": [{**p["templates"][0], "circuit": 5}, *p["templates"][1:]]}, q),
    "an entry with another key": lambda p, q: ({"templates": [{**p["templates"][0], "x": 1}, *p["templates"][1:]]}, q),
    "an entry without its circuit": lambda p, q: ({"templates": [{k: v for k, v in p["templates"][0].items()
                                                                  if k != "circuit"}, *p["templates"][1:]]}, q),
    "an entry without its header": lambda p, q: ({"templates": [{k: v for k, v in p["templates"][0].items()
                                                                 if k != "header"}, *p["templates"][1:]]}, q),
    "a header null": lambda p, q: ({"templates": [{**p["templates"][0], "header": None}, *p["templates"][1:]]}, q),
    "a header a list": lambda p, q: ({"templates": [{**p["templates"][0], "header": [{}]}, *p["templates"][1:]]}, q),
    "a header a string": lambda p, q: ({"templates": [{**p["templates"][0], "header": "h"}, *p["templates"][1:]]}, q),
    "a header empty": lambda p, q: ({"templates": [{**p["templates"][0], "header": {}}, *p["templates"][1:]]}, q),
    "an entry without its registered reads": lambda p, q: ({"templates": [
        {k: v for k, v in p["templates"][0].items() if k != "registered"}, *p["templates"][1:]]}, q),
    "registered reads null": lambda p, q: ({"templates": [{**p["templates"][0], "registered": None},
                                                          *p["templates"][1:]]}, q),
    "registered reads empty": lambda p, q: ({"templates": [{**p["templates"][0], "registered": {}},
                                                           *p["templates"][1:]]}, q),
    "registered reads a list": lambda p, q: ({"templates": [{**p["templates"][0], "registered": [{}]},
                                                            *p["templates"][1:]]}, q),
    "registered reads a string": lambda p, q: ({"templates": [{**p["templates"][0], "registered": "r"},
                                                              *p["templates"][1:]]}, q),
    "an entry not an object": lambda p, q: ({"templates": ["a", *p["templates"][1:]]}, q),
    "the params with another key": lambda p, q: ({**p, "circuit": "0" * 128}, q),
    "the params empty": lambda p, q: ({}, q),
    "the params a list": lambda p, q: (p["templates"], q),
    "templates not a list": lambda p, q: ({"templates": "a"}, q),
    "no params": lambda p, q: (None, q),
    "another query": lambda p, q: (p, {**q, "name": "Q_call"}),
    "another version": lambda p, q: (p, {**q, "version": 1}),
    "no version": lambda p, q: (p, {k: v for k, v in q.items() if k != "version"}),
    "no query": lambda p, q: (p, None),
}


@st.composite
def template_circuit_cases(draw) -> dict:
    """The template_circuits piece's input: the check's template query over one to three templates, one template's
    query or several's, and the params naming each one's circuit file, reference header and registered reads in its
    order; or, half the time, one named change (`_PARAMS_CHANGES`) to a form the params or the query refuse, or one that
    still holds (no params, one template reversed, a version missing that reads as 0, an empty header, registered reads
    null or empty)."""
    ids = draw(st.lists(st.sampled_from(["a/1", "b/2", "c/3", "d/4"]), min_size=1, max_size=3, unique=True))
    query = {"name": "Q_template_instance", "version": 0, "params": {"template": ids[0]}} \
        if len(ids) == 1 and draw(st.booleans()) else \
        {"name": "Q_template_instances", "version": 0, "params": {"templates": ids}}
    circuits = draw(st.lists(st.binary(min_size=64, max_size=64).map(bytes.hex), min_size=len(ids), max_size=len(ids)))
    headers = draw(st.lists(st.sampled_from([_second_reference(ids[0], 0, 2, SECOND_SCHEMAS),
                                             {"scheme": "s", "instances": 3, "units": {"indices": [0]}},
                                             {"set": "served/r/m0", "frame_v3": {"roots": {"x": "aa" * 64}}}]),
                            min_size=len(ids), max_size=len(ids)))
    reads = draw(st.lists(st.sampled_from([
        None, {"format": "verity/registered-reads/v1", "ports": {"x": {"value": BX}}},
        {"ports": {"x": {"value": "x", "root": "aa" * 64}, "o0": {"value": BOUT}}}]),
        min_size=len(ids), max_size=len(ids)))
    proofs = {"templates": [{"template": t, "circuit": c, "header": h, "registered": r}
                            for t, c, h, r in zip(ids, circuits, headers, reads)]}
    if draw(st.booleans()):
        proofs, query = _PARAMS_CHANGES[draw(st.sampled_from(sorted(_PARAMS_CHANGES)))](proofs, query)
    return {"query": query} if proofs is None else {"proofs": proofs, "query": query}


TEMPLATE_CIRCUITS = D.Port(PIECES, "template_circuits", template_circuits,
                           st.one_of(template_circuit_cases(), template_circuit_cases(), JSON), package=PACKAGE)
SELECT = D.Port(PIECES, "select", select, st.one_of(selections(), selections(), changed_selections(1),
                                                     changed_selections(2), JSON), package=PACKAGE)
SLOTS = D.Port(PIECES, "slots", slots, st.one_of(selections(), selections(), changed_selections(1), JSON), package=PACKAGE)
_FITS_HEX = st.binary(max_size=6).map(bytes.hex)
_FITS_JSON = st.recursive(st.none() | st.booleans() | st.integers(-300, 300) | TEXT | st.sampled_from(["\n\r\t\\", "\x1f\x7f"]),
                          lambda inner: st.lists(inner, max_size=3) | st.dictionaries(TEXT, inner, max_size=3),
                          max_leaves=5)
FITS = D.Port(PIECES, "fits", fits, st.one_of(
    st.builds(lambda limit, c, p, ts, program, part, reg, draw: {
        "circuit": c, "draw": draw, "limit": limit, "partition": part, "program": program, "public": p, "registered": reg,
        "tables": ts}, st.integers(0, 24), _FITS_HEX, _FITS_HEX, st.lists(st.tuples(TEXT, _FITS_HEX).map(list), max_size=3),
        st.none() | _FITS_HEX, _FITS_JSON, _FITS_JSON, _FITS_JSON),
    JSON), package=PACKAGE)


@st.composite
def file_populations(draw) -> dict:
    """The file_population piece's input on a random partition: the public file's units mostly its population in order,
    else with one dropped, added or moved, or all shifted by one."""
    desc, query, n = draw(partitions())
    indices = list(range(n))
    kind = draw(st.integers(0, 7))
    if kind == 4 and n:
        del indices[draw(st.integers(0, n - 1))]
    elif kind == 5:
        indices.insert(draw(st.integers(0, n)), draw(st.integers(0, n + 1)))
    elif kind == 6 and n > 1:
        i = draw(st.integers(0, n - 2))
        indices[i], indices[i + 1] = indices[i + 1], indices[i]
    elif kind == 7:
        indices = [u + 1 for u in indices]
    return {"indices": indices, "program": desc, "query": query}


@st.composite
def file_population_publics(draw) -> dict:
    """The file_population piece's input with the public file itself (``public``, in hex): a header written as
    `verity_flock`'s producers write theirs (`json.dumps(head, sort_keys=True)`, a newline, then the body), over
    `file_populations`' units; else that header cut short, without its units, or after a byte that isn't UTF-8."""
    j = draw(file_populations())
    indices = j.pop("indices")
    n, g = len(indices), draw(st.integers(1, 4))
    head = {"format": "flock-instances/v3", "statement": "hm96-sha512-rows", "scheme": "hm96-sha512", "set": "eval",
            "subcircuit": "unit", "range": [0, n], "instances": n, "circuit_sha512": "ab" * 64, "rows": True,
            "units": {"vus_per_block": g, "units_per_vu": draw(st.integers(1, 2)), "indices": indices,
                      "blocks": [indices[b:b + g] for b in range(0, n, g)], "classes": ["cd" * 64] * n},
            "frame_v3": {"roots": {"x": "ef" * 64}, "domain_ids": {"x": "01" * 32}},
            **draw(st.sampled_from([{}, {"identifier": {"name": "caf\u00e9 \u2603"}}, {"shared_rows": {"x": n}},
                                    {"registered": {"w": {"value": "w", "positions": [0, 3], "paths": ["a/b"]}}}]))}
    kind = draw(st.integers(0, 9))
    if kind == 8:
        del head["units"]
    line = json.dumps(head, sort_keys=True).encode()
    if kind == 7:
        line = line[:draw(st.integers(0, len(line) - 1))]
    elif kind == 9:
        line = b"\xff" + line
    return {**j, "public": (line + b"\n" + draw(st.binary(max_size=64))).hex()}


FILE_POPULATION = D.Port(PIECES, "file_population", file_population,
                         st.one_of(file_populations(), file_populations(), file_population_publics(),
                                   file_population_publics(), JSON), package=PACKAGE)
TABLE_DIGEST = st.one_of(st.sampled_from(sorted(PF.MUFU_SHA512.values())), st.binary(max_size=64).map(bytes.hex))
LOOKUPS = st.one_of(st.dictionaries(st.one_of(st.sampled_from(sorted(PF.MUFU_SHA512)), st.text("abx", max_size=4)),
                                    st.just({}), max_size=5),
                    st.none(), st.integers(-9, 9), st.text("abx", max_size=4), st.lists(st.sampled_from(sorted(PF.MUFU_SHA512))))
TABLES_HELD = D.Port(PIECES, "tables_held", tables_held, st.one_of(
    st.builds(lambda ds, lk: {"digests": ds, "lookups": lk}, st.lists(TABLE_DIGEST, max_size=5), LOOKUPS),
    st.builds(lambda ds, lk: {"digests": ds, "lookups": lk}, st.lists(TABLE_DIGEST, max_size=5), LOOKUPS),
    st.lists(TABLE_DIGEST, max_size=5).map(lambda ds: {"digests": ds}),
    st.lists(st.one_of(TABLE_DIGEST, st.sampled_from(["0", "AB", "0g", PF.MUFU_SHA512["rcp"].upper()]), st.integers(0, 3)),
             min_size=1, max_size=4).map(lambda ds: {"digests": ds}),
    JSON), package=PACKAGE)

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
PROBABILITY = st.builds(lambda d, n: str(Fraction(min(n, d), d)), st.integers(1, 1000), st.integers(0, 1000))
BERNOULLI_BOUND = D.Port(PIECES, "bernoulli_bound", bernoulli_bound, st.one_of(
    st.builds(lambda n, p, d: {"delta": d, "p": p, "population": n}, st.integers(0, 5000), PROBABILITY, DELTA),
    st.builds(lambda n, p, d: {"delta": d, "p": p, "population": n}, st.integers(0, 5000),
              st.sampled_from(["0", "1", "1/2", "1/1000", "999/1000"]), DELTA),
    st.builds(lambda n, p, d: {"delta": d, "p": p, "population": n},
              st.one_of(st.integers(0, 50), st.sampled_from([-1, True, "3", None])),
              st.one_of(PROBABILITY, st.sampled_from(["3/2", "2/4", "-1/2", "1/0", "a", 1, None])),
              st.one_of(DELTA, st.sampled_from(["0", "1", "1/1", "3/2", "a"]))),
    JSON), package=PACKAGE)
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
    public = ["population", "window", R.SEED_SALT, "leaf_layer", "anchors", *(f"population:{t}" for t in PROGRAMS_TEMPLATES),
              *(f"{k}:{v['name']}" for v in values for k in ("leaves", "domain", *(("words", "word_bits") if v["registered"] else ())))]
    return change({"public": public, "query": p["query"], "record": a["record"], "scheme": SCHEME, "values": values}, x)


def _record_with(c: dict, **fields) -> dict:
    return {**c, "record": {**c["record"], **fields}}


#: One change each to the rule's input: a declared item or every one dropped, a value dropped, added, twice or the other
#: kind, another query or scheme, a record that is not one, a stratified sampling design's templates, and entries the rule names.
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
    except Exception:  # noqa: BLE001 - a changed parameter the verifier would refuse: the record's own sampling design stands in
        law = a["record"].get("law") if isinstance(a["record"], dict) else None
    out = {"delta": delta, "law": law, "registration": a["record"]}
    return out | {k: e.get(k) for k in ("draw", "instances", "proved_units", "receipt", "served", "session_roots", "sessions",
                                         "statement", "verdicts")}


def _own_reads_case(a: dict, x: int) -> dict:
    """`_record_case` with `own_reads` (or one not a boolean), its first verdict also reporting a read of one of the
    registration's ports, at that port's root or another."""
    c = _record_case(a, "1/4")
    reg, vs = c["registration"], c["verdicts"]
    if isinstance(reg, dict) and isinstance(reg.get("roots"), list) and reg["roots"] and isinstance(vs, dict) and vs:
        r, name = reg["roots"][x % len(reg["roots"])], sorted(vs)[0]
        if isinstance(r, dict) and isinstance(vs[name], dict):
            got = vs[name].get("registered") if isinstance(vs[name].get("registered"), dict) else {}
            read = {"value": r.get("port"), "root": r.get("root") if x % 3 else "cd" * 64}
            c["verdicts"] = {**vs, name: {**vs[name], "registered": {**got, "own": read}}}
    return {**c, "own_reads": [True, True, False, 1, None][x % 5]}


#: δ as the exact ratio of a double (`str(Fraction(delta))`), as the switch passes it.
EXACT_DELTA = st.floats(2 ** -40, 1 - 2 ** -40).map(lambda f: str(Fraction(f))) | st.sampled_from(
    [str(Fraction(f)) for f in (0.5, 0.25, 0.1, 0.05, 0.01, 1e-6, 2 ** -20, 0.999)])
RECORD_PORT = D.Port(PIECES, "record", record, st.one_of(
    st.builds(_record_case, changed_audits(most=0), EXACT_DELTA),
    st.builds(_own_reads_case, changed_audits(most=0), st.integers(0, 63)),
    st.builds(_own_reads_case, changed_audits([REGISTERED_EVIDENCE], 1, 0, REGISTERED_SETUPS), st.integers(0, 63)),
    st.builds(_record_case, changed_audits([EVIDENCE, DRAW_CHANGES, VERDICTS_OF, REGISTERED_EVIDENCE], 1, 1), EXACT_DELTA),
    st.builds(_record_case, changed_audits([RECORD, ENTRIES, REGISTERED_EVIDENCE], 1, 1, REGISTERED_SETUPS), EXACT_DELTA),
    st.builds(_record_case, changed_audits(most=2, least=2), EXACT_DELTA),
    st.builds(_record_case, changed_audits(most=0), st.sampled_from(["0", "1", "3/2", "a", "1/0", "2/4", 5, None])),
    st.builds(lambda a, k: {kk: v for kk, v in _record_case(a, "1/4").items() if kk != k}, changed_audits(most=0),
              st.sampled_from(sorted(RECORD_INPUT_KEYS))),
    JSON), package=PACKAGE)

RUN_SESSIONS = ["5e" * 64, "a1" * 64]


def _law_text(law, n) -> str | None:
    """The rule's law text naming an agreed law over a record of `n` proof blocks: a rate's own, `subset:K` as the rate
    `rate:K/N`, which names `subset:K` again over N, and any other law object's text, which a stream refuses."""
    if isinstance(law, str):
        return law
    if not isinstance(law, dict):
        return None
    if law.get("law") == "subset" and type(law.get("k")) is int and type(n) is int and 1 <= law["k"] <= n:
        return f"rate:{law['k']}/{n}"
    return f"bernoulli:{law.get('p')}" if law.get("law") == "bernoulli" else f"{law.get('law')}:{law.get('k')}"


def _run_case(a: dict, source: str, sessions: int, outputs: list) -> dict:
    """`run`'s input from an audit: the rule naming the verifier's law (`_law_text`), every item
    `_public_items` declares, the challenger's name and place, the coins, and the evidence's statement and verdicts, each
    verdict of record about the delivered sessions and reporting the public outputs."""
    p, e = a["params"], a["evidence"]
    text = _law_text(p.get("law"), a["record"].get("population") if isinstance(a["record"], dict) else None)
    rule = {"rule": "all"} if text is None else {"law": text, "rule": "law"}
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


def _from_challenger(c: dict, rng: random.Random) -> tuple[Drawn, dict]:
    """Run `c` sampling from the auditor's challenger, as a `Stream` over a window runs it: its coins a ledger session's
    challenger opened after the registration's digest and read by the selection rule's sample (`Drawn`), the source the
    bytes read, the evidence the session's and the window root the ledger's open's."""
    rec, p, rule = c["record"], _params_of(c), _rule(c["rule"])
    law = DR.derive(p["program"], p["query"],
                    service.selection_law(rule, rec["population"], service.selection_sizes(rule, p["program"], p["query"])),
                    service.selection_table(rule, p["program"], p["query"], p["work_table"]))

    def reads(coins):
        got = bytearray()

        def source(n):
            b = coins(n)
            got.extend(b)
            return b
        service.select_draw(rule, law, rec["population"], source)
        return bytes(got)
    d = Drawn(rng, before=True, after=bytes.fromhex(R.digest(rec)), reads=reads)
    return d, {**c, "params": {**c["params"], "window_root": d.ledger.window["root"]}, "source": d.read.hex(),
               "coins": d.evidence(d.segment.to_json())}


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


def _as_rate(c: dict, x: int) -> dict:
    """The run's `rate:K/N` rule over the record's N (or a fixed `subset:K`) over tables: one table or a divisor of K,
    which is `rate:K/N` again, tables K doesn't divide, which rounds K up past the record's law (R1) or past N, or N + 1
    tables, past N (`rate`). Any other rule is unchanged."""
    rule, rec = c.get("rule"), c.get("record")
    if not isinstance(rule, dict):
        return c
    m = re.fullmatch(r"subset:([0-9]+)|rate:([0-9]+)/([0-9]+)", str(rule.get("law")))
    n = rec.get("population") if isinstance(rec, dict) else None
    if m is None or not (type(n) is int and m[3] in (None, str(n))):
        return c
    m = [None, m[1] or m[2]]
    if not 1 <= int(m[1]) <= n:
        return c
    k = int(m[1])
    divisors = [t for t in range(1, k + 1) if k % t == 0]
    tables = [None, 1, divisors[x % len(divisors)], n + 1, 2 + x % 5][x % 5]
    return {**c, "rule": {**rule, "law": f"rate:{k}/{n}" + ("" if tables is None else f":{tables}")}}


#: One change each to the run's own inputs: the rule (a sampling design text of any form, its subset sampling design as a rate law, a rule the
#: stream doesn't run, or junk), the declared items and values, the challenger's name and place, the closure map, the
#: coins, where they came from and their evidence, the sessions, a verdict, the outputs, the statement, δ, the proved units and instances, the served units, the
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
    (_with("coin_source", None), st.sampled_from(["challenger", "verifier", "", "Verifier", 5, None])),
    (_with("coins", None), st.sampled_from([{}, {"segment": None}, 5, []])),
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
    (_with("session_roots", None), st.sampled_from([{}, {"x": "00" * 64}, dict(ROOTS), {**ROOTS, "y": "00" * 64}, 5,
                                                    None])),
    (lambda c, k: {kk: v for kk, v in c.items() if kk != k},
     st.sampled_from(["challenger", "closure", "delta", "instances", "outputs", "params", "proved_units",
                      "public",
                      "received", "record", "rule", "served", "session_roots", "sessions", "source", "statement",
                      "values", "verdicts"])),
]


@st.composite
def runs(draw, groups=(), most: int = 0, changes: int = 0) -> dict:
    """A run's input from an audit with up to `most` changes from `groups`, then `changes` of the run's own."""
    a = draw(changed_audits(list(groups), most=most, least=min(most, 1))) if groups else draw(changed_audits(most=0))
    c = _run_case(a, draw(SOURCE), draw(st.integers(1, 2)), draw(st.sampled_from([[], [], ["credited"]])))
    for _ in range(changes):
        change, values = draw(st.sampled_from(RUN_CHANGES))
        c = change(c, draw(values))
    return c


#: The boundary registration's roots, its ports its own; the interior registration's are `ROOTS`.
BOUNDARY_ROOTS = {BOUT: "ce" * 64, BX: "ac" * 64}
#: Two of `BOUNDARY`'s four replay blocks, then one of the three proof blocks in each.
TWO_STAGE_RULE = {"grain": BOUNDARY.query, "k": 1, "replay": "rate:1/2", "rule": "two-stage"}


def _values_and_items(*setups: Setup) -> tuple[list, list]:
    """Each setup's ports as unregistered values, and every item a registration over them may show, declared."""
    values = [[{"name": n, "registered": False} for n in sorted(s.ports)] for s in setups]
    public = ["population", "window", "leaf_layer", "anchors", *(f"population:{t}" for t in [*PROGRAMS_TEMPLATES, BLOCK.id]),
              *(f"{k}:{v['name']}" for vs in values for v in vs for k in ("leaves", "domain"))]
    return values, public


#: Each interior port's row schema, as `Setup` derives its domain.
SECOND_SCHEMAS = {"out": "u16", "x": "hm96-sha512/row/v1"}


def _second_reference(t: str, lo: int, m: int, ports) -> dict:
    """Template `t`'s reference header (`service.template_circuits`), fixed before any run: its reference statement's
    public file's header over the `m` slots from `lo`, each of `ports` a row of its `SECOND_SCHEMAS` schema, in the set
    it was staged in and under that set's bindings, and none of the run's fields."""
    circuit = hashlib.sha512(t.encode()).hexdigest()
    return {"format": "flock-circuit-inputs", "statement": "verity/flock-circuit", "scheme": SCHEME, "set": f"unit/{t}",
            "subcircuit": f"unit/{t}", "range": [0, m], "circuit_sha512": circuit, "rows": False,
            "units": {"vus_per_block": 1, "units_per_vu": 1, "blocks": [[u] for u in range(lo, lo + m)],
                      "classes": [circuit] * m},
            "frame_v3": {"frame": "frame-v3-sha512", "owner": -1, "schemas": {p: SECOND_SCHEMAS[p] for p in ports},
                         "bindings": dict.fromkeys(ports, "00" * 64), "row_key_sha512": "ab" * 64}}


def _second_statement(rec: dict, part: SS.Part, q: int, drawn, named: bool, registered=None, k: int = 0,
                      held=None) -> dict:
    """The statement of `part`'s second-stage session as an honest prover's run makes it (`second_stage`): its template,
    its public file's header the served path's (`flock_inputs`) for the `k`-th part: its template's reference header
    (`_second_reference`) over the part's slots, in the set `served/<run>/m<k>` of the interior record `rec`'s run, at
    `rec`'s roots of the template's ports (each `<template>/<port>` in `rec` when `named`) and at each port of `held`
    the root it maps it to (a wired port's, its writer's), each port bound by the window rule over `rec`'s program,
    partition and run at the port of its root (`ml.commitments.window`); its sample the second stage's units `drawn`
    among them, counted from its first, over a replay sample of `q`; and its registered reads."""
    prefix, lo, m = f"{part.template}/" if named else "", part.lo, q * part.n
    roots = {**{r["port"].removeprefix(prefix): r for r in rec["roots"] if r["port"].startswith(prefix)}, **(held or {})}
    units = [u - lo for u in drawn if lo <= u < lo + m]
    ref, run = _second_reference(part.template, lo, m, roots), rec["window"]["run"]
    fv = ref["frame_v3"]
    return {"template": part.template, "registered": registered,
            "header": {**ref, "set": f"served/{run}/m{k}", "instances": m,
                       "units": {**ref["units"], "indices": list(range(lo, lo + m))},
                       "frame_v3": {**fv, "roots": {p: r["root"] for p, r in roots.items()},
                                    "domain_ids": {p: r["domain"] for p, r in roots.items()},
                                    "bindings": {p: window_binding(rec["program"], rec["partition"], r["port"],
                                                                   fv["schemas"][p], r["leaves"], run).hex()
                                                 for p, r in roots.items()}}},
            "draw": {"law": "subset", "population": m, "k": len(units), "units": units}}


def _checked_names(s) -> dict:
    """What `flock-verify verify`'s verdict names of the session of statement `s` it checked (`second_stage.verified_of`):
    its public file's header and its registered reads by the SHA-512 of their canonical JSON, its sample, and one table,
    as the verifier of record runs it."""
    if not isinstance(s, dict):
        return {}
    out = {"public_header_sha512": _sha(codec.canonical_json(s.get("header"))), "unit_draw": s.get("draw"),
           "session_tables": 1}
    return out if s.get("registered") is None else {**out, "registered_sha512": _sha(codec.canonical_json(s["registered"]))}


def _checked_by(statements) -> dict:
    """An accepting verdict of record for each of `statements`' sessions, naming it as it was checked (`_checked_names`)."""
    return {f"lean-verify/{i}": {"of_record": True, "accepted": True, **_checked_names(s)}
            for i, s in enumerate(statements if isinstance(statements, list) else [])}


def _as_stated(c: dict) -> dict:
    """The second_statements input `c` with each statement's session checked as it states it (`_checked_by`)."""
    if not isinstance(c.get("verdicts"), dict):
        return c
    kept = {k: v for k, v in c["verdicts"].items() if not k.startswith("lean-verify/")}
    return {**c, "verdicts": {**kept, **_checked_by(c.get("statements"))}}


@functools.cache
def _two_stage_text(seed: int, elsewhere: bool = False) -> str:
    """A two-stage run's input (canonical text; the coins' sessions are signed once): the boundary registration over
    `BOUNDARY`'s replay blocks, its first stage from a ledger session's challenger opened after its digest, then the
    interior registration over the replay sample's six proof blocks, logged after it in a window of its own that names
    the replay sample in order, its second stage from another session's challenger, its window's root the one the
    verifier computes from the boundary registration's digest and the replay sample (`second_stage.interior_root`), or
    when `elsewhere` a sampled one, opened after the interior registration's digest; the verdicts accepting, the second-stage session's statement the honest one
    (`_second_statement`), each slot reading the boundary's `x` and `o0` at its PAIR instance's row of its cell's replay
    block, which the verifier names as it checked it (`_checked_names`), and the auditor's
    params naming its template's circuit file, reference header (`_second_reference`) and the registered reads its
    reference statement reads, the statement's, which round 2 compares but for the run's fields
    (`second_stage.blank_reads`)."""
    rng = random.Random(f"two-stage/{seed}")
    rule, b, i = _rule(TWO_STAGE_RULE), BOUNDARY, INTERIOR
    first_law = DR.derive(b.program, b.query, service.selection_law(rule, b.population))
    rec = R.record(b.program, b.query, first_law, SCHEME, [{"port": p, "root": BOUNDARY_ROOTS[p], **b.ports[p]} for p in b.ports],
                   WINDOW, leaf_layer=LEAF)
    sample = []

    def reads(draw_one):
        def reader(coins):
            got = bytearray()

            def source(n):
                out = coins(n)
                got.extend(out)
                return out
            draw_one(source)
            return bytes(got)
        return reader
    d1 = Drawn(rng, before=True, after=bytes.fromhex(R.digest(rec)), reads=reads(
        lambda source: sample.extend(service.select_draw(rule, first_law, b.population, source)["units"])))
    ns = service.replay_cells(Program(BLOCK), i.query)
    ks = service.second_counts(rule.k, [PAIR.id], ns)
    m = len(sample) * sum(ns)
    irec = R.record(i.program, i.query, {"law": "subset", "k": m}, SCHEME,
                    [{"port": p, "root": ROOTS[p], **i.ports[p]} for p in i.ports],
                    {**WINDOW, "interior": True, "replay_sample": list(sample)}, leaf_layer=LEAF, previous=R.digest(rec))
    second = {}

    def second_stage(source):
        second["draw"] = service.select_second(sample, ns, ks, source)
    d2 = Drawn(rng, before=True, after=bytes.fromhex(R.digest(irec)), reads=reads(second_stage),
               window=None if elsewhere else {"root": SS.interior_root(R.digest(rec), sample)})
    n = sum(ns)
    linked = {p: {"value": v, "root": BOUNDARY_ROOTS[v], **b.ports[v], "reads_from": 0,
                  "reads": [b.ports[v]["leaves"] // b.population * sample[u // n] + u % n for u in range(m)]}
              for p, v in (("x", BX), ("o0", BOUT))}
    statements = [_second_statement(irec, SS.Part(i.templates[0], 0, n), len(sample), DR.proved(second["draw"]), False,
                                    {"format": "verity/registered-reads/v1", "ports": linked})]
    (values, ivalues), public = _values_and_items(b, i)

    def params(s: Setup, law, log, d: Drawn) -> dict:
        return {"anchors": [], "delta": "1/10", "law": law, "leaf_layer": LEAF, "log": log, "partition": None,
                "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
                "work_table": None, "window_root": d.ledger.window["root"]}
    verdicts = {"lean": {"accepted": True, "digest": "aa" * 64, "of_record": True, "public": {"credited": 12},
                         "sessions": RUN_SESSIONS,
                         "registered": {p: {"value": e["value"], "root": e["root"]} for p, e in linked.items()},
                         **_checked_names(statements[0])},
                "upstream": {"accepted": True, "of_record": False}}
    return D.canonical({
        "challenger": "verifier-test", "closure": None, "coins": d1.evidence(d1.segment.to_json()), "delta": "1/4",
        "instances": None, "outputs": ["credited"], "params": params(b, first_law, [], d1), "proved_units": None,
        "public": public, "received": RECEIVED, "record": rec, "rule": TWO_STAGE_RULE, "served": None,
        "session_roots": dict(BOUNDARY_ROOTS), "sessions": RUN_SESSIONS, "source": d1.read.hex(),
        "statement": {"instances": len(sample)}, "values": values, "verdicts": verdicts,
        "interior": {"params": {**params(i, irec["law"], [rec], d2), "proofs": {"templates": [
                         {"template": i.templates[0], "circuit": "ab" * 64,
                          "header": _second_reference(i.templates[0], 0, m, i.ports),
                          "registered": statements[0]["registered"]}]}},
                     "replay": _json(codec.encode_program(Program(BLOCK))),
                     "values": ivalues, "record": irec, "received": "2026-10-07T00:01:00Z", "source": d2.read.hex(),
                     "coins": d2.evidence(d2.segment.to_json()), "proved_units": None, "instances": None,
                     "served": None, "session_roots": dict(ROOTS), "statements": statements}})


def _two_stage_case(seed: int = 0, elsewhere: bool = False) -> dict:
    return json.loads(_two_stage_text(seed, elsewhere))


def _inner(change):
    """A change to the interior registration's part of a two-stage run."""
    def apply(c, x):
        if not isinstance(c.get("interior"), dict):
            return c
        return {**c, "interior": change(copy.deepcopy(c["interior"]), x)}
    return apply


def _inner_record(**fields):
    return _inner(lambda it, x: {**it, "record": {**it["record"], **fields}})


def _inner_params(**fields):
    return _inner(lambda it, x: {**it, "params": {**it["params"], **fields}} if isinstance(it.get("params"), dict) else it)


def _inner_window(c, x: int):
    """The interior registration's window root changed: none, one no session opened, or the boundary registration's."""
    boundary = c["params"].get("window_root") if isinstance(c.get("params"), dict) else None
    return _inner_params(window_root=[None, "22" * 64, boundary][x % 3])(c, x)


def _restated(ss, x: int):
    """The `x`-th change to a two-stage run's second-stage statements `ss`: its statement twice, of another template,
    without a field or with one more, or its header's slots, their blocks (which round 2 blanks), instances, scheme, a
    root or the domains, its set (another part's) or its ports' bindings, its read paths named (`registered`, which
    round 2 blanks), its sample's units or
    population, or its registered reads (a boundary port's, read by a port of no interior root or by one of the
    interior's; the interior's `x`, each slot a row `x` on; or not an object) changed."""
    if not (isinstance(ss, list) and ss and isinstance(ss[0], dict)):
        return ss
    s = ss[0]
    h = s["header"] if isinstance(s.get("header"), dict) else {}
    units = h["units"] if isinstance(h.get("units"), dict) else {}
    fv = h["frame_v3"] if isinstance(h.get("frame_v3"), dict) else {}
    roots = fv["roots"] if isinstance(fv.get("roots"), dict) else {}
    domains = fv["domain_ids"] if isinstance(fv.get("domain_ids"), dict) else {}
    bindings = fv["bindings"] if isinstance(fv.get("bindings"), dict) else {}
    d = s["draw"] if isinstance(s.get("draw"), dict) else {}
    link = {"value": BX, "root": BOUNDARY_ROOTS[BX], "reads_from": 0, "reads": [x % 4] * 6, **BOUNDARY.ports[BX]}
    own = {"value": "x", "root": roots.get("x"), "domain": domains.get("x"), "leaves": 6, "reads_from": 0,
           "reads": [(u + x) % 6 for u in range(6)]}
    changed = [
        [s, s], [{**s, "template": "Other_v0"}], [{k: v for k, v in s.items() if k != "draw"}], [{**s, "more": 1}],
        [{**s, "header": {**h, "units": {"indices": list(range(1, 7))}}}],
        [{**s, "header": {**h, "units": {**units, "blocks": [[x % 6]]}}}], [{**s, "header": {**h, "instances": 5}}],
        [{**s, "header": {**h, "scheme": "frame-v3-sha512/other"}}],
        [{**s, "header": {**h, "frame_v3": {**fv, "roots": {**roots, "x": "00" * 64}}}}],
        [{**s, "header": {**h, "frame_v3": {**fv, "domain_ids": {}}}}],
        [{**s, "header": {**h, "set": "served/r-test/m1"}}],
        [{**s, "header": {**h, "frame_v3": {**fv, "bindings": dict.fromkeys(bindings, "00" * 64)}}}],
        [{**s, "header": {**h, "registered": {"x": {"positions": [x % 6]}}}}],
        [{**s, "draw": {**d, "units": [0, 3]}}], [{**s, "draw": {**d, "population": 12}}],
        [{**s, "registered": {"ports": {"y": link}}}], [{**s, "registered": {"ports": {"x": link}}}],
        [{**s, "registered": {"ports": {"x": own}}}], [{**s, "registered": 5}]]
    return changed[x % len(changed)]


def _reproved(it: dict, x: int) -> dict:
    """An interior with its params' reference headers none, of another scheme or with a key more, its reference
    statements' registered reads none, naming another value at each port, or with the run's fields (which round 2
    blanks), or its params in a form `service.template_circuits` refuses: a header null, registered reads a list, no
    templates, not an object."""
    p = it.get("params")
    if not isinstance(p, dict):
        return it
    proofs = p.get("proofs")
    es = proofs.get("templates") if isinstance(proofs, dict) else None
    es = [e for e in es if isinstance(e, dict)] if isinstance(es, list) else []

    def headers(f):
        return {"templates": [{**e, "header": f(e.get("header"))} for e in es]}

    def more(**kw):
        return headers(lambda h: {**h, **kw} if isinstance(h, dict) else h)

    def reads(f):
        return {"templates": [{**e, "registered": f(e.get("registered"))} for e in es]}

    def ported(**kw):
        def each(r):
            if not (isinstance(r, dict) and isinstance(r.get("ports"), dict)):
                return r
            return {**r, "ports": {q: {**e, **kw} if isinstance(e, dict) else e for q, e in r["ports"].items()}}
        return reads(each)
    return {**it, "params": {**p, "proofs": [
        None, more(scheme="s"), more(set="served/r/m0"), headers(lambda h: None), reads(lambda r: None),
        ported(value=BOUT), ported(root="00" * 64, reads=[0]), reads(lambda r: [r]), {"templates": []}, 5][x % 10]}}


#: One change each to a two-stage run: its rule (another k, one per template, one naming another template or one more,
#: a replay law the record's law isn't, another grain), the
#: interior registration missing or not one, its record's population, previous record, window or roots, its values,
#: its copies' query (its params' reference headers then dropped) or program, its params' reference headers
#: (`_reproved`), its replay block, its coins, their evidence, its received, proved units, instances,
#: served units, session roots, second-stage statements (none, not a list, or one changed: `_restated`), links, wires
#: or rows it names (which the verifier derives, so refuses), a verdict that didn't accept, the reads a verdict reports
#: (the interior's `x` or the boundary's `x` at their roots, `x` at another or none), and what it names of the session
#: it checked (another header, none, no registered reads, another sample, two tables or no table count).
TWO_STAGE_CHANGES = [
    (_verdict_change(lambda v, c, x: {**v, **[{"public_header_sha512": "00" * 64}, {"public_header_sha512": 5},
                                              {"registered_sha512": None}, {"unit_draw": None}, {"session_tables": 2},
                                              {"session_tables": None}][(x // 2) % 6]}),
     st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "links": [None, [], 5, [{"template": 1, "port": "x", "value": BX}],
                                          [{"template": INTERIOR.templates[0], "port": "y", "value": BX}]][x % 5]}),
     st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "wires": [None, [], [{"template": INTERIOR.templates[0], "port": "x",
                                                       "writer": INTERIOR.templates[0], "wport": "o0"}]][x % 3]}),
     st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "rows": [None, [], [{"template": INTERIOR.templates[0], "port": "x", "per": 3,
                                                      "runs": [[0, 3, 0, 1]]}]][x % 3]}),
     st.integers(0, 63)),
    (_verdict_change(lambda v, c, x: {**v, "registered": [
        {"x": {"value": "x", "root": ROOTS["x"]}}, {"y": {"value": BX, "root": BOUNDARY_ROOTS[BX]}},
        {"x": {"value": "x", "root": "00" * 64}}, {"x": {"value": "x"}}, {}][x % 5]}), st.integers(0, 63)),
    (lambda c, x: {**c, "rule": {**c["rule"], **[{"k": 2}, {"k": 3}, {"k": 4}, {"replay": "subset:2"},
                                                 {"replay": "rate:3/4"}, {"grain": INTERIOR.query}, {"k": {PAIR.id: 2}},
                                                 {"k": {TRIPLE.id: 1}}, {"k": {PAIR.id: 1, TRIPLE.id: 1}}][x % 9]}},
     st.integers(0, 63)),
    (_with("rule", None), st.sampled_from([{"law": "rate:1/2", "rule": "law"}, {"rule": "all"}])),
    (_with("interior", None), st.sampled_from([None, 5, {}, []])),
    (_inner_record(population=5), st.just(0)), (_inner_record(population=12), st.just(0)),
    (_inner_window, st.integers(0, 63)),
    (_inner_record(previous="11" * 64), st.just(0)), (_inner_record(window=WINDOW), st.just(0)),
    (_inner(lambda it, x: {**it, "record": {k: v for k, v in it["record"].items() if k != "previous"}}), st.just(0)),
    (_inner(lambda it, x: {**it, "values": it["values"][1:] if x % 2 else [*it["values"], {"name": "z", "registered": False}]}),
     st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "params": {**{k: v for k, v in it["params"].items() if k != "proofs"},
                                            "query": [BOUNDARY.query, PO.query(), SETUPS[0].query][x % 3]}}),
     st.integers(0, 63)),
    (_inner(_reproved), st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "params": {**it["params"], "program": [SETUPS[1].desc, BOUNDARY.desc][x % 2]}}),
     st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "replay": [SETUPS[0].desc, SETUPS[1].desc, INTERIOR.desc, 5, {}][x % 5]}),
     st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "source": [str(it.get("source"))[:-2], str(it.get("source"))[:8], "", "zz", 5][x % 5]}),
     st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "coins": [None, {}, {"segment": None}, 5][x % 4]}), st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "received": [5, None, ""][x % 3]}), st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "proved_units": [[], [0], [0, 3], "u", [True]][x % 5]}), st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "instances": [0, 1, 5, "1", True][x % 5]}), st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "served": [[], [0], "s"][x % 3]}), st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "session_roots": [{}, None, dict(BOUNDARY_ROOTS), {**ROOTS, "x": "00" * 64}, 5][x % 5]}),
     st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "statements": [None, [], [5], 5, "s"][x % 5]}), st.integers(0, 63)),
    (_inner(lambda it, x: {**it, "statements": _restated(it.get("statements"), x)}), st.integers(0, 63)),
    (_verdict_change(lambda v, c, x: {**v, "accepted": [False, None][x % 2]}), st.integers(0, 63)),
]


@st.composite
def two_stage_runs(draw, changes: int = 0) -> dict:
    """A two-stage run's input (`_two_stage_case`, one of three), with `changes` of its own or of any run's."""
    c = _two_stage_case(draw(st.integers(0, 2)))
    for _ in range(changes):
        change, values = draw(st.sampled_from(TWO_STAGE_CHANGES + TWO_STAGE_CHANGES + RUN_CHANGES))
        c = change(c, draw(values))
    return c


def _links_of(j):
    """`secondStatements`' `links`: None for null (none derived), else as `SecondStage.linksOf?` reads them, a list of
    objects of exactly `template`, `port` and `value`, strings."""
    if j is None:
        return None
    if not (isinstance(j, list) and all(isinstance(x, dict) and sorted(x) == ["port", "template", "value"]
                                        and all(isinstance(s, str) for s in x.values()) for x in j)):
        raise D.Reject("links is null or a list of {template, port, value}, strings")
    return tuple(SS.Link(x["template"], x["port"], x["value"]) for x in j)


def _wires_of(j):
    """`secondStatements`' `wires`: None for null (none derived), else as `SecondStage.wiresOf?` reads them, a list of
    objects of exactly `template`, `port`, `writer` and `wport`, strings."""
    if j is None:
        return None
    if not (isinstance(j, list) and all(isinstance(x, dict) and sorted(x) == ["port", "template", "wport", "writer"]
                                        and all(isinstance(s, str) for s in x.values()) for x in j)):
        raise D.Reject("wires is null or a list of {template, port, writer, wport}, strings")
    return tuple(SS.Wire(x["template"], x["port"], x["writer"], x["wport"]) for x in j)


def _rows_of(j):
    """`secondStatements`' `rows`: none for null, else as `SecondStage.rowsOf?` reads them (`second_stage.rows_of`)."""
    if j is None:
        return ()
    rows = SS.rows_of(j)
    if rows is None:
        raise D.Reject("rows is null or a list of {template, port, per, runs}, runs [unit, count, row, step] naturals")
    return rows


def second_statements(j):
    """`Pieces.secondStatements`: round 2's statement check (`second_stage.check`) of `statements` (null for none), the
    two-stage form allowed when `allow`, over what `_second_own` reads of `j`."""
    allow, own, statements = _second_own(j)
    return SS.check(allow, own, statements).to_json()


def second_interior_reads(j):
    """`Pieces.secondInteriorReads`: of the second_statements piece's input (`_second_own`), the statements' reads that
    rest on the interior's own values (`second_stage.interior_reads`)."""
    _, own, statements = _second_own(j)
    return SS.interior_reads(own, statements or ())


def _second_own(j):
    """`Pieces.secondOwn`: of `j`, whether the two-stage form is allowed (`allow`), what round 2 holds of the run
    (`second_stage.Own`) and the statements (null for none). Round 2 holds the parts the interior `program`'s strata
    under `query` make for the replay `sample`, the `interior` and `boundary` records, the second stage's units `drawn`,
    the design's `links` and `wires` (each null for none derived) and the `rows` they place its units at (null for
    none), the reads the `verdicts` of record report (`audit.session_registered`) and what they name of the sessions
    they checked (`second_stage.verified_of`), and each template's reference header (`refs`, `{template: header}`, null
    for none) and the registered reads the params give it, taken as given (`ref_reads`, `{template: registered}`, each
    null for none, null for no template's)."""
    j = _json(j)
    if not (isinstance(j, dict) and isinstance(j.get("allow"), bool)):
        raise D.Reject("allow is not a boolean")
    if not (isinstance(j.get("sample"), list) and all(map(_nat, j["sample"]))):
        raise D.Reject("sample is not a list of replay blocks")
    if not (isinstance(j.get("drawn"), list) and all(map(_nat, j["drawn"]))):
        raise D.Reject("drawn is not a list of slots")
    desc, query = _obj(j.get("program"), "program"), _obj(j.get("query"), "query")
    links, wires, rows = _links_of(j.get("links")), _wires_of(j.get("wires")), _rows_of(j.get("rows"))
    verdicts = _obj(j.get("verdicts"), "verdicts")
    if not all(isinstance(v, dict) for v in verdicts.values()):
        raise D.Reject("a verdict is not an object")
    _registered_reads(verdicts)
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    try:
        strata = P.strata(PROGRAMS[sha], query)
    except (P.QueryRefused, ValueError) as e:
        raise D.Reject(str(e)) from None
    statements = j.get("statements")
    if not (statements is None or isinstance(statements, list)):
        raise D.Reject("statements is not a list")
    refs = j.get("refs")
    if not (refs is None or isinstance(refs, dict)):
        raise D.Reject("refs is not an object")
    ref_reads = j.get("ref_reads")
    if not (ref_reads is None or isinstance(ref_reads, dict)):
        raise D.Reject("ref_reads is not an object")
    own = SS.Own(tuple(j["sample"]), SS.parts(len(j["sample"]), strata), j.get("interior"), j.get("boundary"),
                 tuple(j["drawn"]), links, audit.session_registered(verdicts), SS.verified_of(verdicts),
                 tuple((refs or {}).items()), tuple((ref_reads or {}).items()), wires, rows)
    return j["allow"], own, statements


#: A two-template interior, PAIR's six slots then TRIPLE's four, its ports under their templates' names.
SECOND_MIXED = Setup(P.mixed_population_program([(PAIR, 6), (TRIPLE, 4)]), P.template_instances_query([PAIR.id, TRIPLE.id]))
PROGRAMS[SECOND_MIXED.sha512] = SECOND_MIXED.program
#: The boundary record round 2 reads: `BOUNDARY`'s ports at `BOUNDARY_ROOTS`, and registered values `w` and `w2`.
SECOND_BOUNDARY = {"scheme": SCHEME, "population": BOUNDARY.population, "registered": [
    {"value": "w", "root": "77" * 64, "domain": "78" * 64, "leaves": 8},
    {"value": "w2", "root": "79" * 64, "domain": "7a" * 64, "leaves": 8}],
    "roots": [{"port": p, "root": BOUNDARY_ROOTS[p], **BOUNDARY.ports[p]} for p in sorted(BOUNDARY.ports)]}


def _second_case(s: Setup, named: bool, sample: list[int], drawn: list[int], links=(), allow: bool = False,
                 own=(), wires=()) -> dict:
    """The second_statements piece's input for an honest run over interior `s` after the replay sample `sample` of
    `BOUNDARY`'s replay blocks: the interior record, of `s`'s program and partition in a served window that names the
    replay sample in order, with roots of each template's `out` and `x` (each `<template>/<port>` when `named`), a tree
    of the template's slots each; the second stage's units `drawn`, and each part's statement (`_second_statement`);
    the templates in `links` read their `x` from the boundary's `x` at their cells' replay blocks instead, unit `j` of
    a cell its row `j` mod 3 of the block's three (`_runs`), the design linking it and the interior then not
    registering it, those in `own` read it from their own `x`, each slot its own row, and each reader of `wires`,
    ``(reader, writer)`` pairs, reads it from its writer's `out`, each slot its writer's first slot in its cell, the
    design wiring it and the interior then not registering the reader's `x`, which its header holds at its writer's
    root and binding; the design's rows place each linked and wired unit so; the verifier reports those
    reads, and a verdict of
    record names each statement's session as it checked it (`_checked_by`); and each template's reference header is
    its part's (`_second_reference`) over the ports its session has, and its reference statement's registered reads
    are its statement's, which round 2 compares but for the run's fields (`second_stage.blank_reads`)."""
    q, strata, wired = len(sample), P.strata(s.program, s.query), dict(wires)
    parts = [SS.Part(t, lo, (hi - lo) // q) for t, ((lo, hi),) in strata]
    slots, n = {t: hi - lo for t, ((lo, hi),) in strata}, {p.template: p.n for p in parts}
    per = BOUNDARY.ports[BX]["leaves"] // BOUNDARY.population

    def port(t: str, p: str) -> str:
        return f"{t}/{p}" if named else p
    rec = {"scheme": SCHEME, "population": s.population, "program": s.sha512,
           "partition": P.digest(P.partition(s.program, s.query)),
           "window": {**WINDOW, "interior": True, "replay_sample": list(sample)}, "registered": [], "roots": [
               {"port": port(t, p), "root": hashlib.sha512(port(t, p).encode()).hexdigest(), "leaves": slots[t],
                "domain": hashlib.sha512(b"domain/" + port(t, p).encode()).hexdigest()}
               for t in s.templates for p in ("out", "x") if not (p == "x" and (t in links or t in wired))]}
    refs = {p.template: _second_reference(p.template, p.lo, q * p.n, [x for x in ("out", "x")
                                                                        if not (x == "x" and p.template in links)])
            for p in parts}
    statements, reported = [], {}
    for i, p in enumerate(parts):
        held = None
        if p.template in links:
            read = {"value": BX, "root": BOUNDARY_ROOTS[BX], **BOUNDARY.ports[BX], "reads_from": p.lo,
                    "reads": [sample[k // p.n] * per + k % p.n % per for k in range(q * p.n)]}
        elif p.template in own:
            x = next(r for r in rec["roots"] if r["port"] == port(p.template, "x"))
            read = {"value": x["port"], "root": x["root"], "domain": x["domain"], "leaves": x["leaves"],
                    "reads_from": p.lo, "reads": list(range(q * p.n))}
        elif p.template in wired:
            pw = next(w for w in parts if w.template == wired[p.template])
            x = next(r for r in rec["roots"] if r["port"] == port(pw.template, "out"))
            held = {"x": x}
            read = {"value": x["port"], "root": x["root"], "domain": x["domain"], "leaves": x["leaves"],
                    "reads_from": p.lo, "reads": [(k // p.n) * pw.n for k in range(q * p.n)]}
        else:
            statements.append(_second_statement(rec, p, q, drawn, named, k=i))
            continue
        s_ = _second_statement(rec, p, q, drawn, named, {"format": "verity/registered-reads/v1", "ports": {"x": read}},
                               k=i, held=held)
        s_["header"]["frame_v3"]["roots"]["x"], s_["header"]["frame_v3"]["domain_ids"]["x"] = read["root"], read["domain"]
        statements.append(s_)
        reported[f"{p.template}/x"] = {"value": read["value"], "root": read["root"]}
    lean = {"of_record": True, "accepted": True, **({"registered": reported} if reported else {})}
    return {"allow": allow, "sample": sample, "drawn": drawn, "program": s.desc, "query": s.query, "interior": rec,
            "boundary": SECOND_BOUNDARY, "statements": statements, "verdicts": {"lean": lean, **_checked_by(statements)},
            "links": [{"template": t, "port": "x", "value": BX} for t in sorted(links)],
            "wires": [{"template": t, "port": "x", "writer": w, "wport": "out"} for t, w in sorted(wired.items())],
            "rows": [{"template": t, "port": "x", "per": per, "runs": _runs(n[t], per)} for t in sorted(links)] + [
                {"template": t, "port": "x", "per": n[w], "runs": [[0, n[t], 0, 0]]} for t, w in sorted(wired.items())],
            "refs": refs, "ref_reads": {s_["template"]: s_["registered"] for s_ in statements}}


def _runs(n: int, per: int) -> list[list[int]]:
    """Runs placing units `0` to `n - 1` at rows `j` mod `per`, `[unit, count, row, step]` each."""
    return [[u, min(per, n - u), 0, 1] for u in range(0, n, per)]


def _shifted(ss, k: int):
    """Statements `ss` with each registered read moved `k` rows on."""
    if not isinstance(ss, list):
        return ss
    out = []
    for s in ss:
        r = s.get("registered") if isinstance(s, dict) else None
        ports = r.get("ports") if isinstance(r, dict) else None
        if isinstance(ports, dict):
            s = {**s, "registered": {**r, "ports": {
                p: {**e, "reads": [x + k if _int(x) else x for x in e["reads"]]}
                if isinstance(e, dict) and isinstance(e.get("reads"), list) else e for p, e in ports.items()}}}
        out.append(s)
    return out


def _shared_at(c: dict, i: int, ports, reads=()) -> dict:
    """`c` with its `i`-th statement's header, and its template's reference header, sharing rows at `ports`
    (``shared_rows``, a table of one row a slot each), and that statement, and its template's reference statement,
    reading each port of `reads` more from its own rows of the interior record, each slot its own row, which the
    verifier of record reports."""
    s = c["statements"][i]
    t, h, named = s["template"], s["header"], any("/" in r["port"] for r in c["interior"]["roots"])
    shared = dict.fromkeys(ports, h["instances"])
    more = {}
    for p in reads:
        x = next(r for r in c["interior"]["roots"] if r["port"] == (f"{t}/{p}" if named else p))
        more[p] = {"value": x["port"], "root": x["root"], "domain": x["domain"], "leaves": x["leaves"],
                   "reads_from": h["units"]["indices"][0], "reads": list(range(h["instances"]))}
    registered = s["registered"]
    if more:
        registered = {"format": "verity/registered-reads/v1", "ports": {**(registered or {}).get("ports", {}), **more}}
    lean = c["verdicts"]["lean"]
    reported = {**lean.get("registered", {}), **{f"{t}/{p}": {"value": e["value"], "root": e["root"]} for p, e in more.items()}}
    changed = _second_with(c, i, header={**h, "shared_rows": shared}, registered=registered)
    return {**changed, "refs": {**c["refs"], t: {**c["refs"][t], "shared_rows": shared}},
            "ref_reads": {**c["ref_reads"], t: registered},
            "verdicts": {**changed["verdicts"], "lean": {**lean, **({"registered": reported} if reported else {})}}}


def _shared_rows(c: dict, x: int) -> dict:
    """`c` with its first statement's header sharing rows (``shared_rows``) at its `frame_v3` ports, at the ports it
    reads registered, or at none, its template's reference sharing them too or not."""
    ss = c.get("statements")
    if not (isinstance(ss, list) and ss and isinstance(ss[0], dict) and isinstance(ss[0].get("header"), dict)):
        return c
    s, h = ss[0], ss[0]["header"]
    fv = h["frame_v3"] if isinstance(h.get("frame_v3"), dict) else {}
    r = s["registered"] if isinstance(s.get("registered"), dict) else {}
    ports = [fv["schemas"] if isinstance(fv.get("schemas"), dict) else {},
             r["ports"] if isinstance(r.get("ports"), dict) else {}, {}][x % 3]
    shared = dict.fromkeys(ports, h.get("instances"))
    out = {**c, "statements": [{**s, "header": {**h, "shared_rows": shared}}, *ss[1:]]}
    refs, t = c.get("refs"), s.get("template")
    if x % 2 == 0 and isinstance(refs, dict) and isinstance(t, str) and isinstance(refs.get(t), dict):
        out["refs"] = {**refs, t: {**refs[t], "shared_rows": shared}}
    return out


def _relinked(c: dict, x: int) -> dict:
    """`c` with the design's links underived (null), none, of another type, of a template not a string, naming another
    boundary port, or one more for a template of no part."""
    ls = c["links"] if isinstance(c.get("links"), list) else []
    other = [{**x_, "value": BOUT} for x_ in ls if isinstance(x_, dict)]
    more = [*ls, {"template": "Other_v0", "port": "x", "value": BX}]
    return {**c, "links": [None, [], 5, [{"template": 1, "port": "x", "value": BX}], other, more][x % 6]}


def _rewired(c: dict, x: int) -> dict:
    """`c` with the design's wires underived (null), none, of another type, of a writer not a string, each naming its
    writer's `x` instead, or one more for a template of no part."""
    ws = c["wires"] if isinstance(c.get("wires"), list) else []
    other = [{**w, "wport": "x"} for w in ws if isinstance(w, dict)]
    more = [*ws, {"template": "Other_v0", "port": "x", "writer": "Other_v0", "wport": "out"}]
    return {**c, "wires": [None, [], 5, [{"template": "t", "port": "x", "writer": 1, "wport": "out"}], other, more][x % 6]}


def _rerowed(c: dict, x: int) -> dict:
    """`c` with the design's rows none (null or empty), of another type, a run not of four naturals, each one more row
    a block, each first run's row one on, one unit fewer or one more, or its step another, one more for a template of
    no part, or in the other order."""
    rs = [r for r in c["rows"] if isinstance(r, dict)] if isinstance(c.get("rows"), list) else []

    def first(f):
        return [{**r, "runs": [f(*r["runs"][0]), *r["runs"][1:]]} if isinstance(r.get("runs"), list) and r["runs"]
                and isinstance(r["runs"][0], list) and len(r["runs"][0]) == 4 else r for r in rs]
    return {**c, "rows": [
        None, [], 5, [{"template": "t", "port": "x", "per": 1, "runs": [[0, 1, 0]]}],
        [{**r, "per": r["per"] + 1} if _nat(r.get("per")) else r for r in rs],
        first(lambda u, k, w, s: [u, k, w + 1, s]), first(lambda u, k, w, s: [u, max(k - 1, 1), w, s]),
        first(lambda u, k, w, s: [u, k + 1, w, s]), first(lambda u, k, w, s: [u, k, w, 1 - s if s < 2 else 0]),
        [*rs, {"template": "Other_v0", "port": "x", "per": 1, "runs": [[0, 1, 0, 0]]}], rs[::-1]][x % 11]}


def _reported(c: dict, x: int) -> dict:
    """`c` with the verifier of record reporting no reads, a read at another root or of no root, or verdicts that
    aren't."""
    vs = c["verdicts"] if isinstance(c.get("verdicts"), dict) else {}
    lean, checked = vs.get("lean"), {k: v for k, v in vs.items() if k.startswith("lean-verify/")}
    plain = {k: v for k, v in lean.items() if k != "registered"} if isinstance(lean, dict) else {
        "of_record": True, "accepted": True}
    return {**c, "verdicts": [
        {**checked, "lean": plain}, {**checked, "lean": {**plain, "registered": {"r": {"value": BX, "root": "00" * 64}}}},
        {**checked, "lean": {**plain, "registered": {"r": {"value": BX}}}}, 5, {"lean": 5}, {}][x % 6]}


def _unverified(c: dict, x: int) -> dict:
    """`c` with the verdict of record that checked a statement's session gone, not accepting, not of record, or naming
    another header, another or no registered reads, another sample, two tables or no table count."""
    vs = c["verdicts"] if isinstance(c.get("verdicts"), dict) else {}
    names = sorted(k for k in vs if k.startswith("lean-verify/"))
    if not names or not isinstance(vs[names[x % len(names)]], dict):
        return c
    name = names[x % len(names)]
    v = vs[name]
    changed = [None, {**v, "accepted": False}, {**v, "of_record": False}, {**v, "public_header_sha512": "00" * 64},
               {**v, "registered_sha512": "00" * 64}, {k: e for k, e in v.items() if k != "registered_sha512"},
               {**v, "unit_draw": {"law": "subset", "population": 1, "k": 0, "units": []}}, {**v, "session_tables": 2},
               {k: e for k, e in v.items() if k != "session_tables"}][(x // len(names)) % 9]
    return {**c, "verdicts": {k: e for k, e in vs.items() if k != name} if changed is None else {**vs, name: changed}}


def _referenced(c: dict, x: int) -> dict:
    """`c` with the reference headers none, or its first template's dropped, of another scheme, in another set (which
    round 2 places in the run), with a key more in `units`, a row schema more or another port's, or naming the run's
    fields (which round 2 blanks), or not an object."""
    refs = c["refs"] if isinstance(c.get("refs"), dict) else {}
    t = min(refs, default=None)
    if t is None or not isinstance(refs[t], dict):
        return c
    ref, rest = refs[t], {k: h for k, h in refs.items() if k != t}
    units = ref["units"] if isinstance(ref.get("units"), dict) else {}
    fv = ref["frame_v3"] if isinstance(ref.get("frame_v3"), dict) else {}
    schemas = fv["schemas"] if isinstance(fv.get("schemas"), dict) else {}
    ran = {**ref, "instances": 4, "units": {**units, "indices": [1], "blocks": [[1]]},
           "frame_v3": {**fv, "roots": {}, "domain_ids": {}}}
    changed = [{**ref, "scheme": "s"}, {**ref, "set": "served/r/m0"}, {**ref, "units": {**units, "more": [0]}},
               {**ref, "frame_v3": {**fv, "schemas": {**schemas, "y": SECOND_SCHEMAS["x"]}}},
               {**ref, "frame_v3": {**fv, "schemas": dict.fromkeys(schemas, SECOND_SCHEMAS["out"])}}, ran]
    return {**c, "refs": [None, rest, *({**rest, t: h} for h in changed), 5][x % 9]}


def _reordered(c: dict, x: int) -> dict:
    """`c` with the interior record's replay sample (`window.replay_sample`) reversed, one block fewer or more, of
    strings or booleans, or none, or its run another or none."""
    it = c.get("interior")
    w = it.get("window") if isinstance(it, dict) else None
    if not isinstance(w, dict):
        return c
    rs = w["replay_sample"] if isinstance(w.get("replay_sample"), list) else []
    window = [{**w, "replay_sample": rs[::-1]}, {**w, "replay_sample": rs[1:]}, {**w, "replay_sample": [*rs, x % 4]},
              {**w, "replay_sample": [str(u) for u in rs]}, {**w, "replay_sample": [True] * len(rs)},
              {k: v for k, v in w.items() if k != "replay_sample"}, {**w, "run": "r-other"}, {**w, "run": None}][x % 8]
    return {**c, "interior": {**it, "window": window}}


def _valued(c: dict, x: int) -> dict:
    """`c` with the reference statements' registered reads none, or its first template's dropped, null, of another
    format, reading another value at each port, reading a port more or none, or naming other roots, leaves and rows
    (which round 2 blanks, the rows where the run places them), or not an object."""
    refs = c["ref_reads"] if isinstance(c.get("ref_reads"), dict) else {}
    t = min(refs, default=None)
    if t is None:
        return c
    rest = {k: r for k, r in refs.items() if k != t}
    r = refs[t] if isinstance(refs[t], dict) else {"format": "verity/registered-reads/v1"}
    ports = r["ports"] if isinstance(r.get("ports"), dict) else {}

    def each(**kw) -> dict:
        return {p: {**e, **kw} if isinstance(e, dict) else e for p, e in ports.items()}
    changed = [None, {**r, "format": "verity/registered-reads/v0"}, {**r, "ports": each(value=BOUT)},
               {**r, "ports": {**ports, "z": {"value": "w", **dict.fromkeys(SS.READ_RUN_FIELDS)}}}, {**r, "ports": {}},
               {**r, "ports": each(root="00" * 64, leaves=3, reads=[0])}]
    return {**c, "ref_reads": [None, rest, *({**rest, t: h} for h in changed), 5][x % 9]}


def _read_registered(c: dict, x: int) -> dict:
    """`c` with its first statement reading, at a port `z` of no interior root, the boundary's registered value `w` or
    `w2`, slot 0 its first row, which `second-read` takes at a port no link names and the verifier of record reports,
    and its template's reference statement reading `w` or `w2` there, slot 0 its first row, slot 0 its second, or slot 1
    its first: which value a port reads, and at which rows, is the reference's (`second-values`), never the run's
    choice."""
    ss, refs, vs = c.get("statements"), c.get("ref_reads"), c.get("verdicts")
    if not (isinstance(ss, list) and ss and isinstance(ss[0], dict) and isinstance(ss[0].get("template"), str)
            and isinstance(refs, dict) and isinstance(vs, dict) and isinstance(vs.get("lean"), dict)):
        return c
    s, lean = ss[0], vs["lean"]
    value, named = ("w", "w2")[x % 2], ("w", "w2")[(x // 2) % 2]
    first, rows = ((0, [0]), (0, [1]), (1, [0]))[(x // 4) % 3]
    e = next(e for e in SECOND_BOUNDARY["registered"] if e["value"] == value)
    r = s["registered"] if isinstance(s.get("registered"), dict) else {"format": "verity/registered-reads/v1"}
    ports = r["ports"] if isinstance(r.get("ports"), dict) else {}
    registered = {**r, "ports": {**ports, "z": {**e, "reads_from": 0, "reads": [0]}}}
    reported = lean["registered"] if isinstance(lean.get("registered"), dict) else {}
    ref = {**r, "ports": {**ports, "z": {**e, "value": named, "reads_from": first, "reads": rows}}}
    changed = _second_with(c, 0, registered=registered)
    return {**changed, "ref_reads": {**refs, s["template"]: ref}, "verdicts": {**changed["verdicts"], "lean": {
        **lean, "registered": {**reported, f"{s['template']}/z": {"value": value, "root": e["root"]}}}}}


#: Values only the interior registers, in `_read_interior`: a history table `h`, and `opened`, which a read's `ok` port
#: reads: the word 1 a passing read's `ok` holds.
INTERIOR_VALUES = [{"value": "h", "root": "7b" * 64, "domain": "7c" * 64, "leaves": 8},
                   {"value": "opened", "root": "7d" * 64, "domain": "7e" * 64, "leaves": 8}]


def _read_interior(c: dict, x: int) -> dict:
    """`c` with its interior registering `INTERIOR_VALUES`, and its first statement reading `h` at port `z` or `opened`
    at port `ok`, neither an interior root, slot 0 its first row, as its template's reference statement reads it and
    the verifier reports it. By `x`, that read is at its root, domain and leaves, at another root, domain or
    leaf count, or of a value neither registration registers. Only the first passes `second-read`, which takes a value
    only the interior registers at the interior's entry."""
    ss, refs, vs, rec = c.get("statements"), c.get("ref_reads"), c.get("verdicts"), c.get("interior")
    if not (isinstance(ss, list) and ss and isinstance(ss[0], dict) and isinstance(ss[0].get("template"), str)
            and isinstance(refs, dict) and isinstance(vs, dict) and isinstance(vs.get("lean"), dict)
            and isinstance(rec, dict)):
        return c
    s, lean = ss[0], vs["lean"]
    e, port = ((INTERIOR_VALUES[0], "z"), (INTERIOR_VALUES[1], "ok"))[x % 2]
    change = (None, "root", "domain", "leaves", "value")[(x // 2) % 5]
    read = {**e, "reads_from": 0, "reads": [0]}
    if change is not None:
        read[change] = {"root": "00" * 64, "domain": "01" * 64, "leaves": 3, "value": "h3"}[change]
    r = s["registered"] if isinstance(s.get("registered"), dict) else {"format": "verity/registered-reads/v1"}
    ports = r["ports"] if isinstance(r.get("ports"), dict) else {}
    registered = {**r, "ports": {**ports, port: read}}
    reported = lean["registered"] if isinstance(lean.get("registered"), dict) else {}
    held = rec.get("registered") if isinstance(rec.get("registered"), list) else []
    changed = _second_with(c, 0, registered=registered)
    return {**changed, "interior": {**rec, "registered": [*held, *INTERIOR_VALUES]},
            "ref_reads": {**refs, s["template"]: registered}, "verdicts": {**changed["verdicts"], "lean": {
                **lean, "registered": {**reported, f"{s['template']}/{port}": {"value": read["value"],
                                                                               "root": read["root"]}}}}}


#: The root and domain of the value the interior registers under a link's name, in `_read_linked`.
LINKED_VALUE = {"root": "7f" * 64, "domain": "80" * 64}


def _read_linked(c: dict, x: int, kept: bool = False) -> dict:
    """`c` with its interior registering a value of its first link's name (`LINKED_VALUE`, the link's rows a block for
    each sampled block), the boundary record then holding no port of that name unless `kept`, and each statement of the
    linked template reading it at the linked port as round 2 places it, each slot its unit's row of its cell's sampled
    block, as the verifier of record reports it. By `x`, that read is as placed, at the boundary port, at another root,
    domain or leaf count, or at the boundary port's rows."""
    links, rows, sample, ss, b = c.get("links"), c.get("rows"), c.get("sample"), c.get("statements"), c.get("boundary")
    if not (isinstance(links, list) and links and isinstance(rows, list) and isinstance(sample, list)
            and isinstance(ss, list) and isinstance(b, dict)):
        return c
    link = links[0]
    per = next(r["per"] for r in rows if r["template"] == link["template"] and r["port"] == link["port"])
    entry = {"value": link["value"], **LINKED_VALUE, "leaves": len(sample) * per}
    out = c if kept else {**c, "boundary": {**b, "roots": [r for r in b["roots"] if r["port"] != link["value"]]}}
    reported = dict(c["verdicts"]["lean"].get("registered") or {})
    for k, s in enumerate(ss):
        if s.get("template") != link["template"]:
            continue
        e = s["registered"]["ports"][link["port"]]
        placed = {**e, **entry, "reads": [None if r is None else sample.index(r // per) * per + r % per
                                          for r in e["reads"]]}
        read = [placed, e, {**placed, "root": "00" * 64}, {**placed, "domain": "01" * 64},
                {**placed, "leaves": entry["leaves"] + 1}, {**e, **entry}][x]
        out = _second_with(out, k, registered={**s["registered"], "ports": {**s["registered"]["ports"],
                                                                              link["port"]: read}})
        reported[f"{s['template']}/{link['port']}"] = {"value": read["value"], "root": read["root"]}
    return {**out, "interior": {**out["interior"], "registered": [*(out["interior"].get("registered") or []), entry]},
            "verdicts": {**out["verdicts"], "lean": {**out["verdicts"]["lean"], "registered": reported}}}


def _shadowing(c: dict, x: int) -> dict:
    """`c` with its interior registering a value (`INTERIOR_VALUES`' first's root, domain and leaves) of a name the
    boundary record has, by `x` one of its ports' or of the values it registers."""
    rec, b = c.get("interior"), c.get("boundary")
    if not (isinstance(rec, dict) and isinstance(b, dict)):
        return c
    names = [e[k] for f, k in (("roots", "port"), ("registered", "value")) if isinstance(b.get(f), list)
             for e in b[f] if isinstance(e, dict) and isinstance(e.get(k), str)]
    if not names:
        return c
    held = rec.get("registered") if isinstance(rec.get("registered"), list) else []
    return {**c, "interior": {**rec, "registered": [*held, {**INTERIOR_VALUES[0], "value": names[x % len(names)]}]}}

#: A change to a second_statements input, its `x` drawn: a statement changed (`_restated`), reversed, one dropped, none,
#: a slot drawn more, the sample reversed, one replay block more or one fewer, a registered value's read by a port of
#: no interior root, every read moved on (checked as moved, or not), the design's links, wires or rows, the reads the
#: verifier reports, what a verdict of record names of a session it checked (`_unverified`), the reference
#: headers (`_referenced`), the reference statements' registered reads (`_valued`), a registered value read where the
#: reference reads it or another (`_read_registered`), a value only the interior registers read (`_read_interior`),
#: the interior registering a value of a name the boundary record has (`_shadowing`), the interior record's replay
#: sample or run (`_reordered`), or the rows a header shares (`_shared_rows`).
SECOND_CHANGES = [
    lambda c, x: {**c, "statements": _shifted(c["statements"], 1 + x % 3)},
    lambda c, x: _as_stated({**c, "statements": _shifted(c["statements"], 1 + x % 3)}),
    _relinked,
    _rewired,
    _rerowed,
    _reported,
    _unverified,
    _referenced,
    _valued,
    _read_registered,
    _read_interior,
    _shadowing,
    _reordered,
    _shared_rows,
    lambda c, x: {**c, "statements": _restated(c["statements"], x)},
    lambda c, x: {**c, "statements": c["statements"] and c["statements"][::-1]},
    lambda c, x: {**c, "statements": c["statements"] and c["statements"][1:]},
    lambda c, x: {**c, "statements": None},
    lambda c, x: {**c, "drawn": [*c["drawn"], x % (c["interior"]["population"] + 2)]},
    lambda c, x: {**c, "sample": c["sample"][::-1]},
    lambda c, x: {**c, "sample": [*c["sample"], 3 + x % 2]},
    lambda c, x: {**c, "sample": c["sample"][1:]},
    lambda c, x: {**c, "statements": c["statements"] and [
        {**s, "registered": {"ports": {"z": {**SECOND_BOUNDARY["registered"][0], "value": ["w", "v"][x % 2],
                                             "reads": [x % 4]}}}} if isinstance(s, dict) else s
        for s in c["statements"]]},
]


@st.composite
def second_statement_runs(draw) -> dict:
    """The second_statements piece's input: an honest run's (`_second_case`) over the one-template interior (`INTERIOR`,
    its ports unprefixed) or the two-template one (`SECOND_MIXED`), after a replay sample of two of `BOUNDARY`'s four
    replay blocks (or one, or three, which `SECOND_MIXED`'s four TRIPLE slots don't divide), the second stage's units one
    or none in each part's cell of each sampled replay block (or any slots), each part's `x` rooted, linked to the
    boundary, read from its own or wired to the other template's `out` (its own, with one template), and the form allowed
    or not; with up to two changes (`SECOND_CHANGES`), or a field of another type or missing."""
    s, named = draw(st.sampled_from([(INTERIOR, False), (SECOND_MIXED, True)]))
    sample = sorted(draw(st.sets(st.integers(0, 3), min_size=1, max_size=3)))
    q, strata = len(sample), P.strata(s.program, s.query)
    if draw(st.integers(0, 3)):
        drawn = [lo + i * ((hi - lo) // q) + u for _, ((lo, hi),) in strata for i in range(q)
                 if (u := draw(st.one_of(st.none(), st.integers(0, (hi - lo) // q - 1)))) is not None]
    else:
        drawn = draw(st.lists(st.integers(0, s.population + 1), max_size=5))
    kinds = {t: draw(st.sampled_from(["root", "link", "own", "wire"])) for t in s.templates}
    writer = {t: next((u for u in s.templates if u != t), t) for t in s.templates}
    c = _second_case(s, named, sample, drawn, {t for t, k in kinds.items() if k == "link"}, draw(st.booleans()),
                     {t for t, k in kinds.items() if k == "own"},
                     [(t, writer[t]) for t, k in kinds.items() if k == "wire"])
    for _ in range(draw(st.sampled_from([0, 0, 1, 1, 2]))):
        c = draw(st.sampled_from(SECOND_CHANGES))(c, draw(st.integers(0, 63)))
    if draw(st.integers(0, 4)) == 0:
        key, value = draw(st.sampled_from([
            ("allow", 1), ("allow", None), ("sample", [-1]), ("sample", "s"), ("drawn", [True]), ("drawn", None),
            ("program", []), ("query", None), ("query", {}), ("statements", 5), ("interior", 5), ("boundary", None),
            ("links", 5), ("wires", 5), ("wires", [{"template": "t", "port": "x", "writer": "w"}]), ("rows", 5),
            ("rows", [{"template": "t", "port": "x", "per": -1, "runs": []}]),
            ("rows", [{"template": "t", "port": "x", "per": 1, "runs": [[0, 1, True, 0]]}]),
            ("verdicts", None), ("verdicts", {"lean": 5}), ("refs", []), ("refs", "r"), ("ref_reads", []),
            ("ref_reads", "r"), (None, None)]))
        drop = draw(st.sampled_from(sorted(c)))
        c = {k: v for k, v in c.items() if k != drop} if key is None else {**c, key: value}
    return c


SECOND_STATEMENTS = D.Port(PIECES, "second_statements", second_statements,
                           st.one_of(second_statement_runs(), second_statement_runs(), JSON), package=PACKAGE)
#: The second_interior_reads piece's input: second_statements', and runs whose first statement reads a value only the
#: interior registers (`_read_interior`).
SECOND_INTERIOR_READS = D.Port(PIECES, "second_interior_reads", second_interior_reads, st.one_of(
    second_statement_runs(), st.builds(_read_interior, second_statement_runs(), st.integers(0, 9)), JSON),
    package=PACKAGE)

ID8 = bind(CompositeDefinition("CoreSpotId8", 0, (), lambda S: ((("y", Value(8)),), Value(8)),
                               lambda B, S, y: B.call(XOR8, y, y), register=False))
#: Replay blocks to derive round 2's links from (`second_stage.links_of`): `BLOCK`; PAIR over the first parameter then
#: `ID8` over PAIR's output, the second parameter unread; PAIR twice; a primitive over the parameter; two templates'
#: outputs returned as a tuple, the other way round, and as a record; and two fields returned, not each one template's
#: whole output: a parameter beside PAIR's, PAIR's twice about TRIPLE's, and one of a PAIR batch's two beside TRIPLE's.
REPLAYS = [BLOCK] + [bind(CompositeDefinition(f"CoreSpotReplay{name}", 0, (), sig, body, register=False))
                     for name, sig, body in (
    ("Chain", lambda S: ((("x", Array(2, Value(8))), ("w", Value(8))), Value(8)),
     lambda B, S, x, w: B.call(ID8, B.call(PAIR, x))),
    ("Twice", lambda S: ((("x", Array(2, Array(2, Value(8)))),), Value(8)),
     lambda B, S, x: B.call(XOR8, B.call(PAIR, x[0]), B.call(PAIR, x[1]))),
    ("Bare", lambda S: ((("x", Array(2, Value(8))),), Value(8)), lambda B, S, x: B.call(XOR8, x[0], x[1])),
    ("Two", lambda S: ((("x", Array(2, Value(8))), ("z", Array(3, Value(8)))), Tuple(Value(8), Value(8))),
     lambda B, S, x, z: tuple_of(B.call(PAIR, x), B.call(TRIPLE, z))),
    ("Swapped", lambda S: ((("x", Array(2, Value(8))), ("z", Array(3, Value(8)))), Tuple(Value(8), Value(8))),
     lambda B, S, x, z: tuple_of(*reversed((B.call(PAIR, x), B.call(TRIPLE, z))))),
    ("Named", lambda S: ((("x", Array(2, Value(8))), ("z", Array(3, Value(8)))), RecordType(a=Value(8), b=Value(8))),
     lambda B, S, x, z: record_of(a=B.call(PAIR, x), b=B.call(TRIPLE, z))),
    ("Kept", lambda S: ((("x", Array(2, Value(8))), ("w", Value(8))), Tuple(Value(8), Value(8))),
     lambda B, S, x, w: tuple_of(B.call(PAIR, x), w)),
    ("Again", lambda S: ((("x", Array(2, Value(8))), ("z", Array(3, Value(8)))), Tuple(Value(8), Value(8), Value(8))),
     lambda B, S, x, z: (lambda p: tuple_of(p, B.call(TRIPLE, z), p))(B.call(PAIR, x))),
    ("Split", lambda S: ((("x", Array(2, Array(2, Value(8)))), ("z", Array(3, Value(8)))), Tuple(Value(8), Value(8))),
     lambda B, S, x, z: tuple_of(B.batch(PAIR, x, axes=(0,))[0], B.call(TRIPLE, z))))]
#: A boundary program of each of `REPLAYS`, with its replay block's descriptor id, and `BLOCK`'s beside PAIR's, which
#: then calls PAIR twice.
#: Replay blocks to derive round 2's wires from (`second_stage.wires_of`) beside `REPLAYS`' (the chain's, ID8 reading
#: PAIR's output): ID8 batched twice over PAIR's output, a shared argument, each instance reading PAIR's one row; and
#: TRIPLE over PAIR's output beside the parameter's two, a node read with other rows.
WIRE_REPLAYS = [bind(CompositeDefinition(f"CoreSpotReplay{name}", 0, (), sig, body, register=False))
                for name, sig, body in (
    ("Shared", lambda S: ((("x", Array(2, Value(8))),), Array(2, Value(8))),
     lambda B, S, x: B.batch(ID8, B.call(PAIR, x), axes=(None,), n=2)),
    ("Beside", lambda S: ((("x", Array(2, Value(8))),), Value(8)),
     lambda B, S, x: B.call(TRIPLE, array_of([B.call(PAIR, x), x[0], x[1]]))))]
DUO = bind(CompositeDefinition("CoreSpotDuo", 0, (),
                               lambda S: ((("x", Array(2, Value(8))),), Tuple(Value(8), Value(8))),
                               lambda B, S, x: tuple_of(B.call(XOR8, x[0], x[1]), B.call(XOR8, x[1], x[0])),
                               register=False))
SWAP = bind(CompositeDefinition("CoreSpotSwap", 0, (), lambda S: ((("x", Array(2, Value(8))),), Array(2, Value(8))),
                                lambda B, S, x: array_of([B.call(ID8, x[1]), B.call(ID8, x[0])]), register=False))
#: Replay blocks of templates with several callers, to derive round 2's rows from (`second_stage.replay_rows`): PAIR
#: over each of two rows, in order and the other way round, its outputs returned in order; that, and ID8 over the
#: second's output, returned beside them; ID8 over the second output of DUO, a template of two; and, deriving none, PAIR
#: over two parameters, PAIR over a slice not one of its rows, an ID8 batch mapped over a PAIR batch's outputs, PAIR
#: over the parameter and over SWAP's output (a port one caller links and another wires), and PAIR over two ID8 outputs
#: (a port gathered from two writers); then DUO over each of two rows, its first outputs returned, its second, both as
#: two fields, and, deriving none, the first's first beside the second's second.
MULTI_REPLAYS = [bind(CompositeDefinition(f"CoreSpotReplay{name}", 0, (), sig, body, register=False))
                 for name, sig, body in (
    ("Pairs", lambda S: ((("x", Array(2, Array(2, Value(8)))),), Array(2, Value(8))),
     lambda B, S, x: array_of([B.call(PAIR, x[0]), B.call(PAIR, x[1])])),
    ("Crossed", lambda S: ((("x", Array(2, Array(2, Value(8)))),), Array(2, Value(8))),
     lambda B, S, x: array_of([B.call(PAIR, x[1]), B.call(PAIR, x[0])])),
    ("Fan", lambda S: ((("x", Array(2, Array(2, Value(8)))),), Tuple(Array(2, Value(8)), Value(8))),
     lambda B, S, x: (lambda p, q: tuple_of(array_of([p, q]), B.call(ID8, q)))(B.call(PAIR, x[0]), B.call(PAIR, x[1]))),
    ("Field", lambda S: ((("x", Array(2, Value(8))),), Value(8)), lambda B, S, x: B.call(ID8, B.call(DUO, x)[1])),
    ("Apart", lambda S: ((("x", Array(2, Value(8))), ("w", Array(2, Value(8)))), Array(2, Value(8))),
     lambda B, S, x, w: array_of([B.call(PAIR, x), B.call(PAIR, w)])),
    ("Askew", lambda S: ((("x", Array(4, Value(8))),), Value(8)), lambda B, S, x: B.call(PAIR, x[1:3])),
    ("Members", lambda S: ((("x", Array(2, Array(2, Value(8)))),), Array(2, Value(8))),
     lambda B, S, x: B.batch(ID8, B.batch(PAIR, x, axes=(0,)), axes=(0,))),
    ("Kinds", lambda S: ((("x", Array(2, Value(8))),), Array(2, Value(8))),
     lambda B, S, x: array_of([B.call(PAIR, x), B.call(PAIR, B.call(SWAP, x))])),
    ("Gathered", lambda S: ((("x", Array(2, Value(8))),), Value(8)),
     lambda B, S, x: B.call(PAIR, array_of([B.call(ID8, x[0]), B.call(ID8, x[1])]))),
    ("Heads", lambda S: ((("x", Array(2, Array(2, Value(8)))),), Array(2, Value(8))),
     lambda B, S, x: array_of([B.call(DUO, x[0])[0], B.call(DUO, x[1])[0]])),
    ("Tails", lambda S: ((("x", Array(2, Array(2, Value(8)))),), Array(2, Value(8))),
     lambda B, S, x: array_of([B.call(DUO, x[0])[1], B.call(DUO, x[1])[1]])),
    ("Halves", lambda S: ((("x", Array(2, Array(2, Value(8)))),), Tuple(Array(2, Value(8)), Array(2, Value(8)))),
     lambda B, S, x: (lambda a, b: tuple_of(array_of([a[0], b[0]]), array_of([a[1], b[1]])))(
         B.call(DUO, x[0]), B.call(DUO, x[1]))),
    ("Uneven", lambda S: ((("x", Array(2, Array(2, Value(8)))),), Array(2, Value(8))),
     lambda B, S, x: array_of([B.call(DUO, x[0])[0], B.call(DUO, x[1])[1]])))]
ID18 = PrimitiveDefinition("CoreSpotId18", 1, (("a", Value(8)),), Value(18), lambda a: a, register=False)
WIDE = bind(CompositeDefinition("CoreSpotWide", 0, (), lambda S: ((("x", Value(8)),), Value(18)),
                                lambda B, S, x: B.call(ID18, x), register=False))
PACK = bind(CompositeDefinition("CoreSpotPack", 0, (), lambda S: ((("x", Array(8, Value(8))),), Array(8, Value(18))),
                                lambda B, S, x: array_of([B.call(ID18, x[i]) for i in range(8)]), register=False))
#: Replay blocks whose units return fields wider than a word, to count what a replay block returns
#: (`second_stage.returned_bits`): four units of one 18-bit value each, and two of eight such values as one field.
WIDE_REPLAYS = [bind(CompositeDefinition(f"CoreSpotReplay{name}", 0, (), sig, body, register=False))
                for name, sig, body in (
    ("Wides", lambda S: ((("x", Array(4, Value(8))),), Array(4, Value(18))),
     lambda B, S, x: B.batch(WIDE, x, axes=(0,))),
    ("Packs", lambda S: ((("x", Array(2, Array(8, Value(8)))),), Array(2, Array(8, Value(18)))),
     lambda B, S, x: B.batch(PACK, x, axes=(0,))))]
REPLAY_PROGRAMS = [(P.population_program(r, 2), codec.descriptor_id(r)) for r in REPLAYS] + [
    (P.mixed_population_program([(BLOCK, 2), (PAIR, 3)]), codec.descriptor_id(BLOCK))] + [
    (P.population_program(r, 2), codec.descriptor_id(r)) for r in WIRE_REPLAYS + MULTI_REPLAYS + WIDE_REPLAYS]
PROGRAMS.update({PO.program_sha512(p): p for p, _ in REPLAY_PROGRAMS})


def replay_links(j):
    """`Pieces.replayLinks`: the design's links (`second_stage.replay_links`) from the replay block of descriptor id
    `replay` in `program`, each value the port of the `boundary` record; null for none."""
    desc = _obj(j.get("program") if isinstance(j, dict) else None, "program")
    if not isinstance(j.get("replay"), str):
        raise D.Reject("replay is not a string")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    links = SS.replay_links(PROGRAMS[sha], j["replay"], j.get("boundary"))
    return None if links is None else [{"port": x.port, "template": x.template, "value": x.value} for x in links]


@st.composite
def replay_link_inputs(draw) -> dict:
    """The replay_links piece's input: a program of `REPLAY_PROGRAMS`, the descriptor id of its replay block, of another
    of its definitions or of none, and a boundary record naming its ports by template, plainly, none, or not one; or a
    field of another type or missing."""
    prog, key = draw(st.sampled_from(REPLAY_PROGRAMS))
    key = draw(st.sampled_from([key, key, key, codec.descriptor_id(PAIR), codec.descriptor_id(XOR8), "Other_v0"]))
    rec = draw(st.one_of(
        st.sampled_from([{"roots": [{"port": f"{key}/x"}, {"port": f"{key}/o0"}]}, {"roots": [{"port": "x"}]}, None,
                         {"roots": 5}, {"roots": [{"port": 5}, {"port": "a/b"}]}]),
        st.lists(st.text("ab/", max_size=3), max_size=3).map(lambda ps: {"roots": [{"port": p} for p in ps]})))
    c = {"program": _json(codec.encode_program(prog)), "replay": key, "boundary": rec}
    change = draw(st.sampled_from([*[None] * 8, ("program", []), ("program", {}), ("replay", 5), ("replay", None),
                                   ("drop", None)]))
    if change is None:
        return c
    if change[0] == "drop":
        drop = draw(st.sampled_from(sorted(c)))
        return {k: v for k, v in c.items() if k != drop}
    return {**c, change[0]: change[1]}


REPLAY_LINKS = D.Port(PIECES, "replay_links", replay_links,
                      st.integers(0, 3).flatmap(lambda i: JSON if i == 3 else replay_link_inputs()), package=PACKAGE)


def replay_wires(j):
    """`Pieces.replayWires`: the design's wires (`second_stage.replay_wires`) from the replay block of descriptor id
    `replay` in `program`; null for none."""
    desc = _obj(j.get("program") if isinstance(j, dict) else None, "program")
    if not isinstance(j.get("replay"), str):
        raise D.Reject("replay is not a string")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    wires = SS.replay_wires(PROGRAMS[sha], j["replay"])
    return None if wires is None else [{"port": w.port, "template": w.template, "wport": w.wport, "writer": w.writer}
                                       for w in wires]


#: The replay_wires piece's input: replay_links' (`replay_link_inputs`), whose boundary record it doesn't read.
REPLAY_WIRES = D.Port(PIECES, "replay_wires", replay_wires,
                      st.integers(0, 3).flatmap(lambda i: JSON if i == 3 else replay_link_inputs()), package=PACKAGE)


def replay_rows(j):
    """`Pieces.replayRows`: the rows the design places each linked and wired unit at (`second_stage.replay_rows`), from
    the replay block of descriptor id `replay` in `program`; none for none."""
    desc = _obj(j.get("program") if isinstance(j, dict) else None, "program")
    if not isinstance(j.get("replay"), str):
        raise D.Reject("replay is not a string")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    return [r.to_json() for r in SS.replay_rows(PROGRAMS[sha], j["replay"])]


#: The replay_rows piece's input: replay_links' (`replay_link_inputs`), whose boundary record it doesn't read.
REPLAY_ROWS = D.Port(PIECES, "replay_rows", replay_rows,
                     st.integers(0, 3).flatmap(lambda i: JSON if i == 3 else replay_link_inputs()), package=PACKAGE)


def replay_returned(j):
    """`Pieces.replayReturned`: the bits a replay block returns (`second_stage.replay_returned`), the replay block of
    descriptor id `replay` in `program`; null for none."""
    desc = _obj(j.get("program") if isinstance(j, dict) else None, "program")
    if not isinstance(j.get("replay"), str):
        raise D.Reject("replay is not a string")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    return SS.replay_returned(PROGRAMS[sha], j["replay"])


#: The replay_returned piece's input: replay_links' (`replay_link_inputs`), whose boundary record it doesn't read.
REPLAY_RETURNED = D.Port(PIECES, "replay_returned", replay_returned,
                         st.integers(0, 3).flatmap(lambda i: JSON if i == 3 else replay_link_inputs()), package=PACKAGE)


def second_schedule(j):
    """`Pieces.secondSchedule`: why round 2 refuses `record`'s schedule (`schedule.failure`), null for none, and each of
    `blocks`' request and positions under it (`schedule.at`), null where no run holds it or the record commits no list
    of runs."""
    bs = j.get("blocks") if isinstance(j, dict) else None
    if not isinstance(bs, list):
        raise D.Reject("blocks is not a list")
    rec = j.get("record")
    w = rec.get("window") if isinstance(rec, dict) else None
    runs = SCH.runs_of(w.get("schedule") if isinstance(w, dict) else None)
    at = [SCH.at(runs, b) if runs is not None and _nat(b) else None for b in bs]
    return {"failure": SCH.failure(rec), "at": [None if x is None else list(x) for x in at]}


@st.composite
def schedule_inputs(draw) -> dict:
    """The second_schedule piece's input: a record whose window commits runs of requests `a` and `b` that are a schedule
    of its population, or that schedule changed (a run from another block or position, of no block or position, with a
    key added or a field of another type, the population another or not a number, the schedule or the window not one);
    and blocks from before the first to past the last."""
    runs, block, nxt = [], 0, {}
    for _ in range(draw(st.integers(0, 4))):
        r, n, length = draw(st.sampled_from("ab")), draw(st.integers(1, 3)), draw(st.integers(1, 4))
        runs.append({"request": r, "first_span": block, "spans": n, "first_position": nxt.get(r, 0), "length": length})
        block, nxt[r] = block + n, nxt.get(r, 0) + n * length
    rec = {"population": block, "window": {"schedule": runs}}
    change = draw(st.sampled_from([None] * 4 + ["population", "window", "span", "position", "zero", "extra", "type"]))
    if change == "population":
        rec["population"] = draw(st.sampled_from([block + 1, block - 1, str(block), None]))
    elif change == "window":
        rec["window"] = draw(st.sampled_from([None, 5, [], {}, {"schedule": None}, {"schedule": {}}, {"schedule": 5}]))
    elif change is not None and runs:
        k = draw(st.integers(0, len(runs) - 1))
        r = dict(runs[k])
        if change == "span":
            r["first_span"] += draw(st.sampled_from([-1, 1]))
        elif change == "position":
            r["first_position"] += draw(st.sampled_from([-1, 1]))
        elif change == "zero":
            r[draw(st.sampled_from(["spans", "length"]))] = 0
        elif change == "extra":
            r["extra"] = 0
        else:
            r[draw(st.sampled_from(sorted(r)))] = draw(st.sampled_from([True, "1", None, -1, 2]))
        runs[k] = r
    return {"record": rec, "blocks": draw(st.lists(st.integers(-1, block + 1), max_size=4))}


SECOND_SCHEDULE = D.Port(PIECES, "second_schedule", second_schedule,
                         st.integers(0, 3).flatmap(lambda i: JSON if i == 3 else schedule_inputs()), package=PACKAGE)
RUN = D.Port(PIECES, "run", run, st.one_of(
    runs(), runs(), runs(changes=1), runs(changes=1), runs(changes=2), st.builds(_as_rate, runs(), st.integers(0, 63)),
    runs([RECORD], 1), runs([PARAMS], 1), runs([ENTRIES, REGISTERED_PARAMS], 1), runs([EVIDENCE, VERDICTS_OF], 1),
    two_stage_runs(), two_stage_runs(1), two_stage_runs(1), two_stage_runs(2), JSON), package=PACKAGE)

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


def _reading(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _stream_state(s) -> dict:
    """A stream's state as `Stream.State.ofJson` reads it, a covered sample's order that isn't a natural number as null;
    rejected where it refuses."""
    _exactly(s, {"ended", "entries", "failures", "halted", "opened", "salted_at", "size"}, "a state")
    if not (_nat(s["size"]) and _nat(s["failures"]) and isinstance(s["opened"], list) and all(map(_nat, s["opened"]))
            and (s["salted_at"] is None or _reading(s["salted_at"])) and (s["ended"] is None or isinstance(s["ended"], dict))
            and (s["halted"] is None or isinstance(s["halted"], dict)) and isinstance(s["entries"], list)):
        raise D.Reject("a state's counts, salt, end, halt or entries are out of form")
    entries = []
    keys = {"delivered", "draw", "drawn_at", "final", "order", "proved", "registration", "session"}
    for e in s["entries"]:
        if not (isinstance(e, dict) and set(e) in (keys, keys | {"interior"}, keys | {"interior", "second"})
                and all(isinstance(e[k], str) for k in ("interior", "second") if k in e)):
            raise D.Reject("an entry's keys are not its own, with interior then second tokens under a two-stage rule")
        session = e["session"]
        if not (_nat(e["order"]) and isinstance(e["registration"], str) and (e["draw"] is None or isinstance(e["draw"], str))
                and _nat(e["proved"]) and (e["drawn_at"] is None or _reading(e["drawn_at"]))
                and all(isinstance(e[k], bool) for k in ("delivered", "final"))
                and (session is None or isinstance(session, list) and all(
                    isinstance(c, list) and len(c) == 3 and isinstance(c[1], str) and isinstance(c[2], str) for c in session))):
            raise D.Reject("an entry is out of form")
        entries.append({**e, "session": None if session is None else [[c[0] if _nat(c[0]) else None, *c[1:]] for c in session]})
    orders = [e["order"] for e in entries]
    if not (all(o < s["size"] for o in s["opened"] + orders) and orders == sorted(set(orders))):
        raise D.Reject("entries and opened orders are registrations', the entries in order, once each")
    return {**s, "entries": entries}


def stream(j):
    """The proof system stream's rules (`service._step`) on `{"params", "state", "op"}`: the decision without its
    `detail`, which the Lean doesn't write, and the state after it as `canonical_json` writes it. Rejected where the Lean
    refuses: the params as `Stream.Params.ofJson` reads them, the state as `_stream_state`, and an op `_step` raises on."""
    j = _json(j)
    _exactly(j, {"op", "params", "state"}, "the input")
    p = j["params"]
    keys = {"budget", "proof_deadline", "registration_deadline", "rounds", "stream"}
    if not (isinstance(p, dict) and (set(p) == keys or set(p) == keys | {"two_stage"} and p["two_stage"] is True)):
        raise D.Reject("params are not exactly their keys, with two_stage: true under a two-stage rule")
    rd, pd = p["registration_deadline"], p["proof_deadline"]
    if not (_nat(p["budget"]) and p["budget"] >= 1 and _nat(p["rounds"]) and 1 <= p["rounds"] <= service.ROUNDS
            and _reading(pd) and pd > 0 and (rd is None or _reading(rd) and rd > 0)
            and isinstance(p["stream"], str)):
        raise D.Reject("params are out of form")
    state = _stream_state(j["state"])
    try:
        d, after = service._step(p, state, j["op"])
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return {"decision": {k: v for k, v in d.items() if k != "detail"}, "state": codec.canonical_json(after).decode()}


STREAM_DIGEST = hashlib.sha512(b"stream").hexdigest()
SESSION_DIGEST = hashlib.sha512(b"session").hexdigest()
HALT_CAUSE = {"operator": "o", "stream": 1, "at": 4}
#: Readings, whole seconds (the harness writes no float; the service's own tests read the clock's floats through the
#: verifier), so the deadlines fall on both sides and on the edge.
READINGS = st.integers(0, 20)
#: The fields each op reads only once the rules ask for them (`unchecked` until then).
STREAM_LAZY = {"salt": {"at"}, "salt_released": {"released_at", "now"}, "register": {"received_at", "checks", "registration"},
               "select": {"checks", "draw", "proved", "at"}, "interiors": {"checks", "interior"},
               "second": {"checks", "second"}, "deliver": {"now"},
               "deliver_session": {"now"}, "outcome": {"now", "verdicts", "accepted"}}
#: Fields an op may carry out of form, with what the Lean errs or decides on.
MALFORMED = st.sampled_from(["1", True, None, -1, "x", [1], {"a": 1}])


def _one_in(draw, n: int) -> bool:
    """One time in `n`, evenly: `st.integers` samples its bounds more often."""
    return draw(st.sampled_from(range(n))) == 0


def _aim(draw, name: str, target: dict, entries: list, rounds: int) -> list | None:
    """The entries `name` on `target` reads made ready for it: sampled but not yet for `select`, open, under no session
    but `deliver_session`'s (and then a session of its own over consecutive rounds, all of them sampled and open),
    without an interior registration for `interiors`, mostly with one for `second` and sometimes with both stages for
    `deliver`, and delivered or not for `outcome`. A session's covered samples, or None."""
    def open_(e: dict, drawn: bool = True) -> None:
        e.update(draw=f"d{e['order']}" if drawn else None, proved=draw(st.integers(1, 3)) if drawn else 0,
                 drawn_at=draw(READINGS) if drawn else None, final=False, delivered=False, session=None)
    open_(target, name != "select")
    o = target["order"]
    target.pop("interior", None)
    target.pop("second", None)
    if name == "second" and not _one_in(draw, 4) or name == "deliver" and draw(st.booleans()):
        target["interior"] = f"i{o}"
        if name == "deliver":
            target["second"] = f"s{o}"
    if name == "outcome":
        target["delivered"] = draw(st.booleans())
    if name in ("session", "deliver_session"):
        o = target["order"]
        first = draw(st.integers(max(0, o - rounds + 1), o))
        covered = entries[first:min(len(entries), first + rounds)][:draw(st.integers(o - first + 1, rounds))]
        for e in covered:
            open_(e)
        draws = [[c["order"], f"r{c['order']}", f"d{c['order']}"] for c in covered]
        if name == "deliver_session":
            for e in covered:
                e["session"] = [list(d) for d in draws]
        return draws
    return None


@st.composite
def stream_cases(draw) -> dict:
    """A stream's params, one-stage or two-stage, a state of up to 6 registrations holding most of their entries (sampled
    or not, delivered, final, under a session of consecutive rounds, or under a two-stage rule with an interior
    registration and a second stage or not), and one op on it: mostly naming a held entry by its own
    tokens, or another's, or one not received; each field it reads present or not yet (`unchecked`), and sometimes one
    field out of form; sometimes the stream halted. Half the time aimed at the op: the stream open and every entry held,
    with the target's made ready for the op (`_aim`)."""
    name = draw(st.sampled_from([*service._OPS, *service._OPS[3:], "other"]))
    aim = draw(st.booleans())
    rounds = draw(st.integers(1, service.ROUNDS))
    budget = draw(st.integers(1, 4))
    params = {"budget": budget, "rounds": rounds, "registration_deadline": draw(st.sampled_from([None, 5, 2, 10])),
              "proof_deadline": draw(st.sampled_from([3, 7, 1])), "stream": STREAM_DIGEST}
    two_stage = draw(st.booleans()) if name in ("interiors", "second", "deliver", "session") else _one_in(draw, 4)
    if two_stage:
        params["two_stage"] = True
    size = draw(st.integers(1 if aim else 0, 6))
    entries = []
    for o in range(size):
        if not aim and _one_in(draw, 5):
            continue
        drawn = not _one_in(draw, 4)
        e = {"order": o, "registration": f"r{o}", "draw": f"d{o}" if drawn else None,
             "proved": draw(st.integers(0, 3)) if drawn else 0, "drawn_at": draw(READINGS) if drawn else None,
             "delivered": drawn and _one_in(draw, 4), "final": drawn and _one_in(draw, 4), "session": None}
        if two_stage and drawn and draw(st.booleans()):
            e["interior"] = f"i{o}"
            if draw(st.booleans()):
                e["second"] = f"s{o}"
        if not two_stage and drawn and draw(st.booleans()):
            first = draw(st.integers(max(0, o - rounds + 1), o))
            last = draw(st.integers(o, min(size - 1, first + rounds - 1)))
            e["session"] = [[k, f"r{k}", f"d{k}"] for k in range(first, last + 1)]
        entries.append(e)
    opened = sorted(draw(st.sets(st.integers(0, max(size - 1, 0)), max_size=rounds - aim + 1))) if size else []
    salted = None if aim and name.startswith("salt") else draw(st.sampled_from([None, None, 0, 1, 4]))
    ended = None if aim else draw(st.sampled_from([None, None, None, {"received_at": 9, "salted_at": salted}]))
    halted = None if aim else draw(st.sampled_from([None, None, None, HALT_CAUSE]))
    state = {"size": size, "opened": opened, "failures": draw(st.integers(0, budget - aim)), "salted_at": salted,
             "ended": ended, "halted": halted, "entries": entries}
    held = [e for e in entries if e["draw"] is not None] or entries
    target = draw(st.sampled_from(held)) if held and not _one_in(draw, 6) else None
    aimed = _aim(draw, name, target, entries, rounds) if aim and target else None
    o = target["order"] if target else draw(st.sampled_from([size, size + 1, None, -1, "0", True]))
    tokens = (f"r{o}", f"d{o}") if not _one_in(draw, 7) else ("r9", "d9")
    if aimed is not None and not _one_in(draw, 6):
        draws = aimed
    elif target and target["session"] and name in ("session", "deliver_session") and not _one_in(draw, 4):
        draws = [list(c) for c in target["session"]]
    else:
        start = o if _nat(o) else 0
        draws = [[k, f"r{k}", f"d{k}"] for k in range(start, start + draw(st.integers(0, rounds)))]
        if draws and _one_in(draw, 6):
            draws = draws[::-1]
    sessions = draw(st.sampled_from([[SESSION_DIGEST]] * 6 + [[], [SESSION_DIGEST + "\n"], ["x"], None, [SESSION_DIGEST, 1]]))
    fields = {
        "salt": {"at": draw(READINGS)},
        "salt_released": {"released_at": draw(READINGS), "now": draw(READINGS)},
        "register": {"received_at": draw(READINGS), "checks": draw(st.sampled_from([None, None, ["R3-roots"], []])),
                     "registration": f"r{size}"},
        "select": {"order": o, "registration": tokens[0], "checks": draw(st.sampled_from([None, None, ["draw-law"]])),
                   "draw": f"d{o}", "proved": draw(st.integers(0, 3)), "at": draw(READINGS)},
        "interiors": {"order": o, "registration": tokens[0], "draw": tokens[1],
                      "checks": draw(st.sampled_from([None, None, ["interior-population"]])), "interior": f"i{o}"},
        "second": {"order": o, "registration": tokens[0], "draw": tokens[1],
                   "interior": f"i{o}" if not _one_in(draw, 6) else "i9",
                   "checks": draw(st.sampled_from([None, None, ["draw-law"]])), "second": f"s{o}"},
        "session": {"draws": draws},
        "deliver": {"order": o, "registration": tokens[0], "draw": tokens[1], "sessions": sessions, "now": draw(READINGS),
                    **({"second": f"s{o}" if not _one_in(draw, 6) else "s9"} if two_stage else {})},
        "deliver_session": {"stream": draw(st.sampled_from([STREAM_DIGEST] * 5 + ["other"])), "draws": draws,
                            "sessions": sessions, "now": draw(READINGS)},
        "outcome": {"order": o, "registration": tokens[0], "draw": tokens[1], "now": draw(READINGS),
                    "verdicts": draw(st.sampled_from([None, None, None, ["pending"], ["sessions"]])),
                    "accepted": draw(st.booleans())},
        "halt": {"cause": draw(st.sampled_from([HALT_CAUSE] * 4 + [{}, None, "o"]))},
    }.get(name, {})
    op = {"op": name}
    for k, v in fields.items():
        if k in STREAM_LAZY.get(name, ()) and _one_in(draw, 8):
            continue
        op[k] = v
    if fields and _one_in(draw, 12):
        op[draw(st.sampled_from(sorted(fields)))] = draw(MALFORMED)
    change = draw(st.sampled_from(range(40)))
    if change == 0:
        params = {**params, draw(st.sampled_from(sorted(params))): draw(MALFORMED)}
    elif change == 1 and entries:
        state = {**state, "entries": entries[::-1] if len(entries) > 1 else [{**entries[0], "proved": -1}]}
    elif change == 2:
        state = {**state, draw(st.sampled_from(sorted(state))): draw(MALFORMED)}
    return {"params": params, "state": state, "op": op}


STREAM = D.Port(PIECES, "stream", stream, st.one_of(*[stream_cases() for _ in range(5)], JSON), package=PACKAGE)


def _fleet_state(s) -> dict:
    """A fleet's state as `Fleet.State.ofJson` reads it; rejected where it refuses."""
    _exactly(s, {"halted", "size", "streams"}, "a fleet's state")
    if not (_nat(s["size"]) and isinstance(s["streams"], list) and isinstance(s["halted"], list)):
        raise D.Reject("a fleet's size, streams or halts are out of form")
    for a in s["streams"]:
        _exactly(a, {"budget", "operator"}, "a stream")
        if not (isinstance(a["operator"], str) and _nat(a["budget"]) and a["budget"] >= 1):
            raise D.Reject("a stream is an operator and a positive budget")
    for h in s["halted"]:
        _exactly(h, {"cause", "operator"}, "a halt")
        if not (isinstance(h["operator"], str) and isinstance(h["cause"], dict)):
            raise D.Reject("a halt is an operator and its cause")
    return s


def fleet(j):
    """The proof system fleet's rules (`service._fleet_step`) on `{"state", "op"}`: the decision without its `detail`
    and the state after it as `canonical_json` writes it. Rejected where the Lean refuses: the state as `_fleet_state`,
    and an op `_fleet_step` raises on."""
    j = _json(j)
    _exactly(j, {"op", "state"}, "the input")
    state = _fleet_state(j["state"])
    try:
        d, after = service._fleet_step(state, j["op"])
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return {"decision": {k: v for k, v in d.items() if k != "detail"}, "state": codec.canonical_json(after).decode()}


OPERATORS = st.sampled_from(["o", "p", "q"])


@st.composite
def fleet_cases(draw) -> dict:
    """A fleet of up to 5 streams over three operators, some of them halted, and one op on it: mostly naming a stream it
    holds, its failures below, at or past its budget, and sometimes a field or the state out of form."""
    streams = draw(st.lists(st.fixed_dictionaries({"operator": OPERATORS, "budget": st.integers(1, 3)}), max_size=5))
    halted = [{"operator": o, "cause": {"operator": o, "stream": draw(st.integers(0, 4)), "at": draw(st.integers(0, 9))}}
              for o in draw(st.lists(OPERATORS, unique=True, max_size=2))]
    state = {"size": draw(st.integers(0, 9)), "streams": streams, "halted": halted}
    name = draw(st.sampled_from([*service._FLEET_OPS, "failures", "failures", "other"]))
    i = draw(st.integers(0, len(streams))) if not _one_in(draw, 8) else draw(st.sampled_from([-1, None, "0", True]))
    budget = streams[i]["budget"] if _nat(i) and i < len(streams) else 1
    fields = {"attach": {"operator": draw(OPERATORS), "budget": draw(st.integers(1, 3))},
              "register": {"stream": i},
              "failures": {"stream": i, "failures": draw(st.sampled_from([budget - 1, budget, budget, budget + 1, 0]))},
              }.get(name, {})
    op = {"op": name, **fields}
    if fields and _one_in(draw, 8):
        op[draw(st.sampled_from(sorted(fields)))] = draw(MALFORMED)
    change = draw(st.sampled_from(range(20)))
    if change == 0:
        state = {**state, draw(st.sampled_from(sorted(state))): draw(MALFORMED)}
    elif change == 1 and streams:
        state = {**state, "streams": [{**streams[0], "budget": 0}, *streams[1:]]}
    return {"state": state, "op": op}


FLEET = D.Port(PIECES, "fleet", fleet, st.one_of(fleet_cases(), fleet_cases(), JSON), package=PACKAGE)

@st.composite
def hidden_cases(draw, piece: str):
    """A width, a calls table, the honest one over sampled rows or any (overlapping, out of order, with gaps), and
    positions inside and past it; one field malformed or a key changed now and then."""
    width = draw(st.integers(1, 5)) if not _one_in(draw, 10) else draw(st.sampled_from([0, -1, True, "2", None]))
    rows = draw(st.lists(st.integers(0, 6), max_size=6))
    if draw(st.booleans()):
        calls = [list(c) for c in HD.table(width if _nat(width) and width else 1, rows)]
    else:
        calls = draw(st.lists(st.tuples(st.integers(0, 40), st.integers(0, 6)).map(list), max_size=6))
    end = max([o + m * (width if _nat(width) else 1) for o, m in calls], default=0) + 3
    case = {"hidden_offsets": {"rows": rows, "width": width},
            "hidden_locate": {"calls": calls, "positions": draw(st.lists(st.integers(0, end), max_size=8)), "width": width},
            "hidden_witness": {"call": draw(st.integers(0, len(calls) + 1)), "calls": calls,
                               "position": draw(st.integers(0, end)), "width": width}}[piece]
    if _one_in(draw, 10):
        key = draw(st.sampled_from(sorted(case)))
        case[key] = (draw(MALFORMED) if not isinstance(case[key], list) or not case[key] or _one_in(draw, 2)
                     else [*case[key][:-1], draw(st.sampled_from([-1, True, "1", [1], [1, -1], [True, 2], [1, 2, 3]]))])
    elif _one_in(draw, 30):
        del case[draw(st.sampled_from(sorted(case)))]
    elif _one_in(draw, 30):
        case["extra"] = 1
    return case


HIDDEN_OFFSETS = D.Port(PIECES, "hidden_offsets", hidden_offsets, st.one_of(*[hidden_cases("hidden_offsets")] * 4, JSON),
                        package=PACKAGE)
HIDDEN_LOCATE = D.Port(PIECES, "hidden_locate", hidden_locate, st.one_of(*[hidden_cases("hidden_locate")] * 4, JSON),
                       package=PACKAGE)
HIDDEN_WITNESS = D.Port(PIECES, "hidden_witness", hidden_witness, st.one_of(*[hidden_cases("hidden_witness")] * 4, JSON),
                        package=PACKAGE)

pieces = D.server_fixture(DECODE)


PIECE_RUNS = [(DECODE, 300), (PARTITION, 500), (COMMITTED_BITS, 300), (UNIT_GATES, 300), (FANOUT, 400),
              (REGISTRATION, 700), (REGISTERED_ENTRIES, 300), (DOMAIN, 300),
              (PRODUCED, 500), (KNOWN_PORT, 300),
              (CHECK_DRAW, 500), (DRAW, 500), (SELECT, 600), (SLOTS, 400), (FILE_POPULATION, 400), (TABLES_HELD, 300),
              (CHECK_PROVED, 300), (BOUND, 500),
              (BERNOULLI_BOUND, 500), (HIDDEN_OFFSETS, 300), (HIDDEN_LOCATE, 400), (HIDDEN_WITNESS, 400), (VERDICT, 300),
              (MATCHES_PUBLIC, 400), (SESSION_REGISTERED, 400), (RECORD_PORT, 600), (CHECK_PUBLIC, 500), (CHECK_VERDICTS, 400),
              (RUN, 500), (HELLO, 500), (STREAM, 800), (FLEET, 400), (SELECT_SECOND, 300), (SECOND_STATEMENTS, 400),
              (SECOND_INTERIOR_READS, 200), (REPLAY_LINKS, 300), (REPLAY_WIRES, 300), (REPLAY_RETURNED, 300),
              (SECOND_SCHEDULE, 300),
              (FITS, 400), (TEMPLATE_CIRCUITS, 300)]


@pytest.mark.parametrize("port, examples", PIECE_RUNS, ids=[p.fn for p, _ in PIECE_RUNS])
def test_a_piece_agrees(pieces, port, examples):
    stats = D.run(port, examples=examples, server=pieces)
    assert stats.outputs > examples // 4 and stats.rejects > 0, stats


def test_r8_reads_the_readers_python_does(pieces):
    """R8 past a population program, Lean's answers the Python's: two calls of `PAIR` on one pair of `x`, or a batch
    broadcasting it, share the run Input `x` at F 1, and the registration is refused (`R8-shared-input`); a call reading
    another's result, whose result a batch of two maps, joins all three at F 4, which the registration publishes; `x`
    named tied is let through, which the registration publishes where it was shared, and a tied name that isn't a
    parameter is refused (`R8-tied`); and `x` registered, as a registered input or as a parameter the verifier expects
    registered (`held`), is no run Input."""
    for s, f, shared in zip(READER_SETUPS, (1, 1, 4), ([0], [0], [])):
        case = {"program": s.desc, "query": _json(s.query)}
        assert fanout(case) == {"fanout": f, "shared": shared}
        assert D.replay(FANOUT, case, pieces) == ("output", D.canonical(fanout(case)))
        named, expanded = service.params_law("rate:1/2", s.program, s.query, None)
        rec = _record_of(s, None, named, WINDOW, leaf_layer=LEAF, work_table=expanded)
        given = {"params": {"anchors": [], "delta": "1/1000", "law": "rate:1/2", "leaf_layer": LEAF, "log": [],
                            "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": None,
                            "scheme": SCHEME, "work_table": None}, "record": rec}
        reg = registration(given)
        assert reg == {"codes": ["R8-shared-input"] if shared else [], "fanout": f}
        assert D.replay(REGISTRATION, given, pieces) == ("output", D.canonical(reg))
        sig = _json([[name, t.to_json()] for name, t in s.program.fn.signature()[0]])
        for tied, want in ((["x"], {"codes": [], "fanout": f, **({"rests_on_period_check": ["x"]} if shared else {})}),
                           (["nope"], {"codes": ["R8-tied", *(["R8-shared-input"] if shared else [])], "fanout": f})):
            tied_given = {**given, "params": {**given["params"], "parameters": sig, "tied": tied}}
            assert registration(tied_given) == want
            assert D.replay(REGISTRATION, tied_given, pieces) == ("output", D.canonical(want))
    program = _readers(2, (("call", 2, _X0 + _X00),) * 2, True)
    case = {"program": _json(codec.encode_program(program)), "query": _json(P.template_instance_query(PAIR.id))}
    assert D.replay(FANOUT, case, pieces) == ("output", D.canonical({"fanout": 1, "shared": []}))
    s = READER_SETUPS[0]
    case = {"program": s.desc, "query": _json(s.query), "held": [[g.start, len(g)] for g in G.held_gates(s.program, ["x"])]}
    assert case["held"] and fanout(case) == {"fanout": 1, "shared": []}
    assert D.replay(FANOUT, case, pieces) == ("output", D.canonical(fanout(case)))


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


def test_the_select_piece_draws_every_rule(pieces):
    """One sample under each rule `select_draw` samples (the stream's run refuses a Bernoulli sampling design and a closure
    map before it), on a partition whose `PAIR` stratum is two ranges apart, Lean's the Python's whole: `All` every unit,
    reading no coins, and each rate's sample (a work rate's with its closure) reading them, each stratum's count its k,
    which the sampled runs reach only probably, a rate law's a subset's of its count over the 10 units, a stratified or
    work rate's every stratum's k at least its rate of the stratum's units, and a two-stage rule's first stage its replay
    rate's subset. Then the same with every non-ASCII digit `str.isdigit` admits in place of an ASCII one, a rate whose
    count passes the population, a fixed count, as a law or a two-stage rule's replay law, and the retired two-stage
    forms, which all refuse."""
    desc = json.loads(_population(((0, 2), (1, 3), (0, 4), (2, 1))))
    query = _json(P.template_instances_query([t.id for t in TEMPLATES]))
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 2, "work": 1}, QUAD.id: 0}
    closure = [[u, [(u + 1) % 10, (u + 5) % 10]] for u in range(10)]
    source = hashlib.shake_256(b"select").hexdigest(1500)
    rules = [({"rule": "all"}, None), ({"law": "rate:2/5", "rule": "law"}, None), ({"law": "bernoulli:1/3", "rule": "law"}, None),
             ({"law": "stratified-rate:1/2", "rule": "law"}, None), ({"law": "work-rate:2/5", "rule": "law"}, closure),
             ({"law": "rate:1/3:2", "rule": "law"}, None),
             ({"grain": P.template_instance_query(PAIR.id), "k": 1, "replay": "rate:2/5", "rule": "two-stage"}, None),
             ({"grain": P.template_instance_query(PAIR.id), "k": 2, "replay": "rate:1/3:2", "rule": "two-stage"}, None)]
    rates = {"rate:1/3:2": 4, "rate:2/5": 4}                         # 2·⌈10/6⌉, ⌈20/5⌉
    strata = {"stratified-rate:1/2": ((1, 2), [3, 2, 1]), "work-rate:2/5": ((2, 5), [4, 2, 1])}
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
        if (rate := strata.get(rule.get("law"))) is not None:
            (num, den), ks = rate
            assert [s["k"] for s in out["draw"]["strata"]] == ks, (rule, out)
            assert all(den * s["k"] >= num * sum(hi - lo for lo, hi in s["units"]) for s in out["draw"]["strata"])
    retired = [{"check": "all", "replay": "subset:4", "rule": "two-stage"}, "two-stage:subset:4",
               {"grain": P.template_instance_query(PAIR.id), "replay": "subset:4", "rule": "two-stage", "k": 1,
                "check": "all"}]
    fixed = [*({"law": text, "rule": "law"} for text in ("subset:4", "stratified:5", "work:4")),
             *({"grain": P.template_instance_query(PAIR.id), "k": 1, "replay": replay, "rule": "two-stage"}
               for replay in ("subset:4", "stratified-rate:1/2", "work-rate:1/2"))]
    for rule in [*({"law": text, "rule": "law"} for text in ("subset:\u0663", "stratified:\u0665", "work:\u0664",
                                                              "bernoulli:\u0661/3", "bernoulli:1/\u0663", "rate:\u0661/3",
                                                              "rate:1/3:\u0662", "rate:9/10:4", "stratified-rate:\u0661/2",
                                                              "work-rate:1/\u0663")), *retired, *fixed]:
        case = {"closure": None, "program": desc, "query": query, "rule": rule, "source": source, "work_table": table}
        assert D.replay(SELECT, case, pieces)[0] == "reject", rule
        with pytest.raises(D.Reject):
            select(case)


def test_the_slots_follow_the_strata(pieces):
    """On the partition whose `PAIR` stratum is two ranges apart, Lean's slots are the Python's whole: the population's
    slots run through `PAIR`'s units, then `TRIPLE`'s, then `QUAD`'s, so units 2 to 8 move; the proof blocks' gates and
    the input class's are every gate of the program once; and the checked proof blocks are the sampled proof blocks' slots and
    every input slot, in slot order."""
    desc = json.loads(_population(((0, 2), (1, 3), (0, 4), (2, 1))))
    query = _json(P.template_instances_query([t.id for t in TEMPLATES]))
    case = {"closure": None, "program": desc, "query": query, "rule": {"law": "rate:2/5", "rule": "law"},
            "source": hashlib.shake_256(b"slots").hexdigest(400), "work_table": None}
    out = slots(copy.deepcopy(case))
    assert D.replay(SLOTS, case, pieces) == ("output", D.canonical(out))
    assert [[r["unit"] for r in s] for s in out["classes"][0]] == [[0, 1, 5, 6, 7, 8], [2, 3, 4], [9]]
    blocks = [b for c in out["classes"] for s in c for b in s]
    gates = sorted(g for b in blocks for lo, hi in b["pus"] for g in range(lo, hi))
    assert gates == list(range(PROGRAMS[PO.program_sha512(desc)].circuit.size))
    units = [b["unit"] for b in blocks]
    drawn = [units.index(u) for u in select(case)["draw"]["units"]]
    assert out["checked"] == [[v, 0] for v in sorted(drawn + [v for v, u in enumerate(units) if u is None])]


#: Units of the toy and the real-template populations `unit_gates` is held to its walk at: L5's population on one
#: GPU, and past it.
UNIT_GATES_AT = [147, 1001]


@pytest.mark.parametrize("n", UNIT_GATES_AT)
def test_the_unit_gates_by_range_are_the_walks(pieces, n):
    """At `n` units of `PAIR`, `TRIPLE` and `QUAD` at 1 : 6 : 3, Lean's unit gates laid out a range at a time are its walk's
    unit by unit and Python's (`unit_gates`), and its slots, which lay those gates out, are Python's (`slots`)."""
    members = tuple((t, max(1, round(n * w / 10))) for t, w in ((0, 1), (1, 6), (2, 3)))
    desc = json.loads(_population(members))
    query = _json(P.template_instances_query([t.id for t in TEMPLATES]))
    kind, out = D.replay(UNIT_GATES, {"program": desc, "query": query}, pieces)
    assert kind == "output"
    gates, walk = json.loads(out)
    assert len(gates) == n and gates == walk
    case = {"closure": None, "program": desc, "query": query, "rule": {"law": "rate:1/40", "rule": "law"},
            "source": hashlib.shake_256(b"slots").hexdigest(1024), "work_table": None}
    assert D.replay(SLOTS, case, pieces) == ("output", D.canonical(slots(copy.deepcopy(case))))


#: A served model's template mix, per template its units in the population: L5's seven, at their shares of its units.
REAL_TEMPLATES = [
    ('MoeRouterSigmoidBiasSlots_v2{E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"}', 32),
    ("EmbeddingRead_v1{V=256,H=3072}", 131072),
    ("EmbeddingRead_v1{V=256,H=96}", 1024),
    ('MoeExpertFp8BlockGemm_v4{K=6144,N=1,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v2"}}', 131072),
    ("EmbeddingRead_v1{V=256,H=4096}", 49152),
    ("EmbeddingRead_v1{V=256,H=4}", 12288),
    ('MoeExpertFp8BlockGemmW_v4{K=256,N=32,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v2"}}', 49152),
]


@pytest.mark.skipif(not settings.REAL_TEMPLATES,
                    reason="about 18 minutes and 34 GB: set VERITY_SPOT_REAL_TEMPLATES=1")
@pytest.mark.parametrize("n", UNIT_GATES_AT)
def test_the_unit_gates_by_range_are_the_walks_on_real_templates(pieces, n):
    """At `n` units of `REAL_TEMPLATES`, whose Definitions call one another as the catalog's do (the toy's call nothing),
    Lean's unit gates laid out a range at a time are its walk's unit by unit. Python's `partition.unit_gates` isn't asked:
    its partition over these templates takes more memory than a suite has."""
    from verity.circuits.population import mixed_population_program
    from verity.definitions.registry.catalog import load
    from verity.commitments.windows import interface as IF

    load()
    total = sum(c for _t, c in REAL_TEMPLATES)
    counts = [max(1, n * c // total) for _t, c in REAL_TEMPLATES]
    counts[counts.index(max(counts))] += n - sum(counts)
    templates = [IF.resolve(t) for t, _c in REAL_TEMPLATES]
    program = mixed_population_program(list(zip(templates, counts)))
    case = {"program": _json(codec.encode_program(program)), "query": _json(PO.template_instances_query(templates))}
    kind, out = pieces.call("unit_gates", case)
    assert kind == "output", out
    gates, walk = json.loads(out)
    assert len(gates) == n and gates == walk


def test_the_call_queries_cut_call_by_call(pieces):
    """On a program of root Calls, Lean's slots are the Python's whole under `Q_word` at `X = 16` and `Q_call` at `X = 8`:
    the strata are the Calls' Definitions, `FORK`'s `Q_word` units two output units and its committed interior value, the
    first's gate span holding the second's; both refuse the widths one-stage doesn't apply, and agree on a Call whose
    `Q_word` unit commits 32 bits, in the 32-bit class, which `Q_call` at `X = 16` refuses."""
    desc = json.loads(_calls((0, 2, 0, 3)))
    case = {"closure": None, "program": desc, "rule": {"law": "rate:1/4", "rule": "law"},
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
    taken = {**case, "program": wide, "query": _json(PO.query())}
    assert D.replay(SLOTS, taken, pieces) == ("output", D.canonical(slots(copy.deepcopy(taken))))
    for program, query in [(desc, PO.call_query(4)), (desc, PO.call_query(32)), (desc, PO.query(X=8)),
                           (desc, PO.query(version=2)), (wide, PO.call_query(16))]:
        refused = {**case, "program": program, "query": _json(query)}
        assert D.replay(SLOTS, refused, pieces)[0] == "reject", query
        with pytest.raises(D.Reject):
            slots(copy.deepcopy(refused))


def _keyed(key: str, first: bool = False) -> dict:
    """The descriptor of a root of one instance of `PAIR` and one Call of a composite of one gate whose id is the input id
    `key`, the Call the later node or, `first`, the earlier; its program put among `PROGRAMS`."""
    name, version = key.rsplit("_v", 1)
    keyed = bind(CompositeDefinition(name, int(version), (), lambda S: ((("y", Value(8)),), Value(8)),
                                     lambda B, S, y: B.call(XOR8, y, y), register=False))
    assert codec.descriptor_id(keyed) == key

    def body(B, S, x, y):
        calls = [lambda: B.call(keyed, y), lambda: B.call(PAIR, x)]
        return tuple_of(*(call() for call in (calls if first else calls[::-1])))

    program = Program(bind(CompositeDefinition(
        f"CoreSpotKeyed{name}{'First' if first else ''}", 0, (),
        lambda S: ((("x", Array(2, Value(8))), ("y", Value(8))), Tuple(Value(8), Value(8))), body, register=False)))
    PROGRAMS[PO.program_sha512(program)] = program
    return _json(codec.encode_program(program))


@pytest.mark.parametrize("key, first", [("Input16_v1", False), ("Input16_v1", True), ("RegisteredInput16_v1", False)])
def test_a_composite_whose_id_is_an_input_id_is_no_input(pieces, key, first):
    """A root Call of a composite whose id is an input id is no input node (`program.is_input`: only a primitive is one):
    its gate is computed. A template query that doesn't list the composite refuses the program, Lean with the Python's
    reason, in `unit_gates`, `slots` and `select`. One that lists it, and `Q_call` at `X = 16`, which cuts every Call,
    take the Call as a unit, and Lean's slots are the Python's whole: the Call's gate is its unit's and no input block's,
    so the proof blocks' gates and the input class's are every gate of the program once."""
    desc = _keyed(key, first)
    case = {"closure": None, "program": desc, "source": hashlib.shake_256(b"keyed").hexdigest(400), "work_table": None}
    one = _json(P.template_instance_query(PAIR.id))
    refused = {**case, "query": one, "rule": {"rule": "all"}}
    for port, reference, j in [(UNIT_GATES, unit_gates, {"program": desc, "query": one}), (SLOTS, slots, refused),
                               (SELECT, select, refused)]:
        with pytest.raises(D.Reject) as why:
            reference(copy.deepcopy(j))
        assert f"is a call of '{key}'" in str(why.value)
        assert D.replay(port, j, pieces) == ("reject", f"query-refused: {why.value}")
    size = PROGRAMS[PO.program_sha512(desc)].circuit.size
    for query in (P.template_instances_query([PAIR.id, key]), PO.call_query(16)):
        taken = {**case, "query": _json(query), "rule": {"law": "rate:1/2", "rule": "law"}}
        out = slots(copy.deepcopy(taken))
        assert D.replay(SLOTS, taken, pieces) == ("output", D.canonical(out))
        blocks = [b for c in out["classes"] for s in c for b in s]
        assert sorted(g for b in blocks for lo, hi in b["pus"] for g in range(lo, hi)) == list(range(size))
        assert sorted(b["unit"] for b in blocks if b["unit"] is not None) == [0, 1]


XOR64 = PrimitiveDefinition("CoreSpotXor64", 1, (("a", Value(64)), ("b", Value(64))), Value(64),
                            lambda a, b: a ^ b, register=False)
#: A template of one batch of `N` 64-bit XORs: an instance returns `64 N` bits.
WIDE = CompositeDefinition("CoreSpotWide", 0, ("N",),
                           lambda S: ((("x", Array(S.N, Value(64))), ("y", Array(S.N, Value(64)))), Array(S.N, Value(64))),
                           lambda B, S, x, y: B.batch(XOR64, x, y, axes=(0, 0)), register=False)


def _template_population(template, n: int) -> dict:
    """The descriptor of `n` instances of `template`, its program put among `PROGRAMS`."""
    program = P.population_program(template, n)
    PROGRAMS[PO.program_sha512(program)] = program
    return _json(codec.encode_program(program))


def test_a_unit_wider_than_the_widest_class_or_a_template_of_no_gate_is_refused(pieces):
    """Lean refuses as the Python does a template instance of 262,208 bits, one class past the widest, and a population
    with an instance of `IDENTITY`, of no gate; both take an instance of 262,144 bits, the widest class (the measured
    widths' `Gemm_v2{K=2048,N=16384}`), and `SHARED` under `Q_call` at `X = 16`, whose first unit passes `Q_call`'s width
    rule with 16 returned bits while it holds a constant a later unit reads, 24 bits in the 32-bit class, and at `X = 8`,
    where that constant is in no unit, and agree on their slots."""
    case = {"closure": None, "rule": {"law": "rate:1/1000", "rule": "law"},
            "source": hashlib.shake_256(b"refused").hexdigest(400), "work_table": None}
    shared = json.loads(_calls((0, 5)))
    widest, over = bind(WIDE, N=4096), bind(WIDE, N=4097)
    for program, query in [(_template_population(over, 1), P.template_instance_query(over.id)),
                           (json.loads(_population(((0, 1), (4, 2)))), P.template_instances_query([PAIR.id, IDENTITY.id]))]:
        refused = {**case, "program": program, "query": _json(query)}
        assert D.replay(SLOTS, refused, pieces)[0] == "reject", query
        with pytest.raises(D.Reject):
            slots(copy.deepcopy(refused))
    for program, query in [(_template_population(widest, 1), P.template_instance_query(widest.id)),
                           (shared, PO.call_query(16)), (shared, PO.call_query(8))]:
        taken = {**case, "program": program, "query": _json(query)}
        assert D.replay(SLOTS, taken, pieces) == ("output", D.canonical(slots(copy.deepcopy(taken)))), query


R7_CODES = {"R7-registered", "R7-values", "R7-root", "R7-leaves", "R7-words", "R7-domain", "R7-word_bits", "R7-held",
            "R7-supplied"}


def _v2_audit(s: RegisteredSetup, values: dict) -> dict:
    """One valid v2 audit of `s` expecting `values` registered, the log's earlier record registering them under the same
    roots, by a verifier bound to `BINDING` that holds the values it doesn't fill across windows (`held`)."""
    law = {"k": 1, "law": "subset"}
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    earlier = R.record(s.program, s.query, law, SCHEME, roots, EARLIER, leaf_layer=LEAF, registered=s.entries(values))
    rec = _record_of(s, values, law, {**WINDOW, **BINDING}, [earlier], BINDING, leaf_layer=LEAF, previous=R.digest(earlier))
    params = {"anchors": [], "binding": BINDING, "delta": "1/10", "law": law, "leaf_layer": LEAF, "log": [earlier],
              "held": sorted(set(values) - set(s.supplied(values) or {})), "parameters": s.signature,
              "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": values,
              "scheme": SCHEME, "work_table": None, **({"supplied": s.supplied(values)} if s.supplied(values) else {}),
              **({"row_schemas": {v: f for v, f in s.formats.items() if v in values}} if s.formats else {})}
    evidence = {"draw": {**law, "population": s.population, "units": [0]}, "instances": 1, "proved_units": [0],
                "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                "sessions": [], "statement": {},
                "verdicts": {"lean": {"accepted": True, "of_record": True, "registered": reads(rec)}}}
    return _json({"evidence": evidence, "params": params, "record": rec})


def test_every_r7_code_is_reached(pieces):
    """The record's and the parameters' registered changes on v2 audits of every program, Lean and Python agreeing on each:
    together they reach every R7 code, which the sampled runs only probably do, and the unchanged audits none."""
    seen: set[str] = set()
    for s in REGISTERED_SETUPS:
        for values in s.values[:2]:
            a = _v2_audit(s, values)
            honest = registration(_REGISTRATION_OF(a))
            assert honest["codes"] == []
            assert D.replay(REGISTRATION, _REGISTRATION_OF(a), pieces) == ("output", D.canonical(honest))
            for change, x in ((c, x) for c in ENTRIES + REGISTERED_PARAMS + SUPPLIED for x in range(8)):
                b = copy.deepcopy(a)
                try:
                    change(b, x)
                except (KeyError, TypeError, IndexError, ValueError, AttributeError):
                    continue
                kind, out = D.replay(REGISTRATION, _json(_REGISTRATION_OF(b)), pieces)
                seen |= set(json.loads(out)["codes"]) if kind == "output" else set()
    assert R7_CODES <= seen, sorted(R7_CODES - seen)


def test_every_produced_code_is_reached(pieces):
    """Each named change to an honest produced record, Lean and Python agreeing on its codes: each reaches its own code,
    and the honest record none, so every code `produced.check`, `produced.opened` and `produced.shared` give is reached."""
    honest = _produced_case(["partial", "PouwUnitSeeds_v2"], [(32, 16, 16), (1, 64, 16)], ["a1" * 64, "b2" * 64])
    clean = {"codes": [], "opened": [], "shared": []}
    assert D.replay(PRODUCED, honest, pieces) == ("output", D.canonical(clean))
    for name, change, code in PRODUCED_CHANGES:
        c = copy.deepcopy(honest)
        change(c, 0)
        kind, out = D.replay(PRODUCED, _json(c), pieces)
        got = json.loads(out)
        assert kind == "output" and (code in [*got["codes"], *(got["opened"] or []), *(got["shared"] or [])] if code
                                     else got == clean), (name, kind, out)


def test_a_known_port_is_the_one_row_of_its_tree(pieces):
    """`produced.known_port`'s root and opening, Lean and Python agreeing, are the one row of a produced tree committed
    at that content and salt, and another content's are another root."""
    j = {"program": "p", "value": "ok", "row": [1], "salt": KNOWN_SALT, "registration": PRODUCED_REG, "word_bits": 16}
    t = PD.commit("p", "ok", [[1]], PRODUCED_REG, [bytes(range(192))], schema=G.SCHEMA_V2)
    assert D.replay(KNOWN_PORT, j, pieces) == ("output", D.canonical({"root": t.root.hex(), "bc": t.bcs[0].hex()}))
    assert json.loads(D.replay(KNOWN_PORT, {**j, "row": [0]}, pieces)[1])["root"] != t.root.hex()


def test_a_bit_row_values_rows_are_counted_in_its_format(pieces):
    """R7 counts a value's rows in the row format the verifier names for it: `A`'s 8-bit leaves, two to a u16 word in its
    `row/v2` rows, and `B`'s `row-seg/v1` rows are accepted under their formats (or each other's, which count alike), and
    without them `A` is counted as `row/v1` rows of one leaf a word, twice its rows under another domain (`R7-leaves`,
    `R7-domain`), Lean and Python agreeing."""
    s = next(s for s in REGISTERED_SETUPS if s.formats)
    for values in s.values:
        a = _v2_audit(s, values)

        def codes(b):
            kind, out = D.replay(REGISTRATION, _json(_REGISTRATION_OF(b)), pieces)
            return json.loads(out)["codes"] if kind == "output" else kind

        assert codes(a) == [], values
        swapped = copy.deepcopy(a)
        swapped["params"]["row_schemas"] = {v: {G.SCHEMA_V2: G.SCHEMA_SEG, G.SCHEMA_SEG: G.SCHEMA_V2}[f]
                                            for v, f in a["params"]["row_schemas"].items()}
        assert codes(swapped) == [], values
        for drop in (lambda b: b["params"].pop("row_schemas"), lambda b: b["params"].update(row_schemas={})):
            b = copy.deepcopy(a)
            drop(b)
            assert codes(b) == (["R7-leaves", "R7-domain"] if "A" in values else []), values


#: Pearl+'s y trees' binding tag: PoUW's rows domain over a call's served y, at salt nonce 2^61 + the call's index
PEARL_Y_ROWS = "verity/pouw/pearl-plus/y-rows/v1"


def test_an_output_reuse_takes_a_pearl_plus_domain_verbatim(pieces):
    """`T` reusing (`outputs`) Pearl+'s window record's entry for a call's A rows or served y, which PoUW commits
    (`audit.commit_rows` under hm96-sha512) under its rows domain, not one-stage's: `T`'s 32-bit rows as `row-seg/v1`
    rows, or `row/v2` rows (which R7 counts alike), are accepted under that domain and root verbatim, and without
    `outputs` are `R7-domain`. The record's entry under one-stage's own domain for `T` (`registered.domain`, whichever
    binding it takes) is `R7-domain`, under another root `R7-held`, and Pearl+'s entry of 16-bit words or of other rows
    `R7-output`, Lean and Python agreeing."""
    s, values, law = REGISTERED_SETUPS[1], {"T": 2}, {"k": 1, "law": "subset"}
    mine = s.committed["T", 2].entry()
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    salt, u = bytes(range(64)), 7

    def codes(b):
        kind, out = D.replay(REGISTRATION, _json(b), pieces)
        return json.loads(out)["codes"] if kind == "output" else kind

    for name, binding, nonce in (("pearl-a/0/0", PA.activation_binding(salt, u), u),
                                 ("pearl-y/0/0", tagged(PEARL_Y_ROWS, salt, u), (1 << 61) + u)):
        tree = PA.commit_rows(binding, u, [b"".join(w.to_bytes(4, "little") for w in r) for r in s._rows("T", 2)[0]],
                              PA.HM96, salt_key=bytes(range(32)), nonce=nonce)
        theirs = {**mine, "value": name, "root": tree.commitment.root.hex(), "domain": tree.domain.domain_id.hex()}
        assert theirs["domain"] != mine["domain"]

        def case(entry=(), earlier=theirs, drop=(), **change):
            log = [R.record(s.program, s.query, law, SCHEME, roots, EARLIER, leaf_layer=LEAF, registered=[earlier])]
            rec = R.record(s.program, s.query, law, SCHEME, roots, {**WINDOW, **BINDING}, leaf_layer=LEAF,
                           previous=R.digest(log[0]),
                           registered=[{**mine, "root": theirs["root"], "domain": theirs["domain"], **dict(entry)}])
            params = {"anchors": [], "binding": BINDING, "delta": "1/10", "law": law, "leaf_layer": LEAF, "log": log,
                      "outputs": {"T": {"record": R.digest(log[0]), "value": name}}, "parameters": s.signature,
                      "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": values,
                      "row_schemas": {"T": G.SCHEMA_SEG}, "scheme": SCHEME, "work_table": None, **change}
            return {"params": {k: v for k, v in params.items() if k not in drop}, "record": rec}

        assert codes(case()) == [], name
        assert codes(case(row_schemas={"T": G.SCHEMA_V2})) == [], name
        assert codes(case(drop=["outputs"])) == ["R7-domain"], name
        assert codes(case({"domain": mine["domain"]})) == ["R7-domain"], name
        assert codes(case({"root": "ef" * 64})) == ["R7-held"], name
        for e in ({k: v for k, v in theirs.items() if k != G.WIDTH_KEY}, {**theirs, "leaves": 4, "words": 1}):
            assert codes(case(earlier=e))[0] == "R7-output", (name, e)


def test_a_tile_replayed_across_streams_in_one_window_is_refused(pieces):
    """The supplied setup whose seeds read the window's salt and positions, bound to one stream and prover: its record, as
    another stream's (or prover's) verifier checks it in the same window, is refused as it stands (`R6-stream` or
    `R6-prover`, and `R7-supplied`) and relabeled to that verifier's binding (`R7-supplied`: its seeds are the first
    stream's), and by a verifier with no binding (`R7-supplied`), Lean and Python agreeing; every binding change reaches
    those codes or a refused binding, and nothing else."""
    s = next(s for s in SUPPLIED_SETUPS if _reads_window(s.supplied(s.values[0])))
    a = _v2_audit(s, s.values[0])

    def codes(b):
        kind, out = D.replay(REGISTRATION, _json(_REGISTRATION_OF(b)), pieces)
        return json.loads(out)["codes"] if kind == "output" else kind

    assert codes(a) == []
    for k, other in (("stream", "3d" * 64), ("prover", "d1" * 32)):
        b = copy.deepcopy(a)
        b["params"]["binding"] = {**BINDING, k: other}
        assert codes(b) == [f"R6-{k}", "R7-supplied"], k
        b["record"]["window"] = {**b["record"]["window"], k: other}
        assert codes(b) == ["R7-supplied"], k
    b = copy.deepcopy(a)
    del b["params"]["binding"]
    assert codes(b) == ["R7-supplied"]
    seen: set[str] = set()
    for change, x in ((c, x) for c in BINDINGS for x in range(16)):
        b = copy.deepcopy(a)
        change(b, x)
        got = codes(b)
        seen |= {got} if isinstance(got, str) else set(got)
    assert seen == {"R6-stream", "R6-prover", "R7-supplied", "reject"}, seen


def test_a_second_window_registers_its_values_afresh_and_a_replay_is_refused(pieces):
    """Lean and Python agreeing: a verifier naming its window's open root (`WROOT`) and holding W across windows
    accepts a record whose D is a fresh commitment under its window's domain after a log that registered D under an
    earlier window's (`EARLIER_WROOT`) at another root, and W under the same root. The earlier window's D, replayed, is
    refused (`R7-domain`), and so is a W under another root (`R7-held`); a verifier that doesn't hold W takes it per
    window (`R7-domain`), and one that names no window root takes D's domain as before (`R7-domain`). Without the root a
    verifier fails closed: a value the log registered under the domain it takes now that it doesn't hold (W, or a
    fixed-name D) refuses its inputs (`OneStage.paramsOfJson`), whatever the record registers."""
    s, values = REGISTERED_SETUPS[0], {"D": 1, "W": 3}
    law = {"k": 1, "law": "subset"}
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    earlier = R.record(s.program, s.query, law, SCHEME, roots, EARLIER, leaf_layer=LEAF,
                       registered=s.entries(values, EARLIER_WROOT, ["W"]))
    rec = _record_of(s, values, law, WINDOW, [earlier], leaf_layer=LEAF, previous=R.digest(earlier), window_root=WROOT,
                     fixed=["W"])
    params = {"anchors": [], "delta": "1/10", "held": ["W"], "law": law, "leaf_layer": LEAF, "log": [earlier],
              "parameters": s.signature, "partition": None, "ports": s.ports, "program": s.desc, "query": s.query,
              "registered": values, "scheme": SCHEME, "window_root": WROOT, "work_table": None}
    a = {"params": params, "record": rec}

    def codes(b):
        kind, out = D.replay(REGISTRATION, _json(b), pieces)
        return json.loads(out)["codes"] if kind == "output" else kind

    (d0, w0), (d1, w1) = earlier["registered"], rec["registered"]
    assert d0["root"] != d1["root"] and d0["domain"] != d1["domain"] and w0 == w1
    assert codes(a) == []
    replay = copy.deepcopy(a)
    replay["record"]["registered"][0] = earlier["registered"][0]
    assert codes(replay) == ["R7-domain"]
    moved = copy.deepcopy(a)
    moved["record"]["registered"][1] = {**moved["record"]["registered"][1], "root": "12" * 64}
    assert codes(moved) == ["R7-held"]
    assert codes({**a, "params": {**params, "held": []}}) == ["R7-domain"]
    unrooted = {k: v for k, v in params.items() if k != "window_root"}
    assert codes({**a, "params": unrooted}) == ["R7-domain"]
    # fail closed: with no window root, a log that registered W, which the verifier doesn't hold, refuses its inputs, and
    # so does a fixed-name D the log registered under the domain it takes now, replayed or fresh
    assert codes({**a, "params": {k: v for k, v in unrooted.items() if k != "held"}}) == "reject"
    plain = R.record(s.program, s.query, law, SCHEME, roots, EARLIER, leaf_layer=LEAF, registered=s.entries(values))
    for again in (s.entries(values), rec["registered"]):
        b = {"params": {**unrooted, "log": [plain]},
             "record": {**_record_of(s, values, law, WINDOW, [plain], leaf_layer=LEAF, previous=R.digest(plain)),
                        "registered": again}}
        assert codes(b) == "reject"
        assert codes({**b, "params": {**b["params"], "held": ["D", "W"]}}) == ([] if again == plain["registered"]
                                                                                else ["R7-domain", "R7-held"])
    # the window root and held values are the verifier's, never the record's: a record whose window names another root
    # or holds D gets the codes it gets without them, and one carrying them as keys of its own is refused by its form
    claims = {"window_root": EARLIER_WROOT, "held": ["D", "W"]}
    for b in (a, replay, moved):
        claimed = copy.deepcopy(b)
        claimed["record"]["window"] = {**claimed["record"]["window"], **claims}
        assert codes(claimed) == codes(b)
    for k, v in claims.items():
        assert codes({**a, "record": {**rec, k: v}}) == ["keys"]
        assert codes({**a, "record": {**rec, "registered": [{**rec["registered"][0], k: v}, rec["registered"][1]]}}) == [
            "R7-registered"]


def test_a_seed_salt_the_server_chose_is_refused_by_name(pieces):
    """Lean and Python agreeing: a verifier naming its window's open root that read the window's seed salt from its
    challenger accepts a record whose window names that salt, and refuses one naming another salt, one in another form or
    none (`R6-seed-salt`). A verifier that read none (no `seed_salt`, or null) accepts a window naming none and refuses one
    naming a salt (`R6-seed-salt`): the server chose it. A salt the verifier holds in another form, or without the
    window's open root, refuses its inputs."""
    s, values = REGISTERED_SETUPS[0], {"D": 1, "W": 3}
    law = {"k": 1, "law": "subset"}
    rec = _record_of(s, values, law, {**WINDOW, R.SEED_SALT: SEED_SALT}, leaf_layer=LEAF, window_root=WROOT)
    params = {"anchors": [], "delta": "1/10", "law": law, "leaf_layer": LEAF, "log": [], "parameters": s.signature,
              "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": values,
              "scheme": SCHEME, "seed_salt": SEED_SALT, "window_root": WROOT, "work_table": None}

    def codes(window, p=params):
        kind, out = D.replay(REGISTRATION, _json({"params": p, "record": {**rec, "window": window}}), pieces)
        return json.loads(out)["codes"] if kind == "output" else kind

    assert codes(rec["window"]) == []
    for salt in (OTHER_SEED_SALT, SEED_SALT.upper(), None, 5):
        assert codes({**WINDOW, R.SEED_SALT: salt}) == ["R6-seed-salt"], salt
    assert codes(WINDOW) == ["R6-seed-salt"]
    unread = {k: v for k, v in params.items() if k != "seed_salt"}
    for p in (unread, {**params, "seed_salt": None}):
        assert codes(rec["window"], p) == ["R6-seed-salt"]
        assert codes(WINDOW, p) == []
    for bad in (SEED_SALT.upper(), SEED_SALT[:-2], SEED_SALT + "\n", "", 5):
        assert codes(rec["window"], {**params, "seed_salt": bad}) == "reject", bad
    assert codes(rec["window"], {**params, "window_root": None}) == "reject"


def test_a_window_opens_the_pool_its_chain_left(pieces):
    """Lean and Python agreeing on the pool chain: a window opens the pool the previous window of record left, of the
    same program, `pin` the previous record's `pout` entry, its domain and root, whatever the verifier made of that
    window; it opens the empty pool, `pin` filled from `genesis`, at a registration's first window, after the verifier
    refused the window the public lag back, as its verdict log names it (`verdicts`), and where serving resets it
    (`reset`), which the record's window names, with its week and whether the week's reset is spent at it. Any other
    `pin` is refused by name (`R7-pool-chain`), a window naming another reset, week or spent mark than the verifier's
    pool gives, or none, by name (`R6-pool-reset`, `R6-pool-week`, `R6-pool-spent`), a second reset in a week by name
    (`R6-pool-reset-week`), and a pool in another form refuses the verifier's inputs."""
    s, law = POOLED, {"k": 1, "law": "subset"}
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    pool = {"genesis": GENESIS, "lag": 1, "pin": "pin", "pout": "pout", "reset": False, "verdicts": {}, "week": 0}
    params = {"anchors": [], "delta": "1/10", "law": law, "leaf_layer": LEAF, "log": [], "parameters": s.signature,
              "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "pool": pool,
              "registered": {"pin": 16, "pout": 16}, "scheme": SCHEME, "window_root": EARLIER_WROOT, "work_table": None}

    def record(window, pout, pin=None, log=(), reset=False, week=0, **marks):
        """The record after `log`, its window naming the pool fields of the verifier's reset `reset` in week `week`,
        then `marks`."""
        honest = {R.POOL_RESET: reset, R.POOL_WEEK: week,
                  R.POOL_SPENT: reset or R.pool_spent_before({"week": week}, list(log))}
        return R.record(s.program, s.query, law, SCHEME, roots, {**window, **honest, **marks}, leaf_layer=LEAF,
                        previous=R.digest(log[-1]) if log else None, registered=[pout, *([pin] if pin else [])])

    def codes(p, rec):
        kind, out = D.replay(REGISTRATION, _json({"params": p, "record": rec}), pieces)
        return json.loads(out)["codes"] if kind == "output" else kind

    (pout0,), (pout1,) = s.entries({"pout": 16}, EARLIER_WROOT), s.entries({"pout": 16}, WROOT)
    empty = G.supplied_value(s.program, "pin", [[("bytes", GENESIS)]], record(EARLIER, pout0), ["pin"]).entry()
    first = record(EARLIER, pout0, empty)
    assert codes(params, first) == []
    carried = {**pout0, "value": "pin"}
    then = {**params, "log": [first], "window_root": WROOT}
    assert codes(then, record(WINDOW, pout1, carried, [first])) == []
    for pin, want in (({**carried, "root": "ef" * 64}, ["R7-held", "R7-pool-chain"]),
                      (empty, ["R7-domain", "R7-held", "R7-pool-chain"]),
                      ({**pout1, "value": "pin"}, ["R7-domain", "R7-held", "R7-pool-chain"]),
                      ({**carried, "domain": pout1["domain"]}, ["R7-domain", "R7-pool-chain"])):
        assert codes(then, record(WINDOW, pout1, pin, [first])) == want, pin
    for how in ({"verdicts": {R.digest(first): False}}, {"reset": True}):
        restart, reset = {**then, "pool": {**pool, **how}}, how.get("reset", False)
        assert codes(restart, record(WINDOW, pout1, empty, [first], reset)) == [], how
        assert codes(restart, record(WINDOW, pout1, carried, [first], reset)) == [
            "R7-domain", "R7-supplied", "R7-pool-chain"], how
    for verdicts, lag in (({R.digest(first): True}, 1), ({GENESIS: False}, 1), ({R.digest(first): False}, 2)):
        on = {**then, "pool": {**pool, "verdicts": verdicts, "lag": lag}}
        assert codes(on, record(WINDOW, pout1, carried, [first])) == [], (verdicts, lag)
    spent = record(EARLIER, pout0, empty, reset=True)
    assert codes({**params, "pool": {**pool, "reset": True}}, spent) == []
    after = {**then, "log": [spent]}
    again = {**after, "pool": {**pool, "reset": True}}
    assert codes(again, record(WINDOW, pout1, empty, [spent], True)) == ["R6-pool-reset-week"]
    next_week = {**again, "pool": {**pool, "reset": True, "week": 1}}
    assert codes(next_week, record(WINDOW, pout1, empty, [spent], True, 1)) == []
    assert codes(after, record(WINDOW, pout1, carried, [spent])) == []
    for field, named, at, log, code in (
            *((R.POOL_SPENT, n, after, [spent], "R6-pool-spent") for n in (False, None, 1)),
            *((R.POOL_WEEK, n, then, [first], "R6-pool-week") for n in (1, None, False, "0")),
            *((R.POOL_RESET, n, then, [first], "R6-pool-reset") for n in (True, None, 0, "false"))):
        assert codes(at, record(WINDOW, pout1, carried, log, **{field: named})) == [code], (field, named)
    (pin1,) = s.entries({"pin": 16}, WROOT)
    unpooled = {R.POOL_RESET: None, R.POOL_WEEK: None, R.POOL_SPENT: None}
    assert codes({**then, "pool": None}, record(WINDOW, pout1, pin1, [first], **unpooled)) == []
    for field, code in ((R.POOL_RESET, "R6-pool-reset"), (R.POOL_WEEK, "R6-pool-week"),
                        (R.POOL_SPENT, "R6-pool-spent")):
        assert codes({**then, "pool": None}, record(WINDOW, pout1, pin1, [first], **{**unpooled, field: False})) == [
            code], field
    for bad in (*({k: v for k, v in pool.items() if k != b} for b in ("lag", "reset", "verdicts", "week")),
                {**pool, "pin": "pout"}, {**pool, "pin": "x"}, {**pool, "genesis": GENESIS.upper()},
                {**pool, "genesis": GENESIS[:-2]}, {**pool, "lag": 0}, {**pool, "lag": True}, {**pool, "reset": 1},
                {**pool, "week": -1}, {**pool, "week": False}, {**pool, "verdicts": []},
                {**pool, "verdicts": {GENESIS[:-2]: False}}, {**pool, "verdicts": {GENESIS.upper(): False}},
                {**pool, "verdicts": {GENESIS: 0}}, {**pool, "refused": False}, {**pool, "extra": False},
                {**pool, "pout": 5}, []):
        assert codes({**then, "pool": bad}, record(WINDOW, pout1, carried, [first])) == "reject", bad
    for extra in ({"supplied": {"pin": [[{"bytes": GENESIS}]]}}, {"outputs": {"pout": {"port": "x"}}},
                  {"row_schemas": {"pin": G.SCHEMA_SEG}}):
        assert codes({**then, **extra}, record(WINDOW, pout1, carried, [first])) == "reject", extra


def test_an_arrival_cut_at_a_windows_edge_continues_from_the_pool_it_left(pieces):
    """Lean and Python agreeing on cut and continue over three windows. Each window's certified arrivals are the root the
    network certifier signed for it, which the verifier pins (`ing`: the arrivals open at the last window's edge first,
    each at the values it arrived with and the outputs released of it); its rows (`req`) are its own, under its open
    root; and its pool chains. Three windows that carry the pool are accepted, arrivals with a continuation counted
    from the outputs serving computed or the last window's arrivals again are `R7-held`, and a window that restarts its
    pool, by the public rule or by serving's reset, is accepted from the empty pool alone, still serving the
    continuations, its carried pool refused by name."""
    s, law = CONTINUED, {"k": 1, "law": "subset"}
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    windows, wroots = (EARLIER, WINDOW, {**WINDOW, "row": 102}), (EARLIER_WROOT, WROOT, "a8" * 64)

    def stamp(rows, tag):
        salts = [hashlib.shake_256(f"ing/{tag}/{r}".encode()).digest(192) for r in range(len(rows))]
        return G.Registered.commit(s.sha512, "ing", rows, salts).entry()

    stamps = [stamp(rows, i) for i, rows in enumerate((
        [[1, 7, 5], [1, 8, 2], [1, 9, 3], [0, 0, 0]], [[1, 7, 8], [1, 9, 3], [1, 10, 1], [0, 0, 0]],
        [[1, 9, 4], [1, 10, 1], [0, 0, 0], [0, 0, 0]]))]

    def params(i, log, **how):
        pool = {"genesis": GENESIS, "lag": 1, "pin": "pin", "pout": "pout", "reset": False, "verdicts": {}, "week": 0}
        return {"anchors": [], "delta": "1/10", "law": law, "leaf_layer": LEAF, "log": list(log),
                "parameters": s.signature, "partition": None, "ports": s.ports, "program": s.desc, "query": s.query,
                "pool": {**pool, **how}, "pinned": {"ing": stamps[i]["root"]},
                "registered": {"ing": 3, "req": 12, "pin": 16, "pout": 16}, "scheme": SCHEME,
                "window_root": wroots[i], "work_table": None}

    def record(i, pin, log=(), ing=None, reset=False):
        """Window i's record after `log`, `pin` its entry for `pin`, the empty pool filled from `genesis` where it is
        None, its window naming the reset `reset` in week 0 and the spent mark R6 holds it to."""
        marks = {R.POOL_RESET: reset, R.POOL_WEEK: 0,
                 R.POOL_SPENT: reset or R.pool_spent_before({"week": 0}, list(log))}

        def bare(*extra):
            return R.record(s.program, s.query, law, SCHEME, roots, {**windows[i], **marks},
                            leaf_layer=LEAF, previous=R.digest(log[-1]) if log else None,
                            registered=[ing or stamps[i], *s.entries({"req": 12, "pout": 16}, wroots[i]), *extra])

        if pin is None:
            pin = G.supplied_value(s.program, "pin", [[("bytes", GENESIS)]], bare(), ["pin"]).entry()
        return bare(pin)

    def carried(rec):
        return {**next(e for e in rec["registered"] if e["value"] == "pout"), "value": "pin"}

    def codes(p, rec):
        kind, out = D.replay(REGISTRATION, _json({"params": p, "record": rec}), pieces)
        return json.loads(out)["codes"] if kind == "output" else kind

    first = record(0, None)
    second = record(1, carried(first), [first])
    third = record(2, carried(second), [first, second])
    for i, rec in enumerate((first, second, third)):
        assert codes(params(i, [first, second][:i]), rec) == [], i
    served = stamp([[1, 7, 9], [1, 9, 3], [1, 10, 1], [0, 0, 0]], "served")
    for ing in (served, stamps[0]):
        assert codes(params(1, [first]), record(1, carried(first), [first], ing)) == ["R7-held"], ing
    assert codes(params(1, [first]), record(1, None, [first])) == ["R7-domain", "R7-held", "R7-pool-chain"]
    for how in ({"verdicts": {R.digest(second): False}}, {"reset": True}):
        restart, reset = params(2, [first, second], **how), how.get("reset", False)
        assert codes(restart, record(2, None, [first, second], reset=reset)) == [], how
        assert codes(restart, record(2, carried(second), [first, second], reset=reset)) == [
            "R7-domain", "R7-supplied", "R7-pool-chain"], how
        assert codes(restart, record(2, None, [first, second], stamps[1], reset)) == ["R7-held"], how


def test_a_root_with_a_trailing_newline_is_refused(pieces):
    """A registered root and a registered value's 128-character root (127 hex digits and a newline), each with a newline
    last, are refused by Lean and Python alike: Python's `$` matches before a final newline, which `fullmatch` doesn't."""
    s = REGISTERED_SETUPS[0]
    a = _v2_audit(s, s.values[1])
    r3, r7 = copy.deepcopy(a), copy.deepcopy(a)
    r3["record"]["roots"][0]["root"] += "\n"
    r7["record"]["registered"][0]["root"] = r7["record"]["registered"][0]["root"][:127] + "\n"
    for b, codes in ((r3, ["R3-root"]), (r7, ["R7-root", "R7-held"])):
        got = registration(_json(_REGISTRATION_OF(b)))
        assert got["codes"] == codes
        assert D.replay(REGISTRATION, _json(_REGISTRATION_OF(b)), pieces) == ("output", D.canonical(got))


def _many_ports(a: dict, n: int, seed: int = 0) -> dict:
    """`a` with `n` more ports, a MoE window's shape (64 experts' tiles of 8), registered at roots in shuffled order."""
    b = copy.deepcopy(a)
    roots = []
    for i in range(n):
        p = f"w13/e{i // 8:04d}/t{i % 8}"
        b["params"]["ports"][p] = {"domain": hashlib.sha512(p.encode()).hexdigest(), "leaves": 1 + i % 64}
        roots.append({"port": p, "root": f"{i:0128x}", **b["params"]["ports"][p]})
    random.Random(seed).shuffle(roots)
    b["record"]["roots"] = [*b["record"]["roots"], *roots]
    return b


def test_a_many_port_registration_agrees(pieces):
    """R3 over a window of 4096 ports, the shape whose quadratic walk `Registration.portIndex` replaced, Lean and Python
    agreeing: accepted as registered, and each change to one port reaches its code and no other."""
    a = _many_ports(_REGISTRATION_OF(_v1_audit("subset")), 4096)

    def codes(change):
        b = copy.deepcopy(a)
        change(b["record"]["roots"])
        kind, out = D.replay(REGISTRATION, _json(b), pieces)
        return json.loads(out)["codes"] if kind == "output" else kind

    def bump(r, k):
        r[k] = r[k] + 1 if k == "leaves" else "ab" * 64

    assert codes(lambda rs: None) == []
    assert codes(lambda rs: rs.append({**rs[1000], "root": "not hex"})) == ["R3-ports", "R3-root"]
    assert codes(lambda rs: rs.pop(2000)) == ["R3-ports"]
    assert codes(lambda rs: rs[3000].update(port=7)) == ["R3-ports"]
    assert codes(lambda rs: bump(rs[123], "leaves")) == ["R3-leaves"]
    assert codes(lambda rs: bump(rs[4000], "domain")) == ["R3-domain"]


def test_every_public_items_code_is_reached(pieces):
    """Each change to the rule's input on a v1 audit and a v2 one, Lean and Python agreeing on each: together they reach
    every code of `check_public` and a reject, which the sampled runs only probably do, and the unchanged audits none."""
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
    """A v2 audit of each program accepts when the verifier reports the sessions' registered reads opened against
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
    """The whole audit record, field by field, on an accepted audit of each sampling design (with a work sampling design's closure), the same audit
    rejected by its served sample and its sessions' roots, and an accepted v2 audit: every field the Lean writes is
    `audit_record`'s, a Bernoulli law's bound on incorrect proof blocks the exact one (`Bernoulli.count_bound`)."""
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
            p, n = Fraction(case["draw"]["p"]), case["registration"]["population"]
            assert lean["wrong_units"]["bound"] == Bernoulli(p).count_bound(n, float(Fraction(case["delta"])))
    assert [json.loads(json.loads(D.replay(RECORD_PORT, c, pieces)[1]))["accepted"] for c in cases] == [True] * 12 + [False, True]


def _v1_audit(kind: str, s: Setup = SETUPS[3], p: str = "1/1") -> dict:
    """One valid v1 audit of `s` under a sampling design of `kind` (a Bernoulli one at `p`, which a run refuses, or `all`,
    the sampling design All names: `subset` over every unit), its verdicts accepting. `subset`, `stratified` and `work`
    are rates, `rate:3/N`, `stratified-rate:1/2` and `work-rate:1/2`, which the params name and the record holds
    resolved."""
    law = ({"law": "bernoulli", "p": p} if kind == "bernoulli" else {"k": s.population, "law": "subset"} if kind == "all"
           else {"subset": f"rate:3/{s.population}", "stratified": "stratified-rate:1/2", "work": "work-rate:1/2"}[kind])
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}}
    table = {t: w for t, w in table.items() if t in s.templates} if kind == "work" else None
    named, expanded = service.params_law(law, s.program, s.query, table)
    mine = DR.derive(s.program, s.query, named, expanded) if kind != "bernoulli" else {"k": 3}
    rec = R.record(s.program, s.query, named, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, work_table=expanded)
    params = {"anchors": [], "delta": "1/10", "law": law, "leaf_layer": LEAF, "log": [], "partition": None, "ports": s.ports,
              "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME, "work_table": table}
    instances = sum(st_["k"] for st_ in mine["strata"]) if "strata" in mine else mine["k"]
    evidence = {"session_roots": dict(ROOTS), "statement": {"instances": instances},
                "verdicts": {"lean": {"accepted": True, "digest": "aa" * 64, "of_record": True},
                             "upstream": {"accepted": True, "of_record": False}}}
    return _json({"evidence": evidence, "params": params, "record": rec})


def test_the_run_reaches_every_step(pieces):
    """A registration's run under each sampling design, over one unit's population and over three, and a v2 registration's, Lean's
    the Python's whole: each accepts, on the sample from the challenger's coins (`_from_challenger`), showing the
    verdicts' public outputs. The same runs refused at each step (a declared item missing, a record R1-R7 refuses, no
    session, a verdict of record about other sessions) and rejected (a verdict that ran and did not accept), which the
    sampled runs reach only probably. A run that read coins is refused at its select step without their evidence, or with
    empty evidence (`coins`): the provenance check is never skipped. A run asked to sample from the verifier's own coins
    is refused before it starts: the verifier samples no coins of its own."""
    v2 = _v2_audit(REGISTERED_SETUPS[0], REGISTERED_SETUPS[0].values[1])
    v2["evidence"]["verdicts"]["upstream"] = {"accepted": True, "of_record": False}
    seen = []
    laws = ("subset", "stratified", "work")
    for i, a in enumerate([*(_v1_audit(k) for k in laws), *(_v1_audit(k, SETUPS[2]) for k in laws), v2]):
        _, case = _from_challenger(_run_case(a, "", 2, ["credited"]), random.Random(f"run/{i}"))
        other = {**case["verdicts"]["lean"], "sessions": RUN_SESSIONS[::-1]}
        for changed in (case, {**case, "public": case["public"][1:]},
                        {**case, "record": {**case["record"], "previous": "11" * 64}}, {**case, "sessions": []},
                        {**case, "verdicts": {**case["verdicts"], "lean": other}},
                        {**case, "verdicts": {**case["verdicts"], "upstream": {"accepted": False, "of_record": False}}},
                        {k: v for k, v in case.items() if k != "coins"}, {**case, "coins": {}},
                        {**case, "coins": {"segment": None}}):
            out = run(copy.deepcopy(changed))
            assert D.replay(RUN, changed, pieces) == ("output", D.canonical(out)), (a["params"]["law"], out)
            seen.append(out.get("step") or out["symbol"])
            if changed.get("coins") != case["coins"]:
                assert out["codes"] == ["coins"], out
        assert run(copy.deepcopy(case))["public"] == {"credited": 12} and run(copy.deepcopy(case))["read"] > 0
        for own in ("verifier", "os", ""):
            kind, why = D.replay(RUN, {**case, "coin_source": own}, pieces)
            assert kind == "reject" and "the verifier draws no coins of its own" in why, (own, kind, why)
    assert seen == ["accept", "register", "register", "deliver", "outcome", "reject", "select", "select", "select"] * 7


def test_a_two_stage_run_reaches_every_step(pieces):
    """A two-stage run (`_two_stage_case`), Lean's the Python's whole: its first stage two of the four replay blocks and
    its second one proof block in each, its session reading the ports the links derived from `BLOCK` name (PAIR's `x`
    and `o0`) at the boundary, with both audit records, and rejected on round 2's refusal of the two-stage form alone
    (`two-stage-form`), which the Python allowing the form (`second_stage.FORM`) accepts, as it does the run whose
    session also reads the interior's own `out`, each slot its own rows, which its verifier reports and its template's
    reference statement reads; a session reading `out` its reference doesn't is refused (`second-values`), as is one
    reading its linked `x` from the interior's own (`second-read`), a statement whose verifier of record checked its
    session against a public file whose header names another circuit file (`second-verified`), and an interior naming
    links is no run's. Then refused at each step: a grain the registration's query isn't (`grain`); no
    interior registration (`interior`); an interior registration of another population (`interior-population`), in a
    window that is neither the boundary registration's nor the one the verifier computes (`interior-window`: the
    second session opened on a sampled root, which signs and opens as the honest one does; a root no session opened;
    no root), not after the boundary registration (`R6-previous`), or declaring a value the Spec doesn't hold;
    second-stage coins without their evidence, or held to the boundary registration's window, which the second
    session didn't open (`coins`). Rejected where a verdict didn't accept, and refused before it starts for `k` =
    n_rho (it would prove whole replay blocks) and an interior registration under a one-stage rule."""
    case = _two_stage_case()
    out = run(copy.deepcopy(case))
    assert D.replay(RUN, case, pieces) == ("output", D.canonical(out))
    inner = json.loads(out["interior"]["audit"])
    assert out["symbol"] == "reject" and out["public"] == {} and json.loads(out["audit"])["accepted"] is True, out
    assert [k for k, c in inner["checks"].items() if not c["ok"]] == ["statements"]
    assert inner["checks"]["statements"]["codes"] == ["two-stage-form"]
    assert len(out["draw"]["units"]) == 2 and sorted(u // 3 for u in out["interior"]["draw"]["units"]) == [0, 1]
    with mock.patch.object(SS, "FORM", True):
        allowed = run(copy.deepcopy(case))
    assert allowed["symbol"] == "accept" and allowed["public"] == {"credited": 12}, allowed
    def reads(referred=True, **ports):
        def change(c, x):
            it = c["interior"]
            s = it["statements"][0]
            s = {**s, "registered": {**s["registered"], "ports": {
                **s["registered"]["ports"], **{p: {"value": p, "root": ROOTS[p], **INTERIOR.ports[p], "reads_from": 0,
                                                   "reads": rows} for p, rows in ports.items()}}}}
            ps = it["params"]
            es = [{**e, "registered": s["registered"]} if referred else e
                  for e in ps["proofs"]["templates"]]
            return {**c, "interior": {**it, "statements": [s], "params": {**ps, "proofs": {"templates": es}}},
                    "verdicts": {**c["verdicts"], "lean": {**c["verdicts"]["lean"], **_checked_names(s)}}}
        return change

    def reported(*ports):
        return _verdict_change(lambda v, c, x: {**v, "registered": {
            **v["registered"], **{p: {"value": p, "root": ROOTS[p]} for p in ports}}})
    s0 = case["interior"]["statements"][0]
    elsewhere = _verdict_change(lambda v, c, x: {**v, **_checked_names(
        {**s0, "header": {**s0["header"], "circuit_sha512": "ab" * 64}})})(case, 0)
    evens = [2 * u for u in range(6)]
    for changed, codes in ((reported("out")(reads(out=evens)(case, 0), 0), ["two-stage-form"]),
                           (reported("out")(reads(False, out=evens)(case, 0), 0), ["second-values"]),
                           (reported("x")(reads(x=list(range(6)))(case, 0), 0), ["second-read"]),
                           (elsewhere, ["second-verified"])):
        out = run(copy.deepcopy(changed))
        assert D.replay(RUN, changed, pieces) == ("output", D.canonical(out))
        assert json.loads(out["interior"]["audit"])["checks"]["statements"]["codes"] == codes, out
        with mock.patch.object(SS, "FORM", True):
            assert (run(copy.deepcopy(changed))["symbol"] == "accept") == (codes == ["two-stage-form"])
    named = _inner(lambda it, x: {**it, "links": [{"template": INTERIOR.templates[0], "port": "x", "value": BX}]})(
        case, 0)
    assert D.replay(RUN, named, pieces)[0] == "reject"
    with pytest.raises(D.Reject):
        run(copy.deepcopy(named))
    steps = [
        ({**case, "rule": {**case["rule"], "grain": INTERIOR.query}}, ("register", ["grain"])),
        ({**case, "interior": None}, ("interiors", ["interior"])),
        (_inner_record(population=9)(case, 0), ("interiors", ["interior-population"])),
        (_two_stage_case(elsewhere=True), ("interiors", ["interior-window"])),
        (_inner_params(window_root="22" * 64)(case, 0), ("interiors", ["interior-window"])),
        (_inner_params(window_root=None)(case, 0), ("interiors", ["interior-window"])),
        (_inner_record(previous="11" * 64)(case, 0), ("interiors", ["R6-previous"])),
        (_inner(lambda it, x: {**it, "values": it["values"][1:]})(case, 0), ("interiors", None)),
        (_inner(lambda it, x: {**it, "coins": None})(case, 0), ("second", ["coins"])),
        (_inner_params(window_root=case["params"]["window_root"])(case, 0), ("second", ["coins"])),
    ]
    for changed, (step, codes) in steps:
        got = run(copy.deepcopy(changed))
        assert D.replay(RUN, changed, pieces) == ("output", D.canonical(got)), (step, got)
        assert got.get("step") == step and (codes is None or got["codes"] == codes), (step, got)
    rejected = _verdict_change(lambda v, c, x: {**v, "accepted": False})(case, 0)
    got = run(copy.deepcopy(rejected))
    assert D.replay(RUN, rejected, pieces) == ("output", D.canonical(got)) and got["symbol"] == "reject", got
    for refused in ({**case, "rule": {**case["rule"], "k": 3}}, {**case, "rule": {"law": "rate:1/2", "rule": "law"}}):
        assert D.replay(RUN, refused, pieces)[0] == "reject", refused["rule"]
        with pytest.raises(D.Reject):
            run(copy.deepcopy(refused))


def _second_with(c: dict, i: int, **fields) -> dict:
    """`c` with fields of its `i`-th statement replaced, its session checked as it states it."""
    return _as_stated({**c, "statements": [{**s, **fields} if k == i else s for k, s in enumerate(c["statements"])]})


def test_round_two_holds_each_session_to_its_part(pieces):
    """Round 2's statement check, Lean's the Python's whole, over the two-template interior (`SECOND_MIXED`, its ports
    named by template) after a replay sample of two: the honest statements pass, one for each part, by position in the
    parts' order (else `second-order`), with a part's `x` read from the boundary where the design links it, each slot
    at its unit's row of its cell's replay block, or read from its own `x` at its own rows, and fail only on the form
    while it is refused (`two-stage-form`); every part needs a statement, the second stage drawing in it or not
    (`second-cover`); a read of the boundary at another cell's replay block, of a tree of more than `2^64` leaves, of
    its own `x` at another slot's rows, of a boundary port the design doesn't link, or of a linked port anywhere but
    the boundary port its link names (or the interior's value of its name, where the interior registers one), or a
    linked port not read (`second-read`) is refused, as are reads the verifier of record doesn't report
    (`second-reported`); links none derived, one for a template of no part, or one whose rows don't place its part's
    units, are refused (`second-links`);
    a header that isn't its template's reference header placed in the run (`second_stage.at_run`: its set
    `served/<run>/m<k>` at its template's position, its ports' bindings the window rule's) but for the run's fields
    (`second_stage.RUN_FIELDS`, `units.blocks` and `registered` among them), or of a template with none, is refused
    (`second-header`), and one only the run's fields differ in passes it, as does a reference staged in another set or
    at other slots; an interior record whose
    replay sample isn't the verifier's own in order, `True` for 1 included, is refused before any statement
    (`second-interior-order`); and a replay sample of three, which TRIPLE's four slots aren't a multiple of, makes no
    parts (`second-cells`)."""
    pair, triple = SECOND_MIXED.templates
    sample, drawn = [0, 2], [1, 4, 6, 9]
    honest = _second_case(SECOND_MIXED, True, sample, drawn, allow=True)
    h0, h1 = (s["header"] for s in honest["statements"])
    b0 = h0["frame_v3"]["bindings"]
    odd = _second_case(SECOND_MIXED, True, [1, 3], [], allow=True)

    def placed(c: dict, **w) -> dict:
        return {**c, "interior": {**c["interior"], "window": {**c["interior"]["window"], **w}}}

    def wide(c: dict, per: int) -> dict:
        """`c` with the boundary's `x` of `per` rows a replay block, TRIPLE's rows naming them, its slots reading their
        units' rows there."""
        b, s, (r,) = c["boundary"], c["statements"][1], c["rows"]
        e = s["registered"]["ports"]["x"]
        read = {**e, "leaves": b["population"] * per,
                "reads": [x // r["per"] * per + x % r["per"] for x in e["reads"]]}
        return {**_second_with(c, 1, registered={**s["registered"], "ports": {"x": read}}), "rows": [{**r, "per": per}],
                "boundary": {**b, "roots": [{**x, "leaves": read["leaves"]} if x["port"] == BX else x
                                            for x in b["roots"]]}}
    elsewhere = _second_reference(triple, 0, h1["instances"], ("out", "x"))
    assert elsewhere != honest["refs"][triple] and SS.blank_run(elsewhere) == SS.blank_run(honest["refs"][triple])
    assert h0["set"] == "served/r-test/m0" and h1["set"] == "served/r-test/m1" and sorted(b0) == ["out", "x"]
    assert b0["x"] == SS.port_binding(honest["interior"], f"{pair}/x", SECOND_SCHEMAS["x"], "r-test")
    linked = _second_case(SECOND_MIXED, True, sample, drawn, links={triple}, allow=True)
    reads = linked["statements"][1]["registered"]["ports"]["x"]
    bx = _second_case(SECOND_MIXED, True, sample, drawn, links={pair})["statements"][0]["registered"]
    own = _second_case(SECOND_MIXED, True, sample, drawn, links={triple}, own={pair}, allow=True)
    mine = own["statements"][0]["registered"]["ports"]["x"]
    assert [s["draw"]["units"] for s in honest["statements"]] == [[1, 4], [0, 3]] and reads["reads"] == [0, 1, 6, 7]
    assert mine["value"] == f"{pair}/x" and mine["reads"] == list(range(6)) and mine["leaves"] == 6
    cases = [
        (honest, []),
        ({**honest, "allow": False}, ["two-stage-form"]),
        ({**honest, "statements": honest["statements"][::-1]}, ["second-order"]),
        (linked, []),
        (wide(linked, 2 ** 64 // linked["boundary"]["population"]), []),
        (wide(linked, 2 ** 64 // linked["boundary"]["population"] + 1), ["second-read"]),
        (own, []),
        ({**own, "allow": False}, ["two-stage-form"]),
        (_second_case(SECOND_MIXED, True, sample, drawn, links={pair, triple}, allow=True), []),
        (_second_case(SECOND_MIXED, True, sample, [1, 4], allow=True), []),
        ({**_second_case(SECOND_MIXED, True, sample, [1, 4], allow=True), "statements": honest["statements"][:1]},
         ["second-cover"]),
        ({**honest, "statements": honest["statements"][1:]}, ["second-order", "second-cover"]),
        ({**honest, "statements": honest["statements"][:1]}, ["second-cover"]),
        (_second_with(linked, 1, registered={**linked["statements"][1]["registered"], "ports": {
            "x": {**reads, "reads": reads["reads"][::-1]}}}), ["second-read"]),
        (_second_with(own, 0, registered={**own["statements"][0]["registered"], "ports": {
            "x": {**mine, "reads": mine["reads"][::-1]}}}), ["second-read"]),
        ({**linked, "links": []}, ["second-read"]),
        ({**linked, "links": [{"template": triple, "port": "x", "value": BOUT}]}, ["second-read"]),
        ({**honest, "links": linked["links"], "rows": linked["rows"]}, ["second-read"]),
        ({**own, "links": [*own["links"], {"template": pair, "port": "x", "value": BX}],
          "rows": [*own["rows"], {"template": pair, "port": "x", "per": 3, "runs": _runs(3, 3)}]}, ["second-read"]),
        ({**honest, "links": linked["links"]}, ["second-links"]),
        ({**linked, "rows": []}, ["second-links"]),
        ({**own, "verdicts": {**own["verdicts"], "lean": {"of_record": True, "accepted": True}}}, ["second-reported"]),
        (_second_with(honest, 0, registered=bx), ["second-read", "second-values", "second-reported"]),
        ({**linked, "links": None}, ["second-links"]),
        ({**linked, "links": [*linked["links"], {"template": "Other_v0", "port": "x", "value": BX}]}, ["second-links"]),
        (_second_case(SECOND_MIXED, True, [0, 1, 2], drawn, allow=True), ["second-cells"]),
        ({**honest, "refs": None}, ["second-header"]),
        ({**honest, "refs": {pair: honest["refs"][pair]}}, ["second-header"]),
        ({**honest, "refs": {**honest["refs"], triple: {**honest["refs"][triple], "set": "served/r/m1"}}}, []),
        ({**honest, "refs": {**honest["refs"], pair: {**honest["refs"][pair], "scheme": "frame-v3-sha512/other"}}},
         ["second-header"]),
        (_second_with(honest, 0, header={**h0, "range": [0, 7]}), ["second-header"]),
        (_second_with(honest, 0, header={**h0, "units": {**h0["units"], "blocks": [[0]]}}), []),
        (_second_with(honest, 0, header={**h0, "units": {**h0["units"], "units_per_vu": 2}}), ["second-header"]),
        ({**honest, "refs": {**honest["refs"], triple: elsewhere}}, []),
        (_second_with(honest, 0, header={**h0, "set": h1["set"]}), ["second-header"]),
        (_second_with(honest, 1, header={**h1, "set": "served/r-other/m1"}), ["second-header"]),
        (_second_with(honest, 0, header={**h0, "frame_v3": {**h0["frame_v3"], "bindings": {
            "out": b0["x"], "x": b0["out"]}}}), ["second-header"]),
        (_second_with(honest, 0, header={**h0, "frame_v3": {**h0["frame_v3"], "bindings": {"out": b0["out"]}}}),
         ["second-header"]),
        (_second_with(honest, 0, header={**h0, "registered": {f"{pair}/x": {"positions": [0]}}}), []),
        (placed(honest, run="r-other"), ["second-header"]),
        ({**honest, "refs": {**honest["refs"], pair: h0, triple: SS.blank_run(h1)}}, []),
        (placed(honest, replay_sample=sample[::-1]), ["second-interior-order"]),
        (placed(honest, replay_sample=sample[:1]), ["second-interior-order"]),
        (placed(honest, replay_sample=None), ["second-interior-order"]),
        ({**placed(honest, replay_sample=[2]), "statements": None}, ["second-interior-order", "second-statements"]),
        ({**placed(honest, replay_sample=[0, 2, 3]), "statements": honest["statements"][::-1]},
         ["second-interior-order", "second-order"]),
        (odd, []),
        (placed(odd, replay_sample=[True, 3]), ["second-interior-order"]),
    ]
    for case, codes in cases:
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (codes, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (codes, got)


def test_round_two_holds_the_boundary_schedule_to_its_replay_blocks(pieces):
    """Round 2's schedule, Lean's the Python's whole, over the two-template interior after a replay sample of two: a
    boundary record that commits a schedule of its four replay blocks (`window.schedule`: runs of positions of its
    requests, in block order and each request's positions in order) passes, all blocks one run or several, a request's
    positions resumed in a later run, as does a record that commits none; runs that skip a block, repeat one, restart a
    request's positions or skip some, cover fewer or more blocks than the population, hold no block or no position, or
    aren't runs fail ``second-schedule``, after ``second-interior-order`` when the interior names another sample; and
    each block's request and positions are `schedule.at`'s (`second_schedule`)."""
    honest = _second_case(SECOND_MIXED, True, [0, 2], [1, 4, 6, 9], allow=True)

    def scheduled(runs) -> dict:
        return {**honest, "boundary": {**honest["boundary"], "window": {"schedule": runs}}}

    def run(r: str, f: int, n: int, p: int, length: int) -> dict:
        return {"request": r, "first_span": f, "spans": n, "first_position": p, "length": length}
    resumed = [run("a", 0, 1, 0, 2), run("b", 1, 2, 0, 1), run("a", 3, 1, 2, 2)]
    restarted = [run("a", 0, 1, 0, 2), run("a", 1, 3, 3, 2)]
    assert honest["boundary"]["population"] == 4
    cases = [
        (honest, []),
        (scheduled([run("a", 0, 4, 0, 1)]), []),
        (scheduled(resumed), []),
        ({**honest, "boundary": {**honest["boundary"], "window": {}}}, []),
        (scheduled(None), []),
        (scheduled([run("a", 0, 3, 0, 1)]), ["second-schedule"]),
        (scheduled([run("a", 0, 5, 0, 1)]), ["second-schedule"]),
        (scheduled([run("a", 0, 1, 0, 2), run("b", 2, 2, 0, 1)]), ["second-schedule"]),
        (scheduled([run("a", 0, 2, 0, 1), run("b", 1, 3, 0, 1)]), ["second-schedule"]),
        (scheduled([run("a", 0, 1, 0, 2), run("a", 1, 3, 0, 2)]), ["second-schedule"]),
        (scheduled(restarted), ["second-schedule"]),
        (scheduled([run("a", 0, 4, 1, 1)]), ["second-schedule"]),
        (scheduled([run("a", 0, 4, 0, 0)]), ["second-schedule"]),
        (scheduled([run("a", 0, 0, 0, 1), run("a", 0, 4, 0, 1)]), ["second-schedule"]),
        (scheduled([{**run("a", 0, 4, 0, 1), "extra": 0}]), ["second-schedule"]),
        (scheduled([{**run("a", 0, 4, 0, 1), "request": 5}]), ["second-schedule"]),
        (scheduled({"a": run("a", 0, 4, 0, 1)}), ["second-schedule"]),
        ({**scheduled([run("a", 0, 3, 0, 1)]), "interior": {**honest["interior"], "window": {
            **honest["interior"]["window"], "replay_sample": [2, 0]}}}, ["second-interior-order", "second-schedule"]),
    ]
    for case, codes in cases:
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (codes, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (codes, got)
    for c, want in [
        ({"record": scheduled(resumed)["boundary"], "blocks": [0, 1, 2, 3, 4, -1]},
         {"failure": None, "at": [["a", 0, 2], ["b", 0, 1], ["b", 1, 2], ["a", 2, 4], None, None]}),
        ({"record": scheduled(restarted)["boundary"], "blocks": [0, 1]},
         {"failure": "a run of a from position 3, not from position 2", "at": [["a", 0, 2], ["a", 3, 5]]}),
        ({"record": scheduled([run("a", 0, 3, 0, 1)])["boundary"], "blocks": []},
         {"failure": "runs of 3 blocks, not the population's 4", "at": []}),
    ]:
        assert second_schedule(copy.deepcopy(c)) == want
        assert D.replay(SECOND_SCHEDULE, c, pieces) == ("output", D.canonical(want)), c


def test_round_two_holds_a_wired_port_to_its_writers_output(pieces):
    """Round 2's wires, Lean's the Python's whole, over the two-template interior (`SECOND_MIXED`) after a replay sample
    of two, TRIPLE's `x` wired to PAIR's `out`: the interior commits no TRIPLE `x`, and TRIPLE's session reads PAIR's
    `out`, each slot the row of its writer's unit its rows name, its writer's first in its cell, and binds its `x` at
    PAIR's `out` (`second_stage.at_run`), which passes, as does a slot reading nothing; read at another instance's or
    another cell's rows, at PAIR's `x`, or not read, it fails ``second-read``; without the wire, or with one naming
    another writer port, its header fails too (``second-header``: bound at a port the interior doesn't hold it at), as
    does a wire's for a port no session reads; and wires none derived (null or missing), naming a template of no part as
    reader or writer, or one whose rows don't place its reader's units, fail ``second-wires``, after ``second-links``
    when the links are none too, and a wire not of strings is refused."""
    pair, triple = SECOND_MIXED.templates
    sample, drawn = [0, 2], [1, 4, 6, 9]
    wired = _second_case(SECOND_MIXED, True, sample, drawn, wires=[(triple, pair)], allow=True)
    s1 = wired["statements"][1]
    x, h1 = s1["registered"]["ports"]["x"], s1["header"]
    assert f"{triple}/x" not in {r["port"] for r in wired["interior"]["roots"]}
    assert (x["value"], x["reads"], x["leaves"]) == (f"{pair}/out", [0, 0, 3, 3], 6)
    assert h1["frame_v3"]["bindings"]["x"] == SS.port_binding(wired["interior"], f"{pair}/out", SECOND_SCHEMAS["x"],
                                                              "r-test")
    px = next(r for r in wired["interior"]["roots"] if r["port"] == f"{pair}/x")

    def reading(**e) -> dict:
        return _second_with(wired, 1, registered={**s1["registered"], "ports": {"x": {**x, **e}}})
    wire = wired["wires"][0]
    cases = [
        (wired, []),
        ({**wired, "allow": False}, ["two-stage-form"]),
        (reading(reads=[1, 1, 4, 4]), ["second-read"]),
        (reading(reads=[3, 3, 0, 0]), ["second-read"]),
        (reading(reads=[0, 0, 3, None]), []),
        (reading(value=px["port"], root=px["root"], domain=px["domain"]),
         ["second-read", "second-values", "second-reported"]),
        (_second_with(wired, 1, registered=None), ["second-roots", "second-read", "second-values", "second-reported"]),
        ({k: v for k, v in wired.items() if k != "wires"}, ["second-wires"]),
        ({**wired, "wires": []}, ["second-header", "second-read"]),
        ({**wired, "wires": [{**wire, "wport": "x"}]}, ["second-header", "second-read"]),
        ({**wired, "wires": [wire, {"template": pair, "port": "x", "writer": triple, "wport": "out"}],
          "rows": [*wired["rows"], {"template": pair, "port": "x", "per": 2, "runs": [[0, 3, 0, 0]]}]},
         ["second-wires"]),
        ({**wired, "wires": [wire, {"template": pair, "port": "x", "writer": triple, "wport": "out"}]},
         ["second-wires"]),
        ({**wired, "wires": None}, ["second-wires"]),
        ({**wired, "wires": [{**wire, "template": "Other_v0"}]}, ["second-wires"]),
        ({**wired, "wires": [{**wire, "writer": "Other_v0"}]}, ["second-wires"]),
        ({**wired, "links": None, "wires": None}, ["second-links", "second-wires"]),
        ({**wired, "wires": [{**wire, "wport": 5}]}, "reject"),
    ]
    for case, codes in cases:
        if codes == "reject":
            assert D.replay(SECOND_STATEMENTS, case, pieces)[0] == "reject"
            with pytest.raises(D.Reject):
                second_statements(copy.deepcopy(case))
            continue
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (codes, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (codes, got)


def test_round_two_refuses_a_wire_that_reads_a_returned_output(pieces):
    """A wire whose writer's port a link names reads an output the replay block returns: the boundary record commits
    that output and the interior record's port is what the reader is held to, two entries nothing holds to each other.
    The design fails `second-wires`, its links derived, alike in both."""
    pair, triple = SECOND_MIXED.templates
    wired = _second_case(SECOND_MIXED, True, [0, 2], [1, 4, 6, 9], wires=[(triple, pair)], allow=True)
    per, n = BOUNDARY.ports[BX]["leaves"] // BOUNDARY.population, wired["rows"][0]["per"]
    case = {**wired, "links": [{"template": pair, "port": "out", "value": BX}],
            "rows": [*wired["rows"], {"template": pair, "port": "out", "per": per, "runs": _runs(n, per)}]}
    got = second_statements(copy.deepcopy(case))
    assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), got
    assert "second-wires" in got["codes"] and "second-links" not in got["codes"], got


def test_round_two_reads_each_unit_at_the_row_its_rows_name(pieces):
    """Round 2's rows, Lean's the Python's whole, over the two-template interior after a replay sample of two: a slot of
    a linked or wired port reads exactly its unit's row of its cell's replay block, of the rows a block the design
    places the port's value in, never another. TRIPLE's `x`, linked to the boundary's three rows a block, passes at
    rows 0 and 1 of each sampled block, or at rows 1 and 2, 1 and 0, or both at row 0, as its rows name, and is refused
    (`second-read`) at its block's first row only, at its units' rows swapped, at the next block's rows, in a tree of
    one row a block, or with rows that name another `per` or a row past the block's; wired to PAIR's `out`, it passes
    at its writer's units 0 and 1, or 1 and 2, as its rows name, and is refused at the rows of the layout its rows
    don't name or at a unit past its writer's three. Rows that don't hold each of the part's units once, in order (one
    unit, from unit 1, one unit more, for another port, or none), refuse the link (`second-links`), and rows a block
    other than the writer's units a block refuse the wire (`second-wires`)."""
    pair, triple = SECOND_MIXED.templates
    sample, units = [0, 2], [1, 4, 6, 9]
    linked = _second_case(SECOND_MIXED, True, sample, units, links={triple}, allow=True)
    wired = _second_case(SECOND_MIXED, True, sample, units, wires=[(triple, pair)], allow=True)
    lx, wx = (c["statements"][1]["registered"]["ports"]["x"] for c in (linked, wired))
    (lr,), (wr,) = linked["rows"], wired["rows"]
    assert (lr["per"], lr["runs"], lx["reads"], lx["leaves"]) == (3, [[0, 2, 0, 1]], [0, 1, 6, 7], 12)
    assert (wr["per"], wr["runs"], wx["reads"]) == (3, [[0, 2, 0, 0]], [0, 0, 3, 3])

    def at(c: dict, x: dict, runs=None, per=None, **e) -> dict:
        r = {**c["rows"][0], **{k: v for k, v in (("runs", runs), ("per", per)) if v is not None}}
        return {**_second_with(c, 1, registered={**c["statements"][1]["registered"], "ports": {"x": {**x, **e}}}),
                "rows": [r]}
    cases = [
        (at(linked, lx), []),
        (at(linked, lx, [[0, 2, 1, 1]], reads=[1, 2, 7, 8]), []),
        (at(linked, lx, [[0, 1, 1, 0], [1, 1, 0, 0]], reads=[1, 0, 7, 6]), []),
        (at(linked, lx, [[0, 1, 0, 0], [1, 1, 0, 0]], reads=[0, 0, 6, 6]), []),
        (at(linked, lx, reads=[0, 0, 6, 6]), ["second-read"]),
        (at(linked, lx, reads=[1, 0, 7, 6]), ["second-read"]),
        (at(linked, lx, reads=[3, 4, 9, 10]), ["second-read"]),
        (at(linked, lx, [[0, 1, 0, 0], [1, 1, 0, 0]]), ["second-read"]),
        (at(linked, lx, leaves=4, reads=[0, 0, 2, 2]), ["second-read"]),
        (at(linked, lx, per=2, reads=[0, 1, 4, 5]), ["second-read"]),
        (at(linked, lx, [[0, 2, 2, 1]], reads=[2, 3, 8, 9]), ["second-read"]),
        (at(linked, lx, [[0, 1, 0, 1]]), ["second-links"]),
        (at(linked, lx, [[1, 2, 0, 1]]), ["second-links"]),
        (at(linked, lx, [[0, 2, 0, 1], [2, 1, 0, 0]]), ["second-links"]),
        ({**linked, "rows": [{**lr, "port": "out"}]}, ["second-links"]),
        ({**linked, "rows": None}, ["second-links"]),
        (at(wired, wx, [[0, 2, 0, 1]], reads=[0, 1, 3, 4]), []),
        (at(wired, wx, [[0, 2, 1, 1]], reads=[1, 2, 4, 5]), []),
        (at(wired, wx, [[0, 2, 0, 1]]), ["second-read"]),
        (at(wired, wx, reads=[0, 1, 3, 4]), ["second-read"]),
        (at(wired, wx, [[0, 2, 2, 1]], reads=[2, 3, 5, 6]), ["second-read"]),
        (at(wired, wx, per=2), ["second-wires"]),
        (at(wired, wx, [[0, 1, 0, 0]]), ["second-wires"]),
        ({**wired, "rows": []}, ["second-wires"]),
    ]
    for case, codes in cases:
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (codes, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (codes, got)


def test_round_two_holds_each_port_a_header_shares_rows_at_to_a_registered_read(pieces):
    """`second-refs`, Lean's the Python's whole, over the two-template interior after a replay sample of two, PAIR reading
    its own `x` and TRIPLE its linked `x` at the boundary: a header sharing rows (``shared_rows``, as its template's
    reference does) only at ports its statement reads registered passes, PAIR's own `out` read at each slot's own rows
    and TRIPLE's at its own; a shared port left unread, whose refs the session's verifier would only bound by their
    table, is refused, as is a header sharing rows with no registered reads; a shared port read at another slot's rows
    is `second-read`'s."""
    pair, triple = SECOND_MIXED.templates
    sample, drawn = [0, 2], [1, 4, 6, 9]
    honest = _second_case(SECOND_MIXED, True, sample, drawn, allow=True)
    own = _second_case(SECOND_MIXED, True, sample, drawn, links={triple}, own={pair}, allow=True)
    both = _shared_at(own, 0, ("out", "x"), reads=("out",))
    mine = both["statements"][0]["registered"]["ports"]["x"]
    assert both["statements"][0]["header"]["shared_rows"] == {"out": 6, "x": 6} == both["refs"][pair]["shared_rows"]
    cases = [
        (both, []),
        (_shared_at(own, 0, ("x",)), []),
        (_shared_at(own, 1, ("out", "x"), reads=("out",)), []),
        (_shared_at(both, 1, ("out", "x"), reads=("out",)), []),
        (_shared_at(own, 0, ("out", "x")), ["second-refs"]),
        (_shared_at(own, 1, ("out", "x")), ["second-refs"]),
        (_shared_at(honest, 0, ("out", "x")), ["second-refs"]),
        (_second_with(both, 0, registered={**both["statements"][0]["registered"], "ports": {
            **both["statements"][0]["registered"]["ports"], "x": {**mine, "reads": mine["reads"][::-1]}}}),
         ["second-read"]),
        ({**both, "refs": own["refs"]}, ["second-header"]),
    ]
    for case, codes in cases:
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (codes, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (codes, got)


def test_round_two_holds_each_port_to_the_value_its_reference_reads(pieces):
    """`second-values`, Lean's the Python's whole, over the two-template interior after a replay sample of two: a
    statement's registered reads are its template's reference statement's but for the run's fields
    (`second_stage.blank_reads`). So TRIPLE reading its linked `x` passes against a reference reading the boundary's `x`
    with other roots, leaves and rows or none, and is refused against no reference reads, one for no template, null, of
    another format, reading another value, a port more or none; PAIR reading its own `x` is refused against a reference
    reading the boundary's. At a port no link names, which `second-read` lets read any value either registration
    registers at any rows, PAIR reading `w2` passes where its reference reads `w2` at its rows and is refused where it
    reads `w`, and the reverse, and reading either is refused where its reference reads it at other rows
    (`_read_registered`): the run never chooses the value, nor the rows."""
    pair, triple = SECOND_MIXED.templates
    sample, drawn = [0, 2], [1, 4, 6, 9]
    honest = _second_case(SECOND_MIXED, True, sample, drawn, allow=True)
    linked = _second_case(SECOND_MIXED, True, sample, drawn, links={triple}, allow=True)
    own = _second_case(SECOND_MIXED, True, sample, drawn, links={triple}, own={pair}, allow=True)
    ref = linked["ref_reads"][triple]
    x = ref["ports"]["x"]
    assert honest["ref_reads"] == {pair: None, triple: None}
    assert sorted(ref["ports"]) == ["x"] and x["value"] == BX

    def at(c: dict, t: str, r) -> dict:
        return {**c, "ref_reads": {**c["ref_reads"], t: r}}
    cases = [
        (linked, []),
        (at(linked, triple, {**ref, "ports": {"x": {**x, "root": "00" * 64, "leaves": 3, "reads": [0]}}}), []),
        (at(linked, triple, {**ref, "ports": {"x": {"value": BX}}}), []),
        ({**linked, "ref_reads": None}, ["second-values"]),
        ({**linked, "ref_reads": {pair: None}}, ["second-values"]),
        (at(linked, triple, None), ["second-values"]),
        (at(linked, triple, {**ref, "format": "verity/registered-reads/v0"}), ["second-values"]),
        (at(linked, triple, {**ref, "ports": {"x": {**x, "value": BOUT}}}), ["second-values"]),
        (at(linked, triple, {**ref, "ports": {"x": x, "out": {"value": "out"}}}), ["second-values"]),
        (at(linked, triple, {**ref, "ports": {}}), ["second-values"]),
        (at(own, pair, ref), ["second-values"]),
        (at(honest, pair, ref), ["second-values"]),
        (_read_registered(honest, 0), []),
        (_read_registered(honest, 3), []),
        (_read_registered(honest, 1), ["second-values"]),
        (_read_registered(honest, 2), ["second-values"]),
        (_read_registered(own, 1), ["second-values"]),
        (_read_registered(honest, 4), ["second-values"]),
        (_read_registered(honest, 7), ["second-values"]),
        (_read_registered(honest, 8), ["second-values"]),
    ]
    for case, codes in cases:
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (codes, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (codes, got)
    w2 = _read_registered(honest, 1)["statements"][0]["registered"]["ports"]["z"]
    assert w2["value"] == "w2" and w2["root"] == SECOND_BOUNDARY["registered"][1]["root"]


def test_round_two_reads_values_only_the_interior_registers_and_names_each(pieces):
    """Reads of values only the interior registers, Lean's the Python's whole (`second-read`, `second_interior_reads`),
    over the two-template interior after a replay sample of two. PAIR's statement reading such a value
    (`_read_interior`: a history table `h` at `z`, a passing read's `opened` at `ok`) passes at its root, domain and
    leaves. It is refused at another root, domain or leaf count, and reading a value neither registration registers
    (`second-read`). Each such read is named (`second_interior_reads`, at another root too), and none of the honest
    statements, a linked read or a read of the boundary's `w` is. The attached record carries the names where there are
    any (`second_stage.attach`)."""
    pair, triple = SECOND_MIXED.templates
    sample, units = [0, 2], [1, 4, 6, 9]
    honest = _second_case(SECOND_MIXED, True, sample, units, allow=True)
    linked = _second_case(SECOND_MIXED, True, sample, units, links={triple}, allow=True)
    for x in range(10):
        case, codes = _read_interior(honest, x), [] if x < 2 else ["second-read"]
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (x, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (x, got)
    h, opened = ({"template": pair, "port": p, "value": v} for p, v in (("z", "h"), ("ok", "opened")))
    for case, want in [(honest, []), (linked, []), (_read_registered(honest, 0), []), (_read_interior(honest, 0), [h]),
                       (_read_interior(honest, 1), [opened]), (_read_interior(honest, 2), [h]),
                       (_read_interior(honest, 8), [])]:
        got = second_interior_reads(copy.deepcopy(case))
        assert D.replay(SECOND_INTERIOR_READS, case, pieces) == ("output", D.canonical(got)), (want, got)
        assert got == want, (want, got)
    _, own, statements = _second_own(_read_interior(honest, 0))
    v = SS.check(True, own, statements)
    assert v.ok and SS.attach({"checks": {}}, v, SS.interior_reads(own, statements))["interior_reads"] == [h]
    assert "interior_reads" not in SS.attach({"checks": {}}, v)

def test_round_two_reads_a_linked_port_at_the_value_the_interior_registers_for_it(pieces):
    """A port a link names reads the value the interior registers under the link's name, where it registers one (one
    the boundary record has no port of: `second-interior-names`), and never the boundary port then, each slot at its
    unit's row of its cell's sampled block (`readOk`, `wiredRow`): Lean's the Python's whole (`second-read`,
    `second_interior_reads`), over the two-template interior after a replay sample of two with TRIPLE's `x` linked
    (`_read_linked`). The read as placed passes; at the boundary port, at another root, domain or leaf count, or at the
    boundary port's rows it is refused (`second-read`). Each such read is named (`second_interior_reads`, at another
    root too), and with no such value the boundary port is read as before. A value the verifier supplies (a table's
    root, an index) reaches a template's parameter this way."""
    _, triple = SECOND_MIXED.templates
    linked = _second_case(SECOND_MIXED, True, [0, 2], [1, 4, 6, 9], links={triple}, allow=True)
    for x in range(6):
        case, codes = _read_linked(linked, x), [] if x == 0 else ["second-read"]
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (x, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (x, got)
    read = {"template": triple, "port": "x", "value": linked["links"][0]["value"]}
    for case, want in [(linked, []), (_read_linked(linked, 0), [read]), (_read_linked(linked, 2), [read])]:
        got = second_interior_reads(copy.deepcopy(case))
        assert D.replay(SECOND_INTERIOR_READS, case, pieces) == ("output", D.canonical(got)), (want, got)
        assert got == want, (want, got)


def test_round_two_refuses_an_interior_that_registers_a_name_round_one_has(pieces):
    """Each name a session reads is one registration's (`namesOk`): round 2 refuses an interior record registering a
    value of a name the boundary record has, a port or a registered value (`second-interior-names`), Lean's the
    Python's whole, so the interior's value of a name, registered after the first stage's coins, never stands in for
    one round 1 committed. Over the two-template interior after a replay sample of two: a value of each of the
    boundary's ports' names and of each value it registers (`_shadowing`), with no port linked; and with TRIPLE's `x`
    linked, the interior's value of the link's name beside the boundary's port of it (`_read_linked`, `kept`), refused
    though the read as placed passes `second-read`, and with it at the boundary port's root."""
    _, triple = SECOND_MIXED.templates
    honest = _second_case(SECOND_MIXED, True, [0, 2], [1, 4, 6, 9], allow=True)
    linked = _second_case(SECOND_MIXED, True, [0, 2], [1, 4, 6, 9], links={triple}, allow=True)
    names = len(SECOND_BOUNDARY["roots"]) + len(SECOND_BOUNDARY["registered"])
    cases = [*((_shadowing(honest, x), ["second-interior-names"]) for x in range(names)),
             (_read_linked(linked, 0, kept=True), ["second-interior-names"]),
             (_read_linked(linked, 1, kept=True), ["second-interior-names", "second-read"])]
    for case, codes in [(honest, []), (linked, []), *cases]:
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (codes, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (codes, got)


def test_round_two_holds_each_statement_to_the_verdict_that_checked_it(pieces):
    """Round 2 reads a statement only as a verifier of record checked it, Lean's the Python's whole (`second-verified`):
    over the two-template interior the honest statements pass, and a statement that passes every other check is refused
    when the public file its session was checked against has a header naming another circuit file, or its session's
    verdict of record is gone, didn't accept, is not of record, or names another sample, other registered reads or
    none, or reads where it read none. And only as one table (`second-tables`): a verdict of record naming it that set
    its session up as two tables, or names no table count, is refused, even beside one that names one table."""
    triple = SECOND_MIXED.templates[1]
    sample, drawn = [0, 2], [1, 4, 6, 9]
    honest = _second_case(SECOND_MIXED, True, sample, drawn, allow=True)
    linked = _second_case(SECOND_MIXED, True, sample, drawn, links={triple}, allow=True)
    s0, v0, v1 = honest["statements"][0], honest["verdicts"]["lean-verify/0"], linked["verdicts"]["lean-verify/1"]

    def verdict(c: dict, name: str, v: dict) -> dict:
        return {**c, "verdicts": {**c["verdicts"], name: v}}
    elsewhere = {**s0, "header": {**s0["header"], "circuit_sha512": "ab" * 64}}
    cases = [
        (honest, []),
        (linked, []),
        (verdict(honest, "lean-verify/0", {**v0, **_checked_names(elsewhere)}), ["second-verified"]),
        ({**honest, "verdicts": {k: v for k, v in honest["verdicts"].items() if k != "lean-verify/0"}},
         ["second-verified"]),
        (verdict(honest, "lean-verify/0", {**v0, "accepted": False}), ["second-verified"]),
        (verdict(honest, "lean-verify/0", {**v0, "of_record": False}), ["second-verified"]),
        (verdict(honest, "lean-verify/0", {**v0, "unit_draw": {**s0["draw"], "units": [0, 4]}}), ["second-verified"]),
        (verdict(honest, "lean-verify/0", {**v0, "registered_sha512": "00" * 64}), ["second-verified"]),
        (verdict(linked, "lean-verify/1", {**v1, "registered_sha512": "00" * 64}), ["second-verified"]),
        (verdict(linked, "lean-verify/1", {k: v for k, v in v1.items() if k != "registered_sha512"}),
         ["second-verified"]),
        (verdict(honest, "lean-verify/0", {**v0, "session_tables": 2}), ["second-tables"]),
        (verdict(honest, "lean-verify/0", {k: v for k, v in v0.items() if k != "session_tables"}), ["second-tables"]),
        (verdict(honest, "lean-verify/split", {**v0, "session_tables": 2}), ["second-tables"]),
    ]
    for case, codes in cases:
        got = second_statements(copy.deepcopy(case))
        assert D.replay(SECOND_STATEMENTS, case, pieces) == ("output", D.canonical(got)), (codes, got)
        assert got["codes"] == codes and got["ok"] == (codes == []), (codes, got)


def test_round_two_derives_its_links_from_the_replay_block(pieces):
    """The design's links, Lean's the Python's whole, derived from the registered replay block's Definition alone:
    `BLOCK`'s PAIR batch reads its `x` and it returns PAIR's output, so PAIR's `x` and `o0` are the boundary's ports
    (named by template where the record names its ports so, else plainly); the chain's PAIR reads its `x` and it returns
    ID8's output; a block returning PAIR's and TRIPLE's outputs as two fields, a tuple's either way round or a record's,
    links each one's `o0` to its field's port (`o<j>`); and none where the block calls a template twice, a primitive
    reads its parameter, a field is a parameter, a template's output twice or part of one, the program calls PAIR
    elsewhere too, or the key names a template or no Definition."""
    def case(i, key=None, rec=None):
        prog, k = REPLAY_PROGRAMS[i]
        return {"program": _json(codec.encode_program(prog)), "replay": key or k, "boundary": rec}
    chain, pair, id8 = codec.descriptor_id(REPLAYS[1]), codec.descriptor_id(PAIR), codec.descriptor_id(ID8)
    triple = codec.descriptor_id(TRIPLE)
    both = [{"port": "x", "template": pair, "value": "x"}, {"port": "x", "template": triple, "value": "z"}]

    def outs(*ts):
        return [{"port": "o0", "template": t, "value": f"o{j}"} for j, t in enumerate(ts)]
    cases = [
        (case(0, rec=SECOND_BOUNDARY), [{"port": "x", "template": pair, "value": BX},
                                        {"port": "o0", "template": pair, "value": BOUT}]),
        (case(0), [{"port": "x", "template": pair, "value": "x"}, {"port": "o0", "template": pair, "value": "o0"}]),
        (case(1, rec={"roots": [{"port": f"{chain}/x"}]}), [{"port": "x", "template": pair, "value": f"{chain}/x"},
                                                           {"port": "o0", "template": id8, "value": f"{chain}/o0"}]),
        (case(4), both + outs(pair, triple)),
        (case(5), both + outs(triple, pair)),
        (case(6), both + outs(pair, triple)),
        *((case(i), None) for i in (2, 3, 7, 8, 9, 10)),
        (case(0, key=pair), None),
        (case(0, key="Other_v0"), None),
    ]
    for c, want in cases:
        got = replay_links(copy.deepcopy(c))
        assert got == want and D.replay(REPLAY_LINKS, c, pieces) == ("output", D.canonical(got)), (c["replay"], got)


def test_the_designs_wires_lean_the_pythons_whole(pieces):
    """The design's wires, Lean's the Python's whole, derived from the registered replay block's Definition alone: the
    chain's ID8 reads PAIR's whole output, and so does each instance of a batch of ID8 sharing it; none to derive where
    no node reads another (`BLOCK`, a primitive over the parameter, two templates' outputs returned, or the key names a
    template over its parameters); and none where a template reads a node beside other rows, a primitive reads a
    template's output, or the key names a primitive or no Definition."""
    def case(i, key=None):
        prog, k = REPLAY_PROGRAMS[i]
        return {"program": _json(codec.encode_program(prog)), "replay": key or k}
    pair, id8, n = codec.descriptor_id(PAIR), codec.descriptor_id(ID8), len(REPLAYS) + 1
    read = [{"port": "y", "template": id8, "wport": "o0", "writer": pair}]
    cases = [(case(0), []), (case(1), read), (case(3), []), (case(4), []), (case(0, key=pair), []), (case(n), read),
             (case(n + 1), None), (case(2), None), (case(0, key=codec.descriptor_id(XOR8)), None),
             (case(0, key="Other_v0"), None)]
    for c, want in cases:
        got = replay_wires(copy.deepcopy(c))
        assert got == want and D.replay(REPLAY_WIRES, c, pieces) == ("output", D.canonical(got)), (c["replay"], got)


def test_the_designs_rows_place_each_callers_units(pieces):
    """The design's links, wires and rows, Lean's the Python's whole, derived once from the replay block's Definition
    for each template port, its callers' units in body order: `BLOCK`'s PAIR batch reads its member's row of `x` and
    writes its row of `o0`, three a block; PAIR called over each of two rows reads row `j` at unit `j`, or row `1 - j`
    the other way round, and writes row `j` of the output; ID8 over the second PAIR's output reads PAIR's unit 1 of two,
    its own output a field of one row; ID8 over DUO's second output reads it at DUO's `o1`. PAIR's callers reading two
    parameters, or a slice not one of its rows, or one the parameter and the other SWAP's output, make no links, and
    the last no wires either; an ID8 batch mapped over a PAIR batch's outputs, or PAIR over two ID8 outputs, makes no
    wires; the rows are then only those of the others. DUO over each of two rows, returned one field of each, links
    that field's port (`o<f>`), each field returned its own; its first output at one caller and its second at the other
    links neither."""
    pair, id8, duo = (codec.descriptor_id(d) for d in (PAIR, ID8, DUO))
    pairs, crossed, fan, field, apart, askew, members, kinds, gathered, heads, tails, halves, uneven = MULTI_REPLAYS

    def link(t, port, value):
        return {"port": port, "template": t, "value": value}

    def rows(*rs):
        return [{"per": per, "port": port, "runs": runs, "template": t} for t, port, per, runs in rs]
    one, two = [[0, 1, 0, 0]], [[0, 2, 0, 1]]
    cases = [
        (BLOCK, [link(pair, "x", "x"), link(pair, "o0", "o0")], [],
         rows((pair, "x", 3, [[0, 3, 0, 1]]), (pair, "o0", 3, [[0, 3, 0, 1]]))),
        (pairs, [link(pair, "x", "x"), link(pair, "o0", "o0")], [], rows((pair, "x", 2, two), (pair, "o0", 2, two))),
        (crossed, [link(pair, "x", "x"), link(pair, "o0", "o0")], [],
         rows((pair, "x", 2, [[0, 1, 1, 0], [1, 1, 0, 0]]), (pair, "o0", 2, two))),
        (fan, [link(pair, "x", "x"), link(pair, "o0", "o0"), link(id8, "o0", "o1")],
         [{"port": "y", "template": id8, "wport": "o0", "writer": pair}],
         rows((pair, "x", 2, two), (pair, "o0", 2, two), (id8, "o0", 1, one), (id8, "y", 2, [[0, 1, 1, 0]]))),
        (field, [link(duo, "x", "x"), link(id8, "o0", "o0")],
         [{"port": "y", "template": id8, "wport": "o1", "writer": duo}],
         rows((duo, "x", 1, one), (id8, "o0", 1, one), (id8, "y", 1, one))),
        (apart, None, [], []),
        (askew, None, [], []),
        (members, [link(pair, "x", "x"), link(id8, "o0", "o0")], None, rows((pair, "x", 2, two), (id8, "o0", 2, two))),
        (kinds, None, None, []),
        (gathered, [link(id8, "y", "x"), link(pair, "o0", "o0")], None, rows((id8, "y", 2, two), (pair, "o0", 1, one))),
        (heads, [link(duo, "x", "x"), link(duo, "o0", "o0")], [], rows((duo, "x", 2, two), (duo, "o0", 2, two))),
        (tails, [link(duo, "x", "x"), link(duo, "o1", "o0")], [], rows((duo, "x", 2, two), (duo, "o1", 2, two))),
        (halves, [link(duo, "x", "x"), link(duo, "o0", "o0"), link(duo, "o1", "o1")], [],
         rows((duo, "x", 2, two), (duo, "o0", 2, two), (duo, "o1", 2, two))),
        (uneven, None, [], []),
    ]
    for r, links, wires, want in cases:
        c = {"program": _json(codec.encode_program(P.population_program(r, 2))), "replay": codec.descriptor_id(r),
             "boundary": None}
        for port, mirror, expected in ((REPLAY_LINKS, replay_links, links), (REPLAY_WIRES, replay_wires, wires),
                                       (REPLAY_ROWS, replay_rows, want)):
            got = mirror(copy.deepcopy(c))
            assert got == expected and D.replay(port, c, pieces) == ("output", D.canonical(got)), (c["replay"], got)


def test_a_replay_block_returns_its_units_fields_on_words_of_their_own(pieces):
    """The bits a replay block returns, Lean's the Python's whole, counted from its Definition's return: each unit's
    returned field on 16-bit words of its own. `BLOCK` returns three PAIR outputs of 8 bits, a word each; ID8 over DUO's
    second output one word; DUO over each of two rows two words where one field of each is returned and four where
    both are; four units of an 18-bit value two words each, twice their 72 bits; and two units of eight such values in
    one field nine words each, the 144 bits they hold. A replay block whose links can't be derived returns none, as do
    a key naming one of its templates and a key naming no Definition."""
    pairs, crossed, fan, field, apart, askew, members, kinds, gathered, heads, tails, halves, uneven = MULTI_REPLAYS
    wides, packs = WIDE_REPLAYS
    cases = [(BLOCK, None, 3 * 16), (pairs, None, 2 * 16), (fan, None, 3 * 16), (field, None, 16), (heads, None, 2 * 16),
             (halves, None, 4 * 16), (wides, None, 4 * 32), (packs, None, 2 * 144), (apart, None, None),
             (uneven, None, None), (REPLAYS[3], None, None), (BLOCK, codec.descriptor_id(PAIR), None),
             (BLOCK, "Other_v0", None)]
    for r, key, want in cases:
        c = {"program": _json(codec.encode_program(P.population_program(r, 2))),
             "replay": key or codec.descriptor_id(r), "boundary": None}
        got = replay_returned(copy.deepcopy(c))
        assert got == want and D.replay(REPLAY_RETURNED, c, pieces) == ("output", D.canonical(got)), (c["replay"], got)


def test_a_windows_run_from_the_challengers_coins(pieces):
    """A window of two templates' run (`SETUPS[3]`: PAIR and TRIPLE, one stratum each under `stratified-rate:1/2`) sampling from
    the auditor's challenger, as a `Stream` over a window runs it: its coins a ledger session's challenger opened after
    the registration's digest and read by the selection rule's sample (`Drawn`), Lean's `Run.run` the Python's whole. The
    honest run is accepted, each stratum sampled at least its floor. It is refused at its registration for a window of
    other counts (`R2-program`) or the templates in the other order (`query`), and for a record whose sampling design samples a
    stratum short of its floor (`R1-law`); at its select (`coins`) for each mutation of the coins that changes the
    segment or the bytes, coins opened after another digest, coins of a session the ledger doesn't open withheld,
    evidence for another window, and none or empty."""
    rule, s = service.Law("stratified-rate:1/2"), SETUPS[3]
    a = _v1_audit("stratified", s)
    rec = a["record"]
    named = service.selection_law(rule, rec["population"], service.selection_sizes(rule, s.program, s.query))
    law = DR.derive(s.program, s.query, named,
                    service.selection_table(rule, s.program, s.query, None))

    def reads(coins):
        got = bytearray()

        def source(n):
            b = coins(n)
            got.extend(b)
            return b
        service.select_draw(rule, law, rec["population"], source)
        return bytes(got)

    def case_of(d: Drawn, segment: dict, read: bytes, **change) -> dict:
        c = _run_case(a, read.hex(), 2, ["credited"])
        c["params"] = {**c["params"], "window_root": d.ledger.window["root"]}
        return {**c, "coins": d.evidence(segment), **change}

    def ran(c: dict) -> dict:
        out = run(copy.deepcopy(c))
        assert D.replay(RUN, c, pieces) == ("output", D.canonical(out)), out
        return out

    for seed in range(3):
        rng = random.Random(f"window/{seed}")
        d = Drawn(rng, before=True, after=bytes.fromhex(R.digest(rec)), reads=reads)
        honest = case_of(d, d.segment.to_json(), d.read)
        out = ran(honest)
        assert out["symbol"] == "accept" and out["read"] == len(d.read), out
        for stratum in law["strata"]:
            drawn = [u for u in out["draw"]["units"] if any(lo <= u < hi for lo, hi in stratum["units"])]
            assert len(drawn) == stratum["k"] >= 1, (stratum, out["draw"])
        other_counts = json.loads(_population(((0, 3), (1, 2))))
        reordered = Setup(P.mixed_population_program([(TRIPLE, 3), (PAIR, 2)]), P.template_instances_query([TRIPLE.id, PAIR.id]))
        PROGRAMS[reordered.sha512] = reordered.program
        short = copy.deepcopy(rec)
        short["law"]["strata"][0]["k"] = 0
        registers = {"R2-program": {**honest, "params": {**honest["params"], "program": other_counts}},
                     "query": {**honest, "params": {**honest["params"], "program": reordered.desc, "query": reordered.query}},
                     "R1-law": {**honest, "record": short}}
        for code, c in registers.items():
            out = ran(c)
            assert out["step"] == "register" and code in out["codes"], (code, out)
        tampered = bytearray(d.read)
        tampered[rng.randrange(len(tampered))] ^= 1 << rng.randrange(8)
        selects = [case_of(d, _mutated(rng, d, name)[0], d.read) for name in DRAW_MUTATIONS
                   if name not in ("unread", "read-tampered", "other-after")]
        selects.append(case_of(d, d.segment.to_json(), bytes(tampered)))
        elsewhere = Drawn(rng, reads=reads)
        bare = Drawn(rng, before=True, after=bytes.fromhex(R.digest(rec)), reads=reads, withheld=False)
        selects += [case_of(elsewhere, elsewhere.segment.to_json(), elsewhere.read),
                    case_of(bare, bare.segment.to_json(), bare.read),
                    {**honest, "params": {**honest["params"], "window_root": d.other_window(rng)}},
                    {**honest, "coins": None}, {**honest, "coins": {}}]
        for c in selects:
            out = ran(c)
            assert (out.get("step"), out.get("codes")) == ("select", ["coins"]), out


def test_the_audit_reads_its_draw_from_the_challengers_coins(server):
    """A one-stage audit whose verifier names the auditor's challenger (`challenger`, with `window_root`) reads its sample's
    coins through `Challenger.drawCoins` (`coins.draw_coins`): under the verifier's key, on its window, of its family,
    released after the registration's digest, Lean's the Python's. The sample from those coins is accepted; it is rejected
    for another sample, the evidence's coins carrying a key (`coins_reason`'s form, where the prover named the key), none,
    coins under another key, on another window, of another family, opened after another digest, each mutation of the
    segment, or a session the ledger doesn't open withheld (one whose challenger may have released its root before the
    prover's commitment), and refused for a challenger without a window. A verifier that names no challenger decides as
    before, and rejects evidence that carries coins."""
    s, law = SETUPS[3], {"k": 3, "law": "stratified"}
    mine = DR.derive(s.program, s.query, law, None)
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF)
    after = bytes.fromhex(R.digest(rec))

    def drawn_from(src: bytes) -> dict:
        at = [0]

        def source(k: int) -> bytes:
            at[0] += k
            return src[at[0] - k:at[0]]
        return draw_first(OwnRandomness(source), mine, s.population)

    def reads(coins) -> bytes:
        got = bytearray()

        def source(k: int) -> bytes:
            b = coins(k)
            got.extend(b)
            return b
        draw_first(OwnRandomness(source), mine, s.population)
        return bytes(got)

    def audit_of(d: Drawn, draw: dict, coins, **params) -> dict:
        proved = DR.proved(draw)
        evidence = {"draw": draw, "instances": len(proved), "proved_units": proved,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                    "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
        if coins is not None:
            evidence["coins"] = coins
        p = {"anchors": [], "challenger": {"key": d.s.key.to_json(), "label": "draw"}, "delta": "1/20", "law": law,
             "leaf_layer": LEAF, "log": [], "partition": None, "ports": s.ports, "program": s.desc, "query": s.query,
             "registered": None, "scheme": SCHEME, "window_root": d.ledger.window["root"], "work_table": None}
        return {"evidence": evidence, "params": {k: v for k, v in (p | params).items() if k not in params or v is not None},
                "record": rec}

    def verdict(a: dict) -> str:
        return D.replay(ONE_STAGE, _json(_case(a)), server)[0]

    def accepts(a: dict) -> None:
        assert verdict(a) == "accept", answer(_json(_case(a)))["reason"]

    for seed in range(3):
        rng = random.Random(f"one-stage-coins/{seed}")
        d = Drawn(rng, before=True, after=after, reads=reads)
        coins = {k: v for k, v in d.evidence(d.segment.to_json()).items() if k != "key"}
        draw = drawn_from(d.read)
        accepts(audit_of(d, draw, coins))
        other = next(x for x in (drawn_from(rng.randbytes(4 * len(d.read) + 256)) for _ in range(64)) if x["units"] != draw["units"])
        stranger = next(x for x in (Drawn(rng, before=True, after=after, reads=reads) for _ in range(16))
                        if x.s.key != d.s.key)
        elsewhere = Drawn(rng, reads=reads)
        rejected = [audit_of(d, other, coins), audit_of(d, draw, d.evidence(d.segment.to_json())),
                    audit_of(d, draw, None), audit_of(d, draw, coins, challenger={"key": stranger.s.key.to_json(),
                                                                                 "label": "draw"}),
                    audit_of(d, draw, coins, window_root=d.other_window(rng)),
                    audit_of(d, draw, coins, challenger={"key": d.s.key.to_json(), "label": "other"}),
                    audit_of(elsewhere, drawn_from(elsewhere.read),
                             {k: v for k, v in elsewhere.evidence(elsewhere.segment.to_json()).items() if k != "key"}),
                    audit_of(d, draw, coins, challenger=None)]
        rejected += [audit_of(d, draw, {**coins, "segment": _mutated(rng, d, name)[0]}) for name in DRAW_MUTATIONS
                     if name not in ("unread", "read-tampered", "other-after")]
        bare = Drawn(rng, before=True, after=after, reads=reads, withheld=False)
        bare_coins = {k: v for k, v in bare.evidence(bare.segment.to_json()).items() if k != "key"}
        rejected.append(audit_of(bare, drawn_from(bare.read), bare_coins))
        with pytest.raises(CoinError, match="withhold"):
            draw_coins(bare.s.key, bare.ledger.window["root"], "draw", after, bare_coins, lms_verify)
        assert [verdict(a) for a in rejected] == ["reject"] * len(rejected)
        refused = audit_of(d, draw, coins, window_root=None)
        assert answer(_json(_case(refused)))["reason"].startswith("params:") and verdict(refused) == "reject"
        accepts(audit_of(d, draw, None, challenger=None))


def _ran(pieces, case: dict) -> dict:
    out = run(copy.deepcopy(case))
    assert D.replay(RUN, case, pieces) == ("output", D.canonical(out)), out
    return out


def test_the_run_refuses_a_bernoulli_law(pieces):
    """A registration under a Bernoulli sampling design is refused at its registration at any p (`bernoulli`,
    `Draw.bernoulli`), Lean's the Python's whole: at 0 < p < 1 its sessions would set up tables of different shapes, and at
    p = 0 the stream would accept having verified nothing; R1-R7 come first. Under subset, stratified, work and All the run
    goes on to accept."""
    source = hashlib.shake_256(b"run").hexdigest(600)
    for s, p in [(SETUPS[3], "0/1"), (SETUPS[3], "0/4"), (SETUPS[3], "1/1"), (SETUPS[3], "3/3"), (SETUPS[3], "1/2"),
                 (SETUPS[3], "2/3"), (SETUPS[2], "0/1"), (SETUPS[2], "1/1"), (SETUPS[2], "1/2")]:
        case = _run_case(_v1_audit("bernoulli", s, p), source, 2, ["credited"])
        assert _ran(pieces, case) == {"codes": ["bernoulli"], "step": "register"}, p
        out = _ran(pieces, {**case, "record": {**case["record"], "previous": "11" * 64}})
        assert out["step"] == "register" and out["codes"] != ["bernoulli"], (p, out)
    for kind in ("subset", "stratified", "work"):
        assert _ran(pieces, _from_challenger(_run_case(_v1_audit(kind), "", 2, ["credited"]),
                                             random.Random(kind))[1])["symbol"] == "accept", kind
    every = {**_run_case(_v1_audit("all"), "", 2, ["credited"]), "rule": {"rule": "all"}}
    assert _ran(pieces, every)["symbol"] == "accept"


def test_the_run_refuses_a_closure_map(pieces):
    """A work registration's run given a closure map, empty or not, is refused at its select step before any coin is read
    (`closure`), Lean's the Python's whole: a work sample it widens proves a set whose size varies with the sample, which the
    soundness statement doesn't cover. Without one it is accepted."""
    source = hashlib.shake_256(b"run").hexdigest(600)
    for s in (SETUPS[3], SETUPS[2]):
        case = _run_case(_v1_audit("work", s), source, 2, ["credited"])
        n = case["record"]["population"]
        for closure in ([], [[0, [1]]], [[u, [(u + 1) % n]] for u in range(n)]):
            assert _ran(pieces, {**case, "closure": closure}) == {"codes": ["closure"], "step": "select"}, closure
        assert _ran(pieces, _from_challenger(case, random.Random(n))[1])["symbol"] == "accept"


def _covered_case(kind: str, p: str = "1/2") -> tuple[dict, dict, dict]:
    """An honest v1 audit of `SETUPS[3]` under `kind`, as params, record and evidence: `bernoulli` at `p`, `subset`,
    `stratified`, `work`, `all` (the sampling design the stream's All rule names, `subset` over every unit), or `closure`,
    a work sample that carries a closure."""
    s = SETUPS[3]
    n = s.population
    law = ({"law": "bernoulli", "p": p} if kind == "bernoulli" else service.selection_law(service.All(), n) if kind == "all"
           else {"k": 2, "law": "subset"} if kind == "subset" else {"k": 3, "law": "work" if kind == "closure" else kind})
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}} if law["law"] == "work" else None
    mine = DR.derive(s.program, s.query, law, table)
    rec = R.record(s.program, s.query, law, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, work_table=table)
    if law["law"] == "bernoulli":
        units = {Fraction(0): [], Fraction(1): list(range(n))}.get(Fraction(p), [1, 2, 4])
    elif law["law"] == "subset":
        units = list(range(law["k"]))
    else:
        units = sorted(u for st_ in mine["strata"] for u in [u for lo, hi in st_["units"] for u in range(lo, hi)][:st_["k"]])
    d = {**mine, "population": n, "units": units}
    if kind == "closure":
        d["closure"] = [u for u in range(n) if u not in units][:1]
    proved = DR.proved(d)
    evidence = {"draw": d, "instances": len(proved), "proved_units": proved,
                "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
    params = {"anchors": [], "delta": "1/20", "law": law, "leaf_layer": LEAF, "log": [],
              "partition": P.partition(s.program, s.query), "ports": s.ports, "program": s.desc, "query": s.query,
              "registered": None, "scheme": SCHEME, "work_table": table}
    return params, rec, evidence


def test_the_audit_refuses_what_soundness_does_not_cover(server, server_v2):
    """The one-stage audit, v1 (`OneStage.protocol`) and v2 (`OneStageZk`), Lean's the Python's: parameters under a
    Bernoulli law at 0 < p < 1 are refused (`paramsOfJson`), and evidence whose sample carries a closure
    (`evidenceOf`), after its commitment's digest, v2's parameters naming a challenger at the deployed LMS parameters so
    that its refusals of parameters naming none (`NO_CHALLENGER`) or another key (`KEY_PARAMS`) aren't what refuse them;
    the same work audit without the closure is accepted by v1."""
    for kind, stage in (("bernoulli", "params: "), ("closure", "evidence: ")):
        params, rec, evidence = _covered_case(kind)
        case = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        out = server.exchange(case)
        assert out == answer(case) | {"reason": out["reason"]} and out["verdict"] == "reject", (kind, out)
        assert out["reason"].startswith(stage) and "soundness statement doesn't cover" in out["reason"], (kind, out)
        circuit, public, record = STAND_IN_CIRCUIT, b"not a public file", b"{}"
        base = {k: v for k, v in evidence.items() if k != "verdicts"}
        d = Drawn(random.Random(kind), params=[DEPLOYED])
        v2 = {"blobs": [circuit.hex(), public.hex(), record.hex()], "commitment": _hex(rec),
              "evidence": _hex(base | {"proofs": {"sessions": []}}),
              "params": params | {"challenger": {"key": d.s.key.to_json(), "label": "draw"},
                                  "proofs": {"circuit": _sha(circuit), "public": _sha(public), "registered": None},
                                  "window_root": d.ledger.window["root"]}}
        out = server_v2.exchange(_json(v2))
        assert out["verdict"] == "reject" and out["reason"].startswith(stage), (kind, out)
        assert "soundness statement doesn't cover" in out["reason"], (kind, out)
        if kind == "bernoulli":
            for p in ("0/1", "0/4", "1/1", "3/3", "2/3"):
                law = {"law": "bernoulli", "p": p}
                one, two = {**case, "params": {**params, "law": law}}, {**v2, "params": {**v2["params"], "law": law}}
                out = server.exchange(one)
                assert out == answer(one) | {"reason": out["reason"]}, (p, out)
                assert out["verdict"] == "reject" and out["reason"] == "params: " + service.BERNOULLI, (p, out)
                out = server_v2.exchange(_json(two))
                assert out["verdict"] == "reject" and out["reason"] == "params: " + service.BERNOULLI, (p, out)
    for kind in ("subset", "stratified", "work", "all"):
        params, rec, evidence = _covered_case(kind)
        case = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        assert D.replay(ONE_STAGE, case, server)[0] == "accept", kind
        two = {**v2, "params": {**v2["params"], "law": params["law"], "work_table": params["work_table"]}}
        out = server_v2.exchange(_json(two))
        assert not (out["reason"] or "").startswith("params: "), (kind, out)


def test_the_run_refuses_a_population_out_of_order(pieces):
    """A registration whose program's units, read stratum by stratum, are not its N units in order is refused at its
    registration (`population-order`, `Slots.popOk`), Lean's the Python's whole: `SETUPS[4]` puts PAIR's units 0, 3 and 4
    in one stratum ahead of TRIPLE's 1 and 2, and `SETUPS[5]`'s Calls put PAIR's units 0 and 4 ahead of the Fork's. The
    sampled audit's knowledge soundness reads slot `u` as unit `u`. Programs whose strata are blocks in order go on."""
    source = hashlib.shake_256(b"run").hexdigest(600)
    for s, ok in ((SETUPS[4], False), (SETUPS[5], False), (SETUPS[3], True), (SETUPS[2], True), (SETUPS[6], True)):
        assert P.in_order(s.program, s.query, s.population) is ok, s.templates
        case = _run_case(_v1_audit("subset", s), source, 2, ["credited"])
        out = run(copy.deepcopy(case))
        assert D.replay(RUN, case, pieces) == ("output", D.canonical(out)), out
        if ok:
            assert out.get("codes") != ["population-order"], out
        else:
            assert (out.get("step"), out.get("codes")) == ("register", ["population-order"]), out
    assert not P.in_order(SETUPS[3].program, SETUPS[3].query, SETUPS[3].population + 1)


#: The public files Flock's producers wrote, recorded in `verity/protocols/flock/prover/tests`' fetched fixtures
#: (`fixtures/artifacts.json`): `verity_flock`'s staging (`json.dumps(sort_keys=True)`), and the selftest's and the
#: recursive links' (compact, sorted).
RECORDED = Path(__file__).resolve().parents[6] / "verity/protocols/flock/prover/tests/fixtures"
RECORDED_FIXTURES = ("forged-proofs", "public-inputs", "public-inputs-blocks", "rec-coins", "rec-links", "zk-sessions")


def _recorded_publics() -> list[Path]:
    missing = [f"verity/protocols/flock/prover/tests/fixtures/{f}" for f in RECORDED_FIXTURES if not (RECORDED / f).is_dir()]
    assert not missing, f"not fetched: `uv run research data fetch-fixtures {' '.join(missing)}`"
    return sorted(p for f in RECORDED_FIXTURES for p in (RECORDED / f).rglob("*.bin")
                  if re.fullmatch(r"(.+\.)?pub(-\d+)?\.bin", p.name))


def test_the_recorded_public_files_pass_the_population_check(pieces):
    """Each public file the producers recorded (`RECORDED`) passes `FilePart.check` against a population of as many units
    in order as it names, its header read by `FilePart.header` (`Flock.JsonParse.parse`, then `Flock.norm`) as Python's
    `json.loads` reads it: the check refuses none of them."""
    files = _recorded_publics()
    assert len(files) >= 23, files
    query = P.template_instance_query(MEMBERS[0].id)
    for p in files:
        b = p.read_bytes()
        n = len(json.loads(b.split(b"\n", 1)[0])["units"]["indices"])
        case = {"program": json.loads(_population(((0, n),))), "query": query, "public": b.hex()}
        assert file_population(copy.deepcopy(case)) == {"why": None}, p
        assert D.replay(FILE_POPULATION, case, pieces) == ("output", D.canonical({"why": None})), p
