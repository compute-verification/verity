"""`one_stage`'s Python held to SPOT's one-stage audit in Lean (`core_spot`), its definition and verifier (`one_stage.verifier.RECORD_VERDICT`): the whole protocol through its own `serve` (`spot-one-stage`, a
`ProtocolPort`), and its pieces through `spot-difftest` (`Port`s), on sampled programs, sampling designs, records, samples and
verdicts, valid ones and one or two changes to each. A disagreement is the Python's to fix, unless it finds the Lean wrong.

The Python held to it is `one_stage` itself (`registration.check`, `draw.derive`, `check_draw`, `audit.audit_record`, `profile`,
`wrong_units`, `registered.registered_entries` and `domain`) behind the protocol's decoders, which this file writes as the
Lean reads them (`OneStage.paramsOfJson`, `commitmentOf`, `evidenceOf`), for the `draw` piece `coins.draw_first` on
`OwnRandomness` over the given bytes, for the `link_points` piece `link_points` (its sampling design held as `draw_first` holds
one, `coins.sampling_design_problem`), for the `select` piece the challenger's sample from a Spec's selection rule on
(`service.selection_sampling_design`, `draw.derive` and `service.select_draw` on fixed coins, with the bytes it read), for the
`slots` piece that sample in the claims' slots, read off `partition.strata`, `partition.unit_gates` and the root's input
nodes (`slots`), for the `hidden_offsets`, `hidden_locate` and `hidden_witness` pieces `hidden_sample` on honest and
arbitrary calls tables, for `check_public` and `check_verdicts` the proof system's own (`service.check_public`,
`service.check_verdicts`), for the `fits` piece v2's refusal of a length a session's statement can't hold (`proofs.fits_why`, `proofs.draw_fits_why`) at small limits, and for the `run` piece
(a docket's whole run, `Run.run`) those references in a `Stream`'s order.
The `partition`, `select` and `slots` pieces also run on random partitions: mixed population
programs of random members of three templates, and rarely of `FORK` and of `IDENTITY`, of no gate, which one-stage
refuses, under template queries over them, and programs of random root Calls (`call_program`) under `Q_word` and `Q_call`, a 32-bit
Call among them, whose unit `Q_word` takes in the 32-bit class and `Q_call` at `X = 16` refuses, and `SHARED`, whose
`Q_call` unit at `X = 16` holds a constant another unit reads. Records are v1 and, on the programs with root parameters to register
(`REGISTERED_SETUPS`, 16- and 32-bit), mostly v2: the verifier expects values registered (`registered`, with its program
function's `parameters`), and the record (and an earlier one in the log) registers `Registered.commit`'s entries; on
the programs with a pair of digests to fill (`SUPPLIED_SETUPS`, 32- and 1-bit leaves, a digest a row in a bit-row
format, or one 16-bit leaf a row) the verifier fills it itself from
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
import importlib.util
import json
import random
import re
import shutil
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from hypothesis import note, strategies as st

import verity_difftest as D
from verity_difftest import protocol as PR
from verity.commitments import docket_digest as DD
from verity.circuits import boolean as BL, population as PN
from verity.commitments import domains as DM
from verity.protocols.accounting.compute.pouw.verifier import audit as PA
from verity.protocols.accounting.compute.pouw.verifier.protocol import tagged
from verity.protocols.proof.spot.verifier.profile import Bernoulli, Stratified, Stratum
from verity.protocols.proof.spot.verifier.challenger import challenger as CH
from verity.protocols.proof.spot.verifier import service
from verity.protocols.proof.spot.verifier.one_stage import (
    audit,
    caller as C,
    draw as DR,
    hello as H,
    hidden_sample as HD,
    link_points as LP,
    members as M,
    one_circuit as OC,
    partition as P,
    produced as PD,
    proofs as PF,
    registered as G,
    docket as R,
)
from verity.protocols.proof.spot.prover import docket as DK
from verity.protocols.proof.spot.prover import produced as PP
from verity.commitments import registered_value as RV
from verity.protocols.proof.spot.verifier.challenger.challenger import COIN_BYTES, NO_CLOSE, NO_TREE, ChallengerKey, CoinError
from verity.protocols.proof.spot.verifier.challenger.coins import (FAMILY_ROOT, LIMITS, MAX_PATH, OwnRandomness, coins_reason,
                                              coins_within, draw_coins, draw_first, sampling_design_problem)
from verity.protocols.proof.spot.verifier.challenger.coins import ran_out as draw_ran_out
from verity.protocols.proof.spot.verifier.one_stage.proofs import order_why
from verity.protocols.proof.spot.verifier.one_stage.verdict import Verdict
from verity.circuits import codec, partition_object as PO, partition_query as PQ
from verity.circuits import edges as EG, replay_links as RL
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program, input_id, is_input
from verity.circuits.refs import array_of, record_of, tuple_of
from verity.circuits.types import Array, RecordType, Tuple, Value



def _by_path(name: str, path: Path):
    """The module at `path`, loaded once as `name`: this file is imported by path from other suites, where a bare
    `import settings` finds their own registry and `stand_in` is on no path."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


settings = _by_path("spot_verifier_lean_settings", Path(__file__).resolve().with_name("settings.py"))
stand_in = _by_path("stand_in", Path(__file__).resolve().parents[5] / "circuits" / "stand_in.py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_challenger_difftest import DEPLOYED, LANES, SAMPLE_MUTATIONS, ORDERS, Ch, Drawn, _flip, _mutated, lms_verify  # noqa: E402

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proof" / "spot" / "verifier" / "lean")
CALLER_PACKAGE = PACKAGE.parents[2] / "spot_flock/verifier/lean"
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
CONST8 = stand_in.exact(PrimitiveDefinition("CoreSpotConst8", 1, (), Value(8), lambda: 0x5A, register=False),
                         lambda B: [B.call(BL.constant((0x5A >> j) & 1)) for j in range(8)])


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
    """A verifier's program and query, with its descriptor and the ports its layout derives, each `rows` rows a unit, and
    its partition object, with `blocks` (`partition_object.build`) where given."""

    def __init__(self, program, query, names=("out", "x"), rows=(2, 1), blocks=None):
        self.program, self.query = program, query
        self.desc = _json(codec.encode_program(program))
        self.sha512 = PO.program_sha512(program)
        self.population = P.units(program, query)
        self.templates = [t for t, _ in P.strata(program, query)]
        part = self.partition = P.partition(program, query) if blocks is None else PO.build(program, query, blocks)
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


SETUPS = [Setup(PN.population_program(PAIR, n), P.template_instance_query(PAIR.id)) for n in (1, 3, 6)] + [
    Setup(PN.mixed_population_program([(PAIR, 2), (TRIPLE, 3)]), P.template_instances_query([PAIR.id, TRIPLE.id])),
    Setup(PN.mixed_population_program([(PAIR, 1), (TRIPLE, 2), (PAIR, 2)]), P.template_instances_query([TRIPLE.id, PAIR.id])),
    Setup(call_program((0, 2, 0, 3)), PO.query()), Setup(call_program((1, 2, 3)), PO.call_query(8))]
XOR16 = PrimitiveDefinition("CoreSpotXor16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: a ^ b,
                            register=False)
#: `PAIR` and `TRIPLE` over u16 words, the one form the door lowers (`OneStageZk.admitted`), each `x` a 128-byte row,
#: past a compression before its last block, which a Flock block's pin needs.
PAIR16 = bind(CompositeDefinition("CoreSpotPair16", 0, (), lambda S: ((("x", Array(64, Value(16))),), Value(16)),
                                  lambda B, S, x: B.call(XOR16, x[0], x[1]), register=False))
TRIPLE16 = bind(CompositeDefinition("CoreSpotTriple16", 0, (), lambda S: ((("x", Array(64, Value(16))),), Value(16)),
                                    lambda B, S, x: B.call(XOR16, B.call(XOR16, x[0], x[1]), x[2]), register=False))
_MIXED16 = Setup(PN.mixed_population_program([(PAIR16, 2), (TRIPLE16, 3)]),
                 P.template_instances_query([PAIR16.id, TRIPLE16.id]))
#: `SETUPS[3]`'s population over u16 words, with a partition object naming its proof blocks, one unit each: the door
#: derives the drawn block's circuit from them (`one_circuit.block`), and refuses an object naming none.
BLOCKED = Setup(_MIXED16.program, _MIXED16.query, blocks=[[(u, u + 1) for u in range(_MIXED16.population)]])


def _registered_program(name: str, n: int, extra, registered) -> Program:
    """`n` instances of `PAIR` over `x`, beside root parameters `extra` the body does not read, those `registered` names
    bound at docket."""
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
            self.committed[v, w] = RV.Registered.commit(self.sha512, v, *self._rows(v, w))

    def _rows(self, v: str, w: int) -> tuple:
        """`Registered.commit`'s rows, salts, width and row format for `v` at `w` words."""
        schema = self.formats.get(v, RV.SCHEMA)
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
                self._windowed[v, w, window] = RV.Registered.commit(self.sha512, v, *self._rows(v, w), window=window)
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
#: `row-seg/v1` rows); `x`, bound per run, is a run Input, which no verifier expects registered (`R7-run-input`).
REGISTERED_SETUPS = [
    RegisteredSetup(_registered_program("CoreSpotRegistered16", 3, (
        ("W", Array(6, Value(16))), ("D", Array(4, Value(12))), ("E", Array(0, Value(16))), ("M", Tuple(Value(8), Value(16))),
        ("V", Value(40))), ("W", "D")), [{"W": 2}, {"D": 1, "W": 3}, {"D": 4}]),
    RegisteredSetup(_registered_program("CoreSpotRegistered32", 2, (
        ("T", Array(4, Value(32))), ("U", RecordType(("a", Array(3, Value(20))),))), ("T", "U")),
        [{"T": 2}, {"T": 1, "U": 3}, {"U": 1}]),
    RegisteredSetup(_registered_program("CoreSpotRegisteredBits", 2, (
        ("A", Array(8, Value(8))), ("B", Array(4, Value(16)))), ("A", "B")),
        [{"A": 2}, {"A": 1, "B": 2}, {"B": 4}], {"A": G.SCHEMA_V2, "B": G.SCHEMA_SEG})]
#: Programs with a value the verifier fills itself, as PoUW's `pouw.roots`: `S`, each row a pair of digests of 32-bit
#: leaves (rows of x's root and W's registered root, and W's and out's) or of 1-bit leaves (out's root and bytes of its own),
#: or, in a bit-row format, one `Digest` a row in `row/v2` (out's root, then bytes of its own: 32 u16 words a row) or
#: one digest of 1-bit leaves a row in `row-seg/v1` (x's root, then bytes of its own); or `O`, a read's verdict value,
#: each row the verifier's two bytes of 1 as one 16-bit leaf (`half_leaves`).
SUPPLIED_SETUPS = [
    SuppliedSetup(_registered_program("CoreSpotSupplied32", 3, (
        ("S", Array(2, Array(2, Array(16, Value(32))))), ("W", Array(6, Value(16)))), ("S", "W")),
        [{"S": 32, "W": 2}, {"W": 3, "S": 32}, {"W": 2}],
        {"S": [[{"port": "x"}, {"value": "W"}], [{"value": "W"}, {"port": "out"}]]}),
    SuppliedSetup(_registered_program("CoreSpotSupplied1", 2, (
        ("S", Array(1, Array(2, Array(512, Value(1))))),), ("S",)),
        [{"S": 1024}, {"S": 1024}], {"S": [[{"port": "out"}, {"bytes": "5a" * 64}]]}),
    SuppliedSetup(_registered_program("CoreSpotSuppliedPW", 3, (
        ("S", Array(2, Array(2, Array(16, Value(32))))), ("W", Array(6, Value(16)))), ("S", "W")),
        [{"S": 32, "W": 2}, {"W": 3, "S": 32}],
        {"S": [[{"window": COIN}] + [{"position": "0"}] * 8, [{"position": "2"}] * 8 + [{"window": COIN}]]}),
    SuppliedSetup(_registered_program("CoreSpotSuppliedDigestV2", 2, (
        ("S", Array(2, Array(8, Array(64, Value(1))))),), ("S",)),
        [{"S": 32}, {"S": 32}], {"S": [[{"port": "out"}], [{"bytes": "5a" * 64}]]}, {"S": G.SCHEMA_V2}),
    SuppliedSetup(_registered_program("CoreSpotSuppliedBitsSeg", 2, (
        ("S", Array(2, Array(512, Value(1)))),), ("S",)),
        [{"S": 32}, {"S": 32}], {"S": [[{"port": "x"}], [{"bytes": "a5" * 64}]]}, {"S": G.SCHEMA_SEG}),
    SuppliedSetup(_registered_program("CoreSpotSuppliedHalf", 2, (
        ("O", Array(2, Value(16))),), ("O",)),
        [{"O": 1}, {"O": 1}], {"O": [[{"bytes": "0001"}], [{"bytes": "0001"}]]})]
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
#: A two-stage design's two dockets: the boundary one over four replay blocks under the grain, its ports its own,
#: and the interior one over a replay sample of two of them, six proof blocks under the check's query.
#: The boundary's ports: `BLOCK`'s own, under its descriptor id, which the links round 2 derives from it read, each its
#: three PAIR instances' rows a replay block, of PAIR's `x` and output each.
BX, BOUT = (f"{codec.descriptor_id(BLOCK)}/{p}" for p in ("x", "o0"))
BOUNDARY = Setup(PN.population_program(BLOCK, 4), P.template_instance_query(BLOCK.id), names=(BOUT, BX), rows=(3, 3))
INTERIOR = Setup(PN.population_program(PAIR, 6), P.template_instance_query(PAIR.id))
PROGRAMS = {s.sha512: s.program for s in SETUPS + REGISTERED_SETUPS + [POOLED, CONTINUED, BOUNDARY, INTERIOR, BLOCKED]}
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
    program = PN.mixed_population_program([(MEMBERS[t], n) for t, n in members])
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
        raise D.Reject(f"the keys of {what} are not {', '.join(sorted(keys))}")


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


def rule(sampling_design) -> dict:
    """`OneStage.rule`: the agreed sampling design, `{"law": "bernoulli", "p"}` or `{"law": "subset" | "stratified" | "work", "k"}`."""
    kind = sampling_design.get("law") if isinstance(sampling_design, dict) else None
    if kind == "bernoulli":
        _exactly(sampling_design, {"law", "p"}, "the law")
        if not probability(sampling_design["p"]):
            raise D.Reject("the law's p is not NUM/DEN")
    elif kind in ("subset", "stratified", "work"):
        _exactly(sampling_design, {"k", "law"}, "the law")
        if not _nat(sampling_design["k"]):
            raise D.Reject("the law's k is not a natural number")
    else:
        raise D.Reject("the law is not subset, bernoulli, stratified or work")
    return sampling_design


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
    weighed = isinstance(j, dict) and "registration" in j
    given = isinstance(j, dict) and "parameters" in j
    keys = PARAM_KEYS | {"parameters"} if v2 or tied or weighed or given else PARAM_KEYS
    keys = keys | {"tied"} if tied else keys
    keys = keys | {"registration"} if weighed else keys
    keys = keys | {"supplied"} if isinstance(j, dict) and "supplied" in j else keys
    keys = keys | {"pinned"} if isinstance(j, dict) and "pinned" in j else keys
    keys = keys | {"outputs"} if isinstance(j, dict) and "outputs" in j else keys
    keys = keys | {k for k in ("row_schemas", "window_root", "held", "challenger", "seed_salt", "pool", "compiler",
                               "certified", "verdicts", "window_values", "coins", "commitments", "released")
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
        sampling_design, table = service.params_sampling_design(j["law"], PROGRAMS[sha], j["query"], j["work_table"])
    except ValueError as e:
        raise D.Reject(str(e)) from None
    rule(sampling_design)
    j = {**j, "law": sampling_design, "work_table": table}
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
    if part is not None and R.partition_query_refused(part, j["query"]):
        raise D.Reject(R.PARTITION_QUERY)
    if v2 and not all(map(_nat, _obj(j["registered"], "registered").values())):
        raise D.Reject("registered: words that are not a natural number")
    needed = v2 or tied or weighed or bool(R.marked_inputs(PROGRAMS[sha]))
    if needed and not given:
        raise D.Reject("parameters: the program binds inputs at registration, or the verifier expects or pins values, "
                       "and no parameters name them")
    if needed or given:
        _parameters(desc, PROGRAMS[sha], j["parameters"])
    if tied and not (j["tied"] is None or isinstance(j["tied"], list) and all(isinstance(t, str) for t in j["tied"])):
        raise D.Reject("tied values are a list of the program's parameters")
    raw = j
    if "verdicts" in j:
        try:
            R._verdicts(j["verdicts"])
        except ValueError as e:
            raise D.Reject(str(e)) from None
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
    try:
        G.marked_rows(PROGRAMS[sha], j["registered"] or {}, G.sources(j.get("supplied", {})), j.get("coins"),
                      j.get("commitments"), j.get("released"), len(j["log"]),
                      [bytes.fromhex(k) for k in j.get("attester_keys", [])])
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
    if "certified" in j:
        if not isinstance(j["certified"], dict):
            raise D.Reject("certified entries are not an object")
        try:
            R._certified(j["certified"], j["registered"] or {}, G.sources(j.get("supplied", {})))
        except ValueError as e:
            raise D.Reject(str(e)) from None
    try:
        window, fixed = R._window(j.get("window_root"), j.get("held"), j["registered"] or {},
                                  G.sources(j.get("supplied", {})), j.get("pinned") or {}, j.get("outputs") or {})
    except ValueError as e:
        raise D.Reject(str(e)) from None
    try:
        R._seed_salt(j.get("seed_salt"), window)
        R._registered_compiler(j.get("compiler", R.UNREAD))
    except ValueError as e:
        raise D.Reject(str(e)) from None
    if "challenger" in j:
        _challenger(j["challenger"], window)
    if weighed:
        try:
            reg = R.registration_parsed(j["registration"], j.get("challenger"), window, lms_verify)
        except R.RegistrationRefused as e:
            raise D.Reject(str(e)) from None
        raw = {**raw, "weights_of_record": reg[:2]}
    if "window_values" in j:
        _window_values(j["window_values"], j["ports"], j["registered"] or {})
    delta = ratio(j["delta"])
    if not 0 < delta < 1:
        raise D.Reject("delta is not in (0, 1)")
    if R.unwindowed(PROGRAMS[sha], j["registered"] or {}, j.get("row_schemas"), window, fixed, j["log"]):
        raise D.Reject("a value it doesn't hold across windows that the log registered, and no window root")
    return {**raw, "program": PROGRAMS[sha], "delta": delta}


def _window_values(named, ports: dict, registered: dict) -> None:
    """`OneStage.paramsParsed`'s `window_values`: a list of strings, none a port or a value the verifier expects
    registered."""
    if not isinstance(named, list):
        raise D.Reject("window values are a list of strings")
    for v in named:
        if not isinstance(v, str):
            raise D.Reject("window values are a list of strings")
        if v in ports or v in registered:
            raise D.Reject(f"window value {v}: a port or a registered value of the verifier's, not a window's")


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
    return rec, bytes.fromhex(DD.digest(rec))


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


def _docket_verdict(p: dict, rec) -> Verdict:
    return R.check(rec, p["program"], p["query"], p["law"], p["scheme"], p["ports"], p["leaf_layer"], p["anchors"], p["log"],
                   p["partition"], p["work_table"], p["registered"], p.get("supplied"), p.get("pinned"),
                   p.get("binding"), p.get("outputs"), p.get("row_schemas"), p.get("window_root"), p.get("held"),
                   p.get("link_members", ()), seed_salt=p.get("seed_salt"), pool=p.get("pool"), tied=p.get("tied"),
                   compiler=p.get("compiler", R.UNREAD), certified=p.get("certified"), verdicts=p.get("verdicts"),
                   coins=p.get("coins"), commitments=p.get("commitments"), released=p.get("released"),
                   weights_of_record=p.get("weights_of_record"))


def _registration(p: dict, rec) -> tuple[str, ...]:
    return _docket_verdict(p, rec).codes


def _caller_verdict(p: dict, rec) -> Verdict:
    return C.check(rec, p["program"], p["query"], p["law"], p["scheme"], p["ports"], p["leaf_layer"], p["anchors"], p["log"],
                   p["partition"], p["work_table"], p["registered"], p.get("supplied"), p.get("pinned"),
                   p.get("binding"), p.get("outputs"), p.get("row_schemas"), p.get("window_root"), p.get("held"),
                   day=p.get("day"), declarations=p.get("declarations"), seed_salt=p.get("seed_salt"),
                   pool=p.get("pool"), compiler=p.get("compiler", R.UNREAD), certified=p.get("certified"),
                   verdicts=p.get("verdicts"), coins=p.get("coins"), commitments=p.get("commitments"),
                   released=p.get("released"))


def _coins_of(p: dict, c: tuple[dict, bytes], e: dict) -> bytes | None:
    """`OneStage.coinsOf`: with the verifier's challenger, the coins `coins.draw_coins` reads from the evidence's `coins`
    under its key, on the verifier's window, of its family, after the docket's digest; none without one, when the
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
    exactly when `audit_record` accepts, with the sampled and proved units, the bounds, whether every verdict ran (the
    record's ``complete``) and the profile."""
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
                             audit.held_reads(e["verdicts"], rec, p.get("window_values") or ()),
                             sorted(p.get("certified") or ()))
    if not out["accepted"]:
        if not all(x["ok"] for x in out["checks"].values()):
            return "reject", "checks"
        ran = [v for v in e["verdicts"].values() if v.get("accepted") is not None]
        return ("undecided", "verifier of record") if all(v["accepted"] is True for v in ran) else ("reject", "verdict")
    profile = audit.profile(rec, e["draw"], p["delta"]).to_json()
    wrong = out["wrong_units"]
    public = {"bound": wrong["bound"], audit.COMPLETE: out["complete"], "drawn": list(e["draw"]["units"]),
              "per_template": wrong.get("per_template"), "proved": DR.proved(e["draw"])}
    return "accept", public, {**profile, "delta": str(profile["delta"])}


def _ran_out(p: dict, rec: dict, src: bytes) -> bool:
    """`OneStage.ranOut`: whether the coins `src` run out before the sample under the verifier's sampling design as derived ends
    (`coins.ran_out`); false when the sampling design doesn't derive, the record names no population or the sample refuses otherwise."""
    if not _nat(rec.get("population")):
        return False
    try:
        return draw_ran_out(DR.derive(p["program"], p["query"], p["law"], p["work_table"]), rec["population"], src)
    except (P.QueryRefused, ValueError, TypeError):
        return False


def covered_params(j) -> dict:
    """`OneStage.paramsOfJson`: `decode_params` (`paramsParsed`), and every bernoulli sampling design refused, at any p
    (`service.bernoulli`), as are parameters that name certified entries."""
    if isinstance(j, dict) and "certified" in j:
        raise D.Reject("the parameters name certified entries: the sampled audit takes none")
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
    return PR.answer(DM.ONE_STAGE_PROTOCOL, _json(case), params=covered_params, commitment=decode_commitment,
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
    p = _params_of(j)
    if "members" in j:
        p = {**p, "link_members": _link_members(j["members"])}
    v = _docket_verdict(p, j.get("record"))
    return {"codes": list(v.codes), **({} if v.fanout is None else {"fanout": v.fanout}),
            **({"rests_on_period_check": list(v.rests_on_period_check)} if v.rests_on_period_check else {})}


def caller(j):
    """`spot-flock-caller`'s `caller` (`VStar.Recursive.Caller.check`): the caller's codes for a record against the
    parameters (`caller.check`), whatever SPOT's check makes of the record."""
    j = _json(j)
    return {"codes": list(_caller_verdict(_params_of(j), j.get("record")).codes)}


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
        return RV.domain(j["program"], j["value"], j["words"], j["leaves"], j["word_bits"], j.get("window"),
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
    """`draw_first` on the source's bytes, read in order; it raises `ValueError` (and `TypeError` on an unhashable sampling design
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


def ran_out(j):
    """`coins.ran_out` on `draw`'s input: whether that sample runs out of the source's bytes; refused where `draw` is."""
    j = _json(j)
    if not isinstance(j, dict) or not isinstance(j.get("law"), dict):
        raise D.Reject("law is not an object")
    if not _nat(j.get("population")):
        raise D.Reject("population is not a natural number")
    if not isinstance(j.get("source"), str):
        raise D.Reject("source is not hex")
    src = PR.unhex(j["source"])
    try:
        return {"ran_out": draw_ran_out(j["law"], j["population"], src)}
    except (ValueError, TypeError) as e:
        raise D.Reject(str(e)) from None


def _rule(j) -> service.Selection:
    """`Select.Rule.ofJson`: a selection rule as `to_json` writes it, exactly its keys, of the kinds a `Stream` runs."""
    kind = j.get("rule") if isinstance(j, dict) else None
    try:
        if kind == "all" and set(j) == {"rule"}:
            return service.All()
        if kind == "law" and set(j) == {"law", "rule"} and isinstance(j["law"], str):
            return service.SamplingDesign(j["law"])
        if (kind == "two-stage" and set(j) == {"grain", "k", "replay", "rule"} and isinstance(j["replay"], str)
                and (_nat(j["k"]) or isinstance(j["k"], dict)) and service._grain(j["grain"])):
            return service.TwoStage(j["grain"], j["replay"], j["k"])
    except ValueError as e:
        raise D.Reject(str(e)) from None
    raise D.Reject("not a selection rule the stream runs")


def select(j):
    """The challenger's sample as `Stream.register` and `Stream.select` make it: `service.selection_sampling_design` over the
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
        sampling_design = DR.derive(program, query, service.selection_sampling_design(rule, n, service.selection_sizes(rule, program, query)), table)
        obj = service.select_draw(rule, sampling_design, n, coins, closure=closure)
    except (P.QueryRefused, ValueError, service.Refused) as e:
        raise D.Reject(str(e)) from None
    if not DR.check_draw(obj, {"population": n}, sampling_design).ok:
        raise D.Reject("the draw is not well formed under the law")
    return {"draw": obj, "read": at}


def design(j):
    """`Pieces.design`: the sampling design of `select`'s sample (`audit.design`, Lean `Spot.Pieces.designOfSamplingDesign`), Flock's sampling design
    held to core's `Core.SamplingDesign`, the work sampling design's among them."""
    return {"design": _json(audit.design(select(j)["draw"]).to_json())}


def template_circuits(j):
    """A two-stage run's params naming each template's circuit file and its statements' reference header and registered
    reads (`service.template_circuits`), in the order the check's template query names the templates, and the templates
    they record as table reads, each with its table's rows (`service.template_reads`), and the registered tables they
    pin (`service.pinned_tables`); a missing `proofs` names none."""
    j = _json(j)
    proofs, query = (j.get("proofs"), j.get("query")) if isinstance(j, dict) else (None, None)
    try:
        named = service.template_circuits(proofs, query)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return {"circuits": [[t, c, h, r] for t, c, h, r in named],
            "reads": [[t, v] for t, v in service.template_reads(proofs)],
            "tables": [[p, e] for p, e in service.pinned_tables(proofs)]}


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


def pop_order(j):
    """`Slots.popOk` as it runs (`Slots.popOkRanges`: the strata's segments chain `0, ..., n - 1` and every unit has its
    gates) and as it is stated (the layout's population is units `0, ..., n - 1` in slot order): both `partition.in_order`."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    desc, query = _obj(j.get("program"), "program"), _obj(j.get("query"), "query")
    n = j.get("population")
    if not (_int(n) and n >= 0):
        raise D.Reject("population is not a natural number")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    ok = P.in_order(PROGRAMS[sha], query, n)
    return {"layout": ok, "ranges": ok}


def public_units(j):
    """Whether a public file's ``indices`` are exactly the ``[lo, hi)`` ranges ``units``' units, ascending
    (`proofs.units_exactly`, `PublicUnits.exactly`)."""
    j = _json(j)
    units, indices = (j.get("units"), j.get("indices")) if isinstance(j, dict) else (None, None)
    if not (isinstance(units, list)
            and all(isinstance(r, list) and len(r) == 2 and all(map(_nat, r)) and r[0] < r[1] for r in units)):
        raise D.Reject("units is not a list of [lo, hi] with 0 <= lo < hi")
    if not (isinstance(indices, list) and all(map(_nat, indices))):
        raise D.Reject("indices is not a list of natural numbers")
    return {"exactly": PF.units_exactly(units, indices)}


def _link_members(ms) -> list:
    """The members a piece is handed (`LinkPoints.membersOfJson`), as `link_points`' functions and `registration.check`
    read them: each its units' ranges and its circuit's units a proof block."""
    if not isinstance(ms, list):
        raise D.Reject("the members are not a list")
    members = []
    for m in ms:
        if not (isinstance(m, dict) and _nat(m.get("vus_per_block"))):
            raise D.Reject("a member's vus_per_block is not a natural number")
        us = m.get("units")
        if not (isinstance(us, list) and us and all(isinstance(r, list) and len(r) == 2 and all(map(_int, r))
                                                    and 0 <= r[0] < r[1] for r in us)):
            raise D.Reject("a member's units are not a nonempty list of [lo, hi] with 0 <= lo < hi")
        members.append((us, m["vus_per_block"]))
    return members


def link_points(j):
    """Each member's fewest and most sampled proof blocks under the sampling design over the population when the sample ends (None where none
    does), and the members whose link points differ (`link_points.bounds`, `refused`; Lean `LinkPoints.piece`)."""
    j = _json(j)
    if not (isinstance(j, dict) and _nat(j.get("population"))):
        raise D.Reject("the population is not a natural number")
    sampling_design, n = j.get("law"), j["population"]
    if not isinstance(sampling_design, dict) or (why := sampling_design_problem(sampling_design, n)):
        raise D.Reject(f"the law: {why}" if isinstance(sampling_design, dict) else "the law is not an object")
    members = _link_members(j.get("members"))
    try:
        return {"bounds": [None if b is None else list(b) for b in LP.bounds(sampling_design, n, members)],
                "refused": LP.refused(sampling_design, n, members)}
    except ValueError as e:
        raise D.Reject(str(e)) from None


_LOWER_HEX = re.compile(r"(?:[0-9a-f]{2})*")


def _unhex(x) -> bytes:
    if not (isinstance(x, str) and _LOWER_HEX.fullmatch(x)):
        raise D.Reject("not lowercase hex of whole bytes")
    return bytes.fromhex(x)


def fits(j):
    """`OneStageZk.fitsWhy` and `sampleFitsWhy` at ``limit`` (`proofs.fits_why`, `proofs.draw_fits_why`): the files hex,
    the tables ``[name, hex]`` pairs, the program hex or null, and the partition object, registered reads and sample null
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
    """`FilePart.why` (`proofs.population_why`): the docket's strata over the program and query
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


NO_BLOCKS = "restrict: the partition object has no blocks"


@functools.cache
def _blocked_lookups() -> list[str]:
    """The lookups of the circuit file `Own.circuitFile` derives for `BLOCKED`'s block at unit 0, as the door lowers it."""
    c, part, units = b"".join(codec.descriptor_chunks(BLOCKED.program)), BLOCKED.partition, [(0, 1)]
    text = OC.lower(OC.restrict(c, part, units), part, PF.lower_executable, cut=OC.cuts(c, part),
                    formed=OC.formed_in(c, part, units))
    return PF.lookups_of(json.loads(text.split(b"\n")[1].decode()[len("META "):]))


def _tables_case(j):
    """`OneStageZk.tablesCase`: the blobs' `digests` (even-length lowercase hex) and the lookups of the circuit file the
    verifier derives from the case's C and partition (`BLOCKED`'s, `_blocked_lookups`), or the ValueError it refuses
    with where the case names no partition; the case's own ``lookups`` reach no circuit the verifier reads."""
    j = _json(j)
    ds = j.get("digests") if isinstance(j, dict) else None
    if not isinstance(ds, list):
        raise D.Reject("digests is not a list")
    if not all(isinstance(d, str) for d in ds):
        raise D.Reject("digests is not a list of hex strings")
    if not all(len(d) % 2 == 0 and HEX_DIGITS.fullmatch(d) for d in ds):
        raise D.Reject("not lowercase hex")
    if j.get("partition") is None:
        return ds, ValueError(NO_BLOCKS)
    if (j["partition"], j.get("program")) != (_json(BLOCKED.partition), BLOCKED.desc):
        raise RuntimeError("the strategy drew a C and partition other than BLOCKED's")
    return ds, _blocked_lookups()


def tables_held(j):
    """`OneStageZk.tablesPiece` (`proofs.tables_why`): v2's refusal of blobs under `digests` whose pinned MUFU tables
    are not the circuit's lookups (`_tables_case`)."""
    ds, lookups = _tables_case(j)
    if isinstance(lookups, ValueError):
        return {"why": str(lookups)}
    return {"why": PF.tables_why(lookups, ds)}


def own_tables(j):
    """`OneStageZk.ownTablesPiece` (`proofs.own_tables`, `proofs.tables_match_why`): a member's own lookup tables among
    the blobs under `digests`, by name, and their refusal against the circuit's lookups (`_tables_case`)."""
    ds, lookups = _tables_case(j)
    if isinstance(lookups, ValueError):
        return {"tables": None, "why": str(lookups)}
    tables = PF.own_tables(lookups, ds)
    return {"tables": tables, "why": PF.tables_match_why(tables, lookups)}


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
        sampling_design = Stratified(tuple(Stratum(s["name"], s["size"], s["k"]) for s in ss))
        return {"bound": sampling_design.count_bound(delta), "per_stratum": {s.name: sampling_design.stratum_bound(s.name, delta) for s in sampling_design.strata}}
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
    sampling_design = {"k": 2, "law": "subset"}
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports], WINDOW,
                   leaf_layer=LEAF)
    d = {"k": 2, "law": "subset", "population": s.population, "units": [1, 4]}
    return rec, R.receipt(rec, "verifier-test", received=RECEIVED), d, sampling_design


FIXED = _fixed_audit()


def verdict(j):
    j = _json(j)
    vs = j.get("verdicts") if isinstance(j, dict) else None
    if not isinstance(vs, dict) or not all(isinstance(v, dict) for v in vs.values()):
        raise D.Reject("verdicts is not an object of objects")
    rec, receipt, d, sampling_design = FIXED
    out = audit.audit_record(rec, receipt, d, sampling_design, d["units"], 2, {}, vs, Fraction(1, 1000))
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
        sampling_design = rec.get("law")
        strata = sampling_design.get("strata") if isinstance(sampling_design, dict) else None
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
    """`RecordType.inputOf`'s domain, where `audit_record` reads its inputs as the Lean does, `certified` and
    `window_values` lists of strings when given; δ."""
    ints = lambda x: isinstance(x, list) and all(map(_int, x))  # noqa: E731
    _exactly(j, RECORD_INPUT_KEYS | ({"certified", "window_values"} & set(j) if isinstance(j, dict) else set()),
             "the record's input")
    if not (isinstance(j.get("certified", []), list) and all(isinstance(v, str) for v in j.get("certified", []))):
        raise D.Reject("certified is not a list of strings")
    if not (isinstance(j.get("window_values", []), list) and all(isinstance(v, str) for v in j.get("window_values", []))):
        raise D.Reject("window_values is not a list of strings")
    reg, sample_ = j["registration"], j["draw"]
    if not (isinstance(reg, dict) and all(k in reg for k in REGISTRATION_KEYS) and _int(reg["population"])
            and isinstance(reg["anchors"], list) and isinstance(reg["roots"], list)
            and all(isinstance(r, dict) and isinstance(r.get("port"), str) and "root" in r and "leaves" in r for r in reg["roots"])
            and ("registered" not in reg or isinstance(reg["registered"], list) and all(
                isinstance(e, dict) and isinstance(e.get("value"), str) and all(k in e for k in ("root", "leaves", "domain"))
                for e in reg["registered"]))):
        raise D.Reject("the registration")
    if not (isinstance(j["receipt"], dict) and isinstance(sample_, dict) and ints(sample_.get("units"))
            and ("closure" not in sample_ or ints(sample_["closure"])) and ("population" not in sample_ or _int(sample_["population"]))
            and (sample_.get("law") != "subset" or "k" in sample_) and isinstance(j["law"], dict)):
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
    roots the verdicts' own for the docket's values less those of its own ports at their own roots, with those of the
    window values it names less the reads at their window entries' roots (`Record.heldReads`), and with `certified`
    each of those values held to have a reported read."""
    delta = _record_input(j)
    opened = audit.held_reads(j["verdicts"], j["registration"], j.get("window_values") or ())
    return audit.audit_record(j["registration"], j["receipt"], j["draw"], j["law"], j["proved_units"], j["instances"],
                              j["statement"], j["verdicts"], float(delta), j["sessions"], j["served"], j["session_roots"],
                              opened, j.get("certified", ()))


def record(j):
    return codec.canonical_json(audit_record_of(_json(j))).decode()


def _named(params: dict) -> dict:
    """`Run.audit`'s `window_values`: the params' window values, given only when there are some."""
    return {"window_values": list(params["window_values"])} if params.get("window_values") else {}


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
    return {"format": R.RECEIPT, "registration": DD.digest(rec), "verifier": challenger, "received": received}


def run(j):
    """`Run.run`: a `Stream`'s run of one docket from its public check through its verdict, the pieces above in its
    order. `register`: `check_public` on the values and declared items, the population, R1-R7 under
    `service.selection_sampling_design` over it (a rate sampling design's count of none or more than N: `rate`), a bernoulli sampling design at
    any p (`service.bernoulli`: `bernoulli`), a program whose units read stratum by stratum are not its N in order
    nor a disjoint cover of them (`partition.covers`: `population-order`), or a cover under a design not subset at both
    the rule and the docket (`members.subset_designs`: `strata-design`) or under a two-stage rule (`population-order`), and
    the receipt; `select`: a closure map refused (`closure`), then the
    sample from the source's bytes (`select`), the challenger's coins with their evidence when it read any, held to
    the run's `family`, its challenger key (`key`) and its copy of main's history (`main`) (`_coins_code`: code `coins`,
    or `family`; a `coin_source` other than the challenger is refused
    up front: the verifier samples no coins of its own); `deliver`: one or more sessions, each matching
    `service._DIGEST`; `outcome`: `check_verdicts`, then the audit record (`record`) over the sample's proved units unless
    given, its acceptance the symbol and the first verdict of record's public outputs shown on an accept. A step that refuses gives `{"step",
    "codes"}`; inputs the Lean decodes before the run (`Pieces.run`) are refused up front. Under a two-stage rule the
    docket's query is its grain (`grain`), and a run with no interior docket stops after `select` (`interior`); this
    file draws no two-stage run with one."""
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
    for of, what in ((j, "the run's"), (j.get("interior"), "an interior's")):
        if isinstance(of, dict) and not (of.get("family") is None or isinstance(of["family"], str)):
            raise D.Reject(f"{what} family is not a string")
    if j.get("key") is not None:
        try:
            ChallengerKey.from_json(j["key"])
        except CoinError as e:
            raise D.Reject(str(e)) from None
    if not isinstance(j.get("source"), str):
        raise D.Reject("source is not hex")
    PR.unhex(j["source"])
    if j.get("coin_source", "challenger") not in (None, "challenger"):
        raise D.Reject("the verifier draws no coins of its own")
    it = j.get("interior")
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
        sampling_design = service.selection_sampling_design(rule, n, service.selection_sizes(rule, p["program"], p["query"]))
        table = service.selection_table(rule, p["program"], p["query"], p["work_table"])
    except service.Refused as e:
        return {"codes": list(e.verdict.codes), "step": "register"}
    except ValueError as e:
        raise D.Reject(str(e)) from None
    codes = _registration({**p, "law": sampling_design, "work_table": table}, rec)
    if codes:
        return {"codes": list(codes), "step": "register"}
    if service.bernoulli(sampling_design):
        return {"codes": ["bernoulli"], "step": "register"}
    if not P.in_order(p["program"], p["query"], n):
        if not P.covers(p["program"], p["query"], n):
            return {"codes": ["population-order"], "step": "register"}
        if not M.subset_designs(sampling_design, rec):
            return {"codes": ["strata-design"], "step": "register"}
        if isinstance(rule, service.TwoStage):
            return {"codes": ["population-order"], "step": "register"}
    receipt = _receipt(rec, j["challenger"], j["received"])
    try:
        derived = DR.derive(p["program"], p["query"], sampling_design, table)
    except (P.QueryRefused, ValueError) as e:
        raise D.Reject(str(e)) from None
    if closure is not None:
        return {"codes": ["closure"], "step": "select"}
    drawn = select({"closure": closure, "program": j["params"]["program"], "query": j["params"]["query"], "rule": j["rule"],
                    "source": j["source"], "work_table": p["work_table"]})
    code = _coins_code(j, rec, PR.unhex(j["source"])[:drawn["read"]], j.get("key"), j.get("main"))
    if code is not None:
        return {"codes": [code], "step": "select"}
    codes = DR.check_draw(drawn["draw"], rec, derived).codes
    if codes:
        return {"codes": list(codes), "step": "select"}
    if isinstance(rule, service.TwoStage):
        if it is None:
            return {"codes": ["interior"], "step": "interiors"}
        raise RuntimeError("this file draws no two-stage run with an interior docket")
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
                           "verdicts": verdicts, "certified": sorted(p.get("certified") or ()), **_named(p)})
    accepted = out["accepted"] is True
    return {"audit": codec.canonical_json(out).decode(), "draw": drawn["draw"],
            "public": shown if accepted else {}, "read": drawn["read"], "receipt": receipt,
            "symbol": "accept" if accepted else "reject"}


def _coins_code(j: dict, rec: dict, read: bytes, key, main) -> str | None:
    """`Run.coinsCode`: a sample that read no coins carries no evidence; one that read coins read the challenger's and
    carries evidence that holds (`coins_reason`) under the run's challenger key `key` and its copy of main's
    history `main`, a second stage's under the same two, for the verifier's window root, of the run's family, after the docket's digest. The code it
    is refused under, or None: `family` for a family whose log shows another root (`FAMILY_ROOT`), else `coins`, a
    verifier that holds no key among them."""
    if not read:
        return None if j.get("coins") is None else "coins"
    window, family = j["params"].get("window_root"), j.get("family")
    if window is None or family is None or key is None:
        return "coins"
    why = coins_reason(ChallengerKey.from_json(key), main, j.get("coins"), window, family, bytes.fromhex(DD.digest(rec)),
                       read, lms_verify)
    return None if why is None else "family" if why == FAMILY_ROOT else "coins"


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
def sampling_designs(draw, n: int):
    kind = draw(st.sampled_from(["subset", "bernoulli", "stratified", "work"]))
    if kind == "bernoulli":
        den = draw(st.integers(1, 6))
        return {"law": "bernoulli", "p": f"{draw(st.integers(0, den))}/{den}"}
    return {"k": draw(st.integers(0 if kind == "subset" else 1, n)), "law": kind}


@st.composite
def agreed_sampling_designs(draw, n: int):
    """One-stage's agreed sampling design: mostly a rate as a selection rule states it (`OneStage.design`), a rate sampling design's tables a divisor
    of `n` so its count fits, else a sampling design object (`sampling_designs`)."""
    kind = draw(st.sampled_from(["rate", "stratified-rate", "work-rate", "rate", "stratified-rate", "work-rate", "object"]))
    if kind == "object":
        return draw(sampling_designs(n))
    den = draw(st.integers(1, 6))
    tables = draw(st.sampled_from([None, *(t for t in (1, 2, 3) if n % t == 0)])) if kind == "rate" else None
    return f"{kind}:{draw(st.integers(1, den))}/{den}" + ("" if tables is None else f":{tables}")


def _worked(sampling_design) -> bool:
    """Whether an agreed sampling design is expanded over a work table: `work` or a work rate."""
    return (sampling_design.get("law") == "work") if isinstance(sampling_design, dict) else str(sampling_design).startswith("work-rate:")


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


def drawn(draw, sampling_design: dict, n: int) -> dict:
    """The verifier's sample under its expanded sampling design; a work sample's closure, which the audit refuses, one time in four."""
    kind = sampling_design["law"]
    if kind == "subset":
        return {"k": sampling_design["k"], "law": "subset", "population": n, "units": sorted(draw(st.permutations(range(n)))[:sampling_design["k"]])}
    if kind == "bernoulli":
        return {"law": "bernoulli", "p": sampling_design["p"], "population": n,
                "units": sorted(draw(st.sets(st.integers(0, n - 1), max_size=n)))}
    units = sorted(u for s in sampling_design["strata"] for u in draw(st.permutations([u for lo, hi in s["units"] for u in range(lo, hi)]))[:s["k"]])
    d = {"k": sampling_design["k"], "law": kind, "population": n, "strata": sampling_design["strata"], "units": units}
    rest = [u for u in range(n) if u not in units]
    if kind == "work" and rest and draw(st.integers(0, 3)) == 0:
        d["closure"] = sorted(draw(st.sets(st.sampled_from(rest), min_size=1)))
    return d


def _reads_window(supplied: dict) -> bool:
    """Whether a supplied value reads the window's salt, which no verifier without a binding fills."""
    return any(k == "window" for rows in G.sources(supplied).values() for row in rows for k, _ in row)


def _record_of(s: Setup, values, sampling_design: dict, window: dict, log=(), binding=None, output=None, window_root=None, fixed=(),
               **kw) -> dict:
    """`R.record` over `ROOTS` registering `values`: `s`'s commitments (W's under `output`'s root and domain when it is
    an earlier record's output, the rest under `window_root`'s domain but `fixed`'s), and each value the verifier fills
    itself as it fills it from that record, `log` and its `binding` (`registered.supplied_value`)."""
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    sup = s.supplied(values) if isinstance(s, RegisteredSetup) else {}
    entries = s.entries({v: w for v, w in values.items() if v not in sup}, window_root, fixed) if values else []
    entries = [{**e, **output} if output and e["value"] == "W" else e for e in entries]
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, roots, window, registered=entries, **kw)
    if not sup:
        return rec
    filled = [G.supplied_value(s.program, v, rows, rec, sorted(sup), list(log), binding=binding,
                               schema=s.formats.get(v, RV.SCHEMA)).entry()
              for v, rows in G.sources(sup).items()]
    return DK.record(s.program, s.query, sampling_design, SCHEME, roots, window, registered=entries + filled, **kw)


def reads(rec: dict) -> dict:
    """A Lean verdict's `registered`: per port of the statement, the value it reads and the root it opened against (the
    record's)."""
    return {f"{e['value']}.{i}": {"root": e["root"], "value": e["value"]} for e in rec["registered"] for i in range(2)}


@st.composite
def audits(draw, setups=(*SETUPS, *REGISTERED_SETUPS)) -> tuple[dict, dict, dict]:
    """A valid audit: the verifier's parameters, the docket and the evidence (verdicts mostly accepting). On a
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
    sampling_design = draw(agreed_sampling_designs(n))
    table = draw(work_tables(s)) if _worked(sampling_design) else None
    try:
        named, expanded = service.params_sampling_design(sampling_design, s.program, s.query, table)
        mine = DR.derive(s.program, s.query, named, expanded)
    except ValueError:
        table = {t: 1 for t in s.templates}
        named, expanded = service.params_sampling_design(sampling_design, s.program, s.query, table)
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
        log = [DK.record(s.program, s.query, named, SCHEME, roots, EARLIER, leaf_layer=LEAF, partition=part,
                        work_table=expanded, registered=given)]
    anchors = draw(st.sampled_from([[], ["weights"]]))
    binding = BINDING if _reads_window(sup) or draw(st.booleans()) else None
    params = {"anchors": anchors, "delta": draw(st.sampled_from(["1/1000", "1/10", "1/2", "999/1000"])), "law": sampling_design,
              "leaf_layer": LEAF, "log": log, "partition": part, "ports": s.ports, "program": s.desc, "query": s.query,
              "registered": values, "scheme": SCHEME, "work_table": table}
    if values is not None:
        params["parameters"] = s.signature
    if isinstance(s, RegisteredSetup) and s.supplied(values):
        params["supplied"] = s.supplied(values)
    if output:
        kind = {"value": EARLIER_VALUE} if reuse else {"port": EARLIER_OUT}
        params["outputs"] = {"W": kind if not reuse and draw(st.booleans()) else {**kind, "record": DD.digest(log[-1])}}
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
                     previous=DD.digest(log[-1]) if log else None, partition=part, work_table=expanded)
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
    _set("record", "format", lambda a, x: [DM.REGISTRATION, "verity/registration/v0"][x % 2]),
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
                                              {"program": "0" * 128, "query": a["params"]["query"]},
                                              {"format": PO.FORMAT, "program": PO.program_sha512(a["params"]["program"]),
                                               "query": {**a["params"]["query"], "version": 99}}][x % 6]),
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
SAMPLE_CHANGES = DRAW  # `DRAW` is the sample piece's port below
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
    a["record"]["previous"] = DD.digest(last)


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
    _set("record", "format", lambda a, x: [DM.REGISTRATION_V1, "verity/registration/v3"][x % 2]),
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
            return {k: v for k, v in e.items() if k != RV.WIDTH_KEY}
        return {**e, ["leaves", "words", RV.WIDTH_KEY, "domain", "root", "value"][x % 7]:
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
    return DD.digest(a["params"]["log"][-1]) if a["params"]["log"] else "ee" * 64


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
        {"W": {"port": "x", "record": _earlier(a)}}, {"W": {"port": EARLIER_OUT, "record": DD.digest(a["record"])}},
        {"W": {"port": EARLIER_OUT, "record": _earlier(a)}}, {"S": {"port": EARLIER_OUT, "record": _earlier(a)}},
        {"nope": {"port": EARLIER_OUT, "record": _earlier(a)}}, {"W": {"port": EARLIER_OUT, "record": "EE" * 64}},
        {"W": {"port": EARLIER_OUT}}, {"W": {"port": 1, "record": _earlier(a)}},
        {"W": {"port": EARLIER_OUT, "record": _earlier(a), "x": 1}}, {"W": EARLIER_OUT}, ["W"], None, {},
        {"W": {"value": EARLIER_VALUE, "record": "ee" * 64}}, {"W": {"value": "nope", "record": _earlier(a)}},
        {"W": {"value": "W", "record": _earlier(a)}}, {"W": {"value": EARLIER_VALUE, "record": _earlier(a)}},
        {"W": {"value": EARLIER_VALUE, "record": DD.digest(a["record"])}}, {"W": {"value": 1, "record": _earlier(a)}},
        {"W": {"value": EARLIER_VALUE, "port": EARLIER_OUT, "record": _earlier(a)}}, {"W": {"value": EARLIER_VALUE}},
        {"S": {"value": EARLIER_VALUE, "record": _earlier(a)}}, {"W": {"port": "nope"}}, {"W": {"port": 1}},
        {"S": {"port": EARLIER_OUT}}, {"W": {"port": "x"}}][x % 28]),
    lambda a, x: a["params"].pop("outputs"),
    _earlier_out,
    _earlier_value,
    _set("params", "row_schemas", lambda a, x: [
        {"A": RV.SCHEMA}, {"A": G.SCHEMA_SEG}, {"B": G.SCHEMA_V2}, {"W": G.SCHEMA_V2}, {"T": G.SCHEMA_SEG}, {"x": G.SCHEMA_V2},
        {"S": G.SCHEMA_V2}, {"nope": G.SCHEMA_V2}, {"A": "u16"}, {"A": "hm96-sha512/row/v2 "}, {"A": 1}, ["A"], "A", None,
        {}][x % 15]),
    lambda a, x: a["params"].pop("row_schemas", None),
]


def _other_reads(r: dict, x: int) -> dict:
    """The verdict of record's registered reads, changed: one port's root another (so its value's two ports disagree), every
    root upper case, a value missing or added (one the record names as a root port, at that port's root or another, or
    one it doesn't name), only one port's, none."""
    ports = sorted(r)
    p = ports[x % len(ports)]
    v = r[p]["value"]
    return [{**r, p: {**r[p], "root": "12" * 64}}, {k: e for k, e in r.items() if e["value"] != v}, {},
            {**r, "extra.0": {"root": "ab" * 64, "value": "extra"}}, {**r, "x.0": {"root": ROOTS["x"], "value": "x"}},
            {**r, "x.0": {"root": "12" * 64, "value": "x"}},
            {k: {**e, "root": e["root"].upper()} for k, e in r.items()}, {p: r[p]}][x % 8]


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
    dom = RV.domain(rec["program"], e["value"], e["words"], e["leaves"], e.get("word_bits", 16), window).domain_id.hex()
    rec["registered"] = [{**e, "domain": dom} if j == i else f for j, f in enumerate(es)]


def _same_window_log(a: dict, x: int) -> None:
    """The log's last record registered the record's values under the record's own domains (the same window), at other
    roots, so R7 holds them to those."""
    last = a["params"]["log"][-1]
    a["params"]["log"] = [*a["params"]["log"][:-1], {**last, "registered": [
        {**e, "root": "12" * 64} for e in a["record"]["registered"]][:1 + x % len(a["record"]["registered"])]}]
    a["record"]["previous"] = DD.digest(a["params"]["log"][-1])


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
#: A value a verifier holds a certified entry for, none of these programs' parameters (a link's certified rows), and the
#: entry it built from the docket it read under the deployment's key.
CERTIFIED_VALUE = "link-0/in/7"
CERTIFIED_ENTRY = {"domain": "c4" * 64, "leaves": 3, "root": "c5" * 64, "words": 64}


def _certify(mine=None, entry=None):
    """The verifier holding a certified entry for `CERTIFIED_VALUE` (`mine`: its change to the parameter) and the
    record, as a v2 record, registering the value beside its others (`entry`: its change to that entry, None for no
    entry)."""
    def change(a, x):
        held = {CERTIFIED_VALUE: dict(CERTIFIED_ENTRY)}
        a["params"]["certified"] = mine(held, x) if mine else held
        e = {"value": CERTIFIED_VALUE, **CERTIFIED_ENTRY}
        e = entry(e, x) if entry else e
        if e is not None:
            a["record"]["registered"] = [*a["record"].get("registered", []), e]
            a["record"]["format"] = DM.REGISTRATION
    return change


def _uncertified(a: dict, x: int) -> None:
    """The record registers the certified value, and the verifier holds no entry for it."""
    _certify()(a, x)
    del a["params"]["certified"]


#: One change each to the certified entries a verifier holds and the record's entry for the certified value: held and
#: registered alike, at 16 bits and at 32; the record's entry changed or left out; the verifier's entry in another form;
#: its entries in another form or of other values, a parameter it expects registered among them; and an entry the
#: record registers that the verifier holds none for.
CERTIFIED = [
    _certify(),
    _certify(entry=lambda e, x: [None, {**e, "root": "c6" * 64}, {**e, "leaves": 4}, {**e, "words": 32},
                                 {**e, "domain": "c7" * 64}, {**e, "word_bits": 32}, {**e, "word_bits": 16},
                                 {**e, "root": "xy"}, {**e, "leaves": "3"}, {**e, "extra": 1}, {**e, "value": "W"},
                                 {k: v for k, v in e.items() if k != "root"}][x % 12]),
    _certify(mine=lambda c, x: {CERTIFIED_VALUE: {**c[CERTIFIED_VALUE], **[
        {"root": "c6" * 64}, {"leaves": 4}, {"word_bits": 32}, {"word_bits": 16}, {"word_bits": 8}, {"word_bits": "16"},
        {"root": "C5" * 64}, {"root": "c5"}, {"root": 5}, {"leaves": -1}, {"leaves": True}, {"leaves": "3"},
        {"words": "64"}, {"domain": 5}, {"extra": 1}][x % 15]}}),
    _certify(mine=lambda c, x: [{}, {"W": c[CERTIFIED_VALUE]}, {"x": c[CERTIFIED_VALUE]}, {"S": c[CERTIFIED_VALUE]},
                                {**c, "link-0/out/7": c[CERTIFIED_VALUE]}, None, [], "x", {CERTIFIED_VALUE: None},
                                {CERTIFIED_VALUE: []},
                                {CERTIFIED_VALUE: {k: v for k, v in c[CERTIFIED_VALUE].items() if k != "root"}}][x % 11]),
    _certify(mine=lambda c, x: {CERTIFIED_VALUE: {**c[CERTIFIED_VALUE], "word_bits": 32}},
             entry=lambda e, x: {**e, "word_bits": 32}),
    _uncertified,
]


GROUPS = [RECORD, PARAMS, DRAW, EVIDENCE, VERDICTS_OF, ENTRIES, REGISTERED_PARAMS, REGISTERED_EVIDENCE, SUPPLIED, BINDINGS,
          WINDOWS, SEED_SALTS]


@st.composite
def changed_audits(draw, groups=GROUPS, most: int = 2, least: int = 0, setups=(*SETUPS, *REGISTERED_SETUPS)) -> dict:
    """An audit as `{"params", "record", "evidence"}`, with `least` to `most` changes, each from one of `groups`; a change
    that doesn't apply to what an earlier one left (a field it removed, a list it replaced or emptied) is skipped."""
    params, rec, evidence = draw(audits(setups))
    a = {"params": params, "record": rec, "evidence": evidence}
    for _ in range(draw(st.integers(least, most))):
        change = draw(st.sampled_from(draw(st.sampled_from(groups))))
        try:
            change(a, draw(st.integers(0, 63)))
        except (KeyError, TypeError, IndexError, ValueError, AttributeError, ZeroDivisionError):
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

ONE_STAGE = D.ProtocolPort("spot-one-stage", answer, CASES, package=PACKAGE)
server = D.server_fixture(ONE_STAGE)


def test_the_one_stage_audit_agrees(server):
    stats = D.run(ONE_STAGE, examples=600, server=server)
    assert stats.examples == 600 and min(stats.counts.get(v, 0) for v in ("accept", "reject", "undecided")) > 40, stats


def test_an_accepted_case_is_compared_whole(server):
    """One accepted audit of each sampling design the audit covers, field by field: the public outputs and the profile Python
    writes."""
    for sampling_design in ({"k": 2, "law": "subset"}, {"k": 3, "law": "stratified"}, {"k": 3, "law": "work"}, "rate:1/3",
                "stratified-rate:1/2", "work-rate:1/3"):
        s = SETUPS[3]
        table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}} if _worked(sampling_design) else None
        named, expanded = service.params_sampling_design(sampling_design, s.program, s.query, table)
        mine = DR.derive(s.program, s.query, named, expanded)
        rec = DK.record(s.program, s.query, named, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                       WINDOW, leaf_layer=LEAF, work_table=expanded)
        units = [0, 3][:mine["k"]] if mine["law"] == "subset" else sorted(
            u for st_ in mine["strata"] for u in [u for lo, hi in st_["units"] for u in range(lo, hi)][:st_["k"]])
        d = {**{k_: v for k_, v in mine.items()}, "population": s.population, "units": units}
        proved = DR.proved(d)
        evidence = {"draw": d, "instances": len(proved), "proved_units": proved,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                    "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
        params = {"anchors": [], "delta": "1/20", "law": sampling_design, "leaf_layer": LEAF, "log": [], "partition": None,
                  "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
                  "work_table": table}
        case = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        verdict_, line = D.replay(ONE_STAGE, case, server)
        assert verdict_ == "accept" and json.loads(line) == answer(case) | {"reason": None}, (sampling_design, line)


def test_an_accept_says_whether_every_verdict_ran(server):
    """An accept's public ``complete``, Lean's and the Python's alike: true when every verdict ran, and false beside
    one that is not of record and did not run (its ``accepted`` null or absent), which the accept does not wait on."""
    s, design = SETUPS[3], {"k": 2, "law": "subset"}
    mine = DR.derive(s.program, s.query, design, None)
    rec = DK.record(s.program, s.query, design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, work_table=None)
    d = {**mine, "population": s.population, "units": [0, 3]}
    params = {"anchors": [], "delta": "1/20", "law": design, "leaf_layer": LEAF, "log": [], "partition": None,
              "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
              "work_table": None}
    for other, complete in (({}, True), ({"upstream": {"accepted": True, "of_record": False}}, True),
                            ({"upstream": {"accepted": None, "of_record": False}}, False),
                            ({"upstream": {"of_record": False}}, False)):
        evidence = {"draw": d, "instances": 2, "proved_units": DR.proved(d),
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                    "sessions": [], "statement": {},
                    "verdicts": {"lean": {"accepted": True, "of_record": True}, **other}}
        case = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        verdict_, line = D.replay(ONE_STAGE, case, server)
        got = json.loads(line)
        assert verdict_ == "accept" and got == answer(case) | {"reason": None}, (other, line)
        assert got["public"]["complete"] is complete, (other, line)


ONE_STAGE_V2 = D.ProtocolPort("spot-one-stage-zk", answer, CASES, package=PACKAGE)
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


def subset_only(kind: str) -> str:
    """`OneStageZk.SUBSET_ONLY`"""
    return (f"the sampling design is {kind}, which the one-statement audit's soundness statement doesn't cover: "
            "it covers a subset sampling design")


def _reversed(j):
    """`j` with every object's keys in reverse order."""
    return {k: _reversed(j[k]) for k in reversed(list(j))} if isinstance(j, dict) else j


def test_the_parameters_gate_refuses_every_number_form_but_an_integer():
    """`Core.canonicalJson` refuses the other forms of a number, so the query `jeq` and `query_digest` compare is the one
    canonical text."""
    for text in (b'{"v":1.0}', b'{"v":1e0}', b'{"v":10e-1}'):
        with pytest.raises(PR.Reject):
            PR.canonical_json(text)
    assert R.query_digest(_reversed({"b": 1, "a": {"d": 2, "c": 3}})) == R.query_digest({"a": {"c": 3, "d": 2}, "b": 1})


def parsed_params_v2(j) -> dict:
    """`OneStageZk.paramsParsed`'s refusals of v1's part of the parameters: v1's (`covered_params`,
    `OneStage.paramsOfJson`), then parameters naming no challenger (`NO_CHALLENGER`), or one whose key is at other LMS
    parameters than the deployed ones (`KEY_PARAMS`), or a partition object whose query is not `query`
    (`PARTITION_QUERY`). Its sampling design is the sampling design v1 resolved, under any kind v1
    parses."""
    rest = {k: v for k, v in j.items() if k != "proofs"}
    p = covered_params(rest)
    if "challenger" not in rest:
        raise D.Reject(NO_CHALLENGER)
    key = ChallengerKey.from_json(rest["challenger"]["key"])
    if (key.W, key.H) != DEPLOYED:
        raise D.Reject(KEY_PARAMS)
    if rest["partition"] is not None and R.query_digest(rest["partition"]["query"]) != R.query_digest(rest["query"]):
        raise D.Reject(M.PARTITION_QUERY)
    return p


def subset_design(p: dict) -> dict:
    """`OneStageZk.paramsOfJson`'s last refusal, of parsed parameters (`parsed_params_v2`) whose sampling design is no
    `subset` one, by its kind (`subset_only`): a rate stated as text is its design over the population by here."""
    if p["law"]["law"] != "subset":
        raise D.Reject(subset_only(p["law"]["law"]))
    return p


def decode_params_v2(j) -> dict:
    """`OneStageZk.paramsOfJson`'s refusals of v1's part of the parameters: `parsed_params_v2`'s, then a sampling design
    that is no `subset` one (`subset_design`)."""
    return subset_design(parsed_params_v2(j))


def _challenger_sample(rng: random.Random, s: Setup, sampling_design: dict, rec: dict, trees: int = 1,
                     keys: list[str] | None = None, seed: int = 0, order: str = "honest",
                     **shape: bool) -> tuple[Drawn, dict, dict]:
    """A sample under `sampling_design` from the coins of a challenger session its ledger opens withheld, under the key of `seed` at
    the deployed parameters, released after `rec`'s digest and read as `draw_first` reads them, the session's
    challenger sampling and opening `trees` coin trees too (`TreeSession`) on v2's one table (`OneStageZk.sessionTables`),
    their links opened after `keys` (that table when none), its log in `order` (`ORDERS`): the session, the sample, and the
    evidence's `coins` (the session's record without the key, which the verifier's parameters name). `shape`: `Drawn`'s
    `before` or `later`, a session whose family holds another root."""
    mine = DR.derive(s.program, s.query, sampling_design, None)

    def reads(coins) -> bytes:
        got = bytearray()

        def source(k: int) -> bytes:
            b = coins(k)
            got.extend(b)
            return b
        draw_first(OwnRandomness(source), mine, s.population)
        return bytes(got)
    d = Drawn(rng, after=bytes.fromhex(DD.digest(rec)), reads=reads, trees=trees, tables=V2_TABLES,
              keys=keys, params=[DEPLOYED], seed=seed, order=order, **shape)
    at = [0]

    def source(k: int) -> bytes:
        at[0] += k
        return d.read[at[0] - k:at[0]]
    coins = d.evidence(d.segment.to_json())
    return d, draw_first(OwnRandomness(source), mine, s.population), coins


def test_v2_reads_no_verdict_from_the_evidence(server, server_v2):
    """v2 (`Spot.OneStageZk`, `one_stage.proofs.verdicts_v2`) is v1 with the verdicts its own run of the verifier
    on the proofs. An audit v1 accepts on the verdict its evidence hands in and its challenger's coins is, in v2: refused
    for naming verdicts; refused, before any tree is read, with them replaced by proofs naming no session or two (the
    sampled proof blocks are proved in one session, `oneSession`); rejected by the session's own verdict when its files are not
    a statement's; refused before the session is verified when its coins are not a tree the challenger logged or its
    link was opened after other keys than its one table (`LINK_TABLES`; the `root_B` length a statement fixes,
    `LINK_ROOT_LEN`, these files set up none to hold), or the public file is not the docket's
    (`publicHeld`, its codes Python's `matches_public`'s), or its ``units.indices`` are not the population's units in
    order (`indicesHeld`): a header rewritten after the sample, its pin with it, to credit an honest instance as the sampled
    proof block, or naming none; refused when a file it names is in no blob, as the public file the parameters name is when
    the blobs hold another, or its sessions are not the evidence's. The public file is the parameters', committed before
    any coin: evidence naming one of its own, or parameters naming none, are refused. `test_proofs.py` holds the
    verdicts to `verify`'s on a real session, and `test_challenger_difftest.py` the trees' check to its Python."""
    s = BLOCKED
    sampling_design = {"k": 2, "law": "subset"}
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, partition=s.partition)
    rng = random.Random("v2-verdicts")
    circuit, public = STAND_IN_CIRCUIT, _public_of(rec)

    def audit(trees: int, keys: list[str] | None = None) -> tuple[Drawn, dict, dict, dict]:
        ch, d, coins = _challenger_sample(rng, s, sampling_design, rec, trees, keys)
        proved = DR.proved(d)
        evidence = {"coins": coins, "draw": d, "instances": len(proved), "proved_units": proved,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": sorted(d["units"]),
                    "session_roots": {r["port"]: r["root"] for r in rec["roots"]},
                    "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
        params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/20",
                  "law": sampling_design, "leaf_layer": LEAF, "log": [], "partition": s.partition,
                  "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
                  "window_root": ch.ledger.window["root"], "work_table": None}
        v1 = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        assert D.replay(ONE_STAGE, v1, server)[0] == "accept"
        base = {k: v for k, v in evidence.items() if k != "verdicts"}
        return ch, evidence, base, params | {"proofs": {"public": _sha(public), "registered": None}}

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
    assert run(base | {"proofs": proofs(["s"]), "sessions": ["s"]}, blobs_=blobs[1:]) == \
        run(base | {"proofs": proofs(["s"]), "sessions": ["s"]})
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
    assert got == ("reject", "params: the keys of proofs are not public, registered"), got
    for keys in (["circuit", "z"], ["a", "circuit"], ["circuit-00"], []):
        ch_k, _, base_k, p_k = audit(1, keys)
        extra = D.canonical(ch_k.trees[0].record).encode()
        assert run(base_k | {"proofs": proofs(["s"], extra), "sessions": ["s"]},
                   blobs_=[circuit.hex(), public.hex(), extra.hex()], p=p_k) == (
            "reject", f"proofs: session s's coins are not the challenger's: {CH.LINK_TABLES}"), keys


def _audit_v1_accepts(s: Setup, seed: str, server) -> tuple:
    """An audit of `s` that v1 accepts on the verdict its evidence hands in and its challenger's coins, sampled under
    `seed`: the docket, the challenger's session, v1's evidence without its ``sessions`` and
    ``verdicts``, and v1's parameters, which v2 reads with ``proofs``."""
    sampling_design = {"k": 2, "law": "subset"}
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, partition=s.partition)
    ch, d, coins = _challenger_sample(random.Random(seed), s, sampling_design, rec)
    proved = DR.proved(d)
    base = {"coins": coins, "draw": d, "instances": len(proved), "proved_units": proved,
            "receipt": R.receipt(rec, "v", received=RECEIVED), "served": sorted(d["units"]),
            "session_roots": {r["port"]: r["root"] for r in rec["roots"]},
            "statement": {}}
    params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/20",
              "law": sampling_design, "leaf_layer": LEAF, "log": [], "partition": s.partition,
              "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
              "window_root": ch.ledger.window["root"], "work_table": None}
    v1 = base | {"sessions": [], "verdicts": {"lean": {"accepted": True, "of_record": True}}}
    assert D.replay(ONE_STAGE, _json({"commitment": _hex(rec), "evidence": _hex(v1), "params": params}),
                    server)[0] == "accept", s.templates
    return rec, ch, base, params


def test_the_registration_after_the_draw_is_held_in_the_check_as_python_does(server):
    """The registration's half after the draw in the whole check (`OneStage.check`, which `checkAt` runs; its twin
    `docket.registration_after_draw`), on a v1 audit from the challenger's coins whose parameters take the registration
    from the ledger: the ledger that signs the record's digest for the session's lane before the session, in a period
    whose open commits the docket's program as its circuit, accepts; one whose registration was signed after the session
    opened, a lane the session is not on, a ledger after the draw that holds the epoch to another digest than the one
    before it, and a period that commits another circuit (the SHA-512 of the descriptor's bytes as written, or any other) each
    pass the half before the draw and are rejected by the check by name, Lean's reason the Python's."""
    s = BLOCKED
    sampling_design = {"k": 2, "law": "subset"}
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, partition=s.partition)
    body = _record_body([])
    digest = DD.digest(json.loads(body))
    rng = random.Random("registration-after-draw")
    ch, d, _ = _challenger_sample(rng, s, sampling_design, rec)
    while ch.ledger.period_open is None:
        ch, d, _ = _challenger_sample(rng, s, sampling_design, rec)
    lg, proved = ch.ledger, DR.proved(d)
    events, opened = list(lg.events), lg.period_open

    program = P.program_digest(s.program)

    def lines(registration: str, earlier: str | None = None, after: bool = False, circuit: str = program) -> list[str]:
        lg.period_open = {**opened, "registration": registration, "circuit": circuit}
        lg.events = [(k, sid, lg.period_open if b is opened else b) for k, sid, b in events]
        if after:
            lg.events = [e for e in lg.events if e[0] not in ("period-open", "period-close")] + [
                ("period-open", None, lg.period_open)]
        if earlier is not None:
            at = 2 if lg.v1 else 0
            first = {**opened, "lane": earlier, "period": "p-first", "registration": registration}
            lg.events[at:at] = [("period-open", None, first), (
                "period-close", None, {"complete": False, "lane": earlier, "period": "p-first", "windows": []})]
        return lg.lines()

    def run(before: list[str], after: list[str], lane: str = lg.lane) -> tuple[str, str | None, str | None]:
        coins = {**ch.evidence(ch.segment.to_json()), "ledger": after}
        evidence = {"coins": coins, "draw": d, "instances": len(proved), "proved_units": proved,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": sorted(d["units"]),
                    "session_roots": {r["port"]: r["root"] for r in rec["roots"]},
                    "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
        params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/20",
                  "law": sampling_design, "leaf_layer": LEAF, "log": [], "partition": s.partition,
                  "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
                  "window_root": lg.window["root"], "work_table": None,
                  "parameters": _json([[n, t.to_json()] for n, t in s.program.fn.signature()[0]]),
                  "registration": {"record": body.hex(), "ledger": before, "lane": lane}}
        held = R.registration_parsed(params["registration"], params["challenger"], params["window_root"], lms_verify)
        assert held is not None and held[1] == digest
        try:
            R.registration_after_draw(params["registration"], params["challenger"], params["window_root"], digest, coins,
                                      lms_verify, program)
            py = None
        except R.RegistrationRefused as e:
            py = str(e)
        out = server.exchange(_json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params}))
        return out["verdict"], out.get("reason"), py

    honest = lines(digest)
    assert run(honest, honest) == ("accept", None, None)
    other_lane = next(x for x in LANES if x != lg.lane)
    for before, after, lane in ((lines(digest, after=True),) * 2 + (lg.lane,),
                                (lines(digest, earlier=other_lane),) * 2 + (other_lane,),
                                (honest, lines("ee" * 64), lg.lane)):
        verdict, reason, py = run(before, after, lane)
        assert py is not None and py.startswith("registration, after the draw: "), py
        assert (verdict, reason) == ("reject", py)
    written = hashlib.sha512(json.dumps(s.desc).encode()).hexdigest()
    assert written != program
    for circuit in (written, "dd" * 64):
        other = lines(digest, circuit=circuit)
        assert run(honest, other) == ("reject", R.PERIOD_CIRCUIT_WHY, R.PERIOD_CIRCUIT_WHY)


def test_v2_takes_a_cover_of_the_population_under_a_subset_design(server, server_v2):
    """v2's population check (`OneStageZk.popHeld`) takes a docket whose program's proof blocks, read stratum by
    stratum, are not its population's units in order but a disjoint cover of them (`Slots.popCovers`) under its subset
    design, as the service's run does (`test_the_run_takes_a_cover_of_the_units_only_under_a_subset_design`):
    `SETUPS[4]`'s and `SETUPS[5]`'s audits go on past it, as `BLOCKED`'s, in order, does to its session's verdict. v2's one
    statement takes no stratified or work design (`subset_only`)."""
    circuit = STAND_IN_CIRCUIT
    for s, ok in ((SETUPS[4], False), (SETUPS[5], False), (BLOCKED, True)):
        assert P.in_order(s.program, s.query, s.population) is ok, s.templates
        rec, ch, base, params = _audit_v1_accepts(s, f"v2-order-{s.templates}", server)
        public, record = _public_of(rec), D.canonical(ch.trees[0].record).encode()
        v2 = base | {"proofs": {"sessions": [{"name": "s", "record": _sha(record), "reps": {}}]}, "sessions": ["s"]}
        case = {"blobs": [circuit.hex(), public.hex(), record.hex()], "commitment": _hex(rec), "evidence": _hex(v2),
                "params": params | {"proofs": {"public": _sha(public), "registered": None}}}
        out = server_v2.exchange(_json(case))
        assert P.covers(s.program, s.query, s.population) and out["verdict"] == "reject", (s.templates, out)
        assert POPULATION_ORDER not in out["reason"] and "partition: the registration's strata" not in out["reason"], out
        if ok:
            assert out["reason"] == "verdict flock-verify/s did not accept", (s.templates, out)


def test_the_per_member_form_as_written_is_what_v2_parses_and_its_check_reads(tmp_path, server, server_v2):
    """`proofs.members_v2` writes v2's per-member form of ``proofs`` for a window of several members, and
    `OneStageZk`'s parsers read what it wrote (`proofs.members_parsed`, the executable's ``members``:
    `membersParamsOfJson`, `membersEvidenceOf`, `membersHeld`, `sampledHeld`), on an audit v1 accepts: every member
    with its files' digests, its units (its block of the partition object's ``blocks``) and its ``vus_per_block``,
    and a statement for each member with a session, in
    the members' order, each a member the evidence's ``proved_units`` hold a unit of. The command's line says
    ``parsed, not checked`` and names no verdict. The audit's check reads the same case in the form and rejects it at
    the first check that fails: the stand-in sessions' records name no coin tree the challenger's release log shows.
    What the parsers refuse the command refuses with their reason: statements out of the members'
    order, a statement of two sessions, of a member the parameters don't name or of a member the sample holds no unit
    of, v1's ``sessions`` other than the statements', and either file in the one statement's form."""
    blocked = Setup(SETUPS[3].program, SETUPS[3].query, blocks=[[(0, 2), (2, 4), (4, 5)]])
    rec, _, base, params = _audit_v1_accepts(blocked, "v2-members", server)

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
                         PF.file_sha512(at / "pub.bin"), None, ((lo, hi),), 4)

    ms = [member(0, f"{PAIR.id}/c0", 0, 2, "s0"), member(1, f"{TRIPLE.id}/c0", 2, 4),
          member(2, f"{TRIPLE.id}/c1", 4, 5, "s1")]
    pm, pe, blobs = PF.members_v2(ms)
    names = [x["name"] for st_ in pe["statements"] for x in st_["sessions"]]
    assert names == ["s0", "s1"] and [st_["member"] for st_ in pe["statements"]] == [ms[0].name, ms[2].name]
    assert blobs == [f for m in (ms[0], ms[2]) for f in (m.proof.circuit, m.proof.public, m.proof.sessions[0] / "session.json",
                                                         m.proof.sessions[0] / "rep-0.bin")] + [ms[1].proof.circuit, ms[1].proof.public]
    # the sample: a unit of each member with a session, and none of the member without one
    p2, e2 = params | {"proofs": pm}, base | {"proofs": pe, "sessions": names, "proved_units": [1, 4]}
    got, refused = PF.members_parsed(p2, e2)
    assert refused is None, refused
    assert got == {
        "members": sorted(({"name": m.name, "public": m.public_sha512, "registered": False,
                            "units": [list(u) for u in m.units], "vus_per_block": 4} for m in ms), key=lambda m: m["name"]),
        "parsed": "parsed, not checked",
        "statements": [{"member": ms[0].name, "sessions": ["s0"]}, {"member": ms[2].name, "sessions": ["s1"]}]}
    assert PF.PARSED == got["parsed"] and not {"verdict", "accept", "accepted", "public", "profile"} & set(got)

    case = {"blobs": [f.read_bytes().hex() for f in blobs], "commitment": _hex(rec), "evidence": _hex(e2), "params": p2}
    out = server_v2.exchange(_json(case))
    assert (out["verdict"], out["reason"]) == ("reject", f"proofs: session s0's coins are not the challenger's: {NO_TREE}")

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
    assert why(e=e2 | {"proved_units": [1, 3]}) == (
        f"proofs: member {ms[2].name}: the sample holds none of its units, and a member the sample missed has no statement")
    assert why(e=e2 | {"proved_units": [2, 3, 4]}) == (
        f"proofs: member {ms[0].name}: the sample holds none of its units, and a member the sample missed has no statement")
    assert why(e=e2 | {"sessions": names[::-1]}) == (
        "evidence: proofs: the sessions are not the evidence's sessions, in order")
    one = {"sessions": [x for st_ in pe["statements"] for x in st_["sessions"]]}
    assert why(e=e2 | {"proofs": one}) == "evidence: the keys of the evidence's proofs are not statements"
    assert why(p=params | {"proofs": {"public": ms[0].public_sha512, "registered": None}}) == (
        "params: the keys of proofs are not members")
    again = pm["members"] | {"z/c0": pm["members"][ms[0].name]}
    assert why(p=params | {"proofs": {"members": again}}) == (
        "params: proofs: unit 0 is in two members' units: a unit is one member's")
    handed = pm["members"] | {ms[1].name: {**pm["members"][ms[1].name], "units": [[2, 4]]}}
    assert why(p=params | {"proofs": {"members": handed}}) == f"params: proofs: member {ms[1].name}: {M.UNITS_HANDED}"
    past = pm["members"] | {ms[1].name: {**pm["members"][ms[1].name], "block": 3}}
    assert why(p=params | {"proofs": {"members": past}}) == (
        f"params: proofs: member {ms[1].name}: partition: E_BLOCKS: no block 3 among the partition's 3")
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
    s = BLOCKED
    sampling_design = {"k": 2, "law": "subset"}
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, partition=s.partition)
    circuit, public = STAND_IN_CIRCUIT, _public_of(rec)
    for order in ORDERS:
        ch, d, coins = _challenger_sample(random.Random(f"v2-order-{order}"), s, sampling_design, rec, 1, order=order)
        proved = DR.proved(d)
        evidence = {"coins": coins, "draw": d, "instances": len(proved), "proved_units": proved,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": sorted(d["units"]),
                    "session_roots": {r["port"]: r["root"] for r in rec["roots"]},
                    "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
        params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/20",
                  "law": sampling_design, "leaf_layer": LEAF, "log": [], "partition": s.partition,
                  "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
                  "window_root": ch.ledger.window["root"], "work_table": None}
        v1 = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        assert D.replay(ONE_STAGE, v1, server)[0] == "accept", order
        want = None if order == "honest" else f"coins: {Ch.SAMPLE_LINK_FIRST}"
        assert order_why(params, rec, evidence) == want, order
        assert Ch.sample_link_first("draw", bytes.fromhex(DD.digest(rec)), ch.s.lines[1:]) == (want is None), order
        record = D.canonical(ch.trees[0].record).encode()
        proofs = {"sessions": [{"name": "s", "record": _sha(record), "reps": {}}]}
        base = {k: v for k, v in evidence.items() if k != "verdicts"}
        case = {"blobs": [circuit.hex(), public.hex(), record.hex()], "commitment": _hex(rec),
                "evidence": _hex(base | {"proofs": proofs, "sessions": ["s"]}),
                "params": params | {"proofs": {"public": _sha(public), "registered": None}}}
        out = server_v2.exchange(_json(case))
        assert (out["verdict"], out["reason"]) == (
            "reject", "verdict flock-verify/s did not accept" if want is None else f"proofs: {want}"), order


def test_link_points_refuses_a_sampling_design_none_of_whose_samples_ends():
    """`link_points.refused` refuses a sampling design no sample of which ends (`NEVER_ENDS`, `ends`): `subset:11` over 10 units, a
    Bernoulli sampling design of denominator 0, a stratum sampling more units than it has; `subset:10` and `bernoulli 0/0` end. These
    are `LinkPoints`' `#guard`s. The `link_points` piece can't reach them, since `Draw.flockSamplingDesign` (`coins.sampling_design_problem`)
    refuses a subset above the population and a denominator of 0, and a stratum's rule samples at most its units."""
    stratum = {"k": 3, "law": "stratified", "strata": [{"k": 3, "template": "a", "units": [[0, 2]]}]}
    assert LP.ends({"k": 10, "law": "subset"}, 10) and LP.ends({"law": "bernoulli", "p": "0/0"}, 10)
    for sampling_design in ({"k": 11, "law": "subset"}, {"law": "bernoulli", "p": "1/0"}, stratum):
        assert not LP.ends(sampling_design, 10)
        with pytest.raises(ValueError, match=f"^{LP.NEVER_ENDS}$"):
            LP.refused(sampling_design, 10, [([[0, 10]], 1)])


def test_v2_reads_its_sample_from_the_challengers_coins(server_v2):
    """v2 reads its sample's coins as v1 does (`OneStageZk.coinsOf`, v1's `coinsOf`), and only from the auditor's
    challenger under a key at the deployed parameters: parameters naming none, one without a window (`NO_CHALLENGER`) or
    one whose key is at other parameters (`KEY_PARAMS`) are refused, as are parameters whose sampling design is no
    `subset` one (`subset_only`: a `stratified` or `work` one, as an object or as a rate stated as text, while a rate
    that names the audit's subset sampling design over the population is read as that sampling design), and evidence holding more than its limits
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
    s, sampling_design = BLOCKED, {"k": 3, "law": "subset"}
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, partition=s.partition)
    after = bytes.fromhex(DD.digest(rec))
    circuit, public = STAND_IN_CIRCUIT, _public_of(rec)
    failed = {"flock-verify/s": {"accepted": False, "of_record": True}}
    table = {PAIR16.id: 2, TRIPLE16.id: {"floor": 1, "work": 1}}
    uncovered = {"a stratified design": ({"k": 3, "law": "stratified"}, None, "stratified"),
                 "a work design": ({"k": 3, "law": "work"}, table, "work"),
                 "a work design with no table": ({"k": 3, "law": "work"}, None, "work"),
                 "a stratified rate": ("stratified-rate:1/2", None, "stratified"),
                 "a work rate": ("work-rate:1/2", table, "work")}

    def audit_of(d: Drawn, draw: dict, coins, **params) -> dict:
        proved = DR.proved(draw)
        record = D.canonical(d.trees[0].record).encode() if d.trees else b"{}"
        proofs = {"sessions": [{"name": "s", "record": _sha(record), "reps": {}}]}
        evidence = {"draw": draw, "instances": len(proved), "proved_units": proved, "proofs": proofs,
                    "receipt": R.receipt(rec, "v", received=RECEIVED), "served": sorted(draw["units"]),
                    "session_roots": {r["port"]: r["root"] for r in rec["roots"]},
                    "sessions": ["s"], "statement": {}}
        if coins is not None:
            evidence["coins"] = coins
        p = {"anchors": [], "challenger": {"key": d.s.key.to_json(), "label": "draw"}, "delta": "1/20", "law": sampling_design,
             "leaf_layer": LEAF, "log": [], "partition": s.partition, "ports": s.ports,
             "program": s.desc, "proofs": {"public": _sha(public), "registered": None},
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
        d, draw, coins = _challenger_sample(rng, s, sampling_design, rec)
        honest = audit_of(d, draw, coins)
        assert (python(honest), lean(honest)) == ("verdict", "verdict")
        other = next(x for x in (_challenger_sample(rng, s, sampling_design, rec)[1] for _ in range(64)) if x["units"] != draw["units"])
        stranger = _challenger_sample(rng, s, sampling_design, rec, seed=1)[0]
        assert stranger.s.key != d.s.key
        elsewhere = Drawn(rng, reads=lambda c: c(COIN_BYTES), params=[DEPLOYED])
        bare = Drawn(rng, after=after, reads=lambda c: c(COIN_BYTES), withheld=False, params=[DEPLOYED])
        again = {shape: _challenger_sample(rng, s, sampling_design, rec, **{shape: True}) for shape in ("before", "later")}
        undeployed = {**d.s.key.to_json(), **rng.choice([{"H": 6}, {"W": 2}, {"W": 4, "H": 10}])}
        cases = {
            "a partition object in another key order": ("verdict", audit_of(d, draw, coins, partition=_reversed(s.partition))),
            "no challenger": ("params", audit_of(d, draw, coins, challenger=None)),
            "no challenger, no coins": ("params", audit_of(d, draw, None, challenger=None)),
            "no window": ("params", audit_of(d, draw, coins, window_root=None)),
            "a key at other parameters": ("params", audit_of(d, draw, coins, challenger={"key": undeployed,
                                                                                        "label": "draw"})),
            "an opening's path past its limit": ("evidence", audit_of(d, draw, path_of(coins, MAX_PATH + 1))),
            "an opening's path at its limit": ("coins", audit_of(d, draw, path_of(coins, MAX_PATH))),
            "other draw": ("draw", audit_of(d, other, coins)),
            "no coins": ("coins", audit_of(d, draw, None)),
            "the record's own key": ("coins", audit_of(d, draw, {**coins, "key": d.s.key.to_json()})),
            "another key": ("coins", audit_of(d, draw, coins, challenger={"key": stranger.s.key.to_json(),
                                                                             "label": "draw"})),
            "another window": ("coins", audit_of(d, draw, coins, window_root=d.other_window(rng))),
            "another family": ("coins", audit_of(d, draw, coins, challenger={"key": d.s.key.to_json(),
                                                                                "label": "other"})),
            "forged record": ("coins", audit_of(d, draw, signed(coins, "record"))),
            "forged close": ("coins", audit_of(d, draw, signed(coins, "close"))),
            "released after another digest": ("coins", audit_of(
                elsewhere, draw, elsewhere.evidence(elsewhere.segment.to_json()))),
            "not withheld": ("coins", audit_of(
                bare, draw, bare.evidence(bare.segment.to_json()))),
            "a family with a root before the sample's": ("coins", audit_of(*again["before"])),
            "a family with a root after the sample's": ("coins", audit_of(*again["later"])),
            "before": ("coins", audit_of(again["before"][0], again["before"][1], {
                **again["before"][2], "segment": _mutated(rng, again["before"][0], "before")[0]})),
        } | {name: ("coins", audit_of(d, draw, {**coins, "segment": _mutated(rng, d, name)[0]}))
             for name in SAMPLE_MUTATIONS if name not in ("before", "unread", "read-tampered", "other-after")}
        for name, (want, a) in cases.items():
            assert (python(a), lean(a)) == (want, want), (seed, name)
        for name, (other, work, kind) in uncovered.items():
            a = audit_of(d, draw, coins, law=other, **({} if work is None else {"work_table": work}))
            with pytest.raises(D.Reject) as refused:
                decode_params_v2(a["params"])
            out = outcome(a)
            assert (str(refused.value), out["verdict"], out["reason"]) == (
                subset_only(kind), "reject", f"params: {subset_only(kind)}"), (seed, name)
        rate = audit_of(d, draw, coins, law="rate:3/5")
        assert decode_params_v2(rate["params"])["law"] == sampling_design and (python(rate), lean(rate)) == ("verdict", "verdict"), seed
        for name, (c, why) in unclosed(coins).items():
            a = audit_of(d, draw, c)
            e = {k: v for k, v in a["evidence"].items() if k != "proofs"} | {"verdicts": failed}
            with pytest.raises(CoinError) as py:
                _coins_of(decode_params_v2(a["params"]), (rec, after), e)
            out = outcome(a)
            assert (str(py.value), out["verdict"], out["reason"]) == (why, "reject", f"coins: {why}"), (seed, name)


def committed_bits(j):
    """`Q_call` v1 at X over a program of root Calls (`partition_object.evaluate`): per distinct Definition, in Call order,
    its units' committed bits (`partition_query._committed_bits`), twice, as Lean gives them read off its layout and by the walk
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
        bits = PQ._committed_bits(cut.fn, [(lo, hi) for lo, hi, _path in cut.units])
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
    """A template query's fan-out over a program (`partition.fanout`, `Partition.fanout`), refused for another query or
    one that doesn't apply."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    desc = _obj(j.get("program"), "program")
    query = _obj(j.get("query"), "query")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    if not P.is_template_query(query):
        raise D.Reject("not a template query")
    try:
        return {"fanout": P.fanout(PROGRAMS[sha], query)}
    except (P.QueryRefused, ValueError) as e:
        raise D.Reject(str(e)) from None


def _ranges(x, what: str) -> list:
    if not isinstance(x, list) or not all(isinstance(h, list) and len(h) == 2 and all(_nat(v) for v in h) for h in x):
        raise D.Reject(f"{what} is not a list of [first, count]")
    return x


def run_input_roots(j):
    """Each member's `[port, tree]` at C's run input nodes' trees, or the refusal by name (`members.run_input_roots_at`;
    `OneStageZk.runInputRootsAt`), under the window's open root and the tied gates the case names; refused for another
    query or one that doesn't apply."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    desc = _obj(j.get("program"), "program")
    tied = [] if j.get("tied") is None else _ranges(j["tied"], "tied")
    window = j.get("window_root")
    if window is not None and not isinstance(window, str):
        raise D.Reject("window_root is not a string")
    query = _obj(j.get("query"), "query")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    if not P.is_template_query(query):
        raise D.Reject("not a template query")
    members = j.get("members")
    if not isinstance(members, list) or not all(
            isinstance(m, list) and len(m) == 2 and isinstance(m[0], str) and isinstance(m[1], list)
            and all(isinstance(u, list) and len(u) == 2 and all(_nat(x) for x in u) for u in m[1]) for m in members):
        raise D.Reject("members is not a list of [name, [[first, end], ...]]")
    program = PROGRAMS[sha]
    try:
        ranges = P._template(program, query, "the members' partition").ranges
    except (P.QueryRefused, ValueError) as e:
        raise D.Reject(str(e)) from None
    R = EG._Root(desc)
    by_port = {r.get("port"): r for r in j.get("roots") or () if isinstance(r, dict)}
    try:
        got = [[n, [[r["port"], r["tree"]] for r in M.run_input_roots_at(
            R, program, ranges, P.program_digest(program), window, [range(f, f + c) for f, c in tied], n,
            {"units": us}, [], by_port)]] for n, us in members]
    except M._Refused as e:
        return {"roots": None, "refused": str(e)}
    return {"roots": got, "refused": None}


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
        program = PN.population_program(PAIR, draw(st.integers(1, 6)))
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


@st.composite
def run_input_cases(draw) -> dict:
    """`fanout_cases`' programs and queries, the units of a template query that applies split among one to four members
    in runs, the record's roots at a random set of `input/n<w>` trees, each at C's leaf count and domain under the
    window's open root or not, the window's open root or none, and the first parameter's gates tied or none."""
    case = draw(fanout_cases())
    program = PROGRAMS[PO.program_sha512(case["program"])]
    try:
        n = sum(c for *_, c in PO.TemplatePartition(program, case["query"]).ranges)
    except PO.QueryInapplicable as e:
        # `fanout_cases` draws a query that doesn't apply on purpose: the case gets one member of one unit
        note(f"run_input_cases: the query doesn't apply: {e}")
        n = 1
    cuts = sorted(set(draw(st.lists(st.integers(1, n - 1), max_size=3)))) if n > 1 else []
    bounds = [0, *cuts, max(n, 1)]
    case["members"] = [[f"T/m{i}", [[lo, hi]]] for i, (lo, hi) in enumerate(zip(bounds, bounds[1:]))]
    nodes = EG._Root(case["program"]).nodes
    window = draw(st.sampled_from([WROOT, WROOT, None]))
    if window is not None:
        case["window_root"] = window
    prog = P.program_digest(program)

    def entry(w: int) -> dict:
        ln = draw(st.integers(1, 4))
        rows = nodes[w]["out_leaves"] * nodes[w]["members"] // ln
        honest = draw(st.booleans())
        return {"port": f"input/n{w}", "root": "aa" * 64, "leaves": rows if honest else rows + 1,
                "domain": RV.domain(prog, f"input/n{w}", ln, rows, 16, window=window or WROOT).domain_id.hex()
                if honest else "dd" * 64}
    case["roots"] = [entry(w) for w in sorted(set(draw(st.lists(st.integers(0, len(nodes) - 1), max_size=len(nodes)))))]
    if draw(st.booleans()):
        case["tied"] = [[g.start, len(g)] for g in G.held_gates(program, [program.fn.signature()[0][0][0]])]
    return case


PIECES = "spot-difftest"


def _audit_inputs(f, groups, most=1):
    return changed_audits(groups, most).map(f)


DECODE = D.Port(PIECES, "decode", decode, st.one_of(
    changed_audits(most=1).map(_case),
    st.builds(lambda a, i, x: FRAMING[i](_case(a), x), changed_audits(most=0), st.integers(0, len(FRAMING) - 1),
              st.integers(0, 63)), JSON), package=PACKAGE)
PARTITION = D.Port(PIECES, "partition", partition, st.one_of(
    _audit_inputs(lambda a: {"params": a["params"]}, [PARAMS]),
    st.builds(lambda a, q, sampling_design, t, pre: {"params": a["params"] | {
        "query": q, "law": sampling_design, "work_table": t,
        "partition": {"program": PO.program_sha512(a["params"]["program"]), "query": q} if pre else None}},
        changed_audits(most=0), st.sampled_from(QUERIES), st.one_of(sampling_designs(6), st.just({"k": 1, "law": "stratified"})),
        st.one_of(st.none(), st.dictionaries(st.sampled_from(PROGRAMS_TEMPLATES + ["Other_v1"]), WORK_ENTRY | BAD_WORK_ENTRY,
                                             max_size=3)), st.booleans()),
    st.builds(lambda a, pq, sampling_design, t, pre: {"params": a["params"] | {
        "program": pq[0], "query": pq[1], "law": sampling_design, "work_table": t,
        "partition": {"program": PO.program_sha512(pq[0]), "query": pq[1]} if pre else None}},
        changed_audits(most=0, setups=SETUPS), partitions(), sampling_designs(24),
        st.one_of(st.none(), st.fixed_dictionaries({t: WORK_ENTRY for t in WORK_IDS})), st.booleans()),
    JSON), package=PACKAGE)
COMMITTED_BITS = D.Port(PIECES, "committed_bits", committed_bits, st.one_of(
    st.builds(lambda m, x: {"program": json.loads(_calls(tuple(m))), "X": x},
              st.lists(st.sampled_from(range(len(CALLEES))), min_size=1, max_size=5), st.sampled_from([4, 8, 16, 32])),
    JSON), package=PACKAGE)
UNIT_GATES = D.Port(PIECES, "unit_gates", unit_gates, st.one_of(
    st.builds(lambda c: {"program": c[0], "query": c[1]}, partitions()), JSON), package=PACKAGE)
FANOUT = D.Port(PIECES, "fanout", fanout, st.one_of(fanout_cases(), JSON), package=PACKAGE)


def ungated_openers(j):
    """The definitions of a program that reach an opener outside a gated one (`docket.ungated_openers`,
    `Docket.ungatedOpeners`), refused for a malformed `gated`."""
    j = _json(j)
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    desc = _obj(j.get("program"), "program")
    gated = j.get("gated")
    if not isinstance(gated, list) or not all(
            isinstance(g, list) and len(g) == 2 and all(isinstance(x, str) for x in g) for g in gated):
        raise D.Reject("gated: each entry is [id, closure digest]")
    if PO.program_sha512(desc) not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {PO.program_sha512(desc)[:16]}")
    return {"templates": R.ungated_openers([tuple(g) for g in gated], desc)}


#: circuits of one call of a read each: the ungated read and the gated read at two bindings `docket.GATED` doesn't hold
OPENER_READS = ("EmbeddingRead_v2{V=2,H=1}", "EmbeddingRead_v4{V=2,H=1}", "EmbeddingRead_v4{V=4,H=1}")


@functools.cache
def _opener_desc(tid: str) -> dict:
    from verity.commitments.periods import interface as I
    program = PN.population_program(I.resolve(tid), 1)
    PROGRAMS[PO.program_sha512(desc := codec.encode_program(program))] = program
    return desc


def _gated_lists(desc: dict) -> list:
    """`docket.GATED`, none, each of its entries at another digest or under another id, and every read's own digest."""
    own = [[e["id"], codec.closure_digest(desc, k)] for k, e in desc["definitions"].items()
           if e["id"].startswith("EmbeddingRead")]
    held = [list(g) for g in R.GATED]
    return [held, [], [[i, "00" * 64] for i, _ in held], [["EmbeddingRead_v2", d] for _, d in held], own, held + own]


@st.composite
def opener_cases(draw) -> dict:
    """A circuit of `OPENER_READS`, or a population program of `PAIR`, which reaches no opener, under a list of
    `_gated_lists`."""
    if draw(st.integers(0, 3)) == 0:
        program = PN.population_program(PAIR, draw(st.integers(1, 3)))
        PROGRAMS[PO.program_sha512(desc := codec.encode_program(program))] = program
    else:
        desc = _opener_desc(draw(st.sampled_from(OPENER_READS)))
    return {"program": desc, "gated": draw(st.sampled_from(_gated_lists(desc)))}


UNGATED_OPENERS = D.Port(PIECES, "ungated_openers", ungated_openers, st.one_of(opener_cases(), JSON), package=PACKAGE)


def gated_list(j):
    """The verifier's own list of gated templates (`docket.GATED`, `Docket.GATED`), whatever the input."""
    return [list(g) for g in R.GATED]


GATED_LIST = D.Port(PIECES, "gated", gated_list, JSON, package=PACKAGE)
RUN_INPUT_ROOTS = D.Port(PIECES, "run_input_roots", run_input_roots, st.one_of(*[run_input_cases()] * 6, JSON),
                         package=PACKAGE)


@functools.cache
def _periods_tests():
    """The periods tests' module, for its run input programs (`node_slice`)."""
    spec = importlib.util.spec_from_file_location(
        "periods_test_interface_rows", Path(__file__).resolve().parents[7] / "tests/verity/commitments/periods/test_interface.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slice_rows(starts, n, ranges):
    """`node_slice`'s program, its ``x`` port's run input read (`interface.run_input_reads`), and the node's row each unit
    reads by that read, and the piece's input asking the Lean for the same rows."""
    TI = _periods_tests()
    prog = TI.node_slice(starts, n, ranges)
    (read,) = [r for r in TI.I.run_input_reads([(TI.COORD, len(starts))], prog, PO.template_instances_query([TI.COORD]), ())
               if r.port == "x"]
    words = dict(TI.I.layout(TI.COORD).rows)["x"]
    rows = [int(i) // words for i in read.source(np.arange(read.units))]
    case = {"program": json.loads(D.canonical(codec.encode_program(prog))), "indices": list(range(len(starts))),
            "port": "x", "value": read.param, "leaves": n // 64}
    return prog, read, words, rows, case


@pytest.mark.parametrize("starts, n, ranges", [([64, 192, 320], 512, (3,)), ([128, 320, 512, 704], 768, (1, 3))])
def test_a_run_input_slice_reads_the_rows_lean_derives(pieces, starts, n, ranges):
    """A port whose units read its run input node at a first row above zero and a stride above one (rows 1, 3, 5 of 8;
    rows 2, 5, 8, 11 of 12 across two ranges): the rows its `Read` names (``base`` and ``step``) are the rows
    `Flock.OneCircuit.pos` derives from C, at the tree the read names; a `Read` of another stride names other rows, which
    the verifier never opens, and the Lean refuses another tree or another row count."""
    _prog, read, words, rows, case = _slice_rows(starts, n, ranges)
    assert rows == [s // 64 for s in starts] and read.base > 0 and read.step[0] > words
    assert pieces.call("run_input_rows", case) == ("output", D.canonical({"rows": rows, "from": 0}))
    wrong = replace(read, step=(read.step[0] + words,))
    assert [int(i) // words for i in wrong.source(np.arange(wrong.units))] != rows
    assert pieces.call("run_input_rows", {**case, "value": "input/n1"})[0] == "reject"
    assert pieces.call("run_input_rows", {**case, "leaves": n // 64 + 1})[0] == "reject"


@pytest.mark.parametrize("starts, n, ranges, members", [
    ([64, 192, 320], 512, (3,), [["T/m0", [[0, 1]]], ["T/m1", [[1, 3]]]]),
    ([320], 1024, (1,), [["T/m0", [[0, 1]]]]),
    ([0], 1024, (1,), [["T/m0", [[0, 1]]]]),
])
def test_a_members_slice_of_a_run_input_node_opens_the_whole_nodes_tree_in_lean_and_python(pieces, starts, n, ranges,
                                                                                         members):
    """Members whose units read some rows of a run input node, from a first row above zero at a stride (rows 1, 3, 5 of
    8; row 5 of 16; row 0 of 16, the scan's state's shape): `members.run_input_roots_at` and `OneStageZk.runInputRootsAt`
    each hold the member's port at the record's one tree of the whole node, at C's row count, and a tree of only the rows
    read is refused by name by both (`run_tree_why`, leaf count)."""
    prog, read, words, rows, _case = _slice_rows(starts, n, ranges)
    PROGRAMS[PO.program_sha512(prog)] = prog
    assert read.param == "input/n0" and rows == [s // 64 for s in starts] and len(rows) < n // 64
    R, digest = EG._Root(json.loads(D.canonical(codec.encode_program(prog)))), P.program_digest(prog)
    query = PO.template_instances_query([_periods_tests().COORD])
    ranges_ = P._template(prog, query, "q").ranges
    nodes = {w: ln for _m, us in members for _p, w, ln in M.member_inputs(R, ranges_, "T", us)}
    assert nodes[0] == 64

    def tree(w: int, ln: int, rows_: int | None = None) -> dict:
        rs = R.nodes[w]["out_leaves"] * R.nodes[w]["members"] // ln
        wd = ln * (EG._bits(R.nodes[w]["ret"]) // R.nodes[w]["out_leaves"]) // 16
        k = rs if rows_ is None else rows_
        return {"port": f"input/n{w}", "root": f"{w:02x}" * 64, "leaves": k,
                "domain": RV.domain(digest, f"input/n{w}", wd, k, 16, window=WROOT).domain_id.hex()}
    assert tree(0, 64)["leaves"] == n // 64
    base = {"program": _json(codec.encode_program(prog)), "query": _json(query), "members": members,
            "roots": [tree(w, ln) for w, ln in sorted(nodes.items())], "window_root": WROOT}
    got = run_input_roots(base)
    assert got["refused"] is None and all(["x", "input/n0"] in r for _n, r in got["roots"]), got
    assert D.replay(RUN_INPUT_ROOTS, base, pieces) == ("output", D.canonical(got))
    part = {**base, "roots": [tree(0, 64, len(rows)), *base["roots"][1:]]}
    assert run_input_roots(part) == {"roots": None, "refused": M.run_tree_why(0, "leaf count")}
    assert D.replay(RUN_INPUT_ROOTS, part, pieces) == ("output", D.canonical(run_input_roots(part)))


def test_a_run_input_read_off_one_stride_is_refused_by_the_read_and_derived_by_the_lean(pieces):
    """Units at rows 1, 3, 4: `run_input_reads` refuses them (no one stride, so no one `Read`), and the Lean derives
    each unit's row from C all the same, so its verdict rests on no `Read`."""
    TI = _periods_tests()
    prog = TI.node_slice([64, 192, 256], 512, (3,))
    with pytest.raises(ValueError, match="not row 5: its units read the node's rows at one stride"):
        TI.I.run_input_reads([(TI.COORD, 3)], prog, PO.template_instances_query([TI.COORD]), ())
    case = {"program": json.loads(D.canonical(codec.encode_program(prog))), "indices": [0, 1, 2], "port": "x",
            "value": "input/n0", "leaves": 8}
    assert pieces.call("run_input_rows", case) == ("output", D.canonical({"rows": [1, 3, 4], "from": 0}))
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
    changed_audits([CERTIFIED], 1, 1).map(_REGISTRATION_OF),
    changed_audits([CERTIFIED, ENTRIES, REGISTERED_PARAMS, SUPPLIED], 2, 2, REGISTERED_SETUPS).map(_REGISTRATION_OF),
    st.builds(lambda a, r: {"params": a["params"], "record": r}, changed_audits(most=0), JSON)), package=PACKAGE)
CALLER = D.Port("spot-flock-caller", "caller", caller, REGISTRATION.inputs, package=CALLER_PACKAGE)


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
PRODUCED_REG, PRODUCED_SAMPLE, PRODUCED_OTHER = "c3" * 64, "d4" * 64, "e0" * 64


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
    return {"record": PD.record(PRODUCED_REG, PRODUCED_SAMPLE, es, ops), "registration": PRODUCED_REG, "draw": PRODUCED_SAMPLE,
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
    ("a registered value's domain", lambda c, x: _entry(c, x).update(domain=RV.domain(
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
CHECK_SAMPLE = D.Port(PIECES, "check_draw", check_draw, st.one_of(
    st.builds(lambda a, dn, other: {"draw": a["evidence"]["draw"], "law": other or a["record"]["law"],
                                    "population": a["record"]["population"] + dn},
              changed_audits([DRAW], 2), st.sampled_from([0, 0, 0, 1, -1]),
              st.sampled_from([None, None, None, {"k": 2, "law": "subset"}, {"law": "bernoulli", "p": "1/2"}])),
    st.builds(lambda d, n, sampling_design: {"draw": d, "law": sampling_design, "population": n}, JSON, st.integers(0, 6) | JSON,
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
SIMPLE_SAMPLING_DESIGNS = st.builds(lambda k: {"k": k, "law": "subset"}, st.integers(0, 40)) | st.builds(
    lambda num, den: {"law": "bernoulli", "p": f"{num}/{den}"}, st.integers(0, 12), st.integers(1, 12))
SAMPLING_DESIGN_CHANGES = [
    lambda sampling_design, x: sampling_design,
    lambda sampling_design, x: {**sampling_design, "k": sampling_design["k"] + x % 3 - 1} if "k" in sampling_design else {**sampling_design, "p": ["0/1", "1/1", "2/4", "3/2", "1/0", "1"][x % 6]},
    lambda sampling_design, x: {**sampling_design, "strata": sampling_design["strata"][:-1]},
    lambda sampling_design, x: {**sampling_design, "strata": [{**s, "k": s["k"] + 1} if i == x % len(sampling_design["strata"]) else s
                                      for i, s in enumerate(sampling_design["strata"])]},
    lambda sampling_design, x: {**sampling_design, "extra": 1},
    lambda sampling_design, x: {k: v for k, v in sampling_design.items() if k != "law"},
    lambda sampling_design, x: {**sampling_design, "law": ["subset", "bernoulli", "stratified", "work", [1]][x % 5]},
]


def _changed_sampling_design(sampling_design: dict, change, x: int) -> dict:
    try:
        return change(sampling_design, x)
    except (KeyError, ZeroDivisionError):
        return sampling_design


SAMPLE_INPUTS = st.one_of(
    st.builds(lambda a, src: {"law": a["record"]["law"], "population": a["record"]["population"], "source": src},
              changed_audits(most=0), SOURCE),
    st.builds(lambda a, f, x, dn, src: {"law": _changed_sampling_design(a["record"]["law"], f, x),
                                        "population": a["record"]["population"] + dn, "source": src},
              changed_audits(most=0), st.sampled_from(SAMPLING_DESIGN_CHANGES), st.integers(0, 63), st.sampled_from([0, 0, 0, 1, -1]),
              SOURCE),
    st.builds(lambda sampling_design, n, src: {"law": sampling_design, "population": n, "source": src}, SIMPLE_SAMPLING_DESIGNS, st.integers(0, 40), SOURCE),
    st.builds(lambda sampling_design, n, src: {"law": sampling_design, "population": n, "source": src}, SIMPLE_SAMPLING_DESIGNS | sampling_designs(6),
              st.integers(0, 40), SHORT_SOURCE),
    st.builds(lambda sampling_design, n, src: {"law": sampling_design, "population": n, "source": src}, JSON, st.integers(0, 6) | JSON,
              SHORT_SOURCE | st.sampled_from(["", "0", "zz", "AB", "ab cd"]) | JSON))
DRAW = D.Port(PIECES, "draw", draw, SAMPLE_INPUTS, package=PACKAGE)
RAN_OUT = D.Port(PIECES, "ran_out", ran_out, SAMPLE_INPUTS, package=PACKAGE)
#: Sampling design texts `randomness.parse_design` or `randomness.parse_rate` refuses or reads oddly: digits of other scripts, signs, spaces, a
#: second separator, leading zeros, a missing or zero argument.
ODD_SAMPLING_DESIGN_TEXTS = ["subset:", "subset", "", "subset:\u0663", "work:\u00b2", "stratified:\uff11", "bernoulli:1/\u0662",
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
#: A rate sampling design's text, `rate:NUM/DEN[:T]`, the tables written or not.
RATE_TEXTS = st.builds(lambda num, den, t: f"rate:{num}/{den}" + ("" if t is None else f":{t}"),
                       st.integers(0, 12), st.integers(0, 12), st.none() | st.integers(0, 6))
#: A stratified or work rate sampling design's text, `stratified-rate:NUM/DEN` or `work-rate:NUM/DEN`.
STRATA_RATE_TEXTS = st.builds(lambda kind, num, den: f"{kind}:{num}/{den}", st.sampled_from(["stratified-rate", "work-rate"]),
                              st.integers(0, 12), st.integers(0, 12))
SAMPLING_DESIGN_TEXTS = st.one_of(
    st.builds(lambda kind, k: f"{kind}:{k}", st.sampled_from(["subset", "stratified", "work"]), st.integers(0, 30)),
    st.builds(lambda num, den: f"bernoulli:{num}/{den}", st.integers(0, 12), st.integers(0, 12)),
    RATE_TEXTS, STRATA_RATE_TEXTS, st.sampled_from(ODD_SAMPLING_DESIGN_TEXTS))
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
    (_with("rule", None), SAMPLING_DESIGN_TEXTS.map(lambda t: {"law": t, "rule": "law"})),
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

#: What a params entry may add to record its template as a table read (`service.template_reads`): nothing, a read of
#: V rows of H words, and records that are not two positive counts.
_READ_RECORDS = [{}, {}, {"read": {"V": 8, "H": 2}}, {"read": {"V": 131072, "H": 576}}, {"read": {"V": 0, "H": 2}},
                 {"read": {"V": 8}}, {"read": {"V": True, "H": 2}}, {"read": None}, {"read": {"V": 8, "H": 2, "x": 1}},
                 {"read": {"V": "8", "H": 2}}, {"read": {"V": -1, "H": 2}}, {"read": [8, 2]}]

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
    "a pinned table": lambda p, q: ({**p, "tables": {"w": _PINNED}}, q),
    "a pinned table of other words": lambda p, q: ({**p, "tables": {"w": {**_PINNED, "word_bits": 8}}}, q),
    "a pinned table of 0-bit words": lambda p, q: ({**p, "tables": {"w": {**_PINNED, "word_bits": 0}}}, q),
    "a pinned table under another parameter": lambda p, q: ({**p, "tables": {"v": _PINNED}}, q),
    "a pinned table without its domain": lambda p, q: ({**p, "tables": {"w": {
        k: v for k, v in _PINNED.items() if k != "domain"}}}, q),
    "a pinned table with another key": lambda p, q: ({**p, "tables": {"w": {**_PINNED, "rows": 8}}}, q),
    "a pinned table's root uppercase": lambda p, q: ({**p, "tables": {"w": {**_PINNED, "root": "AA" * 64}}}, q),
    "a pinned table's root short": lambda p, q: ({**p, "tables": {"w": {**_PINNED, "root": "aa" * 32}}}, q),
    "a pinned table of no leaves": lambda p, q: ({**p, "tables": {"w": {**_PINNED, "leaves": 0}}}, q),
    "a pinned table's words a boolean": lambda p, q: ({**p, "tables": {"w": {**_PINNED, "words": True}}}, q),
    "a pinned table a list": lambda p, q: ({**p, "tables": {"w": [_PINNED]}}, q),
    "pinned tables a list": lambda p, q: ({**p, "tables": [_PINNED]}, q),
    "pinned tables empty": lambda p, q: ({**p, "tables": {}}, q),
}


#: A registered table's entry the auditor pins (`service._table_entry`): the tables' registration's for parameter ``w``.
_PINNED = {"value": "w", "root": "ab" * 64, "leaves": 8, "words": 4, "domain": "cd" * 64}


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
    recorded = draw(st.lists(st.sampled_from(_READ_RECORDS), min_size=len(ids), max_size=len(ids)))
    proofs = {"templates": [{"template": t, "circuit": c, "header": h, "registered": r, **x}
                            for t, c, h, r, x in zip(ids, circuits, headers, reads, recorded)]}
    if draw(st.booleans()):
        proofs["tables"] = {"w": _PINNED, **({"x": {**_PINNED, "value": "x", "root": "ef" * 64}} if draw(st.booleans())
                                              else {})}
    if draw(st.booleans()):
        proofs, query = _PARAMS_CHANGES[draw(st.sampled_from(sorted(_PARAMS_CHANGES)))](proofs, query)
    return {"query": query} if proofs is None else {"proofs": proofs, "query": query}


TEMPLATE_CIRCUITS = D.Port(PIECES, "template_circuits", template_circuits,
                           st.one_of(template_circuit_cases(), template_circuit_cases(), JSON), package=PACKAGE)
SELECT = D.Port(PIECES, "select", select, st.one_of(selections(), selections(), changed_selections(1),
                                                     changed_selections(2), JSON), package=PACKAGE)
DESIGN = D.Port(PIECES, "design", design, st.one_of(selections(), selections(), changed_selections(1), JSON),
                package=PACKAGE)
SLOTS = D.Port(PIECES, "slots", slots, st.one_of(selections(), selections(), changed_selections(1), JSON), package=PACKAGE)
POP_ORDER = D.Port(PIECES, "pop_order", pop_order, st.one_of(*[st.builds(
    lambda part, dn: {"population": max(0, part[2] + dn), "program": part[0], "query": part[1]}, partitions(),
    st.sampled_from([0, 0, 0, -1, 1]))] * 3, JSON), package=PACKAGE)


def _units_case(units: list, how: int, x: int) -> dict:
    """``units`` with their own units ascending, or those shifted, reversed, short of one, or with ``x`` added."""
    want = [u for lo, hi in sorted(map(tuple, units)) for u in range(lo, hi)]
    at = x % (len(want) + 1)
    return {"indices": [want, [u + 1 for u in want], want[::-1], want[:-1], [*want, x], [*want[:at], x, *want[at:]]][how],
            "units": units}


PUBLIC_UNITS = D.Port(PIECES, "public_units", public_units, st.one_of(
    *[st.builds(_units_case, st.lists(st.builds(lambda lo, n: [lo, lo + n], st.integers(0, 12), st.integers(1, 5)),
                                      max_size=3), st.integers(0, 5), st.integers(0, 20))] * 2,
    st.builds(lambda u, i: {"indices": i, "units": u}, st.lists(st.lists(st.integers(-1, 6), max_size=3), max_size=3),
              st.lists(st.integers(-1, 6), max_size=6)),
    JSON), package=PACKAGE)


def _cut(population: int, cuts: list[int], owners: list[int]) -> list[list[list[int]]]:
    """`[0, population)` cut at `cuts` (each modulo `population + 1`), its `i`-th range `owners[i % len(owners)]`'s: each
    owner's ranges, ascending, the owners in the order of their first range."""
    edges = sorted({0, population, *(c % (population + 1) for c in cuts)})
    out: dict[int, list] = {}
    for i, r in enumerate(zip(edges, edges[1:])):
        out.setdefault(owners[i % len(owners)], []).append(list(r))
    return list(out.values())


def _cut_members(population: int, members: tuple, gs: list[int], bad: int, k: int) -> list[dict]:
    """Members as a piece is handed them (`LinkPoints.membersOfJson`): `[0, population)` cut by `members` (`_cut`), each
    `gs` units a proof block; `bad` from 1 to 4 breaks the `k`-th: a unit past the population, two of its ranges
    meeting, no unit a proof block, or no range."""
    ms = [{"units": us, "vus_per_block": gs[i % len(gs)]} for i, us in enumerate(_cut(population, *members))]
    if 1 <= bad <= 4 and ms:
        m = ms[k % len(ms)]
        m |= [{"units": [*m["units"], [population, population + 1]]}, {"units": [*m["units"], m["units"][0]]},
              {"vus_per_block": 0}, {"units": []}][bad - 1]
    return ms


def _link_case(kind: str, k: int, population: int, strata: tuple, works: list[int], members: tuple, gs: list[int],
               bad: int) -> dict:
    """A `link_points` input: a `kind` sampling design of `k` over `population` units, a stratified or work sampling design's strata cut by
    `strata` (`_cut`) with each stratum's k its rule's (work `works`, floor 1), and members cut by `members`, each `gs`
    units a proof block, one of them broken where `bad` says (`_cut_members`)."""
    n_k = k % population + 1 if population else k
    if kind == "subset":
        sampling_design = {"k": k % (population + 2), "law": kind}
    elif kind == "bernoulli":
        sampling_design = {"law": kind, "p": ["0/1", "1/2", "1/1", "2/2", "3/7", "0/3"][k % 6]}
    elif kind == "stratified":
        sampling_design = {"k": n_k, "law": kind, "strata": [{"k": DR.stratum_k(n_k, LP.size(rs), population), "template": f"t{i}",
                                                  "units": rs} for i, rs in enumerate(_cut(population, *strata))]}
    else:
        ss = [{"floor": 1, "template": f"t{i}", "units": rs, "work": works[i % len(works)]}
              for i, rs in enumerate(_cut(population, *strata))]
        total = sum(t["work"] * LP.size(t["units"]) for t in ss)
        sampling_design = {"k": n_k, "law": kind, "strata": [{**t, "k": DR.work_k(n_k, t["work"], LP.size(t["units"]), total, 1)}
                                                 for t in ss] if total else ss}
    return {"law": sampling_design, "members": _cut_members(population, members, gs, bad, k), "population": population}


CUTS = st.tuples(st.lists(st.integers(0, 60), max_size=6), st.lists(st.integers(0, 3), min_size=1, max_size=4))
LINK_POINTS = D.Port(PIECES, "link_points", link_points, st.one_of(
    *[st.builds(_link_case, st.sampled_from(["subset", "bernoulli", "stratified", "work"]), st.integers(0, 60),
                st.integers(0, 40), CUTS, st.lists(st.integers(0, 5), min_size=1, max_size=3), CUTS,
                st.lists(st.integers(1, 5), min_size=1, max_size=3), st.integers(0, 9))] * 3,
    st.builds(lambda sampling_design, n, ms: {"law": sampling_design, "members": ms, "population": n}, JSON, st.integers(-1, 9), JSON),
    JSON), package=PACKAGE)

#: Windows a member of which can hold nine sampled proof blocks or eight, which open 4 link points and 3 at one unit a
#: proof block: one template's 24 proof blocks, and PAIR's two with TRIPLE's eighteen, a stratum each.
LINK_SETUPS = [Setup(PN.population_program(PAIR, 24), P.template_instance_query(PAIR.id)),
               Setup(PN.mixed_population_program([(PAIR, 2), (TRIPLE, 18)]), P.template_instances_query([PAIR.id, TRIPLE.id]))]
PROGRAMS.update({s.sha512: s.program for s in LINK_SETUPS})


def _docket_members(a: dict, cut: int, gs: list[int], bad: int) -> dict:
    """The `registration` piece's input of the audit `a` with the per-member form's members: the record's population
    cut in two at `cut`, or whole where it holds under two units, each `gs` units a proof block, one broken where
    `bad` says (`_cut_members`)."""
    n = a["record"].get("population") if isinstance(a["record"], dict) else None
    n = n if _nat(n) else 0
    return {**_REGISTRATION_OF(a),
            "members": _cut_members(n, ([cut % (n - 1) + 1] if n > 1 else [], [0, 1]), gs, bad, cut)}


#: `registration` inputs with members: audits of `LINK_SETUPS`' windows and of the small ones, with at most one change.
REGISTRATIONS_WITH_MEMBERS = st.one_of(
    *[st.builds(_docket_members, changed_audits(most=1, setups=setups), st.integers(0, 60),
                st.lists(st.sampled_from([1, 1, 2]), min_size=1, max_size=2), st.integers(0, 20))
      for setups in (LINK_SETUPS, LINK_SETUPS, LINK_SETUPS, SETUPS)],
    st.builds(lambda a, ms: {**_REGISTRATION_OF(a), "members": ms}, changed_audits(most=0), JSON))

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
#: Each case names `BLOCKED`'s C and partition, so the verifier's circuit file is the derived one (`restrict`).
TABLES_CASES = st.one_of(st.one_of(
    st.builds(lambda ds, lk: {"digests": ds, "lookups": lk}, st.lists(TABLE_DIGEST, max_size=5), LOOKUPS),
    st.builds(lambda ds, lk: {"digests": ds, "lookups": lk}, st.lists(TABLE_DIGEST, max_size=5), LOOKUPS),
    st.lists(TABLE_DIGEST, max_size=5).map(lambda ds: {"digests": ds}),
    st.lists(st.one_of(TABLE_DIGEST, st.sampled_from(["0", "AB", "0g", PF.MUFU_SHA512["rcp"].upper()]), st.integers(0, 3)),
             min_size=1, max_size=4).map(lambda ds: {"digests": ds}),
).map(lambda j: j | {"program": BLOCKED.desc, "partition": BLOCKED.partition}), JSON)
TABLES_HELD = D.Port(PIECES, "tables_held", tables_held, TABLES_CASES, package=PACKAGE)
OWN_TABLES = D.Port(PIECES, "own_tables", own_tables, TABLES_CASES, package=PACKAGE)

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


def public_ports(j):
    j = _json(j)
    rec = j.get("record") if isinstance(j, dict) else None
    roots = rec.get("roots") if isinstance(rec, dict) else None
    if not isinstance(roots, list):
        raise D.Reject("the record's roots are not a list")
    if not all(isinstance(r, dict) and isinstance(r.get("port"), str) for r in roots):
        raise D.Reject("a registered root is not an object with a string port")
    try:
        return _json(R.public_ports(rec))
    except ValueError as e:
        raise D.Reject(str(e)) from None


def _roots(ports: list[str], x: int) -> list[dict]:
    """A docket's roots at `ports`, every fifth with no domain or leaves."""
    return [{"domain": f"d{i}", "leaves": i + x % 3, "port": p, "root": f"r{i}"} if (x + i) % 5 else
            {"port": p, "root": f"r{i}"} for i, p in enumerate(ports)]


#: Port names a member's roots carry (one holding a `/`), and members' names (one template's two chunks, another's, one
#: with no `/`)
OWN_PORTS = st.sampled_from(["x", "sx", "o0", "w/row"])
MEMBER_NAMES = st.sampled_from(["T{K=1}/c0", "T{K=1}/c1", "A/c0", "T"])
PUBLIC_PORTS = D.Port(PIECES, "public_ports", public_ports, st.one_of(
    st.builds(lambda ps, x: {"record": {"roots": _roots(ps, x)}}, st.lists(OWN_PORTS, max_size=4), st.integers(0, 63)),
    st.builds(lambda m, ps, x: {"record": {"roots": _roots([f"{m}/{p}" for p in ps], x)}}, MEMBER_NAMES,
              st.lists(OWN_PORTS, min_size=1, max_size=4), st.integers(0, 63)),
    st.builds(lambda ps, x: {"record": {"roots": _roots(ps, x)}}, st.lists(st.one_of(
        OWN_PORTS, st.builds("{}/{}".format, MEMBER_NAMES, OWN_PORTS),
        st.sampled_from(["row-table", "row-table/s0/t", "input/n3", "input/n03", "input/n3/x"]),
        st.builds("{}/{}".format, st.sampled_from(["T{K=1}", "A", "B"]), st.sampled_from(["o0", "x"]))),
        min_size=1, max_size=4), st.integers(0, 63)),
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
    """`audit_record`'s arguments from an audit: the verifier's derived sampling design (or the record's, when the parameters don't
    derive one), and the evidence, with δ."""
    p, e = a["params"], a["evidence"]
    try:
        sampling_design = DR.derive(PROGRAMS[PO.program_sha512(p["program"])], p["query"], p["law"], p["work_table"])
    except Exception:  # noqa: BLE001 - a changed parameter the verifier would refuse: the record's own sampling design stands in
        sampling_design = a["record"].get("law") if isinstance(a["record"], dict) else None
    out = {"delta": delta, "law": sampling_design, "registration": a["record"]}
    return out | {k: e.get(k) for k in ("draw", "instances", "proved_units", "receipt", "served", "session_roots", "sessions",
                                         "statement", "verdicts")}


def _own_reads_case(a: dict, x: int) -> dict:
    """`_record_case` with its first verdict also reporting a read of one of the docket's ports, at that port's
    root or another."""
    c = _record_case(a, "1/4")
    reg, vs = c["registration"], c["verdicts"]
    if isinstance(reg, dict) and isinstance(reg.get("roots"), list) and reg["roots"] and isinstance(vs, dict) and vs:
        r, name = reg["roots"][x % len(reg["roots"])], sorted(vs)[0]
        if isinstance(r, dict) and isinstance(vs[name], dict):
            got = vs[name].get("registered") if isinstance(vs[name].get("registered"), dict) else {}
            read = {"value": r.get("port"), "root": r.get("root") if x % 3 else "cd" * 64}
            c["verdicts"] = {**vs, name: {**vs[name], "registered": {**got, "own": read}}}
    return c


def _certified_case(a: dict, x: int) -> dict:
    """`_record_case` with `certified` (or one not a list of strings): names among the docket's registered
    values, the values its verdicts report reads of and one neither names, some twice."""
    c = _record_case(a, "1/4")
    reg, vs = c["registration"], c["verdicts"]
    listed = reg.get("registered") if isinstance(reg, dict) and isinstance(reg.get("registered"), list) else []
    read = [e for v in (vs.values() if isinstance(vs, dict) else ()) if isinstance(v, dict)
            and isinstance(v.get("registered"), dict) for e in v["registered"].values()]
    names = ["unlisted"] + [e["value"] for e in listed + read if isinstance(e, dict) and isinstance(e.get("value"), str)]
    pick = [names[(x // 9 + i) % len(names)] for i in range(x % 4)]
    return {**c, "certified": [pick, pick, pick + pick[:1], pick, pick, "unlisted", [1], None, {"unlisted": 1}][x % 9]}


#: δ as the exact ratio of a double (`str(Fraction(delta))`), as the switch passes it.
EXACT_DELTA = st.floats(2 ** -40, 1 - 2 ** -40).map(lambda f: str(Fraction(f))) | st.sampled_from(
    [str(Fraction(f)) for f in (0.5, 0.25, 0.1, 0.05, 0.01, 1e-6, 2 ** -20, 0.999)])
RECORD_PORT = D.Port(PIECES, "record", record, st.one_of(
    st.builds(_record_case, changed_audits(most=0), EXACT_DELTA),
    st.builds(_own_reads_case, changed_audits(most=0), st.integers(0, 63)),
    st.builds(_own_reads_case, changed_audits([REGISTERED_EVIDENCE], 1, 0, REGISTERED_SETUPS), st.integers(0, 63)),
    st.builds(_certified_case, changed_audits(most=0), st.integers(0, 63)),
    st.builds(_certified_case, changed_audits([REGISTERED_EVIDENCE], 1, 0, REGISTERED_SETUPS), st.integers(0, 63)),
    st.builds(_record_case, changed_audits([EVIDENCE, SAMPLE_CHANGES, VERDICTS_OF, REGISTERED_EVIDENCE], 1, 1), EXACT_DELTA),
    st.builds(_record_case, changed_audits([RECORD, ENTRIES, REGISTERED_EVIDENCE], 1, 1, REGISTERED_SETUPS), EXACT_DELTA),
    st.builds(_record_case, changed_audits(most=2, least=2), EXACT_DELTA),
    st.builds(_record_case, changed_audits(most=0), st.sampled_from(["0", "1", "3/2", "a", "1/0", "2/4", 5, None])),
    st.builds(lambda a, k: {kk: v for kk, v in _record_case(a, "1/4").items() if kk != k}, changed_audits(most=0),
              st.sampled_from(sorted(RECORD_INPUT_KEYS))),
    JSON), package=PACKAGE)

RUN_SESSIONS = ["5e" * 64, "a1" * 64]


def _sampling_design_text(sampling_design, n) -> str | None:
    """The rule's sampling design text naming an agreed sampling design over a record of `n` proof blocks: a rate's own, `subset:K` as the rate
    `rate:K/N`, which names `subset:K` again over N, and any other sampling design object's text, which a stream refuses."""
    if isinstance(sampling_design, str):
        return sampling_design
    if not isinstance(sampling_design, dict):
        return None
    if sampling_design.get("law") == "subset" and type(sampling_design.get("k")) is int and type(n) is int and 1 <= sampling_design["k"] <= n:
        return f"rate:{sampling_design['k']}/{n}"
    return f"bernoulli:{sampling_design.get('p')}" if sampling_design.get("law") == "bernoulli" else f"{sampling_design.get('law')}:{sampling_design.get('k')}"


def _run_case(a: dict, source: str, sessions: int, outputs: list) -> dict:
    """`run`'s input from an audit: the rule naming the verifier's sampling design (`_sampling_design_text`), every item
    `_public_items` declares, the challenger's name and place, the coins, and the evidence's statement and verdicts, each
    verdict of record about the delivered sessions and reporting the public outputs."""
    p, e = a["params"], a["evidence"]
    text = _sampling_design_text(p.get("law"), a["record"].get("population") if isinstance(a["record"], dict) else None)
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


def _from_challenger(c: dict, rng: random.Random, **shape: bool) -> tuple[Drawn, dict]:
    """Run `c` sampling from the auditor's challenger, as a `Stream` over a window runs it: its coins a ledger session's
    challenger opened after the docket's digest and read by the selection rule's sample (`Drawn`), the source the
    bytes read, the evidence the session's, the family the session's `draw`, the window root the ledger's open's, and
    the key and main's history the session's own (`Drawn.held`).
    `shape`: `Drawn`'s `before` or `later`, a session whose family holds another root."""
    rec, p, rule = c["record"], _params_of(c), _rule(c["rule"])
    sampling_design = DR.derive(p["program"], p["query"],
                    service.selection_sampling_design(rule, rec["population"], service.selection_sizes(rule, p["program"], p["query"])),
                    service.selection_table(rule, p["program"], p["query"], p["work_table"]))

    def reads(coins):
        got = bytearray()

        def source(n):
            b = coins(n)
            got.extend(b)
            return b
        service.select_draw(rule, sampling_design, rec["population"], source)
        return bytes(got)
    d = Drawn(rng, after=bytes.fromhex(DD.digest(rec)), reads=reads, **shape)
    return d, {**c, "params": {**c["params"], "window_root": d.ledger.window["root"]}, "source": d.read.hex(),
               "coins": d.evidence(d.segment.to_json()), "family": "draw", **d.held()}


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
    which is `rate:K/N` again, tables K doesn't divide, which rounds K up past the record's sampling design (R1) or past N, or N + 1
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


#: One change each to the run's own inputs: the rule (a sampling design text of any form, its subset sampling design as a rate sampling design, a rule the
#: stream doesn't run, or junk), the declared items and values, the challenger's name and place, the closure map, the
#: coins, where they came from and their evidence, the sessions, a verdict, the outputs, the statement, δ, the proved units and instances, the served units, the
#: session roots, or a field missing.
RUN_CHANGES = [
    (_with("rule", None), st.one_of(SAMPLING_DESIGN_TEXTS.map(lambda t: {"law": t, "rule": "law"}), st.just({"rule": "all"}),
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

ID8 = bind(CompositeDefinition("CoreSpotId8", 0, (), lambda S: ((("y", Value(8)),), Value(8)),
                               lambda B, S, y: B.call(XOR8, y, y), register=False))
#: Replay blocks to derive round 2's links from (`replay_links.links_of`): `BLOCK`; PAIR over the first parameter then
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
#: Replay blocks to derive round 2's wires from (Lean's `Pieces.replayWires`) beside `REPLAYS`' (the chain's, ID8 reading
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
#: A token select's pieces made small, each token two rows of two logits. Gathered: DUO is a chunk, its two outputs a
#: best and an index, and MERGE the combine, reading logits[0], every chunk's best and every chunk's index. By units:
#: UNIT is a unit of two chunks, its outputs its chunks' bests, their indices and its first logit, and JOIN the combine,
#: each parameter one whole output of one unit, two of them one output port of UNIT at its token's two units. By chunk
#: position, each token three rows: SWAP and KEEP are its first and second chunks, each a Definition of its own with
#: one output, and LAST the combine, over the last chunk's logits and each other chunk's one output, whole.
MERGE = bind(CompositeDefinition(
    "CoreSpotMerge", 0, (), lambda S: ((("x0", Value(8)), ("best", Array(2, Value(8))), ("idx", Array(2, Value(8)))), Value(8)),
    lambda B, S, x0, best, idx: B.call(XOR8, B.call(XOR8, x0, B.call(XOR8, best[0], best[1])), B.call(XOR8, idx[0], idx[1])),
    register=False))
UNIT = bind(CompositeDefinition(
    "CoreSpotUnit", 0, (), lambda S: ((("x", Array(2, Value(8))),), Tuple(Array(2, Value(8)), Array(2, Value(8)), Value(8))),
    lambda B, S, x: tuple_of(array_of([B.call(ID8, x[0]), B.call(ID8, x[1])]),
                             array_of([B.call(XOR8, x[0], x[1]), B.call(XOR8, x[1], x[0])]), x[0]), register=False))
JOIN = bind(CompositeDefinition(
    "CoreSpotJoin", 0, (), lambda S: ((("x0", Value(8)), *((f"{p}{j}", Array(2, Value(8))) for p in ("best", "idx")
                                                            for j in range(2))), Value(8)),
    lambda B, S, x0, *ps: functools.reduce(lambda a, p: B.call(XOR8, a, B.call(XOR8, p[0], p[1])), ps, x0),
    register=False))
KEEP = bind(CompositeDefinition("CoreSpotKeep", 0, (), lambda S: ((("x", Array(2, Value(8))),), Array(2, Value(8))),
                                lambda B, S, x: array_of([B.call(ID8, x[0]), B.call(ID8, x[1])]), register=False))
LAST = bind(CompositeDefinition(
    "CoreSpotLast", 0, (), lambda S: ((("x", Array(2, Value(8))), ("r0", Array(2, Value(8))), ("r1", Array(2, Value(8)))),
                                      Value(8)),
    lambda B, S, x, *rs: functools.reduce(lambda a, p: B.call(XOR8, a, B.call(XOR8, p[0], p[1])), rs,
                                          B.call(XOR8, x[0], x[1])), register=False))


def _select(B, x):
    """`TokenSelect_v4`'s shape: a token's chunks, then its combine gathering one output of each."""
    a, b = B.call(DUO, x[0]), B.call(DUO, x[1])
    return B.call(MERGE, x[0][0], array_of([a[0], b[0]]), array_of([a[1], b[1]]))


def _select_units(B, x):
    """A token's units, then its combine reading each output of each whole."""
    a, b = B.call(UNIT, x[0]), B.call(UNIT, x[1])
    return B.call(JOIN, a[2], a[0], b[0], a[1], b[1])


def _select_places(B, x):
    """`GreedySelect_v1`'s shape: a token's chunks, a Definition a chunk position, then its combine over the last
    chunk's logits and each other chunk's one output, whole."""
    return B.call(LAST, x[2], B.call(SWAP, x[0]), B.call(KEEP, x[1]))


#: Replay blocks of templates with several callers, to derive round 2's rows from (`replay_links.replay_rows`): PAIR
#: over each of two rows, in order and the other way round, its outputs returned in order; that, and ID8 over the
#: second's output, returned beside them; ID8 over the second output of DUO, a template of two; and, deriving none, PAIR
#: over two parameters, PAIR over a slice not one of its rows, an ID8 batch mapped over a PAIR batch's outputs, PAIR
#: over the parameter and over SWAP's output (a port one caller links and another wires), and PAIR over two ID8 outputs
#: (a port gathered from two writers); then DUO over each of two rows, its first outputs returned, its second, both as
#: two fields, and, deriving none, the first's first beside the second's second; then a token select inlined over two
#: tokens, gathered (`_select`: no wires, and its logits read at two row widths), by units (`_select_units`) and by
#: chunk position (`_select_places`).
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
     lambda B, S, x: array_of([B.call(DUO, x[0])[0], B.call(DUO, x[1])[1]])),
    ("Select", lambda S: ((("x", Array(2, Array(2, Array(2, Value(8))))),), Array(2, Value(8))),
     lambda B, S, x: array_of([_select(B, x[0]), _select(B, x[1])])),
    ("SelectUnits", lambda S: ((("x", Array(2, Array(2, Array(2, Value(8))))),), Array(2, Value(8))),
     lambda B, S, x: array_of([_select_units(B, x[0]), _select_units(B, x[1])])),
    ("SelectPlaces", lambda S: ((("x", Array(2, Array(3, Array(2, Value(8))))),), Array(2, Value(8))),
     lambda B, S, x: array_of([_select_places(B, x[0]), _select_places(B, x[1])])))]
ID18 = PrimitiveDefinition("CoreSpotId18", 1, (("a", Value(8)),), Value(18), lambda a: a, register=False)
WIDE = bind(CompositeDefinition("CoreSpotWide", 0, (), lambda S: ((("x", Value(8)),), Value(18)),
                                lambda B, S, x: B.call(ID18, x), register=False))
PACK = bind(CompositeDefinition("CoreSpotPack", 0, (), lambda S: ((("x", Array(8, Value(8))),), Array(8, Value(18))),
                                lambda B, S, x: array_of([B.call(ID18, x[i]) for i in range(8)]), register=False))
#: Replay blocks whose units return fields wider than a word, to count what a replay block returns
#: (Lean's `Pieces.replayReturned`): four units of one 18-bit value each, and two of eight such values as one field.
WIDE_REPLAYS = [bind(CompositeDefinition(f"CoreSpotReplay{name}", 0, (), sig, body, register=False))
                for name, sig, body in (
    ("Wides", lambda S: ((("x", Array(4, Value(8))),), Array(4, Value(18))),
     lambda B, S, x: B.batch(WIDE, x, axes=(0,))),
    ("Packs", lambda S: ((("x", Array(2, Array(8, Value(8)))),), Array(2, Array(8, Value(18)))),
     lambda B, S, x: B.batch(PACK, x, axes=(0,))))]
REPLAY_PROGRAMS = [(PN.population_program(r, 2), codec.descriptor_id(r)) for r in REPLAYS] + [
    (PN.mixed_population_program([(BLOCK, 2), (PAIR, 3)]), codec.descriptor_id(BLOCK))] + [
    (PN.population_program(r, 2), codec.descriptor_id(r)) for r in WIRE_REPLAYS + MULTI_REPLAYS + WIDE_REPLAYS]
PROGRAMS.update({PO.program_sha512(p): p for p, _ in REPLAY_PROGRAMS})


def replay_links(j):
    """`Pieces.replayLinks`: the design's links (`replay_links.replay_links`) from the replay block of descriptor id
    `replay` in `program`, each value the port of the `boundary` record; null for none."""
    desc = _obj(j.get("program") if isinstance(j, dict) else None, "program")
    if not isinstance(j.get("replay"), str):
        raise D.Reject("replay is not a string")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    links = RL.replay_links(PROGRAMS[sha], j["replay"], j.get("boundary"))
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


def replay_rows(j):
    """`Pieces.replayRows`: the rows the design places each linked and wired unit at (`replay_links.replay_rows`), from
    the replay block of descriptor id `replay` in `program`; none for none."""
    desc = _obj(j.get("program") if isinstance(j, dict) else None, "program")
    if not isinstance(j.get("replay"), str):
        raise D.Reject("replay is not a string")
    if not desc:
        raise D.Reject("not a program descriptor")
    sha = PO.program_sha512(desc)
    if sha not in PROGRAMS:
        raise RuntimeError(f"the strategy drew a descriptor this file has no program for: {sha[:16]}")
    return [r.to_json() for r in RL.replay_rows(PROGRAMS[sha], j["replay"])]


#: The replay_rows piece's input: replay_links' (`replay_link_inputs`), whose boundary record it doesn't read.
REPLAY_ROWS = D.Port(PIECES, "replay_rows", replay_rows,
                     st.integers(0, 3).flatmap(lambda i: JSON if i == 3 else replay_link_inputs()), package=PACKAGE)
RUN = D.Port(PIECES, "run", run, st.one_of(
    runs(), runs(), runs(changes=1), runs(changes=1), runs(changes=2), st.builds(_as_rate, runs(), st.integers(0, 63)),
    runs([RECORD], 1), runs([PARAMS], 1), runs([ENTRIES, REGISTERED_PARAMS], 1), runs([EVIDENCE, VERDICTS_OF], 1), JSON),
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
               "select": {"checks", "draw", "proved", "at"}, "deliver": {"now"},
               "deliver_session": {"now"}, "outcome": {"now", "verdicts", "accepted"}}
#: Fields an op may carry out of form, with what the Lean errs or decides on.
MALFORMED = st.sampled_from(["1", True, None, -1, "x", [1], {"a": 1}])


def _one_in(draw, n: int) -> bool:
    """One time in `n`, evenly: `st.integers` samples its bounds more often."""
    return draw(st.sampled_from(range(n))) == 0


def _aim(draw, name: str, target: dict, entries: list, rounds: int) -> list | None:
    """The entries `name` on `target` reads made ready for it: sampled but not yet for `select`, open, under no session
    but `deliver_session`'s (and then a session of its own over consecutive rounds, all of them sampled and open),
    and delivered or not for `outcome`. A session's covered samples, or None."""
    def open_(e: dict, drawn: bool = True) -> None:
        e.update(draw=f"d{e['order']}" if drawn else None, proved=draw(st.integers(1, 3)) if drawn else 0,
                 drawn_at=draw(READINGS) if drawn else None, final=False, delivered=False, session=None)
    open_(target, name != "select")
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
    """A stream's params, a state of up to 6 dockets holding most of their entries (sampled or not, delivered, final,
    or under a session of consecutive rounds), and one op on it: mostly naming a held entry by its own
    tokens, or another's, or one not received; each field it reads present or not yet (`unchecked`), and sometimes one
    field out of form; sometimes the stream halted. Half the time aimed at the op: the stream open and every entry held,
    with the target's made ready for the op (`_aim`)."""
    name = draw(st.sampled_from([*service._OPS, *service._OPS[3:], "other"]))
    aim = draw(st.booleans())
    rounds = draw(st.integers(1, service.ROUNDS))
    budget = draw(st.integers(1, 4))
    params = {"budget": budget, "rounds": rounds, "registration_deadline": draw(st.sampled_from([None, 5, 2, 10])),
              "proof_deadline": draw(st.sampled_from([3, 7, 1])), "stream": STREAM_DIGEST}
    size = draw(st.integers(1 if aim else 0, 6))
    entries = []
    for o in range(size):
        if not aim and _one_in(draw, 5):
            continue
        drawn = not _one_in(draw, 4)
        e = {"order": o, "registration": f"r{o}", "draw": f"d{o}" if drawn else None,
             "proved": draw(st.integers(0, 3)) if drawn else 0, "drawn_at": draw(READINGS) if drawn else None,
             "delivered": drawn and _one_in(draw, 4), "final": drawn and _one_in(draw, 4), "session": None}
        if drawn and draw(st.booleans()):
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
        "session": {"draws": draws},
        "deliver": {"order": o, "registration": tokens[0], "draw": tokens[1], "sessions": sessions, "now": draw(READINGS)},
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
callers = D.server_fixture(CALLER)


PIECE_RUNS = [(DECODE, 300), (PARTITION, 500), (COMMITTED_BITS, 300), (UNIT_GATES, 300), (FANOUT, 400), (UNGATED_OPENERS, 60),
              (RUN_INPUT_ROOTS, 400),
              (REGISTRATION, 700), (REGISTERED_ENTRIES, 300), (DOMAIN, 300),
              (PRODUCED, 500), (KNOWN_PORT, 300),
              (CHECK_SAMPLE, 500), (DRAW, 500), (RAN_OUT, 300), (SELECT, 600), (DESIGN, 300), (SLOTS, 400),
              (FILE_POPULATION, 400), (TABLES_HELD, 300), (OWN_TABLES, 300), (CHECK_PROVED, 300), (BOUND, 500),
              (BERNOULLI_BOUND, 500), (HIDDEN_OFFSETS, 300), (HIDDEN_LOCATE, 400), (HIDDEN_WITNESS, 400), (VERDICT, 300),
              (MATCHES_PUBLIC, 400), (PUBLIC_PORTS, 300), (SESSION_REGISTERED, 400), (RECORD_PORT, 600), (CHECK_PUBLIC, 500),
              (CHECK_VERDICTS, 400),
              (RUN, 500), (HELLO, 500), (STREAM, 800), (FLEET, 400), (POP_ORDER, 300), (PUBLIC_UNITS, 300),
              (LINK_POINTS, 500), (REPLAY_LINKS, 300), (FITS, 400), (TEMPLATE_CIRCUITS, 300)]


@pytest.mark.parametrize("port, examples", PIECE_RUNS, ids=[p.fn for p, _ in PIECE_RUNS])
def test_a_piece_agrees(pieces, port, examples):
    stats = D.run(port, examples=examples, server=pieces)
    assert stats.outputs > examples // 4 and stats.rejects > 0, stats


def test_the_callers_check_agrees(callers):
    """The caller's check (`VStar.Recursive.Caller.check`, `spot-flock-caller`) gives the Python's codes
    (`caller.check`) on the docket piece's inputs: SPOT's check no longer makes these codes, so this port is the
    one that holds them."""
    stats = D.run(CALLER, examples=700, server=callers)
    assert stats.outputs > 175 and stats.rejects > 0, stats


def test_link_points_holds_the_per_member_forms_members_as_the_python_does(pieces):
    """The docket's `link-points` on the per-member form's members, Lean's answers the Python's (the piece's
    ``members``, which `Docket.check` reads as it reads the ones `OneStageZk.membersParamsOfJson` hands it from a
    case's parameters; `registration.check`'s ``members``). Two windows of two members at one unit a proof block:
    PAIR's two proof blocks and TRIPLE's eighteen, a template each, and one template's twenty-four cut in half. A row
    is a sampling design with each member's fewest and most sampled proof blocks over the samples that end, the members
    a sample that ends misses (fewest none) and the members whose fewest and most open different link points (3 up to
    eight proof blocks, 4 up to sixteen, 5 past). The `link_points` piece answers those counts and the members whose link
    points differ, and the docket is refused, for `link-points` alone, exactly where that list holds a member, a member a
    sample misses passing:

    * a stratum inside each member (`stratified:10` over the two templates), or a subset that leaves out one proof
      block (`subset:19`, `subset:23`): no sample misses a member and each has one count, so the docket passes;
    * `subset:9` over two and eighteen misses PAIR's member, which passes, and holds seven to nine of TRIPLE's, 3 link
      points or 4, which is refused;
    * `subset:20` over the halves holds eight to twelve of each, 3 link points or 4, and misses neither;
    * `subset:2` over two and eighteen, and `stratified:1` over the halves, whose one stratum holds both members,
      miss either member and give each one count: link points pass them;
    * a Bernoulli sampling design of rate 0 samples nothing, so each member's fewest and most are none: link points
      pass it.

    With no members, the one-statement form's, every record passes. Then dockets of such windows and of small
    ones, their members cut at random and some broken, agree, a share of them refused for their link points and a
    share passed."""
    two = [{"units": [[0, 2]], "vus_per_block": 1}, {"units": [[2, 20]], "vus_per_block": 1}]
    half = [{"units": [[0, 12]], "vus_per_block": 1}, {"units": [[12, 24]], "vus_per_block": 1}]
    rows = [(1, two, {"k": 10, "law": "stratified"}, [[1, 1], [9, 9]], [], []),
            (1, two, {"k": 19, "law": "subset"}, [[1, 2], [17, 18]], [], []),
            (0, half, {"k": 23, "law": "subset"}, [[11, 12], [11, 12]], [], []),
            (1, two, {"k": 9, "law": "subset"}, [[0, 2], [7, 9]], [0], [1]),
            (0, half, {"k": 20, "law": "subset"}, [[8, 12], [8, 12]], [], [0, 1]),
            (1, two, {"k": 2, "law": "subset"}, [[0, 2], [0, 2]], [0, 1], []),
            (0, half, {"k": 1, "law": "stratified"}, [[0, 1], [0, 1]], [0, 1], []),
            (1, two, {"law": "bernoulli", "p": "0/1"}, [[0, 0], [0, 0]], [0, 1], [])]
    for setup, members, design, bounds, missed, differ in rows:
        s = LINK_SETUPS[setup]
        rec = _record_of(s, None, design, WINDOW, leaf_layer=LEAF)
        assert [i for i, (lo, _) in enumerate(bounds) if lo == 0] == missed, design
        assert [i for i, (lo, hi) in enumerate(bounds) if LP.nbl(1, lo) != LP.nbl(1, hi)] == differ, design
        refused = differ
        counts = {"law": rec["law"], "members": members, "population": rec["population"]}
        assert link_points(counts) == {"bounds": bounds, "refused": refused}, design
        assert D.replay(LINK_POINTS, counts, pieces) == ("output", D.canonical(link_points(counts))), design
        given = {"params": {"anchors": [], "delta": "1/1000", "law": design, "leaf_layer": LEAF, "log": [],
                            "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": None,
                            "scheme": SCHEME, "work_table": None}, "record": rec}
        codes = ["link-points"] if refused else []
        for case, want in (({**given, "members": members}, codes), (given, []), ({**given, "members": []}, [])):
            reg = registration(case)
            assert reg["codes"] == want, (design, case.get("members"))
            assert D.replay(REGISTRATION, case, pieces) == ("output", D.canonical(reg))
        if design["law"] == "stratified":
            assert [t["k"] for t in rec["law"]["strata"]] == ([1, 9] if setup else [1])
    seen: dict[bool, int] = {}

    def counted(j):
        out = registration(j)
        refused = "link-points" in out["codes"]
        seen[refused] = seen.get(refused, 0) + 1
        return out
    stats = D.run(D.Port(PIECES, "registration", counted, REGISTRATIONS_WITH_MEMBERS, package=PACKAGE), examples=400,
                  server=pieces)
    assert stats.outputs > 100 and stats.rejects > 0 and min(seen.get(True, 0), seen.get(False, 0)) > 20, (stats, seen)


def _pin(program: Program, v: str, leaves: int | None = None, domain: str | None = None) -> dict:
    """A pin of `program`'s parameter `v` at one word a row, its domain at program digest ``aa`` * 64 unless `domain` names one."""
    n, b = G.shape(program, v, 1) if leaves is None else leaves, G.width(program, v)
    return {"value": v, "root": "ab" * 64, "leaves": n, "words": 1, **({} if b == RV.WORD_BITS else {RV.WIDTH_KEY: b}),
            "domain": domain or RV.domain("aa" * 64, v, 1, n, b).domain_id.hex()}


def _record_body(pins: list[dict]) -> bytes:
    """A deployment's registration record registering `pins` under the program ``aa…``."""
    return json.dumps({"format": DM.REGISTRATION, "program": "aa" * 64, "deployment": {"kind": "test"},
                       "registered": pins, "previous": None}).encode()


def _weights_doc(pins: list[dict], head: str = R.DEVELOPMENT) -> dict:
    """The parameters' `registration` of the record registering `pins`, on the development path, named."""
    return {"record": _record_body(pins).hex(), "head": head}


def test_r8_holds_registered_inputs_to_the_weights_of_record_as_python_does(pieces):
    """R8 under a weights' registration of record (`docket.weights_held`, `Docket.weightsOf`), Lean's codes the
    Python's: `x` registered and read by two calls passes with the registration and its digest, on a v1 record (the
    accepted pin is no entry R7 reads, and a v2 record is refused `keys`); it is refused without the registration
    (`R8-unpinned`, the case where Lean once passed what Python refused, handed no `parameters`), at a pin's wrong
    domain or shape, and with an entry's key beside its own or a value twice (`R8-weights-form`); a pin of the
    unregistered `x` is not read; and a
    registered input under a query no template query applies to is refused
    (`R8-weights-query`)."""
    marked = _readers(2, (("call", 2, _X0 + _X00),) * 2, True)
    plain = READER_SETUPS[0]
    s = Setup(marked, plain.query)
    reads = sorted({(t, p, "x") for t, p, _ in P.input_reads(marked, s.query, G.held_gates(marked, ["x"]))})
    assert reads
    pin = _pin(marked, "x")
    doc = _weights_doc([pin])
    cases = [(s, doc, [], []), (s, doc, [pin], ["keys"]),
             (s, None, [], ["R8-unpinned"]),
             (s, _weights_doc([_pin(marked, "x", domain="11" * 64)]), [], ["R8-weights-domain", "R8-unpinned"]),
             (s, _weights_doc([_pin(marked, "x", leaves=pin["leaves"] + 1)]), [], ["R8-weights-shape"]),
             (s, _weights_doc([{**pin, "reads": [list(r) for r in reads]}]), [], ["R8-weights-form"]),
             (s, _weights_doc([pin, pin]), [], ["R8-weights-form"]),
             (plain, doc, [], ["R8-shared-input"]),
             (Setup(marked, PO.query()), None, [], None)]
    for setup, weights, entries, want in cases:
        named, expanded = service.params_sampling_design("rate:1/2", setup.program, setup.query, None)
        roots = [{"port": p, "root": ROOTS[p], **setup.ports[p]} for p in setup.ports]
        rec = DK.record(setup.program, setup.query, named, SCHEME, roots, WINDOW, leaf_layer=LEAF, work_table=expanded,
                        registered=entries)
        sig = _json([[name, t.to_json()] for name, t in setup.program.fn.signature()[0]])
        params = {"anchors": [], "delta": "1/1000", "law": "rate:1/2", "leaf_layer": LEAF, "log": [], "partition": None,
                  "ports": setup.ports, "program": setup.desc, "query": setup.query, "registered": None, "scheme": SCHEME,
                  "work_table": None, "parameters": sig, **({} if weights is None else {"registration": weights})}
        given = _json({"params": params, "record": rec})
        reg = registration(given)
        assert reg["codes"] == want if want is not None else "R8-weights-query" in reg["codes"], (weights, reg)
        assert D.replay(REGISTRATION, given, pieces) == ("output", D.canonical(reg)), weights


def test_r8_refuses_a_marked_input_expected_by_name_alone_as_python_does(pieces):
    """R8 on a marked input the verifier expects registered by name (`docket.weights_held`, `Docket.weightsHeld`), Lean's
    codes the Python's: with no pin R8 accepted and no root the verifier holds itself it is refused (`R8-unpinned`)
    whatever root the record names; pinned by the verifier (`pinned`) it passes."""
    marked = _readers(2, (("call", 2, _X0 + _X00),) * 2, True)
    s = Setup(marked, READER_SETUPS[0].query)
    for pinned, want in ((None, ["R8-unpinned"]), ({"x": "ab" * 64}, [])):
        named, expanded = service.params_sampling_design("rate:1/2", s.program, s.query, None)
        roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
        mine = G.registered_entries(s.program, {"x": 1}, None, None, ("x",) if pinned else ())
        entries = [{"value": v, "root": "ab" * 64, **e} for v, e in mine.items()]
        rec = DK.record(s.program, s.query, named, SCHEME, roots, WINDOW, leaf_layer=LEAF, work_table=expanded,
                        registered=entries)
        sig = _json([[name, t.to_json()] for name, t in s.program.fn.signature()[0]])
        params = {"anchors": [], "delta": "1/1000", "law": "rate:1/2", "leaf_layer": LEAF, "log": [], "partition": None,
                  "ports": s.ports, "program": s.desc, "query": s.query, "registered": {"x": 1}, "scheme": SCHEME,
                  "work_table": None, "parameters": sig, **({} if pinned is None else {"pinned": pinned})}
        given = _json({"params": params, "record": rec})
        reg = registration(given)
        assert reg["codes"] == want, (pinned, reg)
        assert D.replay(REGISTRATION, given, pieces) == ("output", D.canonical(reg)), pinned


def test_the_parameters_registration_is_refused_by_name_as_python_does(pieces):
    """`OneStage.registrationParsed` (`docket.registration_parsed`): a record with a ledger and no challenger, a record
    with neither a ledger nor the development path named, a head other than `development`, a record not in lowercase
    hex, and a record that is no JSON are each refused before any check, by name, in Lean as in Python."""
    s = Setup(_readers(2, (("call", 2, _X0 + _X00),) * 2, True), READER_SETUPS[0].query)
    body = _record_body([_pin(s.program, "x")]).hex()
    for reg, why in (({"record": body, "ledger": [], "lane": "circuits"}, R.NO_SIGNED_HEAD_WHY),
                     ({"record": body}, R.REGISTRATION_FORM),
                     ({"record": body, "head": "served"}, R.REGISTRATION_FORM),
                     ({"record": body.upper(), "head": R.DEVELOPMENT}, R.REGISTRATION_FORM),
                     ({"record": b"{".hex(), "head": R.DEVELOPMENT}, R.REGISTRATION_FORM)):
        with pytest.raises(R.RegistrationRefused) as e:
            R.registration_parsed(reg, None, None, lms_verify)
        assert str(e.value) == why
        given = _registration_case(s, [], registration=reg)
        assert D.replay(REGISTRATION, given, pieces)[0] == "reject", reg


def _registration_case(s: Setup, entries: list, **more) -> dict:
    named, expanded = service.params_sampling_design("rate:1/2", s.program, s.query, None)
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    rec = DK.record(s.program, s.query, named, SCHEME, roots, WINDOW, leaf_layer=LEAF, work_table=expanded,
                    registered=entries)
    sig = _json([[name, t.to_json()] for name, t in s.program.fn.signature()[0]])
    params = {"anchors": [], "delta": "1/1000", "law": "rate:1/2", "leaf_layer": LEAF, "log": [], "partition": None,
              "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
              "work_table": None, "parameters": sig, **more}
    return _json({"params": params, "record": rec})


def test_r7_refuses_a_supplied_input_with_no_mark_and_no_source_as_python_does(pieces):
    """R7 on C whose root holds a SuppliedInput node with no mark in C (no `inputs` entry) and no source in the
    parameters' `supplied` (`docket.unsourced_supplied`, `Docket.unsourced`), on a record that registers nothing: refused
    by name (`R7-supplied-source`), Lean's codes the Python's. A source is a registered value's (`supplied`), so its
    case is a v2 record's (`test_one_stage`, `Spot/Test/Docket.lean`)."""
    plain = _readers(2, (("call", 2, _X0 + _X00),) * 2, False)
    s = Setup(Program(plain.fn, supplied=("x",)), READER_SETUPS[0].query)
    PROGRAMS[s.sha512] = s.program
    assert G.marks(s.desc) == {} and R.unsourced_supplied(s.program, {}) == ["x"]
    given = _registration_case(s, [])
    reg = registration(given)
    assert "R7-supplied-source" in reg["codes"], reg
    assert D.replay(REGISTRATION, given, pieces) == ("output", D.canonical(reg))


def test_r7_refuses_a_pinned_values_entry_whose_domain_alone_differs_as_python_does(pieces):
    """R7 on a value the weights' registration of record pins (`docket.off_pin`, `Docket.offPin`): the record's entry
    equal to the pin in root, leaf count and words, its domain alone another, is refused by name (`R7-pin`), Lean's codes
    the Python's; the entry that is the pin is not."""
    s = Setup(_readers(2, (("call", 2, _X0 + _X00),) * 2, True), READER_SETUPS[0].query)
    pin = _pin(s.program, "x")
    entry = {k: pin[k] for k in ("value", "root", "leaves", "words", "domain")}
    for e, refused in ((entry, False), ({**entry, "domain": "11" * 64}, True)):
        body = _record_body([pin])
        assert R.off_pin({}, [e], (body, DD.digest(json.loads(body)))) == (["x"] if refused else [])
        given = _registration_case(s, [e], registered={"x": 1}, registration=_weights_doc([pin]))
        reg = registration(given)
        assert ("R7-pin" in reg["codes"]) == refused, (e, reg)
        assert D.replay(REGISTRATION, given, pieces) == ("output", D.canonical(reg)), e


def test_the_verifiers_gated_list_is_the_lints_generated_one(pieces):
    """The list R9 holds gated templates by, compiled into the Lean (`Docket.GATED`) and in the Python (`docket.GATED`), is
    exactly the entries of circuits' generated list (`tests/lint/openers_gated.json`), each evaluated at its binding: the
    file is what the constant is held to, never what the verifier reads."""
    root = next(p for p in Path(__file__).resolve().parents if (p / "tests" / "lint").is_dir())
    entries = json.loads((root / "tests/lint/openers_gated.json").read_text())["entries"]
    assert all(e["class"] == "gated" and e["evaluated"] is True for e in entries), entries
    want = [[e["template"], e["closure_digest"]] for e in entries]
    assert gated_list(None) == want
    assert D.replay(GATED_LIST, None, pieces) == ("output", D.canonical(want))


def test_r9_refuses_a_read_outside_a_held_gated_template_as_the_python_does(pieces):
    """R9's walk, Lean's answer the Python's: under the verifier's list (`docket.GATED`, the served binding
    `EmbeddingRead_v4{V=128256,H=2048}` alone) each small read is named, the ungated `EmbeddingRead_v2` and the gated
    `EmbeddingRead_v4` at bindings not on it; under a list holding each read's own closure digest, none is named (the
    walk holds what the list holds); under its own id at another digest, each is named."""
    for tid in OPENER_READS:
        desc = _opener_desc(tid)
        own = [[e["id"], codec.closure_digest(desc, k)] for k, e in desc["definitions"].items() if k == tid]
        for held, want in (([list(g) for g in R.GATED], [tid]), (own, []),
                           ([[i, "00" * 64] for i, _ in own], [tid])):
            case = {"program": desc, "gated": held}
            got = ungated_openers(case)
            assert got == {"templates": want}, (tid, held, got)
            assert D.replay(UNGATED_OPENERS, case, pieces) == ("output", D.canonical(got)), (tid, held)


def test_r8_reads_the_readers_python_does(pieces):
    """R8 past a population program, Lean's answers the Python's: two calls of `PAIR` on one pair of `x`, or a batch
    broadcasting it, at F 1, and a call reading another's result, whose result a batch of two maps, joining all three at
    F 4, which the docket publishes; a run Input two proof blocks read is refused in the variant with no members
    (`R8-shared-input`) and not with members, whose check holds its readers to the node's one tree; `x` named tied is
    let through, which the docket publishes, and a tied name that isn't a parameter is refused (`R8-tied`)."""
    for s, f, shared in zip(READER_SETUPS, (1, 1, 4), (True, True, False)):
        case = {"program": s.desc, "query": _json(s.query)}
        assert fanout(case) == {"fanout": f}
        assert D.replay(FANOUT, case, pieces) == ("output", D.canonical(fanout(case)))
        named, expanded = service.params_sampling_design("rate:1/2", s.program, s.query, None)
        rec = _record_of(s, None, named, WINDOW, leaf_layer=LEAF, work_table=expanded)
        given = {"params": {"anchors": [], "delta": "1/1000", "law": "rate:1/2", "leaf_layer": LEAF, "log": [],
                            "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": None,
                            "scheme": SCHEME, "work_table": None}, "record": rec}
        reg = registration(given)
        assert reg == {"codes": ["R8-shared-input"] if shared else [], "fanout": f}
        assert D.replay(REGISTRATION, given, pieces) == ("output", D.canonical(reg))
        n = P.units(s.program, s.query)
        each = {**given, "members": [{"units": [[u, u + 1]], "vus_per_block": 1} for u in range(n)]}
        assert "R8-shared-input" not in registration(each)["codes"]
        assert D.replay(REGISTRATION, each, pieces) == ("output", D.canonical(registration(each)))
        sig = _json([[name, t.to_json()] for name, t in s.program.fn.signature()[0]])
        for tied, want in ((["x"], {"codes": [], "fanout": f, "rests_on_period_check": ["x"]}),
                           (["nope"], {"codes": ["R8-tied", *(["R8-shared-input"] if shared else [])], "fanout": f})):
            tied_given = {**given, "params": {**given["params"], "parameters": sig, "tied": tied}}
            assert registration(tied_given) == want
            assert D.replay(REGISTRATION, tied_given, pieces) == ("output", D.canonical(want))
    program = _readers(2, (("call", 2, _X0 + _X00),) * 2, True)
    case = {"program": _json(codec.encode_program(program)), "query": _json(P.template_instance_query(PAIR.id))}
    assert D.replay(FANOUT, case, pieces) == ("output", D.canonical({"fanout": 1}))


def test_run_input_roots_reads_every_proof_block_as_python_does(pieces):
    """`run_input_roots`, Lean's answers the Python's: two calls of `PAIR` reading one pair of the run Input `x` each
    open the record's one tree of it at C's leaf count and the domain under the window's open root; no tree, another
    leaf count, another domain, no open root, and a tied `x`, each as Python answers."""
    s = READER_SETUPS[0]
    prog = P.program_digest(s.program)
    R = EG._Root(s.desc)
    reads = M.member_inputs(R, P._template(s.program, s.query, "q").ranges, "T/m0", [(0, 1)])
    assert reads and all(R.nodes[w]["input"] for _p, w, _n in reads)
    w, ln = reads[0][1], reads[0][2]
    rows = R.nodes[w]["out_leaves"] * R.nodes[w]["members"] // ln
    words = ln * (EG._bits(R.nodes[w]["ret"]) // R.nodes[w]["out_leaves"]) // 16
    good = {"port": f"input/n{w}", "root": "aa" * 64, "leaves": rows,
            "domain": RV.domain(prog, f"input/n{w}", words, rows, 16, window=WROOT).domain_id.hex()}
    base = {"program": s.desc, "query": _json(s.query), "members": [["T/m0", [[0, 1]]], ["T/m1", [[1, 2]]]],
            "roots": [good], "window_root": WROOT}
    got = run_input_roots(base)
    assert got["refused"] is None and all(any(t == f"input/n{w}" for _p, t in r) for _n, r in got["roots"]), got
    cases = [base, {**base, "roots": []}, {**base, "roots": [{**good, "leaves": rows + 1}]},
             {**base, "roots": [{**good, "domain": "dd" * 64}]}, {k: v for k, v in base.items() if k != "window_root"},
             {**base, "tied": [[g.start, len(g)] for g in G.held_gates(s.program, ["x"])]}]
    whys = [M.run_tree_missing_why("T/m0", reads[0][0], w), M.run_tree_why(w, "leaf count"),
            M.run_tree_why(w, "domain"), M.run_tree_unbound_why(w)]
    assert [run_input_roots(c)["refused"] for c in cases[1:5]] == whys
    assert run_input_roots(cases[5]) == {"roots": [["T/m0", []], ["T/m1", []]], "refused": None}
    for c in cases:
        assert D.replay(RUN_INPUT_ROOTS, c, pieces) == ("output", D.canonical(run_input_roots(c)))


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


def test_the_select_piece_samples_every_rule(pieces):
    """One sample under each rule `select_draw` samples (the stream's run refuses a Bernoulli sampling design and a closure
    map before it), on a partition whose `PAIR` stratum is two ranges apart, Lean's the Python's whole: `All` every unit,
    reading no coins, and each rate's sample (a work rate's with its closure) reading them, each stratum's count its k,
    which the sampled runs reach only probably, a rate sampling design's a subset's of its count over the 10 units, a stratified or
    work rate's every stratum's k at least its rate of the stratum's units, and a two-stage rule's first stage its replay
    rate's subset. Then the same with every non-ASCII digit `str.isdigit` admits in place of an ASCII one, a rate whose
    count passes the population, a fixed count, as a sampling design or a two-stage rule's replay sampling design, and the retired two-stage
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


def test_a_work_samples_design_is_cores(pieces):
    """Flock's work sampling design held to `Core.SamplingDesign`: a work sample's sampling design, Lean's (`Spot.Pieces.designOfSamplingDesign`) the Python's
    (`audit.design`), is the profile's `work` case with each stratum's works and floor, and its sample's design is the
    stratified one at its k_s."""
    desc = json.loads(_population(((0, 2), (1, 3), (0, 4), (2, 1))))
    case = {"closure": None, "program": desc, "query": _json(P.template_instances_query([t.id for t in TEMPLATES])),
            "rule": {"law": "work-rate:2/5", "rule": "law"}, "source": hashlib.shake_256(b"design").hexdigest(1500),
            "work_table": {PAIR.id: 2, TRIPLE.id: {"floor": 2, "work": 1}, QUAD.id: 0}}
    out = design(copy.deepcopy(case))
    assert D.replay(DESIGN, case, pieces) == ("output", D.canonical(out))
    d = out["design"]
    assert d["law"] == "work" and d["k"] == 4 and {s["name"] for s in d["strata"]} == {PAIR.id, TRIPLE.id, QUAD.id}
    assert [s["floor"] for s in d["strata"] if s["name"] == TRIPLE.id] == [2]
    work = audit.design(select(copy.deepcopy(case))["draw"])
    assert work.stratified == Stratified(tuple(Stratum(s["name"], s["size"], s["k"]) for s in d["strata"]))


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


#: Templates of templates: `NEST` calls `PAIR` twice and `DEEP` calls `NEST` and `PAIR`, so a unit's gates and every
#: offset before it are walks of nested Definitions.
NEST = bind(CompositeDefinition("CoreSpotNest", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, B.call(PAIR, x), B.call(PAIR, x)), register=False))
DEEP = bind(CompositeDefinition("CoreSpotDeep", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                lambda B, S, x: B.call(XOR8, B.call(NEST, x), B.call(PAIR, x)), register=False))


def test_the_slots_of_a_population_of_nested_templates(pieces):
    """On 2,000 units in seven root batches of four templates, two of them nested two deep and three of them in two
    batches apart, Lean's slots (`Partition.unitGates`, each callee's gates and each batch's offset read off one table) are
    the Python's whole (`TemplatePartition.gates` per unit), and the population's and the input class's gates are every
    gate of the program once."""
    program = PN.mixed_population_program([(DEEP, 300), (PAIR, 250), (NEST, 400), (DEEP, 170), (QUAD, 500), (NEST, 90),
                                          (PAIR, 290)])
    PROGRAMS[PO.program_sha512(program)] = program
    query = _json(P.template_instances_query([DEEP.id, PAIR.id, NEST.id, QUAD.id]))
    case = {"closure": None, "program": _json(codec.encode_program(program)), "query": query,
            "rule": {"law": "rate:1/125", "rule": "law"}, "source": hashlib.shake_256(b"nested").hexdigest(400),
            "work_table": None}
    out = slots(copy.deepcopy(case))
    assert D.replay(SLOTS, case, pieces) == ("output", D.canonical(out))
    assert [len(s) for s in out["classes"][0]] == [470, 540, 490, 500]
    rus = [r for c in out["classes"] for s in c for r in s]
    gates = sorted(g for r in rus for lo, hi in r["pus"] for g in range(lo, hi))
    assert gates == list(range(program.circuit.size))


def test_the_population_order_from_the_ranges_is_the_layouts(pieces):
    """`Slots.popOk`, which runs from the ranges (`Slots.popOkRanges`), is its statement read off the layout's population,
    and the Python's (`partition.in_order`), at the docket's `n` and either side of it: on 2,000 units in seven root
    batches, whose strata put units apart and so fail it, on one batch of each of those templates, which passes at `n`
    only, and on `SETUPS`, out of order and in order."""
    apart = PN.mixed_population_program([(DEEP, 300), (PAIR, 250), (NEST, 400), (DEEP, 170), (QUAD, 500), (NEST, 90),
                                        (PAIR, 290)])
    blocks = PN.mixed_population_program([(DEEP, 470), (PAIR, 540), (NEST, 490), (QUAD, 500)])
    query = _json(P.template_instances_query([DEEP.id, PAIR.id, NEST.id, QUAD.id]))
    cases = [(apart, query, 2000, False), (blocks, query, 2000, True)]
    cases += [(s.program, _json(s.query), s.population, ok)
              for s, ok in ((SETUPS[4], False), (SETUPS[5], False), (SETUPS[3], True), (SETUPS[2], True), (SETUPS[6], True))]
    for program, query, n, ok in cases:
        PROGRAMS[PO.program_sha512(program)] = program
        for m, want in ((n, ok), (n - 1, False), (n + 1, False)):
            case = {"population": m, "program": _json(codec.encode_program(program)), "query": query}
            assert pop_order(copy.deepcopy(case)) == {"layout": want, "ranges": want}, (m, want)
            assert D.replay(POP_ORDER, case, pieces) == ("output", D.canonical({"layout": want, "ranges": want})), (m, want)


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
    ("EmbeddingRead_v2{V=256,H=3072}", 131072),
    ("EmbeddingRead_v2{V=256,H=96}", 1024),
    ('MoeExpertFp8BlockGemm_v4{K=6144,N=1,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v2"}}', 131072),
    ("EmbeddingRead_v2{V=256,H=4096}", 49152),
    ("EmbeddingRead_v2{V=256,H=4}", 12288),
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
    from verity_vllm.definitions.registry.catalog import load
    from verity.commitments.periods import interface as IF

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
    program = PN.population_program(template, n)
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


def test_glm53s_all_reduce_at_three_rows_is_refused_past_the_widest_class(pieces):
    """GLM-5.3's TP8 MNNVL all-reduce at three rows, as `tp_allreduce.circuit-check.py`'s `_glm53_served` binds it
    (`AllReduceMnnvl_v1{WORLD=8,N=18432,SHOT=1}`), returns 294,912 bits: Lean's slots and the Python's partition refuse
    its instance by the widest class."""
    from verity_vllm.definitions.registry.catalog import load
    from verity.commitments.periods import interface as IF
    load()
    unit = IF.resolve("AllReduceMnnvl_v1{WORLD=8,N=18432,SHOT=1}")
    query = P.template_instance_query(unit.id)
    why = "commits 294912 bits, more than the widest class, 262144"
    with pytest.raises(P.QueryRefused, match=why):
        P.units(PN.population_program(unit, 1), query)
    refused = {"closure": None, "rule": {"law": "rate:1/1000", "rule": "law"},
               "source": hashlib.shake_256(b"refused").hexdigest(400), "work_table": None,
               "program": _template_population(unit, 1), "query": _json(query)}
    kind, body = D.replay(SLOTS, refused, pieces)
    assert kind == "reject" and why in str(body), (kind, body)
    with pytest.raises(D.Reject, match=why):
        slots(copy.deepcopy(refused))


R7_CODES = {"R7-registered", "R7-values", "R7-root", "R7-leaves", "R7-words", "R7-domain", "R7-word_bits", "R7-held",
            "R7-supplied", "R7-certified"}


def _v2_audit(s: RegisteredSetup, values: dict) -> dict:
    """One valid v2 audit of `s` expecting `values` registered, the log's earlier record registering them under the same
    roots, by a verifier bound to `BINDING` that holds the values it doesn't fill across windows (`held`)."""
    sampling_design = {"k": 1, "law": "subset"}
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    earlier = DK.record(s.program, s.query, sampling_design, SCHEME, roots, EARLIER, leaf_layer=LEAF, registered=s.entries(values))
    rec = _record_of(s, values, sampling_design, {**WINDOW, **BINDING}, [earlier], BINDING, leaf_layer=LEAF, previous=DD.digest(earlier))
    params = {"anchors": [], "binding": BINDING, "delta": "1/10", "law": sampling_design, "leaf_layer": LEAF, "log": [earlier],
              "held": sorted(set(values) - set(s.supplied(values) or {})), "parameters": s.signature,
              "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": values,
              "scheme": SCHEME, "work_table": None, **({"supplied": s.supplied(values)} if s.supplied(values) else {}),
              **({"row_schemas": {v: f for v, f in s.formats.items() if v in values}} if s.formats else {})}
    evidence = {"draw": {**sampling_design, "population": s.population, "units": [0]}, "instances": 1, "proved_units": [0],
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
            for change, x in ((c, x) for c in ENTRIES + REGISTERED_PARAMS + SUPPLIED + CERTIFIED for x in range(8)):
                b = copy.deepcopy(a)
                try:
                    change(b, x)
                except (KeyError, TypeError, IndexError, ValueError, AttributeError):
                    continue
                kind, out = D.replay(REGISTRATION, _json(_REGISTRATION_OF(b)), pieces)
                seen |= set(json.loads(out)["codes"]) if kind == "output" else set()
    assert R7_CODES <= seen, sorted(R7_CODES - seen)


def test_r7_takes_the_verifiers_certified_entries(pieces, server):
    """Lean and Python agreeing, on a program none of whose parameters is registered and on one with a value registered:
    a record that registers a certified value beside the verifier's own is accepted by a verifier holding the entry for
    it, at 16 bits and at 32, and refused by one holding none (by its keys where it expects nothing registered, else
    `R7-values`); the verifier's entry with no entry in the record is refused (by the record's keys, else `R7-values`
    and `R7-certified`), an entry at another root is `R7-certified`, and one of other leaves, words, domain or width
    R7's code for that key. A certified value the verifier expects a parameter registered under, or an entry in
    another form, refuses its inputs. The sampled audit refuses parameters that name certified entries, on an audit
    its own checks would accept. A run of record whose parameters name the entry accepts the record that registers the
    certified value and refuses it at its register step without (`R7-values`)."""
    def codes(a, change, x=0):
        b = copy.deepcopy(a)
        change(b, x)
        kind, out = D.replay(REGISTRATION, _json(_REGISTRATION_OF(b)), pieces)
        return json.loads(out)["codes"] if kind == "output" else kind

    alike, entry, mine, forms, wide, uncertified = CERTIFIED
    lone, own = _v1_audit("all", SETUPS[0]), _v2_audit(REGISTERED_SETUPS[0], {"W": 2})
    for a, unheld, unregistered in ((lone, ["keys"], ["keys"]), (own, ["R7-values"], ["R7-values", "R7-certified"])):
        assert codes(a, lambda b, x: None) == []
        assert codes(a, alike) == [] and codes(a, wide) == []
        assert codes(a, uncertified) == unheld and codes(a, forms, 0) == unheld
        assert codes(a, entry, 0) == unregistered
        for x, want in ((1, ["R7-certified"]), (2, ["R7-leaves"]), (3, ["R7-words"]), (4, ["R7-domain"]),
                        (5, ["R7-word_bits"]), (6, ["R7-word_bits"]), (7, ["R7-root", "R7-certified"]),
                        (8, ["R7-leaves"]), (9, ["R7-registered"]), (11, ["R7-registered", "R7-certified"])):
            assert codes(a, entry, x) == want, x
        for x, want in ((0, ["R7-certified"]), (1, ["R7-leaves"]), (2, ["R7-word_bits"]), (3, [])):
            assert codes(a, mine, x) == want, x
        for x in range(4, 15):
            assert codes(a, mine, x) == "reject", x
        for x in range(5, 11):
            assert codes(a, forms, x) == "reject", x
    assert codes(own, forms, 1) == "reject" and codes(lone, forms, 1) == ["R7-values", "R7-certified"]

    full = copy.deepcopy(own)
    alike(full, 0)
    full["evidence"].update(receipt=R.receipt(full["record"], "v", received=RECEIVED), verdicts={
        "lean": {"accepted": True, "of_record": True, "registered": reads(full["record"])}})
    case = _json(_case(full))
    parsed = (decode_params(case["params"]), decode_commitment(PR.unhex(case["commitment"])),
              decode_evidence(PR.unhex(case["evidence"])))
    assert check(*parsed)[0] == "accept"
    assert D.replay(ONE_STAGE, case, server)[0] == "reject"
    assert D.replay(ONE_STAGE, _json(_case(own)), server)[0] == "accept"

    ran = _v2_audit(REGISTERED_SETUPS[0], REGISTERED_SETUPS[0].values[1])
    alike(ran, 0)
    ran["evidence"]["verdicts"] = {"lean": {"accepted": True, "of_record": True, "registered": reads(ran["record"])},
                                   "upstream": {"accepted": True, "of_record": False}}
    _, case = _from_challenger(_run_case(ran, "", 2, ["credited"]), random.Random("run/certified"))
    case = {**case, "values": [*case["values"], {"name": CERTIFIED_VALUE, "registered": True}],
            "public": [*case["public"], *(f"{k}:{CERTIFIED_VALUE}" for k in ("leaves", "domain", "words", "word_bits"))]}
    bare = {**case, "params": {k: v for k, v in case["params"].items() if k != "certified"}}
    for c, want in ((case, ("accept", None, None)), (bare, (None, "register", ["R7-values"]))):
        out = run(copy.deepcopy(c))
        assert (out.get("symbol"), out.get("step"), out.get("codes")) == want, out
        assert D.replay(RUN, c, pieces) == ("output", D.canonical(out)), out


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
    t = PP.commit("p", "ok", [[1]], PRODUCED_REG, [bytes(range(192))], schema=G.SCHEMA_V2)
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
    `outputs` are `R7-domain`. The record's entry under one-stage's own domain for `T` (`registered_value.domain`, whichever
    binding it takes) is `R7-domain`, under another root `R7-held`, and Pearl+'s entry of 16-bit words or of other rows
    `R7-output`, Lean and Python agreeing."""
    s, values, sampling_design = REGISTERED_SETUPS[1], {"T": 2}, {"k": 1, "law": "subset"}
    mine = s.committed["T", 2].entry()
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    salt, u = bytes(range(64)), 7

    def codes(b):
        kind, out = D.replay(REGISTRATION, _json(b), pieces)
        return json.loads(out)["codes"] if kind == "output" else kind

    for name, binding, nonce in (("pearl-a/0/0", PA.activation_binding(salt, u), u),
                                 ("pearl-y/0/0", tagged(PEARL_Y_ROWS, salt, u), (1 << 61) + u)):
        rows = [b"".join(w.to_bytes(4, "little") for w in r) for r in s._rows("T", 2)[0]]
        tree = PA.commit_rows(binding, u, rows, PA.HM96, salts=[hashlib.shake_256(f"salt {nonce} {i}".encode()).digest(PA.HM96_SALT)
                                                                for i in range(len(rows))])
        theirs = {**mine, "value": name, "root": tree.commitment.root.hex(), "domain": tree.domain.domain_id.hex()}
        assert theirs["domain"] != mine["domain"]

        def case(entry=(), earlier=theirs, drop=(), **change):
            log = [DK.record(s.program, s.query, sampling_design, SCHEME, roots, EARLIER, leaf_layer=LEAF, registered=[earlier])]
            rec = DK.record(s.program, s.query, sampling_design, SCHEME, roots, {**WINDOW, **BINDING}, leaf_layer=LEAF,
                           previous=DD.digest(log[0]),
                           registered=[{**mine, "root": theirs["root"], "domain": theirs["domain"], **dict(entry)}])
            params = {"anchors": [], "binding": BINDING, "delta": "1/10", "law": sampling_design, "leaf_layer": LEAF, "log": log,
                      "outputs": {"T": {"record": DD.digest(log[0]), "value": name}}, "parameters": s.signature,
                      "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": values,
                      "row_schemas": {"T": G.SCHEMA_SEG}, "scheme": SCHEME, "work_table": None, **change}
            return {"params": {k: v for k, v in params.items() if k not in drop}, "record": rec}

        assert codes(case()) == [], name
        assert codes(case(row_schemas={"T": G.SCHEMA_V2})) == [], name
        assert codes(case(drop=["outputs"])) == ["R7-domain"], name
        assert codes(case({"domain": mine["domain"]})) == ["R7-domain"], name
        assert codes(case({"root": "ef" * 64})) == ["R7-held"], name
        for e in ({k: v for k, v in theirs.items() if k != RV.WIDTH_KEY}, {**theirs, "leaves": 4, "words": 1}):
            assert codes(case(earlier=e))[0] == "R7-output", (name, e)


def _codes_of(case, pieces, callers) -> list[str] | str:
    """SPOT's codes for `case` (`REGISTRATION`), then the caller's (`CALLER`), Lean and Python agreeing on each; the kind
    of the first that doesn't answer, where the parameters are refused."""
    out: list[str] = []
    for port, server in ((REGISTRATION, pieces), (CALLER, callers)):
        kind, got = D.replay(port, case, server)
        if kind != "output":
            return kind
        out += json.loads(got)["codes"]
    return out


def test_a_tile_replayed_across_streams_in_one_window_is_refused(pieces, callers):
    """The supplied setup whose seeds read the window's salt and positions, bound to one stream and prover: its record, as
    another stream's (or prover's) verifier checks it in the same window, is refused as it stands (`R6-stream` or
    `R6-prover`, and `R7-supplied`) and relabeled to that verifier's binding (`R7-supplied`: its seeds are the first
    stream's), and by a verifier with no binding (`R7-supplied`), Lean and Python agreeing; every binding change reaches
    those codes or a refused binding, and nothing else."""
    s = next(s for s in SUPPLIED_SETUPS if _reads_window(s.supplied(s.values[0])))
    a = _v2_audit(s, s.values[0])

    def codes(b):
        return _codes_of(_json(_REGISTRATION_OF(b)), pieces, callers)

    assert codes(a) == []
    for k, other in (("stream", "3d" * 64), ("prover", "d1" * 32)):
        b = copy.deepcopy(a)
        b["params"]["binding"] = {**BINDING, k: other}
        assert codes(b) == ["R7-supplied", f"R6-{k}"], k
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
    sampling_design = {"k": 1, "law": "subset"}
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    earlier = DK.record(s.program, s.query, sampling_design, SCHEME, roots, EARLIER, leaf_layer=LEAF,
                       registered=s.entries(values, EARLIER_WROOT, ["W"]))
    rec = _record_of(s, values, sampling_design, WINDOW, [earlier], leaf_layer=LEAF, previous=DD.digest(earlier), window_root=WROOT,
                     fixed=["W"])
    params = {"anchors": [], "delta": "1/10", "held": ["W"], "law": sampling_design, "leaf_layer": LEAF, "log": [earlier],
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
    plain = DK.record(s.program, s.query, sampling_design, SCHEME, roots, EARLIER, leaf_layer=LEAF, registered=s.entries(values))
    for again in (s.entries(values), rec["registered"]):
        b = {"params": {**unrooted, "log": [plain]},
             "record": {**_record_of(s, values, sampling_design, WINDOW, [plain], leaf_layer=LEAF, previous=DD.digest(plain)),
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


def test_a_seed_salt_the_server_chose_is_refused_by_name(pieces, callers):
    """Lean and Python agreeing: a verifier naming its window's open root that read the window's seed salt from its
    challenger accepts a record whose window names that salt, and refuses one naming another salt, one in another form or
    none (`R6-seed-salt`). A verifier that read none (no `seed_salt`, or null) accepts a window naming none and refuses one
    naming a salt (`R6-seed-salt`): the server chose it. A salt the verifier holds in another form, or without the
    window's open root, refuses its inputs."""
    s, values = REGISTERED_SETUPS[0], {"D": 1, "W": 3}
    sampling_design = {"k": 1, "law": "subset"}
    rec = _record_of(s, values, sampling_design, {**WINDOW, R.SEED_SALT: SEED_SALT}, leaf_layer=LEAF, window_root=WROOT)
    params = {"anchors": [], "delta": "1/10", "law": sampling_design, "leaf_layer": LEAF, "log": [], "parameters": s.signature,
              "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": values,
              "scheme": SCHEME, "seed_salt": SEED_SALT, "window_root": WROOT, "work_table": None}

    def codes(window, p=params):
        return _codes_of(_json({"params": p, "record": {**rec, "window": window}}), pieces, callers)

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


def test_advice_no_registered_compiler_reads_is_refused_by_name(pieces, callers):
    """Lean and Python agreeing on the registered compiler (`compiler`, the registration log's item, with the public
    length ℓ of the advice it reads): a window naming it by its digest with advice of at most ℓ bits is accepted, one
    naming another compiler or none is refused by name (`R6-compiler`), and advice over ℓ, in another form or none too
    (`R6-advice-length`). Where the log registers none (`compiler` null), a window naming a compiler or advice is refused
    (`R6-compiler`); a verifier that reads no log's compiler holds the window to nothing. A compiler in another form
    refuses the verifier's inputs."""
    s, values = REGISTERED_SETUPS[0], {"D": 1, "W": 3}
    design = {"k": 1, "law": "subset"}
    digest, other = "c1" * 64, "c2" * 64
    item = {"advice": "verity/advice/v1", "cap": 32, "digest": digest, "kind": "builder"}

    def named(d, bits):
        return {**WINDOW, "compiler": {"digest": d, "name": "c"}, "advice": {"bits": bits, "root": "ad" * 64}}

    rec = _record_of(s, values, design, named(digest, 8), leaf_layer=LEAF, window_root=WROOT)
    params = {"anchors": [], "compiler": item, "delta": "1/10", "law": design, "leaf_layer": LEAF, "log": [],
              "parameters": s.signature, "partition": None, "ports": s.ports, "program": s.desc, "query": s.query,
              "registered": values, "scheme": SCHEME, "window_root": WROOT, "work_table": None}
    unread = {k: v for k, v in params.items() if k != "compiler"}
    none = {**params, "compiler": None}

    def codes(window, p=params):
        return _codes_of(_json({"params": p, "record": {**rec, "window": window}}), pieces, callers)

    cases = [(named(digest, b), params, []) for b in (0, 32)]
    cases += [(named(digest, b), params, ["R6-advice-length"]) for b in (33, -1, "8", None, True)]
    cases += [(named(other, 8), params, ["R6-compiler"]), (WINDOW, params, ["R6-compiler", "R6-advice-length"]),
              (named(digest, 8), none, ["R6-compiler"]),
              ({**WINDOW, "advice": {"bits": 0, "root": "ad" * 64}}, none, ["R6-compiler"]), (WINDOW, none, []),
              (named(other, 99), unread, []), (WINDOW, unread, [])]
    for window, p, want in cases:
        assert codes(window, p) == want, (window, p.get("compiler"))
    for bad in ({**item, "kind": "theta"}, {**item, "digest": 5}, {**item, "digest": digest.upper()},
                {k: v for k, v in item.items() if k != "advice"}, {**item, "extra": 1}, "c"):
        assert codes(rec["window"], {**params, "compiler": bad}) == "reject", bad


def test_a_window_opens_the_pool_its_chain_left(pieces, callers):
    """Lean and Python agreeing on the pool chain: a window opens the pool the previous window of record left, of the
    same program, `pin` the previous record's `pout` entry, its domain and root, whatever the verifier made of that
    window; it opens the empty pool, `pin` filled from `genesis`, at a docket's first window, after the verifier
    refused the window the public lag back, as its verdict log names it (`verdicts`), and where serving resets it
    (`reset`), which the record's window names, with its week and whether the week's reset is spent at it. Any other
    `pin` is refused by name (`R7-pool-chain`), a window naming another reset, week or spent mark than the verifier's
    pool gives, or none, by name (`R6-pool-reset`, `R6-pool-week`, `R6-pool-spent`), a second reset in a week by name
    (`R6-pool-reset-week`), and a pool in another form refuses the verifier's inputs."""
    s, sampling_design = POOLED, {"k": 1, "law": "subset"}
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    pool = {"genesis": GENESIS, "lag": 1, "pin": "pin", "pout": "pout", "reset": False, "verdicts": {}, "week": 0}
    params = {"anchors": [], "delta": "1/10", "law": sampling_design, "leaf_layer": LEAF, "log": [], "parameters": s.signature,
              "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "pool": pool,
              "registered": {"pin": 16, "pout": 16}, "scheme": SCHEME, "window_root": EARLIER_WROOT, "work_table": None}

    def record(window, pout, pin=None, log=(), reset=False, week=0, **marks):
        """The record after `log`, its window naming the pool fields of the verifier's reset `reset` in week `week`,
        then `marks`."""
        honest = {R.POOL_RESET: reset, R.POOL_WEEK: week,
                  R.POOL_SPENT: reset or R.pool_spent_before({"week": week}, list(log))}
        return DK.record(s.program, s.query, sampling_design, SCHEME, roots, {**window, **honest, **marks}, leaf_layer=LEAF,
                        previous=DD.digest(log[-1]) if log else None, registered=[pout, *([pin] if pin else [])])

    def codes(p, rec):
        return _codes_of(_json({"params": p, "record": rec}), pieces, callers)

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
    for how in ({"verdicts": {DD.digest(first): False}}, {"reset": True}):
        restart, reset = {**then, "pool": {**pool, **how}}, how.get("reset", False)
        assert codes(restart, record(WINDOW, pout1, empty, [first], reset)) == [], how
        assert codes(restart, record(WINDOW, pout1, carried, [first], reset)) == [
            "R7-domain", "R7-supplied", "R7-pool-chain"], how
    for verdicts, lag in (({DD.digest(first): True}, 1), ({GENESIS: False}, 1), ({DD.digest(first): False}, 2)):
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


def test_a_start_is_carried_only_past_a_record_the_verdict_log_accepted(pieces, callers):
    """Lean and Python agreeing on the caller's verdicts: with the verifier's verdict log (`verdicts`, the pool's form),
    `W` read from an earlier record's port carries where the log names that record accepted or doesn't name it, and is
    `R7-output-refused` where it names it refused, in the history form and the explicit one alike, after SPOT's R7
    codes; `W` reusing that record's registered value is not held to it, and a log in another form refuses the
    verifier's inputs."""
    s, values, sampling_design = REGISTERED_SETUPS[0], {"D": 1, "W": 3}, {"k": 1, "law": "subset"}
    output = {"root": "c1" * 64, "domain": EARLIER_OUT_DOMAIN}
    w = s.committed["W", 3].entry()
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    log = [DK.record(s.program, s.query, sampling_design, SCHEME, [*roots, {"port": EARLIER_OUT, "leaves": w["leaves"], **output}],
                    EARLIER, leaf_layer=LEAF, registered=[{**w, "value": EARLIER_VALUE, **output}])]
    d = DD.digest(log[0])
    rec = _record_of(s, values, sampling_design, WINDOW, log, output=output, fixed=["W"], leaf_layer=LEAF, previous=d)
    params = {"anchors": [], "delta": "1/10", "law": sampling_design, "leaf_layer": LEAF, "log": log, "parameters": s.signature,
              "partition": None, "ports": s.ports, "program": s.desc, "query": s.query, "registered": values,
              "scheme": SCHEME, "work_table": None}

    def codes(outputs, verdicts=(), r=rec):
        """The codes with `outputs` and the verdict log `verdicts`, or none where it is `()`."""
        p = {**params, "outputs": outputs, **({} if verdicts == () else {"verdicts": verdicts})}
        return _codes_of(_json({"params": p, "record": r}), pieces, callers)

    moved = {**rec, "registered": [{**e, "root": "ef" * 64} if e["value"] == "W" else e for e in rec["registered"]]}
    for out in ({"W": {"port": EARLIER_OUT}}, {"W": {"record": d, "port": EARLIER_OUT}}):
        for verdicts in ((), {}, {d: True}, {"ee" * 64: False}):
            assert codes(out, verdicts) == [], (out, verdicts)
        assert codes(out, {d: False}) == ["R7-output-refused"], out
        assert codes(out, {d: False}, moved) == ["R7-held", "R7-output-refused"], out
    assert codes({"W": {"record": d, "value": EARLIER_VALUE}}, {d: False}) == []
    for bad in ([], {d: 0}, {d.upper(): False}, {d[:-2]: False}, None, "x"):
        assert codes({"W": {"port": EARLIER_OUT}}, bad) == "reject", bad


def test_an_arrival_cut_at_a_windows_edge_continues_from_the_pool_it_left(pieces, callers):
    """Lean and Python agreeing on cut and continue over three windows. Each window's certified arrivals are the root the
    network attester signed for it, which the verifier pins (`ing`: the arrivals open at the last window's edge first,
    each at the values it arrived with and the outputs released of it); its rows (`req`) are its own, under its open
    root; and its pool chains. Three windows that carry the pool are accepted, arrivals with a continuation counted
    from the outputs serving computed or the last window's arrivals again are `R7-held`, and a window that restarts its
    pool, by the public rule or by serving's reset, is accepted from the empty pool alone, still serving the
    continuations, its carried pool refused by name."""
    s, sampling_design = CONTINUED, {"k": 1, "law": "subset"}
    roots = [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports]
    windows, wroots = (EARLIER, WINDOW, {**WINDOW, "row": 102}), (EARLIER_WROOT, WROOT, "a8" * 64)

    def stamp(rows, tag):
        salts = [hashlib.shake_256(f"ing/{tag}/{r}".encode()).digest(192) for r in range(len(rows))]
        return RV.Registered.commit(s.sha512, "ing", rows, salts).entry()

    stamps = [stamp(rows, i) for i, rows in enumerate((
        [[1, 7, 5], [1, 8, 2], [1, 9, 3], [0, 0, 0]], [[1, 7, 8], [1, 9, 3], [1, 10, 1], [0, 0, 0]],
        [[1, 9, 4], [1, 10, 1], [0, 0, 0], [0, 0, 0]]))]

    def params(i, log, **how):
        pool = {"genesis": GENESIS, "lag": 1, "pin": "pin", "pout": "pout", "reset": False, "verdicts": {}, "week": 0}
        return {"anchors": [], "delta": "1/10", "law": sampling_design, "leaf_layer": LEAF, "log": list(log),
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
            return DK.record(s.program, s.query, sampling_design, SCHEME, roots, {**windows[i], **marks},
                            leaf_layer=LEAF, previous=DD.digest(log[-1]) if log else None,
                            registered=[ing or stamps[i], *s.entries({"req": 12, "pout": 16}, wroots[i]), *extra])

        if pin is None:
            pin = G.supplied_value(s.program, "pin", [[("bytes", GENESIS)]], bare(), ["pin"]).entry()
        return bare(pin)

    def carried(rec):
        return {**next(e for e in rec["registered"] if e["value"] == "pout"), "value": "pin"}

    def codes(p, rec):
        return _codes_of(_json({"params": p, "record": rec}), pieces, callers)

    first = record(0, None)
    second = record(1, carried(first), [first])
    third = record(2, carried(second), [first, second])
    for i, rec in enumerate((first, second, third)):
        assert codes(params(i, [first, second][:i]), rec) == [], i
    served = stamp([[1, 7, 9], [1, 9, 3], [1, 10, 1], [0, 0, 0]], "served")
    for ing in (served, stamps[0]):
        assert codes(params(1, [first]), record(1, carried(first), [first], ing)) == ["R7-held"], ing
    assert codes(params(1, [first]), record(1, None, [first])) == ["R7-domain", "R7-held", "R7-pool-chain"]
    for how in ({"verdicts": {DD.digest(second): False}}, {"reset": True}):
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
    """R3 over a window of 4096 ports, the shape whose quadratic walk `Docket.portIndex` replaced, Lean and Python
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
    assert codes(lambda rs: rs.append({**rs[1000], "root": "cd" * 64})) == ["R3-ports"]
    assert codes(lambda rs: rs.insert(0, {**rs[1000], "root": "cd" * 64})) == ["R3-ports"]
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


def _reported(a: dict, v: str) -> dict:
    """The reads a v2 audit's verdict of record reports, changed, by name: all of them (`all`); with a value the
    docket doesn't name (`unnamed`); less `v`'s (`unread`); none (`empty`) or no report (`none`); one of `v`'s
    ports at another root (`one-port`) or all of them (`another-root`); with a root port's value at its own root
    (`root-port`) or at another (`root-port-elsewhere`)."""
    opened = a["evidence"]["verdicts"]["lean"]["registered"]
    p = next(k for k in sorted(opened) if opened[k]["value"] == v)
    return {"all": opened, "unnamed": {**opened, "extra.0": {"root": "ab" * 64, "value": "extra"}},
            "unread": {k: e for k, e in opened.items() if e["value"] != v}, "empty": {}, "none": None,
            "one-port": {**opened, p: {**opened[p], "root": "12" * 64}},
            "another-root": {k: {**e, "root": "12" * 64} if e["value"] == v else e for k, e in opened.items()},
            "root-port": {**opened, "x.0": {"root": ROOTS["x"], "value": "x"}},
            "root-port-elsewhere": {**opened, "x.0": {"root": "12" * 64, "value": "x"}}}


def _reporting(a: dict, got) -> dict:
    lean = {**a["evidence"]["verdicts"]["lean"], "registered": got}
    return {**a, "evidence": {**a["evidence"], "verdicts": {"lean": lean}}}


def test_a_v2_audit_reads_the_sessions_registered_roots(server):
    """A v2 audit of each program accepts when each read the verifier reports of the record's own values, less a root
    port's read at that port's own root (its own), is of a registered value, opened against its registered root: every
    value's reads, a value's reads missing, none reported, and a value the docket doesn't name beside them (not
    its to read). It rejects a read opened against another root, by one port of a value or by all of them, and a root
    port's read at another root. Where the docket also registers a value under a root port's name, a read at the
    port's root in place of one of that value's is rejected: the value's reads no longer agree."""
    for s in REGISTERED_SETUPS:
        a = _v2_audit(s, s.values[1])
        v = sorted(e["value"] for e in a["record"]["registered"])[0]
        own = "reject" if "x" in s.values[1] else "accept"
        for name, want in (("all", "accept"), ("unnamed", "accept"), ("unread", "accept"), ("empty", "accept"),
                           ("none", "accept"), ("one-port", "reject"), ("another-root", "reject"), ("root-port", own),
                           ("root-port-elsewhere", "reject")):
            case = _json(_case(_reporting(a, _reported(a, v)[name])))
            assert D.replay(ONE_STAGE, case, server)[0] == want, (s.program.fn.id, name)


def test_the_record_fails_a_read_not_of_a_registered_root_and_no_missing_read(pieces):
    """The record of a v2 audit of each program, Lean's and Python's the same: accepted, with no `registered` check,
    when a registered value has no reported read, by its reads missing or none reported, or a root port's one read is at
    that port's own root; and failing `registered`, naming the value read, when all of a value's reads are opened
    against another root, a root port's read is at another root, or a root port's read at its own root disagrees with
    the reads of a value registered under its name."""
    for s in REGISTERED_SETUPS:
        a = _v2_audit(s, s.values[1])
        named = sorted(e["value"] for e in a["record"]["registered"])
        v = named[0]
        own = ["x"] if "x" in s.values[1] else None
        for name, wrong, opened in (("unread", None, None), ("empty", None, None), ("none", None, None),
                                    ("another-root", [v], named), ("root-port", own, named),
                                    ("root-port-elsewhere", ["x"], sorted({*named, "x"}))):
            case = _record_case(_reporting(a, _reported(a, v)[name]), str(Fraction(0.1)))
            kind, line = D.replay(RECORD_PORT, case, pieces)
            assert kind == "output", line
            rec = json.loads(json.loads(line))
            assert codec.canonical_json(rec["checks"]) == codec.canonical_json(audit_record_of(copy.deepcopy(case))["checks"])
            assert rec["accepted"] is (wrong is None), (s.program.fn.id, name)
            assert rec["checks"].get("registered") == (None if wrong is None else {
                "ok": False, "codes": ["registered"],
                "detail": {"registered": named, "opened": opened, "wrong": wrong}}), (s.program.fn.id, name)


def test_the_record_holds_a_certified_value_to_a_reported_read(pieces):
    """The record of a v2 audit of each program whose verifier holds a certified entry for one of its registered values,
    Lean's and Python's the same: accepted when the verdict of record reports a read of the value at its root; failing
    `certified`, naming the value, when it reports none: its reads missing, none reported, or its ports reported under a
    value the docket does not list and at another root, a read the docket's own names drop. The same
    records without the certified names are accepted: a registered value needs no reported read."""
    for s in REGISTERED_SETUPS:
        a = _v2_audit(s, s.values[1])
        v = sorted(e["value"] for e in a["record"]["registered"])[0]
        got = _reported(a, v)
        renamed = {k: {"root": "12" * 64, "value": "unlisted"} if e["value"] == v else e for k, e in got["all"].items()}
        for name, reads, unread in (("all", got["all"], None), ("unread", got["unread"], [v]), ("empty", {}, [v]),
                                    ("none", None, [v]), ("renamed", renamed, [v])):
            bare = _record_case(_reporting(a, reads), str(Fraction(0.1)))
            for case in ({**bare, "certified": [v]}, bare):
                kind, line = D.replay(RECORD_PORT, case, pieces)
                assert kind == "output", line
                rec = json.loads(json.loads(line))
                assert codec.canonical_json(rec["checks"]) == codec.canonical_json(
                    audit_record_of(copy.deepcopy(case))["checks"])
                failed = "certified" in case and unread is not None
                assert rec["accepted"] is not failed, (s.program.fn.id, name)
                assert rec["checks"].get("certified") == ({
                    "ok": False, "codes": ["certified"], "detail": {"certified": [v], "unread": unread}}
                    if failed else None), (s.program.fn.id, name)


#: A window value's root, committed in the record's window and in no root or registered entry of it.
WINDOW_VALUE_ROOT = "5d" * 64
#: The setups whose values read nothing of the record's window, so a window that also commits a window value's entry
#: registers them as it did.
WINDOW_FREE_SETUPS = [s for s in REGISTERED_SETUPS if '"window"' not in json.dumps(getattr(s, "sup", {}))]


def _window_value_audit(s: RegisteredSetup, named: list, entry: dict | None, reads: dict) -> dict:
    """`_v2_audit` of `s` whose verifier names `named` window values (none: no `window_values`), whose record's window
    holds `entry` in its `activations.values` under `st` (none: no `activations`), and whose verdict also reports
    `reads`."""
    a = _v2_audit(s, s.values[0])
    rec = a["record"]
    if entry is not None:
        rec["window"] = {**rec["window"], "activations": {"values": {"st": entry}}}
    if named is not None:
        a["params"]["window_values"] = named
    lean = a["evidence"]["verdicts"]["lean"]
    a["evidence"] = {**a["evidence"], "receipt": R.receipt(rec, "v", received=RECEIVED),
                     "verdicts": {"lean": {**lean, "registered": {**lean["registered"], **reads}}}}
    return _json(a)


def test_a_named_window_value_is_read_at_its_entrys_root(server):
    """A window value the verifier names is held to the root of its entry in the record's `window.activations.values`:
    a read at that root accepts, as does no read of it; a read at another root, under two roots, or of a named value
    with no entry, or one whose entry names another value, rejects. Named nowhere, the same reads accept as on the
    base, which reads none of a value the docket doesn't name. A name that is a port or a value the verifier
    expects registered is refused with the parameters."""
    at = lambda root: {"st.0": {"root": root, "value": "st"}}  # noqa: E731
    entry = {"root": WINDOW_VALUE_ROOT, "value": "st"}
    for s in WINDOW_FREE_SETUPS:
        for named, ent, reads, want in (
                (["st"], entry, at(WINDOW_VALUE_ROOT), "accept"),
                (["st"], entry, {}, "accept"),
                (["st"], entry, at("12" * 64), "reject"),
                (["st"], entry, {**at(WINDOW_VALUE_ROOT), "st.1": {"root": "12" * 64, "value": "st"}}, "reject"),
                (["st"], None, at(WINDOW_VALUE_ROOT), "reject"),
                (["st"], {**entry, "value": "other"}, at(WINDOW_VALUE_ROOT), "reject"),
                (["st"], {"value": "st"}, at(WINDOW_VALUE_ROOT), "reject"),
                ([], entry, at("12" * 64), "accept"),
                (None, entry, at("12" * 64), "accept"),
                (None, None, at("12" * 64), "accept")):
            kind, line = D.replay(ONE_STAGE, _json(_case(_window_value_audit(s, named, ent, reads))), server)
            assert kind == want, (s.program.fn.id, named, ent, reads, line)
        port, value = sorted(s.ports)[0], sorted(s.values[0])[0]
        for named in ([port], [value], ["st", port], "st", [5]):
            kind, line = D.replay(ONE_STAGE, _json(_case(_window_value_audit(s, named, entry, at(WINDOW_VALUE_ROOT)))),
                                  server)
            assert kind != "accept" and "window value" in line, (s.program.fn.id, named, line)


def test_a_named_window_value_at_another_named_values_root_is_refused(server):
    """With two window values named, each with its own entry, a read of one at the other's entry's root rejects: each
    named value is held to its own entry, never to any named entry's root."""
    a_root, b_root = WINDOW_VALUE_ROOT, "6e" * 64
    entries = {"st": {"root": a_root, "value": "st"}, "su": {"root": b_root, "value": "su"}}
    for s in WINDOW_FREE_SETUPS:
        for reads, want in (({"st.0": {"root": a_root, "value": "st"}, "su.0": {"root": b_root, "value": "su"}}, "accept"),
                            ({"st.0": {"root": b_root, "value": "st"}}, "reject"),
                            ({"su.0": {"root": a_root, "value": "su"}}, "reject")):
            a = _window_value_audit(s, ["st", "su"], None, reads)
            a["record"]["window"] = {**a["record"]["window"], "activations": {"values": entries}}
            a["evidence"]["receipt"] = R.receipt(a["record"], "v", received=RECEIVED)
            kind, line = D.replay(ONE_STAGE, _json(_case(a)), server)
            assert kind == want, (s.program.fn.id, reads, line)


def _unnamed(j):
    """`j` naming no window value (`window_values` empty), where it is an object."""
    return {**j, "window_values": []} if isinstance(j, dict) else j


def _base_held(verdicts, docket, window_values=()):
    """The registered reads `audit_record` held before window values: the docket's own, less its ports' at their
    own roots."""
    return audit.less_own_reads(audit.session_registered(verdicts, docket), docket)


#: The refusals whose text names the keys they read, which a `window_values` added to keys already wrong joins.
KEYS_REFUSED = ("params: the keys of the parameters are not ", "the keys of the record's input are not ")


def _reason(port, got):
    return json.loads(got[1]).get("reason") if port is ONE_STAGE and got[0] != "error" else got[1]


def _naming_window_values(reason: str) -> str:
    """A keys refusal (`KEYS_REFUSED`) as it reads with `window_values` among the keys."""
    head = next(h for h in KEYS_REFUSED if reason.startswith(h))
    return head + ", ".join(sorted({*reason.removeprefix(head).split(", "), "window_values"}))


def _same(port, got, unnamed, ok, changed: bool) -> None:
    """`unnamed`'s outcome is `got`'s, byte for byte, but where naming no window value `changed` the input and `got`
    refused the keys it read: then it refuses them with `window_values` among them."""
    reason = _reason(port, got)
    if changed and got[0] in ("reject", "error") and isinstance(reason, str) and reason.startswith(KEYS_REFUSED):
        want = _naming_window_values(reason)
        if port is ONE_STAGE and got[0] != "error":
            assert unnamed[0] == got[0] and json.loads(unnamed[1]) == {**json.loads(got[1]), "reason": want}, (
                ok, got, unnamed)
        else:
            assert unnamed == (got[0], want), (ok, got, unnamed)
    else:
        assert unnamed == got, (ok, got, unnamed)


def _same_keys_refusal(port, lean, python, ok) -> None:
    """Where Lean refuses the keys it read (`KEYS_REFUSED`), the mirror refuses them in the same words."""
    reason = _reason(port, lean)
    if lean[0] in ("reject", "error") and isinstance(reason, str) and reason.startswith(KEYS_REFUSED):
        assert python[0] == lean[0] and _reason(port, python) == reason, (ok, lean, python)


def test_naming_no_window_value_changes_no_outcome(server, pieces):
    """Naming no window value changes nothing, reason included: on each sampled one-stage case and record-piece input of
    the agreement tests, the mirror with the key absent gives the line it gives holding the reads as before window
    values (`_base_held`), and Lean and the mirror with `window_values` empty give the lines they give with it absent,
    but where the input's keys were already refused, the same refusal with `window_values` among the keys, in the
    mirror's words as in Lean's."""
    import hypothesis
    seen: set[str] = set()

    @hypothesis.settings(derandomize=True, database=None, deadline=None, max_examples=300,
                         suppress_health_check=[hypothesis.HealthCheck.too_slow])
    @hypothesis.given(CASES, RECORD_PORT.inputs)
    def same(case, j):
        named = {**case, "params": _unnamed(case["params"])} if isinstance(case, dict) and "params" in case else case
        with mock.patch.object(audit, "held_reads", _base_held):
            base, base_j = ONE_STAGE.python(case), RECORD_PORT.python(j)
        assert ONE_STAGE.python(case) == base and RECORD_PORT.python(j) == base_j
        lean = ONE_STAGE.lean(server, case)
        seen.add(lean[0])
        changed = named != case
        _same(ONE_STAGE, lean, ONE_STAGE.lean(server, named), case, changed)
        _same(ONE_STAGE, base, ONE_STAGE.python(named), case, changed)
        _same_keys_refusal(ONE_STAGE, lean, base, case)
        lean_j = RECORD_PORT.lean(pieces, j)
        _same_keys_refusal(RECORD_PORT, lean_j, base_j, j)
        _same(RECORD_PORT, lean_j, RECORD_PORT.lean(pieces, _unnamed(j)), j, _unnamed(j) != j)
        _same(RECORD_PORT, base_j, RECORD_PORT.python(_unnamed(j)), j, _unnamed(j) != j)
        seen.add(f"record {base_j[0]}")

    same()
    assert {"accept", "reject", "record output", "record reject"} <= seen, seen


def _accepted_record_case(kind: str, delta: float) -> dict:
    s = SETUPS[3]
    sampling_design = {"law": "bernoulli", "p": "1/2"} if kind == "bernoulli" else {"k": 3, "law": kind}
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}} if kind == "work" else None
    mine = DR.derive(s.program, s.query, sampling_design, table)
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
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
    `audit_record`'s, a Bernoulli sampling design's bound on incorrect proof blocks the exact one (`Bernoulli.count_bound`)."""
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
    sampling_design = ({"law": "bernoulli", "p": p} if kind == "bernoulli" else {"k": s.population, "law": "subset"} if kind == "all"
           else {"subset": f"rate:3/{s.population}", "stratified": "stratified-rate:1/2", "work": "work-rate:1/2"}[kind])
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}}
    table = {t: w for t, w in table.items() if t in s.templates} if kind == "work" else None
    named, expanded = service.params_sampling_design(sampling_design, s.program, s.query, table)
    mine = DR.derive(s.program, s.query, named, expanded) if kind != "bernoulli" else {"k": 3}
    rec = DK.record(s.program, s.query, named, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, work_table=expanded)
    params = {"anchors": [], "delta": "1/10", "law": sampling_design, "leaf_layer": LEAF, "log": [], "partition": None, "ports": s.ports,
              "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME, "work_table": table}
    instances = sum(st_["k"] for st_ in mine["strata"]) if "strata" in mine else mine["k"]
    evidence = {"session_roots": dict(ROOTS), "statement": {"instances": instances},
                "verdicts": {"lean": {"accepted": True, "digest": "aa" * 64, "of_record": True},
                             "upstream": {"accepted": True, "of_record": False}}}
    return _json({"evidence": evidence, "params": params, "record": rec})


def test_the_run_reaches_every_step(pieces):
    """A docket's run under each sampling design, over one unit's population and over three, and a v2 docket's, Lean's
    the Python's whole: each accepts, on the sample from the challenger's coins (`_from_challenger`), showing the
    verdicts' public outputs. The same runs refused at each step (a declared item missing, a record R1-R7 refuses, no
    session, a verdict of record about other sessions) and rejected (a verdict that ran and did not accept), which the
    sampled runs reach only probably. A run that read coins is refused at its select step without their evidence, with
    empty evidence, without the family the stream read them from or under another, without the challenger's key or
    under another than the session is signed under, on another copy of main's history than the evidence carries, and
    on evidence that names a key of its own (`coins`): the provenance check is never skipped, and it is under the key
    and the history the verifier holds. Evidence that holds but for a root other than the docket's digest in the family, before the
    sample's coins or after them, is refused for that (`family`): a family's coins follow one docket. A run asked
    to sample from the verifier's own coins is refused before it starts: the verifier samples no coins of its own."""
    v2 = _v2_audit(REGISTERED_SETUPS[0], REGISTERED_SETUPS[0].values[1])
    v2["evidence"]["verdicts"]["upstream"] = {"accepted": True, "of_record": False}
    seen = []
    sampling_designs = ("subset", "stratified", "work")
    for i, a in enumerate([*(_v1_audit(k) for k in sampling_designs), *(_v1_audit(k, SETUPS[2]) for k in sampling_designs), v2]):
        _, case = _from_challenger(_run_case(a, "", 2, ["credited"]), random.Random(f"run/{i}"))
        other = {**case["verdicts"]["lean"], "sessions": RUN_SESSIONS[::-1]}
        for changed in (case, {**case, "public": case["public"][1:]},
                        {**case, "record": {**case["record"], "previous": "11" * 64}}, {**case, "sessions": []},
                        {**case, "verdicts": {**case["verdicts"], "lean": other}},
                        {**case, "verdicts": {**case["verdicts"], "upstream": {"accepted": False, "of_record": False}}},
                        {k: v for k, v in case.items() if k != "coins"}, {**case, "coins": {}},
                        {**case, "coins": {"segment": None}}, {k: v for k, v in case.items() if k != "family"},
                        {**case, "family": "other"}, {k: v for k, v in case.items() if k != "key"},
                        {**case, "key": {**case["key"], "public": _flip(case["key"]["public"], -1)}},
                        {**case, "main": [*case["main"], "0" * 40]},
                        {**case, "coins": {**case["coins"], "key": case["key"]}}):
            out = run(copy.deepcopy(changed))
            assert D.replay(RUN, changed, pieces) == ("output", D.canonical(out)), (a["params"]["law"], out)
            seen.append(out.get("step") or out["symbol"])
            held = ("coins", "family", "key", "main")
            if [changed.get(k) for k in held] != [case[k] for k in held]:
                assert out["codes"] == ["coins"], out
        for shape in ("before", "later"):
            _, again = _from_challenger(_run_case(a, "", 2, ["credited"]), random.Random(f"run/{i}/{shape}"),
                                        **{shape: True})
            out = run(copy.deepcopy(again))
            assert D.replay(RUN, again, pieces) == ("output", D.canonical(out)), (a["params"]["law"], shape, out)
            assert (out.get("step"), out.get("codes")) == ("select", ["family"]), (a["params"]["law"], shape, out)
        kind, why = D.replay(RUN, {**case, "family": 5}, pieces)
        assert kind == "reject" and "family is not a string" in why, (kind, why)
        with pytest.raises(D.Reject, match="family is not a string"):
            run({**copy.deepcopy(case), "family": 5})
        assert run(copy.deepcopy(case))["public"] == {"credited": 12} and run(copy.deepcopy(case))["read"] > 0
        for own in ("verifier", "os", ""):
            kind, why = D.replay(RUN, {**case, "coin_source": own}, pieces)
            assert kind == "reject" and "the verifier draws no coins of its own" in why, (own, kind, why)
    assert seen == ["accept", "register", "register", "deliver", "outcome", "reject", *["select"] * 9] * 7


def test_the_designs_rows_place_each_callers_units(pieces):
    """The design's links and rows, Lean's the Python's whole, derived once from the replay block's Definition
    for each template port, its callers' units in body order: `BLOCK`'s PAIR batch reads its member's row of `x` and
    writes its row of `o0`, three a block; PAIR called over each of two rows reads row `j` at unit `j`, or row `1 - j`
    the other way round, and writes row `j` of the output; ID8 over the second PAIR's output reads PAIR's unit 1 of two,
    its own output a field of one row; ID8 over DUO's second output reads it at DUO's `o1`. PAIR's callers reading two
    parameters, or a slice not one of its rows, or one the parameter and the other SWAP's output, make no links, and
    their rows are then only those of the others. DUO over each of two rows, returned one field of each, links
    that field's port (`o<f>`), each field returned its own; its first output at one caller and its second at the other
    links neither. A token select inlined over two tokens, its combine gathering one output of each of a token's two
    chunks, has links that read the logits at two row widths, the chunk's four rows a block and, for logits[0], the
    combine's eight; by units, the combine reading each output of each of a token's two units whole, each unit of the
    combine reads its token's units' rows, the logits at the unit's width alone; by chunk position, each chunk a
    Definition of its own, every template reads its own row of its token's three."""
    pair, id8, duo, merge, unit, join, swap, keep, last = (
        codec.descriptor_id(d) for d in (PAIR, ID8, DUO, MERGE, UNIT, JOIN, SWAP, KEEP, LAST))
    (pairs, crossed, fan, field, apart, askew, members, kinds, gathered, heads, tails, halves, uneven, select,
     units, places) = MULTI_REPLAYS

    def link(t, port, value):
        return {"port": port, "template": t, "value": value}

    def rows(*rs):
        return [{"per": per, "port": port, "runs": runs, "template": t} for t, port, per, runs in rs]
    one, two = [[0, 1, 0, 0]], [[0, 2, 0, 1]]
    cases = [
        (BLOCK, [link(pair, "x", "x"), link(pair, "o0", "o0")],
         rows((pair, "x", 3, [[0, 3, 0, 1]]), (pair, "o0", 3, [[0, 3, 0, 1]]))),
        (pairs, [link(pair, "x", "x"), link(pair, "o0", "o0")], rows((pair, "x", 2, two), (pair, "o0", 2, two))),
        (crossed, [link(pair, "x", "x"), link(pair, "o0", "o0")],
         rows((pair, "x", 2, [[0, 1, 1, 0], [1, 1, 0, 0]]), (pair, "o0", 2, two))),
        (fan, [link(pair, "x", "x"), link(pair, "o0", "o0"), link(id8, "o0", "o1")],
         rows((pair, "x", 2, two), (pair, "o0", 2, two), (id8, "o0", 1, one), (id8, "y", 2, [[0, 1, 1, 0]]))),
        (field, [link(duo, "x", "x"), link(id8, "o0", "o0")],
         rows((duo, "x", 1, one), (id8, "o0", 1, one), (id8, "y", 1, one))),
        (apart, None, []),
        (askew, None, []),
        (members, [link(pair, "x", "x"), link(id8, "o0", "o0")], rows((pair, "x", 2, two), (id8, "o0", 2, two))),
        (kinds, None, []),
        (gathered, [link(id8, "y", "x"), link(pair, "o0", "o0")], rows((id8, "y", 2, two), (pair, "o0", 1, one))),
        (heads, [link(duo, "x", "x"), link(duo, "o0", "o0")], rows((duo, "x", 2, two), (duo, "o0", 2, two))),
        (tails, [link(duo, "x", "x"), link(duo, "o1", "o0")], rows((duo, "x", 2, two), (duo, "o1", 2, two))),
        (halves, [link(duo, "x", "x"), link(duo, "o0", "o0"), link(duo, "o1", "o1")],
         rows((duo, "x", 2, two), (duo, "o0", 2, two), (duo, "o1", 2, two))),
        (uneven, None, []),
        (select, [link(duo, "x", "x"), link(merge, "x0", "x"), link(merge, "o0", "o0")],
         rows((duo, "x", 4, [[0, 4, 0, 1]]), (merge, "x0", 8, [[0, 2, 0, 4]]), (merge, "o0", 2, two))),
        (units, [link(unit, "x", "x"), link(join, "o0", "o0")],
         rows((unit, "x", 4, [[0, 4, 0, 1]]), (join, "o0", 2, two), (join, "x0", 4, [[0, 2, 0, 2]]),
              (join, "best0", 4, [[0, 2, 0, 2]]), (join, "best1", 4, [[0, 2, 1, 2]]),
              (join, "idx0", 4, [[0, 2, 0, 2]]), (join, "idx1", 4, [[0, 2, 1, 2]]))),
        (places, [link(swap, "x", "x"), link(keep, "x", "x"), link(last, "x", "x"), link(last, "o0", "o0")],
         rows((swap, "x", 6, [[0, 2, 0, 3]]), (keep, "x", 6, [[0, 2, 1, 3]]), (last, "x", 6, [[0, 2, 2, 3]]),
              (last, "o0", 2, two), (last, "r0", 2, two), (last, "r1", 2, two))),
    ]
    for r, links, want in cases:
        c = {"program": _json(codec.encode_program(PN.population_program(r, 2))), "replay": codec.descriptor_id(r),
             "boundary": None}
        for port, mirror, expected in ((REPLAY_LINKS, replay_links, links), (REPLAY_ROWS, replay_rows, want)):
            got = mirror(copy.deepcopy(c))
            assert got == expected and D.replay(port, c, pieces) == ("output", D.canonical(got)), (c["replay"], got)


def test_a_windows_run_from_the_challengers_coins(pieces):
    """A window of two templates' run (`SETUPS[3]`: PAIR and TRIPLE, one stratum each under `stratified-rate:1/2`) sampling from
    the auditor's challenger, as a `Stream` over a window runs it: its coins a ledger session's challenger opened after
    the docket's digest and read by the selection rule's sample (`Drawn`), Lean's `Run.run` the Python's whole. The
    honest run is accepted, each stratum sampled at least its floor. It is refused at its docket for a window of
    other counts (`R2-program`) or the templates in the other order (`query`), and for a record whose sampling design samples a
    stratum short of its floor (`R1-law`); at its select (`coins`) for each mutation of the coins that changes the
    segment or the bytes, coins opened after another digest, coins of a session the ledger doesn't open withheld,
    evidence for another window, and none or empty; and at its select (`family`) for a session whose family holds
    another root, before the sample's coins or after them, on evidence that holds but for that."""
    rule, s = service.SamplingDesign("stratified-rate:1/2"), SETUPS[3]
    a = _v1_audit("stratified", s)
    rec = a["record"]
    named = service.selection_sampling_design(rule, rec["population"], service.selection_sizes(rule, s.program, s.query))
    sampling_design = DR.derive(s.program, s.query, named,
                    service.selection_table(rule, s.program, s.query, None))

    def reads(coins):
        got = bytearray()

        def source(n):
            b = coins(n)
            got.extend(b)
            return b
        service.select_draw(rule, sampling_design, rec["population"], source)
        return bytes(got)

    def case_of(d: Drawn, segment: dict, read: bytes, **change) -> dict:
        c = _run_case(a, read.hex(), 2, ["credited"])
        c["params"] = {**c["params"], "window_root": d.ledger.window["root"]}
        return {**c, "coins": d.evidence(segment), "family": "draw", **d.held(), **change}

    def ran(c: dict) -> dict:
        out = run(copy.deepcopy(c))
        assert D.replay(RUN, c, pieces) == ("output", D.canonical(out)), out
        return out

    for seed in range(3):
        rng = random.Random(f"window/{seed}")
        d = Drawn(rng, after=bytes.fromhex(DD.digest(rec)), reads=reads)
        honest = case_of(d, d.segment.to_json(), d.read)
        out = ran(honest)
        assert out["symbol"] == "accept" and out["read"] == len(d.read), out
        for stratum in sampling_design["strata"]:
            drawn = [u for u in out["draw"]["units"] if any(lo <= u < hi for lo, hi in stratum["units"])]
            assert len(drawn) == stratum["k"] >= 1, (stratum, out["draw"])
        other_counts = json.loads(_population(((0, 3), (1, 2))))
        reordered = Setup(PN.mixed_population_program([(TRIPLE, 3), (PAIR, 2)]), P.template_instances_query([TRIPLE.id, PAIR.id]))
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
        selects = [case_of(d, _mutated(rng, d, name)[0], d.read) for name in SAMPLE_MUTATIONS
                   if name not in ("before", "unread", "read-tampered", "other-after")]
        selects.append(case_of(d, d.segment.to_json(), bytes(tampered)))
        elsewhere = Drawn(rng, reads=reads)
        bare = Drawn(rng, after=bytes.fromhex(DD.digest(rec)), reads=reads, withheld=False)
        selects += [case_of(elsewhere, elsewhere.segment.to_json(), elsewhere.read),
                    case_of(bare, bare.segment.to_json(), bare.read),
                    {**honest, "params": {**honest["params"], "window_root": d.other_window(rng)}},
                    {**honest, "coins": None}, {**honest, "coins": {}}, {**honest, "family": None},
                    {**honest, "family": "other"}, {**honest, "key": None},
                    {**honest, "key": {**honest["key"], "public": _flip(honest["key"]["public"], -1)}},
                    {**honest, "main": [c for c in honest["main"] if c != d.ledger.opened["pinned"]]},
                    {**honest, "main": [*honest["main"], "0" * 40]},
                    {**honest, "coins": {**honest["coins"], "key": honest["key"]}}]
        for c in selects:
            out = ran(c)
            assert (out.get("step"), out.get("codes")) == ("select", ["coins"]), out
        for shape in ("before", "later"):
            again = Drawn(rng, after=bytes.fromhex(DD.digest(rec)), reads=reads, **{shape: True})
            out = ran(case_of(again, again.segment.to_json(), again.read))
            assert (out.get("step"), out.get("codes")) == ("select", ["family"]), (shape, out)
            if shape == "before":
                out = ran(case_of(again, _mutated(rng, again, "before")[0], again.read))
                assert out.get("step") == "select" and out.get("codes") in (["coins"], ["family"]), out


def test_the_audit_reads_its_sample_from_the_challengers_coins(server):
    """A one-stage audit whose verifier names the auditor's challenger (`challenger`, with `window_root`) reads its sample's
    coins through `Challenger.sampleCoins` (`coins.draw_coins`): under the verifier's key, on its window, of its family,
    released after the docket's digest, Lean's the Python's. The sample from those coins is accepted; it is rejected
    for another sample, the evidence's coins carrying a key (`coins_reason`'s form, where the prover named the key), none,
    coins under another key, on another window, of another family, opened after another digest, each mutation of the
    segment, a family whose log shows another root before the sample's coins or after them (`FAMILY_ROOT`), or a session
    the ledger doesn't open withheld (one whose challenger may have released its root before the prover's commitment),
    and refused for a challenger without a window. A verifier that names no challenger decides as before, and rejects
    evidence that carries coins."""
    s, sampling_design = SETUPS[3], {"k": 3, "law": "stratified"}
    mine = DR.derive(s.program, s.query, sampling_design, None)
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF)
    after = bytes.fromhex(DD.digest(rec))

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
        p = {"anchors": [], "challenger": {"key": d.s.key.to_json(), "label": "draw"}, "delta": "1/20", "law": sampling_design,
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
        d = Drawn(rng, after=after, reads=reads)
        coins = d.evidence(d.segment.to_json())
        draw = drawn_from(d.read)
        accepts(audit_of(d, draw, coins))
        other = next(x for x in (drawn_from(rng.randbytes(4 * len(d.read) + 256)) for _ in range(64)) if x["units"] != draw["units"])
        stranger = next(x for x in (Drawn(rng, after=after, reads=reads) for _ in range(16))
                        if x.s.key != d.s.key)
        elsewhere = Drawn(rng, reads=reads)
        rejected = [audit_of(d, other, coins), audit_of(d, draw, {**coins, "key": d.s.key.to_json()}),
                    audit_of(d, draw, None), audit_of(d, draw, coins, challenger={"key": stranger.s.key.to_json(),
                                                                                 "label": "draw"}),
                    audit_of(d, draw, coins, window_root=d.other_window(rng)),
                    audit_of(d, draw, coins, challenger={"key": d.s.key.to_json(), "label": "other"}),
                    audit_of(elsewhere, drawn_from(elsewhere.read),
                             elsewhere.evidence(elsewhere.segment.to_json())),
                    audit_of(d, draw, coins, challenger=None)]
        rejected += [audit_of(d, draw, {**coins, "segment": _mutated(rng, d, name)[0]}) for name in SAMPLE_MUTATIONS
                     if name not in ("before", "unread", "read-tampered", "other-after")]
        for shape in ("before", "later"):
            again = Drawn(rng, after=after, reads=reads, **{shape: True})
            held = again.evidence(again.segment.to_json())
            rejected.append(audit_of(again, drawn_from(again.read), held))
            with pytest.raises(CoinError, match=re.escape(FAMILY_ROOT)):
                draw_coins(again.s.key, again.ledger.window["root"], "draw", after, held, lms_verify)
            if shape == "before":
                rejected.append(audit_of(again, drawn_from(again.read),
                                         {**held, "segment": _mutated(rng, again, "before")[0]}))
        bare = Drawn(rng, after=after, reads=reads, withheld=False)
        bare_coins = bare.evidence(bare.segment.to_json())
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


def test_the_run_refuses_a_bernoulli_sampling_design(pieces):
    """A docket under a Bernoulli sampling design is refused at its docket at any p (`bernoulli`,
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
    """A work docket's run given a closure map, empty or not, is refused at its select step before any coin is read
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
    sampling_design = ({"law": "bernoulli", "p": p} if kind == "bernoulli" else service.selection_sampling_design(service.All(), n) if kind == "all"
           else {"k": 2, "law": "subset"} if kind == "subset" else {"k": 3, "law": "work" if kind == "closure" else kind})
    table = {PAIR.id: 2, TRIPLE.id: {"floor": 1, "work": 1}} if sampling_design["law"] == "work" else None
    mine = DR.derive(s.program, s.query, sampling_design, table)
    rec = DK.record(s.program, s.query, sampling_design, SCHEME, [{"port": p, "root": ROOTS[p], **s.ports[p]} for p in s.ports],
                   WINDOW, leaf_layer=LEAF, work_table=table)
    if sampling_design["law"] == "bernoulli":
        units = {Fraction(0): [], Fraction(1): list(range(n))}.get(Fraction(p), [1, 2, 4])
    elif sampling_design["law"] == "subset":
        units = list(range(sampling_design["k"]))
    else:
        units = sorted(u for st_ in mine["strata"] for u in [u for lo, hi in st_["units"] for u in range(lo, hi)][:st_["k"]])
    d = {**mine, "population": n, "units": units}
    if kind == "closure":
        d["closure"] = [u for u in range(n) if u not in units][:1]
    proved = DR.proved(d)
    evidence = {"draw": d, "instances": len(proved), "proved_units": proved,
                "receipt": R.receipt(rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": True, "of_record": True}}}
    params = {"anchors": [], "delta": "1/20", "law": sampling_design, "leaf_layer": LEAF, "log": [],
              "partition": P.partition(s.program, s.query), "ports": s.ports, "program": s.desc, "query": s.query,
              "registered": None, "scheme": SCHEME, "work_table": table}
    return params, rec, evidence


def test_the_audit_refuses_what_soundness_does_not_cover(server, server_v2):
    """The one-stage audit, v1 (`OneStage.protocol`) and v2 (`OneStageZk`), Lean's the Python's: parameters under a
    Bernoulli sampling design at 0 < p < 1 are refused (`paramsOfJson`), and evidence whose sample carries a closure
    (`evidenceOf`), after its commitment's digest, v2's parameters naming a challenger at the deployed LMS parameters so
    that its refusals of parameters naming none (`NO_CHALLENGER`) or another key (`KEY_PARAMS`) aren't what refuse them;
    the same work audit without the closure is accepted by v1. v1 accepts an honest audit under a `subset`, `stratified`
    or `work` sampling design; v2, whose one statement's soundness statement covers a subset sampling design alone,
    refuses the parameters of the `stratified` and the `work` one by their kind (`subset_only`), and the `subset` ones'
    it doesn't. So v2 refuses the
    work audit that carries the closure at its parameters, and a subset audit whose sample carries one at its evidence."""
    for kind, stage in (("bernoulli", "params: "), ("closure", "evidence: ")):
        params, rec, evidence = _covered_case(kind)
        case = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        out = server.exchange(case)
        assert out == answer(case) and out["verdict"] == "reject", (kind, out)
        assert out["reason"].startswith(stage) and "soundness statement doesn't cover" in out["reason"], (kind, out)
        circuit, public, record = STAND_IN_CIRCUIT, b"not a public file", b"{}"
        base = {k: v for k, v in evidence.items() if k != "verdicts"}
        d = Drawn(random.Random(kind), params=[DEPLOYED])
        v2 = {"blobs": [circuit.hex(), public.hex(), record.hex()], "commitment": _hex(rec),
              "evidence": _hex(base | {"proofs": {"sessions": []}}),
              "params": params | {"challenger": {"key": d.s.key.to_json(), "label": "draw"},
                                  "proofs": {"public": _sha(public), "registered": None},
                                  "window_root": d.ledger.window["root"]}}
        out = server_v2.exchange(_json(v2))
        if kind == "closure":
            assert (out["verdict"], out["reason"]) == ("reject", f"params: {subset_only('work')}"), out
            covered, held, plain = _covered_case("subset")
            sample = plain["draw"]
            widened = {**sample, "closure": [u for u in range(sample["population"]) if u not in sample["units"]][:1]}
            carried = {k: v for k, v in plain.items() if k != "verdicts"} | {"draw": widened, "proofs": {"sessions": []}}
            out = server_v2.exchange(_json({**v2, "commitment": _hex(held), "evidence": _hex(carried),
                                            "params": {**v2["params"], "law": covered["law"], "work_table": None}}))
        assert out["verdict"] == "reject" and out["reason"].startswith(stage), (kind, out)
        assert "soundness statement doesn't cover" in out["reason"], (kind, out)
        if kind == "bernoulli":
            for p in ("0/1", "0/4", "1/1", "3/3", "2/3"):
                sampling_design = {"law": "bernoulli", "p": p}
                one, two = {**case, "params": {**params, "law": sampling_design}}, {**v2, "params": {**v2["params"], "law": sampling_design}}
                out = server.exchange(one)
                assert out == answer(one), (p, out)
                assert out["verdict"] == "reject" and out["reason"] == "params: " + service.BERNOULLI, (p, out)
                out = server_v2.exchange(_json(two))
                assert out["verdict"] == "reject" and out["reason"] == "params: " + service.BERNOULLI, (p, out)
    for kind in ("subset", "stratified", "work", "all"):
        params, rec, evidence = _covered_case(kind)
        case = _json({"commitment": _hex(rec), "evidence": _hex(evidence), "params": params})
        assert D.replay(ONE_STAGE, case, server)[0] == "accept", kind
        two = {**v2, "params": {**v2["params"], "law": params["law"], "work_table": params["work_table"]}}
        out = server_v2.exchange(_json(two))
        if kind in ("stratified", "work"):
            assert (out["verdict"], out["reason"]) == ("reject", f"params: {subset_only(kind)}"), (kind, out)
        else:
            assert not (out["reason"] or "").startswith("params: "), (kind, out)


def test_the_run_takes_a_cover_of_the_units_only_under_a_subset_design(pieces):
    """A docket whose program's units, read stratum by stratum, are not its N units in order but whose strata are a
    disjoint cover of them passes the run's population check under a subset design (`Slots.popCovers`) and is refused
    at its docket under a stratified one (`strata-design`, `OneStageZk.subsetDesign`), Lean's the Python's whole:
    `SETUPS[4]` puts PAIR's units 0, 3 and 4 in one stratum beside TRIPLE's 1 and 2, and `SETUPS[5]`'s Calls put
    PAIR's units 0 and 4 beside the Fork's. Programs whose strata are blocks in order go on under either."""
    source = hashlib.shake_256(b"run").hexdigest(600)
    for s, ok in ((SETUPS[4], False), (SETUPS[5], False), (SETUPS[3], True), (SETUPS[2], True), (SETUPS[6], True)):
        assert P.in_order(s.program, s.query, s.population) is ok and P.covers(s.program, s.query, s.population)
        for kind in ("subset", "stratified") if s is SETUPS[4] or ok and s is SETUPS[3] else ("subset",):
            case = _run_case(_v1_audit(kind, s), source, 2, ["credited"])
            out = run(copy.deepcopy(case))
            assert D.replay(RUN, case, pieces) == ("output", D.canonical(out)), (kind, out)
            if ok or kind == "subset":
                assert out.get("codes") not in (["population-order"], ["strata-design"]), (kind, out)
            else:
                assert (out.get("step"), out.get("codes")) == ("register", ["strata-design"]), (kind, out)
    assert not P.in_order(SETUPS[3].program, SETUPS[3].query, SETUPS[3].population + 1)
    assert not P.covers(SETUPS[3].program, SETUPS[3].query, SETUPS[3].population + 1)


def test_a_two_stage_run_takes_no_cover(pieces):
    """Under a two-stage rule the run takes the units in order only, Lean's the Python's whole: the interleaved layout
    `SETUPS[4]` under a subset design is refused at its grain (`grain`: its query is two templates', and a grain is one
    template's, whose units are in order), before the cover branch's `population-order`; the contiguous `SETUPS[1]`
    passes the population check and goes on to its sample (`coins`: this case carries no coins' evidence)."""
    source = hashlib.shake_256(b"run").hexdigest(600)
    for s, want in ((SETUPS[4], ("register", ["grain"])), (SETUPS[1], ("select", ["coins"]))):
        case = _run_case(_v1_audit("subset", s), source, 2, ["credited"])
        case["rule"] = {"grain": P.template_instance_query(PAIR.id), "k": 1, "replay": f"rate:3/{s.population}",
                        "rule": "two-stage"}
        out = run(copy.deepcopy(case))
        assert D.replay(RUN, case, pieces) == ("output", D.canonical(out)), (s, out)
        assert (out.get("step"), out.get("codes")) == want, out


#: The public files Flock's producers wrote, recorded in `tests/verity/protocols/proof/flock/prover`' fetched fixtures
#: (`artifacts.json`): `verity_flock`'s staging (`json.dumps(sort_keys=True)`), and the selftest's and the
#: recursive links' (compact, sorted).
RECORDED = Path(__file__).resolve().parents[7] / "tests/verity/protocols/proof/flock/prover/fixtures"
RECORDED_FIXTURES = ("forged-proofs", "public-inputs", "public-inputs-blocks", "rec-coins", "rec-links", "zk-sessions")


def _recorded_publics() -> list[Path]:
    missing = [f"tests/verity/protocols/proof/flock/prover/fixtures/{f}" for f in RECORDED_FIXTURES if not (RECORDED / f).is_dir()]
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
