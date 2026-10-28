"""The streamed program digest (`codec.descriptor_chunks`) against the codec before it: the same bytes, digests and refusals.

`old_encode_program` and `old_jcs_domain_error` are the codec's `encode_program` and `jcs_domain_error` before the digest
streamed, verbatim but for their names (the reference, as `test_ports_one_use.OldUses` is the port check's): every Program
here, random ones and the other tests' own, is encoded by both, under both schemas and at several node batch sizes, and the
descriptor's canonical JSON, `program_digest` and `descriptor_size` must agree byte for byte. The pinned digest itself is
`test_format_spec`'s."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import random
from typing import Any

import pytest
from test_running_example import Add16, Mul16

from verity.ml.circuits import codec as C
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.codec import (
    MAX_SAFE_INT,
    SCHEMA_V0,
    SCHEMAS,
    IdentityConflict,
    _as_specialized,
    _encode_definition,
    _program_fn_functions,
    _spec_id,
    _static_functions,
    _Tables,
)
from verity.ml.circuits.defs import (
    REGISTRY,
    CompositeDefinition,
    FunctionDefinition,
    PrimitiveDefinition,
    SpecializedDefinition,
    bind,
)
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

V16 = Value(16)


# -- the reference: the codec before the streamed digest ---------------------------------------------------------------

def old_encode_program(program, schema: str = C.SCHEMA) -> dict:
    if schema not in SCHEMAS:
        raise ValueError(f"unknown descriptor schema {schema!r}")
    tables = None if schema == SCHEMA_V0 else _Tables()
    defs: dict[str, dict] = {}
    seen: dict[str, FunctionDefinition] = {}
    duplicates: list[tuple[str, FunctionDefinition]] = []
    dup_objects: set[tuple[str, int]] = set()
    visited: dict[int, FunctionDefinition] = {}

    def visit(fn: FunctionDefinition) -> None:
        if visited.get(id(fn)) is fn:
            return
        visited[id(fn)] = fn
        fid = _spec_id(fn)
        prev = seen.get(fid)
        if prev is not None:
            if prev is not fn and (fid, id(fn)) not in dup_objects:
                dup_objects.add((fid, id(fn)))
                duplicates.append((fid, fn))
            return
        seen[fid] = fn
        if isinstance(fn, PrimitiveDefinition):
            defs[fid] = _encode_definition(fn)
            return
        assert isinstance(fn, SpecializedDefinition)
        defs[fid] = {}
        for n in fn.body.nodes:
            visit(n.fn)
        for k in fn.defn.statics:
            for f in _static_functions(fn.bindings[k]):
                visit(_as_specialized(f, fid))
        defs[fid] = _encode_definition(fn, schema, tables)

    root = program.root
    visit(root)
    for f in _program_fn_functions(program):
        visit(f)
    for fid, fn in duplicates:
        if _encode_definition(fn) != _encode_definition(seen[fid]):
            raise IdentityConflict(f"E_IDENTITY: two different definitions claim the specialization {fid!r}")
    desc = {
        "schema": schema,
        "root": _spec_id(root),
        "program_fn": getattr(program, "program_fn_id", None) or _spec_id(program.fn),
        "definitions": defs,
        "annotations": dict(program.annotations),
    }
    if tables is not None:
        desc.update(tables.to_json())
    err = old_jcs_domain_error(C._free_form(desc))
    if err is not None:
        raise ValueError(f"descriptor outside the canonical JSON domain (PROTOCOL.md §1): {err}")
    return desc


def old_jcs_domain_error(v: Any, where: str = "$") -> str | None:
    stack = [(v, where)]
    while stack:
        x, at = stack.pop()
        if isinstance(x, str):
            if not x.isascii() or "\x7f" in x:
                return f"{at}: string {x[:40]!r} is not ASCII without DEL"
        elif isinstance(x, bool) or x is None:
            continue
        elif isinstance(x, int):
            if abs(x) > MAX_SAFE_INT:
                return f"{at}: integer {x} exceeds 2^53 - 1 in magnitude"
        elif isinstance(x, (list, tuple)):
            if set(map(type, x)) <= {int}:
                if x and max(max(x), -min(x)) > MAX_SAFE_INT:
                    i = max(i for i, y in enumerate(x) if abs(y) > MAX_SAFE_INT)
                    return f"{at}[{i}]: integer {x[i]} exceeds 2^53 - 1 in magnitude"
                continue
            stack.extend((y, f"{at}[{i}]") for i, y in enumerate(x))
        elif isinstance(x, dict):
            for k, y in x.items():
                if not isinstance(k, str):
                    return f"{at}: key {k!r} is not a string"
                stack.append((k, f"{at}.<key>"))
                stack.append((y, f"{at}.{k}"))
        else:
            return f"{at}: a {type(x).__name__} is not a JSON integer, string, boolean, null, array or object"
    return None


def old_bytes(program, schema: str = C.SCHEMA) -> bytes:
    return C.canonical_json({k: v for k, v in old_encode_program(program, schema).items() if k != "annotations"})


def outcome(f, *args) -> tuple[str, Any]:
    try:
        return "ok", f(*args)
    except (ValueError, TypeError, IdentityConflict) as e:
        return type(e).__name__, str(e)


def assert_same(program, batches=(1, 3, C._NODE_BATCH)) -> int:
    """Old and new agree on ``program`` under each schema and node batch size: bytes, digest and size, or the same refusal.
    Returns how many schemas it encoded under."""
    encoded = 0
    for schema in SCHEMAS:
        want = outcome(old_bytes, program, schema)
        got = outcome(lambda p, s: C.canonical_json({k: v for k, v in C.encode_program(p, s).items() if k != "annotations"}),
                      program, schema)
        assert got == want, (schema, got[0], want[0])
        for batch in batches:
            C._NODE_BATCH = batch
            try:
                assert outcome(lambda p, s: b"".join(C.descriptor_chunks(p, s)), program, schema) == want, (schema, batch)
                if schema == C.SCHEMA and want[0] == "ok":
                    assert C.program_digest(program) == hashlib.sha512(want[1]).hexdigest()
                    assert C.descriptor_size(program) == len(want[1])
                elif schema == C.SCHEMA:
                    assert outcome(C.program_digest, program) == want
            finally:
                C._NODE_BATCH = NODE_BATCH
        encoded += want[0] == "ok"
    return encoded


NODE_BATCH = C._NODE_BATCH


# -- random Programs: reference sequences that alias, broadcast and scatter; statics of every kind; batch and scan -------

def _sum_body(B, S, x):
    acc = x[0]
    for i in range(1, S.N):
        acc = B.call(Add16, acc, x[i])
    return acc


SUM = CompositeDefinition("CodecStreamSum", 1, ("N",), lambda S: ((("x", Array(S.N, V16)),), V16), _sum_body, register=False)


def _mix_body(B, S, x, y):
    f = S.OPS["f"]
    acc = B.call(f, x[0], y)
    for i in range(1, S.N, 1 + S.SHAPE[1]):
        acc = B.call(S.OPS["g"], acc, x[i])
    return acc


MIX = CompositeDefinition("CodecStreamMix", 1, ("N", "TAG", "EPS", "OPS", "SHAPE"),
                          lambda S: ((("x", Array(S.N, V16)), ("y", V16)), V16), _mix_body, register=False)
STEP = CompositeDefinition("CodecStreamStep", 1, (), lambda S: ((("c", V16), ("x", V16)), Tuple(V16, V16)),
                           lambda B, S, c, x: tuple_of(B.call(Add16, c, x), B.call(Mul16, c, x)), register=False)


class _Gen:
    """A random root over two arrays of 16-bit words: sums of element lists sampled from what is in scope (runs, reversed runs,
    one element repeated, scattered picks, or an earlier list extended, so a later argument's leading parts are an earlier
    one's whole sequence), batches of sums, scans, and a composite with int, string, float, function-bundle and tuple statics,
    returning picks from scope."""

    def __init__(self, seed: int) -> None:
        self.r, self.seed = random.Random(seed), seed

    def elements(self, pool: list, seqs: list) -> list:
        r = self.r
        kind = r.choice(("run", "rev", "same", "picks", "extend", "extend"))
        if kind == "extend" and seqs:
            return r.choice(seqs) + [r.choice(pool) for _ in range(r.randint(0, 3))]
        if kind in ("run", "rev"):
            a = r.randrange(len(pool))
            run = pool[a:a + r.randint(1, 6)]
            return run[::-1] if kind == "rev" else run
        if kind == "same":
            return [r.choice(pool)] * r.randint(2, 5)
        return [r.choice(pool) for _ in range(r.randint(1, 6))]

    def program(self) -> Program:
        r = self.r
        p, q, k = r.randint(2, 9), r.randint(1, 6), r.randint(1, 4)

        def body(B, S, x, y):
            pool = [x[i] for i in range(p)] + [y[j] for j in range(q)]
            seqs: list[list] = []
            for _ in range(r.randint(1, 14)):
                op = r.choice(("sum", "sum", "batch", "add", "mix", "scan"))
                els = self.elements(pool, seqs)
                if op == "sum":
                    pool.append(B.call(bind(SUM, N=len(els)), array_of(els)))
                elif op == "batch":
                    rows = [els] + [self.elements(pool, seqs)[:len(els)] for _ in range(r.randint(0, 2))]
                    rows = [row for row in rows if len(row) == len(els)]
                    out = B.batch(bind(SUM, N=len(els)), array_of([array_of(row) for row in rows]), axes=(0,))
                    pool += [out[i] for i in range(len(rows))]
                elif op == "add":
                    pool.append(B.call(r.choice((Add16, Mul16)), r.choice(pool), r.choice(pool)))
                elif op == "mix":
                    ops = {"f": r.choice((Add16, Mul16)), "g": r.choice((Add16, Mul16))}
                    fn = bind(MIX, N=len(els), TAG=r.choice(("a", "row-k", "")), EPS=r.choice((1e-5, 0.5, -0.0, 3.0)), OPS=ops,
                              SHAPE=(len(els), r.randint(0, 2)))
                    pool.append(B.call(fn, array_of(els), r.choice(pool)))
                else:
                    out = B.scan(STEP, r.choice(pool), xs=[array_of(els)])
                    pool += [out[0]] + [out[1][i] for i in range(len(els))]
                seqs.append(els)
            return array_of([r.choice(pool) for _ in range(k)])

        def sig(S):
            return ((("x", Array(p, V16)), ("y", Array(q, V16))), Array(k, V16))

        return Program(CompositeDefinition(f"CodecStreamRoot{self.seed}", 1, (), sig, body, register=False))


RANDOM = 300


def test_random_programs_have_the_old_codecs_bytes():
    """`_Gen`'s Programs: the streamed descriptor, `encode_program`'s, `program_digest` and `descriptor_size` are the old codec's
    under both schemas, at node batches of 1 and 3 (a batch boundary inside a body, where an alias reaches back across it) and
    the default, as is the partition object's `program`; and the alias rule fires (a v1 body writes `alias`)."""
    aliased = 0
    for seed in range(RANDOM):
        prog = _Gen(seed).program()
        assert assert_same(prog) == len(SCHEMAS)
        assert PO.program_sha512(prog) == hashlib.sha512(old_bytes(prog)).hexdigest()
        aliased += b'"alias"' in old_bytes(prog)
    assert aliased > RANDOM // 10


# -- the other tests' Programs ----------------------------------------------------------------------------------------

MODULES = ("test_codec", "test_running_example", "test_annotations", "test_ports", "test_ports_one_use", "test_stacking",
           "test_format_spec", "test_template_instance")


def _in_a_function(d: Any) -> bool:
    f = getattr(d, "evaluate", None) or getattr(d, "body", None)
    return "<locals>" in getattr(f, "__qualname__", "")


@pytest.fixture
def recorded(monkeypatch):
    """The Programs built while it is in use. What the tests it runs register in their own bodies is dropped afterwards, so their
    own runs register it afresh; a module they import keeps what it registers."""
    seen: list[Program] = []
    init = Program.__init__

    def record(self, *a, **k):
        init(self, *a, **k)
        seen.append(self)

    monkeypatch.setattr(Program, "__init__", record)
    before = dict(REGISTRY.defs)
    yield seen
    for fid, d in list(REGISTRY.defs.items()):
        if before.get(fid) is not d and _in_a_function(d):
            del REGISTRY.defs[fid]


def test_the_other_tests_programs_have_the_old_codecs_bytes(recorded):
    """Every Program the circuits tests' zero-argument tests build (`MODULES`): old and new agree on each, bytes or refusal."""
    for name in MODULES:
        mod = importlib.import_module(name)
        for fname, f in inspect.getmembers(mod, inspect.isfunction):
            if not fname.startswith("test_") or f.__module__ != name or inspect.signature(f).parameters:
                continue
            with contextlib.suppress(Exception):  # only the Programs it built are wanted
                f()
    assert len(recorded) > 50
    assert sum(assert_same(p, batches=(1, C._NODE_BATCH)) for p in recorded) > 50


# -- refusals ---------------------------------------------------------------------------------------------------------

def _one(statics: dict) -> Program:
    defn = CompositeDefinition("CodecStreamStatic", 1, tuple(statics), lambda S: ((("x", V16),), V16),
                               lambda B, S, x: B.call(Add16, x, x), register=False)
    inner = bind(defn, **statics)
    return Program(CompositeDefinition("CodecStreamStaticRoot", 1, (), lambda S: ((("x", V16),), V16),
                                       lambda B, S, x: B.call(inner, x), register=False))


@pytest.mark.parametrize("statics", [{"TAG": "caf\u00e9"}, {"TAG": "a\x7fb"}, {"N": 2**53}, {"N": -(2**60)},
                                     {"T": (1, 2**53, "x")}, {"T": {"a": [3, "\u2603"]}}])
def test_a_descriptor_outside_the_domain_is_refused_as_before(statics):
    """A binding outside the canonical JSON domain: `encode_program`, the streamed descriptor and `program_digest` refuse it with
    the old codec's message, which names its path."""
    prog = _one(statics)
    want = outcome(old_bytes, prog)
    assert want[0] == "ValueError" and "outside the canonical JSON domain" in want[1]
    assert_same(prog)


def test_an_identity_conflict_is_refused_as_before():
    """Two different definitions under one specialization id: the old codec's E_IDENTITY, from each entry point."""
    a = bind(CompositeDefinition("CodecStreamTwin", 1, (), lambda S: ((("x", V16),), V16),
                                 lambda B, S, x: B.call(Add16, x, x), register=False))
    b = bind(CompositeDefinition("CodecStreamTwin", 1, (), lambda S: ((("x", V16),), V16),
                                 lambda B, S, x: B.call(Mul16, x, x), register=False))
    prog = Program(CompositeDefinition("CodecStreamTwins", 1, (), lambda S: ((("x", V16),), V16),
                                       lambda B, S, x: B.call(Add16, B.call(a, x), B.call(b, x)), register=False))
    assert outcome(old_bytes, prog)[0] == "IdentityConflict"
    assert_same(prog)


def _plant(r: random.Random, depth: int) -> Any:
    """A random JSON-ish value with long keys, now and then off the domain (a non-ASCII or DEL string, an integer past 2^53, a
    float, a set, a non-string key)."""
    x = r.random()
    if depth == 0 or x < 0.3:
        return r.choice((0, 7, -3, "k", True, None, 2**53 - 1, [1, 2, 3], [], "x" * 50) if r.random() < 0.97 else
                        ("\u00e9", "a\x7f", 2**53, -(2**54), 1.5, {1, 2}, [1, 2**60, 3]))
    if x < 0.65:
        return [_plant(r, depth - 1) for _ in range(r.randint(0, 4))]
    d = {("key" * r.randint(1, 30)) + str(i): _plant(r, depth - 1) for i in range(r.randint(0, 4))}
    if r.random() < 0.02:
        d[3] = 0
    return d


def test_a_domain_error_names_the_old_path():
    """`jcs_domain_error`, its paths spelled only for the error, names what the old one named (the same first violation and the
    same path) on random nested values, and spells a path only when it reports one."""
    r, spelled, errors = random.Random(0), [], 0
    for _ in range(3000):
        v = _plant(r, 4)
        want = old_jcs_domain_error(v)
        got = C.jcs_domain_error(v)
        assert got == want
        errors += want is not None
    assert 300 < errors < 2700
    path = C._path
    try:
        C._path = lambda at: spelled.append(at) or path(at)
        assert C.jcs_domain_error({"definitions": {"Root" * 25_000: {"ports": [[i, i + 1, {"sink": True}] for i in range(5000)]}}}) is None
        assert spelled == []
        assert C.jcs_domain_error({"a": [0, {"b": 2**53}]}) == "$.a[1].b: integer 9007199254740992 exceeds 2^53 - 1 in magnitude"
        assert len(spelled) == 1
    finally:
        C._path = path


# -- what the digest holds ----------------------------------------------------------------------------------------------

def test_the_digest_streams_a_batch_of_nodes_at_a_time(monkeypatch):
    """`program_digest` of a Program, and so a partition object's `program`, builds no descriptor: `encode_program` is never called, and no piece it hashes holds more than
    one batch of a body's nodes (here 8 of a root's 2000), so what it holds is one definition's head and one batch."""
    n = 2000

    def body(B, S, x):
        acc = x[0]
        for i in range(n):
            acc = B.call(Add16, acc, x[i % 4])
        return acc

    prog = Program(CompositeDefinition("CodecStreamLong", 1, (), lambda S: ((("x", Array(4, V16)),), V16), body, register=False))
    want = old_bytes(prog)
    monkeypatch.setattr(C, "_NODE_BATCH", 8)
    monkeypatch.setattr(C, "encode_program", lambda *a, **k: pytest.fail("program_digest encoded the whole descriptor"))
    pieces = list(C.descriptor_chunks(prog))
    assert b"".join(pieces) == want and C.program_digest(prog) == hashlib.sha512(want).hexdigest() == PO.build(prog)["program"]
    node = len(want) // n
    assert max(map(len, pieces)) < 16 * node < len(want) // 100
