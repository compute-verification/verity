"""JSON descriptor codec.

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

The descriptor encodes a program as: the set of specialized definitions reachable from the root (each
primitive by id/signature, each composite specialization by id + explicit binding record + specialized
signature + body nodes with reference sequences), and the root binding. References are *logical value
references only* (parameter leaves and node-output leaves); the descriptor contains no authentication
locations, no commitments, no gate-output components.

The format is specified in `PROTOCOL.md` beside this module, with pinned vectors (`verity/ml/circuits/tests/format_vectors.json`). The bytes
the program digest hashes are RFC 8785 (JCS) of the descriptor without `annotations`: `canonical_json` is JCS on the domain a
descriptor is held to (strings ASCII without DEL, integers of magnitude at most 2^53 - 1, no floats; `jcs_domain_error`).

Schemas (`schema` field): `verity-ir/descriptor/v0` writes every body verbatim; `verity-ir/descriptor/v1`
(the default) writes the same program with flat concatenations, the reference-sequence alias
and interned callee ids / types (see `SCHEMA_V1` below). Both decode. The digest is a function of the
encoding, so it differs between the schemas for one program; `descriptor_equivalence` is the test that two
descriptors denote the same circuit.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
from typing import Callable, Any, Iterator

from .defs import (Body, Builder, CompositeDefinition, FunctionDefinition, IdentityConflict, Node, PrimitiveDefinition, Registry,
                   SpecializedDefinition, _static_key, bind)
from . import ports as P
from .program import input_id, input_prim, is_input
from .refs import Affine, Coll, Concat, Explicit, PartLog, Refs, Strided, refs_from_json
from .types import type_from_json

SCHEMA_V0 = "verity-ir/descriptor/v0"
SCHEMA_V1 = "verity-ir/descriptor/v1"
SCHEMA = SCHEMA_V1
SCHEMAS = (SCHEMA_V0, SCHEMA_V1)

# v1 differs from v0 only in how the *same* program is written down (what a descriptor denotes — gates,
# producers, ordered reference sequences, definitions — is identical; `descriptor_equivalence` checks that):
#   * nested `concat`s are written flat, and a node argument whose leading parts repeat the whole reference
#     sequence of an earlier node's argument is written as `{"alias": [node, arg]}` followed by the remaining
#     parts — a pass-through reference to the sequence already in the body instead of a re-materialised copy
#     (the root's KV cache: position T's `kc` is position T-1's `kc` plus one row);
#   * a node's callee id (`fn`), argument types (`args[].type`) and output type (`out`) are interned: integers
#     into the top-level `callees` and `types` tables (first-seen order of the deterministic dependency walk).
ALIAS_WINDOW = 8  # how many trailing parts an alias search may leave as the explicit suffix


class _Tables:
    """The v1 interning tables of one descriptor: `callees` (specialization ids) and `types` (type JSON)."""

    def __init__(self) -> None:
        self.callees: dict[str, int] = {}
        self.types: dict[str, int] = {}
        self.type_json: list[Any] = []
        self._seen: dict[int, tuple[Any, int]] = {}  # id(type or definition) -> (it, its index), so each object is keyed once

    def callee(self, fid: str) -> int:
        return self.callees.setdefault(fid, len(self.callees))

    def callee_of(self, f: FunctionDefinition) -> int:
        hit = self._seen.get(id(f))
        if hit is None or hit[0] is not f:
            hit = self._seen[id(f)] = (f, self.callee(_spec_id(f)))
        return hit[1]

    def type(self, t) -> int:
        hit = self._seen.get(id(t))
        if hit is not None and hit[0] is t:
            return hit[1]
        j = t.to_json()
        key = json.dumps(j, sort_keys=True, separators=(",", ":"))
        i = self.types.get(key)
        if i is None:
            i = self.types[key] = len(self.type_json)
            self.type_json.append(j)
        self._seen[id(t)] = (t, i)
        return i

    def to_json(self) -> dict:
        return {"callees": list(self.callees), "types": self.type_json}


def _static_json(v: Any) -> Any:
    if isinstance(v, (PrimitiveDefinition, SpecializedDefinition)):
        return {"fn": _spec_id(v)}
    if isinstance(v, CompositeDefinition):
        return {"fn": v.id}
    if isinstance(v, (list, tuple)):
        return [_static_json(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return {"f64": v.hex()}
    if isinstance(v, dict):
        return {"dict": {k: _static_json(v[k]) for k in sorted(v)}}
    raise TypeError(f"unsupported static value {v!r}")


def _spec_id(fn: FunctionDefinition) -> str:
    """Descriptor id of a specialized definition: `Name_vN` for primitives / static-free composites, else
    `Name_vN{k=v,...}` with the binding record in static-parameter order."""
    if isinstance(fn, PrimitiveDefinition):
        return fn.id
    assert isinstance(fn, SpecializedDefinition)
    if not fn.defn.statics:
        return fn.id
    parts = []
    for k in fn.defn.statics:
        parts.append(f"{k}={json.dumps(_static_json(fn.bindings[k]), sort_keys=True, separators=(',', ':'))}")
    return fn.base_id + "{" + ",".join(parts) + "}"


def descriptor_id(fn: FunctionDefinition) -> str:
    """The id a descriptor names `fn` by (its `definitions` key and `callees` entry): the readable `fn.id` differs for string, float,
    function and dict statics (`RMSNormTriton_v1{N=2048,EPS={"f64":"0x1.4f8b588e368f1p-17"}}` against `...EPS=1e-05}`)."""
    return _spec_id(fn)


def _flat_parts(r: Refs) -> list[Refs]:
    """The leaf parts of a (possibly nested) `Concat`, in order; a non-concat is its own single part."""
    if not isinstance(r, Concat):
        return [r]
    out: list[Refs] = []
    stack = [r]
    while stack:
        p = stack.pop()
        if isinstance(p, Concat):
            stack.extend(reversed(p.parts))
        else:
            out.append(p)
    return out


def _part_key(p: Refs) -> tuple:
    """Value identity of one leaf part (the sequence it denotes, by representation)."""
    if isinstance(p, Affine):
        return ("a", p.space, p.idx, p.base, p.count)
    if isinstance(p, Strided):
        return ("s", p.space, p.idx, p.base, p.inner, p.outer_stride, p.inner_stride, p.count)
    assert isinstance(p, Explicit)
    return ("e", p.items)


def _flat_keys(r: Concat) -> tuple[list[Refs], tuple]:
    """`_flat_parts(r)` and their `_part_key`s. A `Concat` cut from a `PartLog` reuses the keys of the log's parts."""
    if r.prefix is not None:
        log, j = r.prefix
        tail = r.parts[j:]
        if log.leaf_parts(j) and not any(isinstance(p, Concat) for p in tail):
            memo = log.memo.setdefault("codec.part_keys", [])
            if len(memo) < j:
                memo.extend(_part_key(p) for p in log.parts[len(memo):j])
            return r.parts, tuple(memo[:j]) + tuple(_part_key(p) for p in tail)
    parts = _flat_parts(r)
    return parts, tuple(_part_key(p) for p in parts)


class _BodyEncoder:
    """Reference encoding within one body. v0: `Refs.to_json` verbatim. v1: flat `concat`s and the alias rule
    (see `SCHEMA_V1`): the whole reference sequence of every encoded node argument is remembered, and a later
    concatenation whose first `m` parts equal one of them (`m` down to `len - ALIAS_WINDOW`, longest first) is
    written as that argument's alias plus its remaining parts. Aliases point backwards only (to an earlier node
    of the same body, or from `ret` to any node), so decoding is one forward pass: a node's later argument never
    aliases its earlier one (`F32Eq(x, x)`), since the decoder has not yet read the node it would name."""

    def __init__(self, schema: str = SCHEMA_V0, tables: _Tables | None = None) -> None:
        self.schema = schema
        self.tables = tables
        self.seqs: dict[tuple, tuple[int, int]] = {}
        if schema != SCHEMA_V0 and tables is None:
            raise ValueError(f"{schema} bodies are encoded within a descriptor (interning tables)")

    def fn(self, f: FunctionDefinition) -> Any:
        return _spec_id(f) if self.tables is None else self.tables.callee_of(f)

    def type(self, t) -> Any:
        return t.to_json() if self.tables is None else self.tables.type(t)

    def refs(self, r: Refs, at: tuple[int, int] | None = None) -> Any:
        if self.schema == SCHEMA_V0 or not isinstance(r, Concat):
            return r.to_json()
        parts, keys = _flat_keys(r)
        n = len(keys)
        out: list | None = None
        for m in range(n, max(1, n - ALIAS_WINDOW - 1), -1):  # m >= 2, suffix <= ALIAS_WINDOW parts
            hit = self.seqs.get(keys[:m])
            if hit is not None and (at is None or hit[0] < at[0]):
                out = [{"alias": list(hit)}] + [p.to_json() for p in parts[m:]]
                break
        if out is None:
            out = [p.to_json() for p in parts]
        if at is not None and n > 1:
            self.seqs.setdefault(keys, at)
        return out[0] if len(out) == 1 else {"concat": out}


def _encode_node(n: Node, enc: _BodyEncoder | None = None, k: int = 0) -> dict:
    enc = enc or _BodyEncoder()
    d: dict[str, Any] = {
        "form": n.form,
        "fn": enc.fn(n.fn),
        "args": [{"type": enc.type(a.type), "refs": enc.refs(a.refs, (k, i))} for i, a in enumerate(n.args)],
        "out": enc.type(n.out),
    }
    if n.form == "batch":
        d["axes"] = list(n.axes)
        d["n"] = n.n
    if n.form == "scan":
        d["n"] = n.n
        d["n_xs"] = n.n_xs
    return d


def _encode_body(b: Body, schema: str = SCHEMA_V0, tables: _Tables | None = None) -> dict:
    """One body. The default is the v0 (verbatim, self-contained) form, which is also what `definition_digest`
    consumers hash; `encode_program` passes the descriptor's schema and interning tables."""
    enc = _BodyEncoder(schema, tables)
    return {
        "params": [[name, t.to_json()] for name, t in b.params],
        "ret_type": b.ret_type.to_json(),
        "nodes": [_encode_node(n, enc, k) for k, n in enumerate(b.nodes)],
        "ret": enc.refs(b.ret),
    }


def _encode_definition(fn: FunctionDefinition, schema: str = SCHEMA_V0, tables: _Tables | None = None) -> dict:
    if isinstance(fn, PrimitiveDefinition):
        d = {
            "form": "primitive",
            "id": fn.id,
            "params": [[name, t.to_json()] for name, t in fn.params],
            "ret": fn.ret.to_json(),
        }
    else:
        assert isinstance(fn, SpecializedDefinition)
        d = {
            "form": "composite",
            "id": fn.base_id,
            "bindings": {k: _static_json(fn.bindings[k]) for k in fn.defn.statics},
            "body": _encode_body(fn.body, schema, tables),
        }
    ports = P.declared(fn)
    if ports is not None:  # only a Definition that declares ports carries the key (PROTOCOL.md §12)
        d["ports"] = ports
    return d


def definition_entry(fn: FunctionDefinition) -> dict:
    """The self-contained (v0) descriptor entry of one definition, callees named by id: what a digest of the definition alone hashes."""
    return _encode_definition(fn)


def _static_functions(v: Any):
    """The function definitions referenced by a static value, recursing into the same containers `_static_key`
    and `_static_json` support (tuples/lists and string-keyed dicts), in encoding order."""
    if isinstance(v, FunctionDefinition):
        yield v
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _static_functions(x)
    elif isinstance(v, dict):
        for k in sorted(v):
            yield from _static_functions(v[k])


def _as_specialized(v: FunctionDefinition, owner: str) -> FunctionDefinition:
    """A function-valued static as the definition the descriptor records it as: primitives and complete
    specializations as they are; a static-free composite as its (unique) specialization."""
    if isinstance(v, CompositeDefinition):
        if v.statics:
            raise TypeError(f"{owner}: static function value {v.id} has unbound static parameters {list(v.statics)}; "
                            f"bind them before it can be encoded")
        return bind(v)
    if isinstance(v, SpecializedDefinition) and not v.complete:
        missing = [k for k in v.defn.statics if k not in v.bindings]
        raise TypeError(f"{owner}: static function value {v.id} is missing static bindings {missing}")
    return v


def _program_fn_functions(program) -> list[FunctionDefinition]:
    """The functions the program function's own binding record references (called by a body node or not). For a
    `Program` from its `fn`; for a `DecodedProgram` the definitions its descriptor resolved for `program_fn`."""
    pfn = getattr(program, "fn", None)
    if isinstance(pfn, SpecializedDefinition):
        owner = _spec_id(pfn)
        return [_as_specialized(f, owner) for k in pfn.defn.statics for f in _static_functions(pfn.bindings[k])]
    return list(getattr(program, "program_fn_functions", ()))


def encode_program(program, schema: str = SCHEMA) -> dict:
    """Encode `program` (a `Program`) as a descriptor dict. Annotations are carried in a separate top-level
    field and excluded from `program_digest`.

    Dependency discovery visits every function a definition depends on: the callee of each body node and every
    function reference inside its static bindings, *including those nested in record/tuple statics* (the same
    containers the binding key and the serializer recurse into), so a bundle such as `ops={"dot": DotA, ...}`
    resolves to shared definitions and `Row{ops=A}` / `Row{ops=B}` coexist. One fully qualified id maps to one
    definition: a second, different definition under the same id is an `IdentityConflict`."""
    if schema not in SCHEMAS:
        raise ValueError(f"unknown descriptor schema {schema!r}")
    tables = None if schema == SCHEMA_V0 else _Tables()
    done: dict[str, dict] = {}

    def encode(fid: str, fn: FunctionDefinition) -> None:
        done[fid] = _encode_definition(fn) if isinstance(fn, PrimitiveDefinition) else _encode_definition(fn, schema, tables)

    seen = _walk(program, encode)
    desc = {
        "schema": schema,
        "root": _spec_id(program.root),
        "program_fn": getattr(program, "program_fn_id", None) or _spec_id(program.fn),
        "definitions": {fid: done[fid] for fid in seen},
        "annotations": dict(program.annotations),
    }
    if tables is not None:
        desc.update(tables.to_json())
    _check_domain(desc)
    return desc


def _walk(program, done: Callable[[str, FunctionDefinition], None]) -> dict[str, FunctionDefinition]:
    """`encode_program`'s dependency walk: each specialization the descriptor records, by descriptor id in first-visit order.
    `done(fid, fn)` runs once per specialization, after every specialization it depends on: the order v1's tables intern in.
    Raises `IdentityConflict` on two different definitions under one id, once every specialization is done."""
    seen: dict[str, FunctionDefinition] = {}
    duplicates: list[tuple[str, FunctionDefinition]] = []
    # One entry per distinct duplicate OBJECT, not per visit: a workload Program composed from request components decoded
    # separately refers to every shared definition through a re-decoded object from EVERY node that calls it, so the per-visit
    # list held one entry per node (on a large workload, over a thousand visits an object) and the identity check below
    # re-encoded the definitions verbatim that many times.
    dup_objects: set[tuple[str, int]] = set()
    visited: dict[int, FunctionDefinition] = {}  # a second visit of the same object changes nothing

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
        if isinstance(fn, SpecializedDefinition):
            for n in fn.body.nodes:
                visit(n.fn)
            for k in fn.defn.statics:
                for f in _static_functions(fn.bindings[k]):
                    visit(_as_specialized(f, fid))
        else:
            assert isinstance(fn, PrimitiveDefinition)
        done(fid, fn)

    visit(program.root)
    for f in _program_fn_functions(program):  # the program function is inlined into the root: its binding record
        visit(f)  # (`program_fn`) is part of the identity, so every function it references is a definition too
    for fid, fn in duplicates:  # a different object under an already-recorded id must denote the same definition
        if _encode_definition(fn) != _encode_definition(seen[fid]):  # self-contained (v0) forms: no table side effects
            raise IdentityConflict(f"E_IDENTITY: two different definitions claim the specialization {fid!r}")
    return seen


def _intern_body(b: Body, tables: _Tables) -> None:
    """What encoding `b` adds to v1's tables, in the order `_encode_node` adds it: each node's callee, its arguments' types, its output's."""
    for n in b.nodes:
        tables.callee_of(n.fn)
        for a in n.args:
            tables.type(a.type)
        tables.type(n.out)


_NODE_BATCH = 4096  # nodes serialized per `canonical_json` call


def _composite_chunks(fn: SpecializedDefinition, head: dict, schema: str, tables: _Tables | None) -> Iterator[bytes]:
    """`canonical_json(_encode_definition(fn, schema, tables))`, a batch of body nodes at a time. ``head`` is the entry without its
    body (`_entry_head`). The keys are written in sorted order: bindings, body (nodes, params, ret, ret_type), form, id, ports."""
    enc, b = _BodyEncoder(schema, tables), fn.body
    yield b'{"bindings":' + canonical_json(head["bindings"]) + b',"body":{"nodes":['
    for lo in range(0, len(b.nodes), _NODE_BATCH):
        part = canonical_json([_encode_node(n, enc, k) for k, n in enumerate(b.nodes[lo:lo + _NODE_BATCH], lo)])
        yield (b"," if lo else b"") + part[1:-1]
    yield (b'],"params":' + canonical_json([[name, t.to_json()] for name, t in b.params]) + b',"ret":' + canonical_json(enc.refs(b.ret))
           + b',"ret_type":' + canonical_json(b.ret_type.to_json()) + b'},"form":' + canonical_json(head["form"]) + b',"id":'
           + canonical_json(head["id"]))
    if "ports" in head:
        yield b',"ports":' + canonical_json(head["ports"])
    yield b"}"


def _entry_head(fn: SpecializedDefinition) -> dict:
    """`_encode_definition(fn)` but for its body's nodes and ``ret``: the parts `_check_domain` reads (`_free_form`), in its key order."""
    d = {"form": "composite", "id": fn.base_id, "bindings": {k: _static_json(fn.bindings[k]) for k in fn.defn.statics},
         "body": {"params": [[name, t.to_json()] for name, t in fn.body.params], "ret_type": fn.body.ret_type.to_json()}}
    ports = P.declared(fn)
    if ports is not None:
        d["ports"] = ports
    return d


def descriptor_chunks(program, schema: str = SCHEMA) -> Iterator[bytes]:
    """The bytes `program_digest` hashes, `canonical_json` of `encode_program(program, schema)` without ``annotations``, in pieces,
    without the descriptor's dict: one walk of the specializations (`_walk`), interning v1's tables in `encode_program`'s order,
    then each definition in key order, its body a batch of nodes at a time. Raises what `encode_program` raises, `_check_domain`'s
    ValueError once the last piece is out."""
    if schema not in SCHEMAS:
        raise ValueError(f"unknown descriptor schema {schema!r}")
    tables = None if schema == SCHEMA_V0 else _Tables()
    heads: dict[str, dict] = {}

    def intern(fid: str, fn: FunctionDefinition) -> None:
        if isinstance(fn, PrimitiveDefinition):
            heads[fid] = _encode_definition(fn)
            return
        heads[fid] = _entry_head(fn)
        if tables is not None:
            _intern_body(fn.body, tables)

    seen = _walk(program, intern)
    skeleton = {"schema": schema, "root": _spec_id(program.root),
                "program_fn": getattr(program, "program_fn_id", None) or _spec_id(program.fn),
                "definitions": {fid: heads[fid] for fid in seen}}
    if tables is not None:
        skeleton.update(tables.to_json())
        interned = (len(tables.callees), len(tables.type_json))
        yield b'{"callees":' + canonical_json(skeleton["callees"]) + b',"definitions":{'
    else:
        yield b'{"definitions":{'
    for i, fid in enumerate(sorted(seen)):
        yield (b"," if i else b"") + canonical_json(fid) + b":"
        fn = seen[fid]
        if isinstance(fn, PrimitiveDefinition):
            yield canonical_json(heads[fid])
        else:
            yield from _composite_chunks(fn, heads[fid], schema, tables)
    yield (b'},"program_fn":' + canonical_json(skeleton["program_fn"]) + b',"root":' + canonical_json(skeleton["root"])
           + b',"schema":' + canonical_json(schema))
    if tables is not None:
        if (len(tables.callees), len(tables.type_json)) != interned:
            raise AssertionError("descriptor_chunks: encoding a body interned an entry the tables were written without")
        yield b',"types":' + canonical_json(skeleton["types"])
    yield b"}"
    _check_domain(skeleton)


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON, the bytes Lean's `Flock.canon` writes: compact, keys sorted, every character outside printable ASCII
    a lowercase `\\uXXXX` escape (a surrogate pair above U+FFFF), so no byte is at or above 0x7f."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


MAX_SAFE_INT = 2**53 - 1


def jcs_domain_error(v: Any, where: str = "$") -> str | None:
    """Where `v` leaves the domain on which `canonical_json` is RFC 8785 (JCS), or None: JSON values whose strings are ASCII
    without DEL (U+0000..U+007E) and whose numbers are integers of magnitude at most 2^53 - 1."""
    # A path is (parent, step), an int step an index, and is spelled out only for the error: spelling each item's path as it is
    # pushed copied the whole path once per item, and a window's root id (its binding record) is ~100 KB over 1.5M ports.
    stack: list[tuple[Any, Any]] = [(v, (None, where))]
    while stack:
        x, at = stack.pop()
        if isinstance(x, str):
            if not x.isascii() or "\x7f" in x:
                return f"{_path(at)}: string {x[:40]!r} is not ASCII without DEL"
        elif isinstance(x, bool) or x is None:
            continue
        elif isinstance(x, int):
            if abs(x) > MAX_SAFE_INT:
                return f"{_path(at)}: integer {x} exceeds 2^53 - 1 in magnitude"
        elif isinstance(x, (list, tuple)):
            if set(map(type, x)) <= {int}:  # checked whole, as popping its items one by one would: the last one out of range first
                if x and max(max(x), -min(x)) > MAX_SAFE_INT:
                    i = max(i for i, y in enumerate(x) if abs(y) > MAX_SAFE_INT)
                    return f"{_path(at)}[{i}]: integer {x[i]} exceeds 2^53 - 1 in magnitude"
                continue
            stack.extend((y, (at, i)) for i, y in enumerate(x))
        elif isinstance(x, dict):
            for k, y in x.items():
                if not isinstance(k, str):
                    return f"{_path(at)}: key {k!r} is not a string"
                stack.append((k, (at, ".<key>")))
                stack.append((y, (at, "." + k)))
        else:
            return f"{_path(at)}: a {type(x).__name__} is not a JSON integer, string, boolean, null, array or object"
    return None


def _path(at: tuple[Any, Any]) -> str:
    steps = []
    while at is not None:
        at, step = at
        steps.append(f"[{step}]" if isinstance(step, int) else step)
    return "".join(reversed(steps))


def _free_form(desc: dict) -> Any:
    """The parts of a descriptor that are not structural indices: every top-level field but `definitions` and `annotations`,
    and each definition's key, id, bindings, parameters and return type (a body's nodes and `ret` hold only table indices,
    reference indices and the fixed vocabulary, checked by the decoder)."""
    defs = {fid: {k: v for k, v in d.items() if k != "body"} | {k: v for k, v in d.get("body", {}).items() if k in ("params", "ret_type")}
            for fid, d in desc.get("definitions", {}).items()}
    return {k: v for k, v in desc.items() if k not in ("definitions", "annotations")} | {"definitions": defs}


def _check_domain(desc: dict) -> None:
    err = jcs_domain_error(_free_form(desc))
    if err is not None:
        raise ValueError(f"descriptor outside the canonical JSON domain (PROTOCOL.md §1): {err}")


def program_digest(program_or_desc) -> str:
    """SHA-512 of the descriptor's canonical JSON without ``annotations``; a program's is streamed (`descriptor_chunks`)."""
    if isinstance(program_or_desc, dict):
        return hashlib.sha512(canonical_json({k: v for k, v in program_or_desc.items() if k != "annotations"})).hexdigest()
    # The cyclic collector stays off while streaming: encoding frees its pieces by reference counting, and the collector's
    # passes over a big program's live bodies cost more than the encoding itself.
    on = gc.isenabled()
    gc.disable()
    try:
        h = hashlib.sha512()
        for chunk in descriptor_chunks(program_or_desc):
            h.update(chunk)
        return h.hexdigest()
    finally:
        if on:
            gc.enable()


def descriptor_size(program_or_desc) -> int:
    if isinstance(program_or_desc, dict):
        return len(canonical_json({k: v for k, v in program_or_desc.items() if k != "annotations"}))
    return sum(map(len, descriptor_chunks(program_or_desc)))


# ---------------------------------------------------------------------------------------------------------
# Decoding: rebuild specializations with explicit bodies. Primitives are looked up in `registry` by id (their
# evaluators are code, not data).
# ---------------------------------------------------------------------------------------------------------

# Primitive families that are materialised on demand (one definition per width / constant / vocab) when a descriptor is
# decoded in a process that has not built the program: (pattern over the primitive id, maker from the match).  A semantics
# package registers its families with `register_lazy_family` (`verity.ml.circuits` never imports a semantics package).  An id matching
# `program.INPUT_ID` or `program.REGISTERED_ID` always denotes `program`'s own `Input<w>` or `RegisteredInput<w>`, ahead of
# any registry entry.  A pattern matches the whole id, and the member it makes must carry that id: a family has one spelling
# per member.
_LAZY_PRIMITIVE_FAMILIES: list[tuple[re.Pattern[str], Callable[[re.Match[str]], PrimitiveDefinition]]] = []


def register_lazy_family(pattern: str | re.Pattern[str], make: Callable[[re.Match[str]], PrimitiveDefinition]) -> None:
    """Teach `decode_program` to materialise a primitive family by id (e.g. `Const<w>[0x..]_v1`) via `make(match)`."""
    pat = re.compile(pattern) if isinstance(pattern, str) else pattern
    if not any(p.pattern == pat.pattern for p, _ in _LAZY_PRIMITIVE_FAMILIES):
        _LAZY_PRIMITIVE_FAMILIES.append((pat, make))


def _split_record(fid: str) -> list[tuple[str, str]]:
    """The `k=<canonical json>` pairs of a specialization id `Name_vN{k1=..,k2=..}`, in id order (commas inside
    strings/brackets are skipped); `[]` for an id without a record."""
    if "{" not in fid or not fid.endswith("}"):
        return []
    rec = fid[fid.index("{") + 1:-1]
    pairs: list[tuple[str, str]] = []
    depth, in_str, esc, start = 0, False, False, 0
    for pos, ch in enumerate(rec + ","):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 0:
            k, _, v = rec[start:pos].partition("=")
            pairs.append((k, v))
            start = pos + 1
    return pairs


def _statics_order(fid: str, names: tuple[str, ...]) -> tuple[str, ...]:
    """Static-parameter order of a decoded specialization. The `bindings` object loses it under canonical JSON
    (sorted keys); the definition's id `Name_vN{k1=..,k2=..}` keeps it, so read the top-level keys back."""
    if len(names) < 2 or "{" not in fid:
        return names
    order = tuple(k for k, _ in _split_record(fid))
    if sorted(order) != sorted(names):
        raise ValueError(f"definition {fid!r}: binding record keys {sorted(names)} do not match its id")
    return order


def _record_function_ids(fid: str) -> list[str]:
    """Every `{"fn": id}` reference in a specialization id's binding record, in encoding order."""
    out: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            if set(v) == {"fn"}:
                out.append(v["fn"])
            else:
                for k in sorted(v):
                    walk(v[k])
        elif isinstance(v, list):
            for x in v:
                walk(x)

    for _, v in _split_record(fid):
        try:
            walk(json.loads(v))
        except ValueError:  # a readable (non-JSON) record such as the root's `Root[Serve_v1{C={2305f91c1425},..}]`
            pass
    return out


def lookup_primitive(registry: Registry, fid: str) -> PrimitiveDefinition:
    """`registry[fid]`, materialising the on-demand families (`Input<w>`, `Const<w>[bits]`, `GatherBf16x<V>`)
    when a descriptor is decoded in a process that has not built the program (their evaluators are total
    functions of the id).  Public so a semantics package's lazy-family makers (`register_lazy_family`) can
    resolve the primitive they lift or wrap by id."""
    return _lookup_primitive(registry, fid)


def _lookup_primitive(registry: Registry, fid: str) -> PrimitiveDefinition:
    m = input_id(fid)
    if m:
        return input_prim(m[1], m[0])
    if fid in registry:
        return registry[fid]  # type: ignore[return-value]
    for pat, make in _LAZY_PRIMITIVE_FAMILIES:
        m = pat.fullmatch(fid)
        if m:
            p = make(m)
            if p.id != fid:
                raise KeyError(f"primitive {fid} is not in the registry: its family spells that member {p.id}")
            return p
    raise KeyError(f"primitive {fid} is not in the registry")


class DecodedProgram:
    def __init__(self, root: SpecializedDefinition, defs: dict[str, Any], desc: dict,
                 program_fn_functions: tuple[FunctionDefinition, ...] = ()) -> None:
        self.root = root
        self.defs = defs
        self.desc = desc
        from .layout import Circuit
        self.circuit = Circuit(root)
        self.annotations = dict(desc.get("annotations", {}))
        theta = self.annotations.get("theta", ())
        root.theta = tuple(theta) if isinstance(theta, list) else theta  # type: ignore[attr-defined]  # `ports.facts` checks it
        self.program_fn_id = desc["program_fn"]
        self.program_fn_functions = program_fn_functions  # the definitions `program_fn`'s record references

    @property
    def gates(self) -> int:
        return self.root.gates


def _decode_refs(j, nodes: list[Node], schema: str, leaves: Callable[[str, int], int | None] | None = None) -> Refs:
    """v0: `refs_from_json`. v1: also resolves `{"alias": [k, a]}` to the reference sequence node `k` passed as
    argument `a` (already decoded — aliases point backwards), splicing a concatenation's alias parts in flat so
    the decoded representation does not depend on how the writer nested it.  Given `leaves`, every part read is `_checked_refs`."""
    if schema == SCHEMA_V0:
        return _checked_refs(refs_from_json(j), leaves)
    if "alias" in j:
        k, a = j["alias"]
        if not (_nat(k) and _nat(a) and k < len(nodes) and a < len(nodes[k].args)):
            raise ValueError(f"alias to argument {a} of node {k}: aliases name an argument of an earlier node")
        return nodes[k].args[a].refs
    if "concat" in j:
        items = j["concat"]
        log = None
        if items and "alias" in items[0]:
            p = _decode_refs(items[0], nodes, schema, leaves)
            if isinstance(p, Concat) and p.prefix is not None and p.prefix[1] == len(p.prefix[0]) == len(p.parts):
                log = p.prefix[0]  # the alias is the whole log so far (a key sequence one row longer than the call before)
            elif isinstance(p, Concat):
                log = PartLog(p.parts)
            else:
                log = PartLog([p])
            items = items[1:]
        parts: list[Refs] = []
        for x in items:
            p = _decode_refs(x, nodes, schema, leaves)
            parts.extend(p.parts if isinstance(p, Concat) else [p])
        if log is None:
            return Concat(parts)
        for p in parts:
            log.append(p)
        return log.raw_cut(len(log))
    return _checked_refs(refs_from_json(j), leaves)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _nat(x: Any) -> bool:
    return _int(x) and x >= 0


def _checked_refs(r: Refs, leaves: Callable[[str, int], int | None] | None) -> Refs:
    """`r`, if every reference in it names an existing leaf: of a parameter of the body (`p`) or of a node's output (`n`).  A node
    naming itself or a later node decodes (checkers name that defect); the canonical layout assumes it does not (PROTOCOL.md §6).
    `leaves(space, idx)` is that collection's leaf count, None if there is none."""
    if leaves is None:
        return r
    if isinstance(r, Concat):
        for p in r.parts:
            _checked_refs(p, leaves)
        return r
    if isinstance(r, Explicit):
        ok = all(len(x) == 3 and _int(x[2]) for x in r.items)
        spans = [(s, i, j, j) for s, i, j in r.items] if ok else []
    elif isinstance(r, Affine):
        ok = _int(r.base) and _nat(r.count)
        spans = [(r.space, r.idx, r.base, r.base + r.count - 1)] if ok and r.count else []
    else:
        ok = all(map(_int, (r.base, r.outer_stride, r.inner_stride))) and _nat(r.count) and _int(r.inner) and r.inner >= 1
        spans = []
        if ok and r.count:
            q, last = divmod(r.count - 1, r.inner)
            corners = [(0, 0), (0, min(r.inner, r.count) - 1), (q, 0), (q, last)] + ([(q - 1, r.inner - 1)] if q else [])
            vals = [r.base + a * r.outer_stride + b * r.inner_stride for a, b in corners]
            spans = [(r.space, r.idx, min(vals), max(vals))]
    for s, i, lo, hi in spans:
        n = leaves(s, i) if s in ("p", "n") and _nat(i) else None
        ok = ok and n is not None and 0 <= lo and hi < n
    if not ok:
        raise ValueError(f"references {r.to_json()} name no leaf of a parameter or a node")
    return r


def _type(j: Any) -> Any:
    try:
        return type_from_json(j)
    except (TypeError, ValueError, IndexError, KeyError) as e:
        raise ValueError(f"type {j!r}: {e}") from e


def _at(table: list, i: Any, what: str) -> Any:
    if not (isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(table)):
        raise ValueError(f"{what} index {i!r} outside 0..{len(table) - 1}")
    return table[i]


_NODE_KEYS = {"primitive": {"form", "fn", "args", "out"}, "call": {"form", "fn", "args", "out"},
              "batch": {"form", "fn", "args", "out", "axes", "n"}, "scan": {"form", "fn", "args", "out", "n", "n_xs"}}


def _check_node(fid: str, k: int, nd: dict, fn: FunctionDefinition) -> None:
    form = nd.get("form")
    if form not in _NODE_KEYS or set(nd) != _NODE_KEYS[form]:
        raise ValueError(f"definition {fid!r} node {k}: form {form!r} with keys {sorted(nd)}")
    prim = isinstance(fn, PrimitiveDefinition)
    if (form == "primitive" and not prim) or (form in ("call", "scan") and prim):
        raise ValueError(f"definition {fid!r} node {k}: a {form} node does not call {'a primitive' if prim else 'a composite'} ({fn.id})")
    if (form == "batch" and not (isinstance(nd["axes"], list) and all(a is None or _nat(a) for a in nd["axes"]) and _nat(nd["n"]))) or \
            (form == "scan" and not (_nat(nd["n"]) and _nat(nd["n_xs"]) and nd["n_xs"] < len(nd["args"]))):
        raise ValueError(f"definition {fid!r} node {k}: {form} counts {({x: nd[x] for x in ('axes', 'n', 'n_xs') if x in nd})}")


def _build_node(B: Builder, fid: str, k: int, nd: dict, fn: FunctionDefinition, args: tuple[Coll, ...], out: Any) -> None:
    """Rebuild node `k` through the `Builder`, which checks its arguments against the callee's signature and computes its output
    type; the node's declared `out` must be that type."""
    try:
        if nd["form"] in ("primitive", "call"):
            B.call(fn, *args)
        elif nd["form"] == "batch":
            B.batch(fn, *args, axes=nd["axes"], n=nd["n"])
        else:
            x = 1 + nd["n_xs"]
            B.scan(fn, args[0], args[1:x], args[x:], n=nd["n"])
    except (TypeError, IndexError) as e:
        raise ValueError(f"definition {fid!r} node {k}: {e}") from e
    if B.nodes[k].out != out:
        raise ValueError(f"definition {fid!r} node {k}: declares out {out!r}, computes {B.nodes[k].out!r}")


def _record_value(v: Any, fn: Callable[[str], Any]) -> Any:
    """A binding record's static value, `_static_json` inverted; ``fn`` resolves each function reference's id."""
    if isinstance(v, dict) and "fn" in v:
        return fn(v["fn"])
    if isinstance(v, dict) and "f64" in v:
        return float.fromhex(v["f64"])
    if isinstance(v, dict) and "dict" in v:
        return {k: _record_value(x, fn) for k, x in v["dict"].items()}
    if isinstance(v, list):
        return tuple(_record_value(x, fn) for x in v)
    return v


def definition_decoder(desc: dict, registry: Registry, built: dict[str, Any] | None = None) -> Callable[[str], Any]:
    """The per-definition reader `decode_program` uses: each definition of ``desc`` by its id, decoded into ``built``."""
    schema = desc.get("schema")
    if schema not in SCHEMAS:
        raise ValueError(f"unexpected descriptor schema {schema!r}")
    _check_domain(desc)
    ddefs = desc["definitions"]
    built = {} if built is None else built
    if schema == SCHEMA_V0:
        callee_id, arg_type = (lambda f: f), _type
    else:
        callees, types = desc["callees"], [_type(t) for t in desc["types"]]
        callee_id, arg_type = (lambda i: _at(callees, i, "callees")), (lambda i: _at(types, i, "types"))

    def get(fid: str):
        if fid in built:
            return built[fid]
        d = ddefs[fid]
        if d["form"] == "primitive":
            if d["id"] != fid:
                raise ValueError(f"definition {fid!r}: a primitive is keyed by its id, not {d['id']!r}")
            p = _lookup_primitive(registry, d["id"])
            if [[n, t.to_json()] for n, t in p.params] != d["params"] or p.ret.to_json() != d["ret"]:
                raise ValueError(f"primitive {fid} signature differs from the registry")
            if d.get("ports") != P.declared(p):
                raise ValueError(f"primitive {fid} ports differ from the registry")
            built[fid] = p
            return p
        name, _, ver = d["id"].rpartition("_v")
        if d["form"] != "composite" or not (name and ver.isdigit() and f"{name}_v{int(ver)}" == d["id"]):
            raise ValueError(f"definition {fid!r}: form {d['form']!r}, id {d['id']!r} (a composite's id is Name_vN)")
        statics = _statics_order(fid, tuple(d["bindings"].keys()))
        key = d["id"] + ("{" + ",".join(f"{k}={canonical_json(d['bindings'][k]).decode()}" for k in statics) + "}" if statics else "")
        if key != fid:
            raise ValueError(f"definition {fid!r}: a composite is keyed by its id and binding record, {key!r}")
        cdef = CompositeDefinition(f"{name}", int(ver), statics, lambda S: ((), None), lambda *_: None, register=False)
        sp = SpecializedDefinition(cdef, {}, ())
        built[fid] = sp
        sp.bindings.update({k: _record_value(v, get) for k, v in d["bindings"].items()})
        ours(fid)
        b = d["body"]
        params = tuple((n, _type(t)) for n, t in b["params"])
        ret_type = _type(b["ret_type"])
        B = Builder(params, ret_type)
        nodes = B.nodes
        outs = [arg_type(nd["out"]) for nd in b["nodes"]]           # each node's declared output, checked when the node is built

        def leaves(space: str, idx: int) -> int | None:
            if space == "p":
                return params[idx][1].leaves if idx < len(params) else None
            return outs[idx].leaves if idx < len(outs) else None

        for nd in b["nodes"]:
            k = len(nodes)
            fn = get(callee_id(nd["fn"]))
            if fid != desc["root"] and is_input(fn):
                raise ValueError(f"definition {fid!r} node {k}: an input node ({fn.id}) appears only in the root's body")
            _check_node(fid, k, nd, fn)
            try:
                args = tuple(Coll(arg_type(a["type"]), _decode_refs(a["refs"], nodes, schema, leaves)) for a in nd["args"])
            except ValueError as e:
                raise ValueError(f"definition {fid!r} node {k}: {e}") from e
            _build_node(B, fid, k, nd, fn, args, arg_type(nd["out"]))
        try:
            ret = Coll(ret_type, _decode_refs(b["ret"], nodes, schema, leaves))
        except (TypeError, ValueError) as e:
            raise ValueError(f"definition {fid!r} ret: {e}") from e
        try:
            sp.set_body(B.finish(ret))
        except TypeError as e:
            raise ValueError(f"definition {fid!r}: {e}") from e
        sp.set_port_types(P.PortTypes.from_json(fid, d["ports"], params, ret_type, outs if fid == desc["root"] else None)
                          if "ports" in d else None)
        return sp

    admitted: dict[str, Any] = {}

    def ours(fid: str):
        """``fid`` as `bind` would admit it: where the registry holds its composite, that composite specialized at the file's
        binding record (function references the registry's own where it holds them too), once the composite has every static
        the record names and its `refuse` has no reason against the record; otherwise the decoded definition. Uncached, so a
        file's record never enters the registry's binding cache."""
        if fid in admitted:
            return admitted[fid]
        d = ddefs[fid]
        reg = registry[d["id"]] if d["form"] == "composite" and d["id"] in registry else None
        if not isinstance(reg, CompositeDefinition):
            admitted[fid] = get(fid)
            return admitted[fid]
        record = {k: _record_value(v, ours) for k, v in d["bindings"].items()}
        if unknown := [k for k in record if k not in reg.statics]:
            raise ValueError(f"definition {fid!r}: {reg.id} has no static parameter {unknown[0]!r}")
        try:
            why = reg.refuse(record) if reg.refuse is not None else None
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as e:
            why = f"{type(e).__name__} judging the record: {e}"
        if why:
            raise ValueError(f"definition {fid!r}: {reg.id} refuses this binding: {why}")
        admitted[fid] = SpecializedDefinition(reg, record, tuple((k, _static_key(record[k])) for k in reg.statics if k in record))
        return admitted[fid]

    return get


def decode_program(desc: dict, registry: Registry) -> DecodedProgram:
    built: dict[str, Any] = {}
    get = definition_decoder(desc, registry, built)
    root = get(desc["root"])
    if not isinstance(root, SpecializedDefinition) or root.body.params:
        raise ValueError(f"root {desc['root']!r}: the root is a composite without parameters")
    pfn_fns = tuple(get(fid) for fid in _record_function_ids(desc["program_fn"]) if fid in desc["definitions"])
    P.check(root)
    return DecodedProgram(root, built, desc, pfn_fns)
