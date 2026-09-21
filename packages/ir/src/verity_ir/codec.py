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

Status: one deterministic encoding (sorted keys, compact separators, ints only) — `canonical_json` — is what
the program digest hashes. It is **not** a byte-canonical schema in the sense of a frozen external spec
(no field-order/escaping/number-format specification beyond Python's `json` with `sort_keys=True`,
`ensure_ascii=True`); say so wherever the digest is reported. Annotations are excluded from the digest.

Schemas (`schema` field): `verity-ir/descriptor/v0` writes every body verbatim; `verity-ir/descriptor/v1`
(the default since 2026-09-08) writes the same program with flat concatenations, the reference-sequence alias
and interned callee ids / types (see `SCHEMA_V1` below). Both decode. The digest is a function of the
encoding, so it differs between the schemas for one program; `descriptor_equivalence` is the test that two
descriptors denote the same circuit (docs/vllm-poc/descriptor-codec.md).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, Any

from .defs import (Body, CompositeDefinition, FunctionDefinition, IdentityConflict, Node, PrimitiveDefinition, Registry,
                   SpecializedDefinition, bind)
from .refs import Affine, Coll, Concat, Explicit, Refs, Strided, refs_from_json
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

    def callee(self, fid: str) -> int:
        return self.callees.setdefault(fid, len(self.callees))

    def type(self, t) -> int:
        j = t.to_json()
        key = json.dumps(j, sort_keys=True, separators=(",", ":"))
        i = self.types.get(key)
        if i is None:
            i = self.types[key] = len(self.type_json)
            self.type_json.append(j)
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


class _BodyEncoder:
    """Reference encoding within one body. v0: `Refs.to_json` verbatim. v1: flat `concat`s and the alias rule
    (see `SCHEMA_V1`): the whole reference sequence of every encoded node argument is remembered, and a later
    concatenation whose first `m` parts equal one of them (`m` down to `len - ALIAS_WINDOW`, longest first) is
    written as that argument's alias plus its remaining parts. Aliases point backwards only (to an earlier node
    of the same body, or from `ret` to any node), so decoding is one forward pass."""

    def __init__(self, schema: str = SCHEMA_V0, tables: _Tables | None = None) -> None:
        self.schema = schema
        self.tables = tables
        self.seqs: dict[tuple, tuple[int, int]] = {}
        if schema != SCHEMA_V0 and tables is None:
            raise ValueError(f"{schema} bodies are encoded within a descriptor (interning tables)")

    def fn(self, f: FunctionDefinition) -> Any:
        fid = _spec_id(f)
        return fid if self.tables is None else self.tables.callee(fid)

    def type(self, t) -> Any:
        return t.to_json() if self.tables is None else self.tables.type(t)

    def refs(self, r: Refs, at: tuple[int, int] | None = None) -> Any:
        if self.schema == SCHEMA_V0 or not isinstance(r, Concat):
            return r.to_json()
        parts = _flat_parts(r)
        keys = tuple(_part_key(p) for p in parts)
        n = len(keys)
        out: list | None = None
        for m in range(n, max(1, n - ALIAS_WINDOW - 1), -1):  # m >= 2, suffix <= ALIAS_WINDOW parts
            hit = self.seqs.get(keys[:m])
            if hit is not None:
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
        return {
            "form": "primitive",
            "id": fn.id,
            "params": [[name, t.to_json()] for name, t in fn.params],
            "ret": fn.ret.to_json(),
        }
    assert isinstance(fn, SpecializedDefinition)
    return {
        "form": "composite",
        "id": fn.base_id,
        "bindings": {k: _static_json(fn.bindings[k]) for k in fn.defn.statics},
        "body": _encode_body(fn.body, schema, tables),
    }


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
    defs: dict[str, dict] = {}
    seen: dict[str, FunctionDefinition] = {}
    duplicates: list[tuple[str, FunctionDefinition]] = []

    def visit(fn: FunctionDefinition) -> None:
        fid = _spec_id(fn)
        prev = seen.get(fid)
        if prev is not None:
            if prev is not fn:
                duplicates.append((fid, fn))
            return
        seen[fid] = fn
        if isinstance(fn, PrimitiveDefinition):
            defs[fid] = _encode_definition(fn)
            return
        assert isinstance(fn, SpecializedDefinition)
        defs[fid] = {}  # placeholder against cycles
        for n in fn.body.nodes:
            visit(n.fn)
        for k in fn.defn.statics:
            for f in _static_functions(fn.bindings[k]):
                visit(_as_specialized(f, fid))
        defs[fid] = _encode_definition(fn, schema, tables)

    root = program.root
    visit(root)
    for f in _program_fn_functions(program):  # the program function is inlined into the root: its binding record
        visit(f)  # (`program_fn`) is part of the identity, so every function it references is a definition too
    for fid, fn in duplicates:  # a different object under an already-recorded id must denote the same definition
        if _encode_definition(fn) != _encode_definition(seen[fid]):  # self-contained (v0) forms: no table side effects
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
    return desc


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def program_digest(program_or_desc) -> str:
    desc = program_or_desc if isinstance(program_or_desc, dict) else encode_program(program_or_desc)
    d = {k: v for k, v in desc.items() if k != "annotations"}
    return hashlib.sha256(canonical_json(d)).hexdigest()


def descriptor_size(program_or_desc) -> int:
    desc = program_or_desc if isinstance(program_or_desc, dict) else encode_program(program_or_desc)
    return len(canonical_json({k: v for k, v in desc.items() if k != "annotations"}))


# ---------------------------------------------------------------------------------------------------------
# Decoding: rebuild specializations with explicit bodies. Primitives are looked up in `registry` by id (their
# evaluators are code, not data).
# ---------------------------------------------------------------------------------------------------------

# Primitive families that are materialised on demand (one definition per width / constant / vocab) when a descriptor is
# decoded in a process that has not built the program: (pattern over the primitive id, maker from the match).  The IR
# registers only its own `Input<w>`; a semantics package registers its families with `register_lazy_family` (the IR never
# imports a semantics package).
_LAZY_PRIMITIVE_FAMILIES: list[tuple[re.Pattern[str], Callable[[re.Match[str]], PrimitiveDefinition]]] = [
    (re.compile(r"^Input(\d+)_v1$"), lambda m: __import__("verity_ir.program", fromlist=["input_prim"]).input_prim(int(m[1]))),
]


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


def _lookup_primitive(registry: Registry, fid: str) -> PrimitiveDefinition:
    """`registry[fid]`, materialising the on-demand families (`Input<w>`, `Const<w>[bits]`, `GatherBf16x<V>`)
    when a descriptor is decoded in a process that has not built the program (their evaluators are total
    functions of the id)."""
    if fid in registry:
        return registry[fid]  # type: ignore[return-value]
    for pat, make in _LAZY_PRIMITIVE_FAMILIES:
        m = pat.match(fid)
        if m:
            return make(m)
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
        self.program_fn_id = desc["program_fn"]
        self.program_fn_functions = program_fn_functions  # the definitions `program_fn`'s record references

    @property
    def gates(self) -> int:
        return self.root.gates


def _decode_refs(j, nodes: list[Node], schema: str) -> Refs:
    """v0: `refs_from_json`. v1: also resolves `{"alias": [k, a]}` to the reference sequence node `k` passed as
    argument `a` (already decoded — aliases point backwards), splicing a concatenation's alias parts in flat so
    the decoded representation does not depend on how the writer nested it."""
    if schema == SCHEMA_V0:
        return refs_from_json(j)
    if "alias" in j:
        k, a = j["alias"]
        if k >= len(nodes):
            raise ValueError(f"alias to node {k} before it is defined")
        return nodes[k].args[a].refs
    if "concat" in j:
        parts: list[Refs] = []
        for x in j["concat"]:
            p = _decode_refs(x, nodes, schema)
            parts.extend(p.parts if isinstance(p, Concat) else [p])
        return Concat(parts)
    return refs_from_json(j)


def decode_program(desc: dict, registry: Registry) -> DecodedProgram:
    schema = desc.get("schema")
    if schema not in SCHEMAS:
        raise ValueError(f"unexpected descriptor schema {schema!r}")
    ddefs = desc["definitions"]
    built: dict[str, Any] = {}
    if schema == SCHEMA_V0:
        callee_id, arg_type = (lambda f: f), type_from_json
    else:
        callees, types = desc["callees"], [type_from_json(t) for t in desc["types"]]
        callee_id, arg_type = callees.__getitem__, types.__getitem__

    def get(fid: str):
        if fid in built:
            return built[fid]
        d = ddefs[fid]
        if d["form"] == "primitive":
            p = _lookup_primitive(registry, d["id"])
            if [[n, t.to_json()] for n, t in p.params] != d["params"] or p.ret.to_json() != d["ret"]:
                raise ValueError(f"primitive {fid} signature differs from the registry")
            built[fid] = p
            return p
        name, _, ver = d["id"].rpartition("_v")
        statics = _statics_order(fid, tuple(d["bindings"].keys()))
        cdef = CompositeDefinition(f"{name}", int(ver), statics, lambda S: ((), None), lambda *_: None, register=False)
        sp = SpecializedDefinition(cdef, {}, ())
        built[fid] = sp
        def static_of(v):
            if isinstance(v, dict) and "fn" in v:
                return get(v["fn"])
            if isinstance(v, dict) and "f64" in v:
                return float.fromhex(v["f64"])
            if isinstance(v, dict) and "dict" in v:
                return {k2: static_of(v2) for k2, v2 in v["dict"].items()}
            if isinstance(v, list):
                return tuple(static_of(x) for x in v)
            return v

        bindings = {k: static_of(v) for k, v in d["bindings"].items()}
        sp.bindings.update(bindings)
        b = d["body"]
        params = tuple((n, type_from_json(t)) for n, t in b["params"])
        ret_type = type_from_json(b["ret_type"])
        nodes: list[Node] = []
        offsets = [0]
        for nd in b["nodes"]:
            fn = get(callee_id(nd["fn"]))
            args = tuple(Coll(arg_type(a["type"]), _decode_refs(a["refs"], nodes, schema)) for a in nd["args"])
            node = Node(nd["form"], fn, args, arg_type(nd["out"]), axes=tuple(nd.get("axes", ())),
                        n=nd.get("n", 1), n_xs=nd.get("n_xs", 0))
            nodes.append(node)
            offsets.append(offsets[-1] + node.gates)
        sp.set_body(Body(params, ret_type, nodes, _decode_refs(b["ret"], nodes, schema), offsets))
        return sp

    root = get(desc["root"])
    pfn_fns = tuple(get(fid) for fid in _record_function_ids(desc["program_fn"]) if fid in ddefs)
    return DecodedProgram(root, built, desc, pfn_fns)
