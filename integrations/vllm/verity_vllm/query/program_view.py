"""A `verity.verification.query.Program` view of a derived Program.

Two readers of ONE flat body (`program.fn.body.nodes` of the derived request Program, in body order):

* :func:`from_descriptor` -- the Program descriptor (`descriptor.json.gz`, `verity.ir.codec`): the `program_fn` definition's body
  nodes give every Call's Definition, output type and operand references; `annotations["node_hints"]` gives its name.
* :func:`from_instances` -- the Build's instance sequence (`instances.json.gz`, `derive_step._instance_sequence`): the same nodes
  written one row per Call (`i`, `spec`, `form`, `name`, `args` = operand runs) -- a projection of the descriptor body plus the
  name annotation, produced by the same derivation on meta tensors (nothing executed).  The output type is not in a row: it is
  the registered Definition's signature bound to the row's statics (`verity_vllm.registry`), read once per distinct spec.

Both yield the same :class:`ProgramView`: Calls with ports, reads and names; prescribed inputs as Values of negative ids
(`-(k + 1)` for parameter `k`; a record parameter has one port per field, anything else the port `out`); outputs as the reader
knows them (a descriptor's `ret`; an instance sequence carries none).

Ports.  A Definition returning one collection (`Array<N, Value<w>>`, a `Value<w>`) has the one port `out`; a tuple has the ports
`"0"`, `"1"`, ...; a record its field names.  Every port knows its element count and word width (None when the leaves are not
one width) and its leaf interval in the Call's output, so an operand run `("n", j, lo, hi)` resolves to the ports of Call `j` it
overlaps -- the Value(s) read.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from verity.verification.query import ValueRef

__all__ = ["Port", "CallView", "ProgramView", "InputView", "from_instances", "from_descriptor", "definition_ports", "parse_type_repr",
           "spec_base", "spec_statics", "ports_of_type", "REGISTRY_MODULES"]


@dataclass(frozen=True, slots=True)
class Port:
    """One output collection of a Call: name, leaf interval `[lo, hi)` in the Call's output, element count, word width."""

    name: str
    lo: int
    hi: int
    elems: int
    width: int | None


@dataclass(frozen=True, slots=True)
class CallView:
    id: int
    definition: str                  # the full spec id `Name_vN{k=v,..}`
    family: str                      # `Name_vN`
    form: str                        # call | batch | scan
    name: str                        # the recorded naming hint (`<module>/<fx node>[/<member>][row]`), "" if none
    port_table: tuple[Port, ...]
    reads: tuple[ValueRef, ...]

    @property
    def ports(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.port_table)

    def port(self, name: str) -> Port:
        for p in self.port_table:
            if p.name == name:
                return p
        raise KeyError(f"Call {self.id} ({self.definition}) has no port {name!r}")

    @property
    def statics(self) -> dict[str, Any]:
        return spec_statics(self.definition)


@dataclass(frozen=True, slots=True)
class InputView:
    """One prescribed input (a parameter of the Program function): its id (negative), name, type and ports."""

    id: int
    name: str
    type_repr: str
    port_table: tuple[Port, ...]

    @property
    def ports(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.port_table)


@dataclass(frozen=True)
class ProgramView:
    calls: tuple[CallView, ...]
    params: tuple[InputView, ...]
    outputs: tuple[ValueRef, ...]
    digest: str | None
    source: str                      # "descriptor" | "instances"

    @property
    def inputs(self) -> tuple[ValueRef, ...]:
        return tuple(ValueRef(p.id, port.name) for p in self.params for port in p.port_table)

    def call(self, cid: int) -> CallView:
        c = self.calls[cid]
        if c.id != cid:                                  # ids are body positions for both readers; keep the lookup honest
            for cc in self.calls:
                if cc.id == cid:
                    return cc
            raise KeyError(cid)
        return c

    def input(self, iid: int) -> InputView:
        for p in self.params:
            if p.id == iid:
                return p
        raise KeyError(iid)


# ---------------------------------------------------------------------------------------------------------------------------------
# spec ids and types
# ---------------------------------------------------------------------------------------------------------------------------------

def spec_base(spec: str) -> str:
    """`Gemm_v1{K=2304,N=4096}` -> `Gemm_v1` (a fold's `prefix:` before the record dropped; a `:` inside the record is JSON)."""
    head = spec.split("{", 1)[0]
    return head.rsplit(":", 1)[-1]


def _spec_record_id(spec: str) -> str:
    head, sep, rec = spec.partition("{")
    return head.rsplit(":", 1)[-1] + sep + rec


def _static_value(v: Any) -> Any:
    if isinstance(v, dict):
        if set(v) == {"f64"}:
            return float.fromhex(v["f64"])
        if set(v) == {"fn"}:
            return v["fn"]
        if set(v) == {"dict"}:
            return {k: _static_value(x) for k, x in v["dict"].items()}
        return {k: _static_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return tuple(_static_value(x) for x in v)
    return v


def spec_statics(spec: str) -> dict[str, Any]:
    """The binding record of a spec id as Python values (`{"f64": hex}` -> float, lists -> tuples), in id order."""
    from verity.ir.codec import _split_record
    out: dict[str, Any] = {}
    for k, v in _split_record(_spec_record_id(spec)):
        try:
            out[k] = _static_value(json.loads(v))
        except ValueError:
            out[k] = v
    return out


def ports_of_type(t) -> tuple[Port, ...]:
    """The output ports of a `verity.ir.types.Type`: one `out` for a Value / Array, `"0"`.. for a Tuple, field names for a Record."""
    from verity.ir.types import Record, Tuple as TupleT

    def one(name: str, lo: int, ty) -> Port:
        n = ty.leaves
        return Port(name, lo, lo + n, n, ty.homogeneous_width())

    if isinstance(t, TupleT):
        out, lo = [], 0
        for i, f in enumerate(t.fields):
            out.append(one(str(i), lo, f))
            lo += f.leaves
        return tuple(out)
    if isinstance(t, Record):
        out, lo = [], 0
        for k, f in t.fields:
            out.append(one(k, lo, f))
            lo += f.leaves
        return tuple(out)
    return (one("out", 0, t),)


class TypeReprError(ValueError):
    pass


def parse_type_repr(s: str):
    """The `repr` of a `verity.ir.types.Type` back to a Type: `Value<32>`, `Array<31,Value<32>>`, `(A, B)`, `{name: A, other: B}`."""
    from verity.ir.types import Array, Record, Tuple as TupleT, Value
    pos = 0
    n = len(s)

    def skip_ws() -> None:
        nonlocal pos
        while pos < n and s[pos] in " \n\t":
            pos += 1

    def expect(ch: str) -> None:
        nonlocal pos
        skip_ws()
        if pos >= n or s[pos] != ch:
            raise TypeReprError(f"type repr: expected {ch!r} at {pos} in {s[:80]!r}")
        pos += 1

    def parse():
        nonlocal pos
        skip_ws()
        if s[pos:pos + 6] == "Value<":
            pos += len("Value<")
            j = s.index(">", pos)
            w = int(s[pos:j])
            pos = j + 1
            return Value(w)
        if s[pos:pos + 6] == "Array<":
            pos += len("Array<")
            j = s.index(",", pos)
            k = int(s[pos:j])
            pos = j + 1
            elem = parse()
            expect(">")
            return Array(k, elem)
        if s[pos] == "(":
            pos += 1
            fields = []
            skip_ws()
            while s[pos] != ")":
                fields.append(parse())
                skip_ws()
                if s[pos] == ",":
                    pos += 1
                    skip_ws()
            pos += 1
            return TupleT(*fields)
        if s[pos] == "{":
            pos += 1
            pairs = []
            skip_ws()
            while s[pos] != "}":
                j = s.index(":", pos)
                key = s[pos:j].strip()
                pos = j + 1
                pairs.append((key, parse()))
                skip_ws()
                if s[pos] == ",":
                    pos += 1
                    skip_ws()
            pos += 1
            return Record(*pairs)
        raise TypeReprError(f"type repr: unexpected {s[pos:pos + 20]!r} at {pos}")

    t = parse()
    skip_ws()
    if pos != n:
        raise TypeReprError(f"type repr: trailing {s[pos:pos + 20]!r}")
    return t


#: the integration's registry modules whose Definitions a derived Program names (imported once by `definition_ports`; a module that
#: fails to import is reported with the port lookup that needed it, never treated as "not registered")
REGISTRY_MODULES = ("b1", "dense", "fp8", "moe", "sampling", "b1_tp2", "moe_pad")
_REGISTRY: dict[str, Any] = {}


def _registry():
    if "reg" not in _REGISTRY:
        import importlib
        from verity.ir.defs import REGISTRY
        errors = []
        for m in REGISTRY_MODULES:
            try:
                importlib.import_module(f"verity_vllm.registry.{m}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"verity_vllm.registry.{m}: {type(e).__name__}: {e}")
        _REGISTRY["reg"], _REGISTRY["errors"] = REGISTRY, errors
    return _REGISTRY["reg"], _REGISTRY["errors"]


_PORTS_CACHE: dict[str, tuple[Port, ...]] = {}


def definition_ports(spec: str) -> tuple[Port, ...]:
    """The output ports of the Definition a spec id names, bound to its statics.  Raises `KeyError` (not registered) or `TypeError`
    (statics do not bind) -- a Program row whose Definition the registry does not carry is a Program the registry did not build."""
    hit = _PORTS_CACHE.get(spec)
    if hit is not None:
        return hit
    from verity.ir.codec import lookup_primitive
    reg, errors = _registry()
    base = spec_base(spec)
    d = reg.defs.get(base)
    if d is None:
        try:
            d = lookup_primitive(reg, base)
        except KeyError:
            raise KeyError(f"{base} is not a registered Definition" + (f" (registry import errors: {'; '.join(errors)})" if errors else "")) from None
    if hasattr(d, "bind"):
        sp = d.bind(**spec_statics(spec))
        if not sp.complete:
            missing = [k for k in d.statics if k not in sp.bindings]
            raise TypeError(f"{base}: statics {missing} are not in the spec {spec!r}")
        _params, ret = sp.signature()
    else:
        ret = d.ret
    ports = ports_of_type(ret)
    _PORTS_CACHE[spec] = ports
    return ports


# ---------------------------------------------------------------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------------------------------------------------------------

def _params_view(params: Iterable[tuple[str, Any]]) -> tuple[InputView, ...]:
    out = []
    for k, (name, t) in enumerate(params):
        ty = parse_type_repr(t) if isinstance(t, str) else t
        out.append(InputView(-(k + 1), str(name), repr(ty), ports_of_type(ty)))
    return tuple(out)


def _resolve_reads(args: Iterable[Iterable[Sequence[Any]]], port_of: list[tuple[Port, ...]], params: tuple[InputView, ...],
                   forms: list[str], out: list[ValueRef]) -> None:
    """Operand runs `("n"|"p", idx, lo, hi)` -> the Values read, appended to `out` in operand order, each once."""
    seen: set[ValueRef] = set()
    for operand in args:
        for run in operand:
            space, idx, lo, hi = run[0], int(run[1]), int(run[2]), int(run[3])
            if space == "n":
                table = port_of[idx]
                if len(table) == 1:
                    vs = (ValueRef(idx, table[0].name),)
                else:
                    vs = tuple(ValueRef(idx, p.name) for p in table if p.lo < hi and lo < p.hi)
                    if not vs and forms[idx] == "batch":       # a batch member beyond the single-instance signature: one port
                        vs = (ValueRef(idx, table[0].name),)
            elif space == "p":
                pv = params[idx]
                table = pv.port_table
                if len(table) == 1:
                    vs = (ValueRef(pv.id, table[0].name),)
                else:
                    vs = tuple(ValueRef(pv.id, p.name) for p in table if p.lo < hi and lo < p.hi)
            else:
                continue
            for v in vs:
                if v not in seen:
                    seen.add(v)
                    out.append(v)


def from_instances(path_or_doc: str | dict[str, Any]) -> ProgramView:
    """`instances.json.gz` (or its loaded document) -> ProgramView.  Rows are body positions; the Definition's ports come from the
    registry (`definition_ports`, once per distinct spec); a row of a `batch` form carries no `n`, so its output is read as the
    single-instance signature (the reference request Programs are `call` rows throughout)."""
    if isinstance(path_or_doc, str):
        with gzip.open(path_or_doc, "rt") if path_or_doc.endswith(".gz") else open(path_or_doc) as f:
            doc = json.load(f)
    else:
        doc = path_or_doc
    params = _params_view(doc.get("params") or [])
    rows = doc["rows"]
    port_of: list[tuple[Port, ...]] = [()] * len(rows)
    forms: list[str] = [""] * len(rows)
    for r in rows:
        i = int(r["i"])
        port_of[i] = definition_ports(r["spec"])
        forms[i] = str(r.get("form") or "call")
    calls: list[CallView] = []
    for r in rows:
        i = int(r["i"])
        reads: list[ValueRef] = []
        _resolve_reads(r.get("args") or (), port_of, params, forms, reads)
        spec = r["spec"]
        calls.append(CallView(i, spec, spec_base(spec), forms[i], str(r.get("name") or ""), port_of[i], tuple(reads)))
    calls.sort(key=lambda c: c.id)
    return ProgramView(tuple(calls), params, (), doc.get("digest"), "instances")


def from_descriptor(desc: dict[str, Any] | str, registry=None) -> ProgramView:
    """A Program descriptor (dict, or a path to `descriptor.json[.gz]`) -> ProgramView of its `program_fn` body.  Names from
    `annotations["node_hints"]` (node index -> hint); ports from each node's own output type; outputs from the body's `ret`."""
    if isinstance(desc, str):
        with gzip.open(desc, "rt") if desc.endswith(".gz") else open(desc) as f:
            desc = json.load(f)
    from verity.ir.codec import decode_program
    from verity.ir.refs import runs
    if registry is None:
        registry, _errors = _registry()
    decoded = decode_program(desc, registry)
    fn = decoded.defs[desc["program_fn"]]
    body = fn.body
    params = _params_view((n, t) for n, t in body.params)
    hints = {int(k): str(v) for k, v in (desc.get("annotations") or {}).get("node_hints", {}).items()}
    port_of = [ports_of_type(n.out) for n in body.nodes]
    forms = [n.form for n in body.nodes]
    calls: list[CallView] = []
    for k, n in enumerate(body.nodes):
        spec = _spec_id(n.fn)
        reads: list[ValueRef] = []
        _resolve_reads(([tuple(r) for r in runs(a.refs)] for a in n.args), port_of, params, forms, reads)
        calls.append(CallView(k, spec, spec_base(spec), n.form, hints.get(k, getattr(n, "name", "") or ""), port_of[k], tuple(reads)))
    outs: list[ValueRef] = []
    _resolve_reads([[tuple(r) for r in runs(body.ret)]], port_of, params, forms, outs)
    from verity.ir.codec import program_digest
    return ProgramView(tuple(calls), params, tuple(outs), program_digest(desc), "descriptor")


def _spec_id(fn) -> str:
    from verity.ir.codec import _spec_id as sid
    return sid(fn)
