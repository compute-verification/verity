"""PyTorch authoring frontend to `Program` (lane FE, thin prototype; frontend design §2–§3).

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

PyTorch is a frontend to `Program`, full stop: nothing here gives ATen, FX or `torch.export` any Verity
meaning. Authored bodies call *registered* functions through `ops` (custom ops whose fake implementations
supply shapes only and whose real implementations refuse to run), compose them with our own `batch` / `scan`
(leaf custom ops, never `vmap`), and are exported with `torch.export`. The exported graph is then guarded —
every node must be one of: a Verity op minted here, a static view (`slice`/`select`/`cat`/`stack`/`view`/
`permute`/`unbind`/identity aliases → reference rearrangement, no gates) or `getitem` plumbing — and translated
into `verity.ml.circuits` objects (`CompositeDefinition` bodies over `Builder.call/batch/scan`, `Program`). Everything
else fails closed with a `FrontendError` code.

Elaboration order: a `@function` is exported once per specialisation (static bindings). Children are elaborated
in a shape-only pre-pass on meta tensors *before* the parent's export (no nested `torch.export`), so inside the
parent's trace a child call is a cached leaf op whose output shape is its return type.
"""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import math
import operator
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch.utils import _pytree as pytree

from verity.ml.circuits.codec import _encode_body, _spec_id, canonical_json, program_digest
from verity.ml.circuits.defs import (REGISTRY, Builder, CompositeDefinition, FunctionDefinition, PrimitiveDefinition,
                    SpecializedDefinition, _static_key)
from verity.ml.circuits.defs import bind as ir_bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, Ref, Refs, Strided, array_of, concat
from verity.ml.circuits.liveness import DeadGateReport
from verity.ml.circuits.liveness import dead_gates as _dead_gates
from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Type, Value
from verity_vllm.program.frontend.correspond import compare_bodies
from verity_vllm.program.frontend.provenance import MODULE_WEIGHTS_RULE, RuleApplication, registry_version

FRONTEND_VERSION = "verity_vllm.program.frontend.torch_frontend/2026-09-07.fe2"

# ---------------------------------------------------------------------------------------------------------
# errors: the fail-closed contract
# ---------------------------------------------------------------------------------------------------------

CODES = {
    "E_EXPORT": "torch.export failed or the pre-pass raised (message attached)",
    "E_UNREGISTERED_OP": "a node has no Verity translation: ATen op, foreign custom op, Python callable",
    "E_MUTATION": "an in-place / mutating op, or an input/buffer mutation recorded by export",
    "E_HIDDEN_STATE": "lifted parameter, buffer or tensor constant not declared as a prescribed input",
    "E_OPAQUE_HOST": "data-dependent host escape (.item(), Python control on values, host callback)",
    "E_DYNAMIC_CONTROL_FLOW": "cond / while_loop / map or another higher-order op; subgraph attributes",
    "E_UNSUPPORTED_VIEW": "a view outside the affine/strided whitelist (inner-dim cat, expand, step != 1, ...)",
    "E_WIDTH": "tensor width does not match the declared Value<w> of the registered function",
    "E_SHAPE": "shape does not conform to the declared schema / static dimensions",
    "E_STATIC": "a static parameter could not be resolved, or a static value is not a finite binding",
    "E_ARITY": "wrong number of arguments for a registered function",
    "E_RETURN_TYPE": "returned pytree does not match the return annotation / scan step contract",
    "E_IDENTITY": "two different bodies claim one fully qualified specialisation (name + version + statics)",
    "E_DEAD_GATES": "the exported program has dead gates and export was asked to be strict about them",
}


class DeadGateWarning(UserWarning):
    """Authoring-mode warning: the exported program has gates no gate or root output consumes (§8.6)."""


class FrontendError(Exception):
    def __init__(self, code: str, message: str, where: str | None = None) -> None:
        assert code in CODES, code
        self.code, self.message, self.where = code, message, where
        super().__init__(f"{code}{' @ ' + where if where else ''}: {message}")


# ---------------------------------------------------------------------------------------------------------
# Value<w> <-> tensors: width only; numerical interpretation belongs to functions (IR §3.1)
# ---------------------------------------------------------------------------------------------------------

WIDTH_OF_DTYPE = {torch.bool: 1, torch.uint8: 8, torch.int8: 8, torch.int16: 16, torch.uint16: 16, torch.bfloat16: 16,
                  torch.float16: 16, torch.int32: 32, torch.uint32: 32, torch.float32: 32, torch.int64: 64,
                  torch.uint64: 64, torch.float64: 64,
                  # [R15 fp8] the OCP FP8 words vLLM's FP8 linears carry (e4m3 weights/activations of a block-scaled checkpoint):
                  # 8-bit VALUES, carried as uint8 (`CARRIER[8]`); their arithmetic is the registered FP8 Definitions' (registry.fp8)
                  torch.float8_e4m3fn: 8, torch.float8_e5m2: 8}
CARRIER = {1: torch.bool, 8: torch.uint8, 16: torch.bfloat16, 32: torch.float32, 64: torch.int64}


def width_of(dtype: torch.dtype, where: str | None = None) -> int:
    try:
        return WIDTH_OF_DTYPE[dtype]
    except KeyError:
        raise FrontendError("E_WIDTH", f"dtype {dtype} carries no Value<w> width", where) from None


def type_of_tensor(t: torch.Tensor) -> Type:
    T: Type = Value(width_of(t.dtype))
    for n in reversed(tuple(t.shape)):
        T = Array(int(n), T)
    return T


def type_of(obj: Any, where: str | None = None) -> Type:
    """Tensor -> Array<..., Value<w>>; tuple -> Tuple; dict -> Record; list -> Array<n, T> (homogeneous)."""
    if isinstance(obj, torch.Tensor):
        return type_of_tensor(obj)
    if isinstance(obj, tuple):
        return TupleT(*[type_of(o, where) for o in obj])
    if isinstance(obj, dict):
        return Record(*[(k, type_of(v, where)) for k, v in obj.items()])
    if isinstance(obj, list):
        if not obj:
            raise FrontendError("E_SHAPE", "an empty list has no element type", where)
        ts = [type_of(o, where) for o in obj]
        if any(t != ts[0] for t in ts):
            raise FrontendError("E_SHAPE", f"list elements differ in type: {ts[0]!r} vs {[t for t in ts if t != ts[0]][0]!r}", where)
        return Array(len(ts), ts[0])
    raise FrontendError("E_UNREGISTERED_OP", f"a {type(obj).__name__} is not a value (values are tensors/tuples/dicts/lists)", where)


def is_tensor_type(T: Type) -> bool:
    return isinstance(T, Value) or (isinstance(T, Array) and is_tensor_type(T.elem))


def shape_of(T: Type) -> tuple[tuple[int, ...], int]:
    shape: list[int] = []
    while isinstance(T, Array):
        shape.append(T.n)
        T = T.elem
    assert isinstance(T, Value)
    return tuple(shape), T.w


def flat_types(T: Type) -> list[Type]:
    """Maximal tensor chunks of a type, in leaf order (== pytree flatten order of `tensors_of(T)`)."""
    if is_tensor_type(T):
        return [T]
    if isinstance(T, TupleT):
        return [c for f in T.fields for c in flat_types(f)]
    if isinstance(T, Record):
        return [c for _, f in T.fields for c in flat_types(f)]
    assert isinstance(T, Array)
    return [c for _ in range(T.n) for c in flat_types(T.elem)]


def tensors_of(T: Type, device: str = "meta") -> Any:
    if is_tensor_type(T):
        shape, w = shape_of(T)
        return torch.empty(shape, dtype=CARRIER[w], device=device)
    if isinstance(T, TupleT):
        return tuple(tensors_of(f, device) for f in T.fields)
    if isinstance(T, Record):
        return {k: tensors_of(f, device) for k, f in T.fields}
    assert isinstance(T, Array)
    return [tensors_of(T.elem, device) for _ in range(T.n)]


def unflatten(T: Type, flat: list[torch.Tensor]) -> Any:
    it = iter(flat)

    def go(t: Type):
        if is_tensor_type(t):
            return next(it)
        if isinstance(t, TupleT):
            return tuple(go(f) for f in t.fields)
        if isinstance(t, Record):
            return {k: go(f) for k, f in t.fields}
        return [go(t.elem) for _ in range(t.n)]

    out = go(T)
    assert next(it, None) is None
    return out


def split_coll(c: Coll) -> list[Coll]:
    """Mirror of `flat_types` on a collection of references."""
    T = c.type
    if is_tensor_type(T):
        return [c]
    if isinstance(T, (TupleT, Record)):
        return [p for i in range(len(T.fields)) for p in split_coll(c[i])]
    return [p for i in range(T.n) for p in split_coll(c[i])]


def join_coll(T: Type, flats: Sequence[Coll]) -> Coll:
    """Assemble a collection of type `T` from its maximal tensor chunks, nesting the reference runs exactly as
    TA1's `tuple_of`/`record_of`/`array_of` do (a `concat` per structural level), so that a structured return
    such as `((nb, ni, nxt), ni)` encodes as `tuple_of(tuple_of(nb, ni, nxt), ni)` and digests identically."""
    it = iter(flats)

    def go(t: Type) -> Refs:
        if is_tensor_type(t):
            return next(it).refs
        if isinstance(t, TupleT):
            return concat([go(f) for f in t.fields])
        if isinstance(t, Record):
            return concat([go(f) for _, f in t.fields])
        assert isinstance(t, Array)
        return concat([go(t.elem) for _ in range(t.n)])

    refs = go(T)
    if next(it, None) is not None:
        raise FrontendError("E_SHAPE", f"more tensor chunks than {T!r} has leaves", "join")
    return Coll(T, refs)


# ---------------------------------------------------------------------------------------------------------
# shape schema in annotations: Arr["N", "K", BF16], Tup[...], Value(w)
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArrSchema:
    dims: tuple[int | str, ...]
    width: int


@dataclass(frozen=True)
class _TupSchema:
    fields: tuple[Any, ...]


class Arr:
    """`Arr["K", BF16]`, `Arr["N", "K", BF16]`, `Arr[16, BF16]`, `Arr["2*I", BF16]`: dims are ints, static names
    (solved from the argument shape) or `<int>*NAME` / expressions over statics (checked)."""

    def __class_getitem__(cls, item):
        items = item if isinstance(item, tuple) else (item,)
        *dims, last = items
        if isinstance(last, Value):
            return _ArrSchema(tuple(dims), last.w)
        if isinstance(last, _ArrSchema):
            return _ArrSchema(tuple(dims) + last.dims, last.width)
        raise TypeError("Arr[...] must end in a Value<w> (e.g. BF16) or a nested Arr")


class Tup:
    def __class_getitem__(cls, item):
        items = item if isinstance(item, tuple) else (item,)
        return _TupSchema(tuple(_schema(i) for i in items))


class Dep:
    """`w: Dep(lambda S: layer_weights_type(S["C"]))`: a parameter (or return) whose *type* is a function of the
    statics — records and other structured collections a static configuration determines. The argument is
    checked against the computed type once the statics are known; `specialize(**statics)` computes it."""

    def __init__(self, fn: Callable[[dict], Type]) -> None:
        self.fn = fn

    def __repr__(self) -> str:
        return f"Dep({getattr(self.fn, '__name__', 'λ')})"


def _schema(a):
    if a is inspect.Parameter.empty:
        return None
    if isinstance(a, Value):
        return _ArrSchema((), a.w)
    if isinstance(a, (_ArrSchema, _TupSchema, Dep)):
        return a
    raise FrontendError("E_SHAPE", f"unsupported annotation {a!r} (use Arr[...], Tup[...], Dep(...) or a Value<w>)")


_LIN = re.compile(r"^(\d+)\*([A-Za-z_]\w*)$")


def _unify(sch, T: Type, statics: dict, deferred: list, where: str) -> None:
    if sch is None:
        return
    if isinstance(sch, Dep):
        deferred.append((sch, T))  # checked once every static is known
        return
    if isinstance(sch, _TupSchema):
        if not isinstance(T, TupleT) or len(T.fields) != len(sch.fields):
            raise FrontendError("E_SHAPE", f"expected a {len(sch.fields)}-tuple, got {T!r}", where)
        for s, f in zip(sch.fields, T.fields):
            _unify(s, f, statics, deferred, where)
        return
    for d in sch.dims:
        if not isinstance(T, Array):
            raise FrontendError("E_SHAPE", f"schema {sch} needs another array axis, got {T!r}", where)
        if isinstance(d, int):
            if T.n != d:
                raise FrontendError("E_SHAPE", f"axis {d} expected, got {T.n}", where)
        elif d.isidentifier():
            if d in statics:
                if statics[d] != T.n:
                    raise FrontendError("E_SHAPE", f"static {d}={statics[d]} but the argument has axis {T.n}", where)
            else:
                statics[d] = T.n
        else:
            m = _LIN.match(d.replace(" ", ""))
            if m and m.group(2) not in statics:
                a, name = int(m.group(1)), m.group(2)
                if T.n % a:
                    raise FrontendError("E_SHAPE", f"axis {T.n} is not a multiple of {a} ({d})", where)
                statics[name] = T.n // a
            else:
                deferred.append((d, T.n))
        T = T.elem
    if not isinstance(T, Value) or T.w != sch.width:
        code = "E_WIDTH" if isinstance(T, Value) else "E_SHAPE"
        raise FrontendError(code, f"expected Value<{sch.width}>, got {T!r}", where)


def _check_deferred(deferred: list, statics: dict, where: str) -> None:
    """Dimension expressions and `Dep` types that could only be checked once every static was known."""
    for expr, n in deferred:
        if isinstance(expr, Dep):
            want = _dep_type(expr, statics, where)
            if want != n:
                raise FrontendError("E_SHAPE", f"{expr!r} computes {want!r} but the argument has type {n!r}", where)
        elif _eval_dim(expr, statics, where) != n:
            raise FrontendError("E_SHAPE", f"dimension {expr!r} = {_eval_dim(expr, statics, where)} but the argument has {n}", where)


def _dep_type(sch: Dep, statics: dict, where: str) -> Type:
    try:
        t = sch.fn(dict(statics))
    except Exception as e:  # noqa: BLE001
        raise FrontendError("E_STATIC", f"{sch!r} failed on statics {sorted(statics)}: {type(e).__name__}: {e}", where) from None
    if not isinstance(t, Type):
        raise FrontendError("E_STATIC", f"{sch!r} did not return a Type", where)
    return t


def _eval_dim(expr: str, statics: dict, where: str) -> int:
    try:
        v = eval(expr, {"__builtins__": {}}, {k: v for k, v in statics.items() if isinstance(v, int)})
    except Exception as e:  # noqa: BLE001
        raise FrontendError("E_STATIC", f"cannot evaluate dimension {expr!r}: {e}", where) from None
    if not isinstance(v, int):
        raise FrontendError("E_STATIC", f"dimension {expr!r} is not an int", where)
    return v


def _type_of_schema(sch, statics: dict, where: str) -> Type:
    if isinstance(sch, Dep):
        return _dep_type(sch, statics, where)
    if isinstance(sch, _TupSchema):
        return TupleT(*[_type_of_schema(f, statics, where) for f in sch.fields])
    T: Type = Value(sch.width)
    for d in reversed(sch.dims):
        n = d if isinstance(d, int) else _eval_dim(d, statics, where)
        T = Array(n, T)
    return T


# ---------------------------------------------------------------------------------------------------------
# minted custom ops: `verity::<kind>_<hash>(Tensor[] args, str name) -> Tensor[]`
# ---------------------------------------------------------------------------------------------------------

_LIB = torch.library.Library("verity", "FRAGMENT")


@dataclass
class _OpInfo:
    kind: str  # call | batch | scan
    fn: PrimitiveDefinition | SpecializedDefinition
    arg_types: tuple[Type, ...]  # collection type of each argument as passed (batch: Array<n, P> for mapped)
    axes: tuple
    n: int
    n_xs: int
    out_type: Type


_OPINFO: dict[str, _OpInfo] = {}
_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("verity_trace_depth", default=0)


def _tracing() -> bool:
    return _DEPTH.get() > 0


def _hex16(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:16]


def _no_semantics(*_a, **_k):
    raise NotImplementedError("Verity ops have no torch execution semantics; export them to a Program")


def _mint(kind: str, fn, arg_types: Sequence[Type], axes: tuple, n: int, n_xs: int, out_type: Type) -> str:
    key = canonical_json([kind, _spec_id(fn), [t.to_json() for t in arg_types], list(axes), n, n_xs])
    opname = f"{kind}_{_hex16(key)}"
    info = _OPINFO.get(opname)
    if info is not None and info.fn is not fn:
        # another definition object already claims this fully qualified specialisation: it must be the same body
        # (§8.1: never first-wins)
        fid = _spec_id(fn)
        if isinstance(fn, PrimitiveDefinition) or isinstance(info.fn, PrimitiveDefinition):
            if not (isinstance(fn, PrimitiveDefinition) and isinstance(info.fn, PrimitiveDefinition)
                    and fn.params == info.fn.params and fn.ret == info.fn.ret):
                raise FrontendError("E_IDENTITY", f"two different definitions claim {fid}", fid)
        else:
            v = compare_bodies(info.fn.body, fn.body, fid=fid)
            if v.verdict not in ("identical", "same-references"):
                raise FrontendError("E_IDENTITY", f"two different bodies claim {fid} ({v.verdict}: {v.detail}); give one of them a new version", fid)
    if info is None:
        _LIB.define(f"{opname}(Tensor[] args, str name) -> Tensor[]")
        outs = [shape_of(t) for t in flat_types(out_type)]

        def fake(args, name, _outs=outs):
            dev = args[0].device if args else torch.device("meta")
            return [torch.empty(s, dtype=CARRIER[w], device=dev) for s, w in _outs]

        torch.library.register_fake(f"verity::{opname}", fake)
        _LIB.impl(opname, _no_semantics, "CompositeExplicitAutograd")
        _OPINFO[opname] = _OpInfo(kind, fn, tuple(arg_types), tuple(axes), n, n_xs, out_type)
    return opname


def _emit(kind: str, fn, args: Sequence[Any], arg_types: Sequence[Type], out_type: Type, axes=(), n=1, n_xs=0, name: str = ""):
    if not _tracing():  # pre-pass: shapes only
        return tensors_of(out_type, "meta")
    opname = _mint(kind, fn, arg_types, tuple(axes), n, n_xs, out_type)
    flat = [t for a in args for t in pytree.tree_leaves(a)]
    outs = getattr(torch.ops.verity, opname)(flat, name)
    return unflatten(out_type, list(outs))


# ---------------------------------------------------------------------------------------------------------
# callables usable inside authored bodies
# ---------------------------------------------------------------------------------------------------------


class _Callable:
    id: str

    def _resolve(self, arg_types: Sequence[Type], kwargs: dict) -> tuple[Any, tuple[Type, ...], Type]:
        """(fn definition, declared parameter types, return type) for these argument types and statics."""
        raise NotImplementedError

    def bind(self, **statics) -> "Bound":
        return Bound(self, statics)

    def __call__(self, *args, name: str = "", **kwargs):
        return _call(self, args, kwargs, name)


def _check_types(arg_types: Sequence[Type], params: Sequence[Type], where: str) -> None:
    if len(arg_types) != len(params):
        raise FrontendError("E_ARITY", f"expected {len(params)} arguments, got {len(arg_types)}", where)
    for i, (a, p) in enumerate(zip(arg_types, params)):
        if a != p:
            code = "E_WIDTH" if (is_tensor_type(a) and is_tensor_type(p) and shape_of(a)[0] == shape_of(p)[0]) else "E_SHAPE"
            raise FrontendError(code, f"argument {i} has type {a!r}, declared {p!r}", where)


def _call(target: _Callable, args, kwargs, name):
    arg_types = tuple(type_of(a, target.id) for a in args)
    fn, params, ret = target._resolve(arg_types, kwargs)
    _check_types(arg_types, params, target.id)
    return _emit("call", fn, args, arg_types, ret, name=name)


class Prim(_Callable):
    """A registered one-output primitive as a callable (its call is one gate)."""

    def __init__(self, p: PrimitiveDefinition) -> None:
        self.p, self.id = p, p.id

    def _resolve(self, arg_types, kwargs):
        if kwargs:
            raise FrontendError("E_STATIC", f"primitive {self.id} takes no statics {sorted(kwargs)}", self.id)
        return self.p, tuple(t for _, t in self.p.params), self.p.ret


class Registered(_Callable):
    """An already-registered composite (e.g. TA1's `Gemm`): statics must be given explicitly."""

    def __init__(self, d: CompositeDefinition) -> None:
        self.d, self.id = d, d.id

    def _resolve(self, arg_types, kwargs):
        try:
            sd = ir_bind(self.d, **{k: _ir_static(v, self.id) for k, v in kwargs.items()})
            params, ret = sd.signature()
        except TypeError as e:
            raise FrontendError("E_STATIC", str(e), self.id) from None
        return sd, tuple(t for _, t in params), ret

    def specialize(self, **statics) -> SpecializedDefinition:
        return self._resolve((), statics)[0]

    def __call__(self, *args, name: str = "", **kwargs):
        if not args:  # `ops.Gemm(K=32, N=4)` -> bound callable
            return Bound(self, kwargs)
        return _call(self, args, kwargs, name)


class Bound(_Callable):
    def __init__(self, target: _Callable, statics: dict) -> None:
        self.target, self.statics = target, dict(statics)
        self.id = f"{target.id}{{{','.join(f'{k}={_fmt(v)}' for k, v in self.statics.items())}}}"

    def _resolve(self, arg_types, kwargs):
        return self.target._resolve(arg_types, {**self.statics, **kwargs})

    def specialize(self, **statics) -> SpecializedDefinition:
        return self.target.specialize(**{**self.statics, **statics})


def _fmt(v) -> str:
    return getattr(v, "id", None) or repr(v)


def _ir_static(v, where: str):
    """A static value as `verity.ml.circuits` accepts it: a finite binding record, never a closure. An `Ops` bundle is
    the record of its role bindings (§8.3: the selected functions are part of the specialisation's identity)."""
    if isinstance(v, Ops):
        return {k: _ir_static(f, where) for k, f in sorted(v._roles.items())}
    if isinstance(v, Prim):
        return v.p
    if isinstance(v, Registered):
        return v.d
    if isinstance(v, Function):
        if v.static:
            raise FrontendError("E_STATIC", f"{v.id} has statics {v.static}; bind them before passing it as a static", where)
        return v.defn
    if isinstance(v, Bound):
        base = _ir_static(v.target, where)
        return ir_bind(base, **{k: _ir_static(x, where) for k, x in v.statics.items()})
    if isinstance(v, (FunctionDefinition, int, str, bool, float)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return tuple(_ir_static(x, where) for x in v)
    if isinstance(v, dict):
        return {k: _ir_static(x, where) for k, x in v.items()}
    raise FrontendError("E_STATIC", f"{type(v).__name__} is not a finite static binding", where)


@dataclass
class _Spec:
    param_types: tuple[Type, ...]
    ret_type: Type
    ep: Any  # torch.export.ExportedProgram
    erased: tuple  # key of the authoring-level (non-static) bindings that shaped this body
    accepted: set = field(default_factory=set)  # other erased keys verified to produce this same body


class _Wrap(torch.nn.Module):
    def __init__(self, fn: Callable, in_spec, kw: dict) -> None:
        super().__init__()
        self._fn, self._in_spec, self._kw = fn, in_spec, kw

    def forward(self, *flat):
        tok = _DEPTH.set(_DEPTH.get() + 1)
        try:
            out = self._fn(*pytree.tree_unflatten(list(flat), self._in_spec), **self._kw)
        finally:
            _DEPTH.reset(tok)
        return tuple(pytree.tree_leaves(out))


class Function(_Callable):
    """`@function(static=(...))`: an authored composite. Identity `Name_vN{statics}`; the body of each
    specialisation is the guarded translation of its exported graph."""

    def __init__(self, fn: Callable, *, name: str | None = None, version: int = 1, static: Sequence[str] = (),
                 register: bool = False) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.version = version
        self.static = tuple(static)
        self.id = f"{self.name}_v{version}"
        sig = inspect.signature(fn, eval_str=True)
        self.params = [p for p in sig.parameters.values()
                       if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        self.kwonly = {p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY}
        self.schemas = [_schema(p.annotation) for p in self.params]
        self.ret_schema = _schema(sig.return_annotation) if sig.return_annotation is not inspect.Signature.empty else None
        self.defn = CompositeDefinition(self.name, version, self.static, self._signature, self._body,
                                        doc=fn.__doc__ or "", register=register)
        self._specs: dict[tuple, _Spec] = {}

    # -- resolution ---------------------------------------------------------------------------------
    def _resolve(self, arg_types, kwargs):
        if len(arg_types) != len(self.params):
            raise FrontendError("E_ARITY", f"expected {len(self.params)} arguments, got {len(arg_types)}", self.id)
        statics = dict(kwargs)
        deferred: list = []
        for sch, at, p in zip(self.schemas, arg_types, self.params):
            _unify(sch, at, statics, deferred, f"{self.id}.{p.name}")
        _check_deferred(deferred, statics, self.id)
        missing = [s for s in self.static if s not in statics]
        if missing:
            raise FrontendError("E_STATIC", f"unresolved statics {missing}", self.id)
        declared = {k: _ir_static(statics[k], self.id) for k in self.static}
        erased = tuple(sorted((k, _fmt(v)) for k, v in statics.items() if k not in self.static))
        sd = ir_bind(self.defn, **declared)
        param_types = tuple(_type_of_schema(sch, statics, self.id) if sch is not None else at
                            for sch, at in zip(self.schemas, arg_types))
        spec = self._specs.get(sd.key)
        if spec is None:
            spec = self._elaborate(param_types, statics, erased)
            self._specs[sd.key] = spec
        elif spec.param_types != param_types:
            raise FrontendError("E_SHAPE", f"{sd.id} is monomorphic in its unannotated parameters: {spec.param_types!r} vs {param_types!r}", self.id)
        elif spec.erased != erased and erased not in spec.accepted:
            # The same fully qualified specialisation reached under different authoring-level (undeclared)
            # bindings. That is an error exactly when the two bodies differ (§8.1/§8.3) — never first-wins,
            # never an error for bindings the body does not depend on.
            other = self._elaborate(param_types, statics, erased)
            v = compare_bodies(sd.body, self._fresh_body(other), fid=sd.id)
            if v.verdict not in ("identical", "same-references"):
                raise FrontendError("E_IDENTITY", f"two different bodies claim {sd.id}: elaborated with bindings {dict(spec.erased)} "
                                                  f"and now with {dict(erased)} ({v.verdict}: {v.detail}). Declare the binding in "
                                                  f"static=(...) so that it is part of the specialisation's identity", self.id)
            spec.accepted.add(erased)
        return sd, spec.param_types, spec.ret_type

    def _elaborate(self, param_types: tuple[Type, ...], statics: dict, erased: tuple) -> _Spec:
        kw = {k: v for k, v in statics.items() if k in self.kwonly}
        ex = [tensors_of(t, "meta") for t in param_types]
        if _tracing():
            raise FrontendError("E_EXPORT", f"{self.id} was first reached inside a trace (non-deterministic authoring?)", self.id)
        try:  # pre-pass: resolves children, gives the return type
            out = self.fn(*ex, **kw)
        except FrontendError:
            raise
        except Exception as e:  # noqa: BLE001
            raise FrontendError(_classify(e), f"{type(e).__name__}: {str(e).splitlines()[0][:200]}", self.id) from e
        ret_type = type_of(out, self.id)
        if self.ret_schema is not None:
            try:
                deferred: list = []
                _unify(self.ret_schema, ret_type, dict(statics), deferred, self.id)
                _check_deferred(deferred, statics, self.id)
            except FrontendError as e:
                raise FrontendError("E_RETURN_TYPE", f"return {ret_type!r} does not match the annotation ({e.message})", self.id) from None
        flat, in_spec = pytree.tree_flatten(ex)
        try:
            ep = torch.export.export(_Wrap(self.fn, in_spec, kw), tuple(flat), strict=False)
        except FrontendError:
            raise
        except Exception as e:  # noqa: BLE001
            raise FrontendError(_classify(e), f"{type(e).__name__}: {str(e).splitlines()[0][:200]}", self.id) from e
        _guard(ep, self.id)
        return _Spec(param_types, ret_type, ep, erased)

    def _fresh_body(self, spec: _Spec):
        """Translate `spec` into a body of its own (not through the specialisation's cache)."""
        params = tuple((p.name, t) for p, t in zip(self.params, spec.param_types))
        B = Builder(params, spec.ret_type)
        args = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
        return B.finish(_translate(spec, B, args, self.id))

    def specialize(self, **statics) -> SpecializedDefinition:
        """Elaborate from statics alone (every parameter annotated, every dimension determined by the statics)."""
        if any(s is None for s in self.schemas):
            raise FrontendError("E_SHAPE", f"{self.id} has unannotated parameters; use export(fn, example_inputs)", self.id)
        param_types = tuple(_type_of_schema(s, statics, self.id) for s in self.schemas)
        sd, _, _ = self._resolve(param_types, statics)
        return sd

    # -- verity.ml.circuits hooks ----------------------------------------------------------------------------
    def _key(self, S) -> tuple:
        return tuple((k, _static_key(getattr(S, k))) for k in self.static)

    def _signature(self, S):
        spec = self._specs.get(self._key(S))
        if spec is None:
            raise FrontendError("E_STATIC", f"{self.id} was not elaborated for these statics; call it through the frontend", self.id)
        return tuple((p.name, t) for p, t in zip(self.params, spec.param_types)), spec.ret_type

    def _body(self, B: Builder, S, *args: Coll) -> Coll:
        spec = self._specs[self._key(S)]
        return _translate(spec, B, args, self.id)


def _classify(e: Exception) -> str:
    name, msg = type(e).__name__, str(e)
    if "GuardOnDataDependent" in name or "DataDependent" in name or "meta tensor" in msg or "data-dependent" in msg:
        return "E_OPAQUE_HOST"
    if "HigherOrder" in name or "cond" in msg.lower() or "while_loop" in msg:
        return "E_DYNAMIC_CONTROL_FLOW"
    return "E_EXPORT"


def function(fn: Callable | None = None, *, name: str | None = None, version: int = 1, static: Sequence[str] = (),
             register: bool = False):
    """Decorator: `@function`, `@function(static=("K", "N"))`, `@function(name="Gemm", version=1, static=...)`."""
    if fn is not None:
        return Function(fn, name=name, version=version, static=static, register=register)
    return lambda f: Function(f, name=name, version=version, static=static, register=register)


def bind(fn: _Callable, **statics) -> Bound:
    return Bound(fn, statics)


def batch(fn: _Callable, *args, axes: Sequence[int | None], n: int | None = None, name: str = "", **statics):
    """`batch(F, args..., axes=(0, None, ...))`: `n` dataflow-independent calls of `F`; axis 0 maps the leading
    array axis (a tensor's dim 0 or a list's index), `None` shares the argument. Other axes: move them with a
    view first (E_UNSUPPORTED_VIEW otherwise)."""
    where = f"batch({fn.id})"
    axes = tuple(axes)
    if len(axes) != len(args):
        raise FrontendError("E_ARITY", "one axis entry per argument", where)
    arg_types = tuple(type_of(a, where) for a in args)
    members: list[Type] = []
    lengths: set[int] = set()
    for at, ax in zip(arg_types, axes):
        if ax is None:
            members.append(at)
        elif ax == 0:
            if not isinstance(at, Array):
                raise FrontendError("E_SHAPE", f"axis 0 requires an array, got {at!r}", where)
            members.append(at.elem)
            lengths.add(at.n)
        else:
            raise FrontendError("E_UNSUPPORTED_VIEW", f"axis {ax}: only axis 0 / None are supported; permute first", where)
    if len(lengths) > 1:
        raise FrontendError("E_SHAPE", f"mapped axes have different lengths {sorted(lengths)}", where)
    if lengths:
        if n is not None and n != next(iter(lengths)):
            raise FrontendError("E_SHAPE", "explicit n disagrees with the mapped axis length", where)
        n = next(iter(lengths))
    if n is None:
        raise FrontendError("E_SHAPE", "n is required when no argument is mapped", where)
    fdef, params, ret = fn._resolve(tuple(members), statics)
    _check_types(members, params, where)
    return _emit("batch", fdef, args, arg_types, Array(n, ret), axes=axes, n=n, name=name)


def scan(step: _Callable, init, xs: Sequence[Any] = (), shared: Sequence[Any] = (), name: str = "", **statics):
    """`scan(Step, init, xs, shared)`: `Step(carry, x_1..x_k, s_1..s_m) -> (carry', y)`; returns `(carry_n, ys)`."""
    where = f"scan({step.id})"
    xs, shared = tuple(xs), tuple(shared)
    carry_t = type_of(init, where)
    x_types = tuple(type_of(x, where) for x in xs)
    for xt in x_types:
        if not isinstance(xt, Array):
            raise FrontendError("E_SHAPE", f"xs arguments must be arrays, got {xt!r}", where)
    lengths = {xt.n for xt in x_types}
    if len(lengths) != 1:
        raise FrontendError("E_SHAPE", f"xs lengths must agree and be present: {sorted(lengths)}", where)
    n = lengths.pop()
    shared_t = tuple(type_of(s, where) for s in shared)
    members = (carry_t,) + tuple(xt.elem for xt in x_types) + shared_t
    fdef, params, ret = step._resolve(members, statics)
    _check_types(members, params, where)
    if not (isinstance(ret, TupleT) and len(ret.fields) == 2 and ret.fields[0] == carry_t):
        raise FrontendError("E_RETURN_TYPE", f"step must return (carry: {carry_t!r}, y), got {ret!r}", where)
    out_t = TupleT(carry_t, Array(n, ret.fields[1]))
    return _emit("scan", fdef, (init,) + xs + shared, (carry_t,) + x_types + shared_t, out_t, n=n, n_xs=len(xs), name=name)


# ---------------------------------------------------------------------------------------------------------
# ops: static function bindings by name, resolved against the registry (never torch semantics)
# ---------------------------------------------------------------------------------------------------------


class Ops:
    """`ops.Bf16Add`, `ops["AmpereBF16TcDot16_v2"]`, `ops.const(32, 0)`, `ops.Gemm(K=.., N=..)`; role names
    (`ops.dot`, `ops.rmsnorm`) come from a configuration: `Ops(dot=ops.AmpereBF16TcDot16, ...)`."""

    def __init__(self, registry=REGISTRY, **roles: _Callable) -> None:
        self._registry, self._roles = registry, dict(roles)

    @property
    def id(self) -> str:
        """`Ops{dot=AmpereBF16TcDot16_v2,...}`: the bundle is identified by its role bindings only."""
        return "Ops{" + ",".join(f"{k}={_fmt(v)}" for k, v in sorted(self._roles.items())) + "}"

    @property
    def roles(self) -> dict[str, _Callable]:
        return dict(self._roles)

    def with_roles(self, **roles: _Callable) -> "Ops":
        return Ops(self._registry, **{**self._roles, **roles})

    def const(self, w: int, bits: int, name: str = ""):
        """`Const<w>[0x..]`: a zero-arity gate with a fixed correct output (emitted immediately)."""
        from verity_vllm.program.registry.b1 import const
        return Prim(const(w, bits))(name=name)

    def gather_bf16(self, vocab: int) -> Prim:
        from verity_vllm.program.registry.prims import gather_bf16
        return Prim(gather_bf16(vocab))

    def __getitem__(self, fid: str) -> _Callable:
        if fid in self._roles:
            return self._roles[fid]
        vs = [n] if (n := fid if "_v" in fid else f"{fid}_v1") in self._registry.defs or "_v" in fid else [k for k, d in self._registry.defs.items() if k.startswith(f"{fid}_v") and k[len(fid) + 2:].isdigit() and not hasattr(d, "word")]
        if len(vs) != 1 or (d := self._registry.defs.get(vs[0])) is None:     # an unversioned name: its v1, else its one registered word version (never a Boolean one)
            raise FrontendError("E_UNREGISTERED_OP", f"no registered function {fid!r}")
        return Prim(d) if isinstance(d, PrimitiveDefinition) else Registered(d)

    def __getattr__(self, name: str) -> _Callable:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]


ops = Ops()


# ---------------------------------------------------------------------------------------------------------
# the guard: every accepted node has an explicit Verity translation; everything else is an error
# ---------------------------------------------------------------------------------------------------------

aten = torch.ops.aten
_IDENTITY = {aten.alias.default, aten.detach.default, aten.clone.default, aten.contiguous.default, aten.lift_fresh.default}
_RESHAPE = {aten.view.default, aten.reshape.default, aten._unsafe_view.default, aten.flatten.using_ints, aten.squeeze.dim,
            aten.squeeze.default, aten.squeeze.dims, aten.unsqueeze.default}
_VIEWS = _IDENTITY | _RESHAPE | {aten.slice.Tensor, aten.select.int, aten.cat.default, aten.stack.default, aten.expand.default,
                                  aten.permute.default, aten.t.default, aten.transpose.int, aten.unbind.int, aten.view.dtype}


def _guard(ep, where: str) -> None:
    sig = ep.graph_signature
    hidden = dict(sig.inputs_to_parameters) | dict(sig.inputs_to_buffers) | dict(sig.inputs_to_lifted_tensor_constants)
    if hidden:
        raise FrontendError("E_HIDDEN_STATE", f"lifted parameters/buffers/constants {sorted(hidden.values())}: pass them as "
                                              f"explicit inputs (they are Input gates) or author constants with ops.const", where)
    if sig.user_inputs_to_mutate or sig.buffers_to_mutate:
        raise FrontendError("E_MUTATION", f"export recorded mutations {dict(sig.user_inputs_to_mutate)} {dict(sig.buffers_to_mutate)}", where)
    for node in ep.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "get_attr":
            raise FrontendError("E_DYNAMIC_CONTROL_FLOW", f"subgraph attribute {node.target} (higher-order op body)", where)
        if node.op != "call_function":
            raise FrontendError("E_UNREGISTERED_OP", f"{node.op} {node.target}", where)
        t = node.target
        if t is operator.getitem:
            continue
        if isinstance(t, torch._ops.HigherOrderOperator):
            raise FrontendError("E_DYNAMIC_CONTROL_FLOW", f"higher-order op {t}: data-dependent control flow has no Verity meaning", where)
        if isinstance(t, torch._ops.OpOverload):
            if t.namespace == "verity":
                if t.__name__.split(".")[0] not in _OPINFO:
                    raise FrontendError("E_UNREGISTERED_OP", f"{t} was not minted by this frontend", where)
                continue
            if t._schema.is_mutable:
                raise FrontendError("E_MUTATION", f"{t} mutates its operand; Verity bodies are pure dataflow", where)
            if t in _VIEWS:
                continue
            if t.namespace == "aten":
                raise FrontendError("E_UNREGISTERED_OP", f"ATen op {t} has no Verity meaning; call a registered function through ops.*", where)
            raise FrontendError("E_UNREGISTERED_OP", f"custom op {t} has no Verity binding", where)
        raise FrontendError("E_UNREGISTERED_OP", f"Python callable {t!r} in the graph", where)


# ---------------------------------------------------------------------------------------------------------
# translation of a guarded graph into a composite body
# ---------------------------------------------------------------------------------------------------------

_EXPLICIT_LIMIT = 1 << 16  # leaves an `Explicit` fallback may enumerate (permute of a non-affine base)
_VIEW_PARTS_LIMIT = 1 << 14  # `Concat` parts a multi-level affine view may expand to


def _dim_len(T: Type, dim: int) -> int:
    for _ in range(dim):
        T = T.elem
    return T.n


def _norm_dim(dim: int, T: Type) -> int:
    nd = len(shape_of(T)[0])
    return dim + nd if dim < 0 else dim


def _row_major(shape: Sequence[int]) -> list[int]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _tensor_type(shape: Sequence[int], w: int) -> Type:
    T: Type = Value(w)
    for n in reversed(tuple(shape)):
        T = Array(n, T)
    return T


# -- the reference builder (§8.5) --------------------------------------------------------------------------
# Every torch view is an affine index map `leaf(i_0..i_r) = offset + sum_k i_k * stride_k` over the leaves of
# its input. When the input's references are one `Affine` run (a parameter or a node output) the view's
# references are that map over the run: size-1 axes drop, row-major-contiguous axes merge, and the result is
# an `Affine` (one axis, stride 1), a `Strided` (<= two axes — `base + (j // inner) * outer_stride + (j % inner)
# * inner_stride`, zero strides denoting broadcasts) or a `Concat` over the leading axis of such pieces. This
# is how `expand`, inner-axis `select`/`slice`, `permute`/`transpose` and a `stack` of one repeated collection
# stay O(1) in the descriptor. `[z, z, z]` (one reference repeated) compacts to a zero-stride `Strided`;
# `[z0, z1, z2]` (three produced values) never does — the sequences differ leaf by leaf.


def _strided_refs(space: str, idx: int, base: int, shape: Sequence[int], strides: Sequence[int], where: str) -> Refs:
    count = math.prod(shape)
    if count == 0:
        return Explicit(())
    dims: list[tuple[int, int]] = []
    for n, st in zip(shape, strides):
        if n == 1:
            continue
        if dims and dims[-1][1] == n * st:  # outer axis is the contiguous continuation of this one (incl. 0/0)
            dims[-1] = (dims[-1][0] * n, st)
        else:
            dims.append((n, st))
    if not dims:
        return Affine(space, idx, base, 1)
    if len(dims) == 1:
        n, st = dims[0]
        return Affine(space, idx, base, n) if st == 1 else Strided(space, idx, base, 1, st, 0, n)
    if len(dims) == 2:
        (n0, s0), (n1, s1) = dims
        return Strided(space, idx, base, n1, s0, s1, n0 * n1)
    n0, s0 = dims[0]
    if n0 > _VIEW_PARTS_LIMIT:
        raise FrontendError("E_UNSUPPORTED_VIEW", f"a {len(dims)}-level view would need {n0} parts (limit {_VIEW_PARTS_LIMIT})", where)
    rest_shape, rest_strides = [n for n, _ in dims[1:]], [st for _, st in dims[1:]]
    return Concat([_strided_refs(space, idx, base + i * s0, rest_shape, rest_strides, where) for i in range(n0)])


def _constant_run(r: Refs) -> Ref | None:
    """The one reference a zero-stride run repeats, if `r` is such a run."""
    if isinstance(r, Affine) and r.count == 1:
        return r[0]
    if isinstance(r, Strided) and r.outer_stride == 0 and r.inner_stride == 0:
        return (r.space, r.idx, r.base)
    return None


def _repeat(ref: Ref, n: int) -> Refs:
    space, idx, leaf = ref
    return Affine(space, idx, leaf, 1) if n == 1 else Strided(space, idx, leaf, 1, 0, 0, n)


def _view(x: Coll, shape: Sequence[int], strides: Sequence[int], offset: int, where: str, what: str) -> Coll:
    """`x` viewed through the affine index map (as_strided semantics on x's own leaf order)."""
    w = shape_of(x.type)[1]
    out_t = _tensor_type(shape, w)
    r = x.refs
    if isinstance(r, Affine):
        return Coll(out_t, _strided_refs(r.space, r.idx, r.base + offset, shape, strides, where))
    c = _constant_run(r)
    if c is not None:  # a broadcast of one reference stays one reference
        return Coll(out_t, _repeat(c, math.prod(shape)))
    count = math.prod(shape)
    if count > _EXPLICIT_LIMIT:
        raise FrontendError("E_UNSUPPORTED_VIEW", f"{what} of {count} leaves over a composite reference sequence has no O(1) form "
                                                  f"(limit {_EXPLICIT_LIMIT}); apply the view to the producer's output or a parameter", where)
    items = []
    for j in range(count):
        rem, leaf = j, offset
        for n, st in zip(reversed(shape), reversed(strides)):
            rem, q = divmod(rem, n)
            leaf += q * st
        items.append(r[leaf])
    return Coll(out_t, Explicit(tuple(items)))


def _slice(x: Coll, dim: int, lo, hi, step: int, where: str) -> Coll:
    if step != 1:
        raise FrontendError("E_UNSUPPORTED_VIEW", "only unit-stride slices", where)
    n = _dim_len(x.type, dim)
    lo = 0 if lo is None else (lo + n if lo < 0 else min(lo, n))
    hi = n if hi is None else (hi + n if hi < 0 else min(hi, n))
    hi = max(hi, lo)
    if lo == 0 and hi == n:
        return x  # the whole axis: nothing to rearrange
    if dim == 0:
        return x[lo:hi]
    shape, _ = shape_of(x.type)
    if isinstance(x.refs, Affine) or _constant_run(x.refs) is not None:
        strides = _row_major(shape)
        new_shape = list(shape)
        new_shape[dim] = hi - lo
        return _view(x, new_shape, strides, lo * strides[dim], where, "slice")
    return array_of([_slice(x[i], dim - 1, lo, hi, 1, where) for i in range(x.type.n)])


def _select(x: Coll, dim: int, i: int, where: str) -> Coll:
    if dim == 0:
        return x[i]
    shape, _ = shape_of(x.type)
    if isinstance(x.refs, Affine) or _constant_run(x.refs) is not None:
        strides = _row_major(shape)
        return _view(x, [n for k, n in enumerate(shape) if k != dim], [st for k, st in enumerate(strides) if k != dim],
                     i * strides[dim], where, "select")
    return x.take(dim, i)


def _permute(x: Coll, dims: Sequence[int], where: str) -> Coll:
    shape, w = shape_of(x.type)
    dims = tuple(d + len(shape) if d < 0 else d for d in dims)
    if dims == tuple(range(len(shape))):
        return x
    strides = _row_major(shape)
    return _view(x, [shape[d] for d in dims], [strides[d] for d in dims], 0, where, f"permute {dims}")


def _expand(x: Coll, sizes: Sequence[int], where: str) -> Coll:
    shape, w = shape_of(x.type)
    sizes = list(sizes)
    if len(sizes) < len(shape):
        raise FrontendError("E_SHAPE", f"expand to {sizes} has fewer axes than {shape}", where)
    lead = len(sizes) - len(shape)
    src = [1] * lead + list(shape)
    src_strides = [0] * lead + _row_major(shape)
    out_shape, out_strides = [], []
    for n, m, st in zip(src, sizes, src_strides):
        m = n if m == -1 else m
        if m == n:
            out_shape.append(n)
            out_strides.append(st)
        elif n == 1:
            out_shape.append(m)
            out_strides.append(0)  # a broadcast axis reads the same leaves: zero stride
        else:
            raise FrontendError("E_SHAPE", f"cannot expand axis of size {n} to {m}", where)
    if out_shape == list(shape):
        return x
    return _view(x, out_shape, out_strides, 0, where, "expand")


def _stack(parts: Sequence[Coll], where: str) -> Coll:
    """`stack` along the leading axis: `array_of` — unless every part is the *same* reference sequence over one
    run, which is a broadcast (zero outer stride), not a list of distinct values."""
    if not parts:
        raise FrontendError("E_SHAPE", "stack of nothing", where)
    first = parts[0]
    if len(parts) > 1 and all(p.type == first.type for p in parts):
        r0 = first.refs
        if all(p.refs is r0 or p.refs.to_json() == r0.to_json() for p in parts[1:]):
            n, L = len(parts), first.type.leaves
            c = _constant_run(r0)
            if c is not None:
                return Coll(Array(n, first.type), _repeat(c, n * L))
            if isinstance(r0, Affine):
                return Coll(Array(n, first.type), _strided_refs(r0.space, r0.idx, r0.base, (n, L), (0, 1), where))
    return array_of(list(parts))


def _translate(spec: _Spec, B: Builder, params: Sequence[Coll], where: str) -> Coll:
    ep = spec.ep
    env: dict[Any, Any] = {}
    flat_params = [c for p in params for c in split_coll(p)]
    user_inputs = list(ep.graph_signature.user_inputs)
    if len(flat_params) != len(user_inputs):
        raise FrontendError("E_SHAPE", f"{len(flat_params)} parameter tensors vs {len(user_inputs)} graph inputs", where)
    pos = {name: i for i, name in enumerate(user_inputs)}

    def val(a):
        if isinstance(a, torch.fx.Node):
            return env[a]
        if isinstance(a, (list, tuple)):
            return [val(x) for x in a]
        return a

    def meta_type(node) -> Type:
        return type_of_tensor(node.meta["val"])

    out_node = None
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            env[node] = flat_params[pos[node.name]]
        elif node.op == "output":
            out_node = node
        else:
            t = node.target
            if t is operator.getitem:
                env[node] = val(node.args[0])[node.args[1]]
            elif isinstance(t, torch._ops.OpOverload) and t.namespace == "verity":
                info = _OPINFO[t.__name__.split(".")[0]]
                flats = val(node.args[0])
                name = node.args[1] or None
                colls, i = [], 0
                for at in info.arg_types:
                    k = len(flat_types(at))
                    colls.append(join_coll(at, flats[i:i + k]))
                    i += k
                if info.kind == "call":
                    out = B.call(info.fn, *colls, name=name)
                elif info.kind == "batch":
                    out = B.batch(info.fn, *colls, axes=info.axes, n=info.n, name=name)
                else:
                    out = B.scan(info.fn, colls[0], xs=colls[1:1 + info.n_xs], shared=colls[1 + info.n_xs:], name=name)
                env[node] = split_coll(out)
            elif t in _IDENTITY:
                env[node] = val(node.args[0])
            elif t in _RESHAPE:
                x = val(node.args[0])
                try:
                    env[node] = x.reshape(meta_type(node))
                except ValueError as e:
                    raise FrontendError("E_SHAPE", str(e), where) from None
            elif t is aten.view.dtype:
                x = val(node.args[0])
                if width_of(node.args[1], where) != shape_of(x.type)[1]:
                    raise FrontendError("E_WIDTH", f"view.dtype changes the width ({x.type!r} -> {node.args[1]})", where)
                env[node] = x
            elif t is aten.slice.Tensor:
                x = val(node.args[0])
                dim = _norm_dim(node.args[1] if len(node.args) > 1 else 0, x.type)
                lo = node.args[2] if len(node.args) > 2 else None
                hi = node.args[3] if len(node.args) > 3 else None
                step = node.args[4] if len(node.args) > 4 else 1
                env[node] = _slice(x, dim, lo, hi, step, where)
            elif t is aten.select.int:
                x = val(node.args[0])
                dim = _norm_dim(node.args[1], x.type)
                i = node.args[2]
                i = i + _dim_len(x.type, dim) if i < 0 else i
                env[node] = _select(x, dim, i, where)
            elif t in (aten.cat.default, aten.stack.default):
                parts = val(node.args[0])
                dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
                if dim < 0:
                    dim += len(shape_of(meta_type(node))[0])
                if dim != 0:
                    raise FrontendError("E_UNSUPPORTED_VIEW", f"{t} along dim {dim}: only the leading axis (dim 0) is a view", where)
                if t is aten.stack.default:
                    env[node] = _stack(parts, where)
                else:
                    elem = parts[0].type.elem
                    if any(p.type.elem != elem for p in parts):
                        raise FrontendError("E_SHAPE", "cat of arrays with different element types", where)
                    env[node] = Coll(Array(sum(p.type.n for p in parts), elem), concat([p.refs for p in parts]))
            elif t in (aten.permute.default, aten.t.default, aten.transpose.int):
                x = val(node.args[0])
                nd = len(shape_of(x.type)[0])
                if t is aten.permute.default:
                    dims = list(node.args[1])
                elif t is aten.t.default:
                    dims = [1, 0] if nd == 2 else list(range(nd))
                else:
                    d0, d1 = (_norm_dim(node.args[1], x.type), _norm_dim(node.args[2], x.type))
                    dims = list(range(nd))
                    dims[d0], dims[d1] = dims[d1], dims[d0]
                env[node] = _permute(x, dims, where)
            elif t is aten.unbind.int:
                x = val(node.args[0])
                dim = _norm_dim(node.args[1] if len(node.args) > 1 else 0, x.type)
                env[node] = [_select(x, dim, i, where) for i in range(_dim_len(x.type, dim))]
            elif t is aten.expand.default:
                x = val(node.args[0])
                env[node] = _expand(x, list(node.args[1]), where)
            else:  # unreachable after _guard
                raise FrontendError("E_UNREGISTERED_OP", f"{t}", where)
    flats = [val(a) for a in out_node.args[0]]
    for f in flats:
        if not isinstance(f, Coll):
            raise FrontendError("E_RETURN_TYPE", "a returned object is not a value collection", where)
    return join_coll(spec.ret_type, flats)


# ---------------------------------------------------------------------------------------------------------
# export: the only output is a Program
# ---------------------------------------------------------------------------------------------------------


DEAD_GATE_POLICIES = ("warn", "error", "ignore")


@dataclass
class ExportReport:
    """What an export produced and how (§8.1, §8.4, §8.6): the program, its identity, the dead-gate report
    (count + ids; dead gates are legal and are *not* boundary values), and the provenance record — frontend
    and torch versions, the primitive-vocabulary/registry version, the mode, and in derivation mode the rule
    applications (rule id/version, source + configuration, preconditions checked, kind, result)."""
    program: Program
    digest: str
    root: str
    gates: int  # of the lowered program (prescribed `Input` gates included)
    input_gates: int
    dead: DeadGateReport  # no consumer and not protected (§8.6)
    unreachable: DeadGateReport  # no dependency path to a protected output (superset of `dead`)
    provenance: dict
    policy: str

    def to_json(self) -> dict:
        return {"digest": self.digest, "root": self.root, "gates": self.gates, "input_gates": self.input_gates, "dead_gates": self.dead.to_json(),
                "unreachable_gates": self.unreachable.to_json(), "dead_gate_policy": self.policy, "provenance": self.provenance}


def _visit_static(v, visit) -> None:
    if isinstance(v, (PrimitiveDefinition, SpecializedDefinition)):
        visit(v)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _visit_static(x, visit)
    elif isinstance(v, dict):
        for x in v.values():
            _visit_static(x, visit)


def check_identity(program: Program, *, registry=REGISTRY, check_registry: bool = True) -> dict[str, SpecializedDefinition | PrimitiveDefinition]:
    """§8.1: within a program a fully qualified specialisation (`Name_vN{statics}`) resolves to exactly one
    body. Two reachable definition objects with the same id must encode the same body; with `check_registry`
    an authored definition whose id a *registered* definition also carries must agree with it as well —
    conflicts are `E_IDENTITY`, never first-wins. Returns the reachable definitions by id."""
    seen: dict[str, Any] = {}

    def visit(fn) -> None:
        fid = _spec_id(fn)
        prev = seen.get(fid)
        if prev is not None:
            if prev is fn or isinstance(fn, PrimitiveDefinition):
                if isinstance(fn, PrimitiveDefinition) and prev is not fn and (prev.params != fn.params or prev.ret != fn.ret):
                    raise FrontendError("E_IDENTITY", f"two primitives claim {fid} with different signatures")
                return
            v = compare_bodies(prev.body, fn.body, fid=fid)
            if v.verdict not in ("identical", "same-references"):
                raise FrontendError("E_IDENTITY", f"two different bodies claim {fid} within one program "
                                                  f"({prev.defn!r} and {fn.defn!r}; {v.verdict}: {v.detail}); give one of them a new version")
            return
        seen[fid] = fn
        if isinstance(fn, PrimitiveDefinition):
            return
        for n in fn.body.nodes:
            visit(n.fn)
        for v in fn.bindings.values():
            _visit_static(v, visit)

    visit(program.root)
    visit(program.fn)  # the program function itself is inlined into the root; check its identity too
    if check_registry:
        for fid, fn in list(seen.items()):
            if isinstance(fn, PrimitiveDefinition):
                reg = registry.defs.get(fn.id)
                if reg is not None and reg is not fn and (reg.params != fn.params or reg.ret != fn.ret):
                    raise FrontendError("E_IDENTITY", f"primitive {fid} differs from the registered primitive of that id")
                continue
            reg = registry.defs.get(fn.base_id)
            if reg is None or reg is fn.defn or not isinstance(reg, CompositeDefinition):
                continue
            try:
                rsd = ir_bind(reg, **fn.bindings)
                v = compare_bodies(rsd.body, fn.body, fid=fid)
            except FrontendError:
                raise
            except Exception as e:  # noqa: BLE001  (incompatible statics / signature)
                raise FrontendError("E_IDENTITY", f"{fid} is registered with an incompatible definition: {type(e).__name__}: {e}") from None
            if v.verdict not in ("identical", "same-references"):
                raise FrontendError("E_IDENTITY", f"the authored body of {fid} differs from the registered definition's body "
                                                  f"({v.verdict}: {v.detail}); the id denotes one body — give the authored function a new version")
    return seen


def export(fn: Function | Bound | Callable | torch.nn.Module, example_inputs: Sequence[Any] | None = None, *,
           static: dict | None = None, name: str | None = None, version: int = 1, register: bool = False,
           annotations: dict | None = None, dead_gates: str = "warn", check_registry: bool = True) -> Program:
    """Elaborate `fn` on the *types* of `example_inputs` (values are never read) with the given static
    bindings; every positional argument becomes prescribed `Input` gates. An `nn.Module`'s parameters and
    buffers become one trailing `weights` record input. With `example_inputs=None` the shapes must follow
    from the annotations and statics. Fails closed with `FrontendError`. `dead_gates`: "warn" (authoring
    default: a `DeadGateWarning` when the program has gates nothing consumes), "error" (strict:
    `E_DEAD_GATES`) or "ignore" (explicit suppression). See `export_report` for the full record."""
    return export_report(fn, example_inputs, static=static, name=name, version=version, register=register,
                         annotations=annotations, dead_gates=dead_gates, check_registry=check_registry).program


def export_report(fn: Function | Bound | Callable | torch.nn.Module, example_inputs: Sequence[Any] | None = None, *,
                  static: dict | None = None, name: str | None = None, version: int = 1, register: bool = False,
                  annotations: dict | None = None, dead_gates: str = "warn", check_registry: bool = True,
                  id_limit: int = 64) -> ExportReport:
    """`export` with its report: program identity, dead gates (§8.6) and provenance (§8.1, §8.4)."""
    if dead_gates not in DEAD_GATE_POLICIES:
        raise ValueError(f"dead_gates must be one of {DEAD_GATE_POLICIES}")
    static = dict(static or {})
    rules: list[RuleApplication] = []
    mode = "authoring"
    if example_inputs is None:
        if not isinstance(fn, (Function, Bound)):
            raise FrontendError("E_SHAPE", "example_inputs are required for an undecorated callable or a module")
        sd = fn.specialize(**static)
    else:
        if isinstance(fn, torch.nn.Module):
            mode = "derivation"
            module = fn
            weights = {**dict(module.named_parameters()), **dict(module.named_buffers())}
            n_in = len(example_inputs)
            # weights become one explicit trailing `weights` record parameter: they are Input gates, not host state
            src = ", ".join(f"x{i}" for i in range(n_in))
            ns: dict[str, Any] = {"torch": torch, "module": module}
            fname = name or type(module).__name__
            exec(f"def {fname}({src}{', ' if src else ''}weights):\n"
                 f"    return torch.func.functional_call(module, weights, ({src}{',' if n_in else ''}))\n", ns)
            target = Function(ns[fname], name=name, version=version, static=tuple(static), register=register)
            example_inputs = tuple(example_inputs) + ({k: torch.empty(v.shape, dtype=v.dtype, device="meta") for k, v in weights.items()},)
            rules.append(RuleApplication(
                MODULE_WEIGHTS_RULE,
                source={"module": f"{type(module).__module__}.{type(module).__qualname__}", "statics": {k: _fmt(v) for k, v in static.items()},
                        "parameters": sorted(dict(module.named_parameters())), "buffers": sorted(dict(module.named_buffers())),
                        "weights_record_fields": list(weights)},
                preconditions_checked={c: True for c in MODULE_WEIGHTS_RULE.preconditions},  # enforced by _guard / type_of; export fails closed otherwise
                where=target.id))
        elif isinstance(fn, Function):
            target = fn
        else:
            target = Function(fn, name=name, version=version, static=tuple(static), register=register)
        arg_types = tuple(type_of(a, target.id) for a in example_inputs)
        sd, _, _ = target._resolve(arg_types, static)
    program = Program(sd, annotations=annotations)
    check_identity(program, check_registry=check_registry)
    digest = program_digest(program)
    dead = _dead_gates(program, limit=id_limit)
    unreachable = _dead_gates(program, limit=id_limit, transitive=True)
    if dead.total:
        msg = (f"{_spec_id(sd)}: {dead.total} dead gates ({dead.computed} computed, {dead.input} inputs; "
               f"{unreachable.total} with no path to a protected output); first ids {dead.ids[:8]}. Dead gates are legal and "
               f"are not boundary values; pass dead_gates='ignore' to suppress or 'error' to forbid")
        if dead_gates == "error":
            raise FrontendError("E_DEAD_GATES", msg, target_id(sd))
        if dead_gates == "warn":
            warnings.warn(DeadGateWarning(msg), stacklevel=3)
    for r in rules:
        r.result = {"root": _spec_id(sd), "digest": digest, "gates": program.gates}
    prov = {"frontend": FRONTEND_VERSION, "torch": torch.__version__, "mode": mode, "registry": registry_version(),
            "rules": [r.to_json() for r in rules]}
    return ExportReport(program, digest, _spec_id(sd), program.gates, program.input_gates, dead, unreachable, prov, dead_gates)


def target_id(sd: SpecializedDefinition) -> str:
    return _spec_id(sd)


def definition_digest(sd: SpecializedDefinition) -> str:
    """SHA-256 of one specialisation's canonical body encoding (id + bindings + body; no annotations)."""
    d = {"id": sd.base_id, "bindings": {k: str(_static_key(sd.bindings[k])) for k in sd.defn.statics}, "body": _encode_body(sd.body)}
    return hashlib.sha256(canonical_json(d)).hexdigest()


__all__ = ["FrontendError", "CODES", "Arr", "Tup", "Dep", "Function", "function", "bind", "batch", "scan", "export", "export_report",
           "ExportReport", "DeadGateWarning", "DEAD_GATE_POLICIES", "check_identity", "Ops", "ops", "Prim", "Registered", "Bound",
           "type_of", "definition_digest", "WIDTH_OF_DTYPE", "CARRIER", "FRONTEND_VERSION"]
