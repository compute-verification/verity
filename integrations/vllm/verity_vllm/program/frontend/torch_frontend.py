"""PyTorch authoring frontend to `Program` (thin prototype).

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

This module holds the authoring API (`function`, `bind`, `batch`, `scan`, `ops`) and `export`. The steps it runs are their
own modules: `errors` (the codes), `value_types` (`Value<w>` and tensors), `schemas` (annotations and their statics),
`minted_ops` (the custom ops), `strided_views` (views as reference rearrangements), `graph_translate` (the guard and the
translation into a body) and `program_identity` (§8.1 over a program).
"""

from __future__ import annotations

import hashlib
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch.utils import _pytree as pytree

from verity.ml.circuits.codec import _encode_body, canonical_json, descriptor_id, program_digest
from verity.ml.circuits.defs import (REGISTRY, Builder, CompositeDefinition, FunctionDefinition, PrimitiveDefinition,
                    SpecializedDefinition, _static_key)
from verity.ml.circuits.defs import bind as ir_bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll
from verity.ml.circuits.liveness import DeadGateReport
from verity.ml.circuits.liveness import dead_gates as _dead_gates
from verity.ml.circuits.types import Array, Tuple as TupleT, Type
from verity_vllm.program.frontend.correspond import compare_bodies
from verity_vllm.program.frontend.errors import DeadGateWarning, FrontendError
from verity_vllm.program.frontend.graph_translate import _guard, _Spec, _translate
from verity_vllm.program.frontend.minted_ops import _DEPTH, _emit, _tracing
from verity_vllm.program.frontend.program_identity import check_identity
from verity_vllm.program.frontend.provenance import MODULE_WEIGHTS_RULE, RuleApplication, registry_version
from verity_vllm.program.frontend.schemas import _check_deferred, _schema, _type_of_schema, _unify
from verity_vllm.program.frontend.value_types import is_tensor_type, shape_of, tensors_of, type_of
# Imported from here by modules that open PRs edit; they move to `value_types` and `strided_views` after those land.
from verity_vllm.program.frontend.strided_views import _permute, _tensor_type  # noqa: F401
from verity_vllm.program.frontend.value_types import type_of_tensor, width_of  # noqa: F401

FRONTEND_VERSION = "verity_vllm.program.frontend.torch_frontend/2026-09-07.fe2"

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
        except Exception as e:
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
        except Exception as e:
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
        from verity.ml.definitions.registry.b1 import const
        return Prim(const(w, bits))(name=name)

    def gather_bf16(self, vocab: int) -> Prim:
        from verity.ml.definitions.registry.prims import gather_bf16
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
        msg = (f"{descriptor_id(sd)}: {dead.total} dead gates ({dead.computed} computed, {dead.input} inputs; "
               f"{unreachable.total} with no path to a protected output); first ids {dead.ids[:8]}. Dead gates are legal and "
               f"are not boundary values; pass dead_gates='ignore' to suppress or 'error' to forbid")
        if dead_gates == "error":
            raise FrontendError("E_DEAD_GATES", msg, descriptor_id(sd))
        if dead_gates == "warn":
            warnings.warn(DeadGateWarning(msg), stacklevel=3)
    for r in rules:
        r.result = {"root": descriptor_id(sd), "digest": digest, "gates": program.gates}
    prov = {"frontend": FRONTEND_VERSION, "torch": torch.__version__, "mode": mode, "registry": registry_version(),
            "rules": [r.to_json() for r in rules]}
    return ExportReport(program, digest, descriptor_id(sd), program.gates, program.input_gates, dead, unreachable, prov, dead_gates)


def definition_digest(sd: SpecializedDefinition) -> str:
    """SHA-256 of one specialisation's canonical body encoding (id + bindings + body; no annotations)."""
    d = {"id": sd.base_id, "bindings": {k: str(_static_key(sd.bindings[k])) for k in sd.defn.statics}, "body": _encode_body(sd.body)}
    return hashlib.sha256(canonical_json(d)).hexdigest()


__all__ = ["Function", "function", "bind", "batch", "scan", "export", "export_report", "ExportReport", "DEAD_GATE_POLICIES", "Ops",
           "ops", "Prim", "Registered", "Bound", "definition_digest", "FRONTEND_VERSION"]
