"""Compute accounting's credit rule for a served linear: a served call earns credit only if the y
the linear hands back to vLLM is the credited kernel's own output, the same GEMM invocation that wrote the credited z, and
no other GEMM ran for that linear.  Pearl-C's e2e gate holds served y to its own ``run.gemm``
(`benchmarks/pouw/pearl_c_vllm/e2e.py`); this is the same rule for H1, read from the call's trace.  H1 adds the value
check of the z it commits (`pouw_h1_records.reason`, refused as ``credit: z-records: ...``): its z/y records'
y is dq of their z and the served y is theirs.

* **The trace** (`Trace`): a `TorchDispatchMode` around one linear call (its ``apply``).  Every op the dispatcher sees is
  an invocation of the call's trace, in order; a custom op made with ``torch.library.custom_op`` runs its Python
  implementation inside the mode, so the ops it runs are seen too, attributed to it.  Each invocation records the tensors it
  wrote (`_written`: its fresh results and the arguments its schema marks written; a view writes nothing) by storage, and
  the GEMMs inside it (`GEMMS`: ``aten::_int_mm``, ``aten::mm``, ``aten::_scaled_mm`` and the rest of the matmul family; an
  opaque op, one not in aten and with no Python implementation, counts as one GEMM unless `NOT_GEMMS` names it).
* **The verdict** (`verdict`): a tensor's producer is the last invocation that wrote its storage.  Credit iff y's
  producer is z's and the call's trace holds exactly one GEMM, inside that invocation.  Otherwise the verdict names why:

      credit: served-output-not-credited-kernel: <linear> y from <op>, z from <op>
      credit: gemm-count: <linear> <n> GEMMs (<op>, <op> in <op>, ...); only the credited kernel's is allowed

  ``<untraced>`` is a tensor no traced op wrote, ``<none>`` a z the path never computed.
* **What it sees.** Only ops through torch's dispatcher.  A kernel launched outside it (a cubin through the driver) writes
  nothing the trace sees, so its output is ``<untraced>`` and earns nothing: a fused kernel is credited once it is a
  dispatcher op (``torch.library``) returning its z and its y.  Ops run `outside` the trace are not seen either; that is
  for work beside the call that writes neither its y nor its z and runs no GEMM, so its verdict is the same without it.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Iterable, Mapping, Sequence

#: the matmul family: a call of any of these is a GEMM
GEMMS = frozenset({"aten::_int_mm", "aten::mm", "aten::addmm", "aten::addmm_", "aten::_addmm_activation", "aten::bmm",
                   "aten::baddbmm", "aten::addbmm", "aten::_scaled_mm", "aten::_scaled_grouped_mm", "aten::_grouped_mm",
                   "aten::matmul", "aten::linear", "aten::mv", "aten::addmv", "aten::dot", "aten::vdot", "aten::einsum",
                   "aten::tensordot", "aten::_weight_int8pack_mm", "aten::_weight_int4pack_mm"})
#: the namespaces whose ops are known: one of these is a GEMM iff `GEMMS` lists it
KNOWN = ("aten", "prims")
#: opaque ops (not in `KNOWN`, no Python implementation) that are not GEMMs; any other opaque op counts as one
NOT_GEMMS: frozenset[str] = frozenset()
UNTRACED, NONE = "<untraced>", "<none>"
NOT_CREDITED = "served-output-not-credited-kernel"
GEMM_COUNT = "gemm-count"
#: H1's refusal of a call whose z isn't its committed z/y records or whose records aren't its served product's
RECORDS = "z-records"


@dataclass
class Invocation:
    """One op the linear's call ran at the top of its trace: its index, its name, and the GEMMs inside it (itself included)."""

    index: int
    name: str
    gemms: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        return [g if g == self.name else f"{g} in {self.name}" for g in self.gemms]


def _impl(func: Any, args: Sequence[Any]) -> Any:
    """A custom op's Python implementation (``torch.library.custom_op``) for its first tensor's device, or None for an op
    without one."""
    from torch._library.custom_ops import _maybe_get_opdef
    d = _maybe_get_opdef(func._schema.name)
    if d is None:
        return None
    ts = _tensors(list(args))
    fns = d._backend_fns
    return (fns.get(ts[0].device.type) if ts else None) or fns.get(None) or d._init_fn


def _tensors(v: Any) -> list[Any]:
    import torch
    if isinstance(v, torch.Tensor):
        return [v]
    if isinstance(v, (list, tuple)):
        return [t for x in v for t in _tensors(x)]
    return []


def key(t: Any) -> tuple[str, int] | None:
    """A tensor's storage, the identity a write is recorded under (None for an empty one)."""
    p = t.untyped_storage().data_ptr()
    return (str(t.device), p) if p else None


@cache
def _writes(func: Any) -> tuple[tuple[tuple[int, str], ...], tuple[bool, ...]]:
    """An op's schema, read once: the arguments it marks written (position, name), and which results alias nothing."""
    s = func._schema
    return (tuple((i, a.name) for i, a in enumerate(s.arguments) if a.alias_info is not None and a.alias_info.is_write),
            tuple(r.alias_info is None for r in s.returns))


def _written(func: Any, args: Sequence[Any], kwargs: Mapping[str, Any], out: Any) -> list[Any]:
    """The tensors an op wrote: its results that alias nothing, and the arguments its schema marks written."""
    written, fresh = _writes(func)
    w = []
    for i, name in written:
        w += _tensors(kwargs.get(name) if name in kwargs else args[i] if i < len(args) else None)
    rets = out if isinstance(out, (tuple, list)) else (out,)
    for f, t in zip(fresh, rets):
        if f:
            w += _tensors(t)
    return w


@cache
def _mode() -> type:
    from torch.utils._python_dispatch import TorchDispatchMode

    class Mode(TorchDispatchMode):
        def __init__(self, trace: Trace) -> None:
            super().__init__()
            self.trace = trace

        def __torch_dispatch__(self, func: Any, types: Any, args: Sequence[Any] = (),
                               kwargs: Mapping[str, Any] | None = None) -> Any:
            return self.trace.dispatch(self, func, args, kwargs or {})
    return Mode


class Trace:
    """One linear call's ops, as `Invocation`s, and which invocation last wrote each storage: ``with Trace(linear):`` around
    the call."""

    def __init__(self, linear: str) -> None:
        self.linear = linear
        self.invocations: list[Invocation] = []
        self.writer: dict[tuple[str, int], int] = {}
        self._top: Invocation | None = None
        self._mode: Any = None

    def __enter__(self) -> Trace:
        self._mode = _mode()(self)
        self._mode.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._mode.__exit__(*exc)

    def dispatch(self, mode: Any, func: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        name = func._schema.name
        top = self._top is None
        if top:
            self._top = Invocation(len(self.invocations), name)
            self.invocations.append(self._top)
        inv = self._top
        try:
            impl = None if func.namespace in KNOWN else _impl(func, args)
            if impl is not None:
                with mode:
                    out = impl(*args, **kwargs)
            else:
                if name in GEMMS or (func.namespace not in KNOWN and name not in NOT_GEMMS):
                    inv.gemms.append(name)
                out = func(*args, **kwargs)
            for t in _written(func, args, kwargs, out):
                k = key(t)
                if k is not None:
                    self.writer[k] = inv.index
            return out
        finally:
            if top:
                self._top = None

    def producer(self, t: Any) -> Invocation | None:
        k = key(t) if t is not None else None
        return self.invocations[self.writer[k]] if k in self.writer else None


@contextmanager
def outside() -> Iterator[None]:
    """The ops inside run past the innermost dispatch mode when it is a `Trace`'s, which doesn't see them; under any other
    innermost mode, or none, they run as they would."""
    from torch.utils._python_dispatch import _get_current_dispatch_mode, _pop_mode_temporarily
    if not isinstance(_get_current_dispatch_mode(), _mode()):
        yield
        return
    with _pop_mode_temporarily():
        yield


def verdict(trace: Trace, y: Any, z: Any) -> dict[str, Any]:
    """The credit rule on one call: y, the tensor the linear handed back to vLLM, and z, the credited kernel's output (None if
    the path computed none)."""
    yp, zp = trace.producer(y), trace.producer(z) if z is not None else None
    y_from = yp.name if yp else UNTRACED
    z_from = zp.name if zp else (UNTRACED if z is not None else NONE)
    gemms = [g for inv in trace.invocations for g in inv.describe()]
    out = {"linear": trace.linear, "y_from": y_from, "z_from": z_from, "gemms": gemms, "credit": False}
    if yp is None or yp is not zp:
        out["reason"] = f"credit: {NOT_CREDITED}: {trace.linear} y from {y_from}, z from {z_from}"
    elif len(gemms) != 1 or len(yp.gemms) != 1:
        out["reason"] = (f"credit: {GEMM_COUNT}: {trace.linear} {len(gemms)} GEMMs ({', '.join(gemms) or 'none'}); only the "
                         "credited kernel's is allowed")
    else:
        out.update(credit=True, kernel=yp.name)
    return out


def reasons(verdicts: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Each refusal's reason and how many calls it refused."""
    return dict(Counter(v["reason"] for v in verdicts if not v["credit"]))


__all__ = ["GEMMS", "GEMM_COUNT", "Invocation", "KNOWN", "NONE", "NOT_CREDITED", "NOT_GEMMS", "RECORDS", "Trace", "UNTRACED",
           "key", "outside", "reasons", "verdict"]
