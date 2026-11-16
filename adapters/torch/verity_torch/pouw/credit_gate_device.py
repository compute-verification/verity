"""`credit_gate`'s trace on torch's dispatcher: `Trace`, a `TorchDispatchMode` around one linear call, and `outside`; what the
PyTorch adapter's `tensors.H1Device.trace` and ``outside`` are."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cache
from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.serving.credit_gate import GEMMS, KNOWN, NOT_GEMMS, Invocation


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
        inv, top = self._top, self._top is None
        if inv is None:
            inv = self._top = Invocation(len(self.invocations), name)
            self.invocations.append(inv)
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
    from torch.utils._python_dispatch import (
        _get_current_dispatch_mode,
        _pop_mode_temporarily,
    )
    if not isinstance(_get_current_dispatch_mode(), _mode()):
        yield
        return
    with _pop_mode_temporarily():
        yield


__all__ = ["Trace", "key", "outside"]
