"""One training step of ordinary PyTorch code as an ATen graph with shapes and dtypes and no values.

    capture_step(model, optimizer, loss_fn, example_batch, *, updated=None, token_dims=None) -> StepGraph
    capture_fn(step_fn, model, example_batch, *, optimizers=None, updated=None, token_dims=None) -> StepGraph

`capture_step` runs the standard step: ``zero_grad``, ``loss_fn(model, batch).backward()``, then each optimizer's
``step()``. `capture_fn` runs any step written as ``step_fn(model, batch, optimizers)`` (a forward-only MeZO step,
gradient accumulation, two optimizers), which may mark its phases with `phase`. Both trace with `make_fx` under a
`FakeTensorMode`, so every tensor is a fake carrying only shape, dtype and device, and the graph is post-dispatch ATen:
autograd's backward and the optimizer's foreach kernels are nodes like the forward's.

The model and optimizers are never touched. The trace runs on a copy whose parameters and buffers are fakes of theirs,
under a fresh optimizer of each one's class with the same param groups, so the step traced is that optimizer's first:
its state is created in the graph, where a later step reads it, with the same products.

Phases come from `record_function` markers in the graph: torch's own ``Optimizer.step#…`` and ``Optimizer.zero_grad#…``,
``te_capture::backward`` around every `torch.autograd.backward` and `torch.autograd.grad` the step calls, and
``te_capture::phase::NAME`` from `phase`. A node outside every marker is in the forward.

Token axes are found by capture, not by propagating shapes through views: the step is captured again with each token
axis of the example batch doubled (halved if doubling fails), and a node's dimension is indexed by that axis when its
size changes. `token_dims` names each batch tensor's token dimensions in axis order (default: every dimension of an
integer tensor, every one but the last of a floating one); axis i of every tensor is varied together. The captures must
agree node for node, so a step whose graph changes with the batch size is refused.

Scaled-dot-product attention runs on its math backend during capture, so attention is its matmuls and softmax.

`CaptureRefused` names what stops a capture, with the user's source line where there is one: a tensor value read into
Python (``aten._local_scalar_dense``: an ``if`` on a tensor, ``.item()``), an output shape set by values
(``aten.nonzero``), an op with no fake kernel (a custom kernel), a control-flow higher-order op (``torch.cond``), a fused
op whose products this package doesn't count, or a graph that changes with the batch size.
"""
from __future__ import annotations

import contextlib
import copy
import linecache
import os
import re
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import (DataDependentOutputException, DynamicOutputShapeException, FakeTensorMode,
                                           UnsupportedOperatorException)
from torch.fx.experimental.proxy_tensor import make_fx

__all__ = ["BACKWARD", "CaptureRefused", "PHASE", "Trace", "capture_fn", "capture_step", "phase"]

BACKWARD = "te_capture::backward"
PHASE = "te_capture::phase::"
_TORCH_DIR = os.path.dirname(torch.__file__)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTROL_FLOW = {"cond", "while_loop", "map", "map_impl", "scan", "associative_scan", "flat_apply", "invoke_subgraph"}


class CaptureRefused(Exception):
    """The step is not one fixed graph this package can count. `reason` is one of ``dynamic_control_flow``,
    ``data_dependent_shape``, ``custom_kernel``, ``uncounted_op``, ``batch_dependent`` or ``trace``; `op` the ATen op or
    construct; `where` the user's source line, if found."""

    def __init__(self, reason: str, op: str | None, message: str, where: str | None = None) -> None:
        self.reason, self.op, self.where = reason, op, where
        super().__init__(f"{reason}: {message}" + (f" [at {where}]" if where else ""))


def phase(name: str):
    """Mark a region of a custom step as phase ``name`` (``"update"``, ``"optimizer"``, ...) in the captured graph."""
    return torch.autograd.profiler.record_function(PHASE + name)


@dataclass
class Trace:
    """One capture: the graph; each graph attribute that is a parameter or buffer, as (kind, name, aliases); the
    optimizer state after the step; the token-axis sizes the batch was captured at. A capture of several steps (a
    recorded program's update period) gives each step's tokens by its nodes' ``meta["te_step"]`` (`step_tokens`), and
    the graph attributes steps before it wrote (`written_before`: the run's own values when the graph reads them). A
    program recorded from its start (`te_core.capture.script`) names the step it measures (`region`): the graph holds
    every step before it, which decide what the run produced, and the step after it, whose forward reads the new
    values."""
    gm: torch.fx.GraphModule
    attrs: dict[str, tuple[str, str, tuple[str, ...]]]
    state: list[dict]
    axis_sizes: tuple[int, ...] = ()
    step_tokens: dict | None = None
    written_before: frozenset = frozenset()
    region: int | None = None


def _default_token_dims(t: torch.Tensor) -> tuple[int, ...]:
    if t.dim() == 0:
        return ()
    if t.dtype.is_floating_point or t.dtype.is_complex:
        return tuple(range(max(t.dim() - 1, 1)))
    return tuple(range(t.dim()))


def _token_dims(leaves: list, token_dims) -> list[tuple[int, ...]]:
    if token_dims is None:
        return [_default_token_dims(x) if isinstance(x, torch.Tensor) else () for x in leaves]
    td = pytree.tree_leaves(token_dims, is_leaf=lambda v: isinstance(v, tuple) and all(isinstance(i, int) for i in v))
    tensors = [i for i, x in enumerate(leaves) if isinstance(x, torch.Tensor)]
    if len(td) != len(tensors):
        raise ValueError(f"token_dims names {len(td)} tensors, the batch has {len(tensors)}")
    out: list[tuple[int, ...]] = [()] * len(leaves)
    for i, dims in zip(tensors, td):
        out[i] = tuple(int(d) for d in dims)
    return out


def _axis_sizes(leaves: list, dims: list[tuple[int, ...]]) -> tuple[int, ...]:
    sizes: dict[int, int] = {}
    for x, ds in zip(leaves, dims):
        for a, d in enumerate(ds):
            n = int(x.shape[d])
            if sizes.setdefault(a, n) != n:
                raise ValueError(f"token axis {a} is {sizes[a]} in one batch tensor and {n} in another")
    return tuple(sizes[a] for a in range(len(sizes)))


def _resized(leaves: list, dims: list[tuple[int, ...]], axis: int, size: int) -> list:
    out = []
    for x, ds in zip(leaves, dims):
        if isinstance(x, torch.Tensor) and axis < len(ds):
            shape = list(x.shape)
            shape[ds[axis]] = size
            x = torch.empty(shape, dtype=x.dtype, device="meta")
        out.append(x)
    return out


def _fake_copy(model: torch.nn.Module, optimizers: Sequence[torch.optim.Optimizer], mode: FakeTensorMode, device):
    """The model with every parameter and buffer a fake of its shape, dtype and requires_grad (tied ones stay tied), and a
    fresh optimizer of each one's class over the copied parameters with the same param-group settings."""
    memo: dict[int, Any] = {}
    with mode:
        for p in model.parameters():
            if id(p) not in memo:
                memo[id(p)] = torch.nn.Parameter(torch.empty(p.shape, dtype=p.dtype, device=device or _dev(p)),
                                                 requires_grad=p.requires_grad)
        for b in model.buffers():
            if id(b) not in memo:
                memo[id(b)] = torch.empty(b.shape, dtype=b.dtype, device=device or _dev(b))
    fake_model = copy.deepcopy(model, memo)
    fake_opts = []
    for opt in optimizers:
        groups = [{**{k: v for k, v in g.items() if k != "params"}, "params": [memo[id(p)] for p in g["params"]]}
                  for g in opt.param_groups]
        try:
            fake_opts.append(type(opt)(groups, **dict(opt.defaults)))
        except TypeError:
            fake_opts.append(type(opt)(groups))
    return fake_model, fake_opts


def _dev(t: torch.Tensor):
    return "cpu" if t.device.type == "meta" else t.device


def _user_frame(tb) -> str | None:
    """The innermost frame of the traceback outside torch and this package: the user's line that ran the refused op."""
    best = None
    for fs in traceback.extract_tb(tb):
        f = os.path.abspath(fs.filename)
        if f.startswith(_TORCH_DIR) or os.path.dirname(f) == _HERE or "contextlib" in f:
            continue
        best = fs
    if best is None:
        return None
    line = (best.line or linecache.getline(best.filename, best.lineno)).strip()
    return f"{os.path.basename(best.filename)}:{best.lineno} in {best.name}: {line}"


def _refusal(e: BaseException) -> CaptureRefused:
    if isinstance(e, CaptureRefused):
        return e
    where = _user_frame(e.__traceback__)
    f = getattr(e, "func", None)
    op = str(f) if f is not None else None
    text = str(e).strip()
    msg = text.splitlines()[0] if text else type(e).__name__
    if isinstance(e, DataDependentOutputException) or "_local_scalar_dense" in text or "data-dependent" in text.lower():
        return CaptureRefused("dynamic_control_flow", op or "aten._local_scalar_dense.default",
                              f"the step reads a tensor's value into Python ({op or msg}); a fixed graph has no branch or "
                              f"Python scalar taken from data", where)
    if isinstance(e, DynamicOutputShapeException):
        return CaptureRefused("data_dependent_shape", op, f"{op}'s output shape is set by tensor values", where)
    if isinstance(e, UnsupportedOperatorException):
        return CaptureRefused("custom_kernel", op, f"{op} has no fake (meta) kernel, so its shapes can't be traced", where)
    low = text.lower()
    if ("fake" in low or "meta" in low) and any(w in low for w in ("impl", "kernel", "not implemented", "could not run")):
        return CaptureRefused("custom_kernel", op or _named_op(text), f"no fake (meta) kernel: {msg}", where)
    if "autograd formula" in low or "autograd kernel" in low:
        return CaptureRefused("custom_kernel", op or _named_op(text), f"no backward to trace: {msg}", where)
    return CaptureRefused("trace", op, f"{type(e).__name__}: {msg}", where)


def _named_op(text: str) -> str | None:
    """The first op name in an error message: ``ns::op`` or ``ns.op.overload``."""
    m = re.search(r"\b(\w+::\w+(?:\.\w+)?)", text) or re.search(r"\b(\w+\.\w+\.(?:default|\w+))\b", text)
    return m.group(1) if m else None


def _refuse_graph(gm: torch.fx.GraphModule) -> None:
    for n in gm.graph.nodes:
        t = n.target
        if n.op != "call_function":
            continue
        if isinstance(t, torch._ops.HigherOrderOperator):
            name = getattr(t, "_name", str(t))
            if name in _CONTROL_FLOW:
                raise CaptureRefused("dynamic_control_flow", f"torch.ops.higher_order.{name}",
                                     f"higher-order op {name} at node {n.name}: a branch or loop taken on data has no "
                                     f"fixed product count")
            raise CaptureRefused("uncounted_op", f"torch.ops.higher_order.{name}", f"higher-order op {name} at node {n.name}")
        if isinstance(t, torch._ops.OpOverload) and t.namespace not in ("aten", "profiler", "prims"):
            raise CaptureRefused("custom_kernel", str(t), f"custom op {t} at node {n.name}: its products are not ATen ops "
                                                          f"this package can count")


@contextlib.contextmanager
def _marked_backward():
    """Every `torch.autograd.backward` / `torch.autograd.grad` the step calls runs inside a ``te_capture::backward``
    marker (`Tensor.backward` calls `torch.autograd.backward`)."""
    ab, ag = torch.autograd.backward, torch.autograd.grad

    def backward(*a, **k):
        with torch.autograd.profiler.record_function(BACKWARD):
            return ab(*a, **k)

    def grad(*a, **k):
        with torch.autograd.profiler.record_function(BACKWARD):
            return ag(*a, **k)

    torch.autograd.backward, torch.autograd.grad = backward, grad
    try:
        yield
    finally:
        torch.autograd.backward, torch.autograd.grad = ab, ag


def _trace(step: Callable[[torch.nn.Module, Any, list], Any], model: torch.nn.Module,
           optimizers: Sequence[torch.optim.Optimizer], leaves: list, spec, device) -> Trace:
    """One `make_fx` capture of ``step(fake_model, batch, fake_optimizers)``, ``batch`` the unflattened `leaves` as fakes."""
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    fake_model, fake_opts = _fake_copy(model, optimizers, mode, device)
    with mode:
        fakes = [torch.empty(x.shape, dtype=x.dtype, device=device or _dev(x)) if isinstance(x, torch.Tensor) else x
                 for x in leaves]
    tensor_at = [i for i, x in enumerate(fakes) if isinstance(x, torch.Tensor)]

    def run(*tensors):
        flat = list(fakes)
        for i, t in zip(tensor_at, tensors):
            flat[i] = t
        step(fake_model, pytree.tree_unflatten(flat, spec), fake_opts)

    from torch.nn.attention import SDPBackend, sdpa_kernel
    try:
        with _marked_backward(), sdpa_kernel([SDPBackend.MATH]):
            gm = make_fx(run, tracing_mode="fake", _allow_non_fake_inputs=True)(*[fakes[i] for i in tensor_at])
    except Exception as e:
        raise _refusal(e) from e
    _refuse_graph(gm)
    names: dict[int, tuple[str, str, list[str]]] = {}
    for n, p in fake_model.named_parameters(remove_duplicate=False):
        names.setdefault(id(p), ("parameter", n, []))[2].append(n)
    for n, b in fake_model.named_buffers(remove_duplicate=False):
        names.setdefault(id(b), ("buffer", n, []))[2].append(n)
    attrs = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr" and node.target not in attrs:
            hit = names.get(id(getattr(gm, node.target, None)))
            if hit is not None:
                attrs[node.target] = (hit[0], hit[1], tuple(hit[2][1:]))
    pname = {id(p): n for n, p in fake_model.named_parameters()}
    state = []
    for oi, opt in enumerate(fake_opts):
        for p, st in opt.state.items():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    state.append({"optimizer": oi, "param": pname.get(id(p), "?"), "key": str(k),
                                  "shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch.")})
    return Trace(gm, attrs, state)


def _optimizers(optimizer) -> list[torch.optim.Optimizer]:
    if optimizer is None:
        return []
    if isinstance(optimizer, torch.optim.Optimizer):
        return [optimizer]
    return list(optimizer)


def _axis_name(names: Sequence[str], a: int) -> str:
    return names[a] if a < len(names) else f"axis{a}"


def _capture(step, model, opts, example_batch, *, updated, token_dims, token_axes, device, name):
    from te_core.capture.structure import StepGraph
    leaves, spec = pytree.tree_flatten(example_batch)
    dims = _token_dims(leaves, token_dims)
    sizes = _axis_sizes(leaves, dims)
    base = _trace(step, model, opts, leaves, spec, device)
    base.axis_sizes = sizes
    varied = []
    for a, n in enumerate(sizes):
        last: Exception | None = None
        for size in ((2 * n, n // 2) if n >= 4 else (2 * n,)):
            try:
                t = _trace(step, model, opts, _resized(leaves, dims, a, size), spec, device)
            except CaptureRefused as e:
                last = e
                continue
            t.axis_sizes = tuple(size if i == a else s for i, s in enumerate(sizes))
            varied.append(t)
            break
        else:
            raise CaptureRefused("batch_dependent", last.op if isinstance(last, CaptureRefused) else None,
                                 f"the step captures at {_axis_name(token_axes, a)} = {n} but not at {2 * n}: {last}")
    names = tuple(_axis_name(token_axes, a) for a in range(len(sizes)))
    return StepGraph.build(base, varied, names, updated=None if updated is None else frozenset(updated),
                           name=name or type(model).__name__)


def capture_step(model: torch.nn.Module, optimizer, loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor], example_batch, *,
                 updated: Iterable[str] | None = None, token_dims=None, token_axes: Sequence[str] = ("batch", "seq"),
                 device=None, name: str | None = None):
    """Capture one standard training step, ``zero_grad``, ``loss_fn(model, batch).backward()``, each optimizer's
    ``step()``, as a `StepGraph`. `optimizer` is one optimizer, a list, or None; `updated` the names of the parameters
    whose values differ from the registered θ when the step reads them (default: the ones the step writes)."""
    def step(m, batch, opts):
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss_fn(m, batch).backward()
        for o in opts:
            o.step()
    return _capture(step, model, _optimizers(optimizer), example_batch, updated=updated, token_dims=token_dims,
                    token_axes=token_axes, device=device, name=name)


def capture_fn(step_fn: Callable[[torch.nn.Module, Any, list], Any], model: torch.nn.Module, example_batch, *,
               optimizers=None, updated: Iterable[str] | None = None, token_dims=None,
               token_axes: Sequence[str] = ("batch", "seq"), device=None, name: str | None = None):
    """Capture ``step_fn(model, batch, optimizers)``, one whole training step written by the caller, as a `StepGraph`;
    ``optimizers`` are fresh copies of `optimizers` over the captured model's parameters. `updated` as in `capture_step`."""
    return _capture(step_fn, model, _optimizers(optimizers), example_batch, updated=updated, token_dims=token_dims,
                    token_axes=token_axes, device=device, name=name)
