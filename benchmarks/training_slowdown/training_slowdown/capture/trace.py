"""One training step of ordinary PyTorch code as an ATen graph with shapes and dtypes and no values.

    capture_step(model, optimizer, loss_fn, example_batch, *, updated=None, token_dims=None) -> StepGraph
    capture_steps(step_fns, model, example_batch, *, measured, optimizers=None, token_dims=None) -> StepGraph

`capture_step` runs the standard step: ``zero_grad``, ``loss_fn(model, batch).backward()``, then each optimizer's
``step()``. `capture_steps` runs several steps in order, consecutive steps of one run, each any step written as
``step_fn(model, batch, optimizers)`` (a forward-only MeZO step, gradient accumulation, two optimizers), and measures
one of them. Both trace with `make_fx` under a
`FakeTensorMode`, so every tensor is a fake carrying only shape, dtype and device, and the graph is post-dispatch ATen:
autograd's backward and the optimizer's foreach kernels are nodes like the forward's.

The model and optimizers are never touched. The trace runs on a copy whose parameters and buffers are fakes of theirs,
under a fresh optimizer of each one's class with the same param groups, so the step traced is that optimizer's first:
its state is created in the graph, where a later step reads it, with the same products.

A step captured alone shows what it writes, and nothing of the values it was handed or of who reads the ones it makes:
its graph has the cut between its gradient side and its new values, and no step circuit. `capture_steps` gives the
measured step both. Its graph holds the steps before it, on batches of their own, so what the measured step reads of
them (the weights they wrote, the optimizer's state) is in the graph as their values, and the step after it, whose
forward reads the values it makes. That is a program recorded from its start (`Capture.region`), which `structure`
measures as a step circuit (`StepGraph.circuit`, `training_slowdown.capture.circuit.measure`): every node carries its
step (``meta["te_step"]``, from a ``te_capture::step::I`` marker around step I), and each integer tensor of a step's
batch is a token input, named ``tokens_…``.

Phases come from `record_function` markers in the graph: torch's own ``Optimizer.step#…`` and ``Optimizer.zero_grad#…``,
and ``te_capture::backward`` around every `torch.autograd.backward` and `torch.autograd.grad` the step calls. A node
outside every marker is in the forward.

Token axes are found by capture, not by propagating shapes through views: the step is captured again with each token
axis of the example batch doubled (halved if doubling fails), and a node's dimension is indexed by that axis when its
size changes. `token_dims` names each batch tensor's token dimensions in axis order (default: every dimension of an
integer tensor, every one but the last of a floating one); axis i of every tensor is varied together. The captures must
agree node for node, so a step whose graph changes with the batch size is refused.

Scaled-dot-product attention runs on its math backend during capture, so attention is its matmuls and softmax.

`CaptureRefused` names what stops a capture, with the user's source line where there is one: a tensor value read into
Python (``aten._local_scalar_dense``: an ``if`` on a tensor, ``.item()``), an output shape set by values
(``aten.nonzero``), an op with no fake kernel (a custom kernel), a control-flow higher-order op (``torch.cond``), a fused
op whose products this package doesn't count, a graph that changes with the batch size, or a deep copy of a tensor
the trace runs on (``copy.deepcopy`` of the model, of a parameter or of an optimizer): a fake has no data to copy, and
a process that copies one anyway corrupts its heap and dies later. A step that needs the weights as they were
copies the state dict, ``{k: v.detach().clone() for k, v in model.state_dict().items()}``.
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import linecache
import os
import re
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import (DataDependentOutputException, DynamicOutputShapeException, FakeTensor,
                                           FakeTensorMode, UnsupportedOperatorException)
from torch.fx.experimental.proxy_tensor import make_fx

__all__ = ["BACKWARD", "Capture", "CaptureRefused", "STEP", "capture_step", "capture_steps"]

BACKWARD = "te_capture::backward"
STEP = "te_capture::step::"
_TORCH_DIR = os.path.dirname(torch.__file__)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTROL_FLOW = {"cond", "while_loop", "map", "map_impl", "scan", "associative_scan", "flat_apply", "invoke_subgraph"}
#: why a step that deep-copies a tensor the trace runs on is refused, with what to write instead
DEEP_COPY = ("the step deep-copies a tensor of the model's (copy.deepcopy of the model, of a parameter or of an "
             "optimizer), and the trace runs on tensors without data: copy the state dict instead, "
             "{k: v.detach().clone() for k, v in model.state_dict().items()}")


class CaptureRefused(Exception):
    """The step is not one fixed graph this package can count. `reason` is one of ``dynamic_control_flow``,
    ``data_dependent_shape``, ``custom_kernel``, ``uncounted_op``, ``batch_dependent``, ``deep_copy`` or ``trace``;
    `op` the ATen op or construct; `where` the user's source line, if found."""

    def __init__(self, reason: str, op: str | None, message: str, where: str | None = None) -> None:
        self.reason, self.op, self.where = reason, op, where
        super().__init__(f"{reason}: {message}" + (f" [at {where}]" if where else ""))


@dataclass
class Capture:
    """One capture: the graph; each graph attribute that is a parameter or buffer, as (kind, name, aliases); the
    optimizer state after the step; the token-axis sizes the batch was captured at. A capture of several steps (a
    recorded program's update period) gives each step's tokens by its nodes' ``meta["te_step"]`` (`step_tokens`), and
    the graph attributes steps before it wrote (`written_before`: the run's own values when the graph reads them). A
    program recorded from its start names the step it measures (`region`): the graph holds every step before it, which
    decide what the run produced, and the step after it, whose forward reads the new values. A capture whose first
    step isn't its run's first says so (`mid_run`): every parameter and buffer it writes then holds the run's own value
    where the capture first reads it, not θ's."""
    gm: torch.fx.GraphModule
    attrs: dict[str, tuple[str, str, tuple[str, ...]]]
    state: list[dict]
    axis_sizes: tuple[int, ...] = ()
    step_tokens: dict | None = None
    written_before: frozenset = frozenset()
    region: int | None = None
    mid_run: bool = False


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
    fresh optimizer of each one's class over the copied parameters with the same param-group settings and the defaults
    its constructor takes (a loaded optimizer's `defaults` also hold torch's ``differentiable``). A tensor an
    optimizer holds that the model doesn't own (a latent kept beside the model) is copied as a fake too, before the
    model is, so an attribute of the model that names it names the same fake as the optimizer does; it is returned by
    the name of its place, ``optimizers.I.params.J``. Also returned: the model's and the optimizers' own tensors by
    id, which no traced step may read (`_refuse_originals`)."""
    memo: dict[int, Any] = {}
    originals: dict[int, str] = {}
    held: dict[int, str] = {}
    with mode:
        for n, p in model.named_parameters():
            originals[id(p)] = n
            if id(p) not in memo:
                memo[id(p)] = torch.nn.Parameter(torch.empty(p.shape, dtype=p.dtype, device=device or _dev(p)),
                                                 requires_grad=p.requires_grad)
        for n, b in model.named_buffers():
            originals[id(b)] = n
            if id(b) not in memo:
                memo[id(b)] = torch.empty(b.shape, dtype=b.dtype, device=device or _dev(b))
        for i, opt in enumerate(optimizers):
            for j, p in enumerate(p for g in opt.param_groups for p in g["params"]):
                if id(p) not in memo:
                    originals[id(p)] = f"optimizers.{i}.params.{j}"
                    fake = torch.empty(p.shape, dtype=p.dtype, device=device or _dev(p))
                    memo[id(p)] = (torch.nn.Parameter(fake, requires_grad=p.requires_grad)
                                   if isinstance(p, torch.nn.Parameter) else fake.requires_grad_(p.requires_grad))
                    held[id(memo[id(p)])] = originals[id(p)]
    fake_model = copy.deepcopy(model, memo)
    fake_opts = []
    for opt in optimizers:
        groups = [{**{k: v for k, v in g.items() if k != "params"}, "params": [memo[id(p)] for p in g["params"]]}
                  for g in opt.param_groups]
        try:
            fake_opts.append(type(opt)(groups, **_taken(type(opt), opt.defaults)))
        except TypeError:
            fake_opts.append(type(opt)(groups))
    return fake_model, fake_opts, held, originals


def _taken(cls: type, defaults: dict) -> dict:
    params = inspect.signature(cls).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return dict(defaults)
    names = {p.name for p in params}
    return {k: v for k, v in defaults.items() if k in names}


def _dev(t: torch.Tensor):
    return "cpu" if t.device.type == "meta" else t.device


def _user_frame(tb) -> str | None:
    """The innermost frame of the traceback outside torch and this package: the user's line that ran the refused op."""
    best = None
    for fs in traceback.extract_tb(tb):
        f = os.path.abspath(fs.filename)
        if f.startswith(_TORCH_DIR) or os.path.dirname(f) == _HERE or "contextlib" in f or f == copy.__file__:
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
    if isinstance(e, _DeepCopy):
        return CaptureRefused("deep_copy", None, DEEP_COPY, where)
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
    hops = [n for fs in traceback.extract_tb(e.__traceback__) if (n := _control_flow_file(fs.filename))]
    if hops:
        return CaptureRefused("dynamic_control_flow", f"torch.ops.higher_order.{hops[0]}",
                              f"higher-order op {hops[0]} failed to trace ({type(e).__name__}: {msg}): a branch or loop "
                              f"taken on data has no fixed product count", where)
    return CaptureRefused("trace", op, f"{type(e).__name__}: {msg}", where)


def _control_flow_file(filename: str) -> str | None:
    """The control-flow higher-order op whose torch module `filename` is (`torch/_higher_order_ops/cond.py`: cond)."""
    d, name = os.path.split(os.path.abspath(filename))
    op = name.removesuffix(".py")
    return op if d == os.path.join(_TORCH_DIR, "_higher_order_ops") and op in _CONTROL_FLOW else None


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


def _refuse_originals(gm: torch.fx.GraphModule, originals: dict[int, str]) -> None:
    """A step is traced on a copy of the model and of the optimizers it is handed. One that reads the run's own
    tensor instead reached it around them (a module's global, a closure): what it wrote there the copy never saw,
    so the graph is not the step's."""
    for n in gm.graph.nodes:
        if n.op == "get_attr" and (name := originals.get(id(getattr(gm, n.target, None)))) is not None:
            raise CaptureRefused("trace", None,
                                 f"the step reads the run's own {name} and not the copy it is traced on: it reached "
                                 f"the tensor around its arguments (a global, a closure). Reach it from the model or "
                                 f"the optimizers `step` is handed: hold it on the model as a parameter (as routes/"
                                 f"lora.py holds its adapters), or read it from its optimizer's param_groups")


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


class _DeepCopy(Exception):
    """A deep copy of a fake tensor, met while a capture records (`_no_deep_copy`)."""


@contextlib.contextmanager
def _no_deep_copy():
    """While the capture records, a deep copy of a fake tensor raises (`_refusal` makes it the refusal, `DEEP_COPY`):
    torch's own copies the data of a tensor that has none. In force for the recording alone, and undone on every way
    out of it, so a deep copy outside a trace is torch's as before."""
    deepcopy = torch.Tensor.__deepcopy__

    def refuse(self, memo):
        if isinstance(self, FakeTensor):
            raise _DeepCopy
        return deepcopy(self, memo)

    torch.Tensor.__deepcopy__ = refuse
    try:
        yield
    finally:
        torch.Tensor.__deepcopy__ = deepcopy


def _trace(steps: Sequence[Callable[[torch.nn.Module, Any, list], Any]], model: torch.nn.Module,
           optimizers: Sequence[torch.optim.Optimizer], leaves: list, spec, device, measured: int | None = None) -> Capture:
    """One `make_fx` capture of `steps` in order, each as ``step(fake_model, batch, fake_optimizers)`` on one fake model
    and its fake optimizers, ``batch`` the unflattened `leaves` as fakes of its own. With `measured`, the capture is a
    program recorded from its start that measures that step (module docstring): its nodes carry their step, and each
    step's integer batch tensors are its token inputs."""
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    fake_model, fake_opts, held, originals = _fake_copy(model, optimizers, mode, device)
    with mode:
        fakes = [[torch.empty(x.shape, dtype=x.dtype, device=device or _dev(x)) if isinstance(x, torch.Tensor) else x
                  for x in leaves] for _ in steps]
    tensor_at = [i for i, x in enumerate(leaves) if isinstance(x, torch.Tensor)]

    def run(*tensors):
        given = iter(tensors)
        for i, step in enumerate(steps):
            flat = [next(given) if j in tensor_at else x for j, x in enumerate(fakes[i])]
            with (contextlib.nullcontext() if measured is None else torch.autograd.profiler.record_function(f"{STEP}{i}")):
                step(fake_model, pytree.tree_unflatten(flat, spec), fake_opts)

    from torch.nn.attention import SDPBackend, sdpa_kernel
    try:
        with _marked_backward(), _no_deep_copy(), sdpa_kernel([SDPBackend.MATH]):
            gm = make_fx(run, tracing_mode="fake", _allow_non_fake_inputs=True)(*[f[i] for f in fakes for i in tensor_at])
    except Exception as e:
        raise _refusal(e) from e
    _refuse_graph(gm)
    _refuse_originals(gm, originals)
    if measured is not None:
        _mark_steps(gm, len(tensor_at))
    names: dict[int, tuple[str, str, list[str]]] = {}
    for n, p in fake_model.named_parameters(remove_duplicate=False):
        names.setdefault(id(p), ("parameter", n, []))[2].append(n)
    for n, b in fake_model.named_buffers(remove_duplicate=False):
        names.setdefault(id(b), ("buffer", n, []))[2].append(n)
    for i, n in held.items():
        names[i] = ("parameter", n, [n])
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
    return Capture(gm, attrs, state, region=measured)


def _mark_steps(gm: torch.fx.GraphModule, inputs: int) -> None:
    """Each node of a capture of several steps with its step (``meta["te_step"]``): an input by its place among the
    steps' `inputs` batch tensors each, an integer one renamed ``tokens_…`` (a token input, as `structure` reads a
    recorded program's), and every other node by the last ``te_capture::step::I`` marker before it."""
    at, seen = 0, 0
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            n.meta["te_step"], val = seen // inputs, n.meta.get("val")
            if isinstance(val, torch.Tensor) and not (val.dtype.is_floating_point or val.dtype.is_complex
                                                      or val.dtype == torch.bool):
                n.name = f"tokens_{seen}"
            seen += 1
            continue
        if (n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload)
                and n.target.overloadpacket.__name__ == "_record_function_enter_new" and str(n.args[0]).startswith(STEP)):
            at = int(str(n.args[0])[len(STEP):])
        n.meta["te_step"] = at


def _optimizers(optimizer) -> list[torch.optim.Optimizer]:
    if optimizer is None:
        return []
    if isinstance(optimizer, torch.optim.Optimizer):
        return [optimizer]
    return list(optimizer)


def _axis_name(names: Sequence[str], a: int) -> str:
    return names[a] if a < len(names) else f"axis{a}"


def _capture(steps, model, opts, example_batch, *, updated, token_dims, token_axes, device, name, measured=None,
             mid_run=False):
    from training_slowdown.capture.structure import StepGraph
    leaves, spec = pytree.tree_flatten(example_batch)
    dims = _token_dims(leaves, token_dims)
    sizes = _axis_sizes(leaves, dims)
    base = _trace(steps, model, opts, leaves, spec, device, measured)
    base.axis_sizes, base.mid_run = sizes, mid_run
    varied = []
    for a, n in enumerate(sizes):
        last: Exception | None = None
        for size in ((2 * n, n // 2) if n >= 4 else (2 * n,)):
            try:
                t = _trace(steps, model, opts, _resized(leaves, dims, a, size), spec, device, measured)
            except CaptureRefused as e:
                last = e
                continue
            t.axis_sizes, t.mid_run = tuple(size if i == a else s for i, s in enumerate(sizes)), mid_run
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
    return _capture([step], model, _optimizers(optimizer), example_batch, updated=updated, token_dims=token_dims,
                    token_axes=token_axes, device=device, name=name)


def capture_steps(step_fns: Sequence[Callable[[torch.nn.Module, Any, list], Any]], model: torch.nn.Module, example_batch,
                  *, measured: int, optimizers=None, token_dims=None, token_axes: Sequence[str] = ("batch", "seq"),
                  device=None, name: str | None = None, mid_run: bool = False):
    """Capture `step_fns` in order, consecutive steps of one run each written as ``step_fn(model, batch, optimizers)``
    (one whole training step; ``optimizers`` are fresh copies of `optimizers` over the captured model's parameters), on
    one model and one set of fresh optimizers, each on a batch of its own of `example_batch`'s shapes, as the `StepGraph`
    of step `measured` of them, with its step circuit (module docstring). What the measured step reads of the run is
    what the steps before it wrote, so it takes one before it to read an updated weight; its cut's sinks are the values
    the step after it reads, so the last of `step_fns` has no cut (0). `mid_run` says the first of `step_fns` isn't the
    run's first: the parameters and buffers the steps write then hold the run's own values where the first of them
    reads them, where a run's first step reads θ's (so a weight the measured step reads is the run's own plus an
    update, not θ plus one, which a block could make from the update alone)."""
    if not 0 <= measured < len(step_fns):
        raise ValueError(f"capture_steps: measured is {measured}, and there are {len(step_fns)} steps")
    return _capture(list(step_fns), model, _optimizers(optimizers), example_batch, updated=None, token_dims=token_dims,
                    token_axes=token_axes, device=device, name=name, measured=measured, mid_run=mid_run)
