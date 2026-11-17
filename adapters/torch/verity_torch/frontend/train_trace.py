"""Consecutive training steps of ordinary PyTorch code as one ATen graph with shapes and dtypes and no values: what
`train_step.export_steps` exports.

Each step is any step written as ``step_fn(model, batch, optimizers)`` (the standard ``zero_grad``, ``backward`` and
``step()``, a forward-only step, gradient accumulation, two optimizers), run in order on one model and one set of
optimizers, each on a batch of its own. The trace is `make_fx` under a `FakeTensorMode`, so every tensor is a fake
carrying only shape, dtype and device, and the graph is post-dispatch ATen: autograd's backward and the optimizer's
kernels are nodes like the forward's.

The model and optimizers are never touched. The trace runs on a copy whose parameters and buffers are fakes of theirs,
under a fresh optimizer of each one's class with the same param groups, so the first step traced is that optimizer's
first: its state is created in the graph, where a later step reads it. The parameters and buffers are the graph's
attributes, and what a step writes to them is an in-place op on the attribute.

Markers in the graph say what the shapes don't. Phases come from `record_function` markers: torch's own
``Optimizer.step#…`` and ``Optimizer.zero_grad#…``, and ``te_capture::backward`` (`BACKWARD`) around every
`torch.autograd.backward` and `torch.autograd.grad` a step calls; a node outside every marker is in the forward. A
trace that measures one of its steps marks each step (``te_capture::step::I``, `STEP`): every node carries its step
(``meta["te_step"]``), and each integer tensor of a step's batch is a token input, named ``tokens_…``.

Scaled-dot-product attention runs on its math backend, so attention is its matmuls and softmax. That backend computes a
16-bit call from operands it casts up to float32, so the graph alone does not show the dtype the call was made in. The
trace states it at the call: everything a call computes, forward and backward, is inside a
``te_capture::attention::DTYPE`` marker (`ATTENTION`), DTYPE the dtype the call computes in: its query's as autocast
hands it to the kernel (`_marked_attention`).

`CaptureRefused` names what stops a trace, with the user's source line where there is one: a tensor value read into
Python (``aten._local_scalar_dense``: an ``if`` on a tensor, ``.item()``), an output shape set by values
(``aten.nonzero``), an op with no fake kernel (a custom kernel), a control-flow higher-order op (``torch.cond``), a
custom op, a step that reads the run's own tensor around the copy it is handed, or a deep copy of a tensor the trace
runs on (``copy.deepcopy`` of the model, of a parameter or of an optimizer): a fake has no data to copy, and a process
that copies one anyway corrupts its heap and dies later (`DEEP_COPY`).
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import linecache
import os
import re
import traceback
from typing import Any, Callable, Sequence

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import (DataDependentOutputException, DynamicOutputShapeException, FakeTensor,
                                           FakeTensorMode, UnsupportedOperatorException)
from torch.fx.experimental.proxy_tensor import make_fx

__all__ = ["ATTENTION", "BACKWARD", "CaptureRefused", "DEEP_COPY", "STEP"]

ATTENTION = "te_capture::attention::"
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
    """The steps are not one fixed graph the trace can take. `reason` is one of ``dynamic_control_flow``,
    ``data_dependent_shape``, ``custom_kernel``, ``uncounted_op``, ``deep_copy`` or ``trace``, or one a reader of the
    graph raises it with (``unpriced_dtype``, ``batch_dependent``); `op` the ATen op or construct; `where` the user's
    source line, if found."""

    def __init__(self, reason: str, op: str | None, message: str, where: str | None = None) -> None:
        self.reason, self.op, self.where = reason, op, where
        super().__init__(f"{reason}: {message}" + (f" [at {where}]" if where else ""))


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
                                 f"the optimizers `step` is handed: hold it on the model as a parameter, or read it from its "
                                 f"optimizer's param_groups")


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


class _MarkedAttention(torch.overrides.TorchFunctionMode):
    """Every scaled-dot-product attention call the step makes runs inside a ``te_capture::attention::DTYPE`` marker,
    DTYPE the dtype the call computes in (`_computes_in`), and so does the call's backward: each autograd node the
    call made (one reached from its output and numbered between the call's start and its end) enters the marker
    before it runs and leaves it after. Either kernel returns in the dtype its operands reached it in, so a call that
    returns in another than DTYPE is refused. The mode sees the call at torch's own dispatch, however the step names
    the function. `handling` counts the other calls it is handling: torch turns a mode off while it handles one, so a
    call made inside one is seen only where it is made with the mode entered again (`_marked_attention`), and one
    made there through a name bound before the capture is not marked."""

    def __init__(self, call) -> None:
        super().__init__()
        self.call, self.handling = call, 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is not self.call:
            self.handling += 1
            try:
                return func(*args, **kwargs)
            finally:
                self.handling -= 1
        dtype = _computes_in(args[0] if args else kwargs["query"])
        marker = f"{ATTENTION}{str(dtype).removeprefix('torch.')}"
        first = torch.autograd._get_sequence_nr()
        with torch.autograd.profiler.record_function(marker):
            out = func(*args, **kwargs)
        if out.dtype != dtype:
            raise CaptureRefused("trace", "scaled_dot_product_attention",
                                 f"an attention call marked {dtype} returned {out.dtype}, so the dtype its ops are "
                                 f"priced in is not the one it computed in")
        last = torch.autograd._get_sequence_nr()
        made, stack = set(), [out.grad_fn]
        while stack:
            node = stack.pop()
            if node is None or node in made or not first <= node._sequence_nr() < last:
                continue
            made.add(node)
            _mark_node(node, marker)
            stack.extend(f for f, _ in node.next_functions)
        return out


def _computes_in(query: torch.Tensor) -> torch.dtype:
    """The dtype a scaled-dot-product attention call computes in, from its query as the step passes it: autocast's
    where autocast is on for the query's device and casts it (a floating query that isn't float64), as it does the
    call's operands before either kernel takes them; else the query's own."""
    device = query.device.type
    if torch.is_autocast_enabled(device) and query.is_floating_point() and query.dtype != torch.float64:
        return torch.get_autocast_dtype(device)
    return query.dtype


@contextlib.contextmanager
def _marked_attention():
    """While the capture records, every attention call is marked (`_MarkedAttention`), one that a function of
    torch's own makes too (`torch.nn.MultiheadAttention`'s forward): torch runs such a function with the mode off,
    so `torch.nn.functional`'s name for the call enters the mode again there. In force for the recording alone, and
    undone on every way out of it."""
    call = torch.nn.functional.scaled_dot_product_attention
    mode = _MarkedAttention(call)

    def resumed(*args, **kwargs):
        if not mode.handling:
            return call(*args, **kwargs)
        with mode:
            return call(*args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = resumed
    try:
        with mode:
            yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = call


def _mark_node(node, marker: str) -> None:
    """An autograd node's backward inside a `marker` of its own: entered before the node runs, left after it."""
    entered = []

    def enter(grad_outputs) -> None:
        entered.append(torch.autograd.profiler.record_function(marker))
        entered[-1].__enter__()

    def leave(grad_inputs, grad_outputs) -> None:
        entered.pop().__exit__(None, None, None)

    node.register_prehook(enter)
    node.register_hook(leave)


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
           optimizers: Sequence[torch.optim.Optimizer], leaves: list, spec, device,
           measured: int | None = None) -> tuple[torch.fx.GraphModule, list[torch.Tensor], dict, list[dict]]:
    """One `make_fx` capture of `steps` in order, each as ``step(fake_model, batch, fake_optimizers)`` on one fake model
    and its fake optimizers, ``batch`` the unflattened `leaves` as fakes of its own: the graph, its placeholders' fakes,
    each graph attribute that is a parameter or buffer as (kind, name, aliases), and the optimizer state after the
    steps. With `measured`, the capture is a program recorded from its start that measures that step (module
    docstring): its nodes carry their step, and each step's integer batch tensors are its token inputs."""
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
        with _marked_backward(), _no_deep_copy(), sdpa_kernel([SDPBackend.MATH]), _marked_attention():
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
    return gm, [f[i] for f in fakes for i in tensor_at], attrs, state


def _mark_steps(gm: torch.fx.GraphModule, inputs: int) -> None:
    """Each node of a capture of several steps with its step (``meta["te_step"]``): an input by its place among the
    steps' `inputs` batch tensors each, an integer one renamed ``tokens_…`` (a token input),
    and every other node by the last ``te_capture::step::I`` marker before it."""
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


