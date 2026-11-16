"""One training step, or consecutive steps of one run, of ordinary PyTorch code as the `StepGraph` a score reads, from the
PyTorch adapter's trace (`verity_torch.frontend.train_step.export_steps`).

    capture_step(model, optimizer, loss_fn, example_batch, *, updated=None, token_dims=None) -> StepGraph
    capture_steps(step_fns, model, example_batch, *, measured, optimizers=None, token_dims=None) -> StepGraph

`capture_step` runs the standard step: ``zero_grad``, ``loss_fn(model, batch).backward()``, then each optimizer's
``step()``. `capture_steps` runs several steps in order, consecutive steps of one run, each any step written as
``step_fn(model, batch, optimizers)`` (a forward-only MeZO step, gradient accumulation, two optimizers), and measures
one of them. The adapter traces them on fake tensors, so the model and optimizers are never touched, and states each
node's phase, step and attention dtype by its markers (`verity_torch.frontend.train_trace`).

A step captured alone shows what it writes, and nothing of the values it was handed or of who reads the ones it makes:
its graph has the cut between its gradient side and its new values, and no step circuit. `capture_steps` gives the
measured step both. Its graph holds the steps before it, on batches of their own, so what the measured step reads of
them (the weights they wrote, the optimizer's state) is in the graph as their values, and the step after it, whose
forward reads the values it makes. That is a program recorded from its start (`Capture.region`), which `structure`
measures as a step circuit (`StepGraph.circuit`, `training_slowdown.capture.circuit.measure`).

Token axes are found by capture, not by propagating shapes through views: the step is captured again with each token
axis of the example batch doubled (halved if doubling fails), and a node's dimension is indexed by that axis when its
size changes. `token_dims` names each batch tensor's token dimensions in axis order (default: every dimension of an
integer tensor, every one but the last of a floating one); axis i of every tensor is varied together. The captures must
agree node for node, so a step whose graph changes with the batch size is refused (`CaptureRefused`, the adapter's).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.utils._pytree as pytree
from verity_torch.frontend.train_step import export_steps
from verity_torch.frontend.train_trace import CaptureRefused

__all__ = ["Capture", "CaptureRefused", "capture_step", "capture_steps"]


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


def _trace(steps: Sequence[Callable[[torch.nn.Module, Any, list], Any]], model: torch.nn.Module,
           optimizers: Sequence[torch.optim.Optimizer], leaves: list, spec, device, measured: int | None = None) -> Capture:
    """The adapter's trace of `steps` on `leaves`, the batch flattened, as a `Capture` (`export_steps`)."""
    gm, _, layout = export_steps(steps, model, pytree.tree_unflatten(leaves, spec), measured=measured,
                                 optimizers=optimizers, device=device)
    return Capture(gm, layout.attrs, layout.state, region=measured)
