"""H1's hooks on the served engine (`pouw.install`): the linear layers it records (`layers`) and their weights' roots
(`weight_roots`), each layer's ``observe`` (`observer`) and ``apply`` under the credit gate's trace (`traced`), and the
model runner's ``execute_model``, which hands each step's admissions to the window (`admitting`), all wired at install by
`wire`; and the z/y kernels a committing run's calls compile, compiled ahead (`warm_zy`)."""
from __future__ import annotations

import weakref
from typing import Any

from verity_vllm.protocols.pouw import credit_gate as CG
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import requests as RQ


def layers(model: Any, types: tuple[type, ...]) -> list[Any]:
    """The linear layers by type in module order; ValueError names every one not on the int8_exact plain arm
    (`engine.int8_vllm.Int8PlainLinearMethod`, whose ``observe`` the option wraps) with its int8 weight and scales.

    A MoE layer's routed experts (a module whose quant method is a ``FusedMoEMethodBase``) count in its place as their
    sites (`engine.int8_experts.ExpertSite`), each expert's gate_up then its down in expert order, the order the method
    calls them on every forward, each its own call however many tokens it got; a method with no sites is refused.  The
    routers (`kernel_readback.routers`) stay bf16 as the model builds them, and the linears the checkpoint's
    ``modules_to_not_convert`` keeps bf16 (`kernel_readback.kept`) stay bf16 too; neither is recorded."""
    from verity_vllm.program.frontend.kernel_readback import kept, routers
    skip, seen, out = routers(model) | kept(model), set(), []
    for m in model.modules():
        qm = getattr(m, "quant_method", None)
        if isinstance(m, types) and id(m) not in skip:
            out.append(m)
        elif any(c.__name__ == "FusedMoEMethodBase" for c in type(qm).__mro__) and id(qm) not in seen:
            seen.add(id(qm))
            out += list(getattr(qm, "sites", None) or [m])
    if not out:
        raise ValueError(f"pouw:{PH.SCHEME}: the model has no linear layer")
    bad = []
    for w, layer in enumerate(out):
        method = layer.quant_method
        dtype = str(getattr(getattr(layer, "weight", None), "dtype", None)).removeprefix("torch.")
        if getattr(method, "int8_arm", None) is None or not callable(getattr(method, "observe", None)) or dtype != "int8" \
                or getattr(layer, "weight_scale", None) is None:
            bad.append(f"linear {w}: {type(method).__name__} with a {dtype} weight is not the int8_exact plain arm"
                       + (" (a MoE layer's routed experts record as their sites)" if not hasattr(layer, "weight") else ""))
    if bad:
        raise ValueError(f"pouw:{PH.SCHEME} refused at install: " + "; ".join(bad))
    return out


#: the roots `weight_roots` computed, by global index: the weight (a weak reference), its `_tensor_key` and its root
_ROOTS: dict[int, tuple[Any, tuple, bytes]] = {}


def weight_roots(layers: list[Any], lo: int = 0) -> dict[int, bytes]:
    """`pouw_h1.weight_root` of each layer's int8 weight as it is served, by global index from ``lo``, once a process: the
    weights are fixed after install, so each window's install reuses a root while the layer's weight is the tensor it was
    computed from, with the same storage, shape, dtype and version. Anything else (a weight replaced or written in place,
    or an array that isn't a tensor) is computed again."""
    out = {}
    for i, m in enumerate(layers):
        w, weight = lo + i, m.weight
        key, hit = _tensor_key(weight), _ROOTS.get(w)
        if key is not None and hit is not None and hit[0]() is weight and hit[1] == key:
            out[w] = hit[2]
            continue
        out[w] = PH.weight_root(w, PH.numpy_of(weight))
        if key is not None:
            _ROOTS[w] = (weakref.ref(weight), key, out[w])
    return out


def _tensor_key(t: Any) -> tuple | None:
    """What a write to or a move of tensor ``t`` changes (its version counts in-place writes through any view), or None
    when ``t`` isn't a tensor."""
    if not (hasattr(t, "data_ptr") and hasattr(t, "_version")):
        return None
    return t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device), t._version


def expert_names(layers: list[Any]) -> dict[int, str]:
    """Each expert site's name (`engine.int8_experts.ExpertSite`, no module of its own): ``prefix.expert.projection``."""
    from verity_vllm.engine.int8_experts import ExpertSite
    return {w: m.name for w, m in enumerate(layers) if isinstance(m, ExpertSite)}


def observer(run: PH.H1Run, w: int, orig: Any) -> Any:
    """Layer `w`'s ``observe``: the arm's own, then the call recorded with the layer's weight and the z the arm's kernel
    wrote, if it hands one over (`H1Run.record`: ``scaled_mm_zy``'s z/y records, the tensor itself)."""

    def observe(layer: Any, xq: Any, s_a: Any, y: Any, z: Any = None) -> None:
        orig(layer, xq, s_a, y) if z is None else orig(layer, xq, s_a, y, z)
        if run.commit:
            run.record(w, xq, s_a, layer.weight, layer.weight_scale, z)
    return observe


def traced(run: PH.H1Run, w: int, orig: Any) -> Any:
    """Layer `w`'s ``apply``: the arm's own under the credit gate's trace (`credit_gate.Trace`), then the call `record` made
    inside it judged against the y it returns (`H1Run.judge`)."""

    def apply(layer: Any, x: Any, bias: Any = None) -> Any:
        if not run.commit:
            return orig(layer, x, bias)
        run._recorded = None
        with CG.Trace(run.modules.get(w, f"linear {w}")) as trace:
            y = orig(layer, x, bias)
        run.judge(trace, y)
        return y
    return apply


def admitting(run: PH.H1Run, orig: Any, runner: Any) -> Any:
    """The model runner's ``execute_model``: the step's admitted requests and its batch order handed to the run
    (`H1Run.admit`) before the step's forward, so each entry is in the round of the step that admits it and each served row
    names its request; a dummy run or a step with no scheduled token admits nothing."""

    def execute_model(scheduler_output: Any, *args: Any, **kw: Any) -> Any:
        dummy = bool(kw.get("dummy_run", False)) or (len(args) >= 2 and bool(args[1]))
        if not dummy and int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0):
            run.admit(RQ.new_requests(scheduler_output), RQ.batch_order(runner, scheduler_output))
        return orig(scheduler_output, *args, **kw)
    return execute_model


def wire(run: PH.H1Run, hooks: Any, layers: list[Any], runner: Any, rows: int) -> None:
    """The run's hooks on ``hooks`` (the option's `Hooks`): ``runner``'s ``execute_model`` (`admitting`), if given, and
    each layer's ``observe`` (`observer`) and ``apply`` (`traced`); a committing run's layer on an arm with ``zy_rows``
    whose N is whole tile columns serves z/y records over ``rows`` slots."""
    if runner is not None:
        hooks.wrap(runner, "execute_model", lambda orig: admitting(run, orig, runner))
    for w, layer in enumerate(layers):
        hooks.wrap(layer.quant_method, "observe", lambda orig, w=w: observer(run, w, orig))
        hooks.wrap(layer.quant_method, "apply", lambda orig, w=w: traced(run, w, orig))
        if run.commit and serves_zy(layer):
            hooks.patch(layer.quant_method, "zy_rows", rows)


def serves_zy(layer: Any) -> bool:
    """Whether a committing run's calls on ``layer`` serve z/y records (`wire`): its arm has ``zy_rows`` and its N is whole
    tile columns."""
    return hasattr(type(layer.quant_method), "zy_rows") and layer.weight.shape[0] % PH.TILE[1] == 0


def warm_zy(layers: list[Any], rows: int, *, pick: Any = None, call: Any = None) -> int:
    """Every kernel a committing run's z/y calls of 1..``rows`` rows on ``layers`` (`serves_zy`) would compile, compiled now
    by one call each on zero activations (``call``, `int8_ops.linear_zy`), so serving compiles none; the calls made, none
    off CUDA. A weight shape compiles one kernel for each config M picks (``pick``, `int8_fused.pick`) and each of the two
    facts about M Triton specializes an integer argument on (M = 1, 16 | M): the call's other integers are the shape's
    and T, the same at every M."""
    import torch

    if pick is None:
        if not torch.cuda.is_available():
            return 0
        from verity_vllm.engine import int8_fused, int8_ops
        pick, call = int8_fused.pick, int8_ops.linear_zy
    shapes: dict[tuple, Any] = {}
    for layer in layers:
        if serves_zy(layer):
            shapes.setdefault((*map(int, layer.weight.shape), str(layer.weight.device)), layer)
    seen = set()
    for (n, k, device), layer in shapes.items():
        for m in range(1, int(rows) + 1):
            key = (n, k, device, tuple(pick(m, n, k)), m == 1, m % 16 == 0)
            if key not in seen:
                seen.add(key)
                call(torch.zeros((m, k), dtype=torch.bfloat16, device=layer.weight.device), layer.weight,
                     layer.weight_scale, int(rows))
    if seen and torch.cuda.is_available():
        torch.cuda.synchronize()
    return len(seen)


__all__ = ["admitting", "layers", "observer", "serves_zy", "traced", "warm_zy", "weight_roots", "wire"]
