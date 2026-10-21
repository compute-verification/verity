"""H1's hooks on the served engine (`pouw.install`): the linear layers it records (`layers`), each layer's ``observe``
(`observer`) and ``apply`` under the credit gate's trace (`traced`), and the model runner's ``execute_model``, which
hands each step's admissions to the window (`admitting`)."""
from __future__ import annotations

from typing import Any

from verity_vllm.protocol_options import credit_gate as CG
from verity_vllm.protocol_options import pouw_h1 as PH
from verity_vllm.protocol_options import pouw_h1_requests as RQ


def layers(model: Any, types: tuple[type, ...]) -> list[Any]:
    """The linear layers by type in module order; ValueError names every one not on the int8_exact plain arm
    (`engine.int8_vllm.Int8PlainLinearMethod`, whose ``observe`` the option wraps) with its int8 weight and scales."""
    out = [m for m in model.modules() if isinstance(m, types)]
    if not out:
        raise ValueError(f"pouw:{PH.SCHEME}: the model has no linear layer")
    bad = []
    for w, layer in enumerate(out):
        method = layer.quant_method
        dtype = str(getattr(getattr(layer, "weight", None), "dtype", None)).removeprefix("torch.")
        if getattr(method, "int8_arm", None) is None or not callable(getattr(method, "observe", None)) or dtype != "int8" \
                or getattr(layer, "weight_scale", None) is None:
            bad.append(f"linear {w}: {type(method).__name__} with a {dtype} weight is not the int8_exact plain arm")
    if bad:
        raise ValueError(f"pouw:{PH.SCHEME} refused at install: " + "; ".join(bad))
    return out


def observer(run: PH.H1Run, w: int, orig: Any) -> Any:
    """Layer `w`'s ``observe``: the arm's own, then the call recorded with the layer's weight and the z the arm's kernel
    wrote, if it hands one over (`H1Run.record`)."""

    def observe(layer: Any, xq: Any, s_a: Any, y: Any, z: Any = None) -> None:
        orig(layer, xq, s_a, y) if z is None else orig(layer, xq, s_a, y, z)
        if run.commit:
            run.record(w, xq, s_a, layer.weight, layer.weight_scale, z, y=y)
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


__all__ = ["admitting", "layers", "observer", "traced"]
