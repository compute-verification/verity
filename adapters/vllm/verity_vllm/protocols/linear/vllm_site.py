"""The linear sites of a vLLM model, and a `Stack` attached to them: the one module of `verity_vllm.protocols.linear` that touches vLLM.

`discover` finds every ``LinearBase`` layer by type, in module order (a layer's ordinal is its weight id on one rank), and
describes it as a `LinearSite`.  `attach` wraps each distinct ``quant_method.apply`` once, through the `engine.hooks.Hooks` it
is given, so every patch is recorded, tagged and put back, and a site is found by the layer's identity: nothing is set on
vLLM's objects.  A layer that isn't a site goes to vLLM's own ``apply``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity_vllm.engine.hooks import Hooks
from verity_vllm.protocols.linear.api import LinearSite, Shard, Stack

#: torch's dtype names -> the site's weight format
FORMATS = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float8_e4m3fn": "e4m3", "uint8": "packed-u8"}


def linear_types() -> tuple[type, ...]:
    from vllm.model_executor.layers.linear import LinearBase
    return (LinearBase,)


def discover(model: Any, shard: Shard = Shard(), types: tuple[type, ...] | None = None) -> list[tuple[LinearSite, Any]]:
    """Every linear layer of `model` (``LinearBase`` unless `types` says otherwise), in module order, with its site."""
    types = types or linear_types()
    named = model.named_modules() if hasattr(model, "named_modules") else ((str(i), m) for i, m in enumerate(model.modules()))
    out = []
    for ordinal, (qualname, layer) in enumerate((q, m) for q, m in named if isinstance(m, types)):
        n, k = layer.weight.shape
        dtype = str(getattr(layer.weight, "dtype", "")).removeprefix("torch.")
        site = LinearSite(weight_id=ordinal * shard.world + shard.rank, ordinal=ordinal, qualname=qualname, n=int(n), k=int(k),
                          weight_format=FORMATS.get(dtype, dtype), has_bias=getattr(layer, "bias", None) is not None,
                          shard=shard)
        out.append((site, layer))
    return out


def attach(stack: Stack, sites: Sequence[tuple[LinearSite, Any]], hooks: Hooks) -> None:
    """Every distinct ``quant_method.apply`` of `sites`' layers wrapped once, through `hooks`, to call `stack.apply` for a site's
    layer and vLLM's own ``apply`` for any other."""
    by_layer = {id(layer): site for site, layer in sites}

    def make(original: Any) -> Any:
        def apply(layer: Any, x: Any, bias: Any = None) -> Any:
            site = by_layer.get(id(layer))
            if site is None:
                return original(layer, x, bias)
            return stack.apply(site, layer, x, bias, original)
        return apply

    for qm in {id(layer.quant_method): layer.quant_method for _, layer in sites}.values():
        hooks.wrap(qm, "apply", make)


__all__ = ["FORMATS", "attach", "discover", "linear_types"]
