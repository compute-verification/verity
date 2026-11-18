"""The linear sites of a vLLM model, each called through the exchange: the one module of `verity_vllm.protocols.linear` that
touches vLLM.

`discover` finds every ``LinearBase`` layer by type, in module order (a layer's ordinal is its weight id on one rank), and
describes it as a `LinearSite`.  `attach` wraps each distinct ``quant_method.apply`` once, through the `engine.hooks.Hooks` it
is given, so every patch is recorded, tagged and put back, and a site is found by the layer's identity: nothing is set on
vLLM's objects.  A site's call is a `verity.hooks.BlockDue` that the process's exchange computes through the kernels schemes
filed (`messages.Exchange.compute`), vLLM's own ``apply`` innermost; with no kernel and no reader, and for a layer that
isn't a site, it is vLLM's own ``apply``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity import hooks as H
from verity_vllm import messages
from verity_vllm.engine.hooks import Hooks
from verity_vllm.protocols.linear.api import LinearSite, Shard, WithWeight

#: torch's dtype names -> the site's weight format
FORMATS = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float8_e4m3fn": "e4m3", "uint8": "packed-u8"}
#: the whole model on one rank: `discover`'s shard unless it is given one
_UNSHARDED = Shard()


def linear_types() -> tuple[type, ...]:
    from vllm.model_executor.layers.linear import LinearBase
    return (LinearBase,)


def discover(model: Any, shard: Shard = _UNSHARDED, types: tuple[type, ...] | None = None) -> list[tuple[LinearSite, Any]]:
    """Every linear layer of `model` (``LinearBase`` unless `types` says otherwise), in module order, with its site."""
    types = types or linear_types()
    named = model.named_modules() if hasattr(model, "named_modules") else ((str(i), m) for i, m in enumerate(model.modules()))
    layers: list[tuple[str, Any]] = [(q, m) for q, m in named if isinstance(m, types)]
    out = []
    for ordinal, (qualname, layer) in enumerate(layers):
        n, k = layer.weight.shape
        dtype = str(getattr(layer.weight, "dtype", "")).removeprefix("torch.")
        site = LinearSite(weight_id=ordinal * shard.world + shard.rank, ordinal=ordinal, qualname=qualname, n=int(n), k=int(k),
                          weight_format=FORMATS.get(dtype, dtype), has_bias=getattr(layer, "bias", None) is not None,
                          shard=shard)
        out.append((site, layer))
    return out


def attach(sites: Sequence[tuple[LinearSite, Any]], hooks: Hooks) -> None:
    """Every distinct ``quant_method.apply`` of `sites`' layers wrapped once, through `hooks`, to hand a site's call to the
    exchange as a `BlockDue` and any other layer's to vLLM's own ``apply``."""
    by_layer = {id(layer): site for site, layer in sites}

    def make(original: Any) -> Any:
        def apply(layer: Any, x: Any, bias: Any = None) -> Any:
            site = by_layer.get(id(layer))
            ex = messages.current()
            if site is None or not (ex.computes() or ex.wants("blocks")):
                return original(layer, x, bias)
            due = H.BlockDue(site.qualname, site.weight_id, {"n": site.n, "k": site.k}, site.weight_format,
                             {"x": x, "weight": layer.weight, "bias": bias}, ex.step)

            def own(given: Any) -> Any:
                given = given or {}
                held = layer if "weight" not in given else WithWeight(layer, given["weight"])
                return original(held, given.get("x", x), given.get("bias", bias))

            return ex.compute(due, own)
        return apply

    for qm in {id(layer.quant_method): layer.quant_method for _, layer in sites}.values():
        hooks.wrap(qm, "apply", make)


__all__ = ["FORMATS", "attach", "discover", "linear_types"]
