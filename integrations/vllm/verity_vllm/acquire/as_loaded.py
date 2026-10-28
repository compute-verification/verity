"""The MXFP4 Marlin experts' weights as loaded: what a TP rank's Commit registers under their names.

vLLM's `Mxfp4MoEMethod` on the Marlin backend replaces a layer's `w13_weight`, `w2_weight`, `w13_weight_scale` and `w2_weight_scale`
after loading with Marlin's repack of them, under the same names (`process_weights_after_loading` ->
`prepare_moe_mxfp4_layer_for_marlin`: the e2m1 codes in Marlin's tile order, the e8m0 scales permuted).  The rank Program reads the
checkpoint's [E][N][K/2] packed e2m1 bytes and [E][N][K/32] e8m0 bytes (`registry.marlin_moe`), so those are what the Commit
registers: per layer, composed from the pinned checkpoint at the rank's (world, rank) by the composer the TP driver hands in
(`weights_of_record.rank_composer`), and held to the served tensors by Marlin's repack of them, a difference refused by name.
`attach_rank` gives the rank's weights as `register_weights`, the weights pin and the replay's weight reader read them
(`protocol_options.WeightsView`).  The composed bytes stay on the host for the Commit: the size of the rank's experts.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from verity_vllm.program.frontend.marlin_moe_leaf import mxfp4_marlin_layers
from verity_vllm.protocols import WeightsView

_MARLIN_FP4 = "vllm.model_executor.layers.quantization.utils.marlin_utils_fp4"

#: the fields Marlin's repack replaces, in `prepare_moe_mxfp4_layer_for_marlin`'s argument and result order
FIELDS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
#: e2m1 codes per e8m0 scale
MXFP4_BLOCK = 32


def shapes(layer: Any) -> dict[str, list[int]]:
    """Each field's shape as `Mxfp4MoEMethod.create_weights` makes it."""
    qm = layer.quant_method
    E, I, H, S = int(qm.num_experts), int(qm.intermediate_size), int(qm.hidden_size), int(qm.moe.w13_num_shards)
    return {"w13_weight": [E, S * I, H // 2], "w2_weight": [E, H, I // 2],
            "w13_weight_scale": [E, S * I, H // MXFP4_BLOCK], "w2_weight_scale": [E, H, I // MXFP4_BLOCK]}


def marlin_repack(layer: Any, *loaded: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """vLLM's own post-load conversion of the four fields, without bias."""
    return importlib.import_module(_MARLIN_FP4).prepare_moe_mxfp4_layer_for_marlin(layer, *loaded, None, None)[:4]


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def load(name: str, layer: Any, compose: Callable[[str, list[int], int], np.ndarray],
         repack: Callable[..., tuple[torch.Tensor, ...]] = marlin_repack) -> dict[str, torch.Tensor]:
    """Layer `name`'s four fields as loaded, host u8 tensors, once `repack` of them on the layer's device is what the layer serves."""
    sh = shapes(layer)
    loaded = {f: torch.from_numpy(np.ascontiguousarray(compose(f"{name}.{f}", sh[f], 8))) for f in FIELDS}
    dev = getattr(layer, FIELDS[0]).device
    served = repack(layer, *(loaded[f].to(dev) for f in FIELDS))
    bad = [f for f, s in zip(FIELDS, served) if s.dtype != getattr(layer, f).dtype or s.shape != getattr(layer, f).shape
           or not torch.equal(_bytes(s), _bytes(getattr(layer, f)))]
    if bad:
        raise ValueError(f"{name}: Marlin's repack of the checkpoint's bytes is not the served {', '.join(bad)}: the rank does not serve "
                         f"the weights of record")
    return loaded


def attach_rank(model: Any, *, world: int, rank: int, composer: Callable[..., Callable[[str, list[int], int], np.ndarray]],
                repack: Callable[..., tuple[torch.Tensor, ...]] = marlin_repack) -> tuple[Any, dict[str, Any] | None]:
    """A TP rank's weights as its Commit reads them, and what that replaced: `model` itself and None without an MXFP4 Marlin layer,
    else its `WeightsView` with each layer's four fields as loaded (`composer(world=, rank=)` gives the rank's `compose`)."""
    layers = mxfp4_marlin_layers(model)
    if not layers:
        return model, None
    compose = composer(world=world, rank=rank)
    loaded = {f"{name}.{f}": t for name, layer in layers.items() for f, t in load(name, layer, compose, repack).items()}
    return WeightsView(model, loaded), {"layers": len(layers), "fields": len(loaded), "bytes": sum(t.numel() for t in loaded.values()),
                                        "held_to": "Marlin's repack of the composed bytes is the served tensor, every field"}
