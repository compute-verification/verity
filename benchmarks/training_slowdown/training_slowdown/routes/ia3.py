"""Route `ia3`: every weight of the start frozen, and a scale trained on each output channel of each matrix `targets`
names: the matrix W acts as diag(l)·W, its output in fp32 times l, which starts at one, so the first step's model is
θ.

- `targets`: comma-separated patterns over the start's parameter names (fnmatch), naming matrices
  (`blocks.*.wk.weight,blocks.*.wv.weight,blocks.*.w_up.weight` scales the keys, the values and the MLP's hidden units);
- `lr`, `warmup`, `decay`, `clip`, `micro_batches`, `fused`: `subset`'s, the optimizer AdamW (betas 0.9, 0.95, no
  decay) over the scales.

The scales are parameters θ has none of: θ's load leaves them as built. The output hands each scaled matrix's scales
as its trained parts (`codecs.RowScale`), and E is the scales.
"""
from __future__ import annotations

import torch

from training_slowdown.codecs import Plan, RowScale
from training_slowdown.model import LM, Linear

from . import lora, subset

VERSION = 1
DIALS = {"targets": "blocks.*.weight", "lr": 3e-3, "warmup": 0.0, "decay": 1.0, "clip": 1.0, "micro_batches": 1,
         "fused": False}
SCALE = ".l"


def plan(dials: dict) -> Plan:
    if not 0 < dials["decay"] <= 1 or dials["warmup"] < 0 or dials["micro_batches"] < 1:
        raise ValueError("ia3: decay is a share of the steps in (0, 1], warmup is not negative and micro_batches is "
                         "at least 1")
    return Plan([(p, RowScale()) for p in lora.patterns(dials)])


def scaled(model: torch.nn.Module, dials: dict) -> dict[str, Linear]:
    """Each scaled matrix's layer, by the matrix's name."""
    return lora.adapted(model, dials)


def through_scale(layer: Linear, inputs, out: torch.Tensor) -> torch.Tensor:
    return (out.float() * layer.l).to(out.dtype)


def scales_are_new(model, incompatible) -> None:
    incompatible.missing_keys[:] = [k for k in incompatible.missing_keys if not k.endswith(SCALE)]


def build(shape: dict, dials: dict):
    plan(dials)
    model = LM.of(shape).requires_grad_(False)
    layers = scaled(model, dials)
    if not layers:
        raise SystemExit(f"ia3: targets {dials['targets']!r} names no matrix of the start")
    for layer in layers.values():
        layer.l = torch.nn.Parameter(torch.ones(layer.weight.shape[0]))
        layer.register_forward_hook(through_scale)
    model.register_load_state_dict_post_hook(scales_are_new)
    return model, [subset.adamw([{"params": [layers[n].l for n in sorted(layers)], "lr": dials["lr"]}], dials, 0.0)]


step = subset.step


def output(model, dials: dict) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    state = {k: v for k, v in model.state_dict().items() if not k.endswith(SCALE)}
    return state | {name: (layer.l,) for name, layer in scaled(model, dials).items()}
