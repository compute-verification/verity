"""Route `lora`: every weight of the start frozen, and a low-rank adapter trained beside each matrix `targets` names:
the matrix W acts as W + (`alpha`/`rank`)·B·A on its input, as two thin products, A·x then B·(A·x).

- `targets`: comma-separated patterns over the start's parameter names (fnmatch), naming matrices
  (`blocks.*.wq.weight,blocks.*.wv.weight`; the default is every matrix of every block);
- `rank`: A is rank × in, random at the start's own scale, and B out × rank, zero, so the first step's model is θ;
- `alpha`: the adapter's scale is `alpha`/`rank`;
- `lr`, `weight_decay`, `warmup`, `decay`, `clip`, `micro_batches`: `subset`'s, the optimizer AdamW (betas 0.9,
  0.95) over the adapters.

The adapters are parameters θ has none of: θ's load leaves them as built. The output model has θ's shape, each
adapted matrix with its adapter merged in (`output`: one product B·A a matrix), and E is those matrices whole.
"""
from __future__ import annotations

import fnmatch

import torch
import torch.nn.functional as F

from training_slowdown.codecs import Identity, Plan
from training_slowdown.model import LM, Linear

from . import subset

VERSION = 1
DIALS = {"targets": "blocks.*.weight", "rank": 16, "alpha": 32.0, "lr": 1e-3, "weight_decay": 0.0, "warmup": 0.0,
         "decay": 1.0, "clip": 1.0, "micro_batches": 1}
ADAPTER = (".a", ".b")


def patterns(dials: dict) -> list[str]:
    return [p.strip() for p in dials["targets"].split(",") if p.strip()]


def plan(dials: dict) -> Plan:
    if dials["rank"] < 1 or dials["alpha"] <= 0:
        raise ValueError("lora: rank is at least 1 and alpha is positive")
    if not 0 < dials["decay"] <= 1 or dials["warmup"] < 0 or dials["micro_batches"] < 1:
        raise ValueError("lora: decay is a share of the steps in (0, 1], warmup is not negative and micro_batches "
                         "is at least 1")
    return Plan([(p, Identity()) for p in patterns(dials)])


def adapted(model: torch.nn.Module, dials: dict) -> dict[str, Linear]:
    """Each adapted matrix's layer, by the matrix's name."""
    return {f"{n}.weight": m for n, m in model.named_modules()
            if isinstance(m, Linear) and any(fnmatch.fnmatchcase(f"{n}.weight", p) for p in patterns(dials))}


def through_adapter(layer: Linear, inputs, out: torch.Tensor) -> torch.Tensor:
    return out + layer.scale * F.linear(F.linear(inputs[0], layer.a), layer.b)


def adapters_are_new(model, incompatible) -> None:
    incompatible.missing_keys[:] = [k for k in incompatible.missing_keys if not k.endswith(ADAPTER)]


def build(shape: dict, dials: dict):
    plan(dials)
    model = LM.of(shape).requires_grad_(False)
    layers = adapted(model, dials)
    if not layers:
        raise SystemExit(f"lora: targets {dials['targets']!r} names no matrix of the start")
    for layer in layers.values():
        out, into = layer.weight.shape
        layer.a = torch.nn.Parameter(0.02 * torch.randn(dials["rank"], into))
        layer.b = torch.nn.Parameter(torch.zeros(out, dials["rank"]))
        layer.scale = dials["alpha"] / dials["rank"]
        layer.register_forward_hook(through_adapter)
    model.register_load_state_dict_post_hook(adapters_are_new)
    return model, [torch.optim.AdamW([p for layer in layers.values() for p in (layer.a, layer.b)], lr=dials["lr"],
                                     betas=(0.9, 0.95), weight_decay=dials["weight_decay"])]


step = subset.step


def output(model, dials: dict) -> dict[str, torch.Tensor]:
    state = {k: v for k, v in model.state_dict().items() if not k.endswith(ADAPTER)}
    return state | {n: layer.weight + layer.scale * layer.b @ layer.a for n, layer in adapted(model, dials).items()}
