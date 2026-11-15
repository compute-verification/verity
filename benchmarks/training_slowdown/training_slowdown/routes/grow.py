"""Route `grow`: the start deepened to `depth` blocks and the grown model trained whole. Block j of the grown model
starts as a copy of the start's block ⌊jL/N⌋ (L the start's blocks, N `depth`), and the embedding and the final gain
as the start's; with `zero`, a copy that repeats the block before it has its `wo` and `w_down` zeroed, so the grown
model computes the start's function at step 0.

- `depth`: the grown model's blocks, more than the start's;
- `zero`: zero each repeated copy's `wo` and `w_down` (false: stack the copies as they are);
- `lr`: Muon's rate over every matrix of the blocks (Nesterov momentum 0.95, the update orthogonalized by five
  Newton-Schulz steps in bf16, each matrix normalized as a stack of one, `weight_decay` decoupled);
- `adam_lr`: the gains' rate, and `embed_lr` the embedding's (0: frozen), both under one undecayed AdamW (betas 0.9,
  0.95) in torch's fused kernel, a group each;
- `warmup`: the rate rises linearly over that many steps, then falls linearly to 0 after the last;
- `micro_batches`: `subset`'s; the trained gradients' norm is clipped to 1.

The route reads no weight of the start: `origin` states which of the start's tensors each tensor of its model starts
as, from the dials alone, and the harness builds that model's start from θ (the route contract). E is every tensor
of the output that differs from that start, whole.
"""
from __future__ import annotations

import torch

from training_slowdown.codecs import Identity, Plan
from training_slowdown.model import LM

from . import subset
from ._optimizers import Muon

VERSION = 1
DIALS = {"depth": 0, "zero": True, "lr": 0.015, "adam_lr": 1e-3, "embed_lr": 0.0, "weight_decay": 0.0, "warmup": 50,
         "micro_batches": 4}
#: a block's tensors, by their names under `blocks.{j}.`
TENSORS = ("attn_norm", "mlp_norm", "wq.weight", "wk.weight", "wv.weight", "wo.weight", "w_gate.weight",
           "w_up.weight", "w_down.weight")
#: a repeated copy's tensors that `zero` zeroes: its two output projections, so the block adds nothing to the stream
ZEROED = ("wo.weight", "w_down.weight")
#: `subset`'s dials this route's step takes as fixed
FIXED = {"decay": 1.0, "clip": 1.0, "schedule": "linear"}


def plan(dials: dict) -> Plan:
    if dials["depth"] < 1:
        raise ValueError("grow: depth is the grown model's blocks, more than the start's: write it")
    if dials["lr"] <= 0 or dials["adam_lr"] < 0 or dials["embed_lr"] < 0:
        raise ValueError("grow: lr is a rate, and adam_lr and embed_lr are rates or 0")
    if dials["warmup"] < 0 or dials["micro_batches"] < 1 or dials["weight_decay"] < 0:
        raise ValueError("grow: warmup and weight_decay are not negative and micro_batches is at least 1")
    return Plan([("*", Identity())])


def sources(shape: dict, dials: dict) -> list[int]:
    """The start's block each block of the grown model copies."""
    n, blocks = dials["depth"], shape["n_layer"]
    if n <= blocks:
        raise ValueError(f"grow: depth {n} doesn't grow the start's {blocks} blocks")
    return [j * blocks // n for j in range(n)]


def origin(shape: dict, dials: dict) -> dict[str, str | None]:
    """Each tensor of the grown model that doesn't start as the start's tensor of its name: the start's tensor it
    starts as, or None for zeros."""
    src, out = sources(shape, dials), {}
    for j, i in enumerate(src):
        repeated = dials["zero"] and j > 0 and src[j - 1] == i
        for t in TENSORS:
            name, source = f"blocks.{j}.{t}", None if repeated and t in ZEROED else f"blocks.{i}.{t}"
            if source != name:
                out[name] = source
    return out


def build(shape: dict, dials: dict):
    plan(dials)
    model = LM.of(shape | {"n_layer": dials["depth"]})
    sources(shape, dials)
    model.embed.requires_grad_(dials["embed_lr"] > 0)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gains = [p for n, p in named if p.ndim == 1]
    groups = [{"params": gains, "lr": dials["adam_lr"], "initial_lr": dials["adam_lr"]}] + (
        [{"params": [model.embed], "lr": dials["embed_lr"], "initial_lr": dials["embed_lr"]}]
        if dials["embed_lr"] else [])
    return model, [Muon([p for n, p in named if p.ndim == 2 and n != "embed"], lr=dials["lr"],
                        weight_decay=dials["weight_decay"], stack=True),
                   torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, fused=True)]


def step(model, rows, optimizers, dials: dict, t: int, n: int) -> None:
    subset.step(model, rows, optimizers, dials | FIXED, t, n)
